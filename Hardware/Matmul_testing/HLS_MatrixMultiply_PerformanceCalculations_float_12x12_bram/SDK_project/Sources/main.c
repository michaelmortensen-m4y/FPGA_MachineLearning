/*
 * main.c
 *
 *      Author: Michael Mortensen
 */

#include <stdio.h>
#include <math.h>
#include <xparameters.h>
#include <xbramtest.h>
//*** Needed for timer hardware
#include "xil_types.h"
#include "xtmrctr.h"

#define IN_A_ROWS 12
#define IN_A_COLS 12
#define IN_B_ROWS 12
#define IN_B_COLS 12
#define OUT_C_ROWS 12
#define OUT_C_COLS 12

// Pointers to the BRAM controllers (The addresses should match the axi_bram_ctrl_<x> offset address found under address editor in vivado design suite)
float (*testMatrix_a_HW)[IN_A_ROWS] = (float (*)[IN_A_ROWS])0x40000000;
float (*testMatrix_b_HW)[IN_B_ROWS] = (float (*)[IN_B_ROWS])0x42000000;
float (*testResultMatrix_c_HW)[OUT_C_ROWS] = (float (*)[OUT_C_ROWS])0x44000000;


// Conversion function (float to int and int to float)
unsigned int float_to_u32(float val) {
	unsigned int result;
	union float_bytes {
		float v;
		unsigned char bytes[4];
	} data;
	data.v = val;
	result = (data.bytes[3] << 24) + (data.bytes[2] << 16) + (data.bytes[1] << 8) + (data.bytes[0]);
	return result;
}

float u32_to_float(unsigned int val) {
	union {
		float val_float;
		unsigned char bytes[4];
	} data;
	data.bytes[3] = (val >> (8*3)) & 0xff;
	data.bytes[2] = (val >> (8*2)) & 0xff;
	data.bytes[1] = (val >> (8*1)) & 0xff;
	data.bytes[0] = (val >> (8*0)) & 0xff;
	return data.val_float;
}

// bramtest SW version
void bramtest(float a[IN_A_ROWS][IN_A_COLS], float b[IN_B_ROWS][IN_B_COLS], float c[OUT_C_ROWS][OUT_C_COLS]) {
	for (int i = 0; i < IN_A_ROWS; i++) {
		for (int j = 0; j < IN_B_COLS; j++) {
			c[i][j] = 0;
			for (int k = 0; k < IN_B_ROWS; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

int main() {
	//*** Initialize the timer hardware
	XTmrCtr m_AxiTimer;
	unsigned int m_tickCounter1;
	unsigned int m_tickCounter2;
	unsigned int init_time, curr_time, calibration;
	double m_clockPeriodSeconds;
	double m_timerClockFreq;
	double elapsedTimeInSeconds_SW;
	double elapsedTimeInSeconds_HW;
	double speedup;
	unsigned int elapsedClockCycles_SW;
	unsigned int elapsedClockCycles_HW;
	XTmrCtr_Initialize(&m_AxiTimer, XPAR_TMRCTR_0_DEVICE_ID);
	XTmrCtr_SetOptions(&m_AxiTimer, XPAR_AXI_TIMER_0_DEVICE_ID, XTC_ENABLE_ALL_OPTION);
	// Get the clock period in seconds
	m_timerClockFreq = (double) XPAR_AXI_TIMER_0_CLOCK_FREQ_HZ;
	m_clockPeriodSeconds = (double)1/m_timerClockFreq;
	// Calibrate timer
	XTmrCtr_Reset(&m_AxiTimer, XPAR_AXI_TIMER_0_DEVICE_ID);
	init_time = XTmrCtr_GetValue(&m_AxiTimer, XPAR_AXI_TIMER_0_DEVICE_ID);
	curr_time = XTmrCtr_GetValue(&m_AxiTimer, XPAR_AXI_TIMER_0_DEVICE_ID);
	calibration = curr_time - init_time;
	printf("Calibrating the timer:\n");
	printf("init_time: %d cycles\n", init_time);
	printf("curr_time: %d cycles\n", curr_time);
	printf("calibration: %d cycles\n", calibration);
	XTmrCtr_Reset(&m_AxiTimer, XPAR_AXI_TIMER_0_DEVICE_ID);
	m_tickCounter1 = XTmrCtr_GetValue(&m_AxiTimer, XPAR_AXI_TIMER_0_DEVICE_ID);
	for (int i = 0; i < 10000; i++);
	m_tickCounter2 = XTmrCtr_GetValue(&m_AxiTimer, XPAR_AXI_TIMER_0_DEVICE_ID);
	elapsedClockCycles_SW = m_tickCounter2 - m_tickCounter1 - calibration;
	printf("Loop time for %d iterations is %d cycles\n\n", 10000, elapsedClockCycles_SW);

	// Initialize the Bramtest IP core
	int status;
	XBramtest doBramtest;
	XBramtest_Config *doBramtest_cfg;
	doBramtest_cfg = XBramtest_LookupConfig(XPAR_BRAMTEST_0_DEVICE_ID);
	if (!doBramtest_cfg) {
		printf("Error loading config for doBramtest_cfg\n");
	}
	status = XBramtest_CfgInitialize(&doBramtest, doBramtest_cfg);
	if (status != XST_SUCCESS) {
		printf("Error initializing for doBramtest\n");
	}

	// Declare test matrices and other stuff
	float testMatrix_a[IN_A_ROWS][IN_A_COLS];
	float testMatrix_b[IN_B_ROWS][IN_B_COLS];
	float testResultMatrix_c[OUT_C_ROWS][OUT_C_COLS];
	float referenceResultMatrix_c[OUT_C_ROWS][OUT_C_COLS];
	int i, j, k;

	// Generate test matrices for SW and reference matrix
	for (i = 0; i < IN_A_ROWS; i++) {
		for (j = 0; j < IN_A_COLS; j++) {
			testMatrix_a[i][j] = (float)i+j;
		}
	}
	for (i = 0; i < IN_B_ROWS; i++) {
		for (j = 0; j < IN_B_COLS; j++) {
			testMatrix_b[i][j] = (float)i*j;
		}
	}
	for (i = 0; i < IN_A_ROWS; i++) {
		for (j = 0; j < IN_B_COLS; j++) {
			referenceResultMatrix_c[i][j] = 0;
			for (k = 0; k < IN_B_ROWS; k++) {
				referenceResultMatrix_c[i][j] += testMatrix_a[i][k] * testMatrix_b[k][j];
			}
		}
	}
	int error = 0;

	// Run on software
	printf("\n********** Matrix Multiply **********\n\n");
	printf("Running on SW:\n");
	printf("Testing SW started\n");

	//*** Start timer 0
	XTmrCtr_Reset(&m_AxiTimer,0);
	m_tickCounter1 = XTmrCtr_GetValue(&m_AxiTimer, 0);
	XTmrCtr_Start(&m_AxiTimer, 0);

	bramtest(testMatrix_a, testMatrix_b, testResultMatrix_c);

	//*** Stop timer 0
	XTmrCtr_Stop(&m_AxiTimer,0);
	m_tickCounter2 = XTmrCtr_GetValue(&m_AxiTimer, 0);
	elapsedClockCycles_SW = m_tickCounter2 - m_tickCounter1 - calibration;
	elapsedTimeInSeconds_SW = (double) elapsedClockCycles_SW * m_clockPeriodSeconds;

	for (i = 0; i < OUT_C_ROWS; i++) {
		for (j = 0; j < OUT_C_COLS; j++) {
			if (testResultMatrix_c[i][j] != referenceResultMatrix_c[i][j]) {
				error++;
			}
		}
	}
	if (error > 0) {
		printf("Error in SW calculation!\n Error = %d\n", error);
		printf("\n*************************************\n");
		return 0;
	}
	printf("No error in SW calculation!\n");
	printf("Run time for SW on ARM Processor = %f seconds (%d clock cycles)\n\n", elapsedTimeInSeconds_SW, elapsedClockCycles_SW);

	// Reset result and error count
	printf("\nResetting variables\n");
	for (i = 0; i < OUT_C_ROWS; i++) {
		for (j = 0; j < OUT_C_COLS; j++) {
			testResultMatrix_c[i][j] = 0;
			testResultMatrix_c_HW[i][j] = 0;
		}
	}
	error = 0;

	// Run on hardware
	printf("\n");
	printf("Running on HW:\n");
	printf("Testing HW started\n");

	//*** Start timer 0
	XTmrCtr_Reset(&m_AxiTimer,0);
	m_tickCounter1 = XTmrCtr_GetValue(&m_AxiTimer, 0);
	XTmrCtr_Start(&m_AxiTimer, 0);

	// Generate test matrices for HW and reference matrix
	for (i = 0; i < IN_A_ROWS; i++) {
		for (j = 0; j < IN_A_COLS; j++) {
			//testMatrix_a_HW[i][j] = float_to_u32(((float)i+j));
			testMatrix_a_HW[i][j] = (float)i+j;
		}
	}
	for (i = 0; i < IN_B_ROWS; i++) {
		for (j = 0; j < IN_B_COLS; j++) {
			//testMatrix_b_HW[i][j] = float_to_u32(((float)i*j));
			testMatrix_b_HW[i][j] = (float)i*j;
		}
	}

	XBramtest_Start(&doBramtest); // Start IP core

	while (!XBramtest_IsDone(&doBramtest)) {} // Wait until output is ready
	// Since the IP manipulated the c array directly we already have access to the result

	//*** Stop timer 0
	XTmrCtr_Stop(&m_AxiTimer,0);
	m_tickCounter2 = XTmrCtr_GetValue(&m_AxiTimer, 0);
	elapsedClockCycles_HW = m_tickCounter2 - m_tickCounter1 - calibration;
	elapsedTimeInSeconds_HW = (double) elapsedClockCycles_HW * m_clockPeriodSeconds;

	for (i = 0; i < OUT_C_ROWS; i++) {
		for (j = 0; j < OUT_C_COLS; j++) {
			//if (u32_to_float(testResultMatrix_c_HW[i][j]) != referenceResultMatrix_c[i][j]) {
			if (testResultMatrix_c_HW[i][j] != referenceResultMatrix_c[i][j]) {
				error++;
			}
		}
	}
	if (error > 0) {
		printf("Error in HW calculation!\n Error = %d\n", error);
		printf("Result HW:\n");
		for (i = 0; i < OUT_C_ROWS; i++) {
			for (j = 0; j < OUT_C_COLS; j++) {
				printf("[%f]", testResultMatrix_c_HW[i][j]);
			}
			printf("\n");
		}
		printf("\n*************************************\n");
		return 0;
	}
	printf("No error in HW calculation!\n");
	printf("Run time for HW on PL = %f seconds (%d clock cycles)\n\n", elapsedTimeInSeconds_HW, elapsedClockCycles_HW);
	speedup = ((double) elapsedClockCycles_SW) / ((double) elapsedClockCycles_HW);
	printf("Speedup = %f\n", speedup);

	printf("\n*************************************\n");
	return 0;
}








