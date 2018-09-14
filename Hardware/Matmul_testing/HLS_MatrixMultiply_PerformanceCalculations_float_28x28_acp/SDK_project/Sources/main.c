/*
 * main.c
 *
 *  Created on: Jan 13, 2017
 *      Author: Michael Mortensen
 */

// NB: Stack and heap size has to be changed from the default 1KB to 2KB for both

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <xparameters.h>
#include "xaxidma.h"
#include "xil_types.h" // Used for timer
#include "xtmrctr.h" // Used for timer
#include <xfloat28x28.h>

#define IN_A_ROWS 28
#define IN_A_COLS 28
#define IN_B_ROWS 28
#define IN_B_COLS 28
#define OUT_C_ROWS 28
#define OUT_C_COLS 28

// float28x28 SW version
void float28x28(float a[IN_A_ROWS][IN_A_COLS], float b[IN_B_ROWS][IN_B_COLS], float c[OUT_C_ROWS][OUT_C_COLS]) {
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
	printf("\n********** Matrix Multiply **********\n\n");
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

	// Initialize the Float28x28 IP core
	int status;
	XFloat28x28 doFloat28x28;
	XFloat28x28_Config XFloat28x28_config = { 0, XPAR_FLOAT28X28_0_S_AXI_CONTROL_BUS_BASEADDR };

	// Initialize the DMA
	XAxiDma AxiDma;
	XAxiDma_Config *CfgPtr;
	CfgPtr = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
	if(!CfgPtr){
		printf("Error looking for AXI DMA config\n\r");
		printf("\n*************************************\n");
		return XST_FAILURE;
	}
	status = XAxiDma_CfgInitialize(&AxiDma,CfgPtr);
	if(status != XST_SUCCESS){
		printf("Error initializing DMA\n\r");
		printf("\n*************************************\n");
		return XST_FAILURE;
	}
	//check for scatter gather mode
	if(XAxiDma_HasSg(&AxiDma)){
		printf("Error DMA configured in SG mode\n\r");
		printf("\n*************************************\n");
		return XST_FAILURE;
	}
	// Disable interrupts, we use polling mode
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DMA_TO_DEVICE);

	// Fill test matrices
	float testMatrix_a_SW[IN_A_ROWS][IN_A_COLS];
	float testMatrix_b_SW[IN_B_ROWS][IN_B_COLS];
	float testResultMatrix_c_SW[OUT_C_ROWS][OUT_C_COLS];
	float testResultMatrix_c_HW[OUT_C_ROWS][OUT_C_COLS];
	float referenceResultMatrix_c[OUT_C_ROWS][OUT_C_COLS];
    // Generate test matrices and reference matrix
    for (int i = 0; i < IN_A_ROWS; i++) {
        for (int j = 0; j < IN_A_COLS; j++) {
        	testMatrix_a_SW[i][j] = (float)i+j;
        }
    }
    for (int i = 0; i < IN_B_ROWS; i++) {
        for (int j = 0; j < IN_B_COLS; j++) {
        	testMatrix_b_SW[i][j] = (float)i*j;
        }
    }
	for (int i = 0; i < IN_A_ROWS; i++) {
		for (int j = 0; j < IN_B_COLS; j++) {
			referenceResultMatrix_c[i][j] = 0;
			for (int k = 0; k < IN_B_ROWS; k++) {
				referenceResultMatrix_c[i][j] += testMatrix_a_SW[i][k] * testMatrix_b_SW[k][j];
			}
		}
	}
	unsigned int dma_size = (OUT_C_ROWS*OUT_C_COLS) * sizeof(float);


	// Run on software
	printf("Running on SW:\n");
	float error = 0;

	//*** Start timer 0
	XTmrCtr_Reset(&m_AxiTimer,0);
	m_tickCounter1 = XTmrCtr_GetValue(&m_AxiTimer, 0);
	XTmrCtr_Start(&m_AxiTimer, 0);

	float28x28(testMatrix_a_SW, testMatrix_b_SW, testResultMatrix_c_SW);

	//*** Stop timer 0
	XTmrCtr_Stop(&m_AxiTimer,0);
	m_tickCounter2 = XTmrCtr_GetValue(&m_AxiTimer, 0);
	elapsedClockCycles_SW = m_tickCounter2 - m_tickCounter1 - calibration;
	elapsedTimeInSeconds_SW = (double) elapsedClockCycles_SW * m_clockPeriodSeconds;

	// Check for errors
	for (int i = 0; i < OUT_C_ROWS; i++) {
		for (int j = 0; j < OUT_C_COLS; j++) {
			error += fabsf(testResultMatrix_c_SW[i][j] - referenceResultMatrix_c[i][j]);
		}
	}
	if (error > 0) {
		printf("Error in SW calculation!\n Error = %f\n", error);
		printf("\n*************************************\n");
		return 1;
	}
	//printf("Result SW:\n");
	for (int i = 0; i < OUT_C_ROWS; i++) {
		for (int j = 0; j < OUT_C_COLS; j++) {
			//printf("[%d]", testResultMatrix_c_SW[i][j]);
		}
		//printf("\n");
	}
	printf("No error in SW calculation!\n");
	printf("Run time for SW on ARM Processor = %f seconds (%d clock cycles)\n\n", elapsedTimeInSeconds_SW, elapsedClockCycles_SW);

	// Reset result and error count
	printf("Resetting variables\n\n");
	for (int i = 0; i < OUT_C_ROWS; i++) {
		for (int j = 0; j < OUT_C_COLS; j++) {
			testResultMatrix_c_SW[i][j] = 0;
			testResultMatrix_c_HW[i][j] = 0;
		}
	}
	error = 0;

	// Run on hardware
	printf("Running on HW:\n");

	// flush caches
	Xil_DCacheFlushRange((unsigned int)testMatrix_a_SW,dma_size);
	Xil_DCacheFlushRange((unsigned int)testMatrix_b_SW,dma_size);
	Xil_DCacheFlushRange((unsigned int)testResultMatrix_c_HW,dma_size);

	status = XFloat28x28_CfgInitialize(&doFloat28x28, &XFloat28x28_config);
	if (status != XST_SUCCESS) {
		printf("Error initializing for doHlstest7\n");
		return status;
		printf("\n*************************************\n");
	}
	// The interruption are not connected
	XFloat28x28_InterruptGlobalDisable(&doFloat28x28);
	XFloat28x28_InterruptDisable(&doFloat28x28, 1);

	XFloat28x28_Start(&doFloat28x28);

	// flush caches
	Xil_DCacheFlushRange((unsigned int)testMatrix_a_SW,dma_size);
	Xil_DCacheFlushRange((unsigned int)testMatrix_b_SW,dma_size);
	Xil_DCacheFlushRange((unsigned int)testResultMatrix_c_HW,dma_size);

	//*** Start timer 0
	XTmrCtr_Reset(&m_AxiTimer,0);
	m_tickCounter1 = XTmrCtr_GetValue(&m_AxiTimer, 0);
	XTmrCtr_Start(&m_AxiTimer, 0);

	// Transfer testMatrix_a_SW to the HW Accelerator
	status = XAxiDma_SimpleTransfer(&AxiDma, (unsigned int) testMatrix_a_SW, dma_size, XAXIDMA_DMA_TO_DEVICE);
	if (status != XST_SUCCESS) {
		printf("Error: DMA transfer matrix testMatrix_a_SW to HW accelerator failed\n");
		printf("\n*************************************\n");
		return XST_FAILURE;
	}

	// Wait for transfer to be done
	while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) ;

	// Transfer testMatrix_b_SW to the HW Accelerator
	status = XAxiDma_SimpleTransfer(&AxiDma, (unsigned int) testMatrix_b_SW, dma_size, XAXIDMA_DMA_TO_DEVICE);
	if (status != XST_SUCCESS) {
		printf("Error: DMA transfer matrix testMatrix_b_SW to HW accelerator failed\n");
		printf("\n*************************************\n");
		return XST_FAILURE;
	}

	// Wait for transfer to be done
	while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE));
	// Wait for a little while more (but why?)
	for (int i = 0; i < 700; i++){}

	// Get results from the HW Accelerator
	status = XAxiDma_SimpleTransfer(&AxiDma, (unsigned int) testResultMatrix_c_HW, dma_size, XAXIDMA_DEVICE_TO_DMA);
	if (status != XST_SUCCESS) {
		printf("Error: DMA transfer from HW accelerator failed\n");
		printf("\n*************************************\n");
		return XST_FAILURE;
	}

	// Wait for transfer to be done
	while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE));
	// Wait for a little while more (but why?)
	//for (int i = 0; i < 300; i++){}

	//Xil_DCacheFlushRange((unsigned int)testResultMatrix_c_HW,dma_size);

	//*** Stop timer 0
	XTmrCtr_Stop(&m_AxiTimer,0);
	m_tickCounter2 = XTmrCtr_GetValue(&m_AxiTimer, 0);
	elapsedClockCycles_HW = m_tickCounter2 - m_tickCounter1 - calibration;
	elapsedTimeInSeconds_HW = (double) elapsedClockCycles_HW * m_clockPeriodSeconds;

	//Xil_DCacheFlushRange((unsigned int)testResultMatrix_c_HW,dma_size);

	// Check for errors
	for (int i = 0; i < OUT_C_ROWS; i++) {
		for (int j = 0; j < OUT_C_COLS; j++) {
			if (testResultMatrix_c_HW[i][j] != referenceResultMatrix_c[i][j]) {
				error++;
			}
			//error += fabsf(testResultMatrix_c_HW[i][j] - referenceResultMatrix_c[i][j]);
			//if (i == 0) {
			//	printf("|%f - %f| = %f\n", testResultMatrix_c_HW[i][j], referenceResultMatrix_c[i][j], fabsf(testResultMatrix_c_HW[i][j] - referenceResultMatrix_c[i][j]));
			//}
		}
	}
	if (error > 0) {
		printf("Error in HW calculation!\n Error = %f\n", error);
		printf("\n*************************************\n");
		return 1;
	}
	//printf("Result HW:\n");
	for (int i = 0; i < OUT_C_ROWS; i++) {
		for (int j = 0; j < OUT_C_COLS; j++) {
			//printf("[%d]", testResultMatrix_c_HW[i][j]);
		}
		//printf("\n");
	}
	printf("No error in HW calculation!\n");
	printf("Run time for HW on PL = %f seconds (%d clock cycles)\n\n", elapsedTimeInSeconds_HW, elapsedClockCycles_HW);
	speedup = ((double) elapsedClockCycles_SW) / ((double) elapsedClockCycles_HW);
	printf("Speedup = %f\n", speedup);

	printf("\n*************************************\n");
	return 0;
}



