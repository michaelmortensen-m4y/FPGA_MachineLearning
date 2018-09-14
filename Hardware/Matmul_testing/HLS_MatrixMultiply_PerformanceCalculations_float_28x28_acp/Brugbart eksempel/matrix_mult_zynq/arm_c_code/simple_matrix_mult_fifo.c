//************************************************************
// Simple Matrix multiplication
// for advanced core. Send B matrix First.
//
// G. Sutter for teaching purpose.
// march 2014. For Vivado 2013.3
//
//************************************************************
#include <stdio.h>
#include "platform.h"
#include "xil_io.h"
#include "platform.h"
#include "xparameters.h"
#include "xaxidma.h"
#include "xtmrctr.h"
#include "simple_matrix_mult.h"

#include "xil_printf.h"

//#define PRINT_MED_TIME 1

// AXI DMA Instance
XAxiDma AxiDma;
// TIMER Instance
XTmrCtr timer_dev;
// a Matrix multiplicator instance
XMatrixmul_fifo_accel_core XMatrixmul_dev;
XMatrixmul_fifo_accel_core_Config XMatrixmul_config = { 0, XPAR_MATRIXMUL_FIFO_ACCEL_CORE_0_S_AXI_CONTROL_BUS_BASEADDR };

int init_dma(){
	XAxiDma_Config *CfgPtr;
	int status;

	CfgPtr = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
	if(!CfgPtr){
		print("Error looking for AXI DMA config\n\r");
		return XST_FAILURE;
	}
	status = XAxiDma_CfgInitialize(&AxiDma,CfgPtr);
	if(status != XST_SUCCESS){
		print("Error initializing DMA\n\r");
		return XST_FAILURE;
	}
	//check for scatter gather mode
	if(XAxiDma_HasSg(&AxiDma)){
		print("Error DMA configured in SG mode\n\r");
		return XST_FAILURE;
	}
	/* Disable interrupts, we use polling mode */
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DEVICE_TO_DMA);
	XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DMA_TO_DEVICE);

	return XST_SUCCESS;
}
//---------------------------------------------------------------------------
void matrix_multiply_SW(float a[DIM][DIM], float b[DIM][DIM], float res[DIM][DIM])
{
  int ia, ib, id;
  float sum;

  // matrix multiplication of a A*B matrix
  for (ia = 0; ia < DIM; ++ia)
     for (ib = 0; ib < DIM; ++ib)
     { 	 sum = 0;
		 for (id = 0; id < DIM; ++id)
			 sum += a[ia][id] * b[id][ib];
		 res[ia][ib] = sum;
     }
}
//---------------------------------------------------------------------------
void print_accel_status(void)
{
	int isDone, isIdle, isReady;

	isDone = XMatrixmul_fifo_accel_core_IsDone(&XMatrixmul_dev);
	isIdle = XMatrixmul_fifo_accel_core_IsIdle(&XMatrixmul_dev);
	isReady = XMatrixmul_fifo_accel_core_IsReady(&XMatrixmul_dev);
	xil_printf("MatMultAccel Status: isDone %d, isIdle %d, isReady%d\r\n", isDone, isIdle, isReady);
}
//---------------------------------------------------------------------------
int main(){
	int i, j;
	int status, err=0;
	float A[DIM][DIM], B[DIM][DIM];
	float res_hw[DIM][DIM], res_sw[DIM][DIM];
    float acc_factor;
	unsigned int dma_size = SIZE * sizeof(float);
	unsigned int init_time, curr_time, calibration;
	unsigned int begin_time, end_time;
	unsigned int run_time_sw, run_time_hw = 0;
	#ifdef PRINT_MED_TIME
	unsigned int time_1, time_2, time_3, time_4, time_5, time_6;
	#endif

	init_platform();

	xil_printf("\r********************************\n\r");
	xil_printf("\rFP MATRIX MULT in Vivado HLS + DMA, optimized\n\n\r");

	// Init DMA
	status = init_dma();

	if(status != XST_SUCCESS){
		print("\rError: DMA init failed\n");
		return XST_FAILURE;
	}
	print("\rDMA Init done\n\r");

	// Setup timer
	status = XTmrCtr_Initialize(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	if(status != XST_SUCCESS){ print("\rError: timer setup failed\n"); }
	XTmrCtr_SetOptions(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID, XTC_ENABLE_ALL_OPTION);

	// Calibrate timer
	XTmrCtr_Reset(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	init_time = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	curr_time = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	calibration = curr_time - init_time;
	xil_printf("Calibrating the timer:\r\n");
	xil_printf("init_time: %d cycles.\r\n", init_time);
	xil_printf("curr_time: %d cycles.\r\n", curr_time);
	xil_printf("calibration: %d cycles.\r\n", calibration);

	XTmrCtr_Reset(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	begin_time = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	for (i = 0; i< 10000; i++);
	end_time = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	run_time_sw = end_time - begin_time - calibration;
	xil_printf("Loop time for 10000 iterations is %d cycles.\r\n", run_time_sw);

	xil_printf("\rBefore Start\r\n");

	// input data; Matrix Initiation
	for(i = 0; i<DIM; i++)
		for(j = 0; j<DIM; j++)
		{
			A[i][j] = (float)(i+j);
			B[i][j] = (float)(i*j);
			res_sw[i][j] = 0;
			res_hw[i][j] = 0;
		}

	//call the software version of the function
	//print("\r now ARM is running the SW IP\n\r");

	XTmrCtr_Reset(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	begin_time = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);

	matrix_multiply_SW(A, B, res_sw);

	end_time = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	run_time_sw = end_time - begin_time - calibration;
	xil_printf("\r\nTotal run time for SW on ARM Processor (no SO) is %d cycles.\r\n", run_time_sw);

	// flush caches. IT´s necessary???. check it!!!
	Xil_DCacheFlushRange((unsigned int)A,dma_size);
	Xil_DCacheFlushRange((unsigned int)B,dma_size);
	Xil_DCacheFlushRange((unsigned int)res_hw,dma_size);

	// Run the HW Accelerator;
	status = XMatrixmul_fifo_accel_core_CfgInitialize(&XMatrixmul_dev, &XMatrixmul_config);
	if(status != XST_SUCCESS){
		xil_printf("Error: example setup failed\r\n");
		return XST_FAILURE;
	}
    // the interruption are not connected in fact.
	XMatrixmul_fifo_accel_core_InterruptGlobalDisable(&XMatrixmul_dev);
	XMatrixmul_fifo_accel_core_InterruptDisable(&XMatrixmul_dev, 1);

    print_accel_status();

    //start the accelerator
	XMatrixmul_fifo_accel_core_Start(&XMatrixmul_dev);

    print_accel_status();

	//flush the cache
	Xil_DCacheFlushRange((unsigned int)A,dma_size);
	Xil_DCacheFlushRange((unsigned int)B,dma_size);
	Xil_DCacheFlushRange((unsigned int)res_hw,dma_size);
	print("\rCache cleared\n\r");

	xil_printf("\rSetup HW accelerator done\r\n");

	XTmrCtr_Reset(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	begin_time = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);

	//Run_HW_Accelerator(A, B, res_hw, dma_size);

	//transfer B to the Vivado HLS block. B first!!!!
	status = XAxiDma_SimpleTransfer(&AxiDma, (unsigned int) B, dma_size, XAXIDMA_DMA_TO_DEVICE);
	if (status != XST_SUCCESS) {
		xil_printf("Error: DMA transfer matrix B to Vivado HLS block failed\n");
		return XST_FAILURE;
	}
	#ifdef PRINT_MED_TIME
	time_1 = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	#endif
	/* Wait for transfer to be done */
	while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) ;
	#ifdef PRINT_MED_TIME
	time_2 = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	#endif

	//transfer A to the Vivado HLS block
	status = XAxiDma_SimpleTransfer(&AxiDma, (unsigned int) A, dma_size, XAXIDMA_DMA_TO_DEVICE);
	if (status != XST_SUCCESS) {
		xil_printf("Error: DMA transfer matrix A to Vivado HLS block failed\n");
		return XST_FAILURE;
	}
	#ifdef PRINT_MED_TIME
	time_3 = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	#endif
    /* It has no sense to Wait for transfer to be done */

	//get results from the Vivado HLS block
	status = XAxiDma_SimpleTransfer(&AxiDma, (unsigned int) res_hw, dma_size, XAXIDMA_DEVICE_TO_DMA);
	if (status != XST_SUCCESS) {
		xil_printf("Error: DMA transfer from Vivado HLS block failed\n");
		return XST_FAILURE;
	}
	#ifdef PRINT_MED_TIME
	time_5 = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	#endif

	/* It has no sense to Wait for transfer to be done, we only use the measure the time */
	#ifdef PRINT_MED_TIME
	while (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) ;
	time_4 = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	#endif
	/* Wait for transfer to be done */
	while (XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) ;
    #ifdef PRINT_MED_TIME
	time_6 = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	#endif
	//while ((XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA)) || (XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE))) ;

	end_time = XTmrCtr_GetValue(&timer_dev, XPAR_AXI_TIMER_0_DEVICE_ID);
	run_time_hw = end_time - begin_time - calibration;
	xil_printf("Total run time for AXI DMA + HW accelerator is %d cycles.\r\n",	run_time_hw);

	print_accel_status();

	Xil_DCacheFlushRange((unsigned int)res_hw,dma_size);
	//Compare the results from sw and hw
	for (i = 0; i < DIM; i++)
		for (j = 0; j < DIM; j++)
			if (res_sw[i][j] != res_hw[i][j]) {
				err++;
			}

	if (err == 0)
		xil_printf("\rSW and HW results match!\n\r");
	else
		xil_printf("\rERROR: %d results mismatch\n\r", err);

	// HW vs. SW speedup factor
	acc_factor = (float) run_time_sw / (float) run_time_hw;
	xil_printf("\r\033[1mAcceleration factor: %d.%d \033[0m\r\n\r\n",
			(int) acc_factor, (int) (acc_factor * 1000) % 1000);

    #ifdef PRINT_MED_TIME
	xil_printf("Time waiting to send 1st matrix: %d cycles\r\n", time_2 - time_1);
	xil_printf("Time waiting to send 2nd matrix: %d cycles\r\n", time_4 - time_3);
	xil_printf("Time waiting to receive results: %d cycles\r\n", time_6 - time_5);
	#endif
	cleanup_platform();

	return  err;
}
