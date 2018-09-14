//*****************************************************************************
// Based on a original example from Xilinx
//
// Modified by G.Sutter for teaching purposes
// Revision History:
//  - march 2012, for AutoESL 2011.4.2
//  - dic 2012, for Vivado HLS 2012.3
//  - march 2014, for Vivado HLS 2013.3
//
//*****************************************************************************
#ifndef __MATRIXMUL_F_H__
#define __MATRIXMUL_F_H__

#include <cmath>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

using namespace std;

//#define MAT_A_ROWS 32
//#define MAT_A_COLS 32
//#define MAT_B_ROWS 32
//#define MAT_B_COLS 32
#define MAT_DIM 32

typedef float mat_a_t;
typedef float mat_b_t;
typedef float result_t;

typedef ap_axiu<32,4,5,5> AXI_VALUE;

// Prototype of top level function for C-synthesis
/*
void matrixmul(
      mat_a_t a[MAT_DIM][MAT_DIM],
      mat_b_t b[MAT_DIM][MAT_DIM],
      result_t res[MAT_DIM][MAT_DIM]);

void matrixmul_accel_core (
	hls::stream<AXI_VALUE> &in_stream,
	hls::stream<AXI_VALUE> &out_stream);
*/

void matrixmul_fifo_accel_core (
	hls::stream<AXI_VALUE> &in_stream,
	hls::stream<AXI_VALUE> &out_stream);

#endif // __MATRIXMUL_F_H__ not defined

