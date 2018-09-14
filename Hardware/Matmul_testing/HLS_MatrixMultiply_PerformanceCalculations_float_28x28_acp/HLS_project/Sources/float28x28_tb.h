//------------------------------------------//
//			   HLS FLOAT28x28 Bench			//
//------------------------------------------//

#include <cmath>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#define IN_A_ROWS 28
#define IN_A_COLS 28
#define IN_B_ROWS 28
#define IN_B_COLS 28
#define OUT_C_ROWS 28
#define OUT_C_COLS 28

typedef float mat_a_t; // a in matrix
typedef float mat_b_t; // b in matrix
typedef float mat_c_t; // Result matrix

// Declare 32-bit integer with side-channels <TID,TUSR,TKEEP,TLAST> to help interface with other cores (ex. DMA) (but is not used in this example)
typedef ap_axiu<32,4,5,5> AXI_VALUE;

// Prototype of top level function for C-synthesis
void matrixMultiplication(mat_a_t a[IN_A_ROWS][IN_A_COLS], mat_b_t b[IN_B_ROWS][IN_B_COLS], mat_c_t c[OUT_C_ROWS][OUT_C_COLS]);

void float28x28(hls::stream<AXI_VALUE> &in_stream, hls::stream<AXI_VALUE> &out_stream);

