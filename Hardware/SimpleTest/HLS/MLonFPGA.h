//------------------------------------------//
//				  HLS           			//
//------------------------------------------//

#include <cmath>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#define TENSOR_ROWS 4
#define TENSOR_COLS 4
#define TENSOR_MATS 3 // 30000/Wstep = 300 for Wstep = 100

#define X1_ROWS 4
#define X1_COLS 80 // 4*D = 80 for D = 20

#define X2_ROWS 4
#define X2_COLS 80 // 4*D = 80 for D = 20

#define X3_ROWS 16
#define X3_COLS 16

// Declare 32-bit integer with side-channels <TID,TUSR,TKEEP,TLAST> to help interface with other cores (ex. DMA)
typedef ap_axiu<32,1,1,1> AXI_VALUE;

// Prototype of top level function for C-synthesis
unsigned infer(int tensor[TENSOR_ROWS][TENSOR_COLS][TENSOR_MATS], unsigned cl);

unsigned infer_core(hls::stream<AXI_VALUE> &in_stream_tensor);
