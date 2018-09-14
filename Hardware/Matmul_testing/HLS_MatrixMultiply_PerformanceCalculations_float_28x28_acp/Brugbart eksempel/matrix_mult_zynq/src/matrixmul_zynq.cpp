//*****************************************************************************
// Matrix multiplication example
// Zynq coprocessor
// by G.Sutter for teaching purposes
// Revision History:
//  - March 2014, for Vivado HLS 2013.3
//  - June 2014, for Vivado HLS 2014.1
//
//*****************************************************************************

#include "matrixmul_zynq.h"

using namespace hls;

void matrixmul(
      mat_a_t a[MAT_DIM][MAT_DIM],
      mat_b_t b[MAT_DIM][MAT_DIM],
      result_t res[MAT_DIM][MAT_DIM])
{
// partition with half dimension since BRAM has two ports
int const FACTOR = MAT_DIM/2;
#pragma HLS INLINE off
#pragma HLS array_partition variable=a block factor=FACTOR dim=2
#pragma HLS array_partition variable=b block factor=FACTOR dim=1

  result_t accum;
  // Iterate over the rows of the A matrix
  Row: for(int i = 0; i < MAT_DIM; i++) {
    // Iterate over the columns of the B matrix

    Col: for(int j = 0; j < MAT_DIM; j++) {
      // Do the inner product of a row of A and col of B
#pragma HLS PIPELINE II=1
      accum = 0;
      Prod: for(int k = 0; k < MAT_DIM; k++) {
        accum += a[i][k] * b[k][j];
        res[i][j] = accum; //if (k == (MAT_B_ROWS-1)) res[i][j] = accum;
      }

    }
  }
}

// --------------------------------------------------------
// main accelerator function, interfaces with AXI-S channels
void matrixmul_accel_core (
	stream<AXI_VALUE> &in_stream,
	stream<AXI_VALUE> &out_stream)
{

	// Map HLS ports to AXI interfaces
#pragma HLS RESOURCE variable=in_stream  core=AXIS metadata="-bus_bundle INPUT_STREAM"
#pragma HLS RESOURCE variable=out_stream core=AXIS metadata="-bus_bundle OUTPUT_STREAM"
#pragma HLS RESOURCE variable=return core=AXI4LiteS metadata="-bus_bundle CONTROL_BUS"


	mat_a_t mat_a[MAT_DIM][MAT_DIM];
	mat_b_t mat_b[MAT_DIM][MAT_DIM];
    result_t mat_res[MAT_DIM][MAT_DIM];
    AXI_VALUE aValue;
    int i, j;

	// stream in first matrix
	read_a1: for(i=0; i< MAT_DIM; i++){
		read_a2: for(j=0; j< MAT_DIM; j++){
#pragma HLS PIPELINE
			in_stream.read(aValue);
			union {	unsigned int ival; mat_a_t oval; } converter;
			converter.ival = aValue.data;
			mat_a[i][j] = converter.oval;
			//mat_a[i][j] = aValue.data;
		}
	}

	// stream in second matrix
	read_b1: for(i=0; i< MAT_DIM; i++){
		read_b2: for(j=0; j< MAT_DIM; j++){
#pragma HLS PIPELINE
			in_stream.read(aValue);
			union {	unsigned int ival; mat_b_t oval; } converter;
			converter.ival = aValue.data;
			mat_b[i][j] = converter.oval;
			//mat_b[i][j] = aValue.data;
		}
	}

	// do Matrix multiplication
	matrixmul(mat_a, mat_b, mat_res);

	// stream out result matrix
	write_res1: for(i=0; i< MAT_DIM; i++){
		write_res2: for(j=0; j< MAT_DIM; j++){
#pragma HLS PIPELINE
			union {	unsigned int oval; result_t ival; } converter;
			converter.ival = mat_res[i][j];;
			aValue.data = converter.oval;
			//aValue.data = mat_res[i][j];
			aValue.last = ((i==MAT_DIM-1)&&(j==MAT_DIM-1))? 1 : 0;
			aValue.strb = -1;
			aValue.keep = 15; //e.strb;
			aValue.user = 0;
			aValue.id = 0;
			aValue.dest = 0;
			out_stream.write(aValue);
		}
	}
}


