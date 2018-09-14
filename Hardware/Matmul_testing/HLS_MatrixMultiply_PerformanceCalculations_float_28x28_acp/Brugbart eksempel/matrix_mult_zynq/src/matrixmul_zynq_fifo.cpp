//*****************************************************************************
// Matrix multiplication example
// Zynq coprocessor
//
// by G.Sutter for teaching purposes
// Revision History:
//  - March 2014, for Vivado HLS 2013.3
//  - June 2014, for Vivado HLS 2014.1
//
// Receives B matrix first. Then A row by row.
// uses half BRAMs resources and half computation time than previous.
//
// There are several implementations
//
//*****************************************************************************

#include "matrixmul_zynq_fifo.h"
using namespace hls;
#define VER_2


#ifdef VER_2
// --------------------------------------------------------
// main accelerator function, interfaces with AXI-S channels
void matrixmul_fifo_accel_core (
	stream<AXI_VALUE> &in_stream,
	stream<AXI_VALUE> &out_stream)
{

// Map HLS ports to AXI interfaces
#pragma HLS RESOURCE variable=in_stream  core=AXIS metadata="-bus_bundle INPUT_STREAM"
#pragma HLS RESOURCE variable=out_stream core=AXIS metadata="-bus_bundle OUTPUT_STREAM"
#pragma HLS RESOURCE variable=return core=AXI4LiteS metadata="-bus_bundle CONTROL_BUS"

	mat_b_t mat_b[MAT_DIM][MAT_DIM];
    mat_a_t a_row[MAT_DIM];
	mat_a_t a_row_next[MAT_DIM];

#pragma HLS ARRAY_PARTITION variable=a_row block factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=a_row_next block factor=8 dim=1
int const FACTOR = MAT_DIM/2;
#pragma HLS array_partition variable=mat_b block factor=FACTOR dim=1

	result_t accum;
	result_t accum0, accum1, accum2, accum3, accum4, accum5, accum6, accum7, accum8, accum9 ;
	result_t accum10, accum11, accum12, accum13, accum14, accum15, accum16, accum17, accum18, accum19;
	AXI_VALUE aValue;
	int i, j;


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

	Cache_Row_a_0: for(int k = 0; k < MAT_DIM; k++){
#pragma HLS PIPELINE
	 in_stream.read(aValue);
		union {	unsigned int ival; mat_a_t oval; } converter;
		converter.ival = aValue.data;
		a_row[k]= converter.oval;
		//a_row[k] = aValue.data;
	}

	// Iterate over the rows of the A matrix
	Row: for(int i = 0; i < MAT_DIM; i++) {
	// Iterate over the columns of the B matrix
	  Col: for(int j = 0; j < MAT_DIM; j++) {
#pragma HLS PIPELINE II=1
		// Do the inner product of a row of A and col of B
		//accum = 0;
		if (i < MAT_DIM-1){
			in_stream.read(aValue);
			union {	unsigned int ival; mat_a_t oval; } converter;
			converter.ival = aValue.data;
			if (!(i%2))
			   a_row_next[j] = converter.oval;
			else
			   a_row[j] = converter.oval;
		}

		//Product: for(int k = 0; k < MAT_DIM; k++)
		{
		if (!(i%2)){
		  accum0 =  (a_row[0] * mat_b[0][j])   + (a_row[1] * mat_b[1][j]) ;
		  accum1 =  (a_row[2] * mat_b[2][j])   + (a_row[3] * mat_b[3][j]) ;
		  accum2 =  (a_row[4] * mat_b[4][j])   + (a_row[5] * mat_b[5][j]) ;
		  accum3 =  (a_row[6] * mat_b[6][j])   + (a_row[7] * mat_b[7][j]) ;
		  accum4 =  (a_row[8] * mat_b[8][j])   + (a_row[9] * mat_b[9][j]) ;
		  accum5 =  (a_row[10] * mat_b[10][j]) + (a_row[11] * mat_b[11][j]) ;
		  accum6 =  (a_row[12] * mat_b[12][j]) + (a_row[13] * mat_b[13][j]) ;
		  accum7 =  (a_row[14] * mat_b[14][j]) + (a_row[15] * mat_b[15][j]) ;
		  accum8 =  (a_row[16] * mat_b[16][j]) + (a_row[17] * mat_b[17][j]) ;
		  accum9 =  (a_row[18] * mat_b[18][j]) + (a_row[19] * mat_b[19][j]) ;
		  accum10 = (a_row[20] * mat_b[20][j]) + (a_row[21] * mat_b[21][j]) ;
		  accum11 = (a_row[22] * mat_b[22][j]) + (a_row[23] * mat_b[23][j]) ;
		  accum12 = (a_row[24] * mat_b[24][j]) + (a_row[25] * mat_b[25][j]) ;
		  accum13 = (a_row[26] * mat_b[26][j]) + (a_row[27] * mat_b[27][j]) ;
		  accum14 = (a_row[28] * mat_b[28][j]) + (a_row[29] * mat_b[29][j]) ;
		  accum15 = (a_row[30] * mat_b[30][j]) + (a_row[31] * mat_b[31][j]) ;
		  accum16 = (accum0 + accum1) + (accum2 + accum3);
		  accum17 = (accum4 + accum5) + (accum6 + accum7);
		  accum18 = (accum8 + accum9) + (accum10 + accum11);
		  accum19 = (accum12 + accum13) + (accum14 + accum15);
		  accum = (accum16 + accum17) + (accum18 + accum19);
		}
		else{
			  accum0 =  (a_row_next[0] * mat_b[0][j])   + (a_row_next[1] * mat_b[1][j]) ;
			  accum1 =  (a_row_next[2] * mat_b[2][j])   + (a_row_next[3] * mat_b[3][j]) ;
			  accum2 =  (a_row_next[4] * mat_b[4][j])   + (a_row_next[5] * mat_b[5][j]) ;
			  accum3 =  (a_row_next[6] * mat_b[6][j])   + (a_row_next[7] * mat_b[7][j]) ;
			  accum4 =  (a_row_next[8] * mat_b[8][j])   + (a_row_next[9] * mat_b[9][j]) ;
			  accum5 =  (a_row_next[10] * mat_b[10][j]) + (a_row_next[11] * mat_b[11][j]) ;
			  accum6 =  (a_row_next[12] * mat_b[12][j]) + (a_row_next[13] * mat_b[13][j]) ;
			  accum7 =  (a_row_next[14] * mat_b[14][j]) + (a_row_next[15] * mat_b[15][j]) ;
			  accum8 =  (a_row_next[16] * mat_b[16][j]) + (a_row_next[17] * mat_b[17][j]) ;
			  accum9 =  (a_row_next[18] * mat_b[18][j]) + (a_row_next[19] * mat_b[19][j]) ;
			  accum10 = (a_row_next[20] * mat_b[20][j]) + (a_row_next[21] * mat_b[21][j]) ;
			  accum11 = (a_row_next[22] * mat_b[22][j]) + (a_row_next[23] * mat_b[23][j]) ;
			  accum12 = (a_row_next[24] * mat_b[24][j]) + (a_row_next[25] * mat_b[25][j]) ;
			  accum13 = (a_row_next[26] * mat_b[26][j]) + (a_row_next[27] * mat_b[27][j]) ;
			  accum14 = (a_row_next[28] * mat_b[28][j]) + (a_row_next[29] * mat_b[29][j]) ;
			  accum15 = (a_row_next[30] * mat_b[30][j]) + (a_row_next[31] * mat_b[31][j]) ;
			  accum16 = (accum0 + accum1) + (accum2 + accum3);
			  accum17 = (accum4 + accum5) + (accum6 + accum7);
			  accum18 = (accum8 + accum9) + (accum10 + accum11);
			  accum19 = (accum12 + accum13) + (accum14 + accum15);
			  accum = (accum16 + accum17) + (accum18 + accum19);
			}

		}

		union {	unsigned int oval; result_t ival; } converter;
		converter.ival = accum; //res[i][j];;
		aValue.data = converter.oval;
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
#endif

#ifdef VER_1
// --------------------------------------------------------
// main accelerator function, interfaces with AXI-S channels
void matrixmul_fifo_accel_core (
	stream<AXI_VALUE> &in_stream,
	stream<AXI_VALUE> &out_stream)
{

// Map HLS ports to AXI interfaces
#pragma HLS RESOURCE variable=in_stream  core=AXIS metadata="-bus_bundle INPUT_STREAM"
#pragma HLS RESOURCE variable=out_stream core=AXIS metadata="-bus_bundle OUTPUT_STREAM"
#pragma HLS RESOURCE variable=return core=AXI4LiteS metadata="-bus_bundle CONTROL_BUS"

	mat_b_t mat_b[MAT_DIM][MAT_DIM];
    mat_a_t a_row[MAT_DIM];
	mat_a_t a_row_next[MAT_DIM];

#pragma HLS ARRAY_PARTITION variable=a_row complete dim=0
#pragma HLS ARRAY_PARTITION variable=a_row_next complete dim=0
int const FACTOR = MAT_DIM/2;
#pragma HLS array_partition variable=mat_b block factor=FACTOR dim=1

	result_t accum, prod;
	AXI_VALUE aValue;
	int i, j;


	// stream in second matrix
	read_b1: for(i=0; i< MAT_DIM; i++){
		read_b2: for(j=0; j< MAT_DIM; j++){
#pragma HLS PIPELINE
			in_stream.read(aValue);
			mat_b[i][j] = aValue.data;
		}
	}

	Cache_Row_a_0: for(int k = 0; k < MAT_DIM; k++){
#pragma HLS PIPELINE
	 in_stream.read(aValue);
	 a_row[k] = aValue.data;
	}

	// Iterate over the rows of the A matrix
	Row: for(int i = 0; i < MAT_DIM; i++) {
	// Iterate over the columns of the B matrix
	  Col: for(int j = 0; j < MAT_DIM; j++) {
#pragma HLS PIPELINE II=1
		// Do the inner product of a row of A and col of B
		accum = 0;
//		if (i < MAT_DIM-1){
//			in_stream.read(aValue);
//			a_row_next[j] = aValue.data;
//		}
		if (i < MAT_DIM-1){
			in_stream.read(aValue);
			if (!(i%2))
			   a_row_next[j] = aValue.data;
			else
			   a_row[j] = aValue.data;
		}
		Product: for(int k = 0; k < MAT_DIM; k++) {
			if (!(i%2))
				prod = a_row[k] * mat_b[k][j];
			else
			    prod = a_row_next[k] * mat_b[k][j];
			accum += prod;
		}
		aValue.data = accum; //res[i][j] = accum;
		aValue.last = ((i==MAT_DIM-1)&&(j==MAT_DIM-1))? 1 : 0;
		out_stream.write(aValue);
	  }
//	copy_elem: for(int k = 0; k < MAT_DIM; k++) {
//#pragma HLS UNROLL
//		a_row[k] = a_row_next[k]; }
   }
}
#endif

#ifdef VER_0
// --------------------------------------------------------
//It's not working yet.
//The unrolled copy of a_row gives wrong results.
// main accelerator function, interfaces with AXI-S channels
void matrixmul_fifo_accel_core (
	stream<AXI_VALUE> &in_stream,
	stream<AXI_VALUE> &out_stream)
{

// Map HLS ports to AXI interfaces
#pragma HLS RESOURCE variable=in_stream  core=AXIS metadata="-bus_bundle INPUT_STREAM"
#pragma HLS RESOURCE variable=out_stream core=AXIS metadata="-bus_bundle OUTPUT_STREAM"
#pragma HLS RESOURCE variable=return core=AXI4LiteS metadata="-bus_bundle CONTROL_BUS"

	mat_b_t mat_b[MAT_DIM][MAT_DIM];
    mat_a_t a_row[MAT_DIM];
	mat_a_t a_row_next[MAT_DIM];

#pragma HLS ARRAY_PARTITION variable=a_row complete dim=0
#pragma HLS ARRAY_PARTITION variable=a_row_next complete dim=0
int const FACTOR = MAT_DIM/2;
#pragma HLS array_partition variable=mat_b block factor=FACTOR dim=1

	result_t accum;
	AXI_VALUE aValue;
	int i, j;


	// stream in second matrix
	read_b1: for(i=0; i< MAT_DIM; i++){
		read_b2: for(j=0; j< MAT_DIM; j++){
#pragma HLS PIPELINE
			in_stream.read(aValue);
			mat_b[i][j] = aValue.data;
		}
	}

	Cache_Row_a_0: for(int k = 0; k < MAT_DIM; k++){
#pragma HLS PIPELINE
	 in_stream.read(aValue);
	 a_row[k] = aValue.data;
	}

	// Iterate over the rows of the A matrix
	Row: for(int i = 0; i < MAT_DIM; i++) {
	// Iterate over the columns of the B matrix
	  Col: for(int j = 0; j < MAT_DIM; j++) {
#pragma HLS PIPELINE II=1
		// Do the inner product of a row of A and col of B
		accum = 0;
		if (i < MAT_DIM-1){
			in_stream.read(aValue);
			a_row_next[j] = aValue.data;
		}
		Product: for(int k = 0; k < MAT_DIM; k++) {
		  accum += a_row[k] * mat_b[k][j];
		}
		aValue.data = accum; //res[i][j] = accum;
		aValue.last = ((i==MAT_DIM-1)&&(j==MAT_DIM-1))? 1 : 0;
		out_stream.write(aValue);
	  }
	copy_elem: for(int k = 0; k < MAT_DIM; k++) {
#pragma HLS UNROLL
		a_row[k] = a_row_next[k]; }
	}
}
#endif
