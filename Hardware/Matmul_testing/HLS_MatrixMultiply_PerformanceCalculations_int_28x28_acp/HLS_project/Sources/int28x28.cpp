//------------------------------------------//
//				  HLS INT28x28				//
//------------------------------------------//

#include "int28x28.h"

void matrixMultiplication(mat_a_t a[IN_A_ROWS][IN_A_COLS], mat_b_t b[IN_B_ROWS][IN_B_COLS], mat_c_t c[OUT_C_ROWS][OUT_C_COLS]) {
	// partition with half dimension since BRAM has two ports
	int FACTOR = IN_A_ROWS/2;
#pragma HLS INLINE off
#pragma HLS array_partition variable=a block factor=FACTOR dim=2
#pragma HLS array_partition variable=b block factor=FACTOR dim=1

	hlstest8_label0:for (int i = 0; i < IN_A_ROWS; i++) {
		hlstest8_label1:for (int j = 0; j < IN_B_COLS; j++) {
#pragma HLS PIPELINE II=1
			c[i][j] = 0;
			hlstest8_label2:for (int k = 0; k < IN_B_ROWS; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}

void int28x28(hls::stream<AXI_VALUE> &in_stream, hls::stream<AXI_VALUE> &out_stream) {
#pragma HLS RESOURCE variable=in_stream  core=AXIS metadata="-bus_bundle INPUT_STREAM"
#pragma HLS RESOURCE variable=out_stream core=AXIS metadata="-bus_bundle OUTPUT_STREAM"
#pragma HLS RESOURCE variable=return core=AXI4LiteS metadata="-bus_bundle CONTROL_BUS"

	// Define matrices
	mat_a_t matrix_a[IN_A_ROWS][IN_A_COLS];
	mat_b_t matrix_b[IN_B_ROWS][IN_B_COLS];
	mat_c_t matrix_c[OUT_C_ROWS][OUT_C_COLS];
	AXI_VALUE tempValue;
	int i, j, k;

	// Stream in the a matrix
	read_a1: for(i = 0; i < IN_A_ROWS; i++) {
		read_a2: for(j = 0; j < IN_A_COLS; j++) {
#pragma HLS PIPELINE // Pipeline the loop so that it reads in data in parallel
			in_stream.read(tempValue);
			//union {	unsigned int ival; mat_a_t oval; } converter; // For float conversion
			//converter.ival = tempValue.data; // For float
			//matrix_a[i][j] = converter.oval; // For float
			matrix_a[i][j] = tempValue.data; // For int
		}
	}

	// Stream in the b matrix
	read_b1: for(i = 0; i < IN_B_ROWS; i++) {
		read_b2: for(j = 0; j < IN_B_COLS; j++) {
#pragma HLS PIPELINE // Pipeline the loop so that it reads in data in parallel
			in_stream.read(tempValue);
			//union {	unsigned int ival; mat_b_t oval; } converter; // For float conversion
			//converter.ival = tempValue.data; // For float
			//matrix_b[i][j] = converter.oval; // For float
			matrix_b[i][j] = tempValue.data; // For int
		}
	}

	// Do matrix multiplication
	matrixMultiplication(matrix_a, matrix_b, matrix_c);

	// Stream out the result matrix (c matrix)
	write_c1: for(i = 0; i < OUT_C_ROWS; i++) {
		write_c2: for(j = 0; j < OUT_C_COLS; j++) {
#pragma HLS PIPELINE // Pipeline the loop so that it reads in data in parallel
			//union {	unsigned int oval; mat_c_t ival; } converter; // For float conversion
			//converter.ival = matrix_c[i][j]; // For float
			//tempValue.data = converter.oval; // For float
			tempValue.data = matrix_c[i][j]; // For int
			tempValue.last = ((i==OUT_C_ROWS-1)&&(j==OUT_C_COLS-1))? 1 : 0;
			tempValue.strb = -1;
			tempValue.keep = 15; //e.strb;
			tempValue.user = 0;
			tempValue.id = 0;
			tempValue.dest = 0;
			out_stream.write(tempValue);
		}
	}
}
