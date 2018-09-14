//------------------------------------------//
//			   HLS INT28x28 Bench			//
//------------------------------------------//

#include <stdio.h>
#include <math.h>
#include "int28x28_tb.h"

int main() {
	printf("Testing started\n");

	// Declare test matrices and other stuff
	mat_a_t testMatrix_a[IN_A_ROWS][IN_A_COLS];
	mat_b_t testMatrix_b[IN_B_ROWS][IN_B_COLS];
	mat_c_t testResultMatrix_c[OUT_C_ROWS][OUT_C_COLS];
	mat_c_t referenceResultMatrix_c[OUT_C_ROWS][OUT_C_COLS];
	int error = 0;
	int i, j, k;
	hls::stream<AXI_VALUE> in_stream;
    hls::stream<AXI_VALUE> out_stream;
    AXI_VALUE tempValue;

    // Generate test matrices and reference matrix
    for (i = 0; i < IN_A_ROWS; i++) {
        for (j = 0; j < IN_A_COLS; j++) {
        	testMatrix_a[i][j] = i+j;
        }
    }
    for (i = 0; i < IN_B_ROWS; i++) {
        for (j = 0; j < IN_B_COLS; j++) {
        	testMatrix_b[i][j] = i*j;
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

	// Write test matrix a to input stream
	for(i = 0; i < IN_A_ROWS; i++) {
		for(j = 0; j < IN_A_COLS; j++) {
			//union {	unsigned int oval; mat_a_t ival; } converter; // For float conversion
			//converter.ival = testMatrix_a[i][j]; // For float
			//tempValue.data = converter.oval; // For float
			tempValue.data = testMatrix_a[i][j]; // For int
			in_stream.write(tempValue);
		}
	}

	// Write test matrix b to input stream
	for(i = 0; i < IN_B_ROWS; i++) {
		for(j = 0; j < IN_B_COLS; j++) {
			//union {	unsigned int oval; mat_b_t ival; } converter; // For float conversion
			//converter.ival = testMatrix_b[i][j]; // For float
			//tempValue.data = converter.oval; // For float
			tempValue.data = testMatrix_b[i][j]; // For int
			in_stream.write(tempValue);
		}
	}

	// Test run the matrix multiply core
	int28x28(in_stream, out_stream);

	// Read the output matrix c from output stream
	for(i = 0; i < OUT_C_ROWS; i++) {
		for(j = 0; j < OUT_C_COLS; j++) {
			out_stream.read(tempValue);
			//union {	unsigned int ival; mat_c_t oval; } converter; // For float conversion
			//converter.ival = tempValue.data; // For float
			//testResultMatrix_c[i][j] = converter.oval; // For float
			testResultMatrix_c[i][j] = tempValue.data; // For int
		}
	}

	// Check for error
	for (i = 0; i < OUT_C_ROWS; i++) {
		for (j = 0; j < OUT_C_COLS; j++) {
			if (testResultMatrix_c[i][j] != referenceResultMatrix_c[i][j]) {
				error++;
			}
		}
	}

	if (error > 0) {
		printf("Error in calculation! \nError = %d\n", error);
		return 0;
	}
	printf("No error in calculation!\n");

	return 0;
}
