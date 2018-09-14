//------------------------------------------//
//			   HLS BRAM Bench				//
//------------------------------------------//

#include <stdio.h>
#include <math.h>
#include "bramtest_tb.h"

int main() {
	printf("Testing started\n");
	// Declare test matrices and other stuff
	mat_a_t testMatrix_a[IN_A_ROWS][IN_A_COLS];
	mat_b_t testMatrix_b[IN_B_ROWS][IN_B_COLS];
	mat_c_t testResultMatrix_c[OUT_C_ROWS][OUT_C_COLS];
	mat_c_t referenceResultMatrix_c[OUT_C_ROWS][OUT_C_COLS];
	int i, j, k;

	// Generate test matrices and reference matrix
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
	bramtest(testMatrix_a, testMatrix_b, testResultMatrix_c);
	for (i = 0; i < IN_B_COLS; i++) {
		for (j = 0; j < IN_B_COLS; j++) {
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
