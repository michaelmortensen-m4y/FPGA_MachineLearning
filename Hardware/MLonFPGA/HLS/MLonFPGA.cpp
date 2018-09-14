
// HLS for the preprocessing part that exstracts the gesture window and outputs the X1, X2, and X3 unfoldings to BRAM
// All 16 sensor values are inputted in parallel so the for loops that iterates over the matrix elements of the tensor are all fully unrolled
// Note that this makes it not able to sample the first matrix that meets the threshold check but only the ones after (As opposed to the sequential version of the code)
// The operation is done when all elements of D tensor matrices have been saved to X1, X2, and X3 in BRAM

#define HIGHTHRESHOLD 5
#define LOWTHRESHOLD 1
#define D 20
#define TENSOR_DIM1 4  // Rows
#define TENSOR_DIM2 4  // Columns
#define TENSOR_DIM3 30000 // Matrices
#define X1_DIM1 4
#define X1_DIM2 80 // 4*D
#define X2_DIM1 4
#define X2_DIM2 80 // 4*D
#define X3_DIM1 20 // D
#define X3_DIM2 16 // 4*4

void infer_core(int in_stream[TENSOR_DIM1][TENSOR_DIM2][TENSOR_DIM3], bool* error, int X1[X1_DIM1][X1_DIM2], int X2[X2_DIM1][X2_DIM2], int X3[X3_DIM1][X3_DIM2]) {
//#pragma HLS STREAM variable=in_stream
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS array_partition variable=in_stream complete dim=1
#pragma HLS array_partition variable=in_stream complete dim=2
#pragma HLS INTERFACE ap_none port=error
//#pragma HLS array_partition variable=X1 complete dim=0
//#pragma HLS INTERFACE axis port=X1
#pragma HLS INTERFACE bram port=X1 // To use BRAM
//#pragma HLS RESOURCE variable=X1 core=RAM_1P_BRAM // Force using only 1 BRAM port
//#pragma HLS array_partition variable=X2 complete dim=0
//#pragma HLS INTERFACE axis port=X2
#pragma HLS INTERFACE bram port=X2
//#pragma HLS RESOURCE variable=X2 core=RAM_1P_BRAM
//#pragma HLS array_partition variable=X3 complete dim=0
//#pragma HLS INTERFACE axis port=X3
#pragma HLS INTERFACE bram port=X3
//#pragma HLS RESOURCE variable=X3 core=RAM_1P_BRAM

	int i, j, k = 0, Dcount = D, storedCount = 0;
	int x1_j = 0, x2_i = 0, x3_i = 0, x3_j = 0;
	bool storeMatrices = false;
	*error = false;

	while (storedCount < D && k < TENSOR_DIM3) {
		if (!storeMatrices) {
			label_1:for (i = 0; i < TENSOR_DIM1; i++) {
#pragma HLS unroll
				label_2:for (j = 0; j < TENSOR_DIM2; j++) {
#pragma HLS unroll
					if (in_stream[i][j][k] < LOWTHRESHOLD || in_stream[i][j][k] > LOWTHRESHOLD) {
						storeMatrices = true;
					}
				}
			}
		}
		if (storeMatrices) {
			if (Dcount == D) {
				// Store the elements in this tensor matrix in their corresponding positions in X1, X2, and X3:
				label_3:for (i = 0; i < TENSOR_DIM1; i++) { // For number of rows in tensor
#pragma HLS unroll
					label_4:for (j = 0; j < TENSOR_DIM2; j++) { // For number of columns in tensor
#pragma HLS unroll
						X1[i][x1_j+storedCount] = in_stream[i][j][k];
						X2[j][x2_i+storedCount] = in_stream[i][j][k];
						X3[x3_i][x3_j] = in_stream[i][j][k];
						x1_j += D;
						x3_j++;
					}
					x1_j = 0;
					x2_i += D;
				}
				x2_i = 0;
				x3_i++;
				x3_j = 0;

				Dcount = 0;
				storedCount++;
			}
			Dcount++;
		}
		k++;
	}

	if (storedCount != D) {
		*error = true;
	}



}
