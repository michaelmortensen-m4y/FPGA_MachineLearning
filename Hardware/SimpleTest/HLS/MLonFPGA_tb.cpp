//------------------------------------------//
//			   HLS Bench					//
//------------------------------------------//

#include <stdio.h>

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

void infer_core(int in_stream[TENSOR_DIM1][TENSOR_DIM2][TENSOR_DIM3], bool* error, int X1[X1_DIM1][X1_DIM2], int X2[X2_DIM1][X2_DIM2], int X3[X3_DIM1][X3_DIM2]);


int main() {
	printf("Testing started\n");

	int i, j, k;
	bool error = false;
	int test_tensor[TENSOR_DIM1][TENSOR_DIM2][TENSOR_DIM3];
	int X1[X1_DIM1][X1_DIM2];
	int X2[X2_DIM1][X2_DIM2];
	int X3[X3_DIM1][X3_DIM2];

	// For the first 5 matrices in the tensor the core should return the sum to bram of the 16 elements
	int testValue = 0;
	for(i = 0; i < TENSOR_DIM3; i++) {
		for(j = 0; j < TENSOR_DIM1; j++) {
			for(k = 0; k < TENSOR_DIM2; k++) {
				test_tensor[j][k][i] = testValue;
				testValue++;
			}
		}
	}


	if (error) {
		printf("error = true\n");
	} else {
		printf("error = false\n");
	}

	printf("X1:\n");
	for (i = 0; i < X1_DIM1; i++) {
		for (j = 0; j < X1_DIM2; j++) {
			printf("%d ", X1[i][j]);
		}
		printf("\n");
	}
	printf("X2:\n");
	for (i = 0; i < X2_DIM1; i++) {
		for (j = 0; j < X2_DIM2; j++) {
			printf("%d ", X2[i][j]);
		}
		printf("\n");
	}
	printf("X3:\n");
	for (i = 0; i < X3_DIM1; i++) {
		for (j = 0; j < X3_DIM2; j++) {
			printf("%d ", X3[i][j]);
		}
		printf("\n");
	}

	infer_core(test_tensor, &error, X1, X2, X3);

	if (error) {
		printf("error = true\n");
	} else {
		printf("error = false\n");
	}

	printf("X1:\n");
	for (i = 0; i < X1_DIM1; i++) {
		for (j = 0; j < X1_DIM2; j++) {
			printf("%d ", X1[i][j]);
		}
		printf("\n");
	}
	printf("X2:\n");
	for (i = 0; i < X2_DIM1; i++) {
		for (j = 0; j < X2_DIM2; j++) {
			printf("%d ", X2[i][j]);
		}
		printf("\n");
	}
	printf("X3:\n");
	for (i = 0; i < X3_DIM1; i++) {
		for (j = 0; j < X3_DIM2; j++) {
			printf("%d ", X3[i][j]);
		}
		printf("\n");
	}

	return 0;
}
