
// HLS for SVD

#define MATRIX_DIM1 2
#define MATRIX_DIM2 3

#define S_DIM1 2
#define S_DIM2 2

#define V_DIM1 3
#define V_DIM2 3


void svd_test(int X[MATRIX_DIM1][MATRIX_DIM2], int S[S_DIM1][S_DIM2], int V[V_DIM1][V_DIM2]) {
//#pragma HLS STREAM variable=in_stream
#pragma HLS INTERFACE ap_none port=X
#pragma HLS array_partition variable=X complete
#pragma HLS INTERFACE bram port=S // To use BRAM
#pragma HLS INTERFACE bram port=V

	int m = MATRIX_DIM1;
	int n = MATRIX_DIM2;


}
