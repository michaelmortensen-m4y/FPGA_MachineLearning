//------------------------------------------//
//				  HLS BRAM					//
//------------------------------------------//

#include "bramtest.h"

void bramtest(mat_a_t a[IN_A_ROWS][IN_A_COLS], mat_b_t b[IN_B_ROWS][IN_B_COLS], mat_c_t c[OUT_C_ROWS][OUT_C_COLS]) {
#pragma HLS INTERFACE s_axilite port=return bundle=CRTLS_BUS
#pragma HLS INTERFACE bram port=a // To use BRAM
#pragma HLS RESOURCE variable=a core=RAM_1P_BRAM // Force using only 1 BRAM port
#pragma HLS INTERFACE bram port=b
#pragma HLS RESOURCE variable=b core=RAM_1P_BRAM
#pragma HLS INTERFACE bram port=c
#pragma HLS RESOURCE variable=c core=RAM_1P_BRAM
	bramtest_label0:for (int i = 0; i < IN_A_ROWS; i++) {
		bramtest_label1:for (int j = 0; j < IN_B_COLS; j++) {
#pragma HLS PIPELINE II=12
			c[i][j] = 0;
			bramtest_label2:for (int k = 0; k < IN_B_ROWS; k++) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}
}
