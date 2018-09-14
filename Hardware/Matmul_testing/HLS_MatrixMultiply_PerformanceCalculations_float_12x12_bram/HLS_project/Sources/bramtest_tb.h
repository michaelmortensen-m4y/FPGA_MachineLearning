//------------------------------------------//
//			   HLS BRAM Bench				//
//------------------------------------------//

#define IN_A_ROWS 12
#define IN_A_COLS 12
#define IN_B_ROWS 12
#define IN_B_COLS 12
#define OUT_C_ROWS 12
#define OUT_C_COLS 12

typedef float mat_a_t; // a in matrix
typedef float mat_b_t; // b in matrix
typedef float mat_c_t; // Result matrix

// Function prototype
void bramtest(mat_a_t a[IN_A_ROWS][IN_A_COLS], mat_b_t b[IN_B_ROWS][IN_B_COLS], mat_c_t c[OUT_C_ROWS][OUT_C_COLS]);



