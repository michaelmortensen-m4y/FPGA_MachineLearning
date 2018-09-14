
#include <stdio.h>
#include <ff.h>
#include <math.h>
#include <stdlib.h>
#include <xparameters.h>
#include <xLMonFPGA.h>

// SD card stuff:
FATFS FatFs;
FIL file;
FRESULT result;
TCHAR *Path = "0:/";

int main()
{
    printf("\n***** SD Card Read Test. *****\n\n");

    result = f_mount(&FatFs, Path, 0);
    if (result != 0) {
    	printf("ERROR in f_mount: %d.\n", result);
    } else {
    	printf("f_mount success.\n");
    }

    result = f_open(&file, "testfile.txt", FA_READ);
    if (result != 0) {
		printf("ERROR in f_open: %d.\n", result);
	} else {
		printf("f_open success.\n");
	}

    char buff[10];
    unsigned int count;
    result = f_read(&file, buff, 7, &count);
    if (result != 0) {
		printf("ERROR in f_read: %d.\n", result);
	} else {
		printf("f_read success: ");
		printf(buff);
	}

    /* Writing to DDR3 memory address
    int* array = (int*)MY_BASE_ADDRESS;
    while ( 1 ) {
       u32 val = < read from stdin >;
       array[pos] = val;
       pos += 4;
    }
    */

    // Start IP core
	XMLonFPGA_Start(&doMLonFPGA);

	// Wait until output is ready
	while (!XMLonFPGA_IsDone(&doMLonFPGA)) {}

	// Put the result in a variable
	res = (int) XMLonFPGA_Get_return(&doMLonFPGA);

	// Print the result via UART
	printf("Hardware response: ");
	printf(res);

    printf("\n***** ***** *****\n");
    return 0;
}
