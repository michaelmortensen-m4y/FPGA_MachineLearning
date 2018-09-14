#*****************************************************************************
# Create a Zynq coprocessor for Matrix Multiplication
# by G.Sutter for teaching purposes
# Revision History:
# March 2014, for Vivado HLS 2013.3 (ZedBoard board)
# June 2014, for Vivado HLS 2014.1 (ZedBoard board)
#
#*****************************************************************************


# Create the design project
open_project matrixmul_zynq_fifo_prj

# Define the top level function for hardware synthesis
set_top matrixmul_fifo_accel_core

# Select the files for hardware synthesis
# Select the files required for the testbench
add_files      src/matrixmul_zynq_fifo.cpp 
add_files -tb  src/matrixmul_zynq_fifo_test.cpp 

# ###########################################################
open_solution solution1 -reset

# Select the FPGA 
# For atlys xc6slx45csg324-2 and clk 100 MHz
# set_part xc6slx45csg324-2 
# create_clock -period "100MHz" # (Atlys)
# For ZedBoard xc7z020clg484-1 and clk 100 MHz
set_part xc7z020clg484-1  
create_clock -period "200MHz"

exit
