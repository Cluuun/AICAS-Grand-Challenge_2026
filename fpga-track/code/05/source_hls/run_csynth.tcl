open_project -reset fpga_gemm_kernel_csynth
set_top fpga_gemm_kernel
add_files gemm_kernel.cpp
add_files gemm_kernel_utils.h
add_files -tb gemm_kernel_tb.cpp
open_solution -reset -flow_target vitis solution_csynth
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.333 -name default
csynth_design
close_project
exit
