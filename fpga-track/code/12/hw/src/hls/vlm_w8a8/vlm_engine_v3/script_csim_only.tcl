set script_dir [file normalize [file dirname [info script]]]
cd $script_dir

open_project -reset vlm_engine_v3_kv260
set_top vlm_engine_v3
add_files accelerator.cpp
add_files -tb testbench.cpp
open_solution -reset solution1
set_part {xck26-sfvc784-2LV-c}
create_clock -period 3.333 -name default
csim_design
close_project
exit
