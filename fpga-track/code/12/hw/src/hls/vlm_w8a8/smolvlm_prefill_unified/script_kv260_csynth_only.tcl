set script_dir [file normalize [file dirname [info script]]]
cd $script_dir

proc run_csynth {project_name top_name} {
    open_project -reset $project_name
    set_top $top_name
    add_files accelerator.cpp
    add_files -tb testbench.cpp
    open_solution -reset solution1
    set_part {xck26-sfvc784-2LV-c}
    create_clock -period 3.333 -name default
    csim_design
    csynth_design
    close_project
}

run_csynth vlm_engine_v2_kv260 vlm_engine_v2
exit
