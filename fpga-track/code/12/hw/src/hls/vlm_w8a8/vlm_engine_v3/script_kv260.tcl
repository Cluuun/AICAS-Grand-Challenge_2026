set script_dir [file normalize [file dirname [info script]]]
cd $script_dir

proc run_export {project_name top_name} {
    open_project -reset $project_name
    set_top $top_name
    add_files accelerator.cpp
    add_files -tb testbench.cpp
    open_solution -reset solution1
    set_part {xck26-sfvc784-2LV-c}
    create_clock -period 3.333 -name default
    set_directive_allocation $top_name run_ntile -limit 1 -type function
    set_directive_allocation $top_name compute_group -limit 1 -type function
    set_directive_allocation $top_name apply_scale_group -limit 1 -type function
    csim_design
    csynth_design
    # cosim_design
    export_design -format ip_catalog
    close_project
}

run_export vlm_engine_v3_kv260 vlm_engine_v3
exit
