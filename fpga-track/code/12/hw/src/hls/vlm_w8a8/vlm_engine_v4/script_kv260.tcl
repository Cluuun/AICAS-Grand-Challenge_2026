set script_dir [file normalize [file dirname [info script]]]
cd $script_dir

# csim flow: run testbench against the behavioral PE model under V4_CSIM.
# Synthesis flow: skip the testbench so the AXIS streams stay external.
#
# Usage:
#   vitis_hls -f script_kv260.tcl                  ;# csim then csynth + export
#   V4_HLS_MODE=csim  vitis_hls -f script_kv260.tcl ;# csim only
#   V4_HLS_MODE=csynth vitis_hls -f script_kv260.tcl ;# csynth only
#   V4_HLS_MODE=synth  vitis_hls -f script_kv260.tcl ;# csynth + export only
set run_mode "all"
set clock_period "3.333"
if {[info exists ::env(V4_HLS_MODE)]} {
    set env_mode $::env(V4_HLS_MODE)
    if {$env_mode eq "csim" || $env_mode eq "csynth" || $env_mode eq "synth" || $env_mode eq "all"} {
        set run_mode $env_mode
    }
}
if {[info exists ::env(V4_HLS_CLOCK_PERIOD)]} {
    set clock_period $::env(V4_HLS_CLOCK_PERIOD)
}
foreach arg $argv {
    if {$arg eq "csim" || $arg eq "csynth" || $arg eq "synth" || $arg eq "all"} {
        set run_mode $arg
    }
}

proc run_csim {project_name top_name} {
    global clock_period
    open_project -reset $project_name
    set_top $top_name
    add_files accelerator.cpp -cflags "-std=c++14 -DV4_CSIM"
    add_files -tb testbench.cpp -cflags "-std=c++14 -DV4_CSIM"
    open_solution -reset solution_csim
    set_part {xck26-sfvc784-2LV-c}
    create_clock -period $clock_period -name default
    csim_design -clean
    close_project
}

proc run_export {project_name top_name} {
    global clock_period
    open_project -reset $project_name
    set_top $top_name
    add_files accelerator.cpp -cflags "-std=c++14"
    open_solution -reset solution1
    set_part {xck26-sfvc784-2LV-c}
    create_clock -period $clock_period -name default
    csynth_design
    export_design -format ip_catalog -vendor aicas -library v4 -version 1.2
    close_project
}

proc run_csynth {project_name top_name} {
    global clock_period
    open_project -reset $project_name
    set_top $top_name
    add_files accelerator.cpp -cflags "-std=c++14"
    open_solution -reset solution1
    set_part {xck26-sfvc784-2LV-c}
    create_clock -period $clock_period -name default
    csynth_design
    close_project
}

switch -- $run_mode {
    csim {
        run_csim vlm_engine_v4_kv260_csim vlm_engine_v4
    }
    csynth {
        run_csynth vlm_engine_v4_kv260 vlm_engine_v4
    }
    synth {
        run_export vlm_engine_v4_kv260 vlm_engine_v4
    }
    default {
        run_csim   vlm_engine_v4_kv260_csim vlm_engine_v4
        run_export vlm_engine_v4_kv260      vlm_engine_v4
    }
}
exit
