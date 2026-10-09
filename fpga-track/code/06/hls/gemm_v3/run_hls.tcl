# Vitis HLS project script for gemm_hf/gemm_kernel.
#
# Usage:
#   HLS_ACTION=<csim|csynth|package|all> vitis-run --mode hls --tcl run_hls.tcl

set project_root [file normalize [file dirname [info script]]]
if {[info exists ::env(HLS_PROJECT_DIR)]} {
    set project_name [file normalize $::env(HLS_PROJECT_DIR)]
} else {
    set project_name [file join $project_root hls_proj]
}
set solution_name solution1

if {[info exists ::env(TOP_NAME)]} {
    set top_name [string trim $::env(TOP_NAME)]
} else {
    set top_name gemm_kernel
}

if {[info exists ::env(HLS_ACTION)]} {
    set action [string trim $::env(HLS_ACTION)]
} else {
    set action package
}

if {[info exists ::env(IP_REPO_DIR)]} {
    set ip_repo_root [file normalize $::env(IP_REPO_DIR)]
} else {
    set ip_repo_root [file join $project_root build ip_repo]
}

if {[info exists ::env(PACKAGE_DIR)]} {
    set package_dir [file normalize $::env(PACKAGE_DIR)]
} else {
    set package_dir [file join $project_root build package]
}

if {[info exists ::env(HLS_FILE)]} {
    set hls_file [file normalize $::env(HLS_FILE)]
} else {
    set hls_file [file join $project_root gemm_kernel.cpp]
}

if {[info exists ::env(HLS_HEADER)]} {
    set hls_header [file normalize $::env(HLS_HEADER)]
} else {
    set hls_header [file join $project_root gemm_kernel.h]
}

if {[info exists ::env(TB_FILE)]} {
    set tb_file [file normalize $::env(TB_FILE)]
} else {
    set tb_file [file join $project_root gemm_tb.cpp]
}

if {[info exists ::env(PART)]} {
    set part_name [string trim $::env(PART)]
} else {
    set part_name xck26-sfvc784-2LV-c
}

if {[info exists ::env(HLS_STD)]} {
    set hls_std [string trim $::env(HLS_STD)]
} else {
    set hls_std c++17
}

if {[info exists ::env(HLS_CLOCK_PERIOD_NS)]} {
    set hls_clock_period_ns [string trim $::env(HLS_CLOCK_PERIOD_NS)]
} else {
    set hls_clock_period_ns 2.5
}

set supported_actions {csim csynth cosim package all}
if {[lsearch -exact $supported_actions $action] < 0} {
    puts "Usage: HLS_ACTION=<csim|csynth|package|all> vitis-run --mode hls --tcl run_hls.tcl"
    exit 1
}

if {![file exists $hls_file]} {
    error "HLS source file not found: $hls_file"
}
if {![file exists $hls_header]} {
    error "HLS header file not found: $hls_header"
}

proc setup_project {project_name solution_name top_name hls_file hls_header tb_file part_name hls_std hls_clock_period_ns project_root} {
    open_project -reset $project_name
    set_top $top_name

    add_files $hls_file -cflags "-I$project_root -std=$hls_std"
    add_files $hls_header -cflags "-I$project_root -std=$hls_std"

    if {[file exists $tb_file]} {
        add_files -tb $tb_file -cflags "-I$project_root -std=$hls_std -Wno-unknown-pragmas"
    }

    open_solution -reset $solution_name -flow_target vivado
    set_part $part_name
    create_clock -period $hls_clock_period_ns -name default
}

proc package_ip {project_name solution_name exported_ip_repo package_dir} {
    export_design -format ip_catalog -rtl verilog

    set generated_ip_dir [file join $project_name $solution_name impl ip]
    if {![file exists [file join $generated_ip_dir component.xml]]} {
        error "Packaged IP not found under $generated_ip_dir"
    }

    file delete -force $exported_ip_repo
    file mkdir [file dirname $exported_ip_repo]
    file copy -force $generated_ip_dir $exported_ip_repo

    file delete -force $package_dir
    file mkdir $package_dir
    foreach zip_file [glob -nocomplain [file join $generated_ip_dir *.zip]] {
        file copy -force $zip_file $package_dir
    }

    puts "Packaged IP repo copied to: $exported_ip_repo"
    puts "Packaged IP zip copied to: $package_dir"
}

cd $project_root
setup_project $project_name $solution_name $top_name $hls_file $hls_header $tb_file $part_name $hls_std $hls_clock_period_ns $project_root

set has_tb [expr {$tb_file ne "" && [file exists $tb_file]}]

switch -- $action {
    csim {
        if {$has_tb} {
            csim_design
        } else {
            puts "Skipping csim because no testbench file was provided"
        }
    }
    csynth {
        csynth_design
    }
    cosim {
        csynth_design
        if {$has_tb} {
            cosim_design -rtl verilog
        } else {
            puts "Skipping cosim because no testbench file was provided"
        }
    }
    package {
        csynth_design
        package_ip $project_name $solution_name $ip_repo_root $package_dir
    }
    all {
        if {$has_tb} {
            csim_design
        } else {
            puts "Skipping csim because no testbench file was provided"
        }
        csynth_design
        package_ip $project_name $solution_name $ip_repo_root $package_dir
    }
}

exit
