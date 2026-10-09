# ==============================================================================
# KV260 W8A8 linear-accelerator IP 版本：根据导出的 BD Tcl 重建工程
# 目标设计：
#   - linear_accelerator
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ../..]]

set project_name "kv260_w8a8"
set project_dir  [file normalize [file join $script_dir $project_name]]
set script_bd    [file normalize [file join $repo_root src/ipi/kv260_w8a8_v1.bd.tcl]]
set constraints_files [list     [file normalize [file join $script_dir timing.xdc]] ]

set target_part "xck26-sfvc784-2LV-c"
set board_part  "xilinx.com:kv260_som:part0:1.4"

set num_jobs 16
if {[info exists ::env(VIVADO_JOBS)] && [string is integer -strict $::env(VIVADO_JOBS)] && $::env(VIVADO_JOBS) > 0} {
    set num_jobs $::env(VIVADO_JOBS)
}
set_param general.maxThreads $num_jobs
puts "INFO: Vivado jobs/threads set to $num_jobs"

if {![file exists $script_bd]} {
    error "ERROR: Missing BD Tcl script: $script_bd"
}

puts "INFO: Creating Project: ${project_name}"
create_project -force $project_name $project_dir -part $target_part
set_property board_part $board_part [current_project]

source [file normalize [file join $script_dir add_cosim_ip_path.tcl]]
set_property ip_repo_paths $KV260_IP_REPOS [current_project]
update_ip_catalog

puts "INFO: Sourcing BD Tcl script: $script_bd"
source $script_bd

set bd_design [current_bd_design]
if {$bd_design eq ""} {
    error "ERROR: BD Tcl did not leave an active current_bd_design"
}

set bd_name [get_property NAME $bd_design]
set bd_file [get_files -quiet "${bd_name}.bd"]
if {$bd_file eq ""} {
    error "ERROR: Could not resolve generated BD file for design $bd_name"
}

puts "INFO: Current BD design: $bd_name"
puts "INFO: BD file: $bd_file"

validate_bd_design
save_bd_design
generate_target all $bd_file
export_ip_user_files -of_objects $bd_file -no_script -sync -force -quiet

set wrapper_files [make_wrapper -files $bd_file -top -quiet]
if {$wrapper_files eq ""} {
    set wrapper_files [get_files -quiet -all -regexp {.*_wrapper\.(v|vhd)$}]
}
if {$wrapper_files eq ""} {
    error "ERROR: Could not find generated BD wrapper"
}
foreach wrapper_file $wrapper_files {
    if {[llength [get_files -quiet $wrapper_file]] == 0} {
        add_files -norecurse $wrapper_file
    }
}

update_compile_order -fileset sources_1

set has_constraints 0
foreach xdc $constraints_files {
    if {[file exists $xdc]} {
        puts "INFO: Adding constraints file: $xdc"
        add_files -fileset constrs_1 -norecurse $xdc
        set has_constraints 1
    } else {
        puts "WARNING: constraints file not found: $xdc"
    }
}
if {!$has_constraints} {
    puts "WARNING: No constraints file was added."
}

# puts "INFO: Launching Synthesis..."
# launch_runs synth_1 -jobs $num_jobs
# wait_on_run synth_1

# if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
#     error "ERROR: Synthesis failed. Check synth_1 messages."
# }

# puts "INFO: Launching Implementation..."
# set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

# set timing_impl_tcl [file normalize [file join $script_dir timing_impl.tcl]]
# if {[file exists $timing_impl_tcl]} {
#     set_property STEPS.OPT_DESIGN.TCL.PRE $timing_impl_tcl [get_runs impl_1]
#     puts "INFO: Implementation pre-hook set: $timing_impl_tcl"
# } else {
#     puts "WARNING: timing_impl.tcl not found, skipping extra implementation hook."
# }

# launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
# wait_on_run impl_1

# if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
#     error "ERROR: Implementation failed. Check impl_1 messages."
# }

# set xsa_path [file normalize [file join $project_dir "${project_name}.xsa"]]
# puts "INFO: Exporting XSA: $xsa_path"
# write_hw_platform -fixed -include_bit -force -file $xsa_path

# puts "========================================================"
# puts "SUCCESS: KV260 linear accelerator build complete."
# puts "Project      : $project_dir"
# puts "BD name      : $bd_name"
# puts "Bitstream dir: [get_property DIRECTORY [get_runs impl_1]]"
# puts "XSA          : $xsa_path"
# puts "========================================================"

close_project
