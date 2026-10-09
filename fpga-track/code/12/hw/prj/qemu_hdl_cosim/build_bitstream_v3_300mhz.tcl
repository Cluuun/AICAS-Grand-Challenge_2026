# ==============================================================================
# KV260 W8A8 v3 300 MHz build script.
# Rebuilds the v3 BD, applies congestion-oriented implementation directives,
# and emits the post-route reports needed for route-closure triage.
# ==============================================================================

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir ../..]]

set project_name "kv260_w8a8_300mhz_v3"
set project_dir  [file normalize [file join $script_dir $project_name]]
set script_bd    [file normalize [file join $repo_root src/ipi/kv260_w8a8_300mhz_v3.bd.tcl]]
set constraints_files [list [file normalize [file join $script_dir timing.xdc]]]
set congestion_hook [file normalize [file join $script_dir v3_congestion_impl.tcl]]

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
if {![file exists $congestion_hook]} {
    error "ERROR: Missing v3 congestion hook: $congestion_hook"
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

puts "INFO: Launching Synthesis..."
launch_runs synth_1 -jobs $num_jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error "ERROR: Synthesis failed. Check synth_1 messages."
}

puts "INFO: Launching congestion-oriented Implementation..."
set impl_run [get_runs impl_1]
set_property strategy Performance_ExplorePostRoutePhysOpt $impl_run
set_property STEPS.OPT_DESIGN.TCL.PRE $congestion_hook $impl_run
set_property STEPS.WRITE_BITSTREAM.TCL.POST $congestion_hook $impl_run

launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error "ERROR: Implementation failed. Check impl_1 messages."
}

set xsa_path [file normalize [file join $project_dir "${project_name}.xsa"]]
puts "INFO: Exporting XSA: $xsa_path"
write_hw_platform -fixed -include_bit -force -file $xsa_path

puts "========================================================"
puts "SUCCESS: KV260 W8A8 v3 300 MHz build complete."
puts "Project      : $project_dir"
puts "BD name      : $bd_name"
puts "Bitstream dir: [get_property DIRECTORY [get_runs impl_1]]"
puts "Reports      : [file join [get_property DIRECTORY [get_runs impl_1]] v3_route_reports]"
puts "XSA          : $xsa_path"
puts "========================================================"

close_project
