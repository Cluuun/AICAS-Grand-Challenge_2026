# KV260 W8A8 v6 build script.

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize [file join $script_dir ../..]]

set target_mhz "300"
if {[info exists ::env(V6_TARGET_MHZ)] && $::env(V6_TARGET_MHZ) ne ""} {
    set target_mhz $::env(V6_TARGET_MHZ)
}
if {$target_mhz ni {250 300 375}} {
    error "ERROR: V6_TARGET_MHZ must be a KV260 PL0-supported rate: 250, 300, or 375; got: $target_mhz"
}

set project_name "kv260_v6"
if {[info exists ::env(VIVADO_PROJECT_NAME)] && $::env(VIVADO_PROJECT_NAME) ne ""} {
    set project_name $::env(VIVADO_PROJECT_NAME)
}
set project_dir [file normalize [file join $script_dir $project_name]]
set script_bd   [file normalize [file join $repo_root src/ipi/kv260_w8a8_300mhz_v6.bd.tcl]]
set constraints_files [list [file normalize [file join $script_dir timing.xdc]]]
set timing_hook [file normalize [file join $script_dir timing_impl.tcl]]

set target_part "xck26-sfvc784-2LV-c"
set board_part  "xilinx.com:kv260_som:part0:1.4"

set build_step "impl"
if {[info exists ::env(VIVADO_BUILD_STEP)]} {
    set requested_step $::env(VIVADO_BUILD_STEP)
    if {$requested_step in {validate synth impl}} {
        set build_step $requested_step
    } else {
        error "ERROR: VIVADO_BUILD_STEP must be validate, synth, or impl"
    }
}

set num_jobs 4
if {[info exists ::env(VIVADO_JOBS)] && [string is integer -strict $::env(VIVADO_JOBS)] && $::env(VIVADO_JOBS) > 0} {
    set num_jobs $::env(VIVADO_JOBS)
}
set_param general.maxThreads $num_jobs
puts "INFO: Vivado jobs/threads set to $num_jobs"
puts "INFO: V6 build step set to $build_step"
puts "INFO: V6 target frequency set to ${target_mhz} MHz"

proc write_v6_impl_reports {run_dir tag} {
    puts "INFO: Writing v6 implementation reports: $tag"
    catch {report_utilization -file [file join $run_dir "kv260_v6_bd_wrapper_utilization_${tag}.rpt"]}
    catch {report_utilization -hierarchical -file [file join $run_dir "kv260_v6_bd_wrapper_utilization_hier_${tag}.rpt"]}
    catch {report_timing_summary -delay_type max -max_paths 80 -file [file join $run_dir "kv260_v6_bd_wrapper_timing_summary_${tag}.rpt"]}
    catch {report_route_status -file [file join $run_dir "kv260_v6_bd_wrapper_route_status_${tag}.rpt"]}
    catch {report_design_analysis -congestion -file [file join $run_dir "kv260_v6_bd_wrapper_congestion_${tag}.rpt"]}
    catch {report_design_analysis -complexity -file [file join $run_dir "kv260_v6_bd_wrapper_complexity_${tag}.rpt"]}
    catch {report_control_sets -verbose -file [file join $run_dir "kv260_v6_bd_wrapper_control_sets_${tag}.rpt"]}
}

if {![file exists $script_bd]} {
    error "ERROR: Missing BD Tcl script: $script_bd"
}

puts "INFO: Creating Project: ${project_name}"
create_project -force $project_name $project_dir -part $target_part
set_property board_part $board_part [current_project]
config_ip_cache -disable_cache

source [file normalize [file join $script_dir add_v6_ip_path.tcl]]
set_property ip_repo_paths $KV260_IP_REPOS [current_project]
update_ip_catalog

puts "INFO: Sourcing BD Tcl script: $script_bd"
source $script_bd

set bd_design [current_bd_design]
if {$bd_design eq ""} {
    error "ERROR: BD Tcl did not leave an active current_bd_design"
}

set ps_ctrl_width [get_property CONFIG.PSU__MAXIGP0__DATA_WIDTH [get_bd_cells zynq_ps]]
if {$ps_ctrl_width ne "32"} {
    error "ERROR: zynq_ps/M_AXI_HPM0_FPD control data width is $ps_ctrl_width; expected 32"
}
set engine_vlnv [get_property VLNV [get_bd_cells vlm_engine_v6]]
puts "INFO: v6 engine VLNV: $engine_vlnv"

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

if {$build_step eq "validate"} {
    puts "SUCCESS: KV260 W8A8 v6 BD validate/generate complete."
    close_project
    return
}

puts "INFO: Launching Synthesis..."
launch_runs synth_1 -jobs $num_jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    error "ERROR: Synthesis failed. Check synth_1 messages."
}

if {$build_step eq "synth"} {
    puts "SUCCESS: KV260 W8A8 v6 synthesis complete."
    close_project
    return
}

puts "INFO: Launching Implementation..."
set impl_run [get_runs impl_1]
# v6 is route-congestion limited: the 8x128 PE consumes most DSP columns while
# HLS dense/FFN state consumes nearby BRAM/URAM.  Previous failed runs reached
# place_design but stopped with thousands of routing resource conflicts, so bias
# implementation toward routability first; timing is checked after a complete
# route and treated separately by run_bit_gen_v6.sh.
# Vivado 2024.2 routability tuning uses supported run strategy/directive
# properties here, not legacy internal set_param switches.
set_property strategy Performance_ExplorePostRoutePhysOpt $impl_run
catch {set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore $impl_run}
catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE SSI_SpreadLogic_high $impl_run}
catch {set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true $impl_run}
catch {set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $impl_run}
catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AlternateCLBRouting $impl_run}
catch {set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true $impl_run}
catch {set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $impl_run}
if {[file exists $timing_hook]} {
    set_property STEPS.OPT_DESIGN.TCL.PRE $timing_hook $impl_run
    puts "INFO: Implementation pre-hook set: $timing_hook"
} else {
    puts "WARNING: timing_impl.tcl not found, skipping implementation hook."
}

launch_runs impl_1 -to_step write_bitstream -jobs $num_jobs
set impl_wait_failed [catch {wait_on_run impl_1} impl_wait_msg]
if {$impl_wait_failed} {
    puts "WARNING: wait_on_run impl_1 reported failure; keeping project open for report extraction."
    puts "WARNING: wait_on_run message: $impl_wait_msg"
}
set impl_progress [get_property PROGRESS [get_runs impl_1]]
set impl_status [get_property STATUS [get_runs impl_1]]
puts "INFO: impl_1 progress=${impl_progress} status=${impl_status}"
if {$impl_wait_failed || $impl_progress ne "100%"} {
    puts "WARNING: Implementation did not complete to 100%; keeping project artifacts for inspection."
    set run_dir [get_property DIRECTORY [get_runs impl_1]]
    puts "WARNING: impl_1 directory: $run_dir"
    if {![catch {open_run impl_1} open_msg]} {
        write_v6_impl_reports $run_dir partial
    } else {
        puts "WARNING: Could not open partial impl_1 design for reports: $open_msg"
        set routed_error_dcp [file join $run_dir "${bd_name}_wrapper_routed_error.dcp"]
        if {[file exists $routed_error_dcp] && ![catch {open_checkpoint $routed_error_dcp} dcp_msg]} {
            puts "INFO: Opened routed-error checkpoint for partial reports: $routed_error_dcp"
            write_v6_impl_reports $run_dir partial
        } else {
            puts "WARNING: Could not open routed-error checkpoint for reports: $routed_error_dcp"
        }
    }
    close_project
    return
}

set run_dir [get_property DIRECTORY [get_runs impl_1]]
if {![catch {open_run impl_1} open_msg]} {
    write_v6_impl_reports $run_dir final
} else {
    puts "WARNING: Could not open impl_1 design for final reports: $open_msg"
}

set xsa_path [file normalize [file join $project_dir "${project_name}.xsa"]]
puts "INFO: Exporting XSA: $xsa_path"
write_hw_platform -fixed -include_bit -force -file $xsa_path

puts "========================================================"
puts "SUCCESS: KV260 W8A8 v6 ${target_mhz} MHz build complete."
puts "Project      : $project_dir"
puts "BD name      : $bd_name"
puts "Bitstream dir: [get_property DIRECTORY [get_runs impl_1]]"
puts "XSA          : $xsa_path"
puts "========================================================"

close_project
