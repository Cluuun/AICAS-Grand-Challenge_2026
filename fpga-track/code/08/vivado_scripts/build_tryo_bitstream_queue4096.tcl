# Build the reconstructed Tryo accelerator bitstream and HWH from generated Spinal RTL.
#
# Run after:
#   cd ../Tryo && python3 step2_synthesis.py && python3 step3_spinal_flow.py
#   cd ../SPINAL && sbt "runMain generate_tryo_top" && python3 to_vivado.py
#
# Optional environment variables:
#   TRYO_PART    Vivado part name. Default matches the current reconstructed TCL.
#   TRYO_JOBS    Parallel jobs for synth/impl. Default: 8.

set script_dir [file dirname [file normalize [info script]]]
set package_dir [file normalize [file join $script_dir ".."]]
set project_dir [file normalize [file join $script_dir "build_tryo_bitstream_project_queue4096"]]
set output_dir [file normalize [file join $script_dir "output"]]
set project_name "tryo_bitstream"
set bd_name "design_1"

set part_name "xck26-sfvc784-2LV-c"
if {[info exists ::env(TRYO_PART)] && $::env(TRYO_PART) ne ""} {
  set part_name $::env(TRYO_PART)
}

set jobs 8
if {[info exists ::env(TRYO_JOBS)] && $::env(TRYO_JOBS) ne ""} {
  set jobs $::env(TRYO_JOBS)
}

file mkdir $output_dir

set rtl_file [file join $package_dir "verilog" "TryoTop.v"]
if {![file exists $rtl_file]} {
  error "Missing $rtl_file."
}

source [file join $script_dir "package_tryo_ip.tcl"]
close_project

create_project -force $project_name $project_dir -part $part_name
set_property ip_repo_paths [list [file join $script_dir "ip_repo"]] [current_project]
update_ip_catalog

create_bd_design -quiet $bd_name
source [file join $script_dir "design_1_tryo.tcl"]
create_root_design ""
validate_bd_design
save_bd_design

set bd_file [get_files [file join $project_dir "$project_name.srcs" "sources_1" "bd" $bd_name "$bd_name.bd"]]
set wrapper_file [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_file
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1

set bit_candidates [glob -nocomplain [file join $project_dir "$project_name.runs" "impl_1" "*.bit"]]
if {[llength $bit_candidates] == 0} {
  error "Bitstream was not produced under $project_dir"
}
set bit_file [lindex $bit_candidates 0]

set hwh_candidates [glob -nocomplain [file join $project_dir "$project_name.gen" "sources_1" "bd" $bd_name "hw_handoff" "*.hwh"]]
if {[llength $hwh_candidates] == 0} {
  error "HWH was not produced under $project_dir"
}
set hwh_file [lindex $hwh_candidates 0]

set output_prefix "tryo_queue4096"
if {[info exists ::env(TRYO_OUTPUT_PREFIX)] && $::env(TRYO_OUTPUT_PREFIX) ne ""} {
  set output_prefix $::env(TRYO_OUTPUT_PREFIX)
}

file copy -force $bit_file [file join $output_dir "${output_prefix}.bit"]
file copy -force $hwh_file [file join $output_dir "${output_prefix}.hwh"]

puts "Tryo accelerator bitstream output:"
puts "  [file join $output_dir "${output_prefix}.bit"]"
puts "  [file join $output_dir "${output_prefix}.hwh"]"

