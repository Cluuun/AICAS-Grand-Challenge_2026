set script_dir [file normalize [file dirname [info script]]]
set create_tcl [file normalize [file join $script_dir "create_pynq_zu_tpu_pspl_iftest_bd.tcl"]]
set build_dir [file normalize [file join $script_dir "build_tpu_pspl_iftest_bd_direct"]]
set project_name "pynq_zu_tpu_pspl_iftest"
set bd_name "tpu_pspl_iftest"
set target_part "xczu5eg-sfvc784-1-e"

# Keep the direct flow in a single synthesis thread on the shared server.
set_param general.maxThreads 1
set_param synth.maxThreads 1

if {$argc > 1} {
  set build_dir [file normalize [lindex $argv 1]]
}

source $create_tcl

set top_name "${bd_name}_wrapper"
set output_dir [file normalize [file join $build_dir "direct_artifacts"]]
set hwh_src [file normalize [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name "hw_handoff" "${bd_name}.hwh"]]
set bit_out [file normalize [file join $output_dir "${top_name}.bit"]]
set bin_out [file normalize [file join $output_dir "${top_name}.bin"]]
set hwh_out [file normalize [file join $output_dir "${top_name}.hwh"]]

file mkdir $output_dir

update_compile_order -fileset sources_1
synth_design -top $top_name -part $target_part
opt_design
place_design
route_design
write_bitstream -force $bit_out

if {[file exists $hwh_src]} {
  file copy -force $hwh_src $hwh_out
}

write_cfgmem -force -format bin -interface SMAPx32 -loadbit "up 0x0 $bit_out" $bin_out

puts "Direct single-session bitstream build finished for project: $project_name"
puts "Project directory: $build_dir"
puts "Bitstream: $bit_out"
puts "Hardware handoff: $hwh_out"
puts "Bitstream bin: $bin_out"
