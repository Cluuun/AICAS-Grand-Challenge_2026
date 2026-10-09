set script_dir [file normalize [file dirname [info script]]]
set create_tcl [file normalize [file join $script_dir "create_pynq_zu_tpu_pspl_iftest_bd.tcl"]]
set build_dir [file normalize [file join $script_dir "build_tpu_pspl_iftest_bd"]]
set project_name "pynq_zu_tpu_pspl_iftest"

set_param general.maxThreads 1
set_param synth.maxThreads 1

if {$argc > 1} {
  set build_dir [file normalize [lindex $argv 1]]
}

source $create_tcl

reset_run synth_1
launch_runs synth_1 -jobs 1
wait_on_run synth_1
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1

puts "Sequential bitstream build finished for project: $project_name"
puts "Project directory: $build_dir"
