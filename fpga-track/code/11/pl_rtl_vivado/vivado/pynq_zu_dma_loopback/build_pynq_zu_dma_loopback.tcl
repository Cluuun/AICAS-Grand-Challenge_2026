set script_dir   [file normalize [file dirname [info script]]]
set build_dir    [file normalize [file join $script_dir build]]
set export_dir   [file normalize [file join $script_dir export]]
set project_name "pynq_zu_dma_loopback"
set xpr_path     [file join $build_dir ${project_name}.xpr]
set create_tcl   [file join $script_dir create_pynq_zu_dma_loopback.tcl]

if {![file exists $xpr_path]} {
    puts "INFO: project does not exist yet, creating it first via $create_tcl"
    source $create_tcl
}

file mkdir $export_dir

open_project $xpr_path

reset_run synth_1
reset_run impl_1

launch_runs synth_1 -jobs 8
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

open_run impl_1

set bit_path [file join $export_dir ${project_name}.bit]
set xsa_path [file join $export_dir ${project_name}.xsa]

write_bitstream -force $bit_path
write_hw_platform -fixed -include_bit -force -file $xsa_path

puts "BITSTREAM_OUT=$bit_path"
puts "XSA_OUT=$xsa_path"

close_project
