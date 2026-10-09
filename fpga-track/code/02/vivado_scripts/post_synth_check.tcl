# Vivado 2023.2 post-synth resource/timing check for the Spinal top.
# Run from repo root:
# /data/home/songqiangxu/Xilinx/2023.2/Vivado/2023.2/bin/vivado \
#   -mode batch \
#   -source vivado/scripts/post_synth_check.tcl \
#   -journal vivado/logs/post_synth_check.jou \
#   -log vivado/logs/post_synth_check.log

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir "../.."]]
set rtl_dir [file join $repo_root "SPINAL/vivado"]
set work_dir [file join $repo_root "vivado/work/post_synth_check"]
set report_dir [file join $repo_root "vivado/reports"]

file delete -force $work_dir
file mkdir $work_dir
file mkdir $report_dir

create_project vlm_spinal_synth $work_dir -part xck26-sfvc784-2LV-c -force
set kv260_board_part [get_board_parts -quiet xilinx.com:kv260_som:part0:1.4]
if {$kv260_board_part ne ""} {
  set_property board_part $kv260_board_part [current_project]
}
set_property target_language Verilog [current_project]

# The replaced RTL reads ROM .dat files with ./file.dat paths, so synthesize
# from SPINAL/vivado where to_vivado.py places those .dat files.
cd $rtl_dir
read_verilog [file join $rtl_dir "ACCELERATOR_replaced.v"]
read_verilog [file join $rtl_dir "ACCELERATOR_bb_replaced.v"]

synth_design -top ACCELERATOR -part xck26-sfvc784-2LV-c

report_utilization -hierarchical -file [file join $report_dir "post_synth_util_hier.rpt"]
report_utilization -file [file join $report_dir "post_synth_util.rpt"]

report_timing_summary -file [file join $report_dir "post_synth_timing_unconstrained.rpt"]
create_clock -period 2.5 -name clk [get_ports clk]
report_timing_summary -file [file join $report_dir "post_synth_timing_2p5ns.rpt"]

write_checkpoint -force [file join $work_dir "post_synth.dcp"]
puts "VLM_VIVADO_REPORT_DIR=$report_dir"
