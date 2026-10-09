# Vivado 2023.2 block-design and bitstream flow for the KV260-targeted top.
# Prerequisite: run vivado/scripts/package_accelerator_ip.tcl after refreshing
# SPINAL/vivado with `cd SPINAL && python to_vivado.py`.
#
# Run from repo root:
# /data/home/songqiangxu/Xilinx/2023.2/Vivado/2023.2/bin/vivado \
#   -mode batch \
#   -source vivado/scripts/build_kv260_bitstream.tcl \
#   -journal vivado/logs/build_kv260_bitstream.jou \
#   -log vivado/logs/build_kv260_bitstream.log

set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir "../.."]]
set work_dir [file join $repo_root "vivado/work/kv260_bd"]
set ip_repo_dir [file join $repo_root "vivado/ip_repo"]
set report_dir [file join $repo_root "vivado/reports"]
set bit_dir [file join $repo_root "vivado/bitstream"]

set pl_clk_period_ns 4.687
if {[info exists ::env(PL_CLK_PERIOD_NS)] && $::env(PL_CLK_PERIOD_NS) ne ""} {
  set pl_clk_period_ns $::env(PL_CLK_PERIOD_NS)
}

if {[info exists ::env(PL_CLK_FREQ_MHZ)] && $::env(PL_CLK_FREQ_MHZ) ne ""} {
  set pl_clk_freq_mhz $::env(PL_CLK_FREQ_MHZ)
  set pl_clk_period_ns [expr {1000.0 / double($pl_clk_freq_mhz)}]
} else {
  set pl_clk_freq_mhz [expr {1000.0 / double($pl_clk_period_ns)}]
}
set pl_clk_period_fmt [format "%.3f" $pl_clk_period_ns]
set pl_clk_freq_fmt [format "%.3f" $pl_clk_freq_mhz]
set pl_clk_fall_fmt [format "%.3f" [expr {$pl_clk_period_ns / 2.0}]]
set ::env(VLM_REPO_ROOT) $repo_root

puts "VLM_PL_CLK_PERIOD_NS=$pl_clk_period_fmt"
puts "VLM_PL_CLK_FREQ_MHZ=$pl_clk_freq_fmt"

proc patch_ps_pl_clk_xdc {ps_xdc pl_clk_period_fmt pl_clk_fall_fmt tag} {
  set ps_xdc_new [format {create_clock -name clk_pl_0 -period "%s" -waveform {0.000 %s} [get_pins "PS8_i/PLCLK[0]"]} $pl_clk_period_fmt $pl_clk_fall_fmt]
  if {![file exists $ps_xdc]} {
    puts "WARNING: PS XDC not found for PL clock patch ($tag): $ps_xdc"
    return 0
  }

  set fp [open $ps_xdc "r"]
  set ps_xdc_text [read $fp]
  close $fp

  set patched 0
  set out_lines [list]
  foreach line [split $ps_xdc_text "\n"] {
    if {[string first {create_clock -name clk_pl_0 -period } $line] >= 0 &&
        [string first {PS8_i/PLCLK[0]} $line] >= 0} {
      lappend out_lines $ps_xdc_new
      incr patched
    } else {
      lappend out_lines $line
    }
  }

  if {$patched > 0} {
    set fp [open $ps_xdc "w"]
    puts -nonewline $fp [join $out_lines "\n"]
    close $fp
    puts "VLM_PL_CLK_PATCHED_PS_XDC=$ps_xdc"
    puts "VLM_PL_CLK_PATCHED_TAG=$tag"
    puts "VLM_PL_CLK_PATCHED_COUNT=$patched"
    puts "VLM_PL_CLK_PATCHED_LINE=$ps_xdc_new"
    return $patched
  }

  puts "WARNING: did not find clk_pl_0 create_clock in $ps_xdc during $tag"
  return 0
}

file delete -force $work_dir
file mkdir $work_dir
file mkdir $report_dir
file mkdir $bit_dir

create_project vlm_kv260_bd $work_dir -part xck26-sfvc784-2LV-c -force
set kv260_board_part [get_board_parts -quiet xilinx.com:kv260_som:part0:1.4]
if {$kv260_board_part ne ""} {
  set_property board_part $kv260_board_part [current_project]
}
set_property target_language Verilog [current_project]
set_property ip_repo_paths [list $ip_repo_dir] [current_project]
update_ip_catalog

create_bd_design "vlm_system"

# PS 负责 AXI-Lite 配置寄存器访问，并提供 PL 时钟/复位和 DDR 访问入口。
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps
set_property -dict [list \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $pl_clk_freq_fmt \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $pl_clk_freq_fmt \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__MAXIGP0__DATA_WIDTH {64} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP0 {1} \
  CONFIG.PSU__USE__S_AXI_GP1 {1} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} \
  CONFIG.PSU__USE__S_AXI_GP4 {1} \
  CONFIG.PSU__USE__S_AXI_GP5 {1} \
  CONFIG.PSU__USE__S_AXI_GP6 {1} \
  CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP1__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP6__DATA_WIDTH {128} \
] [get_bd_cells ps]

create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_pl0
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_ctrl_smc
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_param_smc
create_bd_cell -type ip -vlnv user.org:user:smolvlm2_accelerator:* accel

# AXI-Lite 只有一个 PS master 和一个加速器 slave。DDR path 避免 7-to-1 SmartConnect：
# 6 个 128-bit master 直连 PS slave 端口，param_gmem
# 用一个 1:1 SmartConnect 做 256->128 宽度转换。
# vit_xm 使用低功耗域 S_AXI_LPD，避免为 vit_xm/kv 恢复 2-to-1 SmartConnect。
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_ctrl_smc]
set_property -dict [list \
  CONFIG.NUM_SI {1} \
  CONFIG.NUM_MI {1} \
  CONFIG.ADVANCED_PROPERTIES {__experimental_features__ {reduced_axi4_area_mode 1}} \
] [get_bd_cells axi_param_smc]

connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins rst_pl0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst_pl0/ext_reset_in]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins accel/clk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/saxihpc0_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/saxihpc1_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/saxihp0_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/saxihp1_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/saxihp2_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/saxihp3_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins ps/saxi_lpd_aclk]
connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] [get_bd_pins accel/resetn]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins axi_ctrl_smc/aclk]
connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] [get_bd_pins axi_ctrl_smc/aresetn]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins axi_param_smc/aclk]
connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] [get_bd_pins axi_param_smc/aresetn]

connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_ctrl_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ctrl_smc/M00_AXI] [get_bd_intf_pins accel/axilite]

connect_bd_intf_net [get_bd_intf_pins accel/param_gmem] [get_bd_intf_pins axi_param_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_param_smc/M00_AXI] [get_bd_intf_pins ps/S_AXI_HPC0_FPD]
connect_bd_intf_net [get_bd_intf_pins accel/weight_gmem1] [get_bd_intf_pins ps/S_AXI_HPC1_FPD]
connect_bd_intf_net [get_bd_intf_pins accel/weight_gmem2] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins accel/state_gmem] [get_bd_intf_pins ps/S_AXI_HP1_FPD]
connect_bd_intf_net [get_bd_intf_pins accel/vit_a_gmem] [get_bd_intf_pins ps/S_AXI_HP2_FPD]
connect_bd_intf_net [get_bd_intf_pins accel/kv_gmem] [get_bd_intf_pins ps/S_AXI_HP3_FPD]
connect_bd_intf_net [get_bd_intf_pins accel/vit_xm_gmem] [get_bd_intf_pins ps/S_AXI_LPD]

assign_bd_address
validate_bd_design
save_bd_design

make_wrapper -files [get_files [file join $work_dir "vlm_kv260_bd.srcs/sources_1/bd/vlm_system/vlm_system.bd"]] -top
add_files -norecurse [file join $work_dir "vlm_kv260_bd.gen/sources_1/bd/vlm_system/hdl/vlm_system_wrapper.v"]
set_property top vlm_system_wrapper [current_fileset]
update_compile_order -fileset sources_1

set ps_xdc [file join $work_dir "vlm_kv260_bd.gen/sources_1/bd/vlm_system/ip/vlm_system_ps_0/vlm_system_ps_0.xdc"]
patch_ps_pl_clk_xdc $ps_xdc $pl_clk_period_fmt $pl_clk_fall_fmt "after_wrapper"

set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property strategy Performance_ExploreWithRemap [get_runs impl_1]
# 当前瓶颈是 route congestion，而不是 place 前 UTLZ。放置阶段主动摊开逻辑，
# 路由阶段使用 UltraScale+ 的 alternate CLB routing，优先拿到合法 route。
# 2026-05-23：SOFTMAX mask 修复后需要重跑实现；顺手把 pre/post-route
# phys_opt 从 Explore 提到 AggressiveExplore，尝试多做复制/重定时/高扇出优化。
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE AltSpreadLogic_high [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AlternateCLBRouting [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

if {[info exists ::env(FMAX02_FORCE_REPLICATION)] && $::env(FMAX02_FORCE_REPLICATION) eq "1"} {
  set fmax02_synth_hook [file join $repo_root "vivado/scripts/fmax02_post_synth.tcl"]
  set fmax02_phys_hook  [file join $repo_root "vivado/scripts/fmax02_post_phys_opt.tcl"]
  set_property STEPS.SYNTH_DESIGN.TCL.POST $fmax02_synth_hook [get_runs synth_1]
  set_property STEPS.PHYS_OPT_DESIGN.TCL.POST $fmax02_phys_hook [get_runs impl_1]
  puts "VLM_FMAX02_FORCE_REPLICATION=1"
  puts "VLM_FMAX02_SYNTH_HOOK=$fmax02_synth_hook"
  puts "VLM_FMAX02_PHYS_HOOK=$fmax02_phys_hook"
}

launch_runs synth_1 -jobs 8
patch_ps_pl_clk_xdc $ps_xdc $pl_clk_period_fmt $pl_clk_fall_fmt "before_synth_wait"
wait_on_run synth_1
open_run synth_1 -name synth_1
report_utilization -file [file join $report_dir "kv260_synth_util.rpt"]
report_timing_summary -file [file join $report_dir "kv260_synth_timing.rpt"]
close_design

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
open_run impl_1
set impl_clk [get_clocks -quiet clk_pl_0]
if {[llength $impl_clk] > 0} {
  puts "VLM_PL_CLK_IMPL_PERIOD_NS=[get_property PERIOD $impl_clk]"
  puts "VLM_PL_CLK_IMPL_WAVEFORM=[get_property WAVEFORM $impl_clk]"
}
report_utilization -file [file join $report_dir "kv260_impl_util.rpt"]
report_utilization -hierarchical -file [file join $report_dir "kv260_impl_util_hier.rpt"]
report_clocks -file [file join $report_dir "kv260_clocks.rpt"]
report_timing_summary -file [file join $report_dir "kv260_impl_timing.rpt"]
report_timing_summary -setup -max_paths 50 -unique_pins -file [file join $report_dir "qk01_timing_summary_unique.rpt"]
report_timing -max_paths 200 -slack_lesser_than 0.5 -file [file join $report_dir "qk01_top200_paths.rpt"]
report_design_analysis -congestion -timing -setup -file [file join $report_dir "qk01_design_analysis.rpt"]
report_route_status -file [file join $report_dir "kv260_route_status.rpt"]
report_clock_utilization -file [file join $report_dir "kv260_clock_util.rpt"]

set bit_src [file join $work_dir "vlm_kv260_bd.runs/impl_1/vlm_system_wrapper.bit"]
if {[file exists $bit_src]} {
  file copy -force $bit_src [file join $bit_dir "smolvlm2_kv260.bit"]
}
write_hw_platform -fixed -include_bit -force -file [file join $bit_dir "smolvlm2_kv260.xsa"]
puts "VLM_VIVADO_BITSTREAM_DIR=$bit_dir"
puts "VLM_VIVADO_REPORT_DIR=$report_dir"
puts "VLM_PL_CLK_PERIOD_NS=$pl_clk_period_fmt"
puts "VLM_PL_CLK_FREQ_MHZ=$pl_clk_freq_fmt"
puts "VLM_PL_CLK_PS_XDC=$ps_xdc"
