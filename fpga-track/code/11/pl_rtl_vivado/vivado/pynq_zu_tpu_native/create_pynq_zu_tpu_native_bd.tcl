set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ".." ".."]]
set repo_root ""
set build_dir  [file normalize [file join $script_dir "build_tpu_native_bd"]]
set project_name "pynq_zu_tpu_native"
set bd_name "pynq_zu_tpu_native_bd"
set target_part "xczu5eg-sfvc784-1-e"
set matrix_d_base "0x0000C000"

foreach candidate [list \
  [file join $project_root "third_party" "tpu_vanilla-fp-32"] \
] {
  set candidate_norm [file normalize $candidate]
  if {[file isdirectory $candidate_norm]} {
    set repo_root $candidate_norm
    break
  }
}

proc scan_verilog_files {dir file_list_var} {
  upvar $file_list_var file_list

  # Skip the standalone VC707-oriented wrapper tree; this project uses the Zynq BD wrapper as top.
  if {[file tail [file normalize $dir]] eq "system_pro"} {
    return
  }

  foreach file [lsort [glob -nocomplain -directory $dir "*.v"]] {
    if {[file tail $file] eq "tpu_top_matmul.v"} {
      continue
    }
    lappend file_list [file normalize $file]
  }

  foreach subdir [lsort [glob -nocomplain -directory $dir "*"]] {
    if {[file isdirectory $subdir]} {
      scan_verilog_files $subdir file_list
    }
  }
}

proc require_latest_ip_vlnv {pattern pretty_name} {
  set ipdefs [lsort [get_ipdefs -all $pattern]]
  if {[llength $ipdefs] == 0} {
    error "Required IP '$pretty_name' not found for pattern '$pattern'. Check that your Vivado installation includes the Zynq UltraScale+ MPSoC/device support."
  }
  return [lindex $ipdefs end]
}

if {$argc > 0} {
  set repo_root [file normalize [lindex $argv 0]]
}
if {$argc > 1} {
  set build_dir [file normalize [lindex $argv 1]]
}
if {$argc > 2} {
  set matrix_d_base [lindex $argv 2]
}

file mkdir $build_dir
create_project $project_name $build_dir -force -part $target_part
set_property target_language Verilog [current_project]

if {$repo_root eq ""} {
  error "TPU RTL repo root not found. Pass it explicitly as the first tclarg."
}

set rtl_dir [file normalize [file join $repo_root "rtl"]]
if {![file isdirectory $rtl_dir]} {
  error "RTL directory not found: $rtl_dir"
}

set rtl_files [list]
scan_verilog_files $rtl_dir rtl_files
if {[llength $rtl_files] == 0} {
  error "No Verilog files found under: $rtl_dir"
}
add_files $rtl_files
update_compile_order -fileset sources_1

create_bd_design $bd_name

set zynq_ps_vlnv [require_latest_ip_vlnv "xilinx.com:ip:zynq_ultra_ps_e:*" "zynq_ultra_ps_e"]
set smartconnect_vlnv [require_latest_ip_vlnv "xilinx.com:ip:smartconnect:*" "smartconnect"]
set proc_sys_reset_vlnv [require_latest_ip_vlnv "xilinx.com:ip:proc_sys_reset:*" "proc_sys_reset"]

puts "Using IP VLNVs:"
puts "  zynq_ultra_ps_e = $zynq_ps_vlnv"
puts "  smartconnect    = $smartconnect_vlnv"
puts "  proc_sys_reset  = $proc_sys_reset_vlnv"

create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ultra_ps_e_0
create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_ctrl_0
create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_data_0
create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_out_0
create_bd_cell -type ip -vlnv $proc_sys_reset_vlnv proc_sys_reset_0
create_bd_cell -type module -reference tpu_top tpu_top_0
set_property -dict [list CONFIG.MATRIX_D_BASE_ADDR $matrix_d_base] [get_bd_cells tpu_top_0]

apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "0"} [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

set_property CONFIG.NUM_MI {1} [get_bd_cells smartconnect_ctrl_0]
set_property CONFIG.NUM_SI {1} [get_bd_cells smartconnect_ctrl_0]
set_property CONFIG.NUM_MI {1} [get_bd_cells smartconnect_data_0]
set_property CONFIG.NUM_SI {1} [get_bd_cells smartconnect_data_0]
set_property CONFIG.NUM_MI {1} [get_bd_cells smartconnect_out_0]
set_property CONFIG.NUM_SI {1} [get_bd_cells smartconnect_out_0]

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_ctrl_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_data_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_out_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_lpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins tpu_top_0/clk]

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins proc_sys_reset_0/ext_reset_in]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_ctrl_0/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_data_0/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_out_0/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins tpu_top_0/rst_n]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins smartconnect_ctrl_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_ctrl_0/M00_AXI] [get_bd_intf_pins tpu_top_0/s_axil]
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smartconnect_data_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_data_0/M00_AXI] [get_bd_intf_pins tpu_top_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins tpu_top_0/m_axi] [get_bd_intf_pins smartconnect_out_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_out_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

assign_bd_address
assign_bd_address -force \
  -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
  -offset 0xA0000000 \
  -range 0x00010000 \
  [get_bd_addr_segs tpu_top_0/s_axi/reg0]
assign_bd_address -force \
  -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
  -offset 0x80000000 \
  -range 0x00001000 \
  [get_bd_addr_segs tpu_top_0/s_axil/reg0]
validate_bd_design
save_bd_design

make_wrapper -files [get_files [file join $build_dir $project_name.srcs sources_1 bd $bd_name $bd_name.bd]] -top
set wrapper_file [file join $build_dir $project_name.gen sources_1 bd $bd_name hdl ${bd_name}_wrapper.v]
add_files -norecurse $wrapper_file
update_compile_order -fileset sources_1
set_property top ${bd_name}_wrapper [get_filesets sources_1]
set_property top_auto_set 0 [get_filesets sources_1]

puts "Created Vivado project at: $build_dir"
puts "Block design name: $bd_name"
puts "Top module: ${bd_name}_wrapper"
puts "TPU RTL repo root: $repo_root"
puts "TPU MATRIX_D_BASE_ADDR: $matrix_d_base"
puts "NOTE: reserve this physical DDR range in Linux before using the userspace demo."
puts "Assigned address segments:"
foreach seg [lsort [get_bd_addr_segs]] {
  puts "  $seg"
}
