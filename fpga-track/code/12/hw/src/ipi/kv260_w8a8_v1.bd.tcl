################################################################
# KV260 SmolVLM unified prefill block design
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

set scripts_vivado_version 2024.2
set current_vivado_version [version -short]
if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   common::send_gid_msg -ssname BD::TCL -id 2040 -severity "CRITICAL WARNING" \
      "This script was generated for Vivado <$scripts_vivado_version> and is now running in <$current_vivado_version>. IP or board preset differences may affect the design."
}

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   create_project project_1 myproj -part xck26-sfvc784-2LV-c
   set_property BOARD_PART xilinx.com:kv260_som:part0:1.4 [current_project]
}

variable design_name
set design_name smolvlm_kv260_bd

if { [get_files -quiet ${design_name}.bd] ne "" } {
   current_bd_design $design_name
} else {
   create_bd_design $design_name
   current_bd_design $design_name
}

set list_check_ips "\
xilinx.com:ip:zynq_ultra_ps_e:* \
xilinx.com:hls:smolvlm_prefill_unified_w8a8:* \
xilinx.com:ip:smartconnect:* \
xilinx.com:ip:proc_sys_reset:*\
"

set list_ips_missing ""
foreach ip_vlnv $list_check_ips {
   set ip_obj [get_ipdefs -all $ip_vlnv]
   if { $ip_obj eq "" } {
      lappend list_ips_missing $ip_vlnv
   }
}
if { $list_ips_missing ne "" } {
   error "The following IPs are not found in the IP catalog: $list_ips_missing"
}

proc create_root_design { parentCell } {
  variable design_name

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     error "Unable to find parent cell <$parentCell>"
  }

  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     error "Parent <$parentObj> has TYPE = <$parentType>. Expected <hier>."
  }

  set oldCurInst [current_bd_instance .]
  current_bd_instance $parentObj

  set existing_cells [get_bd_cells -quiet]
  if { $existing_cells ne "" } {
     delete_bd_objs $existing_cells
  }

  set zynq_ultra_ps_e_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ultra_ps_e_0]
  apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
      -config {apply_board_preset "1"} [get_bd_cells zynq_ultra_ps_e_0]
  set_property -dict [list \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {150} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP1 {0} \
  ] $zynq_ultra_ps_e_0

  set smolvlm_prefill_unified_0 [create_bd_cell -type ip -vlnv xilinx.com:hls:smolvlm_prefill_unified_w8a8 smolvlm_prefill_unified_0]

  set smartconnect_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smartconnect_ctrl]
  set_property -dict [list \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI {1} \
  ] $smartconnect_ctrl

  set smartconnect_data [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smartconnect_data]
  set_property -dict [list \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI {1} \
  ] $smartconnect_data

  set proc_sys_reset_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset proc_sys_reset_0]

  connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smartconnect_ctrl/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins smartconnect_ctrl/M00_AXI] [get_bd_intf_pins smolvlm_prefill_unified_0/s_axi_control]
  connect_bd_intf_net [get_bd_intf_pins smolvlm_prefill_unified_0/m_axi_gmem] [get_bd_intf_pins smartconnect_data/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins smartconnect_data/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

  connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins smolvlm_prefill_unified_0/ap_rst_n] \
    [get_bd_pins smartconnect_ctrl/aresetn] \
    [get_bd_pins smartconnect_data/aresetn]

  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins smartconnect_ctrl/aclk] \
    [get_bd_pins smartconnect_data/aclk] \
    [get_bd_pins smolvlm_prefill_unified_0/ap_clk] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm1_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk]

  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]

  assign_bd_address -offset 0xA0000000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    [get_bd_addr_segs smolvlm_prefill_unified_0/s_axi_control/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 \
    -target_address_space [get_bd_addr_spaces smolvlm_prefill_unified_0/Data_m_axi_gmem] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_LOW] -force
  assign_bd_address -offset 0xFF000000 -range 0x01000000 \
    -target_address_space [get_bd_addr_spaces smolvlm_prefill_unified_0/Data_m_axi_gmem] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_LPS_OCM] -force

  current_bd_instance $oldCurInst
  validate_bd_design
  save_bd_design
}

create_root_design ""
