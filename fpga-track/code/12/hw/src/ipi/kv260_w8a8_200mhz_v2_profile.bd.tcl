################################################################
# Profile-enabled variant of kv260_w8a8_200mhz_v2.bd.tcl
#
# Adds AXI Performance Monitor IPs on the kernel data / weight AXI
# channels so plan 1 metrics can be sampled from software.
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

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   set existing_project [file normalize [file join [pwd] myproj project_1.xpr]]
   if {[file exists $existing_project]} {
      open_project $existing_project
   } else {
      create_project project_1 myproj -part xck26-sfvc784-2LV-c
      set_property BOARD_PART xilinx.com:kv260_som:part0:1.4 [current_project]
   }
}

set base_script [file join $script_folder kv260_w8a8_200mhz_v2.bd.tcl]
if {![file exists $base_script]} {
   error "Missing base BD script: $base_script"
}

set hls_ip_repo [file normalize [file join $script_folder ../hls/vlm_w8a8/smolvlm_prefill_unified/linear_accelerator_kv260/solution1/impl/ip]]
if {[file exists [file join $hls_ip_repo component.xml]]} {
   set ip_repo_paths [get_property ip_repo_paths [current_project]]
   if {[lsearch -exact $ip_repo_paths $hls_ip_repo] < 0} {
      set_property ip_repo_paths [concat $ip_repo_paths [list $hls_ip_repo]] [current_project]
      update_ip_catalog
   }
}

source $base_script

if {[current_bd_design -quiet] eq ""} {
   set existing_bd [get_files -quiet smolvlm_kv260_bd.bd]
   if {$existing_bd ne ""} {
      open_bd_design $existing_bd
      current_bd_design smolvlm_kv260_bd
   }
}

namespace eval profile_bd_patch {
proc add_axi_perf_monitors {} {
    set apm_vlnv "xilinx.com:ip:axi_perf_mon:5.0"

    if {[llength [get_ipdefs -all $apm_vlnv]] == 0} {
        error "AXI Performance Monitor IP not found in catalog: $apm_vlnv"
    }

    current_bd_instance /

    set apm_data [create_bd_cell -type ip -vlnv $apm_vlnv axi_perf_mon_data_0]
    set_property -dict [list \
        CONFIG.C_AXI4LITE_CORE_CLK_ASYNC {0} \
        CONFIG.C_ENABLE_PROFILE {1} \
        CONFIG.C_LITE_ADDRESS_WIDTH {16} \
        CONFIG.C_NUM_OF_COUNTERS {8} \
        CONFIG.C_SLOT_0_AXI_ADDR_WIDTH {64} \
        CONFIG.C_SLOT_0_AXI_DATA_WIDTH {128} \
        CONFIG.C_SLOT_0_AXI_ID_WIDTH {1} \
        CONFIG.C_SLOT_0_AXI_PROTOCOL {AXI4} \
    ] $apm_data

    set apm_weight [create_bd_cell -type ip -vlnv $apm_vlnv axi_perf_mon_weight_0]
    set_property -dict [list \
        CONFIG.C_AXI4LITE_CORE_CLK_ASYNC {0} \
        CONFIG.C_ENABLE_PROFILE {1} \
        CONFIG.C_LITE_ADDRESS_WIDTH {16} \
        CONFIG.C_NUM_OF_COUNTERS {8} \
        CONFIG.C_SLOT_0_AXI_ADDR_WIDTH {64} \
        CONFIG.C_SLOT_0_AXI_DATA_WIDTH {128} \
        CONFIG.C_SLOT_0_AXI_ID_WIDTH {1} \
        CONFIG.C_SLOT_0_AXI_PROTOCOL {AXI4} \
    ] $apm_weight

    set_property -dict [list CONFIG.NUM_MI {3}] [get_bd_cells smartconnect_ctrl]
    connect_bd_intf_net -intf_net smartconnect_ctrl_M01_AXI \
        [get_bd_intf_pins smartconnect_ctrl/M01_AXI] \
        [get_bd_intf_pins axi_perf_mon_data_0/S_AXI]
    connect_bd_intf_net -intf_net smartconnect_ctrl_M02_AXI \
        [get_bd_intf_pins smartconnect_ctrl/M02_AXI] \
        [get_bd_intf_pins axi_perf_mon_weight_0/S_AXI]

    connect_bd_intf_net -intf_net [get_bd_intf_nets smolvlm_prefill_unified_0_m_axi_gmem_d] \
        [get_bd_intf_pins smolvlm_prefill_unified_0/m_axi_gmem_d] \
        [get_bd_intf_pins axi_perf_mon_data_0/SLOT_0_AXI]

    connect_bd_intf_net -intf_net [get_bd_intf_nets smolvlm_prefill_unified_0_m_axi_gmem_w] \
        [get_bd_intf_pins smolvlm_prefill_unified_0/m_axi_gmem_w] \
        [get_bd_intf_pins axi_perf_mon_weight_0/SLOT_0_AXI]

    connect_bd_net -net zynq_ultra_ps_e_0_pl_clk0 \
        [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
        [get_bd_pins axi_perf_mon_data_0/core_aclk] \
        [get_bd_pins axi_perf_mon_data_0/s_axi_aclk] \
        [get_bd_pins axi_perf_mon_data_0/slot_0_axi_aclk] \
        [get_bd_pins axi_perf_mon_weight_0/core_aclk] \
        [get_bd_pins axi_perf_mon_weight_0/s_axi_aclk] \
        [get_bd_pins axi_perf_mon_weight_0/slot_0_axi_aclk]

    connect_bd_net -net proc_sys_reset_0_peripheral_aresetn \
        [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
        [get_bd_pins axi_perf_mon_data_0/core_aresetn] \
        [get_bd_pins axi_perf_mon_data_0/s_axi_aresetn] \
        [get_bd_pins axi_perf_mon_data_0/slot_0_axi_aresetn] \
        [get_bd_pins axi_perf_mon_weight_0/core_aresetn] \
        [get_bd_pins axi_perf_mon_weight_0/s_axi_aresetn] \
        [get_bd_pins axi_perf_mon_weight_0/slot_0_axi_aresetn]

    assign_bd_address -offset 0xA0000000 -range 0x00010000 \
        -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
        [get_bd_addr_segs smolvlm_prefill_unified_0/s_axi_control/Reg] -force
    assign_bd_address -offset 0xA0010000 -range 0x00010000 \
        -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
        [get_bd_addr_segs axi_perf_mon_data_0/S_AXI/Reg] -force
    assign_bd_address -offset 0xA0020000 -range 0x00010000 \
        -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
        [get_bd_addr_segs axi_perf_mon_weight_0/S_AXI/Reg] -force

    validate_bd_design
    save_bd_design
}
}

profile_bd_patch::add_axi_perf_monitors
