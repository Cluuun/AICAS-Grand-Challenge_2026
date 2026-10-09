# kv260_w8a8_300mhz_v5.bd.tcl --- Vivado IPI block design for vlm_engine_v5.
#
# Topology:
#   PS Zynq UltraScale+ ----HPM0_FPD/AXI-Lite---> smartconnect_ctrl --> vlm_engine_v5 (HLS IP)
#                       <---HP0/HP1/HP2/HP3----- vlm_engine_v5 (4 m_axi)
#                                                            |
#                                                  AXIS feeders/drain
#                                                            v
#                                                   pe_array_v5 (RTL IP)
#
# AXI-Stream cross between HLS and PE:
#   a_stream:    HLS m_axis_a_stream    -> axis_register_slice -> axis_data_fifo -> PE s_axis_a
#   w_stream:    HLS m_axis_w_stream    -> axis_register_slice -> axis_data_fifo -> PE s_axis_w
#   ctrl_stream: HLS m_axis_ctrl_stream -> axis_register_slice                   -> PE s_axis_ctrl
#   psum_stream: PE  m_axis_psum  -> axis_data_fifo -> axis_register_slice -> HLS s_axis_psum_stream
#
# IP repo paths the user must add before sourcing:
#   set_property ip_repo_paths [list \
#       <path-to>/hw/src/rtl_ip_repo/pe_array_v5 \
#       <path-to>/hw/src/hls/vlm_w8a8/vlm_engine_v5/vlm_engine_v5_kv260/solution1/impl/ip \
#   ] [current_project]
#   update_ip_catalog

namespace eval _tcl_v5 {
    proc get_script_folder {} {
        return [file dirname [file normalize [info script]]]
    }
}
set script_folder [_tcl_v5::get_script_folder]

set scripts_vivado_version 2024.2
set current_vivado_version  [version -short]
if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
    common::send_gid_msg -ssname BD::TCL -id 2040 -severity "CRITICAL WARNING" \
        "This script was generated for Vivado $scripts_vivado_version; running on $current_vivado_version."
}

# ----- Project bootstrap -----
if {[get_projects -quiet] eq ""} {
    create_project v5_proj v5_proj -part xck26-sfvc784-2LV-c
    set_property BOARD_PART xilinx.com:kv260_som:part0:1.4 [current_project]
}

variable design_name
set design_name kv260_v5_bd
set target_mhz 300
if {[info exists ::env(V5_TARGET_MHZ)] && $::env(V5_TARGET_MHZ) ne ""} {
    set target_mhz $::env(V5_TARGET_MHZ)
}
if {$target_mhz ni {250 300 375}} {
    error "ERROR: V5_TARGET_MHZ must be a KV260 PL0-supported rate: 250, 300, or 375; got: $target_mhz"
}

if {[current_bd_design -quiet] eq ""} {
    create_bd_design $design_name
    current_bd_design $design_name
}

# ----- Helper: instantiate an IP with sane defaults -----
proc cell {name vlnv {props {}}} {
    set c [create_bd_cell -type ip -vlnv $vlnv $name]
    if {[llength $props]} {
        set_property -dict $props $c
    }
    return $c
}

# ----- PS / clocking / reset -----
set zynq [cell zynq_ps xilinx.com:ip:zynq_ultra_ps_e:* [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $target_mhz \
]]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config { \
    apply_board_preset 1 } $zynq
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $target_mhz \
] $zynq

set rst [cell rst_pl_300m xilinx.com:ip:proc_sys_reset:*]
connect_bd_net [get_bd_pins $zynq/pl_clk0]    [get_bd_pins $rst/slowest_sync_clk]
connect_bd_net [get_bd_pins $zynq/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
foreach hpm_clk {maxihpm0_fpd_aclk maxihpm1_fpd_aclk} {
    set hpm_pin [get_bd_pins -quiet $zynq/$hpm_clk]
    if {$hpm_pin ne ""} {
        connect_bd_net [get_bd_pins $zynq/pl_clk0] $hpm_pin
    }
}

# ----- Smartconnect for the AXI-Lite control bus -----
set sc_ctrl [cell sc_ctrl xilinx.com:ip:smartconnect:* [list \
    CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}]]
connect_bd_intf_net [get_bd_intf_pins $zynq/M_AXI_HPM0_FPD] [get_bd_intf_pins $sc_ctrl/S00_AXI]
connect_bd_net [get_bd_pins $zynq/pl_clk0] [get_bd_pins $sc_ctrl/aclk]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $sc_ctrl/aresetn]

# ----- HLS engine -----
set engine [cell vlm_engine_v5 aicas:V5:vlm_engine_v5:*]

connect_bd_intf_net [get_bd_intf_pins $sc_ctrl/M00_AXI]   [get_bd_intf_pins $engine/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins $engine/m_axi_gmem_w0]  [get_bd_intf_pins $zynq/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins $engine/m_axi_gmem_w1]  [get_bd_intf_pins $zynq/S_AXI_HP1_FPD]
connect_bd_intf_net [get_bd_intf_pins $engine/m_axi_gmem_act] [get_bd_intf_pins $zynq/S_AXI_HP2_FPD]
connect_bd_intf_net [get_bd_intf_pins $engine/m_axi_gmem_out] [get_bd_intf_pins $zynq/S_AXI_HP3_FPD]

foreach hp_idx {0 1 2 3} {
    connect_bd_net [get_bd_pins $zynq/pl_clk0] [get_bd_pins $zynq/saxihp${hp_idx}_fpd_aclk]
}

# ----- PE array IP -----
set pe [cell pe_array aicas:user:pe_array_v5:1.0]
set_property CONFIG.ASSOCIATED_BUSIF {s_axis_a:s_axis_w:s_axis_ctrl:m_axis_psum} [get_bd_pins $pe/ap_clk]

# ----- AXIS interconnect: HLS -> PE -----
proc axis_chain {prefix src_ip src_port dst_ip dst_port {fifo_depth 512}} {
    set rs  [cell ${prefix}_rs  xilinx.com:ip:axis_register_slice:*]
    set fifo [cell ${prefix}_fifo xilinx.com:ip:axis_data_fifo:* [list \
        CONFIG.FIFO_DEPTH $fifo_depth \
        CONFIG.HAS_TLAST  {0} ]]
    connect_bd_intf_net [get_bd_intf_pins $src_ip/$src_port]   [get_bd_intf_pins $rs/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins $rs/M_AXIS]          [get_bd_intf_pins $fifo/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins $fifo/M_AXIS]        [get_bd_intf_pins $dst_ip/$dst_port]
    connect_bd_net [get_bd_pins $rs/aclk]    [get_bd_pins zynq_ps/pl_clk0]
    connect_bd_net [get_bd_pins $rs/aresetn] [get_bd_pins rst_pl_300m/peripheral_aresetn]
    connect_bd_net [get_bd_pins $fifo/s_axis_aclk]    [get_bd_pins zynq_ps/pl_clk0]
    connect_bd_net [get_bd_pins $fifo/s_axis_aresetn] [get_bd_pins rst_pl_300m/peripheral_aresetn]
}

# 4 streams: a, w, ctrl (HLS->PE), psum (PE->HLS)
axis_chain a    vlm_engine_v5 m_axis_a_stream    pe_array s_axis_a    512
axis_chain w    vlm_engine_v5 m_axis_w_stream    pe_array s_axis_w    512
axis_chain ctrl vlm_engine_v5 m_axis_ctrl_stream pe_array s_axis_ctrl 64
axis_chain psum pe_array      m_axis_psum        vlm_engine_v5 s_axis_psum_stream 512

# Engine clock + reset
connect_bd_net [get_bd_pins $zynq/pl_clk0] [get_bd_pins $engine/ap_clk]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $engine/ap_rst_n]
connect_bd_net [get_bd_pins $zynq/pl_clk0] [get_bd_pins $pe/ap_clk]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $pe/ap_rst_n]

# ----- Address map -----
assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs vlm_engine_v5/s_axi_control/Reg]
foreach {mport hpseg} {
    gmem_w0  SAXIGP2/HP0_DDR_LOW
    gmem_w1  SAXIGP3/HP1_DDR_LOW
    gmem_act SAXIGP4/HP2_DDR_LOW
    gmem_out SAXIGP5/HP3_DDR_LOW
} {
    assign_bd_address -target_address_space /vlm_engine_v5/Data_m_axi_${mport} \
        [get_bd_addr_segs zynq_ps/${hpseg}]
}

regenerate_bd_layout
validate_bd_design
save_bd_design

puts "v5 BD '$design_name' constructed and saved."
