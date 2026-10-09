# kv260_w8a8_300mhz_v6.bd.tcl --- Vivado IPI block design for vlm_engine_v6.

namespace eval _tcl_v6 {
    proc get_script_folder {} {
        return [file dirname [file normalize [info script]]]
    }
}
set script_folder [_tcl_v6::get_script_folder]

set scripts_vivado_version 2024.2
set current_vivado_version [version -short]
if {[string first $scripts_vivado_version $current_vivado_version] == -1} {
    common::send_gid_msg -ssname BD::TCL -id 2040 -severity "CRITICAL WARNING" \
        "This script was generated for Vivado $scripts_vivado_version; running on $current_vivado_version."
}

if {[get_projects -quiet] eq ""} {
    create_project v6_proj v6_proj -part xck26-sfvc784-2LV-c
    set_property BOARD_PART xilinx.com:kv260_som:part0:1.4 [current_project]
}

variable design_name
set design_name kv260_v6_bd

set target_mhz 300
if {[info exists ::env(V6_TARGET_MHZ)] && $::env(V6_TARGET_MHZ) ne ""} {
    set target_mhz $::env(V6_TARGET_MHZ)
}
if {$target_mhz ni {250 300 375}} {
    error "ERROR: V6_TARGET_MHZ must be a KV260 PL0-supported rate: 250, 300, or 375; got: $target_mhz"
}

if {[current_bd_design -quiet] eq ""} {
    create_bd_design $design_name
    current_bd_design $design_name
}

proc cell {name vlnv {props {}}} {
    set c [create_bd_cell -type ip -vlnv $vlnv $name]
    if {[llength $props]} {
        set_property -dict $props $c
    }
    return $c
}

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

set sc_ctrl [cell sc_ctrl xilinx.com:ip:smartconnect:* [list \
    CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}]]
connect_bd_intf_net [get_bd_intf_pins $zynq/M_AXI_HPM0_FPD] [get_bd_intf_pins $sc_ctrl/S00_AXI]
connect_bd_net [get_bd_pins $zynq/pl_clk0] [get_bd_pins $sc_ctrl/aclk]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $sc_ctrl/aresetn]

set engine [cell vlm_engine_v6 aicas:V6:vlm_engine_v6:*]

connect_bd_intf_net [get_bd_intf_pins $sc_ctrl/M00_AXI]       [get_bd_intf_pins $engine/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins $engine/m_axi_gmem_w0]  [get_bd_intf_pins $zynq/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins $engine/m_axi_gmem_w1]  [get_bd_intf_pins $zynq/S_AXI_HP1_FPD]
connect_bd_intf_net [get_bd_intf_pins $engine/m_axi_gmem_act] [get_bd_intf_pins $zynq/S_AXI_HP2_FPD]
connect_bd_intf_net [get_bd_intf_pins $engine/m_axi_gmem_out] [get_bd_intf_pins $zynq/S_AXI_HP3_FPD]

foreach hp_idx {0 1 2 3} {
    connect_bd_net [get_bd_pins $zynq/pl_clk0] [get_bd_pins $zynq/saxihp${hp_idx}_fpd_aclk]
}

set pe [cell pe_array aicas:user:pe_array_v6:1.0]
set_property CONFIG.ASSOCIATED_BUSIF {s_axis_a:s_axis_w0:s_axis_w1:s_axis_ctrl:m_axis_psum} [get_bd_pins $pe/ap_clk]

proc axis_reg_chain {prefix src_ip src_port dst_ip dst_port} {
    set rs [cell ${prefix}_rs xilinx.com:ip:axis_register_slice:*]
    connect_bd_intf_net [get_bd_intf_pins $src_ip/$src_port] [get_bd_intf_pins $rs/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins $rs/M_AXIS]        [get_bd_intf_pins $dst_ip/$dst_port]
    connect_bd_net [get_bd_pins $rs/aclk]    [get_bd_pins zynq_ps/pl_clk0]
    connect_bd_net [get_bd_pins $rs/aresetn] [get_bd_pins rst_pl_300m/peripheral_aresetn]
}

proc axis_fifo_chain {prefix src_ip src_port dst_ip dst_port {fifo_depth 128}} {
    set rs [cell ${prefix}_rs xilinx.com:ip:axis_register_slice:*]
    set fifo [cell ${prefix}_fifo xilinx.com:ip:axis_data_fifo:* [list \
        CONFIG.FIFO_DEPTH $fifo_depth \
        CONFIG.FIFO_MEMORY_TYPE {distributed} \
        CONFIG.HAS_TLAST {0}]]
    connect_bd_intf_net [get_bd_intf_pins $src_ip/$src_port] [get_bd_intf_pins $rs/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins $rs/M_AXIS]        [get_bd_intf_pins $fifo/S_AXIS]
    connect_bd_intf_net [get_bd_intf_pins $fifo/M_AXIS]      [get_bd_intf_pins $dst_ip/$dst_port]
    connect_bd_net [get_bd_pins $rs/aclk]    [get_bd_pins zynq_ps/pl_clk0]
    connect_bd_net [get_bd_pins $rs/aresetn] [get_bd_pins rst_pl_300m/peripheral_aresetn]
    connect_bd_net [get_bd_pins $fifo/s_axis_aclk]    [get_bd_pins zynq_ps/pl_clk0]
    connect_bd_net [get_bd_pins $fifo/s_axis_aresetn] [get_bd_pins rst_pl_300m/peripheral_aresetn]
}

# The PE input streams are cycle-lockstep and same-clock. Deep FIFOs on the
# weight streams caused severe route congestion in the first 8x128 build
# (router overlap stuck at 143985/52974). Keep only register slices there and
# let PE backpressure stall the HLS producer during drain. Split w0/w1 keeps
# the 8x128 geometry while avoiding a single top-level 1024b bundle.
axis_reg_chain  a    vlm_engine_v6 m_axis_a_stream    pe_array s_axis_a
axis_reg_chain  w0   vlm_engine_v6 m_axis_w0_stream   pe_array s_axis_w0
axis_reg_chain  w1   vlm_engine_v6 m_axis_w1_stream   pe_array s_axis_w1
axis_reg_chain  ctrl vlm_engine_v6 m_axis_ctrl_stream pe_array s_axis_ctrl
# The psum path is consumed by HLS at II=1 during drain/absorb. A full 128-beat
# block-RAM FIFO costs four RAMB36 tiles in an already 96% BRAM-utilized design,
# so keep only a shallow distributed FIFO for backpressure smoothing.
axis_fifo_chain psum pe_array      m_axis_psum        vlm_engine_v6 s_axis_psum_stream 32

connect_bd_net [get_bd_pins $zynq/pl_clk0] [get_bd_pins $engine/ap_clk]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $engine/ap_rst_n]
connect_bd_net [get_bd_pins $zynq/pl_clk0] [get_bd_pins $pe/ap_clk]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $pe/ap_rst_n]

assign_bd_address -target_address_space /zynq_ps/Data \
    [get_bd_addr_segs vlm_engine_v6/s_axi_control/Reg]
foreach {mport hpseg} {
    gmem_w0  SAXIGP2/HP0_DDR_LOW
    gmem_w1  SAXIGP3/HP1_DDR_LOW
    gmem_act SAXIGP4/HP2_DDR_LOW
    gmem_out SAXIGP5/HP3_DDR_LOW
} {
    assign_bd_address -target_address_space /vlm_engine_v6/Data_m_axi_${mport} \
        [get_bd_addr_segs zynq_ps/${hpseg}]
}

regenerate_bd_layout
validate_bd_design
save_bd_design

puts "v6 BD '$design_name' constructed and saved."
