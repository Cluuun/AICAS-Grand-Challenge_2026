set script_dir [file normalize [file dirname [info script]]]
set repo_root  [file normalize [file join $script_dir .. ..]]
set build_dir  [file normalize [file join $script_dir build]]
set project_name "pynq_zu_dma_loopback"
set bd_name "pynq_zu_dma_loopback_bd"

# PYNQ-ZU board files are not installed on the remote Vivado machine, so use
# the device part directly. Board references:
# - PYNQ-ZU overview page: XCZU5EG-SFVC784
# - XUP HLS flow page: xczu5eg-sfvc784-1-e
set target_part "xczu5eg-sfvc784-1-e"

file mkdir $build_dir

create_project $project_name $build_dir -force -part $target_part
set_property target_language Verilog [current_project]

add_files [file join $repo_root rtl ps_pl_dma_loopback.v]
update_compile_order -fileset sources_1

create_bd_design $bd_name

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ultra_ps_e_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_ctrl_0
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0
create_bd_cell -type module -reference ps_pl_dma_loopback ps_pl_dma_loopback_0

apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "0"} [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP1 {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

# Clocking and reset
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_dma_0/s_axi_lite_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_dma_0/m_axi_s2mm_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins ps_pl_dma_loopback_0/s_axi_control_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_ctrl_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_lpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihpc1_fpd_aclk]

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins proc_sys_reset_0/ext_reset_in]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_ctrl_0/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins axi_dma_0/axi_resetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins ps_pl_dma_loopback_0/s_axi_control_aresetn]

set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_include_s2mm_dre {1} \
    CONFIG.c_m_axi_mm2s_data_width {128} \
    CONFIG.c_m_axi_s2mm_data_width {128} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
] [get_bd_cells axi_dma_0]

set_property CONFIG.NUM_MI {2} [get_bd_cells smartconnect_ctrl_0]
set_property CONFIG.NUM_SI {1} [get_bd_cells smartconnect_ctrl_0]

# AXI-Lite control path: PS -> DMA lite + demo CSR
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins smartconnect_ctrl_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_ctrl_0/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins smartconnect_ctrl_0/M01_AXI] [get_bd_intf_pins ps_pl_dma_loopback_0/S_AXI_CONTROL]

# Memory-mapped DMA path: DMA masters -> PS DDR slave ports
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC1_FPD]

# Stream path
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins ps_pl_dma_loopback_0/S_AXIS_IN]
connect_bd_intf_net [get_bd_intf_pins ps_pl_dma_loopback_0/M_AXIS_OUT] [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

assign_bd_address
validate_bd_design
save_bd_design

make_wrapper -files [get_files [file join $build_dir $project_name.srcs sources_1 bd $bd_name $bd_name.bd]] -top
add_files -norecurse [file join $build_dir $project_name.gen sources_1 bd $bd_name hdl ${bd_name}_wrapper.v]
update_compile_order -fileset sources_1

puts "Created Vivado project at: $build_dir"
puts "Block design name: $bd_name"
