# Reconstructed Vivado block design skeleton for the current Tryo accelerator
# decoder core.
#
# This is adapted from the public Tryo integration style, but the IP name and
# AXI topology are for the current design:
#   - TRYO_LLM_0/axilite: PS AXI-Lite control at 0xA0000000
#   - TRYO_LLM_0/gmem1:  decoder X/Y/logit Y and low-half weights
#   - TRYO_LLM_0/gmem2:  high-half weights
#   - TRYO_LLM_0/gmem3:  unified KV cache
#
# Prerequisite:
#   Package SPINAL/vivado/TryoTop_replaced.v as user.org:user:TRYO_LLM:1.0.

proc create_root_design { parentCell } {
  if { $parentCell eq "" } {
    set parentCell [get_bd_cells /]
  }

  current_bd_instance $parentCell

  set zynq_ultra_ps_e_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ultra_ps_e_0 ]
  set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {270} \
  ] $zynq_ultra_ps_e_0

  set rst_ps8_0_99M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps8_0_99M ]

  set smartconnect_ctrl [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_ctrl ]
  set_property CONFIG.NUM_SI {1} $smartconnect_ctrl

  set smartconnect_gmem1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_gmem1 ]
  set_property CONFIG.NUM_SI {1} $smartconnect_gmem1

  set smartconnect_gmem3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_gmem3 ]
  set_property CONFIG.NUM_SI {1} $smartconnect_gmem3

  set TRYO_LLM_0 [ create_bd_cell -type ip -vlnv user.org:user:TRYO_LLM:1.0 TRYO_LLM_0 ]

  connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smartconnect_ctrl/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins smartconnect_ctrl/M00_AXI] [get_bd_intf_pins TRYO_LLM_0/axilite]

  connect_bd_intf_net [get_bd_intf_pins TRYO_LLM_0/gmem1] [get_bd_intf_pins smartconnect_gmem1/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins smartconnect_gmem1/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP2_FPD]

  connect_bd_intf_net [get_bd_intf_pins TRYO_LLM_0/gmem2] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP3_FPD]

  connect_bd_intf_net [get_bd_intf_pins TRYO_LLM_0/gmem3] [get_bd_intf_pins smartconnect_gmem3/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins smartconnect_gmem3/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP1_FPD]

  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_ps8_0_99M/ext_reset_in]
  connect_bd_net [get_bd_pins rst_ps8_0_99M/peripheral_aresetn] \
    [get_bd_pins smartconnect_ctrl/aresetn] \
    [get_bd_pins smartconnect_gmem1/aresetn] \
    [get_bd_pins smartconnect_gmem3/aresetn] \
    [get_bd_pins TRYO_LLM_0/resetn]

  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp1_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp2_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp3_fpd_aclk] \
    [get_bd_pins rst_ps8_0_99M/slowest_sync_clk] \
    [get_bd_pins smartconnect_ctrl/aclk] \
    [get_bd_pins smartconnect_gmem1/aclk] \
    [get_bd_pins smartconnect_gmem3/aclk] \
    [get_bd_pins TRYO_LLM_0/clk]

  assign_bd_address -offset 0xA0000000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
    [get_bd_addr_segs TRYO_LLM_0/axilite/reg0] -force

  assign_bd_address -offset 0x00000000 -range 0x80000000 \
    -target_address_space [get_bd_addr_spaces TRYO_LLM_0/gmem1] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP4/HP2_DDR_LOW] -force



  assign_bd_address -offset 0x00000000 -range 0x80000000 \
    -target_address_space [get_bd_addr_spaces TRYO_LLM_0/gmem2] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP5/HP3_DDR_LOW] -force

  assign_bd_address -offset 0x00000000 -range 0x80000000 \
    -target_address_space [get_bd_addr_spaces TRYO_LLM_0/gmem3] \
    [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP3/HP1_DDR_LOW] -force

  save_bd_design
}
