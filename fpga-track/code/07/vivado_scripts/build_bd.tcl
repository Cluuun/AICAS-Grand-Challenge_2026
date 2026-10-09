# OViA BD — Simplified: manual PS config, all clock/reset pins connected

create_project -force ovia_bd G:/mimo_project/ovia_bd -part xck26-sfvc784-2LV-c
set_property target_language VHDL [current_project]

set_property ip_repo_paths G:/mimo_project/ovia_hls/ovia_prj/solution1/impl/ip [current_project]
update_ip_catalog

create_bd_design "design_1"

# Zynq PS with explicit configuration
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ps
set_property -dict [list \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100.000000} \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__USE__IRQ0 {0} \
] [get_bd_cells zynq_ps]

# OViA core
create_bd_cell -type ip -vlnv ourteam:hls:ovia_core:1.0 ovia_0

# AXI interconnect
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ctrl
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1} CONFIG.STRATEGY {1}] [get_bd_cells axi_ctrl]

# Control path
connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_LPD] [get_bd_intf_pins axi_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ctrl/M00_AXI] [get_bd_intf_pins ovia_0/s_axi_CONTROL]

# PS clock out → all PL clock pins
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins ovia_0/ap_clk]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_ctrl/ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_ctrl/S00_ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_ctrl/M00_ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins zynq_ps/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins zynq_ps/maxihpm0_lpd_aclk]

# PS reset → all PL reset pins
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins ovia_0/ap_rst_n]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_ctrl/ARESETN]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_ctrl/S00_ARESETN]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_ctrl/M00_ARESETN]

# Address assignment
assign_bd_address

# Validate strictly
validate_bd_design

puts "=== BD VALIDATED ==="
puts "Generating HDL wrapper..."
make_wrapper -files [get_files G:/mimo_project/ovia_bd/ovia_bd.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse G:/mimo_project/ovia_bd/ovia_bd.srcs/sources_1/bd/design_1/hdl/design_1_wrapper.vhd

puts "=== Starting Synthesis ==="
launch_runs synth_1 -jobs 4
wait_on_run synth_1
puts "=== Synthesis Complete ==="

puts "=== Starting Implementation ==="
launch_runs impl_1 -jobs 4
wait_on_run impl_1
puts "=== Implementation Complete ==="

puts "=== Generating Bitstream ==="
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

puts "=== DONE ==="
puts [format "Bitstream: %s" [glob G:/mimo_project/ovia_bd/ovia_bd.runs/impl_1/*.bit]]
