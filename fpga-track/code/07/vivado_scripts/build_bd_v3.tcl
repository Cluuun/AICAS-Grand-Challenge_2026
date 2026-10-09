# OViA BD v3 — with S_AXI_HP0_FPD data path working
create_project -force ovia_bd3 G:/mimo_project/ovia_hls/ovia_bd3 -part xck26-sfvc784-2LV-c
set_property target_language VHDL [current_project]
set_property ip_repo_paths G:/mimo_project/ovia_hls/ovia_prj/solution1/impl/ip [current_project]
update_ip_catalog

create_bd_design "design_1"

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 zynq_ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100.000000} \
] [get_bd_cells zynq_ps]

create_bd_cell -type ip -vlnv ourteam:hls:ovia_core:1.0 ovia_0

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ctrl
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] [get_bd_cells axi_ctrl]

create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_data
set_property -dict [list CONFIG.NUM_MI {1} CONFIG.NUM_SI {1}] [get_bd_cells axi_data]

connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ctrl/M00_AXI] [get_bd_intf_pins ovia_0/s_axi_CONTROL]
connect_bd_intf_net [get_bd_intf_pins ovia_0/m_axi_OCM] [get_bd_intf_pins axi_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_data/M00_AXI] [get_bd_intf_pins zynq_ps/S_AXI_HP0_FPD]

connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins ovia_0/ap_clk]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_ctrl/ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_ctrl/S00_ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_ctrl/M00_ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_data/ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_data/S00_ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins axi_data/M00_ACLK]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins zynq_ps/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins zynq_ps/maxihpm0_lpd_aclk]
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins zynq_ps/saxihp0_fpd_aclk]

connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins ovia_0/ap_rst_n]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_ctrl/ARESETN]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_ctrl/S00_ARESETN]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_ctrl/M00_ARESETN]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_data/ARESETN]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_data/S00_ARESETN]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins axi_data/M00_ARESETN]

assign_bd_address
validate_bd_design
puts "=== BD VALIDATED ==="

save_bd_design
generate_target all [get_files G:/mimo_project/ovia_bd3/ovia_bd3.srcs/sources_1/bd/design_1/design_1.bd]
make_wrapper -files [get_files G:/mimo_project/ovia_bd3/ovia_bd3.srcs/sources_1/bd/design_1/design_1.bd] -top
set wrapper_vhd [file join G:/mimo_project/ovia_hls/ovia_bd3 ovia_bd3.gen sources_1 bd design_1 hdl design_1_wrapper.vhd]
if {[file exists $wrapper_vhd]} {
    add_files -norecurse $wrapper_vhd
} else {
    set wrapper_v [file join G:/mimo_project/ovia_hls/ovia_bd3 ovia_bd3.gen sources_1 bd design_1 hdl design_1_wrapper.v]
    if {[file exists $wrapper_v]} {
        add_files -norecurse $wrapper_v
    } else {
        puts "ERROR: No wrapper file found"
    }
}
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -jobs 4
wait_on_run impl_1
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
puts [format "Bitstream: %s" [glob G:/mimo_project/ovia_bd3/ovia_bd3.runs/impl_1/*.bit]]
