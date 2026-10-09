set script_dir [file dirname [file normalize [info script]]]
if {$argc >= 1} {
    set proj_root [file normalize [lindex $argv 0]]
} else {
    set proj_root [file normalize [file join $script_dir ".."]]
}
if {$argc >= 2} {
    set jobs [lindex $argv 1]
} else {
    set jobs 8
}
if {$argc >= 3} {
    set run_tag [lindex $argv 2]
} else {
    set run_tag "manual"
}

set part_name "xczu15eg-ffvb1156-2-i"
set design_name "system"
set project_name "zu15eg_llama_fpga"
set build_dir [file normalize [file join $proj_root "build_vivado_2024_1_$run_tag"]]
set out_dir [file normalize [file join $proj_root "out" $run_tag]]
file mkdir $out_dir

proc note {msg} {
    puts "ZU15EG_LLAMA_FPGA: $msg"
}

proc safe_set_property {obj prop value} {
    if {[catch {set_property $prop $value $obj} err]} {
        puts "ZU15EG_LLAMA_FPGA: WARN: failed to set $prop=$value: $err"
    }
}

proc safe_set_dict {obj dict_values} {
    if {[catch {set_property -dict $dict_values $obj} err]} {
        puts "ZU15EG_LLAMA_FPGA: WARN: failed to set dict on $obj: $err"
    }
}

proc safe_connect_intf {a b} {
    if {[catch {connect_bd_intf_net $a $b} err]} {
        puts "ZU15EG_LLAMA_FPGA: WARN: interface connect failed: $a <-> $b: $err"
    }
}

proc safe_connect_net {args} {
    if {[catch {connect_bd_net {*}$args} err]} {
        puts "ZU15EG_LLAMA_FPGA: WARN: net connect failed: $args: $err"
    }
}

proc safe_assign_addr {addr_space seg offset range} {
    if {$addr_space eq "" || $seg eq ""} {
        puts "ZU15EG_LLAMA_FPGA: WARN: skip address assignment for empty addr_space/segment at $offset/$range"
        return
    }
    if {[catch {assign_bd_address -offset $offset -range $range -target_address_space $addr_space $seg -force} err]} {
        puts "ZU15EG_LLAMA_FPGA: WARN: address assignment failed: $offset/$range -> $seg: $err"
    }
}

note "project root: $proj_root"
note "build dir: $build_dir"
note "out dir: $out_dir"
set_param general.maxThreads $jobs

create_project -force $project_name $build_dir -part $part_name
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

set rtl_file [file join $proj_root "rtl" "DataPath_xN.v"]
if {![file exists $rtl_file]} {
    error "missing RTL: $rtl_file"
}
add_files -norecurse $rtl_file
set debug_rtl_file [file join $proj_root "rtl" "sdf1_debug_axi.v"]
if {![file exists $debug_rtl_file]} {
    error "missing debug RTL: $debug_rtl_file"
}
add_files -norecurse $debug_rtl_file

set xci_files [glob -nocomplain -directory [file join $proj_root "ip"] -types f */*.xci]
if {[llength $xci_files] == 0} {
    error "no XCI files found under [file join $proj_root ip]"
}
foreach xci $xci_files {
    import_ip -files $xci -name [file rootname [file tail $xci]]
}

update_compile_order -fileset sources_1
report_ip_status -file [file join $out_dir "ip_status_initial.rpt"]

set ips [get_ips -quiet]
if {[llength $ips] > 0} {
    catch {upgrade_ip $ips} upgrade_err
    if {[info exists upgrade_err] && $upgrade_err ne ""} {
        note "IP upgrade returned: $upgrade_err"
    }
    report_ip_status -file [file join $out_dir "ip_status_after_upgrade.rpt"]
    generate_target all $ips
}

create_bd_design $design_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0]
safe_set_dict $ps [list \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {150} \
    CONFIG.PSU__USE__M_AXI_GP2 {1} \
    CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
    CONFIG.PSU__USE__S_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__S_AXI_GP3 {1} \
    CONFIG.PSU__USE__S_AXI_GP4 {1} \
    CONFIG.PSU__USE__S_AXI_GP5 {1} \
    CONFIG.PSU__SAXIGP0__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP3__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
    CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
    CONFIG.PSU__PROTECTION__MASTERS {USB1:NonSecure;0|USB0:NonSecure;1|S_AXI_LPD:NA;0|S_AXI_HPC1_FPD:NA;0|S_AXI_HPC0_FPD:NA;1|S_AXI_HP3_FPD:NA;1|S_AXI_HP2_FPD:NA;1|S_AXI_HP1_FPD:NA;1|S_AXI_HP0_FPD:NA;1|S_AXI_ACP:NA;0|S_AXI_ACE:NA;0|SD1:NonSecure;1|SD0:NonSecure;1|SATA1:NonSecure;0|SATA0:NonSecure;0|RPU1:Secure;1|RPU0:Secure;1|QSPI:NonSecure;1|PMU:NA;1|PCIe:NonSecure;1|NAND:NonSecure;0|LDMA:NonSecure;1|GPU:NonSecure;1|GEM3:NonSecure;1|GEM2:NonSecure;0|GEM1:NonSecure;0|GEM0:NonSecure;0|FDMA:NonSecure;1|DP:NonSecure;1|DAP:NA;1|Coresight:NA;1|CSU:NA;1|APU:NA;1} \
]

set axi_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* ps_axi_ctrl]
safe_set_property $axi_ctrl CONFIG.NUM_MI 2

foreach idx {0 1 2 3} {
    set smc($idx) [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc_$idx]
    safe_set_property $smc($idx) CONFIG.NUM_SI 1
    safe_set_property $smc($idx) CONFIG.NUM_MI 1
}
safe_set_property $smc(0) CONFIG.NUM_SI 2

set rst_bus [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_bus]
set rst_datapath [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_datapath]
set rst_gate [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:* rst_gate]
safe_set_property $rst_gate CONFIG.C_SIZE 1
safe_set_property $rst_gate CONFIG.C_OPERATION and

set dp [create_bd_cell -type module -reference DataPath_xN DataPath_xN_0]
set dbg [create_bd_cell -type module -reference sdf1_debug_axi sdf1_debug_axi_0]

set clk_freq [get_property CONFIG.FREQ_HZ [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]]
if {$clk_freq eq ""} {
    set clk_freq 149998505
}
note "using BD clock freq_hz: $clk_freq"
foreach intf {S00_AXIL m_axi_hp_0_0 m_axi_hp_0_1 m_axi_hp_0_2 m_axi_hp_0_3} {
    safe_set_dict [get_bd_intf_pins DataPath_xN_0/$intf] [list CONFIG.FREQ_HZ $clk_freq CONFIG.CLK_DOMAIN ${design_name}_zynq_ultra_ps_e_0_0_pl_clk0]
}
foreach intf {S_AXI M_AXI} {
    safe_set_dict [get_bd_intf_pins sdf1_debug_axi_0/$intf] [list CONFIG.FREQ_HZ $clk_freq CONFIG.CLK_DOMAIN ${design_name}_zynq_ultra_ps_e_0_0_pl_clk0]
}
safe_set_dict [get_bd_pins DataPath_xN_0/S00_ACLK] [list \
    CONFIG.FREQ_HZ $clk_freq \
    CONFIG.CLK_DOMAIN ${design_name}_zynq_ultra_ps_e_0_0_pl_clk0 \
    CONFIG.ASSOCIATED_BUSIF {S00_AXIL:m_axi_hp_0_0:m_axi_hp_0_1:m_axi_hp_0_2:m_axi_hp_0_3} \
    CONFIG.ASSOCIATED_RESET {S00_ARESETN} \
]
safe_set_dict [get_bd_pins sdf1_debug_axi_0/ACLK] [list \
    CONFIG.FREQ_HZ $clk_freq \
    CONFIG.CLK_DOMAIN ${design_name}_zynq_ultra_ps_e_0_0_pl_clk0 \
    CONFIG.ASSOCIATED_BUSIF {S_AXI:M_AXI} \
    CONFIG.ASSOCIATED_RESET {ARESETN} \
]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins ps_axi_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ps_axi_ctrl/M00_AXI] [get_bd_intf_pins sdf1_debug_axi_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins ps_axi_ctrl/M01_AXI] [get_bd_intf_pins DataPath_xN_0/S00_AXIL]

connect_bd_intf_net [get_bd_intf_pins DataPath_xN_0/m_axi_hp_0_0] [get_bd_intf_pins axi_smc_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sdf1_debug_axi_0/M_AXI] [get_bd_intf_pins axi_smc_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins DataPath_xN_0/m_axi_hp_0_1] [get_bd_intf_pins axi_smc_1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_1/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP2_FPD]
connect_bd_intf_net [get_bd_intf_pins DataPath_xN_0/m_axi_hp_0_2] [get_bd_intf_pins axi_smc_2/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_2/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP3_FPD]
connect_bd_intf_net [get_bd_intf_pins DataPath_xN_0/m_axi_hp_0_3] [get_bd_intf_pins axi_smc_3/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_3/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

set pl_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
foreach pin [list \
    [get_bd_pins ps_axi_ctrl/ACLK] \
    [get_bd_pins ps_axi_ctrl/S00_ACLK] \
    [get_bd_pins ps_axi_ctrl/M00_ACLK] \
    [get_bd_pins ps_axi_ctrl/M01_ACLK] \
    [get_bd_pins rst_bus/slowest_sync_clk] \
    [get_bd_pins rst_datapath/slowest_sync_clk] \
    [get_bd_pins DataPath_xN_0/S00_ACLK] \
    [get_bd_pins sdf1_debug_axi_0/ACLK] \
    [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_lpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihpc0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp1_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp2_fpd_aclk] \
    [get_bd_pins zynq_ultra_ps_e_0/saxihp3_fpd_aclk] \
] {
    connect_bd_net $pl_clk $pin
}
foreach idx {0 1 2 3} {
    connect_bd_net $pl_clk [get_bd_pins axi_smc_$idx/aclk]
}

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_bus/ext_reset_in]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_gate/Op2]
connect_bd_net [get_bd_pins DataPath_xN_0/softReset] [get_bd_pins rst_gate/Op1]
connect_bd_net [get_bd_pins rst_gate/Res] [get_bd_pins rst_datapath/ext_reset_in]

foreach pin [list \
    [get_bd_pins ps_axi_ctrl/ARESETN] \
    [get_bd_pins ps_axi_ctrl/S00_ARESETN] \
    [get_bd_pins ps_axi_ctrl/M00_ARESETN] \
    [get_bd_pins ps_axi_ctrl/M01_ARESETN] \
] {
    connect_bd_net [get_bd_pins rst_bus/peripheral_aresetn] $pin
}
foreach idx {0 1 2 3} {
    connect_bd_net [get_bd_pins rst_bus/peripheral_aresetn] [get_bd_pins axi_smc_$idx/aresetn]
}
connect_bd_net [get_bd_pins rst_bus/peripheral_aresetn] [get_bd_pins sdf1_debug_axi_0/ARESETN]
connect_bd_net [get_bd_pins rst_datapath/peripheral_aresetn] [get_bd_pins DataPath_xN_0/S00_ARESETN]

assign_bd_address
safe_assign_addr [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs sdf1_debug_axi_0/S_AXI/reg0] 0x96000000 0x00001000
safe_assign_addr [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs DataPath_xN_0/S00_AXIL/reg0] 0x96010000 0x00001000

safe_assign_addr [get_bd_addr_spaces DataPath_xN_0/m_axi_hp_0_0] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] 0x00000000 0x80000000
safe_assign_addr [get_bd_addr_spaces DataPath_xN_0/m_axi_hp_0_0] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_HIGH] 0x000800000000 0x000800000000
safe_assign_addr [get_bd_addr_spaces sdf1_debug_axi_0/M_AXI] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] 0x00000000 0x80000000
safe_assign_addr [get_bd_addr_spaces sdf1_debug_axi_0/M_AXI] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_HIGH] 0x000800000000 0x000800000000
safe_assign_addr [get_bd_addr_spaces DataPath_xN_0/m_axi_hp_0_1] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP4/HP2_DDR_LOW] 0x00000000 0x80000000
safe_assign_addr [get_bd_addr_spaces DataPath_xN_0/m_axi_hp_0_1] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP4/HP2_DDR_HIGH] 0x000800000000 0x000800000000
safe_assign_addr [get_bd_addr_spaces DataPath_xN_0/m_axi_hp_0_2] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP5/HP3_DDR_LOW] 0x00000000 0x80000000
safe_assign_addr [get_bd_addr_spaces DataPath_xN_0/m_axi_hp_0_2] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP5/HP3_DDR_HIGH] 0x000800000000 0x000800000000
safe_assign_addr [get_bd_addr_spaces DataPath_xN_0/m_axi_hp_0_3] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_LOW] 0x00000000 0x80000000
safe_assign_addr [get_bd_addr_spaces DataPath_xN_0/m_axi_hp_0_3] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP0/HPC0_DDR_HIGH] 0x000800000000 0x000800000000

validate_bd_design
save_bd_design
write_bd_tcl -force [file join $out_dir "system_bd.tcl"]

set wrapper [make_wrapper -files [get_files [file join $build_dir $project_name.srcs sources_1 bd $design_name ${design_name}.bd]] -top]
add_files -norecurse $wrapper
set_property top ${design_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

generate_target all [get_files [file join $build_dir $project_name.srcs sources_1 bd $design_name ${design_name}.bd]]
export_ip_user_files -of_objects [get_files [file join $build_dir $project_name.srcs sources_1 bd $design_name ${design_name}.bd]] -no_script -sync -force -quiet

update_compile_order -fileset sources_1

note "starting project synth_1"
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
note "synth_1 status: $synth_status"
if {[string first "ERROR" $synth_status] >= 0 || [string first "Failed" $synth_status] >= 0 || [string first "failed" $synth_status] >= 0} {
    error "synth_1 failed: $synth_status"
}

open_run synth_1 -name synth_1
write_checkpoint -force [file join $out_dir "post_synth.dcp"]
report_utilization -file [file join $out_dir "utilization_synth.rpt"]
report_timing_summary -file [file join $out_dir "timing_summary_synth.rpt"]
close_design

note "starting project impl_1 to write_bitstream"
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
note "impl_1 status: $impl_status"
if {[string first "ERROR" $impl_status] >= 0 || [string first "Failed" $impl_status] >= 0 || [string first "failed" $impl_status] >= 0} {
    error "impl_1 failed: $impl_status"
}

open_run impl_1 -name impl_1
write_checkpoint -force [file join $out_dir "post_route.dcp"]
report_utilization -file [file join $out_dir "utilization_routed.rpt"]
report_timing_summary -file [file join $out_dir "timing_summary_routed.rpt"]
report_drc -file [file join $out_dir "drc_routed.rpt"]

set run_bit [file join $build_dir "$project_name.runs" impl_1 "${design_name}_wrapper.bit"]
if {[file exists $run_bit]} {
    file copy -force $run_bit [file join $out_dir "system.bit"]
} else {
    note "WARN: expected bitstream not found at $run_bit"
}

write_hw_platform -fixed -include_bit -force -file [file join $out_dir "system.xsa"]

set summary_file [file join $out_dir "build_summary.txt"]
set fh [open $summary_file w]
puts $fh "project=$project_name"
puts $fh "part=$part_name"
puts $fh "vivado=[version -short]"
puts $fh "clock_mhz=150"
puts $fh "control_base=0x96000000"
puts $fh "datapath_control_base=0x96010000"
puts $fh "run_tag=$run_tag"
puts $fh "bit=[file join $out_dir system.bit]"
puts $fh "xsa=[file join $out_dir system.xsa]"
close $fh

note "build complete"
