set script_dir [file dirname [file normalize [info script]]]
set proj_root [file normalize [file join $script_dir ".."]]
if {$argc >= 1} { set proj_root [file normalize [lindex $argv 0]] }
if {$argc >= 2} { set profile [lindex $argv 1] } else { set profile "zu15eg_scale" }
if {$argc >= 3} { set jobs [lindex $argv 2] } else { set jobs 8 }
if {$argc >= 4} { set run_tag [lindex $argv 3] } else { set run_tag "manual" }

if {$profile eq "zu15eg_scale"} {
    set part_name "xczu15eg-ffvb1156-2-i"
    set board_part ""
    set rtl_target "sdf5_shared"
    set clock_mhz 150
    set ctrl_base 0x96000000
    set ctrl_master "M_AXI_HPM0_LPD"
    set ctrl_clk_pin "maxihpm0_lpd_aclk"
    set data_slave "S_AXI_HP0_FPD"
    set data_clk_pin "saxihp0_fpd_aclk"
    set data_seg "SAXIGP2/HP0_DDR_LOW"
    set ps_props [list \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clock_mhz \
        CONFIG.PSU__USE__M_AXI_GP2 {1} \
        CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    ]
} elseif {$profile eq "zu15eg_wide4" || $profile eq "zu15eg_wide4_125"} {
    set part_name "xczu15eg-ffvb1156-2-i"
    set board_part ""
    set rtl_target "sdf5_shared"
    if {$profile eq "zu15eg_wide4_125"} { set clock_mhz 125 } else { set clock_mhz 150 }
    set ctrl_base 0x96000000
    set ctrl_master "M_AXI_HPM0_LPD"
    set ctrl_clk_pin "maxihpm0_lpd_aclk"
    set data_slave "S_AXI_HP0_FPD"
    set data_clk_pin "saxihp0_fpd_aclk"
    set data_seg "SAXIGP2/HP0_DDR_LOW"
    set ps_props [list \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clock_mhz \
        CONFIG.PSU__USE__M_AXI_GP2 {1} \
        CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    ]
} elseif {$profile eq "kv260_safe"} {
    set part_name "xck26-sfvc784-2LV-c"
    set board_part "xilinx.com:kv260_som:part0:1.4"
    set rtl_target "sdf5_shared"
    set clock_mhz 150
    set ctrl_base 0x96000000
    set ctrl_master "M_AXI_HPM0_LPD"
    set ctrl_clk_pin "maxihpm0_lpd_aclk"
    set data_slave "S_AXI_HP0_FPD"
    set data_clk_pin "saxihp0_fpd_aclk"
    set data_seg "SAXIGP2/HP0_DDR_LOW"
    set ps_props [list \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clock_mhz \
        CONFIG.PSU__USE__M_AXI_GP2 {1} \
        CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
        CONFIG.PSU__PROTECTION__MASTERS {USB1:NonSecure;0|USB0:NonSecure;0|S_AXI_LPD:NA;0|S_AXI_HPC1_FPD:NA;0|S_AXI_HPC0_FPD:NA;1|S_AXI_HP3_FPD:NA;1|S_AXI_HP2_FPD:NA;1|S_AXI_HP1_FPD:NA;0|S_AXI_HP0_FPD:NA;1|S_AXI_ACP:NA;0|S_AXI_ACE:NA;0|SD1:NonSecure;1|SD0:NonSecure;0|SATA1:NonSecure;0|SATA0:NonSecure;0|RPU1:Secure;1|RPU0:Secure;1|QSPI:NonSecure;1|PMU:NA;1|PCIe:NonSecure;0|NAND:NonSecure;0|LDMA:NonSecure;1|GPU:NonSecure;1|GEM3:NonSecure;0|GEM2:NonSecure;0|GEM1:NonSecure;0|GEM0:NonSecure;0|FDMA:NonSecure;1|DP:NonSecure;0|DAP:NA;1|Coresight:NA;1|CSU:NA;1|APU:NA;1} \
    ]
} elseif {$profile eq "kv260_fpd_low"} {
    set part_name "xck26-sfvc784-2LV-c"
    set board_part "xilinx.com:kv260_som:part0:1.4"
    set rtl_target "sdf5_shared"
    set clock_mhz 150
    set ctrl_base 0x96000000
    set ctrl_master "M_AXI_HPM0_FPD"
    set ctrl_clk_pin "maxihpm0_fpd_aclk"
    set data_slave "S_AXI_HP0_FPD"
    set data_clk_pin "saxihp0_fpd_aclk"
    set data_seg "SAXIGP2/HP0_DDR_LOW"
    set ps_props [list \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clock_mhz \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
        CONFIG.PSU__PROTECTION__MASTERS {USB1:NonSecure;0|USB0:NonSecure;0|S_AXI_LPD:NA;0|S_AXI_HPC1_FPD:NA;0|S_AXI_HPC0_FPD:NA;1|S_AXI_HP3_FPD:NA;1|S_AXI_HP2_FPD:NA;1|S_AXI_HP1_FPD:NA;0|S_AXI_HP0_FPD:NA;1|S_AXI_ACP:NA;0|S_AXI_ACE:NA;0|SD1:NonSecure;1|SD0:NonSecure;0|SATA1:NonSecure;0|SATA0:NonSecure;0|RPU1:Secure;1|RPU0:Secure;1|QSPI:NonSecure;1|PMU:NA;1|PCIe:NonSecure;0|NAND:NonSecure;0|LDMA:NonSecure;1|GPU:NonSecure;1|GEM3:NonSecure;0|GEM2:NonSecure;0|GEM1:NonSecure;0|GEM0:NonSecure;0|FDMA:NonSecure;1|DP:NonSecure;0|DAP:NA;1|Coresight:NA;1|CSU:NA;1|APU:NA;1} \
    ]
} elseif {$profile eq "kv260_fpd_a000"} {
    set part_name "xck26-sfvc784-2LV-c"
    set board_part "xilinx.com:kv260_som:part0:1.4"
    set rtl_target "sdf5_shared"
    set clock_mhz 150
    set ctrl_base 0xA0000000
    set ctrl_master "M_AXI_HPM0_FPD"
    set ctrl_clk_pin "maxihpm0_fpd_aclk"
    set data_slave "S_AXI_HP0_FPD"
    set data_clk_pin "saxihp0_fpd_aclk"
    set data_seg "SAXIGP2/HP0_DDR_LOW"
    set ps_props [list \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clock_mhz \
        CONFIG.PSU__USE__M_AXI_GP0 {1} \
        CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
        CONFIG.PSU__PROTECTION__MASTERS {USB1:NonSecure;0|USB0:NonSecure;0|S_AXI_LPD:NA;0|S_AXI_HPC1_FPD:NA;0|S_AXI_HPC0_FPD:NA;1|S_AXI_HP3_FPD:NA;1|S_AXI_HP2_FPD:NA;1|S_AXI_HP1_FPD:NA;0|S_AXI_HP0_FPD:NA;1|S_AXI_ACP:NA;0|S_AXI_ACE:NA;0|SD1:NonSecure;1|SD0:NonSecure;0|SATA1:NonSecure;0|SATA0:NonSecure;0|RPU1:Secure;1|RPU0:Secure;1|QSPI:NonSecure;1|PMU:NA;1|PCIe:NonSecure;0|NAND:NonSecure;0|LDMA:NonSecure;1|GPU:NonSecure;1|GEM3:NonSecure;0|GEM2:NonSecure;0|GEM1:NonSecure;0|GEM0:NonSecure;0|FDMA:NonSecure;1|DP:NonSecure;0|DAP:NA;1|Coresight:NA;1|CSU:NA;1|APU:NA;1} \
    ]
} elseif {$profile eq "kv260_wide4" || $profile eq "kv260_wide4_100"} {
    set part_name "xck26-sfvc784-2LV-c"
    set board_part "xilinx.com:kv260_som:part0:1.4"
    set rtl_target "sdf5_shared"
    if {$profile eq "kv260_wide4_100"} { set clock_mhz 100 } else { set clock_mhz 133.333 }
    set ctrl_base 0x96000000
    set ctrl_master "M_AXI_HPM0_LPD"
    set ctrl_clk_pin "maxihpm0_lpd_aclk"
    set data_slave "S_AXI_HP0_FPD"
    set data_clk_pin "saxihp0_fpd_aclk"
    set data_seg "SAXIGP2/HP0_DDR_LOW"
    set ps_props [list \
        CONFIG.PSU__FPGA_PL0_ENABLE {1} \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $clock_mhz \
        CONFIG.PSU__USE__M_AXI_GP2 {1} \
        CONFIG.PSU__MAXIGP2__DATA_WIDTH {32} \
        CONFIG.PSU__USE__S_AXI_GP2 {1} \
        CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
        CONFIG.PSU__PROTECTION__MASTERS {USB1:NonSecure;0|USB0:NonSecure;0|S_AXI_LPD:NA;0|S_AXI_HPC1_FPD:NA;0|S_AXI_HPC0_FPD:NA;1|S_AXI_HP3_FPD:NA;1|S_AXI_HP2_FPD:NA;1|S_AXI_HP1_FPD:NA;0|S_AXI_HP0_FPD:NA;1|S_AXI_ACP:NA;0|S_AXI_ACE:NA;0|SD1:NonSecure;1|SD0:NonSecure;0|SATA1:NonSecure;0|SATA0:NonSecure;0|RPU1:Secure;1|RPU0:Secure;1|QSPI:NonSecure;1|PMU:NA;1|PCIe:NonSecure;0|NAND:NonSecure;0|LDMA:NonSecure;1|GPU:NonSecure;1|GEM3:NonSecure;0|GEM2:NonSecure;0|GEM1:NonSecure;0|GEM0:NonSecure;0|FDMA:NonSecure;1|DP:NonSecure;0|DAP:NA;1|Coresight:NA;1|CSU:NA;1|APU:NA;1} \
    ]
} else {
    error "unknown profile: $profile"
}

set rtl [file join $proj_root "rtl" "generated" $rtl_target "Sdf5PrefillAxi.v"]
if {[info exists ::env(SDF5_RTL_TARGET)] && $::env(SDF5_RTL_TARGET) ne ""} {
    set rtl_target $::env(SDF5_RTL_TARGET)
    set rtl [file join $proj_root "rtl" "generated" $rtl_target "Sdf5PrefillAxi.v"]
}
if {![file exists $rtl]} { error "missing RTL $rtl" }

set design_name "system"
set project_name "sdf5_prefill_${profile}"
set build_dir [file normalize [file join $proj_root "build_${profile}_${run_tag}"]]
set out_dir [file normalize [file join $proj_root "out" "${profile}_${run_tag}"]]
file mkdir $out_dir
set_param general.maxThreads $jobs

proc note {msg} { puts "sdf5_prefill_BD: $msg" }
proc safe_set_dict {obj dict_values} {
    if {[catch {set_property -dict $dict_values $obj} err]} {
        puts "sdf5_prefill_BD: WARN set_property failed on $obj: $err"
    }
}
proc safe_assign_addr {addr_space seg offset range} {
    if {$addr_space eq "" || $seg eq ""} {
        puts "sdf5_prefill_BD: WARN skip addr offset=$offset range=$range"
        return
    }
    if {[catch {assign_bd_address -offset $offset -range $range -target_address_space $addr_space $seg -force} err]} {
        puts "sdf5_prefill_BD: WARN address assign failed offset=$offset range=$range seg=$seg: $err"
    }
}

create_project -force $project_name $build_dir -part $part_name
if {$board_part ne ""} {
    catch {set_property board_part $board_part [current_project]} board_err
    if {[info exists board_err] && $board_err ne ""} { note "WARN board_part: $board_err" }
}
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
add_files -norecurse $rtl
update_compile_order -fileset sources_1

create_bd_design $design_name
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0]
safe_set_dict $ps $ps_props

set axi_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:* ps_axi_ctrl]
set_property CONFIG.NUM_MI 1 $axi_ctrl
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc_data]
set_property CONFIG.NUM_SI 1 $smc
set_property CONFIG.NUM_MI 1 $smc
set rst_bus [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_bus]
set rst_accel [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst_accel]
safe_set_dict $rst_bus [list CONFIG.C_EXT_RESET_HIGH 0]
safe_set_dict $rst_accel [list CONFIG.C_EXT_RESET_HIGH 0]
set accel [create_bd_cell -type module -reference Sdf5PrefillAxi Sdf5PrefillAxi_0]

set clk_freq [get_property CONFIG.FREQ_HZ [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]]
if {$clk_freq eq ""} { set clk_freq 149998505 }
foreach intf {S_AXI M_AXI} {
    safe_set_dict [get_bd_intf_pins Sdf5PrefillAxi_0/$intf] [list CONFIG.FREQ_HZ $clk_freq CONFIG.CLK_DOMAIN ${design_name}_zynq_ultra_ps_e_0_0_pl_clk0]
}
safe_set_dict [get_bd_pins Sdf5PrefillAxi_0/ACLK] [list \
    CONFIG.FREQ_HZ $clk_freq \
    CONFIG.CLK_DOMAIN ${design_name}_zynq_ultra_ps_e_0_0_pl_clk0 \
    CONFIG.ASSOCIATED_BUSIF {S_AXI:M_AXI} \
    CONFIG.ASSOCIATED_RESET {ARESETN} \
]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/$ctrl_master] [get_bd_intf_pins ps_axi_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ps_axi_ctrl/M00_AXI] [get_bd_intf_pins Sdf5PrefillAxi_0/S_AXI]
connect_bd_intf_net [get_bd_intf_pins Sdf5PrefillAxi_0/M_AXI] [get_bd_intf_pins axi_smc_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc_data/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/$data_slave]

set pl_clk [get_bd_pins zynq_ultra_ps_e_0/pl_clk0]
foreach pin [list \
    [get_bd_pins ps_axi_ctrl/ACLK] \
    [get_bd_pins ps_axi_ctrl/S00_ACLK] \
    [get_bd_pins ps_axi_ctrl/M00_ACLK] \
    [get_bd_pins rst_bus/slowest_sync_clk] \
    [get_bd_pins rst_accel/slowest_sync_clk] \
    [get_bd_pins Sdf5PrefillAxi_0/ACLK] \
    [get_bd_pins zynq_ultra_ps_e_0/$ctrl_clk_pin] \
    [get_bd_pins zynq_ultra_ps_e_0/$data_clk_pin] \
    [get_bd_pins axi_smc_data/aclk] \
] { connect_bd_net $pl_clk $pin }
foreach opt_clk_pin {maxihpm0_lpd_aclk maxihpm0_fpd_aclk} {
    set opt_pin [get_bd_pins -quiet zynq_ultra_ps_e_0/$opt_clk_pin]
    if {$opt_pin ne ""} {
        catch {connect_bd_net $pl_clk $opt_pin} opt_clk_err
        if {[info exists opt_clk_err] && $opt_clk_err ne ""} { note "WARN optional clock $opt_clk_pin: $opt_clk_err" }
    }
}

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_bus/ext_reset_in]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_accel/ext_reset_in]
foreach pin [list \
    [get_bd_pins ps_axi_ctrl/ARESETN] \
    [get_bd_pins ps_axi_ctrl/S00_ARESETN] \
    [get_bd_pins ps_axi_ctrl/M00_ARESETN] \
    [get_bd_pins axi_smc_data/aresetn] \
] { connect_bd_net [get_bd_pins rst_bus/peripheral_aresetn] $pin }
connect_bd_net [get_bd_pins rst_accel/peripheral_aresetn] [get_bd_pins Sdf5PrefillAxi_0/ARESETN]

assign_bd_address
safe_assign_addr [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs Sdf5PrefillAxi_0/S_AXI/reg0] $ctrl_base 0x00001000
safe_assign_addr [get_bd_addr_spaces Sdf5PrefillAxi_0/M_AXI] [get_bd_addr_segs zynq_ultra_ps_e_0/$data_seg] 0x00000000 0x80000000

validate_bd_design
save_bd_design
write_bd_tcl -force [file join $out_dir "system_bd.tcl"]

set wrapper [make_wrapper -files [get_files [file join $build_dir $project_name.srcs sources_1 bd $design_name ${design_name}.bd]] -top]
add_files -norecurse $wrapper
set_property top ${design_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1
generate_target all [get_files [file join $build_dir $project_name.srcs sources_1 bd $design_name ${design_name}.bd]]
export_ip_user_files -of_objects [get_files [file join $build_dir $project_name.srcs sources_1 bd $design_name ${design_name}.bd]] -no_script -sync -force -quiet

note "launch synth_1"
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
note "synth_1 status: $synth_status"
if {[string first "ERROR" $synth_status] >= 0 || [string first "Fail" $synth_status] >= 0} { error "synth failed: $synth_status" }

open_run synth_1 -name synth_1
write_checkpoint -force [file join $out_dir "post_synth.dcp"]
report_utilization -hierarchical -file [file join $out_dir "utilization_synth_hier.rpt"]
report_utilization -file [file join $out_dir "utilization_synth.rpt"]
report_timing_summary -file [file join $out_dir "timing_summary_synth.rpt"]

if {[string match "sdf5_*" $rtl_target]} {
    set lut_count [llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]]
    set dsp_count [llength [get_cells -hierarchical -filter {REF_NAME =~ DSP48*}]]
    set ramb36_count [llength [get_cells -hierarchical -filter {REF_NAME =~ RAMB36*}]]
    set ramb18_count [llength [get_cells -hierarchical -filter {REF_NAME =~ RAMB18*}]]
    set uram_count [llength [get_cells -hierarchical -filter {REF_NAME =~ URAM*}]]
    set ram_prim_count [expr {$ramb36_count + $ramb18_count + $uram_count}]
    set min_lut 25000
    set min_dsp 256
    set min_ram 16
    set fh_gate [open [file join $out_dir "synth_resource_gate.txt"] w]
    puts $fh_gate "profile=$profile"
    puts $fh_gate "rtl_target=$rtl_target"
    puts $fh_gate "lut_count=$lut_count min_lut=$min_lut"
    puts $fh_gate "dsp_count=$dsp_count min_dsp=$min_dsp"
    puts $fh_gate "ramb36_count=$ramb36_count"
    puts $fh_gate "ramb18_count=$ramb18_count"
    puts $fh_gate "uram_count=$uram_count"
    puts $fh_gate "ram_prim_count=$ram_prim_count min_ram=$min_ram"
    close $fh_gate
    note "resource gate: LUT=$lut_count DSP=$dsp_count RAM_PRIM=$ram_prim_count"
    if {$lut_count < $min_lut || $dsp_count < $min_dsp || $ram_prim_count < $min_ram} {
        error "resource gate failed after synth: LUT=$lut_count DSP=$dsp_count RAM_PRIM=$ram_prim_count"
    }
}
close_design

note "launch impl_1 to write_bitstream"
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
set impl_status [get_property STATUS [get_runs impl_1]]
note "impl_1 status: $impl_status"
if {[string first "ERROR" $impl_status] >= 0 || [string first "Fail" $impl_status] >= 0} { error "impl failed: $impl_status" }

open_run impl_1 -name impl_1
write_checkpoint -force [file join $out_dir "post_route.dcp"]
report_utilization -hierarchical -file [file join $out_dir "utilization_routed_hier.rpt"]
report_utilization -file [file join $out_dir "utilization_routed.rpt"]
report_timing_summary -file [file join $out_dir "timing_summary_routed.rpt"]
report_drc -file [file join $out_dir "drc_routed.rpt"]
set run_bit [file join $build_dir "$project_name.runs" impl_1 "${design_name}_wrapper.bit"]
if {[file exists $run_bit]} { file copy -force $run_bit [file join $out_dir "system.bit"] }
write_hw_platform -fixed -include_bit -force -file [file join $out_dir "system.xsa"]

set fh [open [file join $out_dir "build_summary.txt"] w]
puts $fh "profile=$profile"
puts $fh "rtl_target=$rtl_target"
puts $fh "part=$part_name"
puts $fh "vivado=[version -short]"
puts $fh "clock_mhz=$clock_mhz"
puts $fh "control_base=$ctrl_base"
puts $fh "control_master=$ctrl_master"
puts $fh "data_slave=$data_slave"
puts $fh "bit=[file join $out_dir system.bit]"
puts $fh "xsa=[file join $out_dir system.xsa]"
close $fh
note "build complete profile=$profile out=$out_dir"

