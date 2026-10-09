set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ".." ".."]]
set repo_root ""
set build_dir  [file normalize [file join $script_dir "build_tpu_pspl_iftest_bd"]]
set project_name "pynq_zu_tpu_pspl_iftest"
set bd_name "tpu_pspl_iftest"
set target_part "xczu5eg-sfvc784-1-e"
set board_part "tul.com.tw:pynqzu:part0:1.1"
set wrapper_file [file normalize [file join $project_root "rtl" "tpu_pspl_iftest_axi_ip.v"]]
set core_file [file normalize [file join $project_root "rtl" "tpu_pspl_iftest_core.v"]]
set matrix_a_base "0x10000000"
set matrix_b_base "0x10004000"
set matrix_c_base "0x10008000"
set matrix_d_base "0xA0000000"
set tpu_ip_vendor "gyq.local"
set tpu_ip_library "user"
set tpu_ip_name "tpu_pspl_iftest_axi_ip"
set tpu_ip_version "1.0"

# Force Vivado into single-threaded behavior for this shared server flow.
set_param general.maxThreads 1
set_param synth.maxThreads 1

foreach candidate [list \
  [file join $project_root "third_party" "tpu_vanilla-fp-32"] \
] {
  set candidate_norm [file normalize $candidate]
  if {[file isdirectory $candidate_norm]} {
    set repo_root $candidate_norm
    break
  }
}

proc scan_verilog_files {dir file_list_var} {
  upvar $file_list_var file_list

  if {[file tail [file normalize $dir]] eq "system_pro"} {
    return
  }

  foreach file [lsort [glob -nocomplain -directory $dir "*.v"]] {
    if {[file tail $file] eq "tpu_top_matmul.v"} {
      continue
    }
    lappend file_list [file normalize $file]
  }

  foreach subdir [lsort [glob -nocomplain -directory $dir "*"]] {
    if {[file isdirectory $subdir]} {
      scan_verilog_files $subdir file_list
    }
  }
}

proc require_latest_ip_vlnv {pattern pretty_name} {
  set ipdefs [lsort [get_ipdefs -all $pattern]]
  if {[llength $ipdefs] == 0} {
    error "Required IP '$pretty_name' not found for pattern '$pattern'."
  }
  return [lindex $ipdefs end]
}

proc package_pspl_iftest_ip {repo_root wrapper_file core_file build_dir target_part ip_vendor ip_library ip_name ip_version} {
  set rtl_dir [file normalize [file join $repo_root "rtl"]]
  set support_root [file normalize [file join [file dirname $build_dir] "[file tail $build_dir]_support"]]
  set pkg_project_dir [file normalize [file join $support_root "${ip_name}_pkg"]]
  set ip_repo_root [file normalize [file join $support_root "ip_repo"]]
  set ip_root_dir [file normalize [file join $ip_repo_root "${ip_name}_${ip_version}"]]
  set rtl_files [list]

  if {![file isdirectory $rtl_dir]} {
    error "RTL directory not found: $rtl_dir"
  }
  foreach local_file [list $wrapper_file $core_file] {
    if {![file exists $local_file]} {
      error "Local interface-test RTL not found: $local_file"
    }
  }

  scan_verilog_files $rtl_dir rtl_files
  if {[llength $rtl_files] == 0} {
    error "No Verilog files found under: $rtl_dir"
  }

  file delete -force $pkg_project_dir $ip_root_dir
  file mkdir $ip_repo_root

  create_project "${ip_name}_pkg" $pkg_project_dir -force -part $target_part
  set_property target_language Verilog [current_project]
  add_files $rtl_files
  add_files -norecurse $core_file
  add_files -norecurse $wrapper_file
  update_compile_order -fileset sources_1
  set_property top $ip_name [get_filesets sources_1]

  ipx::package_project -root_dir $ip_root_dir -vendor $ip_vendor -library $ip_library -taxonomy /UserIP -import_files -force
  set core [ipx::current_core]
  set_property vendor $ip_vendor $core
  set_property library $ip_library $core
  set_property name $ip_name $core
  set_property version $ip_version $core
  set_property display_name "TPU PS-PL Interface Test AXI IP" $core
  set_property description "AXI self-test wrapper for validating PS-PL CSR/CDMA/BRAM interfaces before enabling the full TPU datapath." $core
  if {[llength [ipx::get_memory_maps -quiet S_AXI -of_objects $core]] > 0} {
    set s_axi_mmap [ipx::get_memory_maps S_AXI -of_objects $core]
    if {[llength [ipx::get_address_blocks -quiet reg0 -of_objects $s_axi_mmap]] > 0} {
      set_property USAGE memory [ipx::get_address_blocks reg0 -of_objects $s_axi_mmap]
    }
  }
  foreach busif {S_AXI S_AXIL M_AXI} {
    if {[llength [ipx::get_bus_interfaces -quiet $busif -of_objects $core]] > 0} {
      ipx::associate_bus_interfaces -busif $busif -clock CLK $core
    }
  }
  ipx::save_core $core
  close_project

  return $ip_repo_root
}

if {$argc > 0} {
  set repo_root [file normalize [lindex $argv 0]]
}
if {$argc > 1} {
  set build_dir [file normalize [lindex $argv 1]]
}
if {$argc > 2} {
  set matrix_d_base [lindex $argv 2]
}

if {$repo_root eq ""} {
  error "TPU RTL repo root not found. Pass it explicitly as the first tclarg."
}

file mkdir $build_dir
set ip_repo_root [package_pspl_iftest_ip $repo_root $wrapper_file $core_file $build_dir $target_part $tpu_ip_vendor $tpu_ip_library $tpu_ip_name $tpu_ip_version]

create_project $project_name $build_dir -force -part $target_part
set_property target_language Verilog [current_project]

if {[llength [get_board_parts -quiet $board_part]] > 0} {
  set_property BOARD_PART $board_part [current_project]
  puts "Using board part: $board_part"
} else {
  puts "Board part '$board_part' not found; continuing with target part '$target_part' only."
}

set_property ip_repo_paths [list $ip_repo_root] [current_project]
update_ip_catalog -rebuild

create_bd_design $bd_name

set axi_bram_ctrl_vlnv [require_latest_ip_vlnv "xilinx.com:ip:axi_bram_ctrl:*" "axi_bram_ctrl"]
set axi_cdma_vlnv [require_latest_ip_vlnv "xilinx.com:ip:axi_cdma:*" "axi_cdma"]
set blk_mem_gen_vlnv [require_latest_ip_vlnv "xilinx.com:ip:blk_mem_gen:*" "blk_mem_gen"]
set proc_sys_reset_vlnv [require_latest_ip_vlnv "xilinx.com:ip:proc_sys_reset:*" "proc_sys_reset"]
set smartconnect_vlnv [require_latest_ip_vlnv "xilinx.com:ip:smartconnect:*" "smartconnect"]
set tpu_ip_vlnv [require_latest_ip_vlnv "${tpu_ip_vendor}:${tpu_ip_library}:${tpu_ip_name}:*" $tpu_ip_name]
set zynq_ps_vlnv [require_latest_ip_vlnv "xilinx.com:ip:zynq_ultra_ps_e:*" "zynq_ultra_ps_e"]

puts "Using IP VLNVs:"
puts "  axi_bram_ctrl        = $axi_bram_ctrl_vlnv"
puts "  axi_cdma             = $axi_cdma_vlnv"
puts "  blk_mem_gen          = $blk_mem_gen_vlnv"
puts "  proc_sys_reset       = $proc_sys_reset_vlnv"
puts "  smartconnect         = $smartconnect_vlnv"
puts "  tpu_pspl_iftest_ip   = $tpu_ip_vlnv"
puts "  zynq_ultra_ps_e      = $zynq_ps_vlnv"

create_bd_cell -type ip -vlnv $axi_bram_ctrl_vlnv axi_bram_ctrl_0
set_property -dict [list \
  CONFIG.DATA_WIDTH {64} \
  CONFIG.SINGLE_PORT_BRAM {1} \
] [get_bd_cells axi_bram_ctrl_0]

create_bd_cell -type ip -vlnv $axi_bram_ctrl_vlnv axi_bram_ctrl_1
set_property -dict [list \
  CONFIG.DATA_WIDTH {64} \
  CONFIG.SINGLE_PORT_BRAM {1} \
] [get_bd_cells axi_bram_ctrl_1]

create_bd_cell -type ip -vlnv $axi_cdma_vlnv axi_cdma_0
set_property -dict [list \
  CONFIG.C_INCLUDE_SG {0} \
  CONFIG.C_M_AXI_DATA_WIDTH {64} \
] [get_bd_cells axi_cdma_0]

create_bd_cell -type ip -vlnv $blk_mem_gen_vlnv blk_mem_gen_0
set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_bd_cells blk_mem_gen_0]

create_bd_cell -type ip -vlnv $proc_sys_reset_vlnv rst_ps8_0_100M

create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
set_property -dict [list \
  CONFIG.NUM_MI {3} \
  CONFIG.NUM_SI {2} \
] [get_bd_cells smartconnect_0]

create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_1
set_property -dict [list \
  CONFIG.ADVANCED_PROPERTIES {__experimental_features__ {disable_low_area_mode 1}} \
  CONFIG.NUM_MI {2} \
  CONFIG.NUM_SI {1} \
] [get_bd_cells smartconnect_1]

create_bd_cell -type ip -vlnv $tpu_ip_vlnv tpu_pspl_iftest_0
set_property -dict [list \
  CONFIG.MATRIX_A_BASE_ADDR $matrix_a_base \
  CONFIG.MATRIX_B_BASE_ADDR $matrix_b_base \
  CONFIG.MATRIX_C_BASE_ADDR $matrix_c_base \
  CONFIG.MATRIX_D_BASE_ADDR $matrix_d_base \
] [get_bd_cells tpu_pspl_iftest_0]

create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ultra_ps_e_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "0"} [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
  CONFIG.PSU__USE__IRQ0 {1} \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {1} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTA]
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_1/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTB]
connect_bd_intf_net [get_bd_intf_pins axi_cdma_0/M_AXI] [get_bd_intf_pins smartconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/S_AXI] [get_bd_intf_pins smartconnect_0/M01_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M02_AXI] [get_bd_intf_pins tpu_pspl_iftest_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_cdma_0/S_AXI_LITE] [get_bd_intf_pins smartconnect_1/M00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_1/M01_AXI] [get_bd_intf_pins tpu_pspl_iftest_0/s_axil]
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_1/S_AXI] [get_bd_intf_pins tpu_pspl_iftest_0/m_axi]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/S00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD]
connect_bd_intf_net [get_bd_intf_pins smartconnect_1/S00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM1_FPD]

connect_bd_net [get_bd_pins axi_cdma_0/cdma_introut] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins axi_bram_ctrl_1/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins axi_cdma_0/s_axi_lite_aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins smartconnect_0/aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins smartconnect_1/aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins tpu_pspl_iftest_0/rst_n]

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_bram_ctrl_1/s_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_cdma_0/m_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_cdma_0/s_axi_lite_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins rst_ps8_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_1/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins tpu_pspl_iftest_0/clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm1_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_ps8_0_100M/ext_reset_in]

assign_bd_address
assign_bd_address -offset 0xA0000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
assign_bd_address -offset 0x10000000 -range 0x10000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs tpu_pspl_iftest_0/s_axi/reg0] -force
assign_bd_address -offset 0x40000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force
assign_bd_address -offset 0xA0000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces tpu_pspl_iftest_0/m_axi] [get_bd_addr_segs axi_bram_ctrl_1/S_AXI/Mem0] -force
assign_bd_address -offset 0xA0000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
assign_bd_address -offset 0xB0010000 -range 0x00010000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs axi_cdma_0/S_AXI_LITE/Reg] -force
assign_bd_address -offset 0xB0000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs tpu_pspl_iftest_0/s_axil/reg0] -force

foreach seg [list \
  /zynq_ultra_ps_e_0/Data/SEG_tpu_pspl_iftest_0_reg0 \
  /zynq_ultra_ps_e_0/Data/SEG_zynq_ultra_ps_e_0_HP0_DDR_LOW \
  /zynq_ultra_ps_e_0/Data/SEG_zynq_ultra_ps_e_0_HP0_LPS_OCM \
] {
  if {[llength [get_bd_addr_segs -quiet $seg]] > 0} {
    exclude_bd_addr_seg $seg
  }
}

validate_bd_design
save_bd_design

set bd_file [get_files [file join $build_dir $project_name.srcs sources_1 bd $bd_name $bd_name.bd]]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file

make_wrapper -files $bd_file -top
set wrapper_file [file join $build_dir $project_name.gen sources_1 bd $bd_name hdl ${bd_name}_wrapper.v]
add_files -norecurse $wrapper_file
update_compile_order -fileset sources_1
set_property top ${bd_name}_wrapper [get_filesets sources_1]
set_property top_auto_set 0 [get_filesets sources_1]

puts "Created Vivado project at: $build_dir"
puts "Block design name: $bd_name"
puts "Top module: ${bd_name}_wrapper"
puts "TPU RTL repo root: $repo_root"
puts "TPU IP repo root: $ip_repo_root"
puts "Interface-test MATRIX_D_BASE_ADDR: $matrix_d_base"
puts "Address plan:"
puts "  PS -> BRAM               : 0xA0000000"
puts "  PS -> IFTEST CSR         : 0xB0000000"
puts "  PS -> AXI CDMA regs      : 0xB0010000"
puts "  CDMA -> IFTEST input win : 0x10000000"
puts "  IFTEST A/B/C decode base : $matrix_a_base / $matrix_b_base / $matrix_c_base"
