set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ".." ".."]]
set repo_root ""
set build_dir  [file normalize [file join $script_dir "build_tpu_pspl_native_iftest_bd"]]
set project_name "pynq_zu_tpu_pspl_native_iftest"
set bd_name "tpu_pspl_native_iftest"
set target_part "xczu5eg-sfvc784-1-e"
set board_part "tul.com.tw:pynqzu:part0:1.1"
set wrapper_file [file normalize [file join $project_root "rtl" "tpu_pspl_iftest_axi_ip.v"]]
set core_file [file normalize [file join $project_root "rtl" "tpu_pspl_iftest_core.v"]]
set matrix_a_base "0xA0000000"
set matrix_b_base "0xA0004000"
set matrix_c_base "0xA0008000"
set matrix_d_base "0xA1000000"
set tpu_ip_vendor "gyq.local"
set tpu_ip_library "user"
set tpu_ip_name "tpu_pspl_iftest_axi_ip"
set tpu_ip_version "1.0"
set build_jobs "auto"
set tool_threads 1

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

proc choose_build_jobs {} {
  set cpu_count 1
  set mem_avail_gib 0

  if {![catch {exec nproc} cpu_out]} {
    set cpu_out [string trim $cpu_out]
    if {[string is integer -strict $cpu_out]} {
      set cpu_count $cpu_out
    }
  }

  if {![catch {exec awk {/MemAvailable:/ {printf "%d", int($2/1024/1024)}} /proc/meminfo} mem_out]} {
    set mem_out [string trim $mem_out]
    if {[string is integer -strict $mem_out]} {
      set mem_avail_gib $mem_out
    }
  }

  set cpu_limit $cpu_count
  if {$cpu_limit < 1} {
    set cpu_limit 1
  }

  if {$mem_avail_gib > 2} {
    set mem_limit [expr {$mem_avail_gib - 2}]
  } else {
    set mem_limit 1
  }
  if {$mem_limit < 1} {
    set mem_limit 1
  }

  set jobs $cpu_limit
  if {$mem_limit < $jobs} {
    set jobs $mem_limit
  }
  if {$jobs > 32} {
    set jobs 32
  }
  if {$jobs < 1} {
    set jobs 1
  }

  puts "Auto-selected build jobs: $jobs (cpu_count=$cpu_count, mem_avail_gib=$mem_avail_gib)"
  return $jobs
}

proc package_pspl_iftest_ip {repo_root wrapper_file core_file build_dir target_part ip_vendor ip_library ip_name ip_version} {
  set rtl_dir [file normalize [file join $repo_root "rtl"]]
  set support_root [file normalize [file join [file dirname $build_dir] "[file tail $build_dir]_support"]]
  set pkg_project_dir [file normalize [file join $support_root "${ip_name}_pkg"]]
  set ip_repo_root [file normalize [file join $support_root "ip_repo"]]
  set ip_root_dir [file normalize [file join $ip_repo_root "${ip_name}_${ip_version}"]]
  set helper_files [list \
    [file normalize [file join $rtl_dir "axi4_lite_slave.v"]] \
    [file normalize [file join $rtl_dir "axi4_full_slave.v"]] \
    [file normalize [file join $rtl_dir "axi4_full_master.v"]] \
  ]

  if {![file isdirectory $rtl_dir]} {
    error "RTL directory not found: $rtl_dir"
  }
  foreach src_file [concat $helper_files [list $core_file $wrapper_file]] {
    if {![file exists $src_file]} {
      error "Required interface-test RTL not found: $src_file"
    }
  }

  file delete -force $pkg_project_dir $ip_root_dir
  file mkdir $ip_repo_root

  create_project "${ip_name}_pkg" $pkg_project_dir -force -part $target_part
  set_property target_language Verilog [current_project]
  add_files -norecurse $helper_files
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
  set_property display_name "TPU PS-PL Native Interface Test AXI IP" $core
  set_property description "AXI self-test wrapper for validating PS direct CSR/data and PL writeback before re-enabling the full TPU datapath." $core
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
if {$argc > 3} {
  set build_jobs [lindex $argv 3]
}
if {$build_jobs eq "auto"} {
  set build_jobs [choose_build_jobs]
} elseif {![string is integer -strict $build_jobs]} {
  error "Build jobs must be an integer or 'auto', got: $build_jobs"
}
if {$build_jobs < 1} {
  set build_jobs 1
}
if {$build_jobs > 32} {
  set build_jobs 32
}

set tool_threads $build_jobs
if {$tool_threads > 8} {
  set tool_threads 8
}

set_param general.maxThreads $tool_threads
set_param synth.maxThreads $tool_threads

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
set blk_mem_gen_vlnv [require_latest_ip_vlnv "xilinx.com:ip:blk_mem_gen:*" "blk_mem_gen"]
set proc_sys_reset_vlnv [require_latest_ip_vlnv "xilinx.com:ip:proc_sys_reset:*" "proc_sys_reset"]
set smartconnect_vlnv [require_latest_ip_vlnv "xilinx.com:ip:smartconnect:*" "smartconnect"]
set tpu_ip_vlnv [require_latest_ip_vlnv "${tpu_ip_vendor}:${tpu_ip_library}:${tpu_ip_name}:*" $tpu_ip_name]
set zynq_ps_vlnv [require_latest_ip_vlnv "xilinx.com:ip:zynq_ultra_ps_e:*" "zynq_ultra_ps_e"]

puts "Using IP VLNVs:"
puts "  axi_bram_ctrl             = $axi_bram_ctrl_vlnv"
puts "  blk_mem_gen               = $blk_mem_gen_vlnv"
puts "  proc_sys_reset            = $proc_sys_reset_vlnv"
puts "  smartconnect              = $smartconnect_vlnv"
puts "  tpu_pspl_native_iftest_ip = $tpu_ip_vlnv"
puts "  zynq_ultra_ps_e           = $zynq_ps_vlnv"

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

create_bd_cell -type ip -vlnv $blk_mem_gen_vlnv blk_mem_gen_0
set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_bd_cells blk_mem_gen_0]

create_bd_cell -type ip -vlnv $proc_sys_reset_vlnv proc_sys_reset_0

create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_ctrl_0
set_property -dict [list \
  CONFIG.ADVANCED_PROPERTIES {__experimental_features__ {disable_low_area_mode 1}} \
  CONFIG.NUM_MI {1} \
  CONFIG.NUM_SI {1} \
] [get_bd_cells smartconnect_ctrl_0]

create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_data_0
set_property -dict [list \
  CONFIG.NUM_MI {2} \
  CONFIG.NUM_SI {1} \
] [get_bd_cells smartconnect_data_0]

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
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP2 {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTA]
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_1/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTB]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_LPD] [get_bd_intf_pins smartconnect_ctrl_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_ctrl_0/M00_AXI] [get_bd_intf_pins tpu_pspl_iftest_0/s_axil]

connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smartconnect_data_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_data_0/M00_AXI] [get_bd_intf_pins tpu_pspl_iftest_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins smartconnect_data_0/M01_AXI] [get_bd_intf_pins axi_bram_ctrl_0/S_AXI]

connect_bd_intf_net [get_bd_intf_pins tpu_pspl_iftest_0/m_axi] [get_bd_intf_pins axi_bram_ctrl_1/S_AXI]

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_ctrl_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_data_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_bram_ctrl_1/s_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_lpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins tpu_pspl_iftest_0/clk]

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins proc_sys_reset_0/ext_reset_in]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_ctrl_0/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_data_0/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins axi_bram_ctrl_1/s_axi_aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins tpu_pspl_iftest_0/rst_n]

assign_bd_address
assign_bd_address -force \
  -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
  -offset 0xA0000000 \
  -range 0x00010000 \
  [get_bd_addr_segs tpu_pspl_iftest_0/S_AXI/reg0]
assign_bd_address -force \
  -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
  -offset 0xA1000000 \
  -range 0x00002000 \
  [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0]
assign_bd_address -force \
  -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] \
  -offset 0x80000000 \
  -range 0x00001000 \
  [get_bd_addr_segs tpu_pspl_iftest_0/S_AXIL/reg0]
assign_bd_address -force \
  -target_address_space [get_bd_addr_spaces tpu_pspl_iftest_0/M_AXI] \
  -offset 0xA1000000 \
  -range 0x00002000 \
  [get_bd_addr_segs axi_bram_ctrl_1/S_AXI/Mem0]

validate_bd_design
save_bd_design

set bd_file [get_files [file join $build_dir $project_name.srcs sources_1 bd $bd_name $bd_name.bd]]
set_property synth_checkpoint_mode None $bd_file
generate_target all $bd_file

make_wrapper -files $bd_file -top
set wrapper_out [file join $build_dir $project_name.gen sources_1 bd $bd_name hdl ${bd_name}_wrapper.v]
add_files -norecurse $wrapper_out
update_compile_order -fileset sources_1
set_property top ${bd_name}_wrapper [get_filesets sources_1]
set_property top_auto_set 0 [get_filesets sources_1]

puts "Created Vivado project at: $build_dir"
puts "Block design name: $bd_name"
puts "Top module: ${bd_name}_wrapper"
puts "TPU RTL repo root: $repo_root"
puts "TPU IP repo root: $ip_repo_root"
puts "Address plan:"
puts "  PS -> IFTEST DATA  : 0xA0000000"
puts "  PS -> BRAM         : 0xA1000000"
puts "  PS -> IFTEST CSR   : 0x80000000"
puts "  IFTEST A/B/C base  : $matrix_a_base / $matrix_b_base / $matrix_c_base"
puts "  IFTEST D base      : $matrix_d_base"
