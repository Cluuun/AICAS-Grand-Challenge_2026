set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ".." ".."]]
set repo_root ""
set build_dir  [file normalize [file join $script_dir "build_tpu_soc_bd"]]
set project_name "pynq_zu_tpu_soc"
set bd_name "tpu_soc"
set target_part "xczu5eg-sfvc784-1-e"
set board_part "tul.com.tw:pynqzu:part0:1.1"
set wrapper_file [file normalize [file join $project_root "rtl" "tpu_top_axi_ip.v"]]
if {![file exists $wrapper_file]} {
  set wrapper_file [file normalize [file join $script_dir "tpu_top_axi_ip.v"]]
}
set matrix_a_base "0x10000000"
set matrix_b_base "0x10004000"
set matrix_c_base "0x10008000"
set matrix_d_base "0xA0000000"
set input_port_name "HP0"
set input_ps_intf "S_AXI_HP0_FPD"
set input_ddr_addr_space "SAXIGP2/HP0_DDR_LOW"
set input_ps_aclk_pin "saxihp0_fpd_aclk"
set input_ddr_base "0x40000000"
set input_ddr_range "0x40000000"
set input_ddr_high_base ""
set input_ddr_high_range ""
set dual_input_port_name "HP1"
set dual_input_ps_intf "S_AXI_HP1_FPD"
set dual_input_ddr_addr_space "SAXIGP3/HP1_DDR_LOW"
set dual_input_ps_aclk_pin "saxihp1_fpd_aclk"
set quad_input2_port_name "HPC0"
set quad_input2_ps_intf "S_AXI_HPC0_FPD"
set quad_input2_ddr_addr_space "SAXIGP0/HPC0_DDR_LOW"
set quad_input2_ps_aclk_pin "saxihpc0_fpd_aclk"
set quad_input3_port_name "HP3"
set quad_input3_ps_intf "S_AXI_HP3_FPD"
set quad_input3_ddr_addr_space "SAXIGP5/HP3_DDR_LOW"
set quad_input3_ps_aclk_pin "saxihp3_fpd_aclk"
set output_port_name "HPC1"
set output_ps_intf "S_AXI_HPC1_FPD"
set output_ddr_addr_space "SAXIGP1/HPC1_DDR_LOW"
set output_ps_aclk_pin "saxihpc1_fpd_aclk"
set output_ddr_base "0x40000000"
set output_ddr_range "0x40000000"
set output_ddr_high_base ""
set output_ddr_high_range ""
set enable_ps_s_axi_gp3 0
set enable_ps_s_axi_gp4 0
set enable_ps_s_axi_gp5 0
set enable_system_ila 0
set system_ila_depth 2048
set system_ila_target "OUTPUT"
set enable_prevd_ila 0
set prevd_ila_depth 1024
set tpu_ram_addr_width ""
set tpu_ram_c_addr_width ""
set tpu_ram_d_addr_width ""
set tpu_ram_a_local_addr_width ""
set tpu_ram_b_local_addr_width ""
set tpu_ram_d_local_addr_width ""
set tpu_fifo_depth ""
set tpu_pe_size 8
set tpu_int8_dot_factor 4
set tpu_op_tile_n 128
set tpu_op_tile_k 128
set ab_dual_input 0
set cmat_d_full_word_pack 0
set cmat_streaming_c_adder 0
set tpu_axi_data_width 64
set tpu_ram_data_width 64
set cdma_include_sg 0
set cdma_max_burst_len 256
set tpu_ip_vendor "gyq.local"
set tpu_ip_library "user"
set tpu_ip_name "tpu_top_axi_ip"
set tpu_ip_version "1.0"
set pl_clk0_mhz "100"

if {[info exists ::env(TPU_SOC_PROJECT_NAME)]} {
  set requested_project_name [string trim $::env(TPU_SOC_PROJECT_NAME)]
  if {$requested_project_name ne ""} {
    set project_name $requested_project_name
  }
}

if {[info exists ::env(TPU_SOC_BD_NAME)]} {
  set requested_bd_name [string trim $::env(TPU_SOC_BD_NAME)]
  if {$requested_bd_name ne ""} {
    set bd_name $requested_bd_name
  }
}

if {[info exists ::env(TPU_SOC_PL_CLK0_MHZ)]} {
  set requested_pl_clk0_mhz [string trim $::env(TPU_SOC_PL_CLK0_MHZ)]
  if {$requested_pl_clk0_mhz ne ""} {
    if {![string is double -strict $requested_pl_clk0_mhz] || $requested_pl_clk0_mhz <= 0} {
      error "TPU_SOC_PL_CLK0_MHZ must be a positive number, got '$requested_pl_clk0_mhz'"
    }
    set pl_clk0_mhz $requested_pl_clk0_mhz
  }
}

if {[info exists ::env(TPU_SOC_TARGET_PART)]} {
  set requested_target_part [string trim $::env(TPU_SOC_TARGET_PART)]
  if {$requested_target_part ne ""} {
    set target_part $requested_target_part
  }
}

if {[info exists ::env(TPU_SOC_BOARD_PART)]} {
  set requested_board_part [string trim $::env(TPU_SOC_BOARD_PART)]
  if {$requested_board_part ne ""} {
    set board_part $requested_board_part
  }
}

if {[info exists ::env(TPU_SOC_OUTPUT_PORT)]} {
  set requested_output_port [string toupper [string trim $::env(TPU_SOC_OUTPUT_PORT)]]
  if {$requested_output_port eq "HPC0"} {
    set output_port_name "HPC0"
    set output_ps_intf "S_AXI_HPC0_FPD"
    set output_ddr_addr_space "SAXIGP0/HPC0_DDR_LOW"
    set output_ps_aclk_pin "saxihpc0_fpd_aclk"
  } elseif {$requested_output_port eq "HPC1"} {
    set output_port_name "HPC1"
    set output_ps_intf "S_AXI_HPC1_FPD"
    set output_ddr_addr_space "SAXIGP1/HPC1_DDR_LOW"
    set output_ps_aclk_pin "saxihpc1_fpd_aclk"
  } elseif {$requested_output_port eq "HP0"} {
    set output_port_name "HP0"
    set output_ps_intf "S_AXI_HP0_FPD"
    set output_ddr_addr_space "SAXIGP2/HP0_DDR_LOW"
    set output_ps_aclk_pin "saxihp0_fpd_aclk"
  } elseif {$requested_output_port eq "HP1"} {
    set output_port_name "HP1"
    set output_ps_intf "S_AXI_HP1_FPD"
    set output_ddr_addr_space "SAXIGP3/HP1_DDR_LOW"
    set output_ps_aclk_pin "saxihp1_fpd_aclk"
    set enable_ps_s_axi_gp3 1
  } else {
    error "Unsupported TPU_SOC_OUTPUT_PORT '$requested_output_port' (expected HPC0, HPC1, HP0, or HP1)"
  }
}

if {[info exists ::env(TPU_SOC_INPUT_PORT)]} {
  set requested_input_port [string toupper [string trim $::env(TPU_SOC_INPUT_PORT)]]
  if {$requested_input_port eq "HP0"} {
    set input_port_name "HP0"
    set input_ps_intf "S_AXI_HP0_FPD"
    set input_ddr_addr_space "SAXIGP2/HP0_DDR_LOW"
    set input_ps_aclk_pin "saxihp0_fpd_aclk"
  } elseif {$requested_input_port eq "HP1"} {
    set input_port_name "HP1"
    set input_ps_intf "S_AXI_HP1_FPD"
    set input_ddr_addr_space "SAXIGP3/HP1_DDR_LOW"
    set input_ps_aclk_pin "saxihp1_fpd_aclk"
    set enable_ps_s_axi_gp3 1
  } else {
    error "Unsupported TPU_SOC_INPUT_PORT '$requested_input_port' (expected HP0 or HP1)"
  }
}

if {[info exists ::env(TPU_SOC_AB_DUAL_INPUT)]} {
  set requested_ab_dual_input [string tolower [string trim $::env(TPU_SOC_AB_DUAL_INPUT)]]
  if {$requested_ab_dual_input eq "" ||
      $requested_ab_dual_input eq "0" ||
      $requested_ab_dual_input eq "false" ||
      $requested_ab_dual_input eq "no"} {
    set ab_dual_input 0
  } elseif {$requested_ab_dual_input eq "1" ||
            $requested_ab_dual_input eq "true" ||
            $requested_ab_dual_input eq "yes"} {
    set ab_dual_input 1
  } elseif {$requested_ab_dual_input eq "2"} {
    set ab_dual_input 2
  } else {
    error "TPU_SOC_AB_DUAL_INPUT must be 0, 1, or 2, got '$requested_ab_dual_input'"
  }
}

if {[info exists ::env(TPU_SOC_CMAT_D_FULL_WORD_PACK)]} {
  set requested_cmat_d_full_word_pack [string tolower [string trim $::env(TPU_SOC_CMAT_D_FULL_WORD_PACK)]]
  if {$requested_cmat_d_full_word_pack ne "" &&
      $requested_cmat_d_full_word_pack ne "0" &&
      $requested_cmat_d_full_word_pack ne "false" &&
      $requested_cmat_d_full_word_pack ne "no"} {
    set cmat_d_full_word_pack 1
  }
}

if {[info exists ::env(TPU_SOC_CMAT_STREAMING_C_ADDER)]} {
  set requested_cmat_streaming_c_adder [string tolower [string trim $::env(TPU_SOC_CMAT_STREAMING_C_ADDER)]]
  if {$requested_cmat_streaming_c_adder ne "" &&
      $requested_cmat_streaming_c_adder ne "0" &&
      $requested_cmat_streaming_c_adder ne "false" &&
      $requested_cmat_streaming_c_adder ne "no"} {
    set cmat_streaming_c_adder 1
  }
}

if {$ab_dual_input} {
  if {$input_port_name eq "HP0"} {
    set dual_input_port_name "HP1"
    set dual_input_ps_intf "S_AXI_HP1_FPD"
    set dual_input_ddr_addr_space "SAXIGP3/HP1_DDR_LOW"
    set dual_input_ps_aclk_pin "saxihp1_fpd_aclk"
    set enable_ps_s_axi_gp3 1
  } elseif {$input_port_name eq "HP1"} {
    set dual_input_port_name "HP0"
    set dual_input_ps_intf "S_AXI_HP0_FPD"
    set dual_input_ddr_addr_space "SAXIGP2/HP0_DDR_LOW"
    set dual_input_ps_aclk_pin "saxihp0_fpd_aclk"
  } else {
    error "TPU_SOC_AB_DUAL_INPUT requires HP0/HP1 input port, got '$input_port_name'"
  }
  if {$ab_dual_input >= 2} {
    set enable_ps_s_axi_gp4 0
    set enable_ps_s_axi_gp5 1
  }
}

if {[info exists ::env(TPU_SOC_INPUT_DDR_BASE)]} {
  set requested_input_ddr_base [string trim $::env(TPU_SOC_INPUT_DDR_BASE)]
  if {$requested_input_ddr_base ne ""} {
    set input_ddr_base $requested_input_ddr_base
  }
}

if {[info exists ::env(TPU_SOC_INPUT_DDR_RANGE)]} {
  set requested_input_ddr_range [string trim $::env(TPU_SOC_INPUT_DDR_RANGE)]
  if {$requested_input_ddr_range ne ""} {
    set input_ddr_range $requested_input_ddr_range
  }
}

if {[info exists ::env(TPU_SOC_OUTPUT_DDR_BASE)]} {
  set requested_output_ddr_base [string trim $::env(TPU_SOC_OUTPUT_DDR_BASE)]
  if {$requested_output_ddr_base ne ""} {
    set output_ddr_base $requested_output_ddr_base
  }
}

if {[info exists ::env(TPU_SOC_OUTPUT_DDR_RANGE)]} {
  set requested_output_ddr_range [string trim $::env(TPU_SOC_OUTPUT_DDR_RANGE)]
  if {$requested_output_ddr_range ne ""} {
    set output_ddr_range $requested_output_ddr_range
  }
}

if {[info exists ::env(TPU_SOC_INPUT_DDR_HIGH_BASE)]} {
  set requested_input_ddr_high_base [string trim $::env(TPU_SOC_INPUT_DDR_HIGH_BASE)]
  if {$requested_input_ddr_high_base ne ""} {
    set input_ddr_high_base $requested_input_ddr_high_base
  }
}

if {[info exists ::env(TPU_SOC_INPUT_DDR_HIGH_RANGE)]} {
  set requested_input_ddr_high_range [string trim $::env(TPU_SOC_INPUT_DDR_HIGH_RANGE)]
  if {$requested_input_ddr_high_range ne ""} {
    set input_ddr_high_range $requested_input_ddr_high_range
  }
}

if {[info exists ::env(TPU_SOC_OUTPUT_DDR_HIGH_BASE)]} {
  set requested_output_ddr_high_base [string trim $::env(TPU_SOC_OUTPUT_DDR_HIGH_BASE)]
  if {$requested_output_ddr_high_base ne ""} {
    set output_ddr_high_base $requested_output_ddr_high_base
  }
}

if {[info exists ::env(TPU_SOC_OUTPUT_DDR_HIGH_RANGE)]} {
  set requested_output_ddr_high_range [string trim $::env(TPU_SOC_OUTPUT_DDR_HIGH_RANGE)]
  if {$requested_output_ddr_high_range ne ""} {
    set output_ddr_high_range $requested_output_ddr_high_range
  }
}

if {$input_ps_intf eq $output_ps_intf} {
  error "Input port '$input_port_name' and output port '$output_port_name' cannot share the same PS slave interface ($input_ps_intf)."
}
if {$ab_dual_input && $dual_input_ps_intf eq $output_ps_intf} {
  error "Dual input port '$dual_input_port_name' and output port '$output_port_name' cannot share the same PS slave interface ($dual_input_ps_intf)."
}
if {$ab_dual_input >= 2 && $quad_input2_ps_intf eq $output_ps_intf} {
  error "Quad input port '$quad_input2_port_name' and output port '$output_port_name' cannot share the same PS slave interface ($quad_input2_ps_intf)."
}
if {$ab_dual_input >= 2 && $quad_input3_ps_intf eq $output_ps_intf} {
  error "Quad input port '$quad_input3_port_name' and output port '$output_port_name' cannot share the same PS slave interface ($quad_input3_ps_intf)."
}

if {[info exists ::env(TPU_SOC_ENABLE_SYSTEM_ILA)]} {
  set requested_system_ila [string trim $::env(TPU_SOC_ENABLE_SYSTEM_ILA)]
  if {$requested_system_ila ne "" && $requested_system_ila ne "0"} {
    set enable_system_ila 1
  }
}

if {[info exists ::env(TPU_SOC_SYSTEM_ILA_DEPTH)]} {
  set requested_depth [string trim $::env(TPU_SOC_SYSTEM_ILA_DEPTH)]
  if {$requested_depth ne ""} {
    if {![string is integer -strict $requested_depth] || $requested_depth < 1024} {
      error "TPU_SOC_SYSTEM_ILA_DEPTH must be an integer >= 1024, got '$requested_depth'"
    }
    set system_ila_depth $requested_depth
  }
}

if {[info exists ::env(TPU_SOC_SYSTEM_ILA_TARGET)]} {
  set requested_ila_target [string toupper [string trim $::env(TPU_SOC_SYSTEM_ILA_TARGET)]]
  if {$requested_ila_target eq "" || $requested_ila_target eq "OUTPUT"} {
    set system_ila_target "OUTPUT"
  } elseif {$requested_ila_target eq "INPUT" || $requested_ila_target eq "INPUT_DATA" || $requested_ila_target eq "CDMA_TPU"} {
    set system_ila_target "INPUT"
  } else {
    error "Unsupported TPU_SOC_SYSTEM_ILA_TARGET '$requested_ila_target' (expected OUTPUT or INPUT)"
  }
}

if {[info exists ::env(TPU_SOC_ENABLE_PREVD_ILA)]} {
  set requested_prevd_ila [string trim $::env(TPU_SOC_ENABLE_PREVD_ILA)]
  if {$requested_prevd_ila ne "" && $requested_prevd_ila ne "0"} {
    set enable_prevd_ila 1
  }
}

if {[info exists ::env(TPU_SOC_PREVD_ILA_DEPTH)]} {
  set requested_prevd_ila_depth [string trim $::env(TPU_SOC_PREVD_ILA_DEPTH)]
  if {$requested_prevd_ila_depth ne ""} {
    if {![string is integer -strict $requested_prevd_ila_depth] || $requested_prevd_ila_depth < 1024} {
      error "TPU_SOC_PREVD_ILA_DEPTH must be an integer >= 1024, got '$requested_prevd_ila_depth'"
    }
    set prevd_ila_depth $requested_prevd_ila_depth
  }
}

if {[info exists ::env(TPU_SOC_RAM_ADDR_WIDTH)]} {
  set requested_ram_addr_width [string trim $::env(TPU_SOC_RAM_ADDR_WIDTH)]
  if {$requested_ram_addr_width ne ""} {
    if {![string is integer -strict $requested_ram_addr_width] || $requested_ram_addr_width < 1} {
      error "TPU_SOC_RAM_ADDR_WIDTH must be a positive integer, got '$requested_ram_addr_width'"
    }
    set tpu_ram_addr_width $requested_ram_addr_width
  }
}

if {[info exists ::env(TPU_SOC_RAM_C_ADDR_WIDTH)]} {
  set requested_ram_c_addr_width [string trim $::env(TPU_SOC_RAM_C_ADDR_WIDTH)]
  if {$requested_ram_c_addr_width ne ""} {
    if {![string is integer -strict $requested_ram_c_addr_width] || $requested_ram_c_addr_width < 1} {
      error "TPU_SOC_RAM_C_ADDR_WIDTH must be a positive integer, got '$requested_ram_c_addr_width'"
    }
    set tpu_ram_c_addr_width $requested_ram_c_addr_width
  }
}

if {[info exists ::env(TPU_SOC_RAM_D_ADDR_WIDTH)]} {
  set requested_ram_d_addr_width [string trim $::env(TPU_SOC_RAM_D_ADDR_WIDTH)]
  if {$requested_ram_d_addr_width ne ""} {
    if {![string is integer -strict $requested_ram_d_addr_width] || $requested_ram_d_addr_width < 1} {
      error "TPU_SOC_RAM_D_ADDR_WIDTH must be a positive integer, got '$requested_ram_d_addr_width'"
    }
    set tpu_ram_d_addr_width $requested_ram_d_addr_width
  }
}

if {[info exists ::env(TPU_SOC_RAM_A_LOCAL_ADDR_WIDTH)]} {
  set requested_ram_a_local_addr_width [string trim $::env(TPU_SOC_RAM_A_LOCAL_ADDR_WIDTH)]
  if {$requested_ram_a_local_addr_width ne ""} {
    if {![string is integer -strict $requested_ram_a_local_addr_width] || $requested_ram_a_local_addr_width < 1} {
      error "TPU_SOC_RAM_A_LOCAL_ADDR_WIDTH must be a positive integer, got '$requested_ram_a_local_addr_width'"
    }
    set tpu_ram_a_local_addr_width $requested_ram_a_local_addr_width
  }
}

if {[info exists ::env(TPU_SOC_RAM_B_LOCAL_ADDR_WIDTH)]} {
  set requested_ram_b_local_addr_width [string trim $::env(TPU_SOC_RAM_B_LOCAL_ADDR_WIDTH)]
  if {$requested_ram_b_local_addr_width ne ""} {
    if {![string is integer -strict $requested_ram_b_local_addr_width] || $requested_ram_b_local_addr_width < 1} {
      error "TPU_SOC_RAM_B_LOCAL_ADDR_WIDTH must be a positive integer, got '$requested_ram_b_local_addr_width'"
    }
    set tpu_ram_b_local_addr_width $requested_ram_b_local_addr_width
  }
}

if {[info exists ::env(TPU_SOC_RAM_D_LOCAL_ADDR_WIDTH)]} {
  set requested_ram_d_local_addr_width [string trim $::env(TPU_SOC_RAM_D_LOCAL_ADDR_WIDTH)]
  if {$requested_ram_d_local_addr_width ne ""} {
    if {![string is integer -strict $requested_ram_d_local_addr_width] || $requested_ram_d_local_addr_width < 1} {
      error "TPU_SOC_RAM_D_LOCAL_ADDR_WIDTH must be a positive integer, got '$requested_ram_d_local_addr_width'"
    }
    set tpu_ram_d_local_addr_width $requested_ram_d_local_addr_width
  }
}

if {[info exists ::env(TPU_SOC_FIFO_DEPTH)]} {
  set requested_fifo_depth [string trim $::env(TPU_SOC_FIFO_DEPTH)]
  if {$requested_fifo_depth ne ""} {
    if {![string is integer -strict $requested_fifo_depth] || $requested_fifo_depth < 1} {
      error "TPU_SOC_FIFO_DEPTH must be a positive integer, got '$requested_fifo_depth'"
    }
    set tpu_fifo_depth $requested_fifo_depth
  }
}

if {[info exists ::env(TPU_SOC_PE_SIZE)]} {
  set requested_pe_size [string trim $::env(TPU_SOC_PE_SIZE)]
  if {$requested_pe_size ne ""} {
    if {![string is integer -strict $requested_pe_size] ||
        $requested_pe_size < 1 ||
        ($requested_pe_size & ($requested_pe_size - 1)) != 0} {
      error "TPU_SOC_PE_SIZE must be a positive power-of-two integer, got '$requested_pe_size'"
    }
    set tpu_pe_size $requested_pe_size
  }
}

if {[info exists ::env(TPU_SOC_INT8_DOT_FACTOR)]} {
  set requested_int8_dot_factor [string trim $::env(TPU_SOC_INT8_DOT_FACTOR)]
  if {$requested_int8_dot_factor ne ""} {
    if {![string is integer -strict $requested_int8_dot_factor] ||
        ($requested_int8_dot_factor != 4 && $requested_int8_dot_factor != 8 && $requested_int8_dot_factor != 16)} {
      error "TPU_SOC_INT8_DOT_FACTOR must be 4, 8, or 16, got '$requested_int8_dot_factor'"
    }
    set tpu_int8_dot_factor $requested_int8_dot_factor
  }
}

if {[info exists ::env(TPU_SOC_OP_TILE_N)]} {
  set requested_op_tile_n [string trim $::env(TPU_SOC_OP_TILE_N)]
  if {$requested_op_tile_n ne ""} {
    if {![string is integer -strict $requested_op_tile_n] ||
        ($requested_op_tile_n != 64 && $requested_op_tile_n != 128)} {
      error "TPU_SOC_OP_TILE_N must be 64 or 128, got '$requested_op_tile_n'"
    }
    set tpu_op_tile_n $requested_op_tile_n
  }
}

if {[info exists ::env(TPU_SOC_OP_TILE_K)]} {
  set requested_op_tile_k [string trim $::env(TPU_SOC_OP_TILE_K)]
  if {$requested_op_tile_k ne ""} {
    if {![string is integer -strict $requested_op_tile_k] ||
        ($requested_op_tile_k != 64 && $requested_op_tile_k != 128)} {
      error "TPU_SOC_OP_TILE_K must be 64 or 128, got '$requested_op_tile_k'"
    }
    set tpu_op_tile_k $requested_op_tile_k
  }
}

if {[info exists ::env(TPU_SOC_AXI_DATA_WIDTH)]} {
  set requested_axi_data_width [string trim $::env(TPU_SOC_AXI_DATA_WIDTH)]
  if {$requested_axi_data_width ne ""} {
    if {![string is integer -strict $requested_axi_data_width] ||
        $requested_axi_data_width < 32 ||
        ($requested_axi_data_width % 32) != 0} {
      error "TPU_SOC_AXI_DATA_WIDTH must be a positive multiple of 32, got '$requested_axi_data_width'"
    }
    set tpu_axi_data_width $requested_axi_data_width
  }
}

if {[info exists ::env(TPU_SOC_RAM_DATA_WIDTH)]} {
  set requested_ram_data_width [string trim $::env(TPU_SOC_RAM_DATA_WIDTH)]
  if {$requested_ram_data_width ne ""} {
    if {![string is integer -strict $requested_ram_data_width] ||
        $requested_ram_data_width < 32 ||
        ($requested_ram_data_width % 32) != 0} {
      error "TPU_SOC_RAM_DATA_WIDTH must be a positive multiple of 32, got '$requested_ram_data_width'"
    }
    set tpu_ram_data_width $requested_ram_data_width
  }
}

if {[info exists ::env(TPU_SOC_CDMA_MAX_BURST_LEN)]} {
  set requested_cdma_max_burst_len [string trim $::env(TPU_SOC_CDMA_MAX_BURST_LEN)]
  if {$requested_cdma_max_burst_len ne ""} {
    if {![string is integer -strict $requested_cdma_max_burst_len] ||
        $requested_cdma_max_burst_len < 1 ||
        $requested_cdma_max_burst_len > 256} {
      error "TPU_SOC_CDMA_MAX_BURST_LEN must be an integer in 1..256, got '$requested_cdma_max_burst_len'"
    }
    set cdma_max_burst_len $requested_cdma_max_burst_len
  }
}

if {[info exists ::env(TPU_SOC_CDMA_INCLUDE_SG)]} {
  set requested_cdma_include_sg [string trim $::env(TPU_SOC_CDMA_INCLUDE_SG)]
  if {$requested_cdma_include_sg ne ""} {
    if {$requested_cdma_include_sg eq "0"} {
      set cdma_include_sg 0
    } elseif {$requested_cdma_include_sg eq "1"} {
      set cdma_include_sg 1
    } else {
      error "TPU_SOC_CDMA_INCLUDE_SG must be 0 or 1, got '$requested_cdma_include_sg'"
    }
  }
}

proc resolve_tpu_repo_root {base_root} {
  set base_root_norm [file normalize $base_root]
  set direct_rtl_dir [file join $base_root_norm "rtl"]
  set local_v2_root [file normalize [file join $base_root_norm "third_party" "tpu_vanilla-v2-local"]]
  set local_v2_rtl_dir [file join $local_v2_root "rtl"]
  set prefer_local_third_party 1

  if {[info exists ::env(TPU_SOC_RTL_ROOT)] && $::env(TPU_SOC_RTL_ROOT) ne ""} {
    set explicit_root [file normalize $::env(TPU_SOC_RTL_ROOT)]
    set explicit_rtl_dir [file join $explicit_root "rtl"]
    if {[file isdirectory $explicit_rtl_dir] && [file exists [file join $explicit_rtl_dir "tpu_top.v"]]} {
      return $explicit_root
    }
    error "TPU_SOC_RTL_ROOT does not point to a valid TPU RTL repo: $::env(TPU_SOC_RTL_ROOT)"
  }

  if {[info exists ::env(TPU_SOC_PREFER_LOCAL_THIRD_PARTY)] && $::env(TPU_SOC_PREFER_LOCAL_THIRD_PARTY) ne ""} {
    set requested_prefer_local [string tolower $::env(TPU_SOC_PREFER_LOCAL_THIRD_PARTY)]
    if {$requested_prefer_local in {"0" "false" "no"}} {
      set prefer_local_third_party 0
    } elseif {$requested_prefer_local in {"1" "true" "yes"}} {
      set prefer_local_third_party 1
    } else {
      error "TPU_SOC_PREFER_LOCAL_THIRD_PARTY must be 0/1/true/false/yes/no, got '$::env(TPU_SOC_PREFER_LOCAL_THIRD_PARTY)'"
    }
  }

  # If the caller already passed the actual TPU RTL repo root, honor it first.
  if {[file isdirectory $direct_rtl_dir] && [file exists [file join $direct_rtl_dir "tpu_top.v"]]} {
    return $base_root_norm
  }

  # For this repo, prefer the checked-in local third_party mirror so Vivado
  # packaging matches the RTL we are actively editing and simulating.
  if {$prefer_local_third_party &&
      [file isdirectory $local_v2_rtl_dir] &&
      [file exists [file join $local_v2_rtl_dir "tpu_top.v"]]} {
    return $local_v2_root
  }

  # Prefer the external v2 source-of-truth repo when we are invoked from a
  # wrapper/project tree such as gemm_rtl_v1 or repo_shell.
  set search_root $base_root_norm
  while {1} {
    set sibling_v2_root [file normalize [file join $search_root "tpu_vanilla_v2"]]
    set sibling_v2_rtl_dir [file join $sibling_v2_root "rtl"]
    if {[file isdirectory $sibling_v2_rtl_dir] && [file exists [file join $sibling_v2_rtl_dir "tpu_top.v"]]} {
      return $sibling_v2_root
    }

    set parent_root [file dirname $search_root]
    if {$parent_root eq $search_root} {
      break
    }
    set search_root $parent_root
  }

  set candidates [list \
    [file join $base_root_norm "third_party" "tpu_vanilla-v2-local"] \
    [file join $base_root_norm "third_party" "tpu_vanilla-fp-32"] \
    $base_root_norm \
  ]

  foreach candidate $candidates {
    set candidate_norm [file normalize $candidate]
    set rtl_dir [file join $candidate_norm "rtl"]
    if {[file isdirectory $rtl_dir] && [file exists [file join $rtl_dir "tpu_top.v"]]} {
      return $candidate_norm
    }
  }

  error "TPU RTL repo root not found under base root: $base_root"
}

set repo_root [resolve_tpu_repo_root $project_root]

proc scan_verilog_files {dir file_list_var} {
  upvar $file_list_var file_list

  # Skip the standalone VC707-oriented wrapper tree; this project uses the Zynq BD wrapper as top.
  if {[file tail [file normalize $dir]] eq "system_pro"} {
    return
  }

  foreach file [lsort [glob -nocomplain -directory $dir "*.v"]] {
    # The fp32 tree contains both rtl/tpu_top.v and rtl/tpu_top_matmul.v, and both declare
    # module tpu_top. Keep the canonical top and skip the duplicate wrapper variant.
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

proc package_tpu_axi_ip {repo_root wrapper_file build_dir target_part ip_vendor ip_library ip_name ip_version} {
  set rtl_dir [file normalize [file join $repo_root "rtl"]]
  set support_root [file normalize [file join [file dirname $build_dir] "[file tail $build_dir]_support"]]
  set pkg_project_dir [file normalize [file join $support_root "${ip_name}_pkg"]]
  set ip_repo_root [file normalize [file join $support_root "ip_repo"]]
  set ip_root_dir [file normalize [file join $ip_repo_root "${ip_name}_${ip_version}"]]
  set rtl_files [list]

  if {![file isdirectory $rtl_dir]} {
    error "RTL directory not found: $rtl_dir"
  }
  if {![file exists $wrapper_file]} {
    error "TPU AXI wrapper not found: $wrapper_file"
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
  add_files -norecurse $wrapper_file
  if {[info exists ::env(TPU_SOC_VERILOG_DEFINES)] && [string trim $::env(TPU_SOC_VERILOG_DEFINES)] ne ""} {
    set define_string [string map {"," " "} [string trim $::env(TPU_SOC_VERILOG_DEFINES)]]
    set define_list [list]
    foreach item [split $define_string] {
      if {[string trim $item] ne ""} {
        lappend define_list [string trim $item]
      }
    }
    if {[llength $define_list] > 0} {
      puts "Setting TPU packaged-IP Verilog defines: $define_list"
      set_property verilog_define $define_list [get_filesets sources_1]
    }
  }
  update_compile_order -fileset sources_1
  set_property top $ip_name [get_filesets sources_1]

  ipx::package_project -root_dir $ip_root_dir -vendor $ip_vendor -library $ip_library -taxonomy /UserIP -import_files -force
  set core [ipx::current_core]
  set_property vendor $ip_vendor $core
  set_property library $ip_library $core
  set_property name $ip_name $core
  set_property version $ip_version $core
  set_property display_name "TPU Top AXI IP" $core
  set_property description "Packaged TPU top wrapper with explicit AXI interface metadata for PYNQ-ZU SoC integration." $core
  if {[llength [ipx::get_memory_maps -quiet S_AXI -of_objects $core]] > 0} {
    set s_axi_mmap [ipx::get_memory_maps S_AXI -of_objects $core]
    if {[llength [ipx::get_address_blocks -quiet reg0 -of_objects $s_axi_mmap]] > 0} {
      set_property USAGE memory [ipx::get_address_blocks reg0 -of_objects $s_axi_mmap]
    }
  }
  foreach busif {S_AXI S_AXIL M_AXI_RD M_AXI_RD1 M_AXI_RD2 M_AXI_RD3 M_AXI} {
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
if {![file exists $wrapper_file]} {
  error "TPU AXI wrapper not found: $wrapper_file"
}
set repo_root [resolve_tpu_repo_root $repo_root]
set wrapper_file [file normalize [file join $repo_root "rtl" "tpu_top_axi_ip.v"]]
puts "TPU RTL repo root: $repo_root"
puts "TPU AXI wrapper: $wrapper_file"

file mkdir $build_dir
set ip_repo_root [package_tpu_axi_ip $repo_root $wrapper_file $build_dir $target_part $tpu_ip_vendor $tpu_ip_library $tpu_ip_name $tpu_ip_version]

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
set system_ila_vlnv ""
if {$enable_system_ila} {
  set system_ila_vlnv [require_latest_ip_vlnv "xilinx.com:ip:system_ila:*" "system_ila"]
}
set prevd_ila_vlnv ""
if {$enable_prevd_ila} {
  set prevd_ila_vlnv [require_latest_ip_vlnv "xilinx.com:ip:ila:*" "ila"]
}
set tpu_ip_vlnv [require_latest_ip_vlnv "${tpu_ip_vendor}:${tpu_ip_library}:${tpu_ip_name}:*" $tpu_ip_name]
set zynq_ps_vlnv [require_latest_ip_vlnv "xilinx.com:ip:zynq_ultra_ps_e:*" "zynq_ultra_ps_e"]

puts "Using IP VLNVs:"
puts "  axi_bram_ctrl   = $axi_bram_ctrl_vlnv"
puts "  axi_cdma        = $axi_cdma_vlnv"
puts "  blk_mem_gen     = $blk_mem_gen_vlnv"
puts "  proc_sys_reset  = $proc_sys_reset_vlnv"
puts "  smartconnect    = $smartconnect_vlnv"
if {$enable_system_ila} {
  puts "  system_ila      = $system_ila_vlnv"
  puts "  system_ila_target = $system_ila_target"
}
if {$enable_prevd_ila} {
  puts "  prevd_ila       = $prevd_ila_vlnv"
}
puts "  tpu_top_axi_ip  = $tpu_ip_vlnv"
puts "  zynq_ultra_ps_e = $zynq_ps_vlnv"

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
  CONFIG.C_INCLUDE_SG $cdma_include_sg \
  CONFIG.C_M_AXI_DATA_WIDTH $tpu_axi_data_width \
  CONFIG.C_M_AXI_MAX_BURST_LEN $cdma_max_burst_len \
] [get_bd_cells axi_cdma_0]
create_bd_cell -type ip -vlnv $blk_mem_gen_vlnv blk_mem_gen_0
set_property -dict [list \
  CONFIG.Memory_Type {True_Dual_Port_RAM} \
  CONFIG.PRIM_type_to_Implement {URAM} \
] [get_bd_cells blk_mem_gen_0]

create_bd_cell -type ip -vlnv $proc_sys_reset_vlnv rst_ps8_0_100M

create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_0
set smartconnect_0_num_si 2
if {$cdma_include_sg} {
  set smartconnect_0_num_si 4
} else {
  set smartconnect_0_num_si 3
}
set_property -dict [list \
  CONFIG.NUM_MI {3} \
  CONFIG.NUM_SI $smartconnect_0_num_si \
] [get_bd_cells smartconnect_0]

if {$ab_dual_input} {
  create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_dual_in_0
  set_property -dict [list \
    CONFIG.ADVANCED_PROPERTIES {__experimental_features__ {disable_low_area_mode 1}} \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI {1} \
  ] [get_bd_cells smartconnect_dual_in_0]
}
if {$ab_dual_input >= 2} {
  create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_quad_in2_0
  set_property -dict [list \
    CONFIG.ADVANCED_PROPERTIES {__experimental_features__ {disable_low_area_mode 1}} \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI {1} \
  ] [get_bd_cells smartconnect_quad_in2_0]

  create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_quad_in3_0
  set_property -dict [list \
    CONFIG.ADVANCED_PROPERTIES {__experimental_features__ {disable_low_area_mode 1}} \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_SI {1} \
  ] [get_bd_cells smartconnect_quad_in3_0]
}

create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_1
set_property -dict [list \
  CONFIG.ADVANCED_PROPERTIES {__experimental_features__ {disable_low_area_mode 1}} \
  CONFIG.NUM_MI {2} \
  CONFIG.NUM_SI {1} \
] [get_bd_cells smartconnect_1]

create_bd_cell -type ip -vlnv $smartconnect_vlnv smartconnect_out_0
set_property -dict [list \
  CONFIG.ADVANCED_PROPERTIES {__experimental_features__ {disable_low_area_mode 1}} \
  CONFIG.NUM_MI {2} \
  CONFIG.NUM_SI {1} \
] [get_bd_cells smartconnect_out_0]

if {$enable_system_ila} {
  create_bd_cell -type ip -vlnv $system_ila_vlnv system_ila_out_0
  set_property -dict [list \
    CONFIG.C_MON_TYPE {INTERFACE} \
    CONFIG.C_NUM_MONITOR_SLOTS {1} \
    CONFIG.C_DATA_DEPTH $system_ila_depth \
    CONFIG.C_SLOT_0_INTF_TYPE {xilinx.com:interface:aximm_rtl:1.0} \
    CONFIG.C_SLOT_0_AXI_PROTOCOL {AXI4} \
    CONFIG.C_SLOT_0_AXI_AR_SEL {0} \
    CONFIG.C_SLOT_0_AXI_R_SEL {0} \
  ] [get_bd_cells system_ila_out_0]
}
if {$enable_prevd_ila} {
  create_bd_cell -type ip -vlnv $prevd_ila_vlnv prevd_ila_0
  set_property -dict [list \
    CONFIG.C_DATA_DEPTH $prevd_ila_depth \
    CONFIG.C_MONITOR_TYPE {Native} \
    CONFIG.C_ENABLE_ILA_AXI_MON {false} \
    CONFIG.C_NUM_OF_PROBES {1} \
    CONFIG.C_PROBE0_WIDTH {32} \
  ] [get_bd_cells prevd_ila_0]
}

create_bd_cell -type ip -vlnv $tpu_ip_vlnv tpu_top_0
set tpu_top_cfg [list \
  CONFIG.MATRIX_A_BASE_ADDR $matrix_a_base \
  CONFIG.MATRIX_B_BASE_ADDR $matrix_b_base \
  CONFIG.MATRIX_C_BASE_ADDR $matrix_c_base \
  CONFIG.PE_SIZE $tpu_pe_size \
  CONFIG.INT8_DOT_FACTOR $tpu_int8_dot_factor \
  CONFIG.OP_TILE_N $tpu_op_tile_n \
  CONFIG.OP_TILE_K $tpu_op_tile_k \
  CONFIG.AB_DUAL_INPUT $ab_dual_input \
  CONFIG.CMAT_D_FULL_WORD_PACK $cmat_d_full_word_pack \
  CONFIG.CMAT_STREAMING_C_ADDER $cmat_streaming_c_adder \
  CONFIG.AXI_DATA_WIDTH $tpu_axi_data_width \
  CONFIG.AXI_AWUSER_WIDTH {8} \
  CONFIG.AXI_WUSER_WIDTH {8} \
  CONFIG.AXI_BUSER_WIDTH {8} \
  CONFIG.RAM_DATA_WIDTH $tpu_ram_data_width \
  CONFIG.MATRIX_D_BASE_ADDR $matrix_d_base \
]
if {$tpu_ram_addr_width ne ""} {
  lappend tpu_top_cfg CONFIG.RAM_ADDR_WIDTH $tpu_ram_addr_width
}
if {$tpu_ram_c_addr_width ne ""} {
  lappend tpu_top_cfg CONFIG.RAM_C_ADDR_WIDTH $tpu_ram_c_addr_width
}
if {$tpu_ram_d_addr_width ne ""} {
  lappend tpu_top_cfg CONFIG.RAM_D_ADDR_WIDTH $tpu_ram_d_addr_width
}
if {$tpu_ram_a_local_addr_width ne ""} {
  lappend tpu_top_cfg CONFIG.RAM_A_LOCAL_ADDR_WIDTH $tpu_ram_a_local_addr_width
}
if {$tpu_ram_b_local_addr_width ne ""} {
  lappend tpu_top_cfg CONFIG.RAM_B_LOCAL_ADDR_WIDTH $tpu_ram_b_local_addr_width
}
if {$tpu_ram_d_local_addr_width ne ""} {
  lappend tpu_top_cfg CONFIG.RAM_D_LOCAL_ADDR_WIDTH $tpu_ram_d_local_addr_width
}
if {$tpu_fifo_depth ne ""} {
  lappend tpu_top_cfg CONFIG.FIFO_DEPTH $tpu_fifo_depth
}
set_property -dict $tpu_top_cfg [get_bd_cells tpu_top_0]

create_bd_cell -type ip -vlnv $zynq_ps_vlnv zynq_ultra_ps_e_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "0"} [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list \
  CONFIG.PSU__USE__IRQ0 {1} \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {1} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP0 {1} \
  CONFIG.PSU__USE__S_AXI_GP1 {1} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__USE__S_AXI_GP3 $enable_ps_s_axi_gp3 \
  CONFIG.PSU__USE__S_AXI_GP4 $enable_ps_s_axi_gp4 \
  CONFIG.PSU__USE__S_AXI_GP5 $enable_ps_s_axi_gp5 \
  CONFIG.PSU__SAXIGP4__DATA_WIDTH {128} \
  CONFIG.PSU__SAXIGP5__DATA_WIDTH {128} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $pl_clk0_mhz \
] [get_bd_cells zynq_ultra_ps_e_0]

connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTA]
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_1/BRAM_PORTA] [get_bd_intf_pins blk_mem_gen_0/BRAM_PORTB]
if {$cdma_include_sg} {
  connect_bd_intf_net [get_bd_intf_pins axi_cdma_0/M_AXI_SG] [get_bd_intf_pins smartconnect_0/S02_AXI]
  connect_bd_intf_net [get_bd_intf_pins tpu_top_0/m_axi_rd] [get_bd_intf_pins smartconnect_0/S03_AXI]
} else {
  connect_bd_intf_net [get_bd_intf_pins tpu_top_0/m_axi_rd] [get_bd_intf_pins smartconnect_0/S02_AXI]
}
if {$ab_dual_input} {
  connect_bd_intf_net [get_bd_intf_pins tpu_top_0/m_axi_rd1] [get_bd_intf_pins smartconnect_dual_in_0/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins smartconnect_dual_in_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/$dual_input_ps_intf]
}
if {$ab_dual_input >= 2} {
  connect_bd_intf_net [get_bd_intf_pins tpu_top_0/m_axi_rd2] [get_bd_intf_pins smartconnect_quad_in2_0/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins smartconnect_quad_in2_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/$quad_input2_ps_intf]
  connect_bd_intf_net [get_bd_intf_pins tpu_top_0/m_axi_rd3] [get_bd_intf_pins smartconnect_quad_in3_0/S00_AXI]
  connect_bd_intf_net [get_bd_intf_pins smartconnect_quad_in3_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/$quad_input3_ps_intf]
}
connect_bd_intf_net [get_bd_intf_pins axi_cdma_0/M_AXI] [get_bd_intf_pins smartconnect_0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/$input_ps_intf]
connect_bd_intf_net [get_bd_intf_pins axi_bram_ctrl_0/S_AXI] [get_bd_intf_pins smartconnect_0/M01_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M02_AXI] [get_bd_intf_pins tpu_top_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_cdma_0/S_AXI_LITE] [get_bd_intf_pins smartconnect_1/M00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_1/M01_AXI] [get_bd_intf_pins tpu_top_0/s_axil]
connect_bd_intf_net [get_bd_intf_pins tpu_top_0/m_axi] [get_bd_intf_pins smartconnect_out_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_out_0/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/$output_ps_intf]
if {$enable_system_ila} {
  if {$system_ila_target eq "OUTPUT"} {
    connect_bd_intf_net [get_bd_intf_pins smartconnect_out_0/M00_AXI] [get_bd_intf_pins system_ila_out_0/SLOT_0_AXI]
  } else {
    connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M02_AXI] [get_bd_intf_pins system_ila_out_0/SLOT_0_AXI]
  }
}
if {$enable_prevd_ila} {
  connect_bd_net [get_bd_pins tpu_top_0/ila_prevd_ctrl] [get_bd_pins prevd_ila_0/probe0]
}
connect_bd_intf_net [get_bd_intf_pins smartconnect_out_0/M01_AXI] [get_bd_intf_pins axi_bram_ctrl_1/S_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/S00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD]
connect_bd_intf_net [get_bd_intf_pins smartconnect_1/S00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM1_FPD]

connect_bd_net [get_bd_pins axi_cdma_0/cdma_introut] [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins axi_bram_ctrl_1/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins axi_cdma_0/s_axi_lite_aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins smartconnect_0/aresetn]
if {$ab_dual_input} {
  connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins smartconnect_dual_in_0/aresetn]
}
if {$ab_dual_input >= 2} {
  connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins smartconnect_quad_in2_0/aresetn]
  connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins smartconnect_quad_in3_0/aresetn]
}
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins smartconnect_1/aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins smartconnect_out_0/aresetn]
connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins tpu_top_0/rst_n]
if {$enable_system_ila} {
  connect_bd_net [get_bd_pins rst_ps8_0_100M/peripheral_aresetn] [get_bd_pins system_ila_out_0/resetn]
}

connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_bram_ctrl_1/s_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_cdma_0/m_axi_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins axi_cdma_0/s_axi_lite_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins rst_ps8_0_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_0/aclk]
if {$ab_dual_input} {
  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_dual_in_0/aclk]
}
if {$ab_dual_input >= 2} {
  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_quad_in2_0/aclk]
  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_quad_in3_0/aclk]
}
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_1/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_out_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins tpu_top_0/clk]
if {$enable_system_ila} {
  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins system_ila_out_0/clk]
}
if {$enable_prevd_ila} {
  connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins prevd_ila_0/clk]
}
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/maxihpm1_fpd_aclk]
set ps_slave_aclk_pins [lsort -unique [list \
  saxihp0_fpd_aclk \
  saxihpc0_fpd_aclk \
  saxihpc1_fpd_aclk \
  $input_ps_aclk_pin \
  $dual_input_ps_aclk_pin \
  $quad_input2_ps_aclk_pin \
  $quad_input3_ps_aclk_pin \
  $output_ps_aclk_pin \
]]
foreach ps_aclk_pin $ps_slave_aclk_pins {
  if {[llength [get_bd_pins -quiet zynq_ultra_ps_e_0/$ps_aclk_pin]] > 0} {
    connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/$ps_aclk_pin]
  }
}
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_resetn0] [get_bd_pins rst_ps8_0_100M/ext_reset_in]

proc maybe_assign_optional_ddr_window {label addr_space_name addr_seg_name base range} {
  if {$base eq "" && $range eq ""} {
    return
  }
  if {$base eq "" || $range eq ""} {
    error "Optional DDR window '$label' requires both base and range (base='$base', range='$range')"
  }
  puts "Assigning optional DDR window $label: base=$base range=$range addr_space=$addr_space_name addr_seg=$addr_seg_name"
  if {[catch {
    assign_bd_address -offset $base -range $range \
      -target_address_space [get_bd_addr_spaces $addr_space_name] \
      [get_bd_addr_segs $addr_seg_name] -force
  } msg]} {
    error "Optional DDR window '$label' assignment failed: $msg"
  }
}

assign_bd_address
assign_bd_address -offset 0xA0000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
assign_bd_address -offset 0x10000000 -range 0x10000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs tpu_top_0/s_axi/reg0] -force
assign_bd_address -offset $input_ddr_base -range $input_ddr_range -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs zynq_ultra_ps_e_0/$input_ddr_addr_space] -force
maybe_assign_optional_ddr_window "cdma-input-high" "axi_cdma_0/Data" "zynq_ultra_ps_e_0/$input_ddr_addr_space" $input_ddr_high_base $input_ddr_high_range
if {$cdma_include_sg} {
  assign_bd_address -offset $input_ddr_base -range $input_ddr_range -target_address_space [get_bd_addr_spaces axi_cdma_0/Data_SG] [get_bd_addr_segs zynq_ultra_ps_e_0/$input_ddr_addr_space] -force
  maybe_assign_optional_ddr_window "cdma-sg-input-high" "axi_cdma_0/Data_SG" "zynq_ultra_ps_e_0/$input_ddr_addr_space" $input_ddr_high_base $input_ddr_high_range
}
assign_bd_address -offset 0xA0000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces tpu_top_0/m_axi] [get_bd_addr_segs axi_bram_ctrl_1/S_AXI/Mem0] -force
assign_bd_address -offset $output_ddr_base -range $output_ddr_range -target_address_space [get_bd_addr_spaces tpu_top_0/m_axi] [get_bd_addr_segs zynq_ultra_ps_e_0/$output_ddr_addr_space] -force
maybe_assign_optional_ddr_window "tpu-output-high" "tpu_top_0/m_axi" "zynq_ultra_ps_e_0/$output_ddr_addr_space" $output_ddr_high_base $output_ddr_high_range
assign_bd_address -offset $input_ddr_base -range $input_ddr_range -target_address_space [get_bd_addr_spaces tpu_top_0/m_axi_rd] [get_bd_addr_segs zynq_ultra_ps_e_0/$input_ddr_addr_space] -force
maybe_assign_optional_ddr_window "tpu-input-rd-high" "tpu_top_0/m_axi_rd" "zynq_ultra_ps_e_0/$input_ddr_addr_space" $input_ddr_high_base $input_ddr_high_range
if {$ab_dual_input} {
  assign_bd_address -offset $input_ddr_base -range $input_ddr_range -target_address_space [get_bd_addr_spaces tpu_top_0/m_axi_rd1] [get_bd_addr_segs zynq_ultra_ps_e_0/$dual_input_ddr_addr_space] -force
  maybe_assign_optional_ddr_window "tpu-input-rd1-high" "tpu_top_0/m_axi_rd1" "zynq_ultra_ps_e_0/$dual_input_ddr_addr_space" $input_ddr_high_base $input_ddr_high_range
}
if {$ab_dual_input >= 2} {
  assign_bd_address -offset $input_ddr_base -range $input_ddr_range -target_address_space [get_bd_addr_spaces tpu_top_0/m_axi_rd2] [get_bd_addr_segs zynq_ultra_ps_e_0/$quad_input2_ddr_addr_space] -force
  maybe_assign_optional_ddr_window "tpu-input-rd2-high" "tpu_top_0/m_axi_rd2" "zynq_ultra_ps_e_0/$quad_input2_ddr_addr_space" $input_ddr_high_base $input_ddr_high_range
  assign_bd_address -offset $input_ddr_base -range $input_ddr_range -target_address_space [get_bd_addr_spaces tpu_top_0/m_axi_rd3] [get_bd_addr_segs zynq_ultra_ps_e_0/$quad_input3_ddr_addr_space] -force
  maybe_assign_optional_ddr_window "tpu-input-rd3-high" "tpu_top_0/m_axi_rd3" "zynq_ultra_ps_e_0/$quad_input3_ddr_addr_space" $input_ddr_high_base $input_ddr_high_range
}
assign_bd_address -offset 0xA0000000 -range 0x00002000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
assign_bd_address -offset 0xB0010000 -range 0x00010000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs axi_cdma_0/S_AXI_LITE/Reg] -force
assign_bd_address -offset 0xB0000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs tpu_top_0/s_axil/reg0] -force

# The PS should use the CSR path only. Keep the TPU data slave visible to CDMA, not PS HPM0.
foreach seg [list \
  /zynq_ultra_ps_e_0/Data/SEG_tpu_top_0_reg0 \
  /zynq_ultra_ps_e_0/Data/SEG_zynq_ultra_ps_e_0_HP0_DDR_LOW \
  /zynq_ultra_ps_e_0/Data/SEG_zynq_ultra_ps_e_0_HP0_LPS_OCM \
] {
  if {[llength [get_bd_addr_segs -quiet $seg]] > 0} {
    exclude_bd_addr_seg $seg
  }
}

if {$cdma_include_sg} {
  foreach seg [list \
    /axi_cdma_0/Data_SG/SEG_tpu_top_0_reg0 \
    /axi_cdma_0/Data_SG/SEG_axi_bram_ctrl_0_Mem0 \
    /axi_cdma_0/Data_SG/SEG_zynq_ultra_ps_e_0_HP0_LPS_OCM \
  ] {
    if {[llength [get_bd_addr_segs -quiet $seg]] > 0} {
      exclude_bd_addr_seg $seg
    }
  }
}

foreach seg [list \
  /tpu_top_0/M_AXI_RD/SEG_tpu_top_0_reg0 \
  /tpu_top_0/M_AXI_RD/SEG_axi_bram_ctrl_0_Mem0 \
  /tpu_top_0/M_AXI_RD/SEG_zynq_ultra_ps_e_0_HPC0_LPS_OCM \
  /tpu_top_0/M_AXI_RD/SEG_zynq_ultra_ps_e_0_HPC1_LPS_OCM \
  /tpu_top_0/M_AXI_RD/SEG_zynq_ultra_ps_e_0_HP0_LPS_OCM \
  /tpu_top_0/M_AXI_RD/SEG_zynq_ultra_ps_e_0_HP1_LPS_OCM \
  /tpu_top_0/M_AXI_RD1/SEG_tpu_top_0_reg0 \
  /tpu_top_0/M_AXI_RD1/SEG_axi_bram_ctrl_0_Mem0 \
  /tpu_top_0/M_AXI_RD1/SEG_zynq_ultra_ps_e_0_HPC0_LPS_OCM \
  /tpu_top_0/M_AXI_RD1/SEG_zynq_ultra_ps_e_0_HPC1_LPS_OCM \
  /tpu_top_0/M_AXI_RD1/SEG_zynq_ultra_ps_e_0_HP0_LPS_OCM \
  /tpu_top_0/M_AXI_RD1/SEG_zynq_ultra_ps_e_0_HP1_LPS_OCM \
  /tpu_top_0/M_AXI_RD2/SEG_tpu_top_0_reg0 \
  /tpu_top_0/M_AXI_RD2/SEG_axi_bram_ctrl_0_Mem0 \
  /tpu_top_0/M_AXI_RD2/SEG_zynq_ultra_ps_e_0_HPC0_LPS_OCM \
  /tpu_top_0/M_AXI_RD2/SEG_zynq_ultra_ps_e_0_HPC1_LPS_OCM \
  /tpu_top_0/M_AXI_RD2/SEG_zynq_ultra_ps_e_0_HP0_LPS_OCM \
  /tpu_top_0/M_AXI_RD2/SEG_zynq_ultra_ps_e_0_HP1_LPS_OCM \
  /tpu_top_0/M_AXI_RD2/SEG_zynq_ultra_ps_e_0_HP2_LPS_OCM \
  /tpu_top_0/M_AXI_RD2/SEG_zynq_ultra_ps_e_0_HP3_LPS_OCM \
  /tpu_top_0/M_AXI_RD3/SEG_tpu_top_0_reg0 \
  /tpu_top_0/M_AXI_RD3/SEG_axi_bram_ctrl_0_Mem0 \
  /tpu_top_0/M_AXI_RD3/SEG_zynq_ultra_ps_e_0_HPC0_LPS_OCM \
  /tpu_top_0/M_AXI_RD3/SEG_zynq_ultra_ps_e_0_HPC1_LPS_OCM \
  /tpu_top_0/M_AXI_RD3/SEG_zynq_ultra_ps_e_0_HP0_LPS_OCM \
  /tpu_top_0/M_AXI_RD3/SEG_zynq_ultra_ps_e_0_HP1_LPS_OCM \
  /tpu_top_0/M_AXI_RD3/SEG_zynq_ultra_ps_e_0_HP2_LPS_OCM \
  /tpu_top_0/M_AXI_RD3/SEG_zynq_ultra_ps_e_0_HP3_LPS_OCM \
  /tpu_top_0/M_AXI/SEG_zynq_ultra_ps_e_0_HPC0_LPS_OCM \
  /tpu_top_0/M_AXI/SEG_zynq_ultra_ps_e_0_HPC1_LPS_OCM \
  /tpu_top_0/M_AXI/SEG_zynq_ultra_ps_e_0_HP0_LPS_OCM \
  /tpu_top_0/M_AXI/SEG_zynq_ultra_ps_e_0_HP1_LPS_OCM \
  /tpu_top_0/M_AXI/SEG_zynq_ultra_ps_e_0_HP2_LPS_OCM \
  /tpu_top_0/M_AXI/SEG_zynq_ultra_ps_e_0_HP3_LPS_OCM \
] {
  if {[llength [get_bd_addr_segs -quiet $seg]] > 0} {
    exclude_bd_addr_seg $seg
  }
}

validate_bd_design
save_bd_design

make_wrapper -files [get_files [file join $build_dir $project_name.srcs sources_1 bd $bd_name $bd_name.bd]] -top
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
puts "TPU MATRIX_D_BASE_ADDR: $matrix_d_base"
puts "TPU PL CLK0 MHz: $pl_clk0_mhz"
puts "TPU AB dual input: $ab_dual_input"
if {$ab_dual_input} {
  puts "TPU AB input ports: lane0=$input_port_name lane1=$dual_input_port_name"
}
if {$ab_dual_input >= 2} {
  puts "TPU AB input ports: lane2=$quad_input2_port_name lane3=$quad_input3_port_name"
}
puts "AXI CDMA include SG: $cdma_include_sg"
puts "AXI CDMA max burst len: $cdma_max_burst_len"
puts "Address plan:"
puts "  PS -> BRAM            : 0xA0000000"
puts "  PS -> TPU CSR         : 0xB0000000"
puts "  PS -> AXI CDMA regs   : 0xB0010000"
puts "  CDMA -> TPU input win : 0x10000000"
puts "  TPU A/B/C decode base : $matrix_a_base / $matrix_b_base / $matrix_c_base"
if {$enable_system_ila} {
  if {$system_ila_target eq "OUTPUT"} {
    puts "  ILA monitor           : smartconnect_out_0/M00_AXI (depth=$system_ila_depth)"
  } else {
    puts "  ILA monitor           : smartconnect_0/M02_AXI (CDMA->TPU data path, depth=$system_ila_depth)"
  }
}
if {$enable_prevd_ila} {
  puts "  PREV_D ILA            : control + loader data sample (depth=$prevd_ila_depth)"
}
