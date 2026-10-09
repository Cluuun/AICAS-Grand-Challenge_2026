set script_dir [file normalize [file dirname [info script]]]
set create_tcl [file normalize [file join $script_dir "create_pynq_zu_tpu_soc_bd.tcl"]]
set build_dir [file normalize [file join $script_dir "build_tpu_soc_bd"]]
set project_name "pynq_zu_tpu_soc"
set bd_name "tpu_soc"
set matrix_d_base "0xA0000000"
# Shared-host default: the 14-core / 31 GiB zcq-vivado host has shown large
# headroom with jobs=8 (Vivado peak well under 8 GiB), so default a bit higher
# while still leaving a couple of cores for the OS and ad-hoc work. Explicit
# "auto" remains available via tclargs. Vivado 2022.1 on this host still
# rejects general.maxThreads > 8, so keep batch jobs and per-tool threads
# separate.
set build_jobs "12"
set tool_threads 1

proc env_value_or_empty {name} {
  if {[info exists ::env($name)]} {
    return [string trim $::env($name)]
  }
  return ""
}

proc parse_hwh_parameter {hwh_path param_name} {
  if {![file exists $hwh_path]} {
    return ""
  }
  set fh [open $hwh_path "r"]
  set content [read $fh]
  close $fh
  set pattern [format {<PARAMETER NAME="%s" VALUE="([^"]+)"} $param_name]
  if {[regexp -- $pattern $content -> value]} {
    return $value
  }
  return ""
}

proc parse_timing_summary_metrics {timing_rpt_path} {
  set metrics [dict create wns "" tns "" failing_endpoints ""]
  if {![file exists $timing_rpt_path]} {
    return $metrics
  }
  set fh [open $timing_rpt_path "r"]
  set content [read $fh]
  close $fh
  if {[regexp {Setup\s+:\s+([0-9]+)\s+Failing Endpoints,\s+Worst Slack\s+([-0-9.]+)ns,\s+Total Violation\s+([-0-9.]+)ns} $content -> failing_endpoints wns tns]} {
    dict set metrics wns $wns
    dict set metrics tns $tns
    dict set metrics failing_endpoints $failing_endpoints
  }
  return $metrics
}

proc sha256_or_empty {path} {
  if {![file exists $path]} {
    return ""
  }
  if {[catch {exec sha256sum $path} out]} {
    return ""
  }
  return [lindex [split [string trim $out]] 0]
}

proc maybe_require_exact {label actual expected} {
  if {$expected ne "" && $actual ne $expected} {
    error "$label mismatch: expected $expected, got $actual"
  }
}

proc maybe_set_run_property_from_env {run_name prop_name env_name} {
  set value [env_value_or_empty $env_name]
  if {$value ne ""} {
    puts "Setting $run_name $prop_name from $env_name=$value"
    set_property $prop_name $value [get_runs $run_name]
  }
}

proc write_manifest {manifest_path metadata} {
  set fh [open $manifest_path "w"]
  dict for {key value} $metadata {
    puts $fh "$key=$value"
  }
  close $fh
}

proc choose_build_jobs {} {
  set cpu_count 1
  set mem_avail_gib 0
  set load_avg_1 0.0

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

  if {![catch {exec awk {print $1} /proc/loadavg} load_out]} {
    set load_out [string trim $load_out]
    if {[string is double -strict $load_out]} {
      set load_avg_1 $load_out
    }
  }

  set reserve_cpu [expr {int(ceil($load_avg_1))}]
  if {$reserve_cpu < 0} {
    set reserve_cpu 0
  }

  set cpu_limit [expr {$cpu_count - $reserve_cpu}]
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

  puts "Auto-selected build jobs: $jobs (cpu_count=$cpu_count, mem_avail_gib=$mem_avail_gib, load_avg_1=$load_avg_1)"
  return $jobs
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

source $create_tcl

reset_run synth_1
launch_runs synth_1 -jobs $build_jobs
wait_on_run synth_1

if {[env_value_or_empty TPU_SOC_STOP_AFTER_SYNTH] ne "" && [env_value_or_empty TPU_SOC_STOP_AFTER_SYNTH] ne "0"} {
  set output_dir [file normalize [file join $build_dir "export"]]
  file mkdir $output_dir
  open_run synth_1
  set synth_util_out [file normalize [file join $output_dir "${bd_name}_wrapper_utilization_synth.rpt"]]
  set synth_util_hier_out [file normalize [file join $output_dir "${bd_name}_wrapper_utilization_hier_synth.rpt"]]
  report_utilization -file $synth_util_out
  report_utilization -hierarchical -file $synth_util_hier_out
  puts "Synthesis-only TPU SoC build finished for project: $project_name"
  puts "Project directory: $build_dir"
  puts "Build jobs: $build_jobs"
  puts "Synthesis utilization: $synth_util_out"
  puts "Synthesis hierarchical utilization: $synth_util_hier_out"
  exit 0
}

reset_run impl_1

maybe_set_run_property_from_env impl_1 strategy TPU_SOC_IMPL_STRATEGY
maybe_set_run_property_from_env impl_1 STEPS.PLACE_DESIGN.ARGS.DIRECTIVE TPU_SOC_PLACE_DIRECTIVE
maybe_set_run_property_from_env impl_1 STEPS.PHYS_OPT_DESIGN.IS_ENABLED TPU_SOC_PHYS_OPT_ENABLE
maybe_set_run_property_from_env impl_1 STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE TPU_SOC_PHYS_OPT_DIRECTIVE
maybe_set_run_property_from_env impl_1 STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE TPU_SOC_ROUTE_DIRECTIVE
maybe_set_run_property_from_env impl_1 STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED TPU_SOC_POST_ROUTE_PHYS_OPT_ENABLE
maybe_set_run_property_from_env impl_1 STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE TPU_SOC_POST_ROUTE_PHYS_OPT_DIRECTIVE

launch_runs impl_1 -to_step write_bitstream -jobs $build_jobs
wait_on_run impl_1

set output_dir [file normalize [file join $build_dir "export"]]
file mkdir $output_dir

open_run impl_1
set timing_summary_out [file normalize [file join $output_dir "${bd_name}_wrapper_timing_summary_routed.rpt"]]
report_timing_summary -max_paths 10 -report_unconstrained -file $timing_summary_out -warn_on_violation

set impl_bit [file normalize [file join $build_dir "${project_name}.runs" "impl_1" "${bd_name}_wrapper.bit"]]
set hwh_src [file normalize [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name "hw_handoff" "${bd_name}.hwh"]]
set bit_out [file normalize [file join $output_dir "${bd_name}_wrapper.bit"]]
set hwh_out [file normalize [file join $output_dir "${bd_name}_wrapper.hwh"]]
set bin_out [file normalize [file join $output_dir "${bd_name}_wrapper.bin"]]
set ltx_out [file normalize [file join $output_dir "${bd_name}_wrapper.ltx"]]
set bif_out [file normalize [file join $output_dir "${bd_name}_wrapper_fpga_manager.bif"]]
set manifest_out [file normalize [file join $output_dir "manifest.txt"]]
set impl_dir [file normalize [file join $build_dir "${project_name}.runs" "impl_1"]]

file copy -force $impl_bit $bit_out
if {[file exists $hwh_src]} {
  file copy -force $hwh_src $hwh_out
}

set ltx_candidates [glob -nocomplain -directory $impl_dir *.ltx]
if {[llength $ltx_candidates] == 0} {
  set ltx_candidates [glob -nocomplain -directory $build_dir *.ltx]
}
if {[llength $ltx_candidates] > 0} {
  set ltx_src [lindex $ltx_candidates 0]
  file copy -force $ltx_src $ltx_out
}

set bif_fh [open $bif_out "w"]
puts $bif_fh "all:"
puts $bif_fh "{"
puts $bif_fh "  ${bd_name}_wrapper.bit"
puts $bif_fh "}"
close $bif_fh

set saved_cwd [pwd]
cd $output_dir
if {[catch {exec bootgen -arch zynqmp -image [file tail $bif_out] -process_bitstream bin -w on} bootgen_out]} {
  cd $saved_cwd
  puts $bootgen_out
  error "bootgen failed while generating fpga_manager .bin"
}
cd $saved_cwd

set bootgen_bin [file normalize [file join $output_dir "${bd_name}_wrapper.bit.bin"]]
if {![file exists $bootgen_bin]} {
  error "bootgen did not produce expected output: $bootgen_bin"
}
file copy -force $bootgen_bin $bin_out

set actual_axi_data_width [parse_hwh_parameter $hwh_out "AXI_DATA_WIDTH"]
set actual_ram_data_width [parse_hwh_parameter $hwh_out "RAM_DATA_WIDTH"]
set actual_ram_a_local_addr_width [parse_hwh_parameter $hwh_out "RAM_A_LOCAL_ADDR_WIDTH"]
set actual_ram_b_local_addr_width [parse_hwh_parameter $hwh_out "RAM_B_LOCAL_ADDR_WIDTH"]
set actual_ram_d_local_addr_width [parse_hwh_parameter $hwh_out "RAM_D_LOCAL_ADDR_WIDTH"]
set actual_pe_size [parse_hwh_parameter $hwh_out "PE_SIZE"]
set actual_int8_dot_factor [parse_hwh_parameter $hwh_out "INT8_DOT_FACTOR"]
set actual_op_tile_n [parse_hwh_parameter $hwh_out "OP_TILE_N"]
set actual_op_tile_k [parse_hwh_parameter $hwh_out "OP_TILE_K"]
set actual_ab_dual_input [parse_hwh_parameter $hwh_out "AB_DUAL_INPUT"]
set actual_cmat_d_full_word_pack [parse_hwh_parameter $hwh_out "CMAT_D_FULL_WORD_PACK"]
set actual_cmat_streaming_c_adder [parse_hwh_parameter $hwh_out "CMAT_STREAMING_C_ADDER"]
set actual_m_axi_data_width [parse_hwh_parameter $hwh_out "C_M_AXI_DATA_WIDTH"]
set timing_metrics [parse_timing_summary_metrics $timing_summary_out]
set timing_wns [dict get $timing_metrics wns]
set timing_tns [dict get $timing_metrics tns]
set timing_failing_endpoints [dict get $timing_metrics failing_endpoints]

maybe_require_exact "AXI_DATA_WIDTH" $actual_axi_data_width [env_value_or_empty TPU_SOC_REQUIRE_AXI_DATA_WIDTH]
maybe_require_exact "RAM_DATA_WIDTH" $actual_ram_data_width [env_value_or_empty TPU_SOC_REQUIRE_RAM_DATA_WIDTH]
maybe_require_exact "PE_SIZE" $actual_pe_size [env_value_or_empty TPU_SOC_REQUIRE_PE_SIZE]
maybe_require_exact "INT8_DOT_FACTOR" $actual_int8_dot_factor [env_value_or_empty TPU_SOC_REQUIRE_INT8_DOT_FACTOR]
maybe_require_exact "OP_TILE_N" $actual_op_tile_n [env_value_or_empty TPU_SOC_REQUIRE_OP_TILE_N]
maybe_require_exact "OP_TILE_K" $actual_op_tile_k [env_value_or_empty TPU_SOC_REQUIRE_OP_TILE_K]
maybe_require_exact "AB_DUAL_INPUT" $actual_ab_dual_input [env_value_or_empty TPU_SOC_REQUIRE_AB_DUAL_INPUT]
maybe_require_exact "CMAT_D_FULL_WORD_PACK" $actual_cmat_d_full_word_pack [env_value_or_empty TPU_SOC_REQUIRE_CMAT_D_FULL_WORD_PACK]
maybe_require_exact "CMAT_STREAMING_C_ADDER" $actual_cmat_streaming_c_adder [env_value_or_empty TPU_SOC_REQUIRE_CMAT_STREAMING_C_ADDER]

set require_timing_clean [env_value_or_empty TPU_SOC_REQUIRE_TIMING_CLEAN]
if {$require_timing_clean ne "" && $require_timing_clean ne "0"} {
  if {$timing_wns eq "" || $timing_tns eq ""} {
    error "Timing clean requested but timing summary metrics were not captured"
  }
  if {$timing_wns < 0.0 || $timing_tns < 0.0} {
    error "Timing clean requested but routed timing is not clean (WNS=$timing_wns, TNS=$timing_tns)"
  }
}

set manifest_data [dict create \
  delivery_tag [expr {[env_value_or_empty TPU_SOC_DELIVERY_TAG] ne "" ? [env_value_or_empty TPU_SOC_DELIVERY_TAG] : $project_name}] \
  project_name $project_name \
  bd_name $bd_name \
  build_dir $build_dir \
  export_dir $output_dir \
  build_log_path [env_value_or_empty TPU_SOC_BUILD_LOG_PATH] \
  timing_report_path $timing_summary_out \
  bit_path $bit_out \
  bin_path $bin_out \
  hwh_path $hwh_out \
  ltx_path [expr {[file exists $ltx_out] ? $ltx_out : ""}] \
  requested_axi_data_width [env_value_or_empty TPU_SOC_AXI_DATA_WIDTH] \
  requested_ram_data_width [env_value_or_empty TPU_SOC_RAM_DATA_WIDTH] \
  requested_ram_a_local_addr_width [env_value_or_empty TPU_SOC_RAM_A_LOCAL_ADDR_WIDTH] \
  requested_ram_b_local_addr_width [env_value_or_empty TPU_SOC_RAM_B_LOCAL_ADDR_WIDTH] \
  requested_ram_d_local_addr_width [env_value_or_empty TPU_SOC_RAM_D_LOCAL_ADDR_WIDTH] \
  requested_pe_size [env_value_or_empty TPU_SOC_PE_SIZE] \
  requested_int8_dot_factor [env_value_or_empty TPU_SOC_INT8_DOT_FACTOR] \
  requested_op_tile_n [env_value_or_empty TPU_SOC_OP_TILE_N] \
  requested_op_tile_k [env_value_or_empty TPU_SOC_OP_TILE_K] \
	  requested_ab_dual_input [env_value_or_empty TPU_SOC_AB_DUAL_INPUT] \
	  requested_cmat_d_full_word_pack [env_value_or_empty TPU_SOC_CMAT_D_FULL_WORD_PACK] \
	  requested_cmat_streaming_c_adder [env_value_or_empty TPU_SOC_CMAT_STREAMING_C_ADDER] \
	  requested_verilog_defines [env_value_or_empty TPU_SOC_VERILOG_DEFINES] \
	  requested_input_ddr_base [env_value_or_empty TPU_SOC_INPUT_DDR_BASE] \
	  requested_input_ddr_range [env_value_or_empty TPU_SOC_INPUT_DDR_RANGE] \
	  requested_input_ddr_high_base [env_value_or_empty TPU_SOC_INPUT_DDR_HIGH_BASE] \
	  requested_input_ddr_high_range [env_value_or_empty TPU_SOC_INPUT_DDR_HIGH_RANGE] \
	  requested_output_ddr_base [env_value_or_empty TPU_SOC_OUTPUT_DDR_BASE] \
	  requested_output_ddr_range [env_value_or_empty TPU_SOC_OUTPUT_DDR_RANGE] \
	  requested_output_ddr_high_base [env_value_or_empty TPU_SOC_OUTPUT_DDR_HIGH_BASE] \
	  requested_output_ddr_high_range [env_value_or_empty TPU_SOC_OUTPUT_DDR_HIGH_RANGE] \
	  actual_axi_data_width $actual_axi_data_width \
  actual_ram_data_width $actual_ram_data_width \
  actual_ram_a_local_addr_width $actual_ram_a_local_addr_width \
  actual_ram_b_local_addr_width $actual_ram_b_local_addr_width \
  actual_ram_d_local_addr_width $actual_ram_d_local_addr_width \
  actual_pe_size $actual_pe_size \
  actual_int8_dot_factor $actual_int8_dot_factor \
  actual_op_tile_n $actual_op_tile_n \
  actual_op_tile_k $actual_op_tile_k \
  actual_ab_dual_input $actual_ab_dual_input \
  actual_cmat_d_full_word_pack $actual_cmat_d_full_word_pack \
  actual_cmat_streaming_c_adder $actual_cmat_streaming_c_adder \
  actual_c_m_axi_data_width $actual_m_axi_data_width \
  routed_wns_ns $timing_wns \
  routed_tns_ns $timing_tns \
  routed_failing_endpoints $timing_failing_endpoints \
  sha256_bit [sha256_or_empty $bit_out] \
  sha256_bin [sha256_or_empty $bin_out] \
  sha256_hwh [sha256_or_empty $hwh_out]]
write_manifest $manifest_out $manifest_data

puts "Formal TPU SoC bitstream build finished for project: $project_name"
puts "Project directory: $build_dir"
puts "Build jobs: $build_jobs"
puts "Vivado tool threads: $tool_threads"
puts "Bitstream: $bit_out"
puts "Hardware handoff: $hwh_out"
puts "Bitstream bin: $bin_out"
puts "Timing summary: $timing_summary_out"
puts "Manifest: $manifest_out"
if {[file exists $ltx_out]} {
  puts "Debug probes: $ltx_out"
}
