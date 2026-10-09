set script_dir [file normalize [file dirname [info script]]]
set create_tcl [file normalize [file join $script_dir "create_pynq_zu_tpu_pspl_native_iftest_bd.tcl"]]
set build_dir [file normalize [file join $script_dir "build_tpu_pspl_native_iftest_bd"]]
set project_name "pynq_zu_tpu_pspl_native_iftest"
set bd_name "tpu_pspl_native_iftest"
set build_jobs "auto"
set tool_threads 1

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

  # High-throughput mode: use the full core count for run-level parallelism and
  # keep only a small memory reserve so large builds do not thrash the machine.
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

if {$argc > 1} {
  set build_dir [file normalize [lindex $argv 1]]
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
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs $build_jobs
wait_on_run impl_1

set impl_bit [file normalize [file join $build_dir "${project_name}.runs" "impl_1" "${bd_name}_wrapper.bit"]]
set hwh_src [file normalize [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name "hw_handoff" "${bd_name}.hwh"]]
set output_dir [file normalize [file join $build_dir "export"]]
set bit_out [file normalize [file join $output_dir "${bd_name}_wrapper.bit"]]
set hwh_out [file normalize [file join $output_dir "${bd_name}_wrapper.hwh"]]
set bin_out [file normalize [file join $output_dir "${bd_name}_wrapper.bin"]]

file mkdir $output_dir
file copy -force $impl_bit $bit_out
if {[file exists $hwh_src]} {
  file copy -force $hwh_src $hwh_out
}
write_cfgmem -force -format bin -interface SMAPx32 -loadbit "up 0x0 $bit_out" $bin_out

puts "Native interface-test bitstream build finished for project: $project_name"
puts "Project directory: $build_dir"
puts "Build jobs: $build_jobs"
puts "Vivado tool threads: $tool_threads"
puts "Bitstream: $bit_out"
puts "Hardware handoff: $hwh_out"
puts "Bitstream bin: $bin_out"
