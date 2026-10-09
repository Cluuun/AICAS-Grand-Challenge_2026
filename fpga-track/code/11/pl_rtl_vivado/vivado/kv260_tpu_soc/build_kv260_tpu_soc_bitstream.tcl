set script_dir [file normalize [file dirname [info script]]]
set repo_root [file normalize [file join $script_dir ".." ".."]]
set build_dir [file normalize [file join $script_dir "build_kv260_tpu_soc_bd"]]
set matrix_d_base "0xA0000000"
set build_jobs "12"

proc set_default_env {name value} {
  if {![info exists ::env($name)] || [string trim $::env($name)] eq ""} {
    set ::env($name) $value
  }
}

if {$argc > 0} {
  set build_dir [file normalize [lindex $argv 0]]
}
if {$argc > 1} {
  set matrix_d_base [lindex $argv 1]
}
if {$argc > 2} {
  set build_jobs [lindex $argv 2]
}

set ::env(TPU_SOC_PROJECT_NAME) "kv260_tpu_soc"
set ::env(TPU_SOC_TARGET_PART) "xck26-sfvc784-2LV-c"
set ::env(TPU_SOC_BOARD_PART) "xilinx.com:kv260_som:part0:1.3"
set ::env(TPU_SOC_INPUT_PORT) "HP0"
set ::env(TPU_SOC_OUTPUT_PORT) "HPC0"
set ::env(TPU_SOC_ENABLE_SYSTEM_ILA) "0"
set_default_env TPU_SOC_AXI_DATA_WIDTH "128"
set_default_env TPU_SOC_RAM_DATA_WIDTH "128"
set_default_env TPU_SOC_PE_SIZE "8"
set_default_env TPU_SOC_REQUIRE_AXI_DATA_WIDTH $::env(TPU_SOC_AXI_DATA_WIDTH)
set_default_env TPU_SOC_REQUIRE_RAM_DATA_WIDTH $::env(TPU_SOC_RAM_DATA_WIDTH)
set_default_env TPU_SOC_REQUIRE_PE_SIZE $::env(TPU_SOC_PE_SIZE)
set_default_env TPU_SOC_DELIVERY_TAG "kv260_axi${::env(TPU_SOC_AXI_DATA_WIDTH)}_ram${::env(TPU_SOC_RAM_DATA_WIDTH)}_pe${::env(TPU_SOC_PE_SIZE)}_hpc0"

set argv [list $repo_root $build_dir $matrix_d_base $build_jobs]
set argc [llength $argv]

set build_script [file normalize [file join $script_dir ".." "pynq_zu_tpu_soc" "build_pynq_zu_tpu_soc_bitstream.tcl"]]
source $build_script
