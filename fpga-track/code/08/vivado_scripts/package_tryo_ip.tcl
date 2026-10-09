# Package the generated Tryo accelerator decoder core as a Vivado IP.
#
# Run after:
#   sbt "runMain generate_tryo_top"
#   python3 to_vivado.py
#
# Expected source files in this submission package:
#   ../verilog/TryoTop.v
#   ../verilog/TryoTop_bb.v
#   ../verilog/*.dat

set script_dir [file dirname [file normalize [info script]]]
set package_dir [file normalize [file join $script_dir ".."]]
set rtl_dir    [file normalize [file join $package_dir "verilog"]]
set ip_root    [file normalize [file join $script_dir "ip_repo" "TRYO_LLM_1_0"]]
set part_name  "xck26-sfvc784-2LV-c"

if {[info exists ::env(TRYO_PART)] && $::env(TRYO_PART) ne ""} {
  set part_name $::env(TRYO_PART)
}

if {![file exists [file join $rtl_dir "TryoTop.v"]]} {
  error "Missing $rtl_dir/TryoTop.v."
}
if {![file exists [file join $rtl_dir "TryoTop_bb.v"]]} {
  error "Missing $rtl_dir/TryoTop_bb.v."
}

create_project -force package_tryo_ip [file join $script_dir "package_tryo_ip_project"] -part $part_name
set_property BOARD_PART xilinx.com:kv260_som:part0:1.4 [current_project]
add_files -norecurse [glob -nocomplain [file join $rtl_dir "*.v"]]
add_files -norecurse [glob -nocomplain [file join $rtl_dir "*.dat"]]
set_property top TryoTop [current_fileset]
update_compile_order -fileset sources_1

file delete -force $ip_root
ipx::package_project -root_dir $ip_root -vendor user.org -library user -taxonomy /UserIP -import_files -force

set core [ipx::current_core]
set_property name TRYO_LLM $core
set_property display_name {Tryo Decoder Core} $core
set_property description {Reconstructed Tryo text decoder integration IP} $core
set_property version 1.0 $core

ipx::infer_bus_interface clk xilinx.com:signal:clock_rtl:1.0 $core
ipx::infer_bus_interface resetn xilinx.com:signal:reset_rtl:1.0 $core
ipx::associate_bus_interfaces -clock clk -reset resetn $core

ipx::update_checksums $core
ipx::save_core $core

puts "Packaged TRYO_LLM IP at $ip_root"
