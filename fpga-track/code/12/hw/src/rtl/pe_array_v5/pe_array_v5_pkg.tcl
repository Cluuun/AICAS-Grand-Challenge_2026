# pe_array_v5_pkg.tcl --- Vivado IP packaging for pe_array_axis_v5.
#
# Usage:  vivado -mode batch -source pe_array_v5_pkg.tcl
#
# Produces an IP-Catalog component at hw/src/rtl_ip_repo/pe_array_v5 with:
#   * vendor=aicas, library=user, name=pe_array_v5, version=1.0
#   * 4 AXI-Stream interfaces inferred from naming convention
#   * 1 clock + 1 sync active-low reset

set ip_dir [file normalize [file dirname [info script]]/../../rtl_ip_repo/pe_array_v5]
file mkdir $ip_dir
file mkdir [file join $ip_dir xgui]
file mkdir [file join $ip_dir src]

set proj_dir [file join $ip_dir _pkg_proj]
file delete -force $proj_dir
create_project pe_array_v5_pkg $proj_dir -part xck26-sfvc784-2LV-c -force

set rtl_dir [file normalize [file dirname [info script]]]
add_files -norecurse [list \
    [file join $rtl_dir pe_mac_i8_v5.v] \
    [file join $rtl_dir pe_array_32x32_v5.v] \
    [file join $rtl_dir pe_array_axis_v5.v] \
]

set_property top pe_array_axis_v5 [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $ip_dir \
    -vendor aicas -library user -taxonomy /UserIP \
    -import_files -force

set core [ipx::current_core]
set_property name        pe_array_v5 $core
set_property display_name {AICAS PE Array V5 32x32 INT8 MAC} $core
set_property description  {32x32 INT8 weight-broadcast PE array with AXIS feeders} $core
set_property version      1.0 $core

# Auto-infer AXIS interfaces from prefix
ipx::infer_bus_interfaces xilinx.com:interface:axis_rtl:1.0 $core

ipx::create_xgui_files $core
ipx::update_checksums   $core
ipx::save_core          $core

# Run Out-Of-Context synthesis to evaluate resource utilization (LUT/FF, etc.)
puts "Running Out-Of-Context synthesis for resource estimation..."
synth_design -top pe_array_axis_v5 -part xck26-sfvc784-2LV-c -mode out_of_context
create_clock -period 2.667 -name pe_array_v5_ooc_clk [get_ports ap_clk]
report_utilization -file [file join $ip_dir pe_array_v5_utilization.txt]
report_timing_summary -file [file join $ip_dir pe_array_v5_timing_summary.txt]

close_project
puts "pe_array_v5 packaged at: $ip_dir"
