set ip_dir [file normalize [file dirname [info script]]/../../rtl_ip_repo/pe_array_v6]
file mkdir $ip_dir
file mkdir [file join $ip_dir xgui]
file mkdir [file join $ip_dir src]

set proj_dir [file join $ip_dir _pkg_proj]
file delete -force $proj_dir
create_project pe_array_v6_pkg $proj_dir -part xck26-sfvc784-2LV-c -force

set rtl_dir [file normalize [file dirname [info script]]]
add_files -norecurse [list \
    [file join $rtl_dir pe_mac_i8_v6.v] \
    [file join $rtl_dir pe_array_8x128_v6.v] \
    [file join $rtl_dir pe_array_axis_v6.v] \
]

set_property top pe_array_axis_v6 [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $ip_dir \
    -vendor aicas -library user -taxonomy /UserIP \
    -import_files -force

set core [ipx::current_core]
set_property name        pe_array_v6 $core
set_property display_name {AICAS PE Array V6 8x128 INT8 MAC} $core
set_property description  {8x128 INT8 PE array with decode-friendly N128 AXIS feeder} $core
set_property version      1.0 $core

ipx::infer_bus_interfaces xilinx.com:interface:axis_rtl:1.0 $core
ipx::create_xgui_files $core
ipx::update_checksums   $core
ipx::save_core          $core

puts "Running Out-Of-Context synthesis for resource estimation..."
synth_design -top pe_array_axis_v6 -part xck26-sfvc784-2LV-c -mode out_of_context
create_clock -period 3.333 -name pe_array_v6_ooc_clk [get_ports ap_clk]
report_utilization -file [file join $ip_dir pe_array_v6_utilization.txt]
report_timing_summary -file [file join $ip_dir pe_array_v6_timing_summary.txt]

close_project
puts "pe_array_v6 packaged at: $ip_dir"
