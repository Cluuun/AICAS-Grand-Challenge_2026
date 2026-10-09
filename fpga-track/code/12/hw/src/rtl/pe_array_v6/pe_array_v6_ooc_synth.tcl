set rtl_dir [file normalize [file dirname [info script]]]
set repo_dir [file normalize [file join $rtl_dir ../../rtl_ip_repo/pe_array_v6]]
file mkdir $repo_dir

read_verilog [list \
    [file join $rtl_dir pe_mac_i8_v6.v] \
    [file join $rtl_dir pe_array_8x128_v6.v] \
    [file join $rtl_dir pe_array_axis_v6.v] \
]
read_xdc [file join $rtl_dir pe_array_v6.xdc]

synth_design -top pe_array_axis_v6 -part xck26-sfvc784-2LV-c -mode out_of_context
report_utilization -file [file join $repo_dir pe_array_v6_utilization.txt]
report_timing_summary -file [file join $repo_dir pe_array_v6_timing_summary.txt]
