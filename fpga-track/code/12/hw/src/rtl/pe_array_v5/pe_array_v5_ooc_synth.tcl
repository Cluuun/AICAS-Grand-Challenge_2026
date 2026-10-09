set rtl_dir [file normalize [file dirname [info script]]]
set repo_dir [file normalize [file join $rtl_dir ../../rtl_ip_repo/pe_array_v5]]

read_verilog [list \
    [file join $rtl_dir pe_mac_i8_v5.v] \
    [file join $rtl_dir pe_array_32x32_v5.v] \
    [file join $rtl_dir pe_array_axis_v5.v] \
]
read_xdc [file join $rtl_dir pe_array_v5.xdc]

synth_design -top pe_array_axis_v5 -part xck26-sfvc784-2LV-c -mode out_of_context
report_utilization -file [file join $repo_dir pe_array_v5_utilization.txt]
report_timing_summary -file [file join $repo_dir pe_array_v5_timing_summary.txt]
