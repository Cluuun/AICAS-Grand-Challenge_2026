set rtl_dir [file normalize [file dirname [info script]]]
set repo_dir [file normalize [file join $rtl_dir ../../rtl_ip_repo/pe_array_v4]]

read_verilog [list \
    [file join $rtl_dir pe_mac_i8.v] \
    [file join $rtl_dir pe_array_32x32.v] \
    [file join $rtl_dir pe_array_axis.v] \
]

synth_design -top pe_array_axis -part xck26-sfvc784-2LV-c -mode out_of_context
report_utilization -file [file join $repo_dir pe_array_v4_utilization.txt]
