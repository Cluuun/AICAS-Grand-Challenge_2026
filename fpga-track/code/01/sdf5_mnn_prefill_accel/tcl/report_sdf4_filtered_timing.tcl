set dcp_path [lindex $argv 0]
set out_dir [lindex $argv 1]
file mkdir $out_dir
open_checkpoint $dcp_path

set fp [open "$out_dir/filtered_timing_summary.txt" w]
puts $fp "SDF4 filtered timing query"
puts $fp ""

proc dump_paths {fp title paths} {
  puts $fp $title
  set idx 0
  foreach p $paths {
    incr idx
    set sp [get_property STARTPOINT_PIN $p]
    set ep [get_property ENDPOINT_PIN $p]
    set slack [get_property SLACK $p]
    set group [get_property GROUP $p]
    puts $fp [format "%02d slack=%s group=%s" $idx $slack $group]
    puts $fp "   from=$sp"
    puts $fp "   to  =$ep"
  }
  puts $fp ""
}

set all_setup [get_timing_paths -delay_type max -max_paths 200 -sort_by slack -filter {SLACK < 0}]
dump_paths $fp "Top setup failing paths" $all_setup

set non_acc_setup [get_timing_paths -delay_type max -max_paths 80 -sort_by slack -filter {SLACK < 0 && ENDPOINT_PIN !~ *area_accMem*}]
dump_paths $fp "Setup failing paths excluding area_accMem endpoints" $non_acc_setup

set non_sdf_setup [get_timing_paths -delay_type max -max_paths 80 -sort_by slack -filter {SLACK < 0 && ENDPOINT_PIN !~ *Sdf4DebugAxi* && STARTPOINT_PIN !~ *Sdf4DebugAxi*}]
dump_paths $fp "Setup failing paths excluding Sdf4DebugAxi endpoints/startpoints" $non_sdf_setup

set hold_paths [get_timing_paths -delay_type min -max_paths 80 -sort_by slack]
dump_paths $fp "Top hold paths" $hold_paths

close $fp
close_design
