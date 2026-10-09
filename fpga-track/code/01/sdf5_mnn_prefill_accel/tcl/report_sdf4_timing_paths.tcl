set dcp_path [lindex $argv 0]
set out_dir [lindex $argv 1]
file mkdir $out_dir
open_checkpoint $dcp_path
report_timing -delay_type max -max_paths 80 -nworst 4 -sort_by slack -file "$out_dir/setup_top80.rpt"
report_timing -delay_type min -max_paths 40 -nworst 4 -sort_by slack -file "$out_dir/hold_top40.rpt"
report_timing_summary -delay_type max -max_paths 80 -file "$out_dir/setup_summary_top80.rpt"
report_timing_summary -delay_type min -max_paths 40 -file "$out_dir/hold_summary_top40.rpt"
close_design
