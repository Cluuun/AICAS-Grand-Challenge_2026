# FMAX-02: after default phys_opt, ask Vivado to replicate the MUX high-fanout nets.
# Guarded with catch so unsupported options do not abort the run.

if {[info exists ::env(VLM_REPO_ROOT)] && $::env(VLM_REPO_ROOT) ne ""} {
  set repo_root [file normalize $::env(VLM_REPO_ROOT)]
} else {
  set repo_root [file normalize [file join [pwd] "../.."]]
}
set report_dir [file join $repo_root "vivado/reports"]
file mkdir $report_dir

proc fmax02_mux_nets {} {
  set patterns [list \
    {*inst_mux*do_buffer_merge_2*address1*} \
    {*inst_mux*do_buffer_merge_2*addr*} \
    {*inst_mux*do_buffer_merge_2*select_ln32*} \
    {*inst_mux*do_buffer_merge_2*select*} \
  ]
  set nets [list]
  foreach pattern $patterns {
    foreach net [get_nets -hier -quiet $pattern] {
      lappend nets $net
    }
  }
  return [lsort -unique $nets]
}

set before_report [file join $report_dir "fmax02_high_fanout_pre_force_phys_opt.rpt"]
if {[catch {report_high_fanout_nets -load_types -max_nets 100 -file $before_report} msg]} {
  puts "VLM_FMAX02_REPORT_HIGH_FANOUT_PHYS_FAILED=$msg"
}

set mux_nets [fmax02_mux_nets]
puts "VLM_FMAX02_PHYS_MATCHED_NETS=[llength $mux_nets]"
foreach net $mux_nets {
  puts "VLM_FMAX02_PHYS_NET=$net"
}

if {[llength $mux_nets] > 0} {
  if {[catch {phys_opt_design -force_replication_on_nets $mux_nets} msg]} {
    puts "VLM_FMAX02_FORCE_REPLICATION_FAILED=$msg"
  } else {
    puts "VLM_FMAX02_FORCE_REPLICATION_APPLIED=1"
  }
}

set after_report [file join $report_dir "fmax02_high_fanout_post_force_phys_opt.rpt"]
if {[catch {report_high_fanout_nets -load_types -max_nets 100 -file $after_report} msg]} {
  puts "VLM_FMAX02_REPORT_HIGH_FANOUT_POST_FAILED=$msg"
}
