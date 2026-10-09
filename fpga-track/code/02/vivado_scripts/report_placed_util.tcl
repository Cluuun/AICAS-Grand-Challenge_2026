# Report utilization from the latest placed checkpoint without changing the build project.
set repo_root [file normalize [file join [file dirname [info script]] ../..]]
set placed_dcp [file join $repo_root vivado/work/kv260_bd/vlm_kv260_bd.runs/impl_1/vlm_system_wrapper_placed.dcp]
set out_report [file join $repo_root vivado/reports/kv260_placed_util.rpt]

open_checkpoint $placed_dcp
report_utilization -file $out_report
