# ===========================================================
# KV260 v3 route-closure implementation directives and reports.
#
# Use as implementation run hooks:
#   STEPS.OPT_DESIGN.TCL.PRE       -> this file
#   STEPS.WRITE_BITSTREAM.TCL.POST -> this file
#
# The commands are guarded by the current implementation step, so the same
# file can be attached to both hooks. It keeps the v3 BD topology unchanged
# and biases implementation toward routability at 300 MHz.
# ===========================================================

set _v3_step ""
catch {set _v3_step [get_property STEPS.CURRENT_STEP [current_run]]}
if {$_v3_step eq ""} {
    catch {set _v3_step [get_property CURRENT_STEP [current_run]]}
}

proc _v3_report_dir {} {
    set run_dir [get_property DIRECTORY [current_run]]
    if {$run_dir eq ""} {
        set run_dir [pwd]
    }
    set out_dir [file normalize [file join $run_dir v3_route_reports]]
    file mkdir $out_dir
    return $out_dir
}

proc _v3_emit_route_reports {tag} {
    set out_dir [_v3_report_dir]
    puts "INFO: writing v3 route-closure reports to $out_dir ($tag)"
    catch {report_utilization -hierarchical -file [file join $out_dir "${tag}_utilization_hier.rpt"]}
    catch {report_timing_summary -delay_type max -max_paths 50 -file [file join $out_dir "${tag}_timing_summary.rpt"]}
    catch {report_route_status -file [file join $out_dir "${tag}_route_status.rpt"]}
    catch {report_design_analysis -congestion -file [file join $out_dir "${tag}_congestion.rpt"]}
}

# Pre-opt hook: set directives/params before the implementation run starts.
if {$_v3_step eq "" || [string match -nocase *OPT_DESIGN* $_v3_step]} {
    puts "INFO: applying KV260 v3 congestion-oriented implementation setup"

    # Favor routability over aggressive timing replication in a near-full K26.
    set_param place.enableGlobalHoldIter true
    set_param route.enableGlobalHoldIter true

    set run [current_run]
    catch {set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore $run}
    catch {set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE SSI_SpreadLogic_high $run}
    catch {set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true $run}
    catch {set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $run}
    catch {set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AlternateCLBRouting $run}
    catch {set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true $run}
    catch {set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $run}

    # Emit an early utilization snapshot after opt if this hook is evaluated
    # inside an opened design checkpoint.
    catch {_v3_emit_route_reports pre_route}
}

# Post-bitstream hook: collect the reports requested by the v3 closure plan.
if {[string match -nocase *WRITE_BITSTREAM* $_v3_step] || [string match -nocase *ROUTE_DESIGN* $_v3_step]} {
    _v3_emit_route_reports post_route
}
