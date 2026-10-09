# ===========================================================
# Implementation-stage CDC constraints for XDMA/PCIe clocks
# Executed via impl_1 STEPS.OPT_DESIGN.TCL.PRE
# ===========================================================

proc _apply_async_group_if_nonempty {grp_a grp_b tag} {
    if {[llength $grp_a] > 0 && [llength $grp_b] > 0} {
        set_clock_groups -asynchronous -group $grp_a -group $grp_b
        puts "INFO: applied async clock group: $tag"
    } else {
        puts "INFO: skip async clock group ($tag), empty object(s)"
    }
}

set sys_ref_clks [get_clocks -quiet {sys_clk pcie_refclk_clk_p}]
set axi_clks     [get_clocks -quiet {axi_aclk shell_region_xdma_0_0_axi_aclk}]
set pipe_clks    [get_clocks -quiet pipe_clk]
set gty_tx_clks  [get_clocks -quiet -regexp {^GTYE4_CHANNEL_TXOUTCLK.*$}]

# Main async relations seen in failing timing reports
_apply_async_group_if_nonempty $sys_ref_clks $axi_clks  "sys/ref <-> axi"
_apply_async_group_if_nonempty $sys_ref_clks $pipe_clks "sys/ref <-> pipe"
_apply_async_group_if_nonempty $sys_ref_clks $gty_tx_clks "sys/ref <-> gty_txout"
_apply_async_group_if_nonempty $pipe_clks $gty_tx_clks "pipe <-> gty_txout"

# Limit false-path to the first synchronizer stage only
set xdma_sync_stage0_d_pins [get_pins -quiet -hierarchical -filter {NAME =~ */xdma_0/inst/pcie4c_ip_i/inst/*sync_vec[*].sync_cell_i/sync_reg[0]/D}]
set xdma_cdc_src_clks [concat $sys_ref_clks $axi_clks $pipe_clks $gty_tx_clks]
if {[llength $xdma_sync_stage0_d_pins] > 0 && [llength $xdma_cdc_src_clks] > 0} {
    set_false_path -from $xdma_cdc_src_clks -to $xdma_sync_stage0_d_pins
    puts "INFO: applied xdma sync stage0 false-path"
} else {
    puts "INFO: skip xdma sync stage0 false-path, empty object(s)"
}

