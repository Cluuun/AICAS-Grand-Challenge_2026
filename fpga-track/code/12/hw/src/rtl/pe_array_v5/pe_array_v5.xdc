# pe_array_v5.xdc --- timing constraints for 300/333/375 MHz closure.
# Sourced when pe_array_axis_v5 is the top of an OOC synthesis run.

# Keep PE OOC synthesis honest for the 375 MHz target. Broadcast fanout is
# constrained in RTL with keep/max_fanout attributes because the PE accumulators
# are intentionally absorbed into DSP registers, so acc_r_reg pin names are not
# stable after synthesis.
create_clock -period 2.667 -name pe_array_v5_ooc_clk [get_ports ap_clk]
