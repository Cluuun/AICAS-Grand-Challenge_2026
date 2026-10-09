# pe_array_v4.xdc --- timing constraints for 300 MHz closure.
# Sourced when pe_array_axis is the top of an OOC synthesis run.

# Loosen the cross-quadrant register-to-register paths so the placer can
# absorb any small skew between quadrants. The pipeline registers added in
# pe_array_32x32 already break the long broadcast nets; this is a safety net.
set_max_delay -datapath_only -from [get_pins -hier -filter {NAME =~ */a_lo_q_reg*/C}] \
                                -to   [get_pins -hier -filter {NAME =~ */ROW*/COL*/u_pe/acc_r_reg*/D}] 3.333
set_max_delay -datapath_only -from [get_pins -hier -filter {NAME =~ */a_hi_q_reg*/C}] \
                                -to   [get_pins -hier -filter {NAME =~ */ROW*/COL*/u_pe/acc_r_reg*/D}] 3.333
set_max_delay -datapath_only -from [get_pins -hier -filter {NAME =~ */w_lo_q_reg*/C}] \
                                -to   [get_pins -hier -filter {NAME =~ */ROW*/COL*/u_pe/acc_r_reg*/D}] 3.333
set_max_delay -datapath_only -from [get_pins -hier -filter {NAME =~ */w_hi_q_reg*/C}] \
                                -to   [get_pins -hier -filter {NAME =~ */ROW*/COL*/u_pe/acc_r_reg*/D}] 3.333
