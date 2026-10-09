# AICAS 2026 - 顶层约束文件
# 目标: KV260 ZU5EV

# ============================================================
# 时钟约束
# ============================================================

# 主时钟 (300MHz)
create_clock -period 3.333 -name clk_fpga [get_ports clk]

# DDR时钟 (300MHz)
create_clock -period 3.333 -name clk_ddr [get_ports ddr_clk]

# ============================================================
# 输入/输出延迟
# ============================================================

# AXI-Stream输入延迟
set_input_delay -clock clk_fpga -max 2.0 [get_ports {s_axis_*}]
set_input_delay -clock clk_fpga -min 0.5 [get_ports {s_axis_*}]

# AXI-Stream输出延迟
set_output_delay -clock clk_fpga -max 2.5 [get_ports {m_axis_*}]
set_output_delay -clock clk_fpga -min 0.5 [get_ports {m_axis_*}]

# 控制信号延迟
set_input_delay -clock clk_fpga -max 1.0 [get_ports {reg_*}]
set_output_delay -clock clk_fpga -max 1.5 [get_ports {reg_*}]

# ============================================================
# 时钟域交叉
# ============================================================

# 异步时钟域
set_false_path -from [get_clocks clk_fpga] -to [get_clocks clk_ddr]
set_false_path -from [get_clocks clk_ddr] -to [get_clocks clk_fpga]

# ============================================================
# 多周期路径
# ============================================================

# 控制寄存器 (2周期)
set_multicycle_path -setup 2 -from [get_pins {reg_*[*]/C}] -to [get_pins {reg_*[*]/D}]
set_multicycle_path -hold 1 -from [get_pins {reg_*[*]/C}] -to [get_pins {reg_*[*]/D}]

# ============================================================
# 最大延迟约束
# ============================================================

# 关键路径延迟
set_max_delay 3.0 -from [get_pins {matmul_int8_accel_*/*}] -to [get_pins {matmul_int8_accel_*/*}]
set_max_delay 3.0 -from [get_pins {attention_engine_*/*}] -to [get_pins {attention_engine_*/*}]

# ============================================================
# 物理约束
# ============================================================

# 布局约束
create_pblock pblock_matmul
resize_pblock pblock_matmul -add {SLICE_X0Y0:SLICE_X63Y119}
add_cells_to_pblock pblock_matmul [get_cells -hierarchical -filter {NAME =~ *matmul*}]

create_pblock pblock_attention
resize_pblock pblock_attention -add {SLICE_X64Y0:SLICE_X127Y119}
add_cells_to_pblock pblock_attention [get_cells -hierarchical -filter {NAME =~ *attention*}]

# ============================================================
# 时钟属性
# ============================================================

set_property CLOCK_DEDICATED_ROUTE BACKBONE [get_nets clk_fpga]
set_property HIGH_FANOUT true [get_nets {rst_n}]
