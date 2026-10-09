# AICAS 2026 - 融合注意力引擎HLS综合脚本
# 目标: KV260 ZU5EV, 300MHz

# 打开项目
open_project attention_engine_prj
set_top attention_engine

# 添加源文件
add_files attention_engine.cpp
add_files -tb tb_attention.cpp

# 设置目标器件
open_solution -reset solution1
set_part xczu5ev-sfvc784-1-i

# 设置时钟周期 (300MHz → 3.333ns)
create_clock -period 3.333 -name default

# 运行C模拟
# csim_design


# 运行综合
csynth_design

# 运行C/RTL联合仿真
# cosim_design

# 导出IP
export_design -format ip_catalog -display_name "Attention Engine Accelerator" \
    -description "Fused multi-head attention with GQA support for VLM inference"

exit
