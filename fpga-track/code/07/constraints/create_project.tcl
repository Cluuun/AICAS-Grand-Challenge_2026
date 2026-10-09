# AICAS 2026 - Vivado工程创建脚本

# 创建项目
create_project aicas_accelerator ./aicas_accelerator -part xczu5ev-sfvc784-1-i -force

# 设置项目属性
set_property target_language Verilog [current_project]
set_property simulator_language Verilog [current_project]
set_property board_part xilinx.com:kv260:part0:1.4 [current_project]

# 设置IP仓库
set_property ip_repo_paths ./hls/ip_repo [current_project]
update_ip_catalog

# 添加源文件
add_files -norecurse ./rtl/attention_engine.sv
add_files -norecurse ./rtl/kv_cache_manager.sv
add_files -norecurse ./rtl/dma_controller.sv

# 添加约束文件
add_files -fileset constrs_1 -norecurse ./constraints/top.xdc

# 设置顶层模块
set_property top attention_engine [current_fileset]

# 添加HLS IP
add_files -norecurse ./hls/ip_repo/matmul_int8/xilinx_com_hls_matmul_int8_1_0/hdl/vhdl/matmul_int8_accel.vhd
add_files -norecurse ./hls/ip_repo/attention_engine/xilinx_com_hls_attention_engine_1_0/hdl/vhdl/attention_engine.vhd
add_files -norecurse ./hls/ip_repo/softmax_lut/xilinx_com_hls_softmax_lut_1_0/hdl/vhdl/softmax_lut.vhd

# 设置仿真顶层
set_property top tb_attention [get_filesets sim_1]

puts "Vivado工程创建完成！"
