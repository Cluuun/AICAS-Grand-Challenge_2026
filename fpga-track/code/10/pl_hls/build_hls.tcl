# ============================================================
# Vitis HLS 综合脚本 — KV260 全融合算子套件 (v4: 32×32 + SwiGLU/ViTMLP/RMSNorm)
#
# 器件: xck26-sfvc784-2LV-c (Kria K26 SoM, ZU5EV)
# 时钟: 200MHz 计算域 (period=5.0ns), 100MHz 控制域 (period=10.0ns)
#
# 严格遵循 kv260_maximum_optimized.md §4:
#   - 控制面: s_axilite 独立 bundle (门铃接收端)
#   - 数据面: m_axi 64-bit 寻址, 独立 AXI Master 端口
#   - 最大突发长度 64 (16 字节对齐)
#   - FRP (Flushable Pipeline) 模式
#
# 用法: vitis_hls -f build_hls.tcl
# ============================================================

open_project kv260_accel_prj -reset

# 添加所有源文件
add_files kernel_gemm_int8.cpp      -cflags "-std=c++14"
add_files kernel_attn_block.cpp     -cflags "-std=c++14"
add_files kernel_fused_swiglu.cpp   -cflags "-std=c++14"
add_files kernel_fused_vit_mlp.cpp  -cflags "-std=c++14"
add_files kernel_rmsnorm.cpp        -cflags "-std=c++14"
add_files pl_controller.cpp         -cflags "-std=c++14"

set kv260_part "xck26-sfvc784-2LV-c"
set kv260_fb   "xczu7ev-ffvf1517-2-e"

# ============================================================
# Helper: apply common solution config
# ============================================================
proc config_kv260_solution {} {
    config_compile -pipeline_loops 64
    config_array_partition -complete_threshold 2
    config_interface -m_axi_addr64=true
    config_interface -m_axi_max_read_burst_length=64
    config_interface -m_axi_max_write_burst_length=64
}

# ============================================================
# Solution 1: kernel_gemm_int8 (32×32 脉动阵列)
#   DSP: 1024 | URAM: 64 | BRAM: 4
# ============================================================
set_top kernel_gemm_int8
open_solution "solution_gemm" -reset

if {[catch {set_part $kv260_part}]} { set_part $kv260_fb }
create_clock -period 5.0 -name clk_compute
config_kv260_solution
config_interface -m_axi_offset slave -m_axi_alignment_byte_size 64

puts "\[INFO\] Synthesizing kernel_gemm_int8 (32x32 systolic, 1024 DSP, DSP48E2 2-way)..."
csynth_design
export_design -rtl vhdl -format ip_catalog \
    -description "KV260 32×32 Systolic GEMM (DSP48E2 2-way, URAM-Backed)" \
    -vendor "user" -library "kv260" -version "1.0"
puts "\[INFO\] kernel_gemm_int8 done."

# ============================================================
# Solution 2: kernel_attn_block (56 URAM + RoPE BRAM + Softmax ROM)
#   URAM: 56 | BRAM: 48 | DSP: 30
# ============================================================
set_top kernel_attn_block
open_solution "solution_attn" -reset

if {[catch {set_part $kv260_part}]} { set_part $kv260_fb }
create_clock -period 5.0 -name clk_compute
config_kv260_solution
config_interface -m_axi_offset slave -m_axi_alignment_byte_size 64

puts "\[INFO\] Synthesizing kernel_attn_block (56 URAM, RoPE BRAM LUT, Softmax ROM)..."
csynth_design
export_design -rtl vhdl -format ip_catalog \
    -description "KV260 Attention (Dual-Mode: URAM KV Cache/No-Cache, CORDIC RoPE, Softmax ROM)" \
    -vendor "user" -library "kv260" -version "1.0"
puts "\[INFO\] kernel_attn_block done."

# ============================================================
# Solution 3: kernel_fused_swiglu (§6.2)
#   DSP: 64 (SiLU PWL 32 + Hadamard 32) | URAM: 56 | BRAM: 16
# ============================================================
set_top kernel_fused_swiglu
open_solution "solution_swiglu" -reset

if {[catch {set_part $kv260_part}]} { set_part $kv260_fb }
create_clock -period 5.0 -name clk_compute
config_kv260_solution
config_interface -m_axi_offset slave -m_axi_alignment_byte_size 64

puts "\[INFO\] Synthesizing kernel_fused_swiglu (PWL SiLU 32 DSP + Hadamard 32 DSP)..."
csynth_design
export_design -rtl vhdl -format ip_catalog \
    -description "KV260 SwiGLU Fusion (Gate+Up+SiLU PWL+Hadamard+Down, URAM L1.5)" \
    -vendor "user" -library "kv260" -version "1.0"
puts "\[INFO\] kernel_fused_swiglu done."

# ============================================================
# Solution 4: kernel_fused_vit_mlp (§6.1)
#   DSP: 32 (GELUTanh PWL) | BRAM: 32 (Ping-Pong) | URAM: 64
# ============================================================
set_top kernel_fused_vit_mlp
open_solution "solution_vitmlp" -reset

if {[catch {set_part $kv260_part}]} { set_part $kv260_fb }
create_clock -period 5.0 -name clk_compute
config_kv260_solution
config_interface -m_axi_offset slave -m_axi_alignment_byte_size 64

puts "\[INFO\] Synthesizing kernel_fused_vit_mlp (GELUTanh PWL 32 DSP + Ping-Pong BRAM)..."
csynth_design
export_design -rtl vhdl -format ip_catalog \
    -description "KV260 ViT MLP Fusion (Linear1+GELUTanh PWL+Linear2, Ping-Pong BRAM)" \
    -vendor "user" -library "kv260" -version "1.0"
puts "\[INFO\] kernel_fused_vit_mlp done."

# ============================================================
# Solution 5: kernel_rmsnorm (§6.3)
#   BRAM: 8 (Ping-Pong 线存) | DSP: 4
# ============================================================
set_top kernel_rmsnorm
open_solution "solution_rmsnorm" -reset

if {[catch {set_part $kv260_part}]} { set_part $kv260_fb }
create_clock -period 5.0 -name clk_compute
config_kv260_solution
config_interface -m_axi_offset slave -m_axi_alignment_byte_size 64

puts "\[INFO\] Synthesizing kernel_rmsnorm (8 BRAM Ping-Pong line buffer, CORDIC RSQRT)..."
csynth_design
export_design -rtl vhdl -format ip_catalog \
    -description "KV260 RMSNorm Hardware Closure (8 BRAM Ping-Pong, CORDIC RSQRT, II=1)" \
    -vendor "user" -library "kv260" -version "1.0"
puts "\[INFO\] kernel_rmsnorm done."

# ============================================================
# Solution 6: pl_controller v4 (描述符环预取引擎 + 门铃 FSM)
#   7 m_axi + 2 s_axilite | 100MHz
# ============================================================
set_top pl_controller
open_solution "solution_ctrl" -reset

if {[catch {set_part $kv260_part}]} { set_part $kv260_fb }
create_clock -period 10.0 -name clk_ctrl
config_kv260_solution
config_interface -m_axi_offset slave

puts "\[INFO\] Synthesizing pl_controller v4 (Descriptor Ring Fetch Engine, Doorbell FSM)..."
csynth_design
export_design -rtl vhdl -format ip_catalog \
    -description "KV260 Controller v4 (Descriptor Ring Fetch, Doorbell FSM, 7×m_axi, 10μs WDT)" \
    -vendor "user" -library "kv260" -version "1.0"
puts "\[INFO\] pl_controller done."

close_project
puts "\[INFO\] =============================================="
puts "\[INFO\] All 6 IP cores synthesized successfully."
puts "\[INFO\] Project: kv260_accel_prj"
puts "\[INFO\] =============================================="
exit
