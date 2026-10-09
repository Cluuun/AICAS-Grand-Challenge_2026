/*
 * kernel_attn_block.cpp — Tiling Attention + 56 URAM L1.5 Cache + RoPE BRAM LUT
 *
 * 严格遵循 kv260_maximum_optimized.md §3.3:
 * - 56 URAM 组建 2MB 片上环形锁存缓存 (L1.5 Cache)
 * - RoPE: 16 BRAM cos/sin 查找表, 32 独立只读端口, II=1 流式注入
 * - AXI4 双缓冲权重预取: 16 BRAM (Ping 8 + Pong 8)
 * - Softmax ROM: 16 BRAM TDP, 32 独立读端口, 2048 深度
 * - QK^T 分块矩阵乘法 + Block Softmax + PV 矩阵乘法
 *
 * 资源:
 * URAM: 56 (L1.5 KV Cache, 16 Mbit 净容量)
 * BRAM: 16 (cos/sin ROM) + 16 (AXI4 双缓冲) + 16 (Softmax ROM) = 48
 * DSP: 30 (注意力分数计算)
 * LUT: ~28K, FF: ~42K
 *
 * 时序: 200MHz 计算时钟
 */

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

// ============================================================
// 数据类型
// ============================================================
typedef ap_int<8>    hls_int8;
typedef ap_int<16>   hls_int16;
typedef ap_int<32>   hls_int32;
typedef ap_uint<16>  hls_uint16;
typedef ap_fixed<16,4> hls_fixed16;
typedef ap_fixed<32,8> hls_fixed32;

// 【修改点 1】：引入离线预计算好的常量查找表头文件
// 该文件中包含了 const hls_int8 cos_lut[2048] 和 sin_lut[2048] 的定义
#include "rope_lut.h"

// ============================================================
// 参数
// ============================================================
#define TILE_SIZE          32
#define HEAD_DIM_MAX       1024
#define SEQ_LEN_MAX        4096
#define EXP_LUT_SIZE       4096     // 11-bit 索引空间 (§3.3)

// URAM L1.5 Cache: 64 URAM blocks (~2.3MB), 匹配 head_dim=64 (SmolVLM2-500M)
#define URAM_CACHE_LINES    64
#define URAM_LINE_DEPTH     1024

// ============================================================
// RoPE 旋转位置编码 — BRAM 固化 cos/sin LUT (§3.3)
// ============================================================
static void rope_apply_32ch(
    hls_int8 qk_in[32],
    hls_int8 qk_out[32],
    int position,
    int half_head_dim,
    const hls_int8 cos_lut[2048],
    const hls_int8 sin_lut[2048])
{
#pragma HLS INLINE

// #pragma HLS UNROLL 
    ROPE_32CH: for (int ch = 0; ch < 32; ch += 2) {
        int rot_idx = (ch / 2) % half_head_dim;
        int lut_idx = (position * (half_head_dim / 2) + rot_idx) & 0x7FF;

        hls_int8 cos_val = cos_lut[lut_idx];
        hls_int8 sin_val = sin_lut[lut_idx];

        hls_int8 real_val = qk_in[ch];
        hls_int8 imag_val = qk_in[ch + 1];

        // 2D 旋转: (real + i*imag) * (cos + i*sin)
        hls_int16 rot_real = (hls_int16)real_val * (hls_int16)cos_val
                           - (hls_int16)imag_val * (hls_int16)sin_val;
        hls_int16 rot_imag = (hls_int16)real_val * (hls_int16)sin_val
                           + (hls_int16)imag_val * (hls_int16)cos_val;

        qk_out[ch]     = (hls_int8)(rot_real >> 7);
        qk_out[ch + 1] = (hls_int8)(rot_imag >> 7);
    }
}

// ============================================================
// Block Softmax — 16 BRAM TDP ROM (§3.3)
// ============================================================
static void block_softmax(
    hls::stream<hls_int32> & score_stream,
    hls::stream<hls_int8>  & attn_stream,
    const hls_uint16         exp_lut[EXP_LUT_SIZE],
    int tile_rows, int tile_cols)
{
#pragma HLS INLINE off

    // Score 缓冲区 (BRAM, dim=2 block partition)
    hls_int32 score_buf[TILE_SIZE][TILE_SIZE];
#pragma HLS BIND_STORAGE variable=score_buf type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=score_buf dim=2 block factor=8

    // 读入 score
    
    READ_SCORES: for (int i = 0; i < tile_rows; i++) {
       
        for (int j = 0; j < tile_cols; j++) {
            #pragma HLS PIPELINE II=1 
            score_buf[i][j] = score_stream.read();
        }
    }
    
    hls_fixed16 exp_vals[TILE_SIZE];
#pragma HLS BIND_STORAGE variable=exp_vals type=RAM_2P impl=LUTRAM

    // 逐行 softmax
    ROW_SOFTMAX: for (int i = 0; i < tile_rows; i++) {
        // 找最大值 (数值稳定性)
        hls_int32 max_val = score_buf[i][0];
        
      
        FIND_MAX: for (int j = 1; j < tile_cols; j++) {
            #pragma HLS PIPELINE II=1  
            if (score_buf[i][j] > max_val) max_val = score_buf[i][j];
        }

        // Exp + Sum
        hls_fixed32 exp_sum = 0;

        // #pragma HLS PIPELINE II=1
        EXP_LOOP: for (int j = 0; j < tile_cols; j++) {

            hls_int32 diff = score_buf[i][j] - max_val;

            // CLZ + 移位归一化 → 11-bit 索引
            hls_uint16 lut_idx;
            if (diff > 1023)       lut_idx = 2047;
            else if (diff < -1024) lut_idx = 0;
            else                   lut_idx = (hls_uint16)(diff + 1024);

            hls_uint16 eq = exp_lut[lut_idx];
            hls_fixed16 ev;
            ev.range(15,0) = eq;
            exp_vals[j] = ev;
            exp_sum += (hls_fixed32)ev;
        }

        // 计算倒数 (显式定点类型转换转换，避免 ambiguous 错误)
        hls_fixed32 inv_sum = (hls_fixed32)1.0 / (exp_sum + (hls_fixed32)0.000001); 
        
        // 归一化
       
        NORM_LOOP: for (int j = 0; j < tile_cols; j++) {
            #pragma HLS PIPELINE II=1
            hls_fixed16 nv = exp_vals[j] * inv_sum;
            attn_stream.write((hls_int8)((nv * 127) >> 8));
        }
    }
}

// ============================================================
// 顶层函数
// ============================================================
extern "C" {
void kernel_attn_block(
    volatile hls_int8  * input_q,       // m_axi gmem0: [B×H×S×D]
    volatile hls_int8  * input_k,       // m_axi gmem1
    volatile hls_int8  * input_v,       // m_axi gmem0
    volatile hls_int8  * output,        // m_axi gmem1: [B×H×S×D]
    int seq_len, int num_heads, int head_dim, int batch_size,
    hls_int8             op_flags       // s_axilite
);
}

void kernel_attn_block(
    volatile hls_int8  * input_q,
    volatile hls_int8  * input_k,
    volatile hls_int8  * input_v,
    volatile hls_int8  * output,
    int seq_len, int num_heads, int head_dim, int batch_size,
    hls_int8             op_flags
) {
#pragma HLS INTERFACE m_axi port=input_q  bundle=gmem0 offset=slave depth=SEQ_LEN_MAX*HEAD_DIM_MAX
#pragma HLS INTERFACE m_axi port=input_k  bundle=gmem1 offset=slave depth=SEQ_LEN_MAX*HEAD_DIM_MAX
#pragma HLS INTERFACE m_axi port=input_v  bundle=gmem0 offset=slave depth=SEQ_LEN_MAX*HEAD_DIM_MAX
#pragma HLS INTERFACE m_axi port=output   bundle=gmem1 offset=slave depth=SEQ_LEN_MAX*HEAD_DIM_MAX
#pragma HLS INTERFACE s_axilite port=seq_len   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=num_heads bundle=ctrl
#pragma HLS INTERFACE s_axilite port=head_dim  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=batch_size bundle=ctrl
#pragma HLS INTERFACE s_axilite port=op_flags bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // ── URAM L1.5 KV Cache ──
    static hls_int8 uram_kv_cache[URAM_CACHE_LINES][URAM_LINE_DEPTH];
#pragma HLS BIND_STORAGE variable=uram_kv_cache type=RAM_1P impl=URAM
#pragma HLS ARRAY_PARTITION variable=uram_kv_cache dim=1 block factor=16

    // 【修改点 2】：绑定外部全局只读 const 查找表的硬件存储核心为 ROM_2P
    // 移除了先前的局部静态变量声明与逻辑标志位，杜绝了运行时的写操作通路
#pragma HLS BIND_STORAGE variable=cos_lut type=ROM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=sin_lut type=ROM_2P impl=BRAM

    // 【修改点 3】：彻底删除了先前的 if (!rope_lut_initialized) { INIT_ROPE_LUT ... } 循环块
    // 无需在芯片内部动态计算复杂的 pow, cos, sin 浮点函数，大幅节省了 DSP 与布线资源。

    // ── Softmax Exp LUT ──
    hls_uint16 exp_lut[EXP_LUT_SIZE];
#pragma HLS BIND_STORAGE variable=exp_lut type=RAM_1P impl=BRAM
// #pragma HLS ARRAY_PARTITION variable=exp_lut cyclic factor=4 dim=1

    // 初始化 Exp LUT (生产环境由 PS 通过 AXI-Lite 预加载)
   
    INIT_EXP_LUT: for (int i = 0; i < EXP_LUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1 
        float x = (float)((int)i - 1024) / 128.0f;
        exp_lut[i] = (hls_uint16)(hls::exp(x) * 256.0f);
    }

    // ── 流通道 (BRAM 深 FIFO) ──
    hls::stream<hls_int32> score_s("score");
#pragma HLS STREAM variable=score_s depth=1024
#pragma HLS BIND_STORAGE variable=score_s type=FIFO impl=BRAM

    hls::stream<hls_int8> attn_s("attn");
#pragma HLS STREAM variable=attn_s depth=1024
#pragma HLS BIND_STORAGE variable=attn_s type=FIFO impl=BRAM

    hls::stream<hls_int8> result_s("result");
#pragma HLS STREAM variable=result_s depth=4096
#pragma HLS BIND_STORAGE variable=result_s type=FIFO impl=BRAM


    hls_int8 q_tile_buf[TILE_SIZE][HEAD_DIM_MAX];
    hls_int8 k_tile_buf[TILE_SIZE][HEAD_DIM_MAX];
#pragma HLS BIND_STORAGE variable=q_tile_buf type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=k_tile_buf type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=q_tile_buf dim=2 block factor=16
#pragma HLS ARRAY_PARTITION variable=k_tile_buf dim=2 block factor=16

    bool use_kv_cache = (op_flags & 0x01) != 0;
    int seq_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;

    BATCH_LOOP: for (int b = 0; b < batch_size; b++) {
    HEAD_LOOP:  for (int h = 0; h < num_heads; h++) {
        int h_off = ((b * num_heads + h) * seq_len * head_dim);

        if (use_kv_cache) {
            STAGE_KV: for (int s = 0; s < seq_len && s < URAM_LINE_DEPTH; s++) {
                
                // #pragma HLS DEPENDENCE variable=input_k inter false
               
                for (int d = 0; d < head_dim && d < URAM_CACHE_LINES; d++) {
                    #pragma HLS PIPELINE II=1 
                    uram_kv_cache[d][s] = input_k[h_off + s * head_dim + d];
                }
            }
        }



        // ── Tiling Attention ──
        TILE_I: for (int ti = 0; ti < seq_tiles; ti++) {
        TILE_J: for (int tj = 0; tj < seq_tiles; tj++) {
            int i_start = ti * TILE_SIZE;
            int j_start = tj * TILE_SIZE;
            int i_lim = (i_start + TILE_SIZE < seq_len) ? TILE_SIZE : (seq_len - i_start);
            int j_lim = (j_start + TILE_SIZE < seq_len) ? TILE_SIZE : (seq_len - j_start);

            // hls_int8 raw_q_row[HEAD_DIM_MAX];
            // hls_int8 raw_k_row[HEAD_DIM_MAX];
            hls_int8 raw_q_row[HEAD_DIM_MAX];
            hls_int8 raw_k_row[HEAD_DIM_MAX];
            hls_int8 chunk_in[32];
            hls_int8 chunk_out[32];
            #pragma HLS BIND_STORAGE variable=raw_q_row type=RAM_2P impl=BRAM
            #pragma HLS BIND_STORAGE variable=raw_k_row type=RAM_2P impl=BRAM
            #pragma HLS BIND_STORAGE variable=chunk_in type=RAM_2P impl=BRAM
            #pragma HLS BIND_STORAGE variable=chunk_out type=RAM_2P impl=BRAM


            // 1. 载入 Q 并实时应用 RoPE
            LOAD_AND_ROPE_Q: for (int i = 0; i < i_lim; i++) {
                // hls_int8 raw_q_row[HEAD_DIM_MAX];
                int q_rope_base = h_off + (i_start + i) * head_dim;
                
                FETCH_Q: for (int d = 0; d < head_dim; d++) {
                    #pragma HLS PIPELINE II=1 
                    raw_q_row[d] = input_q[q_rope_base + d];
                }
                
                APPLY_ROPE_Q: for (int ch = 0; ch < head_dim; ch += 32) {
                    // hls_int8 chunk_in[32], chunk_out[32];
                    //    花

                    // #pragma HLS PIPELINE II=1                
                    for (int c = 0; c < 32; c++) { 
                        #pragma HLS UNROLL
                        chunk_in[c] = raw_q_row[ch + c]; 
                    }
                    // 此处直接引用了通过头文件引入的全局只读查找表
                    rope_apply_32ch(chunk_in, chunk_out, i_start + i, head_dim / 2, cos_lut, sin_lut);
                    

                    // #pragma HLS PIPELINE II=1                
                    for (int c = 0; c < 32; c++) {
                        #pragma HLS UNROLL factor=16
                        q_tile_buf[i][ch + c] = chunk_out[c]; 
                    }
                }
            }

            // 2. 载入 K 并实时应用 RoPE
            LOAD_AND_ROPE_K: for (int j = 0; j < j_lim; j++) {
                // hls_int8 raw_k_row[HEAD_DIM_MAX];
                int k_rope_base = h_off + (j_start + j) * head_dim;
                
                
                FETCH_K: for (int d = 0; d < head_dim; d++) {
                    #pragma HLS PIPELINE II=1                
// #pragma HLS PIPELINE II=1    HUA
                    if (use_kv_cache) {
                        raw_k_row[d] = uram_kv_cache[d][j_start + j];
                    } else {
                        raw_k_row[d] = input_k[k_rope_base + d];
                    }
                }
                APPLY_ROPE_K: for (int ch = 0; ch < head_dim; ch += 32) {
                    // hls_int8 chunk_in[32], chunk_out[32];
                    // 花
                    // #pragma HLS PIPELINE II=1
                    
                    for (int c = 0; c < 32; c++) { 
                        #pragma HLS UNROLL                         
                        chunk_in[c] = raw_k_row[ch + c]; 
                    }

                    rope_apply_32ch(chunk_in, chunk_out, j_start + j, head_dim / 2, cos_lut, sin_lut);
                    

                    // #pragma HLS PIPELINE II=1                     
                    for (int c = 0; c < 32; c++) { 
                        #pragma HLS UNROLL factor=16
                        k_tile_buf[j][ch + c] = chunk_out[c]; 
                    }
                }
            }

            
            // ── QK^T 点积 ──
            QK_DOT: for (int i = 0; i < i_lim; i++) {

                for (int j = 0; j < j_lim; j++) {
                    // hls_int8 raw_q_row[HEAD_DIM_MAX];
                    hls_int32 dot = 0;
// #pragma HLS PIPELINE II=2
                    // #pragma HLS UNROLL factor=16
                   
                    QK_REDUCE: for (int d = 0; d < HEAD_DIM_MAX; d++) {
                        #pragma HLS PIPELINE II=1                     
// #pragma HLS UNROLL factor=8
                        hls_int8 qv = q_tile_buf[i][d];
                        hls_int8 kv = k_tile_buf[j][d];
                        // dot += (hls_int32)qv * (hls_int32)kv;
                        raw_q_row[j] = qv * kv;
                    }
                    
                    for (int c = 0; c < HEAD_DIM_MAX; c++) { 

                        dot += (hls_int32)raw_q_row[c];
                    }

                    score_s.write(dot >> 3);
                }
            }

            // ── Block Softmax ──
            block_softmax(score_s, attn_s, exp_lut, i_lim, j_lim);

            // ── 建立片上缓存防止死锁 ──
            hls_int8 local_attn[TILE_SIZE][TILE_SIZE];
#pragma HLS BIND_STORAGE variable=local_attn type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=local_attn dim=2 block factor=8

            // #pragma HLS PIPELINE II=1 
            DRAIN_ATTN_STREAM: for (int i = 0; i < i_lim; i++) {
                

                // #pragma HLS UNROLL factor=8                
                for (int j = 0; j < j_lim; j++) {
                    #pragma HLS PIPELINE II=1
                    local_attn[i][j] = attn_s.read();
                }
            }

            // ── PV 矩阵乘 ──
            PV_MATMUL: for (int i = 0; i < i_lim; i++) {
// #pragma HLS PIPELINE II=1

                for (int d = 0; d < head_dim; d++) {
// #pragma HLS PIPELINE II=1
                    hls_int32 acc = 0;
                    
                    // #pragma HLS UNROLL 

                    PV_REDUCE: for (int j = 0; j < j_lim; j++) {
                        #pragma HLS PIPELINE II=1                         
                        hls_int8 av = local_attn[i][j];
                        hls_int8 vv = input_v[h_off + (j_start + j) * head_dim + d];
                        // acc += (hls_int32)av * (hls_int32)vv;
                        raw_k_row[j] = av * vv;
                    }
                    
                    for (int c = 0; c < head_dim; c++) { 

                        acc += (hls_int32)raw_k_row[c];
                    }

                    result_s.write((hls_int8)(acc >> 8));
                }
            }

            // ── 逐 Tile 写回 ──
            TILE_WRITEBACK: for (int i = 0; i < i_lim; i++) {
                int tile_write_base = h_off + (i_start + i) * head_dim;
               
                for (int d = 0; d < head_dim; d++) {
                    #pragma HLS PIPELINE II=1 
                    output[ tile_write_base + d] = result_s.read();
                }
            }
        }}
    }}
}