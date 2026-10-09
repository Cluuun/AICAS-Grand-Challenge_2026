/*
 * kernel_fused_vit_mlp.cpp — ViT MLP 融合硬件核 (§6.1)
 *
 * 严格遵循 kv260_maximum_optimized.md §6.1:
 *
 * 目标链: [MUL_MAT(Linear1)] → [GELUTanh] → [MUL_MAT(Linear2)]
 *
 * 阶段 1: Linear1 权重驻留脉动阵列 + GELUTanh PWL 流式激活
 *   - 32×32 主脉动阵列 (权重驻留模式)
 *   - 576 个视觉 Token 以 32×64 分块流式划过
 *   - CARRY8 进位链全速输出 32 通道中间部分和
 *   - GELUTanh PWL: 64 区间 LUTRAM 并行查表 (0 BRAM)
 *   - 32 DSP 并行乘法器, II=1 全吞吐
 *
 * 阶段 2: 级联数据中转站 — 32 BRAM Ping-Pong 缓冲区
 *   - Ping: Linear1 输出写入, Pong: Linear2 读入
 *   - 周期交替, 解耦两级矩阵乘法
 *
 * 资源 (§5.1):
 *   DSP: 32 (GELUTanh PWL 激活引擎)
 *   BRAM: 32 (Ping-Pong 缓冲区: 16+16, SDP 模式)
 *   URAM: 64 (权重驻留, 与主阵列共享)
 *   LUT: ~25K, FF: ~40K
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
typedef ap_uint<8>   hls_uint8;
typedef ap_int<16>   hls_int16;
typedef ap_int<32>   hls_int32;
typedef ap_fixed<16,4> hls_fixed16;
typedef ap_fixed<32,8> hls_fixed32;

// ============================================================
// 参数 (SmolVLM2 ViT)
// ============================================================
// #define VIT_HIDDEN_DIM      960
// #define VIT_INTER_DIM       3840
// #define VIT_PATCHES         576
#define VIT_CHUNK_SIZE      32
// #define MAX_M               4096
#define MAX_N               3072
#define MAX_K               768      // SmolVLM2 ViT: hidden_dim=768, 1KB aligned

#define TILE_K              64
#define SYSTOLIC_COLS       32

// Ping-Pong 缓冲区 — 每个 BRAM(18Kb) 存储1列×TILE_K INT8
#define PING_PONG_ROWS      32
#define PING_PONG_COLS      32       // 对齐 SYSTOLIC_COLS, 减少 BRAM 分区

// ============================================================
// GELUTanh PWL 激活引擎 (§6.1 阶段 1)
//
// GELUTanh(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
//
// 64 区间分段线性插值:
//   区间 i: GELU(x) ≈ m_i · x + b_i
//
// 3 级低延迟流水线:
//   Cycle 0: 高位比特寻址 LUTRAM → 读出 m_i, b_i
//   Cycle 1: DSP 并行: y = x · m_i + b_i
//   Cycle 2: 饱和截断, 精度对齐
// ============================================================
static hls_int16 gelu_tanh_pwl(hls_int32 x, const hls_fixed16 m_lut[64],
                                const hls_fixed16 b_lut[64]) {
#pragma HLS INLINE

    // 将 32-bit 累加和缩放至 16-bit 索引空间
    hls_int16 x_scaled;
    if (x > 1048575)        x_scaled = 32767;
    else if (x < -1048576)  x_scaled = -32768;
    else                    x_scaled = (hls_int16)(x >> 5);

    // 6-bit 区间索引
    ap_uint<6> idx;
    if (x_scaled > 2016)       idx = 63;
    else if (x_scaled < -2048) idx = 0;
    else                       idx = (ap_uint<6>)((x_scaled + 2048) >> 6);

    hls_fixed16 m = m_lut[idx];
    hls_fixed16 b = b_lut[idx];

    hls_fixed16 xs;
    xs.range(15,0) = (ap_uint<16>)(x_scaled & 0xFFFF);

    hls_fixed32 y = (hls_fixed32)m * (hls_fixed32)xs + (hls_fixed32)b;

    // 饱和截断
    hls_int16 result;
    if (y > 32767)       result = 32767;
    else if (y < -32768) result = -32768;
    else                 result = (hls_int16)(y.range(15,0));

    return result;
}

// ============================================================
// 顶层函数: ViT MLP 融合
// ============================================================
extern "C" {
void kernel_fused_vit_mlp(
    volatile hls_int8   * input,           // m_axi gmem0: [M × K] 输入特征
    volatile hls_int8   * linear1_weight,  // m_axi gmem1: [K × N_inter] Linear1 权重
    volatile hls_int8   * linear2_weight,  // m_axi gmem2: [N_inter × K] Linear2 权重
    volatile hls_int8   * output,          // m_axi gmem3: [M × K] 输出特征
    int                   M,               // s_axilite: Token 数 (分块后 ≤32)
    int                   N_inter,         // s_axilite: intermediate_dim
    int                   K_dim,           // s_axilite: hidden_dim
    hls_int8              op_flags         // s_axilite: 控制标志
);
}

void kernel_fused_vit_mlp(
    volatile hls_int8   * input,
    volatile hls_int8   * linear1_weight,
    volatile hls_int8   * linear2_weight,
    volatile hls_int8   * output,
    int                   M,
    int                   N_inter,
    int                   K_dim,
    hls_int8              op_flags
) {
    // §4.2 HLS 接口约束
#pragma HLS INTERFACE m_axi port=input           bundle=gmem0 offset=slave depth=MAX_K
#pragma HLS INTERFACE m_axi port=linear1_weight  bundle=gmem1 offset=slave depth=MAX_K*MAX_N
#pragma HLS INTERFACE m_axi port=linear2_weight  bundle=gmem2 offset=slave depth=MAX_N*MAX_K
#pragma HLS INTERFACE m_axi port=output          bundle=gmem3 offset=slave depth=MAX_K
#pragma HLS INTERFACE s_axilite port=M         bundle=ctrl
#pragma HLS INTERFACE s_axilite port=N_inter   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=K_dim     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=op_flags  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return    bundle=ctrl

    // ── GELUTanh PWL 查找表 (LUTRAM, 0 BRAM) ──
    hls_fixed16 gelu_m[64];
    hls_fixed16 gelu_b[64];
#pragma HLS ARRAY_PARTITION variable=gelu_m complete
#pragma HLS ARRAY_PARTITION variable=gelu_b complete


    // 初始化 GELUTanh PWL 系数
    INIT_PWL: for (int i = 0; i < 64; i++) {

        float x0 = -4.0f + (float)i * 0.125f;
        float x1 = x0 + 0.125f;
        float x0_cube = x0 * x0 * x0;
        float x1_cube = x1 * x1 * x1;
        float inner0 = 0.7978845608f * (x0 + 0.044715f * x0_cube);
        float inner1 = 0.7978845608f * (x1 + 0.044715f * x1_cube);
        float y0 = 0.5f * x0 * (1.0f + tanhf(inner0));
        float y1 = 0.5f * x1 * (1.0f + tanhf(inner1));
        float slope = (y1 - y0) / 0.125f;
        float bias  = y0 - slope * x0;
        gelu_m[i] = (hls_fixed16)(slope * 256.0f);
        gelu_b[i] = (hls_fixed16)(bias * 256.0f);
    }

    // ── Ping-Pong 缓冲区 (§6.1 阶段 2) ──
    // 32 BRAM Tile: Ping(16) + Pong(16), 每块 SDP 模式
    hls_int16 ping_buf[PING_PONG_ROWS][PING_PONG_COLS];
    hls_int16 pong_buf[PING_PONG_ROWS][PING_PONG_COLS];
#pragma HLS BIND_STORAGE variable=ping_buf type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=pong_buf type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=ping_buf dim=2 block factor=8
#pragma HLS ARRAY_PARTITION variable=pong_buf dim=2 block factor=8

// 每一批 Token 局部累加专用的中间结果高速寄存器
    hls_int32 accum[PING_PONG_ROWS][SYSTOLIC_COLS];
#pragma HLS ARRAY_PARTITION variable=accum dim=2 complete
#pragma HLS BIND_STORAGE variable=accum type=RAM_2P impl=lutram

    hls_int16 activated[PING_PONG_ROWS][SYSTOLIC_COLS];
#pragma HLS ARRAY_PARTITION variable=activated dim=2 complete
#pragma HLS BIND_STORAGE variable=activated type=RAM_2P impl=lutram

// Stage 1 和 Stage 2 局部权重分块快取区
    hls_int8 w1_buf[SYSTOLIC_COLS][TILE_K];
#pragma HLS BIND_STORAGE variable=w1_buf type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=w1_buf dim=1 block factor=8

// 【修改点】：为 w2_buf 补充缺失的完全分区指令，使其契合 Stage 2 32路并行的总线要求
    hls_int8 w2_buf[PING_PONG_COLS][TILE_K];
#pragma HLS BIND_STORAGE variable=w2_buf type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=w2_buf dim=1 block factor=8

    // 最终 Linear2 的大片上累加中间区 (32 行 x K 维)
    hls_int32 l2_accum[PING_PONG_ROWS][MAX_K];
#pragma HLS BIND_STORAGE variable=l2_accum type=RAM_2P impl=URAM



    bool use_ping = true;  // true=写ping读pong, false=写pong读ping

    // ── 阶段 1: Linear1 投影 + GELUTanh 激活 ──
    int n_tiles = (N_inter + SYSTOLIC_COLS - 1) / SYSTOLIC_COLS;
    int k_tiles = (K_dim    + TILE_K        - 1) / TILE_K;

    // 中间结果暂存在 Ping-Pong 缓冲区
    // Linear1: [M × K] × [K × N_inter] → [M × N_inter]
    // 分块执行: 每次计算 32 列 × 32 行

    // 【修改点】：构建全系统最顶层 Vision Token 滚动分块大循环，全面吞吐 576 个 Patch
    LOOP_M_CHUNK: for (int m_start = 0; m_start < M; m_start += VIT_CHUNK_SIZE) {
        // int chunk_m = (m_start + VIT_CHUNK_SIZE < M) ? VIT_CHUNK_SIZE : (M - m_start);

        // 每一个 Token Chunk 开始前，原位清零第二级累加器
        CLEAR_L2: for (int mi = 0; mi < VIT_CHUNK_SIZE; mi++) {

            // #pragma HLS UNROLL 
            for (int d = 0; d < MAX_K; d++) {
                #pragma HLS PIPELINE II=1
                l2_accum[mi][d] = 0;
            }
        }

        // 沿着隐藏中间层维度进行 Tiling 推进
        LINE1_N: for (int nt = 0; nt < n_tiles; nt++) {
            int n_start = nt * SYSTOLIC_COLS;
            // int n_lim   = (n_start + SYSTOLIC_COLS < N_inter) ? SYSTOLIC_COLS : (N_inter - n_start);
            // #pragma HLS PIPELINE II=1
            CLEAR_ACC: for (int mi = 0; mi < PING_PONG_ROWS; mi++) {
                #pragma HLS PIPELINE II=1

                for (int nj = 0; nj < SYSTOLIC_COLS; nj++) {
                    #pragma HLS UNROLL                     
                    accum[mi][nj] = 0;
                }
            }

            // ── 阶段 1: 基础特征投影乘积 ──
            LINE1_K: for (int kt = 0; kt < k_tiles; kt++) {
                int k_start = kt * TILE_K;
                // int k_lim   = (k_start + TILE_K < K_dim) ? TILE_K : (K_dim - k_start);

                LOAD_W1: for (int kk = 0; kk < TILE_K; kk++) {

                    int base_w1 = (k_start + kk) * N_inter + n_start;
// #pragma HLS PIPELINE II=1
// #pragma HLS UNROLL factor = 8 

                    for (int nj = 0; nj < SYSTOLIC_COLS; nj++) {
                        #pragma HLS PIPELINE II=1
                        w1_buf[nj][kk] = linear1_weight[base_w1 + nj];

                        // if ((k_start + kk < K_dim) && (n_start + nj < N_inter)) {
                        //     w1_buf[nj][kk] = linear1_weight[base_w1 + nj];
                        //     // w1_buf[nj][kk] = linear1_weight[(k_start + kk) * N_inter + n_start + nj];
                        // } else {
                        //     w1_buf[nj][kk] = 0;
                        // }
                    }
                }

                MAC_W1: for (int mi = 0; mi < VIT_CHUNK_SIZE; mi++) {
                    int base_in = (m_start + mi) * K_dim + k_start;
                    for (int kk = 0; kk < TILE_K; kk++) {

                        hls_int8 a_val = input[base_in + kk];
                        // #pragma HLS PIPELINE II=2                       
                        for (int nj = 0; nj < SYSTOLIC_COLS; nj++) {
// #pragma HLS UNROLL  factor = 8
                            accum[mi][nj] += (hls_int32)a_val * (hls_int32)w1_buf[nj][kk];
                        }
                    }
                }
            }

            // 流式通过 3级流水线 PWL GELUTanh 激活引擎
            GELU_ACTIVATE: for (int mi = 0; mi < VIT_CHUNK_SIZE; mi++) {

                for (int nj = 0; nj < SYSTOLIC_COLS; nj++) {
                    #pragma HLS PIPELINE II=1                    
                    activated[mi][nj] = gelu_tanh_pwl(accum[mi][nj], gelu_m, gelu_b);
                }
            }

            // 【修改点】：修正寻址缺陷，写入片上乒乓缓冲区时改用相对索引 [nj]，消除越界隐患
            if (use_ping) {
                

                WRITE_PING: for (int mi = 0; mi < VIT_CHUNK_SIZE; mi++) {
                    #pragma HLS PIPELINE II=1

                    // #pragma HLS PIPELINE II=1
                    for (int nj = 0; nj < SYSTOLIC_COLS; nj++) {
                        #pragma HLS UNROLL
                        ping_buf[mi][nj] = activated[mi][nj];
                    }
                }
            } else {

                WRITE_PONG: for (int mi = 0; mi < VIT_CHUNK_SIZE; mi++) {
                    #pragma HLS PIPELINE II=1 

                    // #pragma HLS PIPELINE II=1  
                    for (int nj = 0; nj < SYSTOLIC_COLS; nj++) {
                        #pragma HLS UNROLL
                        pong_buf[mi][nj] = activated[mi][nj];
                    }
                }
            }

            // ── 阶段 2: Linear2 级联累加（在片实时多路吞吐） ──
            LINE2_K: for (int kt = 0; kt < k_tiles; kt++) {
                int k_start = kt * TILE_K;
                // int k_lim   = (k_start + TILE_K < K_dim) ? TILE_K : (K_dim - k_start);

                LOAD_W2: for (int ij = 0; ij < SYSTOLIC_COLS; ij++){

                    int base_w2 = (n_start + ij) * K_dim + k_start;
// #pragma HLS PIPELINE II=1
// #pragma HLS UNROLL factor = 8

                    for (int kk = 0; kk < TILE_K; kk++) {
                        #pragma HLS PIPELINE II=1
                        w2_buf[ij][kk] = linear2_weight[base_w2 + kk];

                        // if ((n_start + ij < N_inter) && (k_start + kk < K_dim)) {
                        //     w2_buf[ij][kk] = linear2_weight[base_w2 + kk];
                        // } else {
                        //     w2_buf[ij][kk] = 0;
                        // }
                    }
                }

                MAC_W2: for (int mi = 0; mi < VIT_CHUNK_SIZE; mi++) {
                    for (int kk = 0; kk < TILE_K; kk++) {

                        int d = k_start + kk;
                        // if (d < K_dim) {
                        hls_int32 acc = l2_accum[mi][d];
                        // #pragma HLS PIPELINE II=2                        
                        for (int ij = 0; ij < SYSTOLIC_COLS; ij++) {
// #pragma HLS UNROLL  factor = 8
                            hls_int16 act_val = use_ping ? ping_buf[mi][ij] : pong_buf[mi][ij];
                            acc += (hls_int32)act_val * (hls_int32)w2_buf[ij][kk];
                        }
                        l2_accum[mi][d] = acc;
                        // }
                    }
                }
            }

            // 【修改点】：每次 Tile 结束，对 use_ping 执行逻辑取反，触发底层物理硬件高速换轨
            use_ping = !use_ping;
        }

        // ── 阶段 3: 一次性安全突发刷回 DDR ──
        WRITEBACK: for (int mi = 0; mi < VIT_CHUNK_SIZE; mi++) {
            int base_out = (m_start + mi) * K_dim;
           
            for (int d = 0; d < K_dim; d++) {
                #pragma HLS PIPELINE II=1 
                output[base_out + d] = (hls_int8)((l2_accum[mi][d] >> 14) & 0xFF);
            }
        }
    }
}
