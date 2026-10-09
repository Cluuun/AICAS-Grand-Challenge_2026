/**
 * AICAS 2026 - 融合多头注意力HLS实现
 * 支持GQA (Grouped Query Attention)
 * 目标: KV260 ZU5EV, 300MHz
 */

#include "attention_engine.h"
#include <hls_math.h>

// ============================================================
// Softmax查找表 (exp近似)
// ============================================================
const int LUT_SIZE = 256;
fp16_t exp_lut[LUT_SIZE];

// 初始化exp查找表
void init_exp_lut() {
    for (int i = 0; i < LUT_SIZE; i++) {
        // 映射到 [-8, 0] 范围
        fp16_t x = (fp16_t)-8.0f + ((fp16_t)i * (fp16_t)8.0f / (fp16_t)LUT_SIZE);
        fp16_t x2 = x * x;
        fp16_t x3 = x2 * x;
        fp16_t x4 = x3 * x;
        exp_lut[i] = (fp16_t)1.0f + x + x2/(fp16_t)2.0f + x3/(fp16_t)6.0f + x4/(fp16_t)24.0f;
    }
}

// exp近似函数
fp16_t exp_approx(fp16_t x) {
    #pragma HLS PIPELINE II=1

    // 饱和处理
    if (x > 0) return 1.0;
    if (x < -8.0) return 0.0;

    // 查表
    int addr = (int)((x + (fp16_t)8.0f) * (fp16_t)LUT_SIZE / (fp16_t)8.0f);
    if (addr < 0) addr = 0;
    if (addr >= LUT_SIZE) addr = LUT_SIZE - 1;

    return exp_lut[addr];
}

// ============================================================
// 融合多头注意力核心
// ============================================================
void attention_engine_core(
    // 输入
    fp16_t* Q,              // [seq_len, NUM_HEADS, HEAD_DIM]
    fp16_t* K,              // [kv_len, NUM_KV_HEADS, HEAD_DIM]
    fp16_t* V,              // [kv_len, NUM_KV_HEADS, HEAD_DIM]
    fp16_t* output,         // [seq_len, NUM_HEADS, HEAD_DIM]

    // 配置
    int seq_len,
    int kv_len,

    // 流式接口
    hls::stream<fp16_t>& q_stream,
    hls::stream<fp16_t>& k_stream,
    hls::stream<fp16_t>& v_stream,
    hls::stream<fp16_t>& out_stream
) {
    #pragma HLS INTERFACE m_axi port=Q depth=131072 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=K depth=32768 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=V depth=32768 bundle=gmem2
    #pragma HLS INTERFACE m_axi port=output depth=131072 bundle=gmem3

    #pragma HLS INTERFACE axis port=q_stream
    #pragma HLS INTERFACE axis port=k_stream
    #pragma HLS INTERFACE axis port=v_stream
    #pragma HLS INTERFACE axis port=out_stream

    // 初始化exp查找表
    init_exp_lut();

    // ============================================================
    // 第1阶段: QK^T计算
    // ============================================================
    fp16_t scores[MAX_SEQ_LEN];

    QK_COMPUTE:
    for (int h = 0; h < NUM_HEADS; h++) {
        int kv_group = h / HEADS_PER_GROUP;

        QK_SEQ:
        for (int q_pos = 0; q_pos < seq_len; q_pos++) {
            fp16_t max_score = -1e6;

            QK_KV:
            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
                #pragma HLS PIPELINE II=2

                fp16_t acc = 0;

                // QK^T计算 (带缩放)
                DOT_PRODUCT:
                for (int d = 0; d < HEAD_DIM; d++) {
                    #pragma HLS UNROLL
                    acc += Q[q_pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d]
                         * K[kv_pos * NUM_KV_HEADS * HEAD_DIM + kv_group * HEAD_DIM + d];
                }

                // 缩放
                scores[kv_pos] = acc / (fp16_t)sqrtf((float)HEAD_DIM);

                // 更新最大值
                if (scores[kv_pos] > max_score) {
                    max_score = scores[kv_pos];
                }
            }

            // ============================================================
            // 第2阶段: Softmax (融合在QK循环中)
            // ============================================================
            fp16_t sum_exp = 0;

            SOFTMAX:
            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
                #pragma HLS PIPELINE II=1

                // 减去最大值 (数值稳定性)
                scores[kv_pos] -= max_score;

                // exp近似
                scores[kv_pos] = exp_approx(scores[kv_pos]);

                sum_exp += scores[kv_pos];
            }

            // 归一化
            NORMALIZE:
            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
                #pragma HLS PIPELINE II=1
                scores[kv_pos] /= sum_exp;
            }

            // ============================================================
            // 第3阶段: Score @ V
            // ============================================================
            AV_COMPUTE:
            for (int d = 0; d < HEAD_DIM; d++) {
                #pragma HLS PIPELINE II=2

                fp16_t acc = 0;

                AV_KV:
                for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
                    #pragma HLS UNROLL
                    acc += scores[kv_pos]
                         * V[kv_pos * NUM_KV_HEADS * HEAD_DIM + kv_group * HEAD_DIM + d];
                }

                // 写入输出
                output[q_pos * NUM_HEADS * HEAD_DIM + h * HEAD_DIM + d] = acc;
            }
        }
    }
}

// ============================================================
// 顶层函数
// ============================================================
void attention_engine(
    fp16_t* Q,
    fp16_t* K,
    fp16_t* V,
    fp16_t* output,
    int seq_len,
    int kv_len
) {
    #pragma HLS INTERFACE m_axi port=Q depth=131072 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=K depth=32768 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=V depth=32768 bundle=gmem2
    #pragma HLS INTERFACE m_axi port=output depth=131072 bundle=gmem3

    #pragma HLS INTERFACE s_axilite port=seq_len
    #pragma HLS INTERFACE s_axilite port=kv_len
    #pragma HLS INTERFACE s_axilite port=return

    hls::stream<fp16_t> q_stream, k_stream, v_stream, out_stream;

    // 调用核心函数
    attention_engine_core(Q, K, V, output, seq_len, kv_len,
                         q_stream, k_stream, v_stream, out_stream);
}
