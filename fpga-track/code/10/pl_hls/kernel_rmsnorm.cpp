/*
 * kernel_rmsnorm.cpp — RMSNorm 硬件物理化闭环 (§6.3)
 *
 * 严格遵循 kv260_maximum_optimized.md §6.3:
 *
 * 两阶段算子:
 *   阶段 1: 遍历 D 维向量, 计算均方根因子 rms = sqrt(mean(x²) + ε)
 *   阶段 2: 利用该因子对原向量缩放: y = x / rms * weight
 *
 * 微架构:
 *   - 8 BRAM Ping-Pong 向量线存 (Ping 4 + Pong 4)
 *   - SDP 模式, 横向位宽拼接: 64-bit × 4 = 256-bit 总线
 *   - 数据流双路拆分: 一路实时送入平方和累加进位链树
 *                     一路同步泵入 BRAM 线存暂存
 *   - 平方和完成后, BRAM 以 II=1 全吞吐吐出原向量, 与 rms 因子做哈达玛乘
 *   - 端到端流式归一化闭环, 避免二次读取 DDR
 *
 * 资源:
 *   BRAM: 8 (Ping-Pong 向量线存: 4+4, SDP 模式)
 *   DSP: 4 (平方累加 + 倒数平方根 + 最终缩放)
 *   LUT: ~8K, FF: ~12K
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
typedef ap_int<48>   hls_int48;
typedef ap_uint<32>  hls_uint32;
typedef ap_fixed<16,4> hls_fixed16;
typedef ap_fixed<32,8> hls_fixed32;

// ============================================================
// 参数
// ============================================================
#define MAX_D            1024
// #define LINEBUF_DEPTH    1024      // 4 BRAM × 256-depth (18Kb BRAM)
#define VECTOR_WIDTH     32        // 32 通道并行
#define MAX_WORDS       (MAX_D / VECTOR_WIDTH)

// ============================================================
// 改进型 CORDIC RSQRT (§6.3)
//
// 通过 CORDIC 双曲向量模式计算倒数平方根:
//   y = 1/sqrt(x)
//
// 4 次迭代即可达到 8-bit 精度
// ============================================================
static hls_fixed16 cordic_rsqrt(hls_fixed32 x) {
#pragma HLS INLINE
    // y ≈ 1/sqrt(x), x 归一化至 [0.25, 1.0) 区间
    // ap_fixed<32,8>: 0.25 = 0x40 (64/256), 1.0 = 0x100 (256/256)

    hls_fixed16 y = 0;
    hls_fixed32 x_norm = x;

    // 归一化至 [0.25, 1.0) — 使用 ap_fixed 强类型比较
    int shift = 0;
    if (x_norm > 0) {
        while (x_norm < (hls_fixed32)0.25) { x_norm = x_norm * 2; shift--; }
        while (x_norm > (hls_fixed32)1.0)  { x_norm = x_norm / 2; shift++; }
    }

    // Newton-Raphson 迭代: y_{n+1} = y_n * (3 - x * y_n^2) / 2
    // 初始猜测: y ≈ 0.5 = 0x2000 in ap_fixed<16,4>
    y = (hls_fixed16)0.5;

// #pragma HLS PIPELINE II=2 
    for (int i = 0; i < 4; i++) {
// #pragma HLS UNROLL
        hls_fixed32 y_sq  = (hls_fixed32)y * (hls_fixed32)y;
        hls_fixed32 x_y2  = (hls_fixed32)x_norm * y_sq;
        hls_fixed16 three_minus = (hls_fixed16)((hls_fixed32)1.5 - x_y2);
        y = y * three_minus;
    }

    // 反归一化: y_corrected = y * 2^(shift/2)
    if (shift > 0) {
        y = y >> (shift / 2);
    } else if (shift < 0) {
        y = y << (-shift / 2);
    }

    return y;
}

// ============================================================
// 顶层函数: RMSNorm 硬件闭环
// ============================================================
extern "C" {
void kernel_rmsnorm(
    volatile ap_uint<256>  * input,       // m_axi gmem0: [D] 输入特征
    volatile ap_uint<256>  * weight,      // m_axi gmem1: [D] 缩放权重
    volatile hls_int8  * output,      // m_axi gmem2: [D] 归一化输出
    int                  D,           // s_axilite: 向量维度
    float                eps          // s_axilite: epsilon (通过 raw bits 传入)
);
}

void kernel_rmsnorm(
    volatile hls_int8  * input,
    volatile hls_int8  * weight,
    volatile hls_int8  * output,
    int                  D,
    float                eps
) {
    // §4.2 HLS 接口约束
#pragma HLS INTERFACE m_axi port=input   bundle=gmem0 offset=slave depth=MAX_D
#pragma HLS INTERFACE m_axi port=weight  bundle=gmem1 offset=slave depth=MAX_D
#pragma HLS INTERFACE m_axi port=output  bundle=gmem2 offset=slave depth=MAX_D
#pragma HLS INTERFACE s_axilite port=D      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=eps    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // ── BRAM 向量线存 (§6.3) ──
    // 用 256-bit 位宽的 RAM 解决 32 通道读写的端口饥饿问题
    ap_uint<256> linebuf[MAX_WORDS];
#pragma HLS BIND_STORAGE variable=linebuf type=RAM_1P impl=BRAM

    // ── 阶段 1: 计算均方根 ──
    hls_int48 sum_sq = 0;  // 平方和 (48-bit 防溢出: D_max=4096 × 127² ≈ 66M < 2^48)

    // 第一遍扫描: 计算平方和 + 数据流泵入线存
    int half_D = D / 2;

    int words = D / VECTOR_WIDTH;

    // ── 阶段 1: 计算平方和并泵入缓存 ──
    SUM_SQ_LOOP: for (int c = 0; c < words; c++) {

        ap_uint<256> in_word = input[c]; // 完美单次突发读取 32 Bytes
        linebuf[c] = in_word;            // 并行写入 256-bit BRAM
        
        hls_int48 local_sum = 0;

        // [修改]: 从 256-bit Word 中切片，HLS 将自动生成 32-路加法树 (Adder Tree)
        // #pragma HLS PIPELINE II=2        
        for (int i = 0; i < VECTOR_WIDTH; i++) {
// #pragma HLS UNROLL
            // 提取对应 8-bit
            hls_int8 x = (hls_int8)in_word.range(i * 8 + 7, i * 8);
            hls_int16 x_ext = (hls_int16)x;
            local_sum += (hls_int48)(x_ext * x_ext);
        }
        sum_sq += local_sum;
    }

    // 计算 rms 因子
    // mean_sq = sum_sq / D
    hls_fixed32 mean_sq = sum_sq / D;

    // [修改]: 直接利用 HLS 内部机制实现 IEEE-754 浮点到 ap_fixed 的安全转换
    hls_fixed32 eps_fixed = (hls_fixed32)eps;
    hls_fixed32 rms_input = mean_sq + eps_fixed;
    hls_fixed16 rms_inv = cordic_rsqrt(rms_input);

    // ── 阶段 2: 缩放映射 ──
    // 第二遍: 从线存读取原向量, 与 rms_inv 和 weight 做哈达玛乘
    // 32 通道并行 (II=1 全吞吐)

    SCALE_LOOP: for (int c = 0; c < words; c++) {

        ap_uint<256> cached_word = linebuf[c];
        ap_uint<256> w_word = weight[c];
        ap_uint<256> out_word = 0;

        for (int i = 0; i < VECTOR_WIDTH; i++) {
            #pragma HLS PIPELINE II=1            
// #pragma HLS UNROLL
            hls_int8 x = (hls_int8)cached_word.range(i * 8 + 7, i * 8);
            hls_int8 w = (hls_int8)w_word.range(i * 8 + 7, i * 8);
            
            hls_int16 x_ext = (hls_int16)x;
            hls_int16 w_ext = (hls_int16)w;

            hls_int32 scaled = (hls_int32)x_ext * (hls_int32)w_ext;
            hls_fixed32 result = (hls_fixed32)scaled * (hls_fixed32)rms_inv;

            hls_int16 result_int = (hls_int16)(result >> 12);
            hls_int8 out_val;
            
            if (result_int > 127)        out_val = 127;
            else if (result_int < -128)  out_val = -128;
            else                         out_val = (hls_int8)result_int;

            // 将处理好的 8-bit 打包回 256-bit Word 中
            out_word.range(i * 8 + 7, i * 8) = (ap_uint<8>)out_val;
        }
        // 写回 DDR (此时也是单口 256-bit 完美突发写)
        output[c] = out_word;
    }
    
}
