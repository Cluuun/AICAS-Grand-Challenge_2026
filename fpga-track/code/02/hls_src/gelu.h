#ifndef __INT_GELU_H__
#define __INT_GELU_H__

#include "common.h"
#include "utils.h"

// ============================================================================
// SmolVLM2 GELU Activation Function for ViT MLP
// ============================================================================
// GELU(x) = x * Φ(x), where Φ(x) is the CDF of standard normal distribution
// Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
// 使用查找表近似实现，分为主表和移位表两级查找

// GELU 查找表参数
constexpr int GELU_LOG2DENOM        = 14;      // 主表索引右移位数
constexpr int GELU_LOG2DENOM_S      = 21;      // 移位表索引右移位数
constexpr int GELU_ENTRIES          = 4096;    // 主表条目数
constexpr int GELU_ENTRIES_S        = 32;      // 移位表条目数
constexpr int GELU_ALPHA            = -26214400;  // 输入偏移量 (对应输入范围的起始点)

// GELU 输入/输出数据类型定义 (ViT 专用)
typedef VIT_FC1_TRUNC_T    GELU_IN_T;     // GELU 输入类型 (来自 fc1 线性层截断后的输出, 28-bit)
typedef VIT_FC1_GELU_T     GELU_OUT_T;    // GELU 输出类型 (22-bit)

// 查找表 (从外部文件加载)
constexpr int GELU_TABLE[] = {
    #include "ref/ViT/MLP_LAYER0_GELU_TABLE.txt"
};

constexpr int GELU_TABLE_S[] = {
    #include "ref/ViT/MLP_LAYER0_GELU_TABLE_S.txt"
};

// ============================================================================
// GELU 模板类
// ============================================================================
// 模板参数:
//   if_t   - 输入数据类型
//   gelu_t - 输出数据类型
//   T      - Token 总数
//   TP     - Token 并行度
//   C      - 通道总数 (hidden dimension / intermediate_size)
//   CP     - 通道并行度 (接口并行度)
// ============================================================================

template<
    class if_t,
    class gelu_t,
    int T,
    int TP,
    int C,
    int CP
>
class GELU {
public:
    // 计算后的 tile 参数
    static constexpr int TT = T / TP;   // Token 维度 tile 数
    static constexpr int CT = C / CP;   // 通道维度 tile 数

    GELU() {}

    // ========================================================================
    // do_gelu: 执行 GELU 激活函数（packed，单拍 TP*CP）
    // ========================================================================
    // 输入: i_stream - 输入数据流 [TT x CT] 个 vector<TP*CP>
    // 输出: o_stream - 输出数据流 [TT x CT] 个 vector<TP*CP>
    // ========================================================================
    void do_gelu(
        hls::stream<hls::vector<if_t, TP * CP> > &i_stream, 
        hls::stream<hls::vector<gelu_t, TP * CP> > &o_stream
    ) {
        for (int tt = 0; tt < TT; ++tt) {
            for (int ct = 0; ct < CT; ++ct) {
                #pragma HLS pipeline II=1

                // 读取输入向量
                hls::vector<if_t, TP * CP> x_vec = i_stream.read();
                hls::vector<gelu_t, TP * CP> gelu_vec;
                
                // 并行计算所有元素的 GELU
                for (int tp = 0; tp < TP; ++tp) {
                    for (int cp = 0; cp < CP; ++cp) {
                        #pragma HLS unroll
                        
                        int idx = tp * CP + cp;
                        
                        // 计算主表索引 (量化后的输入映射到表索引)
                        int lut_idx = (x_vec[idx] - GELU_ALPHA) >> GELU_LOG2DENOM;
                        // 计算移位表索引
                        int lut_s_idx = (x_vec[idx] - GELU_ALPHA) >> GELU_LOG2DENOM_S;
                        
                        // 边界检查
                        lut_idx = clamp(lut_idx, 0, GELU_ENTRIES - 1);
                        lut_s_idx = clamp(lut_s_idx, 0, GELU_ENTRIES_S - 1);
                        
                        // 查表获取基础值和移位量
                        auto gelu_val = GELU_TABLE[lut_idx];
                        auto gelu_s = GELU_TABLE_S[lut_s_idx];
                        
                        // 组合结果: 基础值左移对应位数
                        gelu_vec[idx] = gelu_val << gelu_s;
                    }
                }

                // 写出结果
                o_stream.write(gelu_vec);

            } // end of ct loop
        } // end of tt loop
    }

    // ========================================================================
    // do_gelu_unpack: 执行 GELU（unpacked，上游顺序 TT -> CT -> TP，每拍 CP）
    // ========================================================================
    // 适配 ViT_DEMUX/类似模块的输出方式：token 并行度 TP 通过拍间展开，而非拍内打包。
    void do_gelu_unpack(
        hls::stream<hls::vector<if_t, CP> > &i_stream,
        hls::stream<hls::vector<gelu_t, CP> > &o_stream
    ) {
        for (int tt = 0; tt < TT; ++tt) {
            for (int ct = 0; ct < CT; ++ct) {
                for (int tp = 0; tp < TP; ++tp) {
                    #pragma HLS pipeline II=1
                    hls::vector<if_t, CP> x_vec = i_stream.read();
                    hls::vector<gelu_t, CP> gelu_vec;

                    for (int cp = 0; cp < CP; ++cp) {
                        #pragma HLS unroll
                        int lut_idx = (x_vec[cp] - GELU_ALPHA) >> GELU_LOG2DENOM;
                        int lut_s_idx = (x_vec[cp] - GELU_ALPHA) >> GELU_LOG2DENOM_S;
                        lut_idx = clamp(lut_idx, 0, GELU_ENTRIES - 1);
                        lut_s_idx = clamp(lut_s_idx, 0, GELU_ENTRIES_S - 1);
                        auto gelu_val = GELU_TABLE[lut_idx];
                        auto gelu_s = GELU_TABLE_S[lut_s_idx];
                        gelu_vec[cp] = gelu_val << gelu_s;
                    }
                    o_stream.write(gelu_vec);
                }
            }
        }
    }
};

#endif // __INT_GELU_H__
