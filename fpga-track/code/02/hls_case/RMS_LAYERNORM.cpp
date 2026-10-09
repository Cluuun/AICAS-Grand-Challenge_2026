#include "../src/reuse_common.h"
#include "../src/rmsnorm.h"
#include "../src/layernorm.h"
#include "../src/utils.h"
#include <vector>

// ============================================================================
// RMS_LAYERNORM：LLM RMSNorm 与 ViT LayerNorm 的 mode-aware 条件复用 IP
// ============================================================================
// 资源策略：
// - 不在 IP 内常驻 32 层 LLM RMSNorm 权重或 12 层 ViT LayerNorm gamma/beta。
// - M_AXI/PYNQ 按当前 mode/layer/pass 从 memory 流式喂入 lnw_stream/lnb_stream。
// - IP 内只缓存当前 pass 的 C 个权重和当前 token 的 C 个 x，避免新增大 BRAM ROM。
//
// LLM mode: x -> RMSNorm(gamma) -> shared quantizer，支持 LLAMA_L CLS/lm_head RMSNorm。
// ViT mode: x -> LayerNorm(mean/gamma/beta) -> shared quantizer，覆盖 vision layer 0..VIT_L-1。
// 两种 mode 输出汇入 common XLN stream，再进入单个 8-lane 动态量化函数。
// ============================================================================

// simulation hyperparameters
constexpr int RN_LLM_POS    = 96;
// ViT 新导出按 batch*patch 保存，当前 vision_0/new_txt 与其他层同批为 13*1024。
constexpr int RN_VIT_T_LOAD = 13312;
constexpr int RN_VIT_POS    = 0;

// shared interface hyperparameters
constexpr int RN_CP      = REUSE_CP;
constexpr int RN_Q_BITS  = DW_AQ;
constexpr int RN_Q_MAX   = +(1 << (RN_Q_BITS - 1)) - 1;
constexpr int RN_Q_MIN   = -(1 << (RN_Q_BITS - 1));
constexpr int RN_A_SMAX  = 15;

// model sizes
constexpr int RN_LLM_T        = REUSE_LLM_TILE_T;
constexpr int RN_LLM_C        = LLAMA_C;
constexpr int RN_LLM_CT       = RN_LLM_C / RN_CP;
constexpr int RN_LLM_VEC      = RN_LLM_T * RN_LLM_C / RN_CP;
constexpr int RN_LLM_NUM_X    = 2 * RN_LLM_T * RN_LLM_C;

constexpr int RN_VIT_T        = VIT_S;
constexpr int RN_VIT_C        = VIT_C;
constexpr int RN_VIT_CT       = RN_VIT_C / RN_CP;
constexpr int RN_VIT_VEC      = RN_VIT_T * RN_VIT_C / RN_CP;
constexpr int RN_VIT_NUM_X    = 2 * RN_VIT_T * RN_VIT_C;

constexpr int RN_MAX_C        = const_max(RN_LLM_C, RN_VIT_C);
constexpr int RN_DW_LNW       = const_max(DW_LNW, VIT_DW_LNW);
typedef ap_int<RN_DW_LNW> RN_LNW_T;
typedef ap_int<64>        RN_LNB_T;

// 权重和当前 token x 都只缓存当前 pass；按 CP 分 bank 保持 II=1 读写。
// 这些 bank 很浅，直接搬到 BRAM 会产生大量小 BRAM 并恶化 K26 OOC 时序，因此仍保留 LUTRAM。
typedef REUSE_X_T   RN_X_T;
typedef REUSE_XLN_T RN_XLN_T;

// ViT LayerNorm 的乘法真实输入远窄于旧代码中的 ap_int<64> 强制转换。
// 收窄后仍保留完整数学精度，但避免 HLS 推出 64x64/129-bit 宽乘法和长 CARRY8 链。
constexpr int RN_VIT_DW_DIFF      = VIT_DW_X + 2;
constexpr int RN_VIT_DW_MUL1_FULL = RN_VIT_DW_DIFF + VIT_DW_X_RSQRT;
constexpr int RN_VIT_DW_MUL1      = RN_VIT_DW_MUL1_FULL - VIT_LAYERNORM_TRUNC_MUL1;
constexpr int RN_VIT_DW_MUL2_FULL = RN_VIT_DW_MUL1 + VIT_DW_LNW;
constexpr int RN_VIT_DW_MUL2      = RN_VIT_DW_MUL2_FULL - VIT_LAYERNORM_TRUNC_MUL2;
typedef ap_int<RN_VIT_DW_DIFF>      RN_VIT_DIFF_T;
typedef ap_int<RN_VIT_DW_MUL1_FULL> RN_VIT_MUL1_FULL_T;
typedef ap_int<RN_VIT_DW_MUL1>      RN_VIT_MUL1_T;
typedef ap_int<RN_VIT_DW_MUL2_FULL> RN_VIT_MUL2_FULL_T;
typedef ap_int<RN_VIT_DW_MUL2>      RN_VIT_MUL2_T;

X_RSQRT_T llm_rsqrt_lookup(X_POW2SUM_T x_pow2sum) {
    #pragma HLS inline
    int lut_idx = 0;
    for (int i = 0; i < RSQRT_NUM_TABLES; ++i) {
        #pragma HLS unroll
        if (x_pow2sum >= RSQRT_ALPHAS[i]) {
            lut_idx = i;
        }
    }
    auto alpha       = RSQRT_ALPHAS[lut_idx];
    auto log2denom   = RSQRT_LOG2DENOMS[lut_idx];
    auto offset_diff = RSQRT_OFFSETS_DIFF[lut_idx];
    auto index       = clamp((x_pow2sum - alpha) >> log2denom, 0, RSQRT_ENTRIES - 1);
    return (X_RSQRT_T)(RSQRT_TABLES[lut_idx * RSQRT_ENTRIES + index] << offset_diff);
}

VIT_X_RSQRT_T vit_rsqrt_lookup(ap_uint<64> x_var) {
    #pragma HLS inline
    int lut_idx = 0;
    for (int i = 0; i < VIT_LAYERNORM_NUM_TABLES; ++i) {
        #pragma HLS unroll
        if (x_var >= (ap_uint<64>)VIT_LAYERNORM_ALPHAS[i]) {
            lut_idx = i;
        }
    }
    auto alpha       = VIT_LAYERNORM_ALPHAS[lut_idx];
    auto log2denom   = VIT_LAYERNORM_LOG2DENOMS[lut_idx];
    auto offset_diff = VIT_LAYERNORM_OFFSETS_DIFF[lut_idx];
    auto index       = clamp((ap_int<64>)(x_var - (ap_uint<64>)alpha) >> log2denom,
                             (ap_int<64>)0,
                             (ap_int<64>)(VIT_LAYERNORM_ENTRIES - 1));
    return ((VIT_X_RSQRT_T)VIT_LAYERNORM_TABLES[lut_idx * VIT_LAYERNORM_ENTRIES + (int)index]) << offset_diff;
}

ap_int<64> floor_div_c(ap_int<64> value, int c) {
    #pragma HLS inline
    ap_int<64> c64 = (ap_int<64>)c;
    if (value >= 0) {
        return (ap_int<64>)(value / c64);
    }
    return (ap_int<64>)(-((-value + c64 - 1) / c64));
}

void load_llm_lnw(
    hls::stream<hls::vector<RN_LNW_T, RN_CP> > &lnw_stream,
    RN_LNW_T lnw_buf[RN_MAX_C]
) {
    for (int ct = 0; ct < RN_LLM_CT; ++ct) {
        #pragma HLS pipeline II=1
        hls::vector<RN_LNW_T, RN_CP> w_vec = lnw_stream.read();
        for (int cp = 0; cp < RN_CP; ++cp) {
            #pragma HLS unroll
            lnw_buf[ct * RN_CP + cp] = w_vec[cp];
        }
    }
}

void load_vit_lnw_lnb(
    hls::stream<hls::vector<RN_LNW_T, RN_CP> > &lnw_stream,
    hls::stream<hls::vector<RN_LNB_T, RN_CP> > &lnb_stream,
    RN_LNW_T lnw_buf[RN_MAX_C],
    RN_LNB_T lnb_buf[RN_MAX_C]
) {
    for (int ct = 0; ct < RN_VIT_CT; ++ct) {
        #pragma HLS pipeline II=1
        hls::vector<RN_LNW_T, RN_CP> w_vec = lnw_stream.read();
        hls::vector<RN_LNB_T, RN_CP> b_vec = lnb_stream.read();
        for (int cp = 0; cp < RN_CP; ++cp) {
            #pragma HLS unroll
            int c = ct * RN_CP + cp;
            lnw_buf[c] = w_vec[cp];
            lnb_buf[c] = b_vec[cp];
        }
    }
}

// LLM RMSNorm：不减 mean、无 beta，使用 LLM rsqrt LUT 和 LLM gamma。
void llm_rmsnorm_to_common(
    hls::stream<hls::vector<RN_X_T,   RN_CP> > &x_stream,
    hls::stream<hls::vector<RN_XLN_T, RN_CP> > &xln_stream,
    RN_LNW_T lnw_buf[RN_MAX_C],
    RN_X_T x_buf[RN_MAX_C]
) {
    for (int t = 0; t < RN_LLM_T; ++t) {
        X_POW2SUM_T x_pow2sum = 0;

        for (int ct = 0; ct < RN_LLM_CT; ++ct) {
            #pragma HLS pipeline II=1
            hls::vector<RN_X_T, RN_CP> x_vec = x_stream.read();
            for (int cp = 0; cp < RN_CP; ++cp) {
                #pragma HLS unroll
                int c = ct * RN_CP + cp;
                X_T x = reuse_truncate<X_T>(x_vec[cp]);
                x_buf[c] = (RN_X_T)x;
                x_pow2sum = x_pow2sum + x * x;
            }
        }

        X_RSQRT_T x_rsqrt = llm_rsqrt_lookup(x_pow2sum);

        for (int ct = 0; ct < RN_LLM_CT; ++ct) {
            #pragma HLS pipeline II=1
            hls::vector<RN_XLN_T, RN_CP> o_vec;
            for (int cp = 0; cp < RN_CP; ++cp) {
                #pragma HLS unroll
                int c = ct * RN_CP + cp;
                X_T x = reuse_truncate<X_T>(x_buf[c]);
                LNW_T w = reuse_truncate<LNW_T>(lnw_buf[c]);
                auto x_mul_rsqrt = (x * x_rsqrt) >> RSQRT_TRUNC_MUL1;
                // LLM final RMSNorm 在 pos352 已出现超过 DW_XLN=22 的合法软件值；
                // 复用通路本来按 ViT 上限提供 24-bit RN_XLN_T，这里避免先落到
                // LLM 私有 XLN_T 后回绕，再交给共享动态量化器。
                RN_XLN_T out = (RN_XLN_T)((x_mul_rsqrt * w) >> RSQRT_TRUNC_MUL2);
                o_vec[cp] = out;
            }
            xln_stream.write(o_vec);
        }
    }
}

// ViT LayerNorm：先求 floor mean，再对中心化数据求方差、rsqrt、gamma/beta。
void vit_layernorm_to_common(
    hls::stream<hls::vector<RN_X_T,   RN_CP> > &x_stream,
    hls::stream<hls::vector<RN_XLN_T, RN_CP> > &xln_stream,
    RN_LNW_T lnw_buf[RN_MAX_C],
    RN_LNB_T lnb_buf[RN_MAX_C],
    RN_X_T x_buf[RN_MAX_C]
) {
    for (int t = 0; t < RN_VIT_T; ++t) {
        ap_int<64> x_sum = 0;

        for (int ct = 0; ct < RN_VIT_CT; ++ct) {
            #pragma HLS pipeline II=1
            hls::vector<RN_X_T, RN_CP> x_vec = x_stream.read();
            for (int cp = 0; cp < RN_CP; ++cp) {
                #pragma HLS unroll
                int c = ct * RN_CP + cp;
                VIT_X_T x = reuse_truncate<VIT_X_T>(x_vec[cp]);
                x_buf[c] = (RN_X_T)x;
                x_sum = x_sum + (ap_int<64>)x;
            }
        }

        ap_int<64> x_mean = floor_div_c(x_sum, RN_VIT_C);
        ap_uint<64> x_var = 0;

        for (int ct = 0; ct < RN_VIT_CT; ++ct) {
            #pragma HLS pipeline II=1
            for (int cp = 0; cp < RN_CP; ++cp) {
                #pragma HLS unroll
                int c = ct * RN_CP + cp;
                RN_VIT_DIFF_T diff = (RN_VIT_DIFF_T)((ap_int<64>)reuse_truncate<VIT_X_T>(x_buf[c]) - x_mean);
                x_var = x_var + (ap_uint<64>)(diff * diff);
            }
        }

        VIT_X_RSQRT_T x_rsqrt = vit_rsqrt_lookup(x_var);

        for (int ct = 0; ct < RN_VIT_CT; ++ct) {
            #pragma HLS pipeline II=1
            hls::vector<RN_XLN_T, RN_CP> o_vec;
            for (int cp = 0; cp < RN_CP; ++cp) {
                #pragma HLS unroll
                int c = ct * RN_CP + cp;
                RN_VIT_DIFF_T diff = (RN_VIT_DIFF_T)((ap_int<64>)reuse_truncate<VIT_X_T>(x_buf[c]) - x_mean);
                RN_VIT_MUL1_FULL_T mul1_full = diff * x_rsqrt;
                #pragma HLS bind_op variable=mul1_full op=mul impl=dsp latency=3
                RN_VIT_MUL1_T x_mul_rsqrt = (RN_VIT_MUL1_T)(mul1_full >> VIT_LAYERNORM_TRUNC_MUL1);
                VIT_LNW_T lnw = reuse_truncate<VIT_LNW_T>(lnw_buf[c]);
                RN_VIT_MUL2_FULL_T mul2_full = x_mul_rsqrt * lnw;
                #pragma HLS bind_op variable=mul2_full op=mul impl=dsp latency=3
                RN_VIT_MUL2_T x_mul_lnw = (RN_VIT_MUL2_T)(mul2_full >> VIT_LAYERNORM_TRUNC_MUL2);
                VIT_XLN_T out = (VIT_XLN_T)((ap_int<64>)x_mul_lnw + lnb_buf[c]);
                o_vec[cp] = (RN_XLN_T)out;
            }
            xln_stream.write(o_vec);
        }
    }
}

// mode 在 pass 粒度选择 RMSNorm 或 LayerNorm；当前 pass 权重先从外部流加载到小缓存。
void shared_norm_to_common(
    REUSE_MODE_T mode,
    hls::stream<hls::vector<RN_X_T,   RN_CP> > &x_stream,
    hls::stream<hls::vector<RN_LNW_T, RN_CP> > &lnw_stream,
    hls::stream<hls::vector<RN_LNB_T, RN_CP> > &lnb_stream,
    hls::stream<hls::vector<RN_XLN_T, RN_CP> > &xln_stream
) {
    RN_LNW_T lnw_buf[RN_MAX_C];
    RN_LNB_T lnb_buf[RN_MAX_C];
    RN_X_T x_buf[RN_MAX_C];
    #pragma HLS array_partition variable=lnw_buf cyclic factor=RN_CP dim=1
    #pragma HLS array_partition variable=lnb_buf cyclic factor=RN_CP dim=1
    #pragma HLS array_partition variable=x_buf   cyclic factor=RN_CP dim=1
    // CLB-02(b)：gamma/beta 只在层开始时装入，后续按 bank 顺序读取；
    // 迁到 BRAM 可从 RMS hot window 移除 LUTRAM。x_buf 位于 norm 读写主路径，
    // 仍保持 LUTRAM，避免复现 2026-05-21 全量转 BRAM 后 CP 变差的问题。
    #pragma HLS bind_storage variable=lnw_buf type=ram_2p impl=BRAM
    #pragma HLS bind_storage variable=lnb_buf type=ram_2p impl=BRAM
    #pragma HLS bind_storage variable=x_buf   type=ram_2p impl=LUTRAM

    if (is_vit_mode(mode)) {
        load_vit_lnw_lnb(lnw_stream, lnb_stream, lnw_buf, lnb_buf);
        vit_layernorm_to_common(x_stream, xln_stream, lnw_buf, lnb_buf, x_buf);
    } else {
        load_llm_lnw(lnw_stream, lnw_buf);
        llm_rmsnorm_to_common(x_stream, xln_stream, lnw_buf, x_buf);
    }
}

// 共享 XLN 动态量化器：固定 CP=G=8，每拍产生 8 个 q 和 1 个 scale。
void shared_xln_quantize(
    int total_vec,
    hls::stream<hls::vector<RN_XLN_T,   RN_CP> > &xln_stream,
    hls::stream<hls::vector<REUSE_AQ_T, RN_CP> > &xlnq_stream,
    hls::stream<hls::vector<REUSE_AS_T, 1    > > &xlns_stream
) {
    for (int vec = 0; vec < total_vec; ++vec) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=960 max=98304
        hls::vector<RN_XLN_T, RN_CP> xln_vec = xln_stream.read();
        hls::vector<REUSE_AQ_T, RN_CP> q_vec;
        hls::vector<REUSE_AS_T, 1> s_vec;

        RN_XLN_T abs_max = 0;
        for (int cp = 0; cp < RN_CP; ++cp) {
            #pragma HLS unroll
            abs_max = max(abs_max, (RN_XLN_T)abs(xln_vec[cp]));
        }

        int8_t s_val_i = log2ceil(abs_max) - (RN_Q_BITS - 1);
        REUSE_AS_T s_val = clamp(s_val_i, 0, RN_A_SMAX);
        s_vec[0] = s_val;

        for (int cp = 0; cp < RN_CP; ++cp) {
            #pragma HLS unroll
            RN_XLN_T q_val = xln_vec[cp];
            if (s_val != 0) {
                q_val = q_val >> (s_val - 1);
                q_val = q_val + 1;
                q_val = q_val >> 1;
            }
            // signed int8 输出按软件 Quantizer 语义做双向 clamp，避免 4-bit scale 饱和后负数回绕。
            q_vec[cp] = clamp(q_val, (RN_XLN_T)RN_Q_MIN, (RN_XLN_T)RN_Q_MAX);
        }

        xlnq_stream.write(q_vec);
        xlns_stream.write(s_vec);
    }
}

// top function
void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    hls::stream<hls::vector<RN_X_T,     RN_CP> > &x_stream,
    hls::stream<hls::vector<RN_LNW_T,   RN_CP> > &lnw_stream,
    hls::stream<hls::vector<RN_LNB_T,   RN_CP> > &lnb_stream,
    hls::stream<hls::vector<REUSE_AQ_T, RN_CP> > &xlnq_stream,
    hls::stream<hls::vector<REUSE_AS_T, 1    > > &xlns_stream
) {
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface ap_none port=mode
    #pragma HLS interface ap_none port=l_begin
    #pragma HLS interface ap_none port=l_close
    #pragma HLS interface axis port=x_stream
    #pragma HLS interface axis port=lnw_stream
    #pragma HLS interface axis port=lnb_stream
    #pragma HLS interface axis port=xlnq_stream
    #pragma HLS interface axis port=xlns_stream

    #pragma HLS aggregate variable=x_stream    compact=bit
    #pragma HLS aggregate variable=lnw_stream  compact=bit
    #pragma HLS aggregate variable=lnb_stream  compact=bit
    #pragma HLS aggregate variable=xlnq_stream compact=bit
    #pragma HLS aggregate variable=xlns_stream compact=bit

    bool vit = is_vit_mode(mode);
    int layer_limit = vit ? VIT_L : (LLAMA_L + 1);
    int pass_vec = vit ? RN_VIT_VEC : RN_LLM_VEC;

    for (int l = l_begin; l < l_close && l < layer_limit; ++l) {
        bool llm_cls = is_llm_mode(mode) && l == LLAMA_L;
        int pass_count = llm_cls ? 1 : 2;
        for (int pass = 0; pass < pass_count; ++pass) {
            hls::stream<hls::vector<RN_XLN_T, RN_CP> > xln_stream("xln_stream");
            #pragma HLS stream variable=xln_stream depth=32
            // CLB-02(b)：这里保持默认 FIFO 映射。depth=32、192-bit FIFO
            // 转 BRAM 的 LUTRAM 收益很小但 BRAM 成本高，容易越过 140-tile guardrail。
            #pragma HLS dataflow

            shared_norm_to_common(mode, x_stream, lnw_stream, lnb_stream, xln_stream);
            shared_xln_quantize(pass_vec, xln_stream, xlnq_stream, xlns_stream);
        }
    }
}

void feed_llm_lnw(int l, hls::stream<hls::vector<RN_LNW_T, RN_CP> > &lnw_stream) {
    bool cls = l == LLAMA_L;
    int pass_count = cls ? 1 : 2;
    for (int pass = 0; pass < pass_count; ++pass) {
        for (int ct = 0; ct < RN_LLM_CT; ++ct) {
            hls::vector<RN_LNW_T, RN_CP> w_vec;
            for (int cp = 0; cp < RN_CP; ++cp) {
                int c = ct * RN_CP + cp;
                int value = cls ? CLS_RMSNORM_LNW[c] : RMSNORM_LNW[l * 2 * RN_LLM_C + pass * RN_LLM_C + c];
                w_vec[cp] = (RN_LNW_T)value;
            }
            lnw_stream.write(w_vec);
        }
    }
}

void feed_vit_lnw_lnb(int l,
                      hls::stream<hls::vector<RN_LNW_T, RN_CP> > &lnw_stream,
                      hls::stream<hls::vector<RN_LNB_T, RN_CP> > &lnb_stream) {
    for (int pass = 0; pass < 2; ++pass) {
        for (int ct = 0; ct < RN_VIT_CT; ++ct) {
            hls::vector<RN_LNW_T, RN_CP> w_vec;
            hls::vector<RN_LNB_T, RN_CP> b_vec;
            for (int cp = 0; cp < RN_CP; ++cp) {
                int c = ct * RN_CP + cp;
                int idx = l * 2 * RN_VIT_C + pass * RN_VIT_C + c;
                w_vec[cp] = (RN_LNW_T)VIT_LAYERNORM_LNW[idx];
                b_vec[cp] = (RN_LNB_T)VIT_LAYERNORM_LNB[idx];
            }
            lnw_stream.write(w_vec);
            lnb_stream.write(b_vec);
        }
    }
}

void test_llm_layer(int l) {
    string file_path = BINARIES_PATH + to_string(l);
    string save_path = CONDENSE_PATH + to_string(l);

    std::vector<int64_t> ref_x(RN_LLM_NUM_X);
    std::vector<int8_t> ref_q(RN_LLM_NUM_X);
    std::vector<int8_t> ref_s(2 * RN_LLM_T * RN_LLM_CT);
    std::vector<int8_t> dut_q(RN_LLM_NUM_X);
    std::vector<int8_t> dut_s(2 * RN_LLM_T * RN_LLM_CT);

    auto condensed_x = read_tensor<int64_t>(save_path + "/CONDENSED_RESIDUAL_O.bin");
    tensor2array<int64_t>(condensed_x, ref_x.data(), 1, 1, 1, 1, RN_LLM_NUM_X, RN_LLM_NUM_X);

    auto mha_q = read_tensor<int8_t>(file_path + "/MHA_XLN_Q.bin");
    auto mha_s = read_tensor<int8_t>(file_path + "/MHA_XLN_S.bin");
    auto mlp_q = read_tensor<int8_t>(file_path + "/MLP_XLN_Q.bin");
    auto mlp_s = read_tensor<int8_t>(file_path + "/MLP_XLN_S.bin");
    // LLM norm golden 按实际文件长度推导 T_LOAD，当前 weights 通常为 877。
    tensor2array_dynamic_t<int8_t>(mha_q, ref_q.data(),                       1, 1, RN_LLM_POS, RN_LLM_T, RN_LLM_C,  RN_LLM_C,  "MHA_XLN_Q");
    tensor2array_dynamic_t<int8_t>(mha_s, ref_s.data(),                       1, 1, RN_LLM_POS, RN_LLM_T, RN_LLM_CT, RN_LLM_CT, "MHA_XLN_S");
    tensor2array_dynamic_t<int8_t>(mlp_q, ref_q.data() + RN_LLM_T * RN_LLM_C,  1, 1, RN_LLM_POS, RN_LLM_T, RN_LLM_C,  RN_LLM_C,  "MLP_XLN_Q");
    tensor2array_dynamic_t<int8_t>(mlp_s, ref_s.data() + RN_LLM_T * RN_LLM_CT, 1, 1, RN_LLM_POS, RN_LLM_T, RN_LLM_CT, RN_LLM_CT, "MLP_XLN_S");

    hls::stream<hls::vector<RN_X_T,     RN_CP> > x_stream("llm_x_stream");
    hls::stream<hls::vector<RN_LNW_T,   RN_CP> > lnw_stream("llm_lnw_stream");
    hls::stream<hls::vector<RN_LNB_T,   RN_CP> > lnb_stream("llm_lnb_stream");
    hls::stream<hls::vector<REUSE_AQ_T, RN_CP> > q_stream("llm_q_stream");
    hls::stream<hls::vector<REUSE_AS_T, 1    > > s_stream("llm_s_stream");

    array2stream<int64_t, RN_X_T, 1, 1, 1, 1, RN_LLM_NUM_X, RN_CP>(ref_x.data(), x_stream, "LLM X", true);
    feed_llm_lnw(l, lnw_stream);

    top(MODE_LLM, l, l + 1, x_stream, lnw_stream, lnb_stream, q_stream, s_stream);

    save_condensed_tensor<int8_t, REUSE_AQ_T, RN_LLM_NUM_X,              RN_CP>(save_path + "/CONDENSED_XLN_Q.bin", q_stream);
    save_condensed_tensor<int8_t, REUSE_AS_T, 2 * RN_LLM_T * RN_LLM_CT,  1    >(save_path + "/CONDENSED_XLN_S.bin", s_stream);

    stream2array<int8_t, REUSE_AQ_T, 2, RN_LLM_T, 1, RN_LLM_C,  RN_CP>(q_stream, dut_q.data(), "LLM XLN_Q", true);
    stream2array<int8_t, REUSE_AS_T, 2, RN_LLM_T, 1, RN_LLM_CT, 1    >(s_stream, dut_s.data(), "LLM XLN_S", true);

    compare<int8_t>(ref_q.data(), dut_q.data(), RN_LLM_NUM_X, "LLM XLN_Q");
    compare<int8_t>(ref_s.data(), dut_s.data(), 2 * RN_LLM_T * RN_LLM_CT, "LLM XLN_S");
}

void test_llm_cls() {
    string file_path = BINARIES_PATH + to_string(LLAMA_L);
    string save_path = CONDENSE_PATH + to_string(LLAMA_L);

    constexpr int NUM_CLS_X = RN_LLM_T * RN_LLM_C;
    std::vector<int64_t> ref_x(NUM_CLS_X);
    std::vector<int8_t> ref_q(NUM_CLS_X);
    std::vector<int8_t> ref_s(RN_LLM_T * RN_LLM_CT);
    std::vector<int8_t> dut_q(NUM_CLS_X);
    std::vector<int8_t> dut_s(RN_LLM_T * RN_LLM_CT);

    auto cls_x = read_tensor<int64_t>(file_path + "/CLS_X.bin");
    auto cls_q = read_tensor<int8_t>(file_path + "/CLS_XLN_Q.bin");
    auto cls_s = read_tensor<int8_t>(file_path + "/CLS_XLN_S.bin");
    // CLS/lm_head 与 decoder+CLS 顶层测试统一使用当前 decode POS 窗口，
    // 避免 final RMSNorm 参考仍停留在文件开头 8-token tile。
    tensor2array_dynamic_t<int64_t>(cls_x, ref_x.data(), 1, 1, RN_LLM_POS, RN_LLM_T, RN_LLM_C,  RN_LLM_C,  "CLS_X");
    tensor2array_dynamic_t<int8_t>(cls_q,  ref_q.data(), 1, 1, RN_LLM_POS, RN_LLM_T, RN_LLM_C,  RN_LLM_C,  "CLS_XLN_Q");
    tensor2array_dynamic_t<int8_t>(cls_s,  ref_s.data(), 1, 1, RN_LLM_POS, RN_LLM_T, RN_LLM_CT, RN_LLM_CT, "CLS_XLN_S");

    hls::stream<hls::vector<RN_X_T,     RN_CP> > x_stream("cls_x_stream");
    hls::stream<hls::vector<RN_LNW_T,   RN_CP> > lnw_stream("cls_lnw_stream");
    hls::stream<hls::vector<RN_LNB_T,   RN_CP> > lnb_stream("cls_lnb_stream");
    hls::stream<hls::vector<REUSE_AQ_T, RN_CP> > q_stream("cls_q_stream");
    hls::stream<hls::vector<REUSE_AS_T, 1    > > s_stream("cls_s_stream");

    array2stream<int64_t, RN_X_T, 1, 1, RN_LLM_T, 1, RN_LLM_C, RN_CP>(ref_x.data(), x_stream, "LLM CLS X", true);
    feed_llm_lnw(LLAMA_L, lnw_stream);

    top(MODE_LLM, LLAMA_L, LLAMA_L + 1, x_stream, lnw_stream, lnb_stream, q_stream, s_stream);

    save_condensed_tensor<int8_t, REUSE_AQ_T, NUM_CLS_X,             RN_CP>(save_path + "/CONDENSED_XLN_Q.bin", q_stream);
    save_condensed_tensor<int8_t, REUSE_AS_T, RN_LLM_T * RN_LLM_CT,  1    >(save_path + "/CONDENSED_XLN_S.bin", s_stream);

    stream2array<int8_t, REUSE_AQ_T, 1, RN_LLM_T, 1, RN_LLM_C,  RN_CP>(q_stream, dut_q.data(), "LLM CLS XLN_Q", true);
    stream2array<int8_t, REUSE_AS_T, 1, RN_LLM_T, 1, RN_LLM_CT, 1    >(s_stream, dut_s.data(), "LLM CLS XLN_S", true);

    compare<int8_t>(ref_q.data(), dut_q.data(), NUM_CLS_X, "LLM CLS XLN_Q");
    compare<int8_t>(ref_s.data(), dut_s.data(), RN_LLM_T * RN_LLM_CT, "LLM CLS XLN_S");
}

void test_vit_layer(int l) {
    string file_path = VIT_BINARIES_PATH + to_string(l);
    string save_path = VIT_CONDENSE_PATH + to_string(l);

    std::vector<int64_t> ref_x(RN_VIT_NUM_X);
    std::vector<int8_t> ref_q(RN_VIT_NUM_X);
    std::vector<int8_t> ref_s(2 * RN_VIT_T * RN_VIT_CT);
    std::vector<int8_t> dut_q(RN_VIT_NUM_X);
    std::vector<int8_t> dut_s(2 * RN_VIT_T * RN_VIT_CT);

    auto condensed_x = read_tensor<int64_t>(save_path + "/CONDENSED_RESIDUAL_O.bin");
    tensor2array<int64_t>(condensed_x, ref_x.data(), 1, 1, 1, 1, RN_VIT_NUM_X, RN_VIT_NUM_X);

    auto mha_q = read_tensor<int8_t>(file_path + "/MHA_XLN_Q.bin");
    auto mha_s = read_tensor<int8_t>(file_path + "/MHA_XLN_S.bin");
    auto mlp_q = read_tensor<int8_t>(file_path + "/MLP_XLN_Q.bin");
    auto mlp_s = read_tensor<int8_t>(file_path + "/MLP_XLN_S.bin");
    tensor2array<int8_t>(mha_q, ref_q.data(),                      1, 1, RN_VIT_T_LOAD, RN_VIT_POS, RN_VIT_T, RN_VIT_C,  RN_VIT_C);
    tensor2array<int8_t>(mha_s, ref_s.data(),                      1, 1, RN_VIT_T_LOAD, RN_VIT_POS, RN_VIT_T, RN_VIT_CT, RN_VIT_CT);
    tensor2array<int8_t>(mlp_q, ref_q.data() + RN_VIT_T * RN_VIT_C, 1, 1, RN_VIT_T_LOAD, RN_VIT_POS, RN_VIT_T, RN_VIT_C,  RN_VIT_C);
    tensor2array<int8_t>(mlp_s, ref_s.data() + RN_VIT_T * RN_VIT_CT,1, 1, RN_VIT_T_LOAD, RN_VIT_POS, RN_VIT_T, RN_VIT_CT, RN_VIT_CT);

    hls::stream<hls::vector<RN_X_T,     RN_CP> > x_stream("vit_x_stream");
    hls::stream<hls::vector<RN_LNW_T,   RN_CP> > lnw_stream("vit_lnw_stream");
    hls::stream<hls::vector<RN_LNB_T,   RN_CP> > lnb_stream("vit_lnb_stream");
    hls::stream<hls::vector<REUSE_AQ_T, RN_CP> > q_stream("vit_q_stream");
    hls::stream<hls::vector<REUSE_AS_T, 1    > > s_stream("vit_s_stream");

    array2stream<int64_t, RN_X_T, 1, 1, 1, 1, RN_VIT_NUM_X, RN_CP>(ref_x.data(), x_stream, "ViT X", true);
    feed_vit_lnw_lnb(l, lnw_stream, lnb_stream);

    top(MODE_VIT, l, l + 1, x_stream, lnw_stream, lnb_stream, q_stream, s_stream);

    save_condensed_tensor<int8_t, REUSE_AQ_T, RN_VIT_NUM_X,             RN_CP>(save_path + "/CONDENSED_XLN_Q.bin", q_stream);
    save_condensed_tensor<int8_t, REUSE_AS_T, 2 * RN_VIT_T * RN_VIT_CT, 1    >(save_path + "/CONDENSED_XLN_S.bin", s_stream);

    stream2array<int8_t, REUSE_AQ_T, 2, RN_VIT_T, 1, RN_VIT_C,  RN_CP>(q_stream, dut_q.data(), "ViT XLN_Q", true);
    stream2array<int8_t, REUSE_AS_T, 2, RN_VIT_T, 1, RN_VIT_CT, 1    >(s_stream, dut_s.data(), "ViT XLN_S", true);

    compare<int8_t>(ref_q.data(), dut_q.data(), RN_VIT_NUM_X, "ViT XLN_Q");
    compare<int8_t>(ref_s.data(), dut_s.data(), 2 * RN_VIT_T * RN_VIT_CT, "ViT XLN_S");
}

#ifndef __SYNTHESIS__
int main() {
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        test_llm_layer(0);
        printf("LLM layer 0 passed\n");

        test_llm_cls();
        printf("LLM CLS passed\n");
    }

    test_vit_layer(vit_layer);
    printf("ViT layer %d passed\n", vit_layer);

    return 0;
}
#endif
