#include "../src/reuse_axi_common.h"

// ============================================================================
// M_AXI 参数搬运器：DEMUX bias + RMS/LayerNorm gamma/beta
// ============================================================================
// 拆分后本 IP 不再搬运 GEMM 权重或 residual state，只负责小/中等参数流：
//   - op=AXI_PARAM_BIAS：ViT DEMUX bias。每层先从 DDR 读一次到片上 cache，再按
//     GEMM 输出顺序重放，避免旧实现对同一个 channel tile 在 TT/TP 内重复读 DDR。
//   - op=AXI_PARAM_NORM：RMS_LAYERNORM 的 gamma/beta。lnw/lnb 体量很小，保持直接
//     256-bit pack 读取，不再为带宽单独做额外缓存。
//
// mode 在一次调用内保持稳定；LLM mode 下 bias op 直接 no-op，ViT mode 下 norm op
// 同时产生 gamma/beta，LLM norm op 只产生 gamma。

static void load_vit_bias_layer(
    int l,
    axi_maxi_t* memory_vit_bias,
    axi_bias_pack_t bias_cache[AXI_VIT_LAYER_BIAS_VECS]
) {
    // 每层 6912 个 int32 bias 只从 DDR 读一次，按 CP=8 lane 存入 224-bit cache word。
    int layer_base = l * AXI_VIT_LAYER_BIAS_COUNT;
    for (int vec_idx = 0; vec_idx < AXI_VIT_LAYER_BIAS_VECS; ++vec_idx) {
        #pragma HLS pipeline II=1
        axi_bias_vec_t vec;
        axi_load_i32_vec<axi_bias_t>(memory_vit_bias, layer_base + vec_idx * AXI_CP, vec);
        bias_cache[vec_idx] = axi_pack_bias_vec(vec);
    }
}

static void replay_cached_bias_vec(
    axi_bias_pack_t bias_cache[AXI_VIT_LAYER_BIAS_VECS],
    int vec_idx,
    hls::stream<axi_bias_vec_t>& bias_stream
) {
    #pragma HLS inline
    bias_stream.write(axi_unpack_bias_vec(bias_cache[vec_idx]));
}

static void replay_vit_bias_layer(
    axi_bias_pack_t bias_cache[AXI_VIT_LAYER_BIAS_VECS],
    hls::stream<axi_bias_vec_t>& bias_stream
) {
    // 重放顺序必须与 DEMUX 的 GEMM_Y 消费一一对齐：
    // Q/K/V: H -> QKV -> TT -> HCT -> TP；O/FC1/FC2: TT -> channel_tile -> TP。
    constexpr int off_bq_vec = 0;
    constexpr int off_bk_vec = off_bq_vec + AXI_VIT_C / AXI_CP;
    constexpr int off_bv_vec = off_bk_vec + AXI_VIT_C / AXI_CP;
    constexpr int off_bo_vec = off_bv_vec + AXI_VIT_C / AXI_CP;
    constexpr int off_b1_vec = off_bo_vec + AXI_VIT_C / AXI_CP;
    constexpr int off_b2_vec = off_b1_vec + AXI_VIT_CM / AXI_CP;
    constexpr int VIT_BIAS_TT = AXI_VIT_T / REUSE_GEMM_TP;

    for (int h = 0; h < VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            for (int tt = 0; tt < VIT_BIAS_TT; ++tt) {
                for (int hct = 0; hct < VIT_HC / AXI_CP; ++hct) {
                    int c_tile = h * (VIT_HC / AXI_CP) + hct;
                    for (int tp = 0; tp < REUSE_GEMM_TP; ++tp) {
                        #pragma HLS pipeline II=1
                        int off = (qkv == 0) ? off_bq_vec : ((qkv == 1) ? off_bk_vec : off_bv_vec);
                        replay_cached_bias_vec(bias_cache, off + c_tile, bias_stream);
                    }
                }
            }
        }
    }

    for (int tt = 0; tt < VIT_BIAS_TT; ++tt) {
        for (int ct = 0; ct < AXI_VIT_C / AXI_CP; ++ct) {
            for (int tp = 0; tp < REUSE_GEMM_TP; ++tp) {
                #pragma HLS pipeline II=1
                replay_cached_bias_vec(bias_cache, off_bo_vec + ct, bias_stream);
            }
        }
    }
    for (int tt = 0; tt < VIT_BIAS_TT; ++tt) {
        for (int cmt = 0; cmt < AXI_VIT_CM / AXI_CP; ++cmt) {
            for (int tp = 0; tp < REUSE_GEMM_TP; ++tp) {
                #pragma HLS pipeline II=1
                replay_cached_bias_vec(bias_cache, off_b1_vec + cmt, bias_stream);
            }
        }
    }
    for (int tt = 0; tt < VIT_BIAS_TT; ++tt) {
        for (int ct = 0; ct < AXI_VIT_C / AXI_CP; ++ct) {
            for (int tp = 0; tp < REUSE_GEMM_TP; ++tp) {
                #pragma HLS pipeline II=1
                replay_cached_bias_vec(bias_cache, off_b2_vec + ct, bias_stream);
            }
        }
    }
}

static void read_vit_bias_cached(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    axi_maxi_t* memory_vit_bias,
    hls::stream<axi_bias_vec_t>& bias_stream
) {
    if (is_llm_mode(mode)) return;

    axi_bias_pack_t bias_cache[AXI_VIT_LAYER_BIAS_VECS];
    #pragma HLS bind_storage variable=bias_cache type=RAM_1P impl=URAM

    for (int l = l_begin; l < l_close && l < VIT_L; ++l) {
        load_vit_bias_layer(l, memory_vit_bias, bias_cache);
        replay_vit_bias_layer(bias_cache, bias_stream);
    }
}

static void read_norm_params(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    axi_maxi_t* memory_llm_lnw,
    axi_maxi_t* memory_cls_lnw,
    axi_maxi_t* memory_vit_lnw,
    axi_maxi_t* memory_vit_lnb,
    hls::stream<axi_lnw_vec_t>& lnw_stream,
    hls::stream<axi_lnb_vec_t>& lnb_stream
) {
    // norm 参数相对 state/bias 很小，保持直接 pack 读取：LLM 只写 gamma，ViT 写 gamma+beta。
    for (int l = l_begin; l < l_close; ++l) {
        if (is_vit_mode(mode)) {
            if (l >= VIT_L) continue;
            for (int pass = 0; pass < 2; ++pass) {
                for (int ct = 0; ct < AXI_VIT_C / AXI_CP; ++ct) {
                    #pragma HLS pipeline II=1
                    axi_lnw_vec_t w_vec;
                    axi_lnb_vec_t b_vec;
                    int base = l * AXI_VIT_LN_PARAM_COUNT + pass * AXI_VIT_C + ct * AXI_CP;
                    axi_load_i32_vec<axi_lnw_t>(memory_vit_lnw, base, w_vec);
                    axi_load_i64_vec(memory_vit_lnb, base, b_vec);
                    lnw_stream.write(w_vec);
                    lnb_stream.write(b_vec);
                }
            }
        } else {
            bool cls = (l == LLAMA_L);
            if (l > LLAMA_L) continue;
            int pass_count = cls ? 1 : 2;
            for (int pass = 0; pass < pass_count; ++pass) {
                for (int ct = 0; ct < AXI_LLM_C / AXI_CP; ++ct) {
                    #pragma HLS pipeline II=1
                    axi_lnw_vec_t w_vec;
                    int base = pass * AXI_LLM_C + ct * AXI_CP;
                    axi_load_i32_vec<axi_lnw_t>(
                        cls ? memory_cls_lnw : memory_llm_lnw,
                        cls ? ct * AXI_CP : l * 2 * AXI_LLM_C + base,
                        w_vec
                    );
                    lnw_stream.write(w_vec);
                }
            }
        }
    }
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    AXI_PARAM_OP_T op,

    axi_maxi_t* memory_vit_bias,
    axi_maxi_t* memory_llm_lnw,
    axi_maxi_t* memory_cls_lnw,
    axi_maxi_t* memory_vit_lnw,
    axi_maxi_t* memory_vit_lnb,

    hls::stream<axi_bias_vec_t>& bias_stream,
    hls::stream<axi_lnw_vec_t>& lnw_stream,
    hls::stream<axi_lnb_vec_t>& lnb_stream
) {
    // top 只做参数搬运；op 由 PYNQ/Spinal 调度窗口选择，避免 bias/norm 两类消费者互相阻塞。
    #pragma HLS interface ap_ctrl_chain port=return

    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_vit_bias depth=VIT_L*AXI_VIT_LAYER_BIAS_PACKS offset=direct
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_llm_lnw  depth=AXI_LLM_LNW_PACKS offset=direct
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_cls_lnw  depth=AXI_CLS_LNW_PACKS offset=direct
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_vit_lnw  depth=AXI_VIT_LNW_PACKS offset=direct
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_vit_lnb  depth=AXI_VIT_LNB_PACKS offset=direct

    #pragma HLS interface axis port=bias_stream
    #pragma HLS interface axis port=lnw_stream
    #pragma HLS interface axis port=lnb_stream
    #pragma HLS aggregate variable=bias_stream compact=bit
    #pragma HLS aggregate variable=lnw_stream compact=bit
    #pragma HLS aggregate variable=lnb_stream compact=bit

    if (op == AXI_PARAM_BIAS) {
        read_vit_bias_cached(mode, l_begin, l_close, memory_vit_bias, bias_stream);
    } else {
        read_norm_params(mode, l_begin, l_close, memory_llm_lnw, memory_cls_lnw,
                         memory_vit_lnw, memory_vit_lnb, lnw_stream, lnb_stream);
    }
}

#ifndef __SYNTHESIS__
static void drain_bias_stream(hls::stream<axi_bias_vec_t>& bias_stream, int count) {
    // step1 smoke test 只验证流量不会阻塞；数值一致性后续由端到端链式 testbench 覆盖。
    for (int i = 0; i < count; ++i) {
        bias_stream.read();
    }
}

static void drain_lnw_stream(hls::stream<axi_lnw_vec_t>& lnw_stream, int count) {
    for (int i = 0; i < count; ++i) {
        lnw_stream.read();
    }
}

static void drain_lnb_stream(hls::stream<axi_lnb_vec_t>& lnb_stream, int count) {
    for (int i = 0; i < count; ++i) {
        lnb_stream.read();
    }
}

static void test_layer() {
    static axi_maxi_t memory_vit_bias[VIT_L * AXI_VIT_LAYER_BIAS_PACKS];
    static axi_maxi_t memory_llm_lnw[AXI_LLM_LNW_PACKS];
    static axi_maxi_t memory_cls_lnw[AXI_CLS_LNW_PACKS];
    static axi_maxi_t memory_vit_lnw[AXI_VIT_LNW_PACKS];
    static axi_maxi_t memory_vit_lnb[AXI_VIT_LNB_PACKS];

    for (int i = 0; i < VIT_L * AXI_VIT_LAYER_BIAS_PACKS; ++i) memory_vit_bias[i] = i;
    for (int i = 0; i < AXI_LLM_LNW_PACKS; ++i) memory_llm_lnw[i] = i;
    for (int i = 0; i < AXI_CLS_LNW_PACKS; ++i) memory_cls_lnw[i] = i;
    for (int i = 0; i < AXI_VIT_LNW_PACKS; ++i) memory_vit_lnw[i] = i;
    for (int i = 0; i < AXI_VIT_LNB_PACKS; ++i) memory_vit_lnb[i] = i;

    hls::stream<axi_bias_vec_t> bias_stream;
    hls::stream<axi_lnw_vec_t> lnw_stream;
    hls::stream<axi_lnb_vec_t> lnb_stream;

    // LLM decoder norm：每层两次 RMSNorm，只产生 gamma。
    top(MODE_LLM, 0, 1, AXI_PARAM_NORM,
        memory_vit_bias, memory_llm_lnw, memory_cls_lnw, memory_vit_lnw, memory_vit_lnb,
        bias_stream, lnw_stream, lnb_stream);
    drain_lnw_stream(lnw_stream, 2 * AXI_LLM_C / AXI_CP);

    // LLM CLS/lm_head norm：只产生一次 gamma。
    top(MODE_LLM, LLAMA_L, LLAMA_L + 1, AXI_PARAM_NORM,
        memory_vit_bias, memory_llm_lnw, memory_cls_lnw, memory_vit_lnw, memory_vit_lnb,
        bias_stream, lnw_stream, lnb_stream);
    drain_lnw_stream(lnw_stream, AXI_LLM_C / AXI_CP);

    // ViT LayerNorm：每层两次，gamma/beta 同步输出。
    top(MODE_VIT, 0, 1, AXI_PARAM_NORM,
        memory_vit_bias, memory_llm_lnw, memory_cls_lnw, memory_vit_lnw, memory_vit_lnb,
        bias_stream, lnw_stream, lnb_stream);
    drain_lnw_stream(lnw_stream, 2 * AXI_VIT_C / AXI_CP);
    drain_lnb_stream(lnb_stream, 2 * AXI_VIT_C / AXI_CP);

    // ViT DEMUX bias：每层 DDR 读一次后按 Q/K/V/O/FC1/FC2 输出顺序重放。
    top(MODE_VIT, 0, 1, AXI_PARAM_BIAS,
        memory_vit_bias, memory_llm_lnw, memory_cls_lnw, memory_vit_lnw, memory_vit_lnb,
        bias_stream, lnw_stream, lnb_stream);
    constexpr int VIT_BIAS_REPLAY =
        VIT_H * 3 * (AXI_VIT_T / REUSE_GEMM_TP) * (VIT_HC / AXI_CP) * REUSE_GEMM_TP +
        (AXI_VIT_T / REUSE_GEMM_TP) * (AXI_VIT_C / AXI_CP) * REUSE_GEMM_TP +
        (AXI_VIT_T / REUSE_GEMM_TP) * (AXI_VIT_CM / AXI_CP) * REUSE_GEMM_TP +
        (AXI_VIT_T / REUSE_GEMM_TP) * (AXI_VIT_C / AXI_CP) * REUSE_GEMM_TP;
    drain_bias_stream(bias_stream, VIT_BIAS_REPLAY);
}

int main() {
    test_layer();
    return 0;
}
#endif
