#include "../src/reuse_common.h"
#include "../src/utils.h"

// ============================================================================
// LLM/ViT 复用 RV_GEMM
// ============================================================================
// 功能：
//   1. 输入统一为 SOFTMAX 输出的 R_Q/R_S，以及 DEMUX 输出的未量化 V。
//   2. LLM mode 先读入历史 V cache，再把当前 decode tile 的 V 量化后覆盖到 cache。
//   3. ViT mode 直接量化完整 1024-token V，cache 端口保持 no-op。
//
// 资源意图：
//   - V 量化、R@V MAC、scale shift、A 输出量化均只有一套共享实现。
//   - mode 分支只放在 head/layer 粒度，用于选择 token 数、截断位、V 输入顺序和 cache 是否启用。
//   - ViT V 输入保持 DEMUX 顺序 H -> TT -> HCT -> TP_D x CP；LLM 视为 8-token tile。
//
// 与 QK_GEMM 的 k_cur 对照：整段序列上的「主表」是 vq_buf/vs_buf（等价于 QK 的 k_buf），
// 当前 tile 的未量化 V 只在每个 HCT 步进内用小块 raw_tile[P_CP][P_SP] 聚齐后做 V 量化，
// quantize_v_group 直接把结果写进 vq_buf 对应列，故不需要再声明一份与 k_cur 对称的「整 tile v_cur」。

constexpr int P_CP         = REUSE_CP;
constexpr int P_SP         = REUSE_CP;
constexpr int P_HC         = LLAMA_HC;
constexpr int P_HCT        = P_HC / P_CP;
constexpr int P_S          = LLAMA_S;
constexpr int P_ST         = P_S / P_SP;
constexpr int P_LLM_T      = REUSE_LLM_TILE_T;
constexpr int P_LLM_TST    = P_LLM_T / P_SP;
constexpr int P_LLM_POS    = 96;
constexpr int P_LLM_TOP_POS = P_LLM_POS + P_LLM_T - 1;
constexpr int P_LLM_H      = LLAMA_H;
constexpr int P_LLM_KVH    = LLAMA_KVH;
constexpr int P_LLM_GQA    = P_LLM_H / P_LLM_KVH;
constexpr int P_VIT_T      = VIT_S;
constexpr int P_VIT_H      = VIT_H;
constexpr int P_VIT_TT_D   = P_VIT_T / P_CP;
constexpr int P_MAX_H      = const_max(P_LLM_H, P_VIT_H);
constexpr int P_MAX_T      = const_max(P_LLM_T, P_VIT_T);
constexpr int P_MAX_DW_A   = const_max(DW_A, VIT_DW_A);

static_assert(P_HC == VIT_HC, "RV_GEMM reuse requires identical LLM/ViT head dim");
static_assert(P_S == VIT_S, "RV_GEMM reuse assumes LLM cache length and ViT token count both use 1024");
static_assert(DW_A_TRUNC == VIT_DW_A_TRUNC, "RV_GEMM reuse expects shared A output width");
static_assert(P_LLM_T == P_SP, "LLM RV_GEMM cache update assumes one 8-token group per decode tile");

typedef REUSE_AQ_T       rv_q_t;
typedef REUSE_AS_T       rv_s_t;
typedef REUSE_V_TRUNC_T  rv_v_t;
typedef ap_int<P_MAX_DW_A> rv_acc_t;
typedef REUSE_A_TRUNC_T  rv_a_t;

constexpr int P_LLM_NUM_RQ       = P_LLM_H * P_LLM_T * P_S;
constexpr int P_LLM_NUM_RS       = P_LLM_H * P_LLM_T * P_ST;
constexpr int P_LLM_NUM_V        = P_LLM_KVH * P_LLM_T * P_HC;
constexpr int P_LLM_NUM_VQ_CACHE = P_LLM_KVH * P_HC * P_S;
constexpr int P_LLM_NUM_VS_CACHE = P_LLM_KVH * P_HC * P_ST;
constexpr int P_LLM_NUM_VQ_CUR   = P_LLM_KVH * P_HC * P_LLM_T;
constexpr int P_LLM_NUM_VS_CUR   = P_LLM_KVH * P_HC * P_LLM_TST;
constexpr int P_LLM_NUM_AQ       = P_LLM_H * P_LLM_T * P_HC;
constexpr int P_LLM_NUM_AS       = P_LLM_H * P_LLM_T * P_HCT;

constexpr int P_VIT_NUM_RQ       = P_VIT_H * P_VIT_T * P_S;
constexpr int P_VIT_NUM_RS       = P_VIT_H * P_VIT_T * P_ST;
constexpr int P_VIT_NUM_V        = P_VIT_H * P_VIT_T * P_HC;
constexpr int P_VIT_NUM_AQ       = P_VIT_H * P_VIT_T * P_HC;
constexpr int P_VIT_NUM_AS       = P_VIT_H * P_VIT_T * P_HCT;

void load_llm_v_cache_q(
    hls::stream<hls::vector<rv_q_t, P_SP> >& cache_i_stream,
    rv_q_t vq_buf[P_HC][P_S]
) {
    // LLM V cache 输入顺序保持旧 RV_GEMM：[HC][S/SP] x SP。
    for (int c = 0; c < P_HC; ++c) {
        for (int st = 0; st < P_ST; ++st) {
            #pragma HLS pipeline II=1
            hls::vector<rv_q_t, P_SP> cache_vec = cache_i_stream.read();
            for (int sp = 0; sp < P_SP; ++sp) {
                #pragma HLS unroll
                vq_buf[c][st * P_SP + sp] = cache_vec[sp];
            }
        }
    }
}

void load_llm_v_cache_s(
    hls::stream<hls::vector<rv_s_t, 1> >& cache_i_stream,
    rv_s_t vs_buf[P_HC][P_ST]
) {
    // scale cache 每个 channel、每 8 个 token 一个 scale。
    for (int c = 0; c < P_HC; ++c) {
        for (int st = 0; st < P_ST; ++st) {
            #pragma HLS pipeline II=1
            hls::vector<rv_s_t, 1> cache_vec = cache_i_stream.read();
            vs_buf[c][st] = cache_vec[0];
        }
    }
}

void load_llm_v_cache_qs(
    hls::stream<hls::vector<rv_q_t, P_SP> >& q_cache_i_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& s_cache_i_stream,
    rv_q_t q_buf[P_HC][P_S],
    rv_s_t s_buf[P_HC][P_ST],
    int valid_st
) {
    // KV_CACHE 读 V cache 时同拍输出 VQ/VS；LLM causal windowing 只消费当前可见
    // cache 前缀，和 SOFTMAX 输出的 R 行长度保持一致。
    const ap_uint<8> valid_st_u = valid_st;
    for (int c = 0; c < P_HC; ++c) {
        #pragma HLS loop_flatten off
        ap_uint<11> s_base = 0;
        for (ap_uint<8> st = 0; st < valid_st_u; ++st) {
            #pragma HLS loop_tripcount min=1 max=P_ST
            #pragma HLS pipeline II=1
            hls::vector<rv_q_t, P_SP> q_vec = q_cache_i_stream.read();
            hls::vector<rv_s_t, 1> s_vec = s_cache_i_stream.read();
            for (int sp = 0; sp < P_SP; ++sp) {
                #pragma HLS unroll
                q_buf[c][s_base + sp] = q_vec[sp];
            }
            s_buf[c][st] = s_vec[0];
            s_base += P_SP;
        }
    }
}

void quantize_v_group(
    int hct,
    int st,
    rv_v_t raw_tile[P_CP][P_SP],
    bool write_cache_stream,
    rv_q_t vq_buf[P_HC][P_S],
    rv_s_t vs_buf[P_HC][P_ST],
    hls::stream<hls::vector<rv_q_t, P_SP> >& vq_cache_o_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& vs_cache_o_stream
) {
    // 共享 V 量化：
    //   raw_tile[CP][SP] 表示同一 HCT 的 8 个 channel、8 个 token。
    //   每个 channel 横跨 8 个 token 共享一个 scale，与旧 QUANTIZER<w_t,...,C=8/1024> 行为一致。
    constexpr int Q_BITS = rv_q_t::width;
    constexpr int Q_MAX = +(1 << (Q_BITS - 1)) - 1;
    constexpr int Q_MIN = -(1 << (Q_BITS - 1));

    for (int cp = 0; cp < P_CP; ++cp) {
        #pragma HLS pipeline II=1
        int c = hct * P_CP + cp;
        rv_v_t abs_max = 0;
        for (int sp = 0; sp < P_SP; ++sp) {
            #pragma HLS unroll
            abs_max = max(abs_max, (rv_v_t)abs(raw_tile[cp][sp]));
        }

        int8_t s_val = log2ceil(abs_max) - (Q_BITS - 1);
        rv_s_t scale = clamp(s_val, 0, 15);
        hls::vector<rv_q_t, P_SP> q_vec;
        hls::vector<rv_s_t, 1> s_vec;
        s_vec[0] = scale;

        for (int sp = 0; sp < P_SP; ++sp) {
            #pragma HLS unroll
            rv_v_t q_val = raw_tile[cp][sp];
            if (scale != 0) {
                q_val = q_val >> (scale - 1);
                q_val = q_val + 1;
                q_val = q_val >> 1;
            }
            // V/A 激活是 signed int8；scale 饱和时必须和软件一样双向 clamp。
            rv_q_t q_out = (rv_q_t)clamp(q_val, (rv_v_t)Q_MIN, (rv_v_t)Q_MAX);
            vq_buf[c][st * P_SP + sp] = q_out;
            q_vec[sp] = q_out;
        }
        vs_buf[c][st] = scale;

        if (write_cache_stream) {
            vq_cache_o_stream.write(q_vec);
            vs_cache_o_stream.write(s_vec);
        }
    }
}

void load_quantize_llm_v(
    int chunk,
    hls::stream<hls::vector<rv_v_t, P_CP> >& v_stream,
    rv_q_t vq_buf[P_HC][P_S],
    rv_s_t vs_buf[P_HC][P_ST],
    hls::stream<hls::vector<rv_q_t, P_SP> >& vq_cache_o_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& vs_cache_o_stream
) {
    // LLM V 来自 DEMUX 当前 8-token tile，顺序为 HCT -> T x CP。
    // 量化后覆盖 chunk 对应位置，并把当前 tile 的 V_Q/V_S 输出给 KV cache 调度。
    // raw_tile：仅本 HCT 的 8×8 原始元素 staging；与 QK 里 k_cur 不同——K 已是量化流，需整 tile
    // 先装入 k_cur 再 merge；此处量化逻辑必须看满 8 token 才能定 scale，写回目标仍是外层 vq_buf。
    int st = chunk;
    for (int hct = 0; hct < P_HCT; ++hct) {
        rv_v_t raw_tile[P_CP][P_SP];
        #pragma HLS array_partition variable=raw_tile complete dim=1
        #pragma HLS array_partition variable=raw_tile complete dim=2

        for (int t = 0; t < P_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            hls::vector<rv_v_t, P_CP> v_vec = v_stream.read();
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                raw_tile[cp][t] = v_vec[cp];
            }
        }
        quantize_v_group(hct, st, raw_tile, true, vq_buf, vs_buf, vq_cache_o_stream, vs_cache_o_stream);
    }
}

void load_quantize_vit_v(
    hls::stream<hls::vector<rv_v_t, P_CP> >& v_stream,
    rv_q_t vq_buf[P_HC][P_S],
    rv_s_t vs_buf[P_HC][P_ST],
    hls::stream<hls::vector<rv_q_t, P_SP> >& vq_cache_o_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& vs_cache_o_stream
) {
    // ViT V 保持 DEMUX 格式：TT -> HCT -> TP_D，TP_D=8 个 token，每拍 CP=8 个 channel。
    // cache 端口 no-op，因此量化结果只写内部 V^T buffer，不产生 cache stream。
    for (int tt = 0; tt < P_VIT_TT_D; ++tt) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            rv_v_t raw_tile[P_CP][P_SP];
            #pragma HLS array_partition variable=raw_tile complete dim=1
            #pragma HLS array_partition variable=raw_tile complete dim=2

            for (int ti = 0; ti < P_CP; ++ti) {
                #pragma HLS pipeline II=1
                hls::vector<rv_v_t, P_CP> v_vec = v_stream.read();
                for (int cp = 0; cp < P_CP; ++cp) {
                    #pragma HLS unroll
                    raw_tile[cp][ti] = v_vec[cp];
                }
            }
            quantize_v_group(hct, tt, raw_tile, false, vq_buf, vs_buf, vq_cache_o_stream, vs_cache_o_stream);
        }
    }
}

void load_r_row(
    hls::stream<hls::vector<rv_q_t, P_SP> >& rq_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& rs_stream,
    rv_q_t rq_buf[P_S],
    rv_s_t rs_buf[P_ST],
    int valid_st
) {
    // R_Q/R_S 来自 SOFTMAX，两个 mode 都按 [H][T][S/SP] 顺序逐行输入。
    for (int st = 0; st < valid_st; ++st) {
        #pragma HLS loop_tripcount min=1 max=P_ST
        #pragma HLS pipeline II=1
        hls::vector<rv_q_t, P_SP> rq_vec = rq_stream.read();
        hls::vector<rv_s_t, 1> rs_vec = rs_stream.read();
        for (int sp = 0; sp < P_SP; ++sp) {
            #pragma HLS unroll
            rq_buf[st * P_SP + sp] = rq_vec[sp];
        }
        rs_buf[st] = rs_vec[0];
    }
}

void quantize_a_vec(
    hls::vector<rv_a_t, P_CP> a_vec,
    hls::stream<hls::vector<rv_q_t, P_CP> >& aq_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& as_stream
) {
    // 共享 A 输出量化：每个 token 的 8 个 head-channel lane 共用一个 scale。
    constexpr int Q_BITS = rv_q_t::width;
    constexpr int Q_MAX = +(1 << (Q_BITS - 1)) - 1;
    constexpr int Q_MIN = -(1 << (Q_BITS - 1));

    rv_a_t abs_max = 0;
    for (int cp = 0; cp < P_CP; ++cp) {
        #pragma HLS unroll
        abs_max = max(abs_max, (rv_a_t)abs(a_vec[cp]));
    }

    int8_t s_val = log2ceil(abs_max) - (Q_BITS - 1);
    rv_s_t scale = clamp(s_val, 0, 15);
    hls::vector<rv_q_t, P_CP> q_vec;
    hls::vector<rv_s_t, 1> s_vec;
    s_vec[0] = scale;

    for (int cp = 0; cp < P_CP; ++cp) {
        #pragma HLS unroll
        rv_a_t q_val = a_vec[cp];
        if (scale != 0) {
            q_val = q_val >> (scale - 1);
            q_val = q_val + 1;
            q_val = q_val >> 1;
        }
        // V/A 激活是 signed int8；scale 饱和时必须和软件一样双向 clamp。
        q_vec[cp] = (rv_q_t)clamp(q_val, (rv_a_t)Q_MIN, (rv_a_t)Q_MAX);
    }

    aq_stream.write(q_vec);
    as_stream.write(s_vec);
}

void shared_rv_bmm_head(
    int num_tokens,
    int valid_st,
    int trunc,
    rv_q_t vq_buf[P_HC][P_S],
    rv_s_t vs_buf[P_HC][P_ST],
    hls::stream<hls::vector<rv_q_t, P_SP> >& rq_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& rs_stream,
    hls::stream<hls::vector<rv_q_t, P_CP> >& aq_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& as_stream
) {
    // 单套 RV BMM 内核：
    //   每行 R[T,S] 与转置量化后的 V[HC,S] 相乘，CP=8 个输出 channel 并行。
    //   LLM/ViT 只改变 num_tokens 和 trunc，MAC/shift/psum/A-Quantizer 共用。
    rv_q_t rq_buf[P_S];
    rv_s_t rs_buf[P_ST];
    // R 行缓存是 1024-token、按 SP=8 分 bank 的局部表；K26 OOC 尝试改 BRAM 后
    // BRAM 增至 10 且 CP=2.781ns，时序变差，因此仍保持 LUTRAM。
    #pragma HLS bind_storage variable=rq_buf type=ram_2p impl=lutram
    #pragma HLS bind_storage variable=rs_buf type=ram_2p impl=lutram
    #pragma HLS array_partition variable=rq_buf cyclic factor=P_SP dim=1

    for (int t = 0; t < P_MAX_T; ++t) {
        #pragma HLS loop_tripcount min=P_LLM_T max=P_VIT_T
        if (t >= num_tokens) {
            continue;
        }

        load_r_row(rq_stream, rs_stream, rq_buf, rs_buf, valid_st);

        for (int hct = 0; hct < P_HCT; ++hct) {
            rv_acc_t psum_vec[P_CP];
            #pragma HLS array_partition variable=psum_vec complete
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                psum_vec[cp] = 0;
            }

            for (int st = 0; st < valid_st; ++st) {
                #pragma HLS loop_tripcount min=1 max=P_ST
                #pragma HLS pipeline II=1
                rv_acc_t tile_psum[P_CP];
                #pragma HLS array_partition variable=tile_psum complete
                for (int cp = 0; cp < P_CP; ++cp) {
                    #pragma HLS unroll
                    tile_psum[cp] = 0;
                }

                for (int cp = 0; cp < P_CP; ++cp) {
                    for (int sp = 0; sp < P_SP; ++sp) {
                        #pragma HLS unroll
                        int s = st * P_SP + sp;
                        rv_acc_t mul_res = rq_buf[s] * vq_buf[hct * P_CP + cp][s];
                        #pragma HLS bind_op variable=mul_res op=mul impl=dsp
                        tile_psum[cp] += mul_res;
                    }
                }

                rv_s_t r_shift = rs_buf[st];
                for (int cp = 0; cp < P_CP; ++cp) {
                    #pragma HLS unroll
                    rv_s_t v_shift = vs_buf[hct * P_CP + cp][st];
                    psum_vec[cp] += tile_psum[cp] << (r_shift + v_shift);
                }
            }

            hls::vector<rv_a_t, P_CP> a_vec;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                a_vec[cp] = (rv_a_t)(psum_vec[cp] >> trunc);
            }
            quantize_a_vec(a_vec, aq_stream, as_stream);
        }
    }
}

void rv_gemm_one_layer(
    REUSE_MODE_T mode,
    int pos,
    hls::stream<hls::vector<rv_q_t, P_SP> >& rq_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& rs_stream,
    hls::stream<hls::vector<rv_v_t, P_CP> >& v_stream,
    hls::stream<hls::vector<rv_q_t, P_SP> >& vq_cache_i_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& vs_cache_i_stream,
    hls::stream<hls::vector<rv_q_t, P_SP> >& vq_cache_o_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& vs_cache_o_stream,
    hls::stream<hls::vector<rv_q_t, P_CP> >& aq_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& as_stream
) {
    // ViT keeps one V table per attention head. LLM uses native GQA: one V table
    // per KV head, reused for the three query heads that share it.
    bool is_vit = is_vit_mode(mode);
    reuse_pos_t decode_pos = reuse_decode_pos(mode, pos);
    int chunk = decode_pos.chunk;
    int valid_st = reuse_attention_valid_st(mode, pos);
    int trunc = is_vit ? VIT_MHA_TRUNC_A : MHA_TRUNC_A;

    rv_q_t vq_buf[P_HC][P_S];
    rv_s_t vs_buf[P_HC][P_ST];
    #pragma HLS bind_storage variable=vq_buf type=RAM_2P impl=URAM latency=2
    #pragma HLS bind_storage variable=vs_buf type=RAM_2P impl=BRAM latency=2
    #pragma HLS array_reshape variable=vq_buf cyclic factor=P_CP dim=1
    #pragma HLS array_partition variable=vq_buf cyclic factor=P_SP dim=2
    #pragma HLS array_reshape variable=vs_buf cyclic factor=P_CP dim=1

    if (is_vit) {
        for (int h = 0; h < P_VIT_H; ++h) {
            #pragma HLS loop_tripcount min=P_VIT_H max=P_VIT_H
            load_quantize_vit_v(v_stream, vq_buf, vs_buf, vq_cache_o_stream, vs_cache_o_stream);
            shared_rv_bmm_head(P_VIT_T, valid_st, trunc, vq_buf, vs_buf, rq_stream, rs_stream, aq_stream, as_stream);
        }
    } else {
        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            #pragma HLS loop_tripcount min=P_LLM_KVH max=P_LLM_KVH
            load_llm_v_cache_qs(vq_cache_i_stream, vs_cache_i_stream, vq_buf, vs_buf, valid_st);
            load_quantize_llm_v(chunk, v_stream, vq_buf, vs_buf, vq_cache_o_stream, vs_cache_o_stream);
            for (int r = 0; r < P_LLM_GQA; ++r) {
                #pragma HLS loop_tripcount min=P_LLM_GQA max=P_LLM_GQA
                shared_rv_bmm_head(P_LLM_T, valid_st, trunc, vq_buf, vs_buf, rq_stream, rs_stream, aq_stream, as_stream);
            }
        }
    }
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    int pos,
    hls::stream<hls::vector<rv_q_t, P_SP> >& rq_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& rs_stream,
    hls::stream<hls::vector<rv_v_t, P_CP> >& v_stream,
    hls::stream<hls::vector<rv_q_t, P_SP> >& vq_cache_i_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& vs_cache_i_stream,
    hls::stream<hls::vector<rv_q_t, P_SP> >& vq_cache_o_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& vs_cache_o_stream,
    hls::stream<hls::vector<rv_q_t, P_CP> >& aq_stream,
    hls::stream<hls::vector<rv_s_t, 1> >& as_stream
) {
    // 冻结后的 RV_GEMM 顶层接口：
    //   mode=0 为 LLM，mode=1 为 ViT；pos 是 LLM decode 绝对位置，ViT mode 忽略。
    //   ViT mode 不读写 V cache 端口，避免误触发 LLM KV cache 资源。
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=rq_stream
    #pragma HLS interface axis port=rs_stream
    #pragma HLS interface axis port=v_stream
    #pragma HLS interface axis port=vq_cache_i_stream
    #pragma HLS interface axis port=vs_cache_i_stream
    #pragma HLS interface axis port=vq_cache_o_stream
    #pragma HLS interface axis port=vs_cache_o_stream
    #pragma HLS interface axis port=aq_stream
    #pragma HLS interface axis port=as_stream
    #pragma HLS aggregate variable=rq_stream         compact=bit
    #pragma HLS aggregate variable=rs_stream         compact=bit
    #pragma HLS aggregate variable=v_stream          compact=bit
    #pragma HLS aggregate variable=vq_cache_i_stream compact=bit
    #pragma HLS aggregate variable=vs_cache_i_stream compact=bit
    #pragma HLS aggregate variable=vq_cache_o_stream compact=bit
    #pragma HLS aggregate variable=vs_cache_o_stream compact=bit
    #pragma HLS aggregate variable=aq_stream         compact=bit
    #pragma HLS aggregate variable=as_stream         compact=bit

    for (int l = l_begin; l < l_close; ++l) {
        #pragma HLS loop_tripcount min=1 max=32
        if (!reuse_is_valid_layer(mode, l, false)) {
            continue;
        }
        rv_gemm_one_layer(
            mode,
            pos,
            rq_stream,
            rs_stream,
            v_stream,
            vq_cache_i_stream,
            vs_cache_i_stream,
            vq_cache_o_stream,
            vs_cache_o_stream,
            aq_stream,
            as_stream
        );
    }
}

// ============================================================================
// Testbench
// ============================================================================
// step1 同时覆盖：
//   - LLM mode decoder layer 0：SOFTMAX_R + DEMUX_V + V cache -> A 和当前 V cache。
//   - ViT mode vision layer 0：SOFTMAX_R + DEMUX_V -> A，cache 端口保持空。

namespace llm_rv_gemm_tb {

void test_layer(int l) {
    int pos = reuse_tb_llm_pos(P_LLM_POS);
    int top_pos = reuse_tb_llm_top_pos(pos + P_LLM_T - 1);
    int chunk = top_pos / P_LLM_T;
    int valid_s = reuse_llm_valid_s(top_pos);
    int valid_st = (valid_s + P_SP - 1) / P_SP;
    int valid_s_padded = valid_st * P_SP;
    int valid_num_rq = P_LLM_H * P_LLM_T * valid_s_padded;
    int valid_num_rs = P_LLM_H * P_LLM_T * valid_st;
    int valid_num_vq_cache = P_LLM_KVH * P_HC * valid_s_padded;
    int valid_num_vs_cache = P_LLM_KVH * P_HC * valid_st;

    const string file_path = BINARIES_PATH + to_string(l);
    const string save_path = CONDENSE_PATH + to_string(l);

    vector<int8_t>  ref_vq_cur(P_LLM_NUM_VQ_CUR);
    vector<int8_t>  ref_vs_cur(P_LLM_NUM_VS_CUR);
    vector<int8_t>  ref_vq_cache(P_LLM_NUM_VQ_CACHE);
    vector<int8_t>  ref_vs_cache(P_LLM_NUM_VS_CACHE);
    vector<int8_t>  ref_vq_cache_valid(valid_num_vq_cache);
    vector<int8_t>  ref_vs_cache_valid(valid_num_vs_cache);
    vector<int8_t>  ref_aq(P_LLM_NUM_AQ);
    vector<int8_t>  ref_as(P_LLM_NUM_AS);
    vector<int8_t>  condensed_rq(valid_num_rq);
    vector<int8_t>  condensed_rs(valid_num_rs);
    vector<int8_t>  condensed_rq_full(P_LLM_NUM_RQ);
    vector<int8_t>  condensed_rs_full(P_LLM_NUM_RS);
    vector<int64_t> condensed_v(P_LLM_NUM_V);
    vector<int8_t>  dut_vq_cur(P_LLM_NUM_VQ_CUR);
    vector<int8_t>  dut_vs_cur(P_LLM_NUM_VS_CUR);
    vector<int8_t>  dut_aq(P_LLM_NUM_AQ);
    vector<int8_t>  dut_as(P_LLM_NUM_AS);

    {
        // 软件参考按绝对 POS 截取当前 8-token tile，并构造因果 V cache 输入。
        auto V_RAW = read_tensor<int64_t>(file_path + "/MHA_V_SPLIT_HEADS.bin");
        auto VT_Q  = read_tensor<int8_t >(file_path + "/MHA_VT_Q.bin");
        auto VT_S  = read_tensor<int8_t >(file_path + "/MHA_VT_S.bin");
        auto R_Q   = read_tensor<int8_t >(file_path + "/MHA_R_Q.bin");
        auto R_S   = read_tensor<int8_t >(file_path + "/MHA_R_S.bin");
        auto A_Q   = read_tensor<int8_t >(file_path + "/MHA_A_Q.bin");
        auto A_S   = read_tensor<int8_t >(file_path + "/MHA_A_S.bin");
        // V cache 参考按 H -> HC -> S_LOAD 保存；A_Q/A_S 按 token 文件长度保存。
        // 当前 weights 里 S_LOAD 通常为 880、T_LOAD 为 877，均按文件长度动态推导。
        int v_t_load  = infer_tensor_t_load<int64_t>(V_RAW, P_LLM_H, P_HC, "MHA_V_SPLIT_HEADS");
        int vq_s_load = infer_tensor_extent<int8_t>(VT_Q, P_LLM_H * P_HC, "MHA_VT_Q");
        int vs_st_load = infer_tensor_extent<int8_t>(VT_S, P_LLM_H * P_HC, "MHA_VT_S");
        int rq_s_load = infer_tensor_extent<int8_t>(R_Q, P_LLM_H * v_t_load, "MHA_R_Q");
        int rs_st_load = infer_tensor_extent<int8_t>(R_S, P_LLM_H * v_t_load, "MHA_R_S");
        int aq_t_load = infer_tensor_t_load<int8_t>(A_Q, 1, P_LLM_H * P_HC,  "MHA_A_Q");
        int as_t_load = infer_tensor_t_load<int8_t>(A_S, 1, P_LLM_H * P_HCT, "MHA_A_S");
        check_tensor_window(v_t_load, pos, P_LLM_T, "MHA_V_SPLIT_HEADS");
        check_tensor_window(vq_s_load, pos, P_LLM_T, "MHA_VT_Q");
        check_tensor_window(vs_st_load, pos / P_SP, P_LLM_TST, "MHA_VT_S");
        check_tensor_window(v_t_load, pos, P_LLM_T, "MHA_R_Q/MHA_R_S");
        check_tensor_window(aq_t_load, pos, P_LLM_T, "MHA_A_Q");
        check_tensor_window(as_t_load, pos, P_LLM_T, "MHA_A_S");

        tensor2array<int8_t>(R_Q, condensed_rq_full.data(), P_LLM_H, P_LLM_H, v_t_load, pos, P_LLM_T, rq_s_load, P_S);
        tensor2array<int8_t>(R_S, condensed_rs_full.data(), P_LLM_H, P_LLM_H, v_t_load, pos, P_LLM_T, rs_st_load, P_ST);
        tensor2array<int8_t>(A_Q,  ref_aq.data(),       1, 1, aq_t_load, pos, P_LLM_T, P_LLM_H * P_HC,  P_LLM_H * P_HC);
        tensor2array<int8_t>(A_S,  ref_as.data(),       1, 1, as_t_load, pos, P_LLM_T, P_LLM_H * P_HCT, P_LLM_H * P_HCT);

        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            int src_head = kv * P_LLM_GQA;
            for (int c = 0; c < P_HC; ++c) {
                for (int s = 0; s < P_S && s < vq_s_load; ++s) {
                    ref_vq_cache[kv * P_HC * P_S + c * P_S + s] =
                        VT_Q[src_head * P_HC * vq_s_load + c * vq_s_load + s];
                }
                for (int st = 0; st < P_ST && st < vs_st_load; ++st) {
                    ref_vs_cache[kv * P_HC * P_ST + c * P_ST + st] =
                        VT_S[src_head * P_HC * vs_st_load + c * vs_st_load + st];
                }
                for (int t = 0; t < P_LLM_T && pos + t < vq_s_load; ++t) {
                    ref_vq_cur[kv * P_HC * P_LLM_T + c * P_LLM_T + t] =
                        VT_Q[src_head * P_HC * vq_s_load + c * vq_s_load + pos + t];
                }
                for (int tst = 0; tst < P_LLM_TST && pos / P_SP + tst < vs_st_load; ++tst) {
                    ref_vs_cur[kv * P_HC * P_LLM_TST + c * P_LLM_TST + tst] =
                        VT_S[src_head * P_HC * vs_st_load + c * vs_st_load + pos / P_SP + tst];
                }
            }
        }

        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            for (int c = 0; c < P_HC; ++c) {
                for (int s = pos; s < P_S; ++s) {
                    ref_vq_cache[kv * P_HC * P_S + c * P_S + s] = 0;
                }
                for (int st = pos / P_SP; st < P_ST; ++st) {
                    ref_vs_cache[kv * P_HC * P_ST + c * P_ST + st] = 0;
                }
            }
        }
        for (int h = 0; h < P_LLM_H; ++h) {
            for (int t = 0; t < P_LLM_T; ++t) {
                for (int s = 0; s < valid_s_padded; ++s) {
                    condensed_rq[h * P_LLM_T * valid_s_padded + t * valid_s_padded + s] =
                        condensed_rq_full[h * P_LLM_T * P_S + t * P_S + s];
                }
                for (int st = 0; st < valid_st; ++st) {
                    condensed_rs[h * P_LLM_T * valid_st + t * valid_st + st] =
                        condensed_rs_full[h * P_LLM_T * P_ST + t * P_ST + st];
                }
            }
            if (h < P_LLM_KVH) {
                for (int c = 0; c < P_HC; ++c) {
                    for (int s = 0; s < valid_s_padded; ++s) {
                        ref_vq_cache_valid[h * P_HC * valid_s_padded + c * valid_s_padded + s] =
                            ref_vq_cache[h * P_HC * P_S + c * P_S + s];
                    }
                    for (int st = 0; st < valid_st; ++st) {
                        ref_vs_cache_valid[h * P_HC * valid_st + c * valid_st + st] =
                            ref_vs_cache[h * P_HC * P_ST + c * P_ST + st];
                    }
                }
            }
        }
        // Native-GQA DEMUX V stream order: KVH -> HCT -> T -> CP. 直接由 raw V dump 构造，
        // 避免被旧链式 condensed 文件的 POS/DEMUX 实验口径污染。
        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            int src_head = kv * P_LLM_GQA;
            for (int hct = 0; hct < P_HCT; ++hct) {
                for (int t = 0; t < P_LLM_T; ++t) {
                    for (int cp = 0; cp < P_CP; ++cp) {
                        int src_idx = src_head * v_t_load * P_HC + (pos + t) * P_HC + hct * P_CP + cp;
                        int dst_idx = kv * P_LLM_T * P_HC + hct * P_LLM_T * P_CP + t * P_CP + cp;
                        condensed_v[dst_idx] = V_RAW[src_idx];
                    }
                }
            }
        }
        split_heads<int8_t>(ref_aq.data(), P_LLM_H, P_LLM_T, P_HC);
        split_heads<int8_t>(ref_as.data(), P_LLM_H, P_LLM_T, P_HCT);
    }

    if (false) {
        // 输入来自前级复用 SOFTMAX/DEMUX 的 condensed 文件，保持链式仿真边界。
        auto R_Q = read_tensor<int8_t >(save_path + "/CONDENSED_SOFTMAX_QUANT_R_Q.bin");
        auto R_S = read_tensor<int8_t >(save_path + "/CONDENSED_SOFTMAX_QUANT_R_S.bin");
        auto V   = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_V.bin");
        tensor2array<int8_t >(R_Q, condensed_rq.data(), 1, 1, 1, 1, P_LLM_NUM_RQ, P_LLM_NUM_RQ);
        tensor2array<int8_t >(R_S, condensed_rs.data(), 1, 1, 1, 1, P_LLM_NUM_RS, P_LLM_NUM_RS);
        tensor2array<int64_t>(V,   condensed_v.data(),  1, 1, 1, 1, P_LLM_NUM_V,  P_LLM_NUM_V);
    }

    hls::stream<hls::vector<rv_q_t, P_SP> > rq_stream("llm_rq_stream");
    hls::stream<hls::vector<rv_s_t, 1> > rs_stream("llm_rs_stream");
    hls::stream<hls::vector<rv_v_t, P_CP> > v_stream("llm_v_stream");
    hls::stream<hls::vector<rv_q_t, P_SP> > vq_cache_i_stream("llm_vq_cache_i_stream");
    hls::stream<hls::vector<rv_s_t, 1> > vs_cache_i_stream("llm_vs_cache_i_stream");
    hls::stream<hls::vector<rv_q_t, P_SP> > vq_cache_o_stream("llm_vq_cache_o_stream");
    hls::stream<hls::vector<rv_s_t, 1> > vs_cache_o_stream("llm_vs_cache_o_stream");
    hls::stream<hls::vector<rv_q_t, P_CP> > aq_stream("llm_aq_stream");
    hls::stream<hls::vector<rv_s_t, 1> > as_stream("llm_as_stream");

    array2stream_runtime<int8_t,  rv_q_t, P_SP>(condensed_rq.data(), rq_stream, valid_num_rq);
    array2stream_runtime<int8_t,  rv_s_t, 1   >(condensed_rs.data(), rs_stream, valid_num_rs);
    array2stream<int64_t, rv_v_t, 1, 1, 1, 1, P_LLM_NUM_V,  P_CP>(condensed_v.data(),  v_stream,  "LLM V",  true);
    array2stream_runtime<int8_t,  rv_q_t, P_SP>(ref_vq_cache_valid.data(), vq_cache_i_stream, valid_num_vq_cache);
    array2stream_runtime<int8_t,  rv_s_t, 1   >(ref_vs_cache_valid.data(), vs_cache_i_stream, valid_num_vs_cache);

    save_tensor<int8_t>(save_path + "/CONDENSED_RV_GEMM_V_Q_CACHE.bin", ref_vq_cache.data(), P_LLM_NUM_VQ_CACHE);
    save_tensor<int8_t>(save_path + "/CONDENSED_RV_GEMM_V_S_CACHE.bin", ref_vs_cache.data(), P_LLM_NUM_VS_CACHE);

    ::top(MODE_LLM, l, l + 1, top_pos, rq_stream, rs_stream, v_stream,
          vq_cache_i_stream, vs_cache_i_stream, vq_cache_o_stream, vs_cache_o_stream,
          aq_stream, as_stream);

    save_condensed_tensor<int8_t, rv_q_t, P_LLM_NUM_VQ_CUR, P_SP>(save_path + "/CONDENSED_RV_GEMM_V_Q.bin", vq_cache_o_stream);
    save_condensed_tensor<int8_t, rv_s_t, P_LLM_NUM_VS_CUR, 1   >(save_path + "/CONDENSED_RV_GEMM_V_S.bin", vs_cache_o_stream);
    save_condensed_tensor<int8_t, rv_q_t, P_LLM_NUM_AQ,     P_CP>(save_path + "/CONDENSED_RV_GEMM_A_Q.bin", aq_stream);
    save_condensed_tensor<int8_t, rv_s_t, P_LLM_NUM_AS,     1   >(save_path + "/CONDENSED_RV_GEMM_A_S.bin", as_stream);

    stream2array<int8_t, rv_q_t, P_LLM_KVH, P_HC,   1, P_LLM_T, P_SP>(vq_cache_o_stream, dut_vq_cur.data(), "LLM compact VQ", true);
    stream2array<int8_t, rv_s_t, P_LLM_KVH, P_HC,   1, P_LLM_TST, 1 >(vs_cache_o_stream, dut_vs_cur.data(), "LLM compact VS", true);
    stream2array<int8_t, rv_q_t, P_LLM_H, P_LLM_T,  1, P_HC, P_CP   >(aq_stream,         dut_aq.data(),     "LLM A_Q", true);
    stream2array<int8_t, rv_s_t, P_LLM_H, P_LLM_T,  1, P_HCT, 1     >(as_stream,         dut_as.data(),     "LLM A_S", true);

    assert(rq_stream.empty());
    assert(rs_stream.empty());
    assert(v_stream.empty());
    assert(vq_cache_i_stream.empty());
    assert(vs_cache_i_stream.empty());
    assert(aq_stream.empty());
    assert(as_stream.empty());

    compare<int8_t>(ref_vq_cur.data(), dut_vq_cur.data(), P_LLM_NUM_VQ_CUR, "LLM V_Q");
    compare<int8_t>(ref_vs_cur.data(), dut_vs_cur.data(), P_LLM_NUM_VS_CUR, "LLM V_S");
    compare<int8_t>(ref_aq.data(),     dut_aq.data(),     P_LLM_NUM_AQ,     "LLM A_Q");
    compare<int8_t>(ref_as.data(),     dut_as.data(),     P_LLM_NUM_AS,     "LLM A_S");
}

} // namespace llm_rv_gemm_tb

namespace vit_rv_gemm_tb {

void test_layer(int l) {
    const string file_path = VIT_BINARIES_PATH + to_string(l);
    const string save_path = VIT_CONDENSE_PATH + to_string(l);

    vector<int8_t>  ref_aq(P_VIT_NUM_AQ);
    vector<int8_t>  ref_as(P_VIT_NUM_AS);
    vector<int8_t>  condensed_rq(P_VIT_NUM_RQ);
    vector<int8_t>  condensed_rs(P_VIT_NUM_RS);
    vector<int64_t> condensed_v(P_VIT_NUM_V);
    vector<int8_t>  dut_aq(P_VIT_NUM_AQ);
    vector<int8_t>  dut_as(P_VIT_NUM_AS);

    {
        // ViT A_Q/A_S 按 batch*patch token-major 保存；不同导出批次的 T_LOAD 按文件长度推导。
        auto A_Q = read_tensor<int8_t>(file_path + "/MHA_A_Q.bin");
        auto A_S = read_tensor<int8_t>(file_path + "/MHA_A_S.bin");
        int aq_t_load = infer_tensor_t_load<int8_t>(A_Q, 1, P_VIT_H * P_HC,  "ViT MHA_A_Q");
        int as_t_load = infer_tensor_t_load<int8_t>(A_S, 1, P_VIT_H * P_HCT, "ViT MHA_A_S");
        check_tensor_window(aq_t_load, 0, P_VIT_T, "ViT MHA_A_Q");
        check_tensor_window(as_t_load, 0, P_VIT_T, "ViT MHA_A_S");
        tensor2array<int8_t>(A_Q, ref_aq.data(), 1, 1, aq_t_load, 0, P_VIT_T, P_VIT_H * P_HC,  P_VIT_H * P_HC);
        tensor2array<int8_t>(A_S, ref_as.data(), 1, 1, as_t_load, 0, P_VIT_T, P_VIT_H * P_HCT, P_VIT_H * P_HCT);
        split_heads<int8_t>(ref_aq.data(), P_VIT_H, P_VIT_T, P_HC);
        split_heads<int8_t>(ref_as.data(), P_VIT_H, P_VIT_T, P_HCT);
    }

    {
        // ViT 输入读取冻结后的无 VIT 前缀 condensed 文件名。
        auto R_Q = read_tensor<int8_t >(save_path + "/CONDENSED_SOFTMAX_R_Q.bin");
        auto R_S = read_tensor<int8_t >(save_path + "/CONDENSED_SOFTMAX_R_S.bin");
        auto V   = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_V.bin");
        tensor2array<int8_t >(R_Q, condensed_rq.data(), 1, 1, 1, 1, P_VIT_NUM_RQ, P_VIT_NUM_RQ);
        tensor2array<int8_t >(R_S, condensed_rs.data(), 1, 1, 1, 1, P_VIT_NUM_RS, P_VIT_NUM_RS);
        tensor2array<int64_t>(V,   condensed_v.data(),  1, 1, 1, 1, P_VIT_NUM_V,  P_VIT_NUM_V);
    }

    hls::stream<hls::vector<rv_q_t, P_SP> > rq_stream("vit_rq_stream");
    hls::stream<hls::vector<rv_s_t, 1> > rs_stream("vit_rs_stream");
    hls::stream<hls::vector<rv_v_t, P_CP> > v_stream("vit_v_stream");
    hls::stream<hls::vector<rv_q_t, P_SP> > vq_cache_i_stream("vit_vq_cache_i_stream");
    hls::stream<hls::vector<rv_s_t, 1> > vs_cache_i_stream("vit_vs_cache_i_stream");
    hls::stream<hls::vector<rv_q_t, P_SP> > vq_cache_o_stream("vit_vq_cache_o_stream");
    hls::stream<hls::vector<rv_s_t, 1> > vs_cache_o_stream("vit_vs_cache_o_stream");
    hls::stream<hls::vector<rv_q_t, P_CP> > aq_stream("vit_aq_stream");
    hls::stream<hls::vector<rv_s_t, 1> > as_stream("vit_as_stream");

    array2stream<int8_t,  rv_q_t, 1, 1, 1, 1, P_VIT_NUM_RQ, P_SP>(condensed_rq.data(), rq_stream, "ViT RQ", true);
    array2stream<int8_t,  rv_s_t, 1, 1, 1, 1, P_VIT_NUM_RS, 1   >(condensed_rs.data(), rs_stream, "ViT RS", true);
    array2stream<int64_t, rv_v_t, 1, 1, 1, 1, P_VIT_NUM_V,  P_CP>(condensed_v.data(),  v_stream,  "ViT V",  true);

    ::top(MODE_VIT, l, l + 1, 0, rq_stream, rs_stream, v_stream,
          vq_cache_i_stream, vs_cache_i_stream, vq_cache_o_stream, vs_cache_o_stream,
          aq_stream, as_stream);

    save_condensed_tensor<int8_t, rv_q_t, P_VIT_NUM_AQ, P_CP>(save_path + "/CONDENSED_RV_GEMM_A_Q.bin", aq_stream);
    save_condensed_tensor<int8_t, rv_s_t, P_VIT_NUM_AS, 1   >(save_path + "/CONDENSED_RV_GEMM_A_S.bin", as_stream);

    stream2array<int8_t, rv_q_t, P_VIT_H, P_VIT_T, 1, P_HC,  P_CP>(aq_stream, dut_aq.data(), "ViT A_Q", true);
    stream2array<int8_t, rv_s_t, P_VIT_H, P_VIT_T, 1, P_HCT, 1   >(as_stream, dut_as.data(), "ViT A_S", true);

    assert(rq_stream.empty());
    assert(rs_stream.empty());
    assert(v_stream.empty());
    assert(vq_cache_i_stream.empty());
    assert(vs_cache_i_stream.empty());
    assert(vq_cache_o_stream.empty());
    assert(vs_cache_o_stream.empty());
    assert(aq_stream.empty());
    assert(as_stream.empty());

    compare<int8_t>(ref_aq.data(), dut_aq.data(), P_VIT_NUM_AQ, "ViT A_Q");
    compare<int8_t>(ref_as.data(), dut_as.data(), P_VIT_NUM_AS, "ViT A_S");
}

} // namespace vit_rv_gemm_tb

#ifndef __SYNTHESIS__
int main() {
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        llm_rv_gemm_tb::test_layer(0);
        printf("LLM RV_GEMM layer 0 passed\n");
    }

    vit_rv_gemm_tb::test_layer(vit_layer);
    printf("ViT RV_GEMM layer %d passed\n", vit_layer);

    return 0;
}
#endif
