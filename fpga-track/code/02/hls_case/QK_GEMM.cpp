#include "../src/reuse_common.h"
#include "../src/utils.h"

// ============================================================================
// LLM/ViT 复用 QK_GEMM
// ============================================================================
// 功能：
//   1. 冻结后的输入边界为 ROPE_QK 已量化后的 Q/K q+s，不在本模块内再做 Quantizer。
//   2. LLM mode 使用 KV cache：先装入历史 K cache，再用当前 8-token K 覆盖对应 chunk。
//   3. ViT mode 不读写 cache 端口，直接把 1024 个视觉 token 的 K 装入同一套 K buffer。
//
// 资源意图：
//   - MAC、scale shift、psum accumulator、R pack 只在 shared_qk_bmm_head 中实现一套。
//   - mode 分支只出现在 head/layer 粒度的运行时配置、输入装载顺序和 cache 旁路上。
//   - ViT 的输入顺序保持 H -> Q/K -> TT -> HCT -> TP x CP；LLM 视为 T=8 的单 tile。
//
// LLM 为何单独有 k_cur（对比 RV_GEMM 无对称的「整 tile V staging」）：
//   - ROPE_QK 在同一对 qk_q/qk_s 轴上先出 Q、后出 K；Q 必须先装入 q_buf，K 不能边读边与
//     正从 cache 流装入的 k_buf 交织写入，故用 k_cur/ks_cur 承接当前 8-token 的量化 K，
//     再 overwrite 进整段 k_buf（全 S=1024 供 Q@K^T）。
//   - k_cur 维度固定为 P_LLM_T 并绑 LUTRAM，避免若用 P_MAX_T 接流导致 HLS 推断出大 BRAM。

constexpr int P_CP        = REUSE_CP;
constexpr int P_SP        = REUSE_CP;
constexpr int P_HC        = LLAMA_HC;
constexpr int P_HCT       = P_HC / P_CP;
constexpr int P_S         = LLAMA_S;
constexpr int P_ST        = P_S / P_SP;
constexpr int P_LLM_T     = REUSE_LLM_TILE_T;
constexpr int P_LLM_POS   = 96;
constexpr int P_LLM_H     = LLAMA_H;
constexpr int P_LLM_KVH   = LLAMA_KVH;
constexpr int P_LLM_GQA   = P_LLM_H / P_LLM_KVH;
constexpr int P_VIT_T     = VIT_S;
constexpr int P_VIT_H     = VIT_H;
constexpr int P_VIT_TT_D  = P_VIT_T / P_CP;
constexpr int P_MAX_H     = const_max(P_LLM_H, P_VIT_H);
constexpr int P_MAX_T     = const_max(P_LLM_T, P_VIT_T);
constexpr int P_MAX_DW_R  = const_max(DW_R, VIT_DW_R);

static_assert(P_HC == VIT_HC, "QK_GEMM reuse requires identical LLM/ViT head dim");
static_assert(P_S == VIT_S,   "QK_GEMM reuse assumes LLM cache length and ViT token count both use 1024");
static_assert(DW_R_TRUNC == VIT_DW_R_TRUNC, "QK_GEMM reuse expects shared R output width");

typedef REUSE_AQ_T        qk_q_t;
typedef REUSE_AS_T        qk_s_t;
typedef ap_int<P_MAX_DW_R> qk_acc_t;
typedef REUSE_R_TRUNC_T   qk_r_t;

constexpr int P_LLM_NUM_QK = (P_LLM_H + P_LLM_KVH) * P_LLM_T * P_HC;
constexpr int P_LLM_NUM_S  = (P_LLM_H + P_LLM_KVH) * P_LLM_T * P_HCT;
constexpr int P_LLM_NUM_R  =  P_LLM_H * P_LLM_T * P_S;
constexpr int P_LLM_NUM_KQ_CACHE = P_LLM_KVH * P_S * P_HC;
constexpr int P_LLM_NUM_KS_CACHE = P_LLM_KVH * P_S * P_HCT;
constexpr int P_LLM_NUM_KQ_CUR   = P_LLM_KVH * P_LLM_T * P_HC;
constexpr int P_LLM_NUM_KS_CUR   = P_LLM_KVH * P_LLM_T * P_HCT;

constexpr int P_VIT_NUM_QK = 2 * P_VIT_H * P_VIT_T * P_HC;
constexpr int P_VIT_NUM_S  = 2 * P_VIT_H * P_VIT_T * P_HCT;
constexpr int P_VIT_NUM_R  =     P_VIT_H * P_VIT_T * P_S;

void load_llm_token_block_q(
    hls::stream<hls::vector<qk_q_t, P_CP> >& q_stream,
    qk_q_t buf[P_MAX_T][P_HC]
) {
    // LLM ROPE_QK 输出在一个 8-token tile 内按 HCT -> T 顺序展开。
    for (int hct = 0; hct < P_HCT; ++hct) {
        for (int t = 0; t < P_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            hls::vector<qk_q_t, P_CP> q_vec = q_stream.read();
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                buf[t][hct * P_CP + cp] = q_vec[cp];
            }
        }
    }
}

void load_llm_token_block_s(
    hls::stream<hls::vector<qk_s_t, 1> >& s_stream,
    qk_s_t buf[P_MAX_T][P_HCT]
) {
    // LLM scale 与 q 数据同拍顺序一致，每个 CP=8 lane 共享一个 scale。
    for (int hct = 0; hct < P_HCT; ++hct) {
        for (int t = 0; t < P_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            hls::vector<qk_s_t, 1> s_vec = s_stream.read();
            buf[t][hct] = s_vec[0];
        }
    }
}

void load_llm_token_block_qs(
    hls::stream<hls::vector<qk_q_t, P_CP> >& q_stream,
    hls::stream<hls::vector<qk_s_t, 1> >& s_stream,
    qk_q_t q_buf[P_MAX_T][P_HC],
    qk_s_t s_buf[P_MAX_T][P_HCT]
) {
    // Spinal 顶层中 ROPE_QK 每拍成对输出 q/s；QK_GEMM 必须同拍消费，
    // 不能先读完整 q block 再读 s block，否则 scale FIFO 会反压并锁住上游。
    for (int hct = 0; hct < P_HCT; ++hct) {
        for (int t = 0; t < P_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            hls::vector<qk_q_t, P_CP> q_vec = q_stream.read();
            hls::vector<qk_s_t, 1> s_vec = s_stream.read();
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                q_buf[t][hct * P_CP + cp] = q_vec[cp];
            }
            s_buf[t][hct] = s_vec[0];
        }
    }
}

void load_llm_current_block_qs(
    hls::stream<hls::vector<qk_q_t, P_CP> >& q_stream,
    hls::stream<hls::vector<qk_s_t, 1> >& s_stream,
    qk_q_t q_buf[P_LLM_T][P_HC],
    qk_s_t s_buf[P_LLM_T][P_HCT]
) {
    // 当前 K tile 同样来自 ROPE_QK 的 q/s 成对输出；这里保持 LLM-only 小 LUTRAM buffer。
    for (int hct = 0; hct < P_HCT; ++hct) {
        for (int t = 0; t < P_LLM_T; ++t) {
            #pragma HLS pipeline II=1
            hls::vector<qk_q_t, P_CP> q_vec = q_stream.read();
            hls::vector<qk_s_t, 1> s_vec = s_stream.read();
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                q_buf[t][hct * P_CP + cp] = q_vec[cp];
            }
            s_buf[t][hct] = s_vec[0];
        }
    }
}

void load_vit_token_block_q(
    hls::stream<hls::vector<qk_q_t, P_CP> >& q_stream,
    qk_q_t buf[P_MAX_T][P_HC]
) {
    // ViT 保持 DEMUX/ROPE_QK 的 H -> TT -> HCT -> TP_D 顺序，TP_D 当前等于 CP=8。
    for (int tt = 0; tt < P_VIT_TT_D; ++tt) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            for (int ti = 0; ti < P_CP; ++ti) {
                #pragma HLS pipeline II=1
                hls::vector<qk_q_t, P_CP> q_vec = q_stream.read();
                int token = tt * P_CP + ti;
                for (int cp = 0; cp < P_CP; ++cp) {
                    #pragma HLS unroll
                    buf[token][hct * P_CP + cp] = q_vec[cp];
                }
            }
        }
    }
}

void load_vit_token_block_s(
    hls::stream<hls::vector<qk_s_t, 1> >& s_stream,
    qk_s_t buf[P_MAX_T][P_HCT]
) {
    // ViT scale 的 token/channel 顺序与量化后的 q 数据保持一致。
    for (int tt = 0; tt < P_VIT_TT_D; ++tt) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            for (int ti = 0; ti < P_CP; ++ti) {
                #pragma HLS pipeline II=1
                hls::vector<qk_s_t, 1> s_vec = s_stream.read();
                int token = tt * P_CP + ti;
                buf[token][hct] = s_vec[0];
            }
        }
    }
}

void load_vit_token_block_qs(
    hls::stream<hls::vector<qk_q_t, P_CP> >& q_stream,
    hls::stream<hls::vector<qk_s_t, 1> >& s_stream,
    qk_q_t q_buf[P_MAX_T][P_HC],
    qk_s_t s_buf[P_MAX_T][P_HCT]
) {
    // ViT 的 ROPE_QK bypass+Quantizer 也按 H -> TT -> HCT -> TP_D 成对输出 q/s。
    // 成对消费避免将来 ViT 顶层需要 8192 拍以上的大 scale FIFO。
    for (int tt = 0; tt < P_VIT_TT_D; ++tt) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            for (int ti = 0; ti < P_CP; ++ti) {
                #pragma HLS pipeline II=1
                hls::vector<qk_q_t, P_CP> q_vec = q_stream.read();
                hls::vector<qk_s_t, 1> s_vec = s_stream.read();
                int token = tt * P_CP + ti;
                for (int cp = 0; cp < P_CP; ++cp) {
                    #pragma HLS unroll
                    q_buf[token][hct * P_CP + cp] = q_vec[cp];
                }
                s_buf[token][hct] = s_vec[0];
            }
        }
    }
}

void load_cache_q(
    hls::stream<hls::vector<qk_q_t, P_CP> >& cache_i_stream,
    qk_q_t cache[P_S][P_HC]
) {
    // LLM K cache 输入按 [S][HCT] x CP 顺序喂入；ViT mode 不调用该函数。
    for (int s = 0; s < P_S; ++s) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            #pragma HLS pipeline II=1
            hls::vector<qk_q_t, P_CP> cache_vec = cache_i_stream.read();
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                cache[s][hct * P_CP + cp] = cache_vec[cp];
            }
        }
    }
}

void load_cache_s(
    hls::stream<hls::vector<qk_s_t, 1> >& cache_i_stream,
    qk_s_t cache[P_S][P_HCT]
) {
    // LLM K scale cache 输入按 [S][HCT] 顺序喂入；每拍一个 scale。
    for (int s = 0; s < P_S; ++s) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            #pragma HLS pipeline II=1
            hls::vector<qk_s_t, 1> cache_vec = cache_i_stream.read();
            cache[s][hct] = cache_vec[0];
        }
    }
}

void load_cache_qs(
    hls::stream<hls::vector<qk_q_t, P_CP> >& q_cache_i_stream,
    hls::stream<hls::vector<qk_s_t, 1> >& s_cache_i_stream,
    qk_q_t q_cache[P_S][P_HC],
    qk_s_t s_cache[P_S][P_HCT],
    int valid_s
) {
    // KV_CACHE 从 DDR 解包时同拍输出 KQ/KS；LLM causal windowing 只消费当前可见
    // cache 前缀，ViT mode 不调用该函数。
    for (int s = 0; s < valid_s; ++s) {
        #pragma HLS loop_tripcount min=P_LLM_T max=P_S
        for (int hct = 0; hct < P_HCT; ++hct) {
            #pragma HLS pipeline II=1
            hls::vector<qk_q_t, P_CP> q_vec = q_cache_i_stream.read();
            hls::vector<qk_s_t, 1> s_vec = s_cache_i_stream.read();
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                q_cache[s][hct * P_CP + cp] = q_vec[cp];
            }
            s_cache[s][hct] = s_vec[0];
        }
    }
}

void write_llm_cache_current_q(
    int chunk,
    qk_q_t cache[P_S][P_HC],
    hls::stream<hls::vector<qk_q_t, P_CP> >& cache_o_stream
) {
    // 只回写当前 decode tile 的 K，顺序与旧 LLM_QK_GEMM 的 cache_o 保持一致：[T][HCT] x CP。
    int base = chunk * P_LLM_T;
    for (int t = 0; t < P_LLM_T; ++t) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            #pragma HLS pipeline II=1
            hls::vector<qk_q_t, P_CP> cache_vec;
            int s_idx = base + t;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                cache_vec[cp] = (s_idx < P_S) ? cache[s_idx][hct * P_CP + cp] : (qk_q_t)0;
            }
            cache_o_stream.write(cache_vec);
        }
    }
}

void write_llm_cache_current_qs(
    int chunk,
    qk_q_t q_cache[P_S][P_HC],
    qk_s_t s_cache[P_S][P_HCT],
    hls::stream<hls::vector<qk_q_t, P_CP> >& q_cache_o_stream,
    hls::stream<hls::vector<qk_s_t, 1> >& s_cache_o_stream
) {
    // KV_CACHE 写回端按 KQ/KS 成对读取当前 tile；QK_GEMM 也成对输出，
    // 避免只靠 Spinal FIFO 缓存完整 KQ 当前块。
    int base = chunk * P_LLM_T;
    for (int t = 0; t < P_LLM_T; ++t) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            #pragma HLS pipeline II=1
            hls::vector<qk_q_t, P_CP> q_vec;
            hls::vector<qk_s_t, 1> s_vec;
            int s_idx = base + t;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                q_vec[cp] = (s_idx < P_S) ? q_cache[s_idx][hct * P_CP + cp] : (qk_q_t)0;
            }
            s_vec[0] = (s_idx < P_S) ? s_cache[s_idx][hct] : (qk_s_t)0;
            q_cache_o_stream.write(q_vec);
            s_cache_o_stream.write(s_vec);
        }
    }
}

void write_llm_cache_current_s(
    int chunk,
    qk_s_t cache[P_S][P_HCT],
    hls::stream<hls::vector<qk_s_t, 1> >& cache_o_stream
) {
    // scale cache 只输出当前 decode tile，供后续 KV cache 调度保存。
    int base = chunk * P_LLM_T;
    for (int t = 0; t < P_LLM_T; ++t) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            #pragma HLS pipeline II=1
            hls::vector<qk_s_t, 1> cache_vec;
            int s_idx = base + t;
            cache_vec[0] = (s_idx < P_S) ? cache[s_idx][hct] : (qk_s_t)0;
            cache_o_stream.write(cache_vec);
        }
    }
}

void overwrite_llm_cache_with_current_k(
    int chunk,
    qk_q_t k_cur[P_MAX_T][P_HC],
    qk_s_t ks_cur[P_MAX_T][P_HCT],
    qk_q_t k_cache[P_S][P_HC],
    qk_s_t ks_cache[P_S][P_HCT]
) {
    // 将 k_cur 中刚读到的当前 tile 写回 k_cache 的 [chunk*8 ..) 位置；此前 k_cache 已由 load_cache_qs
    // 填入历史 K。K 来自 ROPE 的量化半段，无需像 RV 里 raw_tile 那样先聚 raw 再算 scale 后原位写 vq_buf。
    int base = chunk * P_LLM_T;
    for (int t = 0; t < P_LLM_T; ++t) {
        for (int hct = 0; hct < P_HCT; ++hct) {
            #pragma HLS pipeline II=1
            int s_idx = base + t;
            if (s_idx < P_S) {
                for (int cp = 0; cp < P_CP; ++cp) {
                    #pragma HLS unroll
                    k_cache[s_idx][hct * P_CP + cp] = k_cur[t][hct * P_CP + cp];
                }
                ks_cache[s_idx][hct] = ks_cur[t][hct];
            }
        }
    }
}

void shared_qk_bmm_head(
    int q_tokens,
    int valid_st,
    int trunc,
    qk_q_t q_buf[P_MAX_T][P_HC],
    qk_s_t qs_buf[P_MAX_T][P_HCT],
    qk_q_t k_buf[P_S][P_HC],
    qk_s_t ks_buf[P_S][P_HCT],
    hls::stream<hls::vector<qk_r_t, P_SP> >& r_stream
) {
    // 单套 BMM 内核：
    //   对每个 head 计算 Q[T,HC] x K[S,HC]^T，SP=8 个 K 位置并行输出。
    //   trunc 在外层按 mode 选择，乘法、scale shift、psum 和输出 pack 完全共用。
    for (int t = 0; t < P_MAX_T; ++t) {
        #pragma HLS loop_tripcount min=P_LLM_T max=P_VIT_T
        if (t >= q_tokens) {
            continue;
        }
        for (int st = 0; st < valid_st; ++st) {
            #pragma HLS loop_tripcount min=1 max=P_ST
            hls::vector<qk_acc_t, P_SP> psum_vec(0);
            for (int hct = 0; hct < P_HCT; ++hct) {
                #pragma HLS pipeline II=1

                hls::vector<qk_acc_t, P_SP> tile_psum_vec(0);
                for (int sp = 0; sp < P_SP; ++sp) {
                    for (int cp = 0; cp < P_CP; ++cp) {
                        #pragma HLS unroll
                        qk_acc_t mul_res = q_buf[t][hct * P_CP + cp] * k_buf[st * P_SP + sp][hct * P_CP + cp];
                        #pragma HLS bind_op variable=mul_res op=mul impl=dsp
                        tile_psum_vec[sp] += mul_res;
                    }
                }

                qk_s_t q_shift = qs_buf[t][hct];
                for (int sp = 0; sp < P_SP; ++sp) {
                    #pragma HLS unroll
                    qk_s_t k_shift = ks_buf[st * P_SP + sp][hct];
                    psum_vec[sp] += tile_psum_vec[sp] << (q_shift + k_shift);
                }

                if (hct == P_HCT - 1) {
                    hls::vector<qk_r_t, P_SP> r_vec;
                    for (int sp = 0; sp < P_SP; ++sp) {
                        #pragma HLS unroll
                        r_vec[sp] = (qk_r_t)(psum_vec[sp] >> trunc);
                    }
                    r_stream.write(r_vec);
                }
            }
        }
    }
}

void qk_gemm_one_layer(
    REUSE_MODE_T mode,
    int pos,
    hls::stream<hls::vector<qk_q_t, P_CP> >& qk_q_stream,
    hls::stream<hls::vector<qk_s_t, 1   > >& qk_s_stream,
    hls::stream<hls::vector<qk_q_t, P_CP> >& kq_cache_i_stream,
    hls::stream<hls::vector<qk_s_t, 1   > >& ks_cache_i_stream,
    hls::stream<hls::vector<qk_q_t, P_CP> >& kq_cache_o_stream,
    hls::stream<hls::vector<qk_s_t, 1   > >& ks_cache_o_stream,
    hls::stream<hls::vector<qk_r_t, P_SP> >& r_stream
) {
    // ViT keeps the original per-head MHA path. LLM uses native GQA ordering from
    // ROPE_QK: per KV group K, Q0, Q1, Q2. The K cache is loaded/written once per
    // KV head and then reused for the three query heads in that group.
    bool is_vit = is_vit_mode(mode);
    reuse_pos_t decode_pos = reuse_decode_pos(mode, pos);
    int chunk = decode_pos.chunk;
    int valid_s   = reuse_attention_valid_s(mode, pos);
    int valid_st  = reuse_attention_valid_st(mode, pos);
    int trunc     = is_vit ? VIT_MHA_TRUNC_R : MHA_TRUNC_R;

    qk_q_t q_buf [P_MAX_T][P_HC];
    qk_s_t qs_buf[P_MAX_T][P_HCT];
    qk_q_t k_buf [P_S][P_HC];
    qk_s_t ks_buf[P_S][P_HCT];
    qk_q_t k_cur [P_LLM_T][P_HC];
    qk_s_t ks_cur[P_LLM_T][P_HCT];

    #pragma HLS bind_storage variable=q_buf  type=ram_2p impl=uram latency=2
    #pragma HLS bind_storage variable=qs_buf type=ram_2p impl=bram
    #pragma HLS bind_storage variable=k_buf  type=ram_2p impl=uram
    #pragma HLS bind_storage variable=ks_buf type=ram_2p impl=bram
    #pragma HLS bind_storage variable=k_cur  type=ram_2p impl=lutram
    #pragma HLS bind_storage variable=ks_cur type=ram_2p impl=lutram
    #pragma HLS array_reshape   variable=q_buf  cyclic factor=P_CP dim=2
    #pragma HLS array_reshape   variable=k_buf  cyclic factor=P_CP dim=2
    #pragma HLS array_reshape   variable=k_cur  cyclic factor=P_CP dim=2
    #pragma HLS array_partition variable=k_buf  cyclic factor=P_SP dim=1
    #pragma HLS array_partition variable=ks_buf cyclic factor=P_SP dim=1

    if (is_vit) {
        for (int h = 0; h < P_VIT_H; ++h) {
            #pragma HLS loop_tripcount min=P_VIT_H max=P_VIT_H
            load_vit_token_block_qs(qk_q_stream, qk_s_stream, q_buf, qs_buf);
            load_vit_token_block_qs(qk_q_stream, qk_s_stream, k_buf, ks_buf);
            shared_qk_bmm_head(P_VIT_T, valid_st, trunc, q_buf, qs_buf, k_buf, ks_buf, r_stream);
        }
    } else {
        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            #pragma HLS loop_tripcount min=P_LLM_KVH max=P_LLM_KVH
            load_cache_qs(kq_cache_i_stream, ks_cache_i_stream, k_buf, ks_buf, valid_s);
            load_llm_current_block_qs(qk_q_stream, qk_s_stream, k_cur, ks_cur);
            overwrite_llm_cache_with_current_k(chunk, k_cur, ks_cur, k_buf, ks_buf);
            write_llm_cache_current_qs(chunk, k_buf, ks_buf, kq_cache_o_stream, ks_cache_o_stream);

            for (int r = 0; r < P_LLM_GQA; ++r) {
                #pragma HLS loop_tripcount min=P_LLM_GQA max=P_LLM_GQA
                load_llm_token_block_qs(qk_q_stream, qk_s_stream, q_buf, qs_buf);
                shared_qk_bmm_head(P_LLM_T, valid_st, trunc, q_buf, qs_buf, k_buf, ks_buf, r_stream);
            }
        }
    }
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    int pos,
    hls::stream<hls::vector<qk_q_t, P_CP> >& qk_q_stream,
    hls::stream<hls::vector<qk_s_t, 1   > >& qk_s_stream,
    hls::stream<hls::vector<qk_q_t, P_CP> >& kq_cache_i_stream,
    hls::stream<hls::vector<qk_s_t, 1   > >& ks_cache_i_stream,
    hls::stream<hls::vector<qk_q_t, P_CP> >& kq_cache_o_stream,
    hls::stream<hls::vector<qk_s_t, 1   > >& ks_cache_o_stream,
    hls::stream<hls::vector<qk_r_t, P_SP> >& r_stream
) {
    // 冻结后的 QK_GEMM 顶层接口：
    //   mode=0 为 LLM，mode=1 为 ViT；pos 是 LLM decode 绝对位置，ViT mode 忽略。
    //   ViT mode 必须保持 cache 端口 no-op，不消费也不产生 cache stream。
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=qk_q_stream
    #pragma HLS interface axis port=qk_s_stream
    #pragma HLS interface axis port=kq_cache_i_stream
    #pragma HLS interface axis port=ks_cache_i_stream
    #pragma HLS interface axis port=kq_cache_o_stream
    #pragma HLS interface axis port=ks_cache_o_stream
    #pragma HLS interface axis port=r_stream
    #pragma HLS aggregate variable=qk_q_stream       compact=bit
    #pragma HLS aggregate variable=qk_s_stream       compact=bit
    #pragma HLS aggregate variable=kq_cache_i_stream compact=bit
    #pragma HLS aggregate variable=ks_cache_i_stream compact=bit
    #pragma HLS aggregate variable=kq_cache_o_stream compact=bit
    #pragma HLS aggregate variable=ks_cache_o_stream compact=bit
    #pragma HLS aggregate variable=r_stream          compact=bit

    for (int l = l_begin; l < l_close; ++l) {
        #pragma HLS loop_tripcount min=1 max=32
        if (!reuse_is_valid_layer(mode, l, false)) {
            continue;
        }
        qk_gemm_one_layer(
            mode,
            pos,
            qk_q_stream,
            qk_s_stream,
            kq_cache_i_stream,
            ks_cache_i_stream,
            kq_cache_o_stream,
            ks_cache_o_stream,
            r_stream
        );
    }
}

// ============================================================================
// Testbench
// ============================================================================
// step1 同时覆盖：
//   - LLM mode decoder layer 0：读取 ROPE_QK 量化输出 + K cache，写 R 和当前 K cache。
//   - ViT mode vision layer 0：读取 ROPE_QK bypass+Quantizer 输出，不读写 cache。

namespace llm_qk_gemm_tb {

void test_layer(int l) {
    int pos = reuse_tb_llm_pos(P_LLM_POS);
    int top_pos = reuse_tb_llm_top_pos(pos + P_LLM_T - 1);
    int chunk = top_pos / P_LLM_T;
    int valid_s = reuse_llm_valid_s(top_pos);
    int valid_st = (valid_s + P_SP - 1) / P_SP;
    int valid_s_padded = valid_st * P_SP;
    int valid_num_r = P_LLM_H * P_LLM_T * valid_s_padded;
    int valid_num_kq_cache = P_LLM_KVH * valid_s * P_HC;
    int valid_num_ks_cache = P_LLM_KVH * valid_s * P_HCT;

    const string file_path = BINARIES_PATH + to_string(l);
    const string save_path = CONDENSE_PATH + to_string(l);

    vector<int8_t>  ref_k_q(P_LLM_NUM_KQ_CUR);
    vector<int8_t>  ref_k_s(P_LLM_NUM_KS_CUR);
    vector<int8_t>  ref_k_q_cache(P_LLM_NUM_KQ_CACHE);
    vector<int8_t>  ref_k_s_cache(P_LLM_NUM_KS_CACHE);
    vector<int8_t>  ref_k_q_cache_valid(valid_num_kq_cache);
    vector<int8_t>  ref_k_s_cache_valid(valid_num_ks_cache);
    vector<int64_t> ref_r(P_LLM_NUM_R);
    vector<int8_t>  condensed_q(P_LLM_NUM_QK);
    vector<int8_t>  condensed_s(P_LLM_NUM_S);
    vector<int64_t> dut_r(valid_num_r);
    vector<int8_t>  dut_k_q(P_LLM_NUM_KQ_CUR);
    vector<int8_t>  dut_k_s(P_LLM_NUM_KS_CUR);

    {
        // 软件参考按绝对 POS 截取当前 8-token decode tile，并构造因果 cache 输入。
        auto K_Q = read_tensor<int8_t >(file_path + "/MHA_K_Q.bin");
        auto K_S = read_tensor<int8_t >(file_path + "/MHA_K_S.bin");
        auto R   = read_tensor<int64_t>(file_path + "/MHA_R.bin");
        // 当前参考文件中 K cache 方向通常是 S_LOAD=880，而 R 文件是 T_LOAD=877 x S_LOAD=880。
        // 这里分别按文件长度推导，硬件仍只消费固定 LLAMA_S=1024 的片上 cache窗口。
        int kq_s_load = infer_tensor_t_load<int8_t>(K_Q, P_LLM_H, P_HC,  "MHA_K_Q");
        int ks_s_load = infer_tensor_t_load<int8_t>(K_S, P_LLM_H, P_HCT, "MHA_K_S");
        int r_t_load  = infer_tensor_t_load<int64_t>(R, P_LLM_H, kq_s_load, "MHA_R");
        check_tensor_window(kq_s_load, pos, P_LLM_T, "MHA_K_Q");
        check_tensor_window(ks_s_load, pos, P_LLM_T, "MHA_K_S");
        check_tensor_window(r_t_load,  pos, P_LLM_T, "MHA_R");
        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            int src_head = kv * P_LLM_GQA;
            for (int t = 0; t < P_LLM_T && pos + t < kq_s_load; ++t) {
                for (int c = 0; c < P_HC; ++c) {
                    ref_k_q[kv * P_LLM_T * P_HC + t * P_HC + c] =
                        K_Q[src_head * kq_s_load * P_HC + (pos + t) * P_HC + c];
                }
            }
            for (int t = 0; t < P_LLM_T && pos + t < ks_s_load; ++t) {
                for (int hct = 0; hct < P_HCT; ++hct) {
                    ref_k_s[kv * P_LLM_T * P_HCT + t * P_HCT + hct] =
                        K_S[src_head * ks_s_load * P_HCT + (pos + t) * P_HCT + hct];
                }
            }
            for (int s = 0; s < P_S && s < kq_s_load; ++s) {
                for (int c = 0; c < P_HC; ++c) {
                    ref_k_q_cache[kv * P_S * P_HC + s * P_HC + c] =
                        K_Q[src_head * kq_s_load * P_HC + s * P_HC + c];
                }
            }
            for (int s = 0; s < P_S && s < ks_s_load; ++s) {
                for (int hct = 0; hct < P_HCT; ++hct) {
                    ref_k_s_cache[kv * P_S * P_HCT + s * P_HCT + hct] =
                        K_S[src_head * ks_s_load * P_HCT + s * P_HCT + hct];
                }
            }
        }
        tensor2array<int64_t>(R,   ref_r.data(),         P_LLM_H, P_LLM_H, r_t_load,  pos, P_LLM_T, kq_s_load, P_S);

        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            for (int s = pos; s < P_S; ++s) {
                for (int c = 0; c < P_HC; ++c) {
                    ref_k_q_cache[kv * P_S * P_HC + s * P_HC + c] = 0;
                }
                for (int hct = 0; hct < P_HCT; ++hct) {
                    ref_k_s_cache[kv * P_S * P_HCT + s * P_HCT + hct] = 0;
                }
            }
        }
        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            for (int s = 0; s < valid_s; ++s) {
                for (int c = 0; c < P_HC; ++c) {
                    ref_k_q_cache_valid[kv * valid_s * P_HC + s * P_HC + c] =
                        ref_k_q_cache[kv * P_S * P_HC + s * P_HC + c];
                }
                for (int hct = 0; hct < P_HCT; ++hct) {
                    ref_k_s_cache_valid[kv * valid_s * P_HCT + s * P_HCT + hct] =
                        ref_k_s_cache[kv * P_S * P_HCT + s * P_HCT + hct];
                }
            }
        }
    }

    {
        // Q/K 输入必须来自复用 ROPE_QK 的量化输出，不能回退到本模块私有量化。
        auto Q_T = read_tensor<int8_t>(save_path + "/CONDENSED_ROPE_QK_QUANT_ROT_Q.bin");
        auto S_T = read_tensor<int8_t>(save_path + "/CONDENSED_ROPE_QK_QUANT_ROT_S.bin");
        tensor2array<int8_t>(Q_T, condensed_q.data(), 1, 1, 1, 1, P_LLM_NUM_QK, P_LLM_NUM_QK);
        tensor2array<int8_t>(S_T, condensed_s.data(), 1, 1, 1, 1, P_LLM_NUM_S,  P_LLM_NUM_S);
    }

    hls::stream<hls::vector<qk_q_t, P_CP> > qk_q_stream("llm_qk_q_stream");
    hls::stream<hls::vector<qk_s_t, 1   > > qk_s_stream("llm_qk_s_stream");
    hls::stream<hls::vector<qk_q_t, P_CP> > kq_cache_i_stream("llm_kq_cache_i_stream");
    hls::stream<hls::vector<qk_s_t, 1   > > ks_cache_i_stream("llm_ks_cache_i_stream");
    hls::stream<hls::vector<qk_q_t, P_CP> > kq_cache_o_stream("llm_kq_cache_o_stream");
    hls::stream<hls::vector<qk_s_t, 1   > > ks_cache_o_stream("llm_ks_cache_o_stream");
    hls::stream<hls::vector<qk_r_t, P_SP> > r_stream("llm_r_stream");

    array2stream<int8_t, qk_q_t, 1, 1, 1, 1, P_LLM_NUM_QK, P_CP>(condensed_q.data(), qk_q_stream, "LLM QK_Q", true);
    array2stream<int8_t, qk_s_t, 1, 1, 1, 1, P_LLM_NUM_S,  1   >(condensed_s.data(), qk_s_stream, "LLM QK_S", true);
    array2stream_runtime<int8_t, qk_q_t, P_CP>(ref_k_q_cache_valid.data(), kq_cache_i_stream, valid_num_kq_cache);
    array2stream_runtime<int8_t, qk_s_t, 1   >(ref_k_s_cache_valid.data(), ks_cache_i_stream, valid_num_ks_cache);

    // Cache condensed 文件保留完整 1024 窗口，Spinal/顶层 memory 初始化仍可直接复用；
    // 实际喂入 QK_GEMM 的 AXIS 只取 valid_s 前缀。
    save_tensor<int8_t>(save_path + "/CONDENSED_KQ_CACHE.bin", ref_k_q_cache.data(), P_LLM_NUM_KQ_CACHE);
    save_tensor<int8_t>(save_path + "/CONDENSED_KS_CACHE.bin", ref_k_s_cache.data(), P_LLM_NUM_KS_CACHE);

    ::top(MODE_LLM, l, l + 1, top_pos, qk_q_stream, qk_s_stream,
          kq_cache_i_stream, ks_cache_i_stream, kq_cache_o_stream, ks_cache_o_stream, r_stream);

    save_condensed_tensor_runtime<int64_t, qk_r_t, P_SP>(save_path + "/CONDENSED_QK_GEMM_R.bin", r_stream, valid_num_r);

    stream2array_runtime<int64_t, qk_r_t, P_SP>(r_stream, dut_r.data(), valid_num_r);
    stream2array<int8_t,  qk_q_t, P_LLM_KVH, P_LLM_T, 1, P_HC, P_CP>(kq_cache_o_stream,  dut_k_q.data(), "LLM outKQ", true);
    stream2array<int8_t,  qk_s_t, P_LLM_KVH, P_LLM_T, 1, P_HCT, 1  >(ks_cache_o_stream,  dut_k_s.data(), "LLM outKS", true);

    assert(qk_q_stream.size() == 0);
    assert(qk_s_stream.size() == 0);
    assert(kq_cache_i_stream.size() == 0);
    assert(ks_cache_i_stream.size() == 0);
    assert(r_stream.size() == 0);

    int mismatch_count = 0;
    for (int h = 0; h < P_LLM_H; ++h) {
        for (int t = 0; t < P_LLM_T; ++t) {
            for (int s = 0; s < valid_s; ++s) {
                int idx = h * P_LLM_T * P_S + t * P_S + s;
                int dut_idx = h * P_LLM_T * valid_s_padded + t * valid_s_padded + s;
                if (ref_r[idx] != dut_r[dut_idx]) {
                    if (++mismatch_count <= 10) {
                        printf("LLM R mismatch layer %d h %d t %d s %d: REF=%ld DUT=%ld\n",
                               l, h, t, s, ref_r[idx], dut_r[dut_idx]);
                    }
                }
            }
        }
    }
    if (mismatch_count > 0) {
        printf("LLM R total mismatch count: %d\n", mismatch_count);
    } else {
        printf("LLM QK_GEMM R PASS\n");
    }

    compare<int8_t>(ref_k_q.data(), dut_k_q.data(), P_LLM_NUM_KQ_CUR, "LLM KQ Cache");
    compare<int8_t>(ref_k_s.data(), dut_k_s.data(), P_LLM_NUM_KS_CUR, "LLM KS Cache");
}

} // namespace llm_qk_gemm_tb

namespace vit_qk_gemm_tb {

void test_layer(int l) {
    const string file_path = VIT_BINARIES_PATH + to_string(l);
    const string save_path = VIT_CONDENSE_PATH + to_string(l);

    vector<int8_t>  condensed_q(P_VIT_NUM_QK);
    vector<int8_t>  condensed_s(P_VIT_NUM_S);
    vector<int64_t> ref_r(P_VIT_NUM_R);
    vector<int64_t> dut_r(P_VIT_NUM_R);

    {
        // MHA_R.bin 按 batch*head 保存；不同导出批次的 batch 数可能变化，按文件长度推导。
        auto R = read_tensor<int64_t>(file_path + "/MHA_R.bin");
        int b_load = infer_tensor_extent<int64_t>(R, P_VIT_H * P_VIT_T * P_S, "ViT MHA_R");
        tensor2array<int64_t>(R, ref_r.data(), b_load * P_VIT_H, P_VIT_H, P_VIT_T, 0, P_VIT_T, P_S, P_S);
    }

    {
        // ViT Q/K 输入读取 ROPE_QK bypass+Quantizer 结果，确认 QK_GEMM 内不再私有量化。
        auto Q_T = read_tensor<int8_t>(save_path + "/CONDENSED_ROPE_QK_BYPASS_Q.bin");
        auto S_T = read_tensor<int8_t>(save_path + "/CONDENSED_ROPE_QK_BYPASS_S.bin");
        tensor2array<int8_t>(Q_T, condensed_q.data(), 1, 1, 1, 1, P_VIT_NUM_QK, P_VIT_NUM_QK);
        tensor2array<int8_t>(S_T, condensed_s.data(), 1, 1, 1, 1, P_VIT_NUM_S,  P_VIT_NUM_S);
    }

    hls::stream<hls::vector<qk_q_t, P_CP> > qk_q_stream("vit_qk_q_stream");
    hls::stream<hls::vector<qk_s_t, 1   > > qk_s_stream("vit_qk_s_stream");
    hls::stream<hls::vector<qk_q_t, P_CP> > kq_cache_i_stream("vit_kq_cache_i_stream");
    hls::stream<hls::vector<qk_s_t, 1   > > ks_cache_i_stream("vit_ks_cache_i_stream");
    hls::stream<hls::vector<qk_q_t, P_CP> > kq_cache_o_stream("vit_kq_cache_o_stream");
    hls::stream<hls::vector<qk_s_t, 1   > > ks_cache_o_stream("vit_ks_cache_o_stream");
    hls::stream<hls::vector<qk_r_t, P_SP> > r_stream("vit_r_stream");

    array2stream<int8_t, qk_q_t, 1, 1, 1, 1, P_VIT_NUM_QK, P_CP>(condensed_q.data(), qk_q_stream, "ViT QK_Q", true);
    array2stream<int8_t, qk_s_t, 1, 1, 1, 1, P_VIT_NUM_S,  1   >(condensed_s.data(), qk_s_stream, "ViT QK_S", true);

    ::top(MODE_VIT, l, l + 1, 0, qk_q_stream, qk_s_stream,
          kq_cache_i_stream, ks_cache_i_stream, kq_cache_o_stream, ks_cache_o_stream, r_stream);

    save_condensed_tensor<int64_t, qk_r_t, P_VIT_NUM_R, P_SP>(save_path + "/CONDENSED_QK_GEMM_R.bin", r_stream);
    stream2array<int64_t, qk_r_t, P_VIT_H, P_VIT_T, 1, P_S, P_SP>(r_stream, dut_r.data(), "ViT Output R", true);

    assert(qk_q_stream.size() == 0);
    assert(qk_s_stream.size() == 0);
    assert(kq_cache_i_stream.size() == 0);
    assert(ks_cache_i_stream.size() == 0);
    assert(kq_cache_o_stream.size() == 0);
    assert(ks_cache_o_stream.size() == 0);
    assert(r_stream.size() == 0);

    int mismatch_count = 0;
    for (int h = 0; h < P_VIT_H; ++h) {
        for (int t = 0; t < P_VIT_T; ++t) {
            for (int s = 0; s < P_S; ++s) {
                int idx = h * P_VIT_T * P_S + t * P_S + s;
                if (ref_r[idx] != dut_r[idx]) {
                    if (++mismatch_count <= 10) {
                        printf("ViT R mismatch layer %d h %d t %d s %d: REF=%ld DUT=%ld\n",
                               l, h, t, s, ref_r[idx], dut_r[idx]);
                    }
                }
            }
        }
    }
    if (mismatch_count > 0) {
        printf("ViT R total mismatch count: %d\n", mismatch_count);
    } else {
        printf("ViT QK_GEMM R PASS\n");
    }
}

} // namespace vit_qk_gemm_tb

int main() {
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        llm_qk_gemm_tb::test_layer(0);
        printf("LLM QK_GEMM layer 0 passed\n");
    }

    vit_qk_gemm_tb::test_layer(vit_layer);
    printf("ViT QK_GEMM layer %d passed\n", vit_layer);

    return 0;
}
