#include "../src/reuse_common.h"
#include "../src/utils.h"

// ============================================================================
// LLM/ViT 复用 GEMM_DEMUX
// ============================================================================
// 功能：
//   1. 把共享 PERMUTE/GEMM 输出按下游算子拆分为 QK、V、MLP1、O/MLP2、CLS index 五类流。
//   2. 顶层通过 REUSE_MODE_T mode 选择 LLM decode/CLS 或 ViT encoder；mode 在一次 top 调用内保持稳定。
//   3. LLM mode 保持原 QK/V/UG/OD 语义；CLS/lm_head 在 DEMUX 内做 argmax，只输出 token index。
//      ViT mode 保持 QK/V/FC1/OFC2 文件语义且不写 cls_stream。
//   4. ViT QKV 顺序保持旧链路约定：[H][Q/K/V][TT][HCT][TP] x CP；其中每个 Q/K/V 子流内部是 H -> TT -> HCT -> TP x CP。
//
// 资源意图：
//   - DEMUX 没有乘法 datapath，复用重点是避免为 LLM/ViT 放两套路由逻辑。
//   - ViT bias 不再内置为 12 层常量 ROM；外部调度按 GEMM 输出顺序提供 bias_stream，
//     与 PERMUTE 权重流类似，后续由 M_AXI/PYNQ 负责按 mode/layer/tile 喂入。

constexpr int P_CP = REUSE_CP;

typedef REUSE_GEMM_TRUNC_T gemm_t;
typedef REUSE_QKV_TRUNC_T  qkv_out_t;
typedef REUSE_MLP1_TRUNC_T mlp1_out_t;
typedef REUSE_MLP2_TRUNC_T mlp2_out_t;
typedef REUSE_CLS_INDEX_T  cls_index_t;
typedef REUSE_DEMUX_BIAS_T bias_t;

static_assert(LLAMA_TRUNC_BASE == VIT_TRUNC_BASE, "DEMUX expects shared GEMM trunc base");
constexpr int P_TRUNC_BASE = LLAMA_TRUNC_BASE;

// LLM 参数：
//   普通 decoder 层消费 Q/K/V/O/UG/D 共 7 段 GEMM 输出。
//   l==LLAMA_L 时只消费 CLS/lm_head 输出，并在模块内做 argmax 后写 cls_stream。
constexpr int P_LLM_T       = REUSE_LLM_TILE_T;
constexpr int P_LLM_POS     = 96;
constexpr int P_LLM_TOP_POS = P_LLM_POS + P_LLM_T - 1;
constexpr int P_LLM_H       = LLAMA_H;
constexpr int P_LLM_KVH     = LLAMA_KVH;
constexpr int P_LLM_GQA     = P_LLM_H / P_LLM_KVH;
constexpr int P_LLM_C       = LLAMA_C;
constexpr int P_LLM_HC      = LLAMA_HC;
constexpr int P_LLM_CM      = LLAMA_CM;
constexpr int P_LLM_VOCAB   = LLAMA_VOCAB;
constexpr int P_LLM_HCT     = P_LLM_HC / P_CP;
constexpr int P_LLM_CT      = P_LLM_C  / P_CP;
constexpr int P_LLM_CMT     = P_LLM_CM / P_CP;
constexpr int P_LLM_T_Q     = (P_LLM_T * P_LLM_C) / P_CP;
constexpr int P_LLM_T_KV    = (P_LLM_T * P_LLM_KVH * P_LLM_HC) / P_CP;
constexpr int P_LLM_T_QKV   = P_LLM_T_KV + P_LLM_T_Q + P_LLM_T_KV;
constexpr int P_LLM_T_QKVO  = (P_LLM_T * P_LLM_C)     / P_CP;
constexpr int P_LLM_T_UG    = (P_LLM_T * P_LLM_CM)    / P_CP;
constexpr int P_LLM_T_D     = (P_LLM_T * P_LLM_C)     / P_CP;
constexpr int P_LLM_T_CLS   = (P_LLM_T * P_LLM_VOCAB) / P_CP;
constexpr int P_LLM_NUM_Y   = P_LLM_T * (P_LLM_C + 2 * P_LLM_KVH * P_LLM_HC) + P_LLM_T * P_LLM_C +
                              2 * P_LLM_T * P_LLM_CM + P_LLM_T * P_LLM_C;
constexpr int P_LLM_NUM_CLS = P_LLM_T * P_LLM_VOCAB;
constexpr int P_LLM_T_DECODER = P_LLM_T_QKV + P_LLM_T_QKVO + P_LLM_T_UG * 2 + P_LLM_T_D;

// ViT 参数：
//   encoder 层消费 Q/K/V/O/FC1/FC2 共 6 段 GEMM 输出。
//   ViT 没有 CLS，MLP1 表示 FC1，O/MLP2 流表示 O 和 FC2 拼接。
constexpr int P_VIT_T       = VIT_S;
// ViT 新导出按 batch*patch 保存，当前 vision_0/new_txt 与其他层同批为 13*1024。
constexpr int P_VIT_T_LOAD  = 13312;
constexpr int P_VIT_POS     = 0;
constexpr int P_VIT_H       = VIT_H;
constexpr int P_VIT_C       = VIT_C;
constexpr int P_VIT_HC      = VIT_HC;
constexpr int P_VIT_CM      = VIT_MLP_DIM;
constexpr int P_VIT_TP      = REUSE_GEMM_TP;
constexpr int P_VIT_TT      = P_VIT_T / P_VIT_TP;
constexpr int P_VIT_HCT     = P_VIT_HC / P_CP;
constexpr int P_VIT_CT      = P_VIT_C  / P_CP;
constexpr int P_VIT_CMT     = P_VIT_CM / P_CP;
constexpr int P_VIT_T_QKVO  = (P_VIT_T * P_VIT_C)  / P_CP;
constexpr int P_VIT_T_FC1   = (P_VIT_T * P_VIT_CM) / P_CP;
constexpr int P_VIT_T_FC2   = (P_VIT_T * P_VIT_C)  / P_CP;
constexpr int P_VIT_NUM_Y   = 3 * P_VIT_T * P_VIT_C + P_VIT_T * P_VIT_C +
                              P_VIT_T * P_VIT_CM + P_VIT_T * P_VIT_C;
constexpr int P_VIT_T_DECODER = P_VIT_T_QKVO * 4 + P_VIT_T_FC1 + P_VIT_T_FC2;

void demux_llm_decoder(
    int chunk_pos,
    hls::stream<hls::vector<gemm_t,     P_CP> >& gemm_stream,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& qk_stream,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& v_stream,
    hls::stream<hls::vector<mlp1_out_t, P_CP> >& mlp1_stream,
    hls::stream<hls::vector<mlp2_out_t, P_CP> >& od_fc2_stream
) {
    // LLM native-GQA K/V/Q route:
    //   PERMUTE emits per KV group: K, V, Q0, Q1, Q2. Sending V before the
    //   three Q heads breaks the QK -> SOFTMAX -> RV backpressure cycle while
    //   ROPE_QK/QK_GEMM still see their compact K, Q0, Q1, Q2 stream order.
    for (int kv = 0; kv < P_LLM_KVH; ++kv) {
        // K for this KV group.
        for (int hct = 0; hct < P_LLM_HCT; ++hct) {
            for (int t = 0; t < P_LLM_T; ++t) {
                #pragma HLS pipeline II=1
                hls::vector<gemm_t, P_CP> gemm_vec = gemm_stream.read();
                hls::vector<qkv_out_t, P_CP> k_vec;
                for (int cp = 0; cp < P_CP; ++cp) {
                    #pragma HLS unroll
                    GEMM_TRUNC_T local_gemm = (GEMM_TRUNC_T)gemm_vec[cp];
                    k_vec[cp] = (qkv_out_t)((QKV_TRUNC_T)(local_gemm >> (MHA_TRUNC_QK - P_TRUNC_BASE)));
                }
                qk_stream.write(k_vec);
            }
        }

        // V for this KV group. Future lanes are cleared before V quantization, matching LLM-01.
        for (int hct = 0; hct < P_LLM_HCT; ++hct) {
            for (int t = 0; t < P_LLM_T; ++t) {
                #pragma HLS pipeline II=1
                hls::vector<gemm_t, P_CP> gemm_vec = gemm_stream.read();
                hls::vector<qkv_out_t, P_CP> v_vec;
                for (int cp = 0; cp < P_CP; ++cp) {
                    #pragma HLS unroll
                    GEMM_TRUNC_T local_gemm = (GEMM_TRUNC_T)gemm_vec[cp];
                    QKV_TRUNC_T local_v = (QKV_TRUNC_T)(local_gemm >> (MHA_TRUNC_V - P_TRUNC_BASE));
                    if (t > chunk_pos) {
                        local_v = 0;
                    }
                    v_vec[cp] = (qkv_out_t)local_v;
                }
                v_stream.write(v_vec);
            }
        }

        // The three Q heads that share this KV head.
        for (int r = 0; r < P_LLM_GQA; ++r) {
            for (int hct = 0; hct < P_LLM_HCT; ++hct) {
                for (int t = 0; t < P_LLM_T; ++t) {
                    #pragma HLS pipeline II=1
                    hls::vector<gemm_t, P_CP> gemm_vec = gemm_stream.read();
                    hls::vector<qkv_out_t, P_CP> q_vec;
                    for (int cp = 0; cp < P_CP; ++cp) {
                        #pragma HLS unroll
                        GEMM_TRUNC_T local_gemm = (GEMM_TRUNC_T)gemm_vec[cp];
                        q_vec[cp] = (qkv_out_t)((QKV_TRUNC_T)(local_gemm >> (MHA_TRUNC_QK - P_TRUNC_BASE)));
                    }
                    qk_stream.write(q_vec);
                }
            }
        }
    }

    // LLM O/UG/D 路由：
    //   mlp1_stream 对应原 UG；od_fc2_stream 对应原 O 和 D 拼接。
    //   UG 先在 XUG_TRUNC_T 本地位宽内截断，再扩展到共享 MLP1 输出位宽。
    for (int iter = 0; iter < P_LLM_T_QKVO + 2 * P_LLM_T_UG + P_LLM_T_D; ++iter) {
        #pragma HLS pipeline II=1
        hls::vector<gemm_t, P_CP> gemm_vec = gemm_stream.read();
        if (iter < P_LLM_T_QKVO) {
            hls::vector<mlp2_out_t, P_CP> o_vec;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                GEMM_TRUNC_T local_gemm = (GEMM_TRUNC_T)gemm_vec[cp];
                XD_TRUNC_T local_o = (XD_TRUNC_T)(local_gemm >> (MHA_TRUNC_O - P_TRUNC_BASE));
                o_vec[cp] = (mlp2_out_t)local_o;
            }
            od_fc2_stream.write(o_vec);
        } else if (iter < P_LLM_T_QKVO + 2 * P_LLM_T_UG) {
            hls::vector<mlp1_out_t, P_CP> ug_vec;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                GEMM_TRUNC_T local_gemm = (GEMM_TRUNC_T)gemm_vec[cp];
                XUG_TRUNC_T local_ug = (XUG_TRUNC_T)(local_gemm >> (MLP_TRUNC_UG - P_TRUNC_BASE));
                ug_vec[cp] = (mlp1_out_t)local_ug;
            }
            mlp1_stream.write(ug_vec);
        } else {
            hls::vector<mlp2_out_t, P_CP> d_vec;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                GEMM_TRUNC_T local_gemm = (GEMM_TRUNC_T)gemm_vec[cp];
                XD_TRUNC_T local_d = (XD_TRUNC_T)(local_gemm >> (MLP_TRUNC_D - P_TRUNC_BASE));
                d_vec[cp] = (mlp2_out_t)local_d;
            }
            od_fc2_stream.write(d_vec);
        }
    }
}

void demux_llm_cls(
    hls::stream<hls::vector<gemm_t,    P_CP> >& gemm_stream,
    hls::stream<hls::vector<cls_index_t, P_LLM_T> >& cls_stream
) {
    // LLM CLS/lm_head 路径：
    //   l==LLAMA_L 消费 T x VOCAB 个 logits。为避免把完整 logits 写回 DDR，
    //   这里按 token 维护最大值和 vocab index，最终只输出一拍 8-lane token id。
    constexpr int P_CLS_MIN = -(1 << (CLS_TRUNC_T::width - 1));
    hls::vector<cls_index_t, P_LLM_T> cls_idx(0);
    hls::vector<CLS_TRUNC_T, P_LLM_T> cls_max(P_CLS_MIN);

    for (int vocabt = 0; vocabt < P_LLM_VOCAB / P_CP; ++vocabt) {
        for (int t = 0; t < P_LLM_T; ++t) {
            #pragma HLS loop_flatten off
            hls::vector<gemm_t, P_CP> gemm_vec = gemm_stream.read();
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS pipeline II=1
                GEMM_TRUNC_T local_gemm = (GEMM_TRUNC_T)gemm_vec[cp];
                CLS_TRUNC_T cls_val = (CLS_TRUNC_T)(local_gemm >> (CLS_TRUNC - P_TRUNC_BASE));
                if (cls_val > cls_max[t]) {
                    cls_max[t] = cls_val;
                    cls_idx[t] = (cls_index_t)(vocabt * P_CP + cp);
                }
            }
        }
    }
    cls_stream.write(cls_idx);
}

void demux_vit_encoder(
    hls::stream<hls::vector<gemm_t,     P_CP> >& gemm_stream,
    hls::stream<hls::vector<bias_t,     P_CP> >& bias_stream,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& qk_stream,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& v_stream,
    hls::stream<hls::vector<mlp1_out_t, P_CP> >& mlp1_stream,
    hls::stream<hls::vector<mlp2_out_t, P_CP> >& od_fc2_stream
) {
    // ViT Q/K/V 路由：
    //   保持旧链路 [h][qkv][tt][hct][tp] 顺序；每个 q/k/v 子流内部是 H -> TT -> HCT -> TP x CP。
    //   bias_stream 与 GEMM 输出一一对齐，每拍提供当前 channel tile 的 8-lane bias，DEMUX 内不再常驻 12 层 bias ROM。
    for (int h = 0; h < P_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            for (int tt = 0; tt < P_VIT_TT; ++tt) {
                for (int hct = 0; hct < P_VIT_HCT; ++hct) {
                    for (int tp = 0; tp < P_VIT_TP; ++tp) {
                        #pragma HLS pipeline II=1
                        hls::vector<gemm_t, P_CP> gemm_vec = gemm_stream.read();
                        hls::vector<bias_t, P_CP> bias_vec = bias_stream.read();
                        hls::vector<qkv_out_t, P_CP> qkv_vec;
                        for (int cp = 0; cp < P_CP; ++cp) {
                            #pragma HLS unroll
                            VIT_GEMM_TRUNC_T local_gemm = (VIT_GEMM_TRUNC_T)gemm_vec[cp];
                            VIT_QKV_TRUNC_T local_qkv;
                            if (qkv == 0 || qkv == 1) {
                                local_qkv = (VIT_QKV_TRUNC_T)(local_gemm >> (VIT_MHA_TRUNC_QK - P_TRUNC_BASE));
                            } else {
                                local_qkv = (VIT_QKV_TRUNC_T)(local_gemm >> (VIT_MHA_TRUNC_V - P_TRUNC_BASE));
                            }
                            local_qkv += (VIT_QKV_TRUNC_T)bias_vec[cp];
                            qkv_vec[cp] = (qkv_out_t)local_qkv;
                        }
                        if (qkv == 0 || qkv == 1) {
                            qk_stream.write(qkv_vec);
                        } else {
                            v_stream.write(qkv_vec);
                        }
                    }
                }
            }
        }
    }

    // ViT O/FC1/FC2 路由：
    //   od_fc2_stream 先写 O 后写 FC2；mlp1_stream 只写 FC1。
    //   rel/TP 推导输出 channel tile，匹配旧 ViT_DEMUX 的 GEMM 输出重排。
    for (int iter = 3 * P_VIT_T_QKVO; iter < P_VIT_T_DECODER; ++iter) {
        #pragma HLS pipeline II=1
        hls::vector<gemm_t, P_CP> gemm_vec = gemm_stream.read();
        hls::vector<bias_t, P_CP> bias_vec = bias_stream.read();
        if (iter < 4 * P_VIT_T_QKVO) {
            hls::vector<mlp2_out_t, P_CP> o_vec;
            int rel = iter - 3 * P_VIT_T_QKVO;
            (void)rel;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                VIT_GEMM_TRUNC_T local_gemm = (VIT_GEMM_TRUNC_T)gemm_vec[cp];
                VIT_FC2_TRUNC_T local_o = (VIT_FC2_TRUNC_T)(local_gemm >> (VIT_MHA_TRUNC_O - P_TRUNC_BASE));
                local_o += (VIT_FC2_TRUNC_T)bias_vec[cp];
                o_vec[cp] = (mlp2_out_t)local_o;
            }
            od_fc2_stream.write(o_vec);
        } else if (iter < 4 * P_VIT_T_QKVO + P_VIT_T_FC1) {
            hls::vector<mlp1_out_t, P_CP> fc1_vec;
            int rel = iter - 4 * P_VIT_T_QKVO;
            (void)rel;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                VIT_GEMM_TRUNC_T local_gemm = (VIT_GEMM_TRUNC_T)gemm_vec[cp];
                VIT_FC1_TRUNC_T local_fc1 = (VIT_FC1_TRUNC_T)(local_gemm >> (VIT_MLP_TRUNC_FC1 - P_TRUNC_BASE));
                local_fc1 += (VIT_FC1_TRUNC_T)bias_vec[cp];
                fc1_vec[cp] = (mlp1_out_t)local_fc1;
            }
            mlp1_stream.write(fc1_vec);
        } else {
            hls::vector<mlp2_out_t, P_CP> fc2_vec;
            int rel = iter - (4 * P_VIT_T_QKVO + P_VIT_T_FC1);
            (void)rel;
            for (int cp = 0; cp < P_CP; ++cp) {
                #pragma HLS unroll
                VIT_GEMM_TRUNC_T local_gemm = (VIT_GEMM_TRUNC_T)gemm_vec[cp];
                VIT_FC2_TRUNC_T local_fc2 = (VIT_FC2_TRUNC_T)(local_gemm >> (VIT_MLP_TRUNC_FC2 - P_TRUNC_BASE));
                local_fc2 += (VIT_FC2_TRUNC_T)bias_vec[cp];
                fc2_vec[cp] = (mlp2_out_t)local_fc2;
            }
            od_fc2_stream.write(fc2_vec);
        }
    }
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    int pos,
    hls::stream<hls::vector<gemm_t,     P_CP> >& gemm_stream,
    hls::stream<hls::vector<bias_t,     P_CP> >& bias_stream,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& qk_stream,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& v_stream,
    hls::stream<hls::vector<mlp1_out_t, P_CP> >& mlp1_stream,
    hls::stream<hls::vector<mlp2_out_t, P_CP> >& od_fc2_stream,
    hls::stream<hls::vector<cls_index_t, P_LLM_T> >& cls_stream
) {
    // 冻结后的 DEMUX 复用顶层接口：
    //   mode=0 为 LLM，mode=1 为 ViT；pos 是 LLM decode 的绝对位置，ViT mode 忽略。
    //   l_begin/l_close 为半开区间，LLM 允许 LLAMA_L 作为 CLS/lm_head 路径；
    //   CLS path 在 DEMUX 内做 argmax，cls_stream 只输出当前 8-token tile 的 token id。
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=gemm_stream
    #pragma HLS interface axis port=bias_stream
    #pragma HLS interface axis port=qk_stream
    #pragma HLS interface axis port=v_stream
    #pragma HLS interface axis port=mlp1_stream
    #pragma HLS interface axis port=od_fc2_stream
    #pragma HLS interface axis port=cls_stream
    #pragma HLS aggregate variable=gemm_stream   compact=bit
    #pragma HLS aggregate variable=bias_stream   compact=bit
    #pragma HLS aggregate variable=qk_stream     compact=bit
    #pragma HLS aggregate variable=v_stream      compact=bit
    #pragma HLS aggregate variable=mlp1_stream   compact=bit
    #pragma HLS aggregate variable=od_fc2_stream compact=bit
    #pragma HLS aggregate variable=cls_stream    compact=bit

    reuse_pos_t decode_pos = reuse_decode_pos(mode, pos);
    for (int l = l_begin; l < l_close; ++l) {
        #pragma HLS loop_tripcount min=1 max=33
        if (!reuse_is_valid_layer(mode, l, true)) {
            continue;
        }
        if (is_vit_mode(mode)) {
            demux_vit_encoder(gemm_stream, bias_stream, qk_stream, v_stream, mlp1_stream, od_fc2_stream);
        } else if (reuse_is_llm_cls(mode, l)) {
            demux_llm_cls(gemm_stream, cls_stream);
        } else {
            demux_llm_decoder(decode_pos.chunk_pos, gemm_stream, qk_stream, v_stream, mlp1_stream, od_fc2_stream);
        }
    }
}

// ============================================================================
// Testbench helpers
// ============================================================================
// testbench 同时覆盖：
//   - LLM decoder layer 0：产生 DEMUX_QK/V/UG/OD。
//   - LLM CLS/lm_head：DEMUX 内完成 argmax，产生 DEMUX_CLS_INDEX。
//   - ViT vision layer 0：产生 DEMUX_QK/V/FC1/OFC2。

namespace llm_demux_tb {

void slice_i64(vector<int64_t> &tensor, vector<int64_t> &array, int t_start, int t, int c, const string &name) {
    // LLM binaries 当前真实 T_LOAD 由文件长度决定，不能沿用旧 612 常量。
    tensor2array_dynamic_t<int64_t>(tensor, array.data(), 1, 1, t_start, t, c, c, name);
}

void prepare_decoder_data(int l, hls::stream<hls::vector<gemm_t, P_CP> >& gemm_stream) {
    // LLM decoder 输入来自共享 PERMUTE 生成的 CONDENSED_GEMM_Y.bin。
    const string condense_path = CONDENSE_PATH + to_string(l);
    auto CONDENSED_Y = read_tensor<int64_t>(condense_path + "/CONDENSED_GEMM_Y.bin");
    vector<int64_t> ref_condensed_y(P_LLM_NUM_Y);
    tensor2array<int64_t>(CONDENSED_Y, ref_condensed_y.data(), 1, 1, 1, 1, P_LLM_NUM_Y, P_LLM_NUM_Y);
    array2stream<int64_t, gemm_t, 1, 1, 1, 1, P_LLM_NUM_Y, P_CP>(ref_condensed_y.data(), gemm_stream, "LLM Input GEMM_Y");
}

void prepare_cls_data(hls::stream<hls::vector<gemm_t, P_CP> >& gemm_stream) {
    // LLM CLS 输入同样使用 PERMUTE 输出的 CONDENSED_GEMM_Y.bin，但目录为 decoder_LLAMA_L。
    const string condense_path = CONDENSE_PATH + to_string(LLAMA_L);
    auto CONDENSED_CLS = read_tensor<int64_t>(condense_path + "/CONDENSED_GEMM_Y.bin");
    vector<int64_t> ref_condensed_cls(P_LLM_NUM_CLS);
    tensor2array<int64_t>(CONDENSED_CLS, ref_condensed_cls.data(), 1, 1, 1, 1, P_LLM_NUM_CLS, P_LLM_NUM_CLS);
    array2stream<int64_t, gemm_t, 1, 1, 1, 1, P_LLM_NUM_CLS, P_CP>(ref_condensed_cls.data(), gemm_stream, "LLM Input CLS_GEMM_Y");
}

void compare_decoder_data(
    int l,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& qk_stream,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& v_stream,
    hls::stream<hls::vector<mlp1_out_t, P_CP> >& mlp1_stream,
    hls::stream<hls::vector<mlp2_out_t, P_CP> >& od_fc2_stream
) {
    // LLM 输出对比保持旧 DEMUX 的排列：
    //   QK 为 [H][Q/K][T][HC]，V/O 为 [H][T][HC]，UG 为 [CMT][U/G][T][CP]，D 为 [T][C]。
    const string binaries_path = BINARIES_PATH + to_string(l);
    const string condense_path = CONDENSE_PATH + to_string(l);

    auto MHA_Q = read_tensor<int64_t>(binaries_path + "/MHA_Q.bin");
    auto MHA_K = read_tensor<int64_t>(binaries_path + "/MHA_K.bin");
    auto MHA_V = read_tensor<int64_t>(binaries_path + "/MHA_V.bin");
    auto MHA_O = read_tensor<int64_t>(binaries_path + "/MHA_O.bin");
    auto MLP_U = read_tensor<int64_t>(binaries_path + "/MLP_XU.bin");
    auto MLP_G = read_tensor<int64_t>(binaries_path + "/MLP_XG.bin");
    auto MLP_D = read_tensor<int64_t>(binaries_path + "/MLP_XD.bin");

    vector<int64_t> ref_q(P_LLM_T * P_LLM_C);
    vector<int64_t> ref_k(P_LLM_T * P_LLM_C);
    vector<int64_t> ref_v(P_LLM_T * P_LLM_C);
    vector<int64_t> ref_o(P_LLM_T * P_LLM_C);
    vector<int64_t> ref_u(P_LLM_T * P_LLM_CM);
    vector<int64_t> ref_g(P_LLM_T * P_LLM_CM);
    vector<int64_t> ref_d(P_LLM_T * P_LLM_C);
    slice_i64(MHA_Q, ref_q, P_LLM_POS, P_LLM_T, P_LLM_C,  "MHA_Q");
    slice_i64(MHA_K, ref_k, P_LLM_POS, P_LLM_T, P_LLM_C,  "MHA_K");
    slice_i64(MHA_V, ref_v, P_LLM_POS, P_LLM_T, P_LLM_C,  "MHA_V");
    slice_i64(MHA_O, ref_o, P_LLM_POS, P_LLM_T, P_LLM_C,  "MHA_O");
    slice_i64(MLP_U, ref_u, P_LLM_POS, P_LLM_T, P_LLM_CM, "MLP_XU");
    slice_i64(MLP_G, ref_g, P_LLM_POS, P_LLM_T, P_LLM_CM, "MLP_XG");
    slice_i64(MLP_D, ref_d, P_LLM_POS, P_LLM_T, P_LLM_C,  "MLP_XD");

    constexpr int P_LLM_QK_BLOCKS = P_LLM_KVH * (P_LLM_GQA + 1);
    vector<int64_t> ref_perm_qk(P_LLM_QK_BLOCKS * P_LLM_T * P_LLM_HC);
    vector<int64_t> ref_perm_v (P_LLM_KVH * P_LLM_T * P_LLM_HC);
    vector<int64_t> ref_perm_o (P_LLM_H   * P_LLM_T * P_LLM_HC);
    vector<int64_t> ref_perm_ug(P_LLM_CMT * 2 * P_LLM_T * P_CP);
    vector<int64_t> ref_perm_d (P_LLM_T * P_LLM_C);
    for (int kv = 0; kv < P_LLM_KVH; ++kv) {
        int kv_head = kv * P_LLM_GQA;
        for (int block = 0; block < P_LLM_GQA + 1; ++block) {
            int src_head = (block == 0) ? kv_head : kv_head + (block - 1);
            for (int t = 0; t < P_LLM_T; ++t) {
                for (int hc = 0; hc < P_LLM_HC; ++hc) {
                    ref_perm_qk[(kv * (P_LLM_GQA + 1) + block) * P_LLM_T * P_LLM_HC + t * P_LLM_HC + hc] =
                        (block == 0) ? ref_k[t * P_LLM_C + src_head * P_LLM_HC + hc]
                                     : ref_q[t * P_LLM_C + src_head * P_LLM_HC + hc];
                }
            }
        }
        for (int t = 0; t < P_LLM_T; ++t) {
            for (int hc = 0; hc < P_LLM_HC; ++hc) {
                ref_perm_v[kv * P_LLM_T * P_LLM_HC + t * P_LLM_HC + hc] = ref_v[t * P_LLM_C + kv_head * P_LLM_HC + hc];
            }
        }
    }
    for (int h = 0; h < P_LLM_H; ++h) {
        for (int t = 0; t < P_LLM_T; ++t) {
            for (int hc = 0; hc < P_LLM_HC; ++hc) {
                ref_perm_o[h * P_LLM_T * P_LLM_HC + t * P_LLM_HC + hc] = ref_o[t * P_LLM_C + h * P_LLM_HC + hc];
            }
        }
    }
    for (int cot = 0; cot < P_LLM_CMT; ++cot) {
        for (int ug = 0; ug < 2; ++ug) {
            for (int t = 0; t < P_LLM_T; ++t) {
                for (int cop = 0; cop < P_CP; ++cop) {
                    ref_perm_ug[cot * 2 * P_LLM_T * P_CP + ug * P_LLM_T * P_CP + t * P_CP + cop] =
                        (ug == 0) ? ref_u[t * P_LLM_CM + cot * P_CP + cop]
                                  : ref_g[t * P_LLM_CM + cot * P_CP + cop];
                }
            }
        }
    }
    for (int t = 0; t < P_LLM_T; ++t) {
        for (int c = 0; c < P_LLM_C; ++c) {
            ref_perm_d[t * P_LLM_C + c] = ref_d[t * P_LLM_C + c];
        }
    }

    constexpr int NUM_QK = P_LLM_QK_BLOCKS * P_LLM_T * P_LLM_HC;
    constexpr int NUM_V  = P_LLM_KVH * P_LLM_T * P_LLM_HC;
    constexpr int NUM_UG = 2 * P_LLM_T * P_LLM_CM;
    constexpr int NUM_OD = 2 * P_LLM_T * P_LLM_C;
    hls::stream<hls::vector<qkv_out_t,  P_CP> > sim_qk_stream("llm_sim_qk_stream");
    hls::stream<hls::vector<qkv_out_t,  P_CP> > sim_v_stream ("llm_sim_v_stream");
    hls::stream<hls::vector<mlp1_out_t, P_CP> > sim_ug_stream("llm_sim_ug_stream");
    hls::stream<hls::vector<mlp2_out_t, P_CP> > sim_od_stream("llm_sim_od_stream");
    stream2stream<qkv_out_t,  NUM_QK, P_CP>(qk_stream,     sim_qk_stream);
    stream2stream<qkv_out_t,  NUM_V,  P_CP>(v_stream,      sim_v_stream);
    stream2stream<mlp1_out_t, NUM_UG, P_CP>(mlp1_stream,   sim_ug_stream);
    stream2stream<mlp2_out_t, NUM_OD, P_CP>(od_fc2_stream, sim_od_stream);
    save_condensed_tensor<int64_t, qkv_out_t,  NUM_QK, P_CP>(condense_path + "/CONDENSED_DEMUX_QK.bin", sim_qk_stream);
    save_condensed_tensor<int64_t, qkv_out_t,  NUM_V,  P_CP>(condense_path + "/CONDENSED_DEMUX_V.bin",  sim_v_stream);
    save_condensed_tensor<int64_t, mlp1_out_t, NUM_UG, P_CP>(condense_path + "/CONDENSED_DEMUX_UG.bin", sim_ug_stream);
    save_condensed_tensor<int64_t, mlp2_out_t, NUM_OD, P_CP>(condense_path + "/CONDENSED_DEMUX_OD.bin", sim_od_stream);

    vector<int64_t> dut_qk(ref_perm_qk.size());
    vector<int64_t> dut_v (ref_perm_v.size());
    vector<int64_t> dut_o (ref_perm_o.size());
    vector<int64_t> dut_ug(ref_perm_ug.size());
    vector<int64_t> dut_d (ref_perm_d.size());
    stream2array_unpack<int64_t, qkv_out_t,  P_LLM_QK_BLOCKS, P_LLM_T, P_LLM_T, P_LLM_HC, P_CP>(sim_qk_stream, dut_qk.data(), "LLM outQK", true);
    stream2array_unpack<int64_t, qkv_out_t,  P_LLM_KVH,       P_LLM_T, P_LLM_T, P_LLM_HC, P_CP>(sim_v_stream,  dut_v.data(),  "LLM outV",  true);
    stream2array_unpack<int64_t, mlp2_out_t, P_LLM_H,       P_LLM_T, P_LLM_T, P_LLM_HC, P_CP>(sim_od_stream, dut_o.data(),  "LLM Output O",  true);
    stream2array_unpack<int64_t, mlp1_out_t, P_LLM_CMT * 2, P_LLM_T, P_LLM_T, P_CP,     P_CP>(sim_ug_stream, dut_ug.data(), "LLM Output UG", true);
    stream2array_unpack<int64_t, mlp2_out_t, 1,             P_LLM_T, P_LLM_T, P_LLM_C,  P_CP>(sim_od_stream, dut_d.data(),  "LLM Output D",  true);
    assert(sim_qk_stream.size() == 0);
    assert(sim_v_stream.size() == 0);
    assert(sim_ug_stream.size() == 0);
    assert(sim_od_stream.size() == 0);

    compare<int64_t>(ref_perm_qk.data(), dut_qk.data(), ref_perm_qk.size(), "LLM_QK");
    compare<int64_t>(ref_perm_v.data(),  dut_v.data(),  ref_perm_v.size(),  "LLM_V");
    compare<int64_t>(ref_perm_o.data(),  dut_o.data(),  ref_perm_o.size(),  "LLM_O");
    compare<int64_t>(ref_perm_ug.data(), dut_ug.data(), ref_perm_ug.size(), "LLM_UG");
    compare<int64_t>(ref_perm_d.data(),  dut_d.data(),  ref_perm_d.size(),  "LLM_D");
}

void compare_cls_data(hls::stream<hls::vector<cls_index_t, P_LLM_T> >& cls_stream) {
    // LLM CLS 输出已在 DEMUX 内做 argmax：
    //   testbench 仍读取 T x VOCAB golden logits 计算参考 index，只保存 T 个 token id。
    const string binaries_path = BINARIES_PATH + to_string(LLAMA_L);
    const string condense_path = CONDENSE_PATH + to_string(LLAMA_L);
    auto CLS = read_tensor<int64_t>(binaries_path + "/CLS.bin");
    vector<int64_t> ref_cls(P_LLM_NUM_CLS);
    // CLS/lm_head 参考 index 与 PERMUTE/RMS_LAYERNORM 使用同一 decode POS 窗口。
    tensor2array_dynamic_t<int64_t>(CLS, ref_cls.data(), 1, 1, P_LLM_POS, P_LLM_T, P_LLM_VOCAB, P_LLM_VOCAB, "CLS");

    hls::stream<hls::vector<cls_index_t, P_LLM_T> > sim_cls_index_stream("llm_sim_cls_index_stream");
    stream2stream<cls_index_t, P_LLM_T, P_LLM_T>(cls_stream, sim_cls_index_stream);
    save_condensed_tensor<int64_t, cls_index_t, P_LLM_T, P_LLM_T>(condense_path + "/CONDENSED_DEMUX_CLS_INDEX.bin", sim_cls_index_stream);

    vector<int64_t> ref_idx(P_LLM_T);
    for (int t = 0; t < P_LLM_T; ++t) {
        CLS_TRUNC_T max_val = (CLS_TRUNC_T)ref_cls[t * P_LLM_VOCAB];
        cls_index_t max_idx = 0;
        for (int vocab = 1; vocab < P_LLM_VOCAB; ++vocab) {
            CLS_TRUNC_T cur = (CLS_TRUNC_T)ref_cls[t * P_LLM_VOCAB + vocab];
            if (cur > max_val) {
                max_val = cur;
                max_idx = (cls_index_t)vocab;
            }
        }
        ref_idx[t] = (int64_t)max_idx;
    }

    vector<int64_t> dut_idx(P_LLM_T);
    hls::vector<cls_index_t, P_LLM_T> dut_vec = sim_cls_index_stream.read();
    for (int t = 0; t < P_LLM_T; ++t) {
        dut_idx[t] = (int64_t)dut_vec[t];
    }
    assert(sim_cls_index_stream.size() == 0);
    compare<int64_t>(ref_idx.data(), dut_idx.data(), P_LLM_T, "LLM_CLS_INDEX");
}

void test_layer(int l_begin, int l_close) {
    // LLM mode step1 覆盖普通 decoder 和 CLS 两种 layer 类型；pos 使用当前 tile 最后一位以保持旧 V mask 语义。
    hls::stream<hls::vector<gemm_t,     P_CP> > gemm_stream("llm_gemm_stream");
    hls::stream<hls::vector<bias_t,     P_CP> > bias_stream("llm_bias_unused_stream");
    hls::stream<hls::vector<qkv_out_t,  P_CP> > qk_stream("llm_qk_stream");
    hls::stream<hls::vector<qkv_out_t,  P_CP> > v_stream("llm_v_stream");
    hls::stream<hls::vector<mlp1_out_t, P_CP> > mlp1_stream("llm_mlp1_stream");
    hls::stream<hls::vector<mlp2_out_t, P_CP> > od_fc2_stream("llm_od_fc2_stream");
    hls::stream<hls::vector<cls_index_t, P_LLM_T> > cls_stream("llm_cls_index_stream");

    for (int l = l_begin; l < l_close; ++l) {
        if (l == LLAMA_L) {
            prepare_cls_data(gemm_stream);
        } else {
            prepare_decoder_data(l, gemm_stream);
        }
    }

    ::top(MODE_LLM, l_begin, l_close, P_LLM_TOP_POS,
          gemm_stream, bias_stream, qk_stream, v_stream, mlp1_stream, od_fc2_stream, cls_stream);
    assert(gemm_stream.size() == 0);
    assert(bias_stream.size() == 0);

    for (int l = l_begin; l < l_close; ++l) {
        if (l == LLAMA_L) {
            compare_cls_data(cls_stream);
        } else {
            compare_decoder_data(l, qk_stream, v_stream, mlp1_stream, od_fc2_stream);
        }
    }
}

} // namespace llm_demux_tb

namespace vit_demux_tb {

void write_bias_tile(
    const vector<int64_t> &bias,
    int c_tile,
    hls::stream<hls::vector<bias_t, P_CP> >& bias_stream
) {
    // testbench 按外部 M_AXI 将来应采用的格式，把当前 channel tile 的 8 个 bias lane 打成一拍。
    hls::vector<bias_t, P_CP> bias_vec;
    for (int cp = 0; cp < P_CP; ++cp) {
        bias_vec[cp] = (bias_t)bias[c_tile * P_CP + cp];
    }
    bias_stream.write(bias_vec);
}

void prepare_encoder_data(
    int l,
    hls::stream<hls::vector<gemm_t, P_CP> >& gemm_stream,
    hls::stream<hls::vector<bias_t, P_CP> >& bias_stream
) {
    // ViT 输入来自共享 PERMUTE 生成的 CONDENSED_GEMM_Y.bin；bias 由外部流按 GEMM 输出顺序同步输入。
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);
    const string binaries_path = VIT_BINARIES_PATH + to_string(l);
    auto CONDENSED_Y = read_tensor<int64_t>(condense_path + "/CONDENSED_GEMM_Y.bin");
    vector<int64_t> ref_condensed_y(P_VIT_NUM_Y);
    tensor2array<int64_t>(CONDENSED_Y, ref_condensed_y.data(), 1, 1, 1, 1, P_VIT_NUM_Y, P_VIT_NUM_Y);
    array2stream<int64_t, gemm_t, 1, 1, 1, 1, P_VIT_NUM_Y, P_CP>(ref_condensed_y.data(), gemm_stream, "ViT Input GEMM_Y");

    auto MHA_BQ = read_tensor<int64_t>(binaries_path + "/MHA_BQ.bin");
    auto MHA_BK = read_tensor<int64_t>(binaries_path + "/MHA_BK.bin");
    auto MHA_BV = read_tensor<int64_t>(binaries_path + "/MHA_BV.bin");
    auto MHA_BO = read_tensor<int64_t>(binaries_path + "/MHA_BO.bin");
    auto MLP_B1 = read_tensor<int64_t>(binaries_path + "/MLP_B1.bin");
    auto MLP_B2 = read_tensor<int64_t>(binaries_path + "/MLP_B2.bin");

    // Q/K/V bias 顺序与 DEMUX QKV GEMM 输出保持一致：[h][qkv][tt][hct][tp]。
    for (int h = 0; h < P_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            for (int tt = 0; tt < P_VIT_TT; ++tt) {
                for (int hct = 0; hct < P_VIT_HCT; ++hct) {
                    int c_tile = h * P_VIT_HCT + hct;
                    for (int tp = 0; tp < P_VIT_TP; ++tp) {
                        if (qkv == 0) {
                            write_bias_tile(MHA_BQ, c_tile, bias_stream);
                        } else if (qkv == 1) {
                            write_bias_tile(MHA_BK, c_tile, bias_stream);
                        } else {
                            write_bias_tile(MHA_BV, c_tile, bias_stream);
                        }
                    }
                }
            }
        }
    }

    // O/FC1/FC2 bias 顺序与后半段 GEMM 输出保持一致：O 后 FC1 后 FC2，均为 [tt][channel_tile][tp]。
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int ct = 0; ct < P_VIT_CT; ++ct) {
            for (int tp = 0; tp < P_VIT_TP; ++tp) {
                write_bias_tile(MHA_BO, ct, bias_stream);
            }
        }
    }
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int cmt = 0; cmt < P_VIT_CMT; ++cmt) {
            for (int tp = 0; tp < P_VIT_TP; ++tp) {
                write_bias_tile(MLP_B1, cmt, bias_stream);
            }
        }
    }
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int ct = 0; ct < P_VIT_CT; ++ct) {
            for (int tp = 0; tp < P_VIT_TP; ++tp) {
                write_bias_tile(MLP_B2, ct, bias_stream);
            }
        }
    }
}

void compare_encoder_data(
    int l,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& qk_stream,
    hls::stream<hls::vector<qkv_out_t,  P_CP> >& v_stream,
    hls::stream<hls::vector<mlp1_out_t, P_CP> >& mlp1_stream,
    hls::stream<hls::vector<mlp2_out_t, P_CP> >& od_fc2_stream
) {
    // ViT 输出对比保持旧 DEMUX 的排列：
    //   QK 为 [H][Q/K][TT][HCT][TP]，V 为 [H][TT][HCT][TP]，
    //   FC1 为 [TT][CMT][TP]，O/FC2 在同一 od_fc2_stream 中顺序拼接。
    const string binaries_path = VIT_BINARIES_PATH + to_string(l);
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);

    auto MHA_Q    = read_tensor<int64_t>(binaries_path + "/MHA_Q.bin");
    auto MHA_K    = read_tensor<int64_t>(binaries_path + "/MHA_K.bin");
    auto MHA_V    = read_tensor<int64_t>(binaries_path + "/MHA_V.bin");
    auto MHA_O    = read_tensor<int64_t>(binaries_path + "/MHA_O.bin");
    auto MLP_FC1  = read_tensor<int64_t>(binaries_path + "/MLP_XFC1.bin");
    auto MLP_FC2  = read_tensor<int64_t>(binaries_path + "/MLP_XFC2.bin");

    vector<int64_t> ref_q(P_VIT_T * P_VIT_C);
    vector<int64_t> ref_k(P_VIT_T * P_VIT_C);
    vector<int64_t> ref_v(P_VIT_T * P_VIT_C);
    vector<int64_t> ref_o(P_VIT_T * P_VIT_C);
    vector<int64_t> ref_fc1(P_VIT_T * P_VIT_CM);
    vector<int64_t> ref_fc2(P_VIT_T * P_VIT_C);
    tensor2array<int64_t>(MHA_Q,   ref_q.data(),   1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_C,  P_VIT_C);
    tensor2array<int64_t>(MHA_K,   ref_k.data(),   1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_C,  P_VIT_C);
    tensor2array<int64_t>(MHA_V,   ref_v.data(),   1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_C,  P_VIT_C);
    tensor2array<int64_t>(MHA_O,   ref_o.data(),   1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_C,  P_VIT_C);
    tensor2array<int64_t>(MLP_FC1, ref_fc1.data(), 1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_CM, P_VIT_CM);
    tensor2array<int64_t>(MLP_FC2, ref_fc2.data(), 1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_C,  P_VIT_C);

    vector<int64_t> ref_perm_qk(P_VIT_H * 2 * P_VIT_T * P_VIT_HC);
    vector<int64_t> ref_perm_v (P_VIT_H *     P_VIT_T * P_VIT_HC);
    vector<int64_t> ref_perm_o (P_VIT_T * P_VIT_C);
    vector<int64_t> ref_perm_fc1(P_VIT_T * P_VIT_CM);
    vector<int64_t> ref_perm_fc2(P_VIT_T * P_VIT_C);
    for (int h = 0; h < P_VIT_H; ++h) {
        for (int qk = 0; qk < 2; ++qk) {
            for (int t = 0; t < P_VIT_T; ++t) {
                for (int hc = 0; hc < P_VIT_HC; ++hc) {
                    ref_perm_qk[h * 2 * P_VIT_T * P_VIT_HC + qk * P_VIT_T * P_VIT_HC + t * P_VIT_HC + hc] =
                        (qk == 0) ? ref_q[t * P_VIT_C + h * P_VIT_HC + hc]
                                  : ref_k[t * P_VIT_C + h * P_VIT_HC + hc];
                }
            }
        }
        for (int t = 0; t < P_VIT_T; ++t) {
            for (int hc = 0; hc < P_VIT_HC; ++hc) {
                ref_perm_v[h * P_VIT_T * P_VIT_HC + t * P_VIT_HC + hc] = ref_v[t * P_VIT_C + h * P_VIT_HC + hc];
            }
        }
    }
    for (int t = 0; t < P_VIT_T; ++t) {
        for (int c = 0; c < P_VIT_C; ++c) {
            ref_perm_o[t * P_VIT_C + c] = ref_o[t * P_VIT_C + c];
            ref_perm_fc2[t * P_VIT_C + c] = ref_fc2[t * P_VIT_C + c];
        }
        for (int c = 0; c < P_VIT_CM; ++c) {
            ref_perm_fc1[t * P_VIT_CM + c] = ref_fc1[t * P_VIT_CM + c];
        }
    }

    constexpr int NUM_QK   = 2 * P_VIT_T * P_VIT_C;
    constexpr int NUM_V    =     P_VIT_T * P_VIT_C;
    constexpr int NUM_FC1  =     P_VIT_T * P_VIT_CM;
    constexpr int NUM_OFC2 = 2 * P_VIT_T * P_VIT_C;
    hls::stream<hls::vector<qkv_out_t,  P_CP> > sim_qk_stream("vit_sim_qk_stream");
    hls::stream<hls::vector<qkv_out_t,  P_CP> > sim_v_stream("vit_sim_v_stream");
    hls::stream<hls::vector<mlp1_out_t, P_CP> > sim_fc1_stream("vit_sim_fc1_stream");
    hls::stream<hls::vector<mlp2_out_t, P_CP> > sim_ofc2_stream("vit_sim_ofc2_stream");
    stream2stream<qkv_out_t,  NUM_QK,   P_CP>(qk_stream,     sim_qk_stream);
    stream2stream<qkv_out_t,  NUM_V,    P_CP>(v_stream,      sim_v_stream);
    stream2stream<mlp1_out_t, NUM_FC1,  P_CP>(mlp1_stream,   sim_fc1_stream);
    stream2stream<mlp2_out_t, NUM_OFC2, P_CP>(od_fc2_stream, sim_ofc2_stream);
    save_condensed_tensor<int64_t, qkv_out_t,  NUM_QK,   P_CP>(condense_path + "/CONDENSED_DEMUX_QK.bin",   sim_qk_stream);
    save_condensed_tensor<int64_t, qkv_out_t,  NUM_V,    P_CP>(condense_path + "/CONDENSED_DEMUX_V.bin",    sim_v_stream);
    save_condensed_tensor<int64_t, mlp1_out_t, NUM_FC1,  P_CP>(condense_path + "/CONDENSED_DEMUX_FC1.bin",  sim_fc1_stream);
    save_condensed_tensor<int64_t, mlp2_out_t, NUM_OFC2, P_CP>(condense_path + "/CONDENSED_DEMUX_OFC2.bin", sim_ofc2_stream);

    vector<int64_t> dut_qk(ref_perm_qk.size());
    vector<int64_t> dut_v(ref_perm_v.size());
    vector<int64_t> dut_o(ref_perm_o.size());
    vector<int64_t> dut_fc1(ref_perm_fc1.size());
    vector<int64_t> dut_fc2(ref_perm_fc2.size());
    stream2array_unpack<int64_t, qkv_out_t,  P_VIT_H * 2, P_VIT_T, P_VIT_TP, P_VIT_HC, P_CP>(sim_qk_stream,   dut_qk.data(),   "ViT Output QK",  true);
    stream2array_unpack<int64_t, qkv_out_t,  P_VIT_H,     P_VIT_T, P_VIT_TP, P_VIT_HC, P_CP>(sim_v_stream,    dut_v.data(),    "ViT Output V",   true);
    stream2array_unpack<int64_t, mlp2_out_t, 1,           P_VIT_T, P_VIT_TP, P_VIT_C,  P_CP>(sim_ofc2_stream, dut_o.data(),    "ViT Output O",   true);
    stream2array_unpack<int64_t, mlp1_out_t, 1,           P_VIT_T, P_VIT_TP, P_VIT_CM, P_CP>(sim_fc1_stream,  dut_fc1.data(),  "ViT Output FC1", true);
    stream2array_unpack<int64_t, mlp2_out_t, 1,           P_VIT_T, P_VIT_TP, P_VIT_C,  P_CP>(sim_ofc2_stream, dut_fc2.data(),  "ViT Output FC2", true);
    assert(sim_qk_stream.size() == 0);
    assert(sim_v_stream.size() == 0);
    assert(sim_fc1_stream.size() == 0);
    assert(sim_ofc2_stream.size() == 0);

    compare<int64_t>(ref_perm_qk.data(),   dut_qk.data(),   ref_perm_qk.size(),   "ViT_QK");
    compare<int64_t>(ref_perm_v.data(),    dut_v.data(),    ref_perm_v.size(),    "ViT_V");
    compare<int64_t>(ref_perm_o.data(),    dut_o.data(),    ref_perm_o.size(),    "ViT_O");
    compare<int64_t>(ref_perm_fc1.data(),  dut_fc1.data(),  ref_perm_fc1.size(),  "ViT_FC1");
    compare<int64_t>(ref_perm_fc2.data(),  dut_fc2.data(),  ref_perm_fc2.size(),  "ViT_FC2");
}

void test_layer(int l_begin, int l_close) {
    // ViT mode step1 当前验证 vision layer 0；top 的 pos 输入保留但在 ViT mode 中不参与路径。
    hls::stream<hls::vector<gemm_t,     P_CP> > gemm_stream("vit_gemm_stream");
    hls::stream<hls::vector<bias_t,     P_CP> > bias_stream("vit_bias_stream");
    hls::stream<hls::vector<qkv_out_t,  P_CP> > qk_stream("vit_qk_stream");
    hls::stream<hls::vector<qkv_out_t,  P_CP> > v_stream("vit_v_stream");
    hls::stream<hls::vector<mlp1_out_t, P_CP> > mlp1_stream("vit_mlp1_stream");
    hls::stream<hls::vector<mlp2_out_t, P_CP> > od_fc2_stream("vit_od_fc2_stream");
    hls::stream<hls::vector<cls_index_t, P_LLM_T> > cls_stream("vit_cls_unused_stream");

    for (int l = l_begin; l < l_close; ++l) {
        prepare_encoder_data(l, gemm_stream, bias_stream);
    }

    ::top(MODE_VIT, l_begin, l_close, P_VIT_POS,
          gemm_stream, bias_stream, qk_stream, v_stream, mlp1_stream, od_fc2_stream, cls_stream);
    assert(gemm_stream.size() == 0);
    assert(bias_stream.size() == 0);
    assert(cls_stream.size() == 0);

    for (int l = l_begin; l < l_close; ++l) {
        compare_encoder_data(l, qk_stream, v_stream, mlp1_stream, od_fc2_stream);
    }
}

} // namespace vit_demux_tb

int main() {
    // 一个 CSim 同时覆盖 DEMUX 三条实际调用路径，避免只验证单 mode 后遗漏端口、截断或文件命名问题。
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        llm_demux_tb::test_layer(0, 1);
        llm_demux_tb::test_layer(LLAMA_L, LLAMA_L + 1);
    }
    vit_demux_tb::test_layer(vit_layer, vit_layer + 1);
    return 0;
}
