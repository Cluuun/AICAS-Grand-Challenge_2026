#include "../src/reuse_common.h"
#include "../src/rope_qk.h"
#include "../src/cordic.h"
#include "../src/utils.h"

// ============================================================================
// LLM/ViT 条件复用 ROPE_QK
// ============================================================================
// 功能：
//   1. 冻结后的输入边界统一为 DEMUX 的 Q/K 截断流，输出统一为量化后的 Q/K q+s。
//   2. LLM mode 执行 CORDIC + RoPE 旋转 + Quantizer，保持旧 LLM_ROPE_QK 数值路径。
//   3. ViT mode 不启动 CORDIC/RoPE，直接对 DEMUX Q/K 做 bypass+Quantizer。
//
// 资源意图：
//   - mode 只在 layer 调用粒度选择路径，TP/CP 内层不逐元素切换。
//   - QK_GEMM 后续只接收已量化 Q/K，避免在 QK_GEMM 内为 ViT 私有保留 Quantizer。
//   - LLM RoPE 输出和 ViT bypass 输出汇入同一种 23-bit 中间流，再进入同一个共享 Quantizer；
//     不为 LLM/ViT 分别例化两套量化器。

constexpr int P_CP = REUSE_CP;

typedef REUSE_QKV_TRUNC_T qk_i_t;
typedef REUSE_QKV_TRUNC_T qk_mid_t;
typedef REUSE_AQ_T        qk_q_t;
typedef REUSE_AS_T        qk_s_t;
static_assert(qk_mid_t::width == ROT_T::width, "ROPE_QK shared quantizer expects ROT_T and qk_mid_t to share width");

// LLM 参数：ROPE 只覆盖普通 decoder 层，不支持 LLAMA_L 的 CLS/lm_head 路径。
constexpr int P_LLM_T      = REUSE_LLM_TILE_T;
constexpr int P_LLM_POS    = 96;
constexpr int P_LLM_H      = LLAMA_H;
constexpr int P_LLM_KVH    = LLAMA_KVH;
constexpr int P_LLM_GQA    = P_LLM_H / P_LLM_KVH;
constexpr int P_LLM_C      = LLAMA_C;
constexpr int P_LLM_HC     = LLAMA_HC;
constexpr int P_LLM_HCT    = P_LLM_HC / P_CP;
constexpr int P_LLM_NUM_QK = P_LLM_KVH * (P_LLM_GQA + 1) * P_LLM_T * P_LLM_HC;
constexpr int P_LLM_NUM_S  = P_LLM_NUM_QK / LLAMA_G;
constexpr int P_LLM_NUM_TILE = P_LLM_NUM_QK / P_CP;

// ViT 参数：输入顺序沿用 DEMUX 的 [H][Q/K][TT][HCT][TP] x CP，Quantizer 只逐拍处理并保持原顺序。
constexpr int P_VIT_T      = VIT_S;
constexpr int P_VIT_H      = VIT_H;
constexpr int P_VIT_HC     = VIT_HC;
constexpr int P_VIT_HCT    = P_VIT_HC / P_CP;
constexpr int P_VIT_NUM_QK = 2 * P_VIT_H * P_VIT_T * P_VIT_HC;
constexpr int P_VIT_NUM_S  = 2 * P_VIT_H * P_VIT_T * P_VIT_HCT;
constexpr int P_VIT_NUM_TILE = P_VIT_NUM_QK / P_CP;

CORDIC<P_LLM_H, P_LLM_T, 1, P_LLM_C, P_CP / 2> cordic_inst;
ROPE_QK<P_LLM_H, P_LLM_T, P_LLM_T, P_LLM_C, P_CP> rope_qk_inst;

void rope_qk_llm_gqa_apply(
    hls::stream<hls::vector<qk_i_t, P_CP> >& qk_i_stream,
    hls::stream<hls::vector<COS_SIN_T, P_CP> >& cos_sin_stream,
    hls::stream<hls::vector<qk_mid_t, P_CP> >& quant_i_stream
) {
    // Native GQA order from DEMUX is per KV group: K, Q0, Q1, Q2.
    // RoPE angles are shared across heads, so cache one 8-token tile of cos/sin and
    // replay it for each compact K/Q block.
    constexpr int CP2 = P_CP / 2;
    COS_SIN_T cos_sin[2][P_LLM_T][P_LLM_HC / 2];
    #pragma HLS array_reshape variable=cos_sin complete dim=1
    #pragma HLS array_reshape variable=cos_sin cyclic factor=CP2 dim=3
    #pragma HLS bind_storage variable=cos_sin type=ram_2p impl=lutram

    for (int t = 0; t < P_LLM_T; ++t) {
        for (int hct = 0; hct < P_LLM_HCT; ++hct) {
            #pragma HLS pipeline II=1
            hls::vector<COS_SIN_T, P_CP> cs_vec = cos_sin_stream.read();
            for (int cp2 = 0; cp2 < CP2; ++cp2) {
                #pragma HLS unroll
                cos_sin[0][t][hct * CP2 + cp2] = cs_vec[0 * CP2 + cp2];
                cos_sin[1][t][hct * CP2 + cp2] = cs_vec[1 * CP2 + cp2];
            }
        }
    }

    for (int kv = 0; kv < P_LLM_KVH; ++kv) {
        for (int block = 0; block < P_LLM_GQA + 1; ++block) {
            for (int hct = 0; hct < P_LLM_HCT; ++hct) {
                for (int t = 0; t < P_LLM_T; ++t) {
                    #pragma HLS pipeline II=1
                    hls::vector<qk_i_t, P_CP> qk_vec = qk_i_stream.read();
                    hls::vector<qk_mid_t, P_CP> rot_vec;
                    for (int cp2 = 0; cp2 < CP2; ++cp2) {
                        #pragma HLS unroll
                        COS_SIN_T cur_cos = cos_sin[0][t][hct * CP2 + cp2];
                        COS_SIN_T cur_sin = cos_sin[1][t][hct * CP2 + cp2];
                        int idx = cp2 * 2;
                        rot_vec[idx + 0] = (qk_vec[idx + 0] * cur_cos - qk_vec[idx + 1] * cur_sin) >> MHA_TRUNC_ROT;
                        rot_vec[idx + 1] = (qk_vec[idx + 1] * cur_cos + qk_vec[idx + 0] * cur_sin) >> MHA_TRUNC_ROT;
                    }
                    quant_i_stream.write(rot_vec);
                }
            }
        }
    }
}

void rope_qk_llm_rotate(
    int chunk,
    hls::stream<hls::vector<qk_i_t, P_CP> >& qk_i_stream,
    hls::stream<hls::vector<qk_mid_t, P_CP> >& quant_i_stream
) {
    #pragma HLS dataflow

    hls::stream<hls::vector<COS_SIN_T, P_CP> > cos_sin_stream("cos_sin_stream");

    cordic_inst.do_cordic(chunk, cos_sin_stream);
    rope_qk_llm_gqa_apply(qk_i_stream, cos_sin_stream, quant_i_stream);
}

void rope_qk_vit_bypass(
    hls::stream<hls::vector<qk_i_t, P_CP> >& qk_i_stream,
    hls::stream<hls::vector<qk_mid_t, P_CP> >& quant_i_stream
) {
    // ViT 路径：
    //   不做 RoPE，也不读 pos；把 DEMUX Q/K 交错流直接转入共享 Quantizer 的中间流。
    for (int tile = 0; tile < P_VIT_NUM_TILE; ++tile) {
        #pragma HLS pipeline II=1
        quant_i_stream.write(qk_i_stream.read());
    }
}

void rope_qk_prepare_mid(
    REUSE_MODE_T mode,
    int chunk,
    hls::stream<hls::vector<qk_i_t,   P_CP> >& qk_i_stream,
    hls::stream<hls::vector<qk_mid_t, P_CP> >& quant_i_stream
) {
    // mode 分支只决定前处理来源：
    //   LLM 生成 RoPE 后中间流，ViT 做 bypass 中间流；两者的量化硬件在后级统一复用。
    #pragma HLS inline off
    if (is_vit_mode(mode)) {
        rope_qk_vit_bypass(qk_i_stream, quant_i_stream);
    } else {
        rope_qk_llm_rotate(chunk, qk_i_stream, quant_i_stream);
    }
}

void shared_qk_quantize(
    int num_tiles,
    hls::stream<hls::vector<qk_mid_t, P_CP> >& i_stream,
    hls::stream<hls::vector<qk_q_t,   P_CP> >& q_stream,
    hls::stream<hls::vector<qk_s_t,   1   > >& s_stream
) {
    // 单套共享 Quantizer：
    //   LLM 与 ViT 的输入都已规整为 P_CP=8 lane、23-bit 中间流。
    //   num_tiles 在 layer 粒度选择循环次数，内层 absmax/scale/round/clamp 逻辑完全共用。
    #pragma HLS inline off
    constexpr int Q_BITS = qk_q_t::width;
    constexpr int Q_MAX = +(1 << (Q_BITS - 1)) - 1;
    constexpr int Q_MIN = -(1 << (Q_BITS - 1));
    constexpr int A_CLAMP_MAX = 15;

    for (int tile = 0; tile < num_tiles; ++tile) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=P_LLM_NUM_TILE max=P_VIT_NUM_TILE

        hls::vector<qk_mid_t, P_CP> i_vec = i_stream.read();
        hls::vector<qk_q_t,   P_CP> q_vec;
        hls::vector<qk_s_t,   1   > s_vec;

        qk_mid_t abs_max = 0;
        for (int cp = 0; cp < P_CP; ++cp) {
            #pragma HLS unroll
            qk_mid_t cur_abs = (qk_mid_t)abs(i_vec[cp]);
            abs_max = max(abs_max, cur_abs);
        }

        int8_t s_val = log2ceil(abs_max) - (Q_BITS - 1);
        s_vec[0] = clamp(s_val, 0, A_CLAMP_MAX);

        for (int cp = 0; cp < P_CP; ++cp) {
            #pragma HLS unroll
            qk_mid_t q_val = i_vec[cp];
            qk_s_t scale = s_vec[0];
            if (scale != 0) {
                q_val = q_val >> (scale - 1);
                q_val = q_val + 1;
                q_val = q_val >> 1;
            }
            // 与软件 Quantizer 保持一致：signed int8 激活需要同时限制上下界。
            q_vec[cp] = clamp(q_val, (qk_mid_t)Q_MIN, (qk_mid_t)Q_MAX);
        }

        q_stream.write(q_vec);
        s_stream.write(s_vec);
    }
}

void rope_qk_process(
    REUSE_MODE_T mode,
    int chunk,
    int num_tiles,
    hls::stream<hls::vector<qk_i_t, P_CP> >& qk_i_stream,
    hls::stream<hls::vector<qk_q_t, P_CP> >& qk_q_stream,
    hls::stream<hls::vector<qk_s_t, 1   > >& qk_s_stream
) {
    // 统一处理骨架：
    //   前级按 mode 做 LLM RoPE 或 ViT bypass，后级只有一个 shared_qk_quantize 调用。
    //   该结构用于约束 HLS 只生成一套量化 datapath，而不是在两条 mode 分支各放一套。
    #pragma HLS dataflow

    hls::stream<hls::vector<qk_mid_t, P_CP> > quant_i_stream("quant_i_stream");
    #pragma HLS stream variable=quant_i_stream depth=16

    rope_qk_prepare_mid(mode, chunk, qk_i_stream, quant_i_stream);
    shared_qk_quantize(num_tiles, quant_i_stream, qk_q_stream, qk_s_stream);
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    int pos,
    hls::stream<hls::vector<qk_i_t, P_CP> >& qk_i_stream,
    hls::stream<hls::vector<qk_q_t, P_CP> >& qk_q_stream,
    hls::stream<hls::vector<qk_s_t, 1   > >& qk_s_stream
) {
    // 冻结后的 ROPE_QK 顶层接口：
    //   mode=0 为 LLM，mode=1 为 ViT；pos 是 LLM decode 的绝对位置，ViT mode 忽略。
    //   l_begin/l_close 为半开区间，只接受普通 attention 层，不包含 LLAMA_L。
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=qk_i_stream
    #pragma HLS interface axis port=qk_q_stream
    #pragma HLS interface axis port=qk_s_stream
    #pragma HLS aggregate variable=qk_i_stream compact=bit
    #pragma HLS aggregate variable=qk_q_stream compact=bit
    #pragma HLS aggregate variable=qk_s_stream compact=bit

    reuse_pos_t decode_pos = reuse_decode_pos(mode, pos);
    for (int l = l_begin; l < l_close; ++l) {
        #pragma HLS loop_tripcount min=1 max=32
        if (!reuse_is_valid_layer(mode, l, false)) {
            continue;
        }
        int num_tiles = is_vit_mode(mode) ? P_VIT_NUM_TILE : P_LLM_NUM_TILE;
        rope_qk_process(mode, decode_pos.chunk, num_tiles, qk_i_stream, qk_q_stream, qk_s_stream);
    }
}

// ============================================================================
// Testbench helpers
// ============================================================================
// step1 同时覆盖：
//   - LLM mode decoder layer 0：DEMUX_QK -> CORDIC/ROPE -> Quantizer。
//   - ViT mode vision layer 0：DEMUX_QK -> bypass Quantizer，并保存给后续 QK_GEMM。

namespace rope_qk_tb {

int ref_log2ceil_i64(long long x) {
    if (x >= 1) {
        x -= 1;
    }
    for (int i = 63; i >= 1; --i) {
        if ((x >> (i - 1)) & 1LL) {
            return i;
        }
    }
    return 0;
}

template<class in_t, class out_t, class scale_t, int NUM_ELEMS, int CP, int G>
void reference_quant_flat(
    const vector<int64_t> &input,
    vector<int8_t> &ref_q,
    vector<int8_t> &ref_s
) {
    // testbench 独立参考 Quantizer：
    //   按 AXIS 原始拍顺序逐拍量化，每组 G=8 个 lane 共用一个 scale。
    //   该函数只用于验证 ViT bypass+Quantizer，避免拿 DUT 输出自比较。
    constexpr int CPG = CP / G;
            constexpr int Q_BITS = out_t::width;
            constexpr int Q_MAX = (1 << (Q_BITS - 1)) - 1;
            constexpr int Q_MIN = -(1 << (Q_BITS - 1));
            constexpr int A_CLAMP_MAX = 15;
    static_assert(NUM_ELEMS % CP == 0, "NUM_ELEMS must be multiple of CP");

    for (int tile = 0; tile < NUM_ELEMS / CP; ++tile) {
        in_t lane[CP];
        for (int cp = 0; cp < CP; ++cp) {
            lane[cp] = (in_t)input[tile * CP + cp];
        }
        for (int cpg = 0; cpg < CPG; ++cpg) {
            long long abs_max = 0;
            for (int g = 0; g < G; ++g) {
                long long v = (long long)lane[cpg * G + g];
                long long av = (v < 0) ? -v : v;
                if (av > abs_max) {
                    abs_max = av;
                }
            }
            int s_val = ref_log2ceil_i64(abs_max) - (Q_BITS - 1);
            s_val = clamp<int, int>(s_val, 0, A_CLAMP_MAX);
            ref_s[tile * CPG + cpg] = (int8_t)((scale_t)s_val);

            for (int g = 0; g < G; ++g) {
                int idx = cpg * G + g;
                in_t q_val = lane[idx];
                if (s_val != 0) {
                    q_val = q_val >> (s_val - 1);
                    q_val = q_val + 1;
                    q_val = q_val >> 1;
                }
                in_t clipped = clamp(q_val, (in_t)Q_MIN, (in_t)Q_MAX);
                ref_q[tile * CP + idx] = (int8_t)((out_t)clipped);
            }
        }
    }
}

} // namespace rope_qk_tb

namespace llm_rope_qk_tb {

void build_raw_qk_input(int l, int pos, vector<int64_t> &condensed_qk) {
    // 任意 pos 的单模块回归不能依赖默认 pos=96 生成的 DEMUX condensed 文件；
    // 这里直接按 DEMUX QK 输出顺序从软件原始 MHA_Q/MHA_K 切片构造输入。
    const string file_path = BINARIES_PATH + to_string(l);
    auto MHA_Q = read_tensor<int64_t>(file_path + "/MHA_Q.bin");
    auto MHA_K = read_tensor<int64_t>(file_path + "/MHA_K.bin");
    int q_t_load = infer_tensor_t_load<int64_t>(MHA_Q, 1, P_LLM_C, "MHA_Q");
    int k_t_load = infer_tensor_t_load<int64_t>(MHA_K, 1, P_LLM_C, "MHA_K");
    check_tensor_window(q_t_load, pos, P_LLM_T, "MHA_Q");
    check_tensor_window(k_t_load, pos, P_LLM_T, "MHA_K");

    for (int kv = 0; kv < P_LLM_KVH; ++kv) {
        for (int block = 0; block < P_LLM_GQA + 1; ++block) {
            int src_head = (block == 0) ? kv * P_LLM_GQA : kv * P_LLM_GQA + (block - 1);
            for (int hct = 0; hct < P_LLM_HCT; ++hct) {
                for (int t = 0; t < P_LLM_T; ++t) {
                    for (int cp = 0; cp < P_CP; ++cp) {
                        int hc = hct * P_CP + cp;
                        int out_idx = ((kv * (P_LLM_GQA + 1) + block) * P_LLM_HCT + hct) * P_LLM_T * P_CP
                                    + t * P_CP + cp;
                        int src_idx = (pos + t) * P_LLM_C + src_head * P_LLM_HC + hc;
                        condensed_qk[out_idx] = (block == 0)
                            ? MHA_K[src_idx]
                            : MHA_Q[src_idx];
                    }
                }
            }
        }
    }
}

void test_layer(int l) {
    // LLM 参考来自软件的 MHA_Q/K_Q/S，输入来自复用 DEMUX 生成的 CONDENSED_DEMUX_QK.bin。
    // pos 允许通过 VLM_LLM_POS 覆盖，用于直接复测板上 long-prefill 首个失败窗口。
    int pos = reuse_tb_llm_pos(P_LLM_POS);
    const string file_path = BINARIES_PATH + to_string(l);
    const string save_path = CONDENSE_PATH + to_string(l);

    vector<int8_t> ref_q(P_LLM_NUM_QK);
    vector<int8_t> ref_s(P_LLM_NUM_S);
    {
        auto REF_QROT_Q_TENSOR = read_tensor<int8_t>(file_path + "/MHA_Q_Q.bin");
        auto REF_KROT_Q_TENSOR = read_tensor<int8_t>(file_path + "/MHA_K_Q.bin");
        auto REF_QROT_S_TENSOR = read_tensor<int8_t>(file_path + "/MHA_Q_S.bin");
        auto REF_KROT_S_TENSOR = read_tensor<int8_t>(file_path + "/MHA_K_S.bin");
        // Q 与 K 的保存长度可能不同：当前 Q 相关文件为 T_LOAD=877，K cache 相关文件为 S_LOAD=880。
        // 分别按文件长度推导，避免旧 612/616 常量污染 golden。
        int q_q_t_load = infer_tensor_t_load<int8_t>(REF_QROT_Q_TENSOR, P_LLM_H, P_LLM_HC,  "MHA_Q_Q");
        int k_q_t_load = infer_tensor_t_load<int8_t>(REF_KROT_Q_TENSOR, P_LLM_H, P_LLM_HC,  "MHA_K_Q");
        int q_s_t_load = infer_tensor_t_load<int8_t>(REF_QROT_S_TENSOR, P_LLM_H, P_LLM_HCT, "MHA_Q_S");
        int k_s_t_load = infer_tensor_t_load<int8_t>(REF_KROT_S_TENSOR, P_LLM_H, P_LLM_HCT, "MHA_K_S");
        check_tensor_window(q_q_t_load, pos, P_LLM_T, "MHA_Q_Q");
        check_tensor_window(k_q_t_load, pos, P_LLM_T, "MHA_K_Q");
        check_tensor_window(q_s_t_load, pos, P_LLM_T, "MHA_Q_S");
        check_tensor_window(k_s_t_load, pos, P_LLM_T, "MHA_K_S");

        for (int kv = 0; kv < P_LLM_KVH; ++kv) {
            for (int block = 0; block < P_LLM_GQA + 1; ++block) {
                int src_head = (block == 0) ? kv * P_LLM_GQA : kv * P_LLM_GQA + (block - 1);
                for (int t = 0; t < P_LLM_T; ++t) {
                    for (int hc = 0; hc < P_LLM_HC; ++hc) {
                        int out_idx = (kv * (P_LLM_GQA + 1) + block) * P_LLM_T * P_LLM_HC + t * P_LLM_HC + hc;
                        ref_q[out_idx] = (block == 0)
                            ? REF_KROT_Q_TENSOR[src_head * k_q_t_load * P_LLM_HC + (t + pos) * P_LLM_HC + hc]
                            : REF_QROT_Q_TENSOR[src_head * q_q_t_load * P_LLM_HC + (t + pos) * P_LLM_HC + hc];
                    }
                    for (int hct = 0; hct < P_LLM_HCT; ++hct) {
                        int out_idx = (kv * (P_LLM_GQA + 1) + block) * P_LLM_T * P_LLM_HCT + t * P_LLM_HCT + hct;
                        ref_s[out_idx] = (block == 0)
                            ? REF_KROT_S_TENSOR[src_head * k_s_t_load * P_LLM_HCT + (t + pos) * P_LLM_HCT + hct]
                            : REF_QROT_S_TENSOR[src_head * q_s_t_load * P_LLM_HCT + (t + pos) * P_LLM_HCT + hct];
                    }
                }
            }
        }
    }

    vector<int64_t> condensed_qk(P_LLM_NUM_QK);
    build_raw_qk_input(l, pos, condensed_qk);
    save_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_QK.bin", condensed_qk.data(), P_LLM_NUM_QK);

    hls::stream<hls::vector<qk_i_t, P_CP> > qk_i_stream("llm_qk_i_stream");
    hls::stream<hls::vector<qk_q_t, P_CP> > qk_q_stream("llm_qk_q_stream");
    hls::stream<hls::vector<qk_s_t, 1   > > qk_s_stream("llm_qk_s_stream");
    array2stream<int64_t, qk_i_t, 1, 1, 1, 1, P_LLM_NUM_QK, P_CP>(condensed_qk.data(), qk_i_stream, "LLM QK_I", true);

    ::top(MODE_LLM, l, l + 1, pos, qk_i_stream, qk_q_stream, qk_s_stream);
    assert(qk_i_stream.size() == 0);

    save_condensed_tensor<int8_t, qk_q_t, P_LLM_NUM_QK, P_CP>(save_path + "/CONDENSED_ROPE_QK_QUANT_ROT_Q.bin", qk_q_stream);
    save_condensed_tensor<int8_t, qk_s_t, P_LLM_NUM_S,  1   >(save_path + "/CONDENSED_ROPE_QK_QUANT_ROT_S.bin", qk_s_stream);

    vector<int8_t> dut_q(P_LLM_NUM_QK);
    vector<int8_t> dut_s(P_LLM_NUM_S);
    stream2array_unpack<int8_t, qk_q_t, P_LLM_KVH * (P_LLM_GQA + 1), P_LLM_T, P_LLM_T, P_LLM_HC,  P_CP>(qk_q_stream, dut_q.data(), "LLM ropeQ", true);
    stream2array_unpack<int8_t, qk_s_t, P_LLM_KVH * (P_LLM_GQA + 1), P_LLM_T, P_LLM_T, P_LLM_HCT, 1   >(qk_s_stream, dut_s.data(), "LLM ropeS", true);
    assert(qk_q_stream.size() == 0);
    assert(qk_s_stream.size() == 0);

    compare<int8_t>(ref_q.data(), dut_q.data(), P_LLM_NUM_QK, "LLM_ROPE_QK_Q");
    compare<int8_t>(ref_s.data(), dut_s.data(), P_LLM_NUM_S,  "LLM_ROPE_QK_S");
}

} // namespace llm_rope_qk_tb

namespace vit_rope_qk_tb {

void test_layer(int l) {
    // ViT 参考由 testbench 按 Quantizer 公式从 DEMUX_QK 直接计算。
    // 输出文件名采用冻结方案的 BYPASS_Q/S，供后续复用 QK_GEMM 读取。
    const string save_path = VIT_CONDENSE_PATH + to_string(l);
    auto CONDENSED_QK = read_tensor<int64_t>(save_path + "/CONDENSED_DEMUX_QK.bin");
    vector<int64_t> condensed_qk(P_VIT_NUM_QK);
    tensor2array<int64_t>(CONDENSED_QK, condensed_qk.data(), 1, 1, 1, 1, P_VIT_NUM_QK, P_VIT_NUM_QK);

    vector<int8_t> ref_q(P_VIT_NUM_QK);
    vector<int8_t> ref_s(P_VIT_NUM_S);
    rope_qk_tb::reference_quant_flat<qk_i_t, qk_q_t, qk_s_t, P_VIT_NUM_QK, P_CP, VIT_G>(
        condensed_qk, ref_q, ref_s);

    hls::stream<hls::vector<qk_i_t, P_CP> > qk_i_stream("vit_qk_i_stream");
    hls::stream<hls::vector<qk_q_t, P_CP> > qk_q_stream("vit_qk_q_stream");
    hls::stream<hls::vector<qk_s_t, 1   > > qk_s_stream("vit_qk_s_stream");
    array2stream<int64_t, qk_i_t, 1, 1, 1, 1, P_VIT_NUM_QK, P_CP>(condensed_qk.data(), qk_i_stream, "ViT QK_I", true);

    ::top(MODE_VIT, l, l + 1, 0, qk_i_stream, qk_q_stream, qk_s_stream);
    assert(qk_i_stream.size() == 0);

    save_condensed_tensor<int8_t, qk_q_t, P_VIT_NUM_QK, P_CP>(save_path + "/CONDENSED_ROPE_QK_BYPASS_Q.bin", qk_q_stream);
    save_condensed_tensor<int8_t, qk_s_t, P_VIT_NUM_S,  1   >(save_path + "/CONDENSED_ROPE_QK_BYPASS_S.bin", qk_s_stream);

    vector<int8_t> dut_q(P_VIT_NUM_QK);
    vector<int8_t> dut_s(P_VIT_NUM_S);
    stream2array<int8_t, qk_q_t, 1, 1, 1, P_VIT_NUM_QK, P_CP>(qk_q_stream, dut_q.data(), "ViT ROPE_QK_Q", true);
    stream2array<int8_t, qk_s_t, 1, 1, 1, P_VIT_NUM_S,  1   >(qk_s_stream, dut_s.data(), "ViT ROPE_QK_S", true);
    assert(qk_q_stream.size() == 0);
    assert(qk_s_stream.size() == 0);

    compare<int8_t>(ref_q.data(), dut_q.data(), P_VIT_NUM_QK, "ViT_ROPE_QK_Q");
    compare<int8_t>(ref_s.data(), dut_s.data(), P_VIT_NUM_S,  "ViT_ROPE_QK_S");
}

} // namespace vit_rope_qk_tb

int main() {
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        llm_rope_qk_tb::test_layer(0);
        printf("LLM ROPE_QK layer 0 passed\n");
    }

    if (!reuse_tb_only_llm()) {
        vit_rope_qk_tb::test_layer(vit_layer);
        printf("ViT ROPE_QK layer %d passed\n", vit_layer);
    }

    return 0;
}
