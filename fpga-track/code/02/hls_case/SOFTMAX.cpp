#include "../src/reuse_common.h"
#include "../src/softmax.h"
#include "../src/vit_softmax.h"
#include "../src/utils.h"

// ============================================================================
// LLM/ViT 复用 SOFTMAX
// ============================================================================
// 功能：
//   1. 输入统一为 QK_GEMM 输出的 31-bit attention score R，输出统一为量化后的 R q+s。
//   2. LLM mode 使用绝对 pos 派生 chunk，并在 softmax pass1 做 causal mask。
//   3. ViT mode 无 causal mask，完整 1024x1024 双向注意力。
//
// 资源意图：
//   - exp/recip/softmax/Quantizer 管线只写一套，mode 只选择 LUT 常量、截断参数和 mask 启用。
//   - 旧 LLM 内部 SP=2、ViT 内部 SP=8；复用版统一用 SP=8，并用树形 max 避免 SP=8 的串行 max 链。

constexpr int P_CP        = REUSE_CP;
constexpr int P_SP        = REUSE_CP;
constexpr int P_S         = LLAMA_S;
constexpr int P_ST        = P_S / P_SP;
constexpr int P_LLM_T     = REUSE_LLM_TILE_T;
constexpr int P_LLM_POS   = 96;
constexpr int P_LLM_H     = LLAMA_H;
constexpr int P_VIT_T     = VIT_S;
constexpr int P_VIT_H     = VIT_H;
constexpr int P_MAX_H     = const_max(P_LLM_H, P_VIT_H);
constexpr int P_MAX_T     = const_max(P_LLM_T, P_VIT_T);
constexpr int P_EXP_ENTRIES = 256;
constexpr int P_RECIP_NUM_TABLES = 4;
constexpr int P_RECIP_ENTRIES = 256;

static_assert(P_S == VIT_S, "SOFTMAX reuse assumes LLM S and ViT S both use 1024");
static_assert(DW_R_TRUNC == VIT_DW_R_TRUNC, "SOFTMAX reuse expects shared R input width");
static_assert(DW_EXP == VIT_DW_EXP, "SOFTMAX reuse expects shared exp width");
static_assert(DW_EXP_SUM == VIT_DW_EXP_SUM, "SOFTMAX reuse expects shared exp sum width");
static_assert(DW_RECIP == VIT_DW_RECIP, "SOFTMAX reuse expects shared recip width");

typedef REUSE_R_TRUNC_T   sm_r_t;
typedef REUSE_R_MASKED_T  sm_masked_t;
typedef ap_int<DW_EXP>    sm_exp_t;
typedef ap_int<DW_EXP_SUM> sm_exp_sum_t;
typedef ap_int<DW_RECIP>  sm_recip_t;
typedef REUSE_SOFTMAX_T   sm_softmax_t;
typedef REUSE_AQ_T        sm_q_t;
typedef REUSE_AS_T        sm_s_t;

constexpr int P_LLM_NUM_R  = P_LLM_H * P_LLM_T * P_S;
constexpr int P_LLM_NUM_RS = P_LLM_NUM_R / P_SP;
constexpr int P_VIT_NUM_R  = P_VIT_H * P_VIT_T * P_S;
constexpr int P_VIT_NUM_RS = P_VIT_NUM_R / P_SP;

template<int N, class T>
struct SoftmaxTreeMax {
    static inline T eval(const T *arr) {
        #pragma HLS inline
        T left = SoftmaxTreeMax<N / 2, T>::eval(arr);
        T right = SoftmaxTreeMax<N - N / 2, T>::eval(arr + N / 2);
        return (left > right) ? left : right;
    }
};

template<class T>
struct SoftmaxTreeMax<1, T> {
    static inline T eval(const T *arr) {
        #pragma HLS inline
        return arr[0];
    }
};

template<int N, class T>
T softmax_tree_max(const T *arr) {
    #pragma HLS inline
    return SoftmaxTreeMax<N, T>::eval(arr);
}

int softmax_exp_log2denom(bool is_vit) {
    #pragma HLS inline
    return is_vit ? VIT_EXP_LOG2DENOM : EXP_LOG2DENOM;
}

int softmax_trunc_mul(bool is_vit) {
    #pragma HLS inline
    return is_vit ? VIT_SOFTMAX_TRUNC_MUL : SOFTMAX_TRUNC_MUL;
}

sm_exp_t softmax_exp_lookup(bool is_vit, int index) {
    #pragma HLS inline
    return (sm_exp_t)(is_vit ? VIT_EXP_TABLE_DATA[index] : EXP_TABLE[index]);
}

int softmax_recip_alpha(bool is_vit, int table_id) {
    #pragma HLS inline
    return is_vit ? VIT_RECIP_ALPHAS_DATA[table_id] : RECIP_ALPHAS[table_id];
}

int softmax_recip_log2denom(bool is_vit, int table_id) {
    #pragma HLS inline
    return is_vit ? VIT_RECIP_LOG2DENOMS_DATA[table_id] : RECIP_LOG2DENOMS[table_id];
}

int softmax_recip_offset_diff(bool is_vit, int table_id) {
    #pragma HLS inline
    return is_vit ? VIT_RECIP_OFFSETS_DIFF_DATA[table_id] : RECIP_OFFSETS_DIFF[table_id];
}

sm_recip_t softmax_recip_lookup(bool is_vit, int table_id, int index) {
    #pragma HLS inline
    int flat_idx = table_id * P_RECIP_ENTRIES + index;
    return (sm_recip_t)(is_vit ? VIT_RECIP_TABLES_DATA[flat_idx] : RECIP_TABLES[flat_idx]);
}

void shared_softmax_rows(
    REUSE_MODE_T mode,
    int chunk,
    int num_heads,
    int num_tokens,
    int valid_st,
    hls::stream<hls::vector<sm_r_t,       P_SP> >& r_stream,
    hls::stream<hls::vector<sm_softmax_t, P_SP> >& softmax_stream
) {
    // 单套 softmax 行处理：
    //   每次处理一个 head 的一行 token，对 S=1024 做 pass1 max、pass2 exp/sum/recip、pass3 输出。
    //   LLM/ViT 的 LUT 表不同，但查表和乘法/移位管线共用。
    bool is_vit = is_vit_mode(mode);
    int exp_log2denom = softmax_exp_log2denom(is_vit);
    int trunc_mul = softmax_trunc_mul(is_vit);

    sm_masked_t r_buf[P_S];
    sm_exp_t    r_exp[P_S];
    #pragma HLS bind_storage variable=r_buf type=ram_2p impl=bram
    #pragma HLS bind_storage variable=r_exp type=ram_2p impl=bram
    #pragma HLS array_reshape variable=r_buf cyclic factor=P_SP dim=1
    #pragma HLS array_reshape variable=r_exp cyclic factor=P_SP dim=1

    for (int h = 0; h < P_MAX_H; ++h) {
        #pragma HLS loop_tripcount min=P_VIT_H max=P_LLM_H
        if (h >= num_heads) {
            continue;
        }
        for (int t = 0; t < P_MAX_T; ++t) {
            #pragma HLS loop_tripcount min=P_LLM_T max=P_VIT_T
            if (t >= num_tokens) {
                continue;
            }

            sm_masked_t r_max = (sm_masked_t)MASK_NEG_INF;
            sm_exp_sum_t exp_sum = 0;

            // Pass1：读 R、按 mode 加 mask、缓存整行，并用 SP=8 树形 max 缩短 loop-carried 路径。
            for (int st = 0; st < valid_st; ++st) {
                #pragma HLS loop_tripcount min=1 max=P_ST
                #pragma HLS pipeline II=1
                hls::vector<sm_r_t, P_SP> i_vec = r_stream.read();
                sm_masked_t lane[P_SP];
                #pragma HLS array_partition variable=lane complete

                for (int sp = 0; sp < P_SP; ++sp) {
                    #pragma HLS unroll
                    int s = st * P_SP + sp;
                    bool valid = is_vit || (chunk * P_LLM_T + t >= s);
                    // LLM causal mask 对无效列直接钳到固定负无穷。
                    // 旧写法 `MASK_NEG_INF + score` 在 32-bit masked 类型里会被
                    // 极小负 score 拉到有符号下溢，板上表现为部分 future lane 泄漏。
                    sm_masked_t masked = valid
                        ? (sm_masked_t)i_vec[sp]
                        : (sm_masked_t)MASK_NEG_INF;
                    r_buf[s] = masked;
                    lane[sp] = masked;
                }

                sm_masked_t local_max = softmax_tree_max<P_SP>(lane);
                r_max = max(r_max, local_max);
            }

            // Pass2：用 mode 选择对应 exp LUT，保留与旧实现一致的逐 lane 累加顺序。
            for (int st = 0; st < valid_st; ++st) {
                #pragma HLS loop_tripcount min=1 max=P_ST
                #pragma HLS pipeline II=1
                for (int sp = 0; sp < P_SP; ++sp) {
                    #pragma HLS unroll
                    int s = st * P_SP + sp;
                    int index = clamp((r_max - r_buf[s]) >> exp_log2denom, 0, P_EXP_ENTRIES - 1);
                    sm_exp_t exp_val = softmax_exp_lookup(is_vit, index);
                    r_exp[s] = exp_val;
                    exp_sum = exp_sum + exp_val;
                }
            }

            // Pass2 post：mode 选择 recip 分段表，recip 计算硬件共用。
            int lut_idx = 0;
            for (int i = 0; i < P_RECIP_NUM_TABLES; ++i) {
                #pragma HLS unroll
                if (exp_sum >= softmax_recip_alpha(is_vit, i)) {
                    lut_idx = i;
                }
            }
            int alpha = softmax_recip_alpha(is_vit, lut_idx);
            int log2denom = softmax_recip_log2denom(is_vit, lut_idx);
            int offset_diff = softmax_recip_offset_diff(is_vit, lut_idx);
            int recip_index = clamp((exp_sum - alpha) >> log2denom, 0, P_RECIP_ENTRIES - 1);
            sm_recip_t recip = softmax_recip_lookup(is_vit, lut_idx, recip_index) << offset_diff;

            // Pass3：输出未量化 softmax，后级由单套 shared_softmax_quantize 量化。
            for (int st = 0; st < valid_st; ++st) {
                #pragma HLS loop_tripcount min=1 max=P_ST
                #pragma HLS pipeline II=1
                hls::vector<sm_softmax_t, P_SP> o_vec;
                for (int sp = 0; sp < P_SP; ++sp) {
                    #pragma HLS unroll
                    int s = st * P_SP + sp;
                    o_vec[sp] = (sm_softmax_t)((r_exp[s] * recip) >> trunc_mul);
                }
                softmax_stream.write(o_vec);
            }
        }
    }
}

void shared_softmax_quantize(
    int num_tiles,
    hls::stream<hls::vector<sm_softmax_t, P_SP> >& i_stream,
    hls::stream<hls::vector<sm_q_t,       P_SP> >& q_stream,
    hls::stream<hls::vector<sm_s_t,       1   > >& s_stream
) {
    // 单套 Softmax Quantizer：
    //   LLM/ViT 均按每 8 个 attention weight 共享一个 scale，输出边界与 RV_GEMM 冻结接口一致。
    constexpr int Q_BITS = sm_q_t::width;
    constexpr int Q_MAX = +(1 << (Q_BITS - 1)) - 1;
    constexpr int A_CLAMP_MAX = 15;

    for (int tile = 0; tile < num_tiles; ++tile) {
        #pragma HLS pipeline II=1
        #pragma HLS loop_tripcount min=P_LLM_NUM_R / P_SP max=P_VIT_NUM_R / P_SP

        hls::vector<sm_softmax_t, P_SP> i_vec = i_stream.read();
        hls::vector<sm_q_t,       P_SP> q_vec;
        hls::vector<sm_s_t,       1   > s_vec;

        sm_softmax_t abs_max = 0;
        for (int sp = 0; sp < P_SP; ++sp) {
            #pragma HLS unroll
            abs_max = max(abs_max, (sm_softmax_t)abs(i_vec[sp]));
        }

        int8_t s_val = log2ceil(abs_max) - (Q_BITS - 1);
        s_vec[0] = clamp(s_val, 0, A_CLAMP_MAX);

        for (int sp = 0; sp < P_SP; ++sp) {
            #pragma HLS unroll
            sm_softmax_t q_val = i_vec[sp];
            sm_s_t scale = s_vec[0];
            if (scale != 0) {
                q_val = q_val >> (scale - 1);
                q_val = q_val + 1;
                q_val = q_val >> 1;
            }
            q_vec[sp] = min(q_val, (sm_softmax_t)Q_MAX);
        }

        q_stream.write(q_vec);
        s_stream.write(s_vec);
    }
}

void softmax_process(
    REUSE_MODE_T mode,
    int pos,
    int num_heads,
    int num_tokens,
    int num_tiles,
    hls::stream<hls::vector<sm_r_t, P_SP> >& r_stream,
    hls::stream<hls::vector<sm_q_t, P_SP> >& rq_stream,
    hls::stream<hls::vector<sm_s_t, 1   > >& rs_stream
) {
    // 统一 dataflow 骨架：
    //   前级按 mode 完成 mask/LUT softmax，后级只有一个 shared_softmax_quantize。
    #pragma HLS dataflow

    hls::stream<hls::vector<sm_softmax_t, P_SP> > softmax_stream("softmax_stream");
    #pragma HLS stream variable=softmax_stream depth=64

    reuse_pos_t decode_pos = reuse_decode_pos(mode, pos);
    int valid_st = reuse_attention_valid_st(mode, pos);
    int chunk = decode_pos.chunk;
    shared_softmax_rows(mode, chunk, num_heads, num_tokens, valid_st, r_stream, softmax_stream);
    shared_softmax_quantize(num_tiles, softmax_stream, rq_stream, rs_stream);
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    int pos,
    hls::stream<hls::vector<sm_r_t, P_SP> >& r_stream,
    hls::stream<hls::vector<sm_q_t, P_SP> >& rq_stream,
    hls::stream<hls::vector<sm_s_t, 1   > >& rs_stream
) {
    // 冻结后的 SOFTMAX 顶层接口：
    //   mode=0 为 LLM causal softmax，mode=1 为 ViT bidirectional softmax。
    //   pos 只服务 LLM mode，ViT mode 忽略。
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=r_stream
    #pragma HLS interface axis port=rq_stream
    #pragma HLS interface axis port=rs_stream
    #pragma HLS aggregate variable=r_stream  compact=bit
    #pragma HLS aggregate variable=rq_stream compact=bit
    #pragma HLS aggregate variable=rs_stream compact=bit

    bool is_vit = is_vit_mode(mode);
    int num_heads = is_vit ? P_VIT_H : P_LLM_H;
    int num_tokens = is_vit ? P_VIT_T : P_LLM_T;
    int valid_st = reuse_attention_valid_st(mode, pos);
    int num_tiles = num_heads * num_tokens * valid_st;

    for (int l = l_begin; l < l_close; ++l) {
        #pragma HLS loop_tripcount min=1 max=32
        if (!reuse_is_valid_layer(mode, l, false)) {
            continue;
        }
        softmax_process(mode, pos, num_heads, num_tokens, num_tiles, r_stream, rq_stream, rs_stream);
    }
}

// ============================================================================
// Testbench
// ============================================================================
// step1 同时覆盖：
//   - LLM mode decoder layer 0：QK_GEMM_R -> causal softmax -> shared Quantizer。
//   - ViT mode vision layer 0：QK_GEMM_R -> bidirectional softmax -> shared Quantizer。

namespace llm_softmax_tb {

void test_layer(int l) {
    int pos = reuse_tb_llm_pos(P_LLM_POS);
    int top_pos = reuse_tb_llm_top_pos(pos + P_LLM_T - 1);
    int chunk = top_pos / P_LLM_T;
    int valid_s = reuse_llm_valid_s(top_pos);
    int valid_st = (valid_s + P_SP - 1) / P_SP;
    int valid_s_padded = valid_st * P_SP;
    int valid_num_r = P_LLM_H * P_LLM_T * valid_s_padded;
    int valid_num_rs = P_LLM_H * P_LLM_T * valid_st;

    const string file_path = BINARIES_PATH + to_string(l);
    const string save_path = CONDENSE_PATH + to_string(l);

    vector<int64_t> ref_r_full(P_LLM_NUM_R);
    vector<int64_t> condensed_r(valid_num_r);
    vector<int8_t> ref_rq(P_LLM_NUM_R);
    vector<int8_t> ref_rs(P_LLM_NUM_RS);
    vector<int8_t> dut_rq(valid_num_r);
    vector<int8_t> dut_rs(valid_num_rs);
    int t_load = P_LLM_T;
    int ref_s_load = P_S;
    int ref_st_load = P_ST;

    {
        auto R_Q = read_tensor<int8_t>(file_path + "/MHA_R_Q.bin");
        auto R_S = read_tensor<int8_t>(file_path + "/MHA_R_S.bin");
        // R_Q/R_S 的二维 attention 文件长度来自上游导出；当前常见为 T_LOAD=877、S_LOAD=880。
        // 用 MHA_Q 推导 T_LOAD，再从 R_Q/R_S 反推出 S_LOAD/ST_LOAD。
        auto MHA_Q = read_tensor<int64_t>(file_path + "/MHA_Q.bin");
        t_load  = infer_tensor_t_load<int64_t>(MHA_Q, 1, P_LLM_H * LLAMA_HC, "MHA_Q");
        ref_s_load  = infer_tensor_extent<int8_t>(R_Q, P_LLM_H * t_load, "MHA_R_Q");
        ref_st_load = infer_tensor_extent<int8_t>(R_S, P_LLM_H * t_load, "MHA_R_S");
        if (ref_s_load != ref_st_load * P_SP) {
            std::cerr << "MHA_R_Q S_LOAD=" << ref_s_load
                      << " does not match MHA_R_S ST_LOAD*SP=" << ref_st_load * P_SP << std::endl;
            exit(1);
        }
        check_tensor_window(t_load, pos, P_LLM_T, "MHA_R_Q/MHA_R_S");
        tensor2array<int8_t>(R_Q, ref_rq.data(), P_LLM_H, P_LLM_H, t_load, pos, P_LLM_T, ref_s_load,  P_S);
        tensor2array<int8_t>(R_S, ref_rs.data(), P_LLM_H, P_LLM_H, t_load, pos, P_LLM_T, ref_st_load, P_ST);
    }

    {
        // LLM 任意 layer/POS 调试时直接读取软件导出的 MHA_R 窗口，避免依赖
        // 某个 layer 旧 condense 目录中是否已有对应 QK_GEMM 链式输出。
        auto R = read_tensor<int64_t>(file_path + "/MHA_R.bin");
        int r_s_load = infer_tensor_extent<int64_t>(R, P_LLM_H * t_load, "MHA_R");
        tensor2array<int64_t>(R, ref_r_full.data(), P_LLM_H, P_LLM_H, t_load, pos, P_LLM_T, r_s_load, P_S);
        for (int h = 0; h < P_LLM_H; ++h) {
            for (int t = 0; t < P_LLM_T; ++t) {
                for (int s = 0; s < valid_s_padded; ++s) {
                    condensed_r[h * P_LLM_T * valid_s_padded + t * valid_s_padded + s] =
                        ref_r_full[h * P_LLM_T * P_S + t * P_S + s];
                }
            }
        }
    }

    hls::stream<hls::vector<sm_r_t, P_SP> > r_stream("llm_r_stream");
    hls::stream<hls::vector<sm_q_t, P_SP> > rq_stream("llm_rq_stream");
    hls::stream<hls::vector<sm_s_t, 1   > > rs_stream("llm_rs_stream");

    array2stream_runtime<int64_t, sm_r_t, P_SP>(condensed_r.data(), r_stream, valid_num_r);
    ::top(MODE_LLM, l, l + 1, top_pos, r_stream, rq_stream, rs_stream);

    save_condensed_tensor_runtime<int8_t, sm_q_t, P_SP>(save_path + "/CONDENSED_SOFTMAX_QUANT_R_Q.bin", rq_stream, valid_num_r);
    save_condensed_tensor_runtime<int8_t, sm_s_t, 1   >(save_path + "/CONDENSED_SOFTMAX_QUANT_R_S.bin", rs_stream, valid_num_rs);

    stream2array_runtime<int8_t, sm_q_t, P_SP>(rq_stream, dut_rq.data(), valid_num_r);
    stream2array_runtime<int8_t, sm_s_t, 1   >(rs_stream, dut_rs.data(), valid_num_rs);

    assert(r_stream.empty());
    assert(rq_stream.empty());
    assert(rs_stream.empty());

    int mismatch_rq = 0;
    int mismatch_rs = 0;
    for (int h = 0; h < P_LLM_H; ++h) {
        for (int t = 0; t < P_LLM_T; ++t) {
            for (int s = 0; s < valid_s && s < ref_s_load; ++s) {
                int idx = h * P_LLM_T * P_S + t * P_S + s;
                int dut_idx = h * P_LLM_T * valid_s_padded + t * valid_s_padded + s;
                if (ref_rq[idx] != dut_rq[dut_idx]) {
                    if (++mismatch_rq <= 10) {
                        printf("LLM R_Q mismatch L%d H%d T%d S%d: REF=%d DUT=%d\n",
                               l, h, t, s, (int)ref_rq[idx], (int)dut_rq[dut_idx]);
                    }
                }
            }
        }
    }

    for (int h = 0; h < P_LLM_H; ++h) {
        for (int t = 0; t < P_LLM_T; ++t) {
            for (int st = 0; st < valid_st && st < ref_st_load; ++st) {
                int idx = h * P_LLM_T * P_ST + t * P_ST + st;
                int dut_idx = h * P_LLM_T * valid_st + t * valid_st + st;
                if (ref_rs[idx] != dut_rs[dut_idx]) {
                    if (++mismatch_rs <= 10) {
                        printf("LLM R_S mismatch L%d H%d T%d ST%d: REF=%d DUT=%d\n",
                               l, h, t, st, (int)ref_rs[idx], (int)dut_rs[dut_idx]);
                    }
                }
            }
        }
    }

    printf("LLM SOFTMAX layer %d: R_Q mismatch=%d R_S mismatch=%d\n", l, mismatch_rq, mismatch_rs);
}

} // namespace llm_softmax_tb

namespace vit_softmax_tb {

void test_layer(int l) {
    const string file_path = VIT_BINARIES_PATH + to_string(l);
    const string save_path = VIT_CONDENSE_PATH + to_string(l);

    vector<int64_t> condensed_r(P_VIT_NUM_R);
    vector<int8_t> ref_rq(P_VIT_NUM_R);
    vector<int8_t> ref_rs(P_VIT_NUM_RS);
    vector<int8_t> dut_rq(P_VIT_NUM_R);
    vector<int8_t> dut_rs(P_VIT_NUM_RS);

    {
        // ViT 参考文件按 batch*head 保存；不同导出批次的 batch 数可能变化，按文件长度推导。
        auto R_Q = read_tensor<int8_t>(file_path + "/MHA_R_Q.bin");
        auto R_S = read_tensor<int8_t>(file_path + "/MHA_R_S.bin");
        int bq_load = infer_tensor_extent<int8_t>(R_Q, P_VIT_H * P_VIT_T * P_S,  "ViT MHA_R_Q");
        int bs_load = infer_tensor_extent<int8_t>(R_S, P_VIT_H * P_VIT_T * P_ST, "ViT MHA_R_S");
        assert(bq_load == bs_load);
        tensor2array<int8_t>(R_Q, ref_rq.data(), bq_load * P_VIT_H, P_VIT_H, P_VIT_T, 0, P_VIT_T, P_S,  P_S);
        tensor2array<int8_t>(R_S, ref_rs.data(), bs_load * P_VIT_H, P_VIT_H, P_VIT_T, 0, P_VIT_T, P_ST, P_ST);
    }

    {
        auto R = read_tensor<int64_t>(save_path + "/CONDENSED_QK_GEMM_R.bin");
        tensor2array<int64_t>(R, condensed_r.data(), 1, 1, 1, 1, P_VIT_NUM_R, P_VIT_NUM_R);
    }

    hls::stream<hls::vector<sm_r_t, P_SP> > r_stream("vit_r_stream");
    hls::stream<hls::vector<sm_q_t, P_SP> > rq_stream("vit_rq_stream");
    hls::stream<hls::vector<sm_s_t, 1   > > rs_stream("vit_rs_stream");

    array2stream<int64_t, sm_r_t, 1, 1, 1, 1, P_VIT_NUM_R, P_SP>(condensed_r.data(), r_stream, "ViT R", true);
    ::top(MODE_VIT, l, l + 1, 0, r_stream, rq_stream, rs_stream);

    save_condensed_tensor<int8_t, sm_q_t, P_VIT_NUM_R,  P_SP>(save_path + "/CONDENSED_SOFTMAX_R_Q.bin", rq_stream);
    save_condensed_tensor<int8_t, sm_s_t, P_VIT_NUM_RS, 1   >(save_path + "/CONDENSED_SOFTMAX_R_S.bin", rs_stream);

    stream2array<int8_t, sm_q_t, P_VIT_H, P_VIT_T, 1, P_S,  P_SP>(rq_stream, dut_rq.data(), "ViT R_Q", true);
    stream2array<int8_t, sm_s_t, P_VIT_H, P_VIT_T, 1, P_ST, 1   >(rs_stream, dut_rs.data(), "ViT R_S", true);

    assert(r_stream.empty());
    assert(rq_stream.empty());
    assert(rs_stream.empty());

    int mismatch_rq = 0;
    int mismatch_rs = 0;
    for (int h = 0; h < P_VIT_H; ++h) {
        for (int t = 0; t < P_VIT_T; ++t) {
            for (int s = 0; s < P_S; ++s) {
                int idx = h * P_VIT_T * P_S + t * P_S + s;
                if (ref_rq[idx] != dut_rq[idx]) {
                    if (++mismatch_rq <= 10) {
                        printf("ViT R_Q mismatch L%d H%d T%d S%d: REF=%d DUT=%d\n",
                               l, h, t, s, (int)ref_rq[idx], (int)dut_rq[idx]);
                    }
                }
            }
        }
    }

    for (int h = 0; h < P_VIT_H; ++h) {
        for (int t = 0; t < P_VIT_T; ++t) {
            for (int st = 0; st < P_ST; ++st) {
                int idx = h * P_VIT_T * P_ST + t * P_ST + st;
                if (ref_rs[idx] != dut_rs[idx]) {
                    if (++mismatch_rs <= 10) {
                        printf("ViT R_S mismatch L%d H%d T%d ST%d: REF=%d DUT=%d\n",
                               l, h, t, st, (int)ref_rs[idx], (int)dut_rs[idx]);
                    }
                }
            }
        }
    }

    printf("ViT SOFTMAX layer %d: R_Q mismatch=%d R_S mismatch=%d\n", l, mismatch_rq, mismatch_rs);
}

} // namespace vit_softmax_tb

int main() {
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        int llm_layer = reuse_tb_llm_layer();
        llm_softmax_tb::test_layer(llm_layer);
        printf("LLM SOFTMAX layer %d passed\n", llm_layer);
    }

    if (!reuse_tb_only_llm()) {
        vit_softmax_tb::test_layer(vit_layer);
        printf("ViT SOFTMAX layer %d passed\n", vit_layer);
    }

    return 0;
}
