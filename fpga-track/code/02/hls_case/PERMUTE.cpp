#include "../src/reuse_common.h"
#include "../src/utils.h"

// ============================================================================
// LLM/ViT 复用 GEMM_PERMUTE
// ============================================================================
// 功能：
//   1. 把 LLM decoder/CLS 和 ViT encoder 的线性投影 GEMM 合并成一个物理 IP。
//   2. 顶层通过 REUSE_MODE_T mode 选择当前运行 LLM 还是 ViT；mode 只在 layer 调用粒度生效。
//   3. 内部始终复用同一套 8x8 tensor core 和同一套 accumulator，避免综合出两套大 datapath。
//   4. testbench 同时覆盖 LLM decoder layer 0、LLM CLS/lm_head 和 ViT vision layer 0。
//
// 数据边界：
//   - 输入激活 i_stream 每拍 TP*CIP = 8*8 = 64 个 int8。
//   - 权重 w_stream 每拍 COP*CIP = 8*8 = 64 个 int5。
//   - scale 流 s/s1/s2 分别承载 activation scale、weight shift1、weight shift2。
//   - 输出 o_stream 是 accumulator 基础截断后的 GEMM_TRUNC 数据，每拍 COP=8 个元素。
//
// 注意：
//   LLM 与 ViT 的权重/激活/输出文件由不同目录隔离，因此 ViT condense 文件不再额外添加
//   ViT 特殊前缀，统一使用 CONDENSED_GEMM_* 等算子语义文件名。

constexpr int P_TP      = REUSE_GEMM_TP;
constexpr int P_CIP     = REUSE_CP;
constexpr int P_COP     = REUSE_CP;
constexpr int P_LANES   = P_TP * P_CIP;
constexpr int P_TC_GROUPS = 4;
constexpr int P_TC_COP    = P_COP / P_TC_GROUPS;

static_assert(P_CIP == P_COP, "PERMUTE assumes CIP == COP == 8");
static_assert(P_COP % P_TC_GROUPS == 0, "PERMUTE tensor-core groups must divide COP");
static_assert(LLAMA_TRUNC_BASE == VIT_TRUNC_BASE, "PERMUTE base trunc must match across modes");
constexpr int P_TRUNC_BASE = LLAMA_TRUNC_BASE;

typedef REUSE_AQ_T         aq_t;
typedef REUSE_AS_T         as_t;
typedef REUSE_WQ_T         wq_t;
typedef REUSE_WS_T         ws_t;
typedef REUSE_GEMM_T       gemm_t;
typedef REUSE_GEMM_ACC_T   acc_t;
typedef REUSE_GEMM_TRUNC_T of_t;
typedef ap_uint<24>        cycle_t;
typedef ap_uint<10>        acc_cycle_t;
typedef ap_uint<4>         emit_idx_t;
typedef hls::vector<aq_t,   P_TP * P_CIP>     tc_i_vec_t;
typedef hls::vector<wq_t,   P_COP * P_CIP>    tc_w_vec_t;
typedef hls::vector<wq_t,   P_TC_COP * P_CIP> tc_w_group_vec_t;
typedef hls::vector<gemm_t, P_TP * P_TC_COP>  tc_gemm_group_vec_t;
typedef hls::vector<gemm_t, P_TP * P_COP>     tc_gemm_vec_t;
typedef hls::vector<as_t,   P_TP>             tc_s_vec_t;
typedef hls::vector<ws_t,   P_COP>            tc_ws_vec_t;
typedef hls::vector<ws_t,   P_TC_COP>         tc_ws_group_vec_t;
typedef hls::vector<of_t,   P_TC_COP>         tc_of_group_vec_t;
typedef hls::vector<of_t,   P_COP>            tc_of_vec_t;

// LLM 参数：
//   decoder 普通层包含 native-GQA compact K/Q/V/O、U/G、D；CLS 层只跑 lm_head。
//   LLM decode 的硬件 token tile 固定为 8，等价于 TT=1。
constexpr int P_LLM_T       = REUSE_LLM_TILE_T;
// LLM testbench 的 binaries 可能随上游软件导出 token 数变化；硬件只处理当前 8-token tile，
// 因此 CSim 中按文件大小推导 T_LOAD，避免固定历史长度导致仿真失效。
constexpr int P_LLM_POS     = 96;
constexpr int P_LLM_H       = LLAMA_H;
constexpr int P_LLM_KVH     = LLAMA_KVH;
constexpr int P_LLM_GQA     = P_LLM_H / P_LLM_KVH;
constexpr int P_LLM_C       = LLAMA_C;
constexpr int P_LLM_HC      = LLAMA_HC;
constexpr int P_LLM_CM      = LLAMA_CM;
constexpr int P_LLM_VOCAB   = LLAMA_VOCAB;
constexpr int P_LLM_CT      = P_LLM_C     / P_CIP;
constexpr int P_LLM_CMT     = P_LLM_CM    / P_CIP;
constexpr int P_LLM_VOCABT  = P_LLM_VOCAB / P_CIP;
constexpr int P_LLM_N_Q     = P_LLM_C  / P_COP;
constexpr int P_LLM_N_KV    = (P_LLM_KVH * P_LLM_HC) / P_COP;
constexpr int P_LLM_N_QKV   = P_LLM_N_KV + P_LLM_N_Q + P_LLM_N_KV;
constexpr int P_LLM_N_O     = P_LLM_C  / P_COP;
constexpr int P_LLM_N_UG    = P_LLM_CM / P_COP;
constexpr int P_LLM_N_D     = P_LLM_C  / P_COP;
constexpr int P_LLM_N_CLS   = P_LLM_VOCAB / P_COP;
constexpr int P_LLM_ACC_CYCS1   = P_LLM_C  / P_CIP;
constexpr int P_LLM_ACC_CYCS2   = P_LLM_CM / P_CIP;
constexpr int P_LLM_ACC_CYCS3   = P_LLM_C  / P_CIP;
constexpr int P_LLM_ACC_REPEAT1 = P_LLM_N_QKV + P_LLM_N_O + P_LLM_N_UG * 2;
constexpr int P_LLM_ACC_REPEAT2 = P_LLM_N_D;
constexpr int P_LLM_ACC_REPEAT3 = P_LLM_N_CLS;
constexpr int P_LLM_DECODER_TOTAL_CYCS =
    P_LLM_ACC_REPEAT1 * P_LLM_ACC_CYCS1 + P_LLM_ACC_REPEAT2 * P_LLM_ACC_CYCS2;
constexpr int P_LLM_DECODER_BOUND_CYCS = P_LLM_ACC_REPEAT1 * P_LLM_ACC_CYCS1;
constexpr int P_LLM_CLS_TOTAL_CYCS     = P_LLM_ACC_REPEAT3 * P_LLM_ACC_CYCS3;

// ViT 参数：
//   encoder 普通层包含 Q/K/V/O、FC1、FC2 共 6 个矩阵。
//   ViT 一层有 1024 token，因此 testbench 与硬件循环都要保留 TT=1024/8=128。
constexpr int P_VIT_T       = VIT_S;
// ViT 新导出按 batch*patch 保存，当前 vision_0/new_txt 与其他层同批为 13*1024。
constexpr int P_VIT_T_LOAD  = 13312;
constexpr int P_VIT_POS     = 0;
constexpr int P_VIT_H       = VIT_H;
constexpr int P_VIT_C       = VIT_C;
constexpr int P_VIT_HC      = VIT_HC;
constexpr int P_VIT_CM      = VIT_MLP_DIM;
constexpr int P_VIT_TT      = P_VIT_T / P_TP;
constexpr int P_VIT_CT      = P_VIT_C  / P_CIP;
constexpr int P_VIT_CMT     = P_VIT_CM / P_CIP;
constexpr int P_VIT_HCT     = P_VIT_HC / P_CIP;
constexpr int P_VIT_N_QKVO  = P_VIT_C  / P_COP;
constexpr int P_VIT_N_FC1   = P_VIT_CM / P_COP;
constexpr int P_VIT_N_FC2   = P_VIT_C  / P_COP;
constexpr int P_VIT_ACC_CYCS1   = P_VIT_C  / P_CIP;
constexpr int P_VIT_ACC_CYCS2   = P_VIT_CM / P_CIP;
constexpr int P_VIT_ACC_REPEAT1 = P_VIT_TT * (P_VIT_N_QKVO * 4 + P_VIT_N_FC1);
constexpr int P_VIT_ACC_REPEAT2 = P_VIT_TT * P_VIT_N_FC2;
constexpr int P_VIT_TOTAL_CYCS =
    P_VIT_ACC_REPEAT1 * P_VIT_ACC_CYCS1 + P_VIT_ACC_REPEAT2 * P_VIT_ACC_CYCS2;
constexpr int P_VIT_BOUND_CYCS = P_VIT_ACC_REPEAT1 * P_VIT_ACC_CYCS1;

class REUSE_PERMUTE_TENSOR_CORE {
public:
    // 共享 tensor core：
    //   每次读取一个 8-token x 8-channel 激活 tile 和一个 8-output x 8-channel 权重 tile。
    //   当前为了给最终 Vivado 时序换余量，TP/COP/CIP 三维全部展开成 8x8x8=512 个乘法器；
    //   DSP 资源仍低于 KV260 目标上限，换取 tensor core 每拍输出一个完整 8x8 partial-sum tile。
    //   total_iters 由外层 mode 配置给出，LLM decoder、LLM CLS、ViT encoder 只改变循环次数。
    void dispatch_tensor_inputs(
        int total_iters,
        hls::stream<tc_i_vec_t>& i_stream,
        hls::stream<tc_w_vec_t>& w_stream,
        hls::stream<tc_i_vec_t> i_group_stream[P_TC_GROUPS],
        hls::stream<tc_w_group_vec_t> w_group_stream[P_TC_GROUPS]
    ) {
        #pragma HLS inline off
        #pragma HLS array_partition variable=i_group_stream complete dim=1
        #pragma HLS array_partition variable=w_group_stream complete dim=1
        cycle_t total_iters_u = total_iters;
        for (cycle_t n = 0; n < total_iters_u; ++n) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=172800 max=14155776

            tc_i_vec_t i_vec = i_stream.read();
            tc_w_vec_t w_vec = w_stream.read();

            for (int group = 0; group < P_TC_GROUPS; ++group) {
                #pragma HLS unroll
                tc_w_group_vec_t w_group_vec;
                for (int cop = 0; cop < P_TC_COP; ++cop) {
                    for (int cip = 0; cip < P_CIP; ++cip) {
                        #pragma HLS unroll
                        int src_cop = group * P_TC_COP + cop;
                        w_group_vec[cop * P_CIP + cip] = w_vec[src_cop * P_CIP + cip];
                    }
                }
                i_group_stream[group].write(i_vec);
                w_group_stream[group].write(w_group_vec);
            }
        }
    }

    template<int GROUP_ID>
    void do_tensor_core_group(
        int total_iters,
        hls::stream<tc_i_vec_t>& i_stream,
        hls::stream<tc_w_group_vec_t>& w_stream,
        hls::stream<tc_gemm_group_vec_t>& o_stream
    ) {
        #pragma HLS inline off
        cycle_t total_iters_u = total_iters;
        for (cycle_t n = 0; n < total_iters_u; ++n) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=172800 max=14155776

            tc_i_vec_t i_vec = i_stream.read();
            tc_w_group_vec_t w_vec = w_stream.read();
            tc_gemm_group_vec_t o_vec;

            for (int tp = 0; tp < P_TP; ++tp) {
                for (int cop = 0; cop < P_TC_COP; ++cop) {
                    #pragma HLS unroll
                    o_vec[tp * P_TC_COP + cop] = 0;
                }
            }

            // 每个 group 只负责 COP 的一个小块，仍保持全展开数学结构。
            // 这样 512 个 DSP 被拆成 4 个独立控制域，降低单个 loop-enable/CE 网扇出。
            for (int tp = 0; tp < P_TP; ++tp) {
                for (int cop = 0; cop < P_TC_COP; ++cop) {
                    for (int cip = 0; cip < P_CIP; ++cip) {
                        #pragma HLS unroll
                        auto mul_res = i_vec[tp * P_CIP + cip] * w_vec[cop * P_CIP + cip];
                        #pragma HLS bind_op variable=mul_res op=mul impl=dsp
                        o_vec[tp * P_TC_COP + cop] += mul_res;
                    }
                }
            }

            o_stream.write(o_vec);
        }
    }

    void merge_tensor_groups(
        int total_iters,
        hls::stream<tc_gemm_group_vec_t> group_stream[P_TC_GROUPS],
        hls::stream<tc_gemm_vec_t>& o_stream
    ) {
        #pragma HLS inline off
        #pragma HLS array_partition variable=group_stream complete dim=1
        cycle_t total_iters_u = total_iters;
        for (cycle_t n = 0; n < total_iters_u; ++n) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=172800 max=14155776

            tc_gemm_vec_t o_vec;
            for (int group = 0; group < P_TC_GROUPS; ++group) {
                #pragma HLS unroll
                tc_gemm_group_vec_t group_vec = group_stream[group].read();
                for (int tp = 0; tp < P_TP; ++tp) {
                    for (int cop = 0; cop < P_TC_COP; ++cop) {
                        #pragma HLS unroll
                        int dst_cop = group * P_TC_COP + cop;
                        o_vec[tp * P_COP + dst_cop] = group_vec[tp * P_TC_COP + cop];
                    }
                }
            }
            o_stream.write(o_vec);
        }
    }

    void do_tensor_core(
        int total_iters,
        hls::stream<tc_i_vec_t>& i_stream,
        hls::stream<tc_w_vec_t>& w_stream,
        hls::stream<tc_gemm_vec_t>& o_stream
    ) {
        #pragma HLS dataflow

        hls::stream<tc_i_vec_t> i_group_stream[P_TC_GROUPS];
        hls::stream<tc_w_group_vec_t> w_group_stream[P_TC_GROUPS];
        hls::stream<tc_gemm_group_vec_t> gemm_group_stream[P_TC_GROUPS];
        #pragma HLS array_partition variable=i_group_stream complete dim=1
        #pragma HLS array_partition variable=w_group_stream complete dim=1
        #pragma HLS array_partition variable=gemm_group_stream complete dim=1
        #pragma HLS stream variable=i_group_stream depth=2
        #pragma HLS stream variable=w_group_stream depth=2
        #pragma HLS stream variable=gemm_group_stream depth=2
        // Tensor-core group dataflow FIFOs are very wide and used as physical
        // boundaries between dispatch, DSP groups and accumulator. Use
        // block-style wrappers to avoid mapping them into long SRL/LUTRAM paths.
        #pragma HLS bind_storage variable=i_group_stream type=fifo impl=bram
        #pragma HLS bind_storage variable=w_group_stream type=fifo impl=bram
        #pragma HLS bind_storage variable=gemm_group_stream type=fifo impl=bram

        dispatch_tensor_inputs(total_iters, i_stream, w_stream, i_group_stream, w_group_stream);
        do_tensor_core_group<0>(total_iters, i_group_stream[0], w_group_stream[0], gemm_group_stream[0]);
        do_tensor_core_group<1>(total_iters, i_group_stream[1], w_group_stream[1], gemm_group_stream[1]);
        do_tensor_core_group<2>(total_iters, i_group_stream[2], w_group_stream[2], gemm_group_stream[2]);
        do_tensor_core_group<3>(total_iters, i_group_stream[3], w_group_stream[3], gemm_group_stream[3]);
        merge_tensor_groups(total_iters, gemm_group_stream, o_stream);
    }
};

class REUSE_PERMUTE_ACCUMULATOR {
public:
    // 累加器：
    //   对 tensor core 输出的 partial sum 做跨输入通道累加，并把 activation scale 和 weight shift
    //   合并到同一条流水线中。acc_cycs1/acc_cycs2 描述不同矩阵输入通道数：
    //     - LLM/ViT QKVO/FC1/UG 使用 hidden C。
    //     - LLM D 和 ViT FC2 使用 MLP hidden CM。
    //     - LLM CLS 使用 vocab 输出但输入通道仍是 C，由 run_cls 配置映射到 acc_cycs1。
    //   输出边界改为直接按 token 发 8 拍 COP=8 窄包，避免旧版两个 960-bit 半包 FIFO
    //   在完整 Vivado 顶层里形成高扇出、长路由的 post-route 最差路径。
    void accumulate_and_emit(
        int total_cycs,
        int bound_cycs,
        int acc_cycs1,
        int acc_cycs2,
        int num_output,
        hls::stream<hls::vector<gemm_t, P_TP * P_COP> >& i_stream,
        hls::stream<hls::vector<as_t,   P_TP        > >& s_stream,
        hls::stream<hls::vector<ws_t,   P_COP       > >& s1_stream,
        hls::stream<hls::vector<ws_t,   P_COP       > >& s2_stream,
        hls::stream<hls::vector<of_t,   P_COP       > >& o_stream
    ) {
        acc_t psum[P_LANES];
        #pragma HLS array_reshape variable=psum complete dim=1
        hls::vector<of_t, P_COP> emit_vec[P_TP];
        #pragma HLS array_partition variable=emit_vec complete dim=1

        cycle_t total_cycs_u = total_cycs;
        cycle_t bound_cycs_u = bound_cycs;
        acc_cycle_t acc_cycs1_u = acc_cycs1;
        acc_cycle_t acc_cycs2_u = acc_cycs2;
        cycle_t total_steps = total_cycs_u + cycle_t(num_output * P_TP);
        cycle_t cyc = 0;
        acc_cycle_t sub_cyc = 0;
        emit_idx_t emit_tp = P_TP;
        for (cycle_t step = 0; step < total_steps; ++step) {
            #pragma HLS pipeline II=1
            #pragma HLS EXPRESSION_BALANCE OFF
            #pragma HLS loop_tripcount min=182720 max=15040512

            if (emit_tp < P_TP) {
                hls::vector<of_t, P_COP> o_vec;
                // 输出重排固定为 8 个 token buffer，避免 emit_tp*COP 动态索引被 HLS 展开成
                // 1920-bit barrel-shift/大 mux；每拍只选择一个 240-bit token 向下游发送。
                switch (emit_tp) {
                    case 0: o_vec = emit_vec[0]; break;
                    case 1: o_vec = emit_vec[1]; break;
                    case 2: o_vec = emit_vec[2]; break;
                    case 3: o_vec = emit_vec[3]; break;
                    case 4: o_vec = emit_vec[4]; break;
                    case 5: o_vec = emit_vec[5]; break;
                    case 6: o_vec = emit_vec[6]; break;
                    default: o_vec = emit_vec[7]; break;
                }
                o_stream.write(o_vec);
                ++emit_tp;
            } else if (cyc < total_cycs_u) {
                if (sub_cyc == 0) {
                    for (int lane = 0; lane < P_LANES; ++lane) {
                        #pragma HLS unroll
                        psum[lane] = 0;
                    }
                }

                hls::vector<gemm_t, P_TP * P_COP> i_vec  = i_stream.read();
                hls::vector<as_t,   P_TP        > s_vec  = s_stream.read();
                hls::vector<ws_t,   P_COP       > s1_vec = s1_stream.read();
                hls::vector<ws_t,   P_COP       > s2_vec = s2_stream.read();

                constexpr int s2_width = gemm_t::width + (1 << (as_t::width + 1)) - 1;
                typedef ap_int<s2_width> gemm_s2_t;
                typedef ap_uint<as_t::width + ws_t::width> acc_shift_t;

                for (int tp = 0; tp < P_TP; ++tp) {
                    for (int cp = 0; cp < P_COP; ++cp) {
                        #pragma HLS unroll
                        auto x_val   = i_vec [tp * P_COP + cp];
                        auto s_val   = s_vec [tp];
                        auto s1_val  = s1_vec[cp];
                        auto s2_val  = s2_vec[cp];
                        acc_shift_t shift1 = acc_shift_t(s_val) + acc_shift_t(s1_val);
                        acc_shift_t shift2 = acc_shift_t(s_val) + acc_shift_t(s2_val);
                        // 合成 activation/weight shift，避免 64 lane 共享 x_shift 中间网跨区路由。
                        psum[tp * P_COP + cp] += (gemm_s2_t(x_val) << shift1) + (gemm_s2_t(x_val) << shift2);
                    }
                }

                acc_cycle_t sub_cycs = (cyc < bound_cycs_u) ? acc_cycs1_u : acc_cycs2_u;
                acc_cycle_t next_sub_cyc = sub_cyc + 1;
                if (next_sub_cyc == sub_cycs) {
                    sub_cyc = 0;
                    for (int tp = 0; tp < P_TP; ++tp) {
                        for (int cp = 0; cp < P_COP; ++cp) {
                            #pragma HLS unroll
                            emit_vec[tp][cp] = psum[tp * P_COP + cp] >> P_TRUNC_BASE;
                        }
                    }
                    emit_tp = 0;
                } else {
                    sub_cyc = next_sub_cyc;
                }
                ++cyc;
            }
        }
    }

    void do_accumulator(
        int total_cycs,
        int bound_cycs,
        int acc_cycs1,
        int acc_cycs2,
        int num_output,
        hls::stream<hls::vector<gemm_t, P_TP * P_COP> >& i_stream,
        hls::stream<hls::vector<as_t,   P_TP        > >& s_stream,
        hls::stream<hls::vector<ws_t,   P_COP       > >& s1_stream,
        hls::stream<hls::vector<ws_t,   P_COP       > >& s2_stream,
        hls::stream<hls::vector<of_t,   P_COP       > >& o_stream
    ) {
        // accumulator 内部不再用 960-bit 宽 dataflow FIFO；窄包直接输出给下游，
        // 代价是每个 8x8 tile 完成后暂停输入 8 拍做 token 顺序输出。
        accumulate_and_emit(total_cycs, bound_cycs, acc_cycs1, acc_cycs2, num_output,
                            i_stream, s_stream, s1_stream, s2_stream, o_stream);
    }

    void dispatch_accumulator_scales(
        int total_cycs,
        hls::stream<tc_s_vec_t>& s_stream,
        hls::stream<tc_ws_vec_t>& s1_stream,
        hls::stream<tc_ws_vec_t>& s2_stream,
        hls::stream<tc_s_vec_t> s_group_stream[P_TC_GROUPS],
        hls::stream<tc_ws_group_vec_t> s1_group_stream[P_TC_GROUPS],
        hls::stream<tc_ws_group_vec_t> s2_group_stream[P_TC_GROUPS]
    ) {
        #pragma HLS inline off
        #pragma HLS array_partition variable=s_group_stream complete dim=1
        #pragma HLS array_partition variable=s1_group_stream complete dim=1
        #pragma HLS array_partition variable=s2_group_stream complete dim=1
        cycle_t total_cycs_u = total_cycs;
        for (cycle_t cyc = 0; cyc < total_cycs_u; ++cyc) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=172800 max=14155776

            tc_s_vec_t s_vec = s_stream.read();
            tc_ws_vec_t s1_vec = s1_stream.read();
            tc_ws_vec_t s2_vec = s2_stream.read();

            for (int group = 0; group < P_TC_GROUPS; ++group) {
                #pragma HLS unroll
                tc_ws_group_vec_t s1_group_vec;
                tc_ws_group_vec_t s2_group_vec;
                for (int cp = 0; cp < P_TC_COP; ++cp) {
                    #pragma HLS unroll
                    int src_cp = group * P_TC_COP + cp;
                    s1_group_vec[cp] = s1_vec[src_cp];
                    s2_group_vec[cp] = s2_vec[src_cp];
                }
                s_group_stream[group].write(s_vec);
                s1_group_stream[group].write(s1_group_vec);
                s2_group_stream[group].write(s2_group_vec);
            }
        }
    }

    template<int GROUP_ID>
    void accumulate_and_emit_group(
        int total_cycs,
        int bound_cycs,
        int acc_cycs1,
        int acc_cycs2,
        int num_output,
        hls::stream<tc_gemm_group_vec_t>& i_stream,
        hls::stream<tc_s_vec_t>& s_stream,
        hls::stream<tc_ws_group_vec_t>& s1_stream,
        hls::stream<tc_ws_group_vec_t>& s2_stream,
        hls::stream<tc_of_group_vec_t>& o_stream
    ) {
        #pragma HLS inline off
        acc_t psum[P_TP * P_TC_COP];
        #pragma HLS array_reshape variable=psum complete dim=1
        tc_of_group_vec_t emit_vec[P_TP];
        #pragma HLS array_partition variable=emit_vec complete dim=1

        cycle_t total_cycs_u = total_cycs;
        cycle_t bound_cycs_u = bound_cycs;
        acc_cycle_t acc_cycs1_u = acc_cycs1;
        acc_cycle_t acc_cycs2_u = acc_cycs2;
        cycle_t total_steps = total_cycs_u + cycle_t(num_output * P_TP);
        cycle_t cyc = 0;
        acc_cycle_t sub_cyc = 0;
        emit_idx_t emit_tp = P_TP;
        for (cycle_t step = 0; step < total_steps; ++step) {
            #pragma HLS pipeline II=1
            #pragma HLS EXPRESSION_BALANCE OFF
            #pragma HLS loop_tripcount min=182720 max=15040512

            if (emit_tp < P_TP) {
                tc_of_group_vec_t o_vec;
                // 每个 group 只发 2 个 COP lane，merge 阶段再恢复外部 COP=8 格式。
                switch (emit_tp) {
                    case 0: o_vec = emit_vec[0]; break;
                    case 1: o_vec = emit_vec[1]; break;
                    case 2: o_vec = emit_vec[2]; break;
                    case 3: o_vec = emit_vec[3]; break;
                    case 4: o_vec = emit_vec[4]; break;
                    case 5: o_vec = emit_vec[5]; break;
                    case 6: o_vec = emit_vec[6]; break;
                    default: o_vec = emit_vec[7]; break;
                }
                o_stream.write(o_vec);
                ++emit_tp;
            } else if (cyc < total_cycs_u) {
                if (sub_cyc == 0) {
                    for (int lane = 0; lane < P_TP * P_TC_COP; ++lane) {
                        #pragma HLS unroll
                        psum[lane] = 0;
                    }
                }

                tc_gemm_group_vec_t i_vec = i_stream.read();
                tc_s_vec_t s_vec = s_stream.read();
                tc_ws_group_vec_t s1_vec = s1_stream.read();
                tc_ws_group_vec_t s2_vec = s2_stream.read();

                constexpr int s2_width = gemm_t::width + (1 << (as_t::width + 1)) - 1;
                typedef ap_int<s2_width> gemm_s2_t;
                typedef ap_uint<as_t::width + ws_t::width> acc_shift_t;

                for (int tp = 0; tp < P_TP; ++tp) {
                    for (int cp = 0; cp < P_TC_COP; ++cp) {
                        #pragma HLS unroll
                        auto x_val = i_vec[tp * P_TC_COP + cp];
                        auto s_val = s_vec[tp];
                        auto s1_val = s1_vec[cp];
                        auto s2_val = s2_vec[cp];
                        acc_shift_t shift1 = acc_shift_t(s_val) + acc_shift_t(s1_val);
                        acc_shift_t shift2 = acc_shift_t(s_val) + acc_shift_t(s2_val);
                        psum[tp * P_TC_COP + cp] += (gemm_s2_t(x_val) << shift1) + (gemm_s2_t(x_val) << shift2);
                    }
                }

                acc_cycle_t sub_cycs = (cyc < bound_cycs_u) ? acc_cycs1_u : acc_cycs2_u;
                acc_cycle_t next_sub_cyc = sub_cyc + 1;
                if (next_sub_cyc == sub_cycs) {
                    sub_cyc = 0;
                    for (int tp = 0; tp < P_TP; ++tp) {
                        for (int cp = 0; cp < P_TC_COP; ++cp) {
                            #pragma HLS unroll
                            emit_vec[tp][cp] = psum[tp * P_TC_COP + cp] >> P_TRUNC_BASE;
                        }
                    }
                    emit_tp = 0;
                } else {
                    sub_cyc = next_sub_cyc;
                }
                ++cyc;
            }
        }
    }

    void merge_output_groups(
        int num_output,
        hls::stream<tc_of_group_vec_t> group_stream[P_TC_GROUPS],
        hls::stream<tc_of_vec_t>& o_stream
    ) {
        #pragma HLS inline off
        #pragma HLS array_partition variable=group_stream complete dim=1
        cycle_t total_outputs = cycle_t(num_output * P_TP);
        for (cycle_t idx = 0; idx < total_outputs; ++idx) {
            #pragma HLS pipeline II=1
            #pragma HLS loop_tripcount min=9920 max=848736

            tc_of_vec_t o_vec;
            for (int group = 0; group < P_TC_GROUPS; ++group) {
                #pragma HLS unroll
                tc_of_group_vec_t group_vec = group_stream[group].read();
                for (int cp = 0; cp < P_TC_COP; ++cp) {
                    #pragma HLS unroll
                    int dst_cp = group * P_TC_COP + cp;
                    o_vec[dst_cp] = group_vec[cp];
                }
            }
            o_stream.write(o_vec);
        }
    }
};

REUSE_PERMUTE_TENSOR_CORE reuse_tensor_core_inst;
REUSE_PERMUTE_ACCUMULATOR reuse_accumulator_inst;

// 根据 mode 和 layer 类型生成本次 top_helper 调用所需的所有运行时配置。
// 这些配置只在 layer 粒度选择，不进入 TP/CIP/COP 的展开乘加内层，避免把 mode 分支放进关键路径。
void permute_config(
    REUSE_MODE_T mode,
    bool run_cls,
    int &total_cycs,
    int &bound_cycs,
    int &acc_cycs1,
    int &acc_cycs2,
    int &num_output
) {
    #pragma HLS inline
    if (is_vit_mode(mode)) {
        total_cycs = P_VIT_TOTAL_CYCS;
        bound_cycs = P_VIT_BOUND_CYCS;
        acc_cycs1 = P_VIT_ACC_CYCS1;
        acc_cycs2 = P_VIT_ACC_CYCS2;
        num_output = P_VIT_ACC_REPEAT1 + P_VIT_ACC_REPEAT2;
    } else if (run_cls) {
        total_cycs = P_LLM_CLS_TOTAL_CYCS;
        bound_cycs = P_LLM_CLS_TOTAL_CYCS;
        acc_cycs1 = P_LLM_ACC_CYCS3;
        acc_cycs2 = P_LLM_ACC_CYCS3;
        num_output = P_LLM_ACC_REPEAT3;
    } else {
        total_cycs = P_LLM_DECODER_TOTAL_CYCS;
        bound_cycs = P_LLM_DECODER_BOUND_CYCS;
        acc_cycs1 = P_LLM_ACC_CYCS1;
        acc_cycs2 = P_LLM_ACC_CYCS2;
        num_output = P_LLM_ACC_REPEAT1 + P_LLM_ACC_REPEAT2;
    }
}

void top_helper(
    int total_cycs,
    int bound_cycs,
    int acc_cycs1,
    int acc_cycs2,
    int num_output,
    hls::stream<hls::vector<aq_t, P_TP * P_CIP> >& i_stream,
    hls::stream<hls::vector<as_t, P_TP        > >& s_stream,
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> >& w_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s1_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s2_stream,
    hls::stream<hls::vector<of_t, P_COP       > >& o_stream
) {
    // top_helper 是一个 layer 级 dataflow 封装；tensor core 与 accumulator 都按 COP group 拆分。
    // 外部 AXIS 格式保持不变，内部把 512 DSP 和 64-lane accumulator 切成 4 个更局部的路由域。
    #pragma HLS dataflow
    hls::stream<tc_i_vec_t> i_group_stream[P_TC_GROUPS];
    hls::stream<tc_w_group_vec_t> w_group_stream[P_TC_GROUPS];
    hls::stream<tc_gemm_group_vec_t> gemm_group_stream[P_TC_GROUPS];
    hls::stream<tc_s_vec_t> s_group_stream[P_TC_GROUPS];
    hls::stream<tc_ws_group_vec_t> s1_group_stream[P_TC_GROUPS];
    hls::stream<tc_ws_group_vec_t> s2_group_stream[P_TC_GROUPS];
    hls::stream<tc_of_group_vec_t> o_group_stream[P_TC_GROUPS];
    #pragma HLS array_partition variable=i_group_stream complete dim=1
    #pragma HLS array_partition variable=w_group_stream complete dim=1
    #pragma HLS array_partition variable=gemm_group_stream complete dim=1
    #pragma HLS array_partition variable=s_group_stream complete dim=1
    #pragma HLS array_partition variable=s1_group_stream complete dim=1
    #pragma HLS array_partition variable=s2_group_stream complete dim=1
    #pragma HLS array_partition variable=o_group_stream complete dim=1
    #pragma HLS stream variable=i_group_stream depth=2
    #pragma HLS stream variable=w_group_stream depth=2
    #pragma HLS stream variable=gemm_group_stream depth=2
    // Tensor-core group dataflow FIFOs are very wide and feed/leave the 512-DSP
    // island. Keep them as block-style FIFO wrappers to reduce SRL/LUTRAM
    // packing pressure and provide cleaner physical cut points for routing.
    #pragma HLS bind_storage variable=i_group_stream type=fifo impl=bram
    #pragma HLS bind_storage variable=w_group_stream type=fifo impl=bram
    #pragma HLS bind_storage variable=gemm_group_stream type=fifo impl=bram
    #pragma HLS stream variable=s_group_stream depth=2
    #pragma HLS stream variable=s1_group_stream depth=2
    #pragma HLS stream variable=s2_group_stream depth=2
    #pragma HLS stream variable=o_group_stream depth=2

    reuse_tensor_core_inst.dispatch_tensor_inputs(total_cycs, i_stream, w_stream, i_group_stream, w_group_stream);
    reuse_tensor_core_inst.do_tensor_core_group<0>(total_cycs, i_group_stream[0], w_group_stream[0], gemm_group_stream[0]);
    reuse_tensor_core_inst.do_tensor_core_group<1>(total_cycs, i_group_stream[1], w_group_stream[1], gemm_group_stream[1]);
    reuse_tensor_core_inst.do_tensor_core_group<2>(total_cycs, i_group_stream[2], w_group_stream[2], gemm_group_stream[2]);
    reuse_tensor_core_inst.do_tensor_core_group<3>(total_cycs, i_group_stream[3], w_group_stream[3], gemm_group_stream[3]);
    reuse_accumulator_inst.dispatch_accumulator_scales(total_cycs, s_stream, s1_stream, s2_stream,
                                                       s_group_stream, s1_group_stream, s2_group_stream);
    reuse_accumulator_inst.accumulate_and_emit_group<0>(total_cycs, bound_cycs, acc_cycs1, acc_cycs2, num_output,
                                                        gemm_group_stream[0], s_group_stream[0], s1_group_stream[0],
                                                        s2_group_stream[0], o_group_stream[0]);
    reuse_accumulator_inst.accumulate_and_emit_group<1>(total_cycs, bound_cycs, acc_cycs1, acc_cycs2, num_output,
                                                        gemm_group_stream[1], s_group_stream[1], s1_group_stream[1],
                                                        s2_group_stream[1], o_group_stream[1]);
    reuse_accumulator_inst.accumulate_and_emit_group<2>(total_cycs, bound_cycs, acc_cycs1, acc_cycs2, num_output,
                                                        gemm_group_stream[2], s_group_stream[2], s1_group_stream[2],
                                                        s2_group_stream[2], o_group_stream[2]);
    reuse_accumulator_inst.accumulate_and_emit_group<3>(total_cycs, bound_cycs, acc_cycs1, acc_cycs2, num_output,
                                                        gemm_group_stream[3], s_group_stream[3], s1_group_stream[3],
                                                        s2_group_stream[3], o_group_stream[3]);
    reuse_accumulator_inst.merge_output_groups(num_output, o_group_stream, o_stream);
}

void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    hls::stream<hls::vector<aq_t, P_TP * P_CIP> >& i_stream,
    hls::stream<hls::vector<as_t, P_TP        > >& s_stream,
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> >& w_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s1_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s2_stream,
    hls::stream<hls::vector<of_t, P_COP       > >& o_stream
) {
    // 冻结后的复用顶层接口。mode 在一次 top 调用期间应保持稳定；
    // l_begin/l_close 是半开区间，LLM 允许 l==LLAMA_L 表示 CLS/lm_head，ViT 只允许 [0, VIT_L)。
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface axis port=i_stream
    #pragma HLS interface axis port=w_stream
    #pragma HLS interface axis port=s_stream
    #pragma HLS interface axis port=s1_stream
    #pragma HLS interface axis port=s2_stream
    #pragma HLS interface axis port=o_stream
    #pragma HLS aggregate variable=i_stream  compact=bit
    #pragma HLS aggregate variable=w_stream  compact=bit
    #pragma HLS aggregate variable=s_stream  compact=bit
    #pragma HLS aggregate variable=s1_stream compact=bit
    #pragma HLS aggregate variable=s2_stream compact=bit
    #pragma HLS aggregate variable=o_stream  compact=bit

    for (int l = l_begin; l < l_close; ++l) {
        #pragma HLS loop_tripcount min=1 max=33
        if (!reuse_is_valid_layer(mode, l, true)) {
            continue;
        }
        bool run_cls = reuse_is_llm_cls(mode, l);
        int total_cycs;
        int bound_cycs;
        int acc_cycs1;
        int acc_cycs2;
        int num_output;
        permute_config(mode, run_cls, total_cycs, bound_cycs, acc_cycs1, acc_cycs2, num_output);
        top_helper(total_cycs, bound_cycs, acc_cycs1, acc_cycs2, num_output,
                   i_stream, s_stream, w_stream, s1_stream, s2_stream, o_stream);
    }
}

// ============================================================================
// Testbench helpers
// ============================================================================
// testbench 分为 LLM 和 ViT 两个 namespace，便于分别保持原始独立模块的参考数据顺序。
// 两个 namespace 最终都调用同一个 ::top，只通过 MODE_LLM/MODE_VIT 区分路径。

namespace llm_permute_tb {

// 从完整序列文件中截取当前 decode tile。LLM step1 只验证 POS=96 开始的 8 个 token。
template<typename data_t>
int infer_llm_t_load(const vector<data_t> &tensor, int c, const string &name) {
    if (tensor.size() % c != 0) {
        std::cerr << name << " tensor size " << tensor.size() << " is not divisible by C=" << c << std::endl;
        exit(1);
    }
    return tensor.size() / c;
}

void slice_i8(vector<int8_t> &tensor, vector<int8_t> &array, int t_start, int t, int c, const string &name) {
    int t_load = infer_llm_t_load<int8_t>(tensor, c, name);
    tensor2array<int8_t>(tensor, array.data(), 1, 1, t_load, t_start, t, c, c);
}

void slice_i64(vector<int64_t> &tensor, vector<int64_t> &array, int t_start, int t, int c, const string &name) {
    int t_load = infer_llm_t_load<int64_t>(tensor, c, name);
    tensor2array<int64_t>(tensor, array.data(), 1, 1, t_load, t_start, t, c, c);
}

void prepare_decoder_data(
    int l,
    hls::stream<hls::vector<aq_t, P_TP * P_CIP> >& i_stream,
    hls::stream<hls::vector<as_t, P_TP        > >& s_stream,
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> >& w_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s1_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s2_stream
) {
    // LLM decoder 普通层输入顺序必须与原 LLM_PERMUTE 保持一致：
    //   激活：QKV 使用 MHA_XLN，O 使用 MHA_A，UG 使用 MLP_XLN，D 使用 MLP_XM。
    //   权重：Q/K/V 按 head-major 送入，UG 按输出 channel tile 交替送 U/G。
    //   保存：CONDENSED_GEMM_X/W_* 供 M_AXI/链式仿真检查输入边界。
    const string binaries_path = BINARIES_PATH + to_string(l);
    const string condense_path = CONDENSE_PATH + to_string(l);

    auto MHA_XLN_Q = read_tensor<int8_t>(binaries_path + "/MHA_XLN_Q.bin");
    auto MHA_XLN_S = read_tensor<int8_t>(binaries_path + "/MHA_XLN_S.bin");
    auto MHA_A_Q   = read_tensor<int8_t>(binaries_path + "/MHA_A_Q.bin");
    auto MHA_A_S   = read_tensor<int8_t>(binaries_path + "/MHA_A_S.bin");
    auto MLP_XLN_Q = read_tensor<int8_t>(binaries_path + "/MLP_XLN_Q.bin");
    auto MLP_XLN_S = read_tensor<int8_t>(binaries_path + "/MLP_XLN_S.bin");
    auto MLP_XM_Q  = read_tensor<int8_t>(binaries_path + "/MLP_XM_Q.bin");
    auto MLP_XM_S  = read_tensor<int8_t>(binaries_path + "/MLP_XM_S.bin");

    auto MHA_WQ_Q  = read_tensor<int8_t>(binaries_path + "/MHA_WQ_Q.bin");
    auto MHA_WQ_S1 = read_tensor<int8_t>(binaries_path + "/MHA_WQ_S1.bin");
    auto MHA_WQ_S2 = read_tensor<int8_t>(binaries_path + "/MHA_WQ_S2.bin");
    auto MHA_WK_Q  = read_tensor<int8_t>(binaries_path + "/MHA_WK_Q.bin");
    auto MHA_WK_S1 = read_tensor<int8_t>(binaries_path + "/MHA_WK_S1.bin");
    auto MHA_WK_S2 = read_tensor<int8_t>(binaries_path + "/MHA_WK_S2.bin");
    auto MHA_WV_Q  = read_tensor<int8_t>(binaries_path + "/MHA_WV_Q.bin");
    auto MHA_WV_S1 = read_tensor<int8_t>(binaries_path + "/MHA_WV_S1.bin");
    auto MHA_WV_S2 = read_tensor<int8_t>(binaries_path + "/MHA_WV_S2.bin");
    auto MHA_WO_Q  = read_tensor<int8_t>(binaries_path + "/MHA_WO_Q.bin");
    auto MHA_WO_S1 = read_tensor<int8_t>(binaries_path + "/MHA_WO_S1.bin");
    auto MHA_WO_S2 = read_tensor<int8_t>(binaries_path + "/MHA_WO_S2.bin");
    auto MLP_WU_Q  = read_tensor<int8_t>(binaries_path + "/MLP_WU_Q.bin");
    auto MLP_WU_S1 = read_tensor<int8_t>(binaries_path + "/MLP_WU_S1.bin");
    auto MLP_WU_S2 = read_tensor<int8_t>(binaries_path + "/MLP_WU_S2.bin");
    auto MLP_WG_Q  = read_tensor<int8_t>(binaries_path + "/MLP_WG_Q.bin");
    auto MLP_WG_S1 = read_tensor<int8_t>(binaries_path + "/MLP_WG_S1.bin");
    auto MLP_WG_S2 = read_tensor<int8_t>(binaries_path + "/MLP_WG_S2.bin");
    auto MLP_WD_Q  = read_tensor<int8_t>(binaries_path + "/MLP_WD_Q.bin");
    auto MLP_WD_S1 = read_tensor<int8_t>(binaries_path + "/MLP_WD_S1.bin");
    auto MLP_WD_S2 = read_tensor<int8_t>(binaries_path + "/MLP_WD_S2.bin");

    vector<int8_t> REF_MHA_XLN_Q(P_LLM_T * P_LLM_C);
    vector<int8_t> REF_MHA_XLN_S(P_LLM_T * P_LLM_CT);
    vector<int8_t> REF_MHA_A_Q  (P_LLM_T * P_LLM_C);
    vector<int8_t> REF_MHA_A_S  (P_LLM_T * P_LLM_CT);
    vector<int8_t> REF_MLP_XLN_Q(P_LLM_T * P_LLM_C);
    vector<int8_t> REF_MLP_XLN_S(P_LLM_T * P_LLM_CT);
    vector<int8_t> REF_MLP_XM_Q (P_LLM_T * P_LLM_CM);
    vector<int8_t> REF_MLP_XM_S (P_LLM_T * P_LLM_CMT);

    slice_i8(MHA_XLN_Q, REF_MHA_XLN_Q, P_LLM_POS, P_LLM_T, P_LLM_C,   "MHA_XLN_Q");
    slice_i8(MHA_XLN_S, REF_MHA_XLN_S, P_LLM_POS, P_LLM_T, P_LLM_CT,  "MHA_XLN_S");
    slice_i8(MHA_A_Q,   REF_MHA_A_Q,   P_LLM_POS, P_LLM_T, P_LLM_C,   "MHA_A_Q");
    slice_i8(MHA_A_S,   REF_MHA_A_S,   P_LLM_POS, P_LLM_T, P_LLM_CT,  "MHA_A_S");
    slice_i8(MLP_XLN_Q, REF_MLP_XLN_Q, P_LLM_POS, P_LLM_T, P_LLM_C,   "MLP_XLN_Q");
    slice_i8(MLP_XLN_S, REF_MLP_XLN_S, P_LLM_POS, P_LLM_T, P_LLM_CT,  "MLP_XLN_S");
    slice_i8(MLP_XM_Q,  REF_MLP_XM_Q,  P_LLM_POS, P_LLM_T, P_LLM_CM,  "MLP_XM_Q");
    slice_i8(MLP_XM_S,  REF_MLP_XM_S,  P_LLM_POS, P_LLM_T, P_LLM_CMT, "MLP_XM_S");

    hls::stream<hls::vector<aq_t, P_TP * P_CIP> > sim_i_stream("sim_i_stream");
    hls::stream<hls::vector<as_t, P_TP        > > sim_s_stream("sim_s_stream");
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> > sim_w_stream("sim_w_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > sim_s1_stream("sim_s1_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > sim_s2_stream("sim_s2_stream");

    array2stream<int8_t, aq_t, P_LLM_N_QKV, 1, P_LLM_T, P_LLM_T, P_LLM_C,  P_CIP>(REF_MHA_XLN_Q.data(), sim_i_stream, "LLM compQKV", true);
    array2stream<int8_t, aq_t, P_LLM_CT,     1, P_LLM_T, P_LLM_T, P_LLM_C,  P_CIP>(REF_MHA_A_Q.data(),   sim_i_stream, "LLM Input O",   true);
    array2stream<int8_t, aq_t, 2 * P_LLM_CMT,1, P_LLM_T, P_LLM_T, P_LLM_C,  P_CIP>(REF_MLP_XLN_Q.data(), sim_i_stream, "LLM Input UG",  true);
    array2stream<int8_t, aq_t, P_LLM_CT,     1, P_LLM_T, P_LLM_T, P_LLM_CM, P_CIP>(REF_MLP_XM_Q.data(),  sim_i_stream, "LLM Input D",   true);

    array2stream<int8_t, as_t, P_LLM_N_QKV, 1, P_LLM_T, P_LLM_T, P_LLM_CT,  1>(REF_MHA_XLN_S.data(), sim_s_stream, "LLM compQKVS", true);
    array2stream<int8_t, as_t, P_LLM_CT,     1, P_LLM_T, P_LLM_T, P_LLM_CT,  1>(REF_MHA_A_S.data(),   sim_s_stream, "LLM Input O S",   true);
    array2stream<int8_t, as_t, 2 * P_LLM_CMT,1, P_LLM_T, P_LLM_T, P_LLM_CT,  1>(REF_MLP_XLN_S.data(), sim_s_stream, "LLM Input UG S",  true);
    array2stream<int8_t, as_t, P_LLM_CT,     1, P_LLM_T, P_LLM_T, P_LLM_CMT, 1>(REF_MLP_XM_S.data(),  sim_s_stream, "LLM Input D S",   true);

    for (int kv = 0; kv < P_LLM_KVH; ++kv) {
        int kv_head = kv * P_LLM_GQA;
        array2stream<int8_t, wq_t, 1, 1, P_LLM_HC, P_COP, P_LLM_C,  P_CIP>(MHA_WK_Q.data()  + kv_head * P_LLM_HC * P_LLM_C,  sim_w_stream);
        array2stream<int8_t, ws_t, 1, 1, P_LLM_HC, P_COP, P_LLM_CT, 1    >(MHA_WK_S1.data() + kv_head * P_LLM_HC * P_LLM_CT, sim_s1_stream);
        array2stream<int8_t, ws_t, 1, 1, P_LLM_HC, P_COP, P_LLM_CT, 1    >(MHA_WK_S2.data() + kv_head * P_LLM_HC * P_LLM_CT, sim_s2_stream);
        array2stream<int8_t, wq_t, 1, 1, P_LLM_HC, P_COP, P_LLM_C,  P_CIP>(MHA_WV_Q.data()  + kv_head * P_LLM_HC * P_LLM_C,  sim_w_stream);
        array2stream<int8_t, ws_t, 1, 1, P_LLM_HC, P_COP, P_LLM_CT, 1    >(MHA_WV_S1.data() + kv_head * P_LLM_HC * P_LLM_CT, sim_s1_stream);
        array2stream<int8_t, ws_t, 1, 1, P_LLM_HC, P_COP, P_LLM_CT, 1    >(MHA_WV_S2.data() + kv_head * P_LLM_HC * P_LLM_CT, sim_s2_stream);
        for (int r = 0; r < P_LLM_GQA; ++r) {
            int q_head = kv * P_LLM_GQA + r;
            array2stream<int8_t, wq_t, 1, 1, P_LLM_HC, P_COP, P_LLM_C,  P_CIP>(MHA_WQ_Q.data()  + q_head * P_LLM_HC * P_LLM_C,  sim_w_stream);
            array2stream<int8_t, ws_t, 1, 1, P_LLM_HC, P_COP, P_LLM_CT, 1    >(MHA_WQ_S1.data() + q_head * P_LLM_HC * P_LLM_CT, sim_s1_stream);
            array2stream<int8_t, ws_t, 1, 1, P_LLM_HC, P_COP, P_LLM_CT, 1    >(MHA_WQ_S2.data() + q_head * P_LLM_HC * P_LLM_CT, sim_s2_stream);
        }
    }

    array2stream<int8_t, wq_t, 1, 1, P_LLM_C, P_COP, P_LLM_C,   P_CIP>(MHA_WO_Q.data(),  sim_w_stream);
    array2stream<int8_t, ws_t, 1, 1, P_LLM_C, P_COP, P_LLM_CT,  1    >(MHA_WO_S1.data(), sim_s1_stream);
    array2stream<int8_t, ws_t, 1, 1, P_LLM_C, P_COP, P_LLM_CT,  1    >(MHA_WO_S2.data(), sim_s2_stream);

    for (int cot = 0; cot < P_LLM_CMT; ++cot) {
        array2stream<int8_t, wq_t, 1, 1, P_COP, P_COP, P_LLM_C,  P_CIP>(MLP_WU_Q.data()  + cot * P_COP * P_LLM_C,  sim_w_stream);
        array2stream<int8_t, wq_t, 1, 1, P_COP, P_COP, P_LLM_C,  P_CIP>(MLP_WG_Q.data()  + cot * P_COP * P_LLM_C,  sim_w_stream);
        array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_LLM_CT, 1    >(MLP_WU_S1.data() + cot * P_COP * P_LLM_CT, sim_s1_stream);
        array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_LLM_CT, 1    >(MLP_WG_S1.data() + cot * P_COP * P_LLM_CT, sim_s1_stream);
        array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_LLM_CT, 1    >(MLP_WU_S2.data() + cot * P_COP * P_LLM_CT, sim_s2_stream);
        array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_LLM_CT, 1    >(MLP_WG_S2.data() + cot * P_COP * P_LLM_CT, sim_s2_stream);
    }

    array2stream<int8_t, wq_t, 1, 1, P_LLM_C, P_COP, P_LLM_CM,  P_CIP>(MLP_WD_Q.data(),  sim_w_stream);
    array2stream<int8_t, ws_t, 1, 1, P_LLM_C, P_COP, P_LLM_CMT, 1    >(MLP_WD_S1.data(), sim_s1_stream);
    array2stream<int8_t, ws_t, 1, 1, P_LLM_C, P_COP, P_LLM_CMT, 1    >(MLP_WD_S2.data(), sim_s2_stream);

    const int NUM_X  = P_LLM_N_QKV * P_LLM_T * P_LLM_C + P_LLM_CT * P_LLM_T * P_LLM_C +
                       2 * P_LLM_CMT * P_LLM_T * P_LLM_C + P_LLM_CT * P_LLM_T * P_LLM_CM;
    const int NUM_W  = (P_LLM_H + 2 * P_LLM_KVH) * P_LLM_HC * P_LLM_C + P_LLM_C * P_LLM_C +
                       2 * P_LLM_CM * P_LLM_C + P_LLM_C * P_LLM_CM;
    const int NUM_XS = NUM_X / P_CIP;
    const int NUM_WS = NUM_W / P_CIP;
    save_condensed_tensor<int8_t, aq_t, NUM_X,  P_TP * P_CIP>(condense_path + "/CONDENSED_GEMM_X_Q.bin",  sim_i_stream);
    save_condensed_tensor<int8_t, as_t, NUM_XS, P_TP        >(condense_path + "/CONDENSED_GEMM_X_S.bin",  sim_s_stream);
    save_condensed_tensor<int8_t, wq_t, NUM_W,  P_COP * P_CIP>(condense_path + "/CONDENSED_GEMM_W_Q.bin", sim_w_stream);
    save_condensed_tensor<int8_t, ws_t, NUM_WS, P_COP       >(condense_path + "/CONDENSED_GEMM_W_S1.bin", sim_s1_stream);
    save_condensed_tensor<int8_t, ws_t, NUM_WS, P_COP       >(condense_path + "/CONDENSED_GEMM_W_S2.bin", sim_s2_stream);

    stream2stream<aq_t, P_TP * P_CIP>(sim_i_stream,  i_stream);
    stream2stream<as_t, P_TP        >(sim_s_stream,  s_stream);
    stream2stream<wq_t, P_COP * P_CIP>(sim_w_stream, w_stream);
    stream2stream<ws_t, P_COP       >(sim_s1_stream, s1_stream);
    stream2stream<ws_t, P_COP       >(sim_s2_stream, s2_stream);
}

void prepare_cls_data(
    hls::stream<hls::vector<aq_t, P_TP * P_CIP> >& i_stream,
    hls::stream<hls::vector<as_t, P_TP        > >& s_stream,
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> >& w_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s1_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s2_stream
) {
    // LLM 的 l==LLAMA_L 表示 CLS/lm_head 路径，只消费 CLS_XLN 和 CLS_W。
    // 这个路径复用同一个 tensor core/accumulator，但矩阵数量和输出 repeat 与 decoder 普通层不同。
    const string binaries_path = BINARIES_PATH + to_string(LLAMA_L);
    const string condense_path = CONDENSE_PATH + to_string(LLAMA_L);

    auto CLS_XLN_Q = read_tensor<int8_t>(binaries_path + "/CLS_XLN_Q.bin");
    auto CLS_XLN_S = read_tensor<int8_t>(binaries_path + "/CLS_XLN_S.bin");
    auto CLS_W_Q   = read_tensor<int8_t>(binaries_path + "/CLS_W_Q.bin");
    auto CLS_W_S1  = read_tensor<int8_t>(binaries_path + "/CLS_W_S1.bin");
    auto CLS_W_S2  = read_tensor<int8_t>(binaries_path + "/CLS_W_S2.bin");

    vector<int8_t> REF_CLS_XLN_Q(P_LLM_T * P_LLM_C);
    vector<int8_t> REF_CLS_XLN_S(P_LLM_T * P_LLM_CT);
    // CLS/lm_head 输入窗口与 decoder carry 对齐到 P_LLM_POS；
    // 覆盖生成的 CONDENSED_GEMM_X_*，供单独 CLS 与 decoder+CLS 顶层测试共用。
    tensor2array<int8_t>(CLS_XLN_Q, REF_CLS_XLN_Q.data(), 1, 1,
                         infer_llm_t_load<int8_t>(CLS_XLN_Q, P_LLM_C, "CLS_XLN_Q"),
                         P_LLM_POS, P_LLM_T, P_LLM_C,  P_LLM_C);
    tensor2array<int8_t>(CLS_XLN_S, REF_CLS_XLN_S.data(), 1, 1,
                         infer_llm_t_load<int8_t>(CLS_XLN_S, P_LLM_CT, "CLS_XLN_S"),
                         P_LLM_POS, P_LLM_T, P_LLM_CT, P_LLM_CT);

    hls::stream<hls::vector<aq_t, P_TP * P_CIP> > sim_i_stream("sim_i_stream");
    hls::stream<hls::vector<as_t, P_TP        > > sim_s_stream("sim_s_stream");
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> > sim_w_stream("sim_w_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > sim_s1_stream("sim_s1_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > sim_s2_stream("sim_s2_stream");

    array2stream<int8_t, aq_t, P_LLM_VOCABT, 1, P_LLM_T, P_LLM_T, P_LLM_C,  P_CIP>(REF_CLS_XLN_Q.data(), sim_i_stream, "LLM Input CLS X", true);
    array2stream<int8_t, as_t, P_LLM_VOCABT, 1, P_LLM_T, P_LLM_T, P_LLM_CT, 1    >(REF_CLS_XLN_S.data(), sim_s_stream, "LLM Input CLS S", true);
    array2stream<int8_t, wq_t, 1,            1, P_LLM_VOCAB, P_COP, P_LLM_C,  P_CIP>(CLS_W_Q.data(),  sim_w_stream);
    array2stream<int8_t, ws_t, 1,            1, P_LLM_VOCAB, P_COP, P_LLM_CT, 1    >(CLS_W_S1.data(), sim_s1_stream);
    array2stream<int8_t, ws_t, 1,            1, P_LLM_VOCAB, P_COP, P_LLM_CT, 1    >(CLS_W_S2.data(), sim_s2_stream);

    const int NUM_X  = P_LLM_VOCABT * P_LLM_T * P_LLM_C;
    const int NUM_W  = P_LLM_VOCAB * P_LLM_C;
    const int NUM_XS = NUM_X / P_CIP;
    const int NUM_WS = NUM_W / P_CIP;
    save_condensed_tensor<int8_t, aq_t, NUM_X,  P_TP * P_CIP>(condense_path + "/CONDENSED_GEMM_X_Q.bin",  sim_i_stream);
    save_condensed_tensor<int8_t, as_t, NUM_XS, P_TP        >(condense_path + "/CONDENSED_GEMM_X_S.bin",  sim_s_stream);
    save_condensed_tensor<int8_t, wq_t, NUM_W,  P_COP * P_CIP>(condense_path + "/CONDENSED_GEMM_W_Q.bin", sim_w_stream);
    save_condensed_tensor<int8_t, ws_t, NUM_WS, P_COP       >(condense_path + "/CONDENSED_GEMM_W_S1.bin", sim_s1_stream);
    save_condensed_tensor<int8_t, ws_t, NUM_WS, P_COP       >(condense_path + "/CONDENSED_GEMM_W_S2.bin", sim_s2_stream);

    stream2stream<aq_t, P_TP * P_CIP>(sim_i_stream,  i_stream);
    stream2stream<as_t, P_TP        >(sim_s_stream,  s_stream);
    stream2stream<wq_t, P_COP * P_CIP>(sim_w_stream, w_stream);
    stream2stream<ws_t, P_COP       >(sim_s1_stream, s1_stream);
    stream2stream<ws_t, P_COP       >(sim_s2_stream, s2_stream);
}

void compare_decoder_data(int l, hls::stream<hls::vector<of_t, P_COP> >& o_stream) {
    // 按原 LLM_PERMUTE 的输出语义重建参考：
    //   QKV 输出按 h -> qkv -> t -> hc 排列。
    //   O 输出按 h -> t -> hc 排列。
    //   UG 输出按 cot -> u/g -> t -> cop 排列。
    //   D 输出按 t -> c 排列。
    const string binaries_path = BINARIES_PATH + to_string(l);
    const string condense_path = CONDENSE_PATH + to_string(l);

    auto MHA_Q = read_tensor<int64_t>(binaries_path + "/MHA_Q.bin");
    auto MHA_K = read_tensor<int64_t>(binaries_path + "/MHA_K.bin");
    auto MHA_V = read_tensor<int64_t>(binaries_path + "/MHA_V.bin");
    auto MHA_O = read_tensor<int64_t>(binaries_path + "/MHA_O.bin");
    auto MLP_U = read_tensor<int64_t>(binaries_path + "/MLP_XU.bin");
    auto MLP_G = read_tensor<int64_t>(binaries_path + "/MLP_XG.bin");
    auto MLP_D = read_tensor<int64_t>(binaries_path + "/MLP_XD.bin");

    constexpr int P_LLM_QKV_BLOCKS = P_LLM_KVH * (P_LLM_GQA + 2);
    constexpr int P_LLM_NUM_QKV_Y = P_LLM_QKV_BLOCKS * P_LLM_T * P_LLM_HC;
    vector<int64_t> REF_MHA_PERM_QKV(P_LLM_NUM_QKV_Y);
    vector<int64_t> REF_MHA_PERM_O  (P_LLM_H * P_LLM_T * P_LLM_HC);
    vector<int64_t> REF_MLP_PERM_UG (2 * P_LLM_T * P_LLM_CM);
    vector<int64_t> REF_MLP_PERM_D  (P_LLM_T * P_LLM_C);

    for (int kv = 0; kv < P_LLM_KVH; ++kv) {
        for (int block = 0; block < P_LLM_GQA + 2; ++block) {
            int src_head = (block == 0 || block == 1)
                ? kv * P_LLM_GQA
                : kv * P_LLM_GQA + (block - 2);
            for (int t = 0; t < P_LLM_T; ++t) {
                for (int hc = 0; hc < P_LLM_HC; ++hc) {
                    int64_t val = (block == 0) ? MHA_K[(P_LLM_POS + t) * P_LLM_C + src_head * P_LLM_HC + hc] :
                                  (block == 1) ? MHA_V[(P_LLM_POS + t) * P_LLM_C + src_head * P_LLM_HC + hc] :
                                                 MHA_Q[(P_LLM_POS + t) * P_LLM_C + src_head * P_LLM_HC + hc];
                    REF_MHA_PERM_QKV[(kv * (P_LLM_GQA + 2) + block) * P_LLM_T * P_LLM_HC + t * P_LLM_HC + hc] = val;
                }
            }
        }
    }
    for (int h = 0; h < P_LLM_H; ++h) {
        for (int t = 0; t < P_LLM_T; ++t) {
            for (int hc = 0; hc < P_LLM_HC; ++hc) {
                REF_MHA_PERM_O[h * P_LLM_T * P_LLM_HC + t * P_LLM_HC + hc] =
                    MHA_O[(P_LLM_POS + t) * P_LLM_C + h * P_LLM_HC + hc];
            }
        }
    }
    for (int cot = 0; cot < P_LLM_CMT; ++cot) {
        for (int ug = 0; ug < 2; ++ug) {
            for (int t = 0; t < P_LLM_T; ++t) {
                for (int cop = 0; cop < P_COP; ++cop) {
                    REF_MLP_PERM_UG[cot * 2 * P_LLM_T * P_COP + ug * P_LLM_T * P_COP + t * P_COP + cop] =
                        (ug == 0) ? MLP_U[(P_LLM_POS + t) * P_LLM_CM + cot * P_COP + cop] :
                                    MLP_G[(P_LLM_POS + t) * P_LLM_CM + cot * P_COP + cop];
                }
            }
        }
    }
    for (int t = 0; t < P_LLM_T; ++t) {
        for (int c = 0; c < P_LLM_C; ++c) {
            REF_MLP_PERM_D[t * P_LLM_C + c] = MLP_D[(P_LLM_POS + t) * P_LLM_C + c];
        }
    }

    const int NUM_Y = P_LLM_NUM_QKV_Y + P_LLM_H * P_LLM_T * P_LLM_HC +
                      2 * P_LLM_T * P_LLM_CM + P_LLM_T * P_LLM_C;
    hls::stream<hls::vector<of_t, P_COP> > sim_o_stream("sim_o_stream");
    stream2stream<of_t, NUM_Y, P_COP>(o_stream, sim_o_stream);
    save_condensed_tensor<int64_t, of_t, NUM_Y, P_COP>(condense_path + "/CONDENSED_GEMM_Y.bin", sim_o_stream);

    vector<int64_t> DUT_MHA_PERM_QKV(P_LLM_NUM_QKV_Y);
    vector<int64_t> DUT_MHA_PERM_O  (P_LLM_H * P_LLM_T * P_LLM_HC);
    vector<int64_t> DUT_MLP_PERM_UG (2 * P_LLM_T * P_LLM_CM);
    vector<int64_t> DUT_MLP_PERM_D  (P_LLM_T * P_LLM_C);
    stream2array_unpack<int64_t, of_t, P_LLM_QKV_BLOCKS, P_LLM_T, P_LLM_T, P_LLM_HC, P_COP>(sim_o_stream, DUT_MHA_PERM_QKV.data(), "LLM outQKV", true);
    stream2array_unpack<int64_t, of_t, P_LLM_H,     P_LLM_T, P_LLM_T, P_LLM_HC, P_COP>(sim_o_stream, DUT_MHA_PERM_O.data(),   "LLM Output O",   true);
    stream2array_unpack<int64_t, of_t, P_LLM_CMT * 2, P_LLM_T, P_LLM_T, P_COP, P_COP>(sim_o_stream, DUT_MLP_PERM_UG.data(),  "LLM Output UG",  true);
    stream2array_unpack<int64_t, of_t, 1, P_LLM_T, P_LLM_T, P_LLM_C, P_COP>(sim_o_stream, DUT_MLP_PERM_D.data(), "LLM Output D", true);
    assert(sim_o_stream.size() == 0);

    for (int kv = 0; kv < P_LLM_KVH; ++kv) {
        for (int block = 0; block < P_LLM_GQA + 2; ++block) {
            int extra_trunc = (block == 1) ? (MHA_TRUNC_V - P_TRUNC_BASE) : (MHA_TRUNC_QK - P_TRUNC_BASE);
            truncate<int64_t>(DUT_MHA_PERM_QKV.data() + (kv * (P_LLM_GQA + 2) + block) * P_LLM_T * P_LLM_HC,
                              P_LLM_T * P_LLM_HC, extra_trunc);
        }
    }
    truncate<int64_t>(DUT_MHA_PERM_O.data(),  P_LLM_H * P_LLM_T * P_LLM_HC, MHA_TRUNC_O  - P_TRUNC_BASE);
    truncate<int64_t>(DUT_MLP_PERM_UG.data(), 2 * P_LLM_T * P_LLM_CM,       MLP_TRUNC_UG - P_TRUNC_BASE);
    truncate<int64_t>(DUT_MLP_PERM_D.data(),  P_LLM_T * P_LLM_C,            MLP_TRUNC_D  - P_TRUNC_BASE);

    compare<int64_t>(REF_MHA_PERM_QKV.data(), DUT_MHA_PERM_QKV.data(), P_LLM_NUM_QKV_Y, "LLM_MHA_PERM_COMPACT_QKV");
    compare<int64_t>(REF_MHA_PERM_O.data(),   DUT_MHA_PERM_O.data(),   P_LLM_H * P_LLM_T * P_LLM_HC,     "LLM_MHA_PERM_O");
    compare<int64_t>(REF_MLP_PERM_UG.data(),  DUT_MLP_PERM_UG.data(),  2 * P_LLM_T * P_LLM_CM,           "LLM_MLP_PERM_UG");
    compare<int64_t>(REF_MLP_PERM_D.data(),   DUT_MLP_PERM_D.data(),   P_LLM_T * P_LLM_C,                "LLM_MLP_PERM_D");
}

void compare_cls_data(hls::stream<hls::vector<of_t, P_COP> >& o_stream) {
    // CLS/lm_head 输出是 T x VOCAB，仍通过基础截断后的 CONDENSED_GEMM_Y.bin 进入对比。
    const string binaries_path = BINARIES_PATH + to_string(LLAMA_L);
    const string condense_path = CONDENSE_PATH + to_string(LLAMA_L);

    auto CLS = read_tensor<int64_t>(binaries_path + "/CLS.bin");
    vector<int64_t> REF_CLS_PERM(P_LLM_T * P_LLM_VOCAB);
    // CLS golden logits 也按 P_LLM_POS 截取，避免 lm_head 输出与输入 XLN 不在同一 token 窗口。
    tensor2array<int64_t>(CLS, REF_CLS_PERM.data(), 1, 1,
                          infer_llm_t_load<int64_t>(CLS, P_LLM_VOCAB, "CLS"),
                          P_LLM_POS, P_LLM_T, P_LLM_VOCAB, P_LLM_VOCAB);

    const int NUM_Y = P_LLM_T * P_LLM_VOCAB;
    hls::stream<hls::vector<of_t, P_COP> > sim_o_stream("sim_o_stream");
    stream2stream<of_t, NUM_Y, P_COP>(o_stream, sim_o_stream);
    save_condensed_tensor<int64_t, of_t, NUM_Y, P_COP>(condense_path + "/CONDENSED_GEMM_Y.bin", sim_o_stream);

    vector<int64_t> DUT_CLS_PERM(P_LLM_T * P_LLM_VOCAB);
    stream2array_unpack<int64_t, of_t, 1, P_LLM_T, P_LLM_T, P_LLM_VOCAB, P_COP>(sim_o_stream, DUT_CLS_PERM.data(), "LLM Output CLS", true);
    assert(sim_o_stream.size() == 0);
    truncate<int64_t>(DUT_CLS_PERM.data(), P_LLM_T * P_LLM_VOCAB, CLS_TRUNC - P_TRUNC_BASE);
    compare<int64_t>(REF_CLS_PERM.data(), DUT_CLS_PERM.data(), P_LLM_T * P_LLM_VOCAB, "LLM_CLS_PERM");
}

void test_layer(int l_begin, int l_close) {
    // LLM mode step1 覆盖两种调用：
    //   test_layer(0, 1) 验证 decoder 普通层。
    //   test_layer(LLAMA_L, LLAMA_L+1) 验证 CLS/lm_head。
    hls::stream<hls::vector<aq_t, P_TP * P_CIP> > i_stream("llm_i_stream");
    hls::stream<hls::vector<as_t, P_TP        > > s_stream("llm_s_stream");
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> > w_stream("llm_w_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > s1_stream("llm_s1_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > s2_stream("llm_s2_stream");
    hls::stream<hls::vector<of_t, P_COP       > > o_stream("llm_o_stream");

    for (int l = l_begin; l < l_close; ++l) {
        if (l == LLAMA_L) {
            prepare_cls_data(i_stream, s_stream, w_stream, s1_stream, s2_stream);
        } else {
            prepare_decoder_data(l, i_stream, s_stream, w_stream, s1_stream, s2_stream);
        }
    }

    ::top(MODE_LLM, l_begin, l_close, i_stream, s_stream, w_stream, s1_stream, s2_stream, o_stream);
    assert(i_stream.size() == 0);
    assert(s_stream.size() == 0);
    assert(w_stream.size() == 0);
    assert(s1_stream.size() == 0);
    assert(s2_stream.size() == 0);

    for (int l = l_begin; l < l_close; ++l) {
        if (l == LLAMA_L) {
            compare_cls_data(o_stream);
        } else {
            compare_decoder_data(l, o_stream);
        }
    }
}

} // namespace llm_permute_tb

namespace vit_permute_tb {

void prepare_encoder_data(
    int l,
    hls::stream<hls::vector<aq_t, P_TP * P_CIP> >& i_stream,
    hls::stream<hls::vector<as_t, P_TP        > >& s_stream,
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> >& w_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s1_stream,
    hls::stream<hls::vector<ws_t, P_COP       > >& s2_stream
) {
    // ViT encoder 的关键约束是 QKV 保持 H -> Q/K/V -> TT -> HCT -> CT 顺序。
    // O/FC1/FC2 走 TT 外层顺序，权重按 TT 重放，因为同一组权重要广播到 128 个 token tile。
    // 这里仍保存 CONDENSED_GEMM_X/W_*，但目录是 Weights/ViT/condense/vision_N，因此不需要文件名前缀区分 ViT。
    const string binaries_path = VIT_BINARIES_PATH + to_string(l);
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);

    auto MHA_XLN_Q  = read_tensor<int8_t>(binaries_path + "/MHA_XLN_Q.bin");
    auto MHA_XLN_S  = read_tensor<int8_t>(binaries_path + "/MHA_XLN_S.bin");
    auto MHA_A_Q    = read_tensor<int8_t>(binaries_path + "/MHA_A_Q.bin");
    auto MHA_A_S    = read_tensor<int8_t>(binaries_path + "/MHA_A_S.bin");
    auto MLP_XLN_Q  = read_tensor<int8_t>(binaries_path + "/MLP_XLN_Q.bin");
    auto MLP_XLN_S  = read_tensor<int8_t>(binaries_path + "/MLP_XLN_S.bin");
    auto MLP_XM_Q   = read_tensor<int8_t>(binaries_path + "/MLP_XM_Q.bin");
    auto MLP_XM_S   = read_tensor<int8_t>(binaries_path + "/MLP_XM_S.bin");

    auto MHA_WQ_Q   = read_tensor<int8_t>(binaries_path + "/MHA_WQ_Q.bin");
    auto MHA_WQ_S1  = read_tensor<int8_t>(binaries_path + "/MHA_WQ_S1.bin");
    auto MHA_WQ_S2  = read_tensor<int8_t>(binaries_path + "/MHA_WQ_S2.bin");
    auto MHA_WK_Q   = read_tensor<int8_t>(binaries_path + "/MHA_WK_Q.bin");
    auto MHA_WK_S1  = read_tensor<int8_t>(binaries_path + "/MHA_WK_S1.bin");
    auto MHA_WK_S2  = read_tensor<int8_t>(binaries_path + "/MHA_WK_S2.bin");
    auto MHA_WV_Q   = read_tensor<int8_t>(binaries_path + "/MHA_WV_Q.bin");
    auto MHA_WV_S1  = read_tensor<int8_t>(binaries_path + "/MHA_WV_S1.bin");
    auto MHA_WV_S2  = read_tensor<int8_t>(binaries_path + "/MHA_WV_S2.bin");
    auto MHA_WO_Q   = read_tensor<int8_t>(binaries_path + "/MHA_WO_Q.bin");
    auto MHA_WO_S1  = read_tensor<int8_t>(binaries_path + "/MHA_WO_S1.bin");
    auto MHA_WO_S2  = read_tensor<int8_t>(binaries_path + "/MHA_WO_S2.bin");
    auto MLP_WFC1_Q  = read_tensor<int8_t>(binaries_path + "/MLP_WFC1_Q.bin");
    auto MLP_WFC1_S1 = read_tensor<int8_t>(binaries_path + "/MLP_WFC1_S1.bin");
    auto MLP_WFC1_S2 = read_tensor<int8_t>(binaries_path + "/MLP_WFC1_S2.bin");
    auto MLP_WFC2_Q  = read_tensor<int8_t>(binaries_path + "/MLP_WFC2_Q.bin");
    auto MLP_WFC2_S1 = read_tensor<int8_t>(binaries_path + "/MLP_WFC2_S1.bin");
    auto MLP_WFC2_S2 = read_tensor<int8_t>(binaries_path + "/MLP_WFC2_S2.bin");

    vector<int8_t> REF_MHA_XLN_Q(P_VIT_T * P_VIT_C);
    vector<int8_t> REF_MHA_XLN_S(P_VIT_T * P_VIT_CT);
    vector<int8_t> REF_MHA_A_Q  (P_VIT_T * P_VIT_C);
    vector<int8_t> REF_MHA_A_S  (P_VIT_T * P_VIT_CT);
    vector<int8_t> REF_MLP_XLN_Q(P_VIT_T * P_VIT_C);
    vector<int8_t> REF_MLP_XLN_S(P_VIT_T * P_VIT_CT);
    vector<int8_t> REF_MLP_XM_Q (P_VIT_T * P_VIT_CM);
    vector<int8_t> REF_MLP_XM_S (P_VIT_T * P_VIT_CMT);

    tensor2array<int8_t>(MHA_XLN_Q, REF_MHA_XLN_Q.data(), 1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_C,   P_VIT_C);
    tensor2array<int8_t>(MHA_XLN_S, REF_MHA_XLN_S.data(), 1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_CT,  P_VIT_CT);
    tensor2array<int8_t>(MHA_A_Q,   REF_MHA_A_Q.data(),   1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_C,   P_VIT_C);
    tensor2array<int8_t>(MHA_A_S,   REF_MHA_A_S.data(),   1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_CT,  P_VIT_CT);
    tensor2array<int8_t>(MLP_XLN_Q, REF_MLP_XLN_Q.data(), 1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_C,   P_VIT_C);
    tensor2array<int8_t>(MLP_XLN_S, REF_MLP_XLN_S.data(), 1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_CT,  P_VIT_CT);
    tensor2array<int8_t>(MLP_XM_Q,  REF_MLP_XM_Q.data(),  1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_CM,  P_VIT_CM);
    tensor2array<int8_t>(MLP_XM_S,  REF_MLP_XM_S.data(),  1, 1, P_VIT_T_LOAD, P_VIT_POS, P_VIT_T, P_VIT_CMT, P_VIT_CMT);

    hls::stream<hls::vector<aq_t, P_TP * P_CIP> > sim_i_stream("sim_i_stream");
    hls::stream<hls::vector<as_t, P_TP        > > sim_s_stream("sim_s_stream");
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> > sim_w_stream("sim_w_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > sim_s1_stream("sim_s1_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > sim_s2_stream("sim_s2_stream");

    for (int h = 0; h < P_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            for (int tt = 0; tt < P_VIT_TT; ++tt) {
                for (int hct = 0; hct < P_VIT_HCT; ++hct) {
                    (void)hct;
                    for (int ct = 0; ct < P_VIT_CT; ++ct) {
                        hls::vector<aq_t, P_TP * P_CIP> i_vec;
                        hls::vector<as_t, P_TP> s_vec;
                        for (int tp = 0; tp < P_TP; ++tp) {
                            for (int cip = 0; cip < P_CIP; ++cip) {
                                i_vec[tp * P_CIP + cip] = REF_MHA_XLN_Q[(tt * P_TP + tp) * P_VIT_C + ct * P_CIP + cip];
                            }
                            s_vec[tp] = REF_MHA_XLN_S[(tt * P_TP + tp) * P_VIT_CT + ct];
                        }
                        sim_i_stream.write(i_vec);
                        sim_s_stream.write(s_vec);
                    }
                }
            }
        }
    }
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int ct_out = 0; ct_out < P_VIT_CT; ++ct_out) {
            (void)ct_out;
            for (int ct = 0; ct < P_VIT_CT; ++ct) {
                hls::vector<aq_t, P_TP * P_CIP> i_vec;
                hls::vector<as_t, P_TP> s_vec;
                for (int tp = 0; tp < P_TP; ++tp) {
                    for (int cip = 0; cip < P_CIP; ++cip) {
                        i_vec[tp * P_CIP + cip] = REF_MHA_A_Q[(tt * P_TP + tp) * P_VIT_C + ct * P_CIP + cip];
                    }
                    s_vec[tp] = REF_MHA_A_S[(tt * P_TP + tp) * P_VIT_CT + ct];
                }
                sim_i_stream.write(i_vec);
                sim_s_stream.write(s_vec);
            }
        }
    }
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int cmt = 0; cmt < P_VIT_CMT; ++cmt) {
            for (int ct = 0; ct < P_VIT_CT; ++ct) {
                hls::vector<aq_t, P_TP * P_CIP> i_vec;
                hls::vector<as_t, P_TP> s_vec;
                for (int tp = 0; tp < P_TP; ++tp) {
                    for (int cip = 0; cip < P_CIP; ++cip) {
                        i_vec[tp * P_CIP + cip] = REF_MLP_XLN_Q[(tt * P_TP + tp) * P_VIT_C + ct * P_CIP + cip];
                    }
                    s_vec[tp] = REF_MLP_XLN_S[(tt * P_TP + tp) * P_VIT_CT + ct];
                }
                sim_i_stream.write(i_vec);
                sim_s_stream.write(s_vec);
            }
        }
    }
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int ct_out = 0; ct_out < P_VIT_CT; ++ct_out) {
            (void)ct_out;
            for (int cmt = 0; cmt < P_VIT_CMT; ++cmt) {
                hls::vector<aq_t, P_TP * P_CIP> i_vec;
                hls::vector<as_t, P_TP> s_vec;
                for (int tp = 0; tp < P_TP; ++tp) {
                    for (int cip = 0; cip < P_CIP; ++cip) {
                        i_vec[tp * P_CIP + cip] = REF_MLP_XM_Q[(tt * P_TP + tp) * P_VIT_CM + cmt * P_CIP + cip];
                    }
                    s_vec[tp] = REF_MLP_XM_S[(tt * P_TP + tp) * P_VIT_CMT + cmt];
                }
                sim_i_stream.write(i_vec);
                sim_s_stream.write(s_vec);
            }
        }
    }

    for (int h = 0; h < P_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            int8_t *wq_ptr  = (qkv == 0) ? (MHA_WQ_Q.data()  + h * P_VIT_HC * P_VIT_C)  :
                              (qkv == 1) ? (MHA_WK_Q.data()  + h * P_VIT_HC * P_VIT_C)  :
                                           (MHA_WV_Q.data()  + h * P_VIT_HC * P_VIT_C);
            int8_t *ws1_ptr = (qkv == 0) ? (MHA_WQ_S1.data() + h * P_VIT_HC * P_VIT_CT) :
                              (qkv == 1) ? (MHA_WK_S1.data() + h * P_VIT_HC * P_VIT_CT) :
                                           (MHA_WV_S1.data() + h * P_VIT_HC * P_VIT_CT);
            int8_t *ws2_ptr = (qkv == 0) ? (MHA_WQ_S2.data() + h * P_VIT_HC * P_VIT_CT) :
                              (qkv == 1) ? (MHA_WK_S2.data() + h * P_VIT_HC * P_VIT_CT) :
                                           (MHA_WV_S2.data() + h * P_VIT_HC * P_VIT_CT);
            for (int tt = 0; tt < P_VIT_TT; ++tt) {
                array2stream<int8_t, wq_t, 1, 1, P_VIT_HC, P_COP, P_VIT_C,  P_CIP>(wq_ptr,  sim_w_stream);
                array2stream<int8_t, ws_t, 1, 1, P_VIT_HC, P_COP, P_VIT_CT, 1    >(ws1_ptr, sim_s1_stream);
                array2stream<int8_t, ws_t, 1, 1, P_VIT_HC, P_COP, P_VIT_CT, 1    >(ws2_ptr, sim_s2_stream);
            }
        }
    }
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int ct_out = 0; ct_out < P_VIT_CT; ++ct_out) {
            array2stream<int8_t, wq_t, 1, 1, P_COP, P_COP, P_VIT_C,  P_CIP>(MHA_WO_Q.data()  + ct_out * P_COP * P_VIT_C,  sim_w_stream);
            array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_VIT_CT, 1    >(MHA_WO_S1.data() + ct_out * P_COP * P_VIT_CT, sim_s1_stream);
            array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_VIT_CT, 1    >(MHA_WO_S2.data() + ct_out * P_COP * P_VIT_CT, sim_s2_stream);
        }
    }
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int cmt = 0; cmt < P_VIT_CMT; ++cmt) {
            array2stream<int8_t, wq_t, 1, 1, P_COP, P_COP, P_VIT_C,  P_CIP>(MLP_WFC1_Q.data()  + cmt * P_COP * P_VIT_C,  sim_w_stream);
            array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_VIT_CT, 1    >(MLP_WFC1_S1.data() + cmt * P_COP * P_VIT_CT, sim_s1_stream);
            array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_VIT_CT, 1    >(MLP_WFC1_S2.data() + cmt * P_COP * P_VIT_CT, sim_s2_stream);
        }
    }
    for (int tt = 0; tt < P_VIT_TT; ++tt) {
        for (int ct_out = 0; ct_out < P_VIT_CT; ++ct_out) {
            array2stream<int8_t, wq_t, 1, 1, P_COP, P_COP, P_VIT_CM,  P_CIP>(MLP_WFC2_Q.data()  + ct_out * P_COP * P_VIT_CM,  sim_w_stream);
            array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_VIT_CMT, 1    >(MLP_WFC2_S1.data() + ct_out * P_COP * P_VIT_CMT, sim_s1_stream);
            array2stream<int8_t, ws_t, 1, 1, P_COP, P_COP, P_VIT_CMT, 1    >(MLP_WFC2_S2.data() + ct_out * P_COP * P_VIT_CMT, sim_s2_stream);
        }
    }

    const int NUM_X  = 3 * P_VIT_CT * P_VIT_T * P_VIT_C + P_VIT_CT * P_VIT_T * P_VIT_C +
                       P_VIT_CMT * P_VIT_T * P_VIT_C + P_VIT_CT * P_VIT_T * P_VIT_CM;
    const int NUM_W  = P_VIT_TT * (3 * P_VIT_C * P_VIT_C + P_VIT_C * P_VIT_C +
                                   P_VIT_CM * P_VIT_C + P_VIT_C * P_VIT_CM);
    const int NUM_XS = NUM_X / P_CIP;
    const int NUM_WS = NUM_W / P_CIP;
    save_condensed_tensor<int8_t, aq_t, NUM_X,  P_TP * P_CIP>(condense_path + "/CONDENSED_GEMM_X_Q.bin",  sim_i_stream);
    save_condensed_tensor<int8_t, as_t, NUM_XS, P_TP        >(condense_path + "/CONDENSED_GEMM_X_S.bin",  sim_s_stream);
    save_condensed_tensor<int8_t, wq_t, NUM_W,  P_COP * P_CIP>(condense_path + "/CONDENSED_GEMM_W_Q.bin", sim_w_stream);
    save_condensed_tensor<int8_t, ws_t, NUM_WS, P_COP       >(condense_path + "/CONDENSED_GEMM_W_S1.bin", sim_s1_stream);
    save_condensed_tensor<int8_t, ws_t, NUM_WS, P_COP       >(condense_path + "/CONDENSED_GEMM_W_S2.bin", sim_s2_stream);

    stream2stream<aq_t, P_TP * P_CIP>(sim_i_stream,  i_stream);
    stream2stream<as_t, P_TP        >(sim_s_stream,  s_stream);
    stream2stream<wq_t, P_COP * P_CIP>(sim_w_stream, w_stream);
    stream2stream<ws_t, P_COP       >(sim_s1_stream, s1_stream);
    stream2stream<ws_t, P_COP       >(sim_s2_stream, s2_stream);
}

void compare_encoder_data(int l, hls::stream<hls::vector<of_t, P_COP> >& o_stream) {
    // ViT 对比在 GEMM 基础截断后补上各投影 bias，再做各路径额外截断：
    //   Q/K 使用 VIT_MHA_TRUNC_QK，V 使用 VIT_MHA_TRUNC_V。
    //   O/FC1/FC2 分别使用 ViT MHA/MLP 对应 trunc。
    // 这样保持与 software/int_SmolVLM2 的 IntViTAttention/MLP 数值边界一致。
    const string binaries_path = VIT_BINARIES_PATH + to_string(l);
    const string condense_path = VIT_CONDENSE_PATH + to_string(l);

    auto MHA_Q    = read_tensor<int64_t>(binaries_path + "/MHA_Q.bin");
    auto MHA_K    = read_tensor<int64_t>(binaries_path + "/MHA_K.bin");
    auto MHA_V    = read_tensor<int64_t>(binaries_path + "/MHA_V.bin");
    auto MHA_O    = read_tensor<int64_t>(binaries_path + "/MHA_O.bin");
    auto MHA_BQ   = read_tensor<int64_t>(binaries_path + "/MHA_BQ.bin");
    auto MHA_BK   = read_tensor<int64_t>(binaries_path + "/MHA_BK.bin");
    auto MHA_BV   = read_tensor<int64_t>(binaries_path + "/MHA_BV.bin");
    auto MHA_BO   = read_tensor<int64_t>(binaries_path + "/MHA_BO.bin");
    auto MLP_B1   = read_tensor<int64_t>(binaries_path + "/MLP_B1.bin");
    auto MLP_B2   = read_tensor<int64_t>(binaries_path + "/MLP_B2.bin");
    auto MLP_XFC1 = read_tensor<int64_t>(binaries_path + "/MLP_XFC1.bin");
    auto MLP_XFC2 = read_tensor<int64_t>(binaries_path + "/MLP_XFC2.bin");

    vector<int64_t> REF_MHA_PERM_QKV(P_VIT_H * 3 * P_VIT_T * P_VIT_HC);
    vector<int64_t> REF_MHA_PERM_O  (P_VIT_T * P_VIT_C);
    vector<int64_t> REF_MLP_PERM_FC1(P_VIT_T * P_VIT_CM);
    vector<int64_t> REF_MLP_PERM_FC2(P_VIT_T * P_VIT_C);

    for (int h = 0; h < P_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            for (int t = 0; t < P_VIT_T; ++t) {
                for (int hc = 0; hc < P_VIT_HC; ++hc) {
                    int64_t val = (qkv == 0) ? MHA_Q[(P_VIT_POS + t) * P_VIT_C + h * P_VIT_HC + hc] :
                                  (qkv == 1) ? MHA_K[(P_VIT_POS + t) * P_VIT_C + h * P_VIT_HC + hc] :
                                               MHA_V[(P_VIT_POS + t) * P_VIT_C + h * P_VIT_HC + hc];
                    REF_MHA_PERM_QKV[h * 3 * P_VIT_T * P_VIT_HC + qkv * P_VIT_T * P_VIT_HC + t * P_VIT_HC + hc] = val;
                }
            }
        }
    }
    for (int t = 0; t < P_VIT_T; ++t) {
        for (int c = 0; c < P_VIT_C; ++c) {
            REF_MHA_PERM_O[t * P_VIT_C + c] = MHA_O[(P_VIT_POS + t) * P_VIT_C + c];
        }
        for (int c = 0; c < P_VIT_CM; ++c) {
            REF_MLP_PERM_FC1[t * P_VIT_CM + c] = MLP_XFC1[(P_VIT_POS + t) * P_VIT_CM + c];
        }
        for (int c = 0; c < P_VIT_C; ++c) {
            REF_MLP_PERM_FC2[t * P_VIT_C + c] = MLP_XFC2[(P_VIT_POS + t) * P_VIT_C + c];
        }
    }

    const int NUM_Y = P_VIT_H * 3 * P_VIT_T * P_VIT_HC + P_VIT_T * P_VIT_C + P_VIT_T * P_VIT_CM + P_VIT_T * P_VIT_C;
    hls::stream<hls::vector<of_t, P_COP> > sim_o_stream("sim_o_stream");
    stream2stream<of_t, NUM_Y, P_COP>(o_stream, sim_o_stream);
    save_condensed_tensor<int64_t, of_t, NUM_Y, P_COP>(condense_path + "/CONDENSED_GEMM_Y.bin", sim_o_stream);

    vector<int64_t> DUT_MHA_PERM_QKV(P_VIT_H * 3 * P_VIT_T * P_VIT_HC);
    vector<int64_t> DUT_MHA_PERM_O  (P_VIT_T * P_VIT_C);
    vector<int64_t> DUT_MLP_PERM_FC1(P_VIT_T * P_VIT_CM);
    vector<int64_t> DUT_MLP_PERM_FC2(P_VIT_T * P_VIT_C);
    stream2array_unpack<int64_t, of_t, P_VIT_H * 3, P_VIT_T, P_TP, P_VIT_HC, P_COP>(sim_o_stream, DUT_MHA_PERM_QKV.data(), "ViT Output QKV", true);
    stream2array_unpack<int64_t, of_t, 1, P_VIT_T, P_TP, P_VIT_C,  P_COP>(sim_o_stream, DUT_MHA_PERM_O.data(),   "ViT Output O",   true);
    stream2array_unpack<int64_t, of_t, 1, P_VIT_T, P_TP, P_VIT_CM, P_COP>(sim_o_stream, DUT_MLP_PERM_FC1.data(), "ViT Output FC1", true);
    stream2array_unpack<int64_t, of_t, 1, P_VIT_T, P_TP, P_VIT_C,  P_COP>(sim_o_stream, DUT_MLP_PERM_FC2.data(), "ViT Output FC2", true);
    assert(sim_o_stream.size() == 0);

    for (int h = 0; h < P_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            int extra_trunc = (qkv == 0 || qkv == 1) ? (VIT_MHA_TRUNC_QK - P_TRUNC_BASE) : (VIT_MHA_TRUNC_V - P_TRUNC_BASE);
            truncate<int64_t>(DUT_MHA_PERM_QKV.data() + h * 3 * P_VIT_T * P_VIT_HC + qkv * P_VIT_T * P_VIT_HC,
                              P_VIT_T * P_VIT_HC, extra_trunc);
        }
    }
    for (int h = 0; h < P_VIT_H; ++h) {
        for (int qkv = 0; qkv < 3; ++qkv) {
            for (int hc = 0; hc < P_VIT_HC; ++hc) {
                for (int t = 0; t < P_VIT_T; ++t) {
                    int idx = h * 3 * P_VIT_T * P_VIT_HC + qkv * P_VIT_T * P_VIT_HC + t * P_VIT_HC + hc;
                    int c_idx = h * P_VIT_HC + hc;
                    int64_t bias = (qkv == 0) ? MHA_BQ[c_idx] : (qkv == 1) ? MHA_BK[c_idx] : MHA_BV[c_idx];
                    DUT_MHA_PERM_QKV[idx] += bias;
                }
            }
        }
    }
    truncate<int64_t>(DUT_MHA_PERM_O.data(),   P_VIT_T * P_VIT_C,  VIT_MHA_TRUNC_O   - P_TRUNC_BASE);
    truncate<int64_t>(DUT_MLP_PERM_FC1.data(), P_VIT_T * P_VIT_CM, VIT_MLP_TRUNC_FC1 - P_TRUNC_BASE);
    truncate<int64_t>(DUT_MLP_PERM_FC2.data(), P_VIT_T * P_VIT_C,  VIT_MLP_TRUNC_FC2 - P_TRUNC_BASE);
    for (int t = 0; t < P_VIT_T; ++t) {
        for (int c = 0; c < P_VIT_C; ++c) {
            DUT_MHA_PERM_O[t * P_VIT_C + c] += MHA_BO[c];
            DUT_MLP_PERM_FC2[t * P_VIT_C + c] += MLP_B2[c];
        }
        for (int c = 0; c < P_VIT_CM; ++c) {
            DUT_MLP_PERM_FC1[t * P_VIT_CM + c] += MLP_B1[c];
        }
    }

    compare<int64_t>(REF_MHA_PERM_QKV.data(), DUT_MHA_PERM_QKV.data(), P_VIT_H * 3 * P_VIT_T * P_VIT_HC, "ViT_MHA_PERM_QKV");
    compare<int64_t>(REF_MHA_PERM_O.data(),   DUT_MHA_PERM_O.data(),   P_VIT_T * P_VIT_C,                 "ViT_MHA_PERM_O");
    compare<int64_t>(REF_MLP_PERM_FC1.data(), DUT_MLP_PERM_FC1.data(), P_VIT_T * P_VIT_CM,                "ViT_MLP_PERM_FC1");
    compare<int64_t>(REF_MLP_PERM_FC2.data(), DUT_MLP_PERM_FC2.data(), P_VIT_T * P_VIT_C,                 "ViT_MLP_PERM_FC2");
}

void test_layer(int l_begin, int l_close) {
    // ViT mode step1 当前验证 vision layer 0。后续扩展多层时保持同一个 ::top(MODE_VIT, ...) 调用边界。
    hls::stream<hls::vector<aq_t, P_TP * P_CIP> > i_stream("vit_i_stream");
    hls::stream<hls::vector<as_t, P_TP        > > s_stream("vit_s_stream");
    hls::stream<hls::vector<wq_t, P_COP * P_CIP> > w_stream("vit_w_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > s1_stream("vit_s1_stream");
    hls::stream<hls::vector<ws_t, P_COP       > > s2_stream("vit_s2_stream");
    hls::stream<hls::vector<of_t, P_COP       > > o_stream("vit_o_stream");

    for (int l = l_begin; l < l_close; ++l) {
        prepare_encoder_data(l, i_stream, s_stream, w_stream, s1_stream, s2_stream);
    }

    ::top(MODE_VIT, l_begin, l_close, i_stream, s_stream, w_stream, s1_stream, s2_stream, o_stream);
    assert(i_stream.size() == 0);
    assert(s_stream.size() == 0);
    assert(w_stream.size() == 0);
    assert(s1_stream.size() == 0);
    assert(s2_stream.size() == 0);

    for (int l = l_begin; l < l_close; ++l) {
        compare_encoder_data(l, o_stream);
    }
}

} // namespace vit_permute_tb

#ifndef __SYNTHESIS__
int main() {
    // 一个 CSim 同时覆盖共享 PERMUTE 的三条实际调用路径，防止只验证单 mode 后遗漏接口或顺序问题。
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        llm_permute_tb::test_layer(0, 1);
        llm_permute_tb::test_layer(LLAMA_L, LLAMA_L + 1);
    }
    if (!reuse_tb_only_llm()) {
        vit_permute_tb::test_layer(vit_layer, vit_layer + 1);
    }
    return 0;
}
#endif
