#include "../src/reuse_common.h"
#include "../src/utils.h"
#include "../src/buffer.h"
#include <vector>

// ============================================================================
// LLM/ViT 复用 MUX
// ============================================================================
// 功能：
//   1. 将 XLN/A/XM 三路上游量化结果汇聚为主 GEMM 的 q/s 输入流。
//   2. LLM mode 支持 decoder 普通层和 LLAMA_L 的 CLS/lm_head 路径。
//   3. ViT mode 保持 vision encoder 顺序：QKV 使用 [h][qkv][tt][hct][ct]，
//      O/FC1/FC2 使用 TT tile 顺序，输出每拍 TP_TILE*CP = 64 个 q lane。
//
// 资源意图：
//   - 顶层端口统一使用 REUSE_AQ_T/REUSE_AS_T，二者当前与 LLM/ViT 本地 q/s
//     位宽一致，不需要额外 Quantizer。
//   - ViT MHA-QKV 必须缓存 T=1024 全量 XLN；沿用 64-bit q pack + 32-bit
//     scale pack 并绑定 URAM，避免回退到窄多维数组爆 BRAM。
//   - LLM mode 只使用 8-token 小 buffer；mode 分支在 layer 粒度选择路径，
//     TP/CP 内层只做 pack/unpack 或 stream forward。

constexpr int MX_CP      = REUSE_CP;
constexpr int MX_TP_IN   = 1;
constexpr int MX_TP_TILE = REUSE_GEMM_TP;

typedef REUSE_AQ_T mx_aq_t;
typedef REUSE_AS_T mx_as_t;
typedef ap_uint<DW_AQ * MX_CP>      mx_aq_pack_t;
typedef ap_uint<DW_AS * MX_TP_TILE> mx_as_pack_t;

static_assert(MX_CP == 8, "MUX reuse assumes CP=8");
static_assert(MX_TP_TILE == 8, "MUX output tile must stay 8 tokens");
static_assert(DW_AQ == VIT_DW_AQ, "LLM/ViT q width must match");
static_assert(DW_AS == VIT_DW_AS, "LLM/ViT scale width must match");

// LLM 参数：decode tile 固定为 8 tokens，CLS/lm_head 也复用同一输出宽度。
constexpr int MX_LLM_T      = REUSE_LLM_TILE_T;
constexpr int MX_LLM_H      = LLAMA_H;
constexpr int MX_LLM_KVH    = LLAMA_KVH;
constexpr int MX_LLM_GQA    = MX_LLM_H / MX_LLM_KVH;
constexpr int MX_LLM_C      = LLAMA_C;
constexpr int MX_LLM_HC     = LLAMA_HC;
constexpr int MX_LLM_CM     = LLAMA_CM;
constexpr int MX_LLM_VOCAB  = LLAMA_VOCAB;
constexpr int MX_LLM_TT     = MX_LLM_T / MX_TP_IN;
constexpr int MX_LLM_CT     = MX_LLM_C / MX_CP;
constexpr int MX_LLM_CMT    = MX_LLM_CM / MX_CP;
constexpr int MX_LLM_HCT    = MX_LLM_HC / MX_CP;
constexpr int MX_LLM_VOCABT = MX_LLM_VOCAB / MX_CP;
constexpr int MX_LLM_R_Q    = MX_LLM_H   * MX_LLM_HCT;
constexpr int MX_LLM_R_KV   = MX_LLM_KVH * MX_LLM_HCT;
// Native GQA LLM MHA projection order: per KV group K, Q0/Q1/Q2, V; then O.
constexpr int MX_LLM_R_QKV  = MX_LLM_R_KV + MX_LLM_R_Q + MX_LLM_R_KV;
constexpr int MX_LLM_R_O    = MX_LLM_H * 1 * MX_LLM_HCT;
constexpr int MX_LLM_R_M    = 2 * MX_LLM_CMT;
constexpr int MX_LLM_R_D    = MX_LLM_CT;
constexpr int MX_LLM_R_CLS  = MX_LLM_VOCABT;
constexpr int MX_LLM_NUM_X  = (MX_LLM_R_QKV * MX_LLM_T * MX_LLM_C
                             +      MX_LLM_CT * MX_LLM_T * MX_LLM_C
                             + 2 * MX_LLM_CMT * MX_LLM_T * MX_LLM_C
                             +      MX_LLM_CT * MX_LLM_T * MX_LLM_CM);
constexpr int MX_LLM_NUM_WS = MX_LLM_NUM_X / MX_CP;
constexpr int MX_LLM_N_CLS  = MX_LLM_R_CLS * MX_LLM_CT;
constexpr int MX_LLM_NUM_X_CLS  = MX_LLM_N_CLS * MX_LLM_T * MX_CP;
constexpr int MX_LLM_NUM_WS_CLS = MX_LLM_NUM_X_CLS / MX_CP;

// ViT 参数：MHA-QKV 需要 T=1024 全 token 大 buffer。
constexpr int MX_VIT_T      = VIT_S;
constexpr int MX_VIT_H      = VIT_H;
constexpr int MX_VIT_C      = VIT_C;
constexpr int MX_VIT_HC     = VIT_HC;
constexpr int MX_VIT_CM     = VIT_MLP_DIM;
constexpr int MX_VIT_TT     = MX_VIT_T / MX_TP_TILE;
constexpr int MX_VIT_CT     = MX_VIT_C / MX_CP;
constexpr int MX_VIT_CMT    = MX_VIT_CM / MX_CP;
constexpr int MX_VIT_HCT    = MX_VIT_HC / MX_CP;
constexpr int MX_VIT_R_QKV  = MX_VIT_H * 3 * MX_VIT_HCT;
constexpr int MX_VIT_R_O    = MX_VIT_H * 1 * MX_VIT_HCT;
constexpr int MX_VIT_R_M    = MX_VIT_CMT;
constexpr int MX_VIT_R_D    = MX_VIT_CT;
constexpr int MX_VIT_NUM_X  = (3 * MX_VIT_CT * MX_VIT_T * MX_VIT_C
                             +      MX_VIT_CT * MX_VIT_T * MX_VIT_C
                             +      MX_VIT_CMT * MX_VIT_T * MX_VIT_C
                             +      MX_VIT_CT * MX_VIT_T * MX_VIT_CM);
constexpr int MX_VIT_NUM_WS = MX_VIT_NUM_X / MX_CP;

// ViT MHA-QKV 大 buffer：q 每个 token lane 存 64-bit CP pack，scale 每个
// token tile 存 8 个 4-bit scale pack。该布局是独立 ViT_MUX 已验证的 URAM 方案。
mx_aq_pack_t vit_xln_mha_buf_q[MX_TP_TILE][MX_VIT_TT][MX_VIT_CT];
mx_as_pack_t vit_xln_mha_buf_s[MX_VIT_TT][MX_VIT_CT];

static ap_uint<DW_AQ> mx_aq_to_bits(mx_aq_t val) {
    #pragma HLS inline
    ap_uint<DW_AQ> bits = val.range(DW_AQ - 1, 0);
    return bits;
}

static mx_aq_t mx_aq_from_bits(ap_uint<DW_AQ> bits) {
    #pragma HLS inline
    mx_aq_t val;
    val.range(DW_AQ - 1, 0) = bits;
    return val;
}

static mx_aq_pack_t mx_pack_aq_cp(const hls::vector<mx_aq_t, MX_CP>& vec) {
    #pragma HLS inline
    mx_aq_pack_t pack = 0;
    for (int cp = 0; cp < MX_CP; ++cp) {
        #pragma HLS unroll
        pack.range((cp + 1) * DW_AQ - 1, cp * DW_AQ) = mx_aq_to_bits(vec[cp]);
    }
    return pack;
}

static void mx_unpack_aq_cp(mx_aq_pack_t pack, hls::vector<mx_aq_t, MX_CP>& vec) {
    #pragma HLS inline
    for (int cp = 0; cp < MX_CP; ++cp) {
        #pragma HLS unroll
        vec[cp] = mx_aq_from_bits(pack.range((cp + 1) * DW_AQ - 1, cp * DW_AQ));
    }
}

static void mx_set_as_lane(mx_as_pack_t& pack, int tp, mx_as_t val) {
    #pragma HLS inline
    switch (tp) {
        case 0: pack.range(1 * DW_AS - 1, 0 * DW_AS) = val; break;
        case 1: pack.range(2 * DW_AS - 1, 1 * DW_AS) = val; break;
        case 2: pack.range(3 * DW_AS - 1, 2 * DW_AS) = val; break;
        case 3: pack.range(4 * DW_AS - 1, 3 * DW_AS) = val; break;
        case 4: pack.range(5 * DW_AS - 1, 4 * DW_AS) = val; break;
        case 5: pack.range(6 * DW_AS - 1, 5 * DW_AS) = val; break;
        case 6: pack.range(7 * DW_AS - 1, 6 * DW_AS) = val; break;
        default: pack.range(8 * DW_AS - 1, 7 * DW_AS) = val; break;
    }
}

static mx_as_t mx_get_as_lane(mx_as_pack_t pack, int tp) {
    #pragma HLS inline
    switch (tp) {
        case 0: return pack.range(1 * DW_AS - 1, 0 * DW_AS);
        case 1: return pack.range(2 * DW_AS - 1, 1 * DW_AS);
        case 2: return pack.range(3 * DW_AS - 1, 2 * DW_AS);
        case 3: return pack.range(4 * DW_AS - 1, 3 * DW_AS);
        case 4: return pack.range(5 * DW_AS - 1, 4 * DW_AS);
        case 5: return pack.range(6 * DW_AS - 1, 5 * DW_AS);
        case 6: return pack.range(7 * DW_AS - 1, 6 * DW_AS);
        default: return pack.range(8 * DW_AS - 1, 7 * DW_AS);
    }
}

// LLM 小 buffer：沿用原 LLM_MUX 结构，全部为 8-token tile 内局部缓存。
BUFFER<mx_aq_t, 1,        MX_LLM_T, MX_TP_IN, MX_LLM_C,   MX_CP> llm_xln_buffer_q;
BUFFER<mx_as_t, 1,        MX_LLM_T, MX_TP_IN, MX_LLM_CT,  1    > llm_xln_buffer_s;
BUFFER<mx_aq_t, MX_LLM_H, MX_LLM_T, MX_TP_IN, MX_LLM_HC,  MX_CP> llm_a_buffer_q;
BUFFER<mx_as_t, MX_LLM_H, MX_LLM_T, MX_TP_IN, MX_LLM_HCT, 1    > llm_a_buffer_s;

// ViT 小 buffer：O/FC1/FC2 都按 8-token tile 流式处理；只有 MHA-QKV 需要全局大 buffer。
BUFFER<mx_aq_t, MX_VIT_H, MX_TP_TILE, MX_TP_IN, MX_VIT_HC,  MX_CP> vit_a_buffer_q;
BUFFER<mx_as_t, MX_VIT_H, MX_TP_TILE, MX_TP_IN, MX_VIT_HCT, 1    > vit_a_buffer_s;
BUFFER<mx_aq_t, 1,        MX_TP_TILE, MX_TP_IN, MX_VIT_C,   MX_CP> vit_fc1_xln_buf_q;
BUFFER<mx_as_t, 1,        MX_TP_TILE, MX_TP_IN, MX_VIT_CT,  1    > vit_fc1_xln_buf_s;
BUFFER<mx_aq_t, 1,        MX_TP_TILE, MX_TP_IN, MX_VIT_CM,  MX_CP> vit_xm_buf_q;
BUFFER<mx_as_t, 1,        MX_TP_TILE, MX_TP_IN, MX_VIT_CMT, 1    > vit_xm_buf_s;

// LLM XM 小 buffer 专用版本。旧实现为了节省 BRAM 绑定 LUTRAM，但 K26 顶层
// CLB/SLICEM 已成为 legal route 瓶颈；这里改成 BRAM，用约 24 个 BRAM tile
// 换掉 MLP unpack 的分布式 RAM，优先释放 CLB packing 和局部地址线布线压力。
template<class data_t, int H, int T, int TP, int C, int CP>
void llm_do_buffer_unpack_bram(
    bool trigger,
    int repeat,
    hls::stream<hls::vector<data_t, CP> >& i_stream,
    hls::stream<hls::vector<data_t, T * CP> >& o_stream
) {
    if (!trigger) return;

    constexpr int TT = T / TP;
    constexpr int CT = C / CP;
    data_t buf[H][TT][TP][CT][CP];
    #pragma HLS array_partition variable=buf complete dim=2
    #pragma HLS array_reshape variable=buf complete dim=3
    #pragma HLS array_reshape variable=buf complete dim=5
    #pragma HLS bind_storage variable=buf type=ram_2p impl=bram

    // 输入 XM 为 unpacked order：[h][ct][tt][tp]，每拍 CP 个 lane。
    for (int h = 0; h < H; ++h) {
        for (int ct = 0; ct < CT; ++ct) {
            for (int tt = 0; tt < TT; ++tt) {
                for (int tp = 0; tp < TP; ++tp) {
                    #pragma HLS pipeline II=1
                    hls::vector<data_t, CP> i_vec = i_stream.read();
                    for (int cp = 0; cp < CP; ++cp) {
                        #pragma HLS unroll
                        buf[h][tt][tp][ct][cp] = i_vec[cp];
                    }
                }
            }
        }
    }

    // 输出为 GEMM-facing 64-lane tile，重复 R_D 次供 down projection 权重配对。
    for (int h = 0; h < H; ++h) {
        for (int r = 0; r < repeat; ++r) {
            for (int ct = 0; ct < CT; ++ct) {
                #pragma HLS pipeline II=1
                hls::vector<data_t, T * CP> o_vec;
                for (int tt = 0; tt < TT; ++tt) {
                    for (int tp = 0; tp < TP; ++tp) {
                        for (int cp = 0; cp < CP; ++cp) {
                            #pragma HLS unroll
                            o_vec[tt * TP * CP + tp * CP + cp] = buf[h][tt][tp][ct][cp];
                        }
                    }
                }
                o_stream.write(o_vec);
            }
        }
    }
}

// LLM CLS/lm_head 输出选择：只消费 final RMSNorm 的 XLN 流。
void llm_cls_gemm_mux(
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> >& xlnq_buffered_stream,
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > >& xlns_buffered_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    for (int n = 0; n < MX_LLM_R_CLS * MX_LLM_CT; ++n) {
        #pragma HLS pipeline II=1
        q_stream.write(xlnq_buffered_stream.read());
        s_stream.write(xlns_buffered_stream.read());
    }
}

// LLM MHA 输出选择：先按 native GQA compact K/Q/V 重放 XLN，再重放 RV_GEMM 的 A 给 O 权重。
void llm_mha_gemm_mux(
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> >& xlnq_buffered_stream,
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > >& xlns_buffered_stream,
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> >& aq_buffered_stream,
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > >& as_buffered_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    constexpr int n_qkv = MX_LLM_R_QKV * MX_LLM_CT;
    constexpr int n_o   = MX_LLM_R_O   * MX_LLM_CT;
    for (int n = 0; n < n_qkv + n_o; ++n) {
        #pragma HLS pipeline II=1
        if (n < n_qkv) {
            q_stream.write(xlnq_buffered_stream.read());
            s_stream.write(xlns_buffered_stream.read());
        } else {
            q_stream.write(aq_buffered_stream.read());
            s_stream.write(as_buffered_stream.read());
        }
    }
}

// LLM MLP 输出选择：先重放 XLN 给 up/gate 权重，再重放 SiLU 后 XM 给 down 权重。
void llm_mlp_gemm_mux(
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> >& xlnq_buffered_stream,
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > >& xlns_buffered_stream,
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> >& xmq_buffered_stream,
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > >& xms_buffered_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    constexpr int n_up_gate = MX_LLM_R_M * MX_LLM_CT;
    constexpr int n_down    = MX_LLM_R_D * MX_LLM_CMT;
    for (int n = 0; n < n_up_gate + n_down; ++n) {
        #pragma HLS pipeline II=1
        if (n < n_up_gate) {
            q_stream.write(xlnq_buffered_stream.read());
            s_stream.write(xlns_buffered_stream.read());
        } else {
            q_stream.write(xmq_buffered_stream.read());
            s_stream.write(xms_buffered_stream.read());
        }
    }
}

// LLM CLS/lm_head pass：单独 dataflow scope，避免普通层的 A/XM 空分支进入 RTL 调度。
void run_llm_cls_mux_layer(
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& xlnq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& xlns_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    #pragma HLS dataflow
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> > xlnq_buffered_stream("llm_cls_xlnq_buffered_stream");
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > > xlns_buffered_stream("llm_cls_xlns_buffered_stream");
    // 生产者和消费者都是 II=1 且无突发错位，depth=1 避免默认 512-bit/32-bit
    // depth=2 shift FIFO 消耗大量 LUT/FF。
    #pragma HLS stream variable=xlnq_buffered_stream depth=1
    #pragma HLS stream variable=xlns_buffered_stream depth=1
    // CLB-02: after QK-01 freed BRAM tiles, move the CLS/lm_head q replay
    // buffer back to BRAM. This removes one large MUX LUTRAM island while
    // keeping the stream order and dataflow contract unchanged.
    llm_xln_buffer_q.do_buffer_bram(true, MX_LLM_R_CLS, xlnq_stream, xlnq_buffered_stream);
    // CLB-02(b)：CLS/lm_head scale replay 是 run_llm_mux 剩余 LUTRAM 大头，
    // 也转 BRAM；该路径只重放 scale，不改变 q/s 对齐语义。
    llm_xln_buffer_s.do_buffer_bram(true, MX_LLM_R_CLS, xlns_stream, xlns_buffered_stream);
    llm_cls_gemm_mux(xlnq_buffered_stream, xlns_buffered_stream, q_stream, s_stream);
}

// LLM MHA pass：只启用 XLN 与 A 两类输入，输出顺序匹配 QKV 后接 O。
void run_llm_mha_mux_layer(
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& xlnq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& xlns_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& aq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& as_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    #pragma HLS dataflow
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> > xlnq_buffered_stream("llm_mha_xlnq_buffered_stream");
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > > xlns_buffered_stream("llm_mha_xlns_buffered_stream");
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> > aq_buffered_stream("llm_mha_aq_buffered_stream");
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > > as_buffered_stream("llm_mha_as_buffered_stream");
    // MUX 子级内部 FIFO 只用于 dataflow 进程解耦，保持 1 拍寄存即可，减少
    // 多个 512-bit 默认 shift FIFO 对 LUT 的占用。
    #pragma HLS stream variable=xlnq_buffered_stream depth=1
    #pragma HLS stream variable=xlns_buffered_stream depth=1
    #pragma HLS stream variable=aq_buffered_stream depth=1
    #pragma HLS stream variable=as_buffered_stream depth=1
    // MHA replays the same XLN tile for compact native-GQA K/Q/V projections; use BRAM for q data
    // to trade a small amount of block memory for lower K26 LUTRAM/CLB pressure.
    llm_xln_buffer_q.do_buffer_bram(true, MX_LLM_R_QKV, xlnq_stream, xlnq_buffered_stream);
    llm_xln_buffer_s.do_buffer(true, MX_LLM_R_QKV, xlns_stream, xlns_buffered_stream);
    llm_a_buffer_q.do_buffer_merge(true, MX_LLM_R_O, aq_stream, aq_buffered_stream);
    llm_a_buffer_s.do_buffer_merge(true, MX_LLM_R_O, as_stream, as_buffered_stream);
    llm_mha_gemm_mux(xlnq_buffered_stream, xlns_buffered_stream, aq_buffered_stream, as_buffered_stream,
                     q_stream, s_stream);
}

// LLM MLP pass：只启用 XLN 与 XM 两类输入，避免 MHA 的 A 流在第二个 pass 被误调度。
void run_llm_mlp_mux_layer(
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& xlnq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& xlns_stream,
    hls::stream<hls::vector<mx_aq_t, MX_CP           > >& xmq_stream,
    hls::stream<hls::vector<mx_as_t, 1               > >& xms_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    #pragma HLS dataflow
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> > xlnq_buffered_stream("llm_mlp_xlnq_buffered_stream");
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > > xlns_buffered_stream("llm_mlp_xlns_buffered_stream");
    hls::stream<hls::vector<mx_aq_t, MX_LLM_T * MX_CP> > xmq_buffered_stream("llm_mlp_xmq_buffered_stream");
    hls::stream<hls::vector<mx_as_t, MX_LLM_T        > > xms_buffered_stream("llm_mlp_xms_buffered_stream");
    // LLM MLP 两路都是稳定 II=1 流，局部 FIFO depth=1 足够保持 dataflow
    // 闭合，同时削掉默认 depth=2 的宽 shift register。
    #pragma HLS stream variable=xlnq_buffered_stream depth=1
    #pragma HLS stream variable=xlns_buffered_stream depth=1
    #pragma HLS stream variable=xmq_buffered_stream depth=1
    #pragma HLS stream variable=xms_buffered_stream depth=1
    // LLM MLP xlnq 放 BRAM；XM unpack 已有专用 BRAM buffer。该项每层常驻，
    // 相比 CLS-only buffer 更值得用 BRAM 换 SLICEM。
    llm_xln_buffer_q.do_buffer_bram(true, MX_LLM_R_M, xlnq_stream, xlnq_buffered_stream);
    llm_xln_buffer_s.do_buffer(true, MX_LLM_R_M, xlns_stream, xlns_buffered_stream);
    llm_do_buffer_unpack_bram<mx_aq_t, 1, MX_LLM_T, MX_TP_IN, MX_LLM_CM,  MX_CP>(true, MX_LLM_R_D, xmq_stream, xmq_buffered_stream);
    llm_do_buffer_unpack_bram<mx_as_t, 1, MX_LLM_T, MX_TP_IN, MX_LLM_CMT, 1    >(true, MX_LLM_R_D, xms_stream, xms_buffered_stream);
    llm_mlp_gemm_mux(xlnq_buffered_stream, xlns_buffered_stream, xmq_buffered_stream, xms_buffered_stream,
                     q_stream, s_stream);
}

// LLM mode：普通层跑 MHA+MLP 两个 pass；LLAMA_L 只跑 CLS/lm_head 的 XLN path。
void run_llm_mux(
    int l_begin,
    int l_close,
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& xlnq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& xlns_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& aq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& as_stream,
    hls::stream<hls::vector<mx_aq_t, MX_CP           > >& xmq_stream,
    hls::stream<hls::vector<mx_as_t, 1               > >& xms_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    // 保留 LLM 作为一个独立层级，避免 top 级 q/s 输出在 LLM 子 pass 和 ViT
    // 之间形成多路大 mux；top 只需要在 run_llm_mux 与 run_vit_mux 之间选择。
    #pragma HLS inline off
    for (int l = l_begin; l < l_close; ++l) {
        if (l < 0 || l > LLAMA_L) continue;
        if (l == LLAMA_L) {
            run_llm_cls_mux_layer(xlnq_stream, xlns_stream, q_stream, s_stream);
        } else {
            run_llm_mha_mux_layer(xlnq_stream, xlns_stream, aq_stream, as_stream, q_stream, s_stream);
            run_llm_mlp_mux_layer(xlnq_stream, xlns_stream, xmq_stream, xms_stream, q_stream, s_stream);
        }
    }
}

// ViT O/FC1/FC2 是纯 stream forward，单独函数让 dataflow 里的 producer/consumer 边界清晰。
void vit_forward_o(
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& aq_buf,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& as_buf,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    for (int n = 0; n < MX_VIT_R_O * MX_VIT_CT; ++n) {
        #pragma HLS pipeline II=1
        q_stream.write(aq_buf.read());
        s_stream.write(as_buf.read());
    }
}

void vit_forward_fc1(
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& xlnq_buf,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& xlns_buf,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    for (int n = 0; n < MX_VIT_R_M * MX_VIT_CT; ++n) {
        #pragma HLS pipeline II=1
        q_stream.write(xlnq_buf.read());
        s_stream.write(xlns_buf.read());
    }
}

void vit_forward_fc2(
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& xmq_buf,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& xms_buf,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    for (int n = 0; n < MX_VIT_R_D * MX_VIT_CMT; ++n) {
        #pragma HLS pipeline II=1
        q_stream.write(xmq_buf.read());
        s_stream.write(xms_buf.read());
    }
}

// ViT mode：MHA-QKV 先缓存全 T 的 XLN，再按 [h][qkv] 重放；O/FC1/FC2 按 tile 流式。
void run_vit_mux(
    int l_begin,
    int l_close,
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& xlnq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& xlns_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& aq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& as_stream,
    hls::stream<hls::vector<mx_aq_t, MX_CP           > >& xmq_stream,
    hls::stream<hls::vector<mx_as_t, 1               > >& xms_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    mx_as_pack_t xln_mha_s_tile[MX_VIT_CT];
    // CLB-02(b)：ViT MHA scale tile 是小而集中的 MUX LUTRAM 岛，
    // 用少量 BRAM 换掉 SLICEM 压力。
    #pragma HLS bind_storage variable=xln_mha_s_tile type=ram_2p impl=bram

    for (int l = l_begin; l < l_close; ++l) {
        if (l < 0 || l >= VIT_L) continue;

        // MHA-QKV Step 1：读入 MHA 前 XLN，流顺序为 token-major [tt][tp][ct]。
        for (int tt = 0; tt < MX_VIT_TT; ++tt) {
            for (int tp = 0; tp < MX_TP_TILE; ++tp) {
                for (int ct = 0; ct < MX_VIT_CT; ++ct) {
                    #pragma HLS pipeline II=1
                    hls::vector<mx_aq_t, MX_TP_IN * MX_CP> qv = xlnq_stream.read();
                    hls::vector<mx_as_t, MX_TP_IN        > sv = xlns_stream.read();
                    mx_as_pack_t s_pack = 0;
                    if (tp != 0) s_pack = xln_mha_s_tile[ct];
                    vit_xln_mha_buf_q[tp][tt][ct] = mx_pack_aq_cp(qv);
                    mx_set_as_lane(s_pack, tp, sv[0]);
                    xln_mha_s_tile[ct] = s_pack;
                }
            }
            for (int ct = 0; ct < MX_VIT_CT; ++ct) {
                #pragma HLS pipeline II=1
                vit_xln_mha_buf_s[tt][ct] = xln_mha_s_tile[ct];
            }
        }

        // MHA-QKV Step 2：按 PERMUTE 权重顺序重放 XLN。
        for (int h = 0; h < MX_VIT_H; ++h) {
            for (int qkv = 0; qkv < 3; ++qkv) {
                for (int tt = 0; tt < MX_VIT_TT; ++tt) {
                    for (int hct = 0; hct < MX_VIT_HCT; ++hct) {
                        for (int ct = 0; ct < MX_VIT_CT; ++ct) {
                            #pragma HLS pipeline II=1
                            hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> q_vec;
                            hls::vector<mx_as_t, MX_TP_TILE        > s_vec;
                            mx_as_pack_t s_pack = vit_xln_mha_buf_s[tt][ct];
                            for (int tp = 0; tp < MX_TP_TILE; ++tp) {
                                #pragma HLS unroll
                                hls::vector<mx_aq_t, MX_CP> q_lane;
                                mx_unpack_aq_cp(vit_xln_mha_buf_q[tp][tt][ct], q_lane);
                                for (int cp = 0; cp < MX_CP; ++cp) {
                                    #pragma HLS unroll
                                    q_vec[tp * MX_CP + cp] = q_lane[cp];
                                }
                                s_vec[tp] = mx_get_as_lane(s_pack, tp);
                            }
                            q_stream.write(q_vec);
                            s_stream.write(s_vec);
                        }
                    }
                }
            }
        }

        // O 投影：A 来自 RV_GEMM，按每个 tt 单独合并 H 个 head 后输出。
        // 保留本地转发 FIFO，隔离 tile buffer 与顶层 q/s 输出，避免直连后形成更差的时序路径。
        for (int tt = 0; tt < MX_VIT_TT; ++tt) {
            #pragma HLS dataflow
            hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> > aq_buf("vit_aq_buf");
            hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > > as_buf("vit_as_buf");
            // ViT O path 的局部转发 FIFO 只跨同速进程，depth=1 降低宽 FIFO LUT。
            #pragma HLS stream variable=aq_buf depth=1
            #pragma HLS stream variable=as_buf depth=1
            // ViT O/FC1 的 q buffer 是当前 MUX LUTRAM 大头，转 BRAM 以降低 CLB 占用。
            vit_a_buffer_q.do_buffer_merge_bram(true, MX_VIT_R_O, aq_stream, aq_buf);
            vit_a_buffer_s.do_buffer_merge(true, MX_VIT_R_O, as_stream, as_buf);
            vit_forward_o(aq_buf, as_buf, q_stream, s_stream);
        }

        // FC1 必须先输出所有 tt，再输出 FC2，不能与 FC2 按 tt 交错。
        // 本地转发 FIFO 与 O path 一致，用来保持 dataflow 边界和 post-synth 时序。
        for (int tt = 0; tt < MX_VIT_TT; ++tt) {
            #pragma HLS dataflow
            hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> > xlnq_buf("vit_xlnq_buf");
            hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > > xlns_buf("vit_xlns_buf");
            // FC1 转发 FIFO 不承担跨 tile 缓冲，1 拍寄存即可保持流顺序。
            #pragma HLS stream variable=xlnq_buf depth=1
            #pragma HLS stream variable=xlns_buf depth=1
            vit_fc1_xln_buf_q.do_buffer_bram(true, MX_VIT_R_M, xlnq_stream, xlnq_buf);
            vit_fc1_xln_buf_s.do_buffer(true, MX_VIT_R_M, xlns_stream, xlns_buf);
            vit_forward_fc1(xlnq_buf, xlns_buf, q_stream, s_stream);
        }

        // FC2 的 XM 输入是 CP 粒度 unpacked 流；buffer_unpack 完成 TP 聚合后直接输出给 GEMM。
        for (int tt = 0; tt < MX_VIT_TT; ++tt) {
            #pragma HLS dataflow
            hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> > xmq_buf("vit_xmq_buf");
            hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > > xms_buf("vit_xms_buf");
            // FC2 unpack 后立即按 GEMM-facing tile 输出，depth=1 避免默认宽 FIFO。
            #pragma HLS stream variable=xmq_buf depth=1
            #pragma HLS stream variable=xms_buf depth=1
            vit_xm_buf_q.do_buffer_unpack(true, MX_VIT_R_D, xmq_stream, xmq_buf);
            vit_xm_buf_s.do_buffer_unpack(true, MX_VIT_R_D, xms_stream, xms_buf);
            vit_forward_fc2(xmq_buf, xms_buf, q_stream, s_stream);
        }
    }
}

// top：统一 mode-aware MUX 接口。mode 在一次调用期间稳定，外层只选择 LLM/ViT 调度。
void top(
    REUSE_MODE_T mode,
    int l_begin,
    int l_close,
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& xlnq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& xlns_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& aq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& as_stream,
    hls::stream<hls::vector<mx_aq_t, MX_CP           > >& xmq_stream,
    hls::stream<hls::vector<mx_as_t, 1               > >& xms_stream,
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> >& q_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > >& s_stream
) {
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface ap_none port=mode
    #pragma HLS interface ap_none port=l_begin
    #pragma HLS interface ap_none port=l_close
    #pragma HLS interface axis port=xlnq_stream
    #pragma HLS interface axis port=xlns_stream
    #pragma HLS interface axis port=aq_stream
    #pragma HLS interface axis port=as_stream
    #pragma HLS interface axis port=xmq_stream
    #pragma HLS interface axis port=xms_stream
    #pragma HLS interface axis port=q_stream
    #pragma HLS interface axis port=s_stream
    #pragma HLS aggregate variable=xlnq_stream compact=bit
    #pragma HLS aggregate variable=xlns_stream compact=bit
    #pragma HLS aggregate variable=aq_stream compact=bit
    #pragma HLS aggregate variable=as_stream compact=bit
    #pragma HLS aggregate variable=xmq_stream compact=bit
    #pragma HLS aggregate variable=xms_stream compact=bit
    #pragma HLS aggregate variable=q_stream compact=bit
    #pragma HLS aggregate variable=s_stream compact=bit

    #pragma HLS array_partition variable=vit_xln_mha_buf_q dim=1 complete
    // ViT QKV replay 的关键路径来自 URAM 读口到 q_stream 输出寄存器；显式
    // latency=2 让 HLS 在 URAM 输出侧留寄存阶段，换少量 FF 获得时序余量。
    #pragma HLS bind_storage variable=vit_xln_mha_buf_q type=RAM_2P impl=URAM latency=2
    #pragma HLS bind_storage variable=vit_xln_mha_buf_s type=RAM_2P impl=URAM latency=2

    if (is_vit_mode(mode)) {
        run_vit_mux(l_begin, l_close, xlnq_stream, xlns_stream, aq_stream, as_stream,
                    xmq_stream, xms_stream, q_stream, s_stream);
    } else {
        run_llm_mux(l_begin, l_close, xlnq_stream, xlns_stream, aq_stream, as_stream,
                    xmq_stream, xms_stream, q_stream, s_stream);
    }
}

// ============================================================================
// Testbench
// ============================================================================
// ViT MUX 的 golden/output 张量接近 1GB。合并 LLM+ViT 后若放全局数组会让
// CSim 链接阶段 `.bss` 超过 x86_64 small code model 范围；testbench 因此统一
// 用 std::vector 放在 heap 上，综合 top 不依赖这些数据结构。

void test_llm_cls() {
    string save_path = CONDENSE_PATH + to_string(LLAMA_L);

    std::vector<int8_t> ref_cls_xln_q(MX_LLM_T * MX_LLM_C);
    std::vector<int8_t> ref_cls_xln_s(MX_LLM_T * MX_LLM_CT);
    std::vector<int8_t> ref_x_q_cls(MX_LLM_NUM_X_CLS);
    std::vector<int8_t> ref_x_s_cls(MX_LLM_NUM_WS_CLS);
    std::vector<int8_t> dut_x_q_cls(MX_LLM_NUM_X_CLS);
    std::vector<int8_t> dut_x_s_cls(MX_LLM_NUM_WS_CLS);

    auto xln_q = read_tensor<int8_t>(save_path + "/CONDENSED_XLN_Q.bin");
    auto xln_s = read_tensor<int8_t>(save_path + "/CONDENSED_XLN_S.bin");
    auto x_q = read_tensor<int8_t>(save_path + "/CONDENSED_GEMM_X_Q.bin");
    auto x_s = read_tensor<int8_t>(save_path + "/CONDENSED_GEMM_X_S.bin");

    tensor2array<int8_t>(xln_q, ref_cls_xln_q.data(), 1, 1, 1, 1, MX_LLM_T * MX_LLM_C,  MX_LLM_T * MX_LLM_C);
    tensor2array<int8_t>(xln_s, ref_cls_xln_s.data(), 1, 1, 1, 1, MX_LLM_T * MX_LLM_CT, MX_LLM_T * MX_LLM_CT);
    tensor2array<int8_t>(x_q,   ref_x_q_cls.data(),   1, 1, 1, 1, MX_LLM_NUM_X_CLS,     MX_LLM_NUM_X_CLS);
    tensor2array<int8_t>(x_s,   ref_x_s_cls.data(),   1, 1, 1, 1, MX_LLM_NUM_WS_CLS,    MX_LLM_NUM_WS_CLS);

    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> > xlnq_stream("llm_cls_xlnq_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > > xlns_stream("llm_cls_xlns_stream");
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> > aq_stream("llm_cls_aq_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > > as_stream("llm_cls_as_stream");
    hls::stream<hls::vector<mx_aq_t, MX_CP           > > xmq_stream("llm_cls_xmq_stream");
    hls::stream<hls::vector<mx_as_t, 1               > > xms_stream("llm_cls_xms_stream");
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> > q_stream("llm_cls_q_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > > s_stream("llm_cls_s_stream");

    array2stream<int8_t, mx_aq_t, 1, 1, 1, 1, MX_LLM_T * MX_LLM_C,  MX_TP_IN * MX_CP>(ref_cls_xln_q.data(), xlnq_stream, "LLM CLS XLN Q", true);
    array2stream<int8_t, mx_as_t, 1, 1, 1, 1, MX_LLM_T * MX_LLM_CT, MX_TP_IN        >(ref_cls_xln_s.data(), xlns_stream, "LLM CLS XLN S", true);

    top(MODE_LLM, LLAMA_L, LLAMA_L + 1, xlnq_stream, xlns_stream, aq_stream, as_stream,
        xmq_stream, xms_stream, q_stream, s_stream);

    stream2array<int8_t, mx_aq_t, 1, 1, 1, MX_LLM_NUM_X_CLS,  MX_TP_TILE * MX_CP>(q_stream, dut_x_q_cls.data(), "LLM CLS X Q", true);
    stream2array<int8_t, mx_as_t, 1, 1, 1, MX_LLM_NUM_WS_CLS, MX_TP_TILE        >(s_stream, dut_x_s_cls.data(), "LLM CLS X S", true);
    compare<int8_t>(ref_x_q_cls.data(), dut_x_q_cls.data(), MX_LLM_NUM_X_CLS,  "LLM CLS X Q", true);
    compare<int8_t>(ref_x_s_cls.data(), dut_x_s_cls.data(), MX_LLM_NUM_WS_CLS, "LLM CLS X S", true);
}

void test_llm_layer(int l) {
    string save_path = CONDENSE_PATH + to_string(l);

    std::vector<int8_t> ref_xln_q(2 * MX_LLM_T * MX_LLM_C);
    std::vector<int8_t> ref_xln_s(2 * MX_LLM_T * MX_LLM_CT);
    std::vector<int8_t> ref_mha_a_q(MX_LLM_T * MX_LLM_C);
    std::vector<int8_t> ref_mha_a_s(MX_LLM_T * MX_LLM_CT);
    std::vector<int8_t> ref_mlp_xm_q(MX_LLM_T * MX_LLM_CM);
    std::vector<int8_t> ref_mlp_xm_s(MX_LLM_T * MX_LLM_CMT);
    std::vector<int8_t> ref_x_q(MX_LLM_NUM_X);
    std::vector<int8_t> ref_x_s(MX_LLM_NUM_WS);
    std::vector<int8_t> dut_x_q(MX_LLM_NUM_X);
    std::vector<int8_t> dut_x_s(MX_LLM_NUM_WS);

    auto xln_q = read_tensor<int8_t>(save_path + "/CONDENSED_XLN_Q.bin");
    auto xln_s = read_tensor<int8_t>(save_path + "/CONDENSED_XLN_S.bin");
    auto a_q   = read_tensor<int8_t>(save_path + "/CONDENSED_RV_GEMM_A_Q.bin");
    auto a_s   = read_tensor<int8_t>(save_path + "/CONDENSED_RV_GEMM_A_S.bin");
    auto xm_q  = read_tensor<int8_t>(save_path + "/CONDENSED_SILU_EM_QUANT_XM_Q.bin");
    auto xm_s  = read_tensor<int8_t>(save_path + "/CONDENSED_SILU_EM_QUANT_XM_S.bin");
    auto x_q   = read_tensor<int8_t>(save_path + "/CONDENSED_GEMM_X_Q.bin");
    auto x_s   = read_tensor<int8_t>(save_path + "/CONDENSED_GEMM_X_S.bin");

    tensor2array<int8_t>(xln_q, ref_xln_q.data(),    1, 1, 1, 1, 2 * MX_LLM_T * MX_LLM_C,  2 * MX_LLM_T * MX_LLM_C);
    tensor2array<int8_t>(xln_s, ref_xln_s.data(),    1, 1, 1, 1, 2 * MX_LLM_T * MX_LLM_CT, 2 * MX_LLM_T * MX_LLM_CT);
    tensor2array<int8_t>(a_q,   ref_mha_a_q.data(),  1, 1, 1, 1,     MX_LLM_T * MX_LLM_C,      MX_LLM_T * MX_LLM_C);
    tensor2array<int8_t>(a_s,   ref_mha_a_s.data(),  1, 1, 1, 1,     MX_LLM_T * MX_LLM_CT,     MX_LLM_T * MX_LLM_CT);
    tensor2array<int8_t>(xm_q,  ref_mlp_xm_q.data(), 1, 1, 1, 1,     MX_LLM_T * MX_LLM_CM,     MX_LLM_T * MX_LLM_CM);
    tensor2array<int8_t>(xm_s,  ref_mlp_xm_s.data(), 1, 1, 1, 1,     MX_LLM_T * MX_LLM_CMT,    MX_LLM_T * MX_LLM_CMT);
    tensor2array<int8_t>(x_q,   ref_x_q.data(),      1, 1, 1, 1,     MX_LLM_NUM_X,            MX_LLM_NUM_X);
    tensor2array<int8_t>(x_s,   ref_x_s.data(),      1, 1, 1, 1,     MX_LLM_NUM_WS,           MX_LLM_NUM_WS);

    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> > xlnq_stream("llm_xlnq_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > > xlns_stream("llm_xlns_stream");
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> > aq_stream("llm_aq_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > > as_stream("llm_as_stream");
    hls::stream<hls::vector<mx_aq_t, MX_CP           > > xmq_stream("llm_xmq_stream");
    hls::stream<hls::vector<mx_as_t, 1               > > xms_stream("llm_xms_stream");
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> > q_stream("llm_q_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > > s_stream("llm_s_stream");

    array2stream<int8_t, mx_aq_t, 1, 1, 1, 1, 2 * MX_LLM_T * MX_LLM_C,  MX_TP_IN * MX_CP>(ref_xln_q.data(),    xlnq_stream, "LLM XLN Q", true);
    array2stream<int8_t, mx_as_t, 1, 1, 1, 1, 2 * MX_LLM_T * MX_LLM_CT, MX_TP_IN        >(ref_xln_s.data(),    xlns_stream, "LLM XLN S", true);
    array2stream<int8_t, mx_aq_t, 1, 1, 1, 1,     MX_LLM_T * MX_LLM_C,  MX_TP_IN * MX_CP>(ref_mha_a_q.data(),  aq_stream,   "LLM A Q", true);
    array2stream<int8_t, mx_as_t, 1, 1, 1, 1,     MX_LLM_T * MX_LLM_CT, MX_TP_IN        >(ref_mha_a_s.data(),  as_stream,   "LLM A S", true);
    array2stream<int8_t, mx_aq_t, 1, 1, 1, 1,     MX_LLM_T * MX_LLM_CM, MX_CP           >(ref_mlp_xm_q.data(), xmq_stream,  "LLM XM Q", true);
    array2stream<int8_t, mx_as_t, 1, 1, 1, 1,     MX_LLM_T * MX_LLM_CMT, 1              >(ref_mlp_xm_s.data(), xms_stream,  "LLM XM S", true);

    top(MODE_LLM, l, l + 1, xlnq_stream, xlns_stream, aq_stream, as_stream,
        xmq_stream, xms_stream, q_stream, s_stream);

    stream2array<int8_t, mx_aq_t, 1, 1, 1, MX_LLM_NUM_X,  MX_TP_TILE * MX_CP>(q_stream, dut_x_q.data(), "LLM X Q", true);
    stream2array<int8_t, mx_as_t, 1, 1, 1, MX_LLM_NUM_WS, MX_TP_TILE        >(s_stream, dut_x_s.data(), "LLM X S", true);
    compare<int8_t>(ref_x_q.data(), dut_x_q.data(), MX_LLM_NUM_X,  "LLM X Q", true);
    compare<int8_t>(ref_x_s.data(), dut_x_s.data(), MX_LLM_NUM_WS, "LLM X S", true);
}

void feed_vit_a_stream(
    const int8_t* ref_mha_a_q,
    const int8_t* ref_mha_a_s,
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> >& aq_stream,
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > >& as_stream
) {
    // RV_GEMM condensed A 按 [h][token][hct] 存储；ViT MUX 的 O tile buffer
    // 每个 tt 需要 [h][tp][hct] 顺序，所以 testbench 显式重排喂流。
    for (int tt = 0; tt < MX_VIT_TT; ++tt) {
        for (int h = 0; h < MX_VIT_H; ++h) {
            for (int tp = 0; tp < MX_TP_TILE; ++tp) {
                for (int hct = 0; hct < MX_VIT_HCT; ++hct) {
                    hls::vector<mx_aq_t, MX_TP_IN * MX_CP> aq_vec;
                    hls::vector<mx_as_t, MX_TP_IN        > as_vec;
                    for (int cp = 0; cp < MX_CP; ++cp) {
                        #pragma HLS unroll
                        aq_vec[cp] = ref_mha_a_q[h * MX_VIT_T * MX_VIT_HC
                                    + (tt * MX_TP_TILE + tp) * MX_VIT_HC
                                    + hct * MX_CP + cp];
                    }
                    as_vec[0] = ref_mha_a_s[h * MX_VIT_T * MX_VIT_HCT
                              + (tt * MX_TP_TILE + tp) * MX_VIT_HCT + hct];
                    aq_stream.write(aq_vec);
                    as_stream.write(as_vec);
                }
            }
        }
    }
}

void test_vit_layer(int l) {
    string condense_path = VIT_CONDENSE_PATH + to_string(l);

    std::vector<int8_t> ref_xln_q(2 * MX_VIT_T * MX_VIT_C);
    std::vector<int8_t> ref_xln_s(2 * MX_VIT_T * MX_VIT_CT);
    std::vector<int8_t> ref_mha_a_q(MX_VIT_H * MX_VIT_T * MX_VIT_HC);
    std::vector<int8_t> ref_mha_a_s(MX_VIT_H * MX_VIT_T * MX_VIT_HCT);
    std::vector<int8_t> ref_mlp_xm_q(MX_VIT_T * MX_VIT_CM);
    std::vector<int8_t> ref_mlp_xm_s(MX_VIT_T * MX_VIT_CMT);
    std::vector<int8_t> ref_x_q(MX_VIT_NUM_X);
    std::vector<int8_t> ref_x_s(MX_VIT_NUM_WS);
    std::vector<int8_t> dut_x_q(MX_VIT_NUM_X);
    std::vector<int8_t> dut_x_s(MX_VIT_NUM_WS);

    auto xln_q = read_tensor<int8_t>(condense_path + "/CONDENSED_XLN_Q.bin");
    auto xln_s = read_tensor<int8_t>(condense_path + "/CONDENSED_XLN_S.bin");
    auto a_q   = read_tensor<int8_t>(condense_path + "/CONDENSED_RV_GEMM_A_Q.bin");
    auto a_s   = read_tensor<int8_t>(condense_path + "/CONDENSED_RV_GEMM_A_S.bin");
    auto xm_q  = read_tensor<int8_t>(condense_path + "/CONDENSED_GELU_XM_Q.bin");
    auto xm_s  = read_tensor<int8_t>(condense_path + "/CONDENSED_GELU_XM_S.bin");
    auto x_q   = read_tensor<int8_t>(condense_path + "/CONDENSED_GEMM_X_Q.bin");
    auto x_s   = read_tensor<int8_t>(condense_path + "/CONDENSED_GEMM_X_S.bin");

    tensor2array<int8_t>(xln_q, ref_xln_q.data(),    1, 1, 1, 1, 2 * MX_VIT_T * MX_VIT_C,  2 * MX_VIT_T * MX_VIT_C);
    tensor2array<int8_t>(xln_s, ref_xln_s.data(),    1, 1, 1, 1, 2 * MX_VIT_T * MX_VIT_CT, 2 * MX_VIT_T * MX_VIT_CT);
    tensor2array<int8_t>(a_q,   ref_mha_a_q.data(),  1, 1, 1, 1,     MX_VIT_T * MX_VIT_C,      MX_VIT_T * MX_VIT_C);
    tensor2array<int8_t>(a_s,   ref_mha_a_s.data(),  1, 1, 1, 1,     MX_VIT_T * MX_VIT_CT,     MX_VIT_T * MX_VIT_CT);
    tensor2array<int8_t>(xm_q,  ref_mlp_xm_q.data(), 1, 1, 1, 1,     MX_VIT_T * MX_VIT_CM,     MX_VIT_T * MX_VIT_CM);
    tensor2array<int8_t>(xm_s,  ref_mlp_xm_s.data(), 1, 1, 1, 1,     MX_VIT_T * MX_VIT_CMT,    MX_VIT_T * MX_VIT_CMT);
    tensor2array<int8_t>(x_q,   ref_x_q.data(),      1, 1, 1, 1,     MX_VIT_NUM_X,            MX_VIT_NUM_X);
    tensor2array<int8_t>(x_s,   ref_x_s.data(),      1, 1, 1, 1,     MX_VIT_NUM_WS,           MX_VIT_NUM_WS);

    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> > xlnq_stream("vit_xlnq_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > > xlns_stream("vit_xlns_stream");
    hls::stream<hls::vector<mx_aq_t, MX_TP_IN * MX_CP> > aq_stream("vit_aq_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_IN        > > as_stream("vit_as_stream");
    hls::stream<hls::vector<mx_aq_t, MX_CP           > > xmq_stream("vit_xmq_stream");
    hls::stream<hls::vector<mx_as_t, 1               > > xms_stream("vit_xms_stream");
    hls::stream<hls::vector<mx_aq_t, MX_TP_TILE * MX_CP> > q_stream("vit_q_stream");
    hls::stream<hls::vector<mx_as_t, MX_TP_TILE        > > s_stream("vit_s_stream");

    array2stream<int8_t, mx_aq_t, 1, 1, 1, 1, 2 * MX_VIT_T * MX_VIT_C,  MX_TP_IN * MX_CP>(ref_xln_q.data(),    xlnq_stream, "ViT XLN Q", true);
    array2stream<int8_t, mx_as_t, 1, 1, 1, 1, 2 * MX_VIT_T * MX_VIT_CT, MX_TP_IN        >(ref_xln_s.data(),    xlns_stream, "ViT XLN S", true);
    feed_vit_a_stream(ref_mha_a_q.data(), ref_mha_a_s.data(), aq_stream, as_stream);
    array2stream<int8_t, mx_aq_t, 1, 1, 1, 1,     MX_VIT_T * MX_VIT_CM, MX_CP           >(ref_mlp_xm_q.data(), xmq_stream,  "ViT XM Q", true);
    array2stream<int8_t, mx_as_t, 1, 1, 1, 1,     MX_VIT_T * MX_VIT_CMT, 1              >(ref_mlp_xm_s.data(), xms_stream,  "ViT XM S", true);

    top(MODE_VIT, l, l + 1, xlnq_stream, xlns_stream, aq_stream, as_stream,
        xmq_stream, xms_stream, q_stream, s_stream);

    stream2array<int8_t, mx_aq_t, 1, 1, 1, MX_VIT_NUM_X,  MX_TP_TILE * MX_CP>(q_stream, dut_x_q.data(), "ViT X Q", true);
    stream2array<int8_t, mx_as_t, 1, 1, 1, MX_VIT_NUM_WS, MX_TP_TILE        >(s_stream, dut_x_s.data(), "ViT X S", true);

    compare<int8_t>(ref_x_q.data(), dut_x_q.data(), MX_VIT_NUM_X,  "ViT X Q", true);
    compare<int8_t>(ref_x_s.data(), dut_x_s.data(), MX_VIT_NUM_WS, "ViT X S", true);
}

#ifndef __SYNTHESIS__
int main() {
    int vit_layer = reuse_tb_vit_layer();
    if (!reuse_tb_only_vit()) {
        test_llm_layer(0);
        printf("LLM mode layer 0 passed\n");

        test_llm_cls();
        printf("LLM mode CLS passed\n");
    }

    test_vit_layer(vit_layer);
    printf("ViT mode layer %d passed\n", vit_layer);

    return 0;
}
#endif
