#include "../src/reuse_common.h"
#include "../src/utils.h"

#ifndef __SYNTHESIS__
#include <cassert>
#include <iostream>
#endif

// ============================================================================
// ViT XM memory-backed scheduling boundary
// ============================================================================
// 功能：
//   1. 该 IP 只承担 ViT mode 的跨阶段调度；LLM mode 在 Spinal 顶层旁路直连。
//   2. ViT mode 下把 GELU/XM 从 TT -> CMT -> TP -> CP 写入外部 DDR scratch，
//      等 MUX 完整输出 FC1 后再按同一顺序 replay 给 FC2。
//   3. 该边界解除顶层 MLP 反馈环，避免用约 393K 拍 AXIS FIFO 承担跨阶段缓存。

constexpr int XR_CP       = REUSE_CP;
constexpr int XR_VIT_T    = VIT_S;
constexpr int XR_VIT_TP   = REUSE_GEMM_TP;
constexpr int XR_VIT_TT   = XR_VIT_T / XR_VIT_TP;
constexpr int XR_VIT_CM   = VIT_MLP_DIM;
constexpr int XR_VIT_CMT  = XR_VIT_CM / XR_CP;
constexpr int XR_VIT_NUM  = XR_VIT_T * XR_VIT_CMT;

typedef REUSE_AQ_T xr_q_t;
typedef REUSE_AS_T xr_s_t;
typedef hls::vector<xr_q_t, XR_CP> xr_q_vec_t;
typedef hls::vector<xr_s_t, 1>     xr_s_vec_t;
typedef ap_uint<128>               xr_pack_t;

static xr_pack_t pack_xm(xr_q_vec_t q_vec, xr_s_vec_t s_vec) {
    #pragma HLS inline
    xr_pack_t pack = 0;
    for (int cp = 0; cp < XR_CP; ++cp) {
        #pragma HLS unroll
        ap_uint<xr_q_t::width> bits = q_vec[cp].range(xr_q_t::width - 1, 0);
        pack.range((cp + 1) * xr_q_t::width - 1, cp * xr_q_t::width) = bits;
    }
    pack.range(64 + xr_s_t::width - 1, 64) = s_vec[0].range(xr_s_t::width - 1, 0);
    return pack;
}

static void unpack_xm(xr_pack_t pack, xr_q_vec_t &q_vec, xr_s_vec_t &s_vec) {
    #pragma HLS inline
    for (int cp = 0; cp < XR_CP; ++cp) {
        #pragma HLS unroll
        ap_uint<xr_q_t::width> bits = pack.range((cp + 1) * xr_q_t::width - 1, cp * xr_q_t::width);
        q_vec[cp] = (xr_q_t)bits;
    }
    ap_uint<xr_s_t::width> s_bits = pack.range(64 + xr_s_t::width - 1, 64);
    s_vec[0] = (xr_s_t)s_bits;
}

static void run_vit_store(
    xr_pack_t *memory_vit_xm,
    hls::stream<xr_q_vec_t> &silu_q_stream,
    hls::stream<xr_s_vec_t> &silu_s_stream
) {
    // SILU_GELU ViT 输出顺序为 TT -> CMT -> TP -> CP，线性写入即可保留 MUX FC2 需要的顺序。
    for (int tt = 0; tt < XR_VIT_TT; ++tt) {
        for (int cmt = 0; cmt < XR_VIT_CMT; ++cmt) {
            for (int tp = 0; tp < XR_VIT_TP; ++tp) {
                #pragma HLS pipeline II=1
                int idx = (tt * XR_VIT_CMT + cmt) * XR_VIT_TP + tp;
                memory_vit_xm[idx] = pack_xm(silu_q_stream.read(), silu_s_stream.read());
            }
        }
    }
}

static void run_vit_replay(
    xr_pack_t *memory_vit_xm,
    hls::stream<xr_q_vec_t> &mux_q_stream,
    hls::stream<xr_s_vec_t> &mux_s_stream
) {
    // MUX FC2 每个 tt 内按 CMT -> TP 消费 XM；与 store 顺序一致，只承担阶段解耦。
    for (int tt = 0; tt < XR_VIT_TT; ++tt) {
        for (int cmt = 0; cmt < XR_VIT_CMT; ++cmt) {
            for (int tp = 0; tp < XR_VIT_TP; ++tp) {
                #pragma HLS pipeline II=1
                int idx = (tt * XR_VIT_CMT + cmt) * XR_VIT_TP + tp;
                xr_q_vec_t q_vec;
                xr_s_vec_t s_vec;
                unpack_xm(memory_vit_xm[idx], q_vec, s_vec);
                mux_q_stream.write(q_vec);
                mux_s_stream.write(s_vec);
            }
        }
    }
}

void top(
    xr_pack_t *memory_vit_xm,
    hls::stream<xr_q_vec_t> &silu_q_stream,
    hls::stream<xr_s_vec_t> &silu_s_stream,
    hls::stream<xr_q_vec_t> &mux_q_stream,
    hls::stream<xr_s_vec_t> &mux_s_stream
) {
    // XM_REORDER 是 ViT SILU_GELU 与 MUX 之间的阶段调度边界；LLM 在 Spinal 顶层旁路直连。
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_vit_xm depth=XR_VIT_NUM offset=direct latency=0 max_read_burst_length=2 max_write_burst_length=2 num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS interface axis port=silu_q_stream
    #pragma HLS interface axis port=silu_s_stream
    #pragma HLS interface axis port=mux_q_stream
    #pragma HLS interface axis port=mux_s_stream
    #pragma HLS aggregate variable=silu_q_stream compact=bit
    #pragma HLS aggregate variable=silu_s_stream compact=bit
    #pragma HLS aggregate variable=mux_q_stream compact=bit
    #pragma HLS aggregate variable=mux_s_stream compact=bit
    run_vit_store(memory_vit_xm, silu_q_stream, silu_s_stream);
    run_vit_replay(memory_vit_xm, mux_q_stream, mux_s_stream);
}

#ifndef __SYNTHESIS__
static int q_value(int tt, int cmt, int tp, int cp) {
    return ((tt * 7 + cmt * 5 + tp * 3 + cp) & 0xff) - 128;
}

static int s_value(int tt, int cmt, int tp) {
    return (tt + cmt + tp) & 0xf;
}

static void feed_vit(hls::stream<xr_q_vec_t> &q_stream, hls::stream<xr_s_vec_t> &s_stream) {
    for (int tt = 0; tt < XR_VIT_TT; ++tt) {
        for (int cmt = 0; cmt < XR_VIT_CMT; ++cmt) {
            for (int tp = 0; tp < XR_VIT_TP; ++tp) {
                xr_q_vec_t q_vec;
                xr_s_vec_t s_vec;
                for (int cp = 0; cp < XR_CP; ++cp) {
                    q_vec[cp] = q_value(tt, cmt, tp, cp);
                }
                s_vec[0] = s_value(tt, cmt, tp);
                q_stream.write(q_vec);
                s_stream.write(s_vec);
            }
        }
    }
}

static void check_vit(hls::stream<xr_q_vec_t> &q_stream, hls::stream<xr_s_vec_t> &s_stream) {
    for (int tt = 0; tt < XR_VIT_TT; ++tt) {
        for (int cmt = 0; cmt < XR_VIT_CMT; ++cmt) {
            for (int tp = 0; tp < XR_VIT_TP; ++tp) {
                xr_q_vec_t q_vec = q_stream.read();
                xr_s_vec_t s_vec = s_stream.read();
                for (int cp = 0; cp < XR_CP; ++cp) {
                    assert((int)q_vec[cp] == q_value(tt, cmt, tp, cp));
                }
                assert((int)s_vec[0] == s_value(tt, cmt, tp));
            }
        }
    }
}

static void test_layer() {
    static xr_pack_t memory_vit_xm[XR_VIT_NUM];
    hls::stream<xr_q_vec_t> silu_q_stream;
    hls::stream<xr_s_vec_t> silu_s_stream;
    hls::stream<xr_q_vec_t> mux_q_stream;
    hls::stream<xr_s_vec_t> mux_s_stream;

    feed_vit(silu_q_stream, silu_s_stream);
    top(memory_vit_xm, silu_q_stream, silu_s_stream, mux_q_stream, mux_s_stream);
    check_vit(mux_q_stream, mux_s_stream);
    std::cout << "XM_REORDER test passed" << std::endl;
}

int main() {
    test_layer();
    return 0;
}
#endif
