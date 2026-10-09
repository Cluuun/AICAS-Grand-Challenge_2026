#include "../src/reuse_common.h"
#include "../src/utils.h"

#ifndef __SYNTHESIS__
#include <cassert>
#include <iostream>
#endif

// ============================================================================
// ViT A memory-backed reorder
// ============================================================================
// 功能：
//   1. 该 IP 只承担 ViT mode 的跨阶段调度；LLM mode 在 Spinal 顶层旁路直连。
//   2. ViT mode 下把 RV_GEMM 的 A 输出从自然顺序 H -> token -> HCT 写入外部
//      DDR，再按 MUX O 投影消费顺序 TT -> H -> TP -> HCT 读回。
//   3. 该边界替代顶层全层 URAM buffer，避免为 1024x768 A 额外常驻约 25 个
//      URAM；代价转为每层一次 A 写回和一次 A replay 的 DDR 带宽。

constexpr int AR_CP       = REUSE_CP;
constexpr int AR_VIT_T    = VIT_S;
constexpr int AR_VIT_TP   = REUSE_GEMM_TP;
constexpr int AR_VIT_TT   = AR_VIT_T / AR_VIT_TP;
constexpr int AR_VIT_H    = VIT_H;
constexpr int AR_VIT_HC   = VIT_HC;
constexpr int AR_VIT_HCT  = AR_VIT_HC / AR_CP;
constexpr int AR_VIT_NUM  = AR_VIT_H * AR_VIT_T * AR_VIT_HCT;

typedef REUSE_AQ_T ar_q_t;
typedef REUSE_AS_T ar_s_t;
typedef hls::vector<ar_q_t, AR_CP> ar_q_vec_t;
typedef hls::vector<ar_s_t, 1>     ar_s_vec_t;
typedef ap_uint<128>               ar_pack_t;

static ar_pack_t pack_a(ar_q_vec_t q_vec, ar_s_vec_t s_vec) {
    #pragma HLS inline
    ar_pack_t pack = 0;
    for (int cp = 0; cp < AR_CP; ++cp) {
        #pragma HLS unroll
        ap_uint<ar_q_t::width> bits = q_vec[cp].range(ar_q_t::width - 1, 0);
        pack.range((cp + 1) * ar_q_t::width - 1, cp * ar_q_t::width) = bits;
    }
    pack.range(64 + ar_s_t::width - 1, 64) = s_vec[0].range(ar_s_t::width - 1, 0);
    return pack;
}

static void unpack_a(ar_pack_t pack, ar_q_vec_t &q_vec, ar_s_vec_t &s_vec) {
    #pragma HLS inline
    for (int cp = 0; cp < AR_CP; ++cp) {
        #pragma HLS unroll
        ap_uint<ar_q_t::width> bits = pack.range((cp + 1) * ar_q_t::width - 1, cp * ar_q_t::width);
        q_vec[cp] = (ar_q_t)bits;
    }
    ap_uint<ar_s_t::width> s_bits = pack.range(64 + ar_s_t::width - 1, 64);
    s_vec[0] = (ar_s_t)s_bits;
}

static void run_vit_store(
    ar_pack_t *memory_vit_a,
    hls::stream<ar_q_vec_t> &rv_aq_stream,
    hls::stream<ar_s_vec_t> &rv_as_stream
) {
    // RV_GEMM ViT 输出自然顺序为 H -> token -> HCT。这里线性写入 DDR，
    // 地址公式与后续 replay 的 H/token/HCT 索引保持一致。
    for (int h = 0; h < AR_VIT_H; ++h) {
        for (int t = 0; t < AR_VIT_T; ++t) {
            for (int hct = 0; hct < AR_VIT_HCT; ++hct) {
                #pragma HLS pipeline II=1
                int idx = h * AR_VIT_T * AR_VIT_HCT + t * AR_VIT_HCT + hct;
                memory_vit_a[idx] = pack_a(rv_aq_stream.read(), rv_as_stream.read());
            }
        }
    }
}

static void run_vit_replay(
    ar_pack_t *memory_vit_a,
    hls::stream<ar_q_vec_t> &mux_aq_stream,
    hls::stream<ar_s_vec_t> &mux_as_stream
) {
    // MUX O 阶段每个 TT 需要合并所有 head 的 8 个 token，再由 MUX 内部聚合成
    // TP x CP = 64 lanes 的 PERMUTE 输入 tile。
    for (int tt = 0; tt < AR_VIT_TT; ++tt) {
        for (int h = 0; h < AR_VIT_H; ++h) {
            for (int tp = 0; tp < AR_VIT_TP; ++tp) {
                for (int hct = 0; hct < AR_VIT_HCT; ++hct) {
                    #pragma HLS pipeline II=1
                    int t = tt * AR_VIT_TP + tp;
                    int idx = h * AR_VIT_T * AR_VIT_HCT + t * AR_VIT_HCT + hct;
                    ar_q_vec_t q_vec;
                    ar_s_vec_t s_vec;
                    unpack_a(memory_vit_a[idx], q_vec, s_vec);
                    mux_aq_stream.write(q_vec);
                    mux_as_stream.write(s_vec);
                }
            }
        }
    }
}

void top(
    ar_pack_t *memory_vit_a,
    hls::stream<ar_q_vec_t> &rv_aq_stream,
    hls::stream<ar_s_vec_t> &rv_as_stream,
    hls::stream<ar_q_vec_t> &mux_aq_stream,
    hls::stream<ar_s_vec_t> &mux_as_stream
) {
    // A_REORDER 是 ViT RV_GEMM 与 MUX 之间的调度边界；LLM 在 Spinal 顶层旁路直连。
    #pragma HLS interface ap_ctrl_chain port=return
    #pragma HLS interface mode=m_axi bundle=gmem1 port=memory_vit_a depth=AR_VIT_NUM offset=direct latency=0 max_read_burst_length=2 max_write_burst_length=2 num_read_outstanding=2 num_write_outstanding=2
    #pragma HLS interface axis port=rv_aq_stream
    #pragma HLS interface axis port=rv_as_stream
    #pragma HLS interface axis port=mux_aq_stream
    #pragma HLS interface axis port=mux_as_stream
    #pragma HLS aggregate variable=rv_aq_stream compact=bit
    #pragma HLS aggregate variable=rv_as_stream compact=bit
    #pragma HLS aggregate variable=mux_aq_stream compact=bit
    #pragma HLS aggregate variable=mux_as_stream compact=bit
    run_vit_store(memory_vit_a, rv_aq_stream, rv_as_stream);
    run_vit_replay(memory_vit_a, mux_aq_stream, mux_as_stream);
}

#ifndef __SYNTHESIS__
static int q_value(int h, int t, int hct, int cp) {
    return ((h * 17 + t * 5 + hct * 3 + cp) & 0xff) - 128;
}

static int s_value(int h, int t, int hct) {
    return (h + t + hct) & 0xf;
}

static void feed_vit(hls::stream<ar_q_vec_t> &q_stream, hls::stream<ar_s_vec_t> &s_stream) {
    for (int h = 0; h < AR_VIT_H; ++h) {
        for (int t = 0; t < AR_VIT_T; ++t) {
            for (int hct = 0; hct < AR_VIT_HCT; ++hct) {
                ar_q_vec_t q_vec;
                ar_s_vec_t s_vec;
                for (int cp = 0; cp < AR_CP; ++cp) {
                    q_vec[cp] = q_value(h, t, hct, cp);
                }
                s_vec[0] = s_value(h, t, hct);
                q_stream.write(q_vec);
                s_stream.write(s_vec);
            }
        }
    }
}

static void check_vit(hls::stream<ar_q_vec_t> &q_stream, hls::stream<ar_s_vec_t> &s_stream) {
    for (int tt = 0; tt < AR_VIT_TT; ++tt) {
        for (int h = 0; h < AR_VIT_H; ++h) {
            for (int tp = 0; tp < AR_VIT_TP; ++tp) {
                for (int hct = 0; hct < AR_VIT_HCT; ++hct) {
                    int t = tt * AR_VIT_TP + tp;
                    ar_q_vec_t q_vec = q_stream.read();
                    ar_s_vec_t s_vec = s_stream.read();
                    for (int cp = 0; cp < AR_CP; ++cp) {
                        assert((int)q_vec[cp] == q_value(h, t, hct, cp));
                    }
                    assert((int)s_vec[0] == s_value(h, t, hct));
                }
            }
        }
    }
}

static void test_layer() {
    static ar_pack_t memory_vit_a[AR_VIT_NUM];
    hls::stream<ar_q_vec_t> rv_aq_stream;
    hls::stream<ar_s_vec_t> rv_as_stream;
    hls::stream<ar_q_vec_t> mux_aq_stream;
    hls::stream<ar_s_vec_t> mux_as_stream;

    feed_vit(rv_aq_stream, rv_as_stream);
    top(memory_vit_a, rv_aq_stream, rv_as_stream, mux_aq_stream, mux_as_stream);
    check_vit(mux_aq_stream, mux_as_stream);
    std::cout << "A_REORDER test passed" << std::endl;
}

int main() {
    test_layer();
    return 0;
}
#endif
