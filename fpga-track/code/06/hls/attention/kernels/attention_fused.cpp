#include "attention_fused.hpp"
#include "../common/attn_precision.hpp"
#include "../common/attn_exp_lut.hpp"

// A1-P1 fused attention engine (QK^T -> softmax -> AV), W=128 fixed-point.
//
// Architecture (VISION_ATTN_PL_SIZING.md §2.2): one head resident at a time.
//   * K_h / V_h are streamed once from DDR into on-chip caches whose D_HEAD
//     dimension is fully reshaped into a wide word (ARRAY_RESHAPE complete
//     dim=2), so a whole p2 row (dh=64 elements) is read in a single cycle.
//     This is what lets QK^T / AV run their inner loops at II=1 with a dh=64
//     wide reduction instead of the DDR-streamed 16-wide x 4-beat (II=4) form.
//   * QK^T: per p2 there is NO loop-carried dependency, so the 64-wide
//     multiply + reduction tree pipelines freely (II=1, low Tclk).
//   * softmax: row-max subtract, fp32 LUT exp (pipelined), fp32 sum with
//     rotating partial accumulators, fp32 normalize. All II=1.
//   * AV: reduction over p2 -> rotating partial accumulators per channel break
//     the fp32 add recurrence so the channel loop stays II=1 and < 3 ns.

namespace {

constexpr int M  = vh::N_POS;   // 1024
constexpr int C  = vh::C_EMB;   // 768
constexpr int H  = vh::N_HEAD;  // 12
constexpr int DH = vh::D_HEAD;  // 64

constexpr int EXP_LANES = 2;
constexpr int SUM_BANKS = 16;   // softmax fp32 partial sums (fp32 sum: discipline)
constexpr int MX_BANKS  = 8;    // QK row-max rotating partials (break recurrence)

using vh_attn::mac_t;
using vh_attn::acc_t;
using vh_attn::qk_acc_t;
using vh_attn::av_acc_t;
using vh_attn::prob_t;
using vh_attn::logit_t;
using vh_attn::to_mac;
using vh_attn::exp_lut_eval;

// ---- per-head on-chip caches (dh fully reshaped -> 1 wide read / cycle) -----
// Storage/reshape pragmas live in k_attention_fused (HLS pragmas are illegal at
// namespace scope); they reference these globals by name.
static mac_t K_cache[M][DH];
static mac_t V_cache[M][DH];

static void load_head(
    const float * K_ddr,
    const float * V_ddr,
    int           h)
{
    // The cache load is amortized over M=1024 queries per head, so it does NOT
    // need to be dh-wide. We convert with a single pipelined float->fixed lane
    // into a row buffer (CVT, II=1), then write the whole reshaped row in one
    // wide store (STORE) so the cache write width matches the dh=64 read width
    // in QK/AV. This avoids both the 128-converter LUT blow-up AND the narrow-
    // write/wide-read mismatch that fragments the URAM.
LOAD_KV:
    for (int p2 = 0; p2 < M; ++p2) {
        const int base = p2 * C + h * DH;
        mac_t krow[DH];
        mac_t vrow[DH];
#pragma HLS ARRAY_PARTITION variable=krow complete dim=1
#pragma HLS ARRAY_PARTITION variable=vrow complete dim=1
    CVT:
        for (int d = 0; d < DH; ++d) {
#pragma HLS PIPELINE II=1
            krow[d] = to_mac(K_ddr[base + d]);
            vrow[d] = to_mac(V_ddr[base + d]);
        }
    STORE:
        for (int d = 0; d < DH; ++d) {
#pragma HLS UNROLL
            K_cache[p2][d] = krow[d];
            V_cache[p2][d] = vrow[d];
        }
    }
}

static void qk_row(
    const mac_t q_mac[DH],
    logit_t     Lrow[M],
    logit_t   & row_max)
{
    // Rotating partial maxima break the distance-1 row-max recurrence. Without
    // this, mx (loop-carried) depends on the full QK datapath (mul + dh=64 tree
    // + convert), forcing all of it into one II=1 cycle (6.7 ns). With distance
    // = MX_BANKS the datapath pipelines freely across stages.
    logit_t mxb[MX_BANKS];
#pragma HLS ARRAY_PARTITION variable=mxb complete dim=1
    for (int b = 0; b < MX_BANKS; ++b) {
#pragma HLS UNROLL
        mxb[b] = (logit_t) -128;
    }

QK_P2:
    for (int p2 = 0; p2 < M; ++p2) {
#pragma HLS PIPELINE II=1
        // Stage products, then an explicit balanced binary reduction tree.
        // Each tree level is a separate combinational stage; with II=1 and no
        // loop-carried dependency HLS retimes pipeline registers between levels
        // so each stage stays < 3 ns (a flat 64-deep add chain was 6.7 ns).
        qk_acc_t t[DH];
#pragma HLS ARRAY_PARTITION variable=t complete dim=1
    QK_MUL:
        for (int d = 0; d < DH; ++d) {
#pragma HLS UNROLL
            mac_t p;
#pragma HLS BIND_OP variable=p op=mul impl=dsp
            p = q_mac[d] * K_cache[p2][d];
            t[d] = (qk_acc_t) p;
        }
    QK_TREE:
        for (int stride = DH / 2; stride >= 1; stride >>= 1) {
#pragma HLS UNROLL
            for (int i = 0; i < stride; ++i) {
#pragma HLS UNROLL
                t[i] += t[i + stride];
            }
        }
        // scale = 1/sqrt(64) = 1/8 = exact arithmetic right-shift in fixed point
        // (no float conversion -> keeps the logit on the fast fixed-point path).
        const logit_t s = (logit_t) (t[0] >> 3);
        Lrow[p2] = s;
        const int b = p2 & (MX_BANKS - 1);
        if (s > mxb[b]) mxb[b] = s;
    }

    logit_t mx = (logit_t) -128;
    for (int b = 0; b < MX_BANKS; ++b) {
#pragma HLS UNROLL
        if (mxb[b] > mx) mx = mxb[b];
    }
    row_max = mx;
}

static void softmax_row(const logit_t Lrow[M], logit_t mx, float Srow[M])
{
    // Stage 1 — pure map: exp datapath has no loop-carried dependency, so HLS
    // deeply pipelines ROM read + interpolation (II=1, low Tclk). Fixed-point
    // logit subtraction feeds the bit-slice index (no float->fixed barrel shift).
SM_EXP:
    for (int p2 = 0; p2 < M; p2 += EXP_LANES) {
#pragma HLS PIPELINE II=1
        const float e0 = exp_lut_eval((logit_t) (Lrow[p2]     - mx));
        const float e1 = exp_lut_eval((logit_t) (Lrow[p2 + 1] - mx));
        Srow[p2]     = e0;
        Srow[p2 + 1] = e1;
    }

    // Stage 2 — fp32 sum with rotating partial accumulators (carried distance
    // = SUM_BANKS > adder latency, so the recurrence keeps II=1 at < 3 ns).
    float psum[SUM_BANKS];
#pragma HLS ARRAY_PARTITION variable=psum complete dim=1
    for (int b = 0; b < SUM_BANKS; ++b) {
#pragma HLS UNROLL
        psum[b] = 0.0f;
    }
SM_SUM:
    for (int p2 = 0; p2 < M; ++p2) {
#pragma HLS PIPELINE II=1
        const int   b = p2 & (SUM_BANKS - 1);
        float       s = psum[b] + Srow[p2];
#pragma HLS BIND_OP variable=s op=fadd impl=fabric latency=10
        psum[b] = s;
    }
    float ssum = 0.0f;
    for (int b = 0; b < SUM_BANKS; ++b) {
#pragma HLS UNROLL
        ssum += psum[b];
    }

    const float inv = 1.0f / ssum;

SM_NORM:
    for (int p2 = 0; p2 < M; ++p2) {
#pragma HLS PIPELINE II=1
        Srow[p2] *= inv;
    }
}

static void av_row(
    const float Srow[M],
    int         h,
    av_acc_t    o_fp[DH])
{
    // Fixed-point per-channel accumulators. A fixed add is single-cycle, so the
    // distance-1 reduction recurrence still meets II=1 at < 3 ns — no rotating
    // banks (and no fp32 LUT cost) needed.
    av_acc_t acc[DH];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

    for (int d = 0; d < DH; ++d) {
#pragma HLS UNROLL
        acc[d] = 0;
    }

AV_P2:
    for (int p2 = 0; p2 < M; ++p2) {
#pragma HLS PIPELINE II=1
        const prob_t s = (prob_t) Srow[p2];
    AV_D:
        for (int d = 0; d < DH; ++d) {
#pragma HLS UNROLL
            av_acc_t prod;
#pragma HLS BIND_OP variable=prod op=mul impl=dsp
            prod = (av_acc_t) (s * V_cache[p2][d]);
            acc[d] += prod;
        }
    }

    // Keep the output fixed-point here; the float conversion is deferred to the
    // pipelined DDR write loop (1 converter, amortized) to avoid 64 wide
    // fixed->float converters on the AV output.
AV_REDUCE:
    for (int d = 0; d < DH; ++d) {
#pragma HLS UNROLL
        o_fp[d] = acc[d];
    }
}

}  // namespace

extern "C" void k_attention_fused(
    const float * Q,
    const float * K,
    const float * V,
    float       * O_out)
{
// depth only sizes cosim's AXI memory model / test-vector length; it does not
// affect the synthesized datapath. Overridable so small-slice cosim stays fast
// (production builds keep the canonical 786432 = 1024*768).
#ifndef VH_ATTN_AXI_DEPTH
#define VH_ATTN_AXI_DEPTH 786432
#endif
#pragma HLS INTERFACE m_axi port=Q     offset=slave bundle=gmem0 depth=VH_ATTN_AXI_DEPTH
#pragma HLS INTERFACE m_axi port=K     offset=slave bundle=gmem1 depth=VH_ATTN_AXI_DEPTH
#pragma HLS INTERFACE m_axi port=V     offset=slave bundle=gmem2 depth=VH_ATTN_AXI_DEPTH
#pragma HLS INTERFACE m_axi port=O_out offset=slave bundle=gmem3 depth=VH_ATTN_AXI_DEPTH
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS BIND_STORAGE variable=K_cache type=ram_s2p impl=uram
#pragma HLS BIND_STORAGE variable=V_cache type=ram_s2p impl=uram
#pragma HLS ARRAY_RESHAPE variable=K_cache complete dim=2
#pragma HLS ARRAY_RESHAPE variable=V_cache complete dim=2

    static logit_t Lrow[M];   // fixed-point QK logits (one row resident)
    static float   Srow[M];   // fp32 softmax probabilities (one row resident)
#pragma HLS BIND_STORAGE variable=Lrow type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=Srow type=ram_2p impl=bram

LOOP_HEAD:
    for (int h = 0; h < H; ++h) {
#pragma HLS LOOP_TRIPCOUNT min=12 max=12
        load_head(K, V, h);

    LOOP_P1:
        for (int p1 = 0; p1 < M; ++p1) {
#pragma HLS LOOP_TRIPCOUNT min=1024 max=1024
            const float * q = Q + (size_t) p1 * C + h * DH;
            mac_t q_mac[DH];
#pragma HLS ARRAY_PARTITION variable=q_mac complete dim=1
            for (int d = 0; d < DH; ++d) {
#pragma HLS UNROLL
                q_mac[d] = to_mac(q[d]);
            }

            logit_t mx;
            qk_row(q_mac, Lrow, mx);
            softmax_row(Lrow, mx, Srow);

            av_acc_t o_local[DH];
#pragma HLS ARRAY_PARTITION variable=o_local complete dim=1
            av_row(Srow, h, o_local);

            float * o = O_out + (size_t) p1 * C + h * DH;
            for (int d = 0; d < DH; ++d) {
#pragma HLS PIPELINE II=1
                o[d] = (float) o_local[d];
            }
        }
    }
}
