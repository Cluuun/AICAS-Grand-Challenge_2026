// testbench.cpp --- C-sim coverage for vlm_engine_v4 (csim_design only).
//
// This drives the top-level vlm_engine_v4 with the AXIS streams hooked to a
// behavioral PE (csim/pe_behavioral.hpp). Small cases keep scale exponents at
// zero so the engine reduces to Q20 fixed GEMM. Real-shape cases
// also exercise non-16B natural scale rows (vision K=768 -> 24 scale groups
// padded to 32 bytes) and FFN down strides.
//
// Coverage:
//   - GEMM_DENSE_O          M=2 and M=32 output_count=1
//   - GEMM_FUSED_QKV        M=2 and M=32 output_count=3
//   - GEMM_FUSED_TEXT_FFN   M=2 and M=32 K_in=128 K_im=128 K_out=32 SiLU
//   - GEMM_FUSED_VISION_FFN M=2 and M=32 K_in=128 K_im=128 K_out=32 GELU
//   - real-stride text QKV:   M=32 K=960  N={960,320,320}
//   - real-stride vision QKV: M=32 K=768  N={768,768,768}
//   - real-stride text FFN:   M=32 K_in=960 K_im=2560 K_out=960
//   - real-stride vision FFN: M=32 K_in=768 K_im=3072 K_out=768
#ifndef V4_CSIM
#define V4_CSIM
#endif
#include "accelerator.hpp"
#include "csim/pe_behavioral.hpp"
#include "../common/vlm_w8a8_abi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using v4::axi_word_t;
using v4::kAxiBytes;
using v4::kSaM;
using v4::kSaN;
using v4::kTileK;
using v4::kGroupSize;
using v4::kGroupsPerKtile;
using v4::kWeightColsPerPort;
using v4::kWeightWordsPerKtilePerPort;
using v4::kScaleWordsPerKtilePerPort;

static constexpr uint32_t ARENA_WORDS = 1u << 20; // 16 MB / 16 B = 1M words
static constexpr int kOutFrac = 20;
static constexpr int8_t kDisabledExp = -128;

static axi_word_t g_w0[ARENA_WORDS];
static axi_word_t g_w1[ARENA_WORDS];
static axi_word_t g_act[ARENA_WORDS];
static axi_word_t g_out[ARENA_WORDS];

static void reset_arenas() {
    std::memset(g_w0,  0, sizeof(g_w0));
    std::memset(g_w1,  0, sizeof(g_w1));
    std::memset(g_act, 0, sizeof(g_act));
    std::memset(g_out, 0, sizeof(g_out));
    v4_csim::pe_state().reset();
}

static void wb(axi_word_t *arena, uint64_t byte_addr, uint8_t val) {
    const uint64_t w = byte_addr / kAxiBytes;
    const uint32_t b = byte_addr % kAxiBytes;
    axi_word_t word = arena[w];
    word(b * 8 + 7, b * 8) = val;
    arena[w] = word;
}

static uint8_t rb(const axi_word_t *arena, uint64_t byte_addr) {
    const uint64_t w = byte_addr / kAxiBytes;
    const uint32_t b = byte_addr % kAxiBytes;
    return (uint8_t)arena[w](b * 8 + 7, b * 8);
}

static int32_t r_i32(const axi_word_t *arena, uint64_t byte_addr) {
    uint32_t v = 0;
    for (uint32_t i = 0; i < 4; ++i) v |= (uint32_t)rb(arena, byte_addr + i) << (8 * i);
    return (int32_t)v;
}

static uint64_t align_up_u64(uint64_t value, uint64_t align) {
    const uint64_t mask = align - 1;
    return (value + mask) & ~mask;
}

static void write_task(uint64_t off, const linear_task_t &t) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(&t);
    for (uint32_t i = 0; i < sizeof(t); ++i) wb(g_act, off + i, p[i]);
}

// Fill activation: row-major int8. `exp_out` is optional and contains the
// logical, unpadded per-row group exponents used by software golden checks.
static void fill_act(uint64_t q_off, uint64_t s_off, uint32_t rows, uint32_t K,
                     uint32_t row_stride, uint32_t scale_row_stride,
                     int8_t *a_out, int8_t *exp_out = nullptr,
                     bool varied_exp = false) {
    const uint32_t g_per_row = K / kGroupSize;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t k = 0; k < K; ++k) {
            int8_t v = (int8_t)((int)((m * 5 + k * 3) % 13) - 6);
            a_out[m * K + k] = v;
            wb(g_act, q_off + (uint64_t)m * row_stride + k, (uint8_t)v);
        }
        for (uint32_t g = 0; g < g_per_row; ++g) {
            const int8_t e = varied_exp ? (int8_t)((int)((m + g * 3) % 3) - 1) : (int8_t)0;
            if (exp_out != nullptr) {
                exp_out[m * g_per_row + g] = e;
            }
            wb(g_act, s_off + (uint64_t)m * scale_row_stride + g, (uint8_t)e);
        }
    }
}

static void fill_weight(uint32_t seed, uint32_t K, uint32_t N, int8_t *w) {
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t n = 0; n < N; ++n)
            w[k * N + n] = (int8_t)((int)((seed + k * 7 + n * 5) % 11) - 5);
}

// Pack weights to per-port v11 layout and unit APOT scales (e0=0, e1=disabled).
//   For each (n_tile, k_tile): per port:
//     2048 bytes = 128 axi words. lane in 0..15 maps to byte `lane` within
//     each 16-byte word at K-position `k_in_tile`. (Confirm vs weight_loader.)
//
// weight_loader.hpp computes:
//   const uint32_t k_words_per_lane = kTileK / kAxiBytes;  // 8
//   lane   = i / k_words_per_lane;
//   k_word = i % k_words_per_lane;
//   for b in 0..15: dst.bytes[k_word*16+b][port][lane] = v(8*b+7, 8*b)
// So word index `i = lane*8 + k_word`. Fine, host packs that way too.
static void pack_weight_ports(uint64_t q0_off, uint64_t q1_off,
                              uint64_t s0_off, uint64_t s1_off,
                              const int8_t *w, uint32_t K, uint32_t N,
                              uint32_t nt_stride_q, uint32_t nt_stride_s) {
    const uint32_t k_tiles = (K + kTileK - 1) / kTileK;
    const uint32_t n_tiles = (N + kSaN - 1) / kSaN;
    const uint32_t k_words_per_lane = kTileK / kAxiBytes; // 8
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            const uint64_t q0_kt = q0_off + (uint64_t)nt * nt_stride_q
                                + (uint64_t)kt * kWeightWordsPerKtilePerPort * kAxiBytes;
            const uint64_t q1_kt = q1_off + (uint64_t)nt * nt_stride_q
                                + (uint64_t)kt * kWeightWordsPerKtilePerPort * kAxiBytes;
            for (uint32_t lane = 0; lane < kWeightColsPerPort; ++lane) {
                for (uint32_t kw = 0; kw < k_words_per_lane; ++kw) {
                    const uint64_t word_off_bytes = ((uint64_t)lane * k_words_per_lane + kw) * kAxiBytes;
                    for (uint32_t b = 0; b < kAxiBytes; ++b) {
                        const uint32_t k_in_tile = kw * kAxiBytes + b;
                        const uint32_t k_abs = kt * kTileK + k_in_tile;
                        const uint32_t n0 = nt * kSaN + lane;
                        const uint32_t n1 = nt * kSaN + kWeightColsPerPort + lane;
                        const int8_t v0 = (k_abs < K && n0 < N) ? w[k_abs * N + n0] : 0;
                        const int8_t v1 = (k_abs < K && n1 < N) ? w[k_abs * N + n1] : 0;
                        wb(g_w0, q0_kt + word_off_bytes + b, (uint8_t)v0);
                        wb(g_w1, q1_kt + word_off_bytes + b, (uint8_t)v1);
                    }
                }
            }
            const uint64_t s0_kt = s0_off + (uint64_t)nt * nt_stride_s
                                + (uint64_t)kt * kScaleWordsPerKtilePerPort * kAxiBytes;
            const uint64_t s1_kt = s1_off + (uint64_t)nt * nt_stride_s
                                + (uint64_t)kt * kScaleWordsPerKtilePerPort * kAxiBytes;
            for (uint32_t lane = 0; lane < kWeightColsPerPort; ++lane) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    const uint64_t byte_off = ((uint64_t)lane * kGroupsPerKtile + g) * 2;
                    wb(g_w0, s0_kt + byte_off + 0, 0);
                    wb(g_w0, s0_kt + byte_off + 1, (uint8_t)kDisabledExp);
                    wb(g_w1, s1_kt + byte_off + 0, 0);
                    wb(g_w1, s1_kt + byte_off + 1, (uint8_t)kDisabledExp);
                }
            }
        }
    }
}

static void golden_gemm(const int8_t *a, const int8_t *w, int32_t *c,
                        uint32_t M, uint32_t K, uint32_t N) {
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            int32_t s = 0;
            for (uint32_t k = 0; k < K; ++k)
                s += (int32_t)a[m * K + k] * (int32_t)w[k * N + n];
            c[m * N + n] = s;
        }
}

static int32_t sat_i64(int64_t v) {
    if (v > 2147483647LL) return 2147483647;
    if (v < -2147483648LL) return (int32_t)0x80000000u;
    return (int32_t)v;
}

static int32_t sat_i128(__int128 v) {
    if (v > (__int128)2147483647LL) return 2147483647;
    if (v < -(__int128)2147483648LL) return (int32_t)0x80000000u;
    return (int32_t)v;
}

static int32_t sat_add_i32(int32_t a, int32_t b) {
    return sat_i64((int64_t)a + (int64_t)b);
}

static int32_t shift_apply_q20(int32_t x, int8_t e) {
    const int shift = (int)e + kOutFrac;
    int64_t wide = x;
    if (shift >= 0) {
        if (shift >= 31) {
            if (x > 0) return 2147483647;
            if (x < 0) return (int32_t)0x80000000u;
            return 0;
        }
        wide <<= shift;
    } else {
        const int rs = -shift;
        if (rs >= 63) {
            wide = 0;
        } else {
            const int64_t half = 1LL << (rs - 1);
            wide = (wide >= 0) ? (wide + half) : (wide - half);
            wide >>= rs;
        }
    }
    return sat_i64(wide);
}

static int32_t hard_silu_q20(int32_t x) {
    int32_t xp = x + (3 << kOutFrac);
    int32_t six_q20 = 6 << kOutFrac;
    int32_t cl = (xp < 0) ? 0 : ((xp > six_q20) ? six_q20 : xp);
    __int128 p  = (__int128)x * (__int128)cl;
    return sat_i128((p * (__int128)2796203LL) >> 44);
}

static int32_t hard_gelu_q20(int32_t x) {
    int32_t xp = x + (3 << (kOutFrac - 1));
    int32_t three_q20 = 3 << kOutFrac;
    int32_t cl = (xp < 0) ? 0 : ((xp > three_q20) ? three_q20 : xp);
    __int128 p  = (__int128)x * (__int128)cl;
    return sat_i128((p * (__int128)5592405LL) >> 44);
}

static int32_t round_mul_q20_ref(int32_t a, int32_t b) {
    int64_t p = (int64_t)a * (int64_t)b;
    const int64_t half = 1LL << (kOutFrac - 1);
    p = (p >= 0) ? (p + half) : (p - half);
    return sat_i64(p >> kOutFrac);
}

static int8_t choose_pot_exp_q20(int32_t max_abs) {
    if (max_abs <= 0) return 0;
    int selected_shift = kOutFrac;
    bool found = false;
    for (int shift = -10; shift <= 50; ++shift) {
        bool fits = false;
        if (shift >= 0) {
            fits = (shift >= 31) || ((int64_t)max_abs <= (127LL << shift));
        } else {
            fits = (((int64_t)max_abs << (-shift)) <= 127);
        }
        if (fits && !found) {
            selected_shift = shift;
            found = true;
        }
    }
    int exp = selected_shift - kOutFrac;
    if (exp < -30) exp = -30;
    if (exp > 30) exp = 30;
    return (int8_t)exp;
}

static int8_t quantize_q20_exp(int32_t x, int8_t exp) {
    const int shift = (int)exp + kOutFrac;
    int32_t q = 0;
    if (shift >= 0) {
        if (shift >= 31) {
            q = 0;
        } else if (shift == 0) {
            q = x;
        } else {
            const int32_t half = (int32_t)1 << (shift - 1);
            q = (x >= 0) ? ((x + half) >> shift) : ((x - half) >> shift);
        }
    } else {
        q = sat_i64((int64_t)x << (-shift));
    }
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

static void quantize32_q20(const int32_t *v, int8_t *q, int8_t &exp_out) {
    int32_t mx = 0;
    for (uint32_t i = 0; i < kSaN; ++i) {
        int32_t a = (v[i] < 0) ? -v[i] : v[i];
        if (a > mx) mx = a;
    }
    exp_out = choose_pot_exp_q20(mx);
    for (uint32_t i = 0; i < kSaN; ++i) {
        q[i] = quantize_q20_exp(v[i], exp_out);
    }
}

static void golden_gemm_scaled(const int8_t *a, const int8_t *a_exp,
                               const int8_t *w, int32_t *c,
                               uint32_t M, uint32_t K, uint32_t N) {
    const uint32_t k_tiles = (K + kTileK - 1) / kTileK;
    const uint32_t groups_per_row = (K + kGroupSize - 1) / kGroupSize;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            int32_t total = 0;
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                for (uint32_t gg = 0; gg < kGroupsPerKtile; ++gg) {
                    const uint32_t g = kt * kGroupsPerKtile + gg;
                    const uint32_t k0 = g * kGroupSize;
                    const uint32_t k1 = (k0 + kGroupSize < K) ? (k0 + kGroupSize) : K;
                    int32_t part = 0;
                    for (uint32_t k = k0; k < k1; ++k) {
                        part += (int32_t)a[m * K + k] * (int32_t)w[k * N + n];
                    }
                    const int8_t e = (a_exp != nullptr && g < groups_per_row) ?
                        a_exp[m * groups_per_row + g] : (int8_t)0;
                    total = sat_add_i32(total, shift_apply_q20(part, e));
                }
            }
            c[m * N + n] = total;
        }
    }
}

// FFN golden matches the Q20 fused-then-quantize-then-down contract.
static void golden_ffn(const int8_t *a, const int8_t *gw, const int8_t *uw,
                       const int8_t *dw, int32_t *out,
                       uint32_t M, uint32_t K_in, uint32_t K_im,
                       uint32_t N_out, uint32_t act) {
    std::vector<int8_t> mid_q((size_t)M * K_im, 0);
    const uint32_t mid_groups = (K_im + kGroupSize - 1) / kGroupSize;
    std::vector<int8_t> mid_e((size_t)M * mid_groups, 0);
    std::memset(out, 0, sizeof(int32_t) * M * N_out);

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t gmid = 0; gmid < mid_groups; ++gmid) {
            int32_t fused[kGroupSize];
            int32_t max_abs = 0;
            for (uint32_t c = 0; c < kGroupSize; ++c) {
                uint32_t mc = gmid * kGroupSize + c;
                int32_t ga = 0, ua = 0;
                if (mc < K_im) {
                    for (uint32_t k = 0; k < K_in; ++k) {
                        ga += (int32_t)a[m * K_in + k] * (int32_t)gw[k * K_im + mc];
                        ua += (int32_t)a[m * K_in + k] * (int32_t)uw[k * K_im + mc];
                    }
                    ga = shift_apply_q20(ga, 0);
                    ua = shift_apply_q20(ua, 0);
                }
                int32_t av = (act == LINEAR_FFN_ACT_SILU) ? hard_silu_q20(ga) : hard_gelu_q20(ga);
                fused[c] = round_mul_q20_ref(av, ua);
                int32_t abs_v = fused[c] < 0 ? -fused[c] : fused[c];
                if (mc < K_im && abs_v > max_abs) max_abs = abs_v;
            }
            const int8_t exp_v = choose_pot_exp_q20(max_abs);
            for (uint32_t c = 0; c < kGroupSize; ++c) {
                uint32_t mc = gmid * kGroupSize + c;
                if (mc >= K_im) continue;
                mid_q[(size_t)m * K_im + mc] = quantize_q20_exp(fused[c], exp_v);
            }
            mid_e[(size_t)m * mid_groups + gmid] = exp_v;
        }
    }

    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N_out; ++n) {
            int32_t total = 0;
            for (uint32_t gmid = 0; gmid < mid_groups; ++gmid) {
                int32_t part = 0;
                for (uint32_t c = 0; c < kGroupSize; ++c) {
                    uint32_t mc = gmid * kGroupSize + c;
                    if (mc < K_im) part += (int32_t)mid_q[(size_t)m * K_im + mc] * (int32_t)dw[mc * N_out + n];
                }
                total = sat_add_i32(total, shift_apply_q20(part, mid_e[(size_t)m * mid_groups + gmid]));
            }
            out[m * N_out + n] = total;
        }
}

static int compare_out(const char *name, uint64_t off,
                       const int32_t *gold, uint32_t rows, uint32_t cols,
                       uint32_t row_stride_bytes) {
    int err = 0;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t n = 0; n < cols; ++n) {
            int32_t hw  = r_i32(g_out, off + (uint64_t)m * row_stride_bytes + n * 4);
            int32_t ref = gold[m * cols + n];
            if (hw != ref) {
                if (err < 8) std::printf("  %s mismatch [%u,%u]: hw=%d ref=%d\n", name, m, n, hw, ref);
                ++err;
            }
        }
    }
    if (err == 0) std::printf("PASS %s\n", name);
    else          std::printf("FAIL %s err=%d\n", name, err);
    return err;
}

// =============================================================================
//   Test runners
// =============================================================================

static linear_task_t make_task(uint32_t task_type, uint32_t engine,
                               uint32_t rows, uint32_t K,
                               uint64_t act_q, uint64_t act_s,
                               uint32_t scale_groups) {
    linear_task_t t;
    std::memset(&t, 0, sizeof(t));
    t.magic = VLM_W8A8_TASK_MAGIC;
    t.version = VLM_W8A8_ABI_VERSION;
    t.task_type = task_type;
    t.engine = engine;
    t.rows = rows;
    t.input_cols = K;
    t.act_row_stride = ((K + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    t.act_scale_row_stride = ((scale_groups + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    t.act_q_offset_bytes = act_q;
    t.act_scale_offset_bytes = act_s;
    return t;
}

static int run_dense_case(const char *name, uint32_t task_type,
                          uint32_t rows, uint32_t outputs) {
    reset_arenas();
    const uint32_t K = 128, N = 32;
    const uint64_t task_off    = 0;
    const uint64_t act_off     = 4096;
    const uint64_t act_s_off   = 65536;
    // Per-output, per-port arena offsets.
    const uint64_t q0[3]       = { 131072, 196608, 262144 };
    const uint64_t q1[3]       = { 327680, 393216, 458752 };
    const uint64_t s0[3]       = { 524288, 540672, 557056 };
    const uint64_t s1[3]       = { 573440, 589824, 606208 };
    const uint64_t out_base[3] = { 1048576, 1114112, 1179648 };
    const uint32_t row_stride_out = ((N * 4 + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    const uint32_t scale_groups = K / kGroupSize; // 4

    int8_t a[kSaM * 128];
    int8_t w[3][128 * 32];
    int32_t gold[3][kSaM * 32];
    fill_act(act_off, act_s_off, rows, K,
             ((K + kAxiBytes - 1) / kAxiBytes) * kAxiBytes,
             ((scale_groups + kAxiBytes - 1) / kAxiBytes) * kAxiBytes, a);

    linear_task_t t = make_task(task_type, LINEAR_ENGINE_GEMM, rows, K, act_off, act_s_off, scale_groups);
    t.output_count = outputs;
    const uint32_t nt_q_stride = (kWeightWordsPerKtilePerPort * kAxiBytes); // 1 nt has k_tiles ktiles, but K=128 so 1 kt
    const uint32_t nt_s_stride = (kScaleWordsPerKtilePerPort  * kAxiBytes);
    t.weight_n_tile_stride_bytes        = nt_q_stride;
    t.weight_scale_n_tile_stride_bytes  = nt_s_stride;
    for (uint32_t o = 0; o < outputs; ++o) {
        t.out_cols[o] = N;
        t.weight_q_offset_bytes[o][0]     = q0[o];
        t.weight_q_offset_bytes[o][1]     = q1[o];
        t.weight_scale_offset_bytes[o][0] = s0[o];
        t.weight_scale_offset_bytes[o][1] = s1[o];
        t.dst_offset_bytes[o]             = out_base[o];
        t.output_row_stride_bytes[o]      = row_stride_out;
        fill_weight(o * 17 + task_type, K, N, w[o]);
        pack_weight_ports(q0[o], q1[o], s0[o], s1[o], w[o], K, N, nt_q_stride, nt_s_stride);
        golden_gemm_scaled(a, nullptr, w[o], gold[o], rows, K, N);
    }
    write_task(task_off, t);

    // Stream interfaces (consumed/produced by V4_CSIM_PUMP_PE inside the engine).
    v4::a_stream_t   a_st;
    v4::w_stream_t   w_st;
    v4::ctrl_stream_t c_st;
    v4::psum_stream_t p_st;

    vlm_engine_v4(g_w0, g_w1, g_act, g_out, task_off, a_st, w_st, c_st, p_st);

    int err = 0;
    for (uint32_t o = 0; o < outputs; ++o) {
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl), "%s o%u", name, o);
        err += compare_out(lbl, out_base[o], gold[o], rows, N, row_stride_out);
    }
    return err;
}

static int run_dense_real_case(const char *name, uint32_t task_type,
                               uint32_t rows, uint32_t K,
                               const uint32_t *out_cols,
                               uint32_t outputs,
                               bool varied_act_exponents) {
    reset_arenas();
    const uint64_t task_off = 0, act_off = 4096, act_s_off = 65536;
    const uint32_t scale_groups = K / kGroupSize;
    const uint32_t act_row_stride = ((K + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    const uint32_t act_scale_row_stride = ((scale_groups + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    const uint32_t k_tiles = (K + kTileK - 1) / kTileK;
    const uint32_t nt_q_stride = kWeightWordsPerKtilePerPort * kAxiBytes * k_tiles;
    const uint32_t nt_s_stride = kScaleWordsPerKtilePerPort  * kAxiBytes * k_tiles;

    std::vector<int8_t> a((size_t)rows * K);
    std::vector<int8_t> a_exp((size_t)rows * scale_groups, 0);
    fill_act(act_off, act_s_off, rows, K,
             act_row_stride, act_scale_row_stride,
             a.data(), a_exp.data(), varied_act_exponents);

    linear_task_t t = make_task(task_type, LINEAR_ENGINE_GEMM, rows, K, act_off, act_s_off, scale_groups);
    t.output_count = outputs;
    t.weight_n_tile_stride_bytes = nt_q_stride;
    t.weight_scale_n_tile_stride_bytes = nt_s_stride;

    uint64_t w0_cursor = 0, w1_cursor = 0, out_cursor = 0;
    uint64_t q0[3] = {0, 0, 0}, q1[3] = {0, 0, 0};
    uint64_t s0[3] = {0, 0, 0}, s1[3] = {0, 0, 0};
    uint64_t out_base[3] = {0, 0, 0};
    uint32_t out_stride[3] = {0, 0, 0};
    std::vector<std::vector<int8_t> > weights(outputs);
    std::vector<std::vector<int32_t> > gold(outputs);

    for (uint32_t o = 0; o < outputs; ++o) {
        const uint32_t N = out_cols[o];
        const uint32_t n_tiles = (N + kSaN - 1) / kSaN;
        q0[o] = align_up_u64(w0_cursor, 128); w0_cursor = q0[o] + (uint64_t)n_tiles * nt_q_stride;
        s0[o] = align_up_u64(w0_cursor, 128); w0_cursor = s0[o] + (uint64_t)n_tiles * nt_s_stride;
        q1[o] = align_up_u64(w1_cursor, 128); w1_cursor = q1[o] + (uint64_t)n_tiles * nt_q_stride;
        s1[o] = align_up_u64(w1_cursor, 128); w1_cursor = s1[o] + (uint64_t)n_tiles * nt_s_stride;

        out_stride[o] = ((N * 4 + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
        out_base[o] = align_up_u64(out_cursor, 128);
        out_cursor = out_base[o] + (uint64_t)rows * out_stride[o];

        weights[o].assign((size_t)K * N, 0);
        gold[o].assign((size_t)rows * N, 0);
        fill_weight(101 + task_type + o * 23, K, N, weights[o].data());
        pack_weight_ports(q0[o], q1[o], s0[o], s1[o], weights[o].data(), K, N, nt_q_stride, nt_s_stride);
        golden_gemm_scaled(a.data(), varied_act_exponents ? a_exp.data() : nullptr,
                           weights[o].data(), gold[o].data(), rows, K, N);

        t.out_cols[o] = N;
        t.weight_q_offset_bytes[o][0] = q0[o];
        t.weight_q_offset_bytes[o][1] = q1[o];
        t.weight_scale_offset_bytes[o][0] = s0[o];
        t.weight_scale_offset_bytes[o][1] = s1[o];
        t.dst_offset_bytes[o] = out_base[o];
        t.output_row_stride_bytes[o] = out_stride[o];
    }
    write_task(task_off, t);

    v4::a_stream_t a_st; v4::w_stream_t w_st;
    v4::ctrl_stream_t c_st; v4::psum_stream_t p_st;
    vlm_engine_v4(g_w0, g_w1, g_act, g_out, task_off, a_st, w_st, c_st, p_st);

    int err = 0;
    for (uint32_t o = 0; o < outputs; ++o) {
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl), "%s o%u", name, o);
        err += compare_out(lbl, out_base[o], gold[o].data(), rows, out_cols[o], out_stride[o]);
    }
    return err;
}

static int run_ffn_case(const char *name, uint32_t task_type, uint32_t act_kind,
                        uint32_t rows, uint32_t K_in, uint32_t K_im, uint32_t N_out) {
    reset_arenas();
    const uint64_t task_off = 0, act_off = 4096, act_s_off = 65536;
    uint64_t g_q0=131072, g_q1=196608, g_s0=262144, g_s1=278528;
    uint64_t u_q0=327680, u_q1=393216, u_s0=458752, u_s1=475136;
    uint64_t d_q0=524288, d_q1=589824, d_s0=655360, d_s1=671744;
    uint64_t out_off  = 1048576;
    const uint32_t row_stride_out = ((N_out * 4 + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    const uint32_t scale_groups = K_in / kGroupSize;

    const uint32_t kt_in = (K_in + kTileK - 1) / kTileK;
    const uint32_t kt_im = (K_im + kTileK - 1) / kTileK;
    const uint32_t nt_im = (K_im + kSaN - 1) / kSaN;
    const uint32_t nt_out = (N_out + kSaN - 1) / kSaN;
    const uint32_t nt_q_stride_in = kWeightWordsPerKtilePerPort * kAxiBytes * kt_in;
    const uint32_t nt_s_stride_in = kScaleWordsPerKtilePerPort  * kAxiBytes * kt_in;
    const uint32_t nt_q_stride_down = kWeightWordsPerKtilePerPort * kAxiBytes * kt_im;
    const uint32_t nt_s_stride_down = kScaleWordsPerKtilePerPort  * kAxiBytes * kt_im;

    if (K_in != 128 || K_im != 128 || N_out != 32) {
        uint64_t w0_cursor = 0, w1_cursor = 0;
        g_q0 = align_up_u64(w0_cursor, 128); w0_cursor = g_q0 + (uint64_t)nt_im * nt_q_stride_in;
        g_s0 = align_up_u64(w0_cursor, 128); w0_cursor = g_s0 + (uint64_t)nt_im * nt_s_stride_in;
        u_q0 = align_up_u64(w0_cursor, 128); w0_cursor = u_q0 + (uint64_t)nt_im * nt_q_stride_in;
        u_s0 = align_up_u64(w0_cursor, 128); w0_cursor = u_s0 + (uint64_t)nt_im * nt_s_stride_in;
        d_q0 = align_up_u64(w0_cursor, 128); w0_cursor = d_q0 + (uint64_t)nt_out * nt_q_stride_down;
        d_s0 = align_up_u64(w0_cursor, 128); w0_cursor = d_s0 + (uint64_t)nt_out * nt_s_stride_down;

        g_q1 = align_up_u64(w1_cursor, 128); w1_cursor = g_q1 + (uint64_t)nt_im * nt_q_stride_in;
        g_s1 = align_up_u64(w1_cursor, 128); w1_cursor = g_s1 + (uint64_t)nt_im * nt_s_stride_in;
        u_q1 = align_up_u64(w1_cursor, 128); w1_cursor = u_q1 + (uint64_t)nt_im * nt_q_stride_in;
        u_s1 = align_up_u64(w1_cursor, 128); w1_cursor = u_s1 + (uint64_t)nt_im * nt_s_stride_in;
        d_q1 = align_up_u64(w1_cursor, 128); w1_cursor = d_q1 + (uint64_t)nt_out * nt_q_stride_down;
        d_s1 = align_up_u64(w1_cursor, 128); w1_cursor = d_s1 + (uint64_t)nt_out * nt_s_stride_down;

        out_off = 0;
    }

    std::vector<int8_t> a((size_t)rows * K_in);
    std::vector<int8_t> gw((size_t)K_in * K_im);
    std::vector<int8_t> uw((size_t)K_in * K_im);
    std::vector<int8_t> dw((size_t)K_im * N_out);
    std::vector<int32_t> gold((size_t)rows * N_out);

    fill_act(act_off, act_s_off, rows, K_in,
             ((K_in + kAxiBytes - 1) / kAxiBytes) * kAxiBytes,
             ((scale_groups + kAxiBytes - 1) / kAxiBytes) * kAxiBytes, a.data());
    fill_weight(31 + task_type, K_in, K_im, gw.data());
    fill_weight(47 + task_type, K_in, K_im, uw.data());
    fill_weight(73 + task_type, K_im, N_out, dw.data());

    pack_weight_ports(g_q0, g_q1, g_s0, g_s1, gw.data(), K_in, K_im, nt_q_stride_in, nt_s_stride_in);
    pack_weight_ports(u_q0, u_q1, u_s0, u_s1, uw.data(), K_in, K_im, nt_q_stride_in, nt_s_stride_in);
    pack_weight_ports(d_q0, d_q1, d_s0, d_s1, dw.data(), K_im, N_out, nt_q_stride_down, nt_s_stride_down);
    golden_ffn(a.data(), gw.data(), uw.data(), dw.data(), gold.data(), rows, K_in, K_im, N_out, act_kind);

    linear_task_t t = make_task(task_type, LINEAR_ENGINE_GEMM, rows, K_in, act_off, act_s_off, scale_groups);
    t.ffn_activation = act_kind;
    t.ffn_input_cols = K_in;
    t.ffn_intermediate_cols = K_im;
    t.ffn_output_cols = N_out;
    t.weight_n_tile_stride_bytes = nt_q_stride_in;
    t.weight_scale_n_tile_stride_bytes = nt_s_stride_in;
    t.gate_q_offset_bytes[0] = g_q0; t.gate_q_offset_bytes[1] = g_q1;
    t.gate_scale_offset_bytes[0] = g_s0; t.gate_scale_offset_bytes[1] = g_s1;
    t.up_q_offset_bytes[0] = u_q0; t.up_q_offset_bytes[1] = u_q1;
    t.up_scale_offset_bytes[0] = u_s0; t.up_scale_offset_bytes[1] = u_s1;
    t.down_q_offset_bytes[0] = d_q0; t.down_q_offset_bytes[1] = d_q1;
    t.down_scale_offset_bytes[0] = d_s0; t.down_scale_offset_bytes[1] = d_s1;
    t.ffn_dst_offset_bytes = out_off;
    t.output_row_stride_bytes[0] = row_stride_out;
    write_task(task_off, t);

    v4::a_stream_t   a_st; v4::w_stream_t w_st;
    v4::ctrl_stream_t c_st; v4::psum_stream_t p_st;
    vlm_engine_v4(g_w0, g_w1, g_act, g_out, task_off, a_st, w_st, c_st, p_st);

    return compare_out(name, out_off, gold.data(), rows, N_out, row_stride_out);
}

int main() {
    std::printf("vlm_engine_v4 csim testbench (behavioral PE)\n");
    int err = 0;
    const uint32_t text_qkv_cols[3] = {960, 320, 320};
    const uint32_t vision_qkv_cols[3] = {768, 768, 768};
    const uint32_t vision_dense_cols[1] = {768};

    err += run_dense_case  ("GEMM_DENSE_O M2",        GEMM_DENSE_O,           2, 1);
    err += run_dense_case  ("GEMM_DENSE_O M32",       GEMM_DENSE_O,          32, 1);
    err += run_dense_case  ("GEMM_FUSED_QKV M2",      GEMM_FUSED_QKV,         2, 3);
    err += run_dense_case  ("GEMM_FUSED_QKV M32",     GEMM_FUSED_QKV,        32, 3);
    err += run_ffn_case    ("FFN_TEXT M2 SiLU",       GEMM_FUSED_TEXT_FFN,    LINEAR_FFN_ACT_SILU, 2, 128, 128, 32);
    err += run_ffn_case    ("FFN_TEXT M32 SiLU",      GEMM_FUSED_TEXT_FFN,    LINEAR_FFN_ACT_SILU, 32, 128, 128, 32);
    err += run_ffn_case    ("FFN_VISION M2 GELU",     GEMM_FUSED_VISION_FFN,  LINEAR_FFN_ACT_GELU, 2, 128, 128, 32);
    err += run_ffn_case    ("FFN_VISION M32 GELU",    GEMM_FUSED_VISION_FFN,  LINEAR_FFN_ACT_GELU, 32, 128, 128, 32);
    err += run_dense_real_case("REAL_TEXT_QKV M32",   GEMM_FUSED_QKV,        32, 960, text_qkv_cols, 3, false);
    err += run_dense_real_case("REAL_VISION_QKV M32", GEMM_FUSED_QKV,        32, 768, vision_qkv_cols, 3, true);
    err += run_dense_real_case("REAL_VISION_DENSE M32", GEMM_DENSE_O,        32, 768, vision_dense_cols, 1, true);
    err += run_ffn_case    ("REAL_TEXT_FFN M32 SiLU", GEMM_FUSED_TEXT_FFN,    LINEAR_FFN_ACT_SILU, 32, 960, 2560, 960);
    err += run_ffn_case    ("REAL_VISION_FFN M32 GELU", GEMM_FUSED_VISION_FFN, LINEAR_FFN_ACT_GELU, 32, 768, 3072, 768);

    if (err == 0) std::printf("v4 testbench: PASS\n");
    else          std::printf("v4 testbench: FAIL err=%d\n", err);
    return err == 0 ? 0 : 1;
}
