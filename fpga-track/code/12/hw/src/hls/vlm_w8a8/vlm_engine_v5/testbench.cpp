// testbench.cpp --- C-sim coverage for vlm_engine_v5 (csim_design only).
//
// This drives the top-level vlm_engine_v5 with the AXIS streams hooked to a
// behavioral PE (csim/pe_behavioral.hpp). Small cases keep scale exponents at
// zero so the engine reduces to Q22 fixed GEMM. Real-shape cases
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
#ifndef V5_CSIM
#define V5_CSIM
#endif
#include "accelerator.hpp"
#include "csim/pe_behavioral.hpp"
#include "../common/vlm_w8a8_abi.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

using v5::axi_word_t;
using v5::kAxiBytes;
using v5::kSaM;
using v5::kSaN;
using v5::kTileK;
using v5::kGroupSize;
using v5::kGroupsPerKtile;
using v5::kWeightColsPerPort;
using v5::kWeightWordsPerKtilePerPort;
using v5::kScaleWordsPerKtilePerPort;
using v5::kScaleBytesPerSlot;

static constexpr uint32_t ARENA_WORDS = 1u << 20; // 16 MB / 16 B = 1M words
static constexpr int kOutFrac = 22;
static constexpr int kOutAccBits = 40;
static constexpr uint32_t kOutBytes = 8;
typedef int64_t ref_t;
static axi_word_t g_w0[ARENA_WORDS];
static axi_word_t g_w1[ARENA_WORDS];
static axi_word_t g_act[ARENA_WORDS];
static axi_word_t g_out[ARENA_WORDS];

static void reset_arenas() {
    std::memset(g_w0,  0, sizeof(g_w0));
    std::memset(g_w1,  0, sizeof(g_w1));
    std::memset(g_act, 0, sizeof(g_act));
    std::memset(g_out, 0, sizeof(g_out));
    v5_csim::pe_state().reset();
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

static int64_t r_i64(const axi_word_t *arena, uint64_t byte_addr) {
    uint64_t v = 0;
    for (uint32_t i = 0; i < 8; ++i) v |= (uint64_t)rb(arena, byte_addr + i) << (8 * i);
    return (int64_t)v;
}

static void w_i64(axi_word_t *arena, uint64_t byte_addr, int64_t value) {
    const uint64_t v = (uint64_t)value;
    for (uint32_t i = 0; i < 8; ++i) {
        wb(arena, byte_addr + i, (uint8_t)((v >> (8 * i)) & 0xffu));
    }
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
    const uint32_t g_per_row = (K + kGroupSize - 1) / kGroupSize;
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

// Pack weights to per-port v5 K-major layout and unit POT scales (e0=0).
//   For each (n_tile, k_tile): per port:
//     word[k] byte[lane] is W[k_abs, n_tile*32 + port_base + lane].
// This layout lets the HLS ping-pong buffer store one 128b word per K cycle
// instead of fully partitioning 16 lane-local mini memories.
static void pack_weight_ports(uint64_t q0_off, uint64_t q1_off,
                              uint64_t s0_off, uint64_t s1_off,
                              const int8_t *w, uint32_t K, uint32_t N,
                              uint32_t nt_stride_q, uint32_t nt_stride_s) {
    const uint32_t k_tiles = (K + kTileK - 1) / kTileK;
    const uint32_t n_tiles = (N + kSaN - 1) / kSaN;
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            const uint64_t q0_kt = q0_off + (uint64_t)nt * nt_stride_q
                                + (uint64_t)kt * kWeightWordsPerKtilePerPort * kAxiBytes;
            const uint64_t q1_kt = q1_off + (uint64_t)nt * nt_stride_q
                                + (uint64_t)kt * kWeightWordsPerKtilePerPort * kAxiBytes;
            for (uint32_t k = 0; k < kTileK; ++k) {
                const uint32_t k_abs = kt * kTileK + k;
                const uint64_t word_off_bytes = (uint64_t)k * kAxiBytes;
                for (uint32_t lane = 0; lane < kWeightColsPerPort; ++lane) {
                    const uint32_t n0 = nt * kSaN + lane;
                    const uint32_t n1 = nt * kSaN + kWeightColsPerPort + lane;
                    const int8_t v0 = (k_abs < K && n0 < N) ? w[k_abs * N + n0] : 0;
                    const int8_t v1 = (k_abs < K && n1 < N) ? w[k_abs * N + n1] : 0;
                    wb(g_w0, q0_kt + word_off_bytes + lane, (uint8_t)v0);
                    wb(g_w1, q1_kt + word_off_bytes + lane, (uint8_t)v1);
                }
            }
            const uint64_t s0_kt = s0_off + (uint64_t)nt * nt_stride_s
                                + (uint64_t)kt * kScaleWordsPerKtilePerPort * kAxiBytes;
            const uint64_t s1_kt = s1_off + (uint64_t)nt * nt_stride_s
                                + (uint64_t)kt * kScaleWordsPerKtilePerPort * kAxiBytes;
            for (uint32_t lane = 0; lane < kWeightColsPerPort; ++lane) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    const uint64_t byte_off =
                        ((uint64_t)lane * kGroupsPerKtile + g) * kScaleBytesPerSlot;
                    wb(g_w0, s0_kt + byte_off, 0);
                    wb(g_w1, s1_kt + byte_off, 0);
                }
            }
        }
    }
}

static ref_t sat_ref(__int128 v) {
    const __int128 max_v = (((__int128)1) << (kOutAccBits - 1)) - 1;
    const __int128 min_v = -(((__int128)1) << (kOutAccBits - 1));
    if (v > max_v) return (ref_t)max_v;
    if (v < min_v) return (ref_t)min_v;
    return (ref_t)v;
}

static ref_t round_shift_ref(__int128 v, int shift) {
    if (shift <= 0) return sat_ref(v);
    const __int128 half = ((__int128)1) << (shift - 1);
    if (v >= 0) return sat_ref((v + half) >> shift);
    return sat_ref(-(((-v) + half) >> shift));
}

static ref_t sat_add_ref(ref_t a, ref_t b) {
    return sat_ref((__int128)a + (__int128)b);
}

static ref_t shift_apply_q22(int32_t x, int8_t e) {
    const int shift = (int)e + kOutFrac;
    __int128 wide = x;
    if (shift >= 0) {
        wide <<= shift;
        return sat_ref(wide);
    }
    const int rs = -shift;
    if (rs >= 127) {
        return 0;
    }
    return round_shift_ref(wide, rs);
}

static ref_t fixed_from_i32_q22(int32_t x) {
    return sat_ref((__int128)x << kOutFrac);
}

static ref_t fixed_mul_div_q22(ref_t a, ref_t b, int divisor) {
    const __int128 recip = divisor == 3 ? 5592405 : 2796203;
    const __int128 v = (__int128)a * (__int128)b * recip;
    return sat_ref(v >> (kOutFrac + 24));
}

static ref_t hard_silu_q22(ref_t x) {
    const ref_t six = fixed_from_i32_q22(6);
    const ref_t three = fixed_from_i32_q22(3);
    ref_t gate = sat_add_ref(x, three);
    if (gate < 0) gate = 0;
    if (gate > six) gate = six;
    return fixed_mul_div_q22(x, gate, 6);
}

static ref_t hard_gelu_q22(ref_t x) {
    const ref_t three = fixed_from_i32_q22(3);
    const ref_t one_and_half = three >> 1;
    ref_t gate = sat_add_ref(x, one_and_half);
    if (gate < 0) gate = 0;
    if (gate > three) gate = three;
    return fixed_mul_div_q22(x, gate, 3);
}

static ref_t round_mul_q22_ref(ref_t a, ref_t b) {
    __int128 p = (__int128)a * (__int128)b;
    return round_shift_ref(p, kOutFrac);
}

static int8_t choose_pot_exp_q22(ref_t max_abs) {
    if (max_abs <= 0) return 0;
    int selected_shift = kOutFrac;
    bool found = false;
    for (int shift = -20; shift <= 62; ++shift) {
        bool fits = false;
        if (shift >= 0) {
            fits = ((__int128)max_abs <= ((__int128)127 << shift));
        } else {
            fits = (((__int128)max_abs << (-shift)) <= 127);
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

static int8_t quantize_q22_exp(ref_t x, int8_t exp) {
    const int shift = (int)exp + kOutFrac;
    ref_t q = 0;
    if (shift >= 0) {
        q = round_shift_ref((__int128)x, shift);
    } else {
        q = sat_ref((__int128)x << (-shift));
    }
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

static ref_t sat_q32_ref(__int128 v) {
    const __int128 max_v = (((__int128)1) << 47) - 1;
    const __int128 min_v = -(((__int128)1) << 47);
    if (v > max_v) return (ref_t)max_v;
    if (v < min_v) return (ref_t)min_v;
    return (ref_t)v;
}

static ref_t round_shift_q32_ref(__int128 v, int shift) {
    if (shift <= 0) return sat_q32_ref(v);
    const __int128 half = ((__int128)1) << (shift - 1);
    if (v >= 0) return sat_q32_ref((v + half) >> shift);
    return sat_q32_ref(-(((-v) + half) >> shift));
}

static ref_t gelu_slope_q16_ref(ref_t dy) {
    if (dy >= 0) return (dy + 3) / 6;
    return -(((-dy) + 3) / 6);
}

static ref_t gelu_lut128_q32_ref(ref_t x_q22) {
    static const ref_t y0[128] = {
        -25LL, -44LL, -77LL, -132LL, -224LL, -378LL, -631LL, -1045LL,
        -1715LL, -2790LL, -4499LL, -7190LL, -11391LL, -17887LL, -27840LL, -42947LL,
        -65668LL, -99521LL, -149490LL, -222561LL, -328411LL, -480301LL, -696201LL, -1000172LL,
        -1424060LL, -2009509LL, -2810301LL, -3895020LL, -5349982LL, -7282352LL, -9823303LL, -13131025LL,
        -17393304LL, -22829351LL, -29690452LL, -38258985LL, -48845276LL, -61781772LL, -77414015LL, -96087963LL,
        -118133348LL, -143842904LL, -173447583LL, -207088144LL, -244783837LL, -286399309LL, -331611170LL, -379876037LL,
        -430402117LL, -482126559LL, -533700851LL, -583486385LL, -629562026LL, -669745038LL, -701626040LL, -722617931LL,
        -730017800LL, -721079975LL, -693097480LL, -643488452LL, -569883480LL, -470209524LL, -342766022LL, -186289056LL,
        0LL, 216364128LL, 462540346LL, 737750028LL, 1040729256LL, 1369777468LL, 1722821624LL, 2097492313LL,
        2491207672LL, 2901260725LL, 3324905800LL, 3759439986LL, 4202276182LL, 4651005007LL, 5103443725LL, 5557671201LL,
        6012048827LL, 6465228091LL, 6916146142LL, 7364011187LL, 7808279843LL, 8248628720LL, 8684922465LL, 9117180328LL,
        9545543068LL, 9970241637LL, 10391568769LL, 10809854196LL, 11225443876LL, 11638683351LL, 12049905068LL, 12459419353LL,
        12867508584LL, 13274424047LL, 13680384953LL, 14085579088LL, 14490164642LL, 14894272788LL, 15298010691LL, 15701464667LL,
        16104703300LL, 16507780372LL, 16910737527LL, 17313606611LL, 17716411685LL, 18119170719LL, 18521896974LL, 18924600127LL,
        19327287164LL, 19729963069LL, 20132631360LL, 20535294497LL, 20937954177LL, 21340611562LL, 21743267437LL, 22145922330LL,
        22548576589LL, 22951230443LL, 23353884041LL, 23756537478LL, 24159190816LL, 24561844092LL, 24964497331LL, 25367150548LL,
    };
    static const ref_t dy[128] = {
        -19LL, -33LL, -55LL, -92LL, -154LL, -253LL, -414LL, -670LL,
        -1075LL, -1709LL, -2691LL, -4201LL, -6496LL, -9953LL, -15107LL, -22721LL,
        -33853LL, -49969LL, -73071LL, -105850LL, -151890LL, -215900LL, -303971LL, -423888LL,
        -585449LL, -800792LL, -1084719LL, -1454962LL, -1932370LL, -2540951LL, -3307722LL, -4262279LL,
        -5436047LL, -6861101LL, -8568533LL, -10586291LL, -12936496LL, -15632243LL, -18673948LL, -22045385LL,
        -25709556LL, -29604679LL, -33640561LL, -37695693LL, -41615472LL, -45211861LL, -48264867LL, -50526080LL,
        -51724442LL, -51574292LL, -49785534LL, -46075641LL, -40183012LL, -31881002LL, -20991891LL, -7399869LL,
        8937825LL, 27982495LL, 49609028LL, 73604972LL, 99673956LL, 127443502LL, 156476966LL, 186289056LL,
        216364128LL, 246176218LL, 275209682LL, 302979228LL, 329048212LL, 353044156LL, 374670689LL, 393715359LL,
        410053053LL, 423645075LL, 434534186LL, 442836196LL, 448728825LL, 452438718LL, 454227476LL, 454377626LL,
        453179264LL, 450918051LL, 447865045LL, 444268656LL, 440348877LL, 436293745LL, 432257863LL, 428362740LL,
        424698569LL, 421327132LL, 418285427LL, 415589680LL, 413239475LL, 411221717LL, 409514285LL, 408089231LL,
        406915463LL, 405960906LL, 405194135LL, 404585554LL, 404108146LL, 403737903LL, 403453976LL, 403238633LL,
        403077072LL, 402957155LL, 402869084LL, 402805074LL, 402759034LL, 402726255LL, 402703153LL, 402687037LL,
        402675905LL, 402668291LL, 402663137LL, 402659680LL, 402657385LL, 402655875LL, 402654893LL, 402654259LL,
        402653854LL, 402653598LL, 402653437LL, 402653338LL, 402653276LL, 402653239LL, 402653217LL, 402653203LL,
    };
    const ref_t min_x = -(6LL << kOutFrac);
    const ref_t max_x =  (6LL << kOutFrac);
    const ref_t step = (12LL << kOutFrac) / 128LL;
    if (x_q22 <= min_x) return 0;
    if (x_q22 >= max_x) return sat_q32_ref((__int128)x_q22 << (32 - kOutFrac));
    const ref_t rel = x_q22 - min_x;
    int idx = (int)(rel / step);
    if (idx < 0) idx = 0;
    if (idx > 127) idx = 127;
    const ref_t rem = rel - (ref_t)idx * step;
    return sat_q32_ref((__int128)y0[idx] +
                       round_shift_q32_ref((__int128)gelu_slope_q16_ref(dy[idx]) * rem, 16));
}

static int8_t choose_pot_exp_q32(ref_t max_abs) {
    if (max_abs <= 0) return 0;
    int selected_shift = 32;
    bool found = false;
    for (int shift = -20; shift <= 62; ++shift) {
        bool fits = false;
        if (shift >= 0) {
            fits = ((__int128)max_abs <= ((__int128)127 << shift));
        } else {
            fits = (((__int128)max_abs << (-shift)) <= 127);
        }
        if (fits && !found) {
            selected_shift = shift;
            found = true;
        }
    }
    int exp = selected_shift - 32;
    if (exp < -30) exp = -30;
    if (exp > 30) exp = 30;
    return (int8_t)exp;
}

static int8_t quantize_q32_exp(ref_t x, int8_t exp) {
    const int shift = (int)exp + 32;
    ref_t q = 0;
    if (shift >= 0) {
        q = round_shift_ref((__int128)x, shift);
    } else {
        q = sat_ref((__int128)x << (-shift));
    }
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

static void golden_gemm_scaled(const int8_t *a, const int8_t *a_exp,
                               const int8_t *w, ref_t *c,
                               uint32_t M, uint32_t K, uint32_t N) {
    const uint32_t k_tiles = (K + kTileK - 1) / kTileK;
    const uint32_t groups_per_row = (K + kGroupSize - 1) / kGroupSize;
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            ref_t total = 0;
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
                    total = sat_add_ref(total, shift_apply_q22(part, e));
                }
            }
            c[m * N + n] = total;
        }
    }
}

// FFN golden matches the Q22/int40 fused-then-quantize-then-down contract.
static void golden_ffn(const int8_t *a, const int8_t *gw, const int8_t *uw,
                       const int8_t *dw, ref_t *out,
                       uint32_t M, uint32_t K_in, uint32_t K_im,
                       uint32_t N_out, uint32_t act) {
    std::vector<int8_t> mid_q((size_t)M * K_im, 0);
    const uint32_t mid_groups = (K_im + kGroupSize - 1) / kGroupSize;
    std::vector<int8_t> mid_e((size_t)M * mid_groups, 0);
    std::memset(out, 0, sizeof(ref_t) * M * N_out);

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t gmid = 0; gmid < mid_groups; ++gmid) {
            ref_t fused[kGroupSize];
            ref_t max_abs = 0;
            for (uint32_t c = 0; c < kGroupSize; ++c) {
                uint32_t mc = gmid * kGroupSize + c;
                ref_t ga = 0, ua = 0;
                if (mc < K_im) {
                    int32_t ga_dot = 0, ua_dot = 0;
                    for (uint32_t k = 0; k < K_in; ++k) {
                        ga_dot += (int32_t)a[m * K_in + k] * (int32_t)gw[k * K_im + mc];
                        ua_dot += (int32_t)a[m * K_in + k] * (int32_t)uw[k * K_im + mc];
                    }
                    ga = shift_apply_q22(ga_dot, 0);
                    ua = shift_apply_q22(ua_dot, 0);
                }
                ref_t av = (act == LINEAR_FFN_ACT_SILU) ? hard_silu_q22(ga) : hard_gelu_q22(ga);
                fused[c] = round_mul_q22_ref(av, ua);
                ref_t abs_v = fused[c] < 0 ? -fused[c] : fused[c];
                if (mc < K_im && abs_v > max_abs) max_abs = abs_v;
            }
            const int8_t exp_v = choose_pot_exp_q22(max_abs);
            for (uint32_t c = 0; c < kGroupSize; ++c) {
                uint32_t mc = gmid * kGroupSize + c;
                if (mc >= K_im) continue;
                mid_q[(size_t)m * K_im + mc] = quantize_q22_exp(fused[c], exp_v);
            }
            mid_e[(size_t)m * mid_groups + gmid] = exp_v;
        }
    }

    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N_out; ++n) {
            ref_t total = 0;
            for (uint32_t gmid = 0; gmid < mid_groups; ++gmid) {
                int32_t part = 0;
                for (uint32_t c = 0; c < kGroupSize; ++c) {
                    uint32_t mc = gmid * kGroupSize + c;
                    if (mc < K_im) part += (int32_t)mid_q[(size_t)m * K_im + mc] * (int32_t)dw[mc * N_out + n];
                }
                total = sat_add_ref(total, shift_apply_q22(part, mid_e[(size_t)m * mid_groups + gmid]));
            }
            out[m * N_out + n] = total;
        }
}

static void golden_vision_mlp_bias(const int8_t *a, const int8_t *uw,
                                   const int8_t *dw, const ref_t *up_bias,
                                   const ref_t *down_bias, ref_t *out,
                                   uint32_t M, uint32_t K_in,
                                   uint32_t K_im, uint32_t N_out) {
    std::vector<int8_t> mid_q((size_t)M * K_im, 0);
    const uint32_t mid_groups = (K_im + kGroupSize - 1) / kGroupSize;
    std::vector<int8_t> mid_e((size_t)M * mid_groups, 0);
    std::memset(out, 0, sizeof(ref_t) * M * N_out);

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t gmid = 0; gmid < mid_groups; ++gmid) {
            ref_t fused[kGroupSize];
            ref_t max_abs = 0;
            for (uint32_t c = 0; c < kGroupSize; ++c) {
                const uint32_t mc = gmid * kGroupSize + c;
                ref_t acc = (mc < K_im) ? up_bias[mc] : 0;
                if (mc < K_im) {
                    int32_t dot = 0;
                    for (uint32_t k = 0; k < K_in; ++k) {
                        dot += (int32_t)a[m * K_in + k] * (int32_t)uw[k * K_im + mc];
                    }
                    acc = sat_add_ref(acc, shift_apply_q22(dot, 0));
                }
                fused[c] = gelu_lut128_q32_ref(acc);
                const ref_t abs_v = fused[c] < 0 ? -fused[c] : fused[c];
                if (mc < K_im && abs_v > max_abs) max_abs = abs_v;
            }
            const int8_t exp_v = choose_pot_exp_q32(max_abs);
            for (uint32_t c = 0; c < kGroupSize; ++c) {
                const uint32_t mc = gmid * kGroupSize + c;
                if (mc >= K_im) continue;
                mid_q[(size_t)m * K_im + mc] = quantize_q32_exp(fused[c], exp_v);
            }
            mid_e[(size_t)m * mid_groups + gmid] = exp_v;
        }
    }

    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N_out; ++n) {
            ref_t total = down_bias[n];
            for (uint32_t gmid = 0; gmid < mid_groups; ++gmid) {
                int32_t part = 0;
                for (uint32_t c = 0; c < kGroupSize; ++c) {
                    const uint32_t mc = gmid * kGroupSize + c;
                    if (mc < K_im) {
                        part += (int32_t)mid_q[(size_t)m * K_im + mc] * (int32_t)dw[mc * N_out + n];
                    }
                }
                total = sat_add_ref(total, shift_apply_q22(part, mid_e[(size_t)m * mid_groups + gmid]));
            }
            out[m * N_out + n] = total;
        }
    }
}

static int compare_out(const char *name, uint64_t off,
                       const ref_t *gold, uint32_t rows, uint32_t cols,
                       uint32_t row_stride_bytes) {
    int err = 0;
    for (uint32_t m = 0; m < rows; ++m) {
        for (uint32_t n = 0; n < cols; ++n) {
            ref_t hw  = r_i64(g_out, off + (uint64_t)m * row_stride_bytes + n * kOutBytes);
            ref_t ref = gold[m * cols + n];
            if (hw != ref) {
                if (err < 8) std::printf("  %s mismatch [%u,%u]: hw=%lld ref=%lld\n", name, m, n, (long long)hw, (long long)ref);
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
                          uint32_t rows, uint32_t outputs,
                          bool use_bias = false) {
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
    const uint64_t b0[3]       = { 655360, 655616, 655872 };
    const uint64_t out_base[3] = { 1048576, 1114112, 1179648 };
    const uint32_t row_stride_out = ((N * kOutBytes + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    const uint32_t scale_groups = (K + kGroupSize - 1) / kGroupSize;

    std::vector<int8_t> a((size_t)rows * K);
    int8_t w[3][128 * 32];
    ref_t bias[3][32];
    std::vector<ref_t> gold[3];
    fill_act(act_off, act_s_off, rows, K,
             ((K + kAxiBytes - 1) / kAxiBytes) * kAxiBytes,
             ((scale_groups + kAxiBytes - 1) / kAxiBytes) * kAxiBytes, a.data());

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
        gold[o].assign((size_t)rows * N, 0);
        golden_gemm_scaled(a.data(), nullptr, w[o], gold[o].data(), rows, K, N);
        for (uint32_t n = 0; n < N; ++n) {
            bias[o][n] = use_bias ? fixed_from_i32_q22((int32_t)((int)((o + 1) * (n % 7)) - 9)) : 0;
            if (use_bias) {
                w_i64(g_act, b0[o] + (uint64_t)n * kOutBytes, bias[o][n]);
            }
        }
        if (use_bias) {
            t.bias_offset_bytes[o] = b0[o];
            for (uint32_t m = 0; m < rows; ++m) {
                for (uint32_t n = 0; n < N; ++n) {
                    gold[o][m * N + n] += bias[o][n];
                }
            }
        }
    }
    write_task(task_off, t);

    // Stream interfaces (consumed/produced by the v5 C-sim behavioral PE).
    v5::a_stream_t   a_st;
    v5::w_stream_t   w_st;
    v5::ctrl_stream_t c_st;
    v5::psum_stream_t p_st;

    vlm_engine_v5(g_w0, g_w1, g_act, g_out, task_off, a_st, w_st, c_st, p_st);

    int err = 0;
    for (uint32_t o = 0; o < outputs; ++o) {
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl), "%s o%u", name, o);
        err += compare_out(lbl, out_base[o], gold[o].data(), rows, N, row_stride_out);
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
    const uint32_t scale_groups = (K + kGroupSize - 1) / kGroupSize;
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
    std::vector<std::vector<ref_t> > gold(outputs);

    for (uint32_t o = 0; o < outputs; ++o) {
        const uint32_t N = out_cols[o];
        const uint32_t n_tiles = (N + kSaN - 1) / kSaN;
        q0[o] = align_up_u64(w0_cursor, 128); w0_cursor = q0[o] + (uint64_t)n_tiles * nt_q_stride;
        s0[o] = align_up_u64(w0_cursor, 128); w0_cursor = s0[o] + (uint64_t)n_tiles * nt_s_stride;
        q1[o] = align_up_u64(w1_cursor, 128); w1_cursor = q1[o] + (uint64_t)n_tiles * nt_q_stride;
        s1[o] = align_up_u64(w1_cursor, 128); w1_cursor = s1[o] + (uint64_t)n_tiles * nt_s_stride;

        out_stride[o] = ((N * kOutBytes + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
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

    v5::a_stream_t a_st; v5::w_stream_t w_st;
    v5::ctrl_stream_t c_st; v5::psum_stream_t p_st;
    vlm_engine_v5(g_w0, g_w1, g_act, g_out, task_off, a_st, w_st, c_st, p_st);

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
    const uint32_t row_stride_out = ((N_out * kOutBytes + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    const uint32_t scale_groups = (K_in + kGroupSize - 1) / kGroupSize;

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
    std::vector<ref_t> gold((size_t)rows * N_out);

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

    v5::a_stream_t   a_st; v5::w_stream_t w_st;
    v5::ctrl_stream_t c_st; v5::psum_stream_t p_st;
    vlm_engine_v5(g_w0, g_w1, g_act, g_out, task_off, a_st, w_st, c_st, p_st);

    return compare_out(name, out_off, gold.data(), rows, N_out, row_stride_out);
}

static int run_vision_mlp_bias_case(const char *name,
                                    uint32_t rows, uint32_t K_in,
                                    uint32_t K_im, uint32_t N_out,
                                    bool zero_bias) {
    reset_arenas();
    const uint64_t task_off = 0, act_off = 4096, act_s_off = 65536;
    const uint64_t up_b_off = 131072;
    const uint64_t down_b_off = align_up_u64(up_b_off + (uint64_t)VLM_W8A8_VISION_FFN * kOutBytes, 128);
    uint64_t u_q0=0, u_q1=0, u_s0=0, u_s1=0;
    uint64_t d_q0=0, d_q1=0, d_s0=0, d_s1=0;
    uint64_t out_off = 0;
    const uint32_t row_stride_out = ((N_out * kOutBytes + kAxiBytes - 1) / kAxiBytes) * kAxiBytes;
    const uint32_t scale_groups = (K_in + kGroupSize - 1) / kGroupSize;
    const uint32_t kt_in = (K_in + kTileK - 1) / kTileK;
    const uint32_t kt_im = (K_im + kTileK - 1) / kTileK;
    const uint32_t nt_im = (K_im + kSaN - 1) / kSaN;
    const uint32_t nt_out = (N_out + kSaN - 1) / kSaN;
    const uint32_t nt_q_stride_in = kWeightWordsPerKtilePerPort * kAxiBytes * kt_in;
    const uint32_t nt_s_stride_in = kScaleWordsPerKtilePerPort  * kAxiBytes * kt_in;
    const uint32_t nt_q_stride_down = kWeightWordsPerKtilePerPort * kAxiBytes * kt_im;
    const uint32_t nt_s_stride_down = kScaleWordsPerKtilePerPort  * kAxiBytes * kt_im;

    uint64_t w0_cursor = 0, w1_cursor = 0;
    u_q0 = align_up_u64(w0_cursor, 128); w0_cursor = u_q0 + (uint64_t)nt_im * nt_q_stride_in;
    u_s0 = align_up_u64(w0_cursor, 128); w0_cursor = u_s0 + (uint64_t)nt_im * nt_s_stride_in;
    d_q0 = align_up_u64(w0_cursor, 128); w0_cursor = d_q0 + (uint64_t)nt_out * nt_q_stride_down;
    d_s0 = align_up_u64(w0_cursor, 128); w0_cursor = d_s0 + (uint64_t)nt_out * nt_s_stride_down;
    u_q1 = align_up_u64(w1_cursor, 128); w1_cursor = u_q1 + (uint64_t)nt_im * nt_q_stride_in;
    u_s1 = align_up_u64(w1_cursor, 128); w1_cursor = u_s1 + (uint64_t)nt_im * nt_s_stride_in;
    d_q1 = align_up_u64(w1_cursor, 128); w1_cursor = d_q1 + (uint64_t)nt_out * nt_q_stride_down;
    d_s1 = align_up_u64(w1_cursor, 128); w1_cursor = d_s1 + (uint64_t)nt_out * nt_s_stride_down;

    std::vector<int8_t> a((size_t)rows * K_in);
    std::vector<int8_t> uw((size_t)K_in * K_im);
    std::vector<int8_t> dw((size_t)K_im * N_out);
    std::vector<ref_t> up_bias((size_t)K_im, 0);
    std::vector<ref_t> down_bias((size_t)N_out, 0);
    std::vector<ref_t> gold((size_t)rows * N_out);

    fill_act(act_off, act_s_off, rows, K_in,
             ((K_in + kAxiBytes - 1) / kAxiBytes) * kAxiBytes,
             ((scale_groups + kAxiBytes - 1) / kAxiBytes) * kAxiBytes, a.data());
    fill_weight(151, K_in, K_im, uw.data());
    fill_weight(173, K_im, N_out, dw.data());
    for (uint32_t i = 0; i < K_im; ++i) {
        up_bias[i] = zero_bias ? 0 : fixed_from_i32_q22((int32_t)((int)(i % 7) - 3));
        w_i64(g_act, up_b_off + (uint64_t)i * kOutBytes, up_bias[i]);
    }
    for (uint32_t i = 0; i < N_out; ++i) {
        down_bias[i] = zero_bias ? 0 : fixed_from_i32_q22((int32_t)((int)(i % 5) - 2));
        w_i64(g_act, down_b_off + (uint64_t)i * kOutBytes, down_bias[i]);
    }

    pack_weight_ports(u_q0, u_q1, u_s0, u_s1, uw.data(), K_in, K_im, nt_q_stride_in, nt_s_stride_in);
    pack_weight_ports(d_q0, d_q1, d_s0, d_s1, dw.data(), K_im, N_out, nt_q_stride_down, nt_s_stride_down);
    golden_vision_mlp_bias(a.data(), uw.data(), dw.data(),
                           up_bias.data(), down_bias.data(), gold.data(),
                           rows, K_in, K_im, N_out);

    linear_task_t t = make_task(GEMM_FUSED_VISION_MLP_GELU_BIAS, LINEAR_ENGINE_GEMM,
                                rows, K_in, act_off, act_s_off, scale_groups);
    t.ffn_activation = LINEAR_FFN_ACT_GELU;
    t.ffn_input_cols = K_in;
    t.ffn_intermediate_cols = K_im;
    t.ffn_output_cols = N_out;
    t.weight_n_tile_stride_bytes = nt_q_stride_in;
    t.weight_scale_n_tile_stride_bytes = nt_s_stride_in;
    t.up_q_offset_bytes[0] = u_q0; t.up_q_offset_bytes[1] = u_q1;
    t.up_scale_offset_bytes[0] = u_s0; t.up_scale_offset_bytes[1] = u_s1;
    t.down_q_offset_bytes[0] = d_q0; t.down_q_offset_bytes[1] = d_q1;
    t.down_scale_offset_bytes[0] = d_s0; t.down_scale_offset_bytes[1] = d_s1;
    t.ffn_dst_offset_bytes = out_off;
    t.output_row_stride_bytes[0] = row_stride_out;
    t.up_bias_offset_bytes = up_b_off;
    t.down_bias_offset_bytes = down_b_off;
    write_task(task_off, t);

    v5::a_stream_t a_st; v5::w_stream_t w_st;
    v5::ctrl_stream_t c_st; v5::psum_stream_t p_st;
    vlm_engine_v5(g_w0, g_w1, g_act, g_out, task_off, a_st, w_st, c_st, p_st);

    return compare_out(name, out_off, gold.data(), rows, N_out, row_stride_out);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("vlm_engine_v5 csim testbench (behavioral PE)\n");
    int err = 0;
    const uint32_t text_qkv_cols[3] = {960, 320, 320};
    const uint32_t vision_qkv_cols[3] = {768, 768, 768};
    const uint32_t vision_dense_cols[1] = {768};

    err += run_dense_case  ("GEMM_DENSE_O M2",        GEMM_DENSE_O,           2, 1);
    err += run_dense_case  ("GEMM_DENSE_O M32",       GEMM_DENSE_O,          32, 1);
    err += run_dense_case  ("GEMM_FUSED_QKV M2",      GEMM_FUSED_QKV,         2, 3);
    err += run_dense_case  ("GEMM_FUSED_QKV M32",     GEMM_FUSED_QKV,        32, 3);
    err += run_dense_case  ("GEMM_FUSED_QKV M46",     GEMM_FUSED_QKV,        46, 3);
    err += run_dense_case  ("GEMM_DENSE_O_BIAS M32",  GEMM_DENSE_O,          32, 1, true);
    err += run_dense_case  ("GEMM_FUSED_QKV_BIAS M32",GEMM_FUSED_QKV,        32, 3, true);
    err += run_ffn_case    ("FFN_TEXT M2 SiLU",       GEMM_FUSED_TEXT_FFN,    LINEAR_FFN_ACT_SILU, 2, 128, 128, 32);
    err += run_ffn_case    ("FFN_TEXT M32 SiLU",      GEMM_FUSED_TEXT_FFN,    LINEAR_FFN_ACT_SILU, 32, 128, 128, 32);
    err += run_ffn_case    ("FFN_TEXT M46 SiLU",      GEMM_FUSED_TEXT_FFN,    LINEAR_FFN_ACT_SILU, 46, 128, 128, 32);
    err += run_ffn_case    ("FFN_VISION M2 GELU",     GEMM_FUSED_VISION_FFN,  LINEAR_FFN_ACT_GELU, 2, 128, 128, 32);
    err += run_ffn_case    ("FFN_VISION M32 GELU",    GEMM_FUSED_VISION_FFN,  LINEAR_FFN_ACT_GELU, 32, 128, 128, 32);
    err += run_dense_real_case("REAL_TEXT_QKV M32",   GEMM_FUSED_QKV,        32, 960, text_qkv_cols, 3, false);
    err += run_dense_real_case("REAL_VISION_QKV M32", GEMM_FUSED_QKV,        32, 768, vision_qkv_cols, 3, true);
    err += run_dense_real_case("REAL_VISION_DENSE M32", GEMM_DENSE_O,        32, 768, vision_dense_cols, 1, true);
    err += run_ffn_case    ("REAL_TEXT_FFN M32 SiLU", GEMM_FUSED_TEXT_FFN,    LINEAR_FFN_ACT_SILU, 32, 960, 2560, 960);
    err += run_ffn_case    ("REAL_VISION_FFN M32 GELU", GEMM_FUSED_VISION_FFN, LINEAR_FFN_ACT_GELU, 32, 768, 3072, 768);
    err += run_vision_mlp_bias_case("VISION_MLP_BIAS M2 zero", 2, 128, 128, 32, true);
    err += run_vision_mlp_bias_case("VISION_MLP_BIAS M32 signed", 32, 128, 128, 32, false);
    err += run_vision_mlp_bias_case("VISION_MLP_BIAS M46 signed", 46, 128, 128, 32, false);
    err += run_vision_mlp_bias_case("REAL_VISION_MLP_BIAS M32", 32, 768, 3072, 768, false);

    if (err == 0) std::printf("V5 testbench: PASS\n");
    else          std::printf("V5 testbench: FAIL err=%d\n", err);
    return err == 0 ? 0 : 1;
}
