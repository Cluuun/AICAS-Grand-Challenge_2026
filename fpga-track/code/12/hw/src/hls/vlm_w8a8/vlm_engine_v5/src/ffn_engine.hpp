// ffn_engine.hpp --- V5 FFN gate/up/down fused engine.
//
// Layer math (per token row r):
//   gate[r,n] = SiLU/GELU( sum_k act[r,k] * Wg[k,n] )    // intermediate cols
//   up  [r,n] = sum_k act[r,k] * Wu[k,n]                 // intermediate cols
//   mid [r,n] = gate[r,n] * up[r,n]                       // requantized to int8
//   out [r,m] = sum_n mid[r,n] * Wd[n,m]                  // output cols
//
// Per m_tile:
//   1) load_activation_block(K=hidden) once
//   2) for each intermediate-N-tile:
//        run gate sub-tile -> int32 gate_acc[kSaM][kSaN]
//        run up   sub-tile -> int32 up_acc  [kSaM][kSaN]
//        fuse + quantize  -> int8 mid_q[kSaM][kSaN], scale_exp[kSaM][group]
//        store into mid_cache[][...]
//   3) move mid_cache into act_cache (re-binding), set K=intermediate
//   4) for each output-N-tile of down:
//        run dense kernel with mid as activation
//
// We reuse run_ntile_accumulate_batch() from dense_engine.hpp for gate/up.
// Each call re-clears the PE accumulator (clear=true on the first group).
#pragma once

#include "dense_engine.hpp"
#include "bias_loader.hpp"

namespace v5 {

static constexpr uint32_t kVisionMlpUpBiasCols = VLM_W8A8_VISION_FFN;
static constexpr uint32_t kVisionMlpDownBiasCols = VLM_W8A8_VISION_HIDDEN;
static constexpr uint32_t kVisionMlpDownBiasBase = kVisionMlpUpBiasCols;

static void load_vision_mlp_bias_cache(
        const axi_word_t *act_arena,
        uint64_t up_bias_offset,
        uint64_t down_bias_offset,
        out_acc_t bias_cache[kBiasCacheCols]) {
    #pragma HLS INLINE off
    load_bias_vector_q22_i40(act_arena, up_bias_offset,
                             kVisionMlpUpBiasCols, 0, bias_cache);
    load_bias_vector_q22_i40(act_arena, down_bias_offset,
                             kVisionMlpDownBiasCols,
                             kVisionMlpDownBiasBase, bias_cache);
}

static inline out_acc_t fixed_from_i32_q22(int32_t x) {
    #pragma HLS INLINE
    return saturate_i96_to_out(((ap_int<96>)x) << kOutFrac);
}

static inline out_acc_t fixed_mul_div_q22(out_acc_t a, out_acc_t b, int divisor) {
    #pragma HLS INLINE
    const ap_int<24> recip = (divisor == 3) ? (ap_int<24>)5592405 : (ap_int<24>)2796203;
    ap_int<80> product = (ap_int<80>)a * (ap_int<80>)b;
    ap_int<104> scaled = (ap_int<104>)product * (ap_int<104>)recip;
    return saturate_i96_to_out((ap_int<96>)(scaled >> (kOutFrac + 24)));
}

// HardSiLU: x * clamp(x+3, 0, 6) / 6 in Q22 fixed-point.
static inline out_acc_t V5_hard_silu(out_acc_t x) {
    #pragma HLS INLINE
    const out_acc_t six = fixed_from_i32_q22(6);
    const out_acc_t three = fixed_from_i32_q22(3);
    out_acc_t gate = add_saturate_out(x, three);
    if (gate < 0) gate = 0;
    if (gate > six) gate = six;
    return fixed_mul_div_q22(x, gate, 6);
}

// HardGELU: x * clamp(x+1.5, 0, 3) / 3 in Q22 fixed-point.
static inline out_acc_t V5_hard_gelu(out_acc_t x) {
    #pragma HLS INLINE
    const out_acc_t three = fixed_from_i32_q22(3);
    const out_acc_t one_and_half = three >> 1;
    out_acc_t gate = add_saturate_out(x, one_and_half);
    if (gate < 0) gate = 0;
    if (gate > three) gate = three;
    return fixed_mul_div_q22(x, gate, 3);
}

typedef ap_int<48> gelu_q32_t;
typedef ap_int<32> gelu_slope_t;

struct GeluLutSeg {
    gelu_q32_t y0;
    gelu_slope_t slope_q16;
};

static inline gelu_q32_t saturate_i96_to_gelu_q32(ap_int<96> v) {
    #pragma HLS INLINE
    const ap_int<96> max_v = (((ap_int<96>)1) << 47) - 1;
    const ap_int<96> min_v = -(((ap_int<96>)1) << 47);
    if (v > max_v) {
        return (gelu_q32_t)max_v;
    }
    if (v < min_v) {
        return (gelu_q32_t)min_v;
    }
    return (gelu_q32_t)v;
}

static inline gelu_q32_t round_shift_right_to_gelu_q32(ap_int<96> value, uint32_t shift) {
    #pragma HLS INLINE
    if (shift == 0) {
        return saturate_i96_to_gelu_q32(value);
    }
    const ap_int<96> half = ((ap_int<96>)1) << (shift - 1);
    if (value >= 0) {
        return saturate_i96_to_gelu_q32((value + half) >> shift);
    }
    return saturate_i96_to_gelu_q32(-(((-value) + half) >> shift));
}

// GELU erf LUT over [-6, 6], 128 uniform segments.
// Inputs arrive as Q22/int40 accumulators; the output remains Q32 until the
// per-group A8 requant stage to avoid the accuracy loss caused by early Q22
// rounding in the vision MLP fused path.
static inline gelu_q32_t V5_gelu_lut128_q32(out_acc_t x_q22) {
    #pragma HLS INLINE
    static constexpr int64_t kMinXQ22 = -(6LL << kOutFrac);
    static constexpr int64_t kMaxXQ22 =  (6LL << kOutFrac);
    static constexpr int64_t kStepQ22 = (12LL << kOutFrac) / 128LL;

    #define V5_GELU_SLOPE_Q16(dy) \
        ((gelu_slope_t)(((dy) >= 0) ? (((dy) + 3LL) / 6LL) : -(((-(dy)) + 3LL) / 6LL)))
    static const GeluLutSeg lut[128] = {
        { (gelu_q32_t)-25LL, V5_GELU_SLOPE_Q16(-19LL) },
        { (gelu_q32_t)-44LL, V5_GELU_SLOPE_Q16(-33LL) },
        { (gelu_q32_t)-77LL, V5_GELU_SLOPE_Q16(-55LL) },
        { (gelu_q32_t)-132LL, V5_GELU_SLOPE_Q16(-92LL) },
        { (gelu_q32_t)-224LL, V5_GELU_SLOPE_Q16(-154LL) },
        { (gelu_q32_t)-378LL, V5_GELU_SLOPE_Q16(-253LL) },
        { (gelu_q32_t)-631LL, V5_GELU_SLOPE_Q16(-414LL) },
        { (gelu_q32_t)-1045LL, V5_GELU_SLOPE_Q16(-670LL) },
        { (gelu_q32_t)-1715LL, V5_GELU_SLOPE_Q16(-1075LL) },
        { (gelu_q32_t)-2790LL, V5_GELU_SLOPE_Q16(-1709LL) },
        { (gelu_q32_t)-4499LL, V5_GELU_SLOPE_Q16(-2691LL) },
        { (gelu_q32_t)-7190LL, V5_GELU_SLOPE_Q16(-4201LL) },
        { (gelu_q32_t)-11391LL, V5_GELU_SLOPE_Q16(-6496LL) },
        { (gelu_q32_t)-17887LL, V5_GELU_SLOPE_Q16(-9953LL) },
        { (gelu_q32_t)-27840LL, V5_GELU_SLOPE_Q16(-15107LL) },
        { (gelu_q32_t)-42947LL, V5_GELU_SLOPE_Q16(-22721LL) },
        { (gelu_q32_t)-65668LL, V5_GELU_SLOPE_Q16(-33853LL) },
        { (gelu_q32_t)-99521LL, V5_GELU_SLOPE_Q16(-49969LL) },
        { (gelu_q32_t)-149490LL, V5_GELU_SLOPE_Q16(-73071LL) },
        { (gelu_q32_t)-222561LL, V5_GELU_SLOPE_Q16(-105850LL) },
        { (gelu_q32_t)-328411LL, V5_GELU_SLOPE_Q16(-151890LL) },
        { (gelu_q32_t)-480301LL, V5_GELU_SLOPE_Q16(-215900LL) },
        { (gelu_q32_t)-696201LL, V5_GELU_SLOPE_Q16(-303971LL) },
        { (gelu_q32_t)-1000172LL, V5_GELU_SLOPE_Q16(-423888LL) },
        { (gelu_q32_t)-1424060LL, V5_GELU_SLOPE_Q16(-585449LL) },
        { (gelu_q32_t)-2009509LL, V5_GELU_SLOPE_Q16(-800792LL) },
        { (gelu_q32_t)-2810301LL, V5_GELU_SLOPE_Q16(-1084719LL) },
        { (gelu_q32_t)-3895020LL, V5_GELU_SLOPE_Q16(-1454962LL) },
        { (gelu_q32_t)-5349982LL, V5_GELU_SLOPE_Q16(-1932370LL) },
        { (gelu_q32_t)-7282352LL, V5_GELU_SLOPE_Q16(-2540951LL) },
        { (gelu_q32_t)-9823303LL, V5_GELU_SLOPE_Q16(-3307722LL) },
        { (gelu_q32_t)-13131025LL, V5_GELU_SLOPE_Q16(-4262279LL) },
        { (gelu_q32_t)-17393304LL, V5_GELU_SLOPE_Q16(-5436047LL) },
        { (gelu_q32_t)-22829351LL, V5_GELU_SLOPE_Q16(-6861101LL) },
        { (gelu_q32_t)-29690452LL, V5_GELU_SLOPE_Q16(-8568533LL) },
        { (gelu_q32_t)-38258985LL, V5_GELU_SLOPE_Q16(-10586291LL) },
        { (gelu_q32_t)-48845276LL, V5_GELU_SLOPE_Q16(-12936496LL) },
        { (gelu_q32_t)-61781772LL, V5_GELU_SLOPE_Q16(-15632243LL) },
        { (gelu_q32_t)-77414015LL, V5_GELU_SLOPE_Q16(-18673948LL) },
        { (gelu_q32_t)-96087963LL, V5_GELU_SLOPE_Q16(-22045385LL) },
        { (gelu_q32_t)-118133348LL, V5_GELU_SLOPE_Q16(-25709556LL) },
        { (gelu_q32_t)-143842904LL, V5_GELU_SLOPE_Q16(-29604679LL) },
        { (gelu_q32_t)-173447583LL, V5_GELU_SLOPE_Q16(-33640561LL) },
        { (gelu_q32_t)-207088144LL, V5_GELU_SLOPE_Q16(-37695693LL) },
        { (gelu_q32_t)-244783837LL, V5_GELU_SLOPE_Q16(-41615472LL) },
        { (gelu_q32_t)-286399309LL, V5_GELU_SLOPE_Q16(-45211861LL) },
        { (gelu_q32_t)-331611170LL, V5_GELU_SLOPE_Q16(-48264867LL) },
        { (gelu_q32_t)-379876037LL, V5_GELU_SLOPE_Q16(-50526080LL) },
        { (gelu_q32_t)-430402117LL, V5_GELU_SLOPE_Q16(-51724442LL) },
        { (gelu_q32_t)-482126559LL, V5_GELU_SLOPE_Q16(-51574292LL) },
        { (gelu_q32_t)-533700851LL, V5_GELU_SLOPE_Q16(-49785534LL) },
        { (gelu_q32_t)-583486385LL, V5_GELU_SLOPE_Q16(-46075641LL) },
        { (gelu_q32_t)-629562026LL, V5_GELU_SLOPE_Q16(-40183012LL) },
        { (gelu_q32_t)-669745038LL, V5_GELU_SLOPE_Q16(-31881002LL) },
        { (gelu_q32_t)-701626040LL, V5_GELU_SLOPE_Q16(-20991891LL) },
        { (gelu_q32_t)-722617931LL, V5_GELU_SLOPE_Q16(-7399869LL) },
        { (gelu_q32_t)-730017800LL, V5_GELU_SLOPE_Q16(8937825LL) },
        { (gelu_q32_t)-721079975LL, V5_GELU_SLOPE_Q16(27982495LL) },
        { (gelu_q32_t)-693097480LL, V5_GELU_SLOPE_Q16(49609028LL) },
        { (gelu_q32_t)-643488452LL, V5_GELU_SLOPE_Q16(73604972LL) },
        { (gelu_q32_t)-569883480LL, V5_GELU_SLOPE_Q16(99673956LL) },
        { (gelu_q32_t)-470209524LL, V5_GELU_SLOPE_Q16(127443502LL) },
        { (gelu_q32_t)-342766022LL, V5_GELU_SLOPE_Q16(156476966LL) },
        { (gelu_q32_t)-186289056LL, V5_GELU_SLOPE_Q16(186289056LL) },
        { (gelu_q32_t)0LL, V5_GELU_SLOPE_Q16(216364128LL) },
        { (gelu_q32_t)216364128LL, V5_GELU_SLOPE_Q16(246176218LL) },
        { (gelu_q32_t)462540346LL, V5_GELU_SLOPE_Q16(275209682LL) },
        { (gelu_q32_t)737750028LL, V5_GELU_SLOPE_Q16(302979228LL) },
        { (gelu_q32_t)1040729256LL, V5_GELU_SLOPE_Q16(329048212LL) },
        { (gelu_q32_t)1369777468LL, V5_GELU_SLOPE_Q16(353044156LL) },
        { (gelu_q32_t)1722821624LL, V5_GELU_SLOPE_Q16(374670689LL) },
        { (gelu_q32_t)2097492313LL, V5_GELU_SLOPE_Q16(393715359LL) },
        { (gelu_q32_t)2491207672LL, V5_GELU_SLOPE_Q16(410053053LL) },
        { (gelu_q32_t)2901260725LL, V5_GELU_SLOPE_Q16(423645075LL) },
        { (gelu_q32_t)3324905800LL, V5_GELU_SLOPE_Q16(434534186LL) },
        { (gelu_q32_t)3759439986LL, V5_GELU_SLOPE_Q16(442836196LL) },
        { (gelu_q32_t)4202276182LL, V5_GELU_SLOPE_Q16(448728825LL) },
        { (gelu_q32_t)4651005007LL, V5_GELU_SLOPE_Q16(452438718LL) },
        { (gelu_q32_t)5103443725LL, V5_GELU_SLOPE_Q16(454227476LL) },
        { (gelu_q32_t)5557671201LL, V5_GELU_SLOPE_Q16(454377626LL) },
        { (gelu_q32_t)6012048827LL, V5_GELU_SLOPE_Q16(453179264LL) },
        { (gelu_q32_t)6465228091LL, V5_GELU_SLOPE_Q16(450918051LL) },
        { (gelu_q32_t)6916146142LL, V5_GELU_SLOPE_Q16(447865045LL) },
        { (gelu_q32_t)7364011187LL, V5_GELU_SLOPE_Q16(444268656LL) },
        { (gelu_q32_t)7808279843LL, V5_GELU_SLOPE_Q16(440348877LL) },
        { (gelu_q32_t)8248628720LL, V5_GELU_SLOPE_Q16(436293745LL) },
        { (gelu_q32_t)8684922465LL, V5_GELU_SLOPE_Q16(432257863LL) },
        { (gelu_q32_t)9117180328LL, V5_GELU_SLOPE_Q16(428362740LL) },
        { (gelu_q32_t)9545543068LL, V5_GELU_SLOPE_Q16(424698569LL) },
        { (gelu_q32_t)9970241637LL, V5_GELU_SLOPE_Q16(421327132LL) },
        { (gelu_q32_t)10391568769LL, V5_GELU_SLOPE_Q16(418285427LL) },
        { (gelu_q32_t)10809854196LL, V5_GELU_SLOPE_Q16(415589680LL) },
        { (gelu_q32_t)11225443876LL, V5_GELU_SLOPE_Q16(413239475LL) },
        { (gelu_q32_t)11638683351LL, V5_GELU_SLOPE_Q16(411221717LL) },
        { (gelu_q32_t)12049905068LL, V5_GELU_SLOPE_Q16(409514285LL) },
        { (gelu_q32_t)12459419353LL, V5_GELU_SLOPE_Q16(408089231LL) },
        { (gelu_q32_t)12867508584LL, V5_GELU_SLOPE_Q16(406915463LL) },
        { (gelu_q32_t)13274424047LL, V5_GELU_SLOPE_Q16(405960906LL) },
        { (gelu_q32_t)13680384953LL, V5_GELU_SLOPE_Q16(405194135LL) },
        { (gelu_q32_t)14085579088LL, V5_GELU_SLOPE_Q16(404585554LL) },
        { (gelu_q32_t)14490164642LL, V5_GELU_SLOPE_Q16(404108146LL) },
        { (gelu_q32_t)14894272788LL, V5_GELU_SLOPE_Q16(403737903LL) },
        { (gelu_q32_t)15298010691LL, V5_GELU_SLOPE_Q16(403453976LL) },
        { (gelu_q32_t)15701464667LL, V5_GELU_SLOPE_Q16(403238633LL) },
        { (gelu_q32_t)16104703300LL, V5_GELU_SLOPE_Q16(403077072LL) },
        { (gelu_q32_t)16507780372LL, V5_GELU_SLOPE_Q16(402957155LL) },
        { (gelu_q32_t)16910737527LL, V5_GELU_SLOPE_Q16(402869084LL) },
        { (gelu_q32_t)17313606611LL, V5_GELU_SLOPE_Q16(402805074LL) },
        { (gelu_q32_t)17716411685LL, V5_GELU_SLOPE_Q16(402759034LL) },
        { (gelu_q32_t)18119170719LL, V5_GELU_SLOPE_Q16(402726255LL) },
        { (gelu_q32_t)18521896974LL, V5_GELU_SLOPE_Q16(402703153LL) },
        { (gelu_q32_t)18924600127LL, V5_GELU_SLOPE_Q16(402687037LL) },
        { (gelu_q32_t)19327287164LL, V5_GELU_SLOPE_Q16(402675905LL) },
        { (gelu_q32_t)19729963069LL, V5_GELU_SLOPE_Q16(402668291LL) },
        { (gelu_q32_t)20132631360LL, V5_GELU_SLOPE_Q16(402663137LL) },
        { (gelu_q32_t)20535294497LL, V5_GELU_SLOPE_Q16(402659680LL) },
        { (gelu_q32_t)20937954177LL, V5_GELU_SLOPE_Q16(402657385LL) },
        { (gelu_q32_t)21340611562LL, V5_GELU_SLOPE_Q16(402655875LL) },
        { (gelu_q32_t)21743267437LL, V5_GELU_SLOPE_Q16(402654893LL) },
        { (gelu_q32_t)22145922330LL, V5_GELU_SLOPE_Q16(402654259LL) },
        { (gelu_q32_t)22548576589LL, V5_GELU_SLOPE_Q16(402653854LL) },
        { (gelu_q32_t)22951230443LL, V5_GELU_SLOPE_Q16(402653598LL) },
        { (gelu_q32_t)23353884041LL, V5_GELU_SLOPE_Q16(402653437LL) },
        { (gelu_q32_t)23756537478LL, V5_GELU_SLOPE_Q16(402653338LL) },
        { (gelu_q32_t)24159190816LL, V5_GELU_SLOPE_Q16(402653276LL) },
        { (gelu_q32_t)24561844092LL, V5_GELU_SLOPE_Q16(402653239LL) },
        { (gelu_q32_t)24964497331LL, V5_GELU_SLOPE_Q16(402653217LL) },
        { (gelu_q32_t)25367150548LL, V5_GELU_SLOPE_Q16(402653203LL) },
    };
    #pragma HLS BIND_STORAGE variable=lut type=rom_2p impl=bram
    #undef V5_GELU_SLOPE_Q16

    if (x_q22 <= kMinXQ22) {
        return 0;
    }
    if (x_q22 >= kMaxXQ22) {
        return saturate_i96_to_gelu_q32(((ap_int<96>)x_q22) << (32 - kOutFrac));
    }

    const ap_int<48> rel = (ap_int<48>)x_q22 - (ap_int<48>)kMinXQ22;
    ap_uint<8> idx = (ap_uint<8>)(rel / kStepQ22);
    if (idx > 127) {
        idx = 127;
    }
    const ap_int<32> rem = (ap_int<32>)(rel - (ap_int<48>)idx * kStepQ22);
    const GeluLutSeg seg = lut[idx];
    const ap_int<96> interp = (ap_int<96>)seg.slope_q16 * (ap_int<96>)rem;
    return saturate_i96_to_gelu_q32((ap_int<96>)seg.y0 + round_shift_right_to_gelu_q32(interp, 16));
}

static inline out_acc_t round_mul_q22(out_acc_t a, out_acc_t b) {
    #pragma HLS INLINE
    ap_int<96> p = (ap_int<96>)a * (ap_int<96>)b;
    return round_shift_right_to_out(p, kOutFrac);
}

static inline bool pot_shift_fits_i96(ap_int<96> max_abs, int shift) {
    #pragma HLS INLINE
    if (shift >= 0) {
        const ap_int<96> limit = (ap_int<96>)127 << shift;
        return max_abs <= limit;
    }
    const ap_int<96> scaled = max_abs << (-shift);
    return scaled <= 127;
}

static inline int msb_index_u32(ap_uint<32> x) {
    #pragma HLS INLINE
    int idx = 0;
    if (x(31, 16) != 0) {
        x = x >> 16;
        idx += 16;
    }
    if (x(15, 8) != 0) {
        x = x >> 8;
        idx += 8;
    }
    if (x(7, 4) != 0) {
        x = x >> 4;
        idx += 4;
    }
    if (x(3, 2) != 0) {
        x = x >> 2;
        idx += 2;
    }
    if (x(1, 1) != 0) {
        idx += 1;
    }
    return idx;
}

static inline int msb_index_i96(ap_int<96> x) {
    #pragma HLS INLINE
    const ap_uint<96> u = (ap_uint<96>)x;
    if (u(95, 64) != 0) {
        return 64 + msb_index_u32((ap_uint<32>)u(95, 64));
    }
    if (u(63, 32) != 0) {
        return 32 + msb_index_u32((ap_uint<32>)u(63, 32));
    }
    return msb_index_u32((ap_uint<32>)u(31, 0));
}

static inline int choose_pot_shift_i96(ap_int<96> max_abs, int default_shift) {
    #pragma HLS INLINE
    if (max_abs <= 0) {
        return default_shift;
    }
    int shift = msb_index_i96(max_abs) - 6;
    if (shift < -20) shift = -20;
    if (shift > 62) shift = 62;
    if (!pot_shift_fits_i96(max_abs, shift) && shift < 62) {
        shift += 1;
    }
    return shift;
}

static inline int8_t choose_pot_exp_for_q32(gelu_q32_t max_abs) {
    #pragma HLS INLINE
    const int selected_shift = choose_pot_shift_i96((ap_int<96>)max_abs, 32);
    int exp = selected_shift - 32;
    if (exp < -30) exp = -30;
    if (exp > 30) exp = 30;
    return (int8_t)exp;
}

static inline int8_t quantize_q32_with_exp(gelu_q32_t x, int8_t exp) {
    #pragma HLS INLINE
    const int shift = (int)exp + 32;
    out_acc_t q = 0;
    if (shift >= 0) {
        if (shift >= 95) {
            q = 0;
        } else {
            q = round_shift_right_to_out((ap_int<96>)x, (uint32_t)shift);
        }
    } else {
        const int lshift = -shift;
        q = saturate_i96_to_out(((ap_int<96>)x) << lshift);
    }
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

#if 0
// Kept temporarily as a readable reference for the bit-exact math. The
// release datapath below stores only the current 128-col intermediate tile and
// consumes it immediately; it never allocates a full mid_cache.
static void requantize_q32_ktile_into_act_cache(
        const gelu_q32_t fused[kSaM][kTileK],
        uint32_t rows_valid,
        uint32_t kt_col_byte_off,
        uint32_t valid_cols,
        ActCache &mid) {
    #pragma HLS INLINE off
    const uint32_t group_base = kt_col_byte_off / kGroupSize;
    const uint32_t base_word = kt_col_byte_off / 8;

    for (uint32_t r = 0; r < kSaM; ++r) {
        scale_exp_t group_exp[kGroupsPerKtile];
        #pragma HLS ARRAY_PARTITION variable=group_exp complete
        for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
            gelu_q32_t max_abs = 0;
            for (uint32_t c = 0; c < kGroupSize; ++c) {
                #pragma HLS PIPELINE II=1
                const uint32_t col = g * kGroupSize + c;
                if (col < valid_cols) {
                    const gelu_q32_t v = fused[r][col];
                    const gelu_q32_t a = (v < 0) ? (gelu_q32_t)-v : v;
                    if (a > max_abs) {
                        max_abs = a;
                    }
                }
            }
            group_exp[g] = choose_pot_exp_for_q32(max_abs);
            if (r < rows_valid) {
                mid.row_exp[r][group_base + g] = group_exp[g];
            }
        }

        for (uint32_t w = 0; w < kTileK / 8; ++w) {
            #pragma HLS PIPELINE II=1
            ap_uint<64> word = 0;
            for (uint32_t b = 0; b < 8; ++b) {
                #pragma HLS UNROLL
                const uint32_t c = w * 8 + b;
                const uint32_t g = c / kGroupSize;
                int8_t q = 0;
                if (r < rows_valid && c < valid_cols) {
                    q = quantize_q32_with_exp(fused[r][c], group_exp[g]);
                }
                word(b * 8 + 7, b * 8) = static_cast<uint8_t>(q);
            }
            mid.buf[r][base_word + w] = word;
        }
    }
}

// Non-gated vision MLP:
//   mid = hard_gelu(input * up + up_bias)
//   out = mid * down + down_bias
// Bias is injected as the first accumulator value in scale drain.
static void run_prefill_vision_mlp_gelu_bias(
        const axi_word_t *w0,
        const axi_word_t *w1,
        const axi_word_t *act_arena,
        axi_word_t *out_arena,
        const task_t &task,
        ActCache &act_cache,
        ActCache &mid_cache,
        int8_t exp_in[kMaxKtiles][kSaM][kGroupsPerKtile],
        int8_t exp_im[kMaxKtiles][kSaM][kGroupsPerKtile],
        out_acc_t bias_cache[kBiasCacheCols],
        gelu_q32_t fused_kt[kSaM][kTileK],
        OutAcc &work_acc,
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream) {
    #pragma HLS INLINE

    const uint32_t M    = task.rows;
    const uint32_t K_in = task.ffn_input_cols;
    const uint32_t K_im = task.ffn_intermediate_cols;
    const uint32_t N_out= task.ffn_output_cols;
    const uint32_t kt_in = div_ceil(K_in, kTileK);
    const uint32_t kt_im = div_ceil(K_im, kTileK);
    const uint32_t k_groups_in = div_ceil(K_in, kGroupSize);
    const uint32_t k_groups_im = div_ceil(K_im, kGroupSize);
    const uint32_t m_tiles = div_ceil(M, kSaM);
    const uint32_t nt_im = div_ceil(K_im, kSaN);
    const uint32_t nt_out = div_ceil(N_out, kSaN);
    const uint32_t k_words_in_valid = div_ceil(K_in, kAxiBytes);
    const uint32_t k_words_in = kt_in * kActWordsPerKtile;
    const uint32_t down_weight_n_tile_stride_bytes =
        kWeightWordsPerKtilePerPort * kAxiBytes * kt_im;
    const uint32_t down_weight_scale_n_tile_stride_bytes =
        kScaleWordsPerKtilePerPort * kAxiBytes * kt_im;

    load_vision_mlp_bias_cache(act_arena,
                               task.up_bias_offset_bytes,
                               task.down_bias_offset_bytes,
                               bias_cache);

    for (uint32_t mt = 0; mt < m_tiles; ++mt) {
        const uint32_t mt_row0 = mt * kSaM;
        const uint32_t rows_valid = (mt_row0 + kSaM <= M) ? kSaM : (M - mt_row0);

        load_activation_block(act_arena,
                              task.act_q_offset_bytes
                                + (uint64_t)mt_row0 * task.act_row_stride,
                              task.act_scale_offset_bytes
                                + (uint64_t)mt_row0 * task.act_scale_row_stride,
                              task.act_row_stride,
                              task.act_scale_row_stride,
                              rows_valid, k_words_in_valid, k_words_in,
                              k_groups_in, act_cache);
        for (uint32_t kt = 0; kt < kt_in; ++kt) {
            for (uint32_t r = 0; r < kSaM; ++r) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    #pragma HLS UNROLL
                    uint32_t glob = kt * kGroupsPerKtile + g;
                    exp_in[kt][r][g] =
                        (glob < k_groups_in) ? static_cast<int8_t>(act_cache.row_exp[r][glob])
                                             : static_cast<int8_t>(0);
                }
            }
        }

        for (uint32_t nt_base = 0; nt_base < nt_im; nt_base += kNtilesPerKtile) {
            for (uint32_t nt_g = 0; nt_g < kNtilesPerKtile; ++nt_g) {
                const uint32_t nt = nt_base + nt_g;
                const bool nt_valid = nt < nt_im;
                if (nt_valid) {
                    const uint64_t u_w0 = (task.up_q_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_w1 = (task.up_q_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_s0 = (task.up_scale_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_s1 = (task.up_scale_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;

                    out_acc_t bias_tile[kSaN];
                    #pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1
                    load_bias_tile_from_cache(bias_cache, nt * kSaN, bias_tile);

                    run_ntile_accumulate(w0, w1, act_cache, u_w0, u_w1, u_s0, u_s1,
                                          kt_in, rows_valid, exp_in,
                                          a_stream, w_stream, ctrl_stream, psum_stream,
                                          work_acc, bias_tile);

                    for (uint32_t r = 0; r < kSaM; ++r) {
                        for (uint32_t cb = 0; cb < kSaN; cb += kFfnFusionColLanes) {
                            #pragma HLS PIPELINE II=1
                            for (uint32_t lane = 0; lane < kFfnFusionColLanes; ++lane) {
                                #pragma HLS UNROLL
                                const uint32_t c = cb + lane;
                                fused_kt[r][nt_g * kSaN + c] = V5_gelu_lut128_q32(work_acc.v[r][c]);
                            }
                        }
                    }
                } else {
                    for (uint32_t r = 0; r < kSaM; ++r) {
                        for (uint32_t cb = 0; cb < kSaN; cb += kFfnFusionColLanes) {
                            #pragma HLS PIPELINE II=1
                            for (uint32_t lane = 0; lane < kFfnFusionColLanes; ++lane) {
                                #pragma HLS UNROLL
                                const uint32_t c = cb + lane;
                                fused_kt[r][nt_g * kSaN + c] = 0;
                            }
                        }
                    }
                }
            }
            const uint32_t kt_col = nt_base * kSaN;
            const uint32_t valid_cols = (kt_col + kTileK <= K_im) ? kTileK : (K_im - kt_col);
            requantize_q32_ktile_into_act_cache(fused_kt, rows_valid, kt_col, valid_cols, mid_cache);
        }

        for (uint32_t kt = 0; kt < kt_im; ++kt) {
            for (uint32_t r = 0; r < kSaM; ++r) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    #pragma HLS UNROLL
                    uint32_t glob = kt * kGroupsPerKtile + g;
                    exp_im[kt][r][g] =
                        (glob < k_groups_im) ? static_cast<int8_t>(mid_cache.row_exp[r][glob])
                                             : static_cast<int8_t>(0);
                }
            }
        }

        for (uint32_t nt = 0; nt < nt_out; ++nt) {
            const uint64_t d_w0 = (task.down_q_offset_bytes[0]
                                   + (uint64_t)nt * down_weight_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_w1 = (task.down_q_offset_bytes[1]
                                   + (uint64_t)nt * down_weight_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_s0 = (task.down_scale_offset_bytes[0]
                                   + (uint64_t)nt * down_weight_scale_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_s1 = (task.down_scale_offset_bytes[1]
                                   + (uint64_t)nt * down_weight_scale_n_tile_stride_bytes) / kAxiBytes;

            out_acc_t bias_tile[kSaN];
            #pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1
            load_bias_tile_from_cache(bias_cache, kVisionMlpDownBiasBase + nt * kSaN, bias_tile);

            run_ntile_accumulate(w0, w1, mid_cache,
                                  d_w0, d_w1, d_s0, d_s1,
                                  kt_im, rows_valid, exp_im,
                                  a_stream, w_stream, ctrl_stream, psum_stream,
                                  work_acc, bias_tile);

            const uint64_t out_byte_off = task.ffn_dst_offset_bytes
                                        + (uint64_t)mt_row0 * task.output_row_stride_bytes[0];
            const uint32_t out_col_byte_off = nt * kSaN * kOutBytes;
            store_out_acc(out_arena, out_byte_off,
                          task.output_row_stride_bytes[0],
                          rows_valid, out_col_byte_off, work_acc);
        }
    }
}

// Top-level FFN. Phases: (1) gate/up -> mid cache, (2) down -> DDR.
static void run_prefill_ffn_legacy(
        const axi_word_t *w0,
        const axi_word_t *w1,
        const axi_word_t *act_arena,
        axi_word_t *out_arena,
        const task_t &task,
        ActCache &act_cache,
        ActCache &mid_cache,
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream) {
    #pragma HLS INLINE

    static int8_t exp_in[kMaxKtiles][kSaM][kGroupsPerKtile];
    static int8_t exp_im[kMaxKtiles][kSaM][kGroupsPerKtile];
    static out_acc_t zero_bias_tile[kSaN] = {};
    static out_acc_t bias_cache[kBiasCacheCols];
    #pragma HLS ARRAY_PARTITION variable=zero_bias_tile complete dim=1
    #pragma HLS BIND_STORAGE variable=bias_cache type=ram_2p impl=bram

    gelu_q32_t fused_kt[kSaM][kTileK];
    #pragma HLS BIND_STORAGE variable=fused_kt type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=fused_kt cyclic factor=kFfnFusionColLanes dim=2

    OutAcc acc0, acc1;
    #pragma HLS ARRAY_PARTITION variable=acc0.v cyclic factor=kScaleDrainColLanes dim=2
    #pragma HLS ARRAY_PARTITION variable=acc1.v cyclic factor=kScaleDrainColLanes dim=2

    if (task.task_type == GEMM_FUSED_VISION_MLP_GELU_BIAS) {
        run_prefill_vision_mlp_gelu_bias(w0, w1, act_arena, out_arena, task,
            act_cache, mid_cache, exp_in, exp_im, bias_cache, fused_kt, acc0,
            a_stream, w_stream, ctrl_stream, psum_stream);
        return;
    }

    const uint32_t M    = task.rows;
    const uint32_t K_in = task.ffn_input_cols;          // hidden
    const uint32_t K_im = task.ffn_intermediate_cols;   // intermediate
    const uint32_t N_out= task.ffn_output_cols;         // hidden
    const uint32_t kt_in     = div_ceil(K_in, kTileK);
    const uint32_t kt_im     = div_ceil(K_im, kTileK);
    const uint32_t k_groups_in = div_ceil(K_in, kGroupSize);
    const uint32_t k_groups_im = div_ceil(K_im, kGroupSize);
    const uint32_t m_tiles   = div_ceil(M, kSaM);
    const uint32_t nt_im     = div_ceil(K_im, kSaN);
    const uint32_t nt_out    = div_ceil(N_out, kSaN);
    const uint32_t k_words_in_valid = div_ceil(K_in, kAxiBytes);
    const uint32_t k_words_in = kt_in * kActWordsPerKtile;
    const uint32_t down_weight_n_tile_stride_bytes =
        kWeightWordsPerKtilePerPort * kAxiBytes * kt_im;
    const uint32_t down_weight_scale_n_tile_stride_bytes =
        kScaleWordsPerKtilePerPort * kAxiBytes * kt_im;

    for (uint32_t mt = 0; mt < m_tiles; ++mt) {
        const uint32_t mt_row0    = mt * kSaM;
        const uint32_t rows_valid = (mt_row0 + kSaM <= M) ? kSaM : (M - mt_row0);

        // ---- Phase A: load input activation, run gate+up, build mid cache ----
        load_activation_block(act_arena,
                              task.act_q_offset_bytes
                                + (uint64_t)mt_row0 * task.act_row_stride,
                              task.act_scale_offset_bytes
                                + (uint64_t)mt_row0 * task.act_scale_row_stride,
                              task.act_row_stride,
                              task.act_scale_row_stride,
                              rows_valid, k_words_in_valid, k_words_in,
                              k_groups_in, act_cache);
        for (uint32_t kt = 0; kt < kt_in; ++kt) {
            for (uint32_t r = 0; r < kSaM; ++r) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    #pragma HLS UNROLL
                    uint32_t glob = kt * kGroupsPerKtile + g;
                    exp_in[kt][r][g] =
                        (glob < k_groups_in) ? static_cast<int8_t>(act_cache.row_exp[r][glob])
                                             : static_cast<int8_t>(0);
                }
            }
        }

        for (uint32_t nt_base = 0; nt_base < nt_im; nt_base += kNtilesPerKtile) {
            for (uint32_t nt_g = 0; nt_g < kNtilesPerKtile; ++nt_g) {
                const uint32_t nt = nt_base + nt_g;
                const bool nt_valid = nt < nt_im;
                if (nt_valid) {
                    const uint64_t g_w0 = (task.gate_q_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t g_w1 = (task.gate_q_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t g_s0 = (task.gate_scale_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t g_s1 = (task.gate_scale_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_w0 = (task.up_q_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_w1 = (task.up_q_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_s0 = (task.up_scale_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_s1 = (task.up_scale_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;

                    run_ntile_accumulate(w0, w1, act_cache, g_w0, g_w1, g_s0, g_s1,
                                          kt_in, rows_valid, exp_in,
                                          a_stream, w_stream, ctrl_stream, psum_stream,
                                          acc0, zero_bias_tile);
                    run_ntile_accumulate(w0, w1, act_cache, u_w0, u_w1, u_s0, u_s1,
                                          kt_in, rows_valid, exp_in,
                                          a_stream, w_stream, ctrl_stream, psum_stream,
                                          acc1, zero_bias_tile);

                    for (uint32_t r = 0; r < kSaM; ++r) {
                        for (uint32_t cb = 0; cb < kSaN; cb += kFfnFusionColLanes) {
                            #pragma HLS PIPELINE II=1
                            for (uint32_t lane = 0; lane < kFfnFusionColLanes; ++lane) {
                            #pragma HLS UNROLL
                            const uint32_t c = cb + lane;
                            const out_acc_t g = acc0.v[r][c];
                            const out_acc_t u = acc1.v[r][c];
                            const out_acc_t act = (task.ffn_activation == LINEAR_FFN_ACT_GELU)
                                            ? V5_hard_gelu(g) : V5_hard_silu(g);
                            fused_kt[r][nt_g * kSaN + c] =
                                saturate_i96_to_gelu_q32(((ap_int<96>)round_mul_q22(act, u)) << (32 - kOutFrac));
                            }
                        }
                    }
                } else {
                    for (uint32_t r = 0; r < kSaM; ++r) {
                        for (uint32_t cb = 0; cb < kSaN; cb += kFfnFusionColLanes) {
                            #pragma HLS PIPELINE II=1
                            for (uint32_t lane = 0; lane < kFfnFusionColLanes; ++lane) {
                            #pragma HLS UNROLL
                            const uint32_t c = cb + lane;
                            fused_kt[r][nt_g * kSaN + c] = 0;
                            }
                        }
                    }
                }
            }
            const uint32_t kt_col = nt_base * kSaN;
            const uint32_t valid_cols = (kt_col + kTileK <= K_im) ? kTileK : (K_im - kt_col);
            requantize_q32_ktile_into_act_cache(fused_kt, rows_valid, kt_col, valid_cols, mid_cache);
        }

        // ---- Phase B: run down with mid_cache as activation ----
        // Build per-kt exponents for the intermediate cache.
        for (uint32_t kt = 0; kt < kt_im; ++kt) {
            for (uint32_t r = 0; r < kSaM; ++r) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    #pragma HLS UNROLL
                    uint32_t glob = kt * kGroupsPerKtile + g;
                    exp_im[kt][r][g] =
                        (glob < k_groups_im) ? static_cast<int8_t>(mid_cache.row_exp[r][glob])
                                             : static_cast<int8_t>(0);
                }
            }
        }

        for (uint32_t nt = 0; nt < nt_out; ++nt) {
            const uint64_t d_w0 = (task.down_q_offset_bytes[0]
                                   + (uint64_t)nt * down_weight_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_w1 = (task.down_q_offset_bytes[1]
                                   + (uint64_t)nt * down_weight_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_s0 = (task.down_scale_offset_bytes[0]
                                   + (uint64_t)nt * down_weight_scale_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_s1 = (task.down_scale_offset_bytes[1]
                                   + (uint64_t)nt * down_weight_scale_n_tile_stride_bytes) / kAxiBytes;

            const uint64_t out_byte_off = task.ffn_dst_offset_bytes
                                        + (uint64_t)mt_row0 * task.output_row_stride_bytes[0];
            const uint32_t out_col_byte_off = nt * kSaN * kOutBytes;

            run_ntile_accumulate(w0, w1, mid_cache,
                                  d_w0, d_w1, d_s0, d_s1,
                                  kt_im, rows_valid, exp_im,
                                  a_stream, w_stream, ctrl_stream, psum_stream,
                                  acc0, zero_bias_tile);
            store_out_acc(out_arena, out_byte_off,
                          task.output_row_stride_bytes[0],
                          rows_valid, out_col_byte_off, acc0);
        }
    }
}
#endif

static void requantize_q32_slab_into_mid_tile(
        uint32_t rows_valid,
        uint32_t valid_cols,
        uint32_t slab,
        ActCache &cache) {
    #pragma HLS INLINE off
    const uint32_t base_word = mid_slab_word_base(slab);
    const uint32_t fused_base_word = fused_slab_word_base(slab);
    gelu_q32_t fused_row[kTileK];
    #pragma HLS BIND_STORAGE variable=fused_row type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=fused_row cyclic factor=kFfnRequantBanks dim=1
    for (uint32_t r = 0; r < kSaM; ++r) {
        for (uint32_t c = 0; c < kTileK; ++c) {
            #pragma HLS PIPELINE II=1
            fused_row[c] = (gelu_q32_t)cache.buf[r][fused_base_word + c];
        }
        gelu_q32_t max_abs = 0;
        for (uint32_t c = 0; c < kTileK; ++c) {
            #pragma HLS PIPELINE II=1
            if (c < valid_cols) {
                const gelu_q32_t v = fused_row[c];
                const gelu_q32_t a = (v < 0) ? (gelu_q32_t)-v : v;
                if (a > max_abs) max_abs = a;
            }
        }
        const int8_t exp = choose_pot_exp_for_q32(max_abs);
        cache.mid_row_exp[slab][r] = (r < rows_valid) ? exp : (int8_t)0;
        for (uint32_t w = 0; w < kActWords64PerKtile; ++w) {
            #pragma HLS PIPELINE II=1
            ap_uint<64> word = 0;
            for (uint32_t b = 0; b < 8; ++b) {
                #pragma HLS UNROLL
                const uint32_t c = w * 8 + b;
                const int8_t q = (r < rows_valid && c < valid_cols)
                    ? quantize_q32_with_exp(fused_row[c], exp)
                    : (int8_t)0;
                word(b * 8 + 7, b * 8) = (uint8_t)q;
            }
            cache.buf[r][base_word + w] = word;
        }
    }
}

// Fixed BM4 FFN scheduler. Each 128-column intermediate tile is generated,
// requantized into the temporary tail of ActCache, and consumed by every down
// output tile before the temporary storage is reused.
static void run_prefill_ffn(
        const axi_word_t *w0,
        const axi_word_t *w1,
        const axi_word_t *act_arena,
        axi_word_t *out_arena,
        const task_t &task,
        ActCache &act_cache,
        DownCtxPool &down_ctx,
        OutAccBatch &work0,
        OutAccBatch &work1,
        out_acc_t bias_cache[kBiasCacheCols],
        a_stream_t &a_stream,
        w_stream_t &w_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream) {
    #pragma HLS INLINE
    const bool vision_bias = task.task_type == GEMM_FUSED_VISION_MLP_GELU_BIAS;
    const uint32_t M = task.rows;
    const uint32_t K_in = task.ffn_input_cols;
    const uint32_t K_im = task.ffn_intermediate_cols;
    const uint32_t N_out = task.ffn_output_cols;
    const uint32_t kt_in = div_ceil(K_in, kTileK);
    const uint32_t kt_im = div_ceil(K_im, kTileK);
    const uint32_t nt_im = div_ceil(K_im, kSaN);
    const uint32_t nt_out = div_ceil(N_out, kSaN);
    const uint32_t k_groups_in = div_ceil(K_in, kGroupSize);
    const uint32_t m_tiles = div_ceil(M, kSaM);
    const uint32_t k_words_in_valid = div_ceil(K_in, kAxiBytes);
    const uint32_t k_words_in = kt_in * kActWordsPerKtile;
    const uint32_t down_nt_stride_w = kWeightWordsPerKtilePerPort * kt_im;
    const uint32_t down_nt_stride_s = kScaleWordsPerKtilePerPort * kt_im;

    static int8_t exp_in[kBatchMtiles][kMaxKtiles][kSaM][kGroupsPerKtile];
    static int8_t exp_mid[kBatchMtiles][kMaxKtiles][kSaM][kGroupsPerKtile];
    static out_acc_t zero_bias[kSaN] = {};
    #pragma HLS ARRAY_PARTITION variable=zero_bias complete dim=1
    if (vision_bias) {
        load_vision_mlp_bias_cache(act_arena,
                                   task.up_bias_offset_bytes,
                                   task.down_bias_offset_bytes,
                                   bias_cache);
    }

    for (uint32_t mt0 = 0; mt0 < m_tiles; mt0 += kBatchMtiles) {
        const uint32_t slabs_valid =
            (mt0 + kBatchMtiles <= m_tiles) ? kBatchMtiles : (m_tiles - mt0);
        uint32_t rows_valid[kBatchMtiles];
        uint32_t act_word_base[kBatchMtiles];
        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            const uint32_t row0 = (mt0 + slab) * kSaM;
            rows_valid[slab] = (row0 + kSaM <= M) ? kSaM : (M - row0);
            act_word_base[slab] = input_slab_word_base(slab);
            load_activation_block(act_arena,
                                  task.act_q_offset_bytes + (uint64_t)row0 * task.act_row_stride,
                                  task.act_scale_offset_bytes + (uint64_t)row0 * task.act_scale_row_stride,
                                  task.act_row_stride, task.act_scale_row_stride,
                                  rows_valid[slab], k_words_in_valid, k_words_in,
                                  k_groups_in, slab, act_cache);
            for (uint32_t kt = 0; kt < kt_in; ++kt) {
                for (uint32_t r = 0; r < kSaM; ++r) {
                    for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                        #pragma HLS UNROLL
                        const uint32_t glob = kt * kGroupsPerKtile + g;
                        exp_in[slab][kt][r][g] =
                            (glob < k_groups_in) ? (int8_t)act_cache.row_exp[slab][r][glob] : (int8_t)0;
                    }
                }
            }
        }

        for (uint32_t im_kt = 0; im_kt < kt_im; ++im_kt) {
            for (uint32_t nt_g = 0; nt_g < kNtilesPerKtile; ++nt_g) {
                const uint32_t nt = im_kt * kNtilesPerKtile + nt_g;
                if (nt < nt_im) {
                    const uint64_t u_w0 = (task.up_q_offset_bytes[0]
                        + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_w1 = (task.up_q_offset_bytes[1]
                        + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_s0 = (task.up_scale_offset_bytes[0]
                        + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_s1 = (task.up_scale_offset_bytes[1]
                        + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    out_acc_t up_bias[kSaN];
                    #pragma HLS ARRAY_PARTITION variable=up_bias complete dim=1
                    if (vision_bias) {
                        load_bias_tile_from_cache(bias_cache, nt * kSaN, up_bias);
                    } else {
                        for (uint32_t c = 0; c < kSaN; ++c) {
                            #pragma HLS UNROLL
                            up_bias[c] = 0;
                        }
                    }
                    run_ntile_accumulate_batch(w0, w1, act_cache, act_word_base, slabs_valid,
                                               u_w0, u_w1, u_s0, u_s1, kt_in,
                                               exp_in, a_stream, w_stream, ctrl_stream,
                                               psum_stream, work1, up_bias);

                    if (!vision_bias) {
                        const uint64_t g_w0 = (task.gate_q_offset_bytes[0]
                            + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                        const uint64_t g_w1 = (task.gate_q_offset_bytes[1]
                            + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                        const uint64_t g_s0 = (task.gate_scale_offset_bytes[0]
                            + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                        const uint64_t g_s1 = (task.gate_scale_offset_bytes[1]
                            + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                        run_ntile_accumulate_batch(w0, w1, act_cache, act_word_base, slabs_valid,
                                                   g_w0, g_w1, g_s0, g_s1, kt_in,
                                                   exp_in, a_stream, w_stream, ctrl_stream,
                                                   psum_stream, work0, zero_bias);
                    }

                    for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
                        for (uint32_t r = 0; r < kSaM; ++r) {
                            for (uint32_t c = 0; c < kSaN; ++c) {
                                #pragma HLS PIPELINE II=1
                                const out_acc_t u = work1.slab[slab].v[r][c];
                                gelu_q32_t fused = 0;
                                if (vision_bias) {
                                    fused = V5_gelu_lut128_q32(u);
                                } else {
                                    const out_acc_t g = work0.slab[slab].v[r][c];
                                    const out_acc_t act = task.ffn_activation == LINEAR_FFN_ACT_GELU
                                        ? V5_hard_gelu(g) : V5_hard_silu(g);
                                    fused = saturate_i96_to_gelu_q32(
                                        ((ap_int<96>)round_mul_q22(act, u)) << (32 - kOutFrac));
                                }
                                act_cache.buf[r][fused_slab_word_base(slab)
                                                    + nt_g * kSaN + c] =
                                    (ap_uint<64>)(ap_int<64>)fused;
                            }
                        }
                    }
                } else {
                    for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
                        for (uint32_t r = 0; r < kSaM; ++r) {
                            for (uint32_t c = 0; c < kSaN; ++c) {
                                #pragma HLS PIPELINE II=1
                                act_cache.buf[r][fused_slab_word_base(slab)
                                                    + nt_g * kSaN + c] = 0;
                            }
                        }
                    }
                }
            }

            const uint32_t col0 = im_kt * kTileK;
            const uint32_t valid_cols = (col0 + kTileK <= K_im) ? kTileK : (K_im - col0);
            for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
                requantize_q32_slab_into_mid_tile(rows_valid[slab], valid_cols,
                                                  slab, act_cache);
            }
            uint32_t mid_act_word_base[kBatchMtiles];
            for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
                mid_act_word_base[slab] = mid_slab_word_base(slab);
                for (uint32_t r = 0; r < kSaM; ++r) {
                    #pragma HLS PIPELINE II=1
                    exp_mid[slab][0][r][0] = (int8_t)act_cache.mid_row_exp[slab][r];
                }
            }

            for (uint32_t nt = 0; nt < nt_out; ++nt) {
                const uint64_t d_w0 = task.down_q_offset_bytes[0] / kAxiBytes
                    + (uint64_t)nt * down_nt_stride_w
                    + (uint64_t)im_kt * kWeightWordsPerKtilePerPort;
                const uint64_t d_w1 = task.down_q_offset_bytes[1] / kAxiBytes
                    + (uint64_t)nt * down_nt_stride_w
                    + (uint64_t)im_kt * kWeightWordsPerKtilePerPort;
                const uint64_t d_s0 = task.down_scale_offset_bytes[0] / kAxiBytes
                    + (uint64_t)nt * down_nt_stride_s
                    + (uint64_t)im_kt * kScaleWordsPerKtilePerPort;
                const uint64_t d_s1 = task.down_scale_offset_bytes[1] / kAxiBytes
                    + (uint64_t)nt * down_nt_stride_s
                    + (uint64_t)im_kt * kScaleWordsPerKtilePerPort;
                out_acc_t down_bias[kSaN];
                #pragma HLS ARRAY_PARTITION variable=down_bias complete dim=1
                const bool first_im_kt = im_kt == 0;
                if (first_im_kt && vision_bias) {
                    load_bias_tile_from_cache(bias_cache,
                                              kVisionMlpDownBiasBase + nt * kSaN,
                                              down_bias);
                } else {
                    for (uint32_t c = 0; c < kSaN; ++c) {
                        #pragma HLS UNROLL
                        down_bias[c] = 0;
                    }
                }
                run_ntile_accumulate_batch(w0, w1, act_cache,
                                           mid_act_word_base, slabs_valid,
                                           d_w0, d_w1, d_s0, d_s1, 1,
                                           exp_mid,
                                           a_stream, w_stream, ctrl_stream,
                                           psum_stream, work0, down_bias);
                if (first_im_kt) {
                    merge_out_batch_into_down_ctx_first(work0, slabs_valid, nt, down_ctx);
                } else {
                    merge_out_batch_into_down_ctx_accum(work0, slabs_valid, nt, down_ctx);
                }
            }
        }

        store_down_ctx_pool(out_arena,
                            task.ffn_dst_offset_bytes
                                + (uint64_t)mt0 * kSaM * task.output_row_stride_bytes[0],
                            task.output_row_stride_bytes[0],
                            slabs_valid, rows_valid, nt_out, down_ctx);
    }
}

} // namespace v5
