#pragma once

// fp32 softmax exp via BRAM LUT + linear interpolation.
//
// Timing (A1-P1.5): index path is fixed bit-slice + saturating cast (fast).
// Interpolation uses ap_ufixed<8,-8> frac (no uint->float uitofp, was 3.224 ns).
// exp_lut_eval is a multi-cycle submodule (3 pipelined sub-stages, auto depth
// ~18) so ROM + fixed lerp + float export retime across stages, SM_EXP II=1.

#ifndef _GLIBCXX_HAS_GTHREADS
#  define _GLIBCXX_HAS_GTHREADS 0
#endif
#include "ap_fixed.h"
#include "attn_precision.hpp"

namespace vh_attn {

constexpr int   EXP_LUT_N   = 1024;
constexpr float EXP_X_MIN   = -16.0f;
constexpr float EXP_X_MAX   =  0.0f;
constexpr float EXP_LUT_STEP = (EXP_X_MAX - EXP_X_MIN) / (float) (EXP_LUT_N - 1);

typedef ap_ufixed<18, 4, AP_TRN, AP_SAT> exp_u_t;
typedef ap_fixed<25, 8>                  lut_amp_t;

static const float k_exp_lut_rom[EXP_LUT_N] = {
#include "attn_exp_lut_table.inc"
};

// Stage A — index + fraction (combinational, < 1 ns)
inline void exp_lut_index(logit_t x, ap_uint<10>& i0, ap_uint<8>& ff) {
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#endif
    exp_u_t     u   = x - (logit_t) EXP_X_MIN;
    ap_uint<18> raw = u.range(17, 0);
    i0              = raw.range(17, 8);
    ff              = raw.range(7, 0);
}

// Stage B — dual-port ROM read (BRAM latency retimed inside module)
inline void exp_lut_fetch(ap_uint<10> i0, lut_amp_t& v0, lut_amp_t& v1) {
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS BIND_STORAGE variable=k_exp_lut_rom type=rom_np impl=bram latency=3
#endif
    v0 = (lut_amp_t) k_exp_lut_rom[i0];
    const int i1 =
        (i0 < (ap_uint<10>) (EXP_LUT_N - 1)) ? (int) (i0 + 1) : (int) i0;
    v1 = (lut_amp_t) k_exp_lut_rom[i1];
}

// Stage C — fixed fractional lerp (no uitofp / float frac multiply)
inline float exp_lut_interp(lut_amp_t v0, lut_amp_t v1, ap_uint<8> ff) {
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#endif
    ap_ufixed<8, -8> frac;
    frac = ff;
    const lut_amp_t dv = v1 - v0;
    const lut_amp_t acc = v0 + dv * frac;
    return (float) acc;
}

inline float exp_lut_eval(logit_t x) {
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#pragma HLS INLINE off
#endif
    ap_uint<10> i0;
    ap_uint<8>  ff;
    lut_amp_t   v0, v1;
    exp_lut_index(x, i0, ff);
    exp_lut_fetch(i0, v0, v1);
    return exp_lut_interp(v0, v1, ff);
}

}  // namespace vh_attn
