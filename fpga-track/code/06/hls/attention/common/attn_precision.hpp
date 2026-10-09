#pragma once

// A1-P1 attention MAC precision — do NOT use ap_fixed<16,6> (12-layer cos ~0.987).
// QK^T / AV multiply operands: ap_fixed<16,8>; accumulators stay fp32.

#ifndef _GLIBCXX_HAS_GTHREADS
#  define _GLIBCXX_HAS_GTHREADS 0
#endif
#include "ap_fixed.h"

namespace vh_attn {

// Match S2 envelope (<24,8>); avoid <16,6>-class collapse on 12-layer stack.
// Default round/truncate + wrap keeps the float->fixed converters cheap (no
// AP_RND/AP_SAT barrel-shift+saturate logic, which exploded LUT x128 in the
// per-head cache load). Q/K/V magnitudes stay well inside +/-128 here, so wrap
// is safe (verified: cos unchanged).
typedef ap_fixed<24, 8>  mac_t;
typedef float            acc_t;

// Fixed-point reduction accumulators (sizing §2.1: ap_fixed<~,~> accumulator).
// Fixed adders are ~5-10x cheaper in LUT than fp32 adders and complete in one
// cycle, so the dh=64 QK tree and the 1024-long AV reduction stay II=1 / <3 ns
// without the fp32 LUT blow-up.
typedef ap_fixed<48, 24> qk_acc_t;   // sum of 64 (<24,8> x <24,8>) products
typedef ap_fixed<40, 16> av_acc_t;   // sum of 1024 prob*V terms
typedef ap_ufixed<18, 2> prob_t;     // softmax probability in [0,1]
typedef ap_fixed<24, 8>  logit_t;    // QK logit (fixed): lets exp index by pure
                                     // bit-slice instead of a float->fixed
                                     // barrel shift on the softmax timing path

inline mac_t to_mac(float x) { return (mac_t) x; }
inline float to_f(acc_t x) { return (float) x; }

}  // namespace vh_attn
