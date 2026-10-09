#pragma once

#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#  include "../common/vh_dims.hpp"
#else
#  include "../common/types.hpp"
#endif

// A1-P1: W=128 fused per-head attention (QK^T -> softmax -> AV).
// Q/K/V/O DDR interfaces stay FP32; internal QK/AV use ap_fixed<16,8> MAC + fp32 acc.
// P0 FP32 kernel remains in attention.cpp as k_attention.

extern "C" void k_attention_fused(
    const float * Q,
    const float * K,
    const float * V,
    float       * O_out);
