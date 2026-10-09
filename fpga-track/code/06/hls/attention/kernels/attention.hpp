#pragma once

#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#  include "../common/vh_dims.hpp"
#else
#  include "../common/types.hpp"
#endif

// K_ATTN — fused multi-head attention scoring + softmax + value aggregation
// for one transformer layer.
//
// This kernel deliberately fuses Q @ K^T, softmax(scaled), and (S @ V) for
// each head so the 1024x1024 scores tensor never lives in DDR (see
// docs/operator_inventory.md "Critical bottleneck").  Q/K/V are already
// laid out as [M, C] with channel-major head packing (Q[m*C + h*DH + d]).
//
// Inputs:
//   Q, K, V : FP32 [M=1024, C=768]   in DDR via AXI-MM
// Output:
//   O_out   : FP32 [M=1024, C=768]   attention output before W_o projection
//
// Internal scratch O has shape [N_HEAD, M, DH] (one head at a time fits
// in BRAM/URAM; for S1 we use stack arrays sized for the SmolVLM2 config).
extern "C" void k_attention(
    const float * Q,
    const float * K,
    const float * V,
    float       * O_out);
