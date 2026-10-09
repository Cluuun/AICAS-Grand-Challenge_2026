// ffn_fusion.hpp — FFN activation functions and mid-result quantization.
#pragma once

#include "types.hpp"

// HardSiLU approximation: x * clamp(x+3, 0, 6) / 6.
// Uses fixed-point multiply: * 2796203 >> 24 ~= /6.
static inline int32_t hard_silu_fixed(int32_t x) {
    #pragma HLS INLINE
    int32_t x_plus_3 = x + 3;
    int32_t clamped = (x_plus_3 < 0) ? 0 : ((x_plus_3 > 6) ? 6 : x_plus_3);
    int64_t prod = (int64_t)x * (int64_t)clamped;
    return (int32_t)((prod * 2796203LL) >> 24);
}

// HardGELU approximation: x * clamp(x+1.5, 0, 3) / 3.
// Uses fixed-point multiply: * 5592405 >> 24 ~= /3.
static inline int32_t hard_gelu_fixed(int32_t x) {
    #pragma HLS INLINE
    int32_t x_plus_1p5 = x + (3 << 7);  // 1.5 in Q8
    int32_t three_q8 = 3 << 8;
    int32_t clamped = (x_plus_1p5 < 0) ? 0 : ((x_plus_1p5 > three_q8) ? three_q8 : x_plus_1p5);
    int64_t prod = (int64_t)x * (int64_t)clamped;
    return (int32_t)((prod * 5592405LL) >> 24);
}

// Quantize int32 values to int8 with power-of-2 scale.
// Returns the scale exponent and writes quantized values.
static void quantize_row_to_int8(
        const int32_t values[kSaN],
        int8_t out_q[kSaN],
        scale_exp_t &out_scale_exp) {
    #pragma HLS INLINE off

    // Find max absolute value
    int32_t max_abs = 0;
    for (uint32_t i = 0; i < kSaN; ++i) {
        #pragma HLS PIPELINE II=1
        int32_t abs_val = (values[i] < 0) ? -values[i] : values[i];
        if (abs_val > max_abs) max_abs = abs_val;
    }

    // Determine shift: find highest bit position
    uint8_t shift = 0;
    if (max_abs > 127) {
        uint32_t tmp = (uint32_t)max_abs;
        while (tmp > 127) {
            tmp >>= 1;
            shift++;
        }
    }
    out_scale_exp = shift;

    // Quantize with rounding
    for (uint32_t i = 0; i < kSaN; ++i) {
        #pragma HLS PIPELINE II=1
        int32_t rounded = (shift > 0) ? ((values[i] + (1 << (shift - 1))) >> shift) : values[i];
        // Clamp to int8
        if (rounded > 127) rounded = 127;
        if (rounded < -128) rounded = -128;
        out_q[i] = (int8_t)rounded;
    }
}

// Apply FFN activation (gate * up with activation function) and quantize.
// gate_acc and up_acc are the raw int32 accumulator outputs.
static void ffn_activate_and_quantize(
        const int32_t gate_acc[kSaM][kSaN],
        const int32_t up_acc[kSaM][kSaN],
        uint32_t activation_type,
        int8_t mid_q[kSaM][kSaN],
        scale_exp_t mid_scale[kSaM]) {
    #pragma HLS INLINE off

    for (uint32_t row = 0; row < kSaM; ++row) {
        int32_t fused[kSaN];

        for (uint32_t col = 0; col < kSaN; ++col) {
            #pragma HLS PIPELINE II=1
            int32_t g = gate_acc[row][col];
            int32_t u = up_acc[row][col];
            int32_t activated;
            if (activation_type == LINEAR_FFN_ACT_SILU)
                activated = hard_silu_fixed(g);
            else
                activated = hard_gelu_fixed(g);
            fused[col] = (int32_t)(((int64_t)activated * (int64_t)u) >> 16);
        }

        quantize_row_to_int8(fused, mid_q[row], mid_scale[row]);
    }
}
