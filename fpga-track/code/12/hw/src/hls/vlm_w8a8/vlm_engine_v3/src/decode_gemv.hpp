// decode_gemv.hpp — Activation-stationary GEMV for decode (M=1).
// 32 MAC units, bandwidth-matched to 2×128-bit AXI ports.
#pragma once

#include "types.hpp"

// Run GEMV for one output group (32 columns).
// Activation is pre-loaded in act_buf. Weights stream from DDR.
static void decode_gemv_group(
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        uint64_t w0_base_word,
        uint64_t w1_base_word,
        const int8_t act_buf[kGenericMaxDim],
        uint32_t k_dim,
        int32_t acc[kDecodeMac]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=acc complete

    for (uint32_t i = 0; i < kDecodeMac; ++i) {
        #pragma HLS UNROLL
        acc[i] = 0;
    }

    for (uint32_t k_idx = 0; k_idx < k_dim; ++k_idx) {
        #pragma HLS PIPELINE II=1
        vlm_w8a8_axi_t w0_word = weight0_arena[w0_base_word + k_idx];
        vlm_w8a8_axi_t w1_word = weight1_arena[w1_base_word + k_idx];
        int8_t a_val = act_buf[k_idx];

        for (uint32_t b = 0; b < kAxiBytes; ++b) {
            #pragma HLS UNROLL
            int8_t w0_val = (int8_t)(uint8_t)w0_word(b * 8 + 7, b * 8);
            int8_t w1_val = (int8_t)(uint8_t)w1_word(b * 8 + 7, b * 8);
            acc[b] += (int32_t)a_val * (int32_t)w0_val;
            acc[kAxiBytes + b] += (int32_t)a_val * (int32_t)w1_val;
        }
    }
}
