// prefill_ctrl.hpp — Prefill engine control: weight/act loading, compute dispatch.
#pragma once

#include "types.hpp"

// Load one K-tile of weights for one port (128 words = 2048 bytes)
static void load_weight_ktile_port(
        const vlm_w8a8_axi_t *weight_arena,
        uint64_t base_word_addr,
        vlm_w8a8_axi_t weight_buf[kWeightWordsPerPort]) {
    #pragma HLS INLINE off
    for (uint32_t i = 0; i < kWeightWordsPerPort; ++i) {
        #pragma HLS PIPELINE II=1
        weight_buf[i] = weight_arena[base_word_addr + i];
    }
}

// Load both weight ports and build the logical 32-column stream:
//   unified[group * 64 + k * 2 + 0] = cols 0..15 for that k
//   unified[group * 64 + k * 2 + 1] = cols 16..31 for that k
static void load_weight_ktile_dual(
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        uint64_t base0_word_addr,
        uint64_t base1_word_addr,
        vlm_w8a8_axi_t weight_buf[kWeightWordsPerKtile]) {
    #pragma HLS INLINE off
    for (uint32_t i = 0; i < kWeightWordsPerPort; ++i) {
        #pragma HLS PIPELINE II=1
        const uint32_t group = i / kGroupSize;
        const uint32_t k = i % kGroupSize;
        const uint32_t out_idx = group * (kGroupSize * VLM_W8A8_WEIGHT_PORTS) + k * VLM_W8A8_WEIGHT_PORTS;
        weight_buf[out_idx] = weight0_arena[base0_word_addr + i];
        weight_buf[out_idx + 1u] = weight1_arena[base1_word_addr + i];
    }
}

// Load scale exponents for one K-tile from both ports.
static void load_weight_scale_ktile_dual(
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        uint64_t base0_word_addr,
        uint64_t base1_word_addr,
        scale_exp_t w_scale[kGroupsPerKtile][kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=w_scale complete dim=0
    for (uint32_t w = 0; w < kScaleWordsPerPort; ++w) {
        #pragma HLS PIPELINE II=1
        vlm_w8a8_axi_t word0 = weight0_arena[base0_word_addr + w];
        vlm_w8a8_axi_t word1 = weight1_arena[base1_word_addr + w];
        for (uint32_t b = 0; b < kAxiBytes; ++b) {
            #pragma HLS UNROLL
            uint32_t flat_idx = w * kAxiBytes + b;
            uint32_t g = flat_idx / kWeightColsPerPort;
            uint32_t c = flat_idx % kWeightColsPerPort;
            if (g < kGroupsPerKtile) {
                w_scale[g][c] = static_cast<scale_exp_t>(static_cast<ap_int<8> >(word0(b * 8 + 7, b * 8)));
                w_scale[g][kWeightColsPerPort + c] = static_cast<scale_exp_t>(static_cast<ap_int<8> >(word1(b * 8 + 7, b * 8)));
            }
        }
    }
}

// Load one activation K-tile for one M-tile (SA_M rows x 128 K elements).
static void load_activation_ktile(
        const vlm_w8a8_axi_t *act_arena,
        uint64_t base_word_addr,
        uint64_t scale_base_byte_addr,
        uint32_t row_start,
        uint32_t row_stride_words,
        uint32_t scale_groups,
        uint32_t kt,
        uint32_t valid_rows,
        vlm_w8a8_axi_t act_buf[kSaM][kActWordsPerKtile],
        scale_exp_t act_scale[kSaM][kGroupsPerKtile]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=act_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=act_scale complete dim=0

    for (uint32_t row = 0; row < kSaM; ++row) {
        const bool valid = row < valid_rows;
        uint64_t row_addr = base_word_addr + (uint64_t)row * row_stride_words + (uint64_t)kt * kActWordsPerKtile;
        for (uint32_t w = 0; w < kActWordsPerKtile; ++w) {
            #pragma HLS PIPELINE II=1
            act_buf[row][w] = valid ? act_arena[row_addr + w] : vlm_w8a8_axi_t(0);
        }
        for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
            #pragma HLS PIPELINE II=1
            scale_exp_t exp = 0;
            const uint32_t global_group = kt * kGroupsPerKtile + g;
            if (valid && global_group < scale_groups && scale_base_byte_addr != 0u) {
                const uint64_t byte_addr = scale_base_byte_addr + static_cast<uint64_t>(row_start + row) * scale_groups + global_group;
                const vlm_w8a8_axi_t word = act_arena[byte_addr / kAxiBytes];
                const uint32_t byte_idx = byte_addr % kAxiBytes;
                exp = static_cast<scale_exp_t>(static_cast<ap_int<8> >(word(byte_idx * 8 + 7, byte_idx * 8)));
            }
            act_scale[row][g] = exp;
        }
    }
}
