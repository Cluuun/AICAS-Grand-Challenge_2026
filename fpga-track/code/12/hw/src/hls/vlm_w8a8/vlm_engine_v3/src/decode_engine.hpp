// decode_engine.hpp — Decode top-level: GEMV dispatch for M=1 tasks.
#pragma once

#include "types.hpp"
#include "decode_gemv.hpp"

// Load activation vector into local buffer (one row)
static void load_decode_activation(
        const vlm_w8a8_axi_t *act_arena,
        uint64_t base_word_addr,
        uint32_t k_words,
        int8_t act_buf[kGenericMaxDim]) {
    #pragma HLS INLINE off
    for (uint32_t w = 0; w < k_words; ++w) {
        #pragma HLS PIPELINE II=1
        vlm_w8a8_axi_t word = act_arena[base_word_addr + w];
        for (uint32_t b = 0; b < kAxiBytes; ++b) {
            #pragma HLS UNROLL
            act_buf[w * kAxiBytes + b] = (int8_t)(uint8_t)word(b * 8 + 7, b * 8);
        }
    }
}

// Store decode output (32 int32 values as one row segment)
static void store_decode_output(
        vlm_w8a8_axi_t *out_arena,
        uint64_t base_word_addr,
        uint32_t col_offset,
        const int32_t acc[kDecodeMac]) {
    #pragma HLS INLINE off
    const uint32_t words = kDecodeMac * 4 / kAxiBytes;  // 8
    const uint32_t col_word_off = col_offset * 4 / kAxiBytes;

    for (uint32_t w = 0; w < words; ++w) {
        #pragma HLS PIPELINE II=1
        vlm_w8a8_axi_t word = 0;
        for (uint32_t i = 0; i < kAxiBytes / 4; ++i) {
            #pragma HLS UNROLL
            int32_t val = acc[w * (kAxiBytes / 4) + i];
            word((i + 1) * 32 - 1, i * 32) = static_cast<uint32_t>(val);
        }
        out_arena[base_word_addr + col_word_off + w] = word;
    }
}

// Run decode dense task (GEMV path, M=1)
static void run_decode_dense(
        const vlm_w8a8_axi_t *act_arena,
        vlm_w8a8_axi_t *out_arena,
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        const accelerator_task_t &task,
        vlm_w8a8_profile_t &profile) {
    #pragma HLS INLINE off

    const uint32_t input_cols = task.input_cols;
    const uint32_t input_padded = align_up(input_cols, kTileK);
    const uint32_t k_words = input_padded / kAxiBytes;

    int8_t act_buf[kGenericMaxDim];
    #pragma HLS BIND_STORAGE variable=act_buf type=ram_1p impl=bram
    int32_t acc[kDecodeMac];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    // Load activation once
    uint64_t act_base = task_act_offset(task) / kAxiBytes;
    load_decode_activation(act_arena, act_base, k_words, act_buf);

    for (uint32_t out_idx = 0; out_idx < task.output_count; ++out_idx) {
        const uint32_t out_cols = task.out_cols[out_idx];
        const uint32_t n_groups = div_ceil(out_cols, kDecodeMac);
        uint64_t out_base = task_out_offset(task, out_idx) / kAxiBytes;

        for (uint32_t ng = 0; ng < n_groups; ++ng) {
            uint64_t w0_base = task_weight_offset(task, out_idx) / kAxiBytes + (uint64_t)ng * input_padded;
            uint64_t w1_base = task_weight1_offset(task, out_idx) / kAxiBytes + (uint64_t)ng * input_padded;

            decode_gemv_group(weight0_arena, weight1_arena, w0_base, w1_base,
                            act_buf, input_padded, acc);
            store_decode_output(out_arena, out_base, ng * kDecodeMac, acc);
        }
    }
}
