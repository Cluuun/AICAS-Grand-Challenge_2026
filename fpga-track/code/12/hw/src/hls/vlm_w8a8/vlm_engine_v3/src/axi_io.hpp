// axi_io.hpp — AXI burst read/write primitives and task loading.
#pragma once

#include "types.hpp"

// Load task descriptor from activation arena
static void load_task(
        const vlm_w8a8_axi_t *act_arena,
        uint64_t byte_offset,
        accelerator_task_t *task) {
    #pragma HLS INLINE off
    const uint32_t word_offset = byte_offset / kAxiBytes;
    const uint32_t task_words = sizeof(accelerator_task_t) / kAxiBytes;
    ap_uint<8> buf[sizeof(accelerator_task_t)];
    #pragma HLS ARRAY_PARTITION variable=buf complete

    for (uint32_t w = 0; w < task_words; ++w) {
        #pragma HLS PIPELINE II=1
        vlm_w8a8_axi_t word = act_arena[word_offset + w];
        for (uint32_t b = 0; b < kAxiBytes; ++b) {
            #pragma HLS UNROLL
            buf[w * kAxiBytes + b] = word(b * 8 + 7, b * 8);
        }
    }  
    // estimated cycles: 73 (first word estimated) + 336/16 (task size / bytes per cycle) = 73 + 21 = 94 cycles

    uint8_t *dst = reinterpret_cast<uint8_t *>(task);
    for (uint32_t i = 0; i < sizeof(accelerator_task_t); ++i) {
        #pragma HLS UNROLL
        dst[i] = buf[i];
    }
}

// Store profile to output arena
static void store_profile(
        vlm_w8a8_axi_t *out_arena,
        uint64_t byte_offset,
        const vlm_w8a8_profile_t &profile) {
    #pragma HLS INLINE off
    const uint32_t word_offset = byte_offset / kAxiBytes;
    const uint32_t prof_words = sizeof(vlm_w8a8_profile_t) / kAxiBytes;
    const uint8_t *src = reinterpret_cast<const uint8_t *>(&profile);

    for (uint32_t w = 0; w < prof_words; ++w) {
        #pragma HLS PIPELINE II=1
        vlm_w8a8_axi_t word = 0;
        for (uint32_t b = 0; b < kAxiBytes; ++b) {
            #pragma HLS UNROLL
            word(b * 8 + 7, b * 8) = src[w * kAxiBytes + b];
        }
        out_arena[word_offset + w] = word;
    }
}
