// act_cache.hpp --- m_tile-level activation cache.
//
// Layout (URAM-backed):
//   act_cache[row=0..kSaM-1][word=0..kActCacheK/8-1] : 64-bit packed int8
//   Each row holds K=kActCacheK bytes of token-major quantized activation.
//   Per-row scale exponents are kept separately.
//
// Lifecycle:
//   * load_activation_block(): once per m_tile, burst-fills the active padded
//     K range only. Rows beyond rows_valid are zeroed over that same range.
//   * push_act_kt_to_stream(): per K-tile during compute, emits 32 cycles
//     of stream beats (1 cycle per K element, 32 active rows / cycle).
//
// Cycles:
//   load = kSaM * active_words64 + kSaM * active_scale_words + ~10
//   push = kTileK + ~4
#pragma once

#include "streams.hpp"

namespace v4 {

struct ActCache {
    ap_uint<64> buf[kSaM][kActCacheK / 8];
    scale_exp_t row_exp[kSaM][kActCacheK / kGroupSize]; // 96 groups max for K=3072/32
};

// Load activation block for one m_tile. Rows beyond rows_valid are zeroed.
static void load_activation_block(
        const axi_word_t *act_arena,
        uint64_t act_q_byte_offset,
        uint64_t act_scale_byte_offset,
        uint32_t row_stride_bytes,
        uint32_t scale_stride_bytes,
        uint32_t rows_valid,
        uint32_t k_words,           // padded active K / kAxiBytes
        uint32_t k_groups,          // == K / kGroupSize
        ActCache &cache) {
    #pragma HLS INLINE off
    const uint32_t q_word_base    = act_q_byte_offset / kAxiBytes;
    const uint32_t row_stride_w   = row_stride_bytes  / kAxiBytes;
    const uint32_t scale_word_base = act_scale_byte_offset / kAxiBytes;
    const uint32_t scale_stride_w  = scale_stride_bytes / kAxiBytes;

    const uint32_t active_w64 = k_words * 2u;
    const uint32_t scale_words_per_row = (k_groups + kAxiBytes - 1u) / kAxiBytes;

    // Valid rows: straight-line burst reads, no conditional around the AXI
    // load, so HLS can infer bursts for the effective K range.
    for (uint32_t r = 0; r < rows_valid; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kSaM
        axi_word_t pair = 0;
        for (uint32_t w64 = 0; w64 < active_w64; ++w64) {
            #pragma HLS PIPELINE II=1
            if ((w64 & 1u) == 0u) {
                const uint32_t src_w = w64 >> 1;
                pair = act_arena[q_word_base + r * row_stride_w + src_w];
            }
            cache.buf[r][w64] = ((w64 & 1u) == 0u) ? pair(63, 0) : pair(127, 64);
        }
    }

    // Invalid rows still participate in PE compute; zero only the active K
    // range that this task will actually stream.
    for (uint32_t r = rows_valid; r < kSaM; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=kSaM
        for (uint32_t w64 = 0; w64 < active_w64; ++w64) {
            #pragma HLS PIPELINE II=1
            cache.buf[r][w64] = 0;
        }
    }

    // Per-row scale exponents (1 byte / group).
    for (uint32_t r = 0; r < rows_valid; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kSaM
        for (uint32_t w = 0; w < scale_words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            axi_word_t v = act_arena[scale_word_base + r * scale_stride_w + w];
            for (uint32_t b = 0; b < kAxiBytes; ++b) {
                #pragma HLS UNROLL
                uint32_t g = w * kAxiBytes + b;
                if (g < k_groups) {
                    cache.row_exp[r][g] = (int8_t)(uint8_t)v(b * 8 + 7, b * 8);
                }
            }
        }
    }
    for (uint32_t r = rows_valid; r < kSaM; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=kSaM
        for (uint32_t w = 0; w < scale_words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            for (uint32_t b = 0; b < kAxiBytes; ++b) {
                #pragma HLS UNROLL
                uint32_t g = w * kAxiBytes + b;
                if (g < k_groups) {
                    cache.row_exp[r][g] = 0;
                }
            }
        }
    }
}

// Stream one K-tile of activation into a_stream. Per K element we emit one
// 256b beat = 32 act bytes (one byte per row).
// Estimated cycles: kTileK + 4
static void push_act_kt_to_stream(
        const ActCache &cache,
        uint32_t k_tile_idx,
        a_stream_t &a_stream) {
    #pragma HLS INLINE off

    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        const uint32_t k_idx = k_tile_idx * kTileK + k;
        stream_word_t beat = 0;
        for (uint32_t r = 0; r < kSaM; ++r) {
            #pragma HLS UNROLL
            const ap_uint<64> word = cache.buf[r][k_idx / 8];
            const uint32_t byte_idx = k_idx & 7u;
            beat(r * 8 + 7, r * 8) = word(byte_idx * 8 + 7, byte_idx * 8);
        }
        a_stream.write(beat);
    }
}

} // namespace v4
