// act_cache.hpp --- m_tile-level activation cache.
//
// Layout (URAM-backed):
//   act_cache[row=0..kSaM-1][word] : 64-bit packed int8
//   Each row bank holds four input slabs, four temporary A8 FFN tiles, and four
//   temporary Q32 fusion tiles. The 2112-word depth fits in one URAM per row bank.
//   Per-row scale exponents are kept separately.
//
// Lifecycle:
//   * load_activation_block(): once per m_tile, burst-fills the valid K range
//     only, then zero-fills the padded K tail. Rows beyond rows_valid are
//     zeroed over the full padded range.
//   * push_act_kt_to_stream(): per K-tile during compute, emits 32 cycles
//     of stream beats (1 cycle per K element, 32 active rows / cycle).
//
// Cycles:
//   load = kSaM * (valid_words64 + tail_words64) + kSaM * active_scale_words + ~10
//   push = kTileK + ~4
#pragma once

#include "streams.hpp"

namespace v5 {

struct ActCache {
    ap_uint<64> buf[kSaM][kActCacheWords64PerRow];
    scale_exp_t row_exp[kBatchMtiles][kSaM][kActCacheK / kGroupSize];
    scale_exp_t mid_row_exp[kBatchMtiles][kSaM];
};

static inline uint32_t input_slab_word_base(uint32_t slab) {
    #pragma HLS INLINE
    return slab * kActWords64PerRow;
}

static inline uint32_t mid_slab_word_base(uint32_t slab) {
    #pragma HLS INLINE
    return kActInputBatchWords64PerRow + slab * kActWords64PerKtile;
}

static inline uint32_t fused_slab_word_base(uint32_t slab) {
    #pragma HLS INLINE
    return kActInputBatchWords64PerRow + kActMidBatchWords64PerRow + slab * kTileK;
}

// Load activation block for one m_tile. Rows beyond rows_valid are zeroed.
static void load_activation_block(
        const axi_word_t *act_arena,
        uint64_t act_q_byte_offset,
        uint64_t act_scale_byte_offset,
        uint32_t row_stride_bytes,
        uint32_t scale_stride_bytes,
        uint32_t rows_valid,
        uint32_t k_valid_words,     // ceil(valid K / kAxiBytes)
        uint32_t k_padded_words,    // padded active K / kAxiBytes
        uint32_t k_groups,          // ceil(valid K / kGroupSize)
        uint32_t slab,
        ActCache &cache) {
    #pragma HLS INLINE off
    const uint32_t q_word_base    = act_q_byte_offset / kAxiBytes;
    const uint32_t row_stride_w   = row_stride_bytes  / kAxiBytes;
    const uint32_t scale_word_base = act_scale_byte_offset / kAxiBytes;
    const uint32_t scale_stride_w  = scale_stride_bytes / kAxiBytes;

    const uint32_t valid_w64 = k_valid_words * 2u;
    const uint32_t active_w64 = k_padded_words * 2u;
    const uint32_t scale_words_per_row = (k_groups + kAxiBytes - 1u) / kAxiBytes;
    const uint32_t cache_word_base = input_slab_word_base(slab);

    // Valid rows: straight-line burst reads, no conditional around the AXI
    // load, so HLS can infer bursts for the effective K range.
    for (uint32_t r = 0; r < rows_valid; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kSaM
        axi_word_t pair = 0;
        for (uint32_t w64 = 0; w64 < valid_w64; ++w64) {
            #pragma HLS PIPELINE II=1
            if ((w64 & 1u) == 0u) {
                const uint32_t src_w = w64 >> 1;
                pair = act_arena[q_word_base + r * row_stride_w + src_w];
            }
            cache.buf[r][cache_word_base + w64] =
                ((w64 & 1u) == 0u) ? pair(63, 0) : pair(127, 64);
        }
        for (uint32_t w64 = valid_w64; w64 < active_w64; ++w64) {
            #pragma HLS PIPELINE II=1
            cache.buf[r][cache_word_base + w64] = 0;
        }
    }

    // Invalid rows still participate in PE compute; zero only the active K
    // range that this task will actually stream.
    for (uint32_t r = rows_valid; r < kSaM; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=kSaM
        #pragma HLS LOOP_FLATTEN off
        for (uint32_t w64 = 0; w64 < active_w64; ++w64) {
            #pragma HLS PIPELINE II=1
            cache.buf[r][cache_word_base + w64] = 0;
        }
    }

    // Per-row scale exponents (1 byte / group).
    for (uint32_t r = 0; r < rows_valid; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kSaM
        #pragma HLS LOOP_FLATTEN off
        for (uint32_t w = 0; w < scale_words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            axi_word_t v = act_arena[scale_word_base + r * scale_stride_w + w];
            for (uint32_t b = 0; b < kAxiBytes; ++b) {
                #pragma HLS UNROLL
                uint32_t g = w * kAxiBytes + b;
                if (g < k_groups) {
                    cache.row_exp[slab][r][g] = (int8_t)(uint8_t)v(b * 8 + 7, b * 8);
                }
            }
        }
    }
    for (uint32_t r = rows_valid; r < kSaM; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=0 max=kSaM
        #pragma HLS LOOP_FLATTEN off
        for (uint32_t w = 0; w < scale_words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            for (uint32_t b = 0; b < kAxiBytes; ++b) {
                #pragma HLS UNROLL
                uint32_t g = w * kAxiBytes + b;
                if (g < k_groups) {
                    cache.row_exp[slab][r][g] = 0;
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
        uint32_t cache_word_base,
        uint32_t k_tile_idx,
        a_stream_t &a_stream) {
    #pragma HLS INLINE off

    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        const uint32_t k_idx = k_tile_idx * kTileK + k;
        stream_word_t beat = 0;
        for (uint32_t r = 0; r < kSaM; ++r) {
            #pragma HLS UNROLL
            const ap_uint<64> word = cache.buf[r][cache_word_base + k_idx / 8];
            const uint32_t byte_idx = k_idx & 7u;
            beat(r * 8 + 7, r * 8) = word(byte_idx * 8 + 7, byte_idx * 8);
        }
        a_stream.write(beat);
    }
}

} // namespace v5
