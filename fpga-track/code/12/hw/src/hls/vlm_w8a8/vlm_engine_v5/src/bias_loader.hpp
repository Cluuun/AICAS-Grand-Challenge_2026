// bias_loader.hpp --- Shared Q22/int40 bias loading helpers for v5.
#pragma once

#include "scale_drain.hpp"

namespace v5 {

static constexpr uint32_t kBiasCacheCols = 4096;

static inline out_acc_t bias_i64_to_out(ap_int<64> value) {
    #pragma HLS INLINE
    return saturate_i96_to_out((ap_int<96>)value);
}

static inline void unpack_bias_pair_q22_i40(
        axi_word_t word,
        out_acc_t &lo,
        out_acc_t &hi) {
    #pragma HLS INLINE
    lo = bias_i64_to_out((ap_int<64>)(ap_uint<64>)word(63, 0));
    hi = bias_i64_to_out((ap_int<64>)(ap_uint<64>)word(127, 64));
}

static void clear_bias_cache(out_acc_t bias_cache[kBiasCacheCols]) {
    #pragma HLS INLINE off
    for (uint32_t i = 0; i < kBiasCacheCols; ++i) {
        #pragma HLS PIPELINE II=1
        bias_cache[i] = 0;
    }
}

// Bias vectors are written by the host with 128-bit alignment. Each AXI beat
// holds two sign-extended int64 Q22 values, so this loader consumes the full
// 128-bit bus explicitly instead of issuing two reads to the same beat.
static void load_bias_vector_q22_i40(
        const axi_word_t *act_arena,
        uint64_t bias_offset,
        uint32_t cols,
        uint32_t cache_base,
        out_acc_t bias_cache[kBiasCacheCols]) {
    #pragma HLS INLINE off
    if (bias_offset == 0) {
        for (uint32_t c = 0; c < cols; ++c) {
            #pragma HLS PIPELINE II=1
            const uint32_t idx = cache_base + c;
            if (idx < kBiasCacheCols) {
                bias_cache[idx] = 0;
            }
        }
        return;
    }

    const uint32_t pairs = (cols + 1u) >> 1;
    const uint64_t word_base = bias_offset / kAxiBytes;
    for (uint32_t p = 0; p < pairs; ++p) {
        #pragma HLS PIPELINE II=1
        out_acc_t lo = 0;
        out_acc_t hi = 0;
        unpack_bias_pair_q22_i40(act_arena[word_base + p], lo, hi);
        const uint32_t c0 = cache_base + (p << 1);
        const uint32_t c1 = c0 + 1u;
        if (c0 < kBiasCacheCols) {
            bias_cache[c0] = lo;
        }
        if ((p << 1) + 1u < cols && c1 < kBiasCacheCols) {
            bias_cache[c1] = hi;
        }
    }
}

static void load_bias_tile_from_cache(
        const out_acc_t bias_cache[kBiasCacheCols],
        uint32_t base_col,
        out_acc_t bias_tile[kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1
    for (uint32_t c = 0; c < kSaN; ++c) {
        #pragma HLS PIPELINE II=1
        // All v5 call sites pass validated model tensor offsets. Avoiding the
        // bounds mux here keeps the bias-tile loader out of the outer FFN loop
        // control critical path.
        bias_tile[c] = bias_cache[base_col + c];
    }
}

} // namespace v5
