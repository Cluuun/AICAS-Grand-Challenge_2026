// prefill_scale.hpp - scale post-processor.
// Keep this narrower than the dot array: scale is not the throughput limiter,
// and fully parallel dynamic shifts are expensive in LUTs.
#pragma once

#include "types.hpp"

static int32_t apply_exp_shift_i32(int32_t partial, scale_exp_t exp) {
    #pragma HLS INLINE
    const int shift = static_cast<int>(exp);
    if (shift >= 31) {
        return 0;
    }
    if (shift >= 0) {
        return partial << shift;
    }
    const int rshift = -shift;
    if (rshift >= 31) {
        return 0;
    }
    return partial >> rshift;
}

// Apply scale shift to one group's partial sums and accumulate into output buffer.
static void apply_scale_group(
        int32_t pe_acc[kSaM][kSaN],
        const scale_exp_t act_exp[kSaM],
        const scale_exp_t w_exp[kSaN],
        int32_t out_acc[kSaM][kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=pe_acc complete dim=2
    #pragma HLS ARRAY_PARTITION variable=out_acc complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_exp complete

    static constexpr uint32_t kScaleColsPerCycle = 4;
    for (uint32_t row = 0; row < kSaM; ++row) {
        scale_exp_t a_exp = act_exp[row];
        for (uint32_t col_base = 0; col_base < kSaN; col_base += kScaleColsPerCycle) {
            #pragma HLS PIPELINE II=1
            for (uint32_t lane = 0; lane < kScaleColsPerCycle; ++lane) {
                #pragma HLS UNROLL
                const uint32_t col = col_base + lane;
                int32_t partial = pe_acc[row][col];
                scale_exp_t shift = a_exp + w_exp[col];
                int32_t scaled = apply_exp_shift_i32(partial, shift);
                out_acc[row][col] += scaled;
            }
        }
    }
}

// Clear PE accumulator array
static void clear_pe_acc(int32_t pe_acc[kSaM][kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=pe_acc complete dim=2
    for (uint32_t row = 0; row < kSaM; ++row) {
        #pragma HLS PIPELINE II=1
        for (uint32_t col = 0; col < kSaN; ++col) {
            #pragma HLS UNROLL
            pe_acc[row][col] = 0;
        }
    }
}

// Clear output accumulator
static void clear_out_acc(int32_t out_acc[kSaM][kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=out_acc complete dim=2
    for (uint32_t row = 0; row < kSaM; ++row) {
        #pragma HLS PIPELINE II=1
        for (uint32_t col = 0; col < kSaN; ++col) {
            #pragma HLS UNROLL
            out_acc[row][col] = 0;
        }
    }
}
