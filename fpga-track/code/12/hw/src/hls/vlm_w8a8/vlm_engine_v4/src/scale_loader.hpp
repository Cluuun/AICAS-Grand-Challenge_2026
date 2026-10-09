// scale_loader.hpp --- weight scale (per K-group) loader.
//
// Host packed layout per (n_tile, k_tile, port): for each lane in 0..15,
//   for each group in 0..3, two bytes (POT e0 then legacy APOT e1).
// The v4 hardware consumes only e0 and skips e1 for ABI compatibility.
// Total bytes per K-tile per port = 16*4*2 = 128 = 8 axi words.
//
// Cycles: kScaleWordsPerKtilePerPort * 1 = 8 cycles per kt
#pragma once

#include "types.hpp"

namespace v4 {

struct ScaleBuf {
    // [group=0..3][col=0..31] -> POT exponent.
    int8_t e0[kGroupsPerKtile][kSaN];
};

// Load a single K-tile's weight scales from both ports.
// Estimated cycles: kScaleWordsPerKtilePerPort (~8)
static void load_weight_scale_kt(
        const axi_word_t *w0,
        const axi_word_t *w1,
        uint64_t w0_kt_scale_words,
        uint64_t w1_kt_scale_words,
        ScaleBuf &dst) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=dst.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dst.e0 cyclic factor=kScaleDrainColLanes dim=2
    for (uint32_t i = 0; i < kScaleWordsPerKtilePerPort; ++i) {
        #pragma HLS PIPELINE II=1
        axi_word_t v0 = w0[w0_kt_scale_words + i];
        axi_word_t v1 = w1[w1_kt_scale_words + i];
        for (uint32_t b = 0; b < kAxiBytes; ++b) {
            #pragma HLS UNROLL
            uint32_t flat = i * kAxiBytes + b;       // 0..127
            uint32_t entry = flat / kScaleBytesPerSlot;  // 0..63
            uint32_t which_e = flat % kScaleBytesPerSlot; // 0=e0, 1=e1 (skipped)
            uint32_t lane    = entry / kGroupsPerKtile;   // 0..15
            uint32_t group   = entry % kGroupsPerKtile;   // 0..3
            if (which_e == 0u && lane < kWeightColsPerPort && group < kGroupsPerKtile) {
                const int8_t x0 = (int8_t)(uint8_t)v0(b * 8 + 7, b * 8);
                const int8_t x1 = (int8_t)(uint8_t)v1(b * 8 + 7, b * 8);
                dst.e0[group][lane]                      = x0;
                dst.e0[group][lane + kWeightColsPerPort] = x1;
            }
        }
    }
}

} // namespace v4
