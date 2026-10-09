// weight_loader.hpp --- streaming weight K-tile loader with ping-pong buffer.
//
// Source layout (v5, K-major):
//   For each (n_tile_idx, k_tile_idx, k=0..127) on each port:
//     one 128b word carries the 16 port-local N lanes for that K position.
//   Total = 128 AXI words = 2048 B / port / kt.
//
// Streaming protocol:
//   For one K-tile we emit kTileK beats of 256b = 32 weight bytes / cycle.
//   Each cycle's beat carries weights for the 32 columns at one K position.
//   Inside a 256b beat: lane 0..15 from port0, lane 16..31 from port1.
//
// Cycles:
//   load_weight_kt: kWeightWordsPerKtilePerPort + ~6 = ~134 (II=1 dual-port)
//   push_weight_kt_to_stream: kTileK + ~6 = ~134
//
// Ping-pong: buf[2][...]. While compute consumes ping, loader fills pong.
#pragma once

#include "streams.hpp"

namespace v5 {

struct WeightBuf {
    // [port][k] -> 16 N-lane bytes for this K position.
    axi_word_t words[2][kTileK];
};

// Load one K-tile from both HP ports. K-major packing lets the compute stage
// read one word per port per cycle without lane-partitioning tiny memories.
// Estimated cycles: kWeightWordsPerKtilePerPort + 6 (~134)
static void load_weight_kt_dual(
        const axi_word_t *w0,
        const axi_word_t *w1,
        uint64_t w0_kt_base_words,
        uint64_t w1_kt_base_words,
        WeightBuf &dst) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=dst.words complete dim=1
    const uint32_t words = kWeightWordsPerKtilePerPort; // 128
    for (uint32_t i = 0; i < words; ++i) {
        #pragma HLS PIPELINE II=1
        dst.words[0][i] = w0[w0_kt_base_words + i];
        dst.words[1][i] = w1[w1_kt_base_words + i];
    }
}

// Emit kTileK stream beats from a filled WeightBuf. Each beat = 32 cols.
// Estimated cycles: kTileK + 4
static void push_weight_kt_to_stream(
        const WeightBuf &src,
        w_stream_t &w_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=src.words complete dim=1
    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        stream_word_t beat = 0;
        beat(127, 0)   = src.words[0][k];
        beat(255, 128) = src.words[1][k];
        w_stream.write(beat);
    }
}

} // namespace v5
