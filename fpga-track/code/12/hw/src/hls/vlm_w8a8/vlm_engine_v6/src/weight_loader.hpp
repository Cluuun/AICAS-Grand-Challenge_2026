// weight_loader.hpp --- streaming weight K-tile loader with ping-pong buffer.
//
// Source layout (v6, K-major):
//   For each (n_tile_idx, k_tile_idx, k=0..127) on each port:
//     four 128b words carry the 64 port-local N lanes for that K position.
//   Total = 512 AXI words = 8192 B / port / kt.
//
// Streaming protocol:
//   For one K-tile we emit kTileK lockstep beats on two 512b streams:
//     w0 lane 0..63 from port0, w1 lane 64..127 from port1.
//
// Cycles:
//   load_weight_kt: kWeightWordsPerKtilePerPort + ~6 = ~518 (II=1 dual-port)
//   push_weight_kt_to_stream: kTileK + ~6 = ~134
//
// Ping-pong: buf[2][...]. While compute consumes ping, loader fills pong.
#pragma once

#include "streams.hpp"

namespace v6 {

struct WeightBuf {
    // [port][k][word] -> 64 N-lane bytes for this K position per port.
    axi_word_t words[2][kTileK][kWeightWordsPerKPerPort];
};

// Load one K-tile from both HP ports. K-major packing lets the compute stage
// read one word per port per cycle without lane-partitioning tiny memories.
// Estimated cycles: kWeightWordsPerKtilePerPort + 6 (~518)
static void load_weight_kt_dual(
        const axi_word_t *w0,
        const axi_word_t *w1,
        uint64_t w0_kt_base_words,
        uint64_t w1_kt_base_words,
        WeightBuf &dst) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=dst.words complete dim=1
    const uint32_t words = kWeightWordsPerKtilePerPort; // 512
    for (uint32_t i = 0; i < words; ++i) {
        #pragma HLS PIPELINE II=1
        const uint32_t k = i / kWeightWordsPerKPerPort;
        const uint32_t w = i % kWeightWordsPerKPerPort;
        dst.words[0][k][w] = w0[w0_kt_base_words + i];
        dst.words[1][k][w] = w1[w1_kt_base_words + i];
    }
}

// Emit kTileK stream beats from a filled WeightBuf. Each beat = 128 cols.
// Estimated cycles: kTileK + 4
static void push_weight_kt_to_stream(
        const WeightBuf &src,
        w_stream_t &w0_stream,
        w_stream_t &w1_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=src.words complete dim=1
    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        w_half_stream_word_t beat0 = 0;
        w_half_stream_word_t beat1 = 0;
        for (uint32_t w = 0; w < kWeightWordsPerKPerPort; ++w) {
            #pragma HLS UNROLL
            beat0(w * kAxiWidth + kAxiWidth - 1, w * kAxiWidth) = src.words[0][k][w];
            beat1(w * kAxiWidth + kAxiWidth - 1, w * kAxiWidth) = src.words[1][k][w];
        }
        w0_stream.write(beat0);
        w1_stream.write(beat1);
    }
}

} // namespace v6
