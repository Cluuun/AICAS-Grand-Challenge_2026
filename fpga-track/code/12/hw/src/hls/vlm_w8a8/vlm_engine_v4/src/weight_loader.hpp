// weight_loader.hpp --- streaming weight K-tile loader with ping-pong buffer.
//
// Source layout (ABI v11, unchanged from v3):
//   For each (n_tile_idx, k_tile_idx, lane=0..15) on each port:
//     128 contiguous bytes (one K-tile column). Total = 16 * 128 = 2048 B / port / kt.
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

namespace v4 {

struct WeightBuf {
    // [port][lane][k_word] -> 16 contiguous K bytes packed into one AXI word.
    // Partitioning port/lane gives the compute loop 32 independent banks.
    axi_word_t words[2][kWeightColsPerPort][kTileK / kAxiBytes];
};

// Load one K-tile from both HP ports. The host packs each (lane) as a
// 128-byte contiguous run, so 16 lanes/port -> 2048B = 128 beats / port,
// totaling 256 beats but the two ports run in parallel = 128 effective cycles.
// Estimated cycles: kWeightWordsPerKtilePerPort + 6 (~134)
static void load_weight_kt_dual(
        const axi_word_t *w0,
        const axi_word_t *w1,
        uint64_t w0_kt_base_words,
        uint64_t w1_kt_base_words,
        WeightBuf &dst) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=dst.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=dst.words complete dim=2
    // Per port: kWeightColsPerPort lanes * (kTileK/kAxiBytes) words = 16*8 = 128 words
    // We flatten into a single II=1 loop and read both ports per cycle.
    const uint32_t words = kWeightWordsPerKtilePerPort; // 128
    for (uint32_t i = 0; i < words; ++i) {
        #pragma HLS PIPELINE II=1
        axi_word_t v0 = w0[w0_kt_base_words + i];
        axi_word_t v1 = w1[w1_kt_base_words + i];
        // i = lane * (kTileK/kAxiBytes) + k_word
        // Inverse: lane = i / (kTileK/kAxiBytes), k_word = i % (kTileK/kAxiBytes)
        const uint32_t k_words_per_lane = kTileK / kAxiBytes;  // 8
        const uint32_t lane   = i / k_words_per_lane;
        const uint32_t k_word = i % k_words_per_lane;
        dst.words[0][lane][k_word] = v0;
        dst.words[1][lane][k_word] = v1;
    }
}

// Emit kTileK stream beats from a filled WeightBuf. Each beat = 32 cols.
// Estimated cycles: kTileK + 4
static void push_weight_kt_to_stream(
        const WeightBuf &src,
        w_stream_t &w_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=src.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=src.words complete dim=2
    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        const uint32_t k_word = k / kAxiBytes;
        const uint32_t b_idx  = k % kAxiBytes;
        stream_word_t beat = 0;
        for (uint32_t lane = 0; lane < kWeightColsPerPort; ++lane) {
            #pragma HLS UNROLL
            axi_word_t w0 = src.words[0][lane][k_word];
            axi_word_t w1 = src.words[1][lane][k_word];
            beat(lane * 8 + 7, lane * 8) = w0(b_idx * 8 + 7, b_idx * 8);
            uint32_t col_hi = lane + kWeightColsPerPort;
            beat(col_hi * 8 + 7, col_hi * 8) = w1(b_idx * 8 + 7, b_idx * 8);
        }
        w_stream.write(beat);
    }
}

} // namespace v4
