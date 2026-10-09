// pe_driver.hpp --- sequencer that drives a/w/ctrl streams to the PE IP.
//
// One N-tile compute timeline (k_tiles K-tiles, 1 group per K-tile):
//   For each kt:
//     For each group (4):
//       For 128 K cycles:               ctrl = en(=1), clear=(first k)
//         a_stream <- 1 beat (32 act bytes from cache for current K position)
//         w_stream <- 1 beat (32 weight bytes for current K position)
//         ctrl_stream <- ctrl word
//   The final beat of each 32-K group carries commit=1. The PE autonomously
//   drains that group and backpressures the next compute group until the drain
//   completes. There are no standalone drain ctrl beats.
//
// Cycles per N-tile:
//   compute: k_tiles * kTileK    (= 128 cycles per kt)
//   psum beats: kPsumBeatsPerDrain (= 128) per K-tile
//   total ctrl: k_tiles * kTileK
//
// (Per K-tile compute = kTileK = 128 cycles.)
#pragma once

#include "act_cache.hpp"
#include "weight_loader.hpp"

namespace v5 {

// Drive compute portion of one N-tile. Caller has filled WeightBuf for each
// K-tile and ActCache for the m_tile.
// Estimated cycles: k_tiles * kTileK (one beat / cycle)
static void drive_compute_ntile(
        const ActCache &act,
        const WeightBuf wbufs[],   // [k_tiles] filled buffers (consumed in order)
        uint32_t k_tiles,
        bool final_tile,
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off

    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        // K-tile beats: kTileK cycles, II=1
        for (uint32_t k = 0; k < kTileK; ++k) {
            #pragma HLS PIPELINE II=1
            // Activation beat: 32 act bytes for K position kt*kTileK + k
            const uint32_t k_idx         = kt * kTileK + k;
            stream_word_t a_beat = 0;
            for (uint32_t r = 0; r < kSaM; ++r) {
                #pragma HLS UNROLL
                const ap_uint<64> a_word = act.buf[r][k_idx / 8];
                const uint32_t a_bidx = k_idx & 7u;
                a_beat(r * 8 + 7, r * 8) = a_word(a_bidx * 8 + 7, a_bidx * 8);
            }
            a_stream.write(a_beat);

            stream_word_t w_beat = 0;
            w_beat(127, 0)   = wbufs[kt].words[0][k];
            w_beat(255, 128) = wbufs[kt].words[1][k];
            w_stream.write(w_beat);

            const bool group_start = ((k & (kGroupSize - 1u)) == 0u);
            const bool group_end   = ((k & (kGroupSize - 1u)) == (kGroupSize - 1u));
            const bool kt_end      = (kt == k_tiles - 1u) && (k == kTileK - 1u);
            ctrl_stream.write(make_ctrl(group_start, true, group_end,
                                        final_tile && kt_end));
        }
    }
}

} // namespace v5
