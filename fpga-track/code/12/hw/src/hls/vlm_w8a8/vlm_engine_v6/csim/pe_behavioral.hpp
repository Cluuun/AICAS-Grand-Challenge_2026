// pe_behavioral.hpp --- C++ behavioral model of pe_array_v6 for HLS C-sim only.
//
// On hardware, vlm_engine_v6 emits a/w/ctrl AXIS beats to pe_array_v6 (Verilog
// IP) which returns m_axis_psum. C-sim has no Verilog IP, so this header
// supplies a synchronous behavioral PE that the engine drains once per K-tile
// via the V6_CSIM build path.
//
// Protocol exactly matches pe_array_axis.v (see the wrapper for the source of
// truth):
//   ctrl[0] = clear    : zero accumulators on this beat (combined with en)
//   ctrl[1] = en       : MAC step using current a/w beats
//   ctrl[2] = commit   : when en=1, final compute beat of a 32-K group
//   ctrl[3] = last     : final K-tile marker carried on the final group commit
//   psum beat          : kPsumLanes int32 lanes; group-major rows then segments.
//
// The behavioral PE keeps live accumulator state across calls, so per-kt clears
// carried on k0 are honored end-to-end.
#pragma once

#include "../src/types.hpp"
#include "../src/streams.hpp"

#include <cstdio>
#include <cstdlib>

namespace v6_csim {

struct PeState {
    int32_t acc[v6::kSaM][v6::kSaN];
    PeState() { reset(); }
    void reset() {
        for (uint32_t r = 0; r < v6::kSaM; ++r)
            for (uint32_t c = 0; c < v6::kSaN; ++c)
                acc[r][c] = 0;
    }
};

// Singleton state, reset at the start of each task by the csim wrapper.
inline PeState & pe_state() {
    static PeState s;
    return s;
}

static inline void protocol_fail(const char *phase, uint32_t idx, ap_uint<16> ctrl,
                                 const char *why) {
    std::fprintf(stderr,
                 "v6_csim protocol FAIL phase=%s idx=%u ctrl=0x%04x: %s\n",
                 phase, idx, (unsigned)ctrl, why);
    std::abort();
}

static inline void pump_ktiles(
        uint32_t k_tiles,
        v6::a_stream_t   &a_stream,
        v6::w_stream_t   &w0_stream,
        v6::w_stream_t   &w1_stream,
        v6::ctrl_stream_t &ctrl_stream,
        v6::psum_stream_t &psum_stream);

// Pump one or more K-tiles. The new hardware protocol emits psums
// autonomously after the commit beat, so C-sim consumes only compute ctrl beats.
static inline void pump_one_ktile(
        v6::a_stream_t   &a_stream,
        v6::w_stream_t   &w0_stream,
        v6::w_stream_t   &w1_stream,
        v6::ctrl_stream_t &ctrl_stream,
        v6::psum_stream_t &psum_stream) {
    pump_ktiles(1, a_stream, w0_stream, w1_stream, ctrl_stream, psum_stream);
}

static inline void pump_ktiles(
        uint32_t k_tiles,
        v6::a_stream_t   &a_stream,
        v6::w_stream_t   &w0_stream,
        v6::w_stream_t   &w1_stream,
        v6::ctrl_stream_t &ctrl_stream,
        v6::psum_stream_t &psum_stream) {
    PeState &st = pe_state();

    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        // ----- Compute/drain phase: each 32-K group commits and drains -----
        for (uint32_t group = 0; group < v6::kGroupsPerKtile; ++group) {
            for (uint32_t kg = 0; kg < v6::kGroupSize; ++kg) {
                const uint32_t k = group * v6::kGroupSize + kg;
                ap_uint<16> c = ctrl_stream.read();
                bool clear = c[v6::kCtrlClearBit];
                bool en    = c[v6::kCtrlEnBit];
                bool commit= c[v6::kCtrlDrainBit];
                bool last  = c[v6::kCtrlLastBit];
                const bool final_group_k = (kg == v6::kGroupSize - 1);
                const bool final_kt = (kt == k_tiles - 1);
                const bool final_tile_beat =
                    final_kt && group == v6::kGroupsPerKtile - 1 && final_group_k;
                if (!en) protocol_fail("compute", k, c, "compute beat without en");
                if (commit != final_group_k) protocol_fail("compute", k, c, "commit must be asserted only on final group k");
                if (clear != (kg == 0)) protocol_fail("compute", k, c, "clear must be asserted only on group k0");
                if (last != final_tile_beat) protocol_fail("compute", k, c, "last must be final K-tile commit only");
                v6::a_stream_word_t a_beat = a_stream.read();
                v6::w_half_stream_word_t w0_beat = w0_stream.read();
                v6::w_half_stream_word_t w1_beat = w1_stream.read();
                if (clear) st.reset();
                int8_t a_row[v6::kSaM];
                int8_t w_col[v6::kSaN];
                for (uint32_t r = 0; r < v6::kSaM; ++r) {
                    a_row[r] = (int8_t)(uint8_t)a_beat(r * 8 + 7, r * 8);
                }
                for (uint32_t cc = 0; cc < v6::kSaN; ++cc) {
                    if (cc < v6::kWeightColsPerPort) {
                        w_col[cc] = (int8_t)(uint8_t)w0_beat(cc * 8 + 7, cc * 8);
                    } else {
                        const uint32_t local_cc = cc - v6::kWeightColsPerPort;
                        w_col[cc] = (int8_t)(uint8_t)w1_beat(local_cc * 8 + 7, local_cc * 8);
                    }
                }
                for (uint32_t r = 0; r < v6::kSaM; ++r) {
                    for (uint32_t cc = 0; cc < v6::kSaN; ++cc) {
                        st.acc[r][cc] += (int32_t)a_row[r] * (int32_t)w_col[cc];
                    }
                }
            }

            // Autonomous drain: fixed-width row-segment psum beats for this group.
            for (uint32_t row = 0; row < v6::kSaM; ++row) {
                for (uint32_t half = 0; half < v6::kPsumBeatsPerRow; ++half) {
                    v6::psum_word_t beat = 0;
                    const uint32_t base_col = half * v6::kPsumLanes;
                    for (uint32_t l = 0; l < v6::kPsumLanes; ++l) {
                        int32_t v = st.acc[row][base_col + l];
                        beat(l * 32 + 31, l * 32) = (ap_uint<32>)(uint32_t)v;
                    }
                    psum_stream.write(beat);
                }
            }
        }
    }
}

static inline void pump_ktiles_batch(
        uint32_t k_tiles,
        uint32_t slabs_valid,
        v6::a_stream_t   &a_stream,
        v6::w_stream_t   &w0_stream,
        v6::w_stream_t   &w1_stream,
        v6::ctrl_stream_t &ctrl_stream,
        v6::psum_stream_t &psum_stream) {
    PeState &st = pe_state();
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            for (uint32_t group = 0; group < v6::kGroupsPerKtile; ++group) {
                for (uint32_t kg = 0; kg < v6::kGroupSize; ++kg) {
                    const uint32_t k = group * v6::kGroupSize + kg;
                    ap_uint<16> c = ctrl_stream.read();
                    const bool clear = c[v6::kCtrlClearBit];
                    const bool en = c[v6::kCtrlEnBit];
                    const bool commit = c[v6::kCtrlDrainBit];
                    const bool last = c[v6::kCtrlLastBit];
                    const bool final_group_k = (kg == v6::kGroupSize - 1);
                    const bool final_kt = (kt == k_tiles - 1);
                    if (!en) protocol_fail("batch-compute", k, c, "compute beat without en");
                    if (commit != final_group_k) protocol_fail("batch-compute", k, c, "commit must be asserted only on final group k");
                    if (clear != (kg == 0)) protocol_fail("batch-compute", k, c, "clear must be asserted only on group k0");
                    if (last != (final_kt && final_group_k)) protocol_fail("batch-compute", k, c, "last must mark each slab final K-tile");
                    v6::a_stream_word_t a_beat = a_stream.read();
                    v6::w_half_stream_word_t w0_beat = w0_stream.read();
                    v6::w_half_stream_word_t w1_beat = w1_stream.read();
                    if (clear) st.reset();
                    for (uint32_t r = 0; r < v6::kSaM; ++r) {
                        const int8_t av = (int8_t)(uint8_t)a_beat(r * 8 + 7, r * 8);
                        for (uint32_t cc = 0; cc < v6::kSaN; ++cc) {
                            const int8_t wv = (cc < v6::kWeightColsPerPort)
                                ? (int8_t)(uint8_t)w0_beat(cc * 8 + 7, cc * 8)
                                : (int8_t)(uint8_t)w1_beat((cc - v6::kWeightColsPerPort) * 8 + 7,
                                                           (cc - v6::kWeightColsPerPort) * 8);
                            st.acc[r][cc] += (int32_t)av * (int32_t)wv;
                        }
                    }
                }
                for (uint32_t row = 0; row < v6::kSaM; ++row) {
                    for (uint32_t half = 0; half < v6::kPsumBeatsPerRow; ++half) {
                        v6::psum_word_t beat = 0;
                        const uint32_t base_col = half * v6::kPsumLanes;
                        for (uint32_t lane = 0; lane < v6::kPsumLanes; ++lane) {
                            beat(lane * 32 + 31, lane * 32) =
                                (ap_uint<32>)(uint32_t)st.acc[row][base_col + lane];
                        }
                        psum_stream.write(beat);
                    }
                }
            }
        }
    }
}

} // namespace v6_csim

// Engine-side hook: in csim builds, pump the behavioral PE for one K-tile
// between drain ctrl emission and psum absorption. Synthesis builds expand to
// nothing because the real RTL PE consumes/produces the streams directly.
#ifdef V6_CSIM
  #define V6_CSIM_PUMP_PE(a_stream, w0_stream, w1_stream, ctrl_stream, psum_stream) \
      ::v6_csim::pump_one_ktile((a_stream), (w0_stream), (w1_stream), (ctrl_stream), (psum_stream))
#else
  #define V6_CSIM_PUMP_PE(a_stream, w0_stream, w1_stream, ctrl_stream, psum_stream) ((void)0)
#endif
