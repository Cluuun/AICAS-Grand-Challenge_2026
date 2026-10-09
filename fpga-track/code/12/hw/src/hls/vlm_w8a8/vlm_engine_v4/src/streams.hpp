// streams.hpp --- AXIS stream type aliases between HLS feeders and PE array IP.
#pragma once

#include "types.hpp"

namespace v4 {

typedef hls::stream<stream_word_t> a_stream_t;     // 32 act bytes / cycle
typedef hls::stream<stream_word_t> w_stream_t;     // 32 weight bytes / cycle
typedef hls::stream<ap_uint<16>>   ctrl_stream_t;  // {tlast,drain,en,clear}
typedef hls::stream<psum_word_t>   psum_stream_t;  // 8 int32 / beat
typedef hls::stream<scale_group_word_t> scale_stream_t; // 16 quarter-group packets / K-tile

// Ctrl bit positions (must match pe_array_axis.v decode)
static constexpr unsigned kCtrlClearBit = 0;
static constexpr unsigned kCtrlEnBit    = 1;
static constexpr unsigned kCtrlDrainBit = 2;
static constexpr unsigned kCtrlLastBit  = 3;

static inline ap_uint<16> make_ctrl(bool clear, bool en, bool drain, bool last) {
    ap_uint<16> v = 0;
    v[kCtrlClearBit] = clear;
    v[kCtrlEnBit]    = en;
    v[kCtrlDrainBit] = drain;
    v[kCtrlLastBit]  = last;
    return v;
}

} // namespace v4
