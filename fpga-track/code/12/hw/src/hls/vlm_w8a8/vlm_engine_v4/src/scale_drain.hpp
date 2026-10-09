// scale_drain.hpp --- consume per-32 PSUM stream, apply Q20 POT scales,
// accumulate across K-tiles, and write final int32 results to DDR.
//
// PSUM ordering from PE (one drain per K-tile):
//   group-major: 4 scale groups x 32 rows x 4 quarter-row beats. Each beat
//   carries 8 columns of int32 partial sums.
//
// Per N-tile flow:
//   For each kt:
//     Pull kPsumBeatsPerDrain beats (=512) and fold them into out_acc.
//     Apply per-row activation POT exponent and per-column weight POT
//     exponent for the current 32-wide K group.
#pragma once

#include "streams.hpp"
#include "scale_loader.hpp"

namespace v4 {

struct OutAcc {
    int32_t v[kSaM][kSaN];
};

static constexpr int kOutFrac = 20;
static constexpr int8_t kScaleExpDisabled = -128;

static inline int32_t saturate_i64_to_i32(ap_int<64> v) {
    #pragma HLS INLINE
    if (v > 2147483647) {
        return 2147483647;
    }
    if (v < -2147483648LL) {
        return (int32_t)0x80000000u;
    }
    return (int32_t)v;
}

static inline int32_t saturate_i48_to_i32(ap_int<48> v) {
    #pragma HLS INLINE
    if (v > 2147483647) {
        return 2147483647;
    }
    if (v < -2147483648LL) {
        return (int32_t)0x80000000u;
    }
    return (int32_t)v;
}

static inline int32_t saturate_i33_to_i32(ap_int<33> v) {
    #pragma HLS INLINE
    if (v > 2147483647) {
        return 2147483647;
    }
    if (v < -2147483648LL) {
        return (int32_t)0x80000000u;
    }
    return (int32_t)v;
}

static inline int32_t apply_fixed_shift_q20(int32_t x, int scale_exp) {
    #pragma HLS INLINE
    const int shift = scale_exp + kOutFrac;
    if (shift >= 0) {
        if (shift >= 31) {
            if (x > 0) {
                return 2147483647;
            }
            if (x < 0) {
                return (int32_t)0x80000000u;
            }
            return 0;
        }
        const ap_int<33> x33 = x;
        const ap_int<33> pos_limit = ((ap_int<33>)2147483647) >> shift;
        const ap_int<33> neg_limit = ((ap_int<33>)-2147483648LL) >> shift;
        if (x33 > pos_limit) {
            return 2147483647;
        }
        if (x33 < neg_limit) {
            return (int32_t)0x80000000u;
        }
        ap_int<48> wide = (ap_int<48>)x;
        wide <<= shift;
        return (int32_t)wide;
    }

    const int rshift = -shift;
    if (rshift >= 63) {
        return 0;
    }
    if (rshift >= 47) {
        return (x < 0) ? -1 : 0;
    }
    ap_int<48> wide = x;
    if (rshift > 0) {
        const ap_int<48> half = (ap_int<48>)1 << (rshift - 1);
        wide = (wide >= 0) ? (wide + half) : (wide - half);
        wide >>= rshift;
    }
    return saturate_i48_to_i32(wide);
}

static inline int32_t apply_pot_q20(int32_t dot, int act_exp, int8_t w_e0) {
    #pragma HLS INLINE
    if (w_e0 != kScaleExpDisabled) {
        return apply_fixed_shift_q20(dot, act_exp + (int)w_e0);
    }
    return 0;
}

static inline int32_t add_saturate_i32(int32_t a, int32_t b) {
    #pragma HLS INLINE
    return saturate_i33_to_i32((ap_int<33>)a + (ap_int<33>)b);
}

template <uint32_t BASE_COL>
static inline scale_group_word_t pack_scale_group_half(const ScaleBuf &w_scale, uint32_t group) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 cyclic factor=kScaleDrainColLanes dim=2
    scale_group_word_t word = 0;
    for (uint32_t lane = 0; lane < kScaleDrainColLanes; ++lane) {
        #pragma HLS UNROLL
        const uint32_t col = BASE_COL + lane;
        word(lane * 8 + 7, lane * 8) =
            (ap_uint<8>)(uint8_t)w_scale.e0[group][col];
    }
    return word;
}

template <uint32_t LANE>
static inline int8_t scale_half_e0(scale_group_word_t word) {
    #pragma HLS INLINE
    return (int8_t)(uint8_t)word(LANE * 8 + 7, LANE * 8);
}

static void push_scale_groups(const ScaleBuf &w_scale, scale_stream_t &scale_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 cyclic factor=kScaleDrainColLanes dim=2
    for (uint32_t group = 0; group < kGroupsPerKtile; ++group) {
        #pragma HLS PIPELINE II=4
        scale_stream.write(pack_scale_group_half<0>(w_scale, group));
        scale_stream.write(pack_scale_group_half<kScaleDrainColLanes>(w_scale, group));
        scale_stream.write(pack_scale_group_half<2 * kScaleDrainColLanes>(w_scale, group));
        scale_stream.write(pack_scale_group_half<3 * kScaleDrainColLanes>(w_scale, group));
    }
}

template <uint32_t LANE>
static inline void absorb_stream_col(
        psum_half_word_t beat,
        scale_group_word_t scale_word,
        int act_exp,
        bool assign,
        OutAcc &out_acc,
        uint32_t row,
        uint32_t base_col) {
    #pragma HLS INLINE
    const uint32_t col = base_col + LANE;
    ap_uint<32> v32 = beat(LANE * 32 + 31, LANE * 32);
    const int32_t scaled = apply_pot_q20((int32_t)(uint32_t)v32,
                                         act_exp,
                                         scale_half_e0<LANE>(scale_word));
    out_acc.v[row][col] = assign ? scaled : add_saturate_i32(out_acc.v[row][col], scaled);
}

static inline void absorb_stream_half(
        psum_half_word_t beat,
        scale_group_word_t scale_word,
        int act_exp,
        bool assign,
        OutAcc &out_acc,
        uint32_t row,
        uint32_t base_col) {
    #pragma HLS INLINE
    absorb_stream_col< 0>(beat, scale_word, act_exp, assign, out_acc, row, base_col);
    absorb_stream_col< 1>(beat, scale_word, act_exp, assign, out_acc, row, base_col);
    absorb_stream_col< 2>(beat, scale_word, act_exp, assign, out_acc, row, base_col);
    absorb_stream_col< 3>(beat, scale_word, act_exp, assign, out_acc, row, base_col);
    absorb_stream_col< 4>(beat, scale_word, act_exp, assign, out_acc, row, base_col);
    absorb_stream_col< 5>(beat, scale_word, act_exp, assign, out_acc, row, base_col);
    absorb_stream_col< 6>(beat, scale_word, act_exp, assign, out_acc, row, base_col);
    absorb_stream_col< 7>(beat, scale_word, act_exp, assign, out_acc, row, base_col);
}

// Initialize output accumulator to zero. Estimated cycles: 32 (II=1, partition kSaN)
static void out_acc_clear(OutAcc &acc) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=acc.v cyclic factor=kScaleDrainColLanes dim=2
    for (uint32_t r = 0; r < kSaM; ++r) {
        for (uint32_t c = 0; c < kSaN; ++c) {
            #pragma HLS PIPELINE II=1
            acc.v[r][c] = 0;
        }
    }
}

// Pull one drain and fold per-32 POT-scaled partial sums into out_acc.
static void absorb_psum_drain(
        psum_stream_t &psum,
        const ScaleBuf &w_scale,           // per-group, 4x32
        const int8_t   act_row_exp[kSaM][kGroupsPerKtile],  // 4 groups for this kt
        bool first_kt,
        OutAcc &out_acc) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 cyclic factor=kScaleDrainColLanes dim=2
    #pragma HLS ARRAY_PARTITION variable=out_acc.v cyclic factor=kScaleDrainColLanes dim=2

    // Pull: 4 groups x 32 rows x 4 quarter-row beats.
    for (uint32_t i = 0; i < kPsumBeatsPerDrain; ++i) {
        #pragma HLS PIPELINE II=1
        psum_word_t beat = psum.read();
        const uint32_t group = i / kPsumBeatsPerGroup;
        const uint32_t in_group = i % kPsumBeatsPerGroup;
        const uint32_t row   = in_group / kPsumBeatsPerRow;
        const uint32_t half  = in_group % kPsumBeatsPerRow;
        const int act_exp = (int)act_row_exp[row][group];
        const uint32_t base_col = half * kScaleDrainColLanes;
        for (uint32_t lane = 0; lane < kScaleDrainColLanes; ++lane) {
            #pragma HLS UNROLL
            const uint32_t col = base_col + lane;
            ap_uint<32> v32 = beat(lane * 32 + 31, lane * 32);
            const int32_t scaled = apply_pot_q20((int32_t)(uint32_t)v32,
                                                 act_exp,
                                                 w_scale.e0[group][col]);
            out_acc.v[row][col] = (first_kt && group == 0u) ?
                    scaled : add_saturate_i32(out_acc.v[row][col], scaled);
        }
    }
}

// Streaming scale metadata variant used by the pipelined HLS datapath.
// The producer emits one scale packet per psum beat segment, in the same group
// order as the PE psum drain. Keeping packets at 8 columns avoids wide dynamic
// part-select muxes in the hot absorb loop.
static void absorb_psum_drain_stream(
        psum_stream_t &psum,
        scale_stream_t &scale_stream,
        const int8_t   act_row_exp[kSaM][kGroupsPerKtile],
        bool first_kt,
        OutAcc &out_acc) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=out_acc.v cyclic factor=kScaleDrainColLanes dim=2

    for (uint32_t group = 0; group < kGroupsPerKtile; ++group) {
        scale_group_word_t scale_seg[kPsumBeatsPerRow];
        #pragma HLS ARRAY_PARTITION variable=scale_seg complete dim=1
        for (uint32_t seg = 0; seg < kPsumBeatsPerRow; ++seg) {
            #pragma HLS UNROLL
            scale_seg[seg] = scale_stream.read();
        }
        for (uint32_t row = 0; row < kSaM; ++row) {
            const int act_exp = (int)act_row_exp[row][group];
            const bool assign = first_kt && group == 0u;
            for (uint32_t half = 0; half < kPsumBeatsPerRow; ++half) {
                #pragma HLS PIPELINE II=1
                psum_word_t beat = psum.read();
                psum_half_word_t psum_half = (psum_half_word_t)beat;
                scale_group_word_t scale_half = scale_seg[half];
                absorb_stream_half(psum_half, scale_half, act_exp, assign,
                                   out_acc, row, half * kScaleDrainColLanes);
            }
        }
    }
}

static void absorb_all_ktiles(
        psum_stream_t &psum_stream,
        scale_stream_t &scale_stream,
        uint32_t k_tiles,
        const int8_t act_row_exp_kt[/*k_tiles*/][kSaM][kGroupsPerKtile],
        OutAcc &out_acc) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=out_acc.v cyclic factor=kScaleDrainColLanes dim=2
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kMaxKtiles
        absorb_psum_drain_stream(psum_stream, scale_stream,
                                 act_row_exp_kt[kt], kt == 0, out_acc);
    }
}

// Write OutAcc to DDR as int32 row-major. rows_valid <= kSaM allows ragged
// edge tiles. Invalid tail rows must not be written: Q/K/V outputs are packed
// back-to-back, so zeroing rows beyond M would overwrite the next output.
// Estimated cycles max: kSaM * (kSaN * 4 / kAxiBytes) = 32 * 8 = 256
static void store_out_acc(
        axi_word_t *out_arena,
        uint64_t out_byte_offset,
        uint32_t row_stride_bytes,
        uint32_t rows_valid,
        uint32_t n_tile_col_offset, // bytes within a row
        const OutAcc &acc) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=acc.v cyclic factor=kScaleDrainColLanes dim=2
    const uint32_t out_word_base = out_byte_offset / kAxiBytes;
    const uint32_t row_stride_w  = row_stride_bytes / kAxiBytes;
    const uint32_t col_word_off  = n_tile_col_offset / kAxiBytes;
    const uint32_t words_per_row = (kSaN * 4) / kAxiBytes;  // 32*4/16 = 8

    for (uint32_t r = 0; r < rows_valid; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kSaM
        for (uint32_t w = 0; w < words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            axi_word_t word = 0;
            for (uint32_t b = 0; b < 4; ++b) {  // 4 int32 per 16B word
                #pragma HLS UNROLL
                uint32_t c = w * 4 + b;
                int32_t v = acc.v[r][c];
                word(b * 32 + 31, b * 32) = (ap_uint<32>)(uint32_t)v;
            }
            out_arena[out_word_base + r * row_stride_w + col_word_off + w] = word;
        }
    }
}

} // namespace v4
