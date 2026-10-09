// scale_drain.hpp --- consume per-128 PSUM stream, apply Q22 POT scales,
// accumulate across K-tiles, and write final signed 40-bit results to DDR.
//
// PSUM ordering from PE (one drain per K-tile):
//   row-major: 8 rows x 16 half-row beats. Each beat carries 8 columns of
//   int32 partial sums.
//
// Per N-tile flow:
//   For each kt:
//     Pull kPsumBeatsPerDrain beats (=128) and fold them into out_acc.
//     Apply per-row activation POT exponent and per-column weight POT
//     exponent for the current 128-wide K tile.
#pragma once

#include "streams.hpp"
#include "scale_loader.hpp"

namespace v6 {

static constexpr int kOutFrac = 22;
static constexpr int kOutAccBits = 40;
static constexpr int kOutBytesI40 = 8; // FFN output: sign-extended int64 slots
static constexpr int kOutBytesI32 = 4; // v6 dense/QKV/O output: signed i32 Q22 slots
static constexpr int kOutBytes = kOutBytesI40;
static constexpr int8_t kScaleExpDisabled = -128;
typedef ap_int<kOutAccBits> out_acc_t;
typedef ap_int<8> final_shift_t;
typedef ap_uint<64> final_shift_word_t;

struct OutAcc {
    out_acc_t v[kSaM][kSaN];
};

struct OutAccBatch {
    OutAcc slab[kBatchMtiles];
};

// FFN stream-down stores partial output accumulators across intermediate
// K-tiles. Row banking maps exactly to 32 URAMs: each bank has
// BM4 * 30 output tiles * 32 columns = 3840 entries.
struct DownCtxPool {
    out_acc_t v[kSaM][kBatchMtiles * kMaxFfnOutNtiles * kSaN];
};

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

static inline out_acc_t saturate_i96_to_out(ap_int<96> v) {
    #pragma HLS INLINE
    const ap_int<96> max_v = (((ap_int<96>)1) << (kOutAccBits - 1)) - 1;
    const ap_int<96> min_v = -(((ap_int<96>)1) << (kOutAccBits - 1));
    if (v > max_v) {
        return (out_acc_t)max_v;
    }
    if (v < min_v) {
        return (out_acc_t)min_v;
    }
    return (out_acc_t)v;
}

static inline out_acc_t round_shift_right_to_out(ap_int<96> value, uint32_t shift) {
    #pragma HLS INLINE
    if (shift == 0) {
        return saturate_i96_to_out(value);
    }
    const ap_int<96> half = ((ap_int<96>)1) << (shift - 1);
    if (value >= 0) {
        return saturate_i96_to_out((value + half) >> shift);
    }
    return saturate_i96_to_out(-(((-value) + half) >> shift));
}

static inline out_acc_t round_shift_right_i32_to_out(int32_t value, uint32_t shift) {
    #pragma HLS INLINE
    if (shift == 0) {
        return (out_acc_t)value;
    }
    if (shift > 32) {
        return 0;
    }

    const ap_int<34> v = (ap_int<34>)value;
    const ap_int<34> half = ((ap_int<34>)1) << (shift - 1);
    const ap_int<34> mag = (v >= 0) ? v : (ap_int<34>)-v;
    const ap_int<34> rounded = (mag + half) >> shift;
    return (v >= 0) ? (out_acc_t)rounded : (out_acc_t)-rounded;
}

static inline out_acc_t apply_final_shift_q22(int32_t x, final_shift_t final_shift);

static inline out_acc_t apply_final_shift_q22(int32_t x, final_shift_t final_shift) {
    #pragma HLS INLINE
    if (final_shift == (final_shift_t)kScaleExpDisabled) {
        return 0;
    }
    const int shift = (int)final_shift;
    if (shift >= 0) {
        ap_int<96> wide = x;
        wide <<= shift;
        return saturate_i96_to_out(wide);
    }

    const int rshift = -shift;
    return round_shift_right_i32_to_out(x, (uint32_t)rshift);
}

static inline out_acc_t add_saturate_out(out_acc_t a, out_acc_t b) {
    #pragma HLS INLINE
    const out_acc_t sum = (out_acc_t)(a + b);
    const bool overflow = (a[kOutAccBits - 1] == b[kOutAccBits - 1])
                       && (sum[kOutAccBits - 1] != a[kOutAccBits - 1]);
    if (!overflow) {
        return sum;
    }
    const out_acc_t max_v = (out_acc_t)((((ap_int<kOutAccBits + 1>)1)
                                         << (kOutAccBits - 1)) - 1);
    const out_acc_t min_v = (out_acc_t)(-(((ap_int<kOutAccBits + 1>)1)
                                           << (kOutAccBits - 1)));
    return a[kOutAccBits - 1] ? min_v : max_v;
}

template <uint32_t BASE_COL>
static inline scale_group_word_t pack_scale_group_half(const ScaleBuf &w_scale, uint32_t group) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 complete dim=2
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

template <uint32_t LANE>
static inline final_shift_t scale_half_final_shift(final_shift_word_t word) {
    #pragma HLS INLINE
    return (final_shift_t)(uint8_t)word(LANE * 8 + 7, LANE * 8);
}

template <uint32_t LANE>
static inline void set_final_shift_lane(
        final_shift_word_t &out,
        scale_group_word_t scale_word,
        int act_exp) {
    #pragma HLS INLINE
    const int8_t w_exp = scale_half_e0<LANE>(scale_word);
    const final_shift_t shift =
        (w_exp == kScaleExpDisabled) ? (final_shift_t)kScaleExpDisabled
                                     : (final_shift_t)(act_exp + (int)w_exp + kOutFrac);
    out(LANE * 8 + 7, LANE * 8) = (ap_uint<8>)(uint8_t)shift;
}

static inline final_shift_word_t pack_final_shift_half(
        scale_group_word_t scale_word,
        int act_exp) {
    #pragma HLS INLINE
    final_shift_word_t out = 0;
    set_final_shift_lane<0>(out, scale_word, act_exp);
    set_final_shift_lane<1>(out, scale_word, act_exp);
    set_final_shift_lane<2>(out, scale_word, act_exp);
    set_final_shift_lane<3>(out, scale_word, act_exp);
    set_final_shift_lane<4>(out, scale_word, act_exp);
    set_final_shift_lane<5>(out, scale_word, act_exp);
    set_final_shift_lane<6>(out, scale_word, act_exp);
    set_final_shift_lane<7>(out, scale_word, act_exp);
    return out;
}

static void push_scale_groups(const ScaleBuf &w_scale, scale_stream_t &scale_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w_scale.e0 complete dim=2
    for (uint32_t group = 0; group < kGroupsPerKtile; ++group) {
        for (uint32_t base = 0; base < kSaN; base += kScaleDrainColLanes) {
            #pragma HLS PIPELINE II=1
            scale_group_word_t word = 0;
            for (uint32_t lane = 0; lane < kScaleDrainColLanes; ++lane) {
                #pragma HLS UNROLL
                const uint32_t col = base + lane;
                word(lane * 8 + 7, lane * 8) =
                    (ap_uint<8>)(uint8_t)w_scale.e0[group][col];
            }
            scale_stream.write(word);
        }
    }
}

template <uint32_t LANE>
static inline void absorb_stream_col(
        psum_half_word_t beat,
        final_shift_word_t final_shift_word,
        OutAcc &out_acc,
        uint32_t row,
        uint32_t base_col) {
    #pragma HLS INLINE
    const uint32_t col = base_col + LANE;
    ap_uint<32> v32 = beat(LANE * 32 + 31, LANE * 32);
    const out_acc_t scaled = apply_final_shift_q22((int32_t)(uint32_t)v32,
                                                   scale_half_final_shift<LANE>(final_shift_word));
    out_acc.v[row][col] = add_saturate_out(out_acc.v[row][col], scaled);
}

static inline void absorb_stream_half(
        psum_half_word_t beat,
        final_shift_word_t final_shift_word,
        OutAcc &out_acc,
        uint32_t row,
        uint32_t base_col) {
    #pragma HLS INLINE
    absorb_stream_col<0>(beat, final_shift_word, out_acc, row, base_col);
    absorb_stream_col<1>(beat, final_shift_word, out_acc, row, base_col);
    absorb_stream_col<2>(beat, final_shift_word, out_acc, row, base_col);
    absorb_stream_col<3>(beat, final_shift_word, out_acc, row, base_col);
    absorb_stream_col<4>(beat, final_shift_word, out_acc, row, base_col);
    absorb_stream_col<5>(beat, final_shift_word, out_acc, row, base_col);
    absorb_stream_col<6>(beat, final_shift_word, out_acc, row, base_col);
    absorb_stream_col<7>(beat, final_shift_word, out_acc, row, base_col);
}

// Streaming scale metadata variant used by the pipelined HLS datapath.
// The producer emits one scale packet per psum beat segment, in the same group
// order as the PE psum drain. Keeping packets at 8 columns avoids wide dynamic
// part-select muxes in the hot absorb loop.
static void absorb_psum_drain_stream(
        psum_stream_t &psum,
        scale_stream_t &scale_stream,
        const int8_t   act_row_exp[kSaM][kGroupsPerKtile],
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
            for (uint32_t half = 0; half < kPsumBeatsPerRow; ++half) {
                #pragma HLS PIPELINE II=1
                psum_word_t beat = psum.read();
                psum_half_word_t psum_half = (psum_half_word_t)beat;
                const final_shift_word_t final_shift =
                    pack_final_shift_half(scale_seg[half], act_exp);
                absorb_stream_half(psum_half, final_shift, out_acc, row,
                                   half * kScaleDrainColLanes);
            }
        }
    }
}

static void init_out_acc_batch_bias(
        uint32_t slabs_valid,
        OutAccBatch &out_batch,
        const out_acc_t bias_tile[kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=out_batch.slab[0].v cyclic factor=kScaleDrainColLanes dim=2
    for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kBatchMtiles
        for (uint32_t row = 0; row < kSaM; ++row) {
            for (uint32_t base_col = 0; base_col < kSaN;
                 base_col += kScaleDrainColLanes) {
                #pragma HLS PIPELINE II=1
                for (uint32_t lane = 0; lane < kScaleDrainColLanes; ++lane) {
                    #pragma HLS UNROLL
                    const uint32_t col = base_col + lane;
                    out_batch.slab[slab].v[row][col] = bias_tile[col];
                }
            }
        }
    }
}

static void absorb_all_ktiles_batch(
        psum_stream_t &psum_stream,
        scale_stream_t &scale_stream,
        uint32_t k_tiles,
        uint32_t slabs_valid,
        const int8_t act_row_exp_kt[kBatchMtiles][kMaxKtiles][kSaM][kGroupsPerKtile],
        OutAccBatch &out_batch,
        const out_acc_t bias_tile[kSaN]) {
    #pragma HLS INLINE off
    if (k_tiles == 0) {
        return;
    }
    // Keep bias initialization out of the psum absorb loop.  Folding
    // bias + scaled_psum into the first K-tile absorb created the routed
    // worst path from bias BRAM through a wide saturating add into OutAcc.
    // This short init loop costs a small fixed latency but leaves the hot
    // absorb loop as OutAcc += scaled_psum, which is much friendlier at
    // 300 MHz.
    init_out_acc_batch_bias(slabs_valid, out_batch, bias_tile);
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kMaxKtiles
        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=kBatchMtiles
            absorb_psum_drain_stream(psum_stream, scale_stream,
                                     act_row_exp_kt[slab][kt],
                                     out_batch.slab[slab]);
        }
    }
}

template <bool FIRST_KT>
static void merge_out_batch_into_down_ctx_impl(
        const OutAccBatch &batch,
        uint32_t slabs_valid,
        uint32_t nt,
        DownCtxPool &ctx) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=ctx.v cyclic factor=kScaleDrainColLanes dim=2
    for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
        const uint32_t ctx_col_base = (slab * kMaxFfnOutNtiles + nt) * kSaN;
        for (uint32_t row = 0; row < kSaM; ++row) {
            for (uint32_t base_col = 0; base_col < kSaN;
                 base_col += kScaleDrainColLanes) {
                #pragma HLS DEPENDENCE variable=ctx.v inter false
                #pragma HLS PIPELINE II=1
                for (uint32_t lane = 0; lane < kScaleDrainColLanes; ++lane) {
                    #pragma HLS UNROLL
                    const uint32_t col = base_col + lane;
                    const uint32_t ctx_col = ctx_col_base + col;
                    const out_acc_t value = batch.slab[slab].v[row][col];
                    if (FIRST_KT) {
                        ctx.v[row][ctx_col] = value;
                    } else {
                        ctx.v[row][ctx_col] = add_saturate_out(ctx.v[row][ctx_col], value);
                    }
                }
            }
        }
    }
}

static void merge_out_batch_into_down_ctx_first(
        const OutAccBatch &batch,
        uint32_t slabs_valid,
        uint32_t nt,
        DownCtxPool &ctx) {
    #pragma HLS INLINE off
    merge_out_batch_into_down_ctx_impl<true>(batch, slabs_valid, nt, ctx);
}

static void merge_out_batch_into_down_ctx_accum(
        const OutAccBatch &batch,
        uint32_t slabs_valid,
        uint32_t nt,
        DownCtxPool &ctx) {
    #pragma HLS INLINE off
    merge_out_batch_into_down_ctx_impl<false>(batch, slabs_valid, nt, ctx);
}

// Write OutAcc to DDR as sign-extended int64 row-major. rows_valid <= kSaM allows ragged
// edge tiles. Invalid tail rows must not be written: Q/K/V outputs are packed
// back-to-back, so zeroing rows beyond M would overwrite the next output.
// Estimated cycles max: kSaM * ceil(kSaN * 8 / kAxiBytes) = 8 * 64 = 512.
static void store_out_acc(
        axi_word_t *out_arena,
        uint64_t out_byte_offset,
        uint32_t row_stride_bytes,
        uint32_t rows_valid,
        uint32_t cols_valid,
        uint32_t n_tile_col_offset, // bytes within a row
        const OutAcc &acc) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=acc.v cyclic factor=kScaleDrainColLanes dim=2
    const uint32_t out_word_base = out_byte_offset / kAxiBytes;
    const uint32_t row_stride_w  = row_stride_bytes / kAxiBytes;
    const uint32_t col_word_off  = n_tile_col_offset / kAxiBytes;
    const uint32_t words_per_row =
        (cols_valid * kOutBytes + kAxiBytes - 1u) / kAxiBytes;
    uint32_t row_word_base = out_word_base + col_word_off;

    for (uint32_t r = 0; r < rows_valid; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kSaM
        uint32_t dst_word = row_word_base;
        for (uint32_t w = 0; w < words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            axi_word_t word = 0;
            for (uint32_t b = 0; b < 2; ++b) {  // 2 int64 per 16B word
                #pragma HLS UNROLL
                uint32_t c = w * 2 + b;
                if (c < cols_valid) {
                    ap_int<64> v = (ap_int<64>)acc.v[r][c];
                    word(b * 64 + 63, b * 64) = (ap_uint<64>)v;
                }
            }
            out_arena[dst_word] = word;
            dst_word++;
        }
        row_word_base += row_stride_w;
    }
}

static void store_out_acc_i32(
        axi_word_t *out_arena,
        uint64_t out_byte_offset,
        uint32_t row_stride_bytes,
        uint32_t rows_valid,
        uint32_t cols_valid,
        uint32_t n_tile_col_offset,
        const OutAcc &acc) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=acc.v cyclic factor=kScaleDrainColLanes dim=2
    const uint32_t out_word_base = out_byte_offset / kAxiBytes;
    const uint32_t row_stride_w  = row_stride_bytes / kAxiBytes;
    const uint32_t col_word_off  = n_tile_col_offset / kAxiBytes;
    const uint32_t words_per_row =
        (cols_valid * kOutBytesI32 + kAxiBytes - 1u) / kAxiBytes;
    uint32_t row_word_base = out_word_base + col_word_off;

    for (uint32_t r = 0; r < rows_valid; ++r) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kSaM
        uint32_t dst_word = row_word_base;
        for (uint32_t w = 0; w < words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            axi_word_t word = 0;
            for (uint32_t b = 0; b < 4; ++b) {
                #pragma HLS UNROLL
                uint32_t c = w * 4 + b;
                if (c < cols_valid) {
                    const int32_t v = saturate_i64_to_i32((ap_int<64>)acc.v[r][c]);
                    word(b * 32 + 31, b * 32) = (ap_uint<32>)(uint32_t)v;
                }
            }
            out_arena[dst_word] = word;
            dst_word++;
        }
        row_word_base += row_stride_w;
    }
}

static void store_out_acc_batch(
        axi_word_t *out_arena,
        uint64_t out_byte_offset,
        uint32_t row_stride_bytes,
        uint32_t slabs_valid,
        const uint32_t rows_valid[kBatchMtiles],
        uint32_t cols_valid,
        uint32_t n_tile_col_offset,
        const OutAccBatch &batch) {
    #pragma HLS INLINE off
    for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
        store_out_acc(out_arena,
                      out_byte_offset + (uint64_t)slab * kSaM * row_stride_bytes,
                      row_stride_bytes, rows_valid[slab], cols_valid,
                      n_tile_col_offset, batch.slab[slab]);
    }
}

static void store_out_acc_batch_i32(
        axi_word_t *out_arena,
        uint64_t out_byte_offset,
        uint32_t row_stride_bytes,
        uint32_t slabs_valid,
        const uint32_t rows_valid[kBatchMtiles],
        uint32_t cols_valid,
        uint32_t n_tile_col_offset,
        const OutAccBatch &batch) {
    #pragma HLS INLINE off
    for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
        store_out_acc_i32(out_arena,
                          out_byte_offset + (uint64_t)slab * kSaM * row_stride_bytes,
                          row_stride_bytes, rows_valid[slab], cols_valid,
                          n_tile_col_offset, batch.slab[slab]);
    }
}

static void store_down_ctx_pool(
        axi_word_t *out_arena,
        uint64_t out_byte_offset,
        uint32_t row_stride_bytes,
        uint32_t slabs_valid,
        const uint32_t rows_valid[kBatchMtiles],
        uint32_t output_cols,
        uint32_t n_tiles,
        const DownCtxPool &ctx) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=ctx.v cyclic factor=kScaleDrainColLanes dim=2
    for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
        for (uint32_t nt = 0; nt < n_tiles; ++nt) {
            const uint32_t nt_col0 = nt * kSaN;
            const uint32_t cols_valid =
                (nt_col0 + kSaN <= output_cols) ? kSaN : (output_cols - nt_col0);
            const uint32_t words_per_row =
                (cols_valid * kOutBytes + kAxiBytes - 1u) / kAxiBytes;
            const uint32_t ctx_col_base = (slab * kMaxFfnOutNtiles + nt) * kSaN;
            const uint32_t out_word_base =
                (out_byte_offset + (uint64_t)slab * kSaM * row_stride_bytes) / kAxiBytes;
            const uint32_t row_stride_w = row_stride_bytes / kAxiBytes;
            const uint32_t col_word_off = nt * kSaN * kOutBytes / kAxiBytes;
            uint32_t row_word_base = out_word_base + col_word_off;
            for (uint32_t r = 0; r < rows_valid[slab]; ++r) {
                uint32_t dst_word = row_word_base;
                for (uint32_t w = 0; w < words_per_row; ++w) {
                    #pragma HLS PIPELINE II=1
                    axi_word_t word = 0;
                    for (uint32_t b = 0; b < 2; ++b) {
                        #pragma HLS UNROLL
                        const uint32_t c = w * 2 + b;
                        if (c < cols_valid) {
                            word(b * 64 + 63, b * 64) =
                                (ap_uint<64>)(ap_int<64>)ctx.v[r][ctx_col_base + c];
                        }
                    }
                    out_arena[dst_word++] = word;
                }
                row_word_base += row_stride_w;
            }
        }
    }
}

} // namespace v6
