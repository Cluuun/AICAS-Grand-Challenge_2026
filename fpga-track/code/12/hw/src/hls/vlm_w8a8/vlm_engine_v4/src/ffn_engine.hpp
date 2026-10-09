// ffn_engine.hpp --- v4 FFN gate/up/down fused engine.
//
// Layer math (per token row r):
//   gate[r,n] = SiLU/GELU( sum_k act[r,k] * Wg[k,n] )    // intermediate cols
//   up  [r,n] = sum_k act[r,k] * Wu[k,n]                 // intermediate cols
//   mid [r,n] = gate[r,n] * up[r,n]                       // requantized to int8
//   out [r,m] = sum_n mid[r,n] * Wd[n,m]                  // output cols
//
// Per m_tile:
//   1) load_activation_block(K=hidden) once
//   2) for each intermediate-N-tile:
//        run gate sub-tile -> int32 gate_acc[kSaM][kSaN]
//        run up   sub-tile -> int32 up_acc  [kSaM][kSaN]
//        fuse + quantize  -> int8 mid_q[kSaM][kSaN], scale_exp[kSaM][group]
//        store into mid_cache[][...]
//   3) move mid_cache into act_cache (re-binding), set K=intermediate
//   4) for each output-N-tile of down:
//        run dense kernel with mid as activation
//
// We reuse run_ntile_accumulate() from dense_engine.hpp for gate/up/down. Each
// call re-clears the PE accumulator (clear=true on first kt/k of every nt).
#pragma once

#include "dense_engine.hpp"

namespace v4 {

// HardSiLU: x * clamp(x+3, 0, 6) / 6 in Q20 fixed-point.
static inline int32_t v4_hard_silu(int32_t x) {
    #pragma HLS INLINE
    int32_t xp = x + (3 << kOutFrac);
    int32_t six_q20 = 6 << kOutFrac;
    int32_t cl  = (xp < 0) ? 0 : ((xp > six_q20) ? six_q20 : xp);
    ap_int<64> p = static_cast<ap_int<64> >(x) * static_cast<ap_int<64> >(cl);
    ap_int<96> scaled = static_cast<ap_int<96> >(p) * static_cast<ap_int<32> >(2796203);
    return static_cast<int32_t>(scaled >> 44);
}

// HardGELU: x * clamp(x+1.5, 0, 3) / 3 in Q20.
static inline int32_t v4_hard_gelu(int32_t x) {
    #pragma HLS INLINE
    int32_t xp = x + (3 << (kOutFrac - 1));
    int32_t three_q20 = 3 << kOutFrac;
    int32_t cl = (xp < 0) ? 0 : ((xp > three_q20) ? three_q20 : xp);
    ap_int<64> p = static_cast<ap_int<64> >(x) * static_cast<ap_int<64> >(cl);
    ap_int<96> scaled = static_cast<ap_int<96> >(p) * static_cast<ap_int<32> >(5592405);
    return static_cast<int32_t>(scaled >> 44);
}

static inline int32_t round_shift_right_i32(int32_t x, uint32_t shift) {
    #pragma HLS INLINE
    if (shift == 0) {
        return x;
    }
    const int32_t half = (int32_t)1 << (shift - 1);
    return (x >= 0) ? ((x + half) >> shift) : ((x - half) >> shift);
}

static inline int32_t round_mul_q20(int32_t a, int32_t b) {
    #pragma HLS INLINE
    ap_int<64> p = static_cast<ap_int<64> >(a) * static_cast<ap_int<64> >(b);
    const ap_int<64> half = (ap_int<64>)1 << (kOutFrac - 1);
    p = (p >= 0) ? (p + half) : (p - half);
    return saturate_i64_to_i32(p >> kOutFrac);
}

static inline int8_t choose_pot_exp_for_q20(int32_t max_abs) {
    #pragma HLS INLINE
    if (max_abs <= 0) {
        return 0;
    }
    int selected_shift = kOutFrac;
    bool found = false;
    for (int shift = -10; shift <= 50; ++shift) {
        #pragma HLS LOOP_TRIPCOUNT min=61 max=61
        bool fits = false;
        if (shift >= 0) {
            ap_int<64> limit = (ap_int<64>)127 << shift;
            fits = ((ap_int<64>)max_abs <= limit);
        } else {
            ap_int<64> scaled = (ap_int<64>)max_abs << (-shift);
            fits = (scaled <= 127);
        }
        if (fits && !found) {
            selected_shift = shift;
            found = true;
        }
    }
    int exp = selected_shift - kOutFrac;
    if (exp < -30) exp = -30;
    if (exp > 30) exp = 30;
    return (int8_t)exp;
}

static inline int8_t quantize_q20_with_exp(int32_t x, int8_t exp) {
    #pragma HLS INLINE
    const int shift = (int)exp + kOutFrac;
    int32_t q = 0;
    if (shift >= 0) {
        q = (shift >= 31) ? 0 : round_shift_right_i32(x, (uint32_t)shift);
    } else {
        const int lshift = -shift;
        ap_int<64> wide = (ap_int<64>)x << lshift;
        q = saturate_i64_to_i32(wide);
    }
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

// Quantize one row of Q20 fused values to int8 with power-of-2 scale.
static void v4_quantize_row(
        const int32_t fused[kSaN],
        int8_t out_q[kSaN],
        scale_exp_t &out_exp) {
    #pragma HLS INLINE off
    int32_t max_abs = 0;
    for (uint32_t i = 0; i < kSaN; ++i) {
        #pragma HLS UNROLL factor=8
        int32_t a = (fused[i] < 0) ? -fused[i] : fused[i];
        if (a > max_abs) max_abs = a;
    }
    out_exp = choose_pot_exp_for_q20(max_abs);
    for (uint32_t i = 0; i < kSaN; ++i) {
        #pragma HLS UNROLL factor=8
        out_q[i] = quantize_q20_with_exp(fused[i], out_exp);
    }
}

// Re-quantize an OutAcc (int32 row-major) into the activation cache as int8
// at byte offset `nt_byte_off` along K. Group exponents (kSaN/kGroupSize=1 group)
// stored into mid_exp[r][g_global]. Estimated cycles: kSaM*(kSaN+10).
static void requantize_into_act_cache(
        const OutAcc &acc,
        uint32_t rows_valid,
        uint32_t nt_col_byte_off,        // bytes within a row (== nt * kSaN)
        ActCache &mid) {
    #pragma HLS INLINE off
    const uint32_t group_global = nt_col_byte_off / kGroupSize;
    for (uint32_t r = 0; r < kSaM; ++r) {
        int8_t row_q[kSaN];
        #pragma HLS ARRAY_PARTITION variable=row_q complete
        scale_exp_t row_exp = 0;
        v4_quantize_row(acc.v[r], row_q, row_exp);
        if (r < rows_valid) {
            mid.row_exp[r][group_global] = row_exp;
        }
        const uint32_t base_word = nt_col_byte_off / 8;
        for (uint32_t w = 0; w < kSaN / 8; ++w) {
            #pragma HLS PIPELINE II=1
            ap_uint<64> word = 0;
            for (uint32_t b = 0; b < 8; ++b) {
                #pragma HLS UNROLL
                uint32_t c = w * 8 + b;
                int8_t v  = (r < rows_valid) ? row_q[c] : (int8_t)0;
                word(b * 8 + 7, b * 8) = (uint8_t)v;
            }
            mid.buf[r][base_word + w] = word;
        }
    }
}

// Re-quantize four 32-column FFN intermediate tiles as one 128-wide K-tile.
// Each 32-column group keeps its own POT exponent, matching software v2 A8.
static void requantize_ktile_into_act_cache(
        const int32_t fused[kSaM][kTileK],
        uint32_t rows_valid,
        uint32_t kt_col_byte_off,
        uint32_t valid_cols,
        ActCache &mid) {
    #pragma HLS INLINE off
    const uint32_t group_base = kt_col_byte_off / kGroupSize;
    const uint32_t base_word = kt_col_byte_off / 8;

    for (uint32_t r = 0; r < kSaM; ++r) {
        scale_exp_t group_exp[kGroupsPerKtile];
        #pragma HLS ARRAY_PARTITION variable=group_exp complete
        for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
            int32_t max_abs = 0;
            for (uint32_t c = 0; c < kGroupSize; ++c) {
                #pragma HLS PIPELINE II=1
                const uint32_t col = g * kGroupSize + c;
                if (col < valid_cols) {
                    const int32_t v = fused[r][col];
                    const int32_t a = (v < 0) ? -v : v;
                    if (a > max_abs) {
                        max_abs = a;
                    }
                }
            }
            group_exp[g] = choose_pot_exp_for_q20(max_abs);
            if (r < rows_valid) {
                mid.row_exp[r][group_base + g] = group_exp[g];
            }
        }

        for (uint32_t w = 0; w < kTileK / 8; ++w) {
            #pragma HLS PIPELINE II=1
            ap_uint<64> word = 0;
            for (uint32_t b = 0; b < 8; ++b) {
                #pragma HLS UNROLL
                const uint32_t c = w * 8 + b;
                const uint32_t g = c / kGroupSize;
                int8_t q = 0;
                if (r < rows_valid && c < valid_cols) {
                    q = quantize_q20_with_exp(fused[r][c], group_exp[g]);
                }
                word(b * 8 + 7, b * 8) = static_cast<uint8_t>(q);
            }
            mid.buf[r][base_word + w] = word;
        }
    }
}

// Top-level FFN. Phases: (1) gate/up -> mid cache, (2) down -> DDR.
static void run_prefill_ffn(
        const axi_word_t *w0,
        const axi_word_t *w1,
        const axi_word_t *act_arena,
        axi_word_t *out_arena,
        const task_t &task,
        ActCache &act_cache,
        ActCache &mid_cache,
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream) {
    #pragma HLS INLINE

    const uint32_t M    = task.rows;
    const uint32_t K_in = task.ffn_input_cols;          // hidden
    const uint32_t K_im = task.ffn_intermediate_cols;   // intermediate
    const uint32_t N_out= task.ffn_output_cols;         // hidden
    const uint32_t kt_in     = div_ceil(K_in, kTileK);
    const uint32_t kt_im     = div_ceil(K_im, kTileK);
    const uint32_t k_groups_in = div_ceil(K_in, kGroupSize);
    const uint32_t k_groups_im = div_ceil(K_im, kGroupSize);
    const uint32_t m_tiles   = div_ceil(M, kSaM);
    const uint32_t nt_im     = div_ceil(K_im, kSaN);
    const uint32_t nt_out    = div_ceil(N_out, kSaN);
    const uint32_t k_words_in = kt_in * kActWordsPerKtile;
    const uint32_t down_weight_n_tile_stride_bytes =
        kWeightWordsPerKtilePerPort * kAxiBytes * kt_im;
    const uint32_t down_weight_scale_n_tile_stride_bytes =
        kScaleWordsPerKtilePerPort * kAxiBytes * kt_im;

    static int8_t exp_in[kMaxKtiles][kSaM][kGroupsPerKtile];
    static int8_t exp_im[kMaxKtiles][kSaM][kGroupsPerKtile];

    for (uint32_t mt = 0; mt < m_tiles; ++mt) {
        const uint32_t mt_row0    = mt * kSaM;
        const uint32_t rows_valid = (mt_row0 + kSaM <= M) ? kSaM : (M - mt_row0);

        // ---- Phase A: load input activation, run gate+up, build mid cache ----
        load_activation_block(act_arena,
                              task.act_q_offset_bytes
                                + (uint64_t)mt_row0 * task.act_row_stride,
                              task.act_scale_offset_bytes
                                + (uint64_t)mt_row0 * task.act_scale_row_stride,
                              task.act_row_stride,
                              task.act_scale_row_stride,
                              rows_valid, k_words_in, k_groups_in, act_cache);
        for (uint32_t kt = 0; kt < kt_in; ++kt) {
            for (uint32_t r = 0; r < kSaM; ++r) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    #pragma HLS UNROLL
                    uint32_t glob = kt * kGroupsPerKtile + g;
                    exp_in[kt][r][g] =
                        (glob < k_groups_in) ? static_cast<int8_t>(act_cache.row_exp[r][glob])
                                             : static_cast<int8_t>(0);
                }
            }
        }

        for (uint32_t nt_base = 0; nt_base < nt_im; nt_base += kGroupsPerKtile) {
            int32_t fused_kt[kSaM][kTileK];
            #pragma HLS BIND_STORAGE variable=fused_kt type=ram_2p impl=lutram
            #pragma HLS ARRAY_PARTITION variable=fused_kt cyclic factor=kFfnFusionColLanes dim=2

            for (uint32_t nt_g = 0; nt_g < kGroupsPerKtile; ++nt_g) {
                const uint32_t nt = nt_base + nt_g;
                const bool nt_valid = nt < nt_im;
                if (nt_valid) {
                    const uint64_t g_w0 = (task.gate_q_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t g_w1 = (task.gate_q_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t g_s0 = (task.gate_scale_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t g_s1 = (task.gate_scale_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_w0 = (task.up_q_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_w1 = (task.up_q_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_s0 = (task.up_scale_offset_bytes[0]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                    const uint64_t u_s1 = (task.up_scale_offset_bytes[1]
                                           + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;

                    OutAcc gate_acc, up_acc;
                    #pragma HLS ARRAY_PARTITION variable=gate_acc.v cyclic factor=kScaleDrainColLanes dim=2
                    #pragma HLS ARRAY_PARTITION variable=up_acc.v cyclic factor=kScaleDrainColLanes dim=2
                    run_ntile_accumulate(w0, w1, act_cache, g_w0, g_w1, g_s0, g_s1,
                                          kt_in, rows_valid, exp_in,
                                          a_stream, w_stream, ctrl_stream, psum_stream, gate_acc);
                    run_ntile_accumulate(w0, w1, act_cache, u_w0, u_w1, u_s0, u_s1,
                                          kt_in, rows_valid, exp_in,
                                          a_stream, w_stream, ctrl_stream, psum_stream, up_acc);

                    for (uint32_t r = 0; r < kSaM; ++r) {
                        for (uint32_t cb = 0; cb < kSaN; cb += kFfnFusionColLanes) {
                            #pragma HLS PIPELINE II=1
                            for (uint32_t lane = 0; lane < kFfnFusionColLanes; ++lane) {
                            #pragma HLS UNROLL
                            const uint32_t c = cb + lane;
                            const int32_t g = gate_acc.v[r][c];
                            const int32_t u = up_acc.v[r][c];
                            const int32_t act = (task.ffn_activation == LINEAR_FFN_ACT_GELU)
                                            ? v4_hard_gelu(g) : v4_hard_silu(g);
                            fused_kt[r][nt_g * kSaN + c] = round_mul_q20(act, u);
                            }
                        }
                    }
                } else {
                    for (uint32_t r = 0; r < kSaM; ++r) {
                        for (uint32_t cb = 0; cb < kSaN; cb += kFfnFusionColLanes) {
                            #pragma HLS PIPELINE II=1
                            for (uint32_t lane = 0; lane < kFfnFusionColLanes; ++lane) {
                            #pragma HLS UNROLL
                            const uint32_t c = cb + lane;
                            fused_kt[r][nt_g * kSaN + c] = 0;
                            }
                        }
                    }
                }
            }
            const uint32_t kt_col = nt_base * kSaN;
            const uint32_t valid_cols = (kt_col + kTileK <= K_im) ? kTileK : (K_im - kt_col);
            requantize_ktile_into_act_cache(fused_kt, rows_valid, kt_col, valid_cols, mid_cache);
        }

        // ---- Phase B: run down with mid_cache as activation ----
        // Build per-kt exponents for the intermediate cache.
        for (uint32_t kt = 0; kt < kt_im; ++kt) {
            for (uint32_t r = 0; r < kSaM; ++r) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    #pragma HLS UNROLL
                    uint32_t glob = kt * kGroupsPerKtile + g;
                    exp_im[kt][r][g] =
                        (glob < k_groups_im) ? static_cast<int8_t>(mid_cache.row_exp[r][glob])
                                             : static_cast<int8_t>(0);
                }
            }
        }

        for (uint32_t nt = 0; nt < nt_out; ++nt) {
            const uint64_t d_w0 = (task.down_q_offset_bytes[0]
                                   + (uint64_t)nt * down_weight_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_w1 = (task.down_q_offset_bytes[1]
                                   + (uint64_t)nt * down_weight_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_s0 = (task.down_scale_offset_bytes[0]
                                   + (uint64_t)nt * down_weight_scale_n_tile_stride_bytes) / kAxiBytes;
            const uint64_t d_s1 = (task.down_scale_offset_bytes[1]
                                   + (uint64_t)nt * down_weight_scale_n_tile_stride_bytes) / kAxiBytes;

            const uint64_t out_byte_off = task.ffn_dst_offset_bytes
                                        + (uint64_t)mt_row0 * task.output_row_stride_bytes[0];
            const uint32_t out_col_byte_off = nt * kSaN * sizeof(int32_t);

            OutAcc down_acc;
            #pragma HLS ARRAY_PARTITION variable=down_acc.v cyclic factor=kScaleDrainColLanes dim=2
            run_ntile_accumulate(w0, w1, mid_cache,
                                  d_w0, d_w1, d_s0, d_s1,
                                  kt_im, rows_valid, exp_im,
                                  a_stream, w_stream, ctrl_stream, psum_stream,
                                  down_acc);
            store_out_acc(out_arena, out_byte_off,
                          task.output_row_stride_bytes[0],
                          rows_valid, out_col_byte_off, down_acc);
        }
    }
}

} // namespace v4
