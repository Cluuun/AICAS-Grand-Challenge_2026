// dense_engine.hpp --- V5 prefill dense / QKV-fused engine.
//
// Outer loop hierarchy (mt outermost so activation cache is reused):
//   for mt in m_tiles:
//     load_activation_block(K)            // ~5130 cycles for K=2560 (paid once / mt)
//     for out_idx in output_count:
//       for nt in n_tiles[out_idx]:
//         for kt in k_tiles:
//           load_weight_kt_dual + scales            (~134 cycles, DDR-bound)
//           drive 128 cycles of (a/w/ctrl)          (commit once per 128-K tile)
//           PE autonomously drains 128 psum beats   (overlaps next K-tile)
//           absorb_psum_drain (apply scale, fold)
//         store_out_acc                             (~256 cycles)
//
// Stream interfaces are AXIS ports the BD wires through to the pe_array_v5 IP.
#pragma once

#include "axi_io.hpp"
#include "act_cache.hpp"
#include "weight_loader.hpp"
#include "scale_loader.hpp"
#include "pe_driver.hpp"
#include "scale_drain.hpp"
#include "bias_loader.hpp"
#ifdef V5_CSIM
#include "../csim/pe_behavioral.hpp"
#else
#define V5_CSIM_PUMP_PE(a_stream, w_stream, ctrl_stream, psum_stream) ((void)0)
#endif

namespace v5 {

static void load_weight_scale_tile(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        uint64_t w0_kt_base_words,
        uint64_t w1_kt_base_words,
        uint64_t w0_kt_scale_words,
        uint64_t w1_kt_scale_words,
        WeightBuf &wbuf,
        ScaleBuf &sbuf) {
    #pragma HLS INLINE off
    load_weight_kt_dual(w0_arena, w1_arena, w0_kt_base_words, w1_kt_base_words, wbuf);
    load_weight_scale_kt(w0_arena, w1_arena, w0_kt_scale_words, w1_kt_scale_words, sbuf);
}

static void drive_compute_tile(
        const ActCache &act,
        const WeightBuf &wbuf,
        const ScaleBuf &sbuf,
        uint32_t act_word_base,
        uint32_t kt,
        bool final_kt,
        scale_stream_t &scale_stream,
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 cyclic factor=kScaleDrainColLanes dim=2

    push_scale_groups(sbuf, scale_stream);

    for (uint32_t k = 0; k < kTileK; ++k) {
        #pragma HLS PIPELINE II=1
        const uint32_t k_idx  = kt * kTileK + k;
        stream_word_t a_beat = 0, w_beat = 0;
        for (uint32_t r = 0; r < kSaM; ++r) {
            #pragma HLS UNROLL
            const ap_uint<64> a_word = act.buf[r][act_word_base + k_idx / 8];
            const uint32_t a_bidx = k_idx & 7u;
            a_beat(r * 8 + 7, r * 8) = a_word(a_bidx * 8 + 7, a_bidx * 8);
        }
        w_beat(127, 0)   = wbuf.words[0][k];
        w_beat(255, 128) = wbuf.words[1][k];
        const bool group_start = ((k & (kGroupSize - 1u)) == 0u);
        const bool group_end   = ((k & (kGroupSize - 1u)) == (kGroupSize - 1u));
        const bool kt_end      = (k == kTileK - 1u);
        a_stream.write(a_beat);
        w_stream.write(w_beat);
        ctrl_stream.write(make_ctrl(/*clear=*/group_start, true,
                                    /*commit=*/group_end,
                                    /*last=*/final_kt && kt_end));
    }
}

static void weight_pp_load_stage(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        uint64_t w0_nt_base_words,
        uint64_t w1_nt_base_words,
        uint64_t w0_nt_scale_words,
        uint64_t w1_nt_scale_words,
        uint32_t k_tiles,
        WeightBuf &wbuf0,
        WeightBuf &wbuf1,
        ScaleBuf &sbuf0,
        ScaleBuf &sbuf1,
        hls::stream<ap_uint<1> > &loaded_bank_stream,
        hls::stream<ap_uint<1> > &free_bank_stream) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION function instances=load_weight_scale_tile limit=1
    #pragma HLS ARRAY_PARTITION variable=wbuf0.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wbuf1.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 cyclic factor=kScaleDrainColLanes dim=2
    #pragma HLS ARRAY_PARTITION variable=sbuf1.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf1.e0 cyclic factor=kScaleDrainColLanes dim=2

    const uint32_t kt_w_words   = kWeightWordsPerKtilePerPort;
    const uint32_t kt_w_scale_w = kScaleWordsPerKtilePerPort;

    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kMaxKtiles
        // A bank may be reused only after the compute stage has released it.
        if (kt >= 2u) {
            (void)free_bank_stream.read();
        }

        const uint64_t w0_kt_base_words  = w0_nt_base_words  + kt * kt_w_words;
        const uint64_t w1_kt_base_words  = w1_nt_base_words  + kt * kt_w_words;
        const uint64_t w0_kt_scale_words = w0_nt_scale_words + kt * kt_w_scale_w;
        const uint64_t w1_kt_scale_words = w1_nt_scale_words + kt * kt_w_scale_w;
        const ap_uint<1> bank = (ap_uint<1>)(kt & 1u);

        if (bank == 0u) {
            load_weight_scale_tile(w0_arena, w1_arena,
                                   w0_kt_base_words, w1_kt_base_words,
                                   w0_kt_scale_words, w1_kt_scale_words,
                                   wbuf0, sbuf0);
        } else {
            load_weight_scale_tile(w0_arena, w1_arena,
                                   w0_kt_base_words, w1_kt_base_words,
                                   w0_kt_scale_words, w1_kt_scale_words,
                                   wbuf1, sbuf1);
        }
        loaded_bank_stream.write(bank);
    }
}

#if 0
// Disabled single-slab reference path. The release datapath below always uses
// the fixed BM4 scheduler so each resident weight tile serves multiple slabs.
static void weight_pp_compute_stage(
        const ActCache &act,
        uint32_t act_word_base,
        uint32_t k_tiles,
        WeightBuf &wbuf0,
        WeightBuf &wbuf1,
        ScaleBuf &sbuf0,
        ScaleBuf &sbuf1,
        hls::stream<ap_uint<1> > &loaded_bank_stream,
        hls::stream<ap_uint<1> > &free_bank_stream,
        scale_stream_t &scale_stream,
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION function instances=drive_compute_tile limit=1
    #pragma HLS ARRAY_PARTITION variable=wbuf0.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wbuf1.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 cyclic factor=kScaleDrainColLanes dim=2
    #pragma HLS ARRAY_PARTITION variable=sbuf1.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf1.e0 cyclic factor=kScaleDrainColLanes dim=2

    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kMaxKtiles
        const ap_uint<1> bank = loaded_bank_stream.read();
        const bool final_kt = (kt == k_tiles - 1u);
        if (bank == 0u) {
            drive_compute_tile(act, wbuf0, sbuf0, act_word_base, kt, final_kt,
                               scale_stream, a_stream, w_stream, ctrl_stream);
        } else {
            drive_compute_tile(act, wbuf1, sbuf1, act_word_base, kt, final_kt,
                               scale_stream, a_stream, w_stream, ctrl_stream);
        }
        if (kt + 2u < k_tiles) {
            free_bank_stream.write(bank);
        }
    }
}

static void drive_all_ktiles_pp(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        const ActCache &act,
        uint32_t act_word_base,
        uint64_t w0_nt_base_words,
        uint64_t w1_nt_base_words,
        uint64_t w0_nt_scale_words,
        uint64_t w1_nt_scale_words,
        uint32_t k_tiles,
        scale_stream_t &scale_stream,
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off

    const uint32_t kt_w_words   = kWeightWordsPerKtilePerPort;
    const uint32_t kt_w_scale_w = kScaleWordsPerKtilePerPort;

    if (k_tiles == 0) {
        return;
    }

    (void)kt_w_words;
    (void)kt_w_scale_w;

    WeightBuf wbuf0, wbuf1;
    ScaleBuf  sbuf0, sbuf1;
    #pragma HLS BIND_STORAGE variable=wbuf0.words type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=wbuf1.words type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=wbuf0.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=wbuf1.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf0.e0 cyclic factor=kScaleDrainColLanes dim=2
    #pragma HLS ARRAY_PARTITION variable=sbuf1.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf1.e0 cyclic factor=kScaleDrainColLanes dim=2

    hls::stream<ap_uint<1> > loaded_bank_stream("loaded_bank_stream");
    hls::stream<ap_uint<1> > free_bank_stream("free_bank_stream");
    #pragma HLS STREAM variable=loaded_bank_stream depth=2
    #pragma HLS STREAM variable=free_bank_stream depth=2

#ifdef V5_CSIM
    // C-sim executes dataflow processes sequentially. Keep the compact
    // functional path there; synthesis uses the free-bank handshake below.
    WeightBuf wbuf;
    ScaleBuf sbuf;
    #pragma HLS BIND_STORAGE variable=wbuf.words type=ram_2p impl=bram
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        const uint64_t w0_kt_base_words  = w0_nt_base_words  + kt * kt_w_words;
        const uint64_t w1_kt_base_words  = w1_nt_base_words  + kt * kt_w_words;
        const uint64_t w0_kt_scale_words = w0_nt_scale_words + kt * kt_w_scale_w;
        const uint64_t w1_kt_scale_words = w1_nt_scale_words + kt * kt_w_scale_w;
        load_weight_scale_tile(w0_arena, w1_arena,
                               w0_kt_base_words, w1_kt_base_words,
                               w0_kt_scale_words, w1_kt_scale_words,
                               wbuf, sbuf);
        drive_compute_tile(act, wbuf, sbuf, act_word_base, kt, kt + 1u == k_tiles,
                           scale_stream, a_stream, w_stream, ctrl_stream);
    }
#else
    #pragma HLS DATAFLOW disable_start_propagation
    weight_pp_load_stage(w0_arena, w1_arena,
                         w0_nt_base_words, w1_nt_base_words,
                         w0_nt_scale_words, w1_nt_scale_words,
                         k_tiles,
                         wbuf0, wbuf1, sbuf0, sbuf1,
                         loaded_bank_stream, free_bank_stream);
    weight_pp_compute_stage(act, act_word_base, k_tiles,
                            wbuf0, wbuf1, sbuf0, sbuf1,
                            loaded_bank_stream, free_bank_stream,
                            scale_stream, a_stream, w_stream, ctrl_stream);
#endif
}

// Shared matmul datapath for one N-tile. It streams act/weight/control to the
// PE array and absorbs scaled psums into out_acc; callers decide whether to
// store out_acc to DDR or use it on chip.
static void run_ntile_accumulate(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        const ActCache &act,
        uint32_t act_word_base,
        uint64_t w0_nt_base_words,
        uint64_t w1_nt_base_words,
        uint64_t w0_nt_scale_words,
        uint64_t w1_nt_scale_words,
        uint32_t k_tiles,
        uint32_t rows_valid,
        const int8_t act_row_exp_kt[/*k_tiles*/][kSaM][kGroupsPerKtile],
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream,
        OutAcc &out_acc,
        const out_acc_t bias_tile[kSaN]) {
    #pragma HLS INLINE off
    (void)rows_valid;

    #pragma HLS ARRAY_PARTITION variable=out_acc.v cyclic factor=kScaleDrainColLanes dim=2

    scale_stream_t scale_stream("scale_stream");
    #pragma HLS STREAM variable=scale_stream depth=16

#ifdef V5_CSIM
    #pragma HLS DATAFLOW
    drive_all_ktiles_pp(w0_arena, w1_arena, act, act_word_base,
                        w0_nt_base_words, w1_nt_base_words,
                        w0_nt_scale_words, w1_nt_scale_words,
                        k_tiles, scale_stream,
                        a_stream, w_stream, ctrl_stream);
    v5_csim::pump_ktiles(k_tiles, a_stream, w_stream, ctrl_stream, psum_stream);
    absorb_all_ktiles(psum_stream, scale_stream, k_tiles, act_row_exp_kt,
                      out_acc, bias_tile);
#else
    #pragma HLS DATAFLOW disable_start_propagation
    drive_all_ktiles_pp(w0_arena, w1_arena, act, act_word_base,
                        w0_nt_base_words, w1_nt_base_words,
                        w0_nt_scale_words, w1_nt_scale_words,
                        k_tiles, scale_stream,
                        a_stream, w_stream, ctrl_stream);
    absorb_all_ktiles(psum_stream, scale_stream, k_tiles, act_row_exp_kt,
                      out_acc, bias_tile);
#endif
}
#endif

static void weight_pp_compute_stage_batch(
        const ActCache &act,
        const uint32_t act_word_base[kBatchMtiles],
        uint32_t slabs_valid,
        uint32_t k_tiles,
        WeightBuf &wbuf0,
        WeightBuf &wbuf1,
        ScaleBuf &sbuf0,
        ScaleBuf &sbuf1,
        hls::stream<ap_uint<1> > &loaded_bank_stream,
        hls::stream<ap_uint<1> > &free_bank_stream,
        scale_stream_t &scale_stream,
        a_stream_t &a_stream,
        w_stream_t &w_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=kMaxKtiles
        const ap_uint<1> bank = loaded_bank_stream.read();
        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=kBatchMtiles
            if (bank == 0u) {
                drive_compute_tile(act, wbuf0, sbuf0, act_word_base[slab],
                                   kt, kt + 1u == k_tiles,
                                   scale_stream, a_stream, w_stream, ctrl_stream);
            } else {
                drive_compute_tile(act, wbuf1, sbuf1, act_word_base[slab],
                                   kt, kt + 1u == k_tiles,
                                   scale_stream, a_stream, w_stream, ctrl_stream);
            }
        }
        if (kt + 2u < k_tiles) {
            free_bank_stream.write(bank);
        }
    }
}

static void drive_all_ktiles_batch_pp(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        const ActCache &act,
        const uint32_t act_word_base[kBatchMtiles],
        uint32_t slabs_valid,
        uint64_t w0_nt_base_words,
        uint64_t w1_nt_base_words,
        uint64_t w0_nt_scale_words,
        uint64_t w1_nt_scale_words,
        uint32_t k_tiles,
        scale_stream_t &scale_stream,
        a_stream_t &a_stream,
        w_stream_t &w_stream,
        ctrl_stream_t &ctrl_stream) {
    #pragma HLS INLINE off
    WeightBuf wbuf;
    ScaleBuf sbuf;
    #pragma HLS BIND_STORAGE variable=wbuf.words type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=wbuf.words complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sbuf.e0 cyclic factor=kScaleDrainColLanes dim=2

    // Keep the multi-slab weight-stationary reuse inside one dataflow producer.
    // Passing wbuf/sbuf through a separate load-stage dataflow process lets HLS
    // lower each array element to a FIFO. That is not reusable across slabs:
    // slab 0 consumes the FIFO token and slab 1 blocks on an empty sbuf/wbuf
    // channel. A local buffer preserves RAM semantics and still loads each
    // resident K tile only once for all valid slabs.
    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        const uint64_t w0_kt = w0_nt_base_words + kt * kWeightWordsPerKtilePerPort;
        const uint64_t w1_kt = w1_nt_base_words + kt * kWeightWordsPerKtilePerPort;
        const uint64_t s0_kt = w0_nt_scale_words + kt * kScaleWordsPerKtilePerPort;
        const uint64_t s1_kt = w1_nt_scale_words + kt * kScaleWordsPerKtilePerPort;
        load_weight_scale_tile(w0_arena, w1_arena, w0_kt, w1_kt, s0_kt, s1_kt, wbuf, sbuf);
        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            drive_compute_tile(act, wbuf, sbuf, act_word_base[slab],
                               kt, kt + 1u == k_tiles,
                               scale_stream, a_stream, w_stream, ctrl_stream);
        }
    }
}

static void run_ntile_accumulate_batch(
        const axi_word_t *w0_arena,
        const axi_word_t *w1_arena,
        const ActCache &act,
        const uint32_t act_word_base[kBatchMtiles],
        uint32_t slabs_valid,
        uint64_t w0_nt_base_words,
        uint64_t w1_nt_base_words,
        uint64_t w0_nt_scale_words,
        uint64_t w1_nt_scale_words,
        uint32_t k_tiles,
        const int8_t act_row_exp_kt[kBatchMtiles][kMaxKtiles][kSaM][kGroupsPerKtile],
        a_stream_t &a_stream,
        w_stream_t &w_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream,
        OutAccBatch &out_batch,
        const out_acc_t bias_tile[kSaN]) {
    #pragma HLS INLINE off
    scale_stream_t scale_stream("batch_scale_stream");
    #pragma HLS STREAM variable=scale_stream depth=32
#ifdef V5_CSIM
    drive_all_ktiles_batch_pp(w0_arena, w1_arena, act, act_word_base, slabs_valid,
                              w0_nt_base_words, w1_nt_base_words,
                              w0_nt_scale_words, w1_nt_scale_words, k_tiles,
                              scale_stream, a_stream, w_stream, ctrl_stream);
    v5_csim::pump_ktiles_batch(k_tiles, slabs_valid,
                               a_stream, w_stream, ctrl_stream, psum_stream);
    absorb_all_ktiles_batch(psum_stream, scale_stream, k_tiles, slabs_valid,
                            act_row_exp_kt, out_batch, bias_tile);
#else
    #pragma HLS DATAFLOW disable_start_propagation
    drive_all_ktiles_batch_pp(w0_arena, w1_arena, act, act_word_base, slabs_valid,
                              w0_nt_base_words, w1_nt_base_words,
                              w0_nt_scale_words, w1_nt_scale_words, k_tiles,
                              scale_stream, a_stream, w_stream, ctrl_stream);
    absorb_all_ktiles_batch(psum_stream, scale_stream, k_tiles, slabs_valid,
                            act_row_exp_kt, out_batch, bias_tile);
#endif
}

// QKV/O top-level: BM4 slabs outer, then output_count and n_tiles.
static void run_prefill_dense(
        const axi_word_t *w0,
        const axi_word_t *w1,
        const axi_word_t *act_arena,
        axi_word_t *out_arena,
        const task_t &task,
        ActCache &act_cache,
        OutAccBatch &work_batch,
        out_acc_t bias_cache[kBiasCacheCols],
        a_stream_t   &a_stream,
        w_stream_t   &w_stream,
        ctrl_stream_t &ctrl_stream,
        psum_stream_t &psum_stream) {
    #pragma HLS INLINE

    const uint32_t M        = task.rows;
    const uint32_t K        = task.input_cols;
    const uint32_t k_tiles  = div_ceil(K, kTileK);
    const uint32_t k_groups = div_ceil(K, kGroupSize);
    const uint32_t m_tiles  = div_ceil(M, kSaM);
    const uint32_t k_words_valid = div_ceil(K, kAxiBytes);
    const uint32_t k_words_act = k_tiles * kActWordsPerKtile;

    static int8_t act_row_exp_kt[kBatchMtiles][kMaxKtiles][kSaM][kGroupsPerKtile];

    bool has_bias = false;
    for (uint32_t out_idx = 0; out_idx < VLM_W8A8_LINEAR_MAX_OUTPUTS; ++out_idx) {
        #pragma HLS UNROLL
        has_bias = has_bias || (task.bias_offset_bytes[out_idx] != 0);
    }
    if (has_bias) {
        uint32_t bias_base = 0;
        for (uint32_t out_idx = 0; out_idx < task.output_count; ++out_idx) {
            const uint32_t cols = task.out_cols[out_idx];
            load_bias_vector_q22_i40(act_arena,
                                     task.bias_offset_bytes[out_idx],
                                     cols,
                                     bias_base,
                                     bias_cache);
            bias_base += cols;
        }
    }

    for (uint32_t mt0 = 0; mt0 < m_tiles; mt0 += kBatchMtiles) {
        const uint32_t slabs_valid =
            (mt0 + kBatchMtiles <= m_tiles) ? kBatchMtiles : (m_tiles - mt0);
        uint32_t rows_valid[kBatchMtiles];
        uint32_t act_word_base[kBatchMtiles];
        for (uint32_t slab = 0; slab < slabs_valid; ++slab) {
            const uint32_t mt_row0 = (mt0 + slab) * kSaM;
            rows_valid[slab] = (mt_row0 + kSaM <= M) ? kSaM : (M - mt_row0);
            act_word_base[slab] = input_slab_word_base(slab);
            load_activation_block(act_arena,
                                  task.act_q_offset_bytes + (uint64_t)mt_row0 * task.act_row_stride,
                                  task.act_scale_offset_bytes + (uint64_t)mt_row0 * task.act_scale_row_stride,
                                  task.act_row_stride, task.act_scale_row_stride,
                                  rows_valid[slab], k_words_valid, k_words_act,
                                  k_groups, slab, act_cache);
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                for (uint32_t r = 0; r < kSaM; ++r) {
                    for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                        #pragma HLS UNROLL
                        const uint32_t glob_g = kt * kGroupsPerKtile + g;
                        act_row_exp_kt[slab][kt][r][g] =
                            (glob_g < k_groups) ? (int8_t)act_cache.row_exp[slab][r][glob_g] : (int8_t)0;
                    }
                }
            }
        }

        for (uint32_t out_idx = 0; out_idx < task.output_count; ++out_idx) {
            const uint32_t N = task.out_cols[out_idx];
            const uint32_t n_tiles = div_ceil(N, kSaN);

            for (uint32_t nt = 0; nt < n_tiles; ++nt) {
                const uint64_t w0_nt_base = (task.weight_q_offset_bytes[out_idx][0]
                                             + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                const uint64_t w1_nt_base = (task.weight_q_offset_bytes[out_idx][1]
                                             + (uint64_t)nt * task.weight_n_tile_stride_bytes) / kAxiBytes;
                const uint64_t w0_nt_scl  = (task.weight_scale_offset_bytes[out_idx][0]
                                             + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;
                const uint64_t w1_nt_scl  = (task.weight_scale_offset_bytes[out_idx][1]
                                             + (uint64_t)nt * task.weight_scale_n_tile_stride_bytes) / kAxiBytes;

                const uint64_t out_byte_off = task.dst_offset_bytes[out_idx]
                    + (uint64_t)mt0 * kSaM * task.output_row_stride_bytes[out_idx];
                const uint32_t out_col_byte_off = nt * kSaN * kOutBytes;

                out_acc_t bias_tile[kSaN];
                #pragma HLS ARRAY_PARTITION variable=bias_tile complete dim=1
                if (has_bias && task.bias_offset_bytes[out_idx] != 0) {
                    uint32_t bias_cache_base = 0;
                    for (uint32_t bi = 0; bi < out_idx; ++bi) {
                        bias_cache_base += task.out_cols[bi];
                    }
                    load_bias_tile_from_cache(bias_cache,
                                              bias_cache_base + nt * kSaN,
                                              bias_tile);
                } else {
                    for (uint32_t c = 0; c < kSaN; ++c) {
                        #pragma HLS UNROLL
                        bias_tile[c] = 0;
                    }
                }

                run_ntile_accumulate_batch(w0, w1, act_cache, act_word_base, slabs_valid,
                                     w0_nt_base, w1_nt_base, w0_nt_scl, w1_nt_scl,
                                     k_tiles,
                                     act_row_exp_kt,
                                     a_stream, w_stream, ctrl_stream, psum_stream,
                                     work_batch,
                                     bias_tile);
                store_out_acc_batch(out_arena, out_byte_off,
                              task.output_row_stride_bytes[out_idx],
                              slabs_valid, rows_valid, out_col_byte_off, work_batch);
            }
        }
    }
}

} // namespace v5
