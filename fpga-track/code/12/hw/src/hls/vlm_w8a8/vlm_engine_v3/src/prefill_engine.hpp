// prefill_engine.hpp — Prefill top-level: dense GEMM and FFN orchestration.
#pragma once

#include "types.hpp"
#include "prefill_ctrl.hpp"
#include "prefill_scale.hpp"
#include "ffn_fusion.hpp"

// Compute one group (32 K-elements) using weight-broadcast model.
// Reads from weight buffer and activation buffer, accumulates into pe_acc.
static void compute_group(
        const vlm_w8a8_axi_t weight_buf[kWeightWordsPerKtile],
        const vlm_w8a8_axi_t act_buf[kSaM][kActWordsPerKtile],
        uint32_t group_in_ktile,
        int32_t pe_acc[kSaM][kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=act_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=pe_acc complete dim=2

    const uint32_t group_word_base = group_in_ktile * (kGroupSize * VLM_W8A8_WEIGHT_PORTS);

    for (uint32_t k = 0; k < kGroupSize; ++k) {
        #pragma HLS PIPELINE II=1

        // Extract 32 weights for this k from weight buffer (port0: col 0-15, port1: col 16-31)
        // In the unified buffer, all 32 cols are stored: word index = group*32*32/16 + k*32/16
        int8_t w_vals[kSaN];
        #pragma HLS ARRAY_PARTITION variable=w_vals complete
        for (uint32_t wi = 0; wi < VLM_W8A8_WEIGHT_PORTS; ++wi) {
            #pragma HLS UNROLL
            vlm_w8a8_axi_t ww = weight_buf[group_word_base + k * VLM_W8A8_WEIGHT_PORTS + wi];
            for (uint32_t b = 0; b < kAxiBytes; ++b) {
                #pragma HLS UNROLL
                w_vals[wi * kAxiBytes + b] = (int8_t)(uint8_t)ww(b * 8 + 7, b * 8);
            }
        }

        // Extract activation[row][k] for all 32 rows
        uint32_t ktile_k = group_in_ktile * kGroupSize + k;
        uint32_t act_word_idx = ktile_k / kAxiBytes;
        uint32_t act_byte_idx = ktile_k % kAxiBytes;

        for (uint32_t row = 0; row < kSaM; ++row) {
            #pragma HLS UNROLL
            vlm_w8a8_axi_t aw = act_buf[row][act_word_idx];
            int8_t a_val = (int8_t)(uint8_t)aw(act_byte_idx * 8 + 7, act_byte_idx * 8);
            for (uint32_t col = 0; col < kSaN; ++col) {
                #pragma HLS UNROLL
                pe_acc[row][col] += (int32_t)a_val * (int32_t)w_vals[col];
            }
        }
    }
}

// Run one N-tile: iterate over all K-tiles, accumulate scaled results.
static void run_ntile(
        const vlm_w8a8_axi_t *act_arena,
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        uint64_t act_base_word,
        uint64_t act_scale_base_byte,
        uint32_t row_start,
        uint32_t row_stride_words,
        uint32_t scale_groups,
        uint32_t valid_rows,
        uint64_t weight0_base_word,
        uint64_t weight1_base_word,
        uint64_t scale0_base_word,
        uint64_t scale1_base_word,
        uint32_t k_tiles,
        int32_t out_acc[kSaM][kSaN]) {
    #pragma HLS INLINE off

    vlm_w8a8_axi_t w_buf_ping[kWeightWordsPerKtile];
    vlm_w8a8_axi_t act_buf[kSaM][kActWordsPerKtile];
    scale_exp_t act_scale[kSaM][kGroupsPerKtile];
    scale_exp_t w_scale[kGroupsPerKtile][kSaN];
    int32_t pe_acc[kSaM][kSaN];
    #pragma HLS ARRAY_PARTITION variable=act_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=act_scale complete dim=0
    #pragma HLS ARRAY_PARTITION variable=pe_acc complete dim=2
    #pragma HLS ARRAY_PARTITION variable=w_scale complete dim=0
    #pragma HLS BIND_STORAGE variable=act_buf type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=pe_acc type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=w_buf_ping type=ram_2p impl=bram

    clear_out_acc(out_acc);

    for (uint32_t kt = 0; kt < k_tiles; ++kt) {
        uint64_t q0_kt_base = weight0_base_word + (uint64_t)kt * kWeightWordsPerPort;
        uint64_t q1_kt_base = weight1_base_word + (uint64_t)kt * kWeightWordsPerPort;
        uint64_t s0_kt_base = scale0_base_word + (uint64_t)kt * kScaleWordsPerPort;
        uint64_t s1_kt_base = scale1_base_word + (uint64_t)kt * kScaleWordsPerPort;

        load_activation_ktile(act_arena, act_base_word, act_scale_base_byte, row_start,
                              row_stride_words, scale_groups, kt, valid_rows,
                              act_buf, act_scale);
        load_weight_ktile_dual(weight0_arena, weight1_arena, q0_kt_base, q1_kt_base, w_buf_ping);
        load_weight_scale_ktile_dual(weight0_arena, weight1_arena, s0_kt_base, s1_kt_base, w_scale);

        // Compute 4 groups within this K-tile
        for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
            clear_pe_acc(pe_acc);
            compute_group(w_buf_ping, act_buf, g, pe_acc);

            // Scale and accumulate
            scale_exp_t a_exp_group[kSaM];
            #pragma HLS ARRAY_PARTITION variable=a_exp_group complete
            for (uint32_t r = 0; r < kSaM; ++r) {
                #pragma HLS UNROLL
                a_exp_group[r] = act_scale[r][g];
            }
            apply_scale_group(pe_acc, a_exp_group, w_scale[g], out_acc);
        }
    }
}

// Store output tile (SA_M × SA_N int32 values) to output arena
static void store_output_tile(
        vlm_w8a8_axi_t *out_arena,
        uint64_t base_word_addr,
        uint32_t out_row_stride_words,
        const int32_t out_acc[kSaM][kSaN],
        uint32_t valid_rows,
        uint32_t col_offset) {
    #pragma HLS INLINE off
    // Each int32 = 4 bytes, SA_N=32 → 128 bytes = 8 words per row
    const uint32_t words_per_row = kSaN * 4 / kAxiBytes;  // 8
    const uint32_t col_word_offset = col_offset * 4 / kAxiBytes;

    for (uint32_t row = 0; row < valid_rows; ++row) {
        uint64_t row_addr = base_word_addr + (uint64_t)row * out_row_stride_words + col_word_offset;
        for (uint32_t w = 0; w < words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            vlm_w8a8_axi_t word = 0;
            for (uint32_t i = 0; i < kAxiBytes / 4; ++i) {
                #pragma HLS UNROLL
                uint32_t col = w * (kAxiBytes / 4) + i;
                int32_t val = out_acc[row][col];
                word((i + 1) * 32 - 1, i * 32) = static_cast<uint32_t>(val);
            }
            out_arena[row_addr + w] = word;
        }
    }
}

static void zero_output_tile(
        vlm_w8a8_axi_t *out_arena,
        uint64_t base_word_addr,
        uint32_t out_row_stride_words,
        uint32_t valid_rows,
        uint32_t col_offset) {
    #pragma HLS INLINE off
    const uint32_t words_per_row = kSaN * 4 / kAxiBytes;
    const uint32_t col_word_offset = col_offset * 4 / kAxiBytes;

    for (uint32_t row = 0; row < valid_rows; ++row) {
        uint64_t row_addr = base_word_addr + (uint64_t)row * out_row_stride_words + col_word_offset;
        for (uint32_t w = 0; w < words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            out_arena[row_addr + w] = 0;
        }
    }
}

static void add_store_output_tile(
        vlm_w8a8_axi_t *out_arena,
        uint64_t base_word_addr,
        uint32_t out_row_stride_words,
        const int32_t out_acc[kSaM][kSaN],
        uint32_t valid_rows,
        uint32_t col_offset) {
    #pragma HLS INLINE off
    const uint32_t words_per_row = kSaN * 4 / kAxiBytes;
    const uint32_t col_word_offset = col_offset * 4 / kAxiBytes;

    for (uint32_t row = 0; row < valid_rows; ++row) {
        uint64_t row_addr = base_word_addr + (uint64_t)row * out_row_stride_words + col_word_offset;
        for (uint32_t w = 0; w < words_per_row; ++w) {
            #pragma HLS PIPELINE II=1
            vlm_w8a8_axi_t word = out_arena[row_addr + w];
            for (uint32_t i = 0; i < kAxiBytes / 4; ++i) {
                #pragma HLS UNROLL
                uint32_t col = w * (kAxiBytes / 4) + i;
                uint32_t old_bits = word((i + 1) * 32 - 1, i * 32);
                int32_t old_val = static_cast<int32_t>(old_bits);
                int32_t next_val = old_val + out_acc[row][col];
                word((i + 1) * 32 - 1, i * 32) = static_cast<uint32_t>(next_val);
            }
            out_arena[row_addr + w] = word;
        }
    }
}

// Run dense GEMM task (prefill path)
static void run_prefill_dense(
        const vlm_w8a8_axi_t *act_arena,
        vlm_w8a8_axi_t *out_arena,
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        const accelerator_task_t &task,
        vlm_w8a8_profile_t &profile) {
    #pragma HLS INLINE off

    const uint32_t rows = task.rows;
    const uint32_t input_cols = task.input_cols;
    const uint32_t input_padded = align_up(input_cols, kTileK);
    const uint32_t k_tiles = input_padded / kTileK;
    const uint32_t act_row_stride_words = task.act_row_stride / kAxiBytes;
    const uint32_t scale_groups = div_ceil(input_cols, kGroupSize);
    const uint32_t m_tiles = div_ceil(rows, kSaM);

    int32_t out_acc[kSaM][kSaN];
    #pragma HLS ARRAY_PARTITION variable=out_acc complete dim=2
    #pragma HLS BIND_STORAGE variable=out_acc type=ram_2p impl=lutram

    for (uint32_t out_idx = 0; out_idx < task.output_count; ++out_idx) {
        const uint32_t out_cols = task.out_cols[out_idx];
        const uint32_t n_tiles = div_ceil(out_cols, kSaN);
        const uint32_t out_row_stride_words = align_up(out_cols * 4, kAxiBytes) / kAxiBytes;

        for (uint32_t mt = 0; mt < m_tiles; ++mt) {
            const uint32_t row_start = mt * kSaM;
            const uint32_t valid_rows = (row_start + kSaM <= rows) ? kSaM : (rows - row_start);
            const uint64_t act_base = task_act_offset(task) / kAxiBytes + (uint64_t)row_start * act_row_stride_words;

            for (uint32_t nt = 0; nt < n_tiles; ++nt) {
                uint64_t w0_base = task_weight_offset(task, out_idx) / kAxiBytes +
                                   (uint64_t)nt * k_tiles * kWeightWordsPerPort;
                uint64_t w1_base = task_weight1_offset(task, out_idx) / kAxiBytes +
                                   (uint64_t)nt * k_tiles * kWeightWordsPerPort;
                uint64_t s0_base = task_scale_offset(task, out_idx) / kAxiBytes +
                                   (uint64_t)nt * k_tiles * kScaleWordsPerPort;
                uint64_t s1_base = task.weight_scale_offset_bytes_port1[out_idx] / kAxiBytes +
                                   (uint64_t)nt * k_tiles * kScaleWordsPerPort;

                run_ntile(act_arena, weight0_arena, weight1_arena,
                          act_base, task.act_scale_offset_bytes, row_start,
                          act_row_stride_words, scale_groups, valid_rows,
                          w0_base, w1_base, s0_base, s1_base, k_tiles, out_acc);

                uint64_t out_base = task_out_offset(task, out_idx) / kAxiBytes + (uint64_t)row_start * out_row_stride_words;
                store_output_tile(out_arena, out_base, out_row_stride_words, out_acc, valid_rows, nt * kSaN);
            }
        }
    }
}

static void pack_mid_activation_tile(
        const int8_t mid_q[kSaM][kSaN],
        const scale_exp_t mid_scale[kSaM],
        vlm_w8a8_axi_t act_buf[kSaM][kActWordsPerKtile],
        scale_exp_t act_scale[kSaM][kGroupsPerKtile],
        uint32_t group_in_ktile) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=mid_q complete dim=2
    #pragma HLS ARRAY_PARTITION variable=act_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=act_scale complete dim=0

    for (uint32_t row = 0; row < kSaM; ++row) {
        for (uint32_t w = 0; w < kActWordsPerKtile; ++w) {
            #pragma HLS PIPELINE II=1
            vlm_w8a8_axi_t word = act_buf[row][w];
            const uint32_t word_group = (w * kAxiBytes) / kGroupSize;
            if (word_group == group_in_ktile) {
                for (uint32_t b = 0; b < kAxiBytes; ++b) {
                    #pragma HLS UNROLL
                    const uint32_t col = (w * kAxiBytes + b) - group_in_ktile * kGroupSize;
                    word(b * 8 + 7, b * 8) = static_cast<uint8_t>(mid_q[row][col]);
                }
            }
            act_buf[row][w] = word;
        }
        act_scale[row][group_in_ktile] = mid_scale[row];
    }
}

static void clear_local_activation_tile(
        vlm_w8a8_axi_t act_buf[kSaM][kActWordsPerKtile],
        scale_exp_t act_scale[kSaM][kGroupsPerKtile]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=act_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=act_scale complete dim=0
    for (uint32_t row = 0; row < kSaM; ++row) {
        for (uint32_t w = 0; w < kActWordsPerKtile; ++w) {
            #pragma HLS PIPELINE II=1
            act_buf[row][w] = 0;
        }
        for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
            #pragma HLS PIPELINE II=1
            act_scale[row][g] = 0;
        }
    }
}

static scale_exp_t load_scale_exp_byte(const vlm_w8a8_axi_t *arena, uint64_t base_word, uint32_t group, uint32_t col) {
    #pragma HLS INLINE
    const uint32_t flat = group * kWeightColsPerPort + (col % kWeightColsPerPort);
    const vlm_w8a8_axi_t word = arena[base_word + flat / kAxiBytes];
    const uint32_t byte_idx = flat % kAxiBytes;
    return static_cast<scale_exp_t>(static_cast<ap_int<8> >(word(byte_idx * 8 + 7, byte_idx * 8)));
}

static scale_exp_t load_act_scale_exp_byte(
        const vlm_w8a8_axi_t *act_arena,
        uint64_t scale_base_byte_addr,
        uint32_t row_abs,
        uint32_t scale_groups,
        uint32_t group_abs) {
    #pragma HLS INLINE
    if (scale_base_byte_addr == 0u || group_abs >= scale_groups) {
        return 0;
    }
    const uint64_t byte_addr = scale_base_byte_addr + static_cast<uint64_t>(row_abs) * scale_groups + group_abs;
    const vlm_w8a8_axi_t word = act_arena[byte_addr / kAxiBytes];
    const uint32_t byte_idx = byte_addr % kAxiBytes;
    return static_cast<scale_exp_t>(static_cast<ap_int<8> >(word(byte_idx * 8 + 7, byte_idx * 8)));
}

// Lightweight 32-column matmul used for FFN gate/up. It avoids instantiating a
// second 1024-DSP prefill array while preserving the fused task semantics.
static void run_dense32_light_from_ddr(
        const vlm_w8a8_axi_t *act_arena,
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        uint64_t act_base_word,
        uint64_t act_scale_base_byte,
        uint32_t row_start,
        uint32_t row_stride_words,
        uint32_t scale_groups,
        uint32_t valid_rows,
        uint64_t weight0_base_word,
        uint64_t weight1_base_word,
        uint64_t scale0_base_word,
        uint64_t scale1_base_word,
        uint32_t input_padded,
        int32_t out_acc[kSaM][kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=out_acc complete dim=2

    clear_out_acc(out_acc);
    const uint32_t k_tiles = input_padded / kTileK;
    for (uint32_t row = 0; row < kSaM; ++row) {
        if (row < valid_rows) {
            const uint64_t row_word_base = act_base_word + static_cast<uint64_t>(row) * row_stride_words;
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                    int32_t part[kSaN];
                    #pragma HLS ARRAY_PARTITION variable=part complete
                    for (uint32_t c = 0; c < kSaN; ++c) {
                        #pragma HLS PIPELINE II=1
                        part[c] = 0;
                    }

                    for (uint32_t kk = 0; kk < kGroupSize; ++kk) {
                        const uint32_t k_in_tile = g * kGroupSize + kk;
                        const uint32_t k_abs = kt * kTileK + k_in_tile;
                        const vlm_w8a8_axi_t act_word = act_arena[row_word_base + k_abs / kAxiBytes];
                        const int8_t a_val = static_cast<int8_t>(static_cast<uint8_t>(act_word((k_abs % kAxiBytes) * 8 + 7, (k_abs % kAxiBytes) * 8)));
                        const uint64_t woff = static_cast<uint64_t>(kt) * kWeightWordsPerPort + g * kGroupSize + kk;
                        const vlm_w8a8_axi_t w0 = weight0_arena[weight0_base_word + woff];
                        const vlm_w8a8_axi_t w1 = weight1_arena[weight1_base_word + woff];
                        for (uint32_t c_base = 0; c_base < kWeightColsPerPort; c_base += kFfnColsPerCycle) {
                            #pragma HLS PIPELINE II=1
                            for (uint32_t lane = 0; lane < kFfnColsPerCycle; ++lane) {
                                #pragma HLS UNROLL
                                const uint32_t c = c_base + lane;
                                const int8_t w0v = static_cast<int8_t>(static_cast<uint8_t>(w0(c * 8 + 7, c * 8)));
                                const int8_t w1v = static_cast<int8_t>(static_cast<uint8_t>(w1(c * 8 + 7, c * 8)));
                                part[c] += static_cast<int32_t>(a_val) * static_cast<int32_t>(w0v);
                                part[kWeightColsPerPort + c] += static_cast<int32_t>(a_val) * static_cast<int32_t>(w1v);
                            }
                        }
                    }

                    const scale_exp_t a_exp = load_act_scale_exp_byte(act_arena, act_scale_base_byte,
                                                                      row_start + row, scale_groups,
                                                                      kt * kGroupsPerKtile + g);
                    for (uint32_t c = 0; c < kSaN; ++c) {
                        #pragma HLS UNROLL factor=4
                        const scale_exp_t w_exp = (c < kWeightColsPerPort)
                            ? load_scale_exp_byte(weight0_arena, scale0_base_word + static_cast<uint64_t>(kt) * kScaleWordsPerPort, g, c)
                            : load_scale_exp_byte(weight1_arena, scale1_base_word + static_cast<uint64_t>(kt) * kScaleWordsPerPort, g, c);
                        out_acc[row][c] += apply_exp_shift_i32(part[c], a_exp + w_exp);
                    }
                }
            }
        }
    }
}

static void run_dense32_light_from_local(
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        const vlm_w8a8_axi_t act_buf[kSaM][kActWordsPerKtile],
        const scale_exp_t act_scale[kSaM][kGroupsPerKtile],
        uint32_t valid_rows,
        uint64_t weight0_base_word,
        uint64_t weight1_base_word,
        uint64_t scale0_base_word,
        uint64_t scale1_base_word,
        int32_t out_acc[kSaM][kSaN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=act_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=act_scale complete dim=0
    #pragma HLS ARRAY_PARTITION variable=out_acc complete dim=2

    clear_out_acc(out_acc);
    for (uint32_t row = 0; row < kSaM; ++row) {
        if (row < valid_rows) {
            for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                int32_t part[kSaN];
                #pragma HLS ARRAY_PARTITION variable=part complete
                for (uint32_t c = 0; c < kSaN; ++c) {
                    #pragma HLS PIPELINE II=1
                    part[c] = 0;
                }
                for (uint32_t kk = 0; kk < kGroupSize; ++kk) {
                    const uint32_t k_in_tile = g * kGroupSize + kk;
                    const vlm_w8a8_axi_t act_word = act_buf[row][k_in_tile / kAxiBytes];
                    const int8_t a_val = static_cast<int8_t>(static_cast<uint8_t>(act_word((k_in_tile % kAxiBytes) * 8 + 7, (k_in_tile % kAxiBytes) * 8)));
                    const uint64_t woff = g * kGroupSize + kk;
                    const vlm_w8a8_axi_t w0 = weight0_arena[weight0_base_word + woff];
                    const vlm_w8a8_axi_t w1 = weight1_arena[weight1_base_word + woff];
                    for (uint32_t c_base = 0; c_base < kWeightColsPerPort; c_base += kFfnColsPerCycle) {
                        #pragma HLS PIPELINE II=1
                        for (uint32_t lane = 0; lane < kFfnColsPerCycle; ++lane) {
                            #pragma HLS UNROLL
                            const uint32_t c = c_base + lane;
                            const int8_t w0v = static_cast<int8_t>(static_cast<uint8_t>(w0(c * 8 + 7, c * 8)));
                            const int8_t w1v = static_cast<int8_t>(static_cast<uint8_t>(w1(c * 8 + 7, c * 8)));
                            part[c] += static_cast<int32_t>(a_val) * static_cast<int32_t>(w0v);
                            part[kWeightColsPerPort + c] += static_cast<int32_t>(a_val) * static_cast<int32_t>(w1v);
                        }
                    }
                }
                for (uint32_t c = 0; c < kSaN; ++c) {
                    #pragma HLS UNROLL factor=4
                    const scale_exp_t w_exp = (c < kWeightColsPerPort)
                        ? load_scale_exp_byte(weight0_arena, scale0_base_word, g, c)
                        : load_scale_exp_byte(weight1_arena, scale1_base_word, g, c);
                    out_acc[row][c] += apply_exp_shift_i32(part[c], act_scale[row][g] + w_exp);
                }
            }
        }
    }
}

// Fused FFN computes gate/up one 128-wide intermediate K-tile at a time,
// quantizes into a local activation tile, then immediately consumes it for down.
// The FFN MAC is intentionally narrower than the main prefill array so the
// single IP can keep the large dense GEMM core resident and still fit KV260.
static void run_prefill_ffn(
        const vlm_w8a8_axi_t *act_arena,
        vlm_w8a8_axi_t *out_arena,
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        const accelerator_task_t &task,
        vlm_w8a8_profile_t &profile) {
    #pragma HLS INLINE off

    const uint32_t rows = task.rows;
    const uint32_t input_cols = (task.ffn_input_cols != 0u) ? task.ffn_input_cols : task.input_cols;
    const uint32_t mid_cols = task.ffn_intermediate_cols;
    const uint32_t out_cols = task.ffn_output_cols;
    const uint32_t input_padded = align_up(input_cols, kTileK);
    const uint32_t input_k_tiles = input_padded / kTileK;
    const uint32_t input_scale_groups = div_ceil(input_cols, kGroupSize);
    const uint32_t mid_tiles = div_ceil(mid_cols, kSaN);
    const uint32_t mid_ktiles = div_ceil(mid_cols, kTileK);
    const uint32_t out_ntiles = div_ceil(out_cols, kSaN);
    const uint32_t m_tiles = div_ceil(rows, kSaM);
    const uint32_t act_row_stride_words = task.act_row_stride / kAxiBytes;
    const uint32_t out_row_stride_words = align_up(out_cols * 4, kAxiBytes) / kAxiBytes;

    int32_t gate_acc[kSaM][kSaN];
    int32_t up_acc[kSaM][kSaN];
    int32_t down_part[kSaM][kSaN];
    int8_t mid_q[kSaM][kSaN];
    scale_exp_t mid_scale[kSaM];
    vlm_w8a8_axi_t mid_act_buf[kSaM][kActWordsPerKtile];
    scale_exp_t mid_act_scale[kSaM][kGroupsPerKtile];
    #pragma HLS ARRAY_PARTITION variable=gate_acc complete dim=2
    #pragma HLS ARRAY_PARTITION variable=up_acc complete dim=2
    #pragma HLS ARRAY_PARTITION variable=down_part complete dim=2
    #pragma HLS ARRAY_PARTITION variable=mid_q complete dim=2
    #pragma HLS ARRAY_PARTITION variable=mid_scale complete
    #pragma HLS ARRAY_PARTITION variable=mid_act_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=mid_act_scale complete dim=0
    #pragma HLS BIND_STORAGE variable=gate_acc type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=up_acc type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=down_part type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=mid_act_buf type=ram_2p impl=lutram

    for (uint32_t mt = 0; mt < m_tiles; ++mt) {
        const uint32_t row_start = mt * kSaM;
        const uint32_t valid_rows = (row_start + kSaM <= rows) ? kSaM : (rows - row_start);
        const uint64_t act_base = task_act_offset(task) / kAxiBytes + (uint64_t)row_start * act_row_stride_words;
        const uint64_t out_base = task.ffn_dst_offset_bytes / kAxiBytes + (uint64_t)row_start * out_row_stride_words;

        for (uint32_t ont = 0; ont < out_ntiles; ++ont) {
            zero_output_tile(out_arena, out_base, out_row_stride_words, valid_rows, ont * kSaN);
        }

        for (uint32_t mkt = 0; mkt < mid_ktiles; ++mkt) {
            clear_local_activation_tile(mid_act_buf, mid_act_scale);

            for (uint32_t g = 0; g < kGroupsPerKtile; ++g) {
                const uint32_t mid_nt = mkt * kGroupsPerKtile + g;
                if (mid_nt < mid_tiles) {
                    const uint64_t gate_w0 = task.gate_q_offset_bytes / kAxiBytes +
                                             (uint64_t)mid_nt * input_k_tiles * kWeightWordsPerPort;
                    const uint64_t gate_w1 = task.gate_q_offset_bytes_port1 / kAxiBytes +
                                             (uint64_t)mid_nt * input_k_tiles * kWeightWordsPerPort;
                    const uint64_t gate_s0 = task.gate_scale_offset_bytes / kAxiBytes +
                                             (uint64_t)mid_nt * input_k_tiles * kScaleWordsPerPort;
                    const uint64_t gate_s1 = task.gate_scale_offset_bytes_port1 / kAxiBytes +
                                             (uint64_t)mid_nt * input_k_tiles * kScaleWordsPerPort;
                    const uint64_t up_w0 = task.up_q_offset_bytes / kAxiBytes +
                                           (uint64_t)mid_nt * input_k_tiles * kWeightWordsPerPort;
                    const uint64_t up_w1 = task.up_q_offset_bytes_port1 / kAxiBytes +
                                           (uint64_t)mid_nt * input_k_tiles * kWeightWordsPerPort;
                    const uint64_t up_s0 = task.up_scale_offset_bytes / kAxiBytes +
                                           (uint64_t)mid_nt * input_k_tiles * kScaleWordsPerPort;
                    const uint64_t up_s1 = task.up_scale_offset_bytes_port1 / kAxiBytes +
                                           (uint64_t)mid_nt * input_k_tiles * kScaleWordsPerPort;

                    run_dense32_light_from_ddr(act_arena, weight0_arena, weight1_arena,
                                               act_base, task.act_scale_offset_bytes, row_start,
                                               act_row_stride_words, input_scale_groups, valid_rows,
                                               gate_w0, gate_w1, gate_s0, gate_s1,
                                               input_padded, gate_acc);
                    run_dense32_light_from_ddr(act_arena, weight0_arena, weight1_arena,
                                               act_base, task.act_scale_offset_bytes, row_start,
                                               act_row_stride_words, input_scale_groups, valid_rows,
                                               up_w0, up_w1, up_s0, up_s1,
                                               input_padded, up_acc);

                    ffn_activate_and_quantize(gate_acc, up_acc, task.ffn_activation, mid_q, mid_scale);
                    pack_mid_activation_tile(mid_q, mid_scale, mid_act_buf, mid_act_scale, g);
                }
            }

            for (uint32_t ont = 0; ont < out_ntiles; ++ont) {
                const uint64_t down_w0 = task.down_q_offset_bytes / kAxiBytes +
                                         ((uint64_t)ont * mid_ktiles + mkt) * kWeightWordsPerPort;
                const uint64_t down_w1 = task.down_q_offset_bytes_port1 / kAxiBytes +
                                         ((uint64_t)ont * mid_ktiles + mkt) * kWeightWordsPerPort;
                const uint64_t down_s0 = task.down_scale_offset_bytes / kAxiBytes +
                                         ((uint64_t)ont * mid_ktiles + mkt) * kScaleWordsPerPort;
                const uint64_t down_s1 = task.down_scale_offset_bytes_port1 / kAxiBytes +
                                         ((uint64_t)ont * mid_ktiles + mkt) * kScaleWordsPerPort;

                run_dense32_light_from_local(weight0_arena, weight1_arena,
                                             mid_act_buf, mid_act_scale, valid_rows,
                                             down_w0, down_w1, down_s0, down_s1,
                                             down_part);
                add_store_output_tile(out_arena, out_base, out_row_stride_words,
                                      down_part, valid_rows, ont * kSaN);
            }
        }
    }
}
