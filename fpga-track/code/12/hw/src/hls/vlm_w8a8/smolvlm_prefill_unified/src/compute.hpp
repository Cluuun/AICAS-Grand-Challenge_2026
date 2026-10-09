static scaled_acc_t round_shift_right_fixed(scaled_acc_t value, int shift) {
    #pragma HLS INLINE
    if (shift <= 0) {
        return value;
    }
    const scaled_acc_t half = scaled_acc_t(1) << (shift - 1);
    if (value >= 0) {
        value += half;
    } else {
        value -= half;
    }
    return value >> shift;
}

static scaled_acc_t scale_dot_to_fixed(dot_acc_t dot, scale_exp_t exp) {
    #pragma HLS INLINE
    if (exp == scale_exp_t(kScaleExpDisabled)) {
        return 0;
    }
    const int shift = static_cast<int>(exp) + kV2OutFrac;
    scaled_acc_t value = static_cast<scaled_acc_t>(dot);
    if (shift >= 0) {
        return value << shift;
    }
    return round_shift_right_fixed(value, -shift);
}

static float fixed_acc_to_f32(scaled_acc_t value) {
    #pragma HLS INLINE
    const scaled_acc_t max_i32 = scaled_acc_t(2147483647);
    const scaled_acc_t min_i32 = scaled_acc_t(-2147483647) - 1;
    ap_int<32> narrowed = 0;
    if (value > max_i32) {
        narrowed = ap_int<32>(2147483647);
    } else if (value < min_i32) {
        narrowed = ap_int<32>(-2147483647) - 1;
    } else {
        narrowed = static_cast<ap_int<32> >(value);
    }
    const float scaled = static_cast<float>(narrowed);
    return scaled * 5.9604644775390625e-8f;
}

static void clear_tile_acc(scaled_acc_t out_acc[kMaxTileM][kTileN], uint32_t rows) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kAccBankN dim=2

    for (uint32_t mi_base = 0; mi_base < rows; mi_base += kMaxPeM) {
        for (uint32_t lane_base = 0; lane_base < kTileN; lane_base += kReducePeN) {
            #pragma HLS PIPELINE II=1
            for (uint32_t mii = 0; mii < kMaxPeM; ++mii) {
                #pragma HLS UNROLL
                for (uint32_t lii = 0; lii < kReducePeN; ++lii) {
                    #pragma HLS UNROLL
                    out_acc[mi_base + mii][lane_base + lii] = 0;
                }
            }
        }
    }
}

static void compute_row_group_core(
        uint32_t mi_base,
        uint32_t n_offset,
        uint32_t rows,
        const vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        const scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups],
        uint32_t act_base,
        uint32_t scale_count,
        const ap_uint<32> weight_q[kTileN][kWeightChunksPerLane],
        const weight_scale_exp_t weight_scale[kTileN][kScalePerTile],
        scaled_acc_t out_acc[kMaxTileM][kTileN]) {
    #pragma HLS INLINE off
    #pragma HLS DEPENDENCE variable=out_acc inter false
    #pragma HLS ARRAY_PARTITION variable=act_q complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kAccBankN dim=2

    // The inner product packs two row lanes into one DSP multiply
    // (`vlm_w8a8_dsp_mul_2row_i8`) before signs are restored.
    dot_acc_t local_acc[kMaxPeM][kPeN];
    #pragma HLS ARRAY_PARTITION variable=local_acc complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_acc complete dim=2
    scaled_acc_t group_acc[kMaxPeM][kPeN];
    #pragma HLS ARRAY_PARTITION variable=group_acc complete dim=1
    #pragma HLS ARRAY_PARTITION variable=group_acc cyclic factor=kAccBankN dim=2

    const uint32_t scale_group_base = act_base / VLM_W8A8_QK;
    for (uint32_t mii = 0; mii < kMaxPeM; ++mii) {
        #pragma HLS UNROLL
        for (uint32_t ni = 0; ni < kPeN; ++ni) {
            #pragma HLS UNROLL
            group_acc[mii][ni] = 0;
        }
    }

    for (uint32_t sb = 0; sb < scale_count; ++sb) {
        for (uint32_t mii = 0; mii < kMaxPeM; ++mii) {
            #pragma HLS UNROLL
            for (uint32_t ni = 0; ni < kPeN; ++ni) {
                #pragma HLS UNROLL
                local_acc[mii][ni] = 0;
            }
        }

        const uint32_t block_base = act_base + sb * VLM_W8A8_QK;
        for (uint32_t kk_group = 0; kk_group < kQkSubTileCount; ++kk_group) {
            #pragma HLS PIPELINE II=1
            ap_uint<32> act_chunks[kMaxPeM][kKSubTile / 4u];
            #pragma HLS ARRAY_PARTITION variable=act_chunks complete dim=1
            #pragma HLS ARRAY_PARTITION variable=act_chunks complete dim=2
            ap_int<20> act_correction[kMaxPeM];
            #pragma HLS ARRAY_PARTITION variable=act_correction complete
            const uint32_t kk_base = kk_group * kKSubTile;
            for (uint32_t mii = 0; mii < kMaxPeM; ++mii) {
                #pragma HLS UNROLL
                const uint32_t mi = mi_base + mii;
                const bool valid_row = mi < rows;
                for (uint32_t ko_base = 0; ko_base < kKSubTile; ko_base += 4u) {
                    #pragma HLS UNROLL
                    const uint32_t act_byte_idx = block_base + kk_base + ko_base;
                    const uint32_t act_word_idx = act_byte_idx / 16u;
                    const uint32_t act_lane_idx = act_byte_idx % 16u;
                    const vlm_w8a8_axi_t act_word = valid_row ? act_q[mi][act_word_idx] : vlm_w8a8_axi_t(0);
                    ap_uint<32> act_chunk = 0;
                    const uint32_t act_chunk_idx = act_lane_idx >> 2u;
                    if (act_chunk_idx == 0u) {
                        act_chunk = act_word.range(31, 0);
                    } else if (act_chunk_idx == 1u) {
                        act_chunk = act_word.range(63, 32);
                    } else if (act_chunk_idx == 2u) {
                        act_chunk = act_word.range(95, 64);
                    } else {
                        act_chunk = act_word.range(127, 96);
                    }
                    act_chunks[mii][ko_base / 4u] = act_chunk;
                }
                ap_int<11> act_sum = 0;
                for (uint32_t ko_base = 0; ko_base < kKSubTile; ko_base += 4u) {
                    #pragma HLS UNROLL
                    act_sum += vlm_i8x4_sum_pack(act_chunks[mii][ko_base / 4u]);
                }
                act_correction[mii] = static_cast<ap_int<20> >(act_sum) << 7;
            }
            for (uint32_t ni = 0; ni < kPeN; ++ni) {
                #pragma HLS UNROLL
                const uint32_t n_local = n_offset + ni;
                const uint32_t weight_byte_base = sb * VLM_W8A8_QK + kk_base;
                const uint32_t weight_chunk_idx = weight_byte_base / 4u;
                ap_uint<32> w_chunks[kKSubTile / 4u];
                #pragma HLS ARRAY_PARTITION variable=w_chunks complete
                for (uint32_t ko_base = 0; ko_base < kKSubTile; ko_base += 4u) {
                    #pragma HLS UNROLL
                    w_chunks[ko_base / 4u] = weight_q[n_local][weight_chunk_idx + (ko_base / 4u)];
                }
                ap_int<11> weight_sum = 0;
                for (uint32_t ko_base = 0; ko_base < kKSubTile; ko_base += 4u) {
                    #pragma HLS UNROLL
                    weight_sum += vlm_i8x4_sum_pack(w_chunks[ko_base / 4u]);
                }
                const ap_int<20> weight_correction = static_cast<ap_int<20> >(weight_sum) << 7;
                for (uint32_t mii = 0; mii < kMaxPeM; mii += 2u) {
                    #pragma HLS UNROLL
                    dot_acc_t dot0 = local_acc[mii][ni];
                    dot_acc_t dot1 = local_acc[mii + 1u][ni];
                    ap_uint<19> raw0 = 0;
                    ap_uint<19> raw1 = 0;
                    for (uint32_t ko_base = 0; ko_base < kKSubTile; ko_base += 4u) {
                        #pragma HLS UNROLL
                        const ap_uint<36> raw_pair = vlm_i8x2_dot4_u8packed_raw_pack(
                                act_chunks[mii][ko_base / 4u],
                                act_chunks[mii + 1u][ko_base / 4u],
                                w_chunks[ko_base / 4u]);
                        raw0 += raw_pair.range(17, 0);
                        raw1 += raw_pair.range(35, 18);
                    }
                    const dot_acc_t offset_correction = static_cast<dot_acc_t>(kKSubTile * 16384u);
                    const dot_acc_t partial0 = static_cast<ap_int<20> >(raw0) -
                            act_correction[mii] - weight_correction - offset_correction;
                    const dot_acc_t partial1 = static_cast<ap_int<20> >(raw1) -
                            act_correction[mii + 1u] - weight_correction - offset_correction;
                    dot0 += partial0;
                    dot1 += partial1;
                    local_acc[mii][ni] = dot0;
                    local_acc[mii + 1u][ni] = dot1;
                }
            }
        }

        for (uint32_t ni_base = 0; ni_base < kPeN; ni_base += kReducePeN) {
            #pragma HLS PIPELINE II=1
            for (uint32_t mii = 0; mii < kMaxPeM; ++mii) {
                #pragma HLS UNROLL
                const uint32_t mi = mi_base + mii;
                for (uint32_t nii = 0; nii < kReducePeN; ++nii) {
                    #pragma HLS UNROLL
                    const uint32_t ni = ni_base + nii;
                    const uint32_t n_local = n_offset + ni;
                    const scale_exp_t a_exp = act_scale_2d[mi][scale_group_base + sb];
                    const weight_scale_exp_t w_exp = weight_scale[n_local][sb];
                    scaled_acc_t tile_sum = 0;
                    if (w_exp.e0 != scale_exp_t(kScaleExpDisabled)) {
                        tile_sum += scale_dot_to_fixed(local_acc[mii][ni], a_exp + w_exp.e0);
                    }
                    if (w_exp.e1 != scale_exp_t(kScaleExpDisabled)) {
                        tile_sum += scale_dot_to_fixed(local_acc[mii][ni], a_exp + w_exp.e1);
                    }
                    group_acc[mii][ni] += tile_sum;
                }
            }
        }
    }

    for (uint32_t mii = 0; mii < kMaxPeM; ++mii) {
        const uint32_t mi = mi_base + mii;
        for (uint32_t ni_base = 0; ni_base < kPeN; ni_base += kAccBankN) {
            #pragma HLS PIPELINE II=1
            for (uint32_t nii = 0; nii < kAccBankN; ++nii) {
                #pragma HLS UNROLL
                const uint32_t ni = ni_base + nii;
                const uint32_t n_local = n_offset + ni;
                out_acc[mi][n_local] += group_acc[mii][ni];
            }
        }
    }
}

static void compute_tile_generic(
        uint32_t rows,
        const vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        const scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups],
        uint32_t act_base,
        const ap_uint<32> weight_q[kTileN][kWeightChunksPerLane],
        const weight_scale_exp_t weight_scale[kTileN][kScalePerTile],
        uint32_t scale_count,
        scaled_acc_t out_acc[kMaxTileM][kTileN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=act_q complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kAccBankN dim=2

    const uint32_t row_group_count = (rows + kMaxPeM - 1u) / kMaxPeM;
    for (uint32_t n_offset = 0; n_offset < kTileN; n_offset += kPeN) {
        for (uint32_t mi_group = 0; mi_group < row_group_count; ++mi_group) {
            const uint32_t mi_base = mi_group * kMaxPeM;
            compute_row_group_core(
                    mi_base,
                    n_offset,
                    rows,
                    act_q,
                    act_scale_2d,
                    act_base,
                    scale_count,
                    weight_q,
                    weight_scale,
                    out_acc);
        }
    }
}

static void compute_tile_stage(
        bool do_compute,
        uint32_t rows,
        const vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        const scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups],
        uint32_t act_base,
        const ap_uint<32> weight_q[kTileN][kWeightChunksPerLane],
        const weight_scale_exp_t weight_scale[kTileN][kScalePerTile],
        uint32_t scale_count,
        scaled_acc_t out_acc[kMaxTileM][kTileN]) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=act_q complete dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kAccBankN dim=2

    if (!do_compute) {
        return;
    }
    compute_tile_generic(rows, act_q, act_scale_2d, act_base, weight_q, weight_scale, scale_count, out_acc);
}

static void load_full_weight_tile_stage(
        bool do_load,
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        uint64_t q0_offset_bytes,
        uint64_t scale0_offset_bytes,
        uint64_t q1_offset_bytes,
        uint64_t scale1_offset_bytes,
        uint32_t k_tile_count,
        uint32_t n_base,
        uint32_t k_tile_idx,
        ap_uint<32> weight_q[kTileN][kWeightChunksPerLane],
        weight_scale_exp_t weight_scale[kTileN][kScalePerTile]);

static void overlap_full_tile_stage(
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        uint64_t q0_offset_bytes,
        uint64_t scale0_offset_bytes,
        uint64_t q1_offset_bytes,
        uint64_t scale1_offset_bytes,
        uint32_t k_tile_count,
        uint32_t n_base,
        uint32_t rows,
        const vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        const scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups],
        const ap_uint<32> current_weight_q[kTileN][kWeightChunksPerLane],
        const weight_scale_exp_t current_weight_scale[kTileN][kScalePerTile],
        ap_uint<32> next_weight_q[kTileN][kWeightChunksPerLane],
        weight_scale_exp_t next_weight_scale[kTileN][kScalePerTile],
        scaled_acc_t out_acc[kMaxTileM][kTileN],
        bool do_compute,
        uint32_t current_act_base,
        bool do_load,
        uint32_t next_k_tile_idx) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    #pragma HLS ARRAY_PARTITION variable=act_q complete dim=1

    compute_tile_stage(do_compute, rows, act_q, act_scale_2d, current_act_base, current_weight_q, current_weight_scale, kScalePerTile, out_acc);
    load_full_weight_tile_stage(do_load, weight0_arena, weight1_arena, q0_offset_bytes, scale0_offset_bytes, q1_offset_bytes, scale1_offset_bytes, k_tile_count, n_base, next_k_tile_idx, next_weight_q, next_weight_scale);
}

static void load_full_weight_tile_stage(
        bool do_load,
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        uint64_t q0_offset_bytes,
        uint64_t scale0_offset_bytes,
        uint64_t q1_offset_bytes,
        uint64_t scale1_offset_bytes,
        uint32_t k_tile_count,
        uint32_t n_base,
        uint32_t k_tile_idx,
        ap_uint<32> weight_q[kTileN][kWeightChunksPerLane],
        weight_scale_exp_t weight_scale[kTileN][kScalePerTile]) {
    #pragma HLS INLINE off

    if (!do_load) {
        return;
    }
    load_full_weight_tile(weight0_arena, weight1_arena, q0_offset_bytes, scale0_offset_bytes, q1_offset_bytes, scale1_offset_bytes, k_tile_count, n_base, k_tile_idx, weight_q, weight_scale);
}

static void store_output_tile(
        vlm_w8a8_axi_t * out_arena,
        uint64_t dst_offset_bytes,
        uint32_t row_base,
        uint32_t rows,
        uint32_t n_base,
        uint32_t valid_cols,
        const scaled_acc_t out_acc[kMaxTileM][kTileN],
        uint32_t out_cols) {
    #pragma HLS INLINE off

    for (uint32_t mi = 0; mi < rows; ++mi) {
        const uint64_t row_offset = dst_offset_bytes + static_cast<uint64_t>(row_base + mi) * out_cols * sizeof(uint32_t);
        const uint64_t row_word_base = (row_offset / 16u) + n_base / 4u;
        for (uint32_t word_idx = 0; word_idx < valid_cols / 4u; ++word_idx) {
            #pragma HLS PIPELINE II=1
            vlm_w8a8_axi_t word = 0;
            for (uint32_t off = 0; off < 4u; ++off) {
                #pragma HLS UNROLL
                const uint32_t lane = word_idx * 4u + off;
                const uint32_t bits = vlm_w8a8_f32_bits(fixed_acc_to_f32(out_acc[mi][lane]));
                word.range(static_cast<int>((off + 1u) * 32u - 1u), static_cast<int>(off * 32u)) = bits;
            }
            out_arena[row_word_base + word_idx] = word;
        }
    }
}

static void run_dense_output_tile(
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        uint64_t q0_offset_bytes,
        uint64_t scale0_offset_bytes,
        uint64_t q1_offset_bytes,
        uint64_t scale1_offset_bytes,
        uint32_t k_tile_count,
        uint32_t n_base,
        uint32_t rows,
        const vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        const scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups],
        scaled_acc_t out_acc[kMaxTileM][kTileN],
        vlm_w8a8_profile_t & profile) {
    #pragma HLS INLINE off

    ap_uint<32> weight_q_ping[kTileN][kWeightChunksPerLane];
    ap_uint<32> weight_q_pong[kTileN][kWeightChunksPerLane];
    weight_scale_exp_t weight_scale_ping[kTileN][kScalePerTile];
    weight_scale_exp_t weight_scale_pong[kTileN][kScalePerTile];
    #pragma HLS BIND_STORAGE variable=weight_q_ping type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=weight_q_pong type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=weight_q_ping complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight_q_pong complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight_scale_ping complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight_scale_pong complete dim=1
    #pragma HLS ARRAY_PARTITION variable=weight_scale_ping complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight_scale_pong complete dim=2
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kAccBankN dim=2

    clear_tile_acc(out_acc, rows);

    // Prime the ping-pong buffer so every following iteration can compute the
    // current K tile while loading the next one from the weight AXI bundle.
    overlap_full_tile_stage(
            weight0_arena,
            weight1_arena,
            q0_offset_bytes,
            scale0_offset_bytes,
            q1_offset_bytes,
            scale1_offset_bytes,
            k_tile_count,
            n_base,
            rows,
            act_q,
            act_scale_2d,
            weight_q_pong,
            weight_scale_pong,
            weight_q_ping,
            weight_scale_ping,
            out_acc,
            false,
            0u,
            true,
            0u);

    const uint32_t last_k_tile_idx = k_tile_count - 1u;
    for (uint32_t k_tile_idx = 0; k_tile_idx < last_k_tile_idx; ++k_tile_idx) {
        if ((k_tile_idx & 1u) == 0u) {
            overlap_full_tile_stage(
                    weight0_arena,
                    weight1_arena,
                    q0_offset_bytes,
                    scale0_offset_bytes,
                    q1_offset_bytes,
                    scale1_offset_bytes,
                    k_tile_count,
                    n_base,
                    rows,
                    act_q,
                    act_scale_2d,
                    weight_q_ping,
                    weight_scale_ping,
                    weight_q_pong,
                    weight_scale_pong,
                    out_acc,
                    true,
                    k_tile_idx * kTileK,
                    true,
                    k_tile_idx + 1u);
        } else {
            overlap_full_tile_stage(
                    weight0_arena,
                    weight1_arena,
                    q0_offset_bytes,
                    scale0_offset_bytes,
                    q1_offset_bytes,
                    scale1_offset_bytes,
                    k_tile_count,
                    n_base,
                    rows,
                    act_q,
                    act_scale_2d,
                    weight_q_pong,
                    weight_scale_pong,
                    weight_q_ping,
                    weight_scale_ping,
                    out_acc,
                    true,
                    k_tile_idx * kTileK,
                    true,
                    k_tile_idx + 1u);
        }
    }

    if ((last_k_tile_idx & 1u) == 0u) {
        overlap_full_tile_stage(
                weight0_arena,
                weight1_arena,
                q0_offset_bytes,
                scale0_offset_bytes,
                q1_offset_bytes,
                scale1_offset_bytes,
                k_tile_count,
                n_base,
                rows,
                act_q,
                act_scale_2d,
                weight_q_ping,
                weight_scale_ping,
                weight_q_pong,
                weight_scale_pong,
                out_acc,
                true,
                last_k_tile_idx * kTileK,
                false,
                0u);
    } else {
        overlap_full_tile_stage(
                weight0_arena,
                weight1_arena,
                q0_offset_bytes,
                scale0_offset_bytes,
                q1_offset_bytes,
                scale1_offset_bytes,
                k_tile_count,
                n_base,
                rows,
                act_q,
                act_scale_2d,
                weight_q_pong,
                weight_scale_pong,
                weight_q_ping,
                weight_scale_ping,
                out_acc,
                true,
                last_k_tile_idx * kTileK,
                false,
                0u);
    }
    profile.cycles_load_weight += static_cast<uint64_t>(k_tile_count) * profile_cycles_load_weight_tile();
    profile.cycles_compute_a += static_cast<uint64_t>(k_tile_count) * profile_cycles_compute_a(rows, kScalePerTile);
    profile.cycles_compute_b += static_cast<uint64_t>(k_tile_count) * profile_cycles_compute_b(rows);
}
