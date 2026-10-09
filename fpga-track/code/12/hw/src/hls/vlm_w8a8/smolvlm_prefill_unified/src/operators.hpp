static scaled_acc_t fixed_abs_acc(scaled_acc_t value) {
    #pragma HLS INLINE
    if (value < 0) {
        const scaled_acc_t neg = static_cast<scaled_acc_t>(-value);
        return neg;
    }
    return value;
}

static scaled_acc_t fixed_from_i32(int32_t value) {
    #pragma HLS INLINE
    return static_cast<scaled_acc_t>(value) << kV2OutFrac;
}

static scaled_acc_t fixed_mul_q24(scaled_acc_t lhs, scaled_acc_t rhs) {
    #pragma HLS INLINE
    const scaled_acc_t max_i48 = (scaled_acc_t(1) << 47) - 1;
    const scaled_acc_t min_i48 = -(scaled_acc_t(1) << 47);
    ap_int<48> lhs_narrow = 0;
    ap_int<48> rhs_narrow = 0;
    if (lhs > max_i48) {
        lhs_narrow = ap_int<48>(max_i48);
    } else if (lhs < min_i48) {
        lhs_narrow = ap_int<48>(min_i48);
    } else {
        lhs_narrow = static_cast<ap_int<48> >(lhs);
    }
    if (rhs > max_i48) {
        rhs_narrow = ap_int<48>(max_i48);
    } else if (rhs < min_i48) {
        rhs_narrow = ap_int<48>(min_i48);
    } else {
        rhs_narrow = static_cast<ap_int<48> >(rhs);
    }
    const ap_int<96> product = static_cast<ap_int<96> >(lhs_narrow) * static_cast<ap_int<96> >(rhs_narrow);
    return static_cast<scaled_acc_t>(product >> kV2OutFrac);
}

static scaled_acc_t fixed_hardsigmoid_gate(scaled_acc_t value) {
    #pragma HLS INLINE
    const scaled_acc_t three = fixed_from_i32(3);
    const scaled_acc_t six = fixed_from_i32(6);
    scaled_acc_t gate = value + three;
    if (gate < 0) {
        gate = 0;
    } else if (gate > six) {
        gate = six;
    }
    return gate;
}

static scaled_acc_t fixed_hardgelu_gate(scaled_acc_t value) {
    #pragma HLS INLINE
    const scaled_acc_t three = fixed_from_i32(3);
    const scaled_acc_t one_and_half = three >> 1;
    scaled_acc_t gate = value + one_and_half;
    if (gate < 0) {
        gate = 0;
    } else if (gate > three) {
        gate = three;
    }
    return gate;
}

static scaled_acc_t fixed_ffn_activation(scaled_acc_t value, uint32_t activation) {
    #pragma HLS INLINE
    if (activation == LINEAR_FFN_ACT_GELU) {
        const scaled_acc_t gate = fixed_hardgelu_gate(value);
        return fixed_mul_q24(value, gate) / 3;
    }
    const scaled_acc_t gate = fixed_hardsigmoid_gate(value);
    return fixed_mul_q24(value, gate) / 6;
}

static bool fixed_pot_scale_covers(ap_uint<64> max_abs, int exp) {
    #pragma HLS INLINE
    const int shift = exp + kV2OutFrac;
    ap_uint<64> threshold = 0;
    if (shift >= 0) {
        threshold = ap_uint<64>(127) << shift;
    } else {
        threshold = ap_uint<64>(127) >> (-shift);
    }
    return max_abs <= threshold;
}

static scale_exp_t pot_exp_ceil_from_fixed(ap_uint<64> max_abs) {
    #pragma HLS INLINE off
    if (max_abs == 0u) {
        return 0;
    }
    int msb = 0;
    bool found = false;
    for (int bit = 63; bit >= 0; --bit) {
        #pragma HLS PIPELINE II=1
        if (((max_abs >> bit) & 1u) != 0u && !found) {
            msb = bit;
            found = true;
        }
    }
    int exp = msb - kV2OutFrac - 6;
    if (exp < -30) {
        exp = -30;
    } else if (exp > 30) {
        exp = 30;
    }
    return static_cast<scale_exp_t>(exp);
}

static int8_t quantize_fixed_to_i8(scaled_acc_t value, scale_exp_t exp) {
    #pragma HLS INLINE
    const int shift = static_cast<int>(exp) + kV2OutFrac;
    scaled_acc_t q = 0;
    if (shift >= 0) {
        q = round_shift_right_fixed(value, shift);
    } else {
        q = value << (-shift);
    }
    if (q > 127) {
        q = 127;
    } else if (q < -127) {
        q = -127;
    }
    return static_cast<int8_t>(q);
}

static void quantize_mid_tile_to_full(
        uint32_t rows,
        uint32_t mid_base,
        bool use_vision_gelu,
        const scaled_acc_t gate_acc[kMaxTileM][kTileN],
        const scaled_acc_t up_acc[kMaxTileM][kTileN],
        vlm_w8a8_axi_t mid_q_full[kFfnTileM][kFfnMidWordsPerRow],
        scale_exp_t mid_scale_full[kFfnTileM][kFfnScaleGroups]) {
    #pragma HLS INLINE off

    const uint32_t scale_idx = mid_base / VLM_W8A8_QK;
    const uint32_t word_base = mid_base / 16u;

    // FFN fusion materializes only the quantized SiLU(gate) * up tile. The
    // down projection can then reuse the same W8A8 dense datapath.
    const uint32_t activation = use_vision_gelu ? LINEAR_FFN_ACT_GELU : LINEAR_FFN_ACT_SILU;
    for (uint32_t mi = 0; mi < rows; ++mi) {
        scaled_acc_t fused[kTileN];
        #pragma HLS ARRAY_PARTITION variable=fused complete
        for (uint32_t lane = 0; lane < kTileN; ++lane) {
            #pragma HLS PIPELINE II=1
            const scaled_acc_t gate_act = fixed_ffn_activation(gate_acc[mi][lane], activation);
            const scaled_acc_t val = fixed_mul_q24(gate_act, up_acc[mi][lane]);
            fused[lane] = val;
        }

        int8_t q_vals[kTileN];
        #pragma HLS ARRAY_PARTITION variable=q_vals complete
        for (uint32_t group = 0; group < kTileN / VLM_W8A8_QK; ++group) {
            ap_uint<64> max_abs = 0;
            for (uint32_t off = 0; off < VLM_W8A8_QK; ++off) {
                #pragma HLS PIPELINE II=1
                const uint32_t lane = group * VLM_W8A8_QK + off;
                const ap_uint<64> abs_val = static_cast<ap_uint<64> >(fixed_abs_acc(fused[lane]));
                if (abs_val > max_abs) {
                    max_abs = abs_val;
                }
            }
            const scale_exp_t exp = pot_exp_ceil_from_fixed(max_abs);
            mid_scale_full[mi][scale_idx + group] = exp;
            for (uint32_t off = 0; off < VLM_W8A8_QK; ++off) {
                #pragma HLS PIPELINE II=1
                const uint32_t lane = group * VLM_W8A8_QK + off;
                q_vals[lane] = quantize_fixed_to_i8(fused[lane], exp);
            }
        }

        for (uint32_t w = 0; w < kTileN / 16u; ++w) {
            #pragma HLS PIPELINE II=1
            vlm_w8a8_axi_t word = 0;
            for (uint32_t b = 0; b < 16u; ++b) {
                #pragma HLS UNROLL
                word.range(static_cast<int>((b + 1u) * 8u - 1u), static_cast<int>(b * 8u)) =
                        static_cast<uint8_t>(q_vals[w * 16u + b]);
            }
            mid_q_full[mi][word_base + w] = word;
        }
    }
}

static void run_dense_task(
        const vlm_w8a8_axi_t * act_arena,
        vlm_w8a8_axi_t * out_arena,
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        const accelerator_task_t & task,
        vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups],
        vlm_w8a8_profile_t & profile) {
    #pragma HLS INLINE

    scaled_acc_t out_acc[kMaxTileM][kTileN];
    #pragma HLS BIND_STORAGE variable=out_acc type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=out_acc cyclic factor=kAccBankN dim=2

    const uint32_t input_cols = task.input_cols;
    const uint32_t k_padded = align_up_u32(input_cols, kTileK);
    const uint32_t act_words_per_row = k_padded / 16u;
    const uint32_t k_tile_count = k_padded / kTileK;

    const uint32_t tile_m = task_uses_decode_gemv(task) ? kDecodeTileM : kDenseTileM;
    for (uint32_t row_base = 0; row_base < task.rows; row_base += tile_m) {
        #pragma HLS LOOP_FLATTEN off
        const uint32_t rows = (row_base + tile_m <= task.rows) ? tile_m : (task.rows - row_base);
        uint32_t load_act_cycles = 0u;
        load_activation_rows(
                act_arena,
                task.act_q_offset_bytes,
                task.act_scale_offset_bytes,
                task.act_row_stride,
                act_words_per_row,
                k_tile_count * kScalePerTile,
                row_base,
                rows,
                act_q,
                act_scale_2d);
        load_act_cycles = static_cast<uint32_t>(profile_cycles_load_act(act_words_per_row, k_tile_count * kScalePerTile));
        profile.cycles_load_act += load_act_cycles;
        for (uint32_t output_idx = 0; output_idx < task.output_count; ++output_idx) {
            #pragma HLS LOOP_FLATTEN off
            const uint32_t out_cols = task.out_cols[output_idx];
            const uint32_t n_padded = align_up_u32(out_cols, kTileN);
            for (uint32_t n_base = 0; n_base < n_padded; n_base += kTileN) {
                #pragma HLS LOOP_FLATTEN off
                const uint32_t valid_cols = (n_base + kTileN <= out_cols) ? kTileN : (out_cols - n_base);
                run_dense_output_tile(
                        weight0_arena,
                        weight1_arena,
                        task.weight_q_offset_bytes[output_idx],
                        task.weight_scale_offset_bytes[output_idx],
                        task.weight_q_offset_bytes_port1[output_idx],
                        task.weight_scale_offset_bytes_port1[output_idx],
                        k_tile_count,
                        n_base,
                        rows,
                        act_q,
                        act_scale_2d,
                        out_acc,
                        profile);
                store_output_tile(out_arena, task.dst_offset_bytes[output_idx], row_base, rows, n_base, valid_cols, out_acc, out_cols);
                const uint32_t store_cycles = static_cast<uint32_t>(profile_cycles_store(rows, valid_cols));
                profile.cycles_store += store_cycles;
            }
        }
    }
}

static void run_ffn_task(
        const vlm_w8a8_axi_t * act_arena,
        vlm_w8a8_axi_t * out_arena,
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        const accelerator_task_t & task,
        vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups],
        vlm_w8a8_profile_t & profile) {
    #pragma HLS INLINE

    scaled_acc_t gate_acc[kMaxTileM][kTileN];
    scaled_acc_t up_acc[kMaxTileM][kTileN];
    vlm_w8a8_axi_t mid_q_full[kFfnTileM][kFfnMidWordsPerRow];
    scale_exp_t mid_scale_full[kFfnTileM][kFfnScaleGroups];
    scaled_acc_t down_out_acc[kMaxTileM][kTileN];
    #pragma HLS BIND_STORAGE variable=gate_acc type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=up_acc type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=mid_q_full type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=down_out_acc type=ram_2p impl=lutram
    #pragma HLS ARRAY_PARTITION variable=gate_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=gate_acc cyclic factor=kAccBankN dim=2
    #pragma HLS ARRAY_PARTITION variable=up_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=up_acc cyclic factor=kAccBankN dim=2
    #pragma HLS ARRAY_PARTITION variable=mid_q_full cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=mid_scale_full cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=down_out_acc cyclic factor=kMaxPeM dim=1
    #pragma HLS ARRAY_PARTITION variable=down_out_acc cyclic factor=kAccBankN dim=2

    const bool use_vision_gelu = task.ffn_activation == LINEAR_FFN_ACT_GELU;
    const uint32_t ffn_input_cols = task.ffn_input_cols;
    const uint32_t ffn_intermediate_cols = task.ffn_intermediate_cols;
    const uint32_t ffn_output_cols = task.ffn_output_cols;
    const uint32_t ffn_input_padded = align_up_u32(ffn_input_cols, kTileK);
    const uint32_t ffn_intermediate_padded = align_up_u32(ffn_intermediate_cols, kTileK);
    const uint32_t ffn_output_padded = align_up_u32(ffn_output_cols, kTileN);
    const uint32_t input_words_per_row = ffn_input_padded / 16u;
    const uint32_t input_tile_count = ffn_input_padded / kTileK;
    const uint32_t input_scale_groups = input_tile_count * kScalePerTile;
    const uint32_t intermediate_words_per_row = ffn_intermediate_padded / 16u;
    const uint32_t intermediate_tile_count = ffn_intermediate_padded / kTileK;
    const uint32_t intermediate_scale_groups = intermediate_tile_count * kScalePerTile;
    const uint32_t tile_m = task_uses_decode_gemv(task) ? kDecodeTileM : kFfnTileM;
    for (uint32_t row_base = 0; row_base < task.rows; row_base += tile_m) {
        #pragma HLS LOOP_FLATTEN off
        const uint32_t rows = (row_base + tile_m <= task.rows) ? tile_m : (task.rows - row_base);
        uint32_t load_act_cycles = 0u;
        load_activation_rows(
                act_arena,
                task.act_q_offset_bytes,
                task.act_scale_offset_bytes,
                task.act_row_stride,
                input_words_per_row,
                input_scale_groups,
                row_base,
                rows,
                act_q,
                act_scale_2d);
        load_act_cycles = static_cast<uint32_t>(profile_cycles_load_act(input_words_per_row, input_scale_groups));
        profile.cycles_load_act += load_act_cycles;

        for (uint32_t mid_base = 0; mid_base < ffn_intermediate_cols; mid_base += kTileN) {
            #pragma HLS LOOP_FLATTEN off
            const uint64_t load_weight_before = profile.cycles_load_weight;
            const uint64_t compute_a_before = profile.cycles_compute_a;
            const uint64_t compute_b_before = profile.cycles_compute_b;
            run_dense_output_tile(
                    weight0_arena,
                    weight1_arena,
                    task.gate_q_offset_bytes,
                    task.gate_scale_offset_bytes,
                    task.gate_q_offset_bytes_port1,
                    task.gate_scale_offset_bytes_port1,
                    input_tile_count,
                    mid_base,
                    rows,
                    act_q,
                    act_scale_2d,
                    gate_acc,
                    profile);
            profile.cycles_ffn_gate += (profile.cycles_load_weight - load_weight_before) +
                    (profile.cycles_compute_a - compute_a_before) +
                    (profile.cycles_compute_b - compute_b_before);
            const uint64_t load_weight_before_up = profile.cycles_load_weight;
            const uint64_t compute_a_before_up = profile.cycles_compute_a;
            const uint64_t compute_b_before_up = profile.cycles_compute_b;
            run_dense_output_tile(
                    weight0_arena,
                    weight1_arena,
                    task.up_q_offset_bytes,
                    task.up_scale_offset_bytes,
                    task.up_q_offset_bytes_port1,
                    task.up_scale_offset_bytes_port1,
                    input_tile_count,
                    mid_base,
                    rows,
                    act_q,
                    act_scale_2d,
                    up_acc,
                    profile);
            profile.cycles_ffn_up += (profile.cycles_load_weight - load_weight_before_up) +
                    (profile.cycles_compute_a - compute_a_before_up) +
                    (profile.cycles_compute_b - compute_b_before_up);
            quantize_mid_tile_to_full(rows, mid_base, use_vision_gelu, gate_acc, up_acc, mid_q_full, mid_scale_full);
            profile.cycles_ffn_quantize += profile_cycles_quantize_mid(rows);
        }

        for (uint32_t mi = 0; mi < rows; ++mi) {
            for (uint32_t w = 0; w < kFfnMidWordsPerRow; ++w) {
                #pragma HLS PIPELINE II=1
                if (w < intermediate_words_per_row) {
                    act_q[mi][w] = mid_q_full[mi][w];
                }
            }
        }
        profile.cycles_ffn_quantize += profile_cycles_copy_mid_q(rows, intermediate_words_per_row);
        for (uint32_t mi = 0; mi < rows; ++mi) {
            for (uint32_t g = 0; g < kFfnScaleGroups; ++g) {
                #pragma HLS PIPELINE II=1
                if (g < intermediate_scale_groups) {
                    act_scale_2d[mi][g] = mid_scale_full[mi][g];
                }
            }
        }
        profile.cycles_ffn_quantize += profile_cycles_copy_mid_scale(rows, intermediate_scale_groups);

        for (uint32_t n_base = 0; n_base < ffn_output_padded; n_base += kTileN) {
            #pragma HLS LOOP_FLATTEN off
            const uint64_t load_weight_before_down = profile.cycles_load_weight;
            const uint64_t compute_a_before_down = profile.cycles_compute_a;
            const uint64_t compute_b_before_down = profile.cycles_compute_b;
            run_dense_output_tile(
                    weight0_arena,
                    weight1_arena,
                    task.down_q_offset_bytes,
                    task.down_scale_offset_bytes,
                    task.down_q_offset_bytes_port1,
                    task.down_scale_offset_bytes_port1,
                    intermediate_tile_count,
                    n_base,
                    rows,
                    act_q,
                    act_scale_2d,
                    down_out_acc,
                    profile);
            const uint32_t valid_cols = (n_base + kTileN <= ffn_output_cols) ? kTileN : (ffn_output_cols - n_base);
            store_output_tile(out_arena, task.ffn_dst_offset_bytes, row_base, rows, n_base, valid_cols, down_out_acc, ffn_output_cols);
            const uint32_t store_cycles = static_cast<uint32_t>(profile_cycles_store(rows, valid_cols));
            profile.cycles_store += store_cycles;
            profile.cycles_ffn_down += (profile.cycles_load_weight - load_weight_before_down) +
                    (profile.cycles_compute_a - compute_a_before_down) +
                    (profile.cycles_compute_b - compute_b_before_down) +
                    store_cycles;
        }
    }
}
