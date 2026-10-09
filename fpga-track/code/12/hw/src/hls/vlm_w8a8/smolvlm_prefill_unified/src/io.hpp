struct msu_port_t {
    const vlm_w8a8_axi_t * weight0;
    const vlm_w8a8_axi_t * weight1;
    const vlm_w8a8_axi_t * act;
    vlm_w8a8_axi_t * out;
};

static uint64_t msu_word_index(uint64_t byte_offset) {
    #pragma HLS INLINE
    return byte_offset / 16u;
}

static vlm_w8a8_axi_t msu_read128(const vlm_w8a8_axi_t * arena, uint64_t word_idx) {
    #pragma HLS INLINE
    return arena[word_idx];
}

static void msu_write128(vlm_w8a8_axi_t * arena, uint64_t word_idx, vlm_w8a8_axi_t value) {
    #pragma HLS INLINE
    arena[word_idx] = value;
}

static ap_uint<8> msu_extract_u8(vlm_w8a8_axi_t word, uint32_t byte_lane) {
    #pragma HLS INLINE
    const uint32_t shift = byte_lane * 8u;
    const vlm_w8a8_axi_t shifted = word >> shift;
    return shifted.range(7, 0);
}

static uint8_t msu_load_u8(const vlm_w8a8_axi_t * arena, uint64_t byte_offset) {
    #pragma HLS INLINE
    const uint64_t word_idx = msu_word_index(byte_offset);
    const uint32_t byte_lane = static_cast<uint32_t>(byte_offset & 0xfu);
    return static_cast<uint8_t>(msu_extract_u8(msu_read128(arena, word_idx), byte_lane));
}

static void load_task(
        const vlm_w8a8_axi_t * data_arena,
        uint64_t task_offset_bytes,
        accelerator_task_t * task) {
    #pragma HLS INLINE off

    task_buffer_t raw = {};
    const uint64_t first_word = msu_word_index(task_offset_bytes);
    for (uint32_t word_idx = 0; word_idx < kTaskTransferWords; ++word_idx) {
        #pragma HLS PIPELINE II=1
        raw.words[word_idx] = msu_read128(data_arena, first_word + word_idx);
    }
    *task = raw.task;
}

static vlm_w8a8_axi_t pack_u64_pair(uint64_t lo, uint64_t hi) {
    #pragma HLS INLINE

    vlm_w8a8_axi_t word = 0;
    word.range(63, 0) = lo;
    word.range(127, 64) = hi;
    return word;
}

static void store_profile(
        vlm_w8a8_axi_t * out_arena,
        uint64_t profile_offset_bytes,
        const vlm_w8a8_profile_t & profile) {
    #pragma HLS INLINE off

    profile_buffer_t raw = {};
    raw.profile = profile;
    const uint64_t first_word = msu_word_index(profile_offset_bytes);
    for (uint32_t word_idx = 0; word_idx < kProfileTransferWords; ++word_idx) {
        #pragma HLS PIPELINE II=1
        msu_write128(out_arena, first_word + word_idx, raw.words[word_idx]);
    }
}

static void load_activation_rows(
        const vlm_w8a8_axi_t * data_arena,
        uint64_t act_q_offset_bytes,
        uint64_t act_scale_offset_bytes,
        uint32_t act_row_stride,
        uint32_t act_words_per_row,
        uint32_t act_scale_group_count,
        uint32_t row_base,
        uint32_t rows,
        vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups]) {
    #pragma HLS INLINE off

    for (uint32_t mi = 0; mi < kMaxTileM; ++mi) {
        const bool valid_row = mi < rows;
        uint64_t row_word_base = 0;
        if (valid_row) {
            const uint32_t row = row_base + mi;
            row_word_base = (act_q_offset_bytes + static_cast<uint64_t>(row) * act_row_stride) / 16u;
        }
        for (uint32_t word_idx = 0; word_idx < act_words_per_row; ++word_idx) {
            #pragma HLS PIPELINE II=1
            act_q[mi][word_idx] = valid_row ? msu_read128(data_arena, row_word_base + word_idx) : vlm_w8a8_axi_t(0);
        }
        const uint64_t scale_row_byte_base =
                act_scale_offset_bytes + static_cast<uint64_t>(row_base + mi) * act_scale_group_count;
        const uint32_t first_scale_word = static_cast<uint32_t>(msu_word_index(scale_row_byte_base));
        const uint32_t scale_word_byte_offset = static_cast<uint32_t>(scale_row_byte_base % 16u);
        const uint32_t scale_words = (scale_word_byte_offset + act_scale_group_count + 15u) / 16u;
        vlm_w8a8_axi_t scale_buf[kMaxActScaleWordsPerRow];
        #pragma HLS ARRAY_PARTITION variable=scale_buf complete
        for (uint32_t word_idx = 0; word_idx < kMaxActScaleWordsPerRow; ++word_idx) {
            #pragma HLS PIPELINE II=1
            scale_buf[word_idx] = (valid_row && word_idx < scale_words) ? msu_read128(data_arena, first_scale_word + word_idx) : vlm_w8a8_axi_t(0);
        }
        for (uint32_t g = 0; g < act_scale_group_count; ++g) {
            #pragma HLS PIPELINE II=1
            const uint32_t local_byte = scale_word_byte_offset + g;
            const uint32_t word_idx = local_byte / 16u;
            const uint32_t byte_idx = local_byte % 16u;
            const vlm_w8a8_axi_t word = scale_buf[word_idx];
            const ap_uint<8> exp_bits = msu_extract_u8(word, byte_idx);
            act_scale_2d[mi][g] = valid_row ? static_cast<scale_exp_t>(static_cast<int8_t>(exp_bits)) : scale_exp_t(0);
        }
    }
}

static void load_weight_port_tile(
        const vlm_w8a8_axi_t * weight_arena,
        uint64_t q_offset_bytes,
        uint64_t scale_offset_bytes,
        uint32_t k_tile_count,
        uint32_t n_base,
        uint32_t k_tile_idx,
        uint32_t lane_base,
        ap_uint<32> weight_q[kTileN][kWeightChunksPerLane],
        weight_scale_exp_t weight_scale[kTileN][kScalePerTile]) {
    #pragma HLS INLINE off

    const uint32_t n_tile_idx = n_base / kTileN;
    const uint64_t tile_word_base = packed_q_port_byte_offset(q_offset_bytes, k_tile_count, n_tile_idx, k_tile_idx, 0u) / 16u;
    const uint32_t words_per_lane = kWeightWordsPerLane;
    const uint32_t total_words = kWeightPortTileN * words_per_lane;
    for (uint32_t idx = 0; idx < total_words; ++idx) {
        const vlm_w8a8_axi_t word = msu_read128(weight_arena, tile_word_base + idx);
        const uint32_t lane = idx / words_per_lane;
        const uint32_t global_lane = lane_base + lane;
        const uint32_t word_idx = idx % words_per_lane;
        const uint32_t chunk_base = word_idx * 4u;
        for (uint32_t chunk = 0; chunk < 4u; ++chunk) {
            #pragma HLS PIPELINE II=1
            weight_q[global_lane][chunk_base + chunk] =
                    word.range(static_cast<int>((chunk + 1u) * 32u - 1u), static_cast<int>(chunk * 32u));
        }
    }

    vlm_w8a8_axi_t scale_words[(kWeightPortTileN * kScalePerTile * 2u + 15u) / 16u];
    #pragma HLS ARRAY_PARTITION variable=scale_words complete
    const uint32_t scale_word_count = (kWeightPortTileN * kScalePerTile * 2u + 15u) / 16u;
    const uint64_t scale_word_base = packed_scale_port_byte_offset(scale_offset_bytes, k_tile_count, n_tile_idx, k_tile_idx, 0u, 0u) / 16u;
    for (uint32_t word_idx = 0; word_idx < scale_word_count; ++word_idx) {
        #pragma HLS PIPELINE II=1
        scale_words[word_idx] = msu_read128(weight_arena, scale_word_base + word_idx);
    }
    for (uint32_t lane = 0; lane < kWeightPortTileN; ++lane) {
        for (uint32_t sb = 0; sb < kScalePerTile; ++sb) {
            #pragma HLS PIPELINE II=1
            const uint32_t byte_offset = (lane * kScalePerTile + sb) * 2u;
            const uint32_t word_idx = byte_offset / 16u;
            const uint32_t byte_idx = byte_offset % 16u;
            const vlm_w8a8_axi_t word = scale_words[word_idx];
            const int8_t e0 = static_cast<int8_t>(msu_extract_u8(word, byte_idx));
            const int8_t e1 = static_cast<int8_t>(msu_extract_u8(word, byte_idx + 1u));
            const uint32_t global_lane = lane_base + lane;
            weight_scale[global_lane][sb].e0 = static_cast<scale_exp_t>(e0);
            weight_scale[global_lane][sb].e1 = static_cast<scale_exp_t>(e1);
        }
    }
}

static void load_full_weight_tile(
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

    const uint32_t n_tile_idx = n_base / kTileN;
    const uint64_t tile0_word_base = packed_q_port_byte_offset(q0_offset_bytes, k_tile_count, n_tile_idx, k_tile_idx, 0u) / 16u;
    const uint64_t tile1_word_base = packed_q_port_byte_offset(q1_offset_bytes, k_tile_count, n_tile_idx, k_tile_idx, 0u) / 16u;
    const uint32_t words_per_lane = kWeightWordsPerLane;
    const uint32_t total_words = kWeightPortTileN * words_per_lane;
    for (uint32_t idx = 0; idx < total_words; ++idx) {
        const vlm_w8a8_axi_t word0 = msu_read128(weight0_arena, tile0_word_base + idx);
        const vlm_w8a8_axi_t word1 = msu_read128(weight1_arena, tile1_word_base + idx);
        const uint32_t lane = idx / words_per_lane;
        const uint32_t word_idx = idx % words_per_lane;
        const uint32_t chunk_base = word_idx * 4u;
        for (uint32_t chunk = 0; chunk < 4u; ++chunk) {
            #pragma HLS PIPELINE II=1
            weight_q[lane][chunk_base + chunk] =
                    word0.range(static_cast<int>((chunk + 1u) * 32u - 1u), static_cast<int>(chunk * 32u));
            weight_q[kWeightPortTileN + lane][chunk_base + chunk] =
                    word1.range(static_cast<int>((chunk + 1u) * 32u - 1u), static_cast<int>(chunk * 32u));
        }
    }

    vlm_w8a8_axi_t scale_words0[(kWeightPortTileN * kScalePerTile * 2u + 15u) / 16u];
    vlm_w8a8_axi_t scale_words1[(kWeightPortTileN * kScalePerTile * 2u + 15u) / 16u];
    #pragma HLS ARRAY_PARTITION variable=scale_words0 complete
    #pragma HLS ARRAY_PARTITION variable=scale_words1 complete
    const uint32_t scale_word_count = (kWeightPortTileN * kScalePerTile * 2u + 15u) / 16u;
    const uint64_t scale0_word_base = packed_scale_port_byte_offset(scale0_offset_bytes, k_tile_count, n_tile_idx, k_tile_idx, 0u, 0u) / 16u;
    const uint64_t scale1_word_base = packed_scale_port_byte_offset(scale1_offset_bytes, k_tile_count, n_tile_idx, k_tile_idx, 0u, 0u) / 16u;
    for (uint32_t word_idx = 0; word_idx < scale_word_count; ++word_idx) {
        #pragma HLS PIPELINE II=1
        scale_words0[word_idx] = msu_read128(weight0_arena, scale0_word_base + word_idx);
        scale_words1[word_idx] = msu_read128(weight1_arena, scale1_word_base + word_idx);
    }
    for (uint32_t lane = 0; lane < kWeightPortTileN; ++lane) {
        for (uint32_t sb = 0; sb < kScalePerTile; ++sb) {
            #pragma HLS PIPELINE II=1
            const uint32_t byte_offset = (lane * kScalePerTile + sb) * 2u;
            const uint32_t word_idx = byte_offset / 16u;
            const uint32_t byte_idx = byte_offset % 16u;
            const vlm_w8a8_axi_t word0 = scale_words0[word_idx];
            const vlm_w8a8_axi_t word1 = scale_words1[word_idx];
            weight_scale[lane][sb].e0 = static_cast<scale_exp_t>(static_cast<int8_t>(msu_extract_u8(word0, byte_idx)));
            weight_scale[lane][sb].e1 = static_cast<scale_exp_t>(static_cast<int8_t>(msu_extract_u8(word0, byte_idx + 1u)));
            weight_scale[kWeightPortTileN + lane][sb].e0 = static_cast<scale_exp_t>(static_cast<int8_t>(msu_extract_u8(word1, byte_idx)));
            weight_scale[kWeightPortTileN + lane][sb].e1 = static_cast<scale_exp_t>(static_cast<int8_t>(msu_extract_u8(word1, byte_idx + 1u)));
        }
    }
}
