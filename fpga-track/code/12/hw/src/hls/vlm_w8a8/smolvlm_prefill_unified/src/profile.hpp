static uint32_t align_up_u32(uint32_t value, uint32_t align) {
    #pragma HLS INLINE
    return ((value + align - 1u) / align) * align;
}

static uint32_t div_up_u32(uint32_t value, uint32_t div) {
    #pragma HLS INLINE
    return (value + div - 1u) / div;
}

static uint64_t max_u64(uint64_t lhs, uint64_t rhs) {
    #pragma HLS INLINE
    return lhs > rhs ? lhs : rhs;
}

static uint64_t profile_cycles_load_task() {
    #pragma HLS INLINE
    return kTaskTransferWords;
}

static bool valid_engine(uint32_t engine) {
    #pragma HLS INLINE
    return engine == LINEAR_ENGINE_GEMM || engine == LINEAR_ENGINE_GEMV;
}

static bool task_uses_decode_gemv(const accelerator_task_t & task) {
    #pragma HLS INLINE
    return task.engine == LINEAR_ENGINE_GEMV;
}

static bool task_is_gemm_type(uint32_t task_type) {
    #pragma HLS INLINE
    return task_type == GEMM_DENSE_O ||
            task_type == GEMM_FUSED_QKV ||
            task_type == GEMM_FUSED_TEXT_FFN ||
            task_type == GEMM_FUSED_VISION_FFN;
}

static bool task_is_gemv_type(uint32_t task_type) {
    #pragma HLS INLINE
    return task_type == GEMV_DENSE_O ||
            task_type == GEMV_FUSED_DECODE_QKV ||
            task_type == GEMV_FUSED_DECODE_FFN;
}

static bool task_is_dense_or_qkv_type(uint32_t task_type) {
    #pragma HLS INLINE
    return task_type == GEMM_DENSE_O ||
            task_type == GEMM_FUSED_QKV ||
            task_type == GEMV_DENSE_O ||
            task_type == GEMV_FUSED_DECODE_QKV;
}

static bool task_is_ffn_type(uint32_t task_type) {
    #pragma HLS INLINE
    return task_type == GEMM_FUSED_TEXT_FFN ||
            task_type == GEMM_FUSED_VISION_FFN ||
            task_type == GEMV_FUSED_DECODE_FFN;
}

static bool task_is_vision_ffn_type(uint32_t task_type) {
    #pragma HLS INLINE
    return task_type == GEMM_FUSED_VISION_FFN;
}

static bool valid_ffn_activation(uint32_t activation) {
    #pragma HLS INLINE
    return activation == LINEAR_FFN_ACT_SILU || activation == LINEAR_FFN_ACT_GELU;
}

static uint64_t profile_cycles_load_act(uint32_t act_words_per_row, uint32_t act_scale_group_count) {
    #pragma HLS INLINE
    const uint64_t per_row = static_cast<uint64_t>(act_words_per_row) +
            static_cast<uint64_t>(kMaxActScaleWordsPerRow) +
            static_cast<uint64_t>(act_scale_group_count);
    return static_cast<uint64_t>(kMaxTileM) * per_row;
}

static uint64_t profile_cycles_load_weight_tile() {
    #pragma HLS INLINE
    return static_cast<uint64_t>(kTileN) * static_cast<uint64_t>(kWeightChunksPerLane) +
            static_cast<uint64_t>(kTileN) * static_cast<uint64_t>(kScalePerTile) * 2u;
}

static uint64_t profile_cycles_clear_tile_acc(uint32_t rows) {
    #pragma HLS INLINE
    return static_cast<uint64_t>(div_up_u32(rows, kMaxPeM)) * static_cast<uint64_t>(kTileN / 4u) * 2u;
}

static uint64_t profile_cycles_compute_a(uint32_t rows, uint32_t scale_count) {
    #pragma HLS INLINE
    return static_cast<uint64_t>(div_up_u32(rows, kMaxPeM)) *
            static_cast<uint64_t>(kNPePasses) *
            static_cast<uint64_t>(scale_count) *
            static_cast<uint64_t>(kQkSubTileCount);
}

static uint64_t profile_cycles_compute_b(uint32_t rows) {
    #pragma HLS INLINE
    const uint64_t scale_cycles = static_cast<uint64_t>(kPeN / kReducePeN);
    const uint64_t writeback_cycles = static_cast<uint64_t>(kMaxPeM) *
            static_cast<uint64_t>(kPeN / kAccBankN);
    return static_cast<uint64_t>(div_up_u32(rows, kMaxPeM)) *
            static_cast<uint64_t>(kNPePasses) *
            (scale_cycles + writeback_cycles);
}

static uint64_t profile_cycles_compute_elapsed(uint32_t rows, uint32_t scale_count) {
    #pragma HLS INLINE
    return max_u64(profile_cycles_compute_a(rows, scale_count), profile_cycles_compute_b(rows));
}

static uint64_t profile_cycles_dense_elapsed(uint32_t rows, uint32_t k_tile_count, uint32_t scale_count) {
    #pragma HLS INLINE
    if (k_tile_count == 0u) {
        return profile_cycles_clear_tile_acc(rows);
    }
    const uint64_t clear_cycles = profile_cycles_clear_tile_acc(rows);
    const uint64_t load_cycles = profile_cycles_load_weight_tile();
    const uint64_t compute_cycles = profile_cycles_compute_elapsed(rows, scale_count);
    return clear_cycles + load_cycles +
            static_cast<uint64_t>(k_tile_count - 1u) * max_u64(load_cycles, compute_cycles) +
            compute_cycles;
}

static uint64_t profile_cycles_store(uint32_t rows, uint32_t valid_cols) {
    #pragma HLS INLINE
    return static_cast<uint64_t>(rows) * static_cast<uint64_t>(valid_cols / 4u);
}

static uint64_t profile_cycles_quantize_mid(uint32_t rows) {
    #pragma HLS INLINE
    return static_cast<uint64_t>(rows) * static_cast<uint64_t>(kTileN + kTileN + (kTileN / 16u));
}

static uint64_t profile_cycles_copy_mid_q(uint32_t rows, uint32_t words_per_row) {
    #pragma HLS INLINE
    return static_cast<uint64_t>(rows) * static_cast<uint64_t>(words_per_row);
}

static uint64_t profile_cycles_copy_mid_scale(uint32_t rows, uint32_t scale_groups) {
    #pragma HLS INLINE
    return static_cast<uint64_t>(rows) * static_cast<uint64_t>(scale_groups);
}

static uint64_t packed_q_byte_offset(uint64_t base_offset, uint32_t k_tile_count, uint32_t n_tile_idx, uint32_t k_tile_idx, uint32_t lane) {
    #pragma HLS INLINE
    const uint64_t tile_idx = static_cast<uint64_t>(n_tile_idx) * static_cast<uint64_t>(k_tile_count) + k_tile_idx;
    return base_offset + (tile_idx * kTileN + lane) * kTileK;
}

static uint64_t packed_q_port_byte_offset(uint64_t base_offset, uint32_t k_tile_count, uint32_t n_tile_idx, uint32_t k_tile_idx, uint32_t lane) {
    #pragma HLS INLINE
    const uint64_t tile_idx = static_cast<uint64_t>(n_tile_idx) * static_cast<uint64_t>(k_tile_count) + k_tile_idx;
    return base_offset + (tile_idx * kWeightPortTileN + lane) * kTileK;
}

static uint64_t packed_scale_byte_offset(
        uint64_t base_offset,
        uint32_t k_tile_count,
        uint32_t n_tile_idx,
        uint32_t k_tile_idx,
        uint32_t lane,
        uint32_t scale_idx) {
    #pragma HLS INLINE
    const uint64_t tile_idx = static_cast<uint64_t>(n_tile_idx) * static_cast<uint64_t>(k_tile_count) + k_tile_idx;
    return base_offset + (((tile_idx * kTileN + lane) * kScalePerTile) + scale_idx) * 2u;
}

static uint64_t packed_scale_port_byte_offset(
        uint64_t base_offset,
        uint32_t k_tile_count,
        uint32_t n_tile_idx,
        uint32_t k_tile_idx,
        uint32_t lane,
        uint32_t scale_idx) {
    #pragma HLS INLINE
    const uint64_t tile_idx = static_cast<uint64_t>(n_tile_idx) * static_cast<uint64_t>(k_tile_count) + k_tile_idx;
    return base_offset + (((tile_idx * kWeightPortTileN + lane) * kScalePerTile) + scale_idx) * 2u;
}

static void unpack_i8x4_chunk(ap_uint<32> chunk, int8_t out[4]) {
    #pragma HLS INLINE
    out[0] = static_cast<int8_t>(chunk.range(7, 0));
    out[1] = static_cast<int8_t>(chunk.range(15, 8));
    out[2] = static_cast<int8_t>(chunk.range(23, 16));
    out[3] = static_cast<int8_t>(chunk.range(31, 24));
}

static void unpack_i8x8_pair(ap_uint<32> lo, ap_uint<32> hi, int8_t out[8]) {
    #pragma HLS INLINE
    int8_t lo_vals[4];
    int8_t hi_vals[4];
    #pragma HLS ARRAY_PARTITION variable=lo_vals complete
    #pragma HLS ARRAY_PARTITION variable=hi_vals complete
    unpack_i8x4_chunk(lo, lo_vals);
    unpack_i8x4_chunk(hi, hi_vals);
    for (uint32_t i = 0; i < 4u; ++i) {
        #pragma HLS UNROLL
        out[i] = lo_vals[i];
        out[i + 4u] = hi_vals[i];
    }
}
