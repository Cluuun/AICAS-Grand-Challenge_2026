#include "accelerator.hpp"


namespace {

// HLS keeps these helpers in small header-like modules so each file maps to one
// design concern: constants, profiling, arena IO, tile compute, and fused ops.
#include "src/config.hpp"
#include "src/profile.hpp"
#include "src/io.hpp"
#include "src/compute.hpp"
#include "src/operators.hpp"

} // namespace

static bool validate_dense_task(const accelerator_task_t & task, uint32_t & status) {
    #pragma HLS INLINE off
    const uint32_t input_cols = task.input_cols;
    const uint32_t input_padded = align_up_u32(input_cols, kTileK);
    if ((task.task_type == GEMM_DENSE_O || task.task_type == GEMV_DENSE_O) && task.output_count != 1u) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    if ((task.task_type == GEMM_FUSED_QKV || task.task_type == GEMV_FUSED_DECODE_QKV) &&
        task.output_count != VLM_W8A8_LINEAR_MAX_OUTPUTS) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    if (task.output_count == 0u || task.output_count > VLM_W8A8_LINEAR_MAX_OUTPUTS) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    if (task.rows == 0u ||
        input_cols == 0u ||
        input_cols > kGenericMaxDim ||
        task.act_row_stride < input_padded ||
        (task.act_row_stride % kTileK) != 0u) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    for (uint32_t output_idx = 0; output_idx < VLM_W8A8_LINEAR_MAX_OUTPUTS; ++output_idx) {
        #pragma HLS UNROLL
        const bool active_output = output_idx < task.output_count;
        if (active_output &&
            (task.out_cols[output_idx] == 0u ||
             task.out_cols[output_idx] > kGenericMaxDim ||
             (task.out_cols[output_idx] % 4u) != 0u)) {
            status = VLM_W8A8_STATUS_BAD_SHAPE;
            return false;
        }
    }
    return true;
}

static bool validate_ffn_task(const accelerator_task_t & task, uint32_t & status) {
    #pragma HLS INLINE off
    const uint32_t ffn_input_cols = task.ffn_input_cols;
    const uint32_t ffn_intermediate_cols = task.ffn_intermediate_cols;
    const uint32_t ffn_output_cols = task.ffn_output_cols;
    const uint32_t ffn_input_padded = align_up_u32(ffn_input_cols, kTileK);
    const bool text_ffn_shape =
            ffn_input_cols == kHidden &&
            ffn_intermediate_cols == kFfn &&
            ffn_output_cols == kHidden &&
            task.ffn_activation == LINEAR_FFN_ACT_SILU;
    const bool vision_ffn_shape =
            ffn_input_cols == kVisionHidden &&
            ffn_intermediate_cols == kVisionFfn &&
            ffn_output_cols == kVisionHidden &&
            task.ffn_activation == LINEAR_FFN_ACT_GELU;
    if (task.rows == 0u ||
        ffn_input_cols == 0u ||
        ffn_intermediate_cols == 0u ||
        ffn_output_cols == 0u ||
        ffn_input_cols > kGenericMaxDim ||
        ffn_intermediate_cols > kGenericMaxDim ||
        ffn_output_cols > kGenericMaxDim ||
        (ffn_output_cols % 4u) != 0u ||
        task.act_row_stride < ffn_input_padded ||
        (task.act_row_stride % kTileK) != 0u ||
        !valid_ffn_activation(task.ffn_activation)) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    if ((task.task_type == GEMM_FUSED_TEXT_FFN || task.task_type == GEMV_FUSED_DECODE_FFN) && !text_ffn_shape) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    if (task.task_type == GEMM_FUSED_VISION_FFN && !vision_ffn_shape) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    return true;
}

static bool run_task_scheduler(
        const vlm_w8a8_axi_t * act_arena,
        vlm_w8a8_axi_t * out_arena,
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        const accelerator_task_t & task,
        vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow],
        scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups],
        vlm_w8a8_profile_t & profile,
        uint32_t & status) {
    #pragma HLS INLINE off
    #pragma HLS ALLOCATION function instances=compute_row_group_core limit=1
    #pragma HLS ALLOCATION function instances=compute_tile_generic limit=1
    #pragma HLS ALLOCATION function instances=compute_tile_stage limit=1
    #pragma HLS ALLOCATION function instances=run_dense_output_tile limit=1

    if (task.engine == LINEAR_ENGINE_GEMM && !task_is_gemm_type(task.task_type)) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    if (task.engine == LINEAR_ENGINE_GEMV && (!task_is_gemv_type(task.task_type) || task.rows != 1u)) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return false;
    }
    if (task_is_dense_or_qkv_type(task.task_type)) {
        if (!validate_dense_task(task, status)) {
            return false;
        }
        const uint64_t before_load_act = profile.cycles_load_act;
        const uint64_t before_load_weight = profile.cycles_load_weight;
        const uint64_t before_compute_a = profile.cycles_compute_a;
        const uint64_t before_compute_b = profile.cycles_compute_b;
        const uint64_t before_store = profile.cycles_store;
        run_dense_task(act_arena, out_arena, weight0_arena, weight1_arena, task, act_q, act_scale_2d, profile);
        profile.cycles_dense += (profile.cycles_load_act - before_load_act) +
                (profile.cycles_load_weight - before_load_weight) +
                max_u64(profile.cycles_compute_a - before_compute_a, profile.cycles_compute_b - before_compute_b) +
                (profile.cycles_store - before_store);
        profile.task_count_dense += 1u;
        profile.total_rows += task.rows;
        return true;
    }
    if (task_is_ffn_type(task.task_type)) {
        if (!validate_ffn_task(task, status)) {
            return false;
        }
        const uint64_t before_load_act = profile.cycles_load_act;
        const uint64_t before_load_weight = profile.cycles_load_weight;
        const uint64_t before_compute_a = profile.cycles_compute_a;
        const uint64_t before_compute_b = profile.cycles_compute_b;
        const uint64_t before_store = profile.cycles_store;
        const uint64_t before_quantize = profile.cycles_ffn_quantize;
        run_ffn_task(act_arena, out_arena, weight0_arena, weight1_arena, task, act_q, act_scale_2d, profile);
        profile.cycles_ffn += (profile.cycles_load_act - before_load_act) +
                (profile.cycles_load_weight - before_load_weight) +
                max_u64(profile.cycles_compute_a - before_compute_a, profile.cycles_compute_b - before_compute_b) +
                (profile.cycles_store - before_store) +
                (profile.cycles_ffn_quantize - before_quantize);
        profile.task_count_ffn += 1u;
        profile.total_rows += task.rows;
        return true;
    }
    status = VLM_W8A8_STATUS_BAD_SHAPE;
    return false;
}

extern "C" void vlm_engine_v2(
        const vlm_w8a8_axi_t * weight0_arena,
        const vlm_w8a8_axi_t * weight1_arena,
        const vlm_w8a8_axi_t * act_arena,
        vlm_w8a8_axi_t * out_arena,
        ap_uint<64> task_offset_bytes,
        ap_uint<64> profile_offset_bytes,
        uint32_t task_count,
        uint32_t & status) {
    #pragma HLS INTERFACE m_axi port=weight0_arena offset=slave bundle=gmem_w0 depth=kArenaDepthWords max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=weight1_arena offset=slave bundle=gmem_w1 depth=kArenaDepthWords max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=act_arena offset=slave bundle=gmem_act depth=kArenaDepthWords max_read_burst_length=256 num_read_outstanding=16
    #pragma HLS INTERFACE m_axi port=out_arena offset=slave bundle=gmem_out depth=kArenaDepthWords max_write_burst_length=256 num_write_outstanding=16
    #pragma HLS INTERFACE s_axilite port=weight0_arena bundle=control
    #pragma HLS INTERFACE s_axilite port=weight1_arena bundle=control
    #pragma HLS INTERFACE s_axilite port=act_arena bundle=control
    #pragma HLS INTERFACE s_axilite port=out_arena bundle=control
    #pragma HLS INTERFACE s_axilite port=task_offset_bytes bundle=control
    #pragma HLS INTERFACE s_axilite port=profile_offset_bytes bundle=control
    #pragma HLS INTERFACE s_axilite port=task_count bundle=control
    #pragma HLS INTERFACE s_axilite port=status bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS ALLOCATION function instances=compute_row_group_core limit=1
    #pragma HLS ALLOCATION function instances=compute_tile_generic limit=1
    #pragma HLS ALLOCATION function instances=compute_tile_stage limit=1
    #pragma HLS ALLOCATION function instances=run_dense_output_tile limit=1

    vlm_w8a8_axi_t act_q[kMaxTileM][kMaxActWordsPerRow];
    scale_exp_t act_scale_2d[kMaxTileM][kMaxScaleGroups];
    #pragma HLS BIND_STORAGE variable=act_q type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=act_q complete dim=1
    #pragma HLS ARRAY_PARTITION variable=act_scale_2d complete dim=1
    #pragma HLS ARRAY_PARTITION variable=act_scale_2d cyclic factor=kScalePerTile dim=2

    status = VLM_W8A8_STATUS_DONE;
    const uint64_t task_offset = task_offset_bytes;
    const uint64_t profile_offset = profile_offset_bytes;
    if ((task_offset & 0xfu) != 0u || (profile_offset & 0xfu) != 0u) {
        status = VLM_W8A8_STATUS_BAD_SHAPE;
        return;
    }

    vlm_w8a8_profile_t profile = {};
    for (uint32_t i = 0; i < task_count; ++i) {
        accelerator_task_t task;
        load_task(act_arena, task_offset_bytes + static_cast<uint64_t>(i) * sizeof(accelerator_task_t), &task);
        profile.cycles_read_task += profile_cycles_load_task();
        profile.cycles_total += profile_cycles_load_task();
        if (task.magic != VLM_W8A8_TASK_MAGIC || task.version != VLM_W8A8_ABI_VERSION) {
            status = VLM_W8A8_STATUS_BAD_TASK;
            return;
        }
        if (!valid_engine(task.engine)) {
            status = VLM_W8A8_STATUS_BAD_SHAPE;
            return;
        }
        if (!run_task_scheduler(act_arena, out_arena, weight0_arena, weight1_arena, task, act_q, act_scale_2d, profile, status)) {
            return;
        }
    }

    profile.cycles_total += profile.cycles_load_act +
            profile.cycles_load_weight +
            max_u64(profile.cycles_compute_a, profile.cycles_compute_b) +
            profile.cycles_store +
            profile.cycles_ffn_quantize;
    store_profile(out_arena, profile_offset, profile);
}
