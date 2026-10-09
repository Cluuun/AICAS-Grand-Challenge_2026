// accelerator.cpp — vlm_engine_v3 top-level entry point.
// Prefill/Decode dual-engine with shared AXI interface.
// PE array: RTL blackbox (pe_array_32x32.v), HLS controls data movement.
#include "accelerator.hpp"

namespace {
#include "src/types.hpp"
#include "src/axi_io.hpp"
#include "src/prefill_ctrl.hpp"
#include "src/prefill_scale.hpp"
#include "src/prefill_engine.hpp"
#include "src/decode_gemv.hpp"
#include "src/decode_engine.hpp"
#include "src/ffn_fusion.hpp"
} // namespace

static bool run_task_dispatch(
        const vlm_w8a8_axi_t *act_arena,
        vlm_w8a8_axi_t *out_arena,
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        const accelerator_task_t &task,
        vlm_w8a8_profile_t &profile,
        uint32_t &status) {
    #pragma HLS INLINE off

    if (task_is_dense_or_qkv(task.task_type)) {
        if (task.engine == LINEAR_ENGINE_GEMV && task.rows == 1u) {
            run_decode_dense(act_arena, out_arena, weight0_arena, weight1_arena, task, profile);
        } else {
            run_prefill_dense(act_arena, out_arena, weight0_arena, weight1_arena, task, profile);
        }
        profile.task_count_dense += 1u;
        profile.total_rows += task.rows;
        return true;
    }
    if (task_is_ffn(task.task_type)) {
        run_prefill_ffn(act_arena, out_arena, weight0_arena, weight1_arena, task, profile);
        profile.task_count_ffn += 1u;
        profile.total_rows += task.rows;
        return true;
    }
    status = VLM_W8A8_STATUS_BAD_TASK;
    return false;
}

extern "C" void vlm_engine_v3(
        const vlm_w8a8_axi_t *weight0_arena,
        const vlm_w8a8_axi_t *weight1_arena,
        const vlm_w8a8_axi_t *act_arena,
        vlm_w8a8_axi_t *out_arena,
        ap_uint<64> task_offset_bytes,
        ap_uint<64> profile_offset_bytes,
        uint32_t task_count,
        uint32_t &status) {
    #pragma HLS INTERFACE m_axi port=weight0_arena offset=slave bundle=gmem_w0 depth=kArenaDepthWords max_read_burst_length=256 num_read_outstanding=8
    #pragma HLS INTERFACE m_axi port=weight1_arena offset=slave bundle=gmem_w1 depth=kArenaDepthWords max_read_burst_length=256 num_read_outstanding=8
    #pragma HLS INTERFACE m_axi port=act_arena offset=slave bundle=gmem_act depth=kArenaDepthWords max_read_burst_length=256 num_read_outstanding=8
    #pragma HLS INTERFACE m_axi port=out_arena offset=slave bundle=gmem_out depth=kArenaDepthWords max_write_burst_length=256 num_write_outstanding=8
    #pragma HLS INTERFACE s_axilite port=weight0_arena bundle=control
    #pragma HLS INTERFACE s_axilite port=weight1_arena bundle=control
    #pragma HLS INTERFACE s_axilite port=act_arena bundle=control
    #pragma HLS INTERFACE s_axilite port=out_arena bundle=control
    #pragma HLS INTERFACE s_axilite port=task_offset_bytes bundle=control
    #pragma HLS INTERFACE s_axilite port=profile_offset_bytes bundle=control
    #pragma HLS INTERFACE s_axilite port=task_count bundle=control
    #pragma HLS INTERFACE s_axilite port=status bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

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
        // profile.cycles_read_task += 16;  // estimated task load cycles

        if (task.magic != VLM_W8A8_TASK_MAGIC || task.version != VLM_W8A8_ABI_VERSION) {
            status = VLM_W8A8_STATUS_BAD_TASK;
            store_profile(out_arena, profile_offset, profile);
            return;
        }

        if (!run_task_dispatch(act_arena, out_arena, weight0_arena, weight1_arena, task, profile, status)) {
            store_profile(out_arena, profile_offset, profile);
            return;
        }
    }

    store_profile(out_arena, profile_offset, profile);
}
