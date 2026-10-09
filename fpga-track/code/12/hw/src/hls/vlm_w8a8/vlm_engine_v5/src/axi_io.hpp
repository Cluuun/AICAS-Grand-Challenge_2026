// axi_io.hpp --- task descriptor load + scalar-write helpers for V5.
// Profile path is removed (vs v3) to free LUT.
#pragma once

#include "types.hpp"

#include <cstddef>

namespace v5 {

static inline uint32_t task_read_u32(
        const axi_word_t words[sizeof(task_t) / kAxiBytes],
        uint32_t byte_offset) {
    #pragma HLS INLINE
    const uint32_t word_idx = byte_offset / kAxiBytes;
    const uint32_t bit_idx  = (byte_offset % kAxiBytes) * 8u;
    return (uint32_t)((words[word_idx] >> bit_idx) & 0xffffffffu);
}

static inline uint64_t task_read_u64(
        const axi_word_t words[sizeof(task_t) / kAxiBytes],
        uint32_t byte_offset) {
    #pragma HLS INLINE
    const uint32_t word_idx = byte_offset / kAxiBytes;
    const uint32_t bit_idx  = (byte_offset % kAxiBytes) * 8u;
    return (uint64_t)((words[word_idx] >> bit_idx) & 0xffffffffffffffffULL);
}

// Estimated cycles: ~24 (one beat per 16 bytes of task struct, II=1, +setup)
static void load_task(const axi_word_t *act_arena,
                      uint64_t byte_offset,
                      task_t *out_task) {
    #pragma HLS INLINE off
    const uint32_t word_offset = byte_offset / kAxiBytes;
    const uint32_t task_words  = sizeof(task_t) / kAxiBytes;

    axi_word_t words[sizeof(task_t) / kAxiBytes];
    #pragma HLS ARRAY_PARTITION variable=words complete

    for (uint32_t w = 0; w < task_words; ++w) {
        #pragma HLS PIPELINE II=1
        words[w] = act_arena[word_offset + w];
    }

    out_task->magic                  = task_read_u32(words, offsetof(task_t, magic));
    out_task->version                = task_read_u32(words, offsetof(task_t, version));
    out_task->task_type              = task_read_u32(words, offsetof(task_t, task_type));
    out_task->rows                   = task_read_u32(words, offsetof(task_t, rows));
    out_task->act_q_offset_bytes     = task_read_u64(words, offsetof(task_t, act_q_offset_bytes));
    out_task->act_scale_offset_bytes = task_read_u64(words, offsetof(task_t, act_scale_offset_bytes));
    out_task->act_row_stride         = task_read_u32(words, offsetof(task_t, act_row_stride));
    out_task->output_count           = task_read_u32(words, offsetof(task_t, output_count));
    out_task->input_cols             = task_read_u32(words, offsetof(task_t, input_cols));
    out_task->ffn_activation         = task_read_u32(words, offsetof(task_t, ffn_activation));
    out_task->weight_arena_base      = task_read_u64(words, offsetof(task_t, weight_arena_base));
    out_task->engine                 = task_read_u32(words, offsetof(task_t, engine));
    out_task->weight_n_tile_stride_bytes       = task_read_u32(words, offsetof(task_t, weight_n_tile_stride_bytes));
    out_task->weight_scale_n_tile_stride_bytes = task_read_u32(words, offsetof(task_t, weight_scale_n_tile_stride_bytes));
    out_task->act_scale_row_stride             = task_read_u32(words, offsetof(task_t, act_scale_row_stride));

    for (uint32_t i = 0; i < VLM_W8A8_LINEAR_MAX_OUTPUTS; ++i) {
        #pragma HLS UNROLL
        out_task->out_cols[i] = task_read_u32(words, offsetof(task_t, out_cols) + i * sizeof(uint32_t));
        out_task->dst_offset_bytes[i] = task_read_u64(words, offsetof(task_t, dst_offset_bytes) + i * sizeof(uint64_t));
        out_task->bias_offset_bytes[i] = task_read_u64(words, offsetof(task_t, bias_offset_bytes) + i * sizeof(uint64_t));
        out_task->output_row_stride_bytes[i] =
            task_read_u32(words, offsetof(task_t, output_row_stride_bytes) + i * sizeof(uint32_t));
        for (uint32_t p = 0; p < VLM_W8A8_WEIGHT_PORTS; ++p) {
            #pragma HLS UNROLL
            out_task->weight_q_offset_bytes[i][p] =
                task_read_u64(words, offsetof(task_t, weight_q_offset_bytes)
                                   + (i * VLM_W8A8_WEIGHT_PORTS + p) * sizeof(uint64_t));
            out_task->weight_scale_offset_bytes[i][p] =
                task_read_u64(words, offsetof(task_t, weight_scale_offset_bytes)
                                   + (i * VLM_W8A8_WEIGHT_PORTS + p) * sizeof(uint64_t));
        }
    }

    for (uint32_t p = 0; p < VLM_W8A8_WEIGHT_PORTS; ++p) {
        #pragma HLS UNROLL
        out_task->gate_q_offset_bytes[p] =
            task_read_u64(words, offsetof(task_t, gate_q_offset_bytes) + p * sizeof(uint64_t));
        out_task->gate_scale_offset_bytes[p] =
            task_read_u64(words, offsetof(task_t, gate_scale_offset_bytes) + p * sizeof(uint64_t));
        out_task->up_q_offset_bytes[p] =
            task_read_u64(words, offsetof(task_t, up_q_offset_bytes) + p * sizeof(uint64_t));
        out_task->up_scale_offset_bytes[p] =
            task_read_u64(words, offsetof(task_t, up_scale_offset_bytes) + p * sizeof(uint64_t));
        out_task->down_q_offset_bytes[p] =
            task_read_u64(words, offsetof(task_t, down_q_offset_bytes) + p * sizeof(uint64_t));
        out_task->down_scale_offset_bytes[p] =
            task_read_u64(words, offsetof(task_t, down_scale_offset_bytes) + p * sizeof(uint64_t));
    }

    out_task->ffn_dst_offset_bytes      = task_read_u64(words, offsetof(task_t, ffn_dst_offset_bytes));
    out_task->ffn_input_cols            = task_read_u32(words, offsetof(task_t, ffn_input_cols));
    out_task->ffn_intermediate_cols     = task_read_u32(words, offsetof(task_t, ffn_intermediate_cols));
    out_task->ffn_output_cols           = task_read_u32(words, offsetof(task_t, ffn_output_cols));
    out_task->reserved                  = task_read_u32(words, offsetof(task_t, reserved));
    out_task->up_bias_offset_bytes      = task_read_u64(words, offsetof(task_t, up_bias_offset_bytes));
    out_task->down_bias_offset_bytes    = task_read_u64(words, offsetof(task_t, down_bias_offset_bytes));
    out_task->reserved1                 = task_read_u64(words, offsetof(task_t, reserved1));
}

} // namespace v5
