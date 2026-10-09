#ifndef VLM_W8A8_ABI_H
#define VLM_W8A8_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI v14: dense/QKV per-output bias offsets. Changes vs v13:
 *   - Added bias_offset_bytes[3] for dense O and fused QKV outputs. Bias
 *     values use the same signed Q22 sign-extended int64 format as v13 MLP
 *     bias. A zero offset means no bias for that output.
 *
 * ABI v13: vlm_engine_v5 vision MLP GELU+bias fused task. Changes vs v12:
 *   - Added GEMM_FUSED_VISION_MLP_GELU_BIAS.
 *   - Added up/down bias offsets for fused non-gated vision MLP. Bias values
 *     are signed Q22 values stored in sign-extended int64 slots in the data
 *     arena.
 *
 * ABI v12: vlm_engine_v5 q22_i40 output format. Changes vs v11:
 *   - v5 dense/QKV/FFN outputs are signed 40-bit Q22 values stored in
 *     sign-extended int64 slots.
 *
 * ABI v11: vlm_engine_v4. Changes vs v10:
 *   - VLM_W8A8_ABI_VERSION 10 -> 11
 *   - Added v4 architecture constants (kSaM=32, kSaN=32, K-tile=128, act cache K=3072)
 *   - Removed vlm_w8a8_profile_t + linear_task.profile_offset_bytes (saves LUT in HLS)
 *   - Unified per-port weight offsets into 2-D arrays
 *     weight_q_offset_bytes_port[OUT][PORT]
 *     weight_scale_offset_bytes_port[OUT][PORT]
 *     gate_*_port[PORT], up_*_port[PORT], down_*_port[PORT]
 *   - Added per-output N stride and per-output row stride for the output tile
 *
 * v3 path keeps reading the engine via the old struct from a separate header
 * (kept as vlm_w8a8_abi_v10.h if needed). New host always emits v11.
 */
#define VLM_W8A8_ABI_VERSION 14u
#define VLM_W8A8_ABI_VERSION_V5 14u
#define VLM_W8A8_ABI_VERSION_V6 15u
#define VLM_W8A8_TASK_MAGIC  0x384d4c56u

#define VLM_W8A8_QK                 32u
#define VLM_W8A8_TEXT_HIDDEN        960u
#define VLM_W8A8_TEXT_KV            320u
#define VLM_W8A8_TEXT_FFN           2560u
#define VLM_W8A8_VISION_HIDDEN      768u
#define VLM_W8A8_VISION_FFN         3072u
#define VLM_W8A8_LINEAR_TILE_N      64u
#define VLM_W8A8_LINEAR_K_BLOCK     128u
#define VLM_W8A8_LINEAR_MAX_OUTPUTS 3u
#define VLM_W8A8_WEIGHT_PORTS       2u
#define VLM_W8A8_LINEAR_GENERIC_MAX_DIM 3072u
#define VLM_W8A8_ATTN_Q_HEADS       15u

/* vlm_engine_v3 architecture constants (kept for v3 build target) */
#define VLM_W8A8_V3_SA_M            29u
#define VLM_W8A8_V3_SA_N            32u
#define VLM_W8A8_V3_TILE_K          128u
#define VLM_W8A8_V3_DECODE_MAC      32u

/* vlm_engine_v4 architecture constants */
#define VLM_W8A8_V4_SA_M            32u
#define VLM_W8A8_V4_SA_N            32u
#define VLM_W8A8_V4_TILE_K          128u
#define VLM_W8A8_V4_ACT_CACHE_K     VLM_W8A8_LINEAR_GENERIC_MAX_DIM

/* vlm_engine_v5 architecture constants: POT scales, one 128-K group per tile. */
#define VLM_W8A8_V5_QK              128u
#define VLM_W8A8_V5_SA_M            32u
#define VLM_W8A8_V5_SA_N            32u
#define VLM_W8A8_V5_TILE_K          128u
#define VLM_W8A8_V5_ACT_CACHE_K     VLM_W8A8_LINEAR_GENERIC_MAX_DIM

/* vlm_engine_v6 architecture constants: decode-friendly N128 PE geometry. */
#define VLM_W8A8_V6_QK              128u
#define VLM_W8A8_V6_SA_M            8u
#define VLM_W8A8_V6_SA_N            128u
#define VLM_W8A8_V6_TILE_K          128u
#define VLM_W8A8_V6_BATCH_M         4u
#define VLM_W8A8_V6_ACT_CACHE_K     VLM_W8A8_LINEAR_GENERIC_MAX_DIM

#define VLM_W8A8_ATTN_KV_HEADS      5u
#define VLM_W8A8_HEAD_DIM           64u
#define VLM_W8A8_PAGE_TOKENS        16u
#define VLM_W8A8_MAX_SEQ_LEN        2048u

enum vlm_w8a8_status_code {
    VLM_W8A8_STATUS_IDLE         = 0u,
    VLM_W8A8_STATUS_DONE         = 1u,
    VLM_W8A8_STATUS_BAD_TASK     = 2u,
    VLM_W8A8_STATUS_BAD_FORMAT   = 3u,
    VLM_W8A8_STATUS_BAD_SHAPE    = 4u,
    VLM_W8A8_STATUS_OUT_OF_RANGE = 5u,
};

enum linear_engine {
    LINEAR_ENGINE_GEMM = 0u,
    LINEAR_ENGINE_GEMV = 1u,
};

enum linear_task_type {
    GEMM_DENSE_O          = 64u,
    GEMM_FUSED_QKV        = 65u,
    GEMM_FUSED_TEXT_FFN   = 66u,
    GEMM_FUSED_VISION_FFN = 67u,
    GEMM_FUSED_VISION_MLP_GELU_BIAS = 68u,
    GEMV_DENSE_O          = 80u,
    GEMV_FUSED_DECODE_QKV = 81u,
    GEMV_FUSED_DECODE_FFN = 82u,
};

enum linear_ffn_activation {
    LINEAR_FFN_ACT_SILU = 0u,
    LINEAR_FFN_ACT_GELU = 1u,
};

enum decode_attention_task_opcode {
    DECODE_ATTENTION_KV_APPEND_FP16     = 17u,
    DECODE_ATTENTION_QK_SOFTMAX_AV_FP16 = 18u,
};

/* Linear task descriptor (v11). 64-byte aligned, 16-byte multiple length. */
typedef struct linear_task_v11 {
    uint32_t magic;
    uint32_t version;
    uint32_t task_type;
    uint32_t rows;                    /* M */
    uint64_t act_q_offset_bytes;
    uint64_t act_scale_offset_bytes;
    uint32_t act_row_stride;
    uint32_t output_count;
    uint32_t input_cols;              /* K */
    uint32_t ffn_activation;
    uint64_t weight_arena_base;
    uint32_t out_cols[VLM_W8A8_LINEAR_MAX_OUTPUTS];   /* N per output */
    uint32_t engine;
    /* Per-output, per-port weight & scale offsets */
    uint64_t weight_q_offset_bytes[VLM_W8A8_LINEAR_MAX_OUTPUTS][VLM_W8A8_WEIGHT_PORTS];
    uint64_t weight_scale_offset_bytes[VLM_W8A8_LINEAR_MAX_OUTPUTS][VLM_W8A8_WEIGHT_PORTS];
    uint64_t dst_offset_bytes[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    uint64_t bias_offset_bytes[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    uint32_t output_row_stride_bytes[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    uint32_t weight_n_tile_stride_bytes;
    uint32_t weight_scale_n_tile_stride_bytes;
    uint32_t act_scale_row_stride;

    /* FFN-specific (gate, up, down) per-port */
    uint64_t gate_q_offset_bytes[VLM_W8A8_WEIGHT_PORTS];
    uint64_t gate_scale_offset_bytes[VLM_W8A8_WEIGHT_PORTS];
    uint64_t up_q_offset_bytes[VLM_W8A8_WEIGHT_PORTS];
    uint64_t up_scale_offset_bytes[VLM_W8A8_WEIGHT_PORTS];
    uint64_t down_q_offset_bytes[VLM_W8A8_WEIGHT_PORTS];
    uint64_t down_scale_offset_bytes[VLM_W8A8_WEIGHT_PORTS];
    uint64_t ffn_dst_offset_bytes;
    uint32_t ffn_input_cols;
    uint32_t ffn_intermediate_cols;
    uint32_t ffn_output_cols;
    uint32_t reserved;
    uint64_t up_bias_offset_bytes;
    uint64_t down_bias_offset_bytes;
    uint64_t reserved1;
} linear_task_t;

/* Backward-compat struct alias: hosts that still emit v10 must fail in HLS
 * via the version field check; we don't carry profile here. */

/* v11: legacy profile shape kept as a host-side no-op so existing runtime
 * profiling glue continues to compile. The v4 engine never reads or writes
 * this region; runtime should leave it zero. Field names match v10. */
typedef struct vlm_w8a8_profile {
    uint64_t cycles_total;
    uint64_t cycles_read_task;
    uint64_t cycles_dense;
    uint64_t cycles_ffn;
    uint64_t cycles_reserved0;
    uint64_t cycles_load_act;
    uint64_t cycles_load_weight;
    uint64_t cycles_compute_a;
    uint64_t cycles_compute_b;
    uint64_t cycles_store;
    uint64_t cycles_ffn_gate;
    uint64_t cycles_ffn_up;
    uint64_t cycles_ffn_quantize;
    uint64_t cycles_ffn_down;
    uint32_t task_count_dense;
    uint32_t task_count_ffn;
    uint32_t task_count_reserved0;
    uint32_t total_rows;
} vlm_w8a8_profile_t;

typedef struct smolvlm_kv_page_meta {
    uint64_t k_data_offset_bytes;
    uint64_t v_data_offset_bytes;
    uint32_t layer_id;
    uint32_t kv_head;
    uint32_t page_idx;
    uint32_t token_count;
} smolvlm_kv_page_meta_t;

typedef struct decode_attention_task {
    uint32_t magic;
    uint32_t version;
    uint32_t opcode;
    uint32_t layer_id;
    uint32_t token_idx;
    uint32_t seq_len;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t q_offset_bytes;
    uint64_t k_offset_bytes;
    uint64_t v_offset_bytes;
    uint64_t kv_meta_offset_bytes;
    uint64_t dst_offset_bytes;
    /* Legacy: kept so v3 decode host helpers compile; v4 engine ignores. */
    uint64_t profile_offset_bytes;
    uint32_t reserved[10];
} decode_attention_task_t;

#ifdef __cplusplus
}
#endif

#endif
