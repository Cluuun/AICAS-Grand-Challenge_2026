#ifndef VLM_W8A8_ABI_V10_H
#define VLM_W8A8_ABI_V10_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VLM_W8A8_ABI_VERSION 10u
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

/* vlm_engine_v3 architecture constants */
#define VLM_W8A8_V3_SA_M            32u
#define VLM_W8A8_V3_SA_N            32u
#define VLM_W8A8_V3_TILE_K          128u
#define VLM_W8A8_V3_DECODE_MAC      32u

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

typedef struct linear_task {
    uint32_t magic;
    uint32_t version;
    uint32_t task_type;
    uint32_t rows;
    uint64_t act_q_offset_bytes;
    uint64_t act_scale_offset_bytes;
    uint32_t act_row_stride;
    uint32_t output_count;
    uint32_t input_cols;
    uint32_t ffn_activation;
    uint64_t weight_arena_base;
    uint32_t out_cols[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    uint32_t engine;
    /* Port 0 carries the low half of each output-channel tile. */
    uint64_t weight_q_offset_bytes[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    uint64_t weight_scale_offset_bytes[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    /* Port 1 carries the high half of each output-channel tile. */
    uint64_t weight_q_offset_bytes_port1[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    uint64_t weight_scale_offset_bytes_port1[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    uint64_t dst_offset_bytes[VLM_W8A8_LINEAR_MAX_OUTPUTS];
    uint64_t gate_q_offset_bytes;
    uint64_t gate_scale_offset_bytes;
    uint64_t gate_q_offset_bytes_port1;
    uint64_t gate_scale_offset_bytes_port1;
    uint64_t up_q_offset_bytes;
    uint64_t up_scale_offset_bytes;
    uint64_t up_q_offset_bytes_port1;
    uint64_t up_scale_offset_bytes_port1;
    uint64_t down_q_offset_bytes;
    uint64_t down_scale_offset_bytes;
    uint64_t down_q_offset_bytes_port1;
    uint64_t down_scale_offset_bytes_port1;
    uint64_t ffn_dst_offset_bytes;
    uint64_t profile_offset_bytes;
    uint32_t ffn_input_cols;
    uint32_t ffn_intermediate_cols;
    uint32_t ffn_output_cols;
    uint32_t reserved[5];
} linear_task_t;

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
    uint64_t profile_offset_bytes;
    uint32_t reserved[8];
} decode_attention_task_t;

#ifdef __cplusplus
}
#endif

#endif
