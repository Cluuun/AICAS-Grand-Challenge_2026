// types.hpp — Constants, types, and utility functions for vlm_engine_v3.
#pragma once

#include <ap_int.h>
#include <stdint.h>
#include "../../common/vlm_w8a8_abi_v10.h"

// ============================================================
// Architecture constants
// ============================================================

static constexpr uint32_t kSaM = 29;
static constexpr uint32_t kSaN = 32;
static constexpr uint32_t kGroupSize = VLM_W8A8_QK;  // 32
static constexpr uint32_t kTileK = 128;              // K elements per K-tile (4 groups)
static constexpr uint32_t kGroupsPerKtile = kTileK / kGroupSize;  // 4

static_assert(kSaM == VLM_W8A8_V3_SA_M, "v3 HLS SA_M must match the shared ABI");
static_assert(kSaN == VLM_W8A8_V3_SA_N, "v3 HLS SA_N must match the shared ABI");
static_assert(kTileK == VLM_W8A8_V3_TILE_K, "v3 HLS tile K must match the shared ABI");

static constexpr uint32_t kGenericMaxDim = VLM_W8A8_LINEAR_GENERIC_MAX_DIM;
static constexpr uint32_t kHidden = VLM_W8A8_TEXT_HIDDEN;
static constexpr uint32_t kKv = VLM_W8A8_TEXT_KV;
static constexpr uint32_t kFfn = VLM_W8A8_TEXT_FFN;
static constexpr uint32_t kVisionHidden = VLM_W8A8_VISION_HIDDEN;
static constexpr uint32_t kVisionFfn = VLM_W8A8_VISION_FFN;

// AXI word width
static constexpr uint32_t kAxiWidth = 128;
static constexpr uint32_t kAxiBytes = kAxiWidth / 8;  // 16

// Weight layout per N tile and K tile:
//   each port stores group -> k -> 16 output-channel bytes.
//   HP0 carries cols 0..15, HP1 carries cols 16..31.
static constexpr uint32_t kWeightColsPerPort = kSaN / VLM_W8A8_WEIGHT_PORTS; // 16
static constexpr uint32_t kWeightWordsPerPort = kGroupsPerKtile * kGroupSize; // 128
static constexpr uint32_t kWeightWordsPerKtile = kWeightWordsPerPort * VLM_W8A8_WEIGHT_PORTS; // 256
// Scale layout per K tile: group -> cols. Each port stores 4 groups x 16 cols = 64 bytes = 4 words.
static constexpr uint32_t kScaleWordsPerPort = (kGroupsPerKtile * kWeightColsPerPort + kAxiBytes - 1) / kAxiBytes;
static constexpr uint32_t kScaleWordsPerKtile = kScaleWordsPerPort * VLM_W8A8_WEIGHT_PORTS;
static constexpr uint32_t kActWordsPerKtile = kTileK / kAxiBytes; // 8

// Max dimensions for buffer sizing
static constexpr uint32_t kMaxKtiles = (kGenericMaxDim + kTileK - 1) / kTileK;  // 24
static constexpr uint32_t kMaxNtiles = (kGenericMaxDim + kSaN - 1) / kSaN;      // 96
static constexpr uint32_t kMaxActWordsPerRow = kGenericMaxDim / kAxiBytes;       // 192

// Decode constants
static constexpr uint32_t kDecodeMac = 32;
static constexpr uint32_t kDecodeActBufWords = kGenericMaxDim / kAxiBytes;  // 192
static constexpr uint32_t kFfnColsPerCycle = 8;

// Arena depth hint (words)
static constexpr uint32_t kArenaDepthWords = 256 * 1024;

// ============================================================
// Types
// ============================================================

typedef ap_uint<kAxiWidth> vlm_w8a8_axi_t;
typedef ap_int<8> int8_val_t;
typedef ap_int<32> int32_val_t;
typedef ap_int<8> scale_exp_t;

// Task descriptor (matches linear_task_t in ABI)
typedef linear_task_t accelerator_task_t;

// ============================================================
// Utility functions
// ============================================================

static inline uint32_t align_up(uint32_t val, uint32_t align) {
    return ((val + align - 1) / align) * align;
}

static inline uint32_t div_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

static inline bool task_is_gemm_type(uint32_t t) {
    return t == GEMM_DENSE_O || t == GEMM_FUSED_QKV ||
           t == GEMM_FUSED_TEXT_FFN || t == GEMM_FUSED_VISION_FFN;
}

static inline bool task_is_gemv_type(uint32_t t) {
    return t == GEMV_DENSE_O || t == GEMV_FUSED_DECODE_QKV || t == GEMV_FUSED_DECODE_FFN;
}

static inline bool task_is_dense_or_qkv(uint32_t t) {
    return t == GEMM_DENSE_O || t == GEMM_FUSED_QKV ||
           t == GEMV_DENSE_O || t == GEMV_FUSED_DECODE_QKV;
}

static inline bool task_is_ffn(uint32_t t) {
    return t == GEMM_FUSED_TEXT_FFN || t == GEMM_FUSED_VISION_FFN || t == GEMV_FUSED_DECODE_FFN;
}

// Accessors to abstract ABI field names for cleaner engine code
static inline uint64_t task_act_offset(const accelerator_task_t &t) {
    return t.act_q_offset_bytes;
}
static inline uint64_t task_weight_offset(const accelerator_task_t &t, uint32_t idx) {
    return t.weight_q_offset_bytes[idx];
}
static inline uint64_t task_weight1_offset(const accelerator_task_t &t, uint32_t idx) {
    return t.weight_q_offset_bytes_port1[idx];
}
static inline uint64_t task_out_offset(const accelerator_task_t &t, uint32_t idx) {
    return t.dst_offset_bytes[idx];
}
static inline uint64_t task_scale_offset(const accelerator_task_t &t, uint32_t idx) {
    return t.weight_scale_offset_bytes[idx];
}
