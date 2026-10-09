// types.hpp --- V5 architecture constants. PE 32x32, K-tile 128, m_tile 32-row.
#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include "../../common/vlm_w8a8_abi.h"

namespace v5 {

// =============================================================
// PE array geometry (must match pe_array_v5 RTL)
// =============================================================
static constexpr uint32_t kSaM       = 32;        // rows = M-tile size
static constexpr uint32_t kSaN       = 32;        // cols = N-tile size
static constexpr uint32_t kGroupSize = VLM_W8A8_V5_QK; // 128, scale group along K
static constexpr uint32_t kTileK     = 128;       // K elements per K-tile
static constexpr uint32_t kGroupsPerKtile = kTileK / kGroupSize; // 1
static constexpr uint32_t kNtilesPerKtile = kTileK / kSaN;        // 4 FFN mid N-tiles -> one 128-col act K-tile
static constexpr uint32_t kBatchMtiles = 4;                       // fixed BM4 multi-slab schedule

static_assert(kSaM == VLM_W8A8_V5_SA_M,    "V5 SA_M must match ABI v11");
static_assert(kSaN == VLM_W8A8_V5_SA_N,    "V5 SA_N must match ABI v11");
static_assert(kTileK == VLM_W8A8_V5_TILE_K, "V5 TILE_K must match ABI v11");

// =============================================================
// AXI / activation cache sizing
// =============================================================
static constexpr uint32_t kAxiWidth = 128;
static constexpr uint32_t kAxiBytes = kAxiWidth / 8;       // 16
static constexpr uint32_t kStreamWidth = 256;              // A/W side: 32B/cycle
static constexpr uint32_t kStreamBytes = kStreamWidth / 8; // 32
static constexpr uint32_t kPsumWidth = 256;                // quarter row, 8 x int32
static constexpr uint32_t kPsumLanes = kPsumWidth / 32;
static constexpr uint32_t kScaleDrainColLanes = 8;         // scaled cols per cycle
static constexpr uint32_t kFfnFusionColLanes = 2;          // keep HLS DSP within 32x32 PE budget
static constexpr uint32_t kFfnRequantBanks = 4;            // dual-port BRAMs feed 8 A8 lanes / cycle

// Activation cache holds K_max bytes per row, kSaM rows.
static constexpr uint32_t kActCacheK = VLM_W8A8_V5_ACT_CACHE_K; // 3072
static constexpr uint32_t kActWordsPerRow = kActCacheK / kAxiBytes; // 192
static constexpr uint32_t kActWordsPerKtile = kTileK / kAxiBytes;   // 8
static constexpr uint32_t kActWords64PerRow = kActCacheK / 8;       // 384
static constexpr uint32_t kActWords64PerKtile = kTileK / 8;         // 16
static constexpr uint32_t kActInputBatchWords64PerRow =
    kBatchMtiles * kActWords64PerRow;                                // 1536
static constexpr uint32_t kActMidBatchWords64PerRow =
    kBatchMtiles * kActWords64PerKtile;                              // 64
static constexpr uint32_t kActFusedBatchWords64PerRow =
    kBatchMtiles * kTileK;                                           // 512, one Q32 value per word
static constexpr uint32_t kActCacheWords64PerRow =
    kActInputBatchWords64PerRow + kActMidBatchWords64PerRow
        + kActFusedBatchWords64PerRow;                               // 2112

// Weight: per-port carries kSaN/2 = 16 lanes per N-tile.
static constexpr uint32_t kWeightColsPerPort = kSaN / VLM_W8A8_WEIGHT_PORTS; // 16
static constexpr uint32_t kWeightWordsPerKtilePerPort =
    (kTileK * kWeightColsPerPort + kAxiBytes - 1) / kAxiBytes;               // 128

// Weight scale: v5 is POT-only and packs one exponent byte per (group, lane).
static constexpr uint32_t kScaleBytesPerSlot   = 1;
static constexpr uint32_t kScaleBytesPerKtilePerPort =
    kGroupsPerKtile * kWeightColsPerPort * kScaleBytesPerSlot;               // 16
static constexpr uint32_t kScaleWordsPerKtilePerPort =
    (kScaleBytesPerKtilePerPort + kAxiBytes - 1) / kAxiBytes;                // 1

// Max compile-time sizes (text path is the largest)
static constexpr uint32_t kMaxKtiles  = (VLM_W8A8_LINEAR_GENERIC_MAX_DIM + kTileK - 1) / kTileK; // 24
static constexpr uint32_t kMaxNtiles  = (VLM_W8A8_LINEAR_GENERIC_MAX_DIM + kSaN - 1) / kSaN;     // 96
static constexpr uint32_t kMaxFfnOutNtiles =
    (VLM_W8A8_TEXT_HIDDEN + kSaN - 1) / kSaN; // text hidden=960 is the largest FFN output

static constexpr uint32_t kPsumBeatsPerRow = kSaN / kPsumLanes;   // 4 (8 int32 / beat)
static constexpr uint32_t kPsumBeatsPerGroup = kSaM * kPsumBeatsPerRow; // 128
static constexpr uint32_t kPsumBeatsPerDrain = kGroupsPerKtile * kPsumBeatsPerGroup; // 128
// Keep the hardware stream contract compact: the final compute beat of each
// 128-K tile carries commit=1, then the RTL PE wrapper autonomously drains the
// tile. There are no standalone drain ctrl beats.
static constexpr uint32_t kComputeFlushBeats = 0;

// Arena depth hint
static constexpr uint32_t kArenaDepthWords = 256u * 1024u;

// =============================================================
// Types
// =============================================================
typedef ap_uint<kAxiWidth>    axi_word_t;     // 128b DDR beat
typedef ap_uint<kStreamWidth> stream_word_t;  // 256b AXIS beat
typedef ap_uint<kPsumWidth>   psum_word_t;    // 256b PSUM beat
typedef ap_uint<kScaleDrainColLanes * 32> psum_half_word_t; // 8 x int32
typedef ap_uint<kScaleDrainColLanes * 8> scale_group_word_t;  // 8 x e0 int8
typedef ap_int<8>             scale_exp_t;
typedef linear_task_t         task_t;

struct ctrl_word_t {
    ap_uint<16> data;
};

// =============================================================
// Helpers
// =============================================================
static inline uint32_t div_ceil(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

static inline bool task_is_dense_or_qkv(uint32_t t) {
    return t == GEMM_DENSE_O || t == GEMM_FUSED_QKV ||
           t == GEMV_DENSE_O || t == GEMV_FUSED_DECODE_QKV;
}
static inline bool task_is_ffn(uint32_t t) {
    return t == GEMM_FUSED_TEXT_FFN || t == GEMM_FUSED_VISION_FFN ||
           t == GEMM_FUSED_VISION_MLP_GELU_BIAS ||
           t == GEMV_FUSED_DECODE_FFN;
}

} // namespace v5
