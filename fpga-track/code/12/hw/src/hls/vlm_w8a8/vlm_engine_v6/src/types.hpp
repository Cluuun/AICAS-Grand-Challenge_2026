// types.hpp --- V6 architecture constants. PE 8x128, K-tile 128, BM4 slabs.
#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>
#include "../../common/vlm_w8a8_abi.h"

namespace v6 {

// =============================================================
// PE array geometry (must match pe_array_v6 RTL)
// =============================================================
static constexpr uint32_t kSaM       = VLM_W8A8_V6_SA_M; // 8 rows / slab
static constexpr uint32_t kSaN       = VLM_W8A8_V6_SA_N; // 128 output columns
static constexpr uint32_t kGroupSize = VLM_W8A8_V6_QK;   // 128, scale group along K
static constexpr uint32_t kTileK     = 128;       // K elements per K-tile
static constexpr uint32_t kGroupsPerKtile = kTileK / kGroupSize; // 1
static constexpr uint32_t kNtilesPerKtile = kTileK / kSaN;        // 1 FFN mid N-tile -> one 128-col act K-tile
static constexpr uint32_t kBatchMtiles = VLM_W8A8_V6_BATCH_M;     // fixed BM4 multi-slab schedule

static_assert(kSaM == VLM_W8A8_V6_SA_M,     "V6 SA_M must match ABI");
static_assert(kSaN == VLM_W8A8_V6_SA_N,     "V6 SA_N must match ABI");
static_assert(kTileK == VLM_W8A8_V6_TILE_K, "V6 TILE_K must match ABI");

// =============================================================
// AXI / activation cache sizing
// =============================================================
static constexpr uint32_t kAxiWidth = 128;
static constexpr uint32_t kAxiBytes = kAxiWidth / 8;       // 16
static constexpr uint32_t kAStreamWidth = kSaM * 8;         // A side: 8B/cycle
static constexpr uint32_t kWStreamWidth = kSaN * 8;         // W side: 128B/cycle
static constexpr uint32_t kWHalfStreamWidth = kWStreamWidth / VLM_W8A8_WEIGHT_PORTS; // 64B/cycle
static constexpr uint32_t kAStreamBytes = kAStreamWidth / 8;
static constexpr uint32_t kWStreamBytes = kWStreamWidth / 8;
static constexpr uint32_t kWHalfStreamBytes = kWHalfStreamWidth / 8;
static constexpr uint32_t kPsumWidth = 256;                // quarter row, 8 x int32
static constexpr uint32_t kPsumLanes = kPsumWidth / 32;
static constexpr uint32_t kScaleDrainColLanes = 8;         // scaled cols per cycle
static constexpr uint32_t kFfnFusionColLanes = 2;          // keep HLS DSP outside the 8x128 PE budget
static constexpr uint32_t kFfnRequantBanks = 4;            // dual-port BRAMs feed 8 A8 lanes / cycle

// Activation cache holds K_max bytes per row, kSaM rows.
static constexpr uint32_t kActCacheK = VLM_W8A8_V6_ACT_CACHE_K; // 3072
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

// Weight: per-port carries kSaN/2 = 64 lanes per N-tile.
static constexpr uint32_t kWeightColsPerPort = kSaN / VLM_W8A8_WEIGHT_PORTS; // 64
static constexpr uint32_t kWeightWordsPerKPerPort =
    (kWeightColsPerPort + kAxiBytes - 1) / kAxiBytes;                         // 4
static constexpr uint32_t kWeightWordsPerKtilePerPort =
    (kTileK * kWeightColsPerPort + kAxiBytes - 1) / kAxiBytes;               // 512

// Weight scale: v6 is POT-only and packs one exponent byte per (group, lane).
static constexpr uint32_t kScaleBytesPerSlot   = 1;
static constexpr uint32_t kScaleBytesPerKtilePerPort =
    kGroupsPerKtile * kWeightColsPerPort * kScaleBytesPerSlot;               // 64
static constexpr uint32_t kScaleWordsPerKtilePerPort =
    (kScaleBytesPerKtilePerPort + kAxiBytes - 1) / kAxiBytes;                // 4

// Max compile-time sizes (text path is the largest)
static constexpr uint32_t kMaxKtiles  = (VLM_W8A8_LINEAR_GENERIC_MAX_DIM + kTileK - 1) / kTileK; // 24
static constexpr uint32_t kMaxNtiles  = (VLM_W8A8_LINEAR_GENERIC_MAX_DIM + kSaN - 1) / kSaN;     // 96
static constexpr uint32_t kMaxFfnOutNtiles =
    (VLM_W8A8_TEXT_HIDDEN + kSaN - 1) / kSaN; // text hidden=960 is the largest FFN output

static constexpr uint32_t kPsumBeatsPerRow = kSaN / kPsumLanes;   // 16 (8 int32 / beat)
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
typedef ap_uint<kAStreamWidth> a_stream_word_t; // 64b AXIS beat
typedef ap_uint<kWStreamWidth> w_stream_word_t; // legacy concatenated 1024b W beat
typedef ap_uint<kWHalfStreamWidth> w_half_stream_word_t; // 512b AXIS beat
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

} // namespace v6
