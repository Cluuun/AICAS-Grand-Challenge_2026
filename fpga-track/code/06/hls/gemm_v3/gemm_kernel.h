#pragma once

#include <ap_int.h>
#include <cstdint>

/* ---- Quantization block size (Q8_0 style) ---- */
#define QK8_0 32

/* ---- Maximum problem dimensions ---- */
constexpr int GEMM_MAX_M        = 1024;
constexpr int GEMM_MAX_N        = 3072;
constexpr int GEMM_MAX_K        = 3072;
constexpr int GEMM_MAX_K_BLOCKS = GEMM_MAX_K / QK8_0;

/* ---- M panel: macro-block for full B reuse ---- */
constexpr int GEMM_M_PANEL = 128;

/* ---- Tile / parallelism ---- */
constexpr int GEMM_TILE_M  = 8;
constexpr int GEMM_TILE_N  = 32;
constexpr int GEMM_TILE_K  = 128;
constexpr int GEMM_TILE_KB = GEMM_TILE_K / QK8_0;
constexpr int GEMM_M_PAR   = 2;
constexpr int GEMM_N_PAR   = 16;

constexpr int GEMM_PACK_B  = 16;
constexpr int GEMM_PACK_F  = 4;

constexpr int GEMM_ROW_TILES_PER_PANEL = GEMM_M_PANEL / GEMM_TILE_M;

/* ---- Packed A panel storage ----
 *
 * a_panel stores int8 values packed 16-per-word (ap_uint<128>) instead
 * of one int8 per element.  This lets HLS use the full 72-bit URAM
 * data width (2 URAMs side-by-side form a 144-bit row), instead of
 * 8 bit per URAM word as before.  Result: ~8x fewer URAMs.
 *
 * Each QK8_0 block (32 bytes) occupies 2 consecutive packed words.
 */
constexpr int GEMM_A_PANEL_WORDS = GEMM_MAX_K / GEMM_PACK_B;       /* 192 */
constexpr int GEMM_A_SCALE_WORDS = GEMM_MAX_K_BLOCKS / GEMM_PACK_F; /* 24 */
constexpr int GEMM_WORDS_PER_BLOCK = QK8_0 / GEMM_PACK_B;            /* 2 */

using gemm_a_pack_t   = ap_uint<128>;
using gemm_b_pack_t   = ap_uint<128>;
using gemm_b_axi_t    = ap_uint<128>;
using gemm_f32_pack_t = ap_uint<128>;

static_assert(GEMM_M_PANEL % GEMM_TILE_M == 0, "M_PANEL must be multiple of TILE_M");
static_assert(GEMM_TILE_M  % GEMM_M_PAR == 0,  "TILE_M must be multiple of M_PAR");
static_assert(GEMM_TILE_K  % QK8_0 == 0,       "TILE_K must be multiple of QK8_0");
static_assert(GEMM_TILE_K  % GEMM_PACK_B == 0, "TILE_K must align to packed bytes");
static_assert(GEMM_TILE_N  % GEMM_N_PAR == 0,  "TILE_N must be multiple of N_PAR");
static_assert(GEMM_TILE_KB == 4,               "Helpers assume TILE_KB == 4");
static_assert(GEMM_MAX_K_BLOCKS % GEMM_PACK_F == 0, "MAX_K_BLOCKS must be multiple of PACK_F");

/* ---- A is now read as FP32 and quantized on-chip ----
 * a_f32: row-major FP32 A, packed 4 floats per 128-bit word. The kernel
 * performs row-wise Q8_0-style quantization (per-QK8_0 block amax/127 scale)
 * in load_a_panel and stores INT8 bytes + FP32 scales into URAM/BRAM. The
 * downstream compute path stays byte-identical to the INT8 baseline.
 */
constexpr int GEMM_A_F32_WORDS_MAX = GEMM_MAX_K / GEMM_PACK_F;  /* 768 */

void gemm_kernel(
    const gemm_f32_pack_t* __restrict a_f32,
    const gemm_b_axi_t*    __restrict b_t_0,
    const gemm_b_axi_t*    __restrict b_t_1,
    const gemm_f32_pack_t* __restrict b_t_scales,
    gemm_f32_pack_t*       __restrict c,
    int m,
    int n,
    int k);