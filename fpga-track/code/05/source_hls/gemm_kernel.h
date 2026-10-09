#ifndef GEMM_KERNEL_H
#define GEMM_KERNEL_H

#include <cstdint>
#include <ap_int.h>
#include <hls_vector.h>

/*
 * W8A8 GEMM ABI v2.1+wide (2026-04-17, ap_uint<256> port ABI) for the cell_hls kernel.
 *
 * - PS/CPU quantizes F32 activations to Q8 payload + FP16 scales
 * - static weights are prepared in the same Q8 format
 * - PL performs int8 x int8 accumulation and writes FP32 output
 *
 * Port layout (v2.1+wide — outer-N weight to match GGUF block_q8_0 native):
 * - act:           Q8 payload packed as ap_uint<256> beats, [M, K/32]
 * - weight:        Q8 payload packed as ap_uint<256> beats, [N, K/32] (outer N)
 * - act_scales:    FP16 bit patterns, row-major [M, ceil(K / 32)]
 * - weight_scales: FP16 bit patterns, row-major [N, ceil(K / 32)]
 * - out:           FP32 output, row-major [M, N]
 *
 * Rationale:
 * GGUF stores weight tiles outer-N (each row = one output column's K chunks),
 * so the host no longer needs to transpose the weight matrix before DMA.
 *
 * Note:
 * This is a hardware-friendly split-scale Q8 layout. Unlike raw ggml
 * `block_q8_0`, the int8 payload and FP16 scale are exposed as separate ports.
 *
 * ABI evolution (2026-04-15 → 2026-04-17):
 * Path C (2026-04-15) used byte pointers (int8_t*) and relied on
 * `max_widen_bitwidth=256` to hint port widening, but HLS refused to coalesce
 * the 32-wide UNROLL'd byte reads — csynth log showed `[HLS 200-885] limited
 * memory ports` pushing load II 1→32, making the kernel only ~3% faster than
 * CPU on SmolVLM2 ViT prefill.
 *
 * v2.1+wide (2026-04-17) switches act/weight to `ap_uint<256>*` — one beat
 * per read, no UNROLL competing on one port. Step 0A POC confirmed Vitis HLS
 * 2022.1 csynth'd clean with II=1 and 256-bit m_axi RDATA. `ap_uint<256>` has
 * neither of the two A v4 blockers: (1) no clang-LTO failure (the POC linked);
 * (2) no alignas(32) clash with host `std::vector<int8_t>` — the host just
 * needs to ensure the BO base is 32-byte aligned (easy for XRT BOs).
 *
 * Residency buffers inside the kernel still use ACT_VEC/WGT_VEC + AGGREGATE
 * to keep URAM footprint low (A v2.5 learning: 192→32 URAM blocks).
 */

typedef int8_t ACT_DTYPE;
typedef int8_t WGT_DTYPE;
typedef uint16_t SCALE_DTYPE;
typedef int32_t ACC_DTYPE;
typedef float OUT_DTYPE;

static const int GEMM_Q8_GROUP_SIZE = 32;

// Packed Q8 vector types — INTERNAL USE ONLY (kernel residency buffers).
// ACT_VEC/WGT_VEC are used inside the kernel body for residency +
// AGGREGATE compact=bit (URAM footprint reduction from A v2.5).
// Public ABI uses `ap_uint<256>*` instead (see v2.1+wide note above):
// `hls::vector<...>*` on m_axi triggers csynth LTO failure in Vitis HLS
// 2022.1 AND imposes alignas(32) that `std::vector<int8_t>` cannot guarantee.
typedef hls::vector<ACT_DTYPE, GEMM_Q8_GROUP_SIZE> ACT_VEC;
typedef hls::vector<WGT_DTYPE, GEMM_Q8_GROUP_SIZE> WGT_VEC;

static const int GEMM_MAX_M = 1024;
static const int GEMM_MAX_K = 3072;
static const int GEMM_MAX_N = 3072;

// Phase-1 ship shape is VIT ffn_up (M=1024, K=768, N=3072). The HLS on-chip
// residency buffers are sized to GEMM_SHIP_K rather than GEMM_MAX_K so the
// URAM/BRAM budget on KV260 is not oversubscribed at synthesis time. Runtime
// K may be anything in [1, GEMM_SHIP_K]; csim uses K=128 to keep simulation
// fast while the full shipping gate (L2) re-runs at K=GEMM_SHIP_K.
//
// v2.2 尝试 (2026-04-17): SHIP_K 768 → 960 想覆盖 SmolVLM2 LLM K=960，已回退。
// csynth 下 load_act/load_wgt II 从 1 崩到 32（HLS 放弃 256-bit port widening），
// compute loop 被合并成 trip=32767 (unknown) 的单层循环，DSP 683→113 大滑坡。
// 详见 docs/v2.1_hls_kernel_iteration_notes.md 问题 12。回到 768 保 v2.1 可用。
static const int GEMM_SHIP_K = 768;

// Path C tile geometry — conservative baseline. Prior A-path (TILE_M=256,
// TILE_N=128, TM=4, TN=8) passed csim but hit 6 consecutive No-ship
// adversarial reviews on resource/II feasibility. Path C prioritizes
// csynth-pass + full-pipeline walk-through over peak t/s.
static const int TILE_M = 64;
static const int TILE_N = 64;
static const int TILE_K = 32;

// Inner-loop unroll factors used by the kernel body. Path C dropped from
// 4x8x32=1024 MAC/cycle down to 2x2x32=128 MAC/cycle — lets HLS schedule
// honestly and keeps DSP utilization trivially under budget.
static const int TM_UNROLL = 2;
static const int TN_UNROLL = 2;
static const int K_UNROLL  = 32;

// ceil(x/y) (x/y rounding higher)
static inline int gemm_div_up(int x, int y) {
    return (x + y - 1) / y;
}

// num of tiles (@ k dim)
static inline int gemm_k_groups(int k) {
    return gemm_div_up(k, GEMM_Q8_GROUP_SIZE);
}

// num of act
static inline int gemm_act_elems(int m, int k) {
    return m * k;
}

// num of weight
static inline int gemm_weight_elems(int k, int n) {
    return k * n;
}

// num of act_scales
static inline int gemm_act_scale_elems(int m, int k) {
    return m * gemm_k_groups(k);
}

// num of weight_scales
static inline int gemm_weight_scale_elems(int k, int n) {
    return gemm_k_groups(k) * n;
}

// num of out
static inline int gemm_out_elems(int m, int n) {
    return m * n;
}


// Round x up to a multiple of `tile`. Host code is expected to pad M/K/N and
// zero-fill the payload/scale buffers so the HLS kernel can drop all
// inner-loop boundary guards and enable AXI burst + port widening.
static inline int gemm_pad_to_tile(int x, int tile) {
    return ((x + tile - 1) / tile) * tile;
}

// Path C: validity reflects this build's compiled capacity. K is limited to
// GEMM_SHIP_K because the on-chip residency buffers (act_buf, wgt_buf) are
// sized to SHIP_K to respect KV260 URAM/BRAM budget; a runtime K > SHIP_K
// would overflow those buffers. The fix for the adversarial-review #5/#6
// "silent stale output" finding is NOT to widen this check, but to make the
// kernel body explicitly zero `out` when dims are invalid — see the early
// branch in `fpga_gemm_kernel`. Host-side `supports_op` (Task 12) enforces
// the phase-1 shape restriction (1024,768,3072) before this gate.
static inline bool gemm_dims_valid(int m, int k, int n) {
    return m > 0 && k > 0 && n > 0 &&
           m <= GEMM_MAX_M &&
           k <= GEMM_SHIP_K &&
           n <= GEMM_MAX_N &&
           (m % TILE_M == 0) &&
           (k % TILE_K == 0) &&
           (n % TILE_N == 0);
}

// ABI v2.1+wide (2026-04-17):
//   weight is outer-N (row-major [N][K]) to match GGUF block_q8_0 native layout.
//   Host no longer needs to transpose the weight matrix.
//
//   act/weight are 256-bit beat pointers (`ap_uint<256>*`). Each beat packs 32
//   contiguous int8 bytes (one Q8 group). This forces HLS to emit a single
//   256-bit m_axi port per payload, collapsing the prior byte-ABI load II
//   from 32 (root cause: `[HLS 200-885] limited memory ports` under 32-wide
//   byte UNROLL) down to 1.
//
//   Why ap_uint<256> and not hls::vector<int8,32>? A v4 proved
//   `hls::vector<>*` on m_axi triggers clang-LTO failure in Vitis HLS 2022.1
//   AND imposes `alignas(32)` that `std::vector<int8_t>` host buffers cannot
//   guarantee. `ap_uint<256>` has neither blocker — Step 0A POC (2026-04-17)
//   csynth'd clean and achieved II=1.
//
//   Host alignment: the host must write 32-byte-aligned Q8 payload into the
//   XRT BO before launching the kernel (enforced in ggml-xrt pack path).
//   K must be a multiple of GEMM_Q8_GROUP_SIZE (enforced by gemm_dims_valid).
extern "C" void fpga_gemm_kernel(
    ap_uint<256> *act,            // [M][K/32]    row-major, each beat = 32 int8 bytes
    ap_uint<256> *weight,         // [N][K/32]    row-major, each beat = 32 int8 bytes (outer N)
    SCALE_DTYPE  *act_scales,     // [M][K/32]    row-major
    SCALE_DTYPE  *weight_scales,  // [N][K/32]    row-major
    OUT_DTYPE    *out,            // [M][N]       row-major
    int M, int K, int N);

#endif
