#pragma once

// Common HLS types shared by every vision_head kernel.
//
// Bring-up: all activations are FP32, all weights are FP32. This is the
// path that compares 1:1 with `host/vision_head_ref.cpp` and the ggml CPU
// build_siglip output.
//
// W8A8 path: activations stored in DDR as FP32 ping-pong buffers (see plan
// section 4), quantized per-row inside the GEMM call site exactly like
// fpga-gemm.c does today. Weights are stored as INT8 + per-row FP32 scale.
//
// Inside the kernels we use ap_fixed only for accumulators that need to
// match the existing INT8 GEMM engine's dynamic range.

#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
    #include "hls_stream.h"
    #include "hls_math.h"
    // ap_int/ap_fixed pull libstdc++ mutex (TLS) — include only when needed.
#  ifndef VH_TYPES_NO_AP
#    include "ap_int.h"
#    include "ap_fixed.h"
#  endif
#else
    #include <cmath>
    #include <cstdint>
#endif

namespace vh {

// ------- model constants pulled from docs/vision_head_plan.md -------
constexpr int IMG_C    = 3;
constexpr int IMG_HW   = 512;
constexpr int PATCH    = 16;
constexpr int N_GRID   = IMG_HW / PATCH;          // 32
// Cosim-only small-slice override (A1-P4b-G4). Production builds (no -D) keep
// the canonical 1024/768/12 shape; must stay in sync with vh_dims.hpp.
#ifdef VH_N_POS_OVERRIDE
constexpr int N_POS    = VH_N_POS_OVERRIDE;
#else
constexpr int N_POS    = N_GRID * N_GRID;         // 1024
#endif
#ifdef VH_N_HEAD_OVERRIDE
constexpr int N_HEAD   = VH_N_HEAD_OVERRIDE;
constexpr int C_EMB    = VH_N_HEAD_OVERRIDE * 64;
#else
constexpr int C_EMB    = 768;
constexpr int N_HEAD   = 12;
#endif
constexpr int N_LAYER  = 12;
constexpr int D_HEAD   = C_EMB / N_HEAD;          // 64
constexpr int F_FFN    = 3072;
constexpr int SHUF_SF  = 4;
constexpr int N_OUT    = (N_GRID / SHUF_SF) * (N_GRID / SHUF_SF);  // 64
constexpr int C_SHUF   = C_EMB * SHUF_SF * SHUF_SF;                // 12288
constexpr int D_OUT    = 960;

// ------- W8A8 numeric types -------
#if defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#  ifndef VH_TYPES_NO_AP
typedef ap_int<8>            w_int8_t;
typedef ap_int<8>            a_int8_t;
typedef ap_int<32>           acc_i32_t;
typedef ap_fixed<16, 4>      a_fix_t;     // post-LN activation
typedef ap_fixed<32, 16>     ln_acc_t;    // LayerNorm running sum
#  endif
#else
typedef int8_t  w_int8_t;
typedef int8_t  a_int8_t;
typedef int32_t acc_i32_t;
typedef float   a_fix_t;
typedef double  ln_acc_t;
#endif

typedef float f32_t;

// ------- helpers --------
inline float clipf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

}  // namespace vh
