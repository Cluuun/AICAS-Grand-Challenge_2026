"""
[SHLEE] W4A16 single-row GEMV kernel for batch=1 decode on Ampere.

Computes y = x @ dequant(W)^T  where
    x: [1, K] FP16
    W: packed INT4, layout (N, K/8) INT32 — 8 INT4 values per INT32 (LSB first)
    scales: (N, K/GROUP_SIZE) FP16 — symmetric (zero point implicit = 8)
    y: [1, N] FP16

Design notes
------------
- Each block computes one output row (n = blockIdx.x).
- Within a row, threads cooperatively iterate over K/8 INT32 packed weights.
- INT4 dequant happens in registers (bit-shift + subtract 8 + multiply by scale).
- Vectorized 128-bit loads for x (8 halves at a time).
- FP32 internal accumulation.
- No SMEM for weights — pure register pipeline (K is small enough on 2B).

Matches the fast_gemv.py style: load_inline + thin Python wrapper.

Packing layout
--------------
For a weight row of length K, INT4 quantized values q[0..K-1] (each in [0, 15],
where the FP value is `s * (q - 8)`) are packed as:

    qweight[k/8] = (q[k+0] << 0) | (q[k+1] << 4) | ... | (q[k+7] << 28)
"""

import torch

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

// W4A16 GEMV: y = x @ dequant(W)^T
// x: [K] half, W: [N, K/8] int32, scales: [N, K/GROUP] half, y: [N] half
template <int BLOCK_SIZE, int GROUP_SIZE>
__global__ void gemv_w4_kernel(
    const half*    __restrict__ x,
    const int32_t* __restrict__ W,
    const half*    __restrict__ scales,
    half*          __restrict__ y,
    const int K)
{
    static_assert(GROUP_SIZE % 8 == 0, "group must be a multiple of 8");
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;

    const int K8  = K >> 3;            // number of int32 packed words per row
    const int Kg  = K / GROUP_SIZE;    // number of scale groups per row
    const int INT32_PER_GROUP = GROUP_SIZE >> 3;

    const int4* x_i4 = reinterpret_cast<const int4*>(x);
    const int4* w_i4 = reinterpret_cast<const int4*>(W + (long long)n * K8);
    const half* s_row = scales + (long long)n * Kg;

    float acc = 0.0f;

    // Each iteration consumes 4 int32 packed words = 32 INT4 weights = 32 halves of x
    // (matches an int4 vec load of x).  At GROUP_SIZE=128, that's 4 per group.
    const int K8_4 = K8 >> 2;
    #pragma unroll 1
    for (int i = tid; i < K8_4; i += BLOCK_SIZE) {
        int4 wv = w_i4[i];                         // 4 int32s -> 32 INT4 weights
        int4 xv = x_i4[i];                         // 32 halves of x (16 half2)

        // Determine scale for this chunk: at GROUP_SIZE=128, every 4 int32 share
        // a group (32 halves per int4-vec, group_size=128 ⇒ 4 vecs per group).
        const int vec_in_group = INT32_PER_GROUP >> 2;
        int g_idx = i / vec_in_group;
        float s = __half2float(s_row[g_idx]);

        // Decode 8 INT4s per int32
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t packed = ((uint32_t*)&wv)[j];
            uint32_t xword  = ((uint32_t*)&xv)[2*j];
            uint32_t xword2 = ((uint32_t*)&xv)[2*j + 1];
            half2 xa = *reinterpret_cast<const half2*>(&xword);
            half2 xb = *reinterpret_cast<const half2*>(&xword2);
            // 4 halves in xa,xb pair... actually each int4 vec word = 2 halves (32 bits),
            // so we have 4 halves here.  We need 8 halves for 8 INT4 weights.
            // The two xword slots give us exactly that.
            float2 xfa = __half22float2(xa);
            float2 xfb = __half22float2(xb);
            // Extract 8 INT4 weights from packed
            float w0 = (float)((int)((packed >>  0) & 0xF) - 8);
            float w1 = (float)((int)((packed >>  4) & 0xF) - 8);
            float w2 = (float)((int)((packed >>  8) & 0xF) - 8);
            float w3 = (float)((int)((packed >> 12) & 0xF) - 8);
            float w4 = (float)((int)((packed >> 16) & 0xF) - 8);
            float w5 = (float)((int)((packed >> 20) & 0xF) - 8);
            float w6 = (float)((int)((packed >> 24) & 0xF) - 8);
            float w7 = (float)((int)((packed >> 28) & 0xF) - 8);
            // Multiply by scale into accumulator (kept in fp32, scale applied once at end)
            acc += s * (w0*xfa.x + w1*xfa.y + w2*xfb.x + w3*xfb.y);
            // xword/xword2 only had 4 halves; the upper 4 belong to the NEXT int4 of x.
            // But wv contains 32 INT4 weights, and the int4 vector load of x at index i
            // gave us 8 halves -> we need 32 halves to match.  Loop j handles index 0..3
            // of wv's 4 int32 words.  But xv only has 8 halves.  This means we're short
            // by 24 halves of x.
            // ⇒ This kernel actually needs x_i4 indexed by (4*i + j) to walk the proper
            //   x positions.  Rewriting below.
        }
    }
    // (The above inner loop is incorrect — kept here only for shape sanity.  The
    // production loop is separated below.)

    // Block-wide reduction
    constexpr int NWARPS = BLOCK_SIZE / 32;
    __shared__ float smem[NWARPS];
    const int warp = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < NWARPS) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = (NWARPS >> 1); offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if (lane == 0) y[n] = __float2half(acc);
    }
}

// v5: BATCHED M-token W4 GEMV.  Same weight tile loaded ONCE, computes M outputs
//     per output row.  Used for speculative decoding K-token verify path:
//     y[m, n] = sum_k W[n, k] * x[m, k]    for m in [0, M), n in [0, N)
//
//     Memory pattern (BW-bound regime):
//       - weight load: same as M=1 (loaded once, reused for M output rows)
//       - x load: M× more, but x is small → L2 hit
//       - compute: M× more (negligible for BW-bound shapes)
//     → M=4 verify time ≈ 1.1-1.5× M=1 decode time, NOT 4×.
template <int BLOCK_SIZE, int GROUP_SIZE, int M_BATCH>
__global__ void gemv_w4_kernel_v5_batched(
    const half*    __restrict__ X,   // [M, K] half — M input rows
    const int32_t* __restrict__ W,   // [N, K/8] int32
    const half*    __restrict__ scales,// [N, K/group] half
    half*          __restrict__ Y,   // [M, N] half — M output rows
    const int K,
    const int N)
{
    static_assert(GROUP_SIZE % 32 == 0, "v5 requires GROUP_SIZE multiple of 32");
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;
    if (n >= N) return;

    const int K32 = K >> 5;
    const int VEC_PER_GROUP = GROUP_SIZE >> 5;

    const int4* w_i4_row = reinterpret_cast<const int4*>(W + (long long)n * (K >> 3));
    const half* s_row    = scales + (long long)n * (K / GROUP_SIZE);

    // Per-M accumulator
    float acc[M_BATCH];
    #pragma unroll
    for (int m = 0; m < M_BATCH; m++) acc[m] = 0.0f;

    // Per-M x base pointers
    const int4* x_i4_rows[M_BATCH];
    #pragma unroll
    for (int m = 0; m < M_BATCH; m++)
        x_i4_rows[m] = reinterpret_cast<const int4*>(X + (long long)m * K);

    #pragma unroll 1
    for (int i = tid; i < K32; i += BLOCK_SIZE) {
        // Load weight once (4 int32 = 32 weights), cache-streaming
        int4 wv;
        asm("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
            : "l"(w_i4_row + i));
        float s = __half2float(__ldg(s_row + i / VEC_PER_GROUP));

        // Pre-decode weight to 32 floats (once, reused across M)
        float wf[32];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t packed = ((uint32_t*)&wv)[j];
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                int w = (int)((packed >> (4*b)) & 0xF) - 8;
                wf[j*8 + b] = (float)w;
            }
        }

        // For each M input row: load 32 x's + FMA
        #pragma unroll
        for (int m = 0; m < M_BATCH; m++) {
            int4 xv0 = x_i4_rows[m][(i << 2) + 0];
            int4 xv1 = x_i4_rows[m][(i << 2) + 1];
            int4 xv2 = x_i4_rows[m][(i << 2) + 2];
            int4 xv3 = x_i4_rows[m][(i << 2) + 3];
            float xf[32];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int4 xvj = (j==0)?xv0:(j==1)?xv1:(j==2)?xv2:xv3;
                half2 ha = *reinterpret_cast<const half2*>(&xvj.x);
                half2 hb = *reinterpret_cast<const half2*>(&xvj.y);
                half2 hc = *reinterpret_cast<const half2*>(&xvj.z);
                half2 hd = *reinterpret_cast<const half2*>(&xvj.w);
                float2 fa=__half22float2(ha), fb=__half22float2(hb);
                float2 fc=__half22float2(hc), fd=__half22float2(hd);
                xf[j*8+0]=fa.x; xf[j*8+1]=fa.y; xf[j*8+2]=fb.x; xf[j*8+3]=fb.y;
                xf[j*8+4]=fc.x; xf[j*8+5]=fc.y; xf[j*8+6]=fd.x; xf[j*8+7]=fd.y;
            }
            float partial = 0.0f;
            #pragma unroll
            for (int k = 0; k < 32; k++) partial += wf[k] * xf[k];
            acc[m] += s * partial;
        }
    }

    // Per-M reduction
    constexpr int NWARPS = BLOCK_SIZE / 32;
    __shared__ float smem[NWARPS * M_BATCH];
    const int warp = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int m = 0; m < M_BATCH; m++) {
        float a = acc[m];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        if (lane == 0) smem[warp * M_BATCH + m] = a;
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int m = 0; m < M_BATCH; m++) {
            float a = (lane < NWARPS) ? smem[lane * M_BATCH + m] : 0.0f;
            #pragma unroll
            for (int offset = (NWARPS >> 1); offset > 0; offset >>= 1)
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            if (lane == 0) Y[(long long)m * N + n] = __float2half(a);
        }
    }
}

// v3: v2 + PTX cache hints + int4 vector load for weight (4 int32 = 32 weights per iter).
// On A100/PPU, ld.global.cs (cache-streaming) avoids L1 pollution for weight,
// L2::128B prefetch hint for scale.  Less per-thread loop iterations → less ILP gaps.
template <int BLOCK_SIZE, int GROUP_SIZE>
__global__ void gemv_w4_kernel_v3(
    const half*    __restrict__ x,
    const int32_t* __restrict__ W,
    const half*    __restrict__ scales,
    half*          __restrict__ y,
    const int K)
{
    static_assert(GROUP_SIZE % 32 == 0, "v3 requires GROUP_SIZE multiple of 32 (for int4 weight vec)");
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;

    const int K32 = K >> 5;                 // # of int4-vec weight loads per row (each = 4 int32 = 32 weights)
    const int VEC_PER_GROUP = GROUP_SIZE >> 5;   // # of int4-vec per group

    const int4* x_i4   = reinterpret_cast<const int4*>(x);
    const int4* w_i4_row = reinterpret_cast<const int4*>(W + (long long)n * (K >> 3));
    const half* s_row  = scales + (long long)n * (K / GROUP_SIZE);

    float acc = 0.0f;

    #pragma unroll 1
    for (int i = tid; i < K32; i += BLOCK_SIZE) {
        // PTX cache-streaming load of 4 int32 (= 32 weights packed)
        int4 wv;
        asm("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
            : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
            : "l"(w_i4_row + i));

        // 32 halves of x (4 int4 loads)
        int4 xv0 = x_i4[(i << 2) + 0];
        int4 xv1 = x_i4[(i << 2) + 1];
        int4 xv2 = x_i4[(i << 2) + 2];
        int4 xv3 = x_i4[(i << 2) + 3];

        // Scale per group (32-weight int4-vec aligned with group_size multiple of 32)
        int g_idx = i / VEC_PER_GROUP;
        float s = __half2float(__ldg(s_row + g_idx));

        // Process 4 packed int32 → 32 weights → 32 x's
        float partial = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t packed = ((uint32_t*)&wv)[j];
            int4 xvj = (j==0)?xv0:(j==1)?xv1:(j==2)?xv2:xv3;
            half2 xa = *reinterpret_cast<const half2*>(&xvj.x);
            half2 xb = *reinterpret_cast<const half2*>(&xvj.y);
            half2 xc = *reinterpret_cast<const half2*>(&xvj.z);
            half2 xd = *reinterpret_cast<const half2*>(&xvj.w);
            float2 fa = __half22float2(xa);
            float2 fb = __half22float2(xb);
            float2 fc = __half22float2(xc);
            float2 fd = __half22float2(xd);
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                int w = (int)((packed >> (4*b)) & 0xF) - 8;
                float wf = (float)w;
                float xf = (b==0)?fa.x:(b==1)?fa.y:(b==2)?fb.x:(b==3)?fb.y:
                           (b==4)?fc.x:(b==5)?fc.y:(b==6)?fd.x:fd.y;
                partial += wf * xf;
            }
        }
        acc += s * partial;
    }

    constexpr int NWARPS = BLOCK_SIZE / 32;
    __shared__ float smem[NWARPS];
    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < NWARPS) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = (NWARPS >> 1); offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if (lane == 0) y[n] = __float2half(acc);
    }
}

// v4: multi-row kernel.  ROWS_PER_BLOCK output rows per CUDA block, sharing x loads.
//    - Reduces total block count (less SM scheduling overhead)
//    - x loaded ONCE per iter, reused for ROWS rows
//    - Per-row weight load uses cache-streaming (ld.global.cs)
template <int BLOCK_SIZE, int GROUP_SIZE, int ROWS_PER_BLOCK>
__global__ void gemv_w4_kernel_v4(
    const half*    __restrict__ x,
    const int32_t* __restrict__ W,
    const half*    __restrict__ scales,
    half*          __restrict__ y,
    const int K,
    const int N)
{
    static_assert(GROUP_SIZE % 32 == 0, "v4 requires GROUP_SIZE multiple of 32");
    const int n_base = blockIdx.x * ROWS_PER_BLOCK;
    const int tid    = threadIdx.x;
    if (n_base >= N) return;

    const int K32 = K >> 5;                  // # of 32-weight (=4-int32) groups
    const int Kg  = K / GROUP_SIZE;
    const int VEC_PER_GROUP = GROUP_SIZE >> 5;

    const int4* x_i4 = reinterpret_cast<const int4*>(x);

    float acc[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) acc[r] = 0.0f;

    // Per-row pointers
    const int4* w_i4_rows[ROWS_PER_BLOCK];
    const half* s_rows[ROWS_PER_BLOCK];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        int n = n_base + r;
        if (n < N) {
            w_i4_rows[r] = reinterpret_cast<const int4*>(W + (long long)n * (K >> 3));
            s_rows[r]    = scales + (long long)n * Kg;
        }
    }

    #pragma unroll 1
    for (int i = tid; i < K32; i += BLOCK_SIZE) {
        // Load x once (32 halves)
        int4 xv0 = x_i4[(i << 2) + 0];
        int4 xv1 = x_i4[(i << 2) + 1];
        int4 xv2 = x_i4[(i << 2) + 2];
        int4 xv3 = x_i4[(i << 2) + 3];
        // Unpack to float (reusable across rows)
        half2 xa0 = *reinterpret_cast<const half2*>(&xv0.x);
        half2 xa1 = *reinterpret_cast<const half2*>(&xv0.y);
        half2 xa2 = *reinterpret_cast<const half2*>(&xv0.z);
        half2 xa3 = *reinterpret_cast<const half2*>(&xv0.w);
        half2 xb0 = *reinterpret_cast<const half2*>(&xv1.x);
        half2 xb1 = *reinterpret_cast<const half2*>(&xv1.y);
        half2 xb2 = *reinterpret_cast<const half2*>(&xv1.z);
        half2 xb3 = *reinterpret_cast<const half2*>(&xv1.w);
        half2 xc0 = *reinterpret_cast<const half2*>(&xv2.x);
        half2 xc1 = *reinterpret_cast<const half2*>(&xv2.y);
        half2 xc2 = *reinterpret_cast<const half2*>(&xv2.z);
        half2 xc3 = *reinterpret_cast<const half2*>(&xv2.w);
        half2 xd0 = *reinterpret_cast<const half2*>(&xv3.x);
        half2 xd1 = *reinterpret_cast<const half2*>(&xv3.y);
        half2 xd2 = *reinterpret_cast<const half2*>(&xv3.z);
        half2 xd3 = *reinterpret_cast<const half2*>(&xv3.w);
        float2 fa0=__half22float2(xa0), fa1=__half22float2(xa1), fa2=__half22float2(xa2), fa3=__half22float2(xa3);
        float2 fb0=__half22float2(xb0), fb1=__half22float2(xb1), fb2=__half22float2(xb2), fb3=__half22float2(xb3);
        float2 fc0=__half22float2(xc0), fc1=__half22float2(xc1), fc2=__half22float2(xc2), fc3=__half22float2(xc3);
        float2 fd0=__half22float2(xd0), fd1=__half22float2(xd1), fd2=__half22float2(xd2), fd3=__half22float2(xd3);
        float xf[32] = {
            fa0.x, fa0.y, fa1.x, fa1.y, fa2.x, fa2.y, fa3.x, fa3.y,
            fb0.x, fb0.y, fb1.x, fb1.y, fb2.x, fb2.y, fb3.x, fb3.y,
            fc0.x, fc0.y, fc1.x, fc1.y, fc2.x, fc2.y, fc3.x, fc3.y,
            fd0.x, fd0.y, fd1.x, fd1.y, fd2.x, fd2.y, fd3.x, fd3.y,
        };

        int g_idx = i / VEC_PER_GROUP;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            int n = n_base + r;
            if (n >= N) continue;
            // Load weight (4 int32 = 32 weights) with cache-streaming hint
            int4 wv;
            asm("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                : "l"(w_i4_rows[r] + i));
            float s = __half2float(__ldg(s_rows[r] + g_idx));

            float partial = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t packed = ((uint32_t*)&wv)[j];
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    int w = (int)((packed >> (4*b)) & 0xF) - 8;
                    partial += (float)w * xf[j*8 + b];
                }
            }
            acc[r] += s * partial;
        }
    }

    // Per-row reduction
    constexpr int NWARPS = BLOCK_SIZE / 32;
    __shared__ float smem[NWARPS * ROWS_PER_BLOCK];
    const int warp = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; r++) {
        float a = acc[r];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        if (lane == 0) smem[warp * ROWS_PER_BLOCK + r] = a;
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; r++) {
            float a = (lane < NWARPS) ? smem[lane * ROWS_PER_BLOCK + r] : 0.0f;
            #pragma unroll
            for (int offset = (NWARPS >> 1); offset > 0; offset >>= 1)
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            int n = n_base + r;
            if (lane == 0 && n < N) y[n] = __float2half(a);
        }
    }
}

// Corrected straightforward kernel: iterate one int32 packed word at a time;
// load matching 8 halves of x with one int4 vector load.
template <int BLOCK_SIZE, int GROUP_SIZE>
__global__ void gemv_w4_kernel_v2(
    const half*    __restrict__ x,
    const int32_t* __restrict__ W,
    const half*    __restrict__ scales,
    half*          __restrict__ y,
    const int K)
{
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;

    const int K8  = K >> 3;            // # of int32 packed words per row
    const int Kg  = K / GROUP_SIZE;    // # of scale groups per row
    const int VEC_PER_GROUP = GROUP_SIZE >> 3;

    const int4* x_i4 = reinterpret_cast<const int4*>(x);     // each int4 = 8 halves
    const int32_t* w_row = W + (long long)n * K8;
    const half*    s_row = scales + (long long)n * Kg;

    float acc = 0.0f;

    #pragma unroll 2
    for (int i = tid; i < K8; i += BLOCK_SIZE) {
        // Scale for this 8-weight chunk
        int g_idx = i / VEC_PER_GROUP;
        float s = __half2float(s_row[g_idx]);

        // 8 INT4 weights packed into one int32
        uint32_t packed = (uint32_t)w_row[i];

        // 8 halves of x (one int4 vec load)
        int4 xv = x_i4[i];
        half2 x0 = *reinterpret_cast<const half2*>(&xv.x);
        half2 x1 = *reinterpret_cast<const half2*>(&xv.y);
        half2 x2 = *reinterpret_cast<const half2*>(&xv.z);
        half2 x3 = *reinterpret_cast<const half2*>(&xv.w);
        float2 xf0 = __half22float2(x0);
        float2 xf1 = __half22float2(x1);
        float2 xf2 = __half22float2(x2);
        float2 xf3 = __half22float2(x3);

        float w0 = (float)((int)((packed >>  0) & 0xF) - 8);
        float w1 = (float)((int)((packed >>  4) & 0xF) - 8);
        float w2 = (float)((int)((packed >>  8) & 0xF) - 8);
        float w3 = (float)((int)((packed >> 12) & 0xF) - 8);
        float w4 = (float)((int)((packed >> 16) & 0xF) - 8);
        float w5 = (float)((int)((packed >> 20) & 0xF) - 8);
        float w6 = (float)((int)((packed >> 24) & 0xF) - 8);
        float w7 = (float)((int)((packed >> 28) & 0xF) - 8);

        acc += s * (w0*xf0.x + w1*xf0.y + w2*xf1.x + w3*xf1.y
                  + w4*xf2.x + w5*xf2.y + w6*xf3.x + w7*xf3.y);
    }

    // Block-wide reduction
    constexpr int NWARPS = BLOCK_SIZE / 32;
    __shared__ float smem[NWARPS];
    const int warp = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < NWARPS) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = (NWARPS >> 1); offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if (lane == 0) y[n] = __float2half(acc);
    }
}

// Out-of-place dispatcher (allocates output)
torch::Tensor gemv_w4_forward(
    torch::Tensor x,        // [K] or [1,K] half
    torch::Tensor qweight,  // [N, K/8] int32
    torch::Tensor scales,   // [N, K/group] half
    int64_t group_size)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf, "x must be CUDA fp16");
    TORCH_CHECK(qweight.is_cuda() && qweight.scalar_type() == torch::kInt, "qweight must be CUDA int32");
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kHalf, "scales must be CUDA fp16");

    auto xc = x.contiguous();
    auto wc = qweight.contiguous();
    auto sc = scales.contiguous();
    const int N = wc.size(0);
    const int K8 = wc.size(1);
    const int K = K8 * 8;
    TORCH_CHECK(xc.numel() == K, "x length must equal K=qweight.size(1)*8");

    auto y = torch::empty({1, N}, xc.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // PPU 64-SM tuning (measured BLOCK sweep, lower=faster):
    //   o/qkv/gate_up shapes (K=2048):  B=32 wins
    //   down shape         (K=6144):    B=64 wins (longer K → more work/thread)
    //   lm_head            (N≥65536):   B=128 preserves reduction tree for
    //                                   argmax accuracy (B<128 drifts logits)
    int BLOCK;
    if (N >= 65536) BLOCK = 128;      // lm_head — accuracy
    else if (K >= 4096) BLOCK = 64;   // down_proj — bigger K
    else BLOCK = 32;                   // qkv / o / gate_up — small K, max occupancy
    dim3 grid(N);

    #define LAUNCH_B(B, GS) \
        gemv_w4_kernel_v2<B, GS><<<grid, dim3(B), 0, stream>>>( \
            reinterpret_cast<const half*>(xc.data_ptr<at::Half>()), \
            wc.data_ptr<int32_t>(), \
            reinterpret_cast<const half*>(sc.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(y.data_ptr<at::Half>()), \
            K);
    #define DISPATCH_GS(B) \
        do { \
            if      (group_size == 128) LAUNCH_B(B, 128) \
            else if (group_size == 64)  LAUNCH_B(B, 64)  \
            else if (group_size == 32)  LAUNCH_B(B, 32)  \
            else TORCH_CHECK(false, "Unsupported group_size: ", group_size); \
        } while(0)

    if (BLOCK == 32)       DISPATCH_GS(32);
    else if (BLOCK == 64)  DISPATCH_GS(64);
    else                   DISPATCH_GS(128);
    #undef LAUNCH_B
    #undef DISPATCH_GS
    return y;
}

// In-place dispatcher (writes into preallocated `out` — CUDA-graph safe).
void gemv_w4_forward_out(
    torch::Tensor x,         // [K] half
    torch::Tensor qweight,   // [N, K/8] int32
    torch::Tensor scales,    // [N, K/group] half
    torch::Tensor out,       // [1, N] half (writable, preallocated)
    int64_t group_size)
{
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf, "x must be CUDA fp16");
    TORCH_CHECK(qweight.is_cuda() && qweight.scalar_type() == torch::kInt, "qweight must be CUDA int32");
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kHalf, "scales must be CUDA fp16");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kHalf, "out must be CUDA fp16");

    const int N  = qweight.size(0);
    const int K8 = qweight.size(1);
    const int K  = K8 * 8;
    TORCH_CHECK(x.numel() == K, "x length must equal K=qweight.size(1)*8");
    TORCH_CHECK(out.numel() >= N, "out must have at least N elements");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // PPU 64-SM tuning (measured BLOCK sweep) — see gemv_w4_forward for table.
    int BLOCK;
    if (N >= 65536) BLOCK = 128;      // lm_head — accuracy
    else if (K >= 4096) BLOCK = 64;   // down_proj — bigger K
    else BLOCK = 32;                   // qkv / o / gate_up — max occupancy
    dim3 grid(N);

    #define LAUNCH_OUT_B(B, GS) \
        gemv_w4_kernel_v2<B, GS><<<grid, dim3(B), 0, stream>>>( \
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()), \
            qweight.data_ptr<int32_t>(), \
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), \
            K);
    #define DISPATCH_OUT_GS(B) \
        do { \
            if      (group_size == 128) LAUNCH_OUT_B(B, 128) \
            else if (group_size == 64)  LAUNCH_OUT_B(B, 64)  \
            else if (group_size == 32)  LAUNCH_OUT_B(B, 32)  \
            else TORCH_CHECK(false, "Unsupported group_size: ", group_size); \
        } while(0)

    if (BLOCK == 32)       DISPATCH_OUT_GS(32);
    else if (BLOCK == 64)  DISPATCH_OUT_GS(64);
    else                   DISPATCH_OUT_GS(128);
    #undef LAUNCH_OUT_B
    #undef DISPATCH_OUT_GS
}

// v5 batched: K-token verify with single weight load
void gemv_w4_v5_batched_forward_out(
    torch::Tensor X,         // [M, K] half
    torch::Tensor qweight,   // [N, K/8] int32
    torch::Tensor scales,    // [N, K/group] half
    torch::Tensor Y,         // [M, N] half
    int64_t group_size,
    int64_t m_batch,
    int64_t block_size)
{
    const int N  = qweight.size(0);
    const int K8 = qweight.size(1);
    const int K  = K8 * 8;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int BLOCK = (int)block_size;
    if (BLOCK == 0) {
        if (N >= 65536) BLOCK = 128;
        else if (K >= 4096) BLOCK = 64;
        else BLOCK = 32;
    }
    int M = (int)m_batch;
    dim3 grid(N);

    #define LV5(B, GS, MM) \
        gemv_w4_kernel_v5_batched<B, GS, MM><<<grid, dim3(B), 0, stream>>>( \
            reinterpret_cast<const half*>(X.data_ptr<at::Half>()), \
            qweight.data_ptr<int32_t>(), \
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(Y.data_ptr<at::Half>()), K, N);
    #define DISPATCH_M(B, GS) \
        do { if (M == 1) LV5(B, GS, 1) \
             else if (M == 2) LV5(B, GS, 2) \
             else if (M == 4) LV5(B, GS, 4) \
             else if (M == 8) LV5(B, GS, 8) \
             else TORCH_CHECK(false, "M must be 1,2,4,8"); } while(0)
    #define DISPATCH_BGS(B) \
        do { if (group_size==128) DISPATCH_M(B, 128); \
             else if (group_size==64) DISPATCH_M(B, 64); \
             else DISPATCH_M(B, 32); } while(0)

    if (BLOCK == 32)       DISPATCH_BGS(32);
    else if (BLOCK == 64)  DISPATCH_BGS(64);
    else                   DISPATCH_BGS(128);
    #undef LV5
    #undef DISPATCH_M
    #undef DISPATCH_BGS
}

// v4 dispatcher — multirow with configurable ROWS_PER_BLOCK (1, 2, 4, 8)
void gemv_w4_v4_forward_out(
    torch::Tensor x, torch::Tensor qweight, torch::Tensor scales,
    torch::Tensor out, int64_t group_size, int64_t rows_per_block,
    int64_t block_size)
{
    const int N  = qweight.size(0);
    const int K8 = qweight.size(1);
    const int K  = K8 * 8;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int BLOCK = (int)block_size;
    if (BLOCK == 0) {
        if (N >= 65536) BLOCK = 128;
        else if (K >= 4096) BLOCK = 64;
        else BLOCK = 32;
    }
    int ROWS = (int)rows_per_block;
    if (ROWS <= 0) ROWS = 1;
    int n_blocks = (N + ROWS - 1) / ROWS;
    dim3 grid(n_blocks);

    #define LV4(B, GS, R) \
        gemv_w4_kernel_v4<B, GS, R><<<grid, dim3(B), 0, stream>>>( \
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()), \
            qweight.data_ptr<int32_t>(), \
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), K, N);

    #define DISPATCH_R(B, GS) \
        do { if (ROWS == 1) LV4(B, GS, 1) \
             else if (ROWS == 2) LV4(B, GS, 2) \
             else if (ROWS == 4) LV4(B, GS, 4) \
             else if (ROWS == 8) LV4(B, GS, 8) \
             else if (ROWS == 16) LV4(B, GS, 16) \
             else TORCH_CHECK(false, "ROWS must be 1,2,4,8,16"); } while(0)

    #define DISPATCH_BGS(B) \
        do { if (group_size==128) DISPATCH_R(B, 128); \
             else if (group_size==64) DISPATCH_R(B, 64); \
             else DISPATCH_R(B, 32); } while(0)

    if (BLOCK == 32)       DISPATCH_BGS(32);
    else if (BLOCK == 64)  DISPATCH_BGS(64);
    else                   DISPATCH_BGS(128);
    #undef LV4
    #undef DISPATCH_R
    #undef DISPATCH_BGS
}

// v3 dispatcher — uses gemv_w4_kernel_v3 (cache hints + 32-weight chunk)
void gemv_w4_v3_forward_out(
    torch::Tensor x, torch::Tensor qweight, torch::Tensor scales,
    torch::Tensor out, int64_t group_size)
{
    const int N  = qweight.size(0);
    const int K8 = qweight.size(1);
    const int K  = K8 * 8;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int BLOCK;
    if (N >= 65536) BLOCK = 128;
    else if (K >= 4096) BLOCK = 64;
    else BLOCK = 32;
    dim3 grid(N);
    #define LV3(B, GS) \
        gemv_w4_kernel_v3<B, GS><<<grid, dim3(B), 0, stream>>>( \
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()), \
            qweight.data_ptr<int32_t>(), \
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), K);
    #define DV3(B) \
        do { if (group_size==128) LV3(B,128) \
             else if (group_size==64) LV3(B,64) \
             else LV3(B,32); } while(0)
    if (BLOCK == 32)       DV3(32);
    else if (BLOCK == 64)  DV3(64);
    else                   DV3(128);
    #undef LV3
    #undef DV3
}

// =====================================================================
// silu_mul + W4 GEMV  fusion.
// gate_up: [2*K] fp16 — first K is `gate`, second K is `up`.
// Computes y[n] = sum_k W[n,k] * (silu(gate[k]) * up[k]) for n in [0,N).
// =====================================================================
template <int BLOCK_SIZE, int GROUP_SIZE>
__global__ void silu_mul_gemv_w4_kernel(
    const half*    __restrict__ gate_up,    // [2*K] half
    const int32_t* __restrict__ W,           // [N, K/8] int32
    const half*    __restrict__ scales,     // [N, K/group] half
    half*          __restrict__ y,           // [N] half
    const int K)
{
    const int n   = blockIdx.x;
    const int tid = threadIdx.x;

    const int K8  = K >> 3;
    const int Kg  = K / GROUP_SIZE;
    const int VEC_PER_GROUP = GROUP_SIZE >> 3;

    const int4* g_i4 = reinterpret_cast<const int4*>(gate_up);
    const int4* u_i4 = reinterpret_cast<const int4*>(gate_up + K);
    const int32_t* w_row = W + (long long)n * K8;
    const half*    s_row = scales + (long long)n * Kg;

    float acc = 0.0f;

    #pragma unroll 2
    for (int i = tid; i < K8; i += BLOCK_SIZE) {
        int g_idx = i / VEC_PER_GROUP;
        float s = __half2float(s_row[g_idx]);

        uint32_t packed = (uint32_t)w_row[i];

        int4 gv = g_i4[i];
        int4 uv = u_i4[i];
        half2 g0 = *reinterpret_cast<const half2*>(&gv.x);
        half2 g1 = *reinterpret_cast<const half2*>(&gv.y);
        half2 g2 = *reinterpret_cast<const half2*>(&gv.z);
        half2 g3 = *reinterpret_cast<const half2*>(&gv.w);
        half2 u0 = *reinterpret_cast<const half2*>(&uv.x);
        half2 u1 = *reinterpret_cast<const half2*>(&uv.y);
        half2 u2 = *reinterpret_cast<const half2*>(&uv.z);
        half2 u3 = *reinterpret_cast<const half2*>(&uv.w);
        float2 gf0 = __half22float2(g0);
        float2 gf1 = __half22float2(g1);
        float2 gf2 = __half22float2(g2);
        float2 gf3 = __half22float2(g3);
        float2 uf0 = __half22float2(u0);
        float2 uf1 = __half22float2(u1);
        float2 uf2 = __half22float2(u2);
        float2 uf3 = __half22float2(u3);

        #define SILU(x) ((x) / (1.0f + __expf(-(x))))
        float x0 = SILU(gf0.x) * uf0.x;
        float x1 = SILU(gf0.y) * uf0.y;
        float x2 = SILU(gf1.x) * uf1.x;
        float x3 = SILU(gf1.y) * uf1.y;
        float x4 = SILU(gf2.x) * uf2.x;
        float x5 = SILU(gf2.y) * uf2.y;
        float x6 = SILU(gf3.x) * uf3.x;
        float x7 = SILU(gf3.y) * uf3.y;
        #undef SILU

        float w0 = (float)((int)((packed >>  0) & 0xF) - 8);
        float w1 = (float)((int)((packed >>  4) & 0xF) - 8);
        float w2 = (float)((int)((packed >>  8) & 0xF) - 8);
        float w3 = (float)((int)((packed >> 12) & 0xF) - 8);
        float w4 = (float)((int)((packed >> 16) & 0xF) - 8);
        float w5 = (float)((int)((packed >> 20) & 0xF) - 8);
        float w6 = (float)((int)((packed >> 24) & 0xF) - 8);
        float w7 = (float)((int)((packed >> 28) & 0xF) - 8);

        acc += s * (w0*x0 + w1*x1 + w2*x2 + w3*x3
                  + w4*x4 + w5*x5 + w6*x6 + w7*x7);
    }

    constexpr int NWARPS = BLOCK_SIZE / 32;
    __shared__ float smem[NWARPS];
    const int warp = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        acc = (lane < NWARPS) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = (NWARPS >> 1); offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if (lane == 0) y[n] = __float2half(acc);
    }
}

void silu_mul_gemv_w4_forward_out(
    torch::Tensor gate_up,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t group_size)
{
    TORCH_CHECK(gate_up.is_cuda() && gate_up.scalar_type() == torch::kHalf, "gate_up must be CUDA fp16");
    TORCH_CHECK(qweight.is_cuda() && qweight.scalar_type() == torch::kInt, "qweight must be CUDA int32");
    TORCH_CHECK(scales.is_cuda() && scales.scalar_type() == torch::kHalf, "scales must be CUDA fp16");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kHalf, "out must be CUDA fp16");

    const int N  = qweight.size(0);
    const int K8 = qweight.size(1);
    const int K  = K8 * 8;
    TORCH_CHECK(gate_up.numel() == 2 * K, "gate_up length must equal 2*K");
    TORCH_CHECK(out.numel() >= N, "out must have at least N elements");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int BLOCK = 64;   // PPU 64-SM: B=64 sweeps measured 10-29% faster than B=128 at every decoder shape
    dim3 grid(N);
    dim3 block(BLOCK);

    #define LAUNCH_SM(GS) \
        silu_mul_gemv_w4_kernel<BLOCK, GS><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const half*>(gate_up.data_ptr<at::Half>()), \
            qweight.data_ptr<int32_t>(), \
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()), \
            reinterpret_cast<half*>(out.data_ptr<at::Half>()), \
            K);

    if (group_size == 128)      LAUNCH_SM(128)
    else if (group_size == 64)  LAUNCH_SM(64)
    else if (group_size == 32)  LAUNCH_SM(32)
    else TORCH_CHECK(false, "Unsupported group_size: ", group_size);

    #undef LAUNCH_SM
}
"""

_CPP_SRC = """
torch::Tensor gemv_w4_forward(torch::Tensor x, torch::Tensor qweight,
                              torch::Tensor scales, int64_t group_size);
void gemv_w4_forward_out(torch::Tensor x, torch::Tensor qweight,
                         torch::Tensor scales, torch::Tensor out,
                         int64_t group_size);
void gemv_w4_v3_forward_out(torch::Tensor x, torch::Tensor qweight,
                            torch::Tensor scales, torch::Tensor out,
                            int64_t group_size);
void gemv_w4_v4_forward_out(torch::Tensor x, torch::Tensor qweight,
                            torch::Tensor scales, torch::Tensor out,
                            int64_t group_size, int64_t rows_per_block,
                            int64_t block_size);
void gemv_w4_v5_batched_forward_out(torch::Tensor X, torch::Tensor qweight,
                                    torch::Tensor scales, torch::Tensor Y,
                                    int64_t group_size, int64_t m_batch,
                                    int64_t block_size);
void silu_mul_gemv_w4_forward_out(torch::Tensor gate_up, torch::Tensor qweight,
                                  torch::Tensor scales, torch::Tensor out,
                                  int64_t group_size);
"""

def _load():
    global _module
    if _module is not None:
        return _module
    from my_kernel.extension_loader import load_prebuilt
    _module = load_prebuilt("fused_w4_gemv")
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fused_w4_gemv",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["gemv_w4_forward", "gemv_w4_forward_out",
                   "gemv_w4_v3_forward_out", "gemv_w4_v4_forward_out",
                   "gemv_w4_v5_batched_forward_out",
                   "silu_mul_gemv_w4_forward_out"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_80"],
        verbose=False,
    )
    return _module

import os as _os
import json as _json
from pathlib import Path as _Path

_CONFIG_CACHE = None
def _load_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    path = _os.environ.get("MY_KERNEL_W4_CONFIG_JSON")
    if not path:
        path = str(_Path(__file__).with_name("w4_config.json"))
    if _Path(path).exists():
        try:
            with open(path) as f:
                data = _json.load(f)
            _CONFIG_CACHE = {}
            for name, r in data.get("results", {}).items():
                key = (r["N"], r["K"])
                b = r["best"]
                _CONFIG_CACHE[key] = (b["variant"], b["rows"], b["block"])
            print(f"[w4_gemv] loaded config from {path}: {len(_CONFIG_CACHE)} shape entries")
        except Exception as e:
            print(f"[w4_gemv] failed to load {path}: {e}")
            _CONFIG_CACHE = {}
    else:
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE

def _pick_variant(N: int, K: int) -> tuple:
    """Return (fn_name, rows_per_block, block_size)."""
    variant = _os.environ.get("MY_KERNEL_W4_VARIANT", "auto").lower()
    block_override = int(_os.environ.get("MY_KERNEL_W4_BLOCK_OVERRIDE", "0"))
    if variant == "v2": return ("v2", 1, block_override)
    if variant == "v3": return ("v3", 1, block_override)
    if variant.startswith("v4_r"):
        try: r = int(variant.split("r")[-1])
        except ValueError: r = 2
        return ("v4", r, block_override)

    cfg = _load_config()
    if (N, K) in cfg:
        v, r, b = cfg[(N, K)]
        return (v, r, b if block_override == 0 else block_override)

    if cfg:
        candidates = [(n, k) for (n, k) in cfg if k == K and 0.75*N <= n <= 1.25*N]
        if candidates:
            best_key = min(candidates, key=lambda nk: abs(nk[0] - N))
            v, r, b = cfg[best_key]
            return (v, r, b if block_override == 0 else block_override)

    if N >= 50000:                       return ("v4", 2, block_override)
    if N >= 18000:                       return ("v4", 8, block_override)
    if N >= 8000:                        return ("v4", 4, block_override)
    if 4500 <= N <= 6000:                return ("v3", 1, block_override)
    if N == 4096:                        return ("v2", 1, block_override)
    if N <= 2500 and K <= 4096:          return ("v3", 1, block_override)
    if N <= 2500 and K >= 4096:          return ("v2", 1, block_override)
    return ("v3", 1, block_override)

def gemv_w4_dispatch(x, qweight, scales, out, group_size: int):
    """Single entry point with env-driven kernel selection."""
    mod = _load()
    N, K8 = qweight.shape
    K = K8 * 8
    fn, rows, blk = _pick_variant(N, K)
    if fn == "v2":
        mod.gemv_w4_forward_out(x, qweight, scales, out, group_size)
    elif fn == "v3":
        mod.gemv_w4_v3_forward_out(x, qweight, scales, out, group_size)
    elif fn == "v4":
        mod.gemv_w4_v4_forward_out(x, qweight, scales, out, group_size, rows, blk)
    else:
        mod.gemv_w4_forward_out(x, qweight, scales, out, group_size)

@torch.no_grad()
def quantize_pack_w4_sym(W: torch.Tensor, group_size: int = 128):
    """Quantize and pack a (N, K) FP16 weight to (qweight, scales).

    Returns:
        qweight: (N, K/8) int32 — packed INT4 (LSB first per byte).
        scales:  (N, K/group_size) half — per-group symmetric scales.
                 Implicit zero point = 8.
    """
    N, K = W.shape
    assert K % 8 == 0 and K % group_size == 0
    G = K // group_size

    W_f32 = W.detach().to(torch.float32).view(N, G, group_size)
    absmax = W_f32.abs().amax(dim=-1)
    scales_f32 = (absmax / 7.0).clamp(min=1e-8)
    scales = scales_f32.to(W.dtype)

    q_int = torch.round(W_f32 / scales_f32.unsqueeze(-1)).clamp(-8, 7)
    q_uint = (q_int + 8).to(torch.int32).view(N, K)

    q_view = q_uint.view(N, K // 8, 8)
    shifts = (torch.arange(8, device=W.device, dtype=torch.int32) * 4).view(1, 1, 8)
    qweight = (q_view << shifts).sum(dim=-1).to(torch.int32)

    return qweight.contiguous(), scales.contiguous()

def gemv_w4(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
            group_size: int = 128) -> torch.Tensor:
    """Out-of-place forward.  x: (1, K) or (K,) fp16.  Returns (1, N) fp16."""
    _load()
    return _module.gemv_w4_forward(x.reshape(-1), qweight, scales, group_size)

def gemv_w4_out(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor,
                out: torch.Tensor, group_size: int = 128) -> None:
    """In-place forward, writes into preallocated `out` ([1,N] fp16).
    CUDA-graph safe."""
    _load()
    _module.gemv_w4_forward_out(x.reshape(-1), qweight, scales, out, group_size)

def dequant_w4_to_fp16(qweight: torch.Tensor, scales: torch.Tensor,
                       group_size: int, in_features: int) -> torch.Tensor:
    N = qweight.shape[0]
    K = in_features
    Kg = K // group_size
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    nibbles = (qweight.unsqueeze(-1) >> shifts) & 0xF
    signed  = nibbles.reshape(N, K).to(torch.float16) - 8.0
    out = (signed.reshape(N, Kg, group_size) * scales.unsqueeze(-1)).reshape(N, K)
    return out.to(torch.float16).contiguous()

class W4Linear(torch.nn.Module):
    def __init__(self, fp16_weight: torch.Tensor, group_size: int = 128):
        super().__init__()
        qw, sc = quantize_pack_w4_sym(fp16_weight, group_size)
        self.register_buffer("qweight", qw.contiguous())
        self.register_buffer("scales",  sc.contiguous())
        self.group_size   = group_size
        self.in_features  = fp16_weight.shape[1]
        self.out_features = fp16_weight.shape[0]
        self._fp16_weight_T = None

    def _get_fp16_weight_T(self):
        if self._fp16_weight_T is None:
            W = dequant_w4_to_fp16(self.qweight, self.scales,
                                    self.group_size, self.in_features)
            self._fp16_weight_T = W.t().contiguous()
        return self._fp16_weight_T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, self.in_features)
        if x_flat.shape[0] == 1:
            return gemv_w4(x_flat, self.qweight, self.scales, self.group_size)\
                    .reshape(out_shape)
        return torch.mm(x_flat, self._get_fp16_weight_T()).reshape(out_shape)
