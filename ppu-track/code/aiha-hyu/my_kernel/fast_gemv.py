"""
[SHLEE] Optimized FP16 GEMV kernel for single-token decode on Ampere

Computes y = x @ W^T  where x: [1, K], W: [N, K], y: [1, N].
All FP16, FP32 internal accumulation.

Two kernel variants:
  - gemv_small: BLOCK_SIZE=128, for N*K <= 8M (O_proj 2048x2048)
  - gemv_large: BLOCK_SIZE=256, for larger shapes
"""

import torch

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

/*
 * Single-row GEMV with 128-bit (int4) vectorized loads.
 * Each thread loads 8 halves per iteration for maximum memory throughput.
 */
template <int BLOCK_SIZE>
__global__ void gemv_fp16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ W,
    half*       __restrict__ y,
    const int K
) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    const int4* x_i4 = reinterpret_cast<const int4*>(x);
    const int4* w_i4 = reinterpret_cast<const int4*>(W + (long long)n * K);
    const int K8 = K >> 3;

    float acc = 0.0f;
    #pragma unroll 2
    for (int i = tid; i < K8; i += BLOCK_SIZE) {
        int4 wv = w_i4[i];
        int4 xv = x_i4[i];
        half2 w0 = *reinterpret_cast<const half2*>(&wv.x);
        half2 w1 = *reinterpret_cast<const half2*>(&wv.y);
        half2 w2 = *reinterpret_cast<const half2*>(&wv.z);
        half2 w3 = *reinterpret_cast<const half2*>(&wv.w);
        half2 x0 = *reinterpret_cast<const half2*>(&xv.x);
        half2 x1 = *reinterpret_cast<const half2*>(&xv.y);
        half2 x2 = *reinterpret_cast<const half2*>(&xv.z);
        half2 x3 = *reinterpret_cast<const half2*>(&xv.w);
        float2 wf0 = __half22float2(w0); float2 xf0 = __half22float2(x0);
        float2 wf1 = __half22float2(w1); float2 xf1 = __half22float2(x1);
        float2 wf2 = __half22float2(w2); float2 xf2 = __half22float2(x2);
        float2 wf3 = __half22float2(w3); float2 xf3 = __half22float2(x3);
        acc += wf0.x*xf0.x + wf0.y*xf0.y + wf1.x*xf1.x + wf1.y*xf1.y
             + wf2.x*xf2.x + wf2.y*xf2.y + wf3.x*xf3.x + wf3.y*xf3.y;
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
        for (int offset = (NWARPS >> 1) > 0 ? (NWARPS >> 1) : 1; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if (lane == 0)
            y[n] = __float2half(acc);
    }
}

/*
 * Multi-row GEMV with 128-bit vectorized loads.
 * Each block computes RPB rows, 256 threads split into groups.
 */
template <int RPB>
__global__ void gemv_fp16_multirow(
    const half* __restrict__ x,
    const half* __restrict__ W,
    half*       __restrict__ y,
    const int K,
    const int N
) {
    constexpr int THREADS = 256;
    constexpr int THREADS_PER_ROW = THREADS / RPB;
    static_assert(THREADS_PER_ROW >= 32, "need at least one warp per row");

    const int row_in_block = threadIdx.x / THREADS_PER_ROW;
    const int tid_in_row   = threadIdx.x % THREADS_PER_ROW;
    const int global_row   = blockIdx.x * RPB + row_in_block;

    if (global_row >= N) return;

    const int4* x_i4 = reinterpret_cast<const int4*>(x);
    const int4* w_i4 = reinterpret_cast<const int4*>(W + (long long)global_row * K);
    const int K8 = K >> 3;

    float acc = 0.0f;
    #pragma unroll 2
    for (int i = tid_in_row; i < K8; i += THREADS_PER_ROW) {
        int4 wv = w_i4[i];
        int4 xv = x_i4[i];
        half2 w0 = *reinterpret_cast<const half2*>(&wv.x);
        half2 w1 = *reinterpret_cast<const half2*>(&wv.y);
        half2 w2 = *reinterpret_cast<const half2*>(&wv.z);
        half2 w3 = *reinterpret_cast<const half2*>(&wv.w);
        half2 x0 = *reinterpret_cast<const half2*>(&xv.x);
        half2 x1 = *reinterpret_cast<const half2*>(&xv.y);
        half2 x2 = *reinterpret_cast<const half2*>(&xv.z);
        half2 x3 = *reinterpret_cast<const half2*>(&xv.w);
        float2 wf0 = __half22float2(w0); float2 xf0 = __half22float2(x0);
        float2 wf1 = __half22float2(w1); float2 xf1 = __half22float2(x1);
        float2 wf2 = __half22float2(w2); float2 xf2 = __half22float2(x2);
        float2 wf3 = __half22float2(w3); float2 xf3 = __half22float2(x3);
        acc += wf0.x*xf0.x + wf0.y*xf0.y + wf1.x*xf1.x + wf1.y*xf1.y
             + wf2.x*xf2.x + wf2.y*xf2.y + wf3.x*xf3.x + wf3.y*xf3.y;
    }

    constexpr int ROW_WARPS = THREADS_PER_ROW / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    __shared__ float smem[RPB * ((ROW_WARPS > 1) ? ROW_WARPS : 1)];
    const int warp_in_row = tid_in_row / 32;
    const int lane = tid_in_row & 31;

    if constexpr (ROW_WARPS > 1) {
        if (lane == 0) smem[row_in_block * ROW_WARPS + warp_in_row] = acc;
        __syncthreads();
        if (warp_in_row == 0) {
            acc = (lane < ROW_WARPS) ? smem[row_in_block * ROW_WARPS + lane] : 0.0f;
            #pragma unroll
            for (int offset = (ROW_WARPS >> 1) > 0 ? (ROW_WARPS >> 1) : 1; offset > 0; offset >>= 1)
                acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }
    }

    if (tid_in_row == 0)
        y[global_row] = __float2half(acc);
}

/*
 * GEMV with fused residual add: y[n] += x @ W[n,:] (accumulate into output)
 * Used for down_proj + residual add fusion.
 */
template <int BLOCK_SIZE>
__global__ void gemv_fp16_addres_kernel(
    const half* __restrict__ x,
    const half* __restrict__ W,
    half*       __restrict__ y,
    const int K
) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    const float2* x_f2 = reinterpret_cast<const float2*>(x);
    const float2* w_f2 = reinterpret_cast<const float2*>(W + (long long)n * K);

    const int K4 = K >> 2;
    float acc = 0.0f;

    #pragma unroll 4
    for (int i = tid; i < K4; i += BLOCK_SIZE) {
        float2 wv = w_f2[i];
        float2 xv = x_f2[i];

        half2 w0 = *reinterpret_cast<const half2*>(&wv.x);
        half2 w1 = *reinterpret_cast<const half2*>(&wv.y);
        half2 x0 = *reinterpret_cast<const half2*>(&xv.x);
        half2 x1 = *reinterpret_cast<const half2*>(&xv.y);

        float2 wf0 = __half22float2(w0);
        float2 wf1 = __half22float2(w1);
        float2 xf0 = __half22float2(x0);
        float2 xf1 = __half22float2(x1);

        acc += wf0.x * xf0.x + wf0.y * xf0.y
             + wf1.x * xf1.x + wf1.y * xf1.y;
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
        for (int offset = (NWARPS >> 1) > 0 ? (NWARPS >> 1) : 1; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if (lane == 0)
            y[n] = __float2half(acc + __half2float(y[n]));
    }
}

/*
 * Fused SiLU*Mul + GEMV: y = (SiLU(gate) * up) @ W^T
 * gate_up: [2*K] where gate_up[:K] is gate, gate_up[K:] is up
 * W: [N, K], y: [N]
 */
template <int BLOCK_SIZE>
__global__ void silu_mul_gemv_kernel(
    const half* __restrict__ gate_up,
    const half* __restrict__ W,
    half*       __restrict__ y,
    const int K
) {
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    const int4* gate_i4 = reinterpret_cast<const int4*>(gate_up);
    const int4* up_i4   = reinterpret_cast<const int4*>(gate_up + K);
    const int4* w_i4    = reinterpret_cast<const int4*>(W + (long long)n * K);
    const int K8 = K >> 3;

    float acc = 0.0f;
    #pragma unroll 2
    for (int i = tid; i < K8; i += BLOCK_SIZE) {
        int4 gv = gate_i4[i];
        int4 uv = up_i4[i];
        int4 wv = w_i4[i];
        half2 g0 = *reinterpret_cast<const half2*>(&gv.x);
        half2 g1 = *reinterpret_cast<const half2*>(&gv.y);
        half2 g2 = *reinterpret_cast<const half2*>(&gv.z);
        half2 g3 = *reinterpret_cast<const half2*>(&gv.w);
        half2 u0 = *reinterpret_cast<const half2*>(&uv.x);
        half2 u1 = *reinterpret_cast<const half2*>(&uv.y);
        half2 u2 = *reinterpret_cast<const half2*>(&uv.z);
        half2 u3 = *reinterpret_cast<const half2*>(&uv.w);
        half2 w0 = *reinterpret_cast<const half2*>(&wv.x);
        half2 w1 = *reinterpret_cast<const half2*>(&wv.y);
        half2 w2 = *reinterpret_cast<const half2*>(&wv.z);
        half2 w3 = *reinterpret_cast<const half2*>(&wv.w);

        float2 gf0 = __half22float2(g0); float2 uf0 = __half22float2(u0); float2 wf0 = __half22float2(w0);
        float2 gf1 = __half22float2(g1); float2 uf1 = __half22float2(u1); float2 wf1 = __half22float2(w1);
        float2 gf2 = __half22float2(g2); float2 uf2 = __half22float2(u2); float2 wf2 = __half22float2(w2);
        float2 gf3 = __half22float2(g3); float2 uf3 = __half22float2(u3); float2 wf3 = __half22float2(w3);

        #define SILU_MUL_DOT(gf, uf, wf) { \
            float sx = gf.x / (1.0f + __expf(-gf.x)) * uf.x; \
            float sy = gf.y / (1.0f + __expf(-gf.y)) * uf.y; \
            acc += sx * wf.x + sy * wf.y; \
        }
        SILU_MUL_DOT(gf0, uf0, wf0)
        SILU_MUL_DOT(gf1, uf1, wf1)
        SILU_MUL_DOT(gf2, uf2, wf2)
        SILU_MUL_DOT(gf3, uf3, wf3)
        #undef SILU_MUL_DOT
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
        for (int offset = (NWARPS >> 1) > 0 ? (NWARPS >> 1) : 1; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        if (lane == 0)
            y[n] = __float2half(acc);
    }
}

void silu_mul_gemv(
    torch::Tensor gate_up,
    torch::Tensor W,
    torch::Tensor y
) {
    const int N = W.size(0);
    const int K = W.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* gp = reinterpret_cast<const half*>(gate_up.data_ptr<at::Half>());
    const auto* wp = reinterpret_cast<const half*>(W.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());

    silu_mul_gemv_kernel<256><<<N, 256, 0, stream>>>(gp, wp, yp, K);
}

void gemv_fp16_addres(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor y
) {
    const int N = W.size(0);
    const int K = W.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const auto* wp = reinterpret_cast<const half*>(W.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());

    if (K <= 2048) {
        gemv_fp16_addres_kernel<128><<<N, 128, 0, stream>>>(xp, wp, yp, K);
    } else {
        gemv_fp16_addres_kernel<256><<<N, 256, 0, stream>>>(xp, wp, yp, K);
    }
}

void gemv_fp16(
    torch::Tensor x,
    torch::Tensor W,
    torch::Tensor y
) {
    const int N = W.size(0);
    const int K = W.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const auto* wp = reinterpret_cast<const half*>(W.data_ptr<at::Half>());
    auto* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());

    if (K <= 2048 && N <= 2048) {
        constexpr int RPB = 4;
        int nblocks = (N + RPB - 1) / RPB;
        gemv_fp16_multirow<RPB><<<nblocks, 256, 0, stream>>>(xp, wp, yp, K, N);
    } else if (K <= 2048) {
        gemv_fp16_kernel<128><<<N, 128, 0, stream>>>(xp, wp, yp, K);
    } else {
        gemv_fp16_kernel<256><<<N, 256, 0, stream>>>(xp, wp, yp, K);
    }
}
"""

_CPP_SRC = r"""
void gemv_fp16(torch::Tensor x, torch::Tensor W, torch::Tensor y);
void gemv_fp16_addres(torch::Tensor x, torch::Tensor W, torch::Tensor y);
void silu_mul_gemv(torch::Tensor gate_up, torch::Tensor W, torch::Tensor y);
"""

def _load():
    global _module
    if _module is not None:
        return _module

    from my_kernel.extension_loader import load_prebuilt
    _module = load_prebuilt("fast_gemv")
    if _module is not None:
        return _module

    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fast_gemv",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["gemv_fp16", "gemv_fp16_addres", "silu_mul_gemv"],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "--ptxas-options=-v",
        ],
        verbose=False,
    )
    return _module

def gemv(
    x: torch.Tensor,
    W: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """In-place GEMV: out = x @ W.T (single-row matmul)."""
    mod = _load()
    mod.gemv_fp16(x.view(-1), W, out.view(-1))

def gemv_addres(
    x: torch.Tensor,
    W: torch.Tensor,
    res: torch.Tensor,
) -> None:
    """In-place fused GEMV + residual add: res += x @ W.T"""
    mod = _load()
    mod.gemv_fp16_addres(x.view(-1), W, res.view(-1))

def fused_silu_mul_gemv(
    gate_up: torch.Tensor,
    W: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Fused SiLU(gate)*up @ W.T: eliminates separate SiLU*Mul kernel."""
    mod = _load()
    mod.silu_mul_gemv(gate_up.view(-1), W, out.view(-1))
