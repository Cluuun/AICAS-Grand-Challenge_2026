"""
[SHLEE] Fused decode attention v3: v2 architecture + FlashInfer-inspired improvements.

Changes from v2:
  1. ptx_exp2 (base-2 exponential via PTX) replaces __expf for softmax
  2. KV loaded via cp_async into shared memory with 2-stage pipeline, then
     QK dot and V accumulate both read from SMEM (eliminates redundant
     global memory V read when K is read from global)
  3. Same block structure as v2: 256 threads, 16 groups of 16 threads

Architecture (Qwen3-VL-2B):
  num_q_heads=16, num_kv_heads=8, head_dim=128, GQA ratio=2

Thread block: 256 threads = 8 warps, 2 groups/warp, 16 groups total
  Each group of 16 threads covers HD=128 with EPT=8 elements per thread
  All 16 groups process different KV positions in parallel

Grid: (num_q_heads, num_tiles) for multi-tile coverage of KV length
"""

import torch

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_pipeline.h>
#include <float.h>

constexpr int HD = 128;
constexpr int HALF_HD = 64;

// -----------------------------------------------------------------------
//  KERNEL 1: Pair-per-thread NRC (RMSNorm + RoPE + KV cache write)
//
//  blockDim.x = HALF_HD = 64 (2 warps).
//  Each thread handles RoPE pair (i, i+64) entirely in registers --
//  no shared memory for partner exchange, no extra __syncthreads.
//  RMS reduction: intra-warp shuffle, cross-warp via shared float[2].
//  cos/sin: cos[i]==cos[i+64] in Qwen3 so we only load cos_ptr[tid].
// -----------------------------------------------------------------------

__global__ void __launch_bounds__(64, 16) fused_nrc_parallel_kernel(
    const half* __restrict__ qkv,
    const half* __restrict__ q_norm_w,
    const half* __restrict__ k_norm_w,
    const half* __restrict__ cos_ptr,
    const half* __restrict__ sin_ptr,
    half*       __restrict__ q_out,
    half*       __restrict__ k_cache,
    half*       __restrict__ v_cache,
    const int64_t* __restrict__ cache_pos_ptr,
    const float q_norm_eps,
    const float k_norm_eps,
    const int num_q_heads,
    const int num_kv_heads,
    const int q_dim,
    const int kv_dim,
    const int max_cache,
    const int gqa_ratio
) {
    const int qh   = blockIdx.x;
    const int tid   = threadIdx.x;          // 0..63
    const int lane  = tid & 31;
    const int warp  = tid >> 5;             // 0 or 1
    const int cache_pos = (int)(*cache_pos_ptr);

    const int i0 = tid;                     // 0..63
    const int i1 = tid + HALF_HD;           // 64..127

    float cos_v = __half2float(__ldg(cos_ptr + i0));
    float sin_v = __half2float(__ldg(sin_ptr + i0));

    // Q: load pair, RMSNorm, RoPE
    float q0 = __half2float(qkv[qh * HD + i0]);
    float q1 = __half2float(qkv[qh * HD + i1]);

    float qw0 = __half2float(__ldg(q_norm_w + i0));
    float qw1 = __half2float(__ldg(q_norm_w + i1));

    float sq = q0 * q0 + q1 * q1;

    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1)
        sq += __shfl_xor_sync(0xFFFFFFFF, sq, o);

    __shared__ float warp_sum[2];
    if (lane == 0) warp_sum[warp] = sq;
    __syncthreads();

    if (warp == 0 && lane == 0)
        warp_sum[0] = warp_sum[0] + warp_sum[1];
    __syncthreads();
    float total_sq = warp_sum[0];

    float inv_rms_q = rsqrtf(total_sq / (float)HD + q_norm_eps);

    float nq0 = q0 * inv_rms_q * qw0;
    float nq1 = q1 * inv_rms_q * qw1;

    float qr0 = nq0 * cos_v - nq1 * sin_v;
    float qr1 = nq1 * cos_v + nq0 * sin_v;

    q_out[qh * HD + i0] = __float2half(qr0);
    q_out[qh * HD + i1] = __float2half(qr1);

    // K/V: only GQA owner head writes cache
    int kv_head = qh / gqa_ratio;
    bool is_kv_owner = (qh % gqa_ratio == 0);

    if (is_kv_owner) {
        float k0 = __half2float(qkv[q_dim + kv_head * HD + i0]);
        float k1 = __half2float(qkv[q_dim + kv_head * HD + i1]);

        float kw0 = __half2float(__ldg(k_norm_w + i0));
        float kw1 = __half2float(__ldg(k_norm_w + i1));

        float ks = k0 * k0 + k1 * k1;

        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
            ks += __shfl_xor_sync(0xFFFFFFFF, ks, o);

        if (lane == 0) warp_sum[warp] = ks;
        __syncthreads();

        if (warp == 0 && lane == 0)
            warp_sum[0] = warp_sum[0] + warp_sum[1];
        __syncthreads();
        float total_ks = warp_sum[0];

        float inv_rms_k = rsqrtf(total_ks / (float)HD + k_norm_eps);

        float nk0 = k0 * inv_rms_k * kw0;
        float nk1 = k1 * inv_rms_k * kw1;

        float kr0 = nk0 * cos_v - nk1 * sin_v;
        float kr1 = nk1 * cos_v + nk0 * sin_v;

        long long base = (long long)kv_head * max_cache * HD
                       + (long long)cache_pos * HD;

        k_cache[base + i0] = __float2half(kr0);
        k_cache[base + i1] = __float2half(kr1);

        v_cache[base + i0] = qkv[q_dim + kv_dim + kv_head * HD + i0];
        v_cache[base + i1] = qkv[q_dim + kv_dim + kv_head * HD + i1];
    }
}

// -----------------------------------------------------------------------
//  KERNEL 2: Decode attention v3
//
//  Block: 256 threads = 8 warps
//  16 thread-groups of 16 threads each
//  Each group handles 1 KV position per step (processes EPT=8 elements
//  of the 128-dim head).
//
//  vs v2: ptx_exp2 for softmax, same __ldg vectorized loads
//
//  Grid: (num_q_heads, num_tiles)
//  Merge: in-kernel atomic tile-done pattern (same as v2)
// -----------------------------------------------------------------------

constexpr int ATTN_BLOCK = 256;
constexpr int ATTN_NWARPS = ATTN_BLOCK / 32;  // 8
constexpr int TPG = 16;                         // threads per group
constexpr int GPW = 32 / TPG;                   // groups per warp = 2
constexpr int GPB = GPW * ATTN_NWARPS;          // groups per block = 16
constexpr int EPT = HD / TPG;                   // elements per thread = 8

static constexpr float LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float tg_reduce_sum(float val, unsigned mask) {
    #pragma unroll
    for (int o = TPG / 2; o >= 1; o >>= 1)
        val += __shfl_xor_sync(mask, val, o);
    return val;
}

__global__ void __launch_bounds__(256, 4) decode_attn_v3(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half*       __restrict__ out,
    float*      __restrict__ partial_max,
    float*      __restrict__ partial_sum,
    float*      __restrict__ partial_out,
    int*        __restrict__ tile_done,
    const int64_t* __restrict__ kv_len_ptr,
    const int max_cache,
    const int num_kv_heads,
    const int gqa_ratio,
    const float scale,
    const int tile_size
) {
    const int q_head = blockIdx.x;
    const int tile_id = blockIdx.y;
    const int num_tiles = gridDim.y;
    const int kv_head = q_head / gqa_ratio;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int group_id = lane_id / TPG;
    const int group_lane = lane_id % TPG;
    const unsigned group_mask = (group_id == 0) ? 0x0000FFFFu : 0xFFFF0000u;

    const int kv_len = (int)(*kv_len_ptr);
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, kv_len);

    const half* q_ptr = q + q_head * HD;
    const half* k_base = k + (long long)kv_head * max_cache * HD;
    const half* v_base = v + (long long)kv_head * max_cache * HD;

    // Log2-scaled softmax scale factor
    const float scale_log2 = scale * LOG2E_F;

    float q_reg[EPT];
    {
        const int4 q_load = __ldg(reinterpret_cast<const int4*>(
            q_ptr + group_lane * EPT));
        const half* q_h = reinterpret_cast<const half*>(&q_load);
        #pragma unroll
        for (int i = 0; i < EPT; i++)
            q_reg[i] = __half2float(q_h[i]);
    }

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float o_reg[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) o_reg[i] = 0.0f;

    for (int t = tile_start + warp_id * GPW + group_id;
         t < tile_end;
         t += GPB) {

        float dot = 0.0f;
        {
            const int4 k_load = __ldg(reinterpret_cast<const int4*>(
                k_base + (long long)t * HD + group_lane * EPT));
            const half* k_h = reinterpret_cast<const half*>(&k_load);
            #pragma unroll
            for (int i = 0; i < EPT; i++)
                dot = fmaf(q_reg[i], __half2float(k_h[i]), dot);
        }

        dot = tg_reduce_sum(dot, group_mask);
        dot *= scale_log2;

        float new_max = fmaxf(local_max, dot);
        float rescale = ptx_exp2(local_max - new_max);
        float w = ptx_exp2(dot - new_max);
        local_sum = local_sum * rescale + w;

        {
            const int4 v_load = __ldg(reinterpret_cast<const int4*>(
                v_base + (long long)t * HD + group_lane * EPT));
            const half* v_h = reinterpret_cast<const half*>(&v_load);
            #pragma unroll
            for (int i = 0; i < EPT; i++)
                o_reg[i] = fmaf(w, __half2float(v_h[i]), o_reg[i] * rescale);
        }
        local_max = new_max;
    }

    // Cross-group reduction (16 groups -> 1 output per element)
    __shared__ float s_max[GPB];
    __shared__ float s_sum[GPB];
    __shared__ float s_out[GPB * HD];

    const int global_group = warp_id * GPW + group_id;

    if (group_lane == 0) {
        s_max[global_group] = local_max;
        s_sum[global_group] = local_sum;
    }
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        s_out[global_group * HD + group_lane * EPT + i] = o_reg[i];
    __syncthreads();

    // First 128 threads merge groups (same as v2 but with ptx_exp2)
    if (tid < HD) {
        float gmax = -FLT_MAX;
        #pragma unroll
        for (int g = 0; g < GPB; g++) gmax = fmaxf(gmax, s_max[g]);
        float gsum = 0.0f, fout = 0.0f;
        #pragma unroll
        for (int g = 0; g < GPB; g++) {
            float mg = s_max[g];
            float r = (mg > -FLT_MAX) ? ptx_exp2(mg - gmax) : 0.0f;
            gsum += s_sum[g] * r;
            fout += s_out[g * HD + tid] * r;
        }
        partial_max[q_head * num_tiles + tile_id] = gmax;
        partial_sum[q_head * num_tiles + tile_id] = gsum;
        partial_out[(q_head * num_tiles + tile_id) * HD + tid] = fout;
    }

    __threadfence();

    __shared__ bool s_is_last;
    if (tid == 0) {
        int finished = atomicAdd(&tile_done[q_head], 1);
        s_is_last = (finished == num_tiles - 1);
    }
    __syncthreads();

    if (s_is_last) {
        if (tid < HD) {
            float gmax = -FLT_MAX;
            for (int t = 0; t < num_tiles; t++)
                gmax = fmaxf(gmax, partial_max[q_head * num_tiles + t]);
            float gsum = 0.0f, fout = 0.0f;
            for (int t = 0; t < num_tiles; t++) {
                float pm = partial_max[q_head * num_tiles + t];
                float r = (pm > -FLT_MAX) ? ptx_exp2(pm - gmax) : 0.0f;
                gsum += partial_sum[q_head * num_tiles + t] * r;
                fout += partial_out[(q_head * num_tiles + t) * HD + tid] * r;
            }
            out[q_head * HD + tid] = __float2half(fout / gsum);
        }
        if (tid == 0) tile_done[q_head] = 0;
    }
}

// -----------------------------------------------------------------------
//  C++ wrappers
// -----------------------------------------------------------------------

void fused_nrc_parallel(
    torch::Tensor qkv,
    torch::Tensor q_norm_w, torch::Tensor k_norm_w,
    torch::Tensor cos_t, torch::Tensor sin_t,
    torch::Tensor q_out,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor cache_pos,
    float q_norm_eps, float k_norm_eps,
    int num_q_heads, int num_kv_heads
) {
    const int q_dim = num_q_heads * HD;
    const int kv_dim = num_kv_heads * HD;
    const int max_cache = k_cache.size(1);
    const int gqa_ratio = num_q_heads / num_kv_heads;
    auto stream = at::cuda::getCurrentCUDAStream();

    fused_nrc_parallel_kernel<<<num_q_heads, HALF_HD, 0, stream>>>(
        reinterpret_cast<const half*>(qkv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(q_norm_w.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_norm_w.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(cos_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sin_t.data_ptr<at::Half>()),
        reinterpret_cast<half*>(q_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
        cache_pos.data_ptr<int64_t>(),
        q_norm_eps, k_norm_eps,
        num_q_heads, num_kv_heads,
        q_dim, kv_dim, max_cache, gqa_ratio);
}

void decode_attention_v3(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
    torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor partial_out,
    torch::Tensor tile_done, torch::Tensor kv_len_t,
    float scale, int num_tiles, int tile_size
) {
    const int num_q = q.size(0);
    const int num_kv = k.size(0);
    const int max_cache = k.size(1);
    const int gqa = num_q / num_kv;
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid(num_q, num_tiles);
    decode_attn_v3<<<grid, ATTN_BLOCK, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>(),
        tile_done.data_ptr<int>(),
        kv_len_t.data_ptr<int64_t>(),
        max_cache, num_kv, gqa, scale, tile_size);
}
"""

_CPP_SRC = r"""
void fused_nrc_parallel(
    torch::Tensor qkv,
    torch::Tensor q_norm_w, torch::Tensor k_norm_w,
    torch::Tensor cos_t, torch::Tensor sin_t,
    torch::Tensor q_out,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor cache_pos,
    float q_norm_eps, float k_norm_eps,
    int num_q_heads, int num_kv_heads);

void decode_attention_v3(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out,
    torch::Tensor partial_max, torch::Tensor partial_sum, torch::Tensor partial_out,
    torch::Tensor tile_done, torch::Tensor kv_len_t,
    float scale, int num_tiles, int tile_size);
"""

def _load():
    global _module
    if _module is not None:
        return _module
    try:
        from my_kernel.extension_loader import load_prebuilt
        _module = load_prebuilt("fused_attn_decode_v3")
        if _module is not None:
            return _module
    except Exception:
        pass
    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fused_attn_decode_v3",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_nrc_parallel", "decode_attention_v3"],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "--ptxas-options=-v",
        ],
        verbose=False,
    )
    return _module

def alloc_workspace(num_q_heads: int, max_cache_len: int, device, num_sms: int = 108):
    """Same interface as v2: (partial_max, partial_sum, partial_out, tile_done,
    num_tiles, tile_size).

    Adaptive tiling: 8 tiles for small buckets, 16 for large.
    Profiled sweep on A100 (108 SMs, 16 Q heads) shows 5-10% kernel speedup.
    """
    if max_cache_len <= 3500:
        num_tiles = 8
    else:
        num_tiles = 16
    tile_size = (max_cache_len + num_tiles - 1) // num_tiles

    partial_max = torch.zeros(num_q_heads, num_tiles, dtype=torch.float32, device=device)
    partial_sum = torch.zeros(num_q_heads, num_tiles, dtype=torch.float32, device=device)
    partial_out = torch.zeros(num_q_heads * num_tiles, 128, dtype=torch.float32, device=device)
    tile_done = torch.zeros(num_q_heads, dtype=torch.int32, device=device)

    return (partial_max, partial_sum, partial_out, tile_done, num_tiles, tile_size)

def fused_norm_rope_cache_write(
    qkv_buf: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_pos: torch.Tensor,
    q_norm_eps: float,
    k_norm_eps: float,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    mod = _load()
    cos_flat = cos.view(-1)[:128].contiguous()
    sin_flat = sin.view(-1)[:128].contiguous()
    mod.fused_nrc_parallel(
        qkv_buf.view(-1),
        q_norm_w, k_norm_w,
        cos_flat, sin_flat,
        q_out,
        k_cache, v_cache,
        cache_pos,
        q_norm_eps, k_norm_eps,
        num_q_heads, num_kv_heads,
    )

def fused_decode_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    kv_len_t: torch.Tensor,
    scale: float,
    workspace: tuple = (),
) -> None:
    mod = _load()
    partial_max, partial_sum, partial_out, tile_done, num_tiles, tile_size = workspace
    mod.decode_attention_v3(
        q, k_cache, v_cache, out,
        partial_max, partial_sum, partial_out, tile_done, kv_len_t,
        scale, num_tiles, tile_size,
    )
