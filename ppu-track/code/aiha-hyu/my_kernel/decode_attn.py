"""
[SHLEE] Custom fused GQA decode attention kernel for single-token decode on A100.

Single kernel with atomic tile counter for last-block reduction.
Grid = (num_q_heads, NUM_TILES) = 16x32 = 512 blocks.
No separate reduce kernel needed.

kv_len read from device pointer for CUDA graph compatibility.
"""

import torch

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <float.h>

constexpr int BLOCK = 256;
constexpr int NWARPS = BLOCK / 32;
constexpr int HD = 128;
constexpr int EPL = HD / 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, o);
    return val;
}

/*
 * Fused split-tile GQA decode attention.
 *
 * Each block processes one (q_head, tile) pair via online softmax.
 * Partial results (max, sum, weighted_out) are written to global scratch.
 * An atomic counter per q_head tracks how many tiles have finished.
 * The last tile to finish performs the cross-tile reduction and writes
 * the final output -- no second kernel needed.
 *
 * tile_done: [num_q_heads] int32 counters, must be zeroed before launch.
 */
__global__ void decode_attn_fused(
    const half* __restrict__ q,            // [num_q_heads, HD]
    const half* __restrict__ k,            // [num_kv_heads, max_cache, HD]
    const half* __restrict__ v,            // [num_kv_heads, max_cache, HD]
    half*       __restrict__ out,          // [num_q_heads, HD]
    float*      __restrict__ partial_max,  // [num_q_heads, num_tiles]
    float*      __restrict__ partial_sum,  // [num_q_heads, num_tiles]
    float*      __restrict__ partial_out,  // [num_q_heads, num_tiles, HD]
    int*        __restrict__ tile_done,    // [num_q_heads]
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

    const int kv_len = (int)(*kv_len_ptr);
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, kv_len);

    const half* q_ptr = q + q_head * HD;
    const half* k_base = k + (long long)kv_head * max_cache * HD;
    const half* v_base = v + (long long)kv_head * max_cache * HD;

    // Load Q into registers
    float q_reg[EPL];
    #pragma unroll
    for (int i = 0; i < EPL; i++)
        q_reg[i] = __half2float(q_ptr[lane_id * EPL + i]);

    // Online softmax over this tile
    float warp_max = -FLT_MAX;
    float warp_sum = 0.0f;
    float o_reg[EPL] = {0.0f};

    for (int t = tile_start + warp_id; t < tile_end; t += NWARPS) {
        const half* k_ptr = k_base + (long long)t * HD;
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPL; i++)
            dot = fmaf(q_reg[i], __half2float(__ldg(k_ptr + lane_id * EPL + i)), dot);
        dot = warp_reduce_sum(dot) * scale;

        float new_max = fmaxf(warp_max, dot);
        float rescale = __expf(warp_max - new_max);
        float w = __expf(dot - new_max);
        warp_sum = warp_sum * rescale + w;
        #pragma unroll
        for (int i = 0; i < EPL; i++)
            o_reg[i] = o_reg[i] * rescale;
        warp_max = new_max;

        const half* v_ptr = v_base + (long long)t * HD;
        #pragma unroll
        for (int i = 0; i < EPL; i++)
            o_reg[i] = fmaf(w, __half2float(__ldg(v_ptr + lane_id * EPL + i)), o_reg[i]);
    }

    // Reduce warps within block
    __shared__ float s_max[NWARPS];
    __shared__ float s_sum[NWARPS];
    __shared__ float s_out[NWARPS][HD];
    if (lane_id == 0) { s_max[warp_id] = warp_max; s_sum[warp_id] = warp_sum; }
    #pragma unroll
    for (int i = 0; i < EPL; i++)
        s_out[warp_id][lane_id * EPL + i] = o_reg[i];
    __syncthreads();

    // First 128 threads compute this tile's final partial
    if (tid < HD) {
        float gmax = -FLT_MAX;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) gmax = fmaxf(gmax, s_max[w]);
        float gsum = 0.0f, fout = 0.0f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) {
            float r = __expf(s_max[w] - gmax);
            gsum += s_sum[w] * r;
            fout += s_out[w][tid] * r;
        }
        partial_max[q_head * num_tiles + tile_id] = gmax;
        partial_sum[q_head * num_tiles + tile_id] = gsum;
        partial_out[(q_head * num_tiles + tile_id) * HD + tid] = fout;
    }

    __threadfence();

    // Last block for this q_head does the final reduction
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
                float r = __expf(partial_max[q_head * num_tiles + t] - gmax);
                gsum += partial_sum[q_head * num_tiles + t] * r;
                fout += partial_out[(q_head * num_tiles + t) * HD + tid] * r;
            }
            out[q_head * HD + tid] = __float2half(fout / gsum);
        }
        // Reset counter for next invocation (CUDA graph replay)
        if (tid == 0) tile_done[q_head] = 0;
    }
}

void decode_attention(
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
    decode_attn_fused<<<grid, BLOCK, 0, stream>>>(
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
void decode_attention(
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
        _module = load_prebuilt("decode_attn")
        if _module is not None:
            return _module
    except Exception:
        pass

    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="decode_attn",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["decode_attention"],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "--ptxas-options=-v",
        ],
        verbose=False,
    )
    return _module

NUM_TILES = 16

def alloc_attn_workspace(num_q_heads: int, max_cache_len: int, device):
    """Pre-allocate workspace buffers (CUDA-graph safe)."""
    tile_size = (max_cache_len + NUM_TILES - 1) // NUM_TILES
    return (
        torch.empty(num_q_heads, NUM_TILES, dtype=torch.float32, device=device),
        torch.empty(num_q_heads, NUM_TILES, dtype=torch.float32, device=device),
        torch.empty(num_q_heads, NUM_TILES, 128, dtype=torch.float32, device=device),
        torch.zeros(num_q_heads, dtype=torch.int32, device=device),
        tile_size,
    )

def decode_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    kv_len_t: torch.Tensor,
    scale: float,
    workspace: tuple,
) -> None:
    """In-place fused GQA decode attention (single kernel)."""
    mod = _load()
    num_q = q.shape[1]
    p_max, p_sum, p_out, tile_done, tile_size = workspace

    mod.decode_attention(
        q.view(num_q, q.shape[3]),
        k_cache.view(k_cache.shape[1], k_cache.shape[2], k_cache.shape[3]),
        v_cache.view(v_cache.shape[1], v_cache.shape[2], v_cache.shape[3]),
        out.view(num_q, out.shape[3]),
        p_max, p_sum, p_out, tile_done, kv_len_t,
        scale, NUM_TILES, tile_size,
    )
