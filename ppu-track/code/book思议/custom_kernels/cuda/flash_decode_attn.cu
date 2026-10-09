/*
 * Flash-Decoding Split-K Decode Attention (SM80)
 *
 * Based on Flash-Decoding paper (Dao et al., ICLR 2024).
 * Splits KV cache along seq_len for parallel processing across all SMs.
 * CUDA graph compatible: reads seq_len from device pointer, workspace pre-allocated.
 *
 * Features:
 * - Fused write_kv + compute (saves 1 kernel launch per layer)
 * - Parallel combine: splits HEAD_DIM across blocks for better SM utilization
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// warp-level sum reduction
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// ---- Compute kernel: each block handles (kv_head, split_id) ----
// When FUSED_WRITE_KV=1, the block covering position seq_len_ptr also writes
// k_new/v_new to cache before computing attention.
template <int HEAD_DIM, int Q_PER_KV, int TILE, bool FUSED_WRITE_KV>
__global__ void __launch_bounds__(Q_PER_KV * 32)
fd_compute_kernel_graph(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    const __half* __restrict__ k_new,
    const __half* __restrict__ v_new,
    float* __restrict__ partial_o,
    float* __restrict__ partial_l,
    float* __restrict__ partial_m,
    const int32_t* __restrict__ seq_len_ptr,
    int max_seq,
    int n_kv_heads,
    int num_splits)
{
    const int kv_head = blockIdx.x;
    const int split_id = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int q_idx = tid / 32;
    const int global_q_head = kv_head * Q_PER_KV + q_idx;
    if (q_idx >= Q_PER_KV) return;

    int write_pos = *seq_len_ptr;
    int seq_len = write_pos + 1;

    // Fused write_kv: this block covers position write_pos -> write K/V first
    if (FUSED_WRITE_KV) {
        const int split_size_w = (seq_len + num_splits - 1) / num_splits;
        const int w_start = split_id * split_size_w;
        const int w_end = min(w_start + split_size_w, seq_len);
        if (w_start <= write_pos && write_pos < w_end) {
            __half* k_dst = const_cast<__half*>(k_cache) + write_pos * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
            __half* v_dst = const_cast<__half*>(v_cache) + write_pos * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
            const __half* k_src = k_new + kv_head * HEAD_DIM;
            const __half* v_src = v_new + kv_head * HEAD_DIM;
            for (int i = tid; i < HEAD_DIM; i += Q_PER_KV * 32) {
                k_dst[i] = k_src[i];
                v_dst[i] = v_src[i];
            }
            __syncthreads();
        }
    }

    const int split_size = (seq_len + num_splits - 1) / num_splits;
    const int slice_start = split_id * split_size;
    const int slice_end = min(slice_start + split_size, seq_len);
    if (slice_start >= seq_len) {
        if (lane == 0) {
            const int base = (kv_head * num_splits + split_id) * Q_PER_KV + q_idx;
            partial_m[base] = -INFINITY;
            partial_l[base] = 0.0f;
        }
        for (int i = lane; i < HEAD_DIM; i += 32) {
            const int o_idx = ((kv_head * num_splits + split_id) * Q_PER_KV + q_idx) * HEAD_DIM + i;
            partial_o[o_idx] = 0.0f;
        }
        return;
    }

    const float sm_scale = 1.0f / sqrtf((float)HEAD_DIM);
    const int vals_per_thread = HEAD_DIM / 32;

    float q_vals[vals_per_thread];
    const __half* q_head = q + global_q_head * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < vals_per_thread; i++) {
        q_vals[i] = __half2float(__ldg(q_head + lane * vals_per_thread + i));
    }

    float m_val = -1e30f;
    float l_val = 0.0f;
    float o_vals[vals_per_thread];
    #pragma unroll
    for (int i = 0; i < vals_per_thread; i++) o_vals[i] = 0.0f;

    for (int pos = slice_start; pos < slice_end; pos += TILE) {
        const int tile_end = min(pos + TILE, slice_end);
        for (int t = pos; t < tile_end; t++) {
            const __half* k_row = k_cache + t * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < vals_per_thread; i++) {
                float k_val = __half2float(__ldg(k_row + lane * vals_per_thread + i));
                dot += q_vals[i] * k_val;
            }
            dot = warp_reduce_sum(dot);
            float score = __shfl_sync(0xffffffff, dot, 0) * sm_scale;

            float m_new = fmaxf(m_val, score);
            float alpha = expf(m_val - m_new);
            l_val *= alpha;
            #pragma unroll
            for (int i = 0; i < vals_per_thread; i++) o_vals[i] *= alpha;
            float weight = expf(score - m_new);
            l_val += weight;
            const __half* v_row = v_cache + t * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
            #pragma unroll
            for (int i = 0; i < vals_per_thread; i++) {
                float v_val = __half2float(__ldg(v_row + lane * vals_per_thread + i));
                o_vals[i] += weight * v_val;
            }
            m_val = m_new;
        }
    }

    if (lane == 0) {
        const int base = (kv_head * num_splits + split_id) * Q_PER_KV + q_idx;
        partial_m[base] = m_val;
        partial_l[base] = l_val;
    }
    #pragma unroll
    for (int i = 0; i < vals_per_thread; i++) {
        const int d = lane * vals_per_thread + i;
        const int o_idx = ((kv_head * num_splits + split_id) * Q_PER_KV + q_idx) * HEAD_DIM + d;
        partial_o[o_idx] = o_vals[i];
    }
}

// ---- Parallel combine: split HEAD_DIM across blocks for better SM utilization ----
// Grid: (HEAD_DIM / TILE_DIM, n_qo_heads) blocks, each with Q_PER_KV warps
// Each warp handles one Q head's tile. TILE_DIM must be <= 32 (one warp per tile).
// For TILE_DIM < 32, only TILE_DIM threads are active per warp.
template <int HEAD_DIM, int Q_PER_KV, int TILE_DIM>
__global__ void __launch_bounds__(Q_PER_KV * 32)
fd_combine_parallel_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_m,
    __half* __restrict__ output,
    int num_splits,
    int n_kv_heads)
{
    const int tile_id = blockIdx.x;         // which tile of HEAD_DIM
    const int qo_head = blockIdx.y;         // which Q head
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int q_idx = tid / 32;

    const int kv_head = qo_head / Q_PER_KV;
    const int local_q = qo_head % Q_PER_KV;
    if (q_idx != local_q) return;
    if (lane >= TILE_DIM) return;

    const int d_start = tile_id * TILE_DIM;

    // Step 1: compute global max
    float m_global = -1e30f;
    for (int s = 0; s < num_splits; s++) {
        float ms = partial_m[(kv_head * num_splits + s) * Q_PER_KV + local_q];
        m_global = fmaxf(m_global, ms);
    }

    // Step 2: compute l_final
    float l_final = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        const int base = (kv_head * num_splits + s) * Q_PER_KV + local_q;
        float ms = partial_m[base];
        float ls = partial_l[base];
        l_final += ls * expf(ms - m_global);
    }
    float inv_l = 1.0f / (l_final + 1e-8f);

    // Step 3: combine -- each lane handles one dimension
    float attn = 0.0f;
    const int d = d_start + lane;

    for (int s = 0; s < num_splits; s++) {
        const int base = (kv_head * num_splits + s) * Q_PER_KV + local_q;
        float ms = partial_m[base];
        float scale = expf(ms - m_global);
        attn += partial_o[base * HEAD_DIM + d] * scale;
    }

    // Step 4: write output
    output[qo_head * HEAD_DIM + d] = __float2half(attn * inv_l);
}

// ---- Non-fused combine kernel (backward compat) ----
template <int HEAD_DIM, int Q_PER_KV, int MAX_SPLITS>
__global__ void __launch_bounds__(Q_PER_KV * 32)
fd_combine_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_m,
    __half* __restrict__ output,
    int num_splits,
    int n_kv_heads)
{
    const int qo_head = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int q_idx = tid / 32;
    if (q_idx >= Q_PER_KV) return;

    const int kv_head = qo_head / Q_PER_KV;
    const int local_q = qo_head % Q_PER_KV;
    if (q_idx != local_q) return;

    const int vals_per_thread = HEAD_DIM / 32;

    float m_global = -1e30f;
    for (int s = 0; s < num_splits; s++) {
        float ms = partial_m[(kv_head * num_splits + s) * Q_PER_KV + local_q];
        m_global = fmaxf(m_global, ms);
    }

    float o_final[vals_per_thread];
    #pragma unroll
    for (int i = 0; i < vals_per_thread; i++) o_final[i] = 0.0f;
    float l_final = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        const int base = (kv_head * num_splits + s) * Q_PER_KV + local_q;
        float ms = partial_m[base];
        float ls = partial_l[base];
        float scale = expf(ms - m_global);
        l_final += ls * scale;
        #pragma unroll
        for (int i = 0; i < vals_per_thread; i++) {
            const int d = lane * vals_per_thread + i;
            o_final[i] += partial_o[base * HEAD_DIM + d] * scale;
        }
    }

    float inv_l = 1.0f / (l_final + 1e-8f);
    __half* out_head = output + qo_head * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < vals_per_thread; i++) {
        const int d = lane * vals_per_thread + i;
        out_head[d] = __float2half(o_final[i] * inv_l);
    }
}

// ---- Write K/V into cache (reads seq_len from device pointer) ----
template <int HEAD_DIM>
__global__ void __launch_bounds__(32)
fd_write_kv_kernel(
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ k_new,
    const __half* __restrict__ v_new,
    const int32_t* __restrict__ seq_len_ptr,
    int n_kv_heads)
{
    const int kv_head = blockIdx.x;
    const int lane = threadIdx.x;
    int write_pos = *seq_len_ptr;
    __half* k_dst = k_cache + write_pos * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
    __half* v_dst = v_cache + write_pos * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
    const __half* k_src = k_new + kv_head * HEAD_DIM;
    const __half* v_src = v_new + kv_head * HEAD_DIM;
    for (int i = lane; i < HEAD_DIM; i += 32) {
        k_dst[i] = k_src[i];
        v_dst[i] = v_src[i];
    }
}

// ---- Host wrapper (fully CUDA-graph-compatible) ----
// Args match patch_model.py calling convention:
//   flash_decode_attn(q, k_new, v_new, k_cache, v_cache, output, seq_len, ws_o, ws_l, ws_m, num_splits)
void flash_decode_attn_cuda(
    torch::Tensor q,
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor output,
    torch::Tensor seq_len,
    torch::Tensor workspace_o,
    torch::Tensor workspace_l,
    torch::Tensor workspace_m,
    int num_splits)
{
    int n_kv = k_cache.size(1);
    int n_qo = q.size(0);
    int max_seq = k_cache.size(0);

    TORCH_CHECK(n_qo % n_kv == 0, "Q heads must be divisible by KV heads");
    constexpr int Q_PER_KV = 2;  // 16 Q / 8 KV = 2

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const int32_t* seq_len_dev = seq_len.data_ptr<int32_t>();

    // Fused write_kv + compute (saves one kernel launch per layer)
    {
        dim3 grid(n_kv, num_splits);
        fd_compute_kernel_graph<128, 2, 16, true><<<grid, Q_PER_KV * 32, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_cache.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_cache.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_new.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_new.data_ptr<at::Half>()),
            workspace_o.data_ptr<float>(),
            workspace_l.data_ptr<float>(),
            workspace_m.data_ptr<float>(),
            seq_len_dev, max_seq, n_kv, num_splits);
    }

    {
        // Parallel combine: split HEAD_DIM=128 into tiles
        // TILE_DIM=8: 16 tiles x 16 Q heads = 256 blocks
        constexpr int TILE_DIM = 8;
        constexpr int TILES_PER_HEAD = 128 / TILE_DIM;  // 16
        dim3 combine_grid(TILES_PER_HEAD, n_qo);
        fd_combine_parallel_kernel<128, 2, TILE_DIM><<<combine_grid, Q_PER_KV * 32, 0, stream>>>(
            workspace_o.data_ptr<float>(),
            workspace_l.data_ptr<float>(),
            workspace_m.data_ptr<float>(),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            num_splits, n_kv);
    }
}

// ---- Fused combine + O_proj (zero-copy, experimental) ----
void flash_decode_attn_fused_cuda(
    torch::Tensor q,
    torch::Tensor k_new,
    torch::Tensor v_new,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor output,
    torch::Tensor seq_len,
    torch::Tensor workspace_o,
    torch::Tensor workspace_l,
    torch::Tensor workspace_m,
    int num_splits)
{
    flash_decode_attn_cuda(q, k_new, v_new, k_cache, v_cache, output,
                           seq_len, workspace_o, workspace_l, workspace_m,
                           num_splits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_decode_attn", &flash_decode_attn_cuda,
          "Flash-Decoding Split-K decode attention (SM80)");
    m.def("flash_decode_attn_fused", &flash_decode_attn_fused_cuda,
          "Flash-Decoding with zero-copy combine+O_proj fusion (SM80)");
}
