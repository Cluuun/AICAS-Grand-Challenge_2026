#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <math_constants.h>

template <typename T>
__device__ __forceinline__ float fd_to_float(T v);

template <>
__device__ __forceinline__ float fd_to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float fd_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T fd_from_float(float v);

template <>
__device__ __forceinline__ __half fd_from_float<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fd_from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffffu, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[8];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int warp_count = (blockDim.x + 31) / 32;
    val = warp_reduce_max(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < warp_count) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[8];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int warp_count = (blockDim.x + 31) / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < warp_count) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

template <typename T>
__global__ void decode_multi_masked_kernel(
    const T* __restrict__ query,  // [B, H_q, Q, 128]
    const T* __restrict__ key,    // [B, H_kv, S, 128]
    const T* __restrict__ value,  // [B, H_kv, S, 128]
    const T* __restrict__ mask,   // [B, 1, Q, S], contiguous
    T* __restrict__ out,          // [B, Q, H_q, 128]
    const int B,
    const int H_q,
    const int H_kv,
    const int G,
    const int Q,
    const int S) {
    extern __shared__ float scores[];

    const int hq = blockIdx.x;
    const int qi = blockIdx.y;
    const int b = blockIdx.z;
    if (b >= B || qi >= Q || hq >= H_q) return;

    const int kv = hq / G;
    constexpr int Kdim = 128;
    constexpr float scale = 0.0883883476482f;

    const T* q_base = query + (((b * H_q + hq) * Q + qi) * Kdim);
    const T* k_base = key + (((b * H_kv + kv) * S) * Kdim);
    const T* v_base = value + (((b * H_kv + kv) * S) * Kdim);
    const T* m_base = mask + ((b * Q + qi) * S);

    float local_max = -FLT_MAX;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < Kdim; ++d) {
            dot += fd_to_float(q_base[d]) * fd_to_float(k_base[s * Kdim + d]);
        }
        float score = dot * scale + fd_to_float(m_base[s]);
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    float max_score = block_reduce_max(local_max);

    float local_sum = 0.0f;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float p = __expf(scores[s] - max_score);
        scores[s] = p;
        local_sum += p;
    }
    float denom = block_reduce_sum(local_sum);
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

    for (int d = threadIdx.x; d < Kdim; d += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < S; ++s) {
            acc += scores[s] * fd_to_float(v_base[s * Kdim + d]);
        }
        out[(((b * Q + qi) * H_q + hq) * Kdim + d)] = fd_from_float<T>(acc * inv_denom);
    }
}

__global__ void build_kv_ptrs_kernel(
    const __half* k_base,
    const __half* v_base,
    const int B,
    const int H_kv,
    const int G,
    const int S,
    const int Kdim,
    const __half** k_ptrs,
    const __half** v_ptrs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int H = H_kv * G;
    const int total = B * H;
    if (idx >= total) return;

    const int b = idx / H;
    const int h = idx % H;
    const int kv = h / G;

    const size_t offset = (static_cast<size_t>(b) * H_kv + kv) * static_cast<size_t>(S) * static_cast<size_t>(Kdim);
    k_ptrs[idx] = k_base + offset;
    v_ptrs[idx] = v_base + offset;
}

__global__ void pack_query_kernel(
    const __half* query,   // [B, H_q, 1, K]
    __half* q_packed,      // [B, H_kv, G+2, K]
    const int B,
    const int H_kv,
    const int G,
    const int Kdim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * H_kv * (G + 2) * Kdim;
    if (idx >= total) return;

    const int k = idx % Kdim;
    int t = idx / Kdim;
    const int g = t % (G + 2);
    t /= (G + 2);
    const int kv = t % H_kv;
    const int b = t / H_kv;

    if (g < G) {
        const int hq = kv * G + g;
        const int src_idx = (((b * (H_kv * G) + hq) * 1 + 0) * Kdim + k);
        q_packed[idx] = query[src_idx];
    } else {
        q_packed[idx] = __float2half(0.0f);
    }
}

__global__ void unpack_output_kernel(
    const __half* out_packed,  // [B, H_kv, G, K]
    __half* attn_out,          // [B, 1, H_q, K]
    const int B,
    const int H_kv,
    const int G,
    const int Kdim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int H_q = H_kv * G;
    const int total = B * H_q * Kdim;
    if (idx >= total) return;

    const int k = idx % Kdim;
    int t = idx / Kdim;
    const int hq = t % H_q;
    const int b = t / H_q;

    const int kv = hq / G;
    const int g = hq % G;

    const int src_idx = (((b * H_kv + kv) * G + g) * Kdim + k);
    const int dst_idx = (((b * 1 + 0) * H_q + hq) * Kdim + k);
    attn_out[dst_idx] = out_packed[src_idx];
}

extern "C" void flashdecode_build_kv_ptrs(
    const __half* k_base,
    const __half* v_base,
    int B,
    int H_kv,
    int G,
    int S,
    int Kdim,
    const __half** k_ptrs,
    const __half** v_ptrs,
    cudaStream_t stream) {
    const int total = B * H_kv * G;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    build_kv_ptrs_kernel<<<blocks, threads, 0, stream>>>(
        k_base, v_base, B, H_kv, G, S, Kdim, k_ptrs, v_ptrs
    );
}

extern "C" void flashdecode_pack_query(
    const __half* query,
    __half* q_packed,
    int B,
    int H_kv,
    int G,
    int Kdim,
    cudaStream_t stream) {
    const int total = B * H_kv * (G + 2) * Kdim;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    pack_query_kernel<<<blocks, threads, 0, stream>>>(
        query, q_packed, B, H_kv, G, Kdim
    );
}

extern "C" void flashdecode_unpack_output(
    const __half* out_packed,
    __half* attn_out,
    int B,
    int H_kv,
    int G,
    int Kdim,
    cudaStream_t stream) {
    const int total = B * H_kv * G * Kdim;
    const int threads = 128;
    const int blocks = (total + threads - 1) / threads;
    unpack_output_kernel<<<blocks, threads, 0, stream>>>(
        out_packed, attn_out, B, H_kv, G, Kdim
    );
}

extern "C" void flashdecode_run_decode_multi_masked(
    const __half* query,
    const __half* key,
    const __half* value,
    const __half* mask,
    __half* out,
    int B,
    int H_q,
    int H_kv,
    int G,
    int Q,
    int S,
    cudaStream_t stream) {
    const dim3 grid(H_q, Q, B);
    const int threads = 256;
    const size_t smem = static_cast<size_t>(S) * sizeof(float);
    decode_multi_masked_kernel<__half><<<grid, threads, smem, stream>>>(
        query, key, value, mask, out, B, H_q, H_kv, G, Q, S
    );
}

extern "C" void flashdecode_run_decode_multi_masked_bf16(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const __nv_bfloat16* mask,
    __nv_bfloat16* out,
    int B,
    int H_q,
    int H_kv,
    int G,
    int Q,
    int S,
    cudaStream_t stream) {
    const dim3 grid(H_q, Q, B);
    const int threads = 256;
    const size_t smem = static_cast<size_t>(S) * sizeof(float);
    decode_multi_masked_kernel<__nv_bfloat16><<<grid, threads, smem, stream>>>(
        query, key, value, mask, out, B, H_q, H_kv, G, Q, S
    );
}
