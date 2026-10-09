#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>

namespace {

constexpr int kNumQHeads = 16;
constexpr int kNumKVHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kHalfHeadDim = 64;
constexpr int kThreads = 128;
constexpr int kOptThreads = 256;
constexpr int kOptBlockKV = 64;
constexpr int kOptThreads128 = 512;
constexpr int kOptBlockKV128 = 128;
constexpr int kGqa = kNumQHeads / kNumKVHeads;
constexpr float kNegInf = -3.4028234663852886e38F;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ signed char round_i8(float x) {
    x = fminf(127.0f, fmaxf(-127.0f, x));
    float rounded = x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f);
    return static_cast<signed char>(rounded);
}

__device__ __forceinline__ int pack_i8x4(signed char x0, signed char x1, signed char x2, signed char x3) {
    return (static_cast<unsigned char>(x0))
        | (static_cast<unsigned char>(x1) << 8)
        | (static_cast<unsigned char>(x2) << 16)
        | (static_cast<unsigned char>(x3) << 24);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float shared[16];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < warps ? shared[lane] : 0.0f;
    if (warp == 0) {
        v = warp_sum(v);
    }
    if (threadIdx.x == 0) {
        shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ void block_sum_pair(float& a, float& b) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    a = warp_sum(a);
    b = warp_sum(b);
    if (lane == 0) {
        shared[warp] = a;
        shared[warp + 16] = b;
    }
    __syncthreads();
    a = threadIdx.x < warps ? shared[lane] : 0.0f;
    b = threadIdx.x < warps ? shared[lane + 16] : 0.0f;
    if (warp == 0) {
        a = warp_sum(a);
        b = warp_sum(b);
    }
    if (threadIdx.x == 0) {
        shared[0] = a;
        shared[1] = b;
    }
    __syncthreads();
    a = shared[0];
    b = shared[1];
}

__device__ __forceinline__ void block_sum_triple(float& a, float& b, float& c) {
    __shared__ float shared[48];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    a = warp_sum(a);
    b = warp_sum(b);
    c = warp_sum(c);
    if (lane == 0) {
        shared[warp] = a;
        shared[warp + 16] = b;
        shared[warp + 32] = c;
    }
    __syncthreads();
    a = threadIdx.x < warps ? shared[lane] : 0.0f;
    b = threadIdx.x < warps ? shared[lane + 16] : 0.0f;
    c = threadIdx.x < warps ? shared[lane + 32] : 0.0f;
    if (warp == 0) {
        a = warp_sum(a);
        b = warp_sum(b);
        c = warp_sum(c);
    }
    if (threadIdx.x == 0) {
        shared[0] = a;
        shared[1] = b;
        shared[2] = c;
    }
    __syncthreads();
    a = shared[0];
    b = shared[1];
    c = shared[2];
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float shared[16];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    v = warp_max(v);
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < warps ? shared[lane] : kNegInf;
    if (warp == 0) {
        v = warp_max(v);
    }
    if (threadIdx.x == 0) {
        shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

__device__ __forceinline__ float rope_value(
    const __nv_bfloat16* __restrict__ row,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    int d,
    float rstd) {
    const int half = d & (kHalfHeadDim - 1);
    const bool right = d >= kHalfHeadDim;
    const int left_idx = half;
    const int right_idx = half + kHalfHeadDim;
    const float left = bf16_to_float(row[left_idx]) * bf16_to_float(weight[left_idx]) * rstd;
    const float right_v = bf16_to_float(row[right_idx]) * bf16_to_float(weight[right_idx]) * rstd;
    const float c = bf16_to_float(cos[half]);
    const float s = bf16_to_float(sin[half]);
    return right ? (right_v * c + left * s) : (left * c - right_v * s);
}

__device__ __forceinline__ void online_update(
    float score,
    float value,
    float& m,
    float& l,
    float& acc) {
    const float new_m = fmaxf(m, score);
    const float alpha = expf(m - new_m);
    const float beta = expf(score - new_m);
    acc = acc * alpha + beta * value;
    l = l * alpha + beta;
    m = new_m;
}

__device__ __forceinline__ void tile_softmax_weights(
    float* scores0,
    float* scores1,
    int tile_len,
    float& m0,
    float& l0,
    float& m1,
    float& l1) {
    if (threadIdx.x == 0) {
        m0 = kNegInf;
        m1 = kNegInf;
        for (int i = 0; i < tile_len; ++i) {
            m0 = fmaxf(m0, scores0[i]);
            m1 = fmaxf(m1, scores1[i]);
        }
        l0 = 0.0f;
        l1 = 0.0f;
        for (int i = 0; i < tile_len; ++i) {
            const float e0 = expf(scores0[i] - m0);
            const float e1 = expf(scores1[i] - m1);
            scores0[i] = e0;
            scores1[i] = e1;
            l0 += e0;
            l1 += e1;
        }
    }
}

__device__ __forceinline__ void tile_softmax_weights_parallel(
    float* scores0,
    float* scores1,
    int tile_len,
    float& m0,
    float& l0,
    float& m1,
    float& l1) {
    float local_m0 = threadIdx.x < tile_len ? scores0[threadIdx.x] : kNegInf;
    float local_m1 = threadIdx.x < tile_len ? scores1[threadIdx.x] : kNegInf;
    m0 = block_max(local_m0);
    m1 = block_max(local_m1);

    float local_l0 = 0.0f;
    float local_l1 = 0.0f;
    if (threadIdx.x < tile_len) {
        local_l0 = expf(scores0[threadIdx.x] - m0);
        local_l1 = expf(scores1[threadIdx.x] - m1);
        scores0[threadIdx.x] = local_l0;
        scores1[threadIdx.x] = local_l1;
    }
    l0 = local_l0;
    l1 = local_l1;
    block_sum_pair(l0, l1);
}

__global__ void decode_k_i8_v_bf16_single_kernel(
    const __nv_bfloat16* __restrict__ q,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    float softmax_scale,
    __nv_bfloat16* __restrict__ out) {
    const int kvh = blockIdx.x;
    const int d = threadIdx.x;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const int kv_len = cache_seqlens[0];
    const float q0 = bf16_to_float(q[qh0 * kHeadDim + d]);
    const float q1 = bf16_to_float(q[qh1 * kHeadDim + d]);
    float m0 = kNegInf;
    float m1 = kNegInf;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int t = 0; t < kv_len; ++t) {
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const float ks = k_scale[t * kNumKVHeads + kvh];
        const float k = static_cast<float>(k_i8[kv_base + d]) * ks;
        float dot0 = q0 * k;
        float dot1 = q1 * k;
        block_sum_pair(dot0, dot1);
        const float value = bf16_to_float(v[kv_base + d]);
        online_update(dot0 * softmax_scale, value, m0, l0, acc0);
        online_update(dot1 * softmax_scale, value, m1, l1, acc1);
    }

    out[qh0 * kHeadDim + d] = __float2bfloat16(acc0 / fmaxf(l0, 1.0e-20f));
    out[qh1 * kHeadDim + d] = __float2bfloat16(acc1 / fmaxf(l1, 1.0e-20f));
}

__global__ void decode_q_i8_k_i8_v_bf16_single_kernel(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ q_scale,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    float softmax_scale,
    __nv_bfloat16* __restrict__ out) {
    const int kvh = blockIdx.x;
    const int d = threadIdx.x;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const int kv_len = cache_seqlens[0];
    const int q0 = static_cast<int>(q_i8[qh0 * kHeadDim + d]);
    const int q1 = static_cast<int>(q_i8[qh1 * kHeadDim + d]);
    const float qs0 = q_scale[qh0];
    const float qs1 = q_scale[qh1];
    float m0 = kNegInf;
    float m1 = kNegInf;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int t = 0; t < kv_len; ++t) {
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const int k = static_cast<int>(k_i8[kv_base + d]);
        float dot0 = static_cast<float>(q0 * k);
        float dot1 = static_cast<float>(q1 * k);
        block_sum_pair(dot0, dot1);
        const float score_scale = k_scale[t * kNumKVHeads + kvh] * softmax_scale;
        const float value = bf16_to_float(v[kv_base + d]);
        online_update(dot0 * qs0 * score_scale, value, m0, l0, acc0);
        online_update(dot1 * qs1 * score_scale, value, m1, l1, acc1);
    }

    out[qh0 * kHeadDim + d] = __float2bfloat16(acc0 / fmaxf(l0, 1.0e-20f));
    out[qh1 * kHeadDim + d] = __float2bfloat16(acc1 / fmaxf(l1, 1.0e-20f));
}

__global__ void decode_qkv_k_i8_v_bf16_single_kernel(
    const __nv_bfloat16* __restrict__ flat_qkv,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    signed char* __restrict__ k_i8,
    __nv_bfloat16* __restrict__ v,
    float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cache_seqlens_next,
    float softmax_scale,
    float q_eps,
    float k_eps,
    __nv_bfloat16* __restrict__ out) {
    const int kvh = blockIdx.x;
    const int d = threadIdx.x;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const int q_size = kNumQHeads * kHeadDim;
    const int kv_size = kNumKVHeads * kHeadDim;
    const int cache_pos = cache_seqlens[0];
    const int kv_len = cache_seqlens_next[0];

    const __nv_bfloat16* q0_row = flat_qkv + qh0 * kHeadDim;
    const __nv_bfloat16* q1_row = flat_qkv + qh1 * kHeadDim;
    const __nv_bfloat16* k_row = flat_qkv + q_size + kvh * kHeadDim;
    const __nv_bfloat16* v_row = flat_qkv + q_size + kv_size + kvh * kHeadDim;

    const float q0_sq = bf16_to_float(q0_row[d]) * bf16_to_float(q0_row[d]);
    const float q1_sq = bf16_to_float(q1_row[d]) * bf16_to_float(q1_row[d]);
    const float k_sq = bf16_to_float(k_row[d]) * bf16_to_float(k_row[d]);
    const float q0_rstd = rsqrtf(block_sum(q0_sq) / static_cast<float>(kHeadDim) + q_eps);
    const float q1_rstd = rsqrtf(block_sum(q1_sq) / static_cast<float>(kHeadDim) + q_eps);
    const float k_rstd = rsqrtf(block_sum(k_sq) / static_cast<float>(kHeadDim) + k_eps);

    const float q0 = rope_value(q0_row, q_weight, cos, sin, d, q0_rstd);
    const float q1 = rope_value(q1_row, q_weight, cos, sin, d, q1_rstd);
    const float k_cur = rope_value(k_row, k_weight, cos, sin, d, k_rstd);
    const float k_abs_max = block_max(fabsf(k_cur));
    const float ks_cur = k_abs_max > 0.0f ? k_abs_max / 127.0f : 1.0f;

    const int cur_base = (cache_pos * kNumKVHeads + kvh) * kHeadDim;
    k_i8[cur_base + d] = round_i8(k_cur / ks_cur);
    v[cur_base + d] = v_row[d];
    if (d == 0) {
        k_scale[cache_pos * kNumKVHeads + kvh] = ks_cur;
    }
    __syncthreads();

    float m0 = kNegInf;
    float m1 = kNegInf;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const float ks = k_scale[t * kNumKVHeads + kvh];
        const float k = static_cast<float>(k_i8[kv_base + d]) * ks;
        float dot0 = q0 * k;
        float dot1 = q1 * k;
        block_sum_pair(dot0, dot1);
        const float value = bf16_to_float(v[kv_base + d]);
        online_update(dot0 * softmax_scale, value, m0, l0, acc0);
        online_update(dot1 * softmax_scale, value, m1, l1, acc1);
    }

    out[qh0 * kHeadDim + d] = __float2bfloat16(acc0 / fmaxf(l0, 1.0e-20f));
    out[qh1 * kHeadDim + d] = __float2bfloat16(acc1 / fmaxf(l1, 1.0e-20f));
}

__global__ void decode_k_i8_v_bf16_split_partial_kernel(
    const __nv_bfloat16* __restrict__ q,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    int max_splits,
    float softmax_scale,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int d = threadIdx.x;
    const int kv_len = cache_seqlens[0];
    const int start = split * 64;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + 64, kv_len);
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const float q0 = bf16_to_float(q[qh0 * kHeadDim + d]);
    const float q1 = bf16_to_float(q[qh1 * kHeadDim + d]);
    float m0 = kNegInf;
    float m1 = kNegInf;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int t = start; t < end; ++t) {
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const float k = static_cast<float>(k_i8[kv_base + d]) * k_scale[t * kNumKVHeads + kvh];
        float dot0 = q0 * k;
        float dot1 = q1 * k;
        block_sum_pair(dot0, dot1);
        const float value = bf16_to_float(v[kv_base + d]);
        online_update(dot0 * softmax_scale, value, m0, l0, acc0);
        online_update(dot1 * softmax_scale, value, m1, l1, acc1);
    }

    partial_max[qh0 * max_splits + split] = m0;
    partial_sum[qh0 * max_splits + split] = l0;
    partial_out[(qh0 * max_splits + split) * kHeadDim + d] = acc0;
    partial_max[qh1 * max_splits + split] = m1;
    partial_sum[qh1 * max_splits + split] = l1;
    partial_out[(qh1 * max_splits + split) * kHeadDim + d] = acc1;
}

__global__ void decode_k_i8_v_bf16_split_partial_opt_kernel(
    const __nv_bfloat16* __restrict__ q,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    int max_splits,
    float softmax_scale,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out) {
    __shared__ float scores0[kOptBlockKV];
    __shared__ float scores1[kOptBlockKV];
    __shared__ float m_l[4];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_len = cache_seqlens[0];
    const int start = split * kOptBlockKV;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + kOptBlockKV, kv_len);
    const int tile_len = end - start;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;

    for (int ti = warp; ti < tile_len; ti += 8) {
        const int t = start + ti;
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const float ks = k_scale[t * kNumKVHeads + kvh];
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        for (int d = lane; d < kHeadDim; d += 32) {
            const float kval = static_cast<float>(k_i8[kv_base + d]) * ks;
            dot0 += bf16_to_float(q[qh0 * kHeadDim + d]) * kval;
            dot1 += bf16_to_float(q[qh1 * kHeadDim + d]) * kval;
        }
        dot0 = warp_sum(dot0);
        dot1 = warp_sum(dot1);
        if (lane == 0) {
            scores0[ti] = dot0 * softmax_scale;
            scores1[ti] = dot1 * softmax_scale;
        }
    }
    __syncthreads();

    float m0 = kNegInf, l0 = 0.0f, m1 = kNegInf, l1 = 0.0f;
    tile_softmax_weights_parallel(scores0, scores1, tile_len, m0, l0, m1, l1);
    if (tid == 0) {
        m_l[0] = m0;
        m_l[1] = l0;
        m_l[2] = m1;
        m_l[3] = l1;
        partial_max[qh0 * max_splits + split] = m0;
        partial_sum[qh0 * max_splits + split] = l0;
        partial_max[qh1 * max_splits + split] = m1;
        partial_sum[qh1 * max_splits + split] = l1;
    }
    __syncthreads();

    if (tid < kHeadDim) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int ti = 0; ti < tile_len; ++ti) {
            const int kv_base = ((start + ti) * kNumKVHeads + kvh) * kHeadDim;
            const float value = bf16_to_float(v[kv_base + tid]);
            acc0 += scores0[ti] * value;
            acc1 += scores1[ti] * value;
        }
        partial_out[(qh0 * max_splits + split) * kHeadDim + tid] = acc0;
        partial_out[(qh1 * max_splits + split) * kHeadDim + tid] = acc1;
    }
}

__global__ void decode_q_i8_k_i8_v_bf16_split_partial_kernel(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ q_scale,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    int max_splits,
    float softmax_scale,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int d = threadIdx.x;
    const int kv_len = cache_seqlens[0];
    const int start = split * 64;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + 64, kv_len);
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const int q0 = static_cast<int>(q_i8[qh0 * kHeadDim + d]);
    const int q1 = static_cast<int>(q_i8[qh1 * kHeadDim + d]);
    const float qs0 = q_scale[qh0];
    const float qs1 = q_scale[qh1];
    float m0 = kNegInf;
    float m1 = kNegInf;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int t = start; t < end; ++t) {
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const int k = static_cast<int>(k_i8[kv_base + d]);
        float dot0 = static_cast<float>(q0 * k);
        float dot1 = static_cast<float>(q1 * k);
        block_sum_pair(dot0, dot1);
        const float score_scale = k_scale[t * kNumKVHeads + kvh] * softmax_scale;
        const float value = bf16_to_float(v[kv_base + d]);
        online_update(dot0 * qs0 * score_scale, value, m0, l0, acc0);
        online_update(dot1 * qs1 * score_scale, value, m1, l1, acc1);
    }

    partial_max[qh0 * max_splits + split] = m0;
    partial_sum[qh0 * max_splits + split] = l0;
    partial_out[(qh0 * max_splits + split) * kHeadDim + d] = acc0;
    partial_max[qh1 * max_splits + split] = m1;
    partial_sum[qh1 * max_splits + split] = l1;
    partial_out[(qh1 * max_splits + split) * kHeadDim + d] = acc1;
}

__global__ void decode_q_i8_k_i8_v_bf16_split_partial_opt_kernel(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ q_scale,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    int max_splits,
    float softmax_scale,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out) {
    __shared__ float scores0[kOptBlockKV];
    __shared__ float scores1[kOptBlockKV];
    __shared__ float m_l[4];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_len = cache_seqlens[0];
    const int start = split * kOptBlockKV;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + kOptBlockKV, kv_len);
    const int tile_len = end - start;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const float qs0 = q_scale[qh0];
    const float qs1 = q_scale[qh1];
    const int* q0_pack = reinterpret_cast<const int*>(q_i8 + qh0 * kHeadDim);
    const int* q1_pack = reinterpret_cast<const int*>(q_i8 + qh1 * kHeadDim);

    for (int ti = warp; ti < tile_len; ti += 8) {
        const int t = start + ti;
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const int* k_pack = reinterpret_cast<const int*>(k_i8 + kv_base);
        const int kval = k_pack[lane];
        float dot0 = warp_sum(static_cast<float>(__dp4a(q0_pack[lane], kval, 0)));
        float dot1 = warp_sum(static_cast<float>(__dp4a(q1_pack[lane], kval, 0)));
        if (lane == 0) {
            const float ks = k_scale[t * kNumKVHeads + kvh] * softmax_scale;
            scores0[ti] = dot0 * qs0 * ks;
            scores1[ti] = dot1 * qs1 * ks;
        }
    }
    __syncthreads();

    float m0 = kNegInf, l0 = 0.0f, m1 = kNegInf, l1 = 0.0f;
    tile_softmax_weights_parallel(scores0, scores1, tile_len, m0, l0, m1, l1);
    if (tid == 0) {
        m_l[0] = m0;
        m_l[1] = l0;
        m_l[2] = m1;
        m_l[3] = l1;
        partial_max[qh0 * max_splits + split] = m0;
        partial_sum[qh0 * max_splits + split] = l0;
        partial_max[qh1 * max_splits + split] = m1;
        partial_sum[qh1 * max_splits + split] = l1;
    }
    __syncthreads();

    if (tid < kHeadDim) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int ti = 0; ti < tile_len; ++ti) {
            const int kv_base = ((start + ti) * kNumKVHeads + kvh) * kHeadDim;
            const float value = bf16_to_float(v[kv_base + tid]);
            acc0 += scores0[ti] * value;
            acc1 += scores1[ti] * value;
        }
        partial_out[(qh0 * max_splits + split) * kHeadDim + tid] = acc0;
        partial_out[(qh1 * max_splits + split) * kHeadDim + tid] = acc1;
    }
}

__global__ void decode_q_i8_k_i8_v_bf16_split_partial_oneq_kernel(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ q_scale,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    int max_splits,
    float softmax_scale,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out) {
    __shared__ float scores[kOptBlockKV];
    const int qh = blockIdx.x;
    const int split = blockIdx.y;
    const int kvh = qh / kGqa;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_len = cache_seqlens[0];
    const int start = split * kOptBlockKV;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + kOptBlockKV, kv_len);
    const int tile_len = end - start;
    const float qs = q_scale[qh];
    const int* q_pack = reinterpret_cast<const int*>(q_i8 + qh * kHeadDim);

    for (int ti = warp; ti < tile_len; ti += 8) {
        const int t = start + ti;
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const int* k_pack = reinterpret_cast<const int*>(k_i8 + kv_base);
        const int kval = k_pack[lane];
        float dot = warp_sum(static_cast<float>(__dp4a(q_pack[lane], kval, 0)));
        if (lane == 0) {
            scores[ti] = dot * qs * k_scale[t * kNumKVHeads + kvh] * softmax_scale;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float m = kNegInf;
        for (int i = 0; i < tile_len; ++i) {
            m = fmaxf(m, scores[i]);
        }
        float l = 0.0f;
        for (int i = 0; i < tile_len; ++i) {
            const float e = expf(scores[i] - m);
            scores[i] = e;
            l += e;
        }
        partial_max[qh * max_splits + split] = m;
        partial_sum[qh * max_splits + split] = l;
    }
    __syncthreads();

    if (tid < kHeadDim) {
        float acc = 0.0f;
        for (int ti = 0; ti < tile_len; ++ti) {
            const int kv_base = ((start + ti) * kNumKVHeads + kvh) * kHeadDim;
            acc += scores[ti] * bf16_to_float(v[kv_base + tid]);
        }
        partial_out[(qh * max_splits + split) * kHeadDim + tid] = acc;
    }
}

__global__ void decode_q_i8_k_i8_v_bf16_split_partial_opt_bf16po_kernel(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ q_scale,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    int max_splits,
    float softmax_scale,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    __nv_bfloat16* __restrict__ partial_out) {
    __shared__ float scores0[kOptBlockKV];
    __shared__ float scores1[kOptBlockKV];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_len = cache_seqlens[0];
    const int start = split * kOptBlockKV;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + kOptBlockKV, kv_len);
    const int tile_len = end - start;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const float qs0 = q_scale[qh0];
    const float qs1 = q_scale[qh1];
    const int* q0_pack = reinterpret_cast<const int*>(q_i8 + qh0 * kHeadDim);
    const int* q1_pack = reinterpret_cast<const int*>(q_i8 + qh1 * kHeadDim);

    for (int ti = warp; ti < tile_len; ti += 8) {
        const int t = start + ti;
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const int* k_pack = reinterpret_cast<const int*>(k_i8 + kv_base);
        const int kval = k_pack[lane];
        float dot0 = warp_sum(static_cast<float>(__dp4a(q0_pack[lane], kval, 0)));
        float dot1 = warp_sum(static_cast<float>(__dp4a(q1_pack[lane], kval, 0)));
        if (lane == 0) {
            const float ks = k_scale[t * kNumKVHeads + kvh] * softmax_scale;
            scores0[ti] = dot0 * qs0 * ks;
            scores1[ti] = dot1 * qs1 * ks;
        }
    }
    __syncthreads();

    float m0 = kNegInf, l0 = 0.0f, m1 = kNegInf, l1 = 0.0f;
    tile_softmax_weights_parallel(scores0, scores1, tile_len, m0, l0, m1, l1);
    if (tid == 0) {
        partial_max[qh0 * max_splits + split] = m0;
        partial_sum[qh0 * max_splits + split] = l0;
        partial_max[qh1 * max_splits + split] = m1;
        partial_sum[qh1 * max_splits + split] = l1;
    }
    __syncthreads();

    if (tid < kHeadDim) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int ti = 0; ti < tile_len; ++ti) {
            const int kv_base = ((start + ti) * kNumKVHeads + kvh) * kHeadDim;
            const float value = bf16_to_float(v[kv_base + tid]);
            acc0 += scores0[ti] * value;
            acc1 += scores1[ti] * value;
        }
        partial_out[(qh0 * max_splits + split) * kHeadDim + tid] = __float2bfloat16(acc0);
        partial_out[(qh1 * max_splits + split) * kHeadDim + tid] = __float2bfloat16(acc1);
    }
}

__global__ void decode_q_i8_k_i8_v_i8_split_partial_opt_bf16po_kernel(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ q_scale,
    const signed char* __restrict__ k_i8,
    const signed char* __restrict__ v_i8,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int* __restrict__ cache_seqlens,
    int max_splits,
    float softmax_scale,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    __nv_bfloat16* __restrict__ partial_out) {
    __shared__ float scores0[kOptBlockKV];
    __shared__ float scores1[kOptBlockKV];
    __shared__ float v_scales[kOptBlockKV];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_len = cache_seqlens[0];
    const int start = split * kOptBlockKV;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + kOptBlockKV, kv_len);
    const int tile_len = end - start;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const float qs0 = q_scale[qh0];
    const float qs1 = q_scale[qh1];
    const int* q0_pack = reinterpret_cast<const int*>(q_i8 + qh0 * kHeadDim);
    const int* q1_pack = reinterpret_cast<const int*>(q_i8 + qh1 * kHeadDim);

    for (int ti = warp; ti < tile_len; ti += 8) {
        const int t = start + ti;
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const int* k_pack = reinterpret_cast<const int*>(k_i8 + kv_base);
        const int kval = k_pack[lane];
        float dot0 = warp_sum(static_cast<float>(__dp4a(q0_pack[lane], kval, 0)));
        float dot1 = warp_sum(static_cast<float>(__dp4a(q1_pack[lane], kval, 0)));
        if (lane == 0) {
            const float ks = k_scale[t * kNumKVHeads + kvh] * softmax_scale;
            scores0[ti] = dot0 * qs0 * ks;
            scores1[ti] = dot1 * qs1 * ks;
            v_scales[ti] = v_scale[t * kNumKVHeads + kvh];
        }
    }
    __syncthreads();

    float m0 = kNegInf, l0 = 0.0f, m1 = kNegInf, l1 = 0.0f;
    tile_softmax_weights_parallel(scores0, scores1, tile_len, m0, l0, m1, l1);
    if (tid == 0) {
        partial_max[qh0 * max_splits + split] = m0;
        partial_sum[qh0 * max_splits + split] = l0;
        partial_max[qh1 * max_splits + split] = m1;
        partial_sum[qh1 * max_splits + split] = l1;
    }
    __syncthreads();

    if (tid < kHeadDim) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int ti = 0; ti < tile_len; ++ti) {
            const int t = start + ti;
            const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
            const float value = static_cast<float>(v_i8[kv_base + tid]) * v_scales[ti];
            acc0 += scores0[ti] * value;
            acc1 += scores1[ti] * value;
        }
        partial_out[(qh0 * max_splits + split) * kHeadDim + tid] = __float2bfloat16(acc0);
        partial_out[(qh1 * max_splits + split) * kHeadDim + tid] = __float2bfloat16(acc1);
    }
}

__global__ void decode_q_i8_k_i8_v_bf16_split_partial_128_kernel(
    const signed char* __restrict__ q_i8,
    const float* __restrict__ q_scale,
    const signed char* __restrict__ k_i8,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    int max_splits,
    float softmax_scale,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out) {
    __shared__ float scores0[kOptBlockKV128];
    __shared__ float scores1[kOptBlockKV128];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int kv_len = cache_seqlens[0];
    const int start = split * kOptBlockKV128;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + kOptBlockKV128, kv_len);
    const int tile_len = end - start;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const float qs0 = q_scale[qh0];
    const float qs1 = q_scale[qh1];
    const int* q0_pack = reinterpret_cast<const int*>(q_i8 + qh0 * kHeadDim);
    const int* q1_pack = reinterpret_cast<const int*>(q_i8 + qh1 * kHeadDim);

    for (int ti = warp; ti < tile_len; ti += 16) {
        const int t = start + ti;
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const int* k_pack = reinterpret_cast<const int*>(k_i8 + kv_base);
        const int kval = k_pack[lane];
        float dot0 = warp_sum(static_cast<float>(__dp4a(q0_pack[lane], kval, 0)));
        float dot1 = warp_sum(static_cast<float>(__dp4a(q1_pack[lane], kval, 0)));
        if (lane == 0) {
            const float ks = k_scale[t * kNumKVHeads + kvh] * softmax_scale;
            scores0[ti] = dot0 * qs0 * ks;
            scores1[ti] = dot1 * qs1 * ks;
        }
    }
    __syncthreads();

    float m0 = kNegInf, l0 = 0.0f, m1 = kNegInf, l1 = 0.0f;
    tile_softmax_weights_parallel(scores0, scores1, tile_len, m0, l0, m1, l1);
    if (tid == 0) {
        partial_max[qh0 * max_splits + split] = m0;
        partial_sum[qh0 * max_splits + split] = l0;
        partial_max[qh1 * max_splits + split] = m1;
        partial_sum[qh1 * max_splits + split] = l1;
    }
    __syncthreads();

    if (tid < kHeadDim) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int ti = 0; ti < tile_len; ++ti) {
            const int kv_base = ((start + ti) * kNumKVHeads + kvh) * kHeadDim;
            const float value = bf16_to_float(v[kv_base + tid]);
            acc0 += scores0[ti] * value;
            acc1 += scores1[ti] * value;
        }
        partial_out[(qh0 * max_splits + split) * kHeadDim + tid] = acc0;
        partial_out[(qh1 * max_splits + split) * kHeadDim + tid] = acc1;
    }
}

__global__ void decode_qkv_k_i8_v_bf16_split_partial_kernel(
    const __nv_bfloat16* __restrict__ flat_qkv,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    signed char* __restrict__ k_i8,
    __nv_bfloat16* __restrict__ v,
    float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cache_seqlens_next,
    int max_splits,
    float softmax_scale,
    float q_eps,
    float k_eps,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int d = threadIdx.x;
    const int cache_pos = cache_seqlens[0];
    const int kv_len = cache_seqlens_next[0];
    const int start = split * 64;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + 64, kv_len);
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const int q_size = kNumQHeads * kHeadDim;
    const int kv_size = kNumKVHeads * kHeadDim;
    const __nv_bfloat16* q0_row = flat_qkv + qh0 * kHeadDim;
    const __nv_bfloat16* q1_row = flat_qkv + qh1 * kHeadDim;
    const __nv_bfloat16* k_row = flat_qkv + q_size + kvh * kHeadDim;
    const __nv_bfloat16* v_row = flat_qkv + q_size + kv_size + kvh * kHeadDim;

    const float q0_sq = bf16_to_float(q0_row[d]) * bf16_to_float(q0_row[d]);
    const float q1_sq = bf16_to_float(q1_row[d]) * bf16_to_float(q1_row[d]);
    const float k_sq = bf16_to_float(k_row[d]) * bf16_to_float(k_row[d]);
    float q0_sum = q0_sq;
    float q1_sum = q1_sq;
    float k_sum = k_sq;
    block_sum_triple(q0_sum, q1_sum, k_sum);
    const float q0_rstd = rsqrtf(q0_sum / static_cast<float>(kHeadDim) + q_eps);
    const float q1_rstd = rsqrtf(q1_sum / static_cast<float>(kHeadDim) + q_eps);
    const float k_rstd = rsqrtf(k_sum / static_cast<float>(kHeadDim) + k_eps);
    const float q0 = rope_value(q0_row, q_weight, cos, sin, d, q0_rstd);
    const float q1 = rope_value(q1_row, q_weight, cos, sin, d, q1_rstd);
    const float k_cur = rope_value(k_row, k_weight, cos, sin, d, k_rstd);
    const float k_abs_max = block_max(fabsf(k_cur));
    const float ks_cur = k_abs_max > 0.0f ? k_abs_max / 127.0f : 1.0f;
    const signed char k_cur_i8 = round_i8(k_cur / ks_cur);

    if (split == 0) {
        const int cur_base = (cache_pos * kNumKVHeads + kvh) * kHeadDim;
        k_i8[cur_base + d] = k_cur_i8;
        v[cur_base + d] = v_row[d];
        if (d == 0) {
            k_scale[cache_pos * kNumKVHeads + kvh] = ks_cur;
        }
    }

    float m0 = kNegInf;
    float m1 = kNegInf;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int t = start; t < end; ++t) {
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const bool is_current = t == cache_pos;
        const float ks = is_current ? ks_cur : k_scale[t * kNumKVHeads + kvh];
        const float kval = static_cast<float>(is_current ? k_cur_i8 : k_i8[kv_base + d]) * ks;
        float dot0 = q0 * kval;
        float dot1 = q1 * kval;
        block_sum_pair(dot0, dot1);
        const float value = is_current ? bf16_to_float(v_row[d]) : bf16_to_float(v[kv_base + d]);
        online_update(dot0 * softmax_scale, value, m0, l0, acc0);
        online_update(dot1 * softmax_scale, value, m1, l1, acc1);
    }

    partial_max[qh0 * max_splits + split] = m0;
    partial_sum[qh0 * max_splits + split] = l0;
    partial_out[(qh0 * max_splits + split) * kHeadDim + d] = acc0;
    partial_max[qh1 * max_splits + split] = m1;
    partial_sum[qh1 * max_splits + split] = l1;
    partial_out[(qh1 * max_splits + split) * kHeadDim + d] = acc1;
}

__global__ void decode_qkv_k_i8_v_bf16_split_partial_opt_kernel(
    const __nv_bfloat16* __restrict__ flat_qkv,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    signed char* __restrict__ k_i8,
    __nv_bfloat16* __restrict__ v,
    float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    const int* __restrict__ cache_seqlens_next,
    int max_splits,
    float softmax_scale,
    float q_eps,
    float k_eps,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    float* __restrict__ partial_out) {
    __shared__ signed char kcur_s[kHeadDim];
    __shared__ signed char q0_i8_s[kHeadDim];
    __shared__ signed char q1_i8_s[kHeadDim];
    __shared__ int q0_pack_s[kHeadDim / 4];
    __shared__ int q1_pack_s[kHeadDim / 4];
    __shared__ int kcur_pack_s[kHeadDim / 4];
    __shared__ float q_scale_s[2];
    __shared__ float scores0[kOptBlockKV];
    __shared__ float scores1[kOptBlockKV];
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int cache_pos = cache_seqlens[0];
    const int kv_len = cache_seqlens_next[0];
    const int start = split * kOptBlockKV;
    if (start >= kv_len) {
        return;
    }
    const int end = min(start + kOptBlockKV, kv_len);
    const int tile_len = end - start;
    const int qh0 = kvh * kGqa;
    const int qh1 = qh0 + 1;
    const int q_size = kNumQHeads * kHeadDim;
    const int kv_size = kNumKVHeads * kHeadDim;
    const __nv_bfloat16* q0_row = flat_qkv + qh0 * kHeadDim;
    const __nv_bfloat16* q1_row = flat_qkv + qh1 * kHeadDim;
    const __nv_bfloat16* k_row = flat_qkv + q_size + kvh * kHeadDim;
    const __nv_bfloat16* v_row = flat_qkv + q_size + kv_size + kvh * kHeadDim;

    const int d0 = tid;
    float q0_sq = 0.0f;
    float q1_sq = 0.0f;
    float k_sq = 0.0f;
    if (d0 < kHeadDim) {
        const float q0v = bf16_to_float(q0_row[d0]);
        const float q1v = bf16_to_float(q1_row[d0]);
        const float kv = bf16_to_float(k_row[d0]);
        q0_sq = q0v * q0v;
        q1_sq = q1v * q1v;
        k_sq = kv * kv;
    }
    float q0_sum = q0_sq;
    float q1_sum = q1_sq;
    float k_sum = k_sq;
    block_sum_triple(q0_sum, q1_sum, k_sum);
    const float q0_rstd = rsqrtf(q0_sum / static_cast<float>(kHeadDim) + q_eps);
    const float q1_rstd = rsqrtf(q1_sum / static_cast<float>(kHeadDim) + q_eps);
    const float k_rstd = rsqrtf(k_sum / static_cast<float>(kHeadDim) + k_eps);

    float k_abs = 0.0f;
    float q0_abs = 0.0f;
    float q1_abs = 0.0f;
    float k_cur_f = 0.0f;
    float q0_cur_f = 0.0f;
    float q1_cur_f = 0.0f;
    if (d0 < kHeadDim) {
        q0_cur_f = rope_value(q0_row, q_weight, cos, sin, d0, q0_rstd);
        q1_cur_f = rope_value(q1_row, q_weight, cos, sin, d0, q1_rstd);
        k_cur_f = rope_value(k_row, k_weight, cos, sin, d0, k_rstd);
        q0_abs = fabsf(q0_cur_f);
        q1_abs = fabsf(q1_cur_f);
        k_abs = fabsf(k_cur_f);
    }
    const float q0_abs_max = block_max(q0_abs);
    const float q1_abs_max = block_max(q1_abs);
    const float k_abs_max = block_max(k_abs);
    const float q0_scale = q0_abs_max > 0.0f ? q0_abs_max / 127.0f : 1.0f;
    const float q1_scale = q1_abs_max > 0.0f ? q1_abs_max / 127.0f : 1.0f;
    const float ks_cur = k_abs_max > 0.0f ? k_abs_max / 127.0f : 1.0f;
    if (d0 < kHeadDim) {
        q0_i8_s[d0] = round_i8(q0_cur_f / q0_scale);
        q1_i8_s[d0] = round_i8(q1_cur_f / q1_scale);
        const signed char kq = round_i8(k_cur_f / ks_cur);
        kcur_s[d0] = kq;
        if (split == 0) {
            const int cur_base = (cache_pos * kNumKVHeads + kvh) * kHeadDim;
            k_i8[cur_base + d0] = kq;
            v[cur_base + d0] = v_row[d0];
            if (d0 == 0) {
                k_scale[cache_pos * kNumKVHeads + kvh] = ks_cur;
            }
        }
    }
    __syncthreads();
    if (tid < kHeadDim / 4) {
        const int base = tid * 4;
        q0_pack_s[tid] = pack_i8x4(q0_i8_s[base], q0_i8_s[base + 1], q0_i8_s[base + 2], q0_i8_s[base + 3]);
        q1_pack_s[tid] = pack_i8x4(q1_i8_s[base], q1_i8_s[base + 1], q1_i8_s[base + 2], q1_i8_s[base + 3]);
        kcur_pack_s[tid] = pack_i8x4(kcur_s[base], kcur_s[base + 1], kcur_s[base + 2], kcur_s[base + 3]);
    }
    if (tid == 0) {
        q_scale_s[0] = q0_scale;
        q_scale_s[1] = q1_scale;
    }
    __syncthreads();

    for (int ti = warp; ti < tile_len; ti += 8) {
        const int t = start + ti;
        const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
        const bool is_current = t == cache_pos;
        const float ks = is_current ? ks_cur : k_scale[t * kNumKVHeads + kvh];
        const int kpack = is_current ? kcur_pack_s[lane] : reinterpret_cast<const int*>(k_i8 + kv_base)[lane];
        float dot0 = static_cast<float>(__dp4a(q0_pack_s[lane], kpack, 0));
        float dot1 = static_cast<float>(__dp4a(q1_pack_s[lane], kpack, 0));
        dot0 = warp_sum(dot0);
        dot1 = warp_sum(dot1);
        if (lane == 0) {
            const float score_scale = ks * softmax_scale;
            scores0[ti] = dot0 * q_scale_s[0] * score_scale;
            scores1[ti] = dot1 * q_scale_s[1] * score_scale;
        }
    }
    __syncthreads();

    float m0 = kNegInf, l0 = 0.0f, m1 = kNegInf, l1 = 0.0f;
    tile_softmax_weights_parallel(scores0, scores1, tile_len, m0, l0, m1, l1);
    if (tid == 0) {
        partial_max[qh0 * max_splits + split] = m0;
        partial_sum[qh0 * max_splits + split] = l0;
        partial_max[qh1 * max_splits + split] = m1;
        partial_sum[qh1 * max_splits + split] = l1;
    }
    __syncthreads();

    if (tid < kHeadDim) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int ti = 0; ti < tile_len; ++ti) {
            const int t = start + ti;
            const int kv_base = (t * kNumKVHeads + kvh) * kHeadDim;
            const float value = (t == cache_pos) ? bf16_to_float(v_row[tid]) : bf16_to_float(v[kv_base + tid]);
            acc0 += scores0[ti] * value;
            acc1 += scores1[ti] * value;
        }
        partial_out[(qh0 * max_splits + split) * kHeadDim + tid] = acc0;
        partial_out[(qh1 * max_splits + split) * kHeadDim + tid] = acc1;
    }
}

__global__ void qkv_prequant_cache_k_i8_v_bf16_kernel(
    const __nv_bfloat16* __restrict__ flat_qkv,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    signed char* __restrict__ k_i8,
    __nv_bfloat16* __restrict__ v,
    float* __restrict__ k_scale,
    const int* __restrict__ cache_seqlens,
    signed char* __restrict__ q_i8,
    float* __restrict__ q_scale,
    float q_eps,
    float k_eps) {
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    const int q_size = kNumQHeads * kHeadDim;
    const int kv_size = kNumKVHeads * kHeadDim;
    const bool is_q = row < kNumQHeads;
    const int head = is_q ? row : (row - kNumQHeads);
    const __nv_bfloat16* row_ptr = is_q
        ? (flat_qkv + head * kHeadDim)
        : (flat_qkv + q_size + head * kHeadDim);
    const __nv_bfloat16* weight = is_q ? q_weight : k_weight;
    const float eps = is_q ? q_eps : k_eps;

    const float x = bf16_to_float(row_ptr[d]);
    float sumsq = x * x;
    sumsq = block_sum(sumsq);
    const float rstd = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
    const float roped = rope_value(row_ptr, weight, cos, sin, d, rstd);
    const float abs_max = block_max(fabsf(roped));
    const float scale = abs_max > 0.0f ? abs_max / 127.0f : 1.0f;
    const signed char q = round_i8(roped / scale);

    if (is_q) {
        q_i8[head * kHeadDim + d] = q;
        if (d == 0) {
            q_scale[head] = scale;
        }
    } else {
        const int cache_pos = cache_seqlens[0];
        const int cur_base = (cache_pos * kNumKVHeads + head) * kHeadDim;
        k_i8[cur_base + d] = q;
        v[cur_base + d] = flat_qkv[q_size + kv_size + head * kHeadDim + d];
        if (d == 0) {
            k_scale[cache_pos * kNumKVHeads + head] = scale;
        }
    }
}

__global__ void qkv_prequant_cache_k_i8_v_i8_kernel(
    const __nv_bfloat16* __restrict__ flat_qkv,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin,
    signed char* __restrict__ k_i8,
    signed char* __restrict__ v_i8,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
    const int* __restrict__ cache_seqlens,
    signed char* __restrict__ q_i8,
    float* __restrict__ q_scale,
    float q_eps,
    float k_eps) {
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    const int q_size = kNumQHeads * kHeadDim;
    const int kv_size = kNumKVHeads * kHeadDim;
    const int cache_pos = cache_seqlens[0];

    if (row < kNumQHeads + kNumKVHeads) {
        const bool is_q = row < kNumQHeads;
        const int head = is_q ? row : (row - kNumQHeads);
        const __nv_bfloat16* row_ptr = is_q
            ? (flat_qkv + head * kHeadDim)
            : (flat_qkv + q_size + head * kHeadDim);
        const __nv_bfloat16* weight = is_q ? q_weight : k_weight;
        const float eps = is_q ? q_eps : k_eps;

        const float x = bf16_to_float(row_ptr[d]);
        float sumsq = x * x;
        sumsq = block_sum(sumsq);
        const float rstd = rsqrtf(sumsq / static_cast<float>(kHeadDim) + eps);
        const float roped = rope_value(row_ptr, weight, cos, sin, d, rstd);
        const float abs_max = block_max(fabsf(roped));
        const float scale = abs_max > 0.0f ? abs_max / 127.0f : 1.0f;
        const signed char q = round_i8(roped / scale);

        if (is_q) {
            q_i8[head * kHeadDim + d] = q;
            if (d == 0) {
                q_scale[head] = scale;
            }
        } else {
            const int cur_base = (cache_pos * kNumKVHeads + head) * kHeadDim;
            k_i8[cur_base + d] = q;
            const float value = bf16_to_float(flat_qkv[q_size + kv_size + head * kHeadDim + d]);
            const float value_abs_max = block_max(fabsf(value));
            const float value_scale_cur = value_abs_max > 0.0f ? value_abs_max / 127.0f : 1.0f;
            v_i8[cur_base + d] = round_i8(value / value_scale_cur);
            if (d == 0) {
                k_scale[cache_pos * kNumKVHeads + head] = scale;
                v_scale[cache_pos * kNumKVHeads + head] = value_scale_cur;
            }
        }
        return;
    }
}

__global__ void decode_split_reduce_kernel(
    const int* __restrict__ cache_seqlens,
    int max_splits,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_out,
    __nv_bfloat16* __restrict__ out) {
    const int qh = blockIdx.x;
    const int d = threadIdx.x;
    const int kv_len = cache_seqlens[0];
    const int n_splits = (kv_len + kOptBlockKV - 1) / kOptBlockKV;
    float m_part = kNegInf;
    for (int split = d; split < n_splits; split += blockDim.x) {
        m_part = fmaxf(m_part, partial_max[qh * max_splits + split]);
    }
    const float global_m = block_max(m_part);
    float denom_part = 0.0f;
    for (int split = d; split < n_splits; split += blockDim.x) {
        denom_part += expf(partial_max[qh * max_splits + split] - global_m) * partial_sum[qh * max_splits + split];
    }
    const float denom = block_sum(denom_part);
    float acc = 0.0f;
    for (int split = 0; split < n_splits; ++split) {
        const float weight = expf(partial_max[qh * max_splits + split] - global_m);
        acc += weight * partial_out[(qh * max_splits + split) * kHeadDim + d];
    }
    out[qh * kHeadDim + d] = __float2bfloat16(acc / fmaxf(denom, 1.0e-20f));
}

__global__ void decode_split_reduce_128_kernel(
    const int* __restrict__ cache_seqlens,
    int max_splits,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_out,
    __nv_bfloat16* __restrict__ out) {
    const int qh = blockIdx.x;
    const int d = threadIdx.x;
    const int kv_len = cache_seqlens[0];
    const int n_splits = (kv_len + kOptBlockKV128 - 1) / kOptBlockKV128;
    float m_part = kNegInf;
    for (int split = d; split < n_splits; split += blockDim.x) {
        m_part = fmaxf(m_part, partial_max[qh * max_splits + split]);
    }
    const float global_m = block_max(m_part);
    float denom_part = 0.0f;
    for (int split = d; split < n_splits; split += blockDim.x) {
        denom_part += expf(partial_max[qh * max_splits + split] - global_m) * partial_sum[qh * max_splits + split];
    }
    const float denom = block_sum(denom_part);
    float acc = 0.0f;
    for (int split = 0; split < n_splits; ++split) {
        const float weight = expf(partial_max[qh * max_splits + split] - global_m);
        acc += weight * partial_out[(qh * max_splits + split) * kHeadDim + d];
    }
    out[qh * kHeadDim + d] = __float2bfloat16(acc / fmaxf(denom, 1.0e-20f));
}

__global__ void decode_split_reduce_bf16po_kernel(
    const int* __restrict__ cache_seqlens,
    int max_splits,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const __nv_bfloat16* __restrict__ partial_out,
    __nv_bfloat16* __restrict__ out) {
    const int qh = blockIdx.x;
    const int d = threadIdx.x;
    const int kv_len = cache_seqlens[0];
    const int n_splits = (kv_len + kOptBlockKV - 1) / kOptBlockKV;
    float m_part = kNegInf;
    for (int split = d; split < n_splits; split += blockDim.x) {
        m_part = fmaxf(m_part, partial_max[qh * max_splits + split]);
    }
    const float global_m = block_max(m_part);
    float denom_part = 0.0f;
    for (int split = d; split < n_splits; split += blockDim.x) {
        denom_part += expf(partial_max[qh * max_splits + split] - global_m) * partial_sum[qh * max_splits + split];
    }
    const float denom = block_sum(denom_part);
    float acc = 0.0f;
    for (int split = 0; split < n_splits; ++split) {
        const float weight = expf(partial_max[qh * max_splits + split] - global_m);
        acc += weight * bf16_to_float(partial_out[(qh * max_splits + split) * kHeadDim + d]);
    }
    out[qh * kHeadDim + d] = __float2bfloat16(acc / fmaxf(denom, 1.0e-20f));
}

void check_cuda(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_common_cache(
    const torch::Tensor& key_cache_i8,
    const torch::Tensor& value_cache,
    const torch::Tensor& key_scale,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& output) {
    check_cuda(key_cache_i8, "key_cache_i8");
    check_cuda(value_cache, "value_cache");
    check_cuda(key_scale, "key_scale");
    check_cuda(cache_seqlens, "cache_seqlens");
    check_cuda(output, "output");
    TORCH_CHECK(key_cache_i8.scalar_type() == at::kChar, "key_cache_i8 must be int8");
    TORCH_CHECK(value_cache.scalar_type() == at::kBFloat16, "value_cache must be BF16");
    TORCH_CHECK(key_scale.scalar_type() == at::kFloat, "key_scale must be FP32");
    TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt, "cache_seqlens must be int32");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(
        key_cache_i8.dim() == 3 && key_cache_i8.size(1) == kNumKVHeads && key_cache_i8.size(2) == kHeadDim,
        "key_cache_i8 must be [S,8,128]");
    TORCH_CHECK(value_cache.sizes() == key_cache_i8.sizes(), "value_cache must match key_cache_i8 shape");
    TORCH_CHECK(
        key_scale.dim() == 2 && key_scale.size(0) == key_cache_i8.size(0) && key_scale.size(1) == kNumKVHeads,
        "key_scale must be [S,8]");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == kNumQHeads && output.size(1) == kHeadDim, "output must be [16,128]");
    TORCH_CHECK(cache_seqlens.numel() >= 1, "cache_seqlens must have at least one element");
}

void check_common_cache_i8v(
    const torch::Tensor& key_cache_i8,
    const torch::Tensor& value_cache_i8,
    const torch::Tensor& key_scale,
    const torch::Tensor& value_scale,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& output) {
    check_cuda(key_cache_i8, "key_cache_i8");
    check_cuda(value_cache_i8, "value_cache_i8");
    check_cuda(key_scale, "key_scale");
    check_cuda(value_scale, "value_scale");
    check_cuda(cache_seqlens, "cache_seqlens");
    check_cuda(output, "output");
    TORCH_CHECK(key_cache_i8.scalar_type() == at::kChar, "key_cache_i8 must be int8");
    TORCH_CHECK(value_cache_i8.scalar_type() == at::kChar, "value_cache_i8 must be int8");
    TORCH_CHECK(key_scale.scalar_type() == at::kFloat, "key_scale must be FP32");
    TORCH_CHECK(value_scale.scalar_type() == at::kFloat, "value_scale must be FP32");
    TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt, "cache_seqlens must be int32");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(
        key_cache_i8.dim() == 3 && key_cache_i8.size(1) == kNumKVHeads && key_cache_i8.size(2) == kHeadDim,
        "key_cache_i8 must be [S,8,128]");
    TORCH_CHECK(value_cache_i8.sizes() == key_cache_i8.sizes(), "value_cache_i8 must match key_cache_i8 shape");
    TORCH_CHECK(
        key_scale.dim() == 2 && key_scale.size(0) == key_cache_i8.size(0) && key_scale.size(1) == kNumKVHeads,
        "key_scale must be [S,8]");
    TORCH_CHECK(value_scale.sizes() == key_scale.sizes(), "value_scale must match key_scale shape");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == kNumQHeads && output.size(1) == kHeadDim, "output must be [16,128]");
    TORCH_CHECK(cache_seqlens.numel() >= 1, "cache_seqlens must have at least one element");
}

int check_split_workspace(
    const torch::Tensor& partial_max,
    const torch::Tensor& partial_sum,
    const torch::Tensor& partial_out) {
    check_cuda(partial_max, "partial_max");
    check_cuda(partial_sum, "partial_sum");
    check_cuda(partial_out, "partial_out");
    TORCH_CHECK(partial_max.scalar_type() == at::kFloat, "partial_max must be FP32");
    TORCH_CHECK(partial_sum.scalar_type() == at::kFloat, "partial_sum must be FP32");
    TORCH_CHECK(partial_out.scalar_type() == at::kFloat, "partial_out must be FP32");
    TORCH_CHECK(partial_max.dim() == 2 && partial_max.size(0) == kNumQHeads, "partial_max must be [16,max_splits]");
    TORCH_CHECK(partial_sum.sizes() == partial_max.sizes(), "partial_sum must match partial_max");
    TORCH_CHECK(
        partial_out.dim() == 3
            && partial_out.size(0) == kNumQHeads
            && partial_out.size(1) == partial_max.size(1)
            && partial_out.size(2) == kHeadDim,
        "partial_out must be [16,max_splits,128]");
    return static_cast<int>(partial_max.size(1));
}

int check_split_workspace_bf16po(
    const torch::Tensor& partial_max,
    const torch::Tensor& partial_sum,
    const torch::Tensor& partial_out) {
    check_cuda(partial_max, "partial_max");
    check_cuda(partial_sum, "partial_sum");
    check_cuda(partial_out, "partial_out_bf16");
    TORCH_CHECK(partial_max.scalar_type() == at::kFloat, "partial_max must be FP32");
    TORCH_CHECK(partial_sum.scalar_type() == at::kFloat, "partial_sum must be FP32");
    TORCH_CHECK(partial_out.scalar_type() == at::kBFloat16, "partial_out_bf16 must be BF16");
    TORCH_CHECK(partial_max.dim() == 2 && partial_max.size(0) == kNumQHeads, "partial_max must be [16,max_splits]");
    TORCH_CHECK(partial_sum.sizes() == partial_max.sizes(), "partial_sum must match partial_max");
    TORCH_CHECK(
        partial_out.dim() == 3
            && partial_out.size(0) == kNumQHeads
            && partial_out.size(1) == partial_max.size(1)
            && partial_out.size(2) == kHeadDim,
        "partial_out_bf16 must be [16,max_splits,128]");
    return static_cast<int>(partial_max.size(1));
}

}  // namespace

void sage_decode_k_i8_v_bf16_single_cuda(
    torch::Tensor query,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor output,
    double softmax_scale) {
    check_cuda(query, "query");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "query must be BF16");
    TORCH_CHECK(query.dim() == 2 && query.size(0) == kNumQHeads && query.size(1) == kHeadDim, "query must be [16,128]");
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_k_i8_v_bf16_single_kernel<<<kNumKVHeads, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        static_cast<float>(softmax_scale),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_q_i8_k_i8_v_bf16_single_cuda(
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor output,
    double softmax_scale) {
    check_cuda(query_i8, "query_i8");
    check_cuda(query_scale, "query_scale");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    TORCH_CHECK(query_i8.scalar_type() == at::kChar, "query_i8 must be int8");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be FP32");
    TORCH_CHECK(query_i8.dim() == 2 && query_i8.size(0) == kNumQHeads && query_i8.size(1) == kHeadDim, "query_i8 must be [16,128]");
    TORCH_CHECK(query_scale.dim() == 1 && query_scale.size(0) == kNumQHeads, "query_scale must be [16]");
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_q_i8_k_i8_v_bf16_single_kernel<<<kNumKVHeads, kThreads, 0, stream>>>(
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        static_cast<float>(softmax_scale),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_qkv_k_i8_v_bf16_single_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps) {
    check_cuda(flat_qkv, "flat_qkv");
    check_cuda(q_weight, "q_weight");
    check_cuda(k_weight, "k_weight");
    check_cuda(cos, "cos");
    check_cuda(sin, "sin");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    check_cuda(cache_seqlens_next, "cache_seqlens_next");
    TORCH_CHECK(flat_qkv.scalar_type() == at::kBFloat16, "flat_qkv must be BF16");
    TORCH_CHECK(q_weight.scalar_type() == at::kBFloat16, "q_weight must be BF16");
    TORCH_CHECK(k_weight.scalar_type() == at::kBFloat16, "k_weight must be BF16");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16, "cos/sin must be BF16");
    TORCH_CHECK(cache_seqlens_next.scalar_type() == at::kInt, "cache_seqlens_next must be int32");
    TORCH_CHECK(flat_qkv.numel() >= (kNumQHeads + 2 * kNumKVHeads) * kHeadDim, "flat_qkv is too small");
    TORCH_CHECK(q_weight.numel() == kHeadDim && k_weight.numel() == kHeadDim, "q/k weights must be [128]");
    TORCH_CHECK(cos.numel() >= kHeadDim && sin.numel() >= kHeadDim, "cos/sin must have at least 128 elements");
    TORCH_CHECK(cache_seqlens_next.numel() >= 1, "cache_seqlens_next must have at least one element");
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_qkv_k_i8_v_bf16_single_kernel<<<kNumKVHeads, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        cache_seqlens_next.data_ptr<int>(),
        static_cast<float>(softmax_scale),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_k_i8_v_bf16_split_cuda(
    torch::Tensor query,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale) {
    check_cuda(query, "query");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    int max_splits = check_split_workspace(partial_max, partial_sum, partial_out);
    TORCH_CHECK(query.scalar_type() == at::kBFloat16, "query must be BF16");
    TORCH_CHECK(query.dim() == 2 && query.size(0) == kNumQHeads && query.size(1) == kHeadDim, "query must be [16,128]");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(kNumKVHeads, max_splits);
    decode_k_i8_v_bf16_split_partial_opt_kernel<<<grid, kOptThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        max_splits,
        static_cast<float>(softmax_scale),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode_split_reduce_kernel<<<kNumQHeads, kThreads, 0, stream>>>(
        cache_seqlens.data_ptr<int>(),
        max_splits,
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_q_i8_k_i8_v_bf16_split_cuda(
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale) {
    check_cuda(query_i8, "query_i8");
    check_cuda(query_scale, "query_scale");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    int max_splits = check_split_workspace(partial_max, partial_sum, partial_out);
    TORCH_CHECK(query_i8.scalar_type() == at::kChar, "query_i8 must be int8");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be FP32");
    TORCH_CHECK(query_i8.dim() == 2 && query_i8.size(0) == kNumQHeads && query_i8.size(1) == kHeadDim, "query_i8 must be [16,128]");
    TORCH_CHECK(query_scale.dim() == 1 && query_scale.size(0) == kNumQHeads, "query_scale must be [16]");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(kNumKVHeads, max_splits);
    decode_q_i8_k_i8_v_bf16_split_partial_opt_kernel<<<grid, kOptThreads, 0, stream>>>(
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        max_splits,
        static_cast<float>(softmax_scale),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode_split_reduce_kernel<<<kNumQHeads, kThreads, 0, stream>>>(
        cache_seqlens.data_ptr<int>(),
        max_splits,
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_qkv_k_i8_v_bf16_split_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps) {
    check_cuda(flat_qkv, "flat_qkv");
    check_cuda(q_weight, "q_weight");
    check_cuda(k_weight, "k_weight");
    check_cuda(cos, "cos");
    check_cuda(sin, "sin");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    check_cuda(cache_seqlens_next, "cache_seqlens_next");
    int max_splits = check_split_workspace(partial_max, partial_sum, partial_out);
    TORCH_CHECK(flat_qkv.scalar_type() == at::kBFloat16, "flat_qkv must be BF16");
    TORCH_CHECK(q_weight.scalar_type() == at::kBFloat16, "q_weight must be BF16");
    TORCH_CHECK(k_weight.scalar_type() == at::kBFloat16, "k_weight must be BF16");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16, "cos/sin must be BF16");
    TORCH_CHECK(cache_seqlens_next.scalar_type() == at::kInt, "cache_seqlens_next must be int32");
    TORCH_CHECK(flat_qkv.numel() >= (kNumQHeads + 2 * kNumKVHeads) * kHeadDim, "flat_qkv is too small");
    TORCH_CHECK(q_weight.numel() == kHeadDim && k_weight.numel() == kHeadDim, "q/k weights must be [128]");
    TORCH_CHECK(cos.numel() >= kHeadDim && sin.numel() >= kHeadDim, "cos/sin must have at least 128 elements");
    TORCH_CHECK(cache_seqlens_next.numel() >= 1, "cache_seqlens_next must have at least one element");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(kNumKVHeads, max_splits);
    decode_qkv_k_i8_v_bf16_split_partial_opt_kernel<<<grid, kOptThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        static_cast<float>(softmax_scale),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode_split_reduce_kernel<<<kNumQHeads, kThreads, 0, stream>>>(
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_qkv_prequant_k_i8_v_bf16_split_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps) {
    check_cuda(flat_qkv, "flat_qkv");
    check_cuda(q_weight, "q_weight");
    check_cuda(k_weight, "k_weight");
    check_cuda(cos, "cos");
    check_cuda(sin, "sin");
    check_cuda(query_i8, "query_i8");
    check_cuda(query_scale, "query_scale");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    check_cuda(cache_seqlens_next, "cache_seqlens_next");
    int max_splits = check_split_workspace(partial_max, partial_sum, partial_out);
    TORCH_CHECK(flat_qkv.scalar_type() == at::kBFloat16, "flat_qkv must be BF16");
    TORCH_CHECK(q_weight.scalar_type() == at::kBFloat16, "q_weight must be BF16");
    TORCH_CHECK(k_weight.scalar_type() == at::kBFloat16, "k_weight must be BF16");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16, "cos/sin must be BF16");
    TORCH_CHECK(cache_seqlens_next.scalar_type() == at::kInt, "cache_seqlens_next must be int32");
    TORCH_CHECK(query_i8.scalar_type() == at::kChar, "query_i8 must be int8");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be FP32");
    TORCH_CHECK(flat_qkv.numel() >= (kNumQHeads + 2 * kNumKVHeads) * kHeadDim, "flat_qkv is too small");
    TORCH_CHECK(q_weight.numel() == kHeadDim && k_weight.numel() == kHeadDim, "q/k weights must be [128]");
    TORCH_CHECK(cos.numel() >= kHeadDim && sin.numel() >= kHeadDim, "cos/sin must have at least 128 elements");
    TORCH_CHECK(cache_seqlens_next.numel() >= 1, "cache_seqlens_next must have at least one element");
    TORCH_CHECK(query_i8.dim() == 2 && query_i8.size(0) == kNumQHeads && query_i8.size(1) == kHeadDim, "query_i8 must be [16,128]");
    TORCH_CHECK(query_scale.dim() == 1 && query_scale.size(0) == kNumQHeads, "query_scale must be [16]");
    auto stream = at::cuda::getCurrentCUDAStream();
    qkv_prequant_cache_k_i8_v_bf16_kernel<<<kNumQHeads + kNumKVHeads, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 grid(kNumKVHeads, max_splits);
    decode_q_i8_k_i8_v_bf16_split_partial_opt_kernel<<<grid, kOptThreads, 0, stream>>>(
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        static_cast<float>(softmax_scale),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode_split_reduce_kernel<<<kNumQHeads, kThreads, 0, stream>>>(
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_qkv_prequant_k_i8_v_bf16_split_bf16po_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out_bf16,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps) {
    check_cuda(flat_qkv, "flat_qkv");
    check_cuda(q_weight, "q_weight");
    check_cuda(k_weight, "k_weight");
    check_cuda(cos, "cos");
    check_cuda(sin, "sin");
    check_cuda(query_i8, "query_i8");
    check_cuda(query_scale, "query_scale");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    check_cuda(cache_seqlens_next, "cache_seqlens_next");
    int max_splits = check_split_workspace_bf16po(partial_max, partial_sum, partial_out_bf16);
    TORCH_CHECK(flat_qkv.scalar_type() == at::kBFloat16, "flat_qkv must be BF16");
    TORCH_CHECK(q_weight.scalar_type() == at::kBFloat16, "q_weight must be BF16");
    TORCH_CHECK(k_weight.scalar_type() == at::kBFloat16, "k_weight must be BF16");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16, "cos/sin must be BF16");
    TORCH_CHECK(cache_seqlens_next.scalar_type() == at::kInt, "cache_seqlens_next must be int32");
    TORCH_CHECK(query_i8.scalar_type() == at::kChar, "query_i8 must be int8");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be FP32");
    auto stream = at::cuda::getCurrentCUDAStream();
    qkv_prequant_cache_k_i8_v_bf16_kernel<<<kNumQHeads + kNumKVHeads, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 grid(kNumKVHeads, max_splits);
    decode_q_i8_k_i8_v_bf16_split_partial_opt_bf16po_kernel<<<grid, kOptThreads, 0, stream>>>(
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        static_cast<float>(softmax_scale),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(partial_out_bf16.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode_split_reduce_bf16po_kernel<<<kNumQHeads, kThreads, 0, stream>>>(
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(partial_out_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_qkv_prequant_k_i8_v_i8_split_bf16po_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache_i8,
    torch::Tensor key_scale,
    torch::Tensor value_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out_bf16,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps) {
    check_cuda(flat_qkv, "flat_qkv");
    check_cuda(q_weight, "q_weight");
    check_cuda(k_weight, "k_weight");
    check_cuda(cos, "cos");
    check_cuda(sin, "sin");
    check_cuda(query_i8, "query_i8");
    check_cuda(query_scale, "query_scale");
    check_common_cache_i8v(key_cache_i8, value_cache_i8, key_scale, value_scale, cache_seqlens, output);
    check_cuda(cache_seqlens_next, "cache_seqlens_next");
    int max_splits = check_split_workspace_bf16po(partial_max, partial_sum, partial_out_bf16);
    TORCH_CHECK(flat_qkv.scalar_type() == at::kBFloat16, "flat_qkv must be BF16");
    TORCH_CHECK(q_weight.scalar_type() == at::kBFloat16, "q_weight must be BF16");
    TORCH_CHECK(k_weight.scalar_type() == at::kBFloat16, "k_weight must be BF16");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16, "cos/sin must be BF16");
    TORCH_CHECK(cache_seqlens_next.scalar_type() == at::kInt, "cache_seqlens_next must be int32");
    TORCH_CHECK(query_i8.scalar_type() == at::kChar, "query_i8 must be int8");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be FP32");
    TORCH_CHECK(flat_qkv.numel() >= (kNumQHeads + 2 * kNumKVHeads) * kHeadDim, "flat_qkv is too small");
    TORCH_CHECK(q_weight.numel() == kHeadDim && k_weight.numel() == kHeadDim, "q/k weights must be [128]");
    TORCH_CHECK(cos.numel() >= kHeadDim && sin.numel() >= kHeadDim, "cos/sin must have at least 128 elements");
    TORCH_CHECK(cache_seqlens_next.numel() >= 1, "cache_seqlens_next must have at least one element");
    TORCH_CHECK(query_i8.dim() == 2 && query_i8.size(0) == kNumQHeads && query_i8.size(1) == kHeadDim, "query_i8 must be [16,128]");
    TORCH_CHECK(query_scale.dim() == 1 && query_scale.size(0) == kNumQHeads, "query_scale must be [16]");
    auto stream = at::cuda::getCurrentCUDAStream();
    qkv_prequant_cache_k_i8_v_i8_kernel<<<kNumQHeads + kNumKVHeads, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        key_cache_i8.data_ptr<signed char>(),
        value_cache_i8.data_ptr<signed char>(),
        key_scale.data_ptr<float>(),
        value_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 grid(kNumKVHeads, max_splits);
    decode_q_i8_k_i8_v_i8_split_partial_opt_bf16po_kernel<<<grid, kOptThreads, 0, stream>>>(
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        key_cache_i8.data_ptr<signed char>(),
        value_cache_i8.data_ptr<signed char>(),
        key_scale.data_ptr<float>(),
        value_scale.data_ptr<float>(),
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        static_cast<float>(softmax_scale),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(partial_out_bf16.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode_split_reduce_bf16po_kernel<<<kNumQHeads, kThreads, 0, stream>>>(
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(partial_out_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sage_decode_qkv_prequant_k_i8_v_bf16_split128_cuda(
    torch::Tensor flat_qkv,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor key_cache_i8,
    torch::Tensor value_cache,
    torch::Tensor key_scale,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor query_i8,
    torch::Tensor query_scale,
    torch::Tensor partial_max,
    torch::Tensor partial_sum,
    torch::Tensor partial_out,
    torch::Tensor output,
    double softmax_scale,
    double q_eps,
    double k_eps) {
    check_cuda(flat_qkv, "flat_qkv");
    check_cuda(q_weight, "q_weight");
    check_cuda(k_weight, "k_weight");
    check_cuda(cos, "cos");
    check_cuda(sin, "sin");
    check_cuda(query_i8, "query_i8");
    check_cuda(query_scale, "query_scale");
    check_common_cache(key_cache_i8, value_cache, key_scale, cache_seqlens, output);
    check_cuda(cache_seqlens_next, "cache_seqlens_next");
    int max_splits = check_split_workspace(partial_max, partial_sum, partial_out);
    TORCH_CHECK(flat_qkv.scalar_type() == at::kBFloat16, "flat_qkv must be BF16");
    TORCH_CHECK(q_weight.scalar_type() == at::kBFloat16, "q_weight must be BF16");
    TORCH_CHECK(k_weight.scalar_type() == at::kBFloat16, "k_weight must be BF16");
    TORCH_CHECK(cos.scalar_type() == at::kBFloat16 && sin.scalar_type() == at::kBFloat16, "cos/sin must be BF16");
    TORCH_CHECK(cache_seqlens_next.scalar_type() == at::kInt, "cache_seqlens_next must be int32");
    TORCH_CHECK(query_i8.scalar_type() == at::kChar, "query_i8 must be int8");
    TORCH_CHECK(query_scale.scalar_type() == at::kFloat, "query_scale must be FP32");
    auto stream = at::cuda::getCurrentCUDAStream();
    qkv_prequant_cache_k_i8_v_bf16_kernel<<<kNumQHeads + kNumKVHeads, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(flat_qkv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<__nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens.data_ptr<int>(),
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        static_cast<float>(q_eps),
        static_cast<float>(k_eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int tile128_splits = (static_cast<int>(key_cache_i8.size(0)) + kOptBlockKV128 - 1) / kOptBlockKV128;
    dim3 grid(kNumKVHeads, tile128_splits);
    decode_q_i8_k_i8_v_bf16_split_partial_128_kernel<<<grid, kOptThreads128, 0, stream>>>(
        query_i8.data_ptr<signed char>(),
        query_scale.data_ptr<float>(),
        key_cache_i8.data_ptr<signed char>(),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data_ptr<at::BFloat16>()),
        key_scale.data_ptr<float>(),
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        static_cast<float>(softmax_scale),
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    decode_split_reduce_128_kernel<<<kNumQHeads, kThreads, 0, stream>>>(
        cache_seqlens_next.data_ptr<int>(),
        max_splits,
        partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        partial_out.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
