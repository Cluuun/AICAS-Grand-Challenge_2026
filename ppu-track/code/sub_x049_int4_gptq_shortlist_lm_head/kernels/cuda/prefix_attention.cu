#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be fp32")

constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kThreads = 128;
constexpr int kMaxThreads = 1024;

__forceinline__ __device__ float warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__forceinline__ __device__ float warp_max(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(mask, value, offset));
  }
  return value;
}

__forceinline__ __device__ float qk_dot_half2_128(
    const half* __restrict__ query_head,
    const half* __restrict__ key_row,
    int lane) {
  const half2* __restrict__ query2 = reinterpret_cast<const half2*>(query_head);
  const half2* __restrict__ key2 = reinterpret_cast<const half2*>(key_row);
  float2 q01 = __half22float2(query2[lane]);
  float2 k01 = __half22float2(key2[lane]);
  float2 q23 = __half22float2(query2[lane + 32]);
  float2 k23 = __half22float2(key2[lane + 32]);
  float sum = q01.x * k01.x;
  sum = fmaf(q01.y, k01.y, sum);
  sum = fmaf(q23.x, k23.x, sum);
  sum = fmaf(q23.y, k23.y, sum);
  return sum;
}

__forceinline__ __device__ float qk_dot_half2_cached_query_128(
    float2 q01,
    float2 q23,
    const half* __restrict__ key_row,
    int lane) {
  const half2* __restrict__ key2 = reinterpret_cast<const half2*>(key_row);
  float2 k01 = __half22float2(key2[lane]);
  float2 k23 = __half22float2(key2[lane + 32]);
  float sum = q01.x * k01.x;
  sum = fmaf(q01.y, k01.y, sum);
  sum = fmaf(q23.x, k23.x, sum);
  sum = fmaf(q23.y, k23.y, sum);
  return sum;
}

template <int SplitM>
__global__ void prefix_attention_split_stage1_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int max_k,
    int num_splits,
    float scale) {
  int row = blockIdx.x;
  int q_head = row / num_splits;
  int split_idx = row - q_head * num_splits;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  int num_warps = blockDim.x >> 5;
  int kv_head = q_head >> 1;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  if (tile_len <= 0) {
    if (tid == 0) {
      int state_idx = q_head * num_splits + split_idx;
      partial_m[state_idx] = -INFINITY;
      partial_l[state_idx] = 0.0f;
    }
    return;
  }

  __shared__ half q_s[kHeadDim];
  __shared__ float scores[SplitM];

  int q_base = q_head * kHeadDim;
  for (int d = tid; d < kHeadDim; d += blockDim.x) {
    q_s[d] = query[q_base + d];
  }
  __syncthreads();

  float local_m = -INFINITY;
  for (int m = warp_id; m < tile_len; m += num_warps) {
    int k_base = (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum = 0.0f;
#pragma unroll
    for (int d = lane; d < kHeadDim; d += 32) {
      sum += __half2float(q_s[d]) * __half2float(key_cache[k_base + d]);
    }
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  if (tid == 0) {
    float m = -INFINITY;
    for (int i = 0; i < tile_len; ++i) {
      m = fmaxf(m, scores[i]);
    }
    float l = 0.0f;
    for (int i = 0; i < tile_len; ++i) {
      float p = __expf(scores[i] - m);
      scores[i] = p;
      l += p;
    }
    local_m = m;
    int state_idx = q_head * num_splits + split_idx;
    partial_m[state_idx] = m;
    partial_l[state_idx] = l;
  }
  __syncthreads();

  int acc_base = (q_head * num_splits + split_idx) * kHeadDim;
  for (int d = tid; d < kHeadDim; d += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < tile_len; ++i) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + d]);
    }
    partial_acc[acc_base + d] = acc;
  }
}

template <int SplitM, int Threads>
__global__ void prefix_attention_split_stage1_vgroup_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int max_k,
    int workspace_splits,
    int active_splits,
    float scale) {
  int row = blockIdx.x;
  int q_head = row / active_splits;
  int split_idx = row - q_head * active_splits;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;
  int kv_head = q_head >> 1;

  constexpr int lanes_per_dim = Threads >> 7;  // 128->1, 256->2, 512->4, 1024->8.
  constexpr int dims_per_warp = 32 / lanes_per_dim;
  int dim_group = lane / lanes_per_dim;
  int lane_in_dim = lane - dim_group * lanes_per_dim;
  int out_dim = warp_id * dims_per_warp + dim_group;
  bool dim_leader = lane_in_dim == 0;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  if (tile_len <= 0) {
    if (tid == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = -INFINITY;
      partial_l[state_idx] = 0.0f;
    }
    return;
  }

  __shared__ float scores[SplitM];
  __shared__ float shared_state[2];

  int q_base = q_head * kHeadDim;
  float q0 = __half2float(query[q_base + lane]);
  float q1 = __half2float(query[q_base + lane + 32]);
  float q2 = __half2float(query[q_base + lane + 64]);
  float q3 = __half2float(query[q_base + lane + 96]);

  for (int m = warp_id; m < tile_len; m += num_warps) {
    int k_base = (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum =
        q0 * __half2float(key_cache[k_base + lane]) +
        q1 * __half2float(key_cache[k_base + lane + 32]) +
        q2 * __half2float(key_cache[k_base + lane + 64]) +
        q3 * __half2float(key_cache[k_base + lane + 96]);
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_max = -INFINITY;
    for (int i = lane; i < tile_len; i += 32) {
      local_max = fmaxf(local_max, scores[i]);
    }
    float m = warp_max(local_max);
    m = __shfl_sync(0xffffffffu, m, 0);
    float local_l = 0.0f;
    for (int i = lane; i < tile_len; i += 32) {
      float p = __expf(scores[i] - m);
      scores[i] = p;
      local_l += p;
    }
    float l = warp_sum(local_l);
    if (lane == 0) {
      shared_state[0] = m;
      shared_state[1] = l;
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = m;
      partial_l[state_idx] = l;
    }
  }
  __syncthreads();

  if (out_dim < kHeadDim) {
    float acc = 0.0f;
    for (int i = lane_in_dim; i < tile_len; i += lanes_per_dim) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + out_dim]);
    }
    int group_start = dim_group * lanes_per_dim;
    for (int offset = lanes_per_dim >> 1; offset > 0; offset >>= 1) {
      int src_lane = min(lane + offset, 31);
      float other = __shfl_sync(0xffffffffu, acc, src_lane);
      if (lane_in_dim < offset && lane + offset < group_start + lanes_per_dim) {
        acc += other;
      }
    }
    if (dim_leader) {
      int acc_base = (q_head * workspace_splits + split_idx) * kHeadDim;
      partial_acc[acc_base + out_dim] = acc;
    }
  }
}

template <int SplitM, int Threads>
__global__ void prefix_attention_split_stage1_vgroup_2d_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int max_k,
    int workspace_splits,
    float scale) {
  int split_idx = blockIdx.x;
  int q_head = blockIdx.y;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;
  int kv_head = q_head >> 1;

  constexpr int lanes_per_dim = Threads >> 7;  // 128->1, 256->2, 512->4, 1024->8.
  constexpr int dims_per_warp = 32 / lanes_per_dim;
  int dim_group = lane / lanes_per_dim;
  int lane_in_dim = lane - dim_group * lanes_per_dim;
  int out_dim = warp_id * dims_per_warp + dim_group;
  bool dim_leader = lane_in_dim == 0;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  if (tile_len <= 0) {
    if (tid == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = -INFINITY;
      partial_l[state_idx] = 0.0f;
    }
    return;
  }

  __shared__ float scores[SplitM];

  int q_base = q_head * kHeadDim;
  float q0 = __half2float(query[q_base + lane]);
  float q1 = __half2float(query[q_base + lane + 32]);
  float q2 = __half2float(query[q_base + lane + 64]);
  float q3 = __half2float(query[q_base + lane + 96]);

  for (int m = warp_id; m < tile_len; m += num_warps) {
    int k_base = (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum =
        q0 * __half2float(key_cache[k_base + lane]) +
        q1 * __half2float(key_cache[k_base + lane + 32]) +
        q2 * __half2float(key_cache[k_base + lane + 64]) +
        q3 * __half2float(key_cache[k_base + lane + 96]);
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_max = -INFINITY;
    for (int i = lane; i < tile_len; i += 32) {
      local_max = fmaxf(local_max, scores[i]);
    }
    float m = warp_max(local_max);
    m = __shfl_sync(0xffffffffu, m, 0);
    float local_l = 0.0f;
    for (int i = lane; i < tile_len; i += 32) {
      float p = __expf(scores[i] - m);
      scores[i] = p;
      local_l += p;
    }
    float l = warp_sum(local_l);
    if (lane == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = m;
      partial_l[state_idx] = l;
    }
  }
  __syncthreads();

  if (out_dim < kHeadDim) {
    float acc = 0.0f;
    for (int i = lane_in_dim; i < tile_len; i += lanes_per_dim) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + out_dim]);
    }
    int group_start = dim_group * lanes_per_dim;
    for (int offset = lanes_per_dim >> 1; offset > 0; offset >>= 1) {
      int src_lane = min(lane + offset, 31);
      float other = __shfl_sync(0xffffffffu, acc, src_lane);
      if (lane_in_dim < offset && lane + offset < group_start + lanes_per_dim) {
        acc += other;
      }
    }
    if (dim_leader) {
      int acc_base = (q_head * workspace_splits + split_idx) * kHeadDim;
      partial_acc[acc_base + out_dim] = acc;
    }
  }
}

template <int SplitM, int Threads>
__global__ void prefix_attention_split_stage1_vgroup_h2dot_2d_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int max_k,
    int workspace_splits,
    float scale) {
  int split_idx = blockIdx.x;
  int q_head = blockIdx.y;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;
  int kv_head = q_head >> 1;

  constexpr int lanes_per_dim = Threads >> 7;  // 128->1, 256->2, 512->4, 1024->8.
  constexpr int dims_per_warp = 32 / lanes_per_dim;
  int dim_group = lane / lanes_per_dim;
  int lane_in_dim = lane - dim_group * lanes_per_dim;
  int out_dim = warp_id * dims_per_warp + dim_group;
  bool dim_leader = lane_in_dim == 0;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  if (tile_len <= 0) {
    if (tid == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = -INFINITY;
      partial_l[state_idx] = 0.0f;
    }
    return;
  }

  __shared__ float scores[SplitM];

  const half* __restrict__ query_head = query + q_head * kHeadDim;
  for (int m = warp_id; m < tile_len; m += num_warps) {
    const half* __restrict__ key_row =
        key_cache + (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum = qk_dot_half2_128(query_head, key_row, lane);
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_max = -INFINITY;
    for (int i = lane; i < tile_len; i += 32) {
      local_max = fmaxf(local_max, scores[i]);
    }
    float m = warp_max(local_max);
    m = __shfl_sync(0xffffffffu, m, 0);
    float local_l = 0.0f;
    for (int i = lane; i < tile_len; i += 32) {
      float p = __expf(scores[i] - m);
      scores[i] = p;
      local_l += p;
    }
    float l = warp_sum(local_l);
    if (lane == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = m;
      partial_l[state_idx] = l;
    }
  }
  __syncthreads();

  if (out_dim < kHeadDim) {
    float acc = 0.0f;
    for (int i = lane_in_dim; i < tile_len; i += lanes_per_dim) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + out_dim]);
    }
    int group_start = dim_group * lanes_per_dim;
    for (int offset = lanes_per_dim >> 1; offset > 0; offset >>= 1) {
      int src_lane = min(lane + offset, 31);
      float other = __shfl_sync(0xffffffffu, acc, src_lane);
      if (lane_in_dim < offset && lane + offset < group_start + lanes_per_dim) {
        acc += other;
      }
    }
    if (dim_leader) {
      int acc_base = (q_head * workspace_splits + split_idx) * kHeadDim;
      partial_acc[acc_base + out_dim] = acc;
    }
  }
}

template <int SplitM, int Threads>
__global__ void prefix_attention_split_stage1_h2dot_direct_2d_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int max_k,
    int workspace_splits,
    float scale) {
  int split_idx = blockIdx.x;
  int q_head = blockIdx.y;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;
  int kv_head = q_head >> 1;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  if (tile_len <= 0) {
    if (tid == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = -INFINITY;
      partial_l[state_idx] = 0.0f;
    }
    return;
  }

  __shared__ float scores[SplitM];

  const half* __restrict__ query_head = query + q_head * kHeadDim;
  for (int m = warp_id; m < tile_len; m += num_warps) {
    const half* __restrict__ key_row =
        key_cache + (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum = qk_dot_half2_128(query_head, key_row, lane);
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_max = -INFINITY;
    for (int i = lane; i < tile_len; i += 32) {
      local_max = fmaxf(local_max, scores[i]);
    }
    float m = warp_max(local_max);
    m = __shfl_sync(0xffffffffu, m, 0);
    float local_l = 0.0f;
    for (int i = lane; i < tile_len; i += 32) {
      float p = __expf(scores[i] - m);
      scores[i] = p;
      local_l += p;
    }
    float l = warp_sum(local_l);
    if (lane == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = m;
      partial_l[state_idx] = l;
    }
  }
  __syncthreads();

  for (int d = tid; d < kHeadDim; d += Threads) {
    float acc = 0.0f;
    for (int i = 0; i < tile_len; ++i) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + d]);
    }
    int acc_base = (q_head * workspace_splits + split_idx) * kHeadDim;
    partial_acc[acc_base + d] = acc;
  }
}

template <int SplitM, int Threads>
__global__ void prefix_attention_split_stage1_h2dot_direct_qcache_2d_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int max_k,
    int workspace_splits,
    float scale) {
  int split_idx = blockIdx.x;
  int q_head = blockIdx.y;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;
  int kv_head = q_head >> 1;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  if (tile_len <= 0) {
    if (tid == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = -INFINITY;
      partial_l[state_idx] = 0.0f;
    }
    return;
  }

  __shared__ float scores[SplitM];

  const half* __restrict__ query_head = query + q_head * kHeadDim;
  const half2* __restrict__ query2 = reinterpret_cast<const half2*>(query_head);
  float2 q01 = __half22float2(query2[lane]);
  float2 q23 = __half22float2(query2[lane + 32]);
  for (int m = warp_id; m < tile_len; m += num_warps) {
    const half* __restrict__ key_row =
        key_cache + (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum = qk_dot_half2_cached_query_128(q01, q23, key_row, lane);
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_max = -INFINITY;
    for (int i = lane; i < tile_len; i += 32) {
      local_max = fmaxf(local_max, scores[i]);
    }
    float m = warp_max(local_max);
    m = __shfl_sync(0xffffffffu, m, 0);
    float local_l = 0.0f;
    for (int i = lane; i < tile_len; i += 32) {
      float p = __expf(scores[i] - m);
      scores[i] = p;
      local_l += p;
    }
    float l = warp_sum(local_l);
    if (lane == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = m;
      partial_l[state_idx] = l;
    }
  }
  __syncthreads();

  for (int d = tid; d < kHeadDim; d += Threads) {
    float acc = 0.0f;
    for (int i = 0; i < tile_len; ++i) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + d]);
    }
    int acc_base = (q_head * workspace_splits + split_idx) * kHeadDim;
    partial_acc[acc_base + d] = acc;
  }
}

template <int SplitM, int Threads>
__global__ void prefix_attention_split_stage1_direct_2d_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int max_k,
    int workspace_splits,
    float scale) {
  int split_idx = blockIdx.x;
  int q_head = blockIdx.y;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;
  int kv_head = q_head >> 1;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  if (tile_len <= 0) {
    if (tid == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = -INFINITY;
      partial_l[state_idx] = 0.0f;
    }
    return;
  }

  __shared__ float scores[SplitM];

  int q_base = q_head * kHeadDim;
  float q0 = __half2float(query[q_base + lane]);
  float q1 = __half2float(query[q_base + lane + 32]);
  float q2 = __half2float(query[q_base + lane + 64]);
  float q3 = __half2float(query[q_base + lane + 96]);

  for (int m = warp_id; m < tile_len; m += num_warps) {
    int k_base = (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum =
        q0 * __half2float(key_cache[k_base + lane]) +
        q1 * __half2float(key_cache[k_base + lane + 32]) +
        q2 * __half2float(key_cache[k_base + lane + 64]) +
        q3 * __half2float(key_cache[k_base + lane + 96]);
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_max = -INFINITY;
    for (int i = lane; i < tile_len; i += 32) {
      local_max = fmaxf(local_max, scores[i]);
    }
    float m = warp_max(local_max);
    m = __shfl_sync(0xffffffffu, m, 0);
    float local_l = 0.0f;
    for (int i = lane; i < tile_len; i += 32) {
      float p = __expf(scores[i] - m);
      scores[i] = p;
      local_l += p;
    }
    float l = warp_sum(local_l);
    if (lane == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = m;
      partial_l[state_idx] = l;
    }
  }
  __syncthreads();

  for (int d = tid; d < kHeadDim; d += Threads) {
    float acc = 0.0f;
    for (int i = 0; i < tile_len; ++i) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + d]);
    }
    int acc_base = (q_head * workspace_splits + split_idx) * kHeadDim;
    partial_acc[acc_base + d] = acc;
  }
}

template <int SplitM, int Threads>
__global__ void prefix_attention_split_stage1_qpair_direct_2d_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int max_k,
    int workspace_splits,
    float scale) {
  int split_idx = blockIdx.x;
  int kv_head = blockIdx.y;
  int q_head0 = kv_head << 1;
  int q_head1 = q_head0 + 1;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  if (tile_len <= 0) {
    if (tid == 0) {
      int state_idx0 = q_head0 * workspace_splits + split_idx;
      int state_idx1 = q_head1 * workspace_splits + split_idx;
      partial_m[state_idx0] = -INFINITY;
      partial_l[state_idx0] = 0.0f;
      partial_m[state_idx1] = -INFINITY;
      partial_l[state_idx1] = 0.0f;
    }
    return;
  }

  __shared__ float scores0[SplitM];
  __shared__ float scores1[SplitM];

  const half* __restrict__ q0_head = query + q_head0 * kHeadDim;
  const half* __restrict__ q1_head = query + q_head1 * kHeadDim;
  for (int m = warp_id; m < tile_len; m += num_warps) {
    const half* __restrict__ key_row =
        key_cache + (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum0 = qk_dot_half2_128(q0_head, key_row, lane);
    float sum1 = qk_dot_half2_128(q1_head, key_row, lane);
    sum0 = warp_sum(sum0) * scale;
    sum1 = warp_sum(sum1) * scale;
    if (lane == 0) {
      scores0[m] = sum0;
      scores1[m] = sum1;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
    for (int i = lane; i < tile_len; i += 32) {
      local_max0 = fmaxf(local_max0, scores0[i]);
      local_max1 = fmaxf(local_max1, scores1[i]);
    }
    float m0 = warp_max(local_max0);
    float m1 = warp_max(local_max1);
    m0 = __shfl_sync(0xffffffffu, m0, 0);
    m1 = __shfl_sync(0xffffffffu, m1, 0);
    float local_l0 = 0.0f;
    float local_l1 = 0.0f;
    for (int i = lane; i < tile_len; i += 32) {
      float p0 = __expf(scores0[i] - m0);
      float p1 = __expf(scores1[i] - m1);
      scores0[i] = p0;
      scores1[i] = p1;
      local_l0 += p0;
      local_l1 += p1;
    }
    float l0 = warp_sum(local_l0);
    float l1 = warp_sum(local_l1);
    if (lane == 0) {
      int state_idx0 = q_head0 * workspace_splits + split_idx;
      int state_idx1 = q_head1 * workspace_splits + split_idx;
      partial_m[state_idx0] = m0;
      partial_l[state_idx0] = l0;
      partial_m[state_idx1] = m1;
      partial_l[state_idx1] = l1;
    }
  }
  __syncthreads();

  for (int d = tid; d < kHeadDim; d += Threads) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int i = 0; i < tile_len; ++i) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      float v = __half2float(value_cache[v_base + d]);
      acc0 += scores0[i] * v;
      acc1 += scores1[i] * v;
    }
    int acc_base0 = (q_head0 * workspace_splits + split_idx) * kHeadDim;
    int acc_base1 = (q_head1 * workspace_splits + split_idx) * kHeadDim;
    partial_acc[acc_base0 + d] = acc0;
    partial_acc[acc_base1 + d] = acc1;
  }
}

template <int SplitM, int Threads>
__global__ void prefix_attention_split_vgroup_atomic_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_acc,
    int* __restrict__ counters,
    half* __restrict__ out,
    int max_k,
    int workspace_splits,
    int active_splits,
    float scale) {
  int row = blockIdx.x;
  int q_head = row / active_splits;
  int split_idx = row - q_head * active_splits;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;
  int kv_head = q_head >> 1;

  constexpr int lanes_per_dim = Threads >> 7;  // 128->1, 256->2, 512->4, 1024->8.
  constexpr int dims_per_warp = 32 / lanes_per_dim;
  int dim_group = lane / lanes_per_dim;
  int lane_in_dim = lane - dim_group * lanes_per_dim;
  int out_dim = warp_id * dims_per_warp + dim_group;
  bool dim_leader = lane_in_dim == 0;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  int tile_start = split_idx * SplitM;
  int tile_len = valid_len - tile_start;
  tile_len = max(0, min(tile_len, SplitM));

  __shared__ float scores[SplitM];
  __shared__ int should_merge;

  int q_base = q_head * kHeadDim;
  float q0 = __half2float(query[q_base + lane]);
  float q1 = __half2float(query[q_base + lane + 32]);
  float q2 = __half2float(query[q_base + lane + 64]);
  float q3 = __half2float(query[q_base + lane + 96]);

  for (int m = warp_id; m < tile_len; m += num_warps) {
    int k_base = (kv_head * max_k + (tile_start + m)) * kHeadDim;
    float sum =
        q0 * __half2float(key_cache[k_base + lane]) +
        q1 * __half2float(key_cache[k_base + lane + 32]) +
        q2 * __half2float(key_cache[k_base + lane + 64]) +
        q3 * __half2float(key_cache[k_base + lane + 96]);
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    float local_max = -INFINITY;
    for (int i = lane; i < tile_len; i += 32) {
      local_max = fmaxf(local_max, scores[i]);
    }
    float m = warp_max(local_max);
    m = __shfl_sync(0xffffffffu, m, 0);
    float local_l = 0.0f;
    for (int i = lane; i < tile_len; i += 32) {
      float p = __expf(scores[i] - m);
      scores[i] = p;
      local_l += p;
    }
    float l = warp_sum(local_l);
    if (lane == 0) {
      int state_idx = q_head * workspace_splits + split_idx;
      partial_m[state_idx] = m;
      partial_l[state_idx] = l;
    }
  }
  __syncthreads();

  if (out_dim < kHeadDim) {
    float acc = 0.0f;
    for (int i = lane_in_dim; i < tile_len; i += lanes_per_dim) {
      int v_base = (kv_head * max_k + (tile_start + i)) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + out_dim]);
    }
    int group_start = dim_group * lanes_per_dim;
    for (int offset = lanes_per_dim >> 1; offset > 0; offset >>= 1) {
      int src_lane = min(lane + offset, 31);
      float other = __shfl_sync(0xffffffffu, acc, src_lane);
      if (lane_in_dim < offset && lane + offset < group_start + lanes_per_dim) {
        acc += other;
      }
    }
    if (dim_leader) {
      int acc_base = (q_head * workspace_splits + split_idx) * kHeadDim;
      partial_acc[acc_base + out_dim] = acc;
    }
  }
  __syncthreads();

  if (tid == 0) {
    __threadfence();
    int old = atomicAdd(counters + q_head, 1);
    should_merge = ((old + 1) % active_splits) == 0;
  }
  __syncthreads();

  if (should_merge) {
    int state_base = q_head * workspace_splits;
    int out_base = q_head * kHeadDim;
    for (int d = tid; d < kHeadDim; d += blockDim.x) {
      float m = -INFINITY;
      for (int s = 0; s < active_splits; ++s) {
        m = fmaxf(m, partial_m[state_base + s]);
      }
      float l = 0.0f;
      float acc = 0.0f;
      for (int s = 0; s < active_splits; ++s) {
        float weight = __expf(partial_m[state_base + s] - m);
        l += partial_l[state_base + s] * weight;
        int acc_idx = (state_base + s) * kHeadDim + d;
        acc += partial_acc[acc_idx] * weight;
      }
      out[out_base + d] = __float2half_rn(acc / l);
    }
  }
}

template <int BlockM>
__global__ void prefix_attention_tiled_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ out,
    int max_k,
    float scale) {
  int q_head = blockIdx.x;
  int kv_head = q_head >> 1;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  int num_warps = blockDim.x >> 5;

  __shared__ half q_s[kHeadDim];
  __shared__ float scores[BlockM];
  __shared__ float reduce_s[kMaxThreads];

  for (int d = tid; d < kHeadDim; d += blockDim.x) {
    q_s[d] = query[q_head * kHeadDim + d];
  }
  __syncthreads();

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  float running_m = -INFINITY;
  float running_l = 0.0f;
  float acc = 0.0f;
  int out_dim = tid;

  for (int tile_start = 0; tile_start < valid_len; tile_start += BlockM) {
    int tile_len = min(BlockM, valid_len - tile_start);

    for (int m = warp_id; m < tile_len; m += num_warps) {
      int k_base = (kv_head * max_k + tile_start + m) * kHeadDim;
      float sum = 0.0f;
#pragma unroll
      for (int d = lane; d < kHeadDim; d += 32) {
        sum += __half2float(q_s[d]) * __half2float(key_cache[k_base + d]);
      }
      sum = warp_sum(sum) * scale;
      if (lane == 0) {
        scores[m] = sum;
      }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int i = tid; i < tile_len; i += blockDim.x) {
      local_max = fmaxf(local_max, scores[i]);
    }
    reduce_s[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_s[tid] = fmaxf(reduce_s[tid], reduce_s[tid + stride]);
      }
      __syncthreads();
    }
    float block_m = reduce_s[0];
    float new_m = fmaxf(running_m, block_m);
    float alpha = __expf(running_m - new_m);

    float local_l = 0.0f;
    for (int i = tid; i < tile_len; i += blockDim.x) {
      float p = __expf(scores[i] - new_m);
      scores[i] = p;
      local_l += p;
    }
    reduce_s[tid] = local_l;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce_s[tid] += reduce_s[tid + stride];
      }
      __syncthreads();
    }
    float block_l = reduce_s[0];

    if (out_dim < kHeadDim) {
      float block_acc = 0.0f;
      for (int i = 0; i < tile_len; ++i) {
        int v_base = (kv_head * max_k + tile_start + i) * kHeadDim;
        block_acc += scores[i] * __half2float(value_cache[v_base + out_dim]);
      }
      acc = acc * alpha + block_acc;
    }
    running_l = running_l * alpha + block_l;
    running_m = new_m;
    __syncthreads();
  }

  if (out_dim < kHeadDim) {
    out[q_head * kHeadDim + out_dim] = __float2half_rn(acc / running_l);
  }
}

template <int BlockM>
__global__ void prefix_attention_tiled_warpred_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ out,
    int max_k,
    float scale) {
  int q_head = blockIdx.x;
  int kv_head = q_head >> 1;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  int num_warps = blockDim.x >> 5;

  __shared__ float scores[BlockM];
  __shared__ float shared_state[4];

  int q_base = q_head * kHeadDim;
  float q0 = __half2float(query[q_base + lane]);
  float q1 = __half2float(query[q_base + lane + 32]);
  float q2 = __half2float(query[q_base + lane + 64]);
  float q3 = __half2float(query[q_base + lane + 96]);

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  float running_m = -INFINITY;
  float running_l = 0.0f;
  float acc = 0.0f;
  int out_dim = tid;

  for (int tile_start = 0; tile_start < valid_len; tile_start += BlockM) {
    int tile_len = min(BlockM, valid_len - tile_start);

    for (int m = warp_id; m < tile_len; m += num_warps) {
      int k_base = (kv_head * max_k + tile_start + m) * kHeadDim;
      float sum =
          q0 * __half2float(key_cache[k_base + lane]) +
          q1 * __half2float(key_cache[k_base + lane + 32]) +
          q2 * __half2float(key_cache[k_base + lane + 64]) +
          q3 * __half2float(key_cache[k_base + lane + 96]);
      sum = warp_sum(sum) * scale;
      if (lane == 0) {
        scores[m] = sum;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      float local_max = -INFINITY;
      for (int i = lane; i < tile_len; i += 32) {
        local_max = fmaxf(local_max, scores[i]);
      }
      float block_m = warp_max(local_max);
      block_m = __shfl_sync(0xffffffffu, block_m, 0);
      float new_m = fmaxf(running_m, block_m);
      float local_l = 0.0f;
      for (int i = lane; i < tile_len; i += 32) {
        float p = __expf(scores[i] - new_m);
        scores[i] = p;
        local_l += p;
      }
      float block_l = warp_sum(local_l);
      if (lane == 0) {
        shared_state[0] = new_m;
        shared_state[1] = __expf(running_m - new_m);
        shared_state[2] = block_l;
      }
    }
    __syncthreads();

    float new_m = shared_state[0];
    float alpha = shared_state[1];
    float block_l = shared_state[2];

    if (out_dim < kHeadDim) {
      float block_acc = 0.0f;
      for (int i = 0; i < tile_len; ++i) {
        int v_base = (kv_head * max_k + tile_start + i) * kHeadDim;
        block_acc += scores[i] * __half2float(value_cache[v_base + out_dim]);
      }
      acc = acc * alpha + block_acc;
    }
    running_l = running_l * alpha + block_l;
    running_m = new_m;
    __syncthreads();
  }

  if (out_dim < kHeadDim) {
    out[q_head * kHeadDim + out_dim] = __float2half_rn(acc / running_l);
  }
}

template <int BlockM, int Threads>
__global__ void prefix_attention_tiled_vgroup_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ out,
    int max_k,
    float scale) {
  int q_head = blockIdx.x;
  int kv_head = q_head >> 1;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  constexpr int num_warps = Threads >> 5;

  __shared__ float scores[BlockM];
  __shared__ float shared_state[3];

  int q_base = q_head * kHeadDim;
  float q0 = __half2float(query[q_base + lane]);
  float q1 = __half2float(query[q_base + lane + 32]);
  float q2 = __half2float(query[q_base + lane + 64]);
  float q3 = __half2float(query[q_base + lane + 96]);

  constexpr int lanes_per_dim = Threads >> 7;  // 128->1, 256->2, 512->4, 1024->8.
  constexpr int dims_per_warp = 32 / lanes_per_dim;
  int dim_group = lane / lanes_per_dim;
  int lane_in_dim = lane - dim_group * lanes_per_dim;
  int out_dim = warp_id * dims_per_warp + dim_group;
  bool dim_leader = lane_in_dim == 0;

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);
  float running_m = -INFINITY;
  float running_l = 0.0f;
  float acc = 0.0f;

  for (int tile_start = 0; tile_start < valid_len; tile_start += BlockM) {
    int tile_len = min(BlockM, valid_len - tile_start);

    for (int m = warp_id; m < tile_len; m += num_warps) {
      int k_base = (kv_head * max_k + tile_start + m) * kHeadDim;
      float sum =
          q0 * __half2float(key_cache[k_base + lane]) +
          q1 * __half2float(key_cache[k_base + lane + 32]) +
          q2 * __half2float(key_cache[k_base + lane + 64]) +
          q3 * __half2float(key_cache[k_base + lane + 96]);
      sum = warp_sum(sum) * scale;
      if (lane == 0) {
        scores[m] = sum;
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      float local_max = -INFINITY;
      for (int i = lane; i < tile_len; i += 32) {
        local_max = fmaxf(local_max, scores[i]);
      }
      float block_m = warp_max(local_max);
      block_m = __shfl_sync(0xffffffffu, block_m, 0);
      float new_m = fmaxf(running_m, block_m);
      float local_l = 0.0f;
      for (int i = lane; i < tile_len; i += 32) {
        float p = __expf(scores[i] - new_m);
        scores[i] = p;
        local_l += p;
      }
      float block_l = warp_sum(local_l);
      if (lane == 0) {
        shared_state[0] = new_m;
        shared_state[1] = __expf(running_m - new_m);
        shared_state[2] = block_l;
      }
    }
    __syncthreads();

    float new_m = shared_state[0];
    float alpha = shared_state[1];
    float block_l = shared_state[2];

    if (out_dim < kHeadDim) {
      float block_acc = 0.0f;
      for (int i = lane_in_dim; i < tile_len; i += lanes_per_dim) {
        int v_base = (kv_head * max_k + tile_start + i) * kHeadDim;
        block_acc += scores[i] * __half2float(value_cache[v_base + out_dim]);
      }
      int group_start = dim_group * lanes_per_dim;
      for (int offset = lanes_per_dim >> 1; offset > 0; offset >>= 1) {
        int src_lane = min(lane + offset, 31);
        float other = __shfl_sync(0xffffffffu, block_acc, src_lane);
        if (lane_in_dim < offset && lane + offset < group_start + lanes_per_dim) {
          block_acc += other;
        }
      }
      if (dim_leader) {
        acc = acc * alpha + block_acc;
      }
    }
    running_l = running_l * alpha + block_l;
    running_m = new_m;
    __syncthreads();
  }

  if (out_dim < kHeadDim && dim_leader) {
    out[q_head * kHeadDim + out_dim] = __float2half_rn(acc / running_l);
  }
}

__global__ void prefix_attention_single_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int64_t* __restrict__ cache_position,
    half* __restrict__ out,
    int max_k,
    float scale) {
  int q_head = blockIdx.x;
  int kv_head = q_head >> 1;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;
  int num_warps = blockDim.x >> 5;

  extern __shared__ float scores[];
  __shared__ half q_s[kHeadDim];
  __shared__ float reduce_s[kMaxThreads];

  for (int d = tid; d < kHeadDim; d += blockDim.x) {
    q_s[d] = query[q_head * kHeadDim + d];
  }
  __syncthreads();

  int valid_len = static_cast<int>(cache_position[0]) + 1;
  valid_len = min(valid_len, max_k);

  for (int m = warp_id; m < valid_len; m += num_warps) {
    int k_base = (kv_head * max_k + m) * kHeadDim;
    float sum = 0.0f;
#pragma unroll
    for (int d = lane; d < kHeadDim; d += 32) {
      sum += __half2float(q_s[d]) * __half2float(key_cache[k_base + d]);
    }
    sum = warp_sum(sum) * scale;
    if (lane == 0) {
      scores[m] = sum;
    }
  }
  __syncthreads();

  float local_max = -INFINITY;
  for (int i = tid; i < valid_len; i += blockDim.x) {
    local_max = fmaxf(local_max, scores[i]);
  }
  reduce_s[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_s[tid] = fmaxf(reduce_s[tid], reduce_s[tid + stride]);
    }
    __syncthreads();
  }
  float max_score = reduce_s[0];

  float local_l = 0.0f;
  for (int i = tid; i < valid_len; i += blockDim.x) {
    float p = __expf(scores[i] - max_score);
    scores[i] = p;
    local_l += p;
  }
  reduce_s[tid] = local_l;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce_s[tid] += reduce_s[tid + stride];
    }
    __syncthreads();
  }
  float denom = reduce_s[0];

  int out_base = q_head * kHeadDim;
  for (int d = tid; d < kHeadDim; d += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < valid_len; ++i) {
      int v_base = (kv_head * max_k + i) * kHeadDim;
      acc += scores[i] * __half2float(value_cache[v_base + d]);
    }
    out[out_base + d] = __float2half_rn(acc / denom);
  }
}

__global__ void prefix_attention_split_stage2_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ out,
    int workspace_splits,
    int active_splits) {
  int q_head = blockIdx.x;

  int state_base = q_head * workspace_splits;
  int out_base = q_head * kHeadDim;
  for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
    float m = -INFINITY;
    for (int s = 0; s < active_splits; ++s) {
      m = fmaxf(m, partial_m[state_base + s]);
    }
    float l = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < active_splits; ++s) {
      float weight = __expf(partial_m[state_base + s] - m);
      l += partial_l[state_base + s] * weight;
      int acc_idx = (state_base + s) * kHeadDim + d;
      acc += partial_acc[acc_idx] * weight;
    }
    out[out_base + d] = __float2half_rn(acc / l);
  }
}

template <int ActiveSplits>
__global__ void prefix_attention_split_stage2_static_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ out,
    int workspace_splits) {
  int q_head = blockIdx.x;

  int state_base = q_head * workspace_splits;
  int out_base = q_head * kHeadDim;
  for (int d = threadIdx.x; d < kHeadDim; d += blockDim.x) {
    float m = -INFINITY;
#pragma unroll
    for (int s = 0; s < ActiveSplits; ++s) {
      m = fmaxf(m, partial_m[state_base + s]);
    }
    float l = 0.0f;
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < ActiveSplits; ++s) {
      float weight = __expf(partial_m[state_base + s] - m);
      l += partial_l[state_base + s] * weight;
      int acc_idx = (state_base + s) * kHeadDim + d;
      acc += partial_acc[acc_idx] * weight;
    }
    out[out_base + d] = __float2half_rn(acc / l);
  }
}

template <int ActiveSplits, int DimTile>
__global__ void prefix_attention_split_stage2_static_dimtile_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ out,
    int workspace_splits) {
  constexpr int kDimTiles = (kHeadDim + DimTile - 1) / DimTile;
  int q_head = blockIdx.x / kDimTiles;
  int tile_idx = blockIdx.x - q_head * kDimTiles;
  int local_d = threadIdx.x;
  if (local_d >= DimTile) {
    return;
  }
  int d = tile_idx * DimTile + local_d;
  if (d >= kHeadDim) {
    return;
  }

  int state_base = q_head * workspace_splits;
  float m = -INFINITY;
#pragma unroll
  for (int s = 0; s < ActiveSplits; ++s) {
    m = fmaxf(m, partial_m[state_base + s]);
  }
  float l = 0.0f;
  float acc = 0.0f;
#pragma unroll
  for (int s = 0; s < ActiveSplits; ++s) {
    float weight = __expf(partial_m[state_base + s] - m);
    l += partial_l[state_base + s] * weight;
    int acc_idx = (state_base + s) * kHeadDim + d;
    acc += partial_acc[acc_idx] * weight;
  }
  out[q_head * kHeadDim + d] = __float2half_rn(acc / l);
}

template <int ActiveSplits>
__global__ void prefix_attention_split_stage2_coeff_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const float* __restrict__ partial_acc,
    half* __restrict__ out,
    int workspace_splits) {
  __shared__ float coeff[16];
  const int q_head = blockIdx.x;
  const int tid = threadIdx.x;
  const int state_base = q_head * workspace_splits;

  if (tid < 16) {
    const float m_part = tid < ActiveSplits ? partial_m[state_base + tid] : -INFINITY;
    float m = m_part;
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, mask, 32));
    }
    const float weight = tid < ActiveSplits ? __expf(m_part - m) : 0.0f;
    float denom_part = tid < ActiveSplits ? partial_l[state_base + tid] * weight : 0.0f;
#pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
      denom_part += __shfl_xor_sync(0xffffffffu, denom_part, mask, 32);
    }
    if (tid < ActiveSplits) {
      coeff[tid] = weight / denom_part;
    }
  }
  __syncthreads();

  for (int d = tid; d < kHeadDim; d += blockDim.x) {
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < ActiveSplits; ++s) {
      acc += partial_acc[(state_base + s) * kHeadDim + d] * coeff[s];
    }
    out[q_head * kHeadDim + d] = __float2half_rn(acc);
  }
}

void check_prefix_attention_tiled_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    int64_t block_m,
    int64_t num_threads) {
  CHECK_CUDA(query);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_CUDA(cache_position);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key_cache);
  CHECK_CONTIGUOUS(value_cache);
  CHECK_CONTIGUOUS(out);
  CHECK_HALF(query);
  CHECK_HALF(key_cache);
  CHECK_HALF(value_cache);
  CHECK_HALF(out);
  TORCH_CHECK(query.dim() == 4, "query must be [1, 16, 1, 128]");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [1, 8, K, 128]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache shape must match key_cache");
  TORCH_CHECK(query.size(0) == 1, "CUDA prefix attention experiment is specialized for batch=1");
  TORCH_CHECK(key_cache.size(0) == 1, "CUDA prefix attention experiment is specialized for batch=1");
  TORCH_CHECK(query.size(1) == kQueryHeads, "query heads must be 16");
  TORCH_CHECK(key_cache.size(1) == kKvHeads, "KV heads must be 8");
  TORCH_CHECK(query.size(2) == 1, "q_len must be 1");
  TORCH_CHECK(query.size(3) == kHeadDim, "head_dim must be 128");
  TORCH_CHECK(key_cache.size(3) == kHeadDim, "key/value head_dim must be 128");
  TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");
  TORCH_CHECK(out.numel() == kQueryHeads * kHeadDim, "out numel mismatch");
  TORCH_CHECK(block_m == 64 || block_m == 128 || block_m == 256, "block_m must be 64, 128, or 256");
  TORCH_CHECK(
      num_threads == 128 || num_threads == 256 || num_threads == 512 || num_threads == 1024,
      "num_threads must be 128, 256, 512, or 1024");
}

void check_prefix_attention_single_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    int64_t num_threads) {
  CHECK_CUDA(query);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_CUDA(cache_position);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key_cache);
  CHECK_CONTIGUOUS(value_cache);
  CHECK_CONTIGUOUS(out);
  CHECK_HALF(query);
  CHECK_HALF(key_cache);
  CHECK_HALF(value_cache);
  CHECK_HALF(out);
  TORCH_CHECK(query.dim() == 4, "query must be [1, 16, 1, 128]");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [1, 8, K, 128]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache shape must match key_cache");
  TORCH_CHECK(query.size(0) == 1, "CUDA prefix attention experiment is specialized for batch=1");
  TORCH_CHECK(key_cache.size(0) == 1, "CUDA prefix attention experiment is specialized for batch=1");
  TORCH_CHECK(query.size(1) == kQueryHeads, "query heads must be 16");
  TORCH_CHECK(key_cache.size(1) == kKvHeads, "KV heads must be 8");
  TORCH_CHECK(query.size(2) == 1, "q_len must be 1");
  TORCH_CHECK(query.size(3) == kHeadDim, "head_dim must be 128");
  TORCH_CHECK(key_cache.size(3) == kHeadDim, "key/value head_dim must be 128");
  TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");
  TORCH_CHECK(out.numel() == kQueryHeads * kHeadDim, "out numel mismatch");
  TORCH_CHECK(num_threads == 128 || num_threads == 256, "num_threads must be 128 or 256");
}

void check_prefix_attention_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    int64_t split_m) {
  CHECK_CUDA(query);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_CUDA(cache_position);
  CHECK_CUDA(out);
  CHECK_CUDA(partial_m);
  CHECK_CUDA(partial_l);
  CHECK_CUDA(partial_acc);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key_cache);
  CHECK_CONTIGUOUS(value_cache);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(partial_m);
  CHECK_CONTIGUOUS(partial_l);
  CHECK_CONTIGUOUS(partial_acc);
  CHECK_HALF(query);
  CHECK_HALF(key_cache);
  CHECK_HALF(value_cache);
  CHECK_HALF(out);
  CHECK_FLOAT(partial_m);
  CHECK_FLOAT(partial_l);
  CHECK_FLOAT(partial_acc);
  TORCH_CHECK(query.dim() == 4, "query must be [B, H, 1, D]");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [B, KVH, K, D]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache shape must match key_cache");
  TORCH_CHECK(query.size(0) == 1, "CUDA prefix attention experiment is specialized for batch=1");
  TORCH_CHECK(key_cache.size(0) == 1, "CUDA prefix attention experiment is specialized for batch=1");
  TORCH_CHECK(query.size(1) == kQueryHeads, "query heads must be 16");
  TORCH_CHECK(key_cache.size(1) == kKvHeads, "KV heads must be 8");
  TORCH_CHECK(query.size(2) == 1, "q_len must be 1");
  TORCH_CHECK(query.size(3) == kHeadDim, "head_dim must be 128");
  TORCH_CHECK(key_cache.size(3) == kHeadDim, "key/value head_dim must be 128");
  TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");
  TORCH_CHECK(
      split_m == 32 || split_m == 48 || split_m == 64 || split_m == 96 || split_m == 128 || split_m == 256,
      "split_m must be 32, 48, 64, 96, 128, or 256");
  int64_t max_k = key_cache.size(2);
  int64_t num_splits = (max_k + split_m - 1) / split_m;
  TORCH_CHECK(out.numel() == kQueryHeads * kHeadDim, "out numel mismatch");
  TORCH_CHECK(partial_m.sizes() == at::IntArrayRef({kQueryHeads, num_splits}), "partial_m shape mismatch");
  TORCH_CHECK(partial_l.sizes() == at::IntArrayRef({kQueryHeads, num_splits}), "partial_l shape mismatch");
  TORCH_CHECK(
      partial_acc.sizes() == at::IntArrayRef({kQueryHeads, num_splits, kHeadDim}),
      "partial_acc shape mismatch");
}

void check_prefix_attention_split_vgroup_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    int64_t split_m,
    int64_t num_threads) {
  check_prefix_attention_inputs(
      query, key_cache, value_cache, cache_position, out, partial_m, partial_l, partial_acc, split_m);
  TORCH_CHECK(
      num_threads == 128 || num_threads == 256 || num_threads == 512 || num_threads == 1024,
      "num_threads must be 128, 256, 512, or 1024");
}

void check_prefix_attention_stage1_vgroup_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    int64_t split_m,
    int64_t num_threads) {
  CHECK_CUDA(query);
  CHECK_CUDA(key_cache);
  CHECK_CUDA(value_cache);
  CHECK_CUDA(cache_position);
  CHECK_CUDA(partial_m);
  CHECK_CUDA(partial_l);
  CHECK_CUDA(partial_acc);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key_cache);
  CHECK_CONTIGUOUS(value_cache);
  CHECK_CONTIGUOUS(partial_m);
  CHECK_CONTIGUOUS(partial_l);
  CHECK_CONTIGUOUS(partial_acc);
  CHECK_HALF(query);
  CHECK_HALF(key_cache);
  CHECK_HALF(value_cache);
  CHECK_FLOAT(partial_m);
  CHECK_FLOAT(partial_l);
  CHECK_FLOAT(partial_acc);
  TORCH_CHECK(query.dim() == 4, "query must be [B, H, 1, D]");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [B, KVH, K, D]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache shape must match key_cache");
  TORCH_CHECK(query.size(0) == 1, "CUDA prefix attention experiment is specialized for batch=1");
  TORCH_CHECK(key_cache.size(0) == 1, "CUDA prefix attention experiment is specialized for batch=1");
  TORCH_CHECK(query.size(1) == kQueryHeads, "query heads must be 16");
  TORCH_CHECK(key_cache.size(1) == kKvHeads, "KV heads must be 8");
  TORCH_CHECK(query.size(2) == 1, "q_len must be 1");
  TORCH_CHECK(query.size(3) == kHeadDim, "head_dim must be 128");
  TORCH_CHECK(key_cache.size(3) == kHeadDim, "key/value head_dim must be 128");
  TORCH_CHECK(cache_position.numel() == 1, "cache_position must contain one element");
  TORCH_CHECK(
      split_m == 32 || split_m == 48 || split_m == 64 || split_m == 96 || split_m == 128 || split_m == 256,
      "split_m must be 32, 48, 64, 96, 128, or 256");
  TORCH_CHECK(
      num_threads == 128 || num_threads == 256 || num_threads == 512 || num_threads == 1024,
      "num_threads must be 128, 256, 512, or 1024");
  int64_t max_k = key_cache.size(2);
  int64_t num_splits = (max_k + split_m - 1) / split_m;
  TORCH_CHECK(partial_m.sizes() == at::IntArrayRef({kQueryHeads, num_splits}), "partial_m shape mismatch");
  TORCH_CHECK(partial_l.sizes() == at::IntArrayRef({kQueryHeads, num_splits}), "partial_l shape mismatch");
  TORCH_CHECK(
      partial_acc.sizes() == at::IntArrayRef({kQueryHeads, num_splits, kHeadDim}),
      "partial_acc shape mismatch");
}

template <int SplitM>
void launch_prefix_attention_split(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale) {
  int max_k = static_cast<int>(key_cache.size(2));
  int num_splits = static_cast<int>((max_k + SplitM - 1) / SplitM);
  auto stream = at::cuda::getCurrentCUDAStream();
  int stage1_blocks = kQueryHeads * num_splits;
  prefix_attention_split_stage1_kernel<SplitM><<<stage1_blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      cache_position.data_ptr<int64_t>(),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_acc.data_ptr<float>(),
      max_k,
      num_splits,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  prefix_attention_split_stage2_kernel<<<kQueryHeads, kThreads, 0, stream>>>(
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_acc.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      num_splits,
      num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void prefix_attention_tiled_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor out,
    double scale,
    int64_t block_m,
    int64_t num_threads) {
  check_prefix_attention_tiled_inputs(query, key_cache, value_cache, cache_position, out, block_m, num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  int max_k = static_cast<int>(key_cache.size(2));
  int threads = static_cast<int>(num_threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (block_m == 64) {
    prefix_attention_tiled_kernel<64><<<kQueryHeads, threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        max_k,
        static_cast<float>(scale));
  } else if (block_m == 128) {
    prefix_attention_tiled_kernel<128><<<kQueryHeads, threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        max_k,
        static_cast<float>(scale));
  } else {
    prefix_attention_tiled_kernel<256><<<kQueryHeads, threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        max_k,
        static_cast<float>(scale));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_tiled_warpred_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor out,
    double scale,
    int64_t block_m,
    int64_t num_threads) {
  check_prefix_attention_tiled_inputs(query, key_cache, value_cache, cache_position, out, block_m, num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  int max_k = static_cast<int>(key_cache.size(2));
  int threads = static_cast<int>(num_threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (block_m == 64) {
    prefix_attention_tiled_warpred_kernel<64><<<kQueryHeads, threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        max_k,
        static_cast<float>(scale));
  } else if (block_m == 128) {
    prefix_attention_tiled_warpred_kernel<128><<<kQueryHeads, threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        max_k,
        static_cast<float>(scale));
  } else {
    prefix_attention_tiled_warpred_kernel<256><<<kQueryHeads, threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        max_k,
        static_cast<float>(scale));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BlockM, int Threads>
void launch_prefix_attention_tiled_vgroup(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    float scale) {
  int max_k = static_cast<int>(key_cache.size(2));
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_attention_tiled_vgroup_kernel<BlockM, Threads><<<kQueryHeads, Threads, 0, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      cache_position.data_ptr<int64_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      max_k,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Threads>
void launch_prefix_attention_tiled_vgroup_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    float scale,
    int64_t block_m) {
  if (block_m == 64) {
    launch_prefix_attention_tiled_vgroup<64, Threads>(query, key_cache, value_cache, cache_position, out, scale);
  } else if (block_m == 128) {
    launch_prefix_attention_tiled_vgroup<128, Threads>(query, key_cache, value_cache, cache_position, out, scale);
  } else {
    launch_prefix_attention_tiled_vgroup<256, Threads>(query, key_cache, value_cache, cache_position, out, scale);
  }
}

void prefix_attention_tiled_vgroup_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor out,
    double scale,
    int64_t block_m,
    int64_t num_threads) {
  check_prefix_attention_tiled_inputs(query, key_cache, value_cache, cache_position, out, block_m, num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_tiled_vgroup_block<128>(
        query, key_cache, value_cache, cache_position, out, static_cast<float>(scale), block_m);
  } else if (num_threads == 256) {
    launch_prefix_attention_tiled_vgroup_block<256>(
        query, key_cache, value_cache, cache_position, out, static_cast<float>(scale), block_m);
  } else if (num_threads == 512) {
    launch_prefix_attention_tiled_vgroup_block<512>(
        query, key_cache, value_cache, cache_position, out, static_cast<float>(scale), block_m);
  } else {
    launch_prefix_attention_tiled_vgroup_block<1024>(
        query, key_cache, value_cache, cache_position, out, static_cast<float>(scale), block_m);
  }
}

template <int ActiveSplits>
void launch_prefix_attention_split_stage2_static(
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& out,
    int workspace_splits,
    cudaStream_t stream) {
  const char* dimtile_env = std::getenv("AICAS_CUDA_DECODE_PREFIX_ATTENTION_STAGE2_DIM_TILE");
  int dim_tile = dimtile_env == nullptr ? 128 : std::atoi(dimtile_env);
  if (dim_tile == 0) {
    prefix_attention_split_stage2_coeff_kernel<ActiveSplits><<<kQueryHeads, 128, 0, stream>>>(
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        workspace_splits);
  } else if (dim_tile == 16) {
    constexpr int DimTile = 16;
    constexpr int blocks = kQueryHeads * ((kHeadDim + DimTile - 1) / DimTile);
    prefix_attention_split_stage2_static_dimtile_kernel<ActiveSplits, DimTile><<<blocks, DimTile, 0, stream>>>(
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        workspace_splits);
  } else if (dim_tile == 32) {
    constexpr int DimTile = 32;
    constexpr int blocks = kQueryHeads * ((kHeadDim + DimTile - 1) / DimTile);
    prefix_attention_split_stage2_static_dimtile_kernel<ActiveSplits, DimTile><<<blocks, DimTile, 0, stream>>>(
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        workspace_splits);
  } else if (dim_tile == 64) {
    constexpr int DimTile = 64;
    constexpr int blocks = kQueryHeads * ((kHeadDim + DimTile - 1) / DimTile);
    prefix_attention_split_stage2_static_dimtile_kernel<ActiveSplits, DimTile><<<blocks, DimTile, 0, stream>>>(
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        workspace_splits);
  } else {
    prefix_attention_split_stage2_static_kernel<ActiveSplits><<<kQueryHeads, kThreads, 0, stream>>>(
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        workspace_splits);
  }
}

void prefix_attention_split_stage2_gqa2(
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor out,
    int64_t active_splits) {
  CHECK_CUDA(partial_m);
  CHECK_CUDA(partial_l);
  CHECK_CUDA(partial_acc);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(partial_m);
  CHECK_CONTIGUOUS(partial_l);
  CHECK_CONTIGUOUS(partial_acc);
  CHECK_CONTIGUOUS(out);
  CHECK_FLOAT(partial_m);
  CHECK_FLOAT(partial_l);
  CHECK_FLOAT(partial_acc);
  CHECK_HALF(out);
  TORCH_CHECK(partial_m.dim() == 2, "partial_m must be [16, workspace_splits]");
  TORCH_CHECK(partial_l.sizes() == partial_m.sizes(), "partial_l shape mismatch");
  TORCH_CHECK(partial_m.size(0) == kQueryHeads, "partial_m must have 16 query heads");
  int workspace_splits = static_cast<int>(partial_m.size(1));
  TORCH_CHECK(active_splits >= 1 && active_splits <= workspace_splits, "active_splits out of range");
  TORCH_CHECK(active_splits <= 16, "stage2-only wrapper supports active_splits <= 16");
  TORCH_CHECK(
      partial_acc.sizes() == at::IntArrayRef({kQueryHeads, workspace_splits, kHeadDim}),
      "partial_acc must be [16, workspace_splits, 128]");
  TORCH_CHECK(out.numel() == kQueryHeads * kHeadDim, "out must have 2048 fp16 elements");
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (active_splits) {
    case 1:
      launch_prefix_attention_split_stage2_static<1>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 2:
      launch_prefix_attention_split_stage2_static<2>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 3:
      launch_prefix_attention_split_stage2_static<3>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 4:
      launch_prefix_attention_split_stage2_static<4>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 5:
      launch_prefix_attention_split_stage2_static<5>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 6:
      launch_prefix_attention_split_stage2_static<6>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 7:
      launch_prefix_attention_split_stage2_static<7>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 8:
      launch_prefix_attention_split_stage2_static<8>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 9:
      launch_prefix_attention_split_stage2_static<9>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 10:
      launch_prefix_attention_split_stage2_static<10>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 11:
      launch_prefix_attention_split_stage2_static<11>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 12:
      launch_prefix_attention_split_stage2_static<12>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 13:
      launch_prefix_attention_split_stage2_static<13>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 14:
      launch_prefix_attention_split_stage2_static<14>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 15:
      launch_prefix_attention_split_stage2_static<15>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    case 16:
      launch_prefix_attention_split_stage2_static<16>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
      break;
    default:
      TORCH_CHECK(false, "active_splits must be in [1,16]");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int SplitM, int Threads>
void launch_prefix_attention_split_vgroup(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + SplitM - 1) / SplitM);
  int launch_splits = workspace_splits;
  int merge_splits = workspace_splits;
  if (active_splits_hint > 0) {
    launch_splits = static_cast<int>(active_splits_hint);
    launch_splits = std::max(1, std::min(workspace_splits, launch_splits));
    merge_splits = launch_splits;
  }
  int stage1_blocks = kQueryHeads * launch_splits;
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_attention_split_stage1_vgroup_kernel<SplitM, Threads><<<stage1_blocks, Threads, 0, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      cache_position.data_ptr<int64_t>(),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_acc.data_ptr<float>(),
      max_k,
      workspace_splits,
      launch_splits,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (active_splits_hint <= 0) {
    merge_splits = workspace_splits;
  }

  if (merge_splits == 1) {
    launch_prefix_attention_split_stage2_static<1>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 2) {
    launch_prefix_attention_split_stage2_static<2>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 3) {
    launch_prefix_attention_split_stage2_static<3>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 4) {
    launch_prefix_attention_split_stage2_static<4>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 5) {
    launch_prefix_attention_split_stage2_static<5>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 6) {
    launch_prefix_attention_split_stage2_static<6>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 7) {
    launch_prefix_attention_split_stage2_static<7>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 8) {
    launch_prefix_attention_split_stage2_static<8>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 9) {
    launch_prefix_attention_split_stage2_static<9>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 10) {
    launch_prefix_attention_split_stage2_static<10>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 11) {
    launch_prefix_attention_split_stage2_static<11>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 12) {
    launch_prefix_attention_split_stage2_static<12>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 13) {
    launch_prefix_attention_split_stage2_static<13>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 14) {
    launch_prefix_attention_split_stage2_static<14>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 15) {
    launch_prefix_attention_split_stage2_static<15>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else if (merge_splits == 16) {
    launch_prefix_attention_split_stage2_static<16>(partial_m, partial_l, partial_acc, out, workspace_splits, stream);
  } else {
    prefix_attention_split_stage2_kernel<<<kQueryHeads, kThreads, 0, stream>>>(
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        workspace_splits,
        merge_splits);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Threads>
void launch_prefix_attention_split_stage1_vgroup_2d_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + split_m - 1) / split_m);
  int active_splits = workspace_splits;
  if (active_splits_hint > 0) {
    active_splits = static_cast<int>(active_splits_hint);
    active_splits = std::max(1, std::min(workspace_splits, active_splits));
  }
  dim3 grid(active_splits, kQueryHeads, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (split_m == 32) {
    prefix_attention_split_stage1_vgroup_2d_kernel<32, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 48) {
    prefix_attention_split_stage1_vgroup_2d_kernel<48, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 64) {
    prefix_attention_split_stage1_vgroup_2d_kernel<64, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 96) {
    prefix_attention_split_stage1_vgroup_2d_kernel<96, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 128) {
    prefix_attention_split_stage1_vgroup_2d_kernel<128, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else {
    prefix_attention_split_stage1_vgroup_2d_kernel<256, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_split_stage1_vgroup_2d_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_stage1_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_stage1_vgroup_2d_block<128>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_stage1_vgroup_2d_block<256>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_stage1_vgroup_2d_block<512>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else {
    launch_prefix_attention_split_stage1_vgroup_2d_block<1024>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  }
}

template <int Threads>
void launch_prefix_attention_split_stage1_vgroup_h2dot_2d_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + split_m - 1) / split_m);
  int active_splits = workspace_splits;
  if (active_splits_hint > 0) {
    active_splits = static_cast<int>(active_splits_hint);
    active_splits = std::max(1, std::min(workspace_splits, active_splits));
  }
  dim3 grid(active_splits, kQueryHeads, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (split_m == 32) {
    prefix_attention_split_stage1_vgroup_h2dot_2d_kernel<32, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 48) {
    prefix_attention_split_stage1_vgroup_h2dot_2d_kernel<48, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 64) {
    prefix_attention_split_stage1_vgroup_h2dot_2d_kernel<64, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 96) {
    prefix_attention_split_stage1_vgroup_h2dot_2d_kernel<96, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 128) {
    prefix_attention_split_stage1_vgroup_h2dot_2d_kernel<128, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else {
    prefix_attention_split_stage1_vgroup_h2dot_2d_kernel<256, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_split_stage1_vgroup_h2dot_2d_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_stage1_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_stage1_vgroup_h2dot_2d_block<128>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_stage1_vgroup_h2dot_2d_block<256>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_stage1_vgroup_h2dot_2d_block<512>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else {
    launch_prefix_attention_split_stage1_vgroup_h2dot_2d_block<1024>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  }
}

template <int Threads>
void launch_prefix_attention_split_stage1_h2dot_direct_2d_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + split_m - 1) / split_m);
  int active_splits = workspace_splits;
  if (active_splits_hint > 0) {
    active_splits = static_cast<int>(active_splits_hint);
    active_splits = std::max(1, std::min(workspace_splits, active_splits));
  }
  dim3 grid(active_splits, kQueryHeads, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (split_m == 32) {
    prefix_attention_split_stage1_h2dot_direct_2d_kernel<32, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 48) {
    prefix_attention_split_stage1_h2dot_direct_2d_kernel<48, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 64) {
    prefix_attention_split_stage1_h2dot_direct_2d_kernel<64, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 96) {
    prefix_attention_split_stage1_h2dot_direct_2d_kernel<96, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 128) {
    prefix_attention_split_stage1_h2dot_direct_2d_kernel<128, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else {
    prefix_attention_split_stage1_h2dot_direct_2d_kernel<256, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_split_stage1_h2dot_direct_2d_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_stage1_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_stage1_h2dot_direct_2d_block<128>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_stage1_h2dot_direct_2d_block<256>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_stage1_h2dot_direct_2d_block<512>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else {
    launch_prefix_attention_split_stage1_h2dot_direct_2d_block<1024>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  }
}

template <int Threads>
void launch_prefix_attention_split_stage1_h2dot_direct_qcache_2d_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + split_m - 1) / split_m);
  int active_splits = workspace_splits;
  if (active_splits_hint > 0) {
    active_splits = static_cast<int>(active_splits_hint);
    active_splits = std::max(1, std::min(workspace_splits, active_splits));
  }
  dim3 grid(active_splits, kQueryHeads, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (split_m == 32) {
    prefix_attention_split_stage1_h2dot_direct_qcache_2d_kernel<32, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 48) {
    prefix_attention_split_stage1_h2dot_direct_qcache_2d_kernel<48, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 64) {
    prefix_attention_split_stage1_h2dot_direct_qcache_2d_kernel<64, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 96) {
    prefix_attention_split_stage1_h2dot_direct_qcache_2d_kernel<96, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 128) {
    prefix_attention_split_stage1_h2dot_direct_qcache_2d_kernel<128, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else {
    prefix_attention_split_stage1_h2dot_direct_qcache_2d_kernel<256, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_split_stage1_h2dot_direct_qcache_2d_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_stage1_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_stage1_h2dot_direct_qcache_2d_block<128>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_stage1_h2dot_direct_qcache_2d_block<256>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_stage1_h2dot_direct_qcache_2d_block<512>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else {
    launch_prefix_attention_split_stage1_h2dot_direct_qcache_2d_block<1024>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  }
}

template <int Threads>
void launch_prefix_attention_split_stage1_direct_2d_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + split_m - 1) / split_m);
  int active_splits = workspace_splits;
  if (active_splits_hint > 0) {
    active_splits = static_cast<int>(active_splits_hint);
    active_splits = std::max(1, std::min(workspace_splits, active_splits));
  }
  dim3 grid(active_splits, kQueryHeads, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (split_m == 32) {
    prefix_attention_split_stage1_direct_2d_kernel<32, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 48) {
    prefix_attention_split_stage1_direct_2d_kernel<48, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 64) {
    prefix_attention_split_stage1_direct_2d_kernel<64, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 96) {
    prefix_attention_split_stage1_direct_2d_kernel<96, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 128) {
    prefix_attention_split_stage1_direct_2d_kernel<128, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else {
    prefix_attention_split_stage1_direct_2d_kernel<256, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_split_stage1_direct_2d_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_stage1_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_stage1_direct_2d_block<128>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_stage1_direct_2d_block<256>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_stage1_direct_2d_block<512>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else {
    launch_prefix_attention_split_stage1_direct_2d_block<1024>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  }
}

template <int Threads>
void launch_prefix_attention_split_stage1_qpair_direct_2d_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + split_m - 1) / split_m);
  int active_splits = workspace_splits;
  if (active_splits_hint > 0) {
    active_splits = static_cast<int>(active_splits_hint);
    active_splits = std::max(1, std::min(workspace_splits, active_splits));
  }
  dim3 grid(active_splits, kKvHeads, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (split_m == 32) {
    prefix_attention_split_stage1_qpair_direct_2d_kernel<32, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 48) {
    prefix_attention_split_stage1_qpair_direct_2d_kernel<48, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 64) {
    prefix_attention_split_stage1_qpair_direct_2d_kernel<64, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 96) {
    prefix_attention_split_stage1_qpair_direct_2d_kernel<96, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else if (split_m == 128) {
    prefix_attention_split_stage1_qpair_direct_2d_kernel<128, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  } else {
    prefix_attention_split_stage1_qpair_direct_2d_kernel<256, Threads><<<grid, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_split_stage1_qpair_direct_2d_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_stage1_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_stage1_qpair_direct_2d_block<128>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_stage1_qpair_direct_2d_block<256>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_stage1_qpair_direct_2d_block<512>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else {
    launch_prefix_attention_split_stage1_qpair_direct_2d_block<1024>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  }
}

template <int Threads>
void launch_prefix_attention_split_stage1_vgroup_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + split_m - 1) / split_m);
  int active_splits = workspace_splits;
  if (active_splits_hint > 0) {
    active_splits = static_cast<int>(active_splits_hint);
    active_splits = std::max(1, std::min(workspace_splits, active_splits));
  }
  int stage1_blocks = kQueryHeads * active_splits;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (split_m == 32) {
    prefix_attention_split_stage1_vgroup_kernel<32, Threads><<<stage1_blocks, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        active_splits,
        scale);
  } else if (split_m == 48) {
    prefix_attention_split_stage1_vgroup_kernel<48, Threads><<<stage1_blocks, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        active_splits,
        scale);
  } else if (split_m == 64) {
    prefix_attention_split_stage1_vgroup_kernel<64, Threads><<<stage1_blocks, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        active_splits,
        scale);
  } else if (split_m == 96) {
    prefix_attention_split_stage1_vgroup_kernel<96, Threads><<<stage1_blocks, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        active_splits,
        scale);
  } else if (split_m == 128) {
    prefix_attention_split_stage1_vgroup_kernel<128, Threads><<<stage1_blocks, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        active_splits,
        scale);
  } else {
    prefix_attention_split_stage1_vgroup_kernel<256, Threads><<<stage1_blocks, Threads, 0, stream>>>(
        reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
        cache_position.data_ptr<int64_t>(),
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        partial_acc.data_ptr<float>(),
        max_k,
        workspace_splits,
        active_splits,
        scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_split_stage1_vgroup_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_stage1_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_stage1_vgroup_block<128>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_stage1_vgroup_block<256>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_stage1_vgroup_block<512>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  } else {
    launch_prefix_attention_split_stage1_vgroup_block<1024>(
        query, key_cache, value_cache, cache_position, partial_m, partial_l, partial_acc,
        static_cast<float>(scale), split_m, active_splits_hint);
  }
}

template <int Threads>
void launch_prefix_attention_split_vgroup_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  if (split_m == 32) {
    launch_prefix_attention_split_vgroup<32, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        scale,
        active_splits_hint);
  } else if (split_m == 48) {
    launch_prefix_attention_split_vgroup<48, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        scale,
        active_splits_hint);
  } else if (split_m == 64) {
    launch_prefix_attention_split_vgroup<64, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        scale,
        active_splits_hint);
  } else if (split_m == 96) {
    launch_prefix_attention_split_vgroup<96, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        scale,
        active_splits_hint);
  } else if (split_m == 128) {
    launch_prefix_attention_split_vgroup<128, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        scale,
        active_splits_hint);
  } else {
    launch_prefix_attention_split_vgroup<256, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        scale,
        active_splits_hint);
  }
}

void prefix_attention_split_vgroup_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor out,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_split_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      out,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_vgroup_block<128>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        static_cast<float>(scale),
        split_m,
        active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_vgroup_block<256>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        static_cast<float>(scale),
        split_m,
        active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_vgroup_block<512>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        static_cast<float>(scale),
        split_m,
        active_splits_hint);
  } else {
    launch_prefix_attention_split_vgroup_block<1024>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        static_cast<float>(scale),
        split_m,
        active_splits_hint);
  }
}

template <int SplitM, int Threads>
void launch_prefix_attention_split_vgroup_atomic(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& counters,
    float scale,
    int64_t active_splits_hint) {
  int max_k = static_cast<int>(key_cache.size(2));
  int workspace_splits = static_cast<int>((max_k + SplitM - 1) / SplitM);
  int active_splits = workspace_splits;
  if (active_splits_hint > 0) {
    active_splits = static_cast<int>(active_splits_hint);
    active_splits = std::max(1, std::min(workspace_splits, active_splits));
  }
  int stage1_blocks = kQueryHeads * active_splits;
  auto stream = at::cuda::getCurrentCUDAStream();
  prefix_attention_split_vgroup_atomic_kernel<SplitM, Threads><<<stage1_blocks, Threads, 0, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      cache_position.data_ptr<int64_t>(),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_acc.data_ptr<float>(),
      counters.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      max_k,
      workspace_splits,
      active_splits,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Threads>
void launch_prefix_attention_split_vgroup_atomic_block(
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& cache_position,
    const torch::Tensor& out,
    const torch::Tensor& partial_m,
    const torch::Tensor& partial_l,
    const torch::Tensor& partial_acc,
    const torch::Tensor& counters,
    float scale,
    int64_t split_m,
    int64_t active_splits_hint) {
  if (split_m == 32) {
    launch_prefix_attention_split_vgroup_atomic<32, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        scale,
        active_splits_hint);
  } else if (split_m == 48) {
    launch_prefix_attention_split_vgroup_atomic<48, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        scale,
        active_splits_hint);
  } else if (split_m == 64) {
    launch_prefix_attention_split_vgroup_atomic<64, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        scale,
        active_splits_hint);
  } else if (split_m == 96) {
    launch_prefix_attention_split_vgroup_atomic<96, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        scale,
        active_splits_hint);
  } else if (split_m == 128) {
    launch_prefix_attention_split_vgroup_atomic<128, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        scale,
        active_splits_hint);
  } else {
    launch_prefix_attention_split_vgroup_atomic<256, Threads>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        scale,
        active_splits_hint);
  }
}

void prefix_attention_split_vgroup_atomic_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor out,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    torch::Tensor counters,
    double scale,
    int64_t split_m,
    int64_t num_threads,
    int64_t active_splits_hint) {
  check_prefix_attention_split_vgroup_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      out,
      partial_m,
      partial_l,
      partial_acc,
      split_m,
      num_threads);
  CHECK_CUDA(counters);
  CHECK_CONTIGUOUS(counters);
  TORCH_CHECK(counters.scalar_type() == at::kInt, "counters must be int32");
  TORCH_CHECK(counters.numel() == kQueryHeads, "counters numel mismatch");
  c10::cuda::CUDAGuard device_guard(query.device());
  if (num_threads == 128) {
    launch_prefix_attention_split_vgroup_atomic_block<128>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        static_cast<float>(scale),
        split_m,
        active_splits_hint);
  } else if (num_threads == 256) {
    launch_prefix_attention_split_vgroup_atomic_block<256>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        static_cast<float>(scale),
        split_m,
        active_splits_hint);
  } else if (num_threads == 512) {
    launch_prefix_attention_split_vgroup_atomic_block<512>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        static_cast<float>(scale),
        split_m,
        active_splits_hint);
  } else {
    launch_prefix_attention_split_vgroup_atomic_block<1024>(
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        partial_m,
        partial_l,
        partial_acc,
        counters,
        static_cast<float>(scale),
        split_m,
        active_splits_hint);
  }
}

void prefix_attention_single_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor out,
    double scale,
    int64_t num_threads) {
  check_prefix_attention_single_inputs(query, key_cache, value_cache, cache_position, out, num_threads);
  c10::cuda::CUDAGuard device_guard(query.device());
  int max_k = static_cast<int>(key_cache.size(2));
  auto stream = at::cuda::getCurrentCUDAStream();
  size_t shared_bytes = static_cast<size_t>(max_k) * sizeof(float);
  prefix_attention_single_kernel<<<kQueryHeads, static_cast<int>(num_threads), shared_bytes, stream>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      cache_position.data_ptr<int64_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      max_k,
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prefix_attention_split_gqa2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor cache_position,
    torch::Tensor out,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_acc,
    double scale,
    int64_t split_m) {
  check_prefix_attention_inputs(
      query,
      key_cache,
      value_cache,
      cache_position,
      out,
      partial_m,
      partial_l,
      partial_acc,
      split_m);
  c10::cuda::CUDAGuard device_guard(query.device());
  if (split_m == 32) {
    launch_prefix_attention_split<32>(
        query, key_cache, value_cache, cache_position, out, partial_m, partial_l, partial_acc, static_cast<float>(scale));
  } else if (split_m == 48) {
    launch_prefix_attention_split<48>(
        query, key_cache, value_cache, cache_position, out, partial_m, partial_l, partial_acc, static_cast<float>(scale));
  } else if (split_m == 64) {
    launch_prefix_attention_split<64>(
        query, key_cache, value_cache, cache_position, out, partial_m, partial_l, partial_acc, static_cast<float>(scale));
  } else if (split_m == 128) {
    launch_prefix_attention_split<128>(
        query, key_cache, value_cache, cache_position, out, partial_m, partial_l, partial_acc, static_cast<float>(scale));
  } else {
    launch_prefix_attention_split<256>(
        query, key_cache, value_cache, cache_position, out, partial_m, partial_l, partial_acc, static_cast<float>(scale));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "prefix_attention_tiled_gqa2",
      &prefix_attention_tiled_gqa2,
      "Tiled online-softmax CUDA exact decode attention for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_tiled_warpred_gqa2",
      &prefix_attention_tiled_warpred_gqa2,
      "Tiled CUDA exact decode attention with warp-only tile reductions for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_tiled_vgroup_gqa2",
      &prefix_attention_tiled_vgroup_gqa2,
      "Tiled CUDA exact decode attention with grouped lanes per output dim for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_vgroup_gqa2",
      &prefix_attention_split_vgroup_gqa2,
      "Split-prefix CUDA exact decode attention with grouped lanes per output dim for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_stage1_vgroup_gqa2",
      &prefix_attention_split_stage1_vgroup_gqa2,
      "Stage1-only split-prefix CUDA decode attention with grouped lanes per output dim for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_stage1_vgroup_2d_gqa2",
      &prefix_attention_split_stage1_vgroup_2d_gqa2,
      "Stage1-only split-prefix CUDA decode attention with 2D CTA mapping for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_stage1_vgroup_h2dot_2d_gqa2",
      &prefix_attention_split_stage1_vgroup_h2dot_2d_gqa2,
      "Stage1-only split-prefix CUDA decode attention with half2 QK dot and 2D CTA mapping for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_stage1_h2dot_direct_2d_gqa2",
      &prefix_attention_split_stage1_h2dot_direct_2d_gqa2,
      "Stage1-only split-prefix CUDA decode attention with half2 QK dot and direct V accumulation for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_stage1_h2dot_direct_qcache_2d_gqa2",
      &prefix_attention_split_stage1_h2dot_direct_qcache_2d_gqa2,
      "Stage1-only split-prefix CUDA decode attention with query cached QK dot for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_stage1_direct_2d_gqa2",
      &prefix_attention_split_stage1_direct_2d_gqa2,
      "Stage1-only split-prefix CUDA decode attention with direct V accumulation for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_stage1_qpair_direct_2d_gqa2",
      &prefix_attention_split_stage1_qpair_direct_2d_gqa2,
      "Stage1-only split-prefix CUDA decode attention with paired GQA heads and direct V accumulation for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_stage2_gqa2",
      &prefix_attention_split_stage2_gqa2,
      "Stage2-only split-prefix CUDA decode attention merge for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_vgroup_atomic_gqa2",
      &prefix_attention_split_vgroup_atomic_gqa2,
      "Split-prefix CUDA exact decode attention with in-kernel merge for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_single_gqa2",
      &prefix_attention_single_gqa2,
      "Single-kernel CUDA exact decode attention for Qwen3-VL GQA2");
  m.def(
      "prefix_attention_split_gqa2",
      &prefix_attention_split_gqa2,
      "Split-prefix CUDA exact decode attention for Qwen3-VL GQA2");
}
