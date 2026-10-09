#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")

__inline__ __device__ float warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__forceinline__ __device__ float dot_acc_half2(float acc, half2 x, half2 w) {
  float2 xf = __half22float2(x);
  float2 wf = __half22float2(w);
  acc = fmaf(xf.x, wf.x, acc);
  acc = fmaf(xf.y, wf.y, acc);
  return acc;
}

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ uint4 load_uint4_cs(const uint4* ptr) {
  return __ldcs(ptr);
}

__forceinline__ __device__ uint4 load_uint4_cg(const uint4* ptr) {
  return __ldcg(ptr);
}

template <int UNROLL>
__global__ void gate_up_swiglu_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int k_in,
    int rows_per_block) {
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const half* gate_w = weight + static_cast<int64_t>(row) * k_in;
  const half* up_w = weight + static_cast<int64_t>(row + intermediate_size) * k_in;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (int k_base = lane; k_base < k_in; k_base += 32 * UNROLL) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int k = k_base + u * 32;
      if (k < k_in) {
        float xv = __half2float(x[k]);
        gate_acc = fmaf(xv, __half2float(gate_w[k]), gate_acc);
        up_acc = fmaf(xv, __half2float(up_w[k]), up_acc);
      }
    }
  }

  gate_acc = warp_sum(gate_acc);
  up_acc = warp_sum(up_acc);

  if (lane == 0) {
    float swish = gate_acc / (1.0f + expf(-gate_acc));
    out[row] = __float2half_rn(swish * up_acc);
  }
}

__global__ void gate_up_swiglu_legacy_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int k_in,
    int rows_per_block) {
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const half* gate_w = weight + static_cast<int64_t>(row) * k_in;
  const half* up_w = weight + static_cast<int64_t>(row + intermediate_size) * k_in;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
#pragma unroll 4
  for (int k = lane; k < k_in; k += 32) {
    float xv = __half2float(x[k]);
    gate_acc = fmaf(xv, __half2float(gate_w[k]), gate_acc);
    up_acc = fmaf(xv, __half2float(up_w[k]), up_acc);
  }

  gate_acc = warp_sum(gate_acc);
  up_acc = warp_sum(up_acc);

  if (lane == 0) {
    float swish = gate_acc / (1.0f + __expf(-gate_acc));
    out[row] = __float2half_rn(swish * up_acc);
  }
}

template <int UNROLL>
__global__ void gate_up_swiglu_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  constexpr int K_IN = 2048;
  constexpr int STEP = 32 * UNROLL;
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const half* gate_w = weight + static_cast<int64_t>(row) * K_IN;
  const half* up_w = weight + static_cast<int64_t>(row + intermediate_size) * K_IN;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  int k = lane;
  for (; k + STEP - 32 < K_IN; k += STEP) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int ku = k + u * 32;
      float xv = __half2float(x[ku]);
      gate_acc = fmaf(xv, __half2float(gate_w[ku]), gate_acc);
      up_acc = fmaf(xv, __half2float(up_w[ku]), up_acc);
    }
  }
  for (; k < K_IN; k += 32) {
    float xv = __half2float(x[k]);
    gate_acc = fmaf(xv, __half2float(gate_w[k]), gate_acc);
    up_acc = fmaf(xv, __half2float(up_w[k]), up_acc);
  }

  gate_acc = warp_sum(gate_acc);
  up_acc = warp_sum(up_acc);

  if (lane == 0) {
    float swish = gate_acc / (1.0f + __expf(-gate_acc));
    out[row] = __float2half_rn(swish * up_acc);
  }
}

template <int UNROLL>
__global__ void gate_up_swiglu_half2_haccum_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  constexpr int K_IN = 2048;
  constexpr int K_PAIRS = K_IN / 2;
  constexpr int STEP = 32 * UNROLL;
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const half2* gate_w = reinterpret_cast<const half2*>(weight + static_cast<int64_t>(row) * K_IN);
  const half2* up_w = reinterpret_cast<const half2*>(
      weight + static_cast<int64_t>(row + intermediate_size) * K_IN);

  half2 gate_acc = __float2half2_rn(0.0f);
  half2 up_acc = __float2half2_rn(0.0f);
  int k = lane;
  for (; k + STEP - 32 < K_PAIRS; k += STEP) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int ku = k + u * 32;
      half2 xv = x2[ku];
      gate_acc = __hfma2(xv, gate_w[ku], gate_acc);
      up_acc = __hfma2(xv, up_w[ku], up_acc);
    }
  }
  for (; k < K_PAIRS; k += 32) {
    half2 xv = x2[k];
    gate_acc = __hfma2(xv, gate_w[k], gate_acc);
    up_acc = __hfma2(xv, up_w[k], up_acc);
  }

  float gate = __half2float(__low2half(gate_acc)) + __half2float(__high2half(gate_acc));
  float up = __half2float(__low2half(up_acc)) + __half2float(__high2half(up_acc));
  gate = warp_sum(gate);
  up = warp_sum(up);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int UNROLL>
__global__ void gate_up_swiglu_half2_haccum4_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  constexpr int K_IN = 2048;
  constexpr int K_PAIRS = K_IN / 2;
  constexpr int STEP = 32 * UNROLL;
  constexpr int ACCS = 4;
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const half2* gate_w = reinterpret_cast<const half2*>(weight + static_cast<int64_t>(row) * K_IN);
  const half2* up_w = reinterpret_cast<const half2*>(
      weight + static_cast<int64_t>(row + intermediate_size) * K_IN);

  half2 gate_acc[ACCS];
  half2 up_acc[ACCS];
#pragma unroll
  for (int i = 0; i < ACCS; ++i) {
    gate_acc[i] = __float2half2_rn(0.0f);
    up_acc[i] = __float2half2_rn(0.0f);
  }

  int k = lane;
  for (; k + STEP - 32 < K_PAIRS; k += STEP) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int ku = k + u * 32;
      int acc_idx = u & (ACCS - 1);
      half2 xv = x2[ku];
      gate_acc[acc_idx] = __hfma2(xv, gate_w[ku], gate_acc[acc_idx]);
      up_acc[acc_idx] = __hfma2(xv, up_w[ku], up_acc[acc_idx]);
    }
  }
  for (; k < K_PAIRS; k += 32) {
    half2 xv = x2[k];
    gate_acc[0] = __hfma2(xv, gate_w[k], gate_acc[0]);
    up_acc[0] = __hfma2(xv, up_w[k], up_acc[0]);
  }

  float gate = 0.0f;
  float up = 0.0f;
#pragma unroll
  for (int i = 0; i < ACCS; ++i) {
    gate += __half2float(__low2half(gate_acc[i])) + __half2float(__high2half(gate_acc[i]));
    up += __half2float(__low2half(up_acc[i])) + __half2float(__high2half(up_acc[i]));
  }
  gate = warp_sum(gate);
  up = warp_sum(up);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

__global__ void gate_up_swiglu_half2_vec4_haccum_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  constexpr int K_IN = 2048;
  constexpr int K_PAIRS = K_IN / 2;
  constexpr int VEC_PAIRS = 4;
  constexpr int VEC_COUNT = K_PAIRS / VEC_PAIRS;
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const uint4* x4 = reinterpret_cast<const uint4*>(x);
  const uint4* gate_w4 = reinterpret_cast<const uint4*>(weight + static_cast<int64_t>(row) * K_IN);
  const uint4* up_w4 = reinterpret_cast<const uint4*>(
      weight + static_cast<int64_t>(row + intermediate_size) * K_IN);

  half2 gate_acc0 = __float2half2_rn(0.0f);
  half2 gate_acc1 = __float2half2_rn(0.0f);
  half2 gate_acc2 = __float2half2_rn(0.0f);
  half2 gate_acc3 = __float2half2_rn(0.0f);
  half2 up_acc0 = __float2half2_rn(0.0f);
  half2 up_acc1 = __float2half2_rn(0.0f);
  half2 up_acc2 = __float2half2_rn(0.0f);
  half2 up_acc3 = __float2half2_rn(0.0f);

#pragma unroll
  for (int vec = lane; vec < VEC_COUNT; vec += 32) {
    uint4 xv = x4[vec];
    uint4 gv = gate_w4[vec];
    uint4 uv = up_w4[vec];

    half2 x0 = half2_from_u32(xv.x);
    half2 x1 = half2_from_u32(xv.y);
    half2 x2 = half2_from_u32(xv.z);
    half2 x3 = half2_from_u32(xv.w);

    gate_acc0 = __hfma2(x0, half2_from_u32(gv.x), gate_acc0);
    gate_acc1 = __hfma2(x1, half2_from_u32(gv.y), gate_acc1);
    gate_acc2 = __hfma2(x2, half2_from_u32(gv.z), gate_acc2);
    gate_acc3 = __hfma2(x3, half2_from_u32(gv.w), gate_acc3);
    up_acc0 = __hfma2(x0, half2_from_u32(uv.x), up_acc0);
    up_acc1 = __hfma2(x1, half2_from_u32(uv.y), up_acc1);
    up_acc2 = __hfma2(x2, half2_from_u32(uv.z), up_acc2);
    up_acc3 = __hfma2(x3, half2_from_u32(uv.w), up_acc3);
  }

  float gate =
      __half2float(__low2half(gate_acc0)) + __half2float(__high2half(gate_acc0)) +
      __half2float(__low2half(gate_acc1)) + __half2float(__high2half(gate_acc1)) +
      __half2float(__low2half(gate_acc2)) + __half2float(__high2half(gate_acc2)) +
      __half2float(__low2half(gate_acc3)) + __half2float(__high2half(gate_acc3));
  float up =
      __half2float(__low2half(up_acc0)) + __half2float(__high2half(up_acc0)) +
      __half2float(__low2half(up_acc1)) + __half2float(__high2half(up_acc1)) +
      __half2float(__low2half(up_acc2)) + __half2float(__high2half(up_acc2)) +
      __half2float(__low2half(up_acc3)) + __half2float(__high2half(up_acc3));
  gate = warp_sum(gate);
  up = warp_sum(up);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int LOAD_MODE>
__global__ void gate_up_swiglu_half2_vec4_loadmode_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  constexpr int K_IN = 2048;
  constexpr int K_PAIRS = K_IN / 2;
  constexpr int VEC_PAIRS = 4;
  constexpr int VEC_COUNT = K_PAIRS / VEC_PAIRS;
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const uint4* x4 = reinterpret_cast<const uint4*>(x);
  const uint4* gate_w4 = reinterpret_cast<const uint4*>(weight + static_cast<int64_t>(row) * K_IN);
  const uint4* up_w4 = reinterpret_cast<const uint4*>(
      weight + static_cast<int64_t>(row + intermediate_size) * K_IN);

  half2 gate_acc0 = __float2half2_rn(0.0f);
  half2 gate_acc1 = __float2half2_rn(0.0f);
  half2 gate_acc2 = __float2half2_rn(0.0f);
  half2 gate_acc3 = __float2half2_rn(0.0f);
  half2 up_acc0 = __float2half2_rn(0.0f);
  half2 up_acc1 = __float2half2_rn(0.0f);
  half2 up_acc2 = __float2half2_rn(0.0f);
  half2 up_acc3 = __float2half2_rn(0.0f);

#pragma unroll
  for (int vec = lane; vec < VEC_COUNT; vec += 32) {
    uint4 xv = x4[vec];
    uint4 gv;
    uint4 uv;
    if constexpr (LOAD_MODE == 1) {
      gv = load_uint4_cs(gate_w4 + vec);
      uv = load_uint4_cs(up_w4 + vec);
    } else {
      gv = load_uint4_cg(gate_w4 + vec);
      uv = load_uint4_cg(up_w4 + vec);
    }

    half2 x0 = half2_from_u32(xv.x);
    half2 x1 = half2_from_u32(xv.y);
    half2 x2 = half2_from_u32(xv.z);
    half2 x3 = half2_from_u32(xv.w);

    gate_acc0 = __hfma2(x0, half2_from_u32(gv.x), gate_acc0);
    gate_acc1 = __hfma2(x1, half2_from_u32(gv.y), gate_acc1);
    gate_acc2 = __hfma2(x2, half2_from_u32(gv.z), gate_acc2);
    gate_acc3 = __hfma2(x3, half2_from_u32(gv.w), gate_acc3);
    up_acc0 = __hfma2(x0, half2_from_u32(uv.x), up_acc0);
    up_acc1 = __hfma2(x1, half2_from_u32(uv.y), up_acc1);
    up_acc2 = __hfma2(x2, half2_from_u32(uv.z), up_acc2);
    up_acc3 = __hfma2(x3, half2_from_u32(uv.w), up_acc3);
  }

  float gate =
      __half2float(__low2half(gate_acc0)) + __half2float(__high2half(gate_acc0)) +
      __half2float(__low2half(gate_acc1)) + __half2float(__high2half(gate_acc1)) +
      __half2float(__low2half(gate_acc2)) + __half2float(__high2half(gate_acc2)) +
      __half2float(__low2half(gate_acc3)) + __half2float(__high2half(gate_acc3));
  float up =
      __half2float(__low2half(up_acc0)) + __half2float(__high2half(up_acc0)) +
      __half2float(__low2half(up_acc1)) + __half2float(__high2half(up_acc1)) +
      __half2float(__low2half(up_acc2)) + __half2float(__high2half(up_acc2)) +
      __half2float(__low2half(up_acc3)) + __half2float(__high2half(up_acc3));
  gate = warp_sum(gate);
  up = warp_sum(up);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

__global__ void gate_up_swiglu_half2_vec4_haccum_nox_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  constexpr int K_IN = 2048;
  constexpr int K_PAIRS = K_IN / 2;
  constexpr int VEC_PAIRS = 4;
  constexpr int VEC_COUNT = K_PAIRS / VEC_PAIRS;
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const uint4* gate_w4 = reinterpret_cast<const uint4*>(weight + static_cast<int64_t>(row) * K_IN);
  const uint4* up_w4 = reinterpret_cast<const uint4*>(
      weight + static_cast<int64_t>(row + intermediate_size) * K_IN);

  half2 gate_acc0 = __float2half2_rn(0.0f);
  half2 gate_acc1 = __float2half2_rn(0.0f);
  half2 gate_acc2 = __float2half2_rn(0.0f);
  half2 gate_acc3 = __float2half2_rn(0.0f);
  half2 up_acc0 = __float2half2_rn(0.0f);
  half2 up_acc1 = __float2half2_rn(0.0f);
  half2 up_acc2 = __float2half2_rn(0.0f);
  half2 up_acc3 = __float2half2_rn(0.0f);

#pragma unroll
  for (int vec = lane; vec < VEC_COUNT; vec += 32) {
    uint4 gv = gate_w4[vec];
    uint4 uv = up_w4[vec];
    int k = vec * VEC_PAIRS * 2;

    half2 x0 = __halves2half2(x[k + 0], x[k + 1]);
    half2 x1 = __halves2half2(x[k + 2], x[k + 3]);
    half2 x2 = __halves2half2(x[k + 4], x[k + 5]);
    half2 x3 = __halves2half2(x[k + 6], x[k + 7]);

    gate_acc0 = __hfma2(x0, half2_from_u32(gv.x), gate_acc0);
    gate_acc1 = __hfma2(x1, half2_from_u32(gv.y), gate_acc1);
    gate_acc2 = __hfma2(x2, half2_from_u32(gv.z), gate_acc2);
    gate_acc3 = __hfma2(x3, half2_from_u32(gv.w), gate_acc3);
    up_acc0 = __hfma2(x0, half2_from_u32(uv.x), up_acc0);
    up_acc1 = __hfma2(x1, half2_from_u32(uv.y), up_acc1);
    up_acc2 = __hfma2(x2, half2_from_u32(uv.z), up_acc2);
    up_acc3 = __hfma2(x3, half2_from_u32(uv.w), up_acc3);
  }

  float gate =
      __half2float(__low2half(gate_acc0)) + __half2float(__high2half(gate_acc0)) +
      __half2float(__low2half(gate_acc1)) + __half2float(__high2half(gate_acc1)) +
      __half2float(__low2half(gate_acc2)) + __half2float(__high2half(gate_acc2)) +
      __half2float(__low2half(gate_acc3)) + __half2float(__high2half(gate_acc3));
  float up =
      __half2float(__low2half(up_acc0)) + __half2float(__high2half(up_acc0)) +
      __half2float(__low2half(up_acc1)) + __half2float(__high2half(up_acc1)) +
      __half2float(__low2half(up_acc2)) + __half2float(__high2half(up_acc2)) +
      __half2float(__low2half(up_acc3)) + __half2float(__high2half(up_acc3));
  gate = warp_sum(gate);
  up = warp_sum(up);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

__global__ void gate_up_swiglu_half2_vec4_interleaved_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ packed_weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  constexpr int K_IN = 2048;
  constexpr int K_PAIRS = K_IN / 2;
  constexpr int VEC_PAIRS = 4;
  constexpr int VEC_COUNT = K_PAIRS / VEC_PAIRS;
  constexpr int PACKED_ROW_HALFS = K_IN * 2;
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const uint4* x4 = reinterpret_cast<const uint4*>(x);
  const uint4* packed4 = reinterpret_cast<const uint4*>(packed_weight + static_cast<int64_t>(row) * PACKED_ROW_HALFS);

  half2 gate_acc0 = __float2half2_rn(0.0f);
  half2 gate_acc1 = __float2half2_rn(0.0f);
  half2 gate_acc2 = __float2half2_rn(0.0f);
  half2 gate_acc3 = __float2half2_rn(0.0f);
  half2 up_acc0 = __float2half2_rn(0.0f);
  half2 up_acc1 = __float2half2_rn(0.0f);
  half2 up_acc2 = __float2half2_rn(0.0f);
  half2 up_acc3 = __float2half2_rn(0.0f);

#pragma unroll
  for (int vec = lane; vec < VEC_COUNT; vec += 32) {
    uint4 xv = x4[vec];
    uint4 p0 = packed4[vec * 2 + 0];
    uint4 p1 = packed4[vec * 2 + 1];

    half2 x0 = half2_from_u32(xv.x);
    half2 x1 = half2_from_u32(xv.y);
    half2 x2 = half2_from_u32(xv.z);
    half2 x3 = half2_from_u32(xv.w);

    gate_acc0 = __hfma2(x0, half2_from_u32(p0.x), gate_acc0);
    up_acc0 = __hfma2(x0, half2_from_u32(p0.y), up_acc0);
    gate_acc1 = __hfma2(x1, half2_from_u32(p0.z), gate_acc1);
    up_acc1 = __hfma2(x1, half2_from_u32(p0.w), up_acc1);
    gate_acc2 = __hfma2(x2, half2_from_u32(p1.x), gate_acc2);
    up_acc2 = __hfma2(x2, half2_from_u32(p1.y), up_acc2);
    gate_acc3 = __hfma2(x3, half2_from_u32(p1.z), gate_acc3);
    up_acc3 = __hfma2(x3, half2_from_u32(p1.w), up_acc3);
  }

  float gate =
      __half2float(__low2half(gate_acc0)) + __half2float(__high2half(gate_acc0)) +
      __half2float(__low2half(gate_acc1)) + __half2float(__high2half(gate_acc1)) +
      __half2float(__low2half(gate_acc2)) + __half2float(__high2half(gate_acc2)) +
      __half2float(__low2half(gate_acc3)) + __half2float(__high2half(gate_acc3));
  float up =
      __half2float(__low2half(up_acc0)) + __half2float(__high2half(up_acc0)) +
      __half2float(__low2half(up_acc1)) + __half2float(__high2half(up_acc1)) +
      __half2float(__low2half(up_acc2)) + __half2float(__high2half(up_acc2)) +
      __half2float(__low2half(up_acc3)) + __half2float(__high2half(up_acc3));
  gate = warp_sum(gate);
  up = warp_sum(up);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

__global__ void gate_up_swiglu_half2_vec8_haccum_k2048_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int rows_per_block) {
  constexpr int K_IN = 2048;
  constexpr int K_PAIRS = K_IN / 2;
  constexpr int VEC_PAIRS = 4;
  constexpr int VEC_COUNT = K_PAIRS / VEC_PAIRS;
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;
  if (row >= intermediate_size) {
    return;
  }

  const uint4* x4 = reinterpret_cast<const uint4*>(x);
  const uint4* gate_w4 = reinterpret_cast<const uint4*>(weight + static_cast<int64_t>(row) * K_IN);
  const uint4* up_w4 = reinterpret_cast<const uint4*>(
      weight + static_cast<int64_t>(row + intermediate_size) * K_IN);

  half2 gate_acc0 = __float2half2_rn(0.0f);
  half2 gate_acc1 = __float2half2_rn(0.0f);
  half2 gate_acc2 = __float2half2_rn(0.0f);
  half2 gate_acc3 = __float2half2_rn(0.0f);
  half2 gate_acc4 = __float2half2_rn(0.0f);
  half2 gate_acc5 = __float2half2_rn(0.0f);
  half2 gate_acc6 = __float2half2_rn(0.0f);
  half2 gate_acc7 = __float2half2_rn(0.0f);
  half2 up_acc0 = __float2half2_rn(0.0f);
  half2 up_acc1 = __float2half2_rn(0.0f);
  half2 up_acc2 = __float2half2_rn(0.0f);
  half2 up_acc3 = __float2half2_rn(0.0f);
  half2 up_acc4 = __float2half2_rn(0.0f);
  half2 up_acc5 = __float2half2_rn(0.0f);
  half2 up_acc6 = __float2half2_rn(0.0f);
  half2 up_acc7 = __float2half2_rn(0.0f);

#pragma unroll
  for (int vec = lane; vec < VEC_COUNT; vec += 64) {
    uint4 xv0 = x4[vec];
    uint4 gv0 = gate_w4[vec];
    uint4 uv0 = up_w4[vec];

    half2 x0 = half2_from_u32(xv0.x);
    half2 x1 = half2_from_u32(xv0.y);
    half2 x2 = half2_from_u32(xv0.z);
    half2 x3 = half2_from_u32(xv0.w);

    gate_acc0 = __hfma2(x0, half2_from_u32(gv0.x), gate_acc0);
    gate_acc1 = __hfma2(x1, half2_from_u32(gv0.y), gate_acc1);
    gate_acc2 = __hfma2(x2, half2_from_u32(gv0.z), gate_acc2);
    gate_acc3 = __hfma2(x3, half2_from_u32(gv0.w), gate_acc3);
    up_acc0 = __hfma2(x0, half2_from_u32(uv0.x), up_acc0);
    up_acc1 = __hfma2(x1, half2_from_u32(uv0.y), up_acc1);
    up_acc2 = __hfma2(x2, half2_from_u32(uv0.z), up_acc2);
    up_acc3 = __hfma2(x3, half2_from_u32(uv0.w), up_acc3);

    int vec1 = vec + 32;
    uint4 xv1 = x4[vec1];
    uint4 gv1 = gate_w4[vec1];
    uint4 uv1 = up_w4[vec1];

    half2 x4v = half2_from_u32(xv1.x);
    half2 x5 = half2_from_u32(xv1.y);
    half2 x6 = half2_from_u32(xv1.z);
    half2 x7 = half2_from_u32(xv1.w);

    gate_acc4 = __hfma2(x4v, half2_from_u32(gv1.x), gate_acc4);
    gate_acc5 = __hfma2(x5, half2_from_u32(gv1.y), gate_acc5);
    gate_acc6 = __hfma2(x6, half2_from_u32(gv1.z), gate_acc6);
    gate_acc7 = __hfma2(x7, half2_from_u32(gv1.w), gate_acc7);
    up_acc4 = __hfma2(x4v, half2_from_u32(uv1.x), up_acc4);
    up_acc5 = __hfma2(x5, half2_from_u32(uv1.y), up_acc5);
    up_acc6 = __hfma2(x6, half2_from_u32(uv1.z), up_acc6);
    up_acc7 = __hfma2(x7, half2_from_u32(uv1.w), up_acc7);
  }

  float gate =
      __half2float(__low2half(gate_acc0)) + __half2float(__high2half(gate_acc0)) +
      __half2float(__low2half(gate_acc1)) + __half2float(__high2half(gate_acc1)) +
      __half2float(__low2half(gate_acc2)) + __half2float(__high2half(gate_acc2)) +
      __half2float(__low2half(gate_acc3)) + __half2float(__high2half(gate_acc3)) +
      __half2float(__low2half(gate_acc4)) + __half2float(__high2half(gate_acc4)) +
      __half2float(__low2half(gate_acc5)) + __half2float(__high2half(gate_acc5)) +
      __half2float(__low2half(gate_acc6)) + __half2float(__high2half(gate_acc6)) +
      __half2float(__low2half(gate_acc7)) + __half2float(__high2half(gate_acc7));
  float up =
      __half2float(__low2half(up_acc0)) + __half2float(__high2half(up_acc0)) +
      __half2float(__low2half(up_acc1)) + __half2float(__high2half(up_acc1)) +
      __half2float(__low2half(up_acc2)) + __half2float(__high2half(up_acc2)) +
      __half2float(__low2half(up_acc3)) + __half2float(__high2half(up_acc3)) +
      __half2float(__low2half(up_acc4)) + __half2float(__high2half(up_acc4)) +
      __half2float(__low2half(up_acc5)) + __half2float(__high2half(up_acc5)) +
      __half2float(__low2half(up_acc6)) + __half2float(__high2half(up_acc6)) +
      __half2float(__low2half(up_acc7)) + __half2float(__high2half(up_acc7));
  gate = warp_sum(gate);
  up = warp_sum(up);

  if (lane == 0) {
    float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int UNROLL>
__global__ void gate_up_swiglu_shared_x_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int k_in,
    int rows_per_block,
    int block_k) {
  extern __shared__ half x_tile[];
  int row_in_block = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int row = blockIdx.x * rows_per_block + row_in_block;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (int k0 = 0; k0 < k_in; k0 += block_k) {
    int tile_k = min(block_k, k_in - k0);
    for (int k = threadIdx.x; k < tile_k; k += blockDim.x) {
      x_tile[k] = x[k0 + k];
    }
    __syncthreads();

    if (row < intermediate_size) {
      const half* gate_w = weight + static_cast<int64_t>(row) * k_in + k0;
      const half* up_w = weight + static_cast<int64_t>(row + intermediate_size) * k_in + k0;
      for (int k_base = lane; k_base < tile_k; k_base += 32 * UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
          int k = k_base + u * 32;
          if (k < tile_k) {
            float xv = __half2float(x_tile[k]);
            gate_acc = fmaf(xv, __half2float(gate_w[k]), gate_acc);
            up_acc = fmaf(xv, __half2float(up_w[k]), up_acc);
          }
        }
      }
    }
    __syncthreads();
  }

  if (row >= intermediate_size) {
    return;
  }
  gate_acc = warp_sum(gate_acc);
  up_acc = warp_sum(up_acc);

  if (lane == 0) {
    float swish = gate_acc / (1.0f + expf(-gate_acc));
    out[row] = __float2half_rn(swish * up_acc);
  }
}

template <int ROWS_PER_WARP, int UNROLL>
__global__ void gate_up_swiglu_multirow_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int k_in) {
  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int row_base = (blockIdx.x * warps_per_block + warp_in_block) * ROWS_PER_WARP;

  float gate_acc[ROWS_PER_WARP];
  float up_acc[ROWS_PER_WARP];
#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    gate_acc[r] = 0.0f;
    up_acc[r] = 0.0f;
  }

  for (int k_base = lane; k_base < k_in; k_base += 32 * UNROLL) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int k = k_base + u * 32;
      if (k < k_in) {
        float xv = __half2float(x[k]);
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
          int row = row_base + r;
          if (row < intermediate_size) {
            const half* gate_w = weight + static_cast<int64_t>(row) * k_in;
            const half* up_w = weight + static_cast<int64_t>(row + intermediate_size) * k_in;
            gate_acc[r] = fmaf(xv, __half2float(gate_w[k]), gate_acc[r]);
            up_acc[r] = fmaf(xv, __half2float(up_w[k]), up_acc[r]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    int row = row_base + r;
    float gate = warp_sum(gate_acc[r]);
    float up = warp_sum(up_acc[r]);
    if (lane == 0 && row < intermediate_size) {
      float swish = gate / (1.0f + expf(-gate));
      out[row] = __float2half_rn(swish * up);
    }
  }
}

template <int ROWS_PER_WARP, int UNROLL>
__global__ void gate_up_swiglu_half2_multirow_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int intermediate_size,
    int k_in) {
  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int row_base = (blockIdx.x * warps_per_block + warp_in_block) * ROWS_PER_WARP;
  int k_pairs = k_in >> 1;
  const half2* x2 = reinterpret_cast<const half2*>(x);

  float gate_acc[ROWS_PER_WARP];
  float up_acc[ROWS_PER_WARP];
#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    gate_acc[r] = 0.0f;
    up_acc[r] = 0.0f;
  }

  for (int k_base = lane; k_base < k_pairs; k_base += 32 * UNROLL) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int k2 = k_base + u * 32;
      if (k2 < k_pairs) {
        half2 xv = x2[k2];
#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
          int row = row_base + r;
          if (row < intermediate_size) {
            const half2* gate_w = reinterpret_cast<const half2*>(
                weight + static_cast<int64_t>(row) * k_in);
            const half2* up_w = reinterpret_cast<const half2*>(
                weight + static_cast<int64_t>(row + intermediate_size) * k_in);
            gate_acc[r] = dot_acc_half2(gate_acc[r], xv, gate_w[k2]);
            up_acc[r] = dot_acc_half2(up_acc[r], xv, up_w[k2]);
          }
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_WARP; ++r) {
    int row = row_base + r;
    float gate = warp_sum(gate_acc[r]);
    float up = warp_sum(up_acc[r]);
    if (lane == 0 && row < intermediate_size) {
      float swish = gate / (1.0f + expf(-gate));
      out[row] = __float2half_rn(swish * up);
    }
  }
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(out);
  CHECK_HALF(x);
  CHECK_HALF(weight);
  CHECK_HALF(out);
  TORCH_CHECK(weight.dim() == 2, "weight must be [2 * intermediate, K]");
  TORCH_CHECK(out.numel() == intermediate_size, "out must have intermediate_size elements");
  TORCH_CHECK(x.numel() == weight.size(1), "x.numel() must match weight.shape[1]");
  TORCH_CHECK(weight.size(0) >= 2 * intermediate_size, "weight must contain gate rows followed by up rows");
  TORCH_CHECK(rows_per_block >= 1 && rows_per_block <= 32, "rows_per_block must be in [1, 32]");
}

void check_unroll(int64_t unroll) {
  TORCH_CHECK(
      unroll == 1 || unroll == 2 || unroll == 3 || unroll == 4 || unroll == 5 || unroll == 6 ||
          unroll == 8 || unroll == 12 || unroll == 16,
      "unroll must be one of 1, 2, 3, 4, 5, 6, 8, 12, 16");
}

void check_warps_per_block(int64_t warps_per_block) {
  TORCH_CHECK(
      warps_per_block == 1 || warps_per_block == 2 || warps_per_block == 4 || warps_per_block == 8,
      "warps_per_block must be one of 1, 2, 4, 8");
}

void check_rows_per_warp(int64_t rows_per_warp) {
  TORCH_CHECK(
      rows_per_warp == 1 || rows_per_warp == 2 || rows_per_warp == 4 || rows_per_warp == 8,
      "rows_per_warp must be one of 1, 2, 4, 8");
}

void launch_gate_up_swiglu_legacy(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t k_in,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_legacy_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(k_in),
      static_cast<int>(rows_per_block));
}

template <int UNROLL>
void launch_gate_up_swiglu(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t k_in,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_kernel<UNROLL><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(k_in),
      static_cast<int>(rows_per_block));
}

template <int UNROLL>
void launch_gate_up_swiglu_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_k2048_kernel<UNROLL><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
}

template <int UNROLL>
void launch_gate_up_swiglu_half2_haccum_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_haccum_k2048_kernel<UNROLL><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
}

template <int UNROLL>
void launch_gate_up_swiglu_half2_haccum4_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_haccum4_k2048_kernel<UNROLL><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
}

void launch_gate_up_swiglu_half2_vec4_haccum_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_vec4_haccum_k2048_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
}

template <int LOAD_MODE>
void launch_gate_up_swiglu_half2_vec4_loadmode_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_vec4_loadmode_k2048_kernel<LOAD_MODE><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
}

void launch_gate_up_swiglu_half2_vec4_haccum_nox_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_vec4_haccum_nox_k2048_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
}

void launch_gate_up_swiglu_half2_vec4_interleaved_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_vec4_interleaved_k2048_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
}

void launch_gate_up_swiglu_half2_vec8_haccum_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_vec8_haccum_k2048_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(rows_per_block));
}

template <int UNROLL>
void launch_gate_up_swiglu_shared_x(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t k_in,
    int64_t rows_per_block,
    int64_t block_k) {
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(rows_per_block * 32);
  size_t shared_bytes = static_cast<size_t>(block_k) * sizeof(half);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_shared_x_kernel<UNROLL><<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(k_in),
      static_cast<int>(rows_per_block),
      static_cast<int>(block_k));
}

template <int ROWS_PER_WARP, int UNROLL>
void launch_gate_up_swiglu_multirow(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t k_in,
    int64_t warps_per_block) {
  int rows_per_block = static_cast<int>(ROWS_PER_WARP * warps_per_block);
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(warps_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_multirow_kernel<ROWS_PER_WARP, UNROLL><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(k_in));
}

template <int ROWS_PER_WARP, int UNROLL>
void launch_gate_up_swiglu_half2_multirow(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t k_in,
    int64_t warps_per_block) {
  int rows_per_block = static_cast<int>(ROWS_PER_WARP * warps_per_block);
  int blocks = static_cast<int>((intermediate_size + rows_per_block - 1) / rows_per_block);
  int threads = static_cast<int>(warps_per_block * 32);
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_half2_multirow_kernel<ROWS_PER_WARP, UNROLL><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<int>(intermediate_size),
      static_cast<int>(k_in));
}

template <int UNROLL>
void dispatch_rows_multirow(
    int64_t rows_per_warp,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t k_in,
    int64_t warps_per_block) {
  switch (rows_per_warp) {
    case 1:
      launch_gate_up_swiglu_multirow<1, UNROLL>(x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 2:
      launch_gate_up_swiglu_multirow<2, UNROLL>(x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 4:
      launch_gate_up_swiglu_multirow<4, UNROLL>(x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 8:
      launch_gate_up_swiglu_multirow<8, UNROLL>(x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
  }
}

template <int UNROLL>
void dispatch_rows_half2_multirow(
    int64_t rows_per_warp,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& out,
    int64_t intermediate_size,
    int64_t k_in,
    int64_t warps_per_block) {
  switch (rows_per_warp) {
    case 1:
      launch_gate_up_swiglu_half2_multirow<1, UNROLL>(x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 2:
      launch_gate_up_swiglu_half2_multirow<2, UNROLL>(x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 4:
      launch_gate_up_swiglu_half2_multirow<4, UNROLL>(x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 8:
      launch_gate_up_swiglu_half2_multirow<8, UNROLL>(x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
  }
}

}  // namespace

void gate_up_swiglu(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block,
    int64_t unroll) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  check_unroll(unroll);
  int64_t k_in = weight.size(1);

  switch (unroll) {
    case 1:
      launch_gate_up_swiglu<1>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
    case 2:
      launch_gate_up_swiglu<2>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
    case 3:
      launch_gate_up_swiglu<3>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
    case 4:
      launch_gate_up_swiglu<4>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
    case 5:
      launch_gate_up_swiglu<5>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
    case 6:
      launch_gate_up_swiglu<6>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
    case 8:
      launch_gate_up_swiglu<8>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
    case 12:
      launch_gate_up_swiglu<12>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
    case 16:
      launch_gate_up_swiglu<16>(x, weight, out, intermediate_size, k_in, rows_per_block);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_legacy(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  int64_t k_in = weight.size(1);
  launch_gate_up_swiglu_legacy(x, weight, out, intermediate_size, k_in, rows_per_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block,
    int64_t unroll) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  check_unroll(unroll);
  TORCH_CHECK(weight.size(1) == 2048, "gate_up_swiglu_k2048 requires K=2048");

  switch (unroll) {
    case 1:
      launch_gate_up_swiglu_k2048<1>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 2:
      launch_gate_up_swiglu_k2048<2>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 3:
      launch_gate_up_swiglu_k2048<3>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 4:
      launch_gate_up_swiglu_k2048<4>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 5:
      launch_gate_up_swiglu_k2048<5>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 6:
      launch_gate_up_swiglu_k2048<6>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 8:
      launch_gate_up_swiglu_k2048<8>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 12:
      launch_gate_up_swiglu_k2048<12>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 16:
      launch_gate_up_swiglu_k2048<16>(x, weight, out, intermediate_size, rows_per_block);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_half2_haccum_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block,
    int64_t unroll) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  check_unroll(unroll);
  TORCH_CHECK(weight.size(1) == 2048, "gate_up_swiglu_half2_haccum_k2048 requires K=2048");

  switch (unroll) {
    case 1:
      launch_gate_up_swiglu_half2_haccum_k2048<1>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 2:
      launch_gate_up_swiglu_half2_haccum_k2048<2>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 3:
      launch_gate_up_swiglu_half2_haccum_k2048<3>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 4:
      launch_gate_up_swiglu_half2_haccum_k2048<4>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 5:
      launch_gate_up_swiglu_half2_haccum_k2048<5>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 6:
      launch_gate_up_swiglu_half2_haccum_k2048<6>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 8:
      launch_gate_up_swiglu_half2_haccum_k2048<8>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 12:
      launch_gate_up_swiglu_half2_haccum_k2048<12>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 16:
      launch_gate_up_swiglu_half2_haccum_k2048<16>(x, weight, out, intermediate_size, rows_per_block);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_half2_haccum4_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block,
    int64_t unroll) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  check_unroll(unroll);
  TORCH_CHECK(weight.size(1) == 2048, "gate_up_swiglu_half2_haccum4_k2048 requires K=2048");

  switch (unroll) {
    case 1:
      launch_gate_up_swiglu_half2_haccum4_k2048<1>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 2:
      launch_gate_up_swiglu_half2_haccum4_k2048<2>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 3:
      launch_gate_up_swiglu_half2_haccum4_k2048<3>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 4:
      launch_gate_up_swiglu_half2_haccum4_k2048<4>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 5:
      launch_gate_up_swiglu_half2_haccum4_k2048<5>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 6:
      launch_gate_up_swiglu_half2_haccum4_k2048<6>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 8:
      launch_gate_up_swiglu_half2_haccum4_k2048<8>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 12:
      launch_gate_up_swiglu_half2_haccum4_k2048<12>(x, weight, out, intermediate_size, rows_per_block);
      break;
    case 16:
      launch_gate_up_swiglu_half2_haccum4_k2048<16>(x, weight, out, intermediate_size, rows_per_block);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_half2_vec4_haccum_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  TORCH_CHECK(weight.size(1) == 2048, "gate_up_swiglu_half2_vec4_haccum_k2048 requires K=2048");
  TORCH_CHECK(rows_per_block >= 1 && rows_per_block <= 32, "rows_per_block must be in [1, 32]");
  launch_gate_up_swiglu_half2_vec4_haccum_k2048(x, weight, out, intermediate_size, rows_per_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_half2_vec4_loadmode_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block,
    int64_t load_mode) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  TORCH_CHECK(weight.size(1) == 2048, "gate_up_swiglu_half2_vec4_loadmode_k2048 requires K=2048");
  TORCH_CHECK(load_mode == 1 || load_mode == 2, "load_mode must be 1 (cs) or 2 (cg)");
  if (load_mode == 1) {
    launch_gate_up_swiglu_half2_vec4_loadmode_k2048<1>(x, weight, out, intermediate_size, rows_per_block);
  } else {
    launch_gate_up_swiglu_half2_vec4_loadmode_k2048<2>(x, weight, out, intermediate_size, rows_per_block);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_half2_vec4_haccum_nox_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  TORCH_CHECK(weight.size(1) == 2048, "gate_up_swiglu_half2_vec4_haccum_nox_k2048 requires K=2048");
  launch_gate_up_swiglu_half2_vec4_haccum_nox_k2048(x, weight, out, intermediate_size, rows_per_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_half2_vec4_interleaved_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(out);
  CHECK_HALF(x);
  CHECK_HALF(weight);
  CHECK_HALF(out);
  TORCH_CHECK(out.numel() == intermediate_size, "out must have intermediate_size elements");
  TORCH_CHECK(x.numel() == 2048, "x.numel() must be 2048");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) >= intermediate_size, "packed weight must be [intermediate, 4096]");
  TORCH_CHECK(rows_per_block >= 1 && rows_per_block <= 32, "rows_per_block must be in [1, 32]");
  TORCH_CHECK(weight.size(1) == 4096, "gate_up_swiglu_half2_vec4_interleaved_k2048 requires packed K=4096");
  launch_gate_up_swiglu_half2_vec4_interleaved_k2048(x, weight, out, intermediate_size, rows_per_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_half2_vec8_haccum_k2048(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  TORCH_CHECK(weight.size(1) == 2048, "gate_up_swiglu_half2_vec8_haccum_k2048 requires K=2048");
  launch_gate_up_swiglu_half2_vec8_haccum_k2048(x, weight, out, intermediate_size, rows_per_block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_shared_x(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_block,
    int64_t block_k,
    int64_t unroll) {
  check_inputs(x, weight, out, intermediate_size, rows_per_block);
  check_unroll(unroll);
  TORCH_CHECK(block_k >= 32 && block_k <= weight.size(1), "block_k must be in [32, K]");
  int64_t k_in = weight.size(1);

  switch (unroll) {
    case 1:
      launch_gate_up_swiglu_shared_x<1>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
    case 2:
      launch_gate_up_swiglu_shared_x<2>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
    case 3:
      launch_gate_up_swiglu_shared_x<3>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
    case 4:
      launch_gate_up_swiglu_shared_x<4>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
    case 5:
      launch_gate_up_swiglu_shared_x<5>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
    case 6:
      launch_gate_up_swiglu_shared_x<6>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
    case 8:
      launch_gate_up_swiglu_shared_x<8>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
    case 12:
      launch_gate_up_swiglu_shared_x<12>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
    case 16:
      launch_gate_up_swiglu_shared_x<16>(x, weight, out, intermediate_size, k_in, rows_per_block, block_k);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_multirow(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_warp,
    int64_t warps_per_block,
    int64_t unroll) {
  check_inputs(x, weight, out, intermediate_size, 1);
  check_rows_per_warp(rows_per_warp);
  check_warps_per_block(warps_per_block);
  check_unroll(unroll);
  int64_t k_in = weight.size(1);

  switch (unroll) {
    case 1:
      dispatch_rows_multirow<1>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 2:
      dispatch_rows_multirow<2>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 3:
      dispatch_rows_multirow<3>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 4:
      dispatch_rows_multirow<4>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 5:
      dispatch_rows_multirow<5>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 6:
      dispatch_rows_multirow<6>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 8:
      dispatch_rows_multirow<8>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 12:
      dispatch_rows_multirow<12>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 16:
      dispatch_rows_multirow<16>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_up_swiglu_half2_multirow(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor out,
    int64_t intermediate_size,
    int64_t rows_per_warp,
    int64_t warps_per_block,
    int64_t unroll) {
  check_inputs(x, weight, out, intermediate_size, 1);
  check_rows_per_warp(rows_per_warp);
  check_warps_per_block(warps_per_block);
  check_unroll(unroll);
  int64_t k_in = weight.size(1);
  TORCH_CHECK((k_in & 1) == 0, "half2 multirow requires an even K dimension");

  switch (unroll) {
    case 1:
      dispatch_rows_half2_multirow<1>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 2:
      dispatch_rows_half2_multirow<2>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 3:
      dispatch_rows_half2_multirow<3>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 4:
      dispatch_rows_half2_multirow<4>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 5:
      dispatch_rows_half2_multirow<5>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 6:
      dispatch_rows_half2_multirow<6>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 8:
      dispatch_rows_half2_multirow<8>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 12:
      dispatch_rows_half2_multirow<12>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
    case 16:
      dispatch_rows_half2_multirow<16>(rows_per_warp, x, weight, out, intermediate_size, k_in, warps_per_block);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate_up_swiglu", &gate_up_swiglu, "decode gate/up SwiGLU");
  m.def("gate_up_swiglu_legacy", &gate_up_swiglu_legacy, "legacy decode gate/up SwiGLU");
  m.def("gate_up_swiglu_k2048", &gate_up_swiglu_k2048, "decode gate/up SwiGLU specialized for K=2048");
  m.def(
      "gate_up_swiglu_half2_haccum_k2048",
      &gate_up_swiglu_half2_haccum_k2048,
      "decode gate/up SwiGLU specialized for K=2048 with fp16 lane accumulation");
  m.def(
      "gate_up_swiglu_half2_haccum4_k2048",
      &gate_up_swiglu_half2_haccum4_k2048,
      "decode gate/up SwiGLU specialized for K=2048 with four fp16 accumulator chains");
  m.def(
      "gate_up_swiglu_half2_vec4_haccum_k2048",
      &gate_up_swiglu_half2_vec4_haccum_k2048,
      "decode gate/up SwiGLU specialized for K=2048 with uint4 fp16 lane accumulation");
  m.def(
      "gate_up_swiglu_half2_vec4_loadmode_k2048",
      &gate_up_swiglu_half2_vec4_loadmode_k2048,
      "decode gate/up SwiGLU specialized for K=2048 with explicit cache load mode");
  m.def(
      "gate_up_swiglu_half2_vec4_haccum_nox_k2048",
      &gate_up_swiglu_half2_vec4_haccum_nox_k2048,
      "decode gate/up SwiGLU specialized for K=2048 with uint4 weights and scalar x loads");
  m.def(
      "gate_up_swiglu_half2_vec4_interleaved_k2048",
      &gate_up_swiglu_half2_vec4_interleaved_k2048,
      "decode gate/up SwiGLU specialized for K=2048 with interleaved gate/up weights");
  m.def(
      "gate_up_swiglu_half2_vec8_haccum_k2048",
      &gate_up_swiglu_half2_vec8_haccum_k2048,
      "decode gate/up SwiGLU specialized for K=2048 with two uint4 groups per loop");
  m.def("gate_up_swiglu_shared_x", &gate_up_swiglu_shared_x, "decode gate/up SwiGLU with shared x tile");
  m.def("gate_up_swiglu_multirow", &gate_up_swiglu_multirow, "decode gate/up SwiGLU with row reuse inside one warp");
  m.def(
      "gate_up_swiglu_half2_multirow",
      &gate_up_swiglu_half2_multirow,
      "decode gate/up SwiGLU with half2 row reuse inside one warp");
}
