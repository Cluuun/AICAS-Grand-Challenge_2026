#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be fp16")
#define CHECK_BYTE(x) TORCH_CHECK((x).scalar_type() == at::kByte, #x " must be uint8")

constexpr int kHiddenSize = 2048;
constexpr int kIntermediateSize = 6144;
constexpr int kBlockK = 128;
constexpr int kScaleBlocks = kHiddenSize / kBlockK;
constexpr int kPackedBytesPerBlock = kBlockK / 2;
constexpr int kPackedWordsPerBlock = kPackedBytesPerBlock / 4;
constexpr int kKblockWords = 2 * kIntermediateSize * kPackedWordsPerBlock;
constexpr int kKblockScaleStride = 2 * kIntermediateSize;

__forceinline__ __device__ float warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__forceinline__ __device__ float half_warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, 16);
  }
  return value;
}

__forceinline__ __device__ float quarter_warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, 8);
  }
  return value;
}

__forceinline__ __device__ uint32_t load_u32_cg(const uint32_t* ptr) {
  return __ldcg(ptr);
}

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ float dot_acc_int4_pair(float acc, uint32_t byte_value, half2 x_pair) {
  const float2 xv = __half22float2(x_pair);
  const float q0 = static_cast<float>(static_cast<int>(byte_value & 0x0fu) - 8);
  const float q1 = static_cast<float>(static_cast<int>((byte_value >> 4) & 0x0fu) - 8);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  return acc;
}

__forceinline__ __device__ float dot_acc_int4_pair(float acc, uint32_t byte_value, float2 xv) {
  const float q0 = static_cast<float>(static_cast<int>(byte_value & 0x0fu) - 8);
  const float q1 = static_cast<float>(static_cast<int>((byte_value >> 4) & 0x0fu) - 8);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  return acc;
}

__forceinline__ __device__ void dot_acc_u4_pair(uint32_t byte_value, float2 xv, float& acc, float& x_sum) {
  const float q0 = static_cast<float>(byte_value & 0x0fu);
  const float q1 = static_cast<float>((byte_value >> 4) & 0x0fu);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  x_sum += xv.x + xv.y;
}

__forceinline__ __device__ float dot_acc_u4_pair(float acc, uint32_t byte_value, float2 xv) {
  const float q0 = static_cast<float>(byte_value & 0x0fu);
  const float q1 = static_cast<float>((byte_value >> 4) & 0x0fu);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  return acc;
}

__forceinline__ __device__ void dot8_u4xfloat2(
    uint32_t packed,
    float2 x0,
    float2 x1,
    float2 x2,
    float2 x3,
    float& acc,
    float& x_sum) {
  dot_acc_u4_pair(packed & 0xffu, x0, acc, x_sum);
  dot_acc_u4_pair((packed >> 8) & 0xffu, x1, acc, x_sum);
  dot_acc_u4_pair((packed >> 16) & 0xffu, x2, acc, x_sum);
  dot_acc_u4_pair((packed >> 24) & 0xffu, x3, acc, x_sum);
}

__forceinline__ __device__ float dot8_u4xfloat2(
    uint32_t packed,
    float2 x0,
    float2 x1,
    float2 x2,
    float2 x3) {
  float acc = 0.0f;
  acc = dot_acc_u4_pair(acc, packed & 0xffu, x0);
  acc = dot_acc_u4_pair(acc, (packed >> 8) & 0xffu, x1);
  acc = dot_acc_u4_pair(acc, (packed >> 16) & 0xffu, x2);
  acc = dot_acc_u4_pair(acc, (packed >> 24) & 0xffu, x3);
  return acc;
}

__forceinline__ __device__ float dot4_u4xfloat2(
    uint32_t packed,
    float2 x0,
    float2 x1,
    float& x_sum) {
  float acc = 0.0f;
  dot_acc_u4_pair(packed & 0xffu, x0, acc, x_sum);
  dot_acc_u4_pair((packed >> 8) & 0xffu, x1, acc, x_sum);
  return acc;
}

__forceinline__ __device__ float sum4_float2(float2 x0, float2 x1, float2 x2, float2 x3) {
  return x0.x + x0.y + x1.x + x1.y + x2.x + x2.y + x3.x + x3.y;
}

__forceinline__ __device__ float dot8_int4xhalf(uint32_t packed, const half2* __restrict__ x2) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed & 0xffu, x2[0]);
  acc = dot_acc_int4_pair(acc, (packed >> 8) & 0xffu, x2[1]);
  acc = dot_acc_int4_pair(acc, (packed >> 16) & 0xffu, x2[2]);
  acc = dot_acc_int4_pair(acc, (packed >> 24) & 0xffu, x2[3]);
  return acc;
}

__forceinline__ __device__ float dot4_int4xhalf(uint32_t packed, const half2* __restrict__ x2) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed & 0xffu, x2[0]);
  acc = dot_acc_int4_pair(acc, (packed >> 8) & 0xffu, x2[1]);
  return acc;
}

__forceinline__ __device__ float dot8_int4xfloat2(
    uint32_t packed,
    float2 x0,
    float2 x1,
    float2 x2,
    float2 x3) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed & 0xffu, x0);
  acc = dot_acc_int4_pair(acc, (packed >> 8) & 0xffu, x1);
  acc = dot_acc_int4_pair(acc, (packed >> 16) & 0xffu, x2);
  acc = dot_acc_int4_pair(acc, (packed >> 24) & 0xffu, x3);
  return acc;
}

__forceinline__ __device__ float dot16_int4xhalf(
    uint32_t packed0,
    uint32_t packed1,
    const half2* __restrict__ x2) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed0 & 0xffu, x2[0]);
  acc = dot_acc_int4_pair(acc, (packed0 >> 8) & 0xffu, x2[1]);
  acc = dot_acc_int4_pair(acc, (packed0 >> 16) & 0xffu, x2[2]);
  acc = dot_acc_int4_pair(acc, (packed0 >> 24) & 0xffu, x2[3]);
  acc = dot_acc_int4_pair(acc, packed1 & 0xffu, x2[4]);
  acc = dot_acc_int4_pair(acc, (packed1 >> 8) & 0xffu, x2[5]);
  acc = dot_acc_int4_pair(acc, (packed1 >> 16) & 0xffu, x2[6]);
  acc = dot_acc_int4_pair(acc, (packed1 >> 24) & 0xffu, x2[7]);
  return acc;
}

__forceinline__ __device__ float dot16_int4xfloat2(
    uint32_t packed0,
    uint32_t packed1,
    float2 x0,
    float2 x1,
    float2 x2,
    float2 x3,
    float2 x4,
    float2 x5,
    float2 x6,
    float2 x7) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed0 & 0xffu, x0);
  acc = dot_acc_int4_pair(acc, (packed0 >> 8) & 0xffu, x1);
  acc = dot_acc_int4_pair(acc, (packed0 >> 16) & 0xffu, x2);
  acc = dot_acc_int4_pair(acc, (packed0 >> 24) & 0xffu, x3);
  acc = dot_acc_int4_pair(acc, packed1 & 0xffu, x4);
  acc = dot_acc_int4_pair(acc, (packed1 >> 8) & 0xffu, x5);
  acc = dot_acc_int4_pair(acc, (packed1 >> 16) & 0xffu, x6);
  acc = dot_acc_int4_pair(acc, (packed1 >> 24) & 0xffu, x7);
  return acc;
}

__forceinline__ __device__ half2 load_half2_cg(const half2* ptr) {
  return half2_from_u32(__ldcg(reinterpret_cast<const uint32_t*>(ptr)));
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int row_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint16_t* weight_u16 = reinterpret_cast<const uint16_t*>(weight_int4);
  constexpr int kPackedU16PerBlock = kPackedBytesPerBlock / 2;
  constexpr int kKblockU16 = 2 * kIntermediateSize * kPackedU16PerBlock;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint16_t* block_u16 = weight_u16 + block_k * kKblockU16;
    const uint16_t* gate_u16 = block_u16 + row * kPackedU16PerBlock;
    const uint16_t* up_u16 = block_u16 + (kIntermediateSize + row) * kPackedU16PerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane * 2;
    const float gate_block_acc = dot4_int4xhalf(static_cast<uint32_t>(gate_u16[lane]), x2 + x_pair);
    const float up_block_acc = dot4_int4xhalf(static_cast<uint32_t>(up_u16[lane]), x2 + x_pair);

    const float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    const float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = warp_sum(gate_acc);
  const float up = warp_sum(up_acc);
  if (lane == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float gate_block_acc = dot8_int4xhalf(load_u32_cg(gate_w32 + lane16), x2 + x_pair);
    const float up_block_acc = dot8_int4xhalf(load_u32_cg(up_w32 + lane16), x2 + x_pair);

    const float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    const float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    float block_acc = 0.0f;
    block_acc = dot8_int4xhalf(load_u32_cg(gate_w32 + lane16), x2 + x_pair);
    gate_acc = fmaf(
        block_acc,
        __half2float(scales[block_k * kKblockScaleStride + row]),
        gate_acc);

    block_acc = dot8_int4xhalf(load_u32_cg(up_w32 + lane16), x2 + x_pair);
    up_acc = fmaf(
        block_acc,
        __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]),
        up_acc);
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock, bool kBroadcastScale, bool kSharedX, bool kCgX>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  const half2* x2_global = reinterpret_cast<const half2*>(x);
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);
  if constexpr (kSharedX) {
    for (int idx = threadIdx.x; idx < kHiddenSize / 2; idx += blockDim.x) {
      x2_shared[idx] = x2_global[idx];
    }
    __syncthreads();
  }
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = kSharedX ? x2_shared : x2_global;
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    half2 xh0;
    half2 xh1;
    half2 xh2;
    half2 xh3;
    if constexpr (kCgX) {
      xh0 = load_half2_cg(x2 + x_pair);
      xh1 = load_half2_cg(x2 + x_pair + 1);
      xh2 = load_half2_cg(x2 + x_pair + 2);
      xh3 = load_half2_cg(x2 + x_pair + 3);
    } else {
      xh0 = x2[x_pair];
      xh1 = x2[x_pair + 1];
      xh2 = x2[x_pair + 2];
      xh3 = x2[x_pair + 3];
    }
    const float2 x0 = __half22float2(xh0);
    const float2 x1 = __half22float2(xh1);
    const float2 x2v = __half22float2(xh2);
    const float2 x3 = __half22float2(xh3);

    const float gate_block_acc = dot8_int4xfloat2(load_u32_cg(gate_w32 + lane16), x0, x1, x2v, x3);
    const float up_block_acc = dot8_int4xfloat2(load_u32_cg(up_w32 + lane16), x0, x1, x2v, x3);

    float gate_scale;
    float up_scale;
    if constexpr (kBroadcastScale) {
      gate_scale = 0.0f;
      up_scale = 0.0f;
      if (lane16 == 0) {
        gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
        up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
      }
      gate_scale = __shfl_sync(0xffffffffu, gate_scale, 0, 16);
      up_scale = __shfl_sync(0xffffffffu, up_scale, 0, 16);
    } else {
      gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
      up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    }
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <
    int kRowsPerBlock,
    bool kBroadcastScale,
    bool kSharedX,
    bool kCgX,
    bool kBroadcastXSum,
    bool kSplitAccum>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  const half2* x2_global = reinterpret_cast<const half2*>(x);
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);
  if constexpr (kSharedX) {
    for (int idx = threadIdx.x; idx < kHiddenSize / 2; idx += blockDim.x) {
      x2_shared[idx] = x2_global[idx];
    }
    __syncthreads();
  }
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = kSharedX ? x2_shared : x2_global;
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  float gate_acc_alt = 0.0f;
  float up_acc_alt = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    half2 xh0;
    half2 xh1;
    half2 xh2;
    half2 xh3;
    if constexpr (kCgX) {
      xh0 = load_half2_cg(x2 + x_pair);
      xh1 = load_half2_cg(x2 + x_pair + 1);
      xh2 = load_half2_cg(x2 + x_pair + 2);
      xh3 = load_half2_cg(x2 + x_pair + 3);
    } else {
      xh0 = x2[x_pair];
      xh1 = x2[x_pair + 1];
      xh2 = x2[x_pair + 2];
      xh3 = x2[x_pair + 3];
    }
    const float2 x0 = __half22float2(xh0);
    const float2 x1 = __half22float2(xh1);
    const float2 x2v = __half22float2(xh2);
    const float2 x3 = __half22float2(xh3);

    float u4_x_sum = 0.0f;
    float gate_u4_acc = 0.0f;
    if constexpr (kBroadcastXSum) {
      gate_u4_acc = dot8_u4xfloat2(load_u32_cg(gate_w32 + lane16), x0, x1, x2v, x3);
      if (half_id == 0) {
        u4_x_sum = sum4_float2(x0, x1, x2v, x3);
      }
      u4_x_sum = __shfl_sync(0xffffffffu, u4_x_sum, lane16, 32);
    } else {
      dot8_u4xfloat2(load_u32_cg(gate_w32 + lane16), x0, x1, x2v, x3, gate_u4_acc, u4_x_sum);
    }
    const float up_u4_acc = dot8_u4xfloat2(load_u32_cg(up_w32 + lane16), x0, x1, x2v, x3);

    const float gate_block_acc = gate_u4_acc - 8.0f * u4_x_sum;
    const float up_block_acc = up_u4_acc - 8.0f * u4_x_sum;

    float gate_scale;
    float up_scale;
    if constexpr (kBroadcastScale) {
      gate_scale = 0.0f;
      up_scale = 0.0f;
      if (lane16 == 0) {
        gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
        up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
      }
      gate_scale = __shfl_sync(0xffffffffu, gate_scale, 0, 16);
      up_scale = __shfl_sync(0xffffffffu, up_scale, 0, 16);
    } else {
      gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
      up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    }
    if constexpr (kSplitAccum) {
      if ((block_k & 1) == 0) {
        gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
        up_acc = fmaf(up_block_acc, up_scale, up_acc);
      } else {
        gate_acc_alt = fmaf(gate_block_acc, gate_scale, gate_acc_alt);
        up_acc_alt = fmaf(up_block_acc, up_scale, up_acc_alt);
      }
    } else {
      gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
      up_acc = fmaf(up_block_acc, up_scale, up_acc);
    }
  }
  if constexpr (kSplitAccum) {
    gate_acc += gate_acc_alt;
    up_acc += up_acc_alt;
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock, bool kBroadcastScale, bool kSharedX, bool kCgX>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int pair_in_block = warp_id * 2 + half_id;
  const int row0 = blockIdx.x * kRowsPerBlock + pair_in_block * 2;
  const int row1 = row0 + 1;

  const half2* x2_global = reinterpret_cast<const half2*>(x);
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);
  if constexpr (kSharedX) {
    for (int idx = threadIdx.x; idx < kHiddenSize / 2; idx += blockDim.x) {
      x2_shared[idx] = x2_global[idx];
    }
    __syncthreads();
  }
  if (row0 >= kIntermediateSize) {
    return;
  }

  const bool valid_row1 = row1 < kIntermediateSize;
  const half2* x2 = kSharedX ? x2_shared : x2_global;
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate0_acc = 0.0f;
  float up0_acc = 0.0f;
  float gate1_acc = 0.0f;
  float up1_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate0_w32 = block_w32 + row0 * kPackedWordsPerBlock;
    const uint32_t* up0_w32 = block_w32 + (kIntermediateSize + row0) * kPackedWordsPerBlock;
    const uint32_t* gate1_w32 = valid_row1 ? (block_w32 + row1 * kPackedWordsPerBlock) : nullptr;
    const uint32_t* up1_w32 =
        valid_row1 ? (block_w32 + (kIntermediateSize + row1) * kPackedWordsPerBlock) : nullptr;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    half2 xh0;
    half2 xh1;
    half2 xh2;
    half2 xh3;
    if constexpr (kCgX) {
      xh0 = load_half2_cg(x2 + x_pair);
      xh1 = load_half2_cg(x2 + x_pair + 1);
      xh2 = load_half2_cg(x2 + x_pair + 2);
      xh3 = load_half2_cg(x2 + x_pair + 3);
    } else {
      xh0 = x2[x_pair];
      xh1 = x2[x_pair + 1];
      xh2 = x2[x_pair + 2];
      xh3 = x2[x_pair + 3];
    }
    const float2 x0 = __half22float2(xh0);
    const float2 x1 = __half22float2(xh1);
    const float2 x2v = __half22float2(xh2);
    const float2 x3 = __half22float2(xh3);

    float u4_x_sum = 0.0f;
    float gate0_u4_acc = 0.0f;
    dot8_u4xfloat2(load_u32_cg(gate0_w32 + lane16), x0, x1, x2v, x3, gate0_u4_acc, u4_x_sum);
    const float up0_u4_acc = dot8_u4xfloat2(load_u32_cg(up0_w32 + lane16), x0, x1, x2v, x3);
    const float gate0_block_acc = gate0_u4_acc - 8.0f * u4_x_sum;
    const float up0_block_acc = up0_u4_acc - 8.0f * u4_x_sum;

    float gate0_scale;
    float up0_scale;
    if constexpr (kBroadcastScale) {
      gate0_scale = 0.0f;
      up0_scale = 0.0f;
      if (lane16 == 0) {
        gate0_scale = __half2float(scales[block_k * kKblockScaleStride + row0]);
        up0_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row0]);
      }
      gate0_scale = __shfl_sync(0xffffffffu, gate0_scale, 0, 16);
      up0_scale = __shfl_sync(0xffffffffu, up0_scale, 0, 16);
    } else {
      gate0_scale = __half2float(scales[block_k * kKblockScaleStride + row0]);
      up0_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row0]);
    }
    gate0_acc = fmaf(gate0_block_acc, gate0_scale, gate0_acc);
    up0_acc = fmaf(up0_block_acc, up0_scale, up0_acc);

    if (valid_row1) {
      const float gate1_u4_acc = dot8_u4xfloat2(load_u32_cg(gate1_w32 + lane16), x0, x1, x2v, x3);
      const float up1_u4_acc = dot8_u4xfloat2(load_u32_cg(up1_w32 + lane16), x0, x1, x2v, x3);
      const float gate1_block_acc = gate1_u4_acc - 8.0f * u4_x_sum;
      const float up1_block_acc = up1_u4_acc - 8.0f * u4_x_sum;

      float gate1_scale;
      float up1_scale;
      if constexpr (kBroadcastScale) {
        gate1_scale = 0.0f;
        up1_scale = 0.0f;
        if (lane16 == 0) {
          gate1_scale = __half2float(scales[block_k * kKblockScaleStride + row1]);
          up1_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row1]);
        }
        gate1_scale = __shfl_sync(0xffffffffu, gate1_scale, 0, 16);
        up1_scale = __shfl_sync(0xffffffffu, up1_scale, 0, 16);
      } else {
        gate1_scale = __half2float(scales[block_k * kKblockScaleStride + row1]);
        up1_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row1]);
      }
      gate1_acc = fmaf(gate1_block_acc, gate1_scale, gate1_acc);
      up1_acc = fmaf(up1_block_acc, up1_scale, up1_acc);
    }
  }

  const float gate0 = half_warp_sum(gate0_acc);
  const float up0 = half_warp_sum(up0_acc);
  const float gate1 = half_warp_sum(gate1_acc);
  const float up1 = half_warp_sum(up1_acc);
  if (lane16 == 0) {
    const float swish0 = gate0 / (1.0f + __expf(-gate0));
    out[row0] = __float2half_rn(swish0 * up0);
    if (valid_row1) {
      const float swish1 = gate1 / (1.0f + __expf(-gate1));
      out[row1] = __float2half_rn(swish1 * up1);
    }
  }
}

template <int kRowsPerBlock, bool kBroadcastScale, bool kSharedX, bool kCgX>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  const half2* x2_global = reinterpret_cast<const half2*>(x);
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);
  if constexpr (kSharedX) {
    for (int idx = threadIdx.x; idx < kHiddenSize / 2; idx += blockDim.x) {
      x2_shared[idx] = x2_global[idx];
    }
    __syncthreads();
  }
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = kSharedX ? x2_shared : x2_global;
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);
  const int x_pair0 = lane16 * 4;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

  auto load_x4 = [&](int block_k, half2& xh0, half2& xh1, half2& xh2, half2& xh3) {
    const int x_pair = block_k * (kBlockK / 2) + x_pair0;
    if constexpr (kCgX) {
      xh0 = load_half2_cg(x2 + x_pair);
      xh1 = load_half2_cg(x2 + x_pair + 1);
      xh2 = load_half2_cg(x2 + x_pair + 2);
      xh3 = load_half2_cg(x2 + x_pair + 3);
    } else {
      xh0 = x2[x_pair];
      xh1 = x2[x_pair + 1];
      xh2 = x2[x_pair + 2];
      xh3 = x2[x_pair + 3];
    }
  };

  const uint32_t* block_w32_0 = weight_w32;
  const uint32_t* gate_w32_0 = block_w32_0 + row * kPackedWordsPerBlock;
  const uint32_t* up_w32_0 = block_w32_0 + (kIntermediateSize + row) * kPackedWordsPerBlock;
  uint32_t gate_word = load_u32_cg(gate_w32_0 + lane16);
  uint32_t up_word = load_u32_cg(up_w32_0 + lane16);
  half2 xh0;
  half2 xh1;
  half2 xh2;
  half2 xh3;
  load_x4(0, xh0, xh1, xh2, xh3);
  float gate_scale;
  float up_scale;
  if constexpr (kBroadcastScale) {
    gate_scale = 0.0f;
    up_scale = 0.0f;
    if (lane16 == 0) {
      gate_scale = __half2float(scales[row]);
      up_scale = __half2float(scales[kIntermediateSize + row]);
    }
    gate_scale = __shfl_sync(0xffffffffu, gate_scale, 0, 16);
    up_scale = __shfl_sync(0xffffffffu, up_scale, 0, 16);
  } else {
    gate_scale = __half2float(scales[row]);
    up_scale = __half2float(scales[kIntermediateSize + row]);
  }

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    uint32_t next_gate_word = 0u;
    uint32_t next_up_word = 0u;
    half2 next_xh0;
    half2 next_xh1;
    half2 next_xh2;
    half2 next_xh3;
    float next_gate_scale = 0.0f;
    float next_up_scale = 0.0f;
    if (block_k + 1 < kScaleBlocks) {
      const uint32_t* block_w32_next = weight_w32 + (block_k + 1) * kKblockWords;
      const uint32_t* gate_w32_next = block_w32_next + row * kPackedWordsPerBlock;
      const uint32_t* up_w32_next = block_w32_next + (kIntermediateSize + row) * kPackedWordsPerBlock;
      next_gate_word = load_u32_cg(gate_w32_next + lane16);
      next_up_word = load_u32_cg(up_w32_next + lane16);
      load_x4(block_k + 1, next_xh0, next_xh1, next_xh2, next_xh3);
      if constexpr (kBroadcastScale) {
        if (lane16 == 0) {
          next_gate_scale = __half2float(scales[(block_k + 1) * kKblockScaleStride + row]);
          next_up_scale =
              __half2float(scales[(block_k + 1) * kKblockScaleStride + kIntermediateSize + row]);
        }
        next_gate_scale = __shfl_sync(0xffffffffu, next_gate_scale, 0, 16);
        next_up_scale = __shfl_sync(0xffffffffu, next_up_scale, 0, 16);
      } else {
        next_gate_scale = __half2float(scales[(block_k + 1) * kKblockScaleStride + row]);
        next_up_scale =
            __half2float(scales[(block_k + 1) * kKblockScaleStride + kIntermediateSize + row]);
      }
    }

    const float2 x0 = __half22float2(xh0);
    const float2 x1 = __half22float2(xh1);
    const float2 x2v = __half22float2(xh2);
    const float2 x3 = __half22float2(xh3);
    float u4_x_sum = 0.0f;
    float gate_u4_acc = 0.0f;
    dot8_u4xfloat2(gate_word, x0, x1, x2v, x3, gate_u4_acc, u4_x_sum);
    const float up_u4_acc = dot8_u4xfloat2(up_word, x0, x1, x2v, x3);
    gate_acc = fmaf(gate_u4_acc - 8.0f * u4_x_sum, gate_scale, gate_acc);
    up_acc = fmaf(up_u4_acc - 8.0f * u4_x_sum, up_scale, up_acc);

    gate_word = next_gate_word;
    up_word = next_up_word;
    xh0 = next_xh0;
    xh1 = next_xh1;
    xh2 = next_xh2;
    xh3 = next_xh3;
    gate_scale = next_gate_scale;
    up_scale = next_up_scale;
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock, bool kBroadcastScale, bool kSharedX, bool kCgX>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;

  const half2* x2_global = reinterpret_cast<const half2*>(x);
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);
  if constexpr (kSharedX) {
    for (int idx = threadIdx.x; idx < kHiddenSize / 2; idx += blockDim.x) {
      x2_shared[idx] = x2_global[idx];
    }
    __syncthreads();
  }
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = kSharedX ? x2_shared : x2_global;
  const uint32_t* x_w32 = reinterpret_cast<const uint32_t*>(x2);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    uint32_t x0_bits = 0u;
    uint32_t x1_bits = 0u;
    uint32_t x2_bits = 0u;
    uint32_t x3_bits = 0u;
    if (half_id == 0) {
      if constexpr (kCgX) {
        x0_bits = load_u32_cg(x_w32 + x_pair);
        x1_bits = load_u32_cg(x_w32 + x_pair + 1);
        x2_bits = load_u32_cg(x_w32 + x_pair + 2);
        x3_bits = load_u32_cg(x_w32 + x_pair + 3);
      } else {
        x0_bits = x_w32[x_pair];
        x1_bits = x_w32[x_pair + 1];
        x2_bits = x_w32[x_pair + 2];
        x3_bits = x_w32[x_pair + 3];
      }
    }
    x0_bits = __shfl_sync(0xffffffffu, x0_bits, lane16, 32);
    x1_bits = __shfl_sync(0xffffffffu, x1_bits, lane16, 32);
    x2_bits = __shfl_sync(0xffffffffu, x2_bits, lane16, 32);
    x3_bits = __shfl_sync(0xffffffffu, x3_bits, lane16, 32);

    const float2 x0 = __half22float2(half2_from_u32(x0_bits));
    const float2 x1 = __half22float2(half2_from_u32(x1_bits));
    const float2 x2v = __half22float2(half2_from_u32(x2_bits));
    const float2 x3 = __half22float2(half2_from_u32(x3_bits));

    float u4_x_sum = 0.0f;
    float gate_u4_acc = 0.0f;
    dot8_u4xfloat2(load_u32_cg(gate_w32 + lane16), x0, x1, x2v, x3, gate_u4_acc, u4_x_sum);
    const float up_u4_acc = dot8_u4xfloat2(load_u32_cg(up_w32 + lane16), x0, x1, x2v, x3);

    const float gate_block_acc = gate_u4_acc - 8.0f * u4_x_sum;
    const float up_block_acc = up_u4_acc - 8.0f * u4_x_sum;

    float gate_scale;
    float up_scale;
    if constexpr (kBroadcastScale) {
      gate_scale = 0.0f;
      up_scale = 0.0f;
      if (lane16 == 0) {
        gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
        up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
      }
      gate_scale = __shfl_sync(0xffffffffu, gate_scale, 0, 16);
      up_scale = __shfl_sync(0xffffffffu, up_scale, 0, 16);
    } else {
      gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
      up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    }
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const half2 xh0 = x2[x_pair];
    const half2 xh1 = x2[x_pair + 1];
    const half2 xh2 = x2[x_pair + 2];
    const half2 xh3 = x2[x_pair + 3];
    const float2 x0 = __half22float2(xh0);
    const float2 x1 = __half22float2(xh1);
    const float2 x2v = __half22float2(xh2);
    const float2 x3 = __half22float2(xh3);

    const uint32_t gate_word = load_u32_cg(gate_w32 + lane16);
    const uint32_t up_word = load_u32_cg(up_w32 + lane16);

    float gate_lo = 0.0f;
    float gate_hi = 0.0f;
    float up_lo = 0.0f;
    float up_hi = 0.0f;

    gate_lo = dot_acc_u4_pair(gate_lo, gate_word & 0xffu, x0);
    up_lo = dot_acc_u4_pair(up_lo, up_word & 0xffu, x0);
    gate_lo = dot_acc_u4_pair(gate_lo, (gate_word >> 8) & 0xffu, x1);
    up_lo = dot_acc_u4_pair(up_lo, (up_word >> 8) & 0xffu, x1);
    gate_hi = dot_acc_u4_pair(gate_hi, (gate_word >> 16) & 0xffu, x2v);
    up_hi = dot_acc_u4_pair(up_hi, (up_word >> 16) & 0xffu, x2v);
    gate_hi = dot_acc_u4_pair(gate_hi, (gate_word >> 24) & 0xffu, x3);
    up_hi = dot_acc_u4_pair(up_hi, (up_word >> 24) & 0xffu, x3);

    const float u4_x_sum = sum4_float2(x0, x1, x2v, x3);
    const float gate_block_acc = (gate_lo + gate_hi) - 8.0f * u4_x_sum;
    const float up_block_acc = (up_lo + up_hi) - 8.0f * u4_x_sum;

    const float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    const float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const half2 xh0 = x2[x_pair];
    const half2 xh1 = x2[x_pair + 1];
    const half2 xh2 = x2[x_pair + 2];
    const half2 xh3 = x2[x_pair + 3];
    const float2 x0 = __half22float2(xh0);
    const float2 x1 = __half22float2(xh1);
    const float2 x2v = __half22float2(xh2);
    const float2 x3 = __half22float2(xh3);

    float u4_x_sum = 0.0f;
    float gate_u4_acc = 0.0f;
    dot8_u4xfloat2(load_u32_cg(gate_w32 + lane16), x0, x1, x2v, x3, gate_u4_acc, u4_x_sum);

    const uint32_t up_word = load_u32_cg(up_w32 + lane16);
    float up_lo = 0.0f;
    float up_hi = 0.0f;
    up_lo = dot_acc_u4_pair(up_lo, up_word & 0xffu, x0);
    up_lo = dot_acc_u4_pair(up_lo, (up_word >> 8) & 0xffu, x1);
    up_hi = dot_acc_u4_pair(up_hi, (up_word >> 16) & 0xffu, x2v);
    up_hi = dot_acc_u4_pair(up_hi, (up_word >> 24) & 0xffu, x3);

    const float gate_block_acc = gate_u4_acc - 8.0f * u4_x_sum;
    const float up_block_acc = (up_lo + up_hi) - 8.0f * u4_x_sum;

    const float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    const float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const half2 xh0 = x2[x_pair];
    const half2 xh1 = x2[x_pair + 1];
    const half2 xh2 = x2[x_pair + 2];
    const half2 xh3 = x2[x_pair + 3];
    const float2 x0 = __half22float2(xh0);
    const float2 x1 = __half22float2(xh1);
    const float2 x2v = __half22float2(xh2);
    const float2 x3 = __half22float2(xh3);

    const uint32_t gate_word = load_u32_cg(gate_w32 + lane16);
    float gate_lo = 0.0f;
    float gate_hi = 0.0f;
    gate_lo = dot_acc_u4_pair(gate_lo, gate_word & 0xffu, x0);
    gate_lo = dot_acc_u4_pair(gate_lo, (gate_word >> 8) & 0xffu, x1);
    gate_hi = dot_acc_u4_pair(gate_hi, (gate_word >> 16) & 0xffu, x2v);
    gate_hi = dot_acc_u4_pair(gate_hi, (gate_word >> 24) & 0xffu, x3);

    const float u4_x_sum = sum4_float2(x0, x1, x2v, x3);
    const float up_u4_acc = dot8_u4xfloat2(load_u32_cg(up_w32 + lane16), x0, x1, x2v, x3);
    const float gate_block_acc = (gate_lo + gate_hi) - 8.0f * u4_x_sum;
    const float up_block_acc = up_u4_acc - 8.0f * u4_x_sum;

    const float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    const float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float gate_acc_alt = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const half2 xh0 = x2[x_pair];
    const half2 xh1 = x2[x_pair + 1];
    const half2 xh2 = x2[x_pair + 2];
    const half2 xh3 = x2[x_pair + 3];
    const float2 x0 = __half22float2(xh0);
    const float2 x1 = __half22float2(xh1);
    const float2 x2v = __half22float2(xh2);
    const float2 x3 = __half22float2(xh3);

    const uint32_t gate_word = load_u32_cg(gate_w32 + lane16);
    float gate_lo = 0.0f;
    float gate_hi = 0.0f;
    gate_lo = dot_acc_u4_pair(gate_lo, gate_word & 0xffu, x0);
    gate_lo = dot_acc_u4_pair(gate_lo, (gate_word >> 8) & 0xffu, x1);
    gate_hi = dot_acc_u4_pair(gate_hi, (gate_word >> 16) & 0xffu, x2v);
    gate_hi = dot_acc_u4_pair(gate_hi, (gate_word >> 24) & 0xffu, x3);

    const float u4_x_sum = sum4_float2(x0, x1, x2v, x3);
    const float up_u4_acc = dot8_u4xfloat2(load_u32_cg(up_w32 + lane16), x0, x1, x2v, x3);
    const float gate_block_acc = (gate_lo + gate_hi) - 8.0f * u4_x_sum;
    const float up_block_acc = up_u4_acc - 8.0f * u4_x_sum;

    const float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    const float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    if ((block_k & 1) == 0) {
      gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    } else {
      gate_acc_alt = fmaf(gate_block_acc, gate_scale, gate_acc_alt);
    }
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }
  gate_acc += gate_acc_alt;

  const float gate = half_warp_sum(gate_acc);
  const float up = half_warp_sum(up_acc);
  if (lane16 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int row_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint16_t* weight_u16 = reinterpret_cast<const uint16_t*>(weight_int4);
  constexpr int kPackedU16PerBlock = kPackedBytesPerBlock / 2;
  constexpr int kKblockU16 = 2 * kIntermediateSize * kPackedU16PerBlock;

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint16_t* block_u16 = weight_u16 + block_k * kKblockU16;
    const uint16_t* gate_u16 = block_u16 + row * kPackedU16PerBlock;
    const uint16_t* up_u16 = block_u16 + (kIntermediateSize + row) * kPackedU16PerBlock;

    const int x_pair = block_k * (kBlockK / 2) + lane * 2;
    const float2 x0 = __half22float2(x2[x_pair]);
    const float2 x1 = __half22float2(x2[x_pair + 1]);

    float u4_x_sum = 0.0f;
    const float gate_u4_acc = dot4_u4xfloat2(static_cast<uint32_t>(gate_u16[lane]), x0, x1, u4_x_sum);
    float up_x_sum = 0.0f;
    const float up_u4_acc = dot4_u4xfloat2(static_cast<uint32_t>(up_u16[lane]), x0, x1, up_x_sum);

    const float gate_block_acc = gate_u4_acc - 8.0f * u4_x_sum;
    const float up_block_acc = up_u4_acc - 8.0f * u4_x_sum;

    const float gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
    const float up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = warp_sum(gate_acc);
  const float up = warp_sum(up_acc);
  if (lane == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

template <int kRowsPerBlock, bool kReuseX, bool kBroadcastScale>
__global__ void gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    half* __restrict__ out) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group_id = lane >> 3;
  const int lane8 = lane & 7;
  const int row_in_block = warp_id * 4 + group_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kIntermediateSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float gate_acc = 0.0f;
  float up_acc = 0.0f;

#pragma unroll
  for (int block_k = 0; block_k < kScaleBlocks; ++block_k) {
    const uint32_t* block_w32 = weight_w32 + block_k * kKblockWords;
    const uint32_t* gate_w32 = block_w32 + row * kPackedWordsPerBlock;
    const uint32_t* up_w32 = block_w32 + (kIntermediateSize + row) * kPackedWordsPerBlock;

    const int word_offset = lane8 * 2;
    const uint32_t gate0 = load_u32_cg(gate_w32 + word_offset);
    const uint32_t gate1 = load_u32_cg(gate_w32 + word_offset + 1);
    const uint32_t up0 = load_u32_cg(up_w32 + word_offset);
    const uint32_t up1 = load_u32_cg(up_w32 + word_offset + 1);

    const int x_pair = block_k * (kBlockK / 2) + lane8 * 8;
    float gate_block_acc;
    float up_block_acc;
    if constexpr (kReuseX) {
      const float2 x0 = __half22float2(x2[x_pair]);
      const float2 x1 = __half22float2(x2[x_pair + 1]);
      const float2 x2v = __half22float2(x2[x_pair + 2]);
      const float2 x3 = __half22float2(x2[x_pair + 3]);
      const float2 x4 = __half22float2(x2[x_pair + 4]);
      const float2 x5 = __half22float2(x2[x_pair + 5]);
      const float2 x6 = __half22float2(x2[x_pair + 6]);
      const float2 x7 = __half22float2(x2[x_pair + 7]);
      gate_block_acc = dot16_int4xfloat2(gate0, gate1, x0, x1, x2v, x3, x4, x5, x6, x7);
      up_block_acc = dot16_int4xfloat2(up0, up1, x0, x1, x2v, x3, x4, x5, x6, x7);
    } else {
      gate_block_acc = dot16_int4xhalf(gate0, gate1, x2 + x_pair);
      up_block_acc = dot16_int4xhalf(up0, up1, x2 + x_pair);
    }

    float gate_scale;
    float up_scale;
    if constexpr (kBroadcastScale) {
      gate_scale = 0.0f;
      up_scale = 0.0f;
      if (lane8 == 0) {
        gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
        up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
      }
      gate_scale = __shfl_sync(0xffffffffu, gate_scale, 0, 8);
      up_scale = __shfl_sync(0xffffffffu, up_scale, 0, 8);
    } else {
      gate_scale = __half2float(scales[block_k * kKblockScaleStride + row]);
      up_scale = __half2float(scales[block_k * kKblockScaleStride + kIntermediateSize + row]);
    }

    gate_acc = fmaf(gate_block_acc, gate_scale, gate_acc);
    up_acc = fmaf(up_block_acc, up_scale, up_acc);
  }

  const float gate = quarter_warp_sum(gate_acc);
  const float up = quarter_warp_sum(up_acc);
  if (lane8 == 0) {
    const float swish = gate / (1.0f + __expf(-gate));
    out[row] = __float2half_rn(swish * up);
  }
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_int4);
  CHECK_CUDA(scales);
  CHECK_CUDA(out);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_int4);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(out);
  CHECK_HALF(x);
  CHECK_BYTE(weight_int4);
  CHECK_HALF(scales);
  CHECK_HALF(out);
  TORCH_CHECK(x.numel() == kHiddenSize, "x.numel() must be 2048");
  TORCH_CHECK(weight_int4.numel() == kScaleBlocks * 2 * kIntermediateSize * kPackedBytesPerBlock,
              "weight_int4 must contain [16, 2, 6144, 64]");
  TORCH_CHECK(scales.numel() == kScaleBlocks * 2 * kIntermediateSize,
              "scales must contain [16, 2, 6144]");
  TORCH_CHECK(out.numel() == kIntermediateSize, "out must contain 6144 elements");
  TORCH_CHECK(
      rows_per_block == 2 || rows_per_block == 4 || rows_per_block == 8 || rows_per_block == 16,
      "half-warp rows_per_block must be one of 2, 4, 8, or 16");
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_baseline_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = kRowsPerBlock * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kRowsPerBlock, bool kBroadcastScale, bool kSharedX, bool kCgX>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  const int shared_bytes = kSharedX ? static_cast<int>((kHiddenSize / 2) * sizeof(half2)) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048_kernel<
      kRowsPerBlock,
      kBroadcastScale,
      kSharedX,
      kCgX><<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kBroadcastScale, bool kSharedX, bool kCgX>
void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<
          2,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<
          4,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<
          8,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<
          16,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <
    int kRowsPerBlock,
    bool kBroadcastScale,
    bool kSharedX,
    bool kCgX,
    bool kBroadcastXSum,
    bool kSplitAccum>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  const int shared_bytes = kSharedX ? static_cast<int>((kHiddenSize / 2) * sizeof(half2)) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048_kernel<
      kRowsPerBlock,
      kBroadcastScale,
      kSharedX,
      kCgX,
      kBroadcastXSum,
      kSplitAccum><<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <
    bool kBroadcastScale,
    bool kSharedX,
    bool kCgX,
    bool kBroadcastXSum,
    bool kSplitAccum>
void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<
          2,
          kBroadcastScale,
          kSharedX,
          kCgX,
          kBroadcastXSum,
          kSplitAccum>(x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<
          4,
          kBroadcastScale,
          kSharedX,
          kCgX,
          kBroadcastXSum,
          kSplitAccum>(x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<
          8,
          kBroadcastScale,
          kSharedX,
          kCgX,
          kBroadcastXSum,
          kSplitAccum>(x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<
          16,
          kBroadcastScale,
          kSharedX,
          kCgX,
          kBroadcastXSum,
          kSplitAccum>(x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <int kRowsPerBlock, bool kBroadcastScale, bool kSharedX, bool kCgX>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 4 == 0, "pair2 half-warp kernel needs rows_per_block divisible by 4");
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 4) * 32;
  const int shared_bytes = kSharedX ? static_cast<int>((kHiddenSize / 2) * sizeof(half2)) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048_kernel<
      kRowsPerBlock,
      kBroadcastScale,
      kSharedX,
      kCgX><<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kBroadcastScale, bool kSharedX, bool kCgX>
void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<
          4,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<
          8,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<
          16,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "pair2 half-warp rows_per_block must be one of 4, 8, or 16");
  }
}

template <int kRowsPerBlock, bool kBroadcastScale, bool kSharedX, bool kCgX>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "prefetch half-warp kernel needs an even rows_per_block");
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  const int shared_bytes = kSharedX ? static_cast<int>((kHiddenSize / 2) * sizeof(half2)) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048_kernel<
      kRowsPerBlock,
      kBroadcastScale,
      kSharedX,
      kCgX><<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kBroadcastScale, bool kSharedX, bool kCgX>
void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<
          2,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<
          4,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<
          8,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<
          16,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "prefetch half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <int kRowsPerBlock, bool kBroadcastScale, bool kSharedX, bool kCgX>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 2 == 0, "xwarp half-warp kernel needs an even rows_per_block");
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  const int shared_bytes = kSharedX ? static_cast<int>((kHiddenSize / 2) * sizeof(half2)) : 0;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048_kernel<
      kRowsPerBlock,
      kBroadcastScale,
      kSharedX,
      kCgX><<<blocks, threads, shared_bytes, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kBroadcastScale, bool kSharedX, bool kCgX>
void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<
          2,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<
          4,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<
          8,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<
          16,
          kBroadcastScale,
          kSharedX,
          kCgX>(x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "xwarp half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048<2>(
          x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048<4>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048<8>(
          x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048<16>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "pairsplit half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048<2>(
          x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048<4>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048<8>(
          x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048<16>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "upsplit half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048<2>(
          x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048<4>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048<8>(
          x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048<16>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "gatesplit half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048<2>(
          x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048<4>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048<8>(
          x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048<16>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "gatesplit+splitaccum half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

template <int kRowsPerBlock>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = kRowsPerBlock * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048_kernel<kRowsPerBlock>
      <<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
          reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
          reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
          reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048<2>(
          x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048<4>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048<8>(
          x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048<16>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "fullwarp-u4offset rows_per_block must be one of 2, 4, 8, or 16");
  }
}

void check_quarterwarp_rows_per_block(int64_t rows_per_block) {
  TORCH_CHECK(
      rows_per_block == 4 || rows_per_block == 8 || rows_per_block == 16 || rows_per_block == 32,
      "quarter-warp rows_per_block must be one of 4, 8, 16, or 32");
}

template <int kRowsPerBlock, bool kReuseX, bool kBroadcastScale>
void launch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out) {
  static_assert(kRowsPerBlock % 4 == 0, "quarter-warp kernel needs rows_per_block divisible by 4");
  const int blocks = (kIntermediateSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 4) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048_kernel<
      kRowsPerBlock,
      kReuseX,
      kBroadcastScale><<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool kReuseX, bool kBroadcastScale>
void dispatch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& out,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048<4, kReuseX, kBroadcastScale>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048<8, kReuseX, kBroadcastScale>(
          x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048<16, kReuseX, kBroadcastScale>(
          x, weight_int4, scales, out);
      break;
    case 32:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048<32, kReuseX, kBroadcastScale>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "quarter-warp rows_per_block must be one of 4, 8, 16, or 32");
  }
}

}  // namespace

void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_int4, scales, out, rows_per_block);
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_baseline_m6144_k2048<2>(
          x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_baseline_m6144_k2048<4>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_baseline_m6144_k2048<8>(
          x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_baseline_m6144_k2048<16>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

void gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_int4, scales, out, 2);
  switch (rows_per_block) {
    case 1:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048<1>(
          x, weight_int4, scales, out);
      break;
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048<2>(
          x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048<4>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048<8>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "full-warp rows_per_block must be one of 1, 2, 4, or 8");
  }
}

void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block) {
  check_inputs(x, weight_int4, scales, out, rows_per_block);
  switch (rows_per_block) {
    case 2:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048<2>(
          x, weight_int4, scales, out);
      break;
    case 4:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048<4>(
          x, weight_int4, scales, out);
      break;
    case 8:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048<8>(
          x, weight_int4, scales, out);
      break;
    case 16:
      launch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048<16>(
          x, weight_int4, scales, out);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, or 16");
  }
}

void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block,
    bool broadcast_scale,
    bool shared_x,
    bool cg_x) {
  check_inputs(x, weight_int4, scales, out, rows_per_block);
  if (broadcast_scale && shared_x) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<true, true, false>(
        x, weight_int4, scales, out, rows_per_block);
  } else if (broadcast_scale && !shared_x && cg_x) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<true, false, true>(
        x, weight_int4, scales, out, rows_per_block);
  } else if (broadcast_scale && !shared_x && !cg_x) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<true, false, false>(
        x, weight_int4, scales, out, rows_per_block);
  } else if (!broadcast_scale && shared_x) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<false, true, false>(
        x, weight_int4, scales, out, rows_per_block);
  } else if (!broadcast_scale && !shared_x && cg_x) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<false, false, true>(
        x, weight_int4, scales, out, rows_per_block);
  } else {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048<false, false, false>(
        x, weight_int4, scales, out, rows_per_block);
  }
}

void gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block,
    bool broadcast_scale,
    bool shared_x,
    bool cg_x,
    int64_t candidate_idx) {
  check_inputs(x, weight_int4, scales, out, rows_per_block);
  const int candidate = static_cast<int>(candidate_idx);
  TORCH_CHECK(candidate >= 0 && candidate <= 13, "gate/up u4offset candidate_idx must be in [0, 13]");
  if (candidate == 4) {
    TORCH_CHECK(rows_per_block == 4 || rows_per_block == 8 || rows_per_block == 16,
        "pair2 candidate requires rows_per_block in {4, 8, 16}");
    if (broadcast_scale && shared_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<true, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_scale && !shared_x && cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<true, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_scale && !shared_x && !cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<true, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (!broadcast_scale && shared_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<false, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (!broadcast_scale && !shared_x && cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<false, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pair2_m6144_k2048<false, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
    return;
  }
  if (candidate == 5) {
    if (broadcast_scale && shared_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<true, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_scale && !shared_x && cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<true, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_scale && !shared_x && !cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<true, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (!broadcast_scale && shared_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<false, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (!broadcast_scale && !shared_x && cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<false, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_prefetch_m6144_k2048<false, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
    return;
  }
  if (candidate == 6) {
    if (broadcast_scale && shared_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<true, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_scale && !shared_x && cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<true, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_scale && !shared_x && !cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<true, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (!broadcast_scale && shared_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<false, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (!broadcast_scale && !shared_x && cg_x) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<false, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_xwarp_m6144_k2048<false, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
    return;
  }
  if (candidate == 7) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_pairsplit_m6144_k2048(
        x, weight_int4, scales, out, rows_per_block);
    return;
  }
  if (candidate == 8) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, true, false, false, false>(
        x, weight_int4, scales, out, rows_per_block);
    return;
  }
  if (candidate == 9) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, true, false, false>(
        x, weight_int4, scales, out, rows_per_block);
    return;
  }
  if (candidate == 10) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_upsplit_m6144_k2048(
        x, weight_int4, scales, out, rows_per_block);
    return;
  }
  if (candidate == 11) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u4offset_m6144_k2048(
        x, weight_int4, scales, out, rows_per_block);
    return;
  }
  if (candidate == 12) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_m6144_k2048(
        x, weight_int4, scales, out, rows_per_block);
    return;
  }
  if (candidate == 13) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_gatesplit_splitaccum_m6144_k2048(
        x, weight_int4, scales, out, rows_per_block);
    return;
  }
  const bool broadcast_x_sum = candidate == 1 || candidate == 3;
  const bool split_accum = candidate == 2 || candidate == 3;
  if (broadcast_scale && shared_x) {
    if (broadcast_x_sum && split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, true, false, true, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_x_sum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, true, false, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, true, false, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, true, false, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
  } else if (broadcast_scale && !shared_x && cg_x) {
    if (broadcast_x_sum && split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, false, true, true, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_x_sum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, false, true, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, false, true, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, false, true, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
  } else if (broadcast_scale && !shared_x && !cg_x) {
    if (broadcast_x_sum && split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, false, false, true, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_x_sum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, false, false, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, false, false, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<true, false, false, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
  } else if (!broadcast_scale && shared_x) {
    if (broadcast_x_sum && split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, true, false, true, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_x_sum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, true, false, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, true, false, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, true, false, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
  } else if (!broadcast_scale && !shared_x && cg_x) {
    if (broadcast_x_sum && split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, true, true, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_x_sum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, true, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, true, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, true, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
  } else {
    if (broadcast_x_sum && split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, false, true, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (broadcast_x_sum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, false, true, false>(
          x, weight_int4, scales, out, rows_per_block);
    } else if (split_accum) {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, false, false, true>(
          x, weight_int4, scales, out, rows_per_block);
    } else {
      dispatch_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048<false, false, false, false, false>(
          x, weight_int4, scales, out, rows_per_block);
    }
  }
}

void gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor out,
    int64_t rows_per_block,
    bool reuse_x,
    bool broadcast_scale) {
  check_inputs(x, weight_int4, scales, out, 4);
  check_quarterwarp_rows_per_block(rows_per_block);
  if (reuse_x && broadcast_scale) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048<true, true>(
        x, weight_int4, scales, out, rows_per_block);
  } else if (reuse_x && !broadcast_scale) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048<true, false>(
        x, weight_int4, scales, out, rows_per_block);
  } else if (!reuse_x && broadcast_scale) {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048<false, true>(
        x, weight_int4, scales, out, rows_per_block);
  } else {
    dispatch_gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048<false, false>(
        x, weight_int4, scales, out, rows_per_block);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_m6144_k2048",
      &gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_m6144_k2048,
      "Symmetric int4 kblock/kscale half-warp gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048",
      &gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_lowreg_m6144_k2048,
      "Symmetric int4 kblock/kscale low-register half-warp gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048",
      &gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_m6144_k2048,
      "Symmetric int4 kblock/kscale half-warp x-reuse gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048",
      &gate_up_swiglu_int4_sym_kblock_kscale_halfwarp_xreuse_u4offset_m6144_k2048,
      "Symmetric int4 kblock/kscale half-warp x-reuse u4-offset gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048",
      &gate_up_swiglu_int4_sym_kblock_kscale_fullwarp_u16_m6144_k2048,
      "Symmetric int4 kblock/kscale full-warp u16-load gate/up SwiGLU for Qwen3-VL decode");
  m.def(
      "gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048",
      &gate_up_swiglu_int4_sym_kblock_kscale_quarterwarp_m6144_k2048,
      "Symmetric int4 kblock/kscale quarter-warp gate/up SwiGLU for Qwen3-VL decode");
}
