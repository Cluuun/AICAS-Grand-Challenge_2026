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
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be fp32")

constexpr int kHiddenSize = 2048;
constexpr int kBlockK = 128;
constexpr int kPackedBytesPerBlock = kBlockK / 2;
constexpr int kPackedWordsPerBlock = kPackedBytesPerBlock / 4;

__forceinline__ __device__ uint32_t load_u32_cg(const uint32_t* ptr) {
  return __ldcg(ptr);
}

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ half2 load_half2_cg(const half2* ptr) {
  return half2_from_u32(__ldcg(reinterpret_cast<const uint32_t*>(ptr)));
}

__forceinline__ __device__ float dot_acc_int4_pair(float acc, uint32_t byte_value, half2 x_pair) {
  const float2 xv = __half22float2(x_pair);
  const float q0 = static_cast<float>(static_cast<int>(byte_value & 0x0fu) - 8);
  const float q1 = static_cast<float>(static_cast<int>((byte_value >> 4) & 0x0fu) - 8);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  return acc;
}

__forceinline__ __device__ void dot_acc_u4_pair(
    uint32_t byte_value,
    half2 x_pair,
    float& acc,
    float& x_sum) {
  const float2 xv = __half22float2(x_pair);
  const float q0 = static_cast<float>(byte_value & 0x0fu);
  const float q1 = static_cast<float>((byte_value >> 4) & 0x0fu);
  acc = fmaf(q0, xv.x, acc);
  acc = fmaf(q1, xv.y, acc);
  x_sum += xv.x + xv.y;
}

__forceinline__ __device__ float dot8_int4xhalf(uint32_t packed, const half2* __restrict__ x2) {
  float acc = 0.0f;
  acc = dot_acc_int4_pair(acc, packed & 0xffu, x2[0]);
  acc = dot_acc_int4_pair(acc, (packed >> 8) & 0xffu, x2[1]);
  acc = dot_acc_int4_pair(acc, (packed >> 16) & 0xffu, x2[2]);
  acc = dot_acc_int4_pair(acc, (packed >> 24) & 0xffu, x2[3]);
  return acc;
}

__forceinline__ __device__ void dot8_u4xhalf(
    uint32_t packed,
    const half2* __restrict__ x2,
    float& acc,
    float& x_sum) {
  dot_acc_u4_pair(packed & 0xffu, x2[0], acc, x_sum);
  dot_acc_u4_pair((packed >> 8) & 0xffu, x2[1], acc, x_sum);
  dot_acc_u4_pair((packed >> 16) & 0xffu, x2[2], acc, x_sum);
  dot_acc_u4_pair((packed >> 24) & 0xffu, x2[3], acc, x_sum);
}

__forceinline__ __device__ float half_warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, 16);
  }
  return value;
}

__forceinline__ __device__ float halfwarp_broadcast_float(float value, int half_id) {
  return __shfl_sync(0xffffffffu, value, half_id << 4, 32);
}

template <int kKIn, int kSplitK, int kRowsPerBlock>
__global__ void gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    float* __restrict__ partial) {
  constexpr int kScaleBlocks = kKIn / kBlockK;
  constexpr int kChunkBlocks = kScaleBlocks / kSplitK;
  constexpr int kKblockWords = kHiddenSize * kPackedWordsPerBlock;

  const int split = blockIdx.y;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kHiddenSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(load_u32_cg(row_w32 + lane16), x2 + x_pair);
    const float scale = __half2float(scales[block_k * kHiddenSize + row]);
    acc = fmaf(block_acc, scale, acc);
  }

  const float row_acc = half_warp_sum(acc);
  if (lane16 == 0) {
    partial[static_cast<int64_t>(split) * kHiddenSize + row] = row_acc;
  }
}

template <int kKIn, int kSplitK, int kRowsPerBlock>
__global__ void gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp_xreuse_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    float* __restrict__ partial) {
  constexpr int kScaleBlocks = kKIn / kBlockK;
  constexpr int kChunkBlocks = kScaleBlocks / kSplitK;
  constexpr int kKblockWords = kHiddenSize * kPackedWordsPerBlock;

  const int split = blockIdx.y;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kHiddenSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;

    uint32_t x_bits0 = 0;
    uint32_t x_bits1 = 0;
    uint32_t x_bits2 = 0;
    uint32_t x_bits3 = 0;
    if (half_id == 0) {
      x_bits0 = *reinterpret_cast<const uint32_t*>(&load_half2_cg(x2 + x_pair + 0));
      x_bits1 = *reinterpret_cast<const uint32_t*>(&load_half2_cg(x2 + x_pair + 1));
      x_bits2 = *reinterpret_cast<const uint32_t*>(&load_half2_cg(x2 + x_pair + 2));
      x_bits3 = *reinterpret_cast<const uint32_t*>(&load_half2_cg(x2 + x_pair + 3));
    }
    x_bits0 = __shfl_sync(0xffffffffu, x_bits0, lane16, 32);
    x_bits1 = __shfl_sync(0xffffffffu, x_bits1, lane16, 32);
    x_bits2 = __shfl_sync(0xffffffffu, x_bits2, lane16, 32);
    x_bits3 = __shfl_sync(0xffffffffu, x_bits3, lane16, 32);
    half2 x_local[4] = {
        half2_from_u32(x_bits0),
        half2_from_u32(x_bits1),
        half2_from_u32(x_bits2),
        half2_from_u32(x_bits3),
    };

    const float block_acc = dot8_int4xhalf(load_u32_cg(row_w32 + lane16), x_local);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    acc = fmaf(block_acc, scale, acc);
  }

  const float row_acc = half_warp_sum(acc);
  if (lane16 == 0) {
    partial[static_cast<int64_t>(split) * kHiddenSize + row] = row_acc;
  }
}

template <int kKIn, int kSplitK, int kRowsPerBlock>
__global__ void gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp_bscale_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    float* __restrict__ partial) {
  constexpr int kScaleBlocks = kKIn / kBlockK;
  constexpr int kChunkBlocks = kScaleBlocks / kSplitK;
  constexpr int kKblockWords = kHiddenSize * kPackedWordsPerBlock;

  const int split = blockIdx.y;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kHiddenSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(load_u32_cg(row_w32 + lane16), x2 + x_pair);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    acc = fmaf(block_acc, scale, acc);
  }

  const float row_acc = half_warp_sum(acc);
  if (lane16 == 0) {
    partial[static_cast<int64_t>(split) * kHiddenSize + row] = row_acc;
  }
}

template <int kKIn, int kSplitK, int kRowsPerBlock>
__global__ void gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp_sharedx_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    float* __restrict__ partial) {
  constexpr int kScaleBlocks = kKIn / kBlockK;
  constexpr int kChunkBlocks = kScaleBlocks / kSplitK;
  constexpr int kChunkHalf2 = kChunkBlocks * (kBlockK / 2);
  constexpr int kKblockWords = kHiddenSize * kPackedWordsPerBlock;

  const int split = blockIdx.y;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kHiddenSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);

  const int chunk_base = split * kChunkHalf2;
  for (int idx = threadIdx.x; idx < kChunkHalf2; idx += blockDim.x) {
    x2_shared[idx] = load_half2_cg(x2 + chunk_base + idx);
  }
  __syncthreads();

  float acc = 0.0f;

#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_offset * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(load_u32_cg(row_w32 + lane16), x2_shared + x_pair);
    const float scale = __half2float(scales[block_k * kHiddenSize + row]);
    acc = fmaf(block_acc, scale, acc);
  }

  const float row_acc = half_warp_sum(acc);
  if (lane16 == 0) {
    partial[static_cast<int64_t>(split) * kHiddenSize + row] = row_acc;
  }
}

template <int kKIn, int kSplitK, int kRowsPerBlock>
__global__ void gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp_splitaccum_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    float* __restrict__ partial) {
  constexpr int kScaleBlocks = kKIn / kBlockK;
  constexpr int kChunkBlocks = kScaleBlocks / kSplitK;
  constexpr int kKblockWords = kHiddenSize * kPackedWordsPerBlock;

  const int split = blockIdx.y;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kHiddenSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc_even = 0.0f;
  float acc_odd = 0.0f;

#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(load_u32_cg(row_w32 + lane16), x2 + x_pair);
    const float scale = __half2float(scales[block_k * kHiddenSize + row]);
    if ((block_offset & 1) == 0) {
      acc_even = fmaf(block_acc, scale, acc_even);
    } else {
      acc_odd = fmaf(block_acc, scale, acc_odd);
    }
  }

  const float row_acc = half_warp_sum(acc_even + acc_odd);
  if (lane16 == 0) {
    partial[static_cast<int64_t>(split) * kHiddenSize + row] = row_acc;
  }
}

template <int kKIn, int kSplitK, int kRowsPerBlock>
__global__ void gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp_u4offset_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    float* __restrict__ partial) {
  constexpr int kScaleBlocks = kKIn / kBlockK;
  constexpr int kChunkBlocks = kScaleBlocks / kSplitK;
  constexpr int kKblockWords = kHiddenSize * kPackedWordsPerBlock;

  const int split = blockIdx.y;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int row_in_block = warp_id * 2 + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  if (row >= kHiddenSize) {
    return;
  }

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;

#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    float u4_acc = 0.0f;
    float x_sum = 0.0f;
    dot8_u4xhalf(load_u32_cg(row_w32 + lane16), x2 + x_pair, u4_acc, x_sum);
    const float block_acc = u4_acc - 8.0f * x_sum;
    const float scale = __half2float(scales[block_k * kHiddenSize + row]);
    acc = fmaf(block_acc, scale, acc);
  }

  const float row_acc = half_warp_sum(acc);
  if (lane16 == 0) {
    partial[static_cast<int64_t>(split) * kHiddenSize + row] = row_acc;
  }
}

void check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& partial,
    int64_t split_k,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_int4);
  CHECK_CUDA(scales);
  CHECK_CUDA(partial);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_int4);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(partial);
  CHECK_HALF(x);
  CHECK_BYTE(weight_int4);
  CHECK_HALF(scales);
  CHECK_FLOAT(partial);
  TORCH_CHECK(x.numel() == 2048 || x.numel() == 6144, "x.numel() must be 2048 or 6144");
  TORCH_CHECK(
      rows_per_block == 2 || rows_per_block == 4 || rows_per_block == 8 ||
          rows_per_block == 16 || rows_per_block == 32,
      "half-warp rows_per_block must be one of 2, 4, 8, 16, or 32");

  const int64_t scale_blocks = x.numel() / kBlockK;
  TORCH_CHECK(x.numel() % kBlockK == 0, "K must be divisible by 128");
  TORCH_CHECK(scale_blocks % split_k == 0, "K/128 must be divisible by split_k");
  TORCH_CHECK(weight_int4.numel() == scale_blocks * kHiddenSize * kPackedBytesPerBlock,
              "weight_int4 must contain [K/128, 2048, 64]");
  TORCH_CHECK(scales.numel() == scale_blocks * kHiddenSize, "scales must contain [K/128, 2048]");
  TORCH_CHECK(partial.numel() == split_k * kHiddenSize, "partial must be [split_k, 2048]");
}

template <int kKIn, int kSplitK, int kRowsPerBlock>
void launch_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& partial) {
  static_assert(kKIn % kBlockK == 0, "K must be divisible by 128");
  static_assert((kKIn / kBlockK) % kSplitK == 0, "K/128 must be divisible by split_k");
  static_assert(kRowsPerBlock % 2 == 0, "half-warp kernel needs an even rows_per_block");
  const int blocks = (kHiddenSize + kRowsPerBlock - 1) / kRowsPerBlock;
  const int threads = (kRowsPerBlock / 2) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  if constexpr (kKIn == 2048) {
    gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp_splitaccum_kernel<kKIn, kSplitK, kRowsPerBlock>
        <<<dim3(blocks, kSplitK), threads, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<float*>(partial.data_ptr<float>()));
  } else {
    gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp_kernel<kKIn, kSplitK, kRowsPerBlock>
        <<<dim3(blocks, kSplitK), threads, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
            reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<float*>(partial.data_ptr<float>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int kKIn, int kSplitK>
void dispatch_rows(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& partial,
    int64_t rows_per_block) {
  switch (rows_per_block) {
    case 2:
      launch_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp<kKIn, kSplitK, 2>(
          x, weight_int4, scales, partial);
      break;
    case 4:
      launch_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp<kKIn, kSplitK, 4>(
          x, weight_int4, scales, partial);
      break;
    case 8:
      launch_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp<kKIn, kSplitK, 8>(
          x, weight_int4, scales, partial);
      break;
    case 16:
      launch_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp<kKIn, kSplitK, 16>(
          x, weight_int4, scales, partial);
      break;
    case 32:
      launch_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp<kKIn, kSplitK, 32>(
          x, weight_int4, scales, partial);
      break;
    default:
      TORCH_CHECK(false, "half-warp rows_per_block must be one of 2, 4, 8, 16, or 32");
  }
}

void dispatch_split_k2048(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& partial,
    int64_t split_k,
    int64_t rows_per_block) {
    switch (split_k) {
    case 2:
      dispatch_rows<2048, 2>(x, weight_int4, scales, partial, rows_per_block);
      break;
    case 4:
      dispatch_rows<2048, 4>(x, weight_int4, scales, partial, rows_per_block);
      break;
    case 8:
      dispatch_rows<2048, 8>(x, weight_int4, scales, partial, rows_per_block);
      break;
    default:
      TORCH_CHECK(false, "K=2048 supports split_k 2, 4, or 8");
  }
}

void dispatch_split_k6144(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& partial,
    int64_t split_k,
    int64_t rows_per_block) {
  switch (split_k) {
    case 2:
      dispatch_rows<6144, 2>(x, weight_int4, scales, partial, rows_per_block);
      break;
    case 3:
      dispatch_rows<6144, 3>(x, weight_int4, scales, partial, rows_per_block);
      break;
    case 4:
      dispatch_rows<6144, 4>(x, weight_int4, scales, partial, rows_per_block);
      break;
    case 6:
      dispatch_rows<6144, 6>(x, weight_int4, scales, partial, rows_per_block);
      break;
    case 8:
      dispatch_rows<6144, 8>(x, weight_int4, scales, partial, rows_per_block);
      break;
    default:
      TORCH_CHECK(false, "K=6144 supports split_k 2, 3, 4, 6, or 8");
  }
}

}  // namespace

void gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor partial,
    int64_t split_k,
    int64_t rows_per_block) {
  check_inputs(x, weight_int4, scales, partial, split_k, rows_per_block);
  const int64_t k_in = x.numel();
  if (k_in == 2048) {
    TORCH_CHECK(split_k == 2 || split_k == 4 || split_k == 8, "K=2048 supports split_k 2, 4, or 8");
    dispatch_split_k2048(x, weight_int4, scales, partial, split_k, rows_per_block);
  } else if (k_in == 6144) {
    TORCH_CHECK(
        split_k == 2 || split_k == 3 || split_k == 4 || split_k == 6 || split_k == 8,
        "K=6144 supports split_k 2, 3, 4, 6, or 8");
    dispatch_split_k6144(x, weight_int4, scales, partial, split_k, rows_per_block);
  } else {
    TORCH_CHECK(false, "unsupported K for int4 partial GEMV");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp",
      &gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp,
      "Symmetric int4 kblock/kscale half-warp split-K GEMV partial for Qwen3-VL decode");
}
