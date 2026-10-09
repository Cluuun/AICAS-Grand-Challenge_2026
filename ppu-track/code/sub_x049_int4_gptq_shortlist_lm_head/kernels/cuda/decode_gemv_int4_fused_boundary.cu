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
constexpr int kDownK = 6144;
constexpr int kBlockK = 128;
constexpr int kSplitK = 4;
constexpr int kRowsPerBlock = 4;
constexpr int kScaleBlocks = kDownK / kBlockK;
constexpr int kChunkBlocks = kScaleBlocks / kSplitK;
constexpr int kPackedBytesPerBlock = kBlockK / 2;
constexpr int kPackedWordsPerBlock = kPackedBytesPerBlock / 4;
constexpr int kKblockWords = kHiddenSize * kPackedWordsPerBlock;
constexpr int kRowTiles = kHiddenSize / kRowsPerBlock;

__forceinline__ __device__ uint32_t u32_from_half2(half2 value) {
  return *reinterpret_cast<uint32_t*>(&value);
}

__forceinline__ __device__ half2 half2_from_u32(uint32_t value) {
  return *reinterpret_cast<half2*>(&value);
}

__forceinline__ __device__ half2 load_half2_cg(const half2* ptr) {
  return half2_from_u32(__ldcg(reinterpret_cast<const uint32_t*>(ptr)));
}

__forceinline__ __device__ float warp_sum32(float value) {
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

__forceinline__ __device__ float halfwarp_broadcast_float(float value, int half_id) {
  return __shfl_sync(0xffffffffu, value, half_id << 4, 32);
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

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2 + x_pair);
    const float scale = __half2float(scales[block_k * kHiddenSize + row]);
    acc = fmaf(block_acc, scale, acc);
  }

  const float part = half_warp_sum(acc);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_bscale_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2 + x_pair);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    acc = fmaf(block_acc, scale, acc);
  }

  const float part = half_warp_sum(acc);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_splitaccum_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc_even = 0.0f;
  float acc_odd = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2 + x_pair);
    const float scale = __half2float(scales[block_k * kHiddenSize + row]);
    if ((block_offset & 1) == 0) {
      acc_even = fmaf(block_acc, scale, acc_even);
    } else {
      acc_odd = fmaf(block_acc, scale, acc_odd);
    }
  }

  const float part = half_warp_sum(acc_even + acc_odd);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_bscale_splitaccum_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc_even = 0.0f;
  float acc_odd = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2 + x_pair);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    if ((block_offset & 1) == 0) {
      acc_even = fmaf(block_acc, scale, acc_even);
    } else {
      acc_odd = fmaf(block_acc, scale, acc_odd);
    }
  }

  const float part = half_warp_sum(acc_even + acc_odd);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_sharedx_bscale_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  for (int idx = tid; idx < kDownK / 2; idx += blockDim.x) {
    x2_shared[idx] = load_half2_cg(x2 + idx);
  }
  __syncthreads();

  float acc = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2_shared + x_pair);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    acc = fmaf(block_acc, scale, acc);
  }

  const float part = half_warp_sum(acc);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_sharedx_bscale_splitaccum_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];
  extern __shared__ __align__(4) unsigned char shared_bytes[];
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  for (int idx = tid; idx < kDownK / 2; idx += blockDim.x) {
    x2_shared[idx] = load_half2_cg(x2 + idx);
  }
  __syncthreads();

  float acc_even = 0.0f;
  float acc_odd = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2_shared + x_pair);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    if ((block_offset & 1) == 0) {
      acc_even = fmaf(block_acc, scale, acc_even);
    } else {
      acc_odd = fmaf(block_acc, scale, acc_odd);
    }
  }

  const float part = half_warp_sum(acc_even + acc_odd);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_sharedxvec_bscale_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  uint4* x4_shared = reinterpret_cast<uint4*>(shared_bytes);
  half2* x2_shared = reinterpret_cast<half2*>(shared_bytes);

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const uint4* x4 = reinterpret_cast<const uint4*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  constexpr int kXVecCount = kDownK / 8;
  for (int idx = tid; idx < kXVecCount; idx += blockDim.x) {
    x4_shared[idx] = x4[idx];
  }
  __syncthreads();

  float acc = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2_shared + x_pair);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    acc = fmaf(block_acc, scale, acc);
  }

  const float part = half_warp_sum(acc);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_bscale_quadaccum_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2 + x_pair);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    switch (block_offset & 3) {
      case 0:
        acc0 = fmaf(block_acc, scale, acc0);
        break;
      case 1:
        acc1 = fmaf(block_acc, scale, acc1);
        break;
      case 2:
        acc2 = fmaf(block_acc, scale, acc2);
        break;
      default:
        acc3 = fmaf(block_acc, scale, acc3);
        break;
    }
  }

  const float part = half_warp_sum(acc0 + acc1 + acc2 + acc3);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_xreuse_bscale_quadaccum_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
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
      x_bits0 = u32_from_half2(load_half2_cg(x2 + x_pair + 0));
      x_bits1 = u32_from_half2(load_half2_cg(x2 + x_pair + 1));
      x_bits2 = u32_from_half2(load_half2_cg(x2 + x_pair + 2));
      x_bits3 = u32_from_half2(load_half2_cg(x2 + x_pair + 3));
    }
    x_bits0 = __shfl_sync(0xffffffffu, x_bits0, lane16, 32);
    x_bits1 = __shfl_sync(0xffffffffu, x_bits1, lane16, 32);
    x_bits2 = __shfl_sync(0xffffffffu, x_bits2, lane16, 32);
    x_bits3 = __shfl_sync(0xffffffffu, x_bits3, lane16, 32);
    const half2 x_local[4] = {
        half2_from_u32(x_bits0),
        half2_from_u32(x_bits1),
        half2_from_u32(x_bits2),
        half2_from_u32(x_bits3),
    };

    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x_local);
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    switch (block_offset & 3) {
      case 0:
        acc0 = fmaf(block_acc, scale, acc0);
        break;
      case 1:
        acc1 = fmaf(block_acc, scale, acc1);
        break;
      case 2:
        acc2 = fmaf(block_acc, scale, acc2);
        break;
      default:
        acc3 = fmaf(block_acc, scale, acc3);
        break;
    }
  }

  const float part = half_warp_sum(acc0 + acc1 + acc2 + acc3);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_u4offset_bscale_quadaccum_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    float u4_acc = 0.0f;
    float x_sum = 0.0f;
    dot8_u4xhalf(__ldcg(row_w32 + lane16), x2 + x_pair, u4_acc, x_sum);
    const float block_acc = u4_acc - 8.0f * x_sum;
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    switch (block_offset & 3) {
      case 0:
        acc0 = fmaf(block_acc, scale, acc0);
        break;
      case 1:
        acc1 = fmaf(block_acc, scale, acc1);
        break;
      case 2:
        acc2 = fmaf(block_acc, scale, acc2);
        break;
      default:
        acc3 = fmaf(block_acc, scale, acc3);
        break;
    }
  }

  const float part = half_warp_sum(acc0 + acc1 + acc2 + acc3);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(256)
void down_int4_split4_sum_sumsq_xreuse_u4offset_bscale_quadaccum_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][kRowsPerBlock];
  __shared__ float row_sumsq[kRowsPerBlock];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id >> 1;
  const int row_in_tile = ((warp_id & 1) << 1) + half_id;
  const int row = blockIdx.x * kRowsPerBlock + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
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
      x_bits0 = u32_from_half2(load_half2_cg(x2 + x_pair + 0));
      x_bits1 = u32_from_half2(load_half2_cg(x2 + x_pair + 1));
      x_bits2 = u32_from_half2(load_half2_cg(x2 + x_pair + 2));
      x_bits3 = u32_from_half2(load_half2_cg(x2 + x_pair + 3));
    }
    x_bits0 = __shfl_sync(0xffffffffu, x_bits0, lane16, 32);
    x_bits1 = __shfl_sync(0xffffffffu, x_bits1, lane16, 32);
    x_bits2 = __shfl_sync(0xffffffffu, x_bits2, lane16, 32);
    x_bits3 = __shfl_sync(0xffffffffu, x_bits3, lane16, 32);
    const half2 x_local[4] = {
        half2_from_u32(x_bits0),
        half2_from_u32(x_bits1),
        half2_from_u32(x_bits2),
        half2_from_u32(x_bits3),
    };

    float u4_acc = 0.0f;
    float x_sum = 0.0f;
    dot8_u4xhalf(__ldcg(row_w32 + lane16), x_local, u4_acc, x_sum);
    const float block_acc = u4_acc - 8.0f * x_sum;
    float scale = 0.0f;
    if (lane16 == 0) {
      scale = __half2float(scales[block_k * kHiddenSize + row]);
    }
    scale = halfwarp_broadcast_float(scale, half_id);
    switch (block_offset & 3) {
      case 0:
        acc0 = fmaf(block_acc, scale, acc0);
        break;
      case 1:
        acc1 = fmaf(block_acc, scale, acc1);
        break;
      case 2:
        acc2 = fmaf(block_acc, scale, acc2);
        break;
      default:
        acc3 = fmaf(block_acc, scale, acc3);
        break;
    }
  }

  const float part = half_warp_sum(acc0 + acc1 + acc2 + acc3);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < kRowsPerBlock) {
    const int out_row = blockIdx.x * kRowsPerBlock + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1] + row_sumsq[2] + row_sumsq[3];
  }
}

__global__ __launch_bounds__(128)
void down_int4_split4_sum_sumsq_rpb2_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ weight_int4,
    const half* __restrict__ scales,
    const half* __restrict__ residual,
    half* __restrict__ sum_out,
    float* __restrict__ tile_sumsq) {
  __shared__ float split_acc[kSplitK][2];
  __shared__ float row_sumsq[2];

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int half_id = lane >> 4;
  const int lane16 = lane & 15;
  const int split = warp_id;
  const int row_in_tile = half_id;
  const int row = blockIdx.x * 2 + row_in_tile;

  const half2* x2 = reinterpret_cast<const half2*>(x);
  const uint32_t* weight_w32 = reinterpret_cast<const uint32_t*>(weight_int4);

  float acc = 0.0f;
#pragma unroll
  for (int block_offset = 0; block_offset < kChunkBlocks; ++block_offset) {
    const int block_k = split * kChunkBlocks + block_offset;
    const uint32_t* row_w32 = weight_w32 + block_k * kKblockWords + row * kPackedWordsPerBlock;
    const int x_pair = block_k * (kBlockK / 2) + lane16 * 4;
    const float block_acc = dot8_int4xhalf(__ldcg(row_w32 + lane16), x2 + x_pair);
    const float scale = __half2float(scales[block_k * kHiddenSize + row]);
    acc = fmaf(block_acc, scale, acc);
  }

  const float part = half_warp_sum(acc);
  if (lane16 == 0) {
    split_acc[split][row_in_tile] = part;
  }
  __syncthreads();

  if (tid < 2) {
    const int out_row = blockIdx.x * 2 + tid;
    const float total =
        split_acc[0][tid] + split_acc[1][tid] + split_acc[2][tid] + split_acc[3][tid];
    const float residual_f = __half2float(residual[out_row]);
    const half summed_h = __float2half_rn(total + residual_f);
    const float summed_f = __half2float(summed_h);
    sum_out[out_row] = summed_h;
    row_sumsq[tid] = summed_f * summed_f;
  }
  __syncthreads();

  if (tid == 0) {
    tile_sumsq[blockIdx.x] = row_sumsq[0] + row_sumsq[1];
  }
}

__global__ __launch_bounds__(256)
void rmsnorm_from_sum_sumsq_2048_kernel(
    const half* __restrict__ sum_out,
    const float* __restrict__ tile_sumsq,
    const half* __restrict__ norm_weight,
    half* __restrict__ norm_out,
    float eps,
    int tile_count) {
  __shared__ float warp_sums[8];
  __shared__ float rstd_shared;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  float local = 0.0f;
  for (int idx = tid; idx < tile_count; idx += 256) {
    local += tile_sumsq[idx];
  }

  const float warp_total = warp_sum32(local);
  if (lane == 0) {
    warp_sums[warp] = warp_total;
  }
  __syncthreads();

  float total = tid < 8 ? warp_sums[tid] : 0.0f;
  total = warp_sum32(total);
  if (tid == 0) {
    rstd_shared = rsqrtf(total / static_cast<float>(kHiddenSize) + eps);
  }
  __syncthreads();

  const float rstd = rstd_shared;
  const int base = tid * 8;

  const uint4 sum4 = *reinterpret_cast<const uint4*>(sum_out + base);
  const uint4 weight4 = *reinterpret_cast<const uint4*>(norm_weight + base);
  const half2 s01 = half2_from_u32(sum4.x);
  const half2 s23 = half2_from_u32(sum4.y);
  const half2 s45 = half2_from_u32(sum4.z);
  const half2 s67 = half2_from_u32(sum4.w);
  const half2 w01 = half2_from_u32(weight4.x);
  const half2 w23 = half2_from_u32(weight4.y);
  const half2 w45 = half2_from_u32(weight4.z);
  const half2 w67 = half2_from_u32(weight4.w);
  const float2 f01 = __half22float2(s01);
  const float2 f23 = __half22float2(s23);
  const float2 f45 = __half22float2(s45);
  const float2 f67 = __half22float2(s67);

  const half2 n01 = __hmul2(
      __halves2half2(__float2half_rn(f01.x * rstd), __float2half_rn(f01.y * rstd)), w01);
  const half2 n23 = __hmul2(
      __halves2half2(__float2half_rn(f23.x * rstd), __float2half_rn(f23.y * rstd)), w23);
  const half2 n45 = __hmul2(
      __halves2half2(__float2half_rn(f45.x * rstd), __float2half_rn(f45.y * rstd)), w45);
  const half2 n67 = __hmul2(
      __halves2half2(__float2half_rn(f67.x * rstd), __float2half_rn(f67.y * rstd)), w67);

  uint4 norm4;
  norm4.x = u32_from_half2(n01);
  norm4.y = u32_from_half2(n23);
  norm4.z = u32_from_half2(n45);
  norm4.w = u32_from_half2(n67);
  *reinterpret_cast<uint4*>(norm_out + base) = norm4;
}

void check_down_inputs(
    const torch::Tensor& x,
    const torch::Tensor& weight_int4,
    const torch::Tensor& scales,
    const torch::Tensor& residual,
    const torch::Tensor& norm_weight,
    const torch::Tensor& sum_out,
    const torch::Tensor& norm_out,
    const torch::Tensor& tile_sumsq,
    int64_t rows_per_block) {
  CHECK_CUDA(x);
  CHECK_CUDA(weight_int4);
  CHECK_CUDA(scales);
  CHECK_CUDA(residual);
  CHECK_CUDA(norm_weight);
  CHECK_CUDA(sum_out);
  CHECK_CUDA(norm_out);
  CHECK_CUDA(tile_sumsq);
  CHECK_CONTIGUOUS(x);
  CHECK_CONTIGUOUS(weight_int4);
  CHECK_CONTIGUOUS(scales);
  CHECK_CONTIGUOUS(residual);
  CHECK_CONTIGUOUS(norm_weight);
  CHECK_CONTIGUOUS(sum_out);
  CHECK_CONTIGUOUS(norm_out);
  CHECK_CONTIGUOUS(tile_sumsq);
  CHECK_HALF(x);
  CHECK_BYTE(weight_int4);
  CHECK_HALF(scales);
  CHECK_HALF(residual);
  CHECK_HALF(norm_weight);
  CHECK_HALF(sum_out);
  CHECK_HALF(norm_out);
  CHECK_FLOAT(tile_sumsq);
  TORCH_CHECK(x.numel() == kDownK, "x must have 6144 elements");
  TORCH_CHECK(weight_int4.numel() == kScaleBlocks * kHiddenSize * kPackedBytesPerBlock,
              "weight_int4 must contain [48, 2048, 64]");
  TORCH_CHECK(scales.numel() == kScaleBlocks * kHiddenSize, "scales must contain [48, 2048]");
  TORCH_CHECK(residual.numel() == kHiddenSize, "residual must have 2048 elements");
  TORCH_CHECK(norm_weight.numel() == kHiddenSize, "norm_weight must have 2048 elements");
  TORCH_CHECK(sum_out.numel() == kHiddenSize, "sum_out must have 2048 elements");
  TORCH_CHECK(norm_out.numel() == kHiddenSize, "norm_out must have 2048 elements");
  TORCH_CHECK(rows_per_block == 2 || rows_per_block == kRowsPerBlock, "rows_per_block must be 2 or 4");
  TORCH_CHECK(tile_sumsq.numel() == kHiddenSize / rows_per_block, "tile_sumsq has wrong length");
}

}  // namespace

void down_int4_split4_sum_sumsq(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor residual,
    torch::Tensor sum_out,
    torch::Tensor tile_sumsq,
    int64_t rows_per_block) {
  auto stream = at::cuda::getCurrentCUDAStream();
  down_int4_split4_sum_sumsq_kernel<<<kRowTiles, 256, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const uint8_t*>(weight_int4.data_ptr<uint8_t>()),
      reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<half*>(sum_out.data_ptr<at::Half>()),
      reinterpret_cast<float*>(tile_sumsq.data_ptr<float>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rmsnorm_from_sum_sumsq_2048(
    torch::Tensor sum_out,
    torch::Tensor tile_sumsq,
    torch::Tensor norm_weight,
    torch::Tensor norm_out,
    double eps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  rmsnorm_from_sum_sumsq_2048_kernel<<<1, 256, 0, stream>>>(
      reinterpret_cast<const half*>(sum_out.data_ptr<at::Half>()),
      reinterpret_cast<const float*>(tile_sumsq.data_ptr<float>()),
      reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(norm_out.data_ptr<at::Half>()),
      static_cast<float>(eps),
      static_cast<int>(tile_sumsq.numel()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void down_int4_split4_add_rmsnorm_boundary(
    torch::Tensor x,
    torch::Tensor weight_int4,
    torch::Tensor scales,
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor sum_out,
    torch::Tensor norm_out,
    torch::Tensor tile_sumsq,
    double eps,
    int64_t rows_per_block) {
  check_down_inputs(x, weight_int4, scales, residual, norm_weight, sum_out, norm_out, tile_sumsq, rows_per_block);
  TORCH_CHECK(rows_per_block == 4, "down INT4 fused boundary only keeps the validated rows_per_block=4 path");
  down_int4_split4_sum_sumsq(x, weight_int4, scales, residual, sum_out, tile_sumsq, rows_per_block);
  rmsnorm_from_sum_sumsq_2048(sum_out, tile_sumsq, norm_weight, norm_out, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "down_int4_split4_add_rmsnorm_boundary",
      &down_int4_split4_add_rmsnorm_boundary,
      "Experimental down INT4 fused split4 sum/sumsq plus RMSNorm boundary");
  m.def(
      "down_int4_split4_sum_sumsq",
      &down_int4_split4_sum_sumsq,
      "Experimental down INT4 split4 GEMV producing sum and tile sumsq");
  m.def(
      "rmsnorm_from_sum_sumsq_2048",
      &rmsnorm_from_sum_sumsq_2048,
      "RMSNorm from precomputed sum and tile sumsq");
}
