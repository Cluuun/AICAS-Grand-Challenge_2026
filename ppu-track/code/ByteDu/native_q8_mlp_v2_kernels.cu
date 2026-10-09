#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHidden = 2048;
constexpr int kIntermediate = 6144;
constexpr int kQkv = 4096;
constexpr int kGatePackedK = kHidden / 4;
constexpr int kDownPackedK = kIntermediate / 4;
constexpr int kQkvPackedK = kHidden / 4;
constexpr int kThreads = 256;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (warp == 0) {
        v = warp_sum(v);
    }
    return v;
}

__global__ void cast_fp16_to_bf16_kernel(
    const half* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        output[idx] = __float2bfloat16(__half2float(input[idx]));
    }
}

__global__ void add_fp16_residual_sumsq_kernel(
    const half* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ sumsq) {
    const int tid = threadIdx.x;
    float sq = 0.0f;
    for (int idx = tid; idx < kHidden; idx += blockDim.x) {
        const float out = __half2float(input[idx]) + bf16_to_float(residual[idx]);
        residual_out[idx] = __float2bfloat16(out);
        sq += out * out;
    }
    const float sq_sum = block_sum(sq);
    if (tid == 0) {
        sumsq[0] = sq_sum;
    }
}

__global__ void add_rms_norm_fp16_kernel(
    const half* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ residual_out,
    half* __restrict__ normed_output,
    float eps) {
    const int tid = threadIdx.x;
    constexpr int kElemsPerThread = kHidden / kThreads;
    float out_vals[kElemsPerThread];
    float sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < kElemsPerThread; ++it) {
        const int idx = tid + it * kThreads;
        const float out = __half2float(input[idx]) + bf16_to_float(residual[idx]);
        out_vals[it] = out;
        sq += out * out;
    }
    __shared__ float rstd_shared;
    const float sq_sum = block_sum(sq);
    if (tid == 0) {
        rstd_shared = rsqrtf(sq_sum / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;
    #pragma unroll
    for (int it = 0; it < kElemsPerThread; ++it) {
        const int idx = tid + it * kThreads;
        const float out = out_vals[it];
        residual_out[idx] = __float2bfloat16(out);
        normed_output[idx] = __float2half(out * bf16_to_float(norm_weight[idx]) * rstd);
    }
}

__global__ void swiglu_packed_fp16_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int n_rows,
    int n_cols) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_rows * n_cols;
    for (int idx = tid; idx < total; idx += blockDim.x * gridDim.x) {
        const int row = idx / n_cols;
        const int col = idx - row * n_cols;
        const int base = row * (2 * n_cols);
        const float gate = __half2float(input[base + (2 * col)]);
        const float up = __half2float(input[base + (2 * col) + 1]);
        const float silu = gate / (1.0f + expf(-gate));
        output[idx] = __float2half(silu * up);
    }
}

__device__ __forceinline__ void unpack_q8_word(const int word, float& w0, float& w1, float& w2, float& w3) {
    w0 = static_cast<float>(word & 0xFF) - 128.0f;
    w1 = static_cast<float>((word >> 8) & 0xFF) - 128.0f;
    w2 = static_cast<float>((word >> 16) & 0xFF) - 128.0f;
    w3 = static_cast<float>((word >> 24) & 0xFF) - 128.0f;
}

template <int RowsPerBlock, int PackedK>
__device__ __forceinline__ void clear_acc(float (&acc)[RowsPerBlock]) {
    #pragma unroll
    for (int row = 0; row < RowsPerBlock; ++row) {
        acc[row] = 0.0f;
    }
}

template <int RowsPerBlock, int BlockPackedK>
__global__ void gate_up_q8_v2_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    const int first_row = blockIdx.x * RowsPerBlock;
    const int tid = threadIdx.x;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = block_sum(sq);
    if (tid == 0) {
        rstd_shared = rsqrtf(sq_sum / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;

    float gate_acc[RowsPerBlock];
    float up_acc[RowsPerBlock];
    clear_acc<RowsPerBlock, BlockPackedK>(gate_acc);
    clear_acc<RowsPerBlock, BlockPackedK>(up_acc);

    for (int pk = tid; pk < kGatePackedK; pk += blockDim.x) {
        const int base = pk * 4;
        const float x0 = bf16_to_float(residual[base]) * bf16_to_float(norm_weight[base]) * rstd;
        const float x1 = bf16_to_float(residual[base + 1]) * bf16_to_float(norm_weight[base + 1]) * rstd;
        const float x2 = bf16_to_float(residual[base + 2]) * bf16_to_float(norm_weight[base + 2]) * rstd;
        const float x3 = bf16_to_float(residual[base + 3]) * bf16_to_float(norm_weight[base + 3]) * rstd;

        #pragma unroll
        for (int rr = 0; rr < RowsPerBlock; ++rr) {
            const int out_row = first_row + rr;
            if (out_row < kIntermediate) {
                const int gate_row = out_row * 2;
                const int up_row = gate_row + 1;
                float gw0, gw1, gw2, gw3;
                float uw0, uw1, uw2, uw3;
                unpack_q8_word(packed_weight[gate_row * kGatePackedK + pk], gw0, gw1, gw2, gw3);
                unpack_q8_word(packed_weight[up_row * kGatePackedK + pk], uw0, uw1, uw2, uw3);
                gate_acc[rr] += gw0 * x0 + gw1 * x1 + gw2 * x2 + gw3 * x3;
                up_acc[rr] += uw0 * x0 + uw1 * x1 + uw2 * x2 + uw3 * x3;
            }
        }
    }

    #pragma unroll
    for (int rr = 0; rr < RowsPerBlock; ++rr) {
        gate_acc[rr] = block_sum(gate_acc[rr]);
        __syncthreads();
        up_acc[rr] = block_sum(up_acc[rr]);
        __syncthreads();
    }

    if (tid == 0) {
        #pragma unroll
        for (int rr = 0; rr < RowsPerBlock; ++rr) {
            const int out_row = first_row + rr;
            if (out_row < kIntermediate) {
                const int gate_row = out_row * 2;
                const int up_row = gate_row + 1;
                const float gate = gate_acc[rr] * bf16_to_float(scale[gate_row]);
                const float up = up_acc[rr] * bf16_to_float(scale[up_row]);
                const float silu = gate / (1.0f + expf(-gate));
                output[out_row] = __float2bfloat16(silu * up);
            }
        }
    }
}

template <int WarpsPerBlock>
__global__ void gate_up_q8_v2_warp_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;

    float sq = 0.0f;
    for (int k = threadIdx.x; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = block_sum(sq);
    if (threadIdx.x == 0) {
        rstd_shared = rsqrtf(sq_sum / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;

    if (row >= kIntermediate) {
        return;
    }
    const int gate_row = row * 2;
    const int up_row = gate_row + 1;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int pk = lane; pk < kGatePackedK; pk += 32) {
        const int base = pk * 4;
        const float x0 = bf16_to_float(residual[base]) * bf16_to_float(norm_weight[base]) * rstd;
        const float x1 = bf16_to_float(residual[base + 1]) * bf16_to_float(norm_weight[base + 1]) * rstd;
        const float x2 = bf16_to_float(residual[base + 2]) * bf16_to_float(norm_weight[base + 2]) * rstd;
        const float x3 = bf16_to_float(residual[base + 3]) * bf16_to_float(norm_weight[base + 3]) * rstd;
        float gw0, gw1, gw2, gw3;
        float uw0, uw1, uw2, uw3;
        unpack_q8_word(packed_weight[gate_row * kGatePackedK + pk], gw0, gw1, gw2, gw3);
        unpack_q8_word(packed_weight[up_row * kGatePackedK + pk], uw0, uw1, uw2, uw3);
        gate_acc += gw0 * x0 + gw1 * x1 + gw2 * x2 + gw3 * x3;
        up_acc += uw0 * x0 + uw1 * x1 + uw2 * x2 + uw3 * x3;
    }
    gate_acc = warp_sum(gate_acc);
    up_acc = warp_sum(up_acc);
    if (lane == 0) {
        gate_acc *= bf16_to_float(scale[gate_row]);
        up_acc *= bf16_to_float(scale[up_row]);
        const float silu = gate_acc / (1.0f + expf(-gate_acc));
        output[row] = __float2bfloat16(silu * up_acc);
    }
}

__global__ void rms_norm_store_bf16_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ normed_hidden,
    float eps) {
    const int tid = threadIdx.x;
    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = block_sum(sq);
    if (tid == 0) {
        rstd_shared = rsqrtf(sq_sum / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]) * bf16_to_float(norm_weight[k]) * rstd;
        normed_hidden[k] = __float2bfloat16(x);
    }
}

__global__ void rms_norm_store_fp16_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    half* __restrict__ normed_hidden,
    float eps) {
    const int tid = threadIdx.x;
    constexpr int kElemsPerThread = kHidden / kThreads;
    float vals[kElemsPerThread];
    float sq = 0.0f;
    #pragma unroll
    for (int it = 0; it < kElemsPerThread; ++it) {
        const int k = tid + it * kThreads;
        const float x = bf16_to_float(residual[k]);
        vals[it] = x;
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = block_sum(sq);
    if (tid == 0) {
        rstd_shared = rsqrtf(sq_sum / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;
    #pragma unroll
    for (int it = 0; it < kElemsPerThread; ++it) {
        const int k = tid + it * kThreads;
        normed_hidden[k] = __float2half(vals[it] * bf16_to_float(norm_weight[k]) * rstd);
    }
}

__global__ void rms_norm_store_from_sumsq_bf16_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const float* __restrict__ sumsq,
    __nv_bfloat16* __restrict__ normed_hidden,
    float eps) {
    const int tid = threadIdx.x;
    __shared__ float rstd_shared;
    if (tid == 0) {
        rstd_shared = rsqrtf(sumsq[0] / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;
    for (int k = tid; k < kHidden; k += blockDim.x) {
        const float x = bf16_to_float(residual[k]) * bf16_to_float(norm_weight[k]) * rstd;
        normed_hidden[k] = __float2bfloat16(x);
    }
}

template <int WarpsPerBlock>
__global__ void gate_up_q8_v3_warp_kernel(
    const __nv_bfloat16* __restrict__ normed_hidden,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;

    if (row >= kIntermediate) {
        return;
    }
    const int gate_row = row * 2;
    const int up_row = gate_row + 1;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int pk = lane; pk < kGatePackedK; pk += 32) {
        const int base = pk * 4;
        const float x0 = bf16_to_float(normed_hidden[base]);
        const float x1 = bf16_to_float(normed_hidden[base + 1]);
        const float x2 = bf16_to_float(normed_hidden[base + 2]);
        const float x3 = bf16_to_float(normed_hidden[base + 3]);
        float gw0, gw1, gw2, gw3;
        float uw0, uw1, uw2, uw3;
        unpack_q8_word(packed_weight[gate_row * kGatePackedK + pk], gw0, gw1, gw2, gw3);
        unpack_q8_word(packed_weight[up_row * kGatePackedK + pk], uw0, uw1, uw2, uw3);
        gate_acc += gw0 * x0 + gw1 * x1 + gw2 * x2 + gw3 * x3;
        up_acc += uw0 * x0 + uw1 * x1 + uw2 * x2 + uw3 * x3;
    }
    gate_acc = warp_sum(gate_acc);
    up_acc = warp_sum(up_acc);
    if (lane == 0) {
        gate_acc *= bf16_to_float(scale[gate_row]);
        up_acc *= bf16_to_float(scale[up_row]);
        const float silu = gate_acc / (1.0f + expf(-gate_acc));
        output[row] = __float2bfloat16(silu * up_acc);
    }
}

template <int WarpsPerBlock>
__global__ void norm_qkv_q8_v3_warp_kernel(
    const __nv_bfloat16* __restrict__ normed_hidden,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= kQkv) {
        return;
    }
    float acc = 0.0f;
    for (int pk = lane; pk < kQkvPackedK; pk += 32) {
        const int base = pk * 4;
        const float x0 = bf16_to_float(normed_hidden[base]);
        const float x1 = bf16_to_float(normed_hidden[base + 1]);
        const float x2 = bf16_to_float(normed_hidden[base + 2]);
        const float x3 = bf16_to_float(normed_hidden[base + 3]);
        float w0, w1, w2, w3;
        unpack_q8_word(packed_weight[row * kQkvPackedK + pk], w0, w1, w2, w3);
        acc += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        output[row] = __float2bfloat16(acc * bf16_to_float(scale[row]));
    }
}

template <int RowsPerBlock>
__global__ void down_add_sumsq_q8_v2_kernel(
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ partial_sumsq) {
    const int first_row = blockIdx.x * RowsPerBlock;
    const int tid = threadIdx.x;
    float acc[RowsPerBlock];
    clear_acc<RowsPerBlock, kDownPackedK>(acc);

    for (int pk = tid; pk < kDownPackedK; pk += blockDim.x) {
        const int base = pk * 4;
        const float x0 = bf16_to_float(input[base]);
        const float x1 = bf16_to_float(input[base + 1]);
        const float x2 = bf16_to_float(input[base + 2]);
        const float x3 = bf16_to_float(input[base + 3]);

        #pragma unroll
        for (int rr = 0; rr < RowsPerBlock; ++rr) {
            const int row = first_row + rr;
            if (row < kHidden) {
                float w0, w1, w2, w3;
                unpack_q8_word(packed_weight[row * kDownPackedK + pk], w0, w1, w2, w3);
                acc[rr] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
            }
        }
    }

    float block_sq = 0.0f;
    #pragma unroll
    for (int rr = 0; rr < RowsPerBlock; ++rr) {
        acc[rr] = block_sum(acc[rr]);
        __syncthreads();
        if (tid == 0) {
            const int row = first_row + rr;
            if (row < kHidden) {
                const float out = acc[rr] * bf16_to_float(scale[row]) + bf16_to_float(residual[row]);
                residual_out[row] = __float2bfloat16(out);
                block_sq += out * out;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_sumsq[blockIdx.x] = block_sq;
    }
}

template <int WarpsPerBlock>
__global__ void down_add_sumsq_q8_v2_warp_kernel(
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ sumsq) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;

    __shared__ float block_sq;
    if (threadIdx.x == 0) {
        block_sq = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;
    if (row < kHidden) {
        for (int pk = lane; pk < kDownPackedK; pk += 32) {
            const int base = pk * 4;
            const float x0 = bf16_to_float(input[base]);
            const float x1 = bf16_to_float(input[base + 1]);
            const float x2 = bf16_to_float(input[base + 2]);
            const float x3 = bf16_to_float(input[base + 3]);
            float w0, w1, w2, w3;
            unpack_q8_word(packed_weight[row * kDownPackedK + pk], w0, w1, w2, w3);
            acc += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            const float out = acc * bf16_to_float(scale[row]) + bf16_to_float(residual[row]);
            residual_out[row] = __float2bfloat16(out);
            atomicAdd(&block_sq, out * out);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(sumsq, block_sq);
    }
}

template <int RowsPerBlock>
__global__ void norm_qkv_q8_v2_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const float* __restrict__ sumsq,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    const int first_row = blockIdx.x * RowsPerBlock;
    const int tid = threadIdx.x;

    __shared__ float rstd_shared;
    if (tid == 0) {
        rstd_shared = rsqrtf(sumsq[0] / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;

    float acc[RowsPerBlock];
    clear_acc<RowsPerBlock, kQkvPackedK>(acc);
    for (int pk = tid; pk < kQkvPackedK; pk += blockDim.x) {
        const int base = pk * 4;
        const float x0 = bf16_to_float(residual[base]) * bf16_to_float(norm_weight[base]) * rstd;
        const float x1 = bf16_to_float(residual[base + 1]) * bf16_to_float(norm_weight[base + 1]) * rstd;
        const float x2 = bf16_to_float(residual[base + 2]) * bf16_to_float(norm_weight[base + 2]) * rstd;
        const float x3 = bf16_to_float(residual[base + 3]) * bf16_to_float(norm_weight[base + 3]) * rstd;
        #pragma unroll
        for (int rr = 0; rr < RowsPerBlock; ++rr) {
            const int row = first_row + rr;
            if (row < kQkv) {
                float w0, w1, w2, w3;
                unpack_q8_word(packed_weight[row * kQkvPackedK + pk], w0, w1, w2, w3);
                acc[rr] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3;
            }
        }
    }

    #pragma unroll
    for (int rr = 0; rr < RowsPerBlock; ++rr) {
        acc[rr] = block_sum(acc[rr]);
        __syncthreads();
        if (tid == 0) {
            const int row = first_row + rr;
            if (row < kQkv) {
                output[row] = __float2bfloat16(acc[rr] * bf16_to_float(scale[row]));
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// v4: vectorized (128-bit int4 weight loads) + shared-memory normed input.
// Goal: exceed the prebuilt fused kernel's effective HBM bandwidth by issuing
// 128-bit coalesced weight loads and reading the (small) activation vector from
// shared memory instead of redundantly from L2 across every warp.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void unpack_q8_int4(const int4 packed, float (&w)[16]) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int word = (&packed.x)[j];
        w[j * 4 + 0] = static_cast<float>(word & 0xFF) - 128.0f;
        w[j * 4 + 1] = static_cast<float>((word >> 8) & 0xFF) - 128.0f;
        w[j * 4 + 2] = static_cast<float>((word >> 16) & 0xFF) - 128.0f;
        w[j * 4 + 3] = static_cast<float>((word >> 24) & 0xFF) - 128.0f;
    }
}

template <int WarpsPerBlock>
__global__ void gate_up_q8_v4_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ float s_normed[kHidden];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += nthreads) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = block_sum(sq);
    if (tid == 0) {
        rstd_shared = rsqrtf(sq_sum / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;
    for (int k = tid; k < kHidden; k += nthreads) {
        s_normed[k] = bf16_to_float(residual[k]) * bf16_to_float(norm_weight[k]) * rstd;
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= kIntermediate) {
        return;
    }
    const int gate_row = row * 2;
    const int up_row = gate_row + 1;
    const int4* gate_ptr = reinterpret_cast<const int4*>(packed_weight + gate_row * kGatePackedK);
    const int4* up_ptr = reinterpret_cast<const int4*>(packed_weight + up_row * kGatePackedK);
    constexpr int kInt4PerRow = kGatePackedK / 4;  // 512 / 4 = 128

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int g = lane; g < kInt4PerRow; g += 32) {
        const int4 gw = gate_ptr[g];
        const int4 uw = up_ptr[g];
        float gwv[16];
        float uwv[16];
        unpack_q8_int4(gw, gwv);
        unpack_q8_int4(uw, uwv);
        const int ibase = g * 16;
        #pragma unroll
        for (int e = 0; e < 16; ++e) {
            const float x = s_normed[ibase + e];
            gate_acc += gwv[e] * x;
            up_acc += uwv[e] * x;
        }
    }
    gate_acc = warp_sum(gate_acc);
    up_acc = warp_sum(up_acc);
    if (lane == 0) {
        gate_acc *= bf16_to_float(scale[gate_row]);
        up_acc *= bf16_to_float(scale[up_row]);
        const float silu = gate_acc / (1.0f + expf(-gate_acc));
        output[row] = __float2bfloat16(silu * up_acc);
    }
}

template <int WarpsPerBlock>
__global__ void norm_qkv_sumsq_q8_v4_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    float* __restrict__ sumsq_out,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    float eps) {
    __shared__ float s_normed[kHidden];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float sq = 0.0f;
    for (int k = tid; k < kHidden; k += nthreads) {
        const float x = bf16_to_float(residual[k]);
        sq += x * x;
    }
    __shared__ float rstd_shared;
    const float sq_sum = block_sum(sq);
    if (tid == 0) {
        sumsq_out[0] = sq_sum;
        rstd_shared = rsqrtf(sq_sum / static_cast<float>(kHidden) + eps);
    }
    __syncthreads();
    const float rstd = rstd_shared;
    for (int k = tid; k < kHidden; k += nthreads) {
        s_normed[k] = bf16_to_float(residual[k]) * bf16_to_float(norm_weight[k]) * rstd;
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= kQkv) {
        return;
    }
    const int4* w_ptr = reinterpret_cast<const int4*>(packed_weight + row * kQkvPackedK);
    constexpr int kInt4PerRow = kQkvPackedK / 4;  // 512 / 4 = 128
    float acc = 0.0f;
    for (int g = lane; g < kInt4PerRow; g += 32) {
        const int4 pw = w_ptr[g];
        float wv[16];
        unpack_q8_int4(pw, wv);
        const int ibase = g * 16;
        #pragma unroll
        for (int e = 0; e < 16; ++e) {
            acc += wv[e] * s_normed[ibase + e];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) {
        output[row] = __float2bfloat16(acc * bf16_to_float(scale[row]));
    }
}

template <int WarpsPerBlock>
__global__ void down_add_sumsq_q8_v4_kernel(
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ sumsq) {
    __shared__ float s_input[kIntermediate];
    __shared__ float s_partial[WarpsPerBlock];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    for (int k = tid; k < kIntermediate; k += nthreads) {
        s_input[k] = bf16_to_float(input[k]);
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    const int4* w_ptr = reinterpret_cast<const int4*>(packed_weight + row * kDownPackedK);
    constexpr int kInt4PerRow = kDownPackedK / 4;  // 1536 / 4 = 384

    float acc = 0.0f;
    if (row < kHidden) {
        for (int g = lane; g < kInt4PerRow; g += 32) {
            const int4 pw = w_ptr[g];
            float wv[16];
            unpack_q8_int4(pw, wv);
            const int ibase = g * 16;
            #pragma unroll
            for (int e = 0; e < 16; ++e) {
                acc += wv[e] * s_input[ibase + e];
            }
        }
        acc = warp_sum(acc);
    }
    float out = 0.0f;
    if (row < kHidden && lane == 0) {
        out = acc * bf16_to_float(scale[row]) + bf16_to_float(residual[row]);
        residual_out[row] = __float2bfloat16(out);
    }
    if (lane == 0) {
        s_partial[warp] = out * out;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sq = 0.0f;
        #pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            block_sq += s_partial[w];
        }
        atomicAdd(sumsq, block_sq);
    }
}

// ---------------------------------------------------------------------------
// v5: INT4 weight-only down_proj. Weights packed as 8 signed nibbles per int32
// with a per-row BF16 scale. Halves weight HBM traffic vs INT8 packed.
// kDownPackedK4 = kIntermediate / 8 = 768 int32 words per output row.
// ---------------------------------------------------------------------------
constexpr int kDownPackedK4 = kIntermediate / 8;  // 768

__device__ __forceinline__ void unpack_w4_int4(const int4 packed, float (&w)[32]) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned int word = static_cast<unsigned int>((&packed.x)[j]);
        #pragma unroll
        for (int n = 0; n < 8; ++n) {
            // signed 4-bit value in [-8, 7]
            int v = static_cast<int>((word >> (n * 4)) & 0xF);
            v -= (v & 0x8) << 1;  // sign-extend 4-bit
            w[j * 8 + n] = static_cast<float>(v);
        }
    }
}

template <int WarpsPerBlock>
__global__ void down_add_sumsq_w4_v5_kernel(
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ sumsq) {
    __shared__ float s_input[kIntermediate];
    __shared__ float s_partial[WarpsPerBlock];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    for (int k = tid; k < kIntermediate; k += nthreads) {
        s_input[k] = bf16_to_float(input[k]);
    }
    __syncthreads();

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    const int4* w_ptr = reinterpret_cast<const int4*>(packed_weight + row * kDownPackedK4);
    constexpr int kInt4PerRow = kDownPackedK4 / 4;  // 768 / 4 = 192

    float acc = 0.0f;
    if (row < kHidden) {
        for (int g = lane; g < kInt4PerRow; g += 32) {
            const int4 pw = w_ptr[g];
            float wv[32];
            unpack_w4_int4(pw, wv);
            const int ibase = g * 32;
            #pragma unroll
            for (int e = 0; e < 32; ++e) {
                acc += wv[e] * s_input[ibase + e];
            }
        }
        acc = warp_sum(acc);
    }
    float out = 0.0f;
    if (row < kHidden && lane == 0) {
        out = acc * bf16_to_float(scale[row]) + bf16_to_float(residual[row]);
        residual_out[row] = __float2bfloat16(out);
    }
    if (lane == 0) {
        s_partial[warp] = out * out;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sq = 0.0f;
        #pragma unroll
        for (int w = 0; w < WarpsPerBlock; ++w) {
            block_sq += s_partial[w];
        }
        atomicAdd(sumsq, block_sq);
    }
}

__global__ void reduce_sumsq_kernel(const float* __restrict__ partial_sumsq, float* __restrict__ sumsq, int n) {
    float acc = 0.0f;
    for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
        acc += partial_sumsq[idx];
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) {
        sumsq[0] = acc;
    }
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void gate_up_q8_v2_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int rows_per_block) {
    check_tensor(residual, "residual");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(packed_weight, "packed_weight");
    check_tensor(scale, "scale");
    check_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(output.numel() == kIntermediate, "output must have intermediate size 6144");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == 2 * kIntermediate && packed_weight.size(1) == kGatePackedK,
                "packed_weight must be [12288, 512]");
    TORCH_CHECK(scale.numel() >= 2 * kIntermediate, "scale is too small");

    auto stream = at::cuda::getCurrentCUDAStream();
    if (rows_per_block == 2) {
        const int grid = (kIntermediate + 1) / 2;
        gate_up_q8_v2_warp_kernel<2><<<grid, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    } else if (rows_per_block == 8) {
        const int grid = (kIntermediate + 7) / 8;
        gate_up_q8_v2_warp_kernel<8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    } else {
        const int grid = (kIntermediate + 3) / 4;
        gate_up_q8_v2_warp_kernel<4><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    }
}

void gate_up_q8_v3_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor normed_hidden,
    torch::Tensor output,
    double eps,
    int rows_per_block) {
    check_tensor(residual, "residual");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(packed_weight, "packed_weight");
    check_tensor(scale, "scale");
    check_tensor(normed_hidden, "normed_hidden");
    check_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(normed_hidden.scalar_type() == at::kBFloat16, "normed_hidden must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(normed_hidden.numel() == kHidden, "normed_hidden must have hidden size 2048");
    TORCH_CHECK(output.numel() == kIntermediate, "output must have intermediate size 6144");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == 2 * kIntermediate && packed_weight.size(1) == kGatePackedK,
                "packed_weight must be [12288, 512]");
    TORCH_CHECK(scale.numel() >= 2 * kIntermediate, "scale is too small");

    auto stream = at::cuda::getCurrentCUDAStream();
    rms_norm_store_bf16_kernel<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
        static_cast<float>(eps));
    if (rows_per_block == 2) {
        const int grid = (kIntermediate + 1) / 2;
        gate_up_q8_v3_warp_kernel<2><<<grid, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    } else if (rows_per_block == 8) {
        const int grid = (kIntermediate + 7) / 8;
        gate_up_q8_v3_warp_kernel<8><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    } else {
        const int grid = (kIntermediate + 3) / 4;
        gate_up_q8_v3_warp_kernel<4><<<grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()));
    }
}

void rms_norm_store_fp16_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor output,
    double eps) {
    check_tensor(residual, "residual");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kHalf, "output must be FP16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(output.numel() == kHidden, "output must have hidden size 2048");
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_norm_store_fp16_kernel<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        static_cast<float>(eps));
}

void swiglu_packed_fp16_cuda(
    torch::Tensor input,
    torch::Tensor output) {
    check_tensor(input, "input");
    check_tensor(output, "output");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be FP16");
    TORCH_CHECK(output.scalar_type() == at::kHalf, "output must be FP16");
    TORCH_CHECK(input.dim() == 2 && output.dim() == 2, "input/output must be 2D");
    TORCH_CHECK(input.size(0) == output.size(0), "row count mismatch");
    TORCH_CHECK(input.size(1) == output.size(1) * 2, "expected interleaved [gate0,up0,gate1,up1,...] input");
    const int n_rows = static_cast<int>(input.size(0));
    const int n_cols = static_cast<int>(output.size(1));
    const int total = n_rows * n_cols;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = std::max(1, (total + kThreads - 1) / kThreads);
    swiglu_packed_fp16_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        n_rows,
        n_cols);
}

void cast_fp16_to_bf16_cuda(
    torch::Tensor input,
    torch::Tensor output) {
    check_tensor(input, "input");
    check_tensor(output, "output");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be FP16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(input.numel() == output.numel(), "input/output size mismatch");
    const int count = static_cast<int>(input.numel());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = std::max(1, (count + kThreads - 1) / kThreads);
    cast_fp16_to_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        count);
}

void add_norm_qkv_fp16_q8_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor normed_hidden,
    torch::Tensor qkv_packed_weight,
    torch::Tensor qkv_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps,
    int rows_per_block) {
    check_tensor(input, "input");
    check_tensor(residual, "residual");
    check_tensor(residual_out, "residual_out");
    check_tensor(sumsq, "sumsq");
    check_tensor(normed_hidden, "normed_hidden");
    check_tensor(qkv_packed_weight, "qkv_packed_weight");
    check_tensor(qkv_scale, "qkv_scale");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(qkv_output, "qkv_output");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be FP16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(normed_hidden.scalar_type() == at::kBFloat16, "normed_hidden must be BF16");
    TORCH_CHECK(qkv_packed_weight.scalar_type() == at::kInt, "qkv_packed_weight must be int32");
    TORCH_CHECK(qkv_scale.scalar_type() == at::kBFloat16, "qkv_scale must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(qkv_output.scalar_type() == at::kBFloat16, "qkv_output must be BF16");
    TORCH_CHECK(input.numel() == kHidden, "input must have hidden size 2048");
    TORCH_CHECK(residual.numel() == kHidden && residual_out.numel() == kHidden, "residual tensors must have hidden size 2048");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq is too small");
    TORCH_CHECK(normed_hidden.numel() == kHidden, "normed_hidden must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(
        qkv_packed_weight.dim() == 2 && qkv_packed_weight.size(0) == kQkv && qkv_packed_weight.size(1) == kQkvPackedK,
        "qkv_packed_weight must be [4096, 512]");
    TORCH_CHECK(qkv_scale.numel() >= kQkv, "qkv_scale is too small");
    TORCH_CHECK(qkv_output.numel() == kQkv, "qkv_output must have qkv size 4096");

    auto stream = at::cuda::getCurrentCUDAStream();
    add_fp16_residual_sumsq_kernel<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
        sumsq.data_ptr<float>());
    rms_norm_store_from_sumsq_bf16_kernel<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        sumsq.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
        static_cast<float>(eps));

    if (rows_per_block == 2) {
        const int qkv_grid = (kQkv + 1) / 2;
        norm_qkv_q8_v3_warp_kernel<2><<<qkv_grid, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()));
    } else if (rows_per_block == 8) {
        const int qkv_grid = (kQkv + 7) / 8;
        norm_qkv_q8_v3_warp_kernel<8><<<qkv_grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()));
    } else {
        const int qkv_grid = (kQkv + 3) / 4;
        norm_qkv_q8_v3_warp_kernel<4><<<qkv_grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()));
    }
}

void add_rms_norm_fp16_cuda(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor norm_weight,
    torch::Tensor normed_output,
    double eps) {
    check_tensor(input, "input");
    check_tensor(residual, "residual");
    check_tensor(residual_out, "residual_out");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(normed_output, "normed_output");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be FP16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(normed_output.scalar_type() == at::kHalf, "normed_output must be FP16");
    TORCH_CHECK(input.numel() == kHidden, "input must have hidden size 2048");
    TORCH_CHECK(residual.numel() == kHidden && residual_out.numel() == kHidden, "residual tensors must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(normed_output.numel() == kHidden, "normed_output must have hidden size 2048");
    auto stream = at::cuda::getCurrentCUDAStream();
    add_rms_norm_fp16_kernel<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<half*>(normed_output.data_ptr<at::Half>()),
        static_cast<float>(eps));
}

void down_add_sumsq_q8_v2_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor partial_sumsq,
    int rows_per_block) {
    check_tensor(packed_weight, "packed_weight");
    check_tensor(scale, "scale");
    check_tensor(input, "input");
    check_tensor(residual, "residual");
    check_tensor(residual_out, "residual_out");
    check_tensor(sumsq, "sumsq");
    check_tensor(partial_sumsq, "partial_sumsq");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(partial_sumsq.scalar_type() == at::kFloat, "partial_sumsq must be FP32");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kHidden && packed_weight.size(1) == kDownPackedK,
                "packed_weight must be [2048, 1536]");
    TORCH_CHECK(input.numel() == kIntermediate, "input must have intermediate size 6144");
    TORCH_CHECK(residual.numel() == kHidden && residual_out.numel() == kHidden, "residual tensors must have hidden size 2048");
    TORCH_CHECK(scale.numel() >= kHidden, "scale is too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (rows_per_block == 2) {
        const int grid = (kHidden + 1) / 2;
        TORCH_CHECK(partial_sumsq.numel() >= grid, "partial_sumsq is too small");
        down_add_sumsq_q8_v2_kernel<2><<<grid, kThreads, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            partial_sumsq.data_ptr<float>());
        reduce_sumsq_kernel<<<1, kThreads, 0, stream>>>(partial_sumsq.data_ptr<float>(), sumsq.data_ptr<float>(), grid);
    } else if (rows_per_block == 8) {
        const int grid = (kHidden + 7) / 8;
        TORCH_CHECK(partial_sumsq.numel() >= grid, "partial_sumsq is too small");
        down_add_sumsq_q8_v2_kernel<8><<<grid, kThreads, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            partial_sumsq.data_ptr<float>());
        reduce_sumsq_kernel<<<1, kThreads, 0, stream>>>(partial_sumsq.data_ptr<float>(), sumsq.data_ptr<float>(), grid);
    } else {
        const int grid = (kHidden + 3) / 4;
        TORCH_CHECK(partial_sumsq.numel() >= grid, "partial_sumsq is too small");
        down_add_sumsq_q8_v2_kernel<4><<<grid, kThreads, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            partial_sumsq.data_ptr<float>());
        reduce_sumsq_kernel<<<1, kThreads, 0, stream>>>(partial_sumsq.data_ptr<float>(), sumsq.data_ptr<float>(), grid);
    }
}

void down_norm_qkv_q8_v2_cuda(
    torch::Tensor down_packed_weight,
    torch::Tensor down_scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor qkv_packed_weight,
    torch::Tensor qkv_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps,
    int rows_per_block) {
    check_tensor(down_packed_weight, "down_packed_weight");
    check_tensor(down_scale, "down_scale");
    check_tensor(input, "input");
    check_tensor(residual, "residual");
    check_tensor(residual_out, "residual_out");
    check_tensor(sumsq, "sumsq");
    check_tensor(qkv_packed_weight, "qkv_packed_weight");
    check_tensor(qkv_scale, "qkv_scale");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(qkv_output, "qkv_output");
    TORCH_CHECK(down_packed_weight.scalar_type() == at::kInt, "down_packed_weight must be int32");
    TORCH_CHECK(down_scale.scalar_type() == at::kBFloat16, "down_scale must be BF16");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(qkv_packed_weight.scalar_type() == at::kInt, "qkv_packed_weight must be int32");
    TORCH_CHECK(qkv_scale.scalar_type() == at::kBFloat16, "qkv_scale must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(qkv_output.scalar_type() == at::kBFloat16, "qkv_output must be BF16");
    TORCH_CHECK(
        down_packed_weight.dim() == 2 && down_packed_weight.size(0) == kHidden && down_packed_weight.size(1) == kDownPackedK,
        "down_packed_weight must be [2048, 1536]");
    TORCH_CHECK(
        qkv_packed_weight.dim() == 2 && qkv_packed_weight.size(0) == kQkv && qkv_packed_weight.size(1) == kQkvPackedK,
        "qkv_packed_weight must be [4096, 512]");
    TORCH_CHECK(input.numel() == kIntermediate, "input must have intermediate size 6144");
    TORCH_CHECK(residual.numel() == kHidden && residual_out.numel() == kHidden, "residual tensors must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(qkv_output.numel() == kQkv, "qkv_output must have qkv size 4096");
    TORCH_CHECK(down_scale.numel() >= kHidden, "down_scale is too small");
    TORCH_CHECK(qkv_scale.numel() >= kQkv, "qkv_scale is too small");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq is too small");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));

    if (rows_per_block == 2) {
        const int down_grid = (kHidden + 1) / 2;
        const int qkv_grid = (kQkv + 1) / 2;
        down_add_sumsq_q8_v2_warp_kernel<2><<<down_grid, 64, 0, stream>>>(
            down_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(down_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
        norm_qkv_q8_v2_kernel<2><<<qkv_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>(),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    } else if (rows_per_block == 8) {
        const int down_grid = (kHidden + 7) / 8;
        const int qkv_grid = (kQkv + 7) / 8;
        down_add_sumsq_q8_v2_warp_kernel<8><<<down_grid, 256, 0, stream>>>(
            down_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(down_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
        norm_qkv_q8_v2_kernel<8><<<qkv_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>(),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    } else {
        const int down_grid = (kHidden + 3) / 4;
        const int qkv_grid = (kQkv + 3) / 4;
        down_add_sumsq_q8_v2_warp_kernel<4><<<down_grid, 128, 0, stream>>>(
            down_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(down_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
        norm_qkv_q8_v2_kernel<4><<<qkv_grid, kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>(),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    }
}

void gate_up_q8_v4_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block) {
    check_tensor(residual, "residual");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(packed_weight, "packed_weight");
    check_tensor(scale, "scale");
    check_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(output.numel() == kIntermediate, "output must have intermediate size 6144");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == 2 * kIntermediate && packed_weight.size(1) == kGatePackedK,
                "packed_weight must be [12288, 512]");
    TORCH_CHECK(scale.numel() >= 2 * kIntermediate, "scale is too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32) ? warps_per_block : 8;
    const int grid = (kIntermediate + warps - 1) / warps;
    auto launch = [&](auto wpb) {
        constexpr int W = decltype(wpb)::value;
        gate_up_q8_v4_kernel<W><<<grid, W * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    };
    if (warps == 4) {
        launch(std::integral_constant<int, 4>{});
    } else if (warps == 16) {
        launch(std::integral_constant<int, 16>{});
    } else if (warps == 32) {
        launch(std::integral_constant<int, 32>{});
    } else {
        launch(std::integral_constant<int, 8>{});
    }
}

void norm_qkv_sumsq_q8_v4_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor sumsq,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output,
    double eps,
    int warps_per_block) {
    check_tensor(residual, "residual");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(sumsq, "sumsq");
    check_tensor(packed_weight, "packed_weight");
    check_tensor(scale, "scale");
    check_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq is too small");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kQkv && packed_weight.size(1) == kQkvPackedK,
                "packed_weight must be [4096, 512]");
    TORCH_CHECK(scale.numel() >= kQkv, "scale is too small");
    TORCH_CHECK(output.numel() == kQkv, "output must have qkv size 4096");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16 || warps_per_block == 32) ? warps_per_block : 8;
    const int grid = (kQkv + warps - 1) / warps;
    auto launch = [&](auto wpb) {
        constexpr int W = decltype(wpb)::value;
        norm_qkv_sumsq_q8_v4_kernel<W><<<grid, W * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>(),
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
    };
    if (warps == 4) {
        launch(std::integral_constant<int, 4>{});
    } else if (warps == 16) {
        launch(std::integral_constant<int, 16>{});
    } else if (warps == 32) {
        launch(std::integral_constant<int, 32>{});
    } else {
        launch(std::integral_constant<int, 8>{});
    }
}

void down_add_sumsq_q8_v4_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block) {
    check_tensor(packed_weight, "packed_weight");
    check_tensor(scale, "scale");
    check_tensor(input, "input");
    check_tensor(residual, "residual");
    check_tensor(residual_out, "residual_out");
    check_tensor(sumsq, "sumsq");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kHidden && packed_weight.size(1) == kDownPackedK,
                "packed_weight must be [2048, 1536]");
    TORCH_CHECK(input.numel() == kIntermediate, "input must have intermediate size 6144");
    TORCH_CHECK(residual.numel() == kHidden && residual_out.numel() == kHidden, "residual tensors must have hidden size 2048");
    TORCH_CHECK(scale.numel() >= kHidden, "scale is too small");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq is too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));
    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16) ? warps_per_block : 8;
    const int grid = (kHidden + warps - 1) / warps;
    auto launch = [&](auto wpb) {
        constexpr int W = decltype(wpb)::value;
        down_add_sumsq_q8_v4_kernel<W><<<grid, W * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    };
    if (warps == 4) {
        launch(std::integral_constant<int, 4>{});
    } else if (warps == 16) {
        launch(std::integral_constant<int, 16>{});
    } else {
        launch(std::integral_constant<int, 8>{});
    }
}

void down_add_sumsq_w4_v5_cuda(
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    int warps_per_block) {
    check_tensor(packed_weight, "packed_weight");
    check_tensor(scale, "scale");
    check_tensor(input, "input");
    check_tensor(residual, "residual");
    check_tensor(residual_out, "residual_out");
    check_tensor(sumsq, "sumsq");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(0) == kHidden && packed_weight.size(1) == kDownPackedK4,
                "packed_weight must be [2048, 768]");
    TORCH_CHECK(input.numel() == kIntermediate, "input must have intermediate size 6144");
    TORCH_CHECK(residual.numel() == kHidden && residual_out.numel() == kHidden, "residual tensors must have hidden size 2048");
    TORCH_CHECK(scale.numel() >= kHidden, "scale is too small");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq is too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));
    const int warps = (warps_per_block == 4 || warps_per_block == 8 || warps_per_block == 16) ? warps_per_block : 8;
    const int grid = (kHidden + warps - 1) / warps;
    auto launch = [&](auto wpb) {
        constexpr int W = decltype(wpb)::value;
        down_add_sumsq_w4_v5_kernel<W><<<grid, W * 32, 0, stream>>>(
            packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
    };
    if (warps == 4) {
        launch(std::integral_constant<int, 4>{});
    } else if (warps == 16) {
        launch(std::integral_constant<int, 16>{});
    } else {
        launch(std::integral_constant<int, 8>{});
    }
}

void down_norm_qkv_q8_v3_cuda(
    torch::Tensor down_packed_weight,
    torch::Tensor down_scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor normed_hidden,
    torch::Tensor qkv_packed_weight,
    torch::Tensor qkv_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps,
    int rows_per_block) {
    check_tensor(down_packed_weight, "down_packed_weight");
    check_tensor(down_scale, "down_scale");
    check_tensor(input, "input");
    check_tensor(residual, "residual");
    check_tensor(residual_out, "residual_out");
    check_tensor(sumsq, "sumsq");
    check_tensor(normed_hidden, "normed_hidden");
    check_tensor(qkv_packed_weight, "qkv_packed_weight");
    check_tensor(qkv_scale, "qkv_scale");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(qkv_output, "qkv_output");
    TORCH_CHECK(down_packed_weight.scalar_type() == at::kInt, "down_packed_weight must be int32");
    TORCH_CHECK(down_scale.scalar_type() == at::kBFloat16, "down_scale must be BF16");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "input must be BF16");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be BF16");
    TORCH_CHECK(sumsq.scalar_type() == at::kFloat, "sumsq must be FP32");
    TORCH_CHECK(normed_hidden.scalar_type() == at::kBFloat16, "normed_hidden must be BF16");
    TORCH_CHECK(qkv_packed_weight.scalar_type() == at::kInt, "qkv_packed_weight must be int32");
    TORCH_CHECK(qkv_scale.scalar_type() == at::kBFloat16, "qkv_scale must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(qkv_output.scalar_type() == at::kBFloat16, "qkv_output must be BF16");
    TORCH_CHECK(
        down_packed_weight.dim() == 2 && down_packed_weight.size(0) == kHidden && down_packed_weight.size(1) == kDownPackedK,
        "down_packed_weight must be [2048, 1536]");
    TORCH_CHECK(
        qkv_packed_weight.dim() == 2 && qkv_packed_weight.size(0) == kQkv && qkv_packed_weight.size(1) == kQkvPackedK,
        "qkv_packed_weight must be [4096, 512]");
    TORCH_CHECK(input.numel() == kIntermediate, "input must have intermediate size 6144");
    TORCH_CHECK(residual.numel() == kHidden && residual_out.numel() == kHidden, "residual tensors must have hidden size 2048");
    TORCH_CHECK(normed_hidden.numel() == kHidden, "normed_hidden must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(qkv_output.numel() == kQkv, "qkv_output must have qkv size 4096");
    TORCH_CHECK(down_scale.numel() >= kHidden, "down_scale is too small");
    TORCH_CHECK(qkv_scale.numel() >= kQkv, "qkv_scale is too small");
    TORCH_CHECK(sumsq.numel() >= 1, "sumsq is too small");

    auto stream = at::cuda::getCurrentCUDAStream();
    auto status = cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    TORCH_CHECK(status == cudaSuccess, "cudaMemsetAsync failed: ", cudaGetErrorString(status));

    if (rows_per_block == 2) {
        const int down_grid = (kHidden + 1) / 2;
        const int qkv_grid = (kQkv + 1) / 2;
        down_add_sumsq_q8_v2_warp_kernel<2><<<down_grid, 64, 0, stream>>>(
            down_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(down_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
        rms_norm_store_from_sumsq_bf16_kernel<<<1, kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
        norm_qkv_q8_v3_warp_kernel<2><<<qkv_grid, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()));
    } else if (rows_per_block == 8) {
        const int down_grid = (kHidden + 7) / 8;
        const int qkv_grid = (kQkv + 7) / 8;
        down_add_sumsq_q8_v2_warp_kernel<8><<<down_grid, 256, 0, stream>>>(
            down_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(down_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
        rms_norm_store_from_sumsq_bf16_kernel<<<1, kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
        norm_qkv_q8_v3_warp_kernel<8><<<qkv_grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()));
    } else {
        const int down_grid = (kHidden + 3) / 4;
        const int qkv_grid = (kQkv + 3) / 4;
        down_add_sumsq_q8_v2_warp_kernel<4><<<down_grid, 128, 0, stream>>>(
            down_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(down_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>());
        rms_norm_store_from_sumsq_bf16_kernel<<<1, kThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(residual_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
            sumsq.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            static_cast<float>(eps));
        norm_qkv_q8_v3_warp_kernel<4><<<qkv_grid, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(normed_hidden.data_ptr<at::BFloat16>()),
            qkv_packed_weight.data_ptr<int>(),
            reinterpret_cast<const __nv_bfloat16*>(qkv_scale.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(qkv_output.data_ptr<at::BFloat16>()));
    }
}
