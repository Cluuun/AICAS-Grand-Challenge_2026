#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kQuantThreads = 1024;
constexpr int kDownThreads = 256;

template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr, int idx) {
    return static_cast<float>(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half* ptr, int idx) {
    return __half2float(ptr[idx]);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* ptr, int idx, float value) {
    ptr[idx] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* ptr, int idx, float value) {
    ptr[idx] = __float2bfloat16(value);
}

template <>
__device__ __forceinline__ void store_from_float<half>(half* ptr, int idx, float value) {
    ptr[idx] = __float2half(value);
}

__device__ __forceinline__ int8_t quantize_i8(float value, float inv_scale) {
    const float q = nearbyintf(value * inv_scale);
    const float clamped = fminf(127.0f, fmaxf(-128.0f, q));
    return static_cast<int8_t>(clamped);
}

__device__ __forceinline__ int pack_i8x4(int8_t x0, int8_t x1, int8_t x2, int8_t x3) {
    return (static_cast<int>(static_cast<unsigned char>(x0)))
        | (static_cast<int>(static_cast<unsigned char>(x1)) << 8)
        | (static_cast<int>(static_cast<unsigned char>(x2)) << 16)
        | (static_cast<int>(static_cast<unsigned char>(x3)) << 24);
}

template <typename InT>
__global__ void max_abs_scale_kernel(
    const InT* input,
    float* act_scale,
    int n) {
    __shared__ float shared[kQuantThreads];
    const int tid = threadIdx.x;
    float local_max = 0.0f;

    for (int idx = tid; idx < n; idx += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(load_as_float(input, idx)));
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float max_abs = shared[0];
        act_scale[0] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    }
}

template <typename InT>
__global__ void quantize_bf16_like_i8_kernel(
    const InT* input,
    int8_t* act_i8,
    float* act_scale,
    int n) {
    __shared__ float shared[kQuantThreads];
    const int tid = threadIdx.x;
    float local_max = 0.0f;

    for (int idx = tid; idx < n; idx += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(load_as_float(input, idx)));
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    const float max_abs = shared[0];
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    const float inv_scale = 1.0f / scale;
    if (tid == 0) {
        act_scale[0] = scale;
    }
    __syncthreads();

    for (int idx = tid; idx < n; idx += blockDim.x) {
        act_i8[idx] = quantize_i8(load_as_float(input, idx), inv_scale);
    }
}

template <typename InT>
__global__ void quantize_bf16_like_i8_lagged_kernel(
    const InT* input,
    int8_t* act_i8,
    float* act_scale,
    int n) {
    __shared__ float shared[kQuantThreads];
    const int tid = threadIdx.x;
    const float prev_scale = fmaxf(act_scale[0], 1.0e-8f);
    const float inv_prev_scale = 1.0f / prev_scale;
    float local_max = 0.0f;

    for (int idx = tid; idx < n; idx += blockDim.x) {
        const float value = load_as_float(input, idx);
        local_max = fmaxf(local_max, fabsf(value));
        act_i8[idx] = quantize_i8(value, inv_prev_scale);
    }
    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float max_abs = shared[0];
        act_scale[0] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    }
}

template <typename InT>
__global__ void quantize_bf16_like_i8_fixed_kernel(
    const InT* input,
    int8_t* act_i8,
    const float* act_scale,
    int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float inv_scale = 1.0f / fmaxf(act_scale[0], 1.0e-8f);
    for (int idx = tid; idx < n; idx += blockDim.x * gridDim.x) {
        act_i8[idx] = quantize_i8(load_as_float(input, idx), inv_scale);
    }
}

template <typename InT, typename ScaleT, typename ResidualT, int RowsPerBlock>
__global__ void down_add_sumsq_w8a8_onthefly_kernel(
    const int8_t* __restrict__ weight,
    const ScaleT* __restrict__ weight_scale,
    const InT* __restrict__ input,
    const float* __restrict__ act_scale,
    const ResidualT* __restrict__ residual,
    ResidualT* __restrict__ residual_out,
    float* __restrict__ sumsq,
    int rows,
    int cols) {
    __shared__ int partial[RowsPerBlock][kDownThreads];
    const int tid = threadIdx.x;
    const int row_base = blockIdx.x * RowsPerBlock;
    const int cols4 = cols / 4;
    const int tail_start = cols4 * 4;
    const float inv_scale = 1.0f / act_scale[0];
    int acc[RowsPerBlock];
    #pragma unroll
    for (int r = 0; r < RowsPerBlock; ++r) {
        acc[r] = 0;
    }

    for (int col4 = tid; col4 < cols4; col4 += blockDim.x) {
        const int col = col4 * 4;
        const int x4 = pack_i8x4(
            quantize_i8(load_as_float(input, col), inv_scale),
            quantize_i8(load_as_float(input, col + 1), inv_scale),
            quantize_i8(load_as_float(input, col + 2), inv_scale),
            quantize_i8(load_as_float(input, col + 3), inv_scale));
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                const int* __restrict__ weight_i8x4 = reinterpret_cast<const int*>(weight + row * cols);
                acc[r] = __dp4a(weight_i8x4[col4], x4, acc[r]);
            }
        }
    }

    for (int col = tail_start + tid; col < cols; col += blockDim.x) {
        const int x = static_cast<int>(quantize_i8(load_as_float(input, col), inv_scale));
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                acc[r] += static_cast<int>(weight[row * cols + col]) * x;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < RowsPerBlock; ++r) {
        partial[r][tid] = acc[r];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int r = 0; r < RowsPerBlock; ++r) {
                partial[r][tid] += partial[r][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float block_sumsq = 0.0f;
        const float a_scale = act_scale[0];
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                const float w_scale = load_as_float(weight_scale, row);
                const float down = static_cast<float>(partial[r][0]) * a_scale * w_scale;
                const float out = down + load_as_float(residual, row);
                store_from_float(residual_out, row, out);
                block_sumsq += out * out;
            }
        }
        atomicAdd(sumsq, block_sumsq);
    }
}

template <typename ScaleT, typename ResidualT, int RowsPerBlock>
__global__ void down_add_sumsq_w8a8_prequant_kernel(
    const int8_t* __restrict__ weight,
    const ScaleT* __restrict__ weight_scale,
    const int8_t* __restrict__ act_i8,
    const float* __restrict__ act_scale,
    const ResidualT* __restrict__ residual,
    ResidualT* __restrict__ residual_out,
    float* __restrict__ sumsq,
    int rows,
    int cols) {
    __shared__ int partial[RowsPerBlock][kDownThreads];
    const int tid = threadIdx.x;
    const int row_base = blockIdx.x * RowsPerBlock;
    const int cols4 = cols / 4;
    const int tail_start = cols4 * 4;
    const int* __restrict__ act_i8x4 = reinterpret_cast<const int*>(act_i8);
    int acc[RowsPerBlock];
    #pragma unroll
    for (int r = 0; r < RowsPerBlock; ++r) {
        acc[r] = 0;
    }

    for (int col4 = tid; col4 < cols4; col4 += blockDim.x) {
        const int x4 = act_i8x4[col4];
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                const int* __restrict__ weight_i8x4 = reinterpret_cast<const int*>(weight + row * cols);
                acc[r] = __dp4a(weight_i8x4[col4], x4, acc[r]);
            }
        }
    }

    for (int col = tail_start + tid; col < cols; col += blockDim.x) {
        const int x = static_cast<int>(act_i8[col]);
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                acc[r] += static_cast<int>(weight[row * cols + col]) * x;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < RowsPerBlock; ++r) {
        partial[r][tid] = acc[r];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int r = 0; r < RowsPerBlock; ++r) {
                partial[r][tid] += partial[r][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float block_sumsq = 0.0f;
        const float a_scale = act_scale[0];
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                const float w_scale = load_as_float(weight_scale, row);
                const float down = static_cast<float>(partial[r][0]) * a_scale * w_scale;
                const float out = down + load_as_float(residual, row);
                store_from_float(residual_out, row, out);
                block_sumsq += out * out;
            }
        }
        atomicAdd(sumsq, block_sumsq);
    }
}

template <typename ScaleT, typename ResidualT, int RowsPerBlock>
void launch_down_prequant(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& act_i8,
    const torch::Tensor& act_scale,
    const torch::Tensor& residual,
    const torch::Tensor& residual_out,
    const torch::Tensor& sumsq,
    cudaStream_t stream) {
    const int rows = static_cast<int>(weight.size(0));
    const int cols = static_cast<int>(weight.size(1));
    const int blocks = (rows + RowsPerBlock - 1) / RowsPerBlock;
    cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    down_add_sumsq_w8a8_prequant_kernel<ScaleT, ResidualT, RowsPerBlock><<<blocks, kDownThreads, 0, stream>>>(
        weight.data_ptr<int8_t>(),
        reinterpret_cast<const ScaleT*>(weight_scale.data_ptr()),
        act_i8.data_ptr<int8_t>(),
        act_scale.data_ptr<float>(),
        reinterpret_cast<const ResidualT*>(residual.data_ptr()),
        reinterpret_cast<ResidualT*>(residual_out.data_ptr()),
        sumsq.data_ptr<float>(),
        rows,
        cols);
}

template <typename ResidualT, int RowsPerBlock>
void dispatch_scale_down_prequant(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& act_i8,
    const torch::Tensor& act_scale,
    const torch::Tensor& residual,
    const torch::Tensor& residual_out,
    const torch::Tensor& sumsq,
    cudaStream_t stream) {
    if (weight_scale.scalar_type() == torch::kBFloat16) {
        launch_down_prequant<__nv_bfloat16, ResidualT, RowsPerBlock>(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq, stream);
    } else if (weight_scale.scalar_type() == torch::kFloat16) {
        launch_down_prequant<half, ResidualT, RowsPerBlock>(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq, stream);
    } else if (weight_scale.scalar_type() == torch::kFloat32) {
        launch_down_prequant<float, ResidualT, RowsPerBlock>(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq, stream);
    } else {
        TORCH_CHECK(false, "weight_scale must be bf16, fp16, or fp32");
    }
}

template <typename ScaleT, typename ResidualT, typename NormT, typename OutT, int RowsPerBlock>
__global__ void norm_qkv_w8a8_onthefly_kernel(
    const int8_t* __restrict__ weight,
    const ScaleT* __restrict__ weight_scale,
    const ResidualT* __restrict__ residual,
    const float* __restrict__ sumsq,
    const NormT* __restrict__ norm_weight,
    OutT* __restrict__ output,
    float eps,
    int rows,
    int cols) {
    __shared__ int partial[RowsPerBlock][kDownThreads];
    __shared__ float shared_max[kDownThreads];
    __shared__ float rstd_shared;
    __shared__ float act_scale_shared;
    __shared__ float inv_act_scale_shared;

    const int tid = threadIdx.x;
    const int row_base = blockIdx.x * RowsPerBlock;
    const int cols4 = cols / 4;
    const int tail_start = cols4 * 4;

    if (tid == 0) {
        rstd_shared = rsqrtf(sumsq[0] / static_cast<float>(cols) + eps);
    }
    __syncthreads();

    float local_max = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        const float normed = load_as_float(residual, col) * load_as_float(norm_weight, col) * rstd_shared;
        local_max = fmaxf(local_max, fabsf(normed));
    }
    shared_max[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float max_abs = shared_max[0];
        const float act_scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        act_scale_shared = act_scale;
        inv_act_scale_shared = 1.0f / act_scale;
    }
    __syncthreads();

    int acc[RowsPerBlock];
    #pragma unroll
    for (int r = 0; r < RowsPerBlock; ++r) {
        acc[r] = 0;
    }

    for (int col4 = tid; col4 < cols4; col4 += blockDim.x) {
        const int col = col4 * 4;
        const int x4 = pack_i8x4(
            quantize_i8(load_as_float(residual, col) * load_as_float(norm_weight, col) * rstd_shared, inv_act_scale_shared),
            quantize_i8(load_as_float(residual, col + 1) * load_as_float(norm_weight, col + 1) * rstd_shared, inv_act_scale_shared),
            quantize_i8(load_as_float(residual, col + 2) * load_as_float(norm_weight, col + 2) * rstd_shared, inv_act_scale_shared),
            quantize_i8(load_as_float(residual, col + 3) * load_as_float(norm_weight, col + 3) * rstd_shared, inv_act_scale_shared));
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                const int* __restrict__ weight_i8x4 = reinterpret_cast<const int*>(weight + row * cols);
                acc[r] = __dp4a(weight_i8x4[col4], x4, acc[r]);
            }
        }
    }

    for (int col = tail_start + tid; col < cols; col += blockDim.x) {
        const int x = static_cast<int>(quantize_i8(
            load_as_float(residual, col) * load_as_float(norm_weight, col) * rstd_shared,
            inv_act_scale_shared));
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                acc[r] += static_cast<int>(weight[row * cols + col]) * x;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < RowsPerBlock; ++r) {
        partial[r][tid] = acc[r];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int r = 0; r < RowsPerBlock; ++r) {
                partial[r][tid] += partial[r][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float a_scale = act_scale_shared;
        #pragma unroll
        for (int r = 0; r < RowsPerBlock; ++r) {
            const int row = row_base + r;
            if (row < rows) {
                const float w_scale = load_as_float(weight_scale, row);
                store_from_float(output, row, static_cast<float>(partial[r][0]) * a_scale * w_scale);
            }
        }
    }
}

template <typename ScaleT, typename ResidualT, typename NormT, typename OutT, int RowsPerBlock>
void launch_norm_qkv_onthefly(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& residual,
    const torch::Tensor& sumsq,
    const torch::Tensor& norm_weight,
    const torch::Tensor& output,
    double eps,
    cudaStream_t stream) {
    const int rows = static_cast<int>(weight.size(0));
    const int cols = static_cast<int>(weight.size(1));
    const int blocks = (rows + RowsPerBlock - 1) / RowsPerBlock;
    norm_qkv_w8a8_onthefly_kernel<ScaleT, ResidualT, NormT, OutT, RowsPerBlock><<<blocks, kDownThreads, 0, stream>>>(
        weight.data_ptr<int8_t>(),
        reinterpret_cast<const ScaleT*>(weight_scale.data_ptr()),
        reinterpret_cast<const ResidualT*>(residual.data_ptr()),
        sumsq.data_ptr<float>(),
        reinterpret_cast<const NormT*>(norm_weight.data_ptr()),
        reinterpret_cast<OutT*>(output.data_ptr()),
        static_cast<float>(eps),
        rows,
        cols);
}

template <typename ResidualT, typename NormT, typename OutT, int RowsPerBlock>
void dispatch_scale_norm_qkv_onthefly(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& residual,
    const torch::Tensor& sumsq,
    const torch::Tensor& norm_weight,
    const torch::Tensor& output,
    double eps,
    cudaStream_t stream) {
    if (weight_scale.scalar_type() == torch::kBFloat16) {
        launch_norm_qkv_onthefly<__nv_bfloat16, ResidualT, NormT, OutT, RowsPerBlock>(
            weight, weight_scale, residual, sumsq, norm_weight, output, eps, stream);
    } else if (weight_scale.scalar_type() == torch::kFloat16) {
        launch_norm_qkv_onthefly<half, ResidualT, NormT, OutT, RowsPerBlock>(
            weight, weight_scale, residual, sumsq, norm_weight, output, eps, stream);
    } else if (weight_scale.scalar_type() == torch::kFloat32) {
        launch_norm_qkv_onthefly<float, ResidualT, NormT, OutT, RowsPerBlock>(
            weight, weight_scale, residual, sumsq, norm_weight, output, eps, stream);
    } else {
        TORCH_CHECK(false, "weight_scale must be bf16, fp16, or fp32");
    }
}

template <typename NormT, typename OutT, int RowsPerBlock>
void dispatch_residual_norm_qkv_onthefly(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& residual,
    const torch::Tensor& sumsq,
    const torch::Tensor& norm_weight,
    const torch::Tensor& output,
    double eps,
    cudaStream_t stream) {
    if (residual.scalar_type() == torch::kBFloat16) {
        dispatch_scale_norm_qkv_onthefly<__nv_bfloat16, NormT, OutT, RowsPerBlock>(
            weight, weight_scale, residual, sumsq, norm_weight, output, eps, stream);
    } else if (residual.scalar_type() == torch::kFloat16) {
        dispatch_scale_norm_qkv_onthefly<half, NormT, OutT, RowsPerBlock>(
            weight, weight_scale, residual, sumsq, norm_weight, output, eps, stream);
    } else {
        TORCH_CHECK(false, "residual must be bf16 or fp16");
    }
}

template <typename InT, typename ScaleT, typename ResidualT, int RowsPerBlock>
void launch_down_onthefly(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& residual_out,
    const torch::Tensor& sumsq,
    const torch::Tensor& act_scale,
    cudaStream_t stream) {
    const int rows = static_cast<int>(weight.size(0));
    const int cols = static_cast<int>(weight.size(1));
    const int blocks = (rows + RowsPerBlock - 1) / RowsPerBlock;
    max_abs_scale_kernel<InT><<<1, kQuantThreads, 0, stream>>>(
        reinterpret_cast<const InT*>(input.data_ptr()),
        act_scale.data_ptr<float>(),
        cols);
    cudaMemsetAsync(sumsq.data_ptr<float>(), 0, sizeof(float), stream);
    down_add_sumsq_w8a8_onthefly_kernel<InT, ScaleT, ResidualT, RowsPerBlock><<<blocks, kDownThreads, 0, stream>>>(
        weight.data_ptr<int8_t>(),
        reinterpret_cast<const ScaleT*>(weight_scale.data_ptr()),
        reinterpret_cast<const InT*>(input.data_ptr()),
        act_scale.data_ptr<float>(),
        reinterpret_cast<const ResidualT*>(residual.data_ptr()),
        reinterpret_cast<ResidualT*>(residual_out.data_ptr()),
        sumsq.data_ptr<float>(),
        rows,
        cols);
}

template <typename InT, typename ResidualT, int RowsPerBlock>
void dispatch_scale_down_onthefly(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& residual_out,
    const torch::Tensor& sumsq,
    const torch::Tensor& act_scale,
    cudaStream_t stream) {
    if (weight_scale.scalar_type() == torch::kBFloat16) {
        launch_down_onthefly<InT, __nv_bfloat16, ResidualT, RowsPerBlock>(
            weight, weight_scale, input, residual, residual_out, sumsq, act_scale, stream);
    } else if (weight_scale.scalar_type() == torch::kFloat16) {
        launch_down_onthefly<InT, half, ResidualT, RowsPerBlock>(
            weight, weight_scale, input, residual, residual_out, sumsq, act_scale, stream);
    } else if (weight_scale.scalar_type() == torch::kFloat32) {
        launch_down_onthefly<InT, float, ResidualT, RowsPerBlock>(
            weight, weight_scale, input, residual, residual_out, sumsq, act_scale, stream);
    } else {
        TORCH_CHECK(false, "weight_scale must be bf16, fp16, or fp32");
    }
}

template <typename InT, int RowsPerBlock>
void dispatch_residual_down_onthefly(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& residual_out,
    const torch::Tensor& sumsq,
    const torch::Tensor& act_scale,
    cudaStream_t stream) {
    if (residual.scalar_type() == torch::kBFloat16) {
        dispatch_scale_down_onthefly<InT, __nv_bfloat16, RowsPerBlock>(
            weight, weight_scale, input, residual, residual_out, sumsq, act_scale, stream);
    } else if (residual.scalar_type() == torch::kFloat16) {
        dispatch_scale_down_onthefly<InT, half, RowsPerBlock>(
            weight, weight_scale, input, residual, residual_out, sumsq, act_scale, stream);
    } else {
        TORCH_CHECK(false, "residual must be bf16 or fp16");
    }
}

void validate_down_args(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& act_i8,
    const torch::Tensor& act_scale,
    const torch::Tensor& residual,
    const torch::Tensor& residual_out,
    const torch::Tensor& sumsq) {
    TORCH_CHECK(weight.is_cuda() && weight_scale.is_cuda() && act_i8.is_cuda() && act_scale.is_cuda()
        && residual.is_cuda() && residual_out.is_cuda() && sumsq.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
    TORCH_CHECK(act_i8.scalar_type() == torch::kInt8, "act_i8 must be int8");
    TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32, "act_scale must be fp32");
    TORCH_CHECK(sumsq.scalar_type() == torch::kFloat32 && sumsq.numel() >= 1, "sumsq must be fp32 with at least one element");
    TORCH_CHECK(weight.dim() == 2, "weight must be [out, in]");
    TORCH_CHECK(weight_scale.numel() >= weight.size(0), "weight_scale must have at least one value per output row");
    TORCH_CHECK(act_i8.numel() >= weight.size(1), "act_i8 is too small");
    TORCH_CHECK(residual.numel() >= weight.size(0) && residual_out.numel() >= weight.size(0), "residual shape mismatch");
    TORCH_CHECK(residual.scalar_type() == residual_out.scalar_type(), "residual and residual_out dtype mismatch");
}

void validate_down_onthefly_args(
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& residual_out,
    const torch::Tensor& sumsq,
    const torch::Tensor& act_scale) {
    TORCH_CHECK(weight.is_cuda() && weight_scale.is_cuda() && input.is_cuda() && act_scale.is_cuda()
        && residual.is_cuda() && residual_out.is_cuda() && sumsq.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(weight.scalar_type() == torch::kInt8, "weight must be int8");
    TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32, "act_scale must be fp32");
    TORCH_CHECK(sumsq.scalar_type() == torch::kFloat32 && sumsq.numel() >= 1, "sumsq must be fp32 with at least one element");
    TORCH_CHECK(weight.dim() == 2, "weight must be [out, in]");
    TORCH_CHECK(input.numel() == weight.size(1), "input length must match weight input dimension");
    TORCH_CHECK(weight_scale.numel() >= weight.size(0), "weight_scale must have at least one value per output row");
    TORCH_CHECK(residual.numel() >= weight.size(0) && residual_out.numel() >= weight.size(0), "residual shape mismatch");
    TORCH_CHECK(residual.scalar_type() == residual_out.scalar_type(), "residual and residual_out dtype mismatch");
}

void validate_down_norm_qkv_args(
    const torch::Tensor& down_weight,
    const torch::Tensor& down_weight_scale,
    const torch::Tensor& act_i8,
    const torch::Tensor& act_scale,
    const torch::Tensor& residual,
    const torch::Tensor& residual_out,
    const torch::Tensor& sumsq,
    const torch::Tensor& qkv_weight,
    const torch::Tensor& qkv_weight_scale,
    const torch::Tensor& norm_weight,
    const torch::Tensor& qkv_output) {
    validate_down_args(down_weight, down_weight_scale, act_i8, act_scale, residual, residual_out, sumsq);
    TORCH_CHECK(qkv_weight.is_cuda() && qkv_weight_scale.is_cuda() && norm_weight.is_cuda() && qkv_output.is_cuda(),
        "qkv tensors must be CUDA");
    TORCH_CHECK(qkv_weight.scalar_type() == torch::kInt8, "qkv_weight must be int8");
    TORCH_CHECK(qkv_weight.dim() == 2, "qkv_weight must be [out, in]");
    TORCH_CHECK(qkv_weight.size(1) == down_weight.size(0), "qkv_weight input dim must match residual width");
    TORCH_CHECK(qkv_weight_scale.numel() >= qkv_weight.size(0), "qkv_weight_scale must have at least one value per output row");
    TORCH_CHECK(norm_weight.numel() >= qkv_weight.size(1), "norm_weight shape mismatch");
    TORCH_CHECK(norm_weight.scalar_type() == residual_out.scalar_type(), "norm_weight dtype mismatch");
    TORCH_CHECK(qkv_output.numel() >= qkv_weight.size(0), "qkv_output shape mismatch");
}

template <int RowsPerBlock>
torch::Tensor down_add_sumsq_w8a8_prequant_impl(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq) {
    validate_down_args(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (residual.scalar_type() == torch::kBFloat16) {
        dispatch_scale_down_prequant<__nv_bfloat16, RowsPerBlock>(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq, stream);
    } else if (residual.scalar_type() == torch::kFloat16) {
        dispatch_scale_down_prequant<half, RowsPerBlock>(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq, stream);
    } else {
        TORCH_CHECK(false, "residual must be bf16 or fp16");
    }
    return residual_out;
}

template <int RowsPerBlock>
torch::Tensor down_norm_qkv_w8a8_prequant_impl(
    torch::Tensor down_weight,
    torch::Tensor down_weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor qkv_weight,
    torch::Tensor qkv_weight_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps) {
    validate_down_norm_qkv_args(
        down_weight,
        down_weight_scale,
        act_i8,
        act_scale,
        residual,
        residual_out,
        sumsq,
        qkv_weight,
        qkv_weight_scale,
        norm_weight,
        qkv_output);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (residual.scalar_type() == torch::kBFloat16) {
        dispatch_scale_down_prequant<__nv_bfloat16, RowsPerBlock>(
            down_weight, down_weight_scale, act_i8, act_scale, residual, residual_out, sumsq, stream);
        if (norm_weight.scalar_type() == torch::kBFloat16 && qkv_output.scalar_type() == torch::kBFloat16) {
            dispatch_residual_norm_qkv_onthefly<__nv_bfloat16, __nv_bfloat16, RowsPerBlock>(
                qkv_weight, qkv_weight_scale, residual_out, sumsq, norm_weight, qkv_output, eps, stream);
        } else if (norm_weight.scalar_type() == torch::kFloat16 && qkv_output.scalar_type() == torch::kFloat16) {
            dispatch_residual_norm_qkv_onthefly<half, half, RowsPerBlock>(
                qkv_weight, qkv_weight_scale, residual_out, sumsq, norm_weight, qkv_output, eps, stream);
        } else {
            TORCH_CHECK(false, "bf16 residual path requires matching bf16/fp16 norm_weight and qkv_output");
        }
    } else if (residual.scalar_type() == torch::kFloat16) {
        dispatch_scale_down_prequant<half, RowsPerBlock>(
            down_weight, down_weight_scale, act_i8, act_scale, residual, residual_out, sumsq, stream);
        if (norm_weight.scalar_type() == torch::kFloat16 && qkv_output.scalar_type() == torch::kFloat16) {
            dispatch_residual_norm_qkv_onthefly<half, half, RowsPerBlock>(
                qkv_weight, qkv_weight_scale, residual_out, sumsq, norm_weight, qkv_output, eps, stream);
        } else if (norm_weight.scalar_type() == torch::kBFloat16 && qkv_output.scalar_type() == torch::kBFloat16) {
            dispatch_residual_norm_qkv_onthefly<__nv_bfloat16, __nv_bfloat16, RowsPerBlock>(
                qkv_weight, qkv_weight_scale, residual_out, sumsq, norm_weight, qkv_output, eps, stream);
        } else {
            TORCH_CHECK(false, "fp16 residual path requires matching bf16/fp16 norm_weight and qkv_output");
        }
    } else {
        TORCH_CHECK(false, "residual must be bf16 or fp16");
    }
    return qkv_output;
}

}  // namespace

torch::Tensor down_add_sumsq_w8a8_prequant_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq) {
    return down_add_sumsq_w8a8_prequant_impl<4>(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq);
}

torch::Tensor down_add_sumsq_w8a8_prequant_rpb2_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq) {
    return down_add_sumsq_w8a8_prequant_impl<2>(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq);
}

torch::Tensor down_add_sumsq_w8a8_prequant_rpb8_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq) {
    return down_add_sumsq_w8a8_prequant_impl<8>(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq);
}

torch::Tensor down_norm_qkv_w8a8_prequant_cuda(
    torch::Tensor down_weight,
    torch::Tensor down_weight_scale,
    torch::Tensor act_i8,
    torch::Tensor act_scale,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor qkv_weight,
    torch::Tensor qkv_weight_scale,
    torch::Tensor norm_weight,
    torch::Tensor qkv_output,
    double eps) {
    return down_norm_qkv_w8a8_prequant_impl<4>(
        down_weight,
        down_weight_scale,
        act_i8,
        act_scale,
        residual,
        residual_out,
        sumsq,
        qkv_weight,
        qkv_weight_scale,
        norm_weight,
        qkv_output,
        eps);
}

torch::Tensor down_add_sumsq_w8a8_onthefly_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor act_scale) {
    validate_down_onthefly_args(weight, weight_scale, input, residual, residual_out, sumsq, act_scale);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == torch::kBFloat16) {
        dispatch_residual_down_onthefly<__nv_bfloat16, 4>(
            weight, weight_scale, input, residual, residual_out, sumsq, act_scale, stream);
    } else if (input.scalar_type() == torch::kFloat16) {
        dispatch_residual_down_onthefly<half, 4>(
            weight, weight_scale, input, residual, residual_out, sumsq, act_scale, stream);
    } else {
        TORCH_CHECK(false, "input must be bf16 or fp16");
    }
    return residual_out;
}

torch::Tensor quantize_bf16_to_i8_lagged_cuda(
    torch::Tensor input,
    torch::Tensor act_i8,
    torch::Tensor act_scale) {
    TORCH_CHECK(input.is_cuda() && act_i8.is_cuda() && act_scale.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(act_i8.scalar_type() == torch::kInt8, "act_i8 must be int8");
    TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32 && act_scale.numel() >= 1, "act_scale must be fp32");
    TORCH_CHECK(act_i8.numel() >= input.numel(), "act_i8 scratch is too small");
    const int n = static_cast<int>(input.numel());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == torch::kBFloat16) {
        quantize_bf16_like_i8_lagged_kernel<__nv_bfloat16><<<1, kQuantThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            act_i8.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            n);
    } else if (input.scalar_type() == torch::kFloat16) {
        quantize_bf16_like_i8_lagged_kernel<half><<<1, kQuantThreads, 0, stream>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            act_i8.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            n);
    } else {
        TORCH_CHECK(false, "input must be bf16 or fp16");
    }
    return act_i8;
}

torch::Tensor quantize_bf16_to_i8_fixed_cuda(
    torch::Tensor input,
    torch::Tensor act_i8,
    torch::Tensor act_scale) {
    TORCH_CHECK(input.is_cuda() && act_i8.is_cuda() && act_scale.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(act_i8.scalar_type() == torch::kInt8, "act_i8 must be int8");
    TORCH_CHECK(act_scale.scalar_type() == torch::kFloat32 && act_scale.numel() >= 1, "act_scale must be fp32");
    TORCH_CHECK(act_i8.numel() >= input.numel(), "act_i8 scratch is too small");
    const int n = static_cast<int>(input.numel());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    if (input.scalar_type() == torch::kBFloat16) {
        quantize_bf16_like_i8_fixed_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            act_i8.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            n);
    } else if (input.scalar_type() == torch::kFloat16) {
        quantize_bf16_like_i8_fixed_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            act_i8.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            n);
    } else {
        TORCH_CHECK(false, "input must be bf16 or fp16");
    }
    return act_i8;
}

torch::Tensor down_add_sumsq_w8a8_dynamic_cuda(
    torch::Tensor weight,
    torch::Tensor weight_scale,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor residual_out,
    torch::Tensor sumsq,
    torch::Tensor act_i8,
    torch::Tensor act_scale) {
    validate_down_args(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq);
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.numel() == weight.size(1), "input length must match weight input dimension");
    TORCH_CHECK(act_i8.numel() >= input.numel(), "act_i8 scratch is too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int n = static_cast<int>(input.numel());
    if (input.scalar_type() == torch::kBFloat16) {
        quantize_bf16_like_i8_kernel<__nv_bfloat16><<<1, kQuantThreads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            act_i8.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            n);
    } else if (input.scalar_type() == torch::kFloat16) {
        quantize_bf16_like_i8_kernel<half><<<1, kQuantThreads, 0, stream>>>(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            act_i8.data_ptr<int8_t>(),
            act_scale.data_ptr<float>(),
            n);
    } else {
        TORCH_CHECK(false, "input must be bf16 or fp16");
    }
    return down_add_sumsq_w8a8_prequant_cuda(weight, weight_scale, act_i8, act_scale, residual, residual_out, sumsq);
}
