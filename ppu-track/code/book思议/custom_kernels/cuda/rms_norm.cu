/*
 * RMSNorm CUDA核心
 * warp级别规约算方差 每个block处理一行
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ weight,
    const int hidden_size,
    const float eps,
    const int num_rows) {

    const int row = blockIdx.x;
    if (row >= num_rows) return;

    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const scalar_t* row_in = input + row * hidden_size;
    scalar_t* row_out = output + row * hidden_size;

    // 算部分方差
    float partial_var = 0.0f;
    for (int i = tid; i < hidden_size; i += block_dim) {
        float val = static_cast<float>(row_in[i]);
        partial_var += val * val;
    }

    // warp级别规约
    partial_var = warp_reduce_sum<scalar_t>(partial_var);

    // block级别规约
    __shared__ float s_partial[32];
    if (lane_id == 0) {
        s_partial[warp_id] = partial_var;
    }
    __syncthreads();

    float total_var = 0.0f;
    int num_warps = (block_dim + 31) / 32;
    if (warp_id == 0) {
        total_var = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        total_var = warp_reduce_sum<scalar_t>(total_var);
    }

    // 广播rstd
    __shared__ float s_rstd;
    if (tid == 0) {
        s_rstd = 1.0f / sqrtf(total_var / hidden_size + eps);
    }
    __syncthreads();

    // 归一化再乘权重
    for (int i = tid; i < hidden_size; i += block_dim) {
        float val = static_cast<float>(row_in[i]);
        float w = static_cast<float>(weight[i]);
        row_out[i] = static_cast<scalar_t>(val * s_rstd * w);
    }
}


torch::Tensor rms_norm_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
    auto output = torch::empty_like(input);
    const int hidden_size = input.size(-1);
    const int num_rows = input.numel() / hidden_size;

    auto input_contig = input.reshape({num_rows, hidden_size}).contiguous();
    output = output.reshape({num_rows, hidden_size});

    const int threads = 256;
    const int blocks = num_rows;

    AT_DISPATCH_FLOAT_HALF_TYPES(
        input_contig.scalar_type(), "rms_norm_cuda", ([&] {
            rms_norm_kernel<scalar_t><<<blocks, threads>>>(
                input_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                hidden_size,
                static_cast<float>(eps),
                num_rows);
        })
    );

    C10_CUDA_CHECK(cudaGetLastError());
    return output.reshape(input.sizes());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_forward", &rms_norm_cuda, "RMSNorm前向(CUDA)");
}
