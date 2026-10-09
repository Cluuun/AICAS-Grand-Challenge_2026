#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {

constexpr int kArgmaxBlockSize = 256;

__device__ __forceinline__ void update_best(float other_value, int other_index, float& best_value, int& best_index) {
    if (other_value > best_value || (other_value == best_value && other_index < best_index)) {
        best_value = other_value;
        best_index = other_index;
    }
}

__global__ void argmax_fp16_stage1_kernel(
    const half* logits,
    float* partial_values,
    int* partial_indices,
    int n) {
    __shared__ float values[kArgmaxBlockSize];
    __shared__ int indices[kArgmaxBlockSize];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    float best_value = -INFINITY;
    int best_index = 0;

    if (idx < n) {
        best_value = __half2float(logits[idx]);
        best_index = idx;
    }

    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            update_best(values[tid + stride], indices[tid + stride], values[tid], indices[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_values[blockIdx.x] = values[0];
        partial_indices[blockIdx.x] = indices[0];
    }
}

__global__ void cast_bf16_to_fp16_kernel(const __nv_bfloat16* input, half* output, int count) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        output[idx] = __float2half(__bfloat162float(input[idx]));
    }
}


template <typename OutT>
__global__ void argmax_fp16_stage2_kernel(
    const float* partial_values,
    const int* partial_indices,
    OutT* token_out,
    int n_blocks) {
    __shared__ float values[1024];
    __shared__ int indices[1024];
    const int tid = threadIdx.x;
    float best_value = -INFINITY;
    int best_index = 0;

    if (tid < n_blocks) {
        best_value = partial_values[tid];
        best_index = partial_indices[tid];
    }

    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            update_best(values[tid + stride], indices[tid + stride], values[tid], indices[tid]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        token_out[0] = static_cast<OutT>(indices[0]);
    }
}

template <typename CosT, typename OutT>
__global__ void argmax_fp16_stage2_post_step_kernel(
    const float* partial_values,
    const int* partial_indices,
    OutT* token_out,
    const CosT* cos_table,
    const CosT* sin_table,
    CosT* cos_out,
    CosT* sin_out,
    int* cache_seqlens,
    int* cache_seqlens_next,
    const int* base_seqlen,
    int64_t* input_ids,
    int64_t* generated,
    int64_t* step_counter,
    int n_blocks,
    int head_dim) {
    __shared__ float values[1024];
    __shared__ int indices[1024];
    const int tid = threadIdx.x;
    float best_value = -INFINITY;
    int best_index = 0;

    if (tid < n_blocks) {
        best_value = partial_values[tid];
        best_index = partial_indices[tid];
    }

    values[tid] = best_value;
    indices[tid] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            update_best(values[tid + stride], indices[tid + stride], values[tid], indices[tid]);
        }
        __syncthreads();
    }

    const int64_t step = *step_counter;
    const int next_step = static_cast<int>(step + 1);
    if (tid == 0) {
        const int token = indices[0];
        token_out[0] = static_cast<OutT>(token);
        generated[step + 1] = static_cast<int64_t>(token);
        input_ids[0] = static_cast<int64_t>(token);
        const int base = base_seqlen[0];
        cache_seqlens[0] = base + next_step;
        cache_seqlens_next[0] = base + next_step + 1;
        step_counter[0] = static_cast<int64_t>(next_step);
    }

    if (tid < head_dim) {
        const int offset = next_step * head_dim + tid;
        cos_out[tid] = cos_table[offset];
        sin_out[tid] = sin_table[offset];
    }
}

int next_power_of_two(int value) {
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

}  // namespace

torch::Tensor cast_bf16_to_fp16_cuda(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bf16");
    TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be fp16");
    TORCH_CHECK(input.numel() == output.numel(), "input/output size mismatch");
    const int n = static_cast<int>(input.numel());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    cast_bf16_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        n);
    return output;
}


torch::Tensor argmax_fp16_cuda(
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out) {
    TORCH_CHECK(logits.is_cuda() && partial_values.is_cuda() && partial_indices.is_cuda() && token_out.is_cuda(),
        "logits, partial buffers and token_out must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat16, "logits must be fp16");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == 1, "logits must have shape [1, vocab]");
    TORCH_CHECK(partial_values.scalar_type() == torch::kFloat32, "partial_values must be fp32");
    TORCH_CHECK(partial_indices.scalar_type() == torch::kInt32, "partial_indices must be int32");
    TORCH_CHECK(token_out.scalar_type() == torch::kLong || token_out.scalar_type() == torch::kInt32,
        "token_out must be int64 or int32");
    const int n = static_cast<int>(logits.size(1));
    const int n_blocks = (n + kArgmaxBlockSize - 1) / kArgmaxBlockSize;
    TORCH_CHECK(partial_values.numel() >= n_blocks && partial_indices.numel() >= n_blocks,
        "partial buffers are too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    argmax_fp16_stage1_kernel<<<n_blocks, kArgmaxBlockSize, 0, stream>>>(
        reinterpret_cast<const half*>(logits.data_ptr<at::Half>()),
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int>(),
        n);
    const int stage2_threads = next_power_of_two(n_blocks);
    if (token_out.scalar_type() == torch::kLong) {
        argmax_fp16_stage2_kernel<int64_t><<<1, stage2_threads, 0, stream>>>(
            partial_values.data_ptr<float>(),
            partial_indices.data_ptr<int>(),
            token_out.data_ptr<int64_t>(),
            n_blocks);
    } else {
        argmax_fp16_stage2_kernel<int><<<1, stage2_threads, 0, stream>>>(
            partial_values.data_ptr<float>(),
            partial_indices.data_ptr<int>(),
            token_out.data_ptr<int>(),
            n_blocks);
    }
    return token_out;
}

torch::Tensor argmax_fp16_post_step_cuda(
    torch::Tensor logits,
    torch::Tensor partial_values,
    torch::Tensor partial_indices,
    torch::Tensor token_out,
    torch::Tensor cos_table,
    torch::Tensor sin_table,
    torch::Tensor cos_out,
    torch::Tensor sin_out,
    torch::Tensor cache_seqlens,
    torch::Tensor cache_seqlens_next,
    torch::Tensor base_seqlen,
    torch::Tensor input_ids,
    torch::Tensor generated,
    torch::Tensor step_counter,
    int64_t head_dim) {
    TORCH_CHECK(logits.is_cuda() && partial_values.is_cuda() && partial_indices.is_cuda() && token_out.is_cuda(),
        "logits, partial buffers and token_out must be CUDA tensors");
    TORCH_CHECK(cos_table.is_cuda() && sin_table.is_cuda() && cos_out.is_cuda() && sin_out.is_cuda(),
        "cos/sin tensors must be CUDA tensors");
    TORCH_CHECK(cache_seqlens.is_cuda() && cache_seqlens_next.is_cuda() && base_seqlen.is_cuda()
        && input_ids.is_cuda() && generated.is_cuda() && step_counter.is_cuda(), "post-step tensors must be CUDA tensors");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat16, "logits must be fp16");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == 1, "logits must have shape [1, vocab]");
    TORCH_CHECK(partial_values.scalar_type() == torch::kFloat32, "partial_values must be fp32");
    TORCH_CHECK(partial_indices.scalar_type() == torch::kInt32, "partial_indices must be int32");
    TORCH_CHECK(token_out.scalar_type() == torch::kLong || token_out.scalar_type() == torch::kInt32,
        "token_out must be int64 or int32");
    TORCH_CHECK(cos_table.scalar_type() == cos_out.scalar_type() && sin_table.scalar_type() == sin_out.scalar_type()
        && cos_table.scalar_type() == sin_table.scalar_type(), "cos/sin dtype mismatch");
    TORCH_CHECK(cache_seqlens.scalar_type() == torch::kInt32 && cache_seqlens_next.scalar_type() == torch::kInt32
        && base_seqlen.scalar_type() == torch::kInt32, "sequence length tensors must be int32");
    TORCH_CHECK(input_ids.scalar_type() == torch::kLong && generated.scalar_type() == torch::kLong
        && step_counter.scalar_type() == torch::kLong, "input/generated/step tensors must be int64");
    const int n = static_cast<int>(logits.size(1));
    const int n_blocks = (n + kArgmaxBlockSize - 1) / kArgmaxBlockSize;
    TORCH_CHECK(partial_values.numel() >= n_blocks && partial_indices.numel() >= n_blocks,
        "partial buffers are too small");
    TORCH_CHECK(head_dim > 0 && head_dim <= 1024, "head_dim must be in (0, 1024]");
    auto stream = at::cuda::getCurrentCUDAStream();
    argmax_fp16_stage1_kernel<<<n_blocks, kArgmaxBlockSize, 0, stream>>>(
        reinterpret_cast<const half*>(logits.data_ptr<at::Half>()),
        partial_values.data_ptr<float>(),
        partial_indices.data_ptr<int>(),
        n);
    const int stage2_threads = next_power_of_two(std::max(n_blocks, static_cast<int>(head_dim)));
    if (cos_table.scalar_type() == torch::kBFloat16) {
        if (token_out.scalar_type() == torch::kLong) {
            argmax_fp16_stage2_post_step_kernel<__nv_bfloat16, int64_t><<<1, stage2_threads, 0, stream>>>(
                partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), token_out.data_ptr<int64_t>(),
                reinterpret_cast<const __nv_bfloat16*>(cos_table.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(sin_table.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(cos_out.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(sin_out.data_ptr<at::BFloat16>()),
                cache_seqlens.data_ptr<int>(), cache_seqlens_next.data_ptr<int>(), base_seqlen.data_ptr<int>(),
                input_ids.data_ptr<int64_t>(), generated.data_ptr<int64_t>(), step_counter.data_ptr<int64_t>(),
                n_blocks, static_cast<int>(head_dim));
        } else {
            argmax_fp16_stage2_post_step_kernel<__nv_bfloat16, int><<<1, stage2_threads, 0, stream>>>(
                partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), token_out.data_ptr<int>(),
                reinterpret_cast<const __nv_bfloat16*>(cos_table.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(sin_table.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(cos_out.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(sin_out.data_ptr<at::BFloat16>()),
                cache_seqlens.data_ptr<int>(), cache_seqlens_next.data_ptr<int>(), base_seqlen.data_ptr<int>(),
                input_ids.data_ptr<int64_t>(), generated.data_ptr<int64_t>(), step_counter.data_ptr<int64_t>(),
                n_blocks, static_cast<int>(head_dim));
        }
    } else {
        TORCH_CHECK(cos_table.scalar_type() == torch::kFloat16, "cos/sin tensors must be fp16 or bf16");
        if (token_out.scalar_type() == torch::kLong) {
            argmax_fp16_stage2_post_step_kernel<half, int64_t><<<1, stage2_threads, 0, stream>>>(
                partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), token_out.data_ptr<int64_t>(),
                reinterpret_cast<const half*>(cos_table.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(sin_table.data_ptr<at::Half>()),
                reinterpret_cast<half*>(cos_out.data_ptr<at::Half>()),
                reinterpret_cast<half*>(sin_out.data_ptr<at::Half>()),
                cache_seqlens.data_ptr<int>(), cache_seqlens_next.data_ptr<int>(), base_seqlen.data_ptr<int>(),
                input_ids.data_ptr<int64_t>(), generated.data_ptr<int64_t>(), step_counter.data_ptr<int64_t>(),
                n_blocks, static_cast<int>(head_dim));
        } else {
            argmax_fp16_stage2_post_step_kernel<half, int><<<1, stage2_threads, 0, stream>>>(
                partial_values.data_ptr<float>(), partial_indices.data_ptr<int>(), token_out.data_ptr<int>(),
                reinterpret_cast<const half*>(cos_table.data_ptr<at::Half>()),
                reinterpret_cast<const half*>(sin_table.data_ptr<at::Half>()),
                reinterpret_cast<half*>(cos_out.data_ptr<at::Half>()),
                reinterpret_cast<half*>(sin_out.data_ptr<at::Half>()),
                cache_seqlens.data_ptr<int>(), cache_seqlens_next.data_ptr<int>(), base_seqlen.data_ptr<int>(),
                input_ids.data_ptr<int64_t>(), generated.data_ptr<int64_t>(), step_counter.data_ptr<int64_t>(),
                n_blocks, static_cast<int>(head_dim));
        }
    }
    return token_out;
}
