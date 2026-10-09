#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHidden = 2048;
constexpr int kPackedHidden = kHidden / 4;
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
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
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

__global__ void rstd_kernel(const __nv_bfloat16* __restrict__ residual, float* __restrict__ rstd, float eps) {
    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < kHidden; idx += blockDim.x) {
        float x = bf16_to_float(residual[idx]);
        sum += x * x;
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) {
        rstd[0] = rsqrtf(sum / static_cast<float>(kHidden) + eps);
    }
}

template<int WarpsPerBlock>
__global__ void gate_up_q8_prescaled_warp_kernel(
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ norm_weight,
    const float* __restrict__ rstd_ptr,
    const int* __restrict__ packed_weight,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    int intermediate_size) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WarpsPerBlock + warp;
    if (row >= intermediate_size) {
        return;
    }
    const int gate_row = row * 2;
    const int up_row = gate_row + 1;
    const int packed_stride = kPackedHidden;
    const float rstd = rstd_ptr[0];
    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int pk = lane; pk < kPackedHidden; pk += 32) {
        const int base = pk * 4;
        const float x0 = bf16_to_float(residual[base]) * bf16_to_float(norm_weight[base]) * rstd;
        const float x1 = bf16_to_float(residual[base + 1]) * bf16_to_float(norm_weight[base + 1]) * rstd;
        const float x2 = bf16_to_float(residual[base + 2]) * bf16_to_float(norm_weight[base + 2]) * rstd;
        const float x3 = bf16_to_float(residual[base + 3]) * bf16_to_float(norm_weight[base + 3]) * rstd;

        const int gp = packed_weight[gate_row * packed_stride + pk];
        const int up = packed_weight[up_row * packed_stride + pk];
        gate_acc += (static_cast<float>(gp & 0xFF) - 128.0f) * x0;
        gate_acc += (static_cast<float>((gp >> 8) & 0xFF) - 128.0f) * x1;
        gate_acc += (static_cast<float>((gp >> 16) & 0xFF) - 128.0f) * x2;
        gate_acc += (static_cast<float>((gp >> 24) & 0xFF) - 128.0f) * x3;
        up_acc += (static_cast<float>(up & 0xFF) - 128.0f) * x0;
        up_acc += (static_cast<float>((up >> 8) & 0xFF) - 128.0f) * x1;
        up_acc += (static_cast<float>((up >> 16) & 0xFF) - 128.0f) * x2;
        up_acc += (static_cast<float>((up >> 24) & 0xFF) - 128.0f) * x3;
    }

    gate_acc = warp_sum(gate_acc);
    up_acc = warp_sum(up_acc);
    if (lane == 0) {
        gate_acc *= bf16_to_float(scale[gate_row]);
        up_acc *= bf16_to_float(scale[up_row]);
        float gate = gate_acc / (1.0f + expf(-gate_acc));
        output[row] = __float2bfloat16(gate * up_acc);
    }
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void sage_half_layer_rstd_cuda(torch::Tensor residual, torch::Tensor rstd, double eps) {
    check_tensor(residual, "residual");
    check_tensor(rstd, "rstd");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(rstd.scalar_type() == at::kFloat, "rstd must be FP32");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have hidden size 2048");
    auto stream = at::cuda::getCurrentCUDAStream();
    rstd_kernel<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        rstd.data_ptr<float>(),
        static_cast<float>(eps));
}

void sage_half_layer_gate_up_q8_cuda(
    torch::Tensor residual,
    torch::Tensor norm_weight,
    torch::Tensor rstd,
    torch::Tensor packed_weight,
    torch::Tensor scale,
    torch::Tensor output) {
    check_tensor(residual, "residual");
    check_tensor(norm_weight, "norm_weight");
    check_tensor(rstd, "rstd");
    check_tensor(packed_weight, "packed_weight");
    check_tensor(scale, "scale");
    check_tensor(output, "output");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be BF16");
    TORCH_CHECK(norm_weight.scalar_type() == at::kBFloat16, "norm_weight must be BF16");
    TORCH_CHECK(rstd.scalar_type() == at::kFloat, "rstd must be FP32");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "packed_weight must be int32");
    TORCH_CHECK(scale.scalar_type() == at::kBFloat16, "scale must be BF16");
    TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be BF16");
    TORCH_CHECK(residual.numel() == kHidden, "residual must have hidden size 2048");
    TORCH_CHECK(norm_weight.numel() == kHidden, "norm_weight must have hidden size 2048");
    TORCH_CHECK(packed_weight.dim() == 2 && packed_weight.size(1) == kPackedHidden, "packed_weight must be [2N, 512]");
    TORCH_CHECK(packed_weight.size(0) == output.numel() * 2, "packed_weight rows must be 2 * output rows");
    TORCH_CHECK(scale.numel() >= packed_weight.size(0), "scale is too small");
    auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int warps_per_block = 8;
    int grid = (static_cast<int>(output.numel()) + warps_per_block - 1) / warps_per_block;
    gate_up_q8_prescaled_warp_kernel<warps_per_block><<<grid, warps_per_block * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
        rstd.data_ptr<float>(),
        packed_weight.data_ptr<int>(),
        reinterpret_cast<const __nv_bfloat16*>(scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        static_cast<int>(output.numel()));
}
