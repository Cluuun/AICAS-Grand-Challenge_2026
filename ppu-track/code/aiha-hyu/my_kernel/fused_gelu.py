"""
[SHLEE] Fused GELU (tanh approximation) CUDA kernel -- half2 vectorized.

Vision encoder uses gelu_pytorch_tanh:
  GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

hot path: 27 vision blocks x 1 MLP each = 27 invocations per image.

A100 SM 8.0, FP16.
"""
import torch

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float gelu_tanh(float x) {
    const float s = 0.7978845608f; // sqrt(2/pi)
    const float c = 0.044715f;
    float inner = s * (x + c * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void fused_gelu_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const int n)
{
    const int n2 = n / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    const half2* in2  = reinterpret_cast<const half2*>(input);
    half2*       out2 = reinterpret_cast<half2*>(output);

    half2 v = in2[idx];
    float2 f = __half22float2(v);

    f.x = gelu_tanh(f.x);
    f.y = gelu_tanh(f.y);

    out2[idx] = __float22half2_rn(f);
}

torch::Tensor fused_gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kHalf,
                "input must be CUDA fp16");
    auto inp = input.contiguous();
    auto output = torch::empty_like(inp);
    const int n = inp.numel();
    const int n2 = n / 2;
    constexpr int BLOCK = 256;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_gelu_kernel<<<(n2 + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        reinterpret_cast<const half*>(inp.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        n);

    return output;
}
"""

_CPP_SRC = """
torch::Tensor fused_gelu_forward(torch::Tensor input);
"""

def _load():
    global _module
    if _module is not None:
        return _module
    from my_kernel.extension_loader import load_prebuilt
    _module = load_prebuilt("fused_gelu")
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fused_gelu",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_gelu_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        verbose=False,
    )
    return _module

def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    _load()
    return _module.fused_gelu_forward(x)
