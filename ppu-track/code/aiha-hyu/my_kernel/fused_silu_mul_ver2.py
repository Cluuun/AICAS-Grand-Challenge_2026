"""
[SHLEE] Fused SiLU*Mul CUDA kernel -- half2 vectorized, 2x throughput vs scalar.

Hot path: gate/up shape [batch, seq, 6144] for 28 MLP layers.
Element-wise and fully memory-bound, so vectorized loads are critical.
"""
import torch

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <limits>

__global__ void fused_silu_mul_from_gu_kernel(
    const half* __restrict__ gu,
    half* __restrict__ out,
    const int rows,
    const int intermediate)
{
    const int I2 = intermediate >> 1;
    const int total = rows * I2;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int row = idx / I2;
    const int col2 = idx - row * I2;

    const half2* gu2 = reinterpret_cast<const half2*>(gu);
    half2* out2 = reinterpret_cast<half2*>(out);

    const int gu_row_base2 = row * intermediate;
    const int out_row_base2 = row * I2;

    half2 g2 = gu2[gu_row_base2 + col2];
    half2 u2 = gu2[gu_row_base2 + I2 + col2];

    float2 gf = __half22float2(g2);
    float2 uf = __half22float2(u2);

    gf.x = gf.x / (1.0f + __expf(-gf.x)) * uf.x;
    gf.y = gf.y / (1.0f + __expf(-gf.y)) * uf.y;

    out2[out_row_base2 + col2] = __float22half2_rn(gf);
}

torch::Tensor fused_silu_mul_forward_v2(
    torch::Tensor gu,
    int64_t intermediate)
{
    const int64_t last_dim = gu.size(-1);

    const int64_t rows64 = gu.numel() / last_dim;

    const int rows = static_cast<int>(rows64);
    const int I = static_cast<int>(intermediate);
    const int I2 = I >> 1;

    const int64_t total64 = static_cast<int64_t>(rows) * static_cast<int64_t>(I2);
    const int total = static_cast<int>(total64);

    std::vector<int64_t> out_sizes;
    out_sizes.reserve(gu.dim());
    for (int i = 0; i < gu.dim() - 1; ++i) {
        out_sizes.push_back(gu.size(i));
    }
    out_sizes.push_back(intermediate);

    auto output = torch::empty(out_sizes, gu.options());

    constexpr int BLOCK = 256;
    dim3 block(BLOCK);
    dim3 grid((total + BLOCK - 1) / BLOCK);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_silu_mul_from_gu_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(gu.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        rows,
        I
    );

    return output;
}
"""

_CPP_SRC = """
torch::Tensor fused_silu_mul_forward_v2(torch::Tensor gu, int64_t intermediate);
"""

def _load():
    global _module
    if _module is not None:
        return _module
    from my_kernel.extension_loader import load_prebuilt
    _module = load_prebuilt("fused_silu_mul_ver2")
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fused_silu_mul_ver2",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_silu_mul_forward_v2"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        verbose=False,
    )
    return _module

def fused_silu_mul(gu: torch.Tensor, intermediate: int) -> torch.Tensor:
    _load()
    return _module.fused_silu_mul_forward_v2(gu, int(intermediate))
