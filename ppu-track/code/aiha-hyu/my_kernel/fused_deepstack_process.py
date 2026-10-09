import torch

_module = None

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_deepstack_process_kernel(
    half* __restrict__ hidden_states,
    const half* __restrict__ visual_embeds,
    const int start_offset,
    const int V,
    const int D)
{
    const int total_elements = (V * D) / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const half2* visual_embeds2 = reinterpret_cast<const half2*>(visual_embeds);
    half2* hidden_states2 = reinterpret_cast<half2*>(hidden_states) + (start_offset * D / 2);

    half2 v2 = visual_embeds2[idx];
    half2 h2 = hidden_states2[idx];

    half2 res2 = __hadd2(h2, v2);

    hidden_states2[idx] = res2;
}

torch::Tensor fused_deepstack_process_forward(
    torch::Tensor& hidden_states,
    const torch::Tensor& visual_embeds,
    const int start_offset)
{
    TORCH_CHECK(hidden_states.is_cuda() && hidden_states.scalar_type() == torch::kHalf,
                "hidden_states must be CUDA fp16");
    TORCH_CHECK(visual_embeds.is_cuda() && visual_embeds.scalar_type() == torch::kHalf,
                "visual_embeds must be CUDA fp16");

    auto hidden_states_c = hidden_states.contiguous();
    auto visual_embeds_c = visual_embeds.contiguous();

    const int D = hidden_states_c.size(-1);
    auto hidden_states_2d = hidden_states_c.view({-1, D});
    const int V = visual_embeds_c.size(0);

    if (V == 0) {
        return hidden_states_c;
    }

    const int total_elements = (V * D) / 2;
    constexpr int BLOCK = 256;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_deepstack_process_kernel<<<(total_elements + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        reinterpret_cast<half*>(hidden_states_2d.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(visual_embeds_c.data_ptr<at::Half>()),
        start_offset,
        V,
        D);

    return hidden_states_c;
}
"""

_CPP_SRC = """
torch::Tensor fused_deepstack_process_forward(torch::Tensor& hidden_states, const torch::Tensor& visual_embeds, const int start_offset);
"""

def _load():
    global _module
    if _module is not None:
        return _module
    from my_kernel.extension_loader import load_prebuilt
    _module = load_prebuilt("fused_deepstack_process")
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load_inline
    _module = load_inline(
        name="fused_deepstack_process",
        cpp_sources=[_CPP_SRC],
        cuda_sources=[_CUDA_SRC],
        functions=["fused_deepstack_process_forward"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--ptxas-options=-v"],
        verbose=False,
    )
    return _module

def fused_deepstack_process(
    hidden_states: torch.Tensor,
    visual_embeds: torch.Tensor,
    start_offset: int,
) -> torch.Tensor:
    """Add visual_embeds to hidden_states[start_offset : start_offset + V, :], in-place."""
    _load()
    return _module.fused_deepstack_process_forward(hidden_states, visual_embeds, start_offset)
