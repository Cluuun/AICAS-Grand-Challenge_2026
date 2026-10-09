from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "lm_head_top1_int4.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_lm_head_top1_int4"


def _set_default_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


@lru_cache(maxsize=1)
def load_cuda_lm_head_top1_int4_module(verbose: bool = False):
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _set_default_arch_list()
    return load(
        name=_MODULE_NAME,
        sources=[str(_SOURCE)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(_BUILD_DIR),
        verbose=verbose,
    )


def cuda_lm_head_shortlist_compact_top1_int4(
    hidden: torch.Tensor,
    weight_int4: torch.Tensor,
    scales: torch.Tensor,
    token_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_value: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_n: int = 32,
    block_k: int = 128,
    reduce_block: int = 512,
    num_warps: int | None = None,
    num_stages: int | None = None,
    reduce_num_warps: int | None = None,
    variant: int = 0,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA symmetric-INT4 compact shortlist lm_head top1 for harness experiments."""
    if not hidden.is_cuda or not weight_int4.is_cuda or not scales.is_cuda or not token_ids.is_cuda:
        raise ValueError("CUDA lm_head INT4 compact requires CUDA tensors")
    if hidden.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("CUDA lm_head INT4 compact expects fp16 hidden/scales")
    if weight_int4.dtype != torch.uint8 or weight_int4.ndim != 3:
        raise ValueError("weight_int4 must be uint8 [16, n_rows, 64]")
    if token_ids.dtype != torch.long or token_ids.ndim != 1:
        raise ValueError("token_ids must be a CUDA long vector")
    if bias is not None:
        raise ValueError("CUDA lm_head INT4 compact is specialized for Qwen3-VL lm_head bias=False")
    if int(block_n) != 32 or int(block_k) != 128:
        raise ValueError("CUDA lm_head INT4 compact currently supports block_n=32 and block_k=128")
    if int(weight_int4.shape[0]) != 16 or int(weight_int4.shape[2]) != 64:
        raise ValueError("weight_int4 must be [16, n_rows, 64]")
    if int(scales.shape[0]) != 16 or int(scales.shape[1]) != int(weight_int4.shape[1]):
        raise ValueError("scales must be [16, n_rows]")
    if int(token_ids.numel()) != int(weight_int4.shape[1]):
        raise ValueError("compact weight and token_ids sizes do not match")

    hidden_flat = hidden.reshape(-1).contiguous()
    if int(hidden_flat.numel()) != 2048:
        raise ValueError("CUDA lm_head INT4 compact expects K=2048")
    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_value.dtype != torch.float16 or not out_value.is_cuda:
        raise ValueError("out_value must be a CUDA fp16 tensor")
    if out_token.dtype != torch.long or not out_token.is_cuda:
        raise ValueError("out_token must be a CUDA long tensor")
    if workspace is None:
        n_rows = int(weight_int4.shape[1])
        num_blocks = (n_rows + int(block_n) - 1) // int(block_n)
        partial_values = torch.empty((num_blocks,), device=hidden.device, dtype=torch.float32)
        partial_indices = torch.empty((num_blocks,), device=hidden.device, dtype=torch.long)
        atomic_state = torch.empty((1,), device=hidden.device, dtype=torch.long)
    else:
        partial_values, partial_indices, _stage_values, atomic_state = workspace
    module = load_cuda_lm_head_top1_int4_module(verbose=verbose)
    module.lm_head_top1_int4_compact(
        hidden_flat,
        weight_int4.contiguous(),
        scales.contiguous(),
        token_ids.contiguous(),
        partial_values,
        partial_indices,
        atomic_state,
        out_value.reshape(-1),
        out_token.reshape(-1),
    )
    return out_value, out_token
