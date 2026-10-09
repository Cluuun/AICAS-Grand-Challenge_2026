from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "lm_head_top1_fp8.cu"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_cuda_lm_head_top1_fp8"


def _set_default_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


@lru_cache(maxsize=1)
def load_cuda_lm_head_top1_fp8_module(verbose: bool = False):
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


def _make_workspace(
    *,
    device: torch.device | str,
    n_rows: int,
    block_n: int,
    reduce_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = (int(n_rows) + int(block_n) - 1) // int(block_n)
    num_stage_blocks = (num_blocks + int(reduce_block) - 1) // int(reduce_block)
    partial_values = torch.empty((num_blocks,), device=device, dtype=torch.float32)
    partial_indices = torch.empty((num_blocks,), device=device, dtype=torch.long)
    stage_values = torch.empty((num_stage_blocks,), device=device, dtype=torch.float32)
    stage_indices = torch.zeros((num_stage_blocks,), device=device, dtype=torch.long)
    return partial_values, partial_indices, stage_values, stage_indices


def _check_common(
    hidden: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    out_value: torch.Tensor | None,
    out_token: torch.Tensor | None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    *,
    block_n: int,
    block_k: int,
    reduce_block: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    bool,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if not hidden.is_cuda or not weight_fp8.is_cuda or not scales.is_cuda:
        raise ValueError("CUDA lm_head FP8 top1 requires CUDA tensors")
    if hidden.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("CUDA lm_head FP8 top1 expects fp16 hidden/scales")
    if weight_fp8.dtype != torch.uint8 or weight_fp8.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight_fp8 must be uint8 [rows, hidden] and scales [rows, hidden / 128]")
    block_n = int(block_n)
    block_k = int(block_k)
    reduce_block = int(reduce_block)
    if block_n != 32 or block_k != 128:
        raise ValueError("CUDA lm_head FP8 top1 currently supports block_n=32 and block_k=128")
    if reduce_block not in (512, 1024):
        raise ValueError("CUDA lm_head FP8 top1 supports reduce_block 512 or 1024")

    hidden_flat = hidden.reshape(-1).contiguous()
    n_rows, k_in = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
    if int(hidden_flat.numel()) != k_in or k_in % block_k != 0:
        raise ValueError("hidden size mismatch or not divisible by block_k")
    if int(scales.shape[0]) != n_rows or int(scales.shape[1]) != k_in // block_k:
        raise ValueError("scales shape does not match weight_fp8")

    if bias is not None:
        raise ValueError("CUDA lm_head FP8 top1 is specialized for Qwen3-VL lm_head bias=False")
    bias_flat = hidden_flat
    has_bias = False

    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_value.dtype != torch.float16 or not out_value.is_cuda:
        raise ValueError("out_value must be a CUDA fp16 tensor")
    if out_token.dtype != torch.long or not out_token.is_cuda:
        raise ValueError("out_token must be a CUDA long tensor")

    if workspace is None:
        workspace = _make_workspace(device=hidden.device, n_rows=n_rows, block_n=block_n, reduce_block=reduce_block)
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = (n_rows + block_n - 1) // block_n
    num_stage_blocks = (num_blocks + reduce_block - 1) // reduce_block
    if (
        partial_values.dtype != torch.float32
        or partial_indices.dtype != torch.long
        or stage_values.dtype != torch.float32
        or stage_indices.dtype != torch.long
    ):
        raise ValueError("workspace dtypes must be fp32/long/fp32/long")
    if int(partial_values.numel()) < num_blocks or int(partial_indices.numel()) < num_blocks:
        raise ValueError("partial workspace is too small")
    if int(stage_values.numel()) < num_stage_blocks or int(stage_indices.numel()) < num_stage_blocks:
        raise ValueError("stage workspace is too small")

    return (
        hidden_flat,
        bias_flat,
        has_bias,
        out_value,
        out_token,
        out_value.reshape(-1),
        out_token.reshape(-1),
        weight_fp8.contiguous(),
        workspace,
    )


def cuda_lm_head_top1_fp8(
    hidden: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_value: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_n: int = 32,
    block_k: int = 128,
    reduce_block: int = 1024,
    num_warps: int | None = None,
    num_stages: int | None = None,
    reduce_num_warps: int | None = None,
    variant: int = 53,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        hidden_flat,
        bias_flat,
        has_bias,
        out_value,
        out_token,
        out_value_flat,
        out_token_flat,
        weight_fp8,
        workspace,
    ) = _check_common(
        hidden,
        weight_fp8,
        scales,
        bias,
        out_value,
        out_token,
        workspace,
        block_n=int(block_n),
        block_k=int(block_k),
        reduce_block=int(reduce_block),
    )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    module = load_cuda_lm_head_top1_fp8_module(verbose=verbose)
    module.lm_head_top1_fp8_full(
        hidden_flat,
        weight_fp8,
        scales.contiguous(),
        bias_flat,
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        out_value_flat,
        out_token_flat,
        bool(has_bias),
        int(reduce_block),
        int(variant),
    )
    return out_value_flat.view_as(out_value), out_token_flat.view_as(out_token)


def cuda_lm_head_shortlist_compact_top1_fp8(
    hidden: torch.Tensor,
    weight_fp8: torch.Tensor,
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
    variant: int = 52,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_ids.is_cuda or token_ids.dtype != torch.long or token_ids.ndim != 1:
        raise ValueError("token_ids must be a CUDA long vector")
    if int(token_ids.numel()) != int(weight_fp8.shape[0]):
        raise ValueError("compact weight and token_ids sizes do not match")
    (
        hidden_flat,
        bias_flat,
        has_bias,
        out_value,
        out_token,
        out_value_flat,
        out_token_flat,
        weight_fp8,
        workspace,
    ) = _check_common(
        hidden,
        weight_fp8,
        scales,
        bias,
        out_value,
        out_token,
        workspace,
        block_n=int(block_n),
        block_k=int(block_k),
        reduce_block=int(reduce_block),
    )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    module = load_cuda_lm_head_top1_fp8_module(verbose=verbose)
    module.lm_head_top1_fp8_compact(
        hidden_flat,
        weight_fp8,
        scales.contiguous(),
        token_ids.contiguous(),
        bias_flat,
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        out_value_flat,
        out_token_flat,
        bool(has_bias),
        int(reduce_block),
        int(variant),
    )
    return out_value_flat.view_as(out_value), out_token_flat.view_as(out_token)
