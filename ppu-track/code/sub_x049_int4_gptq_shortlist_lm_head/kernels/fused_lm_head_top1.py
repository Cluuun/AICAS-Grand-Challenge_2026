from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import utils as cuda_utils


@triton.jit
def _lm_head_block_top1_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    partial_value_ptr,
    partial_index_ptr,
    n_vocab: tl.constexpr,
    k_in: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_vocab

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * k_in + offs_k[None, :],
            mask=mask_n[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.where(mask_n, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, offs_n, n_vocab), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


@triton.jit
def _lm_head_block_top1_fp8_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    bias_ptr,
    partial_value_ptr,
    partial_index_ptr,
    n_vocab: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_vocab

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        scale_idx: tl.constexpr = k0 // BLOCK_K
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        packed = tl.load(
            w_fp8_ptr + offs_n[:, None] * k_in + offs_k[None, :],
            mask=mask_n[:, None],
            other=0,
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + offs_n * scale_blocks + scale_idx,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    acc = tl.where(mask_n, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, offs_n, n_vocab), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


@triton.jit
def _lm_head_partial_reduce_kernel(
    partial_value_ptr,
    partial_index_ptr,
    stage_value_ptr,
    stage_index_ptr,
    num_blocks: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * REDUCE_BLOCK + tl.arange(0, REDUCE_BLOCK)
    mask = offs < num_blocks
    values = tl.load(partial_value_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
    indices = tl.load(partial_index_ptr + offs, mask=mask, other=2147483647)
    best_value = tl.max(values, axis=0)
    best_index = tl.min(tl.where(values == best_value, indices, 2147483647), axis=0)
    tl.store(stage_value_ptr + pid, best_value)
    tl.store(stage_index_ptr + pid, best_index)


@triton.jit
def _lm_head_shortlist_block_top1_kernel(
    x_ptr,
    w_ptr,
    token_ids_ptr,
    bias_ptr,
    partial_value_ptr,
    partial_index_ptr,
    shortlist_size: tl.constexpr,
    k_in: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_s = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_s = offs_s < shortlist_size
    token_ids = tl.load(token_ids_ptr + offs_s, mask=mask_s, other=0)

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        w = tl.load(
            w_ptr + token_ids[:, None] * k_in + offs_k[None, :],
            mask=mask_s[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + token_ids, mask=mask_s, other=0.0).to(tl.float32)

    acc = tl.where(mask_s, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, token_ids, 2147483647), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


@triton.jit
def _lm_head_shortlist_block_top1_fp8_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    token_ids_ptr,
    bias_ptr,
    partial_value_ptr,
    partial_index_ptr,
    shortlist_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_s = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_s = offs_s < shortlist_size
    token_ids = tl.load(token_ids_ptr + offs_s, mask=mask_s, other=0)

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        scale_idx: tl.constexpr = k0 // BLOCK_K
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        packed = tl.load(
            w_fp8_ptr + token_ids[:, None] * k_in + offs_k[None, :],
            mask=mask_s[:, None],
            other=0,
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + token_ids * scale_blocks + scale_idx,
            mask=mask_s,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    if HAS_BIAS:
        acc += tl.load(bias_ptr + token_ids, mask=mask_s, other=0.0).to(tl.float32)

    acc = tl.where(mask_s, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, token_ids, 2147483647), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


@triton.jit
def _lm_head_shortlist_compact_block_top1_fp8_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    token_ids_ptr,
    bias_ptr,
    partial_value_ptr,
    partial_index_ptr,
    shortlist_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_s = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_s = offs_s < shortlist_size

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        scale_idx: tl.constexpr = k0 // BLOCK_K
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        packed = tl.load(
            w_fp8_ptr + offs_s[:, None] * k_in + offs_k[None, :],
            mask=mask_s[:, None],
            other=0,
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + offs_s * scale_blocks + scale_idx,
            mask=mask_s,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    if HAS_BIAS:
        token_ids = tl.load(token_ids_ptr + offs_s, mask=mask_s, other=0)
        acc += tl.load(bias_ptr + token_ids, mask=mask_s, other=0.0).to(tl.float32)

    acc = tl.where(mask_s, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, offs_s, 2147483647), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


@triton.jit
def _lm_head_final_reduce_kernel(
    stage_value_ptr,
    stage_index_ptr,
    out_value_ptr,
    out_token_ptr,
    num_stage_blocks: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_M)
    mask = offs < num_stage_blocks
    values = tl.load(stage_value_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
    indices = tl.load(stage_index_ptr + offs, mask=mask, other=2147483647)
    best_value = tl.max(values, axis=0)
    best_index = tl.min(tl.where(values == best_value, indices, 2147483647), axis=0)
    tl.store(out_value_ptr, best_value)
    tl.store(out_token_ptr, best_index)


@triton.jit
def _lm_head_final_reduce_mapped_kernel(
    stage_value_ptr,
    stage_index_ptr,
    token_ids_ptr,
    out_value_ptr,
    out_token_ptr,
    num_stage_blocks: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_M)
    mask = offs < num_stage_blocks
    values = tl.load(stage_value_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
    indices = tl.load(stage_index_ptr + offs, mask=mask, other=2147483647)
    best_value = tl.max(values, axis=0)
    best_shortlist_index = tl.min(tl.where(values == best_value, indices, 2147483647), axis=0)
    best_token = tl.load(token_ids_ptr + best_shortlist_index)
    tl.store(out_value_ptr, best_value)
    tl.store(out_token_ptr, best_token)


def create_lm_head_top1_workspace(
    *,
    device: torch.device | str,
    n_vocab: int,
    block_n: int,
    reduce_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = triton.cdiv(int(n_vocab), int(block_n))
    num_stage_blocks = triton.cdiv(num_blocks, int(reduce_block))
    partial_values = torch.empty((num_blocks,), device=device, dtype=torch.float32)
    partial_indices = torch.empty((num_blocks,), device=device, dtype=torch.int64)
    stage_values = torch.empty((num_stage_blocks,), device=device, dtype=torch.float32)
    stage_indices = torch.zeros((num_stage_blocks,), device=device, dtype=torch.int64)
    return partial_values, partial_indices, stage_values, stage_indices


def triton_lm_head_top1(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_value: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_n: int = 32,
    block_k: int = 128,
    reduce_block: int = 1024,
    num_warps: int = 8,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact decode-only fp16 lm_head top-1 over the full vocabulary."""
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("triton_lm_head_top1 requires CUDA tensors")
    if hidden.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("triton_lm_head_top1 currently expects fp16 hidden and weight")
    if weight.ndim != 2:
        raise ValueError("weight must be [vocab, hidden]")

    hidden_flat = hidden.reshape(-1)
    n_vocab, k_in = int(weight.shape[0]), int(weight.shape[1])
    if int(hidden_flat.numel()) != k_in:
        raise ValueError(f"hidden has {hidden_flat.numel()} elements, expected {k_in}")

    if bias is not None:
        if not bias.is_cuda or bias.dtype != hidden.dtype or int(bias.numel()) != n_vocab:
            raise ValueError("bias must be fp16 CUDA tensor with shape [vocab]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = hidden_flat

    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_token.dtype != torch.long:
        raise ValueError("out_token must have dtype torch.long")

    block_n = int(block_n)
    block_k = int(block_k)
    reduce_block = int(reduce_block)
    if k_in % block_k != 0:
        raise ValueError("triton_lm_head_top1 requires hidden size divisible by block_k")
    if block_n <= 0 or reduce_block <= 0:
        raise ValueError("block_n and reduce_block must be positive")

    if workspace is None:
        workspace = create_lm_head_top1_workspace(
            device=hidden.device,
            n_vocab=n_vocab,
            block_n=block_n,
            reduce_block=reduce_block,
        )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = triton.cdiv(n_vocab, block_n)
    num_stage_blocks = triton.cdiv(num_blocks, reduce_block)
    if int(partial_values.numel()) < num_blocks or int(partial_indices.numel()) < num_blocks:
        raise ValueError("partial workspace is too small")
    if int(stage_values.numel()) < num_stage_blocks or int(stage_indices.numel()) < num_stage_blocks:
        raise ValueError("stage workspace is too small")

    _lm_head_block_top1_kernel[(num_blocks,)](
        hidden_flat,
        weight,
        bias_flat,
        partial_values,
        partial_indices,
        n_vocab,
        k_in,
        bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _lm_head_partial_reduce_kernel[(num_stage_blocks,)](
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        num_blocks,
        REDUCE_BLOCK=reduce_block,
        num_warps=int(reduce_num_warps),
        num_stages=1,
    )
    final_block = triton.next_power_of_2(num_stage_blocks)
    _lm_head_final_reduce_kernel[(1,)](
        stage_values,
        stage_indices,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_stage_blocks,
        BLOCK_M=final_block,
        num_warps=1,
        num_stages=1,
    )
    return out_value, out_token


def triton_lm_head_shortlist_top1(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_value: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_n: int = 32,
    block_k: int = 128,
    reduce_block: int = 1024,
    num_warps: int = 8,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate decode-only top-1 over a caller-provided vocabulary subset."""
    if not hidden.is_cuda or not weight.is_cuda or not token_ids.is_cuda:
        raise ValueError("triton_lm_head_shortlist_top1 requires CUDA tensors")
    if hidden.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("triton_lm_head_shortlist_top1 currently expects fp16 hidden and weight")
    if weight.ndim != 2:
        raise ValueError("weight must be [vocab, hidden]")
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a 1D torch.long CUDA tensor")

    hidden_flat = hidden.reshape(-1)
    n_vocab, k_in = int(weight.shape[0]), int(weight.shape[1])
    shortlist_size = int(token_ids.numel())
    if shortlist_size <= 0:
        raise ValueError("token_ids must not be empty")
    if int(hidden_flat.numel()) != k_in:
        raise ValueError(f"hidden has {hidden_flat.numel()} elements, expected {k_in}")

    if bias is not None:
        if not bias.is_cuda or bias.dtype != hidden.dtype or int(bias.numel()) != n_vocab:
            raise ValueError("bias must be fp16 CUDA tensor with shape [vocab]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = hidden_flat

    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_token.dtype != torch.long:
        raise ValueError("out_token must have dtype torch.long")

    block_n = int(block_n)
    block_k = int(block_k)
    reduce_block = int(reduce_block)
    if k_in % block_k != 0:
        raise ValueError("triton_lm_head_shortlist_top1 requires hidden size divisible by block_k")
    if block_n <= 0 or reduce_block <= 0:
        raise ValueError("block_n and reduce_block must be positive")

    if workspace is None:
        workspace = create_lm_head_top1_workspace(
            device=hidden.device,
            n_vocab=shortlist_size,
            block_n=block_n,
            reduce_block=reduce_block,
        )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = triton.cdiv(shortlist_size, block_n)
    num_stage_blocks = triton.cdiv(num_blocks, reduce_block)
    if int(partial_values.numel()) < num_blocks or int(partial_indices.numel()) < num_blocks:
        raise ValueError("partial workspace is too small")
    if int(stage_values.numel()) < num_stage_blocks or int(stage_indices.numel()) < num_stage_blocks:
        raise ValueError("stage workspace is too small")

    _lm_head_shortlist_block_top1_kernel[(num_blocks,)](
        hidden_flat,
        weight,
        token_ids,
        bias_flat,
        partial_values,
        partial_indices,
        shortlist_size,
        k_in,
        bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _lm_head_partial_reduce_kernel[(num_stage_blocks,)](
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        num_blocks,
        REDUCE_BLOCK=reduce_block,
        num_warps=int(reduce_num_warps),
        num_stages=1,
    )
    final_block = triton.next_power_of_2(num_stage_blocks)
    _lm_head_final_reduce_kernel[(1,)](
        stage_values,
        stage_indices,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_stage_blocks,
        BLOCK_M=final_block,
        num_warps=1,
        num_stages=1,
    )
    return out_value, out_token


def triton_lm_head_top1_fp8(
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
    num_warps: int = 8,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact decode-only fp8e4b15 lm_head top-1 over the full vocabulary."""
    if not hidden.is_cuda or not weight_fp8.is_cuda or not scales.is_cuda:
        raise ValueError("triton_lm_head_top1_fp8 requires CUDA tensors")
    if hidden.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("triton_lm_head_top1_fp8 expects fp16 hidden/scales")
    if weight_fp8.dtype != torch.uint8 or weight_fp8.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight_fp8 must be uint8 [vocab, hidden] and scales [vocab, hidden / 128]")

    hidden_flat = hidden.reshape(-1)
    n_vocab, k_in = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_lm_head_top1_fp8 currently requires block_k=128")
    if int(hidden_flat.numel()) != k_in or k_in % block_k != 0:
        raise ValueError("hidden size mismatch or not divisible by block_k")
    if int(scales.shape[0]) != n_vocab or int(scales.shape[1]) != k_in // block_k:
        raise ValueError("scales shape does not match weight_fp8")

    if bias is not None:
        if not bias.is_cuda or bias.dtype != hidden.dtype or int(bias.numel()) != n_vocab:
            raise ValueError("bias must be fp16 CUDA tensor with shape [vocab]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = hidden_flat

    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_token.dtype != torch.long:
        raise ValueError("out_token must have dtype torch.long")

    block_n = int(block_n)
    reduce_block = int(reduce_block)
    if workspace is None:
        workspace = create_lm_head_top1_workspace(
            device=hidden.device,
            n_vocab=n_vocab,
            block_n=block_n,
            reduce_block=reduce_block,
        )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = triton.cdiv(n_vocab, block_n)
    num_stage_blocks = triton.cdiv(num_blocks, reduce_block)

    _lm_head_block_top1_fp8_block128_kernel[(num_blocks,)](
        hidden_flat,
        weight_fp8,
        scales,
        bias_flat,
        partial_values,
        partial_indices,
        n_vocab,
        k_in,
        k_in // block_k,
        bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _lm_head_partial_reduce_kernel[(num_stage_blocks,)](
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        num_blocks,
        REDUCE_BLOCK=reduce_block,
        num_warps=int(reduce_num_warps),
        num_stages=1,
    )
    _lm_head_final_reduce_kernel[(1,)](
        stage_values,
        stage_indices,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_stage_blocks,
        BLOCK_M=triton.next_power_of_2(num_stage_blocks),
        num_warps=1,
        num_stages=1,
    )
    return out_value, out_token


def triton_lm_head_shortlist_top1_fp8(
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
    reduce_block: int = 1024,
    num_warps: int = 8,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate fp8e4b15 top-1 over a caller-provided vocabulary subset."""
    if not hidden.is_cuda or not weight_fp8.is_cuda or not scales.is_cuda or not token_ids.is_cuda:
        raise ValueError("triton_lm_head_shortlist_top1_fp8 requires CUDA tensors")
    if hidden.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("triton_lm_head_shortlist_top1_fp8 expects fp16 hidden/scales")
    if weight_fp8.dtype != torch.uint8 or weight_fp8.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight_fp8 must be uint8 [vocab, hidden] and scales [vocab, hidden / 128]")
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a 1D torch.long CUDA tensor")

    hidden_flat = hidden.reshape(-1)
    n_vocab, k_in = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
    shortlist_size = int(token_ids.numel())
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_lm_head_shortlist_top1_fp8 currently requires block_k=128")
    if shortlist_size <= 0:
        raise ValueError("token_ids must not be empty")
    if int(hidden_flat.numel()) != k_in or k_in % block_k != 0:
        raise ValueError("hidden size mismatch or not divisible by block_k")
    if int(scales.shape[0]) != n_vocab or int(scales.shape[1]) != k_in // block_k:
        raise ValueError("scales shape does not match weight_fp8")

    if bias is not None:
        if not bias.is_cuda or bias.dtype != hidden.dtype or int(bias.numel()) != n_vocab:
            raise ValueError("bias must be fp16 CUDA tensor with shape [vocab]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = hidden_flat

    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_token.dtype != torch.long:
        raise ValueError("out_token must have dtype torch.long")

    block_n = int(block_n)
    reduce_block = int(reduce_block)
    if workspace is None:
        workspace = create_lm_head_top1_workspace(
            device=hidden.device,
            n_vocab=shortlist_size,
            block_n=block_n,
            reduce_block=reduce_block,
        )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = triton.cdiv(shortlist_size, block_n)
    num_stage_blocks = triton.cdiv(num_blocks, reduce_block)

    _lm_head_shortlist_block_top1_fp8_block128_kernel[(num_blocks,)](
        hidden_flat,
        weight_fp8,
        scales,
        token_ids,
        bias_flat,
        partial_values,
        partial_indices,
        shortlist_size,
        k_in,
        k_in // block_k,
        bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _lm_head_partial_reduce_kernel[(num_stage_blocks,)](
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        num_blocks,
        REDUCE_BLOCK=reduce_block,
        num_warps=int(reduce_num_warps),
        num_stages=1,
    )
    _lm_head_final_reduce_kernel[(1,)](
        stage_values,
        stage_indices,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_stage_blocks,
        BLOCK_M=triton.next_power_of_2(num_stage_blocks),
        num_warps=1,
        num_stages=1,
    )
    return out_value, out_token


def triton_lm_head_shortlist_compact_top1_fp8(
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
    reduce_block: int = 1024,
    num_warps: int = 8,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate fp8e4b15 top-1 over a compact caller-provided vocabulary subset."""
    if not hidden.is_cuda or not weight_fp8.is_cuda or not scales.is_cuda or not token_ids.is_cuda:
        raise ValueError("triton_lm_head_shortlist_compact_top1_fp8 requires CUDA tensors")
    if hidden.dtype != torch.float16 or scales.dtype != torch.float16:
        raise ValueError("triton_lm_head_shortlist_compact_top1_fp8 expects fp16 hidden/scales")
    if weight_fp8.dtype != torch.uint8 or weight_fp8.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight_fp8 must be uint8 [shortlist, hidden] and scales [shortlist, hidden / 128]")
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a 1D torch.long CUDA tensor")

    hidden_flat = hidden.reshape(-1)
    shortlist_size, k_in = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_lm_head_shortlist_compact_top1_fp8 currently requires block_k=128")
    if shortlist_size <= 0 or int(token_ids.numel()) != shortlist_size:
        raise ValueError("compact weight and token_ids shortlist sizes do not match")
    if int(hidden_flat.numel()) != k_in or k_in % block_k != 0:
        raise ValueError("hidden size mismatch or not divisible by block_k")
    if int(scales.shape[0]) != shortlist_size or int(scales.shape[1]) != k_in // block_k:
        raise ValueError("scales shape does not match compact weight_fp8")

    if bias is not None:
        if not bias.is_cuda or bias.dtype != hidden.dtype:
            raise ValueError("bias must be fp16 CUDA tensor")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = hidden_flat

    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_token.dtype != torch.long:
        raise ValueError("out_token must have dtype torch.long")

    block_n = int(block_n)
    reduce_block = int(reduce_block)
    if workspace is None:
        workspace = create_lm_head_top1_workspace(
            device=hidden.device,
            n_vocab=shortlist_size,
            block_n=block_n,
            reduce_block=reduce_block,
        )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = triton.cdiv(shortlist_size, block_n)
    num_stage_blocks = triton.cdiv(num_blocks, reduce_block)

    _lm_head_shortlist_compact_block_top1_fp8_block128_kernel[(num_blocks,)](
        hidden_flat,
        weight_fp8,
        scales,
        token_ids,
        bias_flat,
        partial_values,
        partial_indices,
        shortlist_size,
        k_in,
        k_in // block_k,
        bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _lm_head_partial_reduce_kernel[(num_stage_blocks,)](
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        num_blocks,
        REDUCE_BLOCK=reduce_block,
        num_warps=int(reduce_num_warps),
        num_stages=1,
    )
    _lm_head_final_reduce_mapped_kernel[(1,)](
        stage_values,
        stage_indices,
        token_ids,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_stage_blocks,
        BLOCK_M=triton.next_power_of_2(num_stage_blocks),
        num_warps=1,
        num_stages=1,
    )
    return out_value, out_token
