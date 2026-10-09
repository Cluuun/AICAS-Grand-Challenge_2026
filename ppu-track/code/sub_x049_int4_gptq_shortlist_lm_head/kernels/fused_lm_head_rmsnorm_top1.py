from __future__ import annotations

import torch
import triton
import triton.language as tl

from kernels.fused_lm_head_top1 import (
    _lm_head_final_reduce_kernel,
    _lm_head_partial_reduce_kernel,
    create_lm_head_top1_workspace,
)


@triton.jit
def _lm_head_shortlist_rmsnorm_block_top1_kernel(
    x_ptr,
    rms_w_ptr,
    lm_w_ptr,
    token_ids_ptr,
    bias_ptr,
    partial_value_ptr,
    partial_index_ptr,
    shortlist_size: tl.constexpr,
    k_in: tl.constexpr,
    eps,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_s = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_s = offs_s < shortlist_size
    token_ids = tl.load(token_ids_ptr + offs_s, mask=mask_s, other=0)

    sumsq = tl.full((), 0.0, tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    rstd = tl.rsqrt(sumsq / k_in + eps)

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        rms_w = tl.load(rms_w_ptr + offs_k)
        norm_x = ((x * rstd).to(tl.float16) * rms_w).to(tl.float16)
        lm_w = tl.load(
            lm_w_ptr + token_ids[:, None] * k_in + offs_k[None, :],
            mask=mask_s[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(lm_w * norm_x[None, :].to(tl.float32), axis=1)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + token_ids, mask=mask_s, other=0.0).to(tl.float32)

    acc = tl.where(mask_s, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, token_ids, 2147483647), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


@triton.jit
def _lm_head_shortlist_scaled_gather_block_top1_kernel(
    x_ptr,
    rms_w_ptr,
    lm_w_ptr,
    token_ids_ptr,
    partial_value_ptr,
    partial_index_ptr,
    shortlist_size: tl.constexpr,
    k_in: tl.constexpr,
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
        rms_w = tl.load(rms_w_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        lm_w = tl.load(
            lm_w_ptr + token_ids[:, None] * k_in + offs_k[None, :],
            mask=mask_s[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(lm_w * (x * rms_w)[None, :], axis=1)

    acc = tl.where(mask_s, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, token_ids, 2147483647), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


@triton.jit
def _lm_head_shortlist_folded_block_top1_kernel(
    x_ptr,
    folded_w_ptr,
    token_ids_ptr,
    partial_value_ptr,
    partial_index_ptr,
    shortlist_size: tl.constexpr,
    k_in: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_s = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_s = offs_s < shortlist_size

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        folded_w = tl.load(
            folded_w_ptr + offs_s[:, None] * k_in + offs_k[None, :],
            mask=mask_s[:, None],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(folded_w * x[None, :], axis=1)

    token_ids = tl.load(token_ids_ptr + offs_s, mask=mask_s, other=2147483647)
    acc = tl.where(mask_s, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, token_ids, 2147483647), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


@triton.jit
def _lm_head_shortlist_folded_colmajor_block_top1_kernel(
    x_ptr,
    folded_w_ptr,
    token_ids_ptr,
    partial_value_ptr,
    partial_index_ptr,
    shortlist_size: tl.constexpr,
    k_in: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_s = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_s = offs_s < shortlist_size

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        folded_w = tl.load(
            folded_w_ptr + offs_k[:, None] * shortlist_size + offs_s[None, :],
            mask=mask_s[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(folded_w * x[:, None], axis=0)

    token_ids = tl.load(token_ids_ptr + offs_s, mask=mask_s, other=2147483647)
    acc = tl.where(mask_s, acc, -float("inf"))
    best_value = tl.max(acc, axis=0)
    best_index = tl.min(tl.where(acc == best_value, token_ids, 2147483647), axis=0)
    tl.store(partial_value_ptr + pid_n, best_value)
    tl.store(partial_index_ptr + pid_n, best_index)


def create_folded_lm_head_shortlist_weight(
    lm_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    if not (lm_weight.is_cuda and rms_weight.is_cuda and token_ids.is_cuda):
        raise ValueError("folded lm_head shortlist weight requires CUDA tensors")
    if lm_weight.dtype != torch.float16 or rms_weight.dtype != torch.float16:
        raise ValueError("lm_weight and rms_weight must be fp16")
    if lm_weight.ndim != 2 or rms_weight.ndim != 1:
        raise ValueError("lm_weight must be [vocab, hidden] and rms_weight must be [hidden]")
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a 1D torch.long CUDA tensor")
    if int(lm_weight.shape[1]) != int(rms_weight.numel()):
        raise ValueError("rms_weight size must match lm_weight hidden size")

    selected = torch.index_select(lm_weight, 0, token_ids)
    return (selected.float() * rms_weight.float()).to(torch.float16).contiguous()


def create_folded_lm_head_shortlist_weight_colmajor(
    lm_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    return create_folded_lm_head_shortlist_weight(lm_weight, rms_weight, token_ids).t().contiguous()


def _validate_folded_top1_inputs(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    shortlist_size: int,
    k_in: int,
    *,
    block_n: int,
    block_k: int,
    reduce_block: int,
) -> torch.Tensor:
    if not (hidden.is_cuda and token_ids.is_cuda):
        raise ValueError("folded top1 requires CUDA tensors")
    if hidden.dtype != torch.float16:
        raise ValueError("hidden must be fp16")
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a 1D torch.long CUDA tensor")

    hidden_flat = hidden.reshape(-1)
    if int(hidden_flat.numel()) != k_in:
        raise ValueError(f"hidden has {hidden_flat.numel()} elements, expected {k_in}")
    if int(token_ids.numel()) != shortlist_size:
        raise ValueError("token_ids size must match shortlist weight rows")
    if shortlist_size <= 0:
        raise ValueError("shortlist must not be empty")
    if k_in % int(block_k) != 0:
        raise ValueError("hidden size must be divisible by block_k")
    if int(block_n) <= 0 or int(reduce_block) <= 0:
        raise ValueError("block_n and reduce_block must be positive")
    return hidden_flat


def _reduce_lm_head_partials(
    partial_values: torch.Tensor,
    partial_indices: torch.Tensor,
    stage_values: torch.Tensor,
    stage_indices: torch.Tensor,
    out_value: torch.Tensor,
    out_token: torch.Tensor,
    num_blocks: int,
    reduce_block: int,
    reduce_num_warps: int,
    *,
    direct_reduce: bool,
) -> None:
    if direct_reduce:
        _lm_head_final_reduce_kernel[(1,)](
            partial_values,
            partial_indices,
            out_value.reshape(-1),
            out_token.reshape(-1),
            int(num_blocks),
            BLOCK_M=triton.next_power_of_2(int(num_blocks)),
            num_warps=1,
            num_stages=1,
        )
        return

    num_stage_blocks = triton.cdiv(int(num_blocks), int(reduce_block))
    _lm_head_partial_reduce_kernel[(num_stage_blocks,)](
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        int(num_blocks),
        REDUCE_BLOCK=int(reduce_block),
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


def triton_lm_head_shortlist_rmsnorm_top1(
    hidden: torch.Tensor,
    rms_weight: torch.Tensor,
    lm_weight: torch.Tensor,
    token_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
    out_value: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_n: int = 32,
    block_k: int = 128,
    reduce_block: int = 512,
    num_warps: int = 4,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
    direct_reduce: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (hidden.is_cuda and rms_weight.is_cuda and lm_weight.is_cuda and token_ids.is_cuda):
        raise ValueError("triton_lm_head_shortlist_rmsnorm_top1 requires CUDA tensors")
    if hidden.dtype != torch.float16 or rms_weight.dtype != torch.float16 or lm_weight.dtype != torch.float16:
        raise ValueError("hidden, rms_weight and lm_weight must be fp16")
    if lm_weight.ndim != 2:
        raise ValueError("lm_weight must be [vocab, hidden]")
    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must be a 1D torch.long CUDA tensor")

    hidden_flat = hidden.reshape(-1)
    n_vocab, k_in = int(lm_weight.shape[0]), int(lm_weight.shape[1])
    shortlist_size = int(token_ids.numel())
    if shortlist_size <= 0:
        raise ValueError("token_ids must not be empty")
    if int(hidden_flat.numel()) != k_in:
        raise ValueError(f"hidden has {hidden_flat.numel()} elements, expected {k_in}")
    if int(rms_weight.numel()) != k_in:
        raise ValueError(f"rms_weight has {rms_weight.numel()} elements, expected {k_in}")

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
        raise ValueError("hidden size must be divisible by block_k")
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

    _lm_head_shortlist_rmsnorm_block_top1_kernel[(num_blocks,)](
        hidden_flat,
        rms_weight,
        lm_weight,
        token_ids,
        bias_flat,
        partial_values,
        partial_indices,
        shortlist_size,
        k_in,
        float(eps),
        bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _reduce_lm_head_partials(
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_blocks,
        reduce_block,
        reduce_num_warps,
        direct_reduce=bool(direct_reduce),
    )
    return out_value, out_token


def triton_lm_head_shortlist_scaled_gather_top1(
    hidden: torch.Tensor,
    rms_weight: torch.Tensor,
    lm_weight: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    out_value: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_n: int = 32,
    block_k: int = 128,
    reduce_block: int = 512,
    num_warps: int = 4,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
    direct_reduce: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (rms_weight.is_cuda and lm_weight.is_cuda):
        raise ValueError("scaled-gather top1 requires CUDA tensors")
    if rms_weight.dtype != torch.float16 or lm_weight.dtype != torch.float16:
        raise ValueError("rms_weight and lm_weight must be fp16")
    if lm_weight.ndim != 2:
        raise ValueError("lm_weight must be [vocab, hidden]")

    n_vocab, k_in = int(lm_weight.shape[0]), int(lm_weight.shape[1])
    if int(rms_weight.numel()) != k_in:
        raise ValueError("rms_weight size must match lm_weight hidden size")
    hidden_flat = _validate_folded_top1_inputs(
        hidden,
        token_ids,
        int(token_ids.numel()),
        k_in,
        block_n=int(block_n),
        block_k=int(block_k),
        reduce_block=int(reduce_block),
    )
    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_token.dtype != torch.long:
        raise ValueError("out_token must have dtype torch.long")

    shortlist_size = int(token_ids.numel())
    if workspace is None:
        workspace = create_lm_head_top1_workspace(
            device=hidden.device,
            n_vocab=shortlist_size,
            block_n=int(block_n),
            reduce_block=int(reduce_block),
        )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = triton.cdiv(shortlist_size, int(block_n))

    _lm_head_shortlist_scaled_gather_block_top1_kernel[(num_blocks,)](
        hidden_flat,
        rms_weight,
        lm_weight,
        token_ids,
        partial_values,
        partial_indices,
        shortlist_size,
        k_in,
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _reduce_lm_head_partials(
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_blocks,
        reduce_block,
        reduce_num_warps,
        direct_reduce=bool(direct_reduce),
    )
    return out_value, out_token


def triton_lm_head_shortlist_folded_top1(
    hidden: torch.Tensor,
    folded_shortlist_weight: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    out_value: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_n: int = 32,
    block_k: int = 128,
    reduce_block: int = 512,
    num_warps: int = 4,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
    direct_reduce: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not folded_shortlist_weight.is_cuda:
        raise ValueError("folded_shortlist_weight must be a CUDA tensor")
    if folded_shortlist_weight.dtype != torch.float16 or folded_shortlist_weight.ndim != 2:
        raise ValueError("folded_shortlist_weight must be fp16 [shortlist, hidden]")
    if not folded_shortlist_weight.is_contiguous():
        raise ValueError("folded_shortlist_weight must be contiguous")

    shortlist_size, k_in = int(folded_shortlist_weight.shape[0]), int(folded_shortlist_weight.shape[1])
    hidden_flat = _validate_folded_top1_inputs(
        hidden,
        token_ids,
        shortlist_size,
        k_in,
        block_n=int(block_n),
        block_k=int(block_k),
        reduce_block=int(reduce_block),
    )

    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_token.dtype != torch.long:
        raise ValueError("out_token must have dtype torch.long")

    if workspace is None:
        workspace = create_lm_head_top1_workspace(
            device=hidden.device,
            n_vocab=shortlist_size,
            block_n=int(block_n),
            reduce_block=int(reduce_block),
        )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = triton.cdiv(shortlist_size, int(block_n))

    _lm_head_shortlist_folded_block_top1_kernel[(num_blocks,)](
        hidden_flat,
        folded_shortlist_weight,
        token_ids,
        partial_values,
        partial_indices,
        shortlist_size,
        k_in,
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _reduce_lm_head_partials(
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_blocks,
        reduce_block,
        reduce_num_warps,
        direct_reduce=bool(direct_reduce),
    )
    return out_value, out_token


def triton_lm_head_shortlist_folded_colmajor_top1(
    hidden: torch.Tensor,
    folded_shortlist_weight_colmajor: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    out_value: torch.Tensor | None = None,
    out_token: torch.Tensor | None = None,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_n: int = 32,
    block_k: int = 128,
    reduce_block: int = 512,
    num_warps: int = 4,
    num_stages: int = 3,
    reduce_num_warps: int = 8,
    direct_reduce: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not folded_shortlist_weight_colmajor.is_cuda:
        raise ValueError("folded_shortlist_weight_colmajor must be a CUDA tensor")
    if folded_shortlist_weight_colmajor.dtype != torch.float16 or folded_shortlist_weight_colmajor.ndim != 2:
        raise ValueError("folded_shortlist_weight_colmajor must be fp16 [hidden, shortlist]")
    if not folded_shortlist_weight_colmajor.is_contiguous():
        raise ValueError("folded_shortlist_weight_colmajor must be contiguous")

    k_in, shortlist_size = (
        int(folded_shortlist_weight_colmajor.shape[0]),
        int(folded_shortlist_weight_colmajor.shape[1]),
    )
    hidden_flat = _validate_folded_top1_inputs(
        hidden,
        token_ids,
        shortlist_size,
        k_in,
        block_n=int(block_n),
        block_k=int(block_k),
        reduce_block=int(reduce_block),
    )

    if out_value is None:
        out_value = torch.empty((1, 1), device=hidden.device, dtype=hidden.dtype)
    if out_token is None:
        out_token = torch.empty((1, 1), device=hidden.device, dtype=torch.long)
    if out_token.dtype != torch.long:
        raise ValueError("out_token must have dtype torch.long")

    if workspace is None:
        workspace = create_lm_head_top1_workspace(
            device=hidden.device,
            n_vocab=shortlist_size,
            block_n=int(block_n),
            reduce_block=int(reduce_block),
        )
    partial_values, partial_indices, stage_values, stage_indices = workspace
    num_blocks = triton.cdiv(shortlist_size, int(block_n))

    _lm_head_shortlist_folded_colmajor_block_top1_kernel[(num_blocks,)](
        hidden_flat,
        folded_shortlist_weight_colmajor,
        token_ids,
        partial_values,
        partial_indices,
        shortlist_size,
        k_in,
        BLOCK_N=int(block_n),
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _reduce_lm_head_partials(
        partial_values,
        partial_indices,
        stage_values,
        stage_indices,
        out_value.reshape(-1),
        out_token.reshape(-1),
        num_blocks,
        reduce_block,
        reduce_num_warps,
        direct_reduce=bool(direct_reduce),
    )
    return out_value, out_token
