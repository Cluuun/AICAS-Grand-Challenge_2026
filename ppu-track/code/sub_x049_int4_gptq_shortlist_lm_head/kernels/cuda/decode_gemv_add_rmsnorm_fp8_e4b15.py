from __future__ import annotations

import torch
import triton

from kernels.fused_decode_gemv_add_rmsnorm import _splitk_reduce_add_rmsnorm_full_kernel

from .decode_gemv_partial_fp8_e4b15 import (
    cuda_splitk_reduce_add_rmsnorm_2048,
    cuda_gemv_splitk_partial_fp8_e4b15_kblock_kscale,
    cuda_gemv_splitk_partial_fp8_e4b15_kblock_major,
    cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major_local_rowgroup,
    cuda_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup,
)
from .decode_gemv_partial_int4_sym import cuda_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp


def _reduce_add_rmsnorm(
    x: torch.Tensor,
    partial: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None,
    out_sum: torch.Tensor | None,
    out_norm: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_flat = residual.reshape(-1)
    n_out = int(residual_flat.numel())
    if partial.ndim != 2 or int(partial.shape[1]) != n_out:
        raise ValueError("partial must be [split_k, n_out]")
    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
        has_bias = True
    else:
        bias_flat = x.reshape(-1)
        has_bias = False

    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    _splitk_reduce_add_rmsnorm_full_kernel[(1,)](
        partial,
        bias_flat,
        residual_flat,
        norm_weight,
        out_sum_flat,
        out_norm_flat,
        n_out,
        has_bias,
        int(partial.shape[0]),
        float(eps),
        BLOCK_N=triton.next_power_of_2(n_out),
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


def _cuda_reduce_add_rmsnorm(
    partial: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None,
    out_sum: torch.Tensor | None,
    out_norm: torch.Tensor | None,
    reduce_threads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return cuda_splitk_reduce_add_rmsnorm_2048(
        partial,
        residual,
        norm_weight,
        eps,
        bias,
        out_sum=out_sum,
        out_norm=out_norm,
        threads=int(reduce_threads),
    )


def cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_kscale(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    *,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    block_n: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 3,
    variant: int = 0,
    reduce_backend: str = "triton",
    reduce_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA partial GEMV plus existing Triton reduce/add/RMSNorm for down projection ablation."""
    if int(block_n) != 16 or int(block_k) != 128 or int(num_warps) != 8:
        raise ValueError("CUDA FP8 kblock/kscale GEMV is specialized for BN=16, BK=128, warps=8")
    del num_stages
    partial = cuda_gemv_splitk_partial_fp8_e4b15_kblock_kscale(
        x,
        weight_fp8,
        scales,
        split_k=int(split_k),
        variant=int(variant),
    )
    if str(reduce_backend).strip().lower() == "cuda":
        return _cuda_reduce_add_rmsnorm(
            partial, residual, norm_weight, eps, bias, out_sum, out_norm, int(reduce_threads)
        )
    return _reduce_add_rmsnorm(x, partial, residual, norm_weight, eps, bias, out_sum, out_norm)


def cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_major(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    *,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    block_n: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
    split_k: int = 4,
    variant: int = 0,
    reduce_backend: str = "triton",
    reduce_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA partial GEMV plus existing Triton reduce/add/RMSNorm for o projection ablation."""
    if int(block_n) != 16 or int(block_k) != 128 or int(num_warps) != 8:
        raise ValueError("CUDA FP8 kblock-major GEMV is specialized for BN=16, BK=128, warps=8")
    del num_stages
    partial = cuda_gemv_splitk_partial_fp8_e4b15_kblock_major(
        x,
        weight_fp8,
        scales,
        split_k=int(split_k),
        variant=int(variant),
    )
    if str(reduce_backend).strip().lower() == "cuda":
        return _cuda_reduce_add_rmsnorm(
            partial, residual, norm_weight, eps, bias, out_sum, out_norm, int(reduce_threads)
        )
    return _reduce_add_rmsnorm(x, partial, residual, norm_weight, eps, bias, out_sum, out_norm)


def cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_e4b15_kblock_major(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    *,
    active_splits: int,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    block_n: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
    split_k: int = 4,
    variant: int = 32,
    reduce_backend: str = "triton",
    reduce_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused prefix-attention stage2 + O partial GEMV, followed by reduce/add/RMSNorm."""
    if int(block_n) != 16 or int(block_k) != 128 or int(num_warps) != 8:
        raise ValueError("CUDA prefix stage2/O fusion is specialized for BN=16, BK=128, warps=8")
    if int(num_stages) != 4:
        raise ValueError("CUDA prefix stage2/O fusion expects the kblock-major O FP8 stage count")
    if int(split_k) != 4:
        raise ValueError("CUDA prefix stage2/O fusion requires split_k=4")
    if int(variant) != 32:
        raise ValueError("CUDA prefix stage2/O fusion is paired with O FP8 variant 32")
    active_splits = int(active_splits)
    row_variant = -22
    _scratch_x, partial = cuda_prefix_stage2_o_partial_fp8_e4b15_kblock_major_local_rowgroup(
        partial_m,
        partial_l,
        partial_acc,
        weight_fp8,
        scales,
        active_splits=active_splits,
        split_k=4,
        scratch_x=None,
        partial=None,
        row_blocks_per_cta=row_variant,
    )
    if str(reduce_backend).strip().lower() == "cuda":
        return _cuda_reduce_add_rmsnorm(
            partial, residual, norm_weight, eps, bias, out_sum, out_norm, int(reduce_threads)
        )
    return _reduce_add_rmsnorm(residual, partial, residual, norm_weight, eps, bias, out_sum, out_norm)


def cuda_decode_prefix_stage2_o_add_rmsnorm_int4_sym_kblock_kscale_halfwarp(
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    partial_acc: torch.Tensor,
    weight_int4: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    *,
    active_splits: int,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    block_n: int = 2,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 4,
    row_blocks_per_cta: int = -22,
    reduce_backend: str = "triton",
    reduce_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused prefix-attention stage2 + O INT4 partial GEMV, followed by reduce/add/RMSNorm."""
    if int(block_k) != 128 or int(num_warps) != 8:
        raise ValueError("CUDA prefix stage2/O INT4 fusion is specialized for BK=128, warps=8")
    del block_n, num_stages
    active_splits = int(active_splits)
    _scratch_x, partial = cuda_prefix_stage2_o_partial_int4_sym_kblock_major_local_rowgroup(
        partial_m,
        partial_l,
        partial_acc,
        weight_int4,
        scales,
        active_splits=active_splits,
        split_k=int(split_k),
        scratch_x=None,
        partial=None,
        row_blocks_per_cta=int(row_blocks_per_cta),
    )
    if str(reduce_backend).strip().lower() == "cuda":
        return _cuda_reduce_add_rmsnorm(
            partial, residual, norm_weight, eps, bias, out_sum, out_norm, int(reduce_threads)
        )
    return _reduce_add_rmsnorm(residual, partial, residual, norm_weight, eps, bias, out_sum, out_norm)


def cuda_decode_gemv_add_rmsnorm_int4_sym_kblock_kscale_halfwarp(
    x: torch.Tensor,
    weight_int4: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    *,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    block_n: int = 4,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 4,
    reduce_backend: str = "cuda",
    reduce_threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA symmetric-int4 partial GEMV plus existing reduce/add/RMSNorm for down/o ablation."""
    if int(block_k) != 128 or int(num_warps) != 8:
        raise ValueError("CUDA int4 kblock/kscale GEMV is specialized for BK=128, warps=8")
    del num_stages
    partial = cuda_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp(
        x,
        weight_int4,
        scales,
        split_k=int(split_k),
        rows_per_block=int(block_n),
    )
    if str(reduce_backend).strip().lower() == "cuda":
        return _cuda_reduce_add_rmsnorm(
            partial, residual, norm_weight, eps, bias, out_sum, out_norm, int(reduce_threads)
        )
    return _reduce_add_rmsnorm(x, partial, residual, norm_weight, eps, bias, out_sum, out_norm)
