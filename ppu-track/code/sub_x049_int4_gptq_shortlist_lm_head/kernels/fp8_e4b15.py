from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import utils as cuda_utils


@triton.jit
def _quantize_e4b15_block128_kernel(
    weight_ptr,
    weight_fp8_ptr,
    scales_ptr,
    n_rows: tl.constexpr,
    k_in: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // (k_in // BLOCK_K)
    block_k = pid - row * (k_in // BLOCK_K)
    offs_k = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
    values = tl.load(weight_ptr + row * k_in + offs_k).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    scale = tl.maximum(absmax / 1.75, 1.0e-8)
    scaled = (values / scale).to(tl.float16)
    fp8 = cuda_utils.convert_float16_to_fp8e4b15(scaled, has_minx2=True)
    tl.store(weight_fp8_ptr + row * k_in + offs_k, fp8.to(tl.uint8, bitcast=True))
    tl.store(scales_ptr + row * (k_in // BLOCK_K) + block_k, scale)


def triton_quantize_e4b15_block128(
    weight: torch.Tensor,
    *,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp16 row-major weights to Triton fp8e4b15 bytes plus block scales."""
    if not weight.is_cuda or weight.dtype != torch.float16:
        raise ValueError("triton_quantize_e4b15_block128 expects fp16 CUDA weight")
    if weight.ndim != 2:
        raise ValueError("weight must be [N, K]")
    n_rows, k_in = int(weight.shape[0]), int(weight.shape[1])
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_quantize_e4b15_block128 currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")

    weight = weight.contiguous()
    weight_fp8 = torch.empty_like(weight, dtype=torch.uint8)
    scales = torch.empty((n_rows, k_in // block_k), device=weight.device, dtype=torch.float16)
    _quantize_e4b15_block128_kernel[(n_rows * (k_in // block_k),)](
        weight,
        weight_fp8,
        scales,
        n_rows,
        k_in,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return weight_fp8, scales


@triton.jit
def _dequant_e4b15_block128_kernel(
    weight_fp8_ptr,
    scales_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    k_in: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_m[:, None] < n_rows
    packed = tl.load(weight_fp8_ptr + offs_m[:, None] * k_in + offs_k[None, :], mask=mask, other=0)
    values = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
    scale = tl.load(scales_ptr + offs_m * (k_in // BLOCK_K) + pid_k, mask=offs_m < n_rows, other=0.0)
    tl.store(out_ptr + offs_m[:, None] * k_in + offs_k[None, :], values * scale[:, None], mask=mask)


def triton_dequant_e4b15_block128(
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_m: int = 8,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """Materialize fp8e4b15 block-128 weights as fp16 for diagnostics."""
    if not weight_fp8.is_cuda or not scales.is_cuda:
        raise ValueError("triton_dequant_e4b15_block128 requires CUDA tensors")
    if weight_fp8.dtype != torch.uint8 or scales.dtype != torch.float16:
        raise ValueError("expected uint8 fp8e4b15 storage and fp16 scales")
    if weight_fp8.ndim != 2 or scales.ndim != 2:
        raise ValueError("expected weight [N, K] and scales [N, K / 128]")
    n_rows, k_in = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
    block_k = int(block_k)
    if block_k != 128 or k_in % block_k != 0:
        raise ValueError("dequant currently requires block_k=128 and divisible K")
    if int(scales.shape[0]) != n_rows or int(scales.shape[1]) != k_in // block_k:
        raise ValueError("scales shape does not match weight")

    out = torch.empty((n_rows, k_in), device=weight_fp8.device, dtype=torch.float16)
    block_m = int(block_m)
    _dequant_e4b15_block128_kernel[(triton.cdiv(n_rows, block_m), k_in // block_k)](
        weight_fp8,
        scales,
        out,
        n_rows,
        k_in,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out
