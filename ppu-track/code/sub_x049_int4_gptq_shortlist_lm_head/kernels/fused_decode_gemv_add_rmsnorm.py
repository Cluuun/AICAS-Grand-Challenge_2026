import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import utils as cuda_utils


@triton.jit
def _gemv_splitk_partial_kernel(
    x_ptr,
    w_ptr,
    partial_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    split_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    chunk_k = k_in // split_k
    k_begin = pid_k * chunk_k
    k_end = k_begin + chunk_k

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            x_ptr + offs_k,
            eviction_policy="evict_last",
        ).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * k_in + offs_k[None, :],
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(partial_ptr + pid_k * n_out + offs_n, acc)


@triton.jit
def _gemv_splitk_partial_fp8_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    partial_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    split_k: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    chunk_k = k_in // split_k
    k_begin = pid_k * chunk_k
    k_end = k_begin + chunk_k

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        scale_idx: tl.constexpr = k0 // BLOCK_K
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        packed = tl.load(
            w_fp8_ptr + offs_n[:, None] * k_in + offs_k[None, :],
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + offs_n * scale_blocks + scale_idx,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    tl.store(partial_ptr + pid_k * n_out + offs_n, acc)


@triton.jit
def _gemv_splitk_partial_fp8_kblock_major_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    partial_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    split_k: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    chunk_k = k_in // split_k
    k_begin = pid_k * chunk_k
    k_end = k_begin + chunk_k

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        scale_idx: tl.constexpr = k0 // BLOCK_K
        offs_k = k0 + offs_inner
        block_ids = offs_k // BLOCK_K
        inner = offs_k - block_ids * BLOCK_K
        x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
        row_offsets = offs_n * BLOCK_K
        packed = tl.load(
            w_fp8_ptr + block_ids[None, :] * (n_out * BLOCK_K) + row_offsets[:, None] + inner[None, :],
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + offs_n * scale_blocks + scale_idx,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    tl.store(partial_ptr + pid_k * n_out + offs_n, acc)


@triton.jit
def _gemv_add_fp8_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    bias_ptr,
    residual_ptr,
    sum_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_out

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
    residual = tl.load(residual_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    summed = (acc + residual).to(tl.float16, fp_downcast_rounding="rtne")
    tl.store(sum_ptr + offs_n, summed, mask=mask_n)


@triton.jit
def _splitk_reduce_add_rmsnorm_full_kernel(
    partial_ptr,
    bias_ptr,
    residual_ptr,
    norm_weight_ptr,
    sum_ptr,
    norm_ptr,
    n_out: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    split_k: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_out
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for s in range(0, split_k):
        acc += tl.load(partial_ptr + s * n_out + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    residual = tl.load(residual_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    summed = (acc + residual).to(tl.float16, fp_downcast_rounding="rtne")
    summed_fp = summed.to(tl.float32)
    total_sumsq = tl.sum(tl.where(mask_n, summed_fp * summed_fp, 0.0), axis=0)
    rstd = tl.rsqrt(total_sumsq / n_out + eps)
    weight = tl.load(norm_weight_ptr + offs_n, mask=mask_n, other=0.0)
    normed = (summed_fp * rstd).to(tl.float16, fp_downcast_rounding="rtne") * weight
    tl.store(sum_ptr + offs_n, summed, mask=mask_n)
    tl.store(norm_ptr + offs_n, normed, mask=mask_n)


@triton.jit
def _rmsnorm_full_kernel(
    sum_ptr,
    norm_weight_ptr,
    norm_ptr,
    n_out: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_out
    summed = tl.load(sum_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    total_sumsq = tl.sum(tl.where(mask_n, summed * summed, 0.0), axis=0)
    rstd = tl.rsqrt(total_sumsq / n_out + eps)
    weight = tl.load(norm_weight_ptr + offs_n, mask=mask_n, other=0.0)
    normed = (summed * rstd).to(tl.float16, fp_downcast_rounding="rtne") * weight
    tl.store(norm_ptr + offs_n, normed, mask=mask_n)


def triton_decode_gemv_add_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
    *,
    out_sum: torch.Tensor | None = None,
    out_norm: torch.Tensor | None = None,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    mode: str = "splitk-fullnorm",
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp16 split-K GEMV followed by residual add and RMSNorm."""
    if str(mode) != "splitk-fullnorm":
        raise ValueError("triton_decode_gemv_add_rmsnorm only supports splitk-fullnorm")
    if not (x.is_cuda and weight.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm requires CUDA tensors")
    if x.dtype != torch.float16 or weight.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm expects fp16 x/weight/residual")
    if weight.ndim != 2:
        raise ValueError("weight must be [N, K]")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")
    if int(residual_flat.numel()) != n_out:
        raise ValueError(f"residual has {residual_flat.numel()} elements, expected {n_out}")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    block_n = int(block_n)
    block_k = int(block_k)
    split_k = int(split_k)
    if split_k <= 1:
        raise ValueError("splitk-fullnorm requires split_k > 1")
    if n_out > 8192:
        raise ValueError("splitk-fullnorm is intended for one full vector up to 8192 elements")
    if n_out % block_n != 0 or k_in % split_k != 0 or (k_in // split_k) % block_k != 0:
        raise ValueError(
            "splitk-fullnorm requires n_out % block_n == 0 and each split-K chunk divisible by block_k"
        )

    num_blocks = triton.cdiv(n_out, block_n)
    partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    _gemv_splitk_partial_kernel[(num_blocks, split_k)](
        x_flat,
        weight,
        partial,
        n_out,
        k_in,
        split_k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    block_full = triton.next_power_of_2(n_out)
    _splitk_reduce_add_rmsnorm_full_kernel[(1,)](
        partial,
        bias_flat,
        residual_flat,
        norm_weight,
        out_sum_flat,
        out_norm_flat,
        n_out,
        bias is not None,
        split_k,
        float(eps),
        BLOCK_N=block_full,
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


def triton_decode_gemv_add_rmsnorm_fp8_nosplit(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 GEMV/add followed by separate full-vector RMSNorm."""
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_nosplit requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_nosplit expects fp16 x/scales/residual")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")
    if weight_fp8.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight_fp8 must be [N, K] and scales [N, K / 128]")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out, k_in = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")
    if int(residual_flat.numel()) != n_out:
        raise ValueError(f"residual has {residual_flat.numel()} elements, expected {n_out}")
    if int(scales.shape[0]) != n_out or int(scales.shape[1]) != k_in // 128:
        raise ValueError("scales shape does not match weight_fp8")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    block_n = int(block_n)
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_nosplit currently requires block_k=128")
    if n_out % block_n != 0 or k_in % block_k != 0:
        raise ValueError("fp8 nosplit requires n_out % block_n == 0 and K divisible by block_k")

    _gemv_add_fp8_block128_kernel[(triton.cdiv(n_out, block_n),)](
        x_flat,
        weight_fp8,
        scales,
        bias_flat,
        residual_flat,
        out_sum_flat,
        n_out,
        k_in,
        k_in // block_k,
        bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _rmsnorm_full_kernel[(1,)](
        out_sum_flat,
        norm_weight,
        out_norm_flat,
        n_out,
        float(eps),
        BLOCK_N=triton.next_power_of_2(n_out),
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


def triton_decode_gemv_add_rmsnorm_fp8(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 split-K GEMV followed by residual add and RMSNorm."""
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8 requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8 expects fp16 x/scales/residual")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")
    if weight_fp8.ndim != 2 or scales.ndim != 2:
        raise ValueError("weight_fp8 must be [N, K] and scales [N, K / 128]")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out, k_in = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")
    if int(residual_flat.numel()) != n_out:
        raise ValueError(f"residual has {residual_flat.numel()} elements, expected {n_out}")
    if int(scales.shape[0]) != n_out or int(scales.shape[1]) != k_in // 128:
        raise ValueError("scales shape does not match weight_fp8")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    block_n = int(block_n)
    block_k = int(block_k)
    split_k = int(split_k)
    if block_k != 128:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8 currently requires block_k=128")
    if split_k <= 1:
        raise ValueError("fp8 split-K requires split_k > 1")
    if n_out > 8192:
        raise ValueError("fp8 split-K fullnorm is intended for one full vector up to 8192 elements")
    if n_out % block_n != 0 or k_in % split_k != 0 or (k_in // split_k) % block_k != 0:
        raise ValueError(
            "fp8 split-K requires n_out % block_n == 0 and each split-K chunk divisible by block_k"
        )

    partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    _gemv_splitk_partial_fp8_block128_kernel[(triton.cdiv(n_out, block_n), split_k)](
        x_flat,
        weight_fp8,
        scales,
        partial,
        n_out,
        k_in,
        split_k,
        k_in // block_k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    block_full = triton.next_power_of_2(n_out)
    _splitk_reduce_add_rmsnorm_full_kernel[(1,)](
        partial,
        bias_flat,
        residual_flat,
        norm_weight,
        out_sum_flat,
        out_norm_flat,
        n_out,
        bias is not None,
        split_k,
        float(eps),
        BLOCK_N=block_full,
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


def triton_decode_gemv_add_rmsnorm_fp8_kblock_major(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 split-K GEMV over [K/128, N, 128] packed weights."""
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_major requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_major expects fp16 x/scales/residual")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out = int(residual_flat.numel())
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    split_k = int(split_k)
    if block_k != 128:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_major currently requires block_k=128")
    if split_k <= 1:
        raise ValueError("fp8 k-block-major split-K requires split_k > 1")
    if n_out > 8192:
        raise ValueError("fp8 k-block-major fullnorm is intended for one full vector up to 8192 elements")
    if n_out % block_n != 0 or k_in % split_k != 0 or (k_in // split_k) % block_k != 0:
        raise ValueError(
            "fp8 k-block-major requires n_out % block_n == 0 and each split-K chunk divisible by block_k"
        )
    scale_blocks = k_in // block_k
    if int(weight_fp8.numel()) != scale_blocks * n_out * block_k:
        raise ValueError("weight_fp8 must contain [K/128, N, 128]")
    if int(scales.shape[0]) != n_out or int(scales.shape[1]) != scale_blocks:
        raise ValueError("scales must be [N, K / 128]")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    _gemv_splitk_partial_fp8_kblock_major_block128_kernel[(triton.cdiv(n_out, block_n), split_k)](
        x_flat,
        weight_fp8,
        scales,
        partial,
        n_out,
        k_in,
        split_k,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    block_full = triton.next_power_of_2(n_out)
    _splitk_reduce_add_rmsnorm_full_kernel[(1,)](
        partial,
        bias_flat,
        residual_flat,
        norm_weight,
        out_sum_flat,
        out_norm_flat,
        n_out,
        bias is not None,
        split_k,
        float(eps),
        BLOCK_N=block_full,
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


@triton.jit
def _gemv_add_fp8_kblock_major_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    bias_ptr,
    residual_ptr,
    sum_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    kblock_stride = n_out * BLOCK_K

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        block_id = k0 // BLOCK_K
        x = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last").to(tl.float32)
        packed = tl.load(
            w_fp8_ptr + block_id * kblock_stride + row_offsets[:, None] + offs_inner[None, :],
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + offs_n * scale_blocks + block_id,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n).to(tl.float32)
    residual = tl.load(residual_ptr + offs_n).to(tl.float32)
    summed = (acc + residual).to(tl.float16, fp_downcast_rounding="rtne")
    tl.store(sum_ptr + offs_n, summed)


@triton.jit
def _gemv_add_fp8_kblock_kscale_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    bias_ptr,
    residual_ptr,
    sum_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    kblock_stride = n_out * BLOCK_K

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        block_id = k0 // BLOCK_K
        x = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last").to(tl.float32)
        packed = tl.load(
            w_fp8_ptr + block_id * kblock_stride + row_offsets[:, None] + offs_inner[None, :],
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + block_id * n_out + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n).to(tl.float32)
    residual = tl.load(residual_ptr + offs_n).to(tl.float32)
    summed = (acc + residual).to(tl.float16, fp_downcast_rounding="rtne")
    tl.store(sum_ptr + offs_n, summed)


def _triton_decode_gemv_add_rmsnorm_fp8_kblock_nosplit(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None,
    *,
    out_sum: torch.Tensor | None,
    out_norm: torch.Tensor | None,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    kscale: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_nosplit requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_nosplit expects fp16 x/scales/residual")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out = int(residual_flat.numel())
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("fp8 kblock nosplit currently requires block_k=128")
    if n_out > 8192:
        raise ValueError("fp8 kblock nosplit fullnorm is intended for one full vector up to 8192 elements")
    if n_out % block_n != 0 or k_in % block_k != 0:
        raise ValueError("fp8 kblock nosplit requires n_out % block_n == 0 and K divisible by block_k")
    scale_blocks = k_in // block_k
    if int(weight_fp8.numel()) != scale_blocks * n_out * block_k:
        raise ValueError("weight_fp8 must contain [K/128, N, 128]")
    if kscale:
        if int(scales.numel()) != scale_blocks * n_out:
            raise ValueError("scales must contain [K / 128, N]")
    elif int(scales.shape[0]) != n_out or int(scales.shape[1]) != scale_blocks:
        raise ValueError("scales must be [N, K / 128]")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    kernel = _gemv_add_fp8_kblock_kscale_block128_kernel if kscale else _gemv_add_fp8_kblock_major_block128_kernel
    kernel[(triton.cdiv(n_out, block_n),)](
        x_flat,
        weight_fp8,
        scales,
        bias_flat,
        residual_flat,
        out_sum_flat,
        n_out,
        k_in,
        scale_blocks,
        bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    _rmsnorm_full_kernel[(1,)](
        out_sum_flat,
        norm_weight,
        out_norm_flat,
        n_out,
        float(eps),
        BLOCK_N=triton.next_power_of_2(n_out),
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


def triton_decode_gemv_add_rmsnorm_fp8_kblock_major_nosplit(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 single-pass GEMV/add over [K/128, N, 128] weights plus separate RMSNorm."""
    return _triton_decode_gemv_add_rmsnorm_fp8_kblock_nosplit(
        x,
        weight_fp8,
        scales,
        residual,
        norm_weight,
        eps,
        bias,
        out_sum=out_sum,
        out_norm=out_norm,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        kscale=False,
    )


def triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale_nosplit(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 single-pass GEMV/add over kblock weights/scales plus separate RMSNorm."""
    return _triton_decode_gemv_add_rmsnorm_fp8_kblock_nosplit(
        x,
        weight_fp8,
        scales,
        residual,
        norm_weight,
        eps,
        bias,
        out_sum=out_sum,
        out_norm=out_norm,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        kscale=True,
    )


@triton.jit
def _gemv_splitk_partial_fp8_kblock_major_unroll_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    partial_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    split_k: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCKS_PER_ITER: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    chunk_k = k_in // split_k
    k_begin = pid_k * chunk_k
    k_end = k_begin + chunk_k
    kblock_stride = n_out * BLOCK_K

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K * BLOCKS_PER_ITER):
        for u in tl.static_range(0, BLOCKS_PER_ITER):
            block_id = k0 // BLOCK_K + u
            offs_k = k0 + u * BLOCK_K + offs_inner
            x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
            packed = tl.load(
                w_fp8_ptr + block_id * kblock_stride + row_offsets[:, None] + offs_inner[None, :],
                eviction_policy="evict_first",
            )
            w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
            scale = tl.load(
                scales_ptr + offs_n * scale_blocks + block_id,
                eviction_policy="evict_first",
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1) * scale

    tl.store(partial_ptr + pid_k * n_out + offs_n, acc)


def _triton_decode_gemv_add_rmsnorm_fp8_kblock_major_unroll(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None,
    *,
    out_sum: torch.Tensor | None,
    out_norm: torch.Tensor | None,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    blocks_per_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_major_unroll requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_major_unroll expects fp16 x/scales/residual")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out = int(residual_flat.numel())
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    split_k = int(split_k)
    blocks_per_iter = int(blocks_per_iter)
    if block_k != 128:
        raise ValueError("fp8 k-block-major unroll currently requires block_k=128")
    if split_k <= 1:
        raise ValueError("fp8 k-block-major unroll requires split_k > 1")
    if blocks_per_iter <= 0:
        raise ValueError("blocks_per_iter must be positive")
    if n_out > 8192:
        raise ValueError("fp8 k-block-major unroll fullnorm is intended for one full vector up to 8192 elements")
    if (
        n_out % block_n != 0
        or k_in % split_k != 0
        or (k_in // split_k) % (block_k * blocks_per_iter) != 0
    ):
        raise ValueError(
            "fp8 k-block-major unroll requires n_out % block_n == 0 and each split-K chunk divisible by "
            "block_k * blocks_per_iter"
        )
    scale_blocks = k_in // block_k
    if int(weight_fp8.numel()) != scale_blocks * n_out * block_k:
        raise ValueError("weight_fp8 must contain [K/128, N, 128]")
    if int(scales.shape[0]) != n_out or int(scales.shape[1]) != scale_blocks:
        raise ValueError("scales must be [N, K / 128]")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    _gemv_splitk_partial_fp8_kblock_major_unroll_block128_kernel[
        (triton.cdiv(n_out, block_n), split_k)
    ](
        x_flat,
        weight_fp8,
        scales,
        partial,
        n_out,
        k_in,
        split_k,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCKS_PER_ITER=blocks_per_iter,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    block_full = triton.next_power_of_2(n_out)
    _splitk_reduce_add_rmsnorm_full_kernel[(1,)](
        partial,
        bias_flat,
        residual_flat,
        norm_weight,
        out_sum_flat,
        out_norm_flat,
        n_out,
        bias is not None,
        split_k,
        float(eps),
        BLOCK_N=block_full,
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


def triton_decode_gemv_add_rmsnorm_fp8_kblock_major_fast(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 split-K GEMV over k-block-major weights with simplified addressing."""
    return _triton_decode_gemv_add_rmsnorm_fp8_kblock_major_unroll(
        x,
        weight_fp8,
        scales,
        residual,
        norm_weight,
        eps,
        bias,
        out_sum=out_sum,
        out_norm=out_norm,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
        blocks_per_iter=1,
    )


def triton_decode_gemv_add_rmsnorm_fp8_kblock_major_u2(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 split-K GEMV over k-block-major weights, unrolled by two K blocks."""
    return _triton_decode_gemv_add_rmsnorm_fp8_kblock_major_unroll(
        x,
        weight_fp8,
        scales,
        residual,
        norm_weight,
        eps,
        bias,
        out_sum=out_sum,
        out_norm=out_norm,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
        blocks_per_iter=2,
    )


@triton.jit
def _gemv_splitk_partial_fp8_kblock_kscale_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    partial_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    split_k: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    chunk_k = k_in // split_k
    k_begin = pid_k * chunk_k
    k_end = k_begin + chunk_k
    kblock_stride = n_out * BLOCK_K

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        block_id = k0 // BLOCK_K
        x = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last").to(tl.float32)
        packed = tl.load(
            w_fp8_ptr + block_id * kblock_stride + row_offsets[:, None] + offs_inner[None, :],
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + block_id * n_out + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    tl.store(partial_ptr + pid_k * n_out + offs_n, acc)


def triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 split-K GEMV over [K/128, N, 128] weights and [K/128, N] scales."""
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale expects fp16 x/scales/residual")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out = int(residual_flat.numel())
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    split_k = int(split_k)
    if block_k != 128:
        raise ValueError("fp8 kblock+kscale currently requires block_k=128")
    if split_k <= 1:
        raise ValueError("fp8 kblock+kscale requires split_k > 1")
    if n_out > 8192:
        raise ValueError("fp8 kblock+kscale fullnorm is intended for one full vector up to 8192 elements")
    if n_out % block_n != 0 or k_in % split_k != 0 or (k_in // split_k) % block_k != 0:
        raise ValueError(
            "fp8 kblock+kscale requires n_out % block_n == 0 and each split-K chunk divisible by block_k"
        )
    scale_blocks = k_in // block_k
    if int(weight_fp8.numel()) != scale_blocks * n_out * block_k:
        raise ValueError("weight_fp8 must contain [K/128, N, 128]")
    if int(scales.numel()) != scale_blocks * n_out:
        raise ValueError("scales must contain [K / 128, N]")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    _gemv_splitk_partial_fp8_kblock_kscale_block128_kernel[(triton.cdiv(n_out, block_n), split_k)](
        x_flat,
        weight_fp8,
        scales,
        partial,
        n_out,
        k_in,
        split_k,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    block_full = triton.next_power_of_2(n_out)
    _splitk_reduce_add_rmsnorm_full_kernel[(1,)](
        partial,
        bias_flat,
        residual_flat,
        norm_weight,
        out_sum_flat,
        out_norm_flat,
        n_out,
        bias is not None,
        split_k,
        float(eps),
        BLOCK_N=block_full,
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


@triton.jit
def _gemv_splitk_partial_fp8_kblock_kscale_nomask_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    partial_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    split_k: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    chunk_k = k_in // split_k
    k_begin = pid_k * chunk_k
    k_end = k_begin + chunk_k
    kblock_stride = n_out * BLOCK_K

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        block_id = k0 // BLOCK_K
        x = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last").to(tl.float32)
        packed = tl.load(
            w_fp8_ptr + block_id * kblock_stride + row_offsets[:, None] + offs_inner[None, :],
            eviction_policy="evict_first",
        )
        w = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale = tl.load(
            scales_ptr + block_id * n_out + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1) * scale

    tl.store(partial_ptr + pid_k * n_out + offs_n, acc)


@triton.jit
def _gemv_splitk_partial_fp8_kblock_kscale_nomask_u2_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    partial_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    split_k: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    chunk_k = k_in // split_k
    k_begin = pid_k * chunk_k
    k_end = k_begin + chunk_k
    kblock_stride = n_out * BLOCK_K

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(k_begin, k_end, 2 * BLOCK_K):
        block_id0 = k0 // BLOCK_K
        x0 = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last").to(tl.float32)
        packed0 = tl.load(
            w_fp8_ptr + block_id0 * kblock_stride + row_offsets[:, None] + offs_inner[None, :],
            eviction_policy="evict_first",
        )
        w0 = cuda_utils.convert_fp8e4b15_to_float16(packed0.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale0 = tl.load(
            scales_ptr + block_id0 * n_out + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w0 * x0[None, :], axis=1) * scale0

        block_id1 = block_id0 + 1
        x1 = tl.load(x_ptr + k0 + BLOCK_K + offs_inner, eviction_policy="evict_last").to(tl.float32)
        packed1 = tl.load(
            w_fp8_ptr + block_id1 * kblock_stride + row_offsets[:, None] + offs_inner[None, :],
            eviction_policy="evict_first",
        )
        w1 = cuda_utils.convert_fp8e4b15_to_float16(packed1.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale1 = tl.load(
            scales_ptr + block_id1 * n_out + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(w1 * x1[None, :], axis=1) * scale1

    tl.store(partial_ptr + pid_k * n_out + offs_n, acc)


def _triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale_nomask(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scales: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None,
    *,
    out_sum: torch.Tensor | None,
    out_norm: torch.Tensor | None,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    split_k: int,
    unroll2: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale_nomask requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale_nomask expects fp16 x/scales/residual")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out = int(residual_flat.numel())
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    split_k = int(split_k)
    if block_k != 128:
        raise ValueError("fp8 kblock+kscale nomask currently requires block_k=128")
    if split_k <= 1:
        raise ValueError("fp8 kblock+kscale nomask requires split_k > 1")
    if n_out > 8192:
        raise ValueError("fp8 kblock+kscale nomask fullnorm is intended for one full vector up to 8192 elements")
    if n_out % block_n != 0:
        raise ValueError("fp8 kblock+kscale nomask requires n_out divisible by block_n")
    per_split = k_in // split_k
    if k_in % split_k != 0 or per_split % block_k != 0:
        raise ValueError("each split-K chunk must be divisible by block_k")
    if unroll2 and per_split % (2 * block_k) != 0:
        raise ValueError("u2 requires each split-K chunk divisible by 2 * block_k")
    scale_blocks = k_in // block_k
    if int(weight_fp8.numel()) != scale_blocks * n_out * block_k:
        raise ValueError("weight_fp8 must contain [K/128, N, 128]")
    if int(scales.numel()) != scale_blocks * n_out:
        raise ValueError("scales must contain [K / 128, N]")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    kernel = (
        _gemv_splitk_partial_fp8_kblock_kscale_nomask_u2_block128_kernel
        if unroll2
        else _gemv_splitk_partial_fp8_kblock_kscale_nomask_block128_kernel
    )
    kernel[(triton.cdiv(n_out, block_n), split_k)](
        x_flat,
        weight_fp8,
        scales,
        partial,
        n_out,
        k_in,
        split_k,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    block_full = triton.next_power_of_2(n_out)
    _splitk_reduce_add_rmsnorm_full_kernel[(1,)](
        partial,
        bias_flat,
        residual_flat,
        norm_weight,
        out_sum_flat,
        out_norm_flat,
        n_out,
        bias is not None,
        split_k,
        float(eps),
        BLOCK_N=block_full,
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)


def triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale_nomask(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only kblock+kscale FP8 GEMV/add/RMSNorm for fixed divisible decode shapes."""
    return _triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale_nomask(
        x,
        weight_fp8,
        scales,
        residual,
        norm_weight,
        eps,
        bias,
        out_sum=out_sum,
        out_norm=out_norm,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
        unroll2=False,
    )


def triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale_u2(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only kblock+kscale FP8 GEMV/add/RMSNorm, no row mask and unrolled by two K blocks."""
    return _triton_decode_gemv_add_rmsnorm_fp8_kblock_kscale_nomask(
        x,
        weight_fp8,
        scales,
        residual,
        norm_weight,
        eps,
        bias,
        out_sum=out_sum,
        out_norm=out_norm,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
        unroll2=True,
    )


@triton.jit
def _gemv_splitk_partial_fp8_kblock_major_dot_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    partial_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    split_k: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    chunk_k = k_in // split_k
    k_begin = pid_k * chunk_k
    k_end = k_begin + chunk_k
    kblock_stride = n_out * BLOCK_K

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(k_begin, k_end, BLOCK_K):
        block_id = k0 // BLOCK_K
        x = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last")
        packed = tl.load(
            w_fp8_ptr + block_id * kblock_stride + row_offsets[:, None] + offs_inner[None, :],
            eviction_policy="evict_first",
        ).to(tl.float8e4b15, bitcast=True)
        scale = tl.load(
            scales_ptr + offs_n * scale_blocks + block_id,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.reshape(tl.dot(packed, x[:, None], out_dtype=tl.float32), (BLOCK_N,)) * scale

    tl.store(partial_ptr + pid_k * n_out + offs_n, acc)


def triton_decode_gemv_add_rmsnorm_fp8_kblock_major_dot(
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
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
    split_k: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-only fp8e4b15 split-K GEMV over k-block-major weights using tl.dot."""
    if not (x.is_cuda and weight_fp8.is_cuda and scales.is_cuda and residual.is_cuda and norm_weight.is_cuda):
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_major_dot requires CUDA tensors")
    if x.dtype != torch.float16 or scales.dtype != torch.float16 or residual.dtype != torch.float16:
        raise ValueError("triton_decode_gemv_add_rmsnorm_fp8_kblock_major_dot expects fp16 x/scales/residual")
    if weight_fp8.dtype != torch.uint8:
        raise ValueError("weight_fp8 must be uint8 fp8e4b15 storage")

    x_flat = x.reshape(-1)
    residual_flat = residual.reshape(-1)
    n_out = int(residual_flat.numel())
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    split_k = int(split_k)
    if block_k != 128:
        raise ValueError("fp8 k-block-major dot currently requires block_k=128")
    if split_k <= 1:
        raise ValueError("fp8 k-block-major dot requires split_k > 1")
    if n_out > 8192:
        raise ValueError("fp8 k-block-major dot fullnorm is intended for one full vector up to 8192 elements")
    if n_out % block_n != 0 or k_in % split_k != 0 or (k_in // split_k) % block_k != 0:
        raise ValueError(
            "fp8 k-block-major dot requires n_out % block_n == 0 and each split-K chunk divisible by block_k"
        )
    scale_blocks = k_in // block_k
    if int(weight_fp8.numel()) != scale_blocks * n_out * block_k:
        raise ValueError("weight_fp8 must contain [K/128, N, 128]")
    if int(scales.shape[0]) != n_out or int(scales.shape[1]) != scale_blocks:
        raise ValueError("scales must be [N, K / 128]")
    if norm_weight.ndim != 1 or int(norm_weight.numel()) != n_out:
        raise ValueError("norm_weight must be [N]")
    if norm_weight.dtype != x.dtype:
        norm_weight = norm_weight.to(dtype=x.dtype)
    norm_weight = norm_weight.contiguous()

    if bias is not None:
        if not bias.is_cuda or bias.dtype != x.dtype or int(bias.numel()) != n_out:
            raise ValueError("bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out_sum is None:
        out_sum = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    if out_norm is None:
        out_norm = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_sum_flat = out_sum.reshape(-1)
    out_norm_flat = out_norm.reshape(-1)
    if int(out_sum_flat.numel()) != n_out or int(out_norm_flat.numel()) != n_out:
        raise ValueError("out tensors must have N elements")

    partial = torch.empty((split_k, n_out), device=x.device, dtype=torch.float32)
    _gemv_splitk_partial_fp8_kblock_major_dot_block128_kernel[(triton.cdiv(n_out, block_n), split_k)](
        x_flat,
        weight_fp8,
        scales,
        partial,
        n_out,
        k_in,
        split_k,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    block_full = triton.next_power_of_2(n_out)
    _splitk_reduce_add_rmsnorm_full_kernel[(1,)](
        partial,
        bias_flat,
        residual_flat,
        norm_weight,
        out_sum_flat,
        out_norm_flat,
        n_out,
        bias is not None,
        split_k,
        float(eps),
        BLOCK_N=block_full,
        num_warps=8,
        num_stages=1,
    )
    return out_sum_flat.view_as(residual), out_norm_flat.view_as(residual)
