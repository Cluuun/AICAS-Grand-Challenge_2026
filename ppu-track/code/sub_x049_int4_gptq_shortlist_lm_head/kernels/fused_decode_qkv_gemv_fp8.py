from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import utils as cuda_utils


@triton.jit
def _decode_qkv_gemv_fp8_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    bias_ptr,
    out_ptr,
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
    tl.store(out_ptr + offs_n, acc, mask=mask_n)


def triton_decode_qkv_gemv_fp8(
    x: torch.Tensor,
    qkv_weight_fp8: torch.Tensor,
    qkv_scales: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    block_n: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    """Decode-only fp8e4b15 qkv GEMV for packed Qwen3-VL q/k/v projection."""
    if not x.is_cuda or not qkv_weight_fp8.is_cuda or not qkv_scales.is_cuda:
        raise ValueError("triton_decode_qkv_gemv_fp8 requires CUDA tensors")
    if x.dtype != torch.float16 or qkv_scales.dtype != torch.float16:
        raise ValueError("triton_decode_qkv_gemv_fp8 expects fp16 x/scales")
    if qkv_weight_fp8.dtype != torch.uint8:
        raise ValueError("qkv_weight_fp8 must be uint8 fp8e4b15 storage")
    if qkv_weight_fp8.ndim != 2 or qkv_scales.ndim != 2:
        raise ValueError("qkv_weight_fp8 must be [N, K] and qkv_scales [N, K / 128]")

    x_flat = x.reshape(-1)
    n_out, k_in = int(qkv_weight_fp8.shape[0]), int(qkv_weight_fp8.shape[1])
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_decode_qkv_gemv_fp8 currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")
    if int(qkv_scales.shape[0]) != n_out or int(qkv_scales.shape[1]) != k_in // block_k:
        raise ValueError("qkv_scales shape does not match qkv_weight_fp8")

    if qkv_bias is not None:
        if not qkv_bias.is_cuda or qkv_bias.dtype != x.dtype or int(qkv_bias.numel()) != n_out:
            raise ValueError("qkv_bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = qkv_bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out is None:
        out = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != n_out:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {n_out}")

    block_n = int(block_n)
    grid = (triton.cdiv(n_out, block_n),)
    _decode_qkv_gemv_fp8_block128_kernel[grid](
        x_flat,
        qkv_weight_fp8,
        qkv_scales,
        bias_flat,
        out_flat,
        n_out,
        k_in,
        k_in // block_k,
        qkv_bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out_flat.view(*x.shape[:-1], n_out)


def triton_quantize_qkv_int4_sym_block128(
    qkv_weight: torch.Tensor,
    *,
    block_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize packed qkv fp16 weights to symmetric INT4 block-128 storage.

    Returned layout is k-block-major:

    - weight: [K / 128, N, 64] uint8, two signed 4-bit values per byte as q + 8
    - scales: [K / 128, N] fp16
    """
    if not qkv_weight.is_cuda or qkv_weight.dtype != torch.float16:
        raise ValueError("triton_quantize_qkv_int4_sym_block128 expects fp16 CUDA weight")
    if qkv_weight.ndim != 2:
        raise ValueError("qkv_weight must be [N, K]")
    n_out, k_in = int(qkv_weight.shape[0]), int(qkv_weight.shape[1])
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_quantize_qkv_int4_sym_block128 currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")

    weight = qkv_weight.contiguous().float().view(n_out, k_in // block_k, block_k)
    scales = torch.clamp(weight.abs().amax(dim=2) / 7.0, min=1.0e-8).to(torch.float16)
    q = torch.round(weight / scales.float().unsqueeze(-1)).clamp_(-8, 7).to(torch.int16)
    q_u4 = (q + 8).to(torch.uint8)
    lo = q_u4[..., 0::2]
    hi = q_u4[..., 1::2] << 4
    packed = (lo | hi).contiguous().view(n_out, k_in // block_k, block_k // 2)
    return packed.permute(1, 0, 2).contiguous(), scales.permute(1, 0).contiguous()


def quantize_int4_sym_block128_gptq_kblock_pack(
    weight: torch.Tensor,
    activation_samples: torch.Tensor,
    *,
    block_k: int = 128,
    damp_percent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ quantize a [N, K] weight into the existing k-block INT4 layout.

    Returned tensors match :func:`triton_quantize_qkv_int4_sym_block128`:
    weight [K / 128, N, 64] uint8 and scales [K / 128, N] fp16. This is an
    init/bench-time quantization policy; the runtime CUDA kernels are unchanged.
    """
    if not weight.is_cuda or weight.dtype != torch.float16:
        raise ValueError("quantize_int4_sym_block128_gptq_kblock_pack expects fp16 CUDA weight")
    if activation_samples.ndim != 2 or not activation_samples.is_cuda:
        raise ValueError("activation_samples must be CUDA [M, K]")
    if weight.ndim != 2:
        raise ValueError("weight must be [N, K]")
    n_rows, k_in = int(weight.shape[0]), int(weight.shape[1])
    if int(activation_samples.shape[1]) != k_in:
        raise ValueError("activation_samples K must match weight K")
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("quantize_int4_sym_block128_gptq_kblock_pack currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")
    if float(damp_percent) < 0.0:
        raise ValueError("damp_percent must be non-negative")

    samples = activation_samples.detach().float().contiguous()
    work_weight = weight.detach().float().contiguous()
    q_all = torch.empty((n_rows, k_in), device=weight.device, dtype=torch.int16)
    scales_all = torch.empty((n_rows, k_in // block_k), device=weight.device, dtype=torch.float16)
    for start in range(0, k_in, block_k):
        end = start + block_k
        xg = samples[:, start:end]
        block = work_weight[:, start:end].clone()
        scales = torch.clamp(block.abs().amax(dim=1, keepdim=True) / 7.0, min=1.0e-8)
        hessian = xg.transpose(0, 1).matmul(xg)
        diagonal = torch.diag(hessian)
        damp = float(damp_percent) / 100.0 * float(diagonal.mean().item())
        hessian.diagonal().add_(damp + 1.0e-6)
        h_inv = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
        q_block = torch.empty_like(block, dtype=torch.int16)
        scale_vec = scales.view(-1)
        for col in range(block_k):
            original_col = block[:, col]
            q_col = torch.round(original_col / scale_vec).clamp_(-8, 7)
            quant_col = q_col * scale_vec
            q_block[:, col] = q_col.to(torch.int16)
            err = (original_col - quant_col) / h_inv[col, col]
            if col + 1 < block_k:
                block[:, col + 1 :] -= err.unsqueeze(1) * h_inv[col, col + 1 :].unsqueeze(0)
        q_all[:, start:end] = q_block
        scales_all[:, start // block_k] = scales.squeeze(1).to(torch.float16)

    q_u4 = (q_all + 8).to(torch.uint8)
    packed = (q_u4[:, 0::2] | (q_u4[:, 1::2] << 4)).contiguous()
    packed = packed.view(n_rows, k_in // block_k, block_k // 2)
    return packed.permute(1, 0, 2).contiguous(), scales_all.permute(1, 0).contiguous()


@triton.jit
def _decode_qkv_gemv_int4_sym_kblock_kscale_block128_kernel(
    x_ptr,
    w_int4_ptr,
    scales_ptr,
    bias_ptr,
    out_ptr,
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
    offs_pair = tl.arange(0, BLOCK_K // 2)

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        x_even = tl.load(x_ptr + k0 + offs_pair * 2, eviction_policy="evict_last").to(tl.float32)
        x_odd = tl.load(x_ptr + k0 + offs_pair * 2 + 1, eviction_policy="evict_last").to(tl.float32)

        block_base = (k0 // BLOCK_K) * n_out * (BLOCK_K // 2)
        packed = tl.load(
            w_int4_ptr + block_base + offs_n[:, None] * (BLOCK_K // 2) + offs_pair[None, :],
            mask=mask_n[:, None],
            other=0,
            eviction_policy="evict_first",
        ).to(tl.uint32)
        lo = (packed & 15).to(tl.float32) - 8.0
        hi = ((packed >> 4) & 15).to(tl.float32) - 8.0
        scale = tl.load(
            scales_ptr + (k0 // BLOCK_K) * n_out + offs_n,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += tl.sum(lo * x_even[None, :] + hi * x_odd[None, :], axis=1) * scale

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_n, acc, mask=mask_n)


def triton_decode_qkv_gemv_int4_sym_kblock_kscale(
    x: torch.Tensor,
    qkv_weight_int4: torch.Tensor,
    qkv_scales: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    block_n: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only symmetric-INT4 qkv GEMV for packed Qwen3-VL q/k/v projection."""
    if not x.is_cuda or not qkv_weight_int4.is_cuda or not qkv_scales.is_cuda:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale requires CUDA tensors")
    if x.dtype != torch.float16 or qkv_scales.dtype != torch.float16:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale expects fp16 x/scales")
    if qkv_weight_int4.dtype != torch.uint8:
        raise ValueError("qkv_weight_int4 must be uint8 packed symmetric int4 storage")
    if qkv_weight_int4.ndim != 3 or qkv_scales.ndim != 2:
        raise ValueError("qkv_weight_int4 must be [K/128, N, 64] and qkv_scales [K/128, N]")

    x_flat = x.reshape(-1)
    scale_blocks, n_out, packed_k = (int(v) for v in qkv_weight_int4.shape)
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale currently requires block_k=128")
    if packed_k != block_k // 2:
        raise ValueError("qkv_weight_int4 last dimension must be 64 for block_k=128")
    k_in = scale_blocks * block_k
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")
    if int(qkv_scales.shape[0]) != scale_blocks or int(qkv_scales.shape[1]) != n_out:
        raise ValueError("qkv_scales shape does not match qkv_weight_int4")

    if qkv_bias is not None:
        if not qkv_bias.is_cuda or qkv_bias.dtype != x.dtype or int(qkv_bias.numel()) != n_out:
            raise ValueError("qkv_bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = qkv_bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out is None:
        out = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != n_out:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {n_out}")

    block_n = int(block_n)
    grid = (triton.cdiv(n_out, block_n),)
    _decode_qkv_gemv_int4_sym_kblock_kscale_block128_kernel[grid](
        x_flat,
        qkv_weight_int4,
        qkv_scales,
        bias_flat,
        out_flat,
        n_out,
        k_in,
        scale_blocks,
        qkv_bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out_flat.view(*x.shape[:-1], n_out)


@triton.jit
def _decode_qkv_gemv_int4_sym_static_block128_kernel(
    x_ptr,
    w_int4_ptr,
    scales_ptr,
    bias_ptr,
    out_ptr,
    n_out: tl.constexpr,
    k_in: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PAIR_BLOCK: tl.constexpr,
    U4_OFFSET: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pair = tl.arange(0, PAIR_BLOCK)

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        scale = tl.load(scales_ptr + (k0 // BLOCK_K) * n_out + offs_n, eviction_policy="evict_first").to(tl.float32)
        block_acc = tl.zeros((BLOCK_N,), tl.float32)
        x_sum = 0.0
        block_base = (k0 // BLOCK_K) * n_out * (BLOCK_K // 2)
        for pair0 in range(0, BLOCK_K // 2, PAIR_BLOCK):
            pair = pair0 + offs_pair
            x_even = tl.load(x_ptr + k0 + pair * 2, eviction_policy="evict_last").to(tl.float32)
            x_odd = tl.load(x_ptr + k0 + pair * 2 + 1, eviction_policy="evict_last").to(tl.float32)
            packed = tl.load(
                w_int4_ptr + block_base + offs_n[:, None] * (BLOCK_K // 2) + pair[None, :],
                eviction_policy="evict_first",
            ).to(tl.uint32)
            lo = (packed & 15).to(tl.float32)
            hi = ((packed >> 4) & 15).to(tl.float32)
            if U4_OFFSET:
                block_acc += tl.sum(lo * x_even[None, :] + hi * x_odd[None, :], axis=1)
                x_sum += tl.sum(x_even + x_odd, axis=0)
            else:
                block_acc += tl.sum((lo - 8.0) * x_even[None, :] + (hi - 8.0) * x_odd[None, :], axis=1)
        if U4_OFFSET:
            block_acc -= 8.0 * x_sum
        acc += block_acc * scale

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n).to(tl.float32)
    tl.store(out_ptr + offs_n, acc)


def triton_decode_qkv_gemv_int4_sym_kblock_kscale_static(
    x: torch.Tensor,
    qkv_weight_int4: torch.Tensor,
    qkv_scales: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    block_n: int = 8,
    block_k: int = 128,
    pair_block: int = 64,
    u4_offset: bool = False,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Static no-mask qkv INT4 GEMV for fixed decode qkv shapes.

    This keeps the same k-block-major packed layout as
    :func:`triton_decode_qkv_gemv_int4_sym_kblock_kscale`, but removes the
    generic tail masks and lets `pair_block` control unpack/reduction granularity.
    """
    if not x.is_cuda or not qkv_weight_int4.is_cuda or not qkv_scales.is_cuda:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale_static requires CUDA tensors")
    if x.dtype != torch.float16 or qkv_scales.dtype != torch.float16:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale_static expects fp16 x/scales")
    if qkv_weight_int4.dtype != torch.uint8:
        raise ValueError("qkv_weight_int4 must be uint8 packed symmetric int4 storage")
    if qkv_weight_int4.ndim != 3 or qkv_scales.ndim != 2:
        raise ValueError("qkv_weight_int4 must be [K/128, N, 64] and qkv_scales [K/128, N]")

    x_flat = x.reshape(-1)
    scale_blocks, n_out, packed_k = (int(v) for v in qkv_weight_int4.shape)
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale_static currently requires block_k=128")
    if packed_k != block_k // 2:
        raise ValueError("qkv_weight_int4 last dimension must be 64 for block_k=128")
    k_in = scale_blocks * block_k
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")
    if int(qkv_scales.shape[0]) != scale_blocks or int(qkv_scales.shape[1]) != n_out:
        raise ValueError("qkv_scales shape does not match qkv_weight_int4")

    pair_block = int(pair_block)
    if pair_block not in (16, 32, 64):
        raise ValueError("pair_block must be one of 16, 32, 64")
    block_n = int(block_n)
    if n_out % block_n != 0:
        raise ValueError("static qkv INT4 kernel requires block_n to divide n_out")

    if qkv_bias is not None:
        if not qkv_bias.is_cuda or qkv_bias.dtype != x.dtype or int(qkv_bias.numel()) != n_out:
            raise ValueError("qkv_bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = qkv_bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out is None:
        out = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != n_out:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {n_out}")

    grid = (n_out // block_n,)
    _decode_qkv_gemv_int4_sym_static_block128_kernel[grid](
        x_flat,
        qkv_weight_int4,
        qkv_scales,
        bias_flat,
        out_flat,
        n_out,
        k_in,
        qkv_bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        PAIR_BLOCK=pair_block,
        U4_OFFSET=bool(u4_offset),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out_flat.view(*x.shape[:-1], n_out)


@triton.jit
def _decode_qkv_gemv_int4_sym_kblock_kscale_u32_block128_kernel(
    x_ptr,
    w_int4_u32_ptr,
    scales_ptr,
    bias_ptr,
    out_ptr,
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
    offs_word = tl.arange(0, BLOCK_K // 8)

    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        x_base = k0 + offs_word * 8
        x0 = tl.load(x_ptr + x_base + 0, eviction_policy="evict_last").to(tl.float32)
        x1 = tl.load(x_ptr + x_base + 1, eviction_policy="evict_last").to(tl.float32)
        x2 = tl.load(x_ptr + x_base + 2, eviction_policy="evict_last").to(tl.float32)
        x3 = tl.load(x_ptr + x_base + 3, eviction_policy="evict_last").to(tl.float32)
        x4 = tl.load(x_ptr + x_base + 4, eviction_policy="evict_last").to(tl.float32)
        x5 = tl.load(x_ptr + x_base + 5, eviction_policy="evict_last").to(tl.float32)
        x6 = tl.load(x_ptr + x_base + 6, eviction_policy="evict_last").to(tl.float32)
        x7 = tl.load(x_ptr + x_base + 7, eviction_policy="evict_last").to(tl.float32)

        block_base = (k0 // BLOCK_K) * n_out * (BLOCK_K // 8)
        packed = tl.load(
            w_int4_u32_ptr + block_base + offs_n[:, None] * (BLOCK_K // 8) + offs_word[None, :],
            mask=mask_n[:, None],
            other=0,
            eviction_policy="evict_first",
        ).to(tl.uint32)
        b0 = packed & 255
        b1 = (packed >> 8) & 255
        b2 = (packed >> 16) & 255
        b3 = (packed >> 24) & 255
        block_acc = tl.sum(
            ((b0 & 15).to(tl.float32) - 8.0) * x0[None, :]
            + (((b0 >> 4) & 15).to(tl.float32) - 8.0) * x1[None, :]
            + ((b1 & 15).to(tl.float32) - 8.0) * x2[None, :]
            + (((b1 >> 4) & 15).to(tl.float32) - 8.0) * x3[None, :]
            + ((b2 & 15).to(tl.float32) - 8.0) * x4[None, :]
            + (((b2 >> 4) & 15).to(tl.float32) - 8.0) * x5[None, :]
            + ((b3 & 15).to(tl.float32) - 8.0) * x6[None, :]
            + (((b3 >> 4) & 15).to(tl.float32) - 8.0) * x7[None, :],
            axis=1,
        )
        scale = tl.load(
            scales_ptr + (k0 // BLOCK_K) * n_out + offs_n,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        acc += block_acc * scale

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_n, acc, mask=mask_n)


def triton_decode_qkv_gemv_int4_sym_kblock_kscale_u32(
    x: torch.Tensor,
    qkv_weight_int4_u32: torch.Tensor,
    qkv_scales: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    block_n: int = 16,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only symmetric-INT4 qkv GEMV using a uint32 view of packed weights."""
    if not x.is_cuda or not qkv_weight_int4_u32.is_cuda or not qkv_scales.is_cuda:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale_u32 requires CUDA tensors")
    if x.dtype != torch.float16 or qkv_scales.dtype != torch.float16:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale_u32 expects fp16 x/scales")
    if qkv_weight_int4_u32.dtype != torch.int32:
        raise ValueError("qkv_weight_int4_u32 must be an int32 view of packed symmetric int4 storage")
    if qkv_weight_int4_u32.ndim != 3 or qkv_scales.ndim != 2:
        raise ValueError("qkv_weight_int4_u32 must be [K/128, N, 16] and qkv_scales [K/128, N]")

    x_flat = x.reshape(-1)
    scale_blocks, n_out, packed_words = (int(v) for v in qkv_weight_int4_u32.shape)
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_decode_qkv_gemv_int4_sym_kblock_kscale_u32 currently requires block_k=128")
    if packed_words != block_k // 8:
        raise ValueError("qkv_weight_int4_u32 last dimension must be 16 for block_k=128")
    k_in = scale_blocks * block_k
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")
    if int(qkv_scales.shape[0]) != scale_blocks or int(qkv_scales.shape[1]) != n_out:
        raise ValueError("qkv_scales shape does not match qkv_weight_int4_u32")

    if qkv_bias is not None:
        if not qkv_bias.is_cuda or qkv_bias.dtype != x.dtype or int(qkv_bias.numel()) != n_out:
            raise ValueError("qkv_bias must be fp16 CUDA tensor with shape [N]")
        bias_flat = qkv_bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out is None:
        out = torch.empty((n_out,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != n_out:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {n_out}")

    block_n = int(block_n)
    grid = (triton.cdiv(n_out, block_n),)
    _decode_qkv_gemv_int4_sym_kblock_kscale_u32_block128_kernel[grid](
        x_flat,
        qkv_weight_int4_u32,
        qkv_scales,
        bias_flat,
        out_flat,
        n_out,
        k_in,
        scale_blocks,
        qkv_bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out_flat.view(*x.shape[:-1], n_out)
