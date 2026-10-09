import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import utils as cuda_utils


@triton.jit
def _decode_gate_up_swiglu_sum_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_in
        x = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        gate_w = tl.load(
            w_ptr + offs_n[:, None] * k_in + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        up_w = tl.load(
            w_ptr + (offs_n[:, None] + intermediate_size) * k_in + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        gate_acc += tl.sum(gate_w * x[None, :], axis=1)
        up_acc += tl.sum(up_w * x[None, :], axis=1)

    if HAS_BIAS:
        gate_acc += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        up_acc += tl.load(bias_ptr + intermediate_size + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc, mask=mask_n)


def triton_decode_gate_up_swiglu(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int | None = None,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only fp16 fused gate/up projection and SwiGLU activation."""
    if not x.is_cuda or not gate_up_weight.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_weight.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu currently expects fp16 tensors")
    if gate_up_weight.ndim != 2:
        raise ValueError("gate_up_weight must be [2 * intermediate, K]")

    x_flat = x.reshape(-1)
    n_fused, k_in = int(gate_up_weight.shape[0]), int(gate_up_weight.shape[1])
    if intermediate_size is None:
        if n_fused % 2 != 0:
            raise ValueError("gate_up_weight rows must be even when intermediate_size is omitted")
        intermediate_size = n_fused // 2
    intermediate_size = int(intermediate_size)
    if n_fused < 2 * intermediate_size:
        raise ValueError("gate_up_weight must contain gate rows followed by up rows")
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")

    if gate_up_bias is not None:
        if (
            not gate_up_bias.is_cuda
            or gate_up_bias.dtype != torch.float16
            or int(gate_up_bias.numel()) < 2 * intermediate_size
        ):
            raise ValueError("gate_up_bias must be fp16 CUDA tensor with shape [2 * intermediate]")
        bias_flat = gate_up_bias.reshape(-1)
    else:
        bias_flat = x_flat

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    block_n = int(block_n)
    grid = (triton.cdiv(intermediate_size, block_n),)
    _decode_gate_up_swiglu_sum_kernel[grid](
        x_flat,
        gate_up_weight,
        bias_flat,
        out_flat,
        intermediate_size,
        k_in,
        gate_up_bias is not None,
        BLOCK_N=block_n,
        BLOCK_K=int(block_k),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _decode_gate_up_swiglu_fp8_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        scale_idx: tl.constexpr = k0 // BLOCK_K
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        gate_w = tl.load(
            w_fp8_ptr + offs_n[:, None] * k_in + offs_k[None, :],
            mask=mask_n[:, None],
            other=0,
        )
        up_w = tl.load(
            w_fp8_ptr + (offs_n[:, None] + intermediate_size) * k_in + offs_k[None, :],
            mask=mask_n[:, None],
            other=0,
        )
        gate_w = cuda_utils.convert_fp8e4b15_to_float16(gate_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        up_w = cuda_utils.convert_fp8e4b15_to_float16(up_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        gate_scale = tl.load(
            scales_ptr + offs_n * scale_blocks + scale_idx,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        up_scale = tl.load(
            scales_ptr + (offs_n + intermediate_size) * scale_blocks + scale_idx,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        gate_acc += tl.sum(gate_w * x[None, :], axis=1) * gate_scale
        up_acc += tl.sum(up_w * x[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc, mask=mask_n)


def triton_decode_gate_up_swiglu_fp8(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int | None = None,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only e4b15 block-128 gate/up projection and SwiGLU activation."""
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp8 requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp8 expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be uint8 e4b15 storage")
    if gate_up_weight_fp8.ndim != 2:
        raise ValueError("gate_up_weight_fp8 must be [2 * intermediate, K]")
    if gate_up_scales.ndim != 2:
        raise ValueError("gate_up_scales must be [2 * intermediate, K / 128]")

    x_flat = x.reshape(-1)
    n_fused, k_in = int(gate_up_weight_fp8.shape[0]), int(gate_up_weight_fp8.shape[1])
    if int(block_k) != 128:
        raise ValueError("triton_decode_gate_up_swiglu_fp8 currently requires block_k=128")
    if k_in % 128 != 0:
        raise ValueError("triton_decode_gate_up_swiglu_fp8 requires K to be divisible by 128")
    if int(x_flat.numel()) != k_in:
        raise ValueError(f"x has {x_flat.numel()} elements, expected {k_in}")
    if int(gate_up_scales.shape[0]) != n_fused or int(gate_up_scales.shape[1]) != k_in // 128:
        raise ValueError("gate_up_scales must match gate_up_weight_fp8 shape")
    if intermediate_size is None:
        if n_fused % 2 != 0:
            raise ValueError("gate_up_weight_fp8 rows must be even when intermediate_size is omitted")
        intermediate_size = n_fused // 2
    intermediate_size = int(intermediate_size)
    if n_fused < 2 * intermediate_size:
        raise ValueError("gate_up_weight_fp8 must contain gate rows followed by up rows")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    block_n = int(block_n)
    grid = (triton.cdiv(intermediate_size, block_n),)
    _decode_gate_up_swiglu_fp8_block128_kernel[grid](
        x_flat,
        gate_up_weight_fp8,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        k_in // 128,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _decode_gate_up_swiglu_fp8_row_interleaved_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    row_base = offs_n * (2 * k_in)
    scale_base = offs_n * (2 * scale_blocks)
    for k0 in range(0, k_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        scale_idx: tl.constexpr = k0 // BLOCK_K
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        gate_w = tl.load(
            w_fp8_ptr + row_base[:, None] + offs_k[None, :],
            mask=mask_n[:, None],
            other=0,
        )
        up_w = tl.load(
            w_fp8_ptr + row_base[:, None] + k_in + offs_k[None, :],
            mask=mask_n[:, None],
            other=0,
        )
        gate_w = cuda_utils.convert_fp8e4b15_to_float16(gate_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        up_w = cuda_utils.convert_fp8e4b15_to_float16(up_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        gate_scale = tl.load(
            scales_ptr + scale_base + scale_idx,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        up_scale = tl.load(
            scales_ptr + scale_base + scale_blocks + scale_idx,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        gate_acc += tl.sum(gate_w * x[None, :], axis=1) * gate_scale
        up_acc += tl.sum(up_w * x[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc, mask=mask_n)


def triton_decode_gate_up_swiglu_fp8_row_interleaved(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only e4b15 SwiGLU over row-interleaved [gate_row, up_row] FP8 weights."""
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_row_interleaved requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_row_interleaved expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be uint8 e4b15 storage")
    if gate_up_weight_fp8.ndim != 2 or gate_up_scales.ndim != 2:
        raise ValueError("row-interleaved gate/up FP8 pack must be 2D")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    if int(block_k) != 128:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_row_interleaved currently requires block_k=128")
    if k_in % 128 != 0:
        raise ValueError("K must be divisible by 128")
    if int(gate_up_weight_fp8.shape[0]) != intermediate_size or int(gate_up_weight_fp8.shape[1]) != 2 * k_in:
        raise ValueError("gate_up_weight_fp8 must be [intermediate, 2 * K]")
    if int(gate_up_scales.shape[0]) != intermediate_size or int(gate_up_scales.shape[1]) != 2 * (k_in // 128):
        raise ValueError("gate_up_scales must be [intermediate, 2 * K / 128]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    block_n = int(block_n)
    grid = (triton.cdiv(intermediate_size, block_n),)
    _decode_gate_up_swiglu_fp8_row_interleaved_block128_kernel[grid](
        x_flat,
        gate_up_weight_fp8,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        k_in // 128,
        BLOCK_N=block_n,
        BLOCK_K=128,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _decode_gate_up_swiglu_fp8_kblock_major_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    mask_n = offs_n < intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        scale_idx: tl.constexpr = k0 // BLOCK_K
        offs_k = k0 + offs_inner
        block_ids = offs_k // BLOCK_K
        inner = offs_k - block_ids * BLOCK_K
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        row_offsets = offs_n * BLOCK_K
        gate_w = tl.load(
            w_fp8_ptr
            + block_ids[None, :] * (2 * intermediate_size * BLOCK_K)
            + row_offsets[:, None]
            + inner[None, :],
            mask=mask_n[:, None],
            other=0,
        )
        up_w = tl.load(
            w_fp8_ptr
            + block_ids[None, :] * (2 * intermediate_size * BLOCK_K)
            + intermediate_size * BLOCK_K
            + row_offsets[:, None]
            + inner[None, :],
            mask=mask_n[:, None],
            other=0,
        )
        gate_w = cuda_utils.convert_fp8e4b15_to_float16(gate_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        up_w = cuda_utils.convert_fp8e4b15_to_float16(up_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        gate_scale = tl.load(
            scales_ptr + offs_n * scale_blocks + scale_idx,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        up_scale = tl.load(
            scales_ptr + (offs_n + intermediate_size) * scale_blocks + scale_idx,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        gate_acc += tl.sum(gate_w * x[None, :], axis=1) * gate_scale
        up_acc += tl.sum(up_w * x[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc, mask=mask_n)


def triton_decode_gate_up_swiglu_fp8_kblock_major(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only e4b15 SwiGLU over [K/128, 2, intermediate, 128] packed weights."""
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_major requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_major expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be uint8 e4b15 storage")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_major currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by 128")
    scale_blocks = k_in // block_k
    if int(gate_up_weight_fp8.numel()) != scale_blocks * 2 * intermediate_size * block_k:
        raise ValueError("gate_up_weight_fp8 must contain [K/128, 2, intermediate, 128]")
    if int(gate_up_scales.shape[0]) < 2 * intermediate_size or int(gate_up_scales.shape[1]) != scale_blocks:
        raise ValueError("gate_up_scales must be [2 * intermediate, K / 128]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    block_n = int(block_n)
    grid = (triton.cdiv(intermediate_size, block_n),)
    _decode_gate_up_swiglu_fp8_kblock_major_block128_kernel[grid](
        x_flat,
        gate_up_weight_fp8,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _decode_gate_up_swiglu_fp8_kblock_major_unroll_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCKS_PER_ITER: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    mask_n = offs_n < intermediate_size
    kblock_stride = 2 * intermediate_size * BLOCK_K

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K * BLOCKS_PER_ITER):
        for u in tl.static_range(0, BLOCKS_PER_ITER):
            block_id = k0 // BLOCK_K + u
            offs_k = k0 + u * BLOCK_K + offs_inner
            x = tl.load(x_ptr + offs_k, eviction_policy="evict_last").to(tl.float32)
            block_base = block_id * kblock_stride
            gate_w = tl.load(
                w_fp8_ptr + block_base + row_offsets[:, None] + offs_inner[None, :],
                mask=mask_n[:, None],
                other=0,
                eviction_policy="evict_first",
            )
            up_w = tl.load(
                w_fp8_ptr
                + block_base
                + intermediate_size * BLOCK_K
                + row_offsets[:, None]
                + offs_inner[None, :],
                mask=mask_n[:, None],
                other=0,
                eviction_policy="evict_first",
            )
            gate_w = cuda_utils.convert_fp8e4b15_to_float16(gate_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
            up_w = cuda_utils.convert_fp8e4b15_to_float16(up_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
            gate_scale = tl.load(
                scales_ptr + offs_n * scale_blocks + block_id,
                mask=mask_n,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            up_scale = tl.load(
                scales_ptr + (offs_n + intermediate_size) * scale_blocks + block_id,
                mask=mask_n,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            gate_acc += tl.sum(gate_w * x[None, :], axis=1) * gate_scale
            up_acc += tl.sum(up_w * x[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc, mask=mask_n)


def _triton_decode_gate_up_swiglu_fp8_kblock_major_unroll(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None,
    *,
    intermediate_size: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    blocks_per_iter: int,
) -> torch.Tensor:
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_major_unroll requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_major_unroll expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be uint8 e4b15 storage")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    block_k = int(block_k)
    blocks_per_iter = int(blocks_per_iter)
    if block_k != 128:
        raise ValueError("k-block-major unroll currently requires block_k=128")
    if blocks_per_iter <= 0:
        raise ValueError("blocks_per_iter must be positive")
    if k_in % (block_k * blocks_per_iter) != 0:
        raise ValueError("K must be divisible by block_k * blocks_per_iter")
    scale_blocks = k_in // block_k
    if int(gate_up_weight_fp8.numel()) != scale_blocks * 2 * intermediate_size * block_k:
        raise ValueError("gate_up_weight_fp8 must contain [K/128, 2, intermediate, 128]")
    if int(gate_up_scales.shape[0]) < 2 * intermediate_size or int(gate_up_scales.shape[1]) != scale_blocks:
        raise ValueError("gate_up_scales must be [2 * intermediate, K / 128]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    block_n = int(block_n)
    grid = (triton.cdiv(intermediate_size, block_n),)
    _decode_gate_up_swiglu_fp8_kblock_major_unroll_block128_kernel[grid](
        x_flat,
        gate_up_weight_fp8,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCKS_PER_ITER=blocks_per_iter,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def triton_decode_gate_up_swiglu_fp8_kblock_major_fast(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only e4b15 SwiGLU over k-block-major weights with simplified addressing."""
    return _triton_decode_gate_up_swiglu_fp8_kblock_major_unroll(
        x,
        gate_up_weight_fp8,
        gate_up_scales,
        out,
        intermediate_size=intermediate_size,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        blocks_per_iter=1,
    )


def triton_decode_gate_up_swiglu_fp8_kblock_major_u2(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only e4b15 SwiGLU over k-block-major weights, unrolled by two K blocks."""
    return _triton_decode_gate_up_swiglu_fp8_kblock_major_unroll(
        x,
        gate_up_weight_fp8,
        gate_up_scales,
        out,
        intermediate_size=intermediate_size,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        blocks_per_iter=2,
    )


@triton.jit
def _decode_gate_up_swiglu_fp8_kblock_kscale_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    mask_n = offs_n < intermediate_size
    kblock_stride = 2 * intermediate_size * BLOCK_K
    scale_stride = 2 * intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        block_id = k0 // BLOCK_K
        x = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last").to(tl.float32)
        block_base = block_id * kblock_stride
        gate_w = tl.load(
            w_fp8_ptr + block_base + row_offsets[:, None] + offs_inner[None, :],
            mask=mask_n[:, None],
            other=0,
            eviction_policy="evict_first",
        )
        up_w = tl.load(
            w_fp8_ptr
            + block_base
            + intermediate_size * BLOCK_K
            + row_offsets[:, None]
            + offs_inner[None, :],
            mask=mask_n[:, None],
            other=0,
            eviction_policy="evict_first",
        )
        gate_w = cuda_utils.convert_fp8e4b15_to_float16(gate_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        up_w = cuda_utils.convert_fp8e4b15_to_float16(up_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        gate_scale = tl.load(
            scales_ptr + block_id * scale_stride + offs_n,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        up_scale = tl.load(
            scales_ptr + block_id * scale_stride + intermediate_size + offs_n,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        gate_acc += tl.sum(gate_w * x[None, :], axis=1) * gate_scale
        up_acc += tl.sum(up_w * x[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc, mask=mask_n)


def triton_decode_gate_up_swiglu_fp8_kblock_kscale(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only e4b15 SwiGLU over [K/128, 2, intermediate, 128] weights and [K/128, 2, intermediate] scales."""
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_kscale requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_kscale expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be uint8 e4b15 storage")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("kblock+kscale currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")
    scale_blocks = k_in // block_k
    if int(gate_up_weight_fp8.numel()) != scale_blocks * 2 * intermediate_size * block_k:
        raise ValueError("gate_up_weight_fp8 must contain [K/128, 2, intermediate, 128]")
    if int(gate_up_scales.numel()) != scale_blocks * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [K / 128, 2, intermediate]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    block_n = int(block_n)
    grid = (triton.cdiv(intermediate_size, block_n),)
    _decode_gate_up_swiglu_fp8_kblock_kscale_block128_kernel[grid](
        x_flat,
        gate_up_weight_fp8,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _decode_gate_up_swiglu_fp8_kblock_kscale_nomask_unroll_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCKS_PER_ITER: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    kblock_stride = 2 * intermediate_size * BLOCK_K
    scale_stride = 2 * intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K * BLOCKS_PER_ITER):
        for u in tl.static_range(0, BLOCKS_PER_ITER):
            block_id = k0 // BLOCK_K + u
            block_base = block_id * kblock_stride
            x = tl.load(x_ptr + k0 + u * BLOCK_K + offs_inner, eviction_policy="evict_last").to(tl.float32)
            gate_w = tl.load(
                w_fp8_ptr + block_base + row_offsets[:, None] + offs_inner[None, :],
                eviction_policy="evict_first",
            )
            up_w = tl.load(
                w_fp8_ptr
                + block_base
                + intermediate_size * BLOCK_K
                + row_offsets[:, None]
                + offs_inner[None, :],
                eviction_policy="evict_first",
            )
            gate_w = cuda_utils.convert_fp8e4b15_to_float16(gate_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
            up_w = cuda_utils.convert_fp8e4b15_to_float16(up_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
            gate_scale = tl.load(
                scales_ptr + block_id * scale_stride + offs_n,
                eviction_policy="evict_first",
            ).to(tl.float32)
            up_scale = tl.load(
                scales_ptr + block_id * scale_stride + intermediate_size + offs_n,
                eviction_policy="evict_first",
            ).to(tl.float32)
            gate_acc += tl.sum(gate_w * x[None, :], axis=1) * gate_scale
            up_acc += tl.sum(up_w * x[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc)


def _triton_decode_gate_up_swiglu_fp8_kblock_kscale_nomask_unroll(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None,
    *,
    intermediate_size: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    blocks_per_iter: int,
) -> torch.Tensor:
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_kscale_nomask requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_kscale_nomask expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be uint8 e4b15 storage")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    blocks_per_iter = int(blocks_per_iter)
    if block_k != 128:
        raise ValueError("kblock+kscale nomask currently requires block_k=128")
    if blocks_per_iter <= 0:
        raise ValueError("blocks_per_iter must be positive")
    if intermediate_size % block_n != 0:
        raise ValueError("kblock+kscale nomask requires intermediate_size divisible by block_n")
    if k_in % (block_k * blocks_per_iter) != 0:
        raise ValueError("K must be divisible by block_k * blocks_per_iter")
    scale_blocks = k_in // block_k
    if int(gate_up_weight_fp8.numel()) != scale_blocks * 2 * intermediate_size * block_k:
        raise ValueError("gate_up_weight_fp8 must contain [K/128, 2, intermediate, 128]")
    if int(gate_up_scales.numel()) != scale_blocks * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [K / 128, 2, intermediate]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    _decode_gate_up_swiglu_fp8_kblock_kscale_nomask_unroll_block128_kernel[
        (triton.cdiv(intermediate_size, block_n),)
    ](
        x_flat,
        gate_up_weight_fp8,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCKS_PER_ITER=blocks_per_iter,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def triton_decode_gate_up_swiglu_fp8_kblock_kscale_nomask(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only kblock+kscale FP8 SwiGLU for fixed divisible decode shapes."""
    return _triton_decode_gate_up_swiglu_fp8_kblock_kscale_nomask_unroll(
        x,
        gate_up_weight_fp8,
        gate_up_scales,
        out,
        intermediate_size=intermediate_size,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        blocks_per_iter=1,
    )


def triton_decode_gate_up_swiglu_fp8_kblock_kscale_u2(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only kblock+kscale FP8 SwiGLU, no row mask and unrolled by two K blocks."""
    return _triton_decode_gate_up_swiglu_fp8_kblock_kscale_nomask_unroll(
        x,
        gate_up_weight_fp8,
        gate_up_scales,
        out,
        intermediate_size=intermediate_size,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        blocks_per_iter=2,
    )


@triton.jit
def _decode_gate_up_swiglu_fp8_kblock_rowpair_kscale_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * (2 * BLOCK_K)
    kblock_stride = intermediate_size * 2 * BLOCK_K
    scale_stride = intermediate_size * 2

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        block_id = k0 // BLOCK_K
        block_base = block_id * kblock_stride
        x = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last").to(tl.float32)
        gate_w = tl.load(
            w_fp8_ptr + block_base + row_offsets[:, None] + offs_inner[None, :],
            eviction_policy="evict_first",
        )
        up_w = tl.load(
            w_fp8_ptr + block_base + row_offsets[:, None] + BLOCK_K + offs_inner[None, :],
            eviction_policy="evict_first",
        )
        gate_w = cuda_utils.convert_fp8e4b15_to_float16(gate_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        up_w = cuda_utils.convert_fp8e4b15_to_float16(up_w.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
        scale_base = block_id * scale_stride + offs_n * 2
        gate_scale = tl.load(scales_ptr + scale_base, eviction_policy="evict_first").to(tl.float32)
        up_scale = tl.load(scales_ptr + scale_base + 1, eviction_policy="evict_first").to(tl.float32)
        gate_acc += tl.sum(gate_w * x[None, :], axis=1) * gate_scale
        up_acc += tl.sum(up_w * x[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc)


def triton_decode_gate_up_swiglu_fp8_kblock_rowpair_kscale(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only FP8 SwiGLU over [K/128, intermediate, 2, 128] weights and [K/128, intermediate, 2] scales."""
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_rowpair_kscale requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_rowpair_kscale expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be uint8 e4b15 storage")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("kblock rowpair kscale currently requires block_k=128")
    if intermediate_size % block_n != 0:
        raise ValueError("kblock rowpair kscale requires intermediate_size divisible by block_n")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")
    scale_blocks = k_in // block_k
    if int(gate_up_weight_fp8.numel()) != scale_blocks * intermediate_size * 2 * block_k:
        raise ValueError("gate_up_weight_fp8 must contain [K/128, intermediate, 2, 128]")
    if int(gate_up_scales.numel()) != scale_blocks * intermediate_size * 2:
        raise ValueError("gate_up_scales must contain [K / 128, intermediate, 2]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    _decode_gate_up_swiglu_fp8_kblock_rowpair_kscale_block128_kernel[
        (triton.cdiv(intermediate_size, block_n),)
    ](
        x_flat,
        gate_up_weight_fp8,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _decode_gate_up_swiglu_fp8_kblock_major_dot_block128_kernel(
    x_ptr,
    w_fp8_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_inner = tl.arange(0, BLOCK_K)
    row_offsets = offs_n * BLOCK_K
    mask_n = offs_n < intermediate_size
    kblock_stride = 2 * intermediate_size * BLOCK_K

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        block_id = k0 // BLOCK_K
        x = tl.load(x_ptr + k0 + offs_inner, eviction_policy="evict_last")
        block_base = block_id * kblock_stride
        gate_w = tl.load(
            w_fp8_ptr + block_base + row_offsets[:, None] + offs_inner[None, :],
            mask=mask_n[:, None],
            other=0,
            eviction_policy="evict_first",
        ).to(tl.float8e4b15, bitcast=True)
        up_w = tl.load(
            w_fp8_ptr
            + block_base
            + intermediate_size * BLOCK_K
            + row_offsets[:, None]
            + offs_inner[None, :],
            mask=mask_n[:, None],
            other=0,
            eviction_policy="evict_first",
        ).to(tl.float8e4b15, bitcast=True)
        gate_scale = tl.load(
            scales_ptr + offs_n * scale_blocks + block_id,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        up_scale = tl.load(
            scales_ptr + (offs_n + intermediate_size) * scale_blocks + block_id,
            mask=mask_n,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x_col = x[:, None]
        gate_acc += tl.reshape(tl.dot(gate_w, x_col, out_dtype=tl.float32), (BLOCK_N,)) * gate_scale
        up_acc += tl.reshape(tl.dot(up_w, x_col, out_dtype=tl.float32), (BLOCK_N,)) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc, mask=mask_n)


def triton_decode_gate_up_swiglu_fp8_kblock_major_dot(
    x: torch.Tensor,
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only e4b15 SwiGLU over k-block-major weights using tl.dot."""
    if not x.is_cuda or not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_major_dot requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp8_kblock_major_dot expects fp16 x/scales")
    if gate_up_weight_fp8.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp8 must be uint8 e4b15 storage")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("k-block-major dot currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")
    scale_blocks = k_in // block_k
    if int(gate_up_weight_fp8.numel()) != scale_blocks * 2 * intermediate_size * block_k:
        raise ValueError("gate_up_weight_fp8 must contain [K/128, 2, intermediate, 128]")
    if int(gate_up_scales.shape[0]) < 2 * intermediate_size or int(gate_up_scales.shape[1]) != scale_blocks:
        raise ValueError("gate_up_scales must be [2 * intermediate, K / 128]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    block_n = int(block_n)
    grid = (triton.cdiv(intermediate_size, block_n),)
    _decode_gate_up_swiglu_fp8_kblock_major_dot_block128_kernel[grid](
        x_flat,
        gate_up_weight_fp8,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


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


def triton_quantize_gate_up_e4b15_block128(
    gate_up_weight: torch.Tensor,
    *,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp16 gate/up weights to Triton's e4b15 FP8 byte format."""
    if not gate_up_weight.is_cuda or gate_up_weight.dtype != torch.float16:
        raise ValueError("triton_quantize_gate_up_e4b15_block128 expects fp16 CUDA weight")
    if gate_up_weight.ndim != 2:
        raise ValueError("gate_up_weight must be [N, K]")
    n_rows, k_in = int(gate_up_weight.shape[0]), int(gate_up_weight.shape[1])
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_quantize_gate_up_e4b15_block128 currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")

    weight = gate_up_weight.contiguous()
    weight_fp8 = torch.empty_like(weight, dtype=torch.uint8)
    scales = torch.empty((n_rows, k_in // block_k), device=weight.device, dtype=torch.float16)
    grid = (n_rows * (k_in // block_k),)
    _quantize_e4b15_block128_kernel[grid](
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
    packed = tl.load(
        weight_fp8_ptr + offs_m[:, None] * k_in + offs_k[None, :],
        mask=mask,
        other=0,
    )
    values = cuda_utils.convert_fp8e4b15_to_float16(packed.to(tl.float8e4b15, bitcast=True)).to(tl.float32)
    scale = tl.load(scales_ptr + offs_m * (k_in // BLOCK_K) + pid_k, mask=offs_m < n_rows, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * k_in + offs_k[None, :], values * scale[:, None], mask=mask)


def triton_dequant_gate_up_e4b15_block128(
    gate_up_weight_fp8: torch.Tensor,
    gate_up_scales: torch.Tensor,
    *,
    block_m: int = 8,
    block_k: int = 128,
    num_warps: int = 4,
    num_stages: int = 4,
) -> torch.Tensor:
    """Materialize e4b15 block-128 weights as fp16 for diagnostics."""
    if not gate_up_weight_fp8.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_dequant_gate_up_e4b15_block128 requires CUDA tensors")
    if gate_up_weight_fp8.dtype != torch.uint8 or gate_up_scales.dtype != torch.float16:
        raise ValueError("expected uint8 e4b15 storage and fp16 scales")
    if gate_up_weight_fp8.ndim != 2 or gate_up_scales.ndim != 2:
        raise ValueError("expected weight [N, K] and scales [N, K / 128]")
    n_rows, k_in = int(gate_up_weight_fp8.shape[0]), int(gate_up_weight_fp8.shape[1])
    block_k = int(block_k)
    if block_k != 128 or k_in % block_k != 0:
        raise ValueError("dequant currently requires block_k=128 and divisible K")
    if int(gate_up_scales.shape[0]) != n_rows or int(gate_up_scales.shape[1]) != k_in // block_k:
        raise ValueError("gate_up_scales shape does not match weight")

    out = torch.empty((n_rows, k_in), device=gate_up_weight_fp8.device, dtype=torch.float16)
    block_m = int(block_m)
    grid = (triton.cdiv(n_rows, block_m), k_in // block_k)
    _dequant_e4b15_block128_kernel[grid](
        gate_up_weight_fp8,
        gate_up_scales,
        out,
        n_rows,
        k_in,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


def triton_quantize_gate_up_int4_sym_block128(
    gate_up_weight: torch.Tensor,
    *,
    block_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp16 gate/up weights to symmetric int4 bytes plus block scales.

    The packed layout is row-major [N, K / 2]. Each byte stores two values in
    unsigned nibbles as q + 8, where q is clamped to [-8, 7].
    """
    if not gate_up_weight.is_cuda or gate_up_weight.dtype != torch.float16:
        raise ValueError("triton_quantize_gate_up_int4_sym_block128 expects fp16 CUDA weight")
    if gate_up_weight.ndim != 2:
        raise ValueError("gate_up_weight must be [N, K]")
    n_rows, k_in = int(gate_up_weight.shape[0]), int(gate_up_weight.shape[1])
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_quantize_gate_up_int4_sym_block128 currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")

    weight = gate_up_weight.contiguous().float().view(n_rows, k_in // block_k, block_k)
    scales = torch.clamp(weight.abs().amax(dim=2) / 7.0, min=1.0e-8).to(torch.float16)
    q = torch.round(weight / scales.float().unsqueeze(-1)).clamp_(-8, 7).to(torch.int16)
    q_u4 = (q + 8).to(torch.uint8)
    lo = q_u4[..., 0::2]
    hi = q_u4[..., 1::2] << 4
    packed = (lo | hi).contiguous().view(n_rows, k_in // 2)
    return packed, scales.contiguous()


def quantize_gate_up_int4_sym_block128_gptq_pack(
    gate_up_weight: torch.Tensor,
    activation_samples: torch.Tensor,
    *,
    block_k: int = 128,
    damp_percent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ quantize gate/up weights into the existing symmetric INT4 layout.

    This is an initialization-time quantization policy. It returns the exact
    row-major packed layout consumed by the current plain symmetric INT4
    gate/up CUDA kernel, so runtime performance should match the existing
    plain INT4 path.
    """
    if not gate_up_weight.is_cuda or gate_up_weight.dtype != torch.float16:
        raise ValueError("quantize_gate_up_int4_sym_block128_gptq_pack expects fp16 CUDA weight")
    if activation_samples.ndim != 2 or not activation_samples.is_cuda:
        raise ValueError("activation_samples must be CUDA [M, K]")
    if gate_up_weight.ndim != 2:
        raise ValueError("gate_up_weight must be [N, K]")
    n_rows, k_in = int(gate_up_weight.shape[0]), int(gate_up_weight.shape[1])
    if int(activation_samples.shape[1]) != k_in:
        raise ValueError("activation_samples K must match gate_up_weight K")
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("quantize_gate_up_int4_sym_block128_gptq_pack currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")
    if float(damp_percent) < 0.0:
        raise ValueError("damp_percent must be non-negative")

    samples = activation_samples.detach().float().contiguous()
    weight = gate_up_weight.detach().float().contiguous()
    q_all = torch.empty((n_rows, k_in), device=gate_up_weight.device, dtype=torch.int16)
    scales_all = torch.empty((n_rows, k_in // block_k), device=gate_up_weight.device, dtype=torch.float16)
    for start in range(0, k_in, block_k):
        end = start + block_k
        xg = samples[:, start:end]
        block = weight[:, start:end].clone()
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
    lo = q_u4[:, 0::2]
    hi = q_u4[:, 1::2] << 4
    packed = (lo | hi).contiguous()
    return packed, scales_all.contiguous()


_NF4_CODEBOOK = (
    -1.0,
    -0.6961928,
    -0.52507305,
    -0.39491749,
    -0.28444138,
    -0.18477343,
    -0.09105004,
    0.0,
    0.0795803,
    0.1609302,
    0.2461123,
    0.33791524,
    0.44070983,
    0.562617,
    0.72295684,
    1.0,
)
_E2M1_CODEBOOK = (
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)


def triton_quantize_gate_up_fp4_codebook_block128(
    gate_up_weight: torch.Tensor,
    *,
    codebook: str = "nf4",
    block_k: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp16 gate/up weights to a 4-bit codebook plus per-row block scales."""
    if not gate_up_weight.is_cuda or gate_up_weight.dtype != torch.float16:
        raise ValueError("triton_quantize_gate_up_fp4_codebook_block128 expects fp16 CUDA weight")
    if gate_up_weight.ndim != 2:
        raise ValueError("gate_up_weight must be [N, K]")
    n_rows, k_in = int(gate_up_weight.shape[0]), int(gate_up_weight.shape[1])
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("triton_quantize_gate_up_fp4_codebook_block128 currently requires block_k=128")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")

    codebook_name = str(codebook).strip().lower()
    if codebook_name == "nf4":
        values = _NF4_CODEBOOK
        scale_denominator = 1.0
    elif codebook_name in {"e2m1", "fp4-e2m1"}:
        values = _E2M1_CODEBOOK
        scale_denominator = 6.0
    else:
        raise ValueError("codebook must be 'nf4' or 'e2m1'")

    weight = gate_up_weight.contiguous().float().view(n_rows, k_in // block_k, block_k)
    scales = torch.clamp(weight.abs().amax(dim=2) / float(scale_denominator), min=1.0e-8)
    normalized = weight / scales.unsqueeze(-1)
    table = torch.tensor(values, device=gate_up_weight.device, dtype=torch.float32)
    q_u4 = torch.argmin((normalized.unsqueeze(-1) - table).abs(), dim=-1).to(torch.uint8)
    lo = q_u4[..., 0::2]
    hi = q_u4[..., 1::2] << 4
    packed = (lo | hi).contiguous().view(n_rows, k_in // 2)
    return packed, scales.to(torch.float16).contiguous()


@triton.jit
def _decode_gate_up_swiglu_int4_sym_kblock_kscale_block128_kernel(
    x_ptr,
    w_int4_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pair = tl.arange(0, BLOCK_K // 2)
    row_offsets = offs_n * (BLOCK_K // 2)
    kblock_stride = 2 * intermediate_size * (BLOCK_K // 2)
    scale_stride = 2 * intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        block_id = k0 // BLOCK_K
        x_even = tl.load(x_ptr + k0 + offs_pair * 2, eviction_policy="evict_last").to(tl.float32)
        x_odd = tl.load(x_ptr + k0 + offs_pair * 2 + 1, eviction_policy="evict_last").to(tl.float32)
        block_base = block_id * kblock_stride
        gate_packed = tl.load(
            w_int4_ptr + block_base + row_offsets[:, None] + offs_pair[None, :],
            eviction_policy="evict_first",
        ).to(tl.uint32)
        up_packed = tl.load(
            w_int4_ptr
            + block_base
            + intermediate_size * (BLOCK_K // 2)
            + row_offsets[:, None]
            + offs_pair[None, :],
            eviction_policy="evict_first",
        ).to(tl.uint32)

        gate_lo = ((gate_packed & 15).to(tl.float32) - 8.0)
        gate_hi = (((gate_packed >> 4) & 15).to(tl.float32) - 8.0)
        up_lo = ((up_packed & 15).to(tl.float32) - 8.0)
        up_hi = (((up_packed >> 4) & 15).to(tl.float32) - 8.0)

        gate_scale = tl.load(
            scales_ptr + block_id * scale_stride + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)
        up_scale = tl.load(
            scales_ptr + block_id * scale_stride + intermediate_size + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)

        gate_acc += tl.sum(gate_lo * x_even[None, :] + gate_hi * x_odd[None, :], axis=1) * gate_scale
        up_acc += tl.sum(up_lo * x_even[None, :] + up_hi * x_odd[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc)


def triton_decode_gate_up_swiglu_int4_sym_kblock_kscale(
    x: torch.Tensor,
    gate_up_weight_int4: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only symmetric-int4 weight-only SwiGLU over [K/128, 2, intermediate, 64] bytes."""
    if not x.is_cuda or not gate_up_weight_int4.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_int4_sym_kblock_kscale requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_int4_sym_kblock_kscale expects fp16 x/scales")
    if gate_up_weight_int4.dtype != torch.uint8:
        raise ValueError("gate_up_weight_int4 must be uint8 packed symmetric int4 storage")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("int4 kblock+kscale currently requires block_k=128")
    if intermediate_size % block_n != 0:
        raise ValueError("int4 kblock+kscale requires intermediate_size divisible by block_n")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")
    scale_blocks = k_in // block_k
    if int(gate_up_weight_int4.numel()) != scale_blocks * 2 * intermediate_size * (block_k // 2):
        raise ValueError("gate_up_weight_int4 must contain [K/128, 2, intermediate, 64]")
    if int(gate_up_scales.numel()) != scale_blocks * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [K/128, 2, intermediate]")

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    _decode_gate_up_swiglu_int4_sym_kblock_kscale_block128_kernel[
        (triton.cdiv(intermediate_size, block_n),)
    ](
        x_flat,
        gate_up_weight_int4,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out


@triton.jit
def _decode_fp4_e2m1_value(q):
    q_u = q.to(tl.uint32)
    is_pos = q_u >= 8
    idx = tl.where(is_pos, q_u - 8, 7 - q_u)
    mant = (idx & 1).to(tl.float32)
    exp = idx >> 1
    pow2 = tl.where(exp == 1, 1.0, tl.where(exp == 2, 2.0, 4.0))
    normal = (1.0 + 0.5 * mant) * pow2
    mag = tl.where(idx == 0, 0.0, tl.where(idx == 1, 0.5, normal))
    return tl.where(is_pos, mag, -mag)


@triton.jit
def _decode_fp4_codebook_value(q, CODEBOOK: tl.constexpr):
    if CODEBOOK == 1:
        v = q.to(tl.float32)
        v = tl.where(q == 0, -1.0, v)
        v = tl.where(q == 1, -0.6961928, v)
        v = tl.where(q == 2, -0.52507305, v)
        v = tl.where(q == 3, -0.39491749, v)
        v = tl.where(q == 4, -0.28444138, v)
        v = tl.where(q == 5, -0.18477343, v)
        v = tl.where(q == 6, -0.09105004, v)
        v = tl.where(q == 7, 0.0, v)
        v = tl.where(q == 8, 0.0795803, v)
        v = tl.where(q == 9, 0.1609302, v)
        v = tl.where(q == 10, 0.2461123, v)
        v = tl.where(q == 11, 0.33791524, v)
        v = tl.where(q == 12, 0.44070983, v)
        v = tl.where(q == 13, 0.562617, v)
        v = tl.where(q == 14, 0.72295684, v)
        v = tl.where(q == 15, 1.0, v)
    else:
        v = _decode_fp4_e2m1_value(q)
    return v


@triton.jit
def _decode_gate_up_swiglu_fp4_codebook_kblock_kscale_block128_kernel(
    x_ptr,
    w_fp4_ptr,
    scales_ptr,
    y_ptr,
    intermediate_size: tl.constexpr,
    k_in: tl.constexpr,
    scale_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CODEBOOK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pair = tl.arange(0, BLOCK_K // 2)
    row_offsets = offs_n * (BLOCK_K // 2)
    kblock_stride = 2 * intermediate_size * (BLOCK_K // 2)
    scale_stride = 2 * intermediate_size

    gate_acc = tl.zeros((BLOCK_N,), tl.float32)
    up_acc = tl.zeros((BLOCK_N,), tl.float32)
    for k0 in range(0, k_in, BLOCK_K):
        block_id = k0 // BLOCK_K
        x_even = tl.load(x_ptr + k0 + offs_pair * 2, eviction_policy="evict_last").to(tl.float32)
        x_odd = tl.load(x_ptr + k0 + offs_pair * 2 + 1, eviction_policy="evict_last").to(tl.float32)
        block_base = block_id * kblock_stride
        gate_packed = tl.load(
            w_fp4_ptr + block_base + row_offsets[:, None] + offs_pair[None, :],
            eviction_policy="evict_first",
        ).to(tl.uint32)
        up_packed = tl.load(
            w_fp4_ptr
            + block_base
            + intermediate_size * (BLOCK_K // 2)
            + row_offsets[:, None]
            + offs_pair[None, :],
            eviction_policy="evict_first",
        ).to(tl.uint32)

        gate_lo = _decode_fp4_codebook_value(gate_packed & 15, CODEBOOK)
        gate_hi = _decode_fp4_codebook_value((gate_packed >> 4) & 15, CODEBOOK)
        up_lo = _decode_fp4_codebook_value(up_packed & 15, CODEBOOK)
        up_hi = _decode_fp4_codebook_value((up_packed >> 4) & 15, CODEBOOK)

        gate_scale = tl.load(
            scales_ptr + block_id * scale_stride + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)
        up_scale = tl.load(
            scales_ptr + block_id * scale_stride + intermediate_size + offs_n,
            eviction_policy="evict_first",
        ).to(tl.float32)

        gate_acc += tl.sum(gate_lo * x_even[None, :] + gate_hi * x_odd[None, :], axis=1) * gate_scale
        up_acc += tl.sum(up_lo * x_even[None, :] + up_hi * x_odd[None, :], axis=1) * up_scale

    swish = gate_acc * tl.sigmoid(gate_acc)
    tl.store(y_ptr + offs_n, swish * up_acc)


def triton_decode_gate_up_swiglu_fp4_codebook_kblock_kscale(
    x: torch.Tensor,
    gate_up_weight_fp4: torch.Tensor,
    gate_up_scales: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    intermediate_size: int,
    codebook: str = "nf4",
    block_n: int = 32,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 4,
) -> torch.Tensor:
    """Decode-only 4-bit codebook weight-only SwiGLU over [K/128, 2, intermediate, 64] bytes."""
    if not x.is_cuda or not gate_up_weight_fp4.is_cuda or not gate_up_scales.is_cuda:
        raise ValueError("triton_decode_gate_up_swiglu_fp4_codebook_kblock_kscale requires CUDA tensors")
    if x.dtype != torch.float16 or gate_up_scales.dtype != torch.float16:
        raise ValueError("triton_decode_gate_up_swiglu_fp4_codebook_kblock_kscale expects fp16 x/scales")
    if gate_up_weight_fp4.dtype != torch.uint8:
        raise ValueError("gate_up_weight_fp4 must be uint8 packed 4-bit storage")

    x_flat = x.reshape(-1)
    intermediate_size = int(intermediate_size)
    k_in = int(x_flat.numel())
    block_n = int(block_n)
    block_k = int(block_k)
    if block_k != 128:
        raise ValueError("fp4 codebook kblock+kscale currently requires block_k=128")
    if intermediate_size % block_n != 0:
        raise ValueError("fp4 codebook kblock+kscale requires intermediate_size divisible by block_n")
    if k_in % block_k != 0:
        raise ValueError("K must be divisible by block_k")
    scale_blocks = k_in // block_k
    if int(gate_up_weight_fp4.numel()) != scale_blocks * 2 * intermediate_size * (block_k // 2):
        raise ValueError("gate_up_weight_fp4 must contain [K/128, 2, intermediate, 64]")
    if int(gate_up_scales.numel()) != scale_blocks * 2 * intermediate_size:
        raise ValueError("gate_up_scales must contain [K/128, 2, intermediate]")
    codebook_id = 1 if str(codebook).strip().lower() == "nf4" else 2

    if out is None:
        out = torch.empty((intermediate_size,), device=x.device, dtype=x.dtype)
    out_flat = out.reshape(-1)
    if int(out_flat.numel()) != intermediate_size:
        raise ValueError(f"out has {out_flat.numel()} elements, expected {intermediate_size}")

    _decode_gate_up_swiglu_fp4_codebook_kblock_kscale_block128_kernel[
        (triton.cdiv(intermediate_size, block_n),)
    ](
        x_flat,
        gate_up_weight_fp4,
        gate_up_scales,
        out_flat,
        intermediate_size,
        k_in,
        scale_blocks,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        CODEBOOK=codebook_id,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out
