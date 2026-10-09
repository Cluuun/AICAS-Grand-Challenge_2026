from __future__ import annotations

import os

import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/junkrat_triton_cache")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    try:
        from triton.language.extra.libdevice import rsqrt as _tl_rsqrt
    except ModuleNotFoundError:
        try:
            from triton.language.extra.cuda.libdevice import rsqrt as _tl_rsqrt
        except ModuleNotFoundError:
            from triton.language.math import rsqrt as _tl_rsqrt
else:
    _tl_rsqrt = None


DEFAULT_BLOCK_N = int(os.getenv("JUNKRAT_SAGE_HALF_LAYER_BLOCK_N", "8"))
DEFAULT_BLOCK_PACKED_K = int(os.getenv("JUNKRAT_SAGE_HALF_LAYER_BLOCK_PACKED_K", "128"))


if TRITON_AVAILABLE:

    @triton.jit
    def _rms_rstd_kernel(
        residual_ptr,
        rstd_ptr,
        hidden_size,
        eps,
        block_size: tl.constexpr,
    ):
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_size
        x = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sumsq = tl.sum(x * x, axis=0)
        rstd = _tl_rsqrt(sumsq / hidden_size + eps)
        tl.store(rstd_ptr, rstd)


    @triton.jit
    def _gate_up_swiglu_interleaved_q8_prescaled_kernel(
        residual_ptr,
        norm_weight_ptr,
        rstd_ptr,
        packed_weight_ptr,
        scale_ptr,
        output_ptr,
        intermediate_size,
        hidden_size,
        packed_hidden_size,
        packed_row_stride,
        block_n: tl.constexpr,
        block_packed_k: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        n_offsets = pid * block_n + tl.arange(0, block_n)
        n_mask = n_offsets < intermediate_size
        gate_rows = n_offsets * 2
        up_rows = gate_rows + 1
        rstd = tl.load(rstd_ptr).to(tl.float32)

        gate_acc = tl.zeros((block_n,), dtype=tl.float32)
        up_acc = tl.zeros((block_n,), dtype=tl.float32)
        for packed_k_start in tl.range(0, packed_hidden_size, block_packed_k):
            packed_k_offsets = packed_k_start + tl.arange(0, block_packed_k)
            packed_k_mask = packed_k_offsets < packed_hidden_size
            hidden_base_offsets = packed_k_offsets * 4

            h0_mask = hidden_base_offsets < hidden_size
            h1_mask = hidden_base_offsets + 1 < hidden_size
            h2_mask = hidden_base_offsets + 2 < hidden_size
            h3_mask = hidden_base_offsets + 3 < hidden_size
            h0 = tl.load(residual_ptr + hidden_base_offsets, mask=h0_mask, other=0.0).to(tl.float32)
            h1 = tl.load(residual_ptr + hidden_base_offsets + 1, mask=h1_mask, other=0.0).to(tl.float32)
            h2 = tl.load(residual_ptr + hidden_base_offsets + 2, mask=h2_mask, other=0.0).to(tl.float32)
            h3 = tl.load(residual_ptr + hidden_base_offsets + 3, mask=h3_mask, other=0.0).to(tl.float32)
            wnorm0 = tl.load(norm_weight_ptr + hidden_base_offsets, mask=h0_mask, other=0.0).to(tl.float32)
            wnorm1 = tl.load(norm_weight_ptr + hidden_base_offsets + 1, mask=h1_mask, other=0.0).to(tl.float32)
            wnorm2 = tl.load(norm_weight_ptr + hidden_base_offsets + 2, mask=h2_mask, other=0.0).to(tl.float32)
            wnorm3 = tl.load(norm_weight_ptr + hidden_base_offsets + 3, mask=h3_mask, other=0.0).to(tl.float32)
            x0 = h0 * wnorm0 * rstd
            x1 = h1 * wnorm1 * rstd
            x2 = h2 * wnorm2 * rstd
            x3 = h3 * wnorm3 * rstd

            gate_packed = tl.load(
                packed_weight_ptr + gate_rows[:, None] * packed_row_stride + packed_k_offsets[None, :],
                mask=n_mask[:, None] & packed_k_mask[None, :],
                other=0,
            ).to(tl.int32)
            up_packed = tl.load(
                packed_weight_ptr + up_rows[:, None] * packed_row_stride + packed_k_offsets[None, :],
                mask=n_mask[:, None] & packed_k_mask[None, :],
                other=0,
            ).to(tl.int32)

            gw0 = (gate_packed & 0xFF).to(tl.float32) - 128.0
            gw1 = ((gate_packed >> 8) & 0xFF).to(tl.float32) - 128.0
            gw2 = ((gate_packed >> 16) & 0xFF).to(tl.float32) - 128.0
            gw3 = ((gate_packed >> 24) & 0xFF).to(tl.float32) - 128.0
            uw0 = (up_packed & 0xFF).to(tl.float32) - 128.0
            uw1 = ((up_packed >> 8) & 0xFF).to(tl.float32) - 128.0
            uw2 = ((up_packed >> 16) & 0xFF).to(tl.float32) - 128.0
            uw3 = ((up_packed >> 24) & 0xFF).to(tl.float32) - 128.0

            gate_acc += tl.sum(gw0 * x0[None, :] + gw1 * x1[None, :] + gw2 * x2[None, :] + gw3 * x3[None, :], axis=1)
            up_acc += tl.sum(uw0 * x0[None, :] + uw1 * x1[None, :] + uw2 * x2[None, :] + uw3 * x3[None, :], axis=1)

        gate_scale = tl.load(scale_ptr + gate_rows, mask=n_mask, other=0.0).to(tl.float32)
        up_scale = tl.load(scale_ptr + up_rows, mask=n_mask, other=0.0).to(tl.float32)
        gate_acc *= gate_scale
        up_acc *= up_scale
        gate = gate_acc * tl.sigmoid(gate_acc)
        out = gate * up_acc
        tl.store(output_ptr + n_offsets, out.to(output_ptr.dtype.element_ty), mask=n_mask)


def can_use_sage_half_layer_q8(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    rstd: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    return (
        TRITON_AVAILABLE
        and residual.is_cuda
        and norm_weight.is_cuda
        and packed_weight.is_cuda
        and scale.is_cuda
        and rstd.is_cuda
        and output.is_cuda
        and residual.ndim == 1
        and norm_weight.ndim == 1
        and output.ndim == 1
        and residual.dtype in (torch.float16, torch.bfloat16)
        and norm_weight.dtype == residual.dtype
        and output.dtype == residual.dtype
        and packed_weight.dtype == torch.int32
        and rstd.dtype == torch.float32
        and residual.is_contiguous()
        and norm_weight.is_contiguous()
        and packed_weight.is_contiguous()
        and scale.view(-1).is_contiguous()
        and output.is_contiguous()
        and packed_weight.shape[0] == output.numel() * 2
        and packed_weight.shape[1] == (residual.numel() + 3) // 4
        and scale.numel() >= packed_weight.shape[0]
    )


def sage_half_layer_gate_up_q8(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    rstd: torch.Tensor,
    output: torch.Tensor,
    *,
    eps: float = 1e-6,
    block_n: int | None = None,
    block_packed_k: int | None = None,
) -> torch.Tensor:
    if not can_use_sage_half_layer_q8(residual, norm_weight, packed_weight, scale, rstd, output):
        raise RuntimeError("Sage half-layer Q8 gate_up called with unsupported tensors.")
    if block_n is None:
        block_n = max(int(DEFAULT_BLOCK_N), 1)
    if block_packed_k is None:
        block_packed_k = max(int(DEFAULT_BLOCK_PACKED_K), 1)
    hidden_size = residual.numel()
    block_size = triton.next_power_of_2(hidden_size)
    _rms_rstd_kernel[(1,)](
        residual,
        rstd,
        hidden_size,
        float(eps),
        block_size=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    intermediate_size = output.numel()
    grid = (triton.cdiv(intermediate_size, block_n),)
    _gate_up_swiglu_interleaved_q8_prescaled_kernel[grid](
        residual,
        norm_weight,
        rstd,
        packed_weight,
        scale.view(-1),
        output,
        intermediate_size,
        hidden_size,
        packed_weight.shape[1],
        packed_weight.stride(0),
        block_n=block_n,
        block_packed_k=block_packed_k,
        num_warps=4,
    )
    return output


def prewarm_sage_half_layer_q8(
    *,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    rstd: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if not can_use_sage_half_layer_q8(residual, norm_weight, packed_weight, scale, rstd, output):
        return
    sage_half_layer_gate_up_q8(
        residual,
        norm_weight,
        packed_weight,
        scale,
        rstd,
        output,
    )
