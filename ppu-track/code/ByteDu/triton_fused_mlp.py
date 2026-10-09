"""Fused decode MLP: gate_up GEMV + SwiGLU + down GEMV in minimal kernel launches.

For single-token decode (batch=1, seq_len=1), this replaces the 3-kernel
sequence (gate_up GEMM + SwiGLU + down GEMM) with 2 fused Triton kernels:
  1. gate_up GEMV + SwiGLU (fused: reads both gate and up weights, applies
     SwiGLU activation in-register, avoids materializing the full gate_up output)
  2. down GEMV (reads SwiGLU output and down_proj weight, writes final output)

The key win is replacing cuBLAS's M=1 GEMM (which can select suboptimal
split-K algorithms in CUDA graphs) with a dedicated GEMV that directly
saturates HBM bandwidth.

Extended with fused_add_norm variant: fuses the pre-MLP residual-add +
RMSNorm into the GateUp+SwiGLU GEMV, eliminating one kernel launch per
layer (the separate fused_add_rms_norm call).
"""
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

    @triton.jit
    def _fused_gate_up_swiglu_kernel(
        input_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        output_ptr,
        N,
        K,
        weight_stride_n,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            x = tl.load(input_ptr + k_offs, mask=k_mask, other=0).to(tl.float32)

            gw = tl.load(
                gate_weight_ptr + n_offs[:, None] * weight_stride_n + k_offs[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            uw = tl.load(
                up_weight_ptr + n_offs[:, None] * weight_stride_n + k_offs[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)

            gate_acc += tl.sum(gw * x[None, :], axis=1)
            up_acc += tl.sum(uw * x[None, :], axis=1)

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        result = gate_silu * up_acc
        tl.store(output_ptr + n_offs, result.to(output_ptr.dtype.element_ty), mask=n_mask)

    @triton.jit
    def _fused_norm_gate_up_swiglu_kernel(
        hidden_ptr,
        norm_weight_ptr,
        gate_weight_ptr,
        up_weight_ptr,
        output_ptr,
        N,
        K,
        eps,
        weight_stride_n,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused: normed = rms_norm(hidden, norm_weight);
        output = SwiGLU(gate_w @ normed, up_w @ normed).

        Each block redundantly computes the full RMSNorm (cheap for K=2048)
        while tiling the GateUp GEMV across blocks.
        Caller is responsible for computing hidden = residual + attn_out
        before calling this kernel.
        """
        pid = tl.program_id(0).to(tl.int64)
        n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        gate_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        sq_buf = tl.zeros((BLOCK_K,), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K

            hidden = tl.load(hidden_ptr + k_offs, mask=k_mask, other=0).to(tl.float32)

            # Accumulate sum of squares for RMSNorm
            sq_buf += hidden * hidden

            # Pre-multiply by norm weight (rsqrt applied post-loop)
            nw = tl.load(norm_weight_ptr + k_offs, mask=k_mask, other=0).to(tl.float32)
            x_mod = hidden * nw

            # GEMV accumulate with pre-weighted input
            gw = tl.load(
                gate_weight_ptr + n_offs[:, None] * weight_stride_n + k_offs[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            uw = tl.load(
                up_weight_ptr + n_offs[:, None] * weight_stride_n + k_offs[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)

            gate_acc += tl.sum(gw * x_mod[None, :], axis=1)
            up_acc += tl.sum(uw * x_mod[None, :], axis=1)

        # Reduce sum-of-squares and apply rsqrt normalization
        sum_sq = tl.sum(sq_buf)
        rsqrt_val = _tl_rsqrt(sum_sq / K + eps)
        gate_acc *= rsqrt_val
        up_acc *= rsqrt_val

        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        result = gate_silu * up_acc
        tl.store(output_ptr + n_offs, result.to(output_ptr.dtype.element_ty), mask=n_mask)

    @triton.jit
    def _gemv_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        N,
        K,
        weight_stride_n,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_start in tl.range(0, K, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            x = tl.load(input_ptr + k_offs, mask=k_mask, other=0).to(tl.float32)
            w = tl.load(
                weight_ptr + n_offs[:, None] * weight_stride_n + k_offs[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        tl.store(output_ptr + n_offs, acc.to(output_ptr.dtype.element_ty), mask=n_mask)


FUSED_MLP_BLOCK_N = int(os.environ.get("JUNKRAT_FUSED_MLP_BLOCK_N", "32"))
FUSED_MLP_BLOCK_K = int(os.environ.get("JUNKRAT_FUSED_MLP_BLOCK_K", "256"))
FUSED_MLP_NUM_WARPS = int(os.environ.get("JUNKRAT_FUSED_MLP_NUM_WARPS", "4"))
GEMV_BLOCK_N = 64
GEMV_BLOCK_K = 256


def fused_gate_up_swiglu(
    hidden_flat: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Fused gate_up GEMV + SwiGLU.

    Args:
        hidden_flat: (K,) input vector
        gate_weight: (N, K) gate projection weight
        up_weight: (N, K) up projection weight
        output: (N,) pre-allocated output buffer

    Returns:
        output tensor with SwiGLU(gate_proj(x), up_proj(x))
    """
    N, K = gate_weight.shape
    grid = (triton.cdiv(N, FUSED_MLP_BLOCK_N),)
    _fused_gate_up_swiglu_kernel[grid](
        hidden_flat,
        gate_weight,
        up_weight,
        output,
        N,
        K,
        gate_weight.stride(0),
        BLOCK_N=FUSED_MLP_BLOCK_N,
        BLOCK_K=FUSED_MLP_BLOCK_K,
        num_warps=FUSED_MLP_NUM_WARPS,
    )
    return output


def fused_norm_gate_up_swiglu(
    hidden_flat: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    output: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm+GateUp GEMV+SwiGLU.

    Computes: normed = rms_norm(hidden, norm_weight, eps)
              output = SwiGLU(gate_weight @ normed, up_weight @ normed)

    Caller must compute hidden = residual + attn_out before calling.
    """
    N, K = gate_weight.shape
    grid = (triton.cdiv(N, FUSED_MLP_BLOCK_N),)
    _fused_norm_gate_up_swiglu_kernel[grid](
        hidden_flat,
        norm_weight,
        gate_weight,
        up_weight,
        output,
        N,
        K,
        eps,
        gate_weight.stride(0),
        BLOCK_N=FUSED_MLP_BLOCK_N,
        BLOCK_K=FUSED_MLP_BLOCK_K,
        num_warps=FUSED_MLP_NUM_WARPS,
    )
    return output


def triton_gemv(
    hidden_flat: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    block_n: int = GEMV_BLOCK_N,
    block_k: int = GEMV_BLOCK_K,
) -> torch.Tensor:
    """Triton GEMV: output = weight @ hidden_flat.

    Args:
        hidden_flat: (K,) input vector
        weight: (N, K) weight matrix
        output: (N,) pre-allocated output buffer
    """
    N, K = weight.shape
    grid = (triton.cdiv(N, block_n),)
    nw = 4 if block_n <= 128 else 8
    _gemv_kernel[grid](
        hidden_flat,
        weight,
        output,
        N,
        K,
        weight.stride(0),
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=nw,
    )
    return output


def prewarm_fused_mlp(
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    if not TRITON_AVAILABLE or device.type != "cuda":
        return

    with torch.inference_mode():
        x = torch.zeros(hidden_size, device=device, dtype=dtype)
        gw = torch.zeros(intermediate_size, hidden_size, device=device, dtype=dtype)
        uw = torch.zeros_like(gw)
        dw = torch.zeros(hidden_size, intermediate_size, device=device, dtype=dtype)
        out_swiglu = torch.zeros(intermediate_size, device=device, dtype=dtype)
        out_down = torch.zeros(hidden_size, device=device, dtype=dtype)
        fused_gate_up_swiglu(x, gw, uw, out_swiglu)
        triton_gemv(out_swiglu, dw, out_down)
        # Prewarm the norm+gateup variant
        nw = torch.ones(hidden_size, device=device, dtype=dtype)
        fused_norm_gate_up_swiglu(x, nw, gw, uw, out_swiglu, eps=1e-6)
        torch.cuda.synchronize(device=device)
