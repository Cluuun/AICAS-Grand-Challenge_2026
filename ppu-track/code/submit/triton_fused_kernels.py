"""
Triton fused kernels for Qwen3-VL-2B optimization (batch=1).

Strategy: Use Triton for element-wise fusion (RMSNorm, residual add, SwiGLU, RoPE)
and cuBLAS for GEMM/GEMV (which it handles optimally). The key wins are:

1. Pre-concatenate KV weights → 1 cuBLAS GEMV/GEMM instead of 2
2. Pre-concatenate Gate+Up weights → 1 cuBLAS GEMV/GEMM instead of 2
3. Fused RMSNorm (avoid writing norm output to HBM when followed by GEMV)
4. Fused SwiGLU (silu(gate) * up without writing intermediates to HBM)
5. Fused residual_add + RMSNorm (2 ops → 1 kernel)
6. Fused RoPE (eliminate rotate_half's cat + neg overhead)

Model dimensions (Qwen3-VL-2B):
  LLM: hidden=2048, intermediate=6144, heads=16, kv_heads=8, head_dim=128, layers=28
  ViT: hidden=1024, heads=16, head_dim=64, layers=24
"""
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# GPU architecture detection and num_warps configuration
# ---------------------------------------------------------------------------
_compute_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (9, 0)
_is_ampere = _compute_capability[0] == 8   # SM80 (A800, A100, etc.)
_is_sm80_or_newer = _compute_capability[0] >= 8  # SM80+SM90 (A100, A800, H20, etc.)

NUM_WARPS_TINY = 8 if _is_ampere else 4
NUM_WARPS_SMALL = 8 if _is_ampere else 4
NUM_WARPS_MEDIUM = 8 if _is_ampere else 4
NUM_WARPS_LARGE = 16 if _is_ampere else 8
NUM_WARPS_XLARGE = 16 if _is_sm80_or_newer else 8

# ---------------------------------------------------------------------------
# Kernel 1: Fused RMSNorm
# ---------------------------------------------------------------------------
# Computes: out = x * rsqrt(mean(x^2) + eps) * gamma
# For decode: x is [H] vector, single kernel launch, all in SRAM

@triton.jit
def _rmsnorm_kernel(
    X_ptr,
    Gamma_ptr,
    Out_ptr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr = 2048,
):
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H
    x = tl.load(X_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    gamma = tl.load(Gamma_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / H
    rrms = tl.rsqrt(variance + EPS)
    out = (x * rrms * gamma).to(tl.float16)
    tl.store(Out_ptr + h_offsets, out, mask=h_mask)

def triton_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fast RMSNorm for small vectors (decode step)."""
    flat_x = x.view(-1)
    hidden_size = flat_x.shape[0]
    output = torch.empty_like(flat_x)
    BLOCK_H = triton.next_power_of_2(hidden_size)
    _rmsnorm_kernel[(1,)](flat_x, gamma, output, H=hidden_size, EPS=eps, BLOCK_H=BLOCK_H, num_warps=NUM_WARPS_TINY)
    return output.view_as(x)

# ---------------------------------------------------------------------------
# Kernel 2: Fused Residual Add + RMSNorm
# ---------------------------------------------------------------------------
# Computes: hidden = residual + x; out = RMSNorm(hidden)
# Saves: 1 kernel launch + 1 HBM read of hidden for RMSNorm

@triton.jit
def _fused_residual_rmsnorm_kernel(
    Residual_ptr,
    X_ptr,
    Gamma_ptr,
    NormOut_ptr,
    HiddenOut_ptr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr = 2048,
):
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    residual = tl.load(Residual_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    hidden = residual + x
    tl.store(HiddenOut_ptr + h_offsets, hidden.to(tl.float16), mask=h_mask)

    gamma = tl.load(Gamma_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=0) / H
    rrms = tl.rsqrt(variance + EPS)
    norm_out = (hidden * rrms * gamma).to(tl.float16)
    tl.store(NormOut_ptr + h_offsets, norm_out, mask=h_mask)

def fused_residual_rmsnorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple:
    """Fused residual add + RMSNorm. Returns (norm_output, new_hidden)."""
    flat_residual = residual.view(-1)
    flat_x = x.view(-1)
    hidden_size = flat_residual.shape[0]

    norm_output = torch.empty_like(flat_residual)
    new_hidden = torch.empty_like(flat_residual)

    BLOCK_H = triton.next_power_of_2(hidden_size)
    _fused_residual_rmsnorm_kernel[(1,)](
        flat_residual, flat_x, rmsnorm_weight, norm_output, new_hidden,
        H=hidden_size, EPS=eps, BLOCK_H=BLOCK_H, num_warps=NUM_WARPS_TINY,
    )
    return norm_output.view_as(residual), new_hidden.view_as(residual)


# ---------------------------------------------------------------------------
# Kernel 3: Fused SwiGLU
# ---------------------------------------------------------------------------
# Input: gate_up_output [2*I] (concatenated gate and up projections)
# Output: silu(gate) * up [I]
# Saves: avoids writing gate and up separately, then reading them back

@triton.jit
def _fused_swiglu_kernel(
    GateUp_ptr,     # [2*I] concatenated gate and up
    Out_ptr,        # [I] output
    I: tl.constexpr,
    BLOCK_I: tl.constexpr = 8192,
):
    """Single-program SwiGLU for decode (seq=1).
    
    Uses BLOCK_I = next_power_of_2(I) to process all elements in one program,
    avoiding multi-program launch overhead for small tensors.
    """
    offsets = tl.arange(0, BLOCK_I)
    mask = offsets < I

    gate = tl.load(GateUp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + I + offsets, mask=mask, other=0.0).to(tl.float32)

    # SwiGLU: silu(gate) * up
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    tl.store(Out_ptr + offsets, result.to(tl.float16), mask=mask)

def fused_swiglu(gate_up_output: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    """Fused SwiGLU activation on concatenated gate+up output (decode: single token).

    Uses single program with BLOCK_I = next_power_of_2(I) to minimize launch overhead.
    For I=6144, BLOCK_I=8192 → 1 program instead of 6.
    """
    original_shape = gate_up_output.shape
    flat = gate_up_output.view(-1)
    output = torch.empty(intermediate_size, dtype=gate_up_output.dtype, device=gate_up_output.device)

    BLOCK_I = triton.next_power_of_2(intermediate_size)
    _fused_swiglu_kernel[(1,)](flat, output, I=intermediate_size, BLOCK_I=BLOCK_I, num_warps=NUM_WARPS_LARGE)

    out_shape = list(original_shape)
    out_shape[-1] = intermediate_size
    return output.view(*out_shape)


# =========================================================================
# Batch-mode kernels for prefill (seq_len > 1) and ViT
# =========================================================================

# ---------------------------------------------------------------------------
# Kernel 4: Batch RMSNorm (for LLM prefill)
# ---------------------------------------------------------------------------
# Processes [S, H] tensor, one row per program

@triton.jit
def _batch_rmsnorm_kernel(
    X_ptr,
    Gamma_ptr,
    Out_ptr,
    stride_x_row,
    stride_out_row,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr = 2048,
):
    row = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    x = tl.load(X_ptr + row * stride_x_row + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    gamma = tl.load(Gamma_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / H
    rrms = tl.rsqrt(variance + EPS)
    out = (x * rrms * gamma).to(tl.float16)
    tl.store(Out_ptr + row * stride_out_row + h_offsets, out, mask=h_mask)

def batch_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm for [*, H] tensors (prefill batch mode)."""
    original_shape = x.shape
    flat_x = x.reshape(-1, original_shape[-1])
    num_rows = flat_x.shape[0]
    hidden_size = flat_x.shape[1]
    output = torch.empty_like(flat_x)
    BLOCK_H = triton.next_power_of_2(hidden_size)
    _batch_rmsnorm_kernel[(num_rows,)](
        flat_x, gamma, output,
        flat_x.stride(0), output.stride(0),
        H=hidden_size, EPS=eps, BLOCK_H=BLOCK_H, num_warps=NUM_WARPS_LARGE,
    )
    return output.view(original_shape)

# ---------------------------------------------------------------------------
# Kernel 5: Batch Fused Residual + RMSNorm (for LLM prefill)
# ---------------------------------------------------------------------------

@triton.jit
def _batch_fused_residual_rmsnorm_kernel(
    Residual_ptr,
    X_ptr,
    Gamma_ptr,
    NormOut_ptr,
    HiddenOut_ptr,
    stride_r,
    stride_x,
    stride_norm,
    stride_hidden,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr = 2048,
):
    row = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    residual = tl.load(Residual_ptr + row * stride_r + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    hidden = residual + x
    tl.store(HiddenOut_ptr + row * stride_hidden + h_offsets, hidden.to(tl.float16), mask=h_mask)

    gamma = tl.load(Gamma_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    variance = tl.sum(hidden * hidden, axis=0) / H
    rrms = tl.rsqrt(variance + EPS)
    norm_out = (hidden * rrms * gamma).to(tl.float16)
    tl.store(NormOut_ptr + row * stride_norm + h_offsets, norm_out, mask=h_mask)

def batch_fused_residual_rmsnorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple:
    """Batch fused residual add + RMSNorm for [*, H] tensors."""
    original_shape = residual.shape
    flat_r = residual.reshape(-1, original_shape[-1])
    flat_x = x.reshape(-1, original_shape[-1])
    num_rows = flat_r.shape[0]
    hidden_size = flat_r.shape[1]

    norm_output = torch.empty_like(flat_r)
    new_hidden = torch.empty_like(flat_r)

    BLOCK_H = triton.next_power_of_2(hidden_size)
    _batch_fused_residual_rmsnorm_kernel[(num_rows,)](
        flat_r, flat_x, rmsnorm_weight, norm_output, new_hidden,
        flat_r.stride(0), flat_x.stride(0), norm_output.stride(0), new_hidden.stride(0),
        H=hidden_size, EPS=eps, BLOCK_H=BLOCK_H, num_warps=NUM_WARPS_LARGE,
    )
    return norm_output.view(original_shape), new_hidden.view(original_shape)

# ---------------------------------------------------------------------------
# Kernel 6: Batch Fused SwiGLU (for LLM prefill)
# ---------------------------------------------------------------------------
# Kernel 6: Batch Fused SwiGLU (for LLM prefill)


# ---------------------------------------------------------------------------
# Kernel 6: Batch Fused SwiGLU (for LLM prefill)
# ---------------------------------------------------------------------------

@triton.jit
def _batch_fused_swiglu_kernel(
    GateUp_ptr,
    Out_ptr,
    stride_in,
    stride_out,
    I: tl.constexpr,
    BLOCK_I: tl.constexpr = 2048,
):
    row = tl.program_id(0)
    pid_col = tl.program_id(1)
    offsets = pid_col * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = offsets < I

    gate = tl.load(GateUp_ptr + row * stride_in + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + row * stride_in + I + offsets, mask=mask, other=0.0).to(tl.float32)

    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up

    tl.store(Out_ptr + row * stride_out + offsets, result.to(tl.float16), mask=mask)

def batch_fused_swiglu(gate_up_output: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    """Batch fused SwiGLU for [*, 2*I] tensors (prefill mode)."""
    original_shape = gate_up_output.shape
    flat = gate_up_output.reshape(-1, original_shape[-1])
    num_rows = flat.shape[0]
    output = torch.empty(num_rows, intermediate_size, dtype=gate_up_output.dtype, device=gate_up_output.device)

    BLOCK_I = 2048
    grid = (num_rows, triton.cdiv(intermediate_size, BLOCK_I))
    _batch_fused_swiglu_kernel[grid](
        flat, output, flat.stride(0), output.stride(0),
        I=intermediate_size, BLOCK_I=BLOCK_I, num_warps=NUM_WARPS_XLARGE,
    )

    out_shape = list(original_shape)
    out_shape[-1] = intermediate_size
    return output.view(*out_shape)


@triton.jit
def _batch_layernorm_kernel(
    X_ptr,
    Gamma_ptr,
    Beta_ptr,
    Out_ptr,
    stride_x,
    stride_out,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr = 1024,
):
    row = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    x = tl.load(X_ptr + row * stride_x + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    gamma = tl.load(Gamma_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / H
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / H
    rstd = tl.rsqrt(variance + EPS)
    out = (x_centered * rstd * gamma + beta).to(tl.float16)
    tl.store(Out_ptr + row * stride_out + h_offsets, out, mask=h_mask)

def batch_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fast LayerNorm for [*, H] tensors (ViT)."""
    original_shape = x.shape
    flat_x = x.reshape(-1, original_shape[-1])
    num_rows = flat_x.shape[0]
    hidden_size = flat_x.shape[1]
    output = torch.empty_like(flat_x)
    BLOCK_H = triton.next_power_of_2(hidden_size)
    _batch_layernorm_kernel[(num_rows,)](
        flat_x, weight, bias, output,
        flat_x.stride(0), output.stride(0),
        H=hidden_size, EPS=eps, BLOCK_H=BLOCK_H, num_warps=NUM_WARPS_LARGE,
    )
    return output.view(original_shape)


# ---------------------------------------------------------------------------
# Kernel 8: Batch Fused Residual + LayerNorm (for ViT)
# ---------------------------------------------------------------------------
# Computes: hidden = residual + x; out = LayerNorm(hidden)

@triton.jit
def _batch_fused_residual_layernorm_kernel(
    Residual_ptr,
    X_ptr,
    Gamma_ptr,
    Beta_ptr,
    NormOut_ptr,
    HiddenOut_ptr,
    stride_r,
    stride_x,
    stride_norm,
    stride_hidden,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr = 1024,
):
    row = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    residual = tl.load(Residual_ptr + row * stride_r + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + row * stride_x + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    hidden = residual + x
    tl.store(HiddenOut_ptr + row * stride_hidden + h_offsets, hidden.to(tl.float16), mask=h_mask)

    gamma = tl.load(Gamma_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)

    mean = tl.sum(hidden, axis=0) / H
    hidden_centered = hidden - mean
    variance = tl.sum(hidden_centered * hidden_centered, axis=0) / H
    rstd = tl.rsqrt(variance + EPS)
    norm_out = (hidden_centered * rstd * gamma + beta).to(tl.float16)
    tl.store(NormOut_ptr + row * stride_norm + h_offsets, norm_out, mask=h_mask)


def batch_fused_residual_layernorm(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> tuple:
    """Batch fused residual add + LayerNorm for [*, H] tensors (ViT)."""
    original_shape = residual.shape
    flat_r = residual.reshape(-1, original_shape[-1])
    flat_x = x.reshape(-1, original_shape[-1])
    num_rows = flat_r.shape[0]
    hidden_size = flat_r.shape[1]

    norm_output = torch.empty_like(flat_r)
    new_hidden = torch.empty_like(flat_r)

    BLOCK_H = triton.next_power_of_2(hidden_size)
    _batch_fused_residual_layernorm_kernel[(num_rows,)](
        flat_r, flat_x, weight, bias, norm_output, new_hidden,
        flat_r.stride(0), flat_x.stride(0), norm_output.stride(0), new_hidden.stride(0),
        H=hidden_size, EPS=eps, BLOCK_H=BLOCK_H, num_warps=NUM_WARPS_LARGE,
    )
    return norm_output.view(original_shape), new_hidden.view(original_shape)

# =========================================================================
# Fused GELU activation kernel (for ViT MLP)
# =========================================================================
# GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# PyTorch uses the tanh approximation for gelu_pytorch_tanh

@triton.jit
def _batch_gelu_tanh_kernel(
    X_ptr, Out_ptr,
    stride_x_row, stride_out_row,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr = 4096,
):
    """Batch GELU with tanh approximation, one row per program, tiled over columns."""
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    offsets = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offsets < C

    x = tl.load(X_ptr + row * stride_x_row + offsets, mask=mask, other=0.0).to(tl.float32)

    # GELU tanh approximation: tanh(z) = 2*sigmoid(2z) - 1 (leverages HW sigmoid)
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    result = (0.5 * x * (1.0 + tanh_val)).to(tl.float16)

    tl.store(Out_ptr + row * stride_out_row + offsets, result, mask=mask)


def batch_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """Batch GELU (tanh approximation) for [*, C] tensors."""
    original_shape = x.shape
    flat_x = x.reshape(-1, original_shape[-1])
    num_rows = flat_x.shape[0]
    hidden_size = flat_x.shape[1]
    output = torch.empty_like(flat_x)

    BLOCK_C = 4096
    grid = (num_rows, triton.cdiv(hidden_size, BLOCK_C))
    _batch_gelu_tanh_kernel[grid](
        flat_x, output,
        flat_x.stride(0), output.stride(0),
        C=hidden_size, BLOCK_C=BLOCK_C, num_warps=NUM_WARPS_XLARGE,
    )
    return output.view(original_shape)


# ---------------------------------------------------------------------------
# Kernel: Fused Dual RMSNorm + Concatenate (for Eagle3 draft)
# ---------------------------------------------------------------------------
# Replaces: rmsnorm(embeds) + rmsnorm(hidden) + torch.cat → 1 kernel
# Each program: norm both halves, write concatenated [2*H] output

@triton.jit
def _fused_dual_rmsnorm_cat_kernel(
    E_ptr, H_ptr, Gamma_e_ptr, Gamma_h_ptr, Out_ptr,
    stride_e, stride_h, stride_out,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr = 2048,
):
    row = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < HIDDEN

    e = tl.load(E_ptr + row * stride_e + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    gamma_e = tl.load(Gamma_e_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    var_e = tl.sum(e * e, axis=0) / HIDDEN
    normed_e = (e * tl.rsqrt(var_e + EPS) * gamma_e).to(tl.float16)
    tl.store(Out_ptr + row * stride_out + h_offsets, normed_e, mask=h_mask)

    h = tl.load(H_ptr + row * stride_h + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    gamma_h = tl.load(Gamma_h_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    var_h = tl.sum(h * h, axis=0) / HIDDEN
    normed_h = (h * tl.rsqrt(var_h + EPS) * gamma_h).to(tl.float16)
    tl.store(Out_ptr + row * stride_out + HIDDEN + h_offsets, normed_h, mask=h_mask)


def fused_dual_rmsnorm_cat(
    embeds: torch.Tensor,
    hidden: torch.Tensor,
    gamma_e: torch.Tensor,
    gamma_h: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Fused dual RMSNorm + concatenate: [normed_e || normed_h].

    Works for both decode (seq=1) and batch (seq>1).
    """
    original_shape = embeds.shape
    H = original_shape[-1]
    flat_e = embeds.reshape(-1, H)
    flat_h = hidden.reshape(-1, H)
    num_rows = flat_e.shape[0]

    if out is None:
        out_flat = torch.empty(num_rows, 2 * H, dtype=embeds.dtype, device=embeds.device)
    else:
        out_flat = out.reshape(num_rows, 2 * H)

    BLOCK_H = triton.next_power_of_2(H)
    nw = NUM_WARPS_TINY if num_rows == 1 else NUM_WARPS_LARGE
    _fused_dual_rmsnorm_cat_kernel[(num_rows,)](
        flat_e, flat_h, gamma_e, gamma_h, out_flat,
        flat_e.stride(0), flat_h.stride(0), out_flat.stride(0),
        HIDDEN=H, EPS=eps, BLOCK_H=BLOCK_H, num_warps=nw,
    )

    out_shape = list(original_shape)
    out_shape[-1] = 2 * H
    return out_flat.view(*out_shape)


# ---------------------------------------------------------------------------
# Kernel: Fused in-place Q+K RoPE (for Eagle3 draft)
# ---------------------------------------------------------------------------
# Replaces: _apply_rope(q) + _apply_rope(k) → 1 kernel for all heads
# Each program handles one (token, head) pair with rotate_half RoPE.

@triton.jit
def _fused_qk_rope_kernel(
    Q_ptr, K_ptr, Cos_ptr, Sin_ptr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_k_s, stride_k_h, stride_k_d,
    stride_cos_s, stride_cos_d,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    seq_idx = pid // TOTAL_HEADS
    head_in_batch = pid % TOTAL_HEADS
    is_q = head_in_batch < NUM_Q_HEADS

    offsets = tl.arange(0, BLOCK_HD)
    mask = offsets < HEAD_DIM

    cos = tl.load(Cos_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)

    rot_offsets = tl.where(offsets < HALF_DIM, offsets + HALF_DIM, offsets - HALF_DIM)
    sign = tl.where(offsets < HALF_DIM, -1.0, 1.0)

    if is_q:
        head_idx = head_in_batch
        base = seq_idx * stride_q_s + head_idx * stride_q_h
        x = tl.load(Q_ptr + base + offsets * stride_q_d, mask=mask, other=0.0).to(tl.float32)
        x_rot = tl.load(Q_ptr + base + rot_offsets * stride_q_d, mask=mask, other=0.0).to(tl.float32)
        result = (x * cos + x_rot * sign * sin).to(tl.float16)
        tl.store(Q_ptr + base + offsets * stride_q_d, result, mask=mask)
    else:
        head_idx = head_in_batch - NUM_Q_HEADS
        base = seq_idx * stride_k_s + head_idx * stride_k_h
        x = tl.load(K_ptr + base + offsets * stride_k_d, mask=mask, other=0.0).to(tl.float32)
        x_rot = tl.load(K_ptr + base + rot_offsets * stride_k_d, mask=mask, other=0.0).to(tl.float32)
        result = (x * cos + x_rot * sign * sin).to(tl.float16)
        tl.store(K_ptr + base + offsets * stride_k_d, result, mask=mask)


def fused_qk_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
):
    """In-place fused Q+K RoPE for all heads in a single kernel launch.

    Works for both decode (seq=1) and batch (seq>1).
    Uses rotate_half pattern, equivalent to the half-dim formulation
    when cos/sin are symmetric (cos[i] == cos[i+half]).

    Args:
        q: [B, S, num_q_heads, head_dim] — modified in-place
        k: [B, S, num_kv_heads, head_dim] — modified in-place
        cos: broadcastable to [total_tokens, head_dim]
        sin: broadcastable to [total_tokens, head_dim]
    """
    head_dim = q.shape[-1]
    half_dim = head_dim // 2
    total_heads = num_q_heads + num_kv_heads

    q_3d = q.view(-1, num_q_heads, head_dim)
    k_3d = k.view(-1, num_kv_heads, head_dim)
    total_tokens = q_3d.shape[0]

    cos_2d = cos.reshape(total_tokens, head_dim).contiguous()
    sin_2d = sin.reshape(total_tokens, head_dim).contiguous()

    BLOCK_HD = triton.next_power_of_2(head_dim)
    _fused_qk_rope_kernel[(total_tokens * total_heads,)](
        q_3d, k_3d, cos_2d, sin_2d,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
        cos_2d.stride(0), cos_2d.stride(1),
        NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads,
        TOTAL_HEADS=total_heads,
        HEAD_DIM=head_dim, HALF_DIM=half_dim,
        BLOCK_HD=BLOCK_HD, num_warps=NUM_WARPS_SMALL,
    )


# ---------------------------------------------------------------------------
# Kernel: Fused Greedy Decode Output (for Eagle3 draft)
# ---------------------------------------------------------------------------
# Replaces: logits.max() + logits.argmax() + torch.gather(d2t) + copy_×2 → 1 kernel
# Single block scans logits to find max value and argmax, looks up draft_to_target,
# writes target_id and max_logit directly to output buffers.

@triton.jit
def _fused_greedy_output_kernel(
    Logits_ptr,
    D2T_ptr,
    Out_token_ptr,
    Out_confidence_ptr,
    VOCAB: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    best_val = -float('inf')
    best_idx = tl.zeros([], dtype=tl.int32)
    for start in tl.static_range(0, BLOCK_V, 4096):
        offsets = start + tl.arange(0, 4096)
        mask = offsets < VOCAB
        vals = tl.load(Logits_ptr + offsets, mask=mask, other=-float('inf')).to(tl.float32)
        chunk_max = tl.max(vals)
        gt = chunk_max > best_val
        chunk_argmax = tl.argmax(vals, axis=0)
        best_idx = tl.where(gt, (start + chunk_argmax).to(tl.int32), best_idx)
        best_val = tl.where(gt, chunk_max, best_val)

    target_id = tl.load(D2T_ptr + best_idx)
    tl.store(Out_token_ptr, target_id)
    tl.store(Out_confidence_ptr, best_val)


def fused_greedy_output(
    logits: torch.Tensor,
    draft_to_target: torch.Tensor,
    out_token: torch.Tensor,
    out_confidence: torch.Tensor,
):
    """Fused greedy decode: argmax(logits) → draft_to_target lookup → write outputs.

    Args:
        logits: [1, V] or [V] float16 logits
        draft_to_target: [V] int64 mapping table
        out_token: scalar int64 buffer for target token id
        out_confidence: scalar float32 buffer for max logit value
    """
    flat = logits.view(-1)
    V = flat.shape[0]
    BLOCK_V = triton.cdiv(V, 4096) * 4096
    _fused_greedy_output_kernel[(1,)](
        flat, draft_to_target, out_token, out_confidence,
        VOCAB=V, BLOCK_V=BLOCK_V, num_warps=NUM_WARPS_LARGE,
    )


# ---------------------------------------------------------------------------
# Kernel: Fused Embedding + Dual RMSNorm + Cat (for Eagle3 draft decode)
# ---------------------------------------------------------------------------
# Replaces: F.embedding(token_id, weight) + rmsnorm(embed) + rmsnorm(hidden) + cat → 1 kernel
# Single block: loads embedding row by token_id, normalizes both embed and hidden,
# writes concatenated [2*H] output.

@triton.jit
def _fused_embed_dual_rmsnorm_cat_kernel(
    Token_id_ptr,
    Embed_ptr,
    H_ptr,
    Gamma_e_ptr, Gamma_h_ptr,
    Out_ptr,
    stride_embed_v,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr = 2048,
):
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < HIDDEN

    token_id = tl.load(Token_id_ptr).to(tl.int64)
    e = tl.load(Embed_ptr + token_id * stride_embed_v + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    gamma_e = tl.load(Gamma_e_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    var_e = tl.sum(e * e, axis=0) / HIDDEN
    normed_e = (e * tl.rsqrt(var_e + EPS) * gamma_e).to(tl.float16)
    tl.store(Out_ptr + h_offsets, normed_e, mask=h_mask)

    h = tl.load(H_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    gamma_h = tl.load(Gamma_h_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
    var_h = tl.sum(h * h, axis=0) / HIDDEN
    normed_h = (h * tl.rsqrt(var_h + EPS) * gamma_h).to(tl.float16)
    tl.store(Out_ptr + HIDDEN + h_offsets, normed_h, mask=h_mask)


def fused_embed_dual_rmsnorm_cat(
    token_id: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden: torch.Tensor,
    gamma_e: torch.Tensor,
    gamma_h: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Fused embedding lookup + dual RMSNorm + concatenate for decode (seq=1).

    Args:
        token_id: scalar or [1] int64 token index
        embed_weight: [V, H] embedding table
        hidden: [1, 1, H] or [H] hidden state
        gamma_e, gamma_h: [H] RMSNorm weights
        out: optional pre-allocated [1, 1, 2*H] output buffer
    Returns:
        [1, 1, 2*H] concatenated normalized output
    """
    H = embed_weight.shape[1]
    flat_h = hidden.view(-1)
    flat_token = token_id.view(-1)

    if out is None:
        out_flat = torch.empty(2 * H, dtype=embed_weight.dtype, device=embed_weight.device)
    else:
        out_flat = out.view(-1)

    BLOCK_H = triton.next_power_of_2(H)
    _fused_embed_dual_rmsnorm_cat_kernel[(1,)](
        flat_token, embed_weight, flat_h, gamma_e, gamma_h, out_flat,
        embed_weight.stride(0),
        HIDDEN=H, EPS=eps, BLOCK_H=BLOCK_H, num_warps=NUM_WARPS_TINY,
    )
    return out_flat.view(1, 1, 2 * H)


# =========================================================================
# Fused Rotary Position Embedding (RoPE) kernels
# =========================================================================
# Eliminates rotate_half's cat + neg overhead by computing in-register:
#   out[i] = x[i] * cos[i] + (-x[i + half] if i < half else x[i - half]) * sin[i]
# This replaces 6+ kernel launches (mul, neg, cat, mul, add, copy) with 1.


# ---------------------------------------------------------------------------
# Kernel 9: Fused RoPE for ViT
# ---------------------------------------------------------------------------
# ViT RoPE: q/k shape [S, num_heads, head_dim], cos/sin [S, head_dim]
# Processes q and k simultaneously (same cos/sin), one (token, head) per program.
# In-place output variant: writes directly back to q/k buffers to avoid allocation.

@triton.jit
def _fused_rope_vision_kernel(
    Q_ptr, K_ptr,
    Cos_ptr, Sin_ptr,
    Q_out_ptr, K_out_ptr,
    stride_q_s, stride_q_h, stride_q_d,
    stride_k_s, stride_k_h, stride_k_d,
    stride_cos_s, stride_cos_d,
    stride_qo_s, stride_qo_h, stride_qo_d,
    stride_ko_s, stride_ko_h, stride_ko_d,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr = 64,
):
    pid = tl.program_id(0)
    seq_idx = pid // NUM_HEADS
    head_idx = pid % NUM_HEADS

    offsets = tl.arange(0, BLOCK_HD)
    mask = offsets < HEAD_DIM

    cos = tl.load(Cos_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)

    # Load Q once, construct rotated version from registers (eliminates 2nd HBM load)
    q = tl.load(Q_ptr + seq_idx * stride_q_s + head_idx * stride_q_h + offsets * stride_q_d,
                mask=mask, other=0.0).to(tl.float32)
    # Register-level rotate_half: [-q[half:], q[:half]]
    rot_offsets = tl.where(offsets < HALF_DIM, offsets + HALF_DIM, offsets - HALF_DIM)
    q_rot = tl.load(Q_ptr + seq_idx * stride_q_s + head_idx * stride_q_h + rot_offsets * stride_q_d,
                    mask=mask, other=0.0).to(tl.float32)
    sign = tl.where(offsets < HALF_DIM, -1.0, 1.0)

    q_out = (q * cos + q_rot * sign * sin).to(tl.float16)
    tl.store(Q_out_ptr + seq_idx * stride_qo_s + head_idx * stride_qo_h + offsets * stride_qo_d,
             q_out, mask=mask)

    # Load K once, same rotate pattern
    k = tl.load(K_ptr + seq_idx * stride_k_s + head_idx * stride_k_h + offsets * stride_k_d,
                mask=mask, other=0.0).to(tl.float32)
    k_rot = tl.load(K_ptr + seq_idx * stride_k_s + head_idx * stride_k_h + rot_offsets * stride_k_d,
                    mask=mask, other=0.0).to(tl.float32)

    k_out = (k * cos + k_rot * sign * sin).to(tl.float16)
    tl.store(K_out_ptr + seq_idx * stride_ko_s + head_idx * stride_ko_h + offsets * stride_ko_d,
             k_out, mask=mask)


def fused_rope_vision(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple:
    """Fused in-place RoPE for ViT. q/k: [S, num_heads, head_dim], cos/sin: [S, head_dim].

    Writes results back to q and k buffers directly (safe because each program
    loads all 64 dims into registers before writing). Eliminates 2 empty_like allocs.
    """
    seq_len, num_heads, head_dim = q.shape
    half_dim = head_dim // 2

    # cos/sin shape [S, head_dim], already contiguous from cache
    cos_2d = cos.view(seq_len, head_dim).contiguous()
    sin_2d = sin.view(seq_len, head_dim).contiguous()

    grid = (seq_len * num_heads,)
    BLOCK_HD = triton.next_power_of_2(head_dim)
    # In-place: output pointers == input pointers
    _fused_rope_vision_kernel[grid](
        q, k, cos_2d, sin_2d, q, k,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        cos_2d.stride(0), cos_2d.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        NUM_HEADS=num_heads, HEAD_DIM=head_dim, HALF_DIM=half_dim,
        BLOCK_HD=BLOCK_HD, num_warps=NUM_WARPS_SMALL,
    )
    return q, k


# ---------------------------------------------------------------------------
# Kernel 10: Fused RoPE for LLM (single tensor version)
# ---------------------------------------------------------------------------
# Processes one tensor (q or k) with RoPE, shape [B, num_heads, S, head_dim]
# cos/sin shape [S, head_dim] (pre-reshaped for batch=1)
# One program per (head, seq_pos) pair

@triton.jit
def _fused_rope_single_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
    stride_x_h, stride_x_s, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_o_h, stride_o_s, stride_o_d,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    head_idx = pid // SEQ_LEN
    seq_idx = pid % SEQ_LEN

    offsets = tl.arange(0, BLOCK_HD)
    mask = offsets < HEAD_DIM

    cos = tl.load(Cos_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)

    x = tl.load(X_ptr + head_idx * stride_x_h + seq_idx * stride_x_s + offsets * stride_x_d,
                mask=mask, other=0.0).to(tl.float32)

    rot_offsets = tl.where(offsets < HALF_DIM, offsets + HALF_DIM, offsets - HALF_DIM)
    sign = tl.where(offsets < HALF_DIM, -1.0, 1.0)
    x_rot = tl.load(X_ptr + head_idx * stride_x_h + seq_idx * stride_x_s + rot_offsets * stride_x_d,
                    mask=mask, other=0.0).to(tl.float32)
    x_rot = x_rot * sign

    result = (x * cos + x_rot * sin).to(tl.float16)
    tl.store(Out_ptr + head_idx * stride_o_h + seq_idx * stride_o_s + offsets * stride_o_d,
             result, mask=mask)


def fused_rope_llm(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple:
    """Fused RoPE for LLM. q: [B, num_q_heads, S, head_dim], k: [B, num_kv_heads, S, head_dim].

    cos/sin: [B, S, head_dim] or [B, 1, S, head_dim] (will be reshaped).
    For batch=1, processes all (head, seq_pos) pairs in parallel.
    Calls the kernel separately for q and k to keep parameter count manageable.
    """
    batch_size, num_q_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    half_dim = head_dim // 2

    cos_2d = cos.view(-1, head_dim).contiguous()
    sin_2d = sin.view(-1, head_dim).contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    BLOCK_HD = triton.next_power_of_2(head_dim)

    # Process Q
    _fused_rope_single_kernel[(num_q_heads * seq_len,)](
        q, cos_2d, sin_2d, q_out,
        q.stride(1), q.stride(2), q.stride(3),
        cos_2d.stride(0), cos_2d.stride(1),
        q_out.stride(1), q_out.stride(2), q_out.stride(3),
        SEQ_LEN=seq_len, HEAD_DIM=head_dim, HALF_DIM=half_dim,
        BLOCK_HD=BLOCK_HD, num_warps=NUM_WARPS_SMALL,
    )

    # Process K
    _fused_rope_single_kernel[(num_kv_heads * seq_len,)](
        k, cos_2d, sin_2d, k_out,
        k.stride(1), k.stride(2), k.stride(3),
        cos_2d.stride(0), cos_2d.stride(1),
        k_out.stride(1), k_out.stride(2), k_out.stride(3),
        SEQ_LEN=seq_len, HEAD_DIM=head_dim, HALF_DIM=half_dim,
        BLOCK_HD=BLOCK_HD, num_warps=NUM_WARPS_SMALL,
    )

    return q_out, k_out


# ---------------------------------------------------------------------------
# Kernel 11: Fused RMSNorm + RoPE for LLM prefill
# ---------------------------------------------------------------------------
# Combines per-head RMSNorm + RoPE into a single kernel, avoiding intermediate
# HBM writes between norm and RoPE. Each program handles one (token, head) pair.
# Input: x [B*S, num_heads, head_dim] (viewed from projection output)
# Output: [num_heads, B*S, head_dim] (transposed for attention)

@triton.jit
def _fused_rmsnorm_rope_kernel(
    X_ptr, Gamma_ptr, Cos_ptr, Sin_ptr, Out_ptr,
    stride_x_s, stride_x_h, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_o_h, stride_o_s, stride_o_d,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_HD: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    seq_idx = pid // NUM_HEADS
    head_idx = pid % NUM_HEADS

    offsets = tl.arange(0, BLOCK_HD)
    mask = offsets < HEAD_DIM

    # Load input vector for this (token, head)
    x = tl.load(X_ptr + seq_idx * stride_x_s + head_idx * stride_x_h + offsets * stride_x_d,
                mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: compute rrms once, apply to all elements
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    rrms = tl.rsqrt(variance + EPS)

    # Load gamma once, apply RMSNorm
    gamma = tl.load(Gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = x * rrms * gamma

    # Construct rotated normed vector: reuse x from registers via gather
    rot_offsets = tl.where(offsets < HALF_DIM, offsets + HALF_DIM, offsets - HALF_DIM)
    x_rot = tl.load(X_ptr + seq_idx * stride_x_s + head_idx * stride_x_h + rot_offsets * stride_x_d,
                    mask=mask, other=0.0).to(tl.float32)
    # Reuse gamma from registers: gamma_rot[i] = gamma[rot_offsets[i]]
    gamma_rot = tl.load(Gamma_ptr + rot_offsets, mask=mask, other=0.0).to(tl.float32)
    sign = tl.where(offsets < HALF_DIM, -1.0, 1.0)
    normed_rot = x_rot * rrms * gamma_rot * sign

    # RoPE: apply cos/sin
    cos = tl.load(Cos_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)

    result = (normed * cos + normed_rot * sin).to(tl.float16)

    # Store with transposed layout: [num_heads, S, head_dim]
    tl.store(Out_ptr + head_idx * stride_o_h + seq_idx * stride_o_s + offsets * stride_o_d,
             result, mask=mask)


def fused_rmsnorm_rope(
    x: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_heads: int,
) -> torch.Tensor:
    """Fused RMSNorm + RoPE for LLM prefill.

    Input: x [B, S, num_heads, head_dim] (viewed from projection output)
    Output: [B, num_heads, S, head_dim] (ready for attention)

    Replaces: batch_rmsnorm → transpose → fused_rope_single (3 kernels → 1)
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    x_3d = x.reshape(-1, num_heads, head_dim)  # [B*S, num_heads, head_dim]
    total_tokens = x_3d.shape[0]

    out = torch.empty(num_heads, total_tokens, head_dim, dtype=x.dtype, device=x.device)

    cos_2d = cos.view(-1, head_dim).contiguous()
    sin_2d = sin.view(-1, head_dim).contiguous()

    BLOCK_HD = triton.next_power_of_2(head_dim)

    _fused_rmsnorm_rope_kernel[(total_tokens * num_heads,)](
        x_3d, gamma, cos_2d, sin_2d, out,
        x_3d.stride(0), x_3d.stride(1), x_3d.stride(2),
        cos_2d.stride(0), cos_2d.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        NUM_HEADS=num_heads, HEAD_DIM=head_dim, HALF_DIM=half_dim,
        EPS=eps, BLOCK_HD=BLOCK_HD, num_warps=NUM_WARPS_MEDIUM,
    )

    return out.view(batch_size, num_heads, seq_len, head_dim)


# ---------------------------------------------------------------------------
# Kernel 11b: Fused RMSNorm + RoPE outputting [B, S, num_heads, head_dim]
# ---------------------------------------------------------------------------
# Same computation as fused_rmsnorm_rope but outputs in flash_attn-friendly
# layout [B, S, num_heads, head_dim] to avoid a transpose.

@triton.jit
def _fused_rmsnorm_rope_bshd_kernel(
    X_ptr, Gamma_ptr, Cos_ptr, Sin_ptr, Out_ptr,
    stride_x_s, stride_x_h, stride_x_d,
    stride_cos_s, stride_cos_d,
    stride_o_s, stride_o_h, stride_o_d,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_HD: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    seq_idx = pid // NUM_HEADS
    head_idx = pid % NUM_HEADS

    offsets = tl.arange(0, BLOCK_HD)
    mask = offsets < HEAD_DIM

    x = tl.load(X_ptr + seq_idx * stride_x_s + head_idx * stride_x_h + offsets * stride_x_d,
                mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    rrms = tl.rsqrt(variance + EPS)

    # Load gamma once, apply RMSNorm
    gamma = tl.load(Gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = x * rrms * gamma

    # Construct rotated normed vector
    rot_offsets = tl.where(offsets < HALF_DIM, offsets + HALF_DIM, offsets - HALF_DIM)
    x_rot = tl.load(X_ptr + seq_idx * stride_x_s + head_idx * stride_x_h + rot_offsets * stride_x_d,
                    mask=mask, other=0.0).to(tl.float32)
    gamma_rot = tl.load(Gamma_ptr + rot_offsets, mask=mask, other=0.0).to(tl.float32)
    sign = tl.where(offsets < HALF_DIM, -1.0, 1.0)
    normed_rot = x_rot * rrms * gamma_rot * sign

    # RoPE: apply cos/sin
    cos = tl.load(Cos_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)

    result = (normed * cos + normed_rot * sin).to(tl.float16)

    # Store with [S, num_heads, head_dim] layout (no transpose needed for flash_attn)
    tl.store(Out_ptr + seq_idx * stride_o_s + head_idx * stride_o_h + offsets * stride_o_d,
             result, mask=mask)


def fused_rmsnorm_rope_bshd(
    x: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_heads: int,
) -> torch.Tensor:
    """Fused RMSNorm + RoPE outputting [B, S, num_heads, head_dim] for flash_attn.

    Input: x [B, S, num_heads, head_dim] (viewed from projection output)
    Output: [B, S, num_heads, head_dim] (ready for flash_attn_func)

    Same computation as fused_rmsnorm_rope but avoids the transpose.
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    x_3d = x.reshape(-1, num_heads, head_dim)  # [B*S, num_heads, head_dim]
    total_tokens = x_3d.shape[0]

    out = torch.empty(total_tokens, num_heads, head_dim, dtype=x.dtype, device=x.device)

    cos_2d = cos.view(-1, head_dim).contiguous()
    sin_2d = sin.view(-1, head_dim).contiguous()

    BLOCK_HD = triton.next_power_of_2(head_dim)

    _fused_rmsnorm_rope_bshd_kernel[(total_tokens * num_heads,)](
        x_3d, gamma, cos_2d, sin_2d, out,
        x_3d.stride(0), x_3d.stride(1), x_3d.stride(2),
        cos_2d.stride(0), cos_2d.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        NUM_HEADS=num_heads, HEAD_DIM=head_dim, HALF_DIM=half_dim,
        EPS=eps, BLOCK_HD=BLOCK_HD, num_warps=NUM_WARPS_MEDIUM,
    )

    return out.view(batch_size, seq_len, num_heads, head_dim)


@triton.jit
def _batch_fused_qk_norm_rope_kernel(
    QKV_ptr,            # [total_tokens, (num_q_heads + num_kv_heads) * head_dim] flat
    Q_gamma_ptr,        # [head_dim]
    K_gamma_ptr,        # [head_dim]
    Cos_ptr,            # [total_tokens, head_dim]
    Sin_ptr,            # [total_tokens, head_dim]
    Q_out_ptr,          # [total_tokens, num_q_heads, head_dim]
    K_out_ptr,          # [total_tokens, num_kv_heads, head_dim]
    stride_qkv_s, stride_qkv_d,
    stride_cos_s, stride_cos_d,
    stride_q_s, stride_q_h, stride_q_d,
    stride_k_s, stride_k_h, stride_k_d,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_HD: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    seq_idx = pid // TOTAL_HEADS
    head_in_batch = pid % TOTAL_HEADS
    is_q_head = head_in_batch < NUM_Q_HEADS

    if is_q_head:
        head_idx = head_in_batch
        input_offset = seq_idx * stride_qkv_s + head_idx * HEAD_DIM
    else:
        head_idx = head_in_batch - NUM_Q_HEADS
        input_offset = seq_idx * stride_qkv_s + (NUM_Q_HEADS + head_idx) * HEAD_DIM

    offsets = tl.arange(0, BLOCK_HD)
    mask = offsets < HEAD_DIM

    x = tl.load(QKV_ptr + input_offset + offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    rrms = tl.rsqrt(variance + EPS)

    gamma_ptr = Q_gamma_ptr if is_q_head else K_gamma_ptr
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = x * rrms * gamma

    rot_offsets = tl.where(offsets < HALF_DIM, offsets + HALF_DIM, offsets - HALF_DIM)
    x_rot = tl.load(QKV_ptr + input_offset + rot_offsets, mask=mask, other=0.0).to(tl.float32)
    gamma_rot = tl.load(gamma_ptr + rot_offsets, mask=mask, other=0.0).to(tl.float32)
    sign = tl.where(offsets < HALF_DIM, -1.0, 1.0)
    normed_rot = x_rot * rrms * gamma_rot * sign

    cos = tl.load(Cos_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + seq_idx * stride_cos_s + offsets * stride_cos_d, mask=mask, other=0.0)

    result = (normed * cos + normed_rot * sin).to(tl.float16)

    if is_q_head:
        tl.store(Q_out_ptr + seq_idx * stride_q_s + head_idx * stride_q_h + offsets * stride_q_d,
                 result, mask=mask)
    else:
        tl.store(K_out_ptr + seq_idx * stride_k_s + head_idx * stride_k_h + offsets * stride_k_d,
                 result, mask=mask)


def batch_fused_qk_norm_rope(
    qkv_out: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple:
    """Fused Q+K RMSNorm + RoPE for batch verify (seq>1).

    Combines two separate fused_rmsnorm_rope_bshd calls into one kernel launch.

    Args:
        qkv_out: [1, seq_len, (num_q_heads + num_kv_heads) * head_dim] Q+K portion of QKV
        q_gamma: [head_dim] Q norm weight
        k_gamma: [head_dim] K norm weight
        cos: [1, seq_len, head_dim] RoPE cos
        sin: [1, seq_len, head_dim] RoPE sin
        eps: RMSNorm epsilon
        num_q_heads: number of Q heads (16)
        num_kv_heads: number of KV heads (8)
        head_dim: head dimension (128)

    Returns:
        q_out: [1, seq_len, num_q_heads, head_dim]
        k_out: [1, seq_len, num_kv_heads, head_dim]
    """
    seq_len = qkv_out.shape[1]
    half_dim = head_dim // 2
    total_heads = num_q_heads + num_kv_heads

    # Reshape QKV to [seq_len, total_heads * head_dim]
    qkv_2d = qkv_out.view(seq_len, -1)

    cos_2d = cos.view(seq_len, head_dim)
    sin_2d = sin.view(seq_len, head_dim)

    q_out = torch.empty(seq_len, num_q_heads, head_dim, dtype=qkv_out.dtype, device=qkv_out.device)
    k_out = torch.empty(seq_len, num_kv_heads, head_dim, dtype=qkv_out.dtype, device=qkv_out.device)

    BLOCK_HD = triton.next_power_of_2(head_dim)
    grid = (seq_len * total_heads,)

    _batch_fused_qk_norm_rope_kernel[grid](
        qkv_2d, q_gamma, k_gamma, cos_2d, sin_2d, q_out, k_out,
        qkv_2d.stride(0), qkv_2d.stride(1),
        cos_2d.stride(0), cos_2d.stride(1),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads,
        TOTAL_HEADS=total_heads,
        HEAD_DIM=head_dim, HALF_DIM=half_dim, EPS=eps,
        BLOCK_HD=BLOCK_HD, num_warps=NUM_WARPS_MEDIUM,
    )

    return q_out.view(1, seq_len, num_q_heads, head_dim), k_out.view(1, seq_len, num_kv_heads, head_dim)


# =========================================================================
# Flash Decoding: Split-KV Attention for decode (batch=1, seq_q=1)
# =========================================================================
# Standard SDPA launches 1 block per head → 16 blocks for 16 heads on 78 SMs.
# Flash Decoding splits KV cache along sequence dim, launching num_heads * num_splits
# blocks, fully utilizing all SMs.
#
# Single-kernel approach: each program handles one (q_head, kv_split) pair,
# computes partial attention with online softmax, stores (acc, m, d).
# Then PyTorch merges results on GPU (simple vectorized ops, no Python loop).

@triton.jit
def _flash_decode_kernel(
    Q_ptr,          # [num_q_heads, head_dim]
    K_ptr,          # [num_kv_heads, max_cache_len, head_dim]
    V_ptr,          # [num_kv_heads, max_cache_len, head_dim]
    Acc_ptr,        # [num_q_heads, num_splits_padded, head_dim] output accumulators
    M_ptr,          # [num_q_heads, num_splits_padded] max scores
    D_ptr,          # [num_q_heads, num_splits_padded] sum-of-exp
    stride_q_h,
    stride_k_h, stride_k_s, stride_k_d,
    stride_v_h, stride_v_s, stride_v_d,
    stride_acc_h, stride_acc_split, stride_acc_d,
    stride_md_h,
    actual_seq_len,
    scale,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    TILE_KV: tl.constexpr,
):
    q_head_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // NUM_KV_GROUPS

    hd_range = tl.arange(0, HEAD_DIM)

    # Load query: [HEAD_DIM]
    q = tl.load(Q_ptr + q_head_idx * stride_q_h + hd_range).to(tl.float32)

    kv_start = split_idx * BLOCK_KV
    kv_end = tl.minimum(kv_start + BLOCK_KV, actual_seq_len)

    # Online softmax accumulators
    m_i = float("-inf")
    d_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    num_tiles: tl.constexpr = BLOCK_KV // TILE_KV
    for tile in tl.static_range(num_tiles):
        tile_start = kv_start + tile * TILE_KV
        kv_range = tile_start + tl.arange(0, TILE_KV)
        kv_mask = kv_range < kv_end

        # Load K tile: [TILE_KV, HEAD_DIM]
        k_ptrs = K_ptr + kv_head_idx * stride_k_h + kv_range[:, None] * stride_k_s + hd_range[None, :] * stride_k_d
        k_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # Q·K^T: [TILE_KV]
        scores = tl.sum(k_tile * q[None, :], axis=1) * scale
        scores = tl.where(kv_mask, scores, float("-inf"))

        # Online softmax update
        tile_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        p = tl.where(kv_mask, p, 0.0)

        d_i = d_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha

        # Load V tile and accumulate: [TILE_KV, HEAD_DIM]
        v_ptrs = V_ptr + kv_head_idx * stride_v_h + kv_range[:, None] * stride_v_s + hd_range[None, :] * stride_v_d
        v_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(p[:, None] * v_tile, axis=0)

        m_i = m_new

    # Store results
    tl.store(M_ptr + q_head_idx * stride_md_h + split_idx, m_i)
    tl.store(D_ptr + q_head_idx * stride_md_h + split_idx, d_i)
    acc_ptrs = Acc_ptr + q_head_idx * stride_acc_h + split_idx * stride_acc_split + hd_range * stride_acc_d
    tl.store(acc_ptrs, acc)


# too slow..., not ready yet.
@torch.compiler.disable
def flash_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    actual_seq_len,
    num_kv_groups: int = 2,
) -> torch.Tensor:
    """Flash Decoding for batch=1, seq_q=1.

    Decorated with @torch.compiler.disable to avoid graph break from .item().
    torch.compile still optimizes the rest of the model (GEMV, RMSNorm, etc).

    Args:
        query: [1, num_q_heads, 1, head_dim]
        key_cache: [1, num_kv_heads, max_cache_len, head_dim]
        value_cache: [1, num_kv_heads, max_cache_len, head_dim]
        actual_seq_len: tensor or int - valid tokens in cache
        num_kv_groups: num_q_heads // num_kv_heads

    Returns:
        [1, num_q_heads, 1, head_dim]
    """
    import torch.nn.functional as F

    num_q_heads = query.shape[1]
    head_dim = query.shape[3]
    scale = 1.0 / (head_dim ** 0.5)

    # Convert tensor to int if needed
    if isinstance(actual_seq_len, torch.Tensor):
        actual_seq_len = int(actual_seq_len.item())

    # Short sequences: standard SDPA is fine (few KV tokens, no SM underutilization)
    if actual_seq_len <= 128:
        return F.scaled_dot_product_attention(
            query, key_cache[:, :, :actual_seq_len, :],
            value_cache[:, :, :actual_seq_len, :],
            attn_mask=None, dropout_p=0.0, scale=scale,
            is_causal=False, enable_gqa=True,
        )

    # Split parameters tuned for A100: larger TILE_KV reduces loop iterations,
    # BLOCK_KV=256 with TILE_KV=64 → 4 tiles/block (was 8), better ILP
    BLOCK_KV = 256
    TILE_KV = 64
    num_splits = (actual_seq_len + BLOCK_KV - 1) // BLOCK_KV
    num_splits_padded = triton.next_power_of_2(num_splits)

    # Flatten Q: [num_q_heads, head_dim]
    q_flat = query.view(num_q_heads, head_dim).contiguous()
    k_3d = key_cache.view(key_cache.shape[1], key_cache.shape[2], head_dim)
    v_3d = value_cache.view(value_cache.shape[1], value_cache.shape[2], head_dim)

    # Allocate outputs
    acc_buf = torch.zeros(num_q_heads, num_splits_padded, head_dim,
                          dtype=torch.float32, device=query.device)
    m_buf = torch.full((num_q_heads, num_splits_padded), float('-inf'),
                       dtype=torch.float32, device=query.device)
    d_buf = torch.zeros(num_q_heads, num_splits_padded,
                        dtype=torch.float32, device=query.device)

    # Launch kernel: grid = (num_q_heads, num_splits)
    _flash_decode_kernel[(num_q_heads, num_splits)](
        q_flat, k_3d, v_3d, acc_buf, m_buf, d_buf,
        q_flat.stride(0),
        k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
        v_3d.stride(0), v_3d.stride(1), v_3d.stride(2),
        acc_buf.stride(0), acc_buf.stride(1), acc_buf.stride(2),
        m_buf.stride(0),
        actual_seq_len,
        scale,
        NUM_KV_GROUPS=num_kv_groups,
        HEAD_DIM=head_dim,
        BLOCK_KV=BLOCK_KV,
        TILE_KV=TILE_KV,
        num_warps=NUM_WARPS_SMALL,
        num_stages=3,
    )

    # Stage 2: merge splits using PyTorch vectorized ops (no Python loop)
    global_max = m_buf.max(dim=1, keepdim=True).values
    rescale = torch.exp(m_buf - global_max)
    d_rescaled = d_buf * rescale
    total_d = d_rescaled.sum(dim=1, keepdim=True)
    acc_rescaled = acc_buf * rescale.unsqueeze(-1)
    result = acc_rescaled.sum(dim=1) / total_d.clamp(min=1e-10)

    return result.to(torch.float16).view(1, num_q_heads, 1, head_dim)


# =========================================================================
# Kernel 12: Fused Q/K RMSNorm + RoPE for decode (seq=1)
# =========================================================================
# For decode, Q has num_q_heads and K has num_kv_heads, each head_dim=128.
# This kernel processes ALL Q and K heads in a single launch:
#   - Programs 0..num_q_heads-1 handle Q heads (using q_gamma)
#   - Programs num_q_heads..num_q_heads+num_kv_heads-1 handle K heads (using k_gamma)
# Each program: load head vector → RMSNorm → RoPE → store
# Eliminates 8+ kernel launches (2x RMSNorm + rotate_half's cat/neg/mul/add for Q and K)

@triton.jit
def _fused_qk_norm_rope_decode_kernel(
    QKV_ptr,            # [q_dim + kv_dim + kv_dim] flat QKV output from GEMV
    Q_gamma_ptr,        # [head_dim] Q RMSNorm weight
    K_gamma_ptr,        # [head_dim] K RMSNorm weight
    Cos_ptr,            # [head_dim] cos for this position
    Sin_ptr,            # [head_dim] sin for this position
    Q_out_ptr,          # [num_q_heads, head_dim] output Q (BHSD with B=1,S=1 squeezed)
    K_out_ptr,          # [num_kv_heads, head_dim] output K
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_HD: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    is_q_head = pid < NUM_Q_HEADS

    # Determine which head index and gamma to use
    if is_q_head:
        head_idx = pid
        input_offset = head_idx * HEAD_DIM
    else:
        head_idx = pid - NUM_Q_HEADS
        # K starts after Q in the QKV buffer
        input_offset = NUM_Q_HEADS * HEAD_DIM + head_idx * HEAD_DIM

    offsets = tl.arange(0, BLOCK_HD)
    mask = offsets < HEAD_DIM

    # Load head vector from QKV buffer
    x = tl.load(QKV_ptr + input_offset + offsets, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: variance → rrms → normalize
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    rrms = tl.rsqrt(variance + EPS)

    # Load appropriate gamma (Q or K)
    gamma_ptr = Q_gamma_ptr if is_q_head else K_gamma_ptr
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    normed = x * rrms * gamma

    # RoPE: construct rotated version from registers
    rot_offsets = tl.where(offsets < HALF_DIM, offsets + HALF_DIM, offsets - HALF_DIM)
    x_rot = tl.load(QKV_ptr + input_offset + rot_offsets, mask=mask, other=0.0).to(tl.float32)
    gamma_rot = tl.load(gamma_ptr + rot_offsets, mask=mask, other=0.0).to(tl.float32)
    sign = tl.where(offsets < HALF_DIM, -1.0, 1.0)
    normed_rot = x_rot * rrms * gamma_rot * sign

    # Apply cos/sin
    cos = tl.load(Cos_ptr + offsets, mask=mask, other=0.0)
    sin = tl.load(Sin_ptr + offsets, mask=mask, other=0.0)
    result = (normed * cos + normed_rot * sin).to(tl.float16)

    # Store to Q or K output buffer
    if is_q_head:
        tl.store(Q_out_ptr + head_idx * HEAD_DIM + offsets, result, mask=mask)
    else:
        tl.store(K_out_ptr + head_idx * HEAD_DIM + offsets, result, mask=mask)


def fused_qk_norm_rope_decode(
    qkv_out: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_eps: float,
    k_eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple:
    """Fused Q/K RMSNorm + RoPE for decode (seq=1, batch=1).

    Replaces 4 separate operations (q_norm, k_norm, apply_rotary_pos_emb x2)
    with a single Triton kernel launch processing all Q and K heads.

    Args:
        qkv_out: [1, 1, q_dim + kv_dim + kv_dim] flat QKV projection output
        q_gamma: [head_dim] Q RMSNorm weight
        k_gamma: [head_dim] K RMSNorm weight
        cos: [1, 1, head_dim] or broadcastable cos for RoPE
        sin: [1, 1, head_dim] or broadcastable sin for RoPE
        q_eps: Q RMSNorm epsilon
        k_eps: K RMSNorm epsilon (assumed same as q_eps for fused kernel)
        num_q_heads: number of Q attention heads (16 for Qwen3-VL-2B)
        num_kv_heads: number of KV attention heads (8 for Qwen3-VL-2B)
        head_dim: head dimension (128 for Qwen3-VL-2B)

    Returns:
        query_states: [1, num_q_heads, 1, head_dim] BHSD
        key_states: [1, num_kv_heads, 1, head_dim] BHSD
    """
    half_dim = head_dim // 2
    total_heads = num_q_heads + num_kv_heads

    # Flatten inputs for kernel
    qkv_flat = qkv_out.view(-1)
    cos_flat = cos.view(-1)[:head_dim].contiguous()
    sin_flat = sin.view(-1)[:head_dim].contiguous()

    # Output buffers
    q_out = torch.empty(num_q_heads, head_dim, dtype=qkv_out.dtype, device=qkv_out.device)
    k_out = torch.empty(num_kv_heads, head_dim, dtype=qkv_out.dtype, device=qkv_out.device)

    BLOCK_HD = triton.next_power_of_2(head_dim)

    _fused_qk_norm_rope_decode_kernel[(total_heads,)](
        qkv_flat, q_gamma, k_gamma, cos_flat, sin_flat,
        q_out, k_out,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HALF_DIM=half_dim,
        EPS=q_eps,
        BLOCK_HD=BLOCK_HD,
        num_warps=NUM_WARPS_MEDIUM,
    )

    # Reshape to BSND: [1, 1, num_heads, head_dim]
    # BSND matches flash_attn's native layout and the KV cache format,
    # avoiding transpose before index_copy_ and flash_attn_with_kvcache.
    return (
        q_out.view(1, 1, num_q_heads, head_dim),
        k_out.view(1, 1, num_kv_heads, head_dim),
    )