from __future__ import annotations

"""Qwen3-VL decode 阶段的原生/库级融合 patch。"""

from types import MethodType
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from .conf import conf_bool, conf_int, conf_str
from .runtime_int8_quant import runtime_int8_linear

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_BACKEND_STATS = {
    "qkv_backend_fused_linear": 0,
    "qkv_backend_separate_linear": 0,
    "decode_attn_backend_flash_attn": 0,
    "decode_attn_backend_flash_attn_multi_query": 0,
    "decode_attn_backend_flashinfer": 0,
    "decode_attn_backend_flashinfer_fallback": 0,
    "qk_norm_backend_f_rms_norm": 0,
    "qk_norm_backend_manual": 0,
    "qk_norm_backend_triton": 0,
    "qk_rope_backend_separate": 0,
    "qk_rope_backend_triton_fused": 0,
    "qk_rope_backend_triton_fallback": 0,
    "rope_backend_bshd": 0,
    "rope_backend_bshd_triton": 0,
    "rope_backend_bshd_triton_fallback": 0,
    "rope_backend_fa_rotary_experimental": 0,
    "silu_mul_backend_eager": 0,
    "silu_mul_backend_inplace": 0,
    "silu_mul_backend_triton": 0,
    "o_proj_backend_linear": 0,
    "o_proj_backend_mm_out": 0,
    "o_proj_backend_residual_mm_out": 0,
    "down_proj_backend_linear": 0,
    "down_proj_backend_mm_out": 0,
    "down_proj_backend_int8": 0,
    "gate_up_backend_int8": 0,
    "text_rms_norm_fast": 0,
    "text_rms_norm_fallback": 0,
    "layer_tail_fused_decode_hits": 0,
    "layer_tail_fused_multi_query_hits": 0,
}


def _stat_inc(name: str) -> None:
    _BACKEND_STATS[name] = int(_BACKEND_STATS.get(name, 0)) + 1


def reset_decode_backend_stats() -> None:
    for key in _BACKEND_STATS:
        _BACKEND_STATS[key] = 0


def get_decode_backend_stats() -> dict:
    return dict(_BACKEND_STATS)


def _load_flash_attn_with_kvcache():
    try:
        from flash_attn import flash_attn_with_kvcache

        return flash_attn_with_kvcache
    except Exception:
        try:
            from vllm.vllm_flash_attn import flash_attn_with_kvcache

            return flash_attn_with_kvcache
        except Exception:
            return None


def _load_flashinfer_decode():
    try:
        import flashinfer  # type: ignore

        if hasattr(flashinfer, "single_decode_with_kv_cache"):
            return flashinfer.single_decode_with_kv_cache
        decode_ns = getattr(flashinfer, "decode", None)
        if decode_ns is not None and hasattr(decode_ns, "single_decode_with_kv_cache"):
            return decode_ns.single_decode_with_kv_cache
    except Exception:
        return None
    return None


def _num_splits() -> int:
    """传给 flash_attn_with_kvcache；0 表示 FA 自动选 splits（Flash-Decoding 启发式）。"""
    raw = conf_str("FLASH_KVCACHE_NUM_SPLITS", "0", lower=True)
    if raw in ("auto", "0", ""):
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


def _is_static_cache(cache) -> bool:
    return bool(getattr(cache, "is_aicas_static", False) or getattr(cache, "is_static_kv_cache", False))


def _env_on(name: str, default: str = "1") -> bool:
    return conf_bool(name, default)


def _env_str(name: str, default: str) -> str:
    return conf_str(name, default, lower=True)


if _TRITON_AVAILABLE:

    @triton.jit
    def _silu_mul_kernel(gate, up, out, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        gate_val = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
        up_val = tl.load(up + offsets, mask=mask, other=0.0)
        silu = gate_val / (1.0 + tl.exp(-gate_val))
        tl.store(out + offsets, silu * up_val, mask=mask)

    @triton.jit
    def _rms_norm_kernel(x, weight, out, row_stride: tl.constexpr, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        vals = tl.load(x + row * row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        sq = tl.sum(vals * vals, axis=0) / n_cols
        scale = tl.rsqrt(sq + eps)
        w = tl.load(weight + offsets, mask=mask, other=0.0)
        tl.store(out + row * row_stride + offsets, vals * scale * w, mask=mask)

    @triton.jit
    def _rms_rope_kernel(
        x,
        weight,
        cos,
        sin,
        out,
        row_stride: tl.constexpr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        DTYPE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_cols
        vals = tl.load(x + row * row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        sq = tl.sum(vals * vals, axis=0) / n_cols
        scale = tl.rsqrt(sq + eps)

        half = n_cols // 2
        pair_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        pair_vals = tl.load(x + row * row_stride + pair_offsets, mask=mask, other=0.0).to(tl.float32)

        self_w = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        pair_w = tl.load(weight + pair_offsets, mask=mask, other=0.0).to(tl.float32)
        self_norm = vals * scale * self_w
        pair_norm = pair_vals * scale * pair_w
        rot = tl.where(offsets < half, -pair_norm, pair_norm)

        cos_v = tl.load(cos + offsets, mask=mask, other=0.0).to(tl.float32)
        sin_v = tl.load(sin + offsets, mask=mask, other=0.0).to(tl.float32)
        out_v = (self_norm * cos_v) + (rot * sin_v)
        tl.store(out + row * row_stride + offsets, out_v.to(DTYPE), mask=mask)

    @triton.jit
    def _rope_bshd_kernel(
        x,
        cos,
        sin,
        out,
        total,
        seq_len,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        cos_batch: tl.constexpr,
        DTYPE: tl.constexpr,
        COMPUTE_IN_FP32: tl.constexpr,
        MUL_IN_FP32: tl.constexpr,
        ADD_IN_FP32: tl.constexpr,
        PER_MUL_CAST: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        d = offs % head_dim
        tmp = offs // head_dim
        h = tmp % heads
        tmp = tmp // heads
        s = tmp % seq_len
        b = tmp // seq_len
        half = head_dim // 2
        pair_d = tl.where(d < half, d + half, d - half)

        x_base = b * x_stride_b + s * x_stride_s + h * x_stride_h
        self_v = tl.load(x + x_base + d * x_stride_d, mask=mask, other=0.0)
        pair_v = tl.load(x + x_base + pair_d * x_stride_d, mask=mask, other=0.0)
        rot_v = tl.where(d < half, -pair_v, pair_v)
        cos_b = tl.where(cos_batch == 1, 0, b)
        table_base = (cos_b * seq_len + s) * head_dim + d
        cos_v = tl.load(cos + table_base, mask=mask, other=0.0)
        sin_v = tl.load(sin + table_base, mask=mask, other=0.0)

        if COMPUTE_IN_FP32:
            self_v = self_v.to(tl.float32)
            rot_v = rot_v.to(tl.float32)
            cos_v = cos_v.to(tl.float32)
            sin_v = sin_v.to(tl.float32)
        else:
            self_v = self_v.to(DTYPE)
            rot_v = rot_v.to(DTYPE)
            cos_v = cos_v.to(DTYPE)
            sin_v = sin_v.to(DTYPE)

        if PER_MUL_CAST and not COMPUTE_IN_FP32:
            out_v = (self_v * cos_v).to(DTYPE) + (rot_v * sin_v).to(DTYPE)
        else:
            qv = self_v * cos_v
            rv = rot_v * sin_v
            if MUL_IN_FP32:
                qv = qv.to(tl.float32)
                rv = rv.to(tl.float32)
            elif not COMPUTE_IN_FP32:
                qv = qv.to(DTYPE)
                rv = rv.to(DTYPE)
            out_v = qv + rv
            if ADD_IN_FP32:
                out_v = out_v.to(tl.float32)
            if COMPUTE_IN_FP32 or ADD_IN_FP32 or MUL_IN_FP32:
                out_v = out_v.to(DTYPE)
        tl.store(out + offs, out_v, mask=mask)

    @triton.jit
    def _rope_mul_add_bshd_kernel(
        x,
        rot,
        cos,
        sin,
        out,
        total,
        seq_len,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_s: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_d: tl.constexpr,
        cos_batch: tl.constexpr,
        DTYPE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """(x * cos) + (rot * sin) with torch-exact rotate_half already in rot."""
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        d = offs % head_dim
        tmp = offs // head_dim
        h = tmp % heads
        tmp = tmp // heads
        s = tmp % seq_len
        b = tmp // seq_len
        x_base = b * x_stride_b + s * x_stride_s + h * x_stride_h
        self_v = tl.load(x + x_base + d * x_stride_d, mask=mask, other=0.0).to(DTYPE)
        rot_v = tl.load(rot + x_base + d * x_stride_d, mask=mask, other=0.0).to(DTYPE)
        cos_b = tl.where(cos_batch == 1, 0, b)
        table_base = (cos_b * seq_len + s) * head_dim + d
        cos_v = tl.load(cos + table_base, mask=mask, other=0.0).to(DTYPE)
        sin_v = tl.load(sin + table_base, mask=mask, other=0.0).to(DTYPE)
        out_v = (self_v * cos_v).to(DTYPE) + (rot_v * sin_v).to(DTYPE)
        tl.store(out + offs, out_v, mask=mask)


def _triton_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> Optional[torch.Tensor]:
    if not _TRITON_AVAILABLE or not gate.is_cuda or not up.is_cuda or not gate.is_contiguous() or not up.is_contiguous():
        return None
    n_elements = gate.numel()
    if n_elements <= 0:
        return None
    out = torch.empty_like(gate)
    block = min(1024, triton.next_power_of_2(n_elements))
    grid = (triton.cdiv(n_elements, block),)
    _silu_mul_kernel[grid](gate, up, out, n_elements, BLOCK=block)
    return out


def _triton_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> Optional[torch.Tensor]:
    if not _TRITON_AVAILABLE or not x.is_cuda or not x.is_contiguous() or not weight.is_cuda:
        return None
    n_cols = int(x.shape[-1])
    if n_cols <= 0 or n_cols > 1024:
        return None
    out = torch.empty_like(x)
    rows = x.numel() // n_cols
    block = triton.next_power_of_2(n_cols)
    _rms_norm_kernel[(rows,)](x, weight, out, n_cols, n_cols, float(eps), BLOCK=block)
    return out


def _flashinfer_decode_call(fn, q, k_ctx, v_ctx, softmax_scale: float):
    q_flat = q[0, 0].contiguous()
    k_flat = k_ctx[0].contiguous()
    v_flat = v_ctx[0].contiguous()
    try:
        out = fn(q_flat, k_flat, v_flat, kv_layout="NHD", sm_scale=softmax_scale)
    except TypeError:
        out = fn(q_flat, k_flat, v_flat, kv_layout="NHD", softmax_scale=softmax_scale)
    except Exception:
        out = fn(q_flat, k_flat, v_flat)
    return out.unsqueeze(0).unsqueeze(0)


def _rms_norm_decode(x: torch.Tensor, norm_module) -> torch.Tensor:
    if norm_module is None or not hasattr(norm_module, "weight"):
        return x
    eps = float(getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6)))
    backend = _env_str("DECODE_QK_NORM_BACKEND", "f_rms_norm")
    if backend == "triton":
        out = _triton_rms_norm(x, norm_module.weight, eps)
        if out is not None:
            _stat_inc("qk_norm_backend_triton")
            return out
        backend = "f_rms_norm"
    if backend == "manual":
        _stat_inc("qk_norm_backend_manual")
        x_fp32 = x.float()
        inv_rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x_fp32 * inv_rms).to(dtype=x.dtype) * norm_module.weight
    _stat_inc("qk_norm_backend_f_rms_norm")
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), norm_module.weight, eps)
    return norm_module(x)


def _single_token_rope_row(position_embeddings, head_dim: int, dtype: torch.dtype, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if position_embeddings is None:
        return None
    cos, sin = position_embeddings
    if not torch.is_tensor(cos) or not torch.is_tensor(sin):
        return None
    if cos.dim() == 3:
        cos = cos[0, 0]
        sin = sin[0, 0]
    elif cos.dim() == 2:
        cos = cos[0]
        sin = sin[0]
    elif cos.dim() != 1:
        return None
    if cos.numel() < head_dim or sin.numel() < head_dim:
        return None
    return (
        cos[:head_dim].to(dtype=dtype, device=device).contiguous(),
        sin[:head_dim].to(dtype=dtype, device=device).contiguous(),
    )


def _triton_rms_rope(x: torch.Tensor, norm_module, cos_row: torch.Tensor, sin_row: torch.Tensor) -> Optional[torch.Tensor]:
    if norm_module is None or not hasattr(norm_module, "weight"):
        return None
    if not _TRITON_AVAILABLE or not x.is_cuda or not x.is_contiguous():
        return None
    if cos_row.dim() != 1 or sin_row.dim() != 1:
        return None
    n_cols = int(x.shape[-1])
    if n_cols <= 0 or n_cols > 1024 or (n_cols % 2) != 0:
        return None
    if not norm_module.weight.is_cuda or norm_module.weight.numel() < n_cols:
        return None
    out = torch.empty_like(x)
    rows = x.numel() // n_cols
    block = triton.next_power_of_2(n_cols)
    dtype_tl = _rope_triton_dtype_tl(x)
    eps = float(getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6)))
    _rms_rope_kernel[(rows,)](
        x,
        norm_module.weight,
        cos_row,
        sin_row,
        out,
        n_cols,
        n_cols,
        eps,
        DTYPE=dtype_tl,
        BLOCK=block,
    )
    return out


def _text_rms_norm_fast_forward(norm_module, hidden_states: torch.Tensor, *, _original_forward) -> torch.Tensor:
    if (
        not _env_on("ENABLE_TEXT_RMSNORM_FAST", "1")
        or torch.is_grad_enabled()
        or not torch.is_tensor(hidden_states)
        or not hidden_states.is_cuda
        or not hasattr(norm_module, "weight")
    ):
        return _original_forward(hidden_states)

    try:
        eps = float(getattr(norm_module, "variance_epsilon", getattr(norm_module, "eps", 1e-6)))
        if hasattr(F, "rms_norm"):
            _stat_inc("text_rms_norm_fast")
            return F.rms_norm(hidden_states, (hidden_states.shape[-1],), norm_module.weight, eps)
    except Exception:
        _stat_inc("text_rms_norm_fallback")
    return _original_forward(hidden_states)


def _rotate_half_bshd(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _prepare_rope_cos_sin_bshd(cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broadcast cos/sin to [B, S, 1, D] like eager _apply_rope_bshd."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    return cos.to(dtype=q.dtype, device=q.device), sin.to(dtype=q.dtype, device=q.device)


def _apply_rope_bshd_eager(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _prepare_rope_cos_sin_bshd(cos, sin, q)
    q_out = (q * cos) + (_rotate_half_bshd(q) * sin)
    k_out = (k * cos) + (_rotate_half_bshd(k) * sin)
    return q_out.contiguous(), k_out.contiguous()


def _apply_rope_bshd(q: torch.Tensor, k: torch.Tensor, position_embeddings) -> Tuple[torch.Tensor, torch.Tensor]:
    if position_embeddings is None:
        return q, k
    cos, sin = position_embeddings
    if not torch.is_tensor(cos) or not torch.is_tensor(sin):
        return q, k
    if _env_on("ENABLE_TEXT_ROPE_TRITON", "1"):
        out = _apply_rope_bshd_triton(q, k, cos, sin)
        if out is not None:
            _stat_inc("rope_backend_bshd_triton")
            return out
        _stat_inc("rope_backend_bshd_triton_fallback")
    _stat_inc("rope_backend_bshd")
    return _apply_rope_bshd_eager(q, k, cos, sin)


def _decode_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_embeddings,
    q_norm_module,
    k_norm_module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backend = _env_str("DECODE_QK_ROPE_BACKEND", "separate")
    if backend == "triton_fused" and q.dim() == 4 and k.dim() == 4 and q.shape[:2] == (1, 1) and k.shape[:2] == (1, 1):
        rope_row = _single_token_rope_row(position_embeddings, int(q.shape[-1]), q.dtype, q.device)
        if rope_row is not None:
            cos_row, sin_row = rope_row
            q_2d = q.reshape(-1, int(q.shape[-1])).contiguous()
            k_2d = k.reshape(-1, int(k.shape[-1])).contiguous()
            q_out = _triton_rms_rope(q_2d, q_norm_module, cos_row, sin_row)
            k_out = _triton_rms_rope(k_2d, k_norm_module, cos_row, sin_row)
            if q_out is not None and k_out is not None:
                _stat_inc("qk_rope_backend_triton_fused")
                return q_out.view_as(q), k_out.view_as(k)
        _stat_inc("qk_rope_backend_triton_fallback")

    _stat_inc("qk_rope_backend_separate")
    q = _rms_norm_decode(q, q_norm_module)
    k = _rms_norm_decode(k, k_norm_module)
    return _apply_rope_bshd(q, k, position_embeddings)


def _rope_triton_dtype_tl(x: torch.Tensor):
    if x.dtype == torch.float16:
        return tl.float16
    if x.dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


def _rope_cos_sin_layout_ok(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
    if (
        not x.is_cuda
        or not cos.is_cuda
        or not sin.is_cuda
        or x.dim() != 4
        or cos.dim() != 3
        or sin.dim() != 3
    ):
        return False
    batch, seq_len, _, head_dim = [int(v) for v in x.shape]
    if head_dim % 2 != 0 or cos.shape[-2:] != (seq_len, head_dim) or sin.shape != cos.shape:
        return False
    return int(cos.shape[0]) in (1, batch)


def _rope_one_bshd_exact(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Optional[torch.Tensor]:
    """Bit-exact with eager _apply_rope_bshd (torch cat rotate_half + broadcast mul)."""
    if not _rope_cos_sin_layout_ok(x, cos, sin):
        return None
    cos_p, sin_p = _prepare_rope_cos_sin_bshd(cos, sin, x)
    return ((x * cos_p) + (_rotate_half_bshd(x) * sin_p)).contiguous()


_FUSED_ROPE_VARIANTS = frozenset(
    {
        "fused",
        "fused_fp32",
        "fused_fp16_pure",
        "fused_fp16_acc",
        "fused_fp16_mul_fp16",
        "fused_fp16_mul_fp32",
        "fused_fp16_add_fp16",
        "fused_fp16_add_fp32",
        "q_fused_k_exact",
        "q_exact_k_fused",
    }
)


def _rope_fused_kernel_flags(variant: str) -> Tuple[bool, bool, bool, bool]:
    """Return (COMPUTE_IN_FP32, MUL_IN_FP32, ADD_IN_FP32, PER_MUL_CAST)."""
    if variant in ("fused_fp32",):
        return True, True, True, False
    if variant in ("fused_fp16_pure",):
        return False, False, False, False
    if variant in ("fused_fp16_mul_fp16",):
        return False, False, False, False
    if variant in ("fused_fp16_mul_fp32",):
        return False, True, False, False
    if variant in ("fused_fp16_add_fp16",):
        return False, False, False, False
    if variant in ("fused_fp16_add_fp32",):
        return False, False, True, False
    if variant in ("fused", "fused_fp16_acc"):
        return False, False, False, True
    return False, False, False, True


def _rope_one_bshd_triton_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    variant: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Fused rotate+mul Triton kernel."""
    if not _TRITON_AVAILABLE or not _rope_cos_sin_layout_ok(x, cos, sin):
        return None
    variant = variant or _env_str("TEXT_ROPE_TRITON_BACKEND", "exact")
    if variant in ("fused", "fused_fp16_acc"):
        variant = "fused_fp16_acc"
    compute_fp32, mul_fp32, add_fp32, per_mul_cast = _rope_fused_kernel_flags(variant)
    batch, seq_len, heads, head_dim = [int(v) for v in x.shape]
    cos = cos.to(dtype=x.dtype, device=x.device).contiguous()
    sin = sin.to(dtype=x.dtype, device=x.device).contiguous()
    out = torch.empty((batch, seq_len, heads, head_dim), dtype=x.dtype, device=x.device)
    total = batch * seq_len * heads * head_dim
    block = 256
    dtype_tl = _rope_triton_dtype_tl(x)
    _rope_bshd_kernel[(triton.cdiv(total, block),)](
        x,
        cos,
        sin,
        out,
        total,
        seq_len,
        heads,
        head_dim,
        int(x.stride(0)),
        int(x.stride(1)),
        int(x.stride(2)),
        int(x.stride(3)),
        int(cos.shape[0]),
        DTYPE=dtype_tl,
        COMPUTE_IN_FP32=compute_fp32,
        MUL_IN_FP32=mul_fp32,
        ADD_IN_FP32=add_fp32,
        PER_MUL_CAST=per_mul_cast,
        BLOCK=block,
        num_warps=4,
    )
    return out


def _rope_one_bshd_fused_variant(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Optional[torch.Tensor]:
    variant = _env_str("TEXT_ROPE_TRITON_BACKEND", "exact")
    inner = variant
    if variant in ("q_fused_k_exact", "q_exact_k_fused"):
        inner = _env_str("TEXT_ROPE_FUSED_MIXED_KERNEL", "fused_fp32")
    if inner not in _FUSED_ROPE_VARIANTS:
        inner = "fused_fp16_acc"
    return _rope_one_bshd_triton_fused(x, cos, sin, variant=inner)


def _rope_one_bshd_triton_hybrid(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Optional[torch.Tensor]:
    """Torch-exact rotate_half + Triton (x*cos + rot*sin)."""
    if not _TRITON_AVAILABLE or not _rope_cos_sin_layout_ok(x, cos, sin):
        return None
    batch, seq_len, heads, head_dim = [int(v) for v in x.shape]
    cos = cos.to(dtype=x.dtype, device=x.device).contiguous()
    sin = sin.to(dtype=x.dtype, device=x.device).contiguous()
    rot = _rotate_half_bshd(x).contiguous()
    out = torch.empty((batch, seq_len, heads, head_dim), dtype=x.dtype, device=x.device)
    total = batch * seq_len * heads * head_dim
    block = 256
    dtype_tl = _rope_triton_dtype_tl(x)
    _rope_mul_add_bshd_kernel[(triton.cdiv(total, block),)](
        x,
        rot,
        cos,
        sin,
        out,
        total,
        seq_len,
        heads,
        head_dim,
        int(x.stride(0)),
        int(x.stride(1)),
        int(x.stride(2)),
        int(x.stride(3)),
        int(cos.shape[0]),
        DTYPE=dtype_tl,
        BLOCK=block,
        num_warps=4,
    )
    return out


def _rope_one_bshd_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Optional[torch.Tensor]:
    backend = _env_str("TEXT_ROPE_TRITON_BACKEND", "exact")
    if backend in _FUSED_ROPE_VARIANTS:
        return _rope_one_bshd_fused_variant(x, cos, sin)
    if backend == "hybrid":
        return _rope_one_bshd_triton_hybrid(x, cos, sin)
    return _rope_one_bshd_exact(x, cos, sin)


def _apply_rope_bshd_triton(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if torch.is_grad_enabled():
        return None
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    elif cos.dim() != 3:
        return None
    backend = _env_str("TEXT_ROPE_TRITON_BACKEND", "exact")
    try:
        if backend == "exact":
            return _apply_rope_bshd_eager(q, k, cos, sin)
        if backend == "q_fused_k_exact":
            q_out = _rope_one_bshd_fused_variant(q, cos, sin)
            k_out = _rope_one_bshd_exact(k, cos, sin)
            if q_out is None or k_out is None:
                return None
            return q_out, k_out
        if backend == "q_exact_k_fused":
            q_out = _rope_one_bshd_exact(q, cos, sin)
            k_out = _rope_one_bshd_fused_variant(k, cos, sin)
            if q_out is None or k_out is None:
                return None
            return q_out, k_out
        q_out = _rope_one_bshd_triton(q, cos, sin)
        k_out = _rope_one_bshd_triton(k, cos, sin)
        if q_out is None or k_out is None:
            return None
        return q_out, k_out
    except Exception:
        return None


def _fa_rotary_tables(position_embeddings, head_dim: int, dtype: torch.dtype, device: torch.device):
    """Return flash-attn rotary tables for simple text RoPE; MRoPE falls back."""
    if not _env_on("ENABLE_FA_KVCACHE_ROTARY", "0") or position_embeddings is None:
        return None
    cos, sin = position_embeddings
    if not torch.is_tensor(cos) or not torch.is_tensor(sin):
        return None
    if cos.dim() == 3 and int(cos.shape[0]) == 1:
        cos = cos[0]
        sin = sin[0]
    if cos.dim() != 2 or sin.dim() != 2:
        return None
    if cos.shape != sin.shape:
        return None
    rotary_half = head_dim // 2
    if cos.shape[-1] < rotary_half or sin.shape[-1] < rotary_half:
        return None
    cos_fa = cos[:, :rotary_half].to(dtype=dtype, device=device).contiguous()
    sin_fa = sin[:, :rotary_half].to(dtype=dtype, device=device).contiguous()
    return cos_fa, sin_fa


def _get_or_build_fused_qkv(attn_module):
    key = (
        attn_module.q_proj.weight.data_ptr(),
        attn_module.k_proj.weight.data_ptr(),
        attn_module.v_proj.weight.data_ptr(),
        attn_module.q_proj.bias.data_ptr() if attn_module.q_proj.bias is not None else 0,
        attn_module.k_proj.bias.data_ptr() if attn_module.k_proj.bias is not None else 0,
        attn_module.v_proj.bias.data_ptr() if attn_module.v_proj.bias is not None else 0,
    )
    if getattr(attn_module, "_decode_qkv_fused_key", None) == key:
        return attn_module._decode_qkv_fused_weight, attn_module._decode_qkv_fused_bias
    weight = torch.cat([attn_module.q_proj.weight, attn_module.k_proj.weight, attn_module.v_proj.weight], dim=0).contiguous()
    if attn_module.q_proj.bias is not None and attn_module.k_proj.bias is not None and attn_module.v_proj.bias is not None:
        bias = torch.cat([attn_module.q_proj.bias, attn_module.k_proj.bias, attn_module.v_proj.bias], dim=0).contiguous()
    else:
        bias = None
    attn_module._decode_qkv_fused_key = key
    attn_module._decode_qkv_fused_weight = weight
    attn_module._decode_qkv_fused_bias = bias
    return weight, bias


def _get_or_build_fused_gate_up(mlp):
    key = (
        mlp.gate_proj.weight.data_ptr(),
        mlp.up_proj.weight.data_ptr(),
        mlp.gate_proj.bias.data_ptr() if mlp.gate_proj.bias is not None else 0,
        mlp.up_proj.bias.data_ptr() if mlp.up_proj.bias is not None else 0,
    )
    if getattr(mlp, "_decode_gate_up_key", None) == key:
        return mlp._decode_gate_up_weight, mlp._decode_gate_up_bias
    weight = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).contiguous()
    if mlp.gate_proj.bias is not None and mlp.up_proj.bias is not None:
        bias = torch.cat([mlp.gate_proj.bias, mlp.up_proj.bias], dim=0).contiguous()
    else:
        bias = None
    mlp._decode_gate_up_key = key
    mlp._decode_gate_up_weight = weight
    mlp._decode_gate_up_bias = bias
    return weight, bias


def _get_or_build_down_proj_t(mlp):
    key = mlp.down_proj.weight.data_ptr()
    if getattr(mlp, "_decode_down_t_key", None) == key:
        return mlp._decode_down_weight_t
    w_t = mlp.down_proj.weight.transpose(0, 1).contiguous()
    mlp._decode_down_t_key = key
    mlp._decode_down_weight_t = w_t
    return w_t


def _get_or_build_down_ws(mlp, hidden_size: int, dtype: torch.dtype, device: torch.device):
    key = (hidden_size, str(dtype), str(device))
    ws = getattr(mlp, "_decode_down_ws", None)
    if getattr(mlp, "_decode_down_ws_key", None) == key and ws is not None:
        return ws
    ws = torch.empty((1, hidden_size), dtype=dtype, device=device)
    mlp._decode_down_ws_key = key
    mlp._decode_down_ws = ws
    return ws


def _get_or_build_o_proj_t(attn):
    key = attn.o_proj.weight.data_ptr()
    if getattr(attn, "_decode_o_proj_t_key", None) == key:
        return attn._decode_o_proj_weight_t
    w_t = attn.o_proj.weight.transpose(0, 1).contiguous()
    attn._decode_o_proj_t_key = key
    attn._decode_o_proj_weight_t = w_t
    return w_t


def _get_or_build_o_proj_ws(attn, hidden_size: int, dtype: torch.dtype, device: torch.device):
    key = (hidden_size, str(dtype), str(device))
    ws = getattr(attn, "_decode_o_proj_ws", None)
    if getattr(attn, "_decode_o_proj_ws_key", None) == key and ws is not None:
        return ws
    ws = torch.empty((1, hidden_size), dtype=dtype, device=device)
    attn._decode_o_proj_ws_key = key
    attn._decode_o_proj_ws = ws
    return ws


def _decode_o_proj(attn, out: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
    backend = _env_str("DECODE_O_PROJ_BACKEND", "linear")
    if backend == "residual_mm_out" and residual is not None:
        if out.shape[0] == 1 and out.shape[1] == 1 and residual.shape == out.shape:
            _stat_inc("o_proj_backend_residual_mm_out")
            out_2d = out.reshape(1, -1)
            ws = _get_or_build_o_proj_ws(attn, int(attn.o_proj.out_features), out.dtype, out.device)
            torch.mm(out_2d, _get_or_build_o_proj_t(attn), out=ws)
            if attn.o_proj.bias is not None:
                ws.add_(attn.o_proj.bias.view(1, -1))
            ws.add_(residual.reshape(1, -1))
            return ws.view_as(residual)
        _stat_inc("o_proj_backend_linear")
        return F.linear(out, attn.o_proj.weight, attn.o_proj.bias) + residual

    if backend == "mm_out" and out.shape[0] == 1 and out.shape[1] == 1:
        _stat_inc("o_proj_backend_mm_out")
        out_2d = out.reshape(1, -1)
        ws = _get_or_build_o_proj_ws(attn, int(attn.o_proj.out_features), out.dtype, out.device)
        torch.mm(out_2d, _get_or_build_o_proj_t(attn), out=ws)
        if attn.o_proj.bias is not None:
            ws.add_(attn.o_proj.bias.view(1, -1))
        return ws.view(out.shape[0], out.shape[1], -1)

    _stat_inc("o_proj_backend_linear")
    proj = F.linear(out, attn.o_proj.weight, attn.o_proj.bias)
    if residual is not None:
        proj = proj + residual
    return proj


def _decode_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    backend = _env_str("DECODE_SILU_MUL_BACKEND", "eager")
    if backend == "triton":
        out = _triton_silu_mul(gate, up)
        if out is not None:
            _stat_inc("silu_mul_backend_triton")
            return out
        backend = "eager"
    if backend == "inplace":
        _stat_inc("silu_mul_backend_inplace")
        out = F.silu(gate)
        out.mul_(up)
        return out
    _stat_inc("silu_mul_backend_eager")
    return F.silu(gate) * up


def _decode_down_proj(mlp, gated: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
    max_q_len = int(getattr(mlp, "_runtime_int8_max_query_len", 0) or 0)
    if max_q_len > 0 and gated.dim() == 3 and gated.shape[1] <= max_q_len:
        out = runtime_int8_linear(gated, getattr(mlp, "_runtime_int8_down_state", None))
        if out is not None:
            _stat_inc("down_proj_backend_int8")
            if residual is not None:
                out = out + residual
            return out

    backend = _env_str("DECODE_DOWN_PROJ_BACKEND", "linear")
    if backend == "mm_out" and gated.shape[0] == 1 and gated.shape[1] == 1:
        _stat_inc("down_proj_backend_mm_out")
        gated_2d = gated.reshape(1, -1)
        w_t = _get_or_build_down_proj_t(mlp)
        ws = _get_or_build_down_ws(mlp, int(mlp.down_proj.out_features), gated.dtype, gated.device)
        torch.mm(gated_2d, w_t, out=ws)
        if mlp.down_proj.bias is not None:
            ws.add_(mlp.down_proj.bias.view(1, -1))
        if residual is not None:
            ws.add_(residual.reshape(1, -1))
            return ws.view_as(residual)
        return ws.view(gated.shape[0], gated.shape[1], -1)

    _stat_inc("down_proj_backend_linear")
    out = F.linear(gated, mlp.down_proj.weight, mlp.down_proj.bias)
    if residual is not None:
        out = out + residual
    return out


def _decode_only_mlp_forward(mlp, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None):
    max_q_len = int(getattr(mlp, "_runtime_int8_max_query_len", 0) or 0)
    gate_size = int(mlp.gate_proj.out_features)
    up_size = int(mlp.up_proj.out_features)
    gate_up = None
    if max_q_len > 0 and hidden_states.dim() == 3 and hidden_states.shape[1] <= max_q_len:
        gate_up = runtime_int8_linear(hidden_states, getattr(mlp, "_runtime_int8_gate_up_state", None))
        if gate_up is not None:
            _stat_inc("gate_up_backend_int8")
    if gate_up is None:
        gate_up_w, gate_up_b = _get_or_build_fused_gate_up(mlp)
        gate_up = F.linear(hidden_states, gate_up_w, gate_up_b)
    gate, up = torch.split(gate_up, [gate_size, up_size], dim=-1)
    gated = _decode_silu_mul(gate, up)
    return _decode_down_proj(mlp, gated, residual=residual)


def _patch_layer_tail(layer) -> bool:
    if getattr(layer, "_decode_layer_tail_patched", False):
        return False
    if not all(hasattr(layer, name) for name in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")):
        return False
    original_forward = layer.forward

    def decode_layer_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        _original_forward=original_forward,
        **kwargs,
    ):
        q_len = int(hidden_states.shape[1]) if hidden_states.dim() == 3 else 0
        multi_query_ok = q_len > 1 and _env_on("ENABLE_DECODE_LAYER_TAIL_MULTI_QUERY", "1")
        if (
            not _env_on("ENABLE_DECODE_LAYER_TAIL_FUSION", "1")
            or hidden_states.dim() != 3
            or hidden_states.shape[1] < 1
            or (hidden_states.shape[1] != 1 and not multi_query_ok)
            or not _is_static_cache(past_key_values)
        ):
            return _original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        _stat_inc("layer_tail_fused_decode_hits")
        if q_len > 1:
            _stat_inc("layer_tail_fused_multi_query_hits")
        residual = hidden_states
        hidden_states = _rms_norm_decode(hidden_states, self.input_layernorm)
        fuse_attn_residual = _env_str("DECODE_O_PROJ_BACKEND", "linear") == "residual_mm_out"
        attn_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            _residual_for_o_proj=residual if fuse_attn_residual else None,
            **kwargs,
        )
        hidden_after_attn = attn_output if fuse_attn_residual else (residual + attn_output)
        hidden_for_mlp = _rms_norm_decode(hidden_after_attn, self.post_attention_layernorm)
        if _env_on("ENABLE_DECODE_MLP_RESIDUAL_FUSE", "1"):
            return _decode_only_mlp_forward(self.mlp, hidden_for_mlp, residual=hidden_after_attn)
        mlp_out = self.mlp(hidden_for_mlp)
        return hidden_after_attn + mlp_out

    layer.forward = MethodType(decode_layer_forward, layer)
    layer._decode_layer_tail_patched = True
    return True


def _patch_text_rms_norms(language_model) -> bool:
    if not _env_on("ENABLE_TEXT_RMSNORM_FAST", "1"):
        return False
    patched_any = False
    for module in language_model.modules():
        if module.__class__.__name__ != "Qwen3VLTextRMSNorm" or getattr(module, "_text_rmsnorm_fast_patched", False):
            continue
        original_forward = module.forward

        def rms_forward(self, hidden_states, _orig=original_forward):
            return _text_rms_norm_fast_forward(self, hidden_states, _original_forward=_orig)

        module.forward = MethodType(rms_forward, module)
        module._text_rmsnorm_fast_patched = True
        patched_any = True
    return patched_any


def _prewarm_text_rope_triton(language_model) -> None:
    if not _env_on("ENABLE_TEXT_ROPE_TRITON_PREWARM", "1"):
        return
    try:
        first_attn = None
        for layer in getattr(language_model, "layers", []):
            first_attn = getattr(layer, "self_attn", None)
            if first_attn is not None:
                break
        if first_attn is None:
            return
        weight = first_attn.q_proj.weight
        if not weight.is_cuda:
            return
        head_dim = int(first_attn.head_dim)
        q_heads = int(first_attn.q_proj.out_features // head_dim)
        kv_heads = int(first_attn.k_proj.out_features // head_dim)
        cos = torch.empty((1, 1, head_dim), dtype=weight.dtype, device=weight.device)
        sin = torch.empty_like(cos)
        q = torch.empty((1, 1, q_heads, head_dim), dtype=weight.dtype, device=weight.device)
        k = torch.empty((1, 1, kv_heads, head_dim), dtype=weight.dtype, device=weight.device)
        with torch.no_grad():
            _apply_rope_bshd_triton(q, k, cos, sin)
    except Exception:
        _stat_inc("rope_backend_bshd_triton_fallback")


def apply_decode_kernel_fusions(model) -> bool:
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(language_model, "layers", None)
    if not layers:
        return False

    patched_any = False
    if _patch_text_rms_norms(language_model):
        patched_any = True
    _prewarm_text_rope_triton(language_model)

    enable_mlp = _env_on("ENABLE_DECODE_FUSED_MLP", "1")
    for layer in layers:
        if enable_mlp:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and not getattr(mlp, "_decode_fused_mlp_patched", False):
                original_forward = mlp.forward

                def mlp_forward(self, hidden_states: torch.Tensor, _original_forward=original_forward):
                    if hidden_states.dim() != 3 or hidden_states.shape[1] != 1:
                        return _original_forward(hidden_states)
                    return _decode_only_mlp_forward(self, hidden_states)

                mlp.forward = MethodType(mlp_forward, mlp)
                mlp._decode_fused_mlp_patched = True
                patched_any = True

        if _patch_layer_tail(layer):
            patched_any = True

    if patched_any:
        model._decode_kernel_fusions_patched = True
        model._decode_backend_stats = _BACKEND_STATS
        print("[flash_kvcache_attention] decode backend A/B baseline enabled")
    return patched_any


def apply_flash_kvcache_attention(model) -> bool:
    flash_attn_with_kvcache = _load_flash_attn_with_kvcache()
    if flash_attn_with_kvcache is None:
        print("[flash_kvcache_attention] flash_attn_with_kvcache unavailable")
        return False
    flashinfer_decode = _load_flashinfer_decode()

    language_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(language_model, "layers", None)
    if not layers:
        return False

    for layer in layers:
        attn = layer.self_attn
        if getattr(attn, "_flash_kvcache_attention_patched", False):
            continue
        original_forward = attn.forward

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_values=None,
            cache_position: Optional[torch.LongTensor] = None,
            _original_forward=original_forward,
            _flash_attn_with_kvcache=flash_attn_with_kvcache,
            **kwargs,
        ):
            residual_for_o_proj = kwargs.pop("_residual_for_o_proj", None)
            if hidden_states.dim() == 3:
                q_len_in = int(hidden_states.shape[1])
            else:
                q_len_in = 0
            multi_query_ok = (
                q_len_in > 1
                and _env_on("ENABLE_FLASH_KVCACHE_MULTI_QUERY", "1")
                and _env_str("DECODE_ATTN_BACKEND", "flash_attn") == "flash_attn"
            )
            if (
                hidden_states.dim() != 3
                or hidden_states.shape[1] < 1
                or (hidden_states.shape[1] != 1 and not multi_query_ok)
                or past_key_values is None
                or not _is_static_cache(past_key_values)
            ):
                return _original_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            batch, q_len, _ = hidden_states.shape
            head_dim = int(self.head_dim)
            q_heads = int(self.q_proj.out_features // head_dim)
            kv_heads = int(self.k_proj.out_features // head_dim)
            layer_idx = getattr(self, "layer_idx", None)
            if layer_idx is None:
                return _original_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            if _env_on("ENABLE_DECODE_FUSED_QKV", "1"):
                fused_w, fused_b = _get_or_build_fused_qkv(self)
                _stat_inc("qkv_backend_fused_linear")
                qkv = F.linear(hidden_states, fused_w, fused_b)
                q_hidden = int(self.q_proj.out_features)
                kv_hidden = int(self.k_proj.out_features)
                q_part, k_part, v_part = torch.split(qkv, [q_hidden, kv_hidden, kv_hidden], dim=-1)
                q = q_part.view(batch, q_len, q_heads, head_dim)
                k = k_part.view(batch, q_len, kv_heads, head_dim)
                v = v_part.view(batch, q_len, kv_heads, head_dim)
            else:
                _stat_inc("qkv_backend_separate_linear")
                q = self.q_proj(hidden_states).view(batch, q_len, q_heads, head_dim)
                k = self.k_proj(hidden_states).view(batch, q_len, kv_heads, head_dim)
                v = self.v_proj(hidden_states).view(batch, q_len, kv_heads, head_dim)

            q, k = _decode_qk_norm_rope(
                q,
                k,
                position_embeddings,
                getattr(self, "q_norm", None),
                getattr(self, "k_norm", None),
            )
            v = v.contiguous()

            k_cache = past_key_values.k_caches[layer_idx]
            v_cache = past_key_values.v_caches[layer_idx]
            cache_seqlens = past_key_values.cache_seqlens_buf

            decode_backend = _env_str("DECODE_ATTN_BACKEND", "flash_attn")
            fa_rotary = None
            if (
                decode_backend == "flash_attn"
                and getattr(past_key_values, "head_dim_padded", head_dim) == head_dim
                and _env_str("DECODE_QK_ROPE_BACKEND", "separate") == "separate"
            ):
                fa_rotary = _fa_rotary_tables(position_embeddings, head_dim, q.dtype, q.device)
            if fa_rotary is not None:
                _stat_inc("rope_backend_fa_rotary_experimental")
            use_flashinfer = decode_backend == "flashinfer" and flashinfer_decode is not None and q_len == 1
            if use_flashinfer:
                try:
                    # flashinfer 路径手动写入当前 token，再在有效上下文上执行 decode。
                    cache_pos = int(cache_seqlens.reshape(-1)[0].item())
                    k_cache[:, cache_pos : cache_pos + 1, :, :].copy_(k)
                    v_cache[:, cache_pos : cache_pos + 1, :, :].copy_(v)
                    k_ctx = k_cache[:, : cache_pos + 1, :, :]
                    v_ctx = v_cache[:, : cache_pos + 1, :, :]
                    out = _flashinfer_decode_call(flashinfer_decode, q, k_ctx, v_ctx, float(self.scaling))
                    _stat_inc("decode_attn_backend_flashinfer")
                except Exception:
                    out = _flash_attn_with_kvcache(
                        q,
                        k_cache,
                        v_cache,
                        k=k,
                        v=v,
                        cache_seqlens=cache_seqlens,
                        softmax_scale=self.scaling,
                        causal=True,
                        num_splits=_num_splits(),
                    )
                    _stat_inc("decode_attn_backend_flashinfer_fallback")
            else:
                if fa_rotary is None:
                    out = _flash_attn_with_kvcache(
                        q,
                        k_cache,
                        v_cache,
                        k=k,
                        v=v,
                        cache_seqlens=cache_seqlens,
                        softmax_scale=self.scaling,
                        causal=True,
                        num_splits=_num_splits(),
                    )
                else:
                    cos_fa, sin_fa = fa_rotary
                    out = _flash_attn_with_kvcache(
                        q,
                        k_cache,
                        v_cache,
                        k=k,
                        v=v,
                        rotary_cos=cos_fa,
                        rotary_sin=sin_fa,
                        cache_seqlens=cache_seqlens,
                        softmax_scale=self.scaling,
                        causal=True,
                        rotary_interleaved=False,
                        num_splits=_num_splits(),
                    )
                _stat_inc("decode_attn_backend_flash_attn")
                if q_len > 1:
                    _stat_inc("decode_attn_backend_flash_attn_multi_query")
            out = out.reshape(batch, q_len, q_heads * head_dim)
            return _decode_o_proj(self, out, residual=residual_for_o_proj), None

        attn.forward = MethodType(forward, attn)
        attn._flash_kvcache_attention_patched = True

    model._flash_kvcache_attention_patched = True
    model._decode_backend_stats = _BACKEND_STATS
    print("[flash_kvcache_attention] decode attention patched (backend-aware)")
    return True
