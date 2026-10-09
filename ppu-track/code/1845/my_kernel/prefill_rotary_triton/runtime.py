from __future__ import annotations

import os
import threading

import torch

try:
    import triton
except Exception:
    triton = None

from .kernel import qk_rotary_prefill_kernel

_ROTARY_BUFFER_CACHE = {}
_ROTARY_BUFFER_LOCK = threading.Lock()


def _normalize_cos_sin(cos: torch.Tensor, sin: torch.Tensor):
    if cos.ndim == 3 and cos.shape[0] == 1:
        cos = cos[0]
    if sin.ndim == 3 and sin.shape[0] == 1:
        sin = sin[0]
    if cos.ndim == 3 and cos.shape[1] == 1:
        cos = cos[:, 0, :]
    if sin.ndim == 3 and sin.shape[1] == 1:
        sin = sin[:, 0, :]
    return cos, sin


def can_use_prefill_rotary_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    if triton is None:
        return False
    if os.getenv("AICAS_PREFILL_ROTARY_TRITON", "1") != "1":
        return False
    if not (isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor)):
        return False
    if not (q.is_cuda and k.is_cuda):
        return False
    if q.dtype is not torch.bfloat16 or k.dtype != q.dtype:
        return False
    if q.ndim == 3 and k.ndim == 3:
        if q.shape != k.shape:
            return False
        seq_len = int(q.shape[0])
        head_dim = int(q.shape[2])
    elif q.ndim == 4 and k.ndim == 4:
        if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
            return False
        if q.shape[0] != 1:
            return False
        seq_len = int(q.shape[2])
        head_dim = int(q.shape[3])
    else:
        return False
    if head_dim % 2 != 0:
        return False

    cos, sin = _normalize_cos_sin(cos, sin)
    if cos.ndim != 2 or sin.ndim != 2:
        return False
    if cos.shape != sin.shape:
        return False
    if cos.shape[0] != seq_len or cos.shape[1] != head_dim:
        return False
    if not (cos.is_cuda and sin.is_cuda):
        return False
    if cos.dtype != q.dtype or sin.dtype != q.dtype:
        return False
    return True


def _get_rotary_output_buffers(q: torch.Tensor, k: torch.Tensor):
    q_key = (tuple(q.shape), tuple(q.stride()))
    k_key = (tuple(k.shape), tuple(k.stride()))
    device_idx = q.device.index if q.device.index is not None else torch.cuda.current_device()
    key = (device_idx, str(q.dtype), q_key, k_key)
    with _ROTARY_BUFFER_LOCK:
        cached = _ROTARY_BUFFER_CACHE.get(key)
        if cached is not None:
            return cached
        q_out = torch.empty_strided(size=q.shape, stride=q.stride(), device=q.device, dtype=q.dtype)
        k_out = torch.empty_strided(size=k.shape, stride=k.stride(), device=k.device, dtype=k.dtype)
        _ROTARY_BUFFER_CACHE[key] = (q_out, k_out)
        return q_out, k_out


def _as_seq_head_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x
    return x.squeeze(0).transpose(0, 1)


def rotary_qk_prefill_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not can_use_prefill_rotary_triton(q, k, cos, sin):
        return None

    cos, sin = _normalize_cos_sin(cos, sin)
    q_in = _as_seq_head_dim(q)
    k_in = _as_seq_head_dim(k)
    q_out, k_out = _get_rotary_output_buffers(q_in, k_in)

    total = int(q_in.numel() + k_in.numel())
    q_heads = int(q_in.shape[1])
    k_heads = int(k_in.shape[1])
    d = int(q_in.shape[2])
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
    qk_rotary_prefill_kernel[grid](
        q_in,
        k_in,
        cos,
        sin,
        q_out,
        k_out,
        int(q_in.stride(0)),
        int(q_in.stride(1)),
        int(q_in.stride(2)),
        int(k_in.stride(0)),
        int(k_in.stride(1)),
        int(k_in.stride(2)),
        int(q_out.stride(0)),
        int(q_out.stride(1)),
        int(q_out.stride(2)),
        int(k_out.stride(0)),
        int(k_out.stride(1)),
        int(k_out.stride(2)),
        int(cos.stride(0)),
        int(cos.stride(1)),
        int(sin.stride(0)),
        int(sin.stride(1)),
        QH=q_heads,
        KH=k_heads,
        D=d,
        TOTAL=total,
    )
    if q.ndim == 4:
        return q_out.transpose(0, 1).unsqueeze(0), k_out.transpose(0, 1).unsqueeze(0)
    return q_out, k_out


def apply_prefill_rotary_triton(_model=None) -> int:
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_mod
    except Exception:
        return 0
    if getattr(qwen3_vl_mod, "_aicas_prefill_rotary_triton_patched", False):
        return 0

    original_apply_rotary = qwen3_vl_mod.apply_rotary_pos_emb

    def _patched_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        if position_ids is None and unsqueeze_dim == 1:
            out = rotary_qk_prefill_triton(q, k, cos, sin)
            if out is not None:
                return out
        return original_apply_rotary(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)

    qwen3_vl_mod.apply_rotary_pos_emb = _patched_apply_rotary_pos_emb
    qwen3_vl_mod._aicas_prefill_rotary_triton_original = original_apply_rotary
    qwen3_vl_mod._aicas_prefill_rotary_triton_patched = True
    return 1
