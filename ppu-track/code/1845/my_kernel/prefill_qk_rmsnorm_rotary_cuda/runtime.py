from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_EXT = None
_CACHE = {}
_LOCK = threading.Lock()


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    so_path = _THIS_DIR / "prefill_qk_rmsnorm_rotary_ext.so"
    if not so_path.is_file():
        cands = sorted(_THIS_DIR.glob("prefill_qk_rmsnorm_rotary_ext*.so"))
        if not cands:
            return None
        so_path = cands[-1]
    name = "prefill_qk_rmsnorm_rotary_ext"
    if name in sys.modules:
        _EXT = sys.modules[name]
        return _EXT
    spec = importlib.util.spec_from_file_location(name, str(so_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _EXT = mod
    return _EXT


def is_available() -> bool:
    return _load_ext() is not None


def _normalize_cos_sin(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.ndim == 3 and cos.shape[0] == 1:
        cos = cos[0]
    if sin.ndim == 3 and sin.shape[0] == 1:
        sin = sin[0]
    if cos.ndim == 3 and cos.shape[1] == 1:
        cos = cos[:, 0, :]
    if sin.ndim == 3 and sin.shape[1] == 1:
        sin = sin[:, 0, :]
    return cos, sin


def _can_use(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> bool:
    if os.getenv("AICAS_PREFILL_QK_RMS_ROPE_CUDA", "1") != "1":
        return False
    if _load_ext() is None:
        return False
    if not all(isinstance(t, torch.Tensor) for t in (q, k, cos, sin, q_weight, k_weight)):
        return False
    if not all(t.is_cuda for t in (q, k, cos, sin, q_weight, k_weight)):
        return False
    if not all(t.dtype == torch.bfloat16 for t in (q, k, cos, sin, q_weight, k_weight)):
        return False
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
        return False
    if q.shape[1] != k.shape[1] or q.shape[3] != 128 or k.shape[3] != 128:
        return False
    cos, sin = _normalize_cos_sin(cos, sin)
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        return False
    if cos.shape[0] != q.shape[1] or cos.shape[1] != 128:
        return False
    if q_weight.numel() != 128 or k_weight.numel() != 128:
        return False
    return True


def _get_outputs(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device_idx = q.device.index if q.device.index is not None else torch.cuda.current_device()
    key = (device_idx, str(q.dtype), int(q.shape[1]), int(q.shape[2]), int(k.shape[2]), int(q.shape[3]))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        q_out = torch.empty((1, q.shape[2], q.shape[1], q.shape[3]), device=q.device, dtype=q.dtype)
        k_out = torch.empty((1, k.shape[2], k.shape[1], k.shape[3]), device=k.device, dtype=k.dtype)
        _CACHE[key] = (q_out, k_out)
        return q_out, k_out


def run_prefill_qk_rmsnorm_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    cos, sin = _normalize_cos_sin(cos, sin)
    if not _can_use(q, k, cos, sin, q_weight, k_weight):
        return None
    q_out, k_out = _get_outputs(q, k)
    ext = _load_ext()
    ext.run_prefill_qk_rmsnorm_rotary_into(
        q,
        k,
        cos.contiguous(),
        sin.contiguous(),
        q_weight,
        k_weight,
        float(eps),
        q_out,
        k_out,
    )
    return q_out, k_out
