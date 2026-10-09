from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent
_EXT = None
_OUT_CACHE: dict[tuple[int, torch.device, object], tuple[torch.Tensor, torch.Tensor]] = {}


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    so_path = _THIS_DIR / "bf16_qk_rotary_ext.so"
    if not so_path.is_file():
        candidates = sorted(_THIS_DIR.glob("bf16_qk_rotary_ext*.so"))
        if not candidates:
            raise FileNotFoundError(f"Missing bf16 qk rotary extension in {_THIS_DIR}")
        so_path = candidates[-1]

    module_name = "bf16_qk_rotary_ext"
    if module_name in sys.modules:
        _EXT = sys.modules[module_name]
        return _EXT

    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load extension spec from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _EXT = mod
    return mod


def is_available() -> bool:
    try:
        _load_ext()
        return True
    except Exception:
        return False


def _can_use(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
) -> bool:
    if not all(isinstance(t, torch.Tensor) for t in (q, k, cos, sin, q_weight, k_weight)):
        return False
    if not all(t.is_cuda for t in (q, k, cos, sin, q_weight, k_weight)):
        return False
    if not all(t.dtype == torch.bfloat16 for t in (q, k, cos, sin, q_weight, k_weight)):
        return False
    if q.ndim != 2 or k.ndim != 2 or q.shape[1] != 128 or k.shape[1] != 128:
        return False
    if q.shape[0] % 16 != 0 or k.shape[0] % 8 != 0:
        return False
    q_len = int(q.shape[0]) // 16
    if q_len < 1 or q_len > 16 or int(k.shape[0]) // 8 != q_len:
        return False
    if cos.numel() != q_len * 128 or sin.numel() != q_len * 128:
        return False
    if q_weight.numel() != 128 or k_weight.numel() != 128:
        return False
    return True


def run_bf16_qk_rmsnorm_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    cache_tag: object | None = None,
):
    if not _can_use(q, k, cos, sin, q_weight, k_weight):
        return None
    ext = _load_ext()
    q = q.contiguous()
    k = k.contiguous()
    q_len = int(q.shape[0]) // 16
    cache_key = (q_len, q.device, cache_tag)
    cached = _OUT_CACHE.get(cache_key)
    if cached is None or cached[0].device != q.device:
        cached = (
            torch.empty((1, 16, q_len, 128), device=q.device, dtype=torch.bfloat16),
            torch.empty((1, 8, q_len, 128), device=q.device, dtype=torch.bfloat16),
        )
        _OUT_CACHE[cache_key] = cached
    q_out, k_out = cached
    ext.run_bf16_qk_rmsnorm_rotary_into(
        q,
        k,
        cos.reshape(q_len, 128).contiguous(),
        sin.reshape(q_len, 128).contiguous(),
        q_weight.reshape(128).contiguous(),
        k_weight.reshape(128).contiguous(),
        float(eps),
        q_out,
        k_out,
    )
    return q_out, k_out
