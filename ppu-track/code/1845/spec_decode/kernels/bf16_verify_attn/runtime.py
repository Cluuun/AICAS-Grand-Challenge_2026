from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent
_EXT = None
_OUT_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    so_path = _THIS_DIR / "bf16_verify_attn_ext.so"
    if not so_path.is_file():
        candidates = sorted(_THIS_DIR.glob("bf16_verify_attn_ext*.so"))
        if not candidates:
            raise FileNotFoundError(f"Missing bf16 verifier attention extension in {_THIS_DIR}")
        so_path = candidates[-1]

    module_name = "bf16_verify_attn_ext"
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


def _can_use(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    return (
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and query.ndim == 4
        and key.ndim == 4
        and value.ndim == 4
        and query.shape[0] == 1
        and query.shape[1] == 16
        and 1 <= int(query.shape[2]) <= 16
        and query.shape[3] == 128
        and key.shape[0] == 1
        and key.shape[1] == 8
        and key.shape[3] == 128
        and value.shape == key.shape
        and int(key.shape[2]) >= int(query.shape[2])
        and query.device == key.device
        and query.device == value.device
    )


def run_bf16_verify_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    if not _can_use(query, key, value):
        return None
    q = query.contiguous()
    q_len = int(q.shape[2])
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    out_key = (int(device_index), q_len)
    out = _OUT_CACHE.get(out_key)
    if out is None or out.device != q.device:
        out = torch.empty((1, q_len, 16, 128), device=q.device, dtype=torch.bfloat16)
        _OUT_CACHE[out_key] = out
    return run_bf16_verify_attention_into(q, key.contiguous(), value.contiguous(), out)


def run_bf16_verify_attention_into(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
):
    if not _can_use(query, key, value):
        return None
    if not isinstance(out, torch.Tensor) or out.dtype != torch.bfloat16 or not out.is_cuda:
        return None
    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()
    if out.shape != (1, int(q.shape[2]), 16, 128) or out.device != q.device:
        return None
    ext = _load_ext()
    ext.run_bf16_verify_attn_into(q, k, v, out)
    return out
