#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_ORIG_SDPA = F.scaled_dot_product_attention

_VERBOSE = os.environ.get("FLASHATTN_CUDA_VERBOSE", "0") == "1"
_LOG_FALLBACK = os.environ.get("FLASHATTN_CUDA_LOG_FALLBACK", "0") == "1"
_LOG_HITS = os.environ.get("FLASHATTN_CUDA_LOG_HITS", "0") == "1"
_LOG_POLICY = os.environ.get("FLASHATTN_CUDA_LOG_POLICY", "0") == "1"

_POLICY_PATH = Path(os.environ.get("FLASHATTN_CUDA_POLICY_PATH", str(_THIS_DIR / "dispatch_policy.json")))


@dataclass(frozen=True)
class KernelConfig:
    batch: int
    heads_q: int
    heads_kv: int
    seq_len: int
    head_dim: int
    causal: bool


@dataclass(frozen=True)
class KernelSpec:
    name: str
    ext_name: str
    cfg: KernelConfig


@dataclass
class LoadedKernel:
    spec: KernelSpec
    ext: object


_TEXT_KERNEL_SPECS = [
    KernelSpec(
        name="text_prefill_640",
        ext_name="flashattn_cuda_text640_ext",
        cfg=KernelConfig(batch=1, heads_q=16, heads_kv=8, seq_len=640, head_dim=128, causal=True),
    ),
    KernelSpec(
        name="text_prefill_704",
        ext_name="flashattn_cuda_text704_ext",
        cfg=KernelConfig(batch=1, heads_q=16, heads_kv=8, seq_len=704, head_dim=128, causal=True),
    ),
    KernelSpec(
        name="text_prefill_768",
        ext_name="flashattn_cuda_text768_ext",
        cfg=KernelConfig(batch=1, heads_q=16, heads_kv=8, seq_len=768, head_dim=128, causal=True),
    ),
    KernelSpec(
        name="text_prefill_832",
        ext_name="flashattn_cuda_text832_ext",
        cfg=KernelConfig(batch=1, heads_q=16, heads_kv=8, seq_len=832, head_dim=128, causal=True),
    ),
    KernelSpec(
        name="text_prefill_1024",
        ext_name="flashattn_cuda_text1024_ext",
        cfg=KernelConfig(batch=1, heads_q=16, heads_kv=8, seq_len=1024, head_dim=128, causal=True),
    ),
    KernelSpec(
        name="text_prefill_1088",
        ext_name="flashattn_cuda_text1088_ext",
        cfg=KernelConfig(batch=1, heads_q=16, heads_kv=8, seq_len=1088, head_dim=128, causal=True),
    ),
]

_SPECS_BY_NAME = {spec.name: spec for spec in _TEXT_KERNEL_SPECS}

_KERNEL_CACHE: dict[str, Optional[LoadedKernel]] = {}
_EXT_CACHE: dict[str, object] = {}
_PAD_BUFFER_CACHE: dict[tuple[int, str, int, int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_FALLBACK_LOGGED = False
_HIT_LOGGED: set[str] = set()
_POLICY_LOGGED: set[str] = set()

_POLICY_CACHE_LOADED = False
_POLICY_ENTRIES: dict[str, str] = {}
_POLICY_META: dict[str, object] = {}


def _load_extension(ext_name: str) -> object:
    cached = _EXT_CACHE.get(ext_name)
    if cached is not None:
        return cached

    so_path = _THIS_DIR / f"{ext_name}.so"
    if not so_path.is_file():
        raise FileNotFoundError(f"Missing prebuilt extension: {so_path}")

    spec = importlib.util.spec_from_file_location(ext_name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {so_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[ext_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "forward_bhsd"):
        raise AttributeError(f"{so_path} does not export forward_bhsd")

    _EXT_CACHE[ext_name] = mod
    if _VERBOSE:
        print(f"Loaded prebuilt TileLang flashattn extension: {so_path}")
    return mod


def _load_kernel(spec: KernelSpec) -> Optional[LoadedKernel]:
    cached = _KERNEL_CACHE.get(spec.name)
    if cached is not None or spec.name in _KERNEL_CACHE:
        return cached

    try:
        ext = _load_extension(spec.ext_name)
    except Exception:
        _KERNEL_CACHE[spec.name] = None
        if _VERBOSE:
            raise
        return None

    loaded = LoadedKernel(spec=spec, ext=ext)
    _KERNEL_CACHE[spec.name] = loaded
    return loaded


def _load_policy_if_needed() -> None:
    global _POLICY_CACHE_LOADED
    if _POLICY_CACHE_LOADED:
        return

    _POLICY_CACHE_LOADED = True
    if not _POLICY_PATH.is_file():
        if _VERBOSE:
            print(f"[flashattn] policy file not found: {_POLICY_PATH}")
        return

    try:
        data = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        if _VERBOSE:
            print(f"[flashattn] failed to read policy file {_POLICY_PATH}: {exc}")
        return

    entries = data.get("entries", {})
    if isinstance(entries, dict):
        _POLICY_ENTRIES.update({str(k): str(v) for k, v in entries.items()})
    _POLICY_META.update(data.get("meta", {}))

    if _VERBOSE:
        print(f"[flashattn] loaded dispatch policy from {_POLICY_PATH} entries={len(_POLICY_ENTRIES)}")


def _policy_key(is_causal: bool, b: int, hq: int, hkv: int, d: int, q_len: int) -> str:
    return f"{int(bool(is_causal))}|{b}|{hq}|{hkv}|{d}|{q_len}"


def _policy_decision(is_causal: bool, query: torch.Tensor, key: torch.Tensor) -> str:
    _load_policy_if_needed()
    k = _policy_key(
        bool(is_causal),
        int(query.size(0)),
        int(query.size(1)),
        int(key.size(1)),
        int(query.size(3)),
        int(query.size(2)),
    )
    decision = _POLICY_ENTRIES.get(k, "torch")

    if _LOG_POLICY and k not in _POLICY_LOGGED:
        _POLICY_LOGGED.add(k)
        print(f"[flashattn][policy] key={k} decision={decision}")

    return decision


def _log_fallback_once(msg: str) -> None:
    global _FALLBACK_LOGGED
    if _LOG_FALLBACK and (not _FALLBACK_LOGGED):
        print(msg)
        _FALLBACK_LOGGED = True


def _log_hit_once(name: str, cfg: KernelConfig, padded: bool) -> None:
    if not _LOG_HITS:
        return
    key = f"{name}:{int(padded)}"
    if key in _HIT_LOGGED:
        return
    _HIT_LOGGED.add(key)
    mode = "padded" if padded else "exact"
    print(
        f"Using TileLang kernel [{name}] ({mode}) "
        f"B={cfg.batch} Hq={cfg.heads_q} Hkv={cfg.heads_kv} S={cfg.seq_len} D={cfg.head_dim} causal={cfg.causal}"
    )


def _scale_matches(scale: Optional[float], head_dim: int) -> bool:
    if scale is None:
        return True
    expected = 1.0 / math.sqrt(float(head_dim))
    try:
        return abs(float(scale) - expected) <= 1e-5
    except (TypeError, ValueError):
        return False


def _fallback_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
):
    kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = scale

    try:
        return _ORIG_SDPA(query, key, value, enable_gqa=enable_gqa, **kwargs)
    except TypeError:
        if enable_gqa and query.size(1) != key.size(1):
            groups = query.size(1) // key.size(1)
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        return _ORIG_SDPA(query, key, value, **kwargs)


def _common_eligibility(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask,
    dropout_p: float,
) -> bool:
    if attn_mask is not None:
        return False
    if dropout_p != 0.0:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if not (query.dtype == torch.bfloat16 and key.dtype == torch.bfloat16 and value.dtype == torch.bfloat16):
        return False
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False
    if query.device != key.device or query.device != value.device:
        return False
    if query.size(0) != key.size(0) or key.size(0) != value.size(0):
        return False
    if key.shape != value.shape:
        return False
    return True


def _get_pad_buffers(device: torch.device, cfg: KernelConfig, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        device.index if device.index is not None else 0,
        str(dtype),
        cfg.heads_q,
        cfg.heads_kv,
        cfg.seq_len,
        cfg.head_dim,
    )
    cached = _PAD_BUFFER_CACHE.get(key)
    if cached is not None:
        return cached

    q_pad = torch.empty(cfg.batch, cfg.heads_q, cfg.seq_len, cfg.head_dim, device=device, dtype=dtype)
    k_pad = torch.empty(cfg.batch, cfg.heads_kv, cfg.seq_len, cfg.head_dim, device=device, dtype=dtype)
    v_pad = torch.empty(cfg.batch, cfg.heads_kv, cfg.seq_len, cfg.head_dim, device=device, dtype=dtype)
    _PAD_BUFFER_CACHE[key] = (q_pad, k_pad, v_pad)
    return q_pad, k_pad, v_pad


def _try_dispatch_to_kernel(
    loaded: LoadedKernel,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: Optional[float],
) -> Optional[torch.Tensor]:
    cfg = loaded.spec.cfg

    if not _scale_matches(scale, cfg.head_dim):
        return None

    q_batch, q_heads, q_len, q_dim = query.shape
    k_batch, k_heads, k_len, k_dim = key.shape

    if q_batch != cfg.batch or k_batch != cfg.batch:
        return None
    if q_heads != cfg.heads_q or k_heads != cfg.heads_kv:
        return None
    if q_dim != cfg.head_dim or k_dim != cfg.head_dim:
        return None
    if not cfg.causal or (not is_causal):
        return None
    if q_len != k_len:
        return None
    if q_len > cfg.seq_len:
        return None

    if q_len == cfg.seq_len:
        q_exact = query
        k_exact = key
        v_exact = value
        if (not query.is_contiguous()) or (not key.is_contiguous()) or (not value.is_contiguous()):
            q_pad, k_pad, v_pad = _get_pad_buffers(query.device, cfg, query.dtype)
            if not query.is_contiguous():
                q_pad.copy_(query)
                q_exact = q_pad
            if not key.is_contiguous():
                k_pad.copy_(key)
                k_exact = k_pad
            if not value.is_contiguous():
                v_pad.copy_(value)
                v_exact = v_pad
        _log_hit_once(loaded.spec.name, cfg, padded=False)
        return loaded.ext.forward_bhsd(q_exact, k_exact, v_exact)

    q_pad, k_pad, v_pad = _get_pad_buffers(query.device, cfg, query.dtype)
    q_pad[:, :, :q_len, :].copy_(query)
    k_pad[:, :, :k_len, :].copy_(key)
    v_pad[:, :, :k_len, :].copy_(value)

    out_pad = loaded.ext.forward_bhsd(q_pad, k_pad, v_pad)
    _log_hit_once(loaded.spec.name, cfg, padded=True)
    return out_pad[:, :, :q_len, :]


def _dispatch_from_policy(
    decision: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    scale: Optional[float],
) -> Optional[torch.Tensor]:
    if decision == "torch":
        return None

    if not decision.startswith("tile:"):
        return None

    parts = decision.split(":")
    if len(parts) < 2:
        return None

    kernel_name = parts[1]
    spec = _SPECS_BY_NAME.get(kernel_name)
    if spec is None:
        return None

    loaded = _load_kernel(spec)
    if loaded is None:
        return None

    return _try_dispatch_to_kernel(loaded, query, key, value, bool(is_causal), scale)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
):
    if not _common_eligibility(query, key, value, attn_mask, dropout_p):
        _log_fallback_once("Falling back to PyTorch SDPA for unsupported attention config.")
        return _fallback_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    if query.size(1) != key.size(1) and (not enable_gqa):
        _log_fallback_once("Falling back to PyTorch SDPA because enable_gqa=False for grouped heads.")
        return _fallback_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    decision = _policy_decision(bool(is_causal), query, key)
    out = _dispatch_from_policy(decision, query, key, value, bool(is_causal), scale)
    if out is not None:
        return out

    _log_fallback_once("Falling back to PyTorch SDPA for unsupported attention config.")
    return _fallback_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


def patch_torch_sdpa() -> None:
    F.scaled_dot_product_attention = scaled_dot_product_attention


def unpatch_torch_sdpa() -> None:
    F.scaled_dot_product_attention = _ORIG_SDPA
