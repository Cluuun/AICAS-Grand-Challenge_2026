"""Unified KV cache utilities for EAGLE3 speculative decoding.

All cache type conversion and lifecycle operations live here.
Other modules (eagle3.py, eagle3_worker.py, eagle3_verifier.py,
eagle3_acceptor.py) should import from this file.

The mutable _aicas_eagle3_seq_len attribute on cache objects is deprecated.
Callers should track logical sequence length explicitly (e.g. via Eagle3State.seq_len).
"""

from __future__ import annotations

from typing import Any

import torch


# ================================================================
# Length helpers (deprecated pattern - prefer explicit state)
# ================================================================

def get_cache_seq_len(cache) -> int:
    """Read the logical sequence length from a cache object.

    Deprecated: prefer tracking seq_len explicitly in your state object.
    This reads the mutable _aicas_eagle3_seq_len attribute.
    """
    logical = getattr(cache, "_aicas_eagle3_seq_len", None)
    if logical is not None:
        try:
            return int(logical)
        except Exception:
            pass
    from spec_decode.verifier import _cache_seq_len
    return _cache_seq_len(cache)


def set_cache_seq_len(cache, seq_len: int) -> None:
    """Set the logical sequence length on a cache object.

    Deprecated: prefer tracking seq_len explicitly in your state object.
    """
    if cache is None:
        return
    try:
        cache._aicas_eagle3_seq_len = int(seq_len)
    except Exception:
        pass


# ================================================================
# MCL bucket selection
# ================================================================

def _parse_positive_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except Exception:
            continue
        if value > 0:
            values.append(int(value))
    return sorted(set(values))


def get_static_cache_buckets() -> list[int]:
    """Return the configured primary StaticCache bucket list."""
    import os
    raw = (
        os.getenv("AICAS_EAGLE3_TREE_VERIFY_GRAPH_MCL_BUCKETS", "").strip()
        or os.getenv("AICAS_SPEC_STATIC_CACHE_MCL_BUCKETS", "").strip()
    )
    buckets = _parse_positive_int_list(raw)
    return buckets if buckets else [1280]


def get_static_cache_overflow_buckets() -> list[int]:
    """Return the configured overflow StaticCache bucket list."""
    import os
    raw = os.getenv("AICAS_SPEC_STATIC_CACHE_OVERFLOW_BUCKETS", "").strip()
    return _parse_positive_int_list(raw)


def select_static_cache_len(total_len: int) -> int:
    """Select a StaticCache max_cache_len bucket that fits total_len."""
    import os
    buckets = get_static_cache_buckets()
    for bucket in buckets:
        if int(total_len) <= bucket:
            return int(bucket)
    overflow_buckets = get_static_cache_overflow_buckets()
    for bucket in overflow_buckets:
        if int(total_len) <= bucket:
            return int(bucket)
    try:
        overflow_step = int(os.getenv("AICAS_SPEC_STATIC_CACHE_OVERFLOW_STEP", "128"))
    except Exception:
        overflow_step = 128
    overflow_step = max(1, overflow_step)
    max_bucket = max([*buckets, *overflow_buckets]) if overflow_buckets else max(buckets)
    if int(total_len) <= max_bucket:
        return int(max_bucket)
    return int(((int(total_len) + overflow_step - 1) // overflow_step) * overflow_step)


# ================================================================
# Cache tail management
# ================================================================

def clear_static_cache_tail(cache, keep_len: int) -> None:
    """Zero out positions >= keep_len in a StaticCache.

    Does NOT set _aicas_eagle3_seq_len. Callers must manage seq_len themselves.
    Sets cumulative_length on each layer.
    """
    layers = getattr(cache, "layers", None)
    if not layers:
        return
    keep_len = max(0, int(keep_len))
    for layer in layers:
        if hasattr(layer, "cumulative_length"):
            try:
                layer.cumulative_length = keep_len
            except Exception:
                pass
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, torch.Tensor) and int(keys.shape[2]) > keep_len:
            keys[:, :, keep_len:, :].zero_()
        if isinstance(values, torch.Tensor) and int(values.shape[2]) > keep_len:
            values[:, :, keep_len:, :].zero_()


# ================================================================
# Cache copy helpers
# ================================================================

def copy_tree_prefix_into_static(src_cache, dst_cache, start_pos: int) -> None:
    """Copy prefix [0, start_pos) from src_cache into dst_cache (StaticCache).

    Does NOT set _aicas_eagle3_seq_len. Callers must manage seq_len themselves.
    """
    from spec_decode.eagle3 import _copy_cache_into_static
    if src_cache is dst_cache:
        clear_static_cache_tail(dst_cache, start_pos)
        return
    if _copy_cache_into_static is None:
        raise RuntimeError("decode graph static cache helper unavailable")
    _copy_cache_into_static(src_cache, dst_cache, seq_len=start_pos)
    clear_static_cache_tail(dst_cache, start_pos)


# ================================================================
# Cache clone (OPTIMIZED - no more copy.deepcopy)
# ================================================================

def clone_cache_to_len(past_kv, keep_len: int):
    """Create a cloned cache containing only the first keep_len positions.

    Optimized: avoids copy.deepcopy which traverses the full Python object
    graph. Instead uses narrow + copy_ for StaticCache and narrow + clone
    for DynamicCache. Falls back to deepcopy only if the cache type is
    unrecognized.
    """
    keep_len = max(0, int(keep_len))

    # -- StaticCache path (primary) --
    try:
        from transformers.cache_utils import StaticCache
        if isinstance(past_kv, StaticCache):
            mcl = int(past_kv.layers[0].keys.shape[2]) if past_kv.layers else keep_len
            cloned = StaticCache(config=past_kv._config, max_cache_len=mcl)
            for src_layer, dst_layer in zip(past_kv.layers, cloned.layers):
                if not getattr(dst_layer, "is_initialized", False):
                    seed_k = src_layer.keys[:, :, :1, :]
                    seed_v = src_layer.values[:, :, :1, :]
                    dst_layer.lazy_initialization(seed_k)
                copy_len = min(keep_len, int(src_layer.keys.shape[2]))
                dst_layer.keys[:, :, :copy_len, :].copy_(src_layer.keys[:, :, :copy_len, :])
                dst_layer.values[:, :, :copy_len, :].copy_(src_layer.values[:, :, :copy_len, :])
            return cloned
    except Exception:
        pass

    # -- DynamicCache path --
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            cloned = DynamicCache()
            for i in range(len(past_kv)):
                k = past_kv.key_cache[i][:, :, :keep_len, :].contiguous().clone()
                v = past_kv.value_cache[i][:, :, :keep_len, :].contiguous().clone()
                cloned.update(k, v, i)
            return cloned
    except Exception:
        pass

    # -- Fallback (only for unrecognized cache types) --
    import copy
    cloned = copy.deepcopy(past_kv)
    layers = getattr(cloned, "layers", None)
    if layers:
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor):
                layer.keys = keys.narrow(-2, 0, min(keep_len, int(keys.shape[-2]))).contiguous()
            if isinstance(values, torch.Tensor):
                layer.values = values.narrow(-2, 0, min(keep_len, int(values.shape[-2]))).contiguous()
            if hasattr(layer, "cumulative_length"):
                try:
                    layer.cumulative_length = keep_len
                except Exception:
                    pass
    return cloned


# ================================================================
# StaticCache factory (unifies scattered creation points)
# ================================================================

def create_static_cache_for_decode(model_config, past_kv, start_pos: int, q_len: int):
    """Create a properly-sized StaticCache and copy prefix KV into it.

    This is the single entry point for DynamicCache -> StaticCache
    conversion in the EAGLE3 decode path. It replaces 4 previously
    scattered StaticCache() constructor calls, each using a different
    MCL-selection strategy.

    Args:
        model_config: model.config (for layer count / head dims)
        past_kv: source cache (DynamicCache or another StaticCache)
        start_pos: number of valid token positions in past_kv
        q_len: number of new token positions needed for this step

    Returns:
        StaticCache pre-filled with past_kv[0:start_pos], sized to fit
        start_pos + q_len according to the configured MCL bucket policy.
    """
    from transformers.cache_utils import StaticCache
    total_len = int(start_pos) + int(q_len)
    mcl = select_static_cache_len(total_len)
    cache = StaticCache(config=model_config, max_cache_len=mcl)
    copy_tree_prefix_into_static(past_kv, cache, start_pos)
    return cache


# ================================================================
# Backward-compatible re-exports
# ================================================================

_eagle3_cache_seq_len = get_cache_seq_len
_set_eagle3_cache_seq_len = set_cache_seq_len
_select_static_cache_len = select_static_cache_len
_clear_static_cache_tail = clear_static_cache_tail
_copy_tree_prefix_into_static = copy_tree_prefix_into_static
_clone_cache_to_len = clone_cache_to_len
_create_static_cache_for_decode = create_static_cache_for_decode
