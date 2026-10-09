"""KV Cache 管理：压缩、创建、复制、长度操作。"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# 模块级复用 KV cache
_cache = None
_self_spec_draft_cache = None
_self_spec_verify_cache = None


# ---------------------------------------------------------------------------
# KV Cache 压缩 (MixKV-inspired importance + diversity selection)
# ---------------------------------------------------------------------------

def _compact_visual_kv(cache, visual_mask_1d, keep_ratio, debug=None):
    """Compact static KV cache using importance+diversity selection (MixKV-inspired).

    Instead of uniform subsampling, selects visual tokens with the highest VALUE
    norm (importance) while ensuring semantic diversity via KEY cosine distance.

    Args:
        cache: Static KV cache with .layers[i].keys/values [batch, heads, max_seq, dim]
        visual_mask_1d: bool tensor [seq_len] marking visual token positions
        keep_ratio: fraction of visual tokens to keep (0.0-1.0)
        debug: optional dict to store debug info

    Returns:
        new_prefill_len or None if no compaction done
    """
    num_visual = int(visual_mask_1d.sum().item())
    seq_len = visual_mask_1d.shape[0]

    if num_visual <= 1 or keep_ratio >= 1.0:
        if debug is not None:
            debug["kv_compact"] = "skip"
        return None

    visual_indices = torch.nonzero(visual_mask_1d, as_tuple=False).flatten()
    keep_n = max(1, int(round(num_visual * keep_ratio)))
    device = visual_indices.device

    # --- Importance: VALUE L2 norm averaged across last N layers and heads ---
    importance = torch.zeros(num_visual, device=device, dtype=torch.float32)
    n_sample_layers = min(4, len(cache.layers))
    for layer in cache.layers[-n_sample_layers:]:
        # values: [1, num_kv_heads, max_seq, head_dim]
        v_vis = layer.values[0, :, visual_indices, :]  # [num_kv_heads, num_visual, head_dim]
        importance += v_vis.float().norm(dim=-1).mean(dim=0)  # [num_visual]
    importance /= n_sample_layers

    if keep_n >= num_visual:
        kept_offsets = torch.arange(num_visual, device=device)
    elif keep_n == 1:
        kept_offsets = importance.argmax().unsqueeze(0)
    else:
        # --- Diversity: greedy max-min via KEY cosine similarity ---
        # Build candidate pool from top-importance tokens
        pool_size = min(num_visual, max(keep_n * 10, keep_n + 50))
        _, pool_idx = torch.topk(importance, pool_size)

        # Key vectors for pool (last layer, averaged across heads)
        k_pool = cache.layers[-1].keys[0, :, visual_indices[pool_idx], :].float()
        k_avg = k_pool.mean(dim=0)  # [pool_size, head_dim]
        k_norm = k_avg / (k_avg.norm(dim=-1, keepdim=True) + 1e-8)

        # Pairwise cosine similarity matrix
        sim_matrix = torch.mm(k_norm, k_norm.T)  # [pool_size, pool_size]

        # Greedy selection: maximize importance * (1 - max_similarity_to_selected)
        selected = torch.zeros(keep_n, dtype=torch.long, device=device)
        mask = torch.zeros(pool_size, dtype=torch.bool, device=device)
        selected[0] = 0  # highest importance first
        mask[0] = True

        pool_importance = importance[pool_idx]
        for i in range(1, keep_n):
            max_sim = sim_matrix[:, mask].max(dim=1).values  # [pool_size]
            scores = pool_importance * (1.0 - max_sim)
            scores[mask] = -1e9
            best = scores.argmax()
            selected[i] = best
            mask[best] = True

        kept_offsets = pool_idx[selected]

    keep_visual = visual_indices[kept_offsets]

    # Build keep mask: all non-visual + kept visual
    keep_mask = ~visual_mask_1d.clone()
    keep_mask[keep_visual] = True
    keep_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
    new_len = keep_indices.shape[0]

    if new_len >= seq_len:
        if debug is not None:
            debug["kv_compact"] = "skip_no_reduction"
        return None

    # Compact each layer's KV cache
    for layer in cache.layers:
        k_compact = layer.keys[:, :, keep_indices, :]
        v_compact = layer.values[:, :, keep_indices, :]
        layer.keys[:, :, :new_len, :].copy_(k_compact)
        layer.values[:, :, :new_len, :].copy_(v_compact)

    if debug is not None:
        debug["kv_compact"] = "applied_importance_diversity"
        debug["kv_compact_visual_before"] = num_visual
        debug["kv_compact_visual_after"] = int(kept_offsets.numel())
        debug["kv_compact_prefill_before"] = seq_len
        debug["kv_compact_prefill_after"] = new_len

    return new_len


def _maybe_compact_kv(cache, visual_mask_1d, prefill_len, rope_delta,
                      max_new_tokens, debug, source="unknown"):
    """Shared KV compaction gating logic for both prefix_reuse and full_prefill paths.

    Returns (prefill_len, rope_delta) after compaction.
    """
    compact_keep_ratio = float(os.environ.get("AICAS_KV_COMPACT_KEEP_RATIO", "0"))
    _compact_max = int(os.environ.get("AICAS_KV_COMPACT_MAX_TOKENS", "128"))
    if compact_keep_ratio <= 0 or max_new_tokens > _compact_max or cache is None:
        return prefill_len, rope_delta
    if visual_mask_1d is None or (not visual_mask_1d.any()):
        return prefill_len, rope_delta

    # Optional: skip sensitive questions
    _compact_skip_sensitive = os.environ.get("AICAS_KV_COMPACT_SKIP_SENSITIVE", "0") == "1"
    if _compact_skip_sensitive:
        try:
            import _eval_impl.config as _ew_cfg
            from _eval_impl.resolution import _is_prune_sensitive
            if _is_prune_sensitive(_ew_cfg._last_question_text):
                return prefill_len, rope_delta
        except Exception:
            pass

    old_prefill_len = prefill_len
    with torch.inference_mode():
        new_len = _compact_visual_kv(cache, visual_mask_1d, compact_keep_ratio, debug)
    if new_len is not None:
        prefill_len = new_len
        rope_delta += (old_prefill_len - new_len)
        debug[f"kv_compact_{source}"] = "applied"
        debug["kv_compact_prefill_len"] = int(prefill_len)
        debug["kv_compact_rope_delta"] = int(rope_delta)
    return prefill_len, rope_delta


# ---------------------------------------------------------------------------
# Cache 创建与复用
# ---------------------------------------------------------------------------

def _get_or_create_cache(num_layers, num_kv_heads, head_dim, max_seq_len, dtype, device):
    """获取或创建可复用的GraphStaticKVCache"""
    global _cache
    from custom_kernels.cache.graph_cache import GraphStaticKVCache

    if _cache is None or _cache._max_seq_len < max_seq_len:
        _cache = GraphStaticKVCache(
            num_layers=num_layers, num_kv_heads=num_kv_heads,
            head_dim=head_dim, max_seq_len=max_seq_len,
            dtype=dtype, device=device,
        )
    else:
        # 重置复用 — 只清位置，不重新分配
        _cache.reset()
    return _cache


def _get_or_create_self_spec_cache(num_layers, num_kv_heads, head_dim, max_seq_len, dtype, device):
    """获取自投机专用cache，避免污染完整模型cache。"""
    global _self_spec_draft_cache
    from custom_kernels.cache.graph_cache import GraphStaticKVCache

    if _self_spec_draft_cache is None or _self_spec_draft_cache._max_seq_len < max_seq_len:
        _self_spec_draft_cache = GraphStaticKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )
    else:
        _self_spec_draft_cache.reset()
    return _self_spec_draft_cache


def _get_or_create_self_spec_verify_cache(num_layers, num_kv_heads, head_dim, max_seq_len, dtype, device):
    global _self_spec_verify_cache
    from custom_kernels.cache.graph_cache import GraphStaticKVCache

    if _self_spec_verify_cache is None or _self_spec_verify_cache._max_seq_len < max_seq_len:
        _self_spec_verify_cache = GraphStaticKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=device,
        )
    else:
        _self_spec_verify_cache.reset()
    return _self_spec_verify_cache


# ---------------------------------------------------------------------------
# Cache 辅助操作
# ---------------------------------------------------------------------------

def _drop_flash_cache_attrs(cache):
    for attr in ['flash_k_caches', 'flash_v_caches', 'cache_seqlens',
                 'flashinfer_wrapper', 'flashinfer_cache_pos']:
        if hasattr(cache, attr):
            delattr(cache, attr)


def _set_cache_len(cache, seq_len, layer_indices=None):
    """回退/推进静态cache的逻辑长度；旧KV不清零，靠长度裁剪屏蔽。"""
    if layer_indices is None:
        layers = range(len(cache.layers))
    else:
        layers = layer_indices
    for idx in layers:
        layer = cache.layers[idx]
        layer._pos_val = seq_len
        layer.position.fill_(seq_len)
    cache._seq_len = seq_len


def _copy_cache_prefix(src, dst, seq_len):
    with torch.inference_mode():
        for src_layer, dst_layer in zip(src.layers, dst.layers):
            dst_layer.keys[:, :, :seq_len, :].copy_(src_layer.keys[:, :, :seq_len, :])
            dst_layer.values[:, :, :seq_len, :].copy_(src_layer.values[:, :, :seq_len, :])
            dst_layer._pos_val = seq_len
            dst_layer.position.fill_(seq_len)
        dst._seq_len = seq_len


def _copy_cache_range(src, dst, start, end, layer_indices=None):
    if end <= start:
        return
    if layer_indices is None:
        layer_indices = range(len(src.layers))
    with torch.inference_mode():
        for idx in layer_indices:
            src_layer = src.layers[idx]
            dst_layer = dst.layers[idx]
            dst_layer.keys[:, :, start:end, :].copy_(src_layer.keys[:, :, start:end, :])
            dst_layer.values[:, :, start:end, :].copy_(src_layer.values[:, :, start:end, :])


def _parse_skip_layers(skip_spec, num_layers):
    skip_spec = str(skip_spec or "").strip()
    if not skip_spec:
        return {num_layers - 1}
    layers = set()
    for part in skip_spec.replace("+", ",").split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx += num_layers
        if 0 <= idx < num_layers:
            layers.add(idx)
    return layers or {num_layers - 1}
