from __future__ import annotations

"""TTFT 专用多模态 prefill：拆分视觉编码 / 文本 prefill / lm_head，并输出细分耗时。"""

import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .conf import conf_bool, conf_int, conf_str
from .prefill_cuda_graph import (
    _set_deepstack_static_range as _clear_prefill_graph_deepstack_range,
    run_text_prefill_cuda_graph,
)

try:
    from .prefill_kv_cache import make_prefill_kv_cache_key, store_prefill_kv_from_ttft
except Exception:
    def make_prefill_kv_cache_key(_input_ids, _model_kwargs):
        return None

    def store_prefill_kv_from_ttft(*args, **kwargs):
        return None

try:
    from .prefill_last_hidden import probe_language_model_last_hidden_support
except Exception:
    def probe_language_model_last_hidden_support(_language_model):
        return "unknown"

try:
    from .prefill_profiling import collect_prefill_token_fields, log_prefill_token_stats, merge_timing_stats
except Exception:
    def collect_prefill_token_fields(_input_ids, _model_kwargs, model=None):
        return {}

    def log_prefill_token_stats(_stage, _stats):
        return None

    def merge_timing_stats(base, extra):
        merged = dict(base)
        merged.update(extra)
        return merged


def _env_on(name: str, default: str = "1") -> bool:
    return conf_bool(name, default)


def _env_str(name: str, default: str) -> str:
    return conf_str(name, default, lower=True)


def _ms_since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


_PREFILL_ARANGE_CACHE: "OrderedDict[Tuple[int, str], torch.Tensor]" = OrderedDict()
_FIXED_SCATTER_RANGE_CACHE: "OrderedDict[Tuple[int, int], Tuple[int, int]]" = OrderedDict()
_FASTPATH_STATS: Dict[str, int] = {
    "scatter_slice_hit": 0,
    "scatter_slice_fallback": 0,
    "scatter_slice_cache_hit": 0,
    "scatter_slice_cache_miss": 0,
    "prefill_arange_hit": 0,
    "prefill_arange_miss": 0,
}


def _lru_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def _lru_put(cache: OrderedDict, key, value, max_entries: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _get_prefill_cache_position(seq_len: int, device: torch.device) -> torch.Tensor:
    if not _env_on("ENABLE_PREFILL_ARANGE_CACHE", "1"):
        return torch.arange(seq_len, dtype=torch.long, device=device)
    key = (int(seq_len), str(device))
    cached = _lru_get(_PREFILL_ARANGE_CACHE, key)
    if cached is not None:
        _FASTPATH_STATS["prefill_arange_hit"] += 1
        return cached
    _FASTPATH_STATS["prefill_arange_miss"] += 1
    out = torch.arange(seq_len, dtype=torch.long, device=device)
    max_entries = conf_int("PREFILL_ARANGE_CACHE_SIZE", "32", minimum=1)
    _lru_put(_PREFILL_ARANGE_CACHE, key, out, max_entries)
    return out


def _resolve_fixed_image_slice_range(
    input_ids_1d: torch.Tensor,
    image_token_id: int,
    expected_count: int,
) -> Optional[Tuple[int, int]]:
    key = (int(image_token_id), int(expected_count))
    cached = _lru_get(_FIXED_SCATTER_RANGE_CACHE, key)
    if cached is not None:
        _FASTPATH_STATS["scatter_slice_cache_hit"] += 1
        start, end = cached
        if int(input_ids_1d.shape[0]) >= end:
            region = input_ids_1d[start:end]
            has_extra_before = bool(start > 0 and torch.any(input_ids_1d[:start] == int(image_token_id)).item())
            has_extra_after = bool(end < int(input_ids_1d.shape[0]) and torch.any(input_ids_1d[end:] == int(image_token_id)).item())
            if (
                int(region.numel()) == expected_count
                and torch.all(region == int(image_token_id)).item()
                and not has_extra_before
                and not has_extra_after
            ):
                return start, end
        _FASTPATH_STATS["scatter_slice_fallback"] += 1
        return None

    _FASTPATH_STATS["scatter_slice_cache_miss"] += 1
    idx = (input_ids_1d == int(image_token_id)).nonzero(as_tuple=False).flatten()
    if int(idx.numel()) != expected_count or expected_count <= 0:
        return None
    start = int(idx[0].item())
    end = start + expected_count
    expected = torch.arange(start, end, device=idx.device, dtype=idx.dtype)
    if not torch.equal(idx, expected):
        return None
    _lru_put(_FIXED_SCATTER_RANGE_CACHE, key, (start, end), max_entries=8)
    return start, end


def _resolve_position_ids(vlm, input_ids, model_kwargs):
    position_ids = model_kwargs.get("position_ids")
    if position_ids is not None:
        return position_ids
    if not hasattr(vlm, "get_rope_index"):
        raise AttributeError("Qwen3VLModel.get_rope_index not found")
    position_ids, rope_deltas = vlm.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=model_kwargs.get("image_grid_thw"),
        video_grid_thw=model_kwargs.get("video_grid_thw"),
        attention_mask=model_kwargs.get("attention_mask"),
    )
    vlm.rope_deltas = rope_deltas
    return position_ids


def _scatter_image_embeds(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    image_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """用 index 写入替代 expand+masked_scatter，减少大张量 materialize。"""
    if (
        _env_on("ENABLE_FIXED_336_FASTPATH", "1")
        and _env_on("ENABLE_FIXED_IMAGE_SCATTER_SLICE", "1")
        and input_ids.dim() == 2
        and int(input_ids.shape[0]) == 1
        and image_embeds.dim() == 2
    ):
        expected = int(image_embeds.shape[0])
        fixed_range = _resolve_fixed_image_slice_range(input_ids[0], image_token_id, expected)
        if fixed_range is not None:
            start, end = fixed_range
            inputs_embeds[0, start:end] = image_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            _FASTPATH_STATS["scatter_slice_hit"] += 1
            return inputs_embeds

        _FASTPATH_STATS["scatter_slice_fallback"] += 1

    mask = input_ids[0] == image_token_id
    n_placeholders = int(mask.sum().item())
    if n_placeholders != image_embeds.shape[0]:
        # 回退到 HF 语义，保证正确性
        image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))
    inputs_embeds[0, mask] = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return inputs_embeds


def _normalize_image_features_output(image_outputs):
    """兼容 return_dict=True 的 ModelOutput 与 tuple 返回值。"""
    if isinstance(image_outputs, tuple):
        image_embeds_list, deepstack_image_embeds = image_outputs[0], image_outputs[1]
        return image_embeds_list, deepstack_image_embeds
    return image_outputs.pooler_output, image_outputs.deepstack_features


def build_text_only_inputs_embeds(vlm, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
    """只 embed 文本 token，跳过稍后会被视觉特征覆盖的 image/video placeholder。"""
    if not _env_on("ENABLE_TEXT_ONLY_PLACEHOLDER_EMBED", "1") or input_ids.dim() != 2:
        return None
    image_token_id = getattr(vlm.config, "image_token_id", None)
    video_token_id = getattr(vlm.config, "video_token_id", None)
    visual_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if image_token_id is not None:
        visual_mask |= input_ids == int(image_token_id)
    if video_token_id is not None:
        visual_mask |= input_ids == int(video_token_id)
    if not torch.any(visual_mask):
        return None

    embed = vlm.get_input_embeddings()
    weight = embed.weight
    inputs_embeds = torch.empty(
        (*input_ids.shape, int(weight.shape[-1])),
        dtype=weight.dtype,
        device=input_ids.device,
    )
    text_mask = ~visual_mask
    if torch.any(text_mask):
        inputs_embeds[text_mask] = embed(input_ids[text_mask])
    return inputs_embeds


def _build_visual_side(
    vlm,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
    *,
    skip_cache: Optional[bool] = None,
    vision_cache_key=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
    """视觉编码 + 写入 image placeholder。"""
    if skip_cache is None:
        skip_cache = not _env_on("ENABLE_VISION_FEATURE_CACHE_TTFT", "0")
    image_outputs = vlm.get_image_features(
        pixel_values,
        image_grid_thw,
        _ttft_skip_cache=skip_cache,
        _vision_cache_key=vision_cache_key,
    )
    image_embeds_list, deepstack_image_embeds = _normalize_image_features_output(image_outputs)
    image_embeds = torch.cat(image_embeds_list, dim=0)
    image_token_id = int(vlm.config.image_token_id)
    inputs_embeds = _scatter_image_embeds(inputs_embeds, input_ids, image_embeds, image_token_id)
    # language_model._deepstack_process 需要 [B, S] mask
    visual_pos_masks = input_ids == image_token_id
    return inputs_embeds, visual_pos_masks, deepstack_image_embeds


def _argmax_next_token(model, last_hidden: torch.Tensor, backend: str) -> torch.Tensor:
    lm_head = getattr(model, "lm_head", None)
    if backend == "fused_lm_head" and lm_head is not None:
        try:
            from .lm_head_top1 import fused_lm_head_argmax

            return fused_lm_head_argmax(
                last_hidden,
                lm_head.weight,
                getattr(lm_head, "bias", None),
            )
        except Exception:
            pass
    if lm_head is not None:
        logits = lm_head(last_hidden)
        return torch.argmax(logits, dim=-1)
    raise RuntimeError("ttft_fastpath requires model.lm_head")


def _build_ttft_static_kv_cache(model, seq_len: int, device: torch.device, dtype: torch.dtype):
    from .cuda_graph_decode import StaticKVCache, _bucket_max_cache_len

    return StaticKVCache.from_model(
        model,
        max_cache_len=_bucket_max_cache_len(int(seq_len) + 1),
        batch_size=int(1),
        dtype=dtype,
        device=device,
    )


def ttft_forward_and_argmax(
    model,
    input_ids: torch.Tensor,
    model_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    TTFT 专用 forward：get_image_features -> inputs_embeds -> language_model -> lm_head。
    返回 (next_token[B], stats_ms)。
    """
    if not _env_on("ENABLE_TTFT_DEDICATED_FORWARD", "1"):
        return _ttft_fallback_full_model(model, input_ids, model_kwargs)

    vlm = getattr(model, "model", None)
    if vlm is None or not hasattr(vlm, "language_model"):
        return _ttft_fallback_full_model(model, input_ids, model_kwargs)

    token_backend = _env_str("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head")
    stats: Dict[str, float] = {
        "vision_ms": 0.0,
        "embed_merge_ms": 0.0,
        "text_prefill_ms": 0.0,
        "lm_head_ms": 0.0,
        "cache_lookup_ms": 0.0,
        "total_ms": 0.0,
        "path": "dedicated",
        "use_static_kv": 0.0,
    }
    stats["prefill_last_hidden_mode"] = probe_language_model_last_hidden_support(vlm.language_model)
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    inputs_embeds = build_text_only_inputs_embeds(vlm, input_ids)
    if inputs_embeds is None:
        inputs_embeds = vlm.get_input_embeddings()(input_ids)
    stats["embed_merge_ms"] += _ms_since(t0)

    pixel_values = model_kwargs.get("pixel_values")
    image_grid_thw = model_kwargs.get("image_grid_thw")
    visual_pos_masks = None
    deepstack_visual_embeds = None

    if pixel_values is not None:
        t_vision = time.perf_counter()
        try:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = _build_visual_side(
                vlm,
                input_ids,
                inputs_embeds,
                pixel_values,
                image_grid_thw,
                vision_cache_key=model_kwargs.get("_vision_cache_key"),
            )
        except Exception:
            return _ttft_fallback_full_model(model, input_ids, model_kwargs)
        stats["vision_ms"] = _ms_since(t_vision)

    t_pos = time.perf_counter()
    try:
        position_ids = _resolve_position_ids(vlm, input_ids, model_kwargs)
    except Exception:
        return _ttft_fallback_full_model(model, input_ids, model_kwargs)
    stats["embed_merge_ms"] += _ms_since(t_pos)

    seq_len = int(input_ids.shape[-1])
    cache_position = _get_prefill_cache_position(seq_len, input_ids.device)
    use_static_kv = _env_on("ENABLE_TTFT_STATIC_KV", "1")
    past_key_values = None
    use_cache = False

    t_graph_prefill = time.perf_counter()
    graph_result = run_text_prefill_cuda_graph(
        model=model,
        vlm=vlm,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=model_kwargs.get("attention_mask"),
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        past_key_values=None,
        use_cache=True,
    )
    if graph_result is not None:
        stats["text_prefill_ms"] = _ms_since(t_graph_prefill)
        stats["lm_head_ms"] = 0.0
        stats["total_ms"] = _ms_since(t_total)
        stats["path"] = graph_result.mode
        stats["use_static_kv"] = 1.0
        for key, value in graph_result.stats.items():
            if isinstance(value, (int, float, str)):
                stats[f"text_graph_{key}"] = value

        vision_cache = getattr(vlm, "_vision_feature_cache_stats", None)
        if isinstance(vision_cache, dict):
            stats["vision_cache_hits"] = float(vision_cache.get("hits", 0))
            stats["vision_cache_misses"] = float(vision_cache.get("misses", 0))
            stats["cache_lookup_ms"] = float(vision_cache.get("lookup_ms", 0.0))
            stats["vision_graph_capture_success"] = float(vision_cache.get("vision_graph_capture_success", 0))
            stats["vision_graph_capture_fail"] = float(vision_cache.get("vision_graph_capture_fail", 0))
            stats["vision_graph_replay_hit"] = float(vision_cache.get("vision_graph_replay_hit", 0))
            stats["vision_graph_replay_fallback"] = float(vision_cache.get("vision_graph_replay_fallback", 0))
            stats["vision_graph_last_error"] = str(vision_cache.get("vision_graph_last_error", ""))

        return graph_result.next_token.reshape(-1), stats

    _clear_prefill_graph_deepstack_range(vlm, None)

    if use_static_kv:
        try:
            past_key_values = _build_ttft_static_kv_cache(model, seq_len, input_ids.device, inputs_embeds.dtype)
            use_cache = True
            stats["path"] = "dedicated_static_kv"
            stats["use_static_kv"] = 1.0
        except Exception:
            past_key_values = None
            use_cache = False

    t_prefill = time.perf_counter()
    try:
        lm_outputs = vlm.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=model_kwargs.get("attention_mask"),
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            return_dict=True,
        )
    except TypeError:
        # 旧版 signature 无 cache_position
        lm_outputs = vlm.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=model_kwargs.get("attention_mask"),
            past_key_values=past_key_values,
            use_cache=use_cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            return_dict=True,
        )
    stats["text_prefill_ms"] = _ms_since(t_prefill)

    last_hidden = lm_outputs.last_hidden_state
    if last_hidden.shape[1] > 1:
        last_hidden = last_hidden[:, -1:, :]
    t_lm = time.perf_counter()
    lm_backend = "fused_lm_head" if token_backend == "direct_lm_head" else token_backend
    next_token = _argmax_next_token(model, last_hidden, lm_backend)
    stats["lm_head_ms"] = _ms_since(t_lm)
    stats["total_ms"] = _ms_since(t_total)
    stats["scatter_slice_hit"] = float(_FASTPATH_STATS.get("scatter_slice_hit", 0))
    stats["scatter_slice_fallback"] = float(_FASTPATH_STATS.get("scatter_slice_fallback", 0))
    stats["prefill_arange_hit"] = float(_FASTPATH_STATS.get("prefill_arange_hit", 0))
    stats["prefill_arange_miss"] = float(_FASTPATH_STATS.get("prefill_arange_miss", 0))

    vision_cache = getattr(vlm, "_vision_feature_cache_stats", None)
    if isinstance(vision_cache, dict):
        stats["vision_cache_hits"] = float(vision_cache.get("hits", 0))
        stats["vision_cache_misses"] = float(vision_cache.get("misses", 0))
        stats["cache_lookup_ms"] = float(vision_cache.get("lookup_ms", 0.0))
        stats["vision_graph_capture_success"] = float(vision_cache.get("vision_graph_capture_success", 0))
        stats["vision_graph_capture_fail"] = float(vision_cache.get("vision_graph_capture_fail", 0))
        stats["vision_graph_replay_hit"] = float(vision_cache.get("vision_graph_replay_hit", 0))
        stats["vision_graph_replay_fallback"] = float(vision_cache.get("vision_graph_replay_fallback", 0))
        stats["vision_graph_last_error"] = str(vision_cache.get("vision_graph_last_error", ""))

    if use_cache and past_key_values is not None and hasattr(past_key_values, "k_slab"):
        next_cache_position = torch.tensor([seq_len], dtype=torch.long, device=input_ids.device)
        next_position_ids = (position_ids[..., -1:] + 1).contiguous()
        rope_deltas = getattr(vlm, "rope_deltas", None)
        store_prefill_kv_from_ttft(
            model,
            input_ids=input_ids,
            cache_key=make_prefill_kv_cache_key(input_ids, model_kwargs),
            prompt_len=seq_len,
            next_token=next_token,
            static_kv=past_key_values,
            next_position_ids=next_position_ids,
            next_cache_position=next_cache_position,
            rope_deltas=rope_deltas,
        )

    if conf_bool("LOG_PREFILL_TOKEN_STATS", "0"):
        token_fields = collect_prefill_token_fields(input_ids, model_kwargs, model=model)
        if isinstance(vision_cache, dict):
            hits = int(vision_cache.get("hits", 0))
            misses = int(vision_cache.get("misses", 0))
            token_fields["vision_cache"] = "hit" if hits > misses else ("miss" if misses else "n/a")
        log_prefill_token_stats(
            "ttft",
            merge_timing_stats(
                {
                    **token_fields,
                    "path": stats.get("path"),
                    "prefill_last_hidden_mode": stats.get("prefill_last_hidden_mode"),
                },
                stats,
            ),
        )

    return next_token.reshape(-1), stats


def _ttft_fallback_full_model(
    model,
    input_ids: torch.Tensor,
    model_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """回退到完整 CausalLM forward。"""
    from .cuda_graph_decode import _select_next_token_from_outputs

    token_backend = _env_str("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head")
    t0 = time.perf_counter()
    prefill_inputs = dict(model_kwargs)
    prefill_inputs["input_ids"] = input_ids
    prefill_inputs["cache_position"] = _get_prefill_cache_position(
        int(input_ids.shape[-1]), input_ids.device
    )
    prefill_inputs["use_cache"] = False
    prefill_inputs.setdefault("logits_to_keep", 1)
    if token_backend == "fused_lm_head":
        prefill_inputs["output_hidden_states"] = True
    outputs = model(**prefill_inputs, return_dict=True)
    next_token = _select_next_token_from_outputs(model, outputs, token_backend)
    total_ms = _ms_since(t0)
    return next_token.reshape(-1), {
        "vision_ms": 0.0,
        "embed_merge_ms": 0.0,
        "text_prefill_ms": total_ms,
        "lm_head_ms": 0.0,
        "cache_lookup_ms": 0.0,
        "total_ms": total_ms,
        "path": "full_model_fallback",
    }


def log_ttft_breakdown(model, stats: Dict[str, float]) -> None:
    if not _env_on("LOG_TTFT_BREAKDOWN", "1"):
        return
    parts = [
        f"path={stats.get('path', '?')}",
        f"total={stats.get('total_ms', 0):.2f}ms",
        f"vision={stats.get('vision_ms', 0):.2f}ms",
        f"embed_merge={stats.get('embed_merge_ms', 0):.2f}ms",
        f"text_prefill={stats.get('text_prefill_ms', 0):.2f}ms",
        f"lm_head={stats.get('lm_head_ms', 0):.2f}ms",
        f"cache_lookup={stats.get('cache_lookup_ms', 0):.2f}ms",
    ]
    if "vision_cache_hits" in stats:
        parts.append(f"vcache_hit={int(stats['vision_cache_hits'])}")
        parts.append(f"vcache_miss={int(stats['vision_cache_misses'])}")
    if "vision_graph_replay_hit" in stats:
        parts.append(f"vgraph_cap_ok={int(stats['vision_graph_capture_success'])}")
        parts.append(f"vgraph_cap_fail={int(stats['vision_graph_capture_fail'])}")
        parts.append(f"vgraph_replay={int(stats['vision_graph_replay_hit'])}")
        parts.append(f"vgraph_fb={int(stats['vision_graph_replay_fallback'])}")
        if conf_bool("VISION_ENCODER_GRAPH_LOG_STATS", "0") and stats.get("vision_graph_last_error"):
            parts.append(f"vgraph_err={stats.get('vision_graph_last_error')}")
    if "scatter_slice_hit" in stats:
        parts.append(f"scatter_hit={int(stats['scatter_slice_hit'])}")
        parts.append(f"scatter_fb={int(stats['scatter_slice_fallback'])}")
    if "prefill_arange_hit" in stats:
        parts.append(f"arange_hit={int(stats['prefill_arange_hit'])}")
        parts.append(f"arange_miss={int(stats['prefill_arange_miss'])}")
    if "text_graph_replay" in stats:
        parts.append(f"tgraph_cap_ok={int(stats.get('text_graph_capture_success', 0))}")
        parts.append(f"tgraph_cap_fail={int(stats.get('text_graph_capture_fail', 0))}")
        parts.append(f"tgraph_replay={int(stats.get('text_graph_replay', 0))}")
        parts.append(f"tgraph_fb_long={int(stats.get('text_graph_fallback_long', 0))}")
    print("[ttft_breakdown] " + " ".join(parts))
