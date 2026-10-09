from __future__ import annotations

"""Qwen3-VL 视觉特征缓存（可选）."""

from collections import OrderedDict
from contextlib import contextmanager
import os
from types import MethodType
from typing import Any, Dict, Optional, Set, Tuple

import torch
from . import conf as _conf_module
from .conf import conf_bool, conf_int, conf_str


def _env_on(name: str, default: str = "1") -> bool:
    return conf_bool(name, default)


def _normalize_external_key(key: Any) -> Optional[Tuple]:
    if key is None:
        return None
    if torch.is_tensor(key):
        try:
            return tuple(int(x) for x in key.reshape(-1).detach().cpu().tolist())
        except Exception:
            return None
    if isinstance(key, (list, tuple)):
        try:
            return tuple(int(x) for x in key)
        except Exception:
            return tuple(str(x) for x in key)
    try:
        return (int(key),)
    except Exception:
        return (str(key),)


def _cache_key(pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor], external_key: Any = None) -> Tuple:
    """无 GPU 同步的轻量键：shape/dtype/device + grid（grid 仅几个 int）。"""
    image_key = _normalize_external_key(external_key)
    grid_key: Tuple
    if image_grid_thw is None:
        grid_key = ("none",)
    else:
        grid_key = tuple(int(x) for x in image_grid_thw.reshape(-1).detach().cpu().tolist())
    if image_key is None:
        image_key = ("shape_only",)
    return (image_key, tuple(pixel_values.shape), str(pixel_values.dtype), str(pixel_values.device), grid_key)


def _clone_output_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_output_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_output_tree(v) for v in value)
    if isinstance(value, OrderedDict):
        return OrderedDict((k, _clone_output_tree(v)) for k, v in value.items())
    if isinstance(value, dict):
        cloned = {k: _clone_output_tree(v) for k, v in value.items()}
        try:
            return value.__class__(cloned)
        except Exception:
            return cloned
    # transformers.ModelOutput 通常是 dict-like，优先保留原类型
    if hasattr(value, "items") and hasattr(value, "__class__"):
        cloned_map = {k: _clone_output_tree(v) for k, v in value.items()}
        try:
            return value.__class__(**cloned_map)
        except Exception:
            try:
                return value.__class__(cloned_map)
            except Exception:
                return cloned_map
    return value


def _vision_graph_key(vlm, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Tuple:
    grid_key = tuple(int(x) for x in image_grid_thw.reshape(-1).detach().cpu().tolist())
    visual = getattr(vlm, "visual", None)
    visual_id = id(visual) if visual is not None else id(vlm)
    return (
        str(pixel_values.device),
        str(pixel_values.dtype),
        tuple(int(x) for x in pixel_values.shape),
        grid_key,
        str(image_grid_thw.device),
        str(image_grid_thw.dtype),
        tuple(int(x) for x in image_grid_thw.shape),
        int(visual_id),
    )


def _can_use_vision_graph_fastpath(pixel_values, image_grid_thw, kwargs: Dict[str, Any]) -> bool:
    if not _env_on("ENABLE_VISION_ENCODER_CUDA_GRAPH", "1"):
        return False
    if not torch.cuda.is_available() or torch.is_grad_enabled():
        return False
    if not torch.is_tensor(pixel_values) or not pixel_values.is_cuda:
        return False
    if image_grid_thw is None or (not torch.is_tensor(image_grid_thw)):
        return False
    if image_grid_thw.numel() == 0:
        return False
    # graph 内只封装稳定的 vision forward，暂不支持附加 kwargs 分支。
    if len(kwargs) > 0:
        return False
    # 仅支持单图、无视频输入。
    if image_grid_thw.dim() >= 2 and int(image_grid_thw.shape[0]) != 1:
        return False
    if kwargs.get("video_grid_thw", None) is not None or kwargs.get("pixel_values_videos", None) is not None:
        return False
    # 固定 336 fastpath 开启时才尝试 vision graph。
    if conf_str("FIXED_IMAGE_RESOLUTION", "336x336", lower=True) != "336x336":
        return False
    flat = image_grid_thw.reshape(-1)
    if int(flat.numel()) >= 3:
        h = int(flat[-2].item())
        w = int(flat[-1].item())
        if h <= 0 or w <= 0 or h != w:
            return False
    return True


class _VisionEncoderGraphRunner:
    def __init__(self, forward_fn, warmup_iters: int, vlm_owner=None):
        self._forward_fn = forward_fn
        self._warmup_iters = max(1, int(warmup_iters))
        self._vlm_owner = vlm_owner
        self._visual = None
        self._split_sizes: Tuple[int, ...] = ()
        self._static_pos_embeds: Optional[torch.Tensor] = None
        self._static_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._static_cu_seqlens: Optional[torch.Tensor] = None
        self._deepstack_merger_map: Dict[int, Any] = {}
        self.static_pixel_values: Optional[torch.Tensor] = None
        self.static_image_grid_thw: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_outputs: Any = None

    def captured(self) -> bool:
        return self.graph is not None and self.static_outputs is not None

    @staticmethod
    @contextmanager
    def _graph_safe_runtime_overrides():
        # 在 capture 阶段使用保守 eager kernel，避免部分自定义 GEMM kernel 触发 stream-capture invalidated。
        # 只影响 capture/warmup，不影响常规 eager miss 路径。
        overrides = {
            "ENABLE_VISION_KERNEL_FUSIONS": "0",
            "ENABLE_VISION_LINEAR_MM_OUT": "0",
            "VISION_PATCH_EMBED_BACKEND": "linear",
            "ENABLE_VISION_CUBLASLT_EPILOGUE": "0",
            "ENABLE_VISION_LINEAR_CUBLASLT": "0",
        }
        old_values = {k: os.environ.get(k) for k in overrides}
        try:
            for k, v in overrides.items():
                os.environ[k] = v
            if hasattr(_conf_module, "_raw") and hasattr(_conf_module._raw, "cache_clear"):
                _conf_module._raw.cache_clear()
            yield
        finally:
            for k, old in old_values.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old
            if hasattr(_conf_module, "_raw") and hasattr(_conf_module._raw, "cache_clear"):
                _conf_module._raw.cache_clear()

    @staticmethod
    @contextmanager
    def _graph_safe_attention_patch(visual):
        if visual is None:
            yield
            return
        restore_pairs = []
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                apply_rotary_pos_emb_vision,
                eager_attention_forward,
            )
        except Exception:
            yield
            return
        try:
            for blk in getattr(visual, "blocks", []):
                attn = getattr(blk, "attn", None)
                if attn is None:
                    continue
                old_forward = getattr(attn, "forward", None)
                if old_forward is None:
                    continue

                def _graph_safe_attn_forward(
                    self,
                    hidden_states: torch.Tensor,
                    cu_seqlens: torch.Tensor,
                    rotary_pos_emb: Optional[torch.Tensor] = None,
                    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    **kwargs,
                ) -> torch.Tensor:
                    seq_length = hidden_states.shape[0]
                    query_states, key_states, value_states = (
                        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
                    )
                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
                    query_states = query_states.transpose(0, 1).unsqueeze(0)
                    key_states = key_states.transpose(0, 1).unsqueeze(0)
                    value_states = value_states.transpose(0, 1).unsqueeze(0)
                    attn_output, _ = eager_attention_forward(
                        self,
                        query_states,
                        key_states,
                        value_states,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                    )
                    attn_output = attn_output.reshape(seq_length, -1).contiguous()
                    return self.proj(attn_output)

                attn.forward = MethodType(_graph_safe_attn_forward, attn)
                restore_pairs.append((attn, old_forward))
            yield
        finally:
            for attn, old_forward in restore_pairs:
                attn.forward = old_forward

    def _prepare_graph_static_state(self) -> None:
        owner = self._vlm_owner
        visual = getattr(owner, "visual", None) if owner is not None else None
        if visual is None:
            raise RuntimeError("vision graph requires vlm.visual")
        self._visual = visual

        pos_embeds = visual.fast_pos_embed_interpolate(self.static_image_grid_thw)
        self._static_pos_embeds = pos_embeds.detach().clone()
        seq_len = int(self._static_pos_embeds.shape[0])

        rotary_pos_emb = visual.rot_pos_emb(self.static_image_grid_thw).reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        self._static_position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            self.static_image_grid_thw[:, 1] * self.static_image_grid_thw[:, 2],
            self.static_image_grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        self._static_cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        split_sizes = (
            self.static_image_grid_thw.prod(-1) // int(visual.spatial_merge_size) ** 2
        ).detach().cpu().tolist()
        self._split_sizes = tuple(int(x) for x in split_sizes)

        self._deepstack_merger_map = {
            int(layer_idx): merger
            for layer_idx, merger in zip(visual.deepstack_visual_indexes, visual.deepstack_merger_list)
        }

    def _graph_safe_forward_once(self) -> Any:
        visual = self._visual
        hidden_states = self.static_pixel_values.type(visual.dtype)
        hidden_states = visual.patch_embed(hidden_states)
        hidden_states = hidden_states + self._static_pos_embeds

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=self._static_cu_seqlens,
                position_embeddings=self._static_position_embeddings,
            )
            deepstack_merger = self._deepstack_merger_map.get(layer_num)
            if deepstack_merger is not None:
                deepstack_feature_lists.append(deepstack_merger(hidden_states))
        hidden_states = visual.merger(hidden_states)
        image_embeds = torch.split(hidden_states, self._split_sizes)
        return image_embeds, deepstack_feature_lists

    def capture(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> None:
        self.static_pixel_values = torch.empty_like(pixel_values)
        self.static_pixel_values.copy_(pixel_values)
        self.static_image_grid_thw = image_grid_thw.detach().clone()
        self._prepare_graph_static_state()
        with self._graph_safe_runtime_overrides():
            with self._graph_safe_attention_patch(self._visual):
                for _ in range(self._warmup_iters):
                    _ = self._graph_safe_forward_once()
                torch.cuda.synchronize(device=pixel_values.device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, capture_error_mode="relaxed"):
                    self.static_outputs = self._graph_safe_forward_once()
                torch.cuda.synchronize(device=pixel_values.device)

    def replay(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Any:
        if not self.captured():
            raise RuntimeError("vision graph runner is not captured")
        if (
            pixel_values.shape != self.static_pixel_values.shape
            or pixel_values.dtype != self.static_pixel_values.dtype
            or pixel_values.device != self.static_pixel_values.device
            or image_grid_thw.shape != self.static_image_grid_thw.shape
            or image_grid_thw.dtype != self.static_image_grid_thw.dtype
            or image_grid_thw.device != self.static_image_grid_thw.device
        ):
            raise RuntimeError("vision graph input mismatch")
        self.static_pixel_values.copy_(pixel_values)
        self.static_image_grid_thw.copy_(image_grid_thw)
        self.graph.replay()
        if _env_on("VISION_ENCODER_GRAPH_CLONE_OUTPUT", "1"):
            return _clone_output_tree(self.static_outputs)
        return self.static_outputs


def apply_vision_prefill_cache(model) -> bool:
    """给 Qwen3-VL 的 get_image_features 加一层小容量缓存。"""
    if not _env_on("ENABLE_VISION_FEATURE_CACHE", "1"):
        return False
    vlm = getattr(model, "model", None)
    if vlm is None or not hasattr(vlm, "get_image_features"):
        return False
    if getattr(vlm, "_vision_feature_cache_patched", False):
        return True

    original_get_image_features = vlm.get_image_features
    max_entries = conf_int("VISION_FEATURE_CACHE_SIZE", "2", minimum=1)
    cache: "OrderedDict[Tuple, object]" = OrderedDict()
    graph_warmup_iters = conf_int("VISION_ENCODER_GRAPH_WARMUP_ITERS", "2", minimum=1)
    graph_runners: Dict[Tuple, _VisionEncoderGraphRunner] = {}
    graph_disabled: Set[Tuple] = set()
    stats = {
        "hits": 0,
        "misses": 0,
        "lookup_ms": 0.0,
        "stores": 0,
        "vision_graph_capture_success": 0,
        "vision_graph_capture_fail": 0,
        "vision_graph_replay_hit": 0,
        "vision_graph_replay_fallback": 0,
        "vision_graph_last_error": "",
    }

    def cached_get_image_features(self, pixel_values=None, image_grid_thw=None, **kwargs):
        if pixel_values is None:
            return original_get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

        kwargs.pop("return_dict", None)
        external_key = kwargs.pop("_vision_cache_key", None)
        if kwargs.pop("_ttft_skip_cache", False):
            return original_get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

        import time

        t_lookup = time.perf_counter()
        try:
            key = _cache_key(pixel_values, image_grid_thw, external_key)
            cached = cache.get(key)
            stats["lookup_ms"] += (time.perf_counter() - t_lookup) * 1000.0
            if cached is not None:
                stats["hits"] += 1
                cache.move_to_end(key)
                return cached
            stats["misses"] += 1
            out = None
            used_graph = False

            if _can_use_vision_graph_fastpath(pixel_values, image_grid_thw, kwargs):
                gkey = _vision_graph_key(self, pixel_values, image_grid_thw)
                if gkey not in graph_disabled:
                    runner = graph_runners.get(gkey)
                    if runner is None:
                        runner = _VisionEncoderGraphRunner(
                            original_get_image_features,
                            graph_warmup_iters,
                            vlm_owner=self,
                        )
                        try:
                            runner.capture(pixel_values, image_grid_thw)
                            graph_runners[gkey] = runner
                            stats["vision_graph_capture_success"] += 1
                        except Exception as e:
                            graph_disabled.add(gkey)
                            stats["vision_graph_capture_fail"] += 1
                            stats["vision_graph_last_error"] = f"capture_failed:{type(e).__name__}:{str(e)[:160]}"
                            runner = None
                    if runner is not None:
                        try:
                            out = runner.replay(pixel_values, image_grid_thw)
                            stats["vision_graph_replay_hit"] += 1
                            used_graph = True
                        except Exception as e:
                            graph_disabled.add(gkey)
                            stats["vision_graph_replay_fallback"] += 1
                            stats["vision_graph_last_error"] = f"replay_failed:{type(e).__name__}:{str(e)[:160]}"
                            out = None

            if out is None:
                out = original_get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

            # graph replay 的 static output buffer 会在后续 replay 被覆盖；缓存前必须做深拷贝。
            cache_value = _clone_output_tree(out) if used_graph else out
            cache[key] = cache_value
            stats["stores"] += 1
            if len(cache) > max_entries:
                cache.popitem(last=False)
            return cache_value
        except Exception:
            stats["misses"] += 1
            return original_get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    vlm.get_image_features = MethodType(cached_get_image_features, vlm)
    vlm._vision_feature_cache_patched = True
    vlm._vision_feature_cache = cache
    vlm._vision_feature_cache_stats = stats
    vlm._vision_encoder_graph_runners = graph_runners
    vlm._vision_encoder_graph_disabled = graph_disabled
    print("[vision_prefill_cache] vision feature cache patched (no-sync key)")
    return True


def maybe_log_vision_cache_stats(vlm) -> None:
    if not _env_on("VISION_FEATURE_CACHE_LOG_STATS", "0"):
        return
    stats = getattr(vlm, "_vision_feature_cache_stats", None)
    if not isinstance(stats, dict):
        return
    hits = int(stats.get("hits", 0))
    misses = int(stats.get("misses", 0))
    total = hits + misses
    rate = (100.0 * hits / total) if total else 0.0
    print(
        f"[vision_prefill_cache] hits={hits} misses={misses} hit_rate={rate:.1f}% "
        f"lookup_ms={stats.get('lookup_ms', 0):.3f}"
    )
    if _env_on("VISION_ENCODER_GRAPH_LOG_STATS", "0"):
        print(
            "[vision_prefill_cache] "
            f"vision_graph_capture_success={int(stats.get('vision_graph_capture_success', 0))} "
            f"vision_graph_capture_fail={int(stats.get('vision_graph_capture_fail', 0))} "
            f"vision_graph_replay_hit={int(stats.get('vision_graph_replay_hit', 0))} "
            f"vision_graph_replay_fallback={int(stats.get('vision_graph_replay_fallback', 0))}"
        )
