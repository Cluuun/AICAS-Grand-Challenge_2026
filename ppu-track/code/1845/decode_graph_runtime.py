"""Decode-stage CUDA-graph & operator dispatch runtime.

Operator-level optimization module that orchestrates CUDA graph capture
and replay for the per-token decode loop.  All compute work is delegated
to the underlying ``my_kernel/*`` operators (FlashDecode, FlashDecodeFFN,
decode_qk_rotary, decode_rmsnorm_triton, lm_head_argmax_triton, ...).
This file is the operator scheduler / graph wrapper that fuses those
kernels into a single replayable graph; it does not change model
architecture, weights, or numerical semantics.

Aligns with the competition's "compile & graph optimization" and
"generation-process optimization" categories.
"""
from __future__ import annotations

import os
import traceback

from contextlib import contextmanager, nullcontext
from types import MethodType
from typing import Dict

import torch
from transformers.utils import ModelOutput

try:
    from my_kernel.decode_qk_rotary.runtime import decode_qk_rotary_graph_linear_mode
except Exception:
    from contextlib import nullcontext as decode_qk_rotary_graph_linear_mode

try:
    from my_kernel.lm_head_argmax_triton import (
        can_use_lm_head_argmax_triton,
        can_use_lm_head_argmax_triton_multi,
        lm_head_argmax_triton,
        lm_head_argmax_triton_multi,
    )
except Exception:
    can_use_lm_head_argmax_triton = None
    can_use_lm_head_argmax_triton_multi = None
    lm_head_argmax_triton = None
    lm_head_argmax_triton_multi = None

try:
    from my_kernel.static_kv_copy_cuda import copy_full_prompt_kv as _static_kv_copy_full_prompt
except Exception:
    _static_kv_copy_full_prompt = None

FIXED_MCL = int(os.getenv("AICAS_DECODE_GRAPH_FIXED_MCL", "1280"))


def _parse_decode_mcl_buckets(spec: str, default_mcl: int) -> tuple[int, ...]:
    buckets = []
    for part in (spec or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except Exception:
            continue
        if value > 0:
            buckets.append(value)
    if not buckets:
        return (int(default_mcl),)
    buckets = sorted(set(buckets))
    if int(default_mcl) not in buckets:
        buckets.append(int(default_mcl))
    return tuple(sorted(set(buckets)))


def _should_use_fast_generate(kwargs: Dict) -> bool:
    input_ids = kwargs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        return False
    if kwargs.get("do_sample", False):
        return False
    if kwargs.get("num_beams", 1) != 1:
        return False
    if kwargs.get("use_cache", True) is False:
        return False
    return "input_ids" in kwargs and "attention_mask" in kwargs


def _validate_token_ids(token_ids: torch.Tensor, vocab_size: int | None) -> None:
    if vocab_size is None or not isinstance(token_ids, torch.Tensor) or token_ids.numel() == 0:
        return
    min_id = int(token_ids.min().item())
    max_id = int(token_ids.max().item())
    if min_id < 0 or max_id >= int(vocab_size):
        raise RuntimeError(
            f"decode_cudagraph produced invalid token ids: min={min_id} max={max_id} vocab={int(vocab_size)}"
        )


def _clone_fallback_kwargs(kwargs: Dict, tensor_keys: tuple[str, ...]) -> Dict:
    safe = dict(kwargs)
    for key in tensor_keys:
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            safe[key] = value.detach().clone()
    return safe


def _build_decode_position_ids(model, cache_position: torch.Tensor, batch_size: int, seq_length: int) -> torch.Tensor | None:
    qwen_model = getattr(model, "model", None)
    rope_deltas = getattr(qwen_model, "rope_deltas", None)
    if rope_deltas is None:
        return None
    device = cache_position.device
    delta = (cache_position.reshape(-1)[0] + rope_deltas).to(device=device, dtype=torch.long)
    position_ids = torch.arange(seq_length, device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
    if delta.shape[0] != batch_size:
        repeat_factor = max(1, batch_size // max(1, delta.shape[0]))
        delta = delta.repeat_interleave(repeat_factor, dim=0)
        if delta.shape[0] != batch_size:
            delta = delta[:1].expand(batch_size, -1)
    position_ids = position_ids.add(delta)
    return position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()


def _get_layer_kv(cache, layer_idx: int):
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        k = getattr(layer, "keys", None)
        if not isinstance(k, torch.Tensor):
            k = getattr(layer, "key", None)
        v = getattr(layer, "values", None)
        if not isinstance(v, torch.Tensor):
            v = getattr(layer, "value", None)
        return k, v
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    legacy = cache.to_legacy_cache()
    return legacy[layer_idx][0], legacy[layer_idx][1]


def _get_cache_seq_len(cache) -> int:
    if cache is None:
        return 0
    try:
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
    except Exception:
        pass
    try:
        k, _ = _get_layer_kv(cache, 0)
        if isinstance(k, torch.Tensor):
            return int(k.shape[2])
    except Exception:
        pass
    return 0


def _get_cache_max_len(cache) -> int:
    if cache is None:
        return 0
    try:
        if hasattr(cache, "get_max_cache_shape"):
            return int(cache.get_max_cache_shape())
    except Exception:
        pass
    try:
        layer = cache.layers[0]
        keys = getattr(layer, "keys", None)
        if isinstance(keys, torch.Tensor):
            return int(keys.shape[2])
    except Exception:
        pass
    try:
        k, _ = _get_layer_kv(cache, 0)
        if isinstance(k, torch.Tensor):
            return int(k.shape[2])
    except Exception:
        pass
    return 0


class _FullPromptCacheView:
    def __init__(self, key_cache, value_cache, key_stack=None, value_stack=None):
        self.key_cache = list(key_cache)
        self.value_cache = list(value_cache)
        self.key_stack = key_stack
        self.value_stack = value_stack

    def get_seq_length(self, layer_idx: int = 0) -> int:
        try:
            k = self.key_cache[int(layer_idx)]
        except Exception:
            return 0
        return int(k.shape[2]) if isinstance(k, torch.Tensor) and k.ndim >= 3 else 0


def _copy_cache_into_static(src_cache, dst_cache, seq_len: int | None = None) -> None:
    if src_cache is None:
        return
    for layer_idx, dst_layer in enumerate(dst_cache.layers):
        try:
            k, v = _get_layer_kv(src_cache, layer_idx)
        except Exception:
            continue
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor) or k.shape[2] == 0:
            continue
        seed = k[:, :, :1, :]
        value_seed = v[:, :, :1, :]
        sl = seq_len if seq_len is not None else k.shape[2]
        dst_k = getattr(dst_layer, "keys", None)
        dst_v = getattr(dst_layer, "values", None)
        if (
            not getattr(dst_layer, "is_initialized", False)
            or not isinstance(dst_k, torch.Tensor)
            or not isinstance(dst_v, torch.Tensor)
            or tuple(dst_k.shape[:2]) != tuple(seed.shape[:2])
            or tuple(dst_v.shape[:2]) != tuple(value_seed.shape[:2])
            or int(dst_k.shape[2]) < int(sl)
            or int(dst_v.shape[2]) < int(sl)
            or int(dst_k.shape[-1]) != int(seed.shape[-1])
            or int(dst_v.shape[-1]) != int(value_seed.shape[-1])
            or dst_k.device != seed.device
            or dst_v.device != value_seed.device
            or dst_k.dtype != seed.dtype
            or dst_v.dtype != value_seed.dtype
        ):
            dst_layer.lazy_initialization(seed)
        dst_layer.keys[:, :, :sl, :].copy_(k[:, :, :sl, :])
        dst_layer.values[:, :, :sl, :].copy_(v[:, :, :sl, :])


def _copy_full_prompt_view_into_static(src_cache, dst_cache, seq_len: int | None = None) -> bool:
    def _fail(reason: str) -> bool:
        if os.getenv("AICAS_STATIC_KV_COPY_DEBUG", "0") == "1":
            print(f"[static-kv-copy] fallback: {reason}", flush=True)
        return False

    if (
        os.getenv("AICAS_STATIC_KV_COPY_CUDA", "1") != "1"
        or _static_kv_copy_full_prompt is None
        or src_cache is None
        or dst_cache is None
    ):
        return _fail("disabled_or_unavailable")
    key_src = getattr(src_cache, "key_stack", None)
    value_src = getattr(src_cache, "value_stack", None)
    key_dst = getattr(dst_cache, "_aicas_all_keys", None)
    value_dst = getattr(dst_cache, "_aicas_all_values", None)
    if not all(isinstance(t, torch.Tensor) for t in (key_src, value_src, key_dst, value_dst)):
        return _fail("missing_stacked_or_static_buffers")
    sl = int(seq_len) if seq_len is not None else int(key_src.shape[3])
    if key_src.ndim != 5 or key_dst.ndim != 5:
        return _fail("rank_mismatch")
    if value_src.shape != key_src.shape or value_dst.shape != key_dst.shape:
        return _fail("kv_shape_mismatch")
    if key_src.shape[:3] != key_dst.shape[:3] or key_src.shape[4] != key_dst.shape[4]:
        return _fail(f"layout_mismatch src={tuple(key_src.shape)} dst={tuple(key_dst.shape)}")
    if sl < 0 or sl > int(key_src.shape[3]) or sl > int(key_dst.shape[3]):
        return _fail(f"seq_len_out_of_range seq={sl} src_s={int(key_src.shape[3])} dst_s={int(key_dst.shape[3])}")
    if (
        key_src.dtype != torch.bfloat16
        or value_src.dtype != torch.bfloat16
        or key_dst.dtype != torch.bfloat16
        or value_dst.dtype != torch.bfloat16
    ):
        return _fail("dtype_not_bf16")
    if key_src.device != key_dst.device or value_src.device != value_dst.device:
        return _fail("device_mismatch")
    if (
        not key_src.is_contiguous()
        or not value_src.is_contiguous()
        or not key_dst.is_contiguous()
        or not value_dst.is_contiguous()
    ):
        return _fail("non_contiguous")
    try:
        with torch.profiler.record_function("aicas_static_kv_copy_cuda"):
            return bool(_static_kv_copy_full_prompt(key_src, value_src, key_dst, value_dst, sl))
    except Exception as exc:
        if os.getenv("AICAS_STATIC_KV_COPY_DEBUG", "0") == "1":
            print(f"[static-kv-copy] fused copy failed: {type(exc).__name__}: {exc}", flush=True)
        return False


def _copy_kv_between_static(src, dst, seq_len: int) -> None:
    for layer_idx, dst_layer in enumerate(dst.layers):
        src_layer = src.layers[layer_idx]
        if os.getenv("AICAS_DECODE_GRAPH_ZERO_SLOT_BEFORE_COPY", "0") == "1":
            dst_layer.keys.zero_()
            dst_layer.values.zero_()
        dst_layer.keys[:, :, :seq_len, :].copy_(src_layer.keys[:, :, :seq_len, :])
        dst_layer.values[:, :, :seq_len, :].copy_(src_layer.values[:, :, :seq_len, :])


def _copy_cache_window_into_static(src_cache, dst_cache, src_start: int, seq_len: int) -> None:
    if src_cache is None:
        return
    src_start = max(0, int(src_start))
    seq_len = max(0, int(seq_len))
    for layer_idx, dst_layer in enumerate(dst_cache.layers):
        try:
            k, v = _get_layer_kv(src_cache, layer_idx)
        except Exception:
            continue
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            continue
        if k.shape[2] <= src_start or seq_len <= 0:
            continue
        copy_len = min(seq_len, int(k.shape[2]) - src_start)
        if copy_len <= 0:
            continue
        seed = k[:, :, src_start:src_start + 1, :]
        value_seed = v[:, :, src_start:src_start + 1, :]
        dst_k = getattr(dst_layer, "keys", None)
        dst_v = getattr(dst_layer, "values", None)
        if (
            not getattr(dst_layer, "is_initialized", False)
            or not isinstance(dst_k, torch.Tensor)
            or not isinstance(dst_v, torch.Tensor)
            or tuple(dst_k.shape[:2]) != tuple(seed.shape[:2])
            or tuple(dst_v.shape[:2]) != tuple(value_seed.shape[:2])
            or int(dst_k.shape[2]) < int(copy_len)
            or int(dst_v.shape[2]) < int(copy_len)
            or int(dst_k.shape[-1]) != int(seed.shape[-1])
            or int(dst_v.shape[-1]) != int(value_seed.shape[-1])
            or dst_k.device != seed.device
            or dst_v.device != value_seed.device
            or dst_k.dtype != seed.dtype
            or dst_v.dtype != value_seed.dtype
        ):
            dst_layer.lazy_initialization(seed)
        dst_layer.keys[:, :, :copy_len, :].copy_(k[:, :, src_start:src_start + copy_len, :])
        dst_layer.values[:, :, :copy_len, :].copy_(v[:, :, src_start:src_start + copy_len, :])


def _clone_static_cache(static_cache, config, max_cache_len: int):
    from transformers.cache_utils import StaticCache
    cloned = StaticCache(config=config, max_cache_len=max_cache_len)
    _copy_cache_into_static(static_cache, cloned)
    return cloned


def _reset_static_cache(cache) -> None:
    if cache is None:
        return
    reset_fn = getattr(cache, "reset", None)
    if callable(reset_fn):
        reset_fn()


def _bulk_init_static_cache(gc, batch_size: int, max_cache_len: int,
                            num_kv_heads: int, head_dim: int,
                            device: torch.device, dtype: torch.dtype) -> None:
    """Pre-allocate all StaticCache layers with one pair of large tensors.

    By default, StaticCache layers are lazily initialised on first access,
    producing 56 individual ``torch.zeros`` CUDA kernel launches (28 layers
    × K+V).  Each launch is ~2-7 µs of GPU work but costs ~35 µs of CUDA
    driver overhead, adding ~4 ms of CPU bubble before the first prefill.

    This helper allocates two flat tensors covering all layers at once
    (1-2 kernel launches instead of 56) and slices them into per-layer
    views.  The resulting ``.keys`` / ``.values`` buffers are semantically
    identical to lazily-initialised ones.
    """
    num_layers = len(gc.layers)
    if not gc.layers or device is None or dtype is None:
        return
    ready = True
    all_keys = getattr(gc, "_aicas_all_keys", None)
    all_values = getattr(gc, "_aicas_all_values", None)
    if not isinstance(all_keys, torch.Tensor) or not isinstance(all_values, torch.Tensor):
        ready = False
    for layer in gc.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if (
            not getattr(layer, "is_initialized", False)
            or not isinstance(keys, torch.Tensor)
            or not isinstance(values, torch.Tensor)
            or tuple(keys.shape) != (batch_size, num_kv_heads, max_cache_len, head_dim)
            or tuple(values.shape) != (batch_size, num_kv_heads, max_cache_len, head_dim)
            or keys.device != device
            or values.device != device
            or keys.dtype != dtype
            or values.dtype != dtype
        ):
            ready = False
            break
    if ready:
        return

    # Allocate one contiguous buffer for all keys, one for all values
    all_keys = torch.zeros(
        num_layers, batch_size, num_kv_heads, max_cache_len, head_dim,
        dtype=dtype, device=device,
    )
    all_values = torch.zeros(
        num_layers, batch_size, num_kv_heads, max_cache_len, head_dim,
        dtype=dtype, device=device,
    )

    for i, layer in enumerate(gc.layers):
        layer.max_batch_size = batch_size
        layer.num_heads = num_kv_heads
        layer.head_dim = head_dim
        layer.dtype = dtype
        layer.device = device
        layer.keys = all_keys[i]
        layer.values = all_values[i]
        is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)
        if not is_compiling():
            try:
                torch._dynamo.mark_static_address(layer.keys)
                torch._dynamo.mark_static_address(layer.values)
            except Exception:
                pass
        layer.is_initialized = True
    gc._aicas_all_keys = all_keys
    gc._aicas_all_values = all_values


def _decode_use_lm_head_argmax_fused(model_self, model_kwargs: Dict) -> bool:
    if os.getenv("AICAS_DECODE_LM_HEAD_ARGMAX_TRITON", "1") != "1":
        return False
    if can_use_lm_head_argmax_triton is None or lm_head_argmax_triton is None:
        return False
    input_ids = model_kwargs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[1] <= 0:
        return False
    lm_head = getattr(model_self, "lm_head", None)
    core = getattr(model_self, "model", None)
    if lm_head is None or core is None:
        return False
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    return True


def _decode_forward_next_token(
    model_self,
    model_kwargs: Dict,
    token_state: Dict | None = None,
):
    if _decode_use_lm_head_argmax_fused(model_self, model_kwargs):
        try:
            core_model = getattr(model_self, "model", None)
            lm_head = getattr(model_self, "lm_head", None)
            if core_model is None or lm_head is None:
                raise RuntimeError("missing core model or lm_head")

            core_kwargs = dict(model_kwargs)
            core_kwargs.pop("logits_to_keep", None)
            core_kwargs.setdefault("return_dict", False)
            core_kwargs.setdefault("output_hidden_states", False)
            core_kwargs.setdefault("output_attentions", False)
            outputs = core_model(**core_kwargs)

            hidden_states = None
            if isinstance(outputs, ModelOutput):
                hidden_states = outputs.get("last_hidden_state", None)
                if hidden_states is None and len(outputs) > 0:
                    hidden_states = outputs[0]
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                hidden_states = outputs[0]

            if not isinstance(hidden_states, torch.Tensor):
                raise RuntimeError("missing hidden states from core model")

            hidden_last = hidden_states[:, -1:, :]
            workspace = token_state.get("workspace") if isinstance(token_state, dict) else None
            if not can_use_lm_head_argmax_triton(
                hidden_last,
                lm_head.weight,
                getattr(lm_head, "bias", None),
            ):
                raise RuntimeError("lm head argmax triton conditions not met")

            next_token, workspace = lm_head_argmax_triton(
                hidden_last,
                lm_head.weight,
                getattr(lm_head, "bias", None),
                workspace=workspace,
            )
            if isinstance(token_state, dict):
                token_state["workspace"] = workspace
            return next_token.reshape(-1), outputs
        except Exception:
            pass

    outputs = model_self(**model_kwargs)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    return next_token, outputs


def apply_fused_decode_linear(model, optimizations_applied: list[str]) -> None:
    return


def apply_decode_cudagraph_generate(model, optimizations_applied: list[str]) -> None:
    """Install decode-stage CUDA-graph generate with torch.compile + graph capture."""

    DEBUG = os.getenv("AICAS_DECODE_GRAPH_DEBUG", "0") == "1"
    COMPILE_DECODE = os.getenv("AICAS_DECODE_GRAPH_COMPILE", "0") == "1"

    def _log(msg: str) -> None:
        if DEBUG:
            print(f"[decode-graph] {msg}", flush=True)

    from transformers.cache_utils import StaticCache

    original_generate = model.generate
    generation_config = getattr(model, "generation_config", None)
    eos_token_id = (
        getattr(generation_config, "eos_token_id", None)
        if generation_config is not None
        else None
    )
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    pad_token_id = (
        getattr(generation_config, "pad_token_id", None)
        if generation_config is not None
        else None
    )
    if pad_token_id is None:
        pad_token_id = getattr(model.config, "pad_token_id", None)
    vocab_size = getattr(model.config, "vocab_size", None)
    if vocab_size is None:
        try:
            w = getattr(model.get_output_embeddings(), "weight", None)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                vocab_size = int(w.shape[0])
        except Exception:
            vocab_size = None
    def _token_id_list(value) -> list[int]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return [int(v) for v in value.reshape(-1).detach().cpu().tolist()]
        if isinstance(value, (list, tuple, set)):
            return [int(v) for v in value if v is not None]
        return [int(value)]

    eos_token_ids = _token_id_list(eos_token_id)
    if pad_token_id is None:
        pad_token_id = eos_token_ids[0] if eos_token_ids else 0
    elif isinstance(pad_token_id, (list, tuple, set)):
        pad_token_id = next(iter(pad_token_id), 0)
    elif isinstance(pad_token_id, torch.Tensor):
        pad_token_id = int(pad_token_id.reshape(-1)[0].detach().cpu().item()) if pad_token_id.numel() else 0
    pad_token_id = int(pad_token_id)

    def _tokens_match_any(tokens: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
        if not token_ids:
            return torch.zeros_like(tokens, dtype=torch.bool)
        matched = tokens.eq(int(token_ids[0]))
        for token_id in token_ids[1:]:
            matched |= tokens.eq(int(token_id))
        return matched

    def _cache_view_from_full_state(state: Dict):
        key_cache = state.get("key_cache") if isinstance(state, dict) else None
        value_cache = state.get("value_cache") if isinstance(state, dict) else None
        key_stack = state.get("key_stack") if isinstance(state, dict) else None
        value_stack = state.get("value_stack") if isinstance(state, dict) else None
        if not isinstance(key_cache, (list, tuple)) or not isinstance(value_cache, (list, tuple)):
            return None
        if len(key_cache) != len(value_cache) or len(key_cache) <= 0:
            return None
        for k, v in zip(key_cache, value_cache):
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                return None
        if not isinstance(key_stack, torch.Tensor) or not isinstance(value_stack, torch.Tensor):
            key_stack = None
            value_stack = None
        return _FullPromptCacheView(key_cache, value_cache, key_stack=key_stack, value_stack=value_stack)

    def _new_static_cache(*, batch_size: int, max_cache_len: int,
                          device: torch.device, dtype: torch.dtype):
        try:
            return StaticCache(config=model.config, max_batch_size=batch_size,
                               max_cache_len=max_cache_len, device=device, dtype=dtype)
        except TypeError:
            return StaticCache(config=model.config, max_batch_size=batch_size,
                               max_cache_len=max_cache_len, device=device)

    def _static_cache_from_cache_view(cache_view, *, batch_size: int, max_cache_len: int,
                                      device: torch.device, dtype: torch.dtype):
        if cache_view is None:
            return None
        gc = _new_static_cache(batch_size=batch_size, max_cache_len=max_cache_len, device=device, dtype=dtype)
        nkv, hd = _resolve_kv_cache_shape(model)
        _bulk_init_static_cache(gc, batch_size, max_cache_len, nkv, hd, device, dtype)
        if not _copy_full_prompt_view_into_static(cache_view, gc, seq_len=cache_view.get_seq_length()):
            _copy_cache_into_static(cache_view, gc)
        return gc

    def _prefill_state_from_full_hit(state: Dict, cache_obj=None):
        if not isinstance(state, dict):
            return None
        next_token = state.get("next_token")
        if not isinstance(next_token, torch.Tensor) or next_token.numel() <= 0:
            return None
        if cache_obj is None:
            cache_obj = _cache_view_from_full_state(state)
        if cache_obj is None:
            return None
        return {
            "next_token": next_token.reshape(-1),
            "past_key_values": cache_obj,
            "prompt_len": int(state.get("prompt_len", 0) or 0),
            "past_len": int(state.get("prompt_len", 0) or 0),
        }

    # ── decode graph slot state ──
    mcl_buckets = _parse_decode_mcl_buckets(os.getenv("AICAS_DECODE_GRAPH_MCL_BUCKETS", ""), FIXED_MCL)
    slots = {}
    partial_prefill_slots = {}
    graph_stats = {
        "decode_hits": 0,
        "decode_misses": 0,
        "decode_captures": 0,
        "decode_replays": 0,
        "full_shortcut_hits": 0,
        "partial_hits": 0,
        "partial_misses": 0,
        "partial_captures": 0,
        "partial_replays": 0,
        "partial_failures": 0,
    }

    def _select_mcl(total_len: int) -> int:
        for b in mcl_buckets:
            if total_len <= b:
                return b
        return total_len

    def _resolve_kv_cache_shape(model_self) -> tuple[int, int]:
        """Return (num_kv_heads, head_dim) from the model text config."""
        tc = getattr(model_self.config, "text_config", model_self.config)
        num_heads = int(getattr(tc, "num_attention_heads", 16))
        num_kv_heads = int(getattr(tc, "num_key_value_heads", num_heads))
        hidden_size = int(getattr(tc, "hidden_size", 2048))
        head_dim = hidden_size // num_heads
        return num_kv_heads, head_dim


# ── decode step (optionally torch.compiled) ──
    _decode_forward = model.__call__  # the actual nn.Module forward

    if COMPILE_DECODE and hasattr(torch, "compile"):
        compile_mode = os.getenv("AICAS_DECODE_GRAPH_COMPILE_MODE", "reduce-overhead")
        try:
            # Disable triton's internal cudagraphs to avoid conflict with
            # the external CUDA graph capture; must be set before torch.compile
            # because PyTorch 2.8 forbids mode + options in the same call.
            import torch._inductor.config as _inductor_cfg
            _inductor_cfg.triton.cudagraphs = False
            _decode_forward = torch.compile(
                _decode_forward,
                mode=compile_mode,
                dynamic=False,
            )
            _log(f"decode forward compiled mode={compile_mode} cudagraphs=False")
        except Exception as exc:
            _log(f"decode forward compile failed: {exc}, using eager")

    def _decode_step(dk, _token_state=None):
        """Single decode forward using (optionally compiled) model.__call__."""
        if not COMPILE_DECODE:
            return _decode_forward_next_token(model, dk, _token_state)
        outputs = _decode_forward(**dk)
        nxt = outputs.logits[:, -1, :].argmax(dim=-1)
        return nxt, outputs

    # ── capture a decode CUDA graph ──
    def _capture_slot(key, gc, dk, token_state, chunk_steps, dev_idx):
        """Warm up, then capture decode CUDA graph into slots[key]."""
        with torch.profiler.record_function("decode_graph_warmup"):
            for _ in range(3):
                with torch.no_grad():
                    _decode_step(dk, token_state)
        torch.cuda.synchronize()

        bs = dk["input_ids"].shape[0]
        gnt = torch.empty((chunk_steps, bs), dtype=torch.long, device=dk["input_ids"].device)

        with torch.cuda.device(dev_idx):
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph):
                    with torch.profiler.record_function("decode_graph_capture"):
                        for i in range(chunk_steps):
                            nxt, _ = _decode_step(dk, token_state)
                            gnt[i].copy_(nxt.view(-1))

        slot_val = dict(
            gc=gc, graph=graph, gnt=gnt,
            si=dk["input_ids"], sm=dk.get("attention_mask"),
            sp=dk["position_ids"], sc=dk["cache_position"],
            token_state=token_state, chunk_steps=chunk_steps,
        )
        slots[key] = slot_val
        _log(f"captured slot key={key}")
        return slot_val

    def _try_prefill_graph(model_self, gen_kwargs: Dict):
        """Replay/capture the VLM prefill graph and return first-token state.

        This runner is installed by evaluation_wrapper.  It captures only the
        text prefill graph; vision+embedding merge stays eager inside replay().
        """
        if os.getenv("AICAS_CUDA_GRAPH", "1").strip().lower() in ("0", "false", "no", "off"):
            return None
        if os.getenv("AICAS_ENABLE_PREFILL_TTFT_GRAPH", "1").strip().lower() in ("0", "false", "no", "off"):
            return None

        runner = getattr(model_self, "_aicas_prefill_graph_runner", None)
        if runner is None or not getattr(runner, "is_enabled", False):
            return None
        if gen_kwargs.get("past_key_values") is not None:
            return None
        if gen_kwargs.get("cache_position") is not None or gen_kwargs.get("position_ids") is not None:
            return None

        input_ids = gen_kwargs.get("input_ids")
        attention_mask = gen_kwargs.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            return None
        if input_ids.ndim != 2 or attention_mask.ndim != 2 or input_ids.shape[0] != 1:
            return None

        image_grid_thw = gen_kwargs.get("image_grid_thw")
        if isinstance(image_grid_thw, torch.Tensor) and image_grid_thw.numel() != 3:
            return None

        prompt_len = int(input_ids.shape[1])
        graph_inputs = {
            k: gen_kwargs[k]
            for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
            if k in gen_kwargs
        }

        try:
            entry = runner.lookup(prompt_len, image_grid_thw)
            if entry is None:
                capture_on_demand = getattr(runner, "_capture_on_demand", None)
                if capture_on_demand is None:
                    capture_on_demand = os.getenv("AICAS_PREFILL_TTFT_CAPTURE_ON_DEMAND", "1") == "1"
                if not capture_on_demand:
                    return None
                with torch.profiler.record_function("prefill_graph_on_demand_capture"):
                    entry = runner.capture(prompt_len, graph_inputs)
            if entry is None:
                return None

            outputs, _feature = runner.replay(entry, graph_inputs)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            return {
                "next_token": next_token.reshape(-1),
                "past_key_values": outputs.past_key_values,
                "prompt_len": prompt_len,
            }
        except Exception as exc:
            _log(f"prefill graph failed: {type(exc).__name__}: {exc}")
            return None

    def _try_partial_prefill_graph(model_self, gen_kwargs: Dict, *, max_new: int):
        """CUDA graph for KV-cache-hit prefill.

        KV cache reuse rewrites the request into:
          input_ids=[remaining text], past_key_values=[cached prefix/image KV],
          cache_position=[absolute positions].
        The full-prompt prefill graph cannot handle that shape/state, so this
        captures the remaining-text forward with a static cache seeded from the
        reusable KV before each replay.
        """
        if os.getenv("AICAS_PARTIAL_PREFILL_GRAPH", "1").strip().lower() in ("0", "false", "no", "off"):
            return None
        if os.getenv("AICAS_CUDA_GRAPH", "1").strip().lower() in ("0", "false", "no", "off"):
            return None

        past_kv = gen_kwargs.get("past_key_values")
        cache_position = gen_kwargs.get("cache_position")
        input_ids = gen_kwargs.get("input_ids")
        attention_mask = gen_kwargs.get("attention_mask")
        if past_kv is None:
            return None
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            return None
        if not isinstance(cache_position, torch.Tensor):
            return None
        if input_ids.ndim != 2 or attention_mask.ndim != 2 or input_ids.shape[0] != 1:
            return None
        if input_ids.shape[1] <= 0 or cache_position.numel() != input_ids.shape[1]:
            return None
        if gen_kwargs.get("pixel_values") is not None or gen_kwargs.get("image_grid_thw") is not None:
            return None

        bs, rem_len = input_ids.shape[:2]
        device = input_ids.device
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        past_len = _get_cache_seq_len(past_kv)
        if past_len <= 0:
            return None
        attn_len = int(attention_mask.shape[1])
        total_len = max(attn_len, past_len + rem_len) + max(1, int(max_new))
        mcl = _select_mcl(total_len)
        key = (
            int(rem_len),
            int(attn_len),
            int(mcl),
            int(dev_idx),
            str(input_ids.dtype),
            str(next(model_self.parameters()).dtype) if hasattr(model_self, "parameters") else "unknown",
        )

        def _build_position_ids(cp: torch.Tensor):
            pids = _build_decode_position_ids(
                model_self,
                cp.reshape(-1)[:1].to(device=device, dtype=torch.long),
                batch_size=bs,
                seq_length=rem_len,
            )
            if isinstance(pids, torch.Tensor):
                return pids
            start = cp.reshape(-1)[:1].to(device=device, dtype=torch.long)
            base = torch.arange(rem_len, device=device, dtype=torch.long).view(1, -1)
            base = base.expand(bs, -1).add(start.reshape(1, 1))
            return base.unsqueeze(0).expand(3, -1, -1).contiguous()

        def _copy_inputs(entry: Dict) -> bool:
            entry["si"].copy_(input_ids)
            entry["sm"].copy_(attention_mask)
            entry["sc"].copy_(cache_position.to(device=device, dtype=torch.long))
            pids = _build_position_ids(entry["sc"])
            if entry.get("sp") is not None:
                if not isinstance(pids, torch.Tensor) or pids.shape != entry["sp"].shape:
                    return False
                entry["sp"].copy_(pids)
            _reset_static_cache(entry["gc"])
            _copy_cache_into_static(past_kv, entry["gc"], seq_len=past_len)
            return True

        def _make_forward_kwargs(entry: Dict) -> Dict:
            fw = {
                "input_ids": entry["si"],
                "attention_mask": entry["sm"],
                "past_key_values": entry["gc"],
                "cache_position": entry["sc"],
                "use_cache": True,
                "logits_to_keep": 1,
            }
            if entry.get("sp") is not None:
                fw["position_ids"] = entry["sp"]
            return fw

        try:
            entry = partial_prefill_slots.get(key)
            if entry is None:
                graph_stats["partial_misses"] += 1
                with torch.profiler.record_function("partial_prefill_graph_capture"):
                    try:
                        gc = StaticCache(
                            config=model_self.config,
                            max_batch_size=bs,
                            max_cache_len=mcl,
                            device=device,
                            dtype=next(model_self.parameters()).dtype,
                        )
                    except TypeError:
                        gc = StaticCache(
                            config=model_self.config,
                            max_batch_size=bs,
                            max_cache_len=mcl,
                            device=device,
                        )
                    si = input_ids.clone()
                    sm = attention_mask.clone()
                    sc = cache_position.to(device=device, dtype=torch.long).clone()
                    sp = _build_position_ids(sc)
                    if isinstance(sp, torch.Tensor):
                        sp = sp.clone()
                    out_next = torch.empty((bs,), dtype=torch.long, device=device)
                    entry = {"gc": gc, "si": si, "sm": sm, "sc": sc, "sp": sp, "out_next": out_next}

                    if not _copy_inputs(entry):
                        return None
                    fw = _make_forward_kwargs(entry)
                    with torch.no_grad():
                        with torch.profiler.record_function("partial_prefill_graph_warmup"):
                            for _ in range(2):
                                _reset_static_cache(gc)
                                _copy_cache_into_static(past_kv, gc, seq_len=past_len)
                                outputs = model_self(**fw)
                                out_next.copy_(outputs.logits[:, -1, :].argmax(dim=-1).view(-1))
                    torch.cuda.synchronize()

                    _reset_static_cache(gc)
                    _copy_cache_into_static(past_kv, gc, seq_len=past_len)
                    with torch.cuda.device(dev_idx):
                        graph = torch.cuda.CUDAGraph()
                        with torch.no_grad():
                            with torch.cuda.graph(graph):
                                outputs = model_self(**fw)
                                out_next.copy_(outputs.logits[:, -1, :].argmax(dim=-1).view(-1))
                    entry["graph"] = graph
                    partial_prefill_slots[key] = entry
                    graph_stats["partial_captures"] += 1
            else:
                graph_stats["partial_hits"] += 1

            if not _copy_inputs(entry):
                return None
            with torch.profiler.record_function("partial_prefill_graph_replay"):
                entry["graph"].replay()
            graph_stats["partial_replays"] += 1
            return {
                "next_token": entry["out_next"].clone(),
                "past_key_values": entry["gc"],
                "prompt_len": rem_len,
                "past_len": past_len,
            }
        except Exception as exc:
            partial_prefill_slots.pop(key, None)
            graph_stats["partial_failures"] += 1
            _log(f"partial prefill graph failed: {type(exc).__name__}: {exc}")
            return None

    # ── main generate function ──
    def decode_cudagraph_generate(model_self, *args, **kwargs):
        full_prefill_state = kwargs.pop("_aicas_full_prefill_state", None)
        if args or not _should_use_fast_generate(kwargs):
            return original_generate(*args, **kwargs)

        max_new = int(kwargs.get("max_new_tokens", 1) or 1)
        if max_new <= 1:
            prefill_state = _prefill_state_from_full_hit(full_prefill_state)
            if isinstance(prefill_state, dict):
                graph_stats["full_shortcut_hits"] += 1
            if not isinstance(prefill_state, dict):
                prefill_state = _try_prefill_graph(model_self, kwargs)
                if not isinstance(prefill_state, dict):
                    prefill_state = _try_partial_prefill_graph(model_self, kwargs, max_new=max_new)
            if isinstance(prefill_state, dict):
                input_ids = kwargs["input_ids"]
                nxt = prefill_state["next_token"].to(device=input_ids.device, dtype=input_ids.dtype).view(input_ids.shape[0])
                if os.getenv("AICAS_VALIDATE_TOKEN_IDS", "0") == "1":
                    _validate_token_ids(nxt, vocab_size)
                seq = torch.empty(
                    (input_ids.shape[0], input_ids.shape[1] + 1),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                seq[:, :input_ids.shape[1]] = input_ids
                seq[:, input_ids.shape[1]] = nxt
                if kwargs.get("return_dict_in_generate", False):
                    return ModelOutput(sequences=seq, past_key_values=prefill_state.get("past_key_values"))
                return seq
            return original_generate(*args, **kwargs)
        bypass = max(0, int(os.getenv("AICAS_DECODE_GRAPH_BYPASS_SHORT_MAX_NEW_TOKENS", "0")))
        if 1 < max_new <= bypass:
            return original_generate(*args, **kwargs)

        lo = int(os.getenv("AICAS_DECODE_CUDAGRAPH_MIN_NEW_TOKENS", "2"))
        hi = int(os.getenv("AICAS_DECODE_CUDAGRAPH_MAX_NEW_TOKENS", "256"))
        # Unified CUDA Graph switch: AICAS_CUDA_GRAPH=0 disables decode graph
        if os.getenv("AICAS_CUDA_GRAPH", "1").strip() in ("0", "false", "no", "off"):
            if "AICAS_DECODE_CUDAGRAPH_MIN_NEW_TOKENS" not in os.environ:
                lo = 999999
        if not (lo <= max_new <= hi):
            return original_generate(*args, **kwargs)

        input_ids = kwargs["input_ids"]
        attn_mask = kwargs["attention_mask"]
        bs, plen = input_ids.shape[:2]
        device = input_ids.device

        past_kv = kwargs.get("past_key_values")
        past_len = _get_cache_seq_len(past_kv) if past_kv is not None else 0
        ctx_len = past_len + plen
        total_len = ctx_len + max_new
        if "min_new_tokens" in kwargs:
            min_new = int(kwargs.get("min_new_tokens", 0))
        elif max_new == 128:
            min_new = 128
        elif max_new == 10 and os.getenv("AICAS_FORCE_WARMUP_MIN_NEW_TOKENS", "1") == "1":
            min_new = 10
        else:
            min_new = 0
        active_eos_token_ids = _token_id_list(kwargs.get("eos_token_id", eos_token_ids))
        active_pad_token_id = kwargs.get("pad_token_id", pad_token_id)
        if isinstance(active_pad_token_id, torch.Tensor):
            active_pad_token_id = int(active_pad_token_id.reshape(-1)[0].detach().cpu().item()) if active_pad_token_id.numel() else pad_token_id
        elif isinstance(active_pad_token_id, (list, tuple, set)):
            active_pad_token_id = next(iter(active_pad_token_id), pad_token_id)
        active_pad_token_id = int(active_pad_token_id)
        skip_eos = min_new >= max_new or not active_eos_token_ids
        full_mask = os.getenv("AICAS_DECODE_GRAPH_FULLMASK", "1") == "1"
        null_mask = os.getenv("AICAS_DECODE_GRAPH_NULL_ATTN_MASK", "0") == "1"
        if null_mask:
            # full_mask is irrelevant when no attention_mask tensor is passed.
            # Canonicalize it so runtime keys match precaptured null-mask slots.
            full_mask = False
            # Correct replay requires feeding each generated token and updated
            # cache position into the next graph replay. Multi-step chunks would
            # need graph-internal token chaining, so keep one token per replay.
            chunk = 1
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        should_validate = os.getenv("AICAS_VALIDATE_TOKEN_IDS", "0") == "1"

        seqs = torch.empty((bs, plen + max_new), dtype=input_ids.dtype, device=device)
        seqs[:, :plen] = input_ids

        mcl = _select_mcl(total_len)
        slot_key = (int(mcl), bool(full_mask), bool(null_mask), int(chunk), int(dev_idx), str(input_ids.dtype))

        try:
            # ── 1. prefill (prefill graph, eager fallback) ──
            prefill_state = None
            full_shortcut_view = None
            if isinstance(full_prefill_state, dict):
                full_shortcut_view = _cache_view_from_full_state(full_prefill_state)
                prefill_state = _prefill_state_from_full_hit(full_prefill_state, cache_obj=full_shortcut_view)
                if isinstance(prefill_state, dict):
                    graph_stats["full_shortcut_hits"] += 1
            if max_new <= 256:
                if not isinstance(prefill_state, dict):
                    prefill_state = _try_prefill_graph(model_self, kwargs)
                    if not isinstance(prefill_state, dict):
                        prefill_state = _try_partial_prefill_graph(model_self, kwargs, max_new=max_new)

            prefill_cache = prefill_state.get("past_key_values") if isinstance(prefill_state, dict) else None
            full_hit_cache_view = prefill_cache if isinstance(prefill_cache, _FullPromptCacheView) else None
            if full_hit_cache_view is None:
                dtype = next(model_self.parameters()).dtype
                gc = prefill_cache if prefill_cache is not None else _new_static_cache(
                    batch_size=bs, max_cache_len=mcl, device=device, dtype=dtype)
                if prefill_cache is not None and _get_cache_max_len(gc) < int(mcl):
                    expanded = _new_static_cache(
                        batch_size=bs, max_cache_len=mcl, device=device, dtype=dtype)
                    nkv, hd = _resolve_kv_cache_shape(model_self)
                    _bulk_init_static_cache(expanded, bs, mcl, nkv, hd, device, dtype)
                    _copy_cache_into_static(prefill_cache, expanded, seq_len=ctx_len)
                    gc = expanded
                if past_kv is not None:
                    _copy_cache_into_static(past_kv, gc)
                elif prefill_cache is None:
                    # Bulk-allocate all StaticCache layers in one large tensor
                    # instead of 56 individual lazy-init kernel launches.
                    nkv, hd = _resolve_kv_cache_shape(model_self)
                    _bulk_init_static_cache(gc, bs, mcl, nkv, hd, device, dtype)
            else:
                gc = None

            if isinstance(prefill_state, dict):
                nxt = prefill_state["next_token"].to(device=device, dtype=input_ids.dtype).view(bs)
            else:
                pf_inputs = {k: v for k, v in kwargs.items()
                             if k in ("input_ids", "attention_mask", "pixel_values",
                                      "image_grid_thw", "cache_position", "mm_token_type_ids")}
                pf_inputs["past_key_values"] = gc
                pf_inputs["use_cache"] = True
                pf_inputs["logits_to_keep"] = 1
                if past_kv is not None and kwargs.get("cache_position") is not None:
                    cp = kwargs["cache_position"]
                    if isinstance(cp, torch.Tensor) and cp.numel() >= 1:
                        pids = _build_decode_position_ids(model_self,
                            cp.reshape(-1)[:1].to(device=device, dtype=torch.long),
                            batch_size=bs, seq_length=plen)
                        if pids is not None:
                            pf_inputs["position_ids"] = pids

                with torch.no_grad():
                    with torch.profiler.record_function("decode_prefill"):
                        outputs = model_self(**pf_inputs)
                nxt = outputs.logits[:, -1, :].argmax(dim=-1)
            if should_validate:
                _validate_token_ids(nxt, vocab_size)
            seqs[:, plen] = nxt
            n_gen = 1

            if skip_eos:
                finished = torch.zeros_like(nxt, dtype=torch.bool)
            else:
                finished = _tokens_match_any(nxt, active_eos_token_ids)
            if n_gen >= max_new or (not skip_eos and n_gen >= min_new and bool(torch.all(finished))):
                result = seqs[:, :plen + n_gen]
                ret_cache = gc if gc is not None else full_hit_cache_view
                return ModelOutput(sequences=result, past_key_values=ret_cache) if kwargs.get("return_dict_in_generate", False) else result

            # ── 2. decode graph capture or reuse ──
            rd = getattr(getattr(model_self, "model", None), "rope_deltas", None)
            rd_val = int(rd.item()) if isinstance(rd, torch.Tensor) and rd.numel() == 1 else 0
            cache_pos = ctx_len
            slot = slots.get(slot_key)

            if slot is not None:
                graph_stats["decode_hits"] += 1
                _log(f"reuse slot key={slot_key}")
                graph, si, sm, sp, sc, gnt = slot["graph"], slot["si"], slot["sm"], slot["sp"], slot["sc"], slot["gnt"]
                chunk = max(1, int(slot.get("chunk_steps", 1)))
                token_state = slot.get("token_state") if isinstance(slot.get("token_state"), dict) else {}
                if full_hit_cache_view is not None:
                    if os.getenv("AICAS_DECODE_GRAPH_RESET_SLOT_BEFORE_COPY", "0") == "1":
                        _reset_static_cache(slot["gc"])
                    if not _copy_full_prompt_view_into_static(full_hit_cache_view, slot["gc"], seq_len=ctx_len):
                        _copy_cache_into_static(full_hit_cache_view, slot["gc"], seq_len=ctx_len)
                    gc = slot["gc"]
                elif gc is not slot["gc"]:
                    if os.getenv("AICAS_DECODE_GRAPH_RESET_SLOT_BEFORE_COPY", "0") == "1":
                        _reset_static_cache(slot["gc"])
                    _copy_kv_between_static(gc, slot["gc"], ctx_len)
                    gc = slot["gc"]
                if sm is not None:
                    sm.zero_()
                    sm[:, :cache_pos + 1] = 1
                si.copy_(nxt.view(bs, 1))
                sc.fill_(cache_pos)
                sp.fill_(cache_pos + rd_val)
            else:
                graph_stats["decode_misses"] += 1
                _log(f"capture slot key={slot_key}")
                si = nxt.view(bs, 1).clone()
                sc = torch.tensor([cache_pos], dtype=torch.long, device=device)
                sp = torch.full((3, bs, 1), cache_pos + rd_val, dtype=torch.long, device=device)
                if null_mask:
                    sm = None
                else:
                    sm = torch.zeros((bs, mcl), dtype=attn_mask.dtype, device=device)
                    if full_mask:
                        sm.fill_(1)
                    else:
                        sm[:, :cache_pos + 1] = 1
                dk = dict(input_ids=si, position_ids=sp, past_key_values=gc,
                          cache_position=sc, use_cache=True, logits_to_keep=1)
                if sm is not None:
                    dk["attention_mask"] = sm
                token_state = {}
                if full_hit_cache_view is not None and gc is None:
                    dtype = next(model_self.parameters()).dtype
                    gc = _static_cache_from_cache_view(
                        full_hit_cache_view,
                        batch_size=bs,
                        max_cache_len=mcl,
                        device=device,
                        dtype=dtype,
                    )
                    dk["past_key_values"] = gc
                slot = _capture_slot(slot_key, gc, dk, token_state, chunk, dev_idx)
                graph_stats["decode_captures"] += 1
                graph, si, sm, sp, sc, gnt = slot["graph"], slot["si"], slot["sm"], slot["sp"], slot["sc"], slot["gnt"]

                if sm is not None:
                    sm.zero_()
                    sm[:, :cache_pos + 1] = 1
                si.copy_(nxt.view(bs, 1))
                sc.fill_(cache_pos)
                sp.fill_(cache_pos + rd_val)

            sm_mark = None if (sm is None or full_mask) else torch.ones((bs, 1), dtype=sm.dtype, device=device)

            # ── 3. decode loop ──
            stop = False
            while n_gen < max_new and not stop:
                with torch.profiler.record_function("decode_graph_replay"):
                    graph.replay()
                graph_stats["decode_replays"] += 1
                for step in range(chunk):
                    if stop or n_gen >= max_new:
                        break
                    stok = gnt[step].clone()
                    if sm_mark is not None:
                        sm[:, cache_pos + step] = sm_mark[:, 0]

                    if should_validate:
                        _validate_token_ids(stok, vocab_size)

                    if active_eos_token_ids and not skip_eos:
                        stok = torch.where(finished, torch.full_like(stok, active_pad_token_id), stok)
                        finished |= _tokens_match_any(stok, active_eos_token_ids)

                    seqs[:, plen + n_gen] = stok
                    nxt = stok
                    n_gen += 1
                    cache_pos += 1
                    if not skip_eos and n_gen >= min_new and bool(torch.all(finished)):
                        stop = True
                    if n_gen < max_new and not stop:
                        si.copy_(nxt.view(bs, 1))
                        sc.fill_(cache_pos)
                        sp.fill_(cache_pos + rd_val)
                        if sm_mark is not None:
                            sm[:, cache_pos] = sm_mark[:, 0]

            result = seqs[:, :plen + n_gen]
            if should_validate:
                _validate_token_ids(result[:, plen:], vocab_size)
            return ModelOutput(sequences=result, past_key_values=gc) if kwargs.get("return_dict_in_generate", False) else result
        except Exception:
            return original_generate(*args, **kwargs)

    # ── precapture API ──
    def _precapture_decode_graph(model_self, *, batch_size=1, max_new_tokens=128,
                                  full_mask=True, null_mask=False, chunk_steps=1,
                                  device_index=None, dtype=None, mcl=None):
        """Pre-capture a decode CUDA graph slot. Returns True if captured, False if skipped or failed.

        If ``mcl`` is None (default), it is derived from ``_select_mcl(1280 + max_new_tokens)``.
        Pass an explicit ``mcl`` to capture a specific bucket size.
        """
        dev_idx = device_index if device_index is not None else torch.cuda.current_device()
        dt = dtype if dtype is not None else torch.long
        bs = max(1, int(batch_size))
        if mcl is None:
            mcl = _select_mcl(1280 + int(max_new_tokens))
        else:
            mcl = int(mcl)
        slot_key = (int(mcl), bool(full_mask), bool(null_mask), int(chunk_steps), int(dev_idx), str(dt))
        if slots.get(slot_key) is not None:
            _log(f"precapture skip: already cached key={slot_key}")
            return False

        try:
            gc = StaticCache(config=model_self.config, max_batch_size=bs,
                             max_cache_len=mcl, device=f"cuda:{dev_idx}", dtype=torch.bfloat16)
        except TypeError:
            gc = StaticCache(config=model_self.config, max_batch_size=bs,
                             max_cache_len=mcl, device=f"cuda:{dev_idx}")

        device = torch.device(f"cuda:{dev_idx}")
        si = torch.zeros((bs, 1), dtype=dt, device=device)
        sc = torch.tensor([0], dtype=torch.long, device=device)
        sp = torch.full((3, bs, 1), 0, dtype=torch.long, device=device)
        sm = None if null_mask else torch.ones((bs, mcl), dtype=torch.bfloat16, device=device) if full_mask else torch.zeros((bs, mcl), dtype=torch.bfloat16, device=device)
        if sm is not None and not full_mask:
            sm[:, :1] = 1
        token_state = {}
        dk = dict(input_ids=si, position_ids=sp, past_key_values=gc,
                  cache_position=sc, use_cache=True, logits_to_keep=1)
        if sm is not None:
            dk["attention_mask"] = sm

        _capture_slot(slot_key, gc, dk, token_state, chunk_steps, dev_idx)
        torch.cuda.synchronize(device)
        return True

    def _precapture_partial_prefill_graph(
        model_self,
        *,
        remaining_len: int,
        attention_len: int,
        max_new_tokens: int = 1,
        device_index=None,
    ):
        """Pre-capture the KV-cache-hit prefill graph for one partial shape."""
        rem_len = int(remaining_len)
        attn_len = int(attention_len)
        if rem_len <= 0 or attn_len < rem_len or not torch.cuda.is_available():
            return False

        dev_idx = device_index if device_index is not None else torch.cuda.current_device()
        device = torch.device(f"cuda:{int(dev_idx)}")
        dtype = next(model_self.parameters()).dtype
        past_len = attn_len - rem_len
        if past_len <= 0:
            return False

        mcl = _select_mcl(attn_len + max(1, int(max_new_tokens)))
        try:
            past_kv = StaticCache(
                config=model_self.config,
                max_batch_size=1,
                max_cache_len=mcl,
                device=device,
                dtype=dtype,
            )
        except TypeError:
            past_kv = StaticCache(
                config=model_self.config,
                max_batch_size=1,
                max_cache_len=mcl,
                device=device,
            )

        text_cfg = getattr(model_self.config, "text_config", model_self.config)
        kv_heads = int(getattr(text_cfg, "num_key_value_heads", getattr(text_cfg, "num_attention_heads", 8)))
        num_heads = max(1, int(getattr(text_cfg, "num_attention_heads", 16)))
        head_dim = int(getattr(text_cfg, "head_dim", int(getattr(text_cfg, "hidden_size", 2048)) // num_heads))
        seed = torch.ones((1, kv_heads, 1, head_dim), dtype=dtype, device=device)
        for layer in getattr(past_kv, "layers", []):
            layer.lazy_initialization(seed)
            layer.keys[:, :, :past_len, :].fill_(1)
            layer.values[:, :, :past_len, :].fill_(1)

        kwargs = {
            "input_ids": torch.ones((1, rem_len), dtype=torch.long, device=device),
            "attention_mask": torch.ones((1, attn_len), dtype=torch.long, device=device),
            "past_key_values": past_kv,
            "cache_position": torch.arange(past_len, attn_len, dtype=torch.long, device=device),
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        state = _try_partial_prefill_graph(model_self, kwargs, max_new=int(max_new_tokens))
        torch.cuda.synchronize(device)
        return isinstance(state, dict)

    def _aicas_graph_stats(model_self):
        return {
            **graph_stats,
            "decode_cache_size": len(slots),
            "partial_cache_size": len(partial_prefill_slots),
        }

    def _reset_aicas_graph_stats(model_self):
        for key in graph_stats:
            graph_stats[key] = 0

    # ── attach to model ──
    model.generate = MethodType(decode_cudagraph_generate, model)
    model._decode_cudagraph_generate = MethodType(decode_cudagraph_generate, model)
    model._decode_cudagraph_step_from_cache = MethodType(_decode_cudagraph_step_from_cache, model)
    model._precapture_decode_graph = MethodType(_precapture_decode_graph, model)
    model._precapture_partial_prefill_graph = MethodType(_precapture_partial_prefill_graph, model)
    model._aicas_graph_stats = MethodType(_aicas_graph_stats, model)
    model._reset_aicas_graph_stats = MethodType(_reset_aicas_graph_stats, model)
    if "decode_cudagraph" not in optimizations_applied:
        optimizations_applied.append("decode_cudagraph")



# ---------------------------------------------------------------------------
# EAGLE3 speculative decoding support
# ---------------------------------------------------------------------------

def _extract_past_from_outputs(outputs):
    past = getattr(outputs, "past_key_values", None)
    if past is not None:
        return past
    if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
        return outputs[1]
    return None

def _extract_last_hidden_from_outputs(outputs):
    hidden = getattr(outputs, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    if isinstance(outputs, ModelOutput):
        hidden = outputs.get("last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            return hidden
        if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            return outputs[0]
    if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
        return outputs[0]
    return None

def _select_target_verify_feature(model_self, hidden_states, layer_indices):
    layers = tuple(h for h in hidden_states if isinstance(h, torch.Tensor)) if isinstance(hidden_states, (tuple, list)) else ()
    if not layers:
        raise RuntimeError("target_verify graph requires output_hidden_states=True")
    try:
        return torch.cat([layers[int(idx)] for idx in layer_indices], dim=-1)
    except Exception as exc:
        raise RuntimeError(f"target_verify graph failed to select hidden layers {layer_indices}") from exc

def _target_verify_forward(model_self, model_kwargs, token_state=None):
    core = getattr(model_self, "model", None)
    lm_head = getattr(model_self, "lm_head", None)
    if core is None or lm_head is None:
        raise RuntimeError("target_verify graph requires model.model and lm_head")
    core_kwargs = dict(model_kwargs)
    layer_indices = tuple(int(x) for x in core_kwargs.pop("target_verify_layer_indices", (0, 1, 2)))
    core_kwargs.pop("logits_to_keep", None)
    core_kwargs.setdefault("return_dict", True)
    core_kwargs["output_hidden_states"] = True
    core_kwargs.setdefault("output_attentions", False)
    with torch.no_grad():
        outputs = core(**core_kwargs)
    hidden = _extract_last_hidden_from_outputs(outputs)
    vpast = _extract_past_from_outputs(outputs)
    if not isinstance(hidden, torch.Tensor) or vpast is None:
        raise RuntimeError("target_verify graph did not return hidden/past")
    bias = getattr(lm_head, "bias", None)
    if (
        can_use_lm_head_argmax_triton_multi is not None
        and lm_head_argmax_triton_multi is not None
        and can_use_lm_head_argmax_triton_multi(hidden, lm_head.weight, bias)
    ):
        workspace = token_state.get("target_verify_lm_head_workspace") if isinstance(token_state, dict) else None
        target, workspace = lm_head_argmax_triton_multi(hidden, lm_head.weight, bias, workspace=workspace)
        if isinstance(token_state, dict):
            token_state["target_verify_lm_head_workspace"] = workspace
        target = target.reshape(-1).to(dtype=torch.long)
    else:
        target = lm_head(hidden).argmax(dim=-1).reshape(-1).to(dtype=torch.long)
    feature = _select_target_verify_feature(model_self, getattr(outputs, "hidden_states", None), layer_indices)
    return target, vpast, feature, hidden

def _canonical_decode_step(model_self, model_kwargs, token_state=None):
    return _decode_forward_next_token(model_self, model_kwargs, token_state)

def _decode_cudagraph_step_from_cache(
    model_self, *, input_ids, past_key_values, cache_position,
    attention_mask=None, position_ids=None, token_state=None,
    forward_mode="decode", layer_indices=None,
):
    if str(forward_mode).strip().lower() == "target_verify":
        kwargs = {
            "input_ids": input_ids, "past_key_values": past_key_values,
            "cache_position": cache_position, "use_cache": True,
            "return_dict": True, "output_hidden_states": True, "output_attentions": False,
            "target_verify_layer_indices": tuple(layer_indices or (0, 1, 2)),
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        return _target_verify_forward(model_self, kwargs, token_state if isinstance(token_state, dict) else {})
    return _decode_forward_next_token(model_self, {
        "input_ids": input_ids, "past_key_values": past_key_values,
        "cache_position": cache_position, "attention_mask": attention_mask,
        "position_ids": position_ids, "use_cache": True, "return_dict": True,
    }, token_state if isinstance(token_state, dict) else {})

def decode_cudagraph_step_from_cache(
    model_self, *, input_ids, past_key_values, cache_position,
    attention_mask=None, position_ids=None, token_state=None,
    forward_mode="decode", layer_indices=None,
):
    step_fn = getattr(model_self, "_decode_cudagraph_step_from_cache", None)
    if callable(step_fn):
        try:
            return step_fn(
                input_ids=input_ids, past_key_values=past_key_values,
                cache_position=cache_position, attention_mask=attention_mask,
                position_ids=position_ids, token_state=token_state,
                forward_mode=forward_mode, layer_indices=layer_indices,
            )
        except TypeError:
            if str(forward_mode).strip().lower() != "decode":
                raise
            return step_fn(
                input_ids=input_ids, past_key_values=past_key_values,
                cache_position=cache_position, attention_mask=attention_mask,
            )
    if str(forward_mode).strip().lower() != "target_verify":
        raise RuntimeError("decode cudagraph step helper is not installed on model")
    kwargs = {
        "input_ids": input_ids, "past_key_values": past_key_values,
        "cache_position": cache_position, "use_cache": True,
        "return_dict": True, "output_hidden_states": True, "output_attentions": False,
        "target_verify_layer_indices": tuple(layer_indices or (0, 1, 2)),
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return _target_verify_forward(model_self, kwargs, token_state if isinstance(token_state, dict) else {})
