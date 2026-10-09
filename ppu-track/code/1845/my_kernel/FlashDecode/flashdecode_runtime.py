#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch

_THIS_DIR = Path(__file__).resolve().parent
_FLASHDECODE_RUNTIME_ENABLED = True


def set_flashdecode_runtime_enabled(enabled: bool) -> None:
    global _FLASHDECODE_RUNTIME_ENABLED
    _FLASHDECODE_RUNTIME_ENABLED = bool(enabled)


def _load_flashdecode_ext():
    so_path = _THIS_DIR / "flashdecode_ext.so"
    if not so_path.is_file():
        cands = sorted(_THIS_DIR.glob("flashdecode_ext*.so"))
        if not cands:
            raise FileNotFoundError(f"Missing flashdecode extension in {_THIS_DIR}")
        so_path = cands[-1]

    module_name = "flashdecode_ext"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load extension spec from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_model_device(model: torch.nn.Module) -> torch.device:
    dev = getattr(model, "device", None)
    if isinstance(dev, torch.device) and dev.type == "cuda":
        return dev
    for p in model.parameters():
        if p.is_cuda:
            return p.device
    raise RuntimeError("FlashDecode requires a CUDA model device")


def patch_qwen3vl_flashdecode(
    model: torch.nn.Module,
    *,
    min_seq: int = 595,
    max_seq: int = 2067,
    max_S: int = 2112,
) -> Dict[str, int]:
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl_modeling

    if getattr(model, "_flashdecode_attention_installed", False):
        stats = getattr(model, "_flashdecode_stats", None)
        if isinstance(stats, dict):
            return stats
        return {"hit": 0, "decode_fallback": 0, "non_decode": 0, "runtime_disabled": 0}

    ext_mod = _load_flashdecode_ext()
    runner_device = _resolve_model_device(model)
    runner_device_index = runner_device.index
    if runner_device_index is None:
        runner_device_index = torch.cuda.current_device()

    stats = {"hit": 0, "decode_fallback": 0, "non_decode": 0, "runtime_disabled": 0}
    cache = {
        "runner": ext_mod.FlashDecodeRunner(1, 8, 2, max_S, runner_device_index),
        "device": runner_device,
        "attn_out": {},
        "logged_hit": False,
        "logged_fallback": False,
    }

    impl_name = "flashdecode_sdpa_submit"
    original_impl = getattr(model.config, "_attn_implementation", "sdpa")
    if original_impl == impl_name:
        original_impl = "sdpa"
    fallback_attention = qwen3_vl_modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
        original_impl,
        qwen3_vl_modeling.eager_attention_forward,
    )
    expected_scaling = 1.0 / math.sqrt(128.0)

    def _get_attn_out(dtype: torch.dtype) -> torch.Tensor:
        out = cache["attn_out"].get(dtype)
        if out is None:
            out = torch.empty((1, 1, 16, 128), device=cache["device"], dtype=dtype)
            cache["attn_out"][dtype] = out
        return out

    def _get_multi_out(dtype: torch.dtype, q_len: int) -> torch.Tensor:
        key = ("multi", q_len, dtype)
        out = cache["attn_out"].get(key)
        if out is None:
            out = torch.empty((1, q_len, 16, 128), device=cache["device"], dtype=dtype)
            cache["attn_out"][key] = out
        return out

    def _capture_cache_position(attn_module: torch.nn.Module) -> None:
        if getattr(attn_module, "_flashdecode_cache_position_capture_installed", False):
            return

        def _pre_hook(module, args, kwargs=None):
            kwargs = kwargs or {}
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None and len(args) >= 5:
                cache_position = args[4]
            module._flashdecode_cache_position = cache_position

        try:
            handle = attn_module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        except TypeError:
            def _pre_hook_no_kwargs(module, args):
                cache_position = args[4] if len(args) >= 5 else None
                module._flashdecode_cache_position = cache_position

            handle = attn_module.register_forward_pre_hook(_pre_hook_no_kwargs)
        attn_module._flashdecode_cache_position_hook = handle
        attn_module._flashdecode_cache_position_capture_installed = True

    def _install_cache_position_capture() -> None:
        for attn_module in model.modules():
            if (
                hasattr(attn_module, "q_proj")
                and hasattr(attn_module, "k_proj")
                and hasattr(attn_module, "v_proj")
                and hasattr(attn_module, "o_proj")
            ):
                _capture_cache_position(attn_module)

    def _cache_visible_len(module, q_len: int, seq_kv: int) -> Optional[int]:
        cache_position = getattr(module, "_flashdecode_cache_position", None)
        if not isinstance(cache_position, torch.Tensor) or cache_position.numel() < 1:
            return None
        try:
            positions = cache_position.reshape(-1)
            pos = positions[-1] if q_len == 1 else positions[0]
            visible = int(pos.item()) + 1
        except Exception:
            return None
        return max(1, min(int(visible), int(seq_kv)))

    def _mask_visible_len(attention_mask, q_index: int = 0) -> Optional[int]:
        if not isinstance(attention_mask, torch.Tensor):
            return None
        try:
            if attention_mask.ndim == 4:
                row = attention_mask[0, 0, q_index, :]
            elif attention_mask.ndim == 2:
                row = attention_mask[0, :]
            else:
                return None
            if row.dtype == torch.bool:
                return int(row.sum().item())
            if torch.is_floating_point(row):
                return int((row > torch.finfo(row.dtype).min).sum().item())
            return int(row.gt(0).sum().item())
        except Exception:
            return None

    def _cache_pos_desc(module) -> str:
        cache_pos = getattr(module, "_flashdecode_cache_position", None)
        if isinstance(cache_pos, torch.Tensor) and cache_pos.numel() > 0:
            try:
                return str(int(cache_pos.reshape(-1)[-1].item()))
            except Exception:
                return f"shape={tuple(cache_pos.shape)}"
        return "none"

    def _debug_l0(module) -> bool:
        return os.getenv("AICAS_FLASHDECODE_DEBUG", "0") == "1" and int(getattr(module, "layer_idx", -1)) == 0

    def _fallback_with_aligned_mask(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        mask = attention_mask
        key_fb = key
        value_fb = value
        if isinstance(mask, torch.Tensor) and mask.ndim == 4:
            mask_kv = int(mask.shape[-1])
            seq = int(key.shape[-2])
            if mask_kv < seq:
                key_fb = key[:, :, :mask_kv, :].contiguous()
                value_fb = value[:, :, :mask_kv, :].contiguous()
            elif mask_kv > seq:
                mask = mask[..., :seq].contiguous()
        return fallback_attention(
            module,
            query,
            key_fb,
            value_fb,
            mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )

    def _eager_multi_attention_with_aligned_mask(query, key, value, attention_mask, scaling=None):
        mask = attention_mask
        key_fb = key
        value_fb = value
        if isinstance(mask, torch.Tensor) and mask.ndim == 4:
            mask_kv = int(mask.shape[-1])
            seq = int(key.shape[-2])
            if mask_kv < seq:
                key_fb = key[:, :, :mask_kv, :].contiguous()
                value_fb = value[:, :, :mask_kv, :].contiguous()
            elif mask_kv > seq:
                mask = mask[..., :seq].contiguous()
        groups = max(1, int(query.shape[1]) // max(1, int(key_fb.shape[1])))
        key_rep = key_fb.repeat_interleave(groups, dim=1)
        value_rep = value_fb.repeat_interleave(groups, dim=1)
        scale = float(scaling) if scaling is not None else expected_scaling
        scores = torch.matmul(query.float(), key_rep.float().transpose(-1, -2)) * scale
        if isinstance(mask, torch.Tensor):
            scores = scores + mask.float()
        probs = torch.softmax(scores, dim=-1).to(dtype=value_rep.dtype)
        out = torch.matmul(probs, value_rep).transpose(1, 2).contiguous()
        return out, None

    def flashdecode_sdpa(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=0.0,
        scaling=None,
        **kwargs,
    ):
        q_dtype = query.dtype
        q_len = int(query.shape[-2])
        if not cache.get("very_first_call_v2"):
            print(f"[flashdecode] FIRST CALL: q_len={q_len} mask={'none' if attention_mask is None else 'set'}", flush=True)
            cache["very_first_call_v2"] = True
        scale_ok = (scaling is None) or (abs(float(scaling) - expected_scaling) < 1e-7)
        seq_kv = int(key.shape[-2])

        if not _FLASHDECODE_RUNTIME_ENABLED:
            if q_len == 1: stats["decode_fallback"] += 1
            else: stats["non_decode"] += 1
            return fallback_attention(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)

        # Common preconditions for FlashDecode (both single and multi-query)
        _fd_ok = (
            query.is_cuda and key.is_cuda and value.is_cuda
            and q_dtype in (torch.float16, torch.bfloat16)
            and key.dtype == q_dtype and value.dtype == q_dtype
            and query.shape[0] == 1 and query.shape[1] == 16
            and query.shape[-1] == 128
            and key.shape[1] == 8 and key.shape[-1] == 128
            and value.shape[1] == 8 and value.shape[-1] == 128
            and key.shape[-2] == value.shape[-2]
            and min_seq <= seq_kv <= max_seq and seq_kv <= max_S
            and key.is_contiguous() and value.is_contiguous()
            and not kwargs.get("output_attentions", False)
            and query.device == cache["device"]
            and scale_ok
        )

        # Single-query decode (q_len=1).
        # - attention_mask=None: full K/V, causal assumed by kernel
        # - attention_mask present: truncate K/V to visible range so
        #   run_decode_into uses the same kernel as the tree root-split path,
        #   keeping BF16 numerics consistent across decode and verification.
        if q_len == 1 and _fd_ok:
            query_fd = query if query.is_contiguous() else query.contiguous()
            cache_visible = _cache_visible_len(module, q_len, seq_kv)
            mask_visible = _mask_visible_len(attention_mask, 0)
            if mask_visible is None:
                kv_visible = cache_visible if cache_visible is not None else int(key.shape[-2])
            elif cache_visible is not None:
                kv_visible = min(int(mask_visible), int(cache_visible))
            else:
                kv_visible = int(mask_visible)
            kv_visible = max(1, min(int(kv_visible), int(key.shape[-2])))
            if kv_visible == int(key.shape[-2]):
                key_fd = key
                value_fd = value
            else:
                key_fd = key[:, :, :kv_visible, :].contiguous()
                value_fd = value[:, :, :kv_visible, :].contiguous()
            out = _get_attn_out(q_dtype)
            if _debug_l0(module) or not cache.get("logged_q1_hit_v2"):
                mask_desc = "none" if attention_mask is None else f"ndim={attention_mask.ndim} shape={tuple(attention_mask.shape)} dtype={attention_mask.dtype}"
                kv_vis_desc = "full" if kv_visible == seq_kv else kv_visible
                print(
                    f"[flashdecode] q1 HIT v2: layer={getattr(module, 'layer_idx', 'na')} mask={mask_desc} kv_full={seq_kv} kv_vis={kv_vis_desc} cache_pos={_cache_pos_desc(module)}",
                    flush=True,
                )
                cache["logged_q1_hit_v2"] = True
            cache["runner"].run_decode_into(query_fd, key_fd, value_fd, out)
            stats["hit"] += 1
            if not cache["logged_hit"]:
                print(f"[flashdecode] first decode hit: dtype={str(q_dtype).replace('torch.', '')}, seq_kv={seq_kv}")
                cache["logged_hit"] = True
            return out, None

        # Multi-query decode (1 < q_len <= 16): split the two semantics at the
        # kernel boundary. No mask uses the causal virtual_q kernel; explicit
        # masks use the tree/masked multi kernel.
        if 1 < q_len <= 16 and _fd_ok:
            if q_len == 15 and os.getenv("AICAS_FLASHDECODE_ALLOW_Q15_MULTI", "0") != "1":
                stats["non_decode"] += 1
                if not cache.get("logged_q15_multi_fallback"):
                    print(
                        f"[flashdecode] q15 multi fallback: dtype={str(q_dtype).replace('torch.', '')}, seq_kv={seq_kv}",
                        flush=True,
                    )
                    cache["logged_q15_multi_fallback"] = True
                return _eager_multi_attention_with_aligned_mask(
                    query,
                    key,
                    value,
                    attention_mask,
                    scaling=scaling,
                )
            query_fd = query if query.is_contiguous() else query.contiguous()
            multi_out = _get_multi_out(q_dtype, q_len)
            if attention_mask is None:
                cache["runner"].run_decode_multi_into(query_fd, key, value, multi_out)
            elif hasattr(cache["runner"], "run_decode_multi_masked_into"):
                mask_kv = int(attention_mask.shape[-1])
                mask_fd = attention_mask
                if mask_kv < seq_kv:
                    kv_slice = key[:, :, :mask_kv, :]
                    key_fd = kv_slice if kv_slice.is_contiguous() else kv_slice.contiguous()
                    vv_slice = value[:, :, :mask_kv, :]
                    value_fd = vv_slice if vv_slice.is_contiguous() else vv_slice.contiguous()
                else:
                    key_fd = key
                    value_fd = value
                    if mask_kv > seq_kv:
                        mask_fd = attention_mask[..., :seq_kv]
                if mask_fd.dtype != q_dtype:
                    mask_fd = mask_fd.to(dtype=q_dtype)
                if not mask_fd.is_contiguous():
                    mask_fd = mask_fd.contiguous()
                _root_visible = _mask_visible_len(mask_fd, 0)
                root_KV_limit = int(_root_visible) if _root_visible is not None else int(key_fd.shape[-2])
                root_KV_limit = max(1, min(root_KV_limit, int(key_fd.shape[-2])))
                # When root_KV_limit covers the full key_fd, reuse it directly.
                if root_KV_limit == int(key_fd.shape[-2]):
                    root_K = key_fd
                    root_V = value_fd
                else:
                    rk_slice = key[:, :, :root_KV_limit, :]
                    root_K = rk_slice if rk_slice.is_contiguous() else rk_slice.contiguous()
                    rv_slice = value[:, :, :root_KV_limit, :]
                    root_V = rv_slice if rv_slice.is_contiguous() else rv_slice.contiguous()
                root_Q = query_fd[:, :, 0:1, :].contiguous()
                root_out = _get_attn_out(q_dtype)
                if _debug_l0(module):
                    mask_desc = f"ndim={attention_mask.ndim} shape={tuple(attention_mask.shape)} dtype={attention_mask.dtype}"
                    print(
                        f"[flashdecode] root HIT v2: layer={getattr(module, 'layer_idx', 'na')} q_len={q_len} mask={mask_desc} kv_full={seq_kv} root_vis={root_KV_limit} cache_pos={_cache_pos_desc(module)}",
                        flush=True,
                    )
                cache["runner"].run_decode_into(root_Q, root_K, root_V, root_out)
                if q_len > 1:
                    rest_Q = query_fd[:, :, 1:, :].contiguous()
                    rest_mask_slice = mask_fd[:, :, 1:, :]
                    rest_mask = rest_mask_slice if rest_mask_slice.is_contiguous() else rest_mask_slice.contiguous()
                    rest_out = _get_multi_out(q_dtype, q_len - 1)
                    cache["runner"].run_decode_multi_masked_into(rest_Q, key_fd, value_fd, rest_mask, rest_out)
                    # Single cat kernel launch instead of two copy_ calls.
                    torch.cat([root_out, rest_out], dim=1, out=multi_out)
                else:
                    multi_out.copy_(root_out.view(1, 1, 16, 128))
            else:
                return fallback_attention(module, query, key, value, attention_mask,
                                          dropout=dropout, scaling=scaling, **kwargs)
            stats["hit"] += 1
            if not cache.get("logged_multi_hit"):
                print(f"[flashdecode] first multi decode hit: q_len={q_len} seq_kv={seq_kv}", flush=True)
                cache["logged_multi_hit"] = True
            return multi_out, None

        if 1 < q_len <= 16 and not _fd_ok and not cache.get("logged_multi_miss"):
            _miss = []
            if not (query.is_cuda and key.is_cuda and value.is_cuda): _miss.append("cuda")
            if not (q_dtype in (torch.float16, torch.bfloat16)): _miss.append(f"dtype={q_dtype}")
            if not (query.shape[0] == 1 and query.shape[1] == 16): _miss.append(f"q_batch_or_heads={tuple(query.shape[:2])}")
            if not (query.shape[-1] == 128): _miss.append(f"q_dim={query.shape[-1]}")
            if not (key.shape[1] == 8 and key.shape[-1] == 128): _miss.append(f"k_shape={tuple(key.shape)}")
            if not (value.shape[1] == 8 and value.shape[-1] == 128): _miss.append(f"v_shape={tuple(value.shape)}")
            if not (key.shape[-2] == value.shape[-2]): _miss.append("kv_len_mismatch")
            if not (min_seq <= seq_kv <= max_seq): _miss.append(f"seq_kv={seq_kv}_notin[{min_seq},{max_seq}]")
            if not (seq_kv <= max_S): _miss.append(f"seq_kv>{max_S}")
            if not key.is_contiguous(): _miss.append("k_not_contig")
            if not value.is_contiguous(): _miss.append("v_not_contig")
            if not scale_ok: _miss.append("scale")
            if kwargs.get("output_attentions", False): _miss.append("output_attn")
            if not (query.device == cache["device"]): _miss.append("device")
            print(f"[flashdecode] multi miss: q_len={q_len} seq_kv={seq_kv} why={_miss}", flush=True)
            cache["logged_multi_miss"] = True

        # Fallback
        if q_len == 1: stats["decode_fallback"] += 1
        else: stats["non_decode"] += 1
        if not cache["logged_fallback"]:
            print(f"[flashdecode] first decode fallback: dtype={str(q_dtype).replace('torch.', '')}, seq_kv={seq_kv}, q_len={q_len}, mask={'none' if attention_mask is None else 'set'}")
            cache["logged_fallback"] = True
        return fallback_attention(module, query, key, value, attention_mask,
                                  dropout=dropout, scaling=scaling, **kwargs)

    try:
        import torch._dynamo as torch_dynamo

        flashdecode_entry = torch_dynamo.disable(flashdecode_sdpa)
    except Exception:
        flashdecode_entry = flashdecode_sdpa

    qwen3_vl_modeling.ALL_ATTENTION_FUNCTIONS.register(impl_name, flashdecode_entry)
    model.config._attn_implementation = impl_name
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config._attn_implementation = impl_name
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "config"):
        model.model.language_model.config._attn_implementation = impl_name
    _install_cache_position_capture()

    setattr(model, "_flashdecode_attention_installed", True)
    setattr(model, "_flashdecode_stats", stats)
    return stats


def get_flashdecode_stats(model: torch.nn.Module) -> Optional[Dict[str, int]]:
    stats = getattr(model, "_flashdecode_stats", None)
    if isinstance(stats, dict):
        return stats
    return None
