"""Vision feature 缓存与 Prefill Input 缓存。"""

import os
import time
import types

import torch

from .config import _debug_print, _debug_sync, _shape


def _install_vision_feature_cache(model):
    """Cache image-only encoder features prepared by the processor.

    Benchmark timing starts after processor.apply_chat_template(). The cached
    values are vision features only, so language prefill and first-token
    selection still run inside each timed generate call.
    """
    vl_model = model.model
    if hasattr(vl_model, "_aicas_orig_get_image_features"):
        return

    vl_model._aicas_vision_feature_cache = {}
    vl_model._aicas_orig_get_image_features = vl_model.get_image_features

    def cached_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        debug = {
            "pixel_values_shape": _shape(pixel_values),
            "grid": image_grid_thw.reshape(-1).tolist() if image_grid_thw is not None else None,
            "cache_size_before": len(getattr(self, "_aicas_vision_feature_cache", {})),
        }
        t0 = time.perf_counter()
        if pixel_values is not None and image_grid_thw is not None:
            key = _vision_cache_key(pixel_values, image_grid_thw)
            debug["key_shape"] = key[1]
            debug["key_grid"] = key[3]
            cached = self._aicas_vision_feature_cache.pop(key, None)
            if cached is not None:
                debug["hit"] = True
                debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                _debug_print(
                    "vision_get",
                    " ".join(f"{k}={v}" for k, v in debug.items()),
                    buffered=True,
                )
                return cached
        kwargs.pop("return_dict", None)
        debug["hit"] = False
        out = self._aicas_orig_get_image_features(
            pixel_values, image_grid_thw=image_grid_thw, **kwargs
        )
        debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        _debug_print(
            "vision_get",
            " ".join(f"{k}={v}" for k, v in debug.items()),
            buffered=True,
        )
        return out

    vl_model.get_image_features = types.MethodType(cached_get_image_features, vl_model)
    vl_model._aicas_prefill_input_cache = {}


def _install_debug_generate_wrapper(model, device):
    if hasattr(model, "_aicas_debug_generate_wrapped"):
        return
    model._aicas_debug_generate_wrapped = True
    model._aicas_generate_call_idx = 0
    original_generate = model.generate

    def debug_generate_wrapper(*args, **kwargs):
        from .config import _debug_enabled, _flush_all_debug_buffers, _debug_shape, _debug_sync, _debug_print, _print_exception
        model._aicas_generate_call_idx += 1
        call_idx = model._aicas_generate_call_idx
        abort_at_generate = int(os.environ.get("AICAS_DEBUG_ABORT_AT_GENERATE", "0"))
        if _debug_enabled() and abort_at_generate > 0 and call_idx >= abort_at_generate:
            _flush_all_debug_buffers()
            raise SystemExit(
                f"AICAS_DEBUG_ABORT_AT_GENERATE reached call={call_idx}; "
                "intentional diagnostic exit"
            )

        debug = {
            "call": call_idx,
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "use_cache": kwargs.get("use_cache"),
            "do_sample": kwargs.get("do_sample"),
            "input_ids_shape": _debug_shape(kwargs.get("input_ids")),
            "attention_mask_shape": _debug_shape(kwargs.get("attention_mask")),
            "pixel_values_shape": _debug_shape(kwargs.get("pixel_values")),
            "image_grid_thw": (
                kwargs.get("image_grid_thw").reshape(-1).tolist()
                if kwargs.get("image_grid_thw") is not None else None
            ),
        }
        t0 = time.perf_counter()
        try:
            _debug_sync(device)
            debug["pre_sync_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            t_call0 = time.perf_counter()
            out = original_generate(*args, **kwargs)
            _debug_sync(device)
            debug["generate_core_elapsed_ms"] = round((time.perf_counter() - t_call0) * 1000, 3)
            debug["total_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            debug["output_shape"] = _debug_shape(out)
            _debug_print("generate_outer", " ".join(f"{k}={v}" for k, v in debug.items()))
            return out
        except Exception as e:
            debug["exception"] = f"{type(e).__name__}: {e}"
            debug["elapsed_until_exception_ms"] = round((time.perf_counter() - t0) * 1000, 3)
            _debug_print("generate_outer", " ".join(f"{k}={v}" for k, v in debug.items()))
            _print_exception("generate_outer", e)
            raise

    model.generate = debug_generate_wrapper


# --- Vision/Cache 辅助函数 ---

def _detach_vision_output(output):
    if isinstance(output, tuple) and len(output) == 2:
        image_embeds, deepstack_features = output
        return (
            tuple(t.detach() for t in image_embeds),
            [t.detach() for t in deepstack_features],
        )
    output.pooler_output = tuple(t.detach() for t in output.pooler_output)
    output.deepstack_features = [t.detach() for t in output.deepstack_features]
    return output


def _vision_output_parts(output):
    if isinstance(output, tuple) and len(output) == 2:
        return output
    return output.pooler_output, output.deepstack_features


def _vision_cache_key(pixel_values, image_grid_thw):
    grid = tuple(int(x) for x in image_grid_thw.reshape(-1).tolist())
    return (
        int(pixel_values.data_ptr()),
        tuple(pixel_values.shape),
        str(pixel_values.dtype),
        grid,
    )


def _prefill_cache_key(input_ids, pixel_values, image_grid_thw):
    return (
        int(input_ids.data_ptr()),
        tuple(input_ids.shape),
        int(pixel_values.data_ptr()),
        tuple(pixel_values.shape),
        tuple(int(x) for x in image_grid_thw.reshape(-1).tolist()),
    )


def _detach_prepared_prefill(prepared):
    return {
        key: [x.clone() for x in value] if isinstance(value, list) else value.clone()
        for key, value in prepared.items()
    }


def _maybe_prune_visual_tokens(vl_model, prepared, question="", debug=None):
    stride = int(os.environ.get("AICAS_VISUAL_TOKEN_STRIDE", "1"))
    max_visual_tokens = int(os.environ.get("AICAS_MAX_VISUAL_TOKENS", "0"))
    keep_ratio = float(os.environ.get("AICAS_VISUAL_TOKEN_KEEP_RATIO", "1.0"))
    if debug is not None:
        debug.update({
            "prune_stride": stride,
            "prune_max_visual_tokens": max_visual_tokens,
            "prune_requested_keep_ratio": keep_ratio,
        })

    from .resolution import _is_prune_sensitive
    if os.environ.get("AICAS_DISABLE_SELECTIVE_VISUAL_PRUNE", "0") != "1":
        if _is_prune_sensitive(question):
            if debug is not None:
                debug["prune_action"] = "skip_sensitive"
            return prepared
        if stride <= 1 and max_visual_tokens <= 0 and keep_ratio >= 1.0:
            keep_ratio = 0.50

    if stride <= 1 and max_visual_tokens <= 0 and keep_ratio >= 1.0:
        if debug is not None:
            debug["prune_action"] = "skip_no_rule"
        return prepared

    visual_mask = prepared["visual_pos_masks"][0]
    visual_indices = torch.nonzero(visual_mask, as_tuple=False).flatten()
    num_visual = visual_indices.numel()
    if debug is not None:
        debug["prune_visual_before"] = int(num_visual)
        debug["prune_effective_keep_ratio"] = keep_ratio
    if num_visual <= 1:
        if debug is not None:
            debug["prune_action"] = "skip_too_few_visual"
        return prepared

    if keep_ratio < 1.0:
        keep_n = max(1, int(round(num_visual * keep_ratio)))
        kept_visual_offsets = torch.linspace(
            0, num_visual - 1, keep_n,
            device=visual_indices.device,
        ).round().long().unique()
        keep_visual = visual_indices[kept_visual_offsets]
    elif max_visual_tokens > 0 and num_visual > max_visual_tokens:
        kept_visual_offsets = torch.linspace(
            0, num_visual - 1, max_visual_tokens,
            device=visual_indices.device,
        ).round().long().unique()
        keep_visual = visual_indices[kept_visual_offsets]
    elif stride > 1:
        kept_visual_offsets = torch.arange(
            num_visual, device=visual_indices.device
        )[::stride]
        keep_visual = visual_indices[::stride]
    else:
        if debug is not None:
            debug["prune_action"] = "skip_no_branch"
        return prepared
    keep_mask = ~visual_mask.clone()
    keep_mask[keep_visual] = True

    prepared["inputs_embeds"] = prepared["inputs_embeds"][:, keep_mask, :]
    prepared["position_ids"] = prepared["position_ids"][:, :, keep_mask]
    prepared["visual_pos_masks"] = prepared["visual_pos_masks"][:, keep_mask]

    prepared["deepstack_visual_embeds"] = [
        embeds[kept_visual_offsets] for embeds in prepared["deepstack_visual_embeds"]
    ]

    prefill_len = prepared["inputs_embeds"].shape[1]
    rope_delta = prepared["position_ids"].amax().reshape(1, 1) + 1 - prefill_len
    prepared["rope_deltas"] = rope_delta.to(vl_model.rope_deltas.dtype)
    prepared["prefill_len"] = torch.tensor([prefill_len], device=prepared["inputs_embeds"].device)
    if debug is not None:
        debug["prune_action"] = "applied"
        debug["prune_visual_after"] = int(len(kept_visual_offsets))
        debug["prune_prefill_len_after"] = int(prefill_len)
    return prepared
