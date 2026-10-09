"""_PrewarmProcessor：增强的 Processor 封装，带特征缓存。"""

import os
import time

import torch

# Import all helpers from __init__ (where they live as module-level globals)
from . import (
    _debug_abort_at_sample, _debug_enabled, _debug_print, _debug_sync,
    _debug_shape, _shape, _flush_all_debug_buffers, _print_exception,
    _assign_size, _last_question_text,
    _select_image_pixels,
    _vision_cache_key, _prefill_cache_key, _detach_prepared_prefill,
    _detach_vision_output, _vision_output_parts, _maybe_prune_visual_tokens,
)


class _PrewarmProcessor:
    """Processor shim that primes graph_engine's prefill cache before timing."""

    def __init__(self, processor, model_getter, device, default_image_size=None):
        self._processor = processor
        self._model_getter = model_getter
        self._device = device
        self._default_image_size = dict(default_image_size or {})
        self._vision_cache_warned = False
        self._prewarm_warned = False
        self._call_idx = 0

    @property
    def tokenizer(self):
        return self._processor.tokenizer

    def apply_chat_template(self, messages, **kwargs):
        self._call_idx += 1
        call_idx = self._call_idx
        abort_at_sample = _debug_abort_at_sample()
        if abort_at_sample > 0 and call_idx >= abort_at_sample:
            _flush_all_debug_buffers()
            raise SystemExit(
                f"AICAS_DEBUG_ABORT_AT_SAMPLE reached call={call_idx}; "
                "intentional diagnostic exit"
            )
        t0 = time.perf_counter()
        question = ""
        try:
            for item in messages[0].get("content", []):
                if item.get("type") == "text":
                    question = str(item.get("text") or "")
                    break
        except Exception:
            pass
        import _eval_impl.config as _ew_cfg
        _ew_cfg._last_question_text = question

        image_size = None
        try:
            for item in messages[0].get("content", []):
                if item.get("type") == "image":
                    image = item.get("image")
                    image_size = getattr(image, "size", None)
                    break
        except Exception:
            pass

        image_processor = getattr(self._processor, "image_processor", None)
        old_size = None
        debug = {
            "call": call_idx,
            "return_tensors": kwargs.get("return_tensors"),
            "tokenize": kwargs.get("tokenize"),
            "question_head": question[:80].replace("\n", " "),
            "image_size": image_size,
            "processor_size_before": dict(getattr(image_processor, "size", {}) or {}) if image_processor is not None else None,
        }
        selected_pixels = _select_image_pixels(question, image_size)
        debug["selected_pixels"] = int(selected_pixels) if selected_pixels else None
        debug["router_high_res"] = selected_pixels is None
        if selected_pixels and image_processor is not None:
            old_size = dict(image_processor.size)
            image_processor.size["shortest_edge"] = int(selected_pixels)
            image_processor.size["longest_edge"] = int(selected_pixels)
        elif self._default_image_size and image_processor is not None:
            old_size = dict(image_processor.size)
            _assign_size(image_processor.size, self._default_image_size)
        debug["processor_size_used"] = dict(getattr(image_processor, "size", {}) or {}) if image_processor is not None else None
        try:
            t_template0 = time.perf_counter()
            result = self._processor.apply_chat_template(messages, **kwargs)
            _debug_sync(self._device)
            debug["template_core_elapsed_ms"] = round((time.perf_counter() - t_template0) * 1000, 3)
        finally:
            if old_size is not None:
                _assign_size(image_processor.size, old_size)
        debug["processor_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        _gpu_inputs = None
        if kwargs.get("return_tensors") and hasattr(result, "pixel_values"):
            try:
                debug["input_ids_shape_raw"] = _shape(getattr(result, "input_ids", None))
                debug["pixel_values_shape_raw"] = _shape(getattr(result, "pixel_values", None))
                debug["image_grid_thw_raw"] = (
                    getattr(result, "image_grid_thw", None).reshape(-1).tolist()
                    if getattr(result, "image_grid_thw", None) is not None else None
                )
                t_to0 = time.perf_counter()
                inputs = result.to(self._device)
                _gpu_inputs = inputs
                _debug_sync(self._device)
                debug["to_device_elapsed_ms"] = round((time.perf_counter() - t_to0) * 1000, 3)
                model = self._model_getter()
                vl_model = model.model
                pixel_values = getattr(inputs, "pixel_values", None)
                image_grid_thw = getattr(inputs, "image_grid_thw", None)
                input_ids = getattr(inputs, "input_ids", None)
                mm_token_type_ids = getattr(inputs, "mm_token_type_ids", None)
                attention_mask = getattr(inputs, "attention_mask", None)
                if pixel_values is not None and image_grid_thw is not None:
                    key = _vision_cache_key(pixel_values, image_grid_thw)
                    debug["vision_cache_key_shape"] = key[1]
                    debug["vision_cache_key_grid"] = key[3]
                    with torch.inference_mode():
                        t_vision0 = time.perf_counter()
                        output = vl_model._aicas_orig_get_image_features(
                            pixel_values,
                            image_grid_thw=image_grid_thw,
                        )
                        _debug_sync(self._device)
                        output = _detach_vision_output(output)
                        vl_model._aicas_vision_feature_cache[key] = output
                        debug["vision_feature_elapsed_ms"] = round((time.perf_counter() - t_vision0) * 1000, 3)
                        try:
                            image_features, deepstack_features = _vision_output_parts(output)
                            debug["image_feature_shapes"] = [_shape(x) for x in image_features]
                            debug["deepstack_shapes"] = [_shape(x) for x in deepstack_features]
                        except Exception:
                            debug["image_feature_shapes"] = str(type(output))

                        if (
                            os.environ.get("AICAS_ENABLE_PREPARED_PREFILL_CACHE", "0") == "1"
                            and input_ids is not None
                        ):
                            # Compute mm_token_type_ids if processor doesn't provide it
                            if mm_token_type_ids is None:
                                _cfg_m = getattr(vl_model.config, 'text_config', vl_model.config)
                                _img_tid = getattr(_cfg_m, 'image_token_id', None) or getattr(vl_model.config, 'image_token_id', 151655)
                                _vid_tid = getattr(_cfg_m, 'video_token_id', None) or getattr(vl_model.config, 'video_token_id', 151656)
                                mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
                                mm_token_type_ids[input_ids == _img_tid] = 1
                                mm_token_type_ids[input_ids == _vid_tid] = 2
                            t_prepare0 = time.perf_counter()
                            t_embed0 = time.perf_counter()
                            inputs_embeds = vl_model.get_input_embeddings()(input_ids)
                            _debug_sync(self._device)
                            debug["input_embed_elapsed_ms"] = round((time.perf_counter() - t_embed0) * 1000, 3)
                            image_features, deepstack_features = _vision_output_parts(output)
                            t_cat0 = time.perf_counter()
                            image_embeds = torch.cat(image_features, dim=0).to(
                                inputs_embeds.device, inputs_embeds.dtype
                            )
                            _debug_sync(self._device)
                            debug["image_cat_to_elapsed_ms"] = round((time.perf_counter() - t_cat0) * 1000, 3)
                            t_mask0 = time.perf_counter()
                            image_mask, _ = vl_model.get_placeholder_mask(
                                input_ids,
                                inputs_embeds=inputs_embeds,
                                image_features=image_embeds,
                            )
                            _debug_sync(self._device)
                            debug["placeholder_mask_elapsed_ms"] = round((time.perf_counter() - t_mask0) * 1000, 3)
                            t_scatter0 = time.perf_counter()
                            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                            _debug_sync(self._device)
                            debug["masked_scatter_elapsed_ms"] = round((time.perf_counter() - t_scatter0) * 1000, 3)
                            t_pos0 = time.perf_counter()
                            if hasattr(vl_model, 'compute_3d_position_ids'):
                                position_ids = vl_model.compute_3d_position_ids(
                                    input_ids=input_ids,
                                    image_grid_thw=image_grid_thw,
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    mm_token_type_ids=mm_token_type_ids,
                                )
                            else:
                                position_ids, rope_deltas = vl_model.get_rope_index(
                                    input_ids,
                                    image_grid_thw=image_grid_thw,
                                    attention_mask=attention_mask,
                                )
                                vl_model.rope_deltas = rope_deltas
                            _debug_sync(self._device)
                            debug["position_ids_elapsed_ms"] = round((time.perf_counter() - t_pos0) * 1000, 3)
                            prepared = {
                                "inputs_embeds": inputs_embeds,
                                "position_ids": position_ids,
                                "visual_pos_masks": image_mask[..., 0],
                                "deepstack_visual_embeds": deepstack_features,
                                "rope_deltas": vl_model.rope_deltas,
                                "prefill_len": torch.tensor([inputs_embeds.shape[1]], device=inputs_embeds.device),
                            }
                            visual_before = int(torch.nonzero(image_mask[..., 0][0], as_tuple=False).numel())
                            debug["prefill_len_before_prune"] = int(inputs_embeds.shape[1])
                            debug["visual_tokens_before_prune"] = visual_before
                            t_prune0 = time.perf_counter()
                            prepared = _maybe_prune_visual_tokens(vl_model, prepared, question, debug)
                            _debug_sync(self._device)
                            debug["prune_elapsed_ms"] = round((time.perf_counter() - t_prune0) * 1000, 3)
                            pkey = _prefill_cache_key(input_ids, pixel_values, image_grid_thw)
                            vl_model._aicas_prefill_input_cache.clear()
                            t_detach0 = time.perf_counter()
                            vl_model._aicas_prefill_input_cache[pkey] = _detach_prepared_prefill(prepared)
                            _debug_sync(self._device)
                            debug["detach_cache_elapsed_ms"] = round((time.perf_counter() - t_detach0) * 1000, 3)
                            debug["prefill_cache_key_input_shape"] = pkey[1]
                            debug["prefill_cache_key_pixel_shape"] = pkey[3]
                            debug["prefill_cache_key_grid"] = pkey[4]
                            debug["prefill_cache_size"] = len(vl_model._aicas_prefill_input_cache)
                            debug["prepared_prefill_len"] = int(prepared["prefill_len"][0].item())
                            debug["prepared_inputs_embeds_shape"] = _shape(prepared["inputs_embeds"])
                            debug["prepare_elapsed_ms"] = round((time.perf_counter() - t_prepare0) * 1000, 3)
            except Exception as e:
                debug["vision_cache_exception"] = f"{type(e).__name__}: {e}"
                if _debug_enabled():
                    _print_exception("processor_vision_cache", e)
                if not self._vision_cache_warned:
                    print(f"[vision_cache] skipped: {e}")
                    self._vision_cache_warned = True
        else:
            debug["tensor_result"] = bool(kwargs.get("return_tensors"))
            debug["has_pixel_values"] = hasattr(result, "pixel_values")

        debug["total_apply_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        _debug_print("processor", " ".join(f"{k}={v}" for k, v in debug.items()))

        if os.environ.get("AICAS_ENABLE_PREFIX_KV_PREWARM", "0") != "1":
            return _gpu_inputs if _gpu_inputs is not None else result
        if os.environ.get("AICAS_DISABLE_GRAPH_ENGINE", "0") == "1":
            return result
        if not kwargs.get("return_tensors"):
            return result

        try:
            inputs = result.to(self._device)
            with torch.inference_mode():
                self._model_getter().generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=0.0,
                    use_cache=True,
                )
            torch.cuda.synchronize()
        except Exception as e:
            if not self._prewarm_warned:
                print(f"[prewarm] skipped: {e}")
                self._prewarm_warned = True
        return result

    def __getattr__(self, name):
        return getattr(self._processor, name)
