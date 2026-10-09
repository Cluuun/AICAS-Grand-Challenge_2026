from __future__ import annotations

"""Fixed-shape CUDA Graph for the Qwen3-VL text prefill block.

The graph starts after vision features have already been written into
``inputs_embeds`` and ends after ``lm_head`` greedy selection.  Runtime prompts
shorter than the bucket are right padded, but the selected hidden state and the
logical KV length both stay tied to the real prompt length.
"""

from dataclasses import dataclass
from types import MethodType
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .conf import conf_bool, conf_int


_GRAPH_POOL: Dict[Tuple[Any, ...], "TextPrefillCudaGraph"] = {}
_OWNED_KV_POOL: Dict[Tuple[Any, ...], Any] = {}
_STATS: Dict[str, int] = {
    "capture_success": 0,
    "capture_fail": 0,
    "replay": 0,
    "fallback_long": 0,
    "fallback_ineligible": 0,
}
_LAST_ERROR = ""


def _resolve_visual_range(visual_pos_masks: Optional[torch.Tensor], deepstack_visual_embeds) -> Optional[Tuple[int, int]]:
    if (
        not torch.is_tensor(visual_pos_masks)
        or visual_pos_masks.dim() != 2
        or int(visual_pos_masks.shape[0]) != 1
        or not deepstack_visual_embeds
        or not torch.is_tensor(deepstack_visual_embeds[0])
    ):
        return None
    expected = int(deepstack_visual_embeds[0].shape[0])
    if expected <= 0:
        return None
    idx = visual_pos_masks[0].nonzero(as_tuple=False).flatten()
    if int(idx.numel()) != expected:
        return None
    start = int(idx[0].item())
    end = start + expected
    expected_idx = torch.arange(start, end, dtype=idx.dtype, device=idx.device)
    if not torch.equal(idx, expected_idx):
        return None
    return start, end


def _patch_static_deepstack(language_model) -> None:
    if getattr(language_model, "_aicas_static_deepstack_patched", False):
        return
    original = language_model._deepstack_process

    def static_deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds, _original=original):
        if getattr(self, "_aicas_deepstack_static_range_enabled", False):
            span = getattr(self, "_aicas_deepstack_static_range", None)
            if (
                isinstance(span, tuple)
                and len(span) == 2
                and hidden_states.dim() == 3
                and int(hidden_states.shape[0]) == 1
                and torch.is_tensor(visual_embeds)
            ):
                start, end = int(span[0]), int(span[1])
                visual_len = end - start
                if (
                    0 <= start < end <= int(hidden_states.shape[1])
                    and int(visual_embeds.shape[0]) == visual_len
                    and int(visual_embeds.shape[-1]) == int(hidden_states.shape[-1])
                ):
                    add = visual_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype).view(
                        1, visual_len, int(hidden_states.shape[-1])
                    )
                    hidden_states[:, start:end, :] = hidden_states[:, start:end, :] + add
                    return hidden_states
        return _original(hidden_states, visual_pos_masks, visual_embeds)

    language_model._deepstack_process = MethodType(static_deepstack_process, language_model)
    language_model._aicas_static_deepstack_patched = True


def _set_deepstack_static_range(vlm, span: Optional[Tuple[int, int]]) -> None:
    language_model = getattr(vlm, "language_model", None)
    if language_model is None or not hasattr(language_model, "_deepstack_process"):
        return
    _patch_static_deepstack(language_model)
    if span is None:
        language_model._aicas_deepstack_static_range_enabled = False
        language_model._aicas_deepstack_static_range = None
    else:
        language_model._aicas_deepstack_static_range_enabled = True
        language_model._aicas_deepstack_static_range = (int(span[0]), int(span[1]))


@dataclass
class TextPrefillGraphResult:
    logits: torch.Tensor
    next_token: torch.Tensor
    past_key_values: Any
    mode: str
    stats: Dict[str, Any]


def _graph_stats() -> Dict[str, Any]:
    out: Dict[str, Any] = {key: int(value) for key, value in _STATS.items()}
    out["last_error"] = _LAST_ERROR
    return out


def _log_stats() -> None:
    if not conf_bool("TEXT_PREFILL_CUDA_GRAPH_LOG_STATS", "0"):
        return
    stats = _graph_stats()
    print(
        "[text_prefill_graph] "
        f"capture_ok={stats['capture_success']} capture_fail={stats['capture_fail']} "
        f"replay={stats['replay']} fallback_long={stats['fallback_long']} "
        f"fallback_ineligible={stats['fallback_ineligible']} last_error={stats['last_error']}"
    )


def _deepstack_schema(deepstack_visual_embeds) -> Tuple[Any, ...]:
    if not deepstack_visual_embeds:
        return ()
    schema = []
    for tensor in deepstack_visual_embeds:
        if not torch.is_tensor(tensor):
            return ()
        schema.append((tuple(tensor.shape), tensor.dtype, str(tensor.device)))
    return tuple(schema)


def _get_owned_static_kv(model, *, bucket_len: int, device: torch.device, dtype: torch.dtype):
    key = (str(device), str(dtype), int(bucket_len))
    cached = _OWNED_KV_POOL.get(key)
    if cached is not None:
        return cached
    from .cuda_graph_decode import StaticKVCache, _bucket_max_cache_len

    cached = StaticKVCache.from_model(
        model,
        max_cache_len=_bucket_max_cache_len(int(bucket_len) + 8),
        batch_size=1,
        dtype=dtype,
        device=device,
    )
    _OWNED_KV_POOL[key] = cached
    return cached


def _pad_position_ids(dst: torch.Tensor, src: torch.Tensor, real_len: int) -> None:
    dst[..., :real_len].copy_(src[..., :real_len])
    if real_len < int(dst.shape[-1]):
        dst[..., real_len:].copy_(src[..., real_len - 1 : real_len].expand(*src.shape[:-1], dst.shape[-1] - real_len))


class TextPrefillCudaGraph:
    def __init__(
        self,
        *,
        bucket_len: int,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds,
        past_key_values,
        use_cache: bool,
        visual_range: Optional[Tuple[int, int]],
    ) -> None:
        self.bucket_len = int(bucket_len)
        self.device = inputs_embeds.device
        self.dtype = inputs_embeds.dtype
        self.hidden_size = int(inputs_embeds.shape[-1])
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.enabled = False
        self.disabled = False
        self.last_failure: Optional[str] = None
        self.captures = 0
        self.replays = 0
        self.past_key_values = past_key_values
        self.use_cache = bool(use_cache)
        self.visual_range = visual_range

        self.inputs_embeds_buf = torch.empty(
            (1, self.bucket_len, self.hidden_size),
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        self.position_ids_buf = torch.empty(
            (*position_ids.shape[:-1], self.bucket_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        self.attention_mask_buf = None
        if torch.is_tensor(attention_mask):
            self.attention_mask_buf = torch.empty(
                (attention_mask.shape[0], self.bucket_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        self.visual_pos_masks_buf = None
        if torch.is_tensor(visual_pos_masks):
            self.visual_pos_masks_buf = torch.empty(
                (visual_pos_masks.shape[0], self.bucket_len),
                dtype=torch.bool,
                device=visual_pos_masks.device,
            )
        self.deepstack_bufs = (
            [torch.empty_like(tensor) for tensor in deepstack_visual_embeds]
            if deepstack_visual_embeds
            else None
        )
        self.cache_position_buf = torch.arange(self.bucket_len, dtype=torch.long, device=self.device)
        self.last_select_buf = torch.zeros((1, 1, self.bucket_len), dtype=self.dtype, device=self.device)
        self.static_logits: Optional[torch.Tensor] = None
        self.static_token: Optional[torch.Tensor] = None

    def _reset_kv_for_bucket_prefill(self) -> None:
        if self.past_key_values is None:
            return
        self.past_key_values.reset(0)
        self.past_key_values._write_pos_host = 0
        self.past_key_values.cache_seqlens_buf.fill_(0)

    def _copy_inputs(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds,
        real_len: int,
    ) -> None:
        self.inputs_embeds_buf.zero_()
        self.inputs_embeds_buf[:, :real_len, :].copy_(inputs_embeds[:, :real_len, :])
        _pad_position_ids(self.position_ids_buf, position_ids, real_len)
        self.last_select_buf.zero_()
        self.last_select_buf[:, :, real_len - 1].fill_(1)

        if self.attention_mask_buf is not None:
            self.attention_mask_buf.zero_()
            if torch.is_tensor(attention_mask):
                self.attention_mask_buf[:, :real_len].copy_(attention_mask[:, :real_len])
            else:
                self.attention_mask_buf[:, :real_len].fill_(1)

        if self.visual_pos_masks_buf is not None:
            self.visual_pos_masks_buf.zero_()
            if torch.is_tensor(visual_pos_masks):
                self.visual_pos_masks_buf[:, :real_len].copy_(visual_pos_masks[:, :real_len])

        if self.deepstack_bufs is not None:
            for dst, src in zip(self.deepstack_bufs, deepstack_visual_embeds):
                dst.copy_(src)

    def _forward_static(self, model, vlm):
        attention_mask = (
            self.attention_mask_buf
            if conf_bool("TEXT_PREFILL_CUDA_GRAPH_USE_ATTENTION_MASK", "0")
            else None
        )
        lm_outputs = vlm.language_model(
            input_ids=None,
            inputs_embeds=self.inputs_embeds_buf,
            position_ids=self.position_ids_buf,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            cache_position=self.cache_position_buf,
            use_cache=self.use_cache,
            visual_pos_masks=self.visual_pos_masks_buf,
            deepstack_visual_embeds=self.deepstack_bufs,
            return_dict=True,
        )
        last_hidden = torch.bmm(self.last_select_buf, lm_outputs.last_hidden_state)
        logits = F.linear(last_hidden, model.lm_head.weight, getattr(model.lm_head, "bias", None))
        token = torch.argmax(logits, dim=-1)
        return logits, token

    def capture(
        self,
        model,
        vlm,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds,
        real_len: int,
    ) -> bool:
        global _LAST_ERROR
        if not torch.cuda.is_available():
            self.last_failure = "no_cuda"
            _LAST_ERROR = self.last_failure
            return False
        try:
            _set_deepstack_static_range(vlm, self.visual_range)
            warmup_iters = conf_int("TEXT_PREFILL_CUDA_GRAPH_WARMUP_ITERS", "1", minimum=0)
            self._copy_inputs(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                real_len=real_len,
            )
            for _ in range(warmup_iters):
                self._reset_kv_for_bucket_prefill()
                self.static_logits, self.static_token = self._forward_static(model, vlm)
            torch.cuda.synchronize()
            self._reset_kv_for_bucket_prefill()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_logits, self.static_token = self._forward_static(model, vlm)
            _set_deepstack_static_range(vlm, None)
            self.enabled = True
            self.captures += 1
            self.last_failure = None
            _STATS["capture_success"] += 1
            _LAST_ERROR = ""
            return True
        except Exception as exc:
            _set_deepstack_static_range(vlm, None)
            self.enabled = False
            self.disabled = True
            self.graph = None
            self.last_failure = f"{type(exc).__name__}: {exc}"
            _STATS["capture_fail"] += 1
            _LAST_ERROR = self.last_failure
            return False

    def run(
        self,
        model,
        vlm,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        visual_pos_masks: Optional[torch.Tensor],
        deepstack_visual_embeds,
        real_len: int,
    ) -> Optional[TextPrefillGraphResult]:
        if self.disabled:
            _set_deepstack_static_range(vlm, None)
            return None
        _set_deepstack_static_range(vlm, self.visual_range)
        if not self.enabled:
            ok = self.capture(
                model,
                vlm,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                real_len=real_len,
            )
            if not ok:
                _set_deepstack_static_range(vlm, None)
                _log_stats()
                return None

        self._copy_inputs(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            real_len=real_len,
        )
        self._reset_kv_for_bucket_prefill()
        self.graph.replay()
        if self.past_key_values is not None:
            self.past_key_values.reset(real_len)
        _set_deepstack_static_range(vlm, None)
        self.replays += 1
        _STATS["replay"] += 1
        _log_stats()
        return TextPrefillGraphResult(
            logits=self.static_logits,
            next_token=self.static_token,
            past_key_values=self.past_key_values,
            mode=f"text_prefill_graph_{self.bucket_len}{'_kv' if self.use_cache else '_no_kv'}",
            stats=_graph_stats(),
        )


def run_text_prefill_cuda_graph(
    *,
    model,
    vlm,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    visual_pos_masks: Optional[torch.Tensor],
    deepstack_visual_embeds,
    past_key_values=None,
    use_cache: bool = True,
) -> Optional[TextPrefillGraphResult]:
    if not conf_bool("ENABLE_TEXT_PREFILL_CUDA_GRAPH", "1"):
        return None
    if (
        not torch.is_tensor(inputs_embeds)
        or inputs_embeds.dim() != 3
        or int(inputs_embeds.shape[0]) != 1
        or not torch.is_tensor(position_ids)
        or position_ids.shape[-1] != inputs_embeds.shape[1]
        or getattr(model, "lm_head", None) is None
    ):
        _STATS["fallback_ineligible"] += 1
        return None

    bucket_len = conf_int("TEXT_PREFILL_CUDA_GRAPH_BUCKET_LEN", "125", minimum=1)
    real_len = int(inputs_embeds.shape[1])
    if real_len <= 0:
        _STATS["fallback_ineligible"] += 1
        return None
    if real_len > bucket_len:
        _STATS["fallback_long"] += 1
        _set_deepstack_static_range(vlm, None)
        _log_stats()
        return None

    schema = _deepstack_schema(deepstack_visual_embeds)
    if deepstack_visual_embeds and not schema:
        _STATS["fallback_ineligible"] += 1
        _set_deepstack_static_range(vlm, None)
        return None
    visual_range = _resolve_visual_range(visual_pos_masks, deepstack_visual_embeds)
    if deepstack_visual_embeds and visual_range is None:
        _STATS["fallback_ineligible"] += 1
        _set_deepstack_static_range(vlm, None)
        return None
    _set_deepstack_static_range(vlm, visual_range)

    if not use_cache:
        past_key_values = None
        kv_key: Any = "no_cache"
    elif past_key_values is None:
        past_key_values = _get_owned_static_kv(
            model,
            bucket_len=bucket_len,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        kv_key: Any = "owned"
    else:
        if not hasattr(past_key_values, "reset") or not hasattr(past_key_values, "cache_seqlens_buf"):
            _STATS["fallback_ineligible"] += 1
            return None
        kv_key = id(past_key_values)

    key = (
        str(inputs_embeds.device),
        str(inputs_embeds.dtype),
        int(bucket_len),
        int(inputs_embeds.shape[-1]),
        tuple(position_ids.shape[:-1]),
        attention_mask.dtype if torch.is_tensor(attention_mask) else None,
        visual_pos_masks.dtype if torch.is_tensor(visual_pos_masks) else None,
        schema,
        visual_range,
        bool(use_cache),
        kv_key,
    )
    graph = _GRAPH_POOL.get(key)
    if graph is None:
        graph = TextPrefillCudaGraph(
            bucket_len=bucket_len,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            visual_range=visual_range,
        )
        _GRAPH_POOL[key] = graph

    return graph.run(
        model,
        vlm,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=attention_mask,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        real_len=real_len,
    )


def graph_result_to_outputs(result: TextPrefillGraphResult, rope_deltas=None):
    return SimpleNamespace(
        logits=result.logits,
        next_token=result.next_token,
        past_key_values=result.past_key_values,
        rope_deltas=rope_deltas,
        hidden_states=None,
        prefill_graph_stats=result.stats,
    )
