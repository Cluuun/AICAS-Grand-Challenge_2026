"""
Prefill CUDA Graph capture and replay system for AICAS 2026.

Two components:
  1. PrefillGraphRunner  — Captures and replays prefill forward passes as CUDA
     graphs, keyed by (prompt_len, image_grid_thw).  Supports on-demand capture
     and exact matching (padding-free).

  2. GraphWarmupRunner   — Pre-warms ALL CUDA graphs in the framework
     (prefill, decode, full_graph, draft) using small targeted calls, NOT
     end-to-end generation.
"""

from __future__ import annotations

import contextlib
import os
from collections import OrderedDict
from typing import Any

import torch

from transformers.cache_utils import StaticCache

from spec_decode.eagle3 import (
    _disable_compiled_submodules_for_graph,
    _prepare_prefill_forward_inputs,
    _sanitize_qwen3vl_mm_token_type_ids,
    _select_eagle3_feature,
    _remember_eagle3_hidden_layout,
    _temporary_model_attr,
)

# ---------------------------------------------------------------------------
# Dummy input builder (shared by VLMModel + GraphWarmupRunner)
# ---------------------------------------------------------------------------

def build_dummy_prefill_inputs(
    model,
    prompt_len: int,
    grid_thw: tuple,
    device: torch.device,
) -> dict[str, Any]:
    """Build dummy inputs for prefill graph capture from shape params only.

    No dataset, no processor, no PIL images — just correctly-shaped tensors
    matching the Qwen3-VL vision encoder's patch contract.

    Qwen3-VL-2B config:
      - pixel_values:  (T*H*W, in_channels * T_patch * H_patch * W_patch)
                       = (T*H*W, 3 * 2 * 16 * 16) = (T*H*W, 1536)
      - n_img_tokens:  T*H*W // spatial_merge_size**2 = T*H*W // 4
      - image_token_id: 151655
    """
    T, H, W = grid_thw

    # ── Read vision config from model ──
    visual = getattr(getattr(model, "model", None), "visual", None)
    vc = getattr(visual, "config", None) if visual is not None else None
    in_channels = int(getattr(vc, "in_channels", 3))
    t_patch = int(getattr(vc, "temporal_patch_size", 2))
    h_patch = int(getattr(vc, "patch_size", 16))
    w_patch = h_patch
    merge_sz = int(getattr(visual, "spatial_merge_size", 2)) if visual is not None else 2
    img_token_id = getattr(model.config, "image_token_id", 151655)

    n_patches = T * H * W
    pixel_dim = in_channels * t_patch * h_patch * w_patch
    n_img_tokens = n_patches // (merge_sz * merge_sz)
    dtype = next(model.parameters()).dtype

    # ── input_ids: img tokens first, then filler ──
    input_ids = torch.full((1, prompt_len), 1, dtype=torch.long, device=device)
    input_ids[0, :n_img_tokens] = img_token_id

    # ── pixel_values: random patches of correct shape ──
    pixel_values = torch.randn(n_patches, pixel_dim, dtype=dtype, device=device)

    # ── image_grid_thw ──
    image_grid_thw = torch.tensor([[T, H, W]], dtype=torch.long, device=device)

    # ── attention_mask ──
    attention_mask = torch.ones((1, prompt_len), dtype=torch.long, device=device)

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def _get_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, str(int(default))).strip().lower()
    return val in TRUE_VALUES


def _parse_bucket_list(spec: str) -> list[int]:
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
            if v > 0:
                out.append(v)
        except ValueError:
            pass
    return sorted(set(out))


def _fallback_position_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Match Qwen3-VL's text-only fallback position ids."""
    if isinstance(attention_mask, torch.Tensor):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        rope_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        return position_ids.contiguous(), rope_deltas.contiguous()

    position_ids = (
        torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        .view(1, 1, -1)
        .expand(3, input_ids.shape[0], -1)
        .contiguous()
    )
    rope_deltas = torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long)
    return position_ids, rope_deltas


def _compute_qwen3vl_position_state(
    model,
    inputs: dict[str, Any],
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute the same mRoPE state Qwen3-VL would produce in model.forward."""
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise RuntimeError("prefill graph requires tensor input_ids for position_ids")

    core = getattr(model, "model", None)
    get_rope_index = getattr(core, "get_rope_index", None)
    if callable(get_rope_index):
        try:
            with torch.no_grad():
                position_ids, rope_deltas = get_rope_index(
                    input_ids,
                    inputs.get("image_grid_thw"),
                    inputs.get("video_grid_thw"),
                    attention_mask=attention_mask,
                )
            if isinstance(position_ids, torch.Tensor):
                pos = position_ids.to(device=input_ids.device, dtype=torch.long).contiguous()
                delta = (
                    rope_deltas.to(device=input_ids.device, dtype=torch.long).contiguous()
                    if isinstance(rope_deltas, torch.Tensor)
                    else None
                )
                return pos, delta
        except Exception:
            pass

    pos, delta = _fallback_position_ids(input_ids, attention_mask)
    return pos.to(dtype=torch.long).contiguous(), delta.to(dtype=torch.long).contiguous()


def _set_qwen3vl_rope_deltas(model, rope_deltas: torch.Tensor | None) -> None:
    if not isinstance(rope_deltas, torch.Tensor):
        return
    core = getattr(model, "model", None)
    if core is not None:
        core.rope_deltas = rope_deltas


@contextlib.contextmanager
def _temporary_text_attn_implementation(model, impl: str | None):
    """Temporarily select the text attention backend for prefill graph capture."""
    if not impl:
        yield
        return
    configs = []
    for obj in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(getattr(model, "model", None), "config", None), "text_config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ):
        if obj is not None and hasattr(obj, "_attn_implementation") and obj not in configs:
            configs.append(obj)
    old = [(cfg, getattr(cfg, "_attn_implementation")) for cfg in configs]
    try:
        for cfg in configs:
            cfg._attn_implementation = impl
        yield
    finally:
        for cfg, value in old:
            cfg._attn_implementation = value


@contextlib.contextmanager
def _temporary_prefill_maskless_causal_mask(enabled: bool):
    """Use SDPA's built-in causal mode for graph-captured full prefill.

    PrefillGraphRunner only captures batch=1, no-padding full prompts with a
    prompt-sized StaticCache.  In that shape, the explicit 4D causal mask is
    redundant and forces SDPA away from the native flash backend.  Returning
    None lets sdpa_attention_forward pass is_causal=True to PyTorch SDPA.
    """
    if not enabled:
        yield
        return
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_modeling
    except Exception:
        yield
        return
    original = qwen3_vl_modeling.create_causal_mask

    def _maskless_create_causal_mask(*args, **kwargs):
        if kwargs.get("or_mask_function") is not None or kwargs.get("and_mask_function") is not None:
            return original(*args, **kwargs)
        return None

    qwen3_vl_modeling.create_causal_mask = _maskless_create_causal_mask
    try:
        yield
    finally:
        qwen3_vl_modeling.create_causal_mask = original


@contextlib.contextmanager
def _temporary_prefill_packed_qkv_attention(model, enabled: bool):
    """Fuse text prefill q/k/v projections into one packed GEMM.

    Qwen3-VL text attention has dense q_proj/k_proj/v_proj/o_proj.  In the
    graph-captured prefill path, q/k/v are independent GEMMs with the same
    input.  Packing their weights reduces three launches/GEMMs per layer to
    one while preserving the rest of the attention path.
    """
    if not enabled:
        yield
        return
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_modeling
    except Exception:
        yield
        return

    attn_cls = getattr(qwen3_vl_modeling, "Qwen3VLTextAttention", None)
    if attn_cls is None:
        yield
        return

    lm = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(lm, "layers", None) if lm is not None else None
    if not layers:
        yield
        return

    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            continue
        packed_w = torch.cat(
            (attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight),
            dim=0,
        ).contiguous()
        packed_b = None
        if attn.q_proj.bias is not None or attn.k_proj.bias is not None or attn.v_proj.bias is not None:
            biases = []
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
                if proj.bias is None:
                    biases.append(torch.zeros(proj.weight.shape[0], device=proj.weight.device, dtype=proj.weight.dtype))
                else:
                    biases.append(proj.bias)
            packed_b = torch.cat(biases, dim=0).contiguous()
        attn._aicas_prefill_qkv_weight = packed_w
        attn._aicas_prefill_qkv_bias = packed_b
        attn._aicas_prefill_q_size = int(attn.q_proj.weight.shape[0])
        attn._aicas_prefill_k_size = int(attn.k_proj.weight.shape[0])

    original_forward = attn_cls.forward

    def _packed_qkv_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        packed_w = getattr(self, "_aicas_prefill_qkv_weight", None)
        if packed_w is None:
            return original_forward(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        packed_b = getattr(self, "_aicas_prefill_qkv_bias", None)
        qkv = torch.nn.functional.linear(hidden_states, packed_w, packed_b)
        q_size = int(getattr(self, "_aicas_prefill_q_size", self.q_proj.weight.shape[0]))
        k_size = int(getattr(self, "_aicas_prefill_k_size", self.k_proj.weight.shape[0]))

        query_states = self.q_norm(qkv[..., :q_size].view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(qkv[..., q_size:q_size + k_size].view(*input_shape, -1, self.head_dim)).transpose(1, 2)
        value_states = qkv[..., q_size + k_size:].view(*input_shape, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = qwen3_vl_modeling.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = qwen3_vl_modeling.eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = qwen3_vl_modeling.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn_cls.forward = _packed_qkv_forward
    try:
        yield
    finally:
        attn_cls.forward = original_forward


# ---------------------------------------------------------------------------
# Pseudo output object returned by graph replay
# ---------------------------------------------------------------------------

class _GraphPrefillOutput:
    """Lightweight output that mimics a HF CausalLMOutputWithPast for graph
    replay.  Carries pre-computed logits and feature; hidden_states are not
    available (feature is pre-extracted inside the graph)."""

    def __init__(
        self,
        logits: torch.Tensor,
        past_key_values,
        hidden_states=None,
        rope_deltas=None,
    ):
        self.logits = logits
        self.past_key_values = past_key_values
        self._last_hidden_state = hidden_states
        self.rope_deltas = rope_deltas

    @property
    def last_hidden_state(self):
        return self._last_hidden_state


# ---------------------------------------------------------------------------
# PrefillGraphRunner
# ---------------------------------------------------------------------------

class PrefillGraphRunner:
    """CUDA graph capture + replay for prefill forward pass.

    Captures ``model(**forward_kwargs)`` as a CUDA graph for each unique
    ``(prompt_len, image_grid_thw)`` combination encountered.  On replay,
    copies actual inputs into pre-allocated static buffers, replays the
    graph, and returns a lightweight output object with logits, feature,
    and KV cache.

    The feature (concatenated hidden states from ``layer_indices``) is
    extracted **inside** the captured graph so that hidden_states need not
    be materialised at replay time.

    Usage::

        runner = PrefillGraphRunner(model, device, layer_indices)
        runner.enable()
        # During prefill:
        hit = runner.lookup(prompt_len, image_grid_thw)
        if hit is not None:
            outputs, feature = runner.replay(hit, inputs)
        else:
            outputs, feature = eager_prefill(inputs)

    LRU eviction when the cache exceeds ``max_entries``.
    """

    def __init__(
        self,
        model,
        device: torch.device,
        layer_indices: tuple[int, int, int],
        max_entries: int = 64,
    ):
        self.model = model
        self.device = device
        self.layer_indices = layer_indices
        self.max_entries = max(max_entries, 1)  # kept for stats only, no eviction

        # OrderedDict, most-recently-used at the end (no eviction)
        self._cache: OrderedDict[tuple, dict] = OrderedDict()
        self._enabled = False
        self._capture_on_demand = _get_bool(
            "AICAS_PREFILL_TTFT_CAPTURE_ON_DEMAND", True
        )

        # Stats
        self.stats = {"hits": 0, "misses": 0, "captures": 0, "replays": 0}

    # -- properties -------------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def size(self) -> int:
        return len(self._cache)

    # -- public api -------------------------------------------------------

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def clear(self) -> None:
        self._cache.clear()

    def print_stats(self) -> None:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = 100.0 * self.stats["hits"] / max(total, 1)
        print(
            f"[prefill-graph] stats: hits={self.stats['hits']} "
            f"misses={self.stats['misses']} captures={self.stats['captures']} "
            f"replays={self.stats['replays']} "
            f"hit_rate={hit_rate:.1f}% "
            f"cache_size={self.size}",
            flush=True,
        )

    # -- key management ---------------------------------------------------

    def make_key(
        self,
        prompt_len: int,
        image_grid_thw: torch.Tensor | None,
    ) -> tuple:
        """Build a cache key from prompt length and image grid."""
        grid_key: tuple | None = None
        if isinstance(image_grid_thw, torch.Tensor):
            grid_key = tuple(int(x) for x in image_grid_thw.reshape(-1).cpu().tolist())
        dev_idx = self.device.index if self.device.index is not None else 0
        dtype_str = str(next(self.model.parameters()).dtype)
        return (int(prompt_len), grid_key, int(dev_idx), dtype_str)

    def lookup(
        self,
        prompt_len: int,
        image_grid_thw: torch.Tensor | None,
    ) -> dict | None:
        """Look up a cached graph entry.  Returns None on miss."""
        if not self._enabled:
            return None
        key = self.make_key(prompt_len, image_grid_thw)
        entry = self._cache.get(key)
        if entry is not None:
            self.stats["hits"] += 1
            self._cache.move_to_end(key)  # LRU refresh
            return entry
        self.stats["misses"] += 1
        return None

    # -- capture ----------------------------------------------------------

    def capture(
        self,
        prompt_len: int,
        inputs: dict[str, Any],
    ) -> dict | None:
        """Capture a CUDA graph for ``inputs`` at the given prompt length.

        ``inputs`` must contain at least ``input_ids``, ``attention_mask``,
        ``pixel_values``, and ``image_grid_thw`` (as returned by the
        processor).  The full model forward (vision encoder + text
        transformer) is captured.

        Returns the entry dict on success, or None on failure.
        Caller is responsible for progress reporting.
        """
        if not self._enabled:
            return None

        key = self.make_key(prompt_len, inputs.get("image_grid_thw"))
        if key in self._cache:
            return self._cache[key]  # already captured

        try:
            entry = self._do_capture(prompt_len, inputs, key)
        except Exception as exc:
            print(
                f"[prefill-graph] capture failed for len={prompt_len}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None

        self._put_cache(key, entry)
        self.stats["captures"] += 1
        return entry

    @staticmethod
    def _build_inputs_embeds(model, inputs: dict, device: torch.device, dtype) -> tuple:
        """Eagerly run vision encoder + merge to produce inputs_embeds.

        Returns (inputs_embeds, visual_pos_masks, deepstack_visual_embeds, attention_mask).
        vision encoder cannot be CUDA-graph-captured (fast_pos_embed_interpolate
        breaks capture), so this runs eagerly outside any graph.
        """
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')

        embed = model.model.get_input_embeddings()
        inputs_embeds = embed(input_ids)

        visual_pos_masks = None
        deepstack_visual_embeds = None

        if pixel_values is not None:
            image_embeds_list, deepstack_embeds_list = model.model.get_image_features(
                pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds_list, dim=0).to(device, dtype)
            img_token_id = getattr(model.config, "image_token_id", 151655)
            image_pos_mask = (input_ids == img_token_id).to(device=inputs_embeds.device)
            image_mask = image_pos_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_pos_mask
            deepstack_visual_embeds = deepstack_embeds_list

        seq_len = inputs_embeds.shape[1]
        attention_mask = inputs.get('attention_mask')
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds, attention_mask

    def _do_capture(
        self,
        prompt_len: int,
        inputs: dict[str, Any],
        key: tuple,
    ) -> dict:
        """Capture only the text transformer (language_model + lm_head) as a CUDA
        graph.  Vision encoder runs eagerly (ALINPU Conv3d + .item() calls
        are not graph-capturable).  Only language_model + lm_head captured.
        """
        model = self.model
        device = self.device
        dtype = next(model.parameters()).dtype
        layer_indices = self.layer_indices
        vocab_size = getattr(model.config, "vocab_size", 151936)
        hidden_size = getattr(model.config.text_config, "hidden_size", 2048)
        feat_dim = hidden_size * len(layer_indices)
        language_model = model.model.language_model

        # ------------------------------------------------------------------
        # 1. Eager vision + embeddings merge → inputs_embeds
        # ------------------------------------------------------------------
        torch.cuda.synchronize()
        with torch.profiler.record_function("prefill_graph_vision"):
            embeds, vis_masks, deepstack, attn_mask = self._build_inputs_embeds(
                model, inputs, device, dtype)
        position_ids, rope_deltas = _compute_qwen3vl_position_state(model, inputs, attn_mask)
        _set_qwen3vl_rope_deltas(model, rope_deltas)
        seq_len = embeds.shape[1]
        cache_position = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # 2. Static input buffers
        # ------------------------------------------------------------------
        s_embeds = embeds.clone()
        s_attn = attn_mask.clone()
        s_pos = position_ids.clone()
        s_rope_deltas = rope_deltas.clone() if isinstance(rope_deltas, torch.Tensor) else None
        s_cache = cache_position.clone()
        s_vis = vis_masks.clone() if vis_masks is not None else None
        s_deep = [d.clone() for d in deepstack] if deepstack else None

        # Prefill graph only needs prompt KV.  A prompt-sized cache keeps the
        # captured attention shape exact; decode graph expands it when needed.
        max_cache_len = prompt_len
        static_kv = StaticCache(config=model.config, max_batch_size=1,
                               max_cache_len=max_cache_len, device=device, dtype=dtype)
        out_logits = torch.empty((1, 1, vocab_size), dtype=dtype, device=device)
        out_feature = torch.empty((1, 1, feat_dim), dtype=dtype, device=device)
        # FA2's varlen path currently performs a host-side offset increment
        # that is illegal during CUDA graph capture.  Keep the model's default
        # SDPA backend unless explicitly overridden for experiments.
        attn_impl = os.getenv("AICAS_PREFILL_GRAPH_ATTN_IMPL", "").strip() or None
        maskless_sdpa = _get_bool("AICAS_PREFILL_GRAPH_MASKLESS_SDPA", True)
        packed_qkv = _get_bool("AICAS_PREFILL_GRAPH_PACKED_QKV", False)

        # ------------------------------------------------------------------
        # 3. Warmup
        # ------------------------------------------------------------------
        torch.cuda.synchronize()
        with torch.no_grad():
            with _temporary_text_attn_implementation(model, attn_impl):
                with _temporary_prefill_packed_qkv_attention(model, packed_qkv):
                    with _temporary_prefill_maskless_causal_mask(maskless_sdpa):
                        with torch.profiler.record_function("prefill_graph_warmup"):
                            lm_warm = language_model(
                                inputs_embeds=s_embeds, attention_mask=s_attn, position_ids=s_pos,
                                visual_pos_masks=s_vis, deepstack_visual_embeds=s_deep,
                                past_key_values=static_kv, use_cache=True,
                                cache_position=s_cache, output_hidden_states=True,
                            )
        _remember_eagle3_hidden_layout(model, lm_warm.hidden_states)
        torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # 4. Reset & capture
        # ------------------------------------------------------------------
        static_kv.reset()
        s_embeds.copy_(embeds)
        if s_vis is not None and vis_masks is not None:
            s_vis.copy_(vis_masks)
            for i, d in enumerate(deepstack):
                s_deep[i].copy_(d)

        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with _temporary_text_attn_implementation(model, attn_impl):
                with _temporary_prefill_packed_qkv_attention(model, packed_qkv):
                    with _temporary_prefill_maskless_causal_mask(maskless_sdpa):
                        with _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
                            with _disable_compiled_submodules_for_graph(model):
                                with torch.cuda.graph(graph):
                                    with torch.profiler.record_function("prefill_graph_capture"):
                                        lm_out = language_model(
                                            inputs_embeds=s_embeds, attention_mask=s_attn, position_ids=s_pos,
                                            visual_pos_masks=s_vis, deepstack_visual_embeds=s_deep,
                                            past_key_values=static_kv, use_cache=True,
                                            cache_position=s_cache, output_hidden_states=True,
                                        )
                                        feat_full = _select_eagle3_feature(lm_out.hidden_states, layer_indices)
                                        feat = feat_full[:, -1:, :]
                                        logits = model.lm_head(lm_out[0][:, -1:, :])
                                        out_logits.copy_(logits)
                                    out_feature.copy_(feat)

        torch.cuda.synchronize()

        entry = {
            "graph": graph,
            "s_embeds": s_embeds, "s_attn": s_attn,
            "s_pos": s_pos, "s_cache": s_cache,
            "s_rope_deltas": s_rope_deltas,
            "s_vis": s_vis, "s_deep": s_deep,
            "static_kv": static_kv,
            "out_logits": out_logits, "out_feature": out_feature,
            "prompt_len": prompt_len, "key": key,
        }
        return entry

    # -- replay -----------------------------------------------------------

    def replay(
        self,
        entry: dict,
        inputs: dict[str, Any],
    ) -> tuple[_GraphPrefillOutput, torch.Tensor]:
        """Replay captured text-only prefill graph."""
        model = self.model
        device = self.device
        dtype = next(model.parameters()).dtype
        s_embeds = entry["s_embeds"]
        s_attn = entry["s_attn"]
        s_pos = entry["s_pos"]
        s_vis = entry["s_vis"]
        s_deep = entry["s_deep"]
        static_kv = entry["static_kv"]
        static_kv.reset()

        with torch.profiler.record_function("prefill_graph_vision_replay"):
            embeds, vis_masks, deepstack, attn_mask = self._build_inputs_embeds(model, inputs, device, dtype)
        position_ids, rope_deltas = _compute_qwen3vl_position_state(model, inputs, attn_mask)
        s_embeds.copy_(embeds)
        s_attn.copy_(attn_mask)
        s_pos.copy_(position_ids)
        s_rope_deltas = entry.get("s_rope_deltas")
        if isinstance(s_rope_deltas, torch.Tensor) and isinstance(rope_deltas, torch.Tensor):
            s_rope_deltas.copy_(rope_deltas)
            _set_qwen3vl_rope_deltas(model, s_rope_deltas)
        else:
            _set_qwen3vl_rope_deltas(model, rope_deltas)
        if s_vis is not None and vis_masks is not None:
            s_vis.copy_(vis_masks)
            if s_deep is not None and deepstack is not None:
                for i, d in enumerate(deepstack):
                    if i < len(s_deep):
                        s_deep[i].copy_(d)

        with torch.profiler.record_function("prefill_graph_replay"):
            entry["graph"].replay()
        self.stats["replays"] += 1

        output = _GraphPrefillOutput(
            logits=entry["out_logits"],
            past_key_values=static_kv,
            rope_deltas=entry.get("s_rope_deltas"),
        )
        _remember_eagle3_hidden_layout(self.model, None)
        return output, entry["out_feature"]

    # -- cache helpers ----------------------------------------------------

    def _put_cache(self, key: tuple, entry: dict) -> None:
        self._cache[key] = entry
        self._cache.move_to_end(key)


# ---------------------------------------------------------------------------
# GraphWarmupRunner
# ---------------------------------------------------------------------------

class GraphWarmupRunner:
    """Pre-warms ALL CUDA graphs in the framework (prefill, decode,
    full_graph, draft) using small targeted calls.

    NOT end-to-end generation.  Instead, it directly triggers the graph
    capture functions with representative shapes derived from the benchmark
    dataset.
    """

    def __init__(self, model, draft_model, processor, device: torch.device):
        self.model = model
        self.draft_model = draft_model
        self.processor = processor
        self.device = device

    def warmup_all(
        self,
        dataset=None,
        prefill_runner: PrefillGraphRunner | None = None,
        num_probe_samples: int = 20,
    ) -> dict[str, int]:
        """Pre-capture all CUDA graphs (parameterized, no dataset).

        Steps:
          1. Read unique (prompt_len, grid_thw) pairs from shape profile.
          2. Capture a prefill graph for each combination with dummy inputs.
          3. Trigger decode graph warmup via _precapture_decode_graph().
          4. Trigger draft / full-graph warmup (short speculative runs).

        Returns a dict with counts of graphs warmed up per category.
        """
        import tqdm

        counts: dict[str, int] = {"prefill": 0}

        # ------------------------------------------------------------------
        # Phase 1: collect unique shapes from profile (or fallback)
        # ------------------------------------------------------------------
        try:
            from utils import _load_ttft_shape_profile
            shapes = _load_ttft_shape_profile()  # [(grid, prompt_len, freq), ...]
        except Exception:
            shapes = []

        if not shapes:
            print("[graph-warmup] no shapes in profile, skipping prefill capture", flush=True)
        else:
            seen: set = set()
            unique: list[tuple[tuple, int]] = []
            for grid, pl, _freq in shapes:
                key = (tuple(grid), int(pl))
                if key not in seen:
                    seen.add(key)
                    unique.append(key)
            unique = unique[:max(1, num_probe_samples)]

            # ------------------------------------------------------------------
            # Phase 2: capture prefill graphs with tqdm
            # ------------------------------------------------------------------
            if prefill_runner is not None and prefill_runner.is_enabled:
                pbar = tqdm.tqdm(
                    unique, desc="graph-warmup-prefill", unit="shape",
                    dynamic_ncols=True, leave=True,
                )
                for grid_tuple, prompt_len in pbar:
                    pbar.set_postfix_str(f"{prompt_len}×{grid_tuple}")
                    try:
                        dummy = build_dummy_prefill_inputs(
                            self.model, prompt_len, grid_tuple, self.device)
                        cap = prefill_runner.capture(prompt_len, dummy)
                        if cap is not None:
                            counts["prefill"] += 1
                    except Exception as exc:
                        print(
                            f"\n[graph-warmup] prefill capture failed for "
                            f"len={prompt_len} grid={grid_tuple}: {exc}",
                            flush=True,
                        )
                pbar.close()

        # ------------------------------------------------------------------
        # Phase 3: trigger decode graph warmup
        # ------------------------------------------------------------------
        self._warmup_decode_graphs()

        # ------------------------------------------------------------------
        # Phase 4: trigger full-graph / draft warmup
        # ------------------------------------------------------------------
        # (unchanged — spec decode is commented out in evaluation_wrapper)

        print(
            f"[graph-warmup] complete: prefill={counts['prefill']}",
            flush=True,
        )
        return counts

    # -- internal helpers -------------------------------------------------

    def _warmup_decode_graphs(self) -> None:
        """Trigger decode graph capture via direct _precapture_decode_graph() calls.

        No dataset samples, no model.generate() — just parameterized bucket calls.
        """
        import tqdm

        model = self.model
        precap = getattr(model, "_precapture_decode_graph", None)
        if not callable(precap):
            print("[graph-warmup] _precapture_decode_graph not available, skipping decode warmup", flush=True)
            return

        from decode_graph_runtime import FIXED_MCL, _parse_decode_mcl_buckets

        mcl_buckets = _parse_decode_mcl_buckets(
            os.getenv("AICAS_DECODE_GRAPH_MCL_BUCKETS", ""), FIXED_MCL)

        # Build combinations: (full_mask, null_mask, chunk)
        combos = [
            (True, False, 1),
        ]
        if os.getenv("AICAS_DECODE_GRAPH_NULL_ATTN_MASK", "0") == "1":
            combos.append((False, True, 1))

        items = [(mcl, full, null, chunk) for mcl in mcl_buckets for full, null, chunk in combos]

        captured = 0
        pbar = tqdm.tqdm(items, desc="graph-warmup-decode", unit="slot",
                         dynamic_ncols=True, leave=True)
        for mcl, full_mask, null_mask, chunk in pbar:
            pbar.set_postfix_str(f"mcl={mcl} full={full_mask} null={null_mask} chunk={chunk}")
            try:
                ok = precap(
                    batch_size=1, max_new_tokens=16,
                    full_mask=full_mask, null_mask=null_mask,
                    chunk_steps=chunk, mcl=mcl,
                )
                if ok:
                    captured += 1
            except Exception:
                pass
        pbar.close()
        if captured:
            print(f"[graph-warmup] decode graphs captured: {captured}", flush=True)

    def _warmup_spec_graphs(self) -> None:
        """Trigger spec-decode (EAGLE3) graph capture (parameterized)."""
        if os.getenv("AICAS_SPEC_DECODE", "0") != "1":
            return
        if self.draft_model is None:
            return

        spec_ctx = {
            "AICAS_EAGLE3_FULL_GRAPH": "1",
            "AICAS_EAGLE3_FULL_GRAPH_CAPTURE_ON_MISS": "1",
            "AICAS_EAGLE3_GRAPH_CAPTURE_ON_DEMAND_AFTER_PRECAPTURE": "1",
            "AICAS_EAGLE3_SPEC_PRECAPTURE_ACTIVE": "1",
        }
        prev = {}
        for k, v in spec_ctx.items():
            prev[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            # Use dummy prefill inputs from the shape profile
            try:
                from utils import _load_ttft_shape_profile
                shapes = _load_ttft_shape_profile()
            except Exception:
                shapes = []

            shapes = shapes[:5]  # top-5 shapes
            with torch.no_grad():
                for grid, prompt_len, _freq in shapes:
                    try:
                        dummy = build_dummy_prefill_inputs(
                            self.model, prompt_len, tuple(grid), self.device)
                        _ = self.model.generate(
                            **dummy,
                            max_new_tokens=8,
                            min_new_tokens=8,
                            do_sample=False,
                            temperature=0.0,
                            use_cache=True,
                        )
                    except Exception:
                        pass
        finally:
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
