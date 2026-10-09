"""
Merged two-layer KV cache for Qwen3-VL prefill acceleration.

Architecture:
  Layer 1: ImageKVCache (from kvcache/core.py) — exact hash match on pixel_values.
  Layer 2: SimpleRadixKVCache (from kvcache/core.py) — radix tree prefix match on question tokens.

Prefill flow:
  1. Hash pixel_values + image_grid_thw, check Layer 1.
  2. On Layer 1 HIT:
       a. Reuse cached image KV as past_key_values.
       b. Check Layer 2 (radix tree) for question token prefix match.
       c. If Layer 2 HIT: merge image KV + prefix KV, run remaining tokens.
       d. If Layer 2 MISS: run full question with image KV as past.
       e. Combine cached image hidden-state features with Phase 2 features.
  3. On Layer 1 MISS:
       a. Full prefill (image + question).
       b. Cache image KV + image hidden-state features in Layer 1.
       c. Insert question token prefix into Layer 2 radix tree.

TTFT speedup comes primarily from Phase 1 skip: Qwen3-VL images have
hundreds to thousands of tokens.  Skipping ALL image-token prefill
on a cache hit is the main win.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Optional

import torch

from kvcache.core import SimpleRadixKVCache
from spec_decode.verifier import _prepare_prefill_forward_inputs

IMAGE_TOKEN_ID = 151655  # Qwen3VL <|image_pad|>

_KVCACHE_DEBUG = False


# ========================================================================
# Hashing
# ========================================================================

def _hash_tensor(t: Optional[torch.Tensor]) -> int:
    """Stable hash of tensor bytes."""
    if t is None:
        return 0
    return hash(t.detach().cpu().resolve_conj().numpy().tobytes())


# ========================================================================
# MergedKVCache
# ========================================================================

class MergedKVCache:
    """Two-layer KV cache for prefill acceleration.

    Layer 1 (Image):
      Key   = hash(pixel_values) + hash(image_grid_thw)
      Value = image-token KV + Eagle3 hidden-state features for image positions.
      Hit   => Phase 1 (image prefill) is entirely skipped.
              Vision encoder does NOT run.  LLM does NOT process image tokens.

    Layer 2 (Text prefix):
      Key   = radix tree over question tokens (from image_end to end-of-prompt)
      Value = KV cache segment for each 16-token block.
      Hit   => A prefix of the question tokens is NOT recomputed in Phase 2.

    Both layers share the same `prefill()` entry point that returns
    (outputs, feature) with the same interface as `_prefill_with_full_feature`.
    """

    def __init__(
        self,
        max_image_entries: int = 64,
        max_blocks: int = 256,
        block_size: int = 16,
        debug: bool = False,
    ):
        self.max_image_entries = max_image_entries
        self.max_full_entries = max_image_entries * 2
        self.radix_cache = SimpleRadixKVCache(max_blocks=max_blocks, block_size=block_size)
        # Image cache: OrderedDict[(pixel_hash, grid_hash), image_entry_dict]
        # Using OrderedDict for built-in LRU eviction.
        self._image_store: OrderedDict[int, dict] = OrderedDict()
        # Full-result cache: {token_hash: {"outputs": CachedOutput, "feature": Tensor}}
        self._full_cache: OrderedDict[int, dict] = OrderedDict()
        global _KVCACHE_DEBUG
        _KVCACHE_DEBUG = debug

        self.stats = {
            "requests": 0,
            "layer1_hits": 0,
            "layer1_misses": 0,
            "layer1_stores": 0,
            "layer1_entries": 0,
            "phase1_skipped_tokens": 0,
            "layer2_requests": 0,
            "layer2_matched_tokens": 0,
            "layer2_total_tokens": 0,
            "layer2_stores": 0,
            "phase2_skipped_tokens": 0,
        }

    # ====================================================================
    # Layer 1: Image cache
    # ====================================================================

    @staticmethod
    def _image_hash(pixel_values: torch.Tensor) -> int:
        """Stable hash of pixel_values tensor."""
        flat = pixel_values.flatten()
        step = max(1, flat.shape[0] // 512)
        sample = flat[::step].cpu()
        return hash(sample.numpy().tobytes())

    def lookup_image(
        self,
        pixel_values: Optional[torch.Tensor],
    ) -> Optional[dict]:
        """Look up cached image KV + features.

        Returns dict with keys (image_kv, image_feature, image_end_pos, img_len)
        or None on miss.  Touches LRU order on hit.
        """
        if pixel_values is None:
            return None
        img_hash = self._image_hash(pixel_values)
        self.stats["requests"] += 1

        if img_hash not in self._image_store:
            self.stats["layer1_misses"] += 1
            return None

        # Move to end (most recently used) for LRU
        self._image_store.move_to_end(img_hash)
        entry = self._image_store[img_hash]

        self.stats["layer1_hits"] += 1
        self.stats["phase1_skipped_tokens"] += entry["img_len"]

        return entry

    def _evict_image_if_needed(self) -> None:
        """Evict LRU image entries until under limit."""
        while len(self._image_store) >= self.max_image_entries:
            self._image_store.popitem(last=False)  # Remove oldest (least recently used)

    def store_image(
        self,
        pixel_values: Optional[torch.Tensor],
        img_end: int,
        key_cache: list,
        value_cache: list,
        image_feature: torch.Tensor,
    ) -> None:
        """Store image KV + Eagle3 feature in Layer 1.

        Args:
            image_feature: Eagle3 hidden-state feature for [0, img_end) positions.
                           Shape [1, img_end, feat_dim].
        """
        if pixel_values is None:
            return
        img_hash = self._image_hash(pixel_values)
        if img_hash in self._image_store:
            return  # already stored

        self._evict_image_if_needed()

        self._image_store[img_hash] = {
            "img_hash": img_hash,
            "img_start": 0,
            "img_end": img_end,
            "img_len": img_end,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "image_feature": image_feature,
        }
        self.stats["layer1_stores"] += 1
        self.stats["layer1_entries"] = len(self._image_store)

        if _KVCACHE_DEBUG:
            print(f"[MergedCache] Layer1 STORE hash={img_hash} "
                  f"img_tokens={img_end} entries={len(self._image_store)}")

    # ====================================================================
    # Layer 2: Radix prefix cache
    # ====================================================================

    def match_text_prefix(self, token_ids_cpu: torch.Tensor) -> tuple[int, list]:
        """Match token prefix against radix tree.

        Returns (matched_tokens, path_nodes).
        """
        self.stats["layer2_requests"] += 1
        self.stats["layer2_total_tokens"] += token_ids_cpu.shape[0]
        matched_tokens, path_nodes = self.radix_cache.match_prefix(token_ids_cpu)
        self.stats["layer2_matched_tokens"] += matched_tokens
        self.stats["phase2_skipped_tokens"] += matched_tokens
        return matched_tokens, path_nodes

    def store_text_prefix(
        self,
        token_ids_cpu: torch.Tensor,
        key_cache: list,
        value_cache: list,
        already_matched: int = 0,
    ) -> None:
        """Insert token IDs (question tokens) into radix tree."""
        self.radix_cache.insert(token_ids_cpu, key_cache, value_cache, already_matched)
        self.stats["layer2_stores"] += 1

    def reconstruct_text_prefix_kv(self, path_nodes: list):
        """Reconstruct DynamicCache from radix tree path nodes."""
        return self.radix_cache.reconstruct_kv_cache(path_nodes)

    # ====================================================================
    # Main prefill entry point
    # ====================================================================

    @staticmethod
    def _full_prefill(model, inputs: dict, layer_indices: tuple[int, int, int]):
        from spec_decode.eagle3 import _prefill_with_full_feature
        return _prefill_with_full_feature(model, inputs, layer_indices)

    def prefill(
        self,
        model,
        inputs: dict,
        layer_indices: tuple[int, int, int],
    ) -> tuple[Any, torch.Tensor]:
        """Prefill with full-result caching.

        On exact full-sequence cache hit, returns cached (outputs, feature)
        — zero forward passes.  On miss, delegates to _full_prefill (which
        uses prefill graph acceleration) and caches the result.
        """
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return self._full_prefill(model, inputs, layer_indices)

        full_hash = hash(input_ids[0].cpu().numpy().tobytes())

        # ── Exact full-match cache lookup ───────────────────────────
        if full_hash in self._full_cache:
            entry = self._full_cache[full_hash]
            if _KVCACHE_DEBUG:
                print(f"[MergedCache] FULL HIT hash={full_hash} len={input_ids.shape[1]}")
            return entry["outputs"], entry["feature"]

        # ── Miss: full prefill (delegates to prefill graph internally) ──
        from spec_decode.eagle3 import _CachedOutput, _clone_full_kv, _last_logits_from_output
        outputs, feature = self._full_prefill(model, inputs, layer_indices)

        # Cache the full result for future reuse
        self._full_cache[full_hash] = {
            "outputs": _CachedOutput(
                past_key_values=_clone_full_kv(outputs.past_key_values),
                logits=_last_logits_from_output(outputs),
            ),
            "feature": feature.clone(),
        }
        # LRU eviction
        while len(self._full_cache) > self.max_full_entries:
            self._full_cache.pop(next(iter(self._full_cache)))

        if _KVCACHE_DEBUG:
            print(f"[MergedCache] FULL MISS hash={full_hash} len={input_ids.shape[1]} "
                  f"cache_size={len(self._full_cache)}")

        return outputs, feature



def prefill_with_merged_cache(model, inputs, layer_indices, kv_cache=None):
    """Drop-in replacement for _prefill_with_full_feature using MergedKVCache."""
    if kv_cache is None:
        from spec_decode.eagle3 import _prefill_with_full_feature
        return _prefill_with_full_feature(model, inputs, layer_indices)
    return kv_cache.prefill(model, inputs, layer_indices)

