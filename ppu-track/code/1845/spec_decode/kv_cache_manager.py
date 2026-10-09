"""
Two-layer KV prefix cache for Qwen3-VL speculative decoding.

Layer 1: Image KV Cache -- caches KV for system + image tokens.
    Key: hash(pixel_values) + hash(image_grid_thw)
    Guarantee: Same image always produces identical image token KV.

Layer 2: Text Prefix Cache -- caches KV for question tokens (built on Layer 1).
    Key: image_kv_hash + hash(question_tokens)
    Value: text KV from image_end_position onward, plus last_hidden / feature.

Designed for use with the EAGLE3 spec decode prefill in eagle3_worker.py.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Any

import torch

IMAGE_TOKEN_ID = 151655  # Qwen3VL <|image_pad|>


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _hash_tensor(t: torch.Tensor | None) -> int:
    """Stable hash of tensor values (CPU bytes)."""
    if t is None:
        return 0
    return hash(t.detach().cpu().resolve_conj().numpy().tobytes())


# ---------------------------------------------------------------------------
# Cache entries
# ---------------------------------------------------------------------------

class ImageKVEntry:
    """Entry stored in Layer 1 (image KV cache)."""
    __slots__ = ("image_kv", "rope_deltas", "image_end_pos",
                 "pixel_hash", "grid_hash")

    def __init__(
        self,
        image_kv: Any,
        rope_deltas: Any,
        image_end_pos: int,
        pixel_hash: int,
        grid_hash: int,
    ):
        self.image_kv = image_kv          # past_key_values at image_end_pos
        self.rope_deltas = rope_deltas     # rope metadata (may be None)
        self.image_end_pos = image_end_pos
        self.pixel_hash = pixel_hash
        self.grid_hash = grid_hash


class TextPrefixEntry:
    """Entry stored in Layer 2 (text prefix cache)."""
    __slots__ = ("text_kv", "logits", "feature")

    def __init__(self, text_kv: Any, logits: torch.Tensor, feature: torch.Tensor):
        self.text_kv = text_kv            # past_key_values for full sequence
        self.logits = logits              # last-token logits [1, 1, vocab_size]
        self.feature = feature             # eagle3 feature at last position


# ---------------------------------------------------------------------------
# Main cache manager
# ---------------------------------------------------------------------------

class KVPrefixCache:
    """Two-layer KV prefix cache.

    Layer 1: Caches KV for system + image tokens (deterministic by image).
    Layer 2: Caches KV for question tokens (deterministic by image + question).

    Both layers use LRU eviction.
    """

    def __init__(self, max_image_entries: int = 8, max_text_entries: int = 32):
        self.layer1: OrderedDict[tuple[int, int], ImageKVEntry] = OrderedDict()
        self.layer2: OrderedDict[tuple[tuple[int, int], int], TextPrefixEntry] = OrderedDict()
        self.max_image = max_image_entries
        self.max_text = max_text_entries
        self.stats = {
            "layer1_requests": 0,
            "layer1_hits": 0,
            "layer1_stores": 0,
            "layer2_requests": 0,
            "layer2_hits": 0,
            "layer2_stores": 0,
        }

    # ---- Layer 1: Image KV ----

    def _make_layer1_key(self, pixel_values: torch.Tensor | None,
                         image_grid_thw: torch.Tensor | None) -> tuple[int, int]:
        return (_hash_tensor(pixel_values), _hash_tensor(image_grid_thw))

    def lookup_image(self, pixel_values: torch.Tensor | None,
                     image_grid_thw: torch.Tensor | None) -> ImageKVEntry | None:
        """Return cached image KV entry, or None."""
        key = self._make_layer1_key(pixel_values, image_grid_thw)
        self.stats["layer1_requests"] += 1
        entry = self.layer1.get(key)
        if entry is not None:
            self.stats["layer1_hits"] += 1
            self.layer1.move_to_end(key)  # LRU touch
        return entry

    def store_image(self, pixel_values: torch.Tensor | None,
                    image_grid_thw: torch.Tensor | None,
                    image_kv: Any,
                    rope_deltas: Any,
                    image_end_pos: int) -> None:
        """Store KV for image tokens."""
        key = self._make_layer1_key(pixel_values, image_grid_thw)
        if key in self.layer1:
            return
        self._evict_layer1_if_needed()
        self.layer1[key] = ImageKVEntry(
            image_kv, rope_deltas, image_end_pos, key[0], key[1],
        )
        self.stats["layer1_stores"] += 1

    def _evict_layer1_if_needed(self) -> None:
        while len(self.layer1) >= self.max_image:
            self.layer1.popitem(last=False)  # LRU: remove oldest

    # ---- Layer 2: Text Prefix ----

    def _make_layer2_key(self, layer1_entry: ImageKVEntry,
                         question_tokens: torch.Tensor | None) -> tuple:
        image_key = (layer1_entry.pixel_hash, layer1_entry.grid_hash)
        q_hash = _hash_tensor(question_tokens) if question_tokens is not None else 0
        return (image_key, q_hash)

    def lookup_text(self, layer1_entry: ImageKVEntry | None,
                    question_tokens: torch.Tensor) -> TextPrefixEntry | None:
        """Return cached text prefix entry, or None."""
        if layer1_entry is None:
            return None
        key = self._make_layer2_key(layer1_entry, question_tokens)
        self.stats["layer2_requests"] += 1
        entry = self.layer2.get(key)
        if entry is not None:
            self.stats["layer2_hits"] += 1
            self.layer2.move_to_end(key)
        return entry

    def store_text(self, layer1_entry: ImageKVEntry | None,
                   question_tokens: torch.Tensor,
                   text_kv: Any,
                   logits: torch.Tensor,
                   feature: torch.Tensor) -> None:
        """Store KV for question tokens."""
        if layer1_entry is None:
            return
        key = self._make_layer2_key(layer1_entry, question_tokens)
        if key in self.layer2:
            return
        self._evict_layer2_if_needed()
        self.layer2[key] = TextPrefixEntry(text_kv, logits, feature)
        self.stats["layer2_stores"] += 1

    def _evict_layer2_if_needed(self) -> None:
        while len(self.layer2) >= self.max_text:
            self.layer2.popitem(last=False)

    # ---- Stats ----

    def get_stats(self) -> dict[str, int]:
        return dict(self.stats)

    def print_stats(self, prefix: str = "[KVPrefixCache]") -> None:
        s = self.stats
        l1_rate = s["layer1_hits"] / max(s["layer1_requests"], 1)
        l2_rate = s["layer2_hits"] / max(s["layer2_requests"], 1)
        print(
            f"{prefix} L1: {s['layer1_hits']}/{s['layer1_requests']} hits "
            f"({l1_rate * 100:.1f}%) stored={s['layer1_stores']} "
            f"entries={len(self.layer1)} | "
            f"L2: {s['layer2_hits']}/{s['layer2_requests']} hits "
            f"({l2_rate * 100:.1f}%) stored={s['layer2_stores']} "
            f"entries={len(self.layer2)}"
        )


# ---------------------------------------------------------------------------
# Utility: find image boundaries in input_ids
# ---------------------------------------------------------------------------

def find_image_end_pos(input_ids: torch.Tensor) -> int:
    """Return the position (exclusive) after the last image token, or 0."""
    img_mask = (input_ids == IMAGE_TOKEN_ID)
    if not img_mask.any():
        return 0
    last_img_pos = img_mask.nonzero(as_tuple=True)[1][-1].item()
    return last_img_pos + 1


# ---------------------------------------------------------------------------
# "None" cache: disables caching without if/else noise
# ---------------------------------------------------------------------------

class _NullKVPrefixCache(KVPrefixCache):
    """Singleton no-op cache that never stores or hits."""
    def lookup_image(self, pixel_values, image_grid_thw):
        return None
    def store_image(self, *args, **kwargs):
        pass
    def lookup_text(self, layer1_entry, question_tokens):
        return None
    def store_text(self, *args, **kwargs):
        pass


_NULL_CACHE_INSTANCE: _NullKVPrefixCache | None = None


def null_cache() -> KVPrefixCache:
    global _NULL_CACHE_INSTANCE
    if _NULL_CACHE_INSTANCE is None:
        _NULL_CACHE_INSTANCE = _NullKVPrefixCache()
    return _NULL_CACHE_INSTANCE
