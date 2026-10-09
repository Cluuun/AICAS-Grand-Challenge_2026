from __future__ import annotations

import random
from typing import Dict, List

import torch.nn as nn

class ExactQwen3VLText:

    def __init__(self, model: nn.Module, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices

    @property
    def layers(self) -> "nn.ModuleList":
        return self.model.model.language_model.layers

    def remove_layers(self) -> List[int]:
        n = len(self.layers)
        valid = [i for i in self.layer_indices if 0 <= i < n]
        if not valid:
            return []
        self._drop_layers(set(valid))
        return sorted(valid)

    def _drop_layers(self, skip_indices: set):
        lang = self.model.model.language_model
        for idx in sorted(skip_indices, reverse=True):
            del lang.layers[idx]
        for new_idx, layer in enumerate(lang.layers):
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = new_idx
        try:
            self.model.config.get_text_config().num_hidden_layers = len(lang.layers)
        except Exception:
            pass

class ExactQwen3VLVIT:

    def __init__(self, model: nn.Module, block_indices: List[int]):
        self.model = model
        self.block_indices = block_indices

    @property
    def blocks(self) -> "nn.ModuleList":
        return self.model.model.visual.blocks

    def remove_blocks(self) -> List[int]:
        n = len(self.blocks)
        valid = [i for i in self.block_indices if 0 <= i < n]
        if not valid:
            return []
        self._drop_blocks(set(valid))
        return sorted(valid)

    def _drop_blocks(self, skip_indices: set):
        visual = self.model.model.visual
        cfg = visual.config
        old_n = len(visual.blocks)

        new_idx_map: Dict[int, int] = {}
        new_i = 0
        for old_i in range(old_n):
            if old_i not in skip_indices:
                new_idx_map[old_i] = new_i
                new_i += 1

        for idx in sorted(skip_indices, reverse=True):
            del visual.blocks[idx]

        old_ds = list(getattr(cfg, "deepstack_visual_indexes", []) or [])
        if old_ds:
            new_ds = [new_idx_map[di] for di in old_ds if di in new_idx_map]
            try:
                cfg.deepstack_visual_indexes = new_ds
            except Exception:
                pass

        try:
            cfg.depth = len(visual.blocks)
        except Exception:
            pass
