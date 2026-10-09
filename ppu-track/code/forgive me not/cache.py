#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import time
from datasets import load_from_disk
import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling_qwen3_vl
from transformers import StaticCache
import os

import torch

#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
import time
from datasets import load_from_disk
import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling_qwen3_vl
from transformers import StaticCache
import os

import torch

class ASStaticCache(StaticCache):
    def __init__(self, config, max_cache_len, **kwargs):
        super().__init__(config, max_cache_len, **kwargs)
        self._cache_position = None
        self._hidden_size = 2048
        self._device = "cuda:0"
        self._dtype = torch.float16
        self._max_cache_len = max_cache_len

    def reset(self, device: str = "cuda:0"):
        if len(self.layers) == 0:
            return

        for layer in self.layers:
            layer.reset()

            if hasattr(layer, 'cumulative_length'):
                if isinstance(layer.cumulative_length, torch.Tensor):
                    layer.cumulative_length.fill_(self._max_cache_len)

        if not hasattr(self, '_cache_position') or self._cache_position is None:
            self._cache_position = torch.tensor([self._max_cache_len], device=device)
        else:
            self._cache_position.fill_(self._max_cache_len)
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs = None):
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        return k_out, v_out
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()
