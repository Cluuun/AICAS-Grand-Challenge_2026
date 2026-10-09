"""预分配的静态kv cache 解码时不用torch.cat"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class GraphStaticCacheLayer(CacheLayerMixin):
    """预分配的cache层"""

    is_sliding = False
    _graph_mode = False

    def __init__(self, max_seq_len, num_kv_heads, head_dim, dtype, device):
        self.max_seq_len = max_seq_len
        self.keys = torch.zeros(1, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        self.values = torch.zeros(1, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device)
        # 每层自己的position 用于普通模式(prefill)
        self.position = torch.tensor([0], dtype=torch.long, device=device)
        # 用python int追踪位置 不用.item()
        self._pos_val = 0
        # 指向上层cache的引用 构造完再设
        self._cache_ref = None

    def lazy_initialization(self, key_states, value_states):
        pass

    def update(self, key_states, value_states, *args, **kwargs):
        seq_len = key_states.shape[2]

        if self._graph_mode:
            # 图模式: 用共享position 不自增
            pos = self._cache_ref.shared_position
            self.keys.index_copy_(2, pos, key_states)
            self.values.index_copy_(2, pos, value_states)
            return self.keys, self.values
        else:
            # 普通模式: 拷贝到对应slot 推进position 返回sized view
            pos = self._pos_val
            end = pos + seq_len
            self.keys.narrow(2, pos, seq_len).copy_(key_states)
            self.values.narrow(2, pos, seq_len).copy_(value_states)
            self.position.fill_(end)
            self._pos_val = end
            return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_seq_length(self):
        if self._graph_mode and self._cache_ref is not None:
            return self._cache_ref._seq_len
        return self._pos_val

    def get_max_cache_shape(self):
        return self.max_seq_len

    def get_mask_sizes(self, query_length):
        return self.get_seq_length() + query_length, 0

    def prefetch(self):
        pass

    def offload(self):
        pass

    def reset(self):
        self.position.fill_(0)
        self._pos_val = 0
        # 不清keys/values prefill会覆盖
        # seq_len以外的老数据flash_attn不会访问到


class GraphStaticKVCache(Cache):
    """带外部共享position的静态KV cache 用于CUDA graph"""

    def __init__(self, num_layers, num_kv_heads=8, head_dim=128,
                 max_seq_len=4096, dtype=torch.float16, device="cuda"):
        layers = [
            GraphStaticCacheLayer(max_seq_len, num_kv_heads, head_dim, dtype, device)
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)
        # 图模式用的共享position 每次replay前外部更新
        self.shared_position = torch.tensor([0], dtype=torch.long, device=device)
        self._max_seq_len = max_seq_len
        self._graph_mode = False
        # 用python int追踪seq_len 对CUDA graph和torch.compile安全
        self._seq_len = 0
        # 给每层设上反向引用
        for layer in layers:
            layer._cache_ref = self

    def enable_graph_mode(self):
        self._graph_mode = True
        for layer in self.layers:
            layer._graph_mode = True

    def disable_graph_mode(self):
        self._graph_mode = False
        for layer in self.layers:
            layer._graph_mode = False

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].get_seq_length()

    def reset(self):
        self._seq_len = 0
        for layer in self.layers:
            layer.reset()
