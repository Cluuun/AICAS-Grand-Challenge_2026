"""预分配的静态KV Cache，避免每步decode都做torch.cat。"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class StaticCacheLayer(CacheLayerMixin):
    """单层transformer的预分配cache。"""

    is_sliding = False

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False
        self._seq_length = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        batch_size, num_heads, _, head_dim = key_states.shape
        self.keys = torch.zeros(
            batch_size, num_heads, self.max_seq_len, head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.values = torch.zeros(
            batch_size, num_heads, self.max_seq_len, head_dim,
            dtype=self.dtype, device=self.device,
        )
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        seq_len = key_states.shape[2]
        pos = self._seq_length

        # 把新KV直接写到预分配的位置
        self.keys[:, :, pos : pos + seq_len, :].copy_(key_states)
        self.values[:, :, pos : pos + seq_len, :].copy_(value_states)
        self._seq_length = pos + seq_len

        # 返回到当前位置为止的全部cache
        return (
            self.keys[:, :, : self._seq_length, :],
            self.values[:, :, : self._seq_length, :],
        )

    def get_seq_length(self) -> int:
        return self._seq_length

    def get_max_cache_shape(self) -> int:
        return self.max_seq_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        kv_offset = 0
        kv_length = self._seq_length + query_length
        return kv_length, kv_offset


class StaticKVCache(Cache):
    """预分配的KV cache，不用每次decode都cat，兼容HuggingFace的Cache协议。"""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.float16,
        device: str | torch.device = "cuda",
    ):
        # 直接创建静态层，不用懒加载
        self._static_layers = [
            StaticCacheLayer(max_seq_len) for _ in range(num_layers)
        ]
        self._max_seq_len = max_seq_len
        self._num_layers = num_layers
        super().__init__()
        # 用我们的静态层替换掉默认的layers列表
        self.layers = self._static_layers

    def reset(self):
        for layer in self.layers:
            layer._seq_length = 0
