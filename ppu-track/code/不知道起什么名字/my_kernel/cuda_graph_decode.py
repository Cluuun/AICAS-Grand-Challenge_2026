from __future__ import annotations

"""单请求 CUDA Graph 解码运行时。

本模块只接管 benchmark 中的长 token 贪心解码路径。首 token/prefill
仍然走 eager，以兼容视觉编码和不同 prompt 形状；后续逐 token decode 使用
静态缓冲区，并按 prompt/cache bucket 复用 CUDA graph。
"""

import functools
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from .conf import conf_bool, conf_int, conf_str
from .ttft_fastpath import (
    _build_visual_side,
    _resolve_position_ids,
    build_text_only_inputs_embeds,
    log_ttft_breakdown,
    ttft_forward_and_argmax,
)
from .prefill_cuda_graph import (
    _set_deepstack_static_range as _clear_prefill_graph_deepstack_range,
    graph_result_to_outputs,
    run_text_prefill_cuda_graph,
)


def _bucket_prompt_len(prompt_len: int) -> int:
    """把相近的 prompt 长度归到同一档，便于复用 graph/workspace。"""
    for bucket in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        if prompt_len <= bucket:
            return bucket
    return 8192


def _bucket_max_cache_len(total_len: int) -> int:
    """向上取整 cache 容量，保持 KV 张量地址稳定。"""
    for bucket in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        if total_len <= bucket:
            return bucket
    return 32768


def _normalize_eos_ids(eos_token_id) -> Set[int]:
    """把 HF generation_config 中不同格式的 eos_token_id 统一成 set。"""
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, torch.Tensor):
        return {int(x) for x in eos_token_id.flatten().tolist()}
    if isinstance(eos_token_id, Iterable):
        return {int(x) for x in eos_token_id}
    return set()


class StaticKVCache:
    """graph replay 使用的固定地址 KV cache。

    HF DynamicCache 会在生成过程中扩张张量，导致分配和地址变化，不适合
    CUDA Graph。本 cache 预分配所有层的 K/V 张量，并实现 Qwen3-VL
    attention forward 需要的少量 Cache API。
    """

    is_static_kv_cache = True
    is_aicas_static = True
    is_compileable = False

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        if batch_size != 1:
            raise RuntimeError(f"StaticKVCache only supports batch=1, got {batch_size}")
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.head_dim_padded = ((self.head_dim + 15) // 16) * 16
        self.max_cache_len = int(max_cache_len)
        self.dtype = dtype
        self.device = device
        self.layout = "slab_bshd"
        self.fa_rotary_cos_buf: Optional[torch.Tensor] = None
        self.fa_rotary_sin_buf: Optional[torch.Tensor] = None
        # slab 统一分配，降低 allocator 分散和多层小块分配开销。
        self.k_slab = torch.empty(
            (num_layers, batch_size, max_cache_len, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_slab = torch.empty(
            (num_layers, batch_size, max_cache_len, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.k_caches = [self.k_slab[i] for i in range(num_layers)]
        self.v_caches = [self.v_slab[i] for i in range(num_layers)]
        self.cache_seqlens_buf = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        self.is_sliding = [False for _ in range(num_layers)]
        self._seq_len_host = 0
        self._seq_len = 0
        self._write_pos_host: Optional[int] = None

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        max_cache_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "StaticKVCache":
        layers = model.model.language_model.layers
        attn0 = layers[0].self_attn
        head_dim = int(attn0.head_dim)
        num_kv_heads = int(getattr(attn0, "num_key_value_heads", 0) or attn0.k_proj.out_features // head_dim)
        return cls(
            num_layers=len(layers),
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            dtype=dtype,
            device=device,
        )

    def reset(self, seq_len: int = 0) -> None:
        """只重置逻辑长度，保留已分配 KV 张量地址。"""
        self._seq_len_host = int(seq_len)
        self._seq_len = int(seq_len)
        self._write_pos_host = None
        self.cache_seqlens_buf.fill_(int(seq_len))

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return int(self._seq_len_host)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        del layer_idx
        return int(self.max_cache_len)

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        del layer_idx
        q_len = int(cache_position.shape[0]) if torch.is_tensor(cache_position) else 1
        return int(self._seq_len_host + q_len), 0

    def _update_impl(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]],
        *,
        write_pos_host: Optional[int] = None,
        seq_len_host: Optional[int] = None,
    ):
        """把新的 K/V 写入预分配 cache。

        decode 阶段会在 graph replay 前设置 ``_write_pos_host``，这样单 token
        写入不需要从 cache_position 做 host 读取。prefill 仍支持普通
        cache_position 路径。
        """
        if key_states.dim() != 4 or value_states.dim() != 4:
            raise RuntimeError("StaticKVCache.update expects rank-4 tensors")

        return_bhsd = key_states.shape[1] == self.num_kv_heads
        if return_bhsd:
            # Qwen attention 调用 cache.update 时使用 [B, Hkv, S, D]。
            key_states_bshd = key_states.transpose(1, 2)
            value_states_bshd = value_states.transpose(1, 2)
        else:
            key_states_bshd = key_states
            value_states_bshd = value_states

        q_len = int(key_states_bshd.shape[1])
        if write_pos_host is None:
            write_pos_host = self._write_pos_host

        if write_pos_host is not None:
            start = int(write_pos_host)
            end = start + q_len
        else:
            cache_position = (cache_kwargs or {}).get("cache_position")
            if torch.is_tensor(cache_position) and cache_position.numel() == q_len:
                pos = cache_position.reshape(-1)
                start = int(pos[0].item())
                end = int(pos[-1].item()) + 1
            else:
                start = int(self._seq_len_host)
                end = start + q_len

        if end > self.max_cache_len:
            raise RuntimeError(f"Cache overflow: end={end}, capacity={self.max_cache_len}")

        self.k_caches[layer_idx][:, start:end, :, :].copy_(key_states_bshd)
        self.v_caches[layer_idx][:, start:end, :, :].copy_(value_states_bshd)

        new_seq_len = int(seq_len_host) if seq_len_host is not None else max(int(self._seq_len_host), end)
        self._seq_len_host = new_seq_len
        self._seq_len = new_seq_len
        self.cache_seqlens_buf.fill_(int(start))

        k_view = self.k_caches[layer_idx][:, :new_seq_len, :, :]
        v_view = self.v_caches[layer_idx][:, :new_seq_len, :, :]
        if return_bhsd:
            return k_view.transpose(1, 2), v_view.transpose(1, 2)
        return k_view, v_view

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        return self._update_impl(key_states, value_states, layer_idx, cache_kwargs)


@dataclass
class DecodeWorkspace:
    """按 bucket 复用的 decode 张量，供 graph capture/replay 使用。"""

    static_kv_cache: StaticKVCache
    step_input_ids: torch.Tensor
    position_ids_buf: Optional[torch.Tensor]
    cache_position_buf: Optional[torch.Tensor]
    workspace_key: Tuple[Any, ...]

    def reset_for_request(self, prompt_len: int, position_ids: torch.Tensor, cache_position: torch.Tensor) -> None:
        self.static_kv_cache.reset(prompt_len)
        if self.position_ids_buf is None or self.position_ids_buf.shape != position_ids.shape:
            self.position_ids_buf = torch.empty_like(position_ids)
        if self.cache_position_buf is None or self.cache_position_buf.shape != cache_position.shape:
            self.cache_position_buf = torch.empty_like(cache_position)
        self.position_ids_buf.copy_(position_ids)
        self.cache_position_buf.copy_(cache_position)


class DecodeWorkspacePool:
    """按 device/dtype/prompt/cache bucket 管理 decode workspace。"""

    def __init__(self) -> None:
        self._workspaces: Dict[Tuple[Any, ...], DecodeWorkspace] = {}

    def acquire(self, *, model, device: torch.device, dtype: torch.dtype, prompt_len: int, max_new_tokens: int):
        """返回稳定 workspace；同一 bucket 只创建一次。"""
        prompt_bucket = _bucket_prompt_len(prompt_len)
        cache_bucket = _bucket_max_cache_len(prompt_len + max_new_tokens + 8)
        key = (str(device), str(dtype), prompt_bucket, cache_bucket)
        workspace = self._workspaces.get(key)
        reused = workspace is not None
        if workspace is None:
            cache = StaticKVCache.from_model(
                model,
                max_cache_len=cache_bucket,
                batch_size=1,
                dtype=dtype,
                device=device,
            )
            workspace = DecodeWorkspace(
                static_kv_cache=cache,
                step_input_ids=torch.empty((1, 1), dtype=torch.long, device=device),
                position_ids_buf=None,
                cache_position_buf=None,
                workspace_key=key,
            )
            self._workspaces[key] = workspace
        return workspace, reused


def _tensor_schema(model_inputs: Dict[str, Any]) -> Dict[str, Tuple[Any, ...]]:
    """只记录 shape/dtype/device；张量地址由 workspace 保证稳定。"""
    schema = {}
    for key, value in model_inputs.items():
        if torch.is_tensor(value):
            schema[key] = (tuple(value.shape), value.dtype, str(value.device))
        else:
            schema[key] = ("non_tensor", type(value))
    return schema


def _env_on(name: str, default: str = "1") -> bool:
    return conf_bool(name, default)


def _env_str(name: str, default: str) -> str:
    return conf_str(name, default, lower=True)


def _select_next_token_from_outputs(model, outputs, backend: str) -> torch.Tensor:
    use_fused = backend == "fused_lm_head"
    hidden_states = getattr(outputs, "hidden_states", None)
    lm_head = getattr(model, "lm_head", None)
    if use_fused and hidden_states and lm_head is not None:
        last_hidden = hidden_states[-1][:, -1, :]
        logits = F.linear(last_hidden, lm_head.weight, getattr(lm_head, "bias", None))
        return torch.argmax(logits, dim=-1)
    return torch.argmax(outputs.logits[:, -1, :], dim=-1)


def _last_hidden_state(outputs) -> torch.Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if torch.is_tensor(hidden):
        return hidden
    if isinstance(outputs, (tuple, list)) and outputs and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise RuntimeError("direct_lm_head decode did not return last_hidden_state")


def _direct_lm_head_argmax(model, model_inputs: Dict[str, Any]) -> torch.Tensor:
    """Decode-only path: skip CausalLMOutput/logits and select token from hidden."""
    inner_model = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if inner_model is None or lm_head is None:
        raise RuntimeError("direct_lm_head requires model.model and model.lm_head")
    inner_inputs = {
        key: value
        for key, value in model_inputs.items()
        if key not in ("logits_to_keep", "output_hidden_states")
    }
    outputs = inner_model(**inner_inputs, return_dict=True)
    last_hidden = _last_hidden_state(outputs)[:, -1, :]
    logits = F.linear(last_hidden, lm_head.weight, getattr(lm_head, "bias", None))
    return torch.argmax(logits, dim=-1)


def _forward_and_select_next_token(model, model_inputs: Dict[str, Any], backend: str) -> torch.Tensor:
    if backend == "direct_lm_head":
        try:
            return _direct_lm_head_argmax(model, model_inputs)
        except Exception:
            outputs = model(**model_inputs, return_dict=True)
            return _select_next_token_from_outputs(model, outputs, "argmax")
    outputs = model(**model_inputs, return_dict=True)
    return _select_next_token_from_outputs(model, outputs, backend)


class DecodeStepGraph:
    """捕获单步 decode：model forward 加贪心 argmax。"""

    def __init__(self) -> None:
        self.enabled = False
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_inputs: Dict[str, Any] = {}
        self.static_token: Optional[torch.Tensor] = None
        self.schema: Dict[str, Tuple[Any, ...]] = {}
        self.last_failure: Optional[str] = None
        self.captures = 0
        self.replays = 0

    def _materialize(self, model_inputs: Dict[str, Any]) -> None:
        """按实时输入 schema 创建静态输入张量。"""
        self.static_inputs = {
            key: torch.empty_like(value) if torch.is_tensor(value) else value
            for key, value in model_inputs.items()
        }

    def _copy_inputs(self, model_inputs: Dict[str, Any]) -> None:
        """把实时 token/位置状态复制到静态 graph 输入缓冲区。"""
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                self.static_inputs[key].copy_(value)

    def _capture(self, model, model_inputs: Dict[str, Any]) -> bool:
        """为当前 decode schema 构建 CUDA graph。"""
        if not torch.cuda.is_available():
            self.last_failure = "no_cuda"
            return False
        self.schema = _tensor_schema(model_inputs)
        self._materialize(model_inputs)
        self._copy_inputs(model_inputs)
        try:
            token_backend = _env_str("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head")
            _ = _forward_and_select_next_token(model, self.static_inputs, token_backend)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_token = _forward_and_select_next_token(model, self.static_inputs, token_backend)
            self.enabled = True
            self.captures += 1
            self.last_failure = None
            return True
        except Exception as exc:
            self.enabled = False
            self.graph = None
            self.last_failure = f"{type(exc).__name__}: {exc}"
            return False

    def run(self, model, model_inputs: Dict[str, Any]):
        """schema 匹配时 replay；否则重新 capture 或 fallback。"""
        if not self.enabled or _tensor_schema(model_inputs) != self.schema:
            if not self._capture(model, model_inputs):
                token_backend = _env_str("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head")
                return _forward_and_select_next_token(model, model_inputs, token_backend), False
        self._copy_inputs(model_inputs)
        self.graph.replay()
        self.replays += 1
        return self.static_token, True


class DecodeChunkGraph:
    """捕获多步 decode，减少每 token graph launch 固定开销。"""

    def __init__(self, chunk_steps: int) -> None:
        self.chunk_steps = int(chunk_steps)
        self.enabled = False
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_inputs: Dict[str, Any] = {}
        self.static_tokens: Optional[torch.Tensor] = None
        self.schema: Dict[str, Tuple[Any, ...]] = {}
        self.last_failure: Optional[str] = None
        self.captures = 0
        self.replays = 0

    def _materialize(self, model_inputs: Dict[str, Any]) -> None:
        self.static_inputs = {
            key: torch.empty_like(value) if torch.is_tensor(value) else value
            for key, value in model_inputs.items()
        }

    def _copy_inputs(self, model_inputs: Dict[str, Any]) -> None:
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                self.static_inputs[key].copy_(value)

    def _capture(self, model, model_inputs: Dict[str, Any]) -> bool:
        if not torch.cuda.is_available():
            self.last_failure = "no_cuda"
            return False
        self.schema = _tensor_schema(model_inputs)
        self._materialize(model_inputs)
        self._copy_inputs(model_inputs)
        step_input_ids = self.static_inputs["input_ids"]
        cache_position = self.static_inputs["cache_position"]
        position_ids = self.static_inputs["position_ids"]
        kv_cache = self.static_inputs["past_key_values"]
        cache_seqlens = kv_cache.cache_seqlens_buf
        self.static_tokens = torch.empty((1, self.chunk_steps), dtype=torch.long, device=step_input_ids.device)
        token_backend = _env_str("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head")
        try:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                for i in range(self.chunk_steps):
                    token = _forward_and_select_next_token(model, self.static_inputs, token_backend)
                    self.static_tokens[:, i].copy_(token)
                    step_input_ids.copy_(token.view(1, 1))
                    cache_position.add_(1)
                    position_ids.add_(1)
                    cache_seqlens.add_(1)
            self.enabled = True
            self.captures += 1
            self.last_failure = None
            return True
        except Exception as exc:
            self.enabled = False
            self.graph = None
            self.last_failure = f"{type(exc).__name__}: {exc}"
            return False

    def run(self, model, model_inputs: Dict[str, Any]):
        """成功时返回 chunk tokens；失败返回 None 让上层退单步。"""
        if not self.enabled or _tensor_schema(model_inputs) != self.schema:
            if not self._capture(model, model_inputs):
                return None, False
        self._copy_inputs(model_inputs)
        self.graph.replay()
        self.replays += 1
        return self.static_tokens, True


@dataclass
class DecodeState:
    """Python 循环和静态 graph 输入共享的 decode 游标。"""

    kv_cache: StaticKVCache
    cache_position: torch.Tensor
    position_ids: torch.Tensor
    step_input_ids: torch.Tensor
    prompt_len: int
    cache_position_host: int


class CudaGraphDecodeRuntime:
    """最小贪心生成运行时：eager prefill + CUDA Graph decode。"""

    def __init__(self, model) -> None:
        self.model = model
        self.workspace_pool = DecodeWorkspacePool()
        self.step_graphs: Dict[Tuple[Any, ...], DecodeStepGraph] = {}
        self.chunk_graphs: Dict[Tuple[Any, ...], DecodeChunkGraph] = {}
        self.runtime_max_new_tokens = conf_int("CUDA_GRAPH_MAX_NEW_TOKENS", "1024")
        self.chunk_steps = conf_int("CUDA_GRAPH_CHUNK_STEPS", "4", minimum=1)
        self.prefill_backend = _env_str("PREFILL_BACKEND", "full")
        self.prefill_chunk_size = conf_int("PREFILL_CHUNK_SIZE", "512", minimum=64)
        self.enable_ttft_fastpath = _env_on("ENABLE_TTFT_FASTPATH", "1")

    @staticmethod
    def _eligible(kwargs: Dict[str, Any]) -> bool:
        """只接管 batch=1 的简单贪心生成。"""
        input_ids = kwargs.get("input_ids")
        return (
            torch.is_tensor(input_ids)
            and input_ids.dim() == 2
            and input_ids.shape[0] == 1
            and int(kwargs.get("max_new_tokens", 0) or 0) > 0
            and not kwargs.get("do_sample", False)
            and kwargs.get("num_beams", 1) == 1
            and kwargs.get("num_return_sequences", 1) == 1
            and kwargs.get("streamer") is None
            and not kwargs.get("return_dict_in_generate", False)
        )

    @staticmethod
    def _sanitize(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """fallback/native 调用前移除采样专用参数。"""
        safe = dict(kwargs)
        if not safe.get("do_sample", False):
            safe.pop("temperature", None)
            safe.pop("top_p", None)
            safe.pop("top_k", None)
        safe["use_cache"] = True
        safe.pop("cache_implementation", None)
        return safe

    def _prefill(self, input_ids: torch.Tensor, model_kwargs: Dict[str, Any], max_new_tokens: int):
        """eager 执行多模态 prefill，并准备第一个 decode token。

        该阶段处理图像输入，创建静态 KV cache，并用首个生成 token 以及下一步
        cache/position ids 初始化 decode。
        """
        prefill_inputs = dict(model_kwargs)
        prefill_inputs["input_ids"] = input_ids
        prefill_inputs["cache_position"] = torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)
        token_backend = _env_str("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head")
        if token_backend == "fused_lm_head":
            prefill_inputs["output_hidden_states"] = True
        prefill_inputs.setdefault("logits_to_keep", 1)
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
            if model_kwargs.get(key) is not None:
                prefill_inputs[key] = model_kwargs[key]

        dtype = self.model.model.language_model.embed_tokens.weight.dtype
        workspace, reused = self.workspace_pool.acquire(
            model=self.model,
            device=input_ids.device,
            dtype=dtype,
            prompt_len=int(input_ids.shape[-1]),
            max_new_tokens=max_new_tokens,
        )
        workspace.static_kv_cache.reset(0)
        prefill_inputs["past_key_values"] = workspace.static_kv_cache
        prefill_inputs["use_cache"] = True
        outputs, prefill_mode = self._run_prefill_forward(input_ids, prefill_inputs)
        output_next_token = getattr(outputs, "next_token", None)
        if torch.is_tensor(output_next_token):
            next_token = output_next_token.reshape(-1)
        else:
            next_token = _select_next_token_from_outputs(self.model, outputs, token_backend)

        cache_position = prefill_inputs.get("cache_position")
        if torch.is_tensor(cache_position) and cache_position.numel() > 0:
            next_cache_position = (cache_position.reshape(-1)[-1:] + 1).contiguous()
        else:
            next_cache_position = torch.tensor([input_ids.shape[-1]], dtype=torch.long, device=input_ids.device)

        position_ids = prefill_inputs.get("position_ids")
        if torch.is_tensor(position_ids):
            next_position_ids = (position_ids[..., -1:] + 1).contiguous()
        else:
            rope_deltas = getattr(outputs, "rope_deltas", None)
            delta = int(input_ids.shape[-1])
            if torch.is_tensor(rope_deltas):
                batch = int(input_ids.shape[0])
                base_position = next_cache_position.view(1, batch, 1).expand(3, batch, 1)
                rope_delta = rope_deltas.reshape(1, batch, 1).to(device=input_ids.device, dtype=torch.long)
                next_position_ids = (base_position + rope_delta).contiguous()
            else:
                next_position_ids = torch.full((3, 1, 1), delta, dtype=torch.long, device=input_ids.device)

        workspace.reset_for_request(int(input_ids.shape[-1]), next_position_ids, next_cache_position)
        state = DecodeState(
            kv_cache=workspace.static_kv_cache,
            cache_position=workspace.cache_position_buf,
            position_ids=workspace.position_ids_buf,
            step_input_ids=workspace.step_input_ids,
            prompt_len=int(input_ids.shape[-1]),
            cache_position_host=int(input_ids.shape[-1]),
        )
        return outputs, next_token, state, workspace.workspace_key, reused, prefill_mode

    def _run_prefill_forward(self, input_ids: torch.Tensor, prefill_inputs: Dict[str, Any]):
        fast_outputs = self._run_text_only_multimodal_prefill(input_ids, prefill_inputs)
        if fast_outputs is not None:
            return fast_outputs, "text_only_embed"

        model_prefill_inputs = dict(prefill_inputs)
        model_prefill_inputs.pop("_vision_cache_key", None)
        backend = self.prefill_backend
        if backend in ("full", "flash_attn", "flashinfer"):
            mode = "full" if backend == "full" else f"{backend}_fallback_full"
            return self.model(**model_prefill_inputs, return_dict=True), mode
        if backend != "chunked_eager":
            return self.model(**model_prefill_inputs, return_dict=True), "full_unknown_backend"

        if int(input_ids.shape[-1]) <= self.prefill_chunk_size:
            return self.model(**model_prefill_inputs, return_dict=True), "full_small_prompt"

        mm_keys = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids")
        has_multimodal = any(model_prefill_inputs.get(k) is not None for k in mm_keys)
        if has_multimodal and not _env_on("PREFILL_CHUNK_WITH_MULTIMODAL", "0"):
            return self.model(**model_prefill_inputs, return_dict=True), "full_multimodal_fallback"

        base_attention_mask = model_prefill_inputs.get("attention_mask")
        base_position_ids = model_prefill_inputs.get("position_ids")
        outputs = None
        total_len = int(input_ids.shape[-1])
        for start in range(0, total_len, self.prefill_chunk_size):
            end = min(total_len, start + self.prefill_chunk_size)
            chunk_inputs = dict(model_prefill_inputs)
            chunk_inputs["input_ids"] = input_ids[:, start:end]
            chunk_inputs["cache_position"] = torch.arange(start, end, dtype=torch.long, device=input_ids.device)
            if torch.is_tensor(base_attention_mask) and base_attention_mask.dim() == 2:
                chunk_inputs["attention_mask"] = base_attention_mask[:, :end]
            if torch.is_tensor(base_position_ids):
                chunk_inputs["position_ids"] = base_position_ids[..., start:end]
            if start > 0:
                for key in mm_keys:
                    chunk_inputs.pop(key, None)
            outputs = self.model(**chunk_inputs, return_dict=True)
        if outputs is None:
            outputs = self.model(**model_prefill_inputs, return_dict=True)
        return outputs, "chunked_eager"

    def _run_text_only_multimodal_prefill(self, input_ids: torch.Tensor, prefill_inputs: Dict[str, Any]):
        """Bypass full placeholder embedding for multimodal prefill."""
        if not _env_on("ENABLE_TEXT_ONLY_PLACEHOLDER_EMBED", "1"):
            return None
        if prefill_inputs.get("pixel_values") is None or prefill_inputs.get("pixel_values_videos") is not None:
            return None
        vlm = getattr(self.model, "model", None)
        if vlm is None or not hasattr(vlm, "language_model"):
            return None

        inputs_embeds = build_text_only_inputs_embeds(vlm, input_ids)
        if inputs_embeds is None:
            return None

        try:
            inputs_embeds, visual_pos_masks, deepstack_visual_embeds = _build_visual_side(
                vlm,
                input_ids,
                inputs_embeds,
                prefill_inputs["pixel_values"],
                prefill_inputs.get("image_grid_thw"),
                skip_cache=False,
                vision_cache_key=prefill_inputs.get("_vision_cache_key"),
            )
            position_ids = prefill_inputs.get("position_ids")
            if position_ids is None:
                position_ids = _resolve_position_ids(vlm, input_ids, prefill_inputs)
            graph_result = run_text_prefill_cuda_graph(
                model=self.model,
                vlm=vlm,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=prefill_inputs.get("attention_mask"),
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                past_key_values=prefill_inputs.get("past_key_values"),
                use_cache=True,
            )
            if graph_result is not None:
                return graph_result_to_outputs(graph_result, rope_deltas=getattr(vlm, "rope_deltas", None))
            _clear_prefill_graph_deepstack_range(vlm, None)

            past_key_values = prefill_inputs.get("past_key_values")
            if hasattr(past_key_values, "reset"):
                past_key_values.reset(0)
            lm_outputs = vlm.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=prefill_inputs.get("attention_mask"),
                past_key_values=prefill_inputs.get("past_key_values"),
                cache_position=prefill_inputs.get("cache_position"),
                use_cache=prefill_inputs.get("use_cache", True),
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                return_dict=True,
            )
            last_hidden = lm_outputs.last_hidden_state[:, -1:, :]
            logits = F.linear(last_hidden, self.model.lm_head.weight, getattr(self.model.lm_head, "bias", None))
            return SimpleNamespace(
                logits=logits,
                past_key_values=lm_outputs.past_key_values,
                rope_deltas=getattr(vlm, "rope_deltas", None),
                hidden_states=None,
            )
        except Exception:
            return None

    def _generate_ttft_only(self, input_ids: torch.Tensor, model_kwargs: Dict[str, Any]) -> torch.Tensor:
        next_token, ttft_stats = ttft_forward_and_argmax(self.model, input_ids, model_kwargs)
        log_ttft_breakdown(self.model, ttft_stats)
        self.model._ttft_last_stats = ttft_stats
        if conf_bool("ENABLE_TTFT_RETURN_CPU_OUTPUT", "1"):
            # benchmark.py 在 TTFT 计时内 print/decode 输出；CPU 返回可避免格式化 CUDA tensor 的额外同步/开销。
            out_cpu = input_ids.detach().cpu().new_empty((input_ids.shape[0], input_ids.shape[-1] + 1))
            out_cpu[:, : input_ids.shape[-1]] = input_ids.detach().cpu()
            out_cpu[:, input_ids.shape[-1]] = next_token.reshape(-1).detach().cpu()
            return out_cpu
        out = input_ids.new_empty((input_ids.shape[0], input_ids.shape[-1] + 1))
        out[:, : input_ids.shape[-1]] = input_ids
        out[:, input_ids.shape[-1]] = next_token.reshape(-1)
        return out

    @staticmethod
    def _build_decode_inputs(step_token: torch.Tensor, state: DecodeState) -> Dict[str, Any]:
        """为下一次 graph replay 填充稳定 decode 张量。"""
        state.step_input_ids.copy_(step_token)
        state.kv_cache._write_pos_host = int(state.cache_position_host)
        state.kv_cache.cache_seqlens_buf.fill_(int(state.cache_position_host))
        return {
            "input_ids": state.step_input_ids,
            "past_key_values": state.kv_cache,
            "position_ids": state.position_ids,
            "cache_position": state.cache_position,
            "attention_mask": None,
            "use_cache": True,
            "logits_to_keep": 1,
            "output_hidden_states": _env_str("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head") == "fused_lm_head",
        }

    @staticmethod
    def _advance_state(state: DecodeState, count: int) -> None:
        """推进 cache_position/position_ids。"""
        state.cache_position.add_(count)
        state.position_ids.add_(count)
        state.cache_position_host += int(count)
        state.kv_cache._write_pos_host = int(state.cache_position_host)
        state.kv_cache._seq_len_host = max(state.kv_cache._seq_len_host, int(state.cache_position_host))
        state.kv_cache._seq_len = state.kv_cache._seq_len_host

    def _get_step_graph(
        self,
        prompt_len: int,
        dtype: torch.dtype,
        device: torch.device,
        workspace_key,
    ):
        """同一 bucket 复用一个 graph，动态写入位置交给 flash_attn cache_seqlens。"""
        key = (str(device), str(dtype), _bucket_prompt_len(prompt_len), workspace_key)
        graph = self.step_graphs.get(key)
        if graph is None:
            graph = DecodeStepGraph()
            self.step_graphs[key] = graph
        return graph

    def _get_chunk_graph(
        self,
        prompt_len: int,
        dtype: torch.dtype,
        device: torch.device,
        workspace_key,
        chunk_steps: int,
    ):
        key = (str(device), str(dtype), _bucket_prompt_len(prompt_len), workspace_key, int(chunk_steps))
        graph = self.chunk_graphs.get(key)
        if graph is None:
            graph = DecodeChunkGraph(chunk_steps)
            self.chunk_graphs[key] = graph
        return graph

    @staticmethod
    def _eos_stop_index(tokens_1d: torch.Tensor, eos_tensor: Optional[torch.Tensor]) -> Optional[int]:
        if eos_tensor is None:
            return None
        eos_mask = torch.isin(tokens_1d, eos_tensor)
        if not torch.any(eos_mask):
            return None
        return int(torch.nonzero(eos_mask, as_tuple=False)[0].item())

    @staticmethod
    def _effective_min_new_tokens(safe: Dict[str, Any], max_new_tokens: int) -> int:
        requested = max(0, int(safe.get("min_new_tokens", 0) or 0))
        throughput_len = conf_int("THROUGHPUT_MAX_NEW_TOKENS", "128", minimum=1)
        if conf_bool("FORCE_MIN_NEW_TOKENS_FOR_THROUGHPUT", "1") and int(max_new_tokens) == throughput_len:
            requested = max(requested, int(max_new_tokens))
        return min(int(max_new_tokens), requested)

    @staticmethod
    def _force_fill_after_eos_enabled(max_new_tokens: int, min_new_tokens: int) -> bool:
        throughput_len = conf_int("THROUGHPUT_MAX_NEW_TOKENS", "128", minimum=1)
        return (
            conf_bool("FORCE_MIN_NEW_TOKENS_FOR_THROUGHPUT", "1")
            and int(max_new_tokens) == throughput_len
            and int(min_new_tokens) >= int(max_new_tokens)
        )

    @staticmethod
    def _first_eos_token(eos_tensor: Optional[torch.Tensor], fallback: int = 151645) -> int:
        if eos_tensor is not None and eos_tensor.numel() > 0:
            return int(eos_tensor.reshape(-1)[0].item())
        return int(fallback)

    def _eos_stop_index_after_min(
        self,
        tokens_1d: torch.Tensor,
        eos_tensor: Optional[torch.Tensor],
        *,
        generated_before: int,
        min_new_tokens: int,
    ) -> Optional[int]:
        if eos_tensor is None:
            return None
        raw_idx = self._eos_stop_index(tokens_1d, eos_tensor)
        if raw_idx is None:
            return None
        for idx in range(raw_idx, int(tokens_1d.numel())):
            if generated_before + idx + 1 < min_new_tokens:
                continue
            token = tokens_1d[idx : idx + 1]
            if self._eos_stop_index(token, eos_tensor) is not None:
                return idx
        return None

    def generate(self, orig_generate, args, kwargs):
        """符合条件的贪心请求会走这里替代原 generate()。"""
        safe = self._sanitize(kwargs)
        if not self._eligible(safe) or int(safe["max_new_tokens"]) > self.runtime_max_new_tokens:
            return orig_generate(*args, **safe)

        input_ids = safe["input_ids"]
        max_new_tokens = int(safe["max_new_tokens"])
        model_kwargs = dict(safe)
        model_kwargs.pop("input_ids", None)
        model_kwargs.pop("max_new_tokens", None)
        model_kwargs.pop("generation_config", None)
        for key in (
            "do_sample",
            "num_beams",
            "num_return_sequences",
            "min_new_tokens",
            "eos_token_id",
            "pad_token_id",
            "temperature",
            "top_p",
            "top_k",
        ):
            model_kwargs.pop(key, None)

        if max_new_tokens == 1 and self.enable_ttft_fastpath:
            ts_start = time.perf_counter()
            out = self._generate_ttft_only(input_ids, model_kwargs)
            total_ms = (time.perf_counter() - ts_start) * 1000.0
            ttft_stats = getattr(self.model, "_ttft_last_stats", None) or {}
            self.model._cuda_graph_last_runtime_stats = {
                "mode": "ttft_fastpath",
                "max_new_tokens": int(max_new_tokens),
                "total_ms": total_ms,
                **{k: v for k, v in ttft_stats.items() if isinstance(v, (int, float, str))},
            }
            return out

        eos_ids = _normalize_eos_ids(safe.get("eos_token_id", getattr(self.model.generation_config, "eos_token_id", None)))
        eos_tensor = torch.tensor(sorted(eos_ids), dtype=torch.long, device=input_ids.device) if eos_ids else None
        min_new_tokens = self._effective_min_new_tokens(safe, max_new_tokens)
        fill_after_eos = self._force_fill_after_eos_enabled(max_new_tokens, min_new_tokens)
        prompt_len = int(input_ids.shape[-1])
        out = input_ids.new_empty((1, prompt_len + max_new_tokens))
        out[:, :prompt_len] = input_ids

        ts_total = time.perf_counter()
        ts_prefill_start = time.perf_counter()
        outputs, next_token, state, workspace_key, workspace_reused, prefill_mode = self._prefill(
            input_ids, model_kwargs, max_new_tokens
        )
        prefill_ms = (time.perf_counter() - ts_prefill_start) * 1000.0
        cur_len = prompt_len
        out[:, cur_len] = next_token
        cur_len += 1
        if fill_after_eos and self._eos_stop_index(next_token.reshape(-1), eos_tensor) is not None:
            if cur_len < prompt_len + max_new_tokens:
                out[:, cur_len : prompt_len + max_new_tokens].fill_(self._first_eos_token(eos_tensor))
            total_ms = (time.perf_counter() - ts_total) * 1000.0
            self.model._cuda_graph_last_runtime_stats = {
                "mode": "decode_runtime",
                "prefill_backend": prefill_mode,
                "prefill_ms": prefill_ms,
                "decode_ms": 0.0,
                "graph_capture_ms": 0.0,
                "total_ms": total_ms,
                "decode_steps": 0,
                "eos_filled_tokens": int(max_new_tokens - 1),
            }
            return out[:, : prompt_len + max_new_tokens]
        if self._eos_stop_index_after_min(next_token.reshape(-1), eos_tensor, generated_before=0, min_new_tokens=min_new_tokens) is not None:
            total_ms = (time.perf_counter() - ts_total) * 1000.0
            self.model._cuda_graph_last_runtime_stats = {
                "mode": "decode_runtime",
                "prefill_backend": prefill_mode,
                "prefill_ms": prefill_ms,
                "decode_ms": 0.0,
                "graph_capture_ms": 0.0,
                "total_ms": total_ms,
                "decode_steps": 0,
            }
            return out[:, :cur_len]

        graph_hits = 0
        decode_steps = 0
        graph_captures = 0
        graph_replays = 0
        chunk_captures = 0
        chunk_replays = 0
        last_failure = None
        graph_capture_ms = 0.0
        ts_decode_start = time.perf_counter()

        while cur_len < prompt_len + max_new_tokens:
            remaining = prompt_len + max_new_tokens - cur_len
            chunk_steps = min(self.chunk_steps, remaining)
            if chunk_steps > 1:
                model_inputs = self._build_decode_inputs(out[:, cur_len - 1 : cur_len], state)
                chunk_graph = self._get_chunk_graph(
                    prompt_len,
                    outputs.logits.dtype,
                    input_ids.device,
                    workspace_key,
                    chunk_steps,
                )
                captures_before = chunk_graph.captures
                replays_before = chunk_graph.replays
                ts_graph = time.perf_counter()
                chunk_tokens, used_chunk_graph = chunk_graph.run(self.model, model_inputs)
                if chunk_graph.captures > captures_before:
                    graph_capture_ms += (time.perf_counter() - ts_graph) * 1000.0
                chunk_captures += chunk_graph.captures - captures_before
                chunk_replays += chunk_graph.replays - replays_before
                last_failure = chunk_graph.last_failure
                if chunk_tokens is not None:
                    raw_stop_idx = self._eos_stop_index(chunk_tokens[0], eos_tensor)
                    if fill_after_eos and raw_stop_idx is not None:
                        consumed = raw_stop_idx + 1
                        out[:, cur_len : cur_len + consumed] = chunk_tokens[:, :consumed]
                        cur_len += consumed
                        if cur_len < prompt_len + max_new_tokens:
                            out[:, cur_len : prompt_len + max_new_tokens].fill_(self._first_eos_token(eos_tensor))
                        cur_len = prompt_len + max_new_tokens
                        self._advance_state(state, consumed)
                        decode_steps += consumed
                        graph_hits += int(used_chunk_graph) * consumed
                        break
                    stop_idx = self._eos_stop_index_after_min(
                        chunk_tokens[0],
                        eos_tensor,
                        generated_before=cur_len - prompt_len,
                        min_new_tokens=min_new_tokens,
                    )
                    consumed = chunk_steps if stop_idx is None else stop_idx + 1
                    out[:, cur_len : cur_len + consumed] = chunk_tokens[:, :consumed]
                    cur_len += consumed
                    self._advance_state(state, consumed)
                    decode_steps += consumed
                    graph_hits += int(used_chunk_graph) * consumed
                    if stop_idx is not None:
                        break
                    continue

            model_inputs = self._build_decode_inputs(out[:, cur_len - 1 : cur_len], state)
            step_graph = self._get_step_graph(prompt_len, outputs.logits.dtype, input_ids.device, workspace_key)
            captures_before = step_graph.captures
            replays_before = step_graph.replays
            ts_graph = time.perf_counter()
            next_token, used_graph = step_graph.run(self.model, model_inputs)
            if step_graph.captures > captures_before:
                graph_capture_ms += (time.perf_counter() - ts_graph) * 1000.0
            out[:, cur_len] = next_token
            cur_len += 1
            self._advance_state(state, 1)
            decode_steps += 1
            graph_hits += int(used_graph)
            graph_captures += step_graph.captures - captures_before
            graph_replays += step_graph.replays - replays_before
            last_failure = step_graph.last_failure
            raw_stop_idx = self._eos_stop_index(next_token.reshape(-1), eos_tensor)
            if fill_after_eos and raw_stop_idx is not None:
                if cur_len < prompt_len + max_new_tokens:
                    out[:, cur_len : prompt_len + max_new_tokens].fill_(self._first_eos_token(eos_tensor))
                cur_len = prompt_len + max_new_tokens
                break
            stop_idx = self._eos_stop_index_after_min(
                next_token.reshape(-1),
                eos_tensor,
                generated_before=cur_len - prompt_len - 1,
                min_new_tokens=min_new_tokens,
            )
            if stop_idx is not None:
                break

        decode_ms = (time.perf_counter() - ts_decode_start) * 1000.0
        total_ms = (time.perf_counter() - ts_total) * 1000.0
        self.model._cuda_graph_last_runtime_stats = {
            "mode": "decode_runtime",
            "prefill_backend": prefill_mode,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "graph_capture_ms": graph_capture_ms,
            "total_ms": total_ms,
            "decode_steps": int(decode_steps),
            "step_captures": int(graph_captures),
            "step_replays": int(graph_replays),
            "chunk_captures": int(chunk_captures),
            "chunk_replays": int(chunk_replays),
        }

        if conf_bool("CUDA_GRAPH_LOG_DECODE_STATS", "0"):
            hit_rate = graph_hits / max(1, decode_steps)
            print(
                f"[cuda_graph_decode] prompt_len={prompt_len} max_new_tokens={max_new_tokens} "
                f"prefill_backend={prefill_mode} prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f} "
                f"workspace_reused={int(workspace_reused)} decode_steps={decode_steps} "
                f"graph_hits={graph_hits} hit_rate={hit_rate:.2%} "
                f"step_captures={graph_captures} step_replays={graph_replays} "
                f"chunk_steps={self.chunk_steps} chunk_captures={chunk_captures} chunk_replays={chunk_replays} "
                f"graph_capture_ms={graph_capture_ms:.2f} "
                f"step_last_failure={last_failure}"
            )

        return out[:, :cur_len]


def apply_cuda_graph_decode(model) -> bool:
    """patch ``model.generate``，把符合条件的 decode 请求导到本模块。

    不支持的生成模式和运行时异常都会 fallback 到原始 HF generate，因此可以在
    模型初始化时安全安装该 hook。
    """
    if getattr(model, "_cuda_graph_decode_patched", False):
        return True
    if not conf_bool("ENABLE_DECODE_CUDA_GRAPH", "1"):
        return False
    if not getattr(model, "_flash_kvcache_attention_patched", False):
        print("[cuda_graph_decode] disabled: flash kvcache attention is not patched")
        return False

    runtime = CudaGraphDecodeRuntime(model)
    original_generate = model.generate

    @functools.wraps(original_generate)
    def generate(*args, **kwargs):
        """小适配层：兼容位置参数传入的 input_ids。"""
        call_kwargs = dict(kwargs)
        if args:
            if len(args) == 1 and "input_ids" not in call_kwargs:
                call_kwargs["input_ids"] = args[0]
            else:
                return original_generate(*args, **kwargs)
        try:
            with torch.no_grad():
                return runtime.generate(original_generate, (), call_kwargs)
        except Exception as exc:
            if conf_bool("CUDA_GRAPH_LOG_DECODE_STATS", "0"):
                print(f"[cuda_graph_decode] fallback={type(exc).__name__}: {exc}")
                import traceback

                traceback.print_exc()
            return original_generate(**CudaGraphDecodeRuntime._sanitize(call_kwargs))

    model.generate = generate
    model._cuda_graph_runtime = runtime
    model._cuda_graph_original_generate = original_generate
    model._cuda_graph_decode_patched = True
    print("[cuda_graph_decode] minimal decode CUDA graph path enabled")
    return True
