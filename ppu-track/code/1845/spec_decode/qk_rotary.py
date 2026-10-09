from __future__ import annotations

import os
from contextlib import contextmanager
from types import MethodType

import torch

try:
    from my_kernel.decode_qk_rotary.runtime import _try_static_cache_triton_update
except Exception:
    _try_static_cache_triton_update = None


_SPEC_QK_ROTARY_ENABLED = False
_TRACE_ONCE: set[str] = set()


def _trace_once(key: str, message: str) -> None:
    if os.getenv("AICAS_SPEC_KERNEL_TRACE", "0") != "1" or key in _TRACE_ONCE:
        return
    _TRACE_ONCE.add(key)
    print(message)


def _debug_capture_l0(module, stage: str, **tensors) -> None:
    if os.getenv("AICAS_EAGLE3_TREE_CORE_DEBUG_L0", "0") != "1":
        return
    if int(getattr(module, "layer_idx", -1)) != 0:
        return
    token_state = getattr(module, "_aicas_tree_debug_state", None)
    if not isinstance(token_state, dict):
        return
    store = token_state.setdefault("_tree_core_debug_l0", {})
    stage_store = store.setdefault(stage, {})
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        out = stage_store.get(name)
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != tensor.shape
            or out.device != tensor.device
            or out.dtype != tensor.dtype
        ):
            out = torch.empty_like(tensor)
            stage_store[name] = out
        out.copy_(tensor)


def _cos_sin_2d(t: torch.Tensor, q_len: int) -> torch.Tensor | None:
    if not isinstance(t, torch.Tensor):
        return None
    if t.ndim == 3 and t.shape[0] == 1 and t.shape[1] == q_len and t.shape[2] == 128:
        return t[0]
    if t.ndim == 2 and t.shape[0] == q_len and t.shape[1] == 128:
        return t
    if t.numel() == q_len * 128:
        return t.reshape(q_len, 128)
    return None


def _can_use_spec_qk(hidden_states: torch.Tensor, position_embeddings) -> bool:
    if not _SPEC_QK_ROTARY_ENABLED:
        return False
    if os.getenv("AICAS_SPEC_QK_ROTARY", "0") != "1":
        return False
    if not isinstance(hidden_states, torch.Tensor):
        return False
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        return False
    q_len = int(hidden_states.shape[1])
    max_q = int(os.getenv("AICAS_SPEC_KERNEL_MAX_Q", "16"))
    if q_len < 1 or q_len > max_q:
        return False
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        return False
    if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
        return False
    cos = _cos_sin_2d(position_embeddings[0], q_len)
    sin = _cos_sin_2d(position_embeddings[1], q_len)
    return cos is not None and sin is not None


def _layer_stable_cache_sources(module, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key_buf = getattr(module, "_aicas_spec_qk_key_update_src", None)
    if (
        not isinstance(key_buf, torch.Tensor)
        or key_buf.shape != key_states.shape
        or key_buf.device != key_states.device
        or key_buf.dtype != key_states.dtype
    ):
        key_buf = torch.empty_like(key_states)
        module._aicas_spec_qk_key_update_src = key_buf
    value_buf = getattr(module, "_aicas_spec_qk_value_update_src", None)
    if (
        not isinstance(value_buf, torch.Tensor)
        or value_buf.shape != value_states.shape
        or value_buf.device != value_states.device
        or value_buf.dtype != value_states.dtype
    ):
        value_buf = torch.empty_like(value_states)
        module._aicas_spec_qk_value_update_src = value_buf
    key_buf.copy_(key_states)
    value_buf.copy_(value_states)
    return key_buf, value_buf


def apply_spec_qk_rotary(model) -> int:
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    except Exception:
        return 0

    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is None or not hasattr(lm, "layers"):
        return 0

    patched = 0
    for layer in lm.layers:
        inner_layer = getattr(layer, "original_module", layer)
        attn = getattr(inner_layer, "self_attn", None)
        if attn is None or getattr(attn, "_aicas_spec_qk_rotary_patched", False):
            continue
        original_forward = attn.forward

        def _make_forward(orig):
            def _forward(
                self,
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                **kwargs,
            ):
                if not _can_use_spec_qk(hidden_states, position_embeddings):
                    _debug_capture_l0(self, "path_unsupported", hidden=hidden_states[:, :1, :1])
                    return orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        **kwargs,
                    )

                try:
                    from spec_decode.kernels.bf16_qk_rotary.runtime import (
                        run_bf16_qk_rmsnorm_rotary,
                    )

                    batch, q_len, _ = hidden_states.shape
                    if batch != 1:
                        raise RuntimeError("spec qk rotary supports batch=1 only")

                    input_shape = hidden_states.shape[:-1]
                    q_shape = (1, q_len, self.config.num_attention_heads, self.head_dim)
                    kv_shape = (1, q_len, self.config.num_key_value_heads, self.head_dim)

                    q_raw = self.q_proj(hidden_states).view(q_shape).contiguous()
                    k_raw = self.k_proj(hidden_states).view(kv_shape).contiguous()
                    value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2).contiguous()
                    _debug_capture_l0(
                        self,
                        "qkv",
                        q_raw=q_raw,
                        k_raw=k_raw,
                        value=value_states,
                    )

                    cos = _cos_sin_2d(position_embeddings[0], q_len)
                    sin = _cos_sin_2d(position_embeddings[1], q_len)
                    if cos is None or sin is None:
                        raise RuntimeError("unsupported cos/sin shape")

                    qk = run_bf16_qk_rmsnorm_rotary(
                        q_raw.reshape(q_len * self.config.num_attention_heads, self.head_dim),
                        k_raw.reshape(q_len * self.config.num_key_value_heads, self.head_dim),
                        cos,
                        sin,
                        self.q_norm.weight,
                        self.k_norm.weight,
                        float(self.q_norm.variance_epsilon),
                        cache_tag=int(self.layer_idx),
                    )
                    if qk is None:
                        raise RuntimeError("bf16 qk rotary backend rejected inputs")
                    query_states, key_states = qk
                    _debug_capture_l0(
                        self,
                        "rotary",
                        query=query_states,
                        key=key_states,
                        value=value_states,
                    )

                    if past_key_values is not None:
                        key_states, value_states = _layer_stable_cache_sources(self, key_states, value_states)
                        _debug_capture_l0(
                            self,
                            "cache_src",
                            key=key_states,
                            value=value_states,
                        )
                        cache_kwargs = {
                            "sin": position_embeddings[1],
                            "cos": position_embeddings[0],
                            "cache_position": cache_position,
                        }
                        updated = None
                        if _try_static_cache_triton_update is not None:
                            updated = _try_static_cache_triton_update(
                                past_key_values,
                                key_states,
                                value_states,
                                int(self.layer_idx),
                                cache_position,
                            )
                        if updated is None:
                            self._aicas_cache_update_path = "hf"
                            updated = past_key_values.update(
                                key_states,
                                value_states,
                                self.layer_idx,
                                cache_kwargs,
                            )
                        else:
                            self._aicas_cache_update_path = "triton_fixed"
                        key_states, value_states = updated
                        _debug_capture_l0(
                            self,
                            "cache_after",
                            key=key_states,
                            value=value_states,
                        )

                    attention_interface = qwen.ALL_ATTENTION_FUNCTIONS.get_interface(
                        self.config._attn_implementation,
                        qwen.eager_attention_forward,
                    )
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
                    _debug_capture_l0(self, "attn_out", out=attn_output)
                    attn_output = self.o_proj(attn_output)
                    _debug_capture_l0(self, "oproj", out=attn_output)
                    _trace_once(
                        "spec_qk_rotary",
                        f"[spec_kernel] bf16_qk_rotary hit q={q_len}",
                    )
                    return attn_output, attn_weights
                except Exception as exc:
                    if os.getenv("AICAS_SPEC_QK_ROTARY_REQUIRE", "0") == "1":
                        raise
                    _debug_capture_l0(self, "path_fallback", hidden=hidden_states[:, :1, :1])
                    _trace_once(
                        "spec_qk_rotary_fallback",
                        f"[spec_kernel] bf16_qk_rotary fallback: {type(exc).__name__}: {exc}",
                    )
                    return orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        **kwargs,
                    )

            return _forward

        attn.forward = MethodType(_make_forward(original_forward), attn)
        attn._aicas_spec_qk_rotary_patched = True
        patched += 1

    return patched


@contextmanager
def spec_qk_rotary(model):
    global _SPEC_QK_ROTARY_ENABLED
    if os.getenv("AICAS_SPEC_QK_ROTARY", "0") != "1":
        yield
        return
    apply_spec_qk_rotary(model)
    old = _SPEC_QK_ROTARY_ENABLED
    _SPEC_QK_ROTARY_ENABLED = True
    try:
        yield
    finally:
        _SPEC_QK_ROTARY_ENABLED = old
