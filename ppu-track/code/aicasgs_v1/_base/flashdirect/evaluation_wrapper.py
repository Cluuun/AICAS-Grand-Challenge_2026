from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


_BASE_WRAPPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "qkrotary"
    / "evaluation_wrapper.py"
)
_BASE_MODULE_NAME = "aicas_prefill_gpuvisionnorm_prefillqkrotary_flashdirect_20260417"
_BASE_SPEC = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_WRAPPER_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"无法加载基础 wrapper: {_BASE_WRAPPER_PATH}")
_BASE_MODULE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE_MODULE)
_FLASH_UTILS = importlib.import_module("transformers.modeling_flash_attention_utils")


class VLMModel(_BASE_MODULE.VLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "prefill_flash_direct" not in self._optimizations_applied:
            self._optimizations_applied.append("prefill_flash_direct")

    def _run_prefill_attention_fast(
        self,
        mod,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        attn_impl = str(getattr(getattr(mod, "config", None), "_attn_implementation", "") or "")
        if "flash_attention" not in attn_impl:
            return super()._run_prefill_attention_fast(
                mod,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        try:
            input_shape = hidden_states.shape[:-1]
            num_heads = int(getattr(mod, "num_heads", getattr(mod.config, "num_attention_heads", 0)))
            kv_heads = int(
                getattr(
                    mod,
                    "num_key_value_heads",
                    getattr(mod, "num_kv_heads", getattr(mod.config, "num_key_value_heads", num_heads)),
                )
            )
            if num_heads <= 0 or kv_heads <= 0:
                raise RuntimeError("无法读取 prefill attention 头数")

            q_shape = (*input_shape, num_heads, mod.head_dim)
            kv_shape = (*input_shape, kv_heads, mod.head_dim)

            if hasattr(mod, "qkv_proj"):
                qkv_out = mod.qkv_proj(hidden_states)
                split_sizes = getattr(mod, "_packed_qkv_split_sizes", None)
                if split_sizes is None:
                    q_dim = num_heads * mod.head_dim
                    kv_dim = kv_heads * mod.head_dim
                    split_sizes = (q_dim, kv_dim, kv_dim)
                q_out, k_out, v_out = torch.split(qkv_out, split_sizes, dim=-1)
            elif hasattr(mod, "_packed_qkv_weight"):
                qkv_out = F.linear(hidden_states, mod._packed_qkv_weight, mod._packed_qkv_bias)
                q_out, k_out, v_out = torch.split(qkv_out, mod._packed_qkv_split_sizes, dim=-1)
            else:
                q_out = mod.q_proj(hidden_states)
                k_out = mod.k_proj(hidden_states)
                v_out = mod.v_proj(hidden_states)

            q_states = q_out.view(q_shape)
            k_states = k_out.view(kv_shape)
            v_states = v_out.view(kv_shape)

            cos, sin = position_embeddings
            q_states, k_states = self._run_prefill_qk_norm_rotary_pair(
                q_states,
                k_states,
                mod.q_norm.weight,
                mod.k_norm.weight,
                float(mod.q_norm.variance_epsilon),
                float(mod.k_norm.variance_epsilon),
                cos,
                sin,
            )

            flash_k = k_states
            flash_v = v_states
            if past_key_values is not None:
                key_states = k_states.transpose(1, 2)
                value_states = v_states.transpose(1, 2)
                key_states, value_states = past_key_values.update(key_states, value_states, mod.layer_idx)
                flash_k = key_states.transpose(1, 2)
                flash_v = value_states.transpose(1, 2)

            attn_output = _FLASH_UTILS._flash_attention_forward(
                q_states,
                flash_k,
                flash_v,
                attention_mask,
                query_length=int(q_states.shape[1]),
                is_causal=bool(getattr(mod, "is_causal", True)),
                dropout=0.0 if not mod.training else mod.attention_dropout,
                softmax_scale=mod.scaling,
                attn_implementation=attn_impl,
                **kwargs,
            )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
            attn_output = attn_output.reshape(*input_shape, -1)
            attn_output = mod.o_proj(attn_output)
            return attn_output, None
        except Exception as exc:
            if hasattr(self, "_record_fastpath_fallback"):
                self._record_fastpath_fallback("prefill_attention_flash_direct", exc)
            return super()._run_prefill_attention_fast(
                mod,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
