"""LLM-side QKV / MLP fusion + FA2 decode shim.

`install_llm_fusions` patches every language-model layer to use:
  - one fused MLP gate+up GEMM,
  - one fused QKV GEMM,
and routes attention through flash_attn_with_kvcache when no past_key_values
object is supplied (decode/perf path).
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def _make_fused_mlp_forward(gate_up_weight, down_weight, act_fn):
    def _forward(self, x):
        gate_up = F.linear(x, gate_up_weight)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = act_fn(gate) * up
        return F.linear(hidden, down_weight)
    return _forward


def _make_fused_qkv_llm_attention_fa2(attn_module, fa_k_cache_layer,
                                      fa_v_cache_layer, fa_cache_seqlens):
    from flash_attn import flash_attn_with_kvcache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb, eager_attention_forward)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    q_w = attn_module.q_proj.weight
    k_w = attn_module.k_proj.weight
    v_w = attn_module.v_proj.weight
    qkv_weight = torch.cat([q_w, k_w, v_w], dim=0)
    q_dim = q_w.shape[0]
    k_dim = k_w.shape[0]
    v_dim = v_w.shape[0]

    if attn_module.q_proj.bias is not None:
        qkv_bias = torch.cat([
            attn_module.q_proj.bias,
            attn_module.k_proj.bias,
            attn_module.v_proj.bias], dim=0)
    else:
        qkv_bias = None

    o_weight = attn_module.o_proj.weight
    q_norm = attn_module.q_norm
    k_norm = attn_module.k_norm
    head_dim = attn_module.head_dim
    scaling = attn_module.scaling
    layer_idx = attn_module.layer_idx
    attn_impl_key = attn_module.config._attn_implementation
    attention_interface_fn = ALL_ATTENTION_FUNCTIONS.get(
        attn_impl_key, eager_attention_forward)

    def _forward(self, hidden_states, position_embeddings, attention_mask,
                 past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        qkv = F.linear(hidden_states, qkv_weight)
        if qkv_bias is not None:
            qkv = qkv + qkv_bias
        q, k, v = qkv.split([q_dim, k_dim, v_dim], dim=-1)

        query_states = q_norm(q.view(hidden_shape)).transpose(1, 2)
        key_states = k_norm(k.view(hidden_shape)).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        hidden_dtype = hidden_states.dtype
        query_states = query_states.to(hidden_dtype)
        key_states = key_states.to(hidden_dtype)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, layer_idx, cache_kwargs)
            attn_output, attn_weights = attention_interface_fn(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0, scaling=scaling, **kwargs)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return F.linear(attn_output, o_weight), attn_weights
        else:
            q_fa = query_states.transpose(1, 2).contiguous()
            k_fa = key_states.transpose(1, 2).contiguous()
            v_fa = value_states.transpose(1, 2).contiguous()

            attn_output = flash_attn_with_kvcache(
                q_fa, fa_k_cache_layer, fa_v_cache_layer, k_fa, v_fa,
                cache_seqlens=fa_cache_seqlens, causal=True, softmax_scale=scaling)

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return F.linear(attn_output, o_weight), None

    return _forward


def install_llm_fusions(model, fa_k_cache, fa_v_cache, fa_cache_seqlens):
    n_mlp = n_qkv = 0
    for name, module in model.named_modules():
        if 'language_model.layers.' in name and name.endswith('.mlp'):
            if hasattr(module, 'gate_proj') and module.gate_proj.bias is None:
                gate_up_weight = torch.cat(
                    [module.gate_proj.weight, module.up_proj.weight], dim=0)
                down_weight = module.down_proj.weight
                act_fn = module.act_fn
                module.forward = types.MethodType(
                    _make_fused_mlp_forward(gate_up_weight, down_weight, act_fn),
                    module)
                n_mlp += 1
        if 'language_model.layers.' in name and name.endswith('.self_attn'):
            if hasattr(module, 'q_proj') and hasattr(module, 'layer_idx'):
                li = module.layer_idx
                module.forward = types.MethodType(
                    _make_fused_qkv_llm_attention_fa2(
                        module, fa_k_cache[li], fa_v_cache[li], fa_cache_seqlens),
                    module)
                n_qkv += 1
    return n_mlp, n_qkv
