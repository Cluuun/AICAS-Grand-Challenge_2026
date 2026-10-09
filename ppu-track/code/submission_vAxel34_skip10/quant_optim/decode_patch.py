"""Patch vAxel26-style fused QKV/MLP modules with W8A16 weight on M=1 decode path.

Strategy:
  * `_make_fused_qkv_llm_attention_fa2`: the FA2 (else) branch is M=1 decode -> use W8A16.
    The SDPA (if past_key_values is not None) branch is the eager-fallback acc
    path -> keep FP16.
  * `_make_fused_mlp_forward`: dispatch on input M. M=1 -> W8A16; otherwise FP16.
    This makes the patched MLP safe inside the compiled-prefill path (M~hundreds
    keeps FP16) and the compiled-decode path (M=1 takes W8A16).

The module exposes a single function `install_w8a16_decode_patches(...)` that
takes the model, the W8A16 bundle, and the static FA2 cache references; it
mutates each `language_model.layers[i].self_attn.forward` and `.mlp.forward`.
"""
from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn.functional as F

from .w8a16 import W8A16Weight, w8a16_linear
from .llm_decode_quant import W8A16LLMBundle


def _make_w8a16_fused_qkv_attention(
    attn_module,
    qkv_w8: W8A16Weight,
    o_w8: W8A16Weight,
    qkv_bias: Optional[torch.Tensor],
    qkv_fp16: torch.Tensor,
    o_fp16: torch.Tensor,
    q_dim: int,
    k_dim: int,
    v_dim: int,
    fa_k_cache_layer: torch.Tensor,
    fa_v_cache_layer: torch.Tensor,
    fa_cache_seqlens: torch.Tensor,
):
    """Build the patched self_attn.forward replicating vAxel26's FA2 branch
    but using W8A16 on M=1 (FA2 path). FP16 fallback retained for SDPA path."""
    from flash_attn import flash_attn_with_kvcache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb, eager_attention_forward)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

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

        if past_key_values is not None:
            # SDPA acc-fallback path: keep FP16
            qkv = F.linear(hidden_states, qkv_fp16)
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

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, layer_idx, cache_kwargs)
            attn_output, attn_weights = attention_interface_fn(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0, scaling=scaling, **kwargs)
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            return F.linear(attn_output, o_fp16), attn_weights

        # FA2 decode path (M=1): W8A16
        qkv = w8a16_linear(hidden_states, qkv_w8)
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

        q_fa = query_states.transpose(1, 2).contiguous()
        k_fa = key_states.transpose(1, 2).contiguous()
        v_fa = value_states.transpose(1, 2).contiguous()

        attn_output = flash_attn_with_kvcache(
            q_fa, fa_k_cache_layer, fa_v_cache_layer, k_fa, v_fa,
            cache_seqlens=fa_cache_seqlens, causal=True, softmax_scale=scaling)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return w8a16_linear(attn_output, o_w8), None

    return _forward


def _make_w8a16_fused_mlp(
    gate_up_w8: W8A16Weight,
    down_w8: W8A16Weight,
    gate_up_fp16: torch.Tensor,
    down_fp16: torch.Tensor,
    act_fn,
):
    """Build the patched mlp.forward that dispatches on input M:
       M==1 -> W8A16; M>1 -> FP16. The compiled-prefill path (M~hundreds) stays FP16,
       the compiled-decode path (M=1) takes W8A16."""
    def _forward(self, x):
        # x.shape: [batch, seq, hidden] (3D in this model)
        # M is product of leading dims; with batch=1 it's just seq_len.
        if x.shape[-2] == 1:
            gate_up = w8a16_linear(x, gate_up_w8)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = act_fn(gate) * up
            return w8a16_linear(hidden, down_w8)
        else:
            gate_up = F.linear(x, gate_up_fp16)
            gate, up = gate_up.chunk(2, dim=-1)
            hidden = act_fn(gate) * up
            return F.linear(hidden, down_fp16)

    return _forward


def install_w8a16_decode_patches(
    model,
    bundle: W8A16LLMBundle,
    fa_k_cache: torch.Tensor,
    fa_v_cache: torch.Tensor,
    fa_cache_seqlens: torch.Tensor,
) -> tuple[int, int]:
    """Patch each LLM decoder layer's self_attn.forward and mlp.forward
    with the W8A16-on-decode dispatch.

    Returns (n_attn_patched, n_mlp_patched).
    """
    n_attn = n_mlp = 0
    lm = model.model.language_model
    for li, layer in enumerate(lm.layers):
        attn_b = bundle.attns[li]
        mlp_b = bundle.mlps[li]
        if attn_b is None or mlp_b is None:
            continue

        attn = layer.self_attn
        # Reconstruct FP16 fused weights for the SDPA-fallback path.
        qkv_fp16 = torch.cat(
            [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
        )
        o_fp16 = attn.o_proj.weight
        attn.forward = types.MethodType(
            _make_w8a16_fused_qkv_attention(
                attn, attn_b.qkv, attn_b.o, attn_b.qkv_bias,
                qkv_fp16, o_fp16,
                attn_b.q_dim, attn_b.k_dim, attn_b.v_dim,
                fa_k_cache[li], fa_v_cache[li], fa_cache_seqlens,
            ),
            attn,
        )
        n_attn += 1

        mlp = layer.mlp
        if hasattr(mlp, 'gate_proj') and mlp.gate_proj.bias is None:
            gate_up_fp16 = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
            down_fp16 = mlp.down_proj.weight
            mlp.forward = types.MethodType(
                _make_w8a16_fused_mlp(
                    mlp_b.gate_up, mlp_b.down,
                    gate_up_fp16, down_fp16, mlp.act_fn,
                ),
                mlp,
            )
            n_mlp += 1

    return n_attn, n_mlp
