"""Compiled full-prefill builder.

`prefill_one_layer` runs one decoder layer (norm → fused QKV → FA2 with kvcache
→ fused MLP). `make_compiled_full_prefill` stitches NUM_LLM_LAYERS calls
together and wraps the whole thing in torch.compile.

B1 fix: use mode='default' instead of 'max-autotune-no-cudagraphs'. On a cold
container the autotune sweep dominates startup (5-30s per shape × 28 layers ×
many shapes); 'default' uses Inductor's canned templates which start nearly
immediately and still get FA2 (FA2 is invoked through a custom_op, not lowered
by the compiler).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .custom_ops import flash_attn_kvcache_op


def prefill_one_layer(hidden_states, cos_emb, sin_emb, fa_k, fa_v, fa_seqlens,
                      input_layernorm, post_attn_layernorm, mlp,
                      qkv_weight, qkv_bias, o_weight, q_norm, k_norm,
                      head_dim, scaling, q_dim, k_dim, v_dim):
    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, head_dim)

    qkv = F.linear(hidden_states, qkv_weight)
    if qkv_bias is not None:
        qkv = qkv + qkv_bias
    q, k, v = qkv.split([q_dim, k_dim, v_dim], dim=-1)

    query_states = q_norm(q.view(hidden_shape)).transpose(1, 2)
    key_states = k_norm(k.view(hidden_shape)).transpose(1, 2)
    value_states = v.view(hidden_shape).transpose(1, 2)

    cos_q = cos_emb.unsqueeze(1)
    sin_q = sin_emb.unsqueeze(1)

    q1, q2 = query_states.chunk(2, dim=-1)
    query_states = (query_states * cos_q) + (torch.cat((-q2, q1), dim=-1) * sin_q)
    k1, k2 = key_states.chunk(2, dim=-1)
    key_states = (key_states * cos_q) + (torch.cat((-k2, k1), dim=-1) * sin_q)

    query_states = query_states.to(hidden_states.dtype)
    key_states = key_states.to(hidden_states.dtype)

    q_fa = query_states.transpose(1, 2).contiguous()
    k_fa = key_states.transpose(1, 2).contiguous()
    v_fa = value_states.transpose(1, 2).contiguous()

    attn_output = flash_attn_kvcache_op(
        q_fa, fa_k, fa_v, k_fa, v_fa, fa_seqlens, True, scaling)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    hidden_states = F.linear(attn_output, o_weight)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = post_attn_layernorm(hidden_states)
    hidden_states = mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


def make_compiled_full_prefill(
    *, num_layers, layer_params_list,
    fa_k_cache, fa_v_cache,
    llm_norm, lm_head_apply,
):
    def _full_prefill(
        inputs_embeds, cos_emb, sin_emb,
        ds_scattered_0, ds_scattered_1, ds_scattered_2,
        fa_seqlens,
    ):
        hidden_states = inputs_embeds
        for layer_idx in range(num_layers):
            lp = layer_params_list[layer_idx]
            hidden_states = prefill_one_layer(
                hidden_states, cos_emb, sin_emb,
                fa_k_cache[layer_idx], fa_v_cache[layer_idx], fa_seqlens,
                lp['input_layernorm'], lp['post_attention_layernorm'], lp['mlp'],
                lp['qkv_weight'], lp['qkv_bias'], lp['o_weight'],
                lp['q_norm'], lp['k_norm'],
                lp['head_dim'], lp['scaling'],
                lp['q_dim'], lp['k_dim'], lp['v_dim'])

            if layer_idx == 0:
                hidden_states = hidden_states + ds_scattered_0
            elif layer_idx == 1:
                hidden_states = hidden_states + ds_scattered_1
            elif layer_idx == 2:
                hidden_states = hidden_states + ds_scattered_2

        hidden_states = llm_norm(hidden_states)
        last_hidden = hidden_states[:, -1, :]
        return lm_head_apply(last_hidden)

    return torch.compile(_full_prefill, mode='default', dynamic=True)


def make_eager_full_prefill_hidden(
    *, num_layers, layer_params_list,
    fa_k_cache, fa_v_cache,
    llm_norm,
):
    """vAxel31: eager (uncompiled) prefill returning full hidden states.

    NOT torch.compile-wrapped — captured directly by torch.cuda.CUDAGraph in
    vAxel31. dynamo's RNG-state guard at the graph entry is incompatible with
    cudaStreamCaptureStatusActive, so the bucketed path uses an eager Python
    function + CUDA Graph capture (which captures the underlying kernel
    sequence for replay anyway).
    """
    def _full_prefill_hidden(
        inputs_embeds, cos_emb, sin_emb,
        ds_scattered_0, ds_scattered_1, ds_scattered_2,
        fa_seqlens,
    ):
        hidden_states = inputs_embeds
        for layer_idx in range(num_layers):
            lp = layer_params_list[layer_idx]
            hidden_states = prefill_one_layer(
                hidden_states, cos_emb, sin_emb,
                fa_k_cache[layer_idx], fa_v_cache[layer_idx], fa_seqlens,
                lp['input_layernorm'], lp['post_attention_layernorm'], lp['mlp'],
                lp['qkv_weight'], lp['qkv_bias'], lp['o_weight'],
                lp['q_norm'], lp['k_norm'],
                lp['head_dim'], lp['scaling'],
                lp['q_dim'], lp['k_dim'], lp['v_dim'])

            if layer_idx == 0:
                hidden_states = hidden_states + ds_scattered_0
            elif layer_idx == 1:
                hidden_states = hidden_states + ds_scattered_1
            elif layer_idx == 2:
                hidden_states = hidden_states + ds_scattered_2

        hidden_states = llm_norm(hidden_states)
        return hidden_states

    return _full_prefill_hidden


def make_compiled_full_prefill_hidden(
    *, num_layers, layer_params_list,
    fa_k_cache, fa_v_cache,
    llm_norm,
):
    """Deprecated entry kept for back-compat — see make_eager_full_prefill_hidden."""
    return make_eager_full_prefill_hidden(
        num_layers=num_layers, layer_params_list=layer_params_list,
        fa_k_cache=fa_k_cache, fa_v_cache=fa_v_cache, llm_norm=llm_norm)
