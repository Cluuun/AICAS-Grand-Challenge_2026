"""INT4 batch verify: dequantize + torch.mm for multi-token speculative decode.

For each layer, dequantize INT4 weights to FP16 buffer, then use torch.mm
for batch matmul with all verify tokens. Weight reads are shared across
tokens (only dequantize once per layer).
"""
import os
import torch
import torch.nn.functional as F


def _rms_norm(hidden_states, weight, eps):
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    return (hidden_states.float() * torch.rsqrt(variance + eps)).to(hidden_states.dtype) * weight


def _dequantize_weight(module):
    """Dequantize INT4 packed weight to FP16 [out_dim, in_dim]."""
    from custom_kernels.quant.quantize import dequantize_int4_packed

    packed = module._int4_packed
    scale = module._int4_scale
    gs = module._int4_group_size
    fp32 = dequantize_int4_packed(packed, scale, gs)
    return fp32.to(torch.float16)


def _has_int4_weights(module):
    return hasattr(module, '_int4_packed')


def _apply_linear_batch(module, x_batch):
    """Apply linear projection to batch of tokens [B, T, D]."""
    if _has_int4_weights(module):
        W = _dequantize_weight(module)
        # Cache the dequantized weight for this verify cycle
        if not hasattr(module, '_fp16_batch_weight') or module._fp16_batch_weight is None:
            module._fp16_batch_weight = W
        else:
            W = module._fp16_batch_weight
        # x_batch: [T, D], W: [out, D]
        return F.linear(x_batch, W)
    else:
        return module(x_batch)


def int4_batch_verify_forward(language_model, lm_head, cache, verify_ids,
                              rope_cos_table, rope_sin_table, rope_delta,
                              head_dim, num_q_heads, num_kv_heads,
                              hidden_size, q_dim, kv_dim, eps=1e-6):
    """Batch verify with INT4 dequantize + FP16 matmul.

    Processes gamma+1 tokens through all layers in a single pass,
    sharing weight dequantization across all tokens.
    """
    num_tokens = verify_ids.shape[1]
    device = verify_ids.device

    # Pre-dequantize all weights once
    for layer in language_model.layers:
        for mod in [layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                    layer.mlp.gate_up_proj, layer.mlp.down_proj]:
            if _has_int4_weights(mod):
                mod._fp16_batch_weight = _dequantize_weight(mod)
    if _has_int4_weights(lm_head):
        lm_head._fp16_batch_weight = _dequantize_weight(lm_head)

    # Embed tokens
    hidden = language_model.embed_tokens(verify_ids)  # [1, T, hidden]

    # Position IDs
    cache_pos = cache.layers[0]._pos_val
    positions = torch.arange(cache_pos + rope_delta,
                              cache_pos + rope_delta + num_tokens,
                              device=device)

    # RoPE
    if rope_cos_table is not None and all(p < rope_cos_table.shape[0] for p in positions.tolist()):
        cos_batch = rope_cos_table[positions]
        sin_batch = rope_sin_table[positions]
    else:
        theta = 5000000.0
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = positions.float()
        freqs = torch.outer(t, freqs)
        cos_batch = freqs.cos().to(torch.float16)
        sin_batch = freqs.sin().to(torch.float16)

    # Process through layers
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        residual = hidden

        # Input layernorm
        hidden = _rms_norm(hidden, decoder_layer.input_layernorm.weight,
                           decoder_layer.input_layernorm.variance_epsilon)

        # QKV projection (dequantize + batch matmul)
        hidden_2d = hidden.view(num_tokens, hidden_size)
        qkv = _apply_linear_batch(decoder_layer.self_attn.qkv_proj, hidden_2d)
        qkv = qkv.view(1, num_tokens, q_dim + 2 * kv_dim)
        q_raw = qkv[:, :, :q_dim]
        k_raw = qkv[:, :, q_dim:q_dim + kv_dim]
        v_raw = qkv[:, :, q_dim + kv_dim:]

        query = q_raw.view(1, num_tokens, num_q_heads, head_dim)
        key = k_raw.view(1, num_tokens, num_kv_heads, head_dim)
        value = v_raw.view(1, num_tokens, num_kv_heads, head_dim)

        # Q/K norm
        query = _rms_norm(query, decoder_layer.self_attn.q_norm.weight,
                          decoder_layer.self_attn.q_norm.variance_epsilon)
        key = _rms_norm(key, decoder_layer.self_attn.k_norm.weight,
                        decoder_layer.self_attn.k_norm.variance_epsilon)

        # RoPE (rotate_half)
        cos = cos_batch.unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim]
        sin = sin_batch.unsqueeze(0).unsqueeze(2)

        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin

        # Attention with flash_attn_with_kvcache
        has_flash_cache = hasattr(cache, 'flash_k_caches')
        if has_flash_cache:
            from flash_attn import flash_attn_with_kvcache
            attn_output = flash_attn_with_kvcache(
                q=query, k_cache=cache.flash_k_caches[layer_idx],
                v_cache=cache.flash_v_caches[layer_idx],
                k=key, v=value, cache_seqlens=cache.cache_seqlens,
            )
        else:
            from custom_kernels.patch_model import _cache_update
            key_t = key.transpose(1, 2).contiguous()
            value_t = value.transpose(1, 2).contiguous()
            key_cache, value_cache = _cache_update(cache, key_t, value_t, layer_idx)
            key_all = key_cache.transpose(1, 2)
            value_all = value_cache.transpose(1, 2)
            n_rep = num_q_heads // num_kv_heads
            if n_rep > 1:
                key_all = key_all.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(
                    1, key_all.shape[1], num_q_heads, head_dim)
                value_all = value_all.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(
                    1, value_all.shape[1], num_q_heads, head_dim)
            from flash_attn import flash_attn_func
            attn_output = flash_attn_func(
                query, key_all, value_all, causal=True,
                softmax_scale=head_dim ** -0.5,
            )

        attn_output = attn_output.reshape(1, num_tokens, -1).contiguous()
        # O projection
        attn_2d = attn_output.view(num_tokens, hidden_size)
        attn_output = _apply_linear_batch(decoder_layer.self_attn.o_proj, attn_2d)
        attn_output = attn_output.view(1, num_tokens, hidden_size)

        # Residual + post-attention norm
        hidden = residual + attn_output
        residual = hidden
        hidden = _rms_norm(hidden, decoder_layer.post_attention_layernorm.weight,
                           decoder_layer.post_attention_layernorm.variance_epsilon)

        # MLP: gate_up + silu_mul + down
        hidden_2d = hidden.view(num_tokens, hidden_size)
        gate_up = _apply_linear_batch(decoder_layer.mlp.gate_up_proj, hidden_2d)
        gate_up = gate_up.view(1, num_tokens, -1)
        d = gate_up.shape[-1] // 2
        intermediate = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        intermediate_2d = intermediate.view(num_tokens, -1)
        down_out = _apply_linear_batch(decoder_layer.mlp.down_proj, intermediate_2d)
        hidden = residual + down_out.view(1, num_tokens, hidden_size)

    # Final norm
    hidden = _rms_norm(hidden, language_model.norm.weight, eps)

    # lm_head
    hidden_2d = hidden.view(num_tokens, hidden_size)
    logits = _apply_linear_batch(lm_head, hidden_2d.half())
    logits = logits.view(1, num_tokens, -1)

    # Clean up cached weights
    for layer in language_model.layers:
        for mod in [layer.self_attn.qkv_proj, layer.self_attn.o_proj,
                    layer.mlp.gate_up_proj, layer.mlp.down_proj]:
            if hasattr(mod, '_fp16_batch_weight'):
                mod._fp16_batch_weight = None
    if hasattr(lm_head, '_fp16_batch_weight'):
        lm_head._fp16_batch_weight = None

    return logits


def int4_batch_verify_decode(language_model, lm_head, cache, verify_tokens,
                             rope_delta=0, gamma=4):
    """Run INT4 batch verify: dequantize weights + FP16 matmul.

    Args:
        language_model: the language model (patched with INT4)
        lm_head: lm_head layer
        cache: GraphStaticKVCache
        verify_tokens: list of int, length gamma+1
        rope_delta: int
        gamma: int

    Returns:
        pred_tokens: list of int, predictions for each position
    """
    device = next(language_model.parameters()).device
    num_tokens = len(verify_tokens)

    text_config = getattr(language_model.config, 'text_config', language_model.config)
    hidden_size = text_config.hidden_size
    num_q_heads = text_config.num_attention_heads
    num_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, 'head_dim', hidden_size // num_q_heads)
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    rope_cos = getattr(language_model, '_rope_cos_table', None)
    rope_sin = getattr(language_model, '_rope_sin_table', None)

    verify_ids = torch.tensor([verify_tokens], device=device, dtype=torch.long)

    with torch.inference_mode():
        logits = int4_batch_verify_forward(
            language_model, lm_head, cache, verify_ids,
            rope_cos, rope_sin, rope_delta,
            head_dim, num_q_heads, num_kv_heads,
            hidden_size, q_dim, kv_dim,
        )

    pred_tokens = logits[0].argmax(dim=-1).cpu().tolist()
    return pred_tokens
