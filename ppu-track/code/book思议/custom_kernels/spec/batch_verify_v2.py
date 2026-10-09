"""Batch verify for speculative decoding using INT4 batch GEMV.

Processes gamma+1 tokens through all layers with shared weight reads.
Memory traffic ≈ same as single-token decode (weights read once).
Uses flash_attn_with_kvcache for correct causal attention over cache + new tokens.
"""
import os
import torch
import torch.nn.functional as F


def _rms_norm(hidden, weight, eps):
    """RMS norm over last dim. Works for any shape [..., D]."""
    variance = hidden.float().pow(2).mean(-1, keepdim=True)
    return (hidden.float() * torch.rsqrt(variance + eps)).to(hidden.dtype) * weight


def _get_model_config(language_model):
    config = getattr(language_model.config, 'text_config', language_model.config)
    hs = config.hidden_size
    nq = config.num_attention_heads
    nkv = config.num_key_value_heads
    hd = getattr(config, 'head_dim', hs // nq)
    return hs, nq, nkv, hd


def _get_num_groups(module):
    """Get num_groups from INT4 scale shape."""
    if hasattr(module, 'weight_scale_int4'):
        return module.weight_scale_int4.shape[1]
    return module.weight.shape[1] // 128  # fallback


def _compute_rope(positions, head_dim, device):
    """Compute RoPE cos/sin tables for given positions. [T, head_dim] each."""
    theta = 5000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = positions.float()
    freqs = torch.outer(t, freqs)  # [T, head_dim//2]
    cos = freqs.cos().to(torch.float16)  # [T, head_dim//2]
    sin = freqs.sin().to(torch.float16)
    return cos, sin


def _apply_rope(x, cos, sin):
    """Apply RoPE to x [1, T, H, D]. cos/sin [T, D]."""
    cos_b = cos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos_b + rotated * sin_b


def batch_verify_forward(language_model, lm_head, cache, verify_ids,
                         rope_delta=0, gamma=4, gemv_ops=None):
    """Batch verify: INT4 batch GEMV for all layers, shared weight reads.

    Args:
        language_model: INT4-quantized language model (with patched layers)
        lm_head: INT4-quantized lm_head
        cache: GraphStaticKVCache (must have flash_k_caches, flash_v_caches, cache_seqlens)
        verify_ids: list of int, length gamma+1
        rope_delta: position offset for RoPE
        gamma: speculative decoding window size
        gemv_ops: loaded C++ extension module (loaded lazily if None)

    Returns:
        pred_tokens: list of int, model predictions for each position
    """
    if gemv_ops is None:
        from custom_kernels.cuda.gemv_loader import get_gemv_ops
        gemv_ops = get_gemv_ops()

    device = next(language_model.parameters()).device
    T = len(verify_ids)  # gamma + 1

    hs, nq, nkv, hd = _get_model_config(language_model)
    q_dim = nq * hd
    kv_dim = nkv * hd
    eps = 1e-6

    # Get cache position from flash cache
    cache_pos = cache.cache_seqlens[0].item()

    # Embed tokens
    verify_tensor = torch.tensor([verify_ids], device=device, dtype=torch.long)
    hidden = language_model.embed_tokens(verify_tensor)  # [1, T, hs]

    # RoPE: position IDs for each token
    positions = torch.arange(cache_pos + rope_delta,
                             cache_pos + rope_delta + T,
                             device=device)

    # Check if model has precomputed RoPE tables
    rope_cos = getattr(language_model, '_rope_cos_table', None)
    rope_sin = getattr(language_model, '_rope_sin_table', None)

    if rope_cos is not None and all(p < rope_cos.shape[0] for p in positions.tolist()):
        cos_batch = rope_cos[positions]  # [T, hd]
        sin_batch = rope_sin[positions]
    else:
        cos_batch, sin_batch = _compute_rope(positions, hd, device)

    # Process through layers
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        residual = hidden

        # Input RMS norm
        hidden = _rms_norm(hidden, decoder_layer.input_layernorm.weight,
                          decoder_layer.input_layernorm.variance_epsilon)

        # QKV batch INT4 GEMV
        hidden_2d = hidden.view(T, hs)
        qkv_proj = decoder_layer.self_attn.qkv_proj
        if hasattr(qkv_proj, 'weight_int4'):
            ng_qkv = _get_num_groups(qkv_proj)
            qkv_flat = gemv_ops.gemv_int4_batch(
                qkv_proj.weight_int4, hidden_2d,
                qkv_proj.weight_scale_int4, ng_qkv, T)
        else:
            qkv_flat = F.linear(hidden_2d, qkv_proj.weight)
        qkv = qkv_flat.view(1, T, q_dim + 2 * kv_dim)

        # Split QKV
        q_raw = qkv[:, :, :q_dim].view(1, T, nq, hd)
        k_raw = qkv[:, :, q_dim:q_dim + kv_dim].view(1, T, nkv, hd)
        v_raw = qkv[:, :, q_dim + kv_dim:].view(1, T, nkv, hd)

        # Q/K RMS norm
        q = _rms_norm(q_raw, decoder_layer.self_attn.q_norm.weight,
                      decoder_layer.self_attn.q_norm.variance_epsilon)
        k = _rms_norm(k_raw, decoder_layer.self_attn.k_norm.weight,
                      decoder_layer.self_attn.k_norm.variance_epsilon)

        # RoPE
        q = _apply_rope(q, cos_batch, sin_batch)
        k = _apply_rope(k, cos_batch, sin_batch)

        # Attention with flash_attn_with_kvcache (correct causal masking)
        from flash_attn import flash_attn_with_kvcache
        attn_output = flash_attn_with_kvcache(
            q=q,
            k_cache=cache.flash_k_caches[layer_idx],
            v_cache=cache.flash_v_caches[layer_idx],
            k=k, v=v_raw,
            cache_seqlens=cache.cache_seqlens,
        )  # [1, T, nq, hd]

        # O_proj batch INT4 GEMV
        attn_2d = attn_output.reshape(T, hs)
        o_proj = decoder_layer.self_attn.o_proj
        if hasattr(o_proj, 'weight_int4'):
            ng_o = _get_num_groups(o_proj)
            o_flat = gemv_ops.gemv_int4_batch(
                o_proj.weight_int4, attn_2d,
                o_proj.weight_scale_int4, ng_o, T)
        else:
            o_flat = F.linear(attn_2d, o_proj.weight)
        attn_output = o_flat.view(1, T, hs)

        # Residual + post-attention norm
        hidden = residual + attn_output
        residual = hidden
        hidden = _rms_norm(hidden, decoder_layer.post_attention_layernorm.weight,
                          decoder_layer.post_attention_layernorm.variance_epsilon)

        # gate_up batch SiLU+Mul INT4 GEMV
        hidden_2d = hidden.view(T, hs)
        gu_proj = decoder_layer.mlp.gate_up_proj
        if hasattr(gu_proj, 'weight_int4'):
            ng_gu = _get_num_groups(gu_proj)
            gate_up = gemv_ops.gemv_silu_mul_int4_batch(
                gu_proj.weight_int4, hidden_2d,
                gu_proj.weight_scale_int4, ng_gu, T)
        else:
            gate_up_raw = F.linear(hidden_2d, gu_proj.weight)
            d = gate_up_raw.shape[-1] // 2
            gate_up = (F.silu(gate_up_raw[..., :d]) * gate_up_raw[..., d:])

        # down_proj batch INT4 GEMV
        down_proj = decoder_layer.mlp.down_proj
        if hasattr(down_proj, 'weight_int4'):
            ng_down = _get_num_groups(down_proj)
            down_flat = gemv_ops.gemv_int4_batch(
                down_proj.weight_int4, gate_up,
                down_proj.weight_scale_int4, ng_down, T)
        else:
            down_flat = F.linear(gate_up, down_proj.weight)

        hidden = residual + down_flat.view(1, T, hs)

    # Final norm
    hidden = _rms_norm(hidden, language_model.norm.weight, eps)

    # lm_head batch INT4 GEMV
    hidden_2d = hidden.view(T, hs).half()
    if hasattr(lm_head, 'weight_int4'):
        ng_lm = _get_num_groups(lm_head)
        logits_flat = gemv_ops.gemv_lmhead_int4_batch(
            lm_head.weight_int4, hidden_2d,
            lm_head.weight_scale_int4, ng_lm, T)
    else:
        logits_flat = F.linear(hidden_2d, lm_head.weight)

    pred_tokens = logits_flat.argmax(dim=-1).cpu().tolist()
    return pred_tokens


def batch_verify_decode(decoder, language_model, lm_head, cache,
                        verify_tokens, rope_delta=0, gamma=4):
    """High-level API: run batch verify, update cache, return predictions.

    After verification:
    - Flash cache is updated with all T K/V entries (written by flash_attn_with_kvcache)
    - cache_seqlens is advanced by T (all entries)
    - Caller should set cache_seqlens to pos + k (accepted count) after accept/reject

    Args:
        decoder: CUDAGraphDecoder (has gemv_ops)
        language_model: INT4 language model
        lm_head: INT4 lm_head
        cache: GraphStaticKVCache with flash caches
        verify_tokens: list of int, length gamma+1
        rope_delta: int
        gamma: int

    Returns:
        pred_tokens: list of int, predictions for each of the T positions
    """
    gemv_ops = getattr(decoder, '_gemv_ops', None)
    if gemv_ops is None:
        from custom_kernels.cuda.gemv_loader import get_gemv_ops
        gemv_ops = get_gemv_ops()

    with torch.inference_mode():
        return batch_verify_forward(
            language_model, lm_head, cache, verify_tokens,
            rope_delta=rope_delta, gamma=gamma, gemv_ops=gemv_ops,
        )
