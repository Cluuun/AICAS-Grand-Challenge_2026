"""Batch verification for speculative decoding.

Uses the model's own forward pass for correctness, with batch lm_head for
token predictions.  The model's patched attention writes to the flash KV
cache, keeping the cache consistent with CUDA graph decode steps.

Two-tier verification for EAGLE-3 v3:
- batch_verify_fast(): Reduced-layer (e.g. 7 active) fast verification
- full_verify_single(): Full 28-layer single-token verification at rejection
"""

import os
import torch
import torch.nn.functional as F


def _num_groups(module):
    """Derive INT4 group count from the scale tensor shape."""
    if hasattr(module, 'weight_scale_int4'):
        return module.weight_scale_int4.shape[1]
    return module.weight.shape[-1] // 128


def _set_cache_positions(cache, cache_pos):
    """Set all cache position trackers so the model forward reads/writes correctly."""
    cache.cache_seqlens.fill_(cache_pos)
    for i in range(len(cache.layers)):
        cache.layers[i]._pos_val = cache_pos
        cache.layers[i].position.fill_(cache_pos)


def _batch_lmhead(lm_head, hidden_flat, gemv_ops, T):
    """Run batch lm_head on [T, hs] flat hidden states → [T, vocab] logits.

    Uses F.linear unconditionally — the INT4 batch kernel only supports
    specific batch sizes and is not needed for speculative verify (which
    is not the throughput bottleneck).
    """
    if hasattr(lm_head, 'weight'):
        return F.linear(hidden_flat, lm_head.weight)
    else:
        return lm_head(hidden_flat.unsqueeze(0)).squeeze(0)


def batch_verify_forward(language_model, lm_head, cache, verify_ids,
                         cache_pos, gemv_ops):
    """Run a full-model batch forward for T = len(verify_ids) tokens.

    Delegates to language_model() for the forward pass — this uses the exact
    same code path as prefill (fused ops, correct RoPE, flash KV cache writes).
    Caller must set cache.cache_seqlens to the accepted count afterwards.

    Args:
        language_model: model.model.language_model
        lm_head:       model.lm_head
        cache:         GraphStaticKVCache (has flash_k/v_caches, cache_seqlens)
        verify_ids:    list[int] of length T (cur_token + draft tokens)
        cache_pos:     int, starting position in KV cache
        gemv_ops:      GEMV C++ extension (or None)

    Returns:
        pred_tokens: list[int] — model prediction at each position
        hiddens:     [T, hidden_size] fp16 — final-layer hidden states
    """
    device = language_model.embed_tokens.weight.device
    T = len(verify_ids)

    _set_cache_positions(cache, cache_pos)

    verify_tensor = torch.tensor([verify_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = language_model(
            input_ids=verify_tensor,
            past_key_values=cache,
            use_cache=True,
        )

    hidden = outputs.last_hidden_state                  # [1, T, hs]

    h2 = hidden.view(T, -1).half()
    logits = _batch_lmhead(lm_head, h2, gemv_ops, T)

    pred_tokens = logits.argmax(dim=-1).cpu().tolist()
    return pred_tokens, hidden[0].half()                # predictions, [T, hs]


def batch_verify_with_hiddens(language_model, lm_head, cache, verify_ids,
                              cache_pos, gemv_ops, hidden_layer_ids=(0, 1, 2)):
    """Like batch_verify_forward, but also returns intermediate-layer hidden states.

    Used by EAGLE-3 which needs layers [0, 1, 2] hidden concat as draft input.

    Returns:
        pred_tokens:  list[int] — model prediction at each position
        hiddens:      [T, hidden_size] fp16 — final-layer hidden states
        hiddens_3layer: [T, num_layers, hidden_size] fp16 — requested layer hiddens
    """
    device = language_model.embed_tokens.weight.device
    T = len(verify_ids)

    _set_cache_positions(cache, cache_pos)

    verify_tensor = torch.tensor([verify_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = language_model(
            input_ids=verify_tensor,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )

    hidden = outputs.last_hidden_state                  # [1, T, hs]

    # Extract requested layer hidden states
    # outputs.hidden_states: tuple of [1, T, hs], indexed by layer
    all_hiddens = outputs.hidden_states
    layer_hiddens = []
    for lid in hidden_layer_ids:
        if lid < len(all_hiddens):
            layer_hiddens.append(all_hiddens[lid][0].half())  # [T, hs]
        else:
            # Fallback: use final hidden for missing layers
            layer_hiddens.append(hidden[0].half())
    # Stack: [T, num_requested_layers, hs]
    hiddens_3layer = torch.stack(layer_hiddens, dim=1)

    # lm_head
    h2 = hidden.view(T, -1).half()
    logits = _batch_lmhead(lm_head, h2, gemv_ops, T)

    pred_tokens = logits.argmax(dim=-1).cpu().tolist()
    return pred_tokens, hidden[0].half(), hiddens_3layer


# ---------------------------------------------------------------------------
# Config extraction helper (still needed by eagle_decode)
# ---------------------------------------------------------------------------

def extract_model_config(language_model):
    """Extract model dimensions needed by callers."""
    config = getattr(language_model.config, 'text_config', language_model.config)
    hs  = config.hidden_size
    nq  = config.num_attention_heads
    nkv = config.num_key_value_heads
    hd  = getattr(config, 'head_dim', hs // nq)
    return {
        'hidden_size':  hs,
        'num_q_heads':  nq,
        'num_kv_heads': nkv,
        'head_dim':     hd,
        'rope_theta':   float(getattr(config, 'rope_theta', 1_000_000.0)),
        'norm_eps':     float(getattr(config, 'rms_norm_eps', 1e-6)),
    }


# ---------------------------------------------------------------------------
# Two-tier verification for EAGLE-3 v3
# ---------------------------------------------------------------------------

def batch_verify_fast(language_model, lm_head, cache, verify_ids,
                      cache_pos, gemv_ops, skip_layers_str):
    """Reduced-layer fast verification for EAGLE-3 acceptance path.

    Temporarily sets AICAS_BATCH_VERIFY_SKIP_LAYERS so patched_forward
    skips the specified layers during the batch forward.  This is the
    "fast tier" — covering ~95%+ of tokens with ~1ms latency.

    Args:
        skip_layers_str: comma-separated layer indices to SKIP (e.g. "5,6,...,25")
                         Layers NOT in this set are executed (active layers).

    Returns:
        Same as batch_verify_with_hiddens:
        (pred_tokens, hiddens, hiddens_3layer)
    """
    old_val = os.environ.get("AICAS_BATCH_VERIFY_SKIP_LAYERS", "")
    os.environ["AICAS_BATCH_VERIFY_SKIP_LAYERS"] = skip_layers_str
    try:
        result = batch_verify_with_hiddens(
            language_model, lm_head, cache, verify_ids,
            cache_pos, gemv_ops)
    finally:
        if old_val:
            os.environ["AICAS_BATCH_VERIFY_SKIP_LAYERS"] = old_val
        else:
            os.environ.pop("AICAS_BATCH_VERIFY_SKIP_LAYERS", None)
    return result


def full_verify_single(language_model, lm_head, cache, token_id,
                       cache_pos, gemv_ops):
    """Full 28-layer single-token forward for guaranteed-correct output.

    This is the "slow tier" — runs at rejection points and periodic resyncs.
    Processes a single token through ALL 28 layers (no skipping), ensuring
    the prediction and KV cache entry are identical to what the full model
    would produce.

    Returns:
        pred_token:     int — model's predicted next token
        hidden:         [hidden_size] fp16 — final hidden state
        hiddens_3layer: [3, hidden_size] fp16 — layers 0,1,2 for draft input
    """
    device = language_model.embed_tokens.weight.device

    _set_cache_positions(cache, cache_pos)

    tok = torch.tensor([[token_id]], device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = language_model(
            input_ids=tok,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )

    hidden = outputs.last_hidden_state[0, 0].half()   # [hs]

    # lm_head (single token, use F.linear for simplicity)
    if hasattr(lm_head, 'weight'):
        logits = F.linear(hidden, lm_head.weight)
    else:
        logits = lm_head(hidden.unsqueeze(0)).squeeze(0)
    pred_token = logits.argmax(dim=-1).item()

    # Extract 3-layer hiddens for draft model input
    all_hs = outputs.hidden_states
    if all_hs is not None and len(all_hs) >= 3:
        h3 = torch.stack([all_hs[i][0, 0].half() for i in range(3)], dim=0)  # [3, hs]
    else:
        # Fallback: repeat final hidden
        h3 = hidden.unsqueeze(0).expand(3, -1)

    return pred_token, hidden, h3
