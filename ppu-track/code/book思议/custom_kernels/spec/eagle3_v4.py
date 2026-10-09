"""EAGLE-3 v4 — Clean rewrite of speculative decoding.

ROOT CAUSE of 0/150 accuracy (v3 bug):
  GraphStaticKVCache is in GRAPH MODE after CUDA graph decode.
  In graph mode, cache.update() writes K/V at `shared_position` (scalar),
  NOT at the `_pos_val` that _set_cache_positions() sets.
  For T>1 batch verify, index_copy_ fails silently or writes wrong position.
  FIX: disable graph mode for batch verify, re-enable after.

Architecture:
  1. Graph engine does prefill → first token + KV cache (fast, unchanged)
  2. EAGLE-3 takes over for remaining tokens:
     a. Bootstrap: get 3-layer hidden concat from first token position
     b. Draft: trained 1-layer model predicts gamma tokens
     c. Verify: batch forward through target model (graph mode OFF)
     d. Accept/reject: standard speculative decoding logic
     e. Update state for next cycle

Performance strategy:
  - Verify uses AICAS_EAGLE3_FAST_SKIP to skip middle layers (7 active instead of 28)
  - No hidden state collection during verify (only at accepted position for next draft)
  - Future: manual FP16 batch GEMM forward for tensor-core speedup
"""

import logging
import os
import time

import torch
import torch.nn.functional as F

from .batch_verify import (
    _set_cache_positions,
    _batch_lmhead,
)
from .eagle3_draft import load_eagle3_draft

logger = logging.getLogger(__name__)


def _safe_set_cache_pos(cache, pos):
    """Set cache position for ALL tracking mechanisms.

    GraphStaticKVCache has multiple position trackers:
    - cache_seqlens: used by flash_decode_attn
    - shared_position: used by update() in graph mode
    - layers[i]._pos_val: used by update() in normal mode
    - layers[i].position: used by... legacy code
    """
    if hasattr(cache, 'cache_seqlens'):
        cache.cache_seqlens.fill_(pos)
    if hasattr(cache, 'shared_position'):
        cache.shared_position.fill_(pos)
    for i in range(len(cache.layers)):
        cache.layers[i]._pos_val = pos
        cache.layers[i].position.fill_(pos)


def _disable_graph_mode(cache):
    """Temporarily disable graph mode for batch operations."""
    was_graph = getattr(cache, '_graph_mode', False)
    if was_graph:
        cache._graph_mode = False
        for layer in cache.layers:
            layer._graph_mode = False
    return was_graph


def _enable_graph_mode(cache):
    """Re-enable graph mode."""
    if hasattr(cache, '_graph_mode'):
        cache._graph_mode = True
        for layer in cache.layers:
            layer._graph_mode = True


def _batch_verify(language_model, lm_head, cache, verify_ids, cache_pos,
                  collect_hidden_layers=(0, 1, 2), rope_delta=0):
    """Batch verify with graph-mode safety.

    CRITICAL: Disables graph mode before forward, re-enables after.
    In graph mode, update() uses shared_position (scalar) which can't
    handle T>1 inputs. Normal mode uses _pos_val which is correctly
    set for batch operations.

    Returns:
        pred_tokens: list[int] — model prediction at each position
        hidden_final: [T, hidden_size] fp16 — final hidden states
        hiddens_3layer: [T, 3, hidden_size] fp16 — layers 0,1,2 for draft
    """
    device = language_model.embed_tokens.weight.device
    T = len(verify_ids)
    collect = collect_hidden_layers is not None

    # --- Disable graph mode for batch forward ---
    was_graph = _disable_graph_mode(cache)

    # Set position for ALL trackers
    _safe_set_cache_pos(cache, cache_pos)

    verify_tensor = torch.tensor([verify_ids], device=device, dtype=torch.long)

    # CRITICAL: provide position_ids with rope_delta for correct mRoPE.
    # Without this, RoPE positions are wrong → garbage output.
    # Format: [1, T] → model expands to [4, 1, T] internally for mRoPE.
    base_pos = cache_pos + rope_delta
    position_ids = torch.arange(
        base_pos, base_pos + T, device=device, dtype=torch.long
    ).unsqueeze(0)  # [1, T]

    with torch.no_grad():
        outputs = language_model(
            input_ids=verify_tensor,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=collect,
        )

    hidden = outputs.last_hidden_state  # [1, T, hs]
    h_flat = hidden[0].half()           # [T, hs]

    # lm_head
    logits = _batch_lmhead(lm_head, h_flat, None, T)
    pred_tokens = logits.argmax(dim=-1).cpu().tolist()

    # Extract 3-layer hiddens for draft model input
    hiddens_3layer = None
    if collect and outputs.hidden_states is not None:
        all_hs = outputs.hidden_states
        layer_hs = []
        for lid in collect_hidden_layers:
            if lid < len(all_hs):
                layer_hs.append(all_hs[lid][0].half())  # [T, hs]
            else:
                layer_hs.append(h_flat)
        hiddens_3layer = torch.stack(layer_hs, dim=1)  # [T, 3, hs]

    # --- Re-enable graph mode ---
    if was_graph:
        _enable_graph_mode(cache)

    return pred_tokens, h_flat, hiddens_3layer


def eagle3_v4_generate(model, decoder, cache, input_ids, prefill_len,
                       rope_delta, first_token, last_hidden, max_new_tokens,
                       eos_token_ids):
    """EAGLE-3 v4 speculative decoding.

    Args:
        model:        full VLM model
        decoder:      CUDAGraphDecoder (for gemv_ops)
        cache:        GraphStaticKVCache (from graph engine, in graph mode)
        input_ids:    [1, seq_len] prompt
        prefill_len:  prompt length after compaction
        rope_delta:   RoPE position offset
        first_token:  [1, 1] first generated token (from graph engine)
        last_hidden:  [1, hidden_size] or None
        max_new_tokens: max tokens to generate
        eos_token_ids: set of EOS token IDs
    """
    device = input_ids.device
    gamma = int(os.environ.get("AICAS_EAGLE_GAMMA", "5"))
    log_on = os.environ.get("AICAS_EAGLE_LOG", "0") == "1"

    language_model = model.model.language_model
    lm_head = model.lm_head

    # Layer skipping for fast verify
    fast_skip_str = os.environ.get(
        "AICAS_EAGLE3_FAST_SKIP",
        ""  # Empty = no skipping (full 28 layers)
    ).strip()

    # Load draft model
    draft = load_eagle3_draft(device)
    gamma = min(gamma, draft.ttt_length)

    eos_set = set(int(x) for x in (eos_token_ids or ()))

    # Output buffer
    token_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    token_buf[0] = first_token[0, 0]
    num_generated = 1

    # State
    cur_token = first_token[0, 0].item()
    cache_pos = prefill_len  # first token was decoded by graph engine at this pos

    # ==================================================================
    # BOOTSTRAP: get 3-layer hidden concat from the first token position
    # ==================================================================
    # The graph engine already processed first_token at cache_pos.
    # We re-process to collect intermediate layer hidden states.
    # CRITICAL: disable graph mode for this single-token forward too,
    # because _set_cache_positions sets _pos_val but NOT shared_position.
    with torch.no_grad():
        preds, _, hiddens_3layer = _batch_verify(
            language_model, lm_head, cache,
            [cur_token], cache_pos,
            collect_hidden_layers=(0, 1, 2),
            rope_delta=rope_delta,
        )

    # 3-layer hidden at bootstrap position
    cur_hidden_concat = torch.cat(
        [hiddens_3layer[0, i] for i in range(3)], dim=-1
    ).unsqueeze(0).half()  # [1, 6144]

    # The graph engine's first token prediction should match our verify
    # Use the verify's prediction as the "next" token
    bonus_token = preds[0]

    cache_pos += 1  # advance past first token
    if num_generated < max_new_tokens:
        token_buf[num_generated] = bonus_token
        num_generated += 1
    cur_token = bonus_token

    # Apply fast skip env var for subsequent verifies
    if fast_skip_str:
        os.environ["AICAS_BATCH_VERIFY_SKIP_LAYERS"] = fast_skip_str

    if log_on:
        print(f"  [eagle3v4 bootstrap] first={first_token[0,0].item()} "
              f"bonus={bonus_token} cache_pos={cache_pos} gamma={gamma}", flush=True)

    # Stats
    accepted_total = 0
    drafted_total = 0
    cycle_count = 0

    # ==================================================================
    # MAIN LOOP
    # ==================================================================
    try:
        while num_generated < max_new_tokens:
            remaining = max_new_tokens - num_generated
            if remaining <= 1:
                # Last token: single forward for guaranteed correctness
                was_graph = _disable_graph_mode(cache)
                _safe_set_cache_pos(cache, cache_pos)
                with torch.no_grad():
                    tok = torch.tensor([[cur_token]], device=device, dtype=torch.long)
                    # position_ids for single-token mRoPE
                    mrope_pos = cache_pos + rope_delta
                    pos_ids = torch.tensor([[mrope_pos]], device=device, dtype=torch.long)
                    out = language_model(
                        input_ids=tok, position_ids=pos_ids,
                        past_key_values=cache,
                        use_cache=True, output_hidden_states=True,
                    )
                if was_graph:
                    _enable_graph_mode(cache)

                h = out.last_hidden_state[0, 0].half()
                logits = F.linear(h, lm_head.weight) if hasattr(lm_head, 'weight') else lm_head(h.unsqueeze(0)).squeeze(0)
                nt = logits.argmax().item()
                token_buf[num_generated] = nt
                num_generated += 1
                break

            actual_gamma = min(gamma, remaining - 1)
            if actual_gamma <= 0:
                break

            # ============================================================
            # DRAFT: predict future tokens
            # ============================================================
            draft_tokens, draft_logits = draft.draft_tokens(
                cur_hidden_concat, cur_token, actual_gamma,
                position_offset=cache_pos,
            )

            # ============================================================
            # VERIFY: batch forward through target model
            # ============================================================
            verify_ids = [cur_token] + draft_tokens[:actual_gamma]

            pred_tokens, _, hiddens_3layer = _batch_verify(
                language_model, lm_head, cache, verify_ids, cache_pos,
                collect_hidden_layers=(0, 1, 2),
                rope_delta=rope_delta,
            )

            # ============================================================
            # ACCEPT / REJECT
            # ============================================================
            accept_len = 0
            for i in range(actual_gamma):
                if pred_tokens[i] == draft_tokens[i]:
                    accept_len += 1
                else:
                    break

            drafted_total += actual_gamma
            accepted_total += accept_len
            cycle_count += 1

            # Bonus token at first mismatch (or last position)
            bonus_idx = min(accept_len, len(pred_tokens) - 1)
            bonus_token = pred_tokens[bonus_idx]

            if log_on and cycle_count <= 10:
                print(f"  [eagle3v4 #{cycle_count}] pos={cache_pos} γ={actual_gamma} "
                      f"accept={accept_len} draft={draft_tokens[:4]} pred={pred_tokens[:4]}", flush=True)

            # ============================================================
            # WRITE TOKENS
            # ============================================================
            for i in range(min(accept_len, remaining)):
                token_buf[num_generated] = draft_tokens[i]
                num_generated += 1
                if num_generated >= max_new_tokens:
                    break

            if num_generated < max_new_tokens:
                token_buf[num_generated] = bonus_token
                num_generated += 1

            # ============================================================
            # UPDATE STATE
            # ============================================================
            new_cache_pos = cache_pos + accept_len + 1

            # Update hidden concat for next draft
            if hiddens_3layer is not None:
                concat_idx = min(accept_len, hiddens_3layer.shape[0] - 1)
                cur_hidden_concat = torch.cat(
                    [hiddens_3layer[concat_idx, i] for i in range(3)], dim=-1
                ).unsqueeze(0).half()

            cur_token = bonus_token
            cache_pos = new_cache_pos

            # Fix cache positions: verify may have advanced _pos_val beyond accept_len
            # Set back to the correct position for the next cycle
            was_graph = _disable_graph_mode(cache)
            _safe_set_cache_pos(cache, cache_pos)
            if was_graph:
                _enable_graph_mode(cache)

            # EOS check
            if eos_set:
                found_eos = False
                for j in range(max(0, num_generated - accept_len - 1), num_generated):
                    if token_buf[j].item() in eos_set:
                        num_generated = j + 1
                        found_eos = True
                        break
                if found_eos:
                    break

    finally:
        # Clean up env var
        os.environ.pop("AICAS_BATCH_VERIFY_SKIP_LAYERS", None)

    # --- Logging ---
    if log_on and drafted_total > 0:
        rate = accepted_total / drafted_total * 100
        print(
            f"[eagle3v4] gen={num_generated} cycles={cycle_count} "
            f"accept={accepted_total}/{drafted_total} ({rate:.1f}%)",
            flush=True,
        )

    return torch.cat([input_ids, token_buf[:num_generated].unsqueeze(0)], dim=-1)
