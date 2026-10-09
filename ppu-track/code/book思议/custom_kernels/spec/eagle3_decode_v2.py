"""EAGLE-3 v3 decode loop — optimized for maximum throughput.

Key optimizations over v1:
  1. No output_hidden_states — verify uses only final-layer hidden
  2. Draft model takes single hidden (not 3-layer concat)
  3. Two-tier verify: fast skip-layer for accept, full model for rejection
  4. Minimal Python overhead in the hot loop
"""

import logging
import os
import time

import torch
import torch.nn.functional as F

from .batch_verify import (
    batch_verify_forward,
    full_verify_single,
    _set_cache_positions,
    _batch_lmhead,
)
from .eagle3_draft_v2 import load_eagle3_draft_v2

logger = logging.getLogger(__name__)


def eagle3_decode_generate(model, decoder, cache, input_ids, prefill_len,
                           rope_delta, first_token, last_hidden, max_new_tokens,
                           eos_token_ids):
    """EAGLE-3 speculative decoding — maximum throughput version."""
    device = input_ids.device
    gamma = int(os.environ.get("AICAS_EAGLE_GAMMA", "7"))
    log_on = os.environ.get("AICAS_EAGLE_LOG", "0") == "1"

    language_model = model.model.language_model
    lm_head = model.lm_head
    gemv_ops = getattr(decoder, '_gemv_ops', None)

    # Two-tier config
    two_tier = os.environ.get("AICAS_EAGLE3_TWO_TIER", "0") == "1"
    fast_skip_layers = os.environ.get(
        "AICAS_EAGLE3_FAST_SKIP_LAYERS",
        "5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
    ).strip()
    resync_interval = int(os.environ.get("AICAS_EAGLE3_RESYNC_INTERVAL", "20"))

    # Load draft model
    draft = load_eagle3_draft_v2(device)
    gamma = min(gamma, draft.ttt_length)

    # Flash KV cache
    eos_set = set(int(x) for x in (eos_token_ids or ()))

    # Output buffer
    token_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    token_buf[0] = first_token[0, 0]
    num_generated = 1

    cur_token = first_token[0, 0].item()
    cache_pos = prefill_len

    # ================================================================
    # BOOTSTRAP: process first_token to get initial hidden state
    # Using fast single-token verify (no hidden_states collection)
    # ================================================================
    _set_cache_positions(cache, cache_pos)
    tok_tensor = torch.tensor([[cur_token]], device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = language_model(
            input_ids=tok_tensor,
            past_key_values=cache,
            use_cache=True,
        )
    hidden = outputs.last_hidden_state[0, 0].half()  # [hs]

    # lm_head for bonus token
    if hasattr(lm_head, 'weight'):
        logits = F.linear(hidden, lm_head.weight)
    else:
        logits = lm_head(hidden.unsqueeze(0)).squeeze(0)
    bonus_from_bootstrap = logits.argmax(dim=-1).item()

    # Single hidden for draft input
    cur_hidden = hidden.unsqueeze(0).half()  # [1, hs]

    cache_pos += 1
    num_generated += 1
    if num_generated <= max_new_tokens:
        token_buf[num_generated - 1] = bonus_from_bootstrap
    cur_token = bonus_from_bootstrap
    _set_cache_positions(cache, cache_pos)

    if log_on:
        print(f"  [eagle3] bootstrap done: bonus={bonus_from_bootstrap} "
              f"gamma={gamma} two_tier={two_tier} resync={resync_interval}", flush=True)

    # Stats
    accepted_total = 0
    drafted_total = 0
    cycle_count = 0

    # ================================================================
    # MAIN LOOP
    # ================================================================
    while num_generated < max_new_tokens:
        remaining = max_new_tokens - num_generated

        if remaining <= 1:
            _set_cache_positions(cache, cache_pos)
            tok_t = torch.tensor([[cur_token]], device=device, dtype=torch.long)
            with torch.no_grad():
                out = language_model(input_ids=tok_t, past_key_values=cache, use_cache=True)
            h = out.last_hidden_state[0, 0].half()
            if hasattr(lm_head, 'weight'):
                tok = F.linear(h, lm_head.weight).argmax(dim=-1).item()
            else:
                tok = lm_head(h.unsqueeze(0)).squeeze(0).argmax(dim=-1).item()
            token_buf[num_generated] = tok
            num_generated += 1
            break

        actual_gamma = min(gamma, remaining - 1)
        if actual_gamma <= 0:
            break

        # --- DRAFT ---
        draft_tokens_list, draft_logits = draft.draft_tokens(
            cur_hidden, cur_token, actual_gamma,
            position_offset=cache_pos,
        )

        # --- VERIFY (batch forward, NO output_hidden_states) ---
        verify_ids = [cur_token] + draft_tokens_list[:actual_gamma]
        need_resync = (resync_interval > 0 and
                       cycle_count > 0 and
                       cycle_count % resync_interval == 0)

        if two_tier and not need_resync:
            # Fast tier: set skip layers env var temporarily
            old_val = os.environ.get("AICAS_BATCH_VERIFY_SKIP_LAYERS", "")
            os.environ["AICAS_BATCH_VERIFY_SKIP_LAYERS"] = fast_skip_layers
            try:
                with torch.no_grad():
                    pred_tokens, verify_hidden = batch_verify_forward(
                        language_model, lm_head, cache, verify_ids,
                        cache_pos, gemv_ops,
                    )
            finally:
                if old_val:
                    os.environ["AICAS_BATCH_VERIFY_SKIP_LAYERS"] = old_val
                else:
                    os.environ.pop("AICAS_BATCH_VERIFY_SKIP_LAYERS", None)
        else:
            # Full verify: all 28 layers, no hidden_states
            with torch.no_grad():
                pred_tokens, verify_hidden = batch_verify_forward(
                    language_model, lm_head, cache, verify_ids,
                    cache_pos, gemv_ops,
                )

        # --- ACCEPT / REJECT ---
        accept_len = 0
        for i in range(actual_gamma):
            if pred_tokens[i] == draft_tokens_list[i]:
                accept_len += 1
            else:
                break

        drafted_total += actual_gamma
        accepted_total += accept_len
        cycle_count += 1

        # Bonus token at first mismatch position
        bonus_idx = min(accept_len, len(pred_tokens) - 1)
        bonus_token = pred_tokens[bonus_idx]

        if log_on and cycle_count <= 5:
            print(f"  [eagle3 #{cycle_count}] γ={actual_gamma} accept={accept_len} "
                  f"draft={draft_tokens_list[:4]} pred={pred_tokens[:4]}", flush=True)

        # --- TWO-TIER REJECTION: full model correct at rejection point ---
        if two_tier and accept_len < actual_gamma:
            reject_pos = cache_pos + accept_len
            with torch.no_grad():
                full_tok, full_h, _ = full_verify_single(
                    language_model, lm_head, cache,
                    draft_tokens_list[accept_len] if accept_len < actual_gamma else cur_token,
                    reject_pos, gemv_ops,
                )
            bonus_token = full_tok
            cur_hidden = full_h.unsqueeze(0).half()

        # --- WRITE TOKENS ---
        for i in range(min(accept_len, remaining)):
            token_buf[num_generated] = draft_tokens_list[i]
            num_generated += 1

        if num_generated < max_new_tokens:
            token_buf[num_generated] = bonus_token
            num_generated += 1

        # --- UPDATE STATE ---
        new_cache_pos = cache_pos + accept_len + 1

        # Update hidden for next draft (if not set by rejection path)
        if not (two_tier and accept_len < actual_gamma):
            h_idx = min(accept_len, verify_hidden.shape[0] - 1)
            cur_hidden = verify_hidden[h_idx].unsqueeze(0).half()

        cur_token = bonus_token
        cache_pos = new_cache_pos
        _set_cache_positions(cache, cache_pos)

        # EOS check
        if eos_set:
            for j in range(max(0, num_generated - accept_len - 1), num_generated):
                if token_buf[j].item() in eos_set:
                    num_generated = j + 1
                    break
            else:
                continue
            break

    # --- logging ---
    if log_on and drafted_total > 0:
        rate = accepted_total / drafted_total * 100
        print(
            f"[eagle3] gen={num_generated} cycles={cycle_count} "
            f"accept={accepted_total}/{drafted_total} ({rate:.1f}%)",
            flush=True,
        )

    return torch.cat([input_ids, token_buf[:num_generated].unsqueeze(0)], dim=-1)
