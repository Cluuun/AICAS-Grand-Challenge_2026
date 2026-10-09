"""EAGLE-3 v3 speculative decoding loop with two-tier verification.

Architecture:
  1. Draft model (1 Transformer layer) predicts gamma future tokens
  2. Fast verify: reduced-layer (e.g. 7 active) batch verification
  3. On rejection: full 28-layer single-token verification for correct output
  4. Periodic full-model resync to keep KV cache aligned

Key difference from v2:
  - No tail-graph fallback (EAGLE-3 handles ALL decode steps)
  - Two-tier verification: fast 7L for accept path, full 28L at rejection
  - Larger gamma (10+) for higher throughput
  - Configurable via AICAS_EAGLE3_FAST_SKIP_LAYERS and AICAS_EAGLE3_RESYNC_INTERVAL
"""

import logging
import os
import time

import torch

from .batch_verify import (
    batch_verify_with_hiddens,
    batch_verify_fast,
    full_verify_single,
    extract_model_config,
    _set_cache_positions,
)
from .eagle3_draft import load_eagle3_draft

logger = logging.getLogger(__name__)


def eagle3_decode_generate(model, decoder, cache, input_ids, prefill_len,
                           rope_delta, first_token, last_hidden, max_new_tokens,
                           eos_token_ids):
    """EAGLE-3 v3 speculative decoding with two-tier verification.

    Args:
        model:        full VLM model (has .model.language_model, .lm_head)
        decoder:      CUDAGraphDecoder (used for gemv_ops)
        cache:        GraphStaticKVCache
        input_ids:    [1, seq_len] prompt
        prefill_len:  prompt length after compaction
        rope_delta:   RoPE position offset
        first_token:  [1, 1] first generated token
        last_hidden:  [1, hidden_size] hidden state from prefill
        max_new_tokens: max tokens to generate
        eos_token_ids: list/tuple of EOS token IDs
    """
    device = input_ids.device
    gamma = int(os.environ.get("AICAS_EAGLE_GAMMA", "10"))
    log_on = os.environ.get("AICAS_EAGLE_LOG", "0") == "1"

    language_model = model.model.language_model
    lm_head = model.lm_head
    gemv_ops = getattr(decoder, '_gemv_ops', None)

    # Two-tier verification configuration
    two_tier = os.environ.get("AICAS_EAGLE3_TWO_TIER", "0") == "1"
    fast_skip_layers = os.environ.get(
        "AICAS_EAGLE3_FAST_SKIP_LAYERS",
        "5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
    ).strip()
    resync_interval = int(os.environ.get("AICAS_EAGLE3_RESYNC_INTERVAL", "20"))

    # Load EAGLE-3 draft model
    draft = load_eagle3_draft(device)
    gamma = min(gamma, draft.ttt_length)

    # Flash KV cache references
    seqlens = cache.cache_seqlens

    eos_set = set(int(x) for x in (eos_token_ids or ()))

    # Output buffer
    token_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    token_buf[0] = first_token[0, 0]
    num_generated = 1

    # Running state
    cur_token = first_token[0, 0].item()
    cache_pos = prefill_len

    # ==================================================================
    # BOOTSTRAP: process first_token to get initial 3-layer hidden concat
    # ==================================================================
    with torch.no_grad():
        bootstrap_preds, _, hiddens_3layer = batch_verify_with_hiddens(
            language_model, lm_head, cache,
            [cur_token], cache_pos, gemv_ops,
        )

    # 3-layer hidden at bootstrap position
    cur_hidden_concat = torch.cat(
        [hiddens_3layer[0, i] for i in range(3)], dim=-1
    ).unsqueeze(0).half()

    # Bonus token from bootstrap
    bonus_from_bootstrap = bootstrap_preds[0]
    cache_pos += 1
    num_generated += 1
    if num_generated <= max_new_tokens:
        token_buf[num_generated - 1] = bonus_from_bootstrap
    cur_token = bonus_from_bootstrap

    # Update cache positions
    _set_cache_positions(cache, cache_pos)

    if log_on:
        print(f"  [eagle3 bootstrap] first_token={first_token[0,0].item()} "
              f"bonus={bonus_from_bootstrap} cache_pos={cache_pos} "
              f"two_tier={two_tier} gamma={gamma} resync={resync_interval}", flush=True)

    # Stats
    accepted_total = 0
    drafted_total = 0
    resync_count = 0
    full_verify_count = 0

    # ==================================================================
    # MAIN LOOP: EAGLE-3 speculative decoding with two-tier verify
    # ==================================================================
    while num_generated < max_new_tokens:
        remaining = max_new_tokens - num_generated

        if remaining <= 1:
            # Last token: do a full-model forward for guaranteed correctness
            with torch.no_grad():
                full_tok, full_h, full_h3 = full_verify_single(
                    language_model, lm_head, cache,
                    cur_token, cache_pos, gemv_ops)
            token_buf[num_generated] = full_tok
            num_generated += 1
            break

        actual_gamma = min(gamma, remaining - 1)
        if actual_gamma <= 0:
            break

        # ================================================================
        # DRAFT: predict future tokens using EAGLE-3 model
        # ================================================================
        draft_tokens_list, draft_logits = draft.draft_tokens(
            cur_hidden_concat, cur_token, actual_gamma,
            position_offset=cache_pos,
        )

        # ================================================================
        # VERIFY (two-tier or full)
        # ================================================================
        verify_ids = [cur_token] + draft_tokens_list[:actual_gamma]
        need_resync = (resync_interval > 0 and
                       resync_count > 0 and
                       resync_count % resync_interval == 0)

        if two_tier and not need_resync:
            # --- Fast tier: reduced-layer batch verify ---
            with torch.no_grad():
                pred_tokens, verify_hidden, hiddens_3layer = batch_verify_fast(
                    language_model, lm_head, cache, verify_ids,
                    cache_pos, gemv_ops, fast_skip_layers,
                )
        else:
            # --- Full verify: all 28 layers ---
            with torch.no_grad():
                pred_tokens, verify_hidden, hiddens_3layer = batch_verify_with_hiddens(
                    language_model, lm_head, cache, verify_ids,
                    cache_pos, gemv_ops,
                )

        # ================================================================
        # ACCEPT / REJECT
        # ================================================================
        accept_len = 0
        for i in range(actual_gamma):
            if pred_tokens[i] == draft_tokens_list[i]:
                accept_len += 1
            else:
                break

        drafted_total += actual_gamma
        accepted_total += accept_len
        resync_count += 1

        # Bonus token: at first mismatch position
        bonus_idx = min(accept_len, len(pred_tokens) - 1)
        bonus_token = pred_tokens[bonus_idx]

        if log_on and resync_count <= 5:
            tier = "fast" if (two_tier and not need_resync) else "full"
            print(f"  [eagle3 #{resync_count}] pos={cache_pos} γ={actual_gamma} "
                  f"accept={accept_len} tier={tier} "
                  f"draft={draft_tokens_list[:4]} pred={pred_tokens[:4]}", flush=True)

        # ================================================================
        # REJECTION PATH: full-model verify if needed
        # ================================================================
        if two_tier and accept_len < actual_gamma:
            # Fast verify rejected — run full model at rejection point
            # to get guaranteed-correct bonus token and fix KV cache.
            reject_pos = cache_pos + accept_len
            with torch.no_grad():
                full_tok, full_h, full_h3 = full_verify_single(
                    language_model, lm_head, cache,
                    draft_tokens_list[accept_len] if accept_len < actual_gamma else cur_token,
                    reject_pos, gemv_ops,
                )
            bonus_token = full_tok
            full_verify_count += 1

            # Use full model's 3-layer hidden for next draft input
            cur_hidden_concat = torch.cat(
                [full_h3[i] for i in range(3)], dim=-1
            ).unsqueeze(0).half()

        # ================================================================
        # WRITE TOKENS
        # ================================================================
        for i in range(min(accept_len, remaining)):
            token_buf[num_generated] = draft_tokens_list[i]
            num_generated += 1

        if num_generated < max_new_tokens:
            token_buf[num_generated] = bonus_token
            num_generated += 1

        # ================================================================
        # UPDATE STATE
        # ================================================================
        new_cache_pos = cache_pos + accept_len + 1

        # Update hidden concat for next draft (if not already set by rejection path)
        if not (two_tier and accept_len < actual_gamma):
            concat_idx = min(accept_len, hiddens_3layer.shape[0] - 1)
            cur_hidden_concat = torch.cat(
                [hiddens_3layer[concat_idx, i] for i in range(3)], dim=-1
            ).unsqueeze(0).half()

        cur_token = bonus_token
        cache_pos = new_cache_pos

        # Update cache positions for all layers
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
            f"[eagle3] gen={num_generated} cycles={resync_count} "
            f"accept={accepted_total}/{drafted_total} ({rate:.1f}%) "
            f"full_verify={full_verify_count}",
            flush=True,
        )

    return torch.cat([input_ids, token_buf[:num_generated].unsqueeze(0)], dim=-1)
