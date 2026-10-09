"""Batch speculative decoding: Medusa draft + INT4 batch verify.

Pipeline per cycle:
  1. CUDA graph decode_step  → next_token + hidden_buf
  2. MedusaHeads(hidden)     → 3 draft tokens (direct token prediction)
  3. Quick check: d0 == next_token?  (CPU, ~0 cost)
  4. batch_verify_v2          → INT4 batch GEMV through all layers
  5. Accept/reject + KV cache truncate

Works for any max_new_tokens — same code path for throughput and accuracy.
"""

import logging
import os
import time

import torch

from .batch_verify_v2 import batch_verify_forward
from .medusa_heads import load_medusa_heads

logger = logging.getLogger(__name__)


def batch_spec_generate(model, decoder, cache, input_ids, prefill_len,
                        rope_delta, first_token, last_hidden, max_new_tokens,
                        eos_token_ids):
    """Batch speculative decoding with Medusa heads + INT4 batch verify.

    Args:
        model:        full VLM model
        decoder:      CUDAGraphDecoder
        cache:        GraphStaticKVCache (flash_k/v_caches, cache_seqlens)
        input_ids:    [1, seq_len] prompt
        prefill_len:  prompt length after compaction
        rope_delta:   RoPE position offset
        first_token:  [1, 1] first generated token
        last_hidden:  [1, hidden_size] hidden state from prefill
        max_new_tokens: max tokens to generate
        eos_token_ids: EOS token IDs
    """
    device = input_ids.device
    gamma = int(os.environ.get("AICAS_BATCH_SPEC_GAMMA", "3"))
    log_on = os.environ.get("AICAS_BATCH_SPEC_LOG", "0") == "1"

    language_model = model.model.language_model
    lm_head = model.lm_head
    gemv_ops = decoder._gemv_ops
    if gemv_ops is None:
        from custom_kernels.cuda.gemv_loader import get_gemv_ops
        gemv_ops = get_gemv_ops()

    # Medusa draft heads
    medusa = load_medusa_heads(device)
    gamma = min(gamma, medusa.num_heads)

    cache_seqlens = cache.cache_seqlens
    eos_set = set(int(x) for x in (eos_token_ids or ()))

    # Output buffer
    token_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    token_buf[0] = first_token[0, 0]
    num_generated = 1

    # State
    cur_token = first_token.clone()
    step = 0

    # Stats
    cycles = 0
    accepted_total = 0
    drafted_total = 0
    t_verify_total = 0.0
    t_draft_total = 0.0

    while num_generated < max_new_tokens:
        remaining = max_new_tokens - num_generated

        # --- not enough room for a spec cycle: single-step fallback ---
        if remaining <= 2:
            tok_t = cur_token
            for _ in range(remaining):
                if num_generated >= max_new_tokens:
                    break
                next_tok = decoder.decode_step(tok_t, step)
                token_buf[num_generated] = next_tok[0, 0]
                num_generated += 1
                step += 1
                tok_t = next_tok
                if eos_set and tok_t[0, 0].item() in eos_set:
                    break
            break

        actual_gamma = min(gamma, remaining - 2)
        if actual_gamma <= 0:
            next_tok = decoder.decode_step(cur_token, step)
            token_buf[num_generated] = next_tok[0, 0]
            num_generated += 1
            step += 1
            cur_token = next_tok
            continue

        # ================================================================
        # Step 1: CUDA graph decode → next_token + hidden
        # ================================================================
        cache_pos = decoder.prefill_len + step
        next_token = decoder.decode_step(cur_token, step)
        hidden = decoder.hidden_buf[0]  # [hs] fp16

        token_buf[num_generated] = next_token[0, 0]
        num_generated += 1
        step += 1

        if num_generated >= max_new_tokens:
            break

        # ================================================================
        # Step 2: Draft — Medusa heads predict future tokens directly
        # ================================================================
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            draft_tokens = medusa(hidden)  # [gamma] long

        torch.cuda.synchronize()
        t_draft_total += time.perf_counter() - t0

        draft_list = draft_tokens.cpu().tolist()

        # ================================================================
        # Step 3: Quick check — does d0 match next_token?
        # ================================================================
        next_tok_id = next_token[0, 0].item()
        if draft_list[0] != next_tok_id:
            # First draft rejected — just continue with normal decode
            cur_token = next_token
            cycles += 1
            drafted_total += actual_gamma
            if log_on and cycles <= 10:
                print(f"  [bspec #{cycles}] pos={cache_pos} REJECT d0 "
                      f"(d0={draft_list[0]} tok={next_tok_id})", flush=True)
            continue

        # ================================================================
        # Step 4: Batch verify — [next_token, d1, d2, ..., d_{γ-1}]
        # ================================================================
        verify_ids = [next_tok_id] + draft_list[1:actual_gamma]
        T = len(verify_ids)

        # After decode_step: KV valid at [0, cache_pos]
        # Set cache_seqlens = cache_pos + 1 so verify reads [0, cache_pos]
        cache_seqlens.fill_(cache_pos + 1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            preds = batch_verify_forward(
                language_model, lm_head, cache, verify_ids,
                rope_delta=rope_delta, gamma=actual_gamma,
                gemv_ops=gemv_ops,
            )

        torch.cuda.synchronize()
        t_verify_total += time.perf_counter() - t0

        # ================================================================
        # Step 5: Accept/reject
        # ================================================================
        accept_len = 0
        for i in range(actual_gamma - 1):
            if preds[i] == draft_list[i + 1]:
                accept_len += 1
            else:
                break

        drafted_total += actual_gamma
        accepted_total += 1 + accept_len
        cycles += 1

        # Bonus token
        bonus_idx = min(accept_len, len(preds) - 1)
        bonus = preds[bonus_idx]

        if log_on and cycles <= 20:
            print(f"  [bspec #{cycles}] pos={cache_pos} γ={actual_gamma} "
                  f"accept={1 + accept_len} draft={draft_list[:actual_gamma]} "
                  f"pred={preds} bonus={bonus}", flush=True)

        # Write accepted draft tokens (d1..d{accept_len})
        for i in range(1, 1 + accept_len):
            if num_generated >= max_new_tokens:
                break
            token_buf[num_generated] = draft_list[i]
            num_generated += 1

        # Write bonus token
        if num_generated < max_new_tokens:
            token_buf[num_generated] = bonus
            num_generated += 1

        # ================================================================
        # Step 6: Update state
        # ================================================================
        new_cache_pos = cache_pos + 1 + accept_len + 1
        cache_seqlens.fill_(new_cache_pos)
        step = new_cache_pos - decoder.prefill_len

        cur_token = torch.tensor([[bonus]], device=device, dtype=torch.long)

        # EOS check
        if eos_set:
            for j in range(max(0, num_generated - accept_len - 2), num_generated):
                if token_buf[j].item() in eos_set:
                    num_generated = j + 1
                    break
            else:
                continue
            break

    # --- logging ---
    if log_on and cycles > 0:
        rate = accepted_total / drafted_total * 100 if drafted_total > 0 else 0
        avg_d = t_draft_total / cycles * 1000 if cycles else 0
        avg_v = t_verify_total / cycles * 1000 if cycles else 0
        print(
            f"[bspec] gen={num_generated} cycles={cycles} "
            f"accept={accepted_total}/{drafted_total} ({rate:.1f}%) "
            f"draft={avg_d:.2f}ms verify={avg_v:.2f}ms",
            flush=True,
        )

    return torch.cat([input_ids, token_buf[:num_generated].unsqueeze(0)], dim=-1)
