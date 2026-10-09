"""eP-EAGLE speculative decoding loop.

Uses a lightweight draft model to predict future hidden states, then verifies
all draft tokens in a single batched INT4 forward pass.  Output distribution is
identical to autoregressive decoding (lossless).

Works for any max_new_tokens — same code path for throughput and accuracy.
"""

import logging
import os
import time

import torch

from .batch_verify import batch_verify_forward, extract_model_config
from .draft_head import load_draft_head

logger = logging.getLogger(__name__)


def eagle_decode_generate(model, decoder, cache, input_ids, prefill_len,
                          rope_delta, first_token, last_hidden, max_new_tokens,
                          eos_token_ids):
    """EAGLE speculative decoding with batch INT4 verification.

    Args:
        model:        full VLM model (has .model.language_model, .lm_head)
        decoder:      CUDAGraphDecoder (used for single-step fallback)
        cache:        GraphStaticKVCache (has flash_k/v_caches, cache_seqlens)
        input_ids:    [1, seq_len] prompt
        prefill_len:  prompt length after compaction
        rope_delta:   RoPE position offset
        first_token:  [1, 1] first generated token
        last_hidden:  [1, hidden_size] hidden state from prefill
        max_new_tokens: max tokens to generate
        eos_token_ids: list/tuple of EOS token IDs
    """
    device = input_ids.device
    gamma = int(os.environ.get("AICAS_EAGLE_GAMMA", "4"))
    log_on = os.environ.get("AICAS_EAGLE_LOG", "0") == "1"

    language_model = model.model.language_model
    lm_head = model.lm_head
    gemv_ops = getattr(decoder, '_gemv_ops', None)

    # Model config for batch verify
    cfg = extract_model_config(language_model)

    # Draft model
    draft = load_draft_head(device)
    max_draft = draft.horizon - 1                     # position 0 = reconstruction
    gamma = min(gamma, max_draft)

    # Flash KV cache references (set by decoder.setup_for_decode)
    seqlens = cache.cache_seqlens                     # [1] int32

    eos_set = set(int(x) for x in (eos_token_ids or ()))

    # Output buffer
    token_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    token_buf[0] = first_token[0, 0]
    num_generated = 1

    # Running state
    cur_token = first_token[0, 0].item()
    cur_hidden = last_hidden[0].float()               # [hs], float32 for draft
    cache_pos = prefill_len

    # Stats
    accepted_total = 0
    drafted_total = 0
    cycles = 0
    t_draft_total = 0.0
    t_verify_total = 0.0

    while num_generated < max_new_tokens:
        remaining = max_new_tokens - num_generated

        # --- not enough room for a full spec cycle: single-step fallback ---
        if remaining <= 1:
            tok_t = torch.tensor([[cur_token]], device=device, dtype=torch.long)
            step = num_generated - 1
            next_tok = decoder.decode_step(tok_t, step)
            bonus = next_tok[0, 0].item()
            token_buf[num_generated] = bonus
            num_generated += 1
            break

        actual_gamma = min(gamma, remaining - 1)
        if actual_gamma <= 0:
            break

        # ================================================================
        # DRAFT: predict future hidden states → draft tokens
        # ================================================================
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            future_hiddens = draft(cur_hidden.unsqueeze(0))   # [1, horizon, hs]

        # Convert predicted hiddens to draft tokens via lm_head
        draft_h = future_hiddens[0, 1:actual_gamma + 1].half()  # [gamma, hs]
        draft_tokens = _lmhead_argmax(draft_h, lm_head, gemv_ops)

        torch.cuda.synchronize()
        t_draft_total += time.perf_counter() - t0

        # ================================================================
        # VERIFY: batch forward of [cur_token, draft_0, …, draft_{γ-1}]
        # ================================================================
        verify_ids = [cur_token] + draft_tokens[:actual_gamma]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            pred_tokens, verify_hidden = batch_verify_forward(
                language_model, lm_head, cache,
                verify_ids, cache_pos, gemv_ops,
            )

        torch.cuda.synchronize()
        t_verify_total += time.perf_counter() - t0

        # ================================================================
        # ACCEPT / REJECT
        # ================================================================
        accept_len = 0
        for i in range(actual_gamma):
            if pred_tokens[i] == draft_tokens[i]:
                accept_len += 1
            else:
                break

        drafted_total += actual_gamma
        accepted_total += accept_len
        cycles += 1

        # Bonus token = model's prediction at the first rejected (or last) position
        bonus_idx = min(accept_len, len(pred_tokens) - 1)
        bonus_token = pred_tokens[bonus_idx]

        if log_on and cycles <= 5:
            print(f"  [eagle #{cycles}] pos={cache_pos} γ={actual_gamma} "
                  f"accept={accept_len} draft={draft_tokens[:4]} "
                  f"pred={pred_tokens[:4]}", flush=True)

        # Write accepted draft tokens
        for i in range(min(accept_len, remaining)):
            token_buf[num_generated] = draft_tokens[i]
            num_generated += 1

        # Write bonus token
        if num_generated < max_new_tokens:
            token_buf[num_generated] = bonus_token
            num_generated += 1

        # ================================================================
        # UPDATE STATE
        # ================================================================
        # Truncate cache to accepted count (overwrite rejected entries later)
        new_cache_pos = cache_pos + accept_len + 1
        seqlens.fill_(new_cache_pos)

        # Update hidden state from verify result (avoid FP16/INT4 mismatch)
        cur_hidden = verify_hidden[bonus_idx].float().detach()
        cur_token = bonus_token
        cache_pos = new_cache_pos

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
        avg_d = t_draft_total / cycles * 1000 if cycles else 0
        avg_v = t_verify_total / cycles * 1000 if cycles else 0
        print(
            f"[eagle] gen={num_generated} cycles={cycles} "
            f"accept={accepted_total}/{drafted_total} ({rate:.1f}%) "
            f"draft={avg_d:.2f}ms verify={avg_v:.2f}ms",
            flush=True,
        )

    return torch.cat([input_ids, token_buf[:num_generated].unsqueeze(0)], dim=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lmhead_argmax(hiddens, lm_head, gemv_ops):
    """Convert [M, hs] hiddens → list[M] token ids via lm_head argmax."""
    M = hiddens.shape[0]
    if gemv_ops is not None and hasattr(lm_head, 'weight_int4'):
        from .batch_verify import _num_groups
        ng = _num_groups(lm_head)
        logits = gemv_ops.gemv_lmhead_int4_batch(
            lm_head.weight_int4, hiddens, lm_head.weight_scale_int4, ng, M)
    elif hasattr(lm_head, 'weight'):
        import torch.nn.functional as F
        logits = F.linear(hiddens, lm_head.weight)
    else:
        logits = lm_head(hiddens.unsqueeze(0)).squeeze(0)
    return logits.argmax(dim=-1).cpu().tolist()
