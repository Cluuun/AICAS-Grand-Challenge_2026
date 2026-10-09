"""EAGLE speculative decoding with one-by-one CUDA graph verification.

Uses the EAGLE draft model to predict future hidden states, then verifies
each draft token using the existing CUDA graph decode_step.

Key design: after each cycle, use the draft model's PREDICTED hidden state
(at the accepted position) as input for the next draft cycle. This avoids
the FP16/INT4 hidden state mismatch between prefill and decode paths.

Lossless: output distribution identical to autoregressive decoding.
"""
import os
import torch

_draft_model_cache = {}


def _get_draft_model(device):
    if device not in _draft_model_cache:
        ckpt = os.environ.get("AICAS_EAGLE_DRAFT_CKPT", "trash/draft_hidden_12x48_ce.pt")
        from eagle_spec.draft_model import load_draft_model
        _draft_model_cache[device] = load_draft_model(ckpt, device=device)
    return _draft_model_cache[device]


def _batched_lmhead(hidden_flat, lm_head, gemv_ops=None):
    """Batched lm_head: [M, hidden] -> [M] argmax token ids."""
    import torch.nn.functional as F
    M = hidden_flat.shape[0]
    if gemv_ops is not None and hasattr(lm_head, 'weight_int8'):
        tokens = []
        for i in range(M):
            fh = hidden_flat[i:i+1]
            logits_flat = gemv_ops.gemv_lmhead_int8(
                lm_head.weight_int8, fh.reshape(-1), lm_head.weight_scale)
            tokens.append(logits_flat.argmax().item())
        return tokens
    elif hasattr(lm_head, 'weight'):
        logits = F.linear(hidden_flat.half(), lm_head.weight)
        return logits.argmax(dim=-1).tolist()
    else:
        raise RuntimeError("No usable lm_head weights")


def spec_decode_generate(model, decoder, cache, input_ids, prefill_len,
                         rope_delta, first_token, last_hidden, max_new_tokens,
                         eos_token_ids):
    """EAGLE speculative decoding with one-by-one CUDA graph verify.

    Uses the draft model's predicted hidden states (not the decode step's
    hidden states) to avoid FP16/INT4 mismatch. After each cycle, the
    predicted hidden state at the accepted position feeds the next draft.

    Speedup comes from the bonus token when all gamma draft tokens are
    accepted: (gamma+1) tokens in gamma decode_steps + 1 bonus step.
    """
    device = input_ids.device
    gamma = int(os.environ.get("AICAS_SPEC_GAMMA", "4"))
    log_enabled = os.environ.get("AICAS_SPEC_LOG", "0") == "1"

    lm_head = model.lm_head
    gemv_ops = decoder._gemv_ops
    draft_model = _get_draft_model(device)
    max_draft = draft_model.horizon - 1
    gamma = min(gamma, max_draft)

    eos_set = set(int(x) for x in (eos_token_ids or ()))
    token_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    token_buf[0] = first_token[0, 0]
    num_generated = 1

    cur_token = first_token
    cur_hidden = last_hidden[0].float()  # [hidden_size], from prefill
    step = 0

    accepted_total = 0
    drafted_total = 0
    spec_cycles = 0

    while num_generated < max_new_tokens:
        remaining = max_new_tokens - num_generated
        if remaining <= 1:
            next_tok = decoder.decode_step(cur_token, step)
            step += 1
            token_buf[num_generated] = next_tok[0, 0]
            num_generated += 1
            break

        actual_gamma = min(gamma, remaining - 1)
        if actual_gamma <= 0:
            break

        # --- DRAFT: predict future tokens from draft model hidden state ---
        with torch.no_grad():
            future_hiddens = draft_model(cur_hidden.unsqueeze(0))  # [1, horizon, hidden]

        # Convert predicted hidden states to draft tokens
        draft_tokens = _batched_lmhead(
            future_hiddens[0, 1:actual_gamma+1].half(), lm_head, gemv_ops)

        spec_cycles += 1
        drafted_total += actual_gamma

        # --- VERIFY: one-by-one using CUDA graph decode_step ---
        accept_len = 0
        for i in range(actual_gamma):
            next_tok = decoder.decode_step(cur_token, step)
            step += 1
            pred_id = next_tok[0, 0].item()

            if pred_id == draft_tokens[i]:
                token_buf[num_generated] = draft_tokens[i]
                num_generated += 1
                cur_token = next_tok
                accept_len += 1
                if eos_set and draft_tokens[i] in eos_set:
                    break
            else:
                # Reject: use model prediction as bonus
                token_buf[num_generated] = pred_id
                num_generated += 1
                cur_token = next_tok
                break
        else:
            # All draft accepted: bonus decode_step
            if num_generated < max_new_tokens:
                next_tok = decoder.decode_step(cur_token, step)
                step += 1
                bonus = next_tok[0, 0].item()
                token_buf[num_generated] = bonus
                num_generated += 1
                cur_token = next_tok
                if eos_set and bonus in eos_set:
                    break

        accepted_total += accept_len

        # Use draft model's predicted hidden state at the accepted position
        # (avoids FP16/INT4 mismatch from decode step hidden state)
        if accept_len > 0:
            # Position accept_len in future_hiddens corresponds to the last accepted
            # future_hiddens[0, 0] = current position reconstruction
            # future_hiddens[0, k] = predicted hidden at position current+k
            pred_pos = min(accept_len, future_hiddens.shape[1] - 1)
            cur_hidden = future_hiddens[0, pred_pos].detach()

        # EOS check
        if eos_set:
            last_tok = token_buf[num_generated - 1].item()
            if last_tok in eos_set:
                break

    if log_enabled:
        rate = accepted_total / drafted_total * 100 if drafted_total > 0 else 0
        print(
            f"[spec_decode] gen={num_generated} cycles={spec_cycles} "
            f"accepted={accepted_total}/{drafted_total} ({rate:.1f}%)",
            flush=True,
        )

    return torch.cat([input_ids, token_buf[:num_generated].unsqueeze(0)], dim=-1)
