"""N-gram prompt-lookup speculative decoding with streaming verify.

Uses O(1) n-gram lookup to draft tokens, then verifies one-by-one using
the existing decode_step (same INT8 path as normal decode, no FP16 mismatch).

When draft tokens match, we get gamma+1 tokens for the cost of gamma+1 steps.
When they don't match, we get 1 bonus token for 1 step (same as normal).
"""
import os
import torch


class _NgramIndex:
    """O(1) n-gram lookup using a dict {ngram_tuple: latest_match_end_pos}."""

    def __init__(self, ngram_n):
        self.n = ngram_n
        self.idx = {}
        self.history = []
        self._indexed_up_to = 0

    def extend(self, tokens):
        n = self.n
        self.history.extend(tokens)
        L = len(self.history)
        indexable_end = L - n
        while self._indexed_up_to < indexable_end:
            e = self._indexed_up_to
            key = tuple(self.history[e:e + n])
            self.idx[key] = e + n
            self._indexed_up_to += 1

    def lookup(self, max_draft):
        n = self.n
        L = len(self.history)
        if L < n + 1:
            return []
        key = tuple(self.history[-n:])
        pos = self.idx.get(key, -1)
        if pos < 0 or pos >= L:
            return []
        draft = self.history[pos:pos + max_draft]
        return list(draft) if len(draft) > 0 else []


def ngram_spec_generate(model, decoder, cache, input_ids,
                        prefill_len, rope_delta, first_token,
                        max_new_tokens, eos_token_ids):
    """Prompt-lookup speculative decoding with streaming verify."""
    device = input_ids.device
    gamma = max(1, int(os.environ.get("AICAS_NGRAM_GAMMA", "4")))
    ngram_n = max(1, int(os.environ.get("AICAS_NGRAM_N", "2")))
    sync_batch = max(1, int(os.environ.get("AICAS_NGRAM_SYNC_BATCH", "8")))
    log_enabled = os.environ.get("AICAS_NGRAM_LOG", "0") == "1"

    eos_set = set(int(x) for x in (eos_token_ids or ()))
    token_buf = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    token_buf[0] = first_token[0, 0]
    num_generated = 1

    idx = _NgramIndex(ngram_n)
    initial_history = input_ids[0].detach().cpu().tolist()
    initial_history.append(int(first_token[0, 0].item()))
    idx.extend(initial_history)

    last_emit_token = first_token
    base_step = 0

    accepted_total = 0
    drafted_total = 0
    spec_cycles = 0
    normal_cycles = 0

    def _is_eos(tok_id):
        return eos_set and tok_id in eos_set

    while num_generated < max_new_tokens:
        remaining = max_new_tokens - num_generated
        if remaining <= 0:
            break

        # Try n-gram lookup
        draft = idx.lookup(gamma) if remaining >= 2 else []
        has_draft = len(draft) >= 2

        if has_draft:
            spec_cycles += 1
            drafted_total += len(draft)

            # Decode to verify first draft token
            next_tok = decoder.decode_step(last_emit_token, base_step)
            base_step += 1
            pred_id = next_tok[0, 0].item()

            if pred_id == draft[0]:
                # First draft matches
                accept_count = 1
                token_buf[num_generated] = draft[0]
                num_generated += 1
                idx.extend([draft[0]])
                last_emit_token = next_tok
                if _is_eos(draft[0]):
                    break

                # Verify remaining draft tokens
                for di in range(1, min(len(draft), remaining - 1)):
                    next_tok = decoder.decode_step(last_emit_token, base_step)
                    base_step += 1
                    pred_id = next_tok[0, 0].item()

                    if pred_id == draft[di]:
                        accept_count += 1
                        token_buf[num_generated] = draft[di]
                        num_generated += 1
                        idx.extend([draft[di]])
                        last_emit_token = next_tok
                        if _is_eos(draft[di]):
                            break
                    else:
                        # Mismatch: use prediction as bonus
                        if num_generated < max_new_tokens:
                            token_buf[num_generated] = pred_id
                            num_generated += 1
                            idx.extend([pred_id])
                            last_emit_token = next_tok
                            if _is_eos(pred_id):
                                break
                        break
                else:
                    # All draft accepted, add bonus token
                    if num_generated < max_new_tokens:
                        token_buf[num_generated] = pred_id
                        num_generated += 1
                        idx.extend([pred_id])
                        last_emit_token = next_tok
                        if _is_eos(pred_id):
                            break

                accepted_total += accept_count
            else:
                # First draft mismatch: use prediction
                token_buf[num_generated] = pred_id
                num_generated += 1
                idx.extend([pred_id])
                last_emit_token = next_tok
                if _is_eos(pred_id):
                    break
        else:
            # --- Normal batch decode (optimized: non-blocking copies) ---
            normal_cycles += 1
            batch = min(sync_batch, remaining)
            cur_token = last_emit_token
            for b in range(batch):
                if num_generated >= max_new_tokens:
                    break
                next_tok = decoder.decode_step(cur_token, base_step)
                base_step += 1
                # Non-blocking copy (avoids per-token sync)
                token_buf[num_generated].copy_(next_tok[0, 0], non_blocking=True)
                num_generated += 1
                cur_token = next_tok

            last_emit_token = cur_token

            # Single sync to read last token for EOS check + batch index update
            last_id = cur_token[0, 0].item()
            idx.extend(token_buf[num_generated - batch:num_generated].cpu().tolist())

            if _is_eos(last_id):
                break

    if log_enabled:
        rate = accepted_total / drafted_total * 100 if drafted_total > 0 else 0
        print(
            f"[ngram] gen={num_generated} spec={spec_cycles} normal={normal_cycles} "
            f"accepted={accepted_total}/{drafted_total} ({rate:.1f}%)",
            flush=True,
        )

    return torch.cat([input_ids, token_buf[:num_generated].unsqueeze(0)], dim=-1)
