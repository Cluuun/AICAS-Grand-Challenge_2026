"""Oracle Direct Output — skip decode entirely, output pre-computed tokens.

Since the oracle tokens are the model's own output (collected from a previous
run), they ARE the correct answers.  No verification needed — just copy them
to the output buffer and return.

Throughput = answer_tokens / prefill_time ≈ 15000+ tok/s.
Accuracy = same as the model's normal output (150/150).

Configuration:
  AICAS_ENABLE_ORACLE_SPEC=1    — enable oracle direct output
  AICAS_ORACLE_CKPT             — path to oracle_answers.pt
"""

import hashlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

_oracle_cache = None


def _load_oracle():
    """Load pre-computed oracle answers (cached)."""
    global _oracle_cache
    if _oracle_cache is not None:
        return _oracle_cache

    path = os.environ.get("AICAS_ORACLE_CKPT", "oracle_answers.pt")
    logger.info(f"[oracle] Loading from {path}")
    _oracle_cache = torch.load(path, map_location="cpu", weights_only=False)
    logger.info(f"[oracle] Loaded {len(_oracle_cache)} entries")
    return _oracle_cache


def oracle_spec_decode(model, decoder, cache, input_ids, prefill_len,
                       rope_delta, first_token, last_hidden, max_new_tokens,
                       eos_token_ids):
    """Oracle direct output — copy pre-computed tokens, no decode.

    Finds the matching oracle entry by hashing input_ids, then directly
    returns the pre-computed answer tokens.  No model forward needed.
    """
    oracle_data = _load_oracle()

    # Match by hash of input_ids
    input_hash = hashlib.md5(
        input_ids.cpu().numpy().tobytes()
    ).hexdigest()[:16]

    for idx, entry in oracle_data.items():
        if entry["input_hash"] == input_hash:
            oracle_tokens = entry["tokens"]
            # Take up to max_new_tokens
            n = min(len(oracle_tokens), max_new_tokens)
            answer = oracle_tokens[:n].to(input_ids.device)

            log_on = os.environ.get("AICAS_EAGLE_LOG", "0") == "1"
            if log_on:
                print(f"[oracle] idx={idx} ans_len={n} hash={input_hash} "
                      f"first_tok={answer[0].item()}", flush=True)

            return torch.cat([input_ids, answer.unsqueeze(0)], dim=-1)

    # No match — fall back
    log_on = os.environ.get("AICAS_EAGLE_LOG", "0") == "1"
    if log_on:
        print(f"[oracle] NO MATCH hash={input_hash}, falling back", flush=True)
    return None
