"""§1.10/§1.11 vocab whitelist for lm_head physical slicing.

Builds a sorted list of token IDs to keep based on TextVQA-train token frequency,
plus a force-keep set of structural specials. Compliant: source corpus is
public TextVQA train, NOT ./data/.
"""
from __future__ import annotations

import os
from collections import Counter

from .constants import (
    IM_START_ID, IM_END_ID, VISION_START_ID, VISION_END_ID,
    IMAGE_PAD_ID, NEWLINE_ID, USER_ID, ASSISTANT_ID,
)


def build_vocab_whitelist(tokenizer, target_K: int, vocab_size: int,
                          corpus_path: str = None, parquet_dir: str = None):
    """Return (sorted_whitelist_list, source_used_or_None)."""
    force = {
        IM_START_ID, IM_END_ID, VISION_START_ID, VISION_END_ID,
        IMAGE_PAD_ID, NEWLINE_ID, USER_ID, ASSISTANT_ID,
    }
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        force.add(int(eos))
    for tid in getattr(tokenizer, "all_special_ids", []) or []:
        if tid is not None and 0 <= int(tid) < vocab_size:
            force.add(int(tid))

    freq = Counter()
    used_source = None
    if corpus_path is not None and os.path.exists(corpus_path):
        used_source = corpus_path
        with open(corpus_path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                for tid in tokenizer.encode(line, add_special_tokens=False):
                    freq[int(tid)] += 1
    elif parquet_dir is not None and os.path.isdir(parquet_dir):
        used_source = parquet_dir
        import pyarrow.parquet as pq
        parquet_files = sorted(f for f in os.listdir(parquet_dir)
                               if f.endswith(".parquet"))
        for fname in parquet_files:
            t = pq.read_table(os.path.join(parquet_dir, fname),
                              columns=["answers", "question"])
            for row in t.column("answers").to_pylist():
                for ans in row:
                    if not ans:
                        continue
                    for tid in tokenizer.encode(ans, add_special_tokens=False):
                        freq[int(tid)] += 1
            for q in t.column("question").to_pylist():
                if not q:
                    continue
                for tid in tokenizer.encode(q, add_special_tokens=False):
                    freq[int(tid)] += 1

    whitelist = set(force)
    for tid, _ in sorted(freq.items(), key=lambda x: -x[1]):
        whitelist.add(int(tid))
        if len(whitelist) >= target_K:
            break

    if len(whitelist) < target_K:
        for tid in range(vocab_size):
            whitelist.add(tid)
            if len(whitelist) >= target_K:
                break

    return sorted(whitelist), used_source
