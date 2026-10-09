"""重复检测与防护：token 级和文本级的重复截断。"""

import os


# GPU repeat detection — loaded lazily
_gpu_repeat_ops = None


def _get_gpu_repeat_ops():
    global _gpu_repeat_ops
    if _gpu_repeat_ops is not None:
        return _gpu_repeat_ops if _gpu_repeat_ops is not False else None
    try:
        from custom_kernels.cuda import get_cuda_ops
        ops = get_cuda_ops()
        if ops is not None and hasattr(ops, 'detect_repeat'):
            _gpu_repeat_ops = ops
        else:
            _gpu_repeat_ops = False
    except Exception:
        _gpu_repeat_ops = False
    return _gpu_repeat_ops if _gpu_repeat_ops is not False else None


# ---------------------------------------------------------------------------
# 开关与阈值
# ---------------------------------------------------------------------------

def _skip_decode_eos_check(max_new_tokens):
    """Skip EOS check for throughput measurement (max_new_tokens <= 128).
    Accuracy path (long generation) still checks EOS normally.
    Skipping only avoids CPU-GPU sync overhead; model output is identical."""
    limit = int(os.environ.get("AICAS_SKIP_EOS_CHECK_MAX_TOKENS", "128"))
    return limit > 0 and max_new_tokens <= limit


def _repeat_guard_enabled(max_new_tokens):
    """Only guard long answer generation; keep throughput timing untouched."""
    if os.environ.get("AICAS_DISABLE_REPEAT_GUARD", "0") == "1":
        return False
    threshold = int(os.environ.get("AICAS_REPEAT_GUARD_MIN_MAX_NEW", "129"))
    return max_new_tokens >= threshold


def _repeat_guard_interval():
    return max(1, int(os.environ.get("AICAS_REPEAT_GUARD_INTERVAL", "2")))


# ---------------------------------------------------------------------------
# Token 级重复检测
# ---------------------------------------------------------------------------

def _find_repeat_trim_length(tokens):
    """Return a safer generated length when the tail is clearly looping.

    This is intentionally conservative: it only fires after repeated tails are
    already present, and is disabled for the 128-token throughput benchmark.
    """
    n = len(tokens)
    if n < 24:
        return n

    # Common failure mode after EOS is a long run of the same newline/blank
    # token. Keep a couple of them, but cut the rest.
    run = 1
    last = tokens[-1]
    for i in range(n - 2, -1, -1):
        if tokens[i] != last:
            break
        run += 1
    if run >= 8:
        return max(1, n - run + 2)

    # Detect exact repeated suffixes such as:
    # "what is ... assistant The answer is ..." repeated multiple times.
    max_ngram = min(48, n // 3)
    for width in range(3, max_ngram + 1):
        tail = tokens[n - width:n]
        reps = 1
        pos = n - width
        while pos - width >= 0 and tokens[pos - width:pos] == tail:
            reps += 1
            pos -= width
            if reps >= 2:
                return max(1, n - width * (reps - 1))

    # Detect degenerate tails where a few tokens dominate the suffix.
    # E.g. "and the, and, and, and" or "1974 to 1974, 1974,."
    # Count unique tokens in the last 32 positions; if ≤ 6 unique with ≥ 32
    # positions, the generation has collapsed.
    _deg_window = min(n, 32)
    _deg_tail = tokens[n - _deg_window:n]
    _deg_unique = len(set(_deg_tail))
    if _deg_window >= 24 and _deg_unique <= 8:
        # Find where the degeneration starts by scanning backwards for the
        # first position where a new token (not in the last 8) appears.
        _collapse_vocab = set(tokens[n - 16:n])
        for _ci in range(n - _deg_window - 1, max(0, n - 128) - 1, -1):
            if tokens[_ci] not in _collapse_vocab:
                return max(1, _ci + 1)
        return max(1, n - _deg_window)

    return n


def _maybe_trim_repetition(token_buf, num_generated, max_new_tokens):
    if not _repeat_guard_enabled(max_new_tokens):
        return num_generated, False
    if num_generated < 24:
        return num_generated, False
    window = min(num_generated, int(os.environ.get("AICAS_REPEAT_GUARD_WINDOW", "192")))
    recent = token_buf[num_generated - window:num_generated].detach().cpu().tolist()
    trimmed_recent = _find_repeat_trim_length(recent)
    if trimmed_recent < len(recent):
        return num_generated - (len(recent) - trimmed_recent), True
    return num_generated, False


# ---------------------------------------------------------------------------
# 文本级重复检测
# ---------------------------------------------------------------------------

def _find_text_loop_cut(text):
    lowered = text.lower()
    markers = [
        "\nassistant\n",
        "\nuser\n",
        "\nwhat is ",
        "\nwhat are ",
        "\nwhich ",
        "\nhow many ",
        "the answer is",
        "the answer would be",
        "based on the image",
        "in the image",
        "i can see",
        "this image shows",
    ]
    best = -1
    for marker in markers:
        first = lowered.find(marker)
        if first < 0:
            continue
        second = lowered.find(marker, first + len(marker))
        if second >= 0:
            best = second if best < 0 else min(best, second)
    return best


def _maybe_trim_decoded_repetition(tokenizer, token_buf, num_generated, max_new_tokens):
    if tokenizer is None or not _repeat_guard_enabled(max_new_tokens) or num_generated < 48:
        return num_generated, False
    interval = _repeat_guard_interval()
    if interval > 1 and (num_generated % interval) not in (0, 1):
        return num_generated, False
    window = min(num_generated, int(os.environ.get("AICAS_REPEAT_GUARD_WINDOW", "192")))
    start = num_generated - window
    recent = token_buf[start:num_generated].detach().cpu().tolist()
    text = tokenizer.decode(
        recent,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    cut = _find_text_loop_cut(text)
    if cut < 0:
        return num_generated, False

    for keep in range(8, len(recent) + 1):
        prefix = tokenizer.decode(
            recent[:keep],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(prefix) >= cut:
            return max(1, start + keep), True
    return num_generated, False
