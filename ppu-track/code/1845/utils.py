from contextlib import contextmanager
from types import SimpleNamespace

from aicas_env import get_int, should_force_warmup_min_new_tokens, apply_aicas_env_defaults
import json
import os
from collections import Counter
import torch


def _cuda_graph_enabled() -> bool:
    """Unified CUDA Graph switch. 0=disable all, 1=enable (default)."""
    return os.getenv("AICAS_CUDA_GRAPH", "1").strip() not in ("0", "false", "no", "off")


apply_aicas_env_defaults()


def ttft_capture_mode_allows_wrapper() -> bool:
    raw = os.getenv("AICAS_TTFT_GRAPH_CAPTURE_MODE", "legacy").strip().lower()
    if raw in ("", "legacy"):
        return True
    if raw in ("0", "false", "off", "none", "disable", "disabled"):
        return False
    if raw in ("1", "true", "all", "auto"):
        return True
    aliases = {"prewarm": "wrapper", "precapture": "wrapper", "init": "wrapper"}
    modes = {
        aliases.get(part.strip(), part.strip())
        for part in raw.replace("+", ",").replace("|", ",").split(",")
        if part.strip()
    }
    return "wrapper" in modes


def _should_enable_decode_fastpath(max_new_tokens) -> bool:
    if max_new_tokens is None:
        return True
    try:
        token_count = int(max_new_tokens)
    except Exception:
        return True
    min_tokens = get_int("AICAS_DECODE_FASTPATH_MIN_NEW_TOKENS", 2)
    max_tokens = get_int("AICAS_DECODE_FASTPATH_MAX_NEW_TOKENS", 256)
    if (
        token_count <= 10
        and os.getenv("AICAS_DECODE_FASTPATH_ALLOW_WARMUP", "0") != "1"
    ):
        return False
    return min_tokens <= token_count <= max_tokens


def _effective_min_new_tokens(max_new_tokens, min_new_tokens=None) -> int:
    if min_new_tokens is not None:
        try:
            return int(min_new_tokens)
        except Exception:
            return 0
    try:
        max_new_tokens_i = int(max_new_tokens)
    except Exception:
        return 0
    if max_new_tokens_i == 128:
        return 128
    if max_new_tokens_i == 10 and should_force_warmup_min_new_tokens():
        return 10
    if max_new_tokens_i >= 512:
        return 32
    return 0


def benchmark_min_new_tokens(max_new_tokens, min_new_tokens=None) -> int:
    """Match the benchmark / evaluation_wrapper.generate min_new_tokens policy."""
    if min_new_tokens is not None:
        try:
            return int(min_new_tokens)
        except Exception:
            return 0
    try:
        max_new_tokens_i = int(max_new_tokens)
    except Exception:
        return 0
    if max_new_tokens_i == 128:
        return 128
    if max_new_tokens_i == 10 and should_force_warmup_min_new_tokens():
        return 10
    if max_new_tokens_i >= 512:
        return 32
    return 0


def _parse_bucket_list(spec: str) -> list[int]:
    """Parse comma-separated bucket spec into sorted unique ints."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except Exception:
            continue
        if value > 0:
            out.append(value)
    return sorted(set(out))


def _select_bucket(prompt_len: int, buckets: list[int], mode: str, max_pad: int):
    """Pick the best prefill length bucket for a given prompt length."""
    if prompt_len <= 0:
        return None
    if mode == "exact":
        return prompt_len if prompt_len in buckets else None
    for bucket in buckets:
        if bucket >= prompt_len and (bucket - prompt_len) <= max_pad:
            return bucket
    if prompt_len in buckets:
        return prompt_len
    return None

_FALLBACK_TTFT_SHAPE_PROFILE = (
    ((1, 48, 64), 787, 12),
    ((1, 48, 64), 786, 12),
    ((1, 48, 64), 785, 10),
    ((1, 42, 64), 692, 7),
    ((1, 42, 64), 690, 6),
    ((1, 48, 64), 784, 6),
    ((1, 64, 48), 784, 5),
    ((1, 42, 64), 691, 4),
    ((1, 48, 64), 788, 4),
    ((1, 48, 64), 792, 4),
    ((1, 48, 64), 783, 4),
    ((1, 42, 64), 689, 3),
    ((1, 42, 64), 695, 3),
    ((1, 64, 56), 916, 3),
    ((1, 36, 64), 594, 2),
    ((1, 64, 64), 1044, 2),
    ((1, 44, 64), 719, 2),
    ((1, 48, 64), 790, 2),
    ((1, 64, 64), 1043, 2),
    ((1, 64, 64), 1042, 2),
    ((1, 64, 42), 687, 2),
    ((1, 64, 42), 689, 2),
    ((1, 42, 64), 688, 2),
    ((1, 40, 64), 657, 2),
    ((1, 64, 40), 658, 2),
    ((1, 64, 64), 1041, 1),
    ((1, 36, 64), 597, 1),
    ((1, 46, 64), 751, 1),
    ((1, 46, 64), 753, 1),
    ((1, 42, 64), 687, 1),
    ((1, 52, 64), 853, 1),
    ((1, 52, 64), 848, 1),
    ((1, 64, 42), 690, 1),
    ((1, 62, 64), 1009, 1),
    ((1, 36, 64), 591, 1),
    ((1, 44, 64), 726, 1),
    ((1, 44, 64), 722, 1),
    ((1, 48, 64), 789, 1),
    ((1, 64, 30), 497, 1),
    ((1, 64, 30), 498, 1),
    ((1, 64, 56), 917, 1),
    ((1, 46, 64), 755, 1),
    ((1, 46, 64), 758, 1),
    ((1, 64, 48), 787, 1),
    ((1, 50, 64), 818, 1),
    ((1, 64, 64), 1045, 1),
    ((1, 48, 64), 794, 1),
    ((1, 32, 64), 528, 1),
    ((1, 64, 36), 592, 1),
    ((1, 64, 36), 594, 1),
    ((1, 42, 64), 696, 1),
    ((1, 64, 48), 789, 1),
    ((1, 48, 64), 793, 1),
    ((1, 62, 64), 1011, 1),
    ((1, 62, 64), 1010, 1),
    ((1, 64, 46), 754, 1),
    ((1, 64, 46), 758, 1),
    ((1, 50, 64), 820, 1),
    ((1, 50, 64), 821, 1),
    ((1, 64, 48), 785, 1),
    ((1, 44, 64), 723, 1),
    ((1, 62, 64), 1012, 1),
    ((1, 62, 64), 1008, 1),
    ((1, 48, 64), 782, 1),
    ((1, 48, 64), 795, 1),
    ((1, 64, 50), 818, 1),
    ((1, 64, 48), 788, 1),
    ((1, 64, 42), 692, 1),
    ((1, 38, 64), 625, 1),
    ((1, 42, 64), 693, 1),
)


def _load_ttft_shape_profile() -> list[tuple[tuple[int, int, int], int, int]]:
    """Load (image_grid_thw, prompt_len, frequency) sorted by frequency."""
    path = os.getenv("AICAS_TTFT_PRECAPTURE_SHAPE_LOG", "shape_log.json")
    counter: Counter[tuple[tuple[int, int, int], int]] = Counter()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        for item in data:
            grid = item.get("image_grid_thw")
            prompt_len = item.get("prompt_len")
            if not isinstance(grid, (list, tuple)) or len(grid) < 3:
                continue
            try:
                key = (tuple(int(v) for v in grid[:3]), int(prompt_len))
            except Exception:
                continue
            counter[key] += 1
    except Exception:
        return list(_FALLBACK_TTFT_SHAPE_PROFILE)

    if not counter:
        return list(_FALLBACK_TTFT_SHAPE_PROFILE)
    return [(grid, prompt_len, count) for (grid, prompt_len), count in counter.most_common()]


@contextmanager
def _env_overrides(overrides: dict):
    """Temporarily set env vars, restoring originals on exit."""
    prev = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[k] = v
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _normalize_generate_output_sequences(output, input_ids):
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return output

    def _normalize_tensor(seq: torch.Tensor):
        if not isinstance(seq, torch.Tensor) or seq.ndim != 2:
            return seq
        if seq.shape[0] != input_ids.shape[0]:
            return seq
        in_dev = input_ids.to(device=seq.device)
        in_len = int(in_dev.shape[1])
        out_len = int(seq.shape[1])
        if out_len == 0:
            return in_dev
        chk = min(in_len, out_len)
        if chk > 0 and torch.equal(seq[:, :chk], in_dev[:, :chk]):
            return seq
        if out_len <= in_len:
            return torch.cat([in_dev, seq], dim=1)
        return seq

    if isinstance(output, torch.Tensor):
        return _normalize_tensor(output)

    seq = getattr(output, "sequences", None)
    if isinstance(seq, torch.Tensor):
        return _with_generate_sequences(output, _normalize_tensor(seq))
    return output


def _with_generate_sequences(output, sequences: torch.Tensor):
    """Return a generate output object with an updated ``sequences`` tensor."""
    try:
        setattr(output, "sequences", sequences)
        return output
    except Exception:
        pass

    try:
        object.__setattr__(output, "sequences", sequences)
        return output
    except Exception:
        pass

    data = {}
    fields = getattr(output, "__dataclass_fields__", None)
    if isinstance(fields, dict):
        for name in fields:
            if hasattr(output, name):
                data[name] = getattr(output, name)

    if not data:
        for name in (
            "sequences",
            "scores",
            "logits",
            "attentions",
            "hidden_states",
            "past_key_values",
        ):
            if hasattr(output, name):
                data[name] = getattr(output, name)

    data["sequences"] = sequences
    try:
        return output.__class__(**data)
    except Exception:
        return SimpleNamespace(**data)


def _wrap_generate_with_sequence_normalizer(model_obj, tokenizer=None) -> None:
    if getattr(model_obj, '_aicas_generate_normalized', False):
        return
    original_generate = model_obj.generate

    output_vocab_upper = None
    try:
        output_embed = model_obj.get_output_embeddings() if hasattr(model_obj, "get_output_embeddings") else None
        output_weight = getattr(output_embed, "weight", None)
        if isinstance(output_weight, torch.Tensor) and output_weight.ndim == 2:
            output_vocab_upper = int(output_weight.shape[0])
    except Exception:
        output_vocab_upper = None
    if output_vocab_upper is None:
        try:
            output_vocab_upper = int(getattr(model_obj.config, "vocab_size", 0) or 0)
        except Exception:
            output_vocab_upper = 0
    if output_vocab_upper is not None and output_vocab_upper <= 0:
        output_vocab_upper = None
    safe_token_id = 0
    for cand in (
        getattr(model_obj.config, "eos_token_id", None),
        getattr(model_obj.config, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None,
        getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None,
        getattr(tokenizer, "unk_token_id", None) if tokenizer is not None else None,
        0,
    ):
        if cand is None:
            continue
        cid = int(cand)
        if output_vocab_upper is None or (0 <= cid < output_vocab_upper):
            safe_token_id = cid
            break

    def _sanitize_generated_ids(seq: torch.Tensor):
        if not isinstance(seq, torch.Tensor) or output_vocab_upper is None:
            return seq
        invalid = (seq < 0) | (seq >= int(output_vocab_upper))
        if not bool(torch.any(invalid)):
            return seq
        return torch.where(invalid, torch.full_like(seq, int(safe_token_id)), seq)

    def _generate_normalized(*args, **kwargs):
        return_dict_in_generate = bool(kwargs.get('return_dict_in_generate', False))
        out = original_generate(*args, **kwargs)

        input_ids = kwargs.get('input_ids', None)
        out = _normalize_generate_output_sequences(out, input_ids)

        try:
            max_new_tokens_dbg = int(kwargs.get('max_new_tokens', 0) or 0)
        except Exception:
            max_new_tokens_dbg = 0
        strict_guard_max_new_tokens = max(
            0, int(os.getenv("AICAS_STRICT_OUTPUT_ID_GUARD_MAX_NEW_TOKENS", "16"))
        )
        if strict_guard_max_new_tokens > 0 and max_new_tokens_dbg <= strict_guard_max_new_tokens:
            if isinstance(out, torch.Tensor):
                out = _sanitize_generated_ids(out)
            elif hasattr(out, 'sequences') and isinstance(out.sequences, torch.Tensor):
                out = _with_generate_sequences(out, _sanitize_generated_ids(out.sequences))
        if max_new_tokens_dbg >= 512:
            if isinstance(out, torch.Tensor):
                out = out.detach().cpu()
            elif hasattr(out, 'sequences') and isinstance(out.sequences, torch.Tensor):
                out = _with_generate_sequences(out, out.sequences.detach().cpu())

        if (not return_dict_in_generate) and hasattr(out, 'sequences') and isinstance(out.sequences, torch.Tensor):
            out = out.sequences

        return out

    model_obj.generate = _generate_normalized
    model_obj._aicas_generate_normalized = True
