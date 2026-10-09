"""
Phase 4: Exact greedy verifier with detailed timing instrumentation.
"""
import time
import inspect
import copy
import os
from contextlib import nullcontext
import torch

from spec_decode.multitoken_attention import spec_multitoken_attention
from spec_decode.qk_rotary import spec_qk_rotary

try:
    from my_kernel.lm_head_argmax_triton import (
        can_use_lm_head_argmax_triton_multi,
        lm_head_argmax_triton_multi,
    )
except Exception:
    can_use_lm_head_argmax_triton_multi = None
    lm_head_argmax_triton_multi = None

try:
    from decode_graph_runtime import (
        _build_decode_position_ids,
        _copy_cache_into_static,
        _get_layer_kv,
        _get_cache_seq_len,
    )
except Exception:
    _build_decode_position_ids = None
    _copy_cache_into_static = None
    _get_layer_kv = None
    _get_cache_seq_len = None


_FORWARD_CONTROL_KEYS = {
    "use_cache",
    "output_hidden_states",
    "return_dict",
    "past_key_values",
    "cache_position",
}
_TRACE_ONCE: set[str] = set()


def _spec_kernel_trace_once(key: str, message: str) -> None:
    if os.getenv("AICAS_SPEC_KERNEL_TRACE", "0") != "1" or key in _TRACE_ONCE:
        return
    _TRACE_ONCE.add(key)
    print(message)


def _filter_forward_inputs(model, inputs: dict) -> dict:
    """Keep only model.forward kwargs; generate-only kwargs can trigger repr recursion."""
    try:
        allowed = set(inspect.signature(model.forward).parameters)
    except Exception:
        allowed = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
            "position_ids",
        }
    return {
        k: v for k, v in inputs.items()
        if k in allowed and k not in _FORWARD_CONTROL_KEYS
    }


def _last_hidden_module(model):
    core = getattr(model, "model", None)
    language_model = getattr(core, "language_model", None) if core is not None else None
    norm = getattr(language_model, "norm", None) if language_model is not None else None
    if norm is None:
        raise RuntimeError("Cannot find language_model.norm for last-hidden capture")
    return norm


def _forward_with_last_hidden(model, **kwargs):
    box = {}

    def _capture(_module, _inputs, output):
        box["hidden"] = output

    handle = _last_hidden_module(model).register_forward_hook(_capture)
    try:
        out = model(**kwargs, return_dict=True)
    finally:
        handle.remove()
    hidden = box.get("hidden")
    if hidden is None:
        raise RuntimeError("last-hidden hook did not fire")
    return out, hidden


def _runtime_step_with_hidden(model, kwargs: dict, token_state: dict):
    from decode_graph_runtime import _decode_forward_next_token

    box = {}

    def _capture(_module, _inputs, output):
        box["hidden"] = output

    handle = _last_hidden_module(model).register_forward_hook(_capture)
    try:
        next_token, outputs = _decode_forward_next_token(model, kwargs, token_state)
    finally:
        handle.remove()
    hidden = box.get("hidden")
    if isinstance(hidden, torch.Tensor):
        hidden = hidden[:, -1:, :]
    return next_token, outputs, hidden


def _cache_seq_len(cache) -> int:
    if _get_cache_seq_len is not None:
        try:
            return int(_get_cache_seq_len(cache))
        except Exception:
            pass
    if cache is None:
        return 0
    try:
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
    except Exception:
        pass
    return 0


def _decode_position_kwargs(model, past_kv, input_ids: torch.Tensor) -> dict:
    start = _cache_seq_len(past_kv)
    batch_size, seq_len = input_ids.shape
    cache_position = torch.arange(
        start,
        start + seq_len,
        device=input_ids.device,
        dtype=torch.long,
    )
    out = {"cache_position": cache_position}
    if _build_decode_position_ids is not None:
        try:
            pos = _build_decode_position_ids(model, cache_position[:1], batch_size, seq_len)
            if isinstance(pos, torch.Tensor):
                out["position_ids"] = pos
        except Exception:
            pass
    return out


def _prepare_prefill_forward_inputs(model, inputs: dict) -> dict:
    """Prepare prefill kwargs with the same position setup used by generate()."""
    forward_inputs = _filter_forward_inputs(model, inputs)
    input_ids = forward_inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return forward_inputs

    if "cache_position" not in forward_inputs:
        forward_inputs["cache_position"] = torch.arange(
            int(input_ids.shape[1]), device=input_ids.device, dtype=torch.long
        )

    if "position_ids" not in forward_inputs:
        prepare_pos = getattr(model, "_prepare_position_ids_for_generation", None)
        if callable(prepare_pos):
            try:
                position_ids = prepare_pos(input_ids, dict(forward_inputs))
                if isinstance(position_ids, torch.Tensor):
                    forward_inputs["position_ids"] = position_ids
            except Exception:
                pass

    return forward_inputs


def _eos_id_set(eos_id) -> set[int]:
    if eos_id is None:
        return set()
    if isinstance(eos_id, torch.Tensor):
        return {int(x) for x in eos_id.detach().reshape(-1).cpu().tolist()}
    if isinstance(eos_id, (list, tuple, set)):
        out = set()
        for item in eos_id:
            try:
                out.add(int(item))
            except Exception:
                pass
        return out
    try:
        return {int(eos_id)}
    except Exception:
        return set()


def _extract_past_from_core_outputs(outputs):
    if hasattr(outputs, "past_key_values"):
        return outputs.past_key_values
    if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
        return outputs[1]
    return None


def _extract_hidden_from_core_outputs(outputs):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


def _target_to_tensor(target, *, device) -> torch.Tensor:
    if isinstance(target, torch.Tensor):
        return target.reshape(-1).to(device=device, dtype=torch.long)
    return torch.tensor(target, device=device, dtype=torch.long).reshape(-1)


def _tokens_to_list(tokens) -> list[int]:
    if isinstance(tokens, torch.Tensor):
        return [int(x) for x in tokens.reshape(-1).detach().cpu().tolist()]
    return [int(x) for x in tokens]


def _target_to_list(target, limit: int | None = None) -> list[int]:
    if isinstance(target, torch.Tensor):
        flat = target.reshape(-1)
        if limit is not None:
            flat = flat[:limit]
        return [int(x) for x in flat.detach().cpu().tolist()]
    vals = list(target)
    return vals[:limit] if limit is not None else vals


def _maybe_sync():
    if os.getenv("AICAS_SPEC_SYNC_PROFILE", "0") == "1" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _suppress_eos_targets_before_min_length(
    model,
    target_tensor: torch.Tensor,
    hidden_steps: list,
    eos_ids: set[int],
    generated_len: int,
    min_new_tokens: int,
) -> torch.Tensor:
    """Match HF MinNewTokens behavior for verifier argmax targets.

    The verifier usually takes a fast argmax path and only returns token ids.
    If that raw argmax is EOS before min_new_tokens, greedy generate would have
    masked EOS and selected the next best token. Recompute only those rare
    positions from the captured hidden state.
    """
    if not eos_ids or min_new_tokens <= 0 or target_tensor.numel() == 0:
        return target_tensor
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return target_tensor

    eos_list = [int(x) for x in eos_ids]
    out = target_tensor
    for idx in range(int(out.numel())):
        token_pos = int(generated_len) + idx + 1
        if token_pos >= int(min_new_tokens):
            continue
        cur = int(out[idx].item())
        if cur not in eos_ids:
            continue
        hidden = hidden_steps[idx] if idx < len(hidden_steps) else None
        if not isinstance(hidden, torch.Tensor):
            continue
        with torch.no_grad():
            logits = lm_head(hidden[:, -1, :]).float()
            for eos_id in eos_list:
                if 0 <= eos_id < int(logits.shape[-1]):
                    logits[:, eos_id] = -torch.inf
            replacement = logits.argmax(dim=-1).reshape(-1)[0].to(device=out.device, dtype=out.dtype)
        if out is target_tensor:
            out = target_tensor.clone()
        out[idx] = replacement
    return out


def _argmax_with_min_new_tokens(
    logits: torch.Tensor,
    eos_ids: set[int],
    generated_len: int,
    min_new_tokens: int,
) -> int:
    """Take greedy argmax with the same EOS gate as HF min_new_tokens."""
    if not eos_ids or int(generated_len) >= int(min_new_tokens):
        return int(logits.reshape(-1, logits.shape[-1])[0].argmax(dim=-1).item())
    masked = logits.float().clone()
    for eos_id in eos_ids:
        if 0 <= int(eos_id) < int(masked.shape[-1]):
            masked[..., int(eos_id)] = -torch.inf
    return int(masked.reshape(-1, masked.shape[-1])[0].argmax(dim=-1).item())


class _temporary_model_attr:
    def __init__(self, obj, name: str, value):
        self.obj = obj
        self.name = name
        self.value = value
        self.old = None
        self.had = False

    def __enter__(self):
        self.had = hasattr(self.obj, self.name)
        self.old = getattr(self.obj, self.name, None)
        setattr(self.obj, self.name, self.value)

    def __exit__(self, exc_type, exc, tb):
        if self.had:
            setattr(self.obj, self.name, self.old)
        else:
            try:
                delattr(self.obj, self.name)
            except Exception:
                pass


def _fused_verify_chain_core(
    model,
    past_kv,
    candidate_ids: torch.Tensor,
    token_state: dict | None,
    position_kwargs: dict,
):
    """Verify a linear candidate chain with one target-model forward.

    Feed all proposed tokens at once, then compare candidate[i + 1] with the
    target model's argmax at logits[i].
    """
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1:
        raise ValueError("candidate_ids must have shape [1, k]")
    disable_prefill_compile = os.getenv("AICAS_SPEC_FUSED_DISABLE_PREFILL_COMPILE", "1") == "1"
    active_ctx = (
        _temporary_model_attr(model, "_aicas_active_max_new_tokens", 128)
        if disable_prefill_compile
        else nullcontext()
    )
    core = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if core is None or lm_head is None:
        raise RuntimeError("fused verifier requires model.model and lm_head")
    use_core_argmax = (
        os.getenv("AICAS_SPEC_FUSED_CORE_ARGMAX", "1") == "1"
        and can_use_lm_head_argmax_triton_multi is not None
        and lm_head_argmax_triton_multi is not None
    )
    _spec_kernel_trace_once(
        "fused_cache_type",
        f"[spec_kernel] fused verifier cache={type(past_kv).__name__} q={int(candidate_ids.shape[1])}",
    )
    kwargs = {
        "input_ids": candidate_ids,
        "past_key_values": past_kv,
        "use_cache": True,
        "return_dict": False,
        "output_hidden_states": False,
        "output_attentions": False,
        **position_kwargs,
    }
    with active_ctx:
        with torch.no_grad():
            with spec_qk_rotary(model), spec_multitoken_attention(model):
                outputs = core(**kwargs)
    hidden = _extract_hidden_from_core_outputs(outputs)
    vpast = _extract_past_from_core_outputs(outputs)
    if not isinstance(hidden, torch.Tensor) or vpast is None:
        hidden_desc = None if hidden is None else (tuple(hidden.shape), str(hidden.dtype), str(hidden.device))
        raise RuntimeError(f"fused verifier core path did not return hidden/past; hidden={hidden_desc}")

    vhidden_steps = [
        hidden[:, idx:idx + 1, :] for idx in range(int(hidden.shape[1]))
    ]
    if (
        use_core_argmax
        and can_use_lm_head_argmax_triton_multi(
            hidden,
            lm_head.weight,
            getattr(lm_head, "bias", None),
        )
    ):
        workspace = None
        if isinstance(token_state, dict):
            workspace = token_state.get("fused_lm_head_workspace")
        tokens, workspace = lm_head_argmax_triton_multi(
            hidden,
            lm_head.weight,
            getattr(lm_head, "bias", None),
            workspace=workspace,
        )
        if isinstance(token_state, dict):
            token_state["fused_lm_head_workspace"] = workspace
        _spec_kernel_trace_once(
            "lm_head_argmax_multi",
            f"[spec_kernel] lm_head_argmax_triton_multi hit hidden={tuple(hidden.shape)}",
        )
        return tokens.reshape(-1).to(dtype=torch.long), vpast, vhidden_steps
    if os.getenv("AICAS_SPEC_FUSED_REQUIRE_CORE", "0") == "1":
        hidden_desc = tuple(hidden.shape), str(hidden.dtype), str(hidden.device)
        raise RuntimeError(f"fused core argmax unavailable; hidden={hidden_desc}")

    logits = lm_head(hidden)
    return logits.argmax(dim=-1)[0].to(dtype=torch.long), vpast, vhidden_steps


def _fused_verify_chain(model, past_kv, candidate_ids: torch.Tensor, token_state: dict | None = None):
    position_kwargs = _decode_position_kwargs(model, past_kv, candidate_ids)
    return _fused_verify_chain_core(
        model,
        past_kv,
        candidate_ids,
        token_state,
        position_kwargs,
    )


def _direct_text_verify_chain(
    model,
    past_kv,
    candidate_ids: torch.Tensor,
    token_state: dict | None = None,
):
    """Verify candidates through Qwen3-VL text model only.

    This avoids the outer multimodal model wrapper and is the first step toward
    a hand-written small-Q verifier. It still uses the model's decoder layers,
    but it allows spec-only layer patches to own the hot path.
    """
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1:
        raise ValueError("candidate_ids must have shape [1, k]")
    core = getattr(model, "model", None)
    text_model = getattr(core, "language_model", None)
    lm_head = getattr(model, "lm_head", None)
    if text_model is None or lm_head is None:
        raise RuntimeError("direct text verifier requires model.model.language_model and lm_head")
    kwargs = {
        "input_ids": candidate_ids,
        "past_key_values": past_kv,
        "use_cache": True,
        "return_dict": True,
        **_decode_position_kwargs(model, past_kv, candidate_ids),
    }
    disable_prefill_compile = os.getenv("AICAS_SPEC_FUSED_DISABLE_PREFILL_COMPILE", "1") == "1"
    active_ctx = (
        _temporary_model_attr(model, "_aicas_active_max_new_tokens", 128)
        if disable_prefill_compile
        else nullcontext()
    )
    with active_ctx:
        with torch.no_grad():
            with spec_qk_rotary(model), spec_multitoken_attention(model):
                outputs = text_model(**kwargs)

    hidden = getattr(outputs, "last_hidden_state", None)
    vpast = getattr(outputs, "past_key_values", None)
    if not isinstance(hidden, torch.Tensor) or vpast is None:
        raise RuntimeError("direct text verifier did not return hidden/past")
    if (
        can_use_lm_head_argmax_triton_multi is None
        or lm_head_argmax_triton_multi is None
        or not can_use_lm_head_argmax_triton_multi(
            hidden,
            lm_head.weight,
            getattr(lm_head, "bias", None),
        )
    ):
        raise RuntimeError("direct text verifier requires lm_head_argmax_triton_multi")
    workspace = token_state.get("direct_lm_head_workspace") if isinstance(token_state, dict) else None
    tokens, workspace = lm_head_argmax_triton_multi(
        hidden,
        lm_head.weight,
        getattr(lm_head, "bias", None),
        workspace=workspace,
    )
    if isinstance(token_state, dict):
        token_state["direct_lm_head_workspace"] = workspace
    _spec_kernel_trace_once(
        "direct_text_verifier",
        f"[spec_kernel] direct_text verifier hit hidden={tuple(hidden.shape)}",
    )
    vhidden_steps = [
        hidden[:, idx:idx + 1, :] for idx in range(int(hidden.shape[1]))
    ]
    return tokens.reshape(-1).to(dtype=torch.long), vpast, vhidden_steps


def _select_static_cache_len(total_len: int) -> int:
    from spec_decode.eagle3_cache_adapter import select_static_cache_len

    return select_static_cache_len(total_len)


def _static_cache_fused_verify_chain(
    model,
    past_kv,
    candidate_ids: torch.Tensor,
    token_state: dict | None = None,
):
    """Run the existing fused verifier against a reusable StaticCache.

    This is an intermediate backend: it is not the final graph verifier, but it
    forces the small-Q verifier through StaticCache so the cache update kernel
    and future CUDA Graph capture can be validated independently.
    """
    if _copy_cache_into_static is None:
        raise RuntimeError("decode_graph_runtime static cache helpers unavailable")
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != 1:
        raise ValueError("candidate_ids must have shape [1, k]")
    if token_state is None:
        token_state = {}
    mt_backend = os.getenv("AICAS_SPEC_MT_ATTN_BACKEND", "").strip().lower()
    if (
        mt_backend in {"flashdecode", "fd"}
        and os.getenv("AICAS_SPEC_FLASHDECODE_ALLOW_STATIC_CACHE", "0") != "1"
    ):
        return _fused_verify_chain(model, past_kv, candidate_ids, token_state=token_state)

    from transformers.cache_utils import StaticCache

    start_pos = _cache_seq_len(past_kv)
    total_len = start_pos + int(candidate_ids.shape[1])
    max_cache_len = _select_static_cache_len(total_len)
    cache_key = (
        int(max_cache_len),
        int(candidate_ids.device.index if candidate_ids.device.index is not None else 0),
        str(candidate_ids.dtype),
    )
    static_cache = token_state.get("static_fused_cache")
    if (
        static_cache is None
        or token_state.get("static_fused_cache_key") != cache_key
    ):
        static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
        token_state["static_fused_cache"] = static_cache
        token_state["static_fused_cache_key"] = cache_key

    if past_kv is not static_cache:
        reset_fn = getattr(static_cache, "reset", None)
        if callable(reset_fn):
            reset_fn()
        _copy_cache_into_static(past_kv, static_cache, seq_len=start_pos)
    _spec_kernel_trace_once(
        "static_fused_cache",
        f"[spec_kernel] static fused verifier cache mcl={max_cache_len} start={start_pos}",
    )
    return _fused_verify_chain(model, static_cache, candidate_ids, token_state=token_state)


def _copy_static_tail(src_cache, dst_cache, start_pos: int, q_len: int) -> None:
    layers = getattr(dst_cache, "layers", None)
    if not layers:
        return
    for layer_idx, dst_layer in enumerate(layers):
        try:
            k, v = _get_layer_kv(src_cache, layer_idx) if _get_layer_kv is not None else (None, None)
        except Exception:
            k, v = None, None
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            try:
                src_layer = src_cache.layers[layer_idx]
                k = getattr(src_layer, "keys", None)
                v = getattr(src_layer, "values", None)
            except Exception:
                continue
        keys = getattr(dst_layer, "keys", None)
        values = getattr(dst_layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            continue
        end_pos = min(int(start_pos) + int(q_len), int(k.shape[2]), int(keys.shape[2]))
        if end_pos <= start_pos:
            continue
        keys[:, :, start_pos:end_pos, :].copy_(k[:, :, start_pos:end_pos, :])
        values[:, :, start_pos:end_pos, :].copy_(v[:, :, start_pos:end_pos, :])


def _multi_kernel_verify_chain(
    model,
    past_kv,
    candidate_ids: torch.Tensor,
    token_state: dict | None = None,
):
    """Current multi-kernel backend entry.

    The default eager implementation uses direct text-model verification. If
    `AICAS_SPEC_MULTI_KERNEL_STATIC_CACHE=1`, it first copies the input KV to a
    reusable StaticCache so the static cache update path can be validated.
    """
    mt_backend = os.getenv("AICAS_SPEC_MT_ATTN_BACKEND", "").strip().lower()
    flashdecode_static_ok = os.getenv("AICAS_SPEC_FLASHDECODE_ALLOW_STATIC_CACHE", "0") == "1"
    if (
        os.getenv("AICAS_SPEC_MULTI_KERNEL_STATIC_CACHE", "1") != "1"
        or (mt_backend in {"flashdecode", "fd"} and not flashdecode_static_ok)
    ):
        return _direct_text_verify_chain(model, past_kv, candidate_ids, token_state=token_state)
    if token_state is None:
        token_state = {}
    if _copy_cache_into_static is None:
        raise RuntimeError("decode_graph_runtime static cache helpers unavailable")
    from transformers.cache_utils import StaticCache

    start_pos = _cache_seq_len(past_kv)
    total_len = start_pos + int(candidate_ids.shape[1])
    max_cache_len = _select_static_cache_len(total_len)
    cache_key = (
        "direct_text",
        int(max_cache_len),
        int(candidate_ids.device.index if candidate_ids.device.index is not None else 0),
        str(candidate_ids.dtype),
    )
    static_cache = token_state.get("multi_kernel_cache")
    if static_cache is None or token_state.get("multi_kernel_cache_key") != cache_key:
        static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
        token_state["multi_kernel_cache"] = static_cache
        token_state["multi_kernel_cache_key"] = cache_key

    if past_kv is not static_cache:
        reset_fn = getattr(static_cache, "reset", None)
        if callable(reset_fn):
            reset_fn()
        _copy_cache_into_static(past_kv, static_cache, seq_len=start_pos)
    _spec_kernel_trace_once(
        "multi_kernel_cache",
        f"[spec_kernel] multi_kernel static cache mcl={max_cache_len} start={start_pos}",
    )
    return _direct_text_verify_chain(model, static_cache, candidate_ids, token_state=token_state)



def greedy_speculative_generate(
    model, inputs: dict, heads,
    max_new_tokens: int, draft_len: int = 4, debug: bool = False,
    min_new_tokens: int = 0,
):
    device = next(model.parameters()).device
    eos_ids = _eos_id_set(model.generation_config.eos_token_id)
    min_new_tokens = max(0, int(min_new_tokens or 0))

    stats = {"rounds": 0, "accepted": 0, "drafted": 0, "mismatches": 0,
             "draft_ms": 0.0, "verify_ms": 0.0, "rollback_ms": 0.0,
             "fallback": 0, "constant_fastpath_tokens": 0,
             "verifier": {}}
    runtime_token_state = {}

    # Prefill: keep the hidden/logit source aligned with the verifier path.
    forward_inputs = _prepare_prefill_forward_inputs(model, inputs)
    use_runtime_prefill = (
        os.getenv(
            "AICAS_SPEC_PREFILL_RUNTIME",
            "0",
        ) == "1"
        and spec_decode_prefill_step is not None
    )
    if use_runtime_prefill:
        with torch.no_grad():
            next_token_t, prefill_out, hidden = spec_decode_prefill_step(
                model,
                forward_inputs,
                token_state=runtime_token_state,
            )
        past_kv = prefill_out.past_key_values
        if isinstance(hidden, torch.Tensor):
            hidden = hidden[:, -1:, :]
        next_target = _suppress_eos_targets_before_min_length(
            model,
            next_token_t.reshape(-1).to(device=device, dtype=torch.long),
            [hidden],
            eos_ids,
            generated_len=-1,
            min_new_tokens=min_new_tokens,
        )
        next_tok = int(next_target.reshape(-1)[0].item())
    else:
        with torch.no_grad():
            prefill_out, prefill_hidden = _forward_with_last_hidden(
                model,
                **forward_inputs,
                use_cache=True,
                logits_to_keep=1,
            )
        past_kv = prefill_out.past_key_values
        hidden = prefill_hidden[:, -1:, :]                   # [1, 1, hidden]
        logits = prefill_out.logits[:, -1:, :]               # [1, vocab]
        next_tok = _argmax_with_min_new_tokens(
            logits,
            eos_ids,
            generated_len=0,
            min_new_tokens=min_new_tokens,
        )
    if hidden is None:
        raise RuntimeError("spec decode prefill did not return hidden state")
    tokens = []
    candidate_buf = torch.empty((1, draft_len), device=device, dtype=torch.long)
    constant_token_t = None
    constant_token = os.getenv("AICAS_SPEC_DRAFT_CONSTANT_TOKEN", "").strip()
    if constant_token:
        constant_token_t = torch.tensor(int(constant_token), device=device, dtype=torch.long)

    if debug:
        print(f"[spec] prefill done, next={next_tok}")

    while len(tokens) < max_new_tokens:
        # ── Draft k-1 tokens ──
        _maybe_sync()
        t0 = time.perf_counter()
        need_hidden = not bool(constant_token)
        draft_tensor = None
        if constant_token:
            draft_tokens = [int(constant_token)] * max(0, draft_len - 1)
        elif hasattr(heads, "draft_tensor"):
            draft_tensor = heads.draft_tensor(hidden.squeeze(0).squeeze(0), next_tok)
            if draft_tensor.numel() > max(0, draft_len - 1):
                draft_tensor = draft_tensor[:max(0, draft_len - 1)]
            draft_tokens = None
        else:
            draft_tokens = heads.draft(hidden.squeeze(0).squeeze(0), next_tok)
        _maybe_sync()
        draft_ms = (time.perf_counter() - t0) * 1000
        drafted_len = int(draft_tensor.numel()) if isinstance(draft_tensor, torch.Tensor) else len(draft_tokens)
        stats["drafted"] += drafted_len

        # ── Verify: one target forward with k tokens ──
        cand_len = 1 + drafted_len
        if cand_len <= int(candidate_buf.shape[1]):
            cand_ids = candidate_buf[:, :cand_len]
            cand_ids[0, 0].fill_(int(next_tok))
            if cand_len > 1:
                if constant_token_t is not None:
                    cand_ids[0, 1:cand_len].fill_(constant_token_t)
                elif isinstance(draft_tensor, torch.Tensor):
                    cand_ids[0, 1:cand_len].copy_(draft_tensor[:drafted_len].to(device=device, dtype=torch.long))
                else:
                    cand_ids[0, 1:cand_len].copy_(torch.tensor(draft_tokens, device=device, dtype=torch.long))
        else:
            cand_ids = torch.empty((1, cand_len), device=device, dtype=torch.long)
            cand_ids[0, 0].fill_(int(next_tok))
            if drafted_len > 0:
                if isinstance(draft_tensor, torch.Tensor):
                    cand_ids[0, 1:cand_len].copy_(draft_tensor[:drafted_len].to(device=device, dtype=torch.long))
                else:
                    cand_ids[0, 1:cand_len].copy_(torch.tensor(draft_tokens, device=device, dtype=torch.long))
        candidate = None
        _maybe_sync()
        t0 = time.perf_counter()
        verify_backend = os.getenv("AICAS_SPEC_VERIFY_BACKEND", "").strip().lower()
        fused_graph_backends = {"fused_graph", "fused-cudagraph", "fused_cudagraph"}
        decode_graph_backends = {"decode_graph", "graph", "cudagraph"}
        verifier_done = False
        use_multi_kernel = verify_backend == "multi_kernel"
        use_static_fused = verify_backend in {
            "static_fused",
            "static-cache",
            "static_cache",
        }
        use_fused_graph = (
            verify_backend in fused_graph_backends
            and os.getenv("AICAS_SPEC_VERIFY_CUDAGRAPH", "0") == "1"
            and not runtime_token_state.get("disable_fused_graph", False)
        )
        use_fused_verify = (
            os.getenv("AICAS_SPEC_VERIFY_FUSED_FORWARD", "0") == "1"
            and (
                verify_backend not in (decode_graph_backends | {"runtime", "multi_kernel"} | fused_graph_backends)
                or (
                    verify_backend in fused_graph_backends
                    and runtime_token_state.get("disable_fused_graph", False)
                )
            )
        )
        if use_fused_graph:
            verifier_name = "fused_forward"
            vtarget, vpast, vhidden_steps = _fused_verify_chain(
                model,
                past_kv,
                cand_ids,
                token_state=runtime_token_state,
            )
            verifier_done = True
            use_fused_graph = False
            use_fused_verify = False
        if (not verifier_done) and use_multi_kernel:
            verifier_name = "multi_kernel"
            try:
                vtarget, vpast, vhidden_steps = _multi_kernel_verify_chain(
                    model,
                    past_kv,
                    cand_ids,
                    token_state=runtime_token_state,
                )
                verifier_done = True
            except Exception as exc:
                if os.getenv("AICAS_SPEC_MULTI_KERNEL_REQUIRE", "0") == "1":
                    raise
                if debug:
                    print(f"[spec] multi_kernel fallback: {exc}")
                use_multi_kernel = False

        if (not verifier_done) and use_fused_verify:
            verifier_name = "static_fused" if use_static_fused else "fused_forward"
            try:
                if use_static_fused:
                    vtarget, vpast, vhidden_steps = _static_cache_fused_verify_chain(
                        model,
                        past_kv,
                        cand_ids,
                        token_state=runtime_token_state,
                    )
                else:
                    vtarget, vpast, vhidden_steps = _fused_verify_chain(
                        model,
                        past_kv,
                        cand_ids,
                        token_state=runtime_token_state,
                    )
                verifier_done = True
            except Exception as exc:
                if os.getenv("AICAS_SPEC_FUSED_REQUIRE_CORE", "0") == "1":
                    raise
                if debug:
                    print(f"[spec] fused verifier fallback: {exc}")
                use_fused_verify = False

        graph_verify = getattr(model, "_aicas_spec_decode_verify_chain", None)
        use_graph_verify = (
            not verifier_done
            and not use_multi_kernel
            and
            not use_fused_verify
            and not use_fused_graph
            and verify_backend in decode_graph_backends
            and os.getenv("AICAS_SPEC_VERIFY_CUDAGRAPH", "0") == "1"
            and callable(graph_verify)
            and not runtime_token_state.get("disable_graph_verify", False)
        )
        if use_graph_verify:
            verifier_name = "decode_cudagraph"
            try:
                with torch.no_grad():
                    vtarget, vpast, vhidden_steps = graph_verify(
                        past_kv,
                        cand_ids,
                        token_state=runtime_token_state,
                    )
                if (
                    os.getenv("AICAS_SPEC_GRAPH_VALIDATE_FIRST", "1") == "1"
                    and not runtime_token_state.get("graph_validated", False)
                ):
                    ref_target, ref_past, ref_hidden_steps = _fused_verify_chain(
                        model,
                        past_kv,
                        cand_ids,
                        token_state=runtime_token_state,
                    )
                    if not torch.equal(
                        _target_to_tensor(vtarget, device=device),
                        _target_to_tensor(ref_target, device=device),
                    ):
                        runtime_token_state["disable_graph_verify"] = True
                        if debug:
                            print("[spec] graph verifier target mismatch -> fused_forward")
                        verifier_name = "fused_forward"
                        vtarget, vpast, vhidden_steps = ref_target, ref_past, ref_hidden_steps
                        use_graph_verify = False
                        use_fused_verify = True
                    else:
                        runtime_token_state["graph_validated"] = True
                verifier_done = True
            except Exception as exc:
                if debug:
                    print(f"[spec] graph verifier fallback: {exc}")
                use_graph_verify = False
                verifier_name = "fused_forward"
                vtarget, vpast, vhidden_steps = _fused_verify_chain(
                    model,
                    past_kv,
                    cand_ids,
                    token_state=runtime_token_state,
                )
                verifier_done = True
                use_fused_verify = True

        if (not verifier_done) and (not use_multi_kernel) and (not use_fused_verify) and (not use_fused_graph) and (not use_graph_verify) and spec_decode_verify_chain is not None:
            verifier_name = "decode_runtime"
            with torch.no_grad():
                vtarget, vpast, vhidden_steps = spec_decode_verify_chain(
                    model,
                    past_kv,
                    cand_ids,
                    token_state=runtime_token_state,
                )
            verifier_done = True
        elif (not verifier_done) and (not use_multi_kernel) and (not use_fused_verify) and (not use_fused_graph) and (not use_graph_verify):
            verifier_name = "direct_forward"
            with torch.no_grad():
                with spec_qk_rotary(model), spec_multitoken_attention(model):
                    vout, vhidden = _forward_with_last_hidden(
                        model,
                        input_ids=cand_ids,
                        past_key_values=past_kv,
                        use_cache=True,
                        logits_to_keep=cand_len,
                        **_decode_position_kwargs(model, past_kv, cand_ids),
                    )
            vlogits = vout.logits                    # [1, k, vocab]
            vpast = vout.past_key_values
            vtarget = vlogits.argmax(dim=-1)[0].to(dtype=torch.long)
            vhidden_steps = [
                vhidden[:, idx:idx + 1, :] for idx in range(int(vhidden.shape[1]))
            ]
            verifier_done = True
        _maybe_sync()
        verify_ms = (time.perf_counter() - t0) * 1000
        if debug and stats["rounds"] == 0:
            print(f"[spec] verifier={verifier_name}")
            print(f"[spec] candidate={_target_to_list(cand_ids[0], 8)} target={_target_to_list(vtarget, 8)}")
            if verifier_name == "fused_forward":
                print(f"[spec] fused_forward_len={cand_len}")
        stats["verifier"][verifier_name] = stats["verifier"].get(verifier_name, 0) + 1

        # ── Accept prefix match ──
        # vtarget[i] = model's prediction after seeing candidate[:i+1].
        # Compare candidate[i + 1] against vtarget[i].
        target_tensor = _target_to_tensor(vtarget, device=device)
        target_tensor = _suppress_eos_targets_before_min_length(
            model,
            target_tensor,
            vhidden_steps,
            eos_ids,
            generated_len=len(tokens),
            min_new_tokens=min_new_tokens,
        )
        if drafted_len > 0:
            draft_view = cand_ids[0, 1:1 + drafted_len]
            match_tensor = draft_view.eq(target_tensor[:drafted_len])
            mismatch = torch.nonzero(~match_tensor, as_tuple=False)
            if mismatch.numel() == 0:
                accepted = drafted_len
            else:
                accepted = int(mismatch[0, 0].item())
                stats["mismatches"] += 1
        else:
            accepted = 0
        # Keep state exact: on mismatch, append only the verified candidate prefix.
        # The verifier's correction token was predicted but not fed, so it becomes
        # next_tok for the following round instead of being appended immediately.
        append_count = 1 + accepted
        append_count = min(append_count, max_new_tokens - len(tokens))
        tokens.extend(_tokens_to_list(cand_ids[0, :append_count]))
        stats["accepted"] += accepted
        stats["rounds"] += 1
        stats["draft_ms"] += draft_ms
        stats["verify_ms"] += verify_ms

        constant_fastpath = (
            os.getenv("AICAS_SPEC_CONSTANT_FASTPATH", "0") == "1"
            and constant_token_t is not None
            and accepted == drafted_len
            and append_count == cand_len
            and len(tokens) < max_new_tokens
        )
        if constant_fastpath:
            const_id = int(constant_token_t.item())
            target_prefix = target_tensor[:cand_len]
            if bool(target_prefix.eq(const_id).all().item()):
                remaining = max_new_tokens - len(tokens)
                if remaining > 0:
                    tokens.extend([const_id] * remaining)
                    stats["constant_fastpath_tokens"] += remaining
                    if debug:
                        print(f"[spec] constant_fastpath fill={remaining} token={const_id}")
                    break

        # ── KV update + rollback ──
        _maybe_sync()
        t0 = time.perf_counter()
        total_fed = cand_len
        total_kept = append_count
        keep_hidden_idx = max(0, total_kept - 1)
        if total_kept < total_fed:
            crop_by = total_kept - total_fed
            past_kv = _crop_kv(vpast, crop_by)
            new_hidden = vhidden_steps[keep_hidden_idx]
            if new_hidden is not None:
                hidden = new_hidden
            elif need_hidden:
                raise RuntimeError("spec decode missing rollback hidden; disable AICAS_SPEC_GRAPH_LAST_HIDDEN_ONLY")
            next_tok = int(target_tensor[accepted].item())
        else:
            past_kv = vpast
            new_hidden = vhidden_steps[-1]
            if new_hidden is not None:
                hidden = new_hidden
            next_tok = int(target_tensor[-1].item())
        if need_hidden and hidden is None:
            raise RuntimeError("spec decode verifier did not return hidden state")
        _maybe_sync()
        stats["rollback_ms"] += (time.perf_counter() - t0) * 1000

        trace_rounds = int(os.getenv("AICAS_SPEC_TRACE_ROUNDS", "0") or "0")
        if debug and stats["rounds"] <= trace_rounds:
            print(
                f"[spec] round={stats['rounds']} accepted={accepted} "
                f"candidate={_target_to_list(cand_ids[0, :cand_len])} target={_target_to_list(target_tensor, cand_len)}"
            )

        if eos_ids and len(tokens) >= min_new_tokens and tokens and tokens[-1] in eos_ids:
            break

    if debug:
        r = max(stats["rounds"], 1)
        print(f"[spec] rounds={stats['rounds']} accepted={stats['accepted']} "
              f"drafted={stats['drafted']} avg_accept={stats['accepted']/r:.1f} "
              f"draft={stats['draft_ms']/r:.1f}ms verify={stats['verify_ms']/r:.1f}ms "
              f"rollback={stats['rollback_ms']/r:.1f}ms/round "
              f"constant_fastpath={stats['constant_fastpath_tokens']} "
              f"verifier={stats['verifier']}")

    return tokens, stats


def _crop_kv(past_kv, delta: int):
    """Crop last |delta| positions from KV cache. delta is negative for rollback."""
    if delta >= 0:
        return past_kv
    try:
        from transformers.cache_utils import StaticCache
        if isinstance(past_kv, StaticCache):
            cur_len = _cache_seq_len(past_kv)
            keep_len = max(0, cur_len + int(delta))
            for layer in getattr(past_kv, "layers", []):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if isinstance(keys, torch.Tensor) and keys.shape[2] > keep_len:
                    keys[:, :, keep_len:, :].zero_()
                if isinstance(values, torch.Tensor) and values.shape[2] > keep_len:
                    values[:, :, keep_len:, :].zero_()
                if hasattr(layer, "cumulative_length"):
                    try:
                        layer.cumulative_length = keep_len
                    except Exception:
                        pass
            return past_kv
    except Exception:
        pass
    try:
        crop_fn = getattr(past_kv, "crop", None)
        if callable(crop_fn):
            crop_fn(delta)
            return past_kv
    except Exception:
        pass
    try:
        cur_len = _cache_seq_len(past_kv)
        keep_len = max(0, cur_len + int(delta))
        new_cache = copy.deepcopy(past_kv)
        new_crop = getattr(new_cache, "crop", None)
        if callable(new_crop):
            new_crop(keep_len)
            return new_cache
    except Exception:
        pass
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            new_cache = DynamicCache()
            for i in range(len(past_kv)):
                k = past_kv.key_cache[i][:, :, :delta, :].contiguous()
                v = past_kv.value_cache[i][:, :, :delta, :].contiguous()
                new_cache.update(k, v, i)
            return new_cache
    except Exception:
        pass
    # Generic fallback
    try:
        model_cls = type(past_kv)
        new_kv = model_cls()
        for i in range(len(past_kv.key_cache)):
            k = past_kv.key_cache[i][:, :, :delta, :].contiguous()
            v = past_kv.value_cache[i][:, :, :delta, :].contiguous()
            new_kv.update(k, v, i)
        return new_kv
    except Exception:
        return past_kv
