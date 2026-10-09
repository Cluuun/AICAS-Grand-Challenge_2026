from __future__ import annotations

import contextlib
from contextlib import contextmanager
import os
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.masking_utils import create_causal_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Monkey-patch: AttentionInterface (a MutableMapping) lacks get_interface();
# add it so callers can do ALL_ATTENTION_FUNCTIONS.get_interface(key, fallback).
if not hasattr(type(ALL_ATTENTION_FUNCTIONS), "get_interface"):
    def _get_interface(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    type(ALL_ATTENTION_FUNCTIONS).get_interface = _get_interface

from spec_decode.multitoken_attention import spec_multitoken_attention
from spec_decode.qk_rotary import spec_qk_rotary
from spec_decode.verifier import (
    _argmax_with_min_new_tokens,
    _cache_seq_len,
    _crop_kv,
    _decode_position_kwargs,
    _eos_id_set,
    _prepare_prefill_forward_inputs,
    _suppress_eos_targets_before_min_length,
    _target_to_list,
    _target_to_tensor,
    _temporary_model_attr,
    _tokens_to_list,
)
from spec_decode.eagle3_cache_adapter import (
    _clear_static_cache_tail,
    _clone_cache_to_len,
    _copy_tree_prefix_into_static,
    _eagle3_cache_seq_len,
    _select_static_cache_len,
    _set_eagle3_cache_seq_len,
)

try:
    from decode_graph_runtime import (
        _build_decode_position_ids,
        _canonical_decode_step,
        _copy_cache_into_static,
        decode_cudagraph_step_from_cache as _runtime_decode_cudagraph_step_from_cache,
        _decode_forward_next_token,
    )
except Exception:
    _build_decode_position_ids = None
    _canonical_decode_step = None
    _copy_cache_into_static = None
    _runtime_decode_cudagraph_step_from_cache = None
    _decode_forward_next_token = None

try:
    from my_kernel.lm_head_argmax_triton import (
        can_use_lm_head_argmax_triton_multi,
        can_use_lm_head_topk_triton_multi,
        lm_head_argmax_triton_multi,
        lm_head_topk_triton_multi,
    )
except Exception:
    can_use_lm_head_argmax_triton_multi = None
    can_use_lm_head_topk_triton_multi = None
    lm_head_argmax_triton_multi = None
    lm_head_topk_triton_multi = None

try:
    from my_kernel.decode_add_rmsnorm_triton.runtime import (
        _can_use as _can_use_decode_add_rmsnorm,
        _fused_add_rmsnorm as _fused_decode_add_rmsnorm,
    )
except Exception:
    _can_use_decode_add_rmsnorm = None
    _fused_decode_add_rmsnorm = None

try:
    from my_kernel.decode_rmsnorm_triton.runtime import _rmsnorm_decode_only as _decode_rmsnorm_only
except Exception:
    _decode_rmsnorm_only = None

try:
    from my_kernel.decode_qk_rotary.runtime import decode_qk_rotary_graph_linear_mode
except Exception:
    decode_qk_rotary_graph_linear_mode = contextlib.nullcontext

try:
    from my_kernel.embedding_gather_triton import (
        can_use_embedding_gather_triton,
        embedding_gather_triton,
    )
except Exception:
    can_use_embedding_gather_triton = None
    embedding_gather_triton = None

try:
    from spec_decode.kernels.eagle3_expand_topk.runtime import run_eagle3_expand_topk4
except Exception:
    run_eagle3_expand_topk4 = None

try:
    from spec_decode.kernels.eagle3_posterior.runtime import run_eagle3_posterior
except Exception:
    run_eagle3_posterior = None

try:
    from spec_decode.kernels.eagle3_tree_metadata.runtime import run_eagle3_tree_metadata
except Exception:
    run_eagle3_tree_metadata = None

try:
    from spec_decode.kernels.eagle3_draft_tree.runtime import run_eagle3_draft_tree_select_metadata
except Exception:
    run_eagle3_draft_tree_select_metadata = None

try:
    from spec_decode.kernels.eagle3_full_prep.runtime import run_eagle3_full_verify_prep
except Exception:
    run_eagle3_full_verify_prep = None

try:
    from spec_decode.kernels.eagle3_cache_select.runtime import (
        can_use_eagle3_cache_select_compact,
        run_eagle3_cache_select,
        run_eagle3_cache_select_compact,
    )
except Exception:
    can_use_eagle3_cache_select_compact = None
    run_eagle3_cache_select = None
    run_eagle3_cache_select_compact = None


# Module-level cached env vars (read once at import time)
_EAGLE3_FULL_GRAPH_DEBUG = os.getenv("AICAS_EAGLE3_FULL_GRAPH_DEBUG", "0") == "1"

_TRACE_ONCE: set[str] = set()

@contextlib.contextmanager
def _draft_graph_attention_mode():
    """Use the same draft attention path for warmup, capture, replay validation.

    Do not override optimization switches here.  The launch environment owns
    whether fused draft Q/K rotary or any other optimized path is enabled; this
    context only marks the logical scope where graph warmup/capture/validation
    must use the same already-selected path.
    """
    yield


@contextlib.contextmanager
def _disable_compiled_submodules_for_graph(model, draft_model=None):
    """Temporarily revert torch.compile'd submodules to eager originals
    during CUDA graph capture.  torch.compile and CUDA graph capture are
    incompatible because Dynamo may call random ops (e.g. seeding) that
    are illegal inside a capture region.

    Handles both target model layers and draft model submodules.
    Compiled modules are restored immediately after capture so that
    subsequent REPLAY uses the faster compiled kernels.
    """
    saved = {}
    # Target language model layers
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is not None:
        layers = getattr(lm, "layers", None)
        if layers:
            for layer_idx, layer in enumerate(layers):
                for attr_name in ("mlp", "input_layernorm", "post_attention_layernorm"):
                    submod = getattr(layer, attr_name, None)
                    if submod is not None and hasattr(submod, "_orig_mod"):
                        saved[("target", layer_idx, attr_name)] = (layer, submod)
                        setattr(layer, attr_name, submod._orig_mod)
    # Draft model submodules
    if draft_model is not None:
        for attr_name in ("mlp", "input_layernorm", "hidden_norm", "post_attention_layernorm"):
            submod = getattr(getattr(draft_model, "midlayer", None), attr_name, None)
            if submod is not None and hasattr(submod, "_orig_mod"):
                saved[("draft_mid", 0, attr_name)] = (draft_model.midlayer, submod)
                setattr(draft_model.midlayer, attr_name, submod._orig_mod)
        submod = getattr(draft_model, "norm", None)
        if submod is not None and hasattr(submod, "_orig_mod"):
            saved[("draft_norm", 0, "norm")] = (draft_model, submod)
            setattr(draft_model, "norm", submod._orig_mod)
    try:
        yield
    finally:
        for key, (parent, orig_submod) in saved.items():
            setattr(parent, key[2], orig_submod)


def _first_token_mismatch(a: list[int], b: list[int]) -> int | None:
    limit = min(len(a), len(b))
    for idx in range(limit):
        if int(a[idx]) != int(b[idx]):
            return idx
    if len(a) != len(b):
        return limit
    return None


def _trace_once(key: str, message: str) -> None:
    if (
        os.getenv("AICAS_EAGLE3_TRACE", "0") != "1"
    ) or key in _TRACE_ONCE:
        return
    _TRACE_ONCE.add(key)
    print(message)


# ── lightweight graph hit/miss counters (module-level, read by worker) ──
_EAGLE3_GRAPH_COUNTERS: dict[str, int] = {}
_EAGLE3_GRAPH_REASONS: dict[str, str] = {}


def _graph_counter_inc(key: str, delta: int = 1) -> None:
    _EAGLE3_GRAPH_COUNTERS[key] = _EAGLE3_GRAPH_COUNTERS.get(key, 0) + int(delta)


def _graph_counter_get(key: str) -> int:
    return int(_EAGLE3_GRAPH_COUNTERS.get(key, 0))


def _graph_counter_snapshot() -> dict[str, int]:
    return dict(_EAGLE3_GRAPH_COUNTERS)


def _graph_counter_reset() -> None:
    _EAGLE3_GRAPH_COUNTERS.clear()


def _graph_reason_set(key: str, reason: str, *, limit: int = 320) -> None:
    text = str(reason).replace("\n", " | ")
    if len(text) > int(limit):
        text = text[: int(limit) - 3] + "..."
    _EAGLE3_GRAPH_REASONS[str(key)] = text


def _graph_reason_snapshot() -> dict[str, str]:
    return dict(_EAGLE3_GRAPH_REASONS)


def _graph_reason_reset() -> None:
    _EAGLE3_GRAPH_REASONS.clear()


def _is_cuda_fatal_exception(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "device-side assert" in msg
        or "cuda error" in msg
        or "eagle3 cuda failure" in msg
        or "acceleratorerror" in type(exc).__name__.lower()
    )


def _sync_cuda_debug_stage(stage: str) -> None:
    if not _EAGLE3_FULL_GRAPH_DEBUG:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"EAGLE3 CUDA failure after {stage}") from exc


def _eagle3_full_graph_debug_enabled() -> bool:
    return _EAGLE3_FULL_GRAPH_DEBUG


def _debug_tensor_preview(tensor: torch.Tensor, limit: int = 24) -> list[int]:
    try:
        flat = tensor.detach().reshape(-1)[: int(limit)].to(device="cpu", dtype=torch.long)
        return [int(x) for x in flat.tolist()]
    except Exception:
        return []


def _debug_check_int_tensor_range(
    name: str,
    tensor: torch.Tensor,
    *,
    lower: int = 0,
    upper: int | None = None,
    context: str = "",
) -> None:
    if not _eagle3_full_graph_debug_enabled():
        return
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"EAGLE3 full graph debug: {name} is not a tensor ({context})")
    if int(tensor.numel()) <= 0:
        return
    try:
        if tensor.is_cuda:
            torch.cuda.synchronize(tensor.device)
        cpu = tensor.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    except Exception as exc:
        raise RuntimeError(f"EAGLE3 full graph debug sync/copy failed at {name}: {context}") from exc
    bad = cpu < int(lower)
    if upper is not None:
        bad |= cpu >= int(upper)
    if bool(bad.any().item()):
        bad_idx = int(torch.nonzero(bad, as_tuple=False)[0].item())
        raise RuntimeError(
            f"EAGLE3 full graph invalid {name}: "
            f"range=[{int(cpu.min().item())},{int(cpu.max().item())}] "
            f"allowed=[{int(lower)},{'inf' if upper is None else int(upper)}) "
            f"first_bad_index={bad_idx} first_values={_debug_tensor_preview(cpu)} {context}"
        )


def _profile_range(name: str):
    if os.getenv("AICAS_BENCHMARK_PROFILE", "0") != "1" and os.getenv("AICAS_EAGLE3_PROFILE", "0") != "1":
        from contextlib import nullcontext
        return nullcontext()
    return torch.profiler.record_function(name)


@contextlib.contextmanager
def _temporary_preferred_blas(backend: str | None):
    backend = (backend or "").strip().lower()
    if backend not in {"cublas", "cublaslt", "default"}:
        yield
        return
    preferred = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if preferred is None:
        yield
        return
    old = None
    try:
        old = preferred()
    except Exception:
        old = None
    try:
        preferred(backend)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        if old is not None:
            try:
                preferred(old)
            except Exception:
                pass


def _eagle3_allow_ondemand_graph_capture() -> bool:
    if os.getenv("AICAS_EAGLE3_SPEC_PRECAPTURE_DONE", "0") != "1":
        return True
    return os.getenv("AICAS_EAGLE3_GRAPH_CAPTURE_ON_DEMAND_AFTER_PRECAPTURE", "0") == "1"


def _capture_eagle3_feature_layer(hidden_states: torch.Tensor) -> torch.Tensor:
    if os.getenv("AICAS_EAGLE3_CLONE_SELECTED_FEATURES", "0") == "1":
        return hidden_states.clone()
    return hidden_states


def _remember_eagle3_hidden_layout(model, hidden_states) -> None:
    layers = tuple(h for h in hidden_states if isinstance(h, torch.Tensor)) if isinstance(hidden_states, (tuple, list)) else ()
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    n_layers = len(getattr(language_model, "layers", ())) if language_model is not None else 0
    if not layers or n_layers <= 0:
        return
    slot_count = len(layers)
    repeats = 1
    if slot_count > 1 and (slot_count - 1) % n_layers == 0:
        repeats = max(1, (slot_count - 1) // n_layers)
    model._aicas_eagle3_hidden_slot_count = int(slot_count)
    model._aicas_eagle3_hidden_layer_repeats = int(repeats)


def _eagle3_feature_sources(model, layer_indices: tuple[int, int, int], n_layers: int):
    slot_count = int(getattr(model, "_aicas_eagle3_hidden_slot_count", n_layers + 1))
    repeats = int(getattr(model, "_aicas_eagle3_hidden_layer_repeats", 1))
    sources = []
    for idx in layer_indices:
        slot = int(idx)
        if slot <= 0:
            sources.append(("input", 0))
        elif slot >= slot_count - 1:
            sources.append(("norm", n_layers))
        else:
            sources.append(("layer", min(n_layers - 1, (slot - 1) // max(1, repeats))))
    return sources


@contextlib.contextmanager
def _temporary_env(values: dict[str, str]):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _stable_prefill_kernel_context():
    if os.getenv("AICAS_EAGLE3_STABLE_PREFILL_FASTPATHS", "1") == "1":
        return contextlib.nullcontext()
    return _temporary_env({
        "AICAS_EAGLE3_DRAFT_QK_ROTARY": "0",
        "AICAS_EAGLE3_DRAFT_ADD_RMSNORM": "0",
        "AICAS_EAGLE3_DRAFT_RMSNORM": "0",
    })


def _cos_sin_2d(t: torch.Tensor, q_len: int) -> torch.Tensor | None:
    if not isinstance(t, torch.Tensor):
        return None
    if t.ndim == 3 and t.shape[0] == 1 and t.shape[1] == q_len and t.shape[2] == 128:
        return t[0]
    if t.ndim == 2 and t.shape[0] == q_len and t.shape[1] == 128:
        return t
    if t.numel() == q_len * 128:
        return t.reshape(q_len, 128)
    return None


def _eagle3_draft_rmsnorm(norm_module, hidden_states: torch.Tensor) -> torch.Tensor:
    if os.getenv("AICAS_EAGLE3_DRAFT_RMSNORM", "1") != "1":
        return norm_module(hidden_states)
    if _decode_rmsnorm_only is None:
        return norm_module(hidden_states)
    if (
        not isinstance(hidden_states, torch.Tensor)
        or not hidden_states.is_cuda
        or hidden_states.dtype is not torch.bfloat16
        or hidden_states.shape[-1] > 4096
        or torch.is_grad_enabled()
        or getattr(norm_module, "training", False)
    ):
        return norm_module(hidden_states)
    rows = int(hidden_states.numel() // hidden_states.shape[-1])
    max_q = int(os.getenv("AICAS_SPEC_KERNEL_MAX_Q", "16"))
    if rows < 1 or rows > max_q:
        return norm_module(hidden_states)
    try:
        x = hidden_states.contiguous() if not hidden_states.is_contiguous() else hidden_states
        workspaces = getattr(norm_module, "_aicas_eagle3_rmsnorm_workspaces", None)
        if not isinstance(workspaces, dict):
            workspaces = {}
            norm_module._aicas_eagle3_rmsnorm_workspaces = workspaces
        key = (tuple(x.shape), x.device, x.dtype)
        workspace = workspaces.get(key)
        x2d = x.view(-1, x.shape[-1])
        if (
            workspace is None
            or workspace.device != x2d.device
            or workspace.dtype != x2d.dtype
            or workspace.shape != x2d.shape
        ):
            workspace = torch.empty_like(x2d)
            workspaces[key] = workspace
        _trace_once("eagle3_draft_rmsnorm", "[eagle3-draft] decode_rmsnorm hit")
        return _decode_rmsnorm_only(
            x,
            norm_module.weight,
            float(norm_module.variance_epsilon),
            out_buffer=workspace,
        )
    except Exception:
        if os.getenv("AICAS_EAGLE3_DRAFT_RMSNORM_REQUIRE", "0") == "1":
            raise
        return norm_module(hidden_states)


def _parse_layer_indices(spec: str, n_layers: int) -> tuple[int, int, int]:
    if not spec:
        spec = "0,1,2"
    out = []
    for item in spec.split(","):
        token = item.strip().lower()
        if token in {"mid", "middle"}:
            idx = n_layers // 2
        else:
            idx = int(token)
            if idx < 0:
                idx = n_layers + idx
        idx = max(0, min(n_layers - 1, idx))
        out.append(idx)
    while len(out) < 3:
        out.append(out[-1] if out else n_layers - 1)
    return tuple(out[:3])


def _select_eagle3_feature(hidden_states, layer_indices: tuple[int, int, int] | None = None) -> torch.Tensor:
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise RuntimeError("EAGLE3 requires target output_hidden_states=True")
    layers = tuple(h for h in hidden_states if isinstance(h, torch.Tensor))
    if not layers:
        raise RuntimeError("EAGLE3 hidden_states tuple is empty")
    if layer_indices is None:
        layer_indices = _parse_layer_indices(os.getenv("AICAS_EAGLE3_LAYER_INDICES", ""), len(layers))
    last_idx = len(layers) - 1
    selected = [layers[max(0, min(last_idx, int(idx)))] for idx in layer_indices]
    return torch.cat(selected, dim=-1)


def shift_left(tensor: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(tensor[:, -1:])
    return torch.cat((tensor[:, 1:], zero), dim=1)


class Qwen3VLEagle3Attention(nn.Module):
    """Qwen3VL text attention adapted exactly like EAGLE3.

    EAGLE3 concatenates token embeddings with target hidden features before
    q/k/v projection, so q/k/v have input dim 2H and output dim H.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        in_dim = config.hidden_size * 2
        self.q_proj = nn.Linear(in_dim, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(in_dim, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(in_dim, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            key_states = torch.cat((past_key_value[0], key_states), dim=2)
            value_states = torch.cat((past_key_value[1], value_states), dim=2)
        if getattr(self, "_aicas_draft_force_eager_attention", False):
            attention_interface = eager_attention_forward
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation,
                eager_attention_forward,
            )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights, (key_states, value_states)

    def _can_use_spec_qk(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> bool:
        if os.getenv("AICAS_EAGLE3_DRAFT_QK_ROTARY", "1") != "1":
            return False
        if not isinstance(hidden_states, torch.Tensor):
            return False
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            return False
        q_len = int(hidden_states.shape[1])
        max_q = int(os.getenv("AICAS_SPEC_KERNEL_MAX_Q", "16"))
        if q_len < 1 or q_len > max_q:
            return False
        if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
            return False
        if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
            return False
        cos = _cos_sin_2d(position_embeddings[0], q_len)
        sin = _cos_sin_2d(position_embeddings[1], q_len)
        return cos is not None and sin is not None

    def forward_spec_qk(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
    ):
        try:
            from spec_decode.kernels.bf16_qk_rotary.runtime import run_bf16_qk_rmsnorm_rotary

            if not self._can_use_spec_qk(hidden_states, position_embeddings):
                raise RuntimeError("draft qk rotary backend rejected inputs")
            batch, q_len, _ = hidden_states.shape
            if batch != 1:
                raise RuntimeError("draft qk rotary supports batch=1 only")

            input_shape = hidden_states.shape[:-1]
            q_shape = (1, q_len, self.config.num_attention_heads, self.head_dim)
            kv_shape = (1, q_len, self.config.num_key_value_heads, self.head_dim)

            q_raw = self.q_proj(hidden_states).view(q_shape).contiguous()
            k_raw = self.k_proj(hidden_states).view(kv_shape).contiguous()
            value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2).contiguous()

            cos = _cos_sin_2d(position_embeddings[0], q_len)
            sin = _cos_sin_2d(position_embeddings[1], q_len)
            if cos is None or sin is None:
                raise RuntimeError("unsupported draft cos/sin shape")

            qk = run_bf16_qk_rmsnorm_rotary(
                q_raw.reshape(q_len * self.config.num_attention_heads, self.head_dim),
                k_raw.reshape(q_len * self.config.num_key_value_heads, self.head_dim),
                cos,
                sin,
                self.q_norm.weight,
                self.k_norm.weight,
                float(self.q_norm.variance_epsilon),
            )
            if qk is None:
                raise RuntimeError("draft qk rotary backend rejected inputs")
            query_states, key_states = qk
            if past_key_value is not None:
                key_states = torch.cat((past_key_value[0], key_states), dim=2)
                value_states = torch.cat((past_key_value[1], value_states), dim=2)
            else:
                key_states = key_states.clone()
                value_states = value_states.clone()

            if getattr(self, "_aicas_draft_force_eager_attention", False):
                attention_interface = eager_attention_forward
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation,
                    eager_attention_forward,
                )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
            )
            attn_flat = attn_output.reshape(*input_shape, -1)
            oproj_out = self.o_proj(attn_flat.contiguous())
            _trace_once("eagle3_draft_qk_rotary", f"[eagle3-draft] bf16_qk_rotary hit q={q_len}")
            return oproj_out, attn_weights, (key_states, value_states)
        except Exception as exc:
            if os.getenv("AICAS_EAGLE3_DRAFT_QK_ROTARY_REQUIRE", "0") == "1":
                raise
            _trace_once(
                "eagle3_draft_qk_rotary_fallback",
                f"[eagle3-draft] bf16_qk_rotary fallback: {type(exc).__name__}: {exc}",
            )
            return self.forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
            )

    def forward_train_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache_hidden: list[list[torch.Tensor]] | None = None,
    ):
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if cache_hidden is None:
            local_cache_k, local_cache_v = [], []
        else:
            local_cache_k = list(cache_hidden[0])
            local_cache_v = list(cache_hidden[1])
        local_cache_k.append(key_states)
        local_cache_v.append(value_states)

        attn_weights = torch.matmul(query_states, local_cache_k[0].transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        for idx in range(1, len(local_cache_k)):
            same_pos = (query_states * local_cache_k[idx]).sum(-1) * self.scaling
            attn_weights = torch.cat((attn_weights, same_pos[..., None]), dim=-1)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights[..., :q_len], local_cache_v[0])
        for idx in range(1, len(local_cache_v)):
            attn_output = attn_output + attn_weights[..., q_len + idx - 1, None] * local_cache_v[idx]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), [local_cache_k, local_cache_v]


class Qwen3VLEagle3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3VLEagle3Attention(config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        norm_hidden = _eagle3_draft_rmsnorm(self.hidden_norm, hidden_states)
        norm_input = _eagle3_draft_rmsnorm(self.input_layernorm, input_emb)
        attn_input = torch.cat((norm_input, norm_hidden), dim=-1)
        attn_forward = self.self_attn.forward_spec_qk if os.getenv("AICAS_EAGLE3_DRAFT_QK_ROTARY", "1") == "1" else self.self_attn
        attn_out, _, present_key_value = attn_forward(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if (
            os.getenv("AICAS_EAGLE3_DRAFT_ADD_RMSNORM", "1") == "1"
            and
            _can_use_decode_add_rmsnorm is not None
            and _fused_decode_add_rmsnorm is not None
            and _can_use_decode_add_rmsnorm(residual, attn_out)
        ):
            try:
                workspace_key = (residual.device, residual.dtype, tuple(residual.shape))
                workspaces = getattr(self, "_aicas_decode_add_rmsnorm_workspaces", None)
                if not isinstance(workspaces, dict):
                    workspaces = {}
                    self._aicas_decode_add_rmsnorm_workspaces = workspaces
                ws_add, ws_norm, ws_partial, ws_sum = workspaces.get(workspace_key, (None, None, None, None))
                hidden_states, normed, ws_add, ws_norm, ws_partial, ws_sum = _fused_decode_add_rmsnorm(
                    residual,
                    attn_out,
                    self.post_attention_layernorm.weight,
                    float(self.post_attention_layernorm.variance_epsilon),
                    ws_add,
                    ws_norm,
                    ws_partial,
                    ws_sum,
                )
                workspaces[workspace_key] = (ws_add, ws_norm, ws_partial, ws_sum)
                residual = hidden_states
                hidden_states = normed
                _trace_once(
                    f"eagle3_draft_add_rmsnorm",
                    f"[eagle3-draft] decode_add_rmsnorm hit",
                )
            except Exception:
                hidden_states = residual + attn_out
                residual = hidden_states
                hidden_states = _eagle3_draft_rmsnorm(self.post_attention_layernorm, hidden_states)
        else:
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _eagle3_draft_rmsnorm(self.post_attention_layernorm, hidden_states)
        hidden_states = self.mlp(hidden_states)
        out = residual + hidden_states
        return out, present_key_value

    def forward_train_cache(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: list[list[torch.Tensor]],
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        residual = hidden_states
        norm_hidden = self.hidden_norm(hidden_states)
        norm_input = self.input_layernorm(input_emb)
        attn_input = torch.cat((norm_input, norm_hidden), dim=-1)
        attn_out, cache_hidden = self.self_attn.forward_train_cache(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            cache_hidden=cache_hidden,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, cache_hidden


class Eagle3DraftModel(nn.Module):
    """Strict EAGLE3 draft model adapted to Qwen3VL text blocks."""

    checkpoint_format = "qwen3vl_eagle3_strict_v2"
    legacy_checkpoint_formats = {"qwen3vl_eagle3_strict_v1"}

    def __init__(
        self,
        text_config,
        num_heads: int,
        *,
        lm_head_weight: torch.Tensor | None = None,
        embed_weight: torch.Tensor | None = None,
        draft_vocab_ids: torch.Tensor | list[int] | None = None,
    ):
        super().__init__()
        self.config = text_config
        self.hidden_size = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.num_heads = int(num_heads)
        self.padding_idx = text_config.pad_token_id
        if draft_vocab_ids is None:
            draft_vocab_tensor = None
            self.draft_vocab_size = self.vocab_size
        else:
            draft_vocab_tensor = torch.as_tensor(draft_vocab_ids, dtype=torch.long).reshape(-1).detach().cpu()
            if draft_vocab_tensor.numel() <= 0:
                raise ValueError("draft_vocab_ids must be non-empty")
            if int(draft_vocab_tensor.min().item()) < 0 or int(draft_vocab_tensor.max().item()) >= self.vocab_size:
                raise ValueError("draft_vocab_ids contains ids outside the target vocabulary")
            self.draft_vocab_size = int(draft_vocab_tensor.numel())
        if draft_vocab_tensor is None:
            self.register_buffer("draft_vocab_ids", torch.empty(0, dtype=torch.long), persistent=False)
        else:
            self.register_buffer("draft_vocab_ids", draft_vocab_tensor, persistent=True)
        self.fc = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
        self.midlayer = Qwen3VLEagle3DecoderLayer(text_config, layer_idx=0)
        self.norm = Qwen3VLTextRMSNorm(self.hidden_size, eps=text_config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=text_config)
        self.lm_head = nn.Linear(self.hidden_size, self.draft_vocab_size, bias=False)
        self.top_k = min(4, int(self.draft_vocab_size))
        self.total_tokens = 14
        self.depth = 3
        self.tree_mask = None
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self._topk_workspace = None
        self._topk_workspaces = {}
        self._draft_embed_workspaces = {}
        self._expand_topk_workspace = None
        self._tree_metadata_workspace = None
        self._draft_tree_graph_workspaces = {}
        self._draft_all_graph_entries = {}
        self._draft_all_graph_order = []
        self._draft_all_graph_failed_keys = set()
        self._draft_all_graph_failed_reasons = {}
        self._draft_step_graph_entries = {}
        self._draft_step_graph_order = []
        self._draft_step_graph_failed_keys = set()
        self._draft_step_graph_failed_reasons = {}
        self._draft_compiled_fn = None
        self._draft_graph_logical_past_len = None
        self._draft_graph_storage_past_len = None
        if embed_weight is not None:
            self.embed_tokens.weight = nn.Parameter(embed_weight.detach().clone().to(dtype=torch.bfloat16))
        if lm_head_weight is not None:
            weight = lm_head_weight.detach()
            if int(weight.shape[0]) == self.vocab_size and self.draft_vocab_size != self.vocab_size:
                weight = weight.index_select(0, self.draft_vocab_ids.to(device=weight.device))
            if int(weight.shape[0]) != self.draft_vocab_size:
                raise ValueError(
                    f"lm_head_weight rows={int(weight.shape[0])} do not match draft_vocab_size={self.draft_vocab_size}"
                )
            self.lm_head.weight = nn.Parameter(weight.clone().to(dtype=torch.bfloat16))
        self._stable_kv = None
        for param in self.embed_tokens.parameters():
            param.requires_grad_(False)

    def project_feature(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() == 1:
            feature = feature.reshape(1, 1, -1)
        elif feature.dim() == 2:
            feature = feature.unsqueeze(1)
        if int(feature.shape[-1]) == self.hidden_size:
            return feature.to(dtype=self.fc.weight.dtype)
        feature = feature.to(dtype=self.fc.weight.dtype)
        feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
        return self.fc(feature)

    def _embed_input_ids(self, input_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        weight = self.embed_tokens.weight
        use_triton = (
            os.getenv("AICAS_EAGLE3_DRAFT_EMBED_TRITON", "").strip()
            or os.getenv("AICAS_EAGLE3_GRAPH_EMBED_TRITON", "1")
        ) == "1"
        if (
            use_triton
            and can_use_embedding_gather_triton is not None
            and embedding_gather_triton is not None
            and can_use_embedding_gather_triton(input_ids, weight)
        ):
            expected = (int(input_ids.shape[0]), int(input_ids.shape[1]), int(weight.shape[1]))
            workspace_key = (
                input_ids.device,
                input_ids.dtype,
                tuple(input_ids.shape),
                weight.device,
                weight.dtype,
                tuple(weight.shape),
            )
            workspaces = getattr(self, "_draft_embed_workspaces", None)
            if not isinstance(workspaces, dict):
                workspaces = {}
                self._draft_embed_workspaces = workspaces
            out = workspaces.get(workspace_key)
            if (
                not isinstance(out, torch.Tensor)
                or out.shape != expected
                or out.device != input_ids.device
                or out.dtype != weight.dtype
            ):
                out = torch.empty(expected, device=input_ids.device, dtype=weight.dtype)
                workspaces[workspace_key] = out
            embedded = embedding_gather_triton(input_ids, weight, out=out)
            return embedded if embedded.dtype == dtype else embedded.to(dtype=dtype)
        return self.embed_tokens(input_ids).to(dtype=dtype)

    def init_tree(self, total_tokens: int = 15, depth: int = 3, top_k: int = 4) -> None:
        self.total_tokens = max(1, int(total_tokens) - 1)
        self.depth = max(1, int(depth))
        self.top_k = min(max(1, int(top_k)), int(self.draft_vocab_size))
        device = self.embed_tokens.weight.device
        self.tree_mask_init = torch.eye(self.top_k, device=device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=device, dtype=torch.long)
        self._topk_cs_index = torch.arange(self.top_k, device=device, dtype=torch.long)
        self._zero_parent = torch.zeros(1, dtype=torch.long, device=device)

    def reset_kv(self) -> None:
        self._stable_kv = None
        self.tree_mask = None

    def _draft_tree_graph_workspace(self, *, device: torch.device, depth: int, total_tokens: int):
        key = (device, int(depth), int(total_tokens))
        workspaces = getattr(self, "_draft_tree_graph_workspaces", None)
        if not isinstance(workspaces, dict):
            workspaces = {}
            self._draft_tree_graph_workspaces = workspaces
        ws = workspaces.get(key)
        if not isinstance(ws, dict):
            ws = {
                "sample_token": torch.empty((1,), device=device, dtype=torch.long),
                "root_tokens": torch.empty((1, 4), device=device, dtype=torch.long),
                "root_scores": torch.empty((4,), device=device, dtype=torch.float32),
                "topk_ids_slots": torch.empty((int(depth), 4, 4), device=device, dtype=torch.long),
                "tree_select": None,
            }
            workspaces[key] = ws
        return ws

    def _lm_head_logp_topk(self, hidden: torch.Tensor, top_k: int, logits_processor=None):
        if logits_processor is not None:
            raise RuntimeError("EAGLE3 optimized path requires greedy decoding; logits_processor is unsupported")
        hidden = hidden.to(dtype=self.lm_head.weight.dtype)
        bias = getattr(self.lm_head, "bias", None)
        if can_use_lm_head_topk_triton_multi is None or lm_head_topk_triton_multi is None:
            raise RuntimeError("EAGLE3 optimized path requires lm_head_topk_triton_multi")
        if not can_use_lm_head_topk_triton_multi(hidden, self.lm_head.weight, bias, top_k=top_k):
            raise RuntimeError(
                "EAGLE3 optimized topK kernel is not applicable "
                f"hidden={tuple(hidden.shape)} hidden_dtype={hidden.dtype} "
                f"weight_dtype={self.lm_head.weight.dtype} cuda={hidden.is_cuda}"
            )
        hidden_for_key = hidden.reshape(-1, hidden.shape[-1]) if hidden.ndim == 3 else hidden
        workspace_key = (
            hidden_for_key.device,
            hidden_for_key.dtype,
            tuple(hidden_for_key.shape),
            int(top_k),
        )
        workspace = self._topk_workspaces.get(workspace_key)
        with _profile_range("eagle3_draft.lm_head_topk"):
            vals, idxs, workspace = lm_head_topk_triton_multi(
                hidden,
                self.lm_head.weight,
                bias,
                top_k=top_k,
                workspace=workspace,
            )
        self._topk_workspaces[workspace_key] = workspace
        self._topk_workspace = workspace
        if self.draft_vocab_ids.numel() > 0:
            idxs = self.draft_vocab_ids.index_select(0, idxs.reshape(-1)).reshape_as(idxs)
        return idxs, vals

    def _expand_topk_select(
        self,
        topk_index: torch.Tensor,
        topk_p: torch.Tensor,
        scores: torch.Tensor,
        out_hidden: torch.Tensor,
        tree_mask: torch.Tensor,
        top_k: int,
        *,
        score_slot: int,
        score_slots: int,
    ):
        use_fused = os.getenv("AICAS_EAGLE3_FUSED_EXPAND_TOPK", "1") == "1"
        fallback_scores = scores
        if use_fused and run_eagle3_expand_topk4 is not None:
            try:
                check_fused = os.getenv("AICAS_EAGLE3_FUSED_EXPAND_TOPK_CHECK", "0") == "1"
                ref_scores_in = scores.clone() if check_fused else scores
                fallback_scores = ref_scores_in
                result, workspace = run_eagle3_expand_topk4(
                    topk_index,
                    topk_p,
                    scores,
                    out_hidden,
                    tree_mask,
                    workspace=self._expand_topk_workspace,
                    score_slot=score_slot,
                    score_slots=score_slots,
                )
                self._expand_topk_workspace = workspace
                if result is not None:
                    (
                        cu_scores,
                        topk_cs_index,
                        topk_cs_p,
                        out_ids,
                        input_hidden,
                        input_ids,
                        next_tree_mask,
                    ) = result
                    if check_fused:
                        ref_cu = topk_p + ref_scores_in[:, None]
                        ref_topk = torch.topk(ref_cu.view(-1), top_k, dim=-1)
                        ref_idx, ref_scores = ref_topk.indices, ref_topk.values
                        ref_out_ids = ref_idx // top_k
                        ref_input_ids = topk_index.view(-1)[ref_idx][None]
                        ref_input_hidden = out_hidden[:, ref_out_ids]
                        ref_tree_mask = torch.cat((tree_mask[:, :, ref_out_ids], self.tree_mask_init), dim=3)
                        ok = (
                            torch.equal(topk_cs_index, ref_idx)
                            and torch.equal(topk_cs_p, ref_scores)
                            and torch.equal(out_ids, ref_out_ids)
                            and torch.equal(input_ids, ref_input_ids)
                            and torch.equal(input_hidden, ref_input_hidden)
                            and torch.equal(next_tree_mask, ref_tree_mask)
                        )
                        if not ok:
                            raise RuntimeError("fused expand_topk output mismatch")
                    _trace_once("eagle3_fused_expand_topk", "[eagle3-draft] fused expand_topk4 hit")
                    return cu_scores, topk_cs_index, topk_cs_p, out_ids, input_hidden, input_ids, next_tree_mask
            except Exception as exc:
                if os.getenv("AICAS_EAGLE3_FUSED_EXPAND_TOPK_REQUIRE", "0") == "1":
                    raise
                _trace_once(
                    "eagle3_fused_expand_topk_fallback",
                    f"[eagle3-draft] fused expand_topk4 fallback: {type(exc).__name__}: {exc}",
                )

        scores = fallback_scores
        cu_scores = topk_p + scores[:, None]
        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        out_ids = topk_cs_index // top_k
        input_hidden = out_hidden[:, out_ids]
        input_ids = topk_index.view(-1)[topk_cs_index][None]
        next_tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)
        return cu_scores, topk_cs_index, topk_cs_p, out_ids, input_hidden, input_ids, next_tree_mask

    def _tree_metadata_reference(
        self,
        mask_index: torch.Tensor,
        total_tokens: int,
        *,
        logits_processor=None,
    ):
        mask_index_list = mask_index.tolist()

        tree_mask_cpu = torch.eye(total_tokens + 1).bool()
        tree_mask_cpu[:, 0] = True
        for i in range(total_tokens):
            tree_mask_cpu[i + 1].add_(tree_mask_cpu[mask_index_list[i]])
        tree_position_ids = torch.sum(tree_mask_cpu, dim=1) - 1

        tree_mask = tree_mask_cpu.float()[None, None].to(mask_index.device)

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num
        retrieve_indices = torch.zeros(leaf_num, int(max_depth.item()), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()
        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                node_depth = position_ids_list[i]
                for j in reversed(range(node_depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5
            retrieve_indices = sorted(
                retrieve_indices,
                key=lambda row: [x if x >= 0 else maxitem for x in row],
            )

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long, device=mask_index.device)
        tree_position_ids = tree_position_ids.to(mask_index.device)
        return retrieve_indices, tree_mask, tree_position_ids

    def _tree_metadata(
        self,
        mask_index: torch.Tensor,
        total_tokens: int,
        *,
        logits_processor=None,
    ):
        use_fused = (
            logits_processor is None
            and os.getenv("AICAS_EAGLE3_FUSED_TREE_METADATA", "1") == "1"
            and run_eagle3_tree_metadata is not None
        )
        if use_fused:
            try:
                max_depth = int(self.depth) + 2
                result, workspace = run_eagle3_tree_metadata(
                    mask_index,
                    total_tokens=total_tokens,
                    max_depth=max_depth,
                    workspace=self._tree_metadata_workspace,
                )
                self._tree_metadata_workspace = workspace
                if result is not None:
                    tree_mask, retrieve_indices, tree_position_ids, _leaf_count, _actual_depth_tensor = result
                    if os.getenv("AICAS_EAGLE3_FUSED_TREE_METADATA_CHECK", "0") == "1":
                        ref_retrieve, ref_mask, ref_pos = self._tree_metadata_reference(
                            mask_index,
                            total_tokens,
                            logits_processor=logits_processor,
                        )
                        ok = (
                            torch.equal(retrieve_indices[: ref_retrieve.shape[0], : ref_retrieve.shape[1]], ref_retrieve)
                            and torch.equal(tree_mask, ref_mask)
                            and torch.equal(tree_position_ids, ref_pos)
                        )
                        if not ok:
                            raise RuntimeError("fused tree_metadata output mismatch")
                    _trace_once("eagle3_fused_tree_metadata", "[eagle3-draft] fused tree_metadata hit")
                    return retrieve_indices, tree_mask, tree_position_ids
            except Exception as exc:
                if os.getenv("AICAS_EAGLE3_FUSED_TREE_METADATA_REQUIRE", "0") == "1":
                    raise
                _trace_once(
                    "eagle3_fused_tree_metadata_fallback",
                    f"[eagle3-draft] fused tree_metadata fallback: {type(exc).__name__}: {exc}",
                )

        return self._tree_metadata_reference(mask_index, total_tokens, logits_processor=logits_processor)

    def _kv_seq_len(self, past_key_value) -> int:
        if past_key_value is None:
            return 0
        try:
            return int(past_key_value[0].shape[2])
        except Exception:
            return 0

    def _draft_step_graph_bucket_len(self, past_len: int) -> int:
        past_len = max(0, int(past_len))
        raw = os.getenv("AICAS_EAGLE3_DRAFT_STEP_GRAPH_BUCKETS", "").strip()
        if raw:
            buckets: list[int] = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    value = int(part)
                except Exception:
                    continue
                if value > 0:
                    buckets.append(value)
            for bucket in sorted(set(buckets)):
                if past_len <= int(bucket):
                    return int(bucket)
        return int(_select_static_cache_len(past_len))

    def _copy_draft_past_into_bucket(
        self,
        dst_k: torch.Tensor,
        dst_v: torch.Tensor,
        src_k: torch.Tensor,
        src_v: torch.Tensor,
    ) -> int:
        past_len = int(src_k.shape[2])
        bucket_len = int(dst_k.shape[2])
        if past_len > bucket_len or int(src_v.shape[2]) > int(dst_v.shape[2]):
            raise RuntimeError(
                "draft step graph bucket is smaller than past cache: "
                f"past_k={tuple(src_k.shape)} past_v={tuple(src_v.shape)} "
                f"bucket_k={tuple(dst_k.shape)} bucket_v={tuple(dst_v.shape)}"
            )
        dst_k.zero_()
        dst_v.zero_()
        if past_len > 0:
            dst_k[:, :, :past_len, :].copy_(src_k)
            dst_v[:, :, : int(src_v.shape[2]), :].copy_(src_v)
        return past_len

    def _make_draft_position_ids(
        self,
        input_ids: torch.Tensor,
        past_len: int,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(position_ids, torch.Tensor):
            out = position_ids.to(device=input_ids.device, dtype=torch.long)
            if out.dim() == 0:
                out = out.view(1)
            return out
        return torch.arange(
            int(past_len),
            int(past_len) + int(input_ids.shape[1]),
            device=input_ids.device,
            dtype=torch.long,
        ).view(1, -1)

    def _draft_graph_logical_past_tensor(self):
        logical = getattr(self, "_draft_graph_logical_past_len", None)
        if isinstance(logical, torch.Tensor):
            return logical
        return None

    def _draft_graph_storage_past_len_value(self, past_key_value) -> int:
        storage = getattr(self, "_draft_graph_storage_past_len", None)
        if storage is not None:
            try:
                return int(storage)
            except Exception:
                pass
        return self._kv_seq_len(past_key_value)

    def _repack_draft_graph_past(
        self,
        entry: dict,
        past_len: int,
        q_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bucket_len = int(entry["past_bucket_len"])
        total_len = int(past_len) + int(q_len)
        out_k_src = entry["out_k"]
        out_v_src = entry["out_v"]
        ret_k = out_k_src.new_empty(
            (int(out_k_src.shape[0]), int(out_k_src.shape[1]), total_len, int(out_k_src.shape[3]))
        )
        ret_v = out_v_src.new_empty(
            (int(out_v_src.shape[0]), int(out_v_src.shape[1]), total_len, int(out_v_src.shape[3]))
        )
        if past_len > 0:
            ret_k[:, :, :past_len, :].copy_(entry["past_k"][:, :, :past_len, :])
            ret_v[:, :, :past_len, :].copy_(entry["past_v"][:, :, :past_len, :])
        if q_len > 0:
            ret_k[:, :, past_len:total_len, :].copy_(out_k_src[:, :, bucket_len : bucket_len + q_len, :])
            ret_v[:, :, past_len:total_len, :].copy_(out_v_src[:, :, bucket_len : bucket_len + q_len, :])
        return ret_k, ret_v

    def _put_lru_entry(self, store: dict, order: list, key, entry: dict, max_entries: int) -> None:
        store[key] = entry
        if key in order:
            order.remove(key)
        order.append(key)
        while len(order) > max(1, int(max_entries)):
            stale = order.pop(0)
            stale_entry = store.pop(stale, None)
            if isinstance(stale_entry, dict):
                stale_graph = stale_entry.get("graph")
                if isinstance(stale_graph, torch.cuda.CUDAGraph):
                    try:
                        stale_graph.reset()
                    except Exception:
                        pass

    def _draft_step_graph_state(self):
        if os.getenv("AICAS_EAGLE3_DRAFT_GRAPH_GLOBAL_CACHE", "1") != "1":
            return self._draft_step_graph_entries, self._draft_step_graph_order, self._draft_step_graph_failed_keys
        store = getattr(self, "_eagle3_draft_step_graph_entries", None)
        if not isinstance(store, dict):
            store = {}
            self._eagle3_draft_step_graph_entries = store
        order = getattr(self, "_eagle3_draft_step_graph_order", None)
        if not isinstance(order, list):
            order = []
            self._eagle3_draft_step_graph_order = order
        failed = getattr(self, "_eagle3_draft_step_graph_failed_keys", None)
        if not isinstance(failed, set):
            failed = set()
            self._eagle3_draft_step_graph_failed_keys = failed
        return store, order, failed

    def _draft_step_graph_failed_reason_map(self):
        if os.getenv("AICAS_EAGLE3_DRAFT_GRAPH_GLOBAL_CACHE", "1") != "1":
            return self._draft_step_graph_failed_reasons
        reasons = getattr(self, "_eagle3_draft_step_graph_failed_reasons", None)
        if not isinstance(reasons, dict):
            reasons = {}
            self._eagle3_draft_step_graph_failed_reasons = reasons
        return reasons

    def _draft_all_graph_state(self):
        store = getattr(self, "_draft_all_graph_entries", None)
        if not isinstance(store, dict):
            store = {}
            self._draft_all_graph_entries = store
        order = getattr(self, "_draft_all_graph_order", None)
        if not isinstance(order, list):
            order = []
            self._draft_all_graph_order = order
        failed = getattr(self, "_draft_all_graph_failed_keys", None)
        if not isinstance(failed, set):
            failed = set()
            self._draft_all_graph_failed_keys = failed
        return store, order, failed

    def _draft_all_graph_failed_reason_map(self):
        reasons = getattr(self, "_draft_all_graph_failed_reasons", None)
        if not isinstance(reasons, dict):
            reasons = {}
            self._draft_all_graph_failed_reasons = reasons
        return reasons

    def _position_ids(self, input_embeds: torch.Tensor, cache_position: torch.Tensor | None):
        if cache_position is None:
            cache_position = torch.arange(input_embeds.shape[1], device=input_embeds.device)
        return cache_position.view(1, 1, -1).expand(4, input_embeds.shape[0], -1)

    def _attention_mask(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position,
        past_key_value,
        position_ids: torch.Tensor | None,
    ):
        logical_past_tensor = self._draft_graph_logical_past_tensor()
        if logical_past_tensor is not None and past_key_value is not None:
            q_len = int(input_embeds.shape[1])
            storage_past_len = self._draft_graph_storage_past_len_value(past_key_value)
            kv_len = storage_past_len + q_len
            neg = torch.finfo(input_embeds.dtype).min
            out = torch.empty((1, 1, q_len, kv_len), device=input_embeds.device, dtype=input_embeds.dtype)
            out.fill_(neg)

            logical_past = logical_past_tensor.to(device=input_embeds.device, dtype=torch.long).reshape(())
            q_pos = logical_past + torch.arange(q_len, device=input_embeds.device, dtype=torch.long)

            kv_storage = torch.arange(kv_len, device=input_embeds.device, dtype=torch.long)
            current_offset = kv_storage - storage_past_len
            kv_logical = torch.where(
                kv_storage < storage_past_len,
                kv_storage,
                logical_past + current_offset,
            )
            valid_kv = (kv_storage < logical_past) | (kv_storage >= storage_past_len)
            causal = valid_kv.view(1, 1, 1, kv_len) & (
                kv_logical.view(1, 1, 1, kv_len) <= q_pos.view(1, 1, q_len, 1)
            )
            out.masked_fill_(causal, 0.0)

            tree_mask = getattr(self, "tree_mask", None)
            if tree_mask is not None:
                tree_mask = tree_mask.to(device=input_embeds.device)
                _, _, tree_q, tree_k = tree_mask.shape
                if int(tree_q) == q_len and int(tree_k) <= kv_len:
                    logical_kv_len = logical_past + q_len
                    tree_start = logical_kv_len - int(tree_k)
                    tree_logical = tree_start + torch.arange(int(tree_k), device=input_embeds.device, dtype=torch.long)
                    tree_storage = torch.where(
                        tree_logical < logical_past,
                        tree_logical,
                        storage_past_len + (tree_logical - logical_past),
                    )
                    visible = tree_mask.to(dtype=torch.bool) & (
                        tree_logical.view(1, 1, 1, int(tree_k)) <= q_pos.view(1, 1, q_len, 1)
                    )
                    tree_vals = torch.empty((1, 1, q_len, int(tree_k)), device=input_embeds.device, dtype=input_embeds.dtype)
                    tree_vals.fill_(neg)
                    tree_vals.masked_fill_(visible, 0.0)
                    out.index_copy_(3, tree_storage, tree_vals)
            return out

        past_len = self._kv_seq_len(past_key_value)
        q_len = int(input_embeds.shape[1])
        if q_len == 1 and attention_mask is None:
            return None
        kv_len = past_len + q_len
        tree_mask = getattr(self, "tree_mask", None)
        if tree_mask is not None:
            tree_mask = tree_mask.to(device=input_embeds.device)
            _, _, tree_q, tree_k = tree_mask.shape
            if int(tree_q) == q_len and int(tree_k) <= kv_len:
                out = torch.zeros(
                    (1, 1, q_len, kv_len),
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                )
                q_pos = torch.arange(past_len, kv_len, device=input_embeds.device, dtype=torch.long)
                kv_pos = torch.arange(kv_len - int(tree_k), kv_len, device=input_embeds.device, dtype=torch.long)
                visible = tree_mask.to(dtype=torch.bool) & (
                    kv_pos.view(1, 1, 1, int(tree_k)) <= q_pos.view(1, 1, q_len, 1)
                )
                out[:, :, :, -int(tree_k):].masked_fill_(
                    ~visible,
                    torch.finfo(input_embeds.dtype).min,
                )
                return out
        q_pos = torch.arange(past_len, kv_len, device=input_embeds.device, dtype=torch.long)
        kv_pos = torch.arange(kv_len, device=input_embeds.device, dtype=torch.long)
        mask = kv_pos.view(1, 1, 1, kv_len) <= q_pos.view(1, 1, q_len, 1)
        out = torch.empty((1, 1, q_len, kv_len), device=input_embeds.device, dtype=input_embeds.dtype)
        out.fill_(torch.finfo(input_embeds.dtype).min)
        out.masked_fill_(mask, 0.0)
        return out

    def _sequence_attention_mask(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None):
        batch_size, seq_len = hidden_states.shape[:2]
        q_pos = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
        kv_pos = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
        causal = kv_pos.view(1, 1, 1, seq_len) <= q_pos.view(1, 1, seq_len, 1)
        mask = torch.where(
            causal,
            torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype),
            torch.finfo(hidden_states.dtype).min,
        ).expand(batch_size, 1, seq_len, seq_len)
        if attention_mask is None:
            return mask
        valid = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        pad_mask = torch.where(
            valid[:, None, None, :],
            torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype),
            torch.finfo(hidden_states.dtype).min,
        )
        return mask + pad_mask

    def forward_block(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        past_key_value=None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool = True,
    ):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)
        input_ids = input_ids.to(hidden_states.device)
        inputs_embeds = self._embed_input_ids(input_ids, hidden_states.dtype)
        if cache_position is None:
            past_seen = self._kv_seq_len(past_key_value)
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        position_ids = self._position_ids(inputs_embeds, cache_position)
        text_position_ids = position_ids[0]
        mrope_position_ids = position_ids[1:]
        attention_mask = self._attention_mask(inputs_embeds, None, cache_position, past_key_value, text_position_ids)
        position_embeddings = self.rotary_emb(hidden_states, mrope_position_ids)
        hidden_states, present_key_value = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        return hidden_states, present_key_value if use_cache else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
    ):
        hidden_states = self.project_feature(hidden_states)
        input_ids = input_ids.to(hidden_states.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)
        batch_size, seq_len = input_ids.shape
        past_key_value = past_key_values[0] if past_key_values is not None else None
        past_len = self._kv_seq_len(past_key_value)
        inputs_embeds = self._embed_input_ids(input_ids, hidden_states.dtype)
        if position_ids is None:
            cache_position = torch.arange(past_len, past_len + seq_len, device=hidden_states.device, dtype=torch.long)
            mrope_position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
        else:
            position_ids = position_ids.to(device=hidden_states.device, dtype=torch.long)
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            cache_position = position_ids.reshape(-1)[:seq_len]
            mrope_position_ids = position_ids.view(1, batch_size, -1).expand(3, -1, -1)
            if int(mrope_position_ids.shape[2]) != int(seq_len):
                raise RuntimeError(
                    f"position_ids / input_ids length mismatch: "
                    f"position_ids={tuple(position_ids.shape)} input_ids={tuple(input_ids.shape)} "
                    f"batch_size={batch_size} seq_len={seq_len}"
                )
        position_embeddings = self.rotary_emb(hidden_states, mrope_position_ids)
        attn_mask = self._attention_mask(inputs_embeds, attention_mask, cache_position, past_key_value, None)
        hidden_states, present_key_value = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attn_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        if use_cache:
            return hidden_states, (present_key_value,)
        return hidden_states

    def _draft_step_graph_past_pair(self, past_key_values):
        if past_key_values is None:
            return None
        pair = past_key_values
        if (
            isinstance(pair, (tuple, list))
            and len(pair) == 1
            and isinstance(pair[0], (tuple, list))
        ):
            pair = pair[0]
        if not isinstance(pair, (tuple, list)) or len(pair) < 2:
            return None
        k, v = pair[0], pair[1]
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            return None
        return k, v


    # TODO: simplify 
    def _can_use_draft_step_graph(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        past_key_values,
        logits_processor=None,
    ) -> bool:
        if os.getenv("AICAS_EAGLE3_DRAFT_STEP_CUDAGRAPH", "0") != "1":
            return False
        #     return False
        # if torch.is_grad_enabled() or self.training or not torch.cuda.is_available():
        #     return False
        # if not isinstance(hidden_states, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
        #     return False
        # if not hidden_states.is_cuda or not input_ids.is_cuda:
        #     return False
        # if hidden_states.ndim < 3 or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        #     return False
        # if int(input_ids.shape[1]) < 1:
        #     return False
        # if int(self.top_k) != 4:
        #     return False
        # pair = self._draft_step_graph_past_pair(past_key_values)
        # if pair is None:
        #     return False
        # past_k, past_v = pair
        # if not past_k.is_cuda or not past_v.is_cuda:
        #     return False
        # if isinstance(position_ids, torch.Tensor) and not position_ids.is_cuda:
        #     return False
        # tree_mask = getattr(self, "tree_mask", None)
        # if isinstance(tree_mask, torch.Tensor) and not tree_mask.is_cuda:
        #     return False
        return True

    def _draft_step_graph_key(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        past_key_values,
        *,
        topk_last_only: bool = False,
        expand_graph: bool = False,
        expand_scores: torch.Tensor | None = None,
        expand_score_slot: int = -1,
        expand_score_slots: int = -1,
        draft_tree_graph: bool = False,
        draft_tree_total_tokens: int = -1,
        draft_tree_depth: int = -1,
    ):
        pair = self._draft_step_graph_past_pair(past_key_values)
        if pair is None:
            return None
        past_k, past_v = pair
        past_bucket_len = self._draft_step_graph_bucket_len(int(past_k.shape[2]))
        tree_mask = getattr(self, "tree_mask", None)
        device_index = hidden_states.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return (
            int(device_index),
            tuple(hidden_states.shape),
            str(hidden_states.dtype),
            tuple(input_ids.shape),
            str(input_ids.dtype),
            tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor) else None,
            str(position_ids.dtype) if isinstance(position_ids, torch.Tensor) else None,
            (int(past_k.shape[0]), int(past_k.shape[1]), int(past_bucket_len), int(past_k.shape[3])),
            str(past_k.dtype),
            (int(past_v.shape[0]), int(past_v.shape[1]), int(past_bucket_len), int(past_v.shape[3])),
            str(past_v.dtype),
            tuple(tree_mask.shape) if isinstance(tree_mask, torch.Tensor) else None,
            str(tree_mask.dtype) if isinstance(tree_mask, torch.Tensor) else None,
            int(self.top_k),
            bool(topk_last_only),
            bool(expand_graph),
            tuple(expand_scores.shape) if isinstance(expand_scores, torch.Tensor) else None,
            str(expand_scores.dtype) if isinstance(expand_scores, torch.Tensor) else None,
            int(expand_score_slot),
            int(expand_score_slots),
            bool(draft_tree_graph),
            int(draft_tree_total_tokens) if draft_tree_graph else -1,
            int(draft_tree_depth) if draft_tree_graph else -1,
        )

    def _draft_step_eager(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values,
        *,
        position_ids: torch.Tensor | None = None,
        logits_processor=None,
        topk_last_only: bool = False,
    ):
        if position_ids is not None:
            _pids_el = int(position_ids.numel())
            _iids_batch = int(input_ids.shape[0])
            _iids_seq = int(input_ids.shape[1])
            if _pids_el != _iids_batch * _iids_seq:
                _pid_shape = tuple(position_ids.shape) if position_ids.dim() > 0 else _pids_el
                raise RuntimeError(
                    f"_draft_step_eager position_ids mismatch: "
                    f"position_ids_shape={_pid_shape} position_ids_numel={_pids_el} "
                    f"input_ids_shape={tuple(input_ids.shape)} "
                    f"batch_size={_iids_batch} seq_len={_iids_seq} "
                    f"hidden_shape={tuple(hidden_states.shape)} "
                    f"topk_last_only={topk_last_only} "
                    f"stable_kv_len={self._kv_seq_len(past_key_values[0] if isinstance(past_key_values, (tuple, list)) and len(past_key_values) == 1 else past_key_values) if past_key_values is not None else -1}"
                )
        old_force_eager_attention = getattr(
            self.midlayer.self_attn,
            "_aicas_draft_force_eager_attention",
            _EAGLE3_DEBUG_ATTR_MISSING,
        )
        force_eager_attention = os.getenv("AICAS_EAGLE3_DRAFT_FORCE_EAGER_ATTENTION", "0") == "1"
        try:
            if force_eager_attention:
                self.midlayer.self_attn._aicas_draft_force_eager_attention = True
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        finally:
            if force_eager_attention:
                if old_force_eager_attention is _EAGLE3_DEBUG_ATTR_MISSING:
                    try:
                        delattr(self.midlayer.self_attn, "_aicas_draft_force_eager_attention")
                    except Exception:
                        pass
                else:
                    self.midlayer.self_attn._aicas_draft_force_eager_attention = old_force_eager_attention
        topk_hidden = out_hidden[:, -1] if topk_last_only else out_hidden[0]
        topk_index, topk_p = self._lm_head_logp_topk(
            _eagle3_draft_rmsnorm(self.norm, topk_hidden),
            int(self.top_k),
            logits_processor,
        )
        return out_hidden, past_key_values, topk_index, topk_p

    def _draft_step_compiled(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values,
        *,
        position_ids: torch.Tensor | None = None,
        logits_processor=None,
        topk_last_only: bool = False,
    ):
        if self._draft_compiled_fn is None:
            _fn = torch.compile(
                self._draft_step_eager,
                mode="reduce-overhead",
                dynamic=True,
            )
            self._draft_compiled_fn = _fn
        return self._draft_compiled_fn(
            hidden_states,
            input_ids,
            past_key_values,
            position_ids=position_ids,
            logits_processor=logits_processor,
            topk_last_only=topk_last_only,
        )

    def _draft_step_graph(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        past_key_values,
        *,
        position_ids: torch.Tensor | None = None,
        logits_processor=None,
        topk_last_only: bool = False,
        expand_scores: torch.Tensor | None = None,
        expand_tree_mask: torch.Tensor | None = None,
        expand_score_slot: int = -1,
        expand_score_slots: int = -1,
        expand_topk_ids_slots: torch.Tensor | None = None,
        draft_tree_workspace: dict | None = None,
        draft_tree_total_tokens: int = -1,
        draft_tree_depth: int = -1,
    ):
        expand_graph = (
            os.getenv("AICAS_EAGLE3_DRAFT_EXTENDED_GRAPH", "0") == "1"
            and logits_processor is None
            and not bool(topk_last_only)
            and int(self.top_k) == 4
            and run_eagle3_expand_topk4 is not None
            and isinstance(expand_scores, torch.Tensor)
            and isinstance(expand_tree_mask, torch.Tensor)
        )
        draft_tree_graph = (
            expand_graph
            and run_eagle3_draft_tree_select_metadata is not None
            and isinstance(draft_tree_workspace, dict)
            and isinstance(expand_topk_ids_slots, torch.Tensor)
            and int(expand_score_slot) == int(draft_tree_depth) - 1
            and int(draft_tree_total_tokens) > 0
        )
        if not self._can_use_draft_step_graph(
            hidden_states,
            input_ids,
            position_ids,
            past_key_values,
            logits_processor=logits_processor,
        ):
            _graph_counter_inc("draft_step_graph_not_eligible")
            return self._draft_step_eager(
                hidden_states,
                input_ids,
                past_key_values,
                position_ids=position_ids,
                logits_processor=logits_processor,
                topk_last_only=topk_last_only,
            )
        key = self._draft_step_graph_key(
            hidden_states,
            input_ids,
            position_ids,
            past_key_values,
            topk_last_only=topk_last_only,
            expand_graph=bool(expand_graph),
            expand_scores=expand_scores,
            expand_score_slot=int(expand_score_slot) if expand_graph else -1,
            expand_score_slots=int(expand_score_slots) if expand_graph else -1,
            draft_tree_graph=bool(draft_tree_graph),
            draft_tree_total_tokens=int(draft_tree_total_tokens) if draft_tree_graph else -1,
            draft_tree_depth=int(draft_tree_depth) if draft_tree_graph else -1,
        )
        if key is None:
            _graph_counter_inc("draft_step_graph_key_none")
            _trace_once(
                f"eagle3_draft_step_graph_key_rejected_fallback",
                "[eagle3-draft] draft step graph key rejected, fallback to eager",
            )
            return self._draft_step_eager(
                hidden_states,
                input_ids,
                past_key_values,
                position_ids=position_ids,
                logits_processor=logits_processor,
                topk_last_only=topk_last_only,
            )
        store, order, failed_keys = self._draft_step_graph_state()
        failed_reasons = self._draft_step_graph_failed_reason_map()
        if key in failed_keys:
            _graph_counter_inc("draft_step_graph_failed_key")
            reason = failed_reasons.get(key, "unknown")
            _trace_once(
                f"eagle3_draft_step_graph_failed_key_fallback",
                f"[eagle3-draft] draft step graph key previously failed, fallback to eager, reason: {reason}",
            )
            return self._draft_step_eager(
                hidden_states,
                input_ids.detach().clone(),
                past_key_values,
                position_ids=position_ids,
                logits_processor=logits_processor,
                topk_last_only=topk_last_only,
        )
        pair = self._draft_step_graph_past_pair(past_key_values)
        if pair is None:
            _graph_counter_inc("draft_step_graph_no_past")
            _trace_once(
                f"eagle3_draft_step_graph_no_past_fallback",
                "[eagle3-draft] draft step graph missing past key/value, fallback to eager",
            )
            return self._draft_step_eager(
                hidden_states,
                input_ids.detach().clone(),
                past_key_values,
                position_ids=position_ids,
                logits_processor=logits_processor,
                topk_last_only=topk_last_only,
            )
        past_k, past_v = pair
        past_len = int(past_k.shape[2])
        past_bucket_len = self._draft_step_graph_bucket_len(past_len)
        q_len = int(input_ids.shape[1])
        position_buf = self._make_draft_position_ids(input_ids, past_len, position_ids)
        tree_mask = getattr(self, "tree_mask", None)
        entry = store.get(key)
        old_tree_mask = getattr(self, "tree_mask", None)
        old_force_eager_attention = getattr(self.midlayer.self_attn, "_aicas_draft_force_eager_attention", _EAGLE3_DEBUG_ATTR_MISSING)

        if entry is None:
            if not _eagle3_allow_ondemand_graph_capture():
                _graph_counter_inc("draft_step_graph_miss_after_precapture")
                _trace_once(
                    f"eagle3_draft_step_graph_miss_after_precapture",
                    "[eagle3-draft] draft step graph miss after precapture, fallback to eager",
                )
                return self._draft_step_eager(
                    hidden_states,
                    input_ids.detach().clone(),
                    past_key_values,
                    position_ids=position_ids,
                    logits_processor=logits_processor,
                    topk_last_only=topk_last_only,
                )
            try:
                hidden_buf = hidden_states.detach().clone()
                input_buf = input_ids.detach().clone()
                past_k_buf = past_k.new_empty((int(past_k.shape[0]), int(past_k.shape[1]), past_bucket_len, int(past_k.shape[3])))
                past_v_buf = past_v.new_empty((int(past_v.shape[0]), int(past_v.shape[1]), past_bucket_len, int(past_v.shape[3])))
                self._copy_draft_past_into_bucket(past_k_buf, past_v_buf, past_k, past_v)
                tree_mask_buf = tree_mask.detach().clone() if isinstance(tree_mask, torch.Tensor) else None
                static_past = ((past_k_buf, past_v_buf),)
                expand_scores_buf = expand_scores.detach().clone() if expand_graph else None
                expand_tree_mask_buf = expand_tree_mask.detach().clone() if expand_graph else None
                expand_topk_ids_slots_buf = expand_topk_ids_slots if expand_graph and isinstance(expand_topk_ids_slots, torch.Tensor) else None
                try:
                    def _run_graph_expand(_topk, _topk_p, _hidden):
                        if not expand_graph:
                            return None
                        result, workspace = run_eagle3_expand_topk4(
                            _topk,
                            _topk_p,
                            expand_scores_buf,
                            _hidden,
                            expand_tree_mask_buf,
                            workspace=self._expand_topk_workspace,
                            score_slot=int(expand_score_slot),
                            score_slots=int(expand_score_slots),
                        )
                        self._expand_topk_workspace = workspace
                        if result is None:
                            raise RuntimeError("draft extended graph expand_topk rejected inputs")
                        if isinstance(expand_topk_ids_slots_buf, torch.Tensor):
                            expand_topk_ids_slots_buf[int(expand_score_slot)].copy_(_topk)
                        return result

                    def _run_graph_tree_select():
                        if not draft_tree_graph:
                            return None
                        expand_workspace = self._expand_topk_workspace
                        if not isinstance(expand_workspace, dict):
                            raise RuntimeError("draft extended graph missing expand workspace")
                        result, tree_workspace = run_eagle3_draft_tree_select_metadata(
                            draft_tree_workspace["sample_token"],
                            draft_tree_workspace["root_tokens"],
                            draft_tree_workspace["root_scores"],
                            expand_workspace["cu_scores"],
                            expand_topk_ids_slots_buf,
                            expand_workspace["topk_cs_index_slots"],
                            total_tokens=int(draft_tree_total_tokens),
                            depth=int(draft_tree_depth),
                            max_depth=int(draft_tree_depth) + 2,
                            workspace=draft_tree_workspace.get("tree_select"),
                        )
                        draft_tree_workspace["tree_select"] = tree_workspace
                        if result is None:
                            raise RuntimeError("draft extended graph tree_select rejected inputs")
                        return result

                    logical_past_buf = torch.tensor(past_len, device=hidden_states.device, dtype=torch.long)
                    self.tree_mask = tree_mask_buf
                    self.midlayer.self_attn._aicas_draft_force_eager_attention = True
                    self._draft_graph_logical_past_len = logical_past_buf
                    self._draft_graph_storage_past_len = int(past_bucket_len)
                    # Pre-capture warmup: run eager twice so Triton autotuning
                    # caches every kernel config before graph capture begins.
                    _trace_once("recapture_warmup", "[eagle3-draft] draft step graph warmup")
                    with _draft_graph_attention_mode():
                        with decode_qk_rotary_graph_linear_mode(True):
                            # First pass: warm caches.
                            warm_hidden, warm_past, warm_topk, warm_topk_p = self._draft_step_eager(
                                hidden_buf, input_buf, static_past,
                                position_ids=position_buf,
                                logits_processor=logits_processor,
                                topk_last_only=topk_last_only,
                            )
                            _run_graph_expand(warm_topk, warm_topk_p, warm_hidden)
                            _run_graph_tree_select()
                    warm_pair = self._draft_step_graph_past_pair(warm_past)
                    if warm_pair is None:
                        raise RuntimeError("draft step graph warmup produced invalid past")
                    out_hidden_buf = torch.empty_like(warm_hidden)
                    out_k_buf = torch.empty_like(warm_pair[0])
                    out_v_buf = torch.empty_like(warm_pair[1])
                    out_topk_buf = torch.empty_like(warm_topk)
                    out_topk_p_buf = torch.empty_like(warm_topk_p)

                    # Second pass: run with the output buffers pre-allocated.
                    with _draft_graph_attention_mode():
                        with decode_qk_rotary_graph_linear_mode(True):
                            _h, _p, _t, _tp = self._draft_step_eager(
                                hidden_buf, input_buf, static_past,
                                position_ids=position_buf,
                                logits_processor=logits_processor,
                                topk_last_only=topk_last_only,
                            )
                    _run_graph_expand(_t, _tp, _h)
                    graph_tree = _run_graph_tree_select()
                    out_hidden_buf.copy_(_h)
                    _sp = self._draft_step_graph_past_pair(_p)
                    if _sp is None:
                        raise RuntimeError("draft step graph second warmup produced invalid past")
                    out_k_buf.copy_(_sp[0])
                    out_v_buf.copy_(_sp[1])
                    out_topk_buf.copy_(_t)
                    out_topk_p_buf.copy_(_tp)
                    import warnings
                    graph = torch.cuda.CUDAGraph()
                    graph_expand = None
                    graph_empty = False
                    with torch.no_grad():
                        with _draft_graph_attention_mode():
                            with decode_qk_rotary_graph_linear_mode(True):
                                with warnings.catch_warnings(record=True) as _w:
                                    with torch.cuda.graph(graph):
                                        graph_hidden, graph_past, graph_topk, graph_topk_p = self._draft_step_eager(
                                            hidden_buf, input_buf, static_past,
                                            position_ids=position_buf,
                                            logits_processor=logits_processor,
                                            topk_last_only=topk_last_only,
                                        )
                                        graph_pair = self._draft_step_graph_past_pair(graph_past)
                                        if graph_pair is None:
                                            raise RuntimeError("draft step graph capture produced invalid past")
                                        out_hidden_buf.copy_(graph_hidden)
                                        out_k_buf.copy_(graph_pair[0])
                                        out_v_buf.copy_(graph_pair[1])
                                        out_topk_buf.copy_(graph_topk)
                                        out_topk_p_buf.copy_(graph_topk_p)
                                        graph_expand = _run_graph_expand(graph_topk, graph_topk_p, graph_hidden)
                                        graph_tree = _run_graph_tree_select()
                            for _warning in _w:
                                if "CUDA Graph is empty" in str(_warning.message):
                                    graph_empty = True
                                    break
                    if graph_empty:
                        raise RuntimeError("draft step graph capture is empty — Triton autotuning likely fired during capture")
                finally:
                    self.tree_mask = old_tree_mask
                    if old_force_eager_attention is _EAGLE3_DEBUG_ATTR_MISSING:
                        try:
                            delattr(self.midlayer.self_attn, "_aicas_draft_force_eager_attention")
                        except Exception:
                            pass
                    else:
                        self.midlayer.self_attn._aicas_draft_force_eager_attention = old_force_eager_attention
                    self._draft_graph_logical_past_len = None
                    self._draft_graph_storage_past_len = None

                entry = {
                    "graph": graph,
                    "hidden": hidden_buf,
                    "input_ids": input_buf,
                    "position_ids": position_buf,
                    "past_k": past_k_buf,
                    "past_v": past_v_buf,
                    "logical_past_len": logical_past_buf,
                    "past_bucket_len": int(past_bucket_len),
                    "tree_mask": tree_mask_buf,
                    "expand_scores": expand_scores_buf,
                    "expand_tree_mask": expand_tree_mask_buf,
                    "expand_topk_ids_slots": expand_topk_ids_slots_buf,
                    "out_hidden": out_hidden_buf,
                    "out_k": out_k_buf,
                    "out_v": out_v_buf,
                    "topk_ids": out_topk_buf,
                    "topk_vals": out_topk_p_buf,
                    "expand_result": graph_expand if expand_graph else None,
                    "tree_result": graph_tree if draft_tree_graph else None,
                }
                self._put_lru_entry(
                    store,
                    order,
                    key,
                    entry,
                    _draft_step_graph_lru_capacity(),
                )
                _graph_counter_inc("draft_step_graph_capture")
                if expand_graph:
                    _graph_counter_inc("draft_step_extended_graph_capture")
                if draft_tree_graph:
                    _graph_counter_inc("draft_step_tree_graph_capture")
                _trace_once(
                    f"eagle3_draft_step_graph_captured_{tuple(hidden_states.shape)}_{tuple(input_ids.shape)}_{int(past_bucket_len)}_{bool(topk_last_only)}",
                    f"[eagle3] draft step graph captured: hidden={tuple(hidden_states.shape)}",
                )
            except Exception as exc:
                _graph_counter_inc("draft_step_graph_capture_failed")
                store.pop(key, None)
                failed_keys.add(key)
                failed_reasons[key] = (
                    f"{type(exc).__name__}: {exc} "
                    f"hidden={tuple(hidden_states.shape)} input={tuple(input_ids.shape)} "
                    f"past={tuple(past_k.shape)} bucket={int(past_bucket_len)} "
                    f"position={tuple(position_buf.shape) if isinstance(position_buf, torch.Tensor) else None} "
                    f"tree_mask={tuple(tree_mask.shape) if isinstance(tree_mask, torch.Tensor) else None} "
                    f"topk_last_only={bool(topk_last_only)} "
                    f"expand_graph={bool(expand_graph)} draft_tree_graph={bool(draft_tree_graph)} "
                    f"tree_total={int(draft_tree_total_tokens)} tree_depth={int(draft_tree_depth)}\n{traceback.format_exc()}"
                )
                try:
                    torch.cuda.synchronize(hidden_states.device)
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                _trace_once(
                    f"eagle3_draft_step_graph_capture_failed",
                    f"[eagle3-draft] draft step graph capture failed: {failed_reasons[key]}",
                )
                return self._draft_step_eager(
                    hidden_states,
                    input_ids.detach().clone(),
                    past_key_values,
                    position_ids=position_ids,
                    logits_processor=logits_processor,
                    topk_last_only=topk_last_only,
                )

        entry["hidden"].copy_(hidden_states)
        entry["input_ids"].copy_(input_ids)
        if isinstance(entry.get("position_ids"), torch.Tensor):
            entry["position_ids"].copy_(position_buf)
        self._copy_draft_past_into_bucket(entry["past_k"], entry["past_v"], past_k, past_v)
        if isinstance(tree_mask, torch.Tensor) and isinstance(entry.get("tree_mask"), torch.Tensor):
            entry["tree_mask"].copy_(tree_mask)
        if expand_graph and isinstance(expand_scores, torch.Tensor) and isinstance(entry.get("expand_scores"), torch.Tensor):
            entry["expand_scores"].copy_(expand_scores)
        if expand_graph and isinstance(expand_tree_mask, torch.Tensor) and isinstance(entry.get("expand_tree_mask"), torch.Tensor):
            entry["expand_tree_mask"].copy_(expand_tree_mask)
        entry["logical_past_len"].fill_(int(past_len))

        try:
            self.tree_mask = entry.get("tree_mask")
            self.midlayer.self_attn._aicas_draft_force_eager_attention = True
            self._draft_graph_logical_past_len = entry["logical_past_len"]
            self._draft_graph_storage_past_len = int(entry["past_bucket_len"])
            _graph_counter_inc("draft_step_graph_hit")
            if expand_graph and entry.get("expand_result") is not None:
                _graph_counter_inc("draft_step_extended_graph_hit")
            if draft_tree_graph and entry.get("tree_result") is not None:
                _graph_counter_inc("draft_step_tree_graph_hit")
            entry["graph"].replay()
        except Exception as exc:
            _graph_counter_inc("draft_step_graph_replay_fail")
            store.pop(key, None)
            failed_keys.add(key)
            failed_reasons[key] = (
                f"{type(exc).__name__}: {exc} "
                f"hidden={tuple(hidden_states.shape)} input={tuple(input_ids.shape)} "
                f"past={tuple(past_k.shape)} bucket={int(past_bucket_len)} "
                f"position={tuple(position_buf.shape) if isinstance(position_buf, torch.Tensor) else None} "
                f"tree_mask={tuple(tree_mask.shape) if isinstance(tree_mask, torch.Tensor) else None} "
                f"topk_last_only={bool(topk_last_only)} "
                f"expand_graph={bool(expand_graph)} draft_tree_graph={bool(draft_tree_graph)} "
                f"tree_total={int(draft_tree_total_tokens)} tree_depth={int(draft_tree_depth)}\n{traceback.format_exc()}"
            )
            _trace_once(
                f"eagle3_draft_step_graph_replay_failed",
                f"[eagle3-draft] draft step graph replay failed: {failed_reasons[key]}",
            )
            return self._draft_step_eager(
                hidden_states,
                input_ids.detach().clone(),
                past_key_values,
                position_ids=position_ids,
                logits_processor=logits_processor,
                topk_last_only=topk_last_only,
            )
        finally:
            self.tree_mask = old_tree_mask
            if old_force_eager_attention is _EAGLE3_DEBUG_ATTR_MISSING:
                try:
                    delattr(self.midlayer.self_attn, "_aicas_draft_force_eager_attention")
                except Exception:
                    pass
            else:
                self.midlayer.self_attn._aicas_draft_force_eager_attention = old_force_eager_attention
            self._draft_graph_logical_past_len = None
            self._draft_graph_storage_past_len = None
        ret_k, ret_v = self._repack_draft_graph_past(entry, int(past_len), q_len)
        base_result = (
            entry["out_hidden"].clone(),
            ((ret_k.clone(), ret_v.clone()),),
            entry["topk_ids"].clone(),
            entry["topk_vals"].clone(),
        )
        if expand_graph and entry.get("expand_result") is not None:
            return (*base_result, entry["expand_result"], entry.get("tree_result"))
        return base_result

    def training_forward(
        self,
        feature: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        length: int,
    ) -> list[torch.Tensor]:
        hidden_states = self.project_feature(feature)
        input_ids = input_ids.to(hidden_states.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(hidden_states.device)
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
        attn_mask = self._sequence_attention_mask(hidden_states, attention_mask)
        cache_hidden: list[list[torch.Tensor]] = [[], []]
        logits_list: list[torch.Tensor] = []
        cur_input_ids = input_ids
        for idx in range(int(length)):
            inputs_embeds = self.embed_tokens(cur_input_ids).to(dtype=hidden_states.dtype)
            mrope_position_ids = (position_ids + idx).unsqueeze(0).expand(3, -1, -1)
            position_embeddings = self.rotary_emb(hidden_states, mrope_position_ids)
            hidden_states, cache_hidden = self.midlayer.forward_train_cache(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )
            norm_hidden = self.norm(hidden_states)
            logits_list.append(self.lm_head(norm_hidden.to(dtype=self.lm_head.weight.dtype)))
            if idx != int(length) - 1:
                cur_input_ids = shift_left(cur_input_ids)
        return logits_list

    def forward_step(self, hidden_states: torch.Tensor, input_ids: torch.Tensor):
        past_len = self._kv_seq_len(self._stable_kv)
        cache_position = torch.arange(
            past_len,
            past_len + int(input_ids.shape[1]),
            device=input_ids.device,
            dtype=torch.long,
        )
        return self.forward_block(
            hidden_states,
            input_ids,
            past_key_value=self._stable_kv,
            cache_position=cache_position,
            use_cache=True,
        )

    @torch.no_grad()
    def topK_genrate(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, logits_processor=None):
        if not hasattr(self, "tree_mask_init"):
            self.init_tree()
        return self._topK_genrate_impl(hidden_states, input_ids, logits_processor=logits_processor)

    def _topK_genrate_impl_draft_all_graph(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        logits_processor=None,
    ):
        if (
            logits_processor is not None
            or torch.is_grad_enabled()
            or self.training
            or int(self.top_k) != 4
            or run_eagle3_expand_topk4 is None
            or run_eagle3_draft_tree_select_metadata is None
        ):
            _graph_counter_inc("draft_all_graph_not_eligible")
            return None
        if not isinstance(hidden_states, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
            _graph_counter_inc("draft_all_graph_not_eligible")
            return None
        if not hidden_states.is_cuda or not input_ids.is_cuda:
            _graph_counter_inc("draft_all_graph_not_eligible")
            return None
        stable_before = self._stable_kv
        if stable_before is None:
            _graph_counter_inc("draft_all_graph_not_eligible")
            return None

        input_ids_full = input_ids.to(hidden_states.device, dtype=torch.long).contiguous()
        if input_ids_full.ndim != 2 or int(input_ids_full.shape[0]) != 1 or int(input_ids_full.shape[1]) < 2:
            _graph_counter_inc("draft_all_graph_not_eligible")
            return None

        total_tokens = int(self.total_tokens)
        depth = int(self.depth)
        top_k = int(self.top_k)
        if depth < 1 or total_tokens < 1:
            _graph_counter_inc("draft_all_graph_not_eligible")
            return None

        sample_token = input_ids_full[:, -1]
        shifted_input_ids = input_ids_full[:, 1:].contiguous()
        hidden_len = int(hidden_states.shape[1]) if hidden_states.dim() >= 3 else 1
        kv_len = self._kv_seq_len(stable_before[0] if isinstance(stable_before, tuple) else stable_before)
        suffix_len = max(0, int(shifted_input_ids.shape[1]) - int(kv_len))
        if suffix_len != hidden_len:
            _graph_counter_inc("draft_all_graph_not_eligible")
            return None
        prefill_input_ids = shifted_input_ids[:, kv_len:].contiguous()

        pair = self._draft_step_graph_past_pair(stable_before)
        if pair is None:
            _graph_counter_inc("draft_all_graph_no_past")
            return None
        past_k, past_v = pair
        past_len = int(past_k.shape[2])
        past_bucket_len = self._draft_step_graph_bucket_len(past_len)
        position_ids = self._make_draft_position_ids(prefill_input_ids, past_len, None)
        device = hidden_states.device
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        key = (
            int(device_index),
            tuple(hidden_states.shape),
            str(hidden_states.dtype),
            tuple(prefill_input_ids.shape),
            str(prefill_input_ids.dtype),
            (int(past_k.shape[0]), int(past_k.shape[1]), int(past_bucket_len), int(past_k.shape[3])),
            str(past_k.dtype),
            int(depth),
            int(top_k),
            int(total_tokens),
            int(shifted_input_ids.shape[1]),
        )
        store, order, failed_keys = self._draft_all_graph_state()
        failed_reasons = self._draft_all_graph_failed_reason_map()
        if key in failed_keys:
            _graph_counter_inc("draft_all_graph_failed_key")
            _trace_once(
                f"eagle3_draft_all_graph_failed_key_{abs(hash(key))}",
                f"[eagle3-draft] draft_all graph key previously failed, fallback to split draft graph: "
                f"{failed_reasons.get(key, 'unknown')}",
            )
            return None

        entry = store.get(key)
        if entry is None:
            if not _eagle3_allow_ondemand_graph_capture():
                _graph_counter_inc("draft_all_graph_miss_after_precapture")
                return None
            try:
                hidden_buf = hidden_states.detach().clone()
                input_buf = prefill_input_ids.detach().clone()
                sample_token_buf = sample_token.reshape(-1).detach().clone()
                position_buf = position_ids.detach().clone()
                past_k_buf = past_k.new_empty((int(past_k.shape[0]), int(past_k.shape[1]), past_bucket_len, int(past_k.shape[3])))
                past_v_buf = past_v.new_empty((int(past_v.shape[0]), int(past_v.shape[1]), past_bucket_len, int(past_v.shape[3])))
                self._copy_draft_past_into_bucket(past_k_buf, past_v_buf, past_k, past_v)
                static_past = ((past_k_buf, past_v_buf),)
                tree_mask_init_buf = self.tree_mask_init.detach().clone()
                logical_past_bufs = []
                logical_offset = int(prefill_input_ids.shape[1])
                logical_past_bufs.append(torch.tensor(past_len, device=device, dtype=torch.long))
                for _ in range(depth):
                    logical_past_bufs.append(torch.tensor(past_len + logical_offset, device=device, dtype=torch.long))
                    logical_offset += int(top_k)
                draft_tree_ws = {
                    "sample_token": torch.empty((1,), device=device, dtype=torch.long),
                    "root_tokens": torch.empty((1, top_k), device=device, dtype=torch.long),
                    "root_scores": torch.empty((top_k,), device=device, dtype=torch.float32),
                    "topk_ids_slots": torch.empty((depth, top_k, top_k), device=device, dtype=torch.long),
                    "tree_select": None,
                }
                expand_holder = {"workspace": None}
                stable_kv = None

                def _run_all_draft():
                    nonlocal stable_kv
                    self.tree_mask = None
                    self._draft_graph_logical_past_len = logical_past_bufs[0]
                    self._draft_graph_storage_past_len = int(past_bucket_len)
                    out_hidden, past_key_values, topk_index, topk_p = self._draft_step_eager(
                        hidden_buf,
                        input_buf,
                        static_past,
                        position_ids=position_buf,
                        logits_processor=logits_processor,
                        topk_last_only=True,
                    )
                    stable_kv = past_key_values
                    root_scores = topk_p[0]
                    draft_tree_ws["sample_token"].copy_(sample_token_buf)
                    draft_tree_ws["root_tokens"].copy_(topk_index.reshape(1, top_k))
                    draft_tree_ws["root_scores"].copy_(root_scores.reshape(top_k))

                    last_hidden = out_hidden[:, -1]
                    scores = root_scores
                    input_hidden = last_hidden[None].repeat(1, top_k, 1)
                    step_input_ids = topk_index
                    tree_mask = tree_mask_init_buf
                    storage_past_len = int(past_bucket_len) + int(prefill_input_ids.shape[1])
                    for step in range(depth):
                        self.tree_mask = tree_mask
                        self._draft_graph_logical_past_len = logical_past_bufs[step + 1]
                        self._draft_graph_storage_past_len = int(storage_past_len)
                        step_position_ids = int(shifted_input_ids.shape[1]) + step + self.position_ids
                        step_hidden, past_key_values, step_topk, step_topk_p = self._draft_step_eager(
                            input_hidden,
                            step_input_ids,
                            past_key_values,
                            position_ids=step_position_ids,
                            logits_processor=logits_processor,
                            topk_last_only=False,
                        )
                        result, workspace = run_eagle3_expand_topk4(
                            step_topk,
                            step_topk_p,
                            scores,
                            step_hidden,
                            tree_mask,
                            workspace=expand_holder["workspace"],
                            score_slot=step,
                            score_slots=depth,
                        )
                        expand_holder["workspace"] = workspace
                        if result is None:
                            raise RuntimeError("draft_all expand_topk rejected inputs")
                        storage_past_len += int(step_input_ids.shape[1])
                        draft_tree_ws["topk_ids_slots"][step].copy_(step_topk)
                        _cu_scores, _topk_cs_index, scores, _out_ids, input_hidden, step_input_ids, tree_mask = result

                    expand_workspace = expand_holder["workspace"]
                    if not isinstance(expand_workspace, dict):
                        raise RuntimeError("draft_all missing expand workspace")
                    tree_result, tree_workspace = run_eagle3_draft_tree_select_metadata(
                        draft_tree_ws["sample_token"],
                        draft_tree_ws["root_tokens"],
                        draft_tree_ws["root_scores"],
                        expand_workspace["cu_scores"],
                        draft_tree_ws["topk_ids_slots"],
                        expand_workspace["topk_cs_index_slots"],
                        total_tokens=int(total_tokens),
                        depth=int(depth),
                        max_depth=int(depth) + 2,
                        workspace=draft_tree_ws.get("tree_select"),
                    )
                    draft_tree_ws["tree_select"] = tree_workspace
                    if tree_result is None:
                        raise RuntimeError("draft_all tree_select rejected inputs")
                    return tree_result

                old_tree_mask = getattr(self, "tree_mask", None)
                old_force_eager_attention = getattr(self.midlayer.self_attn, "_aicas_draft_force_eager_attention", _EAGLE3_DEBUG_ATTR_MISSING)
                graph_result = None
                try:
                    self.midlayer.self_attn._aicas_draft_force_eager_attention = True
                    self._draft_graph_logical_past_len = logical_past_bufs[0]
                    self._draft_graph_storage_past_len = int(past_bucket_len)
                    with _draft_graph_attention_mode():
                        with decode_qk_rotary_graph_linear_mode(True):
                            _run_all_draft()
                            _run_all_draft()
                    import warnings

                    graph = torch.cuda.CUDAGraph()
                    graph_empty = False
                    with torch.no_grad():
                        with _draft_graph_attention_mode():
                            with decode_qk_rotary_graph_linear_mode(True):
                                with warnings.catch_warnings(record=True) as _w:
                                    with torch.cuda.graph(graph):
                                        graph_result = _run_all_draft()
                                for _warning in _w:
                                    if "CUDA Graph is empty" in str(_warning.message):
                                        graph_empty = True
                                        break
                    if graph_empty:
                        raise RuntimeError("draft_all graph capture is empty")
                finally:
                    self.tree_mask = old_tree_mask
                    if old_force_eager_attention is _EAGLE3_DEBUG_ATTR_MISSING:
                        try:
                            delattr(self.midlayer.self_attn, "_aicas_draft_force_eager_attention")
                        except Exception:
                            pass
                    else:
                        self.midlayer.self_attn._aicas_draft_force_eager_attention = old_force_eager_attention
                    self._draft_graph_logical_past_len = None
                    self._draft_graph_storage_past_len = None

                if graph_result is None:
                    raise RuntimeError("draft_all graph produced no result")
                entry = {
                    "graph": graph,
                    "hidden": hidden_buf,
                    "input_ids": input_buf,
                    "sample_token": sample_token_buf,
                    "position_ids": position_buf,
                    "past_k": past_k_buf,
                    "past_v": past_v_buf,
                    "stable_kv_src": stable_kv,
                    "logical_past_len": logical_past_bufs[0],
                    "logical_past_lens": logical_past_bufs,
                    "past_bucket_len": int(past_bucket_len),
                    "tree_mask_init": tree_mask_init_buf,
                    "draft_tree_workspace": draft_tree_ws,
                    "expand_workspace": expand_holder["workspace"],
                    "result": graph_result,
                }
                self._put_lru_entry(store, order, key, entry, _draft_all_graph_lru_capacity())
                _graph_counter_inc("draft_all_graph_capture")
                _trace_once(
                    f"eagle3_draft_all_graph_captured_{abs(hash(key))}",
                    f"[eagle3-draft] draft_all graph captured: hidden={tuple(hidden_states.shape)} "
                    f"input={tuple(prefill_input_ids.shape)} bucket={int(past_bucket_len)} depth={int(depth)}",
                )
            except Exception as exc:
                _graph_counter_inc("draft_all_graph_capture_failed")
                store.pop(key, None)
                failed_keys.add(key)
                failed_reasons[key] = (
                    f"{type(exc).__name__}: {exc} hidden={tuple(hidden_states.shape)} "
                    f"input={tuple(prefill_input_ids.shape)} past={tuple(past_k.shape)} "
                    f"bucket={int(past_bucket_len)} depth={int(depth)} top_k={int(top_k)} "
                    f"total={int(total_tokens)}\n{traceback.format_exc()}"
                )
                _trace_once(
                    f"eagle3_draft_all_graph_capture_failed_{abs(hash(key))}",
                    f"[eagle3-draft] draft_all graph capture failed, fallback to split draft graph: {failed_reasons[key]}",
                )
                return None

        entry["hidden"].copy_(hidden_states)
        entry["input_ids"].copy_(prefill_input_ids)
        entry["sample_token"].copy_(sample_token.reshape(-1))
        entry["position_ids"].copy_(position_ids)
        self._copy_draft_past_into_bucket(entry["past_k"], entry["past_v"], past_k, past_v)
        entry["logical_past_len"].fill_(int(past_len))
        logical_past_lens = entry.get("logical_past_lens")
        if isinstance(logical_past_lens, list):
            logical_offset = int(prefill_input_ids.shape[1])
            for idx, buf in enumerate(logical_past_lens[1:]):
                if isinstance(buf, torch.Tensor):
                    buf.fill_(int(past_len) + int(logical_offset))
                logical_offset += int(self.top_k)
        old_tree_mask = getattr(self, "tree_mask", None)
        old_force_eager_attention = getattr(self.midlayer.self_attn, "_aicas_draft_force_eager_attention", _EAGLE3_DEBUG_ATTR_MISSING)
        try:
            self.midlayer.self_attn._aicas_draft_force_eager_attention = True
            self._draft_graph_logical_past_len = entry["logical_past_len"]
            self._draft_graph_storage_past_len = int(entry["past_bucket_len"])
            _graph_counter_inc("draft_all_graph_hit")
            entry["graph"].replay()
        except Exception as exc:
            _graph_counter_inc("draft_all_graph_replay_fail")
            store.pop(key, None)
            failed_keys.add(key)
            failed_reasons[key] = (
                f"{type(exc).__name__}: {exc} hidden={tuple(hidden_states.shape)} "
                f"input={tuple(prefill_input_ids.shape)} past={tuple(past_k.shape)} "
                f"bucket={int(past_bucket_len)} depth={int(depth)} top_k={int(top_k)} "
                f"total={int(total_tokens)}\n{traceback.format_exc()}"
            )
            _trace_once(
                f"eagle3_draft_all_graph_replay_failed_{abs(hash(key))}",
                f"[eagle3-draft] draft_all graph replay failed, fallback to split draft graph: {failed_reasons[key]}",
            )
            return None
        finally:
            self.tree_mask = old_tree_mask
            if old_force_eager_attention is _EAGLE3_DEBUG_ATTR_MISSING:
                try:
                    delattr(self.midlayer.self_attn, "_aicas_draft_force_eager_attention")
                except Exception:
                    pass
            else:
                self.midlayer.self_attn._aicas_draft_force_eager_attention = old_force_eager_attention
            self._draft_graph_logical_past_len = None
            self._draft_graph_storage_past_len = None

        stable_pair = self._draft_step_graph_past_pair(entry.get("stable_kv_src"))
        if stable_pair is None:
            self._stable_kv = stable_before
        else:
            ret_k, ret_v = self._repack_draft_graph_past(
                {
                    "past_bucket_len": entry["past_bucket_len"],
                    "past_k": entry["past_k"],
                    "past_v": entry["past_v"],
                    "out_k": stable_pair[0],
                    "out_v": stable_pair[1],
                },
                int(past_len),
                int(prefill_input_ids.shape[1]),
            )
            self._stable_kv = ((ret_k, ret_v),)
        return entry["result"]

    @torch.no_grad()
    def _topK_genrate_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        logits_processor=None,
    ):
        if not hasattr(self, "tree_mask_init"):
            self.init_tree()

        input_ids = input_ids.to(hidden_states.device, dtype=torch.long).contiguous()

        total_tokens = int(self.total_tokens)
        depth = int(self.depth)
        top_k = int(self.top_k)

        sample_token = input_ids[:, -1]
        input_ids = input_ids[:, 1:].contiguous()
        len_posi = int(input_ids.shape[1])
        self.tree_mask = None

        hidden_len = int(hidden_states.shape[1]) if hidden_states.dim() >= 3 else 1
        kv_len = 0
        suffix_len = int(input_ids.shape[1])
        stable_before = self._stable_kv
        if stable_before is not None:
            kv_len = self._kv_seq_len(stable_before[0] if isinstance(stable_before, tuple) else stable_before)
            suffix_len = max(0, int(input_ids.shape[1]) - int(kv_len))
            if suffix_len != hidden_len:
                raise RuntimeError(
                    "EAGLE3 draft KV/input/feature length mismatch: "
                    f"stable_kv_len={kv_len} shifted_input_len={int(input_ids.shape[1])} "
                    f"suffix_len={suffix_len} hidden_len={hidden_len}"
                )
            with _profile_range("eagle3_draft.prefill_forward"):
                with _stable_prefill_kernel_context():
                    out_hidden, past_key_values, topk_index, topk_p = self._draft_step_graph(
                        hidden_states,
                        input_ids=input_ids[:, kv_len:],
                        past_key_values=self._stable_kv,
                        logits_processor=logits_processor,
                        topk_last_only=True,
                    )
        else:
            with _profile_range("eagle3_draft.prefill_forward"):
                out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
                last_hidden_for_topk = out_hidden[:, -1]
                topk_index, topk_p = self._lm_head_logp_topk(
                    _eagle3_draft_rmsnorm(self.norm, last_hidden_for_topk),
                    top_k,
                    logits_processor,
                )

        self._stable_kv = past_key_values

        last_hidden = out_hidden[:, -1]
        scores = topk_p[0]
        scores_list = [scores[None]]
        parents_list = [self._zero_parent]
        sampled_tokens = [topk_index]
        input_ids = topk_index
        draft_tree_ws = None
        if (
            os.getenv("AICAS_EAGLE3_DRAFT_EXTENDED_GRAPH", "0") == "1"
            and logits_processor is None
            and top_k == 4
            and run_eagle3_draft_tree_select_metadata is not None
        ):
            draft_tree_ws = self._draft_tree_graph_workspace(
                device=hidden_states.device,
                depth=depth,
                total_tokens=total_tokens,
            )
            draft_tree_ws["sample_token"].copy_(sample_token.reshape(-1))
            draft_tree_ws["root_tokens"].copy_(topk_index.reshape(1, 4))
            draft_tree_ws["root_scores"].copy_(scores.reshape(4))
        final_tree_result = None

        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = self._topk_cs_index

        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            with _profile_range("eagle3_draft.expand_forward"):
                _draft_step_ret = self._draft_step_graph(
                    input_hidden,
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    logits_processor=logits_processor,
                    expand_scores=scores,
                    expand_tree_mask=tree_mask,
                    expand_score_slot=i,
                    expand_score_slots=depth,
                    expand_topk_ids_slots=draft_tree_ws.get("topk_ids_slots") if isinstance(draft_tree_ws, dict) else None,
                    draft_tree_workspace=draft_tree_ws if i == depth - 1 else None,
                    draft_tree_total_tokens=total_tokens,
                    draft_tree_depth=depth,
                )
                if isinstance(_draft_step_ret, tuple) and len(_draft_step_ret) == 6:
                    out_hidden, past_key_values, topk_index, topk_p, _expand_result, final_tree_result = _draft_step_ret
                elif isinstance(_draft_step_ret, tuple) and len(_draft_step_ret) == 5:
                    out_hidden, past_key_values, topk_index, topk_p, _expand_result = _draft_step_ret
                else:
                    out_hidden, past_key_values, topk_index, topk_p = _draft_step_ret
                    _expand_result = None

            len_posi += 1
            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k ** 2 * bias2 + bias1
            if _expand_result is None:
                with _profile_range("eagle3_draft.expand_topk"):
                    cu_scores, topk_cs_index, scores, out_ids, input_hidden, input_ids, tree_mask = self._expand_topk_select(
                        topk_index,
                        topk_p,
                        scores,
                        out_hidden,
                        tree_mask,
                        top_k,
                        score_slot=i,
                        score_slots=depth,
                    )
            else:
                cu_scores, topk_cs_index, scores, out_ids, input_hidden, input_ids, tree_mask = _expand_result
            if _expand_result is None and isinstance(draft_tree_ws, dict):
                draft_tree_ws["topk_ids_slots"][i].copy_(topk_index)
            parents = topk_cs_index + bias
            parents_list.append(parents)
            sampled_tokens.append(topk_index)
            scores_list.append(cu_scores)

        if final_tree_result is not None:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = final_tree_result
        else:
            with _profile_range("eagle3_draft.tree_select"):
                scores_list = torch.cat(scores_list, dim=0).view(-1)
                sampled_token_list = torch.cat(sampled_tokens, dim=0).view(-1)
                total_tokens = min(total_tokens, int(scores_list.numel()))
                top_scores_index = torch.topk(scores_list, total_tokens, dim=-1).indices
                top_scores_index = torch.sort(top_scores_index).values

                draft_tokens = sampled_token_list[top_scores_index]
                draft_tokens = torch.cat((sample_token.reshape(-1), draft_tokens), dim=0)

                draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
                mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
                mask_index[draft_parents == 0] = -1
                mask_index = mask_index + 1
            with _profile_range("eagle3_draft.tree_metadata"):
                retrieve_indices, tree_mask, tree_position_ids = self._tree_metadata(
                    mask_index,
                    total_tokens,
                    logits_processor=logits_processor,
                )
                draft_tokens = draft_tokens[None]

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @torch.no_grad()
    def draft_tensor(self, feature: torch.Tensor, first_token) -> torch.Tensor:
        self.reset_kv()
        hidden = self.project_feature(feature[:, -1:, :] if feature.dim() == 3 else feature)
        if isinstance(first_token, torch.Tensor):
            prev = first_token.reshape(1, 1).to(device=hidden.device, dtype=torch.long)
        else:
            prev = torch.tensor([[int(first_token)]], device=hidden.device, dtype=torch.long)
        out = torch.empty((self.num_heads,), device=hidden.device, dtype=torch.long)
        for idx in range(self.num_heads):
            hidden, self._stable_kv = self.forward_step(hidden, prev)
            logits = self.lm_head(self.norm(hidden)[:, -1, :].to(dtype=self.lm_head.weight.dtype))
            prev = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)[0]
            # Map from draft index space (0..draft_vocab_size-1) to target vocabulary space
            if self.draft_vocab_ids.numel() > 0:
                prev = self.draft_vocab_ids.to(device=prev.device)[prev]
            out[idx] = prev
            prev = prev.reshape(1, 1)
        return out


def save_eagle3_draft(model: Eagle3DraftModel, path: str, train_length: int, layer_indices: tuple[int, int, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "draft": model.state_dict(),
        "train_length": int(train_length),
        "hidden_size": int(model.hidden_size),
        "vocab_size": int(model.vocab_size),
        "draft_vocab_size": int(model.draft_vocab_size),
        "layer_indices": tuple(int(x) for x in layer_indices),
        "format": Eagle3DraftModel.checkpoint_format,
    }
    if model.draft_vocab_ids.numel() > 0:
        payload["draft_vocab_ids"] = model.draft_vocab_ids.detach().cpu().to(dtype=torch.long)
    torch.save(payload, path)


def load_eagle3_draft(path: str, target_model, draft_len: int, device: str | torch.device) -> Eagle3DraftModel:
    text_config = target_model.config.text_config
    text_model = getattr(getattr(target_model, "model", None), "language_model", None)
    embed_tokens = getattr(text_model, "embed_tokens", None)
    embed_weight = getattr(embed_tokens, "weight", None)
    state = torch.load(path, map_location=device, weights_only=True)
    fmt = state.get("format")
    if fmt != Eagle3DraftModel.checkpoint_format and fmt not in Eagle3DraftModel.legacy_checkpoint_formats:
        raise RuntimeError(
            f"invalid EAGLE3 draft checkpoint format: {fmt!r}; "
            f"expected {Eagle3DraftModel.checkpoint_format!r}"
        )
    if int(state.get("hidden_size", text_config.hidden_size)) != int(text_config.hidden_size):
        raise RuntimeError("EAGLE3 hidden_size mismatch")
    if int(state.get("vocab_size", text_config.vocab_size)) != int(text_config.vocab_size):
        raise RuntimeError("EAGLE3 vocab_size mismatch")
    draft_vocab_ids = state.get("draft_vocab_ids")
    draft = Eagle3DraftModel(
        text_config,
        draft_len - 1,
        lm_head_weight=target_model.lm_head.weight.data,
        embed_weight=embed_weight.data if isinstance(embed_weight, torch.Tensor) else None,
        draft_vocab_ids=draft_vocab_ids,
    )
    draft_state = state["draft"]
    embed_key = "embed_tokens.weight"
    ckpt_embed = draft_state.get(embed_key)
    if isinstance(ckpt_embed, torch.Tensor) and int(ckpt_embed.shape[0]) != int(draft.vocab_size):
        if int(ckpt_embed.shape[1]) != int(draft.hidden_size):
            raise RuntimeError(
                "EAGLE3 draft embed hidden-size mismatch: "
                f"checkpoint={tuple(ckpt_embed.shape)} expected_hidden={int(draft.hidden_size)}"
            )
        full_embed = draft.embed_tokens.weight.detach().clone().to(device=ckpt_embed.device, dtype=ckpt_embed.dtype)
        rows = min(int(ckpt_embed.shape[0]), int(full_embed.shape[0]))
        full_embed[:rows].copy_(ckpt_embed[:rows])
        draft_state = dict(draft_state)
        draft_state[embed_key] = full_embed
        _trace_once(
            "eagle3_draft_embed_expand",
            f"[eagle3-draft] embed expanded ckpt={int(ckpt_embed.shape[0])}→{int(draft.vocab_size)}",
        )
    draft.load_state_dict(draft_state, strict=True)
    draft.layer_indices = tuple(int(x) for x in state.get("layer_indices", (0, 1, 2)))
    draft.to(device=device, dtype=torch.bfloat16)
    draft.eval()
    if os.getenv(
        "AICAS_EAGLE3_DRAFT_FLASHDECODE_FFN",
        os.getenv("AICAS_EAGLE3_DRAFT_FLASH_FFN", "1"),
    ) == "1":
        try:
            from aicas_env import flashdecode_ffn_prealloc
            from my_kernel.flashDecodeFFN.flashdecodeffn import from_torch_qwen3vl_text_mlp

            draft.midlayer.mlp = from_torch_qwen3vl_text_mlp(
                draft.midlayer.mlp,
                use_prealloc=flashdecode_ffn_prealloc(),
            )
            draft.midlayer.mlp.eval()
            draft._aicas_flashdecode_ffn = True
        except Exception as exc:
            draft._aicas_flashdecode_ffn = False
            _trace_once(
                "eagle3_draft_flashffn_fallback",
                f"[eagle3-draft] flashdecodeffn fallback: {type(exc).__name__}: {exc}",
            )
    return draft


def _extract_past(outputs):
    past = getattr(outputs, "past_key_values", None)
    if past is not None:
        return past
    if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
        return outputs[1]
    return None


def _extract_last_hidden(outputs):
    hidden = getattr(outputs, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
        return outputs[0]
    return None


def _prefill_with_feature(model, inputs: dict, layer_indices: tuple[int, int, int]):
    forward_inputs = _prepare_prefill_forward_inputs(model, inputs)
    with torch.no_grad(), _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
        outputs = model(
            **forward_inputs,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
    _remember_eagle3_hidden_layout(model, outputs.hidden_states)
    feature = _select_eagle3_feature(outputs.hidden_states, layer_indices)[:, -1:, :]
    return outputs, feature


def _verify_chain_core(
    model,
    past_kv,
    candidate_ids: torch.Tensor,
    token_state: dict | None,
    position_kwargs: dict,
    layer_indices: tuple[int, int, int],
):
    core = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if core is None or lm_head is None:
        raise RuntimeError("EAGLE3 verifier requires model.model and lm_head")
    kwargs = {
        "input_ids": candidate_ids,
        "past_key_values": past_kv,
        "use_cache": True,
        "return_dict": True,
        "output_hidden_states": True,
        "output_attentions": False,
        **position_kwargs,
    }
    with torch.no_grad():
        with _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
            if os.getenv("AICAS_EAGLE3_CHAIN_SPEC_KERNELS", "0") == "1":
                kernel_ctx = contextlib.ExitStack()
                kernel_ctx.enter_context(spec_qk_rotary(model))
                kernel_ctx.enter_context(spec_multitoken_attention(model))
            else:
                kernel_ctx = contextlib.nullcontext()
            with kernel_ctx:
                outputs = core(**kwargs)
    hidden = _extract_last_hidden(outputs)
    vpast = _extract_past(outputs)
    if not isinstance(hidden, torch.Tensor) or vpast is None:
        raise RuntimeError("EAGLE3 verifier did not return hidden/past")

    bias = getattr(lm_head, "bias", None)
    if can_use_lm_head_argmax_triton_multi is None or lm_head_argmax_triton_multi is None:
        raise RuntimeError("EAGLE3 verifier requires lm_head_argmax_triton_multi")
    if not can_use_lm_head_argmax_triton_multi(hidden, lm_head.weight, bias):
        raise RuntimeError(
            "EAGLE3 verifier argmax kernel is not applicable "
            f"hidden={tuple(hidden.shape)} hidden_dtype={hidden.dtype} "
            f"weight_dtype={lm_head.weight.dtype} cuda={hidden.is_cuda}"
        )
    workspace = token_state.get("eagle3_lm_head_workspace") if isinstance(token_state, dict) else None
    target, workspace = lm_head_argmax_triton_multi(
        hidden,
        lm_head.weight,
        bias,
        workspace=workspace,
    )
    if isinstance(token_state, dict):
        token_state["eagle3_lm_head_workspace"] = workspace
    target = target.reshape(-1).to(dtype=torch.long)

    feature = _select_eagle3_feature(outputs.hidden_states, layer_indices)
    feature_steps = [feature[:, idx:idx + 1, :] for idx in range(int(feature.shape[1]))]
    hidden_steps = [hidden[:, idx:idx + 1, :] for idx in range(int(hidden.shape[1]))]
    return target, vpast, feature_steps, hidden_steps


def _verify_chain(model, past_kv, candidate_ids, token_state, layer_indices):
    position_kwargs = _decode_position_kwargs(model, past_kv, candidate_ids)
    if os.getenv("AICAS_EAGLE3_CHAIN_ATTENTION_MASK", "0") == "1":
        start = _cache_seq_len(past_kv)
        position_kwargs["attention_mask"] = torch.ones(
            (int(candidate_ids.shape[0]), start + int(candidate_ids.shape[1])),
            device=candidate_ids.device,
            dtype=torch.long,
        )
    return _verify_chain_core(
        model,
        past_kv,
        candidate_ids,
        token_state,
        position_kwargs,
        layer_indices,
    )


def _refresh_accepted_chain(
    model,
    past_kv,
    accepted_tokens: torch.Tensor,
    token_state: dict | None,
    layer_indices: tuple[int, int, int],
):
    if os.getenv("AICAS_EAGLE3_DRAFT_EXTEND_CUDAGRAPH", "1") == "1":
        ts = token_state if isinstance(token_state, dict) else {}
        if not ts.get("disable_eagle3_extend_graph", False):
            try:
                return _refresh_accepted_chain_graph(
                    model,
                    past_kv,
                    accepted_tokens,
                    token_state,
                    layer_indices,
                )
            except Exception as exc:
                ts["disable_eagle3_extend_graph"] = True
                if os.getenv("AICAS_EAGLE3_DRAFT_EXTEND_CUDAGRAPH_REQUIRE", "0") == "1":
                    raise
                _trace_once(
                    "eagle3_draft_extend_cudagraph_fallback",
                    f"[eagle3-draft] extend cudagraph fallback: {type(exc).__name__}: {exc}",
                )
    candidate_ids = accepted_tokens.reshape(1, -1).to(dtype=torch.long)
    target, vpast, feature_steps, hidden_steps = _verify_chain(
        model,
        past_kv,
        candidate_ids,
        token_state if isinstance(token_state, dict) else {},
        layer_indices,
    )
    feature = torch.cat(feature_steps, dim=1)
    hidden = torch.cat(hidden_steps, dim=1)
    return target, vpast, feature, hidden


def _refresh_accepted_chain_graph(
    model,
    past_kv,
    accepted_tokens: torch.Tensor,
    token_state: dict | None,
    layer_indices: tuple[int, int, int],
):
    candidate_ids = accepted_tokens.reshape(1, -1).to(dtype=torch.long)
    # Protect original past_kv: decode_graph_runtime validation may modify
    # it in-place during graph capture. Pass a clone so the eager fallback
    # in _refresh_accepted_chain sees an uncorrupted cache.
    safe_kv = _clone_cache_to_len(past_kv, _cache_seq_len(past_kv))
    target, vpast, feature_steps, hidden_steps = _graph_verify_chain(
        model,
        safe_kv,
        candidate_ids,
        token_state if isinstance(token_state, dict) else {},
        layer_indices,
    )
    feature = torch.cat(feature_steps, dim=1)
    hidden = torch.cat(hidden_steps, dim=1)
    return target, vpast, feature, hidden


def _decode_graph_next_for_tokens(model, past_kv, accepted_tokens: torch.Tensor, token_state: dict | None):
    if _decode_forward_next_token is None:
        raise RuntimeError("decode graph forward helper unavailable")
    candidate_ids = accepted_tokens.reshape(1, -1).to(dtype=torch.long)
    position_kwargs = _decode_position_kwargs(model, past_kv, candidate_ids)
    kwargs = {
        "input_ids": candidate_ids,
        "past_key_values": past_kv,
        "use_cache": True,
        "return_dict": False,
        "output_hidden_states": False,
        "output_attentions": False,
        "logits_to_keep": 1,
        **position_kwargs,
    }
    state = token_state if isinstance(token_state, dict) else {}
    with torch.no_grad():
        return _decode_forward_next_token(model, kwargs, state)[0]


def _decode_accept_path(
    model,
    past_kv,
    accepted_tokens: torch.Tensor,
    token_state: dict | None,
    layer_indices: tuple[int, int, int],
    *,
    attention_mask: torch.Tensor | None = None,
):
    if _decode_forward_next_token is None:
        raise RuntimeError("decode graph forward helper unavailable")
    candidate_ids = accepted_tokens.reshape(1, -1).to(dtype=torch.long)
    position_kwargs = _decode_position_kwargs(model, past_kv, candidate_ids)
    kwargs = {
        "input_ids": candidate_ids,
        "past_key_values": past_kv,
        "use_cache": True,
        "return_dict": False,
        "output_hidden_states": False,
        "output_attentions": False,
        "logits_to_keep": 1,
        **position_kwargs,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    state = token_state if isinstance(token_state, dict) else {}
    with torch.no_grad():
        next_token, _outputs = _decode_forward_next_token(model, kwargs, state)
    vpast = _extract_past(_outputs)
    if vpast is None:
        raise RuntimeError("decode accept path did not return past key values")
    return next_token.reshape(-1).to(dtype=torch.long), vpast, None


def _decode_graph_accept_step(model, past_kv, accepted_tokens: torch.Tensor):
    step_fn = getattr(model, "_decode_cudagraph_step_from_cache", None)
    if not callable(step_fn):
        raise RuntimeError("decode graph step helper unavailable")
    flat = accepted_tokens.reshape(-1).to(dtype=torch.long)
    if int(flat.numel()) != 1:
        raise RuntimeError("decode graph accept step currently supports append_count=1")
    cache_pos = torch.tensor(
        [_eagle3_cache_seq_len(past_kv)],
        device=flat.device,
        dtype=torch.long,
    )
    next_token, graph_past = step_fn(
        input_ids=flat.reshape(1, 1),
        past_key_values=past_kv,
        cache_position=cache_pos,
    )
    return next_token.reshape(-1).to(dtype=torch.long), graph_past


def _eagle3_cache_len_diag(cache) -> dict:
    logical = getattr(cache, "_aicas_eagle3_seq_len", None)
    layers = getattr(cache, "layers", None)
    first_layer_len = None
    first_keys_shape = None
    first_cumulative = None
    if layers:
        try:
            layer = layers[0]
            keys = getattr(layer, "keys", None)
            if isinstance(keys, torch.Tensor):
                first_keys_shape = tuple(int(x) for x in keys.shape)
                first_layer_len = int(keys.shape[-2])
            first_cumulative = getattr(layer, "cumulative_length", None)
            if first_cumulative is not None:
                first_cumulative = int(first_cumulative)
        except Exception:
            pass
    return {
        "logical": int(logical) if logical is not None else None,
        "physical": int(_cache_seq_len(cache)),
        "eagle3": int(_eagle3_cache_seq_len(cache)),
        "first_layer_len": first_layer_len,
        "first_cumulative": first_cumulative,
        "first_keys_shape": first_keys_shape,
    }


def _cache_suffix_diff_diag(a_cache, b_cache, start_pos: int, append_count: int, *, label_a: str, label_b: str) -> dict:
    diag = {
        "label_a": label_a,
        "label_b": label_b,
        "start_pos": int(start_pos),
        "append_count": int(append_count),
        "layers": [],
        "max_k": 0.0,
        "max_v": 0.0,
        "max_k_layer": None,
        "max_v_layer": None,
    }
    if a_cache is None or b_cache is None or int(append_count) <= 0:
        diag["error"] = "empty_cache_or_append"
        return diag
    a_layers = getattr(a_cache, "layers", None)
    b_layers = getattr(b_cache, "layers", None)
    if not a_layers or not b_layers:
        diag["error"] = "missing_layers"
        return diag
    layer_count = min(len(a_layers), len(b_layers))
    s0 = int(start_pos)
    s1 = s0 + int(append_count)
    max_records = int(os.getenv("AICAS_EAGLE3_CACHE_SELECT_CHECK_LAYERS", "8"))
    for idx in range(layer_count):
        ak = getattr(a_layers[idx], "keys", None)
        av = getattr(a_layers[idx], "values", None)
        bk = getattr(b_layers[idx], "keys", None)
        bv = getattr(b_layers[idx], "values", None)
        if not all(isinstance(x, torch.Tensor) for x in (ak, av, bk, bv)):
            continue
        if int(ak.shape[-2]) < s1 or int(bk.shape[-2]) < s1:
            rec = {"layer": int(idx), "error": f"short_cache a={tuple(ak.shape)} b={tuple(bk.shape)}"}
            if len(diag["layers"]) < max_records:
                diag["layers"].append(rec)
            continue
        ak_s = ak[..., s0:s1, :].float()
        av_s = av[..., s0:s1, :].float()
        bk_s = bk[..., s0:s1, :].float()
        bv_s = bv[..., s0:s1, :].float()
        dk = (ak_s - bk_s).abs()
        dv = (av_s - bv_s).abs()
        k_max = float(dk.max().item()) if int(dk.numel()) else 0.0
        v_max = float(dv.max().item()) if int(dv.numel()) else 0.0
        k_mean = float(dk.mean().item()) if int(dk.numel()) else 0.0
        v_mean = float(dv.mean().item()) if int(dv.numel()) else 0.0
        if k_max > float(diag["max_k"]):
            diag["max_k"] = k_max
            diag["max_k_layer"] = int(idx)
        if v_max > float(diag["max_v"]):
            diag["max_v"] = v_max
            diag["max_v_layer"] = int(idx)
        if len(diag["layers"]) < max_records:
            k_arg = int(dk.reshape(-1).argmax().item()) if int(dk.numel()) else 0
            v_arg = int(dv.reshape(-1).argmax().item()) if int(dv.numel()) else 0
            dim = int(dk.shape[-1]) if int(dk.ndim) > 0 else 1
            rec = {
                "layer": int(idx),
                "k_max": k_max,
                "k_mean": k_mean,
                "v_max": v_max,
                "v_mean": v_mean,
                "k_arg_flat": k_arg,
                "v_arg_flat": v_arg,
                "k_dim": int(k_arg % dim),
                "v_dim": int(v_arg % dim),
            }
            diag["layers"].append(rec)
    return diag


def _graph_verify_chain(model, past_kv, candidate_ids, token_state, layer_indices):
    if not isinstance(token_state, dict):
        token_state = {}
    if _runtime_decode_cudagraph_step_from_cache is None:
        raise RuntimeError("decode graph step helper unavailable")
    if not isinstance(candidate_ids, torch.Tensor) or candidate_ids.ndim != 2:
        raise RuntimeError("EAGLE3 graph verifier expects candidate_ids=[batch,q_len]")
    position_kwargs = _decode_position_kwargs(model, past_kv, candidate_ids)
    if os.getenv("AICAS_EAGLE3_CHAIN_ATTENTION_MASK", "0") == "1":
        start = _cache_seq_len(past_kv)
        position_kwargs["attention_mask"] = torch.ones(
            (int(candidate_ids.shape[0]), start + int(candidate_ids.shape[1])),
            device=candidate_ids.device,
            dtype=torch.long,
        )
    try:
        target, vpast, feature, hidden = _runtime_decode_cudagraph_step_from_cache(
            model,
            input_ids=candidate_ids,
            past_key_values=past_kv,
            token_state=token_state,
            forward_mode="target_verify",
            layer_indices=tuple(int(x) for x in layer_indices),
            **position_kwargs,
        )
        feature_steps = [feature[:, idx:idx + 1, :] for idx in range(int(feature.shape[1]))]
        hidden_steps = [hidden[:, idx:idx + 1, :] for idx in range(int(hidden.shape[1]))]
        return target.reshape(-1).to(dtype=torch.long), vpast, feature_steps, hidden_steps
    except Exception:
        token_state["disable_eagle3_graph"] = True
        raise


def _prefill_with_full_feature(model, inputs: dict, layer_indices: tuple[int, int, int]):
    # ── Prefill graph cache replay ──────────────────────────────────────
    graph_runner = getattr(model, "_aicas_prefill_graph_runner", None)
    if graph_runner is not None and graph_runner.is_enabled:
        input_ids_tensor = inputs.get("input_ids")
        if input_ids_tensor is not None:
            prompt_len = int(input_ids_tensor.shape[1])
            if prompt_len > 0:
                grid = inputs.get("image_grid_thw")
                entry = graph_runner.lookup(prompt_len, grid)
                if entry is not None:
                    try:
                        outputs, feature = graph_runner.replay(entry, inputs)
                        return outputs, feature
                    except Exception as exc:
                        # Fall through to eager on replay failure
                        if os.getenv("AICAS_EAGLE3_DEBUG", "0") == "1":
                            print(f"[prefill-graph] replay fallback: {exc}", flush=True)

    # ── Eager prefill (original) ────────────────────────────────────────
    forward_inputs = _prepare_prefill_forward_inputs(model, _sanitize_qwen3vl_mm_token_type_ids(model, inputs))
    with torch.no_grad(), _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
        outputs = model(
            **forward_inputs,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
    _remember_eagle3_hidden_layout(model, outputs.hidden_states)
    result = (outputs, _select_eagle3_feature(outputs.hidden_states, layer_indices))

    # ── On-demand capture after eager prefill ──────────────────────────
    if graph_runner is not None and graph_runner.is_enabled:
        input_ids_tensor = inputs.get("input_ids")
        if input_ids_tensor is not None:
            prompt_len = int(input_ids_tensor.shape[1])
            if prompt_len > 0:
                grid = inputs.get("image_grid_thw")
                # Only capture if not already cached
                if graph_runner.lookup(prompt_len, grid) is None:
                    capture_on_demand = os.getenv("AICAS_PREFILL_TTFT_CAPTURE_ON_DEMAND", "1") == "1"
                    if capture_on_demand:
                        graph_runner.capture(prompt_len, inputs)

    return result


# ── KV prefix cache helpers ─────────────────────────────────────────────

def _clone_full_kv(past_kv):
    """Create a full-depth clone of a DynamicCache/StaticCache so the
    original can be safely modified (e.g. by the decode loop) without
    affecting the cached copy."""
    from spec_decode.eagle3_cache_adapter import clone_cache_to_len
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            seq_len = int(past_kv.key_cache[0].shape[2])
            return clone_cache_to_len(past_kv, seq_len)
    except Exception:
        pass
    seq_len = _cache_seq_len(past_kv)
    return clone_cache_to_len(past_kv, seq_len)


def _last_logits_from_output(outputs):
    """Extract last-token logits from model outputs (CausalLMOutputWithPast).
    Returns [1, 1, vocab_size] tensor or None."""
    logits = getattr(outputs, "logits", None)
    if isinstance(logits, torch.Tensor) and logits.numel() > 0:
        return logits[:, -1:, :]
    return None


def _find_image_end_pos(input_ids):
    """Find the position after the last <|image_pad|> token (ID=151655)."""
    img_mask = (input_ids == 151655)
    if not img_mask.any():
        return 0
    last_pos = img_mask.nonzero(as_tuple=True)[1][-1].item()
    return last_pos + 1


def _prefill_with_prefix_cache(
    model,
    inputs: dict,
    layer_indices: tuple[int, int, int],
    kv_cache=None,
    force_full_prefill: bool = False,
):
    """Prefill with merged two-layer KV cache.

    Delegates to MergedKVCache.prefill() for the full two-phase logic.
    Falls back to full prefill when kv_cache is None or force_full_prefill.

    Layer 1 (Image): exact hash match on pixel_values.
      Hit  => skip Phase 1 (image-token prefill), run Phase 2 (question tokens).

    Layer 2 (Text): radix-tree prefix match on question tokens.
      Hit  => optionally skip a prefix of Phase 2.

    Returns:
        (outputs, feature) -- same interface as _prefill_with_full_feature.
    """
    if kv_cache is None or force_full_prefill:
        return _prefill_with_full_feature(model, inputs, layer_indices)

    from spec_decode.kv_cache_merged import prefill_with_merged_cache
    return prefill_with_merged_cache(model, inputs, layer_indices, kv_cache)


def _clone_full_kv_slice(past_kv, length: int):
    """Clone only the first `length` positions of a KV cache."""
    from spec_decode.eagle3_cache_adapter import clone_cache_to_len
    return clone_cache_to_len(past_kv, int(length))


def _make_phase2_inputs(model, inputs: dict, layer1_entry, device):
    """Build forward kwargs for Phase 2 (question token prefill).

    Removes vision inputs and sets past_key_values + cache_position correctly.
    """
    input_ids = inputs["input_ids"]
    image_end_pos = layer1_entry.image_end_pos
    prompt_len = int(input_ids.shape[1])

    phase2_ids = input_ids[:, image_end_pos:]
    if phase2_ids.shape[1] == 0:
        # Edge case: input ends exactly at image_end_pos -- feed the last token
        phase2_ids = input_ids[:, -1:]

    out = dict(inputs)
    out["input_ids"] = phase2_ids

    # Remove vision inputs (already processed in Phase 1)
    out.pop("pixel_values", None)
    out.pop("image_grid_thw", None)
    out.pop("video_grid_thw", None)
    out.pop("second_per_grid_ts", None)

    # Sanitize mm_token_type_ids for the truncated input
    out = _sanitize_qwen3vl_mm_token_type_ids(model, out)

    # Prepare forward kwargs (cache_position will be incorrect -- overridden below)
    filtered = _prepare_prefill_forward_inputs(model, out)

    # Override cache_position to start from image_end_pos
    filtered["cache_position"] = torch.arange(
        image_end_pos, image_end_pos + int(phase2_ids.shape[1]),
        device=device, dtype=torch.long,
    )

    # Build full-length attention mask
    filtered.pop("attention_mask", None)
    filtered["attention_mask"] = torch.ones(
        1, prompt_len, device=device, dtype=torch.long,
    )

    # Attach the cached image KV
    filtered["past_key_values"] = layer1_entry.image_kv

    return filtered


def _resolve_device(model):
    """Get the device of the first parameter."""
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def _hash_tensor(t):
    """Stable hash of tensor bytes."""
    if t is None:
        return 0
    return hash(t.detach().cpu().resolve_conj().numpy().tobytes())


class _CachedOutput:
    """Lightweight output object that mimics a HF model output for cached prefill."""
    def __init__(self, past_key_values=None, logits=None):
        self.past_key_values = past_key_values
        self.logits = logits

    @property
    def last_hidden_state(self):
        return None


def _inputs_with_current_ids(inputs: dict, current_input_ids: torch.Tensor) -> dict:
    out = dict(inputs)
    out["input_ids"] = current_input_ids
    device = current_input_ids.device
    seq_len = int(current_input_ids.shape[1])

    attention_mask = inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        out["attention_mask"] = torch.ones(
            (int(current_input_ids.shape[0]), seq_len),
            device=device,
            dtype=attention_mask.dtype,
        )

    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if isinstance(mm_token_type_ids, torch.Tensor):
        prefix = mm_token_type_ids.to(device=device)
        if int(prefix.shape[1]) >= seq_len:
            out["mm_token_type_ids"] = prefix[:, :seq_len]
        else:
            pad = torch.zeros(
                (int(prefix.shape[0]), seq_len - int(prefix.shape[1])),
                device=device,
                dtype=prefix.dtype,
            )
            out["mm_token_type_ids"] = torch.cat((prefix, pad), dim=1)
    return out


def _sanitize_qwen3vl_mm_token_type_ids(model, inputs: dict) -> dict:
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    input_ids = inputs.get("input_ids")
    if not isinstance(mm_token_type_ids, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
        return inputs
    if mm_token_type_ids.numel() == 0:
        return inputs

    needs_rebuild = False
    try:
        needs_rebuild = bool((mm_token_type_ids.lt(0) | mm_token_type_ids.gt(2)).any().item())
    except Exception:
        needs_rebuild = True
    if not needs_rebuild:
        return inputs

    cfg = getattr(getattr(model, "model", None), "config", getattr(model, "config", None))
    image_token_id = getattr(cfg, "image_token_id", None)
    video_token_id = getattr(cfg, "video_token_id", None)
    rebuilt = torch.zeros_like(input_ids, dtype=mm_token_type_ids.dtype, device=input_ids.device)
    if image_token_id is not None:
        rebuilt = torch.where(
            input_ids.eq(int(image_token_id)),
            torch.ones_like(rebuilt),
            rebuilt,
        )
    if video_token_id is not None:
        rebuilt = torch.where(
            input_ids.eq(int(video_token_id)),
            torch.full_like(rebuilt, 2),
            rebuilt,
        )
    out = dict(inputs)
    out["mm_token_type_ids"] = rebuilt
    _trace_once(
        "eagle3_mm_token_type_ids_rebuilt",
        "[eagle3] rebuilt invalid mm_token_type_ids from input_ids",
    )
    return out


def _full_feature_for_current_ids(
    model,
    inputs: dict,
    current_input_ids: torch.Tensor,
    layer_indices: tuple[int, int, int],
) -> torch.Tensor:
    forward_inputs = _prepare_prefill_forward_inputs(
        model,
        _inputs_with_current_ids(inputs, current_input_ids),
    )
    with torch.no_grad(), _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
        outputs = model(
            **forward_inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
    return _select_eagle3_feature(outputs.hidden_states, layer_indices)


def _full_hidden_for_current_ids(
    model,
    inputs: dict,
    current_input_ids: torch.Tensor,
) -> torch.Tensor:
    forward_inputs = _prepare_prefill_forward_inputs(
        model,
        _inputs_with_current_ids(inputs, current_input_ids),
    )
    with torch.no_grad(), _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
        outputs = model(
            **forward_inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
    hidden = getattr(outputs, "hidden_states", None)
    if not isinstance(hidden, (tuple, list)) or not hidden:
        raise RuntimeError("EAGLE3 full hidden capture failed")
    last = hidden[-1]
    if not isinstance(last, torch.Tensor):
        raise RuntimeError("EAGLE3 full hidden last state missing")
    return last


def _target_next_for_current_ids(
    model,
    inputs: dict,
    current_input_ids: torch.Tensor,
    eos_ids: set[int],
    generated_len: int,
    min_new_tokens: int,
) -> torch.Tensor:
    forward_inputs = _prepare_prefill_forward_inputs(
        model,
        _inputs_with_current_ids(inputs, current_input_ids),
    )
    with torch.no_grad(), _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
        outputs = model(
            **forward_inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            logits_to_keep=1,
        )
    return _argmax_with_min_new_tokens(
        outputs.logits[:, -1:, :],
        eos_ids,
        generated_len=int(generated_len),
        min_new_tokens=int(min_new_tokens),
    )


def _target_topk_for_current_ids(
    model,
    inputs: dict,
    current_input_ids: torch.Tensor,
    eos_ids: set[int],
    generated_len: int,
    min_new_tokens: int,
    k: int = 8,
) -> list[int]:
    forward_inputs = _prepare_prefill_forward_inputs(
        model,
        _inputs_with_current_ids(inputs, current_input_ids),
    )
    with torch.no_grad(), _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
        outputs = model(
            **forward_inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            logits_to_keep=1,
        )
    logits = outputs.logits[:, -1, :].float().reshape(-1)
    if generated_len < min_new_tokens:
        for eos_id in eos_ids:
            if 0 <= int(eos_id) < int(logits.numel()):
                logits[int(eos_id)] = -torch.inf
    top_k = min(int(k), int(logits.numel()))
    return [int(x) for x in logits.topk(top_k, dim=-1).indices.detach().cpu().tolist()]


def _generate_style_next_for_current_ids(
    model,
    inputs: dict,
    current_input_ids: torch.Tensor,
    eos_ids: set[int],
    generated_len: int,
    min_new_tokens: int,
) -> int:
    # Diagnostic only: rerun HF generate to the same prefix and read the next
    # token from the real generation path.
    prompt_len = int(inputs["input_ids"].shape[1])
    target_len = int(current_input_ids.shape[1]) - prompt_len + 1
    if target_len <= 0:
        target_len = 1
    gen_inputs = dict(inputs)
    gen_inputs["max_new_tokens"] = target_len
    gen_inputs["do_sample"] = False
    gen_inputs["temperature"] = 0.0
    gen_inputs["use_cache"] = True
    if min_new_tokens > 0:
        gen_inputs["min_new_tokens"] = int(min_new_tokens)
    with torch.no_grad():
        out = model.generate(**gen_inputs)
    seq = out.sequences if hasattr(out, "sequences") else out
    generated = seq[0, prompt_len:].to(device=current_input_ids.device, dtype=torch.long)
    idx = int(current_input_ids.shape[1]) - prompt_len
    if idx < int(generated.numel()):
        return int(generated[idx].item())
    if int(generated.numel()) > 0:
        return int(generated[-1].item())
    return -1


def _strict_check_accepted_tokens(
    model,
    inputs: dict,
    current_input_ids: torch.Tensor,
    accepted_tokens: torch.Tensor,
    *,
    generated_len: int,
    eos_ids: set[int],
    min_new_tokens: int,
    round_index: int,
    accept_length: int,
) -> None:
    prefix_ids = current_input_ids
    flat = accepted_tokens.reshape(-1).to(device=current_input_ids.device, dtype=torch.long)
    for offset in range(int(flat.numel())):
        expected = int(_target_next_for_current_ids(
            model,
            inputs,
            prefix_ids,
            eos_ids,
            generated_len=int(generated_len) + offset,
            min_new_tokens=int(min_new_tokens),
        ))
        actual = flat[offset].reshape(())
        actual_i = int(actual.item())
        if actual_i != expected:
            raise RuntimeError(
                "EAGLE3 strict greedy mismatch "
                f"round={int(round_index)} offset={offset} "
                f"generated_len={int(generated_len)} accept_length={int(accept_length)} "
                f"actual={actual_i} expected={expected}"
            )
        prefix_ids = torch.cat((prefix_ids, actual.reshape(1, 1)), dim=1)


def _strict_check_next_token(
    model,
    inputs: dict,
    prefix_ids: torch.Tensor,
    next_tok,
    *,
    generated_len: int,
    eos_ids: set[int],
    min_new_tokens: int,
    round_index: int,
    accept_length: int,
    path_target_tokens: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_indices: torch.Tensor,
    best_candidate: torch.Tensor,
    chain_next=None,
) -> None:
    expected = int(_target_next_for_current_ids(
        model,
        inputs,
        prefix_ids,
        eos_ids,
        generated_len=int(generated_len),
        min_new_tokens=int(min_new_tokens),
    ))
    actual = int(next_tok.item() if isinstance(next_tok, torch.Tensor) else next_tok)
    if actual == expected:
        return
    best_i = int(best_candidate.item() if isinstance(best_candidate, torch.Tensor) else best_candidate)
    path = path_target_tokens[best_i].reshape(-1)[:8].detach().cpu().tolist()
    cand = candidates[best_i].reshape(-1)[:8].detach().cpu().tolist()
    retrieve = retrieve_indices[best_i].reshape(-1)[:8].detach().cpu().tolist()
    raise RuntimeError(
        "EAGLE3 strict next mismatch "
        f"round={int(round_index)} generated_len={int(generated_len)} "
        f"accept_length={int(accept_length)} actual={actual} expected={expected} "
        f"chain_next={chain_next} "
        f"path_target={path} candidate={cand} retrieve={retrieve}"
    )


def _strict_tree_reference_debug(
    model,
    past_kv,
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
    tree_target_tokens: torch.Tensor,
    tree_hidden: torch.Tensor,
    retrieve_indices: torch.Tensor,
    best_candidate: torch.Tensor,
    *,
    base_cache_len: int,
    current_len: int,
    layer_indices: tuple[int, int, int],
) -> str:
    device = draft_tokens.device
    q_len = int(draft_tokens.shape[1])
    best_i = int(best_candidate.item() if isinstance(best_candidate, torch.Tensor) else best_candidate)
    row = retrieve_indices[best_i].reshape(-1)
    root_idx = int(row[0].item()) if int(row.numel()) > 0 and int(row[0].item()) >= 0 else 0
    node_count = int(tree_mask.shape[-1])
    root_idx = max(0, min(root_idx, node_count - 1))

    def _small_list(tensor: torch.Tensor, limit: int = 8) -> list[int]:
        return [int(x) for x in tensor.reshape(-1)[:limit].detach().cpu().tolist()]

    details: list[str] = [
        f"base_cache_len={int(base_cache_len)}",
        f"current_len={int(current_len)}",
        f"q_len={q_len}",
        f"root_idx={root_idx}",
        f"tree_pos={_small_list(tree_position_ids, 8)}",
    ]
    try:
        visible = torch.nonzero(
            tree_mask.reshape(node_count, node_count)[root_idx].to(dtype=torch.bool),
            as_tuple=False,
        ).reshape(-1)
        details.append(f"root_visible={_small_list(visible, 16)}")
    except Exception as exc:
        details.append(f"root_visible_error={type(exc).__name__}:{exc}")

    def _summarize(name: str, tokens: torch.Tensor | None) -> None:
        if not isinstance(tokens, torch.Tensor):
            details.append(f"{name}=None")
            return
        flat = tokens.reshape(-1).to(device=device, dtype=torch.long)
        root = int(flat[root_idx].item()) if int(flat.numel()) > root_idx else None
        path = flat[row.to(device=device).clamp_min(0)].reshape(-1)
        details.append(f"{name}_root={root} {name}_path={_small_list(path, 8)}")

    _summarize("fast", tree_target_tokens)
    try:
        eager_tokens = model.lm_head(tree_hidden).argmax(dim=-1).reshape(-1).to(dtype=torch.long)
        _summarize("fast_eager_lm_head", eager_tokens)
    except Exception as exc:
        details.append(f"fast_eager_lm_head_error={type(exc).__name__}:{exc}")

    raw_variants = os.getenv(
        "AICAS_EAGLE3_STRICT_TREE_REF_VARIANTS",
        "no_tree_attn,no_qk_rotary,no_tree_attn_qk,no_smallq",
    )
    variant_env = {
        "no_tree_attn": {"AICAS_SPEC_MT_ATTN": "0"},
        "no_qk_rotary": {"AICAS_SPEC_QK_ROTARY": "0"},
        "no_tree_attn_qk": {"AICAS_SPEC_MT_ATTN": "0", "AICAS_SPEC_QK_ROTARY": "0"},
        "no_smallq": {
            "AICAS_SPEC_MT_ATTN": "0",
            "AICAS_SPEC_QK_ROTARY": "0",
            "AICAS_SPEC_DECODE_RMSNORM_TRITON": "0",
            "AICAS_SPEC_DECODE_ADD_RMSNORM_TRITON": "0",
        },
    }
    dtype = next(model.parameters()).dtype
    cache_position = torch.arange(
        int(base_cache_len),
        int(base_cache_len) + q_len,
        device=device,
        dtype=torch.long,
    )
    attention_mask = _eagle3_tree_attention_mask(
        dtype,
        device,
        int(base_cache_len),
        tree_mask,
        total_kv_len=int(base_cache_len) + q_len,
    )
    position_ids = _eagle3_tree_position_ids(
        model,
        int(current_len),
        tree_position_ids.to(device=device, dtype=torch.long),
        batch_size=1,
    )
    for name in [part.strip() for part in raw_variants.split(",") if part.strip()]:
        env = variant_env.get(name)
        if env is None:
            details.append(f"{name}_error=unknown_variant")
            continue
        try:
            ref_cache = _clone_cache_to_len(past_kv, int(base_cache_len))
            with _temporary_env(env):
                ref_tokens, _, _, _ = _eagle3_tree_core_forward(
                    model,
                    ref_cache,
                    draft_tokens,
                    attention_mask,
                    position_ids,
                    cache_position,
                    layer_indices,
                    {},
                    workspace_key=f"strict_tree_ref_{name}_lm_head_workspace",
                )
            _summarize(name, ref_tokens)
        except Exception as exc:
            details.append(f"{name}_error={type(exc).__name__}:{exc}")
    return "EAGLE3 strict tree refs: " + "; ".join(details)


def _fill_eagle3_tree_attention_mask_(
    out: torch.Tensor,
    start_pos: int,
    tree_mask: torch.Tensor,
) -> torch.Tensor:
    node_count = int(tree_mask.shape[-1])
    if int(out.shape[-2]) != node_count:
        raise RuntimeError(
            f"EAGLE3 tree mask q mismatch: out={tuple(out.shape)} tree={tuple(tree_mask.shape)}"
        )
    if int(start_pos) + node_count > int(out.shape[-1]):
        raise RuntimeError(
            f"EAGLE3 tree mask kv too short: start={int(start_pos)} nodes={node_count} "
            f"kv={int(out.shape[-1])}"
        )
    out.fill_(torch.finfo(out.dtype).min)
    if int(start_pos) > 0:
        out[:, :, :, :int(start_pos)].zero_()
    visible = tree_mask.to(device=out.device, dtype=torch.bool)
    out[:, :, :, int(start_pos):int(start_pos) + node_count].masked_fill_(visible, 0.0)
    return out


def _eagle3_tree_attention_mask(
    dtype,
    device,
    start_pos: int,
    tree_mask: torch.Tensor,
    *,
    total_kv_len: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    tree_mask = tree_mask.to(device=device)
    node_count = int(tree_mask.shape[-1])
    kv_len = int(total_kv_len) if total_kv_len is not None else int(start_pos) + node_count
    shape = (1, 1, node_count, kv_len)
    if out is None or tuple(out.shape) != shape or out.device != device or out.dtype != dtype:
        out = torch.empty(shape, device=device, dtype=dtype)
    return _fill_eagle3_tree_attention_mask_(out, start_pos, tree_mask)


def _eagle3_tree_position_ids(model, current_len: int, tree_position_ids: torch.Tensor, batch_size: int = 1):
    device = tree_position_ids.device
    text_pos = tree_position_ids.to(device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1) + int(current_len)
    rope_deltas = getattr(getattr(model, "model", None), "rope_deltas", None)
    if isinstance(rope_deltas, torch.Tensor):
        delta = rope_deltas.to(device=device, dtype=torch.long)
        if delta.shape[0] != batch_size:
            delta = delta[:1].expand(batch_size, -1)
        mrope_pos = text_pos + delta
    else:
        mrope_pos = text_pos
    return torch.cat(
        (
            text_pos.unsqueeze(0),
            mrope_pos.unsqueeze(0).expand(3, -1, -1),
        ),
        dim=0,
    ).contiguous()


def _eagle3_embed_tokens(
    language_model,
    input_ids: torch.Tensor,
    token_state: dict | None,
    workspace_key: str,
) -> torch.Tensor:
    embed = getattr(language_model, "embed_tokens", None)
    weight = getattr(embed, "weight", None)
    if (
        os.getenv("AICAS_EAGLE3_GRAPH_EMBED_TRITON", "1") == "1"
        and isinstance(token_state, dict)
        and can_use_embedding_gather_triton is not None
        and embedding_gather_triton is not None
        and can_use_embedding_gather_triton(input_ids, weight)
    ):
        expected = (int(input_ids.shape[0]), int(input_ids.shape[1]), int(weight.shape[1]))
        out = token_state.get(workspace_key)
        if (
            not isinstance(out, torch.Tensor)
            or out.shape != expected
            or out.device != input_ids.device
            or out.dtype != weight.dtype
        ):
            out = torch.empty(expected, device=input_ids.device, dtype=weight.dtype)
            token_state[workspace_key] = out
        return embedding_gather_triton(input_ids, weight, out=out)
    return language_model.embed_tokens(input_ids)


_EAGLE3_DEBUG_ATTR_MISSING = object()


def _eagle3_selected_text_forward(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    past_key_values,
    cache_position: torch.Tensor | None,
    layer_indices: tuple[int, int, int],
    token_state: dict | None = None,
    embed_workspace_key: str = "tree_embed_workspace",
):
    """Verifier-only Qwen3VL text forward that records only EAGLE3 layers."""
    if os.getenv("AICAS_EAGLE3_TRACE", "0") == "1":
        _trace_once(
            "selected_target_forward",
            "[eagle3-tree] selected target forward enabled"
        )
    core = getattr(model, "model", None)
    language_model = getattr(core, "language_model", None)
    layers = getattr(language_model, "layers", None) if language_model is not None else None
    if language_model is None or not layers:
        raise RuntimeError("EAGLE3 text verifier requires model.model.language_model.layers")

    if input_ids is None or input_ids.ndim != 2:
        raise RuntimeError("EAGLE3 text verifier expects input_ids=[batch, q_len]")
    if int(input_ids.shape[0]) != 1:
        raise RuntimeError("EAGLE3 text verifier currently supports batch=1")

    inputs_embeds = _eagle3_embed_tokens(language_model, input_ids, token_state, embed_workspace_key)
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + int(inputs_embeds.shape[1]),
            device=inputs_embeds.device,
            dtype=torch.long,
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(4, int(inputs_embeds.shape[0]), -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, int(position_ids.shape[0]), -1)

    if position_ids.ndim == 3 and int(position_ids.shape[0]) == 4:
        text_position_ids = position_ids[0]
        rotary_position_ids = position_ids[1:]
    else:
        text_position_ids = None
        rotary_position_ids = position_ids

    if attention_mask is not None and attention_mask.ndim != 4:
        attention_mask = create_causal_mask(
            config=language_model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

    hidden_states = inputs_embeds
    position_embeddings = language_model.rotary_emb(hidden_states, rotary_position_ids)
    n_layers = len(layers)
    feature_sources = _eagle3_feature_sources(model, layer_indices, n_layers)
    selected: dict[tuple[str, int], torch.Tensor] = {}
    if ("input", 0) in feature_sources:
        selected[("input", 0)] = _capture_eagle3_feature_layer(hidden_states)
    wanted_layers = {idx for kind, idx in feature_sources if kind == "layer"}
    if os.getenv("AICAS_EAGLE3_TRACE", "0") == "1":
        _trace_once(
            "selected_target_feature_sources",
            "[eagle3-tree] selected feature slots "
            f"indices={tuple(int(x) for x in layer_indices)} sources={feature_sources} "
            f"captured={int(getattr(model, '_aicas_eagle3_hidden_slot_count', n_layers + 1))} "
            f"repeats={int(getattr(model, '_aicas_eagle3_hidden_layer_repeats', 1))}",
        )

    use_text_compile = os.getenv("AICAS_EAGLE3_VERIFY_TEXT_COMPILE", "0") == "1"
    compile_ctx = (
        torch.no_grad()
        if use_text_compile
        else _temporary_model_attr(model, "_aicas_disable_prefill_compile", True)
    )
    with compile_ctx:
        for layer_idx, decoder_layer in enumerate(layers):
            with _profile_range(f"eagle3_tree.verify.layer.{layer_idx}"):
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            if layer_idx in wanted_layers:
                selected[("layer", layer_idx)] = _capture_eagle3_feature_layer(hidden_states)

    hidden_states = language_model.norm(hidden_states)
    if ("norm", n_layers) in feature_sources:
        selected[("norm", n_layers)] = _capture_eagle3_feature_layer(hidden_states)

    try:
        feature = torch.cat([selected[source] for source in feature_sources], dim=-1)
    except KeyError as exc:
        raise RuntimeError(
            f"EAGLE3 selected verifier missing hidden-state source {exc.args[0]} "
            f"from requested layers={layer_indices} mapped={feature_sources}; available={sorted(selected)}"
        ) from exc
    return hidden_states, feature, past_key_values


def _eagle3_tree_core_forward(
    model,
    past_kv,
    draft_tokens: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
    layer_indices: tuple[int, int, int],
    token_state: dict | None,
    *,
    workspace_key: str = "tree_lm_head_workspace",
):
    with torch.no_grad():
        with spec_qk_rotary(model), spec_multitoken_attention(model):
            hidden, feature, tree_past = _eagle3_selected_text_forward(
                model,
                input_ids=draft_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_kv,
                cache_position=cache_position,
                layer_indices=layer_indices,
                token_state=token_state,
                embed_workspace_key=f"{workspace_key}_embed",
            )
    lm_head = model.lm_head
    bias = getattr(lm_head, "bias", None)
    if can_use_lm_head_argmax_triton_multi is None or lm_head_argmax_triton_multi is None:
        raise RuntimeError("EAGLE3 optimized verifier requires lm_head_argmax_triton_multi")
    if not can_use_lm_head_argmax_triton_multi(hidden, lm_head.weight, bias):
        raise RuntimeError(
            "EAGLE3 optimized verifier argmax kernel is not applicable "
            f"hidden={tuple(hidden.shape)} hidden_dtype={hidden.dtype} "
            f"weight_dtype={lm_head.weight.dtype} cuda={hidden.is_cuda}"
        )
    workspace = token_state.get(workspace_key) if isinstance(token_state, dict) else None
    target_tokens, workspace = lm_head_argmax_triton_multi(hidden, lm_head.weight, bias, workspace=workspace)
    if isinstance(token_state, dict):
        token_state[workspace_key] = workspace
    return target_tokens.reshape(-1).to(device=draft_tokens.device, dtype=torch.long), feature, hidden, tree_past


def _tree_verify_graph_put_lru(store, order, key, entry, max_entries=4):
    store[key] = entry
    if key in order:
        order.remove(key)
    order.append(key)
    while len(order) > max(1, int(max_entries)):
        stale = order.pop(0)
        stale_entry = store.pop(stale, None)
        if isinstance(stale_entry, dict):
            g = stale_entry.get("graph")
            if isinstance(g, torch.cuda.CUDAGraph):
                try:
                    g.reset()
                except Exception:
                    pass


def _tree_verify_graph_lru_capacity() -> int:
    try:
        from spec_decode.eagle3_cache_adapter import get_static_cache_buckets, get_static_cache_overflow_buckets

        default = max(4, len(get_static_cache_buckets()) + len(get_static_cache_overflow_buckets()))
        return max(default, int(os.getenv("AICAS_EAGLE3_TREE_VERIFY_GRAPH_MAX_ENTRIES", "128")))
    except Exception:
        return 128


def _tree_verify_graph_select_cache_len(total_len: int) -> int:
    total_len = int(total_len)
    max_cache_len = int(_select_static_cache_len(total_len))
    spare = 1
    if spare > 0 and total_len + spare > max_cache_len:
        return int(_select_static_cache_len(total_len + spare))
    return max_cache_len


def _draft_step_graph_lru_capacity() -> int:
    try:
        return max(8, int(os.getenv("AICAS_EAGLE3_DRAFT_GRAPH_MAX_ENTRIES", "1024")))
    except Exception:
        return 1024


def _draft_all_graph_lru_capacity() -> int:
    try:
        return max(4, int(os.getenv("AICAS_EAGLE3_DRAFT_ALL_GRAPH_MAX_ENTRIES", "128")))
    except Exception:
        return 128


def _eagle3_full_graph_lru_capacity() -> int:
    try:
        return max(4, int(os.getenv("AICAS_EAGLE3_FULL_GRAPH_MAX_ENTRIES", "64")))
    except Exception:
        return 64


def _eagle3_full_graph_buffer_len(draft_model, draft_past_len: int) -> int:
    """Draft KV bucket computed from the *current* draft_past_len + one-round margin.

    We only add total_tokens + depth + 8 (≈26), not the entire generate-call
    upper bound.  This keeps the buffer just large enough that its tail never
    contains uninitialised memory polluting softmax reductions.  The value
    still uses the existing bucket infrastructure, so nearby rounds share the
    same graph.
    """
    total_tokens = int(getattr(draft_model, "total_tokens", 15))
    depth = int(getattr(draft_model, "depth", 3))
    upper = max(0, int(draft_past_len)) + int(total_tokens) + int(depth) + 8
    return int(draft_model._draft_step_graph_bucket_len(upper))


def _copy_eagle3_rope_delta_(dst: torch.Tensor, model) -> torch.Tensor:
    core = getattr(model, "model", None)
    live = getattr(core, "rope_deltas", None)
    if isinstance(live, torch.Tensor) and int(live.numel()) > 0:
        src = live.reshape(-1)[:1]
        if src.device == dst.device and src.dtype == dst.dtype:
            dst.copy_(src, non_blocking=True)
        else:
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype), non_blocking=True)
    else:
        dst.zero_()
    return dst


def _eagle3_full_graph_state(model, token_state: dict):
    store = getattr(model, "_eagle3_full_graph_entries", None)
    if not isinstance(store, dict):
        store = {}
        try:
            model._eagle3_full_graph_entries = store
        except Exception:
            pass
    order = getattr(model, "_eagle3_full_graph_order", None)
    if not isinstance(order, list):
        order = []
        try:
            model._eagle3_full_graph_order = order
        except Exception:
            pass
    failed = getattr(model, "_eagle3_full_graph_failed_keys", None)
    if not isinstance(failed, set):
        failed = set()
        try:
            model._eagle3_full_graph_failed_keys = failed
        except Exception:
            pass
    return store, order, failed


def _eagle3_full_graph_failed_reason_map(model, token_state: dict):
    reasons = getattr(model, "_eagle3_full_graph_failed_reasons", None)
    if not isinstance(reasons, dict):
        reasons = {}
        try:
            model._eagle3_full_graph_failed_reasons = reasons
        except Exception:
            pass
    return reasons


def _tree_verify_graph_state(model, token_state: dict):
    """Use model-level graph caches so startup precapture survives into benchmark."""
    if os.getenv("AICAS_EAGLE3_TREE_VERIFY_GRAPH_GLOBAL_CACHE", "1") != "1":
        return (
            token_state.setdefault("tree_verify_graph_entries", {}),
            token_state.setdefault("tree_verify_graph_order", []),
            token_state.setdefault("tree_verify_graph_failed_keys", set()),
        )
    store = getattr(model, "_eagle3_tree_verify_graph_entries", None)
    if not isinstance(store, dict):
        store = {}
        try:
            model._eagle3_tree_verify_graph_entries = store
        except Exception:
            pass
    order = getattr(model, "_eagle3_tree_verify_graph_order", None)
    if not isinstance(order, list):
        order = []
        try:
            model._eagle3_tree_verify_graph_order = order
        except Exception:
            pass
    failed = getattr(model, "_eagle3_tree_verify_graph_failed_keys", None)
    if not isinstance(failed, set):
        failed = set()
        try:
            model._eagle3_tree_verify_graph_failed_keys = failed
        except Exception:
            pass
    return store, order, failed


def _tree_verify_graph_failed_reason_map(model, token_state: dict):
    if os.getenv("AICAS_EAGLE3_TREE_VERIFY_GRAPH_GLOBAL_CACHE", "1") != "1":
        return token_state.setdefault("tree_verify_graph_failed_reasons", {})
    reasons = getattr(model, "_eagle3_tree_verify_graph_failed_reasons", None)
    if not isinstance(reasons, dict):
        reasons = {}
        try:
            model._eagle3_tree_verify_graph_failed_reasons = reasons
        except Exception:
            pass
    return reasons


def _tree_verify_graph_clone_past(tree_past, seq_len: int):
    cloned = _clone_cache_to_len(tree_past, int(seq_len))
    _clear_static_cache_tail(cloned, int(seq_len))
    _set_eagle3_cache_seq_len(cloned, int(seq_len))
    return cloned


def _tree_verify_graph_zero_cache_positions_(cache, cache_position: torch.Tensor | None) -> None:
    if not isinstance(cache_position, torch.Tensor) or cache_position.numel() == 0:
        return
    layers = getattr(cache, "layers", None)
    if not layers:
        return
    pos = cache_position.reshape(-1).to(dtype=torch.long)
    for layer in layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, torch.Tensor) and keys.ndim == 4:
            keys.index_fill_(2, pos, 0)
        if isinstance(values, torch.Tensor) and values.ndim == 4:
            values.index_fill_(2, pos, 0)


def _tree_verify_graph_cache_equal(a, b, seq_len: int) -> bool:
    a_layers = getattr(a, "layers", None)
    b_layers = getattr(b, "layers", None)
    if not a_layers or not b_layers or len(a_layers) != len(b_layers):
        return False
    seq_len = int(seq_len)
    for a_layer, b_layer in zip(a_layers, b_layers):
        a_k = getattr(a_layer, "keys", None)
        a_v = getattr(a_layer, "values", None)
        b_k = getattr(b_layer, "keys", None)
        b_v = getattr(b_layer, "values", None)
        if (
            not isinstance(a_k, torch.Tensor)
            or not isinstance(a_v, torch.Tensor)
            or not isinstance(b_k, torch.Tensor)
            or not isinstance(b_v, torch.Tensor)
            or int(a_k.shape[2]) < seq_len
            or int(a_v.shape[2]) < seq_len
            or int(b_k.shape[2]) < seq_len
            or int(b_v.shape[2]) < seq_len
        ):
            return False
        if not torch.equal(a_k[:, :, :seq_len, :], b_k[:, :, :seq_len, :]):
            return False
        if not torch.equal(a_v[:, :, :seq_len, :], b_v[:, :, :seq_len, :]):
            return False
    return True


def _tree_verify_graph_static_prefix(model, src_cache, seq_len: int, max_cache_len: int):
    from transformers.cache_utils import StaticCache

    static_kv = StaticCache(config=model.config, max_cache_len=int(max_cache_len))
    _copy_cache_into_static(src_cache, static_kv, seq_len=int(seq_len))
    _clear_static_cache_tail(static_kv, int(seq_len))
    _set_eagle3_cache_seq_len(static_kv, int(seq_len))
    return static_kv


def _eagle3_tree_verify_graph(
    model,
    past_kv,
    draft_tokens,
    retrieve_indices,
    tree_mask,
    tree_position_ids,
    current_len,
    layer_indices,
    token_state,
):

    if not isinstance(token_state, dict):
        raise RuntimeError("tree verify graph requires dict token_state")
    if torch.is_grad_enabled() or getattr(model, "training", False) or not torch.cuda.is_available():
        raise RuntimeError("tree verify graph requires inference CUDA mode")

    start_pos = _eagle3_cache_seq_len(past_kv)
    q_len = int(draft_tokens.shape[1])
    total_len = int(start_pos) + q_len
    max_cache_len = _tree_verify_graph_select_cache_len(total_len)
    device = draft_tokens.device
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    dtype = next(model.parameters()).dtype
    graph_attn_backend = os.getenv("AICAS_SPEC_TREE_ATTN_BACKEND", "flashdecode_tree").strip().lower()
    graph_attn_backend = {
        "flashdecode": "flashdecode_tree",
        "fd": "flashdecode_tree",
        "triton": "triton_tree",
    }.get(graph_attn_backend, graph_attn_backend)     # TODO: only remain best impl
    key = (q_len, int(max_cache_len), int(device_index), dtype, graph_attn_backend)
    store, order, failed_keys = _tree_verify_graph_state(model, token_state)
    failed_reasons = _tree_verify_graph_failed_reason_map(model, token_state)
    if key in failed_keys:
        _graph_counter_inc("tree_verify_graph_failed_key")
        reason = failed_reasons.get(key, "unknown")
        _trace_once(
            f"eagle3_tree_verify_graph_failed_key_fallback_{abs(hash(key))}",
            f"[VLMModel][eagle3] tree verify graph key previously failed, fallback to eager: "
            f"q={q_len} mcl={int(max_cache_len)} dev={int(device_index)} dtype={dtype} "
            f"start={int(start_pos)} total={int(total_len)} reason={reason}",
        )
        return _eagle3_tree_decoding(
            model,
            past_kv,
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            current_len,
            layer_indices,
            token_state,
        )
    # store saves graphs
    entry = store.get(key)

    if entry is None:
        try:
            draft_buf = draft_tokens.detach().clone()
            tree_mask_buf = tree_mask.detach().clone()
            tree_pos_buf = tree_position_ids.detach().clone()
            cache_position_buf = torch.arange(
                int(start_pos),
                int(start_pos) + q_len,
                device=device,
                dtype=torch.long,
            )
            attention_mask_buf = _eagle3_tree_attention_mask(
                dtype,
                device,
                int(start_pos),
                tree_mask_buf,
                total_kv_len=int(max_cache_len),
            )
            position_ids_buf = _eagle3_tree_position_ids(
                model,
                int(current_len),
                tree_pos_buf,
                batch_size=1,
            )
            static_kv = _tree_verify_graph_static_prefix(model, past_kv, int(start_pos), int(max_cache_len))

            capture_token_state = dict(token_state)
            capture_token_state.pop("tree_lm_head_workspace", None)
            capture_token_state.pop("tree_graph_lm_head_workspace", None)
            capture_token_state.pop("tree_graph_lm_head_workspace_embed", None)
            _trace_once(
                "eagle3_tree_verify_cudagraph_warmup",
                "[eagle3-tree] verify cudagraph warmup",
            )
            with decode_qk_rotary_graph_linear_mode(True):
                warm_tokens, warm_feature, warm_hidden, warm_past = _eagle3_tree_core_forward(
                    model,
                    static_kv,
                    draft_buf,
                    attention_mask_buf,
                    position_ids_buf,
                    cache_position_buf,
                    layer_indices,
                    capture_token_state,
                    workspace_key="tree_graph_lm_head_workspace",
                )
            out_tokens_buf = torch.empty_like(warm_tokens)
            out_feature_buf = torch.empty_like(warm_feature)
            out_hidden_buf = torch.empty_like(warm_hidden)

            _copy_cache_into_static(past_kv, static_kv, seq_len=int(start_pos))
            _clear_static_cache_tail(static_kv, int(start_pos))
            _tree_verify_graph_zero_cache_positions_(static_kv, cache_position_buf)
            _set_eagle3_cache_seq_len(static_kv, int(start_pos))
            with decode_qk_rotary_graph_linear_mode(True):
                _t, _f, _h, _p = _eagle3_tree_core_forward(
                    model,
                    static_kv,
                    draft_buf,
                    attention_mask_buf,
                    position_ids_buf,
                    cache_position_buf,
                    layer_indices,
                    capture_token_state,
                    workspace_key="tree_graph_lm_head_workspace",
                )
            out_tokens_buf.copy_(_t)
            out_feature_buf.copy_(_f)
            out_hidden_buf.copy_(_h)
            import warnings

            graph = torch.cuda.CUDAGraph()
            graph_empty = False
            _copy_cache_into_static(past_kv, static_kv, seq_len=int(start_pos))
            _clear_static_cache_tail(static_kv, int(start_pos))
            _tree_verify_graph_zero_cache_positions_(static_kv, cache_position_buf)
            _set_eagle3_cache_seq_len(static_kv, int(start_pos))
            with torch.no_grad():
                with decode_qk_rotary_graph_linear_mode(True):
                    with warnings.catch_warnings(record=True) as _w:
                        with torch.cuda.graph(graph):
                            graph_tokens, graph_feature, graph_hidden, graph_past = _eagle3_tree_core_forward(
                                model,
                                static_kv,
                                draft_buf,
                                attention_mask_buf,
                                position_ids_buf,
                                cache_position_buf,
                                layer_indices,
                                capture_token_state,
                                workspace_key="tree_graph_lm_head_workspace",
                            )
                            out_tokens_buf.copy_(graph_tokens)
                            out_feature_buf.copy_(graph_feature)
                            out_hidden_buf.copy_(graph_hidden)
                    for _warning in _w:
                        if "CUDA Graph is empty" in str(_warning.message):
                            graph_empty = True
                            break
            if graph_empty:
                raise RuntimeError("tree verify graph capture is empty — Triton autotuning likely fired during capture")
            entry = {
                "graph": graph,
                "draft_tokens": draft_buf,
                "tree_mask": tree_mask_buf,
                "tree_position_ids": tree_pos_buf,
                "cache_position": cache_position_buf,
                "attention_mask": attention_mask_buf,
                "position_ids": position_ids_buf,
                "static_kv": static_kv,
                "out_tokens": out_tokens_buf,
                "out_feature": out_feature_buf,
                "out_hidden": out_hidden_buf,
                "graph_tokens": graph_tokens,
                "graph_feature": graph_feature,
                "graph_hidden": graph_hidden,
                "graph_past": graph_past,
                "start_pos": int(start_pos),
                "current_len": int(current_len),
                "max_cache_len": int(max_cache_len),
                "workspace": capture_token_state.get("tree_graph_lm_head_workspace"),
                "embed_workspace": capture_token_state.get("tree_graph_lm_head_workspace_embed"),
            }
            _tree_verify_graph_put_lru(
                store,
                order,
                key,
                entry,
                max_entries=_tree_verify_graph_lru_capacity(),
            )
            _graph_counter_inc("tree_verify_graph_capture")
            _trace_once(
                f"eagle3_tree_verify_graph_captured_{q_len}_{int(max_cache_len)}_{int(device_index)}_{graph_attn_backend}",
                f"[eagle3] tree verify graph captured: q={q_len} mcl={int(max_cache_len)}",
            )
        except Exception as exc:
            _graph_counter_inc("tree_verify_graph_capture_failed")
            store.pop(key, None)
            failed_keys.add(key)
            failed_reasons[key] = (
                f"{type(exc).__name__}: {exc} "
                f"q={q_len} start={int(start_pos)} total={int(total_len)} "
                f"mcl={int(max_cache_len)} dev={int(device_index)} dtype={dtype} "
                f"backend={graph_attn_backend} draft={tuple(draft_tokens.shape)} "
                f"tree_mask={tuple(tree_mask.shape) if isinstance(tree_mask, torch.Tensor) else None} "
                f"tree_pos={tuple(tree_position_ids.shape) if isinstance(tree_position_ids, torch.Tensor) else None}\n"
                f"{traceback.format_exc()}"
            )
            _trace_once(
                f"eagle3_tree_verify_graph_capture_failed_{abs(hash(key))}",
                f"[VLMModel][eagle3] tree verify graph capture failed, fallback to eager: {failed_reasons[key]}",
            )
            return _eagle3_tree_decoding(
                model,
                past_kv,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                current_len,
                layer_indices,
                token_state,
            )

    if int(entry.get("start_pos", -1)) != int(start_pos):
        entry["cache_position"].copy_(
            torch.arange(int(start_pos), int(start_pos) + q_len, device=device, dtype=torch.long)
        )
        entry["start_pos"] = int(start_pos)
    # tree_position_ids is a CUDA tensor and torch.equal() would force a sync.
    # The buffer is tiny and already has to be refreshed for replay anyway, so
    # refresh it unconditionally instead of paying a device-side compare each round.
    entry["tree_position_ids"].copy_(tree_position_ids)
    entry["position_ids"].copy_(
        _eagle3_tree_position_ids(model, int(current_len), entry["tree_position_ids"], batch_size=1)
    )
    entry["current_len"] = int(current_len)
    with _profile_range("eagle3.verify.prep"):
        entry["draft_tokens"].copy_(draft_tokens)
        entry["tree_mask"].copy_(tree_mask)
        _fill_eagle3_tree_attention_mask_(entry["attention_mask"], int(start_pos), entry["tree_mask"])
        if past_kv is not entry["static_kv"]:
            _copy_cache_into_static(past_kv, entry["static_kv"], seq_len=int(start_pos))
            _clear_static_cache_tail(entry["static_kv"], int(start_pos))
            _tree_verify_graph_zero_cache_positions_(entry["static_kv"], entry["cache_position"])
        _set_eagle3_cache_seq_len(entry["static_kv"], int(start_pos))

    if entry.get("workspace") is not None:
        token_state["tree_graph_lm_head_workspace"] = entry.get("workspace")
    _graph_counter_inc("tree_verify_graph_hit")
    with _profile_range("eagle3.verify.replay"):
        entry["graph"].replay()
    _set_eagle3_cache_seq_len(entry["static_kv"], int(total_len))

    return (
        entry["out_tokens"].clone(),
        entry["out_feature"].clone(),
        entry["out_hidden"].clone(),
        entry["static_kv"]
        if os.getenv("AICAS_EAGLE3_COMPACT_CACHE_SELECT", "1") == "1"
        else _tree_verify_graph_clone_past(entry["static_kv"], int(total_len)),
    )


def _eagle3_full_graph_generate_verify(
    model,
    draft_model,
    past_kv,
    draft_feature: torch.Tensor,
    draft_input_ids: torch.Tensor,
    current_len: int,
    layer_indices: tuple[int, int, int],
    token_state: dict,
):
    if os.getenv("AICAS_EAGLE3_FULL_GRAPH", "0") != "1":
        return None
    if (
        run_eagle3_expand_topk4 is None
        or run_eagle3_draft_tree_select_metadata is None
        or run_eagle3_full_verify_prep is None
        or getattr(model, "training", False)
        or getattr(draft_model, "training", False)
    ):
        _graph_counter_inc("full_graph_not_eligible")
        return None
    if not isinstance(token_state, dict):
        _graph_counter_inc("full_graph_not_eligible")
        return None
    if not isinstance(draft_feature, torch.Tensor) or not isinstance(draft_input_ids, torch.Tensor):
        _graph_counter_inc("full_graph_not_eligible")
        return None
    if not draft_feature.is_cuda or not draft_input_ids.is_cuda:
        _graph_counter_inc("full_graph_not_eligible")
        return None
    if not hasattr(draft_model, "tree_mask_init"):
        draft_model.init_tree()
    stable_before = getattr(draft_model, "_stable_kv", None)
    if stable_before is None:
        _graph_counter_inc("full_graph_no_draft_past")
        return None
    if int(draft_model.top_k) != 4:
        _graph_counter_inc("full_graph_not_eligible")
        return None

    input_ids_full = draft_input_ids.to(device=draft_feature.device, dtype=torch.long).contiguous()
    if input_ids_full.ndim != 2 or int(input_ids_full.shape[0]) != 1 or int(input_ids_full.shape[1]) < 2:
        _graph_counter_inc("full_graph_not_eligible")
        return None
    shifted_input_ids = input_ids_full[:, 1:].contiguous()
    hidden_len = int(draft_feature.shape[1]) if draft_feature.dim() >= 3 else 1
    draft_kv_len = draft_model._kv_seq_len(stable_before[0] if isinstance(stable_before, tuple) else stable_before)
    suffix_len = max(0, int(shifted_input_ids.shape[1]) - int(draft_kv_len))
    if suffix_len != hidden_len:
        _graph_counter_inc("full_graph_not_eligible")
        return None

    prefill_input_ids = shifted_input_ids[:, draft_kv_len:].contiguous()
    draft_pair = draft_model._draft_step_graph_past_pair(stable_before)
    if draft_pair is None:
        _graph_counter_inc("full_graph_no_draft_past")
        return None
    draft_past_k, draft_past_v = draft_pair
    draft_past_len = int(draft_past_k.shape[2])
    buf_len = _eagle3_full_graph_buffer_len(draft_model, draft_past_len)
    depth = int(draft_model.depth)
    top_k = int(draft_model.top_k)
    total_tokens = int(draft_model.total_tokens)
    q_len = int(total_tokens) + 1

    start_pos = int(_eagle3_cache_seq_len(past_kv))
    # ── hard lower bound, then bucketize so the key stays stable ──
    required_len = max(
        int(buf_len),
        int(start_pos) + q_len,
        int(draft_past_len) + int(suffix_len) + int(depth) * int(top_k),
    )
    buf_len = int(_tree_verify_graph_select_cache_len(required_len))
    device = draft_feature.device
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    dtype = next(model.parameters()).dtype
    graph_attn_backend = os.getenv("AICAS_SPEC_TREE_ATTN_BACKEND", "flashdecode_tree").strip().lower()
    graph_attn_backend = {
        "flashdecode": "flashdecode_tree",
        "fd": "flashdecode_tree",
        "triton": "triton_tree",
    }.get(graph_attn_backend, graph_attn_backend)
    key = (
        int(device_index),
        tuple(draft_feature.shape),
        str(draft_feature.dtype),
        tuple(prefill_input_ids.shape),
        str(prefill_input_ids.dtype),
        int(buf_len),
        int(depth),
        int(top_k),
        int(total_tokens),
        int(q_len),
        dtype,
        graph_attn_backend,
    )
    store, order, failed_keys = _eagle3_full_graph_state(model, token_state)
    failed_reasons = _eagle3_full_graph_failed_reason_map(model, token_state)
    if key in failed_keys:
        _graph_counter_inc("full_graph_failed_key")
        _graph_reason_set("full_graph", failed_reasons.get(key, "unknown"))
        _trace_once(
            f"eagle3_full_graph_failed_key_{abs(hash(key))}",
            f"[eagle3-full] full graph key previously failed: {failed_reasons.get(key, 'unknown')}",
        )
        return None

    entry = store.get(key)
    if entry is None:
        allow_full_capture = (
            os.getenv("AICAS_EAGLE3_SPEC_PRECAPTURE_ACTIVE", "0") == "1"
            or os.getenv("AICAS_EAGLE3_SPEC_PRECAPTURE_DONE", "0") != "1"
            or os.getenv("AICAS_EAGLE3_FULL_GRAPH_CAPTURE_ON_MISS", "0") == "1"
        )
        if not allow_full_capture:
            _graph_counter_inc("full_graph_miss_after_precapture")
            return None
        try:
            feature_buf = draft_feature.detach().clone()
            input_buf = prefill_input_ids.detach().clone()
            sample_token_buf = input_ids_full[:, -1].reshape(-1).detach().clone()
            draft_position_buf = draft_model._make_draft_position_ids(input_buf, draft_past_len, None).detach().clone()
            draft_past_k_buf = draft_past_k.new_empty((int(draft_past_k.shape[0]), int(draft_past_k.shape[1]), int(buf_len), int(draft_past_k.shape[3])))
            draft_past_v_buf = draft_past_v.new_empty((int(draft_past_v.shape[0]), int(draft_past_v.shape[1]), int(buf_len), int(draft_past_v.shape[3])))
            draft_model._copy_draft_past_into_bucket(draft_past_k_buf, draft_past_v_buf, draft_past_k, draft_past_v)
            draft_past_k_buf[:, :, int(draft_past_len):, :].zero_()
            draft_past_v_buf[:, :, int(draft_past_len):, :].zero_()
            draft_static_past = ((draft_past_k_buf, draft_past_v_buf),)
            draft_logical_past_bufs = []
            draft_logical_offset = int(prefill_input_ids.shape[1])
            draft_logical_past_bufs.append(torch.tensor(draft_past_len, device=device, dtype=torch.long))
            for _ in range(depth):
                draft_logical_past_bufs.append(
                    torch.tensor(draft_past_len + draft_logical_offset, device=device, dtype=torch.long)
                )
                draft_logical_offset += int(top_k)
            tree_mask_init_buf = draft_model.tree_mask_init.detach().clone()
            start_pos_buf = torch.tensor(start_pos, device=device, dtype=torch.long)
            static_kv = _tree_verify_graph_static_prefix(model, past_kv, int(start_pos), int(buf_len))
            attention_mask_buf = torch.empty((1, 1, q_len, int(buf_len)), device=device, dtype=dtype)
            cache_position_buf = torch.empty((q_len,), device=device, dtype=torch.long)
            position_ids_buf = torch.empty((4, 1, q_len), device=device, dtype=torch.long)
            out_tokens_buf = torch.empty((q_len,), device=device, dtype=torch.long)
            out_feature_buf = torch.empty((1, q_len, int(draft_feature.shape[-1])), device=device, dtype=draft_feature.dtype)
            language_model = getattr(getattr(model, "model", None), "language_model", None)
            lm_config = getattr(language_model, "config", None)
            hidden_size = int(getattr(lm_config, "hidden_size", getattr(model.config, "hidden_size", draft_feature.shape[-1])))
            out_hidden_buf = torch.empty((1, q_len, hidden_size), device=device, dtype=dtype)
            draft_tree_ws = {
                "sample_token": torch.empty((1,), device=device, dtype=torch.long),
                "root_tokens": torch.empty((1, top_k), device=device, dtype=torch.long),
                "root_scores": torch.empty((top_k,), device=device, dtype=torch.float32),
                "topk_ids_slots": torch.empty((depth, top_k, top_k), device=device, dtype=torch.long),
                "tree_select": None,
            }
            expand_holder = {"workspace": None}
            capture_token_state = dict(token_state)
            capture_token_state.pop("tree_lm_head_workspace", None)
            capture_token_state.pop("tree_graph_lm_head_workspace", None)
            capture_token_state.pop("tree_graph_lm_head_workspace_embed", None)
            stable_kv_src = None

            rope_delta_buf = torch.empty((1,), device=device, dtype=torch.long)
            _copy_eagle3_rope_delta_(rope_delta_buf, model)

            def _run_full_body():
                nonlocal stable_kv_src
                draft_model.tree_mask = None
                draft_model._draft_graph_logical_past_len = draft_logical_past_bufs[0]
                draft_model._draft_graph_storage_past_len = int(buf_len)
                out_hidden, draft_past, topk_index, topk_p = draft_model._draft_step_eager(
                    feature_buf,
                    input_buf,
                    draft_static_past,
                    position_ids=draft_position_buf,
                    logits_processor=None,
                    topk_last_only=True,
                )
                stable_kv_src = draft_past
                root_scores = topk_p[0]
                draft_tree_ws["sample_token"].copy_(sample_token_buf)
                draft_tree_ws["root_tokens"].copy_(topk_index.reshape(1, top_k))
                draft_tree_ws["root_scores"].copy_(root_scores.reshape(top_k))
                last_hidden = out_hidden[:, -1]
                scores = root_scores
                input_hidden = last_hidden[None].repeat(1, top_k, 1)
                step_input_ids = topk_index
                tree_mask = tree_mask_init_buf
                draft_storage_past_len = int(buf_len) + int(prefill_input_ids.shape[1])
                for step in range(depth):
                    draft_model.tree_mask = tree_mask
                    draft_model._draft_graph_logical_past_len = draft_logical_past_bufs[step + 1]
                    draft_model._draft_graph_storage_past_len = int(draft_storage_past_len)
                    step_position_ids = int(shifted_input_ids.shape[1]) + step + draft_model.position_ids
                    step_hidden, draft_past, step_topk, step_topk_p = draft_model._draft_step_eager(
                        input_hidden,
                        step_input_ids,
                        draft_past,
                        position_ids=step_position_ids,
                        logits_processor=None,
                        topk_last_only=False,
                    )
                    result, workspace = run_eagle3_expand_topk4(
                        step_topk,
                        step_topk_p,
                        scores,
                        step_hidden,
                        tree_mask,
                        workspace=expand_holder["workspace"],
                        score_slot=step,
                        score_slots=depth,
                    )
                    expand_holder["workspace"] = workspace
                    if result is None:
                        raise RuntimeError("full graph expand_topk rejected inputs")
                    draft_storage_past_len += int(step_input_ids.shape[1])
                    draft_tree_ws["topk_ids_slots"][step].copy_(step_topk)
                    _cu_scores, _topk_cs_index, scores, _out_ids, input_hidden, step_input_ids, tree_mask = result
                expand_workspace = expand_holder["workspace"]
                if not isinstance(expand_workspace, dict):
                    raise RuntimeError("full graph missing expand workspace")
                tree_result, tree_workspace = run_eagle3_draft_tree_select_metadata(
                    draft_tree_ws["sample_token"],
                    draft_tree_ws["root_tokens"],
                    draft_tree_ws["root_scores"],
                    expand_workspace["cu_scores"],
                    draft_tree_ws["topk_ids_slots"],
                    expand_workspace["topk_cs_index_slots"],
                    total_tokens=int(total_tokens),
                    depth=int(depth),
                    max_depth=int(depth) + 2,
                    workspace=draft_tree_ws.get("tree_select"),
                )
                draft_tree_ws["tree_select"] = tree_workspace
                if tree_result is None:
                    raise RuntimeError("full graph tree_select rejected inputs")
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids = tree_result
                ok = run_eagle3_full_verify_prep(
                    tree_mask,
                    tree_position_ids,
                    start_pos_buf,
                    attention_mask_buf,
                    cache_position_buf,
                    position_ids_buf,
                    rope_delta=rope_delta_buf,
                )
                if not ok:
                    raise RuntimeError("full graph verify prep rejected inputs")
                target_tokens, feature, hidden, tree_past = _eagle3_tree_core_forward(
                    model,
                    static_kv,
                    draft_tokens,
                    attention_mask_buf,
                    position_ids_buf,
                    cache_position_buf,
                    layer_indices,
                    capture_token_state,
                    workspace_key="full_graph_lm_head_workspace",
                )
                out_tokens_buf.copy_(target_tokens)
                out_feature_buf.copy_(feature)
                out_hidden_buf.copy_(hidden)
                return draft_tokens, retrieve_indices, tree_mask, tree_position_ids, tree_past

            old_tree_mask = getattr(draft_model, "tree_mask", None)
            old_force_eager_attention = getattr(draft_model.midlayer.self_attn, "_aicas_draft_force_eager_attention", _EAGLE3_DEBUG_ATTR_MISSING)
            graph_result = None
            try:
                full_graph_blas = os.getenv("AICAS_EAGLE3_FULL_GRAPH_BLAS_BACKEND", "cublas")
                with _temporary_preferred_blas(full_graph_blas):
                    draft_model.midlayer.self_attn._aicas_draft_force_eager_attention = True
                    draft_model._draft_graph_logical_past_len = draft_logical_past_bufs[0]
                    draft_model._draft_graph_storage_past_len = int(buf_len)
                    with _draft_graph_attention_mode():
                        with decode_qk_rotary_graph_linear_mode(True):
                            # Warmup with compiled modules so Dynamo has cached traces
                            # before capture.  We keep compiled modules active during
                            # warmup, then revert to eager only during the actual graph
                            # capture to avoid Dynamo recompilation errors inside the
                            # cuda graph region.
                            _run_full_body()
                            _run_full_body()
                    _copy_cache_into_static(past_kv, static_kv, seq_len=int(start_pos))
                    _clear_static_cache_tail(static_kv, int(start_pos))
                    _tree_verify_graph_zero_cache_positions_(static_kv, cache_position_buf)
                    _set_eagle3_cache_seq_len(static_kv, int(start_pos))
                    import warnings

                    graph = torch.cuda.CUDAGraph()
                    graph_empty = False
                    with torch.no_grad():
                        with _draft_graph_attention_mode():
                            with decode_qk_rotary_graph_linear_mode(True):
                                with _disable_compiled_submodules_for_graph(model, draft_model):
                                    with warnings.catch_warnings(record=True) as _w:
                                        with torch.cuda.graph(graph):
                                            graph_result = _run_full_body()
                                    for _warning in _w:
                                        if "CUDA Graph is empty" in str(_warning.message):
                                            graph_empty = True
                                            break
                    if graph_empty:
                        raise RuntimeError("full graph capture is empty")
            finally:
                draft_model.tree_mask = old_tree_mask
                if old_force_eager_attention is _EAGLE3_DEBUG_ATTR_MISSING:
                    try:
                        delattr(draft_model.midlayer.self_attn, "_aicas_draft_force_eager_attention")
                    except Exception:
                        pass
                else:
                    draft_model.midlayer.self_attn._aicas_draft_force_eager_attention = old_force_eager_attention
                draft_model._draft_graph_logical_past_len = None
                draft_model._draft_graph_storage_past_len = None
            if graph_result is None:
                raise RuntimeError("full graph produced no result")
            entry = {
                "graph": graph,
                "feature": feature_buf,
                "input_ids": input_buf,
                "sample_token": sample_token_buf,
                "draft_position_ids": draft_position_buf,
                "draft_past_k": draft_past_k_buf,
                "draft_past_v": draft_past_v_buf,
                "draft_logical_past_len": draft_logical_past_bufs[0],
                "draft_logical_past_lens": draft_logical_past_bufs,
                "draft_bucket_len": int(buf_len),
                "stable_kv_src": stable_kv_src,
                "start_pos": start_pos_buf,
                "static_kv": static_kv,
                "attention_mask": attention_mask_buf,
                "cache_position": cache_position_buf,
                "position_ids": position_ids_buf,
                "rope_delta": rope_delta_buf,
                "out_tokens": out_tokens_buf,
                "out_feature": out_feature_buf,
                "out_hidden": out_hidden_buf,
                "draft_tree_workspace": draft_tree_ws,
                "expand_workspace": expand_holder["workspace"],
                "result": graph_result,
                "max_cache_len": int(buf_len),
                "q_len": int(q_len),
            }
            _tree_verify_graph_put_lru(store, order, key, entry, max_entries=_eagle3_full_graph_lru_capacity())
            _graph_counter_inc("full_graph_capture")
            _trace_once(
                f"eagle3_full_graph_captured_{abs(hash(key))}",
                f"[eagle3-full] full graph captured: q={q_len} buf_len={int(buf_len)} "
                f"depth={int(depth)}",
            )
        except Exception as exc:
            if _is_cuda_fatal_exception(exc):
                raise
            _graph_counter_inc("full_graph_capture_failed")
            store.pop(key, None)
            failed_keys.add(key)
            failed_reasons[key] = (
                f"{type(exc).__name__}: {exc} start={int(start_pos)} q={q_len} "
                f"buf_len={int(buf_len)} draft_feature={tuple(draft_feature.shape)} "
                f"draft_input={tuple(prefill_input_ids.shape)} depth={int(depth)} top_k={int(top_k)}\n"
                f"{traceback.format_exc()}"
            )
            _graph_reason_set("full_graph", failed_reasons[key])
            _trace_once(
                f"eagle3_full_graph_capture_failed_{abs(hash(key))}",
                f"[eagle3-full] full graph capture failed: {failed_reasons[key]}",
            )
            return None

    entry["feature"].copy_(draft_feature)
    entry["input_ids"].copy_(prefill_input_ids)
    entry["sample_token"].copy_(input_ids_full[:, -1].reshape(-1))
    entry["draft_position_ids"].copy_(draft_model._make_draft_position_ids(prefill_input_ids, draft_past_len, None))
    draft_model._copy_draft_past_into_bucket(entry["draft_past_k"], entry["draft_past_v"], draft_past_k, draft_past_v)
    # Zero tail beyond draft_past_len so softmax never reads uninitialised memory.
    entry["draft_past_k"][:, :, int(draft_past_len):, :].zero_()
    entry["draft_past_v"][:, :, int(draft_past_len):, :].zero_()
    entry["draft_logical_past_len"].fill_(int(draft_past_len))
    draft_logical_past_lens = entry.get("draft_logical_past_lens")
    if isinstance(draft_logical_past_lens, list):
        draft_logical_offset = int(prefill_input_ids.shape[1])
        for idx, buf in enumerate(draft_logical_past_lens[1:]):
            if isinstance(buf, torch.Tensor):
                buf.fill_(int(draft_past_len) + int(draft_logical_offset))
            draft_logical_offset += int(top_k)
    entry["start_pos"].fill_(int(start_pos))
    _copy_eagle3_rope_delta_(entry["rope_delta"], model)
    entry["cache_position"].copy_(
        torch.arange(int(start_pos), int(start_pos) + q_len, device=device, dtype=torch.long)
    )
    with _profile_range("eagle3.full_graph.prep"):
        # Skip the copy when past_kv is already the same cache object
        # (happens from round 2+ — _select_tree_cache_path makes them alias)
        if past_kv is not entry["static_kv"]:
            _copy_cache_into_static(past_kv, entry["static_kv"], seq_len=int(start_pos))
        _clear_static_cache_tail(entry["static_kv"], int(start_pos))
        _tree_verify_graph_zero_cache_positions_(entry["static_kv"], entry["cache_position"])
        _set_eagle3_cache_seq_len(entry["static_kv"], int(start_pos))

    old_tree_mask = getattr(draft_model, "tree_mask", None)
    old_force_eager_attention = getattr(draft_model.midlayer.self_attn, "_aicas_draft_force_eager_attention", _EAGLE3_DEBUG_ATTR_MISSING)
    try:
        draft_model.midlayer.self_attn._aicas_draft_force_eager_attention = True
        draft_model._draft_graph_logical_past_len = entry["draft_logical_past_len"]
        draft_model._draft_graph_storage_past_len = int(entry["draft_bucket_len"])
        _graph_counter_inc("full_graph_hit")
        with _profile_range("eagle3.full_graph.replay"):
            entry["graph"].replay()
    except Exception as exc:
        if _is_cuda_fatal_exception(exc):
            raise
        _graph_counter_inc("full_graph_replay_fail")
        store.pop(key, None)
        failed_keys.add(key)
        failed_reasons[key] = (
            f"{type(exc).__name__}: {exc} start={int(start_pos)} q={q_len} "
            f"buf_len={int(buf_len)} draft_feature={tuple(draft_feature.shape)} "
            f"draft_input={tuple(prefill_input_ids.shape)} depth={int(depth)} top_k={int(top_k)}\n"
            f"{traceback.format_exc()}"
        )
        _graph_reason_set("full_graph", failed_reasons[key])
        _trace_once(
            f"eagle3_full_graph_replay_failed_{abs(hash(key))}",
            f"[eagle3-full] full graph replay failed: {failed_reasons[key]}",
        )
        return None
    finally:
        draft_model.tree_mask = old_tree_mask
        if old_force_eager_attention is _EAGLE3_DEBUG_ATTR_MISSING:
            try:
                delattr(draft_model.midlayer.self_attn, "_aicas_draft_force_eager_attention")
            except Exception:
                pass
        else:
            draft_model.midlayer.self_attn._aicas_draft_force_eager_attention = old_force_eager_attention
        draft_model._draft_graph_logical_past_len = None
        draft_model._draft_graph_storage_past_len = None

    stable_pair = draft_model._draft_step_graph_past_pair(entry.get("stable_kv_src"))
    if stable_pair is not None:
        ret_k, ret_v = draft_model._repack_draft_graph_past(
            {
                "past_bucket_len": entry["draft_bucket_len"],
                "past_k": entry["draft_past_k"],
                "past_v": entry["draft_past_v"],
                "out_k": stable_pair[0],
                "out_v": stable_pair[1],
            },
            int(draft_past_len),
            int(prefill_input_ids.shape[1]),
        )
        draft_model._stable_kv = ((ret_k, ret_v),)
    _set_eagle3_cache_seq_len(entry["static_kv"], int(start_pos) + q_len)
    draft_tokens, retrieve_indices, tree_mask, tree_position_ids, _tree_past = entry["result"]
    return (
        entry["out_tokens"].clone(),
        entry["out_feature"].clone(),
        entry["out_hidden"].clone(),
        entry["static_kv"],
        draft_tokens,
        retrieve_indices,
        tree_mask,
        tree_position_ids,
    )


def _select_tree_cache_path(tree_past, start_pos: int, node_indices):
    layers = getattr(tree_past, "layers", None)
    if not layers:
        return None
    first_k = getattr(layers[0], "keys", None)
    if not isinstance(first_k, torch.Tensor):
        return None
    if isinstance(node_indices, torch.Tensor):
        node_tensor = node_indices.to(device=first_k.device, dtype=torch.long).reshape(-1)
    else:
        node_tensor = torch.tensor(node_indices, device=first_k.device, dtype=torch.long).reshape(-1)
    compact_static = os.getenv("AICAS_EAGLE3_COMPACT_CACHE_SELECT", "1") == "1"
    if compact_static and any(hasattr(layer, "max_cache_len") for layer in layers):
        try:
            append_count = int(node_tensor.numel())
            new_len = int(start_pos) + append_count
            if new_len <= 0:
                return None
            selected = node_tensor + int(start_pos)
            try:
                previous_len = int(getattr(tree_past, "_aicas_eagle3_seq_len", new_len))
            except Exception:
                previous_len = new_len
            use_compact_fused = (
                os.getenv("AICAS_EAGLE3_COMPACT_CACHE_SELECT_FUSED", "1") == "1"
                and run_eagle3_cache_select_compact is not None
                and can_use_eagle3_cache_select_compact is not None
            )
            fused_ok = False
            if use_compact_fused:
                try:
                    # Pre-compute layer data arrays to avoid repeated getattr
                    layer_kv = []
                    for layer in layers:
                        k = getattr(layer, "keys", None)
                        v = getattr(layer, "values", None)
                        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                            return None
                        k_len = int(k.shape[2])
                        if k_len < int(new_len) or int(v.shape[2]) < int(new_len):
                            return None
                        layer_kv.append((k, v, layer))
                    # Validate all layers can use compact fused kernel
                    fused_ok = True
                    for k, v, _ in layer_kv:
                        if not can_use_eagle3_cache_select_compact(
                            k, v, node_tensor,
                            start_pos=int(start_pos),
                            previous_len=int(previous_len),
                        ):
                            fused_ok = False
                            break
                    if not fused_ok:
                        raise RuntimeError("compact fused cache_select rejected inputs")
                    # Execute all layers with pre-computed data
                    for k, v, layer in layer_kv:
                        if not run_eagle3_cache_select_compact(
                            k, v, node_tensor,
                            start_pos=int(start_pos),
                            previous_len=int(previous_len),
                        ):
                            fused_ok = False
                            break
                        if hasattr(layer, "cumulative_length"):
                            try:
                                layer.cumulative_length = new_len
                            except Exception:
                                pass
                    if fused_ok:
                        _set_eagle3_cache_seq_len(tree_past, new_len)
                        try:
                            tree_past._aicas_eagle3_tail_cleared_len = int(new_len)
                        except Exception:
                            pass
                        _graph_counter_inc("cache_select_compact_static_fused_hit")
                        _trace_once("eagle3_compact_cache_select_fused", "[eagle3-tree] compact fused cache_select hit")
                        return tree_past
                except Exception:
                    if os.getenv("AICAS_EAGLE3_COMPACT_CACHE_SELECT_FUSED_REQUIRE", "0") == "1":
                        raise
                    fused_ok = False

            for layer in layers:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                    return None
                if int(keys.shape[2]) < int(new_len) or int(values.shape[2]) < int(new_len):
                    return None
                if append_count > 0:
                    key_selected = keys.index_select(2, selected).contiguous()
                    value_selected = values.index_select(2, selected).contiguous()
                    keys[:, :, int(start_pos):new_len, :].copy_(key_selected)
                    values[:, :, int(start_pos):new_len, :].copy_(value_selected)
                clear_to = min(int(keys.shape[2]), max(int(new_len), int(previous_len)))
                if int(clear_to) > int(new_len):
                    keys[:, :, int(new_len):int(clear_to), :].zero_()
                    values[:, :, int(new_len):int(clear_to), :].zero_()
                if hasattr(layer, "cumulative_length"):
                    try:
                        layer.cumulative_length = new_len
                    except Exception:
                        pass
            _set_eagle3_cache_seq_len(tree_past, new_len)
            try:
                tree_past._aicas_eagle3_tail_cleared_len = int(new_len)
            except Exception:
                pass
            _graph_counter_inc("cache_select_compact_static_hit")
            _trace_once("eagle3_compact_cache_select", "[eagle3-tree] compact cache_select hit")
            return tree_past
        except Exception as exc:
            _graph_counter_inc("cache_select_compact_static_fallback")
            if os.getenv("AICAS_EAGLE3_COMPACT_CACHE_SELECT_REQUIRE", "0") == "1":
                raise
            _trace_once(
                "eagle3_compact_cache_select_fallback",
                f"[eagle3-tree] compact cache_select fallback: {type(exc).__name__}: {exc}",
            )
    if any(hasattr(layer, "max_cache_len") for layer in layers):
        selected = node_tensor + int(start_pos)
        append_count = int(selected.numel())
        new_len = int(start_pos) + append_count
        use_fused = os.getenv("AICAS_EAGLE3_FUSED_CACHE_SELECT", "1") == "1"
        if use_fused and run_eagle3_cache_select is not None:
            try:
                for layer in layers:
                    keys = getattr(layer, "keys", None)
                    values = getattr(layer, "values", None)
                    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                        return None
                    if not run_eagle3_cache_select(keys, values, node_tensor, start_pos=int(start_pos)):
                        raise RuntimeError("cache select kernel rejected inputs")
                    if hasattr(layer, "cumulative_length"):
                        try:
                            layer.cumulative_length = new_len
                        except Exception:
                            pass
                _set_eagle3_cache_seq_len(tree_past, new_len)
                _trace_once("eagle3_fused_cache_select", "[eagle3-tree] fused cache_select hit")
                return tree_past
            except Exception as exc:
                if os.getenv("AICAS_EAGLE3_FUSED_CACHE_SELECT_REQUIRE", "0") == "1":
                    raise
                _trace_once(
                    "eagle3_fused_cache_select_fallback",
                    f"[eagle3-tree] fused cache_select fallback: {type(exc).__name__}: {exc}",
                )

        dest = torch.arange(
            int(start_pos),
            int(start_pos) + append_count,
            device=first_k.device,
            dtype=torch.long,
        )
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                return None
            key_selected = keys.index_select(2, selected).contiguous()
            value_selected = values.index_select(2, selected).contiguous()
            keys.index_copy_(2, dest, key_selected)
            values.index_copy_(2, dest, value_selected)
            if int(keys.shape[2]) > new_len:
                keys[:, :, new_len:, :].zero_()
            if int(values.shape[2]) > new_len:
                values[:, :, new_len:, :].zero_()
            if hasattr(layer, "cumulative_length"):
                try:
                    layer.cumulative_length = new_len
                except Exception:
                    pass
        _set_eagle3_cache_seq_len(tree_past, new_len)
        return tree_past
    if os.getenv("AICAS_EAGLE3_CACHE_SELECT_INPLACE", "0") == "1":
        try:
            selected = node_tensor + int(start_pos)
            append_count = int(selected.numel())
            dest = torch.arange(
                int(start_pos),
                int(start_pos) + append_count,
                device=first_k.device,
                dtype=torch.long,
            )
            new_len = int(start_pos) + append_count
            for layer in layers:
                if hasattr(layer, "max_cache_len"):
                    return None
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                    return None
                key_selected = keys.index_select(-2, selected).contiguous()
                value_selected = values.index_select(-2, selected).contiguous()
                keys.index_copy_(-2, dest, key_selected)
                values.index_copy_(-2, dest, value_selected)
                layer.keys = keys.narrow(-2, 0, new_len)
                layer.values = values.narrow(-2, 0, new_len)
                if hasattr(layer, "cumulative_length"):
                    try:
                        layer.cumulative_length = new_len
                    except Exception:
                        pass
            _set_eagle3_cache_seq_len(tree_past, new_len)
            return tree_past
        except Exception:
            if os.getenv("AICAS_EAGLE3_CACHE_SELECT_INPLACE_REQUIRE", "0") == "1":
                raise
    keep = torch.cat([
        torch.arange(start_pos, device=first_k.device, dtype=torch.long),
        node_tensor + int(start_pos),
    ])
    for layer in layers:
        if hasattr(layer, "max_cache_len"):
            return None
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            return None
        kv_len = int(keys.shape[-2])
        keep_clamped = keep.clamp(0, kv_len - 1)
        if int(keep_clamped.ne(keep).any().item()):
            _debug_keep_bad = keep.detach().cpu().tolist()
            _debug_node_bad = node_tensor.detach().cpu().tolist()
            raise RuntimeError(
                f"_select_tree_cache_path: keep indices out of bounds "
                f"kv_len={kv_len} keep={_debug_keep_bad} node_tensor={_debug_node_bad} "
                f"start_pos={start_pos}"
            )
        layer.keys = keys.index_select(-2, keep).contiguous()
        layer.values = values.index_select(-2, keep).contiguous()
        if hasattr(layer, "cumulative_length"):
            try:
                layer.cumulative_length = int(keep.numel())
            except Exception:
                pass
    _set_eagle3_cache_seq_len(tree_past, int(keep.numel()))
    return tree_past


@contextmanager
def _sdpa_attention_context(model):
    """Temporarily set _attn_implementation='sdpa' on all model configs.

    This routes tree-forward attention through standard SDPA instead of
    FlashDecode or custom multi-token attention.  SDPA dispatches to the
    appropriate backend (FlashAttention for 2D, Math for 4D masks).
    The root position is corrected separately via _canonical_decode_step
    (q_len=1, 2D mask, same backend as baseline).
    """
    cfgs = []
    for cfg in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "model", None), "config", None),
    ):
        if cfg is not None and hasattr(cfg, "_attn_implementation") and cfg not in cfgs:
            cfgs.append(cfg)
    old = [(cfg, getattr(cfg, "_attn_implementation")) for cfg in cfgs]
    try:
        for cfg in cfgs:
            cfg._attn_implementation = "sdpa"
        yield
    finally:
        for cfg, value in old:
            cfg._attn_implementation = value


def _eagle3_tree_decoding(
    model,
    past_kv,
    draft_tokens,
    retrieve_indices,
    tree_mask,
    tree_position_ids,
    current_len,
    layer_indices,
    token_state,
):
    start_pos = _eagle3_cache_seq_len(past_kv)
    dtype = next(model.parameters()).dtype
    device = draft_tokens.device
    cache_position = torch.arange(start_pos, start_pos + int(draft_tokens.shape[1]), device=device, dtype=torch.long)
    _tree_kv_len = None
    if hasattr(past_kv, "layers") and past_kv.layers:
        _tree_keys = getattr(past_kv.layers[0], "keys", None)
        if isinstance(_tree_keys, torch.Tensor) and int(_tree_keys.shape[-2]) > start_pos + int(draft_tokens.shape[1]):
            _tree_kv_len = int(_tree_keys.shape[-2])
    kwargs = {
        "input_ids": draft_tokens,
        "attention_mask": _eagle3_tree_attention_mask(dtype, device, start_pos, tree_mask, total_kv_len=_tree_kv_len),
        "position_ids": _eagle3_tree_position_ids(model, current_len, tree_position_ids, batch_size=1),
        "past_key_values": past_kv,
        "cache_position": cache_position,
        "use_cache": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    use_selected_forward = os.getenv("AICAS_ENABLE_FLASHDECODE_ATTENTION", "0") != "1"
    with torch.no_grad():
        with _profile_range("eagle3_tree.verify.target_forward"):
            with spec_qk_rotary(model):
                if use_selected_forward:
                    hidden, feature, tree_past = _eagle3_selected_text_forward(
                        model,
                        input_ids=draft_tokens,
                        attention_mask=kwargs["attention_mask"],
                        position_ids=kwargs["position_ids"],
                        past_key_values=past_kv,
                        cache_position=cache_position,
                        layer_indices=layer_indices,
                    )
                    outputs = None
                else:
                    with _temporary_model_attr(model, "_aicas_disable_prefill_compile", True):
                        outputs = model.model(**kwargs)
                    hidden = outputs.last_hidden_state
                    feature = _select_eagle3_feature(outputs.hidden_states, layer_indices)
                    tree_past = outputs.past_key_values

    lm_head = model.lm_head
    bias = getattr(lm_head, "bias", None)
    if can_use_lm_head_argmax_triton_multi is None or lm_head_argmax_triton_multi is None:
        raise RuntimeError("EAGLE3 optimized verifier requires lm_head_argmax_triton_multi")
    if not can_use_lm_head_argmax_triton_multi(hidden, lm_head.weight, bias):
        raise RuntimeError(
            "EAGLE3 optimized verifier argmax kernel is not applicable "
            f"hidden={tuple(hidden.shape)} hidden_dtype={hidden.dtype} "
            f"weight_dtype={lm_head.weight.dtype} cuda={hidden.is_cuda}"
        )
    workspace = token_state.get("tree_lm_head_workspace") if isinstance(token_state, dict) else None
    with _profile_range("eagle3_tree.verify.lm_head_argmax"):
        target_tokens, workspace = lm_head_argmax_triton_multi(hidden, lm_head.weight, bias, workspace=workspace)
    if isinstance(token_state, dict):
        token_state["tree_lm_head_workspace"] = workspace
    target_tokens = target_tokens.reshape(-1).to(device=device, dtype=torch.long)

    return target_tokens, feature, hidden, tree_past


def _eagle3_root_lm_head_margin(model, hidden: torch.Tensor, token_state: dict | None = None) -> tuple[float, int, int]:
    if not isinstance(hidden, torch.Tensor) or int(hidden.shape[1]) <= 0:
        return float("inf"), -1, -1
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return float("inf"), -1, -1
    root_hidden = hidden[:, 0:1, :].to(dtype=lm_head.weight.dtype)
    bias = getattr(lm_head, "bias", None)
    workspace_key = "tree_recheck_lm_head_topk_workspace"
    workspace = token_state.get(workspace_key) if isinstance(token_state, dict) else None
    if (
        can_use_lm_head_topk_triton_multi is not None
        and lm_head_topk_triton_multi is not None
        and can_use_lm_head_topk_triton_multi(root_hidden, lm_head.weight, bias, top_k=2)
    ):
        vals, idxs, workspace = lm_head_topk_triton_multi(
            root_hidden,
            lm_head.weight,
            bias,
            top_k=2,
            workspace=workspace,
        )
        if isinstance(token_state, dict):
            token_state[workspace_key] = workspace
    else:
        logits = lm_head(root_hidden).float()
        vals, idxs = torch.topk(logits, k=2, dim=-1)
    vals = vals.reshape(-1).float()
    idxs = idxs.reshape(-1).to(dtype=torch.long)
    if int(vals.numel()) < 2:
        return float("inf"), int(idxs[0].item()) if int(idxs.numel()) else -1, -1
    margin = float((vals[0] - vals[1]).item())
    return margin, int(idxs[0].item()), int(idxs[1].item())


def _maybe_recheck_tree_decoding(
    model,
    past_kv,
    fast_hidden,
    draft_tokens,
    retrieve_indices,
    tree_mask,
    tree_position_ids,
    current_len,
    layer_indices,
    token_state,
    stats: dict,
    *,
    round_index: int,
    debug: bool = False,
):
    if os.getenv("AICAS_EAGLE3_CHAIN_RECHECK", "0") != "1":
        return None
    if os.getenv("AICAS_SPEC_MT_ATTN", "0") != "1":
        return None
    if os.getenv("AICAS_SPEC_MT_ATTN_EAGER_CHAIN", "0") == "1":
        return None

    margin, top1, top2 = _eagle3_root_lm_head_margin(model, fast_hidden, token_state)
    threshold = float(os.getenv("AICAS_EAGLE3_CHAIN_RECHECK_MARGIN", "0.75"))
    stats["chain_recheck_considered"] = stats.get("chain_recheck_considered", 0) + 1
    if margin > threshold:
        return None

    max_per_gen = int(os.getenv("AICAS_EAGLE3_CHAIN_RECHECK_MAX_PER_GEN", "8"))
    if max_per_gen >= 0 and int(stats.get("chain_recheck", 0)) >= max_per_gen:
        stats["chain_recheck_skipped"] = stats.get("chain_recheck_skipped", 0) + 1
        return None

    stats["chain_recheck"] = stats.get("chain_recheck", 0) + 1
    t_recheck = time.perf_counter()
    with _profile_range("eagle3_tree.verify.chain_recheck"):
        with _temporary_env({
            "AICAS_SPEC_MT_ATTN_EAGER_CHAIN": "1",
        }):
            result = _eagle3_tree_decoding(
                model,
                past_kv,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                current_len,
                layer_indices,
                token_state,
            )
    stats["chain_recheck_ms"] = stats.get("chain_recheck_ms", 0.0) + (time.perf_counter() - t_recheck) * 1000
    if debug or os.getenv("AICAS_EAGLE3_CHAIN_RECHECK_TRACE", "0") == "1":
        new_margin, new_top1, new_top2 = _eagle3_root_lm_head_margin(model, result[2], token_state)
        print(
            "[eagle3-tree] chain_recheck "
            f"round={round_index} margin={margin:.4f} top1={top1} top2={top2} "
            f"eager_margin={new_margin:.4f} eager_top1={new_top1} eager_top2={new_top2}",
            flush=True,
        )
    return result


def _gather_tree_path_targets(target_tokens: torch.Tensor, retrieve_indices: torch.Tensor) -> torch.Tensor:
    target_tokens_ext = torch.cat(
        (
            target_tokens.reshape(-1).to(device=retrieve_indices.device, dtype=torch.long),
            torch.full((1,), -1, device=retrieve_indices.device, dtype=torch.long),
        ),
        dim=0,
    )
    return target_tokens_ext[retrieve_indices]


def _suppress_tree_eos_targets_before_min_length(
    model,
    target_tokens: torch.Tensor,
    hidden: torch.Tensor,
    tree_position_ids: torch.Tensor,
    eos_ids: set[int],
    generated_len: int,
    min_new_tokens: int,
) -> torch.Tensor:
    if not eos_ids or min_new_tokens <= 0 or target_tokens.numel() == 0:
        return target_tokens
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not isinstance(hidden, torch.Tensor):
        return target_tokens
    eos_list = [int(x) for x in eos_ids]
    flat_pos = tree_position_ids.reshape(-1).to(device=target_tokens.device, dtype=torch.long)
    flat_tokens = target_tokens.reshape(-1)
    limit = min(int(flat_tokens.numel()), int(flat_pos.numel()), int(hidden.shape[1]))
    if limit <= 0:
        return target_tokens

    active_pos = flat_pos[:limit].add(int(generated_len) + 1).lt(int(min_new_tokens))
    eos_mask = torch.zeros((limit,), device=target_tokens.device, dtype=torch.bool)
    for eos_id in eos_list:
        if 0 <= eos_id < int(getattr(lm_head, "out_features", 0) or 0):
            eos_mask.logical_or_(flat_tokens[:limit].eq(int(eos_id)))
        else:
            eos_mask.logical_or_(flat_tokens[:limit].eq(int(eos_id)))
    replace_idx = torch.nonzero(active_pos & eos_mask, as_tuple=False).reshape(-1)
    if int(replace_idx.numel()) == 0:
        return target_tokens

    with torch.no_grad():
        selected_hidden = hidden.index_select(1, replace_idx)
        logits = lm_head(selected_hidden).float()
        for eos_id in eos_list:
            if 0 <= eos_id < int(logits.shape[-1]):
                logits[..., eos_id] = -torch.inf
        replacement = logits.argmax(dim=-1).reshape(-1).to(device=target_tokens.device, dtype=target_tokens.dtype)

    out_flat = flat_tokens.clone()
    out_flat.index_copy_(0, replace_idx, replacement)
    return out_flat.reshape_as(target_tokens)


def _suppress_tree_eos_targets_before_min_length_exact(
    model,
    inputs: dict,
    current_input_ids: torch.Tensor,
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    tree_mask: torch.Tensor,
    tree_position_ids: torch.Tensor,
    eos_ids: set[int],
    generated_len: int,
    min_new_tokens: int,
) -> torch.Tensor:
    if (
        os.getenv("AICAS_EAGLE3_EXACT_EOS_SUPPRESSION", "0") != "1"
        or not eos_ids
        or min_new_tokens <= 0
        or target_tokens.numel() == 0
    ):
        return target_tokens
    if not isinstance(draft_tokens, torch.Tensor) or draft_tokens.ndim != 2:
        return target_tokens
    node_count = min(
        int(target_tokens.numel()),
        int(tree_position_ids.numel()),
        int(draft_tokens.shape[1]),
        int(tree_mask.shape[-1]),
    )
    if node_count <= 0:
        return target_tokens

    device = target_tokens.device
    flat_tokens = target_tokens.reshape(-1)
    flat_pos = tree_position_ids.reshape(-1).to(device=device, dtype=torch.long)
    active = flat_pos[:node_count].add(int(generated_len) + 1).lt(int(min_new_tokens))
    eos_mask = torch.zeros((node_count,), device=device, dtype=torch.bool)
    for eos_id in eos_ids:
        eos_mask.logical_or_(flat_tokens[:node_count].eq(int(eos_id)))
    replace_idx = torch.nonzero(active & eos_mask, as_tuple=False).reshape(-1)
    if int(replace_idx.numel()) == 0:
        return target_tokens

    out_flat = flat_tokens.clone()
    draft_flat = draft_tokens.reshape(-1).to(device=device, dtype=torch.long)
    visible_mask = tree_mask.reshape(int(tree_mask.shape[-1]), int(tree_mask.shape[-1])).to(
        device=device,
        dtype=torch.bool,
    )
    base_ids = current_input_ids.to(device=device, dtype=torch.long)
    for node_tensor in replace_idx:
        node = int(node_tensor.item())
        if node < 0 or node >= node_count:
            continue
        path_nodes = torch.nonzero(visible_mask[node, :node_count], as_tuple=False).reshape(-1)
        if int(path_nodes.numel()) == 0:
            continue
        path_tokens = draft_flat.index_select(0, path_nodes).reshape(1, -1)
        prefix_ids = torch.cat((base_ids, path_tokens), dim=1)
        replacement = _target_next_for_current_ids(
            model,
            inputs,
            prefix_ids,
            eos_ids,
            generated_len=int(generated_len) + int(path_nodes.numel()),
            min_new_tokens=int(min_new_tokens),
        )
        out_flat[node] = torch.as_tensor(replacement, device=device, dtype=out_flat.dtype).reshape(())

    _trace_once(
        "eagle3_exact_eos_suppression",
        f"[eagle3-tree] exact EOS suppression enabled replacements={int(replace_idx.numel())}",
    )
    return out_flat.reshape_as(target_tokens)


def _evaluate_posterior_greedy_tokens(path_target_tokens: torch.Tensor, candidates: torch.Tensor):
    valid_rows = candidates[:, 0].ge(0) & path_target_tokens[:, 0].ge(0)
    posterior_mask = candidates[:, 1:].to(path_target_tokens.device).eq(path_target_tokens[:, :-1]).int()
    accept_lengths = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
    accept_lengths = torch.where(valid_rows, accept_lengths, torch.full_like(accept_lengths, -1))
    accept_length = accept_lengths.max()
    accept_length_i = max(0, int(accept_length.item()))
    if accept_length_i == 0:
        valid_indices = torch.nonzero(valid_rows, as_tuple=False).reshape(-1)
        if int(valid_indices.numel()) > 0:
            best_candidate = valid_indices[0].to(torch.long)
        else:
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(accept_lengths).to(torch.long)
    return best_candidate, accept_length_i, path_target_tokens[best_candidate, accept_length_i].reshape(())


def _evaluate_tree_posterior_fast(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    retrieve_indices: torch.Tensor,
    token_state: dict | None,
):
    use_fused = os.getenv("AICAS_EAGLE3_FUSED_POSTERIOR", "1") == "1"
    if use_fused and run_eagle3_posterior is not None:
        try:
            workspace = token_state.get("posterior_workspace") if isinstance(token_state, dict) else None
            result, workspace = run_eagle3_posterior(
                target_tokens,
                draft_tokens,
                retrieve_indices,
                workspace=workspace,
            )
            if isinstance(token_state, dict):
                token_state["posterior_workspace"] = workspace
            if result is not None:
                path_target_tokens, candidates, best_candidate, accept_length, next_tok = result
                if os.getenv("AICAS_EAGLE3_FUSED_POSTERIOR_CHECK", "0") == "1":
                    ref_path = _gather_tree_path_targets(target_tokens, retrieve_indices)
                    padding = torch.full((1, 1), -1, device=draft_tokens.device, dtype=torch.long)
                    ref_candidates = torch.cat((draft_tokens, padding), dim=1)[0, retrieve_indices]
                    ref_best, ref_accept, ref_next = _evaluate_posterior_greedy_tokens(ref_path, ref_candidates)
                    ok = (
                        torch.equal(path_target_tokens, ref_path)
                        and torch.equal(candidates, ref_candidates)
                        and torch.equal(best_candidate, ref_best)
                        and int(accept_length.item()) == int(ref_accept)
                        and torch.equal(next_tok, ref_next)
                    )
                    if not ok:
                        raise RuntimeError("fused posterior output mismatch")
                _trace_once("eagle3_fused_posterior", "[eagle3-tree] fused posterior hit")
                return path_target_tokens, candidates, best_candidate, accept_length, next_tok
        except Exception as exc:
            if os.getenv("AICAS_EAGLE3_FUSED_POSTERIOR_REQUIRE", "0") == "1":
                raise
            _trace_once(
                "eagle3_fused_posterior_fallback",
                f"[eagle3-tree] fused posterior fallback: {type(exc).__name__}: {exc}",
            )

    path_target_tokens = _gather_tree_path_targets(target_tokens, retrieve_indices)
    padding = torch.full((1, 1), -1, device=draft_tokens.device, dtype=torch.long)
    candidates = torch.cat((draft_tokens, padding), dim=1)[0, retrieve_indices]
    best_candidate, accept_length, next_tok = _evaluate_posterior_greedy_tokens(path_target_tokens, candidates)
    return path_target_tokens, candidates, best_candidate, accept_length, next_tok


def _truncate_accepted_tokens_at_eos(
    accepted_tokens: torch.Tensor,
    *,
    append_count: int,
    generated_len: int,
    min_new_tokens: int,
    eos_ids: set[int],
) -> tuple[torch.Tensor, int, bool]:
    if not eos_ids or int(append_count) <= 0:
        return accepted_tokens, int(append_count), False
    if int(generated_len) + int(append_count) < int(min_new_tokens):
        return accepted_tokens, int(append_count), False
    flat = accepted_tokens.reshape(-1)
    if int(flat.numel()) <= 0:
        return accepted_tokens, int(append_count), False
    device = flat.device
    eos_mask = torch.zeros_like(flat, dtype=torch.bool)
    for eos_id in eos_ids:
        eos_mask.logical_or_(flat.eq(int(eos_id)))
    if int(min_new_tokens) > 0:
        offsets = torch.arange(1, int(flat.numel()) + 1, device=device, dtype=torch.long)
        eos_mask.logical_and_(offsets.add(int(generated_len)).ge(int(min_new_tokens)))
    if not bool(eos_mask.any().item()):
        return accepted_tokens, int(append_count), False
    first = int(torch.nonzero(eos_mask, as_tuple=False).reshape(-1)[0].item()) + 1
    first = max(1, min(int(first), int(flat.numel())))
    return flat[:first].clone(), first, True


def _accepted_tokens_from_path_plan(
    candidates: torch.Tensor,
    best_candidate,
    accept_length,
    next_tok: torch.Tensor,
    *,
    remaining_tokens: int,
    generated_len: int,
    min_new_tokens: int,
    eos_ids: set[int],
    skip_root: bool,
) -> tuple[torch.Tensor, int, int, bool, int]:
    remaining_tokens = int(remaining_tokens)
    if remaining_tokens <= 0:
        empty = candidates.new_empty((0,), dtype=torch.long)
        return empty, 0, 0, False, 0

    depth = int(candidates.shape[1])
    max_append = min(depth, remaining_tokens)
    if max_append <= 0:
        empty = candidates.new_empty((0,), dtype=torch.long)
        return empty, 0, 0, False, 0

    if isinstance(accept_length, torch.Tensor) and accept_length.is_cuda and candidates.is_cuda:
        device = candidates.device
        best_idx = best_candidate
        if not isinstance(best_idx, torch.Tensor):
            best_idx = torch.tensor(int(best_idx), device=device, dtype=torch.long)
        best_idx = best_idx.reshape(1).to(device=device, dtype=torch.long).clamp_(0, int(candidates.shape[0]) - 1)
        row = candidates.index_select(0, best_idx).reshape(-1).to(dtype=torch.long)

        accept_len_t = accept_length.reshape(()).to(device=device, dtype=torch.long).clamp_(0, depth - 1)
        append_t = torch.minimum(
            accept_len_t + 1,
            accept_len_t.new_tensor(remaining_tokens),
        ).clamp_(0, max_append)

        accepted_full = torch.full((max_append,), -1, device=device, dtype=torch.long)
        if skip_root:
            draft_fill = min(max_append, max(0, depth - 1))
            if draft_fill > 0:
                accepted_full[:draft_fill].copy_(row[1:1 + draft_fill])
        else:
            accepted_full.copy_(row[:max_append])
        last_idx = (append_t - 1).clamp_(0, max_append - 1)
        correction_room = append_t > accept_len_t if skip_root else torch.ones((), device=device, dtype=torch.bool)
        correction_val = torch.where(
            correction_room,
            next_tok.reshape(()).to(device=device, dtype=torch.long),
            accepted_full.gather(0, last_idx.reshape(1)).reshape(()),
        )
        accepted_full.scatter_(0, last_idx.reshape(1), correction_val.reshape(1))

        final_append_t = append_t
        stop_t = append_t.new_zeros(())
        if eos_ids and int(generated_len) + max_append >= int(min_new_tokens):
            offsets = torch.arange(1, max_append + 1, device=device, dtype=torch.long)
            valid = offsets <= append_t
            if int(min_new_tokens) > 0:
                valid = valid & offsets.add(int(generated_len)).ge(int(min_new_tokens))
            eos_mask = torch.zeros((max_append,), device=device, dtype=torch.bool)
            for eos_id in eos_ids:
                eos_mask.logical_or_(accepted_full.eq(int(eos_id)))
            first_or_sentinel = torch.where(valid & eos_mask, offsets, offsets.new_full((), max_append + 1)).min()
            stop_bool = first_or_sentinel.le(max_append)
            final_append_t = torch.where(stop_bool, first_or_sentinel, append_t)
            stop_t = stop_bool.to(dtype=torch.long)

        final_accept_t = torch.minimum(accept_len_t, (final_append_t - 1).clamp_min(0))
        if skip_root:
            no_correction_room = append_t <= accept_len_t
            tree_node_t = torch.where(no_correction_room, append_t + 1, append_t)
            tree_node_t = torch.where(stop_t.bool(), torch.minimum(tree_node_t, final_append_t + 1), tree_node_t)
        else:
            tree_node_t = final_append_t

        plan_cpu = torch.stack((final_append_t, final_accept_t, stop_t, tree_node_t)).detach().cpu()
        append_i, accept_i, stop_raw, tree_node_i = [int(x) for x in plan_cpu.tolist()]
        stop_i = bool(stop_raw)
        accepted_tokens = accepted_full[:append_i].clone()
        return accepted_tokens, append_i, accept_i, stop_i, tree_node_i

    accept_i = max(0, min(int(accept_length.item() if isinstance(accept_length, torch.Tensor) else accept_length), depth - 1))
    append_i = min(accept_i + 1, remaining_tokens, max_append)
    row = candidates[best_candidate].to(dtype=torch.long)
    if skip_root:
        if append_i <= accept_i:
            tree_node_i = append_i + 1
            accepted_tokens = row[1:1 + append_i].clone()
        else:
            tree_node_i = append_i
            parts = []
            accepted_count = max(0, append_i - 1)
            if accepted_count > 0:
                parts.append(row[1:1 + accepted_count])
            parts.append(next_tok.reshape(1).to(device=row.device, dtype=row.dtype))
            accepted_tokens = torch.cat(parts, dim=0).to(dtype=torch.long)
    else:
        tree_node_i = append_i
        accepted_tokens = row[:append_i].clone()
        if append_i > 0:
            accepted_tokens.reshape(-1)[-1].copy_(next_tok.reshape(()).to(device=row.device, dtype=row.dtype))

    accepted_tokens, append_i, stop_i = _truncate_accepted_tokens_at_eos(
        accepted_tokens,
        append_count=append_i,
        generated_len=int(generated_len),
        min_new_tokens=int(min_new_tokens),
        eos_ids=eos_ids,
    )
    if stop_i:
        accept_i = min(int(accept_i), int(append_i) - 1)
        if skip_root:
            tree_node_i = min(int(tree_node_i), int(append_i) + 1)
        else:
            tree_node_i = int(append_i)
    return accepted_tokens, int(append_i), int(accept_i), bool(stop_i), int(tree_node_i)


@torch.no_grad()
def eagle3_tree_speculative_generate(
    model,
    inputs: dict,
    draft_model: Eagle3DraftModel,
    *,
    max_new_tokens: int,
    debug: bool = False,
    min_new_tokens: int = 0,
    layer_indices: tuple[int, int, int] | None = None,
    force_accept_length: int | None = None,
    diagnostic_records: list[dict] | None = None,
    refresh_draft_feature: bool = False,
    rebuild_draft_kv: bool = False,
    compare_rebuild_root: bool = False,
    reference_tokens=None,
):
    device = next(model.parameters()).device
    eos_ids = _eos_id_set(model.generation_config.eos_token_id)
    min_new_tokens = max(0, int(min_new_tokens or 0))
    if layer_indices is None:
        layer_indices = _parse_layer_indices(os.getenv("AICAS_EAGLE3_LAYER_INDICES", ""), 32)
    if not hasattr(draft_model, "tree_mask_init"):
        draft_model.init_tree()
    draft_model.reset_kv()

    with _profile_range("eagle3_tree.prefill_target"):
        prefill_out, feature = _prefill_with_full_feature(model, inputs, layer_indices)
    past_kv = prefill_out.past_key_values
    current_input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
    initial_input_len = int(current_input_ids.shape[1])
    if reference_tokens is None:
        reference_token_tensor = None
    elif isinstance(reference_tokens, torch.Tensor):
        reference_token_tensor = reference_tokens.to(device=device, dtype=torch.long).reshape(-1)
    else:
        reference_token_tensor = torch.tensor(
            [int(x) for x in reference_tokens],
            device=device,
            dtype=torch.long,
        ).reshape(-1)
    _set_eagle3_cache_seq_len(past_kv, int(current_input_ids.shape[1]))
    next_tok = _argmax_with_min_new_tokens(
        prefill_out.logits[:, -1:, :],
        eos_ids,
        generated_len=0,
        min_new_tokens=min_new_tokens,
    )
    sample_token = torch.tensor([[int(next_tok)]], device=device, dtype=torch.long)
    with _profile_range("eagle3_tree.initial_draft"):
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = draft_model.topK_genrate(
            feature,
            torch.cat((current_input_ids, sample_token), dim=1),
            logits_processor=None,
        )

    stats = {
        "rounds": 0,
        "accepted": 0,
        "drafted": 0,
        "fallback": 0,
        "fallback_generated": 0,
        "fallback_reason": "",
        "verify_ms": 0.0,
        "draft_ms": 0.0,
        "rollback_ms": 0.0,
        "accept0": 0,
        "root_checks": 0,
        "root_target_vocab_hits": 0,
        "root_top1_hits": 0,
        "root_topk_hits": 0,
        "raw_root_top1_hits": 0,
        "raw_root_topk_hits": 0,
        "round0_raw_root_topk_hits": 0,
        "round0_root_checks": 0,
        "later_raw_root_topk_hits": 0,
        "later_root_checks": 0,
        "rebuild_root_checks": 0,
        "rebuild_raw_root_top1_hits": 0,
        "rebuild_raw_root_topk_hits": 0,
        "verify_forward_ms": 0.0,
        "verify_lm_head_ms": 0.0,
        "verify_feature_ms": 0.0,
        "chain_recheck": 0,
        "chain_recheck_considered": 0,
        "chain_recheck_skipped": 0,
        "chain_recheck_ms": 0.0,
        "chain_refresh_count": 0,
        "chain_refresh_tokens": 0,
        "chain_refresh_next_corrections": 0,
        "accept_decode_ms": 0.0,
        "accept_decode_next_corrections": 0,
        "posterior_ms": 0.0,
        "cache_select_ms": 0.0,
        "chain_refresh_ms": 0.0,
        "draft_next_ms": 0.0,
    }
    token_state = {}
    token_chunks: list[torch.Tensor] = []
    generated_len = 0
    adaptive_fallback = os.getenv("AICAS_EAGLE3_ADAPTIVE_FALLBACK", "0") == "1"
    adaptive_min_rounds = max(1, int(os.getenv("AICAS_EAGLE3_ADAPTIVE_MIN_ROUNDS", "4")))
    adaptive_min_generated = max(1, int(os.getenv("AICAS_EAGLE3_ADAPTIVE_MIN_GENERATED", "8")))
    adaptive_min_output_per_round = float(os.getenv("AICAS_EAGLE3_ADAPTIVE_MIN_OUTPUT_PER_ROUND", "3.5"))

    if debug:
        print(f"[eagle3-tree] prefill done, next={next_tok}, layers={layer_indices}")

    _prefill_token = torch.tensor([[int(next_tok)]], device=device, dtype=torch.long)
    token_chunks.append(_prefill_token.reshape(-1).detach().clone())
    generated_len = 1
    from spec_decode.eagle3_cache_adapter import _create_static_cache_for_decode
    _prefill_sc = _create_static_cache_for_decode(
        model.config, past_kv, int(current_input_ids.shape[1]), q_len=1
    )
    _prefill_pos = _decode_position_kwargs(model, _prefill_sc, _prefill_token)
    _prefill_kwargs = dict(
        input_ids=_prefill_token, past_key_values=_prefill_sc,
        use_cache=True, return_dict=False, output_hidden_states=False,
        output_attentions=False, logits_to_keep=1, **_prefill_pos,
    )
    _canonical_decode_step(model, _prefill_kwargs, {})
    past_kv = _prefill_sc
    _set_eagle3_cache_seq_len(past_kv, int(current_input_ids.shape[1]) + 1)

    # ── full_graph round state: tracks draft inputs for the next round ──
    _round_draft_feature = feature
    _round_draft_input_ids = torch.cat((current_input_ids, _prefill_token), dim=1)

    while generated_len < max_new_tokens:
        base_cache_len = _eagle3_cache_seq_len(past_kv)
        t0 = time.perf_counter()
        generated_before = int(generated_len)
        accept_kv_source = os.getenv("AICAS_EAGLE3_ACCEPT_KV_SOURCE", "tree").strip().lower()
        round_base_past = (
            _clone_cache_to_len(past_kv, base_cache_len)
            if accept_kv_source in {"decode", "decode_path", "baseline", "generate", "decode_graph", "graph", "canonical"}
            else None
        )
        # verify the whole draft tree and gather target argmax tokens for each retrieved path
        full_graph_used = False
        with _profile_range("eagle3_tree.verify"):
            timing_detail = os.getenv("AICAS_EAGLE3_TIMING_DETAIL", "0") == "1"
            tree_result = None

            # ── full_graph: combined draft + verify in one CUDA graph ──
            if os.getenv("AICAS_EAGLE3_FULL_GRAPH", "0") == "1":
                with torch.no_grad():
                    tree_result = _eagle3_full_graph_generate_verify(
                        model,
                        draft_model,
                        past_kv,
                        _round_draft_feature,
                        _round_draft_input_ids,
                        int(current_input_ids.shape[1]),
                        layer_indices,
                        token_state,
                    )
                if tree_result is not None:
                    full_graph_used = True

            if not full_graph_used:
                use_tree_verify_graph = (
                    os.getenv("AICAS_EAGLE3_TREE_VERIFY_CUDAGRAPH", "0") == "1"
                    and isinstance(token_state, dict)
                    and not token_state.get("disable_tree_verify_graph", False)
                )
                if use_tree_verify_graph:
                    tree_result = _eagle3_tree_verify_graph(
                        model,
                        past_kv,
                        draft_tokens.to(device),
                        retrieve_indices.to(device),
                        tree_mask.to(device),
                        tree_position_ids.to(device),
                        int(current_input_ids.shape[1]),
                        layer_indices,
                        token_state,
                        stats if timing_detail else None,
                    )
                if tree_result is None:
                    tree_result = _eagle3_tree_decoding(
                        model,
                        past_kv,
                        draft_tokens.to(device),
                        retrieve_indices.to(device),
                        tree_mask.to(device),
                        tree_position_ids.to(device),
                        int(current_input_ids.shape[1]),
                        layer_indices,
                        token_state,
                        stats if timing_detail else None,
                    )

            if full_graph_used:
                # full_graph returns: (target_tokens, feature, hidden, static_kv,
                #                     draft_tokens, retrieve_indices, tree_mask, tree_position_ids)
                (tree_target_tokens, tree_feature, tree_hidden, tree_past,
                 draft_tokens, retrieve_indices, tree_mask, tree_position_ids) = tree_result
            else:
                tree_target_tokens, tree_feature, tree_hidden, tree_past = tree_result
            if (
                _canonical_decode_step is not None
                and int(tree_target_tokens.numel()) > 0
                and force_accept_length is not None  # only for force-all-reject; normal mode uses next_tok correction
            ):
                try:
                    _ro_cache = _clone_cache_to_len(round_base_past, base_cache_len) if round_base_past is not None else _clone_cache_to_len(past_kv, base_cache_len)
                    if int(generated_len) == 1:
                        _ro_last = _prefill_token.to(device)
                    else:
                        _ro_last = current_input_ids[:, -1:].to(device)
                    _ro_kw = dict(
                        input_ids=_ro_last,
                        past_key_values=_ro_cache,
                        cache_position=torch.tensor([base_cache_len - 1], device=device, dtype=torch.long),
                        use_cache=True, return_dict=False, output_hidden_states=False,
                        output_attentions=False, logits_to_keep=1,
                    )
                    with torch.no_grad():
                        _ro_next, _ = _canonical_decode_step(model, _ro_kw, {})
                    _ro_corrected = _ro_next.reshape(-1)[0]
                    _ro_old = tree_target_tokens[0].reshape(())
                    if not bool(_ro_corrected.eq(_ro_old).item()):
                        tree_target_tokens[0].copy_(_ro_corrected)
                except Exception:
                    pass
            if os.getenv("AICAS_EAGLE3_ROOT_FWD_COMPARE", "0") == "1":
                try:
                    if round_base_past is not None and _decode_forward_next_token is not None:
                        _diag_cache = _clone_cache_to_len(round_base_past, base_cache_len)
                        _diag_root_tok = draft_tokens[:, 0:1].to(device)
                        _diag_pos_kwargs = _decode_position_kwargs(model, _diag_cache, _diag_root_tok)
                        _diag_kwargs = dict(
                            input_ids=_diag_root_tok,
                            past_key_values=_diag_cache,
                            use_cache=True,
                            return_dict=True,
                            output_hidden_states=True,
                            output_attentions=False,
                            logits_to_keep=1,
                            **_diag_pos_kwargs,
                        )
                        _diag_mcl = int(_diag_cache.layers[0].keys.shape[-2]) if getattr(_diag_cache, "layers", None) else 1280
                        _diag_kwargs["attention_mask"] = torch.ones((1, _diag_mcl), dtype=torch.long, device=device)
                        with torch.no_grad():
                            _diag_next, _diag_out = _decode_forward_next_token(model, _diag_kwargs, {})
                        _diag_root_target = int(tree_target_tokens[0].item()) if int(tree_target_tokens.numel()) > 0 else -1
                        _diag_root_fwd = int(_diag_next.reshape(-1)[0].item())
                        if _diag_root_target != _diag_root_fwd:
                            print(
                                f"[eagle3-root-fwd] ROUND {int(stats['rounds'])} "
                                f"tree_root_target={_diag_root_target} standalone_root_fwd={_diag_root_fwd} "
                                f"MISMATCH",
                                flush=True,
                            )
                except Exception as _exc:
                    print(
                        f"[eagle3-root-fwd] ROUND {int(stats['rounds'])} "
                        f"error: {type(_exc).__name__}: {_exc}",
                        flush=True,
                    )
            if os.getenv("AICAS_EAGLE3_EXACT_EOS_SUPPRESSION", "0") == "1":
                tree_target_tokens = _suppress_tree_eos_targets_before_min_length_exact(
                    model,
                    inputs,
                    current_input_ids,
                    tree_target_tokens,
                    draft_tokens.to(device),
                    tree_mask.to(device),
                    tree_position_ids.to(device),
                    eos_ids,
                    generated_len=generated_before,
                    min_new_tokens=min_new_tokens,
                )
            else:
                tree_target_tokens = _suppress_tree_eos_targets_before_min_length(
                    model,
                    tree_target_tokens,
                    tree_hidden,
                    tree_position_ids.to(device),
                    eos_ids,
                    generated_len=generated_before,
                    min_new_tokens=min_new_tokens,
                )
        stats["verify_ms"] += (time.perf_counter() - t0) * 1000

        t_posterior = time.perf_counter()
        with _profile_range("eagle3_tree.posterior"):
            path_target_tokens, candidates, best_candidate, accept_length, next_tok = _evaluate_tree_posterior_fast(
                tree_target_tokens,
                draft_tokens.to(device),
                retrieve_indices.to(device),
                token_state,
            )
        stats["posterior_ms"] += (time.perf_counter() - t_posterior) * 1000
        root_diag = None
        if (debug or diagnostic_records is not None) and int(candidates.shape[1]) > 1 and int(candidates.shape[0]) > 0:
            root_target = path_target_tokens[0, 0].reshape(())
            root_candidates = candidates[:, 1]
            root_valid = root_candidates.ge(0)
            root_top1_hit = bool(root_candidates[0].eq(root_target).item())
            root_topk_hit = bool(root_candidates[root_valid].eq(root_target).any().item()) if bool(root_valid.any().item()) else False
            raw_root_mask = tree_position_ids.to(device=device).eq(1)
            raw_root_candidates = draft_tokens.to(device=device)[0, raw_root_mask]
            raw_root_top1_hit = bool(raw_root_candidates[0].eq(root_target).item()) if int(raw_root_candidates.numel()) > 0 else False
            raw_root_topk_hit = bool(raw_root_candidates.eq(root_target).any().item()) if int(raw_root_candidates.numel()) > 0 else False
            draft_vocab_ids = getattr(draft_model, "draft_vocab_ids", None)
            if isinstance(draft_vocab_ids, torch.Tensor) and draft_vocab_ids.numel() > 0:
                root_in_vocab = bool(draft_vocab_ids.to(device=root_target.device).eq(root_target).any().item())
            else:
                root_in_vocab = True
            root_diag = {
                "root_target": int(root_target.item()),
                "root_top1": int(root_candidates[0].item()),
                "root_target_in_draft_vocab": root_in_vocab,
                "root_top1_hit": root_top1_hit,
                "root_topk_hit": root_topk_hit,
                "raw_root_top1_hit": raw_root_top1_hit,
                "raw_root_topk_hit": raw_root_topk_hit,
                "raw_root_candidates": [int(x) for x in raw_root_candidates.detach().cpu().tolist()],
            }
            if compare_rebuild_root:
                saved_stable_kv = draft_model._stable_kv
                saved_tree_mask = getattr(draft_model, "tree_mask", None)
                rebuild_diag: dict = {"ok": False}
                try:
                    full_feature = _full_feature_for_current_ids(model, inputs, current_input_ids, layer_indices)
                    draft_model.reset_kv()
                    fresh_draft_tokens, _, _, fresh_tree_position_ids = draft_model.topK_genrate(
                        full_feature,
                        torch.cat((current_input_ids, draft_tokens[:, :1].to(device=device, dtype=torch.long)), dim=1),
                        logits_processor=None,
                    )
                    fresh_root_mask = fresh_tree_position_ids.to(device=device).eq(1)
                    fresh_raw_root = fresh_draft_tokens.to(device=device)[0, fresh_root_mask]
                    fresh_top1_hit = (
                        bool(fresh_raw_root[0].eq(root_target).item())
                        if int(fresh_raw_root.numel()) > 0
                        else False
                    )
                    fresh_topk_hit = (
                        bool(fresh_raw_root.eq(root_target).any().item())
                        if int(fresh_raw_root.numel()) > 0
                        else False
                    )
                    rebuild_diag = {
                        "ok": True,
                        "raw_root_top1_hit": fresh_top1_hit,
                        "raw_root_topk_hit": fresh_topk_hit,
                        "raw_root_candidates": [int(x) for x in fresh_raw_root.detach().cpu().tolist()],
                    }
                    stats["rebuild_root_checks"] += 1
                    stats["rebuild_raw_root_top1_hits"] += int(fresh_top1_hit)
                    stats["rebuild_raw_root_topk_hits"] += int(fresh_topk_hit)
                except Exception as exc:
                    rebuild_diag = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                finally:
                    draft_model._stable_kv = saved_stable_kv
                    draft_model.tree_mask = saved_tree_mask
                root_diag["rebuild"] = rebuild_diag
            stats["root_checks"] += 1
            stats["root_target_vocab_hits"] += int(root_in_vocab)
            stats["root_top1_hits"] += int(root_top1_hit)
            stats["root_topk_hits"] += int(root_topk_hit)
            stats["raw_root_top1_hits"] += int(raw_root_top1_hit)
            stats["raw_root_topk_hits"] += int(raw_root_topk_hit)
            if int(stats["rounds"]) == 0:
                stats["round0_root_checks"] += 1
                stats["round0_raw_root_topk_hits"] += int(raw_root_topk_hit)
            else:
                stats["later_root_checks"] += 1
                stats["later_raw_root_topk_hits"] += int(raw_root_topk_hit)
        if force_accept_length is not None:
            accept_length = max(0, min(int(force_accept_length), int(candidates.shape[1]) - 1))
            if accept_length == 0:
                best_candidate = torch.tensor(0, dtype=torch.long, device=device)
            next_tok = path_target_tokens[best_candidate, accept_length].reshape(())
            if (
                os.getenv("AICAS_EAGLE3_ROOT_FWD_CORRECT", "0") == "1"
                and round_base_past is not None
                and _decode_forward_next_token is not None
                and isinstance(sample_token, torch.Tensor)
                and int(sample_token.numel()) > 0
            ):
                try:
                    _rfc_cache = _clone_cache_to_len(round_base_past, base_cache_len)
                    _rfc_last_tok = sample_token[:, -1:].to(device)
                    # Use base_cache_len-1 so the model outputs prediction for position base_cache_len
                    # (the same position the tree verifier root target is at)
                    _rfc_cpos = torch.tensor([base_cache_len - 1], device=device, dtype=torch.long)
                    _rfc_kwargs = dict(
                        input_ids=_rfc_last_tok,
                        past_key_values=_rfc_cache,
                        cache_position=_rfc_cpos,
                        use_cache=True,
                        return_dict=False,
                        output_hidden_states=False,
                        output_attentions=False,
                        logits_to_keep=1,
                    )
                    with torch.no_grad():
                        _rfc_next, _ = _decode_forward_next_token(model, _rfc_kwargs, {})
                    _rfc_corrected = _rfc_next.reshape(-1)[0]
                    _rfc_old = next_tok.reshape(())
                    if not bool(_rfc_corrected.eq(_rfc_old).item()):
                        stats["root_fwd_corrections"] = stats.get("root_fwd_corrections", 0) + 1
                        next_tok = _rfc_corrected.reshape(())
                    elif os.getenv("AICAS_EAGLE3_ROOT_FWD_DEBUG", "0") == "1":
                        print(
                            f"[eagle3-root-fwd-correct] ROUND {int(stats['rounds'])} "
                            f"MATCH tree={int(_rfc_old.item())} fwd={int(_rfc_corrected.item())}",
                            flush=True,
                        )
                except Exception as _exc:
                    if os.getenv("AICAS_EAGLE3_ROOT_FWD_DEBUG", "0") == "1":
                        print(
                            f"[eagle3-root-fwd-correct] ROUND {int(stats['rounds'])} "
                            f"error: {type(_exc).__name__}: {_exc}",
                            flush=True,
                        )
        accepted_tokens, append_count, accept_length, stop_after_append, _tree_node_count = _accepted_tokens_from_path_plan(
            candidates,
            best_candidate,
            accept_length,
            next_tok,
            remaining_tokens=int(max_new_tokens) - int(generated_len),
            generated_len=int(generated_len),
            min_new_tokens=int(min_new_tokens),
            eos_ids=eos_ids,
            skip_root=False,
        )
        if os.getenv("AICAS_EAGLE3_STRICT_GREEDY_CHECK", "0") == "1":
            _strict_check_accepted_tokens(
                model,
                inputs,
                current_input_ids,
                accepted_tokens,
                generated_len=generated_len,
                eos_ids=eos_ids,
                min_new_tokens=min_new_tokens,
                round_index=int(stats["rounds"]),
                accept_length=int(accept_length),
            )
            if not stop_after_append:
                chain_next = None
                try:
                    chain_base = _clone_cache_to_len(past_kv, base_cache_len)
                    chain_target, _, _, _ = _refresh_accepted_chain(
                        model,
                        chain_base,
                        accepted_tokens.reshape(-1),
                        {},
                        layer_indices,
                    )
                    chain_flat = chain_target.reshape(-1)
                    if int(chain_flat.numel()) >= int(append_count):
                        chain_next = int(chain_flat[int(append_count) - 1].item())
                except Exception as exc:
                    chain_next = f"{type(exc).__name__}: {exc}"
                try:
                    _strict_check_next_token(
                        model,
                        inputs,
                        torch.cat((current_input_ids, accepted_tokens.reshape(1, -1)), dim=1),
                        next_tok,
                        generated_len=generated_len + int(append_count),
                        eos_ids=eos_ids,
                        min_new_tokens=min_new_tokens,
                        round_index=int(stats["rounds"]),
                        accept_length=int(accept_length),
                        path_target_tokens=path_target_tokens,
                        candidates=candidates,
                        retrieve_indices=retrieve_indices,
                        best_candidate=best_candidate,
                        chain_next=chain_next,
                    )
                except RuntimeError as exc:
                    if os.getenv("AICAS_EAGLE3_STRICT_TREE_REF", "0") != "1":
                        raise
                    ref_debug = _strict_tree_reference_debug(
                        model,
                        past_kv,
                        draft_tokens.to(device),
                        tree_mask.to(device),
                        tree_position_ids.to(device),
                        tree_target_tokens,
                        tree_hidden,
                        retrieve_indices.to(device),
                        best_candidate,
                        base_cache_len=int(base_cache_len),
                        current_len=int(current_input_ids.shape[1]),
                        layer_indices=layer_indices,
                    )
                    raise RuntimeError(f"{exc}\n{ref_debug}") from exc
        token_chunks.append(accepted_tokens.reshape(-1).detach().clone())
        generated_len += int(append_count)
        if reference_token_tensor is not None:
            current_tokens = _tokens_to_list(torch.cat(token_chunks, dim=0))
            expected_tokens = _tokens_to_list(reference_token_tensor[: len(current_tokens)])
            mismatch = _first_token_mismatch(current_tokens, expected_tokens)
            if mismatch is not None:
                start = max(0, int(mismatch) - 8)
                end = int(mismatch) + 8
                raise RuntimeError(
                    "EAGLE3 reference mismatch after append: "
                    f"round={int(stats['rounds'])} generated_before={int(generated_before)} "
                    f"append_count={int(append_count)} accept_length={int(accept_length)} "
                    f"mismatch={int(mismatch)} "
                    f"tokens[{start}:{end}]={current_tokens[start:end]} "
                    f"expected[{start}:{end}]={expected_tokens[start:end]} "
                    f"accepted={_target_to_list(accepted_tokens.reshape(-1))} "
                    f"best={int(best_candidate.item() if isinstance(best_candidate, torch.Tensor) else best_candidate)} "
                    f"candidate={_target_to_list(candidates[best_candidate])} "
                    f"path={_target_to_list(path_target_tokens[best_candidate])} "
                    f"retrieve={_target_to_list(retrieve_indices[best_candidate])}"
                )

        t0 = time.perf_counter()
        t_cache_select = time.perf_counter()
        with _profile_range("eagle3_tree.cache_select"):
            selected_nodes = retrieve_indices[best_candidate, :append_count].to(device=device, dtype=torch.long).contiguous()
            selected_past = _select_tree_cache_path(tree_past, base_cache_len, selected_nodes)
            if selected_past is None:
                raise RuntimeError("EAGLE3 tree verifier could not select accepted KV path")
            current_input_ids = torch.cat((current_input_ids, accepted_tokens.reshape(1, -1)), dim=1)
            if (
                os.getenv("AICAS_EAGLE3_STRICT_EOS_CHECK", "0") == "1"
                and eos_ids
                and generated_len < min_new_tokens
                and int(next_tok.item() if isinstance(next_tok, torch.Tensor) else next_tok) in eos_ids
            ):
                raise RuntimeError(
                    "EAGLE3 tree next token is EOS before min_new_tokens after verifier suppression"
                )
            if isinstance(next_tok, torch.Tensor):
                sample_token = next_tok.reshape(1, 1).to(device=device, dtype=torch.long)
            else:
                sample_token = torch.tensor([[int(next_tok)]], device=device, dtype=torch.long)
            accept_feature = tree_feature[:, selected_nodes, :].contiguous()
            draft_input_ids = torch.cat((current_input_ids, sample_token), dim=1)
            # ── full_graph: update round state for next iteration ──
            _round_draft_feature = accept_feature
            _round_draft_input_ids = draft_input_ids
        stats["cache_select_ms"] += (time.perf_counter() - t_cache_select) * 1000

        accept_decode_diag = None
        if accept_kv_source in {"decode", "decode_path", "baseline", "generate", "decode_graph", "graph"}:
            if round_base_past is None:
                round_base_past = _clone_cache_to_len(past_kv, base_cache_len)
            t_accept_decode = time.perf_counter()
            with _profile_range("eagle3_tree.accept_decode_path"):
                _acc_mcl = int(round_base_past.layers[0].keys.shape[-2]) if getattr(round_base_past, "layers", None) else 1280
                _acc_mask = torch.zeros((1, _acc_mcl), dtype=torch.long, device=device)
                _acc_mask[:, :base_cache_len + int(append_count)] = 1
                decode_next, decode_past, decode_feature = _decode_accept_path(
                    model,
                    round_base_past,
                    accepted_tokens.reshape(-1).to(device=device, dtype=torch.long),
                    token_state,
                    layer_indices,
                    attention_mask=_acc_mask,
                )
                graph_next = None
            stats["accept_decode_ms"] = stats.get("accept_decode_ms", 0.0) + (
                time.perf_counter() - t_accept_decode
            ) * 1000
            _set_eagle3_cache_seq_len(decode_past, base_cache_len + append_count)
            old_next = sample_token.reshape(()).to(device=decode_next.device, dtype=decode_next.dtype)
            chosen_next = graph_next if isinstance(graph_next, torch.Tensor) else decode_next
            decode_next_scalar = chosen_next.reshape(-1)[-1].reshape(())
            if not bool(decode_next_scalar.eq(old_next).item()):
                stats["accept_decode_next_corrections"] = stats.get("accept_decode_next_corrections", 0) + 1
                next_tok = decode_next_scalar
                sample_token = decode_next_scalar.reshape(1, 1).to(device=device, dtype=torch.long)
                draft_input_ids = torch.cat((current_input_ids, sample_token), dim=1)
            if diagnostic_records is not None:
                if accept_feature is not None and decode_feature is not None:
                    denom = accept_feature.float().norm().clamp_min(1e-6)
                    rel = float(((accept_feature.float() - decode_feature.float()).norm() / denom).item())
                    tn = float(accept_feature.float().norm().item())
                    dn = float(decode_feature.float().norm().item())
                else:
                    rel = 0.0
                    tn = 0.0
                    dn = 0.0
                accept_decode_diag = {
                    "source": accept_kv_source,
                    "corrected_next": int(decode_next_scalar.item()),
                    "feature_rel_l2": rel,
                    "tree_feature_norm": tn,
                    "decode_feature_norm": dn,
                    "decode_next": [int(x) for x in decode_next.reshape(-1).detach().cpu().tolist()],
                    "graph_next": (
                        [int(x) for x in graph_next.reshape(-1).detach().cpu().tolist()]
                        if isinstance(graph_next, torch.Tensor)
                        else None
                    ),
                }
            past_kv = decode_past
            if decode_feature is not None:
                accept_feature = decode_feature
        elif accept_kv_source == "canonical":
            _canon_acc_tok = accepted_tokens.reshape(-1).to(device=device, dtype=torch.long)
            _canon_acc_base = _clone_cache_to_len(round_base_past, base_cache_len)
            _canon_pos_kwargs = _decode_position_kwargs(model, _canon_acc_base, _canon_acc_tok.reshape(1, -1))
            if int(stats["rounds"]) in {27, 28}:
                _dbg_pos = base_cache_len  # the position being written
                _dbg_kv = _canon_acc_base.layers[0].keys[0, 0, _dbg_pos - 1, :3].float().tolist() if getattr(_canon_acc_base, "layers", None) else []
                print(f"[kv-dbg] R{int(stats['rounds'])} writing pos={_dbg_pos} token={int(accepted_tokens.reshape(-1)[-1].item())} prev_kv={[round(x,4) for x in _dbg_kv]}", flush=True)
            _canon_kwargs = dict(
                input_ids=_canon_acc_tok.reshape(1, -1),
                past_key_values=_canon_acc_base,
                use_cache=True,
                return_dict=False,
                output_hidden_states=False,
                output_attentions=False,
                logits_to_keep=1,
                **_canon_pos_kwargs,
            )
            _canon_next, _canon_out = _canonical_decode_step(model, _canon_kwargs, {})
            decode_next = _canon_next.reshape(-1).to(dtype=torch.long)
            decode_past = _extract_past(_canon_out) if _canon_out is not None else _canon_acc_base
            if decode_past is None:
                decode_past = _canon_acc_base
            _set_eagle3_cache_seq_len(decode_past, base_cache_len + append_count)
            old_next = sample_token.reshape(()).to(device=decode_next.device, dtype=decode_next.dtype)
            decode_next_scalar = decode_next.reshape(-1)[-1].reshape(())
            if not bool(decode_next_scalar.eq(old_next).item()):
                next_tok = decode_next_scalar
                sample_token = decode_next_scalar.reshape(1, 1).to(device=device, dtype=torch.long)
                draft_input_ids = torch.cat((current_input_ids, sample_token), dim=1)
            past_kv = decode_past
        elif accept_kv_source not in {"tree", "selected_tree"}:
            raise RuntimeError(f"unknown AICAS_EAGLE3_ACCEPT_KV_SOURCE={accept_kv_source!r}")

        chain_refresh_diag = None
        cache_select_diag = None
        refresh_policy = os.getenv("AICAS_EAGLE3_ACCEPT_FEATURE_REFRESH", "off").strip().lower()
        if refresh_policy in {"0", "false", "no", "off", "none"}:
            use_chain_refresh = False
            refresh_reason = "off"
        elif refresh_policy in {"1", "true", "yes", "chain", "always", "all"}:
            use_chain_refresh = True
            refresh_reason = "chain"
        elif refresh_policy in {"accept0", "auto"}:
            use_chain_refresh = int(accept_length) == 0
            refresh_reason = "accept0" if use_chain_refresh else "skip_accept"
        else:
            use_chain_refresh = False
            refresh_reason = f"unknown:{refresh_policy}"
        if use_chain_refresh:
            stats["chain_refresh_count"] = stats.get("chain_refresh_count", 0) + 1
            stats["chain_refresh_tokens"] = stats.get("chain_refresh_tokens", 0) + int(append_count)
            t_chain_refresh = time.perf_counter()
            with _profile_range("eagle3_tree.accept_chain_refresh"):
                base_past_for_chain = _clone_cache_to_len(selected_past, base_cache_len)
                chain_target, chain_past, chain_feature, _chain_hidden = _refresh_accepted_chain(
                    model,
                    base_past_for_chain,
                    accepted_tokens.reshape(-1),
                    token_state,
                    layer_indices,
            )
            stats["chain_refresh_ms"] += (time.perf_counter() - t_chain_refresh) * 1000
            _set_eagle3_cache_seq_len(chain_past, base_cache_len + append_count)
            if os.getenv("AICAS_EAGLE3_CACHE_SELECT_CHECK", "0") == "1":
                cache_select_diag = _cache_suffix_diff_diag(
                    selected_past,
                    chain_past,
                    int(base_cache_len),
                    int(append_count),
                    label_a="selected_tree",
                    label_b="chain_refresh",
                )
                print(
                    "[eagle3-tree][cache-select-check] "
                    f"round={int(stats['rounds'])} generated={int(generated_before)} "
                    f"append={int(append_count)} nodes={_target_to_list(selected_nodes)} "
                    f"max_k={cache_select_diag.get('max_k')} layer_k={cache_select_diag.get('max_k_layer')} "
                    f"max_v={cache_select_diag.get('max_v')} layer_v={cache_select_diag.get('max_v_layer')}",
                    flush=True,
                )
                try:
                    tree_pos_dbg = _eagle3_tree_position_ids(
                        model,
                        int(current_input_ids.shape[1] - append_count),
                        tree_position_ids.to(device=device, dtype=torch.long),
                        batch_size=1,
                    )
                    chain_pos_dbg = _decode_position_kwargs(
                        model,
                        _clone_cache_to_len(selected_past, int(base_cache_len)),
                        accepted_tokens.reshape(1, -1).to(device=device, dtype=torch.long),
                    )
                    chain_pos_ids = chain_pos_dbg.get("position_ids")
                    chain_cache_pos = chain_pos_dbg.get("cache_position")
                    cache_select_diag["position_debug"] = {
                        "tree_root_position_ids": (
                            [int(x) for x in tree_pos_dbg[:, 0, 0].detach().cpu().tolist()]
                            if isinstance(tree_pos_dbg, torch.Tensor) and tree_pos_dbg.ndim == 3
                            else None
                        ),
                        "tree_selected_position_ids": (
                            [
                                [int(v) for v in tree_pos_dbg[:, 0, int(n)].detach().cpu().tolist()]
                                for n in selected_nodes.reshape(-1).detach().cpu().tolist()
                                if int(n) >= 0 and int(n) < int(tree_pos_dbg.shape[-1])
                            ]
                            if isinstance(tree_pos_dbg, torch.Tensor) and tree_pos_dbg.ndim == 3
                            else None
                        ),
                        "chain_position_ids": (
                            chain_pos_ids[:, 0, :].detach().cpu().to(dtype=torch.long).tolist()
                            if isinstance(chain_pos_ids, torch.Tensor) and chain_pos_ids.ndim == 3
                            else None
                        ),
                        "chain_cache_position": (
                            [int(x) for x in chain_cache_pos.reshape(-1).detach().cpu().tolist()]
                            if isinstance(chain_cache_pos, torch.Tensor)
                            else None
                        ),
                    }
                    print(
                        "[eagle3-tree][cache-select-pos] "
                        f"round={int(stats['rounds'])} "
                        f"tree_root={cache_select_diag['position_debug']['tree_root_position_ids']} "
                        f"tree_selected={cache_select_diag['position_debug']['tree_selected_position_ids']} "
                        f"chain_pos={cache_select_diag['position_debug']['chain_position_ids']} "
                        f"chain_cache={cache_select_diag['position_debug']['chain_cache_position']}",
                        flush=True,
                    )
                    if os.getenv("AICAS_EAGLE3_CACHE_SELECT_MASK_DEBUG", "0") == "1":
                        try:
                            tm = tree_mask.reshape(int(tree_mask.shape[-2]), int(tree_mask.shape[-1]))
                            selected_visible = []
                            for n in selected_nodes.reshape(-1).detach().cpu().tolist():
                                ni = int(n)
                                if 0 <= ni < int(tm.shape[0]):
                                    visible = torch.nonzero(
                                        tm[ni].to(device=tm.device, dtype=torch.bool),
                                        as_tuple=False,
                                    ).reshape(-1)
                                    selected_visible.append([int(x) for x in visible.detach().cpu().tolist()])
                            cache_select_diag["mask_debug"] = {
                                "selected_visible": selected_visible,
                                "tree_shape": [int(x) for x in tm.shape],
                            }
                            print(
                                "[eagle3-tree][cache-select-mask] "
                                f"round={int(stats['rounds'])} nodes={_target_to_list(selected_nodes)} "
                                f"visible={selected_visible} shape={tuple(int(x) for x in tm.shape)}",
                                flush=True,
                            )
                        except Exception as exc:
                            cache_select_diag["mask_debug"] = f"{type(exc).__name__}: {exc}"
                except Exception as exc:
                    cache_select_diag["position_debug"] = f"{type(exc).__name__}: {exc}"
            chain_flat = chain_target.reshape(-1).to(device=device, dtype=torch.long)
            chain_next = None
            if int(chain_flat.numel()) >= int(append_count) and int(append_count) > 0:
                chain_next = chain_flat[int(append_count) - 1].reshape(())
                old_next = sample_token.reshape(()).to(device=chain_next.device, dtype=chain_next.dtype)
                if not bool(chain_next.eq(old_next).item()):
                    stats["chain_refresh_next_corrections"] = stats.get("chain_refresh_next_corrections", 0) + 1
                    next_tok = chain_next
                    sample_token = chain_next.reshape(1, 1).to(device=device, dtype=torch.long)
                    draft_input_ids = torch.cat((current_input_ids, sample_token), dim=1)
            if os.getenv("AICAS_EAGLE3_STRICT_CHAIN_REFRESH", "0") == "1":
                accepted_flat = accepted_tokens.reshape(-1)
                prefix_expected = accepted_flat[1:]
                prefix_actual = chain_flat[: int(prefix_expected.numel())]
                next_actual = chain_flat[int(append_count) - 1].reshape(())
                next_expected = sample_token.reshape(()).to(device=next_actual.device, dtype=next_actual.dtype)
                prefix_ok = (
                    int(prefix_actual.numel()) == int(prefix_expected.numel())
                    and bool(prefix_actual.eq(prefix_expected).all().item())
                )
                next_ok = bool(next_actual.eq(next_expected).item())
                if not (prefix_ok and next_ok):
                    raise RuntimeError(
                        "EAGLE3 chain refresh target mismatch: "
                        f"accepted={_target_to_list(accepted_flat)} "
                        f"chain={_target_to_list(chain_flat[: int(append_count)])} "
                        f"next={int(next_expected.item())}"
                    )
            if diagnostic_records is not None:
                denom = accept_feature.float().norm().clamp_min(1e-6)
                rel = (accept_feature.float() - chain_feature.float()).norm() / denom
                chain_refresh_diag = {
                    "reason": refresh_reason,
                    "corrected_next": int(chain_next.item()) if isinstance(chain_next, torch.Tensor) else None,
                    "cache_select": cache_select_diag,
                    "feature_rel_l2": float(rel.item()),
                    "tree_feature_norm": float(accept_feature.float().norm().item()),
                    "chain_feature_norm": float(chain_feature.float().norm().item()),
                    "chain_target": [int(x) for x in chain_target.reshape(-1).detach().cpu().tolist()],
                }
            past_kv = chain_past
            accept_feature = chain_feature
        else:
            if accept_kv_source in {"tree", "selected_tree"}:
                past_kv = selected_past

        if refresh_draft_feature or rebuild_draft_kv:
            full_feature = _full_feature_for_current_ids(model, inputs, current_input_ids, layer_indices)
            if rebuild_draft_kv:
                draft_model.reset_kv()
                accept_feature = full_feature
            else:
                accept_feature = full_feature[:, -append_count:, :]
            _round_draft_feature = accept_feature  # full_graph: keep round state in sync
        if diagnostic_records is not None:
            eager_tree_target = None
            tree_root_topk = None
            tree_root_masked_topk = None
            post_select_chain_target = None
            pre_select_chain_target = None
            decode_graph_next_after_accept = None
            full_forward_next_after_accept = None
            full_forward_topk_after_accept = None
            generate_style_next_after_accept = None
            tree_root_hidden_l2 = None
            full_root_hidden_l2 = None
            cache_len_diag = None
            try:
                tree_logits = model.lm_head(tree_hidden).float()
                eager_tree_target = tree_logits.argmax(dim=-1).reshape(-1).to(dtype=torch.long)
                if int(tree_logits.shape[1]) > 0:
                    root_logits = tree_logits[:, 0, :].reshape(-1)
                    k = min(8, int(root_logits.numel()))
                    tree_root_topk = [
                        int(x) for x in root_logits.topk(k, dim=-1).indices.detach().cpu().tolist()
                    ]
                    masked = root_logits.clone()
                    for eos_id in eos_ids:
                        if 0 <= int(eos_id) < int(masked.numel()):
                            masked[int(eos_id)] = -torch.inf
                    tree_root_masked_topk = [
                        int(x) for x in masked.topk(k, dim=-1).indices.detach().cpu().tolist()
                    ]
            except Exception:
                eager_tree_target = None
            if os.getenv("AICAS_EAGLE3_DIAG_POST_SELECT_CHAIN", "0") == "1":
                try:
                    pre_base = _clone_cache_to_len(past_kv, int(base_cache_len))
                    pre_target, _, _, _ = _refresh_accepted_chain(
                        model,
                        pre_base,
                        accepted_tokens.reshape(-1),
                        {},
                        layer_indices,
                    )
                    pre_select_chain_target = [
                        int(x) for x in pre_target.reshape(-1).detach().cpu().tolist()
                    ]
                except Exception as exc:
                    pre_select_chain_target = f"{type(exc).__name__}: {exc}"
                try:
                    diag_base = _clone_cache_to_len(selected_past, int(base_cache_len))
                    diag_target, _, _, _ = _refresh_accepted_chain(
                        model,
                        diag_base,
                        accepted_tokens.reshape(-1),
                        {},
                        layer_indices,
                    )
                    post_select_chain_target = [
                        int(x) for x in diag_target.reshape(-1).detach().cpu().tolist()
                    ]
                except Exception as exc:
                    post_select_chain_target = f"{type(exc).__name__}: {exc}"
                try:
                    full_forward_next_after_accept = int(_target_next_for_current_ids(
                        model,
                        inputs,
                        current_input_ids,
                        eos_ids,
                        generated_len=int(generated_len),
                        min_new_tokens=int(min_new_tokens),
                    ))
                except Exception as exc:
                    full_forward_next_after_accept = f"{type(exc).__name__}: {exc}"
                try:
                    dg_base = _clone_cache_to_len(selected_past, int(base_cache_len))
                    dg_next = _decode_graph_next_for_tokens(
                        model,
                        dg_base,
                        accepted_tokens.reshape(-1).to(device=device, dtype=torch.long),
                        {},
                    )
                    decode_graph_next_after_accept = [
                        int(x) for x in dg_next.reshape(-1).detach().cpu().tolist()
                    ]
                except Exception as exc:
                    decode_graph_next_after_accept = f"{type(exc).__name__}: {exc}"
                try:
                    full_forward_topk_after_accept = _target_topk_for_current_ids(
                        model,
                        inputs,
                        current_input_ids,
                        eos_ids,
                        generated_len=int(generated_len),
                        min_new_tokens=int(min_new_tokens),
                        k=8,
                    )
                except Exception as exc:
                    full_forward_topk_after_accept = f"{type(exc).__name__}: {exc}"
                diag_generate_round = os.getenv("AICAS_EAGLE3_DIAG_GENERATE_STYLE_ROUND", "").strip()
                should_diag_generate = os.getenv("AICAS_EAGLE3_DIAG_GENERATE_STYLE_NEXT", "0") == "1"
                if diag_generate_round:
                    try:
                        should_diag_generate = should_diag_generate and int(diag_generate_round) == int(stats["rounds"])
                    except Exception:
                        should_diag_generate = False
                if should_diag_generate:
                    try:
                        generate_style_next_after_accept = _generate_style_next_for_current_ids(
                            model,
                            inputs,
                            current_input_ids,
                            eos_ids,
                            generated_len=int(generated_len),
                            min_new_tokens=int(min_new_tokens),
                        )
                    except Exception as exc:
                        generate_style_next_after_accept = f"{type(exc).__name__}: {exc}"
                try:
                    full_hidden = _full_hidden_for_current_ids(model, inputs, current_input_ids)
                    tree_root_hidden = tree_hidden[:, 0:1, :].to(dtype=full_hidden.dtype)
                    tree_root_hidden_l2 = float((tree_root_hidden.float() - full_hidden[:, -1:, :].float()).norm().item())
                    full_root_hidden_l2 = float(full_hidden[:, -1:, :].float().norm().item())
                except Exception as exc:
                    tree_root_hidden_l2 = f"{type(exc).__name__}: {exc}"
                    full_root_hidden_l2 = None
                try:
                    cache_len_diag = {
                        "past_kv": _eagle3_cache_len_diag(past_kv),
                        "tree_past": _eagle3_cache_len_diag(tree_past),
                        "selected_past": _eagle3_cache_len_diag(selected_past),
                        "base_cache_len": int(base_cache_len),
                        "current_len_before_append": int(current_input_ids.shape[1] - append_count),
                        "current_len_after_append": int(current_input_ids.shape[1]),
                        "append_count": int(append_count),
                    }
                except Exception as exc:
                    cache_len_diag = {"error": f"{type(exc).__name__}: {exc}"}
            diagnostic_records.append({
                "round": int(stats["rounds"]),
                "generated_before": int(generated_before),
                "base_cache_len": int(base_cache_len),
                "current_input_len": int(current_input_ids.shape[1] - append_count),
                "best_candidate": int(best_candidate.item() if isinstance(best_candidate, torch.Tensor) else best_candidate),
                "accept_length": int(accept_length),
                "append_count": int(append_count),
                "next_token": int(next_tok.item() if isinstance(next_tok, torch.Tensor) else next_tok),
                "selected_nodes": [int(x) for x in selected_nodes.detach().cpu().tolist()],
                "accepted_tokens": _target_to_list(accepted_tokens.reshape(-1)),
                "root_diag": root_diag,
                "candidates": candidates.detach().cpu().tolist(),
                "path_target_tokens": path_target_tokens.detach().cpu().tolist(),
                "accept_decode": accept_decode_diag,
                "chain_refresh": chain_refresh_diag,
                "pre_select_chain_target": pre_select_chain_target,
                "post_select_chain_target": post_select_chain_target,
                "decode_graph_next_after_accept": decode_graph_next_after_accept,
                "full_forward_next_after_accept": full_forward_next_after_accept,
                "full_forward_topk_after_accept": full_forward_topk_after_accept,
                "generate_style_next_after_accept": generate_style_next_after_accept,
                "tree_root_hidden_l2": tree_root_hidden_l2,
                "full_root_hidden_l2": full_root_hidden_l2,
                "cache_len_diag": cache_len_diag,
                "eager_tree_target_tokens": (
                    eager_tree_target.detach().cpu().tolist()
                    if isinstance(eager_tree_target, torch.Tensor)
                    else None
                ),
                "tree_root_topk": tree_root_topk,
                "tree_root_masked_topk": tree_root_masked_topk,
                "retrieve_indices": retrieve_indices.detach().cpu().tolist(),
            })

        t1 = time.perf_counter()
        # draft the next tokens based on the accepted feature and next token, to warm up the KV cache for the next step
        if not full_graph_used:
            with _profile_range("eagle3_tree.draft_next"):
                try:
                    draft_tokens, retrieve_indices, tree_mask, tree_position_ids = draft_model.topK_genrate(
                        accept_feature,
                        draft_input_ids,
                        logits_processor=None,
                    )
                except Exception:
                    if os.getenv("AICAS_EAGLE3_DRAFT_NEXT_ERROR_DUMP", "1") == "1":
                        try:
                            stable_kv = getattr(draft_model, "_stable_kv", None)
                            stable_len = None
                            if stable_kv is not None:
                                stable_obj = stable_kv[0] if isinstance(stable_kv, tuple) else stable_kv
                                stable_len = int(draft_model._kv_seq_len(stable_obj))
                            feat = accept_feature.detach()
                            ids = draft_input_ids.detach()
                            print(
                                "[eagle3-draft-next-error] "
                                f"round={int(stats.get('rounds', -1))} generated={int(generated_len)} "
                                f"append={int(append_count)} accept_source={accept_kv_source} "
                                f"feature_shape={tuple(int(x) for x in feat.shape)} "
                                f"feature_dtype={feat.dtype} feature_min={float(feat.float().min().item()):.6f} "
                                f"feature_max={float(feat.float().max().item()):.6f} "
                                f"feature_nan={bool(torch.isnan(feat.float()).any().item())} "
                                f"draft_input_len={int(ids.shape[1])} "
                                f"draft_input_tail={_target_to_list(ids.reshape(-1)[-8:])} "
                                f"sample_token={_target_to_list(sample_token.reshape(-1))} "
                                f"stable_kv_len={stable_len} "
                                f"accept_decode={accept_decode_diag}",
                                flush=True,
                            )
                        except Exception as dump_exc:
                            print(
                                f"[eagle3-draft-next-error] dump_failed={type(dump_exc).__name__}: {dump_exc}",
                                flush=True,
                            )
                    raise
        draft_next_elapsed = (time.perf_counter() - t1) * 1000
        stats["draft_ms"] += draft_next_elapsed
        stats["draft_next_ms"] += draft_next_elapsed
        stats["rollback_ms"] += (time.perf_counter() - t0) * 1000
        stats["rounds"] += 1
        stats["accept0"] += int(accept_length == 0)
        stats["accepted"] += max(0, append_count - 1)
        stats["drafted"] += max(0, int(draft_tokens.shape[1]) - 1)

        if debug and stats["rounds"] == 1:
            print(
                f"[eagle3-tree] candidates={_target_to_list(candidates[best_candidate], 8)} "
                f"accepted={accept_length} next={next_tok}"
            )

        if stop_after_append:
            break

        if (
            adaptive_fallback
            and generated_len < max_new_tokens
            and stats["rounds"] >= adaptive_min_rounds
            and generated_len >= adaptive_min_generated
        ):
            output_per_round = float(generated_len) / max(1, int(stats["rounds"]))
            if output_per_round < adaptive_min_output_per_round:
                stats["fallback"] = 1
                stats["fallback_generated"] = int(generated_len)
                stats["fallback_reason"] = (
                    f"output_per_round={output_per_round:.2f} "
                    f"< {adaptive_min_output_per_round:.2f}"
                )
                if debug:
                    print(
                        f"[eagle3-tree] adaptive fallback after {generated_len} tokens: "
                        f"{stats['fallback_reason']}"
                    )
                break

    if debug:
        rounds = max(stats["rounds"], 1)
        msg = (
            f"[eagle3-tree] rounds={stats['rounds']} accepted={stats['accepted']} "
            f"avg_accept={stats['accepted'] / rounds:.1f} "
            f"draft={stats['draft_ms'] / rounds:.1f}ms verify={stats['verify_ms'] / rounds:.1f}ms"
        )
        if stats["root_checks"] > 0:
            checks = max(1, int(stats["root_checks"]))
            msg += (
                f" accept0={stats['accept0'] / rounds:.2f} "
                f"root_vocab={stats['root_target_vocab_hits'] / checks:.2f} "
                f"root_top1={stats['root_top1_hits'] / checks:.2f} "
                f"root_topk={stats['root_topk_hits'] / checks:.2f} "
                f"raw_root_top1={stats['raw_root_top1_hits'] / checks:.2f} "
                f"raw_root_topk={stats['raw_root_topk_hits'] / checks:.2f}"
            )
            round0_checks = max(1, int(stats["round0_root_checks"]))
            later_checks = max(1, int(stats["later_root_checks"]))
            msg += (
                f" round0_raw_root_topk={stats['round0_raw_root_topk_hits'] / round0_checks:.2f} "
                f"later_raw_root_topk={stats['later_raw_root_topk_hits'] / later_checks:.2f}"
            )
        if os.getenv("AICAS_EAGLE3_TIMING_DETAIL", "0") == "1":
            msg += (
                f" vfwd={stats['verify_forward_ms'] / rounds:.1f}ms"
                f" vhead={stats['verify_lm_head_ms'] / rounds:.1f}ms"
                f" vfeat={stats['verify_feature_ms'] / rounds:.1f}ms"
                f" recheck={stats['chain_recheck']}/{stats['chain_recheck_considered']}"
                f" recheck_ms={stats['chain_recheck_ms'] / rounds:.1f}ms"
                f" refresh={stats['chain_refresh_count']}/{rounds}"
                f" refresh_next_fix={stats.get('chain_refresh_next_corrections', 0)}"
                f" refresh_ms={stats['chain_refresh_ms'] / rounds:.1f}ms"
                f" accept_decode_ms={stats.get('accept_decode_ms', 0.0) / rounds:.1f}ms"
                f" accept_decode_fix={stats.get('accept_decode_next_corrections', 0)}"
                f" post={stats['posterior_ms'] / rounds:.1f}ms"
                f" csel={stats['cache_select_ms'] / rounds:.1f}ms"
                f" dnext={stats['draft_next_ms'] / rounds:.1f}ms"
            )
        print(msg)

    if token_chunks:
        tokens = _tokens_to_list(torch.cat(token_chunks, dim=0))
        if reference_token_tensor is not None:
            expected = _tokens_to_list(reference_token_tensor[: len(tokens)])
            if tokens != expected:
                mismatch = _first_token_mismatch(tokens, expected)
                start = max(0, int(mismatch or 0) - 8)
                end = int(mismatch or 0) + 8
                raise RuntimeError(
                    "EAGLE3 final token chunk/reference mismatch: "
                    f"mismatch={mismatch} "
                    f"tokens[{start}:{end}]={tokens[start:end]} "
                    f"expected[{start}:{end}]={expected[start:end]}"
                )
    else:
        tokens = []
    return tokens, stats


def eagle3_speculative_generate(
    model,
    inputs: dict,
    draft_model: Eagle3DraftModel,
    *,
    max_new_tokens: int,
    draft_len: int = 4,
    debug: bool = False,
    min_new_tokens: int = 0,
    layer_indices: tuple[int, int, int] | None = None,
    force_accept_length: int | None = None,
    diagnostic_records: list[dict] | None = None,
    refresh_draft_feature: bool = False,
    rebuild_draft_kv: bool = False,
    compare_rebuild_root: bool = False,
):
    del refresh_draft_feature, rebuild_draft_kv, compare_rebuild_root
    device = next(model.parameters()).device
    eos_ids = _eos_id_set(model.generation_config.eos_token_id)
    min_new_tokens = max(0, int(min_new_tokens or 0))
    if layer_indices is None:
        layer_indices = _parse_layer_indices(os.getenv("AICAS_EAGLE3_LAYER_INDICES", ""), 32)

    stats = {
        "rounds": 0,
        "accepted": 0,
        "drafted": 0,
        "mismatches": 0,
        "draft_ms": 0.0,
        "verify_ms": 0.0,
        "rollback_ms": 0.0,
        "verifier": {},
    }
    token_state = {}

    prefill_out, feature = _prefill_with_feature(model, inputs, layer_indices)
    past_kv = prefill_out.past_key_values
    next_tok = _argmax_with_min_new_tokens(
        prefill_out.logits[:, -1:, :],
        eos_ids,
        generated_len=0,
        min_new_tokens=min_new_tokens,
    )
    tokens: list[int] = []
    candidate_buf = torch.empty((1, draft_len), device=device, dtype=torch.long)

    if debug:
        print(f"[eagle3] prefill done, next={next_tok}, layers={layer_indices}")

    while len(tokens) < max_new_tokens:
        generated_before = len(tokens)
        base_cache_len = _cache_seq_len(past_kv)
        t0 = time.perf_counter()
        draft_tensor = draft_model.draft_tensor(feature, next_tok)
        if draft_tensor.numel() > draft_len - 1:
            draft_tensor = draft_tensor[:draft_len - 1]
        drafted_len = int(draft_tensor.numel())
        stats["drafted"] += drafted_len
        stats["draft_ms"] += (time.perf_counter() - t0) * 1000

        cand_len = 1 + drafted_len
        cand_ids = candidate_buf[:, :cand_len]
        cand_ids[0, 0].fill_(int(next_tok))
        if drafted_len > 0:
            cand_ids[0, 1:cand_len].copy_(draft_tensor.to(device=device, dtype=torch.long))

        t0 = time.perf_counter()
        verifier_name = "fused_forward"
        if not token_state.get("disable_eagle3_graph", False):
            try:
                target, vpast, feature_steps, hidden_steps = _graph_verify_chain(
                    model,
                    past_kv,
                    cand_ids,
                    token_state,
                    layer_indices,
                )
                verifier_name = "fused_graph"
            except Exception as exc:
                token_state["disable_eagle3_graph"] = True
                _trace_once(
                    "eagle3_decode_graph_verify_fallback",
                    f"[eagle3] decode graph verifier fallback: {type(exc).__name__}: {exc}",
                )
                target, vpast, feature_steps, hidden_steps = _verify_chain(
                    model,
                    past_kv,
                    cand_ids,
                    token_state,
                    layer_indices,
                )
        else:
            target, vpast, feature_steps, hidden_steps = _verify_chain(
                model,
                past_kv,
                cand_ids,
                token_state,
                layer_indices,
            )
        stats["verify_ms"] += (time.perf_counter() - t0) * 1000
        stats["verifier"][verifier_name] = stats["verifier"].get(verifier_name, 0) + 1

        target_tensor = _target_to_tensor(target, device=device)
        target_tensor = _suppress_eos_targets_before_min_length(
            model,
            target_tensor,
            hidden_steps,
            eos_ids,
            generated_len=len(tokens),
            min_new_tokens=min_new_tokens,
        )
        if drafted_len > 0:
            match = cand_ids[0, 1:1 + drafted_len].eq(target_tensor[:drafted_len])
            mismatch = torch.nonzero(~match, as_tuple=False)
            accepted = drafted_len if mismatch.numel() == 0 else int(mismatch[0, 0].item())
            if mismatch.numel() != 0:
                stats["mismatches"] += 1
        else:
            accepted = 0
        if force_accept_length is not None:
            accepted = max(0, min(int(force_accept_length), drafted_len))

        append_count = min(1 + accepted, max_new_tokens - len(tokens))
        tokens.extend(_tokens_to_list(cand_ids[0, :append_count]))
        stats["accepted"] += accepted
        stats["rounds"] += 1

        if debug and stats["rounds"] == 1:
            print(f"[eagle3] verifier={verifier_name}")
            print(f"[eagle3] candidate={_target_to_list(cand_ids[0], 8)} target={_target_to_list(target_tensor, 8)}")

        if len(tokens) >= max_new_tokens:
            break

        t0 = time.perf_counter()
        total_fed = cand_len
        total_kept = append_count
        keep_idx = max(0, total_kept - 1)
        if total_kept < total_fed:
            past_kv = _crop_kv(vpast, total_kept - total_fed)
            feature = feature_steps[keep_idx]
            next_tok = int(target_tensor[accepted].item())
        else:
            past_kv = vpast
            feature = feature_steps[-1]
            next_tok = int(target_tensor[-1].item())
        stats["rollback_ms"] += (time.perf_counter() - t0) * 1000

        if diagnostic_records is not None:
            eager_target = None
            try:
                hidden_cat = torch.cat(hidden_steps, dim=1)
                eager_target = model.lm_head(hidden_cat).argmax(dim=-1).reshape(-1).to(dtype=torch.long)
            except Exception:
                eager_target = None
            diagnostic_records.append({
                "round": int(stats["rounds"] - 1),
                "generated_before": int(generated_before),
                "base_cache_len": int(base_cache_len),
                "accept_length": int(accepted),
                "append_count": int(append_count),
                "next_token": int(next_tok),
                "candidates": cand_ids.detach().cpu().tolist(),
                "path_target_tokens": target_tensor.reshape(1, -1).detach().cpu().tolist(),
                "eager_target_tokens": (
                    eager_target.reshape(1, -1).detach().cpu().tolist()
                    if isinstance(eager_target, torch.Tensor)
                    else None
                ),
                "retrieve_indices": [list(range(int(cand_len)))],
            })

        if eos_ids and len(tokens) >= min_new_tokens and tokens and tokens[-1] in eos_ids:
            break

    if debug:
        rounds = max(stats["rounds"], 1)
        print(
            f"[eagle3] rounds={stats['rounds']} accepted={stats['accepted']} "
            f"drafted={stats['drafted']} avg_accept={stats['accepted'] / rounds:.1f} "
            f"draft={stats['draft_ms'] / rounds:.1f}ms verify={stats['verify_ms'] / rounds:.1f}ms "
            f"rollback={stats['rollback_ms'] / rounds:.1f}ms verifier={stats['verifier']}"
        )

    return tokens, stats


_legacy_eagle3_tree_speculative_generate = eagle3_tree_speculative_generate
_legacy_eagle3_speculative_generate = eagle3_speculative_generate


def eagle3_tree_speculative_generate(
    model,
    inputs: dict,
    draft_model: Eagle3DraftModel,
    *,
    max_new_tokens: int,
    debug: bool = False,
    min_new_tokens: int = 0,
    layer_indices: tuple[int, int, int] | None = None,
    force_accept_length: int | None = None,
    reference_tokens=None,
):
    """Compatibility entrypoint backed by the refactored EAGLE3 worker."""
    from spec_decode.eagle3_worker import eagle3_generate

    return eagle3_generate(
        model,
        inputs,
        draft_model,
        max_new_tokens=max_new_tokens,
        debug=debug,
        min_new_tokens=min_new_tokens,
        layer_indices=layer_indices,
        force_accept_length=force_accept_length,
        reference_tokens=reference_tokens,
    )


def eagle3_speculative_generate(
    model,
    inputs: dict,
    draft_model: Eagle3DraftModel,
    *,
    max_new_tokens: int,
    draft_len: int = 4,
    debug: bool = False,
    min_new_tokens: int = 0,
    layer_indices: tuple[int, int, int] | None = None,
    force_accept_length: int | None = None,
):
    """Compatibility wrapper for the old linear EAGLE3 entrypoint."""

    del draft_len
    return eagle3_tree_speculative_generate(
        model,
        inputs,
        draft_model,
        max_new_tokens=max_new_tokens,
        debug=debug,
        min_new_tokens=min_new_tokens,
        layer_indices=layer_indices,
        force_accept_length=force_accept_length,
    )
