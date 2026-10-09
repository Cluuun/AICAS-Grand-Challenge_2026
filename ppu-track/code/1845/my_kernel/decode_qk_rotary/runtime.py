from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch

_THIS_DIR = Path(__file__).resolve().parent
_DECODE_QK_ROTARY_EXT_ENABLED = True
try:
    from my_kernel.decode_linear_triton.runtime import (
        can_use_decode_linear_triton,
        can_use_decode_linear_triton_rows,
        linear_decode_triton,
        linear_decode_triton_rows_fixed,
    )
except Exception:
    can_use_decode_linear_triton = None
    can_use_decode_linear_triton_rows = None
    linear_decode_triton = None
    linear_decode_triton_rows_fixed = None

try:
    from my_kernel.decode_linear_cublaslt.runtime import (
        can_use_decode_linear_cublaslt,
        linear_decode_cublaslt,
    )
except Exception:
    can_use_decode_linear_cublaslt = None
    linear_decode_cublaslt = None

try:
    from my_kernel.static_cache_update_triton import (
        can_use_static_cache_multi_token_update,
        can_use_static_cache_single_token_update,
        update_static_cache_multi_token_triton,
        update_static_cache_single_token_triton,
    )
except Exception:
    can_use_static_cache_multi_token_update = None
    can_use_static_cache_single_token_update = None
    update_static_cache_multi_token_triton = None
    update_static_cache_single_token_triton = None


def set_decode_qk_rotary_ext_enabled(enabled: bool) -> None:
    global _DECODE_QK_ROTARY_EXT_ENABLED
    _DECODE_QK_ROTARY_EXT_ENABLED = bool(enabled)


_DECODE_QK_ROTARY_GRAPH_LINEAR_DEPTH = 0


def _patch_linear_with_triton_gemv():
    """Replace F.linear with a Triton GEMV version (capture-safe, no RTC)."""
    original = torch.nn.functional.linear

    def _triton_linear(input, weight, bias=None):
        if linear_decode_triton_rows_fixed is None or can_use_decode_linear_triton_rows is None:
            return original(input, weight, bias)
        in_shape = input.shape
        x2d = input.reshape(-1, input.shape[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        if not can_use_decode_linear_triton_rows(x2d, weight, bias):
            return original(input, weight, bias)
        result = linear_decode_triton_rows_fixed(x2d, weight, bias)
        return result.reshape(*in_shape[:-1], -1)

    torch.nn.functional.linear = _triton_linear
    return original


def _restore_linear(original):
    torch.nn.functional.linear = original


@contextmanager
def decode_qk_rotary_graph_linear_mode(enabled: bool = True):
    global _DECODE_QK_ROTARY_GRAPH_LINEAR_DEPTH
    if not enabled:
        yield
        return
    backend = os.getenv("AICAS_DECODE_GRAPH_LINEAR_BACKEND", "cublaslt").strip().lower()
    if backend in {"lt", "cublaslt"}:
        backend = "cublaslt"
    blas_backend = backend if backend in {"cublas", "cublaslt", "default"} else "cublas"
    prev_blas = None
    switch_blas = (
        torch.cuda.is_available()
        and hasattr(torch.backends.cuda, "preferred_blas_library")
        and blas_backend in {"cublas", "cublaslt", "default"}
    )
    if switch_blas:
        try:
            prev_blas = torch.backends.cuda.preferred_blas_library()
            torch.backends.cuda.preferred_blas_library(blas_backend)
        except Exception:
            switch_blas = False

    use_triton_linear = (
        backend == "triton"
        and linear_decode_triton_rows_fixed is not None
        and can_use_decode_linear_triton_rows is not None
    )
    prev_linear = None
    if use_triton_linear:
        prev_linear = _patch_linear_with_triton_gemv()

    _DECODE_QK_ROTARY_GRAPH_LINEAR_DEPTH += 1
    try:
        yield
    finally:
        _DECODE_QK_ROTARY_GRAPH_LINEAR_DEPTH -= 1
        if prev_linear is not None:
            _restore_linear(prev_linear)
        if switch_blas and prev_blas is not None:
            try:
                torch.backends.cuda.preferred_blas_library(prev_blas)
            except Exception:
                pass


def _use_graph_linear_triton() -> bool:
    return (
        _DECODE_QK_ROTARY_GRAPH_LINEAR_DEPTH > 0
        and os.getenv("AICAS_DECODE_GRAPH_LINEAR_BACKEND", "cublaslt").strip().lower() == "triton"
        and linear_decode_triton_rows_fixed is not None
        and can_use_decode_linear_triton_rows is not None
    )


def _graph_linear_backend() -> str:
    backend = os.getenv("AICAS_DECODE_GRAPH_LINEAR_BACKEND", "cublaslt").strip().lower()
    if backend in {"lt", "cublaslt"}:
        return "cublaslt"
    return backend


def _use_decode_linear_cublaslt_ext() -> bool:
    backend = _graph_linear_backend()
    return (
        backend in {"cublaslt_ext", "lt_ext"}
        or os.getenv("AICAS_DECODE_LINEAR_CUBLASLT", "0") == "1"
    ) and linear_decode_cublaslt is not None and can_use_decode_linear_cublaslt is not None


def _advance_static_cache_layer_length(layer, kv_len: int) -> None:
    if kv_len <= 0:
        return
    current = getattr(layer, "cumulative_length", None)
    try:
        if isinstance(current, torch.Tensor):
            current.add_(int(kv_len))
        elif current is None:
            layer.cumulative_length = int(kv_len)
        else:
            layer.cumulative_length = int(current) + int(kv_len)
    except Exception:
        try:
            layer.cumulative_length = int(kv_len) if current is None else int(current) + int(kv_len)
        except Exception:
            pass


def _try_static_cache_triton_update(past_key_values, key_states, value_states, layer_idx: int, cache_position):
    layers = getattr(past_key_values, "layers", None)
    if not layers or layer_idx < 0 or layer_idx >= len(layers):
        return None
    if not isinstance(cache_position, torch.Tensor):
        return None
    layer = layers[layer_idx]
    # Only apply Triton fast-path updates to actual StaticCache layers (which have
    # a pre-allocated max_cache_len buffer).  DynamicCache / DynamicLayer objects
    # also have .keys / .values tensors with the same shape signatures, but their
    # sequence dimension equals the current token count, not a pre-allocated maximum.
    # Writing at cache_position (which may be >= current length) into a DynamicCache
    # tensor would silently be a no-op (the Triton kernel masks out-of-bounds stores),
    # leaving the cache unchanged and preventing the HF DynamicCache.update fallback
    # from ever running.
    if not hasattr(layer, "max_cache_len"):
        return None
    try:
        if not getattr(layer, "is_initialized", False):
            layer.lazy_initialization(key_states)
    except Exception:
        return None
    key_cache = getattr(layer, "keys", None)
    value_cache = getattr(layer, "values", None)
    if not isinstance(key_cache, torch.Tensor) or not isinstance(value_cache, torch.Tensor):
        return None

    if (
        update_static_cache_single_token_triton is not None
        and can_use_static_cache_single_token_update is not None
        and can_use_static_cache_single_token_update(key_cache, value_cache, key_states, value_states, cache_position)
    ):
        update_static_cache_single_token_triton(key_cache, value_cache, key_states, value_states, cache_position)
        _advance_static_cache_layer_length(layer, int(key_states.shape[2]))
        return key_cache, value_cache
    if (
        update_static_cache_multi_token_triton is not None
        and can_use_static_cache_multi_token_update is not None
        and can_use_static_cache_multi_token_update(key_cache, value_cache, key_states, value_states, cache_position)
    ):
        update_static_cache_multi_token_triton(key_cache, value_cache, key_states, value_states, cache_position)
        _advance_static_cache_layer_length(layer, int(key_states.shape[2]))
        return key_cache, value_cache
    return None


def _layer_stable_cache_sources(module, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key_buf = getattr(module, "_aicas_decode_qk_key_update_src", None)
    if (
        not isinstance(key_buf, torch.Tensor)
        or key_buf.shape != key_states.shape
        or key_buf.device != key_states.device
        or key_buf.dtype != key_states.dtype
    ):
        key_buf = torch.empty_like(key_states)
        module._aicas_decode_qk_key_update_src = key_buf
    value_buf = getattr(module, "_aicas_decode_qk_value_update_src", None)
    if (
        not isinstance(value_buf, torch.Tensor)
        or value_buf.shape != value_states.shape
        or value_buf.device != value_states.device
        or value_buf.dtype != value_states.dtype
    ):
        value_buf = torch.empty_like(value_states)
        module._aicas_decode_qk_value_update_src = value_buf
    key_buf.copy_(key_states)
    value_buf.copy_(value_states)
    return key_buf, value_buf


def _decode_rope_cache_enabled() -> bool:
    return os.getenv("AICAS_DECODE_ROPE_CACHE", "0") == "1"


def _is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _decode_rope_cache_initial_len(rotary) -> int:
    try:
        max_len = int(os.getenv("AICAS_DECODE_ROPE_CACHE_MAX_LEN", "4096") or "4096")
    except Exception:
        max_len = 4096
    return max(max_len, 1)


def _get_decode_rope_tables(rotary, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = getattr(rotary, "inv_freq", None)
    if not isinstance(inv_freq, torch.Tensor):
        raise RuntimeError("rotary embedding has no inv_freq tensor")
    if inv_freq.numel() * 2 != 128:
        raise RuntimeError("decode rope cache only supports head_dim=128")

    required_len = max(int(getattr(rotary, "_aicas_decode_rope_cache_required_len", 1)), 1)
    max_len = max(_decode_rope_cache_initial_len(rotary), required_len)
    key = (
        int(x.device.index if x.device.index is not None else torch.cuda.current_device()),
        x.dtype,
        max_len,
        float(getattr(rotary, "attention_scaling", 1.0)),
        inv_freq.data_ptr(),
        int(getattr(inv_freq, "_version", 0)),
    )
    cached_key = getattr(rotary, "_aicas_decode_rope_cache_key", None)
    cached = getattr(rotary, "_aicas_decode_rope_cache", None)
    if cached_key == key and isinstance(cached, tuple) and len(cached) == 2:
        return cached
    if _is_cuda_graph_capturing():
        raise RuntimeError("decode rope cache table is not initialized before CUDA graph capture")

    device = x.device
    positions = torch.arange(max_len, device=device, dtype=torch.float32)
    inv = inv_freq.to(device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    scale = float(getattr(rotary, "attention_scaling", 1.0))
    cos_table = (emb.cos() * scale).to(dtype=x.dtype).contiguous()
    sin_table = (emb.sin() * scale).to(dtype=x.dtype).contiguous()
    rotary._aicas_decode_rope_cache_key = key
    rotary._aicas_decode_rope_cache = (cos_table, sin_table)
    rotary._aicas_decode_rope_cache_out = {}
    return cos_table, sin_table


def _decode_rope_position_index(rotary, position_ids: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(position_ids, torch.Tensor):
        return None
    if position_ids.ndim == 2:
        if int(position_ids.shape[0]) != 1 or int(position_ids.shape[1]) != 1:
            return None
        return position_ids.reshape(-1).to(dtype=torch.long)
    if position_ids.ndim != 3 or int(position_ids.shape[0]) != 3 or int(position_ids.shape[1]) != 1 or int(position_ids.shape[2]) != 1:
        return None

    if _is_cuda_graph_capturing():
        mode = getattr(rotary, "_aicas_decode_rope_cache_capture_mode", None)
        if mode == "same":
            return position_ids[0].reshape(-1).to(dtype=torch.long)
        return position_ids[:, 0, 0].reshape(-1).to(dtype=torch.long)

    try:
        same_mrope = bool(torch.equal(position_ids[0], position_ids[1]) and torch.equal(position_ids[0], position_ids[2]))
    except Exception:
        same_mrope = False
    if same_mrope:
        rotary._aicas_decode_rope_cache_capture_mode = "same"
        return position_ids[0].reshape(-1).to(dtype=torch.long)
    rotary._aicas_decode_rope_cache_capture_mode = "mrope"
    return position_ids[:, 0, 0].reshape(-1).to(dtype=torch.long)


def _decode_rope_cache_forward(rotary, original_forward, ext_mod, x: torch.Tensor, position_ids: torch.Tensor):
    if not _decode_rope_cache_enabled():
        return original_forward(x, position_ids)
    if ext_mod is None or not hasattr(ext_mod, "rope_cache_lookup"):
        return original_forward(x, position_ids)
    if not isinstance(x, torch.Tensor) or not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        return original_forward(x, position_ids)
    if x.ndim < 2 or int(x.shape[-2]) != 1:
        return original_forward(x, position_ids)

    pos = _decode_rope_position_index(rotary, position_ids)
    if pos is None:
        return original_forward(x, position_ids)

    try:
        if not _is_cuda_graph_capturing():
            max_pos = int(pos.max().item())
            if max_pos < 0 or int(pos.min().item()) < 0:
                return original_forward(x, position_ids)
            if max_pos >= _decode_rope_cache_initial_len(rotary):
                rotary._aicas_decode_rope_cache_required_len = max_pos + 1
        cos_table, sin_table = _get_decode_rope_tables(rotary, x)
        if int(pos.numel()) not in (1, 3):
            return original_forward(x, position_ids)
        mrope_section = getattr(rotary, "mrope_section", [24, 20, 20])
        cos_out, sin_out = ext_mod.rope_cache_lookup(
            cos_table,
            sin_table,
            pos,
            int(mrope_section[1]),
            int(mrope_section[2]),
        )
        return cos_out.view(1, 1, 128), sin_out.view(1, 1, 128)
    except Exception:
        return original_forward(x, position_ids)


def _patch_decode_rope_cache(model, ext_mod=None) -> bool:
    if not _decode_rope_cache_enabled():
        return False
    lm = getattr(getattr(model, "model", None), "language_model", None)
    rotary = getattr(lm, "rotary_emb", None) if lm is not None else None
    if rotary is None or getattr(rotary, "_aicas_decode_rope_cache_patched", False):
        return False
    original_forward = rotary.forward
    rotary._aicas_decode_rope_cache_ext = ext_mod

    def _forward(self, x, position_ids):
        return _decode_rope_cache_forward(self, original_forward, self._aicas_decode_rope_cache_ext, x, position_ids)

    rotary._aicas_decode_rope_cache_original = original_forward
    rotary.forward = MethodType(_forward, rotary)
    rotary._aicas_decode_rope_cache_patched = True
    return True


def _load_ext():
    so_path = _THIS_DIR / 'decode_qk_rotary_ext.so'
    if not so_path.is_file():
        cands = sorted(_THIS_DIR.glob('decode_qk_rotary_ext*.so'))
        if not cands:
            raise FileNotFoundError(f'Missing decode_qk_rotary extension in {_THIS_DIR}')
        so_path = cands[-1]
    module_name = 'decode_qk_rotary_ext'
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load extension spec from {so_path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def apply_decode_qk_rotary_ext_fastpath(model):
    strict_pack_check = os.getenv("AICAS_QK_EXT_STRICT_PACK_CHECK", "0") == "1"

    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen
    except Exception:
        return 0
    ext = _load_ext()

    # Register ext.forward as a PyTorch custom op so Dynamo can trace through it
    # during torch.compile.  Without this, Dynamo inserts a graph break at every
    # ext.forward() call, preventing self_attn from being compiled.
    try:
        _DECODE_QK_ROTARY_EXT_OP_NAME = "aicas::decode_qk_rotary_ext"
        if not hasattr(torch.ops.aicas, 'decode_qk_rotary_ext'):
            @torch.library.custom_op(_DECODE_QK_ROTARY_EXT_OP_NAME, mutates_args=())
            def _decode_qk_rotary_ext_op(
                query_states: torch.Tensor,
                key_states: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                q_norm_eps: float,
                q_heads: int,
                k_heads: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                q_out, k_out = ext.forward(query_states, key_states, cos, sin,
                                           q_norm_weight, k_norm_weight,
                                           q_norm_eps, q_heads, k_heads)
                return q_out, k_out

            @_decode_qk_rotary_ext_op.register_fake
            def _(*args, **__):
                # Return fake outputs with same shapes as inputs
                return (args[0].clone(), args[1].clone())
    except Exception:
        pass

    lm = getattr(getattr(model, 'model', None), 'language_model', None)
    if lm is None or not hasattr(lm, 'layers'):
        return 0
    _patch_decode_rope_cache(model, ext)

    # ── Build ONE shared forward (NOT per-layer closure!) ──
    # Store per-layer values as instance attrs so Dynamo sees a single
    # function across all 28 layers — enables torch.compile reuse.

    def _shared_forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
        seq_len = int(hidden_states.shape[1]) if hidden_states.ndim == 3 else 0
        max_q = int(os.getenv("AICAS_DECODE_QK_ROTARY_EXT_MAX_Q", "16"))
        decode_only = (
            _DECODE_QK_ROTARY_EXT_ENABLED
            and getattr(self, "_aicas_decode_qk_rotary_ext_fastpath", False)
            and hidden_states.ndim == 3
            and hidden_states.shape[0] == 1
            and seq_len == 1  # REAL single-token decode only, not cached prefill remainder
            and hidden_states.is_cuda and hidden_states.dtype in (torch.float16, torch.bfloat16)
        )
        if not decode_only:
            orig_forward = getattr(self, "_aicas_orig_forward", None)
            if orig_forward is not None:
                return orig_forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )
            class_orig_forward = getattr(type(self), "_aicas_orig_forward_class", None)
            if class_orig_forward is not None:
                return class_orig_forward(
                    self,
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )
            raise AttributeError(
                f"{type(self).__name__} is missing _aicas_orig_forward for decode_qk_rotary fallback"
            )

        q_norm_weight_local = self._aicas_q_norm_weight
        k_norm_weight_local = self._aicas_k_norm_weight
        q_norm_eps_local = self._aicas_q_norm_eps

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        packed_w = getattr(self, '_aicas_qkv_packed_weight', None)
        packed_w_t = getattr(self, '_aicas_qkv_packed_weight_t', None)
        packed_b = getattr(self, '_aicas_qkv_packed_bias', None)
        needs_repack = packed_w is None or packed_w_t is None
        pack_key = None
        if strict_pack_check:
                    pack_key = (
                        self.q_proj.weight.data_ptr(), int(getattr(self.q_proj.weight, '_version', 0)),
                        self.k_proj.weight.data_ptr(), int(getattr(self.k_proj.weight, '_version', 0)),
                        self.v_proj.weight.data_ptr(), int(getattr(self.v_proj.weight, '_version', 0)),
                    )
                    if not needs_repack:
                        needs_repack = getattr(self, '_aicas_qkv_pack_key', None) != pack_key
        if needs_repack:
                    packed_w = torch.cat([
                        self.q_proj.weight,
                        self.k_proj.weight,
                        self.v_proj.weight,
                    ], dim=0).contiguous()
                    packed_w_t = packed_w.transpose(0, 1).contiguous()
                    if self.q_proj.bias is not None or self.k_proj.bias is not None or self.v_proj.bias is not None:
                        biases = []
                        for proj in (self.q_proj, self.k_proj, self.v_proj):
                            if proj.bias is None:
                                biases.append(torch.zeros(proj.weight.shape[0], device=proj.weight.device, dtype=proj.weight.dtype))
                            else:
                                biases.append(proj.bias)
                        packed_b = torch.cat(biases, dim=0).contiguous()
                    else:
                        packed_b = None
                    self._aicas_qkv_packed_weight = packed_w
                    self._aicas_qkv_packed_weight_t = packed_w_t
                    self._aicas_qkv_packed_bias = packed_b
                    self._aicas_qkv_addmm_out = None
                    if strict_pack_check and pack_key is not None:
                        self._aicas_qkv_pack_key = pack_key

        use_graph_linear_triton = _use_graph_linear_triton()
        use_decode_linear_triton = (
                    os.getenv("AICAS_DECODE_LINEAR_TRITON", "0") == "1"
                    and linear_decode_triton is not None
                    and can_use_decode_linear_triton is not None
        )
        use_decode_linear_cublaslt_ext = _use_decode_linear_cublaslt_ext()
        use_decode_linear_addmm = os.getenv("AICAS_DECODE_LINEAR_ADDMM", "0") == "1"
        use_single_row_mm = os.getenv("AICAS_DECODE_LINEAR_SINGLE_ROW_MM", "1") == "1"
        if use_decode_linear_cublaslt_ext and packed_w_t is not None:
                    x2d = hidden_states.reshape(-1, hidden_states.shape[-1])
                    if not x2d.is_contiguous():
                        x2d = x2d.contiguous()
                    if can_use_decode_linear_cublaslt(x2d, packed_w_t, packed_b):
                        qkv_out = linear_decode_cublaslt(
                            x2d,
                            packed_w_t,
                            packed_b,
                            out=getattr(self, "_aicas_qkv_cublaslt_out", None),
                        )
                        self._aicas_qkv_cublaslt_out = qkv_out
                        qkv = qkv_out.view(*input_shape, -1)
                    else:
                        qkv = torch.nn.functional.linear(hidden_states, packed_w, packed_b)
        elif (
                    use_graph_linear_triton
                    and can_use_decode_linear_triton_rows(hidden_states, packed_w, packed_b)
        ):
                    qkv_out = linear_decode_triton_rows_fixed(
                        hidden_states,
                        packed_w,
                        packed_b,
                        out=getattr(self, "_aicas_qkv_graph_triton_out", None),
                    )
                    self._aicas_qkv_graph_triton_out = qkv_out
                    qkv = qkv_out.view(*input_shape, -1)
        elif use_decode_linear_triton and can_use_decode_linear_triton(hidden_states, packed_w, packed_b):
                    qkv_out = linear_decode_triton(
                        hidden_states,
                        packed_w,
                        packed_b,
                        out=getattr(self, "_aicas_qkv_triton_out", None),
                    )
                    self._aicas_qkv_triton_out = qkv_out
                    qkv = qkv_out.view(*input_shape, -1)
        elif use_decode_linear_addmm and packed_w_t is not None:
                    x2d = hidden_states.reshape(-1, hidden_states.shape[-1])
                    graph_linear_backend = _graph_linear_backend()
                    if x2d.shape[0] == 1:
                        if use_single_row_mm:
                            qkv_out = getattr(self, "_aicas_qkv_addmm_out", None)
                            expected_shape = (x2d.shape[0], packed_w_t.shape[1])
                            if (
                                qkv_out is None
                                or qkv_out.shape != expected_shape
                                or qkv_out.device != x2d.device
                                or qkv_out.dtype != x2d.dtype
                            ):
                                qkv_out = torch.empty(expected_shape, device=x2d.device, dtype=x2d.dtype)
                                self._aicas_qkv_addmm_out = qkv_out
                            if packed_b is None and graph_linear_backend == "cublaslt":
                                zero_bias = getattr(self, "_aicas_qkv_zero_bias", None)
                                if (
                                    zero_bias is None
                                    or zero_bias.shape != expected_shape
                                    or zero_bias.device != x2d.device
                                    or zero_bias.dtype != x2d.dtype
                                ):
                                    zero_bias = torch.zeros(expected_shape, device=x2d.device, dtype=x2d.dtype)
                                    self._aicas_qkv_zero_bias = zero_bias
                                torch.addmm(zero_bias, x2d, packed_w_t, beta=0.0, out=qkv_out)
                            elif packed_b is not None and graph_linear_backend == "cublaslt":
                                torch.addmm(packed_b, x2d, packed_w_t, out=qkv_out)
                            else:
                                torch.mm(x2d, packed_w_t, out=qkv_out)
                            if packed_b is not None and graph_linear_backend != "cublaslt":
                                qkv_out.add_(packed_b)
                            qkv = qkv_out.view(*input_shape, -1)
                        else:
                            x1d = x2d.view(-1)
                            qkv_vec = getattr(self, "_aicas_qkv_mv_out", None)
                            if (
                                qkv_vec is None
                                or qkv_vec.numel() != packed_w.shape[0]
                                or qkv_vec.device != x2d.device
                                or qkv_vec.dtype != x2d.dtype
                            ):
                                qkv_vec = torch.empty((packed_w.shape[0],), device=x2d.device, dtype=x2d.dtype)
                                self._aicas_qkv_mv_out = qkv_vec
                            if packed_b is None and graph_linear_backend == "cublaslt":
                                zero_bias = getattr(self, "_aicas_qkv_zero_bias_vec", None)
                                expected_vec = (packed_w.shape[0],)
                                if (
                                    zero_bias is None
                                    or zero_bias.shape != expected_vec
                                    or zero_bias.device != x2d.device
                                    or zero_bias.dtype != x2d.dtype
                                ):
                                    zero_bias = torch.zeros(expected_vec, device=x2d.device, dtype=x2d.dtype)
                                    self._aicas_qkv_zero_bias_vec = zero_bias
                                torch.addmv(zero_bias, packed_w, x1d, beta=0.0, out=qkv_vec)
                            elif packed_b is not None and graph_linear_backend == "cublaslt":
                                torch.addmv(packed_b, packed_w, x1d, out=qkv_vec)
                            elif packed_b is None:
                                torch.mv(packed_w, x1d, out=qkv_vec)
                            else:
                                torch.addmv(packed_b, packed_w, x1d, out=qkv_vec)
                            qkv = qkv_vec.view(*input_shape, -1)
                    else:
                        qkv_out = getattr(self, "_aicas_qkv_addmm_out", None)
                        expected_shape = (x2d.shape[0], packed_w_t.shape[1])
                        if (
                            qkv_out is None
                            or qkv_out.shape != expected_shape
                            or qkv_out.device != x2d.device
                            or qkv_out.dtype != x2d.dtype
                        ):
                            qkv_out = torch.empty(expected_shape, device=x2d.device, dtype=x2d.dtype)
                            self._aicas_qkv_addmm_out = qkv_out
                        if packed_b is None:
                            if graph_linear_backend == "cublaslt":
                                zero_bias = getattr(self, "_aicas_qkv_zero_bias", None)
                                if (
                                    zero_bias is None
                                    or zero_bias.shape != expected_shape
                                    or zero_bias.device != x2d.device
                                    or zero_bias.dtype != x2d.dtype
                                ):
                                    zero_bias = torch.zeros(expected_shape, device=x2d.device, dtype=x2d.dtype)
                                    self._aicas_qkv_zero_bias = zero_bias
                                torch.addmm(zero_bias, x2d, packed_w_t, beta=0.0, out=qkv_out)
                            else:
                                torch.mm(x2d, packed_w_t, out=qkv_out)
                        else:
                            torch.addmm(packed_b, x2d, packed_w_t, out=qkv_out)
                        qkv = qkv_out.view(*input_shape, -1)
        else:
                    qkv = torch.nn.functional.linear(hidden_states, packed_w, packed_b)
        q_size = self.q_proj.weight.shape[0]
        k_size = self.k_proj.weight.shape[0]
        q_heads = q_size // self.head_dim
        k_heads = k_size // self.head_dim
        query_states = qkv[..., :q_size].reshape(-1, self.head_dim)
        key_states = qkv[..., q_size:q_size + k_size].reshape(-1, self.head_dim)
        value_states = qkv[..., q_size + k_size:].reshape(hidden_shape).transpose(1, 2).contiguous()
        cos, sin = position_embeddings
        try:
                    q_out, k_out = torch.ops.aicas.decode_qk_rotary_ext(
                        query_states,
                        key_states,
                        cos.reshape(-1).contiguous(),
                        sin.reshape(-1).contiguous(),
                        q_norm_weight_local,
                        k_norm_weight_local,
                        q_norm_eps_local,
                        q_heads,
                        k_heads,
                    )
        except Exception:
                    q_out, k_out = ext.forward(
                        query_states,
                        key_states,
                        cos.reshape(-1).contiguous(),
                        sin.reshape(-1).contiguous(),
                        q_norm_weight_local,
                        k_norm_weight_local,
                        q_norm_eps_local,
                        q_heads,
                        k_heads,
                    )
        query_states = q_out.reshape(1, seq_len, q_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = k_out.reshape(1, seq_len, k_heads, self.head_dim).transpose(1, 2).contiguous()

        if past_key_values is not None:
                    if os.getenv("AICAS_DECODE_STABLE_KV_UPDATE_SRC", "0") == "1":
                        key_states, value_states = _layer_stable_cache_sources(self, key_states, value_states)
                    cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}
                    updated = _try_static_cache_triton_update(
                        past_key_values,
                        key_states,
                        value_states,
                        int(self.layer_idx),
                        cache_position,
                    )
                    if updated is None:
                        self._aicas_cache_update_path = "hf"
                        updated = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
                    else:
                        self._aicas_cache_update_path = "triton"
                    key_states, value_states = updated

        attention_interface = qwen.ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, qwen.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        oproj_w_t = getattr(self, "_aicas_oproj_weight_t", None)
        oproj_b = getattr(self, "_aicas_oproj_bias", None)
        oproj_pack_key = None
        oproj_needs_repack = oproj_w_t is None
        if strict_pack_check:
                    oproj_pack_key = (
                        self.o_proj.weight.data_ptr(), int(getattr(self.o_proj.weight, "_version", 0)),
                        (self.o_proj.bias.data_ptr(), int(getattr(self.o_proj.bias, "_version", 0)))
                        if self.o_proj.bias is not None else None,
                    )
                    if not oproj_needs_repack:
                        oproj_needs_repack = getattr(self, "_aicas_oproj_pack_key", None) != oproj_pack_key
        if oproj_needs_repack:
                    oproj_w_t = self.o_proj.weight.transpose(0, 1).contiguous()
                    oproj_b = self.o_proj.bias.contiguous() if self.o_proj.bias is not None else None
                    self._aicas_oproj_weight_t = oproj_w_t
                    self._aicas_oproj_bias = oproj_b
                    self._aicas_oproj_addmm_out = None
                    if strict_pack_check and oproj_pack_key is not None:
                        self._aicas_oproj_pack_key = oproj_pack_key

        if use_decode_linear_cublaslt_ext and oproj_w_t is not None:
                    attn_2d = attn_output.reshape(-1, attn_output.shape[-1])
                    if not attn_2d.is_contiguous():
                        attn_2d = attn_2d.contiguous()
                    if can_use_decode_linear_cublaslt(attn_2d, oproj_w_t, oproj_b):
                        oproj_out = linear_decode_cublaslt(
                            attn_2d,
                            oproj_w_t,
                            oproj_b,
                            out=getattr(self, "_aicas_oproj_cublaslt_out", None),
                        )
                        self._aicas_oproj_cublaslt_out = oproj_out
                        attn_output = oproj_out.view(*input_shape, self.o_proj.weight.shape[0])
                    else:
                        attn_output = self.o_proj(attn_output)
        elif (
                    use_graph_linear_triton
                    and can_use_decode_linear_triton_rows(
                        attn_output, self.o_proj.weight, self.o_proj.bias
                    )
        ):
                    oproj_out = linear_decode_triton_rows_fixed(
                        attn_output,
                        self.o_proj.weight,
                        self.o_proj.bias,
                        out=getattr(self, "_aicas_oproj_graph_triton_out", None),
                    )
                    self._aicas_oproj_graph_triton_out = oproj_out
                    attn_output = oproj_out.view(*input_shape, self.o_proj.weight.shape[0])
        elif use_decode_linear_triton and can_use_decode_linear_triton(
                    attn_output, self.o_proj.weight, self.o_proj.bias
        ):
                    oproj_out = linear_decode_triton(
                        attn_output,
                        self.o_proj.weight,
                        self.o_proj.bias,
                        out=getattr(self, "_aicas_oproj_triton_out", None),
                    )
                    self._aicas_oproj_triton_out = oproj_out
                    attn_output = oproj_out.view(*input_shape, self.o_proj.weight.shape[0])
        elif use_decode_linear_addmm and oproj_w_t is not None:
                    attn_2d = attn_output.reshape(-1, attn_output.shape[-1])
                    if attn_2d.shape[0] == 1:
                        if use_single_row_mm:
                            oproj_out = getattr(self, "_aicas_oproj_addmm_out", None)
                            expected_shape = (attn_2d.shape[0], oproj_w_t.shape[1])
                            if (
                                oproj_out is None
                                or oproj_out.shape != expected_shape
                                or oproj_out.device != attn_2d.device
                                or oproj_out.dtype != attn_2d.dtype
                            ):
                                oproj_out = torch.empty(expected_shape, device=attn_2d.device, dtype=attn_2d.dtype)
                                self._aicas_oproj_addmm_out = oproj_out
                            if oproj_b is None and graph_linear_backend == "cublaslt":
                                zero_bias = getattr(self, "_aicas_oproj_zero_bias", None)
                                if (
                                    zero_bias is None
                                    or zero_bias.shape != expected_shape
                                    or zero_bias.device != attn_2d.device
                                    or zero_bias.dtype != attn_2d.dtype
                                ):
                                    zero_bias = torch.zeros(expected_shape, device=attn_2d.device, dtype=attn_2d.dtype)
                                    self._aicas_oproj_zero_bias = zero_bias
                                torch.addmm(zero_bias, attn_2d, oproj_w_t, beta=0.0, out=oproj_out)
                            elif oproj_b is not None and graph_linear_backend == "cublaslt":
                                torch.addmm(oproj_b, attn_2d, oproj_w_t, out=oproj_out)
                            else:
                                torch.mm(attn_2d, oproj_w_t, out=oproj_out)
                            if oproj_b is not None and graph_linear_backend != "cublaslt":
                                oproj_out.add_(oproj_b)
                            attn_output = oproj_out.view(*input_shape, self.o_proj.weight.shape[0])
                        else:
                            attn_1d = attn_2d.view(-1)
                            oproj_vec = getattr(self, "_aicas_oproj_mv_out", None)
                            if (
                                oproj_vec is None
                                or oproj_vec.numel() != self.o_proj.weight.shape[0]
                                or oproj_vec.device != attn_2d.device
                                or oproj_vec.dtype != attn_2d.dtype
                            ):
                                oproj_vec = torch.empty((self.o_proj.weight.shape[0],), device=attn_2d.device, dtype=attn_2d.dtype)
                                self._aicas_oproj_mv_out = oproj_vec
                            if oproj_b is None and graph_linear_backend == "cublaslt":
                                zero_bias = getattr(self, "_aicas_oproj_zero_bias_vec", None)
                                expected_vec = (self.o_proj.weight.shape[0],)
                                if (
                                    zero_bias is None
                                    or zero_bias.shape != expected_vec
                                    or zero_bias.device != attn_2d.device
                                    or zero_bias.dtype != attn_2d.dtype
                                ):
                                    zero_bias = torch.zeros(expected_vec, device=attn_2d.device, dtype=attn_2d.dtype)
                                    self._aicas_oproj_zero_bias_vec = zero_bias
                                torch.addmv(zero_bias, self.o_proj.weight, attn_1d, beta=0.0, out=oproj_vec)
                            elif oproj_b is not None and graph_linear_backend == "cublaslt":
                                torch.addmv(oproj_b, self.o_proj.weight, attn_1d, out=oproj_vec)
                            elif oproj_b is None:
                                torch.mv(self.o_proj.weight, attn_1d, out=oproj_vec)
                            else:
                                torch.addmv(oproj_b, self.o_proj.weight, attn_1d, out=oproj_vec)
                            attn_output = oproj_vec.view(*input_shape, self.o_proj.weight.shape[0])
                    else:
                        oproj_out = getattr(self, "_aicas_oproj_addmm_out", None)
                        expected_shape = (attn_2d.shape[0], oproj_w_t.shape[1])
                        if (
                            oproj_out is None
                            or oproj_out.shape != expected_shape
                            or oproj_out.device != attn_2d.device
                            or oproj_out.dtype != attn_2d.dtype
                        ):
                            oproj_out = torch.empty(expected_shape, device=attn_2d.device, dtype=attn_2d.dtype)
                            self._aicas_oproj_addmm_out = oproj_out
                        if oproj_b is None and graph_linear_backend == "cublaslt":
                            zero_bias = getattr(self, "_aicas_oproj_zero_bias", None)
                            if (
                                zero_bias is None
                                or zero_bias.shape != expected_shape
                                or zero_bias.device != attn_2d.device
                                or zero_bias.dtype != attn_2d.dtype
                            ):
                                zero_bias = torch.zeros(expected_shape, device=attn_2d.device, dtype=attn_2d.dtype)
                                self._aicas_oproj_zero_bias = zero_bias
                            torch.addmm(zero_bias, attn_2d, oproj_w_t, beta=0.0, out=oproj_out)
                        elif oproj_b is None:
                            torch.mm(attn_2d, oproj_w_t, out=oproj_out)
                        else:
                            torch.addmm(oproj_b, attn_2d, oproj_w_t, out=oproj_out)
                        attn_output = oproj_out.view(*input_shape, self.o_proj.weight.shape[0])
        else:
                    attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # Assign shared forward to each layer (store per-layer values as attrs)
    patched = 0
    for layer in lm.layers:
        attn = getattr(layer, 'self_attn', None)
        if attn is None or getattr(attn, '_aicas_decode_qk_rotary_ext_fastpath', False):
            continue
        attn._aicas_orig_forward = attn.forward
        attn._aicas_q_norm_weight = attn.q_norm.weight.detach().contiguous()
        attn._aicas_k_norm_weight = attn.k_norm.weight.detach().contiguous()
        attn._aicas_q_norm_eps = float(attn.q_norm.variance_epsilon)
        patched += 1

    # Write shared forward at CLASS level — avoids CLOSURE_MATCH guard
    # (nn_module.py: "forward" not in mod.__dict__ → lightweight guard)
    if patched > 0:
        attn_cls = type(lm.layers[0].self_attn)
        if not hasattr(attn_cls, "_aicas_orig_forward_class"):
            attn_cls._aicas_orig_forward_class = attn_cls.forward
        attn_cls.forward = _shared_forward
        for layer in lm.layers:
            attn = getattr(layer, 'self_attn', None)
            if attn is not None:
                attn._aicas_decode_qk_rotary_ext_fastpath = True

    original_generate = model.generate

    def _generate_with_policy(*args, **kwargs):
        max_new_tokens = int(kwargs.get('max_new_tokens', 0) or 0)
        min_tokens = int(os.getenv("AICAS_DECODE_FASTPATH_MIN_NEW_TOKENS", "2"))
        max_tokens = int(os.getenv("AICAS_DECODE_FASTPATH_MAX_NEW_TOKENS", "256"))
        enable = min_tokens <= max_new_tokens <= max_tokens
        set_decode_qk_rotary_ext_enabled(enable)
        try:
            return original_generate(*args, **kwargs)
        finally:
            set_decode_qk_rotary_ext_enabled(True)

    model.generate = _generate_with_policy
    return patched
