from __future__ import annotations

import os
from types import MethodType

import torch

from .kernel import rmsnorm_rowwise_kernel

_DECODE_RMSNORM_TRITON_ENABLED = True
_TRACE_ONCE: set[str] = set()


def set_decode_rmsnorm_triton_enabled(enabled: bool) -> None:
    global _DECODE_RMSNORM_TRITON_ENABLED
    _DECODE_RMSNORM_TRITON_ENABLED = bool(enabled)


def _trace_once(key: str, message: str) -> None:
    if os.getenv("AICAS_TIMING_PROFILE", "0") != "1" or key in _TRACE_ONCE:
        return
    _TRACE_ONCE.add(key)
    print(message)


def _seq_len_allowed(seq_len: int) -> bool:
    max_seq = 4
    if os.getenv("AICAS_SPEC_DECODE_RMSNORM_TRITON", "1") == "1":
        max_seq = max(max_seq, int(os.getenv("AICAS_SPEC_KERNEL_MAX_Q", "16")))
    return 1 <= seq_len <= max_seq


def _should_use_kernel(hidden_states: torch.Tensor) -> bool:
    seq_len = int(hidden_states.shape[1]) if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3 else 0
    return (
        _DECODE_RMSNORM_TRITON_ENABLED
        and isinstance(hidden_states, torch.Tensor)
        and hidden_states.is_cuda
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and _seq_len_allowed(seq_len)
        and hidden_states.shape[-1] <= 4096
        and hidden_states.dtype is torch.bfloat16
        and hidden_states.is_contiguous()
    )


def _rmsnorm_decode_only(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    x2d = hidden_states.view(-1, hidden_states.shape[-1])
    m, n = x2d.shape
    if (
        out_buffer is None
        or not isinstance(out_buffer, torch.Tensor)
        or out_buffer.device != x2d.device
        or out_buffer.dtype != x2d.dtype
        or out_buffer.shape != x2d.shape
    ):
        out = torch.empty_like(x2d)
    else:
        out = out_buffer
    rmsnorm_rowwise_kernel[(m,)](
        x2d,
        weight,
        out,
        x2d.stride(0),
        n,
        eps,
    )
    return out.view_as(hidden_states)


def _patch_one_norm(norm_module) -> bool:
    if norm_module is None or getattr(norm_module, "_aicas_decode_rmsnorm_triton", False):
        return False
    if not hasattr(norm_module, "weight") or not hasattr(norm_module, "variance_epsilon"):
        return False

    original_forward = norm_module.forward

    def _forward(self, hidden_states):
        if _should_use_kernel(hidden_states):
            try:
                x2d = hidden_states.view(-1, hidden_states.shape[-1])
                workspace = getattr(self, "_aicas_decode_rmsnorm_workspace", None)
                if (
                    workspace is None
                    or workspace.device != x2d.device
                    or workspace.dtype != x2d.dtype
                    or workspace.shape != x2d.shape
                ):
                    workspace = torch.empty_like(x2d)
                    self._aicas_decode_rmsnorm_workspace = workspace
                _trace_once(
                    f"decode_rmsnorm_{tuple(hidden_states.shape)}",
                    f"[spec_kernel] decode_rmsnorm_triton hit shape={tuple(hidden_states.shape)}",
                )
                return _rmsnorm_decode_only(
                    hidden_states,
                    self.weight,
                    float(self.variance_epsilon),
                    out_buffer=workspace,
                )
            except Exception:
                return original_forward(hidden_states)
        return original_forward(hidden_states)

    norm_module.forward = MethodType(_forward, norm_module)
    norm_module._aicas_decode_rmsnorm_triton = True
    return True


def apply_decode_rmsnorm_triton(model) -> int:
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is None or not hasattr(lm, "layers"):
        return 0

    patched = 0
    for layer in lm.layers:
        patched += int(_patch_one_norm(getattr(layer, "input_layernorm", None)))
        patched += int(_patch_one_norm(getattr(layer, "post_attention_layernorm", None)))

    original_generate = model.generate

    def _generate_with_policy(*args, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", None)
        if max_new_tokens is None:
            enable = True
        else:
            try:
                token_count = int(max_new_tokens)
            except Exception:
                token_count = 0
            min_tokens = int(os.getenv("AICAS_DECODE_FASTPATH_MIN_NEW_TOKENS", "2"))
            max_tokens = int(os.getenv("AICAS_DECODE_FASTPATH_MAX_NEW_TOKENS", "256"))
            enable = min_tokens <= token_count <= max_tokens
        set_decode_rmsnorm_triton_enabled(enable)
        try:
            return original_generate(*args, **kwargs)
        finally:
            set_decode_rmsnorm_triton_enabled(True)

    model.generate = _generate_with_policy
    return patched
