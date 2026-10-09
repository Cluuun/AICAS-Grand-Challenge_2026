from __future__ import annotations

import os

import torch

from .kernel import rmsnorm_prefill_rowwise_kernel

_PREFILL_RMSNORM_TRITON_ENABLED = True

try:
    import torch._dynamo as _torch_dynamo
except Exception:
    _torch_dynamo = None


def _is_dynamo_compiling() -> bool:
    if _torch_dynamo is None:
        return False
    try:
        return bool(_torch_dynamo.is_compiling())
    except Exception:
        return False


def set_prefill_rmsnorm_triton_enabled(enabled: bool) -> None:
    global _PREFILL_RMSNORM_TRITON_ENABLED
    _PREFILL_RMSNORM_TRITON_ENABLED = bool(enabled)


def _should_use_kernel(hidden_states: torch.Tensor) -> bool:
    if not _PREFILL_RMSNORM_TRITON_ENABLED:
        return False
    if not isinstance(hidden_states, torch.Tensor):
        return False
    if (not hidden_states.is_cuda) or hidden_states.ndim < 2:
        return False
    if hidden_states.dtype is not torch.bfloat16:
        return False
    hidden_size = int(hidden_states.shape[-1])
    if hidden_size <= 0 or hidden_size > 4096:
        return False
    min_hidden = max(1, int(os.getenv("AICAS_PREFILL_RMSNORM_TRITON_MIN_HIDDEN", "1")))
    if hidden_size < min_hidden:
        return False
    tokens = int(hidden_states.numel() // max(1, hidden_size))
    min_tokens = max(1, int(os.getenv("AICAS_PREFILL_RMSNORM_TRITON_MIN_TOKENS", "2")))
    if tokens < min_tokens:
        return False
    return hidden_states.is_contiguous()


def _launch_config(n: int) -> tuple[int, int]:
    block_size = 1 << (int(n) - 1).bit_length()
    if block_size <= 256:
        num_warps = 1
    elif block_size <= 1024:
        num_warps = 4
    else:
        num_warps = 8
    return block_size, num_warps


def _rmsnorm_prefill(
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
    block_size, num_warps = _launch_config(n)
    rmsnorm_prefill_rowwise_kernel[(m,)](
        x2d,
        weight,
        out,
        x2d.stride(0),
        n,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=3,
    )
    return out.view_as(hidden_states)


def _rmsnorm_triton_forward(self, hidden_states):
    original_forward = getattr(self.__class__, "_aicas_prefill_rmsnorm_triton_original_forward")
    if _is_dynamo_compiling() or not getattr(self, "_aicas_prefill_rmsnorm_triton", False):
        return original_forward(self, hidden_states)
    if _should_use_kernel(hidden_states):
        try:
            x2d = hidden_states.view(-1, hidden_states.shape[-1])
            key = (x2d.device, x2d.dtype, tuple(x2d.shape), tuple(x2d.stride()))
            workspaces = getattr(self, "_aicas_prefill_rmsnorm_workspaces", None)
            if not isinstance(workspaces, dict):
                workspaces = {}
                self._aicas_prefill_rmsnorm_workspaces = workspaces
            workspace = workspaces.get(key)
            if workspace is None:
                workspace = torch.empty_like(x2d)
                workspaces[key] = workspace
            return _rmsnorm_prefill(
                hidden_states,
                self.weight,
                float(self.variance_epsilon),
                out_buffer=workspace,
            )
        except Exception:
            return original_forward(self, hidden_states)
    return original_forward(self, hidden_states)


def _patch_one_norm(norm_module) -> bool:
    if norm_module is None or getattr(norm_module, "_aicas_prefill_rmsnorm_triton", False):
        return False
    if not hasattr(norm_module, "weight") or not hasattr(norm_module, "variance_epsilon"):
        return False
    weight = getattr(norm_module, "weight", None)
    hidden_size = int(weight.shape[0]) if isinstance(weight, torch.Tensor) and weight.ndim == 1 else 0
    min_hidden = max(1, int(os.getenv("AICAS_PREFILL_RMSNORM_TRITON_MIN_HIDDEN", "1")))
    if hidden_size < min_hidden:
        return False

    cls = norm_module.__class__
    if getattr(cls, "_aicas_prefill_rmsnorm_cuda_class_patched", False):
        return False
    if not getattr(cls, "_aicas_prefill_rmsnorm_triton_class_patched", False):
        cls._aicas_prefill_rmsnorm_triton_original_forward = cls.forward
        cls.forward = _rmsnorm_triton_forward
        cls._aicas_prefill_rmsnorm_triton_class_patched = True
    norm_module._aicas_prefill_rmsnorm_triton = True
    return True


def _patch_attention_norms(layer) -> int:
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return 0
    patched = 0
    patched += int(_patch_one_norm(getattr(attn, "q_norm", None)))
    patched += int(_patch_one_norm(getattr(attn, "k_norm", None)))
    return patched


def apply_prefill_rmsnorm_triton(model) -> int:
    if os.getenv("AICAS_PREFILL_RMSNORM_CUDA", "0") == "1":
        return 0
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is None or not hasattr(lm, "layers"):
        return 0

    patched = 0
    for layer in lm.layers:
        patched += int(_patch_one_norm(getattr(layer, "input_layernorm", None)))
        patched += int(_patch_one_norm(getattr(layer, "post_attention_layernorm", None)))
        patched += _patch_attention_norms(layer)
    patched += int(_patch_one_norm(getattr(lm, "norm", None)))
    return patched
