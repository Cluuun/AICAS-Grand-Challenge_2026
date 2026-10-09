from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_EXT = None
_PREFILL_RMSNORM_CUDA_ENABLED = True

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


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    candidates = sorted(_THIS_DIR.glob("prefill_rmsnorm_cuda_ext*.so"))
    if not candidates:
        raise FileNotFoundError(f"Missing prefill_rmsnorm_cuda_ext*.so in {_THIS_DIR}")
    so_path = candidates[0]
    spec = importlib.util.spec_from_file_location("prefill_rmsnorm_cuda_ext", so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXT = mod
    return mod


def is_available() -> bool:
    try:
        _load_ext()
        return True
    except Exception:
        return False


def set_prefill_rmsnorm_cuda_enabled(enabled: bool) -> None:
    global _PREFILL_RMSNORM_CUDA_ENABLED
    _PREFILL_RMSNORM_CUDA_ENABLED = bool(enabled)


def _should_use_kernel(hidden_states: torch.Tensor) -> bool:
    if not _PREFILL_RMSNORM_CUDA_ENABLED:
        return False
    if not isinstance(hidden_states, torch.Tensor):
        return False
    if (not hidden_states.is_cuda) or hidden_states.ndim < 2:
        return False
    if hidden_states.dtype is not torch.bfloat16:
        return False
    if not hidden_states.is_contiguous():
        return False
    hidden_size = int(hidden_states.shape[-1])
    if hidden_size <= 0 or hidden_size > 4096:
        return False
    tokens = int(hidden_states.numel() // max(1, hidden_size))
    return tokens > 1


def _rmsnorm_prefill_cuda(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    ext = _load_ext()
    x2d = hidden_states.view(-1, hidden_states.shape[-1])
    if (
        out_buffer is None
        or not isinstance(out_buffer, torch.Tensor)
        or out_buffer.device != x2d.device
        or out_buffer.dtype != x2d.dtype
        or out_buffer.shape != x2d.shape
        or not out_buffer.is_contiguous()
    ):
        out = torch.empty_like(x2d)
    else:
        out = out_buffer
    ext.rmsnorm_into(x2d, weight, float(eps), out)
    return out.view_as(hidden_states)


def _rmsnorm_cuda_forward(self, hidden_states):
    original_forward = getattr(self.__class__, "_aicas_prefill_rmsnorm_cuda_original_forward")
    if _is_dynamo_compiling():
        return original_forward(self, hidden_states)
    if _should_use_kernel(hidden_states):
        try:
            x2d = hidden_states.view(-1, hidden_states.shape[-1])
            workspace = getattr(self, "_aicas_prefill_rmsnorm_cuda_workspace", None)
            if (
                workspace is None
                or workspace.device != x2d.device
                or workspace.dtype != x2d.dtype
                or workspace.shape != x2d.shape
                or not workspace.is_contiguous()
            ):
                workspace = torch.empty_like(x2d)
                self._aicas_prefill_rmsnorm_cuda_workspace = workspace
            return _rmsnorm_prefill_cuda(
                hidden_states,
                self.weight,
                float(self.variance_epsilon),
                out_buffer=workspace,
            )
        except Exception:
            return original_forward(self, hidden_states)
    return original_forward(self, hidden_states)


def _patch_one_norm(norm_module) -> bool:
    if norm_module is None or getattr(norm_module, "_aicas_prefill_rmsnorm_cuda", False):
        return False
    if not hasattr(norm_module, "weight") or not hasattr(norm_module, "variance_epsilon"):
        return False

    cls = norm_module.__class__
    if not getattr(cls, "_aicas_prefill_rmsnorm_cuda_class_patched", False):
        cls._aicas_prefill_rmsnorm_cuda_original_forward = cls.forward
        cls.forward = _rmsnorm_cuda_forward
        cls._aicas_prefill_rmsnorm_cuda_class_patched = True
    norm_module._aicas_prefill_rmsnorm_cuda = True
    return True


def apply_prefill_rmsnorm_cuda(model) -> int:
    _load_ext()
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is None or not hasattr(lm, "layers"):
        return 0

    patched = 0
    for layer in lm.layers:
        patched += int(_patch_one_norm(getattr(layer, "input_layernorm", None)))
        patched += int(_patch_one_norm(getattr(layer, "post_attention_layernorm", None)))
    return patched
