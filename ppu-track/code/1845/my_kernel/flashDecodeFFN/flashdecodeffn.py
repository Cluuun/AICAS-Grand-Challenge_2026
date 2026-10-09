#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP as TorchQwen3VLTextMLP

from aicas_env import (
    apply_aicas_env_defaults,
    flashdecode_ffn_allow_legacy,
    flashdecode_ffn_backend,
    get_int,
)

_THIS_DIR = Path(__file__).resolve().parent
_FLASHDECODE_RUNTIME_ENABLED = True

apply_aicas_env_defaults()


def _ensure_qwen3vl_language_model_alias() -> None:
    """
    Compatibility shim for wrappers that still access `model.language_model`.
    Newer transformers Qwen3VLForConditionalGeneration exposes `model` only.
    """
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen3_vl_modeling
    except Exception:
        return

    cls = getattr(qwen3_vl_modeling, "Qwen3VLForConditionalGeneration", None)
    if cls is None or hasattr(cls, "language_model"):
        return

    def _get_language_model(self):
        core_model = getattr(self, "model", None)
        if core_model is None:
            raise AttributeError("Qwen3VLForConditionalGeneration has no model attribute")
        nested = getattr(core_model, "language_model", None)
        return nested if nested is not None else core_model

    cls.language_model = property(_get_language_model)


_ensure_qwen3vl_language_model_alias()


def _load_flashdecode_ext() -> object:
    so_path = _THIS_DIR / "flashdecodeffn_ext.so"
    if not so_path.is_file():
        raise FileNotFoundError(f"Missing flashdecode extension: {so_path}")

    module_name = "flashdecodeffn_ext"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load extension spec from {so_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


flashdecodeffn_ext = None


def _load_flashdecode_cublaslt_ext() -> object | None:
    so_dir = _THIS_DIR / "cublaslt_kernel"
    if not so_dir.is_dir():
        return None
    so_path = so_dir / "ffn_cublaslt_ext.so"
    if not so_path.is_file():
        cands = sorted(so_dir.glob("ffn_cublaslt_ext*.so"))
        if not cands:
            return None
        so_path = cands[-1]
    module_name = "ffn_cublaslt_ext"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


flashdecodeffn_cublaslt_ext = None


def _load_flashdecode_v3_ext() -> object | None:
    so_dir = _THIS_DIR / "v3_kernel"
    if not so_dir.is_dir():
        return None
    so_path = so_dir / "flashdecodeffn_v3_ext.so"
    if not so_path.is_file():
        cands = sorted(so_dir.glob("flashdecodeffn_v3_ext*.so"))
        if not cands:
            return None
        so_path = cands[-1]
    module_name = "flashdecodeffn_v3_ext"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, str(so_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


flashdecodeffn_v3_ext = None


def _get_flashdecode_ext() -> object | None:
    """Legacy binary is not PPU-compatible unless explicitly rebuilt/allowed."""
    global flashdecodeffn_ext
    if flashdecodeffn_ext is not None:
        return flashdecodeffn_ext
    if not flashdecode_ffn_allow_legacy():
        return None
    flashdecodeffn_ext = _load_flashdecode_ext()
    return flashdecodeffn_ext


def _get_flashdecode_cublaslt_ext() -> object | None:
    global flashdecodeffn_cublaslt_ext
    if flashdecodeffn_cublaslt_ext is None:
        flashdecodeffn_cublaslt_ext = _load_flashdecode_cublaslt_ext()
    return flashdecodeffn_cublaslt_ext


def _get_flashdecode_v3_ext() -> object | None:
    global flashdecodeffn_v3_ext
    if flashdecodeffn_v3_ext is None:
        flashdecodeffn_v3_ext = _load_flashdecode_v3_ext()
    return flashdecodeffn_v3_ext


def set_flashdecode_runtime_enabled(enabled: bool) -> None:
    global _FLASHDECODE_RUNTIME_ENABLED
    _FLASHDECODE_RUNTIME_ENABLED = bool(enabled)


def is_flashdecode_runtime_enabled() -> bool:
    return _FLASHDECODE_RUNTIME_ENABLED


class FlashDecodeFFN(nn.Module):
    def __init__(self, use_prealloc: bool = False):
        super().__init__()
        self.use_prealloc = use_prealloc
        backend = flashdecode_ffn_backend()
        if backend == "v3" and _get_flashdecode_v3_ext() is not None:
            self.backend = "v3"
        elif backend == "cublaslt" and _get_flashdecode_cublaslt_ext() is not None:
            self.backend = "cublaslt"
        elif backend == "legacy" and _get_flashdecode_ext() is not None:
            self.backend = "legacy"
        else:
            self.backend = "disabled"

        self._w13_packed: Optional[torch.Tensor] = None
        self._w2_packed: Optional[torch.Tensor] = None
        self._pack_key: Optional[Tuple[object, ...]] = None

        self._out13: Optional[torch.Tensor] = None
        self._gated: Optional[torch.Tensor] = None
        self._out: Optional[torch.Tensor] = None

    def _maybe_repack(self, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> None:
        key = (
            w1.data_ptr(),
            int(getattr(w1, "_version", 0)),
            w3.data_ptr(),
            int(getattr(w3, "_version", 0)),
            w2.data_ptr(),
            int(getattr(w2, "_version", 0)),
            tuple(w1.shape),
            tuple(w3.shape),
            tuple(w2.shape),
            w1.device,
            w1.dtype,
        )
        if key == self._pack_key:
            return

        if self.backend == "v3":
            flashdecodeffn_v3_ext = _get_flashdecode_v3_ext()
            if flashdecodeffn_v3_ext is None:
                raise RuntimeError("v3 backend requested but extension not found")
            pack_w13 = getattr(flashdecodeffn_v3_ext, "pack_text_w13_v2", None)
            if pack_w13 is None:
                pack_w13 = getattr(flashdecodeffn_v3_ext, "pack_w13", None)
            pack_w2 = getattr(flashdecodeffn_v3_ext, "pack_text_w2_v2", None)
            if pack_w2 is None:
                pack_w2 = getattr(flashdecodeffn_v3_ext, "pack_w2", None)
            if pack_w13 is None or pack_w2 is None:
                raise RuntimeError("v3 backend pack functions are unavailable")
            self._w13_packed = pack_w13(
                w1.detach().contiguous(),
                w3.detach().contiguous(),
            )
            self._w2_packed = pack_w2(w2.detach().contiguous())
        elif self.backend == "cublaslt":
            flashdecodeffn_cublaslt_ext = _get_flashdecode_cublaslt_ext()
            if flashdecodeffn_cublaslt_ext is None:
                raise RuntimeError("cublaslt backend requested but extension not found")
            self._w13_packed = flashdecodeffn_cublaslt_ext.pack_text_w13(
                w1.detach().contiguous(),
                w3.detach().contiguous(),
            )
            self._w2_packed = flashdecodeffn_cublaslt_ext.pack_text_w2(w2.detach().contiguous())
        else:
            flashdecodeffn_ext = _get_flashdecode_ext()
            if flashdecodeffn_ext is None:
                raise RuntimeError("legacy backend requested but extension not found")
            self._w13_packed = flashdecodeffn_ext.pack_w13(
                w1.detach().contiguous(),
                w3.detach().contiguous(),
            )
            self._w2_packed = flashdecodeffn_ext.pack_w2(w2.detach().contiguous())
        self._pack_key = key

        self._out13 = None
        self._gated = None
        self._out = None

    def _ensure_workspace(
        self,
        tokens: int,
        hidden_size: int,
        intermediate_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        needs_new = (
            self._out13 is None
            or self._out13.shape != (tokens, 2 * intermediate_size)
            or self._out13.device != device
            or self._out13.dtype != dtype
        )
        if not needs_new:
            return

        self._out13 = torch.empty(tokens, 2 * intermediate_size, device=device, dtype=dtype)
        self._gated = torch.empty(tokens, intermediate_size, device=device, dtype=dtype)
        self._out = torch.empty(tokens, hidden_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        self._maybe_repack(w1, w3, w2)

        if self._w13_packed is None or self._w2_packed is None:
            raise RuntimeError("Packed weights are not initialized.")

        if self.backend == "v3":
            flashdecodeffn_v3_ext = _get_flashdecode_v3_ext()
            if flashdecodeffn_v3_ext is None:
                raise RuntimeError("v3 backend extension is unavailable")
            forward_text_v2 = getattr(flashdecodeffn_v3_ext, "forward_text_v2", None)
            if callable(forward_text_v2):
                return forward_text_v2(x, self._w13_packed, self._w2_packed)
            return flashdecodeffn_v3_ext.forward_packed(x, self._w13_packed, self._w2_packed)

        if self.backend == "cublaslt":
            flashdecodeffn_cublaslt_ext = _get_flashdecode_cublaslt_ext()
            if flashdecodeffn_cublaslt_ext is None:
                raise RuntimeError("cublaslt backend extension is unavailable")
            return flashdecodeffn_cublaslt_ext.text_forward(x, self._w13_packed, self._w2_packed)

        flashdecodeffn_ext = _get_flashdecode_ext()
        if flashdecodeffn_ext is None:
            raise RuntimeError("legacy backend extension is unavailable")

        if self.use_prealloc:
            tokens = x.shape[0]
            intermediate_size = w1.shape[0]
            hidden_size = w2.shape[0]
            self._ensure_workspace(tokens, hidden_size, intermediate_size, x.device, x.dtype)
            if self._out13 is None or self._gated is None or self._out is None:
                raise RuntimeError("Failed to allocate preallocated buffers.")
            return flashdecodeffn_ext.forward_packed_into(
                x,
                self._w13_packed,
                self._w2_packed,
                self._out13,
                self._gated,
                self._out,
            )

        return flashdecodeffn_ext.forward_packed(x, self._w13_packed, self._w2_packed)


# Tell Dynamo to skip FlashDecodeFFN.forward (uses PyCapsule C extensions
# that cannot be traced).  Use disable(), not allow_in_graph() — the
# latter would instruct Dynamo to trace *into* the untraceable C code.
try:
    FlashDecodeFFN.forward = torch.compiler.disable(FlashDecodeFFN.forward)
except Exception:
    pass


class _Qwen3VLTextMLP_FDF(TorchQwen3VLTextMLP):
    def __init__(self, config: Qwen3VLTextConfig, use_prealloc: bool = False):
        super().__init__(config)
        self.flashdecodeffn = FlashDecodeFFN(use_prealloc=use_prealloc)
        self._min_flashdecode_tokens = get_int("FLASHDECODE_MIN_TOKENS", 1, minimum=1)
        self._max_flashdecode_tokens = get_int("FLASHDECODE_MAX_TOKENS", 2147483647, minimum=1)

    def _can_use_flashdecodeffn(self, hidden_states: torch.Tensor) -> bool:
        hidden_act = str(getattr(self.config, "hidden_act", "")).lower()
        tokens = hidden_states.numel() // hidden_states.shape[-1]

        if not _FLASHDECODE_RUNTIME_ENABLED:
            return False
        if getattr(self.flashdecodeffn, "backend", "disabled") == "disabled":
            return False
        if not hidden_states.is_cuda:
            return False
        if hidden_states.dtype != torch.bfloat16:
            return False
        if hidden_act != "silu":
            return False
        if self.training:
            return False
        if torch.is_grad_enabled():
            return False
        if tokens < self._min_flashdecode_tokens:
            return False
        if tokens > self._max_flashdecode_tokens:
            return False
        return True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._can_use_flashdecodeffn(hidden_states):
            return super().forward(hidden_states)

        original_shape = hidden_states.shape
        x2d = hidden_states.reshape(-1, original_shape[-1]).contiguous()
        y2d = self.flashdecodeffn(
            x2d,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
        )
        return y2d.reshape(*original_shape[:-1], y2d.shape[-1])


def _looks_like_qwen3vl_text_mlp(module: object) -> bool:
    return (
        isinstance(module, nn.Module)
        and hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
        and hasattr(module, "config")
    )


def from_torch_qwen3vl_text_mlp(base_mlp: nn.Module, *, use_prealloc: bool = False) -> nn.Module:
    if not _looks_like_qwen3vl_text_mlp(base_mlp):
        raise TypeError("base_mlp must be a Qwen3VLTextMLP-like module.")
    config = base_mlp.config
    converted = _Qwen3VLTextMLP_FDF(config, use_prealloc=use_prealloc)
    if getattr(converted, "_min_flashdecode_tokens", 0) >= (1 << 29):
        return base_mlp

    base_inter = int(base_mlp.gate_proj.weight.shape[0])
    conv_inter = int(converted.gate_proj.weight.shape[0])
    if base_inter != conv_inter:
        gw = base_mlp.gate_proj.weight
        uw = base_mlp.up_proj.weight
        dw = base_mlp.down_proj.weight
        hidden = int(gw.shape[1])
        converted.gate_proj = nn.Linear(hidden, base_inter,
                                        bias=base_mlp.gate_proj.bias is not None,
                                        device=gw.device, dtype=gw.dtype)
        converted.up_proj = nn.Linear(hidden, base_inter,
                                      bias=base_mlp.up_proj.bias is not None,
                                      device=uw.device, dtype=uw.dtype)
        converted.down_proj = nn.Linear(base_inter, int(dw.shape[0]),
                                        bias=base_mlp.down_proj.bias is not None,
                                        device=dw.device, dtype=dw.dtype)

    converted.load_state_dict(base_mlp.state_dict(), strict=True)
    gate_weight = base_mlp.gate_proj.weight
    converted = converted.to(device=gate_weight.device, dtype=gate_weight.dtype)
    converted.train(base_mlp.training)
    return converted


def make_qwen3vl_text_mlp_fdf(*, use_prealloc: bool = False) -> Callable[[Qwen3VLTextConfig], nn.Module]:
    def _ctor(config: Qwen3VLTextConfig) -> nn.Module:
        return _Qwen3VLTextMLP_FDF(config, use_prealloc=use_prealloc)

    return _ctor
