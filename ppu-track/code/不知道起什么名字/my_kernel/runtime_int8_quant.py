from __future__ import annotations

"""Lightweight runtime quantization probes for the custom decode path.

This module intentionally avoids qlean/vLLM as inference frameworks.  It can
use small, directly loaded lower-level kernels when available, and leaves the
original FP weights in place so prefill/fallback paths keep working.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Set

import torch
from torch.utils.cpp_extension import load

from .conf import conf_bool, conf_int, conf_str


@dataclass
class RuntimeInt8LinearState:
    qweight: torch.Tensor
    qweight_t: Optional[torch.Tensor]
    weight_scale: torch.Tensor
    bias: Optional[torch.Tensor]
    out_features: int
    in_features: int
    backend: str


_ACEXT_INT8_GEMM = None
_ACEXT_IMPORT_TRIED = False
_ACEXT_WO_EXT = None
_ACEXT_WO_EXT_TRIED = False


def _load_acext_int8_gemm():
    global _ACEXT_IMPORT_TRIED, _ACEXT_INT8_GEMM
    if _ACEXT_IMPORT_TRIED:
        return _ACEXT_INT8_GEMM
    _ACEXT_IMPORT_TRIED = True
    try:
        from acext import int8_gemm

        _ACEXT_INT8_GEMM = int8_gemm
    except Exception as exc:
        print(f"[runtime_int8_quant] acext int8_gemm unavailable: {type(exc).__name__}: {exc}")
        _ACEXT_INT8_GEMM = None
    return _ACEXT_INT8_GEMM


def _load_acext_weightonly_ext():
    global _ACEXT_WO_EXT_TRIED, _ACEXT_WO_EXT
    if _ACEXT_WO_EXT_TRIED:
        return _ACEXT_WO_EXT
    _ACEXT_WO_EXT_TRIED = True
    try:
        import importlib.util
        from pathlib import Path

        acext_spec = importlib.util.find_spec("acext")
        if acext_spec is None or not acext_spec.submodule_search_locations:
            print("[runtime_int8_quant] acext package unavailable for weight-only extension")
            return None
        site_packages = Path(list(acext_spec.submodule_search_locations)[0]).parent
        src = Path(__file__).resolve().parent / "cpp_ext" / "weightonly_ext.cpp"
        _ACEXT_WO_EXT = load(
            name="aicas_weightonly_ext_runtime",
            sources=[str(src)],
            extra_include_paths=[
                str(site_packages / "include" / "acext"),
                "/usr/local/cuda/include",
            ],
            extra_ldflags=[
                f"-L{site_packages / 'lib'}",
                f"-Wl,-rpath,{site_packages / 'lib'}",
                "-lacext",
                "-L/usr/local/cuda/lib64",
                "-Wl,-rpath,/usr/local/cuda/lib64",
                "-lcudart",
            ],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    except Exception as exc:
        print(f"[runtime_int8_quant] acext weight-only extension unavailable: {type(exc).__name__}: {exc}")
        _ACEXT_WO_EXT = None
    return _ACEXT_WO_EXT


def _parse_targets() -> Set[str]:
    raw = conf_str("RUNTIME_INT8_QUANT_TARGETS", "mlp_down", lower=True)
    if raw in ("", "none", "off", "0"):
        return set()
    targets = {part.strip() for part in raw.replace(";", ",").split(",") if part.strip()}
    if "mlp_all" in targets:
        targets.update({"mlp_gate_up", "mlp_down"})
    return targets


def _select_layer_indices(num_layers: int) -> Set[int]:
    spec = conf_str("RUNTIME_INT8_QUANT_LAYERS", "last:4", lower=True)
    if spec in ("all", "*"):
        return set(range(num_layers))
    selected: Set[int] = set()
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("last:"):
            count = max(0, int(part.split(":", 1)[1]))
            selected.update(range(max(0, num_layers - count), num_layers))
        elif part.startswith("first:"):
            count = max(0, int(part.split(":", 1)[1]))
            selected.update(range(min(num_layers, count)))
        elif part.startswith("range:"):
            _, start, end = part.split(":", 2)
            selected.update(range(max(0, int(start)), min(num_layers, int(end))))
        else:
            idx = int(part)
            if idx < 0:
                idx = num_layers + idx
            if 0 <= idx < num_layers:
                selected.add(idx)
    return selected


@torch.no_grad()
def _quantize_weight_rowwise(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp = weight.detach()
    weight_abs = weight_fp.abs().amax(dim=1).clamp(min=1e-6)
    weight_scale = (weight_abs / 127.0).to(torch.float32).contiguous()
    qweight = torch.round(weight_fp.float() / weight_scale[:, None])
    qweight = qweight.clamp(-127, 127).to(torch.int8).contiguous()
    return qweight, weight_scale


@torch.no_grad()
def build_int8_linear_state(weight: torch.Tensor, bias: Optional[torch.Tensor], *, backend: str) -> RuntimeInt8LinearState:
    qweight, weight_scale = _quantize_weight_rowwise(weight)
    qweight_t = (
        qweight.transpose(0, 1).contiguous()
        if backend in ("torch_int_mm", "triton_w8a16_gemv", "acext_wo_gemv")
        else None
    )
    if backend == "acext_wo_gemv":
        ext = _load_acext_weightonly_ext()
        if ext is None:
            raise RuntimeError("acext weight-only extension is unavailable")
        if weight.device.type != "cuda":
            raise RuntimeError("acext_wo_gemv requires CUDA weights")
        qweight_t_cpu = qweight.transpose(0, 1).contiguous().cpu()
        qweight_t = ext.preprocess_int8_weight(qweight_t_cpu, 80).to(device=weight.device, non_blocking=False)
        weight_scale = weight_scale.to(dtype=weight.dtype).contiguous()
    return RuntimeInt8LinearState(
        qweight=qweight,
        qweight_t=qweight_t,
        weight_scale=weight_scale,
        bias=bias.detach().contiguous() if bias is not None else None,
        out_features=int(weight.shape[0]),
        in_features=int(weight.shape[1]),
        backend=backend,
    )


def _quantize_activation_per_token(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    act_abs = x2d.abs().amax(dim=1).clamp(min=1e-6)
    act_scale = (act_abs / 127.0).to(torch.float32).contiguous()
    xq = torch.round(x2d.float() / act_scale[:, None])
    xq = xq.clamp(-127, 127).to(torch.int8).contiguous()
    return xq, act_scale


def runtime_int8_linear(x: torch.Tensor, state: Optional[RuntimeInt8LinearState]) -> Optional[torch.Tensor]:
    if state is None:
        return None
    if not x.is_cuda:
        return None
    if x.shape[-1] != state.in_features:
        return None
    if state.in_features % 16 != 0 or state.out_features % 16 != 0:
        return None

    orig_shape = tuple(x.shape[:-1])
    x2d = x.reshape(-1, state.in_features).contiguous()

    if state.backend == "acext_w8a8":
        xq, act_scale = _quantize_activation_per_token(x2d)
        int8_gemm = _load_acext_int8_gemm()
        if int8_gemm is None:
            return None
        out = int8_gemm(xq, state.qweight, state.weight_scale, act_scale, state.bias, x.dtype)
    elif state.backend == "torch_int_mm":
        if state.qweight_t is None or not hasattr(torch, "_int_mm"):
            return None
        xq, act_scale = _quantize_activation_per_token(x2d)
        out_i32 = torch._int_mm(xq, state.qweight_t)
        out = out_i32.float()
        out.mul_(act_scale.view(-1, 1))
        out.mul_(state.weight_scale.view(1, -1))
        if state.bias is not None:
            out.add_(state.bias.view(1, -1))
        out = out.to(dtype=x.dtype)
    elif state.backend == "triton_w8a16_gemv":
        if state.qweight_t is None:
            return None
        try:
            from .triton_w8a16_gemv import triton_w8a16_gemv
        except Exception as exc:
            print(f"[runtime_int8_quant] triton_w8a16_gemv unavailable: {type(exc).__name__}: {exc}")
            return None
        out = triton_w8a16_gemv(x2d, state.qweight_t, state.weight_scale, state.bias)
        if out is None:
            return None
    elif state.backend == "acext_wo_gemv":
        if state.qweight_t is None:
            return None
        ext = _load_acext_weightonly_ext()
        if ext is None:
            return None
        if x2d.dtype != torch.float16:
            return None
        out = ext.weightonly_gemv(x2d, state.qweight_t, state.weight_scale)
        if state.bias is not None:
            out.add_(state.bias.view(1, -1))
    else:
        return None
    return out.reshape(*orig_shape, state.out_features)


def _cat_bias(modules: Iterable[torch.nn.Linear]) -> Optional[torch.Tensor]:
    biases = [module.bias for module in modules]
    if any(bias is None for bias in biases):
        return None
    return torch.cat([bias for bias in biases if bias is not None], dim=0).contiguous()


def apply_runtime_int8_quant(model) -> bool:
    if not conf_bool("ENABLE_RUNTIME_INT8_QUANT", "0"):
        return False

    backend = conf_str("RUNTIME_INT8_QUANT_BACKEND", "acext_w8a8", lower=True)
    if backend not in ("acext_w8a8", "torch_int_mm", "triton_w8a16_gemv", "acext_wo_gemv"):
        print(f"[runtime_int8_quant] unsupported backend={backend}, disabled")
        return False
    if backend == "acext_w8a8" and _load_acext_int8_gemm() is None:
        return False
    if backend == "torch_int_mm" and not hasattr(torch, "_int_mm"):
        print("[runtime_int8_quant] torch._int_mm unavailable, disabled")
        return False
    if backend == "acext_wo_gemv" and _load_acext_weightonly_ext() is None:
        return False

    targets = _parse_targets()
    if not targets:
        print("[runtime_int8_quant] no targets selected")
        return False

    language_model = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(language_model, "layers", None)
    if not layers:
        return False

    selected = _select_layer_indices(len(layers))
    max_query_len = conf_int("RUNTIME_INT8_MAX_QUERY_LEN", "1", minimum=1)
    patched_gate_up = 0
    patched_down = 0

    for idx, layer in enumerate(layers):
        if idx not in selected:
            continue
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        mlp._runtime_int8_max_query_len = max_query_len
        if "mlp_gate_up" in targets and all(hasattr(mlp, name) for name in ("gate_proj", "up_proj")):
            weight = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).contiguous()
            bias = _cat_bias((mlp.gate_proj, mlp.up_proj))
            mlp._runtime_int8_gate_up_state = build_int8_linear_state(weight, bias, backend=backend)
            patched_gate_up += 1
        if "mlp_down" in targets and hasattr(mlp, "down_proj"):
            mlp._runtime_int8_down_state = build_int8_linear_state(
                mlp.down_proj.weight,
                mlp.down_proj.bias,
                backend=backend,
            )
            patched_down += 1

    patched_any = patched_gate_up > 0 or patched_down > 0
    if patched_any:
        model._runtime_int8_quant_patched = True
        print(
            "[runtime_int8_quant] enabled "
            f"backend={backend} layers={sorted(selected)} "
            f"targets={sorted(targets)} gate_up={patched_gate_up} down={patched_down} "
            f"max_q_len={max_query_len}"
        )
    return patched_any
