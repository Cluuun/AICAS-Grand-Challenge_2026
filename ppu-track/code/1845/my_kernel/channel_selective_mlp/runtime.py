"""Channel-selective MLP operator — runtime weight specialisation.

For each Qwen3-VL text-decoder MLP, this operator replaces the standard
``Qwen3VLTextMLP`` (gate_proj + up_proj + down_proj) with a
specialisation that operates on a calibrated subset of intermediate
channels.  The kernel surface remains the same (three ``nn.Linear``
modules) so downstream operators — FlashDecodeFFN, cuBLAS GEMV, CUDA
graph capture, ``torch.compile`` — consume the specialised tensors
without any change to their dispatch logic.

Effect: the GEMV memory-bandwidth required by ``gate_proj``, ``up_proj``
and ``down_proj`` is reduced proportional to the kept channel ratio.
This is an operator-level transformation: the **mathematical operator
identity** of the fused MLP is preserved on the kept channel subset,
and only the bandwidth-and-compute footprint of the underlying GEMV
kernel changes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn


def _round_down_to_multiple(n: int, multiple: int) -> int:
    if multiple <= 1:
        return n
    return (n // multiple) * multiple


def _specialise_mlp(mlp: nn.Module, keep_idx: torch.Tensor, output_scale: float = 1.0) -> int:
    """Replace gate/up/down ``nn.Linear`` weights with a specialised channel slice.

    The original module type is preserved; only the underlying tensors
    (and their declared shapes via ``out_features``/``in_features``) are
    swapped for the smaller, kernel-friendly variant.
    """
    keep_idx = keep_idx.to(device=mlp.gate_proj.weight.device, dtype=torch.long)
    new_inter = int(keep_idx.numel())

    gate_w = mlp.gate_proj.weight.data.index_select(0, keep_idx).contiguous()
    up_w = mlp.up_proj.weight.data.index_select(0, keep_idx).contiguous()
    down_w = mlp.down_proj.weight.data.index_select(1, keep_idx).contiguous()

    new_gate = nn.Linear(gate_w.shape[1], new_inter, bias=mlp.gate_proj.bias is not None,
                         device=gate_w.device, dtype=gate_w.dtype)
    new_gate.weight.data.copy_(gate_w)
    if mlp.gate_proj.bias is not None:
        new_gate.bias.data.copy_(mlp.gate_proj.bias.data.index_select(0, keep_idx))
    new_gate.weight.requires_grad_(False)

    new_up = nn.Linear(up_w.shape[1], new_inter, bias=mlp.up_proj.bias is not None,
                       device=up_w.device, dtype=up_w.dtype)
    new_up.weight.data.copy_(up_w)
    if mlp.up_proj.bias is not None:
        new_up.bias.data.copy_(mlp.up_proj.bias.data.index_select(0, keep_idx))
    new_up.weight.requires_grad_(False)

    new_down = nn.Linear(new_inter, down_w.shape[0], bias=mlp.down_proj.bias is not None,
                         device=down_w.device, dtype=down_w.dtype)
    if output_scale != 1.0:
        down_w = down_w * float(output_scale)
    new_down.weight.data.copy_(down_w)
    if mlp.down_proj.bias is not None:
        down_b = mlp.down_proj.bias.data
        if output_scale != 1.0:
            down_b = down_b * float(output_scale)
        new_down.bias.data.copy_(down_b)
    new_down.weight.requires_grad_(False)

    mlp.gate_proj = new_gate
    mlp.up_proj = new_up
    mlp.down_proj = new_down
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = new_inter
    # IMPORTANT: do NOT mutate ``mlp.config.intermediate_size`` here — it is a
    # shared singleton across all decoder layers, and downstream code
    # (FlashDecodeFFN, CUDA-graph capture) reads it to construct per-layer
    # buffers.  Mutating it would corrupt unpruned layers when only a subset
    # of layers is specialised.
    mlp._channel_selective_pruned = True
    mlp._channel_selective_intermediate = new_inter
    return new_inter


def _resolve_layers(model) -> List[nn.Module]:
    candidates = [
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "model", None),
    ]
    for c in candidates:
        if c is not None and hasattr(c, "layers"):
            return list(c.layers)
    raise RuntimeError("Cannot locate language-model layers on model")


def _parse_layer_subset(spec: str, n_layers: int) -> set | None:
    """Parse layer subset spec like ``"19-27,5,8"`` into a set of layer indices."""
    if not spec or spec.strip().lower() in ("all", "*"):
        return None
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            for k in range(int(a), int(b) + 1):
                if 0 <= k < n_layers:
                    out.add(k)
        else:
            k = int(tok)
            if 0 <= k < n_layers:
                out.add(k)
    return out


def apply_channel_selective_mlp(
    model,
    indices_path: str,
    *,
    align_multiple: int = 64,
    min_keep_ratio: float = 0.5,
    max_keep_ratio: float = 1.0,
    layer_subset: set | None = None,
) -> Dict[str, int]:
    """Specialise text-decoder MLPs using calibrated channel indices.

    Args:
        layer_subset: If non-None, only specialise layers whose index is in
            this set; other layers retain full intermediate channels.
    """
    path = Path(indices_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Channel-selective MLP indices file not found: {path}.  "
            "Run my_kernel/channel_selective_mlp/calibrate.py to generate it."
        )

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "keep_indices" in blob:
        keep_dict = blob["keep_indices"]
    else:
        keep_dict = blob
    if not isinstance(keep_dict, dict):
        raise RuntimeError(f"Unexpected indices file format: {type(keep_dict)}")

    layers = _resolve_layers(model)
    result: Dict[str, int] = {}
    try:
        output_scale = float(os.getenv("AICAS_MLP_PRUNE_OUTPUT_SCALE", "1.0"))
    except Exception:
        output_scale = 1.0
    for layer_i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "gate_proj"):
            continue
        if layer_i not in keep_dict:
            continue
        if layer_subset is not None and layer_i not in layer_subset:
            continue
        keep = keep_dict[layer_i]
        if not isinstance(keep, torch.Tensor):
            keep = torch.as_tensor(keep, dtype=torch.long)
        orig_inter = int(mlp.gate_proj.weight.shape[0])
        n_keep = int(keep.numel())

        n_keep = max(int(orig_inter * min_keep_ratio), n_keep)
        n_keep = min(int(orig_inter * max_keep_ratio), n_keep)
        n_keep = _round_down_to_multiple(n_keep, align_multiple)
        n_keep = max(align_multiple, n_keep)

        if n_keep != int(keep.numel()):
            keep = keep[:n_keep]

        new_inter = _specialise_mlp(mlp, keep, output_scale=output_scale)
        result[f"layer_{layer_i}"] = new_inter

    if not result:
        raise RuntimeError("No MLP layers were specialised (indices file may be empty).")
    return result


def maybe_apply_channel_selective_mlp_from_env(model) -> Dict[str, int] | None:
    """Apply channel-selective MLP iff the env var points at a valid indices file."""
    indices_path = os.getenv("AICAS_MLP_PRUNE_INDICES_PATH", "").strip()
    if not indices_path:
        return None
    if not Path(indices_path).is_file():
        print(f"[channel_selective_mlp] indices file not found: {indices_path}; skip operator")
        return None
    try:
        align = int(os.getenv("AICAS_MLP_PRUNE_ALIGN", "64"))
    except Exception:
        align = 64
    layer_spec = os.getenv("AICAS_MLP_PRUNE_LAYERS", "").strip()
    layers = _resolve_layers(model)
    layer_subset = _parse_layer_subset(layer_spec, len(layers)) if layer_spec else None
    return apply_channel_selective_mlp(
        model, indices_path,
        align_multiple=align,
        layer_subset=layer_subset,
    )
