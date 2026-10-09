"""
Submission-time structural MLP channel pruning for Qwen3-VL-2B-Instruct.

The keep indices were selected offline with public TextVQA calibration and are
bundled in this repository. No calibration data, training, or weight updates are
used at evaluation time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

_KEEP_INDICES_FILE = Path(__file__).with_name(
    "mlp_keep_indices.json"
)

def apply_mlp_channel_pruning(model: nn.Module) -> Dict[str, object]:
    layers = _get_decoder_layers(model)
    if not layers:
        raise RuntimeError("[mlp_prune] Could not find model.model.language_model.layers")

    original_size = int(layers[0].mlp.gate_proj.weight.shape[0])
    keep_indices, meta = _load_keep_indices(_KEEP_INDICES_FILE, len(layers), original_size)
    keep_size = int(keep_indices[0].numel())

    print(f"[mlp_prune] Decoder MLP channel pruning: {original_size} -> {keep_size}")
    print(f"[mlp_prune] keep_indices={_KEEP_INDICES_FILE.name}")

    with torch.no_grad():
        for layer, keep_idx in zip(layers, keep_indices):
            _prune_one_mlp(layer.mlp, keep_idx)

    _update_config_intermediate_size(model, keep_size)
    _verify_shapes(layers, keep_size, log_first_layers=3)
    print("[mlp_prune] Done")

    return {
        "enabled": True,
        "original_intermediate_size": original_size,
        "keep_size": keep_size,
        "prune_ratio": round(1.0 - keep_size / float(original_size), 6),
        "num_layers": len(layers),
        "keep_indices_file": _KEEP_INDICES_FILE.name,
        "score_method": meta.get("score_method", "loaded_indices"),
    }

def _load_keep_indices(path: Path, num_layers: int, original_size: int):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != num_layers:
        raise ValueError(
            f"[mlp_prune] Invalid keep-index file: expected {num_layers} layers"
        )

    file_original = payload.get("original_intermediate_size")
    if file_original is not None and int(file_original) != original_size:
        raise ValueError(
            f"[mlp_prune] Keep-index original size {file_original} != model size {original_size}"
        )

    keep_indices: List[torch.Tensor] = []
    keep_size = None
    for layer_idx, values in enumerate(layers):
        idx = torch.tensor(values, dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError(f"[mlp_prune] Empty keep indices at layer {layer_idx}")
        if keep_size is None:
            keep_size = int(idx.numel())
        elif int(idx.numel()) != keep_size:
            raise ValueError(f"[mlp_prune] Keep size mismatch at layer {layer_idx}")
        if int(idx.min().item()) < 0 or int(idx.max().item()) >= original_size:
            raise ValueError(f"[mlp_prune] Out-of-range keep index at layer {layer_idx}")
        if torch.unique(idx).numel() != idx.numel():
            raise ValueError(f"[mlp_prune] Duplicate keep index at layer {layer_idx}")
        keep_indices.append(torch.sort(idx).values)

    return keep_indices, payload

def _prune_one_mlp(mlp: nn.Module, keep_idx: torch.Tensor) -> None:
    keep_idx = keep_idx.to(device=mlp.gate_proj.weight.device)
    mlp.gate_proj = _make_pruned_linear(mlp.gate_proj, row_idx=keep_idx)
    mlp.up_proj = _make_pruned_linear(mlp.up_proj, row_idx=keep_idx)
    mlp.down_proj = _make_pruned_linear(mlp.down_proj, col_idx=keep_idx)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = int(keep_idx.numel())

def _make_pruned_linear(
    linear: nn.Linear,
    row_idx: torch.Tensor | None = None,
    col_idx: torch.Tensor | None = None,
) -> nn.Linear:
    weight = linear.weight.detach()
    bias = linear.bias.detach() if linear.bias is not None else None

    if row_idx is not None:
        weight = weight.index_select(0, row_idx)
        if bias is not None:
            bias = bias.index_select(0, row_idx)
    if col_idx is not None:
        weight = weight.index_select(1, col_idx.to(device=weight.device))

    new_linear = nn.Linear(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    new_linear.weight = nn.Parameter(weight.contiguous())
    if bias is not None:
        new_linear.bias = nn.Parameter(bias.contiguous())
    return new_linear

def _get_decoder_layers(model: nn.Module):
    try:
        return list(model.model.language_model.layers)
    except AttributeError:
        return []

def _update_config_intermediate_size(model: nn.Module, keep_size: int) -> None:
    cfg = getattr(model, "config", None)
    configs = []
    if cfg is not None:
        configs.append(cfg)
        if hasattr(cfg, "get_text_config"):
            try:
                configs.append(cfg.get_text_config())
            except Exception:
                pass
        configs.append(getattr(cfg, "text_config", None))
    try:
        configs.append(model.model.language_model.config)
    except AttributeError:
        pass

    seen = set()
    for obj in configs:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "intermediate_size"):
            obj.intermediate_size = keep_size

def _verify_shapes(layers: Sequence[nn.Module], keep_size: int, log_first_layers: int) -> None:
    hidden_size = int(layers[0].mlp.down_proj.weight.shape[0])
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        gate_shape = tuple(mlp.gate_proj.weight.shape)
        up_shape = tuple(mlp.up_proj.weight.shape)
        down_shape = tuple(mlp.down_proj.weight.shape)
        expected_gate = (keep_size, hidden_size)
        expected_down = (hidden_size, keep_size)
        if gate_shape != expected_gate or up_shape != expected_gate or down_shape != expected_down:
            raise RuntimeError(
                f"[mlp_prune] Shape mismatch at layer {i}: "
                f"gate={gate_shape} up={up_shape} down={down_shape}"
            )
