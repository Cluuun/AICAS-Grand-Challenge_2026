"""ViT W8A8 SmoothQuant bundle for Qwen3-VL-2B vision tower.

Narrow Stage 2: only `linear_fc1` of selected blocks is quantized to W8A8.
Op-level bench on PPU shows:
  - vit_fc1 (M~840 post-prune):  W8A8 ≈ 1.06×  vs FP16
  - vit_fc1 (M~1500 pre-prune):  W8A8 ≈ 1.27×  vs FP16
All other ViT shapes (qkv/proj/fc2 across the M range we use) are <1×, so
they are intentionally NOT quantized.

Per ACCURACY_STRATEGY: blocks[0:3] kept FP16 (early-layer feature integrity);
blocks[3:end] eligible. End-of-net safe block window is configurable.

API:
    bundle = build_vit_fc1_w8a8_bundle(model, layer_range=(3, 23))
    bundle.fc1[layer_idx] is W8A8Weight or None

The patch is consumed inside `_make_manual_vit_forward_fused` in
evaluation_wrapper_vAxel26_quant.py: when the bundle slot is non-None,
the per-block fc1 GEMM dispatches through `w8a8_linear`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .w8a8 import W8A8Weight


@dataclass
class ViTW8A8Bundle:
    """Per-block ViT W8A8 weights. None entries -> keep FP16 for that block."""
    fc1: List[Optional[W8A8Weight]]
    layer_range: Tuple[int, int]


def build_vit_fc1_w8a8_bundle(
    model,
    layer_range: Tuple[int, int] = (3, 23),
    activation_stats: Optional[List[Optional[torch.Tensor]]] = None,
    alpha: float = 0.5,
) -> ViTW8A8Bundle:
    """Build W8A8 bundle for ViT linear_fc1 in [layer_range[0], layer_range[1]).

    Args:
        model: Qwen3-VL-2B HF model with `model.visual.blocks`.
        layer_range: half-open block index window to quantize. Default (3, 23):
                     skip blocks 0-2 (early features) and block 23 (final, fed
                     to merger / deepstack consumers).
        activation_stats: optional list aligned with blocks (length == n_blocks);
                     entry `[K]` = per-input-channel activation absmax from
                     `quant_optim.calibration.collect_vit_fc1_activation_stats`.
                     When provided for a block in `layer_range`, the bundle
                     uses `W8A8Weight.from_calibrated` (SmoothQuant α-fold)
                     for that block; otherwise falls back to plain `from_fp16`.
        alpha: SmoothQuant migration ratio (0=all weight, 1=all activation).

    Returns:
        ViTW8A8Bundle with one slot per block; None outside layer_range.
    """
    visual = model.model.visual if hasattr(model.model, "visual") else model.visual
    blocks = visual.blocks
    n_blocks = len(blocks)

    lo, hi = layer_range
    lo = max(0, lo)
    hi = min(n_blocks, hi)

    fc1: List[Optional[W8A8Weight]] = [None] * n_blocks
    for li in range(lo, hi):
        mlp = blocks[li].mlp
        fc1_w = mlp.linear_fc1.weight
        fc1_b = mlp.linear_fc1.bias
        if (activation_stats is not None
                and li < len(activation_stats)
                and activation_stats[li] is not None):
            fc1[li] = W8A8Weight.from_calibrated(
                fc1_w, activation_stats[li], bias=fc1_b, alpha=alpha)
        else:
            fc1[li] = W8A8Weight.from_fp16(fc1_w, bias=fc1_b)
    return ViTW8A8Bundle(fc1=fc1, layer_range=(lo, hi))
