"""LLM-prefill W8A8 weight bundle for Qwen3-VL-2B (PPU SmoothQuant kernel).

Mirrors `llm_decode_quant.W8A16LLMBundle` but for prefill (M~128-384) where
W8A16's bandwidth advantage flips negative and `smoothquant_gemm` is +1.4-1.6×.

Mode 1 (default, no calibration): W8A8Weight.from_fp16 — symmetric per-row
weight quant, dynamic per-token activation quant. Same numerical behavior as
W8A16 from_fp16 (no SmoothQuant α-migration).

Mode 2 (with calibration): W8A8Weight.from_calibrated using per-input-channel
activation absmax — SmoothQuant α=0.5. Use when Mode 1 hurts P.

Both bundles are wired into the wrapper's compiled prefill path; the W8A16
decode bundle stays untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .w8a8 import W8A8Weight


@dataclass
class W8A8AttnPrefillBundle:
    """Per-layer attention W8A8 weights (fused QKV + o_proj) for prefill."""
    qkv: W8A8Weight
    o:   W8A8Weight
    qkv_bias: Optional[torch.Tensor]   # FP16 bias kept dense
    q_dim: int
    k_dim: int
    v_dim: int


@dataclass
class W8A8MLPPrefillBundle:
    """Per-layer MLP W8A8 weights (fused gate_up + down) for prefill."""
    gate_up: W8A8Weight
    down:    W8A8Weight


@dataclass
class W8A8LLMPrefillBundle:
    """All W8A8 weight bundles for the LLM decoder prefill path."""
    attns: List[Optional[W8A8AttnPrefillBundle]]
    mlps:  List[Optional[W8A8MLPPrefillBundle]]


def build_llm_prefill_w8a8_bundle(
    model,
    skip_layers: Optional[List[int]] = None,
    activation_stats: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
    alpha: float = 0.5,
) -> W8A8LLMPrefillBundle:
    """Walk the model, quantize each LLM decoder layer's fused weights to W8A8.

    Args:
        model: Qwen3-VL-2B HF model with `.model.language_model.layers`.
        skip_layers: indices to leave as FP16 (sensitivity escape hatch).
        activation_stats: optional dict[li] -> dict with keys
            "qkv_in" [K_qkv], "o_in" [K_o], "gate_up_in" [K_gu], "down_in" [K_d]
            (per-input-channel absmax). When provided, SmoothQuant α-migration
            is applied per linear; otherwise from_fp16.
        alpha: SmoothQuant α (0.5 = balanced).

    Returns:
        W8A8LLMPrefillBundle with one Attn+MLP entry per layer.
    """
    skip = set(skip_layers or [])
    lm = model.model.language_model
    layers = lm.layers
    n_layers = len(layers)

    attns: List[Optional[W8A8AttnPrefillBundle]] = [None] * n_layers
    mlps:  List[Optional[W8A8MLPPrefillBundle]] = [None] * n_layers

    for li, layer in enumerate(layers):
        if li in skip:
            continue
        attn = layer.self_attn
        q_w = attn.q_proj.weight
        k_w = attn.k_proj.weight
        v_w = attn.v_proj.weight
        qkv_fp = torch.cat([q_w, k_w, v_w], dim=0)
        o_fp = attn.o_proj.weight

        if attn.q_proj.bias is not None:
            qkv_bias = torch.cat([
                attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias
            ], dim=0).contiguous()
        else:
            qkv_bias = None

        stats = activation_stats.get(li) if activation_stats is not None else None
        if stats is not None and "qkv_in" in stats:
            qkv_q = W8A8Weight.from_calibrated(qkv_fp, stats["qkv_in"], alpha=alpha)
        else:
            qkv_q = W8A8Weight.from_fp16(qkv_fp)
        if stats is not None and "o_in" in stats:
            o_q = W8A8Weight.from_calibrated(o_fp, stats["o_in"], alpha=alpha)
        else:
            o_q = W8A8Weight.from_fp16(o_fp)

        attns[li] = W8A8AttnPrefillBundle(
            qkv=qkv_q,
            o=o_q,
            qkv_bias=qkv_bias,
            q_dim=q_w.shape[0],
            k_dim=k_w.shape[0],
            v_dim=v_w.shape[0],
        )

        mlp = layer.mlp
        gate_up_fp = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
        down_fp = mlp.down_proj.weight
        if stats is not None and "gate_up_in" in stats:
            gu_q = W8A8Weight.from_calibrated(gate_up_fp, stats["gate_up_in"], alpha=alpha)
        else:
            gu_q = W8A8Weight.from_fp16(gate_up_fp)
        if stats is not None and "down_in" in stats:
            d_q = W8A8Weight.from_calibrated(down_fp, stats["down_in"], alpha=alpha)
        else:
            d_q = W8A8Weight.from_fp16(down_fp)

        mlps[li] = W8A8MLPPrefillBundle(gate_up=gu_q, down=d_q)

    return W8A8LLMPrefillBundle(attns=attns, mlps=mlps)
