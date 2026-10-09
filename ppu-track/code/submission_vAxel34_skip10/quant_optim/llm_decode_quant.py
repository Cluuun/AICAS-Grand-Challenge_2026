"""LLM-decode W8A16 weight bundle for Qwen3-VL-2B.

Holds W8A16-quantized fused-QKV / o_proj / fused-gate_up / down_proj
weights for each of the 28 decoder layers. Designed to be wired into
the vAxel26-style decode forward (FA2 path), where M=1 is the dominant
shape and weight bandwidth is the bottleneck.

Prefill uses the original FP16 weights (W8A16 hurts at M≥64 on PPU).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .w8a16 import W8A16Weight


@dataclass
class W8A16AttnBundle:
    """Per-layer attention W8A16 weights (fused QKV + o_proj)."""
    qkv: W8A16Weight
    o:   W8A16Weight
    qkv_bias: Optional[torch.Tensor]   # FP16 bias kept dense
    q_dim: int
    k_dim: int
    v_dim: int


@dataclass
class W8A16MLPBundle:
    """Per-layer MLP W8A16 weights (fused gate_up + down)."""
    gate_up: W8A16Weight
    down:    W8A16Weight


@dataclass
class W8A16LLMBundle:
    """All W8A16 weight bundles for the LLM decoder + lm_head."""
    attns: List[W8A16AttnBundle]
    mlps:  List[W8A16MLPBundle]
    lm_head: Optional[W8A16Weight]


def build_llm_decode_w8a16_bundle(
    model,
    quantize_lm_head: bool = True,
    skip_layers: Optional[List[int]] = None,
) -> W8A16LLMBundle:
    """Walk the model, quantize each LLM decoder layer's fused weights to W8A16.

    Args:
        model: Qwen3-VL-2B HuggingFace model with `.model.language_model.layers`.
        quantize_lm_head: also quantize lm_head (M=1, big N=151936 -> 1.79× per bench).
        skip_layers: optional layer indices to leave as FP16 (sensitivity escape hatch).

    Returns:
        W8A16LLMBundle with one Attn+MLP entry per layer (None placeholders for
        skipped layers), plus optional lm_head.
    """
    skip = set(skip_layers or [])
    lm = model.model.language_model
    layers = lm.layers
    n_layers = len(layers)

    attns: List[W8A16AttnBundle] = [None] * n_layers
    mlps:  List[W8A16MLPBundle] = [None] * n_layers

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

        attns[li] = W8A16AttnBundle(
            qkv=W8A16Weight.from_fp16(qkv_fp),
            o=W8A16Weight.from_fp16(o_fp),
            qkv_bias=qkv_bias,
            q_dim=q_w.shape[0],
            k_dim=k_w.shape[0],
            v_dim=v_w.shape[0],
        )

        mlp = layer.mlp
        gate_up_fp = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0)
        down_fp = mlp.down_proj.weight
        mlps[li] = W8A16MLPBundle(
            gate_up=W8A16Weight.from_fp16(gate_up_fp),
            down=W8A16Weight.from_fp16(down_fp),
        )

    lm_head_w8 = None
    if quantize_lm_head:
        lm_head_w8 = W8A16Weight.from_fp16(model.lm_head.weight)

    return W8A16LLMBundle(attns=attns, mlps=mlps, lm_head=lm_head_w8)
