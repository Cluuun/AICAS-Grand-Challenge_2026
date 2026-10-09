from __future__ import annotations

import types
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class _LinearSlice(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        bias: nn.Parameter | None,
        start: int,
        end: int,
        in_features: int,
    ) -> None:
        super().__init__()
        self.register_parameter("_weight", weight)
        if bias is None:
            self.register_parameter("_bias", None)
        else:
            self.register_parameter("_bias", bias)
        self.start = int(start)
        self.end = int(end)
        self.in_features = int(in_features)
        self.out_features = int(end - start)

    @property
    def weight(self) -> torch.Tensor:
        return self._weight[self.start:self.end]

    @property
    def bias(self) -> torch.Tensor | None:
        if self._bias is None:
            return None
        return self._bias[self.start:self.end]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def _make_fused_linear(parts: list[nn.Linear]) -> tuple[nn.Linear, list[_LinearSlice]]:
    if not parts:
        raise ValueError("empty fused linear parts")
    first = parts[0]
    in_features = int(first.in_features)
    device = first.weight.device
    dtype = first.weight.dtype
    use_bias = any(p.bias is not None for p in parts)
    out_features = sum(int(p.out_features) for p in parts)
    fused = nn.Linear(in_features, out_features, bias=use_bias, device=device, dtype=dtype)
    with torch.no_grad():
        offset = 0
        for part in parts:
            width = int(part.out_features)
            fused.weight[offset:offset + width].copy_(part.weight)
            if use_bias:
                if part.bias is None:
                    fused.bias[offset:offset + width].zero_()
                else:
                    fused.bias[offset:offset + width].copy_(part.bias)
            offset += width
    fused.weight.requires_grad_(False)
    if fused.bias is not None:
        fused.bias.requires_grad_(False)

    views: list[_LinearSlice] = []
    offset = 0
    for part in parts:
        width = int(part.out_features)
        views.append(_LinearSlice(fused.weight, fused.bias, offset, offset + width, in_features))
        offset += width
    return fused, views


class FusedGateUpMLP(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise TypeError("MLP must expose gate_proj/up_proj/down_proj")
        self.config = getattr(mlp, "config", None)
        self.act_fn = getattr(mlp, "act_fn")
        self.down_proj = mlp.down_proj
        self.gate_up_proj, views = _make_fused_linear([mlp.gate_proj, mlp.up_proj])
        self.gate_proj, self.up_proj = views
        if hasattr(mlp, "intermediate_size"):
            self.intermediate_size = int(getattr(mlp, "intermediate_size"))
        else:
            self.intermediate_size = int(self.gate_proj.out_features)
        self._aicas_fused_gate_up_mlp = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        gate = F.silu(gate, inplace=True)
        gate.mul_(up)
        return self.down_proj(gate)


def _fused_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    q_out = int(self.q_proj.out_features)
    k_out = int(self.k_proj.out_features)
    v_out = int(self.v_proj.out_features)

    qkv = self.qkv_proj(hidden_states)
    query_raw, key_raw, value_raw = qkv.split((q_out, k_out, v_out), dim=-1)

    query_states = self.q_norm(query_raw.view(*input_shape, -1, self.head_dim)).transpose(1, 2)
    key_states = self.k_norm(key_raw.view(*input_shape, -1, self.head_dim)).transpose(1, 2)
    value_states = value_raw.view(*input_shape, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _resolve_text_layers(model: nn.Module) -> list[nn.Module]:
    candidates = [
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "model", None),
    ]
    for candidate in candidates:
        layers = getattr(candidate, "layers", None) if candidate is not None else None
        if layers is not None:
            return list(layers)
    return []


def apply_text_projection_fusion(
    model: nn.Module,
    *,
    fuse_qkv: bool = True,
    fuse_mlp_gate_up: bool = True,
) -> Dict[str, int]:
    layers = _resolve_text_layers(model)
    qkv_count = 0
    mlp_count = 0

    for layer in layers:
        if fuse_qkv:
            attn = getattr(layer, "self_attn", None)
            if (
                attn is not None
                and not getattr(attn, "_aicas_fused_qkv", False)
                and not getattr(attn, "_aicas_decode_qk_rotary_ext_fastpath", False)
                and all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm"))
            ):
                qkv_proj, views = _make_fused_linear([attn.q_proj, attn.k_proj, attn.v_proj])
                attn.qkv_proj = qkv_proj
                attn.q_proj, attn.k_proj, attn.v_proj = views
                attn.forward = types.MethodType(_fused_attention_forward, attn)
                attn._aicas_fused_qkv = True
                qkv_count += 1

        if fuse_mlp_gate_up:
            mlp = getattr(layer, "mlp", None)
            if (
                mlp is not None
                and not getattr(mlp, "_aicas_fused_gate_up_mlp", False)
                and all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj", "act_fn"))
            ):
                layer.mlp = FusedGateUpMLP(mlp)
                mlp_count += 1

    return {"qkv": qkv_count, "mlp_gate_up": mlp_count}
