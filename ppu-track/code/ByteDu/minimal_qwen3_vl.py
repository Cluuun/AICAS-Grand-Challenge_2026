from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from safetensors.torch import load_file
from transformers import Qwen3VLConfig

try:
    from triton_act_quant import (
        int8_linear, ln_int8_linear, gelu_int8_linear, quantize_weight_per_channel_i8,
    )
except Exception:  # triton unavailable -> int8 vision path stays disabled
    int8_linear = ln_int8_linear = gelu_int8_linear = None
    quantize_weight_per_channel_i8 = None


AttnBackend = Literal["eager", "flash_attn_2"]


@dataclass(frozen=True)
class ShapeSignature:
    grid_t: int
    grid_h: int
    grid_w: int
    text_seq_len: int
    hidden_dtype: torch.dtype
    device: torch.device

    @classmethod
    def from_inputs(
        cls,
        image_grid_thw: torch.Tensor,
        input_ids: torch.Tensor,
        hidden_dtype: torch.dtype,
        device: torch.device,
    ) -> "ShapeSignature":
        if image_grid_thw.ndim != 2 or image_grid_thw.shape != (1, 3):
            raise ValueError(f"Expected `image_grid_thw` with shape (1, 3), got {tuple(image_grid_thw.shape)}")

        return cls(
            grid_t=int(image_grid_thw[0, 0].item()),
            grid_h=int(image_grid_thw[0, 1].item()),
            grid_w=int(image_grid_thw[0, 2].item()),
            text_seq_len=int(input_ids.shape[-1]),
            hidden_dtype=hidden_dtype,
            device=device,
        )

class RuntimeCache:
    def __init__(self) -> None:
        self.vision_abs_pos: dict[tuple, torch.Tensor] = {}
        self.vision_rotary: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.text_rotary: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.causal_masks: dict[tuple, torch.Tensor] = {}
        self.single_image_metadata: dict[tuple, object] = {}

    def clear(self) -> None:
        self.vision_abs_pos.clear()
        self.vision_rotary.clear()
        self.text_rotary.clear()
        self.causal_masks.clear()
        self.single_image_metadata.clear()


@dataclass
class MinimalVisionOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: tuple[torch.Tensor, ...]
    deepstack_start_layer: int


@dataclass
class MinimalBackboneOutput:
    last_hidden_state: torch.Tensor
    rope_deltas: torch.Tensor | None
    position_ids: torch.Tensor
    shape_signature: ShapeSignature


@dataclass
class MinimalCausalLMOutput:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor
    rope_deltas: torch.Tensor | None
    position_ids: torch.Tensor
    shape_signature: ShapeSignature


@dataclass(frozen=True)
class VisualTokenSpan:
    start: int
    count: int


@dataclass(frozen=True)
class PreparedSingleImageMetadata:
    shape_signature: ShapeSignature
    visual_token_span: VisualTokenSpan
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    rope_delta: int
    last_token_index: int
    text_rotary_cache_key: tuple[int, int, int, int, int, int, torch.dtype, torch.device]

    def rotary_cache_key(self, hidden_dtype: torch.dtype) -> tuple[int, int, int, int, int, int, torch.dtype, torch.device]:
        if hidden_dtype == self.shape_signature.hidden_dtype:
            return self.text_rotary_cache_key
        signature = self.shape_signature
        return (
            signature.grid_t,
            signature.grid_h,
            signature.grid_w,
            signature.text_seq_len,
            self.visual_token_span.start,
            self.visual_token_span.count,
            hidden_dtype,
            signature.device,
        )


def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def dequantize_pack_quantized_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if weight_packed.dtype != torch.int32:
        raise TypeError(f"Expected int32 packed weights, got {weight_packed.dtype}.")
    if weight_shape.numel() != 2:
        raise ValueError(f"Expected weight_shape with 2 elements, got {tuple(weight_shape.shape)}.")

    out_features = int(weight_shape[0].item())
    in_features = int(weight_shape[1].item())
    packed_cols = (in_features + 3) // 4
    if weight_packed.shape != (out_features, packed_cols):
        raise ValueError(
            f"Packed weight shape mismatch: expected {(out_features, packed_cols)}, got {tuple(weight_packed.shape)}."
        )
    if weight_scale.shape[0] != out_features:
        raise ValueError(
            f"Scale row count mismatch: expected {out_features}, got {tuple(weight_scale.shape)}."
        )

    shifts = torch.tensor([0, 8, 16, 24], device=weight_packed.device, dtype=torch.int32)
    unpacked = torch.bitwise_right_shift(weight_packed.unsqueeze(-1), shifts)
    unpacked = torch.bitwise_and(unpacked, 0xFF).to(torch.int16) - 128
    unpacked = unpacked.reshape(out_features, packed_cols * 4)[:, :in_features]
    dequantized = unpacked.to(torch.float32) * weight_scale.reshape(out_features, -1).to(torch.float32)
    return dequantized.to(dtype=dtype or weight_scale.dtype)


def dequantize_int8_scaled_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if weight.dtype != torch.int8:
        raise TypeError(f"Expected int8 quantized weights, got {weight.dtype}.")
    if weight_scale.shape[0] != weight.shape[0]:
        raise ValueError(
            f"Scale row count mismatch: expected {weight.shape[0]}, got {tuple(weight_scale.shape)}."
        )
    dequantized = weight.to(torch.float32) * weight_scale.reshape(weight.shape[0], -1).to(torch.float32)
    return dequantized.to(dtype=dtype or weight_scale.dtype)


def quantize_int8_weight_from_scale(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if weight_scale.shape[0] != weight.shape[0]:
        raise ValueError(
            f"Scale row count mismatch: expected {weight.shape[0]}, got {tuple(weight_scale.shape)}."
        )
    scale_view = weight_scale.reshape(weight.shape[0], -1).to(torch.float32)
    if scale_view.shape[1] not in (1, weight.shape[1]):
        raise ValueError(
            f"Unsupported scale shape for int8 quantization: weight={tuple(weight.shape)}, scale={tuple(weight_scale.shape)}."
        )
    quantized = torch.round(weight.to(torch.float32) / scale_view).clamp_(-128.0, 127.0).to(torch.int8)
    return quantized.contiguous()


def pack_int8_weight_to_int32_q8_words(weight_int8: torch.Tensor) -> torch.Tensor:
    if weight_int8.dtype != torch.int8:
        raise TypeError(f"Expected int8 quantized weights, got {weight_int8.dtype}.")
    quantized = weight_int8.to(torch.int32) + 128
    pad = (-quantized.shape[1]) % 4
    if pad:
        quantized = torch.nn.functional.pad(quantized, (0, pad), value=128)
    quantized = quantized.view(quantized.shape[0], -1, 4)
    return (
        quantized[:, :, 0]
        | (quantized[:, :, 1] << 8)
        | (quantized[:, :, 2] << 16)
        | (quantized[:, :, 3] << 24)
    ).contiguous()


def _attach_quantized_linear_buffers(model: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    for key, packed_weight in tensors.items():
        if not key.endswith(".weight_packed"):
            continue
        module_name = key[: -len(".weight_packed")]
        module = modules.get(module_name)
        if module is None or not isinstance(module, nn.Linear):
            continue
        if hasattr(module, "weight_packed"):
            continue
        scale_key = f"{module_name}.weight_scale"
        shape_key = f"{module_name}.weight_shape"
        device = module.weight.device
        module.register_buffer("weight_packed", packed_weight.to(device=device).contiguous(), persistent=False)
        module.register_buffer(
            "weight_scale",
            tensors[scale_key].to(device=device, dtype=module.weight.dtype).contiguous(),
            persistent=False,
        )
        module.register_buffer("weight_shape", tensors[shape_key].to(device=device).contiguous(), persistent=False)
    for key, weight in tensors.items():
        if not key.endswith(".weight") or weight.dtype != torch.int8:
            continue
        module_name = key[: -len(".weight")]
        module = modules.get(module_name)
        if module is None or not isinstance(module, nn.Linear):
            continue
        if hasattr(module, "weight_int8"):
            continue
        scale_key = f"{module_name}.weight_scale"
        if scale_key not in tensors:
            continue
        device = module.weight.device
        module.register_buffer("weight_int8", weight.to(device=device).contiguous(), persistent=False)
        module.register_buffer(
            "weight_scale",
            tensors[scale_key].to(device=device, dtype=module.weight.dtype).contiguous(),
            persistent=False,
        )


def attach_quantized_linear_codebook(model: nn.Module, tensors: dict[str, torch.Tensor]) -> int:
    modules = dict(model.named_modules())
    attached = 0
    attach_int8 = (
        (
            os.getenv("JUNKRAT_ENABLE_W8A8_DECODE", "0") == "1"
            or os.getenv("JUNKRAT_ENABLE_W8A8_LM_HEAD", "0") == "1"
        )
        and os.getenv("JUNKRAT_DISABLE_CODEBOOK_INT8_BUFFERS", "0") != "1"
    )
    for key, scale in tensors.items():
        if not key.endswith(".weight_scale"):
            continue
        module_name = key[: -len(".weight_scale")]
        module = modules.get(module_name)
        if module is None or not isinstance(module, nn.Linear):
            continue
        if hasattr(module, "weight_int8") or hasattr(module, "weight_packed"):
            continue
        scale = scale.to(device=module.weight.device, dtype=module.weight.dtype).contiguous()
        weight_int8 = quantize_int8_weight_from_scale(module.weight.detach(), scale)
        module.register_buffer(
            "weight_packed",
            pack_int8_weight_to_int32_q8_words(weight_int8),
            persistent=False,
        )
        if attach_int8:
            module.register_buffer("weight_int8", weight_int8.contiguous(), persistent=False)
        module.register_buffer(
            "weight_shape",
            torch.tensor(module.weight.shape, device=module.weight.device, dtype=torch.int32),
            persistent=False,
        )
        module.register_buffer("weight_scale", scale, persistent=False)
        attached += 1
    return attached


def resolve_text_rope_config(config) -> tuple[float, list[int]]:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        return rope_parameters["rope_theta"], rope_parameters.get("mrope_section", [24, 20, 20])

    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        raise AttributeError("Text config must provide `rope_parameters[\"rope_theta\"]` or `rope_theta`.")

    rope_scaling = getattr(config, "rope_scaling", None)
    if hasattr(rope_scaling, "to_dict"):
        rope_scaling = rope_scaling.to_dict()
    if rope_scaling is None:
        rope_scaling = {}

    return rope_theta, rope_scaling.get("mrope_section", [24, 20, 20])


def apply_rotary_pos_emb_text(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (query_states * cos) + (rotate_half(query_states) * sin), (key_states * cos) + (
        rotate_half(key_states) * sin
    )


_VISION_ROPE_FUSED = os.getenv("JUNKRAT_DISABLE_VISION_ROPE_FUSED", "0") != "1"
try:
    from triton_vision_rope import apply_rotary_pos_emb_vision_fused as _vision_rope_fused
except Exception:
    _vision_rope_fused = None


def apply_rotary_pos_emb_vision(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Fused single-kernel rope: fp32 compute / bf16 store, bit-identical to the
    # reference below (verified maxdiff=0), ~4x faster than the elementwise path.
    if (
        _VISION_ROPE_FUSED
        and _vision_rope_fused is not None
        and query_states.is_cuda
        and query_states.dim() == 4
        and query_states.shape[0] == 1
        and query_states.shape[-1] % 2 == 0
    ):
        return _vision_rope_fused(query_states, key_states, cos, sin)
    orig_q_dtype = query_states.dtype
    orig_k_dtype = key_states.dtype
    q = query_states.float()
    k = key_states.float()
    cos = cos.unsqueeze(0).unsqueeze(2).float()
    sin = sin.unsqueeze(0).unsqueeze(2).float()
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q.to(orig_q_dtype), k.to(orig_k_dtype)


def eager_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
) -> torch.Tensor:
    query_states = query_states.transpose(1, 2)
    key_states = repeat_kv(key_states.transpose(1, 2), num_key_value_groups)
    value_states = repeat_kv(value_states.transpose(1, 2), num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous()


def attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int,
    causal: bool,
    backend: AttnBackend,
    flash_compatible: bool,
) -> torch.Tensor:
    if backend == "flash_attn_2" and flash_compatible and query_states.device.type == "cuda":
        return flash_attn_func(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            dropout_p=0.0,
            softmax_scale=scaling,
            causal=causal,
        )

    return eager_attention(query_states, key_states, value_states, attention_mask, scaling, num_key_value_groups)


class MinimalRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, use_fast_path: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_fast_path = use_fast_path

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_fast_path:
            return F.rms_norm(hidden_states, (self.weight.shape[0],), self.weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MinimalVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class MinimalVisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self._int8 = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(_gelu_pytorch_tanh(self.linear_fc1(hidden_states)))

    def forward_int8_prenorm(self, hidden_states: torch.Tensor, norm) -> torch.Tensor:
        # A8W8 (fc1 1.66x, fc2 1.65x). norm2 is fused into fc1's int8 quant; the gelu
        # is fused into fc2's int8 quant — both ride on the GEMM's input memory traffic.
        hidden = ln_int8_linear(
            hidden_states, norm.weight, norm.bias, norm.eps,
            self.fc1_wq, self.fc1_ws, self.linear_fc1.bias,
        )
        return gelu_int8_linear(hidden, self.fc2_wq, self.fc2_ws, self.linear_fc2.bias)


class MinimalVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("inv_freq", self._build_inv_freq(device=None), persistent=False)

    def _build_inv_freq(self, device: torch.device | None) -> torch.Tensor:
        return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim))

    def _apply(self, fn):
        super()._apply(fn)
        self.inv_freq = self._build_inv_freq(device=self.inv_freq.device)
        return self

    def forward(self, seq_len: int) -> torch.Tensor:
        seq = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class MinimalVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm: bool) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = self.norm(hidden_states.view(-1, self.hidden_size))
        else:
            hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)
        hidden_states = self.act(self.linear_fc1(hidden_states))
        return self.linear_fc2(hidden_states)


class MinimalVisionAttention(nn.Module):
    def __init__(self, config, attn_backend: AttnBackend) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        self.scaling = self.head_dim**-0.5
        self.attn_backend = attn_backend
        self._int8 = False

    def _attn_from_qkv(self, qkv: torch.Tensor, position_embeddings) -> torch.Tensor:
        seq_len = qkv.shape[0]
        query_states, key_states, value_states = qkv.reshape(
            seq_len, 3, self.num_heads, self.head_dim
        ).unbind(dim=1)
        query_states = query_states.unsqueeze(0)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, *position_embeddings)
        attn_output = attention_forward(
            query_states, key_states, value_states,
            attention_mask=None, scaling=self.scaling, num_key_value_groups=1,
            causal=False, backend=self.attn_backend, flash_compatible=True,
        )
        return self.proj(attn_output.reshape(seq_len, self.dim))  # proj stays bf16 (int8 0.58x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self._attn_from_qkv(self.qkv(hidden_states), position_embeddings)

    def forward_int8_prenorm(self, hidden_states, norm, position_embeddings):
        # qkv is an int8 winner (1.39x); norm1 is fused into qkv's int8 quant.
        qkv = ln_int8_linear(
            hidden_states, norm.weight, norm.bias, norm.eps,
            self.qkv_wq, self.qkv_ws, self.qkv.bias,
        )
        return self._attn_from_qkv(qkv, position_embeddings)


class MinimalVisionBlock(nn.Module):
    def __init__(self, config, attn_backend: AttnBackend) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = MinimalVisionAttention(config, attn_backend=attn_backend)
        self.mlp = MinimalVisionMLP(config)
        self._int8 = False

    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if self._int8:
            # Fused norm->int8-quant->int8 GEMM for qkv/fc1/fc2 (the int8 winners);
            # proj stays bf16. norm1/norm2/gelu ride on the GEMM input traffic (no
            # standalone norm/quant kernels).
            hidden_states = hidden_states + self.attn.forward_int8_prenorm(
                hidden_states, self.norm1, position_embeddings
            )
            hidden_states = hidden_states + self.mlp.forward_int8_prenorm(hidden_states, self.norm2)
            return hidden_states
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), position_embeddings=position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class MinimalVisionModel(nn.Module):
    def __init__(self, config, runtime_cache: RuntimeCache, attn_backend: AttnBackend) -> None:
        super().__init__()
        self.config = config
        self.runtime_cache = runtime_cache
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = MinimalVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = MinimalVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([MinimalVisionBlock(config, attn_backend=attn_backend) for _ in range(config.depth)])
        self.merger = MinimalVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self._deepstack_merger_by_layer = {
            layer_idx: merger_idx for merger_idx, layer_idx in enumerate(self.deepstack_visual_indexes)
        }
        self.deepstack_merger_list = nn.ModuleList(
            [MinimalVisionPatchMerger(config, use_postshuffle_norm=True) for _ in self.deepstack_visual_indexes]
        )

    def _vision_cache_key(self, signature: ShapeSignature) -> tuple[int, int, int, torch.dtype, torch.device]:
        return (
            signature.grid_t,
            signature.grid_h,
            signature.grid_w,
            signature.hidden_dtype,
            signature.device,
        )

    def fast_pos_embed_interpolate(self, signature: ShapeSignature, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        cache_key = self._vision_cache_key(signature)
        cached = self.runtime_cache.vision_abs_pos.get(cache_key)
        if cached is not None:
            return cached

        grid_t = signature.grid_t
        grid_h = signature.grid_h
        grid_w = signature.grid_w
        merge_size = self.config.spatial_merge_size

        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, grid_h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, grid_w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs_floor + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs_floor + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        idx_tensor = torch.stack(
            [
                (base_h[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h[:, None] + w_idxs_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_idxs_ceil[None, :]).flatten(),
            ]
        ).to(device=device, dtype=torch.long)
        weight_tensor = torch.stack(
            [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]
        ).to(device=device, dtype=self.pos_embed.weight.dtype)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = (pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]).repeat(grid_t, 1)
        result = (
            patch_pos_embeds.view(grid_t, grid_h // merge_size, merge_size, grid_w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
            .to(dtype=dtype)
        )
        self.runtime_cache.vision_abs_pos[cache_key] = result
        return result

    def rot_pos_emb(self, signature: ShapeSignature, device: torch.device) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_t = signature.grid_t
        grid_h = signature.grid_h
        grid_w = signature.grid_w
        max_hw = max(grid_h, grid_w)
        freq_table = self.rotary_pos_emb(max_hw).to(device)

        merged_h = grid_h // merge_size
        merged_w = grid_w // merge_size
        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        if grid_t > 1:
            coords = coords.repeat(grid_t, 1)
        return freq_table[coords].flatten(1)

    def get_position_embeddings(
        self,
        signature: ShapeSignature,
        hidden_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = self._vision_cache_key(signature)
        cached = self.runtime_cache.vision_rotary.get(cache_key)
        if cached is not None:
            return cached

        rotary_pos_emb = self.rot_pos_emb(signature, device=device)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        result = (emb.cos(), emb.sin())
        self.runtime_cache.vision_rotary[cache_key] = result
        return result

    def materialize_runtime_cache(
        self,
        signature: ShapeSignature,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.fast_pos_embed_interpolate(signature, dtype=dtype, device=device)
        self.get_position_embeddings(signature, hidden_dtype=dtype, device=device)

    def vision_prefill_core(
        self,
        pixel_values: torch.Tensor,
        pos_embeds: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + pos_embeds
        position_embeddings = (position_cos, position_sin)

        deepstack_features: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, position_embeddings=position_embeddings)
            merger_idx = self._deepstack_merger_by_layer.get(layer_idx)
            if merger_idx is not None:
                deepstack_features.append(self.deepstack_merger_list[merger_idx](hidden_states))

        pooled = self.merger(hidden_states)
        return hidden_states, pooled, tuple(deepstack_features)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        shape_signature: ShapeSignature,
    ) -> MinimalVisionOutput:
        if image_grid_thw.ndim != 2 or image_grid_thw.shape != (1, 3):
            raise ValueError(f"Expected a single image grid with shape (1, 3), got {tuple(image_grid_thw.shape)}")

        pos_embeds = self.fast_pos_embed_interpolate(
            shape_signature,
            dtype=self.patch_embed.proj.weight.dtype,
            device=pixel_values.device,
        )
        position_embeddings = self.get_position_embeddings(
            shape_signature, hidden_dtype=pos_embeds.dtype, device=pos_embeds.device
        )
        hidden_states, pooled, deepstack_features = self.vision_prefill_core(
            pixel_values,
            pos_embeds,
            position_embeddings[0],
            position_embeddings[1],
        )
        return MinimalVisionOutput(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            deepstack_features=deepstack_features,
            deepstack_start_layer=0,
        )


def enable_vision_int8(visual, *, mlp: bool = True, qkv: bool = True) -> int:
    """Quantize vision block weights to per-channel int8 for the A8W8 prefill path.
    Targets the int8 *winners* at clamp-off M: MLP fc1/fc2 (~1.66x/1.65x) and attn
    qkv (~1.39x). proj (0.58x) and merger stay bf16. Returns blocks converted."""
    if ln_int8_linear is None or quantize_weight_per_channel_i8 is None:
        return 0
    n = 0
    for block in visual.blocks:
        if mlp:
            m = block.mlp
            wq, ws = quantize_weight_per_channel_i8(m.linear_fc1.weight)
            m.register_buffer("fc1_wq", wq, persistent=False)
            m.register_buffer("fc1_ws", ws, persistent=False)
            wq, ws = quantize_weight_per_channel_i8(m.linear_fc2.weight)
            m.register_buffer("fc2_wq", wq, persistent=False)
            m.register_buffer("fc2_ws", ws, persistent=False)
            m._int8 = True
        if qkv:
            a = block.attn
            wq, ws = quantize_weight_per_channel_i8(a.qkv.weight)
            a.register_buffer("qkv_wq", wq, persistent=False)
            a.register_buffer("qkv_ws", ws, persistent=False)
            a._int8 = True
        # block drives the fused-norm int8 path; require both targets for the
        # current implementation (the block forward fuses norm1->qkv and norm2->fc1/fc2).
        block._int8 = bool(mlp and qkv)
        n += 1
    return n


class MinimalTextRotaryEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.base, self.mrope_section = resolve_text_rope_config(config)
        self.dim = config.head_dim
        self.register_buffer("inv_freq", self._build_inv_freq(device=None), persistent=False)
        self.attention_scaling = 1.0

    def _build_inv_freq(self, device: torch.device | None) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim))

    def _apply(self, fn):
        super()._apply(fn)
        self.inv_freq = self._build_inv_freq(device=self.inv_freq.device)
        return self

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            freqs_t[..., slice(offset, length, 3)] = freqs[dim, ..., slice(offset, length, 3)]
        return freqs_t

    def compute(self, position_ids: torch.Tensor, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.to(device) @ position_ids_expanded.to(device)).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


class MinimalTextMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MinimalTextAttention(nn.Module):
    def __init__(self, config, layer_idx: int, attn_backend: AttnBackend) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attn_backend = attn_backend
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        use_fast_rms_norm = attn_backend == "flash_attn_2"
        self.q_norm = MinimalRMSNorm(self.head_dim, eps=config.rms_norm_eps, use_fast_path=use_fast_rms_norm)
        self.k_norm = MinimalRMSNorm(self.head_dim, eps=config.rms_norm_eps, use_fast_path=use_fast_rms_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        flash_compatible: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_norm(self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim))
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        query_states, key_states = apply_rotary_pos_emb_text(query_states, key_states, *position_embeddings)
        attn_output = attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            num_key_value_groups=self.num_key_value_groups,
            causal=True,
            backend=self.attn_backend,
            flash_compatible=flash_compatible,
        )
        return self.o_proj(attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim))


class MinimalTextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, attn_backend: AttnBackend) -> None:
        super().__init__()
        use_fast_rms_norm = attn_backend == "flash_attn_2"
        self.self_attn = MinimalTextAttention(config, layer_idx=layer_idx, attn_backend=attn_backend)
        self.mlp = MinimalTextMLP(config)
        self.input_layernorm = MinimalRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, use_fast_path=use_fast_rms_norm
        )
        self.post_attention_layernorm = MinimalRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, use_fast_path=use_fast_rms_norm
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        flash_compatible: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            flash_compatible=flash_compatible,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MinimalTextModel(nn.Module):
    def __init__(self, config, runtime_cache: RuntimeCache, attn_backend: AttnBackend) -> None:
        super().__init__()
        self.config = config
        self.runtime_cache = runtime_cache
        self.attn_backend = attn_backend
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [MinimalTextDecoderLayer(config, layer_idx=idx, attn_backend=attn_backend) for idx in range(config.num_hidden_layers)]
        )
        self.norm = MinimalRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, use_fast_path=attn_backend == "flash_attn_2"
        )
        self.rotary_emb = MinimalTextRotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def get_causal_mask(self, seq_len: int, hidden_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        cache_key = (seq_len, hidden_dtype, device)
        cached = self.runtime_cache.causal_masks.get(cache_key)
        if cached is None:
            min_value = torch.finfo(hidden_dtype).min
            cached = torch.full((1, 1, seq_len, seq_len), min_value, dtype=hidden_dtype, device=device)
            cached = torch.triu(cached, diagonal=1)
            self.runtime_cache.causal_masks[cache_key] = cached
        return cached

    def get_padded_causal_mask(
        self,
        attention_mask: torch.Tensor,
        hidden_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        padded_mask = self.get_causal_mask(attention_mask.shape[-1], hidden_dtype=hidden_dtype, device=device).clone()
        min_value = torch.finfo(hidden_dtype).min
        return padded_mask.masked_fill(attention_mask[:, None, None, :] == 0, min_value)

    def get_position_embeddings(
        self,
        position_ids: torch.Tensor,
        hidden_dtype: torch.dtype,
        device: torch.device,
        signature: ShapeSignature,
        rotary_cache_key: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del signature
        if rotary_cache_key is not None:
            cached = self.runtime_cache.text_rotary.get(rotary_cache_key)
            if cached is not None:
                return cached

        position_embeddings = self.rotary_emb.compute(position_ids, dtype=hidden_dtype, device=device)
        if rotary_cache_key is not None:
            self.runtime_cache.text_rotary[rotary_cache_key] = position_embeddings
        return position_embeddings

    def _inject_deepstack(
        self,
        hidden_states: torch.Tensor,
        visual_pos_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_token_span: VisualTokenSpan | None = None,
    ) -> torch.Tensor:
        if visual_token_span is not None:
            hidden_states.narrow(1, visual_token_span.start, visual_token_span.count).add_(visual_embeds.unsqueeze(0))
            return hidden_states

        hidden_states = hidden_states.clone()
        hidden_states[:, visual_pos_mask, :] = hidden_states[:, visual_pos_mask, :] + visual_embeds.unsqueeze(0)
        return hidden_states

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        visual_pos_mask: torch.Tensor | None = None,
        deepstack_visual_embeds: tuple[torch.Tensor, ...] | None = None,
        shape_signature: ShapeSignature,
        use_padding_mask: bool = False,
        visual_token_span: VisualTokenSpan | None = None,
        rotary_cache_key: tuple | None = None,
        deepstack_start_layer: int = 0,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_len, device=inputs_embeds.device, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        flash_compatible = self.attn_backend == "flash_attn_2" and not use_padding_mask
        if flash_compatible:
            causal_mask = None
        elif use_padding_mask:
            if attention_mask is None:
                raise ValueError("`attention_mask` is required when `use_padding_mask=True`.")
            causal_mask = self.get_padded_causal_mask(
                attention_mask, hidden_dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )
        else:
            causal_mask = self.get_causal_mask(
                inputs_embeds.shape[1], hidden_dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )
        position_embeddings = self.get_position_embeddings(
            position_ids,
            hidden_dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
            signature=shape_signature,
            rotary_cache_key=rotary_cache_key,
        )

        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                flash_compatible=flash_compatible,
            )
            ds_index = layer_idx - deepstack_start_layer
            if (
                deepstack_visual_embeds is not None
                and 0 <= ds_index < len(deepstack_visual_embeds)
                and (visual_pos_mask is not None or visual_token_span is not None)
            ):
                hidden_states = self._inject_deepstack(
                    hidden_states,
                    visual_pos_mask,
                    deepstack_visual_embeds[ds_index],
                    visual_token_span=visual_token_span,
                )

        return self.norm(hidden_states)


class MinimalQwen3VLBackbone(nn.Module):
    def __init__(
        self,
        config,
        runtime_cache: RuntimeCache,
        attn_backend: AttnBackend,
    ) -> None:
        super().__init__()
        self.config = config
        self.runtime_cache = runtime_cache
        self.llm_spatial_merge_size = int(getattr(config.vision_config, "spatial_merge_size", 2))
        self.llm_visual_pooling = os.getenv("JUNKRAT_LLM_VISUAL_POOLING", "mean").strip().lower()
        if self.llm_visual_pooling not in {"mean", "max"}:
            raise ValueError(f"Unsupported JUNKRAT_LLM_VISUAL_POOLING={self.llm_visual_pooling!r}; expected 'mean' or 'max'.")
        self.visual = MinimalVisionModel(config.vision_config, runtime_cache=runtime_cache, attn_backend=attn_backend)
        self.language_model = MinimalTextModel(
            config.text_config, runtime_cache=runtime_cache, attn_backend=attn_backend
        )
        self.rope_deltas: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def set_llm_spatial_merge_size(self, merge_size: int) -> None:
        merge_size = int(merge_size)
        if merge_size <= 0:
            raise ValueError(f"llm spatial merge size must be positive, got {merge_size}")
        visual_merge = int(self.config.vision_config.spatial_merge_size)
        if merge_size < visual_merge or merge_size % visual_merge != 0:
            raise ValueError(
                f"llm spatial merge size {merge_size} must be a multiple of visual merge size {visual_merge}"
            )
        self.llm_spatial_merge_size = merge_size

    def _llm_merge_size(self) -> int:
        return int(getattr(self, "llm_spatial_merge_size", self.config.vision_config.spatial_merge_size))

    def _pool_visual_features_to_llm_merge(
        self,
        features: torch.Tensor,
        *,
        shape_signature: ShapeSignature,
    ) -> torch.Tensor:
        visual_merge = int(self.config.vision_config.spatial_merge_size)
        llm_merge = self._llm_merge_size()
        if llm_merge == visual_merge:
            return features
        if llm_merge % visual_merge != 0:
            raise ValueError(f"llm merge size {llm_merge} must be divisible by visual merge size {visual_merge}")
        factor = llm_merge // visual_merge
        grid_t = shape_signature.grid_t
        visual_h = shape_signature.grid_h // visual_merge
        visual_w = shape_signature.grid_w // visual_merge
        if visual_h % factor != 0 or visual_w % factor != 0:
            raise ValueError(
                f"grid {shape_signature.grid_h}x{shape_signature.grid_w} cannot be pooled from "
                f"visual merge {visual_merge} to llm merge {llm_merge}"
            )
        expected = grid_t * visual_h * visual_w
        if features.shape[0] != expected:
            raise ValueError(f"Expected {expected} visual features before pooling, got {features.shape[0]}")
        hidden = features.shape[-1]
        grouped = features.view(grid_t, visual_h // factor, factor, visual_w // factor, factor, hidden)
        if self.llm_visual_pooling == "max":
            grouped = grouped.amax(dim=(2, 4))
        else:
            grouped = grouped.mean(dim=(2, 4))
        return grouped.reshape(-1, hidden)

    def pool_visual_output_to_llm_merge(
        self,
        image_outputs: MinimalVisionOutput,
        *,
        shape_signature: ShapeSignature,
    ) -> MinimalVisionOutput:
        pooled = self._pool_visual_features_to_llm_merge(
            image_outputs.pooler_output,
            shape_signature=shape_signature,
        )
        deepstack_features = tuple(
            self._pool_visual_features_to_llm_merge(feature, shape_signature=shape_signature)
            for feature in image_outputs.deepstack_features
        )
        return MinimalVisionOutput(
            last_hidden_state=image_outputs.last_hidden_state,
            pooler_output=pooled,
            deepstack_features=deepstack_features,
            deepstack_start_layer=image_outputs.deepstack_start_layer,
        )

    def get_image_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {input_ids.shape[0]}")
        return input_ids[0] == self.config.image_token_id

    def get_image_token_span(self, image_mask: torch.Tensor, expected_tokens: int) -> VisualTokenSpan:
        return VisualTokenSpan(
            start=int(torch.argmax(image_mask.to(torch.int32)).item()),
            count=expected_tokens,
        )

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        llm_grid_t = grid_t
        merge_size = self._llm_merge_size()
        llm_grid_h = grid_h // merge_size
        llm_grid_w = grid_w // merge_size

        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def compute_single_image_position_ids(
        self,
        *,
        seq_len: int,
        image_token_span: VisualTokenSpan,
        shape_signature: ShapeSignature,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_start = image_token_span.start
        image_token_count = image_token_span.count
        grid_t = shape_signature.grid_t
        grid_h = shape_signature.grid_h
        grid_w = shape_signature.grid_w

        vision_position_ids = self.get_vision_position_ids(
            image_start,
            grid_t=grid_t,
            grid_h=grid_h,
            grid_w=grid_w,
            device=device,
        )
        llm_pos_ids_list = []
        if image_start > 0:
            llm_pos_ids_list.append(torch.arange(image_start, device=device).view(1, -1).expand(3, -1))
        llm_pos_ids_list.append(vision_position_ids)

        image_end = image_start + image_token_count
        suffix_len = seq_len - image_end
        if suffix_len > 0:
            suffix_start = image_start + max(grid_h, grid_w) // self._llm_merge_size()
            llm_pos_ids_list.append(torch.arange(suffix_len, device=device).view(1, -1).expand(3, -1) + suffix_start)

        position_ids = torch.cat(llm_pos_ids_list, dim=1).unsqueeze(1).to(dtype=dtype)
        rope_delta = (position_ids[:, 0].max() + 1 - seq_len).reshape(1, 1)
        self.rope_deltas = rope_delta
        return position_ids, rope_delta

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_token_span: VisualTokenSpan | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {input_ids.shape[0]}")
        if image_grid_thw.shape != (1, 3):
            raise ValueError(f"Expected `image_grid_thw` with shape (1, 3), got {tuple(image_grid_thw.shape)}")

        valid_mask = attention_mask[0].bool() if attention_mask is not None else None
        current_token_types = mm_token_type_ids[0]
        if valid_mask is not None:
            current_token_types = current_token_types[valid_mask]

        token_count = int(current_token_types.shape[0])
        if token_count == 0:
            raise ValueError("Received an empty token sequence after applying the attention mask.")

        device = input_ids.device
        unsupported_types = current_token_types[(current_token_types != 0) & (current_token_types != 1)]
        if unsupported_types.numel() > 0:
            raise ValueError(f"Unsupported token type for single-image inference: {int(unsupported_types[0].item())}")

        grid_t = int(image_grid_thw[0, 0].item())
        grid_h = int(image_grid_thw[0, 1].item())
        grid_w = int(image_grid_thw[0, 2].item())
        image_mask = current_token_types == 1

        if bool(image_mask.any()):
            if image_token_span is None:
                image_start = int(torch.argmax(image_mask.to(torch.int32)).item())
                image_end = token_count - int(torch.argmax(image_mask.flip(0).to(torch.int32)).item())
                image_token_count = image_end - image_start
                if int(image_mask.sum().item()) != image_token_count:
                    raise ValueError("Single-image inference expects the image token span to be contiguous.")
            else:
                image_start = image_token_span.start
                image_token_count = image_token_span.count
            llm_positions, rope_delta = self.compute_single_image_position_ids(
                seq_len=token_count,
                image_token_span=VisualTokenSpan(start=image_start, count=image_token_count),
                shape_signature=ShapeSignature(
                    grid_t=grid_t,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    text_seq_len=token_count,
                    hidden_dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
                dtype=input_ids.dtype,
                device=device,
            )
            llm_positions = llm_positions[:, 0]
        else:
            llm_positions = torch.arange(token_count, device=device).view(1, -1).expand(3, -1)
            rope_delta = (llm_positions.max() + 1 - token_count).reshape(1, 1)

        position_ids = torch.zeros(3, 1, input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
        if valid_mask is not None:
            position_ids[:, 0, valid_mask] = llm_positions
        else:
            position_ids[:, 0] = llm_positions
        self.rope_deltas = rope_delta
        return position_ids, rope_delta

    def prepare_single_image_metadata(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> PreparedSingleImageMetadata:
        shape_signature = ShapeSignature.from_inputs(
            image_grid_thw=image_grid_thw,
            input_ids=input_ids,
            hidden_dtype=self.language_model.embed_tokens.weight.dtype,
            device=input_ids.device,
        )
        expected_tokens = shape_signature.grid_t * shape_signature.grid_h * shape_signature.grid_w // (
            self._llm_merge_size() ** 2
        )
        cache_enabled = os.getenv("JUNKRAT_ENABLE_SINGLE_IMAGE_METADATA_CACHE", "0") == "1"
        span_start = int(torch.argmax(mm_token_type_ids[0].to(torch.int32)).item())
        cache_key = (
            shape_signature.grid_t,
            shape_signature.grid_h,
            shape_signature.grid_w,
            shape_signature.text_seq_len,
            span_start,
            expected_tokens,
            shape_signature.hidden_dtype,
            shape_signature.device,
        )
        if cache_enabled:
            cached = self.runtime_cache.single_image_metadata.get(cache_key)
            if cached is not None:
                return cached
        image_token_span = VisualTokenSpan(
            start=span_start,
            count=expected_tokens,
        )
        position_ids, rope_deltas = self.compute_single_image_position_ids(
            seq_len=shape_signature.text_seq_len,
            image_token_span=image_token_span,
            shape_signature=shape_signature,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        rotary_cache_key = (
            shape_signature.grid_t,
            shape_signature.grid_h,
            shape_signature.grid_w,
            shape_signature.text_seq_len,
            image_token_span.start,
            image_token_span.count,
            shape_signature.hidden_dtype,
            shape_signature.device,
        )
        self.language_model.get_position_embeddings(
            position_ids,
            hidden_dtype=shape_signature.hidden_dtype,
            device=input_ids.device,
            signature=shape_signature,
            rotary_cache_key=rotary_cache_key,
        )
        metadata = PreparedSingleImageMetadata(
            shape_signature=shape_signature,
            visual_token_span=image_token_span,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            rope_delta=int(rope_deltas[0, 0].item()),
            last_token_index=shape_signature.text_seq_len - 1,
            text_rotary_cache_key=rotary_cache_key,
        )
        if cache_enabled:
            self.runtime_cache.single_image_metadata[cache_key] = metadata
        return metadata

    def forward_single_image(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        prepared_metadata: PreparedSingleImageMetadata | None = None,
    ) -> MinimalBackboneOutput:
        if input_ids.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got {input_ids.shape[0]}")
        if pixel_values.ndim != 2:
            raise ValueError(f"Expected `pixel_values` with shape (num_patches, 1536), got {tuple(pixel_values.shape)}")

        del attention_mask

        if prepared_metadata is None:
            embed_dtype = self.language_model.embed_tokens.weight.dtype
            signature = ShapeSignature.from_inputs(
                image_grid_thw=image_grid_thw,
                input_ids=input_ids,
                hidden_dtype=embed_dtype,
                device=input_ids.device,
            )
            expected_tokens = signature.grid_t * signature.grid_h * signature.grid_w // (
                self._llm_merge_size() ** 2
            )
            image_mask = self.get_image_token_mask(input_ids)
            image_token_span = self.get_image_token_span(image_mask, expected_tokens)
            position_ids, rope_deltas = self.compute_single_image_position_ids(
                seq_len=signature.text_seq_len,
                image_token_span=image_token_span,
                shape_signature=signature,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            rotary_cache_key = (
                signature.grid_t,
                signature.grid_h,
                signature.grid_w,
                signature.text_seq_len,
                image_token_span.start,
                image_token_span.count,
                embed_dtype,
                signature.device,
            )
        else:
            signature = prepared_metadata.shape_signature
            image_token_span = prepared_metadata.visual_token_span
            position_ids = prepared_metadata.position_ids
            rope_deltas = prepared_metadata.rope_deltas
            rotary_cache_key = prepared_metadata.text_rotary_cache_key

        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_outputs = self.visual(pixel_values, image_grid_thw=image_grid_thw, shape_signature=signature)
        image_outputs = self.pool_visual_output_to_llm_merge(image_outputs, shape_signature=signature)
        image_embeds = image_outputs.pooler_output.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds.narrow(1, image_token_span.start, image_token_span.count).copy_(image_embeds.unsqueeze(0))
        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            visual_pos_mask=None,
            deepstack_visual_embeds=image_outputs.deepstack_features,
            shape_signature=signature,
            use_padding_mask=False,
            visual_token_span=image_token_span,
            rotary_cache_key=rotary_cache_key,
            deepstack_start_layer=image_outputs.deepstack_start_layer,
        )
        return MinimalBackboneOutput(
            last_hidden_state=hidden_states,
            rope_deltas=rope_deltas,
            position_ids=position_ids,
            shape_signature=signature,
        )

class MinimalQwen3VLForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config,
        *,
        attn_backend: AttnBackend = "eager",
    ) -> None:
        super().__init__()
        if attn_backend not in {"eager", "flash_attn_2"}:
            raise ValueError(f"Unsupported attention backend: {attn_backend}")
        self.config = config
        self.runtime_cache = RuntimeCache()
        self.model = MinimalQwen3VLBackbone(
            config=config,
            runtime_cache=self.runtime_cache,
            attn_backend=attn_backend,
        )
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.attn_backend = attn_backend
        self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def clear_runtime_cache(self) -> None:
        self.runtime_cache.clear()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def warmup_single_image(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        prepared_metadata: PreparedSingleImageMetadata | None = None,
    ) -> ShapeSignature:
        with torch.inference_mode():
            output = self.model.forward_single_image(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                prepared_metadata=prepared_metadata,
            )
        return output.shape_signature

    def prepare_single_image_metadata(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
    ) -> PreparedSingleImageMetadata:
        return self.model.prepare_single_image_metadata(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

    def forward_hidden(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        prepared_metadata: PreparedSingleImageMetadata | None = None,
    ) -> MinimalBackboneOutput:
        return self.model.forward_single_image(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            prepared_metadata=prepared_metadata,
        )

    def forward_last_token_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        prepared_metadata: PreparedSingleImageMetadata | None = None,
    ) -> MinimalCausalLMOutput:
        output = self.forward_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            prepared_metadata=prepared_metadata,
        )
        last_hidden = output.last_hidden_state[:, -1:, :]
        logits = F.linear(last_hidden, self.lm_head.weight)
        return MinimalCausalLMOutput(
            logits=logits,
            last_hidden_state=output.last_hidden_state,
            rope_deltas=output.rope_deltas,
            position_ids=output.position_ids,
            shape_signature=output.shape_signature,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        logits_to_keep: int = 1,
        prepared_metadata: PreparedSingleImageMetadata | None = None,
    ) -> MinimalCausalLMOutput:
        output = self.forward_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            prepared_metadata=prepared_metadata,
        )
        hidden_states = output.last_hidden_state[:, -logits_to_keep:, :]
        logits = F.linear(hidden_states, self.lm_head.weight)
        return MinimalCausalLMOutput(
            logits=logits,
            last_hidden_state=output.last_hidden_state,
            rope_deltas=output.rope_deltas,
            position_ids=output.position_ids,
            shape_signature=output.shape_signature,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attn_backend: AttnBackend = "eager",
    ) -> "MinimalQwen3VLForConditionalGeneration":
        model_path = Path(model_path)
        config = Qwen3VLConfig.from_pretrained(model_path)
        model = cls(config, attn_backend=attn_backend)
        raw_tensors = load_file(model_path / "model.safetensors")
        state_dict = {}
        model_state = model.state_dict()
        for key, target_tensor in model_state.items():
            if key in raw_tensors:
                source_tensor = raw_tensors[key]
                scale_key = f"{key}_scale"
                if source_tensor.dtype == torch.int8 and scale_key in raw_tensors:
                    state_dict[key] = dequantize_int8_scaled_weight(
                        source_tensor,
                        raw_tensors[scale_key],
                        dtype=target_tensor.dtype,
                    )
                else:
                    state_dict[key] = source_tensor
                continue
            packed_key = f"{key}_packed"
            scale_key = f"{key}_scale"
            shape_key = f"{key}_shape"
            if packed_key in raw_tensors and scale_key in raw_tensors and shape_key in raw_tensors:
                state_dict[key] = dequantize_pack_quantized_weight(
                    raw_tensors[packed_key],
                    raw_tensors[scale_key],
                    raw_tensors[shape_key],
                    dtype=target_tensor.dtype,
                )
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"lm_head.weight"}
        if set(missing_keys) != allowed_missing:
            raise RuntimeError(f"Unexpected missing keys: {missing_keys}")
        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        model.tie_weights()
        _attach_quantized_linear_buffers(model, raw_tensors)
        model.to(device=device, dtype=dtype)
        model.eval()
        return model
