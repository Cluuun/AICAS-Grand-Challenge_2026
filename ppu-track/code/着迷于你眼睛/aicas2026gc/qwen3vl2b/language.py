import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from contextlib import nullcontext
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
)
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
import math
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from typing import Optional
from aicas2026gc import RuntimeAutotuner
from transformers.cache_utils import Cache
from aicas2026gc.ops.rms_norm import fused_1d_rmsnorm
from aicas2026gc.ops.rope import _launch_k_rope_to_cache, _launch_rms_norm_rope_to_out
from aicas2026gc.ops import algebra_prep_qkv_and_cache
from aicas2026gc.ops.algebra_split_k_decode import algebra_split_k_decode_cudagraph
from aicas2026gc.ops.decode_skip_bridge import fused_decode_skip_bridge_
from aicas2026gc.ops.swiglu import fused_swiglu_decode, fused_swiglu_prefill
from flash_attn import flash_attn_func
from aicas2026gc.ops import (
    gate_up_gemv_sk4,
    o_gemv_sk32_fp16,
)
from transformers.utils.output_capturing import capture_outputs
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
)

from transformers.processing_utils import Unpack

from .basemodel import Qwen3VLPreTrainedModel, BaseModelOutputWithPast
from .const import (
    QWEN3VL_TEXT_ATTENTION_DECODE_RMS_MEAN,
    QWEN3VL_TEXT_MLP_DECODE_RMS_MEAN,
)
from .layer_stats import maybe_create_layer_stats_collector
from aicas2026gc.timing import timing_end, timing_sample_done, timing_start


# Runtime env flags used in this file:
#
# Prefill INT8 MLP:
#   PREFILL_INT8_MLP=0|gate_up|down|both
#       Enables cuBLASLt INT8 only for language prefill MLP GEMMs.
#       Default is both.
#   PREFILL_INT8_ATTN=0|qkv|o|both
#       Enables cuBLASLt INT8 only for language prefill attention GEMMs.
#       Default is both.
#   PREFILL_INT8_MIN_LAYER=0
#       Default INT8 lower bound when explicit layer allowlists are unset.
#   PREFILL_INT8_MLP_LAYERS=all|0,1,2,...
#       Explicit MLP layer allowlist. Overrides PREFILL_INT8_MIN_LAYER.
#   PREFILL_INT8_ATTN_LAYERS=all|0,1,2,...
#       Explicit attention layer allowlist. Overrides PREFILL_INT8_MIN_LAYER.
#   PREFILL_INT8_AUTOTUNE=0|1
#       0: direct INT8 h0 path. 1: RuntimeAutotuner compares FP16 torch and INT8 candidates.
#       Default is 1.
#   PREFILL_INT8_FUSED_QUANT=0|1
#       Fuses language prefill RMSNorm/SwiGLU with rowwise activation quant
#       before qkv/gate_up/down INT8 GEMMs. Default is 1 in eval.sh.
#   PREFILL_INT8_MAX_ALGOS=8
#       Number of cuBLASLt heuristic candidates to benchmark when AUTOTUNE=1.
#   PREFILL_INT8_WORKSPACE_MB=32
#       cuBLASLt workspace size.
#   PREFILL_INT8_MAX_M=1050
#       Max prefill rows using INT8. Longer fallback uses torch.
#
# Current INT8 coverage:
#   language prefill attention qkv_proj, o_proj+residual,
#   and MLP gate_up_proj, down_proj+residual.
#   No decode and no lm_head path.
#
# Prefill CUDAGraph:
#   PREFILL_PHASE1_CUDAGRAPH=1
#       Captures language prefill layers 0-2 plus dense deepstack full-buffer adds.
#       0 restores the old eager phase1 path.
#
# Decode layer skip:
#   DECODE_SKIP_FUSED_BRIDGE=0|1
#       Fuses the decode LS affine + residual lowrank k128 bridge.
#       Default is 1 and only activates for the current safe rank-128 bridge.
#   DECODE_SKIP_PATCH_SPECS=15:path/to/patch.pt[:alpha]
#       Applies Hadamard LinearPatch-style compensation after decode affine/lowrank.
#   DECODE_SKIP_SOURCE_LOWRANK_SPECS=20:7:path/to/source_delta_lowrank.pt[:rms_reg_fix]
#       Caches decode hidden before source_layer, then adds a source-conditioned
#       lowrank delta before before_layer. Used for h6 -> late residual tests.
#
# Prefill layer skip:
#   PREFILL_LAYER_SKIP_LAYERS=7,8
#       Skips language prefill decoder layers. Must be a subset of DECODE_SKIP_LAYERS.
#   PREFILL_LAYER_SKIP_SCALE_SPECS=9:path/to/ls_affine.pt
#       Applies channel-wise LS affine before the first non-skipped layer.
#   PREFILL_LAYER_SKIP_LOWRANK_SPECS=9:path/to/lowrank.pt[:rms_reg_fix]
#       Applies prefill-specific lowrank compensation before the first non-skipped layer.
#   PREFILL_LAYER_SKIP_STEERING_SPECS=9:path/to/steering.pt:alpha[:rms_protect]
#       Adds a prefill-specific residual steering vector after affine/lowrank compensation.
#   PREFILL_LAYER_SKIP_PATCH_SPECS=9:path/to/official_linearpatch.pt
#       Applies official LinearPatch-style Hadamard channel scaling before the first non-skipped layer.


def maybe_autocast(
    device_type: str,
    dtype=None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()


def _parse_layer_set(value: str, num_layers: int) -> set[int]:
    if not value.strip():
        return set()
    layers = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        layer_idx = int(item)
        if not (0 <= layer_idx < num_layers):
            raise ValueError(f"decode skip layer must be in [0, {num_layers}), got {layer_idx}")
        layers.add(layer_idx)
    return layers


def _parse_prefill_int8_ops(value: str, valid_ops: set[str], aliases: dict[str, str]) -> set[str]:
    raw = value.strip().lower()
    if raw in ("", "0", "false", "f", "no", "off"):
        return set()
    if raw in ("1", "true", "t", "yes", "on", "all", "both"):
        return set(valid_ops)
    ops = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return {aliases.get(op, op) for op in ops if aliases.get(op, op) in valid_ops}


def _parse_prefill_int8_mlp_ops(value: str) -> set[str]:
    return _parse_prefill_int8_ops(value, {"gate_up", "down"}, {"gate": "gate_up", "gateup": "gate_up", "up": "gate_up"})


def _parse_prefill_int8_attn_ops(value: str) -> set[str]:
    return _parse_prefill_int8_ops(value, {"qkv", "o"}, {"out": "o", "proj": "o", "o_proj": "o", "qkv_proj": "qkv"})


def _layer_enabled_by_env(value: str | None, layer_idx: int, num_layers: int) -> bool:
    if value is None or value.strip().lower() in ("", "all", "*"):
        return True
    return layer_idx in _parse_layer_set(value, num_layers)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "t", "yes", "on")


def _load_decode_skip_scale_payload(path: str, hidden_size: int, env_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    scale_vector = payload["scale_vector"] if isinstance(payload, dict) else payload
    bias_vector = payload.get("bias_vector", torch.empty(0, dtype=torch.float16)) if isinstance(payload, dict) else torch.empty(0, dtype=torch.float16)
    if scale_vector.shape != (hidden_size,):
        raise ValueError(f"{env_name} scale vector must have shape=({hidden_size},), got {tuple(scale_vector.shape)}")
    scale_vector = scale_vector.to(torch.float16).contiguous()
    if bias_vector.numel() > 0:
        if bias_vector.shape != (hidden_size,):
            raise ValueError(f"{env_name} bias vector must have shape=({hidden_size},), got {tuple(bias_vector.shape)}")
        bias_vector = bias_vector.to(torch.float16).contiguous()
    return scale_vector, bias_vector


def _load_prefill_layer_skip_patch_payload(path: str, hidden_size: int) -> dict[str, torch.Tensor | str]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"PREFILL_LAYER_SKIP_PATCH_SPECS path must contain a payload dict, got {path}")
    mode = str(payload.get("mode", ""))
    if mode not in {"hadamard_scale", "hadamard_residual"}:
        raise ValueError(f"PREFILL_LAYER_SKIP_PATCH_SPECS unsupported mode={mode!r} in {path}")
    scale_vector = payload.get("scale_vector")
    if not isinstance(scale_vector, torch.Tensor) or scale_vector.shape != (hidden_size,):
        raise ValueError(
            "PREFILL_LAYER_SKIP_PATCH_SPECS scale_vector must have "
            f"shape=({hidden_size},), got {None if scale_vector is None else tuple(scale_vector.shape)}"
        )
    bias_vector = payload.get("bias_vector", torch.empty(0, dtype=torch.float16))
    if not isinstance(bias_vector, torch.Tensor):
        raise ValueError("PREFILL_LAYER_SKIP_PATCH_SPECS bias_vector must be a tensor when present")
    if bias_vector.numel() > 0 and bias_vector.shape != (hidden_size,):
        raise ValueError(
            "PREFILL_LAYER_SKIP_PATCH_SPECS bias_vector must have "
            f"shape=({hidden_size},), got {tuple(bias_vector.shape)}"
        )
    return {
        "mode": mode,
        "scale_vector": scale_vector.to(torch.float16).contiguous(),
        "bias_vector": bias_vector.to(torch.float16).contiguous(),
    }


def _parse_decode_skip_scale_specs(value: str, num_layers: int) -> list[tuple[int, str]]:
    specs: list[tuple[int, str]] = []
    seen: set[int] = set()
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"DECODE_SKIP_SCALE_SPECS item must be before_layer:path, got {item!r}")
        before_text, path = item.split(":", 1)
        before_layer = int(before_text)
        path = path.strip()
        if not (0 <= before_layer < num_layers):
            raise ValueError(f"DECODE_SKIP_SCALE_SPECS before_layer must be in [0, {num_layers}), got {before_layer}")
        if not path:
            raise ValueError(f"DECODE_SKIP_SCALE_SPECS path is empty for before_layer={before_layer}")
        if before_layer in seen:
            raise ValueError(f"DECODE_SKIP_SCALE_SPECS has duplicate before_layer={before_layer}")
        seen.add(before_layer)
        specs.append((before_layer, path))
    return specs


def _parse_prefill_layer_skip_patch_specs(value: str, num_layers: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    seen: set[int] = set()
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 2:
            raise ValueError("PREFILL_LAYER_SKIP_PATCH_SPECS item must be before_layer:path[:alpha]")
        before_layer = int(parts[0])
        path = parts[1].strip()
        alpha = float(parts[2]) if len(parts) >= 3 and parts[2] != "" else None
        if not (0 <= before_layer < num_layers):
            raise ValueError(f"PREFILL_LAYER_SKIP_PATCH_SPECS before_layer must be in [0, {num_layers}), got {before_layer}")
        if not path:
            raise ValueError(f"PREFILL_LAYER_SKIP_PATCH_SPECS path is empty for before_layer={before_layer}")
        if before_layer in seen:
            raise ValueError(f"PREFILL_LAYER_SKIP_PATCH_SPECS has duplicate before_layer={before_layer}")
        seen.add(before_layer)
        specs.append({"before_layer": before_layer, "path": path, "alpha": alpha})
    return specs


def _load_decode_skip_steering_payload(path: str, hidden_size: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    steering_vector = payload["steering_vector"] if isinstance(payload, dict) else payload
    if steering_vector.shape != (hidden_size,):
        raise ValueError(f"DECODE_SKIP_STEERING_SPECS steering vector must have shape=({hidden_size},), got {tuple(steering_vector.shape)}")
    return steering_vector.to(torch.float16).contiguous()


def _parse_decode_skip_steering_specs(value: str, num_layers: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 3:
            raise ValueError(
                "DECODE_SKIP_STEERING_SPECS item must be "
                "before_layer:path:alpha[:rms_protect[:decay_steps[:skip_first_step]]]"
            )
        before_layer = int(parts[0])
        path = parts[1].strip()
        alpha = float(parts[2])
        rms_protect = bool(int(parts[3])) if len(parts) >= 4 and parts[3] != "" else True
        decay_steps = int(parts[4]) if len(parts) >= 5 and parts[4] != "" else 0
        skip_first_step = bool(int(parts[5])) if len(parts) >= 6 and parts[5] != "" else True
        if not (0 <= before_layer < num_layers):
            raise ValueError(f"DECODE_SKIP_STEERING_SPECS before_layer must be in [0, {num_layers}), got {before_layer}")
        if not path:
            raise ValueError(f"DECODE_SKIP_STEERING_SPECS path is empty for before_layer={before_layer}")
        if decay_steps < 0:
            raise ValueError(f"DECODE_SKIP_STEERING_SPECS decay_steps must be >= 0, got {decay_steps}")
        specs.append(
            {
                "before_layer": before_layer,
                "path": path,
                "alpha": alpha,
                "rms_protect": rms_protect,
                "decay_steps": decay_steps,
                "skip_first_step": skip_first_step,
            }
        )
    return specs


def _parse_decode_skip_lowrank_specs(value: str, num_layers: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 2:
            raise ValueError("DECODE_SKIP_LOWRANK_SPECS item must be before_layer:path[:rms_reg_fix]")
        before_layer = int(parts[0])
        path = parts[1].strip()
        rms_reg_fix = bool(int(parts[2])) if len(parts) >= 3 and parts[2] != "" else False
        if not (0 <= before_layer < num_layers):
            raise ValueError(f"DECODE_SKIP_LOWRANK_SPECS before_layer must be in [0, {num_layers}), got {before_layer}")
        if not path:
            raise ValueError(f"DECODE_SKIP_LOWRANK_SPECS path is empty for before_layer={before_layer}")
        specs.append({"before_layer": before_layer, "path": path, "rms_reg_fix": rms_reg_fix})
    return specs


def _parse_decode_skip_source_lowrank_specs(value: str, num_layers: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 3:
            raise ValueError("DECODE_SKIP_SOURCE_LOWRANK_SPECS item must be before_layer:source_layer:path[:rms_reg_fix]")
        before_layer = int(parts[0])
        source_layer = int(parts[1])
        path = parts[2].strip()
        rms_reg_fix = bool(int(parts[3])) if len(parts) >= 4 and parts[3] != "" else False
        if not (0 <= before_layer < num_layers):
            raise ValueError(f"DECODE_SKIP_SOURCE_LOWRANK_SPECS before_layer must be in [0, {num_layers}), got {before_layer}")
        if not (0 <= source_layer < num_layers):
            raise ValueError(f"DECODE_SKIP_SOURCE_LOWRANK_SPECS source_layer must be in [0, {num_layers}), got {source_layer}")
        if source_layer >= before_layer:
            raise ValueError(f"DECODE_SKIP_SOURCE_LOWRANK_SPECS source_layer must be < before_layer, got {source_layer}>={before_layer}")
        if not path:
            raise ValueError(f"DECODE_SKIP_SOURCE_LOWRANK_SPECS path is empty for before_layer={before_layer}")
        specs.append({"before_layer": before_layer, "source_layer": source_layer, "path": path, "rms_reg_fix": rms_reg_fix})
    return specs


def _load_decode_skip_lowrank_payload(path: str, hidden_size: int) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"DECODE_SKIP_LOWRANK_SPECS path must contain a payload dict, got {path}")

    mode = str(payload.get("mode", ""))
    if mode not in {"pure_lowrank", "ls_residual_lowrank"}:
        raise ValueError(f"DECODE_SKIP_LOWRANK_SPECS unsupported mode={mode!r} in {path}")

    mean_in = payload.get("mean_in")
    mean_out = payload.get("mean_out")
    low_a = payload.get("low_a")
    low_b = payload.get("low_b")
    if not isinstance(mean_in, torch.Tensor) or mean_in.shape != (hidden_size,):
        raise ValueError(f"DECODE_SKIP_LOWRANK_SPECS mean_in must have shape=({hidden_size},), got {None if mean_in is None else tuple(mean_in.shape)}")
    if not isinstance(mean_out, torch.Tensor) or mean_out.shape != (hidden_size,):
        raise ValueError(f"DECODE_SKIP_LOWRANK_SPECS mean_out must have shape=({hidden_size},), got {None if mean_out is None else tuple(mean_out.shape)}")
    if not isinstance(low_a, torch.Tensor) or low_a.ndim != 2 or low_a.shape[0] != hidden_size:
        raise ValueError(f"DECODE_SKIP_LOWRANK_SPECS low_a must have shape=({hidden_size}, k), got {None if low_a is None else tuple(low_a.shape)}")
    rank = int(low_a.shape[1])
    if not isinstance(low_b, torch.Tensor) or low_b.shape != (rank, hidden_size):
        raise ValueError(f"DECODE_SKIP_LOWRANK_SPECS low_b must have shape=({rank}, {hidden_size}), got {None if low_b is None else tuple(low_b.shape)}")

    rms_reg_w = payload.get("rms_reg_w", torch.tensor(0.0, dtype=torch.float32, device="cpu"))
    rms_reg_b = payload.get("rms_reg_b", torch.tensor(0.0, dtype=torch.float32, device="cpu"))
    if not isinstance(rms_reg_w, torch.Tensor):
        rms_reg_w = torch.tensor(float(rms_reg_w), dtype=torch.float32, device="cpu")
    if not isinstance(rms_reg_b, torch.Tensor):
        rms_reg_b = torch.tensor(float(rms_reg_b), dtype=torch.float32, device="cpu")

    return {
        "mode": mode,
        "rank": rank,
        "mean_in": mean_in.to(torch.float16).contiguous(),
        "mean_out": mean_out.to(torch.float16).contiguous(),
        "low_a": low_a.to(torch.float16).contiguous(),
        "low_b": low_b.to(torch.float16).contiguous(),
        "rms_reg_w": rms_reg_w.to(torch.float32).contiguous(),
        "rms_reg_b": rms_reg_b.to(torch.float32).contiguous(),
    }


def _load_decode_skip_source_lowrank_payload(path: str, hidden_size: int) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"DECODE_SKIP_SOURCE_LOWRANK_SPECS path must contain a payload dict, got {path}")

    mode = str(payload.get("mode", ""))
    if mode != "source_delta_lowrank":
        raise ValueError(f"DECODE_SKIP_SOURCE_LOWRANK_SPECS unsupported mode={mode!r} in {path}")

    mean_in = payload.get("mean_in")
    mean_delta = payload.get("mean_delta", payload.get("mean_out"))
    low_a = payload.get("low_a")
    low_b = payload.get("low_b")
    if not isinstance(mean_in, torch.Tensor) or mean_in.shape != (hidden_size,):
        raise ValueError(
            "DECODE_SKIP_SOURCE_LOWRANK_SPECS mean_in must have "
            f"shape=({hidden_size},), got {None if mean_in is None else tuple(mean_in.shape)}"
        )
    if not isinstance(mean_delta, torch.Tensor) or mean_delta.shape != (hidden_size,):
        raise ValueError(
            "DECODE_SKIP_SOURCE_LOWRANK_SPECS mean_delta must have "
            f"shape=({hidden_size},), got {None if mean_delta is None else tuple(mean_delta.shape)}"
        )
    if not isinstance(low_a, torch.Tensor) or low_a.ndim != 2 or low_a.shape[0] != hidden_size:
        raise ValueError(
            "DECODE_SKIP_SOURCE_LOWRANK_SPECS low_a must have "
            f"shape=({hidden_size}, k), got {None if low_a is None else tuple(low_a.shape)}"
        )
    rank = int(low_a.shape[1])
    if not isinstance(low_b, torch.Tensor) or low_b.shape != (rank, hidden_size):
        raise ValueError(
            "DECODE_SKIP_SOURCE_LOWRANK_SPECS low_b must have "
            f"shape=({rank}, {hidden_size}), got {None if low_b is None else tuple(low_b.shape)}"
        )

    rms_reg_w = payload.get("rms_reg_w", torch.tensor(0.0, dtype=torch.float32, device="cpu"))
    rms_reg_b = payload.get("rms_reg_b", torch.tensor(0.0, dtype=torch.float32, device="cpu"))
    if not isinstance(rms_reg_w, torch.Tensor):
        rms_reg_w = torch.tensor(float(rms_reg_w), dtype=torch.float32, device="cpu")
    if not isinstance(rms_reg_b, torch.Tensor):
        rms_reg_b = torch.tensor(float(rms_reg_b), dtype=torch.float32, device="cpu")

    return {
        "mode": mode,
        "rank": rank,
        "mean_in": mean_in.to(torch.float16).contiguous(),
        "mean_delta": mean_delta.to(torch.float16).contiguous(),
        "low_a": low_a.to(torch.float16).contiguous(),
        "low_b": low_b.to(torch.float16).contiguous(),
        "rms_reg_w": rms_reg_w.to(torch.float32).contiguous(),
        "rms_reg_b": rms_reg_b.to(torch.float32).contiguous(),
    }


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3VLTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)  # [64]
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

        self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])

        # Runtime auto-tuner for RoPE kernel selection (shared across instances)
        self.rope_tuner = RuntimeAutotuner(name=self.__class__.__name__, fallback="table", strict=True)

    def post_register_buffer(self):
        # WOC, WOC, WOC 这个图片的 position embedding 其实没有屁的作用????
        # 在 __init__ 中执行中执行 self.post_register_buffer() self.mrope_indices 全是 0 !!!!!!!!!!!
        # 要在外部执行 self.post_register_buffer
        mrope_indices = torch.zeros_like(self.inv_freq, dtype=torch.long)

        for dim, offset in enumerate((1, 2), start=1):  # 1:H, 2:W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            mrope_indices[idx] = dim  # 记录这一维该取 T(0), H(1), 还是 W(2)

        self.register_buffer("mrope_indices", mrope_indices, persistent=False)

    def post_init_inv_freq_cache_sin_cos(self):
        inv_freq_cache = torch.arange(4096, dtype=torch.float32, device=self.inv_freq.device)[:, None] * self.inv_freq  # TODO
        inv_freq_cache_cos = inv_freq_cache.cos() * self.attention_scaling  # * 1
        inv_freq_cache_sin = inv_freq_cache.sin() * self.attention_scaling  # * 1

        self.register_buffer("inv_freq_cache_cos", inv_freq_cache_cos, persistent=False)
        self.register_buffer("inv_freq_cache_sin", inv_freq_cache_sin, persistent=False)

    @staticmethod
    def compute_default_rope_parameters(  # ONLY FOR INIT
        config: Qwen3VLTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

    @torch.inference_mode()
    def forward(self, position_ids, is_decoding=False, buffer4decode=None, device_type="cuda", dtype=torch.float16):
        # position_ids.shape: [3, 1, 690] or [3, 1, 1], BS=1
        # self.inv_freq.shape: [64]
        if is_decoding:
            return self.decode_forward(position_ids, is_decoding, buffer4decode, device_type, dtype)
        return self.prefill_forward(position_ids, dtype)

    def decode_forward(self, position_ids, is_decoding=False, buffer4decode=None, device_type="cuda", dtype=torch.float16):
        candidates = {
            "calc": lambda position_ids: self.decode_forward_calc(position_ids, is_decoding, buffer4decode, device_type, dtype),
            "table": lambda position_ids: self.decode_forward_table(position_ids, is_decoding, buffer4decode, device_type, dtype),
        }
        return self.rope_tuner.dispatch_once(candidates, name="decode", args=(position_ids,))

    def decode_forward_calc(self, position_ids, is_decoding=False, buffer4decode=None, device_type="cuda", dtype=torch.float16):
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (position_ids[0][0] * self.inv_freq)[None, None]  # Decoding
            cos = freqs.cos() * self.attention_scaling  # * 1
            sin = freqs.sin() * self.attention_scaling  # * 1

        buffer4decode: nn.Tensor  # [2, 1, 1, dim]
        buffer4decode[0].copy_(cos)
        buffer4decode[1].copy_(sin)
        res = buffer4decode[0], buffer4decode[1]
        return res

    def decode_forward_table(self, position_ids, is_decoding=False, buffer4decode=None, device_type="cuda", dtype=torch.float16):
        cos = self.inv_freq_cache_cos[position_ids[0][0]][None]
        sin = self.inv_freq_cache_sin[position_ids[0][0]][None]

        buffer4decode: nn.Tensor  # [2, 1, 1, dim]
        buffer4decode[0].copy_(cos)
        buffer4decode[1].copy_(sin)
        res = buffer4decode[0], buffer4decode[1]
        return res

    def prefill_forward(self, position_ids, dtype=torch.float16):
        candidates = {
            "calc": lambda position_ids: self.prefill_forward_calc(position_ids, dtype),
            "table": lambda position_ids: self.prefill_forward_table(position_ids, dtype),
        }
        return self.rope_tuner.dispatch_once(candidates, name="prefill", args=(position_ids,))

    def prefill_forward_calc(self, position_ids, dtype=torch.float16):
        with maybe_autocast(device_type="cuda", enabled=False):  # Force float32
            freqs = position_ids[self.mrope_indices].permute(1, 2, 0) * self.inv_freq
            cos = freqs.cos() * self.attention_scaling  # * 1
            sin = freqs.sin() * self.attention_scaling  # * 1
        res = cos.to(dtype=dtype), sin.to(dtype=dtype)
        return res

    def prefill_forward_table(self, position_ids, dtype=torch.float16):
        # 查表的
        # [3, 1, 690]-> [1, 690, 64]
        indices_3d = position_ids[self.mrope_indices].permute(1, 2, 0)
        idx_2d = indices_3d.squeeze(0)  # [690, 64]

        # [4096, 64] -> [690, 64]
        cos_2d = torch.gather(self.inv_freq_cache_cos, dim=0, index=idx_2d)  # [690, 64], 输出和 index shape 一致
        sin_2d = torch.gather(self.inv_freq_cache_sin, dim=0, index=idx_2d)
        res = cos_2d[None].to(dtype), sin_2d[None].to(dtype)
        return res


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3VLTextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.hidden_dim = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.decode_norm_tuner = RuntimeAutotuner(name=self.__class__.__name__, fallback="fused_1d", strict=True, check_correctness=True)
        self.prefill_norm_tuner = RuntimeAutotuner(name=self.__class__.__name__, fallback="torch_rms_norm", strict=True)

    def forward(self, hidden_states: torch.Tensor, decode=False, inplace=False, out=None, cg_bucket_size: int | None = None, is_warmup: bool = False) -> torch.Tensor:
        if decode:  # 这个 BS=1
            if out is not None:
                return fused_1d_rmsnorm(hidden_states, self.weight, eps=self.variance_epsilon, inplace=False, out=out)

            decode_candidates = {
                "fused_1d": lambda hidden_states: fused_1d_rmsnorm(hidden_states, self.weight, eps=self.variance_epsilon, inplace=False, out=out),
                "torch_rms_norm": lambda hidden_states: torch.nn.functional.rms_norm(hidden_states, [self.hidden_dim], self.weight, eps=self.variance_epsilon),
                "triton_general": lambda hidden_states: torch.ops.qwen3vl.rmsnorm_forward(
                    hidden_states, self.hidden_dim, weight=self.weight, inplace=False, eps=self.variance_epsilon
                ),
            }
            return self.decode_norm_tuner.dispatch_once(decode_candidates, name="decode", args=(hidden_states,))

        # Prefill
        # [1, 690, 2048], [2048] -> [1, 690, 2048]
        prefill_candidates = {
            "torch_rms_norm": lambda hidden_states: torch.nn.functional.rms_norm(hidden_states, [self.hidden_dim], self.weight, eps=self.variance_epsilon),
            "triton_general": lambda hidden_states: torch.ops.qwen3vl.rmsnorm_forward(hidden_states, self.hidden_dim, weight=self.weight, inplace=False, eps=self.variance_epsilon),
        }
        if cg_bucket_size is not None:
            return self.prefill_norm_tuner.dispatch(cg_bucket_size, prefill_candidates, is_warmup, args=(hidden_states,))
        return self.prefill_norm_tuner.dispatch_once(prefill_candidates, name="prefill", args=(hidden_states,))
        # Torch Native:
        # input_dtype = hidden_states.dtype
        # hidden_states = hidden_states.to(torch.float32)
        # variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def plot_attn_safe(img_path, attn_tensor, token_text):
    """
    attn_tensor: 应该是从 question_to_image_attn 提取出的单头、单token权重
    """
    import cv2
    import matplotlib.pyplot as plt

    # 1. 确保数据在 CPU 上，且转为 float32 (OpenCV 兼容)
    # 2. 移除冗余维度 (Squeeze)
    attn_map = attn_tensor.detach().to(torch.float32).cpu().squeeze().numpy()

    # 检查维度是否为 2D (应该是 24x28 或类似)
    if len(attn_map.shape) != 2:
        print(f"Error: attn_map shape is {attn_map.shape}, needs to be 2D.")
        return

    # 读取原图
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # 3. 归一化到 [0, 1] 方便观察
    # 这一步很重要，因为切片后的 softmax 和不一定是 1
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    # 4. Resize
    # 注意 OpenCV 的 resize 参数是 (width, height)
    heatmap_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    # 使用 jet 或 viridis 这种对比度强的颜色映射
    plt.imshow(heatmap_resized, alpha=0.6, cmap="jet")
    plt.title(f"Attention for: {token_text}")
    plt.axis("off")
    plt.savefig("x.png")


class Qwen3VLTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,  # False
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,  # False
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,  # False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,  # False
        )
        self.qkv_odim = [
            config.num_attention_heads * self.head_dim,
            config.num_key_value_heads * self.head_dim,
            config.num_key_value_heads * self.head_dim,
        ]
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            sum(self.qkv_odim),
            bias=config.attention_bias,  # False
        )

        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        MAX_SEQ_LEN = 4096  # 你的 KV Cache 支持的最大长度
        BLOCK_SEQ = 128  # 我们特化设置的 Chunk 大小
        NUM_Q_HEADS = config.num_attention_heads  # Q 的头数
        HEAD_DIM = self.head_dim  # Head 维度

        MAX_CHUNKS = math.ceil(MAX_SEQ_LEN / BLOCK_SEQ)

        acc_partial_buffer = torch.empty((NUM_Q_HEADS, MAX_CHUNKS, HEAD_DIM))
        m_partial_buffer = torch.empty((NUM_Q_HEADS, MAX_CHUNKS), dtype=torch.float32)
        l_partial_buffer = torch.empty((NUM_Q_HEADS, MAX_CHUNKS), dtype=torch.float32)

        self.register_buffer("acc_partial_buffer", acc_partial_buffer)
        self.register_buffer("m_partial_buffer", m_partial_buffer)
        self.register_buffer("l_partial_buffer", l_partial_buffer)

        self.replaced = False
        self.decode_replace_range = list(range(17, 27 + 1))  # TODO: 这里需要调整
        self.decode_replace_range = []
        self.register_buffer("qkv_proj_decode_weight", None, persistent=False)
        self.prefill_qkv_int8 = None
        self.prefill_o_int8 = None
        self.prefill_int8_fused_quant = _env_flag("PREFILL_INT8_FUSED_QUANT", "0")

    def _maybe_init_prefill_int8_attn(self):
        ops = _parse_prefill_int8_attn_ops(os.environ.get("PREFILL_INT8_ATTN", "both"))
        if not ops:
            return
        layers_env = os.environ.get("PREFILL_INT8_ATTN_LAYERS")
        if layers_env is None:
            min_layer = int(os.environ.get("PREFILL_INT8_MIN_LAYER", "0"))
            if self.layer_idx < min_layer:
                return
        elif not _layer_enabled_by_env(layers_env, self.layer_idx, self.config.num_hidden_layers):
            return

        from aicas2026gc.ops.int8_linear_cublaslt import CublasLtInt8Linear

        workspace_mb = int(os.environ.get("PREFILL_INT8_WORKSPACE_MB", "32"))
        max_algos = int(os.environ.get("PREFILL_INT8_MAX_ALGOS", "8"))
        max_m = int(os.environ.get("PREFILL_INT8_MAX_M", "1050"))
        autotune = _env_flag("PREFILL_INT8_AUTOTUNE", "1")

        if "qkv" in ops:
            self.prefill_qkv_int8 = CublasLtInt8Linear(
                self.qkv_proj.weight,
                self.qkv_proj.bias,
                name="prefill_attn_qkv",
                workspace_mb=workspace_mb,
                max_algos=max_algos,
                max_m=max_m,
                autotune=autotune,
            )
        if "o" in ops:
            self.prefill_o_int8 = CublasLtInt8Linear(
                self.o_proj.weight,
                self.o_proj.bias,
                name="prefill_attn_o_residual",
                workspace_mb=workspace_mb,
                max_algos=max_algos,
                max_m=max_m,
                autotune=autotune,
            )
        print(
            f"[PrefillINT8Attn] layer={self.layer_idx} ops={sorted(ops)} "
            f"max_m={max_m} max_algos={max_algos} workspace_mb={workspace_mb} autotune={int(autotune)}",
            flush=True,
        )

    def linear_post_init(self):
        if self.replaced:
            return

        # [out, in]
        self.qkv_proj.weight.data.copy_(
            torch.cat(
                [
                    self.q_proj.weight.data * self.input_layernorm.weight,
                    self.k_proj.weight.data * self.input_layernorm.weight,
                    self.v_proj.weight.data * self.input_layernorm.weight,
                ],
                dim=0,
            ).contiguous()
        )
        del self.q_proj
        del self.k_proj
        del self.v_proj
        # del self.input_layernorm

        self.replaced = True
        if self.layer_idx in self.decode_replace_range:
            self.qkv_proj_decode_weight = self.qkv_proj.weight.detach().clone()
            self.qkv_proj_decode_weight.mul_(QWEN3VL_TEXT_ATTENTION_DECODE_RMS_MEAN[self.layer_idx])
        self._maybe_init_prefill_int8_attn()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,  # None
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        current_pos: torch.LongTensor | None = None,
        is_decoding: bool = False,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        cache_seqlens: torch.Tensor = None,  # bs=1, FOR CUDAGraph
        decode_norm_buffer: torch.Tensor | None = None,
        decode_attn_output_buffer: torch.Tensor | None = None,
        cg_bucket_size: int | None = None,
        is_warmup: bool = False,
        layer_stats_collector=None,
        write_kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if is_decoding:
            return self._decode_forward(
                hidden_states,
                position_embeddings,
                is_decoding=True,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                decode_norm_buffer=decode_norm_buffer,
                decode_attn_output_buffer=decode_attn_output_buffer,
                cg_bucket_size=cg_bucket_size,
                is_warmup=is_warmup,
                layer_stats_collector=layer_stats_collector,
            )

        return self._prefill_forward(
            hidden_states,
            position_embeddings,
            is_decoding=False,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            cg_bucket_size=cg_bucket_size,
            is_warmup=is_warmup,
            write_kv_cache=write_kv_cache,
        )

    def _prefill_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,  # None
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        current_pos: torch.LongTensor | None = None,
        is_decoding: bool = False,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        cache_seqlens: torch.Tensor = None,  # bs=1, FOR CUDAGraph
        cg_bucket_size: int | None = None,
        is_warmup: bool = False,
        write_kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        residual = hidden_states

        bs, seq_len, hidden_dim = hidden_states.shape
        hidden_shape = [bs, seq_len, -1, self.head_dim]

        m = bs * seq_len
        use_fused_qkv_quant = (
            self.prefill_int8_fused_quant
            and self.prefill_qkv_int8 is not None
            and self.prefill_qkv_int8.can_prequant(m, hidden_states.dtype)
        )
        if use_fused_qkv_quant:
            from aicas2026gc.ops.int8_linear_cublaslt import rmsnorm_rowwise_quant_out

            hidden_2d = hidden_states.view(-1, self.hidden_size)
            x_q, x_scale = self.prefill_qkv_int8.prequant_buffers(m, hidden_2d.device, hidden_2d.dtype)
            rmsnorm_rowwise_quant_out(hidden_2d, x_q, x_scale, self.input_layernorm.variance_epsilon)
            qkv_states = self.prefill_qkv_int8.forward_prequant(
                x_q,
                x_scale,
                input_shape=tuple(hidden_states.shape),
                bucket=cg_bucket_size,
                is_warmup=is_warmup,
            )
        else:
            hidden_states = F.rms_norm(hidden_states, [self.hidden_size], eps=self.input_layernorm.variance_epsilon)
            if self.prefill_qkv_int8 is None:
                qkv_states = self.qkv_proj(hidden_states)
            else:
                qkv_states = self.prefill_qkv_int8(hidden_states, bucket=cg_bucket_size, is_warmup=is_warmup)
        query_states, key_states, value_states = qkv_states.split(self.qkv_odim, dim=-1)

        cos, sin = position_embeddings
        query_states = torch.ops.qwen3vl.fused_rms_norm_rope_triton_bs1(query_states, list(position_embeddings), self.q_norm.weight, hidden_shape, eps=1e-6, inplace=False)

        if write_kv_cache:
            # K: RMSNorm + RoPE -> write directly into persistent k_cache.
            _launch_k_rope_to_cache(
                key_states,
                self.k_norm.weight,
                cos,
                sin,
                k_cache,
                seq_len,
                hidden_shape,
                self.head_dim,
                eps=1e-6,
            )

            # V: copy into persistent v_cache (view makes it contiguous in cache).
            v_cache[:, :seq_len, :, :] = value_states.view(hidden_shape)
            cache_seqlens.fill_(seq_len)

            # flash_attn reads from cache slices — K is already there, V is contiguous in v_cache.
            k_from_cache = k_cache[:, :seq_len, :, :]
            v_from_cache = v_cache[:, :seq_len, :, :]
        else:
            # Decode layer-skip never reads these layers' prompt KV. Keep the
            # current prefill attention exact, but avoid dirtying persistent KV.
            k_from_cache = key_states.view(hidden_shape)
            _launch_rms_norm_rope_to_out(
                key_states,
                self.k_norm.weight,
                cos,
                sin,
                k_from_cache,
                self.head_dim,
                eps=1e-6,
            )
            v_from_cache = value_states.view(hidden_shape)

        # # breakpoint()
        # kv_len = 400
        # k_cache[:, :kv_len] = k_from_cache[:, -kv_len:].clone()
        # v_cache[:, :kv_len] = v_from_cache[:, -kv_len:].clone()
        # k_from_cache = k_cache[:, :kv_len]
        # v_from_cache = v_cache[:, :kv_len]
        # cache_seqlens.fill_(kv_len)

        # if self.layer_idx == 23:
        #     breakpoint()
        # k_rep = torch.repeat_interleave(k_from_cache, repeats=2, dim=2) # [1, 690, 16, 128]
        # Q = query_states.transpose(1, 2)    # [1, 16, 690, 128]
        # k = k_rep.transpose(1, 2)           # [1, 16, 690, 128]
        # attn_logits = torch.matmul(Q, k.transpose(-1, -2)) / math.sqrt(128)
        # mask = torch.triu(torch.ones(690, 690, device='cuda'), diagonal=1).bool()
        # attn_logits.masked_fill_(mask, float('-inf'))
        # attn_probs = torch.softmax(attn_logits, dim=-1) # [1, 16, 690, 690], [42 // 2, 64 // 2]

        # # 4 个基本 token
        # # 672 个图片 token
        # # 14 个问题 token

        # question_to_image_attn = attn_probs[:, :, 677:685, 4:676]
        # head_idx = 0
        # token_idx = 6 # 对应问题中的第 1 个词, 13
        # heatmap = question_to_image_attn[0, head_idx, token_idx, :].reshape(42 // 2, 64 // 2)
        # plot_attn_safe("images/0.jpg", heatmap, "brand")

        # if self.layer_idx == 23:
        #     breakpoint()

        attn_output = flash_attn_func(query_states, k_from_cache, v_from_cache, causal=True)
        attn_output = attn_output.flatten(-2)

        if self.prefill_o_int8 is None:
            attn_output = torch.addmm(
                residual.view(-1, self.hidden_size),
                attn_output.view(-1, self.hidden_size),
                self.o_proj.weight.T,
            ).view_as(residual)
        else:
            attn_output = self.prefill_o_int8(attn_output, bucket=cg_bucket_size, is_warmup=is_warmup, residual=residual)

        return attn_output, None

    def _decode_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,  # None
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        current_pos: torch.LongTensor | None = None,
        is_decoding: bool = False,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        cache_seqlens: torch.Tensor = None,  # bs=1, FOR CUDAGraph
        decode_norm_buffer: torch.Tensor | None = None,
        decode_attn_output_buffer: torch.Tensor | None = None,
        cg_bucket_size: int | None = None,
        is_warmup: bool = False,
        layer_stats_collector=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert cg_bucket_size is not None and cg_bucket_size > 0

        residual = hidden_states

        bs, seq_len, _ = hidden_states.shape
        hidden_shape = (bs, seq_len, -1, self.head_dim)
        if self.layer_idx in self.decode_replace_range:
            qkv_states = F.linear(hidden_states, self.qkv_proj_decode_weight, self.qkv_proj.bias)
        else:
            hidden_states = fused_1d_rmsnorm(hidden_states, None, eps=self.input_layernorm.variance_epsilon, out=decode_norm_buffer)
            qkv_states = self.qkv_proj(hidden_states)

        # TODO: 这个确认就是最好的吗?
        q_states = algebra_prep_qkv_and_cache(
            qkv_states,
            position_embeddings,
            self.q_norm.weight,
            self.k_norm.weight,
            k_cache,
            v_cache,
            cache_seqlens,  # Tensor
            hidden_shape,
            qkv_odim=self.qkv_odim,
            inplace=False,
        )
        if layer_stats_collector is not None:
            layer_stats_collector.record_kv_gap(self.layer_idx, k_cache, v_cache, cache_seqlens)

        attn_output = algebra_split_k_decode_cudagraph(
            q_states,
            k_cache,
            v_cache,
            cache_seqlens,
            self.acc_partial_buffer,
            self.m_partial_buffer,
            self.l_partial_buffer,
            cg_bucket_size,
        ).flatten(2)

        attn_output = torch.addmm(residual.squeeze(0), attn_output.squeeze(0), self.o_proj.weight.T).view_as(residual)
        return attn_output, None


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 2048
        self.intermediate_size = config.intermediate_size  # 6144
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # silu
        self.gate_up_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.post_attention_layernorm = nn.RMSNorm([self.hidden_size])  # TODO, 用纯权重就行
        self.replaced = False
        self.decode_replace_range = list(range(17, 27 + 1))  # TODO: 这里需要调整
        self.decode_replace_range = []
        self.register_buffer("gate_up_proj_decode_weight", None, persistent=False)
        self.prefill_gate_up_int8 = None
        self.prefill_down_int8 = None
        self.prefill_int8_fused_quant = _env_flag("PREFILL_INT8_FUSED_QUANT", "0")

        self.layer_idx = layer_idx  # 记录是第几个 Qwen3VLTextDecoderLayer

    def _maybe_init_prefill_int8_mlp(self):
        ops = _parse_prefill_int8_mlp_ops(os.environ.get("PREFILL_INT8_MLP", "both"))
        if not ops:
            return
        layers_env = os.environ.get("PREFILL_INT8_MLP_LAYERS")
        if layers_env is None:
            min_layer = int(os.environ.get("PREFILL_INT8_MIN_LAYER", os.environ.get("PREFILL_INT8_MLP_MIN_LAYER", "0")))
            if self.layer_idx < min_layer:
                return
        elif not _layer_enabled_by_env(layers_env, self.layer_idx, self.config.num_hidden_layers):
            return

        from aicas2026gc.ops.int8_linear_cublaslt import CublasLtInt8Linear

        workspace_mb = int(os.environ.get("PREFILL_INT8_WORKSPACE_MB", os.environ.get("PREFILL_INT8_MLP_WORKSPACE_MB", "32")))
        max_algos = int(os.environ.get("PREFILL_INT8_MAX_ALGOS", os.environ.get("PREFILL_INT8_MLP_MAX_ALGOS", "8")))
        max_m = int(os.environ.get("PREFILL_INT8_MAX_M", os.environ.get("PREFILL_INT8_MLP_MAX_M", "1050")))
        autotune = _env_flag("PREFILL_INT8_AUTOTUNE", os.environ.get("PREFILL_INT8_MLP_AUTOTUNE", "1"))

        if "gate_up" in ops:
            self.prefill_gate_up_int8 = CublasLtInt8Linear(
                self.gate_up_proj.weight,
                name="prefill_mlp_gate_up",
                workspace_mb=workspace_mb,
                max_algos=max_algos,
                max_m=max_m,
                autotune=autotune,
            )
        if "down" in ops:
            self.prefill_down_int8 = CublasLtInt8Linear(
                self.down_proj.weight,
                name="prefill_mlp_down_residual",
                workspace_mb=workspace_mb,
                max_algos=max_algos,
                max_m=max_m,
                autotune=autotune,
            )
        print(
            f"[PrefillINT8MLP] layer={self.layer_idx} ops={sorted(ops)} "
            f"max_m={max_m} max_algos={max_algos} workspace_mb={workspace_mb} autotune={int(autotune)}",
            flush=True,
        )

    def linear_post_init(self):
        if self.replaced:
            return
        combined_weight = torch.cat(
            [self.gate_proj.weight.detach().data, self.up_proj.weight.detach().data],
            dim=0,
        ).contiguous()
        self.gate_up_proj.weight.data.copy_(combined_weight)
        # 合并前面的 RMSNorm 的权重
        self.gate_up_proj.weight.data.copy_(self.gate_up_proj.weight.data * self.post_attention_layernorm.weight.data)
        self.post_attention_layernorm.weight.data.fill_(1.0)

        del self.gate_proj
        del self.up_proj
        self.replaced = True

        # 此处干掉无参数的 rms_norm, 用常数取代 rms
        if self.layer_idx in self.decode_replace_range:
            self.gate_up_proj_decode_weight = self.gate_up_proj.weight.detach().clone()
            self.gate_up_proj_decode_weight.mul_(QWEN3VL_TEXT_MLP_DECODE_RMS_MEAN[self.layer_idx])

        self._maybe_init_prefill_int8_mlp()

    def _decode_forward(self, x, decode_norm_buffer=None):
        # 就干了一件事，
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # x.shape: [1, 1, 2048]
        residual = x

        if self.layer_idx in self.decode_replace_range:
            # [1, 2048] @ [12288, 2048].T
            gate_up = F.linear(x, self.gate_up_proj_decode_weight)
        else:
            x = fused_1d_rmsnorm(x, None, eps=self.post_attention_layernorm.eps, out=decode_norm_buffer)
            # [1, 2048] @ [12288, 2048].T
            gate_up = self.gate_up_proj(x)

        activated = fused_swiglu_decode(gate_up)
        return torch.addmm(residual.squeeze(0), activated.squeeze(0), self.down_proj.weight.T).view_as(residual)

    def _prefill_forward(self, x, cg_bucket_size=None, is_warmup=False):
        residual = x
        bs, seq_len, _ = x.shape
        m = bs * seq_len
        use_fused_gate_quant = (
            self.prefill_int8_fused_quant
            and self.prefill_gate_up_int8 is not None
            and self.prefill_gate_up_int8.can_prequant(m, x.dtype)
        )
        if use_fused_gate_quant:
            from aicas2026gc.ops.int8_linear_cublaslt import rmsnorm_rowwise_quant_out

            x_2d = x.view(-1, self.hidden_size)
            x_q, x_scale = self.prefill_gate_up_int8.prequant_buffers(m, x_2d.device, x_2d.dtype)
            rmsnorm_rowwise_quant_out(x_2d, x_q, x_scale, self.post_attention_layernorm.eps)
            gate_up = self.prefill_gate_up_int8.forward_prequant(
                x_q,
                x_scale,
                input_shape=tuple(x.shape),
                bucket=cg_bucket_size,
                is_warmup=is_warmup,
            )
        else:
            x_normed = torch.ops.qwen3vl.rmsnorm_forward(x, self.hidden_size, None, False, self.post_attention_layernorm.eps)
            if self.prefill_gate_up_int8 is None:
                gate_up = self.gate_up_proj(x_normed)
            else:
                gate_up = self.prefill_gate_up_int8(x_normed, bucket=cg_bucket_size, is_warmup=is_warmup)

        use_fused_down_quant = (
            self.prefill_int8_fused_quant
            and self.prefill_down_int8 is not None
            and self.prefill_down_int8.can_prequant(m, gate_up.dtype)
        )
        if use_fused_down_quant:
            from aicas2026gc.ops.int8_linear_cublaslt import swiglu_rowwise_quant_out

            gate_up_2d = gate_up.view(-1, self.intermediate_size * 2)
            x_q, x_scale = self.prefill_down_int8.prequant_buffers(m, gate_up_2d.device, gate_up_2d.dtype)
            swiglu_rowwise_quant_out(gate_up_2d, x_q, x_scale)
            return self.prefill_down_int8.forward_prequant(
                x_q,
                x_scale,
                input_shape=(*gate_up.shape[:-1], self.intermediate_size),
                bucket=cg_bucket_size,
                is_warmup=is_warmup,
                residual=residual,
            )

        activated = fused_swiglu_prefill(gate_up)
        if self.prefill_down_int8 is not None:
            return self.prefill_down_int8(activated, bucket=cg_bucket_size, is_warmup=is_warmup, residual=residual)
        return torch.addmm(residual.squeeze(0), activated.squeeze(0), self.down_proj.weight.T).view_as(residual)

    def forward(self, x, decode=False, decode_norm_buffer=None, cg_bucket_size=None, is_warmup=False):
        if decode:
            return self._decode_forward(x, decode_norm_buffer=decode_norm_buffer)
        return self._prefill_forward(x, cg_bucket_size=cg_bucket_size, is_warmup=is_warmup)


class Qwen3VLTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config, layer_idx)

        # 这两 norm 已被融合，但是不能被注释！用来参数中转
        self.input_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_idx = layer_idx

    def mlp_post_init(self):

        self.self_attn.input_layernorm.weight.data.copy_(self.input_layernorm.weight)
        self.self_attn.input_layernorm.eps = self.input_layernorm.variance_epsilon
        del self.input_layernorm

        self.mlp.post_attention_layernorm.weight.data.copy_(self.post_attention_layernorm.weight)
        self.mlp.post_attention_layernorm.eps = self.post_attention_layernorm.variance_epsilon
        del self.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        is_decoding: bool = False,
        attn_k_cache: torch.Tensor | None = None,
        attn_v_cache: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
        decode_attn_norm_buffer: torch.Tensor | None = None,
        decode_attn_output_buffer: torch.Tensor | None = None,
        decode_mlp_norm_buffer: torch.Tensor | None = None,
        cg_bucket_size: int | None = None,
        is_warmup: bool = False,
        layer_stats_collector=None,
        write_kv_cache: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:

        h_in = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            past_key_values=None,
            cache_position=None,
            position_embeddings=position_embeddings,
            is_decoding=is_decoding,
            k_cache=attn_k_cache,
            v_cache=attn_v_cache,
            cache_seqlens=cache_seqlens,
            decode_norm_buffer=decode_attn_norm_buffer,
            decode_attn_output_buffer=decode_attn_output_buffer,
            cg_bucket_size=cg_bucket_size,
            is_warmup=is_warmup,
            layer_stats_collector=layer_stats_collector,
            write_kv_cache=write_kv_cache,
        )
        h_attn = hidden_states
        # if is_decoding:
        #     breakpoint()
        #     attention_sign = (hidden_states > 0).flatten()

        hidden_states = self.mlp(hidden_states, is_decoding, decode_norm_buffer=decode_mlp_norm_buffer, cg_bucket_size=cg_bucket_size, is_warmup=is_warmup)
        if layer_stats_collector is not None:
            layer_stats_collector.record_layer(
                layer_idx=self.layer_idx,
                is_decoding=is_decoding,
                h_in=h_in,
                h_attn=h_attn,
                h_out=hidden_states,
            )

        # if is_decoding:
        #     mlp_sign = (hidden_states > 0).flatten()
        #     breakpoint()
        #     print( (attention_sign == mlp_sign).sum() )
        return hidden_states


@auto_docstring(custom_intro=("Text part of Qwen3VL, not a pure text-only model, as DeepStack integrates visual features into the early hidden states."))
class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLTextConfig
    input_modalities = ("text",)
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)  # 这个就是输入层的 word embedding
        self.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # 最后的输出层 norm，weight 已被融合，forward 可以注释，原因见后面
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # FOR CUDAGRAPH, ONLY FOR DECODE BS=1
        pe_buffer4decode = torch.empty(2, 1, 1, config.head_dim // 2)
        self.register_buffer("pe_buffer4decode", pe_buffer4decode, persistent=False)
        attn_kv_cache = torch.empty(config.num_hidden_layers, 2, 1, 4096, config.num_key_value_heads, config.head_dim)  # [28, 2, 1, 4096, 8, 128] # FOR BS=1
        self.register_buffer("attn_kv_cache", attn_kv_cache, persistent=False)
        cache_seqlens = torch.zeros(1, dtype=torch.int32)  # for flash attn
        self.register_buffer("cache_seqlens", cache_seqlens, persistent=False)
        hidden_states_buffer = torch.randn(1, 1, config.hidden_size)
        self.register_buffer("hidden_states_buffer", hidden_states_buffer, persistent=False)
        position_ids_buffer = torch.empty(3, 1, 1, dtype=torch.int32)
        self.register_buffer("position_ids_buffer", position_ids_buffer, persistent=False)
        decode_attn_norm_buffers = torch.empty(config.num_hidden_layers, 1, 1, config.hidden_size)
        self.register_buffer("decode_attn_norm_buffers", decode_attn_norm_buffers, persistent=False)
        decode_attn_output_buffers = torch.empty(config.num_hidden_layers, 1, 1, config.hidden_size)
        self.register_buffer("decode_attn_output_buffers", decode_attn_output_buffers, persistent=False)
        decode_mlp_norm_buffers = torch.empty(config.num_hidden_layers, 1, 1, config.hidden_size)
        self.register_buffer("decode_mlp_norm_buffers", decode_mlp_norm_buffers, persistent=False)

        # CUDA Graph 序列长度分桶 (Buckets)
        self.bucket_upper_bounds = [512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 4096 - 128]  # TODO: 4096 要改成全局变量
        self.bucket_lower_bounds = [0, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]

        self.decode_graphs = {}  # 存放不同 Bucket 对应的 CUDA Graph 对象
        self.decode_graph_outputs = {}
        self.is_decode_graphs_captured = False  # 防止重复录制的 flag
        self.cache_seqlens_len = 4096

        # FOR CUDA GRAPH, PREFILL
        # fmt: off
        self.prefill_graph_seq_lens = [
            272,   336,   337,   368,   369,   370,   372,   400,   401,   402,
            403,   431,   432,   433,   434,   435,   437,   438,   463,   464,
            465,   466,   467,   468,   469,   470,   472,   473,   495,   496,
            497,   498,   499,   500,   502,   527,   528,   529,   530,   531,
            532,   533,   558,   559,   560,   561,   562,   563,   564,   565,
            566,   591,   592,   593,   594,   595,   596,   597,   598,   599,
            600,   602,   603,   617,   618,   621,   623,   624,   625,   626,
            627,   628,   629,   630,   631,   632,   635,   641,   642,   643,
            655,   656,   657,   658,   659,   660,   661,   662,   663,   664,
            667,   668,   671,   686,   687,   688,   689,   690,   691,   692,
            693,   694,   695,   696,   697,   698,   699,   701,   703,   712,
            715,   718,   719,   720,   721,   722,   723,   724,   725,   726,
            727,   728,   729,   730,   731,   735,   736,   739,   740,   742,
            744,   749,   751,   752,   753,   754,   755,   756,   757,   758,
            759,   760,   761,   762,   763,   764,   782,   783,   784,   785,
            786,   787,   788,   789,   790,   791,   792,   793,   794,   795,
            796,   797,   798,   800,   808,   814,   815,   816,   817,   818,
            819,   820,   821,   822,   823,   846,   847,   848,   849,   850,
            851,   852,   853,   854,   860,   861,   879,   880,   881,   882,
            883,   884,   885,   886,   887,   910,   911,   912,   913,   914,
            915,   916,   917,   918,   919,   921,   943,   944,   945,   946,
            947,   948,   949,   950,   953,   974,   976,   977,   978,   979,
            980,   982,   983,   1007,  1008,  1009,  1010,  1011,  1012,  1013,
            1014,  1038,  1039,  1040,  1041,  1042,  1043,  1044,  1045,  1046,
            1047,  1048,  1049,  1050,
        ]
        # self.prefill_graph_seq_lens = list(range(272, 1100))
        # fmt: on
        self.num_deepstack_layers = 3
        _max_pf_sl = max(self.prefill_graph_seq_lens)
        prefill_phase1_hs_buf = torch.randn(1, _max_pf_sl, config.hidden_size)
        self.register_buffer("prefill_phase1_hs_buffer", prefill_phase1_hs_buf, persistent=False)
        prefill_deepstack_buf = torch.zeros(self.num_deepstack_layers, 1, _max_pf_sl, config.hidden_size)
        self.register_buffer("prefill_deepstack_buffer", prefill_deepstack_buf, persistent=False)
        prefill_hs_buf = torch.randn(1, _max_pf_sl, config.hidden_size)
        self.register_buffer("prefill_hs_buffer", prefill_hs_buf, persistent=False)
        prefill_pe_cos_buf = torch.empty(1, _max_pf_sl, config.head_dim // 2)
        self.register_buffer("prefill_pe_cos_buffer", prefill_pe_cos_buf, persistent=False)
        prefill_pe_sin_buf = torch.empty(1, _max_pf_sl, config.head_dim // 2)
        self.register_buffer("prefill_pe_sin_buffer", prefill_pe_sin_buf, persistent=False)
        self.prefill_phase1_graphs = {}  # {seq_len: (CUDAGraph, output_tensor_ref)}
        self.prefill_graphs = {}  # {seq_len: (CUDAGraph, output_tensor_ref)}
        self.prefill_graphs_captured = False
        self.prefill_phase1_cudagraph_enabled = _env_flag("PREFILL_PHASE1_CUDAGRAPH", "1")

        self.using_cudagraph = os.environ.get("USE_CUDAGRAPH", "1").lower() in ["1", "true", "yes"]
        self.layer_stats = maybe_create_layer_stats_collector(config.num_hidden_layers, config.hidden_size)
        if self.layer_stats is not None and self.using_cudagraph:
            print("[LayerStats] forcing USE_CUDAGRAPH=0 for calibration collection", flush=True)
            self.using_cudagraph = False
        self.decode_num_layers = int(os.environ.get("DECODE_NUM_LAYERS", str(config.num_hidden_layers)))
        if not (1 <= self.decode_num_layers <= config.num_hidden_layers):
            raise ValueError(f"DECODE_NUM_LAYERS must be in [1, {config.num_hidden_layers}], got {self.decode_num_layers}")
        self.decode_skip_layers = _parse_layer_set(os.environ.get("DECODE_SKIP_LAYERS", ""), config.num_hidden_layers)
        self.prefill_layer_skip_layers = _parse_layer_set(os.environ.get("PREFILL_LAYER_SKIP_LAYERS", ""), config.num_hidden_layers)
        if self.prefill_layer_skip_layers and not self.prefill_layer_skip_layers.issubset(self.decode_skip_layers):
            raise ValueError(
                "PREFILL_LAYER_SKIP_LAYERS must be a subset of DECODE_SKIP_LAYERS so skipped prefill KV is never read by decode; "
                f"prefill={sorted(self.prefill_layer_skip_layers)} decode={sorted(self.decode_skip_layers)}"
            )
        prefill_skip_kv_enabled = os.environ.get("PREFILL_SKIP_KV_WRITE", "1").lower() in ["1", "true", "yes"]
        prefill_skip_kv_layers_env = os.environ.get("PREFILL_SKIP_KV_WRITE_LAYERS")
        if not prefill_skip_kv_enabled:
            self.prefill_skip_kv_write_layers = set()
        elif prefill_skip_kv_layers_env is None:
            self.prefill_skip_kv_write_layers = set(self.decode_skip_layers) | set(self.prefill_layer_skip_layers)
        else:
            self.prefill_skip_kv_write_layers = _parse_layer_set(prefill_skip_kv_layers_env, config.num_hidden_layers)
        self.decode_step_index = -1
        self.decode_skip_rms_scale = float(os.environ.get("DECODE_SKIP_RMS_SCALE", "1.0"))
        self.decode_skip_target_rms = float(os.environ.get("DECODE_SKIP_TARGET_RMS", "0.0"))
        self.decode_skip_scale_min = float(os.environ.get("DECODE_SKIP_SCALE_MIN", "0.0"))
        self.decode_skip_scale_max = float(os.environ.get("DECODE_SKIP_SCALE_MAX", "0.0"))
        self.decode_skip_scale_before_layer = int(os.environ.get("DECODE_SKIP_SCALE_BEFORE_LAYER", str(max(self.decode_skip_layers) + 1 if self.decode_skip_layers else -1)))
        decode_skip_scale_vector_path = os.environ.get("DECODE_SKIP_SCALE_VECTOR", "").strip()
        decode_skip_scale_specs_env = os.environ.get("DECODE_SKIP_SCALE_SPECS", "").strip()
        decode_skip_steering_specs_env = os.environ.get("DECODE_SKIP_STEERING_SPECS", "").strip()
        decode_skip_lowrank_specs_env = os.environ.get("DECODE_SKIP_LOWRANK_SPECS", "").strip()
        decode_skip_source_lowrank_specs_env = os.environ.get("DECODE_SKIP_SOURCE_LOWRANK_SPECS", "").strip()
        decode_skip_patch_specs_env = os.environ.get("DECODE_SKIP_PATCH_SPECS", "").strip()
        self.decode_skip_fused_bridge = os.environ.get("DECODE_SKIP_FUSED_BRIDGE", "1").lower() in ["1", "true", "yes"]
        prefill_layer_skip_scale_specs_env = os.environ.get("PREFILL_LAYER_SKIP_SCALE_SPECS", "").strip()
        prefill_layer_skip_lowrank_specs_env = os.environ.get("PREFILL_LAYER_SKIP_LOWRANK_SPECS", "").strip()
        prefill_layer_skip_steering_specs_env = os.environ.get("PREFILL_LAYER_SKIP_STEERING_SPECS", "").strip()
        prefill_layer_skip_patch_specs_env = os.environ.get("PREFILL_LAYER_SKIP_PATCH_SPECS", "").strip()
        raw_scale_specs: list[tuple[int, str]] = []
        if decode_skip_scale_specs_env:
            raw_scale_specs = _parse_decode_skip_scale_specs(decode_skip_scale_specs_env, config.num_hidden_layers)
        elif decode_skip_scale_vector_path:
            raw_scale_specs = [(self.decode_skip_scale_before_layer, decode_skip_scale_vector_path)]

        self.decode_skip_scale_specs: list[dict[str, object]] = []
        for spec_idx, (before_layer, path) in enumerate(raw_scale_specs):
            scale_vector, bias_vector = _load_decode_skip_scale_payload(path, config.hidden_size, "DECODE_SKIP_SCALE_SPECS")
            scale_name = f"decode_skip_scale_vector_{spec_idx}"
            bias_name = f"decode_skip_bias_vector_{spec_idx}"
            self.register_buffer(scale_name, scale_vector, persistent=False)
            self.register_buffer(bias_name, bias_vector, persistent=False)
            self.decode_skip_scale_specs.append(
                {
                    "before_layer": before_layer,
                    "path": path,
                    "scale_name": scale_name,
                    "bias_name": bias_name,
                    "scale_cpu": scale_vector.clone(),
                    "bias_cpu": bias_vector.clone() if bias_vector.numel() > 0 else None,
                }
            )
        self.decode_skip_vector_ready = not self.decode_skip_scale_specs
        raw_steering_specs = _parse_decode_skip_steering_specs(decode_skip_steering_specs_env, config.num_hidden_layers) if decode_skip_steering_specs_env else []
        self.decode_skip_steering_specs: list[dict[str, object]] = []
        for spec_idx, raw_spec in enumerate(raw_steering_specs):
            steering_vector = _load_decode_skip_steering_payload(str(raw_spec["path"]), config.hidden_size)
            steering_name = f"decode_skip_steering_vector_{spec_idx}"
            alpha_name = f"decode_skip_steering_alpha_{spec_idx}"
            self.register_buffer(steering_name, steering_vector, persistent=False)
            self.register_buffer(alpha_name, torch.tensor(0.0, dtype=torch.float32), persistent=False)
            spec = dict(raw_spec)
            spec["steering_name"] = steering_name
            spec["alpha_name"] = alpha_name
            spec["steering_cpu"] = steering_vector.clone()
            self.decode_skip_steering_specs.append(spec)
        self.decode_skip_steering_ready = not self.decode_skip_steering_specs
        raw_lowrank_specs = _parse_decode_skip_lowrank_specs(decode_skip_lowrank_specs_env, config.num_hidden_layers) if decode_skip_lowrank_specs_env else []
        self.decode_skip_lowrank_specs: list[dict[str, object]] = []
        for spec_idx, raw_spec in enumerate(raw_lowrank_specs):
            payload = _load_decode_skip_lowrank_payload(str(raw_spec["path"]), config.hidden_size)
            mean_in_name = f"decode_skip_lowrank_mean_in_{spec_idx}"
            mean_out_name = f"decode_skip_lowrank_mean_out_{spec_idx}"
            low_a_name = f"decode_skip_lowrank_a_{spec_idx}"
            low_b_name = f"decode_skip_lowrank_b_{spec_idx}"
            rms_w_name = f"decode_skip_lowrank_rms_w_{spec_idx}"
            rms_b_name = f"decode_skip_lowrank_rms_b_{spec_idx}"
            self.register_buffer(mean_in_name, payload["mean_in"], persistent=False)
            self.register_buffer(mean_out_name, payload["mean_out"], persistent=False)
            self.register_buffer(low_a_name, payload["low_a"], persistent=False)
            self.register_buffer(low_b_name, payload["low_b"], persistent=False)
            self.register_buffer(rms_w_name, payload["rms_reg_w"], persistent=False)
            self.register_buffer(rms_b_name, payload["rms_reg_b"], persistent=False)
            spec = dict(raw_spec)
            spec.update(
                {
                    "mode": payload["mode"],
                    "rank": payload["rank"],
                    "mean_in_name": mean_in_name,
                    "mean_out_name": mean_out_name,
                    "low_a_name": low_a_name,
                    "low_b_name": low_b_name,
                    "rms_w_name": rms_w_name,
                    "rms_b_name": rms_b_name,
                    "mean_in_cpu": payload["mean_in"].clone(),
                    "mean_out_cpu": payload["mean_out"].clone(),
                    "low_a_cpu": payload["low_a"].clone(),
                    "low_b_cpu": payload["low_b"].clone(),
                    "rms_w_cpu": payload["rms_reg_w"].clone(),
                    "rms_b_cpu": payload["rms_reg_b"].clone(),
                }
            )
            self.decode_skip_lowrank_specs.append(spec)
        self.decode_skip_lowrank_ready = not self.decode_skip_lowrank_specs
        self.register_buffer("decode_skip_fused_bridge_tmp", torch.empty(0, dtype=torch.float32), persistent=False)
        raw_source_lowrank_specs = (
            _parse_decode_skip_source_lowrank_specs(decode_skip_source_lowrank_specs_env, config.num_hidden_layers)
            if decode_skip_source_lowrank_specs_env
            else []
        )
        self.decode_skip_source_lowrank_specs: list[dict[str, object]] = []
        source_cache_names: dict[int, str] = {}
        for spec_idx, raw_spec in enumerate(raw_source_lowrank_specs):
            payload = _load_decode_skip_source_lowrank_payload(str(raw_spec["path"]), config.hidden_size)
            source_layer = int(raw_spec["source_layer"])
            if source_layer not in source_cache_names:
                cache_name = f"decode_skip_source_cache_{source_layer}"
                self.register_buffer(cache_name, torch.empty(0, dtype=torch.float16), persistent=False)
                source_cache_names[source_layer] = cache_name
            else:
                cache_name = source_cache_names[source_layer]
            mean_in_name = f"decode_skip_source_lowrank_mean_in_{spec_idx}"
            mean_delta_name = f"decode_skip_source_lowrank_mean_delta_{spec_idx}"
            low_a_name = f"decode_skip_source_lowrank_a_{spec_idx}"
            low_b_name = f"decode_skip_source_lowrank_b_{spec_idx}"
            rms_w_name = f"decode_skip_source_lowrank_rms_w_{spec_idx}"
            rms_b_name = f"decode_skip_source_lowrank_rms_b_{spec_idx}"
            self.register_buffer(mean_in_name, payload["mean_in"], persistent=False)
            self.register_buffer(mean_delta_name, payload["mean_delta"], persistent=False)
            self.register_buffer(low_a_name, payload["low_a"], persistent=False)
            self.register_buffer(low_b_name, payload["low_b"], persistent=False)
            self.register_buffer(rms_w_name, payload["rms_reg_w"], persistent=False)
            self.register_buffer(rms_b_name, payload["rms_reg_b"], persistent=False)
            spec = dict(raw_spec)
            spec.update(
                {
                    "mode": payload["mode"],
                    "rank": payload["rank"],
                    "cache_name": cache_name,
                    "mean_in_name": mean_in_name,
                    "mean_delta_name": mean_delta_name,
                    "low_a_name": low_a_name,
                    "low_b_name": low_b_name,
                    "rms_w_name": rms_w_name,
                    "rms_b_name": rms_b_name,
                    "mean_in_cpu": payload["mean_in"].clone(),
                    "mean_delta_cpu": payload["mean_delta"].clone(),
                    "low_a_cpu": payload["low_a"].clone(),
                    "low_b_cpu": payload["low_b"].clone(),
                    "rms_w_cpu": payload["rms_reg_w"].clone(),
                    "rms_b_cpu": payload["rms_reg_b"].clone(),
                }
            )
            self.decode_skip_source_lowrank_specs.append(spec)
        self.decode_skip_source_lowrank_ready = not self.decode_skip_source_lowrank_specs
        raw_decode_patch_specs = _parse_prefill_layer_skip_patch_specs(decode_skip_patch_specs_env, config.num_hidden_layers) if decode_skip_patch_specs_env else []
        self.decode_skip_patch_specs: list[dict[str, object]] = []
        for spec_idx, raw_spec in enumerate(raw_decode_patch_specs):
            before_layer = int(raw_spec["before_layer"])
            path = str(raw_spec["path"])
            payload = _load_prefill_layer_skip_patch_payload(path, config.hidden_size)
            scale_name = f"decode_skip_patch_scale_{spec_idx}"
            bias_name = f"decode_skip_patch_bias_{spec_idx}"
            self.register_buffer(scale_name, payload["scale_vector"], persistent=False)
            self.register_buffer(bias_name, payload["bias_vector"], persistent=False)
            self.decode_skip_patch_specs.append(
                {
                    "before_layer": before_layer,
                    "path": path,
                    "alpha": raw_spec["alpha"],
                    "mode": payload["mode"],
                    "scale_name": scale_name,
                    "bias_name": bias_name,
                    "scale_cpu": payload["scale_vector"].clone(),
                    "bias_cpu": payload["bias_vector"].clone() if payload["bias_vector"].numel() > 0 else None,
                }
            )
        self.decode_skip_patch_ready = not self.decode_skip_patch_specs
        raw_prefill_scale_specs = _parse_decode_skip_scale_specs(prefill_layer_skip_scale_specs_env, config.num_hidden_layers) if prefill_layer_skip_scale_specs_env else []
        self.prefill_layer_skip_scale_specs: list[dict[str, object]] = []
        for spec_idx, (before_layer, path) in enumerate(raw_prefill_scale_specs):
            scale_vector, bias_vector = _load_decode_skip_scale_payload(path, config.hidden_size, "PREFILL_LAYER_SKIP_SCALE_SPECS")
            scale_name = f"prefill_layer_skip_scale_vector_{spec_idx}"
            bias_name = f"prefill_layer_skip_bias_vector_{spec_idx}"
            self.register_buffer(scale_name, scale_vector, persistent=False)
            self.register_buffer(bias_name, bias_vector, persistent=False)
            self.prefill_layer_skip_scale_specs.append(
                {
                    "before_layer": before_layer,
                    "path": path,
                    "scale_name": scale_name,
                    "bias_name": bias_name,
                    "scale_cpu": scale_vector.clone(),
                    "bias_cpu": bias_vector.clone() if bias_vector.numel() > 0 else None,
                }
            )
        self.prefill_layer_skip_vector_ready = not self.prefill_layer_skip_scale_specs
        raw_prefill_patch_specs = _parse_prefill_layer_skip_patch_specs(prefill_layer_skip_patch_specs_env, config.num_hidden_layers) if prefill_layer_skip_patch_specs_env else []
        self.prefill_layer_skip_patch_specs: list[dict[str, object]] = []
        for spec_idx, raw_spec in enumerate(raw_prefill_patch_specs):
            before_layer = int(raw_spec["before_layer"])
            path = str(raw_spec["path"])
            payload = _load_prefill_layer_skip_patch_payload(path, config.hidden_size)
            scale_name = f"prefill_layer_skip_patch_scale_{spec_idx}"
            bias_name = f"prefill_layer_skip_patch_bias_{spec_idx}"
            self.register_buffer(scale_name, payload["scale_vector"], persistent=False)
            self.register_buffer(bias_name, payload["bias_vector"], persistent=False)
            alpha = raw_spec["alpha"]
            self.prefill_layer_skip_patch_specs.append(
                {
                    "before_layer": before_layer,
                    "path": path,
                    "alpha": alpha,
                    "mode": payload["mode"],
                    "scale_name": scale_name,
                    "bias_name": bias_name,
                    "scale_cpu": payload["scale_vector"].clone(),
                    "bias_cpu": payload["bias_vector"].clone() if payload["bias_vector"].numel() > 0 else None,
                }
            )
        self.prefill_layer_skip_patch_ready = not self.prefill_layer_skip_patch_specs
        raw_prefill_lowrank_specs = _parse_decode_skip_lowrank_specs(prefill_layer_skip_lowrank_specs_env, config.num_hidden_layers) if prefill_layer_skip_lowrank_specs_env else []
        self.prefill_layer_skip_lowrank_specs: list[dict[str, object]] = []
        for spec_idx, raw_spec in enumerate(raw_prefill_lowrank_specs):
            payload = _load_decode_skip_lowrank_payload(str(raw_spec["path"]), config.hidden_size)
            mean_in_name = f"prefill_layer_skip_lowrank_mean_in_{spec_idx}"
            mean_out_name = f"prefill_layer_skip_lowrank_mean_out_{spec_idx}"
            low_a_name = f"prefill_layer_skip_lowrank_a_{spec_idx}"
            low_b_name = f"prefill_layer_skip_lowrank_b_{spec_idx}"
            rms_w_name = f"prefill_layer_skip_lowrank_rms_w_{spec_idx}"
            rms_b_name = f"prefill_layer_skip_lowrank_rms_b_{spec_idx}"
            self.register_buffer(mean_in_name, payload["mean_in"], persistent=False)
            self.register_buffer(mean_out_name, payload["mean_out"], persistent=False)
            self.register_buffer(low_a_name, payload["low_a"], persistent=False)
            self.register_buffer(low_b_name, payload["low_b"], persistent=False)
            self.register_buffer(rms_w_name, payload["rms_reg_w"], persistent=False)
            self.register_buffer(rms_b_name, payload["rms_reg_b"], persistent=False)
            spec = dict(raw_spec)
            spec.update(
                {
                    "mode": payload["mode"],
                    "rank": payload["rank"],
                    "mean_in_name": mean_in_name,
                    "mean_out_name": mean_out_name,
                    "low_a_name": low_a_name,
                    "low_b_name": low_b_name,
                    "rms_w_name": rms_w_name,
                    "rms_b_name": rms_b_name,
                    "mean_in_cpu": payload["mean_in"].clone(),
                    "mean_out_cpu": payload["mean_out"].clone(),
                    "low_a_cpu": payload["low_a"].clone(),
                    "low_b_cpu": payload["low_b"].clone(),
                    "rms_w_cpu": payload["rms_reg_w"].clone(),
                    "rms_b_cpu": payload["rms_reg_b"].clone(),
                }
            )
            self.prefill_layer_skip_lowrank_specs.append(spec)
        self.prefill_layer_skip_lowrank_ready = not self.prefill_layer_skip_lowrank_specs
        raw_prefill_steering_specs = (
            _parse_decode_skip_steering_specs(prefill_layer_skip_steering_specs_env, config.num_hidden_layers)
            if prefill_layer_skip_steering_specs_env
            else []
        )
        self.prefill_layer_skip_steering_specs: list[dict[str, object]] = []
        for spec_idx, raw_spec in enumerate(raw_prefill_steering_specs):
            steering_vector = _load_decode_skip_steering_payload(str(raw_spec["path"]), config.hidden_size)
            steering_name = f"prefill_layer_skip_steering_vector_{spec_idx}"
            alpha_name = f"prefill_layer_skip_steering_alpha_{spec_idx}"
            self.register_buffer(steering_name, steering_vector, persistent=False)
            self.register_buffer(alpha_name, torch.tensor(float(raw_spec["alpha"]), dtype=torch.float32), persistent=False)
            spec = dict(raw_spec)
            spec["steering_name"] = steering_name
            spec["alpha_name"] = alpha_name
            spec["steering_cpu"] = steering_vector.clone()
            self.prefill_layer_skip_steering_specs.append(spec)
        self.prefill_layer_skip_steering_ready = not self.prefill_layer_skip_steering_specs
        if self.decode_skip_layers:
            vector_text = ", ".join(
                f"{spec['before_layer']}:{spec['path']} bias={'yes' if getattr(self, str(spec['bias_name'])).numel() > 0 else 'no'}"
                for spec in self.decode_skip_scale_specs
            )
            steering_text = ", ".join(
                f"{spec['before_layer']}:{spec['path']}:alpha={float(spec['alpha']):.4f}:"
                f"rms={int(bool(spec['rms_protect']))}:decay={int(spec['decay_steps'])}:"
                f"skip_first={int(bool(spec['skip_first_step']))}"
                for spec in self.decode_skip_steering_specs
            )
            lowrank_text = ", ".join(
                f"{spec['before_layer']}:{spec['path']}:mode={spec['mode']}:rank={spec['rank']}:rms_fix={int(bool(spec['rms_reg_fix']))}"
                for spec in self.decode_skip_lowrank_specs
            )
            source_lowrank_text = ", ".join(
                f"{spec['before_layer']}:{spec['source_layer']}:{spec['path']}:"
                f"mode={spec['mode']}:rank={spec['rank']}:rms_fix={int(bool(spec['rms_reg_fix']))}"
                for spec in self.decode_skip_source_lowrank_specs
            )
            patch_text = ", ".join(
                f"{spec['before_layer']}:{spec['path']}:mode={spec['mode']}:"
                f"alpha={1.0 if spec['alpha'] is None and spec['mode'] == 'hadamard_residual' else ('replace' if spec['alpha'] is None else float(spec['alpha']))}"
                for spec in self.decode_skip_patch_specs
            )
            print(
                "[DecodeSkip] "
                f"layers={sorted(self.decode_skip_layers)} "
                f"vector={vector_text or decode_skip_scale_vector_path or '<none>'} "
                f"steering={steering_text or '<none>'} "
                f"patch={patch_text or '<none>'} "
                f"lowrank={lowrank_text or '<none>'} "
                f"source_lowrank={source_lowrank_text or '<none>'} "
                f"rms_scale={self.decode_skip_rms_scale:.8f} "
                f"target_rms={self.decode_skip_target_rms:.8f} "
                f"scale_min={self.decode_skip_scale_min:.8f} "
                f"scale_max={self.decode_skip_scale_max:.8f} "
                f"scale_before_layer={self.decode_skip_scale_before_layer} "
                f"fused_bridge={int(self.decode_skip_fused_bridge)}",
                flush=True,
            )
        if self.prefill_layer_skip_layers:
            prefill_vector_text = ", ".join(
                f"{spec['before_layer']}:{spec['path']} bias={'yes' if getattr(self, str(spec['bias_name'])).numel() > 0 else 'no'}"
                for spec in self.prefill_layer_skip_scale_specs
            )
            prefill_lowrank_text = ", ".join(
                f"{spec['before_layer']}:{spec['path']}:mode={spec['mode']}:rank={spec['rank']}:rms_fix={int(bool(spec['rms_reg_fix']))}"
                for spec in self.prefill_layer_skip_lowrank_specs
            )
            prefill_steering_text = ", ".join(
                f"{spec['before_layer']}:{spec['path']}:alpha={float(spec['alpha']):.4f}:rms={int(bool(spec['rms_protect']))}"
                for spec in self.prefill_layer_skip_steering_specs
            )
            prefill_patch_text = ", ".join(
                f"{spec['before_layer']}:{spec['path']}:mode={spec['mode']}:"
                f"alpha={'replace' if spec['alpha'] is None else float(spec['alpha'])}"
                for spec in self.prefill_layer_skip_patch_specs
            )
            print(
                "[PrefillLayerSkip] "
                f"layers={sorted(self.prefill_layer_skip_layers)} "
                f"vector={prefill_vector_text or '<none>'} "
                f"patch={prefill_patch_text or '<none>'} "
                f"lowrank={prefill_lowrank_text or '<none>'} "
                f"steering={prefill_steering_text or '<none>'}",
                flush=True,
            )
        if self.prefill_skip_kv_write_layers:
            print(
                f"[PrefillKV] persistent KV write disabled for layers={sorted(self.prefill_skip_kv_write_layers)}",
                flush=True,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def _ensure_decode_skip_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.decode_skip_vector_ready:
            return
        for spec in self.decode_skip_scale_specs:
            scale_name = str(spec["scale_name"])
            bias_name = str(spec["bias_name"])
            setattr(self, scale_name, spec["scale_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            bias_cpu = spec["bias_cpu"]
            if bias_cpu is not None:
                setattr(self, bias_name, bias_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
        self.decode_skip_vector_ready = True

    def _ensure_decode_skip_steering_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.decode_skip_steering_ready:
            return
        for spec in self.decode_skip_steering_specs:
            steering_name = str(spec["steering_name"])
            alpha_name = str(spec["alpha_name"])
            setattr(self, steering_name, spec["steering_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, alpha_name, getattr(self, alpha_name).to(device=hidden_states.device))
        self.decode_skip_steering_ready = True

    def _ensure_decode_skip_lowrank_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.decode_skip_lowrank_ready:
            return
        for spec in self.decode_skip_lowrank_specs:
            setattr(self, str(spec["mean_in_name"]), spec["mean_in_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["mean_out_name"]), spec["mean_out_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["low_a_name"]), spec["low_a_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["low_b_name"]), spec["low_b_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["rms_w_name"]), spec["rms_w_cpu"].to(device=hidden_states.device))
            setattr(self, str(spec["rms_b_name"]), spec["rms_b_cpu"].to(device=hidden_states.device))
        self.decode_skip_lowrank_ready = True

    def _ensure_decode_skip_source_lowrank_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.decode_skip_source_lowrank_ready:
            return
        for spec in self.decode_skip_source_lowrank_specs:
            setattr(self, str(spec["mean_in_name"]), spec["mean_in_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["mean_delta_name"]), spec["mean_delta_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["low_a_name"]), spec["low_a_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["low_b_name"]), spec["low_b_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["rms_w_name"]), spec["rms_w_cpu"].to(device=hidden_states.device))
            setattr(self, str(spec["rms_b_name"]), spec["rms_b_cpu"].to(device=hidden_states.device))
        self.decode_skip_source_lowrank_ready = True

    def _ensure_decode_skip_patch_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.decode_skip_patch_ready:
            return
        for spec in self.decode_skip_patch_specs:
            scale_name = str(spec["scale_name"])
            bias_name = str(spec["bias_name"])
            setattr(self, scale_name, spec["scale_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            bias_cpu = spec["bias_cpu"]
            if bias_cpu is not None:
                setattr(self, bias_name, bias_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
        self.decode_skip_patch_ready = True

    def _ensure_prefill_layer_skip_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.prefill_layer_skip_vector_ready:
            return
        for spec in self.prefill_layer_skip_scale_specs:
            scale_name = str(spec["scale_name"])
            bias_name = str(spec["bias_name"])
            setattr(self, scale_name, spec["scale_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            bias_cpu = spec["bias_cpu"]
            if bias_cpu is not None:
                setattr(self, bias_name, bias_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
        self.prefill_layer_skip_vector_ready = True

    def _ensure_prefill_layer_skip_patch_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.prefill_layer_skip_patch_ready:
            return
        for spec in self.prefill_layer_skip_patch_specs:
            scale_name = str(spec["scale_name"])
            bias_name = str(spec["bias_name"])
            setattr(self, scale_name, spec["scale_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            bias_cpu = spec["bias_cpu"]
            if bias_cpu is not None:
                setattr(self, bias_name, bias_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
        self.prefill_layer_skip_patch_ready = True

    def _ensure_prefill_layer_skip_lowrank_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.prefill_layer_skip_lowrank_ready:
            return
        for spec in self.prefill_layer_skip_lowrank_specs:
            setattr(self, str(spec["mean_in_name"]), spec["mean_in_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["mean_out_name"]), spec["mean_out_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["low_a_name"]), spec["low_a_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["low_b_name"]), spec["low_b_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, str(spec["rms_w_name"]), spec["rms_w_cpu"].to(device=hidden_states.device))
            setattr(self, str(spec["rms_b_name"]), spec["rms_b_cpu"].to(device=hidden_states.device))
        self.prefill_layer_skip_lowrank_ready = True

    def _ensure_prefill_layer_skip_steering_vectors(self, hidden_states: torch.Tensor) -> None:
        if self.prefill_layer_skip_steering_ready:
            return
        for spec in self.prefill_layer_skip_steering_specs:
            steering_name = str(spec["steering_name"])
            alpha_name = str(spec["alpha_name"])
            setattr(self, steering_name, spec["steering_cpu"].to(device=hidden_states.device, dtype=hidden_states.dtype).contiguous())
            setattr(self, alpha_name, getattr(self, alpha_name).to(device=hidden_states.device))
        self.prefill_layer_skip_steering_ready = True

    @staticmethod
    def _per_token_rms(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)

    @staticmethod
    def _fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        n = orig_shape[-1]
        if n <= 0 or (n & (n - 1)) != 0:
            raise ValueError(f"FWHT last dimension must be a power of two, got {n}")
        y = x.float().reshape(-1, n)
        h = 1
        while h < n:
            y = y.reshape(-1, n // (2 * h), 2 * h)
            left = y[..., :h]
            right = y[..., h : 2 * h]
            y = torch.cat((left + right, left - right), dim=-1)
            h *= 2
        return y.reshape(orig_shape) * (1.0 / math.sqrt(n))

    def _update_decode_skip_steering_alphas(self) -> None:
        if not self.decode_skip_steering_specs:
            return
        step = self.decode_step_index
        for spec in self.decode_skip_steering_specs:
            alpha = float(spec["alpha"])
            if bool(spec["skip_first_step"]) and step <= 0:
                alpha = 0.0
            decay_steps = int(spec["decay_steps"])
            if decay_steps > 0 and alpha != 0.0:
                decay_pos = max(0, step - (1 if bool(spec["skip_first_step"]) else 0))
                alpha *= max(0.0, 1.0 - float(decay_pos) / float(decay_steps))
            getattr(self, str(spec["alpha_name"])).fill_(alpha)

    def _apply_decode_skip_steering(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        if not self.decode_skip_steering_specs:
            return
        for spec in self.decode_skip_steering_specs:
            if layer_idx != spec["before_layer"]:
                continue
            steering_vector = getattr(self, str(spec["steering_name"])).view(1, 1, -1)
            alpha = getattr(self, str(spec["alpha_name"])).to(dtype=hidden_states.dtype)
            if bool(spec["rms_protect"]):
                base_rms = self._per_token_rms(hidden_states)
                steered = hidden_states + steering_vector * alpha
                steered_rms = self._per_token_rms(steered).clamp_min(1e-6)
                hidden_states.copy_(steered * (base_rms / steered_rms).to(dtype=hidden_states.dtype))
            else:
                hidden_states.add_(steering_vector * alpha)

    def _has_decode_skip_lowrank_at_layer(self, layer_idx: int) -> bool:
        for spec in self.decode_skip_lowrank_specs:
            if layer_idx == spec["before_layer"]:
                return True
        return False

    def _has_decode_skip_patch_at_layer(self, layer_idx: int) -> bool:
        for spec in self.decode_skip_patch_specs:
            if layer_idx == spec["before_layer"]:
                return True
        return False

    def _decode_skip_scale_specs_at_layer(self, layer_idx: int) -> list[dict[str, object]]:
        return [spec for spec in self.decode_skip_scale_specs if layer_idx == spec["before_layer"]]

    def _decode_skip_lowrank_specs_at_layer(self, layer_idx: int) -> list[dict[str, object]]:
        return [spec for spec in self.decode_skip_lowrank_specs if layer_idx == spec["before_layer"]]

    def _ensure_decode_skip_fused_bridge_tmp(self, hidden_states: torch.Tensor, rank: int) -> None:
        if self.decode_skip_fused_bridge_tmp.numel() >= rank and self.decode_skip_fused_bridge_tmp.device == hidden_states.device:
            return
        self.decode_skip_fused_bridge_tmp = torch.empty(rank, device=hidden_states.device, dtype=torch.float32)

    def _try_apply_decode_skip_fused_bridge(self, layer_idx: int, hidden_states: torch.Tensor) -> bool:
        if not self.decode_skip_fused_bridge:
            return False
        if self._has_decode_skip_patch_at_layer(layer_idx):
            return False
        if not hidden_states.is_cuda or hidden_states.dtype != torch.float16 or hidden_states.numel() != 2048 or not hidden_states.is_contiguous():
            return False

        scale_specs = self._decode_skip_scale_specs_at_layer(layer_idx)
        lowrank_specs = self._decode_skip_lowrank_specs_at_layer(layer_idx)
        if len(scale_specs) != 1 or len(lowrank_specs) != 1:
            return False

        scale_spec = scale_specs[0]
        lowrank_spec = lowrank_specs[0]
        if str(lowrank_spec["mode"]) != "ls_residual_lowrank":
            return False
        if int(lowrank_spec["rank"]) != 128 or bool(lowrank_spec["rms_reg_fix"]):
            return False

        scale_vector = getattr(self, str(scale_spec["scale_name"]))
        bias_vector = getattr(self, str(scale_spec["bias_name"]))
        if bias_vector.numel() != 2048:
            return False

        mean_in = getattr(self, str(lowrank_spec["mean_in_name"]))
        mean_out = getattr(self, str(lowrank_spec["mean_out_name"]))
        low_a = getattr(self, str(lowrank_spec["low_a_name"]))
        low_b = getattr(self, str(lowrank_spec["low_b_name"]))
        if low_a.shape != (2048, 128) or low_b.shape != (128, 2048):
            return False

        self._ensure_decode_skip_fused_bridge_tmp(hidden_states, 128)
        fused_decode_skip_bridge_(
            hidden_states,
            self.decode_skip_fused_bridge_tmp,
            scale_vector,
            bias_vector,
            mean_in,
            mean_out,
            low_a,
            low_b,
        )
        return True

    def _apply_decode_skip_lowrank(self, layer_idx: int, source_states: torch.Tensor, hidden_states: torch.Tensor) -> None:
        if not self.decode_skip_lowrank_specs:
            return
        for spec in self.decode_skip_lowrank_specs:
            if layer_idx != spec["before_layer"]:
                continue
            mean_in = getattr(self, str(spec["mean_in_name"])).view(1, 1, -1)
            mean_out = getattr(self, str(spec["mean_out_name"])).view(1, 1, -1)
            low_a = getattr(self, str(spec["low_a_name"]))
            low_b = getattr(self, str(spec["low_b_name"]))

            centered = source_states - mean_in
            lowrank_hidden = torch.matmul(centered, low_a)
            lowrank_delta = torch.matmul(lowrank_hidden, low_b).contiguous()
            lowrank_delta.add_(mean_out)

            if str(spec["mode"]) == "ls_residual_lowrank":
                updated = hidden_states + lowrank_delta
            else:
                updated = lowrank_delta

            if bool(spec["rms_reg_fix"]):
                source_rms = self._per_token_rms(source_states).clamp_min(1e-6)
                target_rms = source_rms * getattr(self, str(spec["rms_w_name"])).to(dtype=source_rms.dtype)
                target_rms = (target_rms + getattr(self, str(spec["rms_b_name"])).to(dtype=source_rms.dtype)).clamp_min(1e-6)
                updated_rms = self._per_token_rms(updated).clamp_min(1e-6)
                updated = updated * (target_rms / updated_rms).to(dtype=updated.dtype)

            hidden_states.copy_(updated)

    def _maybe_store_decode_skip_source_cache(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        if not self.decode_skip_source_lowrank_specs:
            return
        for spec in self.decode_skip_source_lowrank_specs:
            if layer_idx != spec["source_layer"]:
                continue
            cache_name = str(spec["cache_name"])
            cache = getattr(self, cache_name)
            if cache.shape != hidden_states.shape or cache.device != hidden_states.device or cache.dtype != hidden_states.dtype:
                cache = torch.empty_like(hidden_states)
                setattr(self, cache_name, cache)
            cache.copy_(hidden_states)

    def _apply_decode_skip_source_lowrank(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        if not self.decode_skip_source_lowrank_specs:
            return
        for spec in self.decode_skip_source_lowrank_specs:
            if layer_idx != spec["before_layer"]:
                continue
            source_states = getattr(self, str(spec["cache_name"]))
            if source_states.numel() == 0:
                raise RuntimeError(
                    "DECODE_SKIP_SOURCE_LOWRANK_SPECS source cache is empty; "
                    f"source_layer={spec['source_layer']} before_layer={spec['before_layer']}"
                )
            mean_in = getattr(self, str(spec["mean_in_name"])).view(1, 1, -1)
            mean_delta = getattr(self, str(spec["mean_delta_name"])).view(1, 1, -1)
            low_a = getattr(self, str(spec["low_a_name"]))
            low_b = getattr(self, str(spec["low_b_name"]))

            centered = source_states - mean_in
            lowrank_hidden = torch.matmul(centered, low_a)
            lowrank_delta = torch.matmul(lowrank_hidden, low_b).contiguous()
            lowrank_delta.add_(mean_delta)
            updated = hidden_states + lowrank_delta

            if bool(spec["rms_reg_fix"]):
                source_rms = self._per_token_rms(source_states).clamp_min(1e-6)
                target_rms = source_rms * getattr(self, str(spec["rms_w_name"])).to(dtype=source_rms.dtype)
                target_rms = (target_rms + getattr(self, str(spec["rms_b_name"])).to(dtype=source_rms.dtype)).clamp_min(1e-6)
                updated_rms = self._per_token_rms(updated).clamp_min(1e-6)
                updated = updated * (target_rms / updated_rms).to(dtype=updated.dtype)

            hidden_states.copy_(updated)

    def _apply_decode_skip_patch(self, layer_idx: int, source_states: torch.Tensor, hidden_states: torch.Tensor) -> None:
        if not self.decode_skip_patch_specs:
            return
        for spec in self.decode_skip_patch_specs:
            if layer_idx != spec["before_layer"]:
                continue
            scale_vector = getattr(self, str(spec["scale_name"])).view(1, 1, -1)
            bias_vector = getattr(self, str(spec["bias_name"]))
            patch_hidden = self._fwht_last_dim(source_states)
            patch_hidden = patch_hidden * scale_vector
            if bias_vector.numel() > 0:
                patch_hidden = patch_hidden + bias_vector.view(1, 1, -1)
            patched = self._fwht_last_dim(patch_hidden).to(dtype=hidden_states.dtype)
            alpha = spec["alpha"]
            if str(spec["mode"]) == "hadamard_residual":
                hidden_states.add_(patched, alpha=1.0 if alpha is None else float(alpha))
            elif alpha is None:
                hidden_states.copy_(patched)
            else:
                hidden_states.add_((patched - source_states).to(dtype=hidden_states.dtype), alpha=float(alpha))

    def _has_prefill_layer_skip_lowrank_at_layer(self, layer_idx: int) -> bool:
        for spec in self.prefill_layer_skip_lowrank_specs:
            if layer_idx == spec["before_layer"]:
                return True
        return False

    def _apply_prefill_layer_skip_scale(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        if not self.prefill_layer_skip_scale_specs:
            return
        for spec in self.prefill_layer_skip_scale_specs:
            if layer_idx != spec["before_layer"]:
                continue
            scale_vector = getattr(self, str(spec["scale_name"]))
            bias_vector = getattr(self, str(spec["bias_name"]))
            hidden_states.mul_(scale_vector.view(1, 1, -1))
            if bias_vector.numel() > 0:
                hidden_states.add_(bias_vector.view(1, 1, -1))

    def _has_prefill_layer_skip_patch_at_layer(self, layer_idx: int) -> bool:
        for spec in self.prefill_layer_skip_patch_specs:
            if layer_idx == spec["before_layer"]:
                return True
        return False

    def _apply_prefill_layer_skip_patch(self, layer_idx: int, source_states: torch.Tensor, hidden_states: torch.Tensor) -> None:
        if not self.prefill_layer_skip_patch_specs:
            return
        for spec in self.prefill_layer_skip_patch_specs:
            if layer_idx != spec["before_layer"]:
                continue
            scale_vector = getattr(self, str(spec["scale_name"])).view(1, 1, -1)
            bias_vector = getattr(self, str(spec["bias_name"]))
            patch_hidden = self._fwht_last_dim(source_states)
            patch_hidden = patch_hidden * scale_vector
            if bias_vector.numel() > 0:
                patch_hidden = patch_hidden + bias_vector.view(1, 1, -1)
            patched = self._fwht_last_dim(patch_hidden).to(dtype=hidden_states.dtype)
            alpha = spec["alpha"]
            if str(spec["mode"]) == "hadamard_residual":
                hidden_states.add_(patched, alpha=1.0 if alpha is None else float(alpha))
            elif alpha is None:
                hidden_states.copy_(patched)
            else:
                hidden_states.add_((patched - source_states).to(dtype=hidden_states.dtype), alpha=float(alpha))

    def _apply_prefill_layer_skip_lowrank(self, layer_idx: int, source_states: torch.Tensor, hidden_states: torch.Tensor) -> None:
        if not self.prefill_layer_skip_lowrank_specs:
            return
        for spec in self.prefill_layer_skip_lowrank_specs:
            if layer_idx != spec["before_layer"]:
                continue
            mean_in = getattr(self, str(spec["mean_in_name"])).view(1, 1, -1)
            mean_out = getattr(self, str(spec["mean_out_name"])).view(1, 1, -1)
            low_a = getattr(self, str(spec["low_a_name"]))
            low_b = getattr(self, str(spec["low_b_name"]))

            centered = source_states - mean_in
            lowrank_hidden = torch.matmul(centered, low_a)
            lowrank_delta = torch.matmul(lowrank_hidden, low_b).contiguous()
            lowrank_delta.add_(mean_out)

            if str(spec["mode"]) == "ls_residual_lowrank":
                updated = hidden_states + lowrank_delta
            else:
                updated = lowrank_delta

            if bool(spec["rms_reg_fix"]):
                source_rms = self._per_token_rms(source_states).clamp_min(1e-6)
                target_rms = source_rms * getattr(self, str(spec["rms_w_name"])).to(dtype=source_rms.dtype)
                target_rms = (target_rms + getattr(self, str(spec["rms_b_name"])).to(dtype=source_rms.dtype)).clamp_min(1e-6)
                updated_rms = self._per_token_rms(updated).clamp_min(1e-6)
                updated = updated * (target_rms / updated_rms).to(dtype=updated.dtype)

            hidden_states.copy_(updated)

    def _apply_prefill_layer_skip_steering(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        if not self.prefill_layer_skip_steering_specs:
            return
        for spec in self.prefill_layer_skip_steering_specs:
            if layer_idx != spec["before_layer"]:
                continue
            steering_vector = getattr(self, str(spec["steering_name"])).view(1, 1, -1)
            alpha = getattr(self, str(spec["alpha_name"])).to(dtype=hidden_states.dtype)
            if bool(spec["rms_protect"]):
                base_rms = self._per_token_rms(hidden_states)
                steered = hidden_states + steering_vector * alpha
                steered_rms = self._per_token_rms(steered).clamp_min(1e-6)
                hidden_states.copy_(steered * (base_rms / steered_rms).to(dtype=hidden_states.dtype))
            else:
                hidden_states.add_(steering_vector * alpha)

    def _prefill_phase2_layers(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cg_bucket_size: int | None = None,
        is_warmup: bool = False,
        layer_stats_collector=None,
    ) -> torch.Tensor:
        self._ensure_prefill_layer_skip_vectors(hidden_states)
        self._ensure_prefill_layer_skip_patch_vectors(hidden_states)
        self._ensure_prefill_layer_skip_lowrank_vectors(hidden_states)
        self._ensure_prefill_layer_skip_steering_vectors(hidden_states)

        for layer_idx in range(self.num_deepstack_layers, len(self.layers)):
            needs_source_states = self._has_prefill_layer_skip_lowrank_at_layer(layer_idx) or self._has_prefill_layer_skip_patch_at_layer(layer_idx)
            source_states = hidden_states.clone() if needs_source_states else None
            self._apply_prefill_layer_skip_scale(layer_idx, hidden_states)
            if source_states is not None:
                self._apply_prefill_layer_skip_lowrank(layer_idx, source_states, hidden_states)
                self._apply_prefill_layer_skip_patch(layer_idx, source_states, hidden_states)
            self._apply_prefill_layer_skip_steering(layer_idx, hidden_states)
            if layer_idx in self.prefill_layer_skip_layers:
                continue
            hidden_states = self.layers[layer_idx](
                hidden_states,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                position_embeddings=position_embeddings,
                is_decoding=False,
                attn_k_cache=self.attn_kv_cache[layer_idx][0],
                attn_v_cache=self.attn_kv_cache[layer_idx][1],
                cache_seqlens=self.cache_seqlens,
                cg_bucket_size=cg_bucket_size,
                is_warmup=is_warmup,
                layer_stats_collector=layer_stats_collector,
                write_kv_cache=layer_idx not in self.prefill_skip_kv_write_layers,
            )
        return hidden_states

    def _decode_layers(self, hidden_states, position_ids, cg_bucket_size, is_warmup=False):
        self._ensure_decode_skip_vectors(hidden_states)
        self._ensure_decode_skip_steering_vectors(hidden_states)
        self._ensure_decode_skip_lowrank_vectors(hidden_states)
        self._ensure_decode_skip_source_lowrank_vectors(hidden_states)
        self._ensure_decode_skip_patch_vectors(hidden_states)
        position_embeddings = self.rotary_emb(position_ids, is_decoding=True, buffer4decode=self.pe_buffer4decode, device_type=hidden_states.device.type, dtype=hidden_states.dtype)

        # position_embeddings[0][:] = 0.
        # position_embeddings[1][:] = 0.
        collect_lmhead = self.layer_stats is not None and self.layer_stats.should_collect_lmhead()
        all_layers_hidden_states = [] if collect_lmhead else None

        for layer_idx in range(self.decode_num_layers):
            self._maybe_store_decode_skip_source_cache(layer_idx, hidden_states)
            fused_bridge_applied = self._try_apply_decode_skip_fused_bridge(layer_idx, hidden_states)
            if not fused_bridge_applied:
                needs_source_states = self._has_decode_skip_lowrank_at_layer(layer_idx) or self._has_decode_skip_patch_at_layer(layer_idx)
                lowrank_source_states = hidden_states.clone() if needs_source_states else None
                applied_vector_scale = False
                for spec in self.decode_skip_scale_specs:
                    if layer_idx == spec["before_layer"]:
                        scale_vector = getattr(self, str(spec["scale_name"]))
                        bias_vector = getattr(self, str(spec["bias_name"]))
                        hidden_states.mul_(scale_vector.view(1, 1, -1))
                        if bias_vector.numel() > 0:
                            hidden_states.add_(bias_vector.view(1, 1, -1))
                        applied_vector_scale = True
                if not applied_vector_scale and layer_idx == self.decode_skip_scale_before_layer:
                    if self.decode_skip_target_rms > 0.0:
                        current_rms = hidden_states.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
                        skip_scale = self.decode_skip_target_rms / current_rms
                        if self.decode_skip_scale_min > 0.0:
                            skip_scale = skip_scale.clamp_min(self.decode_skip_scale_min)
                        if self.decode_skip_scale_max > 0.0:
                            skip_scale = skip_scale.clamp_max(self.decode_skip_scale_max)
                        hidden_states.mul_(skip_scale.to(dtype=hidden_states.dtype))
                    elif self.decode_skip_rms_scale != 1.0:
                        hidden_states.mul_(self.decode_skip_rms_scale)
                if lowrank_source_states is not None:
                    self._apply_decode_skip_lowrank(layer_idx, lowrank_source_states, hidden_states)
                    self._apply_decode_skip_patch(layer_idx, lowrank_source_states, hidden_states)
            self._apply_decode_skip_source_lowrank(layer_idx, hidden_states)
            self._apply_decode_skip_steering(layer_idx, hidden_states)
            if layer_idx in self.decode_skip_layers:
                if all_layers_hidden_states is not None:
                    all_layers_hidden_states.append(hidden_states.detach().clone())
                continue
            decoder_layer = self.layers[layer_idx]
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                position_embeddings=position_embeddings,
                is_decoding=True,
                attn_k_cache=self.attn_kv_cache[layer_idx][0],
                attn_v_cache=self.attn_kv_cache[layer_idx][1],
                cache_seqlens=self.cache_seqlens,
                cg_bucket_size=cg_bucket_size,
                is_warmup=is_warmup,
                decode_attn_norm_buffer=self.decode_attn_norm_buffers[layer_idx],
                decode_attn_output_buffer=self.decode_attn_output_buffers[layer_idx],
                decode_mlp_norm_buffer=self.decode_mlp_norm_buffers[layer_idx],
                layer_stats_collector=self.layer_stats,
            )
            if all_layers_hidden_states is not None:
                all_layers_hidden_states.append(hidden_states.detach().clone())

        if all_layers_hidden_states is not None:
            all_layers_hidden_states = tuple(all_layers_hidden_states)
        return hidden_states, all_layers_hidden_states

    def _capture_all_decode_graphs(self):
        if self.is_decode_graphs_captured:
            return

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        mempool = None

        with torch.cuda.stream(s):
            for bucket_size in self.bucket_upper_bounds:
                self.cache_seqlens.fill_(bucket_size - 1)

                for _ in range(3):
                    hidden_states, _ = self._decode_layers(self.hidden_states_buffer, self.position_ids_buffer, bucket_size, is_warmup=True)
                    self.hidden_states_buffer.copy_(hidden_states)

                g = torch.cuda.CUDAGraph()

                with torch.cuda.graph(g, pool=mempool):
                    hidden_states, all_layers_hidden_states = self._decode_layers(
                        self.hidden_states_buffer,
                        self.position_ids_buffer,
                        bucket_size,
                        is_warmup=False,
                    )
                    self.hidden_states_buffer.copy_(hidden_states)
                if mempool is None:
                    mempool = g.pool()

                self.decode_graphs[bucket_size] = g
                self.decode_graph_outputs[bucket_size] = (hidden_states, all_layers_hidden_states)

        torch.cuda.current_stream().wait_stream(s)
        self.is_decode_graphs_captured = True

    def _decode_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        is_decoding: bool = False,
    ):

        timing_token = timing_start("decode_model")
        self.cache_seqlens += 1
        self.cache_seqlens_len += 1
        self.decode_step_index += 1
        self.position_ids_buffer.copy_(position_ids[1:])
        self._update_decode_skip_steering_alphas()
        if self.layer_stats is not None:
            self.layer_stats.begin_decode_step(self.cache_seqlens_len)

        if not self.is_decode_graphs_captured and self.using_cudagraph:
            self._capture_all_decode_graphs()

        target_bucket = self.bucket_upper_bounds[-1]
        for upper_bound in self.bucket_upper_bounds:
            if self.cache_seqlens_len <= upper_bound:
                target_bucket = upper_bound
                break

        try:
            if not self.using_cudagraph:
                hidden_states, all_layers_hidden_states = self._decode_layers(
                    inputs_embeds,
                    position_ids[1:],
                    target_bucket,
                )
            else:
                self.hidden_states_buffer.copy_(inputs_embeds)
                self.decode_graphs[target_bucket].replay()
                hidden_states, all_layers_hidden_states = self.decode_graph_outputs[target_bucket]
        finally:
            if self.layer_stats is not None:
                self.layer_stats.end_decode_step()

        timing_end(timing_token)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            all_layers_hidden_states=all_layers_hidden_states,
        )

    def _fill_prefill_deepstack_buffer(self, seq_len: int, deepstack_visual_embeds, visual_pos_idx) -> None:
        self.prefill_deepstack_buffer[:, :, :seq_len, :].zero_()
        if deepstack_visual_embeds is None or visual_pos_idx is None:
            return
        count = min(self.num_deepstack_layers, len(deepstack_visual_embeds))
        for layer_idx in range(count):
            layer_buf = self.prefill_deepstack_buffer[layer_idx, :, :seq_len, :]
            layer_buf[visual_pos_idx] = deepstack_visual_embeds[layer_idx]

    def _prefill_phase1_graph_forward(self, seq_len, is_warmup=False):
        """Graph-captured forward for layers [0, num_deepstack_layers).

        DeepStack updates are pre-scattered into dense buffers, so the graph only
        performs static full-buffer adds and contains no dynamic indexing.
        """
        hidden_states = self.prefill_phase1_hs_buffer[:, :seq_len, :]
        position_embeddings = (self.prefill_pe_cos_buffer[:, :seq_len, :], self.prefill_pe_sin_buffer[:, :seq_len, :])
        prefill_bucket = ((seq_len + 127) // 128) * 128 + 100000

        for layer_idx in range(self.num_deepstack_layers):
            hidden_states = self.layers[layer_idx](
                hidden_states,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
                position_embeddings=position_embeddings,
                is_decoding=False,
                attn_k_cache=self.attn_kv_cache[layer_idx][0],
                attn_v_cache=self.attn_kv_cache[layer_idx][1],
                cache_seqlens=self.cache_seqlens,
                cg_bucket_size=prefill_bucket,
                is_warmup=is_warmup,
                write_kv_cache=layer_idx not in self.prefill_skip_kv_write_layers,
            )
            hidden_states.add_(self.prefill_deepstack_buffer[layer_idx, :, :seq_len, :])
        return hidden_states

    def _prefill_graph_forward(self, seq_len, is_warmup=False):
        """Graph-captured forward for layers [num_deepstack_layers, ..., last] + norm.
        Reads from prefill static buffers, returns norm output (address fixed after capture)."""
        hidden_states = self.prefill_hs_buffer[:, :seq_len, :]
        position_embeddings = (self.prefill_pe_cos_buffer[:, :seq_len, :], self.prefill_pe_sin_buffer[:, :seq_len, :])
        prefill_bucket = ((seq_len + 127) // 128) * 128 + 100000  # +100000 区分 decode bucket # TODO, 不够优雅，删除

        hidden_states = self._prefill_phase2_layers(
            hidden_states,
            position_embeddings,
            cg_bucket_size=prefill_bucket,
            is_warmup=is_warmup,
        )

        # return self.norm(hidden_states, decode=False, inplace=False, cg_bucket_size=prefill_bucket, is_warmup=is_warmup)
        # 上面注释 self.norm 是因为 RMSNorm.weight 已经融合到 lm_head.weight 中了
        # 下面注释掉是因为 fused_1d_rmsnorm 可以视为一个线性操作，self.lm_head 没有 bias
        # return fused_1d_rmsnorm(hidden_states[:, -1:], weight=None, inplace=False, eps=self.norm.variance_epsilon)  # 这里不需要全量的，只需要最后一个 token 的
        return hidden_states

    def _capture_all_prefill_graphs(self):
        if self.prefill_graphs_captured:
            return

        phase1_text = "+phase1" if self.prefill_phase1_cudagraph_enabled else ""
        print(f"[PrefillCUDAGraph] pre-capturing {len(self.prefill_graph_seq_lens)} seq_lens{phase1_text}: {self.prefill_graph_seq_lens[0]}~{self.prefill_graph_seq_lens[-1]}")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        mempool = None

        with torch.cuda.stream(s):
            for seq_len in self.prefill_graph_seq_lens:
                if self.prefill_phase1_cudagraph_enabled:
                    for _ in range(3):
                        self._prefill_phase1_graph_forward(seq_len, is_warmup=True)

                    g_phase1 = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_phase1, pool=mempool):
                        out_phase1 = self._prefill_phase1_graph_forward(seq_len, is_warmup=False)
                    if mempool is None:
                        mempool = g_phase1.pool()

                    self.prefill_phase1_graphs[seq_len] = (g_phase1, out_phase1)

                # warmup: resolve autotuners + JIT
                for _ in range(3):
                    self._prefill_graph_forward(seq_len, is_warmup=True)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=mempool):
                    out = self._prefill_graph_forward(seq_len, is_warmup=False)
                if mempool is None:
                    mempool = g.pool()

                self.prefill_graphs[seq_len] = (g, out)

        torch.cuda.current_stream().wait_stream(s)
        self.prefill_graphs_captured = True
        print(f"[PrefillCUDAGraph] all {len(self.prefill_graphs)} graphs captured!")

    def _prefill_forward(self, hidden_states, position_ids, deepstack_visual_embeds, visual_pos_idx):

        # hidden_states: [1, 690, 2048]
        # position_ids: [4, 1, 690]
        # deepstack_visual_embeds [672, 2048]、[672, 2048]、[672, 2048]
        # visual_pos_idx: [672] [672]

        all_layers_hidden_states = None
        if not self.prefill_graphs_captured and self.using_cudagraph:
            self._capture_all_prefill_graphs()

        timing_prefill = timing_start("language_prefill")
        self.cache_seqlens[:] = 0
        seq_len = hidden_states.shape[1]
        self.decode_step_index = -1
        self._update_decode_skip_steering_alphas()
        if self.layer_stats is not None:
            self.layer_stats.maybe_begin_sample(seq_len)
        timing_rope = timing_start("prefill_rope")
        position_embeddings = self.rotary_emb(position_ids[1:], is_decoding=False, buffer4decode=None, device_type=hidden_states.device.type, dtype=hidden_states.dtype)
        timing_end(timing_rope)

        # --- Phase 1: first N layers + deepstack full-buffer adds ---
        if self.prefill_phase1_cudagraph_enabled and seq_len in self.prefill_phase1_graphs and self.using_cudagraph:
            timing_phase1 = timing_start("prefill_phase1_graph")
            cos, sin = position_embeddings
            self.prefill_phase1_hs_buffer[:, :seq_len, :].copy_(hidden_states)
            self.prefill_pe_cos_buffer[:, :seq_len, :].copy_(cos)
            self.prefill_pe_sin_buffer[:, :seq_len, :].copy_(sin)
            self._fill_prefill_deepstack_buffer(seq_len, deepstack_visual_embeds, visual_pos_idx)

            graph, output_ref = self.prefill_phase1_graphs[seq_len]
            graph.replay()
            hidden_states = output_ref
            timing_end(timing_phase1)
        else:
            timing_phase1 = timing_start("prefill_phase1_deepstack")
            for layer_idx in range(self.num_deepstack_layers):
                hidden_states = self.layers[layer_idx](
                    hidden_states,
                    position_ids=None,
                    attention_mask=None,
                    past_key_values=None,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                    is_decoding=False,
                    attn_k_cache=self.attn_kv_cache[layer_idx][0],
                    attn_v_cache=self.attn_kv_cache[layer_idx][1],
                    cache_seqlens=self.cache_seqlens,
                    layer_stats_collector=self.layer_stats,
                    write_kv_cache=layer_idx not in self.prefill_skip_kv_write_layers,
                )
                if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                    hidden_states[visual_pos_idx] += deepstack_visual_embeds[layer_idx]
            timing_end(timing_phase1)

        # --- Phase 2: remaining layers + norm ---
        if seq_len in self.prefill_graphs and self.using_cudagraph:
            timing_phase2 = timing_start("prefill_phase2_graph")
            # CUDA Graph path: copy to static buffers, replay, read output
            cos, sin = position_embeddings
            self.prefill_hs_buffer[:, :seq_len, :].copy_(hidden_states)
            self.prefill_pe_cos_buffer[:, :seq_len, :].copy_(cos)
            self.prefill_pe_sin_buffer[:, :seq_len, :].copy_(sin)

            graph, output_ref = self.prefill_graphs[seq_len]
            graph.replay()
            hidden_states = output_ref
            timing_end(timing_phase2)
        else:
            timing_phase2 = timing_start("prefill_phase2_eager")
            hidden_states = self._prefill_phase2_layers(
                hidden_states,
                position_embeddings,
                layer_stats_collector=self.layer_stats,
            )
            # hidden_states = self.norm(hidden_states, decode=False, inplace=False)
            # 上面注释 self.norm 是因为 RMSNorm.weight 已经融合到 lm_head.weight 中了
            # 下面注释掉是因为 fused_1d_rmsnorm 可以视为一个线性操作，self.lm_head 没有 bias
            # hidden_states = fused_1d_rmsnorm(hidden_states[:, -1:], weight=None, inplace=False, eps=self.norm.variance_epsilon)
            timing_end(timing_phase2)

        self.cache_seqlens_len = seq_len

        timing_end(timing_prefill)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            all_layers_hidden_states=all_layers_hidden_states,
        )

    def record_layer_stats_token(self, next_tokens, all_layers_hidden_states, final_hidden_states, lm_head_weight, token_id_map):
        if self.layer_stats is None:
            return
        self.layer_stats.record_generated_token(
            next_tokens=next_tokens,
            all_layers_hidden_states=all_layers_hidden_states,
            final_hidden_states=final_hidden_states,
            lm_head_weight=lm_head_weight,
            token_id_map=token_id_map,
        )

    def end_layer_stats_sample(self, generated_tokens: int | None = None):
        timing_sample_done(generated_tokens)
        if self.layer_stats is not None:
            self.layer_stats.end_sample(generated_tokens=generated_tokens)

    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        is_decoding: bool = False,
        # args for deepstack, ONLY FOR PREFILL
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> tuple | BaseModelOutputWithPast:

        if is_decoding:
            return self._decode_forward(position_ids=position_ids, inputs_embeds=inputs_embeds)

        visual_pos_idx = torch.nonzero(visual_pos_masks, as_tuple=True) if visual_pos_masks is not None else None
        return self._prefill_forward(inputs_embeds, position_ids, deepstack_visual_embeds, visual_pos_idx)
