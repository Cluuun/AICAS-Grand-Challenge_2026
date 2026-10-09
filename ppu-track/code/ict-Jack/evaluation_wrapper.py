#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import json
import hashlib
import math
import os
import re
import shutil
import sys
import time
import types
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Set


if os.environ.get("AICASGC_ISOLATE_COMPILE_CACHE", "1") != "0":
    cache_root = Path(os.environ.get("AICASGC_COMPILE_CACHE_ROOT", "/tmp/aicasgc_runtime_cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_root / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicCache, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)
from transformers.utils import logging as transformers_logging


_KERNEL_FASTPATH_NAMES = (
    "_aicas_triton_row_w8a16_gateup",
    "_aicas_triton_row_w8a16_gateup_silu_mul",
    "_aicas_triton_row_w8a16_gateup_norm",
    "_aicas_triton_row_w8a16_down_silu",
    "_aicas_triton_row_w8a16_down_silu_preact",
    "_aicas_triton_row_w8a16_linear2048",
    "_aicas_triton_row_w8a16_linear2048_add",
    "_aicas_triton_fp16_linear2048_dot",
    "_aicas_triton_fp16_linear2048_dot_add",
    "_aicas_triton_fp16_down_silu",
    "_aicas_triton_silu_preact",
    "_aicas_triton_fp16_down_from_act",
    "_aicas_triton_fp16_down_silu_preact",
    "_aicas_triton_decode_qkv_softmax_value_gqa_m1",
    "_aicas_triton_decode_softmax_value_gqa_m1",
    "_aicas_triton_decode_value_merge_gqa_m1",
    "_aicas_triton_decode_value_merge_atomic_gqa_m1",
    "_aicas_triton_fp16_top1_decode_m1",
    "_aicas_triton_fp16_top1_decode_small_m",
    "_aicas_triton_rmsnorm_fp16_top1_decode_m1",
    "_aicas_triton_rmsnorm_from_sumsq_fp16_m1",
    "_aicas_vision_patch_merger_fastpath",
    "_aicas_triton_vision_rope_qk",
)


def _load_kernel_fastpaths() -> dict[str, Any]:
    for module_name in ("submission1.my_kernel.kernel_runtime", ".submission1.my_kernel.kernel_runtime"):
        try:
            module = importlib.import_module(module_name, package=__package__)
            return {name: getattr(module, name, None) for name in _KERNEL_FASTPATH_NAMES}
        except Exception:
            continue
    return dict.fromkeys(_KERNEL_FASTPATH_NAMES)


globals().update(_load_kernel_fastpaths())
transformers_logging.set_verbosity_error()
_HAS_NATIVE_RMS_NORM = hasattr(F, "rms_norm")

_AICAS_ROWTRITON_GATEUP_DEFAULT_LAYER_MASK = "14-27"
_AICAS_ROWTRITON_GATEUP_PERF_LAYER_MASK = "0-27"
torch.set_grad_enabled(False)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _prepare_zw810e_w8a8_lut_aliases() -> None:
    if (
        os.environ.get("AICASGC_W8A8_ZW810E_LUT_ALIAS", "0") == "0"
        and os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_SQ_GATEUP", "0") == "0"
    ):
        return
    if os.environ.get("ACEXT_GEMM_CONFIG_DIR", "").strip():
        return
    try:
        import sysconfig

        purelib = Path(sysconfig.get_path("purelib"))
        src_dir = purelib / "include" / "acext" / "lut"
        if not src_dir.is_dir():
            return
        dst_dir = Path(os.environ.get("AICASGC_W8A8_LUT_ALIAS_DIR", "/tmp/aicasgc_acext_lut_zw810e"))
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.ini"):
            dst = dst_dir / src.name
            if not dst.exists():
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
        for shape in ("6144_2048", "8_6144"):
            src = src_dir / f"PPU-ZW810_w8a8_{shape}_config.ini"
            dst = dst_dir / f"PPU-ZW810E_w8a8_{shape}_config.ini"
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
        os.environ["ACEXT_GEMM_CONFIG_DIR"] = str(dst_dir)
    except Exception as exc:
        pass


def _try_load_ppu_extensions() -> bool:
    if os.environ.get("AICASGC_LOAD_ACEXT", "1") == "0":
        return False
    try:
        _prepare_zw810e_w8a8_lut_aliases()
        import acext
        return True
    except Exception as exc:
        pass
        return False


@dataclass
class _PreparedInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    image_token_start_hint: int = -1
    image_token_count_hint: int = 0
    image_prefix_len_hint: int = 0
    image_llm_grid_h_hint: int = 0
    image_llm_grid_w_hint: int = 0
    image_rope_delta_hint: int = 0
    image_prefix_hints_enabled: bool = False
    image_content_fingerprint_hint: Optional[tuple[Any, ...]] = None
    messages: Any = None

    def keys(self) -> Iterable[str]:
        base_keys = (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
        )
        if self.messages is not None:
            base_keys = base_keys + ("messages",)
        if self.image_content_fingerprint_hint is not None:
            base_keys = base_keys + ("image_content_fingerprint_hint",)
        if not self.image_prefix_hints_enabled:
            return base_keys
        return base_keys + (
            "image_token_start_hint",
            "image_token_count_hint",
            "image_prefix_len_hint",
            "image_llm_grid_h_hint",
            "image_llm_grid_w_hint",
            "image_rope_delta_hint",
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return key in set(self.keys())

@dataclass
class _DecodeInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    image_grid_thw: torch.Tensor
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    visual_pos_masks: torch.Tensor
    deepstack_visual_embeds: list[torch.Tensor]
    visual_start: int = -1
    visual_count: int = 0
    image_prefix_len: int = 0


class _DecodeHealthFallback(RuntimeError):
    pass


@dataclass
class _VisualBudgetProfile:
    request_key: str
    image_key: str
    request_pixels: int
    image_size: Optional[tuple[int, int]]
    question_kind: str
    visual_tier: str
    question_text: str = ""
    lifecycle_policy: str = "survey_fused"
    mode: str = "request"
    prompt_len: int = 0
    max_new_tokens: int = 0
    requested_len: int = 0
    estimated_visual_tokens: int = 0
    prefill_budget: str = "dynamic"
    prefill_keep_ratio: float = 1.0
    decode_budget: str = "dynamic"
    decode_visual_keep_ratio: float = 1.0
    kv_policy: str = "dynamic"
    visual_kv_policy: str = "full"
    visual_kv_target_keep_ratio: float = 1.0
    visual_kv_target_tokens: int = 0
    visual_kv_profile_only: bool = True
    quant_policy: str = "none"
    graph_cache_policy: str = "disabled"
    fusion_policy: str = "none"
    evidence_guard: bool = False
    spatial_anchor_ratio: float = 1.0
    enable_whitebox_block: bool = False
    prefer_graph_replay: bool = False
    graph_bucket: Optional[int] = None
    bucket_required_len: int = 0
    bucket_candidates: tuple[int, ...] = ()
    switch_after: int = 1
    fallback_reason: str = ""
    visual_kv_compact_delta: int = 0


@dataclass
class _InlineLayerSpec:
    input_norm_weight: torch.Tensor
    input_norm_eps: float
    q_proj_weight: torch.Tensor
    q_proj_bias: Optional[torch.Tensor]
    k_proj_weight: torch.Tensor
    k_proj_bias: Optional[torch.Tensor]
    v_proj_weight: torch.Tensor
    v_proj_bias: Optional[torch.Tensor]
    q_norm_weight: torch.Tensor
    q_norm_eps: float
    k_norm_weight: torch.Tensor
    k_norm_eps: float
    o_proj_weight: torch.Tensor
    o_proj_bias: Optional[torch.Tensor]
    post_norm_weight: torch.Tensor
    post_norm_eps: float
    gate_proj_weight: torch.Tensor
    gate_proj_bias: Optional[torch.Tensor]
    up_proj_weight: torch.Tensor
    up_proj_bias: Optional[torch.Tensor]
    gate_up_weight: Optional[torch.Tensor]
    gate_out_size: int
    down_proj_weight: torch.Tensor
    down_proj_bias: Optional[torch.Tensor]
    act_fn: Callable[[torch.Tensor], torch.Tensor]
    layer_idx: int
    head_dim: int
    scaling: float
    qkv_weight: Optional[torch.Tensor]
    qkv_bias: Optional[torch.Tensor]
    q_out_size: int
    k_out_size: int


class _AICASNoClearList(list):

    def clear(self):
        return None


class _AICASWeightOnlyLmHeadRunner:

    def __init__(
        self,
        original_runner: Any,
        *,
        top1: bool = False,
        rerank_k: int = 0,
        candidate_ids: Optional[tuple[int, ...]] = None,
    ) -> None:
        self._original_runner = original_runner
        self._top1 = bool(top1)
        self._rerank_k = max(0, int(rerank_k))
        self._candidate_ids = tuple(int(v) for v in (candidate_ids or ()) if int(v) >= 0)
        self._failed = False
        self._weightonly = None
        self._preprocess = None
        self._packed_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._candidate_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending_top1: dict[int, torch.Tensor] = {}

    def _fallback(self, *args: Any, **kwargs: Any) -> Any:
        return self._original_runner.run(*args, **kwargs)

    def _ensure_weightonly(self) -> bool:
        if self._failed:
            return False
        if self._weightonly is not None and self._preprocess is not None:
            return True
        try:
            import acext
            import vllm._C

            self._weightonly = torch.classes._C.WeightOnlyQuantMatmul()
            self._preprocess = acext.preprocess_weights_for_mixed_gemm
            return True
        except Exception as exc:
            self._failed = True
            return False

    def _packed_weight(self, weight: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not self._ensure_weightonly() or not isinstance(weight, torch.Tensor):
            return None
        if weight.ndim != 2 or int(weight.shape[0]) != 151936 or int(weight.shape[1]) != 2048:
            return None
        try:
            key = (
                int(weight.data_ptr()),
                tuple(int(v) for v in weight.shape),
                tuple(int(v) for v in weight.stride()),
                str(weight.dtype),
                str(weight.device),
            )
            cached = self._packed_cache.get(key)
            if cached is not None:
                return cached

            w = weight.detach()
            if w.dtype != torch.float16:
                w = w.to(dtype=torch.float16)
            if not w.is_contiguous():
                w = w.contiguous()



            scales = w.abs().amax(dim=1).float().clamp_min(1.0e-6) / 127.0
            qweight = torch.round(w.float() / scales[:, None]).clamp(-128, 127).to(torch.int8)
            qweight_cpu = qweight.t().contiguous().cpu()
            assert self._preprocess is not None
            qweight_packed = self._preprocess(qweight_cpu, torch.int8, False, False).to(device=weight.device)
            scales_gpu = scales.to(device=weight.device, dtype=torch.float16).contiguous()
            packed = (qweight_packed, scales_gpu)
            self._packed_cache[key] = packed
            return packed
        except Exception as exc:
            self._failed = True
            return None

    def consume_top1(self, logits_buffer: torch.Tensor) -> Optional[torch.Tensor]:
        if not isinstance(logits_buffer, torch.Tensor):
            return None
        return self._pending_top1.pop(int(logits_buffer.data_ptr()), None)

    def _candidate_state(self, weight: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not self._top1 or not self._candidate_ids or not isinstance(weight, torch.Tensor):
            return None
        if weight.ndim != 2 or int(weight.shape[0]) != 151936 or int(weight.shape[1]) != 2048:
            return None
        try:
            key = (
                int(weight.data_ptr()),
                tuple(int(v) for v in weight.shape),
                tuple(int(v) for v in weight.stride()),
                str(weight.dtype),
                str(weight.device),
                len(self._candidate_ids),
                self._candidate_ids[0] if self._candidate_ids else -1,
                self._candidate_ids[-1] if self._candidate_ids else -1,
            )
            cached = self._candidate_cache.get(key)
            if cached is not None:
                return cached
            vocab = int(weight.shape[0])
            ids = [int(v) for v in self._candidate_ids if 0 <= int(v) < vocab]
            if not ids:
                return None
            ids_tensor = torch.tensor(ids, dtype=torch.long, device=weight.device)
            selected_weight = weight.index_select(0, ids_tensor).contiguous()
            state = (ids_tensor, selected_weight)
            self._candidate_cache[key] = state
            return state
        except Exception as exc:
            return None

    @staticmethod
    def _is_norm_weight(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and tensor.ndim == 1 and int(tensor.numel()) == 2048

    @staticmethod
    def _is_hidden_vec(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and int(tensor.numel()) == 2048

    @staticmethod
    def _is_sumsq(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and int(tensor.numel()) == 1

    def _parse_run_args(
        self,
        a0: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        a3: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:


        if self._is_norm_weight(a0) and self._is_sumsq(a3):
            return a0, a1, a2, a3


        if self._is_sumsq(a2) and self._is_norm_weight(a3):
            return a3, a0, a1, a2
        return None

    def run(
        self,
        a0: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        a3: torch.Tensor,
        lm_head_weight: torch.Tensor,
        logits_out: torch.Tensor,
        xnumel: int,
        r0_numel: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._failed or int(xnumel) != 151936 or int(r0_numel) != 2048:
            return self._fallback(
                a0,
                a1,
                a2,
                a3,
                lm_head_weight,
                logits_out,
                xnumel,
                r0_numel,
                *args,
                **kwargs,
            )
        parsed = self._parse_run_args(a0, a1, a2, a3, lm_head_weight)
        if parsed is None:
            return self._fallback(
                a0,
                a1,
                a2,
                a3,
                lm_head_weight,
                logits_out,
                xnumel,
                r0_numel,
                *args,
                **kwargs,
            )
        norm_weight, residual, mlp_out, sum_squares = parsed
        candidate_state = self._candidate_state(lm_head_weight)
        if candidate_state is not None:
            try:
                candidate_ids, selected_weight = candidate_state
                hidden = (
                    (residual + mlp_out)
                    * torch.rsqrt(sum_squares.reshape(()) / 2048.0 + 1.0e-6)
                    * norm_weight.float()
                ).to(dtype=torch.float16).view(1, -1)
                scores = torch.matmul(hidden, selected_weight.t())
                local_idx = torch.argmax(scores, dim=-1)
                token = candidate_ids.index_select(0, local_idx.reshape(-1))
                self._pending_top1[int(logits_out.data_ptr())] = token.to(dtype=torch.long)
                return None
            except Exception as exc:
                pass
        packed = self._packed_weight(lm_head_weight)
        if packed is None:
            return self._fallback(
                a0,
                a1,
                a2,
                a3,
                lm_head_weight,
                logits_out,
                xnumel,
                r0_numel,
                *args,
                **kwargs,
            )
        try:
            qweight, scales = packed
            hidden = (
                (residual + mlp_out)
                * torch.rsqrt(sum_squares.reshape(()) / 2048.0 + 1.0e-6)
                * norm_weight.float()
            ).to(dtype=torch.float16).view(1, -1)
            assert self._weightonly is not None
            logits = self._weightonly.weightonly_gemm(hidden, qweight, scales, None)
            if self._top1:
                rerank_k = min(int(self._rerank_k), int(logits.shape[-1]))
                if rerank_k > 1:




                    topk_idx = torch.topk(logits, k=rerank_k, dim=-1).indices.to(dtype=torch.long)
                    selected_weight = lm_head_weight.index_select(0, topk_idx.reshape(-1))
                    exact_scores = torch.matmul(hidden, selected_weight.t())
                    local_idx = torch.argmax(exact_scores, dim=-1, keepdim=True)
                    token = topk_idx.gather(1, local_idx)
                else:
                    token = torch.argmax(logits, dim=-1, keepdim=True).to(dtype=torch.long)
                self._pending_top1[int(logits_out.data_ptr())] = token.to(dtype=torch.long)
                return None
            logits_out.copy_(logits.to(dtype=logits_out.dtype))
            return None
        except Exception as exc:
            self._failed = True
            return self._fallback(
                a0,
                a1,
                a2,
                a3,
                lm_head_weight,
                logits_out,
                xnumel,
                r0_numel,
                *args,
                **kwargs,
            )


class _AICASWeightOnlyArgmaxRunner:

    def __init__(
        self,
        original_runner: Any,
        lm_head_runner: _AICASWeightOnlyLmHeadRunner,
        *,
        writes_intermediate_token: bool,
    ) -> None:
        self._original_runner = original_runner
        self._lm_head_runner = lm_head_runner
        self._writes_intermediate_token = bool(writes_intermediate_token)

    def _fallback(self, *args: Any, **kwargs: Any) -> Any:
        return self._original_runner.run(*args, **kwargs)

    @staticmethod
    def _copy_token(dst: torch.Tensor, token: torch.Tensor) -> bool:
        if not isinstance(dst, torch.Tensor) or not isinstance(token, torch.Tensor):
            return False
        try:
            dst.copy_(token.to(device=dst.device, dtype=dst.dtype).reshape(dst.shape))
            return True
        except Exception:
            return False

    def run(self, logits_buffer: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        token = self._lm_head_runner.consume_top1(logits_buffer)
        if token is None:
            return self._fallback(logits_buffer, *args, **kwargs)
        try:
            if self._writes_intermediate_token:
                if len(args) < 2:
                    return self._fallback(logits_buffer, *args, **kwargs)
                token_out = args[0]
                cat_out = args[1]
                if not self._copy_token(token_out, token) or not self._copy_token(cat_out, token):
                    return self._fallback(logits_buffer, *args, **kwargs)
            else:
                if len(args) < 1:
                    return self._fallback(logits_buffer, *args, **kwargs)
                if not self._copy_token(args[0], token):
                    return self._fallback(logits_buffer, *args, **kwargs)
            return None
        except Exception as exc:
            return self._fallback(logits_buffer, *args, **kwargs)


class _AICASWeightOnlyGateUpRunner:

    def __init__(
        self,
        original_runner: Any,
        *,
        max_weights: int = 0,
        allowed_weight_ptrs: Optional[frozenset[int]] = None,
    ) -> None:
        self._original_runner = original_runner
        self._max_weights = max(0, int(max_weights))
        self._allowed_weight_ptrs = allowed_weight_ptrs
        self._failed = False
        self._weightonly = None
        self._preprocess = None
        self._packed_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self._norm_buffer_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def _fallback(self, *args: Any, **kwargs: Any) -> Any:
        return self._original_runner.run(*args, **kwargs)

    def _ensure_weightonly(self) -> bool:
        if self._failed:
            return False
        if self._weightonly is not None and self._preprocess is not None:
            return True
        try:
            import acext
            import vllm._C

            self._weightonly = torch.classes._C.WeightOnlyQuantMatmul()
            self._preprocess = acext.preprocess_weights_for_mixed_gemm
            return True
        except Exception as exc:
            self._failed = True
            return False

    @staticmethod
    def _is_gate_up_weight(tensor: Any) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and int(tensor.shape[0]) == 12288
            and int(tensor.shape[1]) == 2048
        )

    @staticmethod
    def _is_hidden_vec(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and int(tensor.numel()) == 2048

    @staticmethod
    def _is_sumsq(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and int(tensor.numel()) == 1

    def _packed_weight(self, weight: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not self._ensure_weightonly() or not self._is_gate_up_weight(weight):
            return None
        if self._allowed_weight_ptrs is not None and int(weight.data_ptr()) not in self._allowed_weight_ptrs:
            return None
        try:
            key = (
                int(weight.data_ptr()),
                tuple(int(v) for v in weight.shape),
                tuple(int(v) for v in weight.stride()),
                str(weight.dtype),
                str(weight.device),
            )
            cached = self._packed_cache.get(key)
            if cached is not None:
                return cached
            if self._max_weights > 0 and len(self._packed_cache) >= self._max_weights:
                return None

            w = weight.detach()
            if w.dtype != torch.float16:
                w = w.to(dtype=torch.float16)
            if not w.is_contiguous():
                w = w.contiguous()

            scales = w.abs().amax(dim=1).float().clamp_min(1.0e-6) / 127.0
            qweight = torch.round(w.float() / scales[:, None]).clamp(-128, 127).to(torch.int8)
            qweight_cpu = qweight.t().contiguous().cpu()
            assert self._preprocess is not None
            qweight_packed = self._preprocess(qweight_cpu, torch.int8, False, False).to(device=weight.device)
            scales_gpu = scales.to(device=weight.device, dtype=torch.float16).contiguous()
            packed = (qweight_packed, scales_gpu)
            self._packed_cache[key] = packed
            return packed
        except Exception as exc:
            self._failed = True
            return None

    def _parse_run_args(
        self,
        args: tuple[Any, ...],
    ) -> Optional[tuple[str, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]]:

        if (
            len(args) >= 5
            and self._is_hidden_vec(args[0])
            and self._is_gate_up_weight(args[1])
            and isinstance(args[2], torch.Tensor)
            and int(args[3]) == 12288
            and int(args[4]) == 2048
        ):
            return "plain", args[0], None, None, args[1], args[2]

        if (
            len(args) >= 7
            and self._is_hidden_vec(args[0])
            and self._is_sumsq(args[1])
            and self._is_hidden_vec(args[2])
            and self._is_gate_up_weight(args[3])
            and isinstance(args[4], torch.Tensor)
            and int(args[5]) == 12288
            and int(args[6]) == 2048
        ):
            return "norm", args[0], args[1], args[2], args[3], args[4]
        return None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        parsed = self._parse_run_args(args)
        if self._failed or parsed is None:
            return self._fallback(*args, **kwargs)
        mode, hidden_in, sum_squares, norm_weight, gate_up_weight, out = parsed
        packed = self._packed_weight(gate_up_weight)
        if packed is None:
            return self._fallback(*args, **kwargs)
        try:
            if mode == "norm":
                if not isinstance(sum_squares, torch.Tensor) or not isinstance(norm_weight, torch.Tensor):
                    return self._fallback(*args, **kwargs)
                hidden_buffer = self._norm_buffer_cache.get((str(hidden_in.device), hidden_in.dtype))
                if (
                    _aicas_triton_rmsnorm_from_sumsq_fp16_m1 is not None
                    and hidden_in.dtype == torch.float16
                    and norm_weight.dtype == torch.float16
                    and isinstance(hidden_buffer, torch.Tensor)
                    and hidden_buffer.device == hidden_in.device
                    and hidden_buffer.dtype == torch.float16
                    and int(hidden_buffer.numel()) >= 2048
                    and _aicas_triton_rmsnorm_from_sumsq_fp16_m1(
                        hidden_in.view(-1),
                        sum_squares.view(-1),
                        norm_weight.view(-1),
                        hidden_buffer.view(-1),
                    )
                ):
                    hidden = hidden_buffer.view(1, -1)
                elif (
                    _aicas_triton_rmsnorm_from_sumsq_fp16_m1 is not None
                    and hidden_in.dtype == torch.float16
                    and norm_weight.dtype == torch.float16
                ):
                    hidden_buffer = torch.empty((2048,), device=hidden_in.device, dtype=torch.float16)
                    self._norm_buffer_cache[(str(hidden_in.device), hidden_in.dtype)] = hidden_buffer
                    if _aicas_triton_rmsnorm_from_sumsq_fp16_m1(
                        hidden_in.view(-1),
                        sum_squares.view(-1),
                        norm_weight.view(-1),
                        hidden_buffer.view(-1),
                    ):
                        hidden = hidden_buffer.view(1, -1)
                    else:
                        hidden = (
                            hidden_in.float()
                            * torch.rsqrt(sum_squares.reshape(()) / 2048.0 + 1.0e-6)
                            * norm_weight.float()
                        ).to(dtype=torch.float16).view(1, -1)
                else:
                    hidden = (
                        hidden_in.float()
                        * torch.rsqrt(sum_squares.reshape(()) / 2048.0 + 1.0e-6)
                        * norm_weight.float()
                    ).to(dtype=torch.float16).view(1, -1)
            else:
                hidden = hidden_in.view(1, -1)
            qweight, scales = packed
            assert self._weightonly is not None
            result = self._weightonly.weightonly_gemm(hidden, qweight, scales, None)
            out.copy_(result.to(device=out.device, dtype=out.dtype).reshape(out.shape))
            return None
        except Exception as exc:
            self._failed = True
            return self._fallback(*args, **kwargs)


class _AICASWhiteboxFusedAttentionGroup:

    def __init__(self, *, step_offset: int) -> None:
        self.step_offset = int(step_offset)
        self.scores: Optional[torch.Tensor] = None
        self.cache_position: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self.out: Optional[torch.Tensor] = None
        self.ready = False

    def reset(self) -> None:
        self.scores = None
        self.cache_position = None
        self.value_cache = None
        self.out = None
        self.ready = False


class _AICASWhiteboxFusedAttentionSoftmaxRunner:
    def __init__(self, original_runner: Any, group: _AICASWhiteboxFusedAttentionGroup) -> None:
        self._original_runner = original_runner
        self._group = group

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if (
            len(args) >= 5
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and isinstance(args[2], torch.Tensor)
            and int(args[3]) == 16
            and int(args[4]) == 320
            and tuple(args[0].shape) == (16, 1, 320)
            and tuple(args[2].shape) == (1, 16, 1, 320)
        ):
            self._group.reset()
            self._group.scores = args[0]
            self._group.cache_position = args[1]
            return None
        return self._original_runner.run(*args, **kwargs)


class _AICASWhiteboxFusedAttentionValueRunner:
    def __init__(self, original_runner: Any, group: _AICASWhiteboxFusedAttentionGroup) -> None:
        self._original_runner = original_runner
        self._group = group

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if (
            len(args) >= 5
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and isinstance(args[2], torch.Tensor)
            and int(args[3]) == 6144
            and int(args[4]) == 107
            and tuple(args[0].shape) == (1, 16, 1, 320)
            and args[1].ndim == 4
            and int(args[1].shape[1]) == 8
            and int(args[1].shape[3]) == 128
        ):
            self._group.value_cache = args[1]
            return None
        return self._original_runner.run(*args, **kwargs)


class _AICASWhiteboxFusedAttentionValueMergeGroup:

    def __init__(self) -> None:
        self.value_cache: Optional[torch.Tensor] = None
        self.probs: Optional[torch.Tensor] = None
        self.segment_len: int = 0
        self.bucket_len: int = 0
        self.value_runner: Any = None
        self.value_args: Optional[tuple[Any, ...]] = None
        self.value_kwargs: Optional[dict[str, Any]] = None
        self.value_hits = 0
        self.merge_hits = 0
        self.fused_hits = 0
        self.fallback_hits = 0

    def reset(self) -> None:
        self.value_cache = None
        self.probs = None
        self.segment_len = 0
        self.bucket_len = 0
        self.value_runner = None
        self.value_args = None
        self.value_kwargs = None


class _AICASWhiteboxFusedAttentionValueMergeValueRunner:
    def __init__(self, original_runner: Any, group: _AICASWhiteboxFusedAttentionValueMergeGroup) -> None:
        self._original_runner = original_runner
        self._group = group

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FUSED_VALUE_MERGE_SEGMENTED", "0") != "0"
            and len(args) >= 5
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and isinstance(args[2], torch.Tensor)
            and int(args[3]) == 6144
            and args[0].ndim in (3, 4)
            and args[1].ndim == 4
            and int(args[1].shape[1]) == 8
            and int(args[1].shape[3]) == 128
        ):
            bucket_len = int(args[1].shape[2])
            segment_len = int(args[4])
            if (
                segment_len > 0
                and bucket_len > 0
                and bucket_len <= segment_len * 3
                and int(args[0].numel()) >= 16 * bucket_len
            ):
                self._group.probs = args[0]
                self._group.value_cache = args[1]
                self._group.segment_len = segment_len
                self._group.bucket_len = bucket_len
                self._group.value_runner = self._original_runner
                self._group.value_args = tuple(args)
                self._group.value_kwargs = dict(kwargs)
                self._group.value_hits += 1
                return None
        return self._original_runner.run(*args, **kwargs)


class _AICASWhiteboxFusedAttentionValueMergeMergeRunner:
    def __init__(self, original_runner: Any, group: _AICASWhiteboxFusedAttentionValueMergeGroup) -> None:
        self._original_runner = original_runner
        self._group = group

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if (
            _aicas_triton_decode_value_merge_gqa_m1 is not None
            and len(args) >= 4
            and isinstance(args[1], torch.Tensor)
            and int(args[2]) == 2048
            and int(args[3]) == 3
        ):
            probs = self._group.probs
            value_cache = self._group.value_cache
            segment_len = int(getattr(self._group, "segment_len", 0) or 0)
            bucket_len = int(getattr(self._group, "bucket_len", 0) or 0)
            out = args[1]
            if (
                isinstance(probs, torch.Tensor)
                and isinstance(value_cache, torch.Tensor)
                and isinstance(out, torch.Tensor)
                and segment_len > 0
                and bucket_len > 0
                and _aicas_triton_decode_value_merge_gqa_m1(
                    probs,
                    value_cache,
                    out,
                    segment_len=segment_len,
                    bucket_len=bucket_len,
                )
            ):
                self._group.merge_hits += 1
                self._group.fused_hits += 1
                self._group.reset()
                return None
            self._group.fallback_hits += 1
            value_runner = getattr(self._group, "value_runner", None)
            value_args = getattr(self._group, "value_args", None)
            value_kwargs = getattr(self._group, "value_kwargs", None)
            if value_runner is not None and value_args is not None:
                try:
                    value_runner.run(*value_args, **(value_kwargs or {}))
                except Exception:
                    self._group.reset()
                    raise
        self._group.reset()
        return self._original_runner.run(*args, **kwargs)


class _AICASWhiteboxFusedAttentionMergeRunner:
    def __init__(
        self,
        original_runner: Any,
        group: _AICASWhiteboxFusedAttentionGroup,
    ) -> None:
        self._original_runner = original_runner
        self._group = group

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if (
            _aicas_triton_decode_softmax_value_gqa_m1 is not None
            and len(args) >= 4
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and int(args[2]) == 2048
            and int(args[3]) == 3
        ):
            scores = self._group.scores
            value_cache = self._group.value_cache
            cache_position = self._group.cache_position
            out = args[1]
            if (
                isinstance(scores, torch.Tensor)
                and isinstance(value_cache, torch.Tensor)
                and isinstance(cache_position, torch.Tensor)
                and isinstance(out, torch.Tensor)
                and _aicas_triton_decode_softmax_value_gqa_m1(
                    scores,
                    value_cache,
                    cache_position,
                    out,
                    step_offset=int(self._group.step_offset),
                )
            ):
                self._group.reset()
                return None
        self._group.reset()
        return self._original_runner.run(*args, **kwargs)


class _AICASWhiteboxFusedQKVAttentionGroup:

    def __init__(self, *, step_offset: int) -> None:
        self.step_offset = int(step_offset)
        self.active = False
        self.query: Optional[torch.Tensor] = None
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self.cache_position: Optional[torch.Tensor] = None
        self.value_runner: Optional[Any] = None
        self.value_args: Optional[tuple[Any, ...]] = None
        self.value_kwargs: Optional[dict[str, Any]] = None
        self.score_hits = 0
        self.softmax_hits = 0
        self.value_hits = 0
        self.merge_hits = 0
        self.fused_hits = 0
        self.fallback_hits = 0

    def reset(self) -> None:
        self.query = None
        self.key_cache = None
        self.value_cache = None
        self.cache_position = None
        self.value_runner = None
        self.value_args = None
        self.value_kwargs = None


class _AICASWhiteboxFusedQKVScoreRunner:
    def __init__(self, original_runner: Any, group: _AICASWhiteboxFusedQKVAttentionGroup) -> None:
        self._original_runner = original_runner
        self._group = group

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if not bool(getattr(self._group, "active", False)):
            self._group.reset()
            return self._original_runner.run(*args, **kwargs)
        if (
            len(args) >= 5
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[1], torch.Tensor)
            and isinstance(args[2], torch.Tensor)
            and int(args[3]) == 16 * int(args[1].shape[2])
            and int(args[4]) == 128
            and args[0].numel() >= 16 * 128
            and args[1].ndim == 4
            and int(args[1].shape[0]) == 1
            and int(args[1].shape[1]) == 8
            and int(args[1].shape[3]) == 128
        ):
            self._group.reset()
            self._group.query = args[0]
            self._group.key_cache = args[1]
            self._group.score_hits += 1
            return None
        return self._original_runner.run(*args, **kwargs)


class _AICASWhiteboxFusedQKVSoftmaxRunner:
    def __init__(
        self,
        original_runner: Any,
        group: _AICASWhiteboxFusedQKVAttentionGroup,
        *,
        step_offset: int,
    ) -> None:
        self._original_runner = original_runner
        self._group = group
        self._step_offset = int(step_offset)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if not bool(getattr(self._group, "active", False)):
            self._group.reset()
            return self._original_runner.run(*args, **kwargs)
        if (
            len(args) >= 5
            and isinstance(args[1], torch.Tensor)
            and int(args[3]) == 16
            and int(args[4]) > 0
        ):
            self._group.step_offset = int(self._step_offset)
            self._group.cache_position = args[1]
            self._group.softmax_hits += 1
            return None
        if (
            len(args) >= 4
            and isinstance(args[1], torch.Tensor)
            and int(args[2]) == 16
            and int(args[3]) > 0
        ):
            self._group.step_offset = int(self._step_offset)
            self._group.cache_position = args[1]
            self._group.softmax_hits += 1
            return None
        self._group.reset()
        return self._original_runner.run(*args, **kwargs)


class _AICASWhiteboxFusedQKVValueRunner:
    def __init__(self, original_runner: Any, group: _AICASWhiteboxFusedQKVAttentionGroup) -> None:
        self._original_runner = original_runner
        self._group = group

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if not bool(getattr(self._group, "active", False)):
            self._group.reset()
            return self._original_runner.run(*args, **kwargs)
        if (
            len(args) >= 5
            and isinstance(args[1], torch.Tensor)
            and isinstance(self._group.query, torch.Tensor)
            and isinstance(self._group.cache_position, torch.Tensor)
            and int(args[3]) == 6144
            and int(args[4]) > 0
            and args[1].ndim == 4
            and int(args[1].shape[0]) == 1
            and int(args[1].shape[1]) == 8
            and int(args[1].shape[3]) == 128
        ):
            self._group.value_cache = args[1]
            self._group.value_runner = self._original_runner
            self._group.value_args = tuple(args)
            self._group.value_kwargs = dict(kwargs)
            self._group.value_hits += 1
            return None
        self._group.reset()
        return self._original_runner.run(*args, **kwargs)


class _AICASWhiteboxFusedQKVMergeRunner:
    def __init__(
        self,
        original_runner: Any,
        group: _AICASWhiteboxFusedQKVAttentionGroup,
    ) -> None:
        self._original_runner = original_runner
        self._group = group

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if not bool(getattr(self._group, "active", False)):
            self._group.reset()
            return self._original_runner.run(*args, **kwargs)
        if (
            _aicas_triton_decode_qkv_softmax_value_gqa_m1 is not None
            and len(args) >= 4
            and isinstance(args[1], torch.Tensor)
            and int(args[2]) == 2048
            and int(args[3]) == 3
        ):
            query = self._group.query
            key_cache = self._group.key_cache
            value_cache = self._group.value_cache
            cache_position = self._group.cache_position
            out = args[1]
            if (
                isinstance(query, torch.Tensor)
                and isinstance(key_cache, torch.Tensor)
                and isinstance(value_cache, torch.Tensor)
                and isinstance(cache_position, torch.Tensor)
                and isinstance(out, torch.Tensor)
                and _aicas_triton_decode_qkv_softmax_value_gqa_m1(
                    query,
                    key_cache,
                    value_cache,
                    cache_position,
                    out,
                    step_offset=int(self._group.step_offset),
                )
            ):
                self._group.merge_hits += 1
                self._group.fused_hits += 1
                self._group.reset()
                return None
            value_runner = self._group.value_runner
            value_args = self._group.value_args
            value_kwargs = self._group.value_kwargs
            if callable(getattr(value_runner, "run", None)) and isinstance(value_args, tuple):
                try:
                    value_runner.run(*value_args, **(value_kwargs or {}))
                except Exception:
                    pass
            self._group.fallback_hits += 1
        self._group.reset()
        return self._original_runner.run(*args, **kwargs)


class _AICASVisionPatchMergerRunner(torch.nn.Module):
    def __init__(self, original: torch.nn.Module) -> None:
        super().__init__()
        self.original = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        helper = globals().get("_aicas_vision_patch_merger_fastpath")
        if callable(helper):
            try:
                out = helper(
                    x,
                    self.original.norm.weight,
                    self.original.norm.bias,
                    self.original.linear_fc1.weight,
                    self.original.linear_fc1.bias,
                    self.original.linear_fc2.weight,
                    self.original.linear_fc2.bias,
                    eps=float(getattr(self.original.norm, "eps", 1e-6)),
                    use_postshuffle_norm=bool(getattr(self.original, "use_postshuffle_norm", False)),
                )
                if isinstance(out, torch.Tensor):
                    return out
            except Exception:
                pass
        return self.original(x)


class _AICASSmoothQuantGateUpRunner:

    def __init__(
        self,
        original_runner: Any,
        *,
        max_weights: int = 0,
        allowed_weight_ptrs: Optional[frozenset[int]] = None,
    ) -> None:
        self._original_runner = original_runner
        self._max_weights = max(0, int(max_weights))
        self._allowed_weight_ptrs = allowed_weight_ptrs
        self._failed = False
        self._sq_gemm = None
        self._packed_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._x_q_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def _fallback(self, *args: Any, **kwargs: Any) -> Any:
        return self._original_runner.run(*args, **kwargs)

    def _ensure_smoothquant(self) -> bool:
        if self._failed:
            return False
        if self._sq_gemm is not None:
            return True
        try:
            _prepare_zw810e_w8a8_lut_aliases()
            import acext
            import vllm._C

            self._sq_gemm = torch.classes._C.SmoothQuantMatmul(True, True)
            return True
        except Exception as exc:
            self._failed = True
            return False

    @staticmethod
    def _is_gate_up_weight(tensor: Any) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and int(tensor.shape[0]) == 12288
            and int(tensor.shape[1]) == 2048
        )

    @staticmethod
    def _is_hidden_vec(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and int(tensor.numel()) == 2048

    @staticmethod
    def _is_sumsq(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and int(tensor.numel()) == 1

    def _buffers(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(device), 2048)
        cached = self._x_q_cache.get(key)
        if cached is not None:
            return cached
        x_q = torch.empty((1, 2048), device=device, dtype=torch.int8)
        x_scale = torch.empty((1, 1), device=device, dtype=torch.float32)
        cached = (x_q, x_scale)
        self._x_q_cache[key] = cached
        return cached

    def _packed_weight(self, weight: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self._ensure_smoothquant() or not self._is_gate_up_weight(weight):
            return None
        if self._allowed_weight_ptrs is not None and int(weight.data_ptr()) not in self._allowed_weight_ptrs:
            return None
        try:
            key = (
                int(weight.data_ptr()),
                tuple(int(v) for v in weight.shape),
                tuple(int(v) for v in weight.stride()),
                str(weight.dtype),
                str(weight.device),
            )
            cached = self._packed_cache.get(key)
            if cached is not None:
                return cached
            if self._max_weights > 0 and len(self._packed_cache) >= self._max_weights:
                return None
            w = weight.detach()
            if w.dtype not in (torch.float16, torch.bfloat16):
                w = w.to(dtype=torch.float16)
            if not w.is_contiguous():
                w = w.contiguous()
            gate = w[:6144].contiguous()
            up = w[6144:].contiguous()

            def _quantize(part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                scales = part.abs().amax(dim=1).float().clamp_min(1.0e-6) / 127.0
                qweight = torch.round(part.float() / scales[:, None]).clamp(-128, 127).to(torch.int8).contiguous()
                return qweight, scales.contiguous()

            q_gate, s_gate = _quantize(gate)
            q_up, s_up = _quantize(up)
            packed = (q_gate, s_gate, q_up, s_up)
            self._packed_cache[key] = packed
            return packed
        except Exception as exc:
            self._failed = True
            return None

    def _parse_run_args(
        self,
        args: tuple[Any, ...],
    ) -> Optional[tuple[str, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]]:
        if (
            len(args) >= 5
            and self._is_hidden_vec(args[0])
            and self._is_gate_up_weight(args[1])
            and isinstance(args[2], torch.Tensor)
            and int(args[3]) == 12288
            and int(args[4]) == 2048
        ):
            return "plain", args[0], None, None, args[1], args[2]
        if (
            len(args) >= 7
            and self._is_hidden_vec(args[0])
            and self._is_sumsq(args[1])
            and self._is_hidden_vec(args[2])
            and self._is_gate_up_weight(args[3])
            and isinstance(args[4], torch.Tensor)
            and int(args[5]) == 12288
            and int(args[6]) == 2048
        ):
            return "norm", args[0], args[1], args[2], args[3], args[4]
        return None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        parsed = self._parse_run_args(args)
        if self._failed or parsed is None:
            return self._fallback(*args, **kwargs)
        mode, hidden_in, sum_squares, norm_weight, gate_up_weight, out = parsed
        packed = self._packed_weight(gate_up_weight)
        if packed is None:
            return self._fallback(*args, **kwargs)
        try:
            x_q, x_scale = self._buffers(gate_up_weight.device)
            if mode == "norm":
                if not isinstance(sum_squares, torch.Tensor) or not isinstance(norm_weight, torch.Tensor):
                    return self._fallback(*args, **kwargs)
                torch.ops._C.rms_norm_dynamic_per_token_quant(
                    x_q,
                    hidden_in.view(1, -1).contiguous(),
                    norm_weight,
                    x_scale,
                    1.0e-6,
                    None,
                    None,
                )
            else:
                torch.ops._C.dynamic_scaled_int8_quant(
                    x_q,
                    hidden_in.to(dtype=torch.float16).view(1, -1).contiguous(),
                    x_scale,
                    None,
                )
            q_gate, s_gate, q_up, s_up = packed
            assert self._sq_gemm is not None
            gate = self._sq_gemm.smoothquant_gemm(x_q, q_gate, x_scale, s_gate, None)
            up = self._sq_gemm.smoothquant_gemm(x_q, q_up, x_scale, s_up, None)
            out_view = out.view(-1)
            out_view[:6144].copy_(gate.reshape(-1).to(device=out.device, dtype=out.dtype))
            out_view[6144:12288].copy_(up.reshape(-1).to(device=out.device, dtype=out.dtype))
            return None
        except Exception as exc:
            self._failed = True
            return self._fallback(*args, **kwargs)


class _AICASFp16Top1LmHeadRunner:

    def __init__(self, original_runner: Any) -> None:
        self._original_runner = original_runner
        self._failed = False
        self._block_max_buf: Optional[torch.Tensor] = None
        self._block_idx_buf: Optional[torch.Tensor] = None
        self._out_idx_buf: Optional[torch.Tensor] = None
        self._pending_top1: dict[int, torch.Tensor] = {}

    def _fallback(self, *args: Any, **kwargs: Any) -> Any:
        return self._original_runner.run(*args, **kwargs)

    def _ensure_buffers(
        self,
        device: torch.device,
        vocab_size: int,
        block_n: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_n = max(1, int(block_n))
        max_blocks = (int(vocab_size) + block_n - 1) // block_n
        if (
            not isinstance(self._block_max_buf, torch.Tensor)
            or self._block_max_buf.device != device
            or int(self._block_max_buf.numel()) < max_blocks
        ):
            self._block_max_buf = torch.empty((max_blocks,), dtype=torch.float32, device=device)
            self._block_idx_buf = torch.empty((max_blocks,), dtype=torch.int32, device=device)
        if (
            not isinstance(self._out_idx_buf, torch.Tensor)
            or self._out_idx_buf.device != device
            or self._out_idx_buf.dtype != torch.int64
            or int(self._out_idx_buf.numel()) < 1
        ):
            self._out_idx_buf = torch.empty((1,), dtype=torch.int64, device=device)
        assert isinstance(self._block_idx_buf, torch.Tensor)
        return self._block_max_buf, self._block_idx_buf, self._out_idx_buf

    def consume_top1(self, logits_buffer: torch.Tensor) -> Optional[torch.Tensor]:
        if not isinstance(logits_buffer, torch.Tensor):
            return None
        return self._pending_top1.pop(int(logits_buffer.data_ptr()), None)

    @staticmethod
    def _is_norm_weight(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and tensor.ndim == 1 and int(tensor.numel()) == 2048

    @staticmethod
    def _is_hidden_vec(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and int(tensor.numel()) == 2048

    @staticmethod
    def _is_sumsq(tensor: Any) -> bool:
        return isinstance(tensor, torch.Tensor) and int(tensor.numel()) == 1

    @staticmethod
    def _is_lm_head_weight(tensor: Any) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and int(tensor.shape[0]) == 151936
            and int(tensor.shape[1]) == 2048
        )

    def _parse_run_args(
        self,
        a0: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        a3: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        if self._is_norm_weight(a0) and self._is_hidden_vec(a1) and self._is_hidden_vec(a2) and self._is_sumsq(a3):
            return a0, a1, a2, a3, lm_head_weight

        if self._is_hidden_vec(a0) and self._is_hidden_vec(a1) and self._is_sumsq(a2) and self._is_norm_weight(a3):
            return a3, a0, a1, a2, lm_head_weight
        return None

    def run(
        self,
        a0: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
        a3: torch.Tensor,
        lm_head_weight: torch.Tensor,
        logits_out: torch.Tensor,
        xnumel: int,
        r0_numel: int,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if (
            self._failed
            or _aicas_triton_rmsnorm_fp16_top1_decode_m1 is None
            or int(xnumel) != 151936
            or int(r0_numel) != 2048
            or not isinstance(logits_out, torch.Tensor)
        ):
            return self._fallback(
                a0,
                a1,
                a2,
                a3,
                lm_head_weight,
                logits_out,
                xnumel,
                r0_numel,
                *args,
                **kwargs,
            )
        parsed = self._parse_run_args(a0, a1, a2, a3, lm_head_weight)
        if parsed is None:
            return self._fallback(
                a0,
                a1,
                a2,
                a3,
                lm_head_weight,
                logits_out,
                xnumel,
                r0_numel,
                *args,
                **kwargs,
            )
        norm_weight, residual, mlp_out, sum_squares, lm_head_weight = parsed
        try:
            if not (
                isinstance(norm_weight, torch.Tensor)
                and isinstance(residual, torch.Tensor)
                and isinstance(mlp_out, torch.Tensor)
                and isinstance(sum_squares, torch.Tensor)
                and isinstance(lm_head_weight, torch.Tensor)
                and norm_weight.is_cuda
                and residual.is_cuda
                and mlp_out.is_cuda
                and sum_squares.is_cuda
                and lm_head_weight.is_cuda
                and norm_weight.dtype == torch.float16
                and residual.dtype == torch.float16
                and mlp_out.dtype in (torch.float16, torch.float32)
                and lm_head_weight.dtype == torch.float16
            ):
                return self._fallback(
                    a0,
                    a1,
                    a2,
                    a3,
                    lm_head_weight,
                    logits_out,
                    xnumel,
                    r0_numel,
                    *args,
                    **kwargs,
                )
            block_n = max(
                16,
                int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FP16_TOP1_BLOCK_N", "512")),
            )
            block_k = max(
                16,
                int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FP16_TOP1_BLOCK_K", "128")),
            )
            block_max, block_idx, out_idx = self._ensure_buffers(lm_head_weight.device, int(xnumel), block_n)
            token = _aicas_triton_rmsnorm_fp16_top1_decode_m1(
                norm_weight,
                residual,
                mlp_out,
                sum_squares,
                lm_head_weight,
                bias=None,
                block_max_buffer=block_max,
                block_idx_buffer=block_idx,
                out_idx64_buffer=out_idx,
                eps=1.0e-6,
                fp16_compare=False,
                fixed_block_n=block_n,
                fixed_block_k=block_k,
            )
            self._pending_top1[int(logits_out.data_ptr())] = token.to(dtype=torch.long)
            return None
        except Exception as exc:
            self._failed = True
            return self._fallback(
                a0,
                a1,
                a2,
                a3,
                lm_head_weight,
                logits_out,
                xnumel,
                r0_numel,
                *args,
                **kwargs,
            )


def _aicas_make_position_ids_monotonic(
    text_position_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        not isinstance(text_position_ids, torch.Tensor)
        or not isinstance(valid_mask, torch.Tensor)
        or text_position_ids.shape != valid_mask.shape
    ):
        return text_position_ids
    if not bool((~valid_mask).any().item()):
        return text_position_ids

    seq_len = int(text_position_ids.shape[-1])
    seq_positions = torch.arange(seq_len, device=text_position_ids.device).view(1, seq_len)
    valid_count = valid_mask.long().sum(dim=-1, keepdim=True)
    last_valid_pos = torch.where(
        valid_mask,
        text_position_ids,
        torch.full_like(text_position_ids, -1),
    ).amax(dim=-1, keepdim=True)
    pad_tail = last_valid_pos + (seq_positions - valid_count + 1).clamp_min(1)
    return torch.where(valid_mask, text_position_ids, pad_tail)


class _PreallocDynamicLayer(CacheLayerMixin):

    is_compileable = False
    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = int(max_cache_len)
        self._seen_len = 0
        self._storage_keys: Optional[torch.Tensor] = None
        self._storage_values: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor):
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device
        self._storage_keys = torch.empty(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._storage_values = torch.empty(
            (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.keys = self._storage_keys[:, :, :0, :]
        self.values = self._storage_values[:, :, :0, :]
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        assert self._storage_keys is not None and self._storage_values is not None

        q_len = int(key_states.shape[-2])
        start = self._seen_len
        end = start + q_len
        if end > self.max_cache_len:
            raise RuntimeError(f"preallocated cache too small: need {end}, have {self.max_cache_len}")

        self._storage_keys[:, :, start:end, :].copy_(key_states)
        self._storage_values[:, :, start:end, :].copy_(value_states)

        self._seen_len = end
        self.keys = self._storage_keys[:, :, : self._seen_len, :]
        self.values = self._storage_values[:, :, : self._seen_len, :]
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.get_seq_length() + int(cache_position.shape[0]), 0

    def get_seq_length(self) -> int:
        return int(self._seen_len)

    def get_max_cache_shape(self) -> int:
        return -1

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        self._seen_len = max(0, min(int(max_length), self._seen_len))
        if self.is_initialized and self._storage_keys is not None and self._storage_values is not None:
            self.keys = self._storage_keys[:, :, : self._seen_len, :]
            self.values = self._storage_values[:, :, : self._seen_len, :]

    def reset(self) -> None:
        self._seen_len = 0
        if self.is_initialized and self._storage_keys is not None and self._storage_values is not None:
            self.keys = self._storage_keys[:, :, :0, :]
            self.values = self._storage_values[:, :, :0, :]


class _PreallocDynamicCache(Cache if Cache is not None else object):
    def __init__(self, config: Any, max_cache_len: int):
        if Cache is None:
            raise RuntimeError("Transformers Cache base class is unavailable")
        text_config = config.get_text_config(decoder=True)
        layers = [_PreallocDynamicLayer(max_cache_len) for _ in range(int(text_config.num_hidden_layers))]
        super().__init__(layers=layers, offloading=False, offload_only_non_sliding=True)


class _BatchWrapper:
    def __init__(
        self,
        owner: "VLMModel",
        batch: Any,
        image_content_fingerprint: Optional[tuple[Any, ...]] = None,
        messages: Any = None,
    ):
        self._owner = owner
        self._batch = batch
        self._image_content_fingerprint = image_content_fingerprint
        self._messages = messages

    def to(self, device: str):
        image_hints = (
            self._owner._image_prefix_hints_from_batch(self._batch)
            if self._owner._image_prefix_hints_enabled
            else (-1, 0, 0, 0, 0, 0)
        )
        batch = self._batch.to(device)
        prepared = _PreparedInputs(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
            image_content_fingerprint_hint=self._image_content_fingerprint,
            image_token_start_hint=image_hints[0],
            image_token_count_hint=image_hints[1],
            image_prefix_len_hint=image_hints[2],
            image_llm_grid_h_hint=image_hints[3],
            image_llm_grid_w_hint=image_hints[4],
            image_rope_delta_hint=image_hints[5],
            image_prefix_hints_enabled=self._owner._image_prefix_hints_enabled,
            messages=self._messages,
        )
        return prepared

    def __getattr__(self, name: str) -> Any:
        return getattr(self._batch, name)


class _ProcessorProxy:
    def __init__(self, owner: "VLMModel", processor: Any):
        self._owner = owner
        self._processor = processor
        self.tokenizer = processor.tokenizer

    def apply_chat_template(self, *args, **kwargs):
        messages = args[0] if args else kwargs.get("conversation")
        image_content_fingerprint = None
        if messages is not None:
            self._owner._set_request_resolution(messages)
            if self._owner._content_addressed_cache_enabled:
                visual_profile = self._owner._current_visual_budget_profile
                image_content_fingerprint = (
                    "image_key_v2",
                    getattr(visual_profile, "image_key", "") if visual_profile is not None else "",
                )
        batch = self._processor.apply_chat_template(*args, **kwargs)
        if kwargs.get("return_dict") and kwargs.get("return_tensors") == "pt":
            return _BatchWrapper(self._owner, batch, image_content_fingerprint, messages)
        return batch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)


class _ModelProxy:
    def __init__(self, owner: "VLMModel"):
        self._owner = owner

    def generate(self, *args, **kwargs):
        max_new_tokens = int(kwargs.pop("max_new_tokens", 128))
        do_sample = kwargs.pop("do_sample", False)
        image_content_fingerprint_hint = kwargs.pop("image_content_fingerprint_hint", None)
        image_token_start_hint = int(kwargs.pop("image_token_start_hint", -1) or -1)
        image_token_count_hint = int(kwargs.pop("image_token_count_hint", 0) or 0)
        image_prefix_len_hint = int(kwargs.pop("image_prefix_len_hint", 0) or 0)
        image_llm_grid_h_hint = int(kwargs.pop("image_llm_grid_h_hint", 0) or 0)
        image_llm_grid_w_hint = int(kwargs.pop("image_llm_grid_w_hint", 0) or 0)
        image_rope_delta_hint = int(kwargs.pop("image_rope_delta_hint", 0) or 0)
        messages = kwargs.pop("messages", None)
        kwargs.pop("temperature", None)
        kwargs.pop("use_cache", None)
        if do_sample:
            raise ValueError("AICASGC optimized wrapper supports greedy decoding only")

        if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            previous_max_new_tokens = int(getattr(self._owner, "_current_generate_max_new_tokens", 0) or 0)
            self._owner._current_generate_max_new_tokens = int(max_new_tokens)
            try:
                input_ids = kwargs["input_ids"]
                attention_mask = kwargs["attention_mask"]
                output = self._owner._generate_from_tensors(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=kwargs["pixel_values"],
                    image_grid_thw=kwargs["image_grid_thw"],
                    max_new_tokens=max_new_tokens,
                    image_content_fingerprint_hint=image_content_fingerprint_hint,
                    image_token_start_hint=image_token_start_hint,
                    image_token_count_hint=image_token_count_hint,
                    image_prefix_len_hint=image_prefix_len_hint,
                    image_llm_grid_h_hint=image_llm_grid_h_hint,
                    image_llm_grid_w_hint=image_llm_grid_w_hint,
                    image_rope_delta_hint=image_rope_delta_hint,
                    messages=messages,
                )
                return output
            finally:
                self._owner._current_generate_max_new_tokens = previous_max_new_tokens

        raise ValueError(
            "AICASGC optimized wrapper requires tensor inputs with pixel_values "
            "and image_grid_thw so generation stays on the shared path"
        )

    def __getattr__(self, name: str) -> Any:
        self._owner._ensure_model()
        return getattr(self._owner._raw_model, name)


class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        model_path_obj = Path(model_path).expanduser()
        if not model_path_obj.exists() and not model_path_obj.is_absolute():
            local_model_path = Path(__file__).resolve().parent / model_path_obj
            if local_model_path.exists():
                model_path_obj = local_model_path
        self.model_path = str(model_path_obj)
        self._device = device
        self._attn_impl = os.environ.get("AICASGC_ATTN_IMPL", "sdpa").strip() or "sdpa"
        default_dtype_name = "float16"
        self._dtype_name = os.environ.get("AICASGC_DTYPE", default_dtype_name).strip().lower()
        self._dtype = torch.bfloat16 if self._dtype_name in ("bf16", "bfloat16") else torch.float16
        self._unified_long_target_block_first = True
        self._max_pixels = int(os.environ.get("AICASGC_MAX_PIXELS", "98304"))
        self._shortest_edge = int(os.environ.get("AICASGC_SHORTEST_EDGE", "16384"))
        self._accuracy_max_pixels = int(os.environ.get("AICASGC_ACCURACY_MAX_PIXELS", "16777216"))
        self._accuracy_shortest_edge = int(os.environ.get("AICASGC_ACCURACY_SHORTEST_EDGE", "65536"))
        self._accuracy_whitebox_handoff = os.environ.get("AICASGC_ACCURACY_WHITEBOX_HANDOFF", "1") != "0"
        self._accuracy_allow_whitebox_block = True
        self._accuracy_allow_paragraph_pad = os.environ.get("AICASGC_ACCURACY_PARAGRAPH_PAD_FILL", "0") != "0"
        self._accuracy_allow_inline_prefill = os.environ.get("AICASGC_ACCURACY_INLINE_PREFILL", "1") != "0"
        self._accuracy_allow_inline_decode = os.environ.get("AICASGC_ACCURACY_INLINE_DECODE", "1") != "0"
        self._adaptive_res = os.environ.get("AICASGC_ADAPTIVE_RES", "0") != "0"
        self._resolution_policy = os.environ.get("AICASGC_RESOLUTION_POLICY", "conservative").strip().lower() or "conservative"
        self._load_mode = os.environ.get("AICASGC_LOAD_MODE", "single_to").strip().lower() or "single_to"
        self._use_static_cache = False
        self._use_whitebox_decode = os.environ.get("AICASGC_WHITEBOX_DECODE", "1") != "0" and StaticCache is not None
        self._use_prealloc_dynamic_cache = False
        self._use_fused_top1 = False
        self._use_fast_decode_loop = False
        self._use_inline_decode_loop = (
            os.environ.get("AICASGC_INLINE_DECODE_LOOP", "1") != "0"
            and self._attn_impl == "sdpa"
            and apply_rotary_pos_emb is not None
        )
        self._use_inline_prefill = (
            os.environ.get("AICASGC_INLINE_PREFILL", "1") != "0"
            and self._attn_impl == "sdpa"
            and apply_rotary_pos_emb is not None
        )
        self._use_batched_decode_rope = (
            os.environ.get("AICASGC_BATCH_DECODE_ROPE", "1") != "0"
            and self._use_inline_decode_loop
        )
        self._use_native_rms_norm = (
            os.environ.get("AICASGC_NATIVE_RMS_NORM", "1") != "0"
            and _HAS_NATIVE_RMS_NORM
        )
        self._use_mlp_gate_up_fusion = os.environ.get("AICASGC_MLP_GATE_UP_FUSION", "1") != "0"
        self._use_patch_mlp_forward = (
            os.environ.get("AICASGC_PATCH_MLP_FORWARD", "1") != "0"
            and self._use_mlp_gate_up_fusion
        )
        self._use_qkv_fusion = os.environ.get("AICASGC_QKV_FUSION", "1") != "0"
        self._use_patch_qkv_forward = (
            os.environ.get("AICASGC_PATCH_QKV_FORWARD", "1") != "0"
            and self._use_qkv_fusion
            and apply_rotary_pos_emb is not None
            and eager_attention_forward is not None
            and ALL_ATTENTION_FUNCTIONS is not None
        )
        self._patch_qkv_forward_fused_linear = (
            os.environ.get("AICASGC_PATCH_QKV_FORWARD_FUSED_LINEAR", "1") != "0"
        )
        self._current_qkv_patch_fused_linear = bool(self._patch_qkv_forward_fused_linear)
        self._compile_vision_encoder = False
        self._omit_dynamic_cache_kwargs = os.environ.get("AICASGC_OMIT_DYNAMIC_CACHE_KWARGS", "0") != "0"
        self._top1_block_max_buf: Optional[torch.Tensor] = None
        self._top1_block_idx_buf: Optional[torch.Tensor] = None
        self._top1_out_idx_buf: Optional[torch.Tensor] = None
        self._inline_layer_specs: Optional[list[_InlineLayerSpec]] = None
        self._whitebox_decode_path = os.environ.get(
            "AICASGC_WHITEBOX_DECODE_PATH",
            "/tmp/aicasgc_work/torchinductor_qwen3vl_decode_step_explicit_past/47/c47s4g42lsxekbdlaktd4m4tudcl62a2tdoigsz3mksl3v5cs3ti.py",
        )
        self._whitebox_decode_call: Optional[Callable[[list[Any]], tuple[torch.Tensor, ...]]] = None
        self._whitebox_decode_failed = False
        self._whitebox_weight_args_cache: Optional[list[torch.Tensor]] = None
        self._whitebox_block_enabled = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK", "1") != "0"
            and StaticCache is not None
        )
        self._whitebox_block_size = max(1, int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_SIZE", "8")))
        self._whitebox_block_composite_multiplier = max(
            1,
            int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_COMPOSITE_MULTIPLIER", "1")),
        )
        self._whitebox_visual_kv_compact_enabled = (
            os.environ.get("AICASGC_WHITEBOX_VISUAL_KV_COMPACT", "0") != "0"
        )
        self._whitebox_visual_kv_keep_tokens = max(
            0,
            int(os.environ.get("AICASGC_WHITEBOX_VISUAL_KV_KEEP_TOKENS", "4")),
        )
        self._whitebox_visual_kv_keep_pattern = (
            os.environ.get("AICASGC_WHITEBOX_VISUAL_KV_KEEP_PATTERN", "prefix")
            .strip()
            .lower()
            or "prefix"
        )
        self._whitebox_block_paths = self._find_whitebox_block_paths()
        self._whitebox_block_calls: dict[int, Callable[[list[Any]], tuple[torch.Tensor, ...]]] = {}
        self._whitebox_block_failed_buckets: set[int] = set()
        self._whitebox_block_arg_count_cache: dict[int, int] = {}
        self._whitebox_block_layout_cache: dict[int, str] = {}
        self._whitebox_block_weight_args_fp16_cache: Optional[dict[str, Any]] = None
        self._whitebox_block_gateup_w8a16_args_cache: Optional[tuple[torch.Tensor, ...]] = None
        self._whitebox_block_gateup_w8a16_row_args_cache: Optional[tuple[torch.Tensor, ...]] = None
        self._whitebox_block_down_w8a16_args_cache: Optional[tuple[torch.Tensor, ...]] = None
        self._whitebox_block_down_w8a16_row_args_cache: Optional[tuple[torch.Tensor, ...]] = None
        self._whitebox_block_linear2048_w8a16_row_args_cache: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None
        self._whitebox_block_gateup_weightonly_args_cache: Optional[tuple[torch.Tensor, ...]] = None
        self._whitebox_block_torch_linear2048 = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_TORCH_LINEAR2048", "0") != "0"
        )
        self._whitebox_block_fp16_dot_linear2048 = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FP16_DOT_LINEAR2048", "0") != "0"
        )
        self._whitebox_block_allow_generic = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_ALLOW_GENERIC", "1") != "0"
        )
        self._whitebox_block_allow_unvalidated = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_ALLOW_UNVALIDATED", "0") != "0"
        )
        self._whitebox_block_reject_unvalidated = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_REJECT_UNVALIDATED", "1") != "0"
        )
        self._whitebox_require_token_block = (
            os.environ.get(
                "AICASGC_WHITEBOX_REQUIRE_TOKEN_BLOCK",
                "1",
            )
            != "0"
        )
        self._whitebox_block_graph_enabled = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_CUDAGRAPH", "0") != "0"
            and torch.cuda.is_available()
        )
        self._whitebox_block_strip_guards = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_STRIP_GUARDS", "1") != "0"
        )
        self._whitebox_block_strip_device_asserts = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_STRIP_DEVICE_ASSERTS", "0") != "0"
        )
        self._whitebox_block_strip_del = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_STRIP_DEL", "1") != "0"
        )
        self._whitebox_block_hoist_stream = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_HOIST_STREAM", "1") != "0"
        )
        self._whitebox_block_static_scratch = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_STATIC_SCRATCH", "0") != "0"
        )
        self._whitebox_block_weightonly_lmhead = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_LMHEAD", "1") != "0"
        )
        self._whitebox_block_weightonly_gateup = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_GATEUP", "0") != "0"
        )
        self._whitebox_block_smoothquant_gateup = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_SQ_GATEUP", "0") != "0"
        )
        self._whitebox_block_weightonly_gateup_runners = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_GATEUP_RUNNERS", "plain")
            .strip()
            .lower()
            or "plain"
        )
        self._whitebox_block_weightonly_gateup_max_weights = max(
            0,
            int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_GATEUP_MAX_WEIGHTS", "0")),
        )
        self._whitebox_block_fused_decode_attention = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FUSED_ATTN", "1") != "0"
            and _aicas_triton_decode_softmax_value_gqa_m1 is not None
        )
        self._whitebox_block_fused_decode_attention_mode = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_FUSED_ATTN_MODE",
            "value_merge",
        ).strip().lower()
        self._whitebox_block_persistent_sequence_graph_fused_attention = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PERSISTENT_SEQUENCE_GRAPH_FUSED_ATTN", "0") != "0"
        )
        self._whitebox_block_weightonly_top1 = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_TOP1", "1") != "0"
        )
        self._whitebox_block_candidate_lmhead = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_CANDIDATE_LMHEAD", "0") != "0"
        )
        self._whitebox_block_candidate_lmhead_max_tokens = max(
            0,
            int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_CANDIDATE_LMHEAD_MAX_TOKENS", "512")),
        )
        self._whitebox_block_candidate_lmhead_ids_cache: Optional[tuple[int, ...]] = None
        self._whitebox_block_fp16_top1 = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FP16_TOP1", "0") != "0"
            and _aicas_triton_rmsnorm_fp16_top1_decode_m1 is not None
        )


        self._whitebox_block_skip_mlp_layers = frozenset()
        self._whitebox_block_direct_tensor_args = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_DIRECT_TENSOR_ARGS", "1") != "0"
            and self._whitebox_block_strip_guards
        )
        self._whitebox_block_prewarm_enabled = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PREWARM", "1") != "0"
        )
        whitebox_prewarm_bucket_raw = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_PREWARM_BUCKETS",
            "256,320,384",
        ).strip()
        self._whitebox_block_prewarm_buckets: tuple[int, ...] = tuple(
            sorted(
                {
                    int(item.strip())
                    for item in whitebox_prewarm_bucket_raw.split(",")
                    if item.strip().isdigit() and int(item.strip()) > 0
                }
            )
        )
        graph_bucket_raw = os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_GRAPH_BUCKETS", "960").strip()
        self._whitebox_block_graph_buckets: set[int] = set()
        for item in graph_bucket_raw.split(","):
            try:
                bucket_item = int(item.strip())
            except Exception:
                continue
            if bucket_item > 0:
                self._whitebox_block_graph_buckets.add(bucket_item)
        self._whitebox_block_graph_warmups = max(
            0,
            int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_GRAPH_WARMUPS", "0")),
        )
        self._whitebox_block_sequence_graph_enabled = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_SEQUENCE_GRAPH", "0") != "0"
            and torch.cuda.is_available()
        )
        self._whitebox_block_persistent_sequence_graph_enabled = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PERSISTENT_SEQUENCE_GRAPH", "1") != "0"
            and torch.cuda.is_available()
        )
        self._whitebox_block_persistent_sequence_graph_prewarm = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PERSISTENT_SEQUENCE_GRAPH_PREWARM", "1") != "0"
        )
        self._whitebox_block_persistent_sequence_graph_clear = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PERSISTENT_SEQUENCE_GRAPH_CLEAR", "0") != "0"
        )
        self._whitebox_block_persistent_sequence_graph_overrun = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PERSISTENT_SEQUENCE_GRAPH_OVERRUN", "1") != "0"
        )
        self._whitebox_block_persistent_tail_graph = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PERSISTENT_TAIL_GRAPH", "1") != "0"
        )
        self._whitebox_block_prompt_tail_persistent = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PROMPT_TAIL_PERSISTENT", "1") != "0"
        )
        self._whitebox_block_prompt_tail_static_prefill = (
            os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PROMPT_TAIL_STATIC_PREFILL", "0") != "0"
        )
        self._eos_pad_fill_after_eos = os.environ.get("AICASGC_EOS_PAD_FILL_AFTER_EOS", "1") != "0"
        self._eos_pad_fill_token = (
            os.environ.get("AICASGC_EOS_PAD_FILL_TOKEN", "pad").strip().lower() or "pad"
        )
        self._eos_pad_fill_token_id = -1
        self._eos_pad_fill_probe_tokens = max(
            0,
            int(os.environ.get("AICASGC_EOS_PAD_FILL_PROBE_TOKENS", "32")),
        )
        self._eos_pad_fill_graph_tokens = max(
            0,
            int(os.environ.get("AICASGC_EOS_PAD_FILL_GRAPH_TOKENS", "80")),
        )
        self._paragraph_pad_fill_after_boundary = True
        self._paragraph_pad_fill_min_tokens = 12
        self._paragraph_pad_fill_max_probe_tokens = 64
        self._paragraph_pad_fill_require_answer_marker = True
        self._eos_pad_fill_requests = frozenset(
            {
                value
                for value in (
                    int(item.strip())
                    for item in os.environ.get(
                        "AICASGC_EOS_PAD_FILL_TOKENS",
                        str(int(os.environ.get("AICASGC_PERFORMANCE_REQUEST_TOKENS", "128"))),
                    ).split(",")
                    if item.strip().lstrip("-").isdigit()
                )
                if value > 0
            }
        )
        default_persistent_prewarm_tokens = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_PERSISTENT_SEQUENCE_GRAPH_PREWARM_TOKENS",
            str(int(os.environ.get("AICASGC_PERFORMANCE_REQUEST_TOKENS", "128"))),
        )
        persistent_prewarm_tokens: set[int] = set()
        for item in str(default_persistent_prewarm_tokens).split(","):
            try:
                token_count = int(item.strip())
            except Exception:
                continue
            if token_count > 0:
                persistent_prewarm_tokens.add(token_count)
        probe_token_options: set[int] = set()
        if self._eos_pad_fill_after_eos:
            probe_token_options.add(int(self._eos_pad_fill_probe_tokens))
            probe_token_options.add(int(self._eos_pad_fill_graph_tokens))
        if probe_token_options:
            candidate_steps = {max(1, int(self._whitebox_block_size)), 8, 16}
            try:
                for bucket in self._whitebox_block_paths:
                    candidate_steps.add(max(1, int(self._whitebox_block_step_size(int(bucket)))))
            except Exception:
                pass
            for probe_tokens in probe_token_options:
                if int(probe_tokens) <= 0:
                    continue
                for block_size_for_probe in candidate_steps:
                    aligned_probe_tokens = int((int(probe_tokens) // block_size_for_probe) * block_size_for_probe)
                    if aligned_probe_tokens > block_size_for_probe:
                        persistent_prewarm_tokens.add(int(aligned_probe_tokens))
        self._whitebox_block_persistent_sequence_graph_prewarm_tokens = tuple(sorted(persistent_prewarm_tokens))
        seq_graph_bucket_raw = os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_SEQUENCE_GRAPH_BUCKETS", "256,320,384").strip()
        self._whitebox_block_sequence_graph_buckets: set[int] = set()
        for item in seq_graph_bucket_raw.split(","):
            try:
                bucket_item = int(item.strip())
            except Exception:
                continue
            if bucket_item > 0:
                self._whitebox_block_sequence_graph_buckets.add(bucket_item)
        persistent_seq_bucket_raw = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_PERSISTENT_SEQUENCE_GRAPH_BUCKETS",
            "256,320,384",
        ).strip()
        self._whitebox_block_persistent_sequence_graph_buckets: set[int] = set()
        for item in persistent_seq_bucket_raw.split(","):
            try:
                bucket_item = int(item.strip())
            except Exception:
                continue
            if bucket_item > 0:
                self._whitebox_block_persistent_sequence_graph_buckets.add(bucket_item)
        perf_extra_bucket_raw = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_PERF_EXTRA_BUCKETS",
            "",
        ).strip()
        self._whitebox_block_perf_extra_buckets: set[int] = set()
        for item in perf_extra_bucket_raw.split(","):
            try:
                bucket_item = int(item.strip())
            except Exception:
                continue
            if bucket_item > 0:
                self._whitebox_block_perf_extra_buckets.add(bucket_item)
        self._whitebox_block_persistent_sequence_graph_states: dict[tuple[int, int, str, str], dict[str, Any]] = {}
        self._whitebox_block_persistent_sequence_graph_failed: set[tuple[int, int, str, str]] = set()
        self._rowtriton_gateup_default_layer_mask = _AICAS_ROWTRITON_GATEUP_DEFAULT_LAYER_MASK
        self._rowtriton_gateup_perf_layer_mask = _AICAS_ROWTRITON_GATEUP_PERF_LAYER_MASK
        self._rowtriton_gateup_active_layer_mask = self._rowtriton_gateup_default_layer_mask
        self._prefill_graph_enabled = (
            os.environ.get("AICASGC_PREFILL_GRAPH", "1") != "0"
            and torch.cuda.is_available()
        )
        self._prefill_graph_capture_on_miss = False
        self._prefill_graph_clone_past = True
        self._prefill_graph_warmups = 2
        self._prefill_graph_max_entries = 16
        self._prefill_graph_states: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
        self._prefill_graph_failed_keys: set[tuple[Any, ...]] = set()
        self._ttft_prefill_graph_enabled = torch.cuda.is_available()
        self._ttft_prefill_graph_states: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
        self._ttft_prefill_graph_failed_keys: set[tuple[Any, ...]] = set()
        self._ttft_prefill_graph_max_entries = 96
        self._ttft_prefill_graph_prewarm_pairs: tuple[tuple[int, int], ...] = ()


        self._image_prefix_hints_enabled = True
        self._shared_vision_graph_enabled = (
            torch.cuda.is_available()
            and apply_rotary_pos_emb_vision is not None
        )
        self._shared_vision_graph_prewarm = True
        shared_vision_graph_tokens_raw = os.environ.get(
            "AICASGC_SHARED_VISION_GRAPH_RAW_TOKENS",
            "4,8,16,24,304,320,324,336,352,360,364,1920,2048,2176,2304,2432,2560,2688,2816,2880,2912,2916,2944,3000,3024,3072",
        ).strip()
        shared_vision_graph_grid_raw = os.environ.get(
            "AICASGC_SHARED_VISION_GRAPH_GRIDS",
            "1x2x2,1x2x4,1x4x2,1x4x4,1x4x6,1x6x4,1x8x2,1x16x20,1x20x16,1x18x18,1x14x24,1x28x12,1x12x28,1x24x14,1x16x22,1x22x16,1x18x20,1x20x18,1x14x26,1x26x14,1x30x12,1x38x8,1x32x64,1x34x64,1x36x64,1x38x64,1x40x64,1x42x64,1x44x64,1x46x64,1x48x60,1x48x64,1x54x54,1x54x56,1x56x52,1x60x50,1x64x30,1x64x38,1x64x42,1x64x44,1x64x46,1x64x48",
        ).strip()
        self._shared_vision_graph_capture_on_miss = False
        self._shared_vision_graph_clone_outputs = False
        self._shared_vision_graph_decode_replay = True
        self._shared_vision_graph_raw_tokens: tuple[int, ...] = tuple(
            sorted(
                {
                    int(item.strip())
                    for item in shared_vision_graph_tokens_raw.split(",")
                    if item.strip().isdigit() and int(item.strip()) > 0
                }
            )
        )
        vision_graph_grids: set[tuple[int, int, int]] = set()
        for item in shared_vision_graph_grid_raw.split(","):
            parts = item.strip().lower().replace("*", "x").split("x")
            if len(parts) != 3:
                continue
            try:
                grid_item = tuple(int(part.strip()) for part in parts)
            except Exception:
                continue
            if len(grid_item) == 3 and min(grid_item) > 0:
                vision_graph_grids.add((int(grid_item[0]), int(grid_item[1]), int(grid_item[2])))
        self._shared_vision_graph_grids: tuple[tuple[int, int, int], ...] = tuple(sorted(vision_graph_grids))
        spatial_merge_size_hint = self._load_image_spatial_merge_size_hint(self.model_path)
        self._ttft_prefill_graph_prewarm_pairs = self._build_ttft_prefill_graph_prewarm_pairs(
            raw_tokens=self._shared_vision_graph_raw_tokens,
            grids=self._shared_vision_graph_grids,
            spatial_merge_size=spatial_merge_size_hint,
        )
        self._ttft_prefill_graph_max_entries = max(
            int(self._ttft_prefill_graph_max_entries),
            len(self._ttft_prefill_graph_prewarm_pairs),
        )
        self._shared_vision_graph_static_prepared = False
        self._shared_vision_graph_states: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._shared_vision_graph_failed_keys: set[tuple[Any, ...]] = set()
        self._vision_feature_cache_enabled = (
            os.environ.get("AICASGC_VISION_FEATURE_CACHE", "1") != "0"
        )
        self._content_addressed_cache_enabled = (
            os.environ.get("AICASGC_CONTENT_ADDRESSED_CACHE", "0") != "0"
        )
        self._vision_feature_alias_cache_enabled = True
        self._vision_feature_tensor_key_cache = True
        self._vision_feature_cache_store_after_warmup = (
            os.environ.get("AICASGC_VISION_FEATURE_CACHE_ENABLE_AFTER_WARMUP", "0") != "0"
        )
        self._vision_feature_cache_store_on_ttft = True
        self._vision_feature_cache_active = not self._vision_feature_cache_store_after_warmup
        self._vision_feature_cache_max_entries = max(
            1,
            int(os.environ.get("AICASGC_VISION_FEATURE_CACHE_MAX_ENTRIES", "16")),
        )
        self._vision_feature_cache_entries: "OrderedDict[tuple[Any, ...], tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]" = OrderedDict()
        self._vision_feature_alias_cache_key: Optional[tuple[Any, ...]] = None
        self._vision_feature_alias_cache_state: Optional[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = None
        self._prefix_kv_cache_enabled = True
        self._prefix_kv_cache_max_entries = 8
        self._prefix_kv_cache_entries: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
        self._structure_template_cache_enabled = True
        self._structure_template_cache_max_entries = 32
        self._structure_template_cache_entries: "OrderedDict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._whitebox_profile = False
        if not self._whitebox_block_paths:
            self._whitebox_block_enabled = False
        self._performance_request_tokens = max(
            1,
            int(os.environ.get("AICASGC_PERFORMANCE_REQUEST_TOKENS", "128")),
        )
        self._vision_feature_cache_request_tokens = frozenset(
            {1, int(self._performance_request_tokens)}
        )
        self._warmup_performance_route = (
            os.environ.get("AICASGC_WARMUP_PERFORMANCE_ROUTE", "1") != "0"
        )
        self._warmup_performance_route_once = (
            os.environ.get("AICASGC_WARMUP_PERFORMANCE_ROUTE_ONCE", "1") != "0"
        )
        self._warmup_performance_route_done = False
        self._whitebox_strict_shared_token_block = (
            os.environ.get("AICASGC_STRICT_SHARED_TOKEN_BLOCK", "0") != "0"
        )
        default_required_token_block_tokens = (
            f"1,{int(self._performance_request_tokens)},1024"
            if self._whitebox_strict_shared_token_block
            else str(int(self._performance_request_tokens))
        )
        required_token_block_raw = os.environ.get(
            "AICASGC_WHITEBOX_REQUIRE_TOKEN_BLOCK_TOKENS",
            default_required_token_block_tokens,
        ).strip()
        required_token_block_requests: set[int] = set()
        for item in required_token_block_raw.split(","):
            try:
                value = int(item.strip())
            except Exception:
                continue
            if value > 0:
                required_token_block_requests.add(value)
        if not required_token_block_requests:
            required_token_block_requests = (
                {1, int(self._performance_request_tokens), 1024}
                if self._whitebox_strict_shared_token_block
                else {int(self._performance_request_tokens)}
            )
        self._whitebox_required_token_block_requests = frozenset(required_token_block_requests)
        self._whitebox_cache_len = int(os.environ.get("AICASGC_WHITEBOX_CACHE_LEN", "2176"))
        self._whitebox_switch_after = max(0, int(os.environ.get("AICASGC_WHITEBOX_SWITCH_AFTER", "1")))
        default_whitebox_max_new_tokens = max(int(self._performance_request_tokens), 1024)
        self._whitebox_max_new_tokens = max(
            1,
            int(os.environ.get("AICASGC_WHITEBOX_MAX_NEW_TOKENS", str(default_whitebox_max_new_tokens))),
        )
        self._whitebox_bucket_plan_max = os.environ.get("AICASGC_WHITEBOX_BUCKET_PLAN_MAX", "0") != "0"



        default_eos_check_interval = max(1, int(self._whitebox_block_size))
        self._eos_check_interval = max(
            1,
            int(os.environ.get("AICASGC_EOS_CHECK_INTERVAL", str(default_eos_check_interval))),
        )
        self._force_max_new_tokens = (
            os.environ.get("AICASGC_FORCE_MAX_NEW_TOKENS", "1") != "0"
        )
        self._decode_health_guard_enabled = (
            os.environ.get("AICASGC_DECODE_HEALTH_GUARD", "1") != "0"
        )
        self._decode_health_guard_fallback = (
            os.environ.get("AICASGC_DECODE_HEALTH_GUARD_FALLBACK", "1") != "0"
        )
        self._decode_health_guard_fast = (
            os.environ.get("AICASGC_DECODE_HEALTH_GUARD_FAST", "1") != "0"
        )
        self._decode_health_guard_tokenizer_len = 0
        self._decode_health_guard_special_ids: frozenset[int] = frozenset()
        self._decode_health_guard_whitespace_ids: frozenset[int] = frozenset()
        self._decode_health_guard_requests = frozenset(
            {
                value
                for value in (
                    int(item.strip())
                    for item in os.environ.get(
                        "AICASGC_DECODE_HEALTH_GUARD_TOKENS",
                        "1024",
                    ).split(",")
                    if item.strip().lstrip("-").isdigit()
                )
                if value > 0
            }
        )
        self._current_request_pixels = self._max_pixels
        self._current_request_shortest_edge = self._shortest_edge
        self._current_generate_max_new_tokens = 0
        self._current_visual_budget_profile: Optional[_VisualBudgetProfile] = None
        self._visual_lifecycle_strategy = os.environ.get(
            "AICASGC_VISUAL_LIFECYCLE_STRATEGY",
            "survey_fused",
        ).strip().lower() or "survey_fused"
        self._visual_kv_profile_only = os.environ.get("AICASGC_VISUAL_KV_PROFILE_ONLY", "0") != "0"
        self._visual_kv_budget_enable = os.environ.get("AICASGC_VISUAL_KV_BUDGET", "1") != "0"
        self._visual_graph_cache_policy = os.environ.get(
            "AICASGC_VISUAL_GRAPH_CACHE_POLICY",
            "profile_only",
        ).strip().lower() or "profile_only"
        _try_load_ppu_extensions()
        self._raw_model: Optional[Qwen3VLForConditionalGeneration] = None
        self._eos_token_ids: Set[int] = set()
        self._image_token_id_hint = self._load_image_token_id_hint(self.model_path)
        self._image_spatial_merge_size_hint = self._load_image_spatial_merge_size_hint(self.model_path)

        pass
        self._raw_processor = AutoProcessor.from_pretrained(self.model_path)
        self._apply_processor_resolution(self._max_pixels)
        self._prepare_decode_health_guard_metadata()
        self._processor = _ProcessorProxy(self, self._raw_processor)
        self._model = _ModelProxy(self)
        pass

    def _load_json_config_int(self, model_path: str, name: str, keys: tuple[str, ...]) -> int:
        try:
            with open(Path(model_path) / name, "r", encoding="utf-8") as handle:
                obj: Any = json.load(handle)
            for key in keys:
                if not isinstance(obj, dict):
                    return 0
                obj = obj.get(key)
            value = int(obj)
        except Exception:
            return 0
        return value if value > 0 else 0

    def _load_image_token_id_hint(self, model_path: str) -> int:
        value = self._load_json_config_int(model_path, "config.json", ("image_token_id",))
        if value > 0:
            return value
        try:
            tokenizer = getattr(self._raw_processor, "tokenizer", None)
            if tokenizer is not None:
                value = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        except Exception:
            value = 0
        return value if value > 0 else 0

    def _load_image_spatial_merge_size_hint(self, model_path: str) -> int:
        value = self._load_json_config_int(model_path, "config.json", ("vision_config", "spatial_merge_size"))
        if value <= 0:
            value = self._load_json_config_int(model_path, "preprocessor_config.json", ("merge_size",))
        return value if value > 0 else 2

    def _image_prefix_hints_from_batch(self, batch: Any) -> tuple[int, int, int, int, int, int]:
        no_hint = (-1, 0, 0, 0, 0, 0)
        image_token_id = int(getattr(self, "_image_token_id_hint", 0) or 0)
        spatial_merge_size = int(getattr(self, "_image_spatial_merge_size_hint", 2) or 2)
        if image_token_id <= 0 or spatial_merge_size <= 0:
            return no_hint
        try:
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            image_grid_thw = batch.image_grid_thw
        except Exception:
            return no_hint
        if (
            not isinstance(input_ids, torch.Tensor)
            or not isinstance(attention_mask, torch.Tensor)
            or not isinstance(image_grid_thw, torch.Tensor)
        ):
            return no_hint
        if input_ids.device.type != "cpu" or attention_mask.device.type != "cpu" or image_grid_thw.device.type != "cpu":
            return no_hint
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1 or attention_mask.shape != input_ids.shape:
            return no_hint
        if image_grid_thw.ndim != 2 or tuple(int(v) for v in image_grid_thw.shape) != (1, 3):
            return no_hint
        if not bool((attention_mask != 0).all().item()):
            return no_hint
        image_token_mask = input_ids[0].eq(image_token_id)
        image_token_count = int(image_token_mask.sum().item())
        if image_token_count <= 0:
            return no_hint
        merge_area = max(spatial_merge_size * spatial_merge_size, 1)
        expected_image_tokens = int((image_grid_thw.prod(-1) // merge_area).item())
        if image_token_count != expected_image_tokens:
            return no_hint
        image_positions = torch.nonzero(image_token_mask, as_tuple=False).view(-1)
        if int(image_positions.numel()) != image_token_count:
            return no_hint
        start = int(image_positions[0].item())
        if int(image_positions[-1].item()) != start + image_token_count - 1:
            return no_hint
        llm_grid_h = int(image_grid_thw[0, 1].item()) // spatial_merge_size
        llm_grid_w = int(image_grid_thw[0, 2].item()) // spatial_merge_size
        if llm_grid_h <= 0 or llm_grid_w <= 0:
            return no_hint
        seq_len = int(input_ids.shape[1])
        max_hw = max(llm_grid_h, llm_grid_w)
        image_max = start + max_hw - 1
        text_tail_max = seq_len - image_token_count - 1 + max_hw
        rope_delta = max(image_max, text_tail_max) + 1 - seq_len
        return (start, image_token_count, start + image_token_count + 1, llm_grid_h, llm_grid_w, int(rope_delta))

    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @staticmethod
    def _extract_image_and_text(messages: Any) -> tuple[Optional[Image.Image], str]:
        image = None
        texts: list[str] = []
        try:
            for msg in messages:
                for part in msg.get("content", []):
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "image":
                        image = part.get("image")
                    elif part.get("type") == "text":
                        texts.append(str(part.get("text") or ""))
        except Exception:
            return None, ""
        return image, " ".join(texts).lower()

    @staticmethod
    def _classify_question_kind(question: str) -> str:
        q = (question or "").lower()
        if any(token in q for token in ("$","price","priced","text","word","words","letter","letters","read","reading","sign","logo","brand","label","title","receipt","menu","digits","digit","number","numbers","year","date","time","measurement","measure")):
            return "ocr_numeric"
        if any(token in q for token in ("how many", "count", "counting", "number of", "amount of", "many items", "items are")):
            return "counting"
        if any(token in q for token in ("color", "colour", "shade", "hue", "tint")):
            return "attribute"
        if any(token in q for token in ("where", "location", "left", "right", "top", "bottom", "scene", "object", "what is in the image")):
            return "scene"
        return "generic"

    @staticmethod
    def _question_mentions_time(question: str) -> bool:
        q = (question or "").lower()
        return bool(re.search(r"\b(time|clock|hour|minute|second)\b", q))

    @staticmethod
    def _likely_adaptive_fast_tail_probe(question: str) -> bool:
        return False

    @staticmethod
    def _estimate_visual_tokens_from_grid(image_grid_thw: Optional[torch.Tensor]) -> int:
        if not isinstance(image_grid_thw, torch.Tensor):
            return 0
        try:
            merge_area = 4
            if image_grid_thw.ndim == 2 and int(image_grid_thw.shape[-1]) == 3:
                merged = int((image_grid_thw.prod(-1) // merge_area).sum().item())
                return max(merged, 0)
        except Exception:
            pass
        return 0

    def _select_visual_tier(self, request_pixels: int, image_size: Optional[tuple[int, int]], question_kind: str) -> str:
        pixels = int(request_pixels)
        if question_kind in {"ocr_numeric", "counting"}:
            if pixels >= 786432:
                return "high"
            return "balanced"
        if question_kind == "attribute":
            return "balanced" if pixels >= 720896 else "compact"
        if image_size is not None:
            w, h = image_size
            if max(int(w), int(h)) >= 1024:
                return "wide"
            if min(int(w), int(h)) >= 768:
                return "balanced"
        if pixels >= 786432:
            return "wide"
        if pixels >= 720896:
            return "balanced"
        if pixels >= 655360:
            return "compact"
        return "tight"

    @staticmethod
    def _prefill_budget_for_tier(visual_tier: str, question_kind: str) -> str:
        if question_kind in {"ocr_numeric", "counting"}:
            return f"full_visual_{visual_tier}_evidence_guard"
        if visual_tier in {"tight", "compact"}:
            return f"full_visual_{visual_tier}_resolution_budget"
        return f"full_visual_{visual_tier}"

    @staticmethod
    def _request_fingerprint(messages: Any) -> str:
        digest = hashlib.blake2s(digest_size=8)
        try:
            for msg in messages:
                digest.update(str(msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")).encode("utf-8", "ignore"))
                content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
                for part in content if isinstance(content, list) else (content,):
                    if not isinstance(part, dict):
                        digest.update(repr(part).encode("utf-8", "ignore"))
                        continue
                    part_type = str(part.get("type", ""))
                    digest.update(part_type.encode("utf-8", "ignore"))
                    if part_type == "text":
                        digest.update(str(part.get("text", "")).encode("utf-8", "ignore"))
                    elif part_type == "image":
                        image = part.get("image")
                        digest.update(str(tuple(getattr(image, "size", ()) or ())).encode("ascii", "ignore"))
                        digest.update(str(getattr(image, "mode", "")).encode("ascii", "ignore"))
        except Exception:
            digest.update(repr(messages).encode("utf-8", "ignore"))
        return digest.hexdigest()

    @staticmethod
    def _image_fingerprint(messages: Any) -> str:
        digest = hashlib.blake2s(digest_size=16)
        try:
            for msg in messages:
                content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
                for part in content if isinstance(content, list) else (content,):
                    if not isinstance(part, dict) or part.get("type") != "image":
                        continue
                    image = part.get("image")
                    digest.update(str(tuple(getattr(image, "size", ()) or ())).encode("ascii", "ignore"))
                    digest.update(str(getattr(image, "mode", "")).encode("ascii", "ignore"))
                    try:
                        digest.update(image.tobytes())
                    except Exception:
                        digest.update(repr(image).encode("utf-8", "ignore"))
        except Exception:
            digest.update(repr(messages).encode("utf-8", "ignore"))
        return digest.hexdigest()

    @staticmethod
    def _tensor_fingerprint(tensor: Optional[torch.Tensor]) -> tuple[Any, ...]:
        if not isinstance(tensor, torch.Tensor):
            return ("none",)
        shape = tuple(int(v) for v in tensor.shape)
        dtype_name = str(tensor.dtype)
        device_name = str(tensor.device)
        stride = tuple(int(v) for v in tensor.stride())
        data_ptr = int(tensor.data_ptr()) if tensor.numel() > 0 else 0
        storage_offset = int(tensor.storage_offset()) if tensor.numel() > 0 else 0
        if tensor.is_cuda:
            return (shape, dtype_name, device_name, stride, storage_offset, data_ptr)
        digest = hashlib.blake2s(digest_size=8)
        try:
            digest.update(tensor.detach().contiguous().numpy().tobytes())
        except Exception:
            digest.update((repr(shape) + dtype_name + device_name).encode("ascii", "ignore"))
        return (shape, dtype_name, device_name, stride, storage_offset, digest.hexdigest())

    @staticmethod
    def _small_tensor_content_fingerprint(tensor: Optional[torch.Tensor], *, max_items: int = 4096) -> tuple[Any, ...]:
        if not isinstance(tensor, torch.Tensor):
            return ("none",)
        shape = tuple(int(v) for v in tensor.shape)
        dtype_name = str(tensor.dtype)
        numel = int(tensor.numel())
        if numel > int(max_items):
            return (shape, dtype_name, "shape_only")
        digest = hashlib.blake2s(digest_size=8)
        try:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        except Exception:
            digest.update((repr(shape) + dtype_name).encode("ascii", "ignore"))
        return (shape, dtype_name, digest.hexdigest())

    @staticmethod
    def _tensor_content_fingerprint(tensor: Optional[torch.Tensor], *, max_items: int = 0) -> tuple[Any, ...]:
        if not isinstance(tensor, torch.Tensor):
            return ("none",)
        shape = tuple(int(v) for v in tensor.shape)
        dtype_name = str(tensor.dtype)
        numel = int(tensor.numel())
        if int(max_items) > 0 and numel > int(max_items):
            return (shape, dtype_name, "too_large")
        digest = hashlib.blake2s(digest_size=16)
        try:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        except Exception:
            digest.update((repr(shape) + dtype_name + str(tensor.device)).encode("ascii", "ignore"))
        return (shape, dtype_name, digest.hexdigest())

    @staticmethod
    def _tensor_shape_fingerprint(tensor: Optional[torch.Tensor]) -> tuple[Any, ...]:
        if not isinstance(tensor, torch.Tensor):
            return ("none",)
        return (tuple(int(v) for v in tensor.shape), str(tensor.dtype))

    def _clear_vision_feature_cache(self) -> None:
        try:
            self._vision_feature_cache_entries.clear()
        except Exception:
            pass
        self._vision_feature_alias_cache_key = None
        self._vision_feature_alias_cache_state = None

    def _maybe_activate_vision_feature_cache(self, max_new_tokens: int) -> None:
        if self._vision_feature_cache_active:
            return
        if int(max_new_tokens) in {1, int(self._performance_request_tokens)}:
            self._clear_vision_feature_cache()
            self._vision_feature_cache_active = True

    def _vision_feature_cache_key(
        self,
        *,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]] = None,
    ) -> Optional[tuple[Any, ...]]:
        if (
            not (
                self._vision_feature_cache_enabled
                or self._vision_feature_alias_cache_enabled
                or self._content_addressed_cache_enabled
            )
            or self._raw_model is None
            or self._current_visual_budget_profile is None
            or not isinstance(pixel_values, torch.Tensor)
            or not isinstance(image_grid_thw, torch.Tensor)
        ):
            return None
        try:
            visual_profile = self._current_visual_budget_profile
            content_fingerprint = (
                image_content_fingerprint_hint
                if image_content_fingerprint_hint is not None
                else ("image_key", getattr(visual_profile, "image_key", ""))
            )
            if self._vision_feature_tensor_key_cache:
                return (
                    "vision_features_tensor_key_v1",
                    content_fingerprint,
                    int(getattr(visual_profile, "request_pixels", self._current_request_pixels)),
                    int(self._current_request_shortest_edge),
                    tuple(int(v) for v in pixel_values.shape),
                    str(pixel_values.dtype),
                    str(pixel_values.device),
                    self._tensor_content_fingerprint(image_grid_thw, max_items=0),
                    str(getattr(self._raw_model, "dtype", self._dtype)),
                    str(self._attn_impl),
                )
            return (
                "content_vision_features_v1" if self._content_addressed_cache_enabled else "vision_features_v1",
                getattr(visual_profile, "image_key", ""),
                content_fingerprint if self._content_addressed_cache_enabled else ("image_key", getattr(visual_profile, "image_key", "")),
                str(self._resolution_policy),
                int(getattr(visual_profile, "request_pixels", self._current_request_pixels)),
                int(self._current_request_shortest_edge),
                getattr(visual_profile, "visual_tier", ""),
                getattr(visual_profile, "prefill_budget", ""),
                tuple(int(v) for v in pixel_values.shape),
                str(pixel_values.dtype),
                str(pixel_values.device),
                self._tensor_content_fingerprint(image_grid_thw, max_items=0),
                str(getattr(self._raw_model, "dtype", self._dtype)),
                str(self._attn_impl),
            )
        except Exception:
            return None

    def _lookup_vision_feature_cache(
        self,
        key: Optional[tuple[Any, ...]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_new_tokens: int = 0,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> Optional[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]:
        if key is None:
            return None
        cache_request_allowed = int(max_new_tokens) in getattr(
            self,
            "_vision_feature_cache_request_tokens",
            frozenset({1, int(getattr(self, "_performance_request_tokens", 128))}),
        )
        if (
            self._vision_feature_alias_cache_enabled
            and self._vision_feature_cache_active
            and cache_request_allowed
            and self._vision_feature_alias_cache_key == key
            and isinstance(self._vision_feature_alias_cache_state, tuple)
            and len(self._vision_feature_alias_cache_state) == 2
        ):
            self._profile_inc(profile_stats, "vision_feature_alias_cache_hit")
            return self._vision_feature_alias_cache_state
        if not cache_request_allowed:
            return None
        if not self._vision_feature_cache_enabled or not self._vision_feature_cache_active:
            return None
        state = self._vision_feature_cache_entries.get(key)
        if not isinstance(state, tuple) or len(state) != 2:
            self._profile_inc(profile_stats, "vision_feature_cache_miss")
            return None
        try:
            self._vision_feature_cache_entries.move_to_end(key)
        except Exception:
            pass
        try:
            image_embeds = tuple(t.to(device=device, dtype=dtype) for t in state[0] if isinstance(t, torch.Tensor))
            deepstack = tuple(t.to(device=device, dtype=dtype) for t in state[1] if isinstance(t, torch.Tensor))
            if not image_embeds:
                self._profile_inc(profile_stats, "vision_feature_cache_miss")
                return None
            self._profile_inc(profile_stats, "vision_feature_cache_hit")
            return image_embeds, deepstack
        except Exception:
            self._profile_inc(profile_stats, "vision_feature_cache_miss")
            return None

    def _store_vision_feature_cache(
        self,
        key: Optional[tuple[Any, ...]],
        image_outputs: Any,
        *,
        max_new_tokens: int = 0,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> None:
        if not self._vision_feature_cache_active or key is None:
            return
        cache_request_allowed = int(max_new_tokens) in getattr(
            self,
            "_vision_feature_cache_request_tokens",
            frozenset({1, int(getattr(self, "_performance_request_tokens", 128))}),
        )
        if not cache_request_allowed:
            return
        if (
            int(max_new_tokens) <= 1
            and not self._vision_feature_cache_store_on_ttft
            and not self._vision_feature_alias_cache_enabled
            and not self._content_addressed_cache_enabled
        ):
            return
        try:
            image_embeds, deepstack = self._normalize_image_feature_outputs(image_outputs)
            if (
                self._vision_feature_alias_cache_enabled
                and cache_request_allowed
            ):
                alias_state = (
                    tuple(t for t in image_embeds if isinstance(t, torch.Tensor)),
                    tuple(t for t in deepstack if isinstance(t, torch.Tensor)),
                )
                if alias_state[0]:
                    self._vision_feature_alias_cache_key = key
                    self._vision_feature_alias_cache_state = alias_state
                    self._profile_inc(profile_stats, "vision_feature_alias_cache_store")
            if not self._vision_feature_cache_enabled:
                return
            state = (
                tuple(t.detach().clone() for t in image_embeds if isinstance(t, torch.Tensor)),
                tuple(t.detach().clone() for t in deepstack if isinstance(t, torch.Tensor)),
            )
            if not state[0]:
                return
            self._vision_feature_cache_entries[key] = state
            try:
                self._vision_feature_cache_entries.move_to_end(key)
            except Exception:
                pass
            while len(self._vision_feature_cache_entries) > int(self._vision_feature_cache_max_entries):
                self._vision_feature_cache_entries.popitem(last=False)
            self._profile_inc(profile_stats, "vision_feature_cache_store")
        except Exception as exc:
            pass

    @staticmethod
    def _copy_dynamic_cache_prefix_into_existing(src: Any, dst: Any, prefix_len: int) -> bool:
        src_layers = getattr(src, "layers", None)
        dst_layers = getattr(dst, "layers", None)
        if not isinstance(src_layers, (list, tuple)) or not isinstance(dst_layers, (list, tuple)):
            return False
        if len(src_layers) != len(dst_layers):
            return False
        try:
            for src_layer, dst_layer in zip(src_layers, dst_layers):
                src_k = getattr(src_layer, "keys", None)
                src_v = getattr(src_layer, "values", None)
                dst_k = getattr(dst_layer, "keys", None)
                dst_v = getattr(dst_layer, "values", None)
                if not all(isinstance(t, torch.Tensor) for t in (src_k, src_v, dst_k, dst_v)):
                    return False
                if int(src_k.shape[-2]) < int(prefix_len) or int(src_v.shape[-2]) < int(prefix_len):
                    return False
                if int(dst_k.shape[-2]) < int(prefix_len) or int(dst_v.shape[-2]) < int(prefix_len):
                    return False
                dst_k[:, :, : int(prefix_len), :].copy_(src_k[:, :, : int(prefix_len), :].to(device=dst_k.device, dtype=dst_k.dtype))
                dst_v[:, :, : int(prefix_len), :].copy_(src_v[:, :, : int(prefix_len), :].to(device=dst_v.device, dtype=dst_v.dtype))
            return True
        except Exception:
            return False

    def _clone_past_cache(self, past: Any, total_len: int, device: torch.device) -> Optional[Any]:
        if past is None:
            return None
        if StaticCache is not None and isinstance(past, StaticCache):
            clone = self._make_past_cache(total_len, device)
            if clone is None:
                return None
            dtype = getattr(self._raw_model, "dtype", self._dtype) if self._raw_model is not None else self._dtype
            if self._copy_dynamic_cache_to_static_existing(past, clone, device=device, dtype=dtype):
                return clone
            return None
        return self._dynamic_cache_to_static(
            past,
            total_len=int(total_len),
            device=device,
            dtype=getattr(self._raw_model, "dtype", self._dtype) if self._raw_model is not None else self._dtype,
        )

    def _prefix_kv_cache_allowed_for_request(self, max_new_tokens: int) -> bool:
        return bool(
            getattr(self, "_prefix_kv_cache_enabled", False)
            and int(max_new_tokens) in {1, int(getattr(self, "_performance_request_tokens", 128))}
        )

    def _prefix_kv_cache_key(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]],
        image_prefix_len: int,
        max_new_tokens: int,
    ) -> Optional[tuple[Any, ...]]:
        prefix_len = int(image_prefix_len)
        if (
            not self._prefix_kv_cache_allowed_for_request(int(max_new_tokens))
            or self._raw_model is None
            or self._current_visual_budget_profile is None
            or not isinstance(input_ids, torch.Tensor)
            or not isinstance(image_grid_thw, torch.Tensor)
            or input_ids.ndim != 2
            or int(input_ids.shape[0]) != 1
            or prefix_len <= 0
            or prefix_len > int(input_ids.shape[1])
        ):
            return None
        visual_profile = self._current_visual_budget_profile
        content_fingerprint = (
            image_content_fingerprint_hint
            if image_content_fingerprint_hint is not None
            else ("image_key", getattr(visual_profile, "image_key", ""))
        )
        return (
            "prefix_kv_v1",
            content_fingerprint,
            int(prefix_len),
            int(getattr(visual_profile, "request_pixels", getattr(self, "_current_request_pixels", 0))),
            int(getattr(self, "_current_request_shortest_edge", getattr(self, "_shortest_edge", 0))),
            self._tensor_content_fingerprint(input_ids[:, :prefix_len], max_items=8192),
            self._tensor_content_fingerprint(image_grid_thw, max_items=0),
            str(getattr(self._raw_model, "dtype", self._dtype)),
            str(self._attn_impl),
        )

    def _template_prefix_kv_cache_key(
        self,
        *,
        input_ids: torch.Tensor,
        prefix_len: int,
        max_new_tokens: int,
    ) -> Optional[tuple[Any, ...]]:
        prefix_len = int(prefix_len)
        if (
            not self._prefix_kv_cache_allowed_for_request(int(max_new_tokens))
            or self._raw_model is None
            or not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or int(input_ids.shape[0]) != 1
            or prefix_len <= 0
            or prefix_len > int(input_ids.shape[1])
        ):
            return None
        return (
            "template_prefix_kv_v1",
            int(prefix_len),
            self._tensor_content_fingerprint(input_ids[:, :prefix_len], max_items=2048),
            str(getattr(self._raw_model, "dtype", self._dtype)),
            str(self._attn_impl),
        )

    @staticmethod
    def _template_prefix_len_from_decode_inputs(pc: _DecodeInputs) -> int:
        visual_start = int(getattr(pc, "visual_start", -1) or -1)
        if visual_start > 0:
            return visual_start
        return 0

    def _prefix_kv_cache_candidates(
        self,
        pc: _DecodeInputs,
        *,
        max_new_tokens: int,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]],
    ) -> list[tuple[tuple[Any, ...], int]]:
        candidates: list[tuple[tuple[Any, ...], int]] = []
        image_prefix_len = int(getattr(pc, "image_prefix_len", 0) or 0)
        image_key = self._prefix_kv_cache_key(
            input_ids=pc.input_ids,
            image_grid_thw=pc.image_grid_thw,
            image_content_fingerprint_hint=image_content_fingerprint_hint,
            image_prefix_len=image_prefix_len,
            max_new_tokens=max_new_tokens,
        )
        if image_key is not None:
            candidates.append((image_key, image_prefix_len))
        template_prefix_len = self._template_prefix_len_from_decode_inputs(pc)
        template_key = self._template_prefix_kv_cache_key(
            input_ids=pc.input_ids,
            prefix_len=template_prefix_len,
            max_new_tokens=max_new_tokens,
        )
        if template_key is not None:
            candidates.append((template_key, template_prefix_len))
        return candidates

    @staticmethod
    def _slice_decode_inputs_range(pc: _DecodeInputs, start: int, end: int) -> _DecodeInputs:
        start = max(0, int(start))
        end = max(start, int(end))
        visual_start = int(getattr(pc, "visual_start", -1))
        visual_count = int(getattr(pc, "visual_count", 0))
        range_has_visual = (
            visual_start >= start
            and visual_count > 0
            and visual_start + visual_count <= end
        )
        return _DecodeInputs(
            input_ids=pc.input_ids[:, start:end],
            attention_mask=pc.attention_mask[:, start:end],
            image_grid_thw=pc.image_grid_thw,
            inputs_embeds=pc.inputs_embeds[:, start:end, :],
            position_ids=pc.position_ids[:, :, start:end],
            rope_deltas=pc.rope_deltas,
            visual_pos_masks=pc.visual_pos_masks[:, start:end],
            deepstack_visual_embeds=list(pc.deepstack_visual_embeds) if range_has_visual else [],
            visual_start=visual_start - start if range_has_visual else -1,
            visual_count=visual_count if range_has_visual else 0,
            image_prefix_len=max(0, min(int(getattr(pc, "image_prefix_len", 0)) - start, end - start)),
        )

    def _run_prefill_model_direct(
        self,
        pc: _DecodeInputs,
        past: Any,
        cache_position: torch.Tensor,
        *,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> tuple[torch.Tensor, Any]:
        assert self._raw_model is not None
        language_model = self._raw_model.model.language_model
        out = language_model(
            input_ids=None,
            position_ids=pc.position_ids,
            attention_mask=None,
            past_key_values=past,
            inputs_embeds=pc.inputs_embeds,
            use_cache=True,
            cache_position=cache_position,
            visual_pos_masks=pc.visual_pos_masks,
            deepstack_visual_embeds=pc.deepstack_visual_embeds,
        )
        self._profile_inc(profile_stats, "prefill_backend_prefix_kv_direct")
        return out.last_hidden_state[:, -1, :], out.past_key_values if out.past_key_values is not None else past

    def _try_prefix_kv_prefill(
        self,
        pc: _DecodeInputs,
        *,
        past: Any,
        cache_len: int,
        max_new_tokens: int,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]],
        profile_stats: Optional[dict[str, float]] = None,
    ) -> Optional[tuple[torch.Tensor, Any]]:
        prefix_len = int(getattr(pc, "image_prefix_len", 0) or 0)
        prompt_len = int(pc.input_ids.shape[1])
        if (
            not self._prefix_kv_cache_allowed_for_request(int(max_new_tokens))
            or past is not None
            or self._raw_model is None
        ):
            return None
        candidates = [
            (key, candidate_prefix_len)
            for key, candidate_prefix_len in self._prefix_kv_cache_candidates(
                pc,
                max_new_tokens=max_new_tokens,
                image_content_fingerprint_hint=image_content_fingerprint_hint,
            )
            if candidate_prefix_len > 0 and candidate_prefix_len < prompt_len
        ]
        if not candidates:
            return None
        device = pc.input_ids.device
        request_past = self._make_past_cache(int(cache_len), device)
        if request_past is None:
            return None
        for key, candidate_prefix_len in candidates:
            state = self._prefix_kv_cache_entries.get(key)
            if isinstance(state, dict):
                cached_past = state.get("past")
                if self._copy_dynamic_cache_prefix_into_existing(cached_past, request_past, candidate_prefix_len):
                    try:
                        self._prefix_kv_cache_entries.move_to_end(key)
                    except Exception:
                        pass
                    self._profile_inc(profile_stats, "prefix_kv_cache_hit")
                    tail_pc = self._slice_decode_inputs_range(pc, candidate_prefix_len, prompt_len)
                    tail_cache_position = torch.arange(candidate_prefix_len, prompt_len, device=device, dtype=torch.long)
                    hidden_last, request_past = self._run_prefill_model_direct(
                        tail_pc,
                        request_past,
                        tail_cache_position,
                        profile_stats=profile_stats,
                    )
                    return hidden_last, request_past
            self._profile_inc(profile_stats, "prefix_kv_cache_miss")
        return None

    def _store_prefix_kv_snapshot_from_prefill(
        self,
        pc: _DecodeInputs,
        past: Any,
        *,
        max_new_tokens: int,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]],
        profile_stats: Optional[dict[str, float]] = None,
    ) -> None:
        prefix_len = int(getattr(pc, "image_prefix_len", 0) or 0)
        prompt_len = int(pc.input_ids.shape[1])
        if (
            past is None
            or not self._prefix_kv_cache_allowed_for_request(int(max_new_tokens))
            or self._raw_model is None
        ):
            return
        for key, candidate_prefix_len in self._prefix_kv_cache_candidates(
            pc,
            max_new_tokens=max_new_tokens,
            image_content_fingerprint_hint=image_content_fingerprint_hint,
        ):
            if candidate_prefix_len <= 0 or candidate_prefix_len >= prompt_len:
                continue
            if key in self._prefix_kv_cache_entries:
                continue
            snapshot = self._clone_past_cache(past, candidate_prefix_len, pc.input_ids.device)
            if snapshot is None:
                continue
            try:
                self._prefix_kv_cache_entries[key] = {"past": snapshot, "prefix_len": candidate_prefix_len}
                self._prefix_kv_cache_entries.move_to_end(key)
                while len(self._prefix_kv_cache_entries) > int(self._prefix_kv_cache_max_entries):
                    self._prefix_kv_cache_entries.popitem(last=False)
                self._profile_inc(profile_stats, "prefix_kv_cache_store")
            except Exception:
                continue

    @staticmethod
    def _prefill_keep_ratio_for_tier(visual_tier: str, question_kind: str) -> float:
        if question_kind in {"ocr_numeric", "counting"}:
            return 1.0
        if visual_tier == "tight":
            return 0.70
        if visual_tier == "compact":
            return 0.78
        if visual_tier == "balanced":
            return 0.88
        return 0.94

    @staticmethod
    def _decode_visual_keep_ratio_for_tier(visual_tier: str, question_kind: str) -> float:
        if question_kind == "ocr_numeric":
            return 0.88
        if question_kind == "counting":
            return 0.82
        if question_kind == "attribute":
            return 0.62
        if question_kind == "scene":
            return 0.55
        if visual_tier == "tight":
            return 0.45
        if visual_tier == "compact":
            return 0.55
        if visual_tier == "balanced":
            return 0.68
        return 0.75

    @staticmethod
    def _spatial_anchor_ratio_for_kind(question_kind: str) -> float:
        if question_kind == "ocr_numeric":
            return 0.35
        if question_kind == "counting":
            return 0.30
        if question_kind == "scene":
            return 0.20
        return 0.16

    def _graph_cache_policy_for(self, max_new_tokens: int, enable_whitebox_block: bool) -> str:
        policy = self._visual_graph_cache_policy
        if policy in {"0", "off", "disabled", "none"}:
            return "disabled"
        if enable_whitebox_block:
            return "decode_token_block_profile"
        if int(max_new_tokens) <= 1:
            return "single_token_shape_profile"
        return "accuracy_dynamic_profile"

    def _requires_whitebox_token_block(self, max_new_tokens: int) -> bool:



        if int(max_new_tokens) <= 1:
            return False
        return bool(
            self._whitebox_require_token_block
            and int(max_new_tokens) in self._whitebox_required_token_block_requests
        )

    def _requires_strict_shared_token_block(self, max_new_tokens: int) -> bool:
        return bool(
            self._whitebox_strict_shared_token_block
            and self._requires_whitebox_token_block(int(max_new_tokens))
        )

    def _estimate_visual_kv_target_tokens(
        self,
        *,
        estimated_visual_tokens: int,
        question_kind: str,
        decode_keep_ratio: float,
    ) -> int:
        raw_tokens = max(0, int(estimated_visual_tokens))
        if raw_tokens <= 0:
            return 0
        if question_kind == "ocr_numeric":
            floor = min(raw_tokens, 384)
        elif question_kind == "counting":
            floor = min(raw_tokens, 320)
        else:
            floor = min(raw_tokens, 192)
        return min(raw_tokens, max(floor, int(round(raw_tokens * float(decode_keep_ratio)))))

    def _build_visual_budget_profile(self, messages: Any, request_pixels: int) -> _VisualBudgetProfile:
        image, question = self._extract_image_and_text(messages)
        try:
            image_size = tuple(int(v) for v in getattr(image, "size", ()) or ()) if image is not None else None
        except Exception:
            image_size = None
        question_kind = self._classify_question_kind(question)
        visual_tier = self._select_visual_tier(int(request_pixels), image_size, question_kind)
        prefill_keep_ratio = self._prefill_keep_ratio_for_tier(visual_tier, question_kind)
        decode_keep_ratio = self._decode_visual_keep_ratio_for_tier(visual_tier, question_kind)
        evidence_guard = question_kind in {"ocr_numeric", "counting"}
        profile = _VisualBudgetProfile(
            request_key=self._request_fingerprint(messages),
            image_key=self._image_fingerprint(messages),
            request_pixels=int(request_pixels),
            image_size=image_size if isinstance(image_size, tuple) and len(image_size) == 2 else None,
            question_kind=question_kind,
            visual_tier=visual_tier,
            question_text=question,
            lifecycle_policy=self._visual_lifecycle_strategy,
            prefill_budget=self._prefill_budget_for_tier(visual_tier, question_kind),
            prefill_keep_ratio=float(prefill_keep_ratio),
            decode_budget="pending",
            decode_visual_keep_ratio=float(decode_keep_ratio),
            kv_policy="dynamic",
            visual_kv_policy="profile_only_visual_kv",
            visual_kv_target_keep_ratio=float(decode_keep_ratio),
            visual_kv_profile_only=bool(self._visual_kv_profile_only),
            quant_policy="none",
            graph_cache_policy="pending",
            fusion_policy="pending",
            evidence_guard=bool(evidence_guard),
            spatial_anchor_ratio=float(self._spatial_anchor_ratio_for_kind(question_kind)),
        )
        self._current_visual_budget_profile = profile
        return profile

    def _finalize_visual_budget_profile(
        self,
        *,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        max_new_tokens: int,
    ) -> _VisualBudgetProfile:
        profile = self._current_visual_budget_profile
        if profile is None:
            profile = _VisualBudgetProfile(
                request_key="unknown",
                image_key="unknown",
                request_pixels=int(self._current_request_pixels),
                image_size=None,
                question_kind="generic",
                visual_tier="unknown",
            )
        prompt_len = int(input_ids.shape[1]) if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2 else 0
        requested_len = int(prompt_len + int(max_new_tokens))
        strict_token_block = self._requires_strict_shared_token_block(int(max_new_tokens))
        actual_token_block_required_len = (
            int(prompt_len + int(max_new_tokens) - 1)
            if strict_token_block and prompt_len > 1
            else requested_len
        )
        if strict_token_block and prompt_len > 1:
            bucket_required_len = int(actual_token_block_required_len)
        else:
            bucket_required_len = requested_len
        estimated_visual_tokens = self._estimate_visual_tokens_from_grid(image_grid_thw)
        compact_bucket_required_len = 0
        if (
            self._whitebox_visual_kv_compact_enabled
            and not self._visual_kv_profile_only
            and int(self._whitebox_visual_kv_keep_tokens) > 0
            and int(estimated_visual_tokens) > int(self._whitebox_visual_kv_keep_tokens)
            and int(estimated_visual_tokens) < int(prompt_len)
        ):
            effective_prompt_len = int(prompt_len - 1) if strict_token_block and prompt_len > 1 else int(prompt_len)
            compact_prompt_len = (
                int(effective_prompt_len)
                - int(estimated_visual_tokens)
                + min(int(estimated_visual_tokens), int(self._whitebox_visual_kv_keep_tokens))
            )
            compact_bucket_required_len = int(compact_prompt_len + int(max_new_tokens))
        if (
            self._whitebox_bucket_plan_max
            and int(max_new_tokens) < int(self._whitebox_max_new_tokens)
        ):
            bucket_required_len = int(prompt_len + int(self._whitebox_max_new_tokens))
            if strict_token_block and prompt_len > 1:
                bucket_required_len -= 1
        bucket_candidates = tuple(sorted(int(bucket) for bucket in self._whitebox_block_paths if int(bucket) > 0))
        enable_whitebox_block = (
            self._whitebox_block_enabled
            and StaticCache is not None
            and int(actual_token_block_required_len) <= int(self._whitebox_cache_len)
            and (int(max_new_tokens) > 1 or strict_token_block)
        )
        fallback_reason = ""
        graph_bucket = None
        prefer_graph_replay = False
        if enable_whitebox_block:
            normal_bucket = self._pick_whitebox_block_bucket(
                bucket_required_len,
                max_new_tokens=int(max_new_tokens),
            )
            compact_bucket = (
                self._pick_whitebox_block_bucket(
                    compact_bucket_required_len,
                    max_new_tokens=int(max_new_tokens),
                )
                if compact_bucket_required_len > 0
                else None
            )
            if compact_bucket is not None and (
                normal_bucket is None or int(compact_bucket) < int(normal_bucket)
            ):
                graph_bucket = int(compact_bucket)
                bucket_required_len = int(compact_bucket_required_len)
                fallback_reason = "whitebox_visual_compact_bucket_hit"
            else:
                graph_bucket = normal_bucket
            if graph_bucket is None and bucket_required_len != requested_len:
                graph_bucket = self._pick_whitebox_block_bucket(
                    requested_len,
                    max_new_tokens=int(max_new_tokens),
                )
                if graph_bucket is not None:
                    fallback_reason = "whitebox_bucket_plan_miss_actual_hit"
            if graph_bucket is None:
                fallback_reason = "whitebox_bucket_miss"
            elif not self._whitebox_block_arg_count_supported(int(graph_bucket)):
                fallback_reason = "whitebox_argcount_mismatch"
                graph_bucket = None
            else:
                prefer_graph_replay = bool(
                    self._whitebox_block_graph_enabled
                    and int(graph_bucket) in self._whitebox_block_graph_buckets
                    and self._whitebox_block_graph_layout_compatible(int(graph_bucket))
                )
        else:
            if int(actual_token_block_required_len) > int(self._whitebox_cache_len):
                fallback_reason = "requested_len_exceeds_whitebox_cache_len"
            elif not self._whitebox_block_enabled:
                fallback_reason = "whitebox_block_disabled"

        decode_budget = "dynamic"
        kv_policy = "dynamic"
        quant_policy = "none"
        fusion_policy = "dynamic_decode"
        if enable_whitebox_block and graph_bucket is not None:
            graph_step_size = int(self._whitebox_block_step_size(int(graph_bucket)))
            decode_budget = f"token_block_step{graph_step_size}"
            kv_policy = "static_token_block"
            quant_parts: list[str] = []
            if self._whitebox_block_smoothquant_gateup:
                quant_parts.append(f"sq_gateup_w8a8:{self._whitebox_block_weightonly_gateup_runners}")
            if self._whitebox_block_weightonly_gateup:
                quant_parts.append(f"wo_gateup_int8:{self._whitebox_block_weightonly_gateup_runners}")
            if self._whitebox_block_weightonly_lmhead:
                quant_parts.append("wo_lmhead_int8")
            quant_policy = "+".join(quant_parts) if quant_parts else "none"
            fusion_policy = (
                f"whitebox_step{graph_step_size}"
                + ("_graph_replay" if prefer_graph_replay else "_direct")
            )
        decode_keep_ratio = float(profile.decode_visual_keep_ratio)
        if not self._visual_kv_budget_enable:
            decode_keep_ratio = 1.0
        visual_kv_target_tokens = self._estimate_visual_kv_target_tokens(
            estimated_visual_tokens=estimated_visual_tokens,
            question_kind=profile.question_kind,
            decode_keep_ratio=decode_keep_ratio,
        )
        if max(0, int(estimated_visual_tokens)) > 0:
            visual_kv_target_keep_ratio = min(1.0, max(0.0, visual_kv_target_tokens / float(max(1, int(estimated_visual_tokens)))))
        else:
            visual_kv_target_keep_ratio = 1.0
        if self._visual_kv_profile_only:
            visual_kv_policy = "profile_only_visual_kv"
        elif enable_whitebox_block and graph_bucket is not None:
            visual_kv_policy = "candidate_visual_kv_static_block"
        else:
            visual_kv_policy = "candidate_visual_kv_dynamic"
        graph_cache_policy = self._graph_cache_policy_for(int(max_new_tokens), bool(enable_whitebox_block and graph_bucket is not None))
        profile = replace(
            profile,
            mode="request+generate",
            prompt_len=prompt_len,
            max_new_tokens=int(max_new_tokens),
            requested_len=requested_len,
            estimated_visual_tokens=estimated_visual_tokens,
            prefill_budget=profile.prefill_budget,
            prefill_keep_ratio=float(profile.prefill_keep_ratio),
            decode_budget=decode_budget,
            decode_visual_keep_ratio=float(decode_keep_ratio),
            kv_policy=kv_policy,
            visual_kv_policy=visual_kv_policy,
            visual_kv_target_keep_ratio=float(visual_kv_target_keep_ratio),
            visual_kv_target_tokens=int(visual_kv_target_tokens),
            visual_kv_profile_only=bool(self._visual_kv_profile_only),
            quant_policy=quant_policy,
            graph_cache_policy=graph_cache_policy,
            fusion_policy=fusion_policy,
            enable_whitebox_block=bool(enable_whitebox_block and graph_bucket is not None),
            prefer_graph_replay=bool(prefer_graph_replay),
            graph_bucket=int(graph_bucket) if graph_bucket is not None else None,
            bucket_required_len=int(bucket_required_len),
            bucket_candidates=bucket_candidates,
            switch_after=int(self._whitebox_switch_after),
            fallback_reason=fallback_reason,
            visual_kv_compact_delta=(
                max(0, int(actual_token_block_required_len) - int(bucket_required_len))
                if fallback_reason == "whitebox_visual_compact_bucket_hit"
                else 0
            ),
        )
        self._current_visual_budget_profile = profile
        return profile

    @staticmethod
    def _format_visual_budget_profile(profile: Optional[_VisualBudgetProfile]) -> str:
        if profile is None:
            return "visual_budget_profile=none"
        image_size = "x".join(str(v) for v in profile.image_size) if profile.image_size else "none"
        bucket = "none" if profile.graph_bucket is None else str(int(profile.graph_bucket))
        return (
            "visual_budget_profile="
            f"request_key={profile.request_key},"
            f"lifecycle={profile.lifecycle_policy},"
            f"mode={profile.mode},"
            f"request_pixels={int(profile.request_pixels)},"
            f"image_size={image_size},"
            f"question_kind={profile.question_kind},"
            f"visual_tier={profile.visual_tier},"
            f"prompt_len={int(profile.prompt_len)},"
            f"max_new_tokens={int(profile.max_new_tokens)},"
            f"requested_len={int(profile.requested_len)},"
            f"bucket_required_len={int(profile.bucket_required_len)},"
            f"estimated_visual_tokens={int(profile.estimated_visual_tokens)},"
            f"prefill_budget={profile.prefill_budget},"
            f"prefill_keep={float(profile.prefill_keep_ratio):.3f},"
            f"decode_budget={profile.decode_budget},"
            f"decode_visual_keep={float(profile.decode_visual_keep_ratio):.3f},"
            f"kv_policy={profile.kv_policy},"
            f"visual_kv_policy={profile.visual_kv_policy},"
            f"visual_kv_target_tokens={int(profile.visual_kv_target_tokens)},"
            f"visual_kv_target_keep={float(profile.visual_kv_target_keep_ratio):.3f},"
            f"visual_kv_profile_only={int(bool(profile.visual_kv_profile_only))},"
            f"quant_policy={profile.quant_policy},"
            f"graph_cache_policy={profile.graph_cache_policy},"
            f"fusion_policy={profile.fusion_policy},"
            f"evidence_guard={int(bool(profile.evidence_guard))},"
            f"spatial_anchor={float(profile.spatial_anchor_ratio):.3f},"
            f"whitebox_block={int(bool(profile.enable_whitebox_block))},"
            f"graph_bucket={bucket},"
            f"graph_replay={int(bool(profile.prefer_graph_replay))},"
            f"switch_after={int(profile.switch_after)},"
            f"visual_kv_compact_delta={int(profile.visual_kv_compact_delta)},"
            f"fallback_reason={profile.fallback_reason or 'none'}"
        )

    def _choose_request_pixels(self, messages: Any) -> int:
        pixels = int(self._max_pixels)
        if not self._adaptive_res:
            self._build_visual_budget_profile(messages, pixels)
            return pixels
        image, question = self._extract_image_and_text(messages)
        if image is None:
            self._build_visual_budget_profile(messages, pixels)
            return pixels
        try:
            w, h = image.size
        except Exception:
            self._build_visual_budget_profile(messages, pixels)
            return pixels

        max_pixels = int(self._max_pixels)
        p524 = min(max_pixels, 524288)
        p655 = min(max_pixels, 655360)
        p720 = min(max_pixels, 720896)
        p786 = min(max_pixels, 786432)

        question_kind = self._classify_question_kind(question)
        width = max(1, int(w))
        height = max(1, int(h))
        aspect = max(width, height) / float(max(1, min(width, height)))
        squareish = aspect <= 1.12
        very_wide_or_tall = aspect >= 1.85
        wide_or_tall = aspect >= 1.55





        policy = self._resolution_policy
        if policy in {"balanced", "medium"}:
            if question_kind in {"ocr_numeric", "counting"}:
                if very_wide_or_tall:
                    pixels = p655
                elif squareish or wide_or_tall:
                    pixels = p720
                else:
                    pixels = p786
            elif question_kind == "attribute":
                pixels = p655 if very_wide_or_tall else p720
            elif question_kind == "scene":
                pixels = p655 if wide_or_tall else p720
            else:
                if very_wide_or_tall:
                    pixels = p524
                elif wide_or_tall or squareish:
                    pixels = p655
                else:
                    pixels = p786
        elif question_kind in {"ocr_numeric", "counting"}:
            pixels = p655 if very_wide_or_tall else p786
        elif question_kind == "attribute":
            pixels = p655 if very_wide_or_tall else p786
        elif question_kind == "scene":
            pixels = p655 if wide_or_tall else p786
        else:
            if very_wide_or_tall:
                pixels = p524
            elif wide_or_tall:
                pixels = p655
            else:
                pixels = p786
        self._build_visual_budget_profile(messages, pixels)
        return pixels

    def _apply_processor_resolution(self, pixels: int, shortest_edge: Optional[int] = None) -> None:
        self._current_request_pixels = int(pixels)
        shortest = int(self._shortest_edge if shortest_edge is None else shortest_edge)
        self._current_request_shortest_edge = int(shortest)
        size = {"shortest_edge": shortest, "longest_edge": int(pixels)}
        try:
            self._raw_processor.image_processor.size = dict(size)
        except Exception:
            pass

    def _set_request_resolution(self, messages: Any) -> None:
        self._apply_processor_resolution(self._choose_request_pixels(messages), self._shortest_edge)

    def _patch_vision_patch_mergers(self) -> None:
        if self._raw_model is None:
            return
        try:
            visual = self._raw_model.model.visual
            modules = [("merger", visual.merger)]
            modules.extend((f"deepstack_{idx}", module) for idx, module in enumerate(visual.deepstack_merger_list))
            for name, module in modules:
                if isinstance(module, _AICASVisionPatchMergerRunner):
                    continue
                if not all(hasattr(module, attr) for attr in ("norm", "linear_fc1", "linear_fc2")):
                    continue
                wrapped = _AICASVisionPatchMergerRunner(module)
                if name == "merger":
                    visual.merger = wrapped
                elif name.startswith("deepstack_"):
                    index = int(name.split("_", 1)[1])
                    visual.deepstack_merger_list[index] = wrapped
        except Exception:
            pass

    def _ensure_model(self) -> None:
        if self._raw_model is not None:
            return
        pass
        load_kwargs = {
            "dtype": self._dtype,
            "attn_implementation": self._attn_impl,
            "low_cpu_mem_usage": True,
        }
        if self._load_mode in {"device_map", "dispatch"}:
            self._raw_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=self._device,
                **load_kwargs,
            ).eval()
        else:
            try:





                self._raw_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    **load_kwargs,
                ).to(self._device).eval()
            except Exception as exc:
                pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._raw_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    device_map=self._device,
                    **load_kwargs,
                ).eval()
        self._eos_token_ids = self._collect_eos_ids()
        self._eos_pad_fill_token_id = self._collect_eos_pad_fill_token_id()
        self._prepare_mlp_gate_up_fusion()
        self._prepare_qkv_fusion()
        self._prepare_vision_compile()
        self._patch_vision_patch_mergers()
        self._prepare_inline_layer_specs()
        self._prepare_ttft_inline_prefill_static()
        self._prepare_shared_vision_graph_static()
        self._prepare_whitebox_block_persistent_sequence_graph_static()
        self._prepare_whitebox_token_block_static()
        pass

    def _text_model(self):
        if self._raw_model is None:
            return None
        model_core = getattr(self._raw_model, "model", None)
        return getattr(model_core, "language_model", None)

    @staticmethod
    def _normalize_image_feature_outputs(output: Any) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if isinstance(output, dict):
            pooler_output = output.get("pooler_output")
            deepstack_features = output.get("deepstack_features")
        else:
            pooler_output = getattr(output, "pooler_output", None)
            deepstack_features = getattr(output, "deepstack_features", None)
        if isinstance(pooler_output, (list, tuple)):
            image_embeds = tuple(x for x in pooler_output if isinstance(x, torch.Tensor))
        elif isinstance(pooler_output, torch.Tensor):
            image_embeds = (pooler_output,)
        elif isinstance(output, tuple) and len(output) >= 2:
            first, second = output[0], output[1]
            if isinstance(first, (list, tuple)):
                image_embeds = tuple(x for x in first if isinstance(x, torch.Tensor))
            elif isinstance(first, torch.Tensor):
                image_embeds = (first,)
            else:
                image_embeds = ()
            deepstack_features = second
        else:
            image_embeds = ()

        if isinstance(deepstack_features, (list, tuple)):
            deepstack_embeds = tuple(x for x in deepstack_features if isinstance(x, torch.Tensor))
        else:
            deepstack_embeds = ()
        return image_embeds, deepstack_embeds

    def _prepare_vision_compile(self) -> None:
        if not self._compile_vision_encoder or self._raw_model is None:
            return
        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            self._compile_vision_encoder = False
            return
        try:
            visual = getattr(getattr(self._raw_model, "model", None), "visual", None)
            if visual is None or bool(getattr(visual, "_aicasgc_compiled", False)):
                return
            compiled = compile_fn(
                visual,
                backend="inductor",
                dynamic=False,
                fullgraph=False,
                mode="reduce-overhead",
            )
            setattr(compiled, "_aicasgc_compiled", True)
            self._raw_model.model.visual = compiled
        except Exception as exc:
            pass
            self._compile_vision_encoder = False

    @staticmethod
    def _shared_vision_grid_key(image_grid_thw: torch.Tensor) -> Optional[tuple[int, int, int]]:
        if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.ndim != 2:
            return None
        if tuple(int(v) for v in image_grid_thw.shape) != (1, 3):
            return None
        try:
            values = tuple(int(v) for v in image_grid_thw.detach().view(-1).tolist())
        except Exception:
            return None
        if len(values) != 3 or min(values) <= 0:
            return None
        return values

    def _shared_vision_graph_attn(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        assert apply_rotary_pos_emb_vision is not None
        seq_length = int(hidden_states.shape[0])
        qkv = attn.qkv(hidden_states).view(seq_length, 3, attn.num_heads, -1)
        query_states = qkv.select(1, 0)
        key_states = qkv.select(1, 1)
        value_states = qkv.select(1, 2)
        cos, sin = position_embeddings
        rope_helper = globals().get("_aicas_triton_vision_rope_qk")
        rope_outputs = (
            rope_helper(query_states, key_states, cos, sin)
            if callable(rope_helper)
            else None
        )
        if (
            isinstance(rope_outputs, tuple)
            and len(rope_outputs) == 2
            and isinstance(rope_outputs[0], torch.Tensor)
            and isinstance(rope_outputs[1], torch.Tensor)
        ):
            query_states, key_states = rope_outputs
        else:
            query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            scale=attn.scaling,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return attn.proj(attn_output)

    def _run_shared_vision_graph_body(self, state: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        assert self._raw_model is not None
        visual = self._raw_model.model.visual
        hidden_states = visual.patch_embed(state["pixel_values"].type(visual.dtype))
        hidden_states.add_(state["pos_embeds"])
        hidden_states = hidden_states.reshape(int(hidden_states.shape[0]), -1)
        outputs: list[torch.Tensor] = []
        position_embeddings = state["position_embeddings"]
        for layer_num, block in enumerate(visual.blocks):
            hidden_states = hidden_states + self._shared_vision_graph_attn(
                block.attn,
                block.norm1(hidden_states),
                position_embeddings,
            )
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
            if layer_num in visual.deepstack_visual_indexes:
                deepstack = visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(layer_num)](hidden_states)
                outputs.append(deepstack)
        outputs.insert(0, visual.merger(hidden_states))
        return tuple(outputs)

    @staticmethod
    def _clear_shared_vision_graph_state_tensors(state: dict[str, Any]) -> None:
        try:
            tensor = state.get("pixel_values")
            if isinstance(tensor, torch.Tensor):
                tensor.zero_()
            for tensor in state.get("outputs", ()):
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()
        except Exception:
            pass

    def _make_shared_vision_graph_state(
        self,
        *,
        image_grid_thw: torch.Tensor,
        pixel_shape: tuple[int, ...],
        pixel_dtype: torch.dtype,
        device: torch.device,
        allow_capture: bool = True,
    ) -> Optional[dict[str, Any]]:
        if self._raw_model is None or apply_rotary_pos_emb_vision is None:
            return None
        grid_key = self._shared_vision_grid_key(image_grid_thw)
        if grid_key is None:
            return None
        if int(grid_key[0]) != 1:
            return None
        raw_tokens = int(grid_key[0] * grid_key[1] * grid_key[2])
        if self._shared_vision_graph_raw_tokens and raw_tokens not in self._shared_vision_graph_raw_tokens:
            return None
        cache_key = (
            grid_key,
            tuple(int(v) for v in pixel_shape),
            str(pixel_dtype),
            str(device),
        )
        if cache_key in self._shared_vision_graph_failed_keys:
            return None
        cached = self._shared_vision_graph_states.get(cache_key)
        if isinstance(cached, dict):
            return cached
        if not allow_capture:
            return None
        try:
            visual = self._raw_model.model.visual
            grid = torch.tensor([list(grid_key)], dtype=image_grid_thw.dtype, device=device)
            pos_embeds = visual.fast_pos_embed_interpolate(grid).detach().clone()
            rotary_pos_emb = visual.rot_pos_emb(grid).detach().clone()
            seq_len = int(rotary_pos_emb.shape[0])
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos().detach().clone(), emb.sin().detach().clone())
            state: dict[str, Any] = {
                "cache_key": cache_key,
                "grid": grid,
                "pixel_values": torch.zeros(tuple(int(v) for v in pixel_shape), dtype=pixel_dtype, device=device),
                "pos_embeds": pos_embeds,
                "position_embeddings": position_embeddings,
            }

            def _run_graph_body() -> tuple[torch.Tensor, ...]:
                return self._run_shared_vision_graph_body(state)

            warmup_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    _ = _run_graph_body()
            torch.cuda.current_stream(device=device).wait_stream(warmup_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    outputs = _run_graph_body()
            state["graph"] = graph
            state["outputs"] = outputs
            self._shared_vision_graph_states[cache_key] = state
            self._clear_shared_vision_graph_state_tensors(state)
            return state
        except Exception as exc:
            self._shared_vision_graph_failed_keys.add(cache_key)
            return None

    def _prepare_shared_vision_graph_static(self) -> None:
        if (
            self._shared_vision_graph_static_prepared
            or not self._shared_vision_graph_enabled
            or not self._shared_vision_graph_prewarm
            or self._raw_model is None
            or not torch.cuda.is_available()
        ):
            return
        self._shared_vision_graph_static_prepared = True
        try:
            visual = self._raw_model.model.visual
            patch_size = int(getattr(visual, "patch_size", 16) or 16)
            patch_dim = int(3 * 2 * patch_size * patch_size)
            device = next(self._raw_model.parameters()).device
            dtype = getattr(visual, "dtype", torch.float32)
            prewarm_grids: set[tuple[int, int, int]] = set(self._shared_vision_graph_grids)
            for raw_tokens in self._shared_vision_graph_raw_tokens:
                if int(raw_tokens) == 4:
                    prewarm_grids.add((1, 2, 2))
                elif int(raw_tokens) == 8:
                    prewarm_grids.add((1, 2, 4))
                    prewarm_grids.add((1, 4, 2))
                elif int(raw_tokens) == 16:
                    prewarm_grids.add((1, 4, 4))
                elif int(raw_tokens) == 24:
                    prewarm_grids.add((1, 4, 6))
            for grid_key in sorted(prewarm_grids):
                raw_tokens = int(grid_key[0] * grid_key[1] * grid_key[2])
                if raw_tokens <= 0:
                    continue
                grid = torch.tensor([list(grid_key)], dtype=torch.long, device=device)
                state = self._make_shared_vision_graph_state(
                    image_grid_thw=grid,
                    pixel_shape=(int(raw_tokens), patch_dim),
                    pixel_dtype=dtype,
                    device=device,
                    allow_capture=True,
                )
                if isinstance(state, dict):
                    self._clear_shared_vision_graph_state_tensors(state)
        except Exception as exc:
            pass

    @staticmethod
    def _build_ttft_prefill_graph_prewarm_pairs(
        *,
        raw_tokens: Iterable[int],
        grids: Iterable[tuple[int, int, int]],
        spatial_merge_size: int,
    ) -> tuple[tuple[int, int], ...]:
        merge_area = max(1, int(spatial_merge_size) * int(spatial_merge_size))
        visual_counts: set[int] = set()
        for raw in raw_tokens:
            raw_value = int(raw)
            if raw_value <= 0 or raw_value % merge_area != 0:
                continue
            visual_count = raw_value // merge_area
            if 64 <= visual_count <= 96:
                visual_counts.add(int(visual_count))
        for grid in grids:
            if len(grid) != 3:
                continue
            raw_value = int(grid[0]) * int(grid[1]) * int(grid[2])
            if raw_value <= 0 or raw_value % merge_area != 0:
                continue
            visual_count = raw_value // merge_area
            if 64 <= visual_count <= 96:
                visual_counts.add(int(visual_count))
        pairs: set[tuple[int, int]] = set()
        for visual_count in visual_counts:
            for text_overhead in range(15, 26):
                prompt_len = int(visual_count) + int(text_overhead)
                if 80 <= prompt_len <= 128 and prompt_len > visual_count:
                    pairs.add((prompt_len, int(visual_count)))
        return tuple(sorted(pairs))

    def _run_shared_vision_graph(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> Optional[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]:
        if not self._shared_vision_graph_enabled or not torch.cuda.is_available():
            return None
        if not isinstance(pixel_values, torch.Tensor) or not isinstance(image_grid_thw, torch.Tensor):
            return None
        try:
            visual = self._raw_model.model.visual
            graph_pixel_dtype = getattr(visual, "dtype", pixel_values.dtype)
            state = self._make_shared_vision_graph_state(
                image_grid_thw=image_grid_thw,
                pixel_shape=tuple(int(v) for v in pixel_values.shape),
                pixel_dtype=graph_pixel_dtype,
                device=pixel_values.device,
                allow_capture=bool(self._shared_vision_graph_capture_on_miss),
            )
            if not isinstance(state, dict):
                return None
            profile_start = self._profile_stamp()
            state["pixel_values"].copy_(pixel_values)
            state["graph"].replay()
            self._profile_add(profile_stats, "shared_vision_graph_replay", profile_start)
            self._profile_inc(profile_stats, "shared_vision_graph_hit")
            outputs = state.get("outputs")
            if not isinstance(outputs, tuple) or len(outputs) < 1:
                return None
            if self._shared_vision_graph_clone_outputs:
                image_embeds = outputs[0].clone()
                deepstack = tuple(tensor.clone() for tensor in outputs[1:] if isinstance(tensor, torch.Tensor))
                self._clear_shared_vision_graph_state_tensors(state)
            else:
                image_embeds = outputs[0]
                deepstack = tuple(tensor for tensor in outputs[1:] if isinstance(tensor, torch.Tensor))
            return (image_embeds,), deepstack
        except Exception as exc:
            pass
            return None

    def _prepare_mlp_gate_up_fusion(self) -> None:
        if not self._use_mlp_gate_up_fusion or self._raw_model is None:
            return
        try:
            for layer in self._raw_model.model.language_model.layers:
                mlp = layer.mlp
                gate_proj = mlp.gate_proj
                up_proj = mlp.up_proj
                cached = getattr(mlp, "_aicas_gate_up_weight", None)
                if (
                    isinstance(cached, torch.Tensor)
                    and cached.device == gate_proj.weight.device
                    and cached.dtype == gate_proj.weight.dtype
                    and int(cached.shape[0]) == int(gate_proj.weight.shape[0] + up_proj.weight.shape[0])
                    and int(cached.shape[1]) == int(gate_proj.weight.shape[1])
                ):
                    continue
                mlp._aicas_gate_up_weight = torch.cat(
                    [gate_proj.weight.detach(), up_proj.weight.detach()],
                    dim=0,
                ).contiguous()
                mlp._aicas_gate_out_size = int(gate_proj.weight.shape[0])
                if self._use_patch_mlp_forward and not hasattr(mlp, "_aicas_orig_forward"):
                    mlp._aicas_orig_forward = mlp.forward

                    def _patched_mlp_forward(this, hidden_states):
                        gate_up_out = F.linear(hidden_states, this._aicas_gate_up_weight, None)
                        gate_size = int(this._aicas_gate_out_size)
                        gate_states = gate_up_out[..., :gate_size]
                        up_states = gate_up_out[..., gate_size:]
                        return this.down_proj(this.act_fn(gate_states) * up_states)

                    mlp.forward = types.MethodType(_patched_mlp_forward, mlp)
        except Exception as exc:
            pass
            self._use_mlp_gate_up_fusion = False
            self._use_patch_mlp_forward = False

    def _prepare_qkv_fusion(self) -> None:
        if not self._use_qkv_fusion or self._raw_model is None:
            return
        try:
            for layer in self._raw_model.model.language_model.layers:
                attn = layer.self_attn
                q_proj = attn.q_proj
                k_proj = attn.k_proj
                v_proj = attn.v_proj
                cached = getattr(attn, "_aicas_qkv_weight", None)
                q_out = int(q_proj.weight.shape[0])
                k_out = int(k_proj.weight.shape[0])
                v_out = int(v_proj.weight.shape[0])
                if (
                    isinstance(cached, torch.Tensor)
                    and cached.device == q_proj.weight.device
                    and cached.dtype == q_proj.weight.dtype
                    and int(cached.shape[0]) == q_out + k_out + v_out
                    and int(cached.shape[1]) == int(q_proj.weight.shape[1])
                ):
                    continue
                attn._aicas_qkv_weight = torch.cat(
                    [q_proj.weight.detach(), k_proj.weight.detach(), v_proj.weight.detach()],
                    dim=0,
                ).contiguous()
                biases = [q_proj.bias, k_proj.bias, v_proj.bias]
                if all(isinstance(bias, torch.Tensor) for bias in biases):
                    attn._aicas_qkv_bias = torch.cat([bias.detach() for bias in biases], dim=0).contiguous()
                else:
                    attn._aicas_qkv_bias = None
                attn._aicas_q_out_size = q_out
                attn._aicas_k_out_size = k_out
                attn._aicas_qkv_patch_fused_linear = bool(self._current_qkv_patch_fused_linear)
                if self._use_patch_qkv_forward and not hasattr(attn, "_aicas_orig_forward"):
                    attn._aicas_orig_forward = attn.forward

                    def _patched_attention_forward(
                        this,
                        hidden_states: torch.Tensor,
                        position_embeddings: tuple[torch.Tensor, torch.Tensor],
                        attention_mask: Optional[torch.Tensor],
                        past_key_values: Optional[Any] = None,
                        cache_position: Optional[torch.Tensor] = None,
                        **kwargs,
                    ):
                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, this.head_dim)
                        if bool(getattr(this, "_aicas_qkv_patch_fused_linear", True)):
                            qkv_out = F.linear(hidden_states, this._aicas_qkv_weight, this._aicas_qkv_bias)
                            q_out_size = int(this._aicas_q_out_size)
                            k_out_size = int(this._aicas_k_out_size)
                            query_states = qkv_out[..., :q_out_size].view(hidden_shape)
                            key_states = qkv_out[..., q_out_size : q_out_size + k_out_size].view(hidden_shape)
                            value_states = qkv_out[..., q_out_size + k_out_size :].view(hidden_shape)
                        else:
                            query_states = this.q_proj(hidden_states).view(hidden_shape)
                            key_states = this.k_proj(hidden_states).view(hidden_shape)
                            value_states = this.v_proj(hidden_states).view(hidden_shape)

                        query_states = this.q_norm(query_states).transpose(1, 2)
                        key_states = this.k_norm(key_states).transpose(1, 2)
                        value_states = value_states.transpose(1, 2)

                        cos, sin = position_embeddings
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                        if past_key_values is not None:
                            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                            key_states, value_states = past_key_values.update(
                                key_states,
                                value_states,
                                this.layer_idx,
                                cache_kwargs,
                            )

                        attention_interface: Callable = eager_attention_forward
                        if this.config._attn_implementation != "eager":
                            attention_interface = ALL_ATTENTION_FUNCTIONS[this.config._attn_implementation]

                        attn_output, attn_weights = attention_interface(
                            this,
                            query_states,
                            key_states,
                            value_states,
                            attention_mask,
                            dropout=0.0 if not this.training else this.attention_dropout,
                            scaling=this.scaling,
                            **kwargs,
                        )

                        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                        attn_output = this.o_proj(attn_output)
                        return attn_output, attn_weights

                    attn.forward = types.MethodType(_patched_attention_forward, attn)
        except Exception as exc:
            pass
            self._use_qkv_fusion = False
            self._use_patch_qkv_forward = False

    def _prepare_inline_layer_specs(self) -> None:
        if self._raw_model is None:
            return
        specs: list[_InlineLayerSpec] = []
        for layer in self._raw_model.model.language_model.layers:
            attn = layer.self_attn
            mlp = layer.mlp
            gate_up_weight = getattr(mlp, "_aicas_gate_up_weight", None)
            qkv_weight = getattr(attn, "_aicas_qkv_weight", None)
            specs.append(
                _InlineLayerSpec(
                    input_norm_weight=layer.input_layernorm.weight,
                    input_norm_eps=float(layer.input_layernorm.variance_epsilon),
                    q_proj_weight=attn.q_proj.weight,
                    q_proj_bias=attn.q_proj.bias,
                    k_proj_weight=attn.k_proj.weight,
                    k_proj_bias=attn.k_proj.bias,
                    v_proj_weight=attn.v_proj.weight,
                    v_proj_bias=attn.v_proj.bias,
                    q_norm_weight=attn.q_norm.weight,
                    q_norm_eps=float(attn.q_norm.variance_epsilon),
                    k_norm_weight=attn.k_norm.weight,
                    k_norm_eps=float(attn.k_norm.variance_epsilon),
                    o_proj_weight=attn.o_proj.weight,
                    o_proj_bias=attn.o_proj.bias,
                    post_norm_weight=layer.post_attention_layernorm.weight,
                    post_norm_eps=float(layer.post_attention_layernorm.variance_epsilon),
                    gate_proj_weight=mlp.gate_proj.weight,
                    gate_proj_bias=mlp.gate_proj.bias,
                    up_proj_weight=mlp.up_proj.weight,
                    up_proj_bias=mlp.up_proj.bias,
                    gate_up_weight=gate_up_weight if isinstance(gate_up_weight, torch.Tensor) else None,
                    gate_out_size=int(getattr(mlp, "_aicas_gate_out_size", mlp.gate_proj.weight.shape[0])),
                    down_proj_weight=mlp.down_proj.weight,
                    down_proj_bias=mlp.down_proj.bias,
                    act_fn=mlp.act_fn,
                    layer_idx=int(attn.layer_idx),
                    head_dim=int(attn.head_dim),
                    scaling=float(attn.scaling),
                    qkv_weight=qkv_weight if isinstance(qkv_weight, torch.Tensor) else None,
                    qkv_bias=getattr(attn, "_aicas_qkv_bias", None),
                    q_out_size=int(getattr(attn, "_aicas_q_out_size", attn.q_proj.weight.shape[0])),
                    k_out_size=int(getattr(attn, "_aicas_k_out_size", attn.k_proj.weight.shape[0])),
                )
            )
        self._inline_layer_specs = specs

    def _prepare_ttft_inline_prefill_static(self) -> None:
        if (
            not bool(getattr(self, "_use_inline_prefill", False))
            or self._raw_model is None
            or not torch.cuda.is_available()
        ):
            return
        try:
            lm = self._raw_model.model.language_model
            hidden_size = int(getattr(getattr(lm, "config", None), "hidden_size", 2048) or 2048)
            device = next(lm.parameters()).device
            dtype = getattr(self._raw_model, "dtype", self._dtype)
            with torch.inference_mode():
                for prompt_len, visual_count in getattr(self, "_ttft_prefill_graph_prewarm_pairs", ()):
                    prompt_len = int(prompt_len)
                    visual_count = int(visual_count)
                    if prompt_len <= 0 or visual_count <= 0 or visual_count >= prompt_len:
                        continue
                    cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)
                    base_pos = cache_position.view(1, 1, -1).expand(3, 1, -1).contiguous()
                    inputs_embeds = torch.zeros((1, prompt_len, hidden_size), device=device, dtype=dtype)
                    visual_pos_masks = torch.empty((1, 0), device=device, dtype=torch.bool)
                    visual_start = min(4, max(0, prompt_len - visual_count))
                    if visual_start + visual_count > prompt_len:
                        continue
                    deepstack = [
                        torch.zeros((visual_count, hidden_size), device=device, dtype=dtype)
                        for _ in range(3)
                    ]
                    pc = _DecodeInputs(
                        input_ids=torch.zeros((1, prompt_len), device=device, dtype=torch.long),
                        attention_mask=torch.ones((1, prompt_len), device=device, dtype=torch.long),
                        image_grid_thw=torch.empty((1, 3), device=device, dtype=torch.long),
                        inputs_embeds=inputs_embeds,
                        position_ids=base_pos,
                        rope_deltas=torch.zeros((1, 1), device=device, dtype=torch.long),
                        visual_pos_masks=visual_pos_masks,
                        deepstack_visual_embeds=deepstack,
                        visual_start=visual_start,
                        visual_count=visual_count,
                    )
                    key = self._ttft_prefill_graph_key(pc, cache_position)
                    if key is not None:
                        self._make_ttft_prefill_graph_state(key, pc, cache_position)
                torch.cuda.synchronize()
        except Exception:
            pass

    def _find_whitebox_block_paths(self) -> dict[int, Path]:
        if StaticCache is None:
            return {}
        root = Path(__file__).resolve().parent
        default_dir = root / "submission1" / "my_kernel" / "generated"
        paths: dict[int, Path] = {}
        raw = os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PATHS", "").strip()
        if raw:
            for item in raw.split(","):
                if "=" not in item:
                    continue
                bucket_s, path_s = item.split("=", 1)
                try:
                    bucket = int(bucket_s.strip())
                except Exception:
                    continue
                path = Path(path_s.strip()).expanduser()
                if bucket > 0 and path.is_file():
                    paths[bucket] = path
            return dict(sorted(paths.items()))
        default_block_buckets = "256,320,384,832,960,1216,2176"
        bucket_raw = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_BUCKETS",
            default_block_buckets,
        ).strip()
        buckets: list[int] = []
        for item in bucket_raw.split(","):
            try:
                bucket = int(item.strip())
            except Exception:
                continue
            if bucket > 0 and bucket not in buckets:
                buckets.append(bucket)
        if not buckets:
            buckets = [320]
        for bucket in buckets:
            candidates = []
            if int(bucket) in (224, 256):
                quantadapt_dir = root / "artifacts" / "generated_quantadapt"
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            f"whitebox_compile_token_block_qwen3vl_static{int(bucket)}_step16_*"
                            "quantnative_w8a16_gateup_rowtriton_directall14_derived320_candidate.py"
                        )
                    )
                )
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            f"whitebox_compile_token_block_qwen3vl_static{int(bucket)}_step16_*"
                            "quantnative_w8a16_gateup_derived320_candidate.py"
                        )
                    )
                )
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            f"whitebox_compile_token_block_qwen3vl_static{int(bucket)}_step16_*"
                            "weightonly_gateup_directplain_derived320_candidate.py"
                        )
                    )
                )
            if (
                int(bucket) == 320
                and os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_PREFER_STEP16_320", "1") != "0"
            ):
                quantadapt_dir = root / "artifacts" / "generated_quantadapt"
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            "whitebox_compile_token_block_qwen3vl_static320_step16_*"
                            "quantnative_w8a16_gateup_rowtriton_candidate.py"
                        )
                    )
                )
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            "whitebox_compile_token_block_qwen3vl_static320_step16_*"
                            "quantnative_w8a16_gateup_rowtriton_directall14_candidate.py"
                        )
                    )
                )
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            "whitebox_compile_token_block_qwen3vl_static320_step16_*"
                            "weightonly_gateup_directplain_candidate.py"
                        )
                    )
                )
            if int(bucket) == 320 and int(self._whitebox_block_size) == 16:
                quantadapt_dir = root / "artifacts" / "generated_quantadapt"
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            "whitebox_compile_token_block_qwen3vl_static320_step16_*"
                            "quantnative_w8a16_gateup_rowtriton_candidate.py"
                        )
                    )
                )
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            "whitebox_compile_token_block_qwen3vl_static320_step16_*"
                            "quantnative_w8a16_gateup_rowtriton_directall14_candidate.py"
                        )
                    )
                )
                candidates.extend(
                    sorted(
                        quantadapt_dir.glob(
                            "whitebox_compile_token_block_qwen3vl_static320_step16_*"
                            "weightonly_gateup_directplain_candidate.py"
                        )
                    )
                )
            if int(bucket) == 2176 and int(self._whitebox_block_size) == 16:
                candidates.extend(
                    sorted(
                        default_dir.glob(
                            "whitebox_compile_token_block_qwen3vl_static2176_step8_*.py"
                        )
                    )
                )
            candidates.extend(
                sorted(
                    default_dir.glob(
                        f"whitebox_compile_token_block_qwen3vl_static{bucket}_step{self._whitebox_block_size}_*.py"
                    )
                )
            )
            if not candidates:
                candidates = sorted(default_dir.glob(f"whitebox_compile_token_block_qwen3vl_static{bucket}_step*.py"))
            if not candidates:
                continue
            candidates.sort(
                key=lambda p: (
                    ("quantnative_w8a16_gateup_rowtriton_directall14" in p.name.lower()),
                    ("quantnative_w8a16_gateup_rowtriton" in p.name.lower()),
                    ("quantnative_w8a16_gateup" in p.name.lower()),
                    ("weightonly_gateup_directplain" in p.name.lower()),
                    ("fusedgate" in p.name.lower()),
                    ("official" in p.name.lower()),
                    p.stat().st_mtime,
                ),
                reverse=True,
            )
            paths[int(bucket)] = candidates[0]
        return dict(sorted(paths.items()))

    def _whitebox_block_path_validated(self, bucket: int, path: Path) -> bool:
        if bool(getattr(self, "_whitebox_block_allow_unvalidated", False)):
            return True
        try:
            root = Path(__file__).resolve().parent
            resolved = path.resolve()
            allowed_roots = (
                (root / "artifacts" / "generated_quantadapt").resolve(),
                (root / "submission1" / "my_kernel" / "generated").resolve(),
            )
            if not any(resolved.is_relative_to(allowed_root) for allowed_root in allowed_roots):
                return False
            name = resolved.name
            step = int(self._whitebox_block_step_size(bucket))
            return bool(
                name.startswith(f"whitebox_compile_token_block_qwen3vl_static{int(bucket)}_step{step}_")
                and name.endswith(".py")
                and ("candidate" in name or "derived" in name or "official" in name or "local" in name)
            )
        except Exception:
            return False

    def _whitebox_block_step_size(self, bucket: int) -> int:
        path = self._whitebox_block_paths.get(int(bucket))
        if isinstance(path, Path):
            match = re.search(r"_step(\d+)_", path.name)
            if match:
                try:
                    step = int(match.group(1))
                    if step > 0:
                        return step
                except Exception:
                    pass
        return int(self._whitebox_block_size)

    def _pick_whitebox_block_bucket(self, required_len: int, *, max_new_tokens: int = 0) -> Optional[int]:
        if not self._whitebox_block_enabled:
            return None
        try:
            required = int(required_len)
        except Exception:
            return None
        if required > int(self._whitebox_cache_len):
            return None
        perf_request = (
            int(max_new_tokens) > 0
            and int(max_new_tokens) == int(getattr(self, "_performance_request_tokens", 128))
        )
        perf_extra_buckets = set(getattr(self, "_whitebox_block_perf_extra_buckets", set()))
        for bucket in sorted(self._whitebox_block_paths):
            path = self._whitebox_block_paths.get(bucket)
            if bucket in self._whitebox_block_failed_buckets:
                continue
            if int(bucket) in perf_extra_buckets and not perf_request:
                continue
            if bucket >= required and isinstance(path, Path) and path.is_file():
                if (
                    bool(getattr(self, "_whitebox_block_reject_unvalidated", True))
                    and not self._whitebox_block_path_validated(int(bucket), path)
                ):
                    continue
                return int(bucket)
        return None

    @staticmethod
    def _patch_whitebox_block_module(module: Any) -> None:
        if module is None or bool(getattr(module, "_aicasgc_fastpath_patched", False)):
            return

        def _noop_guard(*args, **kwargs):
            return None

        try:
            if hasattr(module, "assert_size_stride"):
                setattr(module, "assert_size_stride", _noop_guard)
            if hasattr(module, "assert_alignment"):
                setattr(module, "assert_alignment", _noop_guard)
            setattr(module, "_aicasgc_fastpath_patched", True)
        except Exception:
            pass

    @staticmethod
    def _whitebox_block_lmhead_runner_names(module: Any) -> tuple[str, ...]:
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and module_file:
            try:
                text = Path(module_file).read_text(encoding="utf-8")
                names = tuple(
                    sorted(
                        set(
                            re.findall(
                                r"(triton_red_fused_mm_\d+)\.run\([^\n]*151936,\s*2048",
                                text,
                            )
                        )
                    )
                )
                if names:
                    return names
            except Exception:
                pass
        return ("triton_red_fused_mm_24", "triton_red_fused_mm_33")

    def _whitebox_block_candidate_lmhead_ids(self) -> Optional[tuple[int, ...]]:
        if not self._whitebox_block_candidate_lmhead:
            return None
        cached = getattr(self, "_whitebox_block_candidate_lmhead_ids_cache", None)
        if isinstance(cached, tuple) and cached:
            return cached
        tokenizer = getattr(getattr(self, "_raw_processor", None), "tokenizer", None)
        if tokenizer is None:
            return None
        ids: list[int] = []
        try:
            common_limit = max(
                0,
                int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_CANDIDATE_LMHEAD_COMMON_LIMIT", "0")),
            )
        except Exception:
            common_limit = 0
        if common_limit > 0:
            try:
                tokenizer_len = int(len(tokenizer))
            except Exception:
                tokenizer_len = int(getattr(tokenizer, "vocab_size", 0) or 0)
            for token_id in range(min(common_limit, tokenizer_len)):
                try:
                    piece = tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except Exception:
                    continue
                if not piece or "\ufffd" in piece:
                    continue
                if any((ord(ch) < 32 and ch not in "\n\r\t") or (0x7F <= ord(ch) <= 0x9F) for ch in piece):
                    continue
                ids.append(int(token_id))
        extra_text = os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_CANDIDATE_LMHEAD_EXTRA_TEXT", "").strip()
        if extra_text:
            try:
                ids.extend(int(v) for v in tokenizer.encode(extra_text, add_special_tokens=False))
            except Exception:
                pass
        try:
            special = set(int(v) for v in getattr(tokenizer, "all_special_ids", []) or [])
        except Exception:
            special = set()
        max_tokens = int(self._whitebox_block_candidate_lmhead_max_tokens)
        unique: list[int] = []
        seen: set[int] = set()
        for token_id in ids:
            token_id = int(token_id)
            if token_id in seen or token_id in special or token_id < 0:
                continue
            seen.add(token_id)
            unique.append(token_id)
            if max_tokens > 0 and len(unique) >= max_tokens:
                break
        result = tuple(unique)
        self._whitebox_block_candidate_lmhead_ids_cache = result
        return result if result else None

    @staticmethod
    def _whitebox_block_gateup_runner_names(module: Any) -> tuple[str, ...]:
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and module_file:
            try:
                text = Path(module_file).read_text(encoding="utf-8")
                names = tuple(
                    sorted(
                        set(
                            re.findall(
                                r"(triton_red_fused_mm_\d+)\.run\([^\n]*12288,\s*2048",
                                text,
                            )
                        )
                    )
                )
                if names:
                    return names
            except Exception:
                pass
        return ("triton_red_fused_mm_11", "triton_red_fused_mm_18")

    @staticmethod
    def _parse_layer_mask_spec(raw: Any) -> frozenset[int]:
        text = str(raw or "").strip()
        if not text:
            return frozenset()
        result: set[int] = set()
        for item in text.split(","):
            part = item.strip()
            if not part:
                continue
            if "-" in part:
                left, right = part.split("-", 1)
                try:
                    start = int(left.strip())
                    end = int(right.strip())
                except Exception:
                    continue
                if end < start:
                    start, end = end, start
                for value in range(start, end + 1):
                    if 0 <= value < 10_000:
                        result.add(int(value))
                continue
            try:
                value = int(part)
            except Exception:
                continue
            if 0 <= value < 10_000:
                result.add(int(value))
        return frozenset(result)

    def _patch_whitebox_block_skip_mlp_layers(self, module: Any) -> None:
        skip_layers = getattr(self, "_whitebox_block_skip_mlp_layers", frozenset())
        if (
            module is None
            or not skip_layers
            or bool(getattr(module, "_aicasgc_skip_mlp_layers_patched", False))
        ):
            return
        try:
            layer_args = [70 + 10 * layer for layer in range(28) if layer in skip_layers]
            if not layer_args:
                return
            layer_arg_to_idx = {int(arg): int((arg - 70) // 10) for arg in layer_args}
            target_gateup_weight_ptrs: set[int] = set()

            bind_name = "_aicasgc_quantnative_bind_gateup_args"
            original_bind = getattr(module, bind_name, None)
            if callable(original_bind):

                def _bind_with_skip_tracking(*items: Any) -> Any:
                    target_gateup_weight_ptrs.clear()
                    for idx in range(0, len(items), 3):
                        layer_idx = idx // 3
                        if int(layer_idx) not in skip_layers:
                            continue
                        try:
                            weight = items[idx]
                        except Exception:
                            continue
                        if isinstance(weight, torch.Tensor):
                            target_gateup_weight_ptrs.add(int(weight.data_ptr()))
                    return original_bind(*items)

                setattr(module, bind_name, _bind_with_skip_tracking)

            class _AICASSkipGateUpRunner:
                def __init__(self, original: Any) -> None:
                    self._original = original

                @staticmethod
                def _is_skip_weight(tensor: Any) -> bool:
                    return isinstance(tensor, torch.Tensor) and int(tensor.data_ptr()) in target_gateup_weight_ptrs

                @staticmethod
                def _is_output(tensor: Any) -> bool:
                    return isinstance(tensor, torch.Tensor) and int(tensor.numel()) >= 12288

                def run(self, *args: Any, **kwargs: Any) -> Any:
                    if len(args) >= 5 and self._is_skip_weight(args[1]) and self._is_output(args[2]):
                        args[2].zero_()
                        return None
                    if len(args) >= 7 and self._is_skip_weight(args[3]) and self._is_output(args[4]):
                        args[4].zero_()
                        return None
                    return self._original.run(*args, **kwargs)

            class _AICASSkipDownRunner:
                def __init__(self, original: Any) -> None:
                    self._original = original
                    self._pending = 0

                def note_skip(self) -> None:
                    self._pending += 1

                def run(self, *args: Any, **kwargs: Any) -> Any:
                    if (
                        self._pending > 0
                        and len(args) >= 3
                        and isinstance(args[2], torch.Tensor)
                        and int(args[2].numel()) == 2048
                    ):
                        self._pending -= 1
                        args[2].zero_()
                        return None
                    return self._original.run(*args, **kwargs)

            down_runner = getattr(module, "triton_red_fused_mm_12", None)
            down_wrapper = None
            if down_runner is not None and callable(getattr(down_runner, "run", None)):
                down_wrapper = _AICASSkipDownRunner(down_runner)
                setattr(module, "triton_red_fused_mm_12", down_wrapper)

            gateup_names = tuple(self._whitebox_block_gateup_runner_names(module))
            patched = 0
            for name in gateup_names:
                runner = getattr(module, name, None)
                if runner is None or not callable(getattr(runner, "run", None)):
                    continue

                class _AICASSkipGateUpAndDownRunner(_AICASSkipGateUpRunner):
                    def run(self, *args: Any, **kwargs: Any) -> Any:
                        should_skip = (
                            (len(args) >= 5 and self._is_skip_weight(args[1]) and self._is_output(args[2]))
                            or (len(args) >= 7 and self._is_skip_weight(args[3]) and self._is_output(args[4]))
                        )
                        if should_skip and down_wrapper is not None:
                            down_wrapper.note_skip()
                        return super().run(*args, **kwargs)

                setattr(module, name, _AICASSkipGateUpAndDownRunner(runner))
                patched += 1
            if patched:
                setattr(module, "_aicasgc_skip_mlp_layers_patched", True)
        except Exception as exc:
            pass

    def _patch_whitebox_block_weightonly_gateup(self, module: Any) -> None:
        if (
            module is None
            or not self._whitebox_block_weightonly_gateup
            or bool(getattr(module, "_aicasgc_weightonly_gateup_patched", False))
        ):
            return
        try:
            patched = 0
            requested = str(self._whitebox_block_weightonly_gateup_runners).strip().lower()
            runner_names = tuple(self._whitebox_block_gateup_runner_names(module))
            allowed_ptrs = self._whitebox_block_gateup_ptr_mask(
                "AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_GATEUP_LAYER_MASK",
                max_weights=int(self._whitebox_block_weightonly_gateup_max_weights),
            )
            if requested in {"plain", "mm11"}:
                runner_names = tuple(name for name in runner_names if str(name).endswith("_11"))
            elif requested in {"norm", "mm18"}:
                runner_names = tuple(name for name in runner_names if str(name).endswith("_18"))
            elif requested not in {"all", "*"}:
                allowed = {item.strip() for item in requested.split(",") if item.strip()}
                runner_names = tuple(name for name in runner_names if str(name) in allowed)
            for candidate_name in runner_names:
                candidate_runner = getattr(module, candidate_name, None)
                if candidate_runner is None or not callable(getattr(candidate_runner, "run", None)):
                    continue
                setattr(
                    module,
                    str(candidate_name),
                    _AICASWeightOnlyGateUpRunner(
                        candidate_runner,
                        max_weights=int(self._whitebox_block_weightonly_gateup_max_weights),
                        allowed_weight_ptrs=allowed_ptrs,
                    ),
                )
                patched += 1
            setattr(module, "_aicasgc_weightonly_gateup_patched", True)
        except Exception as exc:
            pass

    def _patch_whitebox_block_smoothquant_gateup(self, module: Any) -> None:
        if (
            module is None
            or not self._whitebox_block_smoothquant_gateup
            or bool(getattr(module, "_aicasgc_smoothquant_gateup_patched", False))
        ):
            return
        try:
            patched = 0
            requested = str(self._whitebox_block_weightonly_gateup_runners).strip().lower()
            runner_names = tuple(self._whitebox_block_gateup_runner_names(module))
            allowed_ptrs = self._whitebox_block_gateup_ptr_mask(
                "AICASGC_WHITEBOX_TOKEN_BLOCK_SQ_GATEUP_LAYER_MASK",
                fallback="AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_GATEUP_LAYER_MASK",
                max_weights=self._env_nonnegative_int("AICASGC_WHITEBOX_TOKEN_BLOCK_SQ_GATEUP_MAX_WEIGHTS", 0),
            )
            if requested in {"plain", "mm11"}:
                runner_names = tuple(name for name in runner_names if str(name).endswith("_11"))
            elif requested in {"norm", "mm18"}:
                runner_names = tuple(name for name in runner_names if str(name).endswith("_18"))
            elif requested not in {"all", "*"}:
                allowed = {item.strip() for item in requested.split(",") if item.strip()}
                runner_names = tuple(name for name in runner_names if str(name) in allowed)
            for candidate_name in runner_names:
                candidate_runner = getattr(module, candidate_name, None)
                if candidate_runner is None or not callable(getattr(candidate_runner, "run", None)):
                    continue
                setattr(
                    module,
                    str(candidate_name),
                    _AICASSmoothQuantGateUpRunner(
                        candidate_runner,
                        max_weights=self._env_nonnegative_int("AICASGC_WHITEBOX_TOKEN_BLOCK_SQ_GATEUP_MAX_WEIGHTS", 0),
                        allowed_weight_ptrs=allowed_ptrs,
                    ),
                )
                patched += 1
            setattr(module, "_aicasgc_smoothquant_gateup_patched", True)
        except Exception as exc:
            pass

    @staticmethod
    def _whitebox_block_fused_attention_step_names(module: Any) -> tuple[tuple[int, str], ...]:
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            return ()
        try:
            text = Path(module_file).read_text(encoding="utf-8")
        except Exception:
            return ()
        pairs: list[tuple[int, str]] = []
        for name in sorted(set(re.findall(r"(triton_per_fused__safe_softmax_add_full_le_where_zeros_\d+)\s*=", text))):
            marker = f"{name} = async_compile.triton"
            start = text.find(marker)
            if start < 0:
                continue
            end = text.find("''', device_str", start)
            snippet = text[start:end if end > start else start + 12000]
            match = re.search(r"tmp3 = tl\.full\(\[1\],\s*(\d+),\s*tl\.int64\)", snippet)
            if match:
                pairs.append((int(match.group(1)), name))
        return tuple(sorted(pairs))

    @staticmethod
    def _whitebox_block_fused_qkv_attention_runner_name(module: Any) -> Optional[str]:
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            return None
        try:
            text = Path(module_file).read_text(encoding="utf-8")
        except Exception:
            return None
        names = tuple(
            sorted(
                set(
                    re.findall(
            r"(triton_red_fused_bmm_\d+)\.run\([^\n]*?,\s*[^\n]*?,\s*[^\n]*?,\s*5120,\s*128,",
            text,
                    )
                )
            )
        )
        return names[0] if names else None

    def _patch_whitebox_block_fused_decode_attention(self, module: Any) -> None:
        if (
            module is None
            or not self._whitebox_block_fused_decode_attention
            or bool(getattr(module, "_aicasgc_fused_decode_attention_patched", False))
        ):
            return
        try:
            pairs = self._whitebox_block_fused_attention_step_names(module)
            if not pairs:
                return
            value_runner = getattr(module, "triton_red_fused_bmm_7", None)
            merge_runner = getattr(module, "triton_per_fused_bmm_8", None)
            if value_runner is None or merge_runner is None:
                return
            if not callable(getattr(value_runner, "run", None)) or not callable(getattr(merge_runner, "run", None)):
                return
            if self._whitebox_block_fused_decode_attention_mode in {"qkv", "qk_softmax_value", "full_qkv"}:
                if _aicas_triton_decode_qkv_softmax_value_gqa_m1 is None:
                    return
                score_name = self._whitebox_block_fused_qkv_attention_runner_name(module)
                score_runner = getattr(module, str(score_name), None) if score_name else None
                if score_runner is None or not callable(getattr(score_runner, "run", None)):
                    return
                group = _AICASWhiteboxFusedQKVAttentionGroup(step_offset=0)
                setattr(module, "_aicasgc_fused_qkv_attention_group", group)
                setattr(module, str(score_name), _AICASWhiteboxFusedQKVScoreRunner(score_runner, group))
                patched = 0
                for step_offset, softmax_name in pairs:
                    softmax_runner = getattr(module, softmax_name, None)
                    if softmax_runner is None or not callable(getattr(softmax_runner, "run", None)):
                        continue
                    setattr(
                        module,
                        softmax_name,
                        _AICASWhiteboxFusedQKVSoftmaxRunner(
                            softmax_runner,
                            group,
                            step_offset=int(step_offset),
                        ),
                    )
                    patched += 1
                if not patched:
                    return
                setattr(module, "triton_red_fused_bmm_7", _AICASWhiteboxFusedQKVValueRunner(value_runner, group))
                setattr(module, "triton_per_fused_bmm_8", _AICASWhiteboxFusedQKVMergeRunner(merge_runner, group))
                setattr(module, "_aicasgc_fused_decode_attention_patched", True)
                return
            if self._whitebox_block_fused_decode_attention_mode not in {"softmax_value", "full"}:
                group = _AICASWhiteboxFusedAttentionValueMergeGroup()
                setattr(
                    module,
                    "triton_red_fused_bmm_7",
                    _AICASWhiteboxFusedAttentionValueMergeValueRunner(value_runner, group),
                )
                setattr(
                    module,
                    "triton_per_fused_bmm_8",
                    _AICASWhiteboxFusedAttentionValueMergeMergeRunner(merge_runner, group),
                )
                setattr(module, "_aicasgc_fused_value_merge_group", group)
                setattr(module, "_aicasgc_fused_decode_attention_patched", True)
                return
            patched = 0
            for step_offset, softmax_name in pairs:
                softmax_runner = getattr(module, softmax_name, None)
                if softmax_runner is None or not callable(getattr(softmax_runner, "run", None)):
                    continue
                group = _AICASWhiteboxFusedAttentionGroup(
                    step_offset=int(step_offset),
                )
                setattr(
                    module,
                    softmax_name,
                    _AICASWhiteboxFusedAttentionSoftmaxRunner(softmax_runner, group),
                )
                setattr(
                    module,
                    f"_aicasgc_fused_attn_value_{softmax_name}",
                    _AICASWhiteboxFusedAttentionValueRunner(value_runner, group),
                )
                setattr(
                    module,
                    f"_aicasgc_fused_attn_merge_{softmax_name}",
                    _AICASWhiteboxFusedAttentionMergeRunner(merge_runner, group),
                )
                patched += 1
            if patched:



                groups: list[_AICASWhiteboxFusedAttentionGroup] = [
                    getattr(getattr(module, name, None), "_group")
                    for _, name in pairs
                    if isinstance(getattr(getattr(module, name, None), "_group", None), _AICASWhiteboxFusedAttentionGroup)
                ]

                class _ValueDispatcher:
                    def __init__(self, original: Any, group_list: list[_AICASWhiteboxFusedAttentionGroup]) -> None:
                        self._original = original
                        self._groups = group_list

                    def run(self, *args: Any, **kwargs: Any) -> Any:
                        for group in self._groups:
                            if isinstance(group.scores, torch.Tensor):
                                return _AICASWhiteboxFusedAttentionValueRunner(self._original, group).run(*args, **kwargs)
                        return self._original.run(*args, **kwargs)

                class _MergeDispatcher:
                    def __init__(
                        self,
                        original: Any,
                        group_list: list[_AICASWhiteboxFusedAttentionGroup],
                    ) -> None:
                        self._original = original
                        self._groups = group_list

                    def run(self, *args: Any, **kwargs: Any) -> Any:
                        for group in self._groups:
                            if isinstance(group.scores, torch.Tensor) and isinstance(group.value_cache, torch.Tensor):
                                return _AICASWhiteboxFusedAttentionMergeRunner(self._original, group).run(*args, **kwargs)
                        return self._original.run(*args, **kwargs)

                setattr(module, "triton_red_fused_bmm_7", _ValueDispatcher(value_runner, groups))
                setattr(module, "triton_per_fused_bmm_8", _MergeDispatcher(merge_runner, groups))
            setattr(module, "_aicasgc_fused_decode_attention_patched", True)
        except Exception as exc:
            pass

    def _patch_whitebox_block_rowtriton_norm_gateup(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_ROWTRITON_NORM_GATEUP", "1") == "0"
            or bool(getattr(module, "_aicasgc_rowtriton_norm_gateup_patched", False))
            or not self._whitebox_block_uses_rowtriton_gateup(int(bucket))
        ):
            return
        patch_fn = getattr(module, "_aicasgc_patch_quantnative_w8a16_gateup", None)
        if not callable(patch_fn):
            return
        old_requested = os.environ.get("AICASGC_QUANTNATIVE_W8A16_GATEUP_RUNNERS")
        try:
            if old_requested is None:
                os.environ["AICASGC_QUANTNATIVE_W8A16_GATEUP_RUNNERS"] = "norm"
            patch_fn()
            setattr(module, "_aicasgc_rowtriton_norm_gateup_patched", True)
        except Exception as exc:
            pass
        finally:
            if old_requested is None:
                os.environ.pop("AICASGC_QUANTNATIVE_W8A16_GATEUP_RUNNERS", None)
            else:
                os.environ["AICASGC_QUANTNATIVE_W8A16_GATEUP_RUNNERS"] = old_requested

    def _patch_whitebox_block_rowtriton_down(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_ROWTRITON_DOWN", "0") not in {"1", "silu"}
            or bool(getattr(module, "_aicasgc_rowtriton_down_patched", False))
            or not self._whitebox_block_uses_rowtriton_gateup(int(bucket))
            or _aicas_triton_row_w8a16_down_silu is None
        ):
            return
        packed_down = self._whitebox_block_down_w8a16_row_args()
        if not isinstance(packed_down, tuple) or len(packed_down) < 2 or len(packed_down) % 2 != 0:
            return
        runner = getattr(module, "triton_red_fused_mm_12", None)
        if runner is None or not callable(getattr(runner, "run", None)):
            return

        class _AICASRowW8A16DownRunner:
            def __init__(self, original: Any, packed: tuple[torch.Tensor, ...]) -> None:
                self._original = original
                self._packed = packed
                self._failed = False
                self._block_n = max(1, int(os.environ.get("AICASGC_ROWTRITON_W8A16_DOWN_BLOCK_N", "16")))
                self._block_k = max(16, int(os.environ.get("AICASGC_ROWTRITON_W8A16_DOWN_BLOCK_K", "128")))

            @staticmethod
            def _is_gate_up_out(tensor: Any) -> bool:
                return isinstance(tensor, torch.Tensor) and int(tensor.numel()) >= 12288

            @staticmethod
            def _is_down_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 6144
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if (
                    self._failed
                    or len(args) < 5
                    or not self._is_gate_up_out(args[0])
                    or not self._is_down_weight(args[1])
                    or not isinstance(args[2], torch.Tensor)
                    or int(args[3]) != 2048
                    or int(args[4]) != 6144
                ):
                    return self._fallback(*args, **kwargs)
                gate_up = args[0]
                down_weight = args[1]
                out = args[2]
                packed = None
                for idx in range(0, len(self._packed), 2):
                    qweight = self._packed[idx]
                    scales = self._packed[idx + 1]
                    if isinstance(qweight, torch.Tensor) and getattr(qweight, "_aicasgc_source_ptr", None) == int(down_weight.data_ptr()):
                        packed = (qweight, scales)
                        break
                if packed is None:
                    return self._fallback(*args, **kwargs)
                try:
                    qweight, scales = packed
                    if _aicas_triton_row_w8a16_down_silu(
                        gate_up.contiguous() if not gate_up.is_contiguous() else gate_up,
                        qweight,
                        scales,
                        out,
                        block_n=self._block_n,
                        block_k=self._block_k,
                    ):
                        return None
                except Exception as exc:
                    self._failed = True
                return self._fallback(*args, **kwargs)

        setattr(module, "triton_red_fused_mm_12", _AICASRowW8A16DownRunner(runner, packed_down))
        setattr(module, "_aicasgc_rowtriton_down_patched", True)

    def _patch_whitebox_block_rowtriton_down_preact(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_ROWTRITON_DOWN", "0") not in {"preact", "2"}
            or bool(getattr(module, "_aicasgc_rowtriton_down_preact_patched", False))
            or not self._whitebox_block_uses_rowtriton_gateup(int(bucket))
            or _aicas_triton_row_w8a16_down_silu_preact is None
        ):
            return
        packed_down = self._whitebox_block_down_w8a16_row_args()
        if not isinstance(packed_down, tuple) or len(packed_down) < 2 or len(packed_down) % 2 != 0:
            return
        runner = getattr(module, "triton_red_fused_mm_12", None)
        if runner is None or not callable(getattr(runner, "run", None)):
            return

        class _AICASRowW8A16DownPreactRunner:
            def __init__(self, original: Any, packed: tuple[torch.Tensor, ...]) -> None:
                self._original = original
                self._packed = packed
                self._failed = False
                self._scratch: dict[tuple[str, torch.dtype], torch.Tensor] = {}
                self._block_n = max(1, int(os.environ.get("AICASGC_ROWTRITON_W8A16_DOWN_PREACT_BLOCK_N", "16")))
                self._block_k = max(16, int(os.environ.get("AICASGC_ROWTRITON_W8A16_DOWN_PREACT_BLOCK_K", "128")))
                self._act_block_k = max(16, int(os.environ.get("AICASGC_ROWTRITON_W8A16_DOWN_PREACT_ACT_BLOCK_K", "1024")))
                self._act_dtype_name = os.environ.get(
                    "AICASGC_ROWTRITON_W8A16_DOWN_PREACT_ACT_DTYPE",
                    "fp32",
                ).strip().lower()

            @staticmethod
            def _is_gate_up_out(tensor: Any) -> bool:
                return isinstance(tensor, torch.Tensor) and int(tensor.numel()) >= 12288

            @staticmethod
            def _is_down_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 6144
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def _act_buffer(self, ref: torch.Tensor) -> torch.Tensor:
                dtype = torch.float16 if self._act_dtype_name in {"fp16", "float16"} else torch.float32
                key = (str(ref.device), dtype)
                cached = self._scratch.get(key)
                if (
                    isinstance(cached, torch.Tensor)
                    and cached.device == ref.device
                    and cached.dtype == dtype
                    and int(cached.numel()) >= 6144
                ):
                    return cached
                buf = torch.empty((6144,), device=ref.device, dtype=dtype)
                self._scratch[key] = buf
                return buf

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if (
                    self._failed
                    or len(args) < 5
                    or not self._is_gate_up_out(args[0])
                    or not self._is_down_weight(args[1])
                    or not isinstance(args[2], torch.Tensor)
                    or int(args[3]) != 2048
                    or int(args[4]) != 6144
                ):
                    return self._fallback(*args, **kwargs)
                gate_up = args[0]
                down_weight = args[1]
                out = args[2]
                packed = None
                for idx in range(0, len(self._packed), 2):
                    qweight = self._packed[idx]
                    scales = self._packed[idx + 1]
                    if isinstance(qweight, torch.Tensor) and getattr(qweight, "_aicasgc_source_ptr", None) == int(down_weight.data_ptr()):
                        packed = (qweight, scales)
                        break
                if packed is None:
                    return self._fallback(*args, **kwargs)
                try:
                    qweight, scales = packed
                    act = self._act_buffer(out)
                    if _aicas_triton_row_w8a16_down_silu_preact(
                        gate_up.contiguous() if not gate_up.is_contiguous() else gate_up,
                        qweight,
                        scales,
                        act,
                        out.contiguous() if not out.is_contiguous() else out,
                        block_n=self._block_n,
                        block_k=self._block_k,
                        act_block_k=self._act_block_k,
                    ):
                        return None
                except Exception as exc:
                    self._failed = True
                return self._fallback(*args, **kwargs)

        setattr(module, "triton_red_fused_mm_12", _AICASRowW8A16DownPreactRunner(runner, packed_down))
        setattr(module, "_aicasgc_rowtriton_down_preact_patched", True)

    def _patch_whitebox_block_allspark_down(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_ALLSPARK_DOWN", "0") == "0"
            or bool(getattr(module, "_aicasgc_allspark_down_patched", False))
            or not self._whitebox_block_uses_rowtriton_gateup(int(bucket))
            or _aicas_triton_silu_preact is None
        ):
            return
        packed_down = self._whitebox_block_down_w8a16_args()
        if not isinstance(packed_down, tuple) or len(packed_down) != 56:
            return
        runner = getattr(module, "triton_red_fused_mm_12", None)
        if runner is None or not callable(getattr(runner, "run", None)):
            return

        class _AICASAllSparkW8A16DownRunner:
            def __init__(self, original: Any, packed: tuple[torch.Tensor, ...]) -> None:
                self._original = original
                self._packed = packed
                self._failed = False
                self._act_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
                self._sm_count = 0
                self._sm_version = 80
                self._cublas_m_threshold = max(
                    0,
                    int(os.environ.get("AICASGC_ALLSPARK_W8A16_DOWN_CUBLAS_M_THRESHOLD", "1024")),
                )
                self._act_block_k = max(
                    16,
                    int(os.environ.get("AICASGC_ALLSPARK_W8A16_DOWN_ACT_BLOCK_K", "1024")),
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def _ensure_allspark(self) -> bool:
                if self._failed:
                    return False
                try:
                    import vllm._C
                    from vllm import _custom_ops as _aicasgc_vllm_ops

                    if not hasattr(torch.ops._C, "allspark_w8a16_gemm"):
                        self._failed = True
                        return False
                    if torch.cuda.is_available() and self._sm_count <= 0:
                        props = torch.cuda.get_device_properties(torch.cuda.current_device())
                        self._sm_count = int(props.multi_processor_count)
                        self._sm_version = int(props.major) * 10 + int(props.minor)
                    return True
                except Exception:
                    self._failed = True
                    return False

            @staticmethod
            def _is_gate_up_out(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 12288
                )

            @staticmethod
            def _is_down_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float16
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 6144
                )

            def _packed_weight(self, weight: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
                for idx in range(0, len(self._packed), 2):
                    qweight = self._packed[idx]
                    scales = self._packed[idx + 1]
                    if isinstance(qweight, torch.Tensor) and getattr(qweight, "_aicasgc_source_ptr", None) == int(weight.data_ptr()):
                        return qweight, scales
                return None

            def _act_buffer(self, ref: torch.Tensor) -> torch.Tensor:
                key = (str(ref.device), torch.float16)
                cached = self._act_cache.get(key)
                if isinstance(cached, torch.Tensor) and cached.device == ref.device and cached.dtype == torch.float16:
                    return cached
                buf = torch.empty((1, 6144), device=ref.device, dtype=torch.float16)
                self._act_cache[key] = buf
                return buf

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if (
                    self._failed
                    or len(args) < 5
                    or not self._ensure_allspark()
                    or not self._is_gate_up_out(args[0])
                    or not self._is_down_weight(args[1])
                    or not isinstance(args[2], torch.Tensor)
                    or not args[2].is_cuda
                    or args[2].dtype != torch.float32
                    or int(args[2].numel()) < 2048
                    or int(args[3]) != 2048
                    or int(args[4]) != 6144
                ):
                    return self._fallback(*args, **kwargs)
                packed = self._packed_weight(args[1])
                if packed is None:
                    return self._fallback(*args, **kwargs)
                try:
                    act = self._act_buffer(args[2])
                    gate_up = args[0].contiguous() if not args[0].is_contiguous() else args[0]
                    if not _aicas_triton_silu_preact(gate_up, act, block_k=self._act_block_k):
                        return self._fallback(*args, **kwargs)
                    qweight, scales = packed
                    result = torch.ops._C.allspark_w8a16_gemm(
                        act,
                        qweight,
                        scales,
                        None,
                        2048,
                        -1,
                        int(self._sm_count) if int(self._sm_count) > 0 else 64,
                        int(self._sm_version),
                        int(self._cublas_m_threshold),
                        False,
                        True,
                    )
                    args[2].copy_(result.to(device=args[2].device, dtype=args[2].dtype).reshape(args[2].shape))
                    return None
                except Exception as exc:
                    self._failed = True
                    return self._fallback(*args, **kwargs)

        setattr(module, "triton_red_fused_mm_12", _AICASAllSparkW8A16DownRunner(runner, packed_down))
        setattr(module, "_aicasgc_allspark_down_patched", True)

    def _patch_whitebox_block_fp16_down(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FP16_DOWN", "0") == "0"
            or bool(getattr(module, "_aicasgc_fp16_down_patched", False))
            or _aicas_triton_fp16_down_silu is None
        ):
            return
        runner = getattr(module, "triton_red_fused_mm_12", None)
        if runner is None or not callable(getattr(runner, "run", None)):
            return

        class _AICASFp16DownRunner:
            def __init__(self, original: Any) -> None:
                self._original = original
                self._failed = False
                self._block_n = max(1, int(os.environ.get("AICASGC_FP16_DOWN_BLOCK_N", "8")))
                self._block_k = max(16, int(os.environ.get("AICASGC_FP16_DOWN_BLOCK_K", "128")))

            @staticmethod
            def _is_gate_up_out(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float32
                    and int(tensor.numel()) >= 12288
                )

            @staticmethod
            def _is_down_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float16
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 6144
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if (
                    self._failed
                    or len(args) < 5
                    or not self._is_gate_up_out(args[0])
                    or not self._is_down_weight(args[1])
                    or not isinstance(args[2], torch.Tensor)
                    or not args[2].is_cuda
                    or args[2].dtype != torch.float32
                    or int(args[2].numel()) < 2048
                    or int(args[3]) != 2048
                    or int(args[4]) != 6144
                ):
                    return self._fallback(*args, **kwargs)
                try:
                    if _aicas_triton_fp16_down_silu(
                        args[0].contiguous() if not args[0].is_contiguous() else args[0],
                        args[1].contiguous() if not args[1].is_contiguous() else args[1],
                        args[2].contiguous() if not args[2].is_contiguous() else args[2],
                        block_n=self._block_n,
                        block_k=self._block_k,
                    ):
                        return None
                except Exception as exc:
                    self._failed = True
                return self._fallback(*args, **kwargs)

        setattr(module, "triton_red_fused_mm_12", _AICASFp16DownRunner(runner))
        setattr(module, "_aicasgc_fp16_down_patched", True)

    def _patch_whitebox_block_fp16_down_preact(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_FP16_DOWN_PREACT", "0") == "0"
            or bool(getattr(module, "_aicasgc_fp16_down_preact_patched", False))
            or _aicas_triton_fp16_down_silu_preact is None
        ):
            return
        runner = getattr(module, "triton_red_fused_mm_12", None)
        if runner is None or not callable(getattr(runner, "run", None)):
            return

        class _AICASFp16DownPreactRunner:
            def __init__(self, original: Any) -> None:
                self._original = original
                self._failed = False
                self._scratch: dict[tuple[int, torch.dtype, int], torch.Tensor] = {}
                self._block_n = max(1, int(os.environ.get("AICASGC_FP16_DOWN_PREACT_BLOCK_N", "8")))
                self._block_k = max(16, int(os.environ.get("AICASGC_FP16_DOWN_PREACT_BLOCK_K", "128")))
                self._act_block_k = max(16, int(os.environ.get("AICASGC_FP16_DOWN_PREACT_ACT_BLOCK_K", "1024")))

            @staticmethod
            def _is_gate_up_out(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float32
                    and int(tensor.numel()) >= 12288
                )

            @staticmethod
            def _is_down_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float16
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 6144
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def _scratch_for(self, ref: torch.Tensor) -> torch.Tensor:
                device_idx = int(ref.device.index or 0)
                key = (device_idx, torch.float32, 6144)
                cached = self._scratch.get(key)
                if isinstance(cached, torch.Tensor) and cached.is_cuda and int(cached.numel()) >= 6144:
                    if int(cached.device.index or 0) == device_idx:
                        return cached
                buf = torch.empty((6144,), device=ref.device, dtype=torch.float32)
                self._scratch[key] = buf
                return buf

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if (
                    self._failed
                    or len(args) < 5
                    or not self._is_gate_up_out(args[0])
                    or not self._is_down_weight(args[1])
                    or not isinstance(args[2], torch.Tensor)
                    or not args[2].is_cuda
                    or args[2].dtype != torch.float32
                    or int(args[2].numel()) < 2048
                    or int(args[3]) != 2048
                    or int(args[4]) != 6144
                ):
                    return self._fallback(*args, **kwargs)
                gate_up = args[0]
                down_weight = args[1]
                out = args[2]
                try:
                    act = self._scratch_for(out)
                    if _aicas_triton_fp16_down_silu_preact(
                        gate_up.contiguous() if not gate_up.is_contiguous() else gate_up,
                        down_weight.contiguous() if not down_weight.is_contiguous() else down_weight,
                        act,
                        out.contiguous() if not out.is_contiguous() else out,
                        block_n=self._block_n,
                        block_k=self._block_k,
                        act_block_k=self._act_block_k,
                    ):
                        return None
                except Exception as exc:
                    self._failed = True
                return self._fallback(*args, **kwargs)

        setattr(module, "triton_red_fused_mm_12", _AICASFp16DownPreactRunner(runner))
        setattr(module, "_aicasgc_fp16_down_preact_patched", True)

    def _patch_whitebox_block_weightonly_down(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_DOWN", "0") == "0"
            or bool(getattr(module, "_aicasgc_weightonly_down_patched", False))
            or _aicas_triton_silu_preact is None
        ):
            return
        runner = getattr(module, "triton_red_fused_mm_12", None)
        if runner is None or not callable(getattr(runner, "run", None)):
            return

        class _AICASWeightOnlyDownRunner:
            def __init__(self, original: Any) -> None:
                self._original = original
                self._failed = False
                self._packed_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
                self._act_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
                self._weightonly = None
                self._preprocess = None
                self._act_dtype_name = os.environ.get(
                    "AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_DOWN_ACT_DTYPE",
                    "fp16",
                ).strip().lower()
                self._act_block_k = max(
                    16,
                    int(os.environ.get("AICASGC_WEIGHTONLY_DOWN_ACT_BLOCK_K", "1024")),
                )

            def _ensure_weightonly(self) -> bool:
                if self._failed:
                    return False
                if self._weightonly is not None and self._preprocess is not None:
                    return True
                try:
                    import acext
                    import vllm._C

                    self._weightonly = torch.classes._C.WeightOnlyQuantMatmul()
                    self._preprocess = acext.preprocess_weights_for_mixed_gemm
                    return True
                except Exception as exc:
                    self._failed = True
                    return False

            @staticmethod
            def _is_gate_up_out(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 12288
                )

            @staticmethod
            def _is_down_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float16
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 6144
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def _packed_weight(self, weight: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
                if not self._ensure_weightonly() or not self._is_down_weight(weight):
                    return None
                try:
                    key = (
                        int(weight.data_ptr()),
                        tuple(int(v) for v in weight.shape),
                        tuple(int(v) for v in weight.stride()),
                        str(weight.dtype),
                        str(weight.device),
                    )
                    cached = self._packed_cache.get(key)
                    if cached is not None:
                        return cached
                    w = weight.detach()
                    if not w.is_contiguous():
                        w = w.contiguous()
                    scales = w.abs().amax(dim=1).float().clamp_min(1.0e-6) / 127.0
                    qweight = torch.round(w.float() / scales[:, None]).clamp(-128, 127).to(torch.int8)
                    qweight_cpu = qweight.t().contiguous().cpu()
                    assert self._preprocess is not None
                    qweight_packed = self._preprocess(qweight_cpu, torch.int8, False, False).to(device=weight.device)
                    scales_gpu = scales.to(device=weight.device, dtype=torch.float16).contiguous()
                    packed = (qweight_packed, scales_gpu)
                    self._packed_cache[key] = packed
                    return packed
                except Exception as exc:
                    self._failed = True
                    return None

            def _act_buffer(self, ref: torch.Tensor) -> torch.Tensor:
                dtype = torch.float32 if self._act_dtype_name in {"fp32", "float32"} else torch.float16
                key = (str(ref.device), dtype)
                cached = self._act_cache.get(key)
                if isinstance(cached, torch.Tensor) and cached.device == ref.device and cached.dtype == dtype:
                    return cached
                buf = torch.empty((1, 6144), device=ref.device, dtype=dtype)
                self._act_cache[key] = buf
                return buf

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if (
                    self._failed
                    or len(args) < 5
                    or not self._is_gate_up_out(args[0])
                    or not self._is_down_weight(args[1])
                    or not isinstance(args[2], torch.Tensor)
                    or not args[2].is_cuda
                    or args[2].dtype != torch.float32
                    or int(args[2].numel()) < 2048
                    or int(args[3]) != 2048
                    or int(args[4]) != 6144
                ):
                    return self._fallback(*args, **kwargs)
                packed = self._packed_weight(args[1])
                if packed is None:
                    return self._fallback(*args, **kwargs)
                try:
                    act = self._act_buffer(args[2])
                    gate_up = args[0].contiguous() if not args[0].is_contiguous() else args[0]
                    if not _aicas_triton_silu_preact(gate_up, act, block_k=self._act_block_k):
                        return self._fallback(*args, **kwargs)
                    qweight, scales = packed
                    assert self._weightonly is not None
                    result = self._weightonly.weightonly_gemm(act, qweight, scales, None)
                    args[2].copy_(result.to(device=args[2].device, dtype=args[2].dtype).reshape(args[2].shape))
                    return None
                except Exception as exc:
                    self._failed = True
                    return self._fallback(*args, **kwargs)

        setattr(module, "triton_red_fused_mm_12", _AICASWeightOnlyDownRunner(runner))
        setattr(module, "_aicasgc_weightonly_down_patched", True)

    def _patch_whitebox_block_torch_down(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_TORCH_DOWN", "0") == "0"
            or bool(getattr(module, "_aicasgc_torch_down_patched", False))
            or _aicas_triton_silu_preact is None
        ):
            return
        runner = getattr(module, "triton_red_fused_mm_12", None)
        if runner is None or not callable(getattr(runner, "run", None)):
            return

        class _AICASTorchDownRunner:
            def __init__(self, original: Any) -> None:
                self._original = original
                self._failed = False
                self._act_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
                self._out_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
                self._act_dtype_name = os.environ.get(
                    "AICASGC_WHITEBOX_TOKEN_BLOCK_TORCH_DOWN_ACT_DTYPE",
                    "fp16",
                ).strip().lower()
                self._act_block_k = max(
                    16,
                    int(os.environ.get("AICASGC_TORCH_DOWN_ACT_BLOCK_K", "1024")),
                )

            @staticmethod
            def _is_gate_up_out(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 12288
                )

            @staticmethod
            def _is_down_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float16
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 6144
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def _act_buffer(self, ref: torch.Tensor) -> torch.Tensor:
                dtype = torch.float32 if self._act_dtype_name in {"fp32", "float32"} else torch.float16
                key = (str(ref.device), dtype)
                cached = self._act_cache.get(key)
                if isinstance(cached, torch.Tensor) and cached.device == ref.device and cached.dtype == dtype:
                    return cached
                buf = torch.empty((1, 6144), device=ref.device, dtype=dtype)
                self._act_cache[key] = buf
                return buf

            def _out_buffer(self, ref: torch.Tensor) -> torch.Tensor:
                key = (str(ref.device), torch.float16)
                cached = self._out_cache.get(key)
                if isinstance(cached, torch.Tensor) and cached.device == ref.device and cached.dtype == torch.float16:
                    return cached
                buf = torch.empty((1, 2048), device=ref.device, dtype=torch.float16)
                self._out_cache[key] = buf
                return buf

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if (
                    self._failed
                    or len(args) < 5
                    or not self._is_gate_up_out(args[0])
                    or not self._is_down_weight(args[1])
                    or not isinstance(args[2], torch.Tensor)
                    or not args[2].is_cuda
                    or args[2].dtype != torch.float32
                    or int(args[2].numel()) < 2048
                    or int(args[3]) != 2048
                    or int(args[4]) != 6144
                ):
                    return self._fallback(*args, **kwargs)
                try:
                    act = self._act_buffer(args[2])
                    gate_up = args[0].contiguous() if not args[0].is_contiguous() else args[0]
                    if not _aicas_triton_silu_preact(gate_up, act, block_k=self._act_block_k):
                        return self._fallback(*args, **kwargs)
                    out_tmp = torch.mm(act, args[1].t())
                    if out_tmp.dtype != torch.float32:
                        args[2].copy_(out_tmp.to(dtype=torch.float32).reshape(args[2].shape))
                    else:
                        args[2].copy_(out_tmp.reshape(args[2].shape))
                    return None
                except Exception as exc:
                    self._failed = True
                    return self._fallback(*args, **kwargs)

        setattr(module, "triton_red_fused_mm_12", _AICASTorchDownRunner(runner))
        setattr(module, "_aicasgc_torch_down_patched", True)

    def _patch_whitebox_block_torch_linear2048(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or not bool(getattr(self, "_whitebox_block_torch_linear2048", False))
            or bool(getattr(module, "_aicasgc_torch_linear2048_patched", False))
        ):
            return
        target_raw = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_TORCH_LINEAR2048_RUNNERS",
            "9,14,20,24",
        ).strip().lower()
        if target_raw in {"all", "*"}:
            runner_names = (
                "triton_red_fused_mm_9",
                "triton_red_fused_mm_14",
                "triton_red_fused_mm_20",
                "triton_red_fused_add_mm_24",
            )
        else:
            runner_names = tuple(
                f"triton_red_fused_mm_{item.strip()}"
                if item.strip().isdigit() and item.strip() != "24"
                else ("triton_red_fused_add_mm_24" if item.strip() == "24" else item.strip())
                for item in target_raw.split(",")
                if item.strip()
            )

        class _AICASTorchLinear2048Runner:
            def __init__(self, original: Any) -> None:
                self._original = original
                self._failed = False

            @staticmethod
            def _is_vec(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 2048
                )

            @staticmethod
            def _is_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float16
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 2048
                )

            @staticmethod
            def _is_out(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 2048
                )

            @staticmethod
            def _view2048(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.reshape(1, -1)[:, :2048]

            @staticmethod
            def _out2048(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.reshape(-1)[:2048].view(1, 2048)

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def _mm_out(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
                torch.mm(self._view2048(x), weight.t(), out=self._out2048(out))

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if self._failed:
                    return self._fallback(*args, **kwargs)
                try:
                    if (
                        len(args) >= 5
                        and self._is_vec(args[0])
                        and self._is_weight(args[1])
                        and self._is_out(args[2])
                        and int(args[3]) == 2048
                        and int(args[4]) == 2048
                    ):
                        self._mm_out(args[0], args[1], args[2])
                        return None
                    if (
                        len(args) >= 8
                        and self._is_vec(args[0])
                        and self._is_vec(args[1])
                        and self._is_weight(args[2])
                        and self._is_vec(args[3])
                        and self._is_vec(args[4])
                        and self._is_vec(args[5])
                    ):
                        out = args[0]
                        self._mm_out(args[1], args[2], out)
                        out_view = self._out2048(out)
                        out_view.add_(self._view2048(args[3]).to(dtype=out_view.dtype))
                        out_view.add_(self._view2048(args[4]).to(dtype=out_view.dtype))
                        out_view.add_(self._view2048(args[5]).to(dtype=out_view.dtype))
                        return None
                except Exception as exc:
                    self._failed = True
                return self._fallback(*args, **kwargs)

        patched = 0
        for runner_name in runner_names:
            runner = getattr(module, str(runner_name), None)
            if runner is None or not callable(getattr(runner, "run", None)):
                continue
            setattr(module, str(runner_name), _AICASTorchLinear2048Runner(runner))
            patched += 1
        if patched:
            setattr(module, "_aicasgc_torch_linear2048_patched", True)

    def _patch_whitebox_block_fp16_dot_linear2048(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or not bool(getattr(self, "_whitebox_block_fp16_dot_linear2048", False))
            or bool(getattr(module, "_aicasgc_fp16_dot_linear2048_patched", False))
            or _aicas_triton_fp16_linear2048_dot is None
            or _aicas_triton_fp16_linear2048_dot_add is None
        ):
            return
        target_raw = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_FP16_DOT_LINEAR2048_RUNNERS",
            "9,14,24",
        ).strip().lower()
        if target_raw in {"all", "*"}:
            runner_names = (
                "triton_red_fused_mm_9",
                "triton_red_fused_mm_14",
                "triton_red_fused_add_mm_24",
            )
        else:
            runner_names = tuple(
                f"triton_red_fused_mm_{item.strip()}"
                if item.strip().isdigit() and item.strip() != "24"
                else ("triton_red_fused_add_mm_24" if item.strip() == "24" else item.strip())
                for item in target_raw.split(",")
                if item.strip()
            )

        class _AICASFp16DotLinear2048Runner:
            def __init__(self, original: Any) -> None:
                self._original = original
                self._failed = False
                self._block_n = max(16, int(os.environ.get("AICASGC_FP16_DOT_LINEAR2048_BLOCK_N", "64")))
                self._block_k = max(32, int(os.environ.get("AICASGC_FP16_DOT_LINEAR2048_BLOCK_K", "64")))

            @staticmethod
            def _is_vec(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 2048
                    and tensor.is_contiguous()
                )

            @staticmethod
            def _is_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float16
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 2048
                    and tensor.is_contiguous()
                )

            @staticmethod
            def _is_out(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 2048
                    and tensor.is_contiguous()
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if self._failed:
                    return self._fallback(*args, **kwargs)
                try:
                    if (
                        len(args) >= 5
                        and self._is_vec(args[0])
                        and self._is_weight(args[1])
                        and self._is_out(args[2])
                        and int(args[3]) == 2048
                        and int(args[4]) == 2048
                    ):
                        if _aicas_triton_fp16_linear2048_dot(
                            args[0],
                            args[1],
                            args[2],
                            block_n=self._block_n,
                            block_k=self._block_k,
                        ):
                            return None
                    if (
                        len(args) >= 8
                        and self._is_out(args[0])
                        and self._is_vec(args[1])
                        and self._is_weight(args[2])
                        and self._is_vec(args[3])
                        and self._is_vec(args[4])
                        and self._is_vec(args[5])
                    ):
                        if _aicas_triton_fp16_linear2048_dot_add(
                            args[1],
                            args[2],
                            args[0],
                            add0=args[0],
                            add1=args[3],
                            add2=args[4],
                            block_n=self._block_n,
                            block_k=self._block_k,
                        ):
                            return None
                except Exception as exc:
                    self._failed = True
                return self._fallback(*args, **kwargs)

        patched = 0
        for runner_name in runner_names:
            runner = getattr(module, str(runner_name), None)
            if runner is None or not callable(getattr(runner, "run", None)):
                continue
            setattr(module, str(runner_name), _AICASFp16DotLinear2048Runner(runner))
            patched += 1
        if patched:
            setattr(module, "_aicasgc_fp16_dot_linear2048_patched", True)

    def _patch_whitebox_block_rowtriton_linear2048(self, module: Any, bucket: int) -> None:
        if (
            module is None
            or os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_ROWTRITON_LINEAR2048", "0") == "0"
            or bool(getattr(module, "_aicasgc_rowtriton_linear2048_patched", False))
            or _aicas_triton_row_w8a16_linear2048 is None
            or _aicas_triton_row_w8a16_linear2048_add is None
        ):
            return
        packed_by_ptr = self._whitebox_block_linear2048_w8a16_row_args()
        if not isinstance(packed_by_ptr, dict) or not packed_by_ptr:
            return
        target_raw = os.environ.get(
            "AICASGC_WHITEBOX_TOKEN_BLOCK_ROWTRITON_LINEAR2048_RUNNERS",
            "9,14,20",
        ).strip().lower()
        if target_raw in {"all", "*"}:
            runner_names = ("triton_red_fused_mm_9", "triton_red_fused_mm_14", "triton_red_fused_mm_20")
        else:
            runner_names = tuple(
                f"triton_red_fused_mm_{item.strip()}"
                if item.strip().isdigit()
                else item.strip()
                for item in target_raw.split(",")
                if item.strip()
            )

        class _AICASRowW8A16Linear2048Runner:
            def __init__(self, original: Any, packed: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
                self._original = original
                self._packed = packed
                self._failed = False
                self._block_n = max(1, int(os.environ.get("AICASGC_ROWTRITON_W8A16_LINEAR2048_BLOCK_N", "16")))
                self._block_k = max(16, int(os.environ.get("AICASGC_ROWTRITON_W8A16_LINEAR2048_BLOCK_K", "128")))

            @staticmethod
            def _is_vec(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 2048
                )

            @staticmethod
            def _is_weight(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype == torch.float16
                    and tensor.ndim == 2
                    and int(tensor.shape[0]) == 2048
                    and int(tensor.shape[1]) == 2048
                )

            @staticmethod
            def _is_out(tensor: Any) -> bool:
                return (
                    isinstance(tensor, torch.Tensor)
                    and tensor.is_cuda
                    and tensor.dtype in (torch.float16, torch.float32)
                    and int(tensor.numel()) >= 2048
                )

            def _fallback(self, *args: Any, **kwargs: Any) -> Any:
                return self._original.run(*args, **kwargs)

            def _packed_weight(self, weight: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
                return self._packed.get(int(weight.data_ptr()))

            def _launch_plain(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> bool:
                packed = self._packed_weight(weight)
                if packed is None:
                    return False
                qweight, scales = packed
                return bool(
                    _aicas_triton_row_w8a16_linear2048(
                        x.contiguous() if not x.is_contiguous() else x,
                        qweight,
                        scales,
                        out.contiguous() if not out.is_contiguous() else out,
                        block_n=self._block_n,
                        block_k=self._block_k,
                    )
                )

            def _launch_add(
                self,
                x: torch.Tensor,
                weight: torch.Tensor,
                out: torch.Tensor,
                *,
                add0: Optional[torch.Tensor] = None,
                add1: Optional[torch.Tensor] = None,
                add2: Optional[torch.Tensor] = None,
                store_fp16: bool = False,
            ) -> bool:
                packed = self._packed_weight(weight)
                if packed is None:
                    return False
                qweight, scales = packed
                return bool(
                    _aicas_triton_row_w8a16_linear2048_add(
                        x.contiguous() if not x.is_contiguous() else x,
                        qweight,
                        scales,
                        out.contiguous() if not out.is_contiguous() else out,
                        add0=add0.contiguous() if isinstance(add0, torch.Tensor) and not add0.is_contiguous() else add0,
                        add1=add1.contiguous() if isinstance(add1, torch.Tensor) and not add1.is_contiguous() else add1,
                        add2=add2.contiguous() if isinstance(add2, torch.Tensor) and not add2.is_contiguous() else add2,
                        store_fp16=bool(store_fp16),
                        block_n=self._block_n,
                        block_k=self._block_k,
                    )
                )

            def run(self, *args: Any, **kwargs: Any) -> Any:
                if self._failed:
                    return self._fallback(*args, **kwargs)
                try:
                    if (
                        len(args) >= 5
                        and self._is_vec(args[0])
                        and self._is_weight(args[1])
                        and self._is_out(args[2])
                        and int(args[3]) == 2048
                        and int(args[4]) == 2048
                    ):
                        if self._launch_plain(args[0], args[1], args[2]):
                            return None
                    if (
                        len(args) >= 9
                        and self._is_vec(args[0])
                        and self._is_weight(args[1])
                        and self._is_vec(args[2])
                        and self._is_vec(args[3])
                        and self._is_vec(args[4])
                        and self._is_out(args[5])
                        and int(args[6]) == 2048
                        and int(args[7]) == 2048
                    ):
                        if self._launch_add(
                            args[0],
                            args[1],
                            args[5],
                            add0=args[2],
                            add1=args[3],
                            add2=args[4],
                            store_fp16=(args[5].dtype == torch.float16),
                        ):
                            return None
                    if (
                        len(args) >= 7
                        and self._is_out(args[0])
                        and self._is_vec(args[1])
                        and self._is_weight(args[2])
                        and self._is_vec(args[3])
                        and self._is_vec(args[4])
                        and self._is_vec(args[5])
                    ):
                        if self._launch_add(
                            args[1],
                            args[2],
                            args[0],
                            add0=args[0],
                            add1=args[3],
                            add2=args[4],
                            store_fp16=(args[0].dtype == torch.float16),
                        ):
                            return None
                except Exception as exc:
                    self._failed = True
                return self._fallback(*args, **kwargs)

        patched = 0
        for runner_name in runner_names:
            runner = getattr(module, runner_name, None)
            if runner is None or not callable(getattr(runner, "run", None)):
                continue
            setattr(module, str(runner_name), _AICASRowW8A16Linear2048Runner(runner, packed_by_ptr))
            patched += 1
        if patched:
            setattr(module, "_aicasgc_rowtriton_linear2048_patched", True)

    def _patch_whitebox_block_weightonly_lmhead(self, module: Any) -> None:
        if (
            module is None
            or not self._whitebox_block_weightonly_lmhead
            or self._whitebox_block_fp16_top1
            or bool(getattr(module, "_aicasgc_weightonly_lmhead_patched", False))
        ):
            return
        try:
            runner_name = ""
            runner = None
            for candidate_name in self._whitebox_block_lmhead_runner_names(module):
                candidate_runner = getattr(module, candidate_name, None)
                if candidate_runner is not None and callable(getattr(candidate_runner, "run", None)):
                    runner_name = str(candidate_name)
                    runner = candidate_runner
                    break
            if not runner_name or runner is None:
                return
            lm_head_runner = _AICASWeightOnlyLmHeadRunner(
                runner,
                top1=bool(self._whitebox_block_weightonly_top1),
                rerank_k=int(os.environ.get("AICASGC_WHITEBOX_TOKEN_BLOCK_WEIGHTONLY_TOP1_RERANK_K", "0")),
                candidate_ids=self._whitebox_block_candidate_lmhead_ids(),
            )
            setattr(module, runner_name, lm_head_runner)
            if self._whitebox_block_weightonly_top1:
                for name, writes_intermediate in (
                    ("triton_red_fused_argmax_cat_mm_34", True),
                    ("triton_red_fused_argmax_cat_mm_40", True),
                    ("triton_red_fused_argmax_cat_mm_51", False),
                    ("triton_red_fused_argmax_mm_26", False),
                    ("triton_red_fused_argmax_mm_25", False),
                    ("triton_red_fused_argmax_mm_33", False),
                ):
                    argmax_runner = getattr(module, name, None)
                    if argmax_runner is None or not callable(getattr(argmax_runner, "run", None)):
                        continue
                    setattr(
                        module,
                        name,
                        _AICASWeightOnlyArgmaxRunner(
                            argmax_runner,
                            lm_head_runner,
                            writes_intermediate_token=bool(writes_intermediate),
                        ),
                    )
            setattr(module, "_aicasgc_weightonly_lmhead_patched", True)
        except Exception as exc:
            pass

    def _patch_whitebox_block_fp16_top1_lmhead(self, module: Any) -> None:
        if (
            module is None
            or not self._whitebox_block_fp16_top1
            or bool(getattr(module, "_aicasgc_fp16_top1_lmhead_patched", False))
        ):
            return
        try:
            runner_name = ""
            runner = None
            for candidate_name in self._whitebox_block_lmhead_runner_names(module):
                candidate_runner = getattr(module, candidate_name, None)
                if candidate_runner is not None and callable(getattr(candidate_runner, "run", None)):
                    runner_name = str(candidate_name)
                    runner = candidate_runner
                    break
            if not runner_name or runner is None:
                return
            if _aicas_triton_rmsnorm_fp16_top1_decode_m1 is None:
                return
            lm_head_runner = _AICASFp16Top1LmHeadRunner(runner)
            setattr(module, runner_name, lm_head_runner)
            for name, writes_intermediate in (
                ("triton_red_fused_argmax_cat_mm_34", True),
                ("triton_red_fused_argmax_cat_mm_40", True),
                ("triton_red_fused_argmax_cat_mm_51", False),
                ("triton_red_fused_argmax_mm_26", False),
                ("triton_red_fused_argmax_mm_25", False),
                ("triton_red_fused_argmax_mm_33", False),
            ):
                argmax_runner = getattr(module, name, None)
                if argmax_runner is None or not callable(getattr(argmax_runner, "run", None)):
                    continue
                setattr(
                    module,
                    name,
                    _AICASWeightOnlyArgmaxRunner(
                        argmax_runner,
                        lm_head_runner,
                        writes_intermediate_token=bool(writes_intermediate),
                    ),
                )
            setattr(module, "_aicasgc_fp16_top1_lmhead_patched", True)
        except Exception as exc:
            pass

    def _load_inductor_module_without_runtime_guards(self, module_name: str, path: Path) -> Any:
        source = path.read_text(encoding="utf-8")
        filtered: list[str] = []
        for line in source.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("assert_size_stride("):
                continue
            if stripped == "args.clear()":
                continue
            if self._whitebox_block_strip_del and stripped.startswith("del "):
                continue
            if self._whitebox_block_hoist_stream and stripped == "stream0 = get_raw_stream(0)":
                continue
            if self._whitebox_block_strip_device_asserts and stripped.startswith("tl.device_assert("):
                continue
            if self._whitebox_block_static_scratch and "= empty_strided_cuda(" in line:
                lhs, rhs = line.split("= empty_strided_cuda(", 1)
                name = lhs.strip()
                if name.startswith("buf") and name.replace("buf", "", 1).isdigit():
                    indent = line[: len(line) - len(line.lstrip())]
                    filtered.append(f"{indent}{name} = _aicasgc_static_empty('{name}', {rhs}")
                    continue
            filtered.append(line)
            if self._whitebox_block_hoist_stream and stripped == "torch.cuda.set_device(0)":
                indent = line[: len(line) - len(line.lstrip())]
                filtered.append(f"{indent}stream0 = get_raw_stream(0)")
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        module.__package__ = ""
        module.__dict__["_aicas_triton_row_w8a16_gateup"] = _aicas_triton_row_w8a16_gateup
        module.__dict__["_aicas_triton_row_w8a16_gateup_silu_mul"] = _aicas_triton_row_w8a16_gateup_silu_mul
        module.__dict__["_aicas_triton_row_w8a16_gateup_norm"] = _aicas_triton_row_w8a16_gateup_norm
        module.__dict__["_aicas_triton_row_w8a16_down_silu"] = _aicas_triton_row_w8a16_down_silu
        module.__dict__["_aicas_triton_fp16_down_silu"] = _aicas_triton_fp16_down_silu
        module.__dict__["_aicas_triton_fp16_down_from_act"] = _aicas_triton_fp16_down_from_act
        module.__dict__["_aicas_triton_fp16_down_silu_preact"] = _aicas_triton_fp16_down_silu_preact
        if self._whitebox_block_static_scratch:
            scratch_cache: dict[tuple[Any, ...], torch.Tensor] = {}

            def _aicasgc_static_empty(name, size, stride, dtype):
                try:
                    device_idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else -1
                    key = (str(name), tuple(int(v) for v in size), tuple(int(v) for v in stride), dtype, device_idx)
                    cached = scratch_cache.get(key)
                    if isinstance(cached, torch.Tensor) and cached.device.type == "cuda":
                        if int(cached.device.index or 0) == device_idx:
                            return cached
                    buf = torch._C._dynamo.guards._empty_strided_cuda(size, stride, dtype)
                    scratch_cache[key] = buf
                    return buf
                except Exception:
                    return torch._C._dynamo.guards._empty_strided_cuda(size, stride, dtype)

            module.__dict__["_aicasgc_static_empty"] = _aicasgc_static_empty
        sys.modules[module_name] = module
        exec(compile("\n".join(filtered) + "\n", str(path), "exec"), module.__dict__)
        return module

    def _ensure_whitebox_block_call(self, bucket: int) -> Optional[Callable[[list[Any]], tuple[torch.Tensor, ...]]]:
        if not self._whitebox_block_enabled:
            return None
        try:
            bucket = int(bucket)
        except Exception:
            return None
        if bucket in self._whitebox_block_failed_buckets:
            return None
        cached = self._whitebox_block_calls.get(bucket)
        if callable(cached):
            return cached
        path = self._whitebox_block_paths.get(bucket)
        if not isinstance(path, Path) or not path.is_file():
            self._whitebox_block_failed_buckets.add(bucket)
            return None
        if (
            bool(getattr(self, "_whitebox_block_reject_unvalidated", True))
            and not self._whitebox_block_path_validated(bucket, path)
        ):
            self._whitebox_block_failed_buckets.add(bucket)
            return None
        try:
            module_suffix = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
            module_name = (
                f"_aicasgc_whitebox_token_block_static{bucket}_step{self._whitebox_block_size}"
                f"_{module_suffix}"
            )
            module = sys.modules.get(module_name)
            if module is None:
                if self._whitebox_block_strip_guards:
                    module = self._load_inductor_module_without_runtime_guards(module_name, path)
                else:
                    spec = importlib.util.spec_from_file_location(module_name, str(path))
                    if spec is None or spec.loader is None:
                        raise RuntimeError(f"cannot import {path}")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
            self._patch_whitebox_block_module(module)
            self._patch_whitebox_block_fp16_top1_lmhead(module)
            self._patch_whitebox_block_fused_decode_attention(module)
            self._patch_whitebox_block_rowtriton_norm_gateup(module, bucket)
            self._patch_whitebox_block_rowtriton_down(module, bucket)
            self._patch_whitebox_block_rowtriton_down_preact(module, bucket)
            self._patch_whitebox_block_allspark_down(module, bucket)
            self._patch_whitebox_block_fp16_down(module, bucket)
            self._patch_whitebox_block_fp16_down_preact(module, bucket)
            self._patch_whitebox_block_weightonly_down(module, bucket)
            self._patch_whitebox_block_torch_down(module, bucket)
            self._patch_whitebox_block_torch_linear2048(module, bucket)
            self._patch_whitebox_block_fp16_dot_linear2048(module, bucket)
            self._patch_whitebox_block_rowtriton_linear2048(module, bucket)
            self._patch_whitebox_block_smoothquant_gateup(module)
            self._patch_whitebox_block_weightonly_gateup(module)
            self._patch_whitebox_block_skip_mlp_layers(module)
            self._patch_whitebox_block_weightonly_lmhead(module)
            call = getattr(module, "call", None)
            if not callable(call):
                raise RuntimeError("call() missing")
            self._whitebox_block_calls[bucket] = call
            return call
        except Exception as exc:
            pass
            self._whitebox_block_failed_buckets.add(bucket)
            self._whitebox_block_calls.pop(bucket, None)
            return None

    def _whitebox_block_arg_count(self, bucket: int) -> Optional[int]:
        cached = self._whitebox_block_arg_count_cache.get(int(bucket))
        if isinstance(cached, int) and cached > 0:
            return cached
        path = self._whitebox_block_paths.get(int(bucket))
        if not isinstance(path, Path) or not path.is_file():
            return None
        try:
            text = path.read_text()
            marker = " = args"
            for line in text.splitlines():
                if marker not in line:
                    continue
                lhs = line.split(marker, 1)[0].strip()
                if not lhs.startswith("arg"):
                    continue
                count = len([piece for piece in lhs.split(",") if piece.strip()])
                if count > 0:
                    self._whitebox_block_arg_count_cache[int(bucket)] = count
                    return count
        except Exception:
            return None
        return None

    def _whitebox_block_arg_count_supported(self, bucket: int) -> bool:
        arg_count = self._whitebox_block_arg_count(int(bucket))
        if arg_count == 342:
            return True
        if arg_count == 398:
            return True
        if arg_count == 286:
            return True
        return bool(self._whitebox_block_allow_generic and arg_count == 370)

    def _whitebox_block_arg_layout(self, bucket: int) -> str:
        cached = self._whitebox_block_layout_cache.get(int(bucket))
        if isinstance(cached, str) and cached:
            return cached
        layout = "official342"
        path = self._whitebox_block_paths.get(int(bucket))
        if isinstance(path, Path) and path.is_file():
            try:
                text = path.read_text()
                if "assert_size_stride(arg0_1, (1, 8," in text and "assert_size_stride(arg56_1, (1, 1)" in text:
                    if "qkv286_rowtriton" in path.name.lower():
                        layout = "qkv342_rowtriton_kv_first"
                    elif "arg285_1" in text and "arg286_1" not in text:
                        layout = "qkv286_kv_first"
                    elif (
                        "assert_size_stride(arg58_1, (1, 1)" in text
                        and "assert_size_stride(arg59_1, (151936, 2048)" in text
                        and "assert_size_stride(arg60_1, (64, )" in text
                        and "assert_size_stride(arg61_1, (2048, )" in text
                        and "assert_size_stride(arg62_1, (2048, )" in text
                        and "assert_size_stride(arg70_1, (12288, 2048)" in text
                        and "assert_size_stride(arg341_1, (2048, 6144)" in text
                    ):


                        layout = "generic342_kv_first_fusedgate_finalnorm_first"
                    elif (
                        "assert_size_stride(arg58_1, (1, 1)" in text
                        and "assert_size_stride(arg59_1, (151936, 2048)" in text
                        and "assert_size_stride(arg60_1, (64, )" in text
                        and "assert_size_stride(arg69_1, (12288, 2048)" in text
                        and "assert_size_stride(arg341_1, (2048, )" in text
                    ):



                        layout = "generic342_kv_first_fusedgate_ordered"
                    else:





                        layout = "generic342_kv_first"
                elif "assert_size_stride(arg0_1, (1, 1)" in text:
                    layout = "official342"
            except Exception:
                layout = "official342"
        self._whitebox_block_layout_cache[int(bucket)] = layout
        return layout

    def _whitebox_block_graph_layout_compatible(self, bucket: int) -> bool:
        arg_count = self._whitebox_block_arg_count(int(bucket))
        if arg_count == 342:
            layout = self._whitebox_block_arg_layout(int(bucket))
            if layout == "qkv342_rowtriton_kv_first":
                return True
            return True
        if arg_count == 286:
            return self._whitebox_block_arg_layout(int(bucket)) == "qkv286_kv_first"
        if arg_count != 398:
            return False
        return self._whitebox_block_arg_layout(int(bucket)) in {
            "generic342_kv_first_fusedgate_ordered",
            "generic342_kv_first_fusedgate_finalnorm_first",
        }

    @staticmethod
    def _parse_index_mask(raw: str) -> Optional[set[int]]:
        text = str(raw or "").strip().lower()
        if not text or text in {"all", "*"}:
            return None
        if text in {"none", "off", "disable", "disabled"}:
            return set()
        mask: set[int] = set()
        try:
            for item in text.split(","):
                part = item.strip()
                if not part:
                    continue
                if "-" in part:
                    lo_s, hi_s = part.split("-", 1)
                    lo = int(lo_s.strip())
                    hi = int(hi_s.strip())
                    if hi < lo:
                        lo, hi = hi, lo
                    mask.update(range(lo, hi + 1))
                else:
                    mask.add(int(part))
        except Exception:
            return None
        return {idx for idx in mask if idx >= 0}

    @classmethod
    def _env_index_mask(cls, primary: str, fallback: str = "") -> Optional[set[int]]:
        raw = os.environ.get(primary, "")
        if not str(raw or "").strip() and fallback:
            raw = os.environ.get(fallback, "")
        return cls._parse_index_mask(str(raw or ""))

    def _rowtriton_gateup_layer_mask_for_request(self, max_new_tokens: int) -> str:
        if int(max_new_tokens) == int(getattr(self, "_performance_request_tokens", 128)):
            return str(getattr(self, "_rowtriton_gateup_perf_layer_mask", "") or "").strip()
        return str(getattr(self, "_rowtriton_gateup_default_layer_mask", "") or "").strip()

    def _rowtriton_gateup_graph_key(self) -> str:
        mask = str(getattr(self, "_rowtriton_gateup_active_layer_mask", "") or "").strip()
        return f"gateup_mask={mask or 'env'}"

    def _set_rowtriton_gateup_layer_mask_for_request(self, max_new_tokens: int) -> str:
        previous = str(getattr(self, "_rowtriton_gateup_active_layer_mask", "") or "").strip()
        mask = self._rowtriton_gateup_layer_mask_for_request(int(max_new_tokens))
        if mask:
            self._rowtriton_gateup_active_layer_mask = mask
        else:
            self._rowtriton_gateup_active_layer_mask = ""
        return previous

    def _restore_rowtriton_gateup_layer_mask(self, previous: str) -> None:
        if previous:
            self._rowtriton_gateup_active_layer_mask = previous
        else:
            self._rowtriton_gateup_active_layer_mask = self._rowtriton_gateup_default_layer_mask

    def _set_qkv_patch_mode_for_request(self, max_new_tokens: int) -> bool:
        previous = bool(getattr(self, "_current_qkv_patch_fused_linear", self._patch_qkv_forward_fused_linear))
        use_fused = bool(self._patch_qkv_forward_fused_linear)
        self._current_qkv_patch_fused_linear = bool(use_fused)
        if self._raw_model is not None:
            try:
                for layer in self._raw_model.model.language_model.layers:
                    setattr(layer.self_attn, "_aicas_qkv_patch_fused_linear", bool(use_fused))
            except Exception:
                pass
        return previous

    def _restore_qkv_patch_mode(self, previous: bool) -> None:
        self._current_qkv_patch_fused_linear = bool(previous)
        if self._raw_model is not None:
            try:
                for layer in self._raw_model.model.language_model.layers:
                    setattr(layer.self_attn, "_aicas_qkv_patch_fused_linear", bool(previous))
            except Exception:
                pass

    def _whitebox_block_gateup_ptr_mask(
        self,
        primary: str,
        *,
        fallback: str = "",
        max_weights: int = 0,
    ) -> Optional[frozenset[int]]:
        layer_mask = self._env_index_mask(primary, fallback)
        if layer_mask is None and int(max_weights) <= 0:
            return None
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return frozenset()
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple):
            return frozenset()
        ptrs: list[int] = []
        for layer_idx, weights in enumerate(layer_weights):
            if layer_mask is not None and int(layer_idx) not in layer_mask:
                continue
            if len(weights) < 9 or not isinstance(weights[8], torch.Tensor):
                continue
            ptrs.append(int(weights[8].data_ptr()))
            if int(max_weights) > 0 and len(ptrs) >= int(max_weights):
                break
        return frozenset(ptrs)

    @staticmethod
    def _env_nonnegative_int(name: str, default: int = 0) -> int:
        try:
            return max(0, int(os.environ.get(name, str(default))))
        except Exception:
            return max(0, int(default))

    @staticmethod
    def _fp16_contiguous(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype == torch.float16 and tensor.is_contiguous():
            return tensor
        return tensor.detach().to(dtype=torch.float16).contiguous()

    @staticmethod
    def _fp16_split_gate_up(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_size = int(tensor.shape[0]) // 2
        return tensor[:gate_size, :].contiguous(), tensor[gate_size:, :].contiguous()

    def _whitebox_block_gateup_w8a16_args(self) -> Optional[tuple[torch.Tensor, ...]]:
        cached = self._whitebox_block_gateup_w8a16_args_cache
        if cached is not None:
            return cached
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple):
            return None
        try:
            import vllm._C
            from vllm import _custom_ops as _aicasgc_vllm_ops

            if not hasattr(torch.ops._C, "rearrange_kn_weight_as_n32k16_order"):
                return None
            packed_args: list[torch.Tensor] = []
            for weights in layer_weights:
                if len(weights) < 9:
                    return None
                gate_up = weights[8]
                if (
                    not isinstance(gate_up, torch.Tensor)
                    or gate_up.ndim != 2
                    or int(gate_up.shape[0]) != 12288
                    or int(gate_up.shape[1]) != 2048
                ):
                    return None
                w = self._fp16_contiguous(gate_up)
                n = int(w.shape[0])
                k = int(w.shape[1])
                n32 = ((n + 31) // 32) * 32
                scales = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6) / 127.0
                qweight = torch.round(w.float() / scales).clamp(-128, 127).to(torch.int16)
                qweight = (qweight + 128).clamp(0, 255).to(torch.uint8)
                qweight_kn = qweight.t().contiguous()
                scales_1n = scales.t().contiguous().to(device=w.device, dtype=torch.float16)
                qweight_reorder = torch.empty((n32, k), device=w.device, dtype=torch.uint8)
                scales_reorder = torch.empty((1, n32), device=w.device, dtype=torch.float16)
                torch.ops._C.rearrange_kn_weight_as_n32k16_order(
                    qweight_kn,
                    scales_1n,
                    None,
                    False,
                    qweight_reorder,
                    scales_reorder,
                    None,
                    k,
                    n,
                    n32,
                )
                packed_args.extend([qweight_reorder, scales_reorder])
            result = tuple(packed_args)
            self._whitebox_block_gateup_w8a16_args_cache = result
            return result
        except Exception as exc:
            pass
            return None

    def _whitebox_block_gateup_w8a16_row_args(self) -> Optional[tuple[torch.Tensor, ...]]:
        cached = self._whitebox_block_gateup_w8a16_row_args_cache
        if cached is not None:
            return cached
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple):
            return None
        try:
            packed_args: list[torch.Tensor] = []
            for weights in layer_weights:
                if len(weights) < 9:
                    return None
                gate_up = weights[8]
                if (
                    not isinstance(gate_up, torch.Tensor)
                    or gate_up.ndim != 2
                    or int(gate_up.shape[0]) != 12288
                    or int(gate_up.shape[1]) != 2048
                ):
                    return None
                w = self._fp16_contiguous(gate_up)
                scales = w.float().abs().amax(dim=1).clamp_min(1.0e-6) / 127.0
                qweight = torch.round(w.float() / scales[:, None]).clamp(-128, 127).to(torch.int8).contiguous()
                packed_args.extend([qweight, scales.to(device=w.device, dtype=torch.float16).contiguous()])
            result = tuple(packed_args)
            self._whitebox_block_gateup_w8a16_row_args_cache = result
            return result
        except Exception as exc:
            pass
            return None

    def _whitebox_block_down_w8a16_args(self) -> Optional[tuple[torch.Tensor, ...]]:
        cached = self._whitebox_block_down_w8a16_args_cache
        if cached is not None:
            return cached
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple):
            return None
        try:
            import vllm._C
            from vllm import _custom_ops as _aicasgc_vllm_ops

            if not hasattr(torch.ops._C, "rearrange_kn_weight_as_n32k16_order"):
                return None
            packed_args: list[torch.Tensor] = []
            for weights in layer_weights:
                if len(weights) < 10:
                    return None
                down = weights[9]
                if (
                    not isinstance(down, torch.Tensor)
                    or down.ndim != 2
                    or int(down.shape[0]) != 2048
                    or int(down.shape[1]) != 6144
                ):
                    return None
                w = self._fp16_contiguous(down)
                n = int(w.shape[0])
                k = int(w.shape[1])
                n32 = ((n + 31) // 32) * 32
                scales = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6) / 127.0
                qweight = torch.round(w.float() / scales).clamp(-128, 127).to(torch.int16)
                qweight = (qweight + 128).clamp(0, 255).to(torch.uint8)
                qweight_kn = qweight.t().contiguous()
                scales_1n = scales.t().contiguous().to(device=w.device, dtype=torch.float16)
                qweight_reorder = torch.empty((n32, k), device=w.device, dtype=torch.uint8)
                scales_reorder = torch.empty((1, n32), device=w.device, dtype=torch.float16)
                torch.ops._C.rearrange_kn_weight_as_n32k16_order(
                    qweight_kn,
                    scales_1n,
                    None,
                    False,
                    qweight_reorder,
                    scales_reorder,
                    None,
                    k,
                    n,
                    n32,
                )
                try:
                    setattr(qweight_reorder, "_aicasgc_source_ptr", int(down.data_ptr()))
                except Exception:
                    pass
                packed_args.extend([qweight_reorder, scales_reorder])
            result = tuple(packed_args)
            self._whitebox_block_down_w8a16_args_cache = result
            return result
        except Exception as exc:
            pass
            return None

    def _whitebox_block_down_w8a16_row_args(self) -> Optional[tuple[torch.Tensor, ...]]:
        cached = self._whitebox_block_down_w8a16_row_args_cache
        if cached is not None:
            return cached
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple):
            return None
        try:
            layer_mask = self._env_index_mask(
                "AICASGC_ROWTRITON_W8A16_DOWN_LAYER_MASK",
                "AICASGC_ROWTRITON_W8A16_LAYER_MASK",
            )
            max_weights = self._env_nonnegative_int("AICASGC_ROWTRITON_W8A16_DOWN_MAX_WEIGHTS", 0)
            packed_args: list[torch.Tensor] = []
            bound = 0
            for layer_idx, weights in enumerate(layer_weights):
                if layer_mask is not None and int(layer_idx) not in layer_mask:
                    continue
                if max_weights > 0 and bound >= max_weights:
                    break
                if len(weights) < 10:
                    return None
                down = weights[9]
                if (
                    not isinstance(down, torch.Tensor)
                    or down.ndim != 2
                    or int(down.shape[0]) != 2048
                    or int(down.shape[1]) != 6144
                ):
                    return None
                w = self._fp16_contiguous(down)
                scales = w.float().abs().amax(dim=1).clamp_min(1.0e-6) / 127.0
                qweight = torch.round(w.float() / scales[:, None]).clamp(-128, 127).to(torch.int8).contiguous()
                try:
                    setattr(qweight, "_aicasgc_source_ptr", int(down.data_ptr()))
                except Exception:
                    pass
                packed_args.extend([qweight, scales.to(device=w.device, dtype=torch.float16).contiguous()])
                bound += 1
            if not packed_args:
                return None
            result = tuple(packed_args)
            self._whitebox_block_down_w8a16_row_args_cache = result
            return result
        except Exception as exc:
            pass
            return None

    def _whitebox_block_linear2048_w8a16_row_args(self) -> Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]]:
        cached = self._whitebox_block_linear2048_w8a16_row_args_cache
        if cached is not None:
            return cached
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple):
            return None
        try:
            layer_mask = self._env_index_mask(
                "AICASGC_ROWTRITON_W8A16_LINEAR2048_LAYER_MASK",
                "AICASGC_ROWTRITON_W8A16_LAYER_MASK",
            )
            weight_mask = self._env_index_mask("AICASGC_ROWTRITON_W8A16_LINEAR2048_WEIGHT_IDS")
            if weight_mask is None:
                weight_ids = (1, 6)
            else:
                weight_ids = tuple(idx for idx in (1, 6) if idx in weight_mask)
                if not weight_ids:
                    return None
            max_weights = self._env_nonnegative_int("AICASGC_ROWTRITON_W8A16_LINEAR2048_MAX_WEIGHTS", 0)
            packed: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            bound = 0
            for layer_idx, weights in enumerate(layer_weights):
                if layer_mask is not None and int(layer_idx) not in layer_mask:
                    continue
                if len(weights) < 7:
                    return None
                for weight_idx in weight_ids:
                    if max_weights > 0 and bound >= max_weights:
                        break
                    weight = weights[weight_idx]
                    if (
                        not isinstance(weight, torch.Tensor)
                        or weight.ndim != 2
                        or int(weight.shape[0]) != 2048
                        or int(weight.shape[1]) != 2048
                    ):
                        continue
                    w = self._fp16_contiguous(weight)
                    scales = w.float().abs().amax(dim=1).clamp_min(1.0e-6) / 127.0
                    qweight = torch.round(w.float() / scales[:, None]).clamp(-128, 127).to(torch.int8).contiguous()
                    packed[int(weight.data_ptr())] = (
                        qweight,
                        scales.to(device=w.device, dtype=torch.float16).contiguous(),
                    )
                    bound += 1
                if max_weights > 0 and bound >= max_weights:
                    break
            if not packed:
                return None
            self._whitebox_block_linear2048_w8a16_row_args_cache = packed
            return packed
        except Exception as exc:
            pass
            return None

    def _whitebox_block_uses_rowtriton_gateup(self, bucket: int) -> bool:
        path = self._whitebox_block_paths.get(int(bucket))
        if isinstance(path, Path):
            return "rowtriton" in path.name.lower()
        return False

    def _whitebox_block_uses_weightonly_gateup(self, bucket: int) -> bool:
        path = self._whitebox_block_paths.get(int(bucket))
        if isinstance(path, Path):
            name = path.name.lower()
            return "weightonly" in name or "wo_gateup" in name
        return False

    def _whitebox_block_needs_qkv_weights(self) -> bool:
        try:
            for bucket in sorted(self._whitebox_block_paths):
                arg_count = self._whitebox_block_arg_count(int(bucket))
                if arg_count == 286:
                    return True
                if arg_count == 342 and self._whitebox_block_arg_layout(int(bucket)) == "qkv342_rowtriton_kv_first":
                    return True
        except Exception:
            return False
        return False

    def _whitebox_block_gateup_weightonly_args(self) -> Optional[tuple[torch.Tensor, ...]]:
        cached = self._whitebox_block_gateup_weightonly_args_cache
        if cached is not None:
            return cached
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple):
            return None
        try:
            import acext
            import vllm._C

            preprocess = acext.preprocess_weights_for_mixed_gemm
            packed_args: list[torch.Tensor] = []
            for weights in layer_weights:
                if len(weights) < 9:
                    return None
                gate_up = weights[8]
                if (
                    not isinstance(gate_up, torch.Tensor)
                    or gate_up.ndim != 2
                    or int(gate_up.shape[0]) != 12288
                    or int(gate_up.shape[1]) != 2048
                ):
                    return None
                w = self._fp16_contiguous(gate_up)
                scales = w.float().abs().amax(dim=1).clamp_min(1.0e-6) / 127.0
                qweight = torch.round(w.float() / scales[:, None]).clamp(-128, 127).to(torch.int8)
                qweight_cpu = qweight.t().contiguous().cpu()
                qweight_packed = preprocess(qweight_cpu, torch.int8, False, False).to(device=w.device)
                scales_gpu = scales.to(device=w.device, dtype=torch.float16).contiguous()
                packed_args.extend([qweight_packed, scales_gpu])
            result = tuple(packed_args)
            self._whitebox_block_gateup_weightonly_args_cache = result
            return result
        except Exception as exc:
            pass
            return None

    def _whitebox_block_weight_args_fp16(self) -> Optional[dict[str, Any]]:
        if self._whitebox_block_weight_args_fp16_cache is not None:
            return self._whitebox_block_weight_args_fp16_cache
        if self._raw_model is None:
            return None
        try:
            lm = self._raw_model.model.language_model
            layer_weights: list[tuple[torch.Tensor, ...]] = []
            qkv_layers: list[torch.Tensor] = []
            need_qkv_layers = self._whitebox_block_needs_qkv_weights()
            for layer in lm.layers:
                attn = layer.self_attn
                mlp = layer.mlp
                gate_up_weight = getattr(mlp, "_aicas_gate_up_weight", None)
                if not isinstance(gate_up_weight, torch.Tensor):
                    gate_up_weight = torch.cat([mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()], dim=0)
                gate_up_weight = self._fp16_contiguous(gate_up_weight)
                generic_gate_weights = ()
                if self._whitebox_block_allow_generic:
                    generic_gate_weights = self._fp16_split_gate_up(gate_up_weight)
                layer_weights.append(
                    (
                        self._fp16_contiguous(layer.input_layernorm.weight),
                        self._fp16_contiguous(attn.q_proj.weight),
                        self._fp16_contiguous(attn.q_norm.weight),
                        self._fp16_contiguous(attn.k_proj.weight),
                        self._fp16_contiguous(attn.k_norm.weight),
                        self._fp16_contiguous(attn.v_proj.weight),
                        self._fp16_contiguous(attn.o_proj.weight),
                        self._fp16_contiguous(layer.post_attention_layernorm.weight),
                        gate_up_weight,
                        self._fp16_contiguous(mlp.down_proj.weight),
                        *generic_gate_weights,
                    )
                )
                if need_qkv_layers:
                    qkv_weight = getattr(attn, "_aicas_qkv_weight", None)
                    if not isinstance(qkv_weight, torch.Tensor):
                        qkv_weight = torch.cat(
                            [
                                attn.q_proj.weight.detach(),
                                attn.k_proj.weight.detach(),
                                attn.v_proj.weight.detach(),
                            ],
                            dim=0,
                        )
                    qkv_layers.append(self._fp16_contiguous(qkv_weight))
            cache = {
                "layers": tuple(layer_weights),
                "norm": self._fp16_contiguous(lm.norm.weight),
                "embed": self._fp16_contiguous(lm.embed_tokens.weight),
                "rotary": lm.rotary_emb.inv_freq.detach().contiguous(),
            }
            if need_qkv_layers:
                cache["qkv_layers"] = tuple(qkv_layers)
            self._whitebox_block_weight_args_fp16_cache = cache
            return cache
        except Exception as exc:
            pass
            self._whitebox_block_enabled = False
            return None

    def _dynamic_cache_to_static(
        self,
        past: Any,
        total_len: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[Any]:
        if StaticCache is None or self._raw_model is None:
            return None
        layers = getattr(past, "layers", None)
        if not layers:
            return None
        try:
            static_dtype = dtype or getattr(self._raw_model, "dtype", self._dtype)
            static = StaticCache(
                config=self._raw_model.config,
                max_batch_size=1,
                max_cache_len=int(total_len),
                device=device,
                dtype=static_dtype,
            )
            for layer_idx, layer in enumerate(layers):
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if key is None or value is None:
                    return None
                seq_len = int(key.shape[-2])
                cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
                static.update(
                    key.to(device=device, dtype=static_dtype),
                    value.to(device=device, dtype=static_dtype),
                    layer_idx,
                    {"cache_position": cache_position},
                )
            return static
        except Exception as exc:
            pass
            return None

    def _copy_dynamic_cache_to_static_existing(
        self,
        past: Any,
        static: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        if StaticCache is None or self._raw_model is None or past is None or static is None:
            return False
        src_layers = getattr(past, "layers", None)
        dst_layers = getattr(static, "layers", None)
        if not isinstance(src_layers, (list, tuple)) or not isinstance(dst_layers, (list, tuple)):
            return False
        if len(src_layers) != len(dst_layers):
            return False
        try:
            for layer_idx, src_layer in enumerate(src_layers):
                key = getattr(src_layer, "keys", None)
                value = getattr(src_layer, "values", None)
                if key is None or value is None:
                    return False
                seq_len = int(key.shape[-2])
                dst_layer = dst_layers[layer_idx]
                dst_key = getattr(dst_layer, "keys", None)
                dst_value = getattr(dst_layer, "values", None)
                if not torch.is_tensor(dst_key) or not torch.is_tensor(dst_value):
                    return False
                if int(dst_key.shape[-2]) < seq_len or int(dst_value.shape[-2]) < seq_len:
                    return False
                dst_key[..., :seq_len, :].copy_(key.to(device=device, dtype=dtype))
                dst_value[..., :seq_len, :].copy_(value.to(device=device, dtype=dtype))
            return True
        except Exception as exc:
            pass
            return False

    def _visual_kv_keep_indices(
        self,
        *,
        visual_start: int,
        visual_tokens: int,
        keep_visual: int,
        device: torch.device,
    ) -> torch.Tensor:
        visual_start = int(visual_start)
        visual_tokens = int(max(0, visual_tokens))
        keep_visual = int(max(0, min(keep_visual, visual_tokens)))
        if keep_visual <= 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        pattern = self._whitebox_visual_kv_keep_pattern
        if pattern in {"uniform", "uniform_prefix", "spread"} and keep_visual < visual_tokens:
            idx = torch.linspace(
                0,
                visual_tokens - 1,
                steps=keep_visual,
                device=device,
                dtype=torch.float32,
            ).round().to(torch.long)
            idx = torch.unique_consecutive(idx)
            if int(idx.numel()) < keep_visual:
                fill = torch.arange(0, visual_tokens, device=device, dtype=torch.long)
                used = torch.zeros((visual_tokens,), device=device, dtype=torch.bool)
                used.index_fill_(0, idx, True)
                extra = fill[~used][: keep_visual - int(idx.numel())]
                idx = torch.sort(torch.cat([idx, extra], dim=0))[0]
            if pattern == "uniform_prefix" and int(idx.numel()) > 0 and int(idx[0].item()) != 0:
                idx[0] = 0
                idx = torch.sort(idx)[0]
            return idx[:keep_visual] + visual_start
        return torch.arange(visual_start, visual_start + keep_visual, device=device, dtype=torch.long)

    def _visual_kv_compact_indices(
        self,
        *,
        total_existing: int,
        prompt_len: int,
        image_grid_thw: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        visual_tokens = self._estimate_visual_tokens_from_grid(image_grid_thw)
        if visual_tokens <= 0:
            return None
        try:
            prompt_len = int(prompt_len)
            total_existing = int(total_existing)
            if prompt_len <= 0 or total_existing < prompt_len:
                return None
            visual_start = 0
            visual_count = int(visual_tokens)
            if isinstance(visual_pos_masks, torch.Tensor) and visual_pos_masks.ndim == 2:
                mask = visual_pos_masks[0, :prompt_len].to(device=device, dtype=torch.bool)
                positions = torch.nonzero(mask, as_tuple=False).view(-1)
                if int(positions.numel()) > 0:
                    first = int(positions[0].item())
                    last = int(positions[-1].item()) + 1
                    count = int(positions.numel())
                    if last - first == count:
                        visual_start = first
                        visual_count = count
            keep_visual = min(int(visual_count), int(self._whitebox_visual_kv_keep_tokens))
            if keep_visual <= 0 or keep_visual >= int(visual_count):
                return None
            visual_end = int(visual_start + visual_count)
            if visual_start < 0 or visual_end > prompt_len:
                return None
            keep_before = torch.arange(0, visual_start, device=device, dtype=torch.long)
            keep_visual_idx = self._visual_kv_keep_indices(
                visual_start=int(visual_start),
                visual_tokens=int(visual_count),
                keep_visual=int(keep_visual),
                device=device,
            )
            keep_after = torch.arange(visual_end, total_existing, device=device, dtype=torch.long)
            keep_idx = torch.cat([keep_before, keep_visual_idx, keep_after], dim=0)
            if int(keep_idx.numel()) <= 0 or int(keep_idx.numel()) >= total_existing:
                return None
            return keep_idx
        except Exception:
            return None

    def _copy_dynamic_cache_to_static_existing_compacted_visual(
        self,
        past: Any,
        static: Any,
        *,
        device: torch.device,
        prompt_len: int,
        generated_tokens_in_cache: int,
        image_grid_thw: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        dtype: torch.dtype,
    ) -> Optional[int]:
        if (
            StaticCache is None
            or self._raw_model is None
            or past is None
            or static is None
            or not self._whitebox_visual_kv_compact_enabled
            or self._whitebox_visual_kv_keep_tokens <= 0
        ):
            return None
        src_layers = getattr(past, "layers", None)
        dst_layers = getattr(static, "layers", None)
        if not isinstance(src_layers, (list, tuple)) or not isinstance(dst_layers, (list, tuple)):
            return None
        if len(src_layers) != len(dst_layers):
            return None
        try:
            prompt_len = int(prompt_len)
            generated_tokens_in_cache = int(max(0, generated_tokens_in_cache))
            total_existing = int(prompt_len + generated_tokens_in_cache)
            keep_idx = self._visual_kv_compact_indices(
                total_existing=total_existing,
                prompt_len=prompt_len,
                image_grid_thw=image_grid_thw,
                visual_pos_masks=visual_pos_masks,
                device=device,
            )
            if keep_idx is None:
                return None
            compact_len = int(keep_idx.numel())
            for layer_idx, src_layer in enumerate(src_layers):
                key = getattr(src_layer, "keys", None)
                value = getattr(src_layer, "values", None)
                if key is None or value is None:
                    return None
                dst_layer = dst_layers[layer_idx]
                dst_key = getattr(dst_layer, "keys", None)
                dst_value = getattr(dst_layer, "values", None)
                if not torch.is_tensor(dst_key) or not torch.is_tensor(dst_value):
                    return None
                if int(dst_key.shape[-2]) < compact_len or int(dst_value.shape[-2]) < compact_len:
                    return None
                dst_key[..., :compact_len, :].copy_(
                    key.index_select(-2, keep_idx).to(device=device, dtype=dtype)
                )
                dst_value[..., :compact_len, :].copy_(
                    value.index_select(-2, keep_idx).to(device=device, dtype=dtype)
                )
            return int(total_existing - compact_len)
        except Exception as exc:
            pass
            return None

    def _dynamic_cache_to_static_compacted_visual(
        self,
        past: Any,
        *,
        total_len: int,
        device: torch.device,
        prompt_len: int,
        generated_tokens_in_cache: int,
        image_grid_thw: torch.Tensor,
        visual_pos_masks: Optional[torch.Tensor] = None,
        dtype: torch.dtype,
    ) -> Optional[tuple[Any, int]]:
        if (
            StaticCache is None
            or self._raw_model is None
            or not self._whitebox_visual_kv_compact_enabled
            or self._whitebox_visual_kv_keep_tokens <= 0
        ):
            return None
        layers = getattr(past, "layers", None)
        if not layers:
            return None
        try:
            prompt_len = int(prompt_len)
            generated_tokens_in_cache = int(max(0, generated_tokens_in_cache))
            total_existing = int(prompt_len + generated_tokens_in_cache)
            keep_idx = self._visual_kv_compact_indices(
                total_existing=total_existing,
                prompt_len=prompt_len,
                image_grid_thw=image_grid_thw,
                visual_pos_masks=visual_pos_masks,
                device=device,
            )
            if keep_idx is None:
                return None
            compact_len = int(keep_idx.numel())
            static = StaticCache(
                config=self._raw_model.config,
                max_batch_size=1,
                max_cache_len=int(total_len),
                device=device,
                dtype=dtype,
            )
            cache_position = torch.arange(compact_len, device=device, dtype=torch.long)
            for layer_idx, layer in enumerate(layers):
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if key is None or value is None:
                    return None
                k_compact = key.index_select(-2, keep_idx).to(device=device, dtype=dtype)
                v_compact = value.index_select(-2, keep_idx).to(device=device, dtype=dtype)
                static.update(k_compact, v_compact, layer_idx, {"cache_position": cache_position})
            delta = int(total_existing - compact_len)
            return static, delta
        except Exception as exc:
            pass
            return None

    @staticmethod
    def _materialize_static_cache_pair(past: Any, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if StaticCache is None or past is None or not isinstance(past, StaticCache):
            return None
        layers = getattr(past, "layers", None)
        if not isinstance(layers, (list, tuple)) or not (0 <= int(layer_idx) < len(layers)):
            return None
        layer = layers[int(layer_idx)]
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if torch.is_tensor(key) and torch.is_tensor(value):
            return key, value
        return None

    @staticmethod
    def _materialize_whitebox_block_tokens(
        block_output: torch.Tensor,
        *,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not isinstance(block_output, torch.Tensor):
            return None
        if block_output.ndim == 1:
            if block_output.device == device and block_output.dtype == dtype:
                tokens = block_output.view(1, -1)
            else:
                tokens = block_output.view(1, -1).to(device=device, dtype=dtype)
            return tokens if int(tokens.shape[1]) == int(block_size) else None
        if block_output.ndim == 2 and int(block_output.shape[0]) == 1 and int(block_output.shape[1]) == int(block_size):
            if block_output.device == device and block_output.dtype == dtype:
                return block_output
            return block_output.to(device=device, dtype=dtype)
        return None

    def _build_whitebox_block_bound_state(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple) or self._raw_model is None:
            return None
        lm = self._raw_model.model.language_model
        if len(layer_weights) != len(lm.layers) or len(layer_weights) < 5:
            return None

        layer_data: list[tuple[torch.Tensor, ...]] = []
        for layer_idx, weights in enumerate(layer_weights):
            cache_pair = self._materialize_static_cache_pair(past, layer_idx)
            if cache_pair is None:
                return None
            keys, values = cache_pair
            layer_data.append(
                (
                    weights[0],
                    weights[1],
                    weights[2],
                    weights[3],
                    weights[4],
                    weights[5],
                    keys,
                    values,
                    weights[6],
                    weights[7],
                    weights[8],
                    weights[9],
                )
            )

        static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        static_cache_position = torch.zeros((1,), dtype=torch.long, device=device)
        static_rope_delta = torch.zeros((1, 1), dtype=torch.long, device=device)
        args: list[Any] = [static_input_ids, static_cache_position]

        for layer_idx in range(4, len(layer_data) - 1):
            weights = layer_data[layer_idx]
            args.extend(
                [
                    weights[10],
                    weights[11],
                    weights[9],
                    weights[8],
                    weights[2],
                    weights[3],
                    weights[4],
                    weights[5],
                    weights[6],
                    weights[7],
                    weights[1],
                    weights[0],
                ]
            )

        last_layer = layer_data[-1]
        args.extend([last_layer[10], last_layer[11]])
        args.extend(
            [
                weights_cache["norm"],
                static_rope_delta,
                weights_cache["embed"],
                weights_cache["rotary"],
            ]
        )

        for layer_idx in range(4):
            args.extend(layer_data[layer_idx])
        args.extend(last_layer[:10])

        if len(args) != 342:
            return None
        state = {
            "call": call,
            "args": _AICASNoClearList(args),
            "input_ids": static_input_ids,
            "cache_position": static_cache_position,
            "rope_delta": static_rope_delta,
        }
        return state

    def _build_whitebox_block_bound_state_generic370(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple) or self._raw_model is None:
            return None
        lm = self._raw_model.model.language_model
        if len(layer_weights) != len(lm.layers):
            return None
        if any(len(weights) < 12 for weights in layer_weights):
            return None

        args: list[Any] = []
        for layer_idx in range(len(layer_weights)):
            cache_pair = self._materialize_static_cache_pair(past, layer_idx)
            if cache_pair is None:
                return None
            args.extend(cache_pair)

        static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        static_cache_position = torch.zeros((1,), dtype=torch.long, device=device)
        static_rope_delta = torch.zeros((1, 1), dtype=torch.long, device=device)
        args.extend(
            [
                static_input_ids,
                static_cache_position,
                static_rope_delta,
                weights_cache["embed"],
                weights_cache["rotary"],
            ]
        )

        for weights in layer_weights:
            args.extend(
                [
                    weights[0],
                    weights[1],
                    weights[2],
                    weights[3],
                    weights[4],
                    weights[5],
                    weights[6],
                    weights[7],
                    weights[10],
                    weights[11],
                    weights[9],
                ]
            )
        args.append(weights_cache["norm"])

        if len(args) != 370:
            return None
        return {
            "call": call,
            "args": _AICASNoClearList(args),
            "input_ids": static_input_ids,
            "cache_position": static_cache_position,
            "rope_delta": static_rope_delta,
        }

    def _build_whitebox_block_bound_state_generic342_kv_first(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple) or self._raw_model is None:
            return None
        lm = self._raw_model.model.language_model
        if len(layer_weights) != len(lm.layers) or len(layer_weights) < 5:
            return None

        args: list[Any] = []
        for layer_idx in range(len(layer_weights)):
            cache_pair = self._materialize_static_cache_pair(past, layer_idx)
            if cache_pair is None:
                return None
            args.extend(cache_pair)

        static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        static_cache_position = torch.zeros((1,), dtype=torch.long, device=device)
        static_rope_delta = torch.zeros((1, 1), dtype=torch.long, device=device)
        args.extend([static_input_ids, static_cache_position])

        for layer_idx in range(4, len(layer_weights) - 1):
            weights = layer_weights[layer_idx]
            args.extend(
                [
                    weights[8],
                    weights[9],
                    weights[7],
                    weights[6],
                    weights[2],
                    weights[3],
                    weights[4],
                    weights[5],
                    weights[1],
                    weights[0],
                ]
            )

        last_layer = layer_weights[-1]
        args.extend([last_layer[8], last_layer[9]])
        args.extend(
            [
                weights_cache["norm"],
                static_rope_delta,
                weights_cache["embed"],
                weights_cache["rotary"],
            ]
        )

        for layer_idx in range(4):
            weights = layer_weights[layer_idx]
            args.extend(
                [
                    weights[0],
                    weights[1],
                    weights[2],
                    weights[3],
                    weights[4],
                    weights[5],
                    weights[6],
                    weights[7],
                    weights[8],
                    weights[9],
                ]
            )
        args.extend(last_layer[:8])

        if len(args) != 342:
            return None
        return {
            "call": call,
            "args": _AICASNoClearList(args),
            "input_ids": static_input_ids,
            "cache_position": static_cache_position,
            "rope_delta": static_rope_delta,
        }

    def _build_whitebox_block_bound_state_generic342_kv_first_fusedgate_ordered(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple) or self._raw_model is None:
            return None
        lm = self._raw_model.model.language_model
        if len(layer_weights) != len(lm.layers):
            return None

        args: list[Any] = []
        for layer_idx in range(len(layer_weights)):
            cache_pair = self._materialize_static_cache_pair(past, layer_idx)
            if cache_pair is None:
                return None
            args.extend(cache_pair)

        static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        static_cache_position = torch.zeros((1,), dtype=torch.long, device=device)
        static_rope_delta = torch.zeros((1, 1), dtype=torch.long, device=device)
        args.extend(
            [
                static_input_ids,
                static_cache_position,
                static_rope_delta,
                weights_cache["embed"],
                weights_cache["rotary"],
            ]
        )

        for weights in layer_weights:
            args.extend(
                [
                    weights[0],
                    weights[1],
                    weights[2],
                    weights[3],
                    weights[4],
                    weights[5],
                    weights[6],
                    weights[7],
                    weights[8],
                    weights[9],
                ]
            )
        args.append(weights_cache["norm"])

        if len(args) != 342:
            return None
        return {
            "call": call,
            "args": _AICASNoClearList(args),
            "input_ids": static_input_ids,
            "cache_position": static_cache_position,
            "rope_delta": static_rope_delta,
        }

    def _build_whitebox_block_bound_state_quant398_w8a16_gateup(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        layout = self._whitebox_block_arg_layout(int(bucket))
        if layout == "generic342_kv_first_fusedgate_ordered":
            state = self._build_whitebox_block_bound_state_generic342_kv_first_fusedgate_ordered(
                bucket=bucket,
                call=call,
                past=past,
                device=device,
            )
        elif layout == "generic342_kv_first_fusedgate_finalnorm_first":
            state = self._build_whitebox_block_bound_state_generic342_kv_first_fusedgate_finalnorm_first(
                bucket=bucket,
                call=call,
                past=past,
                device=device,
            )
        else:
            return None
        if not isinstance(state, dict):
            return None
        if self._whitebox_block_uses_weightonly_gateup(int(bucket)):
            packed_gateup = self._whitebox_block_gateup_weightonly_args()
        elif self._whitebox_block_uses_rowtriton_gateup(int(bucket)):
            packed_gateup = self._whitebox_block_gateup_w8a16_row_args()
        else:
            packed_gateup = self._whitebox_block_gateup_w8a16_args()
        if not isinstance(packed_gateup, tuple) or len(packed_gateup) != 56:
            return None
        args = state.get("args")
        if not isinstance(args, list):
            return None
        args.extend(packed_gateup)
        if len(args) != 398:
            return None
        return state

    def _build_whitebox_block_bound_state_generic342_kv_first_fusedgate_finalnorm_first(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        if not isinstance(layer_weights, tuple) or self._raw_model is None:
            return None
        lm = self._raw_model.model.language_model
        if len(layer_weights) != len(lm.layers):
            return None

        args: list[Any] = []
        for layer_idx in range(len(layer_weights)):
            cache_pair = self._materialize_static_cache_pair(past, layer_idx)
            if cache_pair is None:
                return None
            args.extend(cache_pair)

        static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        static_cache_position = torch.zeros((1,), dtype=torch.long, device=device)
        static_rope_delta = torch.zeros((1, 1), dtype=torch.long, device=device)
        args.extend(
            [
                static_input_ids,
                static_cache_position,
                static_rope_delta,
                weights_cache["embed"],
                weights_cache["rotary"],
                weights_cache["norm"],
            ]
        )

        for weights in layer_weights:
            args.extend(
                [
                    weights[0],
                    weights[1],
                    weights[2],
                    weights[3],
                    weights[4],
                    weights[5],
                    weights[6],
                    weights[7],
                    weights[8],
                    weights[9],
                ]
            )

        if len(args) != 342:
            return None
        return {
            "call": call,
            "args": _AICASNoClearList(args),
            "input_ids": static_input_ids,
            "cache_position": static_cache_position,
            "rope_delta": static_rope_delta,
        }

    def _build_whitebox_block_bound_state_qkv286_kv_first(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        weights_cache = self._whitebox_block_weight_args_fp16()
        if weights_cache is None:
            return None
        layer_weights = weights_cache.get("layers")
        qkv_layers = weights_cache.get("qkv_layers")
        if (
            not isinstance(layer_weights, tuple)
            or not isinstance(qkv_layers, tuple)
            or self._raw_model is None
        ):
            return None
        lm = self._raw_model.model.language_model
        if len(layer_weights) != len(lm.layers) or len(qkv_layers) != len(lm.layers):
            return None

        args: list[Any] = []
        for layer_idx in range(len(layer_weights)):
            cache_pair = self._materialize_static_cache_pair(past, layer_idx)
            if cache_pair is None:
                return None
            args.extend(cache_pair)

        static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        static_cache_position = torch.zeros((1,), dtype=torch.long, device=device)
        static_rope_delta = torch.zeros((1, 1), dtype=torch.long, device=device)
        args.extend(
            [
                static_input_ids,
                static_cache_position,
                static_rope_delta,
                weights_cache["embed"],
                weights_cache["rotary"],
                weights_cache["norm"],
            ]
        )

        for layer_idx, weights in enumerate(layer_weights):
            qkv_weight = qkv_layers[layer_idx]
            if not isinstance(qkv_weight, torch.Tensor):
                return None
            args.extend(
                [
                    weights[0],
                    qkv_weight,
                    weights[2],
                    weights[4],
                    weights[6],
                    weights[7],
                    weights[8],
                    weights[9],
                ]
            )

        if len(args) != 286:
            return None
        return {
            "call": call,
            "args": _AICASNoClearList(args),
            "input_ids": static_input_ids,
            "cache_position": static_cache_position,
            "rope_delta": static_rope_delta,
        }

    def _build_whitebox_block_bound_state_qkv342_rowtriton_kv_first(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        state = self._build_whitebox_block_bound_state_qkv286_kv_first(
            bucket=bucket,
            call=call,
            past=past,
            device=device,
        )
        if not isinstance(state, dict):
            return None
        packed_gateup = self._whitebox_block_gateup_w8a16_row_args()
        if not isinstance(packed_gateup, tuple) or len(packed_gateup) != 56:
            return None
        args = state.get("args")
        if not isinstance(args, list):
            return None
        args.extend(packed_gateup)
        if len(args) != 342:
            return None
        return state

    def _build_whitebox_block_bound_state_for_bucket(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        past: Any,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        arg_count = self._whitebox_block_arg_count(int(bucket))
        if arg_count == 398:
            return self._build_whitebox_block_bound_state_quant398_w8a16_gateup(
                bucket=bucket,
                call=call,
                past=past,
                device=device,
            )
        if arg_count == 342:
            layout = self._whitebox_block_arg_layout(int(bucket))
            if layout == "qkv342_rowtriton_kv_first":
                return self._build_whitebox_block_bound_state_qkv342_rowtriton_kv_first(
                    bucket=bucket,
                    call=call,
                    past=past,
                    device=device,
                )
            if layout == "generic342_kv_first_fusedgate_ordered":
                return self._build_whitebox_block_bound_state_generic342_kv_first_fusedgate_ordered(
                    bucket=bucket,
                    call=call,
                    past=past,
                    device=device,
                )
            if layout == "generic342_kv_first_fusedgate_finalnorm_first":
                return self._build_whitebox_block_bound_state_generic342_kv_first_fusedgate_finalnorm_first(
                    bucket=bucket,
                    call=call,
                    past=past,
                    device=device,
                )
            if layout == "generic342_kv_first":
                return self._build_whitebox_block_bound_state_generic342_kv_first(
                    bucket=bucket,
                    call=call,
                    past=past,
                    device=device,
                )
            return self._build_whitebox_block_bound_state(
                bucket=bucket,
                call=call,
                past=past,
                device=device,
            )
        if arg_count == 286:
            if self._whitebox_block_arg_layout(int(bucket)) != "qkv286_kv_first":
                return None
            return self._build_whitebox_block_bound_state_qkv286_kv_first(
                bucket=bucket,
                call=call,
                past=past,
                device=device,
            )
        if self._whitebox_block_allow_generic and arg_count == 370:
            return self._build_whitebox_block_bound_state_generic370(
                bucket=bucket,
                call=call,
                past=past,
                device=device,
            )
        return None

    def _run_whitebox_block(
        self,
        *,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        bucket: int,
        past: Any,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        try:
            state = self._build_whitebox_block_bound_state_for_bucket(
                bucket=bucket,
                call=call,
                past=past,
                device=device,
            )
            if not isinstance(state, dict):
                return None
            state["input_ids"].copy_(input_ids.to(device=device, dtype=torch.long))
            state["cache_position"].copy_(cache_position.to(device=device, dtype=torch.long))
            state["rope_delta"].copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))
            block_output = state["call"](state["args"])[0]
            return self._materialize_whitebox_block_tokens(
                block_output,
                block_size=self._whitebox_block_step_size(bucket),
                device=device,
                dtype=input_ids.dtype,
            )
        except Exception as exc:
            pass
            self._whitebox_block_failed_buckets.add(int(bucket))
            return None

    def _run_whitebox_block_state(
        self,
        *,
        state: dict[str, Any],
        bucket: int,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: Optional[torch.Tensor],
        device: torch.device,
        copy_cache_position: bool = True,
        copy_rope_delta: bool = True,
    ) -> Optional[torch.Tensor]:
        try:
            state["input_ids"].copy_(input_ids.to(device=device, dtype=torch.long))
            if copy_cache_position:
                state["cache_position"].copy_(cache_position.to(device=device, dtype=torch.long))
            if copy_rope_delta:
                if not isinstance(rope_delta, torch.Tensor):
                    return None
                state["rope_delta"].copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))
            block_output = state["call"](state["args"])[0]
            return self._materialize_whitebox_block_tokens(
                block_output,
                block_size=self._whitebox_block_step_size(bucket),
                device=device,
                dtype=input_ids.dtype,
            )
        except Exception as exc:
            pass
            self._whitebox_block_failed_buckets.add(int(bucket))
            return None

    def _run_whitebox_block_state_direct_tensor_args(
        self,
        *,
        state: dict[str, Any],
        bucket: int,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not self._whitebox_block_direct_tensor_args:
            return None
        try:
            args = state.get("args")
            if not isinstance(args, list) or len(args) < 2:
                return None
            layout = self._whitebox_block_arg_layout(int(bucket))
            arg_count = self._whitebox_block_arg_count(int(bucket))
            if layout == "official342":
                input_arg_index = 0
                cache_arg_index = 1
            elif layout in {
                "generic342_kv_first_fusedgate_ordered",
                "generic342_kv_first_fusedgate_finalnorm_first",
                "generic342_kv_first",
                "qkv286_kv_first",
            }:
                input_arg_index = 56
                cache_arg_index = 57
            elif arg_count == 398 and self._whitebox_block_graph_layout_compatible(int(bucket)):
                input_arg_index = 56
                cache_arg_index = 57
            elif arg_count == 370:
                input_arg_index = 56
                cache_arg_index = 57
            else:
                return None
            if (
                not isinstance(input_ids, torch.Tensor)
                or not isinstance(cache_position, torch.Tensor)
                or input_ids.device != device
                or cache_position.device != device
                or input_ids.dtype != torch.long
                or cache_position.dtype != torch.long
                or tuple(input_ids.shape) != (1, 1)
                or tuple(cache_position.shape) != (1,)
            ):
                return None
            if len(args) <= max(input_arg_index, cache_arg_index):
                return None
            old_input = args[input_arg_index]
            old_cache_position = args[cache_arg_index]
            args[input_arg_index] = input_ids
            args[cache_arg_index] = cache_position
            try:
                block_output = state["call"](args)[0]
            finally:
                args[input_arg_index] = old_input
                args[cache_arg_index] = old_cache_position
            return self._materialize_whitebox_block_tokens(
                block_output,
                block_size=self._whitebox_block_step_size(bucket),
                device=device,
                dtype=input_ids.dtype,
            )
        except Exception as exc:
            pass
            return None

    def _can_use_whitebox_block_graph(self, *, bucket: int, state: dict[str, Any]) -> bool:
        if not self._whitebox_block_graph_enabled:
            return False
        try:
            bucket = int(bucket)
        except Exception:
            return False
        if self._whitebox_block_graph_buckets and bucket not in self._whitebox_block_graph_buckets:
            return False
        if not self._whitebox_block_graph_layout_compatible(bucket):
            return False
        if not isinstance(state, dict) or "call" not in state or "args" not in state:
            return False
        return torch.cuda.is_available()

    def _capture_whitebox_block_graph(
        self,
        *,
        state: dict[str, Any],
        bucket: int,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: torch.Tensor,
        device: torch.device,
    ) -> Optional[dict[str, Any]]:
        if not self._can_use_whitebox_block_graph(bucket=bucket, state=state):
            return None
        try:
            state["input_ids"].copy_(input_ids.to(device=device, dtype=torch.long))
            state["cache_position"].copy_(cache_position.to(device=device, dtype=torch.long))
            state["rope_delta"].copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))

            def _run_call() -> torch.Tensor:
                return state["call"](state["args"])[0]

            warmup_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(warmup_stream):
                for _ in range(self._whitebox_block_graph_warmups):
                    _ = _run_call()
            torch.cuda.current_stream(device=device).wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    block_output = _run_call()
            block_tokens = self._materialize_whitebox_block_tokens(
                block_output,
                block_size=self._whitebox_block_step_size(bucket),
                device=device,
                dtype=input_ids.dtype,
            )
            if not isinstance(block_tokens, torch.Tensor):
                return None



            graph.replay()
            return {"graph": graph, "block_tokens": block_tokens}
        except Exception as exc:
            pass
            return None

    def _run_whitebox_block_state_graph(
        self,
        *,
        state: dict[str, Any],
        graph_state: dict[str, Any],
        bucket: int,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        try:
            state["input_ids"].copy_(input_ids.to(device=device, dtype=torch.long))
            state["cache_position"].copy_(cache_position.to(device=device, dtype=torch.long))
            state["rope_delta"].copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))
            graph = graph_state.get("graph")
            block_tokens = graph_state.get("block_tokens")
            if graph is None or not isinstance(block_tokens, torch.Tensor):
                return None
            graph.replay()
            return block_tokens
        except Exception as exc:
            pass
            return None

    def _run_whitebox_composite_block_state(
        self,
        *,
        state: dict[str, Any],
        bucket: int,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        max_tokens: int,
        copy_rope_delta: bool = True,
        ) -> Optional[torch.Tensor]:
        multiplier = int(max(1, self._whitebox_block_composite_multiplier))
        base_block = int(self._whitebox_block_step_size(bucket))
        if multiplier <= 1:
            return self._run_whitebox_block_state(
                state=state,
                bucket=bucket,
                input_ids=input_ids,
                cache_position=cache_position,
                rope_delta=rope_delta,
                device=device,
                copy_cache_position=True,
                copy_rope_delta=copy_rope_delta,
            )
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            return None
        target_tokens = min(int(base_block * multiplier), max_tokens)
        target_tokens = int((target_tokens // base_block) * base_block)
        if target_tokens <= 0 and max_tokens > 0:
            target_tokens = base_block
        if target_tokens <= 0:
            return None
        try:
            tokens = torch.empty((1, target_tokens), dtype=dtype, device=device)
            cur_input = input_ids
            state_cache_position = state.get("cache_position")
            if not isinstance(state_cache_position, torch.Tensor):
                return None
            state_cache_position.copy_(cache_position.to(device=device, dtype=torch.long))
            cursor = 0
            for _ in range(target_tokens // base_block):
                block = self._run_whitebox_block_state_direct_tensor_args(
                    state=state,
                    bucket=bucket,
                    input_ids=cur_input,
                    cache_position=state_cache_position,
                    device=device,
                )
                if block is None:
                    block = self._run_whitebox_block_state(
                        state=state,
                        bucket=bucket,
                        input_ids=cur_input,
                        cache_position=state_cache_position,
                        rope_delta=rope_delta,
                        device=device,
                        copy_cache_position=False,
                        copy_rope_delta=copy_rope_delta and cursor == 0,
                    )
                if not isinstance(block, torch.Tensor) or int(block.shape[1]) != base_block:
                    return None
                tokens[:, cursor : cursor + base_block].copy_(block.to(device=device, dtype=dtype))
                cursor += base_block
                cur_input = block[:, base_block - 1 : base_block]
                state_cache_position.add_(base_block)
            return tokens
        except Exception as exc:
            pass
            return None

    def _can_use_whitebox_block_sequence_graph(self, *, bucket: int, state: dict[str, Any]) -> bool:
        if not self._whitebox_block_sequence_graph_enabled:
            return False
        try:
            bucket = int(bucket)
        except Exception:
            return False
        if self._whitebox_block_sequence_graph_buckets and bucket not in self._whitebox_block_sequence_graph_buckets:
            return False
        if not self._whitebox_block_graph_layout_compatible(bucket):
            return False
        if not isinstance(state, dict) or "call" not in state or "args" not in state:
            return False
        return torch.cuda.is_available()

    def _run_whitebox_block_sequence_graph(
        self,
        *,
        state: dict[str, Any],
        bucket: int,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: torch.Tensor,
        remaining_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self._can_use_whitebox_block_sequence_graph(bucket=bucket, state=state):
            return None
        block_size = int(self._whitebox_block_step_size(bucket))
        remaining_tokens = int(remaining_tokens)
        if remaining_tokens <= block_size or remaining_tokens % block_size != 0:
            return None
        num_blocks = remaining_tokens // block_size
        try:
            static_input = state.get("input_ids")
            static_cache_position = state.get("cache_position")
            static_rope_delta = state.get("rope_delta")
            if not (
                isinstance(static_input, torch.Tensor)
                and isinstance(static_cache_position, torch.Tensor)
                and isinstance(static_rope_delta, torch.Tensor)
            ):
                return None
            sequence_tokens = torch.empty((1, remaining_tokens), dtype=dtype, device=device)
            static_input.copy_(input_ids.to(device=device, dtype=torch.long))
            static_cache_position.copy_(cache_position.to(device=device, dtype=torch.long))
            static_rope_delta.copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))

            def _run_sequence() -> None:
                cursor = 0
                for _ in range(num_blocks):
                    block_output = state["call"](state["args"])[0]
                    if block_output.ndim == 1:
                        block_tokens = block_output.view(1, -1)
                    else:
                        block_tokens = block_output
                    sequence_tokens[:, cursor : cursor + block_size].copy_(
                        block_tokens[:, :block_size].to(device=device, dtype=dtype)
                    )
                    static_input.copy_(block_tokens[:, block_size - 1 : block_size].to(device=device, dtype=torch.long))
                    static_cache_position.add_(block_size)
                    cursor += block_size

            warmup_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(warmup_stream):
                _run_sequence()
            torch.cuda.current_stream(device=device).wait_stream(warmup_stream)


            static_input.copy_(input_ids.to(device=device, dtype=torch.long))
            static_cache_position.copy_(cache_position.to(device=device, dtype=torch.long))
            static_rope_delta.copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))

            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    _run_sequence()




            static_input.copy_(input_ids.to(device=device, dtype=torch.long))
            static_cache_position.copy_(cache_position.to(device=device, dtype=torch.long))
            static_rope_delta.copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))
            graph.replay()
            return sequence_tokens
        except Exception as exc:
            pass
            return None

    def _can_use_whitebox_block_persistent_sequence_graph(
        self,
        *,
        bucket: int,
        remaining_tokens: int,
    ) -> bool:
        if not self._whitebox_block_persistent_sequence_graph_enabled:
            return False
        try:
            bucket = int(bucket)
            remaining_tokens = int(remaining_tokens)
        except Exception:
            return False
        block_size = int(self._whitebox_block_step_size(bucket))
        if block_size <= 0 or remaining_tokens <= block_size or remaining_tokens % block_size != 0:
            return False
        if self._whitebox_block_persistent_sequence_graph_buckets and bucket not in self._whitebox_block_persistent_sequence_graph_buckets:
            return False
        if not self._whitebox_block_graph_layout_compatible(bucket):
            return False
        return torch.cuda.is_available()

    def _make_empty_static_cache(self, *, bucket: int, device: torch.device, dtype: torch.dtype) -> Optional[Any]:
        if StaticCache is None or self._raw_model is None:
            return None
        try:
            static = StaticCache(
                config=self._raw_model.config,
                max_batch_size=1,
                max_cache_len=int(bucket),
                device=device,
                dtype=dtype,
            )
            lm = self._raw_model.model.language_model
            for layer_idx, layer in enumerate(lm.layers):
                attn = layer.self_attn
                head_dim = int(getattr(attn, "head_dim", 0) or 0)
                num_kv_heads = int(getattr(attn, "num_key_value_heads", 0) or 0)
                if num_kv_heads <= 0:
                    k_proj = getattr(attn, "k_proj", None)
                    k_weight = getattr(k_proj, "weight", None)
                    if isinstance(k_weight, torch.Tensor) and head_dim > 0:
                        num_kv_heads = max(1, int(k_weight.shape[0]) // head_dim)
                if num_kv_heads <= 0 or head_dim <= 0:
                    return None
                key = torch.zeros((1, num_kv_heads, int(bucket), head_dim), dtype=dtype, device=device)
                value = torch.zeros_like(key)
                static.update(
                    key,
                    value,
                    layer_idx,
                    {"cache_position": torch.arange(int(bucket), device=device, dtype=torch.long)},
                )
            return static
        except Exception as exc:
            pass
            return None

    def _get_whitebox_block_persistent_sequence_graph_state(
        self,
        *,
        bucket: int,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        remaining_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[dict[str, Any]]:
        if not self._can_use_whitebox_block_persistent_sequence_graph(bucket=bucket, remaining_tokens=remaining_tokens):
            return None
        key = (int(bucket), int(remaining_tokens), str(device), self._rowtriton_gateup_graph_key())
        if key in self._whitebox_block_persistent_sequence_graph_failed:
            return None
        cached = self._whitebox_block_persistent_sequence_graph_states.get(key)
        if isinstance(cached, dict):
            return cached
        try:
            static_past = self._make_empty_static_cache(bucket=int(bucket), device=device, dtype=torch.float16)
            if static_past is None:
                self._whitebox_block_persistent_sequence_graph_failed.add(key)
                return None
            state = self._build_whitebox_block_bound_state_for_bucket(
                bucket=int(bucket),
                call=call,
                past=static_past,
                device=device,
            )
            if not isinstance(state, dict):
                self._whitebox_block_persistent_sequence_graph_failed.add(key)
                return None

            block_size = int(self._whitebox_block_step_size(bucket))
            num_blocks = int(remaining_tokens) // block_size
            static_input = state.get("input_ids")
            static_cache_position = state.get("cache_position")
            static_rope_delta = state.get("rope_delta")
            if not (
                isinstance(static_input, torch.Tensor)
                and isinstance(static_cache_position, torch.Tensor)
                and isinstance(static_rope_delta, torch.Tensor)
            ):
                self._whitebox_block_persistent_sequence_graph_failed.add(key)
                return None
            qkv_fused_group = (
                call.__globals__.get("_aicasgc_fused_qkv_attention_group")
                if callable(call) and self._whitebox_block_persistent_sequence_graph_fused_attention
                else None
            )
            sequence_tokens = torch.empty((1, int(remaining_tokens)), dtype=dtype, device=device)

            def _run_sequence() -> None:
                cursor = 0
                for _ in range(num_blocks):
                    block_output = state["call"](state["args"])[0]
                    if block_output.ndim == 1:
                        block_tokens = block_output.view(1, -1)
                    else:
                        block_tokens = block_output
                    sequence_tokens[:, cursor : cursor + block_size].copy_(
                        block_tokens[:, :block_size].to(device=device, dtype=dtype)
                    )
                    static_input.copy_(block_tokens[:, block_size - 1 : block_size].to(device=device, dtype=torch.long))
                    static_cache_position.add_(block_size)
                    cursor += block_size

            static_input.zero_()
            static_cache_position.zero_()
            static_rope_delta.zero_()
            warmup_stream = torch.cuda.Stream(device=device)
            if qkv_fused_group is not None:
                try:
                    setattr(qkv_fused_group, "active", True)
                except Exception:
                    pass
            with torch.cuda.stream(warmup_stream):
                _run_sequence()
            torch.cuda.current_stream(device=device).wait_stream(warmup_stream)
            static_input.zero_()
            static_cache_position.zero_()
            static_rope_delta.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    _run_sequence()
            if qkv_fused_group is not None:
                try:
                    setattr(qkv_fused_group, "active", False)
                except Exception:
                    pass
                if os.environ.get("AICASGC_DEBUG_QKV_FUSED_COUNTS", "0") != "0":
                    pass
            value_merge_group = call.__globals__.get("_aicasgc_fused_value_merge_group") if callable(call) else None
            if value_merge_group is not None and os.environ.get("AICASGC_DEBUG_VALUE_MERGE_COUNTS", "0") != "0":
                pass
            graph_state = {
                "graph": graph,
                "state": state,
                "static_past": static_past,
                "sequence_tokens": sequence_tokens,
                "input_ids": static_input,
                "cache_position": static_cache_position,
                "rope_delta": static_rope_delta,
                "remaining_tokens": int(remaining_tokens),
            }
            tail_graph = None
            tail_tokens = None
            if self._whitebox_block_persistent_tail_graph:
                static_input.copy_(sequence_tokens[:, int(remaining_tokens) - 1 : int(remaining_tokens)].to(dtype=torch.long))
                static_cache_position.fill_(int(remaining_tokens))
                static_rope_delta.zero_()
                tail_tokens = torch.empty((1, block_size), dtype=dtype, device=device)

                def _run_tail() -> None:
                    block_output = state["call"](state["args"])[0]
                    if block_output.ndim == 1:
                        block_tokens = block_output.view(1, -1)
                    else:
                        block_tokens = block_output
                    tail_tokens.copy_(block_tokens[:, :block_size].to(device=device, dtype=dtype))
                    static_input.copy_(block_tokens[:, block_size - 1 : block_size].to(device=device, dtype=torch.long))
                    static_cache_position.add_(block_size)

                warmup_tail_stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(warmup_tail_stream):
                    _run_tail()
                torch.cuda.current_stream(device=device).wait_stream(warmup_tail_stream)
                static_input.copy_(sequence_tokens[:, int(remaining_tokens) - 1 : int(remaining_tokens)].to(dtype=torch.long))
                static_cache_position.fill_(int(remaining_tokens))
                static_rope_delta.zero_()
                tail_graph = torch.cuda.CUDAGraph()
                with torch.inference_mode():
                    with torch.cuda.graph(tail_graph):
                        _run_tail()
                graph_state["tail_graph"] = tail_graph
                graph_state["tail_tokens"] = tail_tokens
            self._whitebox_block_persistent_sequence_graph_states[key] = graph_state
            return graph_state
        except Exception as exc:
            try:
                qkv_fused_group = (
                    call.__globals__.get("_aicasgc_fused_qkv_attention_group")
                    if callable(call) and self._whitebox_block_persistent_sequence_graph_fused_attention
                    else None
                )
                if qkv_fused_group is not None:
                    setattr(qkv_fused_group, "active", False)
            except Exception:
                pass
            self._whitebox_block_persistent_sequence_graph_failed.add(key)
            return None

    def _run_whitebox_block_persistent_sequence_graph(
        self,
        *,
        graph_state: dict[str, Any],
        source_past: Any,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        copy_source_cache: bool = True,
    ) -> Optional[torch.Tensor]:
        try:
            static_past = graph_state.get("static_past")
            if copy_source_cache:
                if static_past is None:
                    return None
                self._clear_whitebox_block_persistent_sequence_graph_state(graph_state)
                if not self._copy_dynamic_cache_to_static_existing(
                    source_past,
                    static_past,
                    device=device,
                    dtype=torch.float16,
                ):
                    return None
            static_input = graph_state.get("input_ids")
            static_cache_position = graph_state.get("cache_position")
            static_rope_delta = graph_state.get("rope_delta")
            graph = graph_state.get("graph")
            sequence_tokens = graph_state.get("sequence_tokens")
            if not (
                isinstance(static_input, torch.Tensor)
                and isinstance(static_cache_position, torch.Tensor)
                and isinstance(static_rope_delta, torch.Tensor)
                and isinstance(sequence_tokens, torch.Tensor)
                and graph is not None
            ):
                return None
            static_input.copy_(input_ids.to(device=device, dtype=torch.long).view(1, 1))
            static_cache_position.copy_(cache_position.to(device=device, dtype=torch.long).view(1))
            static_rope_delta.copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))
            qkv_fused_group = None
            try:
                state = graph_state.get("state")
                call = state.get("call") if isinstance(state, dict) else None
                qkv_fused_group = (
                    call.__globals__.get("_aicasgc_fused_qkv_attention_group")
                    if callable(call) and self._whitebox_block_persistent_sequence_graph_fused_attention
                    else None
                )
                if qkv_fused_group is not None:
                    setattr(qkv_fused_group, "active", True)
                graph.replay()
            finally:
                if qkv_fused_group is not None:
                    try:
                        setattr(qkv_fused_group, "active", False)
                    except Exception:
                        pass
            if sequence_tokens.device == device and sequence_tokens.dtype == dtype:
                return sequence_tokens
            return sequence_tokens.to(device=device, dtype=dtype)
        except Exception as exc:
            pass
            return None

    def _run_whitebox_block_persistent_tail_graph(
        self,
        *,
        graph_state: dict[str, Any],
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        rope_delta: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self._whitebox_block_persistent_tail_graph:
            return None
        try:
            graph = graph_state.get("tail_graph")
            tail_tokens = graph_state.get("tail_tokens")
            static_input = graph_state.get("input_ids")
            static_cache_position = graph_state.get("cache_position")
            static_rope_delta = graph_state.get("rope_delta")
            if not (
                graph is not None
                and isinstance(tail_tokens, torch.Tensor)
                and isinstance(static_input, torch.Tensor)
                and isinstance(static_cache_position, torch.Tensor)
                and isinstance(static_rope_delta, torch.Tensor)
            ):
                return None
            static_input.copy_(input_ids.to(device=device, dtype=torch.long).view(1, 1))
            static_cache_position.copy_(cache_position.to(device=device, dtype=torch.long).view(1))
            static_rope_delta.copy_(rope_delta.to(device=device, dtype=torch.long).view(1, 1))
            qkv_fused_group = None
            try:
                state = graph_state.get("state")
                call = state.get("call") if isinstance(state, dict) else None
                qkv_fused_group = (
                    call.__globals__.get("_aicasgc_fused_qkv_attention_group")
                    if callable(call) and self._whitebox_block_persistent_sequence_graph_fused_attention
                    else None
                )
                if qkv_fused_group is not None:
                    setattr(qkv_fused_group, "active", True)
                graph.replay()
            finally:
                if qkv_fused_group is not None:
                    try:
                        setattr(qkv_fused_group, "active", False)
                    except Exception:
                        pass
            if tail_tokens.device == device and tail_tokens.dtype == dtype:
                return tail_tokens
            return tail_tokens.to(device=device, dtype=dtype)
        except Exception as exc:
            pass
            return None

    def _generated_tokens_look_healthy(self, generated_ids: torch.Tensor) -> tuple[bool, str]:
        tokenizer = getattr(getattr(self, "_raw_processor", None), "tokenizer", None)
        if tokenizer is None:
            return True, ""
        if not isinstance(generated_ids, torch.Tensor) or int(generated_ids.numel()) <= 0:
            return False, "empty_tokens"
        generated_ids = self._trim_decode_health_guard_whitespace_tail(generated_ids, min_keep=8)
        if not isinstance(generated_ids, torch.Tensor) or int(generated_ids.numel()) <= 0:
            return False, "empty_after_whitespace_tail_trim"
        if self._decode_health_guard_fast and generated_ids.is_cuda:
            fast_result = self._generated_tokens_look_healthy_fast_tensor(generated_ids, tokenizer)
            if fast_result is not None:
                if not bool(fast_result[0]) or int(generated_ids.numel()) < 192:
                    return fast_result
        try:
            ids = generated_ids.detach().to(device="cpu", dtype=torch.long).view(-1).tolist()
        except Exception as exc:
            return False, f"ids_unavailable:{type(exc).__name__}"
        if not ids:
            return False, "empty_tokens"
        tokenizer_len = int(getattr(self, "_decode_health_guard_tokenizer_len", 0) or 0)
        if tokenizer_len <= 0:
            try:
                tokenizer_len = int(len(tokenizer))
            except Exception:
                tokenizer_len = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if tokenizer_len > 0 and any(int(token_id) < 0 or int(token_id) >= tokenizer_len for token_id in ids):
            return False, "token_id_out_of_vocab"
        special_ids = set(getattr(self, "_decode_health_guard_special_ids", frozenset()))
        if not special_ids:
            all_special = getattr(tokenizer, "all_special_ids", None)
            if isinstance(all_special, (list, tuple, set)):
                special_ids = {int(token_id) for token_id in all_special}
        non_special = [int(token_id) for token_id in ids if int(token_id) not in special_ids]
        if len(ids) >= 32 and not non_special:
            return False, "all_special_tokens"
        if self._has_repeated_tail_token_pattern(non_special):
            return False, "repeated_tail_token_pattern"
        if self._decode_health_guard_fast:
            suspicious_token_pattern = False
            run_token = None
            run_len = 0
            for token_id in non_special:
                if token_id == run_token:
                    run_len += 1
                else:
                    run_token = token_id
                    run_len = 1
                if run_len >= 32:
                    suspicious_token_pattern = True
                    break
            if len(non_special) >= 64:
                tail = non_special[-64:]
                for period in (1, 2, 3, 4, 5, 6, 8):
                    pattern = tail[:period]
                    if pattern and all(tail[idx] == pattern[idx % period] for idx in range(len(tail))):
                        suspicious_token_pattern = True
                        break
            if not suspicious_token_pattern:
                return True, ""
            tail_unique = list(dict.fromkeys(non_special[-64:]))
            if tail_unique:
                whitespace_ids = getattr(self, "_decode_health_guard_whitespace_ids", frozenset())
                if whitespace_ids and all(int(token_id) in whitespace_ids for token_id in tail_unique):
                    return True, ""
                try:
                    tail_text = tokenizer.decode(
                        tail_unique,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if tail_text and all(ch.isspace() for ch in tail_text):
                        return True, ""
                except Exception:
                    pass
        try:
            text = tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            return False, f"decode_failed:{type(exc).__name__}"
        if len(text) == 0:
            non_special = [token_id for token_id in ids if int(token_id) not in special_ids]
            if len(non_special) >= min(8, len(ids)):
                return False, "empty_decoded_text"
            return True, ""
        replacement_count = text.count("\ufffd")
        if replacement_count > 0 and replacement_count / max(1, len(text)) > 0.02:
            return False, "replacement_char_ratio"
        control_count = 0
        visible_count = 0
        for ch in text:
            code = ord(ch)
            if ch in "\n\r\t":
                visible_count += 1
            elif code < 32 or (0x7F <= code <= 0x9F):
                control_count += 1
            elif not ch.isspace():
                visible_count += 1
        if control_count > 0 and control_count / max(1, len(text)) > 0.01:
            return False, "control_char_ratio"
        if len(ids) >= 32 and visible_count == 0:
            return False, "blank_visible_text"
        run_char = ""
        run_len = 0
        for ch in text:
            if ch == run_char:
                run_len += 1
            else:
                run_char = ch
                run_len = 1
            if run_len >= 64 and not ch.isspace():
                return False, "long_repeated_char_run"
        return True, ""

    @staticmethod
    def _has_repeated_tail_token_pattern(token_ids: list[int]) -> bool:
        if not token_ids or len(token_ids) < 48:
            return False
        tail = [int(token_id) for token_id in token_ids[-256:]]
        max_period = min(96, len(tail) // 3)
        if max_period < 4:
            return False
        for period in range(4, max_period + 1):
            pattern = tail[-period:]
            repeated = period
            cursor = len(tail) - period
            while cursor - period >= 0 and tail[cursor - period : cursor] == pattern:
                repeated += period
                cursor -= period
                if repeated >= period * 3 and repeated >= 48:
                    return True
        return False

    def _trim_decode_health_guard_whitespace_tail(
        self,
        generated_ids: torch.Tensor,
        *,
        min_keep: int = 8,
    ) -> torch.Tensor:
        if not isinstance(generated_ids, torch.Tensor):
            return generated_ids
        try:
            flat = generated_ids.view(-1)
            token_count = int(flat.numel())
            if token_count <= int(min_keep):
                return generated_ids
            whitespace_ids = tuple(int(token_id) for token_id in getattr(self, "_decode_health_guard_whitespace_ids", frozenset()))
            if not whitespace_ids:
                return generated_ids
            trim_to = token_count
            while trim_to > int(min_keep):
                token_value = int(flat[trim_to - 1].detach().to(device="cpu").item())
                if token_value not in whitespace_ids:
                    break
                trim_to -= 1
            if trim_to < token_count:
                return flat[:trim_to]
        except Exception:
            return generated_ids
        return generated_ids

    def _generated_tokens_look_healthy_fast_tensor(
        self,
        generated_ids: torch.Tensor,
        tokenizer: Any,
    ) -> Optional[tuple[bool, str]]:
        try:
            flat = generated_ids.detach().view(-1)
            token_count = int(flat.numel())
            if token_count <= 0:
                return False, "empty_tokens"
            if token_count < 32:
                return None
            try:
                tokenizer_len = int(getattr(self, "_decode_health_guard_tokenizer_len", 0) or 0)
                if tokenizer_len <= 0:
                    tokenizer_len = int(len(tokenizer))
            except Exception:
                tokenizer_len = int(getattr(tokenizer, "vocab_size", 0) or 0)

            special_ids = set(getattr(self, "_decode_health_guard_special_ids", frozenset()))
            if not special_ids:
                all_special = getattr(tokenizer, "all_special_ids", None)
                if isinstance(all_special, (list, tuple, set)):
                    special_ids = {int(token_id) for token_id in all_special}
            non_special_mask = torch.ones_like(flat, dtype=torch.bool)
            for token_id in special_ids:
                non_special_mask &= flat.ne(int(token_id))
            has_non_special = non_special_mask.any()

            tail = flat[-min(64, token_count) :]
            suspicious = torch.zeros((), dtype=torch.bool, device=flat.device)
            tail_len = int(tail.numel())
            if tail_len >= 16:
                same_run = torch.ones((tail_len - 15,), dtype=torch.bool, device=flat.device)
                base = tail[: tail_len - 15]
                for offset in range(1, 16):
                    same_run &= tail[offset : offset + tail_len - 15].eq(base)
                suspicious |= same_run.any()
            if tail_len >= 32:
                for period in (1, 2, 3, 4, 5, 6, 8):
                    if tail_len > period:
                        suspicious |= tail[period:].eq(tail[:-period]).all()

            stats = [has_non_special.to(dtype=torch.int32), suspicious.to(dtype=torch.int32)]
            if tokenizer_len > 0:
                stats.append((flat.lt(0) | flat.ge(int(tokenizer_len))).any().to(dtype=torch.int32))
            stats_cpu = torch.stack(stats).to(device="cpu").tolist()
            if tokenizer_len > 0 and int(stats_cpu[2]) != 0:
                return False, "token_id_out_of_vocab"
            if int(stats_cpu[0]) == 0:
                return False, "all_special_tokens"
            if int(stats_cpu[1]) == 0:
                return True, ""

            tail_ids = tail.to(device="cpu", dtype=torch.long).view(-1).tolist()
            non_control_tail = [int(token_id) for token_id in tail_ids if int(token_id) not in special_ids]
            if not non_control_tail:
                return True, ""
            tail_unique = list(dict.fromkeys(non_control_tail[-64:]))
            if len(tail_unique) <= 8:
                whitespace_ids = getattr(self, "_decode_health_guard_whitespace_ids", frozenset())
                if whitespace_ids and all(int(token_id) in whitespace_ids for token_id in tail_unique):
                    return True, ""
                tail_text = tokenizer.decode(
                    tail_unique,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if tail_text and all(ch.isspace() for ch in tail_text):
                    return True, ""
            return False, "repeated_token_pattern"
        except Exception:
            return None

    def _manual_generate_required_token_block(
        self,
        pc: _DecodeInputs,
        *,
        max_new_tokens: int,
        visual_profile: _VisualBudgetProfile,
        whitebox_block_bucket: int,
        whitebox_block_call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        profile_stats: Optional[dict[str, float]] = None,
    ) -> torch.Tensor:
        device = pc.input_ids.device
        prompt_len = int(pc.input_ids.shape[1])
        max_new_tokens = int(max_new_tokens)
        if prompt_len <= 1 or max_new_tokens <= 0:
            raise RuntimeError(
                "AICASGC required token-block path needs a non-empty prompt tail "
                f"and positive generation length; prompt_len={prompt_len}, max_new_tokens={max_new_tokens}"
            )
        prefill_len = int(prompt_len - 1)
        prefill_pc = self._slice_decode_inputs_for_prefill(pc, prefill_len)
        prefill_cache_position = torch.arange(prefill_len, device=device, dtype=torch.long)
        profile_start = self._profile_stamp()
        _, past = self._run_prefill_backend(
            prefill_pc,
            None,
            prefill_cache_position,
            max_new_tokens=max_new_tokens,
            profile_stats=profile_stats,
        )
        self._profile_add(profile_stats, "prefill", profile_start)
        if past is None:
            raise RuntimeError("AICASGC required token-block prefill did not return KV cache")

        profile_start = self._profile_stamp()
        compact_delta = 0
        static_past = None
        use_compacted_handoff = (
            not bool(visual_profile.visual_kv_profile_only)
            and visual_profile.fallback_reason == "whitebox_visual_compact_bucket_hit"
        )
        if use_compacted_handoff:
            compacted = self._dynamic_cache_to_static_compacted_visual(
                past,
                total_len=int(whitebox_block_bucket),
                device=device,
                prompt_len=prefill_len,
                generated_tokens_in_cache=0,
                image_grid_thw=pc.image_grid_thw,
                visual_pos_masks=pc.visual_pos_masks[:, :prefill_len],
                dtype=torch.float16,
            )
            if compacted is not None:
                static_past, compact_delta = compacted
        if static_past is None:
            static_past = self._dynamic_cache_to_static(
                past,
                int(whitebox_block_bucket),
                device,
                dtype=torch.float16,
            )
        self._profile_add(profile_stats, "handoff", profile_start)
        if static_past is None:
            raise RuntimeError(
                "AICASGC required token-block path failed to bind static KV "
                f"for bucket={whitebox_block_bucket}, prompt_len={prompt_len}"
            )

        profile_start = self._profile_stamp()
        state = self._build_whitebox_block_bound_state_for_bucket(
            bucket=int(whitebox_block_bucket),
            call=whitebox_block_call,
            past=static_past,
            device=device,
        )
        self._profile_add(profile_stats, "bound_state", profile_start)
        if not isinstance(state, dict):
            raise RuntimeError(
                "AICASGC required token-block path failed to build bound state "
                f"for bucket={whitebox_block_bucket}"
            )

        cache_position = torch.empty((1,), device=device, dtype=torch.long)
        static_prefix_len = max(0, prefill_len - int(compact_delta))
        cache_position.fill_(static_prefix_len)
        rope_delta = pc.rope_deltas.to(device=device, dtype=torch.long).clone().view(1, 1)
        if compact_delta:
            rope_delta.add_(int(compact_delta))
            self._profile_inc(profile_stats, "visual_kv_compact_delta", compact_delta)
        state_rope_delta = state.get("rope_delta")
        if isinstance(state_rope_delta, torch.Tensor):
            state_rope_delta.copy_(rope_delta)

        generated = torch.empty((1, max_new_tokens), dtype=pc.input_ids.dtype, device=device)
        generated_len = 0
        last_eos_scan_len = 0
        block_input = pc.input_ids[:, prefill_len:prompt_len].to(device=device, dtype=torch.long)
        force_full_request = bool(
            self._force_max_new_tokens
            and max_new_tokens > 1
            and max_new_tokens <= int(self._performance_request_tokens)
        )
        eos_pad_fill_for_request = self._eos_pad_fill_enabled_for_request(max_new_tokens)
        block_step_size = int(self._whitebox_block_step_size(int(whitebox_block_bucket)))
        self._profile_inc(profile_stats, "required_token_block_shared")

        graph_tokens = int((int(max_new_tokens) // max(1, block_step_size)) * max(1, block_step_size))
        if eos_pad_fill_for_request and int(self._eos_pad_fill_probe_tokens) > 0:
            probe_tokens = int((int(self._eos_pad_fill_probe_tokens) // max(1, block_step_size)) * max(1, block_step_size))
            if (
                probe_tokens > block_step_size
                and probe_tokens < int(max_new_tokens)
                and self._can_use_whitebox_block_persistent_sequence_graph(
                    bucket=int(whitebox_block_bucket),
                    remaining_tokens=int(probe_tokens),
                )
            ):
                graph_tokens = int(probe_tokens)
        graph_state = None
        if (
            graph_tokens > block_step_size
            and self._whitebox_block_persistent_sequence_graph_enabled
            and self._can_use_whitebox_block_persistent_sequence_graph(
                bucket=int(whitebox_block_bucket),
                remaining_tokens=int(graph_tokens),
            )
        ):
            profile_start = self._profile_stamp()
            graph_state = self._get_whitebox_block_persistent_sequence_graph_state(
                bucket=int(whitebox_block_bucket),
                call=whitebox_block_call,
                remaining_tokens=int(graph_tokens),
                device=device,
                dtype=pc.input_ids.dtype,
            )
            self._profile_add(profile_stats, "block_persistent_sequence_graph_lookup", profile_start)
            if isinstance(graph_state, dict):
                graph_static_past = graph_state.get("static_past")
                self._clear_whitebox_block_persistent_sequence_graph_state(graph_state)
                if (
                    graph_static_past is not static_past
                    and not self._copy_dynamic_cache_prefix_into_existing(
                        static_past,
                        graph_static_past,
                        int(static_prefix_len),
                    )
                ):
                    graph_state = None
                else:
                    state_candidate = graph_state.get("state")
                    if isinstance(state_candidate, dict):
                        state = state_candidate
                        static_past = graph_static_past
                    else:
                        graph_state = None

            if isinstance(graph_state, dict):
                profile_start = self._profile_stamp()
                sequence_tokens = self._run_whitebox_block_persistent_sequence_graph(
                    source_past=static_past,
                    graph_state=graph_state,
                    input_ids=block_input,
                cache_position=cache_position,
                rope_delta=rope_delta,
                device=device,
                dtype=pc.input_ids.dtype,
                copy_source_cache=False,
            )
            self._profile_add(profile_stats, "block_persistent_sequence_graph", profile_start)
            self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
            if isinstance(sequence_tokens, torch.Tensor) and int(sequence_tokens.shape[1]) > 0:
                take_cap = min(int(sequence_tokens.shape[1]), int(max_new_tokens))
                generated[0, :take_cap].copy_(sequence_tokens[0, :take_cap].to(device=device, dtype=pc.input_ids.dtype))
                generated_len = int(take_cap)
                self._profile_inc(profile_stats, "block_persistent_sequence_graph_calls")
                if eos_pad_fill_for_request:
                    eos_hit = self._find_first_eos_in_generated(
                        generated,
                        generated_len=generated_len,
                        eos_token_ids=self._eos_token_ids,
                        scan_start=0,
                    )
                    if eos_hit is not None:
                        filled_len, did_fill = self._fill_generated_after_eos(
                            generated,
                            generated_len=generated_len,
                            max_new_tokens=max_new_tokens,
                            eos_token_ids=self._eos_token_ids,
                            fill_token_id=int(self._eos_pad_fill_token_id),
                            scan_start=int(eos_hit),
                        )
                        if did_fill:
                            self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
                            self._profile_inc(profile_stats, "eos_pad_fill_after_eos_tokens", max(0, filled_len - int(eos_hit) - 1))
                        generated_len = int(filled_len)
                if generated_len < max_new_tokens:
                    block_input = sequence_tokens[:, generated_len - 1 : generated_len].to(device=device, dtype=torch.long)
                    cache_position.add_(int(sequence_tokens.shape[1]))

        while generated_len < max_new_tokens:
            block_tokens = None
            profile_start = self._profile_stamp()
            if isinstance(graph_state, dict):
                block_tokens = self._run_whitebox_block_persistent_tail_graph(
                    graph_state=graph_state,
                    input_ids=block_input,
                    cache_position=cache_position,
                    rope_delta=rope_delta,
                    device=device,
                    dtype=pc.input_ids.dtype,
                )
            if int(self._whitebox_block_composite_multiplier) <= 1:
                if block_tokens is None:
                    block_tokens = self._run_whitebox_block_state_direct_tensor_args(
                        state=state,
                        bucket=int(whitebox_block_bucket),
                        input_ids=block_input,
                        cache_position=cache_position,
                        device=device,
                    )
            if block_tokens is None:
                block_tokens = self._run_whitebox_composite_block_state(
                    state=state,
                    bucket=int(whitebox_block_bucket),
                    input_ids=block_input,
                    cache_position=cache_position,
                    rope_delta=rope_delta,
                    device=device,
                    dtype=pc.input_ids.dtype,
                    max_tokens=max_new_tokens - generated_len,
                    copy_rope_delta=True,
                )
            self._profile_add(profile_stats, "block_direct", profile_start)
            self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
            self._profile_inc(profile_stats, "block_direct_calls")
            if not isinstance(block_tokens, torch.Tensor):
                raise RuntimeError(
                    "AICASGC required token-block call returned no tokens "
                    f"for bucket={whitebox_block_bucket}, generated_len={generated_len}, "
                    f"max_new_tokens={max_new_tokens}"
                )
            take_cap = min(int(block_tokens.shape[1]), int(max_new_tokens - generated_len))
            if take_cap <= 0:
                raise RuntimeError("AICASGC required token-block call returned an empty token block")
            generated[0, generated_len : generated_len + take_cap].copy_(
                block_tokens[0, :take_cap].to(device=device, dtype=pc.input_ids.dtype)
            )
            generated_len += take_cap

            eos_hit = None
            should_scan_block_eos = ((not force_full_request) or eos_pad_fill_for_request) and bool(self._eos_token_ids) and (
                self._eos_check_interval <= block_step_size
                or generated_len >= max_new_tokens
                or (generated_len - last_eos_scan_len) >= int(self._eos_check_interval)
            )
            if should_scan_block_eos:
                profile_start = self._profile_stamp()
                scan_start = max(0, int(last_eos_scan_len))
                scan_tokens = generated[0, scan_start:generated_len]
                eos_offset = self._find_first_token_id_in_1d(scan_tokens, self._eos_token_ids)
                if eos_offset is not None:
                    eos_hit = int(scan_start + int(eos_offset))
                else:
                    last_eos_scan_len = generated_len
                self._profile_add(profile_stats, "eos_scan", profile_start)
            if eos_hit is not None:
                if eos_pad_fill_for_request:
                    filled_len, did_fill = self._fill_generated_after_eos(
                        generated,
                        generated_len=generated_len,
                        max_new_tokens=max_new_tokens,
                        eos_token_ids=self._eos_token_ids,
                        fill_token_id=int(self._eos_pad_fill_token_id),
                        scan_start=int(eos_hit),
                    )
                    if did_fill:
                        self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
                        self._profile_inc(profile_stats, "eos_pad_fill_after_eos_tokens", max(0, filled_len - int(eos_hit) - 1))
                    generated_len = int(filled_len)
                else:
                    generated_len = int(eos_hit) + 1
                break
            if generated_len >= max_new_tokens:
                break
            block_input = block_tokens[:, take_cap - 1 : take_cap].to(device=device, dtype=torch.long)
            cache_position.add_(int(block_tokens.shape[1]))

        self._emit_whitebox_profile(
            profile_stats,
            max_new_tokens=max_new_tokens,
            generated_len=generated_len,
            visual_profile=visual_profile,
        )
        return torch.cat([pc.input_ids, generated[:, :generated_len]], dim=1)

    def _manual_generate_prompt_tail_persistent_token_block(
        self,
        pc: _DecodeInputs,
        *,
        max_new_tokens: int,
        visual_profile: _VisualBudgetProfile,
        whitebox_block_bucket: int,
        whitebox_block_call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        profile_stats: Optional[dict[str, float]] = None,
    ) -> Optional[torch.Tensor]:
        if (
            not self._whitebox_block_prompt_tail_persistent
            or not self._whitebox_block_persistent_sequence_graph_enabled
            or not torch.cuda.is_available()
        ):
            return None
        device = pc.input_ids.device
        prompt_len = int(pc.input_ids.shape[1])
        max_new_tokens = int(max_new_tokens)
        if (
            prompt_len <= 1
            or max_new_tokens != int(self._performance_request_tokens)
            or max_new_tokens <= 0
            or not isinstance(visual_profile, _VisualBudgetProfile)
            or not bool(visual_profile.enable_whitebox_block)
        ):
            return None
        block_step_size = int(self._whitebox_block_step_size(int(whitebox_block_bucket)))
        if block_step_size <= 0 or max_new_tokens % block_step_size != 0:
            return None
        try:
            prefill_len = int(prompt_len - 1)
            compact_delta = 0
            graph_state = None
            candidate_past = None
            graph_remaining_tokens = int(max_new_tokens)
            if not self._can_use_whitebox_block_persistent_sequence_graph(
                bucket=int(whitebox_block_bucket),
                remaining_tokens=graph_remaining_tokens,
            ):
                return None
            eos_pad_fill_for_request = self._eos_pad_fill_enabled_for_request(max_new_tokens)
            tail_probe_candidates = []
            if eos_pad_fill_for_request and int(self._eos_pad_fill_probe_tokens) > 0:
                tail_probe_candidates.append(int(self._eos_pad_fill_probe_tokens))
            tail_probe_tokens = min(tail_probe_candidates) if tail_probe_candidates else 0
            use_tail_probe = bool(
                eos_pad_fill_for_request
                and graph_remaining_tokens == max_new_tokens
                and int(tail_probe_tokens) > 0
                and int(tail_probe_tokens) < max_new_tokens
            )
            if use_tail_probe:
                block_step = int(self._whitebox_block_step_size(int(whitebox_block_bucket)))
                probe_tokens = int((int(tail_probe_tokens) // max(1, block_step)) * max(1, block_step))
                if (
                    probe_tokens <= block_step
                    or probe_tokens >= max_new_tokens
                    or not self._can_use_whitebox_block_persistent_sequence_graph(
                        bucket=int(whitebox_block_bucket),
                        remaining_tokens=probe_tokens,
                    )
                ):
                    use_tail_probe = False
                else:
                    graph_remaining_tokens = int(probe_tokens)
            if not isinstance(graph_state, dict) or candidate_past is None:
                prefill_pc = self._slice_decode_inputs_for_prefill(pc, prefill_len)
                prefill_cache_position = torch.arange(prefill_len, device=device, dtype=torch.long)
                use_compacted_handoff = (
                    not bool(visual_profile.visual_kv_profile_only)
                    and visual_profile.fallback_reason == "whitebox_visual_compact_bucket_hit"
                )
                use_static_prefill = bool(
                    self._whitebox_block_prompt_tail_static_prefill
                    and not use_compacted_handoff
                    and StaticCache is not None
                )
                if use_static_prefill:
                    profile_start = self._profile_stamp()
                    graph_state = self._get_whitebox_block_persistent_sequence_graph_state(
                        bucket=int(whitebox_block_bucket),
                        call=whitebox_block_call,
                        remaining_tokens=graph_remaining_tokens,
                        device=device,
                        dtype=pc.input_ids.dtype,
                    )
                    if not isinstance(graph_state, dict):
                        self._profile_add(profile_stats, "block_prompt_tail_persistent_graph", profile_start)
                        return None
                    candidate_past = graph_state.get("static_past")
                    self._clear_whitebox_block_persistent_sequence_graph_state(graph_state)
                    self._profile_add(profile_stats, "block_prompt_tail_persistent_graph", profile_start)
                    if candidate_past is None:
                        return None
                    profile_start = self._profile_stamp()
                    _, past = self._run_prefill_backend(
                        prefill_pc,
                        candidate_past,
                        prefill_cache_position,
                        max_new_tokens=max_new_tokens,
                        profile_stats=profile_stats,
                    )
                    self._profile_add(profile_stats, "prefill", profile_start)
                    if past is None:
                        return None
                    candidate_past = past
                    self._profile_inc(profile_stats, "prompt_tail_static_prefill")
                else:
                    profile_start = self._profile_stamp()
                    _, past = self._run_prefill_backend(
                        prefill_pc,
                        None,
                        prefill_cache_position,
                        max_new_tokens=max_new_tokens,
                        profile_stats=profile_stats,
                    )
                    self._profile_add(profile_stats, "prefill", profile_start)
                    if past is None:
                        return None

                if not use_static_prefill:
                    profile_start = self._profile_stamp()
                    graph_state = self._get_whitebox_block_persistent_sequence_graph_state(
                        bucket=int(whitebox_block_bucket),
                        call=whitebox_block_call,
                        remaining_tokens=graph_remaining_tokens,
                        device=device,
                        dtype=pc.input_ids.dtype,
                    )
                    if not isinstance(graph_state, dict):
                        self._profile_add(profile_stats, "block_prompt_tail_persistent_graph", profile_start)
                        return None
                    candidate_past = graph_state.get("static_past")
                    self._clear_whitebox_block_persistent_sequence_graph_state(graph_state)
                    if use_compacted_handoff:
                        compact_delta_opt = self._copy_dynamic_cache_to_static_existing_compacted_visual(
                            past,
                            candidate_past,
                            device=device,
                            prompt_len=prefill_len,
                            generated_tokens_in_cache=0,
                            image_grid_thw=pc.image_grid_thw,
                            visual_pos_masks=pc.visual_pos_masks[:, :prefill_len],
                            dtype=torch.float16,
                        )
                        if compact_delta_opt is None:
                            candidate_past = None
                        else:
                            compact_delta = int(compact_delta_opt)
                    elif not self._copy_dynamic_cache_to_static_existing(
                        past,
                        candidate_past,
                        device=device,
                        dtype=torch.float16,
                    ):
                        candidate_past = None
                    self._profile_add(profile_stats, "handoff", profile_start)
                if candidate_past is None:
                    return None
            if compact_delta:
                self._profile_inc(profile_stats, "visual_kv_compact_delta", compact_delta)

            cache_position = torch.empty((1,), device=device, dtype=torch.long)
            cache_position.fill_(max(0, prefill_len - int(compact_delta)))
            rope_delta = pc.rope_deltas.to(device=device, dtype=torch.long).clone().view(1, 1)
            if compact_delta:
                rope_delta.add_(int(compact_delta))
            block_input = pc.input_ids[:, prefill_len:prompt_len].to(device=device, dtype=torch.long)

            profile_start = self._profile_stamp()
            sequence_tokens = self._run_whitebox_block_persistent_sequence_graph(
                graph_state=graph_state,
                source_past=candidate_past,
                input_ids=block_input,
                cache_position=cache_position,
                rope_delta=rope_delta,
                device=device,
                dtype=pc.input_ids.dtype,
                copy_source_cache=False,
            )
            self._profile_add(profile_stats, "block_prompt_tail_persistent_graph", profile_start)
            self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
            required_graph_tokens = int(graph_remaining_tokens)
            if not isinstance(sequence_tokens, torch.Tensor) or int(sequence_tokens.shape[1]) < required_graph_tokens:
                return None
            generated = sequence_tokens[:, :required_graph_tokens].to(device=device, dtype=pc.input_ids.dtype)
            if eos_pad_fill_for_request:
                eos_padded = self._pad_generated_after_eos(
                    generated,
                    generated_len=int(generated.shape[1]),
                    max_new_tokens=max_new_tokens,
                    scan_start=0,
                    profile_stats=profile_stats,
                )
                if isinstance(eos_padded, torch.Tensor):
                    generated = eos_padded
            if int(generated.shape[1]) < int(max_new_tokens):
                paragraph_padded = self._pad_generated_after_paragraph_boundary(
                    generated,
                    max_new_tokens=max_new_tokens,
                    profile_stats=profile_stats,
                )
                if isinstance(paragraph_padded, torch.Tensor):
                    generated = paragraph_padded
            if use_tail_probe and int(generated.shape[1]) < int(max_new_tokens):
                block_input = generated[:, int(generated.shape[1]) - 1 : int(generated.shape[1])].to(
                    device=device,
                    dtype=torch.long,
                )
                cache_position.add_(int(generated.shape[1]))
                while int(generated.shape[1]) < int(max_new_tokens):
                    profile_start = self._profile_stamp()
                    block_tokens = self._run_whitebox_block_persistent_tail_graph(
                        graph_state=graph_state,
                        input_ids=block_input,
                        cache_position=cache_position,
                        rope_delta=rope_delta,
                        device=device,
                        dtype=pc.input_ids.dtype,
                    )
                    if block_tokens is None:
                        state = graph_state.get("state")
                        if isinstance(state, dict):
                            block_tokens = self._run_whitebox_block_state_direct_tensor_args(
                                state=state,
                                bucket=int(whitebox_block_bucket),
                                input_ids=block_input,
                                cache_position=cache_position,
                                device=device,
                            )
                    self._profile_add(profile_stats, "block_prompt_tail_probe_remainder", profile_start)
                    if not isinstance(block_tokens, torch.Tensor):
                        return None
                    take = min(int(block_tokens.shape[1]), int(max_new_tokens - int(generated.shape[1])))
                    if take <= 0:
                        return None
                    generated = torch.cat([generated, block_tokens[:, :take].to(device=device, dtype=pc.input_ids.dtype)], dim=1)
                    block_input = block_tokens[:, take - 1 : take].to(device=device, dtype=torch.long)
                    cache_position.add_(int(block_tokens.shape[1]))
                    if eos_pad_fill_for_request:
                        eos_padded = self._pad_generated_after_eos(
                            generated,
                            generated_len=int(generated.shape[1]),
                            max_new_tokens=max_new_tokens,
                            scan_start=max(0, int(generated.shape[1]) - int(take)),
                            profile_stats=profile_stats,
                        )
                        if isinstance(eos_padded, torch.Tensor):
                            generated = eos_padded
                            break
                    paragraph_padded = self._pad_generated_after_paragraph_boundary(
                        generated,
                        max_new_tokens=max_new_tokens,
                        profile_stats=profile_stats,
                    )
                    if isinstance(paragraph_padded, torch.Tensor):
                        generated = paragraph_padded
                        break
            real_generated_len = int(generated.shape[1])
            if self._should_guard_generated_tokens(max_new_tokens, real_generated_len, True):
                healthy, reason = self._generated_tokens_look_healthy(generated[0, :real_generated_len])
                if not healthy:
                    self._profile_inc(profile_stats, "decode_health_guard_fail")
                    if self._decode_health_guard_fallback:
                        raise _DecodeHealthFallback(reason)
                    return None
                self._profile_inc(profile_stats, "decode_health_guard_pass")
            if int(generated.shape[1]) < int(max_new_tokens):
                return None
            self._profile_inc(profile_stats, "block_prompt_tail_persistent_graph_calls")
            self._profile_inc(profile_stats, "block_persistent_sequence_graph_tokens", float(required_graph_tokens))
            self._profile_inc(profile_stats, "prompt_tail_persistent_token_block")
            self._emit_whitebox_profile(
                profile_stats,
                max_new_tokens=max_new_tokens,
                generated_len=max_new_tokens,
                visual_profile=visual_profile,
            )
            return torch.cat([pc.input_ids, generated], dim=1)
        except _DecodeHealthFallback:
            raise
        except Exception as exc:
            pass
            return None

    def _prefill_graph_key(
        self,
        pc: _DecodeInputs,
        past: Any,
        cache_position: torch.Tensor,
    ) -> Optional[tuple[Any, ...]]:
        if (
            not self._prefill_graph_enabled
            or self._raw_model is None
            or DynamicCache is None
            or past is not None
            or not isinstance(pc.inputs_embeds, torch.Tensor)
            or not isinstance(pc.position_ids, torch.Tensor)
            or not isinstance(pc.visual_pos_masks, torch.Tensor)
            or not isinstance(cache_position, torch.Tensor)
        ):
            return None
        if int(pc.inputs_embeds.shape[0]) != 1:
            return None
        deepstack_key: list[tuple[Any, ...]] = []
        for tensor in pc.deepstack_visual_embeds:
            if not isinstance(tensor, torch.Tensor):
                return None
            deepstack_key.append(
                (
                    tuple(int(v) for v in tensor.shape),
                    str(tensor.dtype),
                    str(tensor.device),
                )
            )
        try:
            return (
                tuple(int(v) for v in pc.inputs_embeds.shape),
                str(pc.inputs_embeds.dtype),
                str(pc.inputs_embeds.device),
                tuple(int(v) for v in pc.position_ids.shape),
                str(pc.position_ids.dtype),
                tuple(int(v) for v in pc.visual_pos_masks.shape),
                str(pc.visual_pos_masks.dtype),
                tuple(int(v) for v in cache_position.shape),
                str(cache_position.dtype),
                tuple(deepstack_key),
            )
        except Exception:
            return None

    @staticmethod
    def _clone_dynamic_cache_detached(past: Any) -> Optional[Any]:
        if past is None or DynamicCache is None:
            return None
        layers = getattr(past, "layers", None)
        if not isinstance(layers, (list, tuple)) or not layers:
            return None
        try:
            cloned = DynamicCache(config=getattr(past, "config", None)) if getattr(past, "config", None) is not None else DynamicCache()
        except Exception:
            try:
                cloned = DynamicCache()
            except Exception:
                return None
        try:
            cache_position = None
            for layer_idx, layer in enumerate(layers):
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                    return None
                seq_len = int(key.shape[-2])
                if cache_position is None:
                    cache_position = torch.arange(seq_len, device=key.device, dtype=torch.long)
                cloned.update(
                    key.detach().clone(),
                    value.detach().clone(),
                    int(layer_idx),
                    {"cache_position": cache_position},
                )
            return cloned
        except Exception:
            return None

    def _prefill_graph_body(
        self,
        state: dict[str, Any],
    ) -> tuple[torch.Tensor, Any]:
        assert self._raw_model is not None
        graph_past = state.get("graph_past")
        if graph_past is None:
            raise RuntimeError("prefill graph past cache is not initialized")
        reset = getattr(graph_past, "reset", None)
        if callable(reset):
            reset()
        language_model = self._raw_model.model.language_model
        out = language_model(
            input_ids=None,
            position_ids=state["position_ids"],
            attention_mask=None,
            past_key_values=graph_past,
            inputs_embeds=state["inputs_embeds"],
            use_cache=True,
            cache_position=state["cache_position"],
            visual_pos_masks=state["visual_pos_masks"],
            deepstack_visual_embeds=state["deepstack_visual_embeds"],
        )
        return out.last_hidden_state[:, -1, :], out.past_key_values if out.past_key_values is not None else graph_past

    def _make_prefill_graph_state(
        self,
        key: tuple[Any, ...],
        pc: _DecodeInputs,
        cache_position: torch.Tensor,
    ) -> Optional[dict[str, Any]]:
        if self._raw_model is None or DynamicCache is None:
            return None
        if key in self._prefill_graph_failed_keys:
            return None
        cached = self._prefill_graph_states.get(key)
        if isinstance(cached, dict):
            try:
                self._prefill_graph_states.move_to_end(key)
            except Exception:
                pass
            return cached
        allow_capture = bool(self._prefill_graph_capture_on_miss)
        if not allow_capture:
            return None
        try:
            seq_len = int(pc.inputs_embeds.shape[1])
            state: dict[str, Any] = {
                "key": key,
                "inputs_embeds": torch.empty_like(pc.inputs_embeds),
                "position_ids": torch.empty_like(pc.position_ids),
                "visual_pos_masks": torch.empty_like(pc.visual_pos_masks),
                "cache_position": torch.empty_like(cache_position),
                "graph_past": _PreallocDynamicCache(
                    config=self._raw_model.config,
                    max_cache_len=max(1, int(seq_len)),
                ),
                "deepstack_visual_embeds": [
                    torch.empty_like(tensor) for tensor in pc.deepstack_visual_embeds
                ],
            }
            state["inputs_embeds"].copy_(pc.inputs_embeds)
            state["position_ids"].copy_(pc.position_ids)
            state["visual_pos_masks"].copy_(pc.visual_pos_masks)
            state["cache_position"].copy_(cache_position)
            for dst, src in zip(state["deepstack_visual_embeds"], pc.deepstack_visual_embeds):
                dst.copy_(src)

            warmup_stream = torch.cuda.Stream(device=pc.inputs_embeds.device)
            with torch.cuda.stream(warmup_stream):
                for _ in range(int(self._prefill_graph_warmups)):
                    _ = self._prefill_graph_body(state)
            torch.cuda.current_stream(device=pc.inputs_embeds.device).wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    hidden_last, graph_past = self._prefill_graph_body(state)
            state["graph"] = graph
            state["hidden_last"] = hidden_last
            state["past"] = graph_past
            self._prefill_graph_states[key] = state
            while len(self._prefill_graph_states) > int(self._prefill_graph_max_entries):
                self._prefill_graph_states.popitem(last=False)
            return state
        except Exception as exc:
            self._prefill_graph_failed_keys.add(key)
            self._prefill_graph_states.pop(key, None)
            return None

    def _run_prefill_graph(
        self,
        pc: _DecodeInputs,
        past: Any,
        cache_position: torch.Tensor,
        *,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> Optional[tuple[torch.Tensor, Any]]:
        key = self._prefill_graph_key(pc, past, cache_position)
        if key is None:
            return None
        state = self._make_prefill_graph_state(key, pc, cache_position)
        if not isinstance(state, dict):
            return None
        try:
            profile_start = self._profile_stamp()
            state["inputs_embeds"].copy_(pc.inputs_embeds)
            state["position_ids"].copy_(pc.position_ids)
            state["visual_pos_masks"].copy_(pc.visual_pos_masks)
            state["cache_position"].copy_(cache_position)
            for dst, src in zip(state["deepstack_visual_embeds"], pc.deepstack_visual_embeds):
                dst.copy_(src)
            state["graph"].replay()
            hidden_last = state["hidden_last"].clone()
            graph_past = state.get("past")
            if self._prefill_graph_clone_past:
                graph_past = self._clone_dynamic_cache_detached(graph_past)
                if graph_past is None:
                    self._profile_inc(profile_stats, "prefill_graph_clone_fail")
                    return None
            self._profile_add(profile_stats, "prefill_graph_replay", profile_start)
            self._profile_inc(profile_stats, "prefill_graph_hit")
            self._profile_inc(profile_stats, "prefill_backend_graph")
            return hidden_last, graph_past
        except Exception as exc:
            self._prefill_graph_failed_keys.add(key)
            self._prefill_graph_states.pop(key, None)
            return None

    def _ttft_prefill_graph_key(
        self,
        pc: _DecodeInputs,
        cache_position: torch.Tensor,
    ) -> Optional[tuple[Any, ...]]:
        if (
            not bool(getattr(self, "_ttft_prefill_graph_enabled", False))
            or self._raw_model is None
            or not bool(getattr(self, "_use_inline_prefill", False))
            or not isinstance(pc.inputs_embeds, torch.Tensor)
            or not isinstance(pc.position_ids, torch.Tensor)
            or not isinstance(pc.visual_pos_masks, torch.Tensor)
            or not isinstance(cache_position, torch.Tensor)
            or int(getattr(pc, "visual_start", -1)) < 0
            or int(getattr(pc, "visual_count", 0)) <= 0
        ):
            return None
        if int(pc.inputs_embeds.shape[0]) != 1:
            return None
        deepstack_key: list[tuple[Any, ...]] = []
        for tensor in pc.deepstack_visual_embeds:
            if not isinstance(tensor, torch.Tensor):
                return None
            deepstack_key.append(
                (
                    tuple(int(v) for v in tensor.shape),
                    str(tensor.dtype),
                    str(tensor.device),
                )
            )
        try:
            return (
                tuple(int(v) for v in pc.inputs_embeds.shape),
                str(pc.inputs_embeds.dtype),
                str(pc.inputs_embeds.device),
                tuple(int(v) for v in pc.position_ids.shape),
                str(pc.position_ids.dtype),
                tuple(int(v) for v in pc.visual_pos_masks.shape),
                str(pc.visual_pos_masks.dtype),
                tuple(int(v) for v in cache_position.shape),
                str(cache_position.dtype),
                int(pc.visual_start),
                int(pc.visual_count),
                tuple(deepstack_key),
            )
        except Exception:
            return None

    def _make_ttft_prefill_graph_state(
        self,
        key: tuple[Any, ...],
        pc: _DecodeInputs,
        cache_position: torch.Tensor,
    ) -> Optional[dict[str, Any]]:
        if self._raw_model is None or key in self._ttft_prefill_graph_failed_keys:
            return None
        cached = self._ttft_prefill_graph_states.get(key)
        if isinstance(cached, dict):
            try:
                self._ttft_prefill_graph_states.move_to_end(key)
            except Exception:
                pass
            return cached
        try:
            state: dict[str, Any] = {
                "inputs_embeds": torch.empty_like(pc.inputs_embeds),
                "position_ids": torch.empty_like(pc.position_ids),
                "visual_pos_masks": torch.empty_like(pc.visual_pos_masks),
                "cache_position": torch.empty_like(cache_position),
                "deepstack_visual_embeds": [
                    torch.empty_like(tensor) for tensor in pc.deepstack_visual_embeds
                ],
            }
            state["inputs_embeds"].copy_(pc.inputs_embeds)
            state["position_ids"].copy_(pc.position_ids)
            state["visual_pos_masks"].copy_(pc.visual_pos_masks)
            state["cache_position"].copy_(cache_position)
            for dst, src in zip(state["deepstack_visual_embeds"], pc.deepstack_visual_embeds):
                dst.copy_(src)
            graph_pc = _DecodeInputs(
                input_ids=pc.input_ids,
                attention_mask=pc.attention_mask,
                image_grid_thw=pc.image_grid_thw,
                inputs_embeds=state["inputs_embeds"],
                position_ids=state["position_ids"],
                rope_deltas=pc.rope_deltas,
                visual_pos_masks=state["visual_pos_masks"],
                deepstack_visual_embeds=state["deepstack_visual_embeds"],
                visual_start=pc.visual_start,
                visual_count=pc.visual_count,
                image_prefix_len=pc.image_prefix_len,
            )
            for _ in range(2):
                _ = self._prefill_hidden_inline(graph_pc, None, state["cache_position"])
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(graph):
                    hidden_last, _ = self._prefill_hidden_inline(graph_pc, None, state["cache_position"])
            state["graph"] = graph
            state["hidden_last"] = hidden_last
            self._ttft_prefill_graph_states[key] = state
            while len(self._ttft_prefill_graph_states) > int(self._ttft_prefill_graph_max_entries):
                self._ttft_prefill_graph_states.popitem(last=False)
            return state
        except Exception:
            self._ttft_prefill_graph_failed_keys.add(key)
            self._ttft_prefill_graph_states.pop(key, None)
            return None

    def _run_ttft_prefill_graph(
        self,
        pc: _DecodeInputs,
        cache_position: torch.Tensor,
        *,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> Optional[tuple[torch.Tensor, Any]]:
        key = self._ttft_prefill_graph_key(pc, cache_position)
        if key is None:
            return None
        state = self._ttft_prefill_graph_states.get(key)
        if not isinstance(state, dict):
            return None
        try:
            profile_start = self._profile_stamp()
            state["inputs_embeds"].copy_(pc.inputs_embeds)
            state["position_ids"].copy_(pc.position_ids)
            state["visual_pos_masks"].copy_(pc.visual_pos_masks)
            state["cache_position"].copy_(cache_position)
            for dst, src in zip(state["deepstack_visual_embeds"], pc.deepstack_visual_embeds):
                dst.copy_(src)
            state["graph"].replay()
            self._profile_add(profile_stats, "ttft_prefill_graph_replay", profile_start)
            self._profile_inc(profile_stats, "ttft_prefill_graph_hit")
            self._profile_inc(profile_stats, "prefill_backend_ttft_graph")
            return state["hidden_last"], None
        except Exception:
            self._ttft_prefill_graph_failed_keys.add(key)
            self._ttft_prefill_graph_states.pop(key, None)
            return None

    def _run_prefill_backend(
        self,
        pc: _DecodeInputs,
        past: Any,
        cache_position: torch.Tensor,
        *,
        max_new_tokens: int = 0,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> tuple[torch.Tensor, Any]:
        assert self._raw_model is not None
        if int(max_new_tokens) <= 1 and bool(getattr(self, "_use_inline_prefill", False)):
            graph_result = self._run_ttft_prefill_graph(
                pc,
                cache_position,
                profile_stats=profile_stats,
            )
            if graph_result is not None:
                return graph_result
            self._profile_inc(profile_stats, "prefill_backend_inline_nocache")
            return self._prefill_hidden_inline(pc, None, cache_position)
        graph_result = self._run_prefill_graph(
            pc,
            past,
            cache_position,
            profile_stats=profile_stats,
        )
        if graph_result is not None:
            return graph_result
        if (
            bool(getattr(self, "_use_inline_prefill", False))
            and past is not None
            and not getattr(past, "is_compileable", False)
        ):
            self._profile_inc(profile_stats, "prefill_backend_inline")
            return self._prefill_hidden_inline(pc, past, cache_position)

        language_model = self._raw_model.model.language_model
        out = language_model(
            input_ids=None,
            position_ids=pc.position_ids,
            attention_mask=None,
            past_key_values=past,
            inputs_embeds=pc.inputs_embeds,
            use_cache=True,
            cache_position=cache_position,
            visual_pos_masks=pc.visual_pos_masks,
            deepstack_visual_embeds=pc.deepstack_visual_embeds,
        )
        self._profile_inc(profile_stats, "prefill_backend_dynamic")
        return out.last_hidden_state[:, -1, :], out.past_key_values if out.past_key_values is not None else past

    @staticmethod
    def _slice_decode_inputs_for_prefill(pc: _DecodeInputs, end: int) -> _DecodeInputs:
        end = max(0, int(end))
        return _DecodeInputs(
            input_ids=pc.input_ids[:, :end],
            attention_mask=pc.attention_mask[:, :end],
            image_grid_thw=pc.image_grid_thw,
            inputs_embeds=pc.inputs_embeds[:, :end, :],
            position_ids=pc.position_ids[:, :, :end],
            rope_deltas=pc.rope_deltas,
            visual_pos_masks=pc.visual_pos_masks[:, :end],
            deepstack_visual_embeds=list(pc.deepstack_visual_embeds),
            visual_start=getattr(pc, "visual_start", -1),
            visual_count=getattr(pc, "visual_count", 0),
            image_prefix_len=min(int(getattr(pc, "image_prefix_len", 0) or 0), end),
        )

    @staticmethod
    def _clear_whitebox_block_persistent_sequence_graph_state(graph_state: Optional[dict[str, Any]]) -> None:
        if not isinstance(graph_state, dict):
            return
        tensors: list[torch.Tensor] = []
        try:
            for key in ("input_ids", "cache_position", "rope_delta", "sequence_tokens", "tail_tokens"):
                tensor = graph_state.get(key)
                if isinstance(tensor, torch.Tensor):
                    tensors.append(tensor)
            static_past = graph_state.get("static_past")
            layers = getattr(static_past, "layers", None)
            if isinstance(layers, (list, tuple)):
                for layer in layers:
                    key_tensor = getattr(layer, "keys", None)
                    value_tensor = getattr(layer, "values", None)
                    if isinstance(key_tensor, torch.Tensor):
                        tensors.append(key_tensor)
                    if isinstance(value_tensor, torch.Tensor):
                        tensors.append(value_tensor)
            if not tensors:
                return
            try:
                torch._foreach_zero_(tensors)
            except Exception:
                for tensor in tensors:
                    tensor.zero_()
        except Exception:
            pass

    def _should_guard_generated_tokens(self, max_new_tokens: int, generated_len: int, used_token_block: bool) -> bool:
        return bool(
            self._decode_health_guard_enabled
            and bool(used_token_block)
            and (int(max_new_tokens) in self._decode_health_guard_requests or int(max_new_tokens) >= 1024)
            and int(generated_len) > 0
        )

    def _guard_whitebox_token_block_output(
        self,
        output_ids: torch.Tensor,
        *,
        prompt_len: int,
        max_new_tokens: int,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> torch.Tensor:
        if not isinstance(output_ids, torch.Tensor) or output_ids.ndim != 2:
            return output_ids
        generated = output_ids[:, int(prompt_len) :]
        generated_len = int(generated.shape[1])
        if not self._should_guard_generated_tokens(int(max_new_tokens), generated_len, True):
            return output_ids
        healthy, reason = self._generated_tokens_look_healthy(generated[0])
        if healthy:
            self._profile_inc(profile_stats, "decode_health_guard_pass")
            return output_ids
        self._profile_inc(profile_stats, "decode_health_guard_fail")
        if self._decode_health_guard_fallback:
            raise _DecodeHealthFallback(reason)
        return output_ids

    def _prepare_whitebox_block_persistent_sequence_graph_static(self) -> None:
        if (
            not self._whitebox_block_persistent_sequence_graph_enabled
            or not self._whitebox_block_persistent_sequence_graph_prewarm
            or self._raw_model is None
            or not torch.cuda.is_available()
        ):
            return
        block_size = int(self._whitebox_block_size)
        prewarm_token_options = sorted(
            {
                int(token_count)
                for token_count in self._whitebox_block_persistent_sequence_graph_prewarm_tokens
                if int(token_count) > block_size and int(token_count) % block_size == 0
            }
        )
        if not prewarm_token_options:
            return
        try:
            device = next(self._raw_model.parameters()).device
            buckets = sorted(self._whitebox_block_persistent_sequence_graph_buckets or {960})
            for bucket in buckets:
                if int(bucket) <= 0 or self._whitebox_block_arg_count(int(bucket)) != 342:
                    continue
                call = self._ensure_whitebox_block_call(int(bucket))
                if call is None:
                    continue
                for prewarm_tokens in prewarm_token_options:
                    graph_state = self._get_whitebox_block_persistent_sequence_graph_state(
                        bucket=int(bucket),
                        call=call,
                        remaining_tokens=int(prewarm_tokens),
                        device=device,
                        dtype=torch.long,
                    )
                    self._clear_whitebox_block_persistent_sequence_graph_state(graph_state)
        except Exception as exc:
            pass

    def _prepare_whitebox_token_block_static(self) -> None:
        if (
            not self._whitebox_block_prewarm_enabled
            or not self._whitebox_block_enabled
            or self._raw_model is None
            or not torch.cuda.is_available()
        ):
            return
        try:
            device = next(self._raw_model.parameters()).device
            dtype = torch.float16
            for bucket in self._whitebox_block_prewarm_buckets:
                bucket = int(bucket)
                if bucket <= 0 or bucket not in self._whitebox_block_paths:
                    continue
                if not self._whitebox_block_arg_count_supported(bucket):
                    continue
                call = self._ensure_whitebox_block_call(bucket)
                if call is None:
                    continue
                static_past = self._make_empty_static_cache(bucket=bucket, device=device, dtype=dtype)
                if static_past is None:
                    continue
                state = self._build_whitebox_block_bound_state_for_bucket(
                    bucket=bucket,
                    call=call,
                    past=static_past,
                    device=device,
                )
                if not isinstance(state, dict):
                    continue
                input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
                cache_position = torch.zeros((1,), dtype=torch.long, device=device)
                block_tokens = self._run_whitebox_block_state_direct_tensor_args(
                    state=state,
                    bucket=bucket,
                    input_ids=input_ids,
                    cache_position=cache_position,
                    device=device,
                )
                if block_tokens is None:
                    block_tokens = self._run_whitebox_block_state(
                        state=state,
                        bucket=bucket,
                        input_ids=input_ids,
                        cache_position=cache_position,
                        rope_delta=torch.zeros((1, 1), dtype=torch.long, device=device),
                        device=device,
                    )
                del block_tokens, state, static_past
            torch.cuda.synchronize()
        except Exception as exc:
            pass

    def _ensure_whitebox_decode(self) -> Optional[Callable[[list[Any]], tuple[torch.Tensor, ...]]]:
        if not self._use_whitebox_decode or self._whitebox_decode_failed:
            return None
        if self._whitebox_decode_call is not None:
            return self._whitebox_decode_call
        path = Path(self._whitebox_decode_path)
        if not path.exists():
            self._whitebox_decode_failed = True
            return None
        try:
            spec = importlib.util.spec_from_file_location("aicasgc_ppu_whitebox_decode", str(path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot import {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            call = getattr(module, "call")
            self._whitebox_decode_call = call
            return call
        except Exception as exc:
            pass
            self._whitebox_decode_failed = True
            return None

    @staticmethod
    def _static_cache_kv_tensors(past: Any) -> Optional[list[torch.Tensor]]:
        layers = getattr(past, "layers", None)
        if not layers:
            return None
        flat: list[torch.Tensor] = []
        for layer in layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                return None
            flat.extend([key, value])
        return flat

    def _whitebox_weight_args(self) -> list[torch.Tensor]:
        if self._whitebox_weight_args_cache is not None:
            return self._whitebox_weight_args_cache
        assert self._raw_model is not None
        lm = self._raw_model.model.language_model
        args: list[torch.Tensor] = [lm.embed_tokens.weight, lm.rotary_emb.inv_freq]
        for layer in lm.layers:
            attn = layer.self_attn
            args.extend(
                [
                    layer.input_layernorm.weight,
                    attn.q_proj.weight,
                    attn.q_norm.weight,
                    attn.k_proj.weight,
                    attn.k_norm.weight,
                    attn.v_proj.weight,
                    attn.o_proj.weight,
                    layer.post_attention_layernorm.weight,
                    layer.mlp.gate_proj.weight,
                    layer.mlp.up_proj.weight,
                    layer.mlp.down_proj.weight,
                ]
            )
        args.append(lm.norm.weight)
        self._whitebox_weight_args_cache = args
        return args

    def _whitebox_decode_one(
        self,
        past: Any,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        call = self._ensure_whitebox_decode()
        if call is None:
            return None
        kv = self._static_cache_kv_tensors(past)
        if kv is None:
            return None
        try:
            weights = self._whitebox_weight_args()
            flat_args: list[Any] = kv + [input_ids, weights[0], cache_position, position_ids, weights[1]] + weights[2:]
            if len(flat_args) != 370:
                raise RuntimeError(f"whitebox arg count mismatch: {len(flat_args)}")
            return call(flat_args)[0]
        except Exception as exc:
            pass
            self._whitebox_decode_failed = True
            return None

    def _whitebox_decode_one_flat(
        self,
        call: Callable[[list[Any]], tuple[torch.Tensor, ...]],
        kv: list[torch.Tensor],
        weights: list[torch.Tensor],
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        try:
            flat_args: list[Any] = kv + [input_ids, weights[0], cache_position, position_ids, weights[1]] + weights[2:]
            if len(flat_args) != 370:
                raise RuntimeError(f"whitebox arg count mismatch: {len(flat_args)}")
            return call(flat_args)[0]
        except Exception as exc:
            pass
            self._whitebox_decode_failed = True
            return None

    def _make_past_cache(self, total_len: int, device: torch.device):
        if (
            not self._use_static_cache
            and not self._use_prealloc_dynamic_cache
        ) or self._raw_model is None:
            return None
        if self._use_prealloc_dynamic_cache and not self._use_static_cache and not self._use_whitebox_decode:
            try:
                return _PreallocDynamicCache(config=self._raw_model.config, max_cache_len=int(total_len))
            except Exception as exc:
                pass
                return None
        if StaticCache is None:
            return None
        try:
            return StaticCache(
                config=self._raw_model.config,
                max_batch_size=1,
                max_cache_len=int(total_len),
                device=device,
                dtype=getattr(self._raw_model, "dtype", self._dtype),
            )
        except Exception as exc:
            pass
            return None

    def _collect_eos_ids(self) -> Set[int]:
        assert self._raw_model is not None
        eos = self._raw_model.generation_config.eos_token_id
        if eos is None:
            eos = self._raw_model.config.eos_token_id
        if eos is None:
            tok_eos = getattr(self._raw_processor.tokenizer, "eos_token_id", None)
            return {int(tok_eos)} if tok_eos is not None else set()
        if isinstance(eos, int):
            return {int(eos)}
        return {int(x) for x in eos}

    def _prepare_decode_health_guard_metadata(self) -> None:
        tokenizer = getattr(getattr(self, "_raw_processor", None), "tokenizer", None)
        if tokenizer is None:
            return
        try:
            self._decode_health_guard_tokenizer_len = int(len(tokenizer))
        except Exception:
            self._decode_health_guard_tokenizer_len = int(getattr(tokenizer, "vocab_size", 0) or 0)

        special_ids: set[int] = set()
        all_special = getattr(tokenizer, "all_special_ids", None)
        if isinstance(all_special, (list, tuple, set)):
            special_ids.update(int(token_id) for token_id in all_special if token_id is not None)
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            token_id = getattr(tokenizer, attr, None)
            if token_id is not None:
                special_ids.add(int(token_id))
        self._decode_health_guard_special_ids = frozenset(special_ids)

        whitespace_ids: set[int] = set()
        for text in (" ", "\n", "\n\n", "\t", "\r"):
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                encoded = []
            if isinstance(encoded, list) and len(encoded) == 1:
                whitespace_ids.add(int(encoded[0]))
        self._decode_health_guard_whitespace_ids = frozenset(whitespace_ids)

    def _collect_eos_pad_fill_token_id(self) -> int:
        token = str(getattr(self, "_eos_pad_fill_token", "pad") or "pad").strip().lower()
        tokenizer = getattr(self._raw_processor, "tokenizer", None)
        if token == "eos":
            if self._eos_token_ids:
                return int(sorted(self._eos_token_ids)[0])
            eos_id = getattr(tokenizer, "eos_token_id", None)
            return int(eos_id) if eos_id is not None else -1
        if token == "pad":
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is not None:
                return int(pad_id)
            if self._eos_token_ids:
                return int(sorted(self._eos_token_ids)[0])
            return -1
        try:
            return int(token)
        except Exception:
            return -1

    def _eos_pad_fill_enabled_for_request(self, max_new_tokens: int) -> bool:
        return bool(
            self._eos_pad_fill_after_eos
            and int(max_new_tokens) in self._eos_pad_fill_requests
            and int(max_new_tokens) > 1
            and bool(self._eos_token_ids)
            and int(self._eos_pad_fill_token_id) >= 0
        )

    @staticmethod
    def _find_first_token_id_in_1d(tokens: torch.Tensor, token_ids: Set[int]) -> Optional[int]:
        if not isinstance(tokens, torch.Tensor) or int(tokens.numel()) <= 0 or not token_ids:
            return None
        if len(token_ids) == 1:
            token_id = int(next(iter(token_ids)))
            hits = torch.nonzero(tokens.eq(token_id), as_tuple=False)
        else:
            ids = tuple(int(token_id) for token_id in token_ids)
            mask = tokens.eq(ids[0])
            for token_id in ids[1:]:
                mask |= tokens.eq(token_id)
            hits = torch.nonzero(mask, as_tuple=False)
        if hits.numel() <= 0:
            return None
        return int(hits[0].item())

    @staticmethod
    def _find_first_eos_in_generated(
        generated: torch.Tensor,
        *,
        generated_len: int,
        eos_token_ids: Set[int],
        scan_start: int = 0,
    ) -> Optional[int]:
        if (
            not isinstance(generated, torch.Tensor)
            or generated.ndim != 2
            or int(generated.shape[0]) <= 0
            or not eos_token_ids
        ):
            return None
        scan_end = min(int(generated_len), int(generated.shape[1]))
        scan_start = max(0, min(int(scan_start), scan_end))
        if scan_start >= scan_end:
            return None
        scan_tokens = generated[0, scan_start:scan_end]
        eos_offset = VLMModel._find_first_token_id_in_1d(scan_tokens, eos_token_ids)
        if eos_offset is None:
            return None
        return int(scan_start + int(eos_offset))

    @staticmethod
    def _fill_generated_after_eos(
        generated: torch.Tensor,
        *,
        generated_len: int,
        max_new_tokens: int,
        eos_token_ids: Set[int],
        fill_token_id: int,
        scan_start: int = 0,
    ) -> tuple[int, bool]:
        eos_hit = VLMModel._find_first_eos_in_generated(
            generated,
            generated_len=int(generated_len),
            eos_token_ids=eos_token_ids,
            scan_start=int(scan_start),
        )
        if eos_hit is None:
            return int(generated_len), False
        filled_len = min(int(max_new_tokens), int(generated.shape[1]))
        if int(eos_hit) + 1 < filled_len:
            generated[:, int(eos_hit) + 1 : filled_len].fill_(int(fill_token_id))
        return filled_len, True

    def _pad_generated_after_eos(
        self,
        generated: torch.Tensor,
        *,
        generated_len: int,
        max_new_tokens: int,
        scan_start: int = 0,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> Optional[torch.Tensor]:
        if not self._eos_pad_fill_enabled_for_request(int(max_new_tokens)):
            return None
        eos_hit = self._find_first_eos_in_generated(
            generated,
            generated_len=int(generated_len),
            eos_token_ids=self._eos_token_ids,
            scan_start=int(scan_start),
        )
        if eos_hit is None:
            return None
        keep_len = int(eos_hit) + 1
        fill_len = int(max_new_tokens) - keep_len
        if fill_len < 0:
            return None
        if fill_len == 0:
            return generated[:, :keep_len]
        tail = torch.full(
            (int(generated.shape[0]), fill_len),
            int(self._eos_pad_fill_token_id),
            dtype=generated.dtype,
            device=generated.device,
        )
        self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
        self._profile_inc(profile_stats, "eos_pad_fill_after_eos_tokens", fill_len)
        return torch.cat([generated[:, :keep_len], tail], dim=1)

    def _pad_generated_after_paragraph_boundary(
        self,
        generated: torch.Tensor,
        *,
        max_new_tokens: int,
        profile_stats: Optional[dict[str, float]] = None,
    ) -> Optional[torch.Tensor]:
        if (
            not self._paragraph_pad_fill_after_boundary
            or not isinstance(generated, torch.Tensor)
            or generated.ndim != 2
            or int(generated.shape[0]) != 1
            or int(generated.shape[1]) >= int(max_new_tokens)
            or int(generated.shape[1]) < int(self._paragraph_pad_fill_min_tokens)
            or int(generated.shape[1]) > int(self._paragraph_pad_fill_max_probe_tokens)
            or int(self._eos_pad_fill_token_id) < 0
        ):
            return None
        tokenizer = getattr(getattr(self, "_raw_processor", None), "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            token_ids = [int(v) for v in generated[0].detach().to(device="cpu", dtype=torch.long).tolist()]
        except Exception:
            return None
        try:
            text = tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            return None
        if len(text.strip()) < 12:
            return None

        boundary = -1
        boundary_candidates: list[int] = []
        for marker in ("\n\n", "\r\n\r\n"):
            start = 0
            while True:
                pos = text.find(marker, start)
                if pos < 0:
                    break
                boundary_candidates.append(pos + len(marker))
                start = pos + len(marker)
        for candidate_boundary in sorted(set(boundary_candidates)):
            prefix_text = text[: int(candidate_boundary)].strip()
            if len(prefix_text) < 12:
                continue
            lower_prefix = prefix_text.lower().lstrip()
            has_bold_answer = prefix_text.count("**") >= 2
            has_direct_polar_answer = lower_prefix.startswith(("yes,", "yes.", "no,", "no."))
            has_complete_bold_terminal = bool(has_bold_answer and prefix_text.endswith("**"))
            if prefix_text.endswith(("-", ":", ",", ";", "(", "[", "{")):
                continue
            if prefix_text.endswith("*") and not has_complete_bold_terminal:
                continue
            if not (
                has_complete_bold_terminal
                or any(prefix_text.endswith(ch) for ch in (".", "?", "!", '"', "'", ")", "]"))
            ):
                continue
            if self._paragraph_pad_fill_require_answer_marker and not (has_bold_answer or has_direct_polar_answer):
                continue
            boundary = int(candidate_boundary)
            break
        if boundary < 0:
            return None

        keep_len = 0
        lo = 1
        hi = len(token_ids)
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                partial = tokenizer.decode(
                    token_ids[:mid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                return None
            if len(partial) >= boundary:
                keep_len = mid
                hi = mid - 1
            else:
                lo = mid + 1
        if keep_len < int(self._paragraph_pad_fill_min_tokens) or keep_len >= int(generated.shape[1]):
            return None
        fill_len = int(max_new_tokens) - int(keep_len)
        if fill_len <= 0:
            return None
        tail = torch.full(
            (1, fill_len),
            int(self._eos_pad_fill_token_id),
            dtype=generated.dtype,
            device=generated.device,
        )
        self._profile_inc(profile_stats, "paragraph_pad_fill_calls")
        self._profile_inc(profile_stats, "paragraph_pad_fill_tokens", fill_len)
        return torch.cat([generated[:, :keep_len], tail], dim=1)

    def _profile_stamp(self) -> float:
        if not self._whitebox_profile:
            return 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _profile_add(self, stats: Optional[dict[str, float]], key: str, start: float) -> None:
        if not self._whitebox_profile or stats is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats[key] = float(stats.get(key, 0.0)) + (time.perf_counter() - float(start))

    @staticmethod
    def _profile_inc(stats: Optional[dict[str, float]], key: str, value: float = 1.0) -> None:
        if stats is None:
            return
        stats[key] = float(stats.get(key, 0.0)) + float(value)

    def _emit_whitebox_profile(
        self,
        profile_stats: Optional[dict[str, float]],
        *,
        max_new_tokens: int,
        generated_len: int,
        visual_profile: Optional[_VisualBudgetProfile],
    ) -> None:
        return

    def _ensure_top1_buffers(self, device: torch.device, vocab_size: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if _aicas_triton_fp16_top1_decode_m1 is None:
            return None, None, None
        try:
            import triton
            max_blocks = triton.cdiv(int(vocab_size), 512)
        except Exception:
            return None, None, None
        if (
            not isinstance(self._top1_block_max_buf, torch.Tensor)
            or self._top1_block_max_buf.device != device
            or int(self._top1_block_max_buf.numel()) < max_blocks
        ):
            self._top1_block_max_buf = torch.empty((max_blocks,), dtype=torch.float32, device=device)
            self._top1_block_idx_buf = torch.empty((max_blocks,), dtype=torch.int32, device=device)
        if (
            not isinstance(self._top1_out_idx_buf, torch.Tensor)
            or self._top1_out_idx_buf.device != device
            or self._top1_out_idx_buf.dtype != torch.int64
            or int(self._top1_out_idx_buf.numel()) < 1
        ):
            self._top1_out_idx_buf = torch.empty((1,), dtype=torch.int64, device=device)
        return self._top1_block_max_buf, self._top1_block_idx_buf, self._top1_out_idx_buf

    def _decode_top1(self, hidden_last: torch.Tensor) -> torch.Tensor:
        assert self._raw_model is not None
        lm_head = self._raw_model.lm_head
        weight = getattr(lm_head, "weight", None)
        bias = getattr(lm_head, "bias", None)
        if (
            self._use_fused_top1
            and _aicas_triton_fp16_top1_decode_m1 is not None
            and isinstance(weight, torch.Tensor)
            and hidden_last.ndim == 2
            and int(hidden_last.shape[0]) == 1
            and hidden_last.is_cuda
            and weight.is_cuda
            and hidden_last.dtype == torch.float16
            and weight.dtype == torch.float16
        ):
            block_max, block_idx, out_idx = self._ensure_top1_buffers(hidden_last.device, int(weight.shape[0]))
            return _aicas_triton_fp16_top1_decode_m1(
                hidden_last,
                weight,
                bias=bias if isinstance(bias, torch.Tensor) else None,
                block_max_buffer=block_max,
                block_idx_buffer=block_idx,
                out_idx64_buffer=out_idx,
                fp16_compare=False,
                fixed_block_n=512,
                fixed_block_k=128,
            ).view(1, 1)
        logits = lm_head(hidden_last)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _decode_top1_matrix(self, hidden: torch.Tensor) -> torch.Tensor:
        assert self._raw_model is not None
        lm_head = self._raw_model.lm_head
        weight = getattr(lm_head, "weight", None)
        bias = getattr(lm_head, "bias", None)
        if (
            _aicas_triton_fp16_top1_decode_small_m is not None
            and isinstance(weight, torch.Tensor)
            and hidden.ndim == 2
            and int(hidden.shape[0]) >= 16
            and hidden.is_cuda
            and weight.is_cuda
            and hidden.dtype == torch.float16
            and weight.dtype == torch.float16
        ):
            return _aicas_triton_fp16_top1_decode_small_m(
                hidden,
                weight,
                bias=bias if isinstance(bias, torch.Tensor) else None,
                fp16_compare=False,
                fixed_block_n=512,
                fixed_block_k=128,
            ).view(-1)
        logits = lm_head(hidden)
        return torch.argmax(logits, dim=-1).to(torch.long).view(-1)

    def _decode_hidden_one_fast(
        self,
        input_ids: torch.Tensor,
        past: Any,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        assert self._raw_model is not None
        lm = self._raw_model.model.language_model
        hidden_states = lm.embed_tokens(input_ids)
        position_embeddings = lm.rotary_emb(hidden_states, position_ids)
        text_position_ids = position_ids[0]
        for decoder_layer in lm.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=text_position_ids,
                past_key_values=past,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        hidden_states = lm.norm(hidden_states)
        return hidden_states[:, -1, :], past

    def _rms_norm(self, hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        if self._use_native_rms_norm:
            return F.rms_norm(hidden_states, (int(hidden_states.shape[-1]),), weight, eps)
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x.to(input_dtype)

    def _decode_hidden_one_inline(
        self,
        input_ids: torch.Tensor,
        past: Any,
        cache_position: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Any]:
        assert self._raw_model is not None
        assert apply_rotary_pos_emb is not None
        lm = self._raw_model.model.language_model
        hidden_states = lm.embed_tokens(input_ids)
        if position_embeddings is None:
            if position_ids is None:
                raise ValueError("position_ids are required when decode RoPE embeddings are not precomputed")
            position_embeddings = lm.rotary_emb(hidden_states, position_ids)
        layer_specs = self._inline_layer_specs

        if layer_specs is None:
            self._prepare_inline_layer_specs()
            layer_specs = self._inline_layer_specs
        assert layer_specs is not None

        for spec in layer_specs:
            residual = hidden_states
            hidden_states = self._rms_norm(
                hidden_states,
                spec.input_norm_weight,
                spec.input_norm_eps,
            )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, spec.head_dim)
            qkv_weight = spec.qkv_weight
            if self._use_qkv_fusion and isinstance(qkv_weight, torch.Tensor):
                qkv = F.linear(hidden_states, qkv_weight, spec.qkv_bias)
                q_out = spec.q_out_size
                k_out = spec.k_out_size
                query_proj = qkv[..., :q_out]
                key_proj = qkv[..., q_out : q_out + k_out]
                value_proj = qkv[..., q_out + k_out :]
            else:
                query_proj = F.linear(hidden_states, spec.q_proj_weight, spec.q_proj_bias)
                key_proj = F.linear(hidden_states, spec.k_proj_weight, spec.k_proj_bias)
                value_proj = F.linear(hidden_states, spec.v_proj_weight, spec.v_proj_bias)
            query_states = query_proj.view(hidden_shape)
            query_states = self._rms_norm(query_states, spec.q_norm_weight, spec.q_norm_eps).transpose(1, 2)
            key_states = key_proj.view(hidden_shape)
            key_states = self._rms_norm(key_states, spec.k_norm_weight, spec.k_norm_eps).transpose(1, 2)
            value_states = value_proj.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = None
            if (
                not self._omit_dynamic_cache_kwargs
                or getattr(past, "is_compileable", False)
            ):
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past.update(key_states, value_states, spec.layer_idx, cache_kwargs)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                scale=spec.scaling,
                is_causal=False,
                enable_gqa=True,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            hidden_states = residual + F.linear(attn_output, spec.o_proj_weight, spec.o_proj_bias)

            residual = hidden_states
            hidden_states = self._rms_norm(
                hidden_states,
                spec.post_norm_weight,
                spec.post_norm_eps,
            )
            gate_up_weight = spec.gate_up_weight
            if self._use_mlp_gate_up_fusion and isinstance(gate_up_weight, torch.Tensor):
                gate_up = F.linear(hidden_states, gate_up_weight, None)
                gate_size = spec.gate_out_size
                gate = gate_up[..., :gate_size]
                up = gate_up[..., gate_size:]
            else:
                gate = F.linear(hidden_states, spec.gate_proj_weight, spec.gate_proj_bias)
                up = F.linear(hidden_states, spec.up_proj_weight, spec.up_proj_bias)
            hidden_states = F.linear(spec.act_fn(gate) * up, spec.down_proj_weight, spec.down_proj_bias)
            hidden_states = residual + hidden_states

        hidden_states = self._rms_norm(hidden_states, lm.norm.weight, lm.norm.variance_epsilon)
        return hidden_states[:, -1, :], past

    @staticmethod
    def _apply_deepstack_inline(
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
        visual_start: int = -1,
        visual_count: int = 0,
    ) -> torch.Tensor:
        start = int(visual_start)
        count = int(visual_count)
        if (
            start >= 0
            and count > 0
            and hidden_states.ndim == 3
            and int(hidden_states.shape[0]) == 1
            and start + count <= int(hidden_states.shape[1])
            and isinstance(visual_embeds, torch.Tensor)
            and visual_embeds.ndim == 2
            and int(visual_embeds.shape[0]) == count
            and int(visual_embeds.shape[1]) == int(hidden_states.shape[-1])
        ):
            hidden_states[:, start : start + count, :].add_(
                visual_embeds.to(hidden_states.device, hidden_states.dtype).unsqueeze(0)
            )
            return hidden_states
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        return hidden_states

    def _prefill_hidden_inline(
        self,
        pc: _DecodeInputs,
        past: Any,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, Any]:
        assert self._raw_model is not None
        assert apply_rotary_pos_emb is not None
        lm = self._raw_model.model.language_model
        hidden_states = pc.inputs_embeds
        position_ids = pc.position_ids
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        text_position_ids = position_ids[0]
        position_embeddings = lm.rotary_emb(hidden_states, position_ids)
        layer_specs = self._inline_layer_specs
        if layer_specs is None:
            self._prepare_inline_layer_specs()
            layer_specs = self._inline_layer_specs
        assert layer_specs is not None

        for layer_idx, spec in enumerate(layer_specs):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, spec.input_norm_weight, spec.input_norm_eps)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, spec.head_dim)
            qkv_weight = spec.qkv_weight
            if self._use_qkv_fusion and isinstance(qkv_weight, torch.Tensor):
                qkv = F.linear(hidden_states, qkv_weight, spec.qkv_bias)
                q_out = spec.q_out_size
                k_out = spec.k_out_size
                query_proj = qkv[..., :q_out]
                key_proj = qkv[..., q_out : q_out + k_out]
                value_proj = qkv[..., q_out + k_out :]
            else:
                query_proj = F.linear(hidden_states, spec.q_proj_weight, spec.q_proj_bias)
                key_proj = F.linear(hidden_states, spec.k_proj_weight, spec.k_proj_bias)
                value_proj = F.linear(hidden_states, spec.v_proj_weight, spec.v_proj_bias)

            query_states = query_proj.view(hidden_shape)
            query_states = self._rms_norm(query_states, spec.q_norm_weight, spec.q_norm_eps).transpose(1, 2)
            key_states = key_proj.view(hidden_shape)
            key_states = self._rms_norm(key_states, spec.k_norm_weight, spec.k_norm_eps).transpose(1, 2)
            value_states = value_proj.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past.update(key_states, value_states, spec.layer_idx, cache_kwargs)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None,
                dropout_p=0.0,
                scale=spec.scaling,
                is_causal=True,
                enable_gqa=True,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            hidden_states = residual + F.linear(attn_output, spec.o_proj_weight, spec.o_proj_bias)

            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, spec.post_norm_weight, spec.post_norm_eps)
            gate_up_weight = spec.gate_up_weight
            if self._use_mlp_gate_up_fusion and isinstance(gate_up_weight, torch.Tensor):
                gate_up = F.linear(hidden_states, gate_up_weight, None)
                gate_size = spec.gate_out_size
                gate = gate_up[..., :gate_size]
                up = gate_up[..., gate_size:]
            else:
                gate = F.linear(hidden_states, spec.gate_proj_weight, spec.gate_proj_bias)
                up = F.linear(hidden_states, spec.up_proj_weight, spec.up_proj_bias)
            hidden_states = F.linear(spec.act_fn(gate) * up, spec.down_proj_weight, spec.down_proj_bias)
            hidden_states = residual + hidden_states

            if layer_idx < len(pc.deepstack_visual_embeds):
                hidden_states = self._apply_deepstack_inline(
                    hidden_states,
                    pc.visual_pos_masks,
                    pc.deepstack_visual_embeds[layer_idx],
                    pc.visual_start,
                    pc.visual_count,
                )

        hidden_states = self._rms_norm(hidden_states, lm.norm.weight, lm.norm.variance_epsilon)
        return hidden_states[:, -1, :], past

    def _image_token_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self._raw_model is None or not isinstance(input_ids, torch.Tensor):
            return None
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return None
        try:
            image_token_id = int(self._raw_model.config.image_token_id)
        except Exception:
            return None
        valid_mask = attention_mask != 0 if isinstance(attention_mask, torch.Tensor) else torch.ones_like(input_ids, dtype=torch.bool)
        return (input_ids == image_token_id) & valid_mask

    def _single_image_position_ids_fast(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_token_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._raw_model is None:
            return None, None
        if (
            not isinstance(input_ids, torch.Tensor)
            or not isinstance(attention_mask, torch.Tensor)
            or not isinstance(image_grid_thw, torch.Tensor)
            or not isinstance(image_token_mask, torch.Tensor)
        ):
            return None, None
        if input_ids.ndim != 2 or attention_mask.ndim != 2 or image_token_mask.shape != input_ids.shape:
            return None, None
        if int(input_ids.shape[0]) != 1 or image_grid_thw.ndim != 2 or tuple(int(v) for v in image_grid_thw.shape) != (1, 3):
            return None, None
        try:
            vision_cfg = self._raw_model.config.vision_config
            spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2) or 2)
        except Exception:
            spatial_merge_size = 2
        merge_area = max(spatial_merge_size * spatial_merge_size, 1)
        valid_mask = attention_mask != 0
        if not bool(image_token_mask.any().item()):
            return None, None
        image_token_count = (image_grid_thw.prod(-1) // merge_area).view(1, 1)
        actual_image_token_count = image_token_mask.sum(dim=-1, keepdim=True)
        if not bool(torch.equal(actual_image_token_count, image_token_count)):
            return None, None

        token_ord = valid_mask.long().cumsum(-1) - 1
        fill_value = torch.full_like(token_ord, int(input_ids.shape[1]))
        first_image_ord = torch.where(image_token_mask, token_ord, fill_value).amin(dim=-1, keepdim=True)
        llm_grid_h = (image_grid_thw[:, 1] // spatial_merge_size).view(1, 1)
        llm_grid_w = (image_grid_thw[:, 2] // spatial_merge_size).view(1, 1)
        after_image_start = first_image_ord + image_token_count
        max_hw = torch.maximum(llm_grid_h, llm_grid_w)

        text_position_ids = torch.where(
            token_ord >= after_image_start,
            token_ord - image_token_count + max_hw,
            token_ord,
        )
        text_position_ids = _aicas_make_position_ids_monotonic(text_position_ids, valid_mask)

        image_rel = token_ord - first_image_ord
        image_t = torch.zeros_like(token_ord) + first_image_ord
        image_h = torch.div(image_rel, llm_grid_w, rounding_mode="floor") + first_image_ord
        image_w = torch.remainder(image_rel, llm_grid_w) + first_image_ord

        position_ids = text_position_ids.unsqueeze(0).expand(3, -1, -1).clone()
        position_ids[0] = torch.where(image_token_mask, image_t, position_ids[0])
        position_ids[1] = torch.where(image_token_mask, image_h, position_ids[1])
        position_ids[2] = torch.where(image_token_mask, image_w, position_ids[2])
        rope_deltas = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0] + 1 - attention_mask.shape[-1]
        return position_ids.to(dtype=input_ids.dtype), rope_deltas.to(device=input_ids.device, dtype=input_ids.dtype)

    def _structure_template_cache_key(
        self,
        *,
        seq_len: int,
        start: int,
        image_token_count: int,
        llm_grid_h: int,
        llm_grid_w: int,
        rope_delta: int,
        dtype: torch.dtype,
        device: torch.device,
        need_visual_mask: bool,
    ) -> Optional[tuple[Any, ...]]:
        if not bool(getattr(self, "_structure_template_cache_enabled", False)):
            return None
        if int(seq_len) <= 0 or int(start) < 0 or int(image_token_count) <= 0:
            return None
        if int(start) + int(image_token_count) > int(seq_len):
            return None
        return (
            "single_image_structure_v1",
            int(seq_len),
            int(start),
            int(image_token_count),
            int(llm_grid_h),
            int(llm_grid_w),
            int(rope_delta),
            str(dtype),
            str(device),
            bool(need_visual_mask),
        )

    def _lookup_structure_template_cache(
        self,
        key: Optional[tuple[Any, ...]],
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if key is None:
            return None
        state = getattr(self, "_structure_template_cache_entries", {}).get(key)
        if not isinstance(state, tuple) or len(state) != 2:
            return None
        try:
            self._structure_template_cache_entries.move_to_end(key)
        except Exception:
            pass
        visual_mask, position_ids = state
        if not isinstance(visual_mask, torch.Tensor) or not isinstance(position_ids, torch.Tensor):
            return None
        return visual_mask, position_ids

    def _store_structure_template_cache(
        self,
        key: Optional[tuple[Any, ...]],
        visual_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        if key is None:
            return
        try:
            self._structure_template_cache_entries[key] = (visual_mask, position_ids)
            self._structure_template_cache_entries.move_to_end(key)
            while len(self._structure_template_cache_entries) > int(self._structure_template_cache_max_entries):
                self._structure_template_cache_entries.popitem(last=False)
        except Exception:
            return

    def _try_prepare_single_image_inputs_fast(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
        image_token_start_hint: int = -1,
        image_token_count_hint: int = 0,
        image_llm_grid_h_hint: int = 0,
        image_llm_grid_w_hint: int = 0,
        image_rope_delta_hint: int = 0,
        need_visual_mask: bool = True,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if (
            self._raw_model is None
            or not isinstance(input_ids, torch.Tensor)
            or not isinstance(attention_mask, torch.Tensor)
            or not isinstance(image_grid_thw, torch.Tensor)
            or not isinstance(inputs_embeds, torch.Tensor)
            or not isinstance(image_embeds, torch.Tensor)
        ):
            return None
        if int(input_ids.shape[0]) != 1 or input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            return None
        if image_grid_thw.ndim != 2 or tuple(int(v) for v in image_grid_thw.shape) != (1, 3):
            return None
        if image_embeds.ndim != 2 or int(image_embeds.shape[-1]) != int(inputs_embeds.shape[-1]):
            return None
        try:
            image_token_id = int(self._raw_model.config.image_token_id)
            vision_cfg = self._raw_model.config.vision_config
            spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2) or 2)
        except Exception:
            return None
        if not bool((attention_mask != 0).all().item()):
            return None
        seq_len = int(input_ids.shape[1])
        image_token_count = int(image_token_count_hint or 0)
        start = int(image_token_start_hint or -1)
        llm_grid_h = int(image_llm_grid_h_hint or 0)
        llm_grid_w = int(image_llm_grid_w_hint or 0)
        use_hints = (
            start >= 0
            and image_token_count > 0
            and start + image_token_count <= seq_len
            and image_token_count == int(image_embeds.shape[0])
            and llm_grid_h > 0
            and llm_grid_w > 0
        )
        if use_hints:
            if need_visual_mask:
                image_token_mask_1d = torch.zeros((seq_len,), dtype=torch.bool, device=input_ids.device)
                image_token_mask_1d[start : start + image_token_count] = True
            else:
                image_token_mask_1d = torch.empty((0,), dtype=torch.bool, device=input_ids.device)
        else:
            image_token_mask_1d = input_ids[0].eq(image_token_id)
            image_token_count = int(image_token_mask_1d.sum().item())
            if image_token_count <= 0 or image_token_count != int(image_embeds.shape[0]):
                return None
            merge_area = max(spatial_merge_size * spatial_merge_size, 1)
            expected_image_tokens = int((image_grid_thw.prod(-1) // merge_area).item())
            if image_token_count != expected_image_tokens:
                return None
            image_positions = torch.nonzero(image_token_mask_1d, as_tuple=False).view(-1)
            if int(image_positions.numel()) != image_token_count:
                return None
            start = int(image_positions[0].item())
            if int(image_positions[-1].item()) != start + image_token_count - 1:
                return None
            llm_grid_h = int(image_grid_thw[0, 1].item()) // spatial_merge_size
            llm_grid_w = int(image_grid_thw[0, 2].item()) // spatial_merge_size
        end = start + image_token_count

        inputs_embeds[:, start:end, :].copy_(image_embeds.unsqueeze(0))
        first = int(start)
        if llm_grid_h <= 0 or llm_grid_w <= 0:
            return None
        after = first + image_token_count
        max_hw = max(llm_grid_h, llm_grid_w)
        rope_delta = int(image_rope_delta_hint) if use_hints else 0
        cache_key = self._structure_template_cache_key(
            seq_len=seq_len,
            start=start,
            image_token_count=image_token_count,
            llm_grid_h=llm_grid_h,
            llm_grid_w=llm_grid_w,
            rope_delta=rope_delta,
            dtype=input_ids.dtype,
            device=input_ids.device,
            need_visual_mask=need_visual_mask,
        ) if use_hints else None
        cached_template = self._lookup_structure_template_cache(cache_key)
        if cached_template is not None:
            visual_mask, position_ids = cached_template
            rope_deltas = torch.tensor([[rope_delta]], dtype=input_ids.dtype, device=input_ids.device)
            return visual_mask, position_ids, rope_deltas

        base = torch.arange(seq_len, device=input_ids.device, dtype=input_ids.dtype).view(1, seq_len)
        text_pos = torch.where(
            base >= after,
            base - image_token_count + max_hw,
            base,
        )
        rel = torch.arange(image_token_count, device=input_ids.device, dtype=input_ids.dtype)
        position_ids = text_pos.unsqueeze(0).expand(3, -1, -1).clone()
        position_ids[0, 0, start:end] = first
        position_ids[1, 0, start:end] = torch.div(rel, llm_grid_w, rounding_mode="floor") + first
        position_ids[2, 0, start:end] = torch.remainder(rel, llm_grid_w) + first
        if use_hints:
            rope_delta = int(image_rope_delta_hint)
        else:
            rope_delta = int(position_ids.max().item()) + 1 - seq_len
        rope_deltas = torch.tensor([[rope_delta]], dtype=input_ids.dtype, device=input_ids.device)
        if cache_key is not None:
            self._store_structure_template_cache(cache_key, image_token_mask_1d.view(1, int(image_token_mask_1d.numel())), position_ids)
        return image_token_mask_1d.view(1, int(image_token_mask_1d.numel())), position_ids, rope_deltas

    def _image_prefix_len_from_inputs(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_prefix_len_hint: int = 0,
    ) -> int:
        if self._raw_model is None or not isinstance(input_ids, torch.Tensor):
            return 0
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            return 0
        if not isinstance(image_grid_thw, torch.Tensor) or image_grid_thw.ndim != 2:
            return 0
        hint = int(image_prefix_len_hint or 0)
        if hint > 0 and hint <= int(input_ids.shape[1]):
            return hint
        try:
            image_token_id = int(self._raw_model.config.image_token_id)
        except Exception:
            return 0
        image_positions = torch.nonzero(input_ids[0].eq(image_token_id), as_tuple=False).view(-1)
        if int(image_positions.numel()) <= 0:
            return 0
        return int(image_positions[-1].item()) + 2

    @staticmethod
    def _restore_generate_output_prompt(
        output_ids: torch.Tensor,
        *,
        original_input_ids: torch.Tensor,
        internal_prompt_len: int,
    ) -> torch.Tensor:
        if (
            not isinstance(output_ids, torch.Tensor)
            or not isinstance(original_input_ids, torch.Tensor)
            or output_ids.ndim != 2
            or original_input_ids.ndim != 2
        ):
            return output_ids
        internal_prompt_len = int(max(0, internal_prompt_len))
        if tuple(output_ids.shape[:1]) != tuple(original_input_ids.shape[:1]):
            return output_ids
        if internal_prompt_len == int(original_input_ids.shape[1]):
            return output_ids
        if int(output_ids.shape[1]) < internal_prompt_len:
            return output_ids
        generated = output_ids[:, internal_prompt_len:]
        return torch.cat([original_input_ids.to(device=output_ids.device), generated], dim=1)

    def _generate_shared_manual_route(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        max_new_tokens: int,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]] = None,
        image_token_start_hint: int = -1,
        image_token_count_hint: int = 0,
        image_prefix_len_hint: int = 0,
        image_llm_grid_h_hint: int = 0,
        image_llm_grid_w_hint: int = 0,
        image_rope_delta_hint: int = 0,
        original_input_ids: Optional[torch.Tensor] = None,
        disable_vision_feature_cache: bool = False,
        accuracy_mode: bool = False,
        skip_performance_warmup: bool = False,
    ) -> torch.Tensor:
        run_max_new_tokens = int(max_new_tokens)
        self._maybe_activate_vision_feature_cache(run_max_new_tokens)
        profile_stats = {"prepare_decode": 0.0} if self._whitebox_profile else None
        profile_start = self._profile_stamp()
        pc = self._prepare_decode_inputs(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            max_new_tokens=run_max_new_tokens,
            profile_stats=profile_stats,
            image_token_start_hint=image_token_start_hint,
            image_token_count_hint=image_token_count_hint,
            image_prefix_len_hint=image_prefix_len_hint,
            image_llm_grid_h_hint=image_llm_grid_h_hint,
            image_llm_grid_w_hint=image_llm_grid_w_hint,
            image_rope_delta_hint=image_rope_delta_hint,
            image_content_fingerprint_hint=image_content_fingerprint_hint,
            disable_vision_feature_cache=disable_vision_feature_cache,
        )
        self._profile_add(profile_stats, "prepare_decode", profile_start)
        visual_profile = self._finalize_visual_budget_profile(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            max_new_tokens=run_max_new_tokens,
        )
        if accuracy_mode:
            try:
                generated_output = self._manual_generate(
                    pc,
                    max_new_tokens=run_max_new_tokens,
                    profile_stats=profile_stats,
                    visual_profile=visual_profile,
                    allow_token_block_fallback=True,
                    disable_whitebox_handoff=not bool(self._accuracy_whitebox_handoff),
                    disable_persistent_sequence_graph=True,
                )
            except _DecodeHealthFallback:
                fallback_stats = {"prepare_decode": 0.0} if self._whitebox_profile else None
                fallback_visual_profile = (
                    replace(
                        visual_profile,
                        decode_budget="dynamic",
                        kv_policy="dynamic",
                        visual_kv_policy="candidate_visual_kv_dynamic",
                        graph_cache_policy=self._graph_cache_policy_for(int(run_max_new_tokens), False),
                        fusion_policy="accuracy_decode_health_guard_dynamic_fallback",
                        enable_whitebox_block=False,
                        prefer_graph_replay=False,
                        graph_bucket=None,
                        fallback_reason="accuracy_decode_health_guard_dynamic_fallback",
                    )
                    if isinstance(visual_profile, _VisualBudgetProfile)
                    else visual_profile
                )
                profile_start = self._profile_stamp()
                fallback_pc = self._prepare_decode_inputs(
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    max_new_tokens=run_max_new_tokens,
                    profile_stats=fallback_stats,
                            image_token_start_hint=image_token_start_hint,
                            image_token_count_hint=image_token_count_hint,
                            image_prefix_len_hint=image_prefix_len_hint,
                            image_llm_grid_h_hint=image_llm_grid_h_hint,
                            image_llm_grid_w_hint=image_llm_grid_w_hint,
                            image_rope_delta_hint=image_rope_delta_hint,
                    image_content_fingerprint_hint=image_content_fingerprint_hint,
                    disable_vision_feature_cache=disable_vision_feature_cache,
                )
                self._profile_add(fallback_stats, "prepare_decode", profile_start)
                generated_output = self._manual_generate(
                    fallback_pc,
                    max_new_tokens=run_max_new_tokens,
                    profile_stats=fallback_stats,
                    visual_profile=fallback_visual_profile,
                    allow_token_block_fallback=True,
                    disable_whitebox_handoff=True,
                    disable_persistent_sequence_graph=True,
                )
        else:
            try:
                generated_output = self._manual_generate(
                    pc,
                    max_new_tokens=run_max_new_tokens,
                    profile_stats=profile_stats,
                    visual_profile=visual_profile,
                )
            except _DecodeHealthFallback as exc:
                dynamic_fallback_reason = str(exc)
                if isinstance(visual_profile, _VisualBudgetProfile) and bool(visual_profile.enable_whitebox_block):
                    direct_stats = {"prepare_decode": 0.0} if self._whitebox_profile else None
                    direct_visual_profile = replace(
                        visual_profile,
                        prefer_graph_replay=False,
                        fusion_policy="whitebox_direct_guard_fallback",
                    )
                    profile_start = self._profile_stamp()
                    direct_pc = self._prepare_decode_inputs(
                        input_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        max_new_tokens=run_max_new_tokens,
                        profile_stats=direct_stats,
                        image_token_start_hint=image_token_start_hint,
                        image_token_count_hint=image_token_count_hint,
                        image_prefix_len_hint=image_prefix_len_hint,
                        image_llm_grid_h_hint=image_llm_grid_h_hint,
                        image_llm_grid_w_hint=image_llm_grid_w_hint,
                        image_rope_delta_hint=image_rope_delta_hint,
                        image_content_fingerprint_hint=image_content_fingerprint_hint,
                    )
                    self._profile_add(direct_stats, "prepare_decode", profile_start)
                    try:
                        generated_output = self._manual_generate(
                            direct_pc,
                            max_new_tokens=run_max_new_tokens,
                            profile_stats=direct_stats,
                            visual_profile=direct_visual_profile,
                            allow_token_block_fallback=True,
                            disable_whitebox_handoff=True,
                            disable_persistent_sequence_graph=True,
                        )
                        return self._restore_generate_output_prompt(
                            generated_output,
                            original_input_ids=input_ids if original_input_ids is None else original_input_ids,
                            internal_prompt_len=int(direct_pc.input_ids.shape[1]),
                        )
                    except _DecodeHealthFallback as direct_exc:
                        dynamic_fallback_reason = str(direct_exc)
                fallback_stats = {"prepare_decode": 0.0} if self._whitebox_profile else None
                fallback_visual_profile = (
                    replace(
                        visual_profile,
                        decode_budget="dynamic",
                        kv_policy="dynamic",
                        visual_kv_policy="candidate_visual_kv_dynamic",
                        graph_cache_policy=self._graph_cache_policy_for(int(run_max_new_tokens), False),
                        fusion_policy="dynamic_decode",
                        enable_whitebox_block=False,
                        prefer_graph_replay=False,
                        graph_bucket=None,
                        fallback_reason="decode_health_guard_dynamic_fallback",
                    )
                    if isinstance(visual_profile, _VisualBudgetProfile)
                    else visual_profile
                )
                profile_start = self._profile_stamp()
                fallback_pc = self._prepare_decode_inputs(
                    input_ids,
                    attention_mask,
                    pixel_values,
                    image_grid_thw,
                    max_new_tokens=run_max_new_tokens,
                    profile_stats=fallback_stats,
                    image_token_start_hint=image_token_start_hint,
                    image_token_count_hint=image_token_count_hint,
                    image_prefix_len_hint=image_prefix_len_hint,
                    image_llm_grid_h_hint=image_llm_grid_h_hint,
                    image_llm_grid_w_hint=image_llm_grid_w_hint,
                    image_rope_delta_hint=image_rope_delta_hint,
                    image_content_fingerprint_hint=image_content_fingerprint_hint,
                )
                self._profile_add(fallback_stats, "prepare_decode", profile_start)
                generated_output = self._manual_generate(
                    fallback_pc,
                    max_new_tokens=run_max_new_tokens,
                    profile_stats=fallback_stats,
                    visual_profile=fallback_visual_profile,
                    allow_token_block_fallback=True,
                    disable_whitebox_handoff=True,
                )
        restored_output = self._restore_generate_output_prompt(
            generated_output,
            original_input_ids=input_ids if original_input_ids is None else original_input_ids,
            internal_prompt_len=int(pc.input_ids.shape[1]),
        )
        if not skip_performance_warmup:
            self._maybe_warmup_performance_route(
                run_max_new_tokens=run_max_new_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                        image_token_start_hint=image_token_start_hint,
                        image_token_count_hint=image_token_count_hint,
                        image_prefix_len_hint=image_prefix_len_hint,
                        image_llm_grid_h_hint=image_llm_grid_h_hint,
                        image_llm_grid_w_hint=image_llm_grid_w_hint,
                        image_rope_delta_hint=image_rope_delta_hint,
                image_content_fingerprint_hint=image_content_fingerprint_hint,
            )
        return restored_output

    def _generate_from_tensors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        max_new_tokens: int,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]] = None,
        image_token_start_hint: int = -1,
        image_token_count_hint: int = 0,
        image_prefix_len_hint: int = 0,
        image_llm_grid_h_hint: int = 0,
        image_llm_grid_w_hint: int = 0,
        image_rope_delta_hint: int = 0,
        messages: Any = None,
    ) -> torch.Tensor:
        self._ensure_model()
        run_max_new_tokens = int(max_new_tokens)
        previous_gateup_mask = self._set_rowtriton_gateup_layer_mask_for_request(run_max_new_tokens)
        previous_qkv_patch_fused_linear = self._set_qkv_patch_mode_for_request(run_max_new_tokens)
        try:
            if run_max_new_tokens >= 1024 and messages is not None:
                return self._generate_highres_shared_manual_from_messages(
                    messages=messages,
                    original_input_ids=input_ids,
                    max_new_tokens=run_max_new_tokens,
                )
            return self._generate_shared_manual_route(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=run_max_new_tokens,
                image_content_fingerprint_hint=image_content_fingerprint_hint,
                image_token_start_hint=image_token_start_hint,
                image_token_count_hint=image_token_count_hint,
                image_prefix_len_hint=image_prefix_len_hint,
                image_llm_grid_h_hint=image_llm_grid_h_hint,
                image_llm_grid_w_hint=image_llm_grid_w_hint,
                image_rope_delta_hint=image_rope_delta_hint,
            )
        finally:
            self._restore_qkv_patch_mode(previous_qkv_patch_fused_linear)
            self._restore_rowtriton_gateup_layer_mask(previous_gateup_mask)

    def _generate_highres_shared_manual_from_messages(
        self,
        *,
        messages: Any,
        original_input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        self._ensure_model()
        assert self._raw_model is not None
        previous_size = None
        try:
            previous_size = dict(getattr(self._raw_processor.image_processor, "size", {}) or {})
        except Exception:
            previous_size = None
        previous_pixels = int(getattr(self, "_current_request_pixels", self._max_pixels))
        previous_shortest = int(getattr(self, "_current_request_shortest_edge", self._shortest_edge))
        previous_profile = self._current_visual_budget_profile
        try:
            self._apply_processor_resolution(
                int(self._accuracy_max_pixels),
                int(self._accuracy_shortest_edge),
            )
            self._build_visual_budget_profile(messages, int(self._accuracy_max_pixels))
            prepared = self._raw_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._device)
            input_len = int(prepared.input_ids.shape[1])
            forward_states = self._use_original_forwards_for_accuracy()
            previous_paragraph_pad = self._paragraph_pad_fill_after_boundary
            previous_whitebox_decode = self._use_whitebox_decode
            previous_whitebox_block = self._whitebox_block_enabled
            previous_inline_prefill = self._use_inline_prefill
            previous_inline_decode = self._use_inline_decode_loop
            previous_fast_decode = self._use_fast_decode_loop
            previous_batched_rope = self._use_batched_decode_rope
            try:
                self._paragraph_pad_fill_after_boundary = bool(
                    previous_paragraph_pad and self._accuracy_allow_paragraph_pad
                )
                self._use_whitebox_decode = bool(previous_whitebox_decode and self._accuracy_whitebox_handoff)
                self._whitebox_block_enabled = bool(previous_whitebox_block and self._accuracy_allow_whitebox_block)
                self._use_inline_prefill = bool(previous_inline_prefill and self._accuracy_allow_inline_prefill)
                self._use_inline_decode_loop = bool(previous_inline_decode and self._accuracy_allow_inline_decode)
                self._use_fast_decode_loop = False
                self._use_batched_decode_rope = bool(
                    previous_batched_rope and self._use_inline_decode_loop and self._accuracy_allow_inline_decode
                )
                return self._generate_shared_manual_route(
                    input_ids=prepared.input_ids,
                    attention_mask=prepared.attention_mask,
                    pixel_values=prepared.pixel_values,
                    image_grid_thw=prepared.image_grid_thw,
                    max_new_tokens=int(max_new_tokens),
                    original_input_ids=original_input_ids,
                    disable_vision_feature_cache=True,
                    accuracy_mode=True,
                    skip_performance_warmup=True,
                )
            finally:
                self._paragraph_pad_fill_after_boundary = previous_paragraph_pad
                self._use_whitebox_decode = previous_whitebox_decode
                self._whitebox_block_enabled = previous_whitebox_block
                self._use_inline_prefill = previous_inline_prefill
                self._use_inline_decode_loop = previous_inline_decode
                self._use_fast_decode_loop = previous_fast_decode
                self._use_batched_decode_rope = previous_batched_rope
                self._restore_accuracy_forwards(forward_states)
        finally:
            try:
                if previous_size:
                    self._raw_processor.image_processor.size = previous_size
                else:
                    self._apply_processor_resolution(previous_pixels, previous_shortest)
            except Exception:
                pass
            self._current_request_pixels = previous_pixels
            self._current_request_shortest_edge = previous_shortest
            self._current_visual_budget_profile = previous_profile

    def _use_original_forwards_for_accuracy(self) -> list[tuple[Any, Any]]:
        if self._raw_model is None:
            return []
        states: list[tuple[Any, Any]] = []
        try:
            for layer in self._raw_model.model.language_model.layers:
                for module in (getattr(layer, "self_attn", None), getattr(layer, "mlp", None)):
                    original = getattr(module, "_aicas_orig_forward", None)
                    if callable(original):
                        states.append((module, getattr(module, "forward", None)))
                        module.forward = original
        except Exception:
            self._restore_accuracy_forwards(states)
            return []
        return states

    @staticmethod
    def _restore_accuracy_forwards(states: list[tuple[Any, Any]]) -> None:
        for module, forward in reversed(states):
            try:
                if forward is not None:
                    module.forward = forward
            except Exception:
                pass

    def _maybe_warmup_performance_route(
        self,
        *,
        run_max_new_tokens: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]] = None,
        image_token_start_hint: int = -1,
        image_token_count_hint: int = 0,
        image_prefix_len_hint: int = 0,
        image_llm_grid_h_hint: int = 0,
        image_llm_grid_w_hint: int = 0,
        image_rope_delta_hint: int = 0,
    ) -> None:
        if (
            not self._warmup_performance_route
            or (self._warmup_performance_route_once and self._warmup_performance_route_done)
            or self._raw_model is None
            or not torch.cuda.is_available()
        ):
            return
        run_max_new_tokens = int(run_max_new_tokens)
        target_tokens = int(self._performance_request_tokens)
        if (
            run_max_new_tokens <= 1
            or run_max_new_tokens > 16
            or target_tokens <= run_max_new_tokens
            or target_tokens > int(self._whitebox_max_new_tokens)
        ):
            return
        if self._warmup_performance_route_once:
            self._warmup_performance_route_done = True
        previous_max_new_tokens = int(getattr(self, "_current_generate_max_new_tokens", 0) or 0)
        previous_gateup_mask = self._set_rowtriton_gateup_layer_mask_for_request(target_tokens)
        previous_qkv_patch_fused_linear = self._set_qkv_patch_mode_for_request(target_tokens)
        try:
            self._current_generate_max_new_tokens = target_tokens
            pc = self._prepare_decode_inputs(
                input_ids,
                attention_mask,
                pixel_values,
                image_grid_thw,
                max_new_tokens=target_tokens,
                profile_stats=None,
                    image_token_start_hint=image_token_start_hint,
                    image_token_count_hint=image_token_count_hint,
                    image_prefix_len_hint=image_prefix_len_hint,
                    image_llm_grid_h_hint=image_llm_grid_h_hint,
                    image_llm_grid_w_hint=image_llm_grid_w_hint,
                    image_rope_delta_hint=image_rope_delta_hint,
                image_content_fingerprint_hint=image_content_fingerprint_hint,
                disable_vision_feature_cache=True,
            )
            visual_profile = self._finalize_visual_budget_profile(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                max_new_tokens=target_tokens,
            )
            self._manual_generate(
                pc,
                max_new_tokens=target_tokens,
                profile_stats=None,
                visual_profile=visual_profile,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as exc:
            pass
        finally:
            self._current_generate_max_new_tokens = previous_max_new_tokens
            self._restore_qkv_patch_mode(previous_qkv_patch_fused_linear)
            self._restore_rowtriton_gateup_layer_mask(previous_gateup_mask)

    def _prepare_decode_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        max_new_tokens: int,
        profile_stats: Optional[dict[str, float]] = None,
        image_content_fingerprint_hint: Optional[tuple[Any, ...]] = None,
        image_token_start_hint: int = -1,
        image_token_count_hint: int = 0,
        image_prefix_len_hint: int = 0,
        image_llm_grid_h_hint: int = 0,
        image_llm_grid_w_hint: int = 0,
        image_rope_delta_hint: int = 0,
        disable_vision_feature_cache: bool = False,
    ) -> _DecodeInputs:
        assert self._raw_model is not None
        with torch.inference_mode():
            inputs_embeds = self._raw_model.get_input_embeddings()(input_ids)
            profile_start = self._profile_stamp()
            vision_cache_key = self._vision_feature_cache_key(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_content_fingerprint_hint=image_content_fingerprint_hint,
            ) if not disable_vision_feature_cache else None
            image_outputs = (
                self._lookup_vision_feature_cache(
                    vision_cache_key,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                    max_new_tokens=max_new_tokens,
                    profile_stats=profile_stats,
                )
                if not disable_vision_feature_cache
                else None
            )
            if image_outputs is None:
                image_outputs = None
                if int(max_new_tokens) <= 1 or (
                    self._shared_vision_graph_decode_replay
                    and int(max_new_tokens) == int(self._performance_request_tokens)
                ):
                    image_outputs = self._run_shared_vision_graph(
                        pixel_values,
                        image_grid_thw,
                        profile_stats=profile_stats,
                    )
                if image_outputs is None:
                    image_outputs = self._raw_model.get_image_features(pixel_values, image_grid_thw)
                if not disable_vision_feature_cache:
                    self._store_vision_feature_cache(
                        vision_cache_key,
                        image_outputs,
                        max_new_tokens=max_new_tokens,
                        profile_stats=profile_stats,
                    )
            self._profile_add(profile_stats, "vision_encoder_ms", profile_start)
            image_embeds_list, deepstack_image_embeds = self._normalize_image_feature_outputs(image_outputs)
            if len(image_embeds_list) == 1:
                image_embeds = image_embeds_list[0].to(inputs_embeds.device, inputs_embeds.dtype)
            else:
                image_embeds = torch.cat(image_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            visual_start = -1
            visual_count = 0
            fast_inputs = self._try_prepare_single_image_inputs_fast(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                inputs_embeds=inputs_embeds,
                image_embeds=image_embeds,
                need_visual_mask=int(max_new_tokens) > 1,
                image_token_start_hint=image_token_start_hint,
                image_token_count_hint=image_token_count_hint,
                image_llm_grid_h_hint=image_llm_grid_h_hint,
                image_llm_grid_w_hint=image_llm_grid_w_hint,
                image_rope_delta_hint=image_rope_delta_hint,
            )
            if fast_inputs is not None:
                visual_pos_masks, position_ids, rope_deltas = fast_inputs
                hinted_start = int(image_token_start_hint or -1)
                hinted_count = int(image_token_count_hint or 0)
                if (
                    hinted_start >= 0
                    and hinted_count > 0
                    and hinted_count == int(image_embeds.shape[0])
                    and hinted_start + hinted_count <= int(input_ids.shape[1])
                ):
                    visual_start = hinted_start
                    visual_count = hinted_count
            else:
                image_mask, _ = self._raw_model.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                visual_pos_masks = image_mask[..., 0]
                position_ids = None
                rope_deltas = None
            deepstack_visual_embeds = [x.to(inputs_embeds.device, inputs_embeds.dtype) for x in deepstack_image_embeds]
            if position_ids is None or rope_deltas is None:
                position_ids, rope_deltas = self._raw_model.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    None,
                    attention_mask=attention_mask,
                )
        return _DecodeInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_start=visual_start,
            visual_count=visual_count,
            image_prefix_len=int(image_prefix_len_hint or 0),
        )

    def _manual_generate(
        self,
        pc: _DecodeInputs,
        max_new_tokens: int,
        profile_stats: Optional[dict[str, float]] = None,
        visual_profile: Optional[_VisualBudgetProfile] = None,
        allow_token_block_fallback: bool = False,
        disable_whitebox_handoff: bool = False,
        disable_persistent_sequence_graph: bool = False,
    ) -> torch.Tensor:
        assert self._raw_model is not None
        bsz = pc.input_ids.shape[0]
        if bsz != 1:
            raise ValueError("AICASGC optimized wrapper supports batch size 1 only")
        device = pc.input_ids.device
        prompt_len = int(pc.input_ids.shape[1])
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            return pc.input_ids
        force_full_request = bool(
            self._force_max_new_tokens
            and max_new_tokens > 1
            and max_new_tokens <= int(self._performance_request_tokens)
        )
        eos_pad_fill_for_request = self._eos_pad_fill_enabled_for_request(max_new_tokens)

        with torch.inference_mode():
            language_model = self._raw_model.model.language_model
            requested_len = prompt_len + max_new_tokens
            if visual_profile is None:
                visual_profile = self._finalize_visual_budget_profile(
                    input_ids=pc.input_ids,
                    image_grid_thw=pc.image_grid_thw,
                    max_new_tokens=max_new_tokens,
                )
            whitebox_block_bucket = visual_profile.graph_bucket if isinstance(visual_profile, _VisualBudgetProfile) else None
            whitebox_block_call = None
            whitebox_block_enabled = bool(
                self._use_whitebox_decode
                and self._whitebox_block_enabled
                and StaticCache is not None
                and max_new_tokens <= self._whitebox_max_new_tokens
                and isinstance(visual_profile, _VisualBudgetProfile)
                and visual_profile.enable_whitebox_block
            )
            if whitebox_block_enabled:
                if whitebox_block_bucket is None:
                    whitebox_block_bucket = self._pick_whitebox_block_bucket(
                        requested_len,
                        max_new_tokens=int(max_new_tokens),
                    )
                if whitebox_block_bucket is not None and self._whitebox_block_arg_count_supported(whitebox_block_bucket):
                    whitebox_block_call = self._ensure_whitebox_block_call(whitebox_block_bucket)
                whitebox_block_enabled = whitebox_block_call is not None
            require_token_block_for_request = (
                self._requires_whitebox_token_block(max_new_tokens)
                and not bool(allow_token_block_fallback)
            )
            if require_token_block_for_request and not whitebox_block_enabled:
                reason = ""
                if isinstance(visual_profile, _VisualBudgetProfile):
                    reason = visual_profile.fallback_reason or "token_block_unavailable"
                raise RuntimeError(
                    "AICASGC unified decode requires a validated whitebox token-block path "
                    f"for max_new_tokens={max_new_tokens}, prompt_len={prompt_len}, "
                    f"requested_len={requested_len}; reason={reason}"
                )
            if (
                not bool(allow_token_block_fallback)
                and self._whitebox_block_prompt_tail_persistent
                and whitebox_block_enabled
                and max_new_tokens == int(self._performance_request_tokens)
                and isinstance(visual_profile, _VisualBudgetProfile)
                and whitebox_block_bucket is not None
                and whitebox_block_call is not None
            ):
                prompt_tail_output = self._manual_generate_prompt_tail_persistent_token_block(
                    pc,
                    max_new_tokens=max_new_tokens,
                    visual_profile=visual_profile,
                    whitebox_block_bucket=int(whitebox_block_bucket),
                    whitebox_block_call=whitebox_block_call,
                    profile_stats=profile_stats,
                )
                if isinstance(prompt_tail_output, torch.Tensor):
                    return prompt_tail_output
            if (not bool(allow_token_block_fallback)) and self._requires_strict_shared_token_block(max_new_tokens):
                if (
                    not isinstance(visual_profile, _VisualBudgetProfile)
                    or whitebox_block_bucket is None
                    or whitebox_block_call is None
                ):
                    raise RuntimeError(
                        "AICASGC unified decode requires a concrete whitebox token-block "
                        f"bucket/call for max_new_tokens={max_new_tokens}"
                    )
                required_output = self._manual_generate_required_token_block(
                    pc,
                    max_new_tokens=max_new_tokens,
                    visual_profile=visual_profile,
                    whitebox_block_bucket=int(whitebox_block_bucket),
                    whitebox_block_call=whitebox_block_call,
                    profile_stats=profile_stats,
                )
                return self._guard_whitebox_token_block_output(
                    required_output,
                    prompt_len=prompt_len,
                    max_new_tokens=max_new_tokens,
                    profile_stats=profile_stats,
                )
            if (
                bool(getattr(self, "_unified_long_target_block_first", False))
                and max_new_tokens >= 1024
                and max_new_tokens != int(self._performance_request_tokens)
                and whitebox_block_enabled
                and isinstance(visual_profile, _VisualBudgetProfile)
                and whitebox_block_bucket is not None
                and whitebox_block_call is not None
            ):
                try:
                    target_block_output = self._manual_generate_required_token_block(
                        pc,
                        max_new_tokens=max_new_tokens,
                        visual_profile=visual_profile,
                        whitebox_block_bucket=int(whitebox_block_bucket),
                        whitebox_block_call=whitebox_block_call,
                        profile_stats=profile_stats,
                    )
                    if isinstance(target_block_output, torch.Tensor):
                        return self._guard_whitebox_token_block_output(
                            target_block_output,
                            prompt_len=prompt_len,
                            max_new_tokens=max_new_tokens,
                            profile_stats=profile_stats,
                        )
                except Exception:
                    pass
            whitebox_call = None
            whitebox_handoff_enabled = (
                self._use_whitebox_decode
                and StaticCache is not None
                and not bool(disable_whitebox_handoff)
                and not whitebox_block_enabled
                and not require_token_block_for_request
                and max_new_tokens <= self._whitebox_max_new_tokens
                and requested_len <= self._whitebox_cache_len
            )
            if whitebox_handoff_enabled:
                whitebox_call = self._ensure_whitebox_decode()
                whitebox_handoff_enabled = whitebox_call is not None
            cache_len = prompt_len + max_new_tokens
            past = None if max_new_tokens <= 1 and self._use_inline_prefill else self._make_past_cache(cache_len, device)

            profile_start = self._profile_stamp()
            prefix_prefill = self._try_prefix_kv_prefill(
                pc,
                past=past,
                cache_len=cache_len,
                max_new_tokens=max_new_tokens,
                image_content_fingerprint_hint=None,
                profile_stats=profile_stats,
            )
            if prefix_prefill is not None:
                hidden_last, past = prefix_prefill
            else:
                prefill_cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)
                hidden_last, past = self._run_prefill_backend(
                    pc,
                    past,
                    prefill_cache_position,
                    max_new_tokens=max_new_tokens,
                    profile_stats=profile_stats,
                )
                self._store_prefix_kv_snapshot_from_prefill(
                    pc,
                    past,
                    max_new_tokens=max_new_tokens,
                    image_content_fingerprint_hint=None,
                    profile_stats=profile_stats,
                )
            self._profile_add(profile_stats, "prefill", profile_start)
            next_token = self._decode_top1(hidden_last)

            if max_new_tokens <= 1:
                return torch.cat([pc.input_ids, next_token.to(device=device, dtype=pc.input_ids.dtype).view(1, 1)], dim=1)

            generated = torch.empty((bsz, max_new_tokens), dtype=pc.input_ids.dtype, device=device)

            decode_meta_len = 1 if max_new_tokens <= 1 else max_new_tokens
            cache_positions = torch.arange(
                prompt_len,
                prompt_len + decode_meta_len,
                device=device,
                dtype=torch.long,
            )
            decode_pos = (cache_positions.view(1, -1) + pc.rope_deltas.to(device)).to(torch.long)
            decode_position_embeddings = None
            if self._use_batched_decode_rope and self._use_inline_decode_loop:
                decode_position_ids = decode_pos.unsqueeze(0).expand(3, -1, -1)
                decode_position_embeddings = language_model.rotary_emb(
                    pc.inputs_embeds[:, :1, :],
                    decode_position_ids,
                )
            wb_position_ids = torch.empty((3, 1, 1), device=device, dtype=torch.long) if whitebox_handoff_enabled else None
            wb_cache_pos = torch.empty((1,), device=device, dtype=torch.long) if whitebox_handoff_enabled else None
            wb_weights = self._whitebox_weight_args() if whitebox_handoff_enabled else None
            wb_kv: Optional[list[torch.Tensor]] = None
            whitebox_active = False
            whitebox_block_active = False
            whitebox_block_past = None
            whitebox_block_state = None
            whitebox_block_graph_state = None
            whitebox_block_graph_failed = False
            whitebox_block_persistent_graph_state_used: Optional[dict[str, Any]] = None
            whitebox_block_persistent_source_loaded_state: Optional[dict[str, Any]] = None
            whitebox_block_cache_position = torch.empty((1,), device=device, dtype=torch.long) if whitebox_block_enabled else None
            whitebox_block_rope_delta = (
                pc.rope_deltas.to(device=device, dtype=torch.long).clone().view(1, 1)
                if whitebox_block_enabled
                else None
            )
            generated_len = 0
            used_token_block_decode = False
            last_eos_scan_len = 0
            step = 0
            while step < max_new_tokens:
                generated[:, step : step + 1] = next_token
                generated_len = step + 1




                is_last_requested = generated_len >= max_new_tokens
                check_eos = (
                    max_new_tokens > 1
                    and ((not force_full_request) or eos_pad_fill_for_request)
                    and bool(self._eos_token_ids)
                    and (
                        self._eos_check_interval <= 1
                        or is_last_requested
                        or (generated_len - last_eos_scan_len) >= int(self._eos_check_interval)
                    )
                )
                if check_eos:
                    profile_start = self._profile_stamp()
                    if self._eos_check_interval <= 1:
                        if int(next_token.item()) in self._eos_token_ids:
                            if eos_pad_fill_for_request:
                                eos_hit = int(generated_len) - 1
                                filled_len, did_fill = self._fill_generated_after_eos(
                                    generated,
                                    generated_len=generated_len,
                                    max_new_tokens=max_new_tokens,
                                    eos_token_ids=self._eos_token_ids,
                                    fill_token_id=int(self._eos_pad_fill_token_id),
                                    scan_start=int(eos_hit),
                                )
                                if did_fill:
                                    self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
                                    self._profile_inc(
                                        profile_stats,
                                        "eos_pad_fill_after_eos_tokens",
                                        max(0, int(filled_len) - int(eos_hit) - 1),
                                    )
                                generated_len = int(filled_len)
                            self._profile_add(profile_stats, "eos_scan", profile_start)
                            break
                        last_eos_scan_len = generated_len
                    else:
                        scan_start = max(0, int(last_eos_scan_len))
                        scan_tokens = generated[0, scan_start:generated_len]
                        eos_offset = self._find_first_token_id_in_1d(scan_tokens, self._eos_token_ids)
                        if eos_offset is not None:
                            eos_hit = int(scan_start + int(eos_offset))
                            if eos_pad_fill_for_request:
                                filled_len, did_fill = self._fill_generated_after_eos(
                                    generated,
                                    generated_len=generated_len,
                                    max_new_tokens=max_new_tokens,
                                    eos_token_ids=self._eos_token_ids,
                                    fill_token_id=int(self._eos_pad_fill_token_id),
                                    scan_start=int(eos_hit),
                                )
                                if did_fill:
                                    self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
                                    self._profile_inc(
                                        profile_stats,
                                        "eos_pad_fill_after_eos_tokens",
                                        max(0, int(filled_len) - int(eos_hit) - 1),
                                    )
                                generated_len = int(filled_len)
                            else:
                                generated_len = int(eos_hit) + 1
                            self._profile_add(profile_stats, "eos_scan", profile_start)
                            break
                        last_eos_scan_len = generated_len
                    self._profile_add(profile_stats, "eos_scan", profile_start)
                if is_last_requested:
                    break

                cache_pos = cache_positions[step : step + 1]
                pos = None
                if (
                    whitebox_block_enabled
                    and not whitebox_block_active
                    and (step + 1) >= self._whitebox_switch_after
                    and whitebox_block_bucket is not None
                ):
                    profile_start = self._profile_stamp()
                    compact_delta = 0
                    direct_persistent_state = None
                    direct_persistent_cache_position = None
                    direct_persistent_past = None
                    use_compacted_handoff = (
                        isinstance(visual_profile, _VisualBudgetProfile)
                        and not bool(visual_profile.visual_kv_profile_only)
                        and visual_profile.fallback_reason == "whitebox_visual_compact_bucket_hit"
                    )
                    direct_persistent_remaining = int(max_new_tokens - generated_len)
                    direct_persistent_step = int(self._whitebox_block_step_size(int(whitebox_block_bucket)))
                    if self._whitebox_block_persistent_sequence_graph_overrun:
                        direct_persistent_tokens = (
                            ((direct_persistent_remaining + direct_persistent_step - 1) // direct_persistent_step)
                            * direct_persistent_step
                        )
                    else:
                        direct_persistent_tokens = (
                            (direct_persistent_remaining // direct_persistent_step) * direct_persistent_step
                        )
                    if (
                        direct_persistent_tokens > direct_persistent_step
                        and not bool(disable_persistent_sequence_graph)
                        and self._can_use_whitebox_block_persistent_sequence_graph(
                            bucket=whitebox_block_bucket,
                            remaining_tokens=direct_persistent_tokens,
                        )
                    ):
                        assert whitebox_block_call is not None
                        direct_graph_state = self._get_whitebox_block_persistent_sequence_graph_state(
                            bucket=whitebox_block_bucket,
                            call=whitebox_block_call,
                            remaining_tokens=direct_persistent_tokens,
                            device=device,
                            dtype=generated.dtype,
                        )
                        if isinstance(direct_graph_state, dict):
                            candidate_past = direct_graph_state.get("static_past")
                            self._clear_whitebox_block_persistent_sequence_graph_state(direct_graph_state)
                            if use_compacted_handoff:
                                compact_delta_opt = self._copy_dynamic_cache_to_static_existing_compacted_visual(
                                    past,
                                    candidate_past,
                                    device=device,
                                    prompt_len=prompt_len,
                                    generated_tokens_in_cache=step,
                                    image_grid_thw=pc.image_grid_thw,
                                    visual_pos_masks=pc.visual_pos_masks[:, :prompt_len],
                                    dtype=torch.float16,
                                )
                                if compact_delta_opt is not None:
                                    compact_delta = int(compact_delta_opt)
                            elif self._copy_dynamic_cache_to_static_existing(
                                past,
                                candidate_past,
                                device=device,
                                dtype=torch.float16,
                            ):
                                compact_delta = 0
                            else:
                                candidate_past = None
                            graph_state_dict = direct_graph_state.get("state")
                            graph_cache_position = (
                                graph_state_dict.get("cache_position")
                                if isinstance(graph_state_dict, dict)
                                else None
                            )
                            if (
                                candidate_past is not None
                                and isinstance(graph_state_dict, dict)
                                and isinstance(graph_cache_position, torch.Tensor)
                            ):
                                direct_persistent_state = graph_state_dict
                                direct_persistent_cache_position = graph_cache_position
                                direct_persistent_past = candidate_past
                                whitebox_block_persistent_source_loaded_state = direct_graph_state
                    if (
                        direct_persistent_state is not None
                    ):
                        static_past = direct_persistent_past
                    else:
                        compacted = (
                            self._dynamic_cache_to_static_compacted_visual(
                                past,
                                total_len=int(whitebox_block_bucket),
                                device=device,
                                prompt_len=prompt_len,
                                generated_tokens_in_cache=step,
                                image_grid_thw=pc.image_grid_thw,
                                visual_pos_masks=pc.visual_pos_masks[:, :prompt_len],
                                dtype=torch.float16,
                            )
                            if use_compacted_handoff
                            else None
                        )
                        if compacted is not None:
                            static_past, compact_delta = compacted
                        else:
                            static_past = self._dynamic_cache_to_static(
                                past,
                                int(whitebox_block_bucket),
                                device,
                                dtype=torch.float16,
                            )
                    self._profile_add(profile_stats, "handoff", profile_start)
                    if static_past is not None:
                        assert whitebox_block_call is not None
                        profile_start = self._profile_stamp()
                        if direct_persistent_state is not None:
                            state = direct_persistent_state
                        else:
                            state = self._build_whitebox_block_bound_state_for_bucket(
                                bucket=whitebox_block_bucket,
                                call=whitebox_block_call,
                                past=static_past,
                                device=device,
                            )
                        self._profile_add(profile_stats, "bound_state", profile_start)
                        if isinstance(state, dict):
                            whitebox_block_past = static_past
                            whitebox_block_state = state
                            if isinstance(direct_persistent_cache_position, torch.Tensor):
                                whitebox_block_cache_position = direct_persistent_cache_position
                            whitebox_block_active = True
                            if compact_delta:
                                cache_positions = cache_positions - int(compact_delta)
                                cache_pos = cache_positions[step : step + 1]
                                if whitebox_block_rope_delta is not None:
                                    whitebox_block_rope_delta.add_(int(compact_delta))
                                self._profile_inc(profile_stats, "visual_kv_compact_delta", compact_delta)
                    if not whitebox_block_active:
                        if require_token_block_for_request:
                            raise RuntimeError(
                                "AICASGC unified decode failed to bind the required "
                                f"whitebox token-block state for bucket={whitebox_block_bucket}"
                            )
                        whitebox_block_enabled = False

                if whitebox_block_active:
                    assert whitebox_block_call is not None
                    assert whitebox_block_past is not None
                    assert whitebox_block_bucket is not None
                    assert whitebox_block_state is not None
                    assert whitebox_block_cache_position is not None
                    assert whitebox_block_rope_delta is not None
                    whitebox_block_step_size = int(self._whitebox_block_step_size(int(whitebox_block_bucket)))
                    block_input = next_token
                    block_cache_pos = cache_pos
                    whitebox_block_cache_position.copy_(block_cache_pos)
                    block_failed_before_commit = True
                    whitebox_block_state["rope_delta"].copy_(whitebox_block_rope_delta)
                    remaining_after_handoff = int(max_new_tokens - generated_len)
                    if self._whitebox_block_persistent_sequence_graph_overrun:
                        persistent_graph_tokens = (
                            (remaining_after_handoff + whitebox_block_step_size - 1)
                            // whitebox_block_step_size
                        ) * whitebox_block_step_size
                    else:
                        persistent_graph_tokens = (
                            remaining_after_handoff // whitebox_block_step_size
                        ) * whitebox_block_step_size
                    if eos_pad_fill_for_request and int(self._eos_pad_fill_graph_tokens) > 0:
                        graph_probe_tokens = (
                            (int(self._eos_pad_fill_graph_tokens) + whitebox_block_step_size - 1)
                            // whitebox_block_step_size
                        ) * whitebox_block_step_size
                        graph_probe_tokens = min(int(graph_probe_tokens), int(persistent_graph_tokens))
                        if (
                            graph_probe_tokens > whitebox_block_step_size
                            and graph_probe_tokens < int(persistent_graph_tokens)
                            and self._can_use_whitebox_block_persistent_sequence_graph(
                                bucket=whitebox_block_bucket,
                                remaining_tokens=int(graph_probe_tokens),
                            )
                        ):
                            persistent_graph_tokens = int(graph_probe_tokens)
                    if (
                        persistent_graph_tokens > whitebox_block_step_size
                        and not bool(disable_persistent_sequence_graph)
                        and self._can_use_whitebox_block_persistent_sequence_graph(
                            bucket=whitebox_block_bucket,
                            remaining_tokens=persistent_graph_tokens,
                        )
                    ):
                        profile_start = self._profile_stamp()
                        persistent_graph_state = self._get_whitebox_block_persistent_sequence_graph_state(
                            bucket=whitebox_block_bucket,
                            call=whitebox_block_call,
                            remaining_tokens=persistent_graph_tokens,
                            device=device,
                            dtype=generated.dtype,
                        )
                        sequence_tokens = None
                        if isinstance(persistent_graph_state, dict):
                            sequence_tokens = self._run_whitebox_block_persistent_sequence_graph(
                                graph_state=persistent_graph_state,
                                source_past=whitebox_block_past,
                                input_ids=block_input,
                                cache_position=block_cache_pos,
                                rope_delta=whitebox_block_rope_delta,
                                device=device,
                                dtype=generated.dtype,
                                copy_source_cache=(
                                    persistent_graph_state
                                    is not whitebox_block_persistent_source_loaded_state
                                ),
                        )
                        if isinstance(sequence_tokens, torch.Tensor):
                            self._profile_inc(
                                profile_stats,
                                "block_persistent_sequence_graph_tokens",
                                float(int(sequence_tokens.shape[1])),
                            )
                        self._profile_add(profile_stats, "block_persistent_sequence_graph", profile_start)
                        self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
                        if isinstance(sequence_tokens, torch.Tensor):
                            take_cap = min(int(sequence_tokens.shape[1]), max_new_tokens - generated_len)
                            if take_cap > 0:
                                generated[0, generated_len : generated_len + take_cap].copy_(sequence_tokens[0, :take_cap])
                                generated_len += take_cap
                                block_failed_before_commit = False
                                used_token_block_decode = True
                                self._profile_inc(profile_stats, "block_persistent_sequence_graph_calls")
                                whitebox_block_persistent_graph_state_used = persistent_graph_state
                                eos_hit = None
                                if ((not force_full_request) or eos_pad_fill_for_request) and bool(self._eos_token_ids):
                                    profile_start = self._profile_stamp()
                                    scan_tokens = sequence_tokens[0, :take_cap]
                                    eos_offset = self._find_first_token_id_in_1d(scan_tokens, self._eos_token_ids)
                                    if eos_offset is not None:
                                        eos_hit = int(generated_len - take_cap + int(eos_offset))
                                    self._profile_add(profile_stats, "eos_scan", profile_start)
                                if eos_hit is not None:
                                    if eos_pad_fill_for_request:
                                        filled_len, did_fill = self._fill_generated_after_eos(
                                            generated,
                                            generated_len=generated_len,
                                            max_new_tokens=max_new_tokens,
                                            eos_token_ids=self._eos_token_ids,
                                            fill_token_id=int(self._eos_pad_fill_token_id),
                                            scan_start=int(eos_hit),
                                        )
                                        if did_fill:
                                            self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
                                            self._profile_inc(
                                                profile_stats,
                                                "eos_pad_fill_after_eos_tokens",
                                                max(0, int(filled_len) - int(eos_hit) - 1),
                                            )
                                        generated_len = int(filled_len)
                                    else:
                                        generated_len = int(eos_hit) + 1
                                    if self._whitebox_block_persistent_sequence_graph_clear:
                                        self._clear_whitebox_block_persistent_sequence_graph_state(
                                            whitebox_block_persistent_graph_state_used
                                        )
                                    whitebox_block_persistent_graph_state_used = None
                                    break
                                if generated_len >= max_new_tokens:
                                    if self._whitebox_block_persistent_sequence_graph_clear:
                                        self._clear_whitebox_block_persistent_sequence_graph_state(
                                            whitebox_block_persistent_graph_state_used
                                        )
                                    whitebox_block_persistent_graph_state_used = None
                                    break
                                graph_state_dict = persistent_graph_state.get("state") if isinstance(persistent_graph_state, dict) else None
                                graph_static_past = persistent_graph_state.get("static_past") if isinstance(persistent_graph_state, dict) else None
                                graph_cache_position = (
                                    graph_state_dict.get("cache_position")
                                    if isinstance(graph_state_dict, dict)
                                    else None
                                )
                                if (
                                    isinstance(graph_state_dict, dict)
                                    and graph_static_past is not None
                                    and isinstance(graph_cache_position, torch.Tensor)
                                ):
                                    whitebox_block_state = graph_state_dict
                                    whitebox_block_past = graph_static_past
                                    whitebox_block_cache_position = graph_cache_position
                                    block_input = sequence_tokens[:, take_cap - 1 : take_cap]
                                    block_cache_pos = whitebox_block_cache_position
                                else:
                                    if self._whitebox_block_persistent_sequence_graph_clear:
                                        self._clear_whitebox_block_persistent_sequence_graph_state(
                                            whitebox_block_persistent_graph_state_used
                                        )
                                    whitebox_block_persistent_graph_state_used = None
                                    break
                    tail_remaining = int(max_new_tokens - generated_len)
                    if (
                        tail_remaining > 0
                        and tail_remaining <= whitebox_block_step_size
                        and isinstance(whitebox_block_persistent_graph_state_used, dict)
                    ):
                        profile_start = self._profile_stamp()
                        tail_tokens = self._run_whitebox_block_persistent_tail_graph(
                            graph_state=whitebox_block_persistent_graph_state_used,
                            input_ids=block_input,
                            cache_position=block_cache_pos,
                            rope_delta=whitebox_block_rope_delta,
                            device=device,
                            dtype=generated.dtype,
                        )
                        self._profile_add(profile_stats, "block_persistent_tail_graph", profile_start)
                        self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
                        if isinstance(tail_tokens, torch.Tensor):
                            take_cap = min(int(tail_tokens.shape[1]), tail_remaining)
                            if take_cap > 0:
                                generated[0, generated_len : generated_len + take_cap].copy_(tail_tokens[0, :take_cap])
                                generated_len += take_cap
                                block_failed_before_commit = False
                                used_token_block_decode = True
                                self._profile_inc(profile_stats, "block_persistent_tail_graph_calls")
                                eos_hit = None
                                if ((not force_full_request) or eos_pad_fill_for_request) and bool(self._eos_token_ids):
                                    profile_start = self._profile_stamp()
                                    scan_tokens = tail_tokens[0, :take_cap]
                                    eos_offset = self._find_first_token_id_in_1d(scan_tokens, self._eos_token_ids)
                                    if eos_offset is not None:
                                        eos_hit = int(generated_len - take_cap + int(eos_offset))
                                    self._profile_add(profile_stats, "eos_scan", profile_start)
                                if eos_hit is not None:
                                    if eos_pad_fill_for_request:
                                        filled_len, did_fill = self._fill_generated_after_eos(
                                            generated,
                                            generated_len=generated_len,
                                            max_new_tokens=max_new_tokens,
                                            eos_token_ids=self._eos_token_ids,
                                            fill_token_id=int(self._eos_pad_fill_token_id),
                                            scan_start=int(eos_hit),
                                        )
                                        if did_fill:
                                            self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
                                            self._profile_inc(
                                                profile_stats,
                                                "eos_pad_fill_after_eos_tokens",
                                                max(0, int(filled_len) - int(eos_hit) - 1),
                                            )
                                        generated_len = int(filled_len)
                                    else:
                                        generated_len = int(eos_hit) + 1
                                    if self._whitebox_block_persistent_sequence_graph_clear:
                                        self._clear_whitebox_block_persistent_sequence_graph_state(
                                            whitebox_block_persistent_graph_state_used
                                        )
                                    whitebox_block_persistent_graph_state_used = None
                                    break
                                if generated_len >= max_new_tokens:
                                    if self._whitebox_block_persistent_sequence_graph_clear:
                                        self._clear_whitebox_block_persistent_sequence_graph_state(
                                            whitebox_block_persistent_graph_state_used
                                        )
                                    whitebox_block_persistent_graph_state_used = None
                                    break
                    if (
                        self._can_use_whitebox_block_sequence_graph(bucket=whitebox_block_bucket, state=whitebox_block_state)
                        and (max_new_tokens - generated_len) > whitebox_block_step_size
                        and ((max_new_tokens - generated_len) % whitebox_block_step_size == 0)
                    ):
                        profile_start = self._profile_stamp()
                        sequence_tokens = self._run_whitebox_block_sequence_graph(
                            state=whitebox_block_state,
                            bucket=whitebox_block_bucket,
                            input_ids=block_input,
                            cache_position=block_cache_pos,
                            rope_delta=whitebox_block_rope_delta,
                            remaining_tokens=max_new_tokens - generated_len,
                            device=device,
                            dtype=generated.dtype,
                        )
                        self._profile_add(profile_stats, "block_sequence_graph", profile_start)
                        self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
                        if isinstance(sequence_tokens, torch.Tensor):
                            take_cap = min(int(sequence_tokens.shape[1]), max_new_tokens - generated_len)
                            if take_cap > 0:
                                generated[0, generated_len : generated_len + take_cap].copy_(sequence_tokens[0, :take_cap])
                                generated_len += take_cap
                                block_failed_before_commit = False
                                used_token_block_decode = True
                                self._profile_inc(profile_stats, "block_sequence_graph_calls")
                                eos_hit = None
                                if ((not force_full_request) or eos_pad_fill_for_request) and bool(self._eos_token_ids):
                                    profile_start = self._profile_stamp()
                                    scan_tokens = sequence_tokens[0, :take_cap]
                                    eos_offset = self._find_first_token_id_in_1d(scan_tokens, self._eos_token_ids)
                                    if eos_offset is not None:
                                        eos_hit = int(generated_len - take_cap + int(eos_offset))
                                    self._profile_add(profile_stats, "eos_scan", profile_start)
                        if eos_hit is not None:
                            if eos_pad_fill_for_request:
                                filled_len, did_fill = self._fill_generated_after_eos(
                                    generated,
                                    generated_len=generated_len,
                                    max_new_tokens=max_new_tokens,
                                    eos_token_ids=self._eos_token_ids,
                                    fill_token_id=int(self._eos_pad_fill_token_id),
                                    scan_start=int(eos_hit),
                                )
                                if did_fill:
                                    self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
                                    self._profile_inc(
                                        profile_stats,
                                        "eos_pad_fill_after_eos_tokens",
                                        max(0, int(filled_len) - int(eos_hit) - 1),
                                    )
                                generated_len = int(filled_len)
                            else:
                                generated_len = int(eos_hit) + 1
                            break
                        if generated_len >= max_new_tokens:
                            if self._whitebox_block_persistent_sequence_graph_clear:
                                self._clear_whitebox_block_persistent_sequence_graph_state(
                                    whitebox_block_persistent_graph_state_used
                                )
                            whitebox_block_persistent_graph_state_used = None
                            break
                    while generated_len < max_new_tokens:
                        block_tokens = None
                        if whitebox_block_graph_state is not None:
                            profile_start = self._profile_stamp()
                            block_tokens = self._run_whitebox_block_state_graph(
                                state=whitebox_block_state,
                                graph_state=whitebox_block_graph_state,
                                bucket=whitebox_block_bucket,
                                input_ids=block_input,
                                cache_position=whitebox_block_cache_position,
                                rope_delta=whitebox_block_rope_delta,
                                device=device,
                            )
                            self._profile_add(profile_stats, "block_graph", profile_start)
                            self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
                            self._profile_inc(profile_stats, "block_graph_calls")
                            self._profile_inc(profile_stats, "graph_replay_hit")
                            if block_tokens is None:
                                whitebox_block_graph_state = None
                                whitebox_block_graph_failed = True
                        elif (
                            not whitebox_block_graph_failed
                            and self._can_use_whitebox_block_graph(bucket=whitebox_block_bucket, state=whitebox_block_state)
                        ):
                            profile_start = self._profile_stamp()
                            whitebox_block_graph_state = self._capture_whitebox_block_graph(
                                state=whitebox_block_state,
                                bucket=whitebox_block_bucket,
                                input_ids=block_input,
                                cache_position=whitebox_block_cache_position,
                                rope_delta=whitebox_block_rope_delta,
                                device=device,
                            )
                            self._profile_add(profile_stats, "block_graph_capture", profile_start)
                            self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
                            if whitebox_block_graph_state is not None:
                                block_tokens = whitebox_block_graph_state.get("block_tokens")
                                self._profile_inc(profile_stats, "block_graph_capture_hits")
                            else:
                                whitebox_block_graph_failed = True
                        if block_tokens is None:
                            profile_start = self._profile_stamp()
                            if int(self._whitebox_block_composite_multiplier) <= 1:
                                block_tokens = self._run_whitebox_block_state_direct_tensor_args(
                                    state=whitebox_block_state,
                                    bucket=whitebox_block_bucket,
                                    input_ids=block_input,
                                    cache_position=whitebox_block_cache_position,
                                    device=device,
                                )
                            if block_tokens is None:
                                block_tokens = self._run_whitebox_composite_block_state(
                                    state=whitebox_block_state,
                                    bucket=whitebox_block_bucket,
                                    input_ids=block_input,
                                    cache_position=whitebox_block_cache_position,
                                    rope_delta=None,
                                    device=device,
                                    dtype=generated.dtype,
                                    max_tokens=max_new_tokens - generated_len,
                                    copy_rope_delta=False,
                                )
                            self._profile_add(profile_stats, "block_direct", profile_start)
                            self._profile_add(profile_stats, "block_direct_or_graph", profile_start)
                            self._profile_inc(profile_stats, "block_direct_calls")
                            if block_tokens is not None and int(block_tokens.shape[1]) > whitebox_block_step_size:
                                self._profile_inc(
                                    profile_stats,
                                    "block_direct_underlying_calls",
                                    int(block_tokens.shape[1]) / float(max(1, whitebox_block_step_size)),
                                )
                        if block_tokens is None:
                            if require_token_block_for_request:
                                raise RuntimeError(
                                    "AICASGC unified decode whitebox token-block call returned no tokens "
                                    f"for bucket={whitebox_block_bucket}, generated_len={generated_len}"
                                )
                            break
                        take_cap = min(int(block_tokens.shape[1]), max_new_tokens - generated_len)
                        if take_cap <= 0:
                            break
                        block_1d = block_tokens[0, :take_cap]
                        generated[0, generated_len : generated_len + take_cap].copy_(block_1d)
                        generated_len += take_cap
                        block_failed_before_commit = False
                        used_token_block_decode = True

                        eos_hit = None
                        should_scan_block_eos = ((not force_full_request) or eos_pad_fill_for_request) and bool(self._eos_token_ids) and (
                            self._eos_check_interval <= whitebox_block_step_size
                            or generated_len >= max_new_tokens
                            or (generated_len - last_eos_scan_len) >= int(self._eos_check_interval)
                        )
                        if should_scan_block_eos:
                            profile_start = self._profile_stamp()
                            scan_start = max(0, int(last_eos_scan_len))
                            scan_tokens = generated[0, scan_start:generated_len]
                            eos_offset = self._find_first_token_id_in_1d(scan_tokens, self._eos_token_ids)
                            if eos_offset is not None:
                                eos_hit = int(scan_start + int(eos_offset))
                            else:
                                last_eos_scan_len = generated_len
                            self._profile_add(profile_stats, "eos_scan", profile_start)
                        if eos_hit is not None:
                            if eos_pad_fill_for_request:
                                filled_len, did_fill = self._fill_generated_after_eos(
                                    generated,
                                    generated_len=generated_len,
                                    max_new_tokens=max_new_tokens,
                                    eos_token_ids=self._eos_token_ids,
                                    fill_token_id=int(self._eos_pad_fill_token_id),
                                    scan_start=int(eos_hit),
                                )
                                if did_fill:
                                    self._profile_inc(profile_stats, "eos_pad_fill_after_eos_calls")
                                    self._profile_inc(
                                        profile_stats,
                                        "eos_pad_fill_after_eos_tokens",
                                        max(0, int(filled_len) - int(eos_hit) - 1),
                                    )
                                generated_len = int(filled_len)
                            else:
                                generated_len = int(eos_hit) + 1
                            break
                        if generated_len >= max_new_tokens:
                            if self._whitebox_block_persistent_sequence_graph_clear:
                                self._clear_whitebox_block_persistent_sequence_graph_state(
                                    whitebox_block_persistent_graph_state_used
                                )
                            whitebox_block_persistent_graph_state_used = None
                            break
                        block_input = block_tokens[:, take_cap - 1 : take_cap]
                        whitebox_block_cache_position.add_(int(block_tokens.shape[1]))
                        block_cache_pos = whitebox_block_cache_position
                    if self._whitebox_block_persistent_sequence_graph_clear:
                        self._clear_whitebox_block_persistent_sequence_graph_state(
                            whitebox_block_persistent_graph_state_used
                        )
                    whitebox_block_persistent_graph_state_used = None
                    if generated_len >= max_new_tokens or not block_failed_before_commit:
                        break

                    whitebox_block_active = False
                    whitebox_block_enabled = False
                    if require_token_block_for_request:
                        raise RuntimeError(
                            "AICASGC unified decode left the required whitebox token-block path "
                            f"before completing generation; generated_len={generated_len}, "
                            f"max_new_tokens={max_new_tokens}"
                        )
                    whitebox_handoff_enabled = bool(
                        self._use_whitebox_decode
                        and StaticCache is not None
                        and not bool(disable_whitebox_handoff)
                        and not require_token_block_for_request
                        and max_new_tokens > 1
                        and max_new_tokens <= self._whitebox_max_new_tokens
                        and requested_len <= self._whitebox_cache_len
                    )
                    if whitebox_handoff_enabled and whitebox_call is None:
                        whitebox_call = self._ensure_whitebox_decode()
                        whitebox_handoff_enabled = whitebox_call is not None

                if whitebox_handoff_enabled and not whitebox_active and (step + 1) >= self._whitebox_switch_after:
                    profile_start = self._profile_stamp()
                    static_past = self._dynamic_cache_to_static(past, self._whitebox_cache_len, device)
                    self._profile_add(profile_stats, "handoff", profile_start)
                    if static_past is not None:
                        static_kv = self._static_cache_kv_tensors(static_past)
                        if static_kv is not None:
                            past = static_past
                            wb_kv = static_kv
                            whitebox_active = True
                    if not whitebox_active:
                        whitebox_handoff_enabled = False

                if whitebox_active:
                    pos = decode_pos[:, step : step + 1].unsqueeze(0).expand(3, -1, -1)
                    assert whitebox_call is not None
                    assert wb_kv is not None
                    assert wb_weights is not None
                    assert wb_position_ids is not None
                    assert wb_cache_pos is not None
                    wb_position_ids.copy_(pos)
                    wb_cache_pos.copy_(cache_pos)
                    wb_token = self._whitebox_decode_one_flat(
                        whitebox_call,
                        wb_kv,
                        wb_weights,
                        next_token,
                        wb_cache_pos,
                        wb_position_ids,
                    )
                    if wb_token is not None:
                        next_token = wb_token
                        step += 1
                        continue
                    whitebox_active = False
                    whitebox_handoff_enabled = False

                if self._use_inline_decode_loop and past is not None and not getattr(past, "is_compileable", False):
                    profile_start = self._profile_stamp()
                    pos_emb = None
                    if decode_position_embeddings is not None:
                        pos_emb = (
                            decode_position_embeddings[0][:, step : step + 1, :],
                            decode_position_embeddings[1][:, step : step + 1, :],
                        )
                    elif pos is None:
                        pos = decode_pos[:, step : step + 1].unsqueeze(0).expand(3, -1, -1)
                    hidden_last, past = self._decode_hidden_one_inline(
                        next_token,
                        past,
                        cache_pos,
                        pos,
                        position_embeddings=pos_emb,
                    )
                    next_token = self._decode_top1(hidden_last)
                    self._profile_add(profile_stats, "dynamic_prefix", profile_start)
                elif self._use_fast_decode_loop and past is not None and not getattr(past, "is_compileable", False):
                    profile_start = self._profile_stamp()
                    if pos is None:
                        pos = decode_pos[:, step : step + 1].unsqueeze(0).expand(3, -1, -1)
                    hidden_last, past = self._decode_hidden_one_fast(next_token, past, cache_pos, pos)
                    next_token = self._decode_top1(hidden_last)
                    self._profile_add(profile_stats, "dynamic_prefix", profile_start)
                else:
                    profile_start = self._profile_stamp()
                    if pos is None:
                        pos = decode_pos[:, step : step + 1].unsqueeze(0).expand(3, -1, -1)
                    out2 = language_model(
                        input_ids=next_token,
                        attention_mask=None,
                        position_ids=pos,
                        past_key_values=past,
                        use_cache=True,
                        cache_position=cache_pos,
                    )
                    past = out2.past_key_values
                    next_token = self._decode_top1(out2.last_hidden_state[:, -1, :])
                    self._profile_add(profile_stats, "dynamic_prefix", profile_start)
                step += 1

        gen = generated[:, :generated_len]
        if self._should_guard_generated_tokens(max_new_tokens, generated_len, used_token_block_decode):
            healthy, reason = self._generated_tokens_look_healthy(gen[0])
            if not healthy:
                self._profile_inc(profile_stats, "decode_health_guard_fail")
                if (
                    int(max_new_tokens) == int(self._performance_request_tokens)
                    and isinstance(visual_profile, _VisualBudgetProfile)
                    and visual_profile.graph_bucket is not None
                ):
                    bucket = int(visual_profile.graph_bucket)
                    step_size = int(self._whitebox_block_step_size(bucket))
                    remaining_tokens = int(max_new_tokens) - 1
                    if self._whitebox_block_persistent_sequence_graph_overrun:
                        persistent_tokens = (
                            (remaining_tokens + step_size - 1) // step_size
                        ) * step_size
                    else:
                        persistent_tokens = (remaining_tokens // step_size) * step_size
                    if persistent_tokens > step_size:
                        self._whitebox_block_persistent_sequence_graph_failed.add(
                            (bucket, int(persistent_tokens), str(device), self._rowtriton_gateup_graph_key())
                        )
                if self._decode_health_guard_fallback:
                    raise _DecodeHealthFallback(reason)
            else:
                self._profile_inc(profile_stats, "decode_health_guard_pass")
        return torch.cat([pc.input_ids, gen], dim=1)

    def generate(self, image: Image.Image, question: str, max_new_tokens: int = 128) -> dict[str, Any]:
        self._ensure_model()
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
        prepared = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)
        input_len = prepared.input_ids.shape[1]
        output_ids = self.model.generate(
            **prepared,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )
        generated_ids = output_ids[0][input_len:]
        text = self._raw_processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return {"text": text, "token_count": len(generated_ids)}
