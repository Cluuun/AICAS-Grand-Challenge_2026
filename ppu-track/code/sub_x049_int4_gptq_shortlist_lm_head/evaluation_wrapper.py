"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
import inspect
import json
import math
import os
from contextlib import contextmanager
from types import MethodType
from typing import Dict, Tuple
from PIL import Image, ImageStat
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import StaticCache


class VLMModel:
    """
    Participant optimization class - modify this to implement optimizations.

    Optimization Architecture:
    - Split optimizations into separate methods for isolation and testing
    - Enable/disable each optimization independently in __init__
    - Each optimization method can be tested individually

    Important Notes:
    1. Benchmark directly calls self.model.generate() for performance testing.
    2. Your optimizations should modify self.model or its operators via Monkey Patch.
    3. All optimizations are applied in __init__ by calling optimization methods.
    """

    _DEFAULT_IMAGE_SIZE_BUCKETS = "auto"

    _ENV_DEFAULTS = {
        "AICAS_CPP_RUNTIME_VERBOSE": "0",
        # 0: prebuilt-only (submission default), 1: force online rebuild (dev override).
        "AICAS_CPP_RUNTIME_FORCE_REBUILD": "0",
        "AICAS_ENABLE_CPP_RUNTIME_TRITON_ADD_RMSNORM": "1",
        "AICAS_ENABLE_CPP_TEXT_INTERLAYER_ADD_RMSNORM": "1",
        "AICAS_ENABLE_CPP_TEXT_FINAL_ADD_RMSNORM": "1",
        "AICAS_ENABLE_CPP_RUNTIME_TRITON_SWIGLU": "1",
        "AICAS_ENABLE_CPP_RUNTIME_TRITON_PREFILL_ROPE": "1",
        "AICAS_ENABLE_CPP_RUNTIME_TRITON_PREFILL_QK_ROPE": "1",
        "AICAS_ENABLE_CPP_RUNTIME_TRITON_PREFILL_KV_CACHE_WRITE": "1",
        "AICAS_ENABLE_CPP_RUNTIME_TRITON_RMSNORM": "1",
        "AICAS_TRITON_PREFILL_ADD_RMSNORM_BLOCK_SIZE": "0",
        "AICAS_TRITON_PREFILL_ADD_RMSNORM_NUM_WARPS": "0",
        "AICAS_TRITON_PREFILL_ADD_RMSNORM_NUM_STAGES": "0",
        "AICAS_TRITON_PREFILL_RMSNORM_BLOCK_SIZE": "0",
        "AICAS_TRITON_PREFILL_RMSNORM_NUM_WARPS": "0",
        "AICAS_TRITON_PREFILL_RMSNORM_NUM_STAGES": "0",
        "AICAS_TRITON_PREFILL_SWIGLU_BLOCK_COL": "0",
        "AICAS_TRITON_PREFILL_SWIGLU_NUM_WARPS": "0",
        "AICAS_TRITON_PREFILL_SWIGLU_NUM_STAGES": "0",
        "AICAS_TRITON_PREFILL_ROPE_BLOCK_D": "0",
        "AICAS_TRITON_PREFILL_ROPE_NUM_WARPS": "0",
        "AICAS_TRITON_PREFILL_ROPE_NUM_STAGES": "0",
        "AICAS_TRITON_PREFILL_QK_ROPE_BLOCK_D": "128",
        "AICAS_TRITON_PREFILL_QK_ROPE_NUM_WARPS": "1",
        "AICAS_TRITON_PREFILL_QK_ROPE_NUM_STAGES": "1",
        "AICAS_TRITON_PREFILL_KV_CACHE_WRITE_BLOCK_T": "8",
        "AICAS_TRITON_PREFILL_KV_CACHE_WRITE_BLOCK_D": "128",
        "AICAS_TRITON_PREFILL_KV_CACHE_WRITE_NUM_WARPS": "4",
        "AICAS_TRITON_PREFILL_KV_CACHE_WRITE_NUM_STAGES": "4",
        "AICAS_TRITON_VISION_ADD_LAYERNORM_BLOCK_SIZE": "0",
        "AICAS_TRITON_VISION_ADD_LAYERNORM_NUM_WARPS": "0",
        "AICAS_TRITON_VISION_ADD_LAYERNORM_NUM_STAGES": "0",
        "AICAS_TRITON_VISION_ROTARY_BLOCK_D": "64",
        "AICAS_TRITON_VISION_ROTARY_NUM_WARPS": "1",
        "AICAS_TRITON_VISION_ROTARY_NUM_STAGES": "1",
        "AICAS_CUDA_GRAPH_DECODE_BUCKET_LARGE": "1024",
        "AICAS_CUDA_GRAPH_DECODE_BUCKET_SMALL": "128",
        "AICAS_CUDA_GRAPH_PREWARM_DECODE_BUCKETS": "128,1024",
        "AICAS_CUDA_GRAPH_PREFILL_BUCKETS": "64,128,256",
        "AICAS_ADAPTIVE_IMAGE_SEQ_BUCKETS": "160,176,192,208,224,240,256,272,288,305",
        "AICAS_CUDA_GRAPH_SEQ_BUCKET_BASE": "1024",
        "AICAS_CUDA_GRAPH_SEQ_BUCKET_GROWTH": "512",
        "AICAS_DISABLE_MIXED16_FUSIONS": "0",
        "AICAS_ENABLE_FP16_PREFILL_LAST_LOGIT_ONLY": "1",
        "AICAS_ENABLE_FP16_PREFILL_OPTIMIZATIONS": "1",
        "AICAS_ENABLE_PREFILL_CUDA_GRAPH": "1",
        "AICAS_ENABLE_PREFILL_CUDA_GRAPH_PREWARM": "1",
        "AICAS_ENABLE_ADAPTIVE_IMAGE_GRAPH_PREWARM": "1",
        "AICAS_ENABLE_FUSED_DECODE_POS_ADVANCE": "1",
        "AICAS_ENABLE_CACHED_DECODE_ROPE": "1",
        "AICAS_ENABLE_PREFILL_TEXT_MAX_BUCKET": "1",
        "AICAS_ENABLE_CPP_RUNTIME": "1",
        "AICAS_ENABLE_CPP_VISION_PATCH_LINEAR": "1",
        "AICAS_ENABLE_CPP_VISION_RUNTIME_LOOP": "1",
        "AICAS_ENABLE_CPP_VISION_SETUP_CACHE": "1",
        "AICAS_ENABLE_CUDA_GRAPH": "1",
        "AICAS_ENABLE_VISUAL_CUDA_GRAPH": "1",
        "AICAS_ENABLE_FIXED_IMAGE_RESHAPE": "1",
        "AICAS_ENABLE_FAST_CONNECTOR_SCATTER": "1",
        "AICAS_ENABLE_FUSED_PREFILL_ROPE_EMBED": "1",
        "AICAS_ENABLE_FUSED_VISION_ADD_LAYERNORM": "1",
        "AICAS_ENABLE_FUSED_VISION_INTERLAYER_ADD_LAYERNORM": "1",
        "AICAS_ENABLE_FUSED_VISION_ROPE": "1",
        "AICAS_ENABLE_IMAGE_RESHAPE": "1",
        "AICAS_ENABLE_IN_REPO_KERNELS": "1",
        "AICAS_ENABLE_DECODE_FP8_QUANT": "1",
        "AICAS_ENABLE_DECODE_FP8_QKV": "1",
        "AICAS_ENABLE_DECODE_FP8_GATE_UP_SWIGLU": "1",
        "AICAS_ENABLE_DECODE_FP8_O_ADD_RMSNORM": "1",
        "AICAS_ENABLE_DECODE_FP8_DOWN_ADD_RMSNORM": "1",
        "AICAS_ENABLE_DECODE_FP8_LM_HEAD": "1",
        "AICAS_ENABLE_CUDA_LM_HEAD_TOP1_FP8": "1",
        "AICAS_ENABLE_DECODE_INT4_LM_HEAD": "1",
        "AICAS_ENABLE_DECODE_FP8_KBLOCK_MAJOR": "1",
        "AICAS_ENABLE_DECODE_INT4_DOWN_ADD_RMSNORM": "1",
        "AICAS_ENABLE_DECODE_INT4_O_ADD_RMSNORM": "1",
        "AICAS_ENABLE_DECODE_GATE_UP_LINEAR_PACK": "1",
        "AICAS_ENABLE_DECODE_QKV_LINEAR_PACK": "1",
        "AICAS_CUDA_DECODE_QKV_FP8_ROWS_PER_BLOCK": "1",
        "AICAS_ENABLE_DECODE_INT4_QKV": "1",
        "AICAS_CUDA_DECODE_QKV_INT4_ROWS_PER_BLOCK": "1",
        "AICAS_DECODE_INT4_QKV_QUANT_POLICY": "gptq",
        "AICAS_DECODE_INT4_QKV_GPTQ_PACK_MODE": "warmup-online",
        "AICAS_DECODE_INT4_QKV_GPTQ_PACK_PATH": "",
        "AICAS_DECODE_INT4_QKV_GPTQ_ONLINE_SAMPLES": "128",
        "AICAS_DECODE_INT4_QKV_GPTQ_ONLINE_OUTPUT_PATH": "kernels/qkv_gptq_online.pt",
        "AICAS_ENABLE_CUDA_DECODE_QKV_QK_UPDATE": "1",
        "AICAS_ENABLE_CUDA_DECODE_QKV_QK_UPDATE_ROPE_CACHE": "0",
        "AICAS_CUDA_DECODE_QKV_QK_UPDATE_NUM_THREADS": "512",
        "AICAS_CUDA_DECODE_FP8_GATE_UP_SWIGLU_ROWS_PER_BLOCK": "2",
        "AICAS_ENABLE_DECODE_INT4_GATE_UP_SWIGLU": "1",
        "AICAS_DECODE_INT4_GATE_UP_QUANT_POLICY": "gptq",
        "AICAS_DECODE_INT4_GATE_UP_GPTQ_PACK_MODE": "warmup-online",
        "AICAS_DECODE_INT4_GATE_UP_GPTQ_PACK_PATH": "",
        "AICAS_DECODE_INT4_GATE_UP_GPTQ_ONLINE_SAMPLES": "128",
        "AICAS_DECODE_INT4_GATE_UP_GPTQ_ONLINE_OUTPUT_PATH": "kernels/gate_up_gptq_online.pt",
        "AICAS_DECODE_INT4_GATE_UP_GPTQ_DAMP_PERCENT": "1.0",
        "AICAS_DECODE_INT4_GPTQ_DAMP_PERCENT": "1.0",
        "AICAS_CUDA_DECODE_INT4_GATE_UP_SWIGLU_ROWS_PER_BLOCK": "4",
        "AICAS_ENABLE_PREFILL_NVTX": "0",
        "AICAS_ENABLE_PREFILL_SDPA_CAUSAL_FASTPATH": "1",
        "AICAS_ENABLE_TRITON_DECODE_ADD_RMSNORM": "1",
        "AICAS_ENABLE_TRITON_DECODE_INTERLAYER_ADD_RMSNORM": "1",
        "AICAS_CUDA_DECODE_DOWN_ADD_RMSNORM_REDUCE_THREADS": "0",
        "AICAS_CUDA_DECODE_O_ADD_RMSNORM_REDUCE_THREADS": "256",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_BLOCK_K": "128",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_BLOCK_N": "32",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_NUM_STAGES": "3",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_NUM_WARPS": "8",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_SPLIT_K": "6",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_BLOCK_K": "128",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_BLOCK_N": "16",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_NUM_STAGES": "3",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_NUM_WARPS": "8",
        "AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_SPLIT_K": "3",
        "AICAS_CUDA_DECODE_INT4_DOWN_ADD_RMSNORM_BLOCK_N": "4",
        "AICAS_CUDA_DECODE_INT4_DOWN_ADD_RMSNORM_SPLIT_K": "4",
        "AICAS_CUDA_DECODE_INT4_DOWN_ADD_RMSNORM_REDUCE_THREADS": "256",
        "AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_QUANT_POLICY": "gptq",
        "AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_PACK_MODE": "warmup-online",
        "AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_PACK_PATH": "",
        "AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_ONLINE_SAMPLES": "128",
        "AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_ONLINE_OUTPUT_PATH": "kernels/down_proj_gptq_online.pt",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_BLOCK_K": "128",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_BLOCK_N": "32",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_NUM_STAGES": "3",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_NUM_WARPS": "8",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_SPLIT_K": "4",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_BLOCK_K": "128",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_BLOCK_N": "16",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_NUM_STAGES": "4",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_NUM_WARPS": "8",
        "AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_SPLIT_K": "4",
        "AICAS_ENABLE_CUDA_DECODE_PREFIX_STAGE2_O_FUSION": "1",
        "AICAS_CUDA_DECODE_INT4_O_ADD_RMSNORM_BLOCK_N": "2",
        "AICAS_CUDA_DECODE_INT4_O_ADD_RMSNORM_SPLIT_K": "2",
        "AICAS_CUDA_DECODE_INT4_O_ADD_RMSNORM_REDUCE_THREADS": "256",
        "AICAS_DECODE_INT4_O_ADD_RMSNORM_QUANT_POLICY": "gptq",
        "AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_PACK_MODE": "warmup-online",
        "AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_PACK_PATH": "",
        "AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_ONLINE_SAMPLES": "128",
        "AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_ONLINE_OUTPUT_PATH": "kernels/o_proj_gptq_online.pt",
        "AICAS_ENABLE_LM_HEAD_SHORTLIST": "1",
        "AICAS_LM_HEAD_SHORTLIST_PATH": "kernels/lm_head_active_vocab_5000x1024.json",
        "AICAS_TRITON_LM_HEAD_TOP1_BLOCK_K": "256",
        "AICAS_TRITON_LM_HEAD_TOP1_BLOCK_N": "16",
        "AICAS_TRITON_LM_HEAD_TOP1_NUM_STAGES": "4",
        "AICAS_TRITON_LM_HEAD_TOP1_NUM_WARPS": "8",
        "AICAS_TRITON_LM_HEAD_TOP1_REDUCE_BLOCK": "512",
        "AICAS_TRITON_LM_HEAD_TOP1_REDUCE_NUM_WARPS": "8",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_BLOCK_K": "128",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_BLOCK_N": "32",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_NUM_STAGES": "3",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_NUM_WARPS": "4",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_REDUCE_BLOCK": "512",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_REDUCE_NUM_WARPS": "8",
        "AICAS_TRITON_LM_HEAD_TOP1_FP8_BLOCK_K": "128",
        "AICAS_TRITON_LM_HEAD_TOP1_FP8_BLOCK_N": "32",
        "AICAS_TRITON_LM_HEAD_TOP1_FP8_NUM_STAGES": "3",
        "AICAS_TRITON_LM_HEAD_TOP1_FP8_NUM_WARPS": "4",
        "AICAS_TRITON_LM_HEAD_TOP1_FP8_REDUCE_BLOCK": "1024",
        "AICAS_TRITON_LM_HEAD_TOP1_FP8_REDUCE_NUM_WARPS": "8",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_FP8_BLOCK_K": "128",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_FP8_BLOCK_N": "32",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_FP8_NUM_STAGES": "4",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_FP8_NUM_WARPS": "4",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_FP8_REDUCE_BLOCK": "512",
        "AICAS_TRITON_LM_HEAD_SHORTLIST_FP8_REDUCE_NUM_WARPS": "8",
        "AICAS_DECODE_INT4_LM_HEAD_QUANT_POLICY": "gptq",
        "AICAS_DECODE_INT4_LM_HEAD_GPTQ_PACK_MODE": "warmup-online",
        "AICAS_DECODE_INT4_LM_HEAD_GPTQ_PACK_PATH": "",
        "AICAS_DECODE_INT4_LM_HEAD_GPTQ_ONLINE_SAMPLES": "128",
        "AICAS_DECODE_INT4_LM_HEAD_GPTQ_ONLINE_OUTPUT_PATH": "kernels/lm_head_gptq_online.pt",
        "AICAS_ENABLE_DECODE_STATIC_ATTENTION_MASK": "0",
        "AICAS_ENABLE_DECODE_PREFIX_ATTENTION": "1",
        "AICAS_DECODE_PREFIX_ATTENTION_BLOCK_M": "128",
        "AICAS_DECODE_PREFIX_ATTENTION_NUM_STAGES": "4",
        "AICAS_DECODE_PREFIX_ATTENTION_NUM_WARPS": "8",
        "AICAS_CUDA_DECODE_PREFIX_ATTENTION_NUM_THREADS": "512",
        "AICAS_CUDA_DECODE_PREFIX_ATTENTION_STAGE2_DIM_TILE": "32",
        "AICAS_TRITON_DECODE_QK_RMSNORM_ROPE_BLOCK_D": "128",
        "AICAS_TRITON_DECODE_QK_RMSNORM_ROPE_NUM_STAGES": "2",
        "AICAS_TRITON_DECODE_QK_RMSNORM_ROPE_NUM_WARPS": "1",
        "AICAS_TRITON_DECODE_QK_KV_UPDATE_BLOCK_D": "128",
        "AICAS_TRITON_DECODE_QK_KV_UPDATE_NUM_STAGES": "1",
        "AICAS_TRITON_DECODE_QK_KV_UPDATE_NUM_WARPS": "8",
        "AICAS_ENABLE_TTFT_KV_REUSE": "1",
        "AICAS_ENABLE_VISUAL_GET_IMAGE_FEATURES": "1",
        "AICAS_PREFILL_TEXT_EXTRA_MAX": "49",
        "AICAS_STRICT_BASELINE": "0",
        "AICAS_IMAGE_RESHAPE_MODE": "long_edge",
        "AICAS_IMAGE_SIZE_BUCKETS": _DEFAULT_IMAGE_SIZE_BUCKETS,
        "AICAS_IMAGE_BUCKET_MIN_EDGE": "160",
        "AICAS_IMAGE_BUCKET_WARMUP_MIN_EDGE": "32",
        "AICAS_IMAGE_BUCKET_STEP": "32",
        "AICAS_ALLOW_LONG_EDGE_BUCKET_UPSCALE": "1",
        "AICAS_LONG_EDGE_ALIGN_MODE": "nearest",
        "AICAS_ENABLE_DYNAMIC_IMAGE_LONG_EDGE_CAP": "0",
        "AICAS_DYNAMIC_IMAGE_LONG_EDGE_SMALL": "512",
        "AICAS_DYNAMIC_IMAGE_LONG_EDGE_MEDIUM": "544",
        "AICAS_DYNAMIC_IMAGE_SMALL_MAX_TOKENS": "180",
        "AICAS_DYNAMIC_IMAGE_MEDIUM_MAX_TOKENS": "0",
        "AICAS_DYNAMIC_IMAGE_PREWARM_EDGES": "",
        "AICAS_ENABLE_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP": "0",
        "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_LONG_EDGE": "512",
        "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_MAX_MEAN_B": "0",
        "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_PIL_MAX_ENTROPY": "0",
        "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_ENTROPY_SIDE": "128",
        "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_MAX_ENTROPY_SMALL": "0",
        "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP_SMALL": "256",
        "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_MAX_ENTROPY_MEDIUM": "0",
        "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP_MEDIUM": "512",
        "AICAS_VISION_BASE_MAX_PIXELS": str(192 * 192),
        "AICAS_VISION_BASE_MIN_PIXELS": "0",
        "AICAS_VISION_FIXED_HEIGHT": "576",
        "AICAS_VISION_FIXED_WIDTH": "576",
    }

    @classmethod
    def _initialize_default_env(cls):
        for key, value in cls._ENV_DEFAULTS.items():
            os.environ.setdefault(key, value)

        cpp_runtime_enabled = os.environ["AICAS_ENABLE_CPP_RUNTIME"] == "1"
        os.environ.setdefault(
            "AICAS_ENABLE_PREFILL_CONTIG_COPY_UPDATE",
            "1" if cpp_runtime_enabled else "0",
        )
        os.environ.setdefault(
            "AICAS_CPP_RUNTIME_VALIDATE",
            "0" if cpp_runtime_enabled else "1",
        )

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.environ[name] == "1"

    @staticmethod
    def _env_int(name: str) -> int:
        return int(os.environ[name])

    @staticmethod
    def _env_float(name: str) -> float:
        return float(os.environ[name])

    @staticmethod
    def _env_int_or_default(name: str, default: int) -> int:
        value = int(os.environ[name])
        return default if value <= 0 else value

    @staticmethod
    def _decode_int4_quant_policy_value(value: str) -> str:
        policy = str(value).strip().lower().replace("_", "-")
        policy = {
            "default": "gptq",
            "gptq-sym": "gptq",
            "sym-gptq": "gptq",
        }.get(policy, policy)
        if policy != "gptq":
            raise ValueError("INT4 quant policy must be 'gptq'")
        return policy

    @staticmethod
    def _decode_int4_quant_policy(env_name: str) -> str:
        policy = VLMModel._decode_int4_quant_policy_value(os.environ[env_name])
        if policy != "gptq":
            raise ValueError(f"{env_name} must be 'gptq'")
        return policy

    @staticmethod
    def _decode_int4_gate_up_quant_policy() -> str:
        return VLMModel._decode_int4_quant_policy("AICAS_DECODE_INT4_GATE_UP_QUANT_POLICY")

    @staticmethod
    def _decode_int4_gate_up_gptq_pack_mode() -> str:
        mode = os.environ["AICAS_DECODE_INT4_GATE_UP_GPTQ_PACK_MODE"].strip().lower().replace("_", "-")
        mode = {
            "default": "warmup-online",
            "online": "warmup-online",
            "warmup": "warmup-online",
            "warmup-online-generate": "warmup-online",
            "static": "prebuilt",
            "file": "prebuilt",
            "offline": "prebuilt",
        }.get(mode, mode)
        if mode not in {"warmup-online", "prebuilt"}:
            raise ValueError(
                "AICAS_DECODE_INT4_GATE_UP_GPTQ_PACK_MODE must be one of 'warmup-online' or 'prebuilt'"
            )
        return mode

    @staticmethod
    def _decode_int4_gptq_pack_mode(env_name: str) -> str:
        mode = os.environ[env_name].strip().lower().replace("_", "-")
        mode = {
            "default": "warmup-online",
            "online": "warmup-online",
            "warmup": "warmup-online",
            "warmup-online-generate": "warmup-online",
            "prebuilt-only": "prebuilt",
            "load": "prebuilt",
        }.get(mode, mode)
        if mode not in {"warmup-online", "prebuilt"}:
            raise ValueError(f"{env_name} must be one of 'warmup-online' or 'prebuilt'")
        return mode

    @staticmethod
    def _decode_int4_qkv_gptq_pack_mode() -> str:
        return VLMModel._decode_int4_gptq_pack_mode("AICAS_DECODE_INT4_QKV_GPTQ_PACK_MODE")

    @staticmethod
    def _decode_int4_down_gptq_pack_mode() -> str:
        return VLMModel._decode_int4_gptq_pack_mode("AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_PACK_MODE")

    @staticmethod
    def _decode_int4_o_gptq_pack_mode() -> str:
        return VLMModel._decode_int4_gptq_pack_mode("AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_PACK_MODE")

    @staticmethod
    def _decode_int4_lm_head_gptq_pack_mode() -> str:
        return VLMModel._decode_int4_gptq_pack_mode("AICAS_DECODE_INT4_LM_HEAD_GPTQ_PACK_MODE")

    @staticmethod
    def _cuda_decode_gemv_reduce_backend(name: str) -> str:
        fixed = {
            "AICAS_CUDA_DECODE_DOWN_ADD_RMSNORM_REDUCE_BACKEND": "cuda",
            "AICAS_CUDA_DECODE_O_ADD_RMSNORM_REDUCE_BACKEND": "triton",
            "AICAS_CUDA_DECODE_INT4_DOWN_ADD_RMSNORM_REDUCE_BACKEND": "cuda",
            "AICAS_CUDA_DECODE_INT4_O_ADD_RMSNORM_REDUCE_BACKEND": "triton",
        }
        if name not in fixed:
            raise ValueError(f"unknown decode GEMV reduce backend key: {name}")
        return fixed[name]

    def _fixed_vision_runtime_shape_enabled(self) -> bool:
        return (
            self._env_flag("AICAS_ENABLE_IMAGE_RESHAPE")
            and self._env_flag("AICAS_ENABLE_FIXED_IMAGE_RESHAPE")
            and self._image_reshape_mode() == "fixed"
        )

    def _shape_specific_vision_runtime_enabled(self) -> bool:
        return (
            self._env_flag("AICAS_ENABLE_IMAGE_RESHAPE")
            and self._env_flag("AICAS_ENABLE_FIXED_IMAGE_RESHAPE")
            and self._image_reshape_mode() in {"fixed", "long_edge", "bounded"}
        )

    @staticmethod
    def _image_reshape_mode() -> str:
        mode = os.environ.get("AICAS_IMAGE_RESHAPE_MODE", "fixed").strip().lower().replace("-", "_")
        mode = {
            "square": "fixed",
            "fixed_square": "fixed",
            "longedge": "long_edge",
            "longest_edge": "long_edge",
            "max_edge": "long_edge",
            "cap": "bounded",
            "bounded_no_upscale": "bounded",
            "no_upscale": "bounded",
            "smart": "pixel_budget",
            "default": "pixel_budget",
        }.get(mode, mode)
        if mode not in {"fixed", "long_edge", "bounded", "pixel_budget"}:
            raise ValueError(
                "AICAS_IMAGE_RESHAPE_MODE must be one of fixed, long_edge, bounded, or pixel_budget"
            )
        return mode

    @staticmethod
    def _long_edge_align_mode() -> str:
        mode = os.environ["AICAS_LONG_EDGE_ALIGN_MODE"].strip().lower().replace("-", "_")
        mode = {
            "floor": "down",
            "round_down": "down",
            "round": "nearest",
        }.get(mode, mode)
        if mode not in {"down", "nearest"}:
            raise ValueError("AICAS_LONG_EDGE_ALIGN_MODE must be one of down or nearest")
        return mode

    @staticmethod
    def _env_int_list(name: str) -> Tuple[int, ...]:
        values = []
        for item in os.environ[name].split(","):
            item = item.strip()
            if item:
                values.append(int(item))
        return tuple(values)

    @staticmethod
    def _env_hw_list(name: str) -> Tuple[Tuple[int, int], ...]:
        values = []
        for item in os.environ[name].split(","):
            item = item.strip().lower()
            if not item:
                continue
            if "x" not in item:
                raise ValueError(f"{name} entries must be formatted as HxW")
            h_str, w_str = item.split("x", 1)
            values.append((int(h_str), int(w_str)))
        return tuple(values)

    @staticmethod
    def _round_down_multiple(value: int, factor: int) -> int:
        value = int(value)
        factor = max(1, int(factor))
        return max(factor, (value // factor) * factor)

    @staticmethod
    def _round_nearest_multiple(value: float, factor: int) -> int:
        factor = max(1, int(factor))
        return max(factor, int(round(float(value) / factor)) * factor)

    @classmethod
    def _aspect_resize_hw(
        cls,
        height: int,
        width: int,
        max_height: int,
        max_width: int,
        factor: int,
        mode: str,
        long_edge_align_mode: str = "down",
    ) -> Tuple[int, int]:
        height = max(1, int(height))
        width = max(1, int(width))
        max_height = max(1, int(max_height))
        max_width = max(1, int(max_width))
        factor = max(1, int(factor))
        if mode == "long_edge":
            scale = min(max_height / height, max_width / width)
            if long_edge_align_mode == "nearest":
                out_h = cls._round_nearest_multiple(height * scale, factor)
                out_w = cls._round_nearest_multiple(width * scale, factor)
            else:
                out_h = cls._round_down_multiple(height * scale, factor)
                out_w = cls._round_down_multiple(width * scale, factor)
        elif mode == "bounded":
            scale = min(1.0, max_height / height, max_width / width)
            out_h = cls._round_down_multiple(height * scale, factor)
            out_w = cls._round_down_multiple(width * scale, factor)
        else:
            raise ValueError(f"unsupported aspect resize mode: {mode}")
        out_h = min(max_height, max(factor, out_h))
        out_w = min(max_width, max(factor, out_w))
        out_h = cls._round_down_multiple(out_h, factor)
        out_w = cls._round_down_multiple(out_w, factor)
        return int(out_h), int(out_w)

    @classmethod
    def _bucket_image_hw(
        cls,
        target_h: int,
        target_w: int,
        max_height: int,
        max_width: int,
        factor: int,
        buckets: Tuple[Tuple[int, int], ...],
        allow_upscale: bool = True,
    ) -> Tuple[int, int]:
        if not buckets:
            return int(target_h), int(target_w)
        target_h = max(1, int(target_h))
        target_w = max(1, int(target_w))
        target_ratio = target_w / target_h
        target_area = target_h * target_w
        candidates = []
        for raw_h, raw_w in buckets:
            h = cls._round_down_multiple(raw_h, factor)
            w = cls._round_down_multiple(raw_w, factor)
            if h > max_height or w > max_width:
                continue
            area = h * w
            if area <= 0:
                continue
            if not allow_upscale and area > target_area:
                continue
            ratio = w / h
            area_penalty = abs(area - target_area) / max(1, target_area)
            ratio_penalty = abs(math.log(max(ratio, 1e-6) / max(target_ratio, 1e-6)))
            candidates.append((ratio_penalty * 4.0 + area_penalty, area < target_area, h, w))
        if not candidates:
            return int(target_h), int(target_w)
        _, _, best_h, best_w = min(candidates)
        return int(best_h), int(best_w)

    def _dynamic_image_resize_bounds(
        self,
        height: int,
        width: int,
        fixed_height: int,
        fixed_width: int,
        factor: int,
        buckets: Tuple[Tuple[int, int], ...],
    ) -> Tuple[int, int]:
        if not self._env_flag("AICAS_ENABLE_DYNAMIC_IMAGE_LONG_EDGE_CAP"):
            return int(fixed_height), int(fixed_width)
        height = max(1, int(height))
        width = max(1, int(width))
        fixed_height = max(1, int(fixed_height))
        fixed_width = max(1, int(fixed_width))
        factor = max(1, int(factor))
        base_h, base_w = self._aspect_resize_hw(
            int(height),
            int(width),
            int(fixed_height),
            int(fixed_width),
            int(factor),
            self._image_reshape_mode(),
            self._long_edge_align_mode(),
        )
        base_h, base_w = self._bucket_image_hw(
            int(base_h),
            int(base_w),
            int(fixed_height),
            int(fixed_width),
            int(factor),
            buckets,
            allow_upscale=(
                self._image_reshape_mode() == "long_edge"
                and self._env_flag("AICAS_ALLOW_LONG_EDGE_BUCKET_UPSCALE")
            ),
        )
        visual_tokens = max(1, int(base_h) // factor) * max(1, int(base_w) // factor)
        cap = max(int(fixed_height), int(fixed_width))
        small_tokens = self._env_int("AICAS_DYNAMIC_IMAGE_SMALL_MAX_TOKENS")
        medium_tokens = self._env_int("AICAS_DYNAMIC_IMAGE_MEDIUM_MAX_TOKENS")
        if small_tokens > 0 and visual_tokens <= small_tokens:
            cap = min(cap, self._env_int("AICAS_DYNAMIC_IMAGE_LONG_EDGE_SMALL"))
        elif medium_tokens > 0 and visual_tokens <= medium_tokens:
            cap = min(cap, self._env_int("AICAS_DYNAMIC_IMAGE_LONG_EDGE_MEDIUM"))
        cap = self._round_down_multiple(max(factor, cap), factor)
        return int(min(fixed_height, cap)), int(min(fixed_width, cap))

    def _simple_visual_dynamic_cap(self, images) -> int:
        if not self._env_flag("AICAS_ENABLE_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP"):
            return 0
        image = images
        while isinstance(image, (list, tuple)) and len(image) == 1:
            image = image[0]
        max_entropy_small = self._env_float("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_MAX_ENTROPY_SMALL")
        max_entropy_medium = self._env_float("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_MAX_ENTROPY_MEDIUM")
        if max_entropy_small > 0.0 or max_entropy_medium > 0.0:
            gray_entropy = self._simple_visual_gray_entropy(image)
            if gray_entropy < 0.0:
                return 0
            if max_entropy_small > 0.0 and gray_entropy <= max_entropy_small:
                return max(0, int(self._env_int("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP_SMALL")))
            if max_entropy_medium > 0.0 and gray_entropy <= max_entropy_medium:
                return max(0, int(self._env_int("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP_MEDIUM")))
            return 0
        max_mean_b = self._env_float("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_MAX_MEAN_B")
        if max_mean_b <= 0.0:
            return 0
        rgb = None
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
            mean_b = float(ImageStat.Stat(rgb).mean[2]) / 255.0
        elif torch.is_tensor(image) and image.ndim == 3:
            tensor = image.detach()
            if tensor.shape[0] in (1, 3):
                if tensor.shape[0] < 3:
                    return 0
                blue = tensor[2]
            elif tensor.shape[-1] in (1, 3):
                if tensor.shape[-1] < 3:
                    return 0
                blue = tensor[..., 2]
            else:
                return 0
            mean_b = float(blue.to(dtype=torch.float32).mean().item())
            if mean_b > 2.0:
                mean_b = mean_b / 255.0
        else:
            return 0
        if mean_b > max_mean_b:
            return 0
        pil_max_entropy = self._env_float("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_PIL_MAX_ENTROPY")
        if pil_max_entropy > 0.0:
            if rgb is None:
                return 0
            hist = rgb.convert("L").histogram()
            total = int(sum(hist))
            if total > 0:
                inv_total = 1.0 / float(total)
                gray_entropy = -sum(
                    (float(count) * inv_total) * math.log2(float(count) * inv_total)
                    for count in hist
                    if count > 0
                ) / 8.0
                if gray_entropy > pil_max_entropy:
                    return 0
        return max(0, int(self._env_int("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_LONG_EDGE")))

    def _simple_visual_gray_entropy(self, image) -> float:
        side = max(16, int(self._env_int("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_ENTROPY_SIDE")))
        if isinstance(image, Image.Image):
            gray = image.convert("L")
            w, h = gray.size
            scale = min(1.0, float(side) / float(max(w, h)))
            if scale < 1.0:
                gray = gray.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.BICUBIC)
            hist = gray.histogram()
            total = int(sum(hist))
            if total <= 0:
                return -1.0
            inv_total = 1.0 / float(total)
            return -sum(
                (float(count) * inv_total) * math.log2(float(count) * inv_total)
                for count in hist
                if count > 0
            ) / 8.0
        if not torch.is_tensor(image) or image.ndim != 3:
            return -1.0
        tensor = image.detach().to(dtype=torch.float32)
        if tensor.shape[0] in (1, 3):
            if tensor.shape[0] < 3:
                return -1.0
            red, green, blue = tensor[0], tensor[1], tensor[2]
        elif tensor.shape[-1] in (1, 3):
            if tensor.shape[-1] < 3:
                return -1.0
            red, green, blue = tensor[..., 0], tensor[..., 1], tensor[..., 2]
        else:
            return -1.0
        if float(torch.maximum(torch.maximum(red.max(), green.max()), blue.max()).item()) > 2.0:
            red = red / 255.0
            green = green / 255.0
            blue = blue / 255.0
        gray = red * 0.299 + green * 0.587 + blue * 0.114
        height, width = int(gray.shape[-2]), int(gray.shape[-1])
        stride = max(1, int(math.floor(float(max(height, width)) / float(side))))
        if stride > 1:
            gray = gray[::stride, ::stride]
        hist = torch.histc(gray.clamp(0.0, 1.0), bins=32, min=0.0, max=1.0)
        total = hist.sum()
        if float(total.item()) <= 0.0:
            return -1.0
        prob = hist / total
        prob = prob[prob > 0.0]
        return float((-(prob * torch.log2(prob)).sum() / math.log2(32.0)).item())

    @classmethod
    def _fixed_square_from_pixels(cls, pixels: int, factor: int) -> int:
        pixels = int(pixels)
        factor = max(1, int(factor))
        if pixels <= 0:
            return 0
        return cls._round_down_multiple(int(pixels ** 0.5), factor)

    @staticmethod
    def _cuda_stream_is_capturing() -> bool:
        if not torch.cuda.is_available():
            return False
        checker = getattr(torch.cuda, "is_current_stream_capturing", None)
        if checker is None:
            return False
        try:
            return bool(checker())
        except Exception:
            return False

    def _get_single_image_grid_meta(self, owner, image_grid_thw):
        if image_grid_thw is None or not torch.is_tensor(image_grid_thw):
            return None
        exact_key = (
            int(image_grid_thw.data_ptr()),
            str(image_grid_thw.device),
            tuple(int(x) for x in image_grid_thw.shape),
            str(image_grid_thw.dtype),
        )
        cached = getattr(owner, "_aicas_cached_single_image_grid_meta", None)
        if cached is not None and cached[0] == exact_key:
            return cached[1]

        shape_key = (
            str(image_grid_thw.device),
            tuple(int(x) for x in image_grid_thw.shape),
            str(image_grid_thw.dtype),
        )
        shape_cached = getattr(owner, "_aicas_cached_single_image_grid_shape_meta", None)
        if (
            self._fixed_vision_runtime_shape_enabled()
            and shape_cached is not None
            and shape_cached[0] == shape_key
        ):
            meta = shape_cached[1]
            owner._aicas_cached_single_image_grid_meta = (exact_key, meta)
            return meta

        if self._cuda_stream_is_capturing():
            raise RuntimeError("single-image grid meta must be cached before CUDA Graph capture")

        grid_row_host = image_grid_thw[0].detach().to(device="cpu", dtype=torch.int64)
        meta = tuple(int(v) for v in grid_row_host.tolist())
        if len(meta) != 3:
            raise ValueError(f"expected single-image grid_thw with 3 values, got {meta}")
        owner._aicas_cached_single_image_grid_meta = (exact_key, meta)
        if self._fixed_vision_runtime_shape_enabled():
            owner._aicas_cached_single_image_grid_shape_meta = (shape_key, meta)
        return meta

    def _get_single_image_token_span(
        self,
        owner,
        input_ids: torch.Tensor,
        image_token_id: int,
        seq_len: int,
        n_image_tokens: int,
    ) -> int:
        key = (
            str(input_ids.device),
            str(input_ids.dtype),
            int(seq_len),
            int(n_image_tokens),
            int(image_token_id),
        )
        cached = getattr(owner, "_aicas_cached_single_image_token_span", None)
        if cached is not None and cached[0] == key:
            return int(cached[1])

        if self._cuda_stream_is_capturing():
            raise RuntimeError("single-image token span must be cached before CUDA Graph capture")

        input_row_cpu = input_ids[0].detach().to(device="cpu", dtype=torch.long)
        nz_cpu = torch.nonzero(input_row_cpu == int(image_token_id), as_tuple=False).view(-1)
        if int(nz_cpu.numel()) != int(n_image_tokens):
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {int(nz_cpu.numel())}, "
                f"features {int(n_image_tokens)}"
            )
        if int(n_image_tokens) > 0:
            start = int(nz_cpu[0].item())
            if int(nz_cpu[-1].item()) - int(start) + 1 != int(n_image_tokens):
                raise ValueError("Image token span must be contiguous for connector fastpath.")
        else:
            start = 0
        owner._aicas_cached_single_image_token_span = (key, int(start))
        return int(start)

    def _decode_int4_gate_up_runtime_ready(self) -> bool:
        return self._decode_int4_gate_up_gptq_online_ready

    def _decode_int4_qkv_runtime_ready(self) -> bool:
        return self._decode_int4_qkv_gptq_online_ready

    def _decode_int4_down_runtime_ready(self) -> bool:
        return self._decode_int4_down_gptq_online_ready

    def _decode_int4_o_runtime_ready(self) -> bool:
        return self._decode_int4_o_gptq_online_ready

    @staticmethod
    def _require_gptq_int4_runtime(feature_name: str, enabled: bool, quant_policy: str) -> None:
        if enabled and quant_policy != "gptq":
            raise ValueError(f"{feature_name} only supports GPTQ INT4 in evaluation_wrapper")

    @staticmethod
    def _repo_relative_path(path: str) -> str:
        path = path.strip()
        if path and not os.path.isabs(path):
            return os.path.join(os.path.dirname(__file__), path)
        return path

    def _load_gptq_prebuilt_payload(self, env_name: str) -> dict:
        pack_path = self._repo_relative_path(os.environ[env_name])
        if not pack_path:
            raise ValueError(f"{env_name} is required when the matching GPTQ_PACK_MODE is prebuilt")
        payload = torch.load(pack_path, map_location=self._device)
        if not isinstance(payload, dict):
            raise ValueError(f"{env_name} must point to a GPTQ payload dict")
        return payload

    def _reset_decode_graph_state_after_gate_up_pack_update(self) -> None:
        self._cuda_graph_cache = {}
        self._last_generation_ctx = None
        self._prefill_cuda_graph_prewarm_done = False
        self._adaptive_image_graph_prewarm_done = False

    def _gate_up_mlp_modules(self) -> list[torch.nn.Module]:
        modules = []
        for module in self._model.modules():
            if all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")):
                gate_weight = getattr(module.gate_proj, "weight", None)
                up_weight = getattr(module.up_proj, "weight", None)
                if (
                    gate_weight is not None
                    and up_weight is not None
                    and gate_weight.ndim == 2
                    and up_weight.ndim == 2
                ):
                    modules.append(module)
        return modules

    def _attention_modules(self) -> list[torch.nn.Module]:
        modules = []
        for module in self._model.modules():
            if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
                modules.append(module)
        return modules

    def _decoder_layers_with_down_proj(self) -> list[torch.nn.Module]:
        layers = []
        for module in self._model.modules():
            if hasattr(module, "mlp") and hasattr(module.mlp, "down_proj"):
                layers.append(module)
        return layers

    def _maybe_collect_qkv_gptq_online_activation(self, owner_module, x: torch.Tensor) -> None:
        if (
            not self._decode_int4_qkv_gptq_online_enabled
            or self._decode_int4_qkv_gptq_online_ready
            or not self._decode_int4_qkv_gptq_online_building
        ):
            return
        if not (
            torch.is_tensor(x)
            and x.is_cuda
            and x.dtype == torch.float16
            and x.ndim >= 2
            and int(x.shape[-1]) == 2048
        ):
            return
        layer_idx = int(getattr(owner_module, "_aicas_layer_idx", getattr(owner_module, "layer_idx", -1)))
        if layer_idx < 0:
            return
        target = self._env_int("AICAS_DECODE_INT4_QKV_GPTQ_ONLINE_SAMPLES")
        counts = getattr(self, "_decode_int4_qkv_gptq_online_counts", None)
        rows = getattr(self, "_decode_int4_qkv_gptq_online_rows", None)
        if counts is None or rows is None or counts.get(layer_idx, 0) >= target:
            return
        flat = x.detach().reshape(-1, int(x.shape[-1]))
        remaining = int(target) - int(counts.get(layer_idx, 0))
        take = min(int(remaining), int(flat.shape[0]))
        if take <= 0:
            return
        rows[layer_idx].append(flat[:take].clone().contiguous())
        counts[layer_idx] = int(counts.get(layer_idx, 0)) + int(take)

    def _maybe_collect_gate_up_gptq_online_activation(self, owner_module, x: torch.Tensor) -> None:
        if (
            not self._decode_int4_gate_up_gptq_online_enabled
            or self._decode_int4_gate_up_gptq_online_ready
            or not self._decode_int4_gate_up_gptq_online_building
        ):
            return
        if not (
            torch.is_tensor(x)
            and x.is_cuda
            and x.dtype == torch.float16
            and x.ndim >= 2
            and int(x.shape[-2]) == 1
        ):
            return
        layer_idx = int(getattr(owner_module, "_aicas_layer_idx", -1))
        if layer_idx < 0:
            return
        target = self._env_int("AICAS_DECODE_INT4_GATE_UP_GPTQ_ONLINE_SAMPLES")
        counts = getattr(self, "_decode_int4_gate_up_gptq_online_counts", None)
        rows = getattr(self, "_decode_int4_gate_up_gptq_online_rows", None)
        if counts is None or rows is None or counts.get(layer_idx, 0) >= target:
            return
        flat = x.detach().reshape(-1, int(x.shape[-1]))
        remaining = int(target) - int(counts.get(layer_idx, 0))
        take = min(int(remaining), int(flat.shape[0]))
        if take <= 0:
            return
        rows[layer_idx].append(flat[:take].clone().contiguous())
        counts[layer_idx] = int(counts.get(layer_idx, 0)) + int(take)

    def _maybe_collect_down_gptq_online_activation(self, owner_module, x: torch.Tensor) -> None:
        if (
            not self._decode_int4_down_gptq_online_enabled
            or self._decode_int4_down_gptq_online_ready
            or not self._decode_int4_down_gptq_online_building
        ):
            return
        if not (
            torch.is_tensor(x)
            and x.is_cuda
            and x.dtype == torch.float16
            and x.ndim >= 1
            and int(x.shape[-1]) == 6144
        ):
            return
        layer_idx = int(getattr(owner_module, "_aicas_layer_idx", -1))
        if layer_idx < 0:
            return
        target = self._env_int("AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_ONLINE_SAMPLES")
        counts = getattr(self, "_decode_int4_down_gptq_online_counts", None)
        rows = getattr(self, "_decode_int4_down_gptq_online_rows", None)
        if counts is None or rows is None or counts.get(layer_idx, 0) >= target:
            return
        flat = x.detach().reshape(-1, int(x.shape[-1]))
        remaining = int(target) - int(counts.get(layer_idx, 0))
        take = min(int(remaining), int(flat.shape[0]))
        if take <= 0:
            return
        rows[layer_idx].append(flat[:take].clone().contiguous())
        counts[layer_idx] = int(counts.get(layer_idx, 0)) + int(take)

    def _maybe_collect_o_gptq_online_activation(self, owner_module, x: torch.Tensor) -> None:
        if (
            not self._decode_int4_o_gptq_online_enabled
            or self._decode_int4_o_gptq_online_ready
            or not self._decode_int4_o_gptq_online_building
        ):
            return
        if not (
            torch.is_tensor(x)
            and x.is_cuda
            and x.dtype == torch.float16
            and x.ndim >= 1
            and int(x.shape[-1]) == 2048
        ):
            return
        layer_idx = int(getattr(owner_module, "_aicas_layer_idx", -1))
        if layer_idx < 0:
            return
        target = self._env_int("AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_ONLINE_SAMPLES")
        counts = getattr(self, "_decode_int4_o_gptq_online_counts", None)
        rows = getattr(self, "_decode_int4_o_gptq_online_rows", None)
        if counts is None or rows is None or counts.get(layer_idx, 0) >= target:
            return
        flat = x.detach().reshape(-1, int(x.shape[-1]))
        remaining = int(target) - int(counts.get(layer_idx, 0))
        take = min(int(remaining), int(flat.shape[0]))
        if take <= 0:
            return
        rows[layer_idx].append(flat[:take].clone().contiguous())
        counts[layer_idx] = int(counts.get(layer_idx, 0)) + int(take)

    def _maybe_collect_lm_head_gptq_online_activation(self, x: torch.Tensor) -> None:
        if (
            not self._decode_int4_lm_head_gptq_online_enabled
            or self._decode_int4_lm_head_gptq_online_ready
            or not self._decode_int4_lm_head_gptq_online_building
        ):
            return
        if not (
            torch.is_tensor(x)
            and x.is_cuda
            and x.dtype == torch.float16
            and x.ndim >= 2
            and int(x.shape[-1]) == 2048
        ):
            return
        target = self._env_int("AICAS_DECODE_INT4_LM_HEAD_GPTQ_ONLINE_SAMPLES")
        count = int(getattr(self, "_decode_int4_lm_head_gptq_online_count", 0))
        rows = getattr(self, "_decode_int4_lm_head_gptq_online_rows", None)
        if rows is None or count >= target:
            return
        flat = x.detach().reshape(-1, int(x.shape[-1]))
        remaining = int(target) - int(count)
        take = min(int(remaining), int(flat.shape[0]))
        if take <= 0:
            return
        rows.append(flat[:take].clone().contiguous())
        self._decode_int4_lm_head_gptq_online_count = count + int(take)

    def _mark_optimization(self, name: str):
        if name not in self._optimizations_applied:
            self._optimizations_applied.append(name)

    def _mixed16_feature_enabled(self, feature_key: str) -> bool:
        return (not self._mixed16_fusions_disabled) and self._env_flag(feature_key)

    def _fixed_shape_mixed16_feature_enabled(self, feature_key: str) -> bool:
        return self._fixed_vision_runtime_shape_enabled() and self._mixed16_feature_enabled(feature_key)

    def _shape_specific_mixed16_feature_enabled(self, feature_key: str) -> bool:
        return self._shape_specific_vision_runtime_enabled() and self._mixed16_feature_enabled(feature_key)

    def _adaptive_image_prefill_graph_enabled(self) -> bool:
        return (
            self._shape_specific_vision_runtime_enabled()
            and not self._fixed_vision_runtime_shape_enabled()
        )

    def _image_size_buckets(self, factor: int | None = None) -> Tuple[Tuple[int, int], ...]:
        raw_value = os.environ.get("AICAS_IMAGE_SIZE_BUCKETS", self._DEFAULT_IMAGE_SIZE_BUCKETS).strip().lower()
        if raw_value in {"", "auto", "full_union", "full-union"}:
            buckets = self._auto_full_union_image_size_buckets(factor)
            return buckets
        if raw_value in {"0", "none", "off", "disable", "disabled"}:
            return ()
        buckets = self._env_hw_list("AICAS_IMAGE_SIZE_BUCKETS")
        if factor is None:
            return buckets
        factor = max(1, int(factor))
        rounded = []
        for raw_h, raw_w in buckets:
            h = self._round_down_multiple(int(raw_h), factor)
            w = self._round_down_multiple(int(raw_w), factor)
            if h > 0 and w > 0:
                rounded.append((int(h), int(w)))
        return tuple(dict.fromkeys(rounded))

    def _auto_full_union_image_size_buckets(self, factor: int | None = None) -> Tuple[Tuple[int, int], ...]:
        factor = max(1, int(factor or self._env_int("AICAS_IMAGE_BUCKET_STEP")))
        max_h = self._round_down_multiple(self._env_int("AICAS_VISION_FIXED_HEIGHT"), factor)
        max_w = self._round_down_multiple(self._env_int("AICAS_VISION_FIXED_WIDTH"), factor)
        min_edge = self._round_down_multiple(
            min(
                self._env_int("AICAS_IMAGE_BUCKET_MIN_EDGE"),
                self._env_int("AICAS_IMAGE_BUCKET_WARMUP_MIN_EDGE"),
            ),
            factor,
        )
        if max_h <= 0 or max_w <= 0:
            return ()
        min_h = min(max_h, max(factor, min_edge))
        min_w = min(max_w, max(factor, min_edge))

        buckets = []
        edges = [(int(max_h), int(max_w))]
        if self._env_flag("AICAS_ENABLE_DYNAMIC_IMAGE_LONG_EDGE_CAP"):
            for value in self._env_int_list("AICAS_DYNAMIC_IMAGE_PREWARM_EDGES"):
                edge = self._round_down_multiple(int(value), factor)
                if edge >= factor:
                    edges.append((min(int(max_h), edge), min(int(max_w), edge)))
        if self._env_flag("AICAS_ENABLE_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP"):
            edge = self._round_down_multiple(self._env_int("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_LONG_EDGE"), factor)
            if edge >= factor:
                edges.append((min(int(max_h), edge), min(int(max_w), edge)))
            entropy_edges = (
                ("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_MAX_ENTROPY_SMALL", "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP_SMALL"),
                ("AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_MAX_ENTROPY_MEDIUM", "AICAS_DYNAMIC_IMAGE_SIMPLE_VISUAL_CAP_MEDIUM"),
            )
            for threshold_name, name in entropy_edges:
                if self._env_float(threshold_name) <= 0.0:
                    continue
                edge = self._round_down_multiple(self._env_int(name), factor)
                if edge >= factor:
                    edges.append((min(int(max_h), edge), min(int(max_w), edge)))
        for edge_h, edge_w in tuple(dict.fromkeys(edges)):
            local_min_h = min(edge_h, max(factor, min_edge))
            local_min_w = min(edge_w, max(factor, min_edge))
            for h in range(int(local_min_h), int(edge_h), int(factor)):
                buckets.append((int(h), int(edge_w)))
            for w in range(int(local_min_w), int(edge_w) + 1, int(factor)):
                buckets.append((int(edge_h), int(w)))
        return tuple(dict.fromkeys(buckets))

    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Initialize model and apply optimizations.

        Args:
            model_path: Qwen3-VL-2B-Instruct model path
            device: CUDA device, e.g., "cuda:0"
        """
        self._device = device
        self.model_path = model_path
        self._initialize_default_env()
        self._strict_baseline_mode = self._env_flag("AICAS_STRICT_BASELINE")

        # Track applied optimizations
        self._optimizations_applied = []

        # Load processor
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)

        # Load model in non-quantized fp16 mode.
        model_load_kwargs = {"dtype": torch.float16}
        if self._strict_baseline_mode:
            print("[VLMModel] Loading model in strict baseline mode (fp16, no optimization)...")
        else:
            print("[VLMModel] Loading model in fp16 mode (non-quantized fp16)...")

        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map=device,
            **model_load_kwargs,
        )
        self._model.eval()
        self._forward_arg_names = set(inspect.signature(self._model.forward).parameters.keys())

        # Initialize generation state fields for both baseline and optimized modes.
        self._cuda_graph_enabled = False
        self._prefill_cuda_graph_enabled = False
        self._prefill_cuda_graph_disabled_reason = None
        self._prefill_cuda_graph_direct_visual = False
        self._prefill_cuda_graph_prewarm_done = False
        self._adaptive_image_graph_prewarm_done = False
        self._visual_cuda_graph_enabled = False
        self._visual_cuda_graph_disabled_reason = None
        self._visual_cuda_graph_cache = {}
        self._cuda_graph_cache = {}
        self._original_generate = None
        self._prefill_graph_buckets = ()
        self._graph_decode_bucket_small = 0
        self._graph_decode_bucket_large = 0
        self._last_generation_ctx = None
        self._fused_decode_position_advance = None
        self._fused_decode_set_start = None
        self._cpp_runtime_module = None
        self._cpp_runtime_enabled = False
        self._cpp_runtime_triton_kernels = {}
        self._cpp_runtime_prefill_qk_rope_launcher = None
        self._cpp_runtime_prefill_qk_rope_function = None
        self._cpp_runtime_prefill_qk_rope_packed_metadata = None
        self._cpp_runtime_prefill_qk_rope_block_d = 0
        self._cpp_runtime_add_rmsnorm_launcher = None
        self._cpp_runtime_add_rmsnorm_function = None
        self._cpp_runtime_add_rmsnorm_packed_metadata = None
        self._cpp_runtime_add_rmsnorm_block_size = 0
        self._cpp_runtime_rmsnorm_launcher = None
        self._cpp_runtime_rmsnorm_function = None
        self._cpp_runtime_rmsnorm_packed_metadata = None
        self._cpp_runtime_rmsnorm_block_size = 0
        self._cpp_runtime_swiglu_launcher = None
        self._cpp_runtime_swiglu_function = None
        self._cpp_runtime_swiglu_packed_metadata = None
        self._cpp_runtime_swiglu_block_size = 0
        self._cpp_runtime_kv_cache_write_launcher = None
        self._cpp_runtime_kv_cache_write_function = None
        self._cpp_runtime_kv_cache_write_packed_metadata = None
        self._cpp_runtime_kv_cache_write_block_t = 0
        self._cpp_runtime_kv_cache_write_block_d = 0
        self._cpp_runtime_add_layernorm_launcher = None
        self._cpp_runtime_add_layernorm_function = None
        self._cpp_runtime_add_layernorm_packed_metadata = None
        self._cpp_runtime_add_layernorm_block_size = 0
        self._cpp_runtime_vision_rotary_launcher = None
        self._cpp_runtime_vision_rotary_function = None
        self._cpp_runtime_vision_rotary_packed_metadata = None
        self._cpp_runtime_vision_rotary_block_d = 0
        self._cpp_runtime_prefill_rope_launcher = None
        self._cpp_runtime_prefill_rope_function = None
        self._cpp_runtime_prefill_rope_packed_metadata = None
        self._cpp_runtime_prefill_rope_block_d = 0
        self._cpp_runtime_cuda_current_stream = None
        self._prefill_nvtx_enabled = self._env_flag("AICAS_ENABLE_PREFILL_NVTX")
        self._nvtx_prefill_active = False
        self._prefill_nvtx_hook_handles = []
        self._decode_static_attention_mask_enabled = self._env_flag("AICAS_ENABLE_DECODE_STATIC_ATTENTION_MASK")
        self._triton_lm_head_top1_enabled = False
        self._triton_lm_head_top1_fp8 = None
        self._triton_lm_head_shortlist_top1_fp8 = None
        self._lm_head_shortlist_top1_int4 = None
        self._lm_head_shortlist_token_ids = None
        self._lm_head_weight_fp8 = None
        self._lm_head_scales_fp8 = None
        self._lm_head_shortlist_weight_fp8 = None
        self._lm_head_shortlist_scales_fp8 = None
        self._lm_head_shortlist_weight_int4 = None
        self._lm_head_shortlist_scales_int4 = None
        self._create_lm_head_top1_workspace = None
        self._in_repo_kernels_enabled = self._env_flag("AICAS_ENABLE_IN_REPO_KERNELS")
        self._decode_fp8_quant_enabled = (
            self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_DECODE_FP8_QUANT")
        )
        self._decode_fp8_qkv_enabled = (
            self._decode_fp8_quant_enabled and self._env_flag("AICAS_ENABLE_DECODE_FP8_QKV")
        )
        self._decode_int4_qkv_enabled = (
            self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_DECODE_INT4_QKV")
        )
        self._decode_int4_qkv_quant_policy_value = self._decode_int4_quant_policy(
            "AICAS_DECODE_INT4_QKV_QUANT_POLICY"
        )
        self._decode_int4_qkv_gptq_pack_mode_value = self._decode_int4_qkv_gptq_pack_mode()
        self._decode_int4_qkv_gptq_online_enabled = (
            self._decode_int4_qkv_enabled
            and self._decode_int4_qkv_quant_policy_value == "gptq"
            and self._decode_int4_qkv_gptq_pack_mode_value == "warmup-online"
        )
        self._decode_int4_qkv_gptq_prebuilt_enabled = (
            self._decode_int4_qkv_enabled
            and self._decode_int4_qkv_quant_policy_value == "gptq"
            and self._decode_int4_qkv_gptq_pack_mode_value == "prebuilt"
        )
        self._decode_int4_qkv_gptq_online_ready = not (
            self._decode_int4_qkv_gptq_online_enabled or self._decode_int4_qkv_gptq_prebuilt_enabled
        )
        self._decode_int4_qkv_gptq_online_building = False
        self._decode_int4_qkv_gptq_online_payload = None
        self._decode_fp8_gate_up_enabled = (
            self._decode_fp8_quant_enabled and self._env_flag("AICAS_ENABLE_DECODE_FP8_GATE_UP_SWIGLU")
        )
        self._decode_int4_gate_up_enabled = (
            self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_DECODE_INT4_GATE_UP_SWIGLU")
        )
        self._decode_int4_gate_up_quant_policy_value = self._decode_int4_gate_up_quant_policy()
        self._decode_int4_gate_up_gptq_pack_mode_value = self._decode_int4_gate_up_gptq_pack_mode()
        self._decode_int4_gate_up_gptq_online_enabled = (
            self._decode_int4_gate_up_enabled
            and self._decode_int4_gate_up_quant_policy_value == "gptq"
            and self._decode_int4_gate_up_gptq_pack_mode_value == "warmup-online"
        )
        self._decode_int4_gate_up_gptq_prebuilt_enabled = (
            self._decode_int4_gate_up_enabled
            and self._decode_int4_gate_up_quant_policy_value == "gptq"
            and self._decode_int4_gate_up_gptq_pack_mode_value == "prebuilt"
        )
        self._decode_int4_gate_up_gptq_online_ready = not (
            self._decode_int4_gate_up_gptq_online_enabled or self._decode_int4_gate_up_gptq_prebuilt_enabled
        )
        self._decode_int4_gate_up_gptq_online_building = False
        self._decode_int4_gate_up_gptq_online_payload = None
        self._decode_fp8_o_add_rmsnorm_enabled = (
            self._decode_fp8_quant_enabled and self._env_flag("AICAS_ENABLE_DECODE_FP8_O_ADD_RMSNORM")
        )
        self._decode_fp8_down_add_rmsnorm_enabled = (
            self._decode_fp8_quant_enabled and self._env_flag("AICAS_ENABLE_DECODE_FP8_DOWN_ADD_RMSNORM")
        )
        self._decode_int4_down_add_rmsnorm_enabled = (
            self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_DECODE_INT4_DOWN_ADD_RMSNORM")
        )
        self._decode_int4_down_quant_policy_value = self._decode_int4_quant_policy(
            "AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_QUANT_POLICY"
        )
        self._decode_int4_down_gptq_pack_mode_value = self._decode_int4_down_gptq_pack_mode()
        self._decode_int4_down_gptq_online_enabled = (
            self._decode_int4_down_add_rmsnorm_enabled
            and self._decode_int4_down_quant_policy_value == "gptq"
            and self._decode_int4_down_gptq_pack_mode_value == "warmup-online"
        )
        self._decode_int4_down_gptq_prebuilt_enabled = (
            self._decode_int4_down_add_rmsnorm_enabled
            and self._decode_int4_down_quant_policy_value == "gptq"
            and self._decode_int4_down_gptq_pack_mode_value == "prebuilt"
        )
        self._decode_int4_down_gptq_online_ready = not (
            self._decode_int4_down_gptq_online_enabled or self._decode_int4_down_gptq_prebuilt_enabled
        )
        self._decode_int4_down_gptq_online_building = False
        self._decode_int4_down_gptq_online_payload = None
        self._decode_int4_o_add_rmsnorm_enabled = (
            self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_DECODE_INT4_O_ADD_RMSNORM")
        )
        self._decode_int4_o_quant_policy_value = self._decode_int4_quant_policy(
            "AICAS_DECODE_INT4_O_ADD_RMSNORM_QUANT_POLICY"
        )
        self._decode_int4_o_gptq_pack_mode_value = self._decode_int4_o_gptq_pack_mode()
        self._decode_int4_o_gptq_online_enabled = (
            self._decode_int4_o_add_rmsnorm_enabled
            and self._decode_int4_o_quant_policy_value == "gptq"
            and self._decode_int4_o_gptq_pack_mode_value == "warmup-online"
        )
        self._decode_int4_o_gptq_prebuilt_enabled = (
            self._decode_int4_o_add_rmsnorm_enabled
            and self._decode_int4_o_quant_policy_value == "gptq"
            and self._decode_int4_o_gptq_pack_mode_value == "prebuilt"
        )
        self._decode_int4_o_gptq_online_ready = not (
            self._decode_int4_o_gptq_online_enabled or self._decode_int4_o_gptq_prebuilt_enabled
        )
        self._decode_int4_o_gptq_online_building = False
        self._decode_int4_o_gptq_online_payload = None
        self._decode_fp8_lm_head_enabled = (
            self._decode_fp8_quant_enabled and self._env_flag("AICAS_ENABLE_DECODE_FP8_LM_HEAD")
        )
        self._cuda_lm_head_top1_fp8_enabled = (
            self._decode_fp8_lm_head_enabled and self._env_flag("AICAS_ENABLE_CUDA_LM_HEAD_TOP1_FP8")
        )
        self._decode_int4_lm_head_enabled = (
            self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_DECODE_INT4_LM_HEAD")
        )
        self._decode_int4_lm_head_quant_policy_value = self._decode_int4_quant_policy(
            "AICAS_DECODE_INT4_LM_HEAD_QUANT_POLICY"
        )
        self._decode_int4_lm_head_gptq_pack_mode_value = self._decode_int4_lm_head_gptq_pack_mode()
        self._decode_int4_lm_head_gptq_online_enabled = (
            self._decode_int4_lm_head_enabled
            and self._decode_int4_lm_head_quant_policy_value == "gptq"
            and self._decode_int4_lm_head_gptq_pack_mode_value == "warmup-online"
        )
        self._decode_int4_lm_head_gptq_prebuilt_enabled = (
            self._decode_int4_lm_head_enabled
            and self._decode_int4_lm_head_quant_policy_value == "gptq"
            and self._decode_int4_lm_head_gptq_pack_mode_value == "prebuilt"
        )
        self._decode_int4_lm_head_gptq_online_ready = not (
            self._decode_int4_lm_head_gptq_online_enabled or self._decode_int4_lm_head_gptq_prebuilt_enabled
        )
        self._decode_int4_lm_head_gptq_online_building = False
        self._decode_int4_lm_head_gptq_online_payload = None
        self._require_gptq_int4_runtime(
            "decode qkv INT4",
            self._decode_int4_qkv_enabled,
            self._decode_int4_qkv_quant_policy_value,
        )
        self._require_gptq_int4_runtime(
            "decode gate/up INT4",
            self._decode_int4_gate_up_enabled,
            self._decode_int4_gate_up_quant_policy_value,
        )
        self._require_gptq_int4_runtime(
            "decode down/add RMSNorm INT4",
            self._decode_int4_down_add_rmsnorm_enabled,
            self._decode_int4_down_quant_policy_value,
        )
        self._require_gptq_int4_runtime(
            "decode o/add RMSNorm INT4",
            self._decode_int4_o_add_rmsnorm_enabled,
            self._decode_int4_o_quant_policy_value,
        )
        self._require_gptq_int4_runtime(
            "decode lm_head INT4",
            self._decode_int4_lm_head_enabled,
            self._decode_int4_lm_head_quant_policy_value,
        )
        self._lm_head_top1_block_n = 0
        self._lm_head_top1_block_k = 0
        self._lm_head_top1_reduce_block = 0
        self._lm_head_top1_num_warps = 0
        self._lm_head_top1_num_stages = 0
        self._lm_head_top1_reduce_num_warps = 0
        self._lm_head_shortlist_block_n = 0
        self._lm_head_shortlist_block_k = 0
        self._lm_head_shortlist_reduce_block = 0
        self._lm_head_shortlist_num_warps = 0
        self._lm_head_shortlist_num_stages = 0
        self._lm_head_shortlist_reduce_num_warps = 0
        self._cpp_runtime_master_enabled = (
            self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_CPP_RUNTIME")
        )
        if self._model.dtype != torch.float16:
            raise RuntimeError(
                f"Official runtime requires fp16 model dtype, but got {self._model.dtype}"
            )
        self._mixed16_fusions_disabled = (
            (not self._in_repo_kernels_enabled) or self._env_flag("AICAS_DISABLE_MIXED16_FUSIONS")
        )
        self._enable_fp16_prefill_optimizations = (
            self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_FP16_PREFILL_OPTIMIZATIONS")
        )
        if self._strict_baseline_mode:
            print("[VLMModel] Strict baseline mode enabled: no kernel patch, no CUDA Graph.")
            print(f"[VLMModel] Model loaded successfully on {device}")
            return

        self._configure_processor_pixel_budget()
        self._init_cpp_runtime_bridge()
        self._enable_cpp_prefill_text_runtime_stub()
        self._enable_prefill_nvtx_annotations()
        self._enable_triton_decode_rmsnorm_rope()
        self._enable_triton_decode_add_rmsnorm()
        self._enable_decode_gate_up_linear_pack()
        self._enable_triton_lm_head_top1()
        self._install_gptq_prebuilt_payloads()
        self._enable_fused_decode_position_advance()
        self._enable_cached_decode_rope_embeddings()
        self._apply_fp16_prefill_optimizations()
        self._init_cpp_runtime_triton_handles()

        # CUDA Graph state (for max_new_tokens=1 path)
        # Enabled when running on CUDA.
        self._cuda_graph_enabled = (
            torch.cuda.is_available()  and str(device).startswith("cuda")
        )
        self._prefill_cuda_graph_enabled = (
            self._cuda_graph_enabled
            and self._env_flag("AICAS_ENABLE_PREFILL_CUDA_GRAPH")
        )
        self._graph_seq_bucket_base = self._env_int("AICAS_CUDA_GRAPH_SEQ_BUCKET_BASE")
        self._graph_seq_bucket_growth = self._env_int("AICAS_CUDA_GRAPH_SEQ_BUCKET_GROWTH")
        self._prefill_graph_buckets = tuple(sorted(set(self._env_int_list("AICAS_CUDA_GRAPH_PREFILL_BUCKETS"))))
        self._graph_decode_bucket_small = self._env_int("AICAS_CUDA_GRAPH_DECODE_BUCKET_SMALL")
        self._graph_decode_bucket_large = self._env_int("AICAS_CUDA_GRAPH_DECODE_BUCKET_LARGE")
        if not self._prefill_graph_buckets:
            raise ValueError("AICAS_CUDA_GRAPH_PREFILL_BUCKETS must contain at least one bucket")
        if any(bucket < 2 for bucket in self._prefill_graph_buckets):
            raise ValueError("AICAS_CUDA_GRAPH_PREFILL_BUCKETS values must be >= 2")
        if self._graph_seq_bucket_base < 2:
            raise ValueError("AICAS_CUDA_GRAPH_SEQ_BUCKET_BASE must be >= 2")
        if self._graph_seq_bucket_growth < 64:
            raise ValueError("AICAS_CUDA_GRAPH_SEQ_BUCKET_GROWTH must be >= 64")
        if self._graph_decode_bucket_small < 2:
            raise ValueError("AICAS_CUDA_GRAPH_DECODE_BUCKET_SMALL must be >= 2")
        if self._graph_decode_bucket_large < self._graph_decode_bucket_small:
            raise ValueError(
                "AICAS_CUDA_GRAPH_DECODE_BUCKET_LARGE must be >= AICAS_CUDA_GRAPH_DECODE_BUCKET_SMALL"
            )
        self._ttft_kv_reuse_enabled = self._env_flag("AICAS_ENABLE_TTFT_KV_REUSE")
        self._visual_cuda_graph_enabled = (
            torch.cuda.is_available()
            and str(device).startswith("cuda")
            and self._cpp_runtime_enabled
            and self._env_flag("AICAS_ENABLE_VISUAL_CUDA_GRAPH")
            and self._env_flag("AICAS_ENABLE_IMAGE_RESHAPE")
            and self._env_flag("AICAS_ENABLE_FIXED_IMAGE_RESHAPE")
            and self._shape_specific_vision_runtime_enabled()
            and self._env_flag("AICAS_ENABLE_CPP_VISION_RUNTIME_LOOP")
        )

        # Keep CUDA Graph generate path enabled.
        if not self._env_flag("AICAS_ENABLE_CUDA_GRAPH"):
            self._cuda_graph_enabled = False
            self._prefill_cuda_graph_enabled = False
        self._enable_cuda_graph_generate()

        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")

    # ================================================================
    # Optimization Methods - Implement your optimizations here
    # ================================================================

    def _configure_processor_pixel_budget(self):
        """Apply the configured single-image resize before benchmark tokenization."""
        if not self._env_flag("AICAS_ENABLE_IMAGE_RESHAPE"):
            return
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is None:
            return

        base_min_pixels = self._env_int("AICAS_VISION_BASE_MIN_PIXELS")
        base_max_pixels = self._env_int("AICAS_VISION_BASE_MAX_PIXELS")
        fixed_height = self._env_int("AICAS_VISION_FIXED_HEIGHT")
        fixed_width = self._env_int("AICAS_VISION_FIXED_WIDTH")
        reshape_mode = self._image_reshape_mode()
        if base_min_pixels < 0 or base_max_pixels < 0:
            raise ValueError("AICAS_VISION_BASE_*_PIXELS must be >= 0")
        if fixed_height < 0 or fixed_width < 0:
            raise ValueError("AICAS_VISION_FIXED_{HEIGHT,WIDTH} must be >= 0")
        if reshape_mode in {"long_edge", "bounded"} and fixed_height <= 0 and fixed_width <= 0:
            raise ValueError(
                "AICAS_VISION_FIXED_HEIGHT/WIDTH must provide the resize bounds for "
                "AICAS_IMAGE_RESHAPE_MODE=long_edge or bounded"
            )

        patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
        merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
        factor = max(1, patch_size * merge_size)

        if fixed_height <= 0 and fixed_width <= 0 and self._fixed_vision_runtime_shape_enabled():
            side = self._fixed_square_from_pixels(base_max_pixels, factor)
            if side > 0:
                fixed_height = side
                fixed_width = side

        if (fixed_height > 0) ^ (fixed_width > 0):
            raise ValueError("AICAS_VISION_FIXED_HEIGHT and AICAS_VISION_FIXED_WIDTH must be set together")

        if fixed_height > 0 and fixed_width > 0 and reshape_mode in {"fixed", "long_edge", "bounded"}:
            fixed_height = self._round_down_multiple(fixed_height, factor)
            fixed_width = self._round_down_multiple(fixed_width, factor)
            image_size_buckets = self._image_size_buckets(factor) if reshape_mode != "fixed" else ()
            from transformers.models.qwen2_vl import image_processing_qwen2_vl_fast

            original_preprocess = image_processor._preprocess

            def _fixed_preprocess(
                images,
                do_resize,
                size,
                interpolation,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
                patch_size,
                temporal_patch_size,
                merge_size,
                disable_grouping,
                return_tensors,
                **kwargs,
            ):
                original_smart_resize = image_processing_qwen2_vl_fast.smart_resize
                visual_cap = self._simple_visual_dynamic_cap(images)

                def _aicas_smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
                    if reshape_mode == "fixed":
                        return int(fixed_height), int(fixed_width)
                    bound_h, bound_w = self._dynamic_image_resize_bounds(
                        int(height),
                        int(width),
                        int(fixed_height),
                        int(fixed_width),
                        int(factor),
                        image_size_buckets,
                    )
                    if visual_cap > 0:
                        cap = self._round_down_multiple(max(int(factor), int(visual_cap)), int(factor))
                        bound_h = min(int(bound_h), int(cap))
                        bound_w = min(int(bound_w), int(cap))
                    target_h, target_w = self._aspect_resize_hw(
                        int(height),
                        int(width),
                        int(bound_h),
                        int(bound_w),
                        int(factor),
                        reshape_mode,
                        self._long_edge_align_mode(),
                    )
                    return self._bucket_image_hw(
                        int(target_h),
                        int(target_w),
                        int(bound_h),
                        int(bound_w),
                        int(factor),
                        image_size_buckets,
                        allow_upscale=(
                            reshape_mode == "long_edge"
                            and self._env_flag("AICAS_ALLOW_LONG_EDGE_BUCKET_UPSCALE")
                        ),
                    )

                image_processing_qwen2_vl_fast.smart_resize = _aicas_smart_resize
                try:
                    return original_preprocess(
                        images,
                        do_resize,
                        size,
                        interpolation,
                        do_rescale,
                        rescale_factor,
                        do_normalize,
                        image_mean,
                        image_std,
                        patch_size,
                        temporal_patch_size,
                        merge_size,
                        disable_grouping,
                        return_tensors,
                        **kwargs,
                    )
                finally:
                    image_processing_qwen2_vl_fast.smart_resize = original_smart_resize

            image_processor._preprocess = _fixed_preprocess
            max_pixels = int(fixed_height * fixed_width)
            image_processor.size = {"shortest_edge": max_pixels, "longest_edge": max_pixels}
            image_processor.min_pixels = max_pixels if reshape_mode == "fixed" else 1
            image_processor.max_pixels = max_pixels
            if reshape_mode == "fixed":
                image_processor._aicas_fixed_resize_hw = (int(fixed_height), int(fixed_width))
            else:
                image_processor._aicas_resize_bound_hw = (int(fixed_height), int(fixed_width))
                image_processor._aicas_image_size_buckets = tuple(image_size_buckets)
            image_processor._aicas_fixed_resize_factor = int(factor)
            self._mark_optimization(f"image_{reshape_mode}_resize")
            return

        changed = False
        if base_min_pixels > 0:
            if base_max_pixels > 0 and base_min_pixels > base_max_pixels:
                raise ValueError("AICAS_VISION_BASE_MIN_PIXELS must be <= AICAS_VISION_BASE_MAX_PIXELS")
            image_processor.min_pixels = int(base_min_pixels)
            changed = True

        if base_max_pixels > 0:
            if base_min_pixels <= 0:
                image_processor.min_pixels = 1
            image_processor.max_pixels = int(base_max_pixels)
            changed = True

        if changed:
            self._mark_optimization("image_reshape_pixel_budget")

    def _init_cpp_runtime_bridge(self):
        """Optionally build and load the stage-1 C++ runtime bridge."""
        if not self._cpp_runtime_master_enabled:
            return
        from cpp_runtime import load_prefill_text_runtime_module

        runtime_module = load_prefill_text_runtime_module(
            verbose=self._env_flag("AICAS_CPP_RUNTIME_VERBOSE")
        )
        from kernels.fused_add_rmsnorm import fused_add_rmsnorm

        metadata = runtime_module.runtime_metadata()
        self._cpp_runtime_module = runtime_module
        self._cpp_runtime_enabled = True
        self._cpp_runtime_fused_add_rmsnorm = fused_add_rmsnorm
        self._init_cpp_runtime_prefill_linear_packs()
        self._mark_optimization("cpp_runtime_bridge")
        print(f"[VLMModel] C++ runtime bridge ready: {dict(metadata)}")

    def _init_cpp_runtime_prefill_linear_packs(self):
        """Prebuild prefill-side fused linear packs so cpp runtime can avoid per-layer Python linear fanout."""
        if not self._cpp_runtime_enabled:
            return

        text_model = getattr(self._model, "language_model", None)
        if text_model is None:
            return

        packed_dtype = self._model.dtype
        for layer in list(getattr(text_model, "layers", [])):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
                self._get_fused_linear_pack(
                    attn,
                    (attn.q_proj, attn.k_proj, attn.v_proj),
                    packed_dtype,
                    "_aicas_qkv_fused_linear_cache",
                )

            mlp = getattr(layer, "mlp", None)
            if mlp is not None and all(hasattr(mlp, name) for name in ("gate_proj", "up_proj")):
                self._get_fused_linear_pack(
                    mlp,
                    (mlp.gate_proj, mlp.up_proj),
                    packed_dtype,
                    "_aicas_gate_up_fused_linear_cache",
                )

    def _enable_cpp_prefill_text_runtime_stub(self):
        """Patch text-model prefill entry with a C++ runtime bridge boundary hook."""
        if not self._cpp_runtime_enabled:
            return
        if not self._shape_specific_vision_runtime_enabled():
            return
        if self._mixed16_fusions_disabled:
            return

        from transformers.models.qwen3_vl import modeling_qwen3_vl

        text_cls = modeling_qwen3_vl.Qwen3VLTextModel
        if getattr(text_cls, "_aicas_cpp_prefill_text_runtime_stub", False):
            self._mark_optimization("cpp_prefill_text_runtime_stub")
            return

        def _patched_forward(
            module,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **kwargs,
        ):
            seq_len = int(
                inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            )
            is_prefill = seq_len > 1
            if is_prefill:
                hidden_states, past_key_values = self._cpp_runtime_module.prefill_text_model_forward_aten(
                    module,
                    input_ids,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    inputs_embeds,
                    use_cache,
                    cache_position,
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    modeling_qwen3_vl.create_causal_mask,
                    DynamicCache,
                    self._cpp_runtime_fused_add_rmsnorm,
                    kwargs,
                    self._cpp_runtime_cuda_current_stream,
                    self._cpp_runtime_prefill_qk_rope_launcher,
                    self._cpp_runtime_prefill_qk_rope_function,
                    self._cpp_runtime_prefill_qk_rope_packed_metadata,
                    self._cpp_runtime_prefill_qk_rope_block_d,
                    self._cpp_runtime_add_rmsnorm_launcher,
                    self._cpp_runtime_add_rmsnorm_function,
                    self._cpp_runtime_add_rmsnorm_packed_metadata,
                    self._cpp_runtime_add_rmsnorm_block_size,
                    self._cpp_runtime_rmsnorm_launcher,
                    self._cpp_runtime_rmsnorm_function,
                    self._cpp_runtime_rmsnorm_packed_metadata,
                    self._cpp_runtime_rmsnorm_block_size,
                    self._cpp_runtime_swiglu_launcher,
                    self._cpp_runtime_swiglu_function,
                    self._cpp_runtime_swiglu_packed_metadata,
                    self._cpp_runtime_swiglu_block_size,
                    self._cpp_runtime_prefill_rope_launcher,
                    self._cpp_runtime_prefill_rope_function,
                    self._cpp_runtime_prefill_rope_packed_metadata,
                    self._cpp_runtime_prefill_rope_block_d,
                    self._cpp_runtime_kv_cache_write_launcher,
                    self._cpp_runtime_kv_cache_write_function,
                    self._cpp_runtime_kv_cache_write_packed_metadata,
                    self._cpp_runtime_kv_cache_write_block_t,
                    self._cpp_runtime_kv_cache_write_block_d,
                    module.norm.weight,
                    module.norm.variance_epsilon,
                )
            else:
                if (input_ids is None) ^ (inputs_embeds is not None):
                    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
                if use_cache and past_key_values is None and not torch.jit.is_tracing():
                    past_key_values = DynamicCache(config=module.config)
                if inputs_embeds is None:
                    inputs_embeds = module.embed_tokens(input_ids)
                if cache_position is None:
                    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                    )
                if position_ids is None:
                    position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
                elif position_ids.ndim == 2:
                    position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
                if position_ids.ndim == 3 and position_ids.shape[0] == 4:
                    text_position_ids = position_ids[0]
                    position_ids = position_ids[1:]
                else:
                    text_position_ids = position_ids[0]
                attention_mask = modeling_qwen3_vl.create_causal_mask(
                    config=module.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )
                hidden_states = inputs_embeds
                position_embeddings = module.rotary_emb(hidden_states, position_ids)
                for layer_idx, decoder_layer in enumerate(module.layers):
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

                    if deepstack_visual_embeds is not None:
                        hidden_states = module._deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_idx],
                        )
                hidden_states = module.norm(hidden_states)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
            )

        text_cls.forward = _patched_forward
        text_cls._aicas_cpp_prefill_text_runtime_stub = True
        self._mark_optimization("cpp_prefill_text_runtime_stub")

    def _init_cpp_runtime_triton_handles(self):
        """Warm up Triton kernels once and keep direct launcher handles for C++ runtime."""
        if not self._cpp_runtime_enabled:
            return
        if self._env_flag("AICAS_ENABLE_CPP_RUNTIME_TRITON_PREFILL_QK_ROPE"):
            self._init_cpp_runtime_triton_prefill_qk_rope_handle()
        if self._env_flag("AICAS_ENABLE_CPP_RUNTIME_TRITON_ADD_RMSNORM"):
            self._init_cpp_runtime_triton_add_rmsnorm_handle()
        if self._env_flag("AICAS_ENABLE_CPP_RUNTIME_TRITON_RMSNORM"):
            self._init_cpp_runtime_triton_rmsnorm_handle()
        if self._env_flag("AICAS_ENABLE_CPP_RUNTIME_TRITON_SWIGLU"):
            self._init_cpp_runtime_triton_swiglu_handle()
        if self._env_flag("AICAS_ENABLE_CPP_RUNTIME_TRITON_PREFILL_KV_CACHE_WRITE"):
            self._init_cpp_runtime_triton_prefill_kv_cache_write_handle()
        if self._env_flag("AICAS_ENABLE_FUSED_VISION_ADD_LAYERNORM"):
            self._init_cpp_runtime_triton_add_layernorm_handle()
        if self._env_flag("AICAS_ENABLE_FUSED_VISION_ROPE"):
            self._init_cpp_runtime_triton_vision_rotary_handle()
        # Text deepstack is handled by the C++ prefill runtime as a contiguous
        # add fastpath. Do not prewarm the legacy Triton scatter kernel in the
        # clean runtime path.
        if (
            self._env_flag("AICAS_ENABLE_FUSED_PREFILL_ROPE_EMBED")
            and self._env_flag("AICAS_ENABLE_CPP_RUNTIME_TRITON_PREFILL_ROPE")
        ):
            self._init_cpp_runtime_triton_prefill_rope_handle()

    def _init_cpp_runtime_triton_prefill_qk_rope_handle(self):
        if self._cpp_runtime_prefill_qk_rope_launcher is not None:
            return

        import triton
        from kernels.fused_prefill_qk_rmsnorm_rope import _fused_prefill_qk_rmsnorm_rope_kernel

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        first_attn = self._model.language_model.layers[0].self_attn
        q_heads = int(first_attn.config.num_attention_heads)
        kv_heads = int(first_attn.config.num_key_value_heads)
        head_dim = int(first_attn.head_dim)
        seq_len = 4
        batch_size = 1
        q = torch.randn(batch_size, q_heads, seq_len, head_dim, device=device, dtype=self._model.dtype)
        k = torch.randn(batch_size, kv_heads, seq_len, head_dim, device=device, dtype=self._model.dtype)
        q_weight = torch.ones(head_dim, device=device, dtype=self._model.dtype)
        k_weight = torch.ones(head_dim, device=device, dtype=self._model.dtype)
        cos = torch.randn(batch_size, seq_len, head_dim, device=device, dtype=self._model.dtype)
        sin = torch.randn_like(cos)
        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)
        default_block_d = max(16, int(triton.next_power_of_2(head_dim)))
        block_d = self._env_int_or_default("AICAS_TRITON_PREFILL_QK_ROPE_BLOCK_D", default_block_d)
        num_warps = self._env_int_or_default(
            "AICAS_TRITON_PREFILL_QK_ROPE_NUM_WARPS",
            4 if block_d <= 128 else 8,
        )
        num_stages = self._env_int_or_default("AICAS_TRITON_PREFILL_QK_ROPE_NUM_STAGES", 2)
        q_rows = batch_size * q_heads * seq_len
        compiled_kernel = _fused_prefill_qk_rmsnorm_rope_kernel.run(
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            out_q,
            out_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            cos.stride(0),
            cos.stride(1),
            cos.stride(2),
            sin.stride(0),
            sin.stride(1),
            sin.stride(2),
            out_q.stride(0),
            out_q.stride(1),
            out_q.stride(2),
            out_q.stride(3),
            out_k.stride(0),
            out_k.stride(1),
            out_k.stride(2),
            out_k.stride(3),
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            batch_size,
            1e-6,
            1e-6,
            q_rows,
            grid=(q_rows + batch_size * kv_heads * seq_len,),
            warmup=True,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        self._cpp_runtime_triton_kernels["fused_prefill_qk_rmsnorm_rope"] = compiled_kernel
        self._cpp_runtime_prefill_qk_rope_launcher = compiled_kernel.run
        self._cpp_runtime_prefill_qk_rope_function = compiled_kernel.function
        self._cpp_runtime_prefill_qk_rope_packed_metadata = compiled_kernel.packed_metadata
        self._cpp_runtime_prefill_qk_rope_block_d = block_d
        self._cpp_runtime_cuda_current_stream = torch.cuda.current_stream
        self._mark_optimization("cpp_runtime_triton_prefill_qk_rope")

    def _init_cpp_runtime_triton_add_rmsnorm_handle(self):
        if self._cpp_runtime_add_rmsnorm_launcher is not None:
            return

        import triton
        from kernels.fused_add_rmsnorm import _fused_add_rmsnorm_kernel

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        hidden_size = int(self._model.language_model.layers[0].post_attention_layernorm.weight.shape[0])
        default_block_size = max(16, int(triton.next_power_of_2(hidden_size)))
        block_size = self._env_int_or_default("AICAS_TRITON_PREFILL_ADD_RMSNORM_BLOCK_SIZE", default_block_size)
        num_warps = self._env_int_or_default(
            "AICAS_TRITON_PREFILL_ADD_RMSNORM_NUM_WARPS",
            4 if block_size <= 1024 else 8,
        )
        num_stages = self._env_int_or_default("AICAS_TRITON_PREFILL_ADD_RMSNORM_NUM_STAGES", 2)
        rows = 4
        x = torch.randn(rows, hidden_size, device=device, dtype=self._model.dtype)
        y = torch.randn_like(x)
        weight = torch.ones(hidden_size, device=device, dtype=self._model.dtype)
        sum_out = torch.empty_like(x)
        norm_out = torch.empty_like(x)
        compiled_kernel = _fused_add_rmsnorm_kernel.run(
            x,
            y,
            weight,
            sum_out,
            norm_out,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            sum_out.stride(0),
            sum_out.stride(1),
            norm_out.stride(0),
            norm_out.stride(1),
            hidden_size,
            1e-6,
            grid=(rows,),
            warmup=True,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        self._cpp_runtime_triton_kernels["fused_add_rmsnorm"] = compiled_kernel
        self._cpp_runtime_add_rmsnorm_launcher = compiled_kernel.run
        self._cpp_runtime_add_rmsnorm_function = compiled_kernel.function
        self._cpp_runtime_add_rmsnorm_packed_metadata = compiled_kernel.packed_metadata
        self._cpp_runtime_add_rmsnorm_block_size = block_size
        self._cpp_runtime_cuda_current_stream = torch.cuda.current_stream
        self._mark_optimization("cpp_runtime_triton_add_rmsnorm")

    def _init_cpp_runtime_triton_rmsnorm_handle(self):
        if self._cpp_runtime_rmsnorm_launcher is not None:
            return

        import triton
        from kernels.fused_rmsnorm import _fused_rmsnorm_kernel

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        hidden_size = int(self._model.language_model.layers[0].input_layernorm.weight.shape[0])
        default_block_size = max(16, int(triton.next_power_of_2(hidden_size)))
        block_size = self._env_int_or_default("AICAS_TRITON_PREFILL_RMSNORM_BLOCK_SIZE", default_block_size)
        num_warps = self._env_int_or_default(
            "AICAS_TRITON_PREFILL_RMSNORM_NUM_WARPS",
            4 if block_size <= 1024 else 8,
        )
        num_stages = self._env_int_or_default("AICAS_TRITON_PREFILL_RMSNORM_NUM_STAGES", 2)
        rows = 4
        x = torch.randn(rows, hidden_size, device=device, dtype=self._model.dtype)
        weight = torch.ones(hidden_size, device=device, dtype=self._model.dtype)
        out = torch.empty_like(x)
        compiled_kernel = _fused_rmsnorm_kernel.run(
            x,
            weight,
            out,
            x.stride(0),
            x.stride(1),
            out.stride(0),
            out.stride(1),
            hidden_size,
            1e-6,
            grid=(rows,),
            warmup=True,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        self._cpp_runtime_triton_kernels["fused_rmsnorm"] = compiled_kernel
        self._cpp_runtime_rmsnorm_launcher = compiled_kernel.run
        self._cpp_runtime_rmsnorm_function = compiled_kernel.function
        self._cpp_runtime_rmsnorm_packed_metadata = compiled_kernel.packed_metadata
        self._cpp_runtime_rmsnorm_block_size = block_size
        self._cpp_runtime_cuda_current_stream = torch.cuda.current_stream
        self._mark_optimization("cpp_runtime_triton_rmsnorm")

    def _init_cpp_runtime_triton_swiglu_handle(self):
        if self._cpp_runtime_swiglu_launcher is not None:
            return

        import triton
        from kernels.fused_swiglu import _fused_swiglu_kernel

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        intermediate_size = int(self._model.language_model.layers[0].mlp.gate_proj.out_features)
        default_block_size = min(512, max(16, int(triton.next_power_of_2(intermediate_size))))
        block_size = self._env_int_or_default("AICAS_TRITON_PREFILL_SWIGLU_BLOCK_COL", default_block_size)
        num_warps = self._env_int_or_default("AICAS_TRITON_PREFILL_SWIGLU_NUM_WARPS", 4)
        num_stages = self._env_int_or_default("AICAS_TRITON_PREFILL_SWIGLU_NUM_STAGES", 2)
        rows = 4
        gate = torch.randn(rows, intermediate_size, device=device, dtype=self._model.dtype)
        up = torch.randn_like(gate)
        out = torch.empty_like(gate)
        compiled_kernel = _fused_swiglu_kernel.run(
            gate,
            up,
            out,
            gate.stride(0),
            gate.stride(1),
            up.stride(0),
            up.stride(1),
            out.stride(0),
            out.stride(1),
            intermediate_size,
            grid=(rows, triton.cdiv(intermediate_size, block_size)),
            warmup=True,
            BLOCK_COL=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        self._cpp_runtime_triton_kernels["fused_swiglu"] = compiled_kernel
        self._cpp_runtime_swiglu_launcher = compiled_kernel.run
        self._cpp_runtime_swiglu_function = compiled_kernel.function
        self._cpp_runtime_swiglu_packed_metadata = compiled_kernel.packed_metadata
        self._cpp_runtime_swiglu_block_size = block_size
        self._cpp_runtime_cuda_current_stream = torch.cuda.current_stream
        self._mark_optimization("cpp_runtime_triton_swiglu")

    def _init_cpp_runtime_triton_prefill_kv_cache_write_handle(self):
        if self._cpp_runtime_kv_cache_write_launcher is not None:
            return

        import triton
        from kernels.fused_prefill_kv_cache_write import _fused_prefill_kv_cache_write_kernel

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        first_attn = self._model.language_model.layers[0].self_attn
        kv_heads = int(first_attn.config.num_key_value_heads)
        head_dim = int(first_attn.head_dim)
        batch_size = 1
        seq_len = 4
        cache_len = 16
        key = torch.randn(batch_size, kv_heads, seq_len, head_dim, device=device, dtype=self._model.dtype)
        value_base = torch.randn(batch_size, seq_len, kv_heads, head_dim, device=device, dtype=self._model.dtype)
        value = value_base.transpose(1, 2)
        key_cache = torch.empty(batch_size, kv_heads, cache_len, head_dim, device=device, dtype=self._model.dtype)
        value_cache = torch.empty_like(key_cache)
        cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
        default_block_d = max(16, int(triton.next_power_of_2(head_dim)))
        block_t = self._env_int_or_default("AICAS_TRITON_PREFILL_KV_CACHE_WRITE_BLOCK_T", 8)
        block_d = self._env_int_or_default("AICAS_TRITON_PREFILL_KV_CACHE_WRITE_BLOCK_D", default_block_d)
        num_warps = self._env_int_or_default("AICAS_TRITON_PREFILL_KV_CACHE_WRITE_NUM_WARPS", 4)
        num_stages = self._env_int_or_default("AICAS_TRITON_PREFILL_KV_CACHE_WRITE_NUM_STAGES", 4)
        compiled_kernel = _fused_prefill_kv_cache_write_kernel.run(
            key,
            value,
            key_cache,
            value_cache,
            cache_position,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            kv_heads,
            seq_len,
            head_dim,
            grid=(triton.cdiv(seq_len, block_t), batch_size * kv_heads),
            warmup=True,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        self._cpp_runtime_triton_kernels["fused_prefill_kv_cache_write"] = compiled_kernel
        self._cpp_runtime_kv_cache_write_launcher = compiled_kernel.run
        self._cpp_runtime_kv_cache_write_function = compiled_kernel.function
        self._cpp_runtime_kv_cache_write_packed_metadata = compiled_kernel.packed_metadata
        self._cpp_runtime_kv_cache_write_block_t = block_t
        self._cpp_runtime_kv_cache_write_block_d = block_d
        self._cpp_runtime_cuda_current_stream = torch.cuda.current_stream
        self._mark_optimization("cpp_runtime_triton_prefill_kv_cache_write")

    def _init_cpp_runtime_triton_add_layernorm_handle(self):
        if self._cpp_runtime_add_layernorm_launcher is not None:
            return

        import triton
        from kernels.fused_add_layernorm import _fused_add_layernorm_kernel

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        hidden_size = int(self._model.visual.blocks[0].norm2.weight.shape[0])
        default_block_size = max(16, min(8192, int(triton.next_power_of_2(hidden_size))))
        block_size = self._env_int_or_default("AICAS_TRITON_VISION_ADD_LAYERNORM_BLOCK_SIZE", default_block_size)
        num_warps = self._env_int_or_default(
            "AICAS_TRITON_VISION_ADD_LAYERNORM_NUM_WARPS",
            8 if hidden_size >= 2048 else 4,
        )
        num_stages = self._env_int_or_default("AICAS_TRITON_VISION_ADD_LAYERNORM_NUM_STAGES", 2)
        rows = 4
        x = torch.randn(rows, hidden_size, device=device, dtype=self._model.dtype)
        y = torch.randn_like(x)
        weight = torch.ones(hidden_size, device=device, dtype=self._model.dtype)
        bias = torch.zeros(hidden_size, device=device, dtype=self._model.dtype)
        sum_out = torch.empty_like(x)
        norm_out = torch.empty_like(x)
        compiled_kernel = _fused_add_layernorm_kernel.run(
            x,
            y,
            weight,
            bias,
            sum_out,
            norm_out,
            x.stride(0),
            x.stride(1),
            y.stride(0),
            y.stride(1),
            sum_out.stride(0),
            sum_out.stride(1),
            norm_out.stride(0),
            norm_out.stride(1),
            hidden_size,
            1e-6,
            grid=(rows,),
            warmup=True,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        self._cpp_runtime_triton_kernels["fused_add_layernorm"] = compiled_kernel
        self._cpp_runtime_add_layernorm_launcher = compiled_kernel.run
        self._cpp_runtime_add_layernorm_function = compiled_kernel.function
        self._cpp_runtime_add_layernorm_packed_metadata = compiled_kernel.packed_metadata
        self._cpp_runtime_add_layernorm_block_size = block_size
        self._cpp_runtime_cuda_current_stream = torch.cuda.current_stream
        self._mark_optimization("cpp_runtime_triton_add_layernorm")

    def _init_cpp_runtime_triton_vision_rotary_handle(self):
        if self._cpp_runtime_vision_rotary_launcher is not None:
            return

        import triton
        from kernels.fused_vision_rotary import _fused_apply_rotary_pos_emb_vision_kernel

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        first_block = self._model.visual.blocks[0]
        num_heads = int(getattr(first_block.attn, "num_heads", 8))
        head_dim = int(first_block.attn.qkv.weight.shape[0] // (3 * num_heads))
        seq_len = 4
        default_block_d = 128 if head_dim > 64 else 64
        block_d = self._env_int_or_default("AICAS_TRITON_VISION_ROTARY_BLOCK_D", default_block_d)
        num_warps = self._env_int_or_default("AICAS_TRITON_VISION_ROTARY_NUM_WARPS", 4)
        num_stages = self._env_int_or_default("AICAS_TRITON_VISION_ROTARY_NUM_STAGES", 2)
        q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=self._model.dtype)
        k = torch.randn_like(q)
        # Runtime vision rotary receives fp32 cos/sin from
        # Qwen3VLVisionRotaryEmbedding. Warm up with the same dtype so cached
        # launch metadata matches real execution.
        cos = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        sin = torch.randn_like(cos)
        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)
        compiled_kernel = _fused_apply_rotary_pos_emb_vision_kernel.run(
            q,
            k,
            cos,
            sin,
            out_q,
            out_k,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            out_q.stride(0),
            out_q.stride(1),
            out_q.stride(2),
            out_k.stride(0),
            out_k.stride(1),
            out_k.stride(2),
            num_heads,
            head_dim,
            head_dim // 2,
            grid=(seq_len * num_heads, triton.cdiv(head_dim, block_d)),
            warmup=True,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        self._cpp_runtime_triton_kernels["fused_vision_rotary"] = compiled_kernel
        self._cpp_runtime_vision_rotary_launcher = compiled_kernel.run
        self._cpp_runtime_vision_rotary_function = compiled_kernel.function
        self._cpp_runtime_vision_rotary_packed_metadata = compiled_kernel.packed_metadata
        self._cpp_runtime_vision_rotary_block_d = block_d
        self._cpp_runtime_cuda_current_stream = torch.cuda.current_stream
        self._mark_optimization("cpp_runtime_triton_vision_rotary")

    def _init_cpp_runtime_triton_prefill_rope_handle(self):
        if self._cpp_runtime_prefill_rope_launcher is not None:
            return

        import triton
        from kernels.fused_prefill_rope_embeddings import _fused_prefill_rope_embeddings_kernel

        device = next(self._model.parameters()).device
        if device.type != "cuda":
            return

        rotary = self._model.language_model.rotary_emb
        half_dim = int(rotary.inv_freq.numel())
        seq_len = 4
        batch_size = 1
        head_dim = half_dim * 2
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len).expand(3, batch_size, seq_len).contiguous()
        inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float32).contiguous()
        out_cos = torch.empty(batch_size, seq_len, head_dim, device=device, dtype=self._model.dtype)
        out_sin = torch.empty_like(out_cos)
        mrope_section = getattr(rotary, "mrope_section", None)
        if mrope_section is None:
            mrope_section = (0, 0, 0)
        len_h = int(mrope_section[1]) * 3
        len_w = int(mrope_section[2]) * 3
        len_h = max(0, min(len_h, half_dim))
        len_w = max(0, min(len_w, half_dim))
        default_block_d = 128 if half_dim > 64 else 64
        block_d = self._env_int_or_default("AICAS_TRITON_PREFILL_ROPE_BLOCK_D", default_block_d)
        num_warps = self._env_int_or_default("AICAS_TRITON_PREFILL_ROPE_NUM_WARPS", 4)
        num_stages = self._env_int_or_default("AICAS_TRITON_PREFILL_ROPE_NUM_STAGES", 2)
        compiled_kernel = _fused_prefill_rope_embeddings_kernel.run(
            position_ids,
            inv_freq,
            out_cos,
            out_sin,
            position_ids.stride(0),
            position_ids.stride(1),
            position_ids.stride(2),
            out_cos.stride(0),
            out_cos.stride(1),
            out_cos.stride(2),
            out_sin.stride(0),
            out_sin.stride(1),
            out_sin.stride(2),
            seq_len,
            half_dim,
            len_h,
            len_w,
            float(rotary.attention_scaling),
            grid=(batch_size * seq_len, triton.cdiv(half_dim, block_d)),
            warmup=True,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        self._cpp_runtime_triton_kernels["fused_prefill_rope_embeddings"] = compiled_kernel
        self._cpp_runtime_prefill_rope_launcher = compiled_kernel.run
        self._cpp_runtime_prefill_rope_function = compiled_kernel.function
        self._cpp_runtime_prefill_rope_packed_metadata = compiled_kernel.packed_metadata
        self._cpp_runtime_prefill_rope_block_d = block_d
        self._cpp_runtime_cuda_current_stream = torch.cuda.current_stream
        self._mark_optimization("cpp_runtime_triton_prefill_rope")

    def _cpp_runtime_launch_prefill_rope(self, position_ids, inv_freq, attention_scaling, mrope_section):
        if not self._cpp_runtime_enabled:
            return None
        if self._cpp_runtime_prefill_rope_launcher is None or self._cpp_runtime_cuda_current_stream is None:
            raise RuntimeError(
                "C++ prefill-rope launcher is not initialized while C++ runtime is enabled."
            )
        batch_size = int(position_ids.shape[1])
        seq_len = int(position_ids.shape[2])
        half_dim = int(inv_freq.shape[0])
        head_dim = half_dim * 2
        out_cos = torch.empty((batch_size, seq_len, head_dim), device=position_ids.device, dtype=torch.float16)
        out_sin = torch.empty_like(out_cos)
        if mrope_section is None:
            mrope_section = (0, 0, 0)
        len_h = int(mrope_section[1]) * 3
        len_w = int(mrope_section[2]) * 3
        len_h = max(0, min(len_h, half_dim))
        len_w = max(0, min(len_w, half_dim))
        return self._cpp_runtime_module.launch_prefill_rope_cached_triton(
            self._cpp_runtime_prefill_rope_launcher,
            self._cpp_runtime_prefill_rope_function,
            self._cpp_runtime_prefill_rope_packed_metadata,
            self._cpp_runtime_cuda_current_stream,
            self._cpp_runtime_prefill_rope_block_d,
            position_ids,
            inv_freq,
            out_cos,
            out_sin,
            seq_len,
            float(attention_scaling),
            len_h,
            len_w,
        )

    def _cpp_runtime_vision_forward(
        self,
        module,
        hidden_states,
        first_norm1,
        cu_seqlens,
        position_embeddings,
        chunk_lengths,
        kwargs,
    ):
        if not self._cpp_runtime_enabled:
            return None
        if not self._env_flag("AICAS_ENABLE_CPP_VISION_RUNTIME_LOOP"):
            return None
        if not self._shape_specific_vision_runtime_enabled():
            return None
        vision_forward = getattr(self._cpp_runtime_module, "vision_runtime_forward", None)
        if vision_forward is None:
            return None
        cos, sin = position_embeddings
        if cos.dtype != torch.float32:
            cos = cos.to(dtype=torch.float32)
        if sin.dtype != torch.float32:
            sin = sin.to(dtype=torch.float32)
        if not cos.is_contiguous():
            cos = cos.contiguous()
        if not sin.is_contiguous():
            sin = sin.contiguous()
        cached_norm_handoff_enabled = self._shape_specific_mixed16_feature_enabled(
            "AICAS_ENABLE_FUSED_VISION_ADD_LAYERNORM"
        )
        interlayer_fuse_enabled = self._env_flag("AICAS_ENABLE_FUSED_VISION_INTERLAYER_ADD_LAYERNORM")
        return vision_forward(
            list(module.blocks),
            tuple(int(x) for x in module.deepstack_visual_indexes),
            list(module.deepstack_merger_list),
            module.merger,
            hidden_states,
            first_norm1,
            cu_seqlens,
            cos,
            sin,
            chunk_lengths,
            dict(kwargs),
            self._cpp_runtime_add_layernorm_launcher,
            self._cpp_runtime_add_layernorm_function,
            self._cpp_runtime_add_layernorm_packed_metadata,
            self._cpp_runtime_cuda_current_stream,
            int(self._cpp_runtime_add_layernorm_block_size),
            bool(interlayer_fuse_enabled),
            bool(cached_norm_handoff_enabled),
            self._cpp_runtime_vision_rotary_launcher,
            self._cpp_runtime_vision_rotary_function,
            self._cpp_runtime_vision_rotary_packed_metadata,
            int(self._cpp_runtime_vision_rotary_block_d),
        )

    def _cpp_runtime_vision_get_image_features(self, visual, pixel_values, image_grid_thw):
        if not self._cpp_runtime_enabled:
            return None
        if not self._env_flag("AICAS_ENABLE_CPP_VISION_RUNTIME_LOOP"):
            return None
        if not self._shape_specific_vision_runtime_enabled():
            return None
        vision_get_image_features = getattr(self._cpp_runtime_module, "vision_get_image_features_aten", None)
        if vision_get_image_features is None:
            return None
        cached_norm_handoff_enabled = self._shape_specific_mixed16_feature_enabled(
            "AICAS_ENABLE_FUSED_VISION_ADD_LAYERNORM"
        )
        interlayer_fuse_enabled = self._env_flag("AICAS_ENABLE_FUSED_VISION_INTERLAYER_ADD_LAYERNORM")
        grid_meta = self._get_single_image_grid_meta(visual, image_grid_thw)
        if grid_meta is None:
            grid_t, grid_h, grid_w = 0, 0, 0
        else:
            grid_t, grid_h, grid_w = grid_meta
        return vision_get_image_features(
            visual,
            pixel_values,
            image_grid_thw,
            self._cpp_runtime_add_layernorm_launcher,
            self._cpp_runtime_add_layernorm_function,
            self._cpp_runtime_add_layernorm_packed_metadata,
            self._cpp_runtime_cuda_current_stream,
            int(self._cpp_runtime_add_layernorm_block_size),
            bool(interlayer_fuse_enabled),
            bool(cached_norm_handoff_enabled),
            self._cpp_runtime_vision_rotary_launcher,
            self._cpp_runtime_vision_rotary_function,
            self._cpp_runtime_vision_rotary_packed_metadata,
            int(self._cpp_runtime_vision_rotary_block_d),
            int(grid_t),
            int(grid_h),
            int(grid_w),
        )

    def _visual_graph_key(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Tuple:
        grid_meta = self._get_single_image_grid_meta(self._model.model.visual, image_grid_thw)
        return (
            str(pixel_values.device),
            str(pixel_values.dtype),
            tuple(int(x) for x in pixel_values.shape),
            str(image_grid_thw.device),
            str(image_grid_thw.dtype),
            tuple(int(x) for x in image_grid_thw.shape),
            tuple(int(x) for x in grid_meta) if grid_meta is not None else (),
        )

    def _clone_visual_graph_output(self, out):
        image_embeds, deepstack_image_embeds = out
        if isinstance(image_embeds, tuple):
            image_embeds = tuple(x.clone() for x in image_embeds)
        elif isinstance(image_embeds, list):
            image_embeds = [x.clone() for x in image_embeds]
        else:
            image_embeds = image_embeds.clone()
        deepstack_image_embeds = [x.clone() for x in list(deepstack_image_embeds)]
        return image_embeds, deepstack_image_embeds

    def _run_visual_graph_forward(self, module, static_pixel_values: torch.Tensor, static_image_grid_thw: torch.Tensor):
        pixel_values = static_pixel_values.type(module.visual.dtype)
        return self._cpp_runtime_vision_get_image_features(module.visual, pixel_values, static_image_grid_thw)

    def _visual_get_image_features_with_graph(self, module, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        if (
            not self._visual_cuda_graph_enabled
            or self._prefill_cuda_graph_direct_visual
            or self._cuda_stream_is_capturing()
            or module.training
            or pixel_values is None
            or image_grid_thw is None
            or not torch.is_tensor(pixel_values)
            or not torch.is_tensor(image_grid_thw)
            or pixel_values.device.type != "cuda"
        ):
            return None

        visual = getattr(module, "visual", None)
        if visual is None:
            return None
        if image_grid_thw.device != pixel_values.device:
            return None

        try:
            graph_key = self._visual_graph_key(pixel_values, image_grid_thw)
            entry = self._visual_cuda_graph_cache.get(graph_key)
            if entry is None:
                entry = {
                    "static_pixel_values": torch.empty_like(pixel_values),
                    "static_image_grid_thw": image_grid_thw.clone(),
                    "graph": None,
                    "captured": False,
                    "out": None,
                }
                self._visual_cuda_graph_cache[graph_key] = entry

            entry["static_pixel_values"].copy_(pixel_values)
            static_pixel_values = entry["static_pixel_values"]
            static_image_grid_thw = entry["static_image_grid_thw"]
            self._get_single_image_grid_meta(visual, static_image_grid_thw)

            if not entry["captured"]:
                warmup_out = self._run_visual_graph_forward(module, static_pixel_values, static_image_grid_thw)
                if warmup_out is None:
                    return None
                warmup_out = self._run_visual_graph_forward(module, static_pixel_values, static_image_grid_thw)
                if warmup_out is None:
                    return None

                graph = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    entry["out"] = self._run_visual_graph_forward(module, static_pixel_values, static_image_grid_thw)
                entry["graph"] = graph
                entry["captured"] = True
                print("[VLMModel] CUDA Graph captured (visual replay path)")

            entry["graph"].replay()
            return self._clone_visual_graph_output(entry["out"])
        except Exception as exc:
            self._visual_cuda_graph_disabled_reason = str(exc)
            self._visual_cuda_graph_enabled = False
            print(f"[VLMModel] Visual CUDA Graph disabled: {exc}")
            return None

    @contextmanager
    def _nvtx_range(self, name: str):
        pushed = False
        if self._prefill_nvtx_enabled and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
        try:
            yield
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()

    @contextmanager
    def _prefill_active_scope(self):
        prev_prefill_flag = self._nvtx_prefill_active
        self._nvtx_prefill_active = True
        try:
            yield
        finally:
            self._nvtx_prefill_active = prev_prefill_flag

    def _apply_fp16_prefill_optimizations(self):
        """Apply fp16 prefill-side fusions and caches as one group."""
        if not self._enable_fp16_prefill_optimizations:
            return

        self._enable_fp16_visual_get_image_features()
        self._enable_fp16_fast_connector_scatter()

    def _enable_fused_decode_position_advance(self):
        if not (self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_FUSED_DECODE_POS_ADVANCE")):
            return

        from kernels.fused_decode_position_advance import (
            fused_decode_position_advance,
            fused_decode_set_start,
        )

        self._fused_decode_position_advance = fused_decode_position_advance
        self._fused_decode_set_start = fused_decode_set_start
        self._mark_optimization("fused_decode_pos_advance")
        self._mark_optimization("fused_decode_set_start")

    def _enable_cached_decode_rope_embeddings(self):
        if not (self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_CACHED_DECODE_ROPE")):
            return

        from transformers.models.qwen3_vl import modeling_qwen3_vl

        rotary_cls = modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding

        def _cache_len(module):
            return max(
                int(getattr(module, "max_seq_len_cached", 0)),
                int(getattr(module, "original_max_seq_len", 0)),
                int(getattr(module.config, "max_position_embeddings", 0)),
                1,
            )

        def _ensure_decode_rope_cache(module, device, dtype):
            if str(getattr(module, "rope_type", "default")) != "default":
                raise RuntimeError("cached decode RoPE currently supports only default rope_type")
            key = (str(device), str(dtype))
            cache = getattr(module, "_aicas_decode_rope_cache", {})
            required_len = _cache_len(module)
            payload = cache.get(key)
            if payload is not None:
                rope_cache, cached_len = payload
                if int(cached_len) >= required_len:
                    return rope_cache

            inv_freq = module.inv_freq.detach().to(device="cpu", dtype=torch.float32)
            positions = torch.arange(required_len, device="cpu", dtype=torch.float32)
            freqs = positions[:, None] * inv_freq[None, :]
            emb = torch.cat((freqs, freqs), dim=-1)
            scale = float(module.attention_scaling)
            rope_cache = (
                torch.stack((emb.cos() * scale, emb.sin() * scale), dim=1)
                .to(dtype=dtype)
                .to(device=device, non_blocking=True)
            )
            cache[key] = (rope_cache, required_len)
            module._aicas_decode_rope_cache = cache
            return rope_cache

        if getattr(rotary_cls, "_aicas_cached_decode_rope", False):
            for module in self._model.modules():
                if isinstance(module, rotary_cls):
                    _ensure_decode_rope_cache(module, module.inv_freq.device, self._model.dtype)
            self._mark_optimization("cached_decode_rope")
            return

        original_forward = rotary_cls.forward

        @torch.no_grad()
        def _patched_forward(module, x, position_ids):
            is_decode_step = (
                not module.training
                and x.is_cuda
                and x.dtype == torch.float16
                and x.shape[1] == 1
                and position_ids is not None
            )
            if not is_decode_step:
                return original_forward(module, x, position_ids)
            if position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            if position_ids.ndim != 3 or int(position_ids.shape[0]) != 3 or int(position_ids.shape[-1]) != 1:
                raise RuntimeError("cached decode RoPE expects position_ids shape [3, batch, 1]")

            key = (str(x.device), str(x.dtype))
            payload = getattr(module, "_aicas_decode_rope_cache", {}).get(key)
            if payload is None:
                raise RuntimeError("cached decode RoPE table was not initialized before decode graph capture")
            rope_cache, _cached_len = payload
            pos = position_ids[0, :, 0].to(dtype=torch.long)
            if os.environ.get("AICAS_ENABLE_CUDA_DECODE_QKV_QK_UPDATE_ROPE_CACHE", "0") == "1":
                return None, None, rope_cache, pos
            rope = torch.index_select(rope_cache, 0, pos)
            cos = rope[:, 0:1, :]
            sin = rope[:, 1:2, :]
            return cos, sin

        rotary_cls.forward = _patched_forward
        rotary_cls._aicas_cached_decode_rope = True
        for module in self._model.modules():
            if isinstance(module, rotary_cls):
                _ensure_decode_rope_cache(module, module.inv_freq.device, self._model.dtype)
        self._mark_optimization("cached_decode_rope")

    def _enable_prefill_nvtx_annotations(self):
        """Add two-level NVTX ranges for prefill stage profiling."""
        if not self._prefill_nvtx_enabled or not torch.cuda.is_available():
            return
        if getattr(self._model, "_aicas_prefill_nvtx_enabled", False):
            self._mark_optimization("prefill_nvtx")
            return

        def _register(module, tag: str):
            if module is None:
                return
            if getattr(module, "_aicas_prefill_nvtx_registered", False):
                return

            depth_attr = "_aicas_prefill_nvtx_depth"

            def _pre_hook(mod, _inputs):
                if not (self._prefill_nvtx_enabled and self._nvtx_prefill_active):
                    return
                torch.cuda.nvtx.range_push(tag)
                setattr(mod, depth_attr, int(getattr(mod, depth_attr, 0)) + 1)

            def _post_hook(mod, _inputs, output):
                depth = int(getattr(mod, depth_attr, 0))
                if depth > 0:
                    torch.cuda.nvtx.range_pop()
                    setattr(mod, depth_attr, depth - 1)
                return output

            pre_h = module.register_forward_pre_hook(_pre_hook)
            post_h = module.register_forward_hook(_post_hook)
            self._prefill_nvtx_hook_handles.extend([pre_h, post_h])
            module._aicas_prefill_nvtx_registered = True

        qwen = getattr(self._model, "model", None)
        text_model = getattr(qwen, "language_model", None) if qwen is not None else None

        _register(getattr(qwen, "visual", None), "prefill/module.visual")
        _register(text_model, "prefill/module.text_model")
        self._model._aicas_prefill_nvtx_enabled = True
        self._mark_optimization("prefill_nvtx")

    def _enable_fp16_visual_get_image_features(self):
        """Route single-image visual prefill through graph or C++ runtime only."""
        if not self._shape_specific_mixed16_feature_enabled("AICAS_ENABLE_VISUAL_GET_IMAGE_FEATURES"):
            return

        from transformers.models.qwen3_vl import modeling_qwen3_vl

        qwen_model_cls = modeling_qwen3_vl.Qwen3VLModel

        if not getattr(qwen_model_cls, "_aicas_fp16_visual_get_image_features", False):
            def _patched_get_image_features(module, pixel_values: torch.FloatTensor, image_grid_thw: torch.Tensor = None):
                if image_grid_thw is None or pixel_values is None:
                    raise RuntimeError("competition path requires single-image pixel_values and image_grid_thw")

                graph_out = self._visual_get_image_features_with_graph(module, pixel_values, image_grid_thw)
                if graph_out is not None:
                    return graph_out

                pixel_values = pixel_values.type(module.visual.dtype)
                cpp_out = self._cpp_runtime_vision_get_image_features(module.visual, pixel_values, image_grid_thw)
                if cpp_out is not None:
                    return cpp_out

                raise RuntimeError("C++ visual runtime failed on the competition path")

            qwen_model_cls.get_image_features = _patched_get_image_features
            qwen_model_cls._aicas_fp16_visual_get_image_features = True

        self._mark_optimization("fp16_visual_get_image_features")

    def _enable_fp16_fast_connector_scatter(self):
        """Fast-path image connector: contiguous placeholder write without masked_scatter."""
        if not self._shape_specific_mixed16_feature_enabled("AICAS_ENABLE_FAST_CONNECTOR_SCATTER"):
            return

        from transformers.models.qwen3_vl import modeling_qwen3_vl

        qwen_model_cls = modeling_qwen3_vl.Qwen3VLModel
        if getattr(qwen_model_cls, "_aicas_fast_connector_scatter", False):
            self._mark_optimization("fp16_fast_connector_scatter")
            return

        original_forward = qwen_model_cls.forward

        def _get_cached_visual_mask(module, device, seq_len: int, start: int, length: int):
            key = (str(device), int(seq_len), int(start), int(length))
            cache = getattr(module, "_aicas_connector_visual_mask_cache", {})
            order = getattr(module, "_aicas_connector_visual_mask_order", [])

            mask = cache.get(key)
            if mask is None:
                mask = torch.zeros((1, int(seq_len)), dtype=torch.bool, device=device)
                if int(length) > 0:
                    mask[:, int(start): int(start) + int(length)] = True
                cache[key] = mask
                order.append(key)
                setattr(module, "_aicas_connector_visual_mask_cache", cache)
                setattr(module, "_aicas_connector_visual_mask_order", order)
            return mask

        def _build_single_image_rope(module, input_ids, inputs_embeds, image_grid_thw, seq_len, start, n_image_tokens):
            grid_meta = self._get_single_image_grid_meta(module, image_grid_thw)
            if grid_meta is None:
                raise RuntimeError("competition connector requires single-image grid metadata")
            t, h, w = grid_meta
            sms = int(module.config.vision_config.spatial_merge_size)
            if sms <= 0:
                raise RuntimeError("invalid spatial_merge_size for competition connector")
            llm_h = int(h) // sms
            llm_w = int(w) // sms
            expected_tokens = int(t) * int(llm_h) * int(llm_w)
            if expected_tokens != int(n_image_tokens):
                raise RuntimeError(
                    f"single-image rope token mismatch: expected {expected_tokens}, got {int(n_image_tokens)}"
                )

            cache_key = (
                str(inputs_embeds.device),
                int(seq_len),
                int(start),
                int(n_image_tokens),
                int(t),
                int(llm_h),
                int(llm_w),
                str(input_ids.dtype),
            )
            cache = getattr(module, "_aicas_single_image_rope_index_cache", {})
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

            position_ids = torch.empty(
                (3, 1, int(seq_len)),
                dtype=input_ids.dtype,
                device=inputs_embeds.device,
            )
            if int(start) > 0:
                prefix = torch.arange(int(start), device=inputs_embeds.device, dtype=input_ids.dtype)
                position_ids[:, 0, : int(start)] = prefix.view(1, -1).expand(3, -1)
            t_index = torch.arange(int(t), device=inputs_embeds.device, dtype=input_ids.dtype).view(
                -1, 1
            ).expand(-1, int(llm_h) * int(llm_w)).flatten()
            h_index = torch.arange(int(llm_h), device=inputs_embeds.device, dtype=input_ids.dtype).view(
                1, -1, 1
            ).expand(int(t), -1, int(llm_w)).flatten()
            w_index = torch.arange(int(llm_w), device=inputs_embeds.device, dtype=input_ids.dtype).view(
                1, 1, -1
            ).expand(int(t), int(llm_h), -1).flatten()
            vision = torch.stack((t_index, h_index, w_index), dim=0) + int(start)
            image_end = int(start) + int(n_image_tokens)
            position_ids[:, 0, int(start): image_end] = vision
            tail_len = int(seq_len) - int(image_end)
            tail_start = int(start) + max(int(t) - 1, int(llm_h) - 1, int(llm_w) - 1) + 1
            if tail_len > 0:
                tail = torch.arange(tail_len, device=inputs_embeds.device, dtype=input_ids.dtype) + int(tail_start)
                position_ids[:, 0, image_end:] = tail.view(1, -1).expand(3, -1)
            rope_delta_val = int(tail_start + max(tail_len - 1, 0) + 1 - int(seq_len))
            rope_deltas = torch.tensor(
                [[rope_delta_val]],
                dtype=input_ids.dtype,
                device=inputs_embeds.device,
            )
            cache[cache_key] = (position_ids, rope_deltas)
            module._aicas_single_image_rope_index_cache = cache
            return position_ids, rope_deltas

        def _patched_forward(
            module,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            cache_position=None,
            **kwargs,
        ):
            if pixel_values is None:
                return original_forward(
                    module,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    cache_position=cache_position,
                    **kwargs,
                )
            if module.training:
                raise RuntimeError("competition path is inference-only")
            if input_ids is None or inputs_embeds is not None:
                raise RuntimeError("competition prefill requires input_ids and lets connector build inputs_embeds")
            if int(input_ids.shape[0]) != 1:
                raise RuntimeError("competition path only supports batch size 1")
            if pixel_values_videos is not None or video_grid_thw is not None:
                raise RuntimeError("competition path does not support video inputs")
            if image_grid_thw is None or not torch.is_tensor(image_grid_thw) or int(image_grid_thw.shape[0]) != 1:
                raise RuntimeError("competition path requires exactly one image grid")

            try:
                cpp_embed_tokens = getattr(self._cpp_runtime_module, "embed_tokens_aten", None)
                if self._cpp_runtime_enabled and cpp_embed_tokens is not None:
                    inputs_embeds = cpp_embed_tokens(module.language_model, input_ids)
                else:
                    inputs_embeds = module.get_input_embeddings()(input_ids)

                image_embeds, deepstack_image_embeds = module.get_image_features(pixel_values, image_grid_thw)
                if not torch.is_tensor(image_embeds):
                    image_embeds = image_embeds[0] if len(image_embeds) == 1 else torch.cat(image_embeds, dim=0)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                n_image_tokens = int(image_embeds.shape[0])
                seq_len = int(input_ids.shape[1])
                image_token_id = int(module.config.image_token_id)

                connector_nvtx = False
                if self._prefill_nvtx_enabled and self._nvtx_prefill_active:
                    torch.cuda.nvtx.range_push("prefill/module.connector")
                    connector_nvtx = True

                start = self._get_single_image_token_span(
                    module,
                    input_ids,
                    image_token_id,
                    seq_len,
                    n_image_tokens,
                )
                module._aicas_image_token_start = int(start)

                if n_image_tokens > 0:
                    inputs_embeds[:, int(start): int(start) + n_image_tokens, :].copy_(
                        image_embeds.view(1, n_image_tokens, -1)
                    )
                module.language_model._aicas_visual_token_offset = int(start)
                module.language_model._aicas_visual_token_count = int(n_image_tokens)
                module.language_model._aicas_visual_token_seq_len = int(seq_len)
                image_mask_2d = _get_cached_visual_mask(module, inputs_embeds.device, seq_len, int(start), n_image_tokens)
                module.language_model._aicas_visual_pos_mask_ptr = int(image_mask_2d.data_ptr())
                module.language_model._aicas_visual_pos_mask_shape = tuple(int(x) for x in image_mask_2d.shape)
                visual_pos_masks = image_mask_2d
                deepstack_visual_embeds = deepstack_image_embeds

                if position_ids is None:
                    position_ids, rope_deltas = _build_single_image_rope(
                        module,
                        input_ids,
                        inputs_embeds,
                        image_grid_thw,
                        seq_len,
                        start,
                        n_image_tokens,
                    )
                    module.rope_deltas = rope_deltas
            finally:
                if "connector_nvtx" in locals() and connector_nvtx:
                    torch.cuda.nvtx.range_pop()

            outputs = module.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

            return modeling_qwen3_vl.Qwen3VLModelOutputWithPast(
                last_hidden_state=outputs.last_hidden_state,
                past_key_values=outputs.past_key_values,
                rope_deltas=module.rope_deltas,
            )

        qwen_model_cls.forward = _patched_forward
        qwen_model_cls._aicas_fast_connector_scatter = True
        self._mark_optimization("fp16_fast_connector_scatter")

    def _enable_triton_decode_rmsnorm_rope(self):
        """Patch decode text attention so narrow decode gates can replace specific substeps."""
        enable_decode_qk_rmsnorm_rope = True
        enable_decode_qk_rmsnorm_rope_gqa2 = True
        enable_decode_qk_kv_update = True
        enable_decode_qkv_linear_pack = self._env_flag("AICAS_ENABLE_DECODE_QKV_LINEAR_PACK")
        enable_int4_decode_qkv_gemv = (
            enable_decode_qkv_linear_pack
            and self._in_repo_kernels_enabled
            and self._env_flag("AICAS_ENABLE_DECODE_INT4_QKV")
        )
        enable_fp8_decode_qkv_gemv = (
            enable_decode_qkv_linear_pack
            and self._decode_fp8_qkv_enabled
            and (not enable_int4_decode_qkv_gemv or self._decode_int4_qkv_quant_policy_value == "gptq")
        )
        enable_cuda_fp8_decode_qkv_gemv = enable_fp8_decode_qkv_gemv
        qkv_gptq_online_fp8_calibration = self._decode_int4_qkv_gptq_online_enabled
        if qkv_gptq_online_fp8_calibration:
            enable_cuda_fp8_decode_qkv_gemv = True
        enable_cuda_qkv_qk_update = (
            (enable_cuda_fp8_decode_qkv_gemv or enable_int4_decode_qkv_gemv)
            and self._env_flag("AICAS_ENABLE_CUDA_DECODE_QKV_QK_UPDATE")
        )
        enable_cuda_qkv_qk_update_rope_cache = (
            enable_cuda_qkv_qk_update
            and self._env_flag("AICAS_ENABLE_CUDA_DECODE_QKV_QK_UPDATE_ROPE_CACHE")
        )
        if enable_cuda_qkv_qk_update_rope_cache:
            if not self._env_flag("AICAS_ENABLE_CACHED_DECODE_ROPE"):
                raise ValueError("CUDA qkv qk update rope-cache path requires AICAS_ENABLE_CACHED_DECODE_ROPE=1")
        enable_decode_prefix_attention = self._env_flag("AICAS_ENABLE_DECODE_PREFIX_ATTENTION")
        enable_o_add_rmsnorm_gemv = True
        enable_cuda_prefix_stage2_o_fusion = self._env_flag("AICAS_ENABLE_CUDA_DECODE_PREFIX_STAGE2_O_FUSION")
        cuda_decode_prefix_attention_num_threads = self._env_int("AICAS_CUDA_DECODE_PREFIX_ATTENTION_NUM_THREADS")
        decode_qk_block_d = self._env_int("AICAS_TRITON_DECODE_QK_RMSNORM_ROPE_BLOCK_D")
        decode_qk_num_warps = self._env_int("AICAS_TRITON_DECODE_QK_RMSNORM_ROPE_NUM_WARPS")
        decode_qk_num_stages = self._env_int("AICAS_TRITON_DECODE_QK_RMSNORM_ROPE_NUM_STAGES")
        decode_qk_kv_block_d = self._env_int("AICAS_TRITON_DECODE_QK_KV_UPDATE_BLOCK_D")
        decode_qk_kv_num_warps = self._env_int("AICAS_TRITON_DECODE_QK_KV_UPDATE_NUM_WARPS")
        decode_qk_kv_num_stages = self._env_int("AICAS_TRITON_DECODE_QK_KV_UPDATE_NUM_STAGES")
        cuda_fp8_decode_qkv_rows_per_block = self._env_int("AICAS_CUDA_DECODE_QKV_FP8_ROWS_PER_BLOCK")
        cuda_fp8_decode_qkv_mode = "fp8e4b15-block128-hscale-cg-static-k2048"
        cuda_int4_decode_qkv_rows_per_block = self._env_int("AICAS_CUDA_DECODE_QKV_INT4_ROWS_PER_BLOCK")
        cuda_int4_decode_qkv_mode = "int4-sym-kblock-kscale-fullwarp-u16-static-k2048"
        qkv_int4_quant_policy = self._decode_int4_quant_policy("AICAS_DECODE_INT4_QKV_QUANT_POLICY")
        cuda_qkv_qk_update_num_threads = self._env_int("AICAS_CUDA_DECODE_QKV_QK_UPDATE_NUM_THREADS")
        cuda_qkv_qk_update_mode = "kvquad-static-contig"
        want_decode_attention_patch = (
            self._in_repo_kernels_enabled
            and (
                enable_decode_qk_rmsnorm_rope
                or enable_decode_qkv_linear_pack
                or enable_decode_prefix_attention
                or enable_o_add_rmsnorm_gemv
            )
        )
        if not want_decode_attention_patch:
            return

        from transformers.models.qwen3_vl import modeling_qwen3_vl

        fused_decode_qk_rmsnorm_rope_gqa2 = None
        fused_decode_qk_rmsnorm_rope_kv_update_gqa2 = None
        if enable_decode_qk_rmsnorm_rope:
            from kernels.fused_decode_qk_rmsnorm_rope_gqa2 import (
                fused_decode_qk_rmsnorm_rope_gqa2,
                fused_decode_qk_rmsnorm_rope_kv_update_gqa2,
            )
        attention_cls = modeling_qwen3_vl.Qwen3VLTextAttention
        if getattr(attention_cls, "_aicas_triton_decode_text_attention", False):
            if enable_decode_qk_rmsnorm_rope:
                self._mark_optimization("triton_decode_qk_rmsnorm_rope")
            if enable_decode_qk_kv_update:
                self._mark_optimization("triton_decode_qk_kv_update")
            if enable_decode_qkv_linear_pack:
                self._mark_optimization("decode_qkv_linear_pack")
            if enable_cuda_fp8_decode_qkv_gemv:
                self._mark_optimization("cuda_decode_qkv_gemv_fp8")
            if enable_int4_decode_qkv_gemv:
                self._mark_optimization("cuda_decode_qkv_gemv_int4")
            if enable_cuda_qkv_qk_update:
                self._mark_optimization("cuda_decode_qkv_qk_update")
            if enable_cuda_qkv_qk_update_rope_cache:
                self._mark_optimization("cuda_decode_qkv_qk_update_rope_cache")
            if enable_decode_prefix_attention:
                self._mark_optimization("cuda_decode_prefix_attention")
            if enable_cuda_prefix_stage2_o_fusion:
                self._mark_optimization("cuda_decode_prefix_stage2_o_fusion")
                self._mark_optimization("cuda_decode_prefix_stage1:h2dot_direct_qcache_2d")
            if enable_o_add_rmsnorm_gemv:
                self._mark_optimization("triton_decode_o_add_rmsnorm_gemv")
            return

        original_forward = attention_cls.forward
        fused_decode_prefix_attention_gqa2 = None
        if enable_decode_prefix_attention:
            from kernels.cuda.prefix_attention import (
                cuda_decode_prefix_attention_bucketed_gqa2 as fused_decode_prefix_attention_gqa2,
                cuda_decode_prefix_attention_bucket_config,
                cuda_decode_prefix_attention_stage1_h2dot_direct_qcache_2d_gqa2,
                make_prefix_attention_bucketed_workspace,
            )
        cuda_decode_qkv_gemv_fp8 = None
        cuda_decode_qkv_gemv_int4 = None
        cuda_decode_qkv_qk_update = None
        cuda_decode_qkv_qk_update_rope_cache = None
        if enable_cuda_fp8_decode_qkv_gemv:
            from kernels.cuda import cuda_decode_qkv_gemv_fp8
        if enable_int4_decode_qkv_gemv:
            from kernels.cuda import cuda_decode_qkv_gemv_int4_sym_kblock_kscale as cuda_decode_qkv_gemv_int4
        if enable_cuda_qkv_qk_update:
            from kernels.cuda import cuda_decode_qkv_qk_rmsnorm_rope_kv_update_gqa2 as cuda_decode_qkv_qk_update
            if enable_cuda_qkv_qk_update_rope_cache:
                from kernels.cuda import (
                    cuda_decode_qkv_qk_rmsnorm_rope_cache_kv_update_gqa2
                    as cuda_decode_qkv_qk_update_rope_cache,
                )
        if enable_decode_qkv_linear_pack:
            for module in self._model.modules():
                if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj")):
                    self._get_fused_linear_fp8_pack(
                        module,
                        (module.q_proj, module.k_proj, module.v_proj),
                        self._model.dtype,
                        "_aicas_decode_qkv_linear_pack",
                        "_aicas_decode_qkv_linear_fp8_pack",
                    )

        def _patched_forward(
            module,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            is_decode_step = (
                not module.training
                and hidden_states.is_cuda
                and hidden_states.dtype == torch.float16
                and hidden_states.shape[1] == 1
            )
            if not is_decode_step:
                return original_forward(
                    module,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            if self._decode_int4_qkv_gptq_online_building:
                self._maybe_collect_qkv_gptq_online_activation(module, hidden_states)

            if enable_decode_qkv_linear_pack:
                qkv_int4_runtime_ready = self._decode_int4_qkv_runtime_ready()
                if enable_int4_decode_qkv_gemv and qkv_int4_runtime_ready:
                    fused_w_int4, fused_scales, fused_b, split_sizes = self._get_fused_linear_int4_qkv_pack(
                        module,
                        (module.q_proj, module.k_proj, module.v_proj),
                        hidden_states.dtype,
                        "_aicas_decode_qkv_linear_pack",
                        "_aicas_decode_qkv_linear_int4_pack",
                        activation_samples=hidden_states,
                    )
                    if cuda_decode_qkv_gemv_int4 is None:
                        raise RuntimeError("CUDA decode qkv INT4 kernel is required")
                    qkv_states = cuda_decode_qkv_gemv_int4(
                        hidden_states,
                        fused_w_int4,
                        fused_scales,
                        fused_b,
                        rows_per_block=cuda_int4_decode_qkv_rows_per_block,
                        mode=cuda_int4_decode_qkv_mode,
                    )
                elif enable_fp8_decode_qkv_gemv:
                    if enable_int4_decode_qkv_gemv and not self._decode_int4_qkv_gptq_online_building:
                        raise RuntimeError("qkv GPTQ INT4 pack is not ready")
                    fused_w_fp8, fused_scales, fused_b, split_sizes = self._get_fused_linear_fp8_pack(
                        module,
                        (module.q_proj, module.k_proj, module.v_proj),
                        hidden_states.dtype,
                        "_aicas_decode_qkv_linear_pack",
                        "_aicas_decode_qkv_linear_fp8_pack",
                    )
                    if cuda_decode_qkv_gemv_fp8 is None:
                        raise RuntimeError("CUDA decode qkv FP8 kernel is required")
                    qkv_states = cuda_decode_qkv_gemv_fp8(
                        hidden_states,
                        fused_w_fp8,
                        fused_scales,
                        fused_b,
                        rows_per_block=cuda_fp8_decode_qkv_rows_per_block,
                        mode=cuda_fp8_decode_qkv_mode,
                    )
                q_states, k_states, v_states = torch.split(qkv_states, split_sizes, dim=-1)
            else:
                raise RuntimeError("decode qkv linear pack is required")

            query_states = q_states.view(hidden_shape).transpose(1, 2)
            key_states = k_states.view(hidden_shape).transpose(1, 2)
            value_states = v_states.view(hidden_shape).transpose(1, 2)
            if len(position_embeddings) == 4:
                cos, sin, rope_cache, rope_position = position_embeddings
            else:
                cos, sin = position_embeddings
                rope_cache = None
                rope_position = None

            use_qk_kv_update = False
            use_qk_gqa2 = (
                enable_decode_qk_rmsnorm_rope_gqa2
                and query_states.shape[1] == 2 * key_states.shape[1]
            )
            cache_layer = None
            if (
                enable_decode_qk_kv_update
                and fused_decode_qk_rmsnorm_rope_kv_update_gqa2 is not None
                and past_key_values is not None
                and cache_position is not None
                and torch.is_tensor(cache_position)
                and cache_position.numel() == 1
                and hasattr(past_key_values, "layers")
                and module.layer_idx < len(past_key_values.layers)
            ):
                cache_layer = past_key_values.layers[module.layer_idx]
                if not getattr(cache_layer, "is_initialized", False):
                    cache_layer.lazy_initialization(key_states)
                use_qk_kv_update = (
                    getattr(cache_layer, "is_initialized", False)
                    and getattr(cache_layer, "keys", None) is not None
                    and getattr(cache_layer, "values", None) is not None
                    and cache_layer.keys.shape[0] == key_states.shape[0]
                    and cache_layer.keys.shape[1] == key_states.shape[1]
                    and cache_layer.keys.shape[3] == key_states.shape[3]
                    and hasattr(cache_layer, "max_cache_len")
                    and int(cache_layer.keys.shape[2]) == int(cache_layer.max_cache_len)
                    and cache_layer.keys.dtype == key_states.dtype
                    and cache_layer.values.dtype == value_states.dtype
                )

            use_qk_kv_update = (
                use_qk_kv_update
                and use_qk_gqa2
                and fused_decode_qk_rmsnorm_rope_kv_update_gqa2 is not None
            )
            use_cuda_qkv_qk_update = (
                use_qk_kv_update
                and enable_cuda_qkv_qk_update
                and cuda_decode_qkv_qk_update is not None
                and enable_decode_qkv_linear_pack
                and (enable_cuda_fp8_decode_qkv_gemv or enable_int4_decode_qkv_gemv)
                and "qkv_states" in locals()
                and query_states.shape[0] == 1
                and query_states.shape[1] == 2 * key_states.shape[1]
                and query_states.shape[3] == 128
            )
            if use_cuda_qkv_qk_update:
                query_out = torch.empty_like(query_states)
                if enable_cuda_qkv_qk_update_rope_cache:
                    if cuda_decode_qkv_qk_update_rope_cache is None or rope_cache is None or rope_position is None:
                        raise RuntimeError("CUDA qkv qk update rope-cache kernel is required")
                    query_states = cuda_decode_qkv_qk_update_rope_cache(
                        qkv_states,
                        cache_layer.keys,
                        cache_layer.values,
                        cache_position,
                        module.q_norm.weight,
                        module.q_norm.variance_epsilon,
                        module.k_norm.weight,
                        module.k_norm.variance_epsilon,
                        rope_cache,
                        rope_position,
                        out_query=query_out,
                        num_threads=cuda_qkv_qk_update_num_threads,
                        mode=cuda_qkv_qk_update_mode,
                    )
                else:
                    query_states = cuda_decode_qkv_qk_update(
                        qkv_states,
                        cache_layer.keys,
                        cache_layer.values,
                        cache_position,
                        module.q_norm.weight,
                        module.q_norm.variance_epsilon,
                        module.k_norm.weight,
                        module.k_norm.variance_epsilon,
                        cos,
                        sin,
                        out_query=query_out,
                        num_threads=cuda_qkv_qk_update_num_threads,
                        mode=cuda_qkv_qk_update_mode,
                    )
                key_states, value_states = cache_layer.keys, cache_layer.values
            elif use_qk_kv_update:
                query_states = fused_decode_qk_rmsnorm_rope_kv_update_gqa2(
                    cache_layer.keys,
                    cache_layer.values,
                    query_states,
                    key_states,
                    value_states,
                    cache_position,
                    module.q_norm.weight,
                    module.q_norm.variance_epsilon,
                    module.k_norm.weight,
                    module.k_norm.variance_epsilon,
                    cos,
                    sin,
                    out_query=query_states,
                    block_d=decode_qk_kv_block_d,
                    num_warps=decode_qk_kv_num_warps,
                    num_stages=decode_qk_kv_num_stages,
                )
                key_states, value_states = cache_layer.keys, cache_layer.values
            elif enable_decode_qk_rmsnorm_rope and use_qk_gqa2 and fused_decode_qk_rmsnorm_rope_gqa2 is not None:
                query_states, key_states = fused_decode_qk_rmsnorm_rope_gqa2(
                    query_states,
                    key_states,
                    module.q_norm.weight,
                    module.q_norm.variance_epsilon,
                    module.k_norm.weight,
                    module.k_norm.variance_epsilon,
                    cos,
                    sin,
                    out_query=query_states,
                    out_key=key_states,
                    block_d=decode_qk_block_d,
                    num_warps=decode_qk_num_warps,
                    num_stages=decode_qk_num_stages,
                )
            else:
                raise RuntimeError("decode q/k RMSNorm + RoPE requires fused GQA2 kernels")

            if past_key_values is not None and not use_qk_kv_update:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, module.layer_idx, cache_kwargs
                )

            use_prefix_attention = (
                enable_decode_prefix_attention
                and fused_decode_prefix_attention_gqa2 is not None
                and cache_position is not None
                and torch.is_tensor(cache_position)
                and cache_position.numel() == 1
                and query_states.shape[1] == 2 * key_states.shape[1]
                and key_states.ndim == 4
                and value_states.ndim == 4
            )
            use_cuda_prefix_stage2_o_fusion = (
                enable_cuda_prefix_stage2_o_fusion
                and (
                    (not self._decode_int4_o_gptq_online_enabled)
                    or self._decode_int4_o_gptq_online_ready
                )
            )
            if use_prefix_attention:
                bucket_len = int(key_states.shape[2])
                split_m, _kind = cuda_decode_prefix_attention_bucket_config(bucket_len)
                active_splits_value = (bucket_len + split_m - 1) // split_m
                workspace_key = (
                    int(query_states.shape[1]),
                    int(key_states.shape[1]),
                    bucket_len,
                    int(query_states.shape[3]),
                    query_states.device.index,
                )
                workspace_cache = getattr(module, "_aicas_cuda_prefix_attention_workspace_cache", None)
                if workspace_cache is None:
                    workspace_cache = {}
                    module._aicas_cuda_prefix_attention_workspace_cache = workspace_cache
                workspace = workspace_cache.get(workspace_key)
                if workspace is None:
                    workspace = make_prefix_attention_bucketed_workspace(
                        query_states,
                        key_states,
                        prefix_len=bucket_len,
                    )
                    workspace_cache[workspace_key] = workspace
                if use_cuda_prefix_stage2_o_fusion:
                    cuda_decode_prefix_attention_stage1_h2dot_direct_qcache_2d_gqa2(
                        query_states,
                        key_states,
                        value_states,
                        cache_position,
                        module.scaling,
                        split_m=split_m,
                        num_threads=cuda_decode_prefix_attention_num_threads,
                        workspace=workspace,
                        active_splits=active_splits_value,
                    )
                    module._aicas_decode_prefix_stage2_o_workspace = (
                        workspace[0],
                        workspace[1],
                        workspace[2],
                        active_splits_value,
                    )
                    attn_output = query_states.reshape(-1)
                else:
                    attn_output = fused_decode_prefix_attention_gqa2(
                        query_states,
                        key_states,
                        value_states,
                        cache_position,
                        module.scaling,
                        num_threads=cuda_decode_prefix_attention_num_threads,
                        workspace=workspace,
                        output_layout="flat",
                    )
            else:
                raise RuntimeError(
                    "AICAS_ENABLE_DECODE_PREFIX_ATTENTION requires decode q_len=1, "
                    "GQA2 cache tensors, and scalar cache_position."
                )
            attn_weights = None

            if not use_cuda_prefix_stage2_o_fusion:
                module._aicas_decode_raw_attn_output = attn_output
            return attn_output, attn_weights

        attention_cls.forward = _patched_forward
        attention_cls._aicas_triton_decode_text_attention = True
        if enable_decode_qk_rmsnorm_rope:
            self._mark_optimization("triton_decode_qk_rmsnorm_rope")
        if enable_decode_qk_kv_update:
            self._mark_optimization("triton_decode_qk_kv_update")
        if enable_decode_qkv_linear_pack:
            self._mark_optimization("decode_qkv_linear_pack")
        if enable_cuda_fp8_decode_qkv_gemv:
            self._mark_optimization("cuda_decode_qkv_gemv_fp8")
        if enable_int4_decode_qkv_gemv:
            self._mark_optimization("cuda_decode_qkv_gemv_int4")
            if qkv_int4_quant_policy == "gptq":
                self._mark_optimization("cuda_decode_qkv_gemv_int4_gptq_pack")
        if enable_cuda_qkv_qk_update:
            self._mark_optimization(f"cuda_decode_qkv_qk_update:{cuda_qkv_qk_update_mode}")
        if enable_decode_prefix_attention:
            self._mark_optimization("cuda_decode_prefix_attention")
        if enable_cuda_prefix_stage2_o_fusion:
            self._mark_optimization("cuda_decode_prefix_stage2_o_fusion")
            self._mark_optimization("cuda_decode_prefix_stage1:h2dot_direct_qcache_2d")
        if enable_o_add_rmsnorm_gemv:
            self._mark_optimization("triton_decode_o_add_rmsnorm_gemv")

    def _enable_triton_decode_add_rmsnorm(self):
        """Fuse decode-only residual add followed by local or next-layer RMSNorm."""
        if not (self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_TRITON_DECODE_ADD_RMSNORM")):
            return

        from transformers.models.qwen3_vl import modeling_qwen3_vl

        decoder_cls = modeling_qwen3_vl.Qwen3VLTextDecoderLayer
        interlayer_fuse_enabled = self._env_flag("AICAS_ENABLE_TRITON_DECODE_INTERLAYER_ADD_RMSNORM")
        enable_down_add_rmsnorm_gemv = True
        enable_o_add_rmsnorm_gemv = True
        enable_fp8_down_add_rmsnorm_gemv = enable_down_add_rmsnorm_gemv and self._decode_fp8_down_add_rmsnorm_enabled
        enable_fp8_o_add_rmsnorm_gemv = enable_o_add_rmsnorm_gemv and self._decode_fp8_o_add_rmsnorm_enabled
        enable_int4_down_add_rmsnorm_gemv = (
            enable_down_add_rmsnorm_gemv and self._decode_int4_down_add_rmsnorm_enabled
        )
        enable_int4_o_add_rmsnorm_gemv = enable_o_add_rmsnorm_gemv and self._decode_int4_o_add_rmsnorm_enabled
        down_gemv_fp8_backend = "cuda"
        o_gemv_fp8_backend = "cuda"
        down_gemv_cuda_reduce_backend = self._cuda_decode_gemv_reduce_backend(
            "AICAS_CUDA_DECODE_DOWN_ADD_RMSNORM_REDUCE_BACKEND"
        )
        o_gemv_cuda_reduce_backend = self._cuda_decode_gemv_reduce_backend(
            "AICAS_CUDA_DECODE_O_ADD_RMSNORM_REDUCE_BACKEND"
        )
        cuda_down_gemv_fp8_variant = 27
        cuda_o_gemv_fp8_variant = 32
        enable_fp8_kblock_major = self._env_flag("AICAS_ENABLE_DECODE_FP8_KBLOCK_MAJOR")
        down_gemv_fp8_uses_kscale = True
        o_gemv_fp8_uses_kscale = False
        cuda_decode_gemv_add_rmsnorm_fp8_kblock_kscale = None
        cuda_decode_gemv_add_rmsnorm_fp8_kblock_major = None
        cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_kblock_major = None
        cuda_decode_gemv_add_rmsnorm_int4 = None
        cuda_decode_prefix_stage2_o_add_rmsnorm_int4 = None
        if enable_fp8_down_add_rmsnorm_gemv or enable_fp8_o_add_rmsnorm_gemv:
            if enable_fp8_down_add_rmsnorm_gemv and down_gemv_fp8_backend == "cuda":
                if not enable_fp8_kblock_major:
                    raise ValueError("CUDA decode down/add RMSNorm FP8 requires k-block-major FP8 weights")
                from kernels.cuda import (
                    cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_kscale
                    as cuda_decode_gemv_add_rmsnorm_fp8_kblock_kscale,
                )
            if enable_fp8_o_add_rmsnorm_gemv and o_gemv_fp8_backend == "cuda":
                if not enable_fp8_kblock_major:
                    raise ValueError("CUDA decode o/add RMSNorm FP8 requires k-block-major FP8 weights")
                from kernels.cuda import (
                    cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_major
                    as cuda_decode_gemv_add_rmsnorm_fp8_kblock_major,
                    cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_e4b15_kblock_major
                    as cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_kblock_major,
                )
        if enable_int4_down_add_rmsnorm_gemv or enable_int4_o_add_rmsnorm_gemv:
            from kernels.cuda import (
                cuda_decode_gemv_add_rmsnorm_int4_sym_kblock_kscale_halfwarp
                as cuda_decode_gemv_add_rmsnorm_int4,
            )
            if enable_int4_o_add_rmsnorm_gemv:
                from kernels.cuda import (
                    cuda_decode_prefix_stage2_o_add_rmsnorm_int4_sym_kblock_kscale_halfwarp
                    as cuda_decode_prefix_stage2_o_add_rmsnorm_int4,
                )
        down_gemv_fp8_block_n = self._env_int("AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_BLOCK_N")
        down_gemv_fp8_block_k = self._env_int("AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_BLOCK_K")
        down_gemv_fp8_split_k = self._env_int("AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_SPLIT_K")
        down_gemv_fp8_num_warps = self._env_int("AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_NUM_WARPS")
        down_gemv_fp8_num_stages = self._env_int("AICAS_TRITON_DECODE_DOWN_ADD_RMSNORM_FP8_NUM_STAGES")
        down_gemv_cuda_reduce_threads = self._env_int("AICAS_CUDA_DECODE_DOWN_ADD_RMSNORM_REDUCE_THREADS")
        down_gemv_int4_block_n = self._env_int("AICAS_CUDA_DECODE_INT4_DOWN_ADD_RMSNORM_BLOCK_N")
        down_gemv_int4_split_k = self._env_int("AICAS_CUDA_DECODE_INT4_DOWN_ADD_RMSNORM_SPLIT_K")
        down_gemv_int4_reduce_backend = self._cuda_decode_gemv_reduce_backend(
            "AICAS_CUDA_DECODE_INT4_DOWN_ADD_RMSNORM_REDUCE_BACKEND"
        )
        down_gemv_int4_reduce_threads = self._env_int("AICAS_CUDA_DECODE_INT4_DOWN_ADD_RMSNORM_REDUCE_THREADS")
        o_gemv_fp8_block_n = self._env_int("AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_BLOCK_N")
        o_gemv_fp8_block_k = self._env_int("AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_BLOCK_K")
        o_gemv_fp8_split_k = self._env_int("AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_SPLIT_K")
        o_gemv_fp8_num_warps = self._env_int("AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_NUM_WARPS")
        o_gemv_fp8_num_stages = self._env_int("AICAS_TRITON_DECODE_O_ADD_RMSNORM_FP8_NUM_STAGES")
        o_gemv_cuda_reduce_threads = self._env_int("AICAS_CUDA_DECODE_O_ADD_RMSNORM_REDUCE_THREADS")
        o_gemv_int4_block_n = self._env_int("AICAS_CUDA_DECODE_INT4_O_ADD_RMSNORM_BLOCK_N")
        o_gemv_int4_split_k = self._env_int("AICAS_CUDA_DECODE_INT4_O_ADD_RMSNORM_SPLIT_K")
        o_gemv_int4_reduce_backend = self._cuda_decode_gemv_reduce_backend(
            "AICAS_CUDA_DECODE_INT4_O_ADD_RMSNORM_REDUCE_BACKEND"
        )
        o_gemv_int4_reduce_threads = self._env_int("AICAS_CUDA_DECODE_INT4_O_ADD_RMSNORM_REDUCE_THREADS")
        down_int4_quant_policy = self._decode_int4_quant_policy(
            "AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_QUANT_POLICY"
        )
        o_int4_quant_policy = self._decode_int4_quant_policy("AICAS_DECODE_INT4_O_ADD_RMSNORM_QUANT_POLICY")
        enable_gate_up_linear_pack = self._env_flag("AICAS_ENABLE_DECODE_GATE_UP_LINEAR_PACK")
        enable_fp8_gate_up_swiglu = enable_gate_up_linear_pack and self._decode_fp8_gate_up_enabled
        enable_int4_gate_up_swiglu = enable_gate_up_linear_pack and self._decode_int4_gate_up_enabled
        enable_int4_gate_up_swiglu_initial = enable_int4_gate_up_swiglu and self._decode_int4_gate_up_runtime_ready()
        cuda_gate_up_swiglu_fp8 = None
        if enable_down_add_rmsnorm_gemv and enable_fp8_gate_up_swiglu:
            from kernels.cuda.decode_gate_up_fp8_e4b15 import (
                cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_hscale
                as cuda_gate_up_swiglu_fp8,
            )
        cuda_gate_up_swiglu_fp8_rows_per_block = self._env_int(
            "AICAS_CUDA_DECODE_FP8_GATE_UP_SWIGLU_ROWS_PER_BLOCK"
        )
        cuda_gate_up_swiglu_int4_rows_per_block = self._env_int(
            "AICAS_CUDA_DECODE_INT4_GATE_UP_SWIGLU_ROWS_PER_BLOCK"
        )
        cuda_gate_up_swiglu_int4 = None
        if enable_down_add_rmsnorm_gemv and enable_int4_gate_up_swiglu:
            from kernels.cuda import cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp

            cuda_gate_up_swiglu_int4 = cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp

        def _patch_final_norm_forward(final_norm, owner):
            if final_norm is None or getattr(final_norm, "_aicas_triton_decode_final_norm_cache", False):
                return
            original_forward = final_norm.forward

            def _patched_final_norm(module, hidden_states: torch.Tensor):
                cached_norm = self._consume_final_norm_cache(owner, hidden_states)
                if cached_norm is not None:
                    return cached_norm
                return original_forward(hidden_states)

            final_norm.forward = MethodType(_patched_final_norm, final_norm)
            final_norm._aicas_triton_decode_final_norm_cache = True
            final_norm._aicas_owner_text_model = owner

        def _wire_decoder_layers():
            text_models = [m for m in self._model.modules() if hasattr(m, "layers") and hasattr(m, "norm")]
            for text_model in text_models:
                decoder_layers = list(getattr(text_model, "layers", []))
                if not decoder_layers:
                    continue
                final_norm = getattr(text_model, "norm", None)
                self._clear_final_norm_cache(text_model)
                if interlayer_fuse_enabled:
                    _patch_final_norm_forward(final_norm, text_model)
                for i, layer in enumerate(decoder_layers):
                    next_layer = decoder_layers[i + 1] if i + 1 < len(decoder_layers) else None
                    layer._aicas_layer_idx = i
                    layer._aicas_use_int4_gate_up_swiglu = enable_int4_gate_up_swiglu_initial
                    if hasattr(layer, "mlp"):
                        layer.mlp._aicas_layer_idx = i
                        layer.mlp._aicas_use_int4_gate_up_swiglu = layer._aicas_use_int4_gate_up_swiglu
                    layer._aicas_triton_decode_interlayer_add_rmsnorm = interlayer_fuse_enabled
                    layer._aicas_next_decoder_layer = next_layer if interlayer_fuse_enabled else None
                    layer._aicas_next_input_layernorm = (
                        getattr(next_layer, "input_layernorm", None)
                        if interlayer_fuse_enabled and next_layer is not None
                        else None
                    )
                    layer._aicas_final_norm = (
                        final_norm if interlayer_fuse_enabled and i + 1 == len(decoder_layers) else None
                    )
                    layer._aicas_owner_text_model = text_model
                    layer._aicas_cached_pre_norm = None
                    if enable_fp8_o_add_rmsnorm_gemv and hasattr(layer.self_attn, "o_proj"):
                        self._get_fp8_weight_pack(
                            layer.self_attn.o_proj,
                            layer.self_attn.o_proj.weight,
                            "_aicas_decode_o_proj_fp8_pack",
                            kblock_major=enable_fp8_kblock_major,
                            kscale=enable_fp8_kblock_major and o_gemv_fp8_uses_kscale,
                        )
                    if enable_fp8_down_add_rmsnorm_gemv and hasattr(layer.mlp, "down_proj"):
                        self._get_fp8_weight_pack(
                            layer.mlp.down_proj,
                            layer.mlp.down_proj.weight,
                            "_aicas_decode_down_proj_fp8_pack",
                            kblock_major=enable_fp8_kblock_major,
                            kscale=enable_fp8_kblock_major and down_gemv_fp8_uses_kscale,
                        )
                    if layer._aicas_use_int4_gate_up_swiglu and hasattr(layer.mlp, "gate_proj") and hasattr(
                        layer.mlp, "up_proj"
                    ):
                        self._get_fused_linear_int4_gate_up_pack(
                            layer.mlp,
                            (layer.mlp.gate_proj, layer.mlp.up_proj),
                            self._model.dtype,
                            "_aicas_decode_gate_up_linear_pack",
                            "_aicas_decode_gate_up_linear_int4_pack",
                        )

        if getattr(decoder_cls, "_aicas_triton_decode_add_rmsnorm", False):
            _wire_decoder_layers()
            self._mark_optimization("triton_decode_add_rmsnorm")
            if interlayer_fuse_enabled:
                self._mark_optimization("triton_decode_interlayer_add_rmsnorm")
                self._mark_optimization("triton_decode_final_add_rmsnorm")
                if enable_down_add_rmsnorm_gemv:
                    self._mark_optimization("triton_decode_down_add_rmsnorm_gemv")
            if enable_o_add_rmsnorm_gemv:
                self._mark_optimization("triton_decode_o_add_rmsnorm_gemv")
            if enable_fp8_o_add_rmsnorm_gemv:
                self._mark_optimization(f"{o_gemv_fp8_backend}_decode_o_add_rmsnorm_gemv_fp8")
                if enable_fp8_kblock_major:
                    self._mark_optimization(f"{o_gemv_fp8_backend}_decode_o_add_rmsnorm_gemv_fp8_kblock")
                    if o_gemv_fp8_uses_kscale:
                        self._mark_optimization(
                            f"{o_gemv_fp8_backend}_decode_o_add_rmsnorm_gemv_fp8_"
                            "kblock"
                        )
            if enable_int4_o_add_rmsnorm_gemv:
                self._mark_optimization("cuda_decode_o_add_rmsnorm_gemv_int4")
                self._mark_optimization("cuda_decode_o_add_rmsnorm_gemv_int4_kblock_kscale")
            if enable_down_add_rmsnorm_gemv:
                if enable_int4_gate_up_swiglu:
                    self._mark_optimization("cuda_decode_gate_up_swiglu_int4")
                    self._mark_optimization("cuda_decode_gate_up_swiglu_int4_kblock_kscale")
                    if self._decode_int4_gate_up_quant_policy() == "gptq":
                        self._mark_optimization("cuda_decode_gate_up_swiglu_int4_gptq_pack")
                if enable_fp8_gate_up_swiglu:
                    self._mark_optimization("cuda_decode_gate_up_swiglu_fp8")
                    self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_kblock_kscale")
                    self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_halfwarp")
                    self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_halfwarp_hscale")
                if enable_fp8_down_add_rmsnorm_gemv:
                    self._mark_optimization(f"{down_gemv_fp8_backend}_decode_down_add_rmsnorm_gemv_fp8")
                    if enable_fp8_kblock_major:
                        self._mark_optimization(f"{down_gemv_fp8_backend}_decode_down_add_rmsnorm_gemv_fp8_kblock")
                        if down_gemv_fp8_uses_kscale:
                            self._mark_optimization(
                                f"{down_gemv_fp8_backend}_decode_down_add_rmsnorm_gemv_fp8_"
                                "kblock_kscale_nomask"
                            )
                if enable_int4_down_add_rmsnorm_gemv:
                    self._mark_optimization("cuda_decode_down_add_rmsnorm_gemv_int4")
                    self._mark_optimization("cuda_decode_down_add_rmsnorm_gemv_int4_kblock_kscale")
            return

        original_forward = decoder_cls.forward

        def _patched_forward(
            module,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ) -> torch.Tensor:
            is_decode_step = (
                not module.training
                and hidden_states.is_cuda
                and hidden_states.dtype == torch.float16
                and hidden_states.shape[1] == 1
            )
            if not is_decode_step:
                module._aicas_cached_pre_norm = None
                final_norm_module = getattr(module, "_aicas_final_norm", None)
                if final_norm_module is not None:
                    owner = getattr(module, "_aicas_owner_text_model", None)
                    self._clear_final_norm_cache(owner)
                return original_forward(
                    module,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

            interlayer_enabled = bool(getattr(module, "_aicas_triton_decode_interlayer_add_rmsnorm", False))
            cached_pre_norm = getattr(module, "_aicas_cached_pre_norm", None)
            use_cached_pre_norm = interlayer_enabled and cached_pre_norm is not None
            module._aicas_cached_pre_norm = None

            residual = hidden_states
            if use_cached_pre_norm:
                hidden_states = cached_pre_norm
            else:
                hidden_states = module.input_layernorm(hidden_states)
            hidden_states, _ = module.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            raw_attn_output = getattr(module.self_attn, "_aicas_decode_raw_attn_output", None)
            module.self_attn._aicas_decode_raw_attn_output = None
            prefix_stage2_o_workspace = getattr(module.self_attn, "_aicas_decode_prefix_stage2_o_workspace", None)
            module.self_attn._aicas_decode_prefix_stage2_o_workspace = None
            if (
                enable_o_add_rmsnorm_gemv
                and (
                    cuda_decode_gemv_add_rmsnorm_int4 is not None
                    or cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_kblock_major is not None
                    or cuda_decode_prefix_stage2_o_add_rmsnorm_int4 is not None
                    or cuda_decode_gemv_add_rmsnorm_fp8_kblock_major is not None
                )
                and (raw_attn_output is hidden_states or prefix_stage2_o_workspace is not None)
                and hasattr(module.self_attn, "o_proj")
            ):
                if self._decode_int4_o_gptq_online_building and raw_attn_output is not None:
                    if getattr(self, "_decode_int4_o_gptq_online_rows", None) is not None:
                        self._maybe_collect_o_gptq_online_activation(module, raw_attn_output)
                o_int4_runtime_ready = (
                    (not self._decode_int4_o_gptq_online_enabled) or self._decode_int4_o_gptq_online_ready
                )
                if enable_int4_o_add_rmsnorm_gemv and o_int4_runtime_ready:
                    if cuda_decode_gemv_add_rmsnorm_int4 is None:
                        raise RuntimeError("Decode o/add RMSNorm int4 kernel is required")
                    o_weight_int4, o_scales = self._get_int4_weight_pack(
                        module.self_attn.o_proj,
                        module.self_attn.o_proj.weight,
                        "_aicas_decode_o_proj_int4_pack",
                        activation_samples=None if prefix_stage2_o_workspace is not None else raw_attn_output,
                    )
                    if prefix_stage2_o_workspace is not None:
                        if cuda_decode_prefix_stage2_o_add_rmsnorm_int4 is None:
                            raise RuntimeError("prefix stage2/O INT4 fusion kernel is required")
                        partial_m, partial_l, partial_acc, active_splits_value = prefix_stage2_o_workspace
                        hidden_states, normed = cuda_decode_prefix_stage2_o_add_rmsnorm_int4(
                            partial_m,
                            partial_l,
                            partial_acc,
                            o_weight_int4,
                            o_scales,
                            residual,
                            module.post_attention_layernorm.weight,
                            module.post_attention_layernorm.variance_epsilon,
                            module.self_attn.o_proj.bias,
                            active_splits=int(active_splits_value),
                            block_n=o_gemv_int4_block_n,
                            block_k=128,
                            num_warps=8,
                            num_stages=3,
                            split_k=4,
                            row_blocks_per_cta=-22,
                            reduce_backend=o_gemv_int4_reduce_backend,
                            reduce_threads=o_gemv_int4_reduce_threads,
                        )
                    else:
                        hidden_states, normed = cuda_decode_gemv_add_rmsnorm_int4(
                            raw_attn_output,
                            o_weight_int4,
                            o_scales,
                            residual,
                            module.post_attention_layernorm.weight,
                            module.post_attention_layernorm.variance_epsilon,
                            module.self_attn.o_proj.bias,
                            block_n=o_gemv_int4_block_n,
                            block_k=128,
                            num_warps=8,
                            num_stages=3,
                            split_k=o_gemv_int4_split_k,
                            reduce_backend=o_gemv_int4_reduce_backend,
                            reduce_threads=o_gemv_int4_reduce_threads,
                        )
                elif enable_fp8_o_add_rmsnorm_gemv and (
                    not enable_int4_o_add_rmsnorm_gemv or not self._decode_int4_o_gptq_online_ready
                ):
                    if enable_int4_o_add_rmsnorm_gemv and not self._decode_int4_o_gptq_online_building:
                        raise RuntimeError("O GPTQ INT4 pack is not ready")
                    decode_o_fp8 = cuda_decode_gemv_add_rmsnorm_fp8_kblock_major
                    if decode_o_fp8 is None:
                        raise RuntimeError("Decode o/add RMSNorm FP8 kernel is required")
                    o_weight_fp8, o_scales = self._get_fp8_weight_pack(
                        module.self_attn.o_proj,
                        module.self_attn.o_proj.weight,
                        "_aicas_decode_o_proj_fp8_pack",
                        kblock_major=enable_fp8_kblock_major,
                        kscale=enable_fp8_kblock_major and o_gemv_fp8_uses_kscale,
                    )
                    if prefix_stage2_o_workspace is not None:
                        if cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_kblock_major is None:
                            raise RuntimeError("prefix stage2/O fusion requires CUDA FP8 O backend")
                        partial_m, partial_l, partial_acc, active_splits_value = prefix_stage2_o_workspace
                        hidden_states, normed = cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_kblock_major(
                            partial_m,
                            partial_l,
                            partial_acc,
                            o_weight_fp8,
                            o_scales,
                            residual,
                            module.post_attention_layernorm.weight,
                            module.post_attention_layernorm.variance_epsilon,
                            module.self_attn.o_proj.bias,
                            active_splits=int(active_splits_value),
                            block_n=o_gemv_fp8_block_n,
                            block_k=o_gemv_fp8_block_k,
                            num_warps=o_gemv_fp8_num_warps,
                            num_stages=o_gemv_fp8_num_stages,
                            split_k=o_gemv_fp8_split_k,
                            variant=cuda_o_gemv_fp8_variant,
                            reduce_backend=o_gemv_cuda_reduce_backend,
                            reduce_threads=o_gemv_cuda_reduce_threads,
                        )
                    else:
                        hidden_states, normed = decode_o_fp8(
                            raw_attn_output,
                            o_weight_fp8,
                            o_scales,
                            residual,
                            module.post_attention_layernorm.weight,
                            module.post_attention_layernorm.variance_epsilon,
                            module.self_attn.o_proj.bias,
                            block_n=o_gemv_fp8_block_n,
                            block_k=o_gemv_fp8_block_k,
                            num_warps=o_gemv_fp8_num_warps,
                            num_stages=o_gemv_fp8_num_stages,
                            split_k=o_gemv_fp8_split_k,
                            variant=cuda_o_gemv_fp8_variant,
                            reduce_backend=o_gemv_cuda_reduce_backend,
                            reduce_threads=o_gemv_cuda_reduce_threads,
                        )
                else:
                    raise RuntimeError("decode O/add RMSNorm requires FP8 or GPTQ INT4")
            else:
                raise RuntimeError("decode O/add RMSNorm requires fused O projection output")

            residual = hidden_states
            next_layer = getattr(module, "_aicas_next_decoder_layer", None)
            next_input_ln = getattr(module, "_aicas_next_input_layernorm", None)
            final_norm_module = getattr(module, "_aicas_final_norm", None)
            can_fuse_interlayer = interlayer_enabled and next_layer is not None and next_input_ln is not None
            can_fuse_final_norm = interlayer_enabled and next_layer is None and final_norm_module is not None
            down_norm_module = next_input_ln if can_fuse_interlayer else final_norm_module if can_fuse_final_norm else None
            if (
                enable_down_add_rmsnorm_gemv
                and down_norm_module is not None
                and hasattr(module.mlp, "gate_proj")
                and hasattr(module.mlp, "up_proj")
                and hasattr(module.mlp, "down_proj")
            ):
                hidden_act = str(getattr(module.mlp.config, "hidden_act", "")).lower()
                if hidden_act in {"silu", "swish"}:
                    if not enable_gate_up_linear_pack:
                        raise RuntimeError("decode gate/up linear pack is required for fused SwiGLU")
                    self._maybe_collect_gate_up_gptq_online_activation(module.mlp, normed)
                    use_int4_gate_up = bool(getattr(module, "_aicas_use_int4_gate_up_swiglu", False))
                    if use_int4_gate_up:
                        if cuda_gate_up_swiglu_int4 is None:
                            raise RuntimeError("Decode gate/up SwiGLU int4 kernel is required")
                        fused_w_int4, fused_scales, fused_b, split_sizes = self._get_fused_linear_int4_gate_up_pack(
                            module.mlp,
                            (module.mlp.gate_proj, module.mlp.up_proj),
                            normed.dtype,
                            "_aicas_decode_gate_up_linear_pack",
                            "_aicas_decode_gate_up_linear_int4_pack",
                        )
                        if fused_b is not None:
                            raise RuntimeError("int4 gate/up SwiGLU path does not support bias")
                        hidden = cuda_gate_up_swiglu_int4(
                            normed,
                            fused_w_int4,
                            fused_scales,
                            intermediate_size=int(split_sizes[0]),
                            rows_per_block=cuda_gate_up_swiglu_int4_rows_per_block,
                            mode="int4-sym-kblock-kscale-halfwarp-xreuse-u4offset-nobcast-m6144-k2048",
                        ).view(*normed.shape[:-1], -1)
                    elif enable_fp8_gate_up_swiglu:
                        gate_up_fp8 = cuda_gate_up_swiglu_fp8
                        if gate_up_fp8 is None:
                            raise RuntimeError("Decode gate/up SwiGLU FP8 kernel is required")
                        fused_w_fp8, fused_scales, fused_b, split_sizes = self._get_fused_linear_fp8_pack(
                            module.mlp,
                            (module.mlp.gate_proj, module.mlp.up_proj),
                            normed.dtype,
                            "_aicas_decode_gate_up_linear_pack",
                            "_aicas_decode_gate_up_linear_fp8_pack",
                            kblock_major=enable_fp8_kblock_major,
                            kscale=enable_fp8_kblock_major,
                        )
                        if fused_b is not None:
                            raise RuntimeError("FP8 gate/up SwiGLU path does not support bias")
                        hidden = gate_up_fp8(
                            normed,
                            fused_w_fp8,
                            fused_scales,
                            intermediate_size=int(split_sizes[0]),
                            rows_per_block=cuda_gate_up_swiglu_fp8_rows_per_block,
                        ).view(*normed.shape[:-1], -1)
                    else:
                        raise RuntimeError("decode gate/up SwiGLU requires FP8 or GPTQ INT4")
                    if self._decode_int4_down_gptq_online_building:
                        if getattr(self, "_decode_int4_down_gptq_online_rows", None) is not None:
                            self._maybe_collect_down_gptq_online_activation(module, hidden)
                    down_int4_runtime_ready = self._decode_int4_down_runtime_ready()
                    if enable_int4_down_add_rmsnorm_gemv and down_int4_runtime_ready:
                        if cuda_decode_gemv_add_rmsnorm_int4 is None:
                            raise RuntimeError("Decode down/add RMSNorm int4 kernel is required")
                        down_weight_int4, down_scales = self._get_int4_weight_pack(
                            module.mlp.down_proj,
                            module.mlp.down_proj.weight,
                            "_aicas_decode_down_proj_int4_pack",
                            activation_samples=hidden,
                        )
                        hidden_states, next_pre_norm = cuda_decode_gemv_add_rmsnorm_int4(
                            hidden,
                            down_weight_int4,
                            down_scales,
                            residual,
                            down_norm_module.weight,
                            down_norm_module.variance_epsilon,
                            module.mlp.down_proj.bias,
                            block_n=down_gemv_int4_block_n,
                            block_k=128,
                            num_warps=8,
                            num_stages=3,
                            split_k=down_gemv_int4_split_k,
                            reduce_backend=down_gemv_int4_reduce_backend,
                            reduce_threads=down_gemv_int4_reduce_threads,
                        )
                    elif enable_fp8_down_add_rmsnorm_gemv and (
                        not enable_int4_down_add_rmsnorm_gemv or not self._decode_int4_down_gptq_online_ready
                    ):
                        if enable_int4_down_add_rmsnorm_gemv and not self._decode_int4_down_gptq_online_building:
                            raise RuntimeError("down GPTQ INT4 pack is not ready")
                        decode_down_fp8 = cuda_decode_gemv_add_rmsnorm_fp8_kblock_kscale
                        if decode_down_fp8 is None:
                            raise RuntimeError("Decode down/add RMSNorm FP8 kernel is required")
                        down_weight_fp8, down_scales = self._get_fp8_weight_pack(
                            module.mlp.down_proj,
                            module.mlp.down_proj.weight,
                            "_aicas_decode_down_proj_fp8_pack",
                            kblock_major=enable_fp8_kblock_major,
                            kscale=enable_fp8_kblock_major and down_gemv_fp8_uses_kscale,
                        )
                        hidden_states, next_pre_norm = decode_down_fp8(
                            hidden,
                            down_weight_fp8,
                            down_scales,
                            residual,
                            down_norm_module.weight,
                            down_norm_module.variance_epsilon,
                            module.mlp.down_proj.bias,
                            block_n=down_gemv_fp8_block_n,
                            block_k=down_gemv_fp8_block_k,
                            num_warps=down_gemv_fp8_num_warps,
                            num_stages=down_gemv_fp8_num_stages,
                            split_k=down_gemv_fp8_split_k,
                            variant=cuda_down_gemv_fp8_variant,
                            reduce_backend=down_gemv_cuda_reduce_backend,
                            reduce_threads=down_gemv_cuda_reduce_threads,
                        )
                    else:
                        raise RuntimeError("decode down/add RMSNorm requires FP8 or GPTQ INT4")
                    if can_fuse_interlayer:
                        next_layer._aicas_cached_pre_norm = next_pre_norm
                    else:
                        owner = getattr(module, "_aicas_owner_text_model", None)
                        self._store_final_norm_cache(owner, hidden_states, next_pre_norm)
                    return hidden_states

            raise RuntimeError("decode MLP requires fused gate/up + down/add RMSNorm path")

        decoder_cls.forward = _patched_forward
        decoder_cls._aicas_triton_decode_add_rmsnorm = True
        _wire_decoder_layers()
        self._mark_optimization("triton_decode_add_rmsnorm")
        if interlayer_fuse_enabled:
            self._mark_optimization("triton_decode_interlayer_add_rmsnorm")
            self._mark_optimization("triton_decode_final_add_rmsnorm")
        if enable_down_add_rmsnorm_gemv:
            self._mark_optimization("triton_decode_down_add_rmsnorm_gemv")
        if enable_fp8_down_add_rmsnorm_gemv:
            self._mark_optimization(f"{down_gemv_fp8_backend}_decode_down_add_rmsnorm_gemv_fp8")
            if enable_fp8_kblock_major:
                self._mark_optimization(f"{down_gemv_fp8_backend}_decode_down_add_rmsnorm_gemv_fp8_kblock")
                if down_gemv_fp8_uses_kscale:
                    self._mark_optimization(
                        f"{down_gemv_fp8_backend}_decode_down_add_rmsnorm_gemv_fp8_"
                        "kblock_kscale_nomask"
                    )
        if enable_int4_down_add_rmsnorm_gemv:
            self._mark_optimization("cuda_decode_down_add_rmsnorm_gemv_int4")
            self._mark_optimization("cuda_decode_down_add_rmsnorm_gemv_int4_kblock_kscale")
            if down_int4_quant_policy == "gptq":
                self._mark_optimization("cuda_decode_down_add_rmsnorm_gemv_int4_gptq_pack")
        if enable_o_add_rmsnorm_gemv:
            self._mark_optimization("triton_decode_o_add_rmsnorm_gemv")
        if enable_fp8_o_add_rmsnorm_gemv:
            self._mark_optimization(f"{o_gemv_fp8_backend}_decode_o_add_rmsnorm_gemv_fp8")
            if enable_fp8_kblock_major:
                self._mark_optimization(f"{o_gemv_fp8_backend}_decode_o_add_rmsnorm_gemv_fp8_kblock")
                if o_gemv_fp8_uses_kscale:
                    self._mark_optimization(
                        f"{o_gemv_fp8_backend}_decode_o_add_rmsnorm_gemv_fp8_"
                        "kblock"
                    )
        if enable_int4_o_add_rmsnorm_gemv:
            self._mark_optimization("cuda_decode_o_add_rmsnorm_gemv_int4")
            self._mark_optimization("cuda_decode_o_add_rmsnorm_gemv_int4_kblock_kscale")
            if o_int4_quant_policy == "gptq":
                self._mark_optimization("cuda_decode_o_add_rmsnorm_gemv_int4_gptq_pack")
        if enable_down_add_rmsnorm_gemv:
            if enable_int4_gate_up_swiglu:
                self._mark_optimization("cuda_decode_gate_up_swiglu_int4")
                self._mark_optimization("cuda_decode_gate_up_swiglu_int4_kblock_kscale")
                if self._decode_int4_gate_up_quant_policy() == "gptq":
                    self._mark_optimization("cuda_decode_gate_up_swiglu_int4_gptq_pack")
            if enable_fp8_gate_up_swiglu:
                self._mark_optimization("cuda_decode_gate_up_swiglu_fp8")
                self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_kblock_kscale")
                self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_halfwarp")
                self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_halfwarp_hscale")

    def _enable_decode_gate_up_linear_pack(self):
        """Pack decode-only gate/up projections and optionally fuse SwiGLU."""
        if not (self._in_repo_kernels_enabled and self._env_flag("AICAS_ENABLE_DECODE_GATE_UP_LINEAR_PACK")):
            return

        from transformers.models.qwen3_vl import modeling_qwen3_vl

        mlp_cls = modeling_qwen3_vl.Qwen3VLTextMLP
        if getattr(mlp_cls, "_aicas_decode_gate_up_linear_pack_only", False):
            self._mark_optimization("decode_gate_up_linear_pack")
            return

        original_forward = mlp_cls.forward
        enable_fp8_gate_up_swiglu = self._decode_fp8_gate_up_enabled
        enable_int4_gate_up_swiglu = self._decode_int4_gate_up_enabled
        enable_int4_gate_up_swiglu_initial = enable_int4_gate_up_swiglu and self._decode_int4_gate_up_runtime_ready()
        enable_fp8_kblock_major = self._env_flag("AICAS_ENABLE_DECODE_FP8_KBLOCK_MAJOR")
        cuda_gate_up_swiglu_fp8_rows_per_block = self._env_int(
            "AICAS_CUDA_DECODE_FP8_GATE_UP_SWIGLU_ROWS_PER_BLOCK"
        )
        cuda_gate_up_swiglu_int4_rows_per_block = self._env_int(
            "AICAS_CUDA_DECODE_INT4_GATE_UP_SWIGLU_ROWS_PER_BLOCK"
        )
        cuda_decode_gate_up_swiglu_fp8 = None
        cuda_decode_gate_up_swiglu_int4 = None
        if enable_int4_gate_up_swiglu:
            from kernels.cuda import cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp

            cuda_decode_gate_up_swiglu_int4 = cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp
        if enable_fp8_gate_up_swiglu:
            from kernels.cuda.decode_gate_up_fp8_e4b15 import (
                cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale_halfwarp_hscale
                as cuda_decode_gate_up_swiglu_fp8,
            )

        for module in self._model.modules():
            if all(hasattr(module, name) for name in ("gate_proj", "up_proj")):
                use_int4_for_module = enable_int4_gate_up_swiglu_initial
                module._aicas_use_int4_gate_up_swiglu = use_int4_for_module
                if use_int4_for_module:
                    self._get_fused_linear_int4_gate_up_pack(
                        module,
                        (module.gate_proj, module.up_proj),
                        self._model.dtype,
                        "_aicas_decode_gate_up_linear_pack",
                        "_aicas_decode_gate_up_linear_int4_pack",
                    )
                if enable_fp8_gate_up_swiglu and not use_int4_for_module:
                    self._get_fused_linear_fp8_pack(
                        module,
                        (module.gate_proj, module.up_proj),
                        self._model.dtype,
                        "_aicas_decode_gate_up_linear_pack",
                        "_aicas_decode_gate_up_linear_fp8_pack",
                        kblock_major=enable_fp8_kblock_major,
                        kscale=enable_fp8_kblock_major,
                    )

        def _patched_forward(module, x):
            hidden_act = str(getattr(module.config, "hidden_act", "")).lower()
            is_decode_step = (
                not module.training
                and x.is_cuda
                and x.dtype == torch.float16
                and x.shape[1] == 1
                and hidden_act in {"silu", "swish"}
            )
            if not is_decode_step:
                return original_forward(module, x)

            self._maybe_collect_gate_up_gptq_online_activation(module, x)
            use_int4_gate_up = bool(getattr(module, "_aicas_use_int4_gate_up_swiglu", enable_int4_gate_up_swiglu))
            if use_int4_gate_up:
                if cuda_decode_gate_up_swiglu_int4 is None:
                    raise RuntimeError("Decode gate/up SwiGLU int4 kernel is required")
                fused_w_int4, fused_scales, fused_b, split_sizes = self._get_fused_linear_int4_gate_up_pack(
                    module,
                    (module.gate_proj, module.up_proj),
                    x.dtype,
                    "_aicas_decode_gate_up_linear_pack",
                    "_aicas_decode_gate_up_linear_int4_pack",
                )
                if fused_b is not None:
                    raise RuntimeError("int4 gate/up SwiGLU path does not support bias")
                hidden = cuda_decode_gate_up_swiglu_int4(
                    x,
                    fused_w_int4,
                    fused_scales,
                    intermediate_size=int(split_sizes[0]),
                    rows_per_block=cuda_gate_up_swiglu_int4_rows_per_block,
                    mode="int4-sym-kblock-kscale-halfwarp-xreuse-u4offset-nobcast-m6144-k2048",
                ).view(*x.shape[:-1], -1)
            elif enable_fp8_gate_up_swiglu:
                gate_up_fp8 = cuda_decode_gate_up_swiglu_fp8
                if gate_up_fp8 is None:
                    raise RuntimeError("Decode gate/up SwiGLU FP8 kernel is required")
                fused_w_fp8, fused_scales, fused_b, split_sizes = self._get_fused_linear_fp8_pack(
                    module,
                    (module.gate_proj, module.up_proj),
                    x.dtype,
                    "_aicas_decode_gate_up_linear_pack",
                    "_aicas_decode_gate_up_linear_fp8_pack",
                    kblock_major=enable_fp8_kblock_major,
                    kscale=enable_fp8_kblock_major,
                )
                if fused_b is not None:
                    raise RuntimeError("FP8 gate/up SwiGLU path does not support bias")
                hidden = gate_up_fp8(
                    x,
                    fused_w_fp8,
                    fused_scales,
                    intermediate_size=int(split_sizes[0]),
                    rows_per_block=cuda_gate_up_swiglu_fp8_rows_per_block,
                ).view(*x.shape[:-1], -1)
            else:
                raise RuntimeError("decode gate/up SwiGLU requires FP8 or GPTQ INT4")
            return module.down_proj(hidden)

        mlp_cls.forward = _patched_forward
        mlp_cls._aicas_decode_gate_up_linear_pack_only = True
        self._mark_optimization("decode_gate_up_linear_pack")
        if enable_int4_gate_up_swiglu:
            self._mark_optimization("cuda_decode_gate_up_swiglu_int4")
            self._mark_optimization("cuda_decode_gate_up_swiglu_int4_kblock_kscale")
            if self._decode_int4_gate_up_quant_policy() == "gptq":
                self._mark_optimization("cuda_decode_gate_up_swiglu_int4_gptq_pack")
            if enable_fp8_gate_up_swiglu:
                self._mark_optimization("cuda_decode_gate_up_swiglu_fp8")
                self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_kblock_kscale")
                self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_halfwarp")
                self._mark_optimization("cuda_decode_gate_up_swiglu_fp8_halfwarp_hscale")

    def _get_decode_weight(self, linear_module, out_dtype: torch.dtype, cache: bool = True):
        weight = linear_module.weight
        if weight.dtype == out_dtype and weight.is_contiguous():
            return weight

        if not cache:
            return weight.to(dtype=out_dtype).contiguous()
        cache = getattr(linear_module, "_aicas_decode_weight_cache", {})
        cache_key = str(out_dtype)
        if cache_key in cache:
            return cache[cache_key]
        cast_weight = weight.to(dtype=out_dtype).contiguous()
        cache[cache_key] = cast_weight
        linear_module._aicas_decode_weight_cache = cache
        return cast_weight

    def _get_decode_bias(self, linear_module, out_dtype: torch.dtype):
        bias = getattr(linear_module, "bias", None)
        if bias is None:
            return None
        if bias.dtype == out_dtype and bias.is_contiguous():
            return bias
        cache = getattr(linear_module, "_aicas_decode_bias_cache", {})
        cache_key = str(out_dtype)
        if cache_key in cache:
            return cache[cache_key]
        cast_bias = bias.to(dtype=out_dtype).contiguous()
        cache[cache_key] = cast_bias
        linear_module._aicas_decode_bias_cache = cache
        return cast_bias

    def _get_fused_linear_pack(self, owner_module, linear_modules, out_dtype: torch.dtype, cache_attr: str):
        cache = getattr(owner_module, cache_attr, {})
        cache_key = str(out_dtype)
        if cache_key in cache:
            return cache[cache_key]

        weights = []
        biases = []
        split_sizes = []
        has_bias = False
        for linear in linear_modules:
            # Fused path keeps a single concatenated cache to avoid duplicated per-layer dequant buffers.
            w = self._get_decode_weight(linear, out_dtype, cache=False)
            b = self._get_decode_bias(linear, out_dtype)
            split_sizes.append(int(w.shape[0]))
            weights.append(w)
            biases.append(b)
            has_bias = has_bias or b is not None

        fused_weight = torch.cat(weights, dim=0).contiguous()
        if has_bias:
            fused_bias_parts = []
            for linear, bias in zip(linear_modules, biases):
                if bias is not None:
                    fused_bias_parts.append(bias)
                else:
                    fused_bias_parts.append(
                        torch.zeros((int(linear.weight.shape[0]),), device=fused_weight.device, dtype=out_dtype)
                    )
            fused_bias = torch.cat(fused_bias_parts, dim=0).contiguous()
        else:
            fused_bias = None

        pack = (fused_weight, fused_bias, tuple(split_sizes))
        cache[cache_key] = pack
        setattr(owner_module, cache_attr, cache)
        return pack

    def _get_fp8_weight_pack(
        self,
        owner_module,
        weight: torch.Tensor,
        cache_attr: str,
        *,
        kblock_major: bool = False,
        kscale: bool = False,
    ):
        effective_cache_attr = cache_attr + ("_kblock_kscale" if kscale else "_kblock" if kblock_major else "")
        cache = getattr(owner_module, effective_cache_attr, None)
        if cache is not None:
            return cache
        from kernels.fp8_e4b15 import triton_quantize_e4b15_block128

        weight_fp8, scales = triton_quantize_e4b15_block128(weight)
        if kblock_major:
            n_out, k_in = int(weight_fp8.shape[0]), int(weight_fp8.shape[1])
            if k_in % 128 != 0:
                raise ValueError("FP8 k-block-major pack requires K divisible by 128")
            scale_blocks = k_in // 128
            weight_fp8 = weight_fp8.view(n_out, k_in // 128, 128).permute(1, 0, 2).contiguous()
            if kscale:
                scales = scales.view(n_out, scale_blocks).permute(1, 0).contiguous()
        pack = (weight_fp8, scales)
        setattr(owner_module, effective_cache_attr, pack)
        return pack

    def _get_int4_weight_pack(
        self,
        owner_module,
        weight: torch.Tensor,
        cache_attr: str,
        *,
        activation_samples: torch.Tensor | None = None,
    ):
        cache = getattr(owner_module, cache_attr, None)
        if cache is None:
            cache = getattr(owner_module, f"{cache_attr}_gptq", None)
        if cache is not None:
            return cache
        if weight.ndim != 2:
            raise ValueError("INT4 weight pack expects [N, K] weight")
        n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
        if n_out != 2048 or k_in % 128 != 0:
            raise ValueError("INT4 GEMV pack expects N=2048 and K divisible by 128")
        if activation_samples is None:
            raise RuntimeError("GPTQ INT4 weight pack requires activation_samples")
        from kernels.fused_decode_qkv_gemv_fp8 import quantize_int4_sym_block128_gptq_kblock_pack

        packed_weight, packed_scales = quantize_int4_sym_block128_gptq_kblock_pack(
            weight,
            activation_samples.reshape(-1, k_in).contiguous(),
            damp_percent=self._env_float("AICAS_DECODE_INT4_GPTQ_DAMP_PERCENT"),
        )
        pack = (packed_weight, packed_scales)
        setattr(owner_module, cache_attr, pack)
        return pack

    def _get_lm_head_shortlist_int4_pack(
        self,
        owner_module,
        weight: torch.Tensor,
        cache_attr: str,
        *,
        activation_samples: torch.Tensor | None = None,
    ):
        cache = getattr(owner_module, cache_attr, None)
        if cache is None:
            cache = getattr(owner_module, f"{cache_attr}_gptq", None)
        if cache is not None:
            return cache
        if weight.ndim != 2:
            raise ValueError("lm_head INT4 shortlist pack expects [N, K] weight")
        n_out, k_in = int(weight.shape[0]), int(weight.shape[1])
        if k_in != 2048:
            raise ValueError("lm_head INT4 shortlist pack expects K=2048")
        if activation_samples is None:
            raise RuntimeError("GPTQ lm_head INT4 shortlist pack requires activation_samples")
        from kernels.fused_decode_qkv_gemv_fp8 import quantize_int4_sym_block128_gptq_kblock_pack

        packed_weight, packed_scales = quantize_int4_sym_block128_gptq_kblock_pack(
            weight,
            activation_samples.reshape(-1, k_in).contiguous(),
            damp_percent=self._env_float("AICAS_DECODE_INT4_GPTQ_DAMP_PERCENT"),
        )
        pack = (packed_weight, packed_scales)
        setattr(owner_module, cache_attr, pack)
        return pack

    def _get_gate_up_fp8_kblock_pack(self, owner_module, weight_fp8: torch.Tensor, scales: torch.Tensor, split_sizes):
        cache_attr = "_aicas_decode_gate_up_linear_fp8_kblock_pack"
        cache = getattr(owner_module, cache_attr, None)
        if cache is not None:
            return cache
        intermediate = int(split_sizes[0])
        k_in = int(weight_fp8.shape[1])
        if k_in % 128 != 0:
            raise ValueError("gate/up FP8 k-block-major pack requires K divisible by 128")
        scale_blocks = k_in // 128
        gate = weight_fp8[:intermediate].view(intermediate, scale_blocks, 128)
        up = weight_fp8[intermediate : 2 * intermediate].view(intermediate, scale_blocks, 128)
        packed_weight = torch.stack((gate, up), dim=1).permute(2, 1, 0, 3).contiguous()
        pack = (packed_weight, scales)
        setattr(owner_module, cache_attr, pack)
        return pack

    def _get_gate_up_fp8_kblock_kscale_pack(
        self, owner_module, weight_fp8: torch.Tensor, scales: torch.Tensor, split_sizes
    ):
        cache_attr = "_aicas_decode_gate_up_linear_fp8_kblock_kscale_pack"
        cache = getattr(owner_module, cache_attr, None)
        if cache is not None:
            return cache
        packed_weight, row_scales = self._get_gate_up_fp8_kblock_pack(owner_module, weight_fp8, scales, split_sizes)
        intermediate = int(split_sizes[0])
        scale_blocks = int(row_scales.shape[1])
        gate_scales = row_scales[:intermediate].view(intermediate, scale_blocks)
        up_scales = row_scales[intermediate : 2 * intermediate].view(intermediate, scale_blocks)
        packed_scales = torch.stack((gate_scales, up_scales), dim=1).permute(2, 1, 0).contiguous()
        pack = (packed_weight, packed_scales)
        setattr(owner_module, cache_attr, pack)
        return pack

    def _get_gate_up_int4_kblock_kscale_pack(
        self,
        owner_module,
        weight_int4: torch.Tensor,
        scales: torch.Tensor,
        split_sizes,
    ):
        cache_attr = "_aicas_decode_gate_up_linear_int4_kblock_kscale_pack"
        cache = getattr(owner_module, cache_attr, None)
        if cache is not None:
            return cache
        intermediate = int(split_sizes[0])
        k_packed = int(weight_int4.shape[1])
        if k_packed % 64 != 0:
            raise ValueError("gate/up int4 k-block pack requires packed K divisible by 64")
        scale_blocks = k_packed // 64
        gate = weight_int4[:intermediate].view(intermediate, scale_blocks, 64)
        up = weight_int4[intermediate : 2 * intermediate].view(intermediate, scale_blocks, 64)
        gate_scales = scales[:intermediate].view(intermediate, scale_blocks)
        up_scales = scales[intermediate : 2 * intermediate].view(intermediate, scale_blocks)
        packed_weight = torch.stack((gate, up), dim=1).permute(2, 1, 0, 3).contiguous()
        packed_scales = torch.stack((gate_scales, up_scales), dim=1).permute(2, 1, 0).contiguous()
        pack = (packed_weight, packed_scales)
        setattr(owner_module, cache_attr, pack)
        return pack

    def _get_gate_up_int4_gptq_rowmajor_pack(
        self,
        owner_module,
        fused_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_attr = "_aicas_decode_gate_up_linear_int4_gptq_rowmajor_pack"
        cache = getattr(owner_module, cache_attr, None)
        if cache is not None:
            return cache

        pack_mode = self._decode_int4_gate_up_gptq_pack_mode()
        if pack_mode == "warmup-online":
            payload = self._decode_int4_gate_up_gptq_online_payload
            if not self._decode_int4_gate_up_gptq_online_ready or payload is None:
                raise RuntimeError("online gate/up GPTQ pack is not ready")
            layer_idx = int(getattr(owner_module, "_aicas_layer_idx", -1))
            if not isinstance(payload, dict):
                raise ValueError("gate/up GPTQ online payload must be a dict")
            layers = payload.get("layers") if isinstance(payload, dict) else None
            if not isinstance(layers, dict):
                raise ValueError("gate/up GPTQ online payload must contain a 'layers' dict")
            layer_payload = layers.get(str(layer_idx), layers.get(layer_idx))
            if not isinstance(layer_payload, dict):
                raise ValueError(f"gate/up GPTQ online payload missing layer {layer_idx}")
            weight_int4 = layer_payload.get("weight_int4")
            scales = layer_payload.get("scales")
            if not torch.is_tensor(weight_int4) or not torch.is_tensor(scales):
                raise ValueError(f"gate/up GPTQ online payload layer {layer_idx} must contain tensors")
            weight_int4 = weight_int4.to(device=fused_w.device, dtype=torch.uint8).contiguous()
            scales = scales.to(device=fused_w.device, dtype=torch.float16).contiguous()
        else:
            if not self._decode_int4_gate_up_gptq_online_ready:
                raise RuntimeError("prebuilt gate/up GPTQ pack is not installed")
            cache = getattr(owner_module, cache_attr, None)
            if cache is None:
                raise RuntimeError("prebuilt gate/up GPTQ pack did not install this layer")
            return cache

        if weight_int4.dtype != torch.uint8 or scales.dtype != torch.float16:
            raise ValueError("gate/up GPTQ row-major pack must be uint8 weight_int4 and fp16 scales")
        expected_weight_shape = (int(fused_w.shape[0]), int(fused_w.shape[1]) // 2)
        expected_scales_shape = (int(fused_w.shape[0]), int(fused_w.shape[1]) // 128)
        if tuple(weight_int4.shape) != expected_weight_shape:
            raise ValueError(
                f"gate/up GPTQ weight_int4 must be {expected_weight_shape}, got {tuple(weight_int4.shape)}"
            )
        if tuple(scales.shape) != expected_scales_shape:
            raise ValueError(f"gate/up GPTQ scales must be {expected_scales_shape}, got {tuple(scales.shape)}")
        pack = (weight_int4, scales)
        setattr(owner_module, cache_attr, pack)
        return pack

    def _install_gate_up_gptq_online_payload(self, payload: dict, source_label: str = "warmup_online") -> None:
        if source_label not in {"warmup_online", "prebuilt"}:
            raise ValueError("gate/up GPTQ source_label must be warmup_online or prebuilt")
        if not isinstance(payload, dict):
            raise ValueError("online gate/up GPTQ payload must be a dict")
        layers_payload = payload.get("layers")
        if not isinstance(layers_payload, dict):
            raise ValueError("online gate/up GPTQ payload must contain a 'layers' dict")
        for layer_idx, module in enumerate(self._gate_up_mlp_modules()):
            module._aicas_layer_idx = int(getattr(module, "_aicas_layer_idx", layer_idx))
            layer_key = str(int(module._aicas_layer_idx))
            layer_payload = layers_payload.get(layer_key, layers_payload.get(int(module._aicas_layer_idx)))
            if not isinstance(layer_payload, dict):
                raise ValueError(f"online gate/up GPTQ payload missing layer {layer_key}")
            weight_int4 = layer_payload.get("weight_int4")
            scales = layer_payload.get("scales")
            if not torch.is_tensor(weight_int4) or not torch.is_tensor(scales):
                raise ValueError(f"online gate/up GPTQ payload layer {layer_key} missing tensors")
            module._aicas_decode_gate_up_linear_int4_gptq_rowmajor_pack = (
                weight_int4.to(device=module.gate_proj.weight.device, dtype=torch.uint8).contiguous(),
                scales.to(device=module.gate_proj.weight.device, dtype=torch.float16).contiguous(),
            )
            for cache_attr in (
                "_aicas_decode_gate_up_linear_int4_kblock_kscale_pack",
                "_aicas_decode_gate_up_linear_int4_pack",
            ):
                if hasattr(module, cache_attr):
                    delattr(module, cache_attr)
            module._aicas_use_int4_gate_up_swiglu = True
        for layer in self._model.modules():
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj") and hasattr(layer.mlp, "up_proj"):
                layer._aicas_use_int4_gate_up_swiglu = True
                layer.mlp._aicas_use_int4_gate_up_swiglu = True
        self._decode_int4_gate_up_gptq_online_payload = payload
        self._decode_int4_gate_up_gptq_online_ready = True
        self._reset_decode_graph_state_after_gate_up_pack_update()
        self._mark_optimization(f"cuda_decode_gate_up_swiglu_int4_gptq_{source_label}")

    def _install_qkv_gptq_online_payload(self, payload: dict, source_label: str = "warmup_online") -> None:
        if source_label not in {"warmup_online", "prebuilt"}:
            raise ValueError("qkv GPTQ source_label must be warmup_online or prebuilt")
        if not isinstance(payload, dict):
            raise ValueError("online qkv GPTQ payload must be a dict")
        layers_payload = payload.get("layers")
        if not isinstance(layers_payload, dict):
            raise ValueError("online qkv GPTQ payload must contain a 'layers' dict")
        for module in self._attention_modules():
            layer_idx = int(getattr(module, "_aicas_layer_idx", getattr(module, "layer_idx", -1)))
            if layer_idx < 0:
                continue
            layer_key = str(layer_idx)
            layer_payload = layers_payload.get(layer_key, layers_payload.get(layer_idx))
            if not isinstance(layer_payload, dict):
                raise ValueError(f"online qkv GPTQ payload missing layer {layer_key}")
            weight_int4 = layer_payload.get("weight_int4")
            scales = layer_payload.get("scales")
            fused_b = layer_payload.get("bias")
            split_sizes = layer_payload.get("split_sizes")
            if not torch.is_tensor(weight_int4) or not torch.is_tensor(scales):
                raise ValueError(f"online qkv GPTQ payload layer {layer_key} missing tensors")
            if not isinstance(split_sizes, (list, tuple)):
                raise ValueError(f"online qkv GPTQ payload layer {layer_key} missing split_sizes")
            if fused_b is not None and not torch.is_tensor(fused_b):
                raise ValueError(f"online qkv GPTQ payload layer {layer_key} bias must be a tensor or None")
            module._aicas_decode_qkv_linear_int4_pack_gptq = (
                weight_int4.to(device=module.q_proj.weight.device, dtype=torch.uint8).contiguous(),
                scales.to(device=module.q_proj.weight.device, dtype=torch.float16).contiguous(),
                None
                if fused_b is None
                else fused_b.to(device=module.q_proj.weight.device, dtype=torch.float16).contiguous(),
                tuple(int(x) for x in split_sizes),
            )
        self._decode_int4_qkv_gptq_online_payload = payload
        self._decode_int4_qkv_gptq_online_ready = True
        self._reset_decode_graph_state_after_gate_up_pack_update()
        self._mark_optimization(f"cuda_decode_qkv_gemv_int4_gptq_{source_label}")

    def _install_down_gptq_online_payload(self, payload: dict, source_label: str = "warmup_online") -> None:
        if source_label not in {"warmup_online", "prebuilt"}:
            raise ValueError("down GPTQ source_label must be warmup_online or prebuilt")
        if not isinstance(payload, dict):
            raise ValueError("online down GPTQ payload must be a dict")
        layers_payload = payload.get("layers")
        if not isinstance(layers_payload, dict):
            raise ValueError("online down GPTQ payload must contain a 'layers' dict")
        for layer in self._decoder_layers_with_down_proj():
            layer_idx = int(getattr(layer, "_aicas_layer_idx", getattr(layer.mlp, "_aicas_layer_idx", -1)))
            if layer_idx < 0:
                continue
            layer_key = str(layer_idx)
            layer_payload = layers_payload.get(layer_key, layers_payload.get(layer_idx))
            if not isinstance(layer_payload, dict):
                raise ValueError(f"online down GPTQ payload missing layer {layer_key}")
            weight_int4 = layer_payload.get("weight_int4")
            scales = layer_payload.get("scales")
            if not torch.is_tensor(weight_int4) or not torch.is_tensor(scales):
                raise ValueError(f"online down GPTQ payload layer {layer_key} missing tensors")
            layer.mlp.down_proj._aicas_decode_down_proj_int4_pack_gptq = (
                weight_int4.to(device=layer.mlp.down_proj.weight.device, dtype=torch.uint8).contiguous(),
                scales.to(device=layer.mlp.down_proj.weight.device, dtype=torch.float16).contiguous(),
            )
        self._decode_int4_down_gptq_online_payload = payload
        self._decode_int4_down_gptq_online_ready = True
        self._reset_decode_graph_state_after_gate_up_pack_update()
        self._mark_optimization(f"cuda_decode_down_add_rmsnorm_gemv_int4_gptq_{source_label}")

    def _install_o_gptq_online_payload(self, payload: dict, source_label: str = "warmup_online") -> None:
        if source_label not in {"warmup_online", "prebuilt"}:
            raise ValueError("O GPTQ source_label must be warmup_online or prebuilt")
        if not isinstance(payload, dict):
            raise ValueError("online O GPTQ payload must be a dict")
        layers_payload = payload.get("layers")
        if not isinstance(layers_payload, dict):
            raise ValueError("online O GPTQ payload must contain a 'layers' dict")
        for module in self._model.modules():
            if not (hasattr(module, "self_attn") and hasattr(module.self_attn, "o_proj")):
                continue
            layer_idx = int(getattr(module, "_aicas_layer_idx", getattr(module.self_attn, "layer_idx", -1)))
            if layer_idx < 0:
                continue
            layer_key = str(layer_idx)
            layer_payload = layers_payload.get(layer_key, layers_payload.get(layer_idx))
            if not isinstance(layer_payload, dict):
                raise ValueError(f"online O GPTQ payload missing layer {layer_key}")
            weight_int4 = layer_payload.get("weight_int4")
            scales = layer_payload.get("scales")
            if not torch.is_tensor(weight_int4) or not torch.is_tensor(scales):
                raise ValueError(f"online O GPTQ payload layer {layer_key} missing tensors")
            module.self_attn.o_proj._aicas_decode_o_proj_int4_pack_gptq = (
                weight_int4.to(device=module.self_attn.o_proj.weight.device, dtype=torch.uint8).contiguous(),
                scales.to(device=module.self_attn.o_proj.weight.device, dtype=torch.float16).contiguous(),
            )
        self._decode_int4_o_gptq_online_payload = payload
        self._decode_int4_o_gptq_online_ready = True
        self._reset_decode_graph_state_after_gate_up_pack_update()
        self._mark_optimization(f"cuda_decode_o_add_rmsnorm_gemv_int4_gptq_{source_label}")

    def _install_lm_head_gptq_online_payload(self, payload: dict, source_label: str = "warmup_online") -> None:
        if source_label not in {"warmup_online", "prebuilt"}:
            raise ValueError("lm_head GPTQ source_label must be warmup_online or prebuilt")
        if not isinstance(payload, dict):
            raise ValueError("online lm_head GPTQ payload must be a dict")
        weight_int4 = payload.get("weight_int4")
        scales = payload.get("scales")
        if not torch.is_tensor(weight_int4) or not torch.is_tensor(scales):
            raise ValueError("online lm_head GPTQ payload must contain tensors")
        lm_head = getattr(self._model, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("online lm_head GPTQ requires lm_head")
        device = lm_head.weight.device
        self._lm_head_shortlist_weight_int4 = weight_int4.to(device=device, dtype=torch.uint8).contiguous()
        self._lm_head_shortlist_scales_int4 = scales.to(device=device, dtype=torch.float16).contiguous()
        lm_head._aicas_lm_head_shortlist_int4_pack_gptq = (
            self._lm_head_shortlist_weight_int4,
            self._lm_head_shortlist_scales_int4,
        )
        self._decode_int4_lm_head_gptq_online_payload = payload
        self._decode_int4_lm_head_gptq_online_ready = True
        self._reset_decode_graph_state_after_gate_up_pack_update()
        self._mark_optimization(f"cuda_lm_head_shortlist_top1_int4_gptq_{source_label}")

    def _install_gptq_prebuilt_payloads(self) -> None:
        if (
            self._decode_int4_qkv_enabled
            and self._decode_int4_qkv_quant_policy_value == "gptq"
            and self._decode_int4_qkv_gptq_pack_mode_value == "prebuilt"
        ):
            payload = self._load_gptq_prebuilt_payload("AICAS_DECODE_INT4_QKV_GPTQ_PACK_PATH")
            self._install_qkv_gptq_online_payload(payload, source_label="prebuilt")

        if (
            self._decode_int4_gate_up_enabled
            and self._decode_int4_gate_up_quant_policy_value == "gptq"
            and self._decode_int4_gate_up_gptq_pack_mode_value == "prebuilt"
        ):
            payload = self._load_gptq_prebuilt_payload("AICAS_DECODE_INT4_GATE_UP_GPTQ_PACK_PATH")
            self._install_gate_up_gptq_online_payload(payload, source_label="prebuilt")

        if (
            self._decode_int4_down_add_rmsnorm_enabled
            and self._decode_int4_down_quant_policy_value == "gptq"
            and self._decode_int4_down_gptq_pack_mode_value == "prebuilt"
        ):
            payload = self._load_gptq_prebuilt_payload("AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_PACK_PATH")
            self._install_down_gptq_online_payload(payload, source_label="prebuilt")

        if (
            self._decode_int4_o_add_rmsnorm_enabled
            and self._decode_int4_o_quant_policy_value == "gptq"
            and self._decode_int4_o_gptq_pack_mode_value == "prebuilt"
        ):
            payload = self._load_gptq_prebuilt_payload("AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_PACK_PATH")
            self._install_o_gptq_online_payload(payload, source_label="prebuilt")

        if (
            self._decode_int4_lm_head_enabled
            and self._decode_int4_lm_head_quant_policy_value == "gptq"
            and self._decode_int4_lm_head_gptq_pack_mode_value == "prebuilt"
        ):
            payload = self._load_gptq_prebuilt_payload("AICAS_DECODE_INT4_LM_HEAD_GPTQ_PACK_PATH")
            self._install_lm_head_gptq_online_payload(payload, source_label="prebuilt")

    def _maybe_prepare_all_gptq_online(self, prefill_kwargs: Dict[str, torch.Tensor], max_new_tokens: int) -> None:
        if int(max_new_tokens) <= 0:
            return

        need_qkv = self._decode_int4_qkv_gptq_online_enabled and not self._decode_int4_qkv_gptq_online_ready
        need_gate_up = (
            self._decode_int4_gate_up_gptq_online_enabled and not self._decode_int4_gate_up_gptq_online_ready
        )
        need_down = self._decode_int4_down_gptq_online_enabled and not self._decode_int4_down_gptq_online_ready
        need_o = self._decode_int4_o_gptq_online_enabled and not self._decode_int4_o_gptq_online_ready
        need_lm_head = (
            self._decode_int4_lm_head_gptq_online_enabled and not self._decode_int4_lm_head_gptq_online_ready
        )
        if not (need_qkv or need_gate_up or need_down or need_o or need_lm_head):
            return

        qkv_modules = self._attention_modules() if need_qkv else []
        gate_up_modules = self._gate_up_mlp_modules() if need_gate_up else []
        down_layers = self._decoder_layers_with_down_proj() if need_down else []
        o_layers = [
            module
            for module in self._model.modules()
            if hasattr(module, "self_attn") and hasattr(module.self_attn, "o_proj")
        ] if need_o else []
        lm_head = getattr(self._model, "lm_head", None) if need_lm_head else None
        if need_qkv and not qkv_modules:
            raise RuntimeError("online qkv GPTQ requires attention modules")
        if need_gate_up and not gate_up_modules:
            raise RuntimeError("online gate/up GPTQ requires MLP modules")
        if need_down and not down_layers:
            raise RuntimeError("online down GPTQ requires decoder layers with mlp.down_proj")
        if need_o and not o_layers:
            raise RuntimeError("online O GPTQ requires decoder layers with self_attn.o_proj")
        if need_lm_head and (lm_head is None or not torch.is_tensor(getattr(lm_head, "weight", None))):
            raise RuntimeError("online lm_head GPTQ requires lm_head weight")

        qkv_target = self._env_int("AICAS_DECODE_INT4_QKV_GPTQ_ONLINE_SAMPLES") if need_qkv else 0
        gate_up_target = self._env_int("AICAS_DECODE_INT4_GATE_UP_GPTQ_ONLINE_SAMPLES") if need_gate_up else 0
        down_target = self._env_int("AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_ONLINE_SAMPLES") if need_down else 0
        o_target = self._env_int("AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_ONLINE_SAMPLES") if need_o else 0
        lm_head_target = self._env_int("AICAS_DECODE_INT4_LM_HEAD_GPTQ_ONLINE_SAMPLES") if need_lm_head else 0
        for enabled, name, target in (
            (need_qkv, "AICAS_DECODE_INT4_QKV_GPTQ_ONLINE_SAMPLES", qkv_target),
            (need_gate_up, "AICAS_DECODE_INT4_GATE_UP_GPTQ_ONLINE_SAMPLES", gate_up_target),
            (need_down, "AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_ONLINE_SAMPLES", down_target),
            (need_o, "AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_ONLINE_SAMPLES", o_target),
            (need_lm_head, "AICAS_DECODE_INT4_LM_HEAD_GPTQ_ONLINE_SAMPLES", lm_head_target),
        ):
            if enabled and target <= 0:
                raise ValueError(f"{name} must be positive")
        target_tokens = max(qkv_target, gate_up_target, down_target, o_target, lm_head_target)
        if target_tokens <= 0:
            raise ValueError("enabled online GPTQ sample counts must be positive")

        input_ids = prefill_kwargs["input_ids"]
        previous_lm_head_int4_enabled = self._decode_int4_lm_head_enabled
        previous_qkv_building = self._decode_int4_qkv_gptq_online_building
        previous_gate_up_building = self._decode_int4_gate_up_gptq_online_building
        previous_down_building = self._decode_int4_down_gptq_online_building
        previous_o_building = self._decode_int4_o_gptq_online_building
        previous_lm_head_building = self._decode_int4_lm_head_gptq_online_building
        previous_layer_flags = []

        if need_qkv:
            self._decode_int4_qkv_gptq_online_building = True
            qkv_layer_indices = [
                int(getattr(module, "_aicas_layer_idx", getattr(module, "layer_idx", idx)))
                for idx, module in enumerate(qkv_modules)
            ]
            self._decode_int4_qkv_gptq_online_rows = {int(layer_idx): [] for layer_idx in qkv_layer_indices}
            self._decode_int4_qkv_gptq_online_counts = {int(layer_idx): 0 for layer_idx in qkv_layer_indices}
        if need_gate_up:
            self._decode_int4_gate_up_gptq_online_building = True
            gate_up_layer_indices = [
                int(getattr(module, "_aicas_layer_idx", idx)) for idx, module in enumerate(gate_up_modules)
            ]
            self._decode_int4_gate_up_gptq_online_rows = {int(layer_idx): [] for layer_idx in gate_up_layer_indices}
            self._decode_int4_gate_up_gptq_online_counts = {int(layer_idx): 0 for layer_idx in gate_up_layer_indices}
            for layer in self._model.modules():
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj") and hasattr(layer.mlp, "up_proj"):
                    previous_layer_flags.append(
                        (
                            layer,
                            getattr(layer, "_aicas_use_int4_gate_up_swiglu", False),
                            getattr(layer.mlp, "_aicas_use_int4_gate_up_swiglu", False),
                        )
                    )
                    layer._aicas_use_int4_gate_up_swiglu = False
                    layer.mlp._aicas_use_int4_gate_up_swiglu = False
                elif hasattr(layer, "gate_proj") and hasattr(layer, "up_proj"):
                    previous_layer_flags.append(
                        (layer, getattr(layer, "_aicas_use_int4_gate_up_swiglu", False), None)
                    )
                    layer._aicas_use_int4_gate_up_swiglu = False
        if need_down:
            self._decode_int4_down_gptq_online_building = True
            down_layer_indices = [
                int(getattr(layer, "_aicas_layer_idx", getattr(layer.mlp, "_aicas_layer_idx", idx)))
                for idx, layer in enumerate(down_layers)
            ]
            self._decode_int4_down_gptq_online_rows = {int(layer_idx): [] for layer_idx in down_layer_indices}
            self._decode_int4_down_gptq_online_counts = {int(layer_idx): 0 for layer_idx in down_layer_indices}
        if need_o:
            self._decode_int4_o_gptq_online_building = True
            o_layer_indices = [
                int(getattr(layer, "_aicas_layer_idx", getattr(layer.self_attn, "layer_idx", idx)))
                for idx, layer in enumerate(o_layers)
            ]
            self._decode_int4_o_gptq_online_rows = {int(layer_idx): [] for layer_idx in o_layer_indices}
            self._decode_int4_o_gptq_online_counts = {int(layer_idx): 0 for layer_idx in o_layer_indices}
        if need_lm_head:
            self._decode_int4_lm_head_gptq_online_building = True
            self._decode_int4_lm_head_gptq_online_rows = []
            self._decode_int4_lm_head_gptq_online_count = 0
            self._decode_int4_lm_head_enabled = False

        try:
            current_seq_len = int(input_ids.shape[1])
            state = self._create_decode_state(
                device=input_ids.device,
                input_dtype=input_ids.dtype,
                max_supported_seq_len=max(current_seq_len, self._round_seq_bucket(current_seq_len, prefill_kwargs)),
                max_decode_tokens=max(int(target_tokens), int(max_new_tokens), 2),
            )
            first_token, rope_scalar = self._prefill_first_token(prefill_kwargs, state)
            self._set_decode_start(state, first_token, current_seq_len, rope_scalar, 0)
            for _token_idx in range(int(target_tokens)):
                self._run_decode_model_and_select_token(
                    {
                        "input_ids": state["static_decode_token"],
                        "past_key_values": state["cache"],
                        "use_cache": True,
                        "return_dict": True,
                        "attention_mask": self._decode_attention_mask(state),
                        "position_ids": state["static_position_ids"],
                        "cache_position": state["static_cache_position"],
                        "logits_to_keep": 1,
                    },
                    state,
                )
                self._advance_decode_positions(state)
                done = True
                if need_qkv:
                    done = done and all(
                        int(v) >= int(qkv_target) for v in self._decode_int4_qkv_gptq_online_counts.values()
                    )
                if need_gate_up:
                    done = done and all(
                        int(v) >= int(gate_up_target)
                        for v in self._decode_int4_gate_up_gptq_online_counts.values()
                    )
                if need_down:
                    done = done and all(
                        int(v) >= int(down_target) for v in self._decode_int4_down_gptq_online_counts.values()
                    )
                if need_o:
                    done = done and all(
                        int(v) >= int(o_target) for v in self._decode_int4_o_gptq_online_counts.values()
                    )
                if need_lm_head:
                    done = done and int(self._decode_int4_lm_head_gptq_online_count) >= int(lm_head_target)
                if done:
                    break
            torch.cuda.synchronize()

            if need_qkv:
                layers_payload = {}
                for idx, module in enumerate(qkv_modules):
                    actual_layer_idx = int(getattr(module, "_aicas_layer_idx", getattr(module, "layer_idx", idx)))
                    rows = self._decode_int4_qkv_gptq_online_rows.get(actual_layer_idx, [])
                    if not rows:
                        raise RuntimeError(f"missing online qkv GPTQ activation samples for layer {actual_layer_idx}")
                    samples = torch.cat(rows, dim=0).contiguous()
                    if int(samples.shape[0]) < int(qkv_target):
                        raise RuntimeError(
                            f"online qkv GPTQ collected {int(samples.shape[0])} samples for layer "
                            f"{actual_layer_idx}, expected {int(qkv_target)}"
                        )
                    samples = samples[: int(qkv_target)].to(
                        device=module.q_proj.weight.device,
                        dtype=torch.float16,
                    ).contiguous()
                    _fused_w, fused_b, _split_sizes = self._get_fused_linear_pack(
                        module,
                        (module.q_proj, module.k_proj, module.v_proj),
                        self._model.dtype,
                        "_aicas_decode_qkv_linear_pack",
                    )
                    weight_int4, scales, _fused_b, split_sizes = self._get_fused_linear_int4_qkv_pack(
                        module,
                        (module.q_proj, module.k_proj, module.v_proj),
                        self._model.dtype,
                        "_aicas_decode_qkv_linear_pack",
                        "_aicas_decode_qkv_linear_int4_pack",
                        activation_samples=samples,
                    )
                    layers_payload[str(actual_layer_idx)] = {
                        "weight_int4": weight_int4.contiguous(),
                        "scales": scales.contiguous(),
                        "bias": None if fused_b is None else fused_b.contiguous(),
                        "split_sizes": tuple(int(x) for x in split_sizes),
                    }
                payload = {
                    "format": "qkv_int4_sym_block128_gptq_kblock_v1",
                    "source": "warmup-online",
                    "damp_percent": self._env_float("AICAS_DECODE_INT4_GPTQ_DAMP_PERCENT"),
                    "num_samples": int(qkv_target),
                    "num_layers": len(layers_payload),
                    "layers": layers_payload,
                }
                output_path = self._repo_relative_path(os.environ["AICAS_DECODE_INT4_QKV_GPTQ_ONLINE_OUTPUT_PATH"])
                if output_path:
                    disk_layers = {}
                    for key, layer_payload in layers_payload.items():
                        disk_layers[key] = {
                            "weight_int4": layer_payload["weight_int4"].detach().cpu().contiguous(),
                            "scales": layer_payload["scales"].detach().cpu().contiguous(),
                            "bias": None
                            if layer_payload["bias"] is None
                            else layer_payload["bias"].detach().cpu().contiguous(),
                            "split_sizes": tuple(int(x) for x in layer_payload["split_sizes"]),
                        }
                    disk_payload = dict(payload)
                    disk_payload["layers"] = disk_layers
                    output_dir = os.path.dirname(output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    torch.save(disk_payload, output_path)
                    payload = torch.load(output_path, map_location=input_ids.device)
                self._install_qkv_gptq_online_payload(payload)
                print(
                    f"[VLMModel] Online qkv GPTQ pack generated: "
                    f"layers={len(layers_payload)}, samples={int(qkv_target)}"
                )

            if need_gate_up:
                from kernels.fused_decode_gate_up_swiglu import quantize_gate_up_int4_sym_block128_gptq_pack

                layers_payload = {}
                for idx, module in enumerate(gate_up_modules):
                    actual_layer_idx = int(getattr(module, "_aicas_layer_idx", idx))
                    rows = self._decode_int4_gate_up_gptq_online_rows.get(actual_layer_idx, [])
                    if not rows:
                        raise RuntimeError(
                            f"missing online gate/up GPTQ activation samples for layer {actual_layer_idx}"
                        )
                    samples = torch.cat(rows, dim=0).contiguous()
                    if int(samples.shape[0]) < int(gate_up_target):
                        raise RuntimeError(
                            f"online gate/up GPTQ collected {int(samples.shape[0])} samples for layer "
                            f"{actual_layer_idx}, expected {int(gate_up_target)}"
                        )
                    samples = samples[: int(gate_up_target)].to(
                        device=module.gate_proj.weight.device,
                        dtype=torch.float16,
                    ).contiguous()
                    fused_w, _fused_b, _split_sizes = self._get_fused_linear_pack(
                        module,
                        (module.gate_proj, module.up_proj),
                        self._model.dtype,
                        "_aicas_decode_gate_up_linear_pack",
                    )
                    weight_int4, scales = quantize_gate_up_int4_sym_block128_gptq_pack(
                        fused_w,
                        samples,
                        damp_percent=self._env_float("AICAS_DECODE_INT4_GATE_UP_GPTQ_DAMP_PERCENT"),
                    )
                    layers_payload[str(actual_layer_idx)] = {
                        "weight_int4": weight_int4.contiguous(),
                        "scales": scales.contiguous(),
                    }
                payload = {
                    "format": "gate_up_int4_sym_block128_gptq_rowmajor_v1",
                    "source": "warmup-online",
                    "damp_percent": self._env_float("AICAS_DECODE_INT4_GATE_UP_GPTQ_DAMP_PERCENT"),
                    "num_samples": int(gate_up_target),
                    "num_layers": len(layers_payload),
                    "layers": layers_payload,
                }
                output_path = self._repo_relative_path(
                    os.environ["AICAS_DECODE_INT4_GATE_UP_GPTQ_ONLINE_OUTPUT_PATH"]
                )
                if output_path:
                    disk_layers = {}
                    for key, layer_payload in layers_payload.items():
                        disk_layers[key] = {
                            "weight_int4": layer_payload["weight_int4"].detach().cpu().contiguous(),
                            "scales": layer_payload["scales"].detach().cpu().contiguous(),
                        }
                    disk_payload = dict(payload)
                    disk_payload["layers"] = disk_layers
                    output_dir = os.path.dirname(output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    torch.save(disk_payload, output_path)
                    payload = torch.load(output_path, map_location=input_ids.device)
                self._install_gate_up_gptq_online_payload(payload)
                print(
                    f"[VLMModel] Online gate/up GPTQ pack generated: "
                    f"layers={len(layers_payload)}, samples={int(gate_up_target)}"
                )

            if need_down:
                layers_payload = {}
                for idx, layer in enumerate(down_layers):
                    actual_layer_idx = int(getattr(layer, "_aicas_layer_idx", getattr(layer.mlp, "_aicas_layer_idx", idx)))
                    rows = self._decode_int4_down_gptq_online_rows.get(actual_layer_idx, [])
                    if not rows:
                        raise RuntimeError(f"missing online down GPTQ activation samples for layer {actual_layer_idx}")
                    samples = torch.cat(rows, dim=0).contiguous()
                    if int(samples.shape[0]) < int(down_target):
                        raise RuntimeError(
                            f"online down GPTQ collected {int(samples.shape[0])} samples for layer "
                            f"{actual_layer_idx}, expected {int(down_target)}"
                        )
                    samples = samples[: int(down_target)].to(
                        device=layer.mlp.down_proj.weight.device,
                        dtype=torch.float16,
                    ).contiguous()
                    weight_int4, scales = self._get_int4_weight_pack(
                        layer.mlp.down_proj,
                        layer.mlp.down_proj.weight,
                        "_aicas_decode_down_proj_int4_pack",
                        activation_samples=samples,
                    )
                    layers_payload[str(actual_layer_idx)] = {
                        "weight_int4": weight_int4.contiguous(),
                        "scales": scales.contiguous(),
                    }
                payload = {
                    "format": "down_proj_int4_sym_block128_gptq_kblock_v1",
                    "source": "warmup-online",
                    "damp_percent": self._env_float("AICAS_DECODE_INT4_GPTQ_DAMP_PERCENT"),
                    "num_samples": int(down_target),
                    "num_layers": len(layers_payload),
                    "layers": layers_payload,
                }
                output_path = self._repo_relative_path(
                    os.environ["AICAS_DECODE_INT4_DOWN_ADD_RMSNORM_GPTQ_ONLINE_OUTPUT_PATH"]
                )
                if output_path:
                    disk_layers = {}
                    for key, layer_payload in layers_payload.items():
                        disk_layers[key] = {
                            "weight_int4": layer_payload["weight_int4"].detach().cpu().contiguous(),
                            "scales": layer_payload["scales"].detach().cpu().contiguous(),
                        }
                    disk_payload = dict(payload)
                    disk_payload["layers"] = disk_layers
                    output_dir = os.path.dirname(output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    torch.save(disk_payload, output_path)
                    payload = torch.load(output_path, map_location=input_ids.device)
                self._install_down_gptq_online_payload(payload)
                print(
                    f"[VLMModel] Online down GPTQ pack generated: "
                    f"layers={len(layers_payload)}, samples={int(down_target)}"
                )

            if need_o:
                layers_payload = {}
                for idx, layer in enumerate(o_layers):
                    actual_layer_idx = int(getattr(layer, "_aicas_layer_idx", getattr(layer.self_attn, "layer_idx", idx)))
                    rows = self._decode_int4_o_gptq_online_rows.get(actual_layer_idx, [])
                    if not rows:
                        raise RuntimeError(f"missing online O GPTQ activation samples for layer {actual_layer_idx}")
                    samples = torch.cat(rows, dim=0).contiguous()
                    if int(samples.shape[0]) < int(o_target):
                        raise RuntimeError(
                            f"online O GPTQ collected {int(samples.shape[0])} samples for layer "
                            f"{actual_layer_idx}, expected {int(o_target)}"
                        )
                    samples = samples[: int(o_target)].to(
                        device=layer.self_attn.o_proj.weight.device,
                        dtype=torch.float16,
                    ).contiguous()
                    weight_int4, scales = self._get_int4_weight_pack(
                        layer.self_attn.o_proj,
                        layer.self_attn.o_proj.weight,
                        "_aicas_decode_o_proj_int4_pack",
                        activation_samples=samples,
                    )
                    layers_payload[str(actual_layer_idx)] = {
                        "weight_int4": weight_int4.contiguous(),
                        "scales": scales.contiguous(),
                    }
                payload = {
                    "format": "o_proj_int4_sym_block128_gptq_kblock_v1",
                    "source": "warmup-online",
                    "damp_percent": self._env_float("AICAS_DECODE_INT4_GPTQ_DAMP_PERCENT"),
                    "num_samples": int(o_target),
                    "num_layers": len(layers_payload),
                    "layers": layers_payload,
                }
                output_path = self._repo_relative_path(
                    os.environ["AICAS_DECODE_INT4_O_ADD_RMSNORM_GPTQ_ONLINE_OUTPUT_PATH"]
                )
                if output_path:
                    disk_layers = {}
                    for key, layer_payload in layers_payload.items():
                        disk_layers[key] = {
                            "weight_int4": layer_payload["weight_int4"].detach().cpu().contiguous(),
                            "scales": layer_payload["scales"].detach().cpu().contiguous(),
                        }
                    disk_payload = dict(payload)
                    disk_payload["layers"] = disk_layers
                    output_dir = os.path.dirname(output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    torch.save(disk_payload, output_path)
                    payload = torch.load(output_path, map_location=input_ids.device)
                self._install_o_gptq_online_payload(payload)
                print(
                    f"[VLMModel] Online O GPTQ pack generated: "
                    f"layers={len(layers_payload)}, samples={int(o_target)}"
                )

            if need_lm_head:
                rows = self._decode_int4_lm_head_gptq_online_rows
                if not rows:
                    raise RuntimeError("missing online lm_head GPTQ activation samples")
                samples = torch.cat(rows, dim=0).contiguous()
                if int(samples.shape[0]) < int(lm_head_target):
                    raise RuntimeError(
                        f"online lm_head GPTQ collected {int(samples.shape[0])} samples, "
                        f"expected {int(lm_head_target)}"
                    )
                samples = samples[: int(lm_head_target)].to(
                    device=lm_head.weight.device,
                    dtype=torch.float16,
                ).contiguous()
                token_ids = self._lm_head_shortlist_token_ids
                if token_ids is None:
                    token_ids = torch.arange(int(lm_head.weight.shape[0]), device=lm_head.weight.device, dtype=torch.long)
                    self._lm_head_shortlist_token_ids = token_ids
                pack_weight = torch.index_select(lm_head.weight, 0, token_ids).contiguous()
                weight_int4, scales = self._get_lm_head_shortlist_int4_pack(
                    lm_head,
                    pack_weight,
                    "_aicas_lm_head_shortlist_int4_pack",
                    activation_samples=samples,
                )
                payload = {
                    "format": "lm_head_int4_sym_block128_gptq_kblock_v1",
                    "source": "warmup-online",
                    "damp_percent": self._env_float("AICAS_DECODE_INT4_GPTQ_DAMP_PERCENT"),
                    "num_samples": int(lm_head_target),
                    "num_tokens": int(token_ids.numel()),
                    "weight_int4": weight_int4.contiguous(),
                    "scales": scales.contiguous(),
                }
                output_path = self._repo_relative_path(
                    os.environ["AICAS_DECODE_INT4_LM_HEAD_GPTQ_ONLINE_OUTPUT_PATH"]
                )
                if output_path:
                    disk_payload = dict(payload)
                    disk_payload["weight_int4"] = payload["weight_int4"].detach().cpu().contiguous()
                    disk_payload["scales"] = payload["scales"].detach().cpu().contiguous()
                    output_dir = os.path.dirname(output_path)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    torch.save(disk_payload, output_path)
                    payload = torch.load(output_path, map_location=input_ids.device)
                self._install_lm_head_gptq_online_payload(payload)
                print(
                    f"[VLMModel] Online lm_head GPTQ pack generated: "
                    f"tokens={int(token_ids.numel())}, samples={int(lm_head_target)}"
                )
        finally:
            self._decode_int4_lm_head_enabled = previous_lm_head_int4_enabled
            if need_gate_up and not self._decode_int4_gate_up_gptq_online_ready:
                for entry in previous_layer_flags:
                    module = entry[0]
                    module._aicas_use_int4_gate_up_swiglu = entry[1]
                    if len(entry) > 2 and entry[2] is not None and hasattr(module, "mlp"):
                        module.mlp._aicas_use_int4_gate_up_swiglu = entry[2]
            if need_qkv:
                self._decode_int4_qkv_gptq_online_building = previous_qkv_building
                self._decode_int4_qkv_gptq_online_rows = None
                self._decode_int4_qkv_gptq_online_counts = None
            if need_gate_up:
                self._decode_int4_gate_up_gptq_online_building = previous_gate_up_building
                self._decode_int4_gate_up_gptq_online_rows = None
                self._decode_int4_gate_up_gptq_online_counts = None
            if need_down:
                self._decode_int4_down_gptq_online_building = previous_down_building
                self._decode_int4_down_gptq_online_rows = None
                self._decode_int4_down_gptq_online_counts = None
            if need_o:
                self._decode_int4_o_gptq_online_building = previous_o_building
                self._decode_int4_o_gptq_online_rows = None
                self._decode_int4_o_gptq_online_counts = None
            if need_lm_head:
                self._decode_int4_lm_head_gptq_online_building = previous_lm_head_building
                self._decode_int4_lm_head_gptq_online_rows = None
                self._decode_int4_lm_head_gptq_online_count = 0

    def _get_fused_linear_int4_gate_up_pack(
        self,
        owner_module,
        linear_modules,
        out_dtype: torch.dtype,
        cache_attr: str,
        int4_cache_attr: str,
    ):
        cache = getattr(owner_module, int4_cache_attr, None)
        if cache is not None:
            return cache
        fused_w, fused_b, split_sizes = self._get_fused_linear_pack(
            owner_module,
            linear_modules,
            out_dtype,
            cache_attr,
        )
        weight_int4, scales = self._get_gate_up_int4_gptq_rowmajor_pack(owner_module, fused_w)
        weight_int4, scales = self._get_gate_up_int4_kblock_kscale_pack(
            owner_module,
            weight_int4,
            scales,
            split_sizes,
        )
        pack = (weight_int4, scales, fused_b, split_sizes)
        setattr(owner_module, int4_cache_attr, pack)
        return pack

    def _get_fused_linear_int4_qkv_pack(
        self,
        owner_module,
        linear_modules,
        out_dtype: torch.dtype,
        cache_attr: str,
        int4_cache_attr: str,
        *,
        activation_samples: torch.Tensor | None = None,
    ):
        cache = getattr(owner_module, int4_cache_attr, None)
        if cache is None:
            cache = getattr(owner_module, f"{int4_cache_attr}_gptq", None)
        if cache is not None:
            return cache
        fused_w, fused_b, split_sizes = self._get_fused_linear_pack(
            owner_module,
            linear_modules,
            out_dtype,
            cache_attr,
        )
        if activation_samples is None:
            raise RuntimeError("GPTQ qkv INT4 pack requires activation_samples")
        from kernels.fused_decode_qkv_gemv_fp8 import quantize_int4_sym_block128_gptq_kblock_pack

        weight_int4, scales = quantize_int4_sym_block128_gptq_kblock_pack(
            fused_w,
            activation_samples.reshape(-1, int(fused_w.shape[1])).contiguous(),
            damp_percent=self._env_float("AICAS_DECODE_INT4_GPTQ_DAMP_PERCENT"),
        )
        pack = (weight_int4, scales, fused_b, split_sizes)
        setattr(owner_module, int4_cache_attr, pack)
        return pack

    def _get_fused_linear_fp8_pack(
        self,
        owner_module,
        linear_modules,
        out_dtype: torch.dtype,
        cache_attr: str,
        fp8_cache_attr: str,
        *,
        kblock_major: bool = False,
        kscale: bool = False,
    ):
        effective_cache_attr = fp8_cache_attr + ("_kblock_kscale" if kscale else "_kblock" if kblock_major else "")
        cache = getattr(owner_module, effective_cache_attr, None)
        if cache is not None:
            return cache
        fused_w, fused_b, split_sizes = self._get_fused_linear_pack(
            owner_module,
            linear_modules,
            out_dtype,
            cache_attr,
        )
        weight_fp8, scales = self._get_fp8_weight_pack(owner_module, fused_w, fp8_cache_attr + "_weight")
        if kscale:
            weight_fp8, scales = self._get_gate_up_fp8_kblock_kscale_pack(owner_module, weight_fp8, scales, split_sizes)
        elif kblock_major:
            weight_fp8, scales = self._get_gate_up_fp8_kblock_pack(owner_module, weight_fp8, scales, split_sizes)
        pack = (weight_fp8, scales, fused_b, split_sizes)
        setattr(owner_module, effective_cache_attr, pack)
        return pack

    def _enable_triton_lm_head_top1(self):
        """Use an exact full-vocab Triton top-1 kernel for decode lm_head."""
        if not self._in_repo_kernels_enabled:
            return

        from kernels.fused_lm_head_top1 import (
            create_lm_head_top1_workspace,
            triton_lm_head_shortlist_top1_fp8,
            triton_lm_head_top1_fp8,
        )
        if self._cuda_lm_head_top1_fp8_enabled:
            from kernels.cuda.lm_head_top1_fp8 import (
                cuda_lm_head_shortlist_compact_top1_fp8,
                cuda_lm_head_top1_fp8,
            )
        if self._decode_int4_lm_head_enabled:
            from kernels.cuda.lm_head_top1_int4 import cuda_lm_head_shortlist_compact_top1_int4
        lm_head = getattr(self._model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            return
        weight = getattr(lm_head, "weight", None)
        if not torch.is_tensor(weight) or weight.dtype != torch.float16 or weight.device.type != "cuda":
            return

        self._triton_lm_head_top1_enabled = True
        if self._decode_fp8_lm_head_enabled:
            if self._cuda_lm_head_top1_fp8_enabled:
                self._triton_lm_head_top1_fp8 = cuda_lm_head_top1_fp8
                self._triton_lm_head_shortlist_top1_fp8 = cuda_lm_head_shortlist_compact_top1_fp8
            else:
                self._triton_lm_head_top1_fp8 = triton_lm_head_top1_fp8
                self._triton_lm_head_shortlist_top1_fp8 = triton_lm_head_shortlist_top1_fp8
        if self._decode_int4_lm_head_enabled:
            self._lm_head_shortlist_top1_int4 = cuda_lm_head_shortlist_compact_top1_int4
        self._create_lm_head_top1_workspace = create_lm_head_top1_workspace
        top1_prefix = "AICAS_TRITON_LM_HEAD_TOP1_FP8" if self._decode_fp8_lm_head_enabled else "AICAS_TRITON_LM_HEAD_TOP1"
        shortlist_prefix = (
            "AICAS_TRITON_LM_HEAD_SHORTLIST_FP8"
            if self._decode_fp8_lm_head_enabled
            else "AICAS_TRITON_LM_HEAD_SHORTLIST"
        )
        self._lm_head_top1_block_n = self._env_int(f"{top1_prefix}_BLOCK_N")
        self._lm_head_top1_block_k = self._env_int(f"{top1_prefix}_BLOCK_K")
        self._lm_head_top1_reduce_block = self._env_int(f"{top1_prefix}_REDUCE_BLOCK")
        self._lm_head_top1_num_warps = self._env_int(f"{top1_prefix}_NUM_WARPS")
        self._lm_head_top1_num_stages = self._env_int(f"{top1_prefix}_NUM_STAGES")
        self._lm_head_top1_reduce_num_warps = self._env_int(f"{top1_prefix}_REDUCE_NUM_WARPS")
        self._lm_head_shortlist_block_n = self._env_int(f"{shortlist_prefix}_BLOCK_N")
        self._lm_head_shortlist_block_k = self._env_int(f"{shortlist_prefix}_BLOCK_K")
        self._lm_head_shortlist_reduce_block = self._env_int(f"{shortlist_prefix}_REDUCE_BLOCK")
        self._lm_head_shortlist_num_warps = self._env_int(f"{shortlist_prefix}_NUM_WARPS")
        self._lm_head_shortlist_num_stages = self._env_int(f"{shortlist_prefix}_NUM_STAGES")
        self._lm_head_shortlist_reduce_num_warps = self._env_int(f"{shortlist_prefix}_REDUCE_NUM_WARPS")
        self._load_lm_head_shortlist(weight)
        if self._decode_fp8_lm_head_enabled:
            if self._lm_head_shortlist_token_ids is not None:
                shortlist_weight = torch.index_select(weight, 0, self._lm_head_shortlist_token_ids).contiguous()
                self._lm_head_shortlist_weight_fp8, self._lm_head_shortlist_scales_fp8 = self._get_fp8_weight_pack(
                    lm_head,
                    shortlist_weight,
                    "_aicas_lm_head_shortlist_fp8_pack",
                )
            else:
                self._lm_head_weight_fp8, self._lm_head_scales_fp8 = self._get_fp8_weight_pack(
                    lm_head,
                    weight,
                    "_aicas_lm_head_fp8_pack",
                )
        if self._decode_int4_lm_head_enabled:
            if self._lm_head_shortlist_token_ids is None:
                self._lm_head_shortlist_token_ids = torch.arange(
                    int(weight.shape[0]),
                    device=weight.device,
                    dtype=torch.long,
                )
        if self._decode_fp8_lm_head_enabled:
            self._mark_optimization(
                "cuda_lm_head_top1_fp8" if self._cuda_lm_head_top1_fp8_enabled else "triton_lm_head_top1_fp8"
            )
        if self._lm_head_shortlist_token_ids is not None:
            self._mark_optimization("lm_head_shortlist_top1")
            if self._decode_fp8_lm_head_enabled:
                self._mark_optimization(
                    "cuda_lm_head_shortlist_top1_fp8"
                    if self._cuda_lm_head_top1_fp8_enabled
                    else "lm_head_shortlist_top1_fp8"
                )
            if self._decode_int4_lm_head_enabled:
                if int(self._lm_head_shortlist_token_ids.numel()) == int(weight.shape[0]):
                    self._mark_optimization("cuda_lm_head_full_top1_int4")
                else:
                    self._mark_optimization("cuda_lm_head_shortlist_top1_int4")

    def _lm_head_top1_launch_params(self, use_shortlist: bool):
        if use_shortlist:
            return (
                self._lm_head_shortlist_block_n,
                self._lm_head_shortlist_block_k,
                self._lm_head_shortlist_reduce_block,
                self._lm_head_shortlist_num_warps,
                self._lm_head_shortlist_num_stages,
                self._lm_head_shortlist_reduce_num_warps,
            )
        return (
            self._lm_head_top1_block_n,
            self._lm_head_top1_block_k,
            self._lm_head_top1_reduce_block,
            self._lm_head_top1_num_warps,
            self._lm_head_top1_num_stages,
            self._lm_head_top1_reduce_num_warps,
        )

    def _load_lm_head_shortlist(self, weight: torch.Tensor) -> None:
        if not self._env_flag("AICAS_ENABLE_LM_HEAD_SHORTLIST"):
            return
        path = os.environ.get("AICAS_LM_HEAD_SHORTLIST_PATH", "").strip()
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        token_ids = None
        if isinstance(payload, dict):
            if "token_ids" in payload:
                token_ids = payload["token_ids"]
            elif "shortlist" in payload:
                token_ids = payload["shortlist"]
        elif isinstance(payload, list):
            token_ids = payload

        if not token_ids:
            raise ValueError(f"failed to load lm_head shortlist tokens from {path}")

        vocab_size = int(weight.shape[0])
        seen = set()
        for token_id in token_ids:
            token_int = int(token_id)
            if token_int < 0 or token_int >= vocab_size:
                raise ValueError(f"lm_head shortlist token id out of range: {token_int}")
            seen.add(token_int)
        unique = sorted(seen)
        if not unique:
            raise ValueError(f"lm_head shortlist is empty after de-duplication: {path}")

        self._lm_head_shortlist_token_ids = torch.tensor(
            unique,
            device=weight.device,
            dtype=torch.long,
        )
        print(f"[VLMModel] LM head shortlist loaded: {len(unique)} token(s) from {path}")

    def _lm_head_weight_bias(self):
        lm_head = getattr(self._model, "lm_head", None)
        if lm_head is None:
            return None, None
        weight = getattr(lm_head, "weight", None)
        if not torch.is_tensor(weight):
            return None, None
        bias = getattr(lm_head, "bias", None)
        if bias is not None and not torch.is_tensor(bias):
            bias = None
        return weight, bias

    def _prepare_lm_head_top1_state(self, state):
        if not self._triton_lm_head_top1_enabled:
            return
        weight, _ = self._lm_head_weight_bias()
        if weight is None:
            return
        token_ids = self._lm_head_shortlist_token_ids
        use_compact = token_ids is not None and (
            self._decode_int4_lm_head_enabled or int(token_ids.numel()) != int(weight.shape[0])
        )
        n_vocab = int(token_ids.numel()) if use_compact else int(weight.shape[0])
        block_n, _, reduce_block, _, _, _ = self._lm_head_top1_launch_params(use_compact)
        workspace_key = (n_vocab, block_n, reduce_block, str(weight.device), use_compact)
        if state.get("lm_head_top1_workspace_key") == workspace_key:
            return
        state["lm_head_top1_workspace"] = self._create_lm_head_top1_workspace(
            device=weight.device,
            n_vocab=n_vocab,
            block_n=block_n,
            reduce_block=reduce_block,
        )
        state["lm_head_top1_workspace_key"] = workspace_key

    def _run_lm_head_top1_on_hidden(
        self,
        hidden_states: torch.Tensor,
        state,
        *,
        out_value: torch.Tensor,
        out_token: torch.Tensor,
        token_index: torch.Tensor | int | None = None,
    ) -> None:
        self._prepare_lm_head_top1_state(state)
        weight, bias = self._lm_head_weight_bias()
        if weight is None:
            raise RuntimeError("decode lm_head top1 requires a CUDA lm_head weight")

        if token_index is None:
            selected_hidden = hidden_states[:, -1:, :]
        elif torch.is_tensor(token_index):
            selected_hidden = torch.index_select(hidden_states, 1, token_index.reshape(-1))
        else:
            idx = int(token_index)
            selected_hidden = hidden_states[:, idx : idx + 1, :]

        self._maybe_collect_lm_head_gptq_online_activation(selected_hidden)

        token_ids = self._lm_head_shortlist_token_ids
        use_compact = token_ids is not None and (
            self._decode_int4_lm_head_enabled or int(token_ids.numel()) != int(weight.shape[0])
        )

        if use_compact:
            if self._decode_int4_lm_head_enabled:
                if self._lm_head_shortlist_weight_int4 is None or self._lm_head_shortlist_scales_int4 is None:
                    raise RuntimeError("lm_head GPTQ INT4 pack is not ready")
                if (
                    self._lm_head_shortlist_top1_int4 is None
                    or self._lm_head_shortlist_weight_int4 is None
                    or self._lm_head_shortlist_scales_int4 is None
                ):
                    raise RuntimeError("AICAS_ENABLE_DECODE_INT4_LM_HEAD requires shortlist INT4 top1 kernel")
            elif self._decode_fp8_lm_head_enabled:
                if (
                    self._triton_lm_head_shortlist_top1_fp8 is None
                    or self._lm_head_shortlist_weight_fp8 is None
                    or self._lm_head_shortlist_scales_fp8 is None
                ):
                    raise RuntimeError("AICAS_ENABLE_DECODE_FP8_LM_HEAD requires shortlist FP8 top1 kernel")
            block_n, block_k, reduce_block, num_warps, num_stages, reduce_num_warps = (
                self._lm_head_top1_launch_params(True)
            )
            if self._decode_int4_lm_head_enabled:
                self._lm_head_shortlist_top1_int4(
                    selected_hidden,
                    self._lm_head_shortlist_weight_int4,
                    self._lm_head_shortlist_scales_int4,
                    token_ids,
                    bias,
                    out_value=out_value,
                    out_token=out_token,
                    workspace=state.get("lm_head_top1_workspace"),
                    block_n=block_n,
                    block_k=block_k,
                    reduce_block=reduce_block,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    reduce_num_warps=reduce_num_warps,
                )
            elif self._decode_fp8_lm_head_enabled:
                self._triton_lm_head_shortlist_top1_fp8(
                    selected_hidden,
                    self._lm_head_shortlist_weight_fp8,
                    self._lm_head_shortlist_scales_fp8,
                    token_ids,
                    bias,
                    out_value=out_value,
                    out_token=out_token,
                    workspace=state.get("lm_head_top1_workspace"),
                    block_n=block_n,
                    block_k=block_k,
                    reduce_block=reduce_block,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    reduce_num_warps=reduce_num_warps,
                )
            else:
                raise RuntimeError("decode lm_head shortlist requires FP8 or GPTQ INT4")
        else:
            if self._decode_int4_lm_head_enabled:
                raise RuntimeError("AICAS_ENABLE_DECODE_INT4_LM_HEAD requires full-vocab compact token ids")
            if self._decode_fp8_lm_head_enabled:
                if (
                    self._triton_lm_head_top1_fp8 is None
                    or self._lm_head_weight_fp8 is None
                    or self._lm_head_scales_fp8 is None
                ):
                    raise RuntimeError("AICAS_ENABLE_DECODE_FP8_LM_HEAD requires FP8 top1 kernel")
            block_n, block_k, reduce_block, num_warps, num_stages, reduce_num_warps = (
                self._lm_head_top1_launch_params(False)
            )
            if self._decode_fp8_lm_head_enabled:
                self._triton_lm_head_top1_fp8(
                    selected_hidden,
                    self._lm_head_weight_fp8,
                    self._lm_head_scales_fp8,
                    bias,
                    out_value=out_value,
                    out_token=out_token,
                    workspace=state.get("lm_head_top1_workspace"),
                    block_n=block_n,
                    block_k=block_k,
                    reduce_block=reduce_block,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    reduce_num_warps=reduce_num_warps,
                )
            else:
                raise RuntimeError("decode lm_head full-vocab top1 requires FP8")

    def _run_decode_model_and_select_token(self, model_kwargs: Dict[str, torch.Tensor], state):
        if self._triton_lm_head_top1_enabled:
            outputs = self._model.model(
                input_ids=model_kwargs.get("input_ids"),
                pixel_values=model_kwargs.get("pixel_values"),
                pixel_values_videos=model_kwargs.get("pixel_values_videos"),
                image_grid_thw=model_kwargs.get("image_grid_thw"),
                video_grid_thw=model_kwargs.get("video_grid_thw"),
                position_ids=model_kwargs.get("position_ids"),
                attention_mask=model_kwargs.get("attention_mask"),
                past_key_values=model_kwargs.get("past_key_values"),
                inputs_embeds=model_kwargs.get("inputs_embeds"),
                cache_position=model_kwargs.get("cache_position"),
                use_cache=model_kwargs.get("use_cache"),
            )
            hidden_states = outputs[0]
            self._run_lm_head_top1_on_hidden(
                hidden_states,
                state,
                out_value=state["static_argmax_value"],
                out_token=state["static_decode_token"],
            )
            return outputs

        outputs = self._model.forward(**model_kwargs)
        torch.max(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
            out=(state['static_argmax_value'], state['static_decode_token']),
        )
        return outputs

    def _advance_decode_positions(self, state):
        advance_fn = self._fused_decode_position_advance
        if advance_fn is not None:
            advance_fn(state['static_cache_position'], state['static_position_ids'])
            if self._decode_static_attention_mask_enabled:
                self._refresh_decode_attention_mask(state)
            return
        state['static_cache_position'].add_(1)
        state['static_position_ids'].add_(1)
        if self._decode_static_attention_mask_enabled:
            self._refresh_decode_attention_mask(state)

    def _refresh_decode_attention_mask(self, state):
        valid_mask = state['static_decode_valid_index'] <= state['static_cache_position'].view(1, 1, 1, 1)
        attention_mask = state['static_decode_attention_mask']
        attention_mask.masked_fill_(valid_mask, 0.0)
        attention_mask.masked_fill_(~valid_mask, torch.finfo(attention_mask.dtype).min)

    def _decode_attention_mask(self, state):
        if not self._decode_static_attention_mask_enabled:
            return None
        return state['static_decode_attention_mask']

    def _clear_final_norm_cache(self, owner):
        if owner is None:
            return
        owner._aicas_final_norm_cache = None
        owner._aicas_final_norm_input_ptr = None
        owner._aicas_final_norm_input_shape = None

    def _store_final_norm_cache(self, owner, hidden_states: torch.Tensor, final_normed: torch.Tensor):
        if owner is None:
            return
        owner._aicas_final_norm_cache = final_normed
        owner._aicas_final_norm_input_ptr = hidden_states.data_ptr()
        owner._aicas_final_norm_input_shape = tuple(hidden_states.shape)

    def _consume_final_norm_cache(self, owner, hidden_states: torch.Tensor):
        if owner is None:
            return None
        cached_norm = getattr(owner, "_aicas_final_norm_cache", None)
        cached_ptr = getattr(owner, "_aicas_final_norm_input_ptr", None)
        cached_shape = getattr(owner, "_aicas_final_norm_input_shape", None)
        if (
            cached_norm is not None
            and cached_ptr == hidden_states.data_ptr()
            and cached_shape == tuple(hidden_states.shape)
        ):
            self._clear_final_norm_cache(owner)
            return cached_norm
        self._clear_final_norm_cache(owner)
        return None

    def _enable_cuda_graph_generate(self):
        """Monkey patch generate() to use CUDA Graph replay for greedy decoding."""
        if not self._cuda_graph_enabled:
            return

        self._original_generate = self._model.generate

        def _patched_generate(*args, **kwargs):
            if args or not self._is_cuda_graph_candidate(kwargs):
                return self._original_generate(*args, **kwargs)
            with torch.inference_mode():
                return self._cuda_graph_generate(kwargs)

        self._model.generate = _patched_generate
        self._mark_optimization('cuda_graph_generate')
        if self._ttft_kv_reuse_enabled:
            self._mark_optimization("ttft_kv_reuse")
        if (
            self._env_flag("AICAS_ENABLE_FP16_PREFILL_LAST_LOGIT_ONLY")
        ):
            self._mark_optimization("fp16_prefill_last_logit_only")
        if self._visual_cuda_graph_enabled:
            self._mark_optimization("visual_cuda_graph")
        if self._prefill_cuda_graph_enabled:
            self._mark_optimization("prefill_cuda_graph")

    def _is_cuda_graph_candidate(self, generate_kwargs: Dict) -> bool:
        input_ids = generate_kwargs.get('input_ids')
        if not torch.is_tensor(input_ids):
            return False
        if input_ids.device.type != 'cuda':
            return False
        if int(generate_kwargs.get('max_new_tokens', 0)) <= 0:
            return False
        return True

    def _extract_prefill_kwargs(self, generate_kwargs: Dict) -> Dict[str, torch.Tensor]:
        prefill_kwargs = {}
        for name, value in generate_kwargs.items():
            if name in self._forward_arg_names and torch.is_tensor(value):
                prefill_kwargs[name] = value
        if 'input_ids' not in prefill_kwargs:
            raise RuntimeError('CUDA Graph path requires input_ids in forward kwargs.')
        return prefill_kwargs

    def _build_cuda_graph_key(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        seq_bucket: int,
        decode_bucket: int,
    ) -> Tuple:
        input_ids = prefill_kwargs['input_ids']
        # Isolate CUDA Graph states by both sequence bucket and decode bucket.
        # This prevents 1024-token runs from overwriting the 128-token fast path.
        return (
            str(input_ids.device),
            str(input_ids.dtype),
            int(seq_bucket),
            int(decode_bucket),
            self._prefill_tensor_signature(prefill_kwargs),
        )

    def _prefill_tensor_signature(self, prefill_kwargs: Dict[str, torch.Tensor]) -> Tuple:
        signature = []
        bucketed = {"input_ids", "attention_mask"}
        unsupported = {"position_ids", "cache_position", "inputs_embeds"}
        for name, value in sorted(prefill_kwargs.items()):
            if name in bucketed or name in unsupported or not torch.is_tensor(value):
                continue
            extra = ()
            if name == "image_grid_thw":
                visual = getattr(getattr(self._model, "model", None), "visual", None)
                if visual is not None:
                    grid_meta = self._get_single_image_grid_meta(visual, value)
                    extra = tuple(int(x) for x in grid_meta) if grid_meta is not None else ()
            signature.append((name, str(value.device), str(value.dtype), tuple(int(x) for x in value.shape), extra))
        return tuple(signature)

    def _capture_prefill_identity(self, prefill_kwargs: Dict[str, torch.Tensor]) -> Dict[str, object]:
        tensors = {}
        for name, value in prefill_kwargs.items():
            if torch.is_tensor(value):
                tensors[name] = value
        return {
            "keys": tuple(sorted(tensors.keys())),
            "tensors": tensors,
        }

    def _prefill_identity_matches(self, cached_identity: Dict[str, object], prefill_kwargs: Dict[str, torch.Tensor]) -> bool:
        if cached_identity is None:
            return False
        cached_keys = cached_identity.get("keys")
        cached_tensors = cached_identity.get("tensors")
        if cached_keys is None or cached_tensors is None:
            return False

        current_tensors = {}
        for name, value in prefill_kwargs.items():
            if torch.is_tensor(value):
                current_tensors[name] = value

        if cached_keys != tuple(sorted(current_tensors.keys())):
            return False
        for name, value in current_tensors.items():
            if cached_tensors.get(name) is not value:
                return False
        return True

    def _round_seq_bucket(self, seq_len: int, prefill_kwargs: Dict[str, torch.Tensor] | None = None) -> int:
        seq_len = int(seq_len)
        if self._adaptive_image_prefill_graph_enabled():
            fixed_shape_max_seq_len = self._adaptive_image_prefill_max_seq_len(seq_len, prefill_kwargs)
            if fixed_shape_max_seq_len is not None:
                return int(fixed_shape_max_seq_len)
            adaptive_buckets = tuple(sorted(set(self._env_int_list("AICAS_ADAPTIVE_IMAGE_SEQ_BUCKETS"))))
            for bucket in adaptive_buckets:
                if seq_len <= int(bucket):
                    return int(bucket)
            if adaptive_buckets:
                growth = max(16, int(adaptive_buckets[-1]) - int(adaptive_buckets[-2]) if len(adaptive_buckets) > 1 else 16)
                return ((seq_len + growth - 1) // growth) * growth
            return seq_len
        if self._env_flag("AICAS_ENABLE_PREFILL_TEXT_MAX_BUCKET"):
            fixed_shape_max_seq_len = self._fixed_shape_prefill_max_seq_len(seq_len)
            if fixed_shape_max_seq_len is not None:
                return int(fixed_shape_max_seq_len)
        for bucket in self._prefill_graph_buckets:
            if seq_len <= int(bucket):
                return int(bucket)
        base_bucket = int(self._graph_seq_bucket_base)
        growth_bucket = int(self._graph_seq_bucket_growth)
        if seq_len <= base_bucket:
            return base_bucket
        return ((seq_len + growth_bucket - 1) // growth_bucket) * growth_bucket

    def _fixed_shape_prefill_max_seq_len(self, seq_len: int) -> int | None:
        """Return the fixed-shape prompt upper bound used by CUDA Graph padding."""
        if not self._fixed_vision_runtime_shape_enabled():
            return None
        try:
            image_processor = getattr(self._processor, "image_processor", None)
            fixed_hw = getattr(image_processor, "_aicas_fixed_resize_hw", None)
            if fixed_hw is None:
                return None
            fixed_h, fixed_w = fixed_hw
            patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
            merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
            factor = max(1, patch_size * merge_size)
            image_tokens = (int(fixed_h) // factor) * (int(fixed_w) // factor)
            text_extra_max = self._env_int("AICAS_PREFILL_TEXT_EXTRA_MAX")
        except Exception:
            return None
        if image_tokens <= 0 or text_extra_max <= 0:
            return None
        max_seq_len = int(image_tokens) + int(text_extra_max)
        return max_seq_len if int(seq_len) <= max_seq_len else None

    def _adaptive_image_prefill_max_seq_len(
        self,
        seq_len: int,
        prefill_kwargs: Dict[str, torch.Tensor] | None = None,
    ) -> int | None:
        """Return the bucket-specific prompt upper bound for adaptive image graphs."""
        if not self._adaptive_image_prefill_graph_enabled():
            return None
        try:
            text_extra_max = self._env_int("AICAS_PREFILL_TEXT_EXTRA_MAX")
            if text_extra_max <= 0:
                return None
            if prefill_kwargs is not None:
                image_tokens = self._prefill_image_token_count(prefill_kwargs)
                if image_tokens > 0:
                    max_seq_len = int(image_tokens) + int(text_extra_max)
                    return max_seq_len if int(seq_len) <= max_seq_len else None
            image_processor = getattr(self._processor, "image_processor", None)
            factor = int(getattr(image_processor, "_aicas_fixed_resize_factor", 0) or 0)
            if factor <= 0:
                patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
                merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
                factor = max(1, patch_size * merge_size)
            buckets = []
            for h, w in self._image_size_buckets(factor):
                image_tokens = (int(h) // factor) * (int(w) // factor)
                if image_tokens > 0:
                    buckets.append(int(image_tokens) + int(text_extra_max))
        except Exception:
            return None
        for bucket in sorted(set(buckets)):
            if int(seq_len) <= int(bucket):
                return int(bucket)
        return None

    def _round_decode_bucket(self, max_new_tokens: int) -> int:
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= self._graph_decode_bucket_small:
            return int(self._graph_decode_bucket_small)
        if max_new_tokens <= self._graph_decode_bucket_large:
            return int(self._graph_decode_bucket_large)
        growth_bucket = 512
        return ((max_new_tokens + growth_bucket - 1) // growth_bucket) * growth_bucket

    def _prewarm_decode_capacities(self, max_new_tokens: int) -> Tuple[int, ...]:
        capacities = {self._round_decode_bucket(int(max_new_tokens))}
        try:
            for value in self._env_int_list("AICAS_CUDA_GRAPH_PREWARM_DECODE_BUCKETS"):
                if int(value) > 1:
                    capacities.add(self._round_decode_bucket(int(value)))
        except Exception:
            pass
        return tuple(sorted(capacities))

    def _create_decode_state(
        self,
        device: torch.device,
        input_dtype: torch.dtype,
        max_supported_seq_len: int,
        max_decode_tokens: int,
    ):
        cache = StaticCache(
            self._model.config,
            max_cache_len=int(max_supported_seq_len) + int(max_decode_tokens),
            max_batch_size=1,
            device=str(device),
            dtype=self._model.dtype,
        )
        text_model = getattr(getattr(self._model, "model", None), "language_model", None)
        first_layer = None
        if text_model is not None:
            layers = getattr(text_model, "layers", None)
            if layers:
                first_layer = layers[0]
        first_attn = getattr(first_layer, "self_attn", None) if first_layer is not None else None
        if first_attn is not None and hasattr(cache, "early_initialization"):
            cache.early_initialization(
                batch_size=1,
                num_heads=int(first_attn.config.num_key_value_heads),
                head_dim=int(first_attn.head_dim),
                dtype=self._model.dtype,
                device=device,
            )
        return {
            'cache': cache,
            'max_supported_seq_len': int(max_supported_seq_len),
            'max_decode_tokens': int(max_decode_tokens),
            'graph': None,
            'captured': False,
            'prefill_graph': None,
            'prefill_captured': False,
            'prefill_failed': False,
            'static_prefill_tensors': {},
            'static_prefill_input_ids': torch.empty((1, int(max_supported_seq_len)), device=device, dtype=input_dtype),
            'static_prefill_attention_mask': torch.empty((1, int(max_supported_seq_len)), device=device, dtype=input_dtype),
            'static_prefill_position_ids': torch.empty(
                (3, 1, int(max_supported_seq_len)), device=device, dtype=torch.long
            ),
            'static_prefill_cache_position': torch.arange(
                int(max_supported_seq_len), device=device, dtype=torch.long
            ),
            'static_prefill_logits_index': torch.empty((1,), device=device, dtype=torch.long),
            'static_prefill_rope_deltas': torch.empty((1, 1), device=device, dtype=torch.long),
            'static_decode_token': torch.empty((1, 1), device=device, dtype=input_dtype),
            'static_argmax_value': torch.empty((1, 1), device=device, dtype=self._model.dtype),
            'static_first_token': torch.empty((1, 1), device=device, dtype=input_dtype),
            'static_first_value': torch.empty((1, 1), device=device, dtype=self._model.dtype),
            'static_decode_attention_mask': torch.empty(
                (1, 1, 1, int(max_supported_seq_len) + int(max_decode_tokens)),
                device=device,
                dtype=self._model.dtype,
            ),
            'static_decode_valid_index': torch.arange(
                int(max_supported_seq_len) + int(max_decode_tokens),
                device=device,
                dtype=torch.long,
            ).view(1, 1, 1, -1),
            'static_position_ids': torch.empty((3, 1, 1), device=device, dtype=torch.long),
            'static_cache_position': torch.empty((1,), device=device, dtype=torch.long),
        }

    def _prefill_first_token(self, prefill_kwargs: Dict[str, torch.Tensor], state) -> Tuple[torch.Tensor, int]:
        with self._nvtx_range("prefill/total"):
            return self._prefill_first_token_uncaptured(prefill_kwargs, state)

    def _prefill_first_token_uncaptured(self, prefill_kwargs: Dict[str, torch.Tensor], state) -> Tuple[torch.Tensor, int]:
        state['cache'].reset()

        with self._prefill_active_scope():
            if self._triton_lm_head_top1_enabled:
                outputs = self._model.model(
                    input_ids=prefill_kwargs.get("input_ids"),
                    pixel_values=prefill_kwargs.get("pixel_values"),
                    pixel_values_videos=prefill_kwargs.get("pixel_values_videos"),
                    image_grid_thw=prefill_kwargs.get("image_grid_thw"),
                    video_grid_thw=prefill_kwargs.get("video_grid_thw"),
                    position_ids=prefill_kwargs.get("position_ids"),
                    attention_mask=prefill_kwargs.get("attention_mask"),
                    past_key_values=state["cache"],
                    inputs_embeds=prefill_kwargs.get("inputs_embeds"),
                    cache_position=prefill_kwargs.get("cache_position"),
                    use_cache=True,
                )
            else:
                model_kwargs = {
                    **prefill_kwargs,
                    "past_key_values": state['cache'],
                    "use_cache": True,
                    "return_dict": False,
                }
                if (
                    self._env_flag("AICAS_ENABLE_FP16_PREFILL_LAST_LOGIT_ONLY")
                    and "logits_to_keep" in self._forward_arg_names
                ):
                    model_kwargs["logits_to_keep"] = 1
                outputs = self._model(**model_kwargs)

        if self._triton_lm_head_top1_enabled:
            self._run_lm_head_top1_on_hidden(
                outputs[0],
                state,
                out_value=state["static_first_value"],
                out_token=state["static_first_token"],
            )
        else:
            torch.max(
                outputs[0][:, -1, :],
                dim=-1,
                keepdim=True,
                out=(state['static_first_value'], state['static_first_token']),
            )
        rope_deltas = self._model.model.rope_deltas
        rope_scalar = int(rope_deltas.view(-1)[0].item()) if torch.is_tensor(rope_deltas) else 0
        first_token = state['static_first_token'].clone()
        return first_token, rope_scalar

    def _prefill_pad_token_id(self) -> int:
        tokenizer = getattr(self._processor, "tokenizer", None)
        for owner in (tokenizer, self._model.generation_config, self._model.config):
            if owner is None:
                continue
            pad_id = getattr(owner, "pad_token_id", None)
            if pad_id is not None:
                return int(pad_id)
        eos_id = getattr(self._model.generation_config, "eos_token_id", None)
        if torch.is_tensor(eos_id):
            eos_id = int(eos_id.reshape(-1)[0].item())
        elif hasattr(eos_id, "__iter__"):
            eos_id = next(iter(eos_id), 0)
        return int(eos_id) if eos_id is not None else 0

    def _can_use_prefill_cuda_graph(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        state,
        current_seq_len: int,
    ) -> bool:
        if (
            not self._prefill_cuda_graph_enabled
            or state.get("prefill_failed", False)
            or "logits_to_keep" not in self._forward_arg_names
            or int(current_seq_len) > int(state["max_supported_seq_len"])
        ):
            return False
        input_ids = prefill_kwargs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.device.type != "cuda":
            return False
        if int(input_ids.shape[0]) != 1 or int(input_ids.shape[1]) != int(current_seq_len):
            return False
        if any(name in prefill_kwargs for name in ("position_ids", "cache_position", "inputs_embeds")):
            return False
        if prefill_kwargs.get("pixel_values_videos") is not None or prefill_kwargs.get("video_grid_thw") is not None:
            return False
        return True

    def _static_prefill_tensor(self, state, name: str, value: torch.Tensor) -> torch.Tensor:
        static_tensors = state["static_prefill_tensors"]
        static_value = static_tensors.get(name)
        if (
            static_value is None
            or static_value.shape != value.shape
            or static_value.dtype != value.dtype
            or static_value.device != value.device
        ):
            static_value = torch.empty_like(value)
            static_tensors[name] = static_value
        return static_value

    def _prefill_image_token_count(self, prefill_kwargs: Dict[str, torch.Tensor]) -> int:
        image_grid_thw = prefill_kwargs.get("image_grid_thw")
        if image_grid_thw is None or not torch.is_tensor(image_grid_thw):
            return 0
        qwen = self._model.model
        grid_meta = self._get_single_image_grid_meta(qwen, image_grid_thw)
        if grid_meta is None:
            return 0
        t, h, w = grid_meta
        merge = int(qwen.config.vision_config.spatial_merge_size)
        return int(t) * (int(h) // merge) * (int(w) // merge)

    def _get_or_build_prefill_position_ids(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        bucket: int,
        current_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        qwen = self._model.model
        input_ids = prefill_kwargs["input_ids"]
        image_grid_thw = prefill_kwargs.get("image_grid_thw")
        video_grid_thw = prefill_kwargs.get("video_grid_thw")
        if video_grid_thw is not None:
            raise RuntimeError("competition prefill graph does not support video inputs")
        if image_grid_thw is None or not torch.is_tensor(image_grid_thw) or int(image_grid_thw.shape[0]) != 1:
            raise RuntimeError("competition prefill graph requires exactly one image grid")

        grid_meta = self._get_single_image_grid_meta(qwen, image_grid_thw)
        if grid_meta is None:
            raise RuntimeError("competition prefill graph requires cached single-image grid metadata")
        t, h, w = grid_meta
        merge = int(qwen.config.vision_config.spatial_merge_size)
        llm_h = int(h) // merge
        llm_w = int(w) // merge
        n_image_tokens = int(t) * int(llm_h) * int(llm_w)
        if n_image_tokens <= 0:
            raise RuntimeError("competition prefill graph requires image tokens")

        image_token_id = int(qwen.config.image_token_id)
        start = self._get_single_image_token_span(
            qwen,
            input_ids,
            image_token_id,
            int(current_seq_len),
            int(n_image_tokens),
        )
        tail_len = int(current_seq_len) - int(start) - int(n_image_tokens)
        if tail_len < 0:
            raise RuntimeError("image token span exceeds current sequence length")

        cache_key = (
            str(input_ids.device),
            int(bucket),
            int(current_seq_len),
            int(start),
            int(n_image_tokens),
            int(t),
            int(llm_h),
            int(llm_w),
        )
        cache = getattr(qwen, "_aicas_prefill_padded_rope_index_cache", {})
        cached = cache.get(cache_key)
        if cached is not None:
            position_ids, rope_deltas, rope_scalar = cached
            return position_ids, rope_deltas, int(rope_scalar)

        position_ids = torch.empty(
            (3, 1, int(bucket)),
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids.fill_(1)
        if int(start) > 0:
            prefix = torch.arange(int(start), device=input_ids.device, dtype=torch.long)
            position_ids[:, 0, : int(start)] = prefix.view(1, -1).expand(3, -1)

        t_index = torch.arange(int(t), device=input_ids.device, dtype=torch.long).view(
            -1, 1
        ).expand(-1, int(llm_h) * int(llm_w)).flatten()
        h_index = torch.arange(int(llm_h), device=input_ids.device, dtype=torch.long).view(
            1, -1, 1
        ).expand(int(t), -1, int(llm_w)).flatten()
        w_index = torch.arange(int(llm_w), device=input_ids.device, dtype=torch.long).view(
            1, 1, -1
        ).expand(int(t), int(llm_h), -1).flatten()
        vision = torch.stack((t_index, h_index, w_index), dim=0) + int(start)
        image_end = int(start) + int(n_image_tokens)
        position_ids[:, 0, int(start): image_end] = vision

        tail_start = int(start) + max(int(t) - 1, int(llm_h) - 1, int(llm_w) - 1) + 1
        if tail_len > 0:
            tail = torch.arange(tail_len, device=input_ids.device, dtype=torch.long) + int(tail_start)
            position_ids[:, 0, image_end: int(current_seq_len)] = tail.view(1, -1).expand(3, -1)
            max_position = int(tail_start) + int(tail_len) - 1
        else:
            max_position = int(start) + max(int(t) - 1, int(llm_h) - 1, int(llm_w) - 1)
        rope_scalar = int(max_position) + 1 - int(current_seq_len)
        rope_deltas = torch.tensor(
            [[rope_scalar]],
            dtype=torch.long,
            device=input_ids.device,
        )
        cache[cache_key] = (position_ids, rope_deltas, int(rope_scalar))
        qwen._aicas_prefill_padded_rope_index_cache = cache
        return position_ids, rope_deltas, int(rope_scalar)

    def _prepare_static_prefill_inputs(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        state,
        current_seq_len: int,
    ) -> int:
        bucket = int(state["max_supported_seq_len"])
        input_ids = prefill_kwargs["input_ids"]
        pad_token_id = self._prefill_pad_token_id()

        state["static_prefill_input_ids"].fill_(pad_token_id)
        state["static_prefill_input_ids"][:, : int(current_seq_len)].copy_(input_ids)

        static_attention_mask = state["static_prefill_attention_mask"]
        static_attention_mask.zero_()
        attention_mask = prefill_kwargs.get("attention_mask")
        if attention_mask is not None and torch.is_tensor(attention_mask):
            static_attention_mask[:, : int(current_seq_len)].copy_(attention_mask)
        else:
            static_attention_mask[:, : int(current_seq_len)].fill_(1)

        position_ids, rope_deltas, rope_scalar = self._get_or_build_prefill_position_ids(
            prefill_kwargs,
            bucket,
            int(current_seq_len),
        )
        state["static_prefill_position_ids"].copy_(position_ids)
        state["static_prefill_rope_deltas"].copy_(rope_deltas)
        state["static_prefill_logits_index"].fill_(int(current_seq_len) - 1)

        skipped = {"input_ids", "attention_mask", "position_ids", "cache_position", "inputs_embeds"}
        for name, value in prefill_kwargs.items():
            if name in skipped or not torch.is_tensor(value):
                continue
            self._static_prefill_tensor(state, name, value).copy_(value)

        qwen = self._model.model
        n_image_tokens = self._prefill_image_token_count(prefill_kwargs)
        if n_image_tokens > 0:
            self._get_single_image_token_span(
                qwen,
                state["static_prefill_input_ids"],
                int(qwen.config.image_token_id),
                int(bucket),
                int(n_image_tokens),
            )
        visual = getattr(qwen, "visual", None)
        static_grid = state["static_prefill_tensors"].get("image_grid_thw")
        if visual is not None and static_grid is not None:
            self._get_single_image_grid_meta(visual, static_grid)
            self._get_single_image_grid_meta(qwen, static_grid)

        return int(rope_scalar)

    def _static_prefill_model_kwargs(self, prefill_kwargs: Dict[str, torch.Tensor], state) -> Dict[str, torch.Tensor]:
        model_kwargs = {
            "input_ids": state["static_prefill_input_ids"],
            "attention_mask": state["static_prefill_attention_mask"],
            "position_ids": state["static_prefill_position_ids"],
            "cache_position": state["static_prefill_cache_position"],
            "past_key_values": state["cache"],
            "use_cache": True,
            "return_dict": False,
        }
        if not self._triton_lm_head_top1_enabled:
            model_kwargs["logits_to_keep"] = state["static_prefill_logits_index"]
        skipped = {"input_ids", "attention_mask", "position_ids", "cache_position", "inputs_embeds"}
        for name, value in prefill_kwargs.items():
            if name in skipped or not torch.is_tensor(value):
                continue
            model_kwargs[name] = state["static_prefill_tensors"][name]
        return model_kwargs

    def _run_static_prefill_forward(self, prefill_kwargs: Dict[str, torch.Tensor], state):
        self._model.model.rope_deltas = state["static_prefill_rope_deltas"]
        with self._prefill_active_scope():
            if self._triton_lm_head_top1_enabled:
                outputs = self._model.model(**self._static_prefill_model_kwargs(prefill_kwargs, state))
            else:
                outputs = self._model(**self._static_prefill_model_kwargs(prefill_kwargs, state))
        if self._triton_lm_head_top1_enabled:
            self._run_lm_head_top1_on_hidden(
                outputs[0],
                state,
                out_value=state["static_first_value"],
                out_token=state["static_first_token"],
                token_index=state["static_prefill_logits_index"],
            )
        else:
            torch.max(
                outputs[0][:, -1, :],
                dim=-1,
                keepdim=True,
                out=(state["static_first_value"], state["static_first_token"]),
            )

    def _prefill_first_token_graph(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        state,
        current_seq_len: int,
    ) -> Tuple[torch.Tensor, int]:
        if not self._can_use_prefill_cuda_graph(prefill_kwargs, state, current_seq_len):
            return self._prefill_first_token(prefill_kwargs, state)

        try:
            rope_scalar = self._prepare_static_prefill_inputs(prefill_kwargs, state, current_seq_len)
            direct_visual_prev = self._prefill_cuda_graph_direct_visual
            self._prefill_cuda_graph_direct_visual = True
            try:
                if not state["prefill_captured"]:
                    self._run_static_prefill_forward(prefill_kwargs, state)
                    self._run_static_prefill_forward(prefill_kwargs, state)
                    torch.cuda.synchronize()

                    graph = torch.cuda.CUDAGraph()
                    with self._nvtx_range("prefill/total"):
                        with torch.cuda.graph(graph):
                            self._run_static_prefill_forward(prefill_kwargs, state)
                    state["prefill_graph"] = graph
                    state["prefill_captured"] = True
                    print("[VLMModel] CUDA Graph captured (prefill replay path)")
                else:
                    self._model.model.rope_deltas = state["static_prefill_rope_deltas"]
                    with self._nvtx_range("prefill/total"):
                        state["prefill_graph"].replay()
            finally:
                self._prefill_cuda_graph_direct_visual = direct_visual_prev
            return state["static_first_token"].clone(), int(rope_scalar)
        except Exception as exc:
            state["prefill_failed"] = True
            self._prefill_cuda_graph_disabled_reason = str(exc)
            print(f"[VLMModel] Prefill CUDA Graph disabled for this state: {exc}")
            return self._prefill_first_token(prefill_kwargs, state)

    def _set_decode_start(self, state, token: torch.Tensor, current_seq_len: int, rope_scalar: int, start_step: int):
        state['static_decode_token'].copy_(token)
        position_value = int(current_seq_len) + int(start_step) + rope_scalar
        cache_value = int(current_seq_len) + int(start_step)
        set_start_fn = self._fused_decode_set_start
        if set_start_fn is not None:
            set_start_fn(
                state['static_cache_position'],
                state['static_position_ids'],
                cache_value,
                position_value,
            )
            if self._decode_static_attention_mask_enabled:
                self._refresh_decode_attention_mask(state)
            return
        state['static_position_ids'].fill_(position_value)
        state['static_cache_position'].fill_(cache_value)
        if self._decode_static_attention_mask_enabled:
            self._refresh_decode_attention_mask(state)

    def _capture_decode_graph(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        state,
        rope_scalar: int,
        first_token: torch.Tensor,
        current_seq_len: int,
    ):
        self._set_decode_start(state, first_token, current_seq_len, rope_scalar, 0)

        self._run_decode_model_and_select_token(
            {
                "input_ids": state['static_decode_token'],
                "past_key_values": state['cache'],
                "use_cache": True,
                "return_dict": True,
                "attention_mask": self._decode_attention_mask(state),
                "position_ids": state['static_position_ids'],
                "cache_position": state['static_cache_position'],
                "logits_to_keep": 1,
            },
            state,
        )
        self._advance_decode_positions(state)

        first_token, rope_scalar = self._prefill_first_token(prefill_kwargs, state)
        self._set_decode_start(state, first_token, current_seq_len, rope_scalar, 0)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            self._run_decode_model_and_select_token(
                {
                    "input_ids": state['static_decode_token'],
                    "past_key_values": state['cache'],
                    "use_cache": True,
                    "return_dict": True,
                    "attention_mask": self._decode_attention_mask(state),
                    "position_ids": state['static_position_ids'],
                    "cache_position": state['static_cache_position'],
                    "logits_to_keep": 1,
                },
                state,
            )
            self._advance_decode_positions(state)

        state['graph'] = graph
        state['captured'] = True
        print('[VLMModel] CUDA Graph captured (decode replay path)')

    def _adaptive_image_prewarm_buckets(self, prefill_kwargs: Dict[str, torch.Tensor]) -> Tuple[Tuple[int, int], ...]:
        image_processor = getattr(self._processor, "image_processor", None)
        factor = int(getattr(image_processor, "_aicas_fixed_resize_factor", 0) or 0)
        if factor <= 0:
            patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
            merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
            factor = max(1, patch_size * merge_size)
        buckets = self._image_size_buckets(factor)
        if buckets:
            return buckets

        image_grid_thw = prefill_kwargs.get("image_grid_thw")
        grid_meta = self._get_single_image_grid_meta(self._model.model, image_grid_thw)
        if grid_meta is None:
            return ()
        _, grid_h, grid_w = grid_meta
        patch_size = max(1, factor // 2)
        return ((int(grid_h) * patch_size, int(grid_w) * patch_size),)

    def _make_adaptive_image_prefill_kwargs(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        bucket_h: int,
        bucket_w: int,
    ) -> Tuple[Dict[str, torch.Tensor], int] | None:
        input_ids = prefill_kwargs.get("input_ids")
        attention_mask = prefill_kwargs.get("attention_mask")
        pixel_values = prefill_kwargs.get("pixel_values")
        image_grid_thw = prefill_kwargs.get("image_grid_thw")
        if (
            not torch.is_tensor(input_ids)
            or not torch.is_tensor(attention_mask)
            or not torch.is_tensor(pixel_values)
            or not torch.is_tensor(image_grid_thw)
            or int(input_ids.shape[0]) != 1
            or attention_mask.shape != input_ids.shape
            or int(pixel_values.ndim) != 2
        ):
            return None

        qwen = self._model.model
        image_token_id = int(qwen.config.image_token_id)
        old_image_tokens = self._prefill_image_token_count(prefill_kwargs)
        if old_image_tokens <= 0:
            return None
        current_seq_len = int(input_ids.shape[1])
        start = self._get_single_image_token_span(
            qwen,
            input_ids,
            image_token_id,
            current_seq_len,
            old_image_tokens,
        )

        image_processor = getattr(self._processor, "image_processor", None)
        patch_size = int(getattr(image_processor, "patch_size", 16) or 16)
        merge_size = int(getattr(image_processor, "merge_size", 2) or 2)
        bucket_h = self._round_down_multiple(int(bucket_h), patch_size * merge_size)
        bucket_w = self._round_down_multiple(int(bucket_w), patch_size * merge_size)
        grid_h = int(bucket_h) // int(patch_size)
        grid_w = int(bucket_w) // int(patch_size)
        if grid_h <= 0 or grid_w <= 0 or grid_h % merge_size != 0 or grid_w % merge_size != 0:
            return None
        new_image_tokens = (grid_h // merge_size) * (grid_w // merge_size)
        if new_image_tokens <= 0:
            return None

        prefix = input_ids[:, :start]
        suffix = input_ids[:, int(start) + int(old_image_tokens):]
        image_span = torch.full(
            (1, int(new_image_tokens)),
            int(image_token_id),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        new_input_ids = torch.cat((prefix, image_span, suffix), dim=1).contiguous()
        new_attention_mask = torch.ones_like(new_input_ids, dtype=attention_mask.dtype, device=attention_mask.device)
        new_pixel_values = torch.zeros(
            (int(grid_h) * int(grid_w), int(pixel_values.shape[1])),
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        new_image_grid = torch.tensor(
            [[1, int(grid_h), int(grid_w)]],
            dtype=image_grid_thw.dtype,
            device=image_grid_thw.device,
        )

        warm_kwargs = dict(prefill_kwargs)
        warm_kwargs["input_ids"] = new_input_ids
        warm_kwargs["attention_mask"] = new_attention_mask
        warm_kwargs["pixel_values"] = new_pixel_values
        warm_kwargs["image_grid_thw"] = new_image_grid
        return warm_kwargs, int(new_input_ids.shape[1])

    def _prewarm_adaptive_image_cuda_graphs(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        max_new_tokens: int,
    ) -> bool:
        if (
            self._adaptive_image_graph_prewarm_done
            or not self._env_flag("AICAS_ENABLE_ADAPTIVE_IMAGE_GRAPH_PREWARM")
            or not self._adaptive_image_prefill_graph_enabled()
        ):
            return False

        input_ids = prefill_kwargs.get("input_ids")
        if not torch.is_tensor(input_ids) or input_ids.device.type != "cuda":
            return False

        decode_capacities = self._prewarm_decode_capacities(int(max_new_tokens))
        captured = 0
        failed = 0
        last_ctx = self._last_generation_ctx
        try:
            for bucket_h, bucket_w in self._adaptive_image_prewarm_buckets(prefill_kwargs):
                made = self._make_adaptive_image_prefill_kwargs(prefill_kwargs, int(bucket_h), int(bucket_w))
                if made is None:
                    failed += 1
                    continue
                warm_kwargs, current_seq_len = made
                target_bucket = self._round_seq_bucket(current_seq_len, warm_kwargs)
                if target_bucket < current_seq_len:
                    failed += 1
                    continue
                for decode_capacity in decode_capacities:
                    graph_key = self._build_cuda_graph_key(warm_kwargs, target_bucket, int(decode_capacity))
                    state = self._cuda_graph_cache.get(graph_key)
                    if state is None:
                        state = self._create_decode_state(
                            device=input_ids.device,
                            input_dtype=input_ids.dtype,
                            max_supported_seq_len=target_bucket,
                            max_decode_tokens=int(decode_capacity),
                        )
                        self._cuda_graph_cache[graph_key] = state
                    if not state.get("prefill_captured", False):
                        first_token, rope_scalar = self._prefill_first_token_graph(
                            warm_kwargs,
                            state,
                            int(current_seq_len),
                        )
                        if int(decode_capacity) > 1 and not state.get("captured", False):
                            self._capture_decode_graph(
                                warm_kwargs,
                                state,
                                rope_scalar,
                                first_token,
                                int(current_seq_len),
                            )
                        torch.cuda.synchronize()
                    captured += 1
        except Exception as exc:
            failed += 1
            print(f"[VLMModel] Adaptive image CUDA Graph prewarm skipped: {exc}")
        finally:
            self._last_generation_ctx = last_ctx

        self._adaptive_image_graph_prewarm_done = True
        if captured > 0:
            self._mark_optimization("adaptive_image_cuda_graph_prewarm")
            print(
                f"[VLMModel] Adaptive image CUDA Graph prewarm done: "
                f"{captured} state(s), {failed} failed, decode_capacities={decode_capacities}"
            )
            return True
        return False

    def _maybe_prewarm_prefill_cuda_graphs(
        self,
        prefill_kwargs: Dict[str, torch.Tensor],
        max_new_tokens: int,
    ) -> None:
        if (
            not self._env_flag("AICAS_ENABLE_PREFILL_CUDA_GRAPH_PREWARM")
            or not self._prefill_cuda_graph_enabled
            or not self._cuda_graph_enabled
            or int(max_new_tokens) <= 1
        ):
            return
        if self._prewarm_adaptive_image_cuda_graphs(prefill_kwargs, max_new_tokens):
            return
        if self._prefill_cuda_graph_prewarm_done and not self._adaptive_image_prefill_graph_enabled():
            return
        input_ids = prefill_kwargs.get("input_ids")
        attention_mask = prefill_kwargs.get("attention_mask")
        if (
            not torch.is_tensor(input_ids)
            or input_ids.device.type != "cuda"
            or int(input_ids.shape[0]) != 1
            or not torch.is_tensor(attention_mask)
            or attention_mask.shape != input_ids.shape
        ):
            return
        current_seq_len = int(input_ids.shape[1])
        target_bucket = self._round_seq_bucket(current_seq_len, prefill_kwargs)
        if target_bucket < current_seq_len:
            return

        decode_capacity = self._round_decode_bucket(int(max_new_tokens))
        try:
            warm_kwargs = dict(prefill_kwargs)
            graph_key = self._build_cuda_graph_key(warm_kwargs, target_bucket, decode_capacity)
            state = self._cuda_graph_cache.get(graph_key)
            if state is not None and state.get("prefill_captured", False):
                if not self._adaptive_image_prefill_graph_enabled():
                    self._prefill_cuda_graph_prewarm_done = True
                return
            if state is None:
                state = self._create_decode_state(
                    device=input_ids.device,
                    input_dtype=input_ids.dtype,
                    max_supported_seq_len=target_bucket,
                    max_decode_tokens=decode_capacity,
                )
                self._cuda_graph_cache[graph_key] = state
            if not state.get("prefill_captured", False):
                first_token, rope_scalar = self._prefill_first_token_graph(
                    warm_kwargs,
                    state,
                    int(current_seq_len),
                )
                if int(max_new_tokens) > 1 and not state.get("captured", False):
                    self._capture_decode_graph(warm_kwargs, state, rope_scalar, first_token, int(current_seq_len))
            torch.cuda.synchronize()
            if not self._adaptive_image_prefill_graph_enabled():
                self._prefill_cuda_graph_prewarm_done = True
            self._mark_optimization("prefill_cuda_graph_prewarm")
        except Exception as exc:
            if not self._adaptive_image_prefill_graph_enabled():
                self._prefill_cuda_graph_prewarm_done = True
            print(f"[VLMModel] Prefill CUDA Graph prewarm skipped: {exc}")

    def _get_eos_token_ids(self, generate_kwargs: Dict):
        eos_token_id = generate_kwargs.get('eos_token_id', self._model.generation_config.eos_token_id)
        if eos_token_id is None:
            return set()
        if torch.is_tensor(eos_token_id):
            return {int(x) for x in eos_token_id.reshape(-1).tolist()}
        if hasattr(eos_token_id, "__iter__"):
            return {int(x) for x in eos_token_id}
        return {int(eos_token_id)}

    def _cuda_graph_generate(self, generate_kwargs: Dict):
        max_new_tokens = int(generate_kwargs['max_new_tokens'])
        prefill_kwargs = self._extract_prefill_kwargs(generate_kwargs)
        self._maybe_prepare_all_gptq_online(prefill_kwargs, max_new_tokens)
        self._maybe_prewarm_prefill_cuda_graphs(prefill_kwargs, max_new_tokens)

        input_ids = prefill_kwargs['input_ids']
        current_seq_len = int(input_ids.shape[1])
        seq_bucket = self._round_seq_bucket(current_seq_len, prefill_kwargs)
        decode_capacity = self._round_decode_bucket(max_new_tokens)
        if max_new_tokens == 1 and self._prefill_cuda_graph_enabled and not self._ttft_kv_reuse_enabled:
            decode_capacity = 1
        graph_key = self._build_cuda_graph_key(prefill_kwargs, seq_bucket, decode_capacity)

        state = self._cuda_graph_cache.get(graph_key)
        if (
            state is None
            or current_seq_len > state['max_supported_seq_len']
            or decode_capacity > state['max_decode_tokens']
        ):
            state = self._create_decode_state(
                device=input_ids.device,
                input_dtype=input_ids.dtype,
                max_supported_seq_len=seq_bucket,
                max_decode_tokens=decode_capacity,
            )
            self._cuda_graph_cache[graph_key] = state
            self._last_generation_ctx = None

        if max_new_tokens == 1:
            eos_token_ids = self._get_eos_token_ids(generate_kwargs)
            prefill_identity = self._capture_prefill_identity(prefill_kwargs)
            first_token, rope_scalar = self._prefill_first_token_graph(prefill_kwargs, state, current_seq_len)
            self._last_generation_ctx = {
                'prefill_identity': prefill_identity,
                'state': state,
                'seq_len': current_seq_len,
                'rope_scalar': int(rope_scalar),
                'generated_tokens': first_token.clone(),
                'last_token': first_token.clone(),
                'finished': bool(eos_token_ids and int(first_token[0, 0].item()) in eos_token_ids),
            }
            return torch.cat([input_ids, first_token], dim=1)

        eos_token_ids = self._get_eos_token_ids(generate_kwargs)
        ctx = self._last_generation_ctx
        if (
            self._ttft_kv_reuse_enabled
            and ctx is not None
            and state['captured']
            and self._prefill_identity_matches(ctx.get('prefill_identity'), prefill_kwargs)
            and ctx['state'] is state
            and ctx['seq_len'] == current_seq_len
        ):
            generated_tokens = ctx['generated_tokens']
            generated_count = int(generated_tokens.shape[1])
            if ctx['finished'] or max_new_tokens <= generated_count:
                out_tokens = generated_tokens[:, :max_new_tokens]
                return torch.cat([input_ids, out_tokens], dim=1)

            result_tokens = torch.empty((1, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
            result_tokens[:, :generated_count].copy_(generated_tokens)

            rope_scalar = int(ctx['rope_scalar'])
            self._set_decode_start(state, ctx['last_token'], current_seq_len, rope_scalar, generated_count - 1)

            finished = False
            actual_len = generated_count
            for token_idx in range(generated_count, max_new_tokens):
                state['graph'].replay()
                result_tokens[:, token_idx:token_idx + 1].copy_(state['static_decode_token'])
                actual_len = token_idx + 1
                if eos_token_ids and int(state['static_decode_token'][0, 0].item()) in eos_token_ids:
                    finished = True
                    break

            generated_tokens = result_tokens[:, :actual_len]
            self._last_generation_ctx = {
                'prefill_identity': self._capture_prefill_identity(prefill_kwargs),
                'state': state,
                'seq_len': current_seq_len,
                'rope_scalar': rope_scalar,
                'generated_tokens': generated_tokens,
                'last_token': generated_tokens[:, -1:].clone(),
                'finished': finished,
            }
            return torch.cat([input_ids, generated_tokens], dim=1)

        first_token, rope_scalar = self._prefill_first_token_graph(prefill_kwargs, state, current_seq_len)

        if max_new_tokens > 1 and not state['captured']:
            self._capture_decode_graph(prefill_kwargs, state, rope_scalar, first_token, current_seq_len)
            first_token, rope_scalar = self._prefill_first_token_graph(prefill_kwargs, state, current_seq_len)

        result_tokens = torch.empty((1, max_new_tokens), device=input_ids.device, dtype=input_ids.dtype)
        result_tokens[:, 0:1].copy_(first_token)
        actual_len = 1
        finished = eos_token_ids and int(first_token[0, 0].item()) in eos_token_ids

        if max_new_tokens > 1 and not finished:
            self._set_decode_start(state, first_token, current_seq_len, rope_scalar, 0)
            for token_idx in range(1, max_new_tokens):
                state['graph'].replay()
                result_tokens[:, token_idx:token_idx + 1].copy_(state['static_decode_token'])
                actual_len = token_idx + 1
                if eos_token_ids and int(state['static_decode_token'][0, 0].item()) in eos_token_ids:
                    finished = True
                    break

        generated_tokens = result_tokens[:, :actual_len]
        self._last_generation_ctx = {
            'prefill_identity': self._capture_prefill_identity(prefill_kwargs),
            'state': state,
            'seq_len': current_seq_len,
            'rope_scalar': int(rope_scalar),
            'generated_tokens': generated_tokens,
            'last_token': generated_tokens[:, -1:].clone(),
            'finished': bool(finished),
        }
        return torch.cat([input_ids, generated_tokens], dim=1)

    # Required properties for benchmark
    @property
    def processor(self):
        """
        Required by benchmark for input processing.

        Benchmark uses this to prepare inputs with unified tokenizer.
        """
        return self._processor

    @property
    def model(self):
        """
        Required by benchmark for direct model.generate() calls.

        Benchmark directly calls self.model.generate() for performance testing.
        Your optimizations should modify this model object or its operators.
        """
        return self._model

    @property
    def device(self):
        """
        Required by benchmark for device information.
        """
        return self._device

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 128
    ) -> Dict:
        """
        Generate answer (optional method, mainly for debugging).

        Note: Benchmark uses self.model.generate() directly for performance testing.
        This method is provided for convenience and debugging purposes.

        Args:
            image: PIL Image object
            question: Question text
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict: {
                "text": str,        # Generated text answer
                "token_count": int  # Generated token count
            }
        """
        # Build Qwen3-VL message format
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]

        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)

        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True
            )

        # Extract generated tokens (remove input part)
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]

        # Decode
        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return {
            "text": text,
            "token_count": len(generated_ids)
        }
