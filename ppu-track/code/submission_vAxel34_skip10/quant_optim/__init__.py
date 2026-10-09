"""quant_optim — modular post-training quantization utilities for Qwen3-VL on PPU.

Public surface (Stage 1 = LLM-decode W8A16, no calibration, RTN per-channel):
    W8A16Weight              — packed INT8 weight + FP16 scales + acext layout
    w8a16_linear             — y = x @ W.T (+ bias) via WeightOnlyQuantMatmul
    numerical_check          — op-level cosine sanity vs FP16 reference
    W8A16LLMBundle           — per-layer fused QKV/MLP + lm_head bundle
    build_llm_decode_w8a16_bundle — factory: walk a Qwen3-VL model, build bundle

Stage 2 (ViT W8A8 SmoothQuant) lives in `.w8a8` once Stage 1 ships.
"""
from .w8a16 import (
    W8A16Weight,
    w8a16_linear,
    numerical_check,
)
from .llm_decode_quant import (
    W8A16AttnBundle,
    W8A16MLPBundle,
    W8A16LLMBundle,
    build_llm_decode_w8a16_bundle,
)
from .decode_patch import install_w8a16_decode_patches
from .w8a8 import W8A8Weight, w8a8_linear, numerical_check_w8a8
from .vit_quant import ViTW8A8Bundle, build_vit_fc1_w8a8_bundle
from .llm_prefill_quant import (
    W8A8AttnPrefillBundle,
    W8A8MLPPrefillBundle,
    W8A8LLMPrefillBundle,
    build_llm_prefill_w8a8_bundle,
)
from .calibration import (
    TextVQACalibrationLoader,
    collect_vit_fc1_activation_stats,
)

__all__ = [
    "W8A16Weight",
    "w8a16_linear",
    "numerical_check",
    "W8A16AttnBundle",
    "W8A16MLPBundle",
    "W8A16LLMBundle",
    "build_llm_decode_w8a16_bundle",
    "install_w8a16_decode_patches",
    "W8A8Weight",
    "w8a8_linear",
    "numerical_check_w8a8",
    "ViTW8A8Bundle",
    "build_vit_fc1_w8a8_bundle",
    "W8A8AttnPrefillBundle",
    "W8A8MLPPrefillBundle",
    "W8A8LLMPrefillBundle",
    "build_llm_prefill_w8a8_bundle",
    "TextVQACalibrationLoader",
    "collect_vit_fc1_activation_stats",
]
