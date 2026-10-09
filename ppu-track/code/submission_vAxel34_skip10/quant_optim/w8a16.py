"""W8A16 (weight-only INT8) GEMM via acext WeightOnlyQuantMatmul.

PPU-only path: `torch.classes._C.WeightOnlyQuantMatmul.weightonly_gemm` from
`vllm/_C.abi3.so` + `acext.preprocess_weights_for_mixed_gemm`.

Designed for batch=1 decode + lm_head (M=1) where weight bandwidth dominates.
Per quantization_workspace bench (M=1):
- lm_head 2048→151936: 1.79×
- gate_up 2048→12288:  1.11×
- qkv/o/down:          0.97-1.03×
Prefill (M≥64): negative, NEVER use here.

Modular API:
    from quant_optim.w8a16 import W8A16Weight, w8a16_linear

    w8 = W8A16Weight.from_fp16(fp16_weight)   # [N, K] FP16 -> packed
    out = w8a16_linear(x, w8, bias)           # x [*, K] -> [*, N]
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional

import torch

_LIBRARY_LOCK = threading.Lock()
_LIBRARY_LOADED = False


def _ensure_acext_loaded():
    """Idempotent load of vllm/_C.abi3.so. Safe across multiple imports."""
    global _LIBRARY_LOADED
    if _LIBRARY_LOADED:
        return
    with _LIBRARY_LOCK:
        if _LIBRARY_LOADED:
            return
        path = os.environ.get(
            "ACEXT_VLLM_C_PATH",
            "/usr/local/lib/python3.12/site-packages/vllm/_C.abi3.so",
        )
        try:
            torch.ops.load_library(path)
        except Exception as e:
            # Re-loading after the same lib was already loaded by vllm import
            # raises a duplicate-registration error; treat as success.
            if "duplicate" not in str(e).lower():
                raise
        _LIBRARY_LOADED = True


# Lazy singletons for the script class instance (it's stateless).
_WO_MATMUL_CACHE: Optional[object] = None


def _get_wo_matmul():
    global _WO_MATMUL_CACHE
    if _WO_MATMUL_CACHE is None:
        _ensure_acext_loaded()
        _WO_MATMUL_CACHE = torch.classes._C.WeightOnlyQuantMatmul()
    return _WO_MATMUL_CACHE


@dataclass
class W8A16Weight:
    """Packed INT8 weight + per-channel FP16 scales for acext W8A16 GEMM.

    Layout (set by acext.preprocess_weights_for_mixed_gemm):
        qweight: [K, N] interleaved INT8 (column-major-interleaved)
        scales:  [N] FP16/BF16
    Use via `w8a16_linear(x, weight, bias)`.

    Attributes:
        qweight: preprocessed int8 weight, shape [K, N]
        scales:  per-output-channel FP16 scales, shape [N]
        out_features: N
        in_features:  K
    """
    qweight: torch.Tensor
    scales: torch.Tensor
    out_features: int
    in_features: int

    @classmethod
    def from_fp16(
        cls,
        fp16_weight: torch.Tensor,
        clip_percentile: Optional[float] = None,
    ) -> "W8A16Weight":
        """Quantize an FP16 weight tensor of shape [N, K] (PyTorch convention).

        Args:
            fp16_weight: [N_out, K_in] FP16/BF16 tensor on any device.
            clip_percentile: optional value in (0, 1]; if set, scale uses the
                given percentile of |W| per row instead of absmax. Default
                None = pure absmax (RTN per-channel symmetric).
        """
        _ensure_acext_loaded()
        from acext import preprocess_weights_for_mixed_gemm

        assert fp16_weight.dim() == 2, f"weight must be 2D [N,K], got {fp16_weight.shape}"
        N, K = fp16_weight.shape
        orig_dtype = fp16_weight.dtype
        orig_device = fp16_weight.device

        w_f = fp16_weight.detach().float()
        if clip_percentile is None or clip_percentile >= 1.0:
            absmax = w_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            absmax = torch.quantile(
                w_f.abs(), clip_percentile, dim=1, keepdim=True
            ).clamp(min=1e-8)
        scales = (absmax / 127.0).to(orig_dtype)
        w_q = (w_f / scales.float()).round().clamp(-127, 127).to(torch.int8)

        # acext preprocess expects [K, N] on CPU
        qweight_cpu = w_q.t().contiguous().cpu()
        qweight_pp = preprocess_weights_for_mixed_gemm(
            qweight_cpu, torch.int8, False, False
        )

        return cls(
            qweight=qweight_pp.to(orig_device).contiguous(),
            scales=scales.squeeze(-1).contiguous().to(orig_device),
            out_features=N,
            in_features=K,
        )

    def to(self, device) -> "W8A16Weight":
        return W8A16Weight(
            qweight=self.qweight.to(device),
            scales=self.scales.to(device),
            out_features=self.out_features,
            in_features=self.in_features,
        )

    @property
    def shape_fp16(self):
        return (self.out_features, self.in_features)


def w8a16_linear(
    x: torch.Tensor,
    weight: W8A16Weight,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W8A16 linear: y = x @ W.T (+ bias). x [*, K] -> [*, N].

    `weight.qweight` must already be acext-preprocessed.
    The acext kernel itself only accepts 2D [M, K] inputs; we flatten
    leading dims and reshape back.
    """
    inst = _get_wo_matmul()
    orig_shape = x.shape
    K = weight.in_features
    N = weight.out_features
    x2d = x.reshape(-1, K).contiguous()
    out2d = inst.weightonly_gemm(x2d, weight.qweight, weight.scales, bias)
    return out2d.reshape(*orig_shape[:-1], N)


def numerical_check(
    fp16_weight: torch.Tensor,
    sample_input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cos_threshold: float = 0.99,
) -> tuple[bool, float, float]:
    """Op-level golden check: compare W8A16 output vs FP16 reference.

    Returns (passed, cosine_similarity, max_rel_err).
    """
    w8 = W8A16Weight.from_fp16(fp16_weight)
    out_q = w8a16_linear(sample_input, w8, bias)
    out_ref = torch.nn.functional.linear(sample_input, fp16_weight, bias)
    flat_q = out_q.flatten().float()
    flat_r = out_ref.flatten().float()
    cos = torch.nn.functional.cosine_similarity(flat_q, flat_r, dim=0).item()
    rel = (out_q - out_ref).abs().max().item() / max(out_ref.abs().max().item(), 1e-3)
    return (cos >= cos_threshold), cos, rel
