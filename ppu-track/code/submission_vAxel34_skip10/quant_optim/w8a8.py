"""W8A8 (per-token activation, per-channel weight) GEMM via acext SmoothQuantMatmul.

PPU-only path: `torch.classes._C.SmoothQuantMatmul(True, True).smoothquant_gemm`.
Designed for prefill / ViT (M >= 256) where W8A16's bandwidth advantage flips
to negative; the SmoothQuant kernel is +1.4-1.6x at those shapes.

Modular API:
    from quant_optim.w8a8 import W8A8Weight, w8a8_linear

    w8 = W8A8Weight.from_fp16(fp16_weight, bias)        # [N,K] FP16 -> int8
    out = w8a8_linear(x, w8)                            # x [*,K] -> [*,N]

Notes:
- Per-token (per-row) activation absmax is computed at runtime; no calibration.
- Output dtype follows input dtype (FP16 in, FP16 out).
- The acext SmoothQuant kernel demands FP32 scales; we convert at call time
  if the stored weight scale is in another dtype.
- The kernel only takes 2D inputs; we flatten leading dims and reshape back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .w8a16 import _ensure_acext_loaded


_SQ_MATMUL_CACHE: Optional[object] = None


def _get_sq_matmul():
    global _SQ_MATMUL_CACHE
    if _SQ_MATMUL_CACHE is None:
        _ensure_acext_loaded()
        # (per_token_activation, per_channel_weight)
        _SQ_MATMUL_CACHE = torch.classes._C.SmoothQuantMatmul(True, True)
    return _SQ_MATMUL_CACHE


@dataclass
class W8A8Weight:
    """INT8 weight in [N, K] layout + FP32 per-output-channel scales.

    Optionally carries a fused FP16 bias for the linear; the SmoothQuant
    kernel takes the bias as an optional Tensor argument.

    `smooth_inv_s` (optional, [K]): per-input-channel inverse migration
    scale produced by SmoothQuant α-fold calibration. When present,
    `w8a8_linear` multiplies the input by `smooth_inv_s` (broadcast on
    the last dim) before the GEMM. The corresponding scale is folded
    into `qweight` at build time, so accuracy is mathematically equivalent
    to the FP16 path up to per-channel rescale + INT8 round-trip.
    """
    qweight: torch.Tensor    # [N, K] int8
    w_scale: torch.Tensor    # [N] float32
    bias: Optional[torch.Tensor]
    out_features: int
    in_features: int
    smooth_inv_s: Optional[torch.Tensor] = None  # [K] same dtype as runtime input

    @classmethod
    def from_fp16(
        cls,
        fp16_weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        clip_percentile: Optional[float] = None,
    ) -> "W8A8Weight":
        """Quantize FP16 weight [N,K] to INT8 with per-row symmetric scale."""
        assert fp16_weight.dim() == 2, f"expected [N,K], got {fp16_weight.shape}"
        N, K = fp16_weight.shape
        device = fp16_weight.device

        w_f = fp16_weight.detach().float()
        if clip_percentile is None or clip_percentile >= 1.0:
            absmax = w_f.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            absmax = torch.quantile(
                w_f.abs(), clip_percentile, dim=1, keepdim=True
            ).clamp(min=1e-8)
        scale = (absmax / 127.0)  # [N,1] float32
        w_int8 = (w_f / scale).round().clamp(-127, 127).to(torch.int8)

        return cls(
            qweight=w_int8.contiguous(),
            w_scale=scale.squeeze(-1).contiguous().to(device, dtype=torch.float32),
            bias=(bias.detach().contiguous() if bias is not None else None),
            out_features=N,
            in_features=K,
        )

    @classmethod
    def from_calibrated(
        cls,
        fp16_weight: torch.Tensor,
        act_absmax: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        s_min: float = 1e-5,
        s_max: float = 1e5,
    ) -> "W8A8Weight":
        """SmoothQuant α-migration W8A8.

        Migration scale per input channel:
            s_k = max|X|_k^α / max|W|_k^(1-α)
        Y = X @ W^T  ≡  (X / s) @ (W * s)^T   (s broadcast on K axis)

        We fold `s` into the weight (W' = W * s), quantize W' per-row,
        and store `inv_s = 1/s` for the runtime input rescale. The
        SmoothQuant kernel call is unchanged; only `w8a8_linear` checks
        `smooth_inv_s` and applies it before quantizing the activation.

        Args:
            fp16_weight: [N, K] FP16 weight (HF Linear weight layout).
            act_absmax:  [K] per-input-channel activation absmax (collected
                         from `collect_vit_fc1_activation_stats`).
            bias:        optional [N] FP16 bias.
            alpha:       0.0 = all migration to weight, 1.0 = all to act,
                         0.5 = balanced (textbook SmoothQuant default).
            s_min/s_max: clamp on `s` to avoid catastrophic rescale on
                         dead channels.

        Numerics: the migration is exact in FP32 before quantization;
        accuracy delta vs `from_fp16` comes from how the rescale
        flattens the per-row max distribution (smaller per-row dynamic
        range → smaller quant noise on heavy-outlier channels).
        """
        assert fp16_weight.dim() == 2, f"expected [N,K], got {fp16_weight.shape}"
        N, K = fp16_weight.shape
        device = fp16_weight.device
        out_dtype = fp16_weight.dtype

        assert act_absmax.numel() == K, (
            f"act_absmax must have shape [K]={K}, got {tuple(act_absmax.shape)}"
        )

        w_f = fp16_weight.detach().float()
        a = act_absmax.detach().float().to(device).clamp(min=1e-8)
        w_per_k = w_f.abs().amax(dim=0).clamp(min=1e-8)  # [K]

        s = (a.pow(alpha) / w_per_k.pow(1.0 - alpha)).clamp(min=s_min, max=s_max)
        # Fold s into weight on K-axis. W'_{n,k} = W_{n,k} * s_k.
        w_scaled = w_f * s.unsqueeze(0)

        absmax = w_scaled.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 127.0
        w_int8 = (w_scaled / scale).round().clamp(-127, 127).to(torch.int8)

        inv_s = (1.0 / s).to(out_dtype).contiguous().to(device)

        return cls(
            qweight=w_int8.contiguous(),
            w_scale=scale.squeeze(-1).contiguous().to(device, dtype=torch.float32),
            bias=(bias.detach().contiguous() if bias is not None else None),
            out_features=N,
            in_features=K,
            smooth_inv_s=inv_s,
        )

    def to(self, device) -> "W8A8Weight":
        return W8A8Weight(
            qweight=self.qweight.to(device),
            w_scale=self.w_scale.to(device),
            bias=(self.bias.to(device) if self.bias is not None else None),
            out_features=self.out_features,
            in_features=self.in_features,
            smooth_inv_s=(
                self.smooth_inv_s.to(device)
                if self.smooth_inv_s is not None else None
            ),
        )

    @property
    def shape_fp16(self):
        return (self.out_features, self.in_features)


def w8a8_linear(x: torch.Tensor, weight: W8A8Weight) -> torch.Tensor:
    """W8A8 linear: y = x @ W.T (+ bias). x [*, K] -> [*, N].

    - Per-token activation absmax is computed at runtime.
    - If `weight.smooth_inv_s` is set (SmoothQuant α-migration), the input
      is rescaled by `inv_s` per input-channel before activation quant.
    - Bias is folded via the kernel's optional bias argument.
    """
    inst = _get_sq_matmul()
    K = weight.in_features
    N = weight.out_features
    orig_shape = x.shape
    if weight.smooth_inv_s is not None:
        x = x * weight.smooth_inv_s
    x2d = x.reshape(-1, K).contiguous()

    # Per-token (per-row) absmax -> int8
    a_absmax = x2d.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    a_scale = (a_absmax / 127.0).to(torch.float32)        # [M,1]
    x_int8 = (x2d.float() / a_scale).round().clamp(-127, 127).to(torch.int8)

    out2d = inst.smoothquant_gemm(
        x_int8, weight.qweight, a_scale.squeeze(-1).contiguous(),
        weight.w_scale, weight.bias)
    return out2d.reshape(*orig_shape[:-1], N).to(x.dtype)


def numerical_check_w8a8(
    fp16_weight: torch.Tensor,
    sample_input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cos_threshold: float = 0.99,
) -> tuple[bool, float, float]:
    """Op-level golden check vs FP16 reference."""
    w8 = W8A8Weight.from_fp16(fp16_weight, bias)
    out_q = w8a8_linear(sample_input, w8)
    out_ref = torch.nn.functional.linear(sample_input, fp16_weight, bias)
    flat_q = out_q.flatten().float()
    flat_r = out_ref.flatten().float()
    cos = torch.nn.functional.cosine_similarity(flat_q, flat_r, dim=0).item()
    denom = max(out_ref.abs().max().item(), 1e-3)
    rel = (out_q - out_ref).abs().max().item() / denom
    return (cos >= cos_threshold), cos, rel
