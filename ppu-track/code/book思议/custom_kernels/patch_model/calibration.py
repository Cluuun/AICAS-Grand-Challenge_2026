"""DASH 缩放校准：per-channel 最小二乘法。"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def _calibrate_mlp_skip_scales(model, skip_layers):
    """DASH scaling: per-channel least-squares calibration.

    For each skip layer, we approximate:
        attn_out + MLP(LN(residual + attn_out)) ≈ scale_c * attn_out
    Solving per-channel via least squares:
        scale_c = sum(attn_c * (attn_c + mlp_c)) / (sum(attn_c^2) + eps)
    The scale vector is stored as layer._mlp_skip_scale (shape [hidden_size]).
    """
    use_per_channel = os.environ.get("AICAS_DASH_PER_CHANNEL", "1") != "0"
    lm = model.model.language_model
    device = next(lm.parameters()).device
    hidden_dim = lm.config.hidden_size
    n_cal = int(os.environ.get("AICAS_DASH_CALIB_SAMPLES", "500"))

    calibrated = 0
    for li in sorted(skip_layers):
        if li >= len(lm.layers):
            continue
        layer = lm.layers[li]
        mlp = layer.mlp
        norm_w = layer.post_attention_layernorm.weight
        eps = layer.post_attention_layernorm.variance_epsilon

        if use_per_channel:
            # Per-channel least-squares: accumulate numerator and denominator
            sum_attn2 = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
            sum_attn_target = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
            for _ in range(n_cal):
                with torch.no_grad():
                    residual = torch.randn(1, 1, hidden_dim, device=device, dtype=torch.float16) * 0.5
                    attn_out = torch.randn(1, 1, hidden_dim, device=device, dtype=torch.float16) * 0.3

                    combined = residual + attn_out
                    variance = combined.float().pow(2).mean(dim=-1, keepdim=True)
                    normed = (combined.float() * torch.rsqrt(variance + eps) * norm_w.float()).half()

                    mlp_out = mlp(normed)

                    a = attn_out.float().reshape(-1)     # [hidden_dim]
                    t = (attn_out.float() + mlp_out.float()).reshape(-1)  # target
                    sum_attn2 += a * a
                    sum_attn_target += a * t

            scale_vec = sum_attn_target / (sum_attn2 + 1e-4)
            # Clamp to reasonable range
            scale_vec = scale_vec.clamp(0.01, 20.0)
            # Store as a parameter-like tensor on the correct device
            layer._mlp_skip_scale = scale_vec.half().to(device)
            calibrated += 1
            scalar_eq = scale_vec.mean().item()
            logger.info(f"[mlp_skip_calibrate] Layer {li}: per-channel scale, mean={scalar_eq:.4f}, "
                        f"std={scale_vec.std().item():.4f}, range=[{scale_vec.min().item():.4f}, {scale_vec.max().item():.4f}]")
        else:
            # Scalar DASH (original L1-norm calibration)
            ratios = []
            for _ in range(n_cal):
                with torch.no_grad():
                    residual = torch.randn(1, 1, hidden_dim, device=device, dtype=torch.float16) * 0.5
                    attn_out = torch.randn(1, 1, hidden_dim, device=device, dtype=torch.float16) * 0.3

                    combined = residual + attn_out
                    variance = combined.float().pow(2).mean(dim=-1, keepdim=True)
                    normed = (combined.float() * torch.rsqrt(variance + eps) * norm_w.float()).half()

                    mlp_out = mlp(normed)

                    attn_l1 = attn_out.float().abs().mean().item()
                    mlp_l1 = mlp_out.float().abs().mean().item()
                    if attn_l1 > 1e-6:
                        ratios.append(mlp_l1 / attn_l1)

            if ratios:
                scale = sum(ratios) / len(ratios)
                scale = max(0.01, min(scale, 10.0))
                layer._mlp_skip_scale = scale
                calibrated += 1
                logger.info(f"[mlp_skip_calibrate] Layer {li}: scale={scale:.4f}")

    if calibrated:
        mode = "per-channel LS" if use_per_channel else "scalar L1"
        print(f"[mlp_skip_calibrate] Calibrated {calibrated} layers ({mode})")
