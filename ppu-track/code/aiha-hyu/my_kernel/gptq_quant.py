"""[SHLEE] GPTQ-style weight quantization, output format unchanged.

Produces the same (qweight, scales) tuple as quantize_pack_w4_sym so the
fused_w4_gemv kernel is reused without modification.

Algorithm (Frantar et al., 2022):
  1. Compute per-layer input-activation Hessian H = X^T X over a small
     calibration set.
  2. Sequentially quantize the weight columns: rounded value of column j
     is fixed, residual error is back-propagated to all remaining columns
     using the inverse Hessian, so the residual quantization error is
     compensated optimally in the L2 sense.

Used after AWQ scale absorption: the AWQ-scaled weights are passed to
GPTQ, which further reduces L2 quantization error.  Final output is then
packed via the standard W4 sym layout.
"""
from __future__ import annotations
import torch
import torch.nn as nn


@torch.no_grad()
def _per_group_round_w4_sym(col: torch.Tensor, group_scales: torch.Tensor,
                            group_idx: int) -> torch.Tensor:
    """Round a single column (out_features,) to its assigned per-group scale.
    group_scales: (out, n_groups). Returns the dequantized (fp16-equivalent)
    rounded value as fp32."""
    s = group_scales[:, group_idx].clamp_min(1e-8)
    q = (col / s).round().clamp(-8, 7)
    return q * s


@torch.no_grad()
def _compute_group_scales(W: torch.Tensor, group: int) -> torch.Tensor:
    """W: (out, in) fp32.  Returns scales (out, n_groups) fp32."""
    out, in_dim = W.shape
    n_groups = (in_dim + group - 1) // group
    W_grouped = W.view(out, n_groups, group)
    max_abs = W_grouped.abs().amax(dim=-1)
    scales = max_abs / 7.0
    return scales


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor,
                  group: int = 128, percdamp: float = 0.01,
                  blocksize: int = 128) -> torch.Tensor:
    """Apply GPTQ-style sequential quantization.

    Args:
        W:    (out, in) fp16 — input weights (already AWQ-scaled).
        H:    (in, in)  fp32 — input activation Hessian X^T X.
        group: group size for W4 quantization.
        percdamp:  Hessian diagonal damping (fraction of mean diag).
        blocksize: inner block size for batched updates.

    Returns:
        Q:    (out, in) fp16 — rounded (dequantized) weight.  Pass to
              quantize_pack_w4_sym to obtain the int4 packed form.
    """
    n_rows, n_cols = W.shape
    device = W.device
    dtype  = W.dtype
    W_f = W.float().clone()
    H_f = H.float().clone()

    dead = torch.diag(H_f) == 0
    H_f[dead, dead] = 1.0
    W_f[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(H_f)).clamp_min(1e-8)
    diag_idx = torch.arange(n_cols, device=device)
    H_f[diag_idx, diag_idx] += damp

    try:
        L = torch.linalg.cholesky(H_f)
        H_inv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(H_inv, upper=True)
    except Exception as e:
        print(f"[gptq] Cholesky failed ({e}); falling back to plain RTN")
        scales = _compute_group_scales(W_f, group)
        Q = torch.zeros_like(W_f)
        for j in range(n_cols):
            Q[:, j] = _per_group_round_w4_sym(W_f[:, j], scales, j // group)
        return Q.to(dtype)

    scales = _compute_group_scales(W_f, group)
    Q = torch.zeros_like(W_f)

    for i in range(0, n_cols, blocksize):
        i2 = min(i + blocksize, n_cols)
        count = i2 - i

        W_block = W_f[:, i:i2].clone()
        Q_block = torch.zeros_like(W_block)
        err_block = torch.zeros_like(W_block)
        U_block = U[i:i2, i:i2]

        for j in range(count):
            col_idx = i + j
            w = W_block[:, j]
            d = U_block[j, j]
            q = _per_group_round_w4_sym(w, scales, col_idx // group)
            Q_block[:, j] = q
            e = (w - q) / d
            err_block[:, j] = e
            if j + 1 < count:
                W_block[:, j+1:count] -= (
                    e.unsqueeze(1) * U_block[j, j+1:count].unsqueeze(0)
                )

        Q[:, i:i2] = Q_block
        if i2 < n_cols:
            W_f[:, i2:] -= err_block @ U[i:i2, i2:]

    return Q.to(dtype)


@torch.no_grad()
def collect_hessian_per_proj(model, samples, processor, device: str,
                             max_pixels: int) -> dict:
    """Register hooks on input_layernorm + post_attention_layernorm so we
    accumulate H = X^T X over the calibration set.

    Returns:
        { "in_ln_{i}":   (in_dim, in_dim) fp32,
          "post_ln_{i}": (in_dim, in_dim) fp32, ... }
    """
    lang = model.model.language_model
    n_layers = len(lang.layers)
    H: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def make_hook(key: str):
        def _hook(module, inp, out):
            x = out if isinstance(out, torch.Tensor) else out[0]
            x2 = x.detach().reshape(-1, x.shape[-1]).float()
            cov = x2.t() @ x2
            cnt = x2.shape[0]
            cov = cov.cpu()
            if key in H:
                H[key] += cov
                counts[key] += cnt
            else:
                H[key] = cov
                counts[key] = cnt
        return _hook

    handles = []
    for i, layer in enumerate(lang.layers):
        handles.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"in_ln_{i}")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"post_ln_{i}")))

    try:
        for idx, (img, q) in enumerate(samples):
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": q},
            ]}]
            try:
                inputs = processor.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ).to(device)
            except Exception as e:
                print(f"[gptq_calib] sample {idx} template fail: {e}")
                continue
            try:
                model(**inputs, use_cache=False, return_dict=True)
            except Exception as e:
                print(f"[gptq_calib] sample {idx} forward fail: {e}")
                continue
            if (idx + 1) % 8 == 0:
                print(f"[gptq_calib] {idx+1}/{len(samples)} samples")
    finally:
        for h in handles:
            h.remove()

    for k in list(H.keys()):
        if counts.get(k, 0):
            H[k] = H[k] / counts[k]
    return H


@torch.no_grad()
def apply_gptq_inplace(model, calibration_samples, processor,
                       device: str = "cuda:0", max_pixels: int = 300000,
                       group: int = 128, percdamp: float = 0.01):
    """Apply GPTQ to qkv_proj and gate_up_proj using the input-activation
    Hessian.  Run *after* apply_awq_inplace so the AWQ-scaled weights are
    the input to GPTQ.

    Weights are modified in-place (proj.weight.data <- Q_dq).  Downstream
    consumers (lean_decode's per-bucket quantize_pack_w4_sym, eager
    forward) then see weights whose RTN re-quantization yields the same
    qweight that GPTQ computed.
    """
    lang = model.model.language_model
    n_layers = len(lang.layers)

    print(f"[gptq] collecting Hessians over "
          f"{len(calibration_samples)} samples...")
    H_dict = collect_hessian_per_proj(
        model, calibration_samples, processor, device, max_pixels)

    print(f"[gptq] applying GPTQ to {n_layers} layers")
    n_done = 0
    for i, layer in enumerate(lang.layers):
        for h_key, proj_paths in [
            (f"in_ln_{i}",
             [("self_attn", "fused_qkv", "qkv_proj"),
              ("self_attn", "q_proj"),
              ("self_attn", "k_proj"),
              ("self_attn", "v_proj")]),
            (f"post_ln_{i}",
             [("mlp", "gate_up_proj"),
              ("mlp", "gate_proj"),
              ("mlp", "up_proj")]),
        ]:
            H = H_dict.get(h_key)
            if H is None:
                continue

            for proj_path in proj_paths:
                mod = layer
                for attr in proj_path:
                    mod = getattr(mod, attr, None)
                    if mod is None: break
                if mod is None or not hasattr(mod, "weight"):
                    continue

                W = mod.weight.data
                H_dev = H.to(W.device)
                Q = gptq_quantize(W, H_dev, group=group, percdamp=percdamp)
                mod.weight.data.copy_(Q)
                n_done += 1
                if n_done % 4 == 0:
                    print(f"[gptq] {n_done} projections processed")

    print(f"[gptq] done — modified {n_done} projection weights")
    return n_done
