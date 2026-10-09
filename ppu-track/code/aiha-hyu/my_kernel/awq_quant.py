"""[SHLEE] In-process AWQ scale computation + RMSNorm absorption.

Output is identical to RTN's quantize_pack_w4_sym format:
  - qweight: int4 packed
  - scales:  fp16 per-group
so fused_w4_gemv kernel is reused unchanged.

AWQ mechanism (Lin et al., 2023):
  1. Find per-input-channel scale s minimizing
       || W*x - quant(W*s) * (x/s) ||^2
     for each input projection.
  2. Absorb 1/s into the preceding RMSNorm weight:
       gamma_new = gamma_old / s
     so the activation x sees the scaled-down value naturally.
  3. Quantize (W * s) using RTN — high-activation channels survive
     with relatively larger scale → lower quantization error.

We absorb into the preceding norm for two projections per decoder layer:
  - input_layernorm  → fused_qkv.qkv_proj
  - post_attention_layernorm → mlp.gate_up_proj
Other projections (o_proj, down_proj) have no preceding norm; we apply
plain RTN there (or absorb into the previous matmul output scale, omitted
for simplicity).
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn


@torch.no_grad()
def _compute_awq_scale(W: torch.Tensor, x_abs_mean: torch.Tensor,
                       n_alpha: int = 20,
                       group: int = 128) -> torch.Tensor:
    """Grid-search alpha in [0, 1] to find s = x_abs_mean ** alpha
    that minimizes round-trip quantization error on W*x_abs_mean.

    W:           (out, in) fp16/fp32
    x_abs_mean:  (in,)    fp32 — per-channel activation magnitude
    Returns s:   (in,)    fp16
    """
    from my_kernel.fused_w4_gemv import quantize_pack_w4_sym, dequant_w4_to_fp16

    device = W.device
    W_f32 = W.float()
    x = x_abs_mean.float().to(device).clamp_min(1e-5)
    x = x / x.mean()

    best_err = None
    best_s = None
    for i in range(n_alpha):
        alpha = (i + 1) / n_alpha
        s = x.pow(alpha).clamp_min(1e-4).to(W.dtype)
        W_scaled = (W_f32 * s.float()).to(W.dtype)
        qw, sc = quantize_pack_w4_sym(W_scaled, group)
        W_dq = dequant_w4_to_fp16(qw, sc, group, W.shape[1]).to(W.dtype)
        W_recovered = W_dq.float() / s.float().unsqueeze(0)
        err = ((W_recovered - W_f32) ** 2).sum().item()
        if best_err is None or err < best_err:
            best_err = err
            best_s = s
    return best_s


@torch.no_grad()
def _collect_input_stats(model, samples, processor, device: str,
                         max_pixels: int) -> dict:
    """Run a forward pass per sample and hook input_layernorm /
    post_attention_layernorm outputs to accumulate per-channel
    activation magnitude.

    Returns:
        { "in_ln_{i}":   tensor (in_features,),
          "post_ln_{i}": tensor (in_features,), ... }
    """
    lang = model.model.language_model
    n_layers = len(lang.layers)
    stats: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}

    def make_hook(key: str):
        def _hook(module, inp, out):
            x = out if isinstance(out, torch.Tensor) else out[0]
            x_abs = x.detach().float().abs()
            x_abs = x_abs.reshape(-1, x_abs.shape[-1]).mean(dim=0)
            if key in stats:
                stats[key] += x_abs.cpu()
                counts[key] += 1
            else:
                stats[key] = x_abs.cpu()
                counts[key] = 1
        return _hook

    handles = []
    for i, layer in enumerate(lang.layers):
        handles.append(layer.input_layernorm.register_forward_hook(
            make_hook(f"in_ln_{i}")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(
            make_hook(f"post_ln_{i}")))

    try:
        for i, (img, q) in enumerate(samples):
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
                print(f"[awq_calib] sample {i} template fail: {e}")
                continue
            try:
                model(**inputs, use_cache=False, return_dict=True)
            except Exception as e:
                print(f"[awq_calib] sample {i} forward fail: {e}")
                continue
            if (i + 1) % 16 == 0:
                print(f"[awq_calib] {i+1}/{len(samples)} samples")
    finally:
        for h in handles:
            h.remove()

    for k in list(stats.keys()):
        if counts.get(k, 0):
            stats[k] = stats[k] / counts[k]
    return stats


@torch.no_grad()
def apply_awq_inplace(model, calibration_samples, processor,
                      device: str = "cuda:0", max_pixels: int = 300000,
                      group: int = 128, n_alpha: int = 20):
    """Compute AWQ per-input-channel scales using activation statistics,
    absorb 1/s into the preceding RMSNorm weight, and overwrite the
    projection weights with the *scaled* version.

    The decoder weights are then ready for downstream RTN quantization
    via the standard quantize_pack_w4_sym path — no kernel changes
    needed.

    Apply to:
      - input_layernorm     -> fused_qkv.qkv_proj   (per-input scale on hidden)
      - post_attention_layernorm -> mlp.gate_up_proj (per-input scale on hidden)
    """
    lang = model.model.language_model
    n_layers = len(lang.layers)

    print(f"[awq] collecting activation stats over "
          f"{len(calibration_samples)} samples...")
    stats = _collect_input_stats(
        model, calibration_samples, processor, device, max_pixels)

    print(f"[awq] computing scales + absorbing into norms ({n_layers} layers)")
    n_done = 0
    for i, layer in enumerate(lang.layers):
        for ln_key, ln_attr, proj_paths in [
            (f"in_ln_{i}",   "input_layernorm",
             [("self_attn", "fused_qkv", "qkv_proj"),
              ("self_attn", "q_proj"),
              ("self_attn", "k_proj"),
              ("self_attn", "v_proj")]),
            (f"post_ln_{i}", "post_attention_layernorm",
             [("mlp", "gate_up_proj"),
              ("mlp", "gate_proj"),
              ("mlp", "up_proj")]),
        ]:
            x_stats = stats.get(ln_key)
            if x_stats is None:
                continue

            targets = []
            for proj_path in proj_paths:
                mod = layer
                for attr in proj_path:
                    mod = getattr(mod, attr, None)
                    if mod is None:
                        break
                if mod is not None and hasattr(mod, "weight"):
                    targets.append(mod)
            if not targets:
                continue

            norm_mod = getattr(layer, ln_attr)
            W_ref = targets[0].weight.data
            s = _compute_awq_scale(
                W_ref, x_stats.to(W_ref.device),
                n_alpha=n_alpha, group=group,
            )
            inv_s = (1.0 / s.float()).to(norm_mod.weight.dtype)
            norm_mod.weight.data.mul_(inv_s)
            for mod in targets:
                W = mod.weight.data
                W_scaled = W.float() * s.float().unsqueeze(0)
                mod.weight.data.copy_(W_scaled.to(W.dtype))
                n_done += 1

    print(f"[awq] applied AWQ to {n_done} projections "
          f"(qkv_proj + gate_up_proj across {n_layers} layers)")
    return n_done


def _load_calibration_samples(data: str, split: str, num_samples: int,
                              max_pixels: int):
    """Get (img, q) pairs.  Prefers the prepacked
    my_kernel/calib_samples.pkl shipped with the submission; falls back
    to a live HF datasets load if the pickle is missing.
    """
    from pathlib import Path
    from PIL import Image as _Image
    import io, pickle

    pkl_path = Path(__file__).with_name("calib_samples.pkl")
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            packed = pickle.load(f)
        samples = []
        for entry in packed[:num_samples]:
            img = _Image.open(io.BytesIO(entry["image_jpeg"])).convert("RGB")
            samples.append((img, entry["question"]))
        print(f"[calib] loaded {len(samples)} pre-packed samples from "
              f"{pkl_path.name}")
        return samples

    from datasets import load_from_disk, load_dataset
    p = Path(data)
    if p.exists() and ((p / "dataset_info.json").exists()
                       or (p / "state.json").exists()):
        ds = load_from_disk(str(p))
        if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
            ds = ds[split]
    else:
        ds = load_dataset(data, split=split, trust_remote_code=True)

    cols = set(ds.column_names)
    img_col = next(c for c in ("image", "img", "input_image") if c in cols)
    q_col   = next(c for c in ("question", "query", "input") if c in cols)
    samples = []
    for i in range(min(num_samples, len(ds))):
        s = ds[i]
        img = s[img_col]
        if not hasattr(img, "size"):
            img = (_Image.open(img).convert("RGB")
                   if isinstance(img, (str, Path))
                   else _Image.fromarray(img).convert("RGB"))
        samples.append((img, s[q_col]))
    return samples
