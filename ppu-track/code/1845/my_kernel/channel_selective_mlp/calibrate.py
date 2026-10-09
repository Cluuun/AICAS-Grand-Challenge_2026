#!/usr/bin/env python3
"""Calibrate per-layer channel importance for the Channel-Selective MLP operator.

Hooks ``silu(gate_proj(x)) * up_proj(x)`` of every text-decoder MLP layer and
accumulates per-channel importance over a small calibration set drawn from the
benchmark dataset.  Saves a dict mapping ``layer_index -> top-K channel
indices`` to a ``.pt`` file consumed at startup by
``my_kernel/channel_selective_mlp/runtime.py``.

Usage:
    python my_kernel/channel_selective_mlp/calibrate.py \
        --model-path /root/autodl-fs/aicas26/QWen3-VL-2B-Instruct \
        --dataset-path /root/aicas26-lhd/baseline/data \
        --output mlp_channel_indices.pt \
        --num-samples 64 \
        --keep-ratio 0.85

Importance metric: mean of ``|silu(gate)*up|`` across all tokens in the
calibration sequences (prefill + a few decode steps).  Channels with the
largest mean activation magnitude are kept and consumed by the operator at
runtime.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _attach_hooks(layers):
    """Wrap each layer's MLP forward to record |silu(gate)*up| per channel.

    Note: we wrap ``forward`` rather than using a hook because we need the
    intermediate activation before ``down_proj``, which is not exposed by a
    standard module hook on the MLP.
    """
    importance = {}
    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "gate_proj"):
            continue
        importance[i] = None

        orig_forward = mlp.forward

        def _wrapped(x, *, _mlp=mlp, _idx=i, _orig=orig_forward):
            with torch.no_grad():
                g = _mlp.gate_proj(x)
                u = _mlp.up_proj(x)
                act = torch.nn.functional.silu(g) * u
                flat = act.detach().reshape(-1, act.shape[-1]).abs()
                score = flat.mean(dim=0).float().cpu()
            cur = importance[_idx]
            importance[_idx] = score if cur is None else (cur + score)
            return _orig(x)

        mlp._calib_orig_forward = orig_forward
        mlp.forward = _wrapped
    return importance, []


def _detach_hooks(layers, _handles):
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "_calib_orig_forward"):
            mlp.forward = mlp._calib_orig_forward
            del mlp._calib_orig_forward


def _resolve_layers(model):
    candidates = [
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(model, "model", None),
    ]
    for c in candidates:
        if c is not None and hasattr(c, "layers"):
            return list(c.layers)
    raise RuntimeError("Cannot locate language-model layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--max-decode-tokens", type=int, default=8)
    ap.add_argument("--keep-ratio", type=float, default=0.85)
    ap.add_argument(
        "--metric",
        choices=("activation", "output_norm"),
        default="output_norm",
        help="Channel ranking metric. output_norm weights activation by down_proj column norm.",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["AICAS_MLP_PRUNE_INDICES_PATH"] = ""
    os.environ["AICAS_LM_HEAD_VOCAB_SIZE"] = "0"
    os.environ["AICAS_DECODE_GRAPH_SANITY_VALIDATE_MAX_NEW_TOKENS"] = "0"
    os.environ["AICAS_DISABLE_DECODE_CUDAGRAPH"] = "1"
    os.environ["AICAS_DISABLE_VISION_GRAPH_REPLAY"] = "1"
    os.environ["AICAS_VISION_ENABLE_COMPILE"] = "0"

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from datasets import load_from_disk

    print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Loading model (BF16) ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    n = min(args.num_samples, len(dataset))
    print(f"Calibrating on {n} samples (decode tokens={args.max_decode_tokens})")

    layers = _resolve_layers(model)
    print(f"Found {len(layers)} text-decoder layers")

    importance, handles = _attach_hooks(layers)

    try:
        for i in range(n):
            sample = dataset[i]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["question"]},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(args.device)
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=args.max_decode_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            if (i + 1) % 8 == 0:
                print(f"  calibrated {i + 1}/{n} samples")
    finally:
        _detach_hooks(layers, handles)

    keep_indices = {}
    sample_layer = next(iter(layers))
    inter = int(sample_layer.mlp.gate_proj.weight.shape[0])
    keep_n_target = int(round(inter * args.keep_ratio))
    keep_n_target = max(16, min(inter, keep_n_target))
    print(f"\nIntermediate size: {inter}, keeping top-{keep_n_target} channels per layer "
          f"({100.0 * keep_n_target / inter:.1f}%)")

    for i in sorted(importance.keys()):
        score = importance[i]
        if score is None:
            print(f"  layer {i}: NO calibration data, keeping all channels")
            keep_indices[i] = torch.arange(inter, dtype=torch.long)
            continue
        if args.metric == "output_norm":
            down = getattr(getattr(layers[i], "mlp", None), "down_proj", None)
            weight = getattr(down, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                score = score * weight.detach().float().norm(dim=0).cpu()
        # Keep indices ranked by descending calibration importance.  The
        # runtime may later truncate this list for a smaller max_keep ratio;
        # sorting by channel id here would turn that truncation into an
        # arbitrary low-index channel slice rather than top-importance prune.
        _, idx = torch.topk(score, k=keep_n_target, largest=True, sorted=True)
        keep_indices[i] = idx.long()
        kept_score = score.index_select(0, idx).sum().item()
        total_score = score.sum().item()
        retention = kept_score / max(1e-9, total_score)
        print(f"  layer {i:>2}: kept {idx.numel():>5} / {inter} channels, "
              f"importance retention = {retention*100:.2f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "keep_indices": keep_indices,
        "intermediate_size": inter,
        "keep_ratio": args.keep_ratio,
        "num_samples": n,
        "metric": args.metric,
        "ordered_by_importance": True,
    }
    torch.save(blob, out_path)
    print(f"\nSaved calibration indices to {out_path}")


def calibrate_channel_importance():
    """Programmatic alias for the CLI entry-point."""
    return main()


if __name__ == "__main__":
    main()
