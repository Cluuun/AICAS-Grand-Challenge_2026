"""Init-time PTQ activation calibration for ViT/LLM quantization.

Compliance: AICAS allows on-the-fly calibration with TextVQA *train* split
(forbidden uploads = pre-quantized weights / offline scales as artifact).
This module runs entirely in `__init__` against
`quant_optim_workspace/textvqa_train/*.parquet` and produces in-memory
per-channel activation statistics. Nothing is persisted to disk.

Public API:
    TextVQACalibrationLoader(parquet_dir, n_samples, image_processor, device)
        — streaming loader yielding (pixel_values, image_grid_thw)
    collect_vit_fc1_activation_stats(model, loader)
        — returns List[Optional[Tensor[K]]] indexed by block id, per-input-
          channel activation absmax over all calibration samples

Usage in evaluation wrapper:
    loader = TextVQACalibrationLoader(
        "/root/aicas/quant_optim_workspace/textvqa_train",
        n_samples=32,
        image_processor=processor.image_processor,
        device=device,
    )
    stats = collect_vit_fc1_activation_stats(model, loader)
    bundle = build_vit_fc1_w8a8_bundle(model, activation_stats=stats)
"""
from __future__ import annotations

import glob
import io
import os
from typing import Iterator, List, Optional, Tuple

import torch


class TextVQACalibrationLoader:
    """Streaming loader over textvqa_train parquet shards.

    Yields (pixel_values_fp16, image_grid_thw) pairs ready to feed into
    `model.model.get_image_features` (or directly into the ViT forward).

    Lazy import of pyarrow and PIL so the module is importable without them.
    """

    def __init__(
        self,
        parquet_dir: str,
        n_samples: int,
        image_processor,
        device: str = "cuda:0",
    ):
        self.parquet_dir = parquet_dir
        self.n_samples = int(n_samples)
        self.image_processor = image_processor
        self.device = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        import pyarrow.parquet as pq
        from PIL import Image

        files = sorted(glob.glob(os.path.join(self.parquet_dir, "train-*.parquet")))
        if not files:
            raise FileNotFoundError(
                f"no train-*.parquet under {self.parquet_dir}"
            )

        count = 0
        for fpath in files:
            if count >= self.n_samples:
                return
            table = pq.read_table(fpath, columns=["image"])
            n_rows = table.num_rows
            img_col = table.column("image")
            for i in range(n_rows):
                if count >= self.n_samples:
                    return
                rec = img_col[i].as_py()
                img_bytes = rec["bytes"]
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                proc = self.image_processor([img], return_tensors="pt")
                pv = proc["pixel_values"].to(self.device, dtype=torch.float16)
                grid = proc["image_grid_thw"].to(self.device)
                yield pv, grid
                count += 1


def collect_vit_fc1_activation_stats(
    model,
    loader: TextVQACalibrationLoader,
) -> List[Optional[torch.Tensor]]:
    """Hook ViT block.mlp.linear_fc1 inputs, return per-input-channel absmax.

    Returns:
        List of length `len(visual.blocks)`. Each entry is either:
        - None (block produced no calibration input — should not happen
          with normal forward but treated defensively), or
        - Tensor of shape [K_in] containing element-wise running absmax
          across all calibration samples (FP32, on `model` device).
    """
    visual = model.model.visual if hasattr(model.model, "visual") else model.visual
    blocks = visual.blocks
    n_blocks = len(blocks)

    stats: List[Optional[torch.Tensor]] = [None] * n_blocks
    handles = []

    def _make_hook(idx: int):
        def _hook(module, inputs):
            x = inputs[0]
            x_flat = x.reshape(-1, x.shape[-1]).detach()
            cur = x_flat.abs().amax(dim=0).float()
            if stats[idx] is None:
                stats[idx] = cur.clone()
            else:
                stats[idx] = torch.maximum(stats[idx], cur)
        return _hook

    for li, blk in enumerate(blocks):
        h = blk.mlp.linear_fc1.register_forward_pre_hook(_make_hook(li))
        handles.append(h)

    try:
        with torch.no_grad():
            for pv, grid in loader:
                _ = model.model.get_image_features(
                    pixel_values=pv, image_grid_thw=grid
                )
    finally:
        for h in handles:
            h.remove()

    return stats
