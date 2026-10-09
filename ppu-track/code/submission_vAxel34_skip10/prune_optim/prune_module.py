"""prune_module — Method D + in-block ViT mask pruning core utilities.

Reference implementation for the optimizations on top of vAxel26 that produced
official AICAS 2026 Total = 2.0085 (P 0.6978 / TTFT 23.71 / Thr 308.30).

Two independent levers:

1. Method D (Resolution-Adaptive max_pixels cap) — set via
   `configure_processor_method_d(image_processor, max_pixels=327680)`.
   This caps the input pixel area before ViT, shrinking N_patches.

2. In-block ViT mask prune — applied inside the manual ViT forward after
   layer 5 (where ds_feat_0 is captured). The wrapper calls
   `compute_n_keep`, `select_keep_blocks`, `gather_blocks_4row` to drop
   `prune_ratio` of merged blocks ranked by ds_feat_0 L2-norm. Block
   granularity preserves the merger's `view(N,4,d).mean(dim=1)` invariant.

CRITICAL: `n_keep` MUST be bucketed (compute_n_keep does this when
config.bucket > 1). Without bucketing, every grid produces a unique
post-prune seq length and torch.compile(dynamic=True) pays a per-shape
recompile tax on the official harness — observed −0.89 score on AICAS v1
submission. With bucket=32, n_keep collapses to ~7 unique values, and
the warmup envelopes below cover all of them.
"""
from dataclasses import dataclass

import torch


@dataclass
class PruneConfig:
    """All knobs in one place. Pass to compute_n_keep / select_keep_blocks etc."""
    # In-block ViT mask prune
    enabled: bool = True              # master switch; if False, disables prune (Method D still applies)
    prune_ratio: float = 0.7          # fraction of merged blocks to drop
    min_k: int = 64                   # floor on n_keep (protects small grids)
    bucket: int = 32                  # snap n_keep up to multiples; 1 disables bucketing
    # Method D — resolution-adaptive
    max_pixels: int = 327680          # image_processor.size["longest_edge"]
    # Compatibility with manual ViT
    rows_per_block: int = 4           # spatial_merge_size**2 for Qwen3-VL


# ─── Method D ────────────────────────────────────────────────────────────

def configure_processor_method_d(image_processor, max_pixels: int) -> None:
    """Patch a HF image_processor to cap the input pixel area.

    Modifies `image_processor.size["longest_edge"]` in place. Also bumps
    `image_processor.max_pixels` if that attribute exists. Idempotent.

    Args:
        image_processor: an HF AutoImageProcessor (e.g. from AutoProcessor)
        max_pixels: target longest-edge pixel area (e.g. 327680)
    """
    if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
        size = dict(image_processor.size)
        prev = size.get("longest_edge")
        size["longest_edge"] = max_pixels
        image_processor.size = size
        print(f"[prune_optim] Method D: longest_edge {prev} -> {max_pixels}")
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels


# ─── In-block ViT mask prune primitives ──────────────────────────────────

def compute_n_keep(n_merged: int, config: PruneConfig) -> int:
    """Decide n_keep for a given merged-block count.

    Pipeline: raw = max(round(n_merged * (1-r)), min_k);
              bucketed = ceil(raw / bucket) * bucket; clamp to n_merged.

    Returns n_merged (no-op) when prune is disabled or n_keep would equal n_merged.
    """
    if (not config.enabled) or config.prune_ratio <= 0.0:
        return n_merged
    raw = max(int(round(n_merged * (1.0 - config.prune_ratio))), config.min_k)
    if config.bucket > 1:
        bucketed = ((raw + config.bucket - 1) // config.bucket) * config.bucket
    else:
        bucketed = raw
    return min(bucketed, n_merged)


def select_keep_blocks(importance: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Top-k blocks by importance (e.g. ds_feat_0.norm(dim=-1)), returned sorted.

    Sorting preserves spatial locality so M-RoPE / DeepStack scatter remain
    monotonic in the kept indices.
    """
    keep_blocks = torch.topk(importance, n_keep, largest=True, sorted=False).indices
    keep_blocks, _ = torch.sort(keep_blocks)
    return keep_blocks


def gather_blocks_4row(tensor_2d: torch.Tensor,
                       keep_blocks: torch.Tensor,
                       rows_per_block: int = 4) -> torch.Tensor:
    """Gather a (n_merged*rows_per_block, dim) tensor at block granularity.

    Used for hidden_states, cos_pe, sin_pe — anything in row-major patch order
    that needs to stay aligned with the merger's `view(N,4,d).mean(dim=1)`.

    Output shape: (n_keep * rows_per_block, dim)
    """
    n_rows, dim = tensor_2d.shape
    n_merged = n_rows // rows_per_block
    view = tensor_2d.view(n_merged, rows_per_block, dim)
    n_keep = keep_blocks.shape[0]
    return view.index_select(0, keep_blocks).reshape(n_keep * rows_per_block, dim)


# ─── Warmup envelopes ────────────────────────────────────────────────────
# These cover the post-prune ViT and prefill shape distributions observed
# across mp=327680 + r=0.7 + bucket=32. Wrappers should iterate these during
# init to seed the dynamic-shape compile cache before the first measured
# sample. Without these warmups, the official harness pays per-shape
# autotuning costs (see AICAS v1 regression: TTFT +24 ms, Thr -135 tok/s).

PRUNE_WARMUP_GRIDS = [
    # Small/medium grids — typical for mp=327680
    (1, 24, 32),  # merged 192 -> n_keep 64
    (1, 28, 36),  # merged 252 -> n_keep 96
    (1, 30, 40),  # merged 300 -> n_keep 96
    (1, 28, 42),  # merged 294 -> n_keep 96
    (1, 34, 36),  # merged 306 -> n_keep 96
    (1, 46, 26),  # merged 299 -> n_keep 96
    # Larger grids (rare but possible on full-resolution paths)
    (1, 36, 48),  # merged 432 -> n_keep 160
    (1, 40, 48),  # merged 480 -> n_keep 160
    (1, 48, 64),  # merged 768 -> n_keep 256
    (1, 42, 64),  # merged 672 -> n_keep 224
    (1, 32, 56),  # merged 448 -> n_keep 160
    (1, 26, 50),  # merged 325 -> n_keep 128
]

PRUNE_PREFILL_WARMUP_SEQS = [128, 160, 192, 224, 256, 288, 320, 384]
