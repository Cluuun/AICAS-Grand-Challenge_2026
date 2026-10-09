"""prune_optim — reusable Method D + in-block ViT mask pruning utilities.

Designed as a drop-in for any vAxel26-derived wrapper. Other optimizations
(speculative decoding, quantization) can hard-fork the wrapper and import
this module to inherit the prune+resolution-cap behavior unchanged.

Public surface:
    PruneConfig                  — dataclass holding all knobs
    configure_processor_method_d — patch HF image_processor (longest_edge cap)
    compute_n_keep               — bucket+min_k clamp on n_keep
    select_keep_blocks           — top-k by importance, returns sorted indices
    gather_blocks_4row           — gather (n_merged*r, dim) at block granularity
    PRUNE_WARMUP_GRIDS           — list of (T,H,W) grids covering pruned envelope
    PRUNE_PREFILL_WARMUP_SEQS    — list of seq_lens covering pruned prefill envelope
"""
from .prune_module import (
    PruneConfig,
    configure_processor_method_d,
    compute_n_keep,
    select_keep_blocks,
    gather_blocks_4row,
    PRUNE_WARMUP_GRIDS,
    PRUNE_PREFILL_WARMUP_SEQS,
)

__all__ = [
    "PruneConfig",
    "configure_processor_method_d",
    "compute_n_keep",
    "select_keep_blocks",
    "gather_blocks_4row",
    "PRUNE_WARMUP_GRIDS",
    "PRUNE_PREFILL_WARMUP_SEQS",
]
