from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Callable, Literal

import torch


AcceptKVSource = Literal["tree", "canonical"]
DiagnosticCallback = Callable[["Eagle3State", "VerifyOutput | None", "AcceptOutput | None"], None]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _accept_kv_source_from_env() -> AcceptKVSource:
    raw = os.getenv("AICAS_EAGLE3_ACCEPT_KV_SOURCE", "tree").strip().lower()
    if raw in {"", "tree", "selected_tree"}:
        return "tree"
    if raw == "canonical":
        return "canonical"
    raise RuntimeError(
        "EAGLE3 refactor only supports AICAS_EAGLE3_ACCEPT_KV_SOURCE="
        f"'tree' or 'canonical', got {raw!r}"
    )


@dataclass(slots=True)
class Eagle3Config:
    """Runtime knobs for EAGLE3 tree speculative decoding.

    This object is the only place where the refactored worker reads
    environment variables.  Lower-level kernels may still read their own
    feature flags, but the generation workflow is fixed to tree verification
    plus either selected-tree KV or canonical refresh KV.
    """

    accept_kv_source: AcceptKVSource = "tree"
    selected_target_forward: bool = True
    exact_eos_suppression: bool = False
    timing_detail: bool = False
    trace: bool = False
    feature_refresh_policy: str = "off"
    adaptive_fallback: bool = False
    adaptive_min_rounds: int = 4
    adaptive_min_generated: int = 8
    adaptive_min_output_per_round: float = 3.5
    checks_enabled: bool = True
    force_accept_length: int | None = None

    @classmethod
    def from_env(cls, *, force_accept_length: int | None = None) -> "Eagle3Config":
        return cls(
            accept_kv_source=_accept_kv_source_from_env(),
            selected_target_forward=_env_flag("AICAS_EAGLE3_SELECTED_TARGET_FORWARD", True),
            exact_eos_suppression=_env_flag("AICAS_EAGLE3_EXACT_EOS_SUPPRESSION", False),
            timing_detail=_env_flag("AICAS_EAGLE3_TIMING_DETAIL", False),
            trace=_env_flag("AICAS_EAGLE3_TRACE", False),
            feature_refresh_policy=os.getenv("AICAS_EAGLE3_ACCEPT_FEATURE_REFRESH", "off").strip().lower(),
            adaptive_fallback=_env_flag("AICAS_EAGLE3_ADAPTIVE_FALLBACK", False),
            adaptive_min_rounds=max(1, _env_int("AICAS_EAGLE3_ADAPTIVE_MIN_ROUNDS", 4)),
            adaptive_min_generated=max(1, _env_int("AICAS_EAGLE3_ADAPTIVE_MIN_GENERATED", 8)),
            adaptive_min_output_per_round=_env_float("AICAS_EAGLE3_ADAPTIVE_MIN_OUTPUT_PER_ROUND", 3.5),
            # checks_enabled=_env_flag("AICAS_EAGLE3_CHECKS", True),
            checks_enabled=_env_flag("AICAS_EAGLE3_CHECKS", False),
            force_accept_length=force_accept_length,
        )

    def verifier_env(self) -> dict[str, str]:
        return {
            "AICAS_EAGLE3_SELECTED_TARGET_FORWARD": "1" if self.selected_target_forward else "0",
        }


def new_eagle3_stats() -> dict[str, Any]:
    return {
        "rounds": 0,
        "accepted": 0,
        "drafted": 0,
        "fallback": 0,
        "fallback_generated": 0,
        "fallback_reason": "",
        "verify_ms": 0.0,
        "draft_ms": 0.0,
        "rollback_ms": 0.0,
        "accept0": 0,
        "root_checks": 0,
        "root_target_vocab_hits": 0,
        "root_top1_hits": 0,
        "root_topk_hits": 0,
        "raw_root_top1_hits": 0,
        "raw_root_topk_hits": 0,
        "round0_raw_root_topk_hits": 0,
        "round0_root_checks": 0,
        "later_raw_root_topk_hits": 0,
        "later_root_checks": 0,
        "rebuild_root_checks": 0,
        "rebuild_raw_root_top1_hits": 0,
        "rebuild_raw_root_topk_hits": 0,
        "verify_forward_ms": 0.0,
        "verify_lm_head_ms": 0.0,
        "verify_feature_ms": 0.0,
        "chain_recheck": 0,
        "chain_recheck_considered": 0,
        "chain_recheck_skipped": 0,
        "chain_recheck_ms": 0.0,
        "chain_refresh_count": 0,
        "chain_refresh_tokens": 0,
        "chain_refresh_next_corrections": 0,
        "accept_decode_ms": 0.0,
        "accept_decode_next_corrections": 0,
        "posterior_ms": 0.0,
        "cache_select_ms": 0.0,
        "chain_refresh_ms": 0.0,
        "draft_next_ms": 0.0,
    }


@dataclass(slots=True)
class DraftTree:
    draft_tokens: torch.Tensor
    retrieve_indices: torch.Tensor
    tree_mask: torch.Tensor
    tree_position_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "DraftTree":
        return DraftTree(
            draft_tokens=self.draft_tokens.to(device),
            retrieve_indices=self.retrieve_indices.to(device),
            tree_mask=self.tree_mask.to(device),
            tree_position_ids=self.tree_position_ids.to(device),
        )


@dataclass(slots=True)
class VerifyOutput:
    target_tokens: torch.Tensor
    feature: torch.Tensor
    hidden: torch.Tensor
    tree_past: Any
    draft_tree: DraftTree


@dataclass(slots=True)
class PathSelection:
    path_target_tokens: torch.Tensor
    candidates: torch.Tensor
    best_candidate: torch.Tensor
    accept_length: int | torch.Tensor
    next_tok: torch.Tensor


@dataclass(slots=True)
class AcceptOutput:
    accepted_tokens: torch.Tensor
    selected_nodes: torch.Tensor
    accepted_past: Any
    accept_feature: torch.Tensor
    next_tok: torch.Tensor
    sample_token: torch.Tensor
    append_count: int
    accept_length: int
    stop_after_append: bool = False
    path: PathSelection | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Eagle3State:
    config: Eagle3Config
    past_kv: Any
    feature: torch.Tensor
    next_tok: torch.Tensor
    sample_token: torch.Tensor
    token_state: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=new_eagle3_stats)
    token_chunks: list[torch.Tensor] = field(default_factory=list)
    generated_len: int = 0
    initial_input_len: int = 0
    token_chunk_debug: list[list[int]] | None = None

    @property
    def device(self) -> torch.device:
        return self.feature.device
