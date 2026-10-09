from __future__ import annotations

import os

import torch

from spec_decode.eagle3 import (
    _eagle3_tree_decoding,
    _eagle3_tree_verify_graph,
    _suppress_tree_eos_targets_before_min_length,
    _suppress_tree_eos_targets_before_min_length_exact,
    _temporary_env,
)
from spec_decode.eagle3_cache_adapter import (
    _eagle3_cache_seq_len,
)
from spec_decode.eagle3_state import DraftTree, Eagle3State, VerifyOutput

# Module-level cached env vars
_EAGLE3_TREE_VERIFY_CUDAGRAPH = os.getenv("AICAS_EAGLE3_TREE_VERIFY_CUDAGRAPH", "0") == "1"


def _verify_tree_raw(
    model,
    state: Eagle3State,
    draft_tree: DraftTree,
    current_input_ids: torch.Tensor,
    layer_indices: tuple[int, int, int],
):
    use_graph = (
        _EAGLE3_TREE_VERIFY_CUDAGRAPH
        and not state.token_state.get("disable_tree_verify_graph", False)
    )
    current_len = int(_eagle3_cache_seq_len(state.past_kv))

    with _temporary_env(state.config.verifier_env()):
        if use_graph:
            return _eagle3_tree_verify_graph(
                model,
                state.past_kv,
                draft_tree.draft_tokens.to(state.device),
                draft_tree.retrieve_indices.to(state.device),
                draft_tree.tree_mask.to(state.device),
                draft_tree.tree_position_ids.to(state.device),
                current_len,
                layer_indices,
                state.token_state,
            )
        return _eagle3_tree_decoding(
            model,
            state.past_kv,
            draft_tree.draft_tokens,
            draft_tree.retrieve_indices,
            draft_tree.tree_mask,
            draft_tree.tree_position_ids,
            current_len,
            layer_indices,
            state.token_state,
        )


def verify_tree(
    model,
    inputs: dict,
    state: Eagle3State,
    current_input_ids: torch.Tensor,
    draft_tree: DraftTree,
    *,
    layer_indices: tuple[int, int, int],
    eos_ids: set[int],
    min_new_tokens: int,
) -> VerifyOutput:
    """Run target-model verification for the current draft tree."""
    draft_tree = draft_tree.to(state.device)
    target_tokens, feature, hidden, tree_past = _verify_tree_raw(
        model,
        state,
        draft_tree,
        current_input_ids,
        layer_indices,
    )

    if state.config.exact_eos_suppression:
        target_tokens = _suppress_tree_eos_targets_before_min_length_exact(
            model,
            inputs,
            current_input_ids,
            target_tokens,
            draft_tree.draft_tokens,
            draft_tree.tree_mask,
            draft_tree.tree_position_ids,
            eos_ids,
            generated_len=int(state.generated_len),
            min_new_tokens=int(min_new_tokens),
        )
    else:
        target_tokens = _suppress_tree_eos_targets_before_min_length(
            model,
            target_tokens,
            hidden,
            draft_tree.tree_position_ids,
            eos_ids,
            generated_len=int(state.generated_len),
            min_new_tokens=int(min_new_tokens),
        )

    return VerifyOutput(
        target_tokens=target_tokens,
        feature=feature,
        hidden=hidden,
        tree_past=tree_past,
        draft_tree=draft_tree,
    )
