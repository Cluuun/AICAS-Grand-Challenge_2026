from __future__ import annotations

from typing import Any

import torch

from spec_decode.eagle3 import (
    Eagle3DraftModel,
    _profile_range,
    load_eagle3_draft,
    save_eagle3_draft,
)
from spec_decode.eagle3_state import AcceptOutput, DraftTree, Eagle3State


class Eagle3DraftTopKRunner:
    """TopK tree draft runner separated from the worker loop.

    The trained `Eagle3DraftModel` module is not changed.  This runner calls
    the existing public/protected methods with the same tensor shapes and owns
    no parameters, so existing checkpoints remain binary-compatible.
    """

    def __init__(self, draft_model: Eagle3DraftModel) -> None:
        self.draft_model = draft_model

    def generate(self, feature: torch.Tensor, input_ids: torch.Tensor, logits_processor=None) -> DraftTree:
        if not hasattr(self.draft_model, "tree_mask_init"):
            self.draft_model.init_tree()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.draft_model.topK_genrate(
            feature,
            input_ids,
            logits_processor=logits_processor,
        )
        return DraftTree(
            draft_tokens=draft_tokens,
            retrieve_indices=retrieve_indices,
            tree_mask=tree_mask,
            tree_position_ids=tree_position_ids,
        )


def get_draft_runner(state: Eagle3State, draft_model: Eagle3DraftModel) -> Eagle3DraftTopKRunner:
    runner = state.token_state.get("draft_topk_runner")
    if not isinstance(runner, Eagle3DraftTopKRunner) or runner.draft_model is not draft_model:
        runner = Eagle3DraftTopKRunner(draft_model)
        state.token_state["draft_topk_runner"] = runner
    return runner


def draft_next(
    state: Eagle3State,
    draft_model: Eagle3DraftModel,
    accept: AcceptOutput,
    draft_input_ids: torch.Tensor,
    *,
    logits_processor: Any = None,
) -> DraftTree:
    """Generate the next draft tree from the accepted feature and next token."""
    with _profile_range("eagle3_tree.draft_next"):
        runner = get_draft_runner(state, draft_model)
        draft_tree = runner.generate(
            accept.accept_feature,
            draft_input_ids,
            logits_processor=logits_processor,
        )
    return draft_tree


__all__ = [
    "Eagle3DraftModel",
    "Eagle3DraftTopKRunner",
    "DraftTree",
    "draft_next",
    "get_draft_runner",
    "load_eagle3_draft",
    "save_eagle3_draft",
]
