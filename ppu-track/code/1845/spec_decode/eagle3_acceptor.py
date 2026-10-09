from __future__ import annotations

import os
import time
from typing import Any

import torch

from spec_decode.eagle3 import (
    _canonical_decode_step,
    _decode_position_kwargs,
    _accepted_tokens_from_path_plan,
    _evaluate_tree_posterior_fast,
    _extract_past,
    _profile_range,
    _refresh_accepted_chain,
    _select_tree_cache_path,
    _target_to_list,
)
from spec_decode.eagle3_cache_adapter import (
    _clone_cache_to_len,
    _eagle3_cache_seq_len,
    _set_eagle3_cache_seq_len,
)
from spec_decode.eagle3_state import AcceptOutput, Eagle3State, PathSelection, VerifyOutput
from spec_decode.eagle3_cache_adapter import clear_static_cache_tail


_PERF_TIMING_ENABLED: bool | None = None


def _perf_timing_enabled() -> bool:
    global _PERF_TIMING_ENABLED
    if _PERF_TIMING_ENABLED is None:
        _PERF_TIMING_ENABLED = os.getenv("AICAS_EAGLE3_PERF_STATS", "0") == "1"
    return _PERF_TIMING_ENABLED


def select_best_path(verify: VerifyOutput, token_state: dict[str, Any] | None = None) -> PathSelection:
    """Choose the greedy path with the longest accepted prefix."""

    path_target_tokens, candidates, best_candidate, accept_length, next_tok = _evaluate_tree_posterior_fast(
        verify.target_tokens,
        verify.draft_tree.draft_tokens,
        verify.draft_tree.retrieve_indices,
        token_state if isinstance(token_state, dict) else {},
    )
    return PathSelection(
        path_target_tokens=path_target_tokens,
        candidates=candidates,
        best_candidate=best_candidate,
        accept_length=accept_length,
        next_tok=next_tok,
    )


class CanonicalRefreshRunner:
    """Standard decode refresh for accepted tokens.

    This mirrors SGLang's split between accepting a path and running a
    fixed-shape correction pass.  It intentionally does not own model
    weights or alter the draft model ABI.
    """

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def refresh(self, model, base_past, accepted_tokens: torch.Tensor, base_cache_len: int):
        if _canonical_decode_step is None:
            raise RuntimeError("canonical decode helper unavailable")
        tokens = accepted_tokens.reshape(1, -1).to(dtype=torch.long)
        kwargs = dict(
            input_ids=tokens,
            past_key_values=base_past,
            use_cache=True,
            return_dict=False,
            output_hidden_states=False,
            output_attentions=False,
            logits_to_keep=1,
            **_decode_position_kwargs(model, base_past, tokens),
        )
        with torch.no_grad():
            next_token, outputs = _canonical_decode_step(model, kwargs, self.state)
        refreshed_past = _extract_past(outputs) if outputs is not None else base_past
        if refreshed_past is None:
            refreshed_past = base_past
        _set_eagle3_cache_seq_len(refreshed_past, int(base_cache_len) + int(tokens.shape[1]))
        return next_token.reshape(-1).to(dtype=torch.long), refreshed_past


def _forced_path(path: PathSelection, force_accept_length: int | None) -> PathSelection:
    if force_accept_length is None:
        return path
    accept_length = max(0, min(int(force_accept_length), int(path.candidates.shape[1]) - 1))
    if accept_length == 0:
        best_candidate = torch.tensor(0, dtype=torch.long, device=path.candidates.device)
    else:
        best_candidate = path.best_candidate
    next_tok = path.path_target_tokens[best_candidate, accept_length].reshape(())
    return PathSelection(
        path_target_tokens=path.path_target_tokens,
        candidates=path.candidates,
        best_candidate=best_candidate,
        accept_length=accept_length,
        next_tok=next_tok,
    )


def _accepted_tokens_for_path(
    path: PathSelection,
    *,
    remaining_tokens: int,
    eos_ids: set[int],
    generated_len: int,
    min_new_tokens: int,
) -> tuple[torch.Tensor, int, int, bool, int]:
    return _accepted_tokens_from_path_plan(
        path.candidates,
        path.best_candidate,
        path.accept_length,
        path.next_tok,
        remaining_tokens=int(remaining_tokens),
        generated_len=int(generated_len),
        min_new_tokens=int(min_new_tokens),
        eos_ids=eos_ids,
        skip_root=True,
    )


def canonical_refresh(
    model,
    state: Eagle3State,
    base_past,
    accepted_tokens: torch.Tensor,
    *,
    base_cache_len: int,
) -> tuple[torch.Tensor, Any]:
    """Refresh accepted KV with the target model's canonical decode path."""

    base = _clone_cache_to_len(base_past, int(base_cache_len))
    clear_static_cache_tail(base, int(base_cache_len))
    _set_eagle3_cache_seq_len(base, int(base_cache_len))
    runner = state.token_state.get("canonical_refresh_runner")
    if not isinstance(runner, CanonicalRefreshRunner):
        runner = CanonicalRefreshRunner()
        state.token_state["canonical_refresh_runner"] = runner
    return runner.refresh(model, base, accepted_tokens, int(base_cache_len))


def _maybe_chain_refresh(
    model,
    state: Eagle3State,
    selected_past,
    accepted_tokens: torch.Tensor,
    accept_feature: torch.Tensor,
    sample_token: torch.Tensor,
    *,
    base_cache_len: int,
    append_count: int,
    layer_indices: tuple[int, int, int],
) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    policy = state.config.feature_refresh_policy
    if policy in {"0", "false", "no", "off", "none", "tree"}:
        return selected_past, accept_feature, sample_token, None
    if policy in {"1", "true", "yes", "chain", "always", "all"}:
        use_refresh = True
        reason = "chain"
    elif policy in {"accept0", "auto"}:
        use_refresh = int(append_count) == 1
        reason = "accept0" if use_refresh else "skip_accept"
    else:
        return selected_past, accept_feature, sample_token, {"reason": f"unknown:{policy}"}
    if not use_refresh:
        return selected_past, accept_feature, sample_token, {"reason": reason}

    state.stats["chain_refresh_count"] = state.stats.get("chain_refresh_count", 0) + 1
    state.stats["chain_refresh_tokens"] = state.stats.get("chain_refresh_tokens", 0) + int(append_count)
    with _profile_range("eagle3_tree.accept_chain_refresh"):
        base_for_chain = _clone_cache_to_len(selected_past, int(base_cache_len))
        chain_target, chain_past, chain_feature, _ = _refresh_accepted_chain(
            model,
            base_for_chain,
            accepted_tokens.reshape(-1),
            state.token_state,
            layer_indices,
        )
    _set_eagle3_cache_seq_len(chain_past, int(base_cache_len) + int(append_count))

    chain_flat = chain_target.reshape(-1).to(device=accepted_tokens.device, dtype=torch.long)
    if int(chain_flat.numel()) >= int(append_count) and int(append_count) > 0:
        chain_next = chain_flat[int(append_count) - 1].reshape(())
        old_next = sample_token.reshape(()).to(device=chain_next.device, dtype=chain_next.dtype)
        if not bool(chain_next.eq(old_next).item()):
            state.stats["chain_refresh_next_corrections"] = state.stats.get("chain_refresh_next_corrections", 0) + 1
            sample_token = chain_next.reshape(1, 1).to(device=accepted_tokens.device, dtype=torch.long)
    else:
        chain_next = None

    diag = {
        "reason": reason,
        "corrected_next": int(chain_next.item()) if isinstance(chain_next, torch.Tensor) else None,
    }
    return chain_past, chain_feature, sample_token, diag


def accept_path(
    model,
    draft_model,
    state: Eagle3State,
    current_input_ids: torch.Tensor,
    verify: VerifyOutput,
    path: PathSelection,
    *,
    base_past,
    base_cache_len: int,
    max_new_tokens: int,
    eos_ids: set[int],
    min_new_tokens: int,
    layer_indices: tuple[int, int, int],
) -> tuple[AcceptOutput, torch.Tensor]:
    """Select accepted tokens, update accepted KV, and prepare draft input."""

    if int(state.stats['rounds']) == 7:
        pass

    path = _forced_path(path, state.config.force_accept_length)
    remaining = int(max_new_tokens) - int(state.generated_len)
    accepted_tokens, append_count, accept_length, stop_after_append, tree_node_count = _accepted_tokens_for_path(
        path,
        remaining_tokens=remaining,
        eos_ids=eos_ids,
        generated_len=int(state.generated_len),
        min_new_tokens=int(min_new_tokens),
    )
    if int(append_count) <= 0:
        raise RuntimeError("EAGLE3 accept_path produced no token to append")
    # if state.config.checks_enabled and int(accepted_tokens.min().item()) < 0:
    #     raise RuntimeError(
    #         "EAGLE3 tree selected an invalid padded token; "
    #         f"candidate={_target_to_list(path.candidates[path.best_candidate])} accept_length={accept_length}"
    #     )

    t_cache_select = time.perf_counter() if _perf_timing_enabled() else None
    try:
        with _profile_range("eagle3_tree.cache_select"):
            if int(tree_node_count) > 0:
                selected_nodes = verify.draft_tree.retrieve_indices[
                    path.best_candidate, :int(tree_node_count)
                ].to(device=state.device, dtype=torch.long).contiguous()
                selected_past = _select_tree_cache_path(verify.tree_past, int(base_cache_len), selected_nodes)
                if selected_past is None:
                    raise RuntimeError("EAGLE3 tree verifier could not select accepted KV path")
                accept_feature = verify.feature[:, selected_nodes, :].contiguous()
            else:
                selected_nodes = torch.empty((0,), device=state.device, dtype=torch.long)
                selected_past = _clone_cache_to_len(state.past_kv, int(base_cache_len))
                accept_feature = verify.feature[:, :0, :].contiguous()
    finally:
        if t_cache_select is not None:
            state.stats["cache_select_ms"] = state.stats.get("cache_select_ms", 0.0) + (
                time.perf_counter() - t_cache_select
            ) * 1000.0

    next_tok = path.next_tok.reshape(()).to(device=state.device, dtype=torch.long)
    sample_token = next_tok.reshape(1, 1)
    prefix_tokens = path.candidates[path.best_candidate, :tree_node_count].to(
        device=state.device,
        dtype=torch.long,
    ).contiguous()
    accepted_past = selected_past
    # Zero tail beyond accepted prefix so get_seq_length() returns correct value
    cleared_len = int(base_cache_len) + int(tree_node_count)
    if int(getattr(accepted_past, "_aicas_eagle3_tail_cleared_len", -1)) != cleared_len:
        clear_static_cache_tail(accepted_past, cleared_len)
    try:
        accepted_past._aicas_eagle3_tail_cleared_len = cleared_len
    except Exception:
        pass
    diagnostics: dict[str, Any] = {}

    # Root correction: single-query FlashDecode may differ from multi-query tree verify.
    # Run canonical decode on the last known correct token to fix next_tok without changing KV.
    # if (

    if state.config.accept_kv_source == "canonical":
        if base_past is None:
            base_past = _clone_cache_to_len(state.past_kv, int(base_cache_len))
        decode_next, accepted_past = canonical_refresh(
            model,
            state,
            base_past,
            accepted_tokens.reshape(-1).to(device=state.device, dtype=torch.long),
            base_cache_len=int(base_cache_len),
        )
        decode_next_scalar = decode_next.reshape(-1)[-1].reshape(())
        if not bool(decode_next_scalar.eq(sample_token.reshape(())).item()):
            state.stats["accept_decode_next_corrections"] = state.stats.get("accept_decode_next_corrections", 0) + 1
            next_tok = decode_next_scalar.to(device=state.device, dtype=torch.long)
            sample_token = next_tok.reshape(1, 1)
            accepted_tokens.reshape(-1)[-1].copy_(next_tok.to(device=accepted_tokens.device, dtype=accepted_tokens.dtype))
        diagnostics["canonical_next"] = int(decode_next_scalar.item())

    accepted_past, accept_feature, sample_token, chain_diag = _maybe_chain_refresh(
        model,
        state,
        accepted_past,
        accepted_tokens,
        accept_feature,
        sample_token,
        base_cache_len=int(base_cache_len),
        append_count=int(append_count),
        layer_indices=layer_indices,
    )
    if chain_diag is not None:
        diagnostics["chain_refresh"] = chain_diag
        next_tok = sample_token.reshape(()).to(device=state.device, dtype=torch.long)

    current_after = torch.cat((current_input_ids, prefix_tokens.reshape(1, -1)), dim=1)
    draft_input_ids = torch.cat((current_after, sample_token.to(device=state.device, dtype=torch.long)), dim=1)

    # ── DEBUG: check accept_path output for invalid ids ──
    # _dbg_vs = getattr(getattr(draft_model, "embed_tokens", None), "num_embeddings", None)
    # if isinstance(_dbg_vs, int) and int(_dbg_vs) > 0:
    #     _dbg_draft_cpu = draft_input_ids.detach().to(device="cpu", dtype=torch.long)
    #     _dbg_bad = (_dbg_draft_cpu < 0) | (_dbg_draft_cpu >= int(_dbg_vs))
    #     if bool(_dbg_bad.any().item()):
    #         raise RuntimeError(
    #             "EAGLE3 CORRUPTION at accept_path draft_input_ids construction: "
    #             f"round={int(state.stats.get('rounds', 0))} "
    #             f"generated={int(state.generated_len)} "
    #             f"draft_input_ids=({_draft_input_ids_debug_desc(draft_input_ids, int(_dbg_vs))}) "
    #             f"state_current=({_draft_input_ids_debug_desc(current_input_ids.detach().to(device='cpu', dtype=torch.long), int(_dbg_vs))}) "
    #             f"prefix=({_draft_input_ids_debug_desc(prefix_tokens.detach().to(device='cpu', dtype=torch.long), int(_dbg_vs))}) "
    #             f"sample=({_draft_input_ids_debug_desc(sample_token.detach().to(device='cpu', dtype=torch.long), int(_dbg_vs))})"
    #         )

    _set_eagle3_cache_seq_len(accepted_past, int(base_cache_len) + int(tree_node_count))
    return (
        AcceptOutput(
            accepted_tokens=accepted_tokens,
            selected_nodes=selected_nodes,
            accepted_past=accepted_past,
            accept_feature=accept_feature,
            next_tok=next_tok,
            sample_token=sample_token.to(device=state.device, dtype=torch.long),
            append_count=int(append_count),
            accept_length=int(accept_length),
            stop_after_append=bool(stop_after_append),
            path=path,
            diagnostics=diagnostics,
        ),
        draft_input_ids,
    )
