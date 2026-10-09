from __future__ import annotations

import contextlib
import os
import time
from typing import Any

import torch

from spec_decode.eagle3 import (
    _argmax_with_min_new_tokens,
    _eos_id_set,
    _eagle3_full_graph_generate_verify,
    _parse_layer_indices,
    _prefill_with_full_feature,
    _prefill_with_prefix_cache,
    _profile_range,
    _suppress_tree_eos_targets_before_min_length,
    _suppress_tree_eos_targets_before_min_length_exact,
    _target_to_list,
)
from spec_decode.eagle3_cache_adapter import (
    _eagle3_cache_seq_len,
    _set_eagle3_cache_seq_len,
)
from spec_decode.eagle3_acceptor import accept_path, select_best_path
from spec_decode.eagle3_draft import Eagle3DraftModel, Eagle3DraftTopKRunner, draft_next
from spec_decode.eagle3_state import (
    AcceptOutput,
    DraftTree,
    Eagle3Config,
    Eagle3State,
    VerifyOutput,
)
from spec_decode.eagle3_verifier import verify_tree

# ── unified perf timing (single env var gate) ──

_PERF_TIMING_ENABLED: bool | None = None


def _perf_timing_enabled() -> bool:
    global _PERF_TIMING_ENABLED
    if _PERF_TIMING_ENABLED is None:
        _PERF_TIMING_ENABLED = os.getenv("AICAS_EAGLE3_PERF_STATS", "0") == "1"
    return _PERF_TIMING_ENABLED


@contextlib.contextmanager
def _timed_stage(stats: dict, name: str):
    """Non-invasive timing gate.  Zero overhead when perf stats are disabled."""
    if not _perf_timing_enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - t0) * 1000.0
        key = f"{name}_ms"
        stats[key] = stats.get(key, 0.0) + elapsed


# Module-level cached env vars (read once at import time)
_EAGLE3_FULL_GRAPH = os.getenv("AICAS_EAGLE3_FULL_GRAPH", "0") == "1"


def prepare_initial_state(
    model,
    inputs: dict,
    draft_model: Eagle3DraftModel,
    *,
    config: Eagle3Config,
    layer_indices: tuple[int, int, int],
    eos_ids: set[int],
    min_new_tokens: int,
    reference_tokens=None,
) -> tuple[Eagle3State, torch.Tensor, DraftTree, torch.Tensor | None]:
    device = next(model.parameters()).device
    if not hasattr(draft_model, "tree_mask_init"):
        draft_model.init_tree()
    draft_model.reset_kv()

    kv_cache = getattr(model, '_aicas_kv_cache', None)
    force_full = os.getenv("AICAS_KV_PREFIX_CACHE_FORCE_FULL", "0") == "1"

    with _profile_range("eagle3_tree.prefill_target"):
        prefill_out, feature = (
            _prefill_with_prefix_cache(model, inputs, layer_indices, kv_cache=kv_cache)
            if kv_cache is not None and not force_full
            else _prefill_with_full_feature(model, inputs, layer_indices)
        )
    past_kv = prefill_out.past_key_values
    current_input_ids = inputs["input_ids"].to(device=device, dtype=torch.long)
    _set_eagle3_cache_seq_len(past_kv, int(current_input_ids.shape[1]))

    if kv_cache is not None and os.getenv("AICAS_KV_CACHE_DEBUG", "0") == "1":
        kv_cache.print_stats("[kv-cache]")

    next_tok = _argmax_with_min_new_tokens(
        prefill_out.logits[:, -1:, :],
        eos_ids,
        generated_len=0,
        min_new_tokens=int(min_new_tokens),
    )
    sample_token = torch.tensor([[int(next_tok)]], device=device, dtype=torch.long)
    draft_input_ids = torch.cat((current_input_ids, sample_token), dim=1)
    runner = Eagle3DraftTopKRunner(draft_model)
    with _profile_range("eagle3_tree.initial_draft"):
        draft_tree = runner.generate(feature, draft_input_ids, logits_processor=None)

    token_chunks = [sample_token.reshape(-1).detach().clone()]
    token_chunk_debug = None

    state = Eagle3State(
        config=config,
        past_kv=past_kv,
        feature=feature,
        next_tok=torch.as_tensor(next_tok, device=device, dtype=torch.long).reshape(()),
        sample_token=sample_token,
        token_state={"draft_topk_runner": runner},
        token_chunks=token_chunks,
        generated_len=1,
        initial_input_len=int(current_input_ids.shape[1]),
        token_chunk_debug=token_chunk_debug,
    )
    state.token_state["draft_vocab_size"] = int(draft_model.embed_tokens.num_embeddings)
    state.token_state["eagle3_initial_target_cache_len"] = int(current_input_ids.shape[1])
    try:
        stable_pair = draft_model._draft_step_graph_past_pair(getattr(draft_model, "_stable_kv", None))
        if stable_pair is not None:
            state.token_state["eagle3_initial_draft_cache_len"] = int(stable_pair[0].shape[2])
    except Exception:
        pass
    reference_tensor = None
    return state, current_input_ids, draft_tree, reference_tensor


def update_state(
    state: Eagle3State,
    accept: AcceptOutput,
    draft_input_ids: torch.Tensor,
    *,
    generated_before: int | None = None,
    base_cache_len: int | None = None,
) -> torch.Tensor:
    current_input_ids = draft_input_ids[:, :-1].contiguous()
    state.past_kv = accept.accepted_past
    state.feature = accept.accept_feature
    state.next_tok = accept.next_tok.reshape(()).to(device=state.device, dtype=torch.long)
    state.sample_token = accept.sample_token
    return current_input_ids


def _should_adaptive_fallback(state: Eagle3State, max_new_tokens: int, debug: bool) -> bool:
    cfg = state.config
    if (
        not cfg.adaptive_fallback
        or int(state.generated_len) >= int(max_new_tokens)
        or int(state.stats["rounds"]) < int(cfg.adaptive_min_rounds)
        or int(state.generated_len) < int(cfg.adaptive_min_generated)
    ):
        return False
    output_per_round = float(state.generated_len) / max(1, int(state.stats["rounds"]))
    if output_per_round >= float(cfg.adaptive_min_output_per_round):
        return False
    state.stats["fallback"] = 1
    state.stats["fallback_generated"] = int(state.generated_len)
    state.stats["fallback_reason"] = (
        f"output_per_round={output_per_round:.2f} < {cfg.adaptive_min_output_per_round:.2f}"
    )
    if debug:
        print(
            f"[eagle3-tree] adaptive fallback after {state.generated_len} tokens: "
            f"{state.stats['fallback_reason']}"
        )
    return True


# ── graph counter helpers ──

def _flush_graph_counters_to_stats(state: Eagle3State) -> dict[str, int]:
    from spec_decode.eagle3 import (
        _graph_counter_snapshot,
        _graph_counter_reset,
        _graph_reason_snapshot,
        _graph_reason_reset,
    )

    graph = _graph_counter_snapshot()
    for k, v in graph.items():
        state.stats[f"graph_{k}"] = v
    reasons = _graph_reason_snapshot()
    _graph_counter_reset()
    _graph_reason_reset()
    if reasons:
        graph["_reasons"] = reasons
    return graph


def _print_debug_summary(state: Eagle3State, graph: dict[str, int] | None = None) -> None:
    rounds = max(int(state.stats["rounds"]), 1)
    stats = state.stats

    # ── core metrics ──
    avg_accept = stats["accepted"] / rounds
    parts = [
        f"rounds={rounds}",
        f"avg_accept={avg_accept:.2f}",
        f"avg_accept_len={avg_accept + 1.0:.2f}",
        f"accept0={stats['accept0']}/{rounds}",
        f"generated={int(state.generated_len)}",
    ]

    # ── per-round stage timing (all major segments) ──
    timing_keys = [
        ("clone_cache_ms",     "clone_cache"),
        ("verify_ms",          "verify"),
        ("posterior_ms",       "posterior"),
        ("accept_ms",          "accept"),
        ("accept_output_ms",   "accept_out"),
        ("cache_select_ms",    "cache_select"),
        ("draft_ms",           "draft"),
        ("update_state_ms",    "update"),
    ]
    timing_parts = []
    for key, label in timing_keys:
        val = stats.get(key, 0.0)
        if val > 0:
            timing_parts.append(f"{label}={val / rounds:.1f}ms")
    if timing_parts:
        parts.append("timing(" + " ".join(timing_parts) + ")")

    # ── graph hit/miss (fused full_graph is the only graph path) ──
    graph = graph or {}
    if graph:
        full_hit = graph.get("full_graph_hit", 0)
        full_capture = graph.get("full_graph_capture", 0)
        if full_hit or full_capture:
            parts.append(f"full_graph={full_hit}")
        if full_capture:
            parts.append(f"full_capture={full_capture}")
        full_miss_parts = []
        for k in ("not_eligible", "no_draft_past", "capture_failed", "failed_key", "miss_after_precapture", "replay_fail"):
            v = graph.get(f"full_graph_{k}", 0)
            if v:
                full_miss_parts.append(f"{k}={v}")
        if full_miss_parts:
            parts.append(f"full_miss({','.join(full_miss_parts)})")
            reasons = graph.get("_reasons")
            if isinstance(reasons, dict):
                reason = reasons.get("full_graph")
                if reason:
                    parts.append(f"full_fail={reason}")
        compact_hit = graph.get("cache_select_compact_static_hit", 0) + graph.get("cache_select_compact_static_fused_hit", 0)
        compact_fallback = graph.get("cache_select_compact_static_fallback", 0)
        if compact_hit or compact_fallback:
            parts.append(f"compact_select={compact_hit}/{compact_hit + compact_fallback}")
        if full_capture:
            parts.append(f"captures(full={full_capture})")

    # ── detail breakdown (only with timing_detail) ──
    if state.config.timing_detail and _perf_timing_enabled():
        detail_keys = [
            ("verify_forward_ms",    "vfwd"),
            ("verify_lm_head_ms",    "vhead"),
            ("chain_refresh_ms",     "chain_refresh"),
            ("accept_decode_ms",     "accept_decode"),
        ]
        detail_parts = []
        for key, label in detail_keys:
            val = stats.get(key, 0.0)
            if val > 0:
                detail_parts.append(f"{label}={val / rounds:.1f}ms")
        if detail_parts:
            parts.append("detail(" + " ".join(detail_parts) + ")")

    print("[eagle3] " + " ".join(parts))


# ── main entry point ──

def eagle3_generate(
    model,
    inputs: dict,
    draft_model: Eagle3DraftModel,
    *,
    max_new_tokens: int,
    debug: bool = False,
    min_new_tokens: int = 0,
    layer_indices: tuple[int, int, int] | None = None,
    force_accept_length: int | None = None,
    reference_tokens=None,
) -> tuple[list[int], dict[str, Any]]:
    eos_ids = _eos_id_set(model.generation_config.eos_token_id)
    min_new_tokens = max(0, int(min_new_tokens or 0))
    if layer_indices is None:
        layer_indices = _parse_layer_indices(os.getenv("AICAS_EAGLE3_LAYER_INDICES", ""), 32)
    config = Eagle3Config.from_env(force_accept_length=force_accept_length)
    state, current_input_ids, draft_tree, _ref_tensor = prepare_initial_state(
        model, inputs, draft_model,
        config=config, layer_indices=layer_indices,
        eos_ids=eos_ids, min_new_tokens=min_new_tokens,
        reference_tokens=reference_tokens,
    )
    state.token_state["eagle3_max_new_tokens"] = int(max_new_tokens)

    # ── Pre-allocate GPU output buffer to avoid GPU→CPU sync per round ──
    # This buffer lives outside the full_graph's memory pool so its contents
    # are never clobbered by CUDA graph replay.
    device = state.device
    output_tokens_cpu = []
    if state.token_chunks:
        output_tokens_cpu.append(state.token_chunks[0].to(device='cpu', dtype=torch.long))
        state.token_chunks = []

    while int(state.generated_len) < int(max_new_tokens):
        generated_before = int(state.generated_len)
        base_cache_len = _eagle3_cache_seq_len(state.past_kv)

        # ── clone_cache (rare: only when accept_kv_source == "canonical") ──
        base_past = None
        best_path = None

        full_graph_result = None
        use_full_graph = _EAGLE3_FULL_GRAPH
        if use_full_graph:
            draft_input_for_full = torch.cat((current_input_ids, state.sample_token), dim=1)
            with _timed_stage(state.stats, "verify"), _profile_range("eagle3.full_graph"):
                full_graph_result = _eagle3_full_graph_generate_verify(
                    model, draft_model, state.past_kv, state.feature,
                    draft_input_for_full, _eagle3_cache_seq_len(state.past_kv),
                    layer_indices, state.token_state,
                )

        if full_graph_result is not None:
            (
                target_tokens, verify_feature, verify_hidden, tree_past,
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
            ) = full_graph_result
            draft_tree = DraftTree(
                draft_tokens, retrieve_indices, tree_mask, tree_position_ids,
            )
            if state.config.exact_eos_suppression:
                target_tokens = _suppress_tree_eos_targets_before_min_length_exact(
                    model, inputs, current_input_ids, target_tokens,
                    draft_tree.draft_tokens, draft_tree.tree_mask,
                    draft_tree.tree_position_ids, eos_ids,
                    generated_len=int(state.generated_len),
                    min_new_tokens=int(min_new_tokens),
                )
            else:
                target_tokens = _suppress_tree_eos_targets_before_min_length(
                    model, target_tokens, verify_hidden,
                    draft_tree.tree_position_ids, eos_ids,
                    generated_len=int(state.generated_len),
                    min_new_tokens=int(min_new_tokens),
                )
            verify = VerifyOutput(
                target_tokens=target_tokens, feature=verify_feature,
                hidden=verify_hidden, tree_past=tree_past, draft_tree=draft_tree,
            )
        else:
            # ── Eager fallback: draft then verify ──
            if draft_tree is None:
                runner = state.token_state.get("draft_topk_runner")
                if not isinstance(runner, Eagle3DraftTopKRunner) or runner.draft_model is not draft_model:
                    runner = Eagle3DraftTopKRunner(draft_model)
                    state.token_state["draft_topk_runner"] = runner
                eager_draft_input = torch.cat((current_input_ids, state.sample_token), dim=1)
                with _timed_stage(state.stats, "draft"), _profile_range("eagle3.draft"):
                    draft_tree = runner.generate(state.feature, eager_draft_input, logits_processor=None)
                    state.stats["drafted"] += max(0, int(draft_tree.draft_tokens.shape[1]) - 1)
            with _timed_stage(state.stats, "verify"), _profile_range("eagle3.verify"):
                verify = verify_tree(
                    model, inputs, state, current_input_ids, draft_tree,
                    layer_indices=layer_indices, eos_ids=eos_ids,
                    min_new_tokens=min_new_tokens,
                )

        # ── stage: posterior (path selection) ──
        with _timed_stage(state.stats, "posterior"), _profile_range("eagle3.posterior"):
            path = select_best_path(verify, state.token_state)

        # ── stage: accept (path + cache select + output) ──
        with _timed_stage(state.stats, "accept"), _profile_range("eagle3.accept"):
            accept, draft_input_ids = accept_path(
                model, draft_model, state,
                current_input_ids, verify, path,
                base_past=base_past,
                base_cache_len=int(base_cache_len),
                max_new_tokens=int(max_new_tokens),
                eos_ids=eos_ids, min_new_tokens=int(min_new_tokens),
                layer_indices=layer_indices,
            )

        # ── stage: accept output processing ──
        with _timed_stage(state.stats, "accept_output"):
            append_count = int(accept.append_count)
            # GPU→GPU copy into the pre-allocated buffer (async, no sync needed)
            accepted_cpu = accept.accepted_tokens.reshape(-1)[:append_count].to(device='cpu', dtype=torch.long)
            output_tokens_cpu.append(accepted_cpu)
            state.generated_len += append_count
            state.stats["rounds"] += 1
            state.stats["accept0"] += int(accept.accept_length == 0)
            state.stats["accepted"] += max(0, int(accept.append_count) - 1)

        current_input_ids = None
        draft_tree = None
        path = None

        if accept.stop_after_append or int(state.generated_len) >= int(max_new_tokens):
            draft_input_ids = None
            verify = None
            break

        # ── stage: draft next (only when not using full_graph) ──
        if not use_full_graph:
            with _timed_stage(state.stats, "draft"), _profile_range("eagle3.draft"):
                draft_tree = draft_next(state, draft_model, accept, draft_input_ids, logits_processor=None)

        # ── stage: update state ──
        with _timed_stage(state.stats, "update_state"), _profile_range("eagle3.update"):
            current_input_ids = update_state(
                state, accept, draft_input_ids,
                generated_before=generated_before,
                base_cache_len=int(base_cache_len),
            )
            if draft_tree is not None:
                state.stats["drafted"] += max(0, int(draft_tree.draft_tokens.shape[1]) - 1)

        draft_input_ids = None
        verify = None

        if debug and int(state.stats["rounds"]) == 1 and accept.path is not None:
            print(
                f"[eagle3-tree] candidates={_target_to_list(accept.path.candidates[accept.path.best_candidate], 8)} "
                f"accepted={accept.accept_length} next={int(accept.next_tok.item())}"
            )

        if _should_adaptive_fallback(state, int(max_new_tokens), debug):
            break

    if output_tokens_cpu:
        all_tokens = torch.cat(output_tokens_cpu)
        tokens = [int(x) for x in all_tokens.reshape(-1).tolist()]
    else:
        tokens = []
    if _perf_timing_enabled():
        graph = _flush_graph_counters_to_stats(state)
        if debug:
            _print_debug_summary(state, graph)
    return tokens, state.stats


__all__ = ["eagle3_generate", "prepare_initial_state", "update_state"]
