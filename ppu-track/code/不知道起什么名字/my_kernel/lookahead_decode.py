from __future__ import annotations

"""Exact block lookahead decoding for the CUDA graph runtime.

This follows the faster implementation style in ``~/wq/my_kernel``:
use a context-keyed n-gram pool, verify a fixed candidate block with one
multi-token forward, and immediately fall back to the existing chunk graph when
the candidate is missing or accepts zero tokens.
"""

import functools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .conf import conf_bool, conf_int
from .cuda_graph_decode import CudaGraphDecodeRuntime, DecodeState, _normalize_eos_ids


@dataclass
class LookaheadGateStats:
    fail_candidate_missing: int = 0
    fail_candidate_shape: int = 0
    fail_remaining_slots: int = 0
    fail_stride_skip: int = 0
    fail_verify_zero_accept: int = 0
    fail_verify_exception: int = 0
    fail_key_context: int = 0


@dataclass
class LookaheadStats:
    eligible_rounds: int = 0
    hits: int = 0
    accept_0: int = 0
    accepted_tokens_total: int = 0
    verify_calls: int = 0
    verify_fallbacks: int = 0
    source_prompt: int = 0
    source_output: int = 0
    source_model: int = 0
    jacobi_drafts: int = 0
    jacobi_rejects: int = 0
    disabled_after_exception: bool = False
    verify_total_ms: float = 0.0
    verify_max_ms: float = 0.0
    gates: LookaheadGateStats = field(default_factory=LookaheadGateStats)

    @property
    def avg_accept(self) -> float:
        return self.accepted_tokens_total / max(1, self.hits)


class LookaheadCandidatePool:
    """Frequency-ranked fixed-block n-gram pool keyed by recent context."""

    def __init__(self, *, block_size: int, key_len: int, max_model_entries: int) -> None:
        self.block_size = int(block_size)
        self.key_len = max(1, int(key_len))
        self.max_model_entries = max(8, int(max_model_entries))
        self._prompt: Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)
        self._output: Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)
        self._model: Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)

    @staticmethod
    def _bump(
        into: Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, ...]]]],
        key: Tuple[int, ...],
        block: Tuple[int, ...],
    ) -> None:
        bucket = into[key]
        for idx, (freq, old_block) in enumerate(bucket):
            if old_block == block:
                bucket[idx] = (freq + 1, old_block)
                return
        bucket.append((1, block))

    @classmethod
    def _add_seq(
        cls,
        tokens: torch.Tensor,
        *,
        block_size: int,
        key_len: int,
        into: Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, ...]]]],
    ) -> None:
        seq = [int(x) for x in tokens.flatten().detach().cpu().tolist()]
        need = key_len + block_size
        if len(seq) < need:
            return
        for start in range(0, len(seq) - need + 1):
            key = tuple(seq[start : start + key_len])
            block = tuple(seq[start + key_len : start + key_len + block_size])
            cls._bump(into, key, block)

    def add_prompt_ngrams(self, tokens: torch.Tensor) -> None:
        self._add_seq(tokens, block_size=self.block_size, key_len=self.key_len, into=self._prompt)

    def add_output_ngrams(self, tokens: torch.Tensor) -> None:
        self._add_seq(tokens, block_size=self.block_size, key_len=self.key_len, into=self._output)

    def add_model_ngrams(self, tokens: torch.Tensor) -> None:
        self._add_seq(tokens, block_size=self.block_size, key_len=self.key_len, into=self._model)
        while len(self._model) > self.max_model_entries:
            del self._model[next(iter(self._model))]

    def get_candidate(self, last_tokens: torch.Tensor) -> Optional[Tuple[torch.Tensor, str]]:
        seq = [int(x) for x in last_tokens.flatten().detach().cpu().tolist()]
        if len(seq) < self.key_len:
            return None
        key = tuple(seq[-self.key_len :])

        def pick(bucket: Dict[Tuple[int, ...], List[Tuple[int, Tuple[int, ...]]]], tag: str):
            items = bucket.get(key)
            if not items:
                return None
            items.sort(key=lambda x: (-x[0], x[1]))
            return torch.tensor([list(items[0][1])], dtype=torch.long, device=last_tokens.device), tag

        for tag, bucket in (("output", self._output), ("prompt", self._prompt), ("model", self._model)):
            got = pick(bucket, tag)
            if got is not None:
                return got
        return None


class LookaheadDecodeRuntime:
    def __init__(self, model) -> None:
        self.model = model
        self.cuda_runtime: CudaGraphDecodeRuntime = model._cuda_graph_runtime
        self.block_size = conf_int("LOOKAHEAD_BLOCK_SIZE", "8", minimum=1)
        self.key_len = conf_int("LOOKAHEAD_KEY_LEN", "1", minimum=1)
        self.max_model_entries = conf_int("LOOKAHEAD_MAX_MODEL_CANDIDATES", "64", minimum=8)
        self.persistent_pool = conf_bool("LOOKAHEAD_PERSISTENT_POOL", "1")
        self.verify_stride = conf_int("LOOKAHEAD_VERIFY_STRIDE", "8", minimum=1)
        self.jacobi_on_miss = conf_bool("LOOKAHEAD_JACOBI_ON_MISS", "0")
        self.jacobi_iters = conf_int("LOOKAHEAD_JACOBI_ITERS", "2", minimum=1)
        self.jacobi_min_accept = conf_int("LOOKAHEAD_JACOBI_MIN_ACCEPT", "2", minimum=1)
        self.min_new_tokens = conf_int("LOOKAHEAD_MIN_NEW_TOKENS", "8", minimum=1)
        self._persistent_pool: Optional[LookaheadCandidatePool] = None

    def _new_pool(self) -> LookaheadCandidatePool:
        return LookaheadCandidatePool(
            block_size=self.block_size,
            key_len=self.key_len,
            max_model_entries=self.max_model_entries,
        )

    def _get_pool(self) -> LookaheadCandidatePool:
        if not self.persistent_pool:
            return self._new_pool()
        if (
            self._persistent_pool is None
            or self._persistent_pool.block_size != self.block_size
            or self._persistent_pool.key_len != self.key_len
        ):
            self._persistent_pool = self._new_pool()
        return self._persistent_pool

    def generate(self, orig_generate, args, kwargs):
        safe = CudaGraphDecodeRuntime._sanitize(kwargs)
        max_new_tokens = int(safe.get("max_new_tokens", 0) or 0)
        if (
            not self.cuda_runtime._eligible(safe)
            or max_new_tokens < self.min_new_tokens
            or max_new_tokens > self.cuda_runtime.runtime_max_new_tokens
        ):
            return orig_generate(*args, **safe)

        input_ids = safe["input_ids"]
        model_kwargs = self._model_kwargs(safe)
        eos_ids = _normalize_eos_ids(
            safe.get("eos_token_id", getattr(self.model.generation_config, "eos_token_id", None))
        )
        eos_tensor = torch.tensor(sorted(eos_ids), dtype=torch.long, device=input_ids.device) if eos_ids else None
        min_new_tokens = CudaGraphDecodeRuntime._effective_min_new_tokens(safe, max_new_tokens)
        fill_after_eos = CudaGraphDecodeRuntime._force_fill_after_eos_enabled(max_new_tokens, min_new_tokens)

        prompt_len = int(input_ids.shape[-1])
        target_len = prompt_len + max_new_tokens
        out = input_ids.new_empty((1, target_len))
        out[:, :prompt_len] = input_ids

        ts_total = time.perf_counter()
        ts_prefill = time.perf_counter()
        outputs, next_token, state, workspace_key, workspace_reused, prefill_mode = self.cuda_runtime._prefill(
            input_ids, model_kwargs, max_new_tokens
        )
        prefill_ms = (time.perf_counter() - ts_prefill) * 1000.0

        cur_len = prompt_len
        out[:, cur_len] = next_token
        cur_len += 1
        if fill_after_eos and self.cuda_runtime._eos_stop_index(next_token.reshape(-1), eos_tensor) is not None:
            if cur_len < target_len:
                out[:, cur_len:target_len].fill_(CudaGraphDecodeRuntime._first_eos_token(eos_tensor))
            return out[:, :target_len]
        stop = self.cuda_runtime._eos_stop_index_after_min(
            next_token.reshape(-1),
            eos_tensor,
            generated_before=0,
            min_new_tokens=min_new_tokens,
        ) is not None

        pool = self._get_pool()
        pool.add_prompt_ngrams(out[:, :prompt_len])
        stats = LookaheadStats()
        graph_tokens = 0
        graph_hits = 0
        graph_capture_ms = 0.0
        decode_steps = 0
        last_la_anchor = -(10**9)
        ts_decode = time.perf_counter()

        while not stop and cur_len < target_len:
            remaining = target_len - cur_len
            if remaining >= self.block_size and not stats.disabled_after_exception:
                if decode_steps - last_la_anchor < self.verify_stride:
                    stats.gates.fail_stride_skip += 1
                else:
                    last_la_anchor = int(decode_steps)
                    la_try = self._try_lookahead_step(
                        state=state,
                        out=out,
                        cur_len=cur_len,
                        pool=pool,
                        stats=stats,
                        remaining=remaining,
                    )
                    if la_try is not None:
                        accepted_count, accepted_tokens, _src = la_try
                        raw_stop_idx = self.cuda_runtime._eos_stop_index(accepted_tokens.reshape(-1), eos_tensor)
                        if fill_after_eos and raw_stop_idx is not None:
                            consumed = raw_stop_idx + 1
                            out[:, cur_len : cur_len + consumed] = accepted_tokens[:, :consumed]
                            cur_len += consumed
                            if cur_len < target_len:
                                out[:, cur_len:target_len].fill_(CudaGraphDecodeRuntime._first_eos_token(eos_tensor))
                            cur_len = target_len
                            self._rewind_state(state, accepted_count - consumed)
                            break
                        stop_idx = self.cuda_runtime._eos_stop_index_after_min(
                            accepted_tokens.reshape(-1),
                            eos_tensor,
                            generated_before=cur_len - prompt_len,
                            min_new_tokens=min_new_tokens,
                        )
                        consumed = accepted_count if stop_idx is None else stop_idx + 1
                        out[:, cur_len : cur_len + consumed] = accepted_tokens[:, :consumed]
                        cur_len += consumed
                        if consumed < accepted_count:
                            self._rewind_state(state, accepted_count - consumed)
                        pool.add_output_ngrams(out[:, prompt_len:cur_len])
                        pool.add_model_ngrams(out[:, max(prompt_len, cur_len - 32) : cur_len])
                        stop = stop_idx is not None
                        continue
            elif remaining < self.block_size:
                stats.gates.fail_remaining_slots += 1

            accepted, used_graph, capture_ms = self._decode_fallback_chunk(
                out, cur_len, target_len, state, outputs, workspace_key
            )
            decode_steps += int(accepted.numel())
            graph_tokens += int(accepted.numel())
            graph_hits += int(used_graph) * int(accepted.numel())
            graph_capture_ms += capture_ms
            raw_stop_idx = self.cuda_runtime._eos_stop_index(accepted.reshape(-1), eos_tensor)
            if fill_after_eos and raw_stop_idx is not None:
                consumed = raw_stop_idx + 1
                self.cuda_runtime._advance_state(state, consumed)
                out[:, cur_len : cur_len + consumed] = accepted[:consumed].view(1, consumed)
                cur_len += consumed
                if cur_len < target_len:
                    out[:, cur_len:target_len].fill_(CudaGraphDecodeRuntime._first_eos_token(eos_tensor))
                cur_len = target_len
                break
            stop_idx = self.cuda_runtime._eos_stop_index_after_min(
                accepted.reshape(-1),
                eos_tensor,
                generated_before=cur_len - prompt_len,
                min_new_tokens=min_new_tokens,
            )
            consumed = int(accepted.numel()) if stop_idx is None else stop_idx + 1
            self.cuda_runtime._advance_state(state, consumed)
            out[:, cur_len : cur_len + consumed] = accepted[:consumed].view(1, consumed)
            cur_len += consumed
            pool.add_output_ngrams(out[:, prompt_len:cur_len])
            pool.add_model_ngrams(out[:, max(prompt_len, cur_len - 32) : cur_len])
            stop = stop_idx is not None

        decode_ms = (time.perf_counter() - ts_decode) * 1000.0
        total_ms = (time.perf_counter() - ts_total) * 1000.0
        self.model._cuda_graph_last_runtime_stats = {
            "mode": "lookahead_decode",
            "prefill_backend": prefill_mode,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "graph_capture_ms": graph_capture_ms,
            "total_ms": total_ms,
            "verify_calls": int(stats.verify_calls),
            "lookahead_hits": int(stats.hits),
            "accepted_from_candidates": int(stats.accepted_tokens_total),
            "lookahead_avg_accept": float(stats.avg_accept),
            "lookahead_verify_ms": float(stats.verify_total_ms),
            "graph_tokens": int(graph_tokens),
            "workspace_reused": int(workspace_reused),
        }
        if conf_bool("LOOKAHEAD_LOG_STATS", "0"):
            generated = max(1, cur_len - prompt_len)
            gates = stats.gates
            print(
                f"[lookahead_decode] prompt_len={prompt_len} max_new_tokens={max_new_tokens} "
                f"prefill_backend={prefill_mode} prefill_ms={prefill_ms:.2f} decode_ms={decode_ms:.2f} "
                f"block={self.block_size} key_len={self.key_len} stride={self.verify_stride} "
                f"attempts={stats.eligible_rounds} verify_calls={stats.verify_calls} hits={stats.hits} "
                f"accepted={stats.accepted_tokens_total} accept_rate={stats.accepted_tokens_total / generated:.2%} "
                f"avg_accept={stats.avg_accept:.2f} verify_ms={stats.verify_total_ms:.2f} "
                f"src_prompt={stats.source_prompt} src_output={stats.source_output} src_model={stats.source_model} "
                f"jacobi_drafts={stats.jacobi_drafts} jacobi_rejects={stats.jacobi_rejects} "
                f"fail_missing={gates.fail_candidate_missing} fail_zero={gates.fail_verify_zero_accept} "
                f"fail_stride={gates.fail_stride_skip} graph_tokens={graph_tokens} graph_hits={graph_hits}"
            )
        return out[:, :cur_len]

    @staticmethod
    def _model_kwargs(safe: Dict[str, Any]) -> Dict[str, Any]:
        model_kwargs = dict(safe)
        model_kwargs.pop("input_ids", None)
        model_kwargs.pop("max_new_tokens", None)
        model_kwargs.pop("generation_config", None)
        for key in (
            "do_sample",
            "num_beams",
            "num_return_sequences",
            "min_new_tokens",
            "eos_token_id",
            "pad_token_id",
            "temperature",
            "top_p",
            "top_k",
        ):
            model_kwargs.pop(key, None)
        return model_kwargs

    def _try_lookahead_step(
        self,
        *,
        state: DecodeState,
        out: torch.Tensor,
        cur_len: int,
        pool: LookaheadCandidatePool,
        stats: LookaheadStats,
        remaining: int,
    ) -> Optional[Tuple[int, torch.Tensor, str]]:
        if remaining < self.block_size:
            stats.gates.fail_remaining_slots += 1
            return None
        if cur_len < self.key_len:
            stats.gates.fail_key_context += 1
            return None
        stats.eligible_rounds += 1
        got = pool.get_candidate(out[:, cur_len - self.key_len : cur_len])
        if got is None:
            stats.gates.fail_candidate_missing += 1
            if not self.jacobi_on_miss:
                return None
            candidate = self._draft_jacobi_candidate(out[:, cur_len - 1 : cur_len], state, self.block_size)
            if candidate is None:
                return None
            src = "model"
            stats.jacobi_drafts += 1
            stats.source_model += 1
        else:
            candidate, src = got
            if src == "prompt":
                stats.source_prompt += 1
            elif src == "output":
                stats.source_output += 1
            else:
                stats.source_model += 1
        if int(candidate.shape[-1]) != self.block_size:
            stats.gates.fail_candidate_shape += 1
            return None
        stats.verify_calls += 1
        try:
            accepted_count, accepted = self._verify_candidate_block(candidate, out[:, cur_len - 1 : cur_len], state, stats)
        except BaseException:
            stats.verify_fallbacks += 1
            stats.gates.fail_verify_exception += 1
            stats.disabled_after_exception = True
            return None
        if accepted_count <= 0:
            stats.accept_0 += 1
            stats.verify_fallbacks += 1
            stats.gates.fail_verify_zero_accept += 1
            return None
        if src == "model" and accepted_count < self.jacobi_min_accept:
            self._rewind_state(state, accepted_count)
            stats.jacobi_rejects += 1
            stats.verify_fallbacks += 1
            return None
        stats.hits += 1
        stats.accepted_tokens_total += int(accepted_count)
        if src == "model":
            pool.add_model_ngrams(torch.cat([out[:, max(0, cur_len - self.key_len) : cur_len], accepted], dim=-1))
        return accepted_count, accepted, src

    def _draft_jacobi_candidate(
        self,
        last_token: torch.Tensor,
        state: DecodeState,
        block_size: int,
    ) -> Optional[torch.Tensor]:
        candidate = last_token.expand(1, block_size).contiguous()
        base_pos = int(state.cache_position_host)
        saved_seq = int(getattr(state.kv_cache, "_seq_len_host", state.kv_cache.get_seq_length()))
        try:
            for _ in range(self.jacobi_iters):
                block_input = torch.cat([last_token.view(1, 1), candidate[:, :-1]], dim=-1)
                inputs = self._build_block_inputs(block_input, state, block_size)
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs.logits[:, -block_size:, :]
                candidate = torch.argmax(logits, dim=-1).to(dtype=torch.long).contiguous()
                self._rollback_state_after_verify(state, base_pos, saved_seq)
            return candidate
        except BaseException:
            return None
        finally:
            self._rollback_state_after_verify(state, base_pos, saved_seq)

    def _verify_candidate_block(
        self,
        candidate: torch.Tensor,
        last_token: torch.Tensor,
        state: DecodeState,
        stats: LookaheadStats,
    ) -> Tuple[int, torch.Tensor]:
        block_size = int(candidate.shape[-1])
        block_input = torch.cat([last_token.view(1, 1), candidate[:, :-1]], dim=-1)
        base_pos = int(state.cache_position_host)
        saved_seq = int(getattr(state.kv_cache, "_seq_len_host", state.kv_cache.get_seq_length()))
        t0 = time.perf_counter()
        try:
            inputs = self._build_block_inputs(block_input, state, block_size)
            outputs = self.model(**inputs, return_dict=True)
            logits = outputs.logits[:, -block_size:, :]
            accepted_count = 0
            for idx in range(block_size):
                greedy = int(torch.argmax(logits[:, idx, :], dim=-1).reshape(-1)[0].item())
                expected = int(candidate[0, idx].item())
                if greedy != expected:
                    break
                accepted_count += 1
            if accepted_count == 0:
                self._rollback_state_after_verify(state, base_pos, saved_seq)
                return 0, candidate[:, :0]
            self._commit_state_after_verify(state, base_pos, accepted_count)
            return accepted_count, candidate[:, :accepted_count].contiguous()
        except BaseException:
            self._rollback_state_after_verify(state, base_pos, saved_seq)
            raise
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            stats.verify_total_ms += dt_ms
            stats.verify_max_ms = max(stats.verify_max_ms, dt_ms)

    @staticmethod
    def _build_block_inputs(block_input: torch.Tensor, state: DecodeState, block_size: int) -> Dict[str, Any]:
        offsets = torch.arange(block_size, dtype=state.cache_position.dtype, device=block_input.device)
        cache_position = (state.cache_position.reshape(-1)[0] + offsets).contiguous()
        lead = max(0, state.position_ids.ndim - 1)
        pos_offsets = offsets.view((1,) * lead + (block_size,))
        position_ids = (state.position_ids[..., -1:] + pos_offsets).contiguous()
        state.kv_cache._write_pos_host = None
        state.kv_cache.cache_seqlens_buf.fill_(int(state.cache_position_host))
        return {
            "input_ids": block_input,
            "past_key_values": state.kv_cache,
            "position_ids": position_ids,
            "cache_position": cache_position,
            "attention_mask": None,
            "use_cache": True,
            "logits_to_keep": block_size,
            "output_hidden_states": False,
        }

    @staticmethod
    def _commit_state_after_verify(state: DecodeState, base_pos: int, accepted_count: int) -> None:
        delta = int(accepted_count)
        state.cache_position.add_(delta)
        state.position_ids.add_(delta)
        state.cache_position_host = int(base_pos + delta)
        state.kv_cache._seq_len_host = int(base_pos + delta)
        state.kv_cache._seq_len = int(base_pos + delta)
        state.kv_cache._write_pos_host = int(base_pos + delta)
        state.kv_cache.cache_seqlens_buf.fill_(int(base_pos + delta))

    @staticmethod
    def _rollback_state_after_verify(state: DecodeState, base_pos: int, saved_seq: int) -> None:
        state.cache_position_host = int(base_pos)
        state.kv_cache._seq_len_host = int(saved_seq)
        state.kv_cache._seq_len = int(saved_seq)
        state.kv_cache._write_pos_host = int(base_pos)
        state.kv_cache.cache_seqlens_buf.fill_(int(saved_seq))

    @staticmethod
    def _rewind_state(state: DecodeState, count: int) -> None:
        if count <= 0:
            return
        state.cache_position.add_(-int(count))
        state.position_ids.add_(-int(count))
        state.cache_position_host -= int(count)
        state.kv_cache._seq_len_host = int(state.cache_position_host)
        state.kv_cache._seq_len = int(state.cache_position_host)
        state.kv_cache._write_pos_host = int(state.cache_position_host)
        state.kv_cache.cache_seqlens_buf.fill_(int(state.cache_position_host))

    def _decode_fallback_chunk(
        self,
        out: torch.Tensor,
        cur_len: int,
        target_len: int,
        state: DecodeState,
        outputs,
        workspace_key,
    ) -> Tuple[torch.Tensor, bool, float]:
        remaining = target_len - cur_len
        chunk_steps = min(self.cuda_runtime.chunk_steps, remaining)
        capture_ms = 0.0
        if chunk_steps > 1:
            model_inputs = self.cuda_runtime._build_decode_inputs(out[:, cur_len - 1 : cur_len], state)
            graph = self.cuda_runtime._get_chunk_graph(
                state.prompt_len,
                outputs.logits.dtype,
                out.device,
                workspace_key,
                chunk_steps,
            )
            captures_before = graph.captures
            ts_graph = time.perf_counter()
            tokens, used_graph = graph.run(self.model, model_inputs)
            if graph.captures > captures_before:
                capture_ms = (time.perf_counter() - ts_graph) * 1000.0
            if tokens is not None:
                return tokens.reshape(-1), used_graph, capture_ms

        model_inputs = self.cuda_runtime._build_decode_inputs(out[:, cur_len - 1 : cur_len], state)
        graph = self.cuda_runtime._get_step_graph(state.prompt_len, outputs.logits.dtype, out.device, workspace_key)
        captures_before = graph.captures
        ts_graph = time.perf_counter()
        token, used_graph = graph.run(self.model, model_inputs)
        if graph.captures > captures_before:
            capture_ms = (time.perf_counter() - ts_graph) * 1000.0
        return token.reshape(-1), used_graph, capture_ms


def apply_lookahead_decode(model) -> bool:
    if getattr(model, "_lookahead_decode_patched", False):
        return True
    if not conf_bool("ENABLE_LOOKAHEAD_DECODE", "1"):
        return False
    if not getattr(model, "_cuda_graph_decode_patched", False) or not hasattr(model, "_cuda_graph_runtime"):
        print("[lookahead_decode] disabled: cuda graph decode runtime is required")
        return False

    runtime = LookaheadDecodeRuntime(model)
    original_generate = model.generate

    @functools.wraps(original_generate)
    def generate(*args, **kwargs):
        call_kwargs = dict(kwargs)
        if args:
            if len(args) == 1 and "input_ids" not in call_kwargs:
                call_kwargs["input_ids"] = args[0]
            else:
                return original_generate(*args, **kwargs)
        try:
            with torch.no_grad():
                return runtime.generate(original_generate, (), call_kwargs)
        except Exception as exc:
            if conf_bool("LOOKAHEAD_LOG_STATS", "0"):
                print(f"[lookahead_decode] fallback={type(exc).__name__}: {exc}")
                import traceback

                traceback.print_exc()
            return original_generate(**CudaGraphDecodeRuntime._sanitize(call_kwargs))

    model.generate = generate
    model._lookahead_runtime = runtime
    model._lookahead_original_generate = original_generate
    model._lookahead_decode_patched = True
    print("[lookahead_decode] exact block lookahead path enabled")
    return True
