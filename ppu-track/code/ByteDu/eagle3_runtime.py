from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import torch


Eagle3HiddenState = tuple[torch.Tensor, ...]
DraftFn = Callable[[torch.Tensor, Eagle3HiddenState | None, int], torch.Tensor | None]
TargetDecodeFn = Callable[[torch.Tensor, int], tuple[torch.Tensor, Eagle3HiddenState | None]]


@dataclass
class Eagle3Stats:
    total_rounds: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    full_accept_rounds: int = 0
    reject_count: int = 0
    target_forwards: int = 0
    generated_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    rollback_count: int = 0
    fallback_count: int = 0
    output_mismatch_count: int = 0
    reject_depth_counts: dict[int, int] = field(default_factory=dict)
    disabled_reason: str | None = None
    checkpoint_available: bool = False
    layer_ids: tuple[int, ...] = ()
    draft_len: int = 0

    def to_dict(self) -> dict:
        avg_draft_len = self.total_draft_tokens / self.total_rounds if self.total_rounds else 0.0
        avg_accepted = self.total_accepted_tokens / self.total_rounds if self.total_rounds else 0.0
        return {
            "total_rounds": self.total_rounds,
            "avg_draft_len": avg_draft_len,
            "avg_accepted_per_round": avg_accepted,
            "full_accept_rate": self.full_accept_rounds / self.total_rounds if self.total_rounds else 0.0,
            "reject_rate_by_depth": {str(k): v for k, v in sorted(self.reject_depth_counts.items())},
            "target_forwards_per_generated_token": (
                self.target_forwards / self.generated_tokens if self.generated_tokens else 0.0
            ),
            "draft_time_ms": self.draft_time_ms,
            "verify_time_ms": self.verify_time_ms,
            "rollback_count": self.rollback_count,
            "fallback_count": self.fallback_count,
            "throughput_delta": None,
            "ttft_delta": None,
            "output_mismatch_count": self.output_mismatch_count,
            "reject_count": self.reject_count,
            "checkpoint_available": self.checkpoint_available,
            "disabled_reason": self.disabled_reason,
            "layer_ids": list(self.layer_ids),
            "draft_len": self.draft_len,
        }


class Eagle3SpeculativeRuntime:
    def __init__(
        self,
        *,
        draft_len: int,
        min_acceptance: float,
        min_rounds_before_disable: int,
        stats_path: str | Path,
        layer_ids: Sequence[int] = (),
        checkpoint_available: bool = False,
        enabled: bool = True,
        disabled_reason: str | None = None,
    ) -> None:
        self.draft_len = max(1, int(draft_len))
        self.min_acceptance = float(min_acceptance)
        self.min_rounds_before_disable = max(1, int(min_rounds_before_disable))
        self.stats_path = Path(stats_path)
        self.enabled = bool(enabled)
        self.stats = Eagle3Stats(
            checkpoint_available=checkpoint_available,
            disabled_reason=disabled_reason if not enabled else None,
            layer_ids=tuple(int(x) for x in layer_ids),
            draft_len=self.draft_len,
        )

    def can_generate(self) -> bool:
        return self.enabled and self.stats.checkpoint_available

    def mark_fallback(self, reason: str) -> None:
        self.stats.fallback_count += 1
        self.stats.disabled_reason = reason
        self.write_stats()

    def write_stats(self) -> None:
        payload = self.stats.to_dict()
        tmp_path = self.stats_path.with_suffix(self.stats_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.stats_path)

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0

    @staticmethod
    def _contains_eos(token: torch.Tensor, eos_token_ids: set[int]) -> bool:
        return bool(eos_token_ids) and int(token.view(-1)[0].item()) in eos_token_ids

    def _maybe_disable_for_low_acceptance(self) -> None:
        if self.stats.total_rounds < self.min_rounds_before_disable:
            return
        acceptance = self.stats.total_accepted_tokens / max(self.stats.total_draft_tokens, 1)
        if acceptance < self.min_acceptance:
            self.enabled = False
            self.stats.fallback_count += 1
            self.stats.disabled_reason = (
                f"acceptance {acceptance:.4f} below threshold {self.min_acceptance:.4f}"
            )

    def generate(
        self,
        *,
        original_input_ids: torch.Tensor,
        first_token: torch.Tensor,
        initial_hidden_state: Eagle3HiddenState | None,
        prompt_len: int,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        draft_fn: DraftFn,
        target_decode_fn: TargetDecodeFn,
    ) -> torch.Tensor:
        generated = torch.empty((1, max_new_tokens), device=original_input_ids.device, dtype=torch.long)
        generated[:, 0] = first_token.view(1)
        num_generated = 1
        self.stats.generated_tokens += 1
        eos_set = {int(token_id) for token_id in eos_token_ids}
        if max_new_tokens <= 1 or self._contains_eos(first_token, eos_set):
            self.write_stats()
            return torch.cat([original_input_ids, generated[:, :num_generated]], dim=1)

        current_input_ids = first_token.view(1, 1)
        hidden_state = initial_hidden_state
        prefix_len = int(prompt_len)

        while num_generated < max_new_tokens:
            remaining = max_new_tokens - num_generated
            candidate_budget = min(self.draft_len, max(0, remaining - 1))
            if candidate_budget <= 0 or not self.can_generate():
                start_verify = time.perf_counter()
                next_token, hidden_state = target_decode_fn(current_input_ids, prefix_len)
                self.stats.verify_time_ms += self._elapsed_ms(start_verify)
                self.stats.target_forwards += 1
                prefix_len += 1
                generated[:, num_generated] = next_token.view(1)
                num_generated += 1
                self.stats.generated_tokens += 1
                current_input_ids = next_token.view(1, 1)
                if self._contains_eos(next_token, eos_set):
                    break
                continue

            start_draft = time.perf_counter()
            candidates = draft_fn(current_input_ids, hidden_state, candidate_budget)
            self.stats.draft_time_ms += self._elapsed_ms(start_draft)
            if candidates is None or candidates.numel() == 0:
                self.stats.fallback_count += 1
                candidate_budget = 0
            else:
                candidates = candidates.view(-1).to(device=original_input_ids.device, dtype=torch.long)
                candidate_budget = min(candidate_budget, int(candidates.numel()))

            if candidate_budget <= 0:
                start_verify = time.perf_counter()
                next_token, hidden_state = target_decode_fn(current_input_ids, prefix_len)
                self.stats.verify_time_ms += self._elapsed_ms(start_verify)
                self.stats.target_forwards += 1
                prefix_len += 1
                generated[:, num_generated] = next_token.view(1)
                num_generated += 1
                self.stats.generated_tokens += 1
                current_input_ids = next_token.view(1, 1)
                if self._contains_eos(next_token, eos_set):
                    break
                continue

            self.stats.total_rounds += 1
            self.stats.total_draft_tokens += candidate_budget
            accepted_this_round = 0
            rejected = False

            for depth in range(candidate_budget):
                start_verify = time.perf_counter()
                target_token, target_hidden = target_decode_fn(current_input_ids, prefix_len)
                self.stats.verify_time_ms += self._elapsed_ms(start_verify)
                self.stats.target_forwards += 1
                prefix_len += 1
                hidden_state = target_hidden
                draft_token = candidates[depth].view(1, 1)
                if int(target_token.view(-1)[0].item()) == int(draft_token.view(-1)[0].item()):
                    generated[:, num_generated] = draft_token.view(1)
                    num_generated += 1
                    self.stats.generated_tokens += 1
                    accepted_this_round += 1
                    current_input_ids = draft_token
                    if self._contains_eos(draft_token, eos_set) or num_generated >= max_new_tokens:
                        rejected = True
                        break
                    continue

                self.stats.reject_count += 1
                self.stats.reject_depth_counts[depth] = self.stats.reject_depth_counts.get(depth, 0) + 1
                generated[:, num_generated] = target_token.view(1)
                num_generated += 1
                self.stats.generated_tokens += 1
                current_input_ids = target_token.view(1, 1)
                rejected = True
                if self._contains_eos(target_token, eos_set):
                    break
                break

            self.stats.total_accepted_tokens += accepted_this_round
            if not rejected and accepted_this_round == candidate_budget and num_generated < max_new_tokens:
                self.stats.full_accept_rounds += 1
                start_verify = time.perf_counter()
                next_token, hidden_state = target_decode_fn(current_input_ids, prefix_len)
                self.stats.verify_time_ms += self._elapsed_ms(start_verify)
                self.stats.target_forwards += 1
                prefix_len += 1
                generated[:, num_generated] = next_token.view(1)
                num_generated += 1
                self.stats.generated_tokens += 1
                current_input_ids = next_token.view(1, 1)
                if self._contains_eos(next_token, eos_set):
                    break

            self._maybe_disable_for_low_acceptance()

        self.write_stats()
        return torch.cat([original_input_ids, generated[:, :num_generated]], dim=1)
