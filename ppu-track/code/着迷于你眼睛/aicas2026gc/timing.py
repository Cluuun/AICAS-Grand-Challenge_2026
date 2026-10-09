import atexit
import os
from collections import defaultdict

import torch


class _TimingCollector:
    def __init__(self):
        self.enabled = os.environ.get("AICAS_TIMING_BREAKDOWN", "0").strip().lower() in {"1", "true", "yes", "on"}
        phases = os.environ.get("AICAS_TIMING_PHASES", "throughput")
        self.phases = {phase.strip() for phase in phases.split(",") if phase.strip()}
        self.print_every = int(os.environ.get("AICAS_TIMING_PRINT_EVERY", "50"))
        self.pending = []
        self.totals_ms = defaultdict(float)
        self.counts = defaultdict(int)
        self.samples = 0
        self.generated_tokens = 0
        self.decode_steps = 0
        self._printed_final = False

    def active(self) -> bool:
        if not self.enabled or not torch.cuda.is_available():
            return False
        phase = os.environ.get("AICAS_BENCHMARK_PHASE", "")
        return phase in self.phases

    def start(self, name: str):
        if not self.active():
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return name, event

    def end(self, token) -> None:
        if token is None:
            return
        name, start_event = token
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self.pending.append((name, start_event, end_event))

    def sample_done(self, generated_tokens: int | None) -> None:
        if not self.enabled or not self.pending:
            return

        # One sync per sample keeps decode-step timing usable without synchronizing every step.
        self.pending[-1][2].synchronize()
        for name, start_event, end_event in self.pending:
            self.totals_ms[name] += float(start_event.elapsed_time(end_event))
            self.counts[name] += 1
        self.pending.clear()

        self.samples += 1
        if generated_tokens is not None:
            tokens = int(generated_tokens)
            self.generated_tokens += tokens
            self.decode_steps += max(tokens - 1, 0)

        if self.print_every > 0 and self.samples % self.print_every == 0:
            self.print_summary(prefix="[TimingBreakdown]")

    def print_summary(self, prefix: str = "[TimingBreakdownFinal]") -> None:
        if not self.enabled or self.samples == 0:
            return

        main_names = ["vision", "language_prefill", "decode_model"]
        main_total = sum(self.totals_ms.get(name, 0.0) for name in main_names)
        avg_tokens = self.generated_tokens / max(self.samples, 1)

        print(
            f"{prefix} samples={self.samples} generated_tokens={self.generated_tokens} "
            f"avg_tokens={avg_tokens:.2f} main_total_ms={main_total:.3f}",
            flush=True,
        )
        for name in [
            "vision",
            "language_prefill",
            "prefill_rope",
            "prefill_phase1_deepstack",
            "prefill_phase2_graph",
            "prefill_phase2_eager",
            "decode_model",
        ]:
            total = self.totals_ms.get(name, 0.0)
            if total <= 0.0:
                continue
            count = self.counts.get(name, 0)
            share = (total / main_total * 100.0) if main_total > 0 else 0.0
            avg = total / max(count, 1)
            print(f"{prefix} {name}: total={total:.3f}ms avg={avg:.3f}ms count={count} share={share:.2f}%", flush=True)

        decode_total = self.totals_ms.get("decode_model", 0.0)
        if self.decode_steps > 0 and decode_total > 0.0:
            print(
                f"{prefix} decode_model_per_step={decode_total / self.decode_steps:.4f}ms "
                f"decode_steps={self.decode_steps}",
                flush=True,
            )

    def final_summary(self) -> None:
        if self._printed_final:
            return
        self._printed_final = True
        if self.pending:
            self.sample_done(None)
        self.print_summary()


_COLLECTOR = _TimingCollector()
atexit.register(_COLLECTOR.final_summary)


def timing_start(name: str):
    return _COLLECTOR.start(name)


def timing_end(token) -> None:
    _COLLECTOR.end(token)


def timing_sample_done(generated_tokens: int | None) -> None:
    _COLLECTOR.sample_done(generated_tokens)
