"""
Lightweight CUDA-event-based section timer for measuring where TTFT goes.

Enable with env var MY_KERNEL_PROFILE=1.  When enabled, instrumentation
points in lean_vision_forward and lean_prefill_forward record GPU events;
at the end of a forward, elapsed ms per section is appended to an in-process
ring buffer.  Call profiler.report() to dump aggregate stats.

Default off — zero overhead in production.

Usage in code:
    from my_kernel.profiler import profiler

    profiler.mark("vit_patch_embed")
    ... work ...
    profiler.mark("vit_blocks_0_5")
    ... work ...
    profiler.flush("vit")      # finalize current group with this prefix

Usage in test script:
    os.environ["MY_KERNEL_PROFILE"] = "1"
    ... run a few generations ...
    from my_kernel.profiler import profiler
    profiler.report()
"""

from __future__ import annotations

import os
import statistics
import sys
from collections import defaultdict
from typing import List, Tuple

import torch

def _enabled() -> bool:
    return os.environ.get("MY_KERNEL_PROFILE", "0") == "1"

class _Profiler:
    def __init__(self):

        self._open: List[Tuple[str, torch.cuda.Event]] = []

        self._hist: dict = defaultdict(list)

    @property
    def enabled(self) -> bool:
        return _enabled()

    def mark(self, label: str):
        """Record a CUDA event with this label.  Pairs against the next mark
        with the same group-prefix flush()."""
        if not self.enabled:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._open.append((label, e))

    def flush(self, group: str | None = None):
        """Finalize and append timings of open events into history.

        Computes elapsed_time between consecutive events; the FIRST event in
        the open list is the anchor, subsequent events form deltas.  After
        flushing, the open list is cleared.

        `group` (optional) is prefixed to each label for namespacing.
        """
        if not self.enabled or len(self._open) < 2:
            self._open.clear()
            return
        torch.cuda.synchronize()
        prefix = (group + ".") if group else ""
        prev_label, prev_event = self._open[0]
        for label, event in self._open[1:]:
            ms = prev_event.elapsed_time(event)
            self._hist[prefix + label].append(ms)
            prev_label, prev_event = label, event
        self._open.clear()

    def report(self, top_n: int = 30, file=sys.stderr):
        if not self._hist:
            print("[profiler] no data", file=file)
            return
        rows = []
        for name, samples in self._hist.items():
            n = len(samples)
            mean = statistics.mean(samples)
            p50 = statistics.median(samples)
            p95 = sorted(samples)[max(0, int(n * 0.95) - 1)] if n > 0 else 0
            rows.append((mean, name, n, mean, p50, p95))
        rows.sort(reverse=True)

        print("", file=file)
        print("=" * 78, file=file)
        print(f"[profiler] timings over {sum(len(s) for s in self._hist.values())} marks",
              file=file)
        print("=" * 78, file=file)
        print(f"{'section':<42}{'n':>6}{'mean':>10}{'p50':>10}{'p95':>10}",
              file=file)
        print("-" * 78, file=file)
        for _, name, n, mean, p50, p95 in rows[:top_n]:
            print(f"{name:<42}{n:>6}{mean:>9.3f}ms{p50:>9.3f}ms{p95:>9.3f}ms",
                  file=file)
        print("=" * 78, file=file)

    def reset(self):
        self._open.clear()
        self._hist.clear()

profiler = _Profiler()
