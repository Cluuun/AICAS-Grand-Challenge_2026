#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import gzip
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def iter_trace_events(path: Path) -> Iterable[dict[str, Any]]:
    """Stream events from a Chrome trace JSON file without loading it all."""

    in_events = False
    collecting = False
    buf: list[str] = []
    depth = 0

    with _open_text(path) as f:
        for line in f:
            if not in_events:
                if '"traceEvents"' in line:
                    in_events = True
                continue

            stripped = line.lstrip()
            if not collecting:
                if stripped.startswith("{"):
                    collecting = True
                    buf = [line]
                    depth = line.count("{") - line.count("}")
                    if depth == 0:
                        text = "".join(buf).rstrip().rstrip(",")
                        try:
                            yield json.loads(text)
                        except json.JSONDecodeError:
                            pass
                        collecting = False
                        buf = []
                elif stripped.startswith("]"):
                    break
            else:
                buf.append(line)
                depth += line.count("{") - line.count("}")
                if depth == 0:
                    text = "".join(buf).rstrip().rstrip(",")
                    try:
                        yield json.loads(text)
                    except json.JSONDecodeError:
                        pass
                    collecting = False
                    buf = []


def _add_aggregate(
    counts: collections.Counter[str],
    totals: dict[str, float],
    maxes: dict[str, float],
    name: str,
    dur: float,
) -> None:
    counts[name] += 1
    totals[name] += dur
    if dur > maxes[name]:
        maxes[name] = dur


def _interesting_name(name: str) -> bool:
    lowered = name.lower()
    return (
        "eagle3" in lowered
        or "spec_" in lowered
        or "lm_head" in lowered
        or name.startswith("aicas")
        or "cudaGraph" in name
        or name in {
            "cudaLaunchKernel",
            "cudaMemcpyAsync",
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "cudaGraphLaunch",
            "cudaGraphInstantiateWithFlags",
            "cudaGraphExecDestroy",
            "cudaGraphDestroy",
        }
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    pos = (len(values) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    frac = pos - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)


def _top_rows(counts: collections.Counter[str], totals: dict[str, float], maxes: dict[str, float], limit: int):
    rows = []
    for name, count in counts.items():
        total = float(totals[name])
        rows.append({
            "name": name,
            "count": int(count),
            "total_us": total,
            "avg_us": total / max(1, int(count)),
            "max_us": float(maxes[name]),
        })
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return rows[:limit]


def _collect_benchmark_intervals(path: Path) -> list[tuple[float, float, str]]:
    """First pass: find all aicas_benchmark.generate.* intervals.

    Returns list of (start_us, end_us, name) sorted by start_us.
    """
    intervals: list[tuple[float, float, str]] = []
    for ev in iter_trace_events(path):
        if ev.get("ph") != "X":
            continue
        name = str(ev.get("name", ""))
        if not name.startswith("aicas_benchmark.generate."):
            continue
        ts = ev.get("ts")
        dur = ev.get("dur", 0.0) or 0.0
        if not isinstance(ts, (int, float)):
            continue
        intervals.append((float(ts), float(ts) + float(dur), name))
    intervals.sort(key=lambda x: x[0])
    return intervals


def _in_benchmark_intervals(
    ts: float,
    end: float,
    intervals: list[tuple[float, float, str]],
    *,
    margin_us: float = 0.0,
) -> bool:
    """Check if [ts, end] overlaps with any benchmark interval (with optional margin)."""
    lo = ts - margin_us
    hi = end + margin_us
    # Binary search for the first interval that could overlap
    import bisect
    idx = bisect.bisect_left([iv[0] for iv in intervals], hi)
    # Check this interval and a few before it
    for i in range(max(0, idx - 1), min(idx + 1, len(intervals))):
        iv_start, iv_end, _ = intervals[i]
        if lo < iv_end and hi > iv_start:
            return True
    return False


def analyze_trace(
    path: Path,
    *,
    gap_threshold_us: float,
    top: int,
    progress_every: int = 0,
    benchmark_only: bool = False,
) -> dict[str, Any]:
    # If benchmark-only, first collect the benchmark intervals
    benchmark_intervals: list[tuple[float, float, str]] | None = None
    if benchmark_only:
        benchmark_intervals = _collect_benchmark_intervals(path)
        if not benchmark_intervals:
            print(
                f"[analyze] WARNING: --benchmark-only set but no "
                f"aicas_benchmark.generate.* intervals found in {path}",
                flush=True,
            )

    cat_counts: collections.Counter[str] = collections.Counter()
    cat_totals: dict[str, float] = collections.defaultdict(float)
    cat_maxes: dict[str, float] = collections.defaultdict(float)

    interesting_counts: collections.Counter[str] = collections.Counter()
    interesting_totals: dict[str, float] = collections.defaultdict(float)
    interesting_maxes: dict[str, float] = collections.defaultdict(float)

    kernel_counts: collections.Counter[str] = collections.Counter()
    kernel_totals: dict[str, float] = collections.defaultdict(float)
    kernel_maxes: dict[str, float] = collections.defaultdict(float)

    cuda_counts: collections.Counter[str] = collections.Counter()
    cuda_totals: dict[str, float] = collections.defaultdict(float)
    cuda_maxes: dict[str, float] = collections.defaultdict(float)

    phase_counts: collections.Counter[str] = collections.Counter()
    phase_totals: dict[str, float] = collections.defaultdict(float)
    phase_maxes: dict[str, float] = collections.defaultdict(float)

    min_ts = float("inf")
    max_end = float("-inf")
    total_events = 0
    complete_events = 0
    benchmark_filtered_events = 0

    prev_kernel_by_stream: dict[tuple[Any, Any], tuple[float, float, str]] = {}
    gap_values: list[float] = []
    gap_total_us = 0.0
    gap_top_heap: list[tuple[float, int, dict[str, Any]]] = []
    gap_serial = 0

    for ev in iter_trace_events(path):
        total_events += 1
        if progress_every > 0 and total_events % int(progress_every) == 0:
            print(f"[progress] {path}: events={total_events}", flush=True)

        if ev.get("ph") != "X":
            continue

        name = str(ev.get("name", ""))
        cat = str(ev.get("cat", ""))
        ts = ev.get("ts")
        dur = ev.get("dur", 0.0) or 0.0
        if not isinstance(ts, (int, float)):
            continue
        ts_f = float(ts)
        dur_f = float(dur)
        end = ts_f + dur_f

        # ── Benchmark-only filter ──
        if benchmark_intervals is not None:
            if not _in_benchmark_intervals(ts_f, end, benchmark_intervals):
                benchmark_filtered_events += 1
                continue

        complete_events += 1
        min_ts = min(min_ts, ts_f)
        max_end = max(max_end, end)

        _add_aggregate(cat_counts, cat_totals, cat_maxes, cat, dur_f)

        if _interesting_name(name):
            _add_aggregate(interesting_counts, interesting_totals, interesting_maxes, name, dur_f)

        if "eagle3" in name.lower() or name.startswith("aicas"):
            _add_aggregate(phase_counts, phase_totals, phase_maxes, name, dur_f)

        if cat == "cuda_runtime":
            _add_aggregate(cuda_counts, cuda_totals, cuda_maxes, name, dur_f)

        if cat == "kernel":
            _add_aggregate(kernel_counts, kernel_totals, kernel_maxes, name, dur_f)

            stream = (ev.get("pid"), ev.get("tid"))
            prev = prev_kernel_by_stream.get(stream)
            if prev is not None:
                prev_end, _prev_dur, prev_name = prev
                gap = ts_f - prev_end
                if gap > float(gap_threshold_us):
                    gap_serial += 1
                    gap_values.append(gap)
                    gap_total_us += gap
                    row = {
                        "gap_us": gap,
                        "at_ts_us": prev_end,
                        "stream": stream,
                        "prev_kernel": prev_name,
                        "next_kernel": name,
                    }
                    if len(gap_top_heap) < top:
                        heapq.heappush(gap_top_heap, (gap, gap_serial, row))
                    elif gap > gap_top_heap[0][0]:
                        heapq.heapreplace(gap_top_heap, (gap, gap_serial, row))
            prev_kernel_by_stream[stream] = (end, dur_f, name)

    span_us = max(0.0, max_end - min_ts) if max_end > min_ts else 0.0
    gap_top = [row for _gap, _serial, row in sorted(gap_top_heap, key=lambda x: x[0], reverse=True)]

    graph_counts = collections.Counter({
        name: count for name, count in cuda_counts.items()
        if "Graph" in name or "graph" in name
    })
    graph_totals = {name: cuda_totals[name] for name in graph_counts}
    graph_maxes = {name: cuda_maxes[name] for name in graph_counts}

    result: dict[str, Any] = {
        "path": str(path),
        "events": total_events,
        "complete_events": complete_events,
        "span_us": span_us,
        "span_ms": span_us / 1000.0,
        "categories": _top_rows(cat_counts, cat_totals, cat_maxes, top),
        "interesting": _top_rows(interesting_counts, interesting_totals, interesting_maxes, top),
        "phases": _top_rows(phase_counts, phase_totals, phase_maxes, top),
        "kernels": _top_rows(kernel_counts, kernel_totals, kernel_maxes, top),
        "cuda_runtime": _top_rows(cuda_counts, cuda_totals, cuda_maxes, top),
        "cuda_graph": _top_rows(graph_counts, graph_totals, graph_maxes, top),
        "kernel_gaps": {
            "threshold_us": float(gap_threshold_us),
            "count": len(gap_values),
            "total_us": gap_total_us,
            "total_ms": gap_total_us / 1000.0,
            "avg_us": gap_total_us / max(1, len(gap_values)),
            "p50_us": _percentile(gap_values, 0.50),
            "p90_us": _percentile(gap_values, 0.90),
            "p99_us": _percentile(gap_values, 0.99),
            "max_us": max(gap_values) if gap_values else 0.0,
            "top": gap_top,
        },
    }
    if benchmark_only:
        result["benchmark_only"] = True
        result["benchmark_interval_count"] = len(benchmark_intervals) if benchmark_intervals else 0
        result["benchmark_filtered_events"] = benchmark_filtered_events
    return result


def _fmt_ms(us: float) -> str:
    return f"{us / 1000.0:,.3f} ms"


def print_rows(title: str, rows: list[dict[str, Any]], *, top: int) -> None:
    print(f"\n## {title}")
    if not rows:
        print("(none)")
        return
    for row in rows[:top]:
        name = str(row["name"])
        if len(name) > 120:
            name = name[:117] + "..."
        print(
            f"{row['count']:>8d}  total={_fmt_ms(row['total_us']):>14s}  "
            f"avg={row['avg_us']:>9.3f} us  max={row['max_us']:>10.3f} us  {name}"
        )


def print_summary(result: dict[str, Any], *, top: int) -> None:
    print("\n" + "=" * 100)
    print(result["path"])
    if result.get("benchmark_only"):
        print(
            f"(filtered to {result.get('benchmark_interval_count', 0)} benchmark intervals; "
            f"skipped {result.get('benchmark_filtered_events', 0):,} non-benchmark events)"
        )
    print("=" * 100)
    print(
        f"events={result['events']:,} complete_events={result['complete_events']:,} "
        f"span={result['span_ms']:,.3f} ms"
    )

    gaps = result["kernel_gaps"]
    print(
        f"kernel_gaps>{gaps['threshold_us']:.1f}us: count={gaps['count']:,} "
        f"total={gaps['total_ms']:,.3f} ms avg={gaps['avg_us']:.3f} us "
        f"p50={gaps['p50_us']:.3f} us p90={gaps['p90_us']:.3f} us "
        f"p99={gaps['p99_us']:.3f} us max={gaps['max_us']:.3f} us"
    )

    print_rows("Categories", result["categories"], top=top)
    print_rows("EAGLE3 / AICAS Phases", result["phases"], top=top)
    print_rows("Interesting Events", result["interesting"], top=top)
    print_rows("CUDA Runtime", result["cuda_runtime"], top=top)
    print_rows("CUDA Graph Runtime", result["cuda_graph"], top=top)
    print_rows("Kernels", result["kernels"], top=top)

    print("\n## Top Kernel Gaps")
    top_gaps = gaps["top"][:top]
    if not top_gaps:
        print("(none)")
    for row in top_gaps:
        prev_name = row["prev_kernel"]
        next_name = row["next_kernel"]
        if len(prev_name) > 58:
            prev_name = prev_name[:55] + "..."
        if len(next_name) > 58:
            next_name = next_name[:55] + "..."
        print(
            f"gap={row['gap_us']:>10.3f} us stream={row['stream']} "
            f"prev={prev_name} next={next_name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze PyTorch/Chrome trace JSON or JSON.GZ files."
    )
    parser.add_argument("traces", nargs="+", help="Trace JSON or JSON.GZ files")
    parser.add_argument("--gap-threshold-us", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json-output", default="", help="Optional path to write machine-readable summary")
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N events")
    parser.add_argument(
        "--benchmark-only", action="store_true", default=False,
        help="Only analyze events within aicas_benchmark.generate.* intervals "
             "(ignores warmup, graph capture, and other non-benchmark phases).",
    )
    args = parser.parse_args()

    results = []
    for trace in args.traces:
        path = Path(trace)
        if not path.exists():
            raise FileNotFoundError(path)
        result = analyze_trace(
            path,
            gap_threshold_us=float(args.gap_threshold_us),
            top=max(1, int(args.top)),
            progress_every=max(0, int(args.progress_every)),
            benchmark_only=bool(args.benchmark_only),
        )
        results.append(result)
        print_summary(result, top=max(1, int(args.top)))

    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
