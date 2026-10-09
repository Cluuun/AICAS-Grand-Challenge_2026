from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SUMMARY_RE = re.compile(
    r"SUMMARY\s+(?P<label>\S+)\s+count\s+(?P<count>\d+)\s+"
    r"avg_ttft_ms\s+(?P<ttft>\S+)\s+avg_thr\s+(?P<thr>\S+)"
)


def _float_or_none(value: str):
    if value in ("None", "nan"):
        return None
    return float(value)


def _run_one(args: argparse.Namespace, layer_spec: str) -> dict:
    label = f"{args.label_prefix}_{layer_spec.replace(':', '_').replace(',', '_')}"
    env = os.environ.copy()
    env.update(
        {
            "ENABLE_LOOKAHEAD_DECODE": "0",
            "ENABLE_RUNTIME_INT8_QUANT": "1",
            "RUNTIME_INT8_QUANT_BACKEND": args.backend,
            "RUNTIME_INT8_QUANT_TARGETS": args.targets,
            "RUNTIME_INT8_QUANT_LAYERS": layer_spec,
            "RUNTIME_INT8_MAX_QUERY_LEN": str(args.max_query_len),
            "LOG_TTFT_BREAKDOWN": "0",
            "CUDA_GRAPH_LOG_DECODE_STATS": "0",
            "LOOKAHEAD_LOG_STATS": "0",
        }
    )

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_runtime_quant_probe.py")),
        "--model-path",
        args.model_path,
        "--dataset-path",
        args.dataset_path,
        "--warmup-samples",
        str(args.warmup_samples),
        "--start",
        str(args.start),
        "--samples",
        str(args.samples),
        "--label",
        label,
    ]
    print("RUN", layer_spec, flush=True)
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    stdout = proc.stdout
    stderr = proc.stderr

    row = {
        "layer_spec": layer_spec,
        "label": label,
        "returncode": proc.returncode,
        "avg_ttft_ms": None,
        "avg_thr": None,
        "count": 0,
        "stats": {},
        "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-20:]),
    }
    for line in stdout.splitlines():
        match = SUMMARY_RE.search(line)
        if match:
            row["count"] = int(match.group("count"))
            row["avg_ttft_ms"] = _float_or_none(match.group("ttft"))
            row["avg_thr"] = _float_or_none(match.group("thr"))
        if line.startswith("STATS "):
            try:
                row["stats"] = ast.literal_eval(line[len("STATS ") :])
            except Exception:
                row["stats"] = {"raw": line[len("STATS ") :]}

    print(
        "DONE",
        layer_spec,
        "rc",
        row["returncode"],
        "ttft",
        row["avg_ttft_ms"],
        "thr",
        row["avg_thr"],
        "int8_hits",
        row["stats"].get("gate_up_backend_int8"),
        flush=True,
    )
    if proc.returncode != 0:
        print(row["stderr_tail"], file=sys.stderr, flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential runtime int8 layer sweep.")
    parser.add_argument("--model-path", default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", default="./data")
    parser.add_argument("--output", default="quant_layer_sweep.json")
    parser.add_argument("--label-prefix", default="layer")
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--backend", default="acext_wo_gemv")
    parser.add_argument("--targets", default="mlp_gate_up")
    parser.add_argument("--max-query-len", type=int, default=1)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=[
            "all",
            "first:7",
            "first:14",
            "last:7",
            "last:14",
            "range:7:21",
            "range:0:7,range:21:28",
        ],
        help="Layer specs accepted by RUNTIME_INT8_QUANT_LAYERS.",
    )
    args = parser.parse_args()

    rows = [_run_one(args, spec) for spec in args.layers]
    out = Path(args.output)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out}")

    valid = [row for row in rows if row["avg_thr"] is not None]
    valid.sort(key=lambda row: row["avg_thr"], reverse=True)
    print("RANK")
    for row in valid:
        print(
            row["layer_spec"],
            "ttft",
            row["avg_ttft_ms"],
            "thr",
            row["avg_thr"],
            "int8_hits",
            row["stats"].get("gate_up_backend_int8"),
        )


if __name__ == "__main__":
    main()
