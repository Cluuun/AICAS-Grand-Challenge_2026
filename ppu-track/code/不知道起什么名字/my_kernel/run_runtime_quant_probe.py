from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from statistics import mean

import torch
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import generate_answer, measure_performance
from evaluation_wrapper import VLMModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Small runtime quantization performance probe.")
    parser.add_argument("--model-path", default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", default="./data")
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--label", default="probe")
    args = parser.parse_args()

    model = VLMModel(args.model_path)
    dataset = load_from_disk(args.dataset_path)

    warmup_end = min(args.warmup_samples, len(dataset))
    for idx in range(warmup_end):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                generate_answer(model, dataset[idx]["image"], dataset[idx]["question"], max_new_tokens=10)
        except Exception as exc:
            print(f"WARMUP_FAIL {idx} {type(exc).__name__}: {exc}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    ttfts_ms = []
    throughputs = []
    end = min(args.start + args.samples, len(dataset))
    for idx in range(args.start, end):
        with contextlib.redirect_stdout(io.StringIO()):
            ttft, throughput, token_count = measure_performance(
                model,
                dataset[idx]["image"],
                dataset[idx]["question"],
            )
        print(
            "MEASURE",
            args.label,
            idx,
            "ttft_ms",
            round(ttft * 1000, 2),
            "thr",
            round(throughput, 2),
            "tokens",
            token_count,
        )
        if ttft != float("inf") and throughput > 0:
            ttfts_ms.append(ttft * 1000)
            throughputs.append(throughput)

    avg_ttft = round(mean(ttfts_ms), 2) if ttfts_ms else None
    avg_thr = round(mean(throughputs), 2) if throughputs else None
    print(f"SUMMARY {args.label} count {len(ttfts_ms)} avg_ttft_ms {avg_ttft} avg_thr {avg_thr}")
    print("STATS", getattr(model.model, "_decode_backend_stats", {}))


if __name__ == "__main__":
    main()
