#!/usr/bin/env python3
"""
统一读取 benchmark 结果，计算 accuracy / 提升率 / 加权总分。

默认基线:
- /opt/AICASGC/score_baseline.json

要求:
- accuracy 通过 compute_accuracy.py 的同一套口径计算
- score 公式:
  score = accuracy * 0.4 + ttft_improvement * 0.3 + throughput_improvement * 0.3
"""

import argparse
import json
from pathlib import Path

from compute_accuracy import compute_class_hit_rate


DEFAULT_BASELINE_PATH = "/opt/AICASGC/score_baseline.json"
DEFAULT_DATASET_PATH = "/opt/AICASGC/data"


def _load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_perf(result_obj: dict) -> dict:
    perf = result_obj.get("performance", {}) or {}
    return {
        "ttft_ms": float(perf.get("avg_ttft_ms", 0.0) or 0.0),
        "throughput": float(perf.get("avg_throughput_tokens_per_sec", 0.0) or 0.0),
        "samples": int(perf.get("samples_evaluated", len(result_obj.get("answers", []) or [])) or 0),
    }


def _compute_score(current: dict, baseline: dict) -> dict:
    b_ttft = float(baseline["ttft_ms"])
    b_tp = float(baseline["throughput"])
    b_acc = float(baseline["accuracy"])

    c_ttft = float(current["ttft_ms"])
    c_tp = float(current["throughput"])
    c_acc = float(current["accuracy"])

    ttft_improvement = ((b_ttft - c_ttft) / b_ttft) if b_ttft > 0 else 0.0
    throughput_improvement = ((c_tp - b_tp) / b_tp) if b_tp > 0 else 0.0
    score = c_acc * 0.4 + ttft_improvement * 0.3 + throughput_improvement * 0.3
    min_allowed_accuracy = b_acc * 0.95

    return {
        "ttft_improvement": ttft_improvement,
        "throughput_improvement": throughput_improvement,
        "score": score,
        "accuracy_valid": c_acc >= min_allowed_accuracy,
        "min_allowed_accuracy": min_allowed_accuracy,
        "delta_vs_baseline_score": score - (b_acc * 0.4),
    }


def evaluate_result(result_path: str, dataset_path: str, baseline_path: str) -> dict:
    result_obj = _load_json(result_path)
    baseline_obj = _load_json(baseline_path)
    perf = _extract_perf(result_obj)

    accuracy_report = compute_class_hit_rate(result_path, dataset_path)
    current = {
        "result_path": str(Path(result_path).resolve()),
        "dataset_path": str(Path(dataset_path).resolve()),
        "ttft_ms": perf["ttft_ms"],
        "throughput": perf["throughput"],
        "accuracy": float(accuracy_report.get("hit_rate", 0.0) or 0.0),
        "hits": int(accuracy_report.get("hits", 0) or 0),
        "total": int(accuracy_report.get("total", 0) or 0),
        "samples": perf["samples"],
    }
    score = _compute_score(current, baseline_obj)

    return {
        "baseline": baseline_obj,
        "current": current,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser(description="AICASGC benchmark result scorer")
    parser.add_argument("--result", type=str, required=True, help="benchmark 输出的 result.json 路径")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH, help="数据集路径")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE_PATH, help="基线 JSON 路径")
    parser.add_argument("--output", type=str, default="", help="可选: 输出评分 JSON 路径")
    args = parser.parse_args()

    report = evaluate_result(args.result, args.dataset_path, args.baseline)

    baseline = report["baseline"]
    current = report["current"]
    score = report["score"]

    print("\n" + "=" * 60)
    print("统一评分结果")
    print("=" * 60)
    print(f"基线文件: {args.baseline}")
    print(
        f"基线: TTFT={baseline['ttft_ms']:.2f} ms, "
        f"TP={baseline['throughput']:.2f} tok/s, "
        f"Acc={baseline['accuracy'] * 100:.2f}%"
    )
    print(
        f"当前: TTFT={current['ttft_ms']:.2f} ms, "
        f"TP={current['throughput']:.2f} tok/s, "
        f"Acc={current['accuracy'] * 100:.2f}% "
        f"({current['hits']}/{current['total']})"
    )
    print(f"TTFT提升率: {score['ttft_improvement'] * 100:.2f}%")
    print(f"吞吐量提升率: {score['throughput_improvement'] * 100:.2f}%")
    print(f"加权总分: {score['score']:.6f}")
    print(
        f"准确率约束: {'通过' if score['accuracy_valid'] else '失败'} "
        f"(下限 {score['min_allowed_accuracy'] * 100:.2f}%)"
    )
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"评分结果已写入: {output_path}")


if __name__ == "__main__":
    main()
