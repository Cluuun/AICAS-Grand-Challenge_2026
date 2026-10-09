from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import generate_answer
from evaluation_wrapper import VLMModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a few answers for quantization drift checks.")
    parser.add_argument("--model-path", default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", default="./data")
    parser.add_argument("--warmup-samples", type=int, default=10)
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    model = VLMModel(args.model_path)

    for idx in range(min(args.warmup_samples, len(dataset))):
        with contextlib.redirect_stdout(io.StringIO()):
            generate_answer(model, dataset[idx]["image"], dataset[idx]["question"], max_new_tokens=10)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    rows = []
    end = min(args.start + args.samples, len(dataset))
    for idx in range(args.start, end):
        item = dataset[idx]
        with contextlib.redirect_stdout(io.StringIO()):
            answer = generate_answer(
                model,
                item["image"],
                item["question"],
                max_new_tokens=args.max_new_tokens,
            )
        rows.append(
            {
                "idx": idx,
                "question_id": item.get("question_id", idx),
                "question": item["question"],
                "prediction": answer["text"],
                "token_count": answer["token_count"],
            }
        )
        print("ANSWER", idx, "tokens", answer["token_count"], repr(answer["text"][:160]))

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
