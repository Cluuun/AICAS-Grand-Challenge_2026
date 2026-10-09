from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_from_disk
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_wrapper import VLMModel
from my_kernel.cuda_graph_decode import CudaGraphDecodeRuntime, _forward_and_select_next_token


def _event_cuda_time_us(event: Any, *, self_time: bool) -> float:
    names = (
        ("self_cuda_time_total", "self_device_time_total", "self_device_time")
        if self_time
        else ("cuda_time_total", "device_time_total", "device_time")
    )
    for name in names:
        if hasattr(event, name):
            value = getattr(event, name)
            if value is not None:
                return float(value)
    return 0.0


def _resolve_existing_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _prepare_sample(model: VLMModel, dataset, sample_idx: int):
    item = dataset[sample_idx]
    processor = model.processor
    device = model.model.device
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": item["image"]},
                {"type": "text", "text": item["question"] + "。"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    return item, inputs


def _print_cuda_summary(prof, *, row_limit: int) -> None:
    events = list(prof.key_averages())
    cuda_events = [e for e in events if _event_cuda_time_us(e, self_time=False) > 0]
    cuda_events.sort(key=lambda e: _event_cuda_time_us(e, self_time=True), reverse=True)

    total_cuda_us = sum(_event_cuda_time_us(e, self_time=True) for e in cuda_events)
    if total_cuda_us <= 0:
        total_cuda_us = sum(_event_cuda_time_us(e, self_time=False) for e in cuda_events)

    print("\n" + "=" * 100)
    print("Decode CUDA operator summary")
    print("=" * 100)
    if not cuda_events:
        print("No CUDA/device events were captured.")
        return

    print(f"Total grouped CUDA self time: {total_cuda_us / 1000.0:.3f} ms")
    print("-" * 100)
    print(f"{'operator':<52} {'calls':>8} {'cuda total':>12} {'cuda self':>12} {'self %':>8}")
    print("-" * 100)
    for event in cuda_events[:row_limit]:
        total_us = _event_cuda_time_us(event, self_time=False)
        self_us = _event_cuda_time_us(event, self_time=True)
        pct = (self_us / total_cuda_us * 100.0) if total_cuda_us > 0 else 0.0
        name = str(event.key)
        if len(name) > 52:
            name = name[:49] + "..."
        print(
            f"{name:<52} {int(event.count):>8} "
            f"{total_us / 1000.0:>10.3f}ms {self_us / 1000.0:>10.3f}ms {pct:>7.2f}%"
        )
    print("-" * 100)


def _print_table(prof, *, sort_by: str, row_limit: int) -> None:
    print("\n" + "=" * 100)
    print(f"Profiler table sorted by {sort_by}")
    print("=" * 100)
    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))


def _profile_decode_only(
    model_obj: VLMModel,
    inputs,
    *,
    max_new_tokens: int,
    warmup_tokens: int,
    warmup_runs: int,
):
    runtime = CudaGraphDecodeRuntime(model_obj.model)

    with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
        for _ in range(max(0, warmup_runs)):
            runtime._prefill(
                inputs.input_ids,
                {k: v for k, v in inputs.items() if k != "input_ids"},
                max(2, warmup_tokens),
            )

    with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
        outputs, next_token, state, _workspace_key, _reused, _prefill_mode = runtime._prefill(
            inputs.input_ids,
            {k: v for k, v in inputs.items() if k != "input_ids"},
            max_new_tokens,
        )

    token_backend = os.environ.get("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head")
    model_inputs = runtime._build_decode_inputs(next_token.reshape(1, 1), state)

    decode_steps = max(0, int(max_new_tokens) - 1)
    if decode_steps <= 0:
        return outputs, 0, 0.0, None

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            with record_function("decode_only_profile"):
                for _ in range(decode_steps):
                    step_token = _forward_and_select_next_token(model_obj.model, model_inputs, token_backend)
                    runtime._advance_state(state, 1)
                    model_inputs = runtime._build_decode_inputs(step_token.reshape(1, 1), state)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return outputs, decode_steps, elapsed, prof


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile decode-only CUDA operators for the root runtime.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--trace", default="decode_only_trace.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler.")

    # Diagnostic mode: disable decode CUDA graph so operator-level events remain visible.
    os.environ["ENABLE_DECODE_CUDA_GRAPH"] = "0"
    os.environ["CUDA_GRAPH_LOG_DECODE_STATS"] = "0"
    os.environ["LOG_THROUGHPUT_BREAKDOWN"] = "0"

    dataset = load_from_disk(str(_resolve_existing_path(args.dataset_path)))
    item = dataset[int(args.sample_index)]

    print("=" * 80)
    print("AICAS decode-only profiler")
    print("=" * 80)
    print(f"model_path={args.model_path}")
    print(f"dataset_path={args.dataset_path}")
    print(f"sample_index={args.sample_index}")
    print(f"question={str(item.get('question', ''))[:120]}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print("ENABLE_DECODE_CUDA_GRAPH=0 (diagnostic profile mode)")

    model_obj = VLMModel(str(_resolve_existing_path(args.model_path)))
    _, inputs = _prepare_sample(model_obj, dataset, int(args.sample_index))
    prompt_len = int(inputs.input_ids.shape[-1])
    print(f"prompt_len={prompt_len}")

    _, decode_steps, elapsed, prof = _profile_decode_only(
        model_obj,
        inputs,
        max_new_tokens=int(args.max_new_tokens),
        warmup_tokens=int(args.warmup_tokens),
        warmup_runs=int(args.warmup_runs),
    )
    throughput = (decode_steps / elapsed) if elapsed > 0 else 0.0
    print(
        f"\nDecode-only elapsed={elapsed * 1000.0:.2f} ms "
        f"decode_steps={decode_steps} throughput={throughput:.2f} tok/s"
    )

    _print_cuda_summary(prof, row_limit=int(args.row_limit))
    _print_table(prof, sort_by="cuda_time_total", row_limit=int(args.row_limit))
    _print_table(prof, sort_by="self_cuda_time_total", row_limit=int(args.row_limit))

    trace_path = Path(args.trace)
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace saved to: {trace_path.resolve()}")
    print("Open chrome://tracing or Perfetto UI and load this trace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
