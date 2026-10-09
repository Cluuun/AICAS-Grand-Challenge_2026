from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation_wrapper import VLMModel
from my_kernel.conf import _raw
from my_kernel.cuda_graph_decode import CudaGraphDecodeRuntime

try:
    from my_kernel.cuda_graph_decode import reset_cuda_graph_caches
except ImportError:
    def reset_cuda_graph_caches(_model) -> None:
        return None


BASE_ENV = {
    "LOG_THROUGHPUT_BREAKDOWN": "0",
    "CUDA_GRAPH_LOG_DECODE_STATS": "0",
    "ENABLE_FUSED_LM_HEAD_TOP1": "0",
    "FORCE_MIN_NEW_TOKENS_FOR_THROUGHPUT": "1",
    "THROUGHPUT_MAX_NEW_TOKENS": "128",
}


def _clear_conf_cache() -> None:
    _raw.cache_clear()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _parse_csv(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


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
    device = model.device
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


def _replay_chunk_graph_only(chunk_graph) -> None:
    if not chunk_graph.enabled or chunk_graph.graph is None:
        raise RuntimeError(f"chunk graph not captured: {chunk_graph.last_failure}")
    chunk_graph.graph.replay()
    chunk_graph.replays += 1


def _combo_name(env: Dict[str, str]) -> str:
    return (
        f"splits={env['FLASH_KVCACHE_NUM_SPLITS']}"
        f"__chunk={env['CUDA_GRAPH_CHUNK_STEPS']}"
    )


def _run_profile(args: argparse.Namespace, env: Dict[str, str]) -> Dict[str, Any]:
    for key, value in env.items():
        os.environ[key] = str(value)
    _clear_conf_cache()

    resolved_dataset_path = _resolve_existing_path(args.dataset_path)
    resolved_model_path = _resolve_existing_path(args.model_path)
    dataset = load_from_disk(str(resolved_dataset_path))
    with contextlib.redirect_stdout(io.StringIO()):
        model = VLMModel(str(resolved_model_path))
    item, inputs = _prepare_sample(model, dataset, args.sample_idx)

    runtime = CudaGraphDecodeRuntime(model.model)
    max_new_tokens = int(args.max_new_tokens)

    with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
        for warm_idx_offset in range(max(0, args.model_warmup_samples)):
            warm_idx = (args.sample_idx + warm_idx_offset) % len(dataset)
            _, warm_inputs = _prepare_sample(model, dataset, warm_idx)
            runtime._prefill(
                warm_inputs.input_ids,
                {k: v for k, v in warm_inputs.items() if k != "input_ids"},
                10,
            )

        reset_cuda_graph_caches(model.model)
        runtime = CudaGraphDecodeRuntime(model.model)

        outputs, next_token, state, workspace_key, _reused, _prefill_mode = runtime._prefill(
            inputs.input_ids,
            {k: v for k, v in inputs.items() if k != "input_ids"},
            max_new_tokens,
        )

        token_backend = os.environ.get("DECODE_LM_HEAD_ARGMAX_BACKEND", "direct_lm_head")
        from my_kernel.cuda_graph_decode import _forward_and_select_next_token

        model_inputs = runtime._build_decode_inputs(next_token.reshape(1, 1), state)
        chunk_steps = int(runtime.chunk_steps)

        for _ in range(max(0, args.decode_warmup_steps)):
            step_token = _forward_and_select_next_token(model.model, model_inputs, token_backend)
            runtime._advance_state(state, 1)
            model_inputs = runtime._build_decode_inputs(step_token.reshape(1, 1), state)

        chunk_graph = runtime._get_chunk_graph(
            state.prompt_len,
            outputs.logits.dtype,
            inputs.input_ids.device,
            workspace_key,
            chunk_steps,
        )

        captures_before = int(chunk_graph.captures)
        ts_capture = time.perf_counter()
        tokens, captured = chunk_graph.run(model.model, dict(model_inputs))
        _sync()
        capture_ms = (time.perf_counter() - ts_capture) * 1000.0
        if not captured or tokens is None:
            raise RuntimeError(f"chunk graph capture failed: {chunk_graph.last_failure}")
        if chunk_graph.captures <= captures_before:
            capture_ms = 0.0

        runtime._advance_state(state, chunk_steps)
        model_inputs = runtime._build_decode_inputs(tokens[:, -1:].reshape(1, 1), state)

        for _ in range(max(0, args.warmup_replays)):
            chunk_graph._copy_inputs(model_inputs)
            _replay_chunk_graph_only(chunk_graph)
            _sync()
            runtime._advance_state(state, chunk_steps)
            model_inputs = runtime._build_decode_inputs(chunk_graph.static_tokens[:, -1:].reshape(1, 1), state)

        replay_times_ms: List[float] = []
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        for _ in range(args.replay_count):
            chunk_graph._copy_inputs(model_inputs)
            start_ev.record()
            _replay_chunk_graph_only(chunk_graph)
            end_ev.record()
            _sync()
            replay_times_ms.append(float(start_ev.elapsed_time(end_ev)))
            runtime._advance_state(state, chunk_steps)
            model_inputs = runtime._build_decode_inputs(
                chunk_graph.static_tokens[:, -1:].reshape(1, 1),
                state,
            )

    total_replay_ms = sum(replay_times_ms)
    decode_steps = args.replay_count * chunk_steps
    sorted_ms = sorted(replay_times_ms)
    p50 = sorted_ms[len(sorted_ms) // 2] if sorted_ms else 0.0
    p90_index = min(len(sorted_ms) - 1, int(len(sorted_ms) * 0.9)) if sorted_ms else 0
    p90 = sorted_ms[p90_index] if sorted_ms else 0.0

    return {
        "name": _combo_name(env),
        "status": "ok",
        "sample_idx": args.sample_idx,
        "question_preview": (item.get("question") or "")[:80],
        "prompt_len": int(state.prompt_len),
        "chunk_steps": chunk_steps,
        "decode_steps_measured": decode_steps,
        "graph_capture_ms": capture_ms,
        "total_replay_ms": total_replay_ms,
        "avg_replay_ms_per_chunk": total_replay_ms / max(1, args.replay_count),
        "avg_replay_ms_per_step": total_replay_ms / max(1, decode_steps),
        "p50_replay_ms_per_chunk": p50,
        "p90_replay_ms_per_chunk": p90,
        "tokens_per_sec_est": (1000.0 * decode_steps / total_replay_ms) if total_replay_ms > 0 else 0.0,
        "resolved_model_path": str(resolved_model_path),
        "resolved_dataset_path": str(resolved_dataset_path),
        "env": dict(env),
    }


def _run_worker(args: argparse.Namespace) -> None:
    env = json.loads(args.env_json)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        row = _run_profile(args, env)
    except Exception as exc:
        row = {
            "name": _combo_name(env),
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "resolved_model_path": str(_resolve_existing_path(args.model_path)),
            "resolved_dataset_path": str(_resolve_existing_path(args.dataset_path)),
            "env": env,
        }
    out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False, indent=2))


def _spawn_worker(script: Path, python: str, output_path: Path, args: argparse.Namespace, env: Dict[str, str]) -> Dict[str, Any]:
    cmd = [
        python,
        str(script),
        "--worker",
        "--env-json",
        json.dumps(env, ensure_ascii=False),
        "--output",
        str(output_path),
        "--model-path",
        args.model_path,
        "--dataset-path",
        args.dataset_path,
        "--sample-idx",
        str(args.sample_idx),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--model-warmup-samples",
        str(args.model_warmup_samples),
        "--decode-warmup-steps",
        str(args.decode_warmup_steps),
        "--warmup-replays",
        str(args.warmup_replays),
        "--replay-count",
        str(args.replay_count),
    ]
    print(f"[sweep] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    return json.loads(output_path.read_text(encoding="utf-8"))


def _build_combos(args: argparse.Namespace) -> List[Dict[str, str]]:
    combos: List[Dict[str, str]] = []
    split_values = _parse_csv(args.splits)
    chunk_values = _parse_csv(args.chunk_steps)
    for split_value in split_values:
        for chunk_value in chunk_values:
            env = dict(BASE_ENV)
            env["FLASH_KVCACHE_NUM_SPLITS"] = split_value
            env["CUDA_GRAPH_CHUNK_STEPS"] = chunk_value
            combos.append(env)
    return combos


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    err_rows = [row for row in rows if row.get("status") != "ok"]
    ok_rows.sort(key=lambda row: float(row["avg_replay_ms_per_step"]))

    best = ok_rows[0] if ok_rows else None
    baseline = None
    for row in ok_rows:
        env = row.get("env", {})
        if (
            env.get("FLASH_KVCACHE_NUM_SPLITS") == "1"
            and env.get("CUDA_GRAPH_CHUNK_STEPS") == "1"
        ):
            baseline = row
            break

    ranked = []
    baseline_ms = float(baseline["avg_replay_ms_per_step"]) if baseline else None
    for row in ok_rows:
        step_ms = float(row["avg_replay_ms_per_step"])
        delta_ms = (baseline_ms - step_ms) if baseline_ms is not None else None
        speedup_pct = ((delta_ms / baseline_ms) * 100.0) if baseline_ms and baseline_ms > 0 else None
        ranked.append(
            {
                "name": row["name"],
                "avg_replay_ms_per_step": step_ms,
                "tokens_per_sec_est": row["tokens_per_sec_est"],
                "delta_vs_baseline_ms": delta_ms,
                "speedup_vs_baseline_pct": speedup_pct,
                "env": row["env"],
            }
        )

    return {
        "best": best,
        "baseline": baseline,
        "ranked": ranked,
        "errors": err_rows,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk-step and split joint sweep")
    parser.add_argument("--model-path", default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", default="./data")
    parser.add_argument("--sample-idx", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-warmup-samples", type=int, default=1)
    parser.add_argument("--decode-warmup-steps", type=int, default=16)
    parser.add_argument("--warmup-replays", type=int, default=10)
    parser.add_argument("--replay-count", type=int, default=100)
    parser.add_argument("--splits", default="0,1,2,4")
    parser.add_argument("--chunk-steps", default="1,2,4,8")
    parser.add_argument("--output-dir", default="/tmp/chunk_split_sweep")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--env-json", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.worker:
        if not args.output:
            raise ValueError("--output is required in worker mode")
        if not args.env_json:
            raise ValueError("--env-json is required in worker mode")
        _run_worker(args)
        return

    script = Path(__file__).resolve()
    python = sys.executable
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for env in _build_combos(args):
        out_path = out_dir / f"{_combo_name(env)}.json"
        rows.append(_spawn_worker(script, python, out_path, args, env))

    summary = _summarize(rows)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"WROTE {summary_path}")


if __name__ == "__main__":
    main()
