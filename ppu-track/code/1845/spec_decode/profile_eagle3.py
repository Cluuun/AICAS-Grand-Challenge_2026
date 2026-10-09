#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_from_disk


def _find_sample_index(ds, question_id: int | None, fallback: int) -> int:
    if question_id is None:
        return int(fallback)
    for idx in range(len(ds)):
        try:
            if int(ds[idx].get("question_id")) == int(question_id):
                return int(idx)
        except Exception:
            continue
    raise RuntimeError(f"question_id={question_id} not found in dataset")


def _prepare_inputs(processor, item, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": item["image"]},
            {"type": "text", "text": item["question"]},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _print_key_table(prof, *, row_limit: int, cuda: bool) -> None:
    sort_by = "self_cuda_time_total" if cuda else "self_cpu_time_total"
    print("\n=== top ops ===")
    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))

    def _time_us(evt, *names: str) -> float:
        for name in names:
            value = getattr(evt, name, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
        return 0.0

    print("\n=== eagle3 ranges ===")
    events = []
    for evt in prof.key_averages():
        key = str(evt.key)
        if "eagle3" in key or "spec_" in key or "lm_head" in key:
            events.append(evt)
    if cuda:
        events.sort(
            key=lambda e: _time_us(e, "self_cuda_time_total", "self_device_time_total"),
            reverse=True,
        )
    else:
        events.sort(key=lambda e: _time_us(e, "self_cpu_time_total"), reverse=True)
    for evt in events[:row_limit]:
        if cuda:
            self_us = _time_us(evt, "self_cuda_time_total", "self_device_time_total")
            total_us = _time_us(evt, "cuda_time_total", "device_time_total")
        else:
            self_us = _time_us(evt, "self_cpu_time_total")
            total_us = _time_us(evt, "cpu_time_total")
        print(
            f"{evt.key:64s} calls={evt.count:5d} "
            f"self={self_us / 1000.0:9.3f}ms total={total_us / 1000.0:9.3f}ms"
        )


def _copy_gzip_checked(src_path: Path, dst_path: Path) -> None:
    tmp_path = dst_path.with_name(f".{dst_path.name}.tmp.{os.getpid()}")
    try:
        with open(src_path, "rb") as src, gzip.open(tmp_path, "wb", compresslevel=3) as dst:
            shutil.copyfileobj(src, dst)

        # Force gzip footer/CRC validation before publishing the final path.
        with gzip.open(tmp_path, "rb") as check:
            while check.read(8 * 1024 * 1024):
                pass
        tmp_path.replace(dst_path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _export_chrome_trace_atomic(prof, trace_file: Path) -> None:
    tmp_path = trace_file.with_name(f".{trace_file.name}.tmp.{os.getpid()}")
    try:
        prof.export_chrome_trace(str(tmp_path))
        tmp_path.replace(trace_file)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", default="../data")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--question-id", type=int, default=34617)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--prof-wait", type=int, default=0)
    parser.add_argument("--prof-warmup", type=int, default=0)
    parser.add_argument("--prof-active", type=int, default=1)
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--output", default="")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--no-spec", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--timing-detail", action="store_true")
    parser.add_argument("--kernel-trace", action="store_true")
    parser.add_argument("--gzip-trace", action="store_true", default=True)
    parser.add_argument("--no-gzip-trace", action="store_false", dest="gzip_trace")
    parser.add_argument("--draft-path", default="")
    parser.add_argument("--tree-total-tokens", type=int, default=None)
    parser.add_argument("--tree-depth", type=int, default=None)
    parser.add_argument("--tree-top-k", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("AICAS_SPEC_DECODE", "0" if args.no_spec else "1")
    os.environ.setdefault("AICAS_SPEC_MT_ATTN", "1")
    os.environ.setdefault("AICAS_EAGLE3_PROFILE_RANGES", "1")
    os.environ.setdefault("AICAS_EAGLE3_DEBUG", "1" if args.debug else "0")
    if args.timing_detail:
        os.environ["AICAS_EAGLE3_TIMING_DETAIL"] = "1"
    if args.kernel_trace:
        os.environ["AICAS_SPEC_KERNEL_TRACE"] = "1"
        os.environ["AICAS_SPEC_MT_ATTN_LOG_HIT"] = "1"
    if args.draft_path:
        os.environ["AICAS_EAGLE3_DRAFT_PATH"] = args.draft_path
    if args.tree_total_tokens is not None:
        os.environ["AICAS_EAGLE3_TREE_TOTAL_TOKENS"] = str(int(args.tree_total_tokens))
    if args.tree_depth is not None:
        os.environ["AICAS_EAGLE3_TREE_DEPTH"] = str(int(args.tree_depth))
    if args.tree_top_k is not None:
        os.environ["AICAS_EAGLE3_TREE_TOP_K"] = str(int(args.tree_top_k))

    from evaluation_wrapper import VLMModel

    wrapper = VLMModel(args.model_path)
    model = wrapper.model
    processor = wrapper.processor
    device = wrapper.device

    ds = load_from_disk(args.dataset_path)
    sample_index = _find_sample_index(ds, args.question_id, args.sample_index)
    item = ds[sample_index]
    inputs = _prepare_inputs(processor, item, device)
    input_len = int(inputs.input_ids.shape[1])

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(args.max_new_tokens),
        do_sample=False,
        temperature=0.0,
        use_cache=True,
    )

    for idx in range(max(0, int(args.warmup_iters))):
        _sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**gen_kwargs)
        _sync()
        print(f"warmup[{idx}] {time.perf_counter() - t0:.3f}s")

    trace_path = args.output.strip()
    if not trace_path:
        qid = item.get("question_id", sample_index)
        trace_path = f"spec_decode/traces/eagle3_q{qid}_{int(args.max_new_tokens)}.json"
    trace_file = Path(trace_path)
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    repeat = max(1, int(args.repeat))
    use_schedule = int(args.prof_wait) > 0 or int(args.prof_warmup) > 0 or int(args.prof_active) != repeat
    prof_kwargs = dict(
        activities=activities,
        record_shapes=bool(args.record_shapes),
        profile_memory=bool(args.profile_memory),
        with_stack=bool(args.with_stack),
    )
    if use_schedule:
        prof_kwargs["schedule"] = torch.profiler.schedule(
            wait=max(0, int(args.prof_wait)),
            warmup=max(0, int(args.prof_warmup)),
            active=max(1, int(args.prof_active)),
            repeat=1,
        )

    output_ids = None
    _sync()
    t0 = time.perf_counter()
    with torch.profiler.profile(**prof_kwargs) as prof:
        for rep in range(repeat):
            with torch.profiler.record_function(f"eagle3_profile.generate.{rep}"):
                with torch.no_grad():
                    output_ids = model.generate(**gen_kwargs)
            _sync()
            if use_schedule:
                prof.step()
    wall = time.perf_counter() - t0

    if output_ids is None:
        raise RuntimeError("profile did not run any generation")
    generated = output_ids[0, input_len:]
    token_count = int(generated.numel())
    throughput = (token_count * repeat) / wall if wall > 0 else 0.0
    _export_chrome_trace_atomic(prof, trace_file)
    gz_trace_file = None
    if args.gzip_trace:
        gz_trace_file = trace_file.with_suffix(trace_file.suffix + ".gz")
        _copy_gzip_checked(trace_file, gz_trace_file)

    print(
        f"profile question_id={item.get('question_id', sample_index)} "
        f"sample_index={sample_index} tokens={token_count} repeat={repeat} wall={wall:.3f}s "
        f"throughput={throughput:.2f} tok/s"
    )
    print(f"chrome_trace={trace_file}")
    if gz_trace_file is not None:
        print(f"chrome_trace_gz={gz_trace_file}")
    _print_key_table(prof, row_limit=max(1, int(args.row_limit)), cuda=torch.cuda.is_available())


if __name__ == "__main__":
    main()
