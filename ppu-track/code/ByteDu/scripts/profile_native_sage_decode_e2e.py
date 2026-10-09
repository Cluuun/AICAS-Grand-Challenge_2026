from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

import sage_decode_attention as sage_decode
from evaluation_wrapper import VLMModel


@dataclass(frozen=True)
class RooflineCalibration:
    hbm_gbps: float
    l2_gbps: float
    bf16_gemv_tflops: float
    int8_dp4a_tops: float
    graph_launch_floor_us: float


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _roofline_calibration() -> RooflineCalibration:
    return RooflineCalibration(
        hbm_gbps=_env_float("JUNKRAT_ROOFLINE_HBM_GBPS", 1550.0),
        l2_gbps=_env_float("JUNKRAT_ROOFLINE_L2_GBPS", 4500.0),
        bf16_gemv_tflops=_env_float("JUNKRAT_ROOFLINE_BF16_GEMV_TFLOPS", 80.0),
        int8_dp4a_tops=_env_float("JUNKRAT_ROOFLINE_INT8_DP4A_TOPS", 120.0),
        graph_launch_floor_us=_env_float("JUNKRAT_ROOFLINE_GRAPH_LAUNCH_FLOOR_US", 2.5),
    )


def _prepare_inputs(model: VLMModel, image, question: str):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    return model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def _run_generate(model: VLMModel, inputs, max_new_tokens: int):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = int(output_ids.shape[1] - inputs.input_ids.shape[1])
    return output_ids, elapsed, generated


def _summarize_profiler(prof: torch.profiler.profile) -> dict:
    events = prof.events()
    names = Counter(evt.name for evt in events)
    device_time_us = Counter()
    for evt in events:
        try:
            device_time_us[evt.name] += float(evt.device_time_total)
        except Exception:
            pass
    cuda_events = [
        evt for evt in events
        if "cuda" in evt.name.lower() or "hggc" in evt.name.lower() or evt.device_type.name == "CUDA"
    ]
    kernel_events = [
        evt for evt in events
        if evt.device_type.name == "CUDA" or "kernel" in evt.name.lower()
    ]
    alloc_events = [
        evt for evt in events
        if "alloc" in evt.name.lower() or "malloc" in evt.name.lower() or "empty" in evt.name.lower()
    ]
    sync_events = [
        evt for evt in events
        if "synchron" in evt.name.lower() or "cudaDeviceSynchronize" in evt.name or "cudaStreamSynchronize" in evt.name
    ]
    interesting = {
        name: count
        for name, count in names.items()
        if any(token in name for token in (
            "sage",
            "decode",
            "Graph",
            "graph",
            "cudaLaunch",
            "cudaGraph",
            "cudaMalloc",
            "cudaFree",
            "cudaMemcpy",
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "empty",
            "zero_",
            "fill_",
            "copy_",
        ))
    }
    return {
        "total_events": len(events),
        "cuda_related_events": len(cuda_events),
        "kernel_like_events": len(kernel_events),
        "alloc_like_events": len(alloc_events),
        "sync_like_events": len(sync_events),
        "interesting_counts": dict(sorted(interesting.items())),
        "top_cuda_or_kernel_events": names.most_common(80),
        "top_device_time_us": device_time_us.most_common(80),
    }


def _sum_device_ms(prof_summary: dict, patterns: tuple[str, ...]) -> float:
    total_us = 0.0
    for name, value in prof_summary.get("top_device_time_us", []):
        if any(pattern in name for pattern in patterns):
            total_us += float(value)
    return total_us / 1000.0


def _roofline_row(name: str, actual_ms: float, theoretical_ms: float, bottleneck: str, next_step: str) -> dict:
    theoretical_ms = max(float(theoretical_ms), 1.0e-6)
    actual_ms = max(float(actual_ms), 0.0)
    return {
        "name": name,
        "actual_ms": actual_ms,
        "theoretical_ms": theoretical_ms,
        "gap_x": actual_ms / theoretical_ms if actual_ms > 0 else 0.0,
        "bottleneck": bottleneck,
        "next_step": next_step,
    }


def _build_roofline(result: dict, inner, input_len: int, prof_summary: dict | None) -> dict:
    cfg = inner.config.text_config
    layers = int(cfg.num_hidden_layers)
    hidden = int(cfg.hidden_size)
    heads = int(cfg.num_attention_heads)
    kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(cfg.head_dim)
    inter = int(cfg.intermediate_size)
    vocab = int(cfg.vocab_size)
    generated = int(result["generated_tokens"])
    decode_steps = max(generated - 1, 0)
    avg_cache_len = float(input_len) + max(decode_steps - 1, 0) / 2.0
    cal = _roofline_calibration()
    prof_summary = prof_summary or {}

    def hbm_ms(bytes_count: float) -> float:
        return (bytes_count / (cal.hbm_gbps * 1.0e9)) * 1000.0

    def bf16_ms(flops: float) -> float:
        return (flops / (cal.bf16_gemv_tflops * 1.0e12)) * 1000.0

    def int8_ms(ops: float) -> float:
        return (ops / (cal.int8_dp4a_tops * 1.0e12)) * 1000.0

    layer_steps = max(layers * decode_steps, 1)
    qkv_flops = layer_steps * 2.0 * hidden * ((heads + 2 * kv_heads) * head_dim)
    oproj_flops = layer_steps * 2.0 * hidden * hidden
    mlp_flops = layer_steps * 2.0 * ((2 * hidden * inter) + (inter * hidden))
    attn_int8_ops = layer_steps * heads * avg_cache_len * head_dim * 2.0
    kv_bytes = layer_steps * avg_cache_len * kv_heads * ((head_dim * 1.0) + (head_dim * 2.0) + 4.0)
    partial_bytes = layer_steps * heads * ((avg_cache_len + 63.0) // 64.0) * head_dim * 4.0 * 2.0
    lm_flops = max(decode_steps, 1) * 2.0 * hidden * vocab

    actual = {
        "sage_preprocess": _sum_device_ms(prof_summary, ("qkv_prequant_cache", "sage_qkv_cache")),
        "sage_partial": _sum_device_ms(prof_summary, ("sage_decode_qkv_k_i8_v_bf16_partial", "decode_q_i8_k_i8_v_bf16_split_partial")),
        "sage_reduce": _sum_device_ms(prof_summary, ("sage_decode_reduce_kernel", "decode_split_reduce_kernel")),
        "o_proj": _sum_device_ms(prof_summary, ("addmv_bn1_kernel",)),
        "mlp_gate_up": _sum_device_ms(prof_summary, ("fused_norm_gate_up_swiglu",)),
        "mlp_down_norm": _sum_device_ms(prof_summary, ("down_add_sumsq", "norm_gemv_sumsq")),
        "lm_head_argmax": _sum_device_ms(prof_summary, ("triton_linear_argmax", "_triton_linear_argmax")),
        "graph_launch_api": (prof_summary.get("interesting_counts", {}).get("cudaGraphLaunch", 0) * cal.graph_launch_floor_us) / 1000.0,
    }

    rows = [
        _roofline_row("QKV projection", _sum_device_ms(prof_summary, ("gemm_ktype",)) * 0.20, bf16_ms(qkv_flops), "BF16/Q8 GEMV throughput and launch mix", "Use per-layer GEMV attribution; fuse next-layer norm+QKV when possible."),
        _roofline_row("Q/K norm + RoPE + quant + KV write", actual["sage_preprocess"], hbm_ms(layer_steps * (heads + kv_heads) * head_dim * 8.0), "small reductions + scalar cache writes", "Keep preprocess once; fuse with default partial only if it removes a launch."),
        _roofline_row("Sage INT8 QK + V accumulate", actual["sage_partial"], max(int8_ms(attn_int8_ops), hbm_ms(kv_bytes)), "source partial kernel below prebuilt; dp4a/V-load scheduling", "Reduce partial_out traffic, improve vectorized K/V loads, compare occupancy/stalls in Asight Compute."),
        _roofline_row("Split softmax reduce/output", actual["sage_reduce"], hbm_ms(partial_bytes), "partial_out HBM read/write and separate reduce launch", "Fuse reduce with output layout/o_proj boundary or compress partial_out."),
        _roofline_row("o_proj/addmv", actual["o_proj"], bf16_ms(oproj_flops), "GEMV bandwidth/launch", "Evaluate reduce->o_proj fusion only after partial kernel is near prebuilt."),
        _roofline_row("MLP gate/up + SwiGLU", actual["mlp_gate_up"], bf16_ms(layer_steps * 2.0 * 2 * hidden * inter), "Q8 GEMV + activation intermediate", "Fuse RMSNorm/gate/up/SwiGLU and keep activations in scratch/registers where possible."),
        _roofline_row("MLP down + RMSNorm chain", actual["mlp_down_norm"], bf16_ms(layer_steps * 2.0 * inter * hidden), "down GEMV plus residual/norm HBM", "Continue fused down_add_sumsq/norm_gemv path; inspect bandwidth gap."),
        _roofline_row("lm_head + argmax", actual["lm_head_argmax"], bf16_ms(lm_flops), "vocab scan", "Keep direct argmax/no full logits; tune candidate blocks/refine path."),
        _roofline_row("CUDA Graph replay floor", actual["graph_launch_api"], (prof_summary.get("interesting_counts", {}).get("cudaGraphLaunch", 0) * cal.graph_launch_floor_us) / 1000.0, "host graph launch overhead", "Use block replay only if 30-sample throughput improves."),
    ]
    rows.sort(key=lambda row: row["actual_ms"], reverse=True)
    return {
        "calibration": cal.__dict__,
        "dimensions": {
            "layers": layers,
            "hidden": hidden,
            "intermediate": inter,
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "vocab": vocab,
            "input_len": input_len,
            "decode_steps": decode_steps,
            "avg_cache_len_est": avg_cache_len,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", default="./data")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=10)
    parser.add_argument("--output", default="analysis_outputs/native_sage_decode_profile.json")
    parser.add_argument("--with-profiler", action="store_true")
    parser.add_argument("--with-roofline", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    dataset = load_from_disk(args.dataset_path)
    sample = dataset[args.sample_index]
    model = VLMModel(args.model_path)
    inputs = _prepare_inputs(model, sample["image"], sample["question"])

    _run_generate(model, inputs, args.warmup_tokens)
    reset_warmup_state = getattr(model, "reset_warmup_state", None)
    if reset_warmup_state is not None:
        reset_warmup_state()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    sage_decode.reset_stats()
    if args.with_profiler:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            output_ids, elapsed, generated = _run_generate(model, inputs, args.max_new_tokens)
    else:
        prof = None
        output_ids, elapsed, generated = _run_generate(model, inputs, args.max_new_tokens)

    text = model.processor.tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    inner = model.model
    result = {
        "env": {
            "JUNKRAT_ENABLE_NATIVE_SAGE_SPLITK": os.getenv("JUNKRAT_ENABLE_NATIVE_SAGE_SPLITK", "0"),
            "JUNKRAT_DISABLE_DECODE_GRAPH": os.getenv("JUNKRAT_DISABLE_DECODE_GRAPH", "0"),
            "JUNKRAT_DECODE_GRAPH_BLOCK_STEPS": os.getenv("JUNKRAT_DECODE_GRAPH_BLOCK_STEPS", "1"),
        },
        "generated_tokens": generated,
        "elapsed_s": elapsed,
        "throughput_tok_s": generated / elapsed if elapsed > 0 else 0.0,
        "non_empty_answer": bool(text.strip()),
        "decode_graph_enabled": bool(getattr(inner, "_decode_graph_enabled", False)),
        "decode_graph_block_enabled": bool(getattr(inner, "_decode_graph_block_enabled", False)),
        "decode_graph_block_size": int(getattr(inner, "_decode_graph_block_size", 1)),
        "native_sage_backend": getattr(inner, "_native_sage_decode_backend", "unknown"),
        "sage_stats": sage_decode.get_stats(),
    }
    prof_summary = _summarize_profiler(prof) if prof is not None else None
    if prof_summary is not None:
        result["profiler"] = prof_summary
    if args.with_roofline:
        result["roofline"] = _build_roofline(result, inner, int(inputs.input_ids.shape[1]), prof_summary)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
