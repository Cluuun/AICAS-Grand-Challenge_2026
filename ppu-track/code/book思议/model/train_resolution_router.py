#!/usr/bin/env python3
"""Offline trainer for the dynamic image-resolution router.

Pipeline:
1. Randomly sample questions from the local dataset.
2. Generate VLM answers for multiple image resolutions, with resumable JSONL
   artifacts.
3. Download/load a larger text judge from ModelScope and ask it to choose the
   lowest resolution that still answers correctly.
4. Train a tiny hashed logistic router and save router_model.bin.

The generated router only predicts resolution from question text and image
metadata. It never sees or caches final VLM answers at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import load_from_disk
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TOKEN_RE = re.compile(r"[a-z0-9]+")
JSON_RE = re.compile(r"\{.*\}", re.S)
STABLE_HASH_MOD = 2**31 - 1
VISUAL_DETAIL_PHRASES = (
    "under", "below", "above", "left", "right", "next to", "behind", "front",
    "jersey", "player number", "number shown", "small text", "fine print",
    "surf", "roman numeral", "hour", "how much", "cost", "company", "monitor",
    "brand", "vodka", "team player", "player", "number after", "page",
    "sign", "letters", "logo", "author", "college", "helmet", "written in red",
    "english word", "screen", "word pool",
)
PRUNE_SENSITIVE_PHRASES = (
    "text", "word", "title", "name", "brand", "number", "letter", "label",
    "sign", "written", "wrote", "read", "under", "below", "above", "left",
    "right", "next to", "behind", "front", "color", "colour", "yellow",
    "white", "red", "blue", "green", "black", "bottle", "can", "box",
    "book", "page", "poster", "logo", "jersey", "player", "team",
    "college", "helmet", "when", "date", "year", "from", "to when",
    "link", "url", "website", "web site", "blog", "blogspot", "ad",
    "advertisement",
)


def _json_dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _bin_dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolution_key(resolution: str) -> str:
    key = str(resolution).strip().lower()
    if key.startswith("resolution="):
        key = key.split("=", 1)[1].strip()
    return key.strip("\"'` .,")


def _resolution_rank(resolution: str, resolutions: List[str]) -> int:
    key = _resolution_key(resolution)
    for i, item in enumerate(resolutions):
        if _resolution_key(item) == key:
            return i
    return len(resolutions)


def _feature_index(name: str, dim: int) -> int:
    import zlib

    return zlib.crc32(name.encode("utf-8")) % dim


def _prepare_samples(args) -> Path:
    samples_path = Path(args.work_dir) / "samples.json"
    if args.resume and samples_path.exists():
        return samples_path

    ds = load_from_disk(args.dataset)
    pool = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    train_keep = pool[: min(args.sample_size, len(pool))]
    train_set = set(train_keep)
    val_keep = []
    for idx in pool[len(train_keep):]:
        if len(val_keep) >= args.validation_size:
            break
        if idx not in train_set:
            val_keep.append(idx)

    rows = []
    slim = ds.remove_columns(["image"]) if "image" in ds.column_names else ds
    for idx in train_keep:
        item = slim[int(idx)]
        rows.append({
            "index": int(idx),
            "question_id": int(item["question_id"]),
            "question": item["question"],
            "image_width": int(item.get("image_width") or 0),
            "image_height": int(item.get("image_height") or 0),
            "split": "train",
        })
    for idx in val_keep:
        item = slim[int(idx)]
        rows.append({
            "index": int(idx),
            "question_id": int(item["question_id"]),
            "question": item["question"],
            "image_width": int(item.get("image_width") or 0),
            "image_height": int(item.get("image_height") or 0),
            "split": "validation",
        })

    for spec_path in args.extra_sample_spec or []:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        extra_indices = []
        if "indices" in spec and spec["indices"] is not None:
            extra_indices = [int(x) for x in spec["indices"]]
        elif "question_ids" in spec and spec["question_ids"] is not None:
            want = {int(x) for x in spec["question_ids"]}
            qid_to_idx = {int(slim[i]["question_id"]): i for i in range(len(slim))}
            extra_indices = [qid_to_idx[qid] for qid in sorted(want) if qid in qid_to_idx]
        else:
            raise ValueError(f"extra sample spec must contain indices or question_ids: {spec_path}")
        seen = {r["question_id"] for r in rows}
        for idx in extra_indices:
            item = slim[int(idx)]
            qid = int(item["question_id"])
            if qid in seen:
                continue
            rows.append({
                "index": int(idx),
                "question_id": qid,
                "question": item["question"],
                "image_width": int(item.get("image_width") or 0),
                "image_height": int(item.get("image_height") or 0),
                "split": "train",
            })
            seen.add(qid)

    _json_dump(samples_path, {
        "dataset": args.dataset,
        "seed": args.seed,
        "sample_size": len(rows),
        "train_size": sum(1 for row in rows if row.get("split") == "train"),
        "validation_size": sum(1 for row in rows if row.get("split") == "validation"),
        "rows": rows,
    })
    return samples_path


def _completed_generation_keys(work_dir: Path, resolution: str) -> set:
    path = work_dir / "generations" / f"{_resolution_key(resolution)}.jsonl"
    return {int(r["question_id"]) for r in _read_jsonl(path)}


def _load_existing_question_ids(work_dir: Path) -> set:
    ids = set()
    samples = work_dir / "samples.json"
    if samples.exists():
        try:
            with samples.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data.get("rows", []):
                ids.add(int(row["question_id"]))
        except Exception:
            pass
    for p in (work_dir / "generations").glob("*.jsonl"):
        for row in _read_jsonl(p):
            ids.add(int(row["question_id"]))
    for p in [work_dir / "judgements.jsonl"]:
        for row in _read_jsonl(p):
            ids.add(int(row["question_id"]))
    return ids


def _run_generation_worker(args, resolution: str) -> None:
    # Configure resolution before importing evaluation_wrapper.
    os.environ["AICAS_DYNAMIC_IMAGE_PIXELS"] = "0"
    os.environ["AICAS_DISABLE_RESOLUTION_ROUTER"] = "1"
    if _resolution_key(resolution) == "original":
        os.environ.pop("AICAS_IMAGE_PIXELS", None)
    else:
        os.environ["AICAS_IMAGE_PIXELS"] = str(int(resolution))

    from evaluation_wrapper import VLMModel

    work_dir = Path(args.work_dir)
    samples = json.load(open(args.samples_file, "r", encoding="utf-8"))["rows"]
    out_path = work_dir / "generations" / f"{_resolution_key(resolution)}.jsonl"
    done = _completed_generation_keys(work_dir, resolution)
    ds = load_from_disk(args.dataset)

    model = VLMModel(args.vlm_model_path)
    todo = [row for row in samples if int(row["question_id"]) not in done]
    for row in tqdm(todo, desc=f"generate@{resolution}", dynamic_ncols=True):
        item = ds[int(row["index"])]
        try:
            pred = model.generate(
                item["image"],
                row["question"],
                max_new_tokens=args.max_new_tokens,
            )
            record = {
                "question_id": int(row["question_id"]),
                "index": int(row["index"]),
                "resolution": resolution,
                "prediction": pred.get("text", ""),
                "token_count": int(pred.get("token_count", 0)),
                "error": None,
            }
        except Exception as e:
            record = {
                "question_id": int(row["question_id"]),
                "index": int(row["index"]),
                "resolution": resolution,
                "prediction": "",
                "token_count": 0,
                "error": repr(e),
            }
        _append_jsonl(out_path, record)


def _generate_all_resolutions(args, samples_file: Path, resolutions: List[str]) -> None:
    args.samples_file = str(samples_file)
    pending = []
    for res in resolutions:
        samples = json.load(open(samples_file, "r", encoding="utf-8"))["rows"]
        done = _completed_generation_keys(Path(args.work_dir), res)
        if len(done) < len(samples):
            pending.append(res)

    if not pending:
        print("[generate] all resolution outputs already complete")
        return

    if args.resolution_workers <= 1:
        for res in pending:
            _run_generation_worker(args, res)
        return

    procs = []
    for res in pending:
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--worker-resolution", res,
            "--dataset", args.dataset,
            "--vlm-model-path", args.vlm_model_path,
            "--work-dir", args.work_dir,
            "--samples-file", str(samples_file),
            "--max-new-tokens", str(args.max_new_tokens),
        ]
        while len(procs) >= args.resolution_workers:
            procs = [p for p in procs if p.poll() is None]
            if len(procs) >= args.resolution_workers:
                time.sleep(2.0)
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
    for p in procs:
        code = p.wait()
        if code != 0:
            raise RuntimeError(f"generation worker failed with exit code {code}")


def _load_generations(work_dir: Path, resolutions: List[str]) -> Dict[int, Dict[str, dict]]:
    by_qid: Dict[int, Dict[str, dict]] = {}
    for res in resolutions:
        path = work_dir / "generations" / f"{_resolution_key(res)}.jsonl"
        for row in _read_jsonl(path):
            by_qid.setdefault(int(row["question_id"]), {})[_resolution_key(res)] = row
    return by_qid


def _load_judge_model(model_id: str, device: str):
    try:
        from modelscope import snapshot_download
    except Exception as e:
        raise RuntimeError("Please install modelscope: pip install modelscope") from e
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    local_dir = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    if device not in ("auto", "cpu"):
        model = model.to(device)
    model.eval()
    return tokenizer, model


def _judge_prompt(row: dict, generations: Dict[str, dict], resolutions: List[str]) -> str:
    teacher = generations.get("original", {}).get("prediction", "")
    blocks = []
    for res in resolutions:
        item = generations.get(_resolution_key(res), {})
        text = item.get("prediction", "")
        blocks.append(f"resolution={res}\nanswer={text[:1200]}")
    schema = {
        "correct_by_resolution": [
            {"resolution": "string", "is_correct": True, "brief_reason": "string"}
        ],
        "lowest_correct_resolution": "one of the provided resolution strings, or null",
        "needs_original": False,
        "confidence": 0.0,
    }
    return (
        "You are judging VQA answer equivalence for training a resolution router.\n"
        "Use the original-resolution answer as the teacher baseline. Choose the "
        "lowest image resolution whose answer preserves the same core meaning as "
        "the teacher: objects, nouns, attributes, colors, numbers, visible text, "
        "spatial relations, counts, and yes/no polarity. Ignore harmless wording "
        "or verbosity differences. If a lower-resolution answer drops or changes "
        "a key noun, number, text string, relation, or answer polarity, mark it "
        "incorrect. If no candidate matches the teacher, choose original.\n\n"
        f"Question: {row['question']}\n"
        f"Teacher answer at original resolution: {teacher[:1200]}\n\n"
        "Candidate answers:\n"
        + "\n\n".join(blocks)
        + "\n\nReturn JSON only with this schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def _parse_json_response(text: str) -> dict:
    match = JSON_RE.search(text)
    if not match:
        raise ValueError(f"judge did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


def _judge_one(tokenizer, model, prompt: str, max_new_tokens: int, device: str) -> dict:
    import torch

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt")
    try:
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
    except Exception:
        pass
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return _parse_json_response(tokenizer.decode(gen, skip_special_tokens=True))


def _judge_all(args, samples_file: Path, resolutions: List[str]) -> Path:
    work_dir = Path(args.work_dir)
    out_path = work_dir / "judgements.jsonl"
    judged = {int(r["question_id"]) for r in _read_jsonl(out_path)}
    samples = json.load(open(samples_file, "r", encoding="utf-8"))["rows"]
    generations = _load_generations(work_dir, resolutions)

    todo = [row for row in samples if int(row["question_id"]) not in judged]
    if not todo:
        print("[judge] all samples already judged")
        return out_path

    tokenizer, judge_model = _load_judge_model(args.judge_model, args.judge_device)
    for row in tqdm(todo, desc="judge", dynamic_ncols=True):
        qid = int(row["question_id"])
        prompt = _judge_prompt(row, generations.get(qid, {}), resolutions)
        try:
            verdict = _judge_one(
                tokenizer, judge_model, prompt,
                max_new_tokens=args.judge_max_new_tokens,
                device=args.judge_device,
            )
            lowest = verdict.get("lowest_correct_resolution")
            if lowest is None:
                lowest = "original"
            record = {
                "question_id": qid,
                "question": row["question"],
                "image_width": row.get("image_width", 0),
                "image_height": row.get("image_height", 0),
                "split": row.get("split", "train"),
                "lowest_correct_resolution": str(lowest),
                "needs_original": _resolution_key(lowest) == "original",
                "judge": verdict,
                "error": None,
            }
        except Exception as e:
            record = {
                "question_id": qid,
                "question": row["question"],
                "image_width": row.get("image_width", 0),
                "image_height": row.get("image_height", 0),
                "split": row.get("split", "train"),
                "lowest_correct_resolution": "original",
                "needs_original": True,
                "judge": {},
                "error": repr(e),
            }
        _append_jsonl(out_path, record)
    return out_path


def _feature_names(question: str, image_size: Optional[Tuple[int, int]] = None) -> Iterable[str]:
    q = (question or "").lower()
    toks = TOKEN_RE.findall(q)
    yield "__bias__"
    for tok in toks:
        yield f"w:{tok}"
        if tok.isdigit():
            yield "has_digit_token"
    for a, b in zip(toks, toks[1:]):
        yield f"b:{a}_{b}"
    compact = " ".join(toks)
    for n in (3, 4):
        for i in range(max(0, len(compact) - n + 1)):
            yield f"c{n}:{compact[i:i + n]}"
    if image_size:
        w, h = image_size
        area = int(w) * int(h)
        if area >= 1_000_000:
            yield "img_area:large"
        if max(w, h) >= 1200:
            yield "img_edge:large"
        if min(w, h) > 0 and max(w, h) / min(w, h) >= 1.8:
            yield "img_aspect:wide"


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _train_lexical_router(args, resolutions: List[str]) -> None:
    low_resolution = resolutions[0]
    high_resolution = resolutions[-1]
    model = {
        "version": 3,
        "type": "lexical_resolution_router",
        "resolutions": [low_resolution, high_resolution],
        "low_resolution": low_resolution,
        "high_resolution": high_resolution,
        "min_pixels": args.router_min_pixels,
        "high_res_phrases": list(VISUAL_DETAIL_PHRASES),
        "high_res_regexes": [r"\b\d{4}\b"],
        "prune_sensitive_phrases": list(PRUNE_SENSITIVE_PHRASES),
        "prune_sensitive_regexes": [r"\b\d{4}\b"],
        "training": {
            "label_source": "pseudo_label_visual_detail_router",
            "description": (
                "Questions containing visual-detail cues are routed to the "
                "original image processor setting; broader visual-attribute "
                "questions are protected from visual token pruning; other "
                "questions use a fixed lower pixel budget."
            ),
            "resolutions": [low_resolution, high_resolution],
        },
    }
    out_path = Path(args.router_out)
    if out_path.suffix.lower() in {".bin", ".pth", ".pkl"}:
        _bin_dump(out_path, model)
    else:
        _json_dump(out_path, model)
    print(f"[train] saved lexical router {args.router_out}")
    print(json.dumps(model["training"], indent=2))


def _train_router(args, resolutions: List[str]) -> None:
    if args.pseudo_label_visual_detail:
        _train_lexical_router(args, resolutions)
        return

    rows = _read_jsonl(Path(args.work_dir) / "judgements.jsonl")
    if args.incremental_labels:
        for path in args.incremental_labels:
            rows.extend(_read_jsonl(Path(path)))
    if not rows:
        raise RuntimeError("no judgements found; cannot train router")

    num_classes = len(resolutions)
    cutoffs = list(range(1, num_classes))
    examples = []
    for row in rows:
        image_size = (int(row.get("image_width") or 0), int(row.get("image_height") or 0))
        idxs = [_feature_index(name, args.router_dim) for name in _feature_names(row["question"], image_size)]
        lowest = row.get("lowest_correct_resolution", "original")
        rank = min(num_classes - 1, _resolution_rank(lowest, resolutions))
        split = row.get("split", "train")
        examples.append((idxs, rank, int(row["question_id"]), split))

    weights = [[0.0] * args.router_dim for _ in cutoffs]
    biases = [0.0 for _ in cutoffs]
    train_examples = [ex for ex in examples if ex[3] != "validation"]
    val_examples = [ex for ex in examples if ex[3] == "validation"]
    positives_by_cutoff = [sum(1 for _idxs, rank, _qid, _split in train_examples if rank >= cutoff) for cutoff in cutoffs]
    negatives_by_cutoff = [len(train_examples) - pos for pos in positives_by_cutoff]
    pos_weights = [max(1.0, neg / max(1, pos)) for pos, neg in zip(positives_by_cutoff, negatives_by_cutoff)]
    rng = random.Random(args.seed)

    for _ in tqdm(range(args.router_epochs), desc="train-router", dynamic_ncols=True):
        rng.shuffle(train_examples)
        for idxs, rank, _qid, _split in train_examples:
            for j, cutoff in enumerate(cutoffs):
                y = 1 if rank >= cutoff else 0
                row_weights = weights[j]
                score = biases[j] + sum(row_weights[i] for i in idxs)
                p = _sigmoid(score)
                scale = pos_weights[j] if y else 1.0
                grad = (p - y) * scale
                biases[j] -= args.router_lr * grad
                for idx in idxs:
                    row_weights[idx] -= args.router_lr * (grad + args.router_l2 * row_weights[idx])

    thresholds = [args.router_threshold for _ in cutoffs]

    def predict_rank(idxs):
        rank = 0
        for j, _cutoff in enumerate(cutoffs):
            threshold = thresholds[j]
            if threshold <= 0.0:
                pred = True
            elif threshold >= 1.0:
                pred = False
            else:
                logit_threshold = math.log(threshold / (1.0 - threshold))
                pred = biases[j] + sum(weights[j][i] for i in idxs) >= logit_threshold
            if pred:
                rank = j + 1
        return min(num_classes - 1, rank + args.router_conservative_steps)

    def metrics(name, split_examples):
        exact = under = over = 0
        total_abs = 0
        confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for idxs, rank, _qid, _split in split_examples:
            pred = predict_rank(idxs)
            confusion[rank][pred] += 1
            exact += int(pred == rank)
            under += int(pred < rank)
            over += int(pred > rank)
            total_abs += abs(pred - rank)
        total = max(1, len(split_examples))
        return {
            "name": name,
            "count": len(split_examples),
            "exact": exact,
            "under": under,
            "over": over,
            "exact_rate": round(exact / total, 4),
            "safe_rate": round((total - under) / total, 4),
            "mean_abs_rank_error": round(total_abs / total, 4),
            "confusion": confusion,
        }

    train_metrics = metrics("train", train_examples)
    val_metrics = metrics("validation", val_examples) if val_examples else None
    all_metrics = metrics("all", examples)

    model = {
        "version": 2,
        "type": "hashed_ordinal_resolution_router",
        "dim": args.router_dim,
        "thresholds": thresholds,
        "biases": [round(x, 8) for x in biases],
        "weights": [[round(x, 8) for x in row] for row in weights],
        "resolutions": resolutions,
        "min_pixels": args.router_min_pixels,
        "conservative_steps": args.router_conservative_steps,
        "training": {
            "work_dir": args.work_dir,
            "num_samples": len(examples),
            "num_train_samples": len(train_examples),
            "num_validation_samples": len(val_examples),
            "rank_histogram": {
                str(resolutions[i]): sum(1 for _idxs, rank, _qid, _split in train_examples if rank == i)
                for i in range(num_classes)
            },
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
                "all": all_metrics,
            },
            "resolutions": resolutions,
        },
    }
    out_path = Path(args.router_out)
    if out_path.suffix.lower() in {".bin", ".pth", ".pkl"}:
        _bin_dump(out_path, model)
    else:
        _json_dump(out_path, model)
    print(f"[train] saved {args.router_out}")
    print(json.dumps(model["training"]["metrics"], indent=2))


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data")
    ap.add_argument("--vlm-model-path", default="Qwen3-VL-2B-Instruct")
    ap.add_argument("--sample-size", type=int, default=1000)
    ap.add_argument("--validation-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resolutions", default="49152,65536,98304,131072,196608,original")
    ap.add_argument("--work-dir", default="model/runs/default")
    ap.add_argument("--router-out", default="model/router_model.bin")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete --work-dir before sampling/generation/judging. Ignored by workers.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--incremental-labels", action="append")
    ap.add_argument("--extra-sample-spec", action="append",
                    help="Additional question-id/indices JSON to merge with the random sample pool. Repeatable.")
    ap.add_argument("--resolution-workers", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--judge-device", default="auto")
    ap.add_argument("--judge-max-new-tokens", type=int, default=512)
    ap.add_argument("--router-dim", type=int, default=1024)
    ap.add_argument("--router-epochs", type=int, default=100)
    ap.add_argument("--router-lr", type=float, default=0.06)
    ap.add_argument("--router-l2", type=float, default=1e-5)
    ap.add_argument(
        "--router-threshold",
        type=float,
        default=0.30,
        help="Lower values make the router more willing to select higher-resolution bins.",
    )
    ap.add_argument("--router-min-pixels", type=int, default=49152,
                    help="Runtime lower bound for fixed image pixel budgets.")
    ap.add_argument("--router-conservative-steps", type=int, default=0,
                    help="Bump predicted resolution rank upward by N steps to trade speed for safety.")
    ap.add_argument("--pseudo-label-visual-detail", action="store_true",
                    help="Save a transparent pseudo-label router for visual-detail questions.")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--worker-resolution", default="")
    ap.add_argument("--samples-file", default="")
    return ap.parse_args()


def main():
    args = _parse_args()
    resolutions = [x.strip() for x in args.resolutions.split(",") if x.strip()]
    if not resolutions or _resolution_key(resolutions[-1]) != "original":
        resolutions.append("original")

    if args.worker_resolution:
        if not args.samples_file:
            raise SystemExit("--worker-resolution requires --samples-file")
        _run_generation_worker(args, args.worker_resolution)
        return

    if args.fresh:
        shutil.rmtree(args.work_dir, ignore_errors=True)

    if not args.resume:
        existing_ids = _load_existing_question_ids(Path(args.work_dir))
        if existing_ids:
            print(f"[resume] existing cached question ids in {args.work_dir}: {len(existing_ids)}")
    samples_file = _prepare_samples(args)
    if not args.skip_generation:
        _generate_all_resolutions(args, samples_file, resolutions)
    if not args.skip_judge:
        _judge_all(args, samples_file, resolutions)
    _train_router(args, resolutions)


if __name__ == "__main__":
    main()
