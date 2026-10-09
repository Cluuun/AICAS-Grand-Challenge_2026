#!/usr/bin/env python3
"""
AICAS 2026 - Self-Testing Benchmark Tool

Measures TTFT and Throughput, generates result.json for self-testing.

Note: It is recommended not to modify this file. This benchmark is intended for 
self-testing purposes only. The final evaluation will be conducted using a 
separate official benchmark system on standardized hardware by the competition 
committee.
"""
import sys
from pathlib import Path

# 评测 cwd 常为 job_root；默认 sys.path[0] 为 ''（即 cwd），会在「import site / .pth」阶段
# 先于标准库搜索选手目录，若含 types.py 或与三方包撞名，会错链到 pyarrow/types 等。
# -P（安全路径）去掉该隐式前缀；提交目录须在「标准库与 torch 等已 import 之后」再插入
# sys.path，否则选手包内的 json.py / types.py 等会遮蔽标准库（torch 依赖 json.JSONEncoder）。
_JOB_ROOT = Path(__file__).resolve().parent


def _prepend_submission_sys_path() -> None:
    r = str(_JOB_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)
    seen = {r}
    for p in _JOB_ROOT.rglob("src"):
        try:
            if not p.is_dir():
                continue
            parent = str(p.parent.resolve())
            if parent not in seen:
                seen.add(parent)
                sys.path.insert(0, parent)
        except OSError:
            continue


import json
import os
import random
import time
import argparse
import platform
import subprocess
from datetime import datetime

import torch
from PIL import Image
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

_prepend_submission_sys_path()
from evaluation_wrapper import VLMModel

# Fixed parameters - Not recommended to modify
MAX_NEW_TOKENS = 128          # Token length for performance testing
ACCURACY_MAX_TOKENS = 1024    # Token length for accuracy testing
WARMUP_SAMPLES = 10           # Warmup samples for GPU stabilization
PERFORMANCE_SAMPLES = None    # Performance test samples (None = all samples)
VAL_SAMPLES = 50000           # Total validation samples (TextVQA train ~45K)
# 与 --ttft-trim-top-ratio 默认一致：按比例去掉 TTFT 最大（最慢）的若干条后再对 TTFT/吞吐做简单平均
TTFT_TRIM_TOP_RATIO_DEFAULT = 0.04


def _print_cuda_hint_for_custom_extensions() -> None:
    """加载选手模型前提示：预编译 .so 需与当前 PyTorch/CUDA 与驱动兼容（细节见 result.json system_info）。"""
    print("\n[CUDA / 自定义扩展] 预编译 .so 需与评测机 PyTorch 的 CUDA 大版本及驱动兼容；无法由脚本替你验证每个 .so。")
    print(
        f"  torch={torch.__version__}  torch.version.cuda={getattr(torch.version, 'cuda', None)}  "
        f"cuda_available={torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        try:
            print(f"  GPU0={torch.cuda.get_device_name(0)}")
        except Exception:
            pass


def get_system_info() -> dict:
    """Collect system information (hardware and software environment)"""
    info = {
        "timestamp": datetime.now().isoformat(),
    }
    
    # Python environment
    info["python_version"] = sys.version.split()[0]
    info["python_full_version"] = sys.version
    
    # PyTorch information
    info["torch_version"] = torch.__version__
    
    # CUDA information
    if torch.cuda.is_available():
        info["cuda_available"] = True
        info["cuda_version"] = torch.version.cuda if hasattr(torch.version, 'cuda') else "N/A"
        try:
            if torch.backends.cudnn.is_available():
                info["cudnn_version"] = str(torch.backends.cudnn.version())
            else:
                info["cudnn_version"] = "N/A"
        except:
            info["cudnn_version"] = "N/A"
        
        # GPU information
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        
        # GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            info["gpu_memory_gb"] = round(gpu_memory, 2)
        except:
            info["gpu_memory_gb"] = "N/A"
        
        # GPU compute capability
        try:
            compute_capability = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
            info["gpu_compute_capability"] = f"{compute_capability[0]}.{compute_capability[1]}"
        except:
            info["gpu_compute_capability"] = "N/A"
    else:
        info["cuda_available"] = False
        info["cuda_version"] = "N/A"
        info["gpu_count"] = 0
        info["gpu_name"] = "N/A"
    
    # CPU information
    info["cpu_processor"] = platform.processor() or "N/A"
    
    if HAS_PSUTIL:
        try:
            info["cpu_count_physical"] = psutil.cpu_count(logical=False)
            info["cpu_count_logical"] = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info["cpu_freq_mhz"] = round(cpu_freq.current, 2) if cpu_freq.current else "N/A"
            else:
                info["cpu_freq_mhz"] = "N/A"
        except:
            info["cpu_count_physical"] = "N/A"
            info["cpu_count_logical"] = "N/A"
            info["cpu_freq_mhz"] = "N/A"
    else:
        info["cpu_count_physical"] = "N/A"
        info["cpu_count_logical"] = "N/A"
        info["cpu_freq_mhz"] = "N/A"
    
    # Try to get CPU model from /proc/cpuinfo (Linux)
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line.lower():
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
                    elif "Processor" in line and ":" in line:
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
    except:
        pass
    
    if "cpu_model" not in info:
        info["cpu_model"] = platform.processor() or "N/A"
    
    # System information
    info["platform_system"] = platform.system()
    info["platform_release"] = platform.release()
    info["platform_version"] = platform.version()
    info["platform_machine"] = platform.machine()
    info["platform_architecture"] = platform.architecture()[0]
    
    # PPU information (if available)
    info["ppu_available"] = False
    info["ppu_info"] = {}
    
    # Check for PPU-related devices
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "ppu" in gpu_name or "pu" in gpu_name:
                info["ppu_available"] = True
                info["ppu_info"] = {
                    "name": torch.cuda.get_device_name(0),
                    "type": "detected_from_gpu_name"
                }
    except:
        pass
    
    # Try to get detailed GPU info via nvidia-smi (if available)
    if torch.cuda.is_available() and platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if lines:
                    parts = lines[0].split(",")
                    if len(parts) >= 3:
                        info["gpu_driver_version"] = parts[1].strip() if len(parts) > 1 else "N/A"
                        info["gpu_memory_total"] = parts[2].strip() if len(parts) > 2 else "N/A"
        except:
            pass
    
    # Memory information
    if HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024**3), 2)
            info["memory_available_gb"] = round(mem.available / (1024**3), 2)
        except:
            pass
    
    return info


def measure_performance(model: VLMModel, image: Image.Image, question: str) -> tuple:
    """
    Measure performance metrics (TTFT and Throughput)
    
    TTFT measurement: Full model call time (generating 1 token)
    Includes: image encoding, text encoding, cross-modal interaction, prefill, first token generation
    
    Args:
        model: VLMModel instance (must expose processor and model attributes)
        image: PIL Image
        question: Question text
    
    Returns:
        tuple: (ttft, throughput, token_count)
    """
    if not hasattr(model, 'processor') or not hasattr(model, 'model'):
        raise AttributeError("Model must expose 'processor' and 'model' attributes")
    
    processor = model.processor
    device = model.device
    model_obj = model.model
    
    # Clear GPU state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Prepare inputs
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)
    
    input_len = inputs.input_ids.shape[1]
    
    # Step 1: Measure TTFT (generate 1 token, includes all preprocessing)
    try:
        torch.cuda.synchronize()
        start_ttft = time.perf_counter()
        
        # Direct call to underlying model
        with torch.no_grad():
            output_ids_ttft = model_obj.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                temperature=0.0,
                use_cache=True
            )
        
        torch.cuda.synchronize()
        ttft = time.perf_counter() - start_ttft
        
    except torch.cuda.OutOfMemoryError as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[Error] OOM during TTFT measurement: {e}")
        return float('inf'), 0.0, 0
    except Exception as e:
        print(f"[Error] Error during TTFT measurement: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), 0.0, 0
    
    # Clear state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(0.01)  # Ensure state reset
    
    # Step 2: Measure full generation (for Throughput)
    try:
        torch.cuda.synchronize()
        start_full = time.perf_counter()
        
        # Direct call to underlying model
        with torch.no_grad():
            output_ids = model_obj.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                use_cache=True
            )
        
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_full
        
        # Extract generated tokens
        generated_ids = output_ids[0][input_len:]
        token_count = len(generated_ids)
        
    except torch.cuda.OutOfMemoryError as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[Error] OOM during full generation: {e}")
        return ttft, 0.0, 0
    except Exception as e:
        print(f"[Error] Error during full generation: {e}")
        import traceback
        traceback.print_exc()
        return ttft, 0.0, 0
    
    # Calculate throughput
    if total_time > 0.001 and token_count > 0:
        throughput = token_count / total_time
    else:
        throughput = 0.0
    
    return ttft, throughput, token_count


def generate_answer(model: VLMModel, image: Image.Image, question: str, max_new_tokens: int = ACCURACY_MAX_TOKENS) -> dict:
    """
    Generate full answer (for accuracy evaluation)
    
    Args:
        model: VLMModel instance
        image: PIL Image
        question: Question text
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        dict: {"text": str, "token_count": int}
    """
    if not hasattr(model, 'processor') or not hasattr(model, 'model'):
        # Fallback: use generate method
        return model.generate(image, question, max_new_tokens=max_new_tokens)
    
    processor = model.processor
    device = model.device
    model_obj = model.model
    
    # Prepare inputs
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device)
    
    input_len = inputs.input_ids.shape[1]
    
    # Generate answer using underlying model
    with torch.no_grad():
        output_ids = model_obj.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True
        )
    
    # Extract generated tokens
    generated_ids = output_ids[0][input_len:]
    text = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return {
        "text": text,
        "token_count": len(generated_ids)
    }


def _load_eval_subset_indices(spec_path: str, dataset, pool_size: int) -> list:
    """Load benchmark row indices from JSON (keys: indices, or question_ids)."""
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    if "indices" in spec and spec["indices"] is not None:
        indices = [int(x) for x in spec["indices"]]
    elif "question_ids" in spec and spec["question_ids"] is not None:
        want = [int(x) for x in spec["question_ids"]]
        if "image" in dataset.column_names:
            slim = dataset.remove_columns(["image"])
        else:
            slim = dataset
        id_to_i = {}
        for i in range(len(slim)):
            id_to_i[int(slim[i]["question_id"])] = i
        try:
            indices = [id_to_i[q] for q in want]
        except KeyError as e:
            raise ValueError(f"question_id not in dataset (pool 0..{pool_size - 1}): {e}") from e
    else:
        raise ValueError(f"Eval subset JSON must contain non-empty 'indices' or 'question_ids': {spec_path}")
    for idx in indices:
        if idx < 0 or idx >= pool_size:
            raise ValueError(f"Eval subset index out of range [0, {pool_size}): {idx}")
    return indices


def _trim_ttft_outliers(
    ttfts: list,
    throughputs: list,
    trim_top_ratio: float,
):
    """
    去掉 TTFT 最大（最慢）的约 trim_top_ratio 比例样本，再对剩余样本的 TTFT 与吞吐做简单平均；
    两条序列按同一下标对齐剔除。n<10 时不截尾（与极小样本数行为稳定）。
    Returns: (ttfts_used, throughputs_used, trimmed_count)
    """
    n = len(ttfts)
    if n == 0 or trim_top_ratio <= 0:
        return ttfts, throughputs, 0
    # Keep behavior stable for tiny sample counts.
    if n < 10:
        return ttfts, throughputs, 0

    trim_k = int(n * trim_top_ratio)
    # Trim at least 1 when ratio > 0 and n is large enough; keep >=1 sample.
    trim_k = max(1, trim_k)
    trim_k = min(trim_k, n - 1)
    if trim_k <= 0:
        return ttfts, throughputs, 0

    order = sorted(range(n), key=lambda i: ttfts[i], reverse=True)
    drop = set(order[:trim_k])
    keep_idx = [i for i in range(n) if i not in drop]
    return [ttfts[i] for i in keep_idx], [throughputs[i] for i in keep_idx], trim_k


import re as _re


def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy matching."""
    s = s.lower().strip()
    s = _re.sub(r'[^\w\s]', '', s)
    s = _re.sub(r'\s+', ' ', s).strip()
    return s


def _word_match(text: str, pattern: str) -> bool:
    """Check if pattern appears as a whole word (or multi-word phrase) in text."""
    # For multi-word patterns, use substring (spaces act as boundaries)
    if ' ' in pattern:
        return pattern in text
    # For single-word patterns, use word boundary (negative lookbehind/ahead for letters)
    return bool(_re.search(r'(?<![a-z])' + _re.escape(pattern) + r'(?![a-z])', text))


def _smart_match(prediction: str, accepted_answers: list) -> bool:
    """
    Smart match: case-insensitive, ignore punctuation.
    Checks: exact → raw substring → prediction contains answer (word-boundary) → answer contains prediction.
    """
    pred_lower = prediction.lower().strip()
    pred_norm = _normalize_text(prediction)
    if not pred_norm:
        return False

    for ans in accepted_answers:
        ans_norm = _normalize_text(ans)
        if not ans_norm:
            continue
        # exact match (normalized)
        if pred_norm == ans_norm:
            return True
        # raw case-insensitive substring with word boundary for short answers
        ans_raw = ans.lower().strip()
        if ' ' not in ans_raw:
            # Single word: require word boundary in raw text
            if _re.search(r'(?<![a-z])' + _re.escape(ans_raw) + r'(?![a-z])', pred_lower):
                return True
        else:
            if ans_raw in pred_lower:
                return True
        # prediction contains answer (word-boundary on normalized)
        if _word_match(pred_norm, ans_norm):
            return True
        # answer contains prediction (short predictions like numbers)
        if len(pred_norm) <= 5 and _word_match(ans_norm, pred_norm):
            return True

    return False


def score_accuracy(predictions: list, samples: list = None, answers_path: str = None) -> dict:
    """
    Score predictions against ground-truth answers.

    Priority:
    1. Inline answers from samples (dataset "answers" field)
    2. External JSON answers file (answers_path)

    Returns dict with accuracy, correct/total counts, per-sample details.
    """
    # Build ground-truth map: question_id -> [answers]
    gt_answers = {}

    # Source 1: inline answers from dataset samples
    if samples:
        for s in samples:
            if "answers" in s and s["answers"]:
                gt_answers[str(s["question_id"])] = s["answers"]

    # Source 2: external JSON file (fills in missing or overrides)
    if answers_path and os.path.exists(answers_path):
        with open(answers_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        file_answers = gt.get("answers", {})
        if file_answers:
            for qid, ans in file_answers.items():
                if str(qid) not in gt_answers:
                    gt_answers[str(qid)] = ans

    if not gt_answers:
        print(f"[Warning] No ground-truth answers found (checked dataset + {answers_path})")
        return {"accuracy": None, "correct": 0, "total": 0, "details": []}

    correct = 0
    total = 0
    details = []

    for pred_item in predictions:
        qid = str(pred_item["question_id"])
        prediction_text = pred_item["prediction"]

        if qid not in gt_answers:
            continue

        total += 1
        accepted = gt_answers[qid]
        match = _smart_match(prediction_text, accepted)

        if match:
            correct += 1

        details.append({
            "question_id": int(qid),
            "prediction": prediction_text[:200],
            "expected": accepted,
            "correct": match,
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "details": details,
    }


def run_benchmark(
    model_class,
    model_path: str,
    dataset_path: str,
    output_path: str,
    num_samples: int = None,
    random_seed: int = None,
    eval_subset_path: str = None,
    ttft_trim_top_ratio: float = TTFT_TRIM_TOP_RATIO_DEFAULT,
    answers_path: str = None,
):
    """
    Run benchmark evaluation
    
    Process:
    1. Load participant model
    2. Measure TTFT and Throughput
    3. Generate answers
    4. Calculate statistics
    5. Save results
    
    Args:
        random_seed: Random seed for reproducibility
    """
    # Set random seed (if provided)
    if random_seed is not None:
        import numpy as np
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Load dataset
    print("=" * 60)
    print("AICAS 2026 Benchmark Tool")
    print("=" * 60)
    print(f"\nLoading dataset from: {dataset_path}")

    # Auto-detect format: arrow (load_from_disk) vs parquet directory
    import glob as _glob
    if _glob.glob(os.path.join(dataset_path, "*.parquet")):
        # Prefer validation split for evaluation; fall back to all parquets
        val_files = _glob.glob(os.path.join(dataset_path, "validation-*.parquet"))
        if val_files:
            print(f"  Detected parquet format, loading validation split ({len(val_files)} shards)...")
            dataset = load_dataset("parquet", data_files=val_files, split="train")
        else:
            print(f"  Detected parquet format, loading all shards...")
            dataset = load_dataset("parquet", data_files=os.path.join(dataset_path, "*.parquet"), split="train")
    else:
        dataset = load_from_disk(dataset_path)
    pool_size = min(VAL_SAMPLES, len(dataset))

    # 固定子集 JSON：优先于 --num-samples 随机子集
    if eval_subset_path:
        indices = _load_eval_subset_indices(eval_subset_path, dataset, pool_size)
        total_samples = len(indices)
        if num_samples is not None:
            print(f"[Note] --eval-subset is set; ignoring --num-samples={num_samples}.")
        print(f"Sample pool size (cap VAL_SAMPLES={VAL_SAMPLES}, dataset len={len(dataset)}): {pool_size}")
        print(f"Using fixed eval subset: {eval_subset_path} ({total_samples} rows).")
    elif num_samples is not None:
        k = min(int(num_samples), pool_size)
        indices = random.sample(range(pool_size), k)
        total_samples = len(indices)
        print(f"Sample pool size (cap VAL_SAMPLES={VAL_SAMPLES}, dataset len={len(dataset)}): {pool_size}")
        print(f"Randomly selected {total_samples} sample(s) without replacement from pool.")
    else:
        indices = list(range(pool_size))
        total_samples = pool_size
        print(f"Total samples: {total_samples} (sequential, first {pool_size} in pool)")

    # Performance test samples
    if PERFORMANCE_SAMPLES is None:
        perf_samples = total_samples  # Test all samples
    else:
        perf_samples = min(PERFORMANCE_SAMPLES, total_samples)

    print(f"Performance test samples: {perf_samples}")

    samples = []
    for i in indices:
        item = dataset[i]
        sample = {
            "question_id": item.get("question_id", i),
            "image": item["image"],
            "question": item["question"],
        }
        # Include ground-truth answers from dataset if available
        if "answers" in item:
            sample["answers"] = item["answers"]
        samples.append(sample)
    
    results = {
        "system_info": get_system_info(),
        "performance": {},
        "answers": []
    }
    
    # Load and test participant model
    print("\n" + "=" * 60)
    print("Running Model Benchmark")
    print("=" * 60)
    
    _print_cuda_hint_for_custom_extensions()
    model = model_class(model_path)
    
    # Warmup
    print(f"\nWarming up ({WARMUP_SAMPLES} samples)...")
    for i in range(min(WARMUP_SAMPLES, len(samples))):
        try:
            generate_answer(model, samples[i]["image"], samples[i]["question"], max_new_tokens=10)
        except Exception as e:
            print(f"[Warning] Warmup sample {i} failed: {e}")
    
    # Clear state after warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Performance testing + answer generation
    ttfts = []
    throughputs = []
    predictions = []
    
    print(f"\nMeasuring performance & generating answers...")
    
    # Performance test samples: measure performance + generate full answers
    for sample in tqdm(samples[:perf_samples], desc="Performance"):
        # Clear state before each measurement for fairness
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        try:
            # Step 1: Measure performance
            ttft, throughput, token_count = measure_performance(
                model, sample["image"], sample["question"]
            )
            
            # Check for failures
            if ttft == float('inf') or throughput == 0.0:
                print(f"[Warning] Sample {sample['question_id']} failed (TTFT={ttft}, Throughput={throughput})")
            else:
                ttfts.append(ttft)
                throughputs.append(throughput)
            
            # Clear state again before generating full answer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Step 2: Generate full answer (for accuracy evaluation)
            try:
                result_full = generate_answer(
                    model,
                    sample["image"], 
                    sample["question"],
                    max_new_tokens=ACCURACY_MAX_TOKENS
                )
                
                predictions.append({
                    "question_id": sample["question_id"],
                    "prediction": result_full["text"]
                })
            except Exception as e:
                print(f"[Error] Error generating full answer for sample {sample['question_id']}: {e}")
                predictions.append({
                    "question_id": sample["question_id"],
                    "prediction": ""
                })
                
        except Exception as e:
            print(f"[Error] Sample {sample['question_id']} failed: {e}")
            predictions.append({
                "question_id": sample["question_id"],
                "prediction": ""
            })
            continue
    
    # If there are remaining samples, only generate answers
    if total_samples > perf_samples:
        for sample in tqdm(samples[perf_samples:], desc="Accuracy"):
            try:
                result = generate_answer(
                    model,
                    sample["image"], 
                    sample["question"],
                    max_new_tokens=ACCURACY_MAX_TOKENS
                )
                predictions.append({
                    "question_id": sample["question_id"],
                    "prediction": result["text"]
                })
            except Exception as e:
                print(f"[Error] Error generating answer for sample {sample['question_id']}: {e}")
                predictions.append({
                    "question_id": sample["question_id"],
                    "prediction": ""
                })
    
    # Calculate statistics
    if len(ttfts) > 0:
        ttfts_used, throughputs_used, trimmed_count = _trim_ttft_outliers(
            ttfts, throughputs, ttft_trim_top_ratio
        )
        avg_ttft = sum(ttfts_used) / len(ttfts_used) * 1000  # Convert to ms
        avg_throughput = sum(throughputs_used) / len(throughputs_used)
        if trimmed_count > 0:
            print(
                f"[TTFT trim] Removed top {trimmed_count}/{len(ttfts)} "
                f"samples by TTFT (ratio={ttft_trim_top_ratio:.2f})."
            )
    else:
        avg_ttft = float('inf')
        avg_throughput = 0.0
    
    # Build performance results
    performance = {
        "avg_ttft_ms": round(avg_ttft, 2) if avg_ttft != float('inf') else None,
        "avg_throughput_tokens_per_sec": round(avg_throughput, 2),
    }
    
    results["performance"] = performance
    results["answers"] = predictions

    # Accuracy scoring — uses dataset answers if available, falls back to answers_path
    accuracy_result = None
    has_inline_answers = any("answers" in s for s in samples)
    if has_inline_answers or answers_path:
        accuracy_result = score_accuracy(predictions, samples=samples, answers_path=answers_path)
        results["accuracy"] = {
            "accuracy": accuracy_result["accuracy"],
            "correct": accuracy_result["correct"],
            "total": accuracy_result["total"],
            "details": accuracy_result["details"],
        }

    # Print summary
    if len(ttfts) > 0:
        print(f"\n✓ TTFT: {avg_ttft:.2f} ms")
        print(f"✓ Throughput: {avg_throughput:.2f} tokens/sec")
    else:
        print(f"\n✗ All samples failed!")

    if accuracy_result:
        print(f"✓ Accuracy: {accuracy_result['accuracy']:.4f} ({accuracy_result['correct']}/{accuracy_result['total']})")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    print(f"\n📊 Results Summary:")
    if len(ttfts) > 0:
        print(f"   TTFT: {avg_ttft:.2f} ms")
        print(f"   Throughput: {avg_throughput:.2f} tokens/sec")
    else:
        print(f"   ⚠ All samples failed!")
    if accuracy_result:
        print(f"   Accuracy: {accuracy_result['accuracy']:.4f} ({accuracy_result['correct']}/{accuracy_result['total']})")
    print(f"   Samples evaluated: {total_samples}")
    print(f"\n💾 Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="AICAS 2026 Benchmark Tool")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct", help="Path to model weights")
    parser.add_argument("--dataset-path", type=str, default="/root/textvqa/data", help="Path to validation dataset")
    parser.add_argument("--output", type=str, default="result.json", help="Output JSON file path")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate; random subset without replacement from min(5000,dataset_len) (default: use full pool sequentially)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed (subset sampling + torch/numpy); omit for non-deterministic subset each run",
    )
    parser.add_argument(
        "--eval-subset",
        type=str,
        default="eval_subset_150.json",
        help="JSON with 'indices' or 'question_ids' (default: eval_subset_150.json)",
    )
    parser.add_argument(
        "--ttft-trim-top-ratio",
        type=float,
        default=TTFT_TRIM_TOP_RATIO_DEFAULT,
        help="Trim slowest-TTFT samples by this ratio before averaging TTFT/throughput (default: 0.10; 0 to disable)",
    )
    parser.add_argument(
        "--answers",
        type=str,
        default=None,
        help="JSON with ground-truth answers for accuracy scoring (default: use dataset 'answers' field)",
    )
    
    args = parser.parse_args()
    
    # Use VLMModel (participants modify this class in evaluation_wrapper.py)
    print("=" * 60)
    print("Using VLMModel (modify evaluation_wrapper.py to add optimizations)")
    print("=" * 60)
    
    # Run benchmark
    run_benchmark(
        model_class=VLMModel,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        eval_subset_path=args.eval_subset,
        ttft_trim_top_ratio=args.ttft_trim_top_ratio,
        answers_path=args.answers,
    )


if __name__ == "__main__":
    main()