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
import json
import time
import argparse
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from datasets import load_from_disk
from tqdm import tqdm

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from evaluation_wrapper import VLMModel

# Fixed parameters - Not recommended to modify
MAX_NEW_TOKENS = 128          # Token length for performance testing
ACCURACY_MAX_TOKENS = 1024    # Token length for accuracy testing
WARMUP_SAMPLES = 10           # Warmup samples for GPU stabilization
PERFORMANCE_SAMPLES = None    # Performance test samples (None = all samples)
VAL_SAMPLES = 5000            # Total validation samples


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
        time.sleep(0.005)  # Ensure state reset
    
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


def run_benchmark(
    model_class,
    model_path: str,
    dataset_path: str,
    output_path: str,
    num_samples: int = None,
    random_seed: int = None
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
        import random
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
    
    dataset = load_from_disk(dataset_path)
    total_samples = num_samples or min(VAL_SAMPLES, len(dataset))
    
    # Performance test samples
    if PERFORMANCE_SAMPLES is None:
        perf_samples = total_samples  # Test all samples
    else:
        perf_samples = min(PERFORMANCE_SAMPLES, total_samples)
    
    print(f"Total samples: {total_samples}")
    print(f"Performance test samples: {perf_samples}")
    
    # Prepare samples (fixed order: first N samples)
    samples = []
    for i in range(total_samples):
        item = dataset[i]
        samples.append({
            "question_id": item.get("question_id", i),
            "image": item["image"],
            "question": item["question"],
        })
    
    results = {
        "system_info": get_system_info(),
        "performance": {},
        "answers": []
    }
    
    # Load and test participant model
    print("\n" + "=" * 60)
    print("Running Model Benchmark")
    print("=" * 60)
    
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

    # Start chrome trace recording (deferred from init)
    if hasattr(model, 'start_trace'):
        model.start_trace()

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
        avg_ttft = sum(ttfts) / len(ttfts) * 1000  # Convert to ms
        avg_throughput = sum(throughputs) / len(throughputs)
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
    
    # Print summary
    if len(ttfts) > 0:
        print(f"\n✓ TTFT: {avg_ttft:.2f} ms")
        print(f"✓ Throughput: {avg_throughput:.2f} tokens/sec")
    else:
        print(f"\n✗ All samples failed!")
    
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
    print(f"   Samples evaluated: {total_samples}")
    print(f"\n💾 Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="AICAS 2026 Benchmark Tool")
    parser.add_argument("--model-path", type=str, default="./Qwen3-VL-2B-Instruct", help="Path to model weights")
    parser.add_argument("--dataset-path", type=str, default="./data", help="Path to validation dataset")
    parser.add_argument("--output", type=str, default="result.json", help="Output JSON file path")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--random-seed", type=int, default=None, help="Random seed for reproducibility")
    
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
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
