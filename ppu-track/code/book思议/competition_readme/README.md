# AICAS 2026 - Vision-Language Model Optimization Competition

## Table of Contents
- [Overview](#overview)
- [Code Structure](#code-structure)
- [Core Files](#core-files)
- [Quick Start](#quick-start)
- [Evaluation Metrics](#evaluation-metrics)
- [Competition Rules](#competition-rules)
- [Important Notes](#important-notes)
- [Submission Guidelines](#submission-guidelines)

## Overview

This competition focuses on optimizing Vision-Language Models (VLM) for inference performance. Participants are required to modify the `VLMModel` class in `evaluation_wrapper.py` to achieve better Time-To-First-Token (TTFT) and Throughput while maintaining accuracy.

## Code Structure

```
AICASGC/
├── benchmark.py              # Benchmark script (not recommended to modify)
├── evaluation_wrapper.py     # Model wrapper (participants implement optimizations here)
├── requirements.txt          # Python dependencies
├── data/                     # Validation dataset
│   ├── data-*.arrow          # Dataset files
│   ├── dataset_info.json     # Dataset metadata
│   └── state.json            # Dataset state
├── Qwen3-VL-2B-Instruct/    # Model weights directory (participants need to download)
└── README.md / README_CN.md   # Documentation
```


## Core Files

- **`benchmark.py`** - Self-testing benchmark script (⚠️ **Not recommended to modify**)
- **`evaluation_wrapper.py`** - Model wrapper where participants implement optimizations
- **`Qwen3-VL-2B-Instruct/`** - Competition model weights (participants need to download, see "Quick Start" section)
- **`data/`** - Validation dataset
- **`requirements.txt`** - Python dependencies

## Quick Start

### 0. Download Model (First Time)

The model files are large and need to be downloaded separately. Please create the model directory first, then download the model:

```bash
# Create model directory
mkdir -p Qwen3-VL-2B-Instruct

# Install huggingface_hub (if not installed)
pip install -U huggingface_hub

# Set mirror endpoint (recommended for users in China, faster download)
export HF_ENDPOINT=https://hf-mirror.com

# Download model to specified directory
huggingface-cli download \
  --resume-download \
  Qwen/Qwen3-VL-2B-Instruct \
  --local-dir ./Qwen3-VL-2B-Instruct \
  --local-dir-use-symlinks False
```

**Note:**
- Model size is approximately 4-5GB, download may take some time
- If download is interrupted, you can rerun the command and it will resume automatically (`--resume-download`)
- After download completes, the `Qwen3-VL-2B-Instruct/` folder will contain all model files
- Ensure you have sufficient disk space (at least 5GB)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Test

```bash
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result.json \
    --num-samples 100
```

### 3. Implement Your Optimizations

Edit the `VLMModel` class in `evaluation_wrapper.py`. The optimization architecture uses **modular design**, where each optimization direction corresponds to an independent method.

#### 3.1 Explore Model Structure (Optional)

Before starting optimizations, you can explore the model structure to understand optimization targets:

```python
class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        # ... load model ...
        
        # Optional: Explore model structure
        self._explore_model_structure()  # Will print model structure information
```

#### 3.2 Enable Optimization Methods

In the `__init__` method, enable/disable different optimizations by commenting/uncommenting:

```python
class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        # ... load model ...
        
        # ================================================================
        # Participant Optimization Area - Enable/disable optimization methods
        # ================================================================
        
        # 1. Vision Encoder Acceleration (optimize high-resolution image processing)
        # self._optimize_vision_encoder()
        
        # 2. KV Cache Management (optimize memory fragmentation during generation)
        # self._optimize_kv_cache()
        
        # 3. Cross-modal Connector Optimization (optimize Cross-modal Connector)
        # self._optimize_cross_modal_connector()
        
        # 4. Flash Attention Optimization
        # self._enable_flash_attention()
        
        # 5. Quantization Optimization
        # self._apply_quantization()
```

#### 3.3 Implement Optimization Code

Implement your optimization logic in each optimization method. For example, optimizing Vision Encoder:

```python
def _optimize_vision_encoder(self):
    """Find this method in evaluation_wrapper.py and implement your optimization"""
    
    # Example: Replace attention operator
    # from your_optimization import optimized_attention
    # if hasattr(self._model, 'vision_model'):
    #     for layer in self._model.vision_model.encoder.layers:
    #         layer.self_attn.forward = optimized_attention
    
    # TODO: Implement your Vision Encoder optimization
    pass
```


**Important Notes:**
- Benchmark directly calls `self.model.generate()` for performance testing
- Your optimizations should modify `self.model` or its operators via Monkey Patch in optimization methods
- All optimization methods are called in `__init__`, and optimizations take effect automatically
- The `generate()` method is optional and mainly for debugging

### 4. Test Your Optimized Model

```bash
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result_optimized.json \
    --num-samples 100
```

### 5. Generate Full Results for Submission

```bash
python benchmark.py \
    --model-path ./Qwen3-VL-2B-Instruct \
    --dataset-path ./data \
    --output result.json \
    --num-samples 5000
```

## Evaluation Metrics

The final score is calculated as:

```
Final Score = 0.4 × Accuracy + 0.3 × TTFT_Improvement + 0.3 × Throughput_Improvement
```

### Metrics Explained

- **TTFT (Time To First Token)**: Time from input preparation to first token generation (in milliseconds)
  - Includes: image encoding, text encoding, cross-modal interaction, prefill stage, first token generation
  - Baseline: ~80ms
  - Improvement = (Baseline - Your_TTFT) / Baseline

- **Throughput**: End-to-end token generation rate (tokens per second)
  - Baseline: ~55 tokens/sec
  - Improvement = (Your_Throughput - Baseline) / Baseline

- **Accuracy**: VQA accuracy on validation set (5000 samples)
  - Soft matching with multiple ground truth answers

## Competition Rules

### Critical Rules

1. **Do not modify `benchmark.py`**
   - This benchmark script is for self-testing only
   - Final evaluation will use a separate official benchmark system
   - Modifying this file may lead to inconsistencies between your local results and final evaluation results

2. **Only modify `evaluation_wrapper.py`**


3. **Maintain required properties**
   - The `VLMModel` class must expose `processor`, `model`, and `device` properties
   - Benchmark uses these properties to access the model and processor
   - The `generate()` method is optional and mainly for debugging

4. **Prohibited behaviors**
   - Do not hardcode answers
   - Do not modify the dataset
   - Do not use external APIs or services
   - All optimizations must be local and self-contained




### Optimization Directions

- Encouraged: Operator replacement and kernel optimization - Rewrite or replace standard operator implementations (such as Attention, LayerNorm, Conv2d, etc.) using Triton, CUDA C++, etc.

- Encouraged: Memory and cache optimization - Optimize KV Cache memory layout, reduce memory fragmentation, optimize GPU memory access patterns

- Encouraged: Compilation and graph optimization - Use torch.compile for computation graph optimization, custom kernel scheduling

- Encouraged: Attention mechanism optimization - Implement Flash Attention, memory-efficient attention, sparse attention

- Encouraged: Generation process optimization - Optimize decoding strategies, cache management, generation configuration parameters

**Not Permitted:**
- Using external services: Prohibited from calling external APIs, cloud services, or any functionality requiring network connection

- Data and answer cheating: Prohibited from training on test data, pre-computing answers, hardcoding outputs

- Model replacement and tampering: Participants should focus on operator-level optimization. Do not use additional datasets to train the model, change model architecture, or directly modify weight values.

- Overfitting optimization: Prohibited from using conditional branches or special processing for specific evaluation samples

- Black-box tool application: Behavior of only modifying configuration files without substantive code contributions is not recognized

- Environment manipulation: Prohibited from interfering with fair evaluation by modifying system environment, GPU frequency locking, etc.



## Important Notes

### Sample Selection

- The provided `benchmark.py` uses **fixed order** (first N samples from index 0)
- When you run `--num-samples 100`, it evaluates samples 0-99
- This ensures reproducibility for local self-testing
- **Note**: The official evaluation system used by the competition committee may employ 
  different sampling strategies (including random sampling) for final verification

### Hardware Information

The benchmark automatically records detailed hardware information:
- Python version, PyTorch version, CUDA version
- GPU name, memory, compute capability
- CPU model, cores, frequency
- System information (OS, kernel, architecture)
- PPU information (if available)

This information is saved in `result.json` under `system_info` for statistical analysis.

### Performance Measurement

- **Warmup**: 10 samples are used for GPU warmup before actual measurement
- **TTFT Measurement**: Measures time from input preparation to first token (includes all preprocessing)
- **Throughput Measurement**: Measures end-to-end generation time for 128 tokens
- **State Isolation**: GPU cache is cleared between measurements to ensure fairness

### Random Seed

- The `--random-seed` parameter only affects PyTorch's random number generator
- It does **NOT** affect sample selection order (which is always fixed)
- Use it for reproducibility of model inference randomness

### Output Format

The `result.json` file contains:
```json
{
  "system_info": {
    "timestamp": "...",
    "python_version": "...",
    "torch_version": "...",
    "cuda_version": "...",
    "gpu_name": "...",
    ...
  },
  "performance": {
    "avg_ttft_ms": 90.55,
    "avg_throughput_tokens_per_sec": 57.77
  },
  "answers": [
    {
      "question_id": 34602,
      "prediction": "your answer text here"
    },
    ...
  ]
}
```

## Submission Guidelines

### Required Files for Preliminary Submission

1. **`result.json`** - Generated by running `benchmark.py`
   - Contains predictions for all samples
   - Must include valid `performance` metrics
   - **Important**: The `result.json` uploaded to the Tianchi platform is for reference only. Final scores will be evaluated by the competition committee using standardized hardware and the official evaluation system.

2. **Your optimized code** - `evaluation_wrapper.py` containing your optimized `VLMModel` class

3. **Docker image** - Container with your optimized environment

### Evaluation Process

1. **Self-Testing**: Use the provided `benchmark.py` to test your optimizations locally
2. **Submission**: Upload your `result.json` to the Tianchi platform (for reference only)
3. **Official Evaluation**: The competition committee will evaluate your code using:
   - Docker image submission
   - Standardized hardware environment
   - Official evaluation code
   - Full validation set with random sampling for verification
4. **Final Ranking**: Based on the final score calculated by the official evaluation system



## Good Luck!

We hope you will focus on operator-level optimization, kernel replacement, and efficient memory management. Remember: accuracy and speed are equally important! Good luck!
