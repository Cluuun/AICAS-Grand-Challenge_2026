# 2026 AICASGC — PPU Track — Season 2 — Submission Guide


This document describes the **ZIP layout**, **dependency notes** (`requirements.txt`), and **current evaluation-machine software stack** used when the Tianchi evaluation Worker pulls, extracts, and runs your submission. Use it to align your local environment. **The deployed Worker is authoritative**; if the organizers change anything, follow their latest notice.

---

## What to put in the submission ZIP (recommended layout)

The evaluation Worker extracts the ZIP and invokes your **`evaluation_wrapper.py`** using the official **`benchmark.py`**. Below is a **minimal viable** layout; files may also live in **subdirectories** (`evaluation_wrapper.py` is searched across the entire extracted tree).

```text
your_submission.zip
├── evaluation_wrapper.py      # Required: implement VLMModel
├── requirements.txt           # Optional: omit or match official exactly
├── my_kernel/                 # Optional: custom ops, helpers (.py / prebuilt .so, etc.)
│   ├── __init__.py
│   └── ...
└── README.md                  # Optional: deps & build notes (not read by scripts; for humans)
```

| Path / file | Required? | Notes |
|-------------|-----------|-------|
| `evaluation_wrapper.py` | **Yes** | Implement `VLMModel`. |
| `requirements.txt` | Optional | Omit entirely or keep identical to the official list (see organizer materials). |
| Other `.py`, assets, **prebuilt** extensions (e.g. `.so`) | Optional | Merged into the job working directory; load via `load_library` / `import` / `ctypes` from `evaluation_wrapper`. |
| `benchmark.py` | **Do not upload** | Any `benchmark.py` inside the ZIP **is ignored**; the official benchmark is always used. |

**Do not** pack large unrelated files or binaries that are not needed for evaluation; they slow download and extraction.

---

## 1. How to submit

### 1.1 Archive format

- The submission must be a **ZIP archive** (`.zip`); do not use rar/7z, etc.
- Paths must be safe (avoid Zip Slip; no `..` or paths that escape the target directory).

### 1.2 Required file: `evaluation_wrapper.py`

- The evaluator searches for **`evaluation_wrapper.py`** under the extract directory (including subdirectories).
- It must implement **`VLMModel`** (same contract as the official template) so `benchmark.py` can load it with `from evaluation_wrapper import VLMModel`.

### 1.3 Do not rely on uploading `benchmark.py`

- If the ZIP contains `benchmark.py`, it **is ignored**. Metrics and protocol follow the official `benchmark.py`.

### 1.4 Optional file: `requirements.txt`

- Optional. If you include it, follow the same rules as the official baseline (typically: omit or match exactly to avoid unexpected package changes on the evaluator).

### 1.5 Other files

- Everything else in the ZIP (after the rules above) is merged into the job working directory; keep size and necessity under control.

---

## 2. Evaluation-machine software environment

| Item | Example |
|------|---------|
| OS | Linux (x86_64) |
| GPU | NVIDIA A800 80GB PCIe |
| NVIDIA driver | 590.48.01 |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 (wheel built for CUDA 12.8) |
| CUDA runtime bundled with PyTorch | 12.8 (bound to the `torch` wheel, used for inference) |
| CUDA Toolkit (`nvcc`) | 12.8 (`nvcc` reports 12.8.61) |

Notes:

- If you develop with a **full CUDA Toolkit**, **document that the Toolkit major version matches PyTorch’s CUDA major version** (examples here use **12.8**) to avoid ABI mismatches.
- **The default evaluation flow does not compile** `.cu` sources from the ZIP automatically. **Build extensions locally** on a machine with the **same CUDA major version** as the evaluator, then submit **prebuilt `.so` (and Python glue)** in the ZIP, or rely on **Triton / official PyTorch extension wheels** from pip (no `nvcc` needed when a wheel exists).

---

## 3. FAQ

### Q1: Must my CUDA major version match the evaluation machine?

Yes.

- When building custom `.so` files, use the same CUDA major version as the evaluator to reduce ABI / runtime incompatibility risk.
- Example versions for this machine are listed in the table above:  
  - PyTorch CUDA: `12.8` (`torch 2.8.0+cu128`)  
  - CUDA Toolkit (`nvcc`): `12.8` (`V12.8.61`)

### Q2: What is the current evaluation logic? How long does one run take?

- Evaluation uses a **fixed subset of 150 samples** from the original dataset.
- One evaluation run typically takes about **15–30 minutes**.

### Q3: What metrics appear on the leaderboard? How is the preliminary score calculated?

- **Leaderboard total score** = 0.4 × *Ratio_accuracy* + 0.3 × *TTFT_improvement* + 0.3 × *Throughput_improvement*.
- **Preliminary score per team** (differs from the leaderboard formula) = 0.4 × (team accuracy / highest accuracy) + 0.3 × (TTFT improvement / highest TTFT improvement) + 0.3 × (throughput improvement / highest throughput improvement).
- The **top 16 teams** in the preliminary round advance to the finals.

## 4. 30-second pre-submit checklist

- ZIP contains `evaluation_wrapper.py`
- `VLMModel` runs end-to-end locally in a matching environment
- If using `.so`: built for Linux x86_64; CUDA major version matches the evaluator

---

If submission rules or the environment change, follow the latest documentation from the organizers / evaluation team.
