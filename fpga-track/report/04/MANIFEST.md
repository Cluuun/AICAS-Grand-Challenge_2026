# Source Code Package

This directory collects the source code used for the AICAS KV260 submission.

## RTL Source

Path:

- `rtl/VersaVLM/`

Copied from:

- `/mnt/c/VersaVLM`

Source git commit:

- `4cba50aee4fde262a5350d3c088507f391d9349a`

Notes:

- Copied as source, excluding only `.git/`.
- Contains the prefill and decode RTL trees.

## Host Inference Framework

Path:

- `host_inference/llama.cpp-kv260-20260407/`

Copied from:

- `/home/gugugu/work/llama.cpp-kv260-20260407`

Source git commit:

- `bf5e05be8ba6710c959e4a400c586ec00e0ae4ca`

Notes:

- The original working tree is about 51 GB and is too large to include directly.
- This is a source-focused snapshot. It keeps the llama.cpp codebase, KV260 NPU backend, `npuruntime`, AICAS scripts, CMake/build scripts, and model conversion/quantization tools.
- Generated files and large runtime artifacts were excluded, including `.git/`, build directories, virtual environments, model files, calibration/output data, result logs, core dumps, and compiled binaries/libraries.
- The runnable overlay and driver binaries are packaged separately under `../01_bitstream_overlay/`.
- Historical submission payloads and board-ready binary packages were also removed from this source snapshot to keep this directory source-only.

Key excluded categories:

- `build*/`
- `AICAS/venv/`, `AICAS/.venv-acc-eval/`
- `AICAS/output/`, `AICAS/data/`, `AICAS/gguf/`, `AICAS/kv260_results/`
- `AICAS2026/*/results/`
- `AICAS2026/*/payload/gguf/`
- historical payload directories such as `AICAS2026/V1/`, `AICAS2026/V3/`, `AICAS2026/VersaVLM/`, and `AICAS2026/overlay/`
- `models/`, `tmp/`, `outputs/`, `artifacts/`
- `*.gguf`, `*.safetensors`, `*.pt`, `*.pth`, `*.onnx`
- compiled objects/libraries/modules such as `*.o`, `*.a`, `*.so`, `*.ko`
- ELF test/benchmark binaries under `npuruntime/`
