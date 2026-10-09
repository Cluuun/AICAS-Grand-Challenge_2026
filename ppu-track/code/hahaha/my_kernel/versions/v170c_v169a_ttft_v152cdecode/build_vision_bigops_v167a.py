#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build_vision_bigops_v167a"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0+PTX")

ext = load(
    name="qwen3vl_vision_bigops_v167a",
    sources=[
        str(ROOT / "csrc" / "torch_bindings_v167a_vision.cpp"),
        str(ROOT / "csrc" / "vision_bigops_v167a.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--ptxas-options=-v",
    ],
    build_directory=str(BUILD_DIR),
    verbose=True,
)

print(ext.__file__)
