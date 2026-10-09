#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0+PTX")

ext = load(
    name="qwen3vl_vision_qkv_rope_v165c",
    sources=[
        str(ROOT / "csrc" / "torch_bindings_v165c_vision.cpp"),
        str(ROOT / "csrc" / "vision_qkv_rope_v165c.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "--ptxas-options=-v",
    ],
    build_directory=str(ROOT / "build_vision_qkv_rope_v165c"),
    verbose=True,
)

print(ext.__file__)
