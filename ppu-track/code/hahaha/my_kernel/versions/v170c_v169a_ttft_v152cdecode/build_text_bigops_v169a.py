#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sysconfig
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build_text_bigops_v169a"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0+PTX")

ext = load(
    name="qwen3vl_text_bigops_v169a",
    sources=[
        str(ROOT / "csrc" / "torch_bindings_v169a_text.cpp"),
        str(ROOT / "csrc" / "text_bigops_v169a.cu"),
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

ext_path = Path(ext.__file__).resolve()
default_target = ROOT / ext_path.name
if ext_path != default_target.resolve():
    shutil.copy2(ext_path, default_target)

ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
canonical_target = ROOT / f"qwen3vl_text_bigops_v169a{ext_suffix}"
if default_target.resolve() != canonical_target.resolve():
    shutil.copy2(default_target, canonical_target)

print(canonical_target)
