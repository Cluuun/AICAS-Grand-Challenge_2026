#!/usr/bin/env python3
"""Build the active v173f decode extension: qwen3vl_megaqwen_v152c.

This script is included for source-to-binary traceability of the clean
submission package. The official submission uses the prebuilt .so and does not
JIT-compile at evaluation time.
"""

from __future__ import annotations

import os
import shutil
import sysconfig
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main() -> None:
    version_dir = Path(__file__).resolve().parent
    csrc_dir = version_dir / "csrc"
    build_dir = version_dir / "build_decode_v152c"
    build_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0+PTX"
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        major, minor = torch.cuda.get_device_capability(0)
        print(f"Arch:    sm_{major}{minor}")
    print(f"TORCH_CUDA_ARCH_LIST: {os.environ['TORCH_CUDA_ARCH_LIST']}")

    module = load(
        name="qwen3vl_megaqwen_v152c",
        sources=[
            str(csrc_dir / "qwen3vl_megaqwen_v152c.cu"),
            str(csrc_dir / "torch_bindings_v152c.cpp"),
            str(version_dir / "sm_profiler.cu"),
        ],
        extra_cflags=["-O3", "-std=c++17", "-I" + str(version_dir)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-std=c++17",
            "-I" + str(version_dir),
        ],
        build_directory=str(build_dir),
        verbose=True,
    )

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output_path = version_dir / f"qwen3vl_megaqwen_v152c{ext_suffix}"
    shutil.copy2(Path(module.__file__), output_path)
    print("Build succeeded")
    print(f"Extension: {module.__file__}")
    print(f"Copied to: {output_path}")


if __name__ == "__main__":
    main()
