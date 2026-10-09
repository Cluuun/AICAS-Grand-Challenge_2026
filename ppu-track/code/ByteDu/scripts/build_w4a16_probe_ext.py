#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "w4a16_probe_ext"
OUT_PATH = ROOT / "prebuilt" / "w4a16_probe_ext.so"


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    module = load(
        name="w4a16_probe_ext",
        sources=[
            str(ROOT / "csrc" / "w4a16_probe_ext.cpp"),
            str(ROOT / "csrc" / "w4a16_probe_kernel.cu"),
        ],
        build_directory=str(BUILD_DIR),
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-Xptxas=-v",
            "-gencode=arch=compute_80,code=sm_80",
        ],
        with_cuda=True,
        is_python_module=True,
        verbose=True,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(module.__file__).resolve(), OUT_PATH)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
