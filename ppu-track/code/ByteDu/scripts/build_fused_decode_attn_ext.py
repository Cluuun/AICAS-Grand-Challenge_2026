#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "fused_decode_attn"
OUT_PATH = ROOT / "prebuilt" / "fused_decode_attn.so"


def main() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    module = load(
        name="fused_decode_attn",
        sources=[str(ROOT / "csrc" / "fused_decode_attn.cu")],
        build_directory=str(BUILD_DIR),
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
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
