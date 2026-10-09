from __future__ import annotations

import shutil
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build" / "native_sage_decode_ext"
BUILD_DIR.mkdir(parents=True, exist_ok=True)

module = load(
    name="native_sage_decode_ext",
    sources=[
        str(ROOT / "native_sage_decode_ext.cpp"),
        str(ROOT / "native_sage_decode_kernels.cu"),
    ],
    build_directory=str(BUILD_DIR),
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)

target = ROOT / "native_sage_decode_ext.so"
shutil.copy2(module.__file__, target)
print(target)
