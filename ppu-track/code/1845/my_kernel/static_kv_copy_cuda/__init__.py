from __future__ import annotations

import os
import sys
from pathlib import Path

import torch  # noqa: F401 - load torch shared libraries before importing the extension


_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import static_kv_copy_cuda_ext as _ext
except Exception as exc:  # pragma: no cover - optional CUDA extension
    _ext = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def available() -> bool:
    return _ext is not None


def copy_full_prompt_kv(key_src, value_src, key_dst, value_dst, seq_len: int) -> bool:
    if _ext is None:
        if os.getenv("AICAS_STATIC_KV_COPY_DEBUG", "0") == "1":
            print(f"[static-kv-copy] extension unavailable: {_IMPORT_ERROR}", flush=True)
        return False
    _ext.copy_full_prompt_kv(key_src, value_src, key_dst, value_dst, int(seq_len))
    return True
