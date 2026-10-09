from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load

_EXT = None
_LOAD_TRIED = False
_WORKSPACES = {}
_WORKSPACE_BYTES = 4 * 1024 * 1024


def _load_ext():
    global _EXT, _LOAD_TRIED
    if _LOAD_TRIED:
        return _EXT
    _LOAD_TRIED = True
    try:
        src = Path(__file__).resolve().parent / "cpp_ext" / "cublaslt_ext.cpp"
        _EXT = load(
            name="aicas_cublaslt_epilogue_ext",
            sources=[str(src)],
            extra_include_paths=["/usr/local/cuda/include"],
            extra_cflags=["-O3", "-std=c++17"],
            extra_ldflags=[
                "-L/usr/local/cuda/lib64",
                "-Wl,-rpath,/usr/local/cuda/lib64",
                "-lcublasLt",
                "-lcublas",
                "-lcudart",
            ],
            verbose=False,
        )
    except Exception as exc:
        print(f"[cublaslt_epilogue] extension unavailable: {type(exc).__name__}: {exc}")
        _EXT = None
    return _EXT


def cublaslt_matmul_bias(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not a.is_cuda or not b.is_cuda or not out.is_cuda:
        return None
    if a.dim() != 2 or b.dim() != 2 or out.dim() != 2:
        return None
    if a.dtype not in (torch.float16, torch.bfloat16):
        return None
    if a.dtype != b.dtype or a.dtype != out.dtype:
        return None
    if a.shape[1] != b.shape[0] or out.shape != (a.shape[0], b.shape[1]):
        return None
    if bias is not None and (not bias.is_cuda or bias.dtype != out.dtype or bias.numel() != out.shape[1]):
        return None
    ext = _load_ext()
    if ext is None:
        return None
    try:
        device = a.device
        workspace = _WORKSPACES.get(device)
        if workspace is None:
            workspace = torch.empty((_WORKSPACE_BYTES,), dtype=torch.uint8, device=device)
            _WORKSPACES[device] = workspace
        return ext.matmul_bias(
            a.contiguous() if not a.is_contiguous() else a,
            b.contiguous() if not b.is_contiguous() else b,
            bias.contiguous() if bias is not None and not bias.is_contiguous() else bias,
            out,
            workspace,
        )
    except Exception as exc:
        print(f"[cublaslt_epilogue] matmul_bias failed: {type(exc).__name__}: {exc}")
        return None
