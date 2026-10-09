import os

import torch
import torch.nn.functional as F

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/junkrat_triton_cache")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


MAX_TRITON_FUSED_SIZE = 65536
DEFAULT_MIN_TRITON_SWIGLU_ROWS = int(os.environ.get("JUNKRAT_TRITON_SWIGLU_MIN_ROWS", "1"))


def _is_hip_backend() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _calculate_swiglu_launch_config(n_cols: int) -> tuple[int, int]:
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_TRITON_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton SwiGLU kernel because n_cols={n_cols} exceeds {MAX_TRITON_FUSED_SIZE}."
        )

    num_warps = 4
    if block_size >= 32768:
        num_warps = 32 if not _is_hip_backend() else 16
    elif block_size >= 8192:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    return block_size, num_warps


if TRITON_AVAILABLE:

    @triton.jit
    def _triton_silu(x):
        return x * tl.sigmoid(x)


    @triton.jit
    def _swiglu_inference_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        row_stride,
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)

        gate_ptr += row_idx * row_stride
        up_ptr += row_idx * row_stride
        out_ptr += row_idx * row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        gate = tl.load(gate_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(up_ptr + col_offsets, mask=mask, other=0)
        out = _triton_silu(gate).to(up.dtype) * up
        tl.store(out_ptr + col_offsets, out, mask=mask)


    @triton.jit
    def _swiglu_packed_kernel(
        gate_up_ptr,
        out_ptr,
        gate_up_stride0,
        out_stride0,
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """SwiGLU on packed [gate | up] tensor, avoiding .contiguous() copies."""
        row_idx = tl.program_id(0).to(tl.int64)

        base = gate_up_ptr + row_idx * gate_up_stride0
        out_base = out_ptr + row_idx * out_stride0

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        gate = tl.load(base + col_offsets, mask=mask, other=0).to(tl.float32)
        up = tl.load(base + n_cols + col_offsets, mask=mask, other=0)
        out = _triton_silu(gate).to(up.dtype) * up
        tl.store(out_base + col_offsets, out, mask=mask)


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def _num_rows(x: torch.Tensor) -> int:
    return x.numel() // x.shape[-1]


def can_use_triton_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    min_rows: int = DEFAULT_MIN_TRITON_SWIGLU_ROWS,
) -> bool:
    return (
        TRITON_AVAILABLE
        and gate.is_cuda
        and up.is_cuda
        and gate.dtype in (torch.float16, torch.bfloat16)
        and gate.dtype == up.dtype
        and gate.shape == up.shape
        and gate.numel() > 0
        and _num_rows(gate) >= min_rows
    )


def _validate_out_tensor(out: torch.Tensor, reference: torch.Tensor) -> None:
    if out.shape != reference.shape:
        raise ValueError(f"Expected out tensor shape {reference.shape}, got {out.shape}")
    if out.dtype != reference.dtype:
        raise ValueError(f"Expected out tensor dtype {reference.dtype}, got {out.dtype}")
    if out.device != reference.device:
        raise ValueError(f"Expected out tensor device {reference.device}, got {out.device}")


def swiglu_inference(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    min_rows: int = DEFAULT_MIN_TRITON_SWIGLU_ROWS,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not can_use_triton_swiglu(gate, up, min_rows=min_rows):
        result = swiglu_reference(gate, up)
        if out is None:
            return result
        _validate_out_tensor(out, result)
        out.copy_(result)
        return out

    n_cols = gate.shape[-1]
    block_size, num_warps = _calculate_swiglu_launch_config(n_cols)

    gate_2d = gate.contiguous().view(-1, n_cols)
    up_2d = up.contiguous().view(-1, n_cols)
    if out is None:
        out_2d = torch.empty_like(gate_2d)
    else:
        _validate_out_tensor(out, gate)
        out_2d = out.view(-1, n_cols)

    _swiglu_inference_kernel[(gate_2d.shape[0],)](
        gate_2d,
        up_2d,
        out_2d,
        gate_2d.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out_2d.view_as(gate)


def swiglu_packed_inference(
    gate_up: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """SwiGLU on packed [gate | up] tensor without .contiguous() copies.

    gate_up: shape [..., 2 * intermediate_size], contiguous
    Returns: shape [..., intermediate_size]
    """
    total_cols = gate_up.shape[-1]
    n_cols = total_cols // 2

    if (
        not TRITON_AVAILABLE
        or not gate_up.is_cuda
        or gate_up.dtype not in (torch.float16, torch.bfloat16)
        or gate_up.numel() == 0
        or os.environ.get("JUNKRAT_DISABLE_TRITON_SWIGLU") == "1"
    ):
        gate, up = gate_up.chunk(2, dim=-1)
        result = F.silu(gate) * up
        if out is not None:
            out.copy_(result)
            return out
        return result

    block_size, num_warps = _calculate_swiglu_launch_config(n_cols)
    gate_up_2d = gate_up.view(-1, total_cols)
    n_rows = gate_up_2d.shape[0]

    if out is None:
        out_2d = torch.empty((n_rows, n_cols), device=gate_up.device, dtype=gate_up.dtype)
    else:
        out_2d = out.view(-1, n_cols)

    _swiglu_packed_kernel[(n_rows,)](
        gate_up_2d,
        out_2d,
        gate_up_2d.stride(0),
        out_2d.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    orig_shape = list(gate_up.shape)
    orig_shape[-1] = n_cols
    return out_2d.view(orig_shape)


def prewarm_triton_swiglu(
    *,
    device: torch.device,
    dtype: torch.dtype,
    intermediate_size: int,
    rows: int | None = None,
) -> None:
    if not TRITON_AVAILABLE or device.type != "cuda":
        return

    warm_rows = max(rows or DEFAULT_MIN_TRITON_SWIGLU_ROWS, 1)
    dummy_gate = torch.zeros((warm_rows, intermediate_size), device=device, dtype=dtype)
    dummy_up = torch.zeros_like(dummy_gate)
    with torch.inference_mode():
        swiglu_inference(dummy_gate, dummy_up)
    dummy_gate_up = torch.zeros((warm_rows, 2 * intermediate_size), device=device, dtype=dtype)
    with torch.inference_mode():
        swiglu_packed_inference(dummy_gate_up)
    torch.cuda.synchronize(device=device)
