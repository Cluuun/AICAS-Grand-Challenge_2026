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

if TRITON_AVAILABLE:
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        try:
            from triton.language.extra.cuda.libdevice import rsqrt
        except ModuleNotFoundError:
            from triton.language.math import rsqrt
else:
    rsqrt = None


MAX_TRITON_RMS_NORM_COLS = 65536
DEFAULT_MIN_TRITON_RMS_NORM_ROWS = int(os.environ.get("JUNKRAT_TRITON_RMS_NORM_MIN_ROWS", "1"))
DEFAULT_MIN_TRITON_FUSED_ADD_RMS_NORM_ROWS = int(os.environ.get("JUNKRAT_TRITON_FUSED_ADD_RMS_NORM_MIN_ROWS", "1"))


def _calculate_rms_norm_launch_config(n_cols: int) -> tuple[int, int]:
    block_size = triton.next_power_of_2(n_cols)
    if block_size > MAX_TRITON_RMS_NORM_COLS:
        raise RuntimeError(
            f"Cannot launch Triton RMSNorm kernel because n_cols={n_cols} exceeds {MAX_TRITON_RMS_NORM_COLS}."
        )

    num_warps = 4
    if block_size >= 4096:
        num_warps = 16
    elif block_size >= 2048:
        num_warps = 8
    return block_size, num_warps


if TRITON_AVAILABLE:

    @triton.jit
    def _rms_norm_inference_kernel(
        y_ptr,
        y_row_stride,
        x_ptr,
        x_row_stride,
        w_ptr,
        w_row_stride,
        n_cols,
        eps,
        elementwise_affine: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        col_offsets = tl.arange(0, block_size)
        mask = col_offsets < n_cols

        y_base = y_ptr + row_idx * y_row_stride
        x_base = x_ptr + row_idx * x_row_stride

        x_row = tl.load(x_base + col_offsets, mask=mask, other=0)
        x_row_dtype = x_row.dtype
        x_row_fp = x_row.to(tl.float32)

        mean_square = tl.sum(x_row_fp * x_row_fp, axis=0) / n_cols
        rstd = rsqrt(mean_square + eps)
        x_row_fp = x_row_fp * rstd

        if elementwise_affine:
            w_row = tl.load(w_ptr + col_offsets, mask=mask, other=0)
            y_row = x_row_fp.to(x_row_dtype) * w_row
        else:
            y_row = x_row_fp.to(x_row_dtype)

        tl.store(y_base + col_offsets, y_row, mask=mask)


    @triton.jit
    def _block_rms_norm_inference_kernel(
        y_ptr,
        y_row_stride,
        x_ptr,
        x_row_stride,
        w_ptr,
        w_row_stride,
        n_rows,
        n_cols,
        eps,
        elementwise_affine: tl.constexpr,
        block_size: tl.constexpr,
        block_row: tl.constexpr,
    ):
        row_idx = tl.program_id(0) * block_row + tl.arange(0, block_row)
        col_offsets = tl.arange(0, block_size)
        row_mask = row_idx < n_rows
        col_mask = col_offsets < n_cols

        x_row = tl.load(
            x_ptr + row_idx[:, None] * x_row_stride + col_offsets[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0,
        )
        x_row_dtype = x_row.dtype
        x_row_fp = x_row.to(tl.float32)

        mean_square = tl.sum(x_row_fp * x_row_fp, axis=1) / n_cols
        rstd = rsqrt(mean_square + eps)
        x_row_fp = x_row_fp * rstd[:, None]

        if elementwise_affine:
            w_row = tl.load(w_ptr + col_offsets, mask=col_mask, other=0)
            y_row = x_row_fp.to(x_row_dtype) * w_row[None, :]
        else:
            y_row = x_row_fp.to(x_row_dtype)

        tl.store(
            y_ptr + row_idx[:, None] * y_row_stride + col_offsets[None, :],
            y_row,
            mask=row_mask[:, None] & col_mask[None, :],
        )


    @triton.jit
    def _fused_add_rms_norm_inference_kernel(
        y_ptr,
        y_row_stride,
        s_ptr,
        s_row_stride,
        x_ptr,
        x_row_stride,
        r_ptr,
        r_row_stride,
        w_ptr,
        w_row_stride,
        n_cols,
        eps,
        block_size: tl.constexpr,
    ):
        row_idx = tl.program_id(0).to(tl.int64)
        col_offsets = tl.arange(0, block_size)
        mask = col_offsets < n_cols

        y_base = y_ptr + row_idx * y_row_stride
        s_base = s_ptr + row_idx * s_row_stride
        x_base = x_ptr + row_idx * x_row_stride
        r_base = r_ptr + row_idx * r_row_stride

        x_row = tl.load(x_base + col_offsets, mask=mask, other=0)
        r_row = tl.load(r_base + col_offsets, mask=mask, other=0)
        s_row = x_row + r_row
        tl.store(s_base + col_offsets, s_row, mask=mask)

        s_row_dtype = s_row.dtype
        s_row_fp = s_row.to(tl.float32)
        w_row = tl.load(w_ptr + col_offsets, mask=mask, other=0)

        mean_square = tl.sum(s_row_fp * s_row_fp, axis=0) / n_cols
        rstd = rsqrt(mean_square + eps)
        y_row = s_row_fp * rstd
        y_row = y_row.to(s_row_dtype) * w_row

        tl.store(y_base + col_offsets, y_row, mask=mask)


def _num_rows(x: torch.Tensor) -> int:
    return x.numel() // x.shape[-1]


def _has_supported_layout(x: torch.Tensor) -> bool:
    return x.ndim >= 1 and x.is_contiguous() and x.stride(-1) == 1


def rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    normalized_shape = (x.shape[-1],)
    return F.rms_norm(x, normalized_shape, weight, eps)


def fused_add_rms_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = residual + x
    normalized_shape = (summed.shape[-1],)
    return F.rms_norm(summed, normalized_shape, weight, eps), summed


def can_use_triton_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    min_rows: int = DEFAULT_MIN_TRITON_RMS_NORM_ROWS,
) -> bool:
    return (
        TRITON_AVAILABLE
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.numel() > 0
        and _has_supported_layout(x)
        and _num_rows(x) >= min_rows
        and (
            weight is None
            or (
                weight.is_cuda
                and weight.ndim == 1
                and weight.shape[0] == x.shape[-1]
                and weight.dtype == x.dtype
                and weight.is_contiguous()
            )
        )
    )


def rms_norm_inference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    eps: float = 1e-6,
    min_rows: int = DEFAULT_MIN_TRITON_RMS_NORM_ROWS,
) -> torch.Tensor:
    if not can_use_triton_rms_norm(x, weight, min_rows=min_rows):
        return rms_norm_reference(x, weight, eps=eps)

    shape = x.shape
    n_cols = shape[-1]
    x_2d = x.view(-1, n_cols)
    n_rows = x_2d.shape[0]
    block_size, num_warps = _calculate_rms_norm_launch_config(n_cols)

    y = torch.empty_like(x_2d)
    elementwise_affine = weight is not None
    weight_ptr = weight if weight is not None else torch.empty(1, device=x.device, dtype=x.dtype)

    if block_size > 256 or n_rows < 4096 * 8:
        _rms_norm_inference_kernel[(n_rows,)](
            y,
            y.stride(0),
            x_2d,
            x_2d.stride(0),
            weight_ptr,
            weight_ptr.stride(0),
            n_cols,
            eps,
            elementwise_affine=elementwise_affine,
            block_size=block_size,
            num_warps=num_warps,
        )
    else:
        block_row = 16
        _block_rms_norm_inference_kernel[(triton.cdiv(n_rows, block_row),)](
            y,
            y.stride(0),
            x_2d,
            x_2d.stride(0),
            weight_ptr,
            weight_ptr.stride(0),
            n_rows,
            n_cols,
            eps,
            elementwise_affine=elementwise_affine,
            block_size=block_size,
            block_row=block_row,
            num_warps=num_warps,
        )

    return y.view(*shape)


def can_use_triton_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    min_rows: int = DEFAULT_MIN_TRITON_FUSED_ADD_RMS_NORM_ROWS,
) -> bool:
    return (
        TRITON_AVAILABLE
        and x.is_cuda
        and residual.is_cuda
        and x.device == residual.device == weight.device
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.dtype == residual.dtype == weight.dtype
        and x.shape == residual.shape
        and x.numel() > 0
        and _has_supported_layout(x)
        and _has_supported_layout(residual)
        and _num_rows(x) >= min_rows
        and weight.ndim == 1
        and weight.shape[0] == x.shape[-1]
        and weight.is_contiguous()
    )


def fused_add_rms_norm_inference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    min_rows: int = DEFAULT_MIN_TRITON_FUSED_ADD_RMS_NORM_ROWS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not can_use_triton_fused_add_rms_norm(x, residual, weight, min_rows=min_rows):
        return fused_add_rms_norm_reference(x, residual, weight, eps=eps)

    shape = x.shape
    n_cols = shape[-1]
    x_2d = x.view(-1, n_cols)
    residual_2d = residual.view(-1, n_cols)
    n_rows = x_2d.shape[0]
    block_size, num_warps = _calculate_rms_norm_launch_config(n_cols)

    y = torch.empty_like(x_2d)
    summed = torch.empty_like(x_2d)
    _fused_add_rms_norm_inference_kernel[(n_rows,)](
        y,
        y.stride(0),
        summed,
        summed.stride(0),
        x_2d,
        x_2d.stride(0),
        residual_2d,
        residual_2d.stride(0),
        weight,
        weight.stride(0),
        n_cols,
        eps,
        block_size=block_size,
        num_warps=num_warps,
    )
    return y.view(*shape), summed.view(*shape)


def prewarm_triton_rms_norm(
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_dims: tuple[int, ...],
    rows: int | None = None,
) -> None:
    if not TRITON_AVAILABLE or device.type != "cuda":
        return

    warm_rows = max(rows or DEFAULT_MIN_TRITON_RMS_NORM_ROWS, 1)
    with torch.inference_mode():
        for hidden_dim in sorted(set(hidden_dims)):
            dummy_input = torch.zeros((warm_rows, hidden_dim), device=device, dtype=dtype)
            dummy_weight = torch.ones((hidden_dim,), device=device, dtype=dtype)
            rms_norm_inference(dummy_input, dummy_weight, min_rows=warm_rows)
    torch.cuda.synchronize(device=device)


def prewarm_triton_fused_add_rms_norm(
    *,
    device: torch.device,
    dtype: torch.dtype,
    hidden_dims: tuple[int, ...],
    rows: int | None = None,
) -> None:
    if not TRITON_AVAILABLE or device.type != "cuda":
        return

    warm_rows = max(rows or DEFAULT_MIN_TRITON_FUSED_ADD_RMS_NORM_ROWS, 1)
    with torch.inference_mode():
        for hidden_dim in sorted(set(hidden_dims)):
            dummy_x = torch.zeros((warm_rows, hidden_dim), device=device, dtype=dtype)
            dummy_residual = torch.zeros_like(dummy_x)
            dummy_weight = torch.ones((hidden_dim,), device=device, dtype=dtype)
            fused_add_rms_norm_inference(
                dummy_x,
                dummy_residual,
                dummy_weight,
                min_rows=warm_rows,
            )
    torch.cuda.synchronize(device=device)
