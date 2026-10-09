import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_swiglu_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    stride_gate_row,
    stride_gate_col,
    stride_up_row,
    stride_up_col,
    stride_out_row,
    stride_out_col,
    n_cols,
    BLOCK_COL: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    col_offsets = col_block * BLOCK_COL + tl.arange(0, BLOCK_COL)
    mask = col_offsets < n_cols

    gate_ptrs = gate_ptr + row * stride_gate_row + col_offsets * stride_gate_col
    up_ptrs = up_ptr + row * stride_up_row + col_offsets * stride_up_col
    out_ptrs = out_ptr + row * stride_out_row + col_offsets * stride_out_col

    gate_in = tl.load(gate_ptrs, mask=mask, other=0.0)
    up = tl.load(up_ptrs, mask=mask, other=0.0)
    gate = gate_in.to(tl.float32)
    swish = (gate * tl.sigmoid(gate)).to(gate_in.dtype)
    out = swish * up

    tl.store(out_ptrs, out, mask=mask)


def reference_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    out: torch.Tensor | None = None,
    block_size: int = 512,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("fused_swiglu requires CUDA tensors")
    if gate.shape != up.shape:
        raise ValueError("gate and up tensors must have the same shape")
    if gate.dtype != up.dtype:
        raise ValueError("gate and up tensors must have the same dtype")
    if out is not None:
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.shape != gate.shape:
            raise ValueError("out must have the same shape as gate/up")
        if out.dtype != gate.dtype:
            raise ValueError("out must have the same dtype as gate/up")

    block_size = int(block_size)
    num_warps = int(num_warps)
    num_stages = int(num_stages)
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if num_warps <= 0:
        raise ValueError("num_warps must be > 0")
    if num_stages <= 0:
        raise ValueError("num_stages must be > 0")

    if gate.numel() == 0:
        return gate if out is None else out

    cols = int(gate.shape[-1])
    rows = int(gate.numel() // cols)
    gate2d = gate.reshape(rows, cols)
    up2d = up.reshape(rows, cols)
    if out is None:
        out = torch.empty_like(gate)
    out2d = out.reshape(rows, cols)

    _fused_swiglu_kernel[(rows, triton.cdiv(cols, block_size))](
        gate2d,
        up2d,
        out2d,
        gate2d.stride(0),
        gate2d.stride(1),
        up2d.stride(0),
        up2d.stride(1),
        out2d.stride(0),
        out2d.stride(1),
        cols,
        BLOCK_COL=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
