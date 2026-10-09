import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fast_swiglu_inplace_decode_kernel(
    x_ptr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    gate_idx = offsets
    up_idx = offsets + D

    gate = tl.load(x_ptr + gate_idx)
    up = tl.load(x_ptr + up_idx)

    gate_f32 = gate.to(tl.float32)
    sig = tl.sigmoid(gate_f32)
    out = (gate_f32 * sig * up.to(tl.float32)).to(gate.dtype)

    tl.store(x_ptr + gate_idx, out)


def fused_swiglu_decode(gate_up):
    D = gate_up.shape[-1] // 2
    block_size = 1024

    grid = (triton.cdiv(D, block_size),)

    fast_swiglu_inplace_decode_kernel[grid](
        gate_up,
        D,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=1,
    )

    return gate_up[:, :, :D]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=1),
    ],
    key=["num_rows_bucket"],
)
@triton.jit
def _swiglu_prefill_kernel(
    x_ptr,
    out_ptr,
    num_rows_bucket,
    BLOCK_D: tl.constexpr,
):
    D: tl.constexpr = 6144
    GATE_UP_D: tl.constexpr = 12288

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    d_offs = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)

    x_row = x_ptr + pid_row * GATE_UP_D
    gate = tl.load(x_row + d_offs)
    up = tl.load(x_row + D + d_offs)

    gate_f32 = gate.to(tl.float32)
    sig = tl.sigmoid(gate_f32)
    result = (gate_f32 * sig * up.to(tl.float32)).to(gate.dtype)

    tl.store(out_ptr + pid_row * D + d_offs, result)


def swiglu_eager_prefill(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def fused_swiglu_prefill(gate_up):
    orig_shape = gate_up.shape
    x_2d = gate_up.view(-1, 12288)
    num_rows = x_2d.shape[0]
    num_rows_bucket = ((num_rows + 127) // 128) * 128
    out = torch.empty(num_rows, 6144, dtype=gate_up.dtype, device=gate_up.device)

    def grid(meta):
        return (num_rows, 6144 // meta["BLOCK_D"])

    _swiglu_prefill_kernel[grid](x_2d, out, num_rows_bucket)
    return out.view(*orig_shape[:-1], 6144)


__all__ = [
    "fused_swiglu_decode",
    "fused_swiglu_prefill",
    "swiglu_eager_prefill",
]


if __name__ == "__main__":
    try:
        from aicas2026gc.autotuner import RuntimeAutotuner
    except ModuleNotFoundError:
        autotuner_path = Path(__file__).resolve().parents[1] / "autotuner.py"
        spec = importlib.util.spec_from_file_location("swiglu_autotuner", autotuner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        RuntimeAutotuner = module.RuntimeAutotuner

    device = "cuda"
    dtype = torch.float16

    prefill_gate_up = torch.randn(1, 512, 12288, device=device, dtype=dtype)
    prefill_tuner = RuntimeAutotuner(name="swiglu_prefill", fallback="eager", strict=False, atol=1e-2, rtol=1e-2)
    prefill_candidates = {
        "eager": swiglu_eager_prefill,
        "triton": fused_swiglu_prefill,
    }
    prefill_out = prefill_tuner.dispatch_once(prefill_candidates, name="M512", args=(prefill_gate_up,))
    prefill_ref = swiglu_eager_prefill(prefill_gate_up)
    print(f"prefill max_diff={(prefill_ref - prefill_out).abs().max().item():.6f}")

    decode_gate_up = torch.randn(1, 1, 12288, device=device, dtype=dtype)
    decode_ref = swiglu_eager_prefill(decode_gate_up)
    decode_out = fused_swiglu_decode(decode_gate_up.clone())
    print(f"decode max_diff={(decode_ref - decode_out).abs().max().item():.6f}")
