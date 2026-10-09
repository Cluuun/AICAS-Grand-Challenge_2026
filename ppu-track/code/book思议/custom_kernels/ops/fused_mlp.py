"""融合SwiGLU MLP，把gate和up合成一次矩阵乘。"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _silu_and_mul_kernel(
    GATE, UP, OUT,
    stride_g, stride_u, stride_o,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """一个kernel搞定silu(gate) * up。"""
    row = tl.program_id(0)
    GATE += row * stride_g
    UP += row * stride_u
    OUT += row * stride_o

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    gate = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)

    # silu(gate) = gate * sigmoid(gate)
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(OUT + offs, out, mask=mask)


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """融合silu(gate) * up。"""
    assert gate.shape == up.shape
    shape = gate.shape
    hidden = shape[-1]
    gate = gate.reshape(-1, hidden)
    up = up.reshape(-1, hidden)
    M = gate.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(hidden)
    out = torch.empty_like(gate)

    _silu_and_mul_kernel[(M,)](
        gate, up, out,
        gate.stride(0), up.stride(0), out.stride(0),
        N=hidden,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(shape)


def _make_fused_mlp_forward(gate_up_weight, down_weight, gate_size, up_size):
    """生成融合后的MLP forward函数。"""

    def forward(self, x):
        # gate + up 一次矩阵乘
        gate_up = F.linear(x, gate_up_weight)
        gate, up = gate_up.split([gate_size, up_size], dim=-1)

        # SwiGLU: silu(gate) * up
        intermediate = F.silu(gate) * up

        # 下投影
        return F.linear(intermediate, down_weight)

    return forward


def patch_mlp(model: torch.nn.Module):
    """把所有Qwen3VLTextMLP模块patch成融合SwiGLU版本。"""
    import types
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

    count = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen3VLTextMLP):
            # 把gate和up的权重拼起来
            gate_up_weight = torch.cat(
                [module.gate_proj.weight, module.up_proj.weight], dim=0
            ).contiguous()
            down_weight = module.down_proj.weight.contiguous()

            gate_size = module.gate_proj.weight.shape[0]
            up_size = module.up_proj.weight.shape[0]

            fwd = _make_fused_mlp_forward(gate_up_weight, down_weight, gate_size, up_size)
            module.forward = types.MethodType(fwd, module)
            count += 1

    return count
