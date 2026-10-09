"""
[SHLEE] Fused Gate+Up projection for MLP: replaces two separate Linear layers
(gate_proj, up_proj) with a single matmul, then splits and applies fused SiLU*Mul.

For Qwen3-VL-2B:
  hidden_size       = 2048
  intermediate_size = 6144
  fused weight: [hidden_size, 2 * intermediate_size] = [2048, 12288]
"""

import torch
import torch.nn as nn

from my_kernel.fused_silu_mul_ver2 import fused_silu_mul

class FusedGateUpProjection(nn.Module):
    """Single matmul for gate+up, then fused SiLU*Mul, then down_proj."""

    def __init__(self, mlp: nn.Module):
        super().__init__()

        w_gate = mlp.gate_proj.weight.data
        w_up = mlp.up_proj.weight.data
        self.intermediate_size = int(w_gate.shape[0])

        fused_w = torch.cat([w_gate, w_up], dim=0)
        self.gate_up_proj = nn.Linear(
            fused_w.shape[1], fused_w.shape[0], bias=False,
        )
        self.gate_up_proj.weight = nn.Parameter(fused_w)

        self.down_proj = mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        return self.down_proj(fused_silu_mul(gate_up, self.intermediate_size))
