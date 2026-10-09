"""MLP 优化：gate+up 权重融合 + 自定义 silu_and_mul。"""

import types
import logging

import torch
import torch.nn as nn

from .cuda_ops import _get_cuda_ops, _get_gemv_ops, _silu_and_mul_cuda, _int8_gemv

logger = logging.getLogger(__name__)


def _make_mlp_forward():
    # MLP forward: 融合gate+up权重 用自定义核心做silu*mul
    # INT8 decode: 用INT8 GEMV省50%权重带宽
    cuda_ops = _get_cuda_ops()
    gemv_ops = _get_gemv_ops()

    if gemv_ops is not None and cuda_ops is not None:
        def patched_forward(self, x):
            if x.shape[0] == 1 and x.shape[1] == 1:
                x_flat = x.reshape(-1)
                # INT4 decode path, with mixed INT4/INT8 fallback per projection.
                if hasattr(self.gate_up_proj, 'weight_int4'):
                    mid = gemv_ops.gemv_silu_mul_int4(
                        self.gate_up_proj.weight_int4, x_flat,
                        self.gate_up_proj.weight_scale_int4,
                        self.gate_up_proj.weight_scale_int4.shape[1])
                elif hasattr(self.gate_up_proj, 'weight_int8'):
                    mid = _int8_gemv(gemv_ops, 'gemv_silu_mul_int8',
                        self.gate_up_proj.weight_int8, x_flat, self.gate_up_proj.weight_scale)
                else:
                    mid = gemv_ops.gemv_silu_mul(self.gate_up_proj.weight, x_flat)

                if hasattr(self.down_proj, 'weight_int4'):
                    out = gemv_ops.gemv_int4(
                        self.down_proj.weight_int4, mid,
                        self.down_proj.weight_scale_int4,
                        self.down_proj.weight_scale_int4.shape[1])
                elif hasattr(self.down_proj, 'weight_int8'):
                    out = _int8_gemv(gemv_ops, 'gemv_int8',
                        self.down_proj.weight_int8, mid, self.down_proj.weight_scale)
                else:
                    out = gemv_ops.gemv(self.down_proj.weight, mid)
                return out.reshape(1, 1, -1)
            else:
                # prefill: cuBLAS GEMM + cuda silu_and_mul
                if hasattr(self.gate_up_proj, 'weight_int8_prefill') and x.reshape(-1, x.shape[-1]).shape[0] > 16:
                    from custom_kernels.quant.int8_prefill import int8_linear_prefill
                    gate_up = int8_linear_prefill(
                        x, self.gate_up_proj.weight_int8_prefill, self.gate_up_proj.weight_scale_prefill)
                else:
                    gate_up = self.gate_up_proj(x)
                intermediate = _silu_and_mul_cuda(cuda_ops, gate_up)
                if hasattr(self.down_proj, 'weight_int8_prefill') and intermediate.reshape(-1, intermediate.shape[-1]).shape[0] > 16:
                    from custom_kernels.quant.int8_prefill import int8_linear_prefill
                    return int8_linear_prefill(
                        intermediate, self.down_proj.weight_int8_prefill, self.down_proj.weight_scale_prefill)
                return self.down_proj(intermediate)
    elif cuda_ops is not None:
        def patched_forward(self, x):
            gate_up = self.gate_up_proj(x)
            intermediate = _silu_and_mul_cuda(cuda_ops, gate_up)
            return self.down_proj(intermediate)
    else:
        def patched_forward(self, x):
            gate_up = self.gate_up_proj(x)
            from custom_kernels.cuda.triton_kernels import triton_silu_and_mul
            intermediate = triton_silu_and_mul(gate_up)
            return self.down_proj(intermediate)

    return patched_forward


def patch_mlp(model):
    # 把gate_proj和up_proj的权重拼起来
    patched = 0
    forward_fn = _make_mlp_forward()

    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3VLTextMLP":
            gate_w = module.gate_proj.weight
            up_w = module.up_proj.weight
            gate_up_w = torch.cat([gate_w, up_w], dim=0)

            gate_up_proj = nn.Linear(
                gate_w.shape[1], gate_up_w.shape[0],
                bias=False,
            )
            gate_up_proj.weight = nn.Parameter(gate_up_w)
            gate_up_proj = gate_up_proj.to(
                device=gate_w.device, dtype=gate_w.dtype
            )

            module.gate_up_proj = gate_up_proj
            del module.gate_proj
            del module.up_proj
            del module.act_fn

            module.forward = types.MethodType(forward_fn, module)
            patched += 1

    return patched
