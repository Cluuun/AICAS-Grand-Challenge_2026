"""RMS Norm 优化：CUDA C++ / Triton 核心替换。"""

import types
import logging

from .cuda_ops import _get_cuda_ops, _rms_norm_cuda

logger = logging.getLogger(__name__)


def _make_rms_norm_forward():
    # 用cuda c++核心做rms_norm 比triton快3-4x
    cuda_ops = _get_cuda_ops()

    if cuda_ops is not None:
        def patched_forward(self, hidden_states):
            return _rms_norm_cuda(cuda_ops, hidden_states, self.weight, self.variance_epsilon)
    else:
        def patched_forward(self, hidden_states):
            from custom_kernels.cuda.triton_kernels import triton_rms_norm
            return triton_rms_norm(hidden_states, self.weight, self.variance_epsilon)
    return patched_forward


def patch_rms_norm(model):
    # 替换所有RMSNorm的forward
    patched = 0
    forward_fn = _make_rms_norm_forward()

    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3VLTextRMSNorm":
            module.forward = types.MethodType(forward_fn, module)
            patched += 1

    return patched
