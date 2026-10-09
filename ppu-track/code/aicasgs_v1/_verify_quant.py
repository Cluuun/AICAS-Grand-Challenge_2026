"""快速验证：用一个简单的 nn.Module 测试量化的正确性（非 vLLM 上下文测试）"""
import torch, sys, json, os

# 从 site-packages 的补丁模块中导入量化函数
spec = importlib.util.spec_from_file_location(
    "_aicas_dequant_impl",
    "/usr/local/lib/python3.12/site-packages/_aicas_dequant_impl.py"
)
if spec and spec.loader:
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

# 创建一个简单的线性层
w = torch.randn(2048, 4096, dtype=torch.bfloat16)
before = w.clone()
mod._quantize_model_weights(torch.nn.Linear(4096, 2048, bias=False).to(w.dtype))

import importlib.util
