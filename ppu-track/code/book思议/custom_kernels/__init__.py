"""Custom inference kernels.
通过    
from custom_kernels import apply_optimizations
    apply_optimizations(self._model)
调用。

做了gate_up_proj, qkv_proj的权重融合
解码阶段用的是Eager attention，预分配静态kv cache，固定cuda图
把解码阶段的RMS norm, SiLU, RoPE用Triton自定义内核替换掉以便于图复用
"""

import logging
import os
import torch

# 注册自定义算子 (torch.ops.custom.*)
import custom_kernels.torch_ops  # noqa: F401

logger = logging.getLogger(__name__)


def _warmup_cuda_cpp_kernels(device='cuda:0'):
    # 预热c++核心
    try:
        from custom_kernels.cuda import get_cuda_ops
        ops = get_cuda_ops()
        if ops is None:
            return

        with torch.inference_mode():
            x = torch.randn(4, 2048, dtype=torch.float16, device=device)
            w = torch.randn(2048, dtype=torch.float16, device=device)
            eps = 1e-6

            # 预热rms_norm prefill和decode两个shape
            for _ in range(3):
                ops.rms_norm(torch.empty_like(x), x, w, eps)
                ops.rms_norm(torch.empty_like(x[:1]), x[:1], w, eps)

            # 预热silu_and_mul
            gu = torch.randn(4, 6144, dtype=torch.float16, device=device)
            gu_out = torch.empty(4, 3072, dtype=torch.float16, device=device)
            gu1_out = torch.empty(1, 3072, dtype=torch.float16, device=device)
            for _ in range(3):
                ops.silu_and_mul(gu_out, gu)
                ops.silu_and_mul(gu1_out, gu[:1])

            # 预热fused_add_rms_norm
            a = torch.randn(4, 2048, dtype=torch.float16, device=device)
            b = torch.randn(4, 2048, dtype=torch.float16, device=device)
            for _ in range(3):
                ops.fused_add_rms_norm(a, b, w, eps)
                ops.fused_add_rms_norm(a[:1], b[:1], w, eps)

        logger.info("[warmup] CUDA C++ kernels warmed up")
    except Exception as e:
        logger.warning(f"[warmup] CUDA C++ kernel warmup failed: {e}")


def _warmup_gemv_kernels(device='cuda:0'):
    # 预热GEMV核心
    try:
        from custom_kernels.cuda.gemv_loader import get_gemv_ops
        gemv_ops = get_gemv_ops()
        if gemv_ops is None:
            return

        with torch.inference_mode():
            # 按Qwen3-VL-2B decode的尺寸来预热
            x = torch.randn(2048, dtype=torch.float16, device=device)
            w_qkv = torch.randn(4096, 2048, dtype=torch.float16, device=device)
            w_o = torch.randn(2048, 2048, dtype=torch.float16, device=device)
            w_gu = torch.randn(6144, 2048, dtype=torch.float16, device=device)
            w_down = torch.randn(2048, 3072, dtype=torch.float16, device=device)

            for _ in range(3):
                gemv_ops.gemv(w_qkv, x)
                gemv_ops.gemv(w_o, x)
                gemv_ops.gemv_silu_mul(w_gu, x)
                mid = torch.randn(3072, dtype=torch.float16, device=device)
                gemv_ops.gemv(w_down, mid)

            # INT4 GEMV warmup (packed uint8 weights)
            try:
                ng_2048 = 2048 // 128  # 16
                ng_3072 = 3072 // 128  # 24
                w_qkv_p = torch.randint(0, 255, (4096, 1024), dtype=torch.uint8, device=device)
                s_qkv = torch.randn(4096, ng_2048, dtype=torch.float32, device=device)
                w_o_p = torch.randint(0, 255, (2048, 1024), dtype=torch.uint8, device=device)
                s_o = torch.randn(2048, ng_2048, dtype=torch.float32, device=device)
                w_gu_p = torch.randint(0, 255, (6144, 1024), dtype=torch.uint8, device=device)
                s_gu = torch.randn(6144, ng_2048, dtype=torch.float32, device=device)
                w_down_p = torch.randint(0, 255, (2048, 1536), dtype=torch.uint8, device=device)
                s_down = torch.randn(2048, ng_3072, dtype=torch.float32, device=device)
                w_lm_p = torch.randint(0, 255, (151936, 1024), dtype=torch.uint8, device=device)
                s_lm = torch.randn(151936, ng_2048, dtype=torch.float32, device=device)
                x = torch.randn(2048, dtype=torch.float16, device=device)
                mid = torch.randn(3072, dtype=torch.float16, device=device)
                res = torch.randn(2048, dtype=torch.float16, device=device)

                for _ in range(2):
                    gemv_ops.gemv_int4(w_qkv_p, x, s_qkv, ng_2048)
                    gemv_ops.gemv_int4(w_o_p, x, s_o, ng_2048)
                    gemv_ops.gemv_silu_mul_int4(w_gu_p, x, s_gu, ng_2048)
                    gemv_ops.gemv_add_int4(w_down_p, mid, s_down, res.clone(), ng_3072)
                    gemv_ops.gemv_lmhead_int4(w_lm_p, x, s_lm, ng_2048)
            except Exception as e2:
                logger.warning(f"[warmup] INT4 GEMV warmup failed: {e2}")

        logger.info("[warmup] GEMV kernels warmed up")
    except Exception as e:
        logger.warning(f"[warmup] GEMV kernel warmup failed: {e}")


def _warmup_triton_kernels(device='cuda:0'):
    # 提前编译所有triton核心 避免第一次调用时编译开销
    try:
        from custom_kernels.cuda.triton_kernels import (
            triton_rms_norm,
            triton_silu_and_mul,
            triton_fused_add_rms_norm,
            triton_fused_rope,
            triton_fused_rope_prefill,
            triton_vision_rope,
            triton_fused_qkv_norm_rope_prefill,
        )

        with torch.inference_mode():
            # 预热rms_norm prefill和decode
            x = torch.randn(4, 2048, dtype=torch.float16, device=device)
            w = torch.randn(2048, dtype=torch.float16, device=device)
            triton_rms_norm(x, w, 1e-6)
            x1 = torch.randn(1, 2048, dtype=torch.float16, device=device)
            triton_rms_norm(x1, w, 1e-6)

            # 预热silu_and_mul
            gu = torch.randn(4, 6144, dtype=torch.float16, device=device)
            triton_silu_and_mul(gu)
            gu1 = torch.randn(1, 6144, dtype=torch.float16, device=device)
            triton_silu_and_mul(gu1)

            # 预热fused_add_rms_norm
            a = torch.randn(4, 2048, dtype=torch.float16, device=device)
            b = torch.randn(4, 2048, dtype=torch.float16, device=device)
            triton_fused_add_rms_norm(a, b, w, 1e-6)
            a1 = torch.randn(1, 2048, dtype=torch.float16, device=device)
            b1 = torch.randn(1, 2048, dtype=torch.float16, device=device)
            triton_fused_add_rms_norm(a1, b1, w, 1e-6)

            # 预热decode用fused_rope
            q = torch.randn(1, 16, 1, 128, dtype=torch.float16, device=device)
            k = torch.randn(1, 8, 1, 128, dtype=torch.float16, device=device)
            cos = torch.randn(1, 1, 1, 128, dtype=torch.float16, device=device)
            sin = torch.randn(1, 1, 1, 128, dtype=torch.float16, device=device)
            triton_fused_rope(q, k, cos, sin)

            # 预热prefill用fused_rope
            qp = torch.randn(1, 16, 4, 128, dtype=torch.float16, device=device)
            kp = torch.randn(1, 8, 4, 128, dtype=torch.float16, device=device)
            cosp = torch.randn(1, 1, 4, 128, dtype=torch.float16, device=device)
            sinp = torch.randn(1, 1, 4, 128, dtype=torch.float16, device=device)
            triton_fused_rope_prefill(qp, kp, cosp, sinp)

            # 预热vision_rope
            vq = torch.randn(16, 4, 64, dtype=torch.float16, device=device)
            vk = torch.randn(16, 4, 64, dtype=torch.float16, device=device)
            vc = torch.randn(16, 64, dtype=torch.float16, device=device)
            vs = torch.randn(16, 64, dtype=torch.float16, device=device)
            triton_vision_rope(vq, vk, vc, vs)

            # 预热fused_qkv_norm_rope_prefill
            qkv_warm = torch.randn(1, 4, 4096, dtype=torch.float16, device=device)
            q_w = torch.randn(128, dtype=torch.float16, device=device)
            k_w = torch.randn(128, dtype=torch.float16, device=device)
            cos_w = torch.randn(1, 1, 4, 128, dtype=torch.float16, device=device)
            sin_w = torch.randn(1, 1, 4, 128, dtype=torch.float16, device=device)
            triton_fused_qkv_norm_rope_prefill(
                qkv_warm, q_w, k_w, cos_w, sin_w,
                2048, 1024, 128,
            )

        logger.info("[warmup] All Triton kernels pre-compiled")
    except Exception as e:
        logger.warning(f"[warmup] Triton kernel warmup failed: {e}")


def apply_optimizations(model, verbose=True):
    # 给模型打补丁 + 预热核心 + 挂载cuda图解码引擎
    if verbose:
        print("[custom_kernels] Applying optimizations...")

    results = {}

    if os.environ.get("AICAS_SAFE_MODE", "0") == "1":
        if verbose:
            print("  [safe_mode] Skipping experimental graph/GEMV/quantization patches")
        return model, results

    # 1. 打补丁: 权重融合 + triton核心
    if os.environ.get("AICAS_DISABLE_PATCH_MODEL", "0") != "1":
        from custom_kernels.patch_model import patch_model_for_graph
        try:
            patch_results = patch_model_for_graph(model)
            results.update(patch_results)
            if verbose:
                for k, v in patch_results.items():
                    print(f"  [{k}] Patched {v} modules")
        except Exception as e:
            if verbose:
                print(f"  [patch_model] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 2. 视觉编码器优化 (fused LN + add+LN)
    if os.environ.get("AICAS_DISABLE_VISION_OPT", "0") != "1":
        try:
            from custom_kernels.vision.opt import patch_vision
            n = patch_vision(model)
            results["vision_opt"] = n
            if verbose:
                print(f"  [vision_opt] Patched {n} vision components (fused LN + add+LN)")
        except Exception as e:
            if verbose:
                print(f"  [vision_opt] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Vision CUDA graph was slower on the original A800 baseline and currently
    # breaks native generate on PPU through inference-tensor in-place writes.
    if os.environ.get("AICAS_ENABLE_VISION_GRAPH", "0") == "1":
        try:
            from custom_kernels.vision.opt import patch_vision_forward_with_graph
            n = patch_vision_forward_with_graph(model)
            results["vision_graph"] = n
            if verbose:
                print(f"  [vision_graph] Patched vision forward with CUDA graph engine ({n})")
        except Exception as e:
            if verbose:
                print(f"  [vision_graph] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 3. 预热所有核心
    try:
        _warmup_cuda_cpp_kernels()
        _warmup_gemv_kernels()
        _warmup_triton_kernels()
        from custom_kernels.vision.opt import warmup_vision_kernels
        warmup_vision_kernels()
        results["kernel_warmup"] = True
        if verbose:
            print("  [kernel_warmup] All kernels pre-compiled (CUDA C++ + Triton + Vision)")

        # Warmup fused GEMV kernels (disabled by default — causes CUDA corruption on some GPUs;
        # the kernels are still used during decode, this is just JIT pre-compilation)
        if os.environ.get("AICAS_DISABLE_FUSED_GEMV_WARMUP", "1") != "1":
            try:
                from custom_kernels.cuda.fused_gemv_loader import get_fused_gemv_ops
                fg = get_fused_gemv_ops()
                if fg is not None:
                    fg_device = 'cuda:0'
                    with torch.inference_mode():
                        x = torch.randn(2048, dtype=torch.float16, device=fg_device)
                        nw = torch.randn(2048, dtype=torch.float16, device=fg_device)
                        W = torch.randint(0, 255, (4096, 1024), dtype=torch.uint8, device=fg_device)
                        s = torch.randn(4096, dtype=torch.float32, device=fg_device)
                        for _ in range(3):
                            fg.gemv_rmsnorm_int4(W, x, nw, s, 1e-6)
                        W_gu = torch.randint(0, 255, (6144, 1024), dtype=torch.uint8, device=fg_device)
                        s_gu = torch.randn(6144, dtype=torch.float32, device=fg_device)
                        res = torch.randn(2048, dtype=torch.float16, device=fg_device)
                        attn = torch.randn(2048, dtype=torch.float16, device=fg_device)
                        for _ in range(3):
                            fg.gemv_addrmsnorm_silu_mul_int4(W_gu, res.clone(), attn, nw, s_gu, 1e-6)
                    if verbose:
                        print("  [fused_gemv_warmup] Fused GEMV kernels warmed up")
            except Exception as e:
                if verbose:
                    print(f"  [fused_gemv_warmup] SKIPPED: {e}")
    except Exception as e:
        if verbose:
            print(f"  [kernel_warmup] FAILED: {e}")

    # Pre-capture vision CUDA graphs for common patch counts
    if "vision_graph" in results:
        try:
            from custom_kernels.vision.opt import pre_capture_vision_graphs
            n = pre_capture_vision_graphs(model)
            results["vision_graph_precapture"] = n
            if verbose:
                print(f"  [vision_graph_precapture] Pre-captured {n} vision CUDA graphs")
        except Exception as e:
            if verbose:
                print(f"  [vision_graph_precapture] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 3.5. Load skip MLP adapters (if trained)
    if os.environ.get("AICAS_SKIP_MLP_ADAPTER", "0") == "1":
        try:
            from train_skip_adapter import load_adapters
            adapter_path = os.environ.get("AICAS_SKIP_MLP_ADAPTER_PATH", "model/skip_adapters.pt")
            n = load_adapters(model, adapter_path)
            results["skip_mlp_adapter"] = n
            if verbose:
                print(f"  [skip_mlp_adapter] Loaded {n} adapters")
        except Exception as e:
            if verbose:
                print(f"  [skip_mlp_adapter] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 4. 挂载cuda图解码引擎
    if os.environ.get("AICAS_DISABLE_GRAPH_ENGINE", "0") != "1":
        try:
            from custom_kernels.graph_engine import make_custom_generate
            model.generate = make_custom_generate(model)
            results["graph_engine"] = True
            if verbose:
                print("  [graph_engine] Custom generate with CUDA graph decode")
        except Exception as e:
            if verbose:
                print(f"  [graph_engine] FAILED: {e}")
            import traceback
            traceback.print_exc()

    # 注: torch.compile实验性功能已禁用 还有图断裂的问题
    # try:
    #     for i, layer in enumerate(model.model.language_model.layers):
    #         model.model.language_model.layers[i] = torch.compile(layer, mode='default', fullgraph=False)
    #     results["torch_compile_lm"] = True
    #     if verbose:
    #         print("  [torch_compile_lm] Compiled LM decoder layers")
    # except Exception as e:
    #     if verbose:
    #         print(f"  [torch_compile_lm] FAILED: {e}")
    #     import traceback
    #     traceback.print_exc()

    if verbose:
        total = sum(v for v in results.values() if isinstance(v, int))
        print(f"[custom_kernels] Done. {len(results)} optimization groups applied.")

    return model, results
