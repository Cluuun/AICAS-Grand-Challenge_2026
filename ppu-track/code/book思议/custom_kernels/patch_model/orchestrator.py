"""补丁编排入口：按顺序应用所有优化补丁。"""

import logging
import os

import torch

from .cuda_ops import _get_cuda_ops
from .rms_norm import patch_rms_norm
from .mlp import patch_mlp
from .attention import patch_qkv
from .decoder_layer import patch_decoder_layers
from .vision import patch_vision_rope, patch_vision_caches, patch_deepstack
from .language_model import patch_language_model_forward
from .calibration import _calibrate_mlp_skip_scales

logger = logging.getLogger(__name__)

# 包在 custom_kernels/patch_model/ 下，需要 3 层 dirname 才到项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def patch_model_for_graph(model):
    # 打所有图安全的补丁: 权重融合 + flash解码 + 自定义核心
    results = {}

    # 先加载cuda算子看看能不能用
    cuda_ops = _get_cuda_ops()

    # 1. RMSNorm: triton核心
    if os.environ.get("AICAS_DISABLE_RMS_NORM_PATCH", "0") != "1":
        try:
            n = patch_rms_norm(model)
            results["rms_norm"] = n
            logger.info(f"[patch] Patched {n} RMSNorm modules (Triton)")
        except Exception as e:
            logger.warning(f"[patch] RMSNorm patch failed: {e}")

    # 2. MLP: 权重融合 + silu_and_mul
    if os.environ.get("AICAS_DISABLE_MLP_PATCH", "0") != "1":
        try:
            n = patch_mlp(model)
            results["mlp"] = n
            logger.info(f"[patch] Fused {n} MLP modules (gate_up + silu_and_mul)")
        except Exception as e:
            logger.warning(f"[patch] MLP patch failed: {e}")

    # 3. QKV融合 + flash解码
    if os.environ.get("AICAS_DISABLE_QKV_PATCH", "0") != "1":
        try:
            n = patch_qkv(model)
            results["qkv"] = n
            logger.info(f"[patch] Fused {n} attention QKV + flash decode")
        except Exception as e:
            logger.warning(f"[patch] QKV fusion patch failed: {e}")

    # 4. Decoder层: 融合残差加+norm
    if os.environ.get("AICAS_DISABLE_DECODER_LAYER_PATCH", "0") != "1":
        try:
            n = patch_decoder_layers(model)
            results["decoder_layer"] = n
            logger.info(f"[patch] Patched {n} decoder layers (fused residual+norm)")
        except Exception as e:
            logger.warning(f"[patch] Decoder layer patch failed: {e}")

    # 5. Vision RoPE: 用triton替换 避免FP16→FP32→FP16
    if os.environ.get("AICAS_DISABLE_VISION_ROPE_PATCH", "0") != "1":
        try:
            n = patch_vision_rope()
            results["vision_rope"] = n
            logger.info(f"[patch] Patched vision encoder RoPE ({n} ops replaced)")
        except Exception as e:
            logger.warning(f"[patch] Vision RoPE patch failed: {e}")

    # 5b. Vision缓存: pos_embed + RoPE 按grid_thw做缓存
    if os.environ.get("AICAS_DISABLE_VISION_CACHE_PATCH", "0") != "1":
        try:
            n = patch_vision_caches(model)
            results["vision_cache"] = n
            logger.info(f"[patch] Patched vision pos_embed + RoPE caching ({n} caches)")
        except Exception as e:
            logger.warning(f"[patch] Vision cache patch failed: {e}")

    # 6. DeepStack: 图安全版
    if os.environ.get("AICAS_DISABLE_DEEPSTACK_PATCH", "0") != "1":
        try:
            n = patch_deepstack(model)
            results["deepstack"] = n
            logger.info(f"[patch] Patched deepstack process (graph-safe)")
        except Exception as e:
            logger.warning(f"[patch] DeepStack patch failed: {e}")

    # 6b. Language model forward: skip unnecessary ops during decode
    if os.environ.get("AICAS_DISABLE_LM_FORWARD_PATCH", "0") != "1":
        try:
            n = patch_language_model_forward(model)
            results["lm_forward"] = n
            logger.info(f"[patch] Patched language model forward (skip mask+deepstack in decode)")
        except Exception as e:
            logger.warning(f"[patch] Language model forward patch failed: {e}")

    # 7. Runtime INT8 quantization
    if os.environ.get("AICAS_DISABLE_INT8", "0") != "1":
        try:
            from custom_kernels.quant.quantize import quantize_model_weights
            n = quantize_model_weights(model)
            results["quantize"] = n
        except Exception as e:
            logger.warning(f"[patch] INT8 quantization failed: {e}")
            print(f"[quantize] Failed: {e}")

    # MLP neuron pruning: after quantization, prune less important neurons
    if os.environ.get("AICAS_MLP_PRUNE_LAYERS", "").strip():
        try:
            from custom_kernels.compress.neuron_prune import setup_mlp_neuron_pruning
            n = setup_mlp_neuron_pruning(model)
            results["neuron_prune"] = n
        except Exception as e:
            logger.warning(f"[patch] MLP neuron pruning failed: {e}")
            print(f"[neuron_prune] Failed: {e}")

    if os.environ.get("AICAS_ENABLE_PREFILL_INT8", "0") == "1":
        try:
            _has_int_mm = hasattr(torch, '_int_mm')
            if not _has_int_mm:
                raise RuntimeError("torch._int_mm not available on this build")
            from custom_kernels.quant.prefill_int8 import prefill_quantize_weights
            n = prefill_quantize_weights(model)
            results["prefill_int8"] = n
        except Exception as e:
            logger.warning(f"[patch] Prefill INT8 quantization failed: {e}")
            print(f"[prefill_int8] Failed: {e}")

    # 8. INT4 per-group量化
    if os.environ.get("AICAS_ENABLE_INT4_SAFE", "0") == "1":
        try:
            from custom_kernels.quant.quantize import quantize_model_weights_int4_safe
            n = quantize_model_weights_int4_safe(model)
            results["quantize_int4_safe"] = n
        except Exception as e:
            logger.warning(f"[patch] INT4 safe quantization failed: {e}")
            print(f"[quantize_int4_safe] Failed: {e}")
    elif os.environ.get("AICAS_ENABLE_INT4", "0") == "1":
        try:
            from custom_kernels.quant.quantize import quantize_model_weights_int4
            n = quantize_model_weights_int4(model)
            results["quantize_int4"] = n
        except Exception as e:
            logger.warning(f"[patch] INT4 quantization failed: {e}")
            print(f"[quantize_int4] Failed: {e}")

    # Low-rank MLP approximation for target decode layers
    if os.environ.get("AICAS_LOWRANK_MLP_LAYERS", "").strip():
        try:
            from custom_kernels.compress.lowrank_mlp import setup_lowrank_mlp
            n = setup_lowrank_mlp(model)
            if n > 0:
                results["lowrank_mlp"] = n
        except Exception as e:
            logger.warning(f"[patch] Low-rank MLP setup failed: {e}")
            print(f"[lowrank_mlp] Failed: {e}")
            import traceback
            traceback.print_exc()

    # DASH scaling calibration for MLP skip layers
    _main_mlp = os.environ.get("AICAS_DECODE_SKIP_MLP_LAYERS", "").strip()
    _tail_mlp = os.environ.get("AICAS_DECODE_TAIL_SKIP_MLP_LAYERS", "").strip()
    _combined_mlp = ",".join(x for x in [_main_mlp, _tail_mlp] if x)
    if _combined_mlp:
        try:
            _skip_set = {int(x) for x in _combined_mlp.replace("+", ",").split(",") if x.strip()}
            # Optional: restrict per-channel DASH to specific layers
            _pc_only_str = os.environ.get("AICAS_DASH_PER_CHANNEL_LAYERS", "").strip()
            _pc_only_set = {int(x) for x in _pc_only_str.split(",") if x.strip()} if _pc_only_str else None
            _use_pc_file = _pc_only_set is not None
            # Try loading pre-computed per-channel scales from real activations
            _pc_path = os.path.join(_PROJECT_ROOT, "model", "per_channel_dash.pt")
            if _use_pc_file and os.environ.get("AICAS_DASH_PER_CHANNEL", "1") != "0" and os.path.exists(_pc_path):
                _pc_data = torch.load(_pc_path, map_location="cpu", weights_only=True)
                _device = next(model.model.parameters()).device
                _loaded = 0
                _pc_layers = set()
                for li in _skip_set:
                    if _pc_only_set is not None and li not in _pc_only_set:
                        continue
                    key = f"layer_{li}"
                    if key in _pc_data["scales"]:
                        scale_vec = _pc_data["scales"][key].half().to(_device)
                        model.model.language_model.layers[li]._mlp_skip_scale = scale_vec
                        _loaded += 1
                        _pc_layers.add(li)
                if _loaded > 0:
                    print(f"[dash] Loaded {_loaded} per-channel scales from {_pc_path} (layers {sorted(_pc_layers)})")
                # Calibration for remaining layers without per-channel
                _remaining = _skip_set - _pc_layers
                if _remaining:
                    _calibrate_mlp_skip_scales(model, _remaining)
            else:
                _calibrate_mlp_skip_scales(model, _skip_set)
        except Exception as e:
            logger.warning(f"[patch] DASH scaling calibration failed: {e}")
            print(f"[dash] Calibration FAILED: {e}")

    # Load per-layer MLP skip bias compensation
    _disable_bias = os.environ.get("AICAS_DISABLE_MLP_SKIP_BIAS", "0") == "1"
    _bias_path = os.path.join(_PROJECT_ROOT, "model", "mlp_skip_biases.pt")
    if not _disable_bias and os.path.exists(_bias_path) and _combined_mlp:
        try:
            _bias_data = torch.load(_bias_path, map_location="cpu", weights_only=True)
            _device = next(model.model.parameters()).device
            _loaded_bias = 0
            for li in _skip_set:
                key = f"layer_{li}"
                if key in _bias_data.get("biases", {}):
                    bias_vec = _bias_data["biases"][key].half().to(_device)
                    model.model.language_model.layers[li]._mlp_skip_bias = bias_vec
                    _loaded_bias += 1
            if _loaded_bias > 0:
                print(f"[bias] Loaded {_loaded_bias} layer biases from {_bias_path}")
        except Exception as e:
            logger.warning(f"[patch] Bias loading failed: {e}")

    # Load low-rank linear compensation for MLP skip
    _lr_rank = int(os.environ.get("AICAS_LR_COMP_RANK", "0"))
    if _lr_rank > 0 and _combined_mlp:
        _lr_path = os.path.join(_PROJECT_ROOT, "model", f"lowrank_comp_r{_lr_rank}.pt")
        if os.path.exists(_lr_path):
            try:
                _lr_data = torch.load(_lr_path, map_location="cpu", weights_only=True)
                _device = next(model.model.parameters()).device
                _loaded_lr = 0
                for li in _skip_set:
                    key_U = f"layer_{li}_U"
                    key_V = f"layer_{li}_V"
                    key_b = f"layer_{li}_bias"
                    if key_U in _lr_data and key_V in _lr_data and key_b in _lr_data:
                        U = _lr_data[key_U].half().to(_device)   # [H, r]
                        V = _lr_data[key_V].half().to(_device)   # [r, H]
                        bias = _lr_data[key_b].half().to(_device) # [H]
                        model.model.language_model.layers[li]._lr_comp = (U, V, bias)
                        _loaded_lr += 1
                if _loaded_lr > 0:
                    print(f"[lr_comp] Loaded {_loaded_lr} layer low-rank (rank={_lr_rank}) from {_lr_path}")
            except Exception as e:
                logger.warning(f"[patch] Low-rank compensation loading failed: {e}")

    # Load SwiGLU adapters for MLP skip layers
    _adapters_default = os.environ.get("AICAS_ENABLE_SKIP_ADAPTERS", "1")
    if _adapters_default != "0":
        try:
            from train_skip_adapter import load_adapters
            n = load_adapters(model)
            if n > 0:
                results["skip_adapters"] = n
        except Exception as e:
            logger.warning(f"[patch] Skip adapter loading failed: {e}")

    return results
