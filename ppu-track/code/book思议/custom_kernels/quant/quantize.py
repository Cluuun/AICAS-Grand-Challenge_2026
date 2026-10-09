"""自适应动态量化: 根据权重分布自动选择量化精度，最大程度保留能力。"""

import torch
import logging
import os

logger = logging.getLogger(__name__)


def quantize_per_channel(weight):
    """Per-channel symmetric INT8: 每行一个scale，精度损失最小。"""
    # weight: [N, K] fp16
    scale = weight.abs().amax(dim=1) / 127.0  # [N]
    scale = scale.clamp(min=1e-8)  # 避免除零
    w_int8 = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.to(torch.float32)  # scale用fp32避免精度损失


def quantize_per_group(weight, group_size=128):
    """Per-group symmetric INT8: 每128个元素一个scale，精度更好。"""
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    num_groups = K // group_size

    w_grouped = weight.reshape(N, num_groups, group_size)
    scale = w_grouped.abs().amax(dim=2) / 127.0  # [N, num_groups]
    scale = scale.clamp(min=1e-8)
    w_int8 = (w_grouped / scale.unsqueeze(2)).round().clamp(-128, 127).to(torch.int8)
    return w_int8.reshape(N, K), scale.to(torch.float32)


def compute_quant_error(weight, w_int8, scale):
    """计算量化误差 (相对MSE)。"""
    if scale.dim() == 1:
        # per-channel
        reconstructed = w_int8.float() * scale.unsqueeze(1)
    else:
        # per-group
        N, K = weight.shape
        group_size = K // scale.shape[1]
        reconstructed = w_int8.float() * scale.unsqueeze(2).expand(-1, -1, group_size).reshape(N, K)

    mse = (weight.float() - reconstructed).pow(2).mean()
    signal_power = weight.float().pow(2).mean()
    return (mse / signal_power).item()


def analyze_weight(weight):
    """分析权重分布特征，返回敏感度指标。"""
    w = weight.float()
    # Kurtosis: 越大说明outlier越多，量化越难
    mean = w.mean()
    std = w.std()
    if std < 1e-8:
        return 0.0
    kurtosis = ((w - mean) ** 4).mean() / (std ** 4)
    # Outlier ratio: 超过3sigma的比例
    outlier_ratio = ((w - mean).abs() > 3 * std).float().mean().item()
    # 组合指标
    return kurtosis * outlier_ratio


def adaptive_quantize(weight, name="", threshold=0.001):
    """自适应量化: 根据权重特征选择per-channel或per-group。

    返回: (w_int8, scale, quant_type)
        quant_type: "per_channel" or "per_group"
    """
    sensitivity = analyze_weight(weight)

    # 先试 per-channel
    w_int8_ch, scale_ch = quantize_per_channel(weight)
    err_ch = compute_quant_error(weight, w_int8_ch, scale_ch)

    if err_ch < threshold:
        return w_int8_ch, scale_ch, "per_channel"

    # per-channel 精度不够，用 per-group
    w_int8_gr, scale_gr = quantize_per_group(weight, group_size=128)
    err_gr = compute_quant_error(weight, w_int8_gr, scale_gr)

    if err_gr < threshold * 2:
        return w_int8_gr, scale_gr, "per_group"

    # 都不够好，还是用 per-channel (比不量化好)
    logger.info(f"  [{name}] high sensitivity ({sensitivity:.4f}), per_channel err={err_ch:.6f}")
    return w_int8_ch, scale_ch, "per_channel"


def quantize_int4_per_channel(weight):
    """Per-channel symmetric INT4: 2 values packed per byte, [-8, 7] range.

    Returns packed [N, K/2] uint8 and scale [N, 1] float32.
    Kernel uses bias correction trick for efficiency.
    """
    N, K = weight.shape
    scale = weight.float().abs().amax(dim=1) / 7.0  # [N]
    scale = scale.clamp(min=1e-8)
    quantized = torch.clamp(torch.round(weight.float() / scale.unsqueeze(1)), -8, 7).to(torch.int16)
    biased = quantized + 8  # [-8,7] → [0,15] unsigned
    packed = (biased[:, 0::2] | (biased[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.unsqueeze(1).to(torch.float32)  # [N, 1] for per-group interface


def quantize_int4_per_group(weight, group_size=128):
    """Per-group symmetric INT4: group_size elements share one scale.

    Returns packed [N, K/2] uint8 and scale [N, num_groups] float32.
    Much more accurate than per-channel for weights with outliers.
    """
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    num_groups = K // group_size

    w_grouped = weight.float().reshape(N, num_groups, group_size)
    scale = w_grouped.abs().amax(dim=2) / 7.0  # [N, num_groups]
    scale = scale.clamp(min=1e-8)

    quantized = torch.clamp(torch.round(w_grouped / scale.unsqueeze(2)), -8, 7).to(torch.int16)
    quantized = quantized.reshape(N, K)
    biased = quantized + 8
    packed = (biased[:, 0::2] | (biased[:, 1::2] << 4)).to(torch.uint8)
    return packed, scale.to(torch.float32)


def dequantize_int4_packed(packed, scale, group_size=128):
    """把 packed int4 解回近似权重，只用于离线误差评估。"""
    if scale.dim() == 1 or (scale.dim() == 2 and scale.shape[1] == 1):
        N = packed.shape[0]
        K = packed.shape[1] * 2
        unpacked = torch.empty(N, K, dtype=torch.int16, device=packed.device)
        unpacked[:, 0::2] = (packed & 0x0F).to(torch.int16) - 8
        unpacked[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.int16) - 8
        scale_flat = scale.reshape(N, 1)
        return unpacked.float() * scale_flat
    N = packed.shape[0]
    num_groups = scale.shape[1]
    group_size = int(group_size)
    K = num_groups * group_size
    vals = torch.empty(N, K, dtype=torch.int16, device=packed.device)
    vals[:, 0::2] = (packed & 0x0F).to(torch.int16) - 8
    vals[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    vals = vals.view(N, num_groups, group_size)
    return (vals.float() * scale.unsqueeze(2)).reshape(N, K)


def _parse_layer_set(value):
    if not value:
        return None
    layers = set()
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            layers.update(range(lo, hi + 1))
        else:
            layers.add(int(part))
    return layers


def _make_layer_filter(layer_start, layer_end):
    include_layers = _parse_layer_set(os.environ.get("AICAS_INT4_LAYERS", ""))
    skip_layers = _parse_layer_set(os.environ.get("AICAS_INT4_SKIP_LAYERS", "")) or set()

    def enabled(layer_idx):
        if include_layers is not None and layer_idx not in include_layers:
            return False
        if layer_idx in skip_layers:
            return False
        return layer_start <= layer_idx < layer_end

    return enabled


def quantize_model_weights_int4(model):
    """INT4 per-group量化所有LM层权重。group_size=128。

    INT4节省75%权重带宽，配合fused norm kernel省掉单独的norm。
    """
    quantized_count = 0
    total_saved_mb = 0.0
    target = os.environ.get("AICAS_INT4_TARGET", "all").lower()
    layer_start = int(os.environ.get("AICAS_INT4_LAYER_START", "0"))
    layer_end = int(os.environ.get("AICAS_INT4_LAYER_END", "9999"))
    layer_enabled = _make_layer_filter(layer_start, layer_end)

    def want(kind):
        if target in ("all", "*"):
            return True
        parts = {p.strip() for p in target.replace("+", ",").split(",") if p.strip()}
        if kind in parts:
            return True
        if kind in ("qkv", "o") and "attn" in parts:
            return True
        if kind in ("gate_up", "down") and "mlp" in parts:
            return True
        return False

    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3VLTextDecoderLayer":
            try:
                layer_idx = int(name.split('.')[-1])
            except Exception:
                layer_idx = 0
            if not layer_enabled(layer_idx):
                continue
            # qkv_proj
            if want("qkv") and hasattr(module, 'self_attn') and hasattr(module.self_attn, 'qkv_proj'):
                w = module.self_attn.qkv_proj.weight.data
                packed, scale = quantize_int4_per_group(w)
                module.self_attn.qkv_proj.weight_int4 = packed
                module.self_attn.qkv_proj.weight_scale_int4 = scale
                saved_mb = w.numel() * 0.5 / 1024 / 1024
                total_saved_mb += saved_mb
                quantized_count += 1

            # o_proj
            if want("o") and hasattr(module, 'self_attn') and hasattr(module.self_attn, 'o_proj'):
                w = module.self_attn.o_proj.weight.data
                packed, scale = quantize_int4_per_group(w)
                module.self_attn.o_proj.weight_int4 = packed
                module.self_attn.o_proj.weight_scale_int4 = scale
                saved_mb = w.numel() * 0.5 / 1024 / 1024
                total_saved_mb += saved_mb
                quantized_count += 1

            # gate_up_proj
            if want("gate_up") and hasattr(module, 'mlp') and hasattr(module.mlp, 'gate_up_proj'):
                w = module.mlp.gate_up_proj.weight.data
                packed, scale = quantize_int4_per_group(w)
                module.mlp.gate_up_proj.weight_int4 = packed
                module.mlp.gate_up_proj.weight_scale_int4 = scale
                saved_mb = w.numel() * 0.5 / 1024 / 1024
                total_saved_mb += saved_mb
                quantized_count += 1

            # down_proj
            if want("down") and hasattr(module, 'mlp') and hasattr(module.mlp, 'down_proj'):
                w = module.mlp.down_proj.weight.data
                packed, scale = quantize_int4_per_group(w)
                module.mlp.down_proj.weight_int4 = packed
                module.mlp.down_proj.weight_scale_int4 = scale
                saved_mb = w.numel() * 0.5 / 1024 / 1024
                total_saved_mb += saved_mb
                quantized_count += 1

    # lm_head is very sensitive: small logit perturbations can flip greedy
    # argmax and cause long repeated answers. Keep it FP16 unless explicitly
    # enabled for speed experiments.
    if hasattr(model, 'lm_head') and os.environ.get("AICAS_QUANTIZE_LM_HEAD", "0") == "1":
        w = model.lm_head.weight.data
        packed, scale = quantize_int4_per_group(w)
        model.lm_head.weight_int4 = packed
        model.lm_head.weight_scale_int4 = scale
        saved_mb = w.numel() * 0.5 / 1024 / 1024
        total_saved_mb += saved_mb
        quantized_count += 1

    logger.info(f"[quantize] INT4: {quantized_count} weights quantized, {total_saved_mb:.0f}MB weight bandwidth")
    print(f"[quantize] INT4: {quantized_count} weights quantized, {total_saved_mb:.0f}MB weight bandwidth")
    return quantized_count


def quantize_model_weights_int4_safe(model):
    """混合 INT4/INT8: 只把低误差层升级为 INT4，其余保留 INT8。

    适合竞赛里的保守提速：先维持可用精度，再尽量吃掉带宽收益。
    """
    # 先建立默认 INT8 量化基线，保证未升级层仍然有可用的低精度权重。
    base_count = quantize_model_weights(model)

    target = os.environ.get("AICAS_INT4_TARGET", "all").lower()
    layer_start = int(os.environ.get("AICAS_INT4_LAYER_START", "0"))
    layer_end = int(os.environ.get("AICAS_INT4_LAYER_END", "9999"))
    layer_enabled = _make_layer_filter(layer_start, layer_end)
    keep_edge_layers = int(os.environ.get("AICAS_INT4_KEEP_EDGE_LAYERS", "4"))
    max_rel_err = float(os.environ.get("AICAS_INT4_MAX_REL_ERR", "0.0025"))
    prefer_group = os.environ.get("AICAS_INT4_PREFER_GROUP", "1") != "0"

    def want(kind):
        if target in ("all", "*"):
            return True
        parts = {p.strip() for p in target.replace("+", ",").split(",") if p.strip()}
        if kind in parts:
            return True
        if kind in ("qkv", "o") and "attn" in parts:
            return True
        if kind in ("gate_up", "down") and "mlp" in parts:
            return True
        return False

    def maybe_assign_int4(module, attr_name, weight, label, module_idx, kind):
        if not want(kind):
            return 0
        if not layer_enabled(module_idx):
            return 0
        if keep_edge_layers > 0:
            last_edge_start = max(layer_start, layer_end - keep_edge_layers)
            if module_idx >= last_edge_start:
                return 0
        if prefer_group:
            packed, scale = quantize_int4_per_group(weight)
            reconstructed = dequantize_int4_packed(packed, scale, group_size=128)
            err = ((weight.float() - reconstructed).pow(2).mean() /
                   weight.float().pow(2).mean().clamp(min=1e-8)).item()
            if err <= max_rel_err:
                setattr(getattr(module, attr_name), "weight_int4", packed)
                setattr(getattr(module, attr_name), "weight_scale_int4", scale)
                return 1
        packed_ch, scale_ch = quantize_int4_per_channel(weight)
        reconstructed_ch = dequantize_int4_packed(packed_ch, scale_ch, group_size=weight.shape[1] // packed_ch.shape[1])
        err_ch = ((weight.float() - reconstructed_ch).pow(2).mean() /
                  weight.float().pow(2).mean().clamp(min=1e-8)).item()
        if err_ch <= max_rel_err * 1.5:
            setattr(getattr(module, attr_name), "weight_int4", packed_ch)
            setattr(getattr(module, attr_name), "weight_scale_int4", scale_ch)
            return 1
        if not prefer_group:
            packed, scale = quantize_int4_per_group(weight)
            reconstructed = dequantize_int4_packed(packed, scale, group_size=128)
            err = ((weight.float() - reconstructed).pow(2).mean() /
                   weight.float().pow(2).mean().clamp(min=1e-8)).item()
            if err <= max_rel_err:
                setattr(getattr(module, attr_name), "weight_int4", packed)
                setattr(getattr(module, attr_name), "weight_scale_int4", scale)
                return 1
        return 0

    upgraded = 0
    total_saved_mb = 0.0
    for name, module in model.named_modules():
        if type(module).__name__ != "Qwen3VLTextDecoderLayer":
            continue
        try:
            layer_idx = int(name.split('.')[-1])
        except Exception:
            layer_idx = 0
        if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'qkv_proj') and hasattr(module.self_attn.qkv_proj, 'weight'):
            w = module.self_attn.qkv_proj.weight.data
            if maybe_assign_int4(module.self_attn, 'qkv_proj', w, f"{layer_idx}.qkv", layer_idx, "qkv"):
                total_saved_mb += w.numel() * 0.5 / 1024 / 1024
                upgraded += 1
        if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'o_proj') and hasattr(module.self_attn.o_proj, 'weight'):
            w = module.self_attn.o_proj.weight.data
            if maybe_assign_int4(module.self_attn, 'o_proj', w, f"{layer_idx}.o", layer_idx, "o"):
                total_saved_mb += w.numel() * 0.5 / 1024 / 1024
                upgraded += 1
        if hasattr(module, 'mlp') and hasattr(module.mlp, 'gate_up_proj') and hasattr(module.mlp.gate_up_proj, 'weight'):
            w = module.mlp.gate_up_proj.weight.data
            if maybe_assign_int4(module.mlp, 'gate_up_proj', w, f"{layer_idx}.gate_up", layer_idx, "gate_up"):
                total_saved_mb += w.numel() * 0.5 / 1024 / 1024
                upgraded += 1
        if hasattr(module, 'mlp') and hasattr(module.mlp, 'down_proj') and hasattr(module.mlp.down_proj, 'weight'):
            w = module.mlp.down_proj.weight.data
            if maybe_assign_int4(module.mlp, 'down_proj', w, f"{layer_idx}.down", layer_idx, "down"):
                total_saved_mb += w.numel() * 0.5 / 1024 / 1024
                upgraded += 1

    logger.info(
        f"[quantize] INT4-safe: upgraded {upgraded} weights on top of {base_count} INT8 weights, "
        f"saved {total_saved_mb:.0f}MB extra bandwidth"
    )
    print(
        f"[quantize] INT4-safe: upgraded {upgraded} weights on top of {base_count} INT8 weights, "
        f"saved {total_saved_mb:.0f}MB extra bandwidth"
    )
    return base_count + upgraded


def quantize_model_weights(model):
    """量化所有LM层权重: qkv, o_proj, gate_up, down, lm_head。

    在权重融合后调用，quantized weights存为module attribute。
    Decode时用INT8 GEMV，prefill时用FP16 GEMM。
    """
    quantized_count = 0
    total_saved_mb = 0.0

    # 1. 量化attention和MLP的融合权重 (28个decoder layer)
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3VLTextDecoderLayer":
            layer_name = name.split('.')[-1]

            # qkv_proj (融合后)
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'qkv_proj'):
                w = module.self_attn.qkv_proj.weight.data
                w_int8, scale, qtype = adaptive_quantize(w, f"{layer_name}.qkv")
                module.self_attn.qkv_proj.weight_int8 = w_int8
                module.self_attn.qkv_proj.weight_scale = scale
                module.self_attn.qkv_proj.quant_type = qtype
                saved_mb = w.numel() * 1 / 1024 / 1024
                total_saved_mb += saved_mb
                quantized_count += 1

            # o_proj
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'o_proj'):
                w = module.self_attn.o_proj.weight.data
                w_int8, scale, qtype = adaptive_quantize(w, f"{layer_name}.o_proj")
                module.self_attn.o_proj.weight_int8 = w_int8
                module.self_attn.o_proj.weight_scale = scale
                module.self_attn.o_proj.quant_type = qtype
                saved_mb = w.numel() * 1 / 1024 / 1024
                total_saved_mb += saved_mb
                quantized_count += 1

            # gate_up_proj (融合后)
            if hasattr(module, 'mlp') and hasattr(module.mlp, 'gate_up_proj'):
                w = module.mlp.gate_up_proj.weight.data
                w_int8, scale, qtype = adaptive_quantize(w, f"{layer_name}.gate_up")
                module.mlp.gate_up_proj.weight_int8 = w_int8
                module.mlp.gate_up_proj.weight_scale = scale
                module.mlp.gate_up_proj.quant_type = qtype
                saved_mb = w.numel() * 1 / 1024 / 1024
                total_saved_mb += saved_mb
                quantized_count += 1

            # down_proj
            if hasattr(module, 'mlp') and hasattr(module.mlp, 'down_proj'):
                w = module.mlp.down_proj.weight.data
                w_int8, scale, qtype = adaptive_quantize(w, f"{layer_name}.down")
                module.mlp.down_proj.weight_int8 = w_int8
                module.mlp.down_proj.weight_scale = scale
                module.mlp.down_proj.quant_type = qtype
                saved_mb = w.numel() * 1 / 1024 / 1024
                total_saved_mb += saved_mb
                quantized_count += 1

    # 2. lm_head is intentionally left in FP16 by default. Quantizing the
    # output projection tends to degrade greedy decoding much more than hidden
    # projections, especially on OCR-style VQA answers.
    if hasattr(model, 'lm_head') and os.environ.get("AICAS_QUANTIZE_LM_HEAD", "0") == "1":
        w = model.lm_head.weight.data
        w_int8, scale, qtype = adaptive_quantize(w, "lm_head")
        model.lm_head.weight_int8 = w_int8
        model.lm_head.weight_scale = scale
        model.lm_head.quant_type = qtype
        saved_mb = w.numel() * 1 / 1024 / 1024
        total_saved_mb += saved_mb
        quantized_count += 1

        # Optional INT4 lm_head (saves ~50% weight bandwidth for vocab projection)
        if os.environ.get("AICAS_QUANTIZE_LM_HEAD_INT4", "0") == "1":
            packed, scale4 = quantize_int4_per_group(w, group_size=128)
            model.lm_head.weight_int4 = packed
            model.lm_head.weight_scale_int4 = scale4
            saved_mb += w.numel() * 0.5 / 1024 / 1024
            total_saved_mb += w.numel() * 0.5 / 1024 / 1024
            print(f"[quantize] lm_head INT4: {packed.shape} (saves {w.numel() * 0.5 / 1024 / 1024:.0f}MB extra)")

    logger.info(f"[quantize] {quantized_count} weights quantized, saved {total_saved_mb:.0f}MB")
    print(f"[quantize] {quantized_count} weights quantized, saved {total_saved_mb:.0f}MB weight bandwidth")
    return quantized_count
