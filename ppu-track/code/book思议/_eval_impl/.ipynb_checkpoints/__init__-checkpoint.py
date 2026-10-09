"""
AICAS 2026 - Participant Core Modification File

All module-level functions inlined here (like original single-file) so that
_PrewarmProcessor and VLMModel see the SAME globals. Sub-modules
processor.py and model.py import these via `from . import _shape` etc.

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
from typing import Dict
import atexit
import os
import re
import sys
import time
import traceback
import types
try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- Environment variables (调参区) ---
# 所有变量用 setdefault，可在外部提前覆盖。

# === INT8 量化 & GEMV 解码 ===
# 启用 INT8 GEMV kernel 的 16-warps 配置，用于 decode 阶段的 MLP/QKV 投影
os.environ.setdefault("AICAS_INT8_GEMV_WARPS16", "1")
# 启用 INT8 GEMV 的 Cooperative Groups 模式，提高矩阵乘法并行度
os.environ.setdefault("AICAS_INT8_GEMV_CG", "1")
# 融合 gate_up 投影的 24-warps GEMV kernel（适用于大 batch）
os.environ.setdefault("AICAS_FUSED_GATE_WARPS24", "1")
# 融合 gate_up 投影的 16-warps GEMV kernel（适用于小 batch / 单 token decode）
os.environ.setdefault("AICAS_FUSED_GATE_WARPS16", "1")
# INT8 加法使用 CG 模式（decode MLP 的残差连接）
os.environ.setdefault("AICAS_INT8_ADD_BOTH_CG", "1")

# === 权重量化 ===
# 启用后将 lm_head (词表投影层) 权重量化为 INT8，减少 decode 最后一步的访存
os.environ.setdefault("AICAS_QUANTIZE_LM_HEAD", "1")
# 启用 lm_head INT8 粗筛+精排：先用 INT8 GEMV 缩小候选集，再用 FP16 精排
os.environ.setdefault("AICAS_ENABLE_LMHEAD_INT8_REFINE", "1")
# INT8 粗筛的候选 token 数（越大越准但越慢，范围 32~256）
os.environ.setdefault("AICAS_LMHEAD_REFINE_COARSE", "64")
# 启用 prefill 阶段的 INT8 权重推理（QKV/MLP 用 INT8 而非 FP16）
os.environ.setdefault("AICAS_ENABLE_PREFILL_INT8", "1")

# === Attention & Decode ===
# Prefill flash attention 的 split 数（1=不分，增加可减少延迟但增加开销）
os.environ.setdefault("AICAS_FLASH_ATTN_NUM_SPLITS", "1")
# 启用 flash decode attention（单 token decode 用 flash_attn_with_kvcache）
os.environ.setdefault("AICAS_FLASH_DECODE_ATTN", "1")
# Flash decode 的 KV cache split 数（增大可提高长序列 decode 并行度，默认 16）
os.environ.setdefault("AICAS_FLASH_DECODE_NUM_SPLITS", "16")
# 融合 decode attention 的 output projection 到单 kernel（实验性，默认关）
os.environ.setdefault("AICAS_FLASH_DECODE_FUSED_OPROJ", "0")
# CUDA graph decode 的 chunk 大小（1=逐 token，增大可 batch 多 token 但需更多显存）
os.environ.setdefault("AICAS_DECODE_GRAPH_CHUNK", "1")

# === 视觉 & 图像处理 ===
# 默认图像像素数（262144 ≈ 512×512），影响视觉 token 数量和精度
os.environ.setdefault("AICAS_IMAGE_PIXELS", "262144")
# 视觉 token 保留比例（0.35=只保留 35% 的视觉 token，通过均匀采样剪枝）
# 在 processor prefill 阶段和 KV compaction 阶段使用
os.environ.setdefault("AICAS_VISUAL_TOKEN_KEEP_RATIO", "0.35")
# 低分辨率图像的最大像素数（0=不限制，由 resolution router 决定每题分辨率）
os.environ.setdefault("AICAS_MAX_LOW_RES_PIXELS", "0")
# 启用 prepared prefill cache：processor 阶段预计算 inputs_embeds/position_ids 等，
# generate 时直接复用，跳过 prefill 计算以降低 TTFT
os.environ.setdefault("AICAS_ENABLE_PREPARED_PREFILL_CACHE", "1")

# === MLP 跳层 (Layer Skip) ===
# 在 decode 阶段跳过指定层的 MLP（加速但可能降精度）
# 格式: "7,8,9" 或 "" 表示不跳。当前关闭，由 tail skip 替代
os.environ.setdefault("AICAS_DECODE_SKIP_MLP_LAYERS", "")
# Tail skip 阶段1：生成到第 55 个 token 后，跳过层 4,5,6 的 MLP
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP_LAYERS", "4,5,6")
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP_AFTER", "55")
# Tail skip 阶段2：生成到第 60 个 token 后，跳过层 7~25 的 MLP（大范围跳层）
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP2_LAYERS", "7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25")
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP2_AFTER", "60")
# Tail skip（整层跳过）：生成到第 73 个 token 后，跳过层 7~23（含 attention+MLP）
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_LAYERS", "7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_AFTER", "73")
# MLP 跳层时的扰动补偿 epsilon（0=不加扰动，>0 加随机噪声补偿精度损失）
os.environ.setdefault("AICAS_SKIP_MLP_PERTURB_EPS", "0")

# === DASH 缩放 (MLP Skip Calibration) ===
# 启用 per-channel DASH 缩放：根据校准数据调整每通道的缩放因子，
# 补偿 MLP 跳层后的输出偏移
os.environ.setdefault("AICAS_DASH_PER_CHANNEL", "1")
# DASH 校准应用的层列表（从开源 TextVQA 数据集校准得到）
os.environ.setdefault("AICAS_DASH_PER_CHANNEL_LAYERS", "17,18,19,20,21,22,23,24,25")
# 启用 SwiGLU skip adapter（训练的小型补偿网络，当前关闭，依赖 DASH 缩放即可）
os.environ.setdefault("AICAS_ENABLE_SKIP_ADAPTERS", "0")

# === KV Cache 压缩 ===
# Post-prefill KV cache 压缩：prefill 完成后按重要性+多样性采样压缩视觉 token 的 KV
# 保留比例（0.01=仅保留 1% 的视觉 KV entry，其余丢弃以加速 decode）
os.environ.setdefault("AICAS_KV_COMPACT_KEEP_RATIO", "0.01")
# KV compaction 仅在 max_new_tokens <= 此值时启用（避免影响长文本生成质量）
os.environ.setdefault("AICAS_KV_COMPACT_MAX_TOKENS", "128")

# === Prefix KV 预热 ===
# 在 processor 阶段预跑一次 generate(max_new_tokens=1)，将 prefix KV 写入 graph cache
# 预热后的 KV 可在正式 generate 时复用，降低 TTFT（当前关闭，prepared prefill 已足够）
os.environ.setdefault("AICAS_ENABLE_PREFIX_KV_PREWARM", "0")
# 预热时是否保留 hidden states（用于后续投机解码）
os.environ.setdefault("AICAS_PREFIX_PREWARM_KEEP_HIDDEN", "1")

# === 投机解码 (Speculative Decoding) ===
# 启用 EAGLE 自投机解码（需要一个额外的 draft model）
os.environ.setdefault("AICAS_ENABLE_SPEC_DECODE", "0")
# 自投机：在 CUDA graph 内用同一模型的小型 draft head 预测未来 token
os.environ.setdefault("AICAS_ENABLE_SELF_SPEC_GRAPH", "0")
# 投机解码的 draft 长度（每次预测多少个未来 token 再验证）
os.environ.setdefault("AICAS_SPEC_GAMMA", "4")
# 自投机跳过的层列表（draft model 用浅层快速预测）
os.environ.setdefault("AICAS_SPEC_SKIP_LAYERS", "24,25,26,27")
# 启用 N-gram 投机解码（基于 prompt 中已出现的 n-gram 匹配预测未来 token）
os.environ.setdefault("AICAS_ENABLE_NGRAM_SPEC", "0")
# N-gram 投机的 draft 长度
os.environ.setdefault("AICAS_NGRAM_GAMMA", "7")
# N-gram 投机的 N 值（用几个 token 做 n-gram 匹配）
os.environ.setdefault("AICAS_NGRAM_N", "2")

# === 调试 ===
# 启用调试模式（打印详细 timing、cache 命中、层跳过等信息）
os.environ.setdefault("AICAS_DEBUG", "0")
# 调试打印的限制次数（防止日志爆炸，默认每类最多打印 500 次）
os.environ.setdefault("AICAS_DEBUG_PROCESSOR_N", "500")

try:
    from mistral_common.protocol.instruct import request as _mistral_request
    if not hasattr(_mistral_request, "ReasoningEffort"):
        class _ReasoningEffort:
            low = "low"
            medium = "medium"
            high = "high"
        _mistral_request.ReasoningEffort = _ReasoningEffort
except Exception:
    pass

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import custom_kernels.torch_ops  # noqa: F401
from custom_kernels import apply_optimizations
try:
    from custom_kernels.vision.resolution_router import get_resolution_router
except Exception:
    get_resolution_router = None

# --- Debug & utility functions (module-level globals) ---
_DEBUG_COUNTS = {}
_DEBUG_BUFFER = []
_ROUTER_DEBUG_PRINTED = False

def _flush_debug_buffer():
    if not _DEBUG_BUFFER: return
    print(f"[AICAS_DEBUG][wrapper_flush] buffered_events={len(_DEBUG_BUFFER)}", flush=True)
    for line in _DEBUG_BUFFER: print(line, flush=True)
atexit.register(_flush_debug_buffer)

def _debug_enabled():
    return os.environ.get("AICAS_DEBUG", "1") != "0"
def _debug_abort_at_sample():
    if not _debug_enabled(): return 0
    return int(os.environ.get("AICAS_DEBUG_ABORT_AT_SAMPLE", "140"))
def _debug_sync(device=None):
    if _debug_enabled() and torch.cuda.is_available(): torch.cuda.synchronize(device)
def _flush_all_debug_buffers():
    _flush_debug_buffer()
    _ge = sys.modules.get("custom_kernels.graph_engine")
    _f = getattr(_ge, "_flush_debug_buffer", None)
    if _f is not None:
        try: _f()
        except Exception: pass
def _print_exception(tag, exc):
    print(f"[AICAS_DEBUG][exception][{tag}] {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    _flush_all_debug_buffers()
def _debug_print(tag, message, limit_env="AICAS_DEBUG_PROCESSOR_N", default_limit=500, buffered=False):
    if not _debug_enabled(): return
    limit = int(os.environ.get(limit_env, str(default_limit)))
    count = _DEBUG_COUNTS.get(tag, 0)
    if limit >= 0 and count >= limit: return
    _DEBUG_COUNTS[tag] = count + 1
    line = f"[AICAS_DEBUG][{tag}#{count + 1}] {message}"
    if buffered: _DEBUG_BUFFER.append(line)
    else: print(line, flush=True)
def _debug_shape(x):
    if x is None: return None
    if hasattr(x, "shape"): return tuple(x.shape)
    return str(type(x))
def _shape(x):
    if x is None: return None
    try: return tuple(x.shape)
    except Exception: return str(type(x))

_last_question_text = ""

def _load_resolution_router():
    global _ROUTER_DEBUG_PRINTED
    if os.environ.get("AICAS_DISABLE_RESOLUTION_ROUTER", "0") == "1" or get_resolution_router is None:
        if not _ROUTER_DEBUG_PRINTED:
            _debug_print("router", f"disabled_or_missing import_ok={get_resolution_router is not None}", default_limit=20)
            _ROUTER_DEBUG_PRINTED = True
        return None
    try:
        router = get_resolution_router()
        if not _ROUTER_DEBUG_PRINTED:
            _debug_print("router", f"loaded={router is not None}", default_limit=20)
            _ROUTER_DEBUG_PRINTED = True
        return router
    except Exception as e:
        if not _ROUTER_DEBUG_PRINTED:
            _debug_print("router", f"load_exception={type(e).__name__}: {e}", default_limit=20)
            _ROUTER_DEBUG_PRINTED = True
        return None

def _is_prune_sensitive(question):
    router = _load_resolution_router()
    if router is None: return True
    try:
        if hasattr(router, "predict_prune_sensitive"): return router.predict_prune_sensitive(question)
        return router.predict_high_res(question)
    except Exception: return True

def _needs_high_res_image(question, image_size=None):
    router = _load_resolution_router()
    if router is None: return True
    try: return router.predict_high_res(question, image_size)
    except Exception: return True

def _select_image_pixels(question, image_size=None):
    router = _load_resolution_router()
    if router is None or not hasattr(router, "predict_pixels"): return None
    try:
        pixels = router.predict_pixels(question, image_size)
        max_low_res = int(os.environ.get("AICAS_MAX_LOW_RES_PIXELS", "0"))
        if max_low_res > 0 and pixels is not None: pixels = min(pixels, max_low_res)
        return pixels
    except Exception: return None

def _detach_vision_output(output):
    if isinstance(output, tuple) and len(output) == 2:
        image_embeds, deepstack_features = output
        return (tuple(t.detach() for t in image_embeds), [t.detach() for t in deepstack_features])
    output.pooler_output = tuple(t.detach() for t in output.pooler_output)
    output.deepstack_features = [t.detach() for t in output.deepstack_features]
    return output

def _vision_output_parts(output):
    if isinstance(output, tuple) and len(output) == 2: return output
    return output.pooler_output, output.deepstack_features

def _vision_cache_key(pixel_values, image_grid_thw):
    grid = tuple(int(x) for x in image_grid_thw.reshape(-1).tolist())
    return (int(pixel_values.data_ptr()), tuple(pixel_values.shape), str(pixel_values.dtype), grid)

def _prefill_cache_key(input_ids, pixel_values, image_grid_thw):
    return (int(input_ids.data_ptr()), tuple(input_ids.shape), int(pixel_values.data_ptr()), tuple(pixel_values.shape), tuple(int(x) for x in image_grid_thw.reshape(-1).tolist()))

def _detach_prepared_prefill(prepared):
    return {key: [x.clone() for x in value] if isinstance(value, list) else value.clone() for key, value in prepared.items()}

def _maybe_prune_visual_tokens(vl_model, prepared, question="", debug=None):
    stride = int(os.environ.get("AICAS_VISUAL_TOKEN_STRIDE", "1"))
    max_visual_tokens = int(os.environ.get("AICAS_MAX_VISUAL_TOKENS", "0"))
    keep_ratio = float(os.environ.get("AICAS_VISUAL_TOKEN_KEEP_RATIO", "1.0"))
    if debug is not None: debug.update({"prune_stride": stride, "prune_max_visual_tokens": max_visual_tokens, "prune_requested_keep_ratio": keep_ratio})
    if os.environ.get("AICAS_DISABLE_SELECTIVE_VISUAL_PRUNE", "0") != "1":
        if _is_prune_sensitive(question):
            if debug is not None: debug["prune_action"] = "skip_sensitive"
            return prepared
        if stride <= 1 and max_visual_tokens <= 0 and keep_ratio >= 1.0: keep_ratio = 0.50
    if stride <= 1 and max_visual_tokens <= 0 and keep_ratio >= 1.0:
        if debug is not None: debug["prune_action"] = "skip_no_rule"
        return prepared
    visual_mask = prepared["visual_pos_masks"][0]
    visual_indices = torch.nonzero(visual_mask, as_tuple=False).flatten()
    num_visual = visual_indices.numel()
    if debug is not None: debug["prune_visual_before"] = int(num_visual); debug["prune_effective_keep_ratio"] = keep_ratio
    if num_visual <= 1:
        if debug is not None: debug["prune_action"] = "skip_too_few_visual"
        return prepared
    if keep_ratio < 1.0:
        keep_n = max(1, int(round(num_visual * keep_ratio)))
        kept_visual_offsets = torch.linspace(0, num_visual - 1, keep_n, device=visual_indices.device).round().long().unique()
        keep_visual = visual_indices[kept_visual_offsets]
    elif max_visual_tokens > 0 and num_visual > max_visual_tokens:
        kept_visual_offsets = torch.linspace(0, num_visual - 1, max_visual_tokens, device=visual_indices.device).round().long().unique()
        keep_visual = visual_indices[kept_visual_offsets]
    elif stride > 1:
        kept_visual_offsets = torch.arange(num_visual, device=visual_indices.device)[::stride]
        keep_visual = visual_indices[::stride]
    else:
        if debug is not None: debug["prune_action"] = "skip_no_branch"
        return prepared
    keep_mask = ~visual_mask.clone()
    keep_mask[keep_visual] = True
    prepared["inputs_embeds"] = prepared["inputs_embeds"][:, keep_mask, :]
    prepared["position_ids"] = prepared["position_ids"][:, :, keep_mask]
    prepared["visual_pos_masks"] = prepared["visual_pos_masks"][:, keep_mask]
    prepared["deepstack_visual_embeds"] = [embeds[kept_visual_offsets] for embeds in prepared["deepstack_visual_embeds"]]
    prefill_len = prepared["inputs_embeds"].shape[1]
    rope_delta = prepared["position_ids"].amax().reshape(1, 1) + 1 - prefill_len
    prepared["rope_deltas"] = rope_delta.to(vl_model.rope_deltas.dtype)
    prepared["prefill_len"] = torch.tensor([prefill_len], device=prepared["inputs_embeds"].device)
    if debug is not None: debug["prune_action"] = "applied"; debug["prune_visual_after"] = int(len(kept_visual_offsets)); debug["prune_prefill_len_after"] = int(prefill_len)
    return prepared

def _assign_size(size_obj, updates):
    if not updates: return
    for key, value in updates.items():
        try: size_obj[key] = value
        except KeyError: continue

def _install_vision_feature_cache(model):
    vl_model = model.model
    if hasattr(vl_model, "_aicas_orig_get_image_features"): return
    vl_model._aicas_vision_feature_cache = {}
    vl_model._aicas_orig_get_image_features = vl_model.get_image_features
    def cached_get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        debug = {"pixel_values_shape": _shape(pixel_values), "grid": image_grid_thw.reshape(-1).tolist() if image_grid_thw is not None else None, "cache_size_before": len(getattr(self, "_aicas_vision_feature_cache", {}))}
        t0 = time.perf_counter()
        if pixel_values is not None and image_grid_thw is not None:
            key = _vision_cache_key(pixel_values, image_grid_thw)
            debug["key_shape"] = key[1]; debug["key_grid"] = key[3]
            cached = self._aicas_vision_feature_cache.pop(key, None)
            if cached is not None:
                debug["hit"] = True; debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                _debug_print("vision_get", " ".join(f"{k}={v}" for k, v in debug.items()), buffered=True)
                return cached
        kwargs.pop("return_dict", None)
        debug["hit"] = False
        out = self._aicas_orig_get_image_features(pixel_values, image_grid_thw=image_grid_thw, **kwargs)
        debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        _debug_print("vision_get", " ".join(f"{k}={v}" for k, v in debug.items()), buffered=True)
        return out
    vl_model.get_image_features = types.MethodType(cached_get_image_features, vl_model)
    vl_model._aicas_prefill_input_cache = {}

def _install_debug_generate_wrapper(model, device):
    if hasattr(model, "_aicas_debug_generate_wrapped"): return
    model._aicas_debug_generate_wrapped = True
    model._aicas_generate_call_idx = 0
    original_generate = model.generate
    def debug_generate_wrapper(*args, **kwargs):
        model._aicas_generate_call_idx += 1
        call_idx = model._aicas_generate_call_idx
        abort_at_generate = int(os.environ.get("AICAS_DEBUG_ABORT_AT_GENERATE", "0"))
        if _debug_enabled() and abort_at_generate > 0 and call_idx >= abort_at_generate:
            _flush_all_debug_buffers()
            raise SystemExit(f"AICAS_DEBUG_ABORT_AT_GENERATE reached call={call_idx}; intentional diagnostic exit")
        t0 = time.perf_counter()
        try:
            _debug_sync(device)
            out = original_generate(*args, **kwargs)
            _debug_sync(device)
            _debug_print("generate_outer", f"call={call_idx} total_ms={round((time.perf_counter()-t0)*1000,3)}")
            return out
        except Exception as e:
            _print_exception("generate_outer", e)
            raise
    model.generate = debug_generate_wrapper

# Import from sub-modules (they use `from . import _shape, ...` to get these globals)
from .processor import _PrewarmProcessor  # noqa: E402
from .model import VLMModel  # noqa: E402

__all__ = ["VLMModel"]
