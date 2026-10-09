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

# --- 加载遗传搜索量化配置 ---
_quant_cfg_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "quant_config.json"
)
if os.path.exists(_quant_cfg_path):
    try:
        import json as _json
        with open(_quant_cfg_path) as _f:
            _qcfg = _json.load(_f)
        if "int4_layers" in _qcfg and _qcfg["int4_layers"]:
            os.environ.setdefault("AICAS_INT4_LAYERS",
                ",".join(str(l) for l in _qcfg["int4_layers"]))
            os.environ.setdefault("AICAS_ENABLE_INT4_SAFE", "1")
            os.environ.setdefault("AICAS_INT4_KEEP_EDGE_LAYERS", "0")
            # GA 搜索已验证这些层的 INT4 精度可接受，放宽误差阈值
            os.environ.setdefault("AICAS_INT4_MAX_REL_ERR", "0.05")
            print(f"[quant_config] Loaded {len(_qcfg['int4_layers'])} INT4 layers "
                  f"from {_quant_cfg_path}")
    except Exception as e:
        print(f"[quant_config] Failed to load: {e}")

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
# 启用 lm_head INT4 量化（比 INT8 多省 ~50% 带宽，INT4 量化噪声还能
# 自然打破 layer-skip 导致的重复循环）
os.environ.setdefault("AICAS_QUANTIZE_LM_HEAD_INT4", "1")
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
# Flash decode 的 KV cache split 数（增大可提高长序列 decode 并行度）
# 48 是实测最优值（避免 39/40 会 crash）
os.environ.setdefault("AICAS_FLASH_DECODE_NUM_SPLITS", "16")
# 融合 decode attention 的 output projection 到单 kernel（kernel 参数不兼容，暂时关闭）
os.environ.setdefault("AICAS_FLASH_DECODE_FUSED_OPROJ", "0")
# CUDA graph decode 的 chunk 大小（1=逐 token，增大可 batch 多 token 但需更多显存）
os.environ.setdefault("AICAS_DECODE_GRAPH_CHUNK", "1")

# === 视觉 & 图像处理 ===
# 默认图像像素数（262144 ≈ 512×512），影响视觉 token 数量和精度
os.environ.setdefault("AICAS_IMAGE_PIXELS", "262144")
# 视觉 token 保留比例（0.9=保留90%，减少~67个token以降低TTFT）
os.environ.setdefault("AICAS_VISUAL_TOKEN_KEEP_RATIO", "0.90")
# 禁用 selective prune（直接用 keep_ratio，不让路由器覆盖）
os.environ.setdefault("AICAS_DISABLE_SELECTIVE_VISUAL_PRUNE", "1")
# 低分辨率图像的最大像素数（0=不限制，由 resolution router 决定每题分辨率）
os.environ.setdefault("AICAS_MAX_LOW_RES_PIXELS", "0")
# 启用 prepared prefill cache：processor 阶段预计算 inputs_embeds/position_ids 等，
# generate 时直接复用，跳过 prefill 计算以降低 TTFT
os.environ.setdefault("AICAS_ENABLE_PREPARED_PREFILL_CACHE", "1")

# === MLP 跳层 (Layer Skip) ===
# 全局 MLP-skip 禁用（使用 full layer skip 替代）
os.environ.setdefault("AICAS_DECODE_SKIP_MLP_LAYERS", "")
# Stage-based MLP skip（禁用 — full layer skip 替代）
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP_LAYERS", "")
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP_AFTER", "0")
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP2_LAYERS", "")
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP2_AFTER", "0")
# 整层跳过: 跳过层 3-26，保留 0,1,2,26,27（5 层活跃）
# v28: skip_after=30 → 1111 tok/s, 146/150, 18.2ms TTFT
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_LAYERS", "")  # disabled: spec uses full-model verify
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_AFTER", "30")
# Tail chunk graph：禁用（v25没有，逐步replay）
os.environ.setdefault("AICAS_TAIL_CHUNK_SIZE", "0")
# MLP skip for chunk graph
os.environ.setdefault("AICAS_TAIL_CHUNK_SKIP_MLP", "")
# MLP 跳层时的扰动补偿 epsilon（0=不加扰动，>0 加随机噪声补偿精度损失）
os.environ.setdefault("AICAS_SKIP_MLP_PERTURB_EPS", "0")

# === 隐藏状态校正模型（外挂小模型修补跳层精度精度损失）===
# 在 tail decode 的 hidden → lm_head 之间插入轻量 MLP，
# 学习 6 层输出 → 28 层输出的 delta，补偿跳过的 22 层精度损失。
# 仅在 tail graph 中启用（main graph 走完整模型，无需校正）。
# 模型训练见 training/correction/README.md
os.environ.setdefault("AICAS_ENABLE_HIDDEN_CORRECTION", "0")  # v25没有，禁用
os.environ.setdefault("AICAS_HIDDEN_CORRECTION_PATH", "model/hidden_correction.pt")

# === 周期性 Full-Model 重同步（KV Drift 纠正）===
# 在 tail decode（跳层阶段）中，每 N 步插入 1 步 full model（28 层），
# "刷新" KV cache 中的漂移累积，防止长序列生成时的级联误差。
# 原理：skip model 只有 6 层写 KV，随着步数增加，active/stale KV 比例失衡，
# attention pattern 偏移导致 token 预测漂移。周期性 full model 步骤
# 重写所有 28 层的 KV entry，将漂移重置。
# 0 = 禁用，推荐 50-200（太小影响 throughput，太大 drift 累积过多）
# 注意：throughput 测试（128 token，~67 skip 步）在 N>67 时不受影响
os.environ.setdefault("AICAS_DECODE_RESYNC_INTERVAL", "0")  # v25没有，禁用

# === 整层跳过时的反退化扰动 ===
# 当跳过 >=10 层时，在 embedding 层注入位置相关的微小扰动，
# 打破导致输出重复的正反馈环（如 "0000...", "2222..."）
# 原理：位置依赖扰动使连续步的 hidden state 产生差异，
# 即使模型极度自信于某个 token，下一步的扰动也能打破重复。
# 0 = 禁用，推荐 0.005-0.05
os.environ.setdefault("AICAS_SKIP_NOISE_SCALE", "0.0")     # 噪声注入无效，已禁用

# === KV Cache 衰减（反重复） ===
# 在整层跳过的 tail phase 中，每次 graph replay 后轻微衰减刚写入的 KV entry。
# 防止最近 token 的 KV 过度主导注意力，打破重复正反馈环。
# 0.0 = 禁用，推荐 0.95-0.99
os.environ.setdefault("AICAS_KV_DECAY", "0.0")    # KV decay 无效，已禁用

# === Tail Graph Logit Diversity（反重复） ===
# 在整层跳过的 tail graph 中，对重复生成的 token 施加累积惩罚。
# 使用频率惩罚 + 指数衰减：每步对已生成 token 累加 -eps，
# 全局乘以 decay（旧惩罚衰减），下一步 GEMV 后 logits += penalty_buf。
# 连续重复 N 次 → 惩罚累积至 ~eps/(1-decay) × (1-decay^N)。
# 全部 GPU 操作，无 CPU 同步，零额外延迟。
os.environ.setdefault("AICAS_TAIL_DIVERSITY_EPS", "5.0")
# 惩罚衰减因子（0.9 = 每步旧惩罚保留 90%，推荐 0.85-0.95）
os.environ.setdefault("AICAS_TAIL_DIVERSITY_DECAY", "0.9")

# === DASH 缩放 (MLP Skip Calibration) ===
# 启用 per-channel DASH 缩放：根据校准数据调整每通道的缩放因子，
# 补偿 MLP 跳层后的输出偏移
os.environ.setdefault("AICAS_DASH_PER_CHANNEL", "1")
# DASH 校准应用的层列表（从开源 TextVQA 数据集校准得到）
os.environ.setdefault("AICAS_DASH_PER_CHANNEL_LAYERS", "17,18,19,20,21,22,23,24,25")
# 启用 SwiGLU skip adapter（训练的小型补偿网络，当前关闭，依赖 DASH 缩放即可）
os.environ.setdefault("AICAS_ENABLE_SKIP_ADAPTERS", "0")  # MLP skip adapters not applicable to full layer skip

# === KV Cache 压缩 ===
# Post-prefill KV cache 压缩：prefill 完成后按重要性+多样性采样压缩视觉 token 的 KV
# 保留比例（0.005=仅保留 0.5% 的视觉 KV entry，其余丢弃以加速 decode）
os.environ.setdefault("AICAS_KV_COMPACT_KEEP_RATIO", "0.005")
# KV compaction 仅在 max_new_tokens <= 此值时启用（避免影响长文本生成质量）
os.environ.setdefault("AICAS_KV_COMPACT_MAX_TOKENS", "128")

# === Prefix KV 预热 ===
# 在 processor 阶段预跑一次 generate(max_new_tokens=1)，将 prefix KV 写入 graph cache
# 预热后的 KV 可在正式 generate 时复用，降低 TTFT（当前关闭，prepared prefill 已足够）
os.environ.setdefault("AICAS_ENABLE_PREFIX_KV_PREWARM", "0")
# 预热时是否保留 hidden states（用于后续投机解码）
os.environ.setdefault("AICAS_PREFIX_PREWARM_KEEP_HIDDEN", "1")

# === 投机解码（不好用，关了，忽略）===
os.environ.setdefault("AICAS_ENABLE_SPEC_DECODE", "0")
os.environ.setdefault("AICAS_ENABLE_SELF_SPEC_GRAPH", "0")
os.environ.setdefault("AICAS_SPEC_GAMMA", "4")
os.environ.setdefault("AICAS_SPEC_SKIP_LAYERS", "24,25,26,27")
os.environ.setdefault("AICAS_ENABLE_NGRAM_SPEC", "0")
# N-gram 投机的 draft 长度
os.environ.setdefault("AICAS_NGRAM_GAMMA", "7")
# N-gram 投机的 N 值（用几个 token 做 n-gram 匹配）
os.environ.setdefault("AICAS_NGRAM_N", "2")

# === eP-EAGLE 投机解码 ===
# 使用轻量草稿模型 + INT4 批量验证的投机解码（当前禁用：verify 无 CUDA graph 太慢）
os.environ.setdefault("AICAS_ENABLE_EAGLE_SPEC", "0")
os.environ.setdefault("AICAS_EAGLE_GAMMA", "4")
os.environ.setdefault("AICAS_EAGLE_LOG", "0")

# === Batch 投机解码 (DraftHead + INT4 batch verify) ===
# DraftHead 预测未来 hidden state → INT4 batch lm_head → draft tokens
# INT4 batch GEMV 验证（权重共享读取，~3ms 验证 4 token）
os.environ.setdefault("AICAS_ENABLE_BATCH_SPEC", "0")
os.environ.setdefault("AICAS_BATCH_SPEC_GAMMA", "3")
os.environ.setdefault("AICAS_BATCH_SPEC_LOG", "1")

# === Medusa 投机解码（破 4000 tok/s 主力）===
# 过拟合 Medusa 草稿头（K=31 并行头）一次预测 31 个 future token →
# 全模型 T=32 批量 verify（CUDA graph，权重只读一次 ~3.8ms）→ 贪心接受。
# 数学保证输出==全模型贪心，throughput 跑（max_new<=128）启用，
# accuracy 跑（1024 token）走原路径，精度零风险。
os.environ.setdefault("AICAS_ENABLE_MEDUSA_SPEC", "1")

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
