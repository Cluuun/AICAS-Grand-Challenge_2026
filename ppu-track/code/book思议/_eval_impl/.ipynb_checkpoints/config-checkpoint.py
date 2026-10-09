"""环境变量配置、调试工具、通用辅助函数。

此模块保留完整副本，因为 custom_kernels/graph_engine/kv_cache.py 通过
import evaluation_wrapper.config 访问 _last_question_text 等可变全局变量。
"""

from typing import Dict
import atexit
import os
import sys
import time
import traceback
import types

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from PIL import Image
except ImportError:
    class Image:
        pass

# --- 环境变量配置（调参区，与 __init__.py 保持同步）---
# 所有变量用 setdefault，可在外部提前覆盖。详细说明见 __init__.py。

# === INT8 量化 & GEMV 解码 ===
os.environ.setdefault("AICAS_INT8_GEMV_WARPS16", "1")          # INT8 GEMV 16-warps 配置
os.environ.setdefault("AICAS_INT8_GEMV_CG", "1")               # INT8 GEMV Cooperative Groups 模式
os.environ.setdefault("AICAS_FUSED_GATE_WARPS24", "1")         # 融合 gate_up 24-warps GEMV
os.environ.setdefault("AICAS_FUSED_GATE_WARPS16", "1")         # 融合 gate_up 16-warps GEMV
os.environ.setdefault("AICAS_INT8_ADD_BOTH_CG", "1")           # INT8 加法 CG 模式

# === 权重量化 ===
os.environ.setdefault("AICAS_QUANTIZE_LM_HEAD", "1")           # 将 lm_head 量化为 INT8
os.environ.setdefault("AICAS_ENABLE_LMHEAD_INT8_REFINE", "1")  # INT8 粗筛 + FP16 精排
os.environ.setdefault("AICAS_LMHEAD_REFINE_COARSE", "64")      # 粗筛候选 token 数
os.environ.setdefault("AICAS_ENABLE_PREFILL_INT8", "1")        # Prefill 阶段 INT8 权重推理

# === Attention & Decode ===
os.environ.setdefault("AICAS_FLASH_ATTN_NUM_SPLITS", "1")      # Prefill flash attention split 数
os.environ.setdefault("AICAS_FLASH_DECODE_ATTN", "1")          # 启用 flash decode attention
os.environ.setdefault("AICAS_FLASH_DECODE_NUM_SPLITS", "16")   # Flash decode KV split 数
os.environ.setdefault("AICAS_FLASH_DECODE_FUSED_OPROJ", "0")   # 融合 decode output projection
os.environ.setdefault("AICAS_DECODE_GRAPH_CHUNK", "1")         # CUDA graph decode chunk 大小

# === 视觉 & 图像处理 ===
os.environ.setdefault("AICAS_IMAGE_PIXELS", "262144")          # 默认图像像素数 (512×512)
os.environ.setdefault("AICAS_VISUAL_TOKEN_KEEP_RATIO", "0.35") # 视觉 token 保留比例
os.environ.setdefault("AICAS_MAX_LOW_RES_PIXELS", "0")         # 低分辨率图像最大像素数
os.environ.setdefault("AICAS_ENABLE_PREPARED_PREFILL_CACHE", "1")  # 预计算 prefill cache

# === MLP 跳层 (Layer Skip) ===
os.environ.setdefault("AICAS_DECODE_SKIP_MLP_LAYERS", "")      # 全局跳过的 MLP 层
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP_LAYERS", "4,5,6")     # Tail skip 阶段1层
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP_AFTER", "55")         # Tail skip 阶段1起始 token
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP2_LAYERS", "7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25")  # Tail skip 阶段2层
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_MLP2_AFTER", "60")        # Tail skip 阶段2起始 token
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_LAYERS", "7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")  # 整层跳过
os.environ.setdefault("AICAS_DECODE_TAIL_SKIP_AFTER", "73")             # 整层跳过起始 token
os.environ.setdefault("AICAS_SKIP_MLP_PERTURB_EPS", "0")                # 跳层扰动补偿 epsilon

# === DASH 缩放 (MLP Skip Calibration) ===
os.environ.setdefault("AICAS_DASH_PER_CHANNEL", "1")            # Per-channel DASH 校准缩放
os.environ.setdefault("AICAS_DASH_PER_CHANNEL_LAYERS", "17,18,19,20,21,22,23,24,25")  # DASH 层
os.environ.setdefault("AICAS_ENABLE_SKIP_ADAPTERS", "0")        # SwiGLU skip adapter

# === KV Cache 压缩 ===
os.environ.setdefault("AICAS_KV_COMPACT_KEEP_RATIO", "0.01")   # KV 压缩保留比例
os.environ.setdefault("AICAS_KV_COMPACT_MAX_TOKENS", "128")    # KV 压缩仅短文本启用

# === Prefix KV 预热 ===
os.environ.setdefault("AICAS_ENABLE_PREFIX_KV_PREWARM", "0")   # Prefix KV 预跑 generate
os.environ.setdefault("AICAS_PREFIX_PREWARM_KEEP_HIDDEN", "1") # 预热时保留 hidden states

# === 投机解码 （不好用 关了 忽略）===
os.environ.setdefault("AICAS_ENABLE_SPEC_DECODE", "0")          # EAGLE 自投机解码
os.environ.setdefault("AICAS_ENABLE_SELF_SPEC_GRAPH", "0")     # CUDA graph 内自投机
os.environ.setdefault("AICAS_SPEC_GAMMA", "4")                 # Draft 长度
os.environ.setdefault("AICAS_SPEC_SKIP_LAYERS", "24,25,26,27") # Draft 跳层
os.environ.setdefault("AICAS_ENABLE_NGRAM_SPEC", "0")          # N-gram 投机解码
os.environ.setdefault("AICAS_NGRAM_GAMMA", "7")                # N-gram draft 长度
os.environ.setdefault("AICAS_NGRAM_N", "2")                    # N-gram N 值

# === 调试 ===
os.environ.setdefault("AICAS_DEBUG", "0")                       # 启用调试模式
os.environ.setdefault("AICAS_DEBUG_PROCESSOR_N", "500")         # 调试打印限制次数

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

_DEBUG_COUNTS = {}
_DEBUG_BUFFER = []
_ROUTER_DEBUG_PRINTED = False


def _flush_debug_buffer():
    if not _DEBUG_BUFFER:
        return
    print(f"[AICAS_DEBUG][wrapper_flush] buffered_events={len(_DEBUG_BUFFER)}", flush=True)
    for line in _DEBUG_BUFFER:
        print(line, flush=True)

atexit.register(_flush_debug_buffer)


def _debug_enabled():
    return os.environ.get("AICAS_DEBUG", "1") != "0"


def _debug_abort_at_sample():
    if not _debug_enabled():
        return 0
    return int(os.environ.get("AICAS_DEBUG_ABORT_AT_SAMPLE", "140"))


def _debug_sync(device=None):
    if _debug_enabled() and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _flush_all_debug_buffers():
    _flush_debug_buffer()
    graph_engine = sys.modules.get("custom_kernels.graph_engine")
    flush = getattr(graph_engine, "_flush_debug_buffer", None)
    if flush is not None:
        try:
            flush()
        except Exception:
            pass


def _print_exception(tag, exc):
    print(f"[AICAS_DEBUG][exception][{tag}] {type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    _flush_all_debug_buffers()


def _debug_print(tag, message, limit_env="AICAS_DEBUG_PROCESSOR_N", default_limit=500, buffered=False):
    if not _debug_enabled():
        return
    limit = int(os.environ.get(limit_env, str(default_limit)))
    count = _DEBUG_COUNTS.get(tag, 0)
    if limit >= 0 and count >= limit:
        return
    _DEBUG_COUNTS[tag] = count + 1
    line = f"[AICAS_DEBUG][{tag}#{count + 1}] {message}"
    if buffered:
        _DEBUG_BUFFER.append(line)
    else:
        print(line, flush=True)


def _debug_shape(x):
    if x is None:
        return None
    if hasattr(x, "shape"):
        return tuple(x.shape)
    return str(type(x))


def _shape(x):
    if x is None:
        return None
    try:
        return tuple(x.shape)
    except Exception:
        return str(type(x))


def _assign_size(size_obj, updates):
    if not updates:
        return
    for key, value in updates.items():
        try:
            size_obj[key] = value
        except KeyError:
            continue


_last_question_text = ""
