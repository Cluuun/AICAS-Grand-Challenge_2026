"""Static IDs, dims, and env-driven config knobs for vAxel30."""
from __future__ import annotations

import os

import torch
import torch._dynamo

from prune_optim import PruneConfig


# ───── env-driven flags (mirror vAxel29) ─────
AICAS_QUANT_ENABLE = os.environ.get("AICAS_QUANT_ENABLE", "1") == "1"
AICAS_QUANT_LM_HEAD = os.environ.get("AICAS_QUANT_LM_HEAD", "1") == "1"
AICAS_QUANT_DECODE_LLM = os.environ.get("AICAS_QUANT_DECODE_LLM", "1") == "1"
AICAS_QUANT_VIT_FC1 = os.environ.get("AICAS_QUANT_VIT_FC1", "0") == "1"
AICAS_QUANT_VIT_FC1_LO = int(os.environ.get("AICAS_QUANT_VIT_FC1_LO", "3"))
AICAS_QUANT_VIT_FC1_HI = int(os.environ.get("AICAS_QUANT_VIT_FC1_HI", "23"))
AICAS_QUANT_VIT_CALIBRATE = os.environ.get("AICAS_QUANT_VIT_CALIBRATE", "0") == "1"
AICAS_QUANT_N_CALIB = int(os.environ.get("AICAS_QUANT_N_CALIB", "32"))
AICAS_QUANT_VIT_ALPHA = float(os.environ.get("AICAS_QUANT_VIT_ALPHA", "0.5"))
AICAS_QUANT_TEXTVQA_DIR = os.environ.get(
    "AICAS_QUANT_TEXTVQA_DIR",
    "/root/aicas/quant_optim_workspace/textvqa_train")

AICAS_VOCAB_WHITELIST = os.environ.get("AICAS_VOCAB_WHITELIST", "1") == "1"
AICAS_VOCAB_K = int(os.environ.get("AICAS_VOCAB_K", "32000"))
AICAS_VOCAB_TEXTVQA_DIR = os.environ.get(
    "AICAS_VOCAB_TEXTVQA_DIR",
    "/root/aicas/quant_optim_workspace/textvqa_train")
AICAS_VOCAB_CORPUS = os.environ.get(
    "AICAS_VOCAB_CORPUS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "textvqa_calib_corpus.txt"))

AICAS_MAX_PIXELS = int(os.environ.get("AICAS_MAX_PIXELS", "327680"))
AICAS_VIT_PRUNE_RATIO = float(os.environ.get("AICAS_VIT_PRUNE_RATIO", "0.7"))
AICAS_VIT_PRUNE_MIN_K = int(os.environ.get("AICAS_VIT_PRUNE_MIN_K", "64"))
AICAS_VIT_PRUNE_BUCKET = int(os.environ.get("AICAS_VIT_PRUNE_BUCKET", "32"))


# B4 fix: 64 was speculative — keep dynamo's eager fallback within reach so
# unexpected recompiles surface fast instead of churning autotune for minutes.
torch._dynamo.config.recompile_limit = 8


PRUNE_CONFIG = PruneConfig(
    enabled=AICAS_VIT_PRUNE_RATIO > 0.0,
    prune_ratio=AICAS_VIT_PRUNE_RATIO,
    min_k=AICAS_VIT_PRUNE_MIN_K,
    bucket=AICAS_VIT_PRUNE_BUCKET,
    max_pixels=AICAS_MAX_PIXELS,
)


MAX_CACHE_LEN = 1280
MAX_CACHE_LEN_ACC = 2176
N_DECODE_STEPS = 32

NUM_LLM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_Q_HEADS = 16

IM_START_ID = 151644
IM_END_ID = 151645
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655
NEWLINE_ID = 198
USER_ID = 872
ASSISTANT_ID = 77091

CHAT_PREFIX = [IM_START_ID, USER_ID, NEWLINE_ID, VISION_START_ID]
CHAT_SUFFIX_BEFORE_Q = [VISION_END_ID]
CHAT_SUFFIX_AFTER_Q = [IM_END_ID, NEWLINE_ID, IM_START_ID, ASSISTANT_ID, NEWLINE_ID]
