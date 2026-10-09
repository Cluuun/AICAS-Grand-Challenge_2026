from __future__ import annotations

"""Processor-side image resolution bucketing for Qwen3-VL inputs."""

import math
from copy import deepcopy
from types import MethodType
from typing import Iterable, List, Optional, Tuple

from PIL import Image

from .conf import conf_bool, conf_str


Size = Tuple[int, int]


def _parse_size(value: str, default: Size) -> Size:
    text = (value or "").strip().lower().replace("*", "x")
    if "x" not in text:
        return default
    left, right = text.split("x", 1)
    try:
        w = max(1, int(left.strip()))
        h = max(1, int(right.strip()))
    except Exception:
        return default
    return w, h


def _parse_buckets(value: str) -> List[Size]:
    buckets: List[Size] = []
    for part in (value or "").split(","):
        size = _parse_size(part, (0, 0))
        if size[0] > 0 and size[1] > 0:
            buckets.append(size)
    return buckets


def _nearest_bucket(width: int, height: int, buckets: Iterable[Size]) -> Size:
    src_aspect = max(width, 1) / max(height, 1)
    src_area = max(width * height, 1)
    best: Optional[Tuple[float, Size]] = None
    for bw, bh in buckets:
        b_aspect = bw / bh
        b_area = bw * bh
        # Aspect ratio dominates; area is a tie-breaker for nearby buckets.
        score = 2.0 * abs(math.log(src_aspect / b_aspect)) + 0.25 * abs(math.log(src_area / b_area))
        item = (score, (bw, bh))
        if best is None or item < best:
            best = item
    if best is None:
        return width, height
    return best[1]


def _resize_image(image: Image.Image, size: Size) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.BICUBIC)


def _rewrite_content_item(item, policy: str, fixed_size: Size, buckets: List[Size]):
    if not isinstance(item, dict) or item.get("type") != "image":
        return item
    image = item.get("image")
    if not isinstance(image, Image.Image):
        return item
    if policy == "fixed":
        target = fixed_size
    elif policy == "bucket":
        target = _nearest_bucket(image.width, image.height, buckets)
    else:
        return item
    new_item = dict(item)
    new_item["image"] = _resize_image(image, target)
    return new_item


def _rewrite_messages(messages, policy: str, fixed_size: Size, buckets: List[Size]):
    if policy == "dynamic":
        return messages
    rewritten = deepcopy(messages)
    for message in rewritten:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = [_rewrite_content_item(item, policy, fixed_size, buckets) for item in content]
    return rewritten


def apply_resolution_bucket_processor(processor) -> bool:
    if not conf_bool("ENABLE_VISION_RESOLUTION_POLICY", "0"):
        return False
    if getattr(processor, "_vision_resolution_policy_patched", False):
        return True

    policy = conf_str("VISION_RESOLUTION_POLICY", "bucket", lower=True)
    if policy not in ("dynamic", "fixed", "bucket"):
        policy = "dynamic"
    if policy == "dynamic":
        return False

    fixed_size = _parse_size(conf_str("VISION_FIXED_RESOLUTION", "364x364", lower=True), (364, 364))
    buckets = _parse_buckets(
        conf_str(
            "VISION_BUCKET_RESOLUTIONS",
            "280x280,336x336,364x364,448x252,252x448,392x280,280x392",
            lower=True,
        )
    )
    if not buckets:
        buckets = [fixed_size]

    original_apply_chat_template = processor.apply_chat_template

    def apply_chat_template_with_resolution(self, messages, *args, _orig=original_apply_chat_template, **kwargs):
        rewritten = _rewrite_messages(messages, policy, fixed_size, buckets)
        return _orig(rewritten, *args, **kwargs)

    processor.apply_chat_template = MethodType(apply_chat_template_with_resolution, processor)
    processor._vision_resolution_policy_patched = True
    processor._vision_resolution_policy = policy
    processor._vision_resolution_buckets = buckets
    print(f"[vision_resolution] policy={policy} fixed={fixed_size} buckets={buckets}")
    return True
