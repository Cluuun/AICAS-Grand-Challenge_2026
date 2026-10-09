from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from vision_route_policy import question_route


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw.strip()) if raw is not None and raw.strip() else int(default)


def _env_csv_ints(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _env_csv_floats(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class VisionGridSpec:
    grid_t: int
    grid_h: int
    grid_w: int
    family: str
    long_edge_px: int
    aspect_ratio: float
    bucket_id: str

    @property
    def grid_thw(self) -> tuple[int, int, int]:
        return (self.grid_t, self.grid_h, self.grid_w)


def _round_grid_cells(value: float, grid_align: int) -> int:
    if grid_align <= 1:
        return max(1, int(round(value)))
    return max(grid_align, int(round(value / grid_align) * grid_align))


def _build_vision_resize_whitelist(*, patch_size: int, align: int) -> tuple[VisionGridSpec, ...]:
    common_long_edges = _env_csv_ints("AICAS_VISION_COMMON_LONG_EDGES", (1024, 896, 768))
    aggressive_long_edges = _env_csv_ints(
        "AICAS_VISION_AGGRESSIVE_LONG_EDGES",
        (736, 704, 640, 608, 576, 512, 480, 448, 384, 320, 288),
    )
    whitelist_mode = os.getenv("AICAS_VISION_GRID_WHITELIST_MODE", "common+aggressive").strip().lower()
    aspect_ratios = _env_csv_floats(
        "AICAS_VISION_WHITELIST_ASPECT_RATIOS",
        (1.0 / 3.0, 1.0 / 2.0, 9.0 / 16.0, 3.0 / 4.0, 1.0, 4.0 / 3.0, 16.0 / 9.0, 2.0, 3.0),
    )
    grid_align = max(1, align // max(1, patch_size))
    families: list[tuple[str, tuple[int, ...]]] = [
        ("common", tuple(int(edge) for edge in common_long_edges if int(edge) > 0)),
    ]
    if whitelist_mode not in {"common", "hot", "hotspots"}:
        families.append(("aggressive", tuple(int(edge) for edge in aggressive_long_edges if int(edge) > 0)))

    specs: dict[tuple[int, int, int], VisionGridSpec] = {}
    for family, long_edges in families:
        for long_edge_px in long_edges:
            long_cells = _round_grid_cells(long_edge_px / float(max(1, patch_size)), grid_align)
            for aspect_ratio in aspect_ratios:
                if aspect_ratio <= 0:
                    continue
                if aspect_ratio >= 1.0:
                    grid_w = long_cells
                    grid_h = _round_grid_cells(long_cells / aspect_ratio, grid_align)
                else:
                    grid_h = long_cells
                    grid_w = _round_grid_cells(long_cells * aspect_ratio, grid_align)
                spec = VisionGridSpec(
                    grid_t=1,
                    grid_h=int(grid_h),
                    grid_w=int(grid_w),
                    family=family,
                    long_edge_px=int(long_edge_px),
                    aspect_ratio=float(aspect_ratio),
                    bucket_id=f"{family}:l{long_edge_px}:g{grid_h}x{grid_w}",
                )
                existing = specs.get(spec.grid_thw)
                if existing is None or (existing.family != "common" and spec.family == "common"):
                    specs[spec.grid_thw] = spec
    return tuple(sorted(specs.values(), key=lambda spec: (spec.long_edge_px, spec.grid_h, spec.grid_w)))


class FusionResolutionProcessor:
    """Processor wrapper that disables resampling by default and only pads when alignment is required."""

    _DETAIL_KEYWORDS = (
        "text",
        "word",
        "letter",
        "sign",
        "label",
        "number",
        "read",
        "written",
        "table",
        "chart",
        "graph",
        "price",
        "date",
        "time",
        "count",
        "how many",
        "small",
        "logo",
        "menu",
        "document",
    )

    def __init__(self, processor, model_ref=None) -> None:
        self._processor = processor
        self._disabled = _truthy("JUNKRAT_DISABLE_RESOLUTION_POLICY", "0")
        self._enabled = _truthy("AICAS_VISION_TOKEN_OPT", "1")
        self._mode = os.getenv("AICAS_VISION_RESIZE_MODE", "off").strip().lower()
        self._route_policy = os.getenv("AICAS_VISION_ROUTE_POLICY", "offline_v1").strip().lower()
        self._adaptive_profile = os.getenv("AICAS_VISION_ADAPTIVE_PROFILE", "conservative").strip().lower()
        self._align = max(1, _env_int("AICAS_VISION_BUCKET_ALIGN", _env_int("AICAS_VISION_ALIGN", 32)))
        self._fixed_long_edge = _env_int("AICAS_VISION_LONG_EDGE", 768)
        self._general_long_edge = _env_int("AICAS_VISION_GENERAL_LONG_EDGE", 560)
        self._ocr_long_edge = _env_int("AICAS_VISION_OCR_LONG_EDGE", 896)
        self._route_targets = {
            "hard_text": _env_int("AICAS_VISION_ROUTE_HARD_TEXT_LONG_EDGE", 640),
            "label_text": _env_int("AICAS_VISION_ROUTE_LABEL_TEXT_LONG_EDGE", 608),
            "numeric_text": _env_int("AICAS_VISION_ROUTE_NUMERIC_TEXT_LONG_EDGE", 608),
            "counting": _env_int("AICAS_VISION_ROUTE_COUNTING_LONG_EDGE", 512),
            "chart_table": _env_int("AICAS_VISION_ROUTE_CHART_LONG_EDGE", 608),
            "coarse": _env_int("AICAS_VISION_ROUTE_COARSE_LONG_EDGE", 320),
        }
        self._max_visual_tokens = _env_int("AICAS_VISION_MAX_VISUAL_TOKENS", 576)
        self._target_visual_tokens = _env_int("AICAS_VISION_TARGET_VISUAL_TOKENS", 0)
        bucket_env = os.getenv("AICAS_VISION_BUCKETS", "")
        self._long_edge_buckets = tuple(sorted({int(part.strip()) for part in bucket_env.split(",") if part.strip()}))
        self._bucket_mode = os.getenv("AICAS_VISION_BUCKET_MODE", "long_edge").strip().lower()
        self._bucket_strict = _truthy("AICAS_VISION_BUCKET_STRICT", "0")
        self._bucket_high_risk_up_only = _truthy("AICAS_VISION_BUCKET_HIGH_RISK_UP_ONLY", "0")
        self._log_enabled = _truthy("AICAS_VISION_LOG", "0") or _truthy("AICAS_VISION_ANALYSIS", "0")
        self._log_path = Path(os.getenv("AICAS_VISION_LOG_PATH", "analysis_outputs/vision_token_processor_log.jsonl"))

        vision_config = getattr(getattr(getattr(model_ref, "inner", None), "config", None), "vision_config", None)
        self._patch_size = int(getattr(vision_config, "patch_size", 16))
        self._merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
        image_processor = getattr(processor, "image_processor", None)
        processor_size = getattr(image_processor, "size", {}) or {}
        self._min_pixels = int(processor_size.get("shortest_edge", 0) or 0)
        self._max_pixels = int(processor_size.get("longest_edge", 0) or 0)
        self._processor_resize_factor = max(1, self._patch_size * self._merge_size)
        if image_processor is not None:
            image_processor.do_resize = False
        self._whitelist_specs = _build_vision_resize_whitelist(patch_size=self._patch_size, align=self._align)
        self._whitelist_lookup = {spec.grid_thw: spec for spec in self._whitelist_specs}
        self._whitelist_by_long_edge: dict[int, tuple[VisionGridSpec, ...]] = {}
        for spec in self._whitelist_specs:
            self._whitelist_by_long_edge.setdefault(spec.long_edge_px, []).append(spec)
        self._whitelist_by_long_edge = {
            int(long_edge): tuple(sorted(specs, key=lambda spec: (spec.aspect_ratio, spec.grid_h, spec.grid_w)))
            for long_edge, specs in self._whitelist_by_long_edge.items()
        }
        self._whitelist_long_edges = tuple(sorted(self._whitelist_by_long_edge))

    def __getattr__(self, name):
        return getattr(self._processor, name)

    @classmethod
    def _question_category(cls, text: str) -> str:
        normalized = text.lower()
        if "chart" in normalized or "graph" in normalized or "table" in normalized:
            return "chart"
        if "how many" in normalized or "count" in normalized:
            return "counting"
        if any(keyword in normalized for keyword in cls._DETAIL_KEYWORDS):
            return "ocr"
        if normalized.strip():
            return "general"
        return "other"

    @staticmethod
    def _question_text(content) -> str:
        if not isinstance(content, list):
            return ""
        return " ".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    @staticmethod
    def _nearest_value(value: int, candidates: tuple[int, ...], *, prefer_lower: bool = True) -> int:
        if not candidates:
            return value
        return min(candidates, key=lambda candidate: (abs(candidate - value), candidate if prefer_lower else -candidate))

    @staticmethod
    def _ceil_to_multiple(value: int, multiple: int) -> int:
        multiple = max(1, int(multiple))
        value = max(1, int(value))
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _padding_box(*, original_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = original_size
        target_w, target_h = target_size
        pad_w = max(0, target_w - width)
        pad_h = max(0, target_h - height)
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return left, top, right, bottom

    @staticmethod
    def _pad_image_without_resize(image: Image.Image, padding: tuple[int, int, int, int]) -> Image.Image:
        if not any(padding):
            return image
        left, top, right, bottom = padding
        array = np.asarray(image)
        if array.ndim < 2:
            raise ValueError(f"Unsupported image rank {array.ndim}; expected at least 2 dimensions.")
        pad_spec = [(top, bottom), (left, right)]
        for _ in range(array.ndim - 2):
            pad_spec.append((0, 0))
        padded = np.pad(array, tuple(pad_spec), mode="edge")
        result = Image.fromarray(padded)
        if result.mode != image.mode:
            result = result.convert(image.mode)
        return result

    def _alignment_canvas_size(self, *, original_size: tuple[int, int]) -> tuple[int, int]:
        width, height = original_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size {original_size}.")
        if self._max_pixels > 0 and width * height > self._max_pixels:
            raise ValueError(
                f"Image size {width}x{height} exceeds processor max_pixels={self._max_pixels}; "
                "downsampling is disabled, so the sample cannot be resized to fit."
            )

        factor = self._processor_resize_factor
        target_w = self._ceil_to_multiple(width, factor)
        target_h = self._ceil_to_multiple(height, factor)
        aspect_ratio = width / max(1.0, float(height))

        while self._min_pixels > 0 and target_w * target_h < self._min_pixels:
            candidates = (
                (self._ceil_to_multiple(target_w + factor, factor), target_h),
                (target_w, self._ceil_to_multiple(target_h + factor, factor)),
            )
            viable = []
            for candidate_w, candidate_h in candidates:
                candidate_pixels = candidate_w * candidate_h
                if self._max_pixels > 0 and candidate_pixels > self._max_pixels:
                    continue
                viable.append((candidate_w, candidate_h))
            if not viable:
                raise ValueError(
                    f"Image size {width}x{height} cannot satisfy processor min_pixels={self._min_pixels} "
                    "without resizing."
                )
            target_w, target_h = min(
                viable,
                key=lambda size: (
                    abs(math.log((size[0] / max(1.0, float(size[1]))) / max(aspect_ratio, 1e-6))),
                    size[0] * size[1],
                    size[1],
                    size[0],
                ),
            )

        if self._max_pixels > 0 and target_w * target_h > self._max_pixels:
            raise ValueError(
                f"Aligned canvas {target_w}x{target_h} exceeds processor max_pixels={self._max_pixels}; "
                "downsampling is disabled."
            )
        return target_w, target_h

    def _snap_long_edge_bucket(self, requested_long_edge: int) -> int:
        candidate_buckets = self._long_edge_buckets or self._whitelist_long_edges
        if requested_long_edge <= 0 or not candidate_buckets:
            return requested_long_edge
        return self._nearest_value(requested_long_edge, candidate_buckets)

    def _bucket_snap_long_edge(self, requested_long_edge: int, category: str) -> int:
        candidate_buckets = self._long_edge_buckets or self._whitelist_long_edges
        if not candidate_buckets:
            return requested_long_edge
        if self._bucket_high_risk_up_only and category in {"ocr", "counting", "chart"}:
            for candidate in sorted(candidate_buckets):
                if candidate >= requested_long_edge:
                    return candidate
            return max(candidate_buckets)
        return self._snap_long_edge_bucket(requested_long_edge)

    def _select_whitelist_spec(self, *, requested_long_edge: int, original_size: tuple[int, int]) -> VisionGridSpec:
        snapped_long_edge = self._snap_long_edge_bucket(requested_long_edge)
        original_w, original_h = original_size
        aspect_ratio = original_w / max(1.0, float(original_h))
        original_long = max(1.0, float(max(original_size)))
        def best_spec_for_edge(long_edge: int) -> VisionGridSpec | None:
            specs = self._whitelist_by_long_edge.get(long_edge)
            if not specs:
                return None
            return min(
                specs,
                key=lambda spec: (
                    abs(math.log(max(aspect_ratio, 1e-6) / max(spec.aspect_ratio, 1e-6))),
                    abs(math.log(original_long / max(float(spec.long_edge_px), 1.0))),
                    spec.grid_h * spec.grid_w,
                ),
            )

        selected = best_spec_for_edge(snapped_long_edge)
        if selected is None:
            raise RuntimeError(f"no whitelist grids available for long edge {snapped_long_edge}")
        if self._min_pixels <= 0:
            return selected
        if (selected.grid_w * self._patch_size) * (selected.grid_h * self._patch_size) >= self._min_pixels:
            return selected
        for long_edge in sorted(edge for edge in self._whitelist_long_edges if edge > snapped_long_edge):
            candidate = best_spec_for_edge(long_edge)
            if candidate is None:
                continue
            if (candidate.grid_w * self._patch_size) * (candidate.grid_h * self._patch_size) >= self._min_pixels:
                return candidate
        return selected

    def _estimate_visual_tokens_for_long_edge(self, *, original_size: tuple[int, int], target_long_edge: int) -> int:
        spec = self._select_whitelist_spec(requested_long_edge=target_long_edge, original_size=original_size)
        return int(spec.grid_t * spec.grid_h * spec.grid_w // (self._merge_size**2))

    def _target_long_edge(self, *, question: str, original_size: tuple[int, int]) -> int | None:
        if not self._enabled or self._mode == "off":
            return None
        category = self._question_category(question)
        candidate_buckets = self._long_edge_buckets or self._whitelist_long_edges
        if self._mode == "fixed":
            target_long_edge = self._fixed_long_edge
            if self._target_visual_tokens > 0 and candidate_buckets:
                for candidate in sorted(candidate_buckets, reverse=True):
                    if self._estimate_visual_tokens_for_long_edge(
                        original_size=original_size,
                        target_long_edge=candidate,
                    ) <= self._target_visual_tokens:
                        target_long_edge = candidate
                        break
            return self._bucket_snap_long_edge(target_long_edge, category)
        if self._mode == "adaptive":
            route = question_route(question)
            base = self._ocr_long_edge if route in {"hard_text", "label_text", "numeric_text", "counting", "chart_table"} else self._general_long_edge
            target_long_edge = self._bucket_snap_long_edge(base, category)
            if self._adaptive_profile != "hybrid" or not candidate_buckets:
                return target_long_edge
            for candidate in sorted(candidate_buckets, reverse=True):
                if candidate <= target_long_edge and self._estimate_visual_tokens_for_long_edge(
                    original_size=original_size,
                    target_long_edge=candidate,
                ) <= self._max_visual_tokens:
                    return candidate
            return min(candidate_buckets) if candidate_buckets else target_long_edge
        if self._route_policy == "off":
            return None
        route = question_route(question)
        return self._bucket_snap_long_edge(self._route_targets.get(route, self._route_targets["coarse"]), category)

    def _resize_image(self, image, *, question: str):
        if not isinstance(image, Image.Image):
            return image, None
        width, height = (int(image.size[0]), int(image.size[1]))
        original_size = (width, height)
        if self._disabled or not self._enabled or self._mode == "off":
            target_w, target_h = self._alignment_canvas_size(original_size=original_size)
            padding = self._padding_box(original_size=original_size, target_size=(target_w, target_h))
            prepared = self._pad_image_without_resize(image, padding)
            return prepared, {
                "policy": "off",
                "route": question_route(question),
                "original_size": [width, height],
                "resized_size": [width, height],
                "prepared_size": [target_w, target_h],
                "padding": [int(x) for x in padding],
                "resample_applied": False,
            }
        target_long = self._target_long_edge(question=question, original_size=original_size)
        route = question_route(question)
        if target_long is None or target_long <= 0:
            target_w, target_h = self._alignment_canvas_size(original_size=original_size)
            padding = self._padding_box(original_size=original_size, target_size=(target_w, target_h))
            prepared = self._pad_image_without_resize(image, padding)
            return prepared, {
                "policy": "off",
                "route": route,
                "original_size": [width, height],
                "resized_size": [width, height],
                "prepared_size": [target_w, target_h],
                "padding": [int(x) for x in padding],
                "resample_applied": False,
            }
        target_long = min(int(target_long), max(width, height))
        spec = self._select_whitelist_spec(requested_long_edge=target_long, original_size=original_size)
        target_w = int(spec.grid_w * self._patch_size)
        target_h = int(spec.grid_h * self._patch_size)
        resized = image if (target_w, target_h) == (width, height) else image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        return resized, {
            "policy": self._mode,
            "route": route,
            "original_size": [width, height],
            "resized_size": [target_w, target_h],
            "prepared_size": [target_w, target_h],
            "padding": [0, 0, 0, 0],
            "target_long_edge": target_long,
            "target_bucket": int(spec.long_edge_px),
            "bucket_id": spec.bucket_id,
            "target_grid_thw": [int(spec.grid_t), int(spec.grid_h), int(spec.grid_w)],
            "estimated_visual_tokens": int(spec.grid_t * spec.grid_h * spec.grid_w // (self._merge_size**2)),
            "align": self._align,
            "resample_applied": (target_w, target_h) != (width, height),
        }

    @staticmethod
    def _grid_tuple_from_tensor(image_grid_thw) -> tuple[int, int, int] | None:
        if image_grid_thw is None:
            return None
        values = tuple(int(x) for x in image_grid_thw.detach().view(-1).cpu().tolist())
        return values if len(values) == 3 else None

    def _rewrite_messages(self, messages):
        rewritten = copy.deepcopy(messages)
        records = []
        resize_ms = 0.0
        for message in rewritten:
            content = message.get("content") if isinstance(message, dict) else None
            question = self._question_text(content)
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                start = time.perf_counter()
                image, record = self._resize_image(item.get("image"), question=question)
                resize_ms += (time.perf_counter() - start) * 1000.0
                item["image"] = image
                if record is not None:
                    records.append(record)
        return rewritten, records, resize_ms

    def apply_chat_template(self, messages, *args, **kwargs):
        start = time.perf_counter()
        rewritten, records, resize_ms = self._rewrite_messages(messages)
        kwargs.setdefault("do_resize", False)
        result = self._processor.apply_chat_template(rewritten, *args, **kwargs)
        actual_grid = self._grid_tuple_from_tensor(getattr(result, "image_grid_thw", None))
        if records and actual_grid is not None:
            if self._mode != "off":
                expected_grid = tuple(records[0].get("target_grid_thw") or ())
                if expected_grid and actual_grid != expected_grid:
                    if self._bucket_strict:
                        raise AssertionError(f"target whitelist grid {expected_grid} != actual grid {actual_grid}")
                    records[0]["grid_miss_reason"] = "processor_adjusted_grid"
            records[0]["actual_grid_thw"] = list(actual_grid)
            records[0]["whitelist_hit"] = actual_grid in self._whitelist_lookup
        if self._log_enabled:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "processor_apply_chat_template_ms": (time.perf_counter() - start) * 1000.0,
                "resize_time_ms": resize_ms,
                "resolution": records,
            }
            if hasattr(result, "image_grid_thw"):
                payload["image_grid_thw"] = result.image_grid_thw.detach().cpu().tolist()
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return result
