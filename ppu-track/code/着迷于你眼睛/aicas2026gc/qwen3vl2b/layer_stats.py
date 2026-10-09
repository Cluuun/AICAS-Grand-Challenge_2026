from __future__ import annotations

import atexit
import csv
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _parse_cross_pairs(value: str, num_layers: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"LAYER_STATS_CROSS_PAIRS item must be source:target, got {item!r}")
        source_text, target_text = item.split(":", 1)
        source = int(source_text)
        target = int(target_text)
        if not (0 <= source < target < num_layers):
            raise ValueError(f"LAYER_STATS_CROSS_PAIRS pair must satisfy 0 <= source < target < {num_layers}, got {item!r}")
        pair = (source, target)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def maybe_create_layer_stats_collector(num_layers: int, hidden_size: int) -> "LayerStatsCollector | None":
    if not _env_flag("LAYER_STATS", False):
        return None
    return LayerStatsCollector(num_layers=num_layers, hidden_size=hidden_size)


class LayerStatsCollector:
    stage_names = ("prefill", "decode")
    tensor_names = ("h_in", "h_attn", "h_out")

    def __init__(self, num_layers: int, hidden_size: int):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_dir = Path(os.environ.get("LAYER_STATS_OUT", "layer_stats/latest"))
        self.phases = {
            phase.strip()
            for phase in os.environ.get("LAYER_STATS_PHASES", "answer,accuracy").split(",")
            if phase.strip()
        }
        self.max_samples = _env_int("LAYER_STATS_MAX_SAMPLES", 20)
        self.max_decode_steps = _env_int("LAYER_STATS_MAX_DECODE_STEPS", 64)
        self.lmhead_enabled = _env_flag("LAYER_STATS_LMHEAD", True)
        self.lmhead_max_steps = _env_int("LAYER_STATS_LMHEAD_MAX_STEPS", 8)
        self.lmhead_topk = _env_int("LAYER_STATS_LMHEAD_TOPK", 5)
        self.outlier_topk = _env_int("LAYER_STATS_OUTLIER_TOPK", 32)
        self.outlier_scale = _env_float("LAYER_STATS_OUTLIER_SCALE", 6.0)
        self.reservoir_size = _env_int("LAYER_STATS_RESERVOIR", 256)
        self.quantile_sample = _env_int("LAYER_STATS_QUANTILE_SAMPLE", 65536)
        self.save_every = max(1, _env_int("LAYER_STATS_SAVE_EVERY", 1))
        self.cross_pairs = _parse_cross_pairs(os.environ.get("LAYER_STATS_CROSS_PAIRS", ""), num_layers)

        stage_count = len(self.stage_names)
        tensor_count = len(self.tensor_names)
        stat_shape = (stage_count, num_layers, tensor_count, hidden_size)
        scalar_shape = (stage_count, num_layers, tensor_count)
        delta_shape = (stage_count, num_layers)

        self.count = torch.zeros(scalar_shape, dtype=torch.long, device="cpu")
        self.channel_sum = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")
        self.channel_sumsq = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")
        self.channel_abs_sum = torch.zeros(stat_shape, dtype=torch.float64, device="cpu")
        self.channel_max_abs = torch.zeros(stat_shape, dtype=torch.float32, device="cpu")
        self.outlier_topk_counts = torch.zeros(stat_shape, dtype=torch.int32, device="cpu")
        self.outlier_threshold_counts = torch.zeros(stat_shape, dtype=torch.int64, device="cpu")

        self.scalar_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(scalar_shape, dtype=torch.float64, device="cpu")
            for name in ("mean", "std", "rms", "norm", "max_abs", "abs_p50", "abs_p90", "abs_p95", "abs_p99")
        }
        self.delta_sums: dict[str, torch.Tensor] = {
            name: torch.zeros(delta_shape, dtype=torch.float64, device="cpu")
            for name in (
                "attn_delta_norm",
                "mlp_delta_norm",
                "attn_delta_ratio",
                "mlp_delta_ratio",
                "attn_mlp_delta_ratio",
                "cos_in_attn",
                "cos_attn_out",
                "cos_in_out",
                "bi",
            )
        }
        self.delta_count = torch.zeros(delta_shape, dtype=torch.long, device="cpu")

        self.kv_gap_count = torch.zeros(num_layers, dtype=torch.long, device="cpu")
        self.k_gap_sum = torch.zeros(num_layers, dtype=torch.float64, device="cpu")
        self.v_gap_sum = torch.zeros(num_layers, dtype=torch.float64, device="cpu")

        self.layer_cos_sum = torch.zeros((num_layers, num_layers), dtype=torch.float64, device="cpu")
        self.layer_cos_count = torch.zeros((num_layers, num_layers), dtype=torch.long, device="cpu")
        self.cross_sumxy = torch.zeros((stage_count, len(self.cross_pairs), hidden_size), dtype=torch.float64, device="cpu")
        self.cross_count = torch.zeros((stage_count, len(self.cross_pairs)), dtype=torch.long, device="cpu")

        self.lmhead_rows: list[dict[str, Any]] = []
        self.token_rows: list[dict[str, Any]] = []
        self.reservoir: list[dict[str, Any]] = []

        self.active = False
        self.sample_idx = -1
        self.samples_started = 0
        self.samples_finished = 0
        self.prompt_len = 0
        self.phase = ""
        self.decode_step = -1
        self.collect_current_step = False
        self._current_layer_outputs: dict[int, torch.Tensor] = {}
        self._saved = False

        atexit.register(self.save)
        print(
            "[LayerStats] enabled "
            f"out={self.out_dir} phases={sorted(self.phases)} "
            f"max_samples={self.max_samples} max_decode_steps={self.max_decode_steps} "
            f"lmhead={self.lmhead_enabled} lmhead_max_steps={self.lmhead_max_steps} "
            f"cross_pairs={self.cross_pairs}",
            flush=True,
        )

    def phase_enabled(self) -> bool:
        phase = os.environ.get("AICAS_BENCHMARK_PHASE", "")
        return "all" in self.phases or phase in self.phases

    def maybe_begin_sample(self, prompt_len: int) -> None:
        if self.active:
            self.end_sample()
        self.active = False
        if not self.phase_enabled():
            return
        if self.samples_started >= self.max_samples:
            return

        self.sample_idx = self.samples_started
        self.samples_started += 1
        self.prompt_len = int(prompt_len)
        self.phase = os.environ.get("AICAS_BENCHMARK_PHASE", "")
        self.decode_step = -1
        self.collect_current_step = False
        self._current_layer_outputs = {}
        self.active = True
        self._saved = False

    def end_sample(self, generated_tokens: int | None = None) -> None:
        if not self.active:
            return
        if generated_tokens is not None:
            self.token_rows.append(
                {
                    "sample": self.sample_idx,
                    "phase": self.phase,
                    "step": "summary",
                    "prompt_len": self.prompt_len,
                    "generated_tokens": int(generated_tokens),
                }
            )
        self.active = False
        self.samples_finished += 1
        if self.samples_finished % self.save_every == 0:
            self.save()

    def begin_decode_step(self, cache_len: int) -> None:
        if not self.active:
            self.collect_current_step = False
            return
        self.decode_step += 1
        self.collect_current_step = self.decode_step < self.max_decode_steps
        self._current_layer_outputs = {}

    def end_decode_step(self) -> None:
        if not (self.active and self.collect_current_step):
            return
        if len(self._current_layer_outputs) < 2:
            return
        layer_ids = sorted(self._current_layer_outputs)
        stacked = torch.stack([self._current_layer_outputs[layer_idx].float() for layer_idx in layer_ids])
        norms = stacked.norm(dim=-1).clamp_min(1e-12)
        cos = (stacked @ stacked.T) / (norms[:, None] * norms[None, :])
        for i, layer_i in enumerate(layer_ids):
            for j, layer_j in enumerate(layer_ids):
                self.layer_cos_sum[layer_i, layer_j] += float(cos[i, j].item())
                self.layer_cos_count[layer_i, layer_j] += 1
        if self.cross_pairs:
            for pair_idx, (source_layer, target_layer) in enumerate(self.cross_pairs):
                source = self._current_layer_outputs.get(source_layer)
                target = self._current_layer_outputs.get(target_layer)
                if source is None or target is None:
                    continue
                self.cross_sumxy[1, pair_idx] += (source.float() * target.float()).cpu().double()
                self.cross_count[1, pair_idx] += 1

    def should_collect_lmhead(self) -> bool:
        return (
            self.active
            and self.collect_current_step
            and self.lmhead_enabled
            and self.decode_step < self.lmhead_max_steps
        )

    def _stage_idx(self, is_decoding: bool) -> int:
        return 1 if is_decoding else 0

    def _flatten_hidden(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().reshape(-1, self.hidden_size)

    def _record_tensor_stats(self, stage_idx: int, layer_idx: int, tensor_idx: int, tensor: torch.Tensor) -> None:
        mat = self._flatten_hidden(tensor)
        if mat.numel() == 0:
            return

        n = mat.shape[0]
        abs_mat = mat.abs()
        idx = (stage_idx, layer_idx, tensor_idx)
        self.count[idx] += n
        self.channel_sum[idx] += mat.sum(dim=0).cpu().double()
        self.channel_sumsq[idx] += (mat * mat).sum(dim=0).cpu().double()
        self.channel_abs_sum[idx] += abs_mat.sum(dim=0).cpu().double()
        self.channel_max_abs[idx] = torch.maximum(self.channel_max_abs[idx], abs_mat.max(dim=0).values.cpu())

        self.scalar_sums["mean"][idx] += float(mat.mean().item())
        self.scalar_sums["std"][idx] += float(mat.std(unbiased=False).item())
        self.scalar_sums["rms"][idx] += float(torch.sqrt((mat * mat).mean()).item())
        self.scalar_sums["norm"][idx] += float(mat.norm(dim=-1).mean().item())
        self.scalar_sums["max_abs"][idx] += float(abs_mat.max().item())

        flat_abs = abs_mat.flatten()
        if self.quantile_sample > 0 and flat_abs.numel() > self.quantile_sample:
            step = max(1, flat_abs.numel() // self.quantile_sample)
            sample_abs = flat_abs[::step][: self.quantile_sample]
        else:
            sample_abs = flat_abs
        quantiles = torch.quantile(sample_abs, torch.tensor([0.5, 0.9, 0.95, 0.99], device=sample_abs.device))
        for name, value in zip(("abs_p50", "abs_p90", "abs_p95", "abs_p99"), quantiles):
            self.scalar_sums[name][idx] += float(value.item())

        channel_score = abs_mat.max(dim=0).values
        topk = min(self.outlier_topk, channel_score.numel())
        top_idx = torch.topk(channel_score, k=topk).indices.cpu()
        self.outlier_topk_counts[idx][top_idx] += 1

        median_abs = torch.median(sample_abs)
        threshold = median_abs * self.outlier_scale
        channel_hits = (abs_mat > threshold).sum(dim=0).cpu().to(torch.int64)
        self.outlier_threshold_counts[idx] += channel_hits

    def _mean_cos(self, lhs: torch.Tensor, rhs: torch.Tensor) -> float:
        return float(F.cosine_similarity(lhs.float(), rhs.float(), dim=-1, eps=1e-12).mean().item())

    def record_layer(
        self,
        *,
        layer_idx: int,
        is_decoding: bool,
        h_in: torch.Tensor,
        h_attn: torch.Tensor,
        h_out: torch.Tensor,
    ) -> None:
        if not self.active:
            return
        if is_decoding and not self.collect_current_step:
            return

        stage_idx = self._stage_idx(is_decoding)
        self._record_tensor_stats(stage_idx, layer_idx, 0, h_in)
        self._record_tensor_stats(stage_idx, layer_idx, 1, h_attn)
        self._record_tensor_stats(stage_idx, layer_idx, 2, h_out)

        h_in_m = self._flatten_hidden(h_in)
        h_attn_m = self._flatten_hidden(h_attn)
        h_out_m = self._flatten_hidden(h_out)
        attn_delta = h_attn_m - h_in_m
        mlp_delta = h_out_m - h_attn_m

        h_in_norm = h_in_m.norm(dim=-1).clamp_min(1e-12)
        h_attn_norm = h_attn_m.norm(dim=-1).clamp_min(1e-12)
        attn_delta_norm = attn_delta.norm(dim=-1)
        mlp_delta_norm = mlp_delta.norm(dim=-1)

        idx = (stage_idx, layer_idx)
        self.delta_count[idx] += 1
        self.delta_sums["attn_delta_norm"][idx] += float(attn_delta_norm.mean().item())
        self.delta_sums["mlp_delta_norm"][idx] += float(mlp_delta_norm.mean().item())
        self.delta_sums["attn_delta_ratio"][idx] += float((attn_delta_norm / h_in_norm).mean().item())
        self.delta_sums["mlp_delta_ratio"][idx] += float((mlp_delta_norm / h_attn_norm).mean().item())
        self.delta_sums["attn_mlp_delta_ratio"][idx] += float((attn_delta_norm / mlp_delta_norm.clamp_min(1e-12)).mean().item())
        self.delta_sums["cos_in_attn"][idx] += self._mean_cos(h_in_m, h_attn_m)
        self.delta_sums["cos_attn_out"][idx] += self._mean_cos(h_attn_m, h_out_m)
        cos_in_out = self._mean_cos(h_in_m, h_out_m)
        self.delta_sums["cos_in_out"][idx] += cos_in_out
        self.delta_sums["bi"][idx] += 1.0 - cos_in_out

        if is_decoding:
            self._current_layer_outputs[layer_idx] = h_out_m[-1].detach().cpu()

        if self.reservoir_size > 0 and len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(
                {
                    "sample": self.sample_idx,
                    "phase": self.phase,
                    "stage": self.stage_names[stage_idx],
                    "step": self.decode_step if is_decoding else -1,
                    "layer": layer_idx,
                    "h_out": h_out_m[-1].detach().cpu().to(torch.float16),
                }
            )

    def record_kv_gap(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor, cache_seqlens: torch.Tensor) -> None:
        if not (self.active and self.collect_current_step):
            return
        pos = int(cache_seqlens.item()) - 1
        if pos <= 0:
            return
        cur_k = k_cache[0, pos].detach().float().flatten()
        prev_k = k_cache[0, pos - 1].detach().float().flatten()
        cur_v = v_cache[0, pos].detach().float().flatten()
        prev_v = v_cache[0, pos - 1].detach().float().flatten()
        self.k_gap_sum[layer_idx] += float(F.cosine_similarity(cur_k[None], prev_k[None], dim=-1, eps=1e-12).item())
        self.v_gap_sum[layer_idx] += float(F.cosine_similarity(cur_v[None], prev_v[None], dim=-1, eps=1e-12).item())
        self.kv_gap_count[layer_idx] += 1

    def record_generated_token(
        self,
        next_tokens: torch.Tensor,
        all_layers_hidden_states: tuple[torch.Tensor, ...] | None,
        final_hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        token_id_map: torch.Tensor | None,
    ) -> None:
        if not self.active:
            return

        token_id = int(next_tokens.flatten()[0].detach().item())
        self.token_rows.append(
            {
                "sample": self.sample_idx,
                "phase": self.phase,
                "step": self.decode_step,
                "prompt_len": self.prompt_len,
                "token_id": token_id,
            }
        )

        if not self.should_collect_lmhead():
            return
        if not all_layers_hidden_states:
            return

        hidden_stack = torch.cat([hidden[:, -1:, :].reshape(1, self.hidden_size) for hidden in all_layers_hidden_states], dim=0)
        logits = F.linear(hidden_stack.to(dtype=lm_head_weight.dtype), lm_head_weight).float()
        probs = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(logits, k=min(self.lmhead_topk, logits.shape[-1]), dim=-1)
        top_probs = torch.gather(probs, dim=-1, index=top_indices)
        entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1)

        if token_id_map is not None and token_id_map.numel() > 0:
            top_tokens = token_id_map[top_indices].detach().cpu()
        else:
            top_tokens = top_indices.detach().cpu()

        final_top_set = set(int(x) for x in top_tokens[-1].tolist())
        for layer_idx in range(hidden_stack.shape[0]):
            layer_top_tokens = [int(x) for x in top_tokens[layer_idx].tolist()]
            margin = 0.0
            if top_values.shape[-1] >= 2:
                margin = float((top_values[layer_idx, 0] - top_values[layer_idx, 1]).item())
            self.lmhead_rows.append(
                {
                    "sample": self.sample_idx,
                    "phase": self.phase,
                    "step": self.decode_step,
                    "layer": layer_idx,
                    "top1_token": layer_top_tokens[0],
                    "final_top1_token": token_id,
                    "top1_match": int(layer_top_tokens[0] == token_id),
                    "top1_prob": float(top_probs[layer_idx, 0].item()),
                    "entropy": float(entropy[layer_idx].item()),
                    "margin": margin,
                    "topk_overlap": len(set(layer_top_tokens) & final_top_set),
                }
            )

    def _safe_div(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1), torch.zeros_like(numerator, dtype=torch.float64))

    def _mean_delta(self, name: str) -> torch.Tensor:
        return self._safe_div(self.delta_sums[name], self.delta_count.to(torch.float64))

    def _write_csv(self, path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _build_layer_bi_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for stage_idx, stage in enumerate(self.stage_names):
            for layer_idx in range(self.num_layers):
                count = int(self.delta_count[stage_idx, layer_idx].item())
                row = {"stage": stage, "layer": layer_idx, "count": count}
                for name in self.delta_sums:
                    value = 0.0 if count == 0 else float((self.delta_sums[name][stage_idx, layer_idx] / count).item())
                    row[name] = value
                rows.append(row)
        return rows

    def _build_kv_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer_idx in range(self.num_layers):
            count = int(self.kv_gap_count[layer_idx].item())
            rows.append(
                {
                    "layer": layer_idx,
                    "count": count,
                    "k_cos": 0.0 if count == 0 else float((self.k_gap_sum[layer_idx] / count).item()),
                    "v_cos": 0.0 if count == 0 else float((self.v_gap_sum[layer_idx] / count).item()),
                }
            )
        return rows

    def _build_layer_cos_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i in range(self.num_layers):
            for j in range(self.num_layers):
                count = int(self.layer_cos_count[i, j].item())
                rows.append(
                    {
                        "layer_i": i,
                        "layer_j": j,
                        "count": count,
                        "cos": 0.0 if count == 0 else float((self.layer_cos_sum[i, j] / count).item()),
                    }
                )
        return rows

    def _build_outlier_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        topk = min(self.outlier_topk, self.hidden_size)
        for stage_idx, stage in enumerate(self.stage_names):
            payload[stage] = {}
            for tensor_idx, tensor_name in enumerate(self.tensor_names):
                payload[stage][tensor_name] = []
                for layer_idx in range(self.num_layers):
                    count_scores = self.outlier_topk_counts[stage_idx, layer_idx, tensor_idx]
                    max_scores = self.channel_max_abs[stage_idx, layer_idx, tensor_idx]
                    count_values, count_indices = torch.topk(count_scores, k=topk)
                    max_values, max_indices = torch.topk(max_scores, k=topk)
                    payload[stage][tensor_name].append(
                        {
                            "layer": layer_idx,
                            "topk_by_hit_count": [
                                {"channel": int(ch), "count": int(cnt)}
                                for ch, cnt in zip(count_indices.tolist(), count_values.tolist())
                                if int(cnt) > 0
                            ],
                            "topk_by_max_abs": [
                                {"channel": int(ch), "max_abs": float(value)}
                                for ch, value in zip(max_indices.tolist(), max_values.tolist())
                                if float(value) > 0.0
                            ],
                        }
                    )
        return payload

    def _build_summary(self) -> dict[str, Any]:
        decode_bi = self._mean_delta("bi")[1]
        decode_cos = self._mean_delta("cos_in_out")[1]
        decode_attn_mlp = self._mean_delta("attn_mlp_delta_ratio")[1]
        kv_rows = self._build_kv_rows()

        low_bi = torch.argsort(decode_bi)[: min(8, self.num_layers)].tolist()
        high_cos = torch.argsort(decode_cos, descending=True)[: min(8, self.num_layers)].tolist()
        high_kv = sorted(kv_rows, key=lambda row: min(row["k_cos"], row["v_cos"]), reverse=True)[:8]

        return {
            "config": {
                "num_layers": self.num_layers,
                "hidden_size": self.hidden_size,
                "phases": sorted(self.phases),
                "max_samples": self.max_samples,
                "max_decode_steps": self.max_decode_steps,
                "lmhead_enabled": self.lmhead_enabled,
                "lmhead_max_steps": self.lmhead_max_steps,
            },
            "samples_started": self.samples_started,
            "samples_finished": self.samples_finished,
            "decode_low_bi_layers": [
                {"layer": int(layer), "bi": float(decode_bi[layer].item()), "cos_in_out": float(decode_cos[layer].item())}
                for layer in low_bi
            ],
            "decode_high_cos_layers": [
                {"layer": int(layer), "bi": float(decode_bi[layer].item()), "cos_in_out": float(decode_cos[layer].item())}
                for layer in high_cos
            ],
            "decode_attn_mlp_delta_ratio": [
                {"layer": layer, "ratio": float(decode_attn_mlp[layer].item())}
                for layer in range(self.num_layers)
            ],
            "high_kv_gap_layers": high_kv,
            "lmhead_rows": len(self.lmhead_rows),
        }

    def _write_skip_candidates(self, path: Path, summary: dict[str, Any]) -> None:
        lines = [
            "# Skip Candidates",
            "",
            "This file is generated from calibration statistics. Treat it as a ranking, not a decision.",
            "",
            "## Low Decode BI",
        ]
        for row in summary["decode_low_bi_layers"]:
            lines.append(f"- layer {row['layer']}: BI={row['bi']:.6f}, cos={row['cos_in_out']:.6f}")
        lines += ["", "## High KV Adjacent Similarity"]
        for row in summary["high_kv_gap_layers"]:
            lines.append(f"- layer {row['layer']}: K={row['k_cos']:.6f}, V={row['v_cos']:.6f}, count={row['count']}")
        if self.lmhead_rows:
            per_layer: dict[int, list[dict[str, Any]]] = {}
            for row in self.lmhead_rows:
                per_layer.setdefault(int(row["layer"]), []).append(row)
            lines += ["", "## LM Head Agreement"]
            for layer_idx in sorted(per_layer):
                rows = per_layer[layer_idx]
                match = sum(int(row["top1_match"]) for row in rows) / max(1, len(rows))
                prob = sum(float(row["top1_prob"]) for row in rows) / max(1, len(rows))
                entropy = sum(float(row["entropy"]) for row in rows) / max(1, len(rows))
                lines.append(f"- layer {layer_idx}: match={match:.4f}, prob={prob:.4f}, entropy={entropy:.4f}, count={len(rows)}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save(self) -> None:
        if self._saved and not self.active:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)

        summary = self._build_summary()
        with (self.out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        layer_bi_rows = self._build_layer_bi_rows()
        self._write_csv(
            self.out_dir / "layer_bi.csv",
            layer_bi_rows,
            ["stage", "layer", "count", *self.delta_sums.keys()],
        )
        self._write_csv(self.out_dir / "kv_gap.csv", self._build_kv_rows(), ["layer", "count", "k_cos", "v_cos"])
        self._write_csv(self.out_dir / "layer_cosine.csv", self._build_layer_cos_rows(), ["layer_i", "layer_j", "count", "cos"])

        if self.lmhead_rows:
            self._write_csv(
                self.out_dir / "lmhead_agreement.csv",
                self.lmhead_rows,
                [
                    "sample",
                    "phase",
                    "step",
                    "layer",
                    "top1_token",
                    "final_top1_token",
                    "top1_match",
                    "top1_prob",
                    "entropy",
                    "margin",
                    "topk_overlap",
                ],
            )
        if self.token_rows:
            fieldnames = sorted({key for row in self.token_rows for key in row})
            self._write_csv(self.out_dir / "tokens.csv", self.token_rows, fieldnames)

        with (self.out_dir / "outlier_channels.json").open("w", encoding="utf-8") as f:
            json.dump(self._build_outlier_payload(), f, indent=2)

        torch.save(
            {
                "count": self.count,
                "channel_sum": self.channel_sum,
                "channel_sumsq": self.channel_sumsq,
                "channel_abs_sum": self.channel_abs_sum,
                "channel_max_abs": self.channel_max_abs,
                "outlier_topk_counts": self.outlier_topk_counts,
                "outlier_threshold_counts": self.outlier_threshold_counts,
                "delta_count": self.delta_count,
                "delta_sums": self.delta_sums,
                "kv_gap_count": self.kv_gap_count,
                "k_gap_sum": self.k_gap_sum,
                "v_gap_sum": self.v_gap_sum,
                "layer_cos_sum": self.layer_cos_sum,
                "layer_cos_count": self.layer_cos_count,
                "cross_pairs": self.cross_pairs,
                "cross_sumxy": self.cross_sumxy,
                "cross_count": self.cross_count,
                "reservoir": self.reservoir,
            },
            self.out_dir / "layer_channel_stats.pt",
        )
        self._write_skip_candidates(self.out_dir / "skip_candidates.md", summary)
        self._saved = not self.active
