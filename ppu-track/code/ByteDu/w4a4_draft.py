from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from triton_decode_packed_qkv_cache import decode_packed_qkv_cache_inplace
from triton_w4a4_gemv import TRITON_AVAILABLE, pack_q4_values, pack_weight_q4_rowwise, w4a4_gemv_inference


TargetStepFn = Callable[[torch.Tensor, int], torch.Tensor]
TargetBatchFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
RollbackFn = Callable[[int, int], None]
DraftFn = Callable[[torch.Tensor, int, int], torch.Tensor | None]


@dataclass
class W4A4LinearWeight:
    packed: torch.Tensor
    scale: torch.Tensor
    in_features: int
    out_features: int
    name: str


@dataclass
class W4A4DraftLayer:
    qkv: W4A4LinearWeight
    o_proj: W4A4LinearWeight
    gate_up: W4A4LinearWeight
    down_proj: W4A4LinearWeight


@dataclass
class W4A4SpecStats:
    enabled: bool = False
    disabled_reason: str | None = None
    k: int = 4
    min_new_tokens: int = 64
    quantization_algorithm: str = "groupwise_symmetric_w4_dynamic_per_token_groupwise_a4"
    total_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_tokens: int = 0
    rollback_count: int = 0
    speculative_rounds: int = 0
    full_accept_rounds: int = 0
    generated_tokens: int = 0
    draft_decode_cost_ms: float = 0.0
    target_verification_cost_ms: float = 0.0
    target_verification_steps: int = 0
    avoided_target_verification_steps: int = 0
    end_to_end_cost_ms: float = 0.0
    end_to_end_throughput: float = 0.0
    fallback_count: int = 0
    groups: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        acceptance = self.accepted_draft_tokens / max(self.total_draft_tokens, 1)
        avg_accepted = self.accepted_draft_tokens / max(self.speculative_rounds, 1)
        rollback_rate = self.rollback_count / max(self.speculative_rounds, 1)
        return {
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "k": self.k,
            "min_new_tokens": self.min_new_tokens,
            "quantization_algorithm": self.quantization_algorithm,
            "total_draft_tokens": self.total_draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "rejected_tokens": self.rejected_tokens,
            "acceptance_rate": acceptance,
            "avg_accepted_tokens_per_round": avg_accepted,
            "rollback_rate": rollback_rate,
            "rollback_count": self.rollback_count,
            "speculative_rounds": self.speculative_rounds,
            "full_accept_rounds": self.full_accept_rounds,
            "target_verification_cost_ms": self.target_verification_cost_ms,
            "target_verification_steps": self.target_verification_steps,
            "avoided_target_verification_steps": self.avoided_target_verification_steps,
            "avg_target_verification_steps_per_round": self.target_verification_steps
            / max(self.speculative_rounds, 1),
            "draft_decode_cost_ms": self.draft_decode_cost_ms,
            "end_to_end_cost_ms": self.end_to_end_cost_ms,
            "end_to_end_throughput": self.end_to_end_throughput,
            "generated_tokens": self.generated_tokens,
            "fallback_count": self.fallback_count,
            "groups": self.groups,
        }


class W4A4SpeculativeRuntime:
    def __init__(
        self,
        *,
        k: int,
        min_new_tokens: int,
        stats_path: str | Path,
        jsonl_path: str | Path,
        enabled: bool,
        disabled_reason: str | None = None,
    ) -> None:
        if k not in (2, 4, 8):
            raise ValueError(f"W4A4_DRAFT_K must be one of 2, 4, 8; got {k}.")
        self.k = int(k)
        self.min_new_tokens = int(min_new_tokens)
        self.stats_path = Path(stats_path)
        self.jsonl_path = Path(jsonl_path)
        self.stats = W4A4SpecStats(
            enabled=enabled,
            disabled_reason=disabled_reason,
            k=self.k,
            min_new_tokens=self.min_new_tokens,
        )

    def can_generate(self, max_new_tokens: int) -> bool:
        return self.stats.enabled and max_new_tokens >= self.min_new_tokens

    def mark_fallback(self, reason: str) -> None:
        self.stats.fallback_count += 1
        self.stats.disabled_reason = reason
        self.write_summary()

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0

    @staticmethod
    def _contains_eos(token: torch.Tensor, eos_token_ids: set[int]) -> bool:
        return bool(eos_token_ids) and int(token.view(-1)[0].item()) in eos_token_ids

    @staticmethod
    def _prompt_bucket(prompt_len: int) -> str:
        if prompt_len < 256:
            return "short_prompt"
        if prompt_len < 768:
            return "medium_prompt"
        return "long_prompt"

    def _record_group(
        self,
        *,
        prompt_len: int,
        max_new_tokens: int,
        generated_tokens: int,
        elapsed_ms: float,
        accepted_draft_tokens: int,
        total_draft_tokens: int,
        rejected_tokens: int,
    ) -> str:
        key = f"max_new={max_new_tokens}|k={self.k}|{self._prompt_bucket(prompt_len)}"
        entry = self.stats.groups.setdefault(
            key,
            {
                "runs": 0,
                "generated_tokens": 0,
                "elapsed_ms": 0.0,
                "accepted_draft_tokens": 0,
                "total_draft_tokens": 0,
                "rejected_tokens": 0,
            },
        )
        entry["runs"] += 1
        entry["generated_tokens"] += generated_tokens
        entry["elapsed_ms"] += elapsed_ms
        entry["accepted_draft_tokens"] += accepted_draft_tokens
        entry["total_draft_tokens"] += total_draft_tokens
        entry["rejected_tokens"] += rejected_tokens
        entry["throughput"] = entry["generated_tokens"] / max(entry["elapsed_ms"] / 1000.0, 1e-9)
        entry["acceptance_rate"] = entry["accepted_draft_tokens"] / max(entry["total_draft_tokens"], 1)
        return key

    def write_summary(self) -> None:
        payload = self.stats.to_dict()
        tmp = self.stats_path.with_suffix(self.stats_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.stats_path)

    def append_record(self, record: dict) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def generate(
        self,
        *,
        original_input_ids: torch.Tensor,
        first_token: torch.Tensor,
        prompt_len: int,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        draft_fn: DraftFn,
        target_step_fn: TargetStepFn,
        rollback_fn: RollbackFn,
        target_batch_fn: TargetBatchFn | None = None,
        use_batch_verifier: bool = False,
    ) -> torch.Tensor:
        start_e2e = time.perf_counter()
        start_total_draft = self.stats.total_draft_tokens
        start_accepted = self.stats.accepted_draft_tokens
        start_rejected = self.stats.rejected_tokens
        generated = torch.empty((1, max_new_tokens), device=original_input_ids.device, dtype=torch.long)
        generated[:, 0] = first_token.view(1)
        num_generated = 1
        prefix_len = int(prompt_len)
        current_input_ids = first_token.view(1, 1)
        eos_set = {int(token_id) for token_id in eos_token_ids}

        if max_new_tokens <= 1 or self._contains_eos(first_token, eos_set):
            elapsed = self._elapsed_ms(start_e2e)
            self.stats.generated_tokens += num_generated
            self.stats.end_to_end_cost_ms += elapsed
            self.stats.end_to_end_throughput = self.stats.generated_tokens / max(self.stats.end_to_end_cost_ms / 1000.0, 1e-9)
            self.write_summary()
            return torch.cat([original_input_ids, generated[:, :num_generated]], dim=1)

        while num_generated < max_new_tokens:
            round_prefix_len = prefix_len
            remaining = max_new_tokens - num_generated
            candidate_budget = min(self.k, max(remaining - 1, 0))
            candidates = torch.empty((0,), device=original_input_ids.device, dtype=torch.long)
            if candidate_budget > 0:
                draft_start = time.perf_counter()
                drafted = draft_fn(current_input_ids, prefix_len, candidate_budget)
                self.stats.draft_decode_cost_ms += self._elapsed_ms(draft_start)
                if drafted is not None and drafted.numel() > 0:
                    candidates = drafted.view(-1).to(device=original_input_ids.device, dtype=torch.long)
                    candidates = candidates[:candidate_budget]
                    candidate_budget = int(candidates.numel())
                else:
                    candidate_budget = 0

            accepted_this_round = 0
            rejected = False
            self.stats.speculative_rounds += 1
            self.stats.total_draft_tokens += candidate_budget
            possible_target_steps = candidate_budget + 1

            if use_batch_verifier and target_batch_fn is not None:
                verify_start = time.perf_counter()
                target_predictions = target_batch_fn(current_input_ids, candidates, prefix_len).view(-1)
                self.stats.target_verification_cost_ms += self._elapsed_ms(verify_start)
                if target_predictions.numel() < possible_target_steps:
                    raise RuntimeError(
                        f"Target verifier returned {target_predictions.numel()} tokens for {candidate_budget} candidates."
                    )
                self.stats.target_verification_steps += possible_target_steps

                for depth in range(candidate_budget):
                    target_token = target_predictions[depth].view(1, 1)
                    draft_token = candidates[depth].view(1, 1)
                    if int(target_token.item()) == int(draft_token.item()):
                        generated[:, num_generated] = draft_token.view(1)
                        num_generated += 1
                        accepted_this_round += 1
                        current_input_ids = draft_token
                        if self._contains_eos(draft_token, eos_set) or num_generated >= max_new_tokens:
                            rejected = True
                            prefix_len = round_prefix_len + accepted_this_round
                            rollback_fn(round_prefix_len, prefix_len)
                            break
                        continue

                    generated[:, num_generated] = target_token.view(1)
                    num_generated += 1
                    current_input_ids = target_token
                    self.stats.rejected_tokens += 1
                    self.stats.rollback_count += 1
                    rejected = True
                    prefix_len = round_prefix_len + accepted_this_round + 1
                    rollback_fn(round_prefix_len, prefix_len)
                    if self._contains_eos(target_token, eos_set):
                        break
                    break

                self.stats.accepted_draft_tokens += accepted_this_round
                if rejected:
                    if num_generated >= max_new_tokens or self._contains_eos(current_input_ids, eos_set):
                        break
                    continue

                self.stats.full_accept_rounds += 1
                bonus_token = target_predictions[candidate_budget].view(1, 1)
                prefix_len = round_prefix_len + candidate_budget + 1
                rollback_fn(round_prefix_len, prefix_len)
                generated[:, num_generated] = bonus_token.view(1)
                num_generated += 1
                current_input_ids = bonus_token
                if self._contains_eos(bonus_token, eos_set):
                    break
                continue

            actual_target_steps = 0
            verify_start = time.perf_counter()

            for depth in range(candidate_budget):
                target_token = target_step_fn(current_input_ids, prefix_len).view(1, 1)
                actual_target_steps += 1
                draft_token = candidates[depth].view(1, 1)
                prefix_len += 1
                if int(target_token.item()) == int(draft_token.item()):
                    generated[:, num_generated] = draft_token.view(1)
                    num_generated += 1
                    accepted_this_round += 1
                    current_input_ids = draft_token
                    if self._contains_eos(draft_token, eos_set) or num_generated >= max_new_tokens:
                        rejected = True
                        rollback_fn(round_prefix_len, prefix_len)
                        break
                    continue

                generated[:, num_generated] = target_token.view(1)
                num_generated += 1
                current_input_ids = target_token
                self.stats.rejected_tokens += 1
                self.stats.rollback_count += 1
                rejected = True
                rollback_fn(round_prefix_len, prefix_len)
                if self._contains_eos(target_token, eos_set):
                    break
                break

            if not rejected:
                bonus_token = target_step_fn(current_input_ids, prefix_len).view(1, 1)
                actual_target_steps += 1
                prefix_len += 1

            self.stats.target_verification_cost_ms += self._elapsed_ms(verify_start)
            self.stats.target_verification_steps += actual_target_steps
            self.stats.avoided_target_verification_steps += max(possible_target_steps - actual_target_steps, 0)
            self.stats.accepted_draft_tokens += accepted_this_round
            if rejected:
                if num_generated >= max_new_tokens or self._contains_eos(current_input_ids, eos_set):
                    break
                continue

            self.stats.full_accept_rounds += 1
            rollback_fn(round_prefix_len, prefix_len)
            generated[:, num_generated] = bonus_token.view(1)
            num_generated += 1
            current_input_ids = bonus_token
            if self._contains_eos(bonus_token, eos_set):
                break

        elapsed = self._elapsed_ms(start_e2e)
        self.stats.generated_tokens += num_generated
        self.stats.end_to_end_cost_ms += elapsed
        self.stats.end_to_end_throughput = self.stats.generated_tokens / max(self.stats.end_to_end_cost_ms / 1000.0, 1e-9)
        group_key = self._record_group(
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            generated_tokens=num_generated,
            elapsed_ms=elapsed,
            accepted_draft_tokens=self.stats.accepted_draft_tokens - start_accepted,
            total_draft_tokens=self.stats.total_draft_tokens - start_total_draft,
            rejected_tokens=self.stats.rejected_tokens - start_rejected,
        )
        record = self.stats.to_dict()
        record.update(
            {
                "prompt_len": prompt_len,
                "prompt_bucket": self._prompt_bucket(prompt_len),
                "max_new_tokens": max_new_tokens,
                "generated_tokens_this_run": num_generated,
                "elapsed_ms_this_run": elapsed,
                "group_key": group_key,
            }
        )
        self.append_record(record)
        self.write_summary()
        print(
            "[W4A4_SPEC] "
            f"k={self.k} prompt_len={prompt_len} max_new={max_new_tokens} "
            f"acceptance={record['acceptance_rate']:.4f} "
            f"avg_accept={record['avg_accepted_tokens_per_round']:.3f} "
            f"rollback={record['rollback_rate']:.4f} "
            f"draft_ms={self.stats.draft_decode_cost_ms:.3f} "
            f"verify_ms={self.stats.target_verification_cost_ms:.3f} "
            f"throughput={record['end_to_end_throughput']:.3f}"
        )
        return torch.cat([original_input_ids, generated[:, :num_generated]], dim=1)


class W4A4DraftModel:
    def __init__(
        self,
        target_inner,
        *,
        device: torch.device,
        dtype: torch.dtype,
        activation_clip_ratio: float = 1.0,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is required for W4A4 draft GEMV.")
        self.target_inner = target_inner
        self.text_model = target_inner.model.language_model
        self.config = target_inner.config.text_config
        self.device = device
        self.dtype = dtype
        self.activation_clip_ratio = float(activation_clip_ratio)
        self.layers: list[W4A4DraftLayer] = []
        self.lm_head: W4A4LinearWeight | None = None
        self._activation_buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._max_cache_len = 0
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.cache_seqlens = torch.zeros((1,), device=device, dtype=torch.int32)
        self.cache_seqlens_next = torch.ones((1,), device=device, dtype=torch.int32)
        self._decode_graph: torch.cuda.CUDAGraph | None = None
        self._decode_graph_cache_len = 0
        self._graph_input_ids = torch.zeros((1, 1), device=device, dtype=torch.long)
        self._graph_position_cos = torch.empty((1, 1, self.config.head_dim), device=device, dtype=dtype)
        self._graph_position_sin = torch.empty_like(self._graph_position_cos)
        self._graph_next_token = torch.empty((1, 1), device=device, dtype=torch.long)
        self._query_out = torch.empty(
            (1, self.config.num_attention_heads, self.config.head_dim),
            device=device,
            dtype=dtype,
        )
        self._qkv_out = torch.empty(
            ((self.config.num_attention_heads + 2 * self.config.num_key_value_heads) * self.config.head_dim,),
            device=device,
            dtype=dtype,
        )
        self._attn_out = torch.empty((self.config.hidden_size,), device=device, dtype=dtype)
        self._gate_up_out = torch.empty((2 * self.config.intermediate_size,), device=device, dtype=dtype)
        self._down_out = torch.empty((self.config.hidden_size,), device=device, dtype=dtype)
        self._lm_logits = torch.empty((self.config.vocab_size,), device=device, dtype=dtype)
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self._load_checkpoint(checkpoint_path)
        else:
            self._build_from_target()

    @staticmethod
    def calibration_clip_ratio(calibration_path: str | Path | None) -> float:
        if calibration_path is None:
            return 1.0
        path = Path(calibration_path)
        if not path.exists():
            return 1.0
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload.get("activation_clip_ratio", 1.0))

    def _quant_linear(self, name: str, weight: torch.Tensor) -> W4A4LinearWeight:
        packed, scale = pack_weight_q4_rowwise(weight.to(device=self.device, dtype=self.dtype))
        return W4A4LinearWeight(
            packed=packed.to(device=self.device),
            scale=scale.to(device=self.device, dtype=self.dtype),
            in_features=int(weight.shape[1]),
            out_features=int(weight.shape[0]),
            name=name,
        )

    def _build_from_target(self) -> None:
        self.layers.clear()
        for layer_idx, layer in enumerate(self.text_model.layers):
            sa = layer.self_attn
            mlp = layer.mlp
            qkv = torch.cat((sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight), dim=0).contiguous()
            gate_up = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0).contiguous()
            self.layers.append(
                W4A4DraftLayer(
                    qkv=self._quant_linear(f"layers.{layer_idx}.qkv", qkv),
                    o_proj=self._quant_linear(f"layers.{layer_idx}.o_proj", sa.o_proj.weight),
                    gate_up=self._quant_linear(f"layers.{layer_idx}.gate_up", gate_up),
                    down_proj=self._quant_linear(f"layers.{layer_idx}.down_proj", mlp.down_proj.weight),
                )
            )
        self.lm_head = self._quant_linear("lm_head", self.target_inner.lm_head.weight)

    def state_dict_payload(self) -> dict:
        tensors = {}
        for idx, layer in enumerate(self.layers):
            for field_name in ("qkv", "o_proj", "gate_up", "down_proj"):
                item = getattr(layer, field_name)
                prefix = f"layers.{idx}.{field_name}"
                tensors[f"{prefix}.packed"] = item.packed.detach().cpu()
                tensors[f"{prefix}.scale"] = item.scale.detach().cpu()
                tensors[f"{prefix}.shape"] = torch.tensor([item.out_features, item.in_features], dtype=torch.int64)
        tensors["lm_head.packed"] = self.lm_head.packed.detach().cpu()
        tensors["lm_head.scale"] = self.lm_head.scale.detach().cpu()
        tensors["lm_head.shape"] = torch.tensor([self.lm_head.out_features, self.lm_head.in_features], dtype=torch.int64)
        return {
            "metadata": {
                "format": "junkrat_w4a4_draft_v1",
                "quantization_algorithm": "groupwise symmetric W4 weights + dynamic per-token groupwise symmetric A4 activations",
                "activation_clip_ratio": self.activation_clip_ratio,
                "num_hidden_layers": len(self.layers),
            },
            "tensors": tensors,
        }

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(self.state_dict_payload(), Path(path))

    def _load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        metadata = payload.get("metadata", {})
        if metadata.get("format") != "junkrat_w4a4_draft_v1":
            raise ValueError(f"Unsupported W4A4 draft checkpoint format: {metadata.get('format')!r}")
        tensors = payload["tensors"]

        def load_linear(prefix: str) -> W4A4LinearWeight:
            shape = tensors[f"{prefix}.shape"]
            packed = tensors[f"{prefix}.packed"]
            in_features = int(shape[1].item())
            if packed.dtype != torch.uint8 and packed.shape[1] == in_features:
                packed = pack_q4_values(packed.to(torch.int8))
            return W4A4LinearWeight(
                packed=packed.to(device=self.device, dtype=torch.uint8).contiguous(),
                scale=tensors[f"{prefix}.scale"].to(device=self.device, dtype=self.dtype).contiguous(),
                out_features=int(shape[0].item()),
                in_features=in_features,
                name=prefix,
            )

        self.layers.clear()
        for idx in range(int(metadata["num_hidden_layers"])):
            self.layers.append(
                W4A4DraftLayer(
                    qkv=load_linear(f"layers.{idx}.qkv"),
                    o_proj=load_linear(f"layers.{idx}.o_proj"),
                    gate_up=load_linear(f"layers.{idx}.gate_up"),
                    down_proj=load_linear(f"layers.{idx}.down_proj"),
                )
            )
        self.lm_head = load_linear("lm_head")
        self.activation_clip_ratio = float(metadata.get("activation_clip_ratio", self.activation_clip_ratio))

    def ensure_cache(self, target_len: int) -> None:
        if self.k_cache is not None and target_len <= self._max_cache_len:
            return
        cfg = self.config
        cache_shape = (cfg.num_hidden_layers, 1, target_len, cfg.num_key_value_heads, cfg.head_dim)
        self.k_cache = torch.empty(cache_shape, device=self.device, dtype=self.dtype)
        self.v_cache = torch.empty_like(self.k_cache)
        self._max_cache_len = int(target_len)
        self._decode_graph = None
        self._decode_graph_cache_len = 0

    def start_from_target_cache(self, target_k_cache: torch.Tensor, target_v_cache: torch.Tensor, prefix_len: int) -> None:
        self.ensure_cache(target_k_cache.shape[2])
        self.k_cache[:, :, :prefix_len].copy_(target_k_cache[:, :, :prefix_len])
        self.v_cache[:, :, :prefix_len].copy_(target_v_cache[:, :, :prefix_len])
        self.cache_seqlens.fill_(prefix_len)
        self.cache_seqlens_next.fill_(prefix_len + 1)

    def sync_target_cache_slice(
        self,
        target_k_cache: torch.Tensor,
        target_v_cache: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        self.ensure_cache(target_k_cache.shape[2])
        start = max(int(start), 0)
        end = max(int(end), start)
        if end > start:
            self.k_cache[:, :, start:end].copy_(target_k_cache[:, :, start:end])
            self.v_cache[:, :, start:end].copy_(target_v_cache[:, :, start:end])
        self.cache_seqlens.fill_(end)
        self.cache_seqlens_next.fill_(end + 1)

    def _activation_workspace(self, in_features: int, num_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
        packed_k = int(in_features)
        cache_key = (packed_k, int(num_groups))
        cached = self._activation_buffers.get(cache_key)
        if cached is None:
            cached = (
                torch.empty((packed_k,), device=self.device, dtype=torch.int8),
                torch.empty((num_groups,), device=self.device, dtype=torch.float32),
            )
            self._activation_buffers[cache_key] = cached
        return cached

    def _linear(self, weight: W4A4LinearWeight, input_vec: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        num_groups = int(weight.scale.shape[1]) if weight.scale.ndim >= 2 else 1
        packed_act, act_scale = self._activation_workspace(weight.in_features, num_groups)
        return w4a4_gemv_inference(
            input_vec.contiguous(),
            weight.packed,
            weight.scale,
            output=output,
            packed_activation=packed_act,
            activation_scale=act_scale,
            activation_clip_ratio=self.activation_clip_ratio,
        )

    def _position_embeddings(self, position: int, steps: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = torch.arange(position, position + steps, device=self.device, dtype=torch.long)
        position_ids = position_ids.view(1, 1, steps).expand(3, -1, -1)
        return self.text_model.rotary_emb.compute(position_ids, dtype=self.dtype, device=self.device)

    def _decode_one_impl(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        output_token: torch.Tensor | None = None,
        set_cache_seqlens: bool = True,
    ) -> torch.Tensor:
        hidden_states = self.text_model.embed_tokens(input_ids).to(dtype=self.dtype)
        residual = hidden_states
        hidden_states = F.rms_norm(
            residual,
            (self.config.hidden_size,),
            self.text_model.layers[0].input_layernorm.weight,
            self.text_model.layers[0].input_layernorm.variance_epsilon,
        )
        if set_cache_seqlens:
            self.cache_seqlens.fill_(prefix_len)
            self.cache_seqlens_next.fill_(prefix_len + 1)

        for layer_idx, layer in enumerate(self.text_model.layers):
            draft_layer = self.layers[layer_idx]
            sa = layer.self_attn
            self._linear(draft_layer.qkv, hidden_states.view(-1), self._qkv_out)
            q_eps = getattr(sa.q_norm, "variance_epsilon", getattr(sa.q_norm, "eps", 1e-6))
            k_eps = getattr(sa.k_norm, "variance_epsilon", getattr(sa.k_norm, "eps", 1e-6))
            cos, sin = position_embeddings
            decode_packed_qkv_cache_inplace(
                self._qkv_out.view(1, -1),
                sa.q_norm.weight,
                sa.k_norm.weight,
                cos.view(1, -1),
                sin.view(1, -1),
                self.k_cache[layer_idx, 0],
                self.v_cache[layer_idx, 0],
                self.cache_seqlens,
                self._query_out,
                num_heads=sa.num_heads,
                num_key_value_heads=sa.num_key_value_heads,
                head_dim=sa.head_dim,
                q_eps=q_eps,
                k_eps=k_eps,
            )
            attn_out = flash_attn_with_kvcache(
                self._query_out.view(1, 1, sa.num_heads, sa.head_dim),
                self.k_cache[layer_idx],
                self.v_cache[layer_idx],
                k=None,
                v=None,
                cache_seqlens=self.cache_seqlens_next,
                softmax_scale=sa.scaling,
                causal=True,
            )
            self._linear(draft_layer.o_proj, attn_out.view(-1), self._attn_out)
            residual = residual + self._attn_out.view(1, 1, -1)
            hidden_states = F.rms_norm(
                residual,
                (self.config.hidden_size,),
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon,
            )
            self._linear(draft_layer.gate_up, hidden_states.view(-1), self._gate_up_out)
            gate, up = self._gate_up_out.chunk(2, dim=0)
            mlp_hidden = F.silu(gate) * up
            self._linear(draft_layer.down_proj, mlp_hidden.contiguous(), self._down_out)
            residual = residual + self._down_out.view(1, 1, -1)
            if layer_idx + 1 < len(self.text_model.layers):
                next_norm = self.text_model.layers[layer_idx + 1].input_layernorm
            else:
                next_norm = self.text_model.norm
            hidden_states = F.rms_norm(
                residual,
                (self.config.hidden_size,),
                next_norm.weight,
                next_norm.variance_epsilon,
            )

        self._linear(self.lm_head, hidden_states.view(-1), self._lm_logits)
        token = torch.argmax(self._lm_logits, dim=0).view(1, 1)
        if output_token is not None:
            output_token.copy_(token)
            return output_token
        return token

    def _capture_decode_graph(self) -> None:
        if self.k_cache is None or self.v_cache is None:
            return
        if self._decode_graph is not None and self._decode_graph_cache_len == self._max_cache_len:
            return
        saved_k0 = self.k_cache[:, :, :1].clone()
        saved_v0 = self.v_cache[:, :, :1].clone()
        saved_seqlens = self.cache_seqlens.clone()
        saved_seqlens_next = self.cache_seqlens_next.clone()
        self._graph_input_ids.zero_()
        graph_pos = self._position_embeddings(0, 1)
        self._graph_position_cos.copy_(graph_pos[0])
        self._graph_position_sin.copy_(graph_pos[1])
        self.cache_seqlens.zero_()
        self.cache_seqlens_next.fill_(1)
        self._decode_one_impl(
            self._graph_input_ids,
            0,
            (self._graph_position_cos, self._graph_position_sin),
            output_token=self._graph_next_token,
            set_cache_seqlens=False,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._decode_one_impl(
                self._graph_input_ids,
                0,
                (self._graph_position_cos, self._graph_position_sin),
                output_token=self._graph_next_token,
                set_cache_seqlens=False,
            )
        self.k_cache[:, :, :1].copy_(saved_k0)
        self.v_cache[:, :, :1].copy_(saved_v0)
        self.cache_seqlens.copy_(saved_seqlens)
        self.cache_seqlens_next.copy_(saved_seqlens_next)
        self._decode_graph = graph
        self._decode_graph_cache_len = self._max_cache_len

    def _decode_one(
        self,
        input_ids: torch.Tensor,
        prefix_len: int,
        rope_delta: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if position_embeddings is None:
            position_embeddings = self._position_embeddings(prefix_len + rope_delta, 1)
        if os.getenv("W4A4_DRAFT_DISABLE_CUDA_GRAPH", "0") != "1":
            self._capture_decode_graph()
            if self._decode_graph is not None:
                self._graph_input_ids.copy_(input_ids.view(1, 1).to(device=self.device, dtype=torch.long))
                self._graph_position_cos.copy_(position_embeddings[0])
                self._graph_position_sin.copy_(position_embeddings[1])
                self.cache_seqlens.fill_(prefix_len)
                self.cache_seqlens_next.fill_(prefix_len + 1)
                self._decode_graph.replay()
                return self._graph_next_token
        return self._decode_one_impl(input_ids, prefix_len, position_embeddings)

    @torch.inference_mode()
    def draft(self, previous_token: torch.Tensor, *, prefix_len: int, rope_delta: int, steps: int) -> torch.Tensor:
        if steps <= 0:
            return torch.empty((1, 0), device=self.device, dtype=torch.long)
        current = previous_token.view(1, 1).to(device=self.device, dtype=torch.long)
        out = torch.empty((1, steps), device=self.device, dtype=torch.long)
        position_embeddings = self._position_embeddings(prefix_len + rope_delta, steps)
        for step in range(steps):
            step_pe = (
                position_embeddings[0][:, step : step + 1, :],
                position_embeddings[1][:, step : step + 1, :],
            )
            token = self._decode_one(current, prefix_len + step, rope_delta, step_pe)
            out[:, step] = token.view(1)
            current = token
        return out
