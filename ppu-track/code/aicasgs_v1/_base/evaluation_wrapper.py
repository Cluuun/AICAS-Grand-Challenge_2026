from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import types
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache


_THIS_WRAPPER_PATH = Path(__file__).resolve()
_ROOT_WRAPPER_PATH = _THIS_WRAPPER_PATH.parents[1] / "evaluation_wrapper.py"
_COMBO_WRAPPER_PATH = (
    _THIS_WRAPPER_PATH.parent / "flashdirect_visionpack" / "evaluation_wrapper.py"
)


def _load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


_COMBO_MODULE = _load_module(
    "aicas_submission_prefill_flashdirect_visionpack_combo_20260417",
    _COMBO_WRAPPER_PATH,
)
_QWEN3_VL_MODELING = importlib.import_module(
    "transformers.models.qwen3_vl.modeling_qwen3_vl"
)


class VLMModel(_COMBO_MODULE.VLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "prefill_bare_runner" not in self._optimizations_applied:
            self._optimizations_applied.append("prefill_bare_runner")
        self._configure_prefill_graph_support()
        self._bare_prefill_warned = False
        print("[VLMModel] bare multimodal prefill runner 已启用: scope=ttft+manual_prefill")
        self._source_config = self._build_source_config()

    def _configure_prefill_graph_support(self) -> None:
        cfg = getattr(self, "_text_opt_config", None)
        if not isinstance(cfg, dict):
            cfg = {}
            self._text_opt_config = cfg
        cfg.setdefault("prefill_graph_enabled", True)
        cfg.setdefault(
            "prefill_graph_warmup_steps",
            max(int(cfg.get("decode_graph_warmup_steps", 3) or 3), 1),
        )
        cfg.setdefault("prefill_graph_prompt_bucket", 64)
        cfg.setdefault("prefill_graph_max_prompt_len", 2048)
        cfg.setdefault("prefill_graph_max_runtimes", 8)
        self._prefill_graph_runtimes = {}
        self._prefill_graph_disable_reason = None
        self._prefill_graph_warned = False
        self._prefill_graph_primed = 0
        self._manual_vision_grid_runtimes = {}
        self._bare_visual_warned = False

    def _build_source_config(self) -> dict[str, object]:
        config = super()._build_source_config()
        if not isinstance(config, dict):
            return {}

        wrapper_sha256 = _sha256_file(_ROOT_WRAPPER_PATH)
        config["wrapper_path"] = str(_ROOT_WRAPPER_PATH)
        config["wrapper_sha256"] = wrapper_sha256

        applied = list(config.get("applied_optimizations", []) or [])
        if "prefill_flash_direct" in getattr(self, "_optimizations_applied", []):
            tag = "prefill_attention:flash_direct"
            if tag not in applied:
                applied.append(tag)
        if "vision_qkv_rotary_pack_attention" in getattr(self, "_optimizations_applied", []):
            tag = "vision_attention:qkv_rotary_pack"
            if tag not in applied:
                applied.append(tag)
        if "prefill_bare_runner" in getattr(self, "_optimizations_applied", []):
            tag = "prefill_runtime:bare_runner"
            if tag not in applied:
                applied.append(tag)
        config["applied_optimizations"] = applied

        enabled = list(config.get("enabled_optimizations", []) or [])
        for tag in ("prefill_attention", "vision_attention", "prefill_runtime"):
            if tag not in enabled:
                enabled.append(tag)
        config["enabled_optimizations"] = enabled

        variants = dict(config.get("variants", {}) or {})
        variants["prefill_attention"] = "flash_direct_qkrotary"
        variants["vision_attention"] = "triton_qkv_rotary_pack"
        variants["prefill_runtime"] = "bare_runner_flashdirect"
        config["variants"] = variants

        decode_runtime = dict(config.get("decode_runtime", {}) or {})
        decode_runtime["runtime_components"] = list(getattr(self, "_optimizations_applied", []) or [])
        decode_runtime["prefill_backend"] = "bare_runner"
        decode_runtime["prefill_bare_runner"] = (
            "prefill_bare_runner" in getattr(self, "_optimizations_applied", [])
        )
        decode_runtime["prefill_qk_norm_rotary_attention"] = (
            "prefill_qk_norm_rotary_attention" in getattr(self, "_optimizations_applied", [])
        )
        decode_runtime["prefill_flash_direct"] = (
            "prefill_flash_direct" in getattr(self, "_optimizations_applied", [])
        )
        decode_runtime["vision_qkv_rotary_pack_attention"] = (
            "vision_qkv_rotary_pack_attention" in getattr(self, "_optimizations_applied", [])
        )
        decode_runtime["prefill_graph_enabled"] = bool(
            getattr(self, "_text_opt_config", {}).get("prefill_graph_enabled", False)
        )
        decode_runtime["prefill_graph_prompt_bucket"] = int(
            getattr(self, "_text_opt_config", {}).get("prefill_graph_prompt_bucket", 0) or 0
        )
        config["decode_runtime"] = decode_runtime

        signature_payload = {
            "profile": config.get("profile"),
            "variant": self._resolved_decode_runtime_variant,
            "model_dtype": self._requested_model_dtype_name,
            "processor": dict(config.get("processor", {}) or {}),
            "decode_runtime": decode_runtime,
            "wrapper_sha256": wrapper_sha256,
        }
        config["wrapper_source_signature"] = _stable_sha256(signature_payload)
        return config

    def _prefill_graph_is_enabled(self) -> bool:
        cfg = getattr(self, "_text_opt_config", {}) or {}
        if not bool(cfg.get("prefill_graph_enabled", False)):
            return False
        if not torch.cuda.is_available():
            return False
        if getattr(self, "_runtime_profiling_enabled", False):
            return False
        if getattr(self, "_prefill_graph_disable_reason", None):
            return False
        return True

    def _align_prefill_graph_bucket_len(self, seq_len: int) -> int:
        cfg = getattr(self, "_text_opt_config", {}) or {}
        bucket = max(int(cfg.get("prefill_graph_prompt_bucket", 64) or 64), 1)
        return max(((max(int(seq_len), 1) + bucket - 1) // bucket) * bucket, 1)

    def _normalize_bare_prefill_position_ids(
        self,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, position_ids.shape[0], -1)
        if position_ids.ndim != 3:
            raise RuntimeError(f"unsupported position_ids shape for bare prefill: {tuple(position_ids.shape)}")
        if int(position_ids.shape[0]) == 4:
            return position_ids[0], position_ids[1:]
        if int(position_ids.shape[0]) == 3:
            return None, position_ids
        raise RuntimeError(f"unsupported prefill position_ids rank: {tuple(position_ids.shape)}")

    def _expand_bare_prefill_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, position_ids.shape[0], -1)
        if position_ids.ndim != 3:
            raise RuntimeError(f"unsupported position_ids shape for bare prefill: {tuple(position_ids.shape)}")
        return position_ids

    def _build_visual_grid_signature(self, image_grid_thw: torch.Tensor) -> tuple[int, ...]:
        if not torch.is_tensor(image_grid_thw):
            raise RuntimeError("manual vision runtime requires image_grid_thw tensor")
        return tuple(
            int(x) for x in image_grid_thw.detach().to("cpu", dtype=torch.long).view(-1).tolist()
        )

    def _get_or_create_manual_vision_grid_runtime(self, image_grid_thw: torch.Tensor) -> dict:
        signature = self._build_visual_grid_signature(image_grid_thw)
        runtime = self._manual_vision_grid_runtimes.get(signature)
        if runtime is not None:
            return runtime

        multimodal_root = getattr(self._model, "model", None)
        visual = getattr(multimodal_root, "visual", None)
        if visual is None:
            raise RuntimeError("manual vision runtime cannot resolve qwen3-vl visual encoder")

        grid_cpu = image_grid_thw.detach().to("cpu", dtype=torch.long)
        grid_cuda = image_grid_thw.detach().to(device=self._device, dtype=torch.long)

        pos_embeds = visual.fast_pos_embed_interpolate(grid_cpu).to(
            device=self._device, dtype=visual.dtype
        )
        rotary_pos_emb = visual.rot_pos_emb(grid_cpu)
        seq_len = int(pos_embeds.shape[0])
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        rotary_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (rotary_emb.cos(), rotary_emb.sin())

        token_lengths = torch.repeat_interleave(
            grid_cuda[:, 1] * grid_cuda[:, 2],
            grid_cuda[:, 0],
        ).to(dtype=torch.int32)
        cu_seqlens = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=self._device),
                token_lengths.cumsum(dim=0, dtype=torch.int32),
            ],
            dim=0,
        )

        deepstack_index_map = {
            int(layer_idx): idx
            for idx, layer_idx in enumerate(list(getattr(visual, "deepstack_visual_indexes", []) or []))
        }
        runtime = {
            "signature": signature,
            "pos_embeds": pos_embeds,
            "position_embeddings": position_embeddings,
            "cu_seqlens": cu_seqlens,
            "deepstack_index_map": deepstack_index_map,
        }
        self._manual_vision_grid_runtimes[signature] = runtime
        return runtime

    def _run_manual_vision_encoder(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        visual_runtime: dict | None = None,
    ):
        multimodal_root = getattr(self._model, "model", None)
        visual = getattr(multimodal_root, "visual", None)
        if visual is None:
            raise RuntimeError("manual vision runner cannot resolve qwen3-vl visual encoder")
        if visual_runtime is None:
            visual_runtime = self._get_or_create_manual_vision_grid_runtime(image_grid_thw)

        hidden_states = visual.patch_embed(pixel_values.type(visual.dtype))
        hidden_states = hidden_states + visual_runtime["pos_embeds"].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        position_embeddings = visual_runtime["position_embeddings"]
        cu_seqlens = visual_runtime["cu_seqlens"]
        deepstack_index_map = visual_runtime["deepstack_index_map"]

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(visual.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            merger_idx = deepstack_index_map.get(int(layer_num))
            if merger_idx is not None:
                deepstack_feature = visual.deepstack_merger_list[merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = visual.merger(hidden_states)
        return types.SimpleNamespace(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )

    def _build_prefill_graph_bucket_key(self, model_kwargs: dict) -> tuple | None:
        if not self._prefill_graph_is_enabled():
            return None
        if not isinstance(model_kwargs, dict):
            return None
        input_ids = model_kwargs.get("input_ids")
        mm_token_type_ids = model_kwargs.get("mm_token_type_ids")
        pixel_values = model_kwargs.get("pixel_values")
        image_grid_thw = model_kwargs.get("image_grid_thw")
        position_ids = model_kwargs.get("position_ids")
        if (
            not torch.is_tensor(input_ids)
            or input_ids.dim() != 2
            or int(input_ids.shape[0]) != 1
            or not input_ids.is_cuda
            or model_kwargs.get("past_key_values") is not None
            or model_kwargs.get("pixel_values_videos") is not None
            or model_kwargs.get("video_grid_thw") is not None
            or not torch.is_tensor(mm_token_type_ids)
            or mm_token_type_ids.shape != input_ids.shape
            or not torch.is_tensor(pixel_values)
            or not torch.is_tensor(image_grid_thw)
            or not torch.is_tensor(position_ids)
        ):
            return None
        if int(input_ids.shape[1]) <= 0:
            return None

        bucket_len = self._align_prefill_graph_bucket_len(int(input_ids.shape[1]))
        cfg = getattr(self, "_text_opt_config", {}) or {}
        max_len = int(cfg.get("prefill_graph_max_prompt_len", 2048) or 2048)
        if bucket_len > max_len:
            return None
        try:
            position_ids = self._expand_bare_prefill_position_ids(position_ids)
        except Exception:
            return None
        try:
            image_grid_signature = self._build_visual_grid_signature(image_grid_thw)
        except Exception:
            return None
        return (
            bucket_len,
            tuple(int(x) for x in pixel_values.shape),
            str(pixel_values.dtype),
            image_grid_signature,
            tuple(int(x) for x in position_ids.shape[:-1]),
        )

    def _build_prefill_graph_model_kwargs(self, model_kwargs: dict) -> dict | None:
        if not isinstance(model_kwargs, dict):
            return None
        graph_kwargs = dict(model_kwargs)
        graph_kwargs["use_cache"] = False
        graph_kwargs.pop("past_key_values", None)
        graph_kwargs.pop("cache_position", None)
        graph_kwargs.pop("cache_implementation", None)
        self._clear_generation_config_cache_impl(graph_kwargs)
        graph_kwargs.setdefault("logits_to_keep", 1)
        self._maybe_attach_prefill_position_ids(graph_kwargs)
        if self._build_prefill_graph_bucket_key(graph_kwargs) is None:
            return None
        return graph_kwargs

    def _copy_prefill_graph_inputs_into_slots(self, runtime: dict, model_kwargs: dict) -> int:
        input_ids = model_kwargs["input_ids"]
        mm_token_type_ids = model_kwargs["mm_token_type_ids"]
        pixel_values = model_kwargs["pixel_values"]
        image_grid_thw = model_kwargs["image_grid_thw"]
        position_ids = model_kwargs["position_ids"]
        if not torch.is_tensor(position_ids):
            raise RuntimeError("prefill graph requires position_ids")
        position_ids = self._expand_bare_prefill_position_ids(position_ids)

        actual_len = int(input_ids.shape[1])
        bucket_len = int(runtime["input_ids"].shape[1])
        if actual_len > bucket_len:
            raise RuntimeError(f"prefill graph bucket overflow: actual={actual_len}, bucket={bucket_len}")

        runtime["input_ids"].zero_()
        runtime["mm_token_type_ids"].zero_()
        runtime["position_ids"].zero_()
        runtime["input_ids"][:, :actual_len].copy_(input_ids)
        runtime["mm_token_type_ids"][:, :actual_len].copy_(mm_token_type_ids)
        runtime["position_ids"][..., :actual_len].copy_(position_ids)
        runtime["pixel_values"].copy_(pixel_values)
        runtime["image_grid_thw"].copy_(image_grid_thw)
        return actual_len

    def _create_prefill_graph_runtime(self, bucket_key: tuple, model_kwargs: dict) -> dict:
        if not torch.cuda.is_available():
            raise RuntimeError("prefill CUDA Graph 仅支持 CUDA 环境")

        input_ids = model_kwargs["input_ids"]
        mm_token_type_ids = model_kwargs["mm_token_type_ids"]
        pixel_values = model_kwargs["pixel_values"]
        image_grid_thw = model_kwargs["image_grid_thw"]
        position_ids = model_kwargs["position_ids"]
        position_ids = self._expand_bare_prefill_position_ids(position_ids)

        bucket_len = int(bucket_key[0])
        cfg = getattr(self, "_text_opt_config", {}) or {}
        warmup_steps = max(int(cfg.get("prefill_graph_warmup_steps", 3) or 3), 1)

        static_input_ids = torch.zeros((1, bucket_len), dtype=input_ids.dtype, device=self._device)
        static_mm_token_type_ids = torch.zeros(
            (1, bucket_len), dtype=mm_token_type_ids.dtype, device=self._device
        )
        static_position_ids = torch.zeros(
            (*tuple(int(x) for x in position_ids.shape[:-1]), bucket_len),
            dtype=position_ids.dtype,
            device=self._device,
        )
        static_pixel_values = torch.zeros_like(pixel_values, device=self._device)
        static_image_grid_thw = torch.zeros_like(image_grid_thw, device=self._device)

        runtime = {
            "bucket_key": bucket_key,
            "input_ids": static_input_ids,
            "mm_token_type_ids": static_mm_token_type_ids,
            "position_ids": static_position_ids,
            "pixel_values": static_pixel_values,
            "image_grid_thw": static_image_grid_thw,
        }
        self._copy_prefill_graph_inputs_into_slots(runtime, model_kwargs)
        visual_runtime = self._get_or_create_manual_vision_grid_runtime(image_grid_thw)
        visual_positions = (mm_token_type_ids[0] == 1).nonzero(as_tuple=False).flatten()
        runtime["visual_positions"] = visual_positions

        static_kwargs = {
            "input_ids": static_input_ids,
            "mm_token_type_ids": static_mm_token_type_ids,
            "position_ids": static_position_ids,
            "pixel_values": static_pixel_values,
            "image_grid_thw": static_image_grid_thw,
            "_prefill_visual_runtime": visual_runtime,
            "_prefill_visual_positions": visual_positions,
            "use_cache": False,
            "return_dict": True,
        }

        def _prefill_body():
            hidden_states, _, _ = self._run_bare_prefill_hidden_states(static_kwargs)
            return hidden_states

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream(device=self._device)
        with torch.inference_mode():
            for _ in range(warmup_steps):
                _ = _prefill_body()
            with torch.cuda.stream(capture_stream):
                capture_stream.synchronize()
            with torch.cuda.graph(graph, stream=capture_stream):
                captured_hidden_states = _prefill_body()

        runtime["graph"] = graph
        runtime["hidden_states"] = captured_hidden_states
        return runtime

    def _get_or_create_prefill_graph_runtime(self, model_kwargs: dict) -> tuple[dict | None, tuple | None]:
        bucket_key = self._build_prefill_graph_bucket_key(model_kwargs)
        if bucket_key is None:
            return None, None
        runtime = self._prefill_graph_runtimes.get(bucket_key)
        if runtime is not None:
            return runtime, bucket_key

        cfg = getattr(self, "_text_opt_config", {}) or {}
        max_runtimes = max(int(cfg.get("prefill_graph_max_runtimes", 8) or 8), 1)
        if len(self._prefill_graph_runtimes) >= max_runtimes:
            return None, bucket_key

        runtime = self._create_prefill_graph_runtime(bucket_key, model_kwargs)
        self._prefill_graph_runtimes[bucket_key] = runtime
        print(
            "[VLMModel] TTFT prefill CUDA Graph 已捕获: "
            f"bucket_len={int(bucket_key[0])}, pixel_shape={bucket_key[1]}, image_grid={bucket_key[3]}"
        )
        return runtime, bucket_key

    def _maybe_prime_ttft_prefill_graph(self, model_kwargs: dict) -> None:
        active = getattr(self, "_active_generation_max_new_tokens", None)
        if active is None or int(active) > 10:
            return
        graph_kwargs = self._build_prefill_graph_model_kwargs(model_kwargs)
        if graph_kwargs is None:
            return
        bucket_key = self._build_prefill_graph_bucket_key(graph_kwargs)
        if bucket_key is None or bucket_key in self._prefill_graph_runtimes:
            return
        try:
            runtime, created_key = self._get_or_create_prefill_graph_runtime(graph_kwargs)
            if runtime is not None and created_key is not None:
                self._prefill_graph_primed += 1
        except Exception as exc:
            self._prefill_graph_disable_reason = str(exc)
            if not self._prefill_graph_warned:
                print(f"[VLMModel] TTFT prefill CUDA Graph 已禁用，回退 eager bare prefill: {exc}")
                self._prefill_graph_warned = True

    def _try_run_ttft_prefill_graph(self, model_kwargs: dict):
        graph_kwargs = self._build_prefill_graph_model_kwargs(model_kwargs)
        if graph_kwargs is None:
            return None
        bucket_key = self._build_prefill_graph_bucket_key(graph_kwargs)
        if bucket_key is None:
            return None
        runtime = self._prefill_graph_runtimes.get(bucket_key)
        if runtime is None:
            return None
        actual_visual_positions = (graph_kwargs["mm_token_type_ids"][0] == 1).nonzero(
            as_tuple=False
        ).flatten()
        if not torch.equal(actual_visual_positions, runtime["visual_positions"]):
            return None

        actual_len = self._copy_prefill_graph_inputs_into_slots(runtime, graph_kwargs)
        runtime["graph"].replay()

        text_model, lm_head = self._get_text_stack()
        last_hidden = runtime["hidden_states"].narrow(1, actual_len - 1, 1)
        logits = self._run_decode_lm_head(last_hidden, lm_head=lm_head)
        qwen_vl_model = getattr(self._model, "model", None)
        rope_deltas = getattr(qwen_vl_model, "rope_deltas", None)
        output_cls = getattr(_QWEN3_VL_MODELING, "Qwen3VLCausalLMOutputWithPast", None)
        if output_cls is not None:
            return output_cls(logits=logits, past_key_values=None, rope_deltas=rope_deltas)
        return types.SimpleNamespace(logits=logits, past_key_values=None, rope_deltas=rope_deltas)

    def _build_prefill_forward_kwargs(self, kwargs):
        model_kwargs = super()._build_prefill_forward_kwargs(kwargs)
        self._maybe_attach_prefill_position_ids(model_kwargs)
        return model_kwargs

    def _build_prefill_mrope_position_ids_fast(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not torch.is_tensor(input_ids)
            or not torch.is_tensor(mm_token_type_ids)
            or input_ids.dim() != 2
            or mm_token_type_ids.shape != input_ids.shape
            or int(input_ids.shape[0]) != 1
        ):
            return None
        if video_grid_thw is not None:
            return None
        if not torch.is_tensor(image_grid_thw) or image_grid_thw.numel() < 3:
            return None
        if attention_mask is not None and (
            not torch.is_tensor(attention_mask) or attention_mask.shape != input_ids.shape
        ):
            return None

        qwen_vl_model = getattr(self._model, "model", None)
        get_vision_position_ids = getattr(qwen_vl_model, "get_vision_position_ids", None)
        vision_config = getattr(getattr(qwen_vl_model, "config", None), "vision_config", None)
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", None)
        if not callable(get_vision_position_ids) or spatial_merge_size is None:
            return None

        token_types = mm_token_type_ids[0]
        valid_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask[0].bool()
            if int(valid_mask.sum().item()) <= 0:
                return None
            token_types = token_types[valid_mask]
        token_types = token_types.to(dtype=torch.long)
        if token_types.numel() <= 0 or not bool((token_types == 1).any().item()):
            return None

        run_types, run_counts = torch.unique_consecutive(token_types, return_counts=True)
        if run_types.numel() <= 0:
            return None

        device = input_ids.device
        current_pos = 0
        image_index = 0
        position_chunks = []
        for modality_type, run_count in zip(run_types.tolist(), run_counts.tolist()):
            if modality_type == 0:
                chunk = (
                    torch.arange(run_count, device=device, dtype=torch.long)
                    .view(1, -1)
                    .expand(3, -1)
                    + current_pos
                )
                current_pos += run_count
            elif modality_type == 1:
                if image_index >= int(image_grid_thw.shape[0]):
                    return None
                grid_thw = image_grid_thw[image_index]
                chunk = get_vision_position_ids(
                    current_pos,
                    grid_thw,
                    1,
                    int(spatial_merge_size),
                    device=device,
                )
                if int(chunk.shape[-1]) != int(run_count):
                    return None
                current_pos += max(int(grid_thw[1].item()), int(grid_thw[2].item())) // int(spatial_merge_size)
                image_index += 1
            else:
                return None
            position_chunks.append(chunk)

        if not position_chunks:
            return None

        llm_positions = torch.cat(position_chunks, dim=1).reshape(3, -1)
        valid_len = int(token_types.shape[0])
        rope_delta_value = int(llm_positions.max().item()) + 1 - valid_len
        rope_deltas = torch.tensor([[rope_delta_value]], device=device, dtype=torch.long)
        if valid_mask is not None:
            position_ids = torch.zeros((3, 1, int(input_ids.shape[1])), device=device, dtype=torch.long)
            position_ids[:, 0, valid_mask] = llm_positions
        else:
            position_ids = llm_positions.view(3, 1, -1)
        return position_ids, rope_deltas

    def _maybe_attach_prefill_position_ids(self, model_kwargs: dict) -> None:
        if not isinstance(model_kwargs, dict) or model_kwargs.get("position_ids") is not None:
            return
        if model_kwargs.get("past_key_values") is not None:
            return
        built = self._build_prefill_mrope_position_ids_fast(
            input_ids=model_kwargs.get("input_ids"),
            mm_token_type_ids=model_kwargs.get("mm_token_type_ids"),
            image_grid_thw=model_kwargs.get("image_grid_thw"),
            video_grid_thw=model_kwargs.get("video_grid_thw"),
            attention_mask=model_kwargs.get("attention_mask"),
        )
        if built is None:
            return
        position_ids, rope_deltas = built
        model_kwargs["position_ids"] = position_ids
        qwen_vl_model = getattr(self._model, "model", None)
        if qwen_vl_model is not None:
            try:
                qwen_vl_model.rope_deltas = rope_deltas
            except Exception:
                pass

    def _coalesce_visual_feature_tensor(self, features):
        if torch.is_tensor(features):
            return features
        if isinstance(features, (list, tuple)):
            tensors = [feat for feat in features if torch.is_tensor(feat)]
            if not tensors:
                return None
            if len(tensors) == 1:
                return tensors[0]
            return torch.cat(tensors, dim=0)
        return None

    def _apply_bare_prefill_deepstack(
        self,
        hidden_states: torch.Tensor,
        visual_positions: torch.Tensor | None,
        visual_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            visual_positions is None
            or visual_embeds is None
            or int(visual_positions.numel()) <= 0
        ):
            return hidden_states
        if int(visual_embeds.shape[0]) != int(visual_positions.numel()):
            raise RuntimeError(
                f"deepstack visual length mismatch: pos={int(visual_positions.numel())}, embeds={int(visual_embeds.shape[0])}"
            )
        hidden_states[0].index_add_(
            0,
            visual_positions.to(device=hidden_states.device, dtype=torch.long),
            visual_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        return hidden_states

    def _prepare_bare_prefill_inputs(self, model_kwargs: dict):
        text_model, _ = self._get_text_stack()
        input_ids = model_kwargs.get("input_ids")
        if not torch.is_tensor(input_ids):
            raise RuntimeError("bare prefill runner requires input_ids")

        inputs_embeds = text_model.embed_tokens(input_ids)
        pixel_values = model_kwargs.get("pixel_values")
        if pixel_values is None:
            return inputs_embeds, None, None

        mm_token_type_ids = model_kwargs.get("mm_token_type_ids")
        image_grid_thw = model_kwargs.get("image_grid_thw")
        if not torch.is_tensor(mm_token_type_ids) or mm_token_type_ids.shape != input_ids.shape:
            raise RuntimeError("bare prefill runner requires mm_token_type_ids matching input_ids")
        if not torch.is_tensor(image_grid_thw):
            raise RuntimeError("bare prefill runner requires image_grid_thw for multimodal prefill")

        visual_runtime = model_kwargs.get("_prefill_visual_runtime")
        try:
            image_outputs = self._run_manual_vision_encoder(
                pixel_values,
                image_grid_thw,
                visual_runtime=visual_runtime if isinstance(visual_runtime, dict) else None,
            )
        except Exception as exc:
            multimodal_root = getattr(self._model, "model", None)
            if multimodal_root is None or not hasattr(multimodal_root, "get_image_features"):
                raise RuntimeError("bare prefill runner cannot resolve qwen3-vl image feature path") from exc
            if not self._bare_visual_warned:
                print(f"[VLMModel] manual vision runner 回退到 get_image_features: {exc}")
                self._bare_visual_warned = True
            image_outputs = multimodal_root.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True,
            )
        image_features = self._coalesce_visual_feature_tensor(
            getattr(image_outputs, "pooler_output", None)
        )
        if image_features is None:
            raise RuntimeError("bare prefill runner failed to collect image features")

        visual_positions = model_kwargs.get("_prefill_visual_positions")
        if torch.is_tensor(visual_positions):
            visual_positions = visual_positions.to(device=input_ids.device, dtype=torch.long)
        else:
            visual_positions = (mm_token_type_ids[0].to(device=input_ids.device) == 1).nonzero(
                as_tuple=False
            ).flatten()
        if int(visual_positions.numel()) != int(image_features.shape[0]):
            raise RuntimeError(
                f"visual token count mismatch: pos={int(visual_positions.numel())}, image={int(image_features.shape[0])}"
            )
        inputs_embeds[0].index_copy_(
            0,
            visual_positions,
            image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

        deepstack_visual_embeds = None
        raw_deepstack = getattr(image_outputs, "deepstack_features", None)
        if isinstance(raw_deepstack, list):
            deepstack_visual_embeds = [
                feat.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                for feat in raw_deepstack
                if torch.is_tensor(feat)
            ]
        return inputs_embeds, visual_positions, deepstack_visual_embeds

    def _should_use_bare_prefill_runner(self, model_kwargs: dict) -> bool:
        if not isinstance(model_kwargs, dict):
            return False
        input_ids = model_kwargs.get("input_ids")
        if (
            not torch.is_tensor(input_ids)
            or input_ids.dim() != 2
            or int(input_ids.shape[0]) != 1
            or not input_ids.is_cuda
        ):
            return False
        if model_kwargs.get("inputs_embeds") is not None:
            return False
        if model_kwargs.get("pixel_values_videos") is not None or model_kwargs.get("video_grid_thw") is not None:
            return False

        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is not None:
            if not torch.is_tensor(attention_mask) or attention_mask.shape != input_ids.shape:
                return False
            if not bool(attention_mask.bool().all().item()):
                return False

        position_ids = model_kwargs.get("position_ids")
        if not torch.is_tensor(position_ids):
            return False

        past_key_values = model_kwargs.get("past_key_values")
        pixel_values = model_kwargs.get("pixel_values")
        if pixel_values is not None:
            mm_token_type_ids = model_kwargs.get("mm_token_type_ids")
            image_grid_thw = model_kwargs.get("image_grid_thw")
            if past_key_values is not None:
                return False
            if not torch.is_tensor(mm_token_type_ids) or mm_token_type_ids.shape != input_ids.shape:
                return False
            if not torch.is_tensor(image_grid_thw):
                return False
            return True

        return past_key_values is not None

    def _run_bare_prefill_model_forward(self, model_kwargs: dict):
        hidden_states, past_key_values, rope_deltas = self._run_bare_prefill_hidden_states(model_kwargs)
        _, lm_head = self._get_text_stack()
        logits = self._run_decode_lm_head(hidden_states[:, -1:, :], lm_head=lm_head)
        use_cache = bool(model_kwargs.get("use_cache", True))
        output_cls = getattr(_QWEN3_VL_MODELING, "Qwen3VLCausalLMOutputWithPast", None)
        if output_cls is not None:
            return output_cls(
                logits=logits,
                past_key_values=past_key_values if use_cache else None,
                rope_deltas=rope_deltas,
            )
        return types.SimpleNamespace(
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            rope_deltas=rope_deltas,
        )

    def _run_bare_prefill_hidden_states(
        self,
        model_kwargs: dict,
    ) -> tuple[torch.Tensor, object | None, torch.Tensor | None]:
        text_model, _ = self._get_text_stack()
        qwen_vl_model = getattr(self._model, "model", None)
        if qwen_vl_model is None:
            raise RuntimeError("bare prefill runner requires qwen3-vl root model")

        inputs_embeds, visual_positions, deepstack_visual_embeds = self._prepare_bare_prefill_inputs(
            model_kwargs
        )
        position_ids = model_kwargs.get("position_ids")
        if not torch.is_tensor(position_ids):
            raise RuntimeError("bare prefill runner requires explicit position_ids")
        text_position_ids, rotary_position_ids = self._normalize_bare_prefill_position_ids(position_ids)

        use_cache = bool(model_kwargs.get("use_cache", True))
        past_key_values = model_kwargs.get("past_key_values")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=text_model.config)

        cache_position = model_kwargs.get("cache_position")
        hidden_states = inputs_embeds
        position_embeddings = text_model.rotary_emb(hidden_states, rotary_position_ids)

        layer_kwargs = {}
        if cache_position is not None:
            layer_kwargs["cache_position"] = cache_position

        for layer_idx, decoder_layer in enumerate(text_model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **layer_kwargs,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._apply_bare_prefill_deepstack(
                    hidden_states,
                    visual_positions,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = text_model.norm(hidden_states)
        rope_deltas = getattr(qwen_vl_model, "rope_deltas", None)
        return hidden_states, past_key_values if use_cache else None, rope_deltas

    def _run_prefill_model_forward(self, model_kwargs: dict, restore_reason: str):
        with self._visual_prefix_restore_reason_context(restore_reason):
            if self._should_use_bare_prefill_runner(model_kwargs):
                try:
                    return self._run_bare_prefill_model_forward(model_kwargs)
                except Exception as exc:
                    self._record_fastpath_fallback("prefill_bare_runner", exc)
                    if not self._bare_prefill_warned:
                        print(f"[VLMModel] bare prefill runner 回退到 HF forward: {exc}")
                        self._bare_prefill_warned = True
            return self._model(**model_kwargs)

    def _run_ttft_direct_greedy(self, args, kwargs):
        input_ids = self._get_input_ids_for_output(args, kwargs)
        active_plan = self._active_visual_prefix_plan if isinstance(self._active_visual_prefix_plan, dict) else None
        cached_prefix = self._get_cached_visual_prefix_entry(active_plan)
        preferred_impl = str(self._kv_cache_opt_config.get("impl", "static")).lower()

        topk_values_buffer = torch.zeros((1, 1, 1), dtype=torch.float16, device=self._device)
        topk_indices_buffer = torch.zeros((1, 1, 1), dtype=torch.long, device=self._device)

        if cached_prefix is not None and preferred_impl == "static" and "static" not in self._kv_cache_disable_reason:
            pool_key = None
            cache_obj = None
            try:
                batch_size, bucket_len = self._estimate_kv_cache_len_from_prompt_len(
                    prompt_len=int(input_ids.shape[1]),
                    kwargs=kwargs,
                    batch_size=int(input_ids.shape[0]),
                )
                cache_obj, pool_key = self._borrow_kv_cache("static", max_cache_len=bucket_len, batch_size=batch_size)
                _, outputs, _ = self._run_prefill_for_manual_decode(
                    args,
                    kwargs,
                    cache_obj=cache_obj,
                    restore_reason="ttft_direct_static",
                )
                with torch.inference_mode():
                    logits = outputs.logits[:, -1:, :]
                    torch.topk(logits, k=1, dim=-1, out=(topk_values_buffer, topk_indices_buffer))
                    next_token = topk_indices_buffer.squeeze(-1)
                return torch.cat([input_ids, next_token], dim=-1)
            except Exception as exc:
                print(f"[VLMModel] TTFT direct StaticCache 前缀恢复失败，回退动态路径: {exc}")
            finally:
                self._release_kv_cache(cache_obj, pool_key)

        model_kwargs = self._build_prefill_forward_kwargs(kwargs)
        model_kwargs["use_cache"] = False
        model_kwargs.pop("past_key_values", None)
        model_kwargs.pop("cache_position", None)
        model_kwargs.pop("cache_implementation", None)
        self._clear_generation_config_cache_impl(model_kwargs)
        model_kwargs.setdefault("logits_to_keep", 1)
        with torch.inference_mode():
            outputs = self._try_run_ttft_prefill_graph(model_kwargs)
            if outputs is None:
                outputs = self._run_prefill_model_forward(model_kwargs, restore_reason="ttft_direct")
            logits = outputs.logits[:, -1:, :]
            torch.topk(logits, k=1, dim=-1, out=(topk_values_buffer, topk_indices_buffer))
            next_token = topk_indices_buffer.squeeze(-1)
        return torch.cat([input_ids, next_token], dim=-1)

    def _run_prefill_for_manual_decode(self, args, kwargs, cache_obj=None, restore_reason: str | None = None):
        input_ids = self._get_input_ids_for_output(args, kwargs)
        model_kwargs = self._build_prefill_forward_kwargs(kwargs)
        model_kwargs.setdefault("logits_to_keep", 1)
        if cache_obj is not None:
            model_kwargs["past_key_values"] = cache_obj
            cache_position = self._build_prefill_cache_position(input_ids)
            if cache_position is not None:
                model_kwargs["cache_position"] = cache_position
            model_kwargs.pop("cache_implementation", None)
            self._clear_generation_config_cache_impl(model_kwargs)
            try:
                self._try_prepare_static_prefix_prefill(model_kwargs, cache_obj)
            except Exception as exc:
                print(f"[VLMModel] Static visual prefix 恢复失败，回退常规 prefill: {exc}")

        def _run_prefill():
            with torch.inference_mode():
                reason = restore_reason or ("manual_prefill_static" if cache_obj is not None else "manual_prefill_dynamic")
                return self._run_prefill_model_forward(model_kwargs, restore_reason=reason)

        outputs, elapsed_ms = self._measure_transition_elapsed_ms(_run_prefill)
        self._record_decode_transition_metric("prefill_forward_ms", elapsed_ms)
        if cache_obj is not None:
            self._record_decode_transition_metric("static_prefill_direct_ms", elapsed_ms)
        self._maybe_prime_ttft_prefill_graph(model_kwargs)
        prefill_cache = self._get_prefill_cache_object(outputs, explicit_cache=cache_obj)
        if prefill_cache is None:
            raise RuntimeError("prefill 未返回 past_key_values，无法执行手写 decode")
        if cache_obj is not None:
            expected_seq_len = int(input_ids.shape[1])
            actual_seq_len = self._get_cache_seq_len(prefill_cache)
            if actual_seq_len != expected_seq_len:
                raise RuntimeError(
                    f"StaticCache prefill 长度异常: actual={actual_seq_len}, expected={expected_seq_len}"
                )
        return input_ids, outputs, prefill_cache
