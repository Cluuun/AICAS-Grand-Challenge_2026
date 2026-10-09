from __future__ import annotations

import importlib
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from transformers import AutoProcessor
except Exception:  # pragma: no cover
    AutoProcessor = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from PIL import Image as _PILImage
except Exception:  # pragma: no cover
    _PILImage = None

try:
    import xxhash as _xxhash
except Exception:  # pragma: no cover
    _xxhash = None

try:
    from safetensors import safe_open as _safe_open
except Exception:  # pragma: no cover
    _safe_open = None

try:
    from safetensors.torch import save_file as _save_safetensors_file
except Exception:  # pragma: no cover
    _save_safetensors_file = None


_VLLM_PROMPT_KWARG = "_aicas_vllm_prompt"
_VLLM_IMAGE_KWARG = "_aicas_vllm_image"
_VLLM_IMAGE_HASH_KWARG = "_aicas_vllm_image_hash"
_VLLM_PROMPT_TOKEN_IDS_KWARG = "_aicas_vllm_prompt_token_ids"
_VLLM_REQUEST_KWARGS = (
    _VLLM_PROMPT_KWARG,
    _VLLM_IMAGE_KWARG,
    _VLLM_IMAGE_HASH_KWARG,
    _VLLM_PROMPT_TOKEN_IDS_KWARG,
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_text(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip()


def _env_optional_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_awq_prequantized_model_path(model_path: str) -> bool:
    path_lower = str(model_path or "").lower()
    return any(tag in path_lower for tag in ("awq", "w4a16", "compressed"))


class _MinimalPreparedInputs:
    def __init__(self, batch, extra: dict[str, object]):
        self._batch = batch
        self._extra = extra

    def to(self, *args, **kwargs):
        self._batch = self._batch.to(*args, **kwargs)
        return self

    def __getattr__(self, name):
        return getattr(self._batch, name)

    def __getitem__(self, key):
        if key in self._extra:
            return self._extra[key]
        return self._batch[key]

    def __iter__(self):
        for key in self._batch.keys():
            yield key
        for key in self._extra.keys():
            yield key

    def __len__(self):
        return len(self._batch) + len(self._extra)

    def keys(self):
        return list(iter(self))

    def items(self):
        for key in self._batch.keys():
            yield key, self._batch[key]
        for key, value in self._extra.items():
            yield key, value

    def values(self):
        for _, value in self.items():
            yield value


class _MinimalProcessorWrapper:
    def __init__(self, owner, processor):
        self._owner = owner
        self._processor = processor

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def __call__(self, *args, **kwargs):
        return self._processor(*args, **kwargs)

    def apply_chat_template(self, messages, *args, **kwargs):
        if not isinstance(messages, list) or len(messages) != 1:
            return self._processor.apply_chat_template(messages, *args, **kwargs)

        normalized_messages, image_obj = self._owner._normalize_messages_for_vllm(messages)
        prompt_kwargs = dict(kwargs)
        prompt_kwargs["tokenize"] = False
        prompt_kwargs.pop("return_dict", None)
        prompt_kwargs.pop("return_tensors", None)
        prompt = self._processor.apply_chat_template(normalized_messages, *args, **prompt_kwargs)

        extras = {
            _VLLM_PROMPT_KWARG: prompt,
            _VLLM_IMAGE_KWARG: image_obj,
            _VLLM_IMAGE_HASH_KWARG: self._owner._hash_visual_image_input(image_obj) if image_obj is not None else None,
        }
        batch = self._processor.apply_chat_template(normalized_messages, *args, **kwargs)
        return _MinimalPreparedInputs(batch, extras)


class _MinimalVLLMGenerateProxy:
    def __init__(self, owner):
        self._owner = owner
        eos = getattr(owner, "_eos_token_id", None)
        pad = getattr(owner, "_pad_token_id", None)
        self.config = type("MinimalConfig", (), {"eos_token_id": eos, "pad_token_id": pad})()

    def generate(self, *args, **kwargs):
        return self._owner._generate_with_vllm_only(*args, **kwargs)


class _MinimalVLLMVLMModel:
    def __init__(self, model_path: str, device: str | None = "cuda"):
        if AutoProcessor is None:
            raise RuntimeError("transformers AutoProcessor 不可用")
        self.model_path = model_path
        self._device = self._resolve_device(device)
        self._source_config: dict[str, object] = {"backend": "minimal_vllm"}
        self._vllm_module = self._import_vllm()
        self._vllm_sampling_params_cls = getattr(self._vllm_module, "SamplingParams")
        self._vllm_runtime_config = self._build_vllm_runtime_config()
        self._vllm_runtime_config = self._maybe_enable_prequantized_vllm_model(self._vllm_runtime_config, model_path=model_path)
        self._vllm_model_path = self._prepare_model_path_for_vllm(model_path=model_path, config=self._vllm_runtime_config)
        self._vllm_llm = self._build_llm(self._vllm_module, self._vllm_runtime_config, self._vllm_model_path)
        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._processor_wrapper = _MinimalProcessorWrapper(self, self._processor)
        self._eos_token_id = getattr(getattr(self._processor, "tokenizer", None), "eos_token_id", None)
        self._pad_token_id = getattr(getattr(self._processor, "tokenizer", None), "pad_token_id", None)
        if self._pad_token_id is None:
            self._pad_token_id = self._eos_token_id
        self._model = _MinimalVLLMGenerateProxy(self)
        self._source_config["vllm_bridge"] = {
            "enabled": True,
            "mode": "minimal_vllm",
            "quantization": self._vllm_runtime_config.get("quantization", ""),
            "dtype": self._vllm_runtime_config.get("dtype", ""),
            "max_model_len": self._vllm_runtime_config.get("max_model_len", 0),
            "gpu_memory_utilization": self._vllm_runtime_config.get("gpu_memory_utilization", 0.0),
            "kv_cache_dtype": self._vllm_runtime_config.get("kv_cache_dtype", ""),
            "calculate_kv_scales": self._vllm_runtime_config.get("calculate_kv_scales"),
            "quantization_param_path": self._vllm_runtime_config.get("quantization_param_path", ""),
            "quantization_param_source": self._vllm_runtime_config.get("quantization_param_source", ""),
            "kv_scale_calibration": self._vllm_runtime_config.get("kv_scale_calibration", ""),
            "attention_backend": self._vllm_runtime_config.get("attention_backend", ""),
            "mm_encoder_attn_backend": self._vllm_runtime_config.get("mm_encoder_attn_backend", ""),
            "flash_attn_version": self._vllm_runtime_config.get("flash_attn_version"),
            "override_attention_dtype": self._vllm_runtime_config.get("override_attention_dtype", ""),
            "enable_fp8_vit_attn": self._vllm_runtime_config.get("enable_fp8_vit_attn", False),
            "skip_mm_profiling": self._vllm_runtime_config.get("skip_mm_profiling", False),
            "speculative_config": self._vllm_runtime_config.get("speculative_config"),
            "effective_model_path": self._vllm_model_path,
            "materialized_quant_model_dir": self._vllm_runtime_config.get("materialized_quant_model_dir", ""),
        }

    def _resolve_device(self, device: str | None) -> str:
        requested = str(device or "cuda").strip().lower()
        if requested in {"cpu", "mps"}:
            return requested
        if requested in {"", "cuda"}:
            return "cuda:0"
        return str(device)

    def _import_vllm(self):
        for path in (
            os.environ.get("AICAS_VLLM_SITE_PACKAGES"),
            "/tmp/aicas_vllm",
            str(Path.home() / f".local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"),
        ):
            if path and os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        return importlib.import_module("vllm")

    def _build_vllm_runtime_config(self) -> dict[str, object]:
        kv_cache_dtype = self._resolve_vllm_kv_cache_dtype()
        calculate_kv_scales = self._resolve_vllm_calculate_kv_scales(kv_cache_dtype)
        quantization_param_path, quantization_param_source = self._resolve_vllm_quantization_param_path(kv_cache_dtype)
        attention_backend = self._normalize_attention_backend(_env_text("AICAS_VLLM_ATTENTION_BACKEND", "auto"))
        mm_encoder_attn_backend = self._normalize_attention_backend(
            _env_text("AICAS_VLLM_MM_ENCODER_ATTN_BACKEND", attention_backend or "auto")
        )
        flash_attn_version = self._resolve_flash_attn_version()
        override_attention_dtype = _env_text("AICAS_VLLM_OVERRIDE_ATTENTION_DTYPE", "")
        fp8_vit_enabled = _env_flag("AICAS_VLLM_ENABLE_FP8_VIT_ATTN", False)
        if fp8_vit_enabled and not override_attention_dtype:
            override_attention_dtype = _env_text("AICAS_VLLM_FP8_VIT_ATTN_DTYPE", "fp8") or "fp8"
        return {
            "enabled": _env_flag("AICAS_ENABLE_VLLM_BRIDGE", True),
            "dtype": _env_text("AICAS_VLLM_DTYPE", "float16"),
            "gpu_memory_utilization": float(os.environ.get("AICAS_VLLM_GPU_MEMORY_UTIL", 0.88) or 0.88),
            "max_model_len": int(os.environ.get("AICAS_VLLM_MAX_MODEL_LEN", 1024) or 1024),
            "enable_prefix_caching": _env_flag("AICAS_VLLM_ENABLE_PREFIX_CACHING", True),
            "warmup": _env_flag("AICAS_VLLM_WARMUP", True),
            "warmup_size": int(os.environ.get("AICAS_VLLM_WARMUP_SIZE", 448) or 448),
            "omit_media_on_cache_hit": _env_flag("AICAS_VLLM_OMIT_MEDIA_ON_CACHE_HIT", True),
            "kv_cache_dtype": kv_cache_dtype,
            "calculate_kv_scales": calculate_kv_scales,
            "quantization_param_path": quantization_param_path,
            "quantization_param_source": quantization_param_source,
            "kv_scale_calibration": _env_text("AICAS_VLLM_KV_SCALE_CALIBRATION", "auto").lower(),
            "attention_backend": attention_backend,
            "mm_encoder_attn_backend": mm_encoder_attn_backend,
            "flash_attn_version": flash_attn_version,
            "override_attention_dtype": override_attention_dtype,
            "enable_fp8_vit_attn": fp8_vit_enabled,
            "skip_mm_profiling": _env_flag("AICAS_VLLM_SKIP_MM_PROFILING", False),
            "speculative_config": self._resolve_speculative_config(),
        }

    def _normalize_vllm_kv_cache_dtype(self, raw_value: str) -> str:
        value = str(raw_value or "").strip().lower()
        if value in {"", "auto"}:
            return "auto"
        if value in {"float16", "half", "fp16"}:
            return "float16"
        if value in {"bfloat16", "bf16"}:
            return "bfloat16"
        if value in {"fp8", "fp8_auto"}:
            return "fp8"
        if value in {"fp8_e4m3", "e4m3", "fp8-e4m3"}:
            return "fp8_e4m3"
        if value in {"fp8_e5m2", "e5m2", "fp8-e5m2"}:
            return "fp8_e5m2"
        return value

    def _normalize_attention_backend(self, raw_value: str) -> str:
        value = str(raw_value or "").strip()
        if not value or value.lower() in {"auto", "none", "default"}:
            return ""
        return value.upper()

    def _resolve_flash_attn_version(self) -> int | None:
        raw = _env_text("AICAS_VLLM_FLASH_ATTN_VERSION", "")
        if not raw:
            return None
        try:
            version = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"AICAS_VLLM_FLASH_ATTN_VERSION 非法: {raw}") from exc
        if version not in {2, 3, 4}:
            raise RuntimeError(f"AICAS_VLLM_FLASH_ATTN_VERSION 仅支持 2/3/4，当前: {version}")
        return version

    def _resolve_speculative_config(self) -> dict[str, object] | None:
        method = _env_text("AICAS_VLLM_SPEC_METHOD", "").strip().lower()
        if not method:
            return None
        cfg: dict[str, object] = {"method": method}
        num_spec = _env_text("AICAS_VLLM_NUM_SPEC_TOKENS", "")
        if num_spec:
            cfg["num_speculative_tokens"] = int(num_spec)
        reject_method = _env_text("AICAS_VLLM_SPEC_REJECTION_METHOD", "").strip().lower()
        if reject_method:
            cfg["rejection_sample_method"] = reject_method
        if method == "ngram":
            pl_min = _env_text("AICAS_VLLM_PROMPT_LOOKUP_MIN", "")
            pl_max = _env_text("AICAS_VLLM_PROMPT_LOOKUP_MAX", "")
            if pl_min:
                cfg["prompt_lookup_min"] = int(pl_min)
            if pl_max:
                cfg["prompt_lookup_max"] = int(pl_max)
        elif method == "suffix":
            tree_depth = _env_text("AICAS_VLLM_SUFFIX_MAX_TREE_DEPTH", "")
            cached_req = _env_text("AICAS_VLLM_SUFFIX_MAX_CACHED_REQUESTS", "")
            spec_factor = _env_text("AICAS_VLLM_SUFFIX_MAX_SPEC_FACTOR", "")
            min_prob = _env_text("AICAS_VLLM_SUFFIX_MIN_TOKEN_PROB", "")
            if tree_depth:
                cfg["suffix_decoding_max_tree_depth"] = int(tree_depth)
            if cached_req:
                cfg["suffix_decoding_max_cached_requests"] = int(cached_req)
            if spec_factor:
                cfg["suffix_decoding_max_spec_factor"] = float(spec_factor)
            if min_prob:
                cfg["suffix_decoding_min_token_prob"] = float(min_prob)
        return cfg

    def _resolve_vllm_kv_cache_dtype(self) -> str:
        requested_dtype = self._normalize_vllm_kv_cache_dtype(_env_text("AICAS_VLLM_KV_CACHE_DTYPE", "auto"))
        calibration_mode = _env_text("AICAS_VLLM_KV_SCALE_CALIBRATION", "auto").strip().lower()
        quant_path_raw = _env_text("AICAS_VLLM_QUANTIZATION_PARAM_PATH") or _env_text("AICAS_VLLM_KV_SCALE_JSON_PATH")
        calculate_kv_scales = _env_optional_flag("AICAS_VLLM_CALCULATE_KV_SCALES")
        allow_fp8_kv = _env_flag("AICAS_VLLM_ENABLE_FP8_KV_EXPERIMENT", False)
        wants_fp8 = (
            requested_dtype.startswith("fp8")
            or calibration_mode in {"none", "random", "dataset", "file", "json"}
            or quant_path_raw != ""
            or calculate_kv_scales is not None
        )
        if wants_fp8:
            if allow_fp8_kv:
                return requested_dtype if requested_dtype != "auto" else "fp8"
            # 默认关闭 KV 量化。即便外部显式请求 FP8 KV cache，
            # 也只有在显式实验开关开启时才真正放行。
            return "float16"
        if requested_dtype in {"float16", "bfloat16"}:
            # 对当前 Qwen3-VL + vLLM 0.19.1 + FlashAttention 路线，
            # 显式传 kv_cache_dtype=float16 会触发 backend 报错；
            # 使用 auto 会稳定回到模型默认 dtype（本仓默认是 float16）。
            return "auto"
        if requested_dtype == "auto":
            return "auto"
        return requested_dtype

    def _resolve_vllm_calculate_kv_scales(self, kv_cache_dtype: str) -> bool | None:
        explicit = _env_optional_flag("AICAS_VLLM_CALCULATE_KV_SCALES")
        if explicit is not None:
            return False if not str(kv_cache_dtype or "").startswith("fp8") else explicit
        if not str(kv_cache_dtype or "").startswith("fp8"):
            return None
        calibration_mode = _env_text("AICAS_VLLM_KV_SCALE_CALIBRATION", "auto").strip().lower()
        if calibration_mode in {"random", "online", "on_the_fly", "warmup"}:
            return True
        if calibration_mode in {"none", "dataset", "file", "json", "off", "disabled"}:
            return False
        return False

    def _resolve_vllm_quantization_param_path(self, kv_cache_dtype: str) -> tuple[str, str]:
        if not str(kv_cache_dtype or "").startswith("fp8"):
            return "", "disabled"
        raw_path = _env_text("AICAS_VLLM_QUANTIZATION_PARAM_PATH") or _env_text("AICAS_VLLM_KV_SCALE_JSON_PATH")
        if not raw_path:
            return "", "none"
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            raise RuntimeError(f"未找到 FP8 KV cache scale 文件: {path}")
        return str(path), "env"

    def _maybe_enable_prequantized_vllm_model(self, config: dict[str, object], *, model_path: str) -> dict[str, object]:
        requested_method = _env_text("AICAS_VLLM_PREQUANT_METHOD", "auto").lower()
        if requested_method in {"", "none", "disabled", "off"}:
            return config
        if requested_method == "auto" and _is_awq_prequantized_model_path(model_path):
            config = dict(config)
            config["quantization"] = "compressed-tensors"
            config["load_format"] = "auto"
            return config
        if requested_method == "compressed-tensors":
            config = dict(config)
            config["quantization"] = "compressed-tensors"
            config["load_format"] = "auto"
            return config
        return config

    def _build_llm(self, vllm_module, config: dict[str, object], model_path: str):
        llm_cls = getattr(vllm_module, "LLM")
        kwargs = {
            "model": model_path,
            "tokenizer": model_path,
            "trust_remote_code": True,
            "dtype": config["dtype"],
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "gpu_memory_utilization": float(config["gpu_memory_utilization"]),
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "disable_log_stats": True,
            "enable_prefix_caching": bool(config["enable_prefix_caching"]),
            "enforce_eager": _env_flag("AICAS_VLLM_ENFORCE_EAGER", False),
        }
        attention_config: dict[str, object] = {}
        attention_backend = str(config.get("attention_backend", "") or "")
        if attention_backend:
            attention_config["backend"] = attention_backend
        if _env_flag("AICAS_VLLM_USE_PREFILL_DECODE_ATTENTION", False):
            attention_config["use_prefill_decode_attention"] = True
        trtllm_attention = _env_optional_flag("AICAS_VLLM_USE_TRTLLM_ATTENTION")
        if trtllm_attention is not None:
            attention_config["use_trtllm_attention"] = bool(trtllm_attention)
        disable_flashinfer_prefill = _env_optional_flag("AICAS_VLLM_DISABLE_FLASHINFER_PREFILL")
        if disable_flashinfer_prefill is not None:
            attention_config["disable_flashinfer_prefill"] = bool(disable_flashinfer_prefill)
        flash_attn_version = config.get("flash_attn_version")
        if flash_attn_version is not None:
            attention_config["flash_attn_version"] = int(flash_attn_version)
        if attention_config:
            kwargs["attention_config"] = attention_config
        mm_encoder_attn_backend = str(config.get("mm_encoder_attn_backend", "") or "")
        if mm_encoder_attn_backend:
            kwargs["mm_encoder_attn_backend"] = mm_encoder_attn_backend
        override_attention_dtype = str(config.get("override_attention_dtype", "") or "")
        if override_attention_dtype:
            kwargs["override_attention_dtype"] = override_attention_dtype
        if bool(config.get("skip_mm_profiling", False)):
            kwargs["skip_mm_profiling"] = True
        if config.get("speculative_config"):
            kwargs["speculative_config"] = dict(config["speculative_config"])
        if int(config["max_model_len"]) > 0:
            kwargs["max_model_len"] = int(config["max_model_len"])
        if config.get("quantization"):
            kwargs["quantization"] = config["quantization"]
        if config.get("load_format"):
            kwargs["load_format"] = config["load_format"]
        kv_cache_dtype = str(config.get("kv_cache_dtype", "") or "")
        if kv_cache_dtype and kv_cache_dtype != "auto":
            kwargs["kv_cache_dtype"] = kv_cache_dtype
        if kv_cache_dtype.startswith("fp8"):
            if config.get("calculate_kv_scales") is not None:
                kwargs["calculate_kv_scales"] = bool(config["calculate_kv_scales"])
            if config.get("quantization_param_path") and not config.get("materialized_quant_model_dir"):
                kwargs["quantization_param_path"] = str(config["quantization_param_path"])
        kwargs = self._filter_vllm_llm_kwargs(llm_cls, kwargs, config)
        return llm_cls(**kwargs)

    def _prepare_model_path_for_vllm(self, *, model_path: str, config: dict[str, object]) -> str:
        quant_path = str(config.get("quantization_param_path", "") or "")
        if not quant_path:
            return model_path
        if str(config.get("vllm_quant_param_direct", "")) == "1":
            return model_path
        return self._materialize_kv_scales_model_dir(model_path=model_path, quantization_param_path=quant_path, config=config)

    def _materialize_kv_scales_model_dir(
        self,
        *,
        model_path: str,
        quantization_param_path: str,
        config: dict[str, object],
    ) -> str:
        if _safe_open is None or _save_safetensors_file is None:
            raise RuntimeError("需要 safetensors 才能将 KV scale materialize 到临时模型目录")
        src_dir = Path(model_path).expanduser().resolve()
        if not src_dir.is_dir():
            raise RuntimeError(f"模型目录不存在，无法注入 KV scale: {src_dir}")
        quant_path = Path(quantization_param_path).expanduser().resolve()
        if not quant_path.is_file():
            raise RuntimeError(f"KV scale 文件不存在: {quant_path}")

        temp_dir = Path(tempfile.mkdtemp(prefix="aicas_fp8kv_", dir=os.environ.get("AICAS_VLLM_TMPDIR") or None))
        self._copy_model_dir_lightweight(src_dir=src_dir, dst_dir=temp_dir)
        scale_tensors = self._load_kv_scale_tensors_from_json(quant_path)
        scale_filename = "kv_cache_scales.safetensors"
        _save_safetensors_file(scale_tensors, str(temp_dir / scale_filename))
        self._write_quantized_model_index(dst_dir=temp_dir, base_dir=src_dir, extra_filename=scale_filename, extra_tensor_names=list(scale_tensors.keys()))
        self._patch_model_config_for_kv_cache(dst_dir=temp_dir, model_path=src_dir, config=config, has_scales=bool(scale_tensors))
        config["materialized_quant_model_dir"] = str(temp_dir)
        config["quantization_param_source"] = f"{config.get('quantization_param_source', 'env')}+materialized"
        return str(temp_dir)

    def _copy_model_dir_lightweight(self, *, src_dir: Path, dst_dir: Path):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for child in src_dir.iterdir():
            target = dst_dir / child.name
            if child.is_symlink():
                target.symlink_to(os.readlink(child))
                continue
            if child.is_dir():
                shutil.copytree(child, target, symlinks=True)
                continue
            if child.suffix == ".safetensors":
                target.symlink_to(child)
                continue
            shutil.copy2(child, target)

    def _load_kv_scale_tensors_from_json(self, quant_path: Path) -> dict[str, "torch.Tensor"]:
        if torch is None:
            raise RuntimeError("torch 不可用，无法构建 KV scale tensor")
        payload = json.loads(quant_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("KV scale JSON 格式非法：顶层必须是 object")
        tensors: dict[str, torch.Tensor] = {}
        for name, value in payload.items():
            if not isinstance(name, str):
                continue
            if not (name.endswith(".k_scale") or name.endswith(".v_scale") or name.endswith(".q_scale")):
                continue
            if isinstance(value, dict):
                if "values" in value:
                    value = value["values"]
                elif "scale" in value:
                    value = value["scale"]
            if isinstance(value, (int, float)):
                tensor = torch.tensor([float(value)], dtype=torch.float32)
            elif isinstance(value, list):
                flat: list[float] = []
                for item in value:
                    if isinstance(item, list):
                        flat.extend(float(x) for x in item)
                    else:
                        flat.append(float(item))
                tensor = torch.tensor(flat, dtype=torch.float32)
            else:
                raise RuntimeError(f"KV scale JSON 中存在不支持的值格式: {name}")
            remapped_name = self._remap_kv_scale_name_for_compressed_tensors(name)
            tensors[remapped_name] = tensor
        if not tensors:
            raise RuntimeError("KV scale JSON 未解析出任何 *.k_scale/*.v_scale/*.q_scale 张量")
        return tensors

    def _remap_kv_scale_name_for_compressed_tensors(self, name: str) -> str:
        if ".self_attn.attn.k_scale" in name:
            return name.replace(".self_attn.attn.k_scale", ".self_attn.k_proj.k_scale")
        if ".self_attn.attn.v_scale" in name:
            return name.replace(".self_attn.attn.v_scale", ".self_attn.v_proj.v_scale")
        if ".self_attn.attn.q_scale" in name:
            return name.replace(".self_attn.attn.q_scale", ".self_attn.q_proj.q_scale")
        return name

    def _write_quantized_model_index(
        self,
        *,
        dst_dir: Path,
        base_dir: Path,
        extra_filename: str,
        extra_tensor_names: list[str],
    ):
        index_path = base_dir / "model.safetensors.index.json"
        weight_map: dict[str, str] = {}
        total_size = 0
        if index_path.is_file():
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map.update(index_payload.get("weight_map", {}))
            total_size = int(index_payload.get("metadata", {}).get("total_size", 0) or 0)
        else:
            base_safetensors = list(base_dir.glob("*.safetensors"))
            if len(base_safetensors) != 1:
                raise RuntimeError("当前模型目录没有标准 safetensors index，且不是单文件 safetensors，无法增量注入 KV scale")
            only_file = base_safetensors[0]
            if _safe_open is None:
                raise RuntimeError("需要 safetensors 才能读取原始权重索引")
            with _safe_open(str(only_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = only_file.name
            total_size = only_file.stat().st_size
        for tensor_name in extra_tensor_names:
            weight_map[tensor_name] = extra_filename
        extra_size = (dst_dir / extra_filename).stat().st_size
        new_index = {
            "metadata": {"total_size": total_size + extra_size},
            "weight_map": weight_map,
        }
        (dst_dir / "model.safetensors.index.json").write_text(
            json.dumps(new_index, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _patch_model_config_for_kv_cache(
        self,
        *,
        dst_dir: Path,
        model_path: Path,
        config: dict[str, object],
        has_scales: bool,
    ):
        config_path = dst_dir / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        quant_cfg = payload.get("quantization_config")
        if not isinstance(quant_cfg, dict):
            raise RuntimeError("模型 config.json 缺少 quantization_config，无法注入 kv_cache_scheme")
        kv_cache_scheme = quant_cfg.get("kv_cache_scheme")
        if not isinstance(kv_cache_scheme, dict):
            kv_cache_scheme = {
                "num_bits": 8,
                "type": "float",
                "symmetric": True,
                "strategy": "tensor",
                "dynamic": False,
                "observer": "static_minmax",
                "observer_kwargs": {},
            }
        scale_dtype = str(config.get("kv_cache_dtype", "") or "").lower()
        if scale_dtype in {"fp8_e4m3", "fp8"}:
            kv_cache_scheme["type"] = "float"
            kv_cache_scheme["num_bits"] = 8
        elif scale_dtype == "fp8_e5m2":
            kv_cache_scheme["type"] = "float"
            kv_cache_scheme["num_bits"] = 8
        kv_cache_scheme["symmetric"] = True
        kv_cache_scheme["dynamic"] = False
        quant_cfg["kv_cache_scheme"] = kv_cache_scheme
        payload["quantization_config"] = quant_cfg
        payload["aicas_fp8_kv_cache"] = {
            "enabled": True,
            "materialized_from": str(model_path),
            "quantization_param_path": str(config.get("quantization_param_path", "")),
            "has_scales": bool(has_scales),
        }
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def _filter_vllm_llm_kwargs(self, llm_cls, kwargs: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        try:
            signature = inspect.signature(llm_cls.__init__)
        except (TypeError, ValueError):
            return kwargs
        parameters = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return kwargs
        supported = {name for name in parameters.keys() if name != "self"}
        filtered = {key: value for key, value in kwargs.items() if key in supported}
        dropped = [key for key in kwargs.keys() if key not in supported]
        if not dropped:
            return filtered
        explicit_fp8_request = (
            str(config.get("kv_cache_dtype", "")).startswith("fp8")
            or bool(config.get("quantization_param_path"))
            or config.get("calculate_kv_scales") is not None
        )
        kv_related = {"kv_cache_dtype", "calculate_kv_scales", "quantization_param_path"}
        if explicit_fp8_request and any(key in kv_related for key in dropped):
            raise RuntimeError(
                "当前 vLLM 版本不支持所请求的 FP8 KV cache 参数: "
                + ", ".join(key for key in dropped if key in kv_related)
            )
        requested_attention_related = {
            "attention_config": bool(config.get("attention_backend") or config.get("flash_attn_version") is not None),
            "mm_encoder_attn_backend": bool(config.get("mm_encoder_attn_backend")),
            "override_attention_dtype": bool(config.get("override_attention_dtype")),
            "skip_mm_profiling": bool(config.get("skip_mm_profiling")),
        }
        unsupported_requested = [key for key, requested in requested_attention_related.items() if requested and key in dropped]
        if unsupported_requested:
            config["unsupported_requested_vllm_features"] = unsupported_requested
        if dropped:
            config["ignored_llm_kwargs"] = dropped
        return filtered

    def _hash_visual_image_input(self, image_obj):
        if _PILImage is not None and isinstance(image_obj, _PILImage.Image):
            try:
                work = image_obj.convert("RGB") if image_obj.mode != "RGB" else image_obj
                payload = work.tobytes()
                meta = f"{work.size}|{work.mode}".encode("utf-8")
                if _xxhash is not None:
                    return _xxhash.xxh3_128_hexdigest(meta + payload)
                return hashlib.md5(meta + payload).hexdigest()
            except Exception:
                return None
        return None

    def _normalize_messages_for_vllm(self, messages):
        try:
            message = messages[0]
            content = message["content"]
            patched_content = []
            image_obj = None
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_obj = self._maybe_resize_image(item.get("image"))
                    patched_item = dict(item)
                    patched_item["image"] = image_obj
                    patched_content.append(patched_item)
                else:
                    patched_content.append(item)
            patched_message = dict(message)
            patched_message["content"] = patched_content
            return [patched_message], image_obj
        except Exception:
            return messages, None

    def _extract_single_image(self, messages):
        _, image_obj = self._normalize_messages_for_vllm(messages)
        return image_obj

    def _maybe_resize_image(self, image_obj):
        if _PILImage is None or not isinstance(image_obj, _PILImage.Image):
            return image_obj
        try:
            work = image_obj.convert("RGB") if image_obj.mode != "RGB" else image_obj
            width, height = work.size
            if width <= 0 or height <= 0:
                return work
            # 贴近高分版策略：尽量少做侵入式 resize，仅限制极端大图。
            max_side = int(os.environ.get("AICAS_AWQ_VLLM_RESIZE_MAX_SIDE", 768) or 768)
            max_pixels = int(os.environ.get("AICAS_AWQ_VLLM_RESIZE_MAX_PIXELS", 768 * 768) or (768 * 768))
            scale = min(max_side / float(max(width, height)), (max_pixels / float(width * height)) ** 0.5, 1.0)
            if scale >= 0.999:
                return work
            new_w = max(28, int(round(width * scale)))
            new_h = max(28, int(round(height * scale)))
            resample = getattr(getattr(_PILImage, "Resampling", _PILImage), "LANCZOS", _PILImage.BICUBIC)
            return work.resize((new_w, new_h), resample=resample)
        except Exception:
            return image_obj

    def _generate_with_vllm_only(self, *args, **kwargs):
        request_payload = {key: kwargs.pop(key, None) for key in _VLLM_REQUEST_KWARGS}
        prompt = request_payload.get(_VLLM_PROMPT_KWARG)
        image_obj = request_payload.get(_VLLM_IMAGE_KWARG)
        image_hash = request_payload.get(_VLLM_IMAGE_HASH_KWARG)
        if not prompt or image_obj is None:
            raise RuntimeError("AWQ+vLLM 路径缺少 prompt 或 image")

        sampling_kwargs = {
            "n": 1,
            "max_tokens": int(kwargs.get("max_new_tokens", 128) or 128),
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": float(os.environ.get("AICAS_REPETITION_PENALTY", "1.1")),
            "skip_special_tokens": False,
        }
        if self._eos_token_id is not None:
            sampling_kwargs["stop_token_ids"] = [int(self._eos_token_id)]
        sampling_params = self._vllm_sampling_params_cls(**sampling_kwargs)

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        def _make_inputs(local_image, local_hash):
            llm_inputs = {"prompt": prompt}
            if local_image is not None:
                llm_inputs["multi_modal_data"] = {"image": local_image}
            if local_hash:
                llm_inputs["multi_modal_uuids"] = {"image": local_hash}
            return llm_inputs

        llm_inputs = _make_inputs(image_obj, image_hash)
        outputs = self._vllm_llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)
        if not outputs:
            raise RuntimeError("vLLM 未返回输出")
        request_output = outputs[0]
        candidate = request_output.outputs[0] if getattr(request_output, "outputs", None) else request_output
        token_ids = list(getattr(candidate, "token_ids", None) or [])
        if not token_ids:
            text = str(getattr(candidate, "text", "") or "")
            token_ids = self._processor.tokenizer.encode(text, add_special_tokens=False) if text else []

        if input_ids is None:
            raise RuntimeError("缺少 input_ids")
        prompt_ids = input_ids[0]
        gen_ids = torch.tensor(token_ids, dtype=prompt_ids.dtype, device=prompt_ids.device)
        return torch.cat((prompt_ids, gen_ids), dim=0).unsqueeze(0)

    @property
    def processor(self):
        return self._processor_wrapper

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def source_config(self):
        return self._source_config

    def get_benchmark_metadata(self):
        return dict(self._source_config)

    def generate(self, image, question: str, max_new_tokens: int = 128):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            use_cache=True,
        )
        token_ids = list(out[0][inputs.input_ids.shape[1]:].tolist())
        text = self.processor.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"text": text, "token_count": len(token_ids)}
