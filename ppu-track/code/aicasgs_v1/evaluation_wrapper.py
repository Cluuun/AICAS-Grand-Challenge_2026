from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parent
_LATEST_PATH = _ROOT / "evaluation_wrapper_latest.py"
_AWQ_VLLM_PATH = _ROOT / "_base" / "evaluation_wrapper_awq_vllm.py"
_VLLM_MINIMAL_PATH = _ROOT / "_base" / "evaluation_wrapper_vllm_minimal.py"
_COMMON_MODEL_DIRS = (
    Path("/root/model"),
    Path("/mnt/workspace/model"),
    Path("/mnt/workspace"),
)
_VLLM_PATCHED_MODEL_CLS = None
_DEFAULT_MIXED_W8A8_MODEL_NAME = "Qwen3-VL-2B-Instruct-W8A8-MIX-26_27"
_DEFAULT_MIXED_W8A8_RECIPE_PATH = Path("/root/AICASGC/qwen3_vl_2b_mix_recipe.yaml")


def _load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_text(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw)


def _resolve_model_path(model_path: str) -> str:
    raw_path = str(model_path or "").strip()
    if not raw_path:
        return raw_path

    requested = Path(raw_path).expanduser()
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add_candidate(path: Path | None):
        if path is None:
            return
        resolved = path.expanduser().resolve(strict=False)
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidates.append(resolved)

    env_override = os.environ.get("AICAS_MODEL_PATH", "").strip()
    if env_override:
        _add_candidate(Path(env_override))

    if requested.is_absolute():
        _add_candidate(requested)
    else:
        _add_candidate(Path.cwd() / requested)
        _add_candidate(_ROOT / requested)
        for base_dir in _COMMON_MODEL_DIRS:
            _add_candidate(base_dir / requested.name)

    _add_candidate(requested)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return str(candidates[0]) if candidates else raw_path


def _resolve_default_mixed_quant_model_path() -> str:
    for base_dir in _COMMON_MODEL_DIRS:
        candidate = (base_dir / _DEFAULT_MIXED_W8A8_MODEL_NAME).expanduser().resolve(strict=False)
        if candidate.exists():
            return str(candidate)
    return str((_COMMON_MODEL_DIRS[0] / _DEFAULT_MIXED_W8A8_MODEL_NAME).resolve(strict=False))


def _materialize_default_mixed_quant_model_if_needed(model_path: str) -> str:
    target_path = Path(_resolve_default_mixed_quant_model_path()).expanduser().resolve(strict=False)
    if target_path.exists():
        return str(target_path)

    if not _env_flag("AICAS_AUTO_BUILD_DEFAULT_PPU_W8A8_MIX", True):
        return str(target_path)

    model_dir = Path(_resolve_model_path(model_path)).expanduser().resolve(strict=False)
    if not model_dir.exists():
        return str(target_path)

    recipe_path = _DEFAULT_MIXED_W8A8_RECIPE_PATH
    if not recipe_path.is_file():
        return str(target_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "qlean",
        "--model_name",
        "Qwen/Qwen3-VL-2B-Instruct",
        "--model_path",
        str(model_dir),
        "--save_path",
        str(target_path),
        "--recipe",
        str(recipe_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL if not _env_flag("AICAS_DEBUG_AUTO_QLEAN", False) else None,
            stderr=subprocess.DEVNULL if not _env_flag("AICAS_DEBUG_AUTO_QLEAN", False) else None,
            timeout=int(os.environ.get("AICAS_AUTO_QLEAN_TIMEOUT_SEC", "3600") or "3600"),
        )
    except Exception:
        return str(target_path)
    return str(target_path)


def _is_ppu_runtime() -> bool:
    if _env_flag("AICAS_FORCE_PPU_RUNTIME", False):
        return True
    for env_name in ("PPU_SDK", "PPU_HOME", "PPU_PATH", "PPU_VERSION", "USE_ALIBABACLOUD_GPU"):
        if os.environ.get(env_name):
            return True
    return False


def _allow_awq_vllm_on_ppu() -> bool:
    return _env_flag("AICAS_PPU_ALLOW_AWQ_VLLM", False) or _env_flag("AICAS_FORCE_AWQ_VLLM", False)


def _read_json_dict(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _normalize_quantization_name(raw_value: str) -> str:
    value = str(raw_value or "").strip().lower()
    if value in {"", "auto"}:
        return value
    if value in {"none", "off", "disabled", "false", "no"}:
        return ""
    if value in {"compressed_tensors", "compressed-tensors", "compressed"}:
        return "compressed-tensors"
    if value in {"w8a8_int8", "w8a8", "a8w8", "int8"}:
        return "compressed-tensors"
    return value


def _extract_bits_from_compression_config(compression_cfg: dict[str, Any]) -> tuple[int | None, int | None]:
    config_groups = compression_cfg.get("config_groups")
    if not isinstance(config_groups, dict):
        return None, None
    for group_cfg in config_groups.values():
        if not isinstance(group_cfg, dict):
            continue
        weights = group_cfg.get("weights")
        input_acts = group_cfg.get("input_activations")
        weight_bits = weights.get("num_bits") if isinstance(weights, dict) else None
        input_bits = input_acts.get("num_bits") if isinstance(input_acts, dict) else None
        if weight_bits is not None or input_bits is not None:
            return weight_bits, input_bits
    return None, None


def _detect_quant_model_metadata(model_path: str) -> dict[str, Any]:
    resolved_path = _resolve_model_path(model_path)
    model_dir = Path(resolved_path).expanduser()
    path_lower = resolved_path.lower()
    metadata: dict[str, Any] = {
        "model_path": resolved_path,
        "is_quantized": False,
        "family": "",
        "preferred_vllm_quantization": "",
        "config_source": "",
        "path_hint": "",
        "weight_bits": None,
        "activation_bits": None,
    }

    def _mark(*, family: str, quantization: str, source: str, weight_bits=None, activation_bits=None):
        metadata["is_quantized"] = True
        metadata["family"] = family
        metadata["preferred_vllm_quantization"] = quantization
        metadata["config_source"] = source
        metadata["weight_bits"] = weight_bits
        metadata["activation_bits"] = activation_bits

    config_payload = None
    for filename in ("config.json", "configuration.json"):
        config_payload = _read_json_dict(model_dir / filename)
        if config_payload is not None:
            break

    if isinstance(config_payload, dict):
        compression_cfg = config_payload.get("compression_config")
        if isinstance(compression_cfg, dict):
            weight_bits, activation_bits = _extract_bits_from_compression_config(compression_cfg)
            family = "compressed-tensors"
            if weight_bits == 8 and activation_bits == 8:
                family = "w8a8_int8"
            elif weight_bits == 4 and activation_bits == 8:
                family = "w4a8"
            elif weight_bits == 4 and activation_bits in {None, 16}:
                family = "w4a16"
            _mark(
                family=family,
                quantization="compressed-tensors",
                source="config.compression_config",
                weight_bits=weight_bits,
                activation_bits=activation_bits,
            )
            return metadata

        quant_cfg = config_payload.get("quantization_config")
        if isinstance(quant_cfg, dict):
            quant_method = _normalize_quantization_name(quant_cfg.get("quant_method", ""))
            if quant_method in {"awq", "gptq", "compressed-tensors", "fp8", "awq_acext", "gptq_acext"}:
                _mark(
                    family=quant_method,
                    quantization=quant_method,
                    source="config.quantization_config",
                    weight_bits=quant_cfg.get("bits"),
                )
                return metadata

    quantize_config = _read_json_dict(model_dir / "quantize_config.json")
    if isinstance(quantize_config, dict):
        quant_method = _normalize_quantization_name(quantize_config.get("quant_method", ""))
        bits = quantize_config.get("bits") or quantize_config.get("w_bit")
        if quant_method == "awq" or (
            bits == 4 and "zero_point" in quantize_config and "group_size" in quantize_config
        ):
            _mark(
                family="awq",
                quantization="awq",
                source="quantize_config.json",
                weight_bits=bits,
            )
            return metadata
        if quant_method == "gptq" or (
            bits is not None and "desc_act" in quantize_config and "group_size" in quantize_config
        ):
            _mark(
                family="gptq",
                quantization="gptq",
                source="quantize_config.json",
                weight_bits=bits,
            )
            return metadata

    if "gptq" in path_lower:
        metadata["path_hint"] = "gptq"
        _mark(family="gptq", quantization="gptq", source="path_hint:gptq")
        return metadata
    if any(tag in path_lower for tag in ("awq", "w4a16", "compressed")):
        metadata["path_hint"] = "awq_or_compressed"
        # 仓库历史 AWQ 主路径是 compressed-tensors 产物，因此 path hint 优先映射到 compressed-tensors。
        _mark(family="awq_or_compressed", quantization="compressed-tensors", source="path_hint:awq_or_compressed")
        return metadata
    if any(tag in path_lower for tag in ("w8a8", "a8w8", "int8")):
        metadata["path_hint"] = "w8a8_int8"
        _mark(family="w8a8_int8", quantization="compressed-tensors", source="path_hint:w8a8_int8")
        return metadata

    return metadata


def _resolve_requested_quantization_method() -> tuple[bool, str]:
    for env_name in ("AICAS_VLLM_QUANTIZATION", "AICAS_VLLM_PREQUANT_METHOD"):
        raw_value = _env_text(env_name, "").strip()
        if not raw_value:
            continue
        normalized = _normalize_quantization_name(raw_value)
        if normalized == "auto":
            return True, "auto"
        return True, normalized
    return False, ""


def _select_vllm_quantization_method(model_path: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or _detect_quant_model_metadata(model_path)
    explicit, requested = _resolve_requested_quantization_method()
    if explicit:
        if not requested:
            return ""
        if requested != "auto":
            return requested

    detected = str(metadata.get("preferred_vllm_quantization", "") or "")
    if not detected:
        return ""
    config_source = str(metadata.get("config_source", "") or "")
    if config_source.startswith("path_hint:") and not _env_flag("AICAS_VLLM_FORCE_PATH_HINT_QUANTIZATION", False):
        return ""

    if _env_flag("AICAS_VLLM_PREFER_PPU_ACEXT", False):
        if detected == "awq":
            return "awq_acext"
        if detected == "gptq":
            return "gptq_acext"
    return detected


def _resolve_vllm_load_format(quantization_method: str) -> str:
    explicit = _env_text("AICAS_VLLM_LOAD_FORMAT", "").strip()
    if explicit:
        return explicit
    if quantization_method == "compressed-tensors":
        return "auto"
    return ""


def _sanitize_quant_config_for_local_runtime(model_path: str, metadata: dict[str, Any] | None = None) -> None:
    metadata = metadata or _detect_quant_model_metadata(model_path)
    quant_method = str(metadata.get("preferred_vllm_quantization", "") or "")
    config_source = str(metadata.get("config_source", "") or "")
    if quant_method != "compressed-tensors":
        return
    if not config_source.startswith("config."):
        return
    config_path = Path(_resolve_model_path(model_path)) / "config.json"
    payload = _read_json_dict(config_path)
    if not isinstance(payload, dict):
        return
    quant_cfg = payload.get("quantization_config")
    if not isinstance(quant_cfg, dict):
        return
    changed = False
    config_groups = quant_cfg.get("config_groups")
    if isinstance(config_groups, dict):
        for group_cfg in config_groups.values():
            if not isinstance(group_cfg, dict):
                continue
            weights = group_cfg.get("weights")
            if not isinstance(weights, dict):
                continue
            for legacy_key in ("scale_dtype", "zp_dtype"):
                if legacy_key in weights:
                    weights.pop(legacy_key, None)
                    changed = True
    if changed:
        payload["quantization_config"] = quant_cfg
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _should_use_vllm_route(model_path: str, metadata: dict[str, Any] | None = None) -> bool:
    metadata = metadata or _detect_quant_model_metadata(model_path)
    explicit_quant, requested_quant = _resolve_requested_quantization_method()
    if explicit_quant and requested_quant != "":
        return True
    if bool(metadata.get("is_quantized", False)):
        return True
    if _env_flag("AICAS_USE_MINIMAL_VLLM_WRAPPER", False) or _env_flag("AICAS_ENABLE_VLLM_BRIDGE", False):
        return True
    return False


def _should_redirect_to_default_mixed_quant(model_path: str, metadata: dict[str, Any] | None = None) -> bool:
    if not _env_flag("AICAS_USE_DEFAULT_PPU_W8A8_MIX", True):
        return False
    explicit_quant, requested_quant = _resolve_requested_quantization_method()
    if explicit_quant and requested_quant == "":
        return False
    if explicit_quant and requested_quant not in {"auto", "compressed-tensors"}:
        return False
    metadata = metadata or _detect_quant_model_metadata(model_path)
    if bool(metadata.get("is_quantized", False)):
        return False
    model_name = Path(str(model_path)).name
    return model_name == "Qwen3-VL-2B-Instruct"


def _install_w8a8_quant_pth_hook():
    """在 site-packages 创建 .pth + 补丁模块。
    EngineCore 子进程加载模型后，对所有 2D 线性权重执行 per-channel W8A8 量化（int8），
    然后立即反量化回 bf16，从而在保持 W8A8 量化精度的同时利用 bf16 Tensor Core 加速。
    """
    import sys as _sys
    import site as _site_module
    _flag = os.environ.get("AICAS_VLLM_DEQUANT_W8A8", "1").strip()
    if _flag not in {"1", "true", "yes"}:
        return
    try:
        _site_dirs = _site_module.getsitepackages()
    except Exception:
        _site_dirs = [p for p in _sys.path if 'site-packages' in p]
    if not _site_dirs:
        return
    _pth = os.path.join(_site_dirs[0], 'aicas_dequant.pth')
    _mod = os.path.join(_site_dirs[0], '_aicas_dequant_impl.py')
    if os.path.exists(_pth) and os.path.exists(_mod):
        return
    # 写入补丁模块（内嵌代码，无外部依赖）
    with open(_mod, 'w') as _f:
        _f.write('''"""AICAS W8A8 per-channel quant + bf16 Tensor Core 推理补丁"""
import os, builtins
if os.environ.get("AICAS_VLLM_DEQUANT_W8A8", "1").strip() in {"1", "true", "yes"}:
    _PATCHED = _PATCHING = False
    _REAL_IMPORT = builtins.__import__
    def _quantize_model_weights(model):
        """遍历模型所有 2D 线性权重，执行 per-channel W8A8 量化后恢复 bf16。
        量化步骤：absmax → scale = absmax/127 → round(w/scale) → clamp → int8 → *scale → bf16
        推理时使用 bf16 Tensor Core matmul（PPU 上比 int8 GEMM 快 3-4x）。"""
        import torch
        qc = 0
        for _name, _param in model.named_parameters():
            if _param.ndim != 2 or _param.dtype not in (torch.bfloat16, torch.float16):
                continue
            _absmax = _param.data.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
            _scale = _absmax / 127.0
            _w_q = (_param.data / _scale).round().clamp(-128, 127).to(torch.int8)
            _param.data = _w_q.to(_param.dtype) * _scale
            qc += 1
        return qc
    def _try_patch():
        global _PATCHED, _PATCHING
        if _PATCHED or _PATCHING: return True
        _PATCHING = True
        try:
            import logging as _logging
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner
            _orig_load = GPUModelRunner.load_model
            def _patched_load(self, *a, **kw):
                _result = _orig_load(self, *a, **kw)
                _qc = _quantize_model_weights(self.model)
                if _qc:
                    _logging.getLogger("aicas").info(
                        "AICAS: per-channel W8A8 quantized %d linear layers", _qc)
                return _result
            GPUModelRunner.load_model = _patched_load
            _PATCHED = True; return True
        except Exception:
            return False
        finally:
            _PATCHING = False
    def _hi(n, *a, **kw):
        r = _REAL_IMPORT(n, *a, **kw)
        if not _PATCHED and isinstance(n, str) and n.startswith("vllm"): _try_patch()
        return r
    builtins.__import__ = _hi
''')
    # 写入 .pth 引导文件
    with open(_pth, 'w') as _f:
        _f.write('import _aicas_dequant_impl\n')


def _load_patched_vllm_model_cls():
    global _VLLM_PATCHED_MODEL_CLS
    if _VLLM_PATCHED_MODEL_CLS is not None:
        return _VLLM_PATCHED_MODEL_CLS

    _install_w8a8_quant_pth_hook()

    module = _load_vllm_minimal_module()
    base_cls = module._MinimalVLLMVLMModel

    class _PatchedMinimalVLLMVLMModel(base_cls):
        def __init__(self, model_path: str, device: str | None = "cuda"):
            self._aicas_quant_metadata = _detect_quant_model_metadata(model_path)
            self._aicas_requested_quantization = _select_vllm_quantization_method(
                model_path, self._aicas_quant_metadata
            )
            self._aicas_seen_mm_hashes: set[str] = set()
            _sanitize_quant_config_for_local_runtime(model_path, self._aicas_quant_metadata)
            if _is_ppu_runtime():
                os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
                os.environ.setdefault("AICAS_VLLM_OMIT_MEDIA_ON_CACHE_HIT", "0")
                os.environ.setdefault("AICAS_VLLM_DTYPE", "bfloat16")
                os.environ.setdefault("AICAS_VLLM_GPU_MEMORY_UTIL", "0.20")
                os.environ.setdefault("AICAS_VLLM_MAX_MODEL_LEN", "1024")
                os.environ.setdefault("AICAS_VLLM_ENABLE_CHUNKED_PREFILL", "1")
                os.environ.setdefault("AICAS_VLLM_SKIP_MM_PROFILING", "1")
                os.environ.setdefault("AICAS_VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
                os.environ.setdefault("AICAS_VLLM_MM_ENCODER_ATTN_BACKEND", "FLASH_ATTN")
                os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
                os.environ.setdefault("AICAS_VLLM_ENABLE_PREFIX_CACHING", "1")
                os.environ.setdefault("AICAS_VLLM_KV_CACHE_DTYPE", "auto")
            requested_backend = _env_text("AICAS_VLLM_ATTENTION_BACKEND", "").strip()
            if requested_backend:
                os.environ["VLLM_ATTENTION_BACKEND"] = requested_backend.upper()
            super().__init__(model_path=model_path, device=device)
            bridge = self._source_config.setdefault("vllm_bridge", {})
            bridge["resolved_quantization"] = self._aicas_requested_quantization
            bridge["quant_detection"] = dict(self._aicas_quant_metadata)
            bridge["ppu_runtime"] = _is_ppu_runtime()
            bridge["env_attention_backend"] = _env_text("VLLM_ATTENTION_BACKEND", "")

        def _generate_with_vllm_only(self, *args, **kwargs):
            import torch

            request_payload = {key: kwargs.pop(key, None) for key in self._vllm_module_wrapper_request_kwargs}
            prompt = request_payload.get(self._vllm_module_wrapper_prompt_kwarg)
            image_obj = request_payload.get(self._vllm_module_wrapper_image_kwarg)
            image_hash = request_payload.get(self._vllm_module_wrapper_image_hash_kwarg)
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

            omit_media_on_cache_hit = bool(self._vllm_runtime_config.get("omit_media_on_cache_hit", False))
            can_skip_image = bool(image_hash) and omit_media_on_cache_hit and image_hash in self._aicas_seen_mm_hashes

            llm_inputs = {"prompt": prompt}
            if can_skip_image:
                llm_inputs["multi_modal_data"] = {"image": None}
                llm_inputs["multi_modal_uuids"] = {"image": image_hash}
            else:
                llm_inputs["multi_modal_data"] = {"image": image_obj}
                if image_hash:
                    llm_inputs["multi_modal_uuids"] = {"image": image_hash}

            outputs = self._vllm_llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)
            if not outputs:
                raise RuntimeError("vLLM 未返回输出")
            if image_hash:
                self._aicas_seen_mm_hashes.add(image_hash)

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
            }
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
            # 通过环境变量 AICAS_VLLM_COMPILATION_CONFIG 传入 compilation_config dict
            # 例如: {"cudagraph_mode": 2} 强制 FULL 模式
            compilation_opts = _env_text("AICAS_VLLM_COMPILATION_CONFIG", "").strip()
            if compilation_opts:
                try:
                    kwargs["compilation_config"] = json.loads(compilation_opts)
                except json.JSONDecodeError:
                    pass
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

        def _maybe_enable_prequantized_vllm_model(
            self, config: dict[str, object], *, model_path: str
        ) -> dict[str, object]:
            quantization_method = getattr(self, "_aicas_requested_quantization", "") or ""
            if not quantization_method:
                return config
            config = dict(config)
            config["quantization"] = quantization_method
            load_format = _resolve_vllm_load_format(quantization_method)
            if load_format:
                config["load_format"] = load_format
            config["selected_quantization_method"] = quantization_method
            return config

    _PatchedMinimalVLLMVLMModel._vllm_module_wrapper_request_kwargs = tuple(
        getattr(module, "_VLLM_REQUEST_KWARGS")
    )
    _PatchedMinimalVLLMVLMModel._vllm_module_wrapper_prompt_kwarg = getattr(module, "_VLLM_PROMPT_KWARG")
    _PatchedMinimalVLLMVLMModel._vllm_module_wrapper_image_kwarg = getattr(module, "_VLLM_IMAGE_KWARG")
    _PatchedMinimalVLLMVLMModel._vllm_module_wrapper_image_hash_kwarg = getattr(module, "_VLLM_IMAGE_HASH_KWARG")

    _VLLM_PATCHED_MODEL_CLS = _PatchedMinimalVLLMVLMModel
    return _VLLM_PATCHED_MODEL_CLS


def _load_latest_module():
    return _load_module("aicas_latest_router", _LATEST_PATH)


def _load_awq_vllm_module():
    return _load_module("aicas_awq_vllm_router", _AWQ_VLLM_PATH)


def _load_vllm_minimal_module():
    return _load_module("aicas_minimal_vllm_router", _VLLM_MINIMAL_PATH)


def _is_awq_prequantized_model_path(model_path: str) -> bool:
    path_lower = str(model_path or "").lower()
    return any(tag in path_lower for tag in ("awq", "w4a16", "compressed"))


class VLMModel:
    def __new__(cls, model_path: str, device: str | None = "cuda"):
        if cls is not VLMModel:
            return super().__new__(cls)
        resolved_model_path = _resolve_model_path(model_path)
        return _load_patched_vllm_model_cls()(model_path=resolved_model_path, device=device)
