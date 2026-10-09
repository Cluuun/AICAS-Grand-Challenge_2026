from __future__ import annotations

import importlib.util
from pathlib import Path


_VLLM_MINIMAL_PATH = Path(__file__).resolve().parent / "evaluation_wrapper_vllm_minimal.py"


def _load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VLLM_MINIMAL_MODULE = _load_module("aicas_awq_vllm_minimal_base", _VLLM_MINIMAL_PATH)
_MinimalVLLMVLMModel = _VLLM_MINIMAL_MODULE._MinimalVLLMVLMModel


class VLMModel(_MinimalVLLMVLMModel):
    """AWQ + vLLM 薄封装入口，直接复用独立的 vLLM minimal base。"""

    def __init__(self, model_path: str, device: str | None = "cuda"):
        super().__init__(model_path=model_path, device=device)
        if isinstance(getattr(self, "_source_config", None), dict):
            self._source_config["backend"] = "awq_vllm"
            self._source_config["awq_vllm_route"] = True
            bridge = self._source_config.setdefault("vllm_bridge", {})
            if isinstance(bridge, dict):
                bridge["awq_vllm_route"] = True
                bridge["mode"] = "awq_vllm_reuse_minimal_base"
