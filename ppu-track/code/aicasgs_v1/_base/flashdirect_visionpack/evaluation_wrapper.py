from __future__ import annotations

import importlib.util
from pathlib import Path


_FLASH_WRAPPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "flashdirect"
    / "evaluation_wrapper.py"
)
_VISIONPACK_WRAPPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "visionpack"
    / "evaluation_wrapper.py"
)


def _load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FLASH_MODULE = _load_module("aicas_prefill_flashdirect_visionpack_flash_20260417", _FLASH_WRAPPER_PATH)
_VISIONPACK_MODULE = _load_module("aicas_prefill_flashdirect_visionpack_pack_20260417", _VISIONPACK_WRAPPER_PATH)


class VLMModel(_FLASH_MODULE.VLMModel):
    _get_triton_vision_qkv_rotary_pack = _VISIONPACK_MODULE.VLMModel._get_triton_vision_qkv_rotary_pack
    _run_vision_qkv_rotary_pack = _VISIONPACK_MODULE.VLMModel._run_vision_qkv_rotary_pack
    _enable_fused_vision_qkv_rotary_attention = _VISIONPACK_MODULE.VLMModel._enable_fused_vision_qkv_rotary_attention

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vision_pack_warned_modules = set()
        self._enable_fused_vision_qkv_rotary_attention()
