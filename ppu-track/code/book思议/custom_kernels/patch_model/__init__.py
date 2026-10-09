"""给Qwen3-VL-2B做monkey-patch 挂上融合算子 — 模块拆分后的入口。"""

from .orchestrator import patch_model_for_graph  # noqa: F401

__all__ = ["patch_model_for_graph"]
