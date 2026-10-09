"""竞赛入口文件。

竞赛门控检测此文件是否存在，benchmark.py 通过 `from evaluation_wrapper import VLMModel` 导入。
"""

from _eval_impl import VLMModel  # noqa: F401

__all__ = ["VLMModel"]
