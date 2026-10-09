"""带cuda图解码的自定义generate — 模块拆分后的入口。"""

import atexit

from .utils import _flush_debug_buffer  # noqa: F401

atexit.register(_flush_debug_buffer)

from .generate import custom_generate, make_custom_generate  # noqa: F401, E402

__all__ = ["custom_generate", "make_custom_generate"]
