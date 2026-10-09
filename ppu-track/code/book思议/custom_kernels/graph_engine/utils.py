"""调试工具、环境上下文管理、通用辅助函数。"""

import os

from custom_kernels import debug as profile_debug

# 调试状态
_DEBUG_COUNTS = {}
_DEBUG_BUFFER = []


def _flush_debug_buffer():
    if not _DEBUG_BUFFER:
        return
    print(f"[AICAS_DEBUG][generate_flush] buffered_events={len(_DEBUG_BUFFER)}", flush=True)
    for line in _DEBUG_BUFFER:
        print(line, flush=True)


def _debug_enabled():
    return os.environ.get("AICAS_DEBUG", "0") != "0"


def _debug_sync(device=None):
    profile_debug.debug_sync(device)


def _debug_print(tag, message, limit_env="AICAS_DEBUG_GENERATE_N", default_limit=1000):
    if not _debug_enabled():
        return
    limit = int(os.environ.get(limit_env, str(default_limit)))
    count = _DEBUG_COUNTS.get(tag, 0)
    if limit >= 0 and count >= limit:
        return
    _DEBUG_COUNTS[tag] = count + 1
    _DEBUG_BUFFER.append(f"[AICAS_DEBUG][{tag}#{count + 1}] {message}")


def _shape(x):
    if x is None:
        return None
    try:
        return tuple(x.shape)
    except Exception:
        return str(type(x))


class _temporary_env:
    """临时环境变量上下文管理器，用于 CUDA graph capture 时切换解码策略。"""

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.old = None

    def __enter__(self):
        self.old = os.environ.get(self.name)
        if self.value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = str(self.value)

    def __exit__(self, exc_type, exc, tb):
        if self.old is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.old


def _prefill_input_cache_key(input_ids, pixel_values, image_grid_thw):
    return (
        int(input_ids.data_ptr()),
        tuple(input_ids.shape),
        int(pixel_values.data_ptr()),
        tuple(pixel_values.shape),
        tuple(int(x) for x in image_grid_thw.reshape(-1).tolist()),
    )
