"""
[SHLEE] Load prebuilt CUDA extensions (.so) from my_kernel/prebuilt/.
Falls back to torch.utils.cpp_extension.load_inline if the prebuilt
binary is missing or ABI-incompatible.
"""

import importlib
import importlib.util
import os
import sys

_PREBUILT_DIR = os.path.join(os.path.dirname(__file__), "prebuilt")

def load_prebuilt(name: str):
    """Try to load my_kernel/prebuilt/<name>.so.  Returns module or None."""
    so_path = os.path.join(_PREBUILT_DIR, f"{name}.so")
    if not os.path.isfile(so_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, so_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        sys.modules.pop(name, None)
        return None
