import os
import sys
import importlib.util
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_SOURCE = _ROOT / "prefill_text_runtime.cpp"
_BUILD_DIR = _ROOT / ".build"
_MODULE_NAME = "aicas_prefill_text_runtime"
_MODE_ENV = "AICAS_CPP_RUNTIME_FORCE_REBUILD"


def _force_rebuild_enabled() -> bool:
    raw = os.environ.get(_MODE_ENV, "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{_MODE_ENV} must be '0' or '1', got {raw!r}")
    return raw == "1"


def _prebuilt_so_path() -> Path:
    return _BUILD_DIR / f"{_MODULE_NAME}.so"


def _load_prebuilt_module(verbose: bool):
    so_path = _prebuilt_so_path()
    if not so_path.exists():
        raise RuntimeError(
            f"Prebuilt runtime not found: {so_path}. "
            f"Set {_MODE_ENV}=1 once to force rebuild."
        )
    if verbose:
        print(f"[cpp_runtime.loader] loading prebuilt module: {so_path}")

    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _remove_stale_build_outputs():
    # Remove target artifacts so mode=1 always rebuilds from source.
    so_path = _prebuilt_so_path()
    if so_path.exists():
        so_path.unlink()
    obj_path = _BUILD_DIR / "prefill_text_runtime.o"
    if obj_path.exists():
        obj_path.unlink()


def _build_from_source(verbose: bool):
    _remove_stale_build_outputs()
    extra_cflags = ["-O3", "-std=c++17"]
    if os.environ.get("AICAS_CPP_RUNTIME_DEBUG", "0") == "1":
        extra_cflags.extend(["-g", "-O0"])
    if verbose:
        print(f"[cpp_runtime.loader] rebuilding from source: {_SOURCE}")
    return load(
        name=_MODULE_NAME,
        sources=[str(_SOURCE)],
        extra_cflags=extra_cflags,
        build_directory=str(_BUILD_DIR),
        verbose=verbose,
    )


@lru_cache(maxsize=1)
def load_prefill_text_runtime_module(verbose: bool = False):
    """Load prebuilt runtime or rebuild from source based on AICAS_CPP_RUNTIME_FORCE_REBUILD."""
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not _force_rebuild_enabled():
        so_path = _prebuilt_so_path()
        if so_path.exists():
            return _load_prebuilt_module(verbose=verbose)
        if verbose:
            print(f"[cpp_runtime.loader] prebuilt module missing, building once: {so_path}")
        return _build_from_source(verbose=verbose)

    return _build_from_source(verbose=verbose)
