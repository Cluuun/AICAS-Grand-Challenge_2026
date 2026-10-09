import os
import importlib.util
import logging
import shutil
import torch

logger = logging.getLogger(__name__)

_fd_ops = None


def _prebuilt_is_fresh(so_path, src_path):
    if os.environ.get("AICAS_CHECK_PREBUILT_FRESH", "0") != "1":
        return True
    try:
        return os.path.getmtime(so_path) >= os.path.getmtime(src_path)
    except OSError:
        return False


def _install_built_artifact(module, target_path):
    try:
        src_path = getattr(module, "__file__", None)
        if src_path and os.path.exists(src_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(src_path, target_path)
            print(f"[flash_decode] Installed rebuilt {os.path.basename(target_path)}")
    except Exception as e:
        print(f"[flash_decode] Failed to install rebuilt artifact: {e}")


def get_flash_decode_ops():
    global _fd_ops
    if _fd_ops is not None:
        return _fd_ops if _fd_ops is not False else None

    if not torch.cuda.is_available():
        _fd_ops = False
        return None

    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    cu_file = os.path.join(cuda_dir, "flash_decode_attn.cu")
    if not os.path.exists(cu_file):
        _fd_ops = False
        return None

    # 1. Try prebuilt .so
    major, minor = torch.cuda.get_device_capability()
    so_name = f"flash_decode_attn_sm{major}{minor}.so"
    build_dir = os.path.join(cuda_dir, "build")
    so_path = os.path.join(build_dir, so_name)

    if os.environ.get("AICAS_SKIP_PREBUILT", "0") == "1":
        print(f"[flash_decode] Skipping prebuilt {so_name} by AICAS_SKIP_PREBUILT=1")
    elif os.path.exists(so_path) and _prebuilt_is_fresh(so_path, cu_file):
        try:
            spec = importlib.util.spec_from_file_location("flash_decode_attn", so_path)
            _fd_ops = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_fd_ops)
            print(f"[flash_decode] Loaded prebuilt {so_name}")
            return _fd_ops
        except Exception as e:
            print(f"[flash_decode] Prebuilt {so_name} load failed: {e}")
    elif os.path.exists(so_path):
        print(f"[flash_decode] Prebuilt {so_name} is stale; rebuilding from source")
    else:
        print(f"[flash_decode] Prebuilt {so_name} not found at {so_path}")

    # 2. Fallback: JIT compile
    from torch.utils.cpp_extension import load

    arch = f"{major}.{minor}"
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    try:
        from custom_kernels.cuda import get_ppu_compile_flags
        logger.info("[flash_decode] JIT compiling flash_decode_attn...")
        _fd_ops = load(
            name="flash_decode_attn",
            sources=[cu_file],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math",
                *get_ppu_compile_flags(),
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
            ],
            extra_cflags=["-O3"],
            verbose=False,
        )
        _install_built_artifact(_fd_ops, so_path)
        return _fd_ops
    except Exception:
        _fd_ops = False
        return None
