import os
import importlib.util
import logging
import shutil
import torch

logger = logging.getLogger(__name__)

_async_ops = None


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
            print(f"[int8_async] Installed rebuilt {os.path.basename(target_path)}")
    except Exception as e:
        print(f"[int8_async] Failed to install rebuilt artifact: {e}")


def get_int8_async_ops():
    global _async_ops
    if _async_ops is not None:
        return _async_ops if _async_ops is not False else None

    if not torch.cuda.is_available():
        _async_ops = False
        return None

    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    cu_file = os.path.join(cuda_dir, "int8_async_gemv.cu")
    if not os.path.exists(cu_file):
        _async_ops = False
        return None

    major, minor = torch.cuda.get_device_capability()
    so_name = f"int8_async_gemv_sm{major}{minor}.so"
    build_dir = os.path.join(cuda_dir, "build")
    so_path = os.path.join(build_dir, so_name)

    if os.environ.get("AICAS_SKIP_PREBUILT", "0") == "1":
        print(f"[int8_async] Skipping prebuilt {so_name}")
    elif os.path.exists(so_path) and _prebuilt_is_fresh(so_path, cu_file):
        try:
            spec = importlib.util.spec_from_file_location("int8_async_gemv", so_path)
            _async_ops = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_async_ops)
            print(f"[int8_async] Loaded prebuilt {so_name}")
            return _async_ops
        except Exception as e:
            print(f"[int8_async] Prebuilt {so_name} load failed: {e}")
    elif os.path.exists(so_path):
        print(f"[int8_async] Prebuilt {so_name} is stale; rebuilding")
    else:
        print(f"[int8_async] Prebuilt {so_name} not found")

    from torch.utils.cpp_extension import load

    arch = f"{major}.{minor}"
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    try:
        from custom_kernels.cuda import get_ppu_compile_flags
        logger.info("[int8_async] JIT compiling int8_async_gemv...")
        _async_ops = load(
            name="int8_async_gemv",
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
        _install_built_artifact(_async_ops, so_path)
        return _async_ops
    except Exception as e:
        logger.warning(f"[int8_async] JIT compile failed: {e}")
        _async_ops = False
        return None
