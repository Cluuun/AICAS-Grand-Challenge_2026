import os
import importlib.util
import logging
import torch

logger = logging.getLogger(__name__)

_cuda_ext = None


def get_ppu_compile_flags():
    """Return CUDA-wrapper flags for Zhenwu PPU builds."""
    ppu_arch = os.environ.get("AICAS_PPU_ARCH", "ppu0015")
    return [f"--ppu-arch={ppu_arch}"]


def _try_load_prebuilt(name_prefix):
    """尝试加载预编译的 .so，按当前 GPU SM 版本匹配"""
    if not torch.cuda.is_available():
        return None
    if os.environ.get("AICAS_SKIP_PREBUILT", "0") == "1":
        print(f"[custom_kernels/cuda] Skipping prebuilt {name_prefix} by AICAS_SKIP_PREBUILT=1")
        return None
    major, minor = torch.cuda.get_device_capability()
    so_name = f"{name_prefix}_sm{major}{minor}.so"
    build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    so_path = os.path.join(build_dir, so_name)
    if os.path.exists(so_path):
        try:
            spec = importlib.util.spec_from_file_location(name_prefix, so_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[custom_kernels/cuda] Loaded prebuilt {so_name}")
            return mod
        except Exception as e:
            print(f"[custom_kernels/cuda] Prebuilt {so_name} load failed: {e}")
    else:
        print(f"[custom_kernels/cuda] Prebuilt {so_name} not found at {so_path}")
    return None


def _jit_compile(name, cu_file):
    """JIT 编译 CUDA kernel（预编译 .so 不存在时的 fallback）"""
    from torch.utils.cpp_extension import load

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}"
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    logger.info(f"[custom_kernels/cuda] JIT compiling {name}...")
    return load(
        name=name,
        sources=[cu_file],
        extra_cuda_cflags=[
            "-O3",
            *get_ppu_compile_flags(),
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )


def get_cuda_ops():
    global _cuda_ext
    if _cuda_ext is not None:
        return _cuda_ext if _cuda_ext is not False else None

    if not torch.cuda.is_available():
        _cuda_ext = False
        return None

    # 1. 尝试加载预编译 .so
    mod = _try_load_prebuilt("custom_fused_kernels")
    if mod is not None:
        _cuda_ext = mod
        return mod

    # 2. Fallback: JIT 编译
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_file = os.path.join(cuda_dir, "fused_kernels.cu")
    cu_file = fallback_file

    if cu_file is None or not os.path.exists(cu_file):
        logger.warning("[custom_kernels/cuda] No .cu file found")
        _cuda_ext = False
        return None

    try:
        _cuda_ext = _jit_compile("custom_fused_kernels", cu_file)
        logger.info("[custom_kernels/cuda] JIT compilation OK")
        return _cuda_ext
    except Exception as e:
        logger.warning(f"[custom_kernels/cuda] JIT compilation failed: {e}")
        _cuda_ext = False
        return None
