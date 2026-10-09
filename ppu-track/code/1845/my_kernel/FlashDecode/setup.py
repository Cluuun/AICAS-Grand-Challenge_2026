from setuptools import setup
import os
import re
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _gencode_flags_for_arch(arch: str) -> list[str]:
    return [
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        f"-gencode=arch=compute_{arch},code=compute_{arch}",
    ]


def _parse_torch_cuda_arch_list(value: str) -> set[str]:
    arches: set[str] = set()
    for token in value.split(";"):
        token = token.strip()
        if not token:
            continue
        token = token.split("+", 1)[0]
        m = re.fullmatch(r"(\d+)\.(\d+)", token)
        if m:
            arches.add(f"{m.group(1)}{m.group(2)}")
            continue
        m = re.fullmatch(r"(\d+)", token)
        if m:
            arches.add(m.group(1))
            continue
    return arches


nvcc_args = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
]

arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
if arch_env:
    target_arches = _parse_torch_cuda_arch_list(arch_env)
else:
    target_arches = {"80"}

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    target_arches.add(f"{major}{minor}")

for arch in sorted(target_arches):
    nvcc_args.extend(_gencode_flags_for_arch(arch))

ext_modules = [
    CUDAExtension(
        name="flashdecode_ext",
        sources=["flashdecode_ext.cpp", "flashdecode_ext_kernel.cu", "flashdecode-mma.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_args,
        },
    )
]

setup(
    name="flashdecode_ext",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
