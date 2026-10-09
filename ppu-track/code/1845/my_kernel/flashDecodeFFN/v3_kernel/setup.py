from setuptools import setup
import os
import re
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _gencode_flags_for_arch(arch: str) -> list[str]:
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}", f"-gencode=arch=compute_{arch},code=compute_{arch}"]


def _parse_torch_cuda_arch_list(value: str) -> set[str]:
    arches = set()
    for token in value.split(';'):
        token = token.strip()
        if not token:
            continue
        token = token.split('+', 1)[0]
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
    "--use_fast_math",
]

arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
target_arches = _parse_torch_cuda_arch_list(arch_env) if arch_env else {"80"}
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    target_arches.add(f"{major}{minor}")

for arch in sorted(target_arches):
    nvcc_args.extend(_gencode_flags_for_arch(arch))

ext_name = os.environ.get("FLASHDECODEFFN_EXT_NAME", "flashdecodeffn_v3_ext")

setup(
    name=ext_name,
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name=ext_name,
            sources=["flashdecodeffn_v3_ext.cpp", "flashdecodeffn_v3_ext_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
