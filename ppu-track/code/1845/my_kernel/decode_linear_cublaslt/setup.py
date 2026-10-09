from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _gencode_flags_for_arch(arch: str) -> list[str]:
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


nvcc_args = ["-O3", "-std=c++17", "--use_fast_math"]
target_arches = {"80", "89"}
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    target_arches.add(f"{major}{minor}")
for arch in sorted(target_arches):
    nvcc_args.extend(_gencode_flags_for_arch(arch))

ext_modules = [
    CUDAExtension(
        name="decode_linear_cublaslt_ext",
        sources=["decode_linear_cublaslt_ext.cpp", "decode_linear_cublaslt_kernel.cu"],
        extra_compile_args={"cxx": ["-O3", "-std=c++17"], "nvcc": nvcc_args},
    )
]

setup(
    name="decode_linear_cublaslt_ext",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
