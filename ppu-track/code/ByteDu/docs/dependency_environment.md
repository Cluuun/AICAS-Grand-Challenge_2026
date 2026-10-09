# Dependency environment

## Verified local environment

- Python: 3.12.3
- OS: Linux 5.10 x86_64
- GPU: PPU-ZW810E
- Compute capability: 8.0
- PyTorch: 2.8.0
- CUDA runtime reported by PyTorch: 12.9

## Python packages used by the fused runtime

| package | verified version | purpose |
|---|---:|---|
| torch | 2.8.0 | model runtime, CUDA graph, extension ABI |
| transformers | 4.57.0 | AutoProcessor / GenerationConfig compatibility |
| flash_attn | 2.7.4.post1 | prefill/decode attention kernels |
| triton | 3.4.0 | custom Triton kernels |
| safetensors | 0.7.0 | model and optional quant codebook loading |
| acext | 1.5.1 / SDK-provided | PPU Acext A8W8 prefill GEMM and opt-in decode lm_head experiments |
| datasets | 4.8.5 | local benchmark dataset loading |
| Pillow | 12.1.0 | image resize/processing |
| numpy | 2.2.6 | processor-side utility dependency |
| tqdm | 4.67.1 | local benchmark progress |
| psutil | 7.2.2 | local benchmark system info |

## Submission notes

- Submission archives are emitted as `submission_fusion_<timestamp>.zip` and include `requirements.txt` for environment documentation and optional dependency resolution.
- The official worker is expected to provide `benchmark.py`, model weights, data, CUDA, PyTorch, FlashAttention and Triton-capable runtime.
- On PPU SDK v2.x images, Acext is loaded from the SDK/Python environment when available; the runtime falls back to BF16/Q8 paths if `acext.int8_gemm` is unavailable or if speed/error auto-gating rejects every A8W8 prefill target.
- Custom CUDA operators are shipped as prebuilt `.so` files under `prebuilt/`; the runtime should not need to compile `.cu` files during evaluation. `prebuilt/a8w8_decode_ext.so` provides opt-in PerToken/PerChannel W8A8 down_proj fused variants plus fixed/lagged/on-the-fly activation quant ablations, while `prebuilt/acext_weightonly_ext.so` is loaded by the default Acext A16W8 int8_pc decode lm_head path when Acext is available. The runtime falls back to Q8 paths if optional extensions are unavailable.
- The prebuilt extensions were validated on Torch 2.8.0 / CUDA 12.9 / SM80. If the worker ABI changes, rebuild the `.so` files for that environment.
