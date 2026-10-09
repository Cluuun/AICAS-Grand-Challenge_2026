# cuBLASLt FFN Trial

This directory contains a standalone cuBLASLt FFN prototype for Qwen3VL hot shapes.

## Build

```bash
cd /root/aicas26-lhd/flashDecodeFFN/cutlass_kernel
/root/miniconda3/envs/aicas26/bin/python3 setup.py build_ext --inplace
```

## Benchmark

```bash
cd /root/aicas26-lhd/flashDecodeFFN/cutlass_kernel
/root/miniconda3/envs/aicas26/bin/python3 benchmark_cublaslt.py --warmup 50 --iters 300
```

It prints:
- text `N=1,H=2048,I=6144`
- text `N=690,H=2048,I=6144`
- vision `N=2688,H=1024,I=4096`
- numerical error and speedup vs PyTorch
- selected cuBLASLt heuristic summary

