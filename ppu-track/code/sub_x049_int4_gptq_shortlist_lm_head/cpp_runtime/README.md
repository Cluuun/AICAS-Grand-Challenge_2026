# CPP Runtime Notes

This directory contains the optional AICAS prefill C++ runtime bridge.

The current runtime module is built from:

- [prefill_text_runtime.cpp](/data/home/tianjianyang/code/AICAS/cpp_runtime/prefill_text_runtime.cpp)

and loaded by:

- [loader.py](/data/home/tianjianyang/code/AICAS/cpp_runtime/loader.py)

## Current Status

The clean prefill direction keeps the C++ runtime as an execution-boundary/runtime experiment.

The historical stateful prefill q/k + rope + kv-update fused kernel is intentionally **not**
wired into the current runtime path. The kernel file is kept only as a legacy reference.

## Required Environment

When using the C++ runtime on this server, do not rely on the shell's default `PATH` and
`LD_LIBRARY_PATH`.

Use:

```bash
export PATH=/data/home/tianjianyang/miniconda3/envs/aicas/bin:$PATH
export LD_LIBRARY_PATH=/data/home/tianjianyang/miniconda3/envs/aicas/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
```

Why this matters:

- `torch.utils.cpp_extension.load(...)` requires `ninja`
- `ninja` is present in the `aicas` conda env, but may not be visible from the default shell path
- the prebuilt `.so` depends on PyTorch shared libraries under `torch/lib`
- if `LD_LIBRARY_PATH` does not include `torch/lib`, loading the module can fail with unresolved
  `libtorch*.so` / `libc10.so` or follow-on symbol errors

## Typical Failure Modes

### 1. `RuntimeError: Ninja is required to load C++ extensions`

This usually does **not** mean `ninja` is missing from the conda env.

On this machine, the common cause was that `PATH` did not include:

```bash
/data/home/tianjianyang/miniconda3/envs/aicas/bin
```

Check with:

```bash
PATH=/data/home/tianjianyang/miniconda3/envs/aicas/bin:$PATH which ninja
PATH=/data/home/tianjianyang/miniconda3/envs/aicas/bin:$PATH ninja --version
```

### 2. Import/load failure from `aicas_prefill_text_runtime.so`

Observed failure shape:

```text
ImportError: ...aicas_prefill_text_runtime.so: undefined symbol: ...
```

Before assuming this is a compiler/ABI issue, first check whether the PyTorch shared libraries
are resolvable:

```bash
ldd /data/home/tianjianyang/code/AICAS/cpp_runtime/.build/aicas_prefill_text_runtime.so
```

If `libtorch.so`, `libtorch_cpu.so`, `libtorch_python.so`, or `libc10.so` show as `not found`,
fix `LD_LIBRARY_PATH` first.

## Rebuild Rules

The loader uses:

- `AICAS_CPP_RUNTIME_FORCE_REBUILD=0`: load prebuilt `.so`, or build once when it is missing
- `AICAS_CPP_RUNTIME_FORCE_REBUILD=1`: delete stale outputs and rebuild from source

First rebuild after source changes:

```bash
export PATH=/data/home/tianjianyang/miniconda3/envs/aicas/bin:$PATH
export LD_LIBRARY_PATH=/data/home/tianjianyang/miniconda3/envs/aicas/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export AICAS_CPP_RUNTIME_FORCE_REBUILD=1

/data/home/tianjianyang/miniconda3/envs/aicas/bin/python - <<'PY'
from cpp_runtime import load_prefill_text_runtime_module
m = load_prefill_text_runtime_module(verbose=True)
print(m.runtime_metadata())
PY
```

After the rebuild succeeds, later runs can switch back to:

```bash
export AICAS_CPP_RUNTIME_FORCE_REBUILD=0
```

If the `.so` is absent in this mode, the loader now builds it once from
`prefill_text_runtime.cpp` instead of failing immediately.

## Sanity Check

Minimal module load check:

```bash
export PATH=/data/home/tianjianyang/miniconda3/envs/aicas/bin:$PATH
export LD_LIBRARY_PATH=/data/home/tianjianyang/miniconda3/envs/aicas/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH
export AICAS_CPP_RUNTIME_FORCE_REBUILD=0

/data/home/tianjianyang/miniconda3/envs/aicas/bin/python - <<'PY'
from cpp_runtime import load_prefill_text_runtime_module
m = load_prefill_text_runtime_module(verbose=False)
print(m.runtime_metadata())
PY
```

Expected output shape:

```text
{'name': 'aicas_prefill_text_runtime', 'stage': 'runner_aten_ops', 'has_cuda': True, ...}
```

## Practical Advice

If the C++ runtime suddenly stops loading on this server:

1. check `PATH` for `aicas/bin`
2. check `LD_LIBRARY_PATH` for `torch/lib`
3. run `ldd` on the built `.so`
4. only after that consider compiler / ABI debugging

Do not jump to "upgrade gcc" first unless the above checks are already clean and the rebuild still
fails.
