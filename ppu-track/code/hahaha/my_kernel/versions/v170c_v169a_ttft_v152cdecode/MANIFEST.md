# v170c_v169a_ttft_v152cdecode

## Goal

Combine the two branches requested by the user:

- Keep `v169a` TTFT text big-ops:
  - packed QKV text prefill path
  - packed gate/up text MLP path
  - vision merger/bigops path
- Replace full-generate decode backend with `v152c` device-side first-token handoff:
  - `generate_from_device_nosync(first_token_tensor, ...)`
  - no full KV cache `zero_()` before prefill KV copy
  - decode RoPE bank precompute on a side CUDA stream while copying prefill KV

This version is intended to test whether `v169a`'s stronger TTFT path and `v170b`'s better local decode throughput are complementary.

## Files

- Wrapper: `/home/howard/Workspace/AICASGC/versions/evaluation_wrapper_v170c_v169a_ttft_v152cdecode_a800.py`
- Text bigops extension: `qwen3vl_text_bigops_v169a.cpython-312-x86_64-linux-gnu.so`
- Decode extension: `qwen3vl_megaqwen_v152c.cpython-312-x86_64-linux-gnu.so`
- Vision extension: `qwen3vl_vision_bigops_v167a.cpython-312-x86_64-linux-gnu.so`
- Source backups are kept under `csrc/`.

## Validation Notes

`py_compile` passed.

Submission-shape smoke:

- command shape: `benchmark.py --num-samples 1`
- timeout: 240s
- result: `/home/howard/Workspace/AICASGC/submissions/v170c_v169a_ttft_v152cdecode_submit_candidate/result/result_v170c_smoke_1.json`
- local RTX 4070S metrics:
  - TTFT: `105.66 ms`
  - throughput: `107.31 tok/s`

Short local benchmark:

- command shape: `benchmark.py --num-samples 10`
- timeout: 300s
- result: `/home/howard/Workspace/AICASGC/submissions/v170c_v169a_ttft_v152cdecode_submit_candidate/result/result_v170c_local10_timeout300.json`
- local RTX 4070S metrics:
  - TTFT: `98.02 ms`
  - throughput: `99.50 tok/s`

Full local benchmark:

- command shape: `benchmark.py --num-samples 150`
- timeout: 900s
- result: `/home/howard/Workspace/AICASGC/submissions/v170c_v169a_ttft_v152cdecode_submit_candidate/result/result_v170c_local150_timeout900.json`
- local RTX 4070S metrics:
  - TTFT: `105.79 ms`
  - throughput: `100.85 tok/s`

## Local Interpretation

Compared with existing local 150-sample results:

- `v169a`: `106.23 ms / 99.75 tok/s`
- `v170b`: `108.54 ms / 101.33 tok/s`
- `v170c`: `105.79 ms / 100.85 tok/s`

`v170c` keeps the TTFT advantage of `v169a` and recovers most of the `v170b` throughput benefit, but it does not beat `v170b` on throughput. This is a useful official A800 candidate if the goal is to test whether the two branches are complementary on target hardware.
