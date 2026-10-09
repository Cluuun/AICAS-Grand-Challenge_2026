# KV260 mmproj NPU Acceleration Design

## 1. Purpose

This document explains, from zero context, the real goal of the work, the hard constraints, the acceptable implementation scope, the feasible technical options, and the acceptance criteria for accelerating `mmproj` in this repository with a custom NPU on KV260.

The intended audience is:

- the person implementing the NPU-side GEMM operator
- the person integrating that operator into `llama.cpp`
- anyone reviewing whether the proposed implementation actually solves the right problem

This is a design and execution document. It is not a user manual.

## 2. Background

This repository contains `llama.cpp` plus multimodal support under `tools/mtmd/`. In the multimodal path, image features are processed by a vision encoder / projector pipeline commonly referred to here as `mmproj`.

For KV260 throughput evaluation, `mmproj` is a meaningful bottleneck. Existing profiling in this repo shows:

- `mmproj` total time is dominated by `GGML_OP_MUL_MAT`
- `MUL_MAT` accounts for about `94.6%` of profiled `mmproj` time in the current workload
- the important hotspot class for this project is:
  - quantized weight: `q8_0`
  - activation: `f32`
  - output: `f32`

The target hardware is a custom NPU connected through a runtime already present in this repository under:

- [npuruntime/kv260/runtime/npu_runtime.h](/home/gugugu/work/llama.cpp-kv260-20260407/npuruntime/kv260/runtime/npu_runtime.h)
- [npuruntime/kv260/runtime/npu_runtime.cpp](/home/gugugu/work/llama.cpp-kv260-20260407/npuruntime/kv260/runtime/npu_runtime.cpp)

That runtime already provides:

- a runtime-managed CMA buffer exposed through `mmap`
- sub-allocation from that CMA heap via `npu_mem_alloc()`
- internal virtual-to-physical translation inside the runtime
- lower-level DMA / compute primitives
- access to the KV260 device node `/dev/npu_kv260`

Therefore, the integration does not need `ggml` or `llama.cpp` to provide physical addresses for ordinary host allocations.

## 3. Real Goal

The real goal is:

1. accelerate the dominant `mmproj` linear layers on KV260 using the NPU
2. keep the integration narrow enough that it can be implemented and validated without rewriting the whole graph system
3. preserve correctness against the current CPU / FP16 reference to an explicitly defined acceptance threshold

The project is not trying to solve all of `llama.cpp`. It is specifically focused on the `mmproj` path used during multimodal throughput evaluation.

## 4. Non-Goals

The following are explicitly out of scope for the first usable version:

- accelerating all `GGML_OP_MUL_MAT` in the entire model stack
- accelerating `f32 * f32` attention matmuls in the first version
- changing the generic `ggml` scheduler design
- requiring `ggml` to expose physical addresses for generic host buffers
- implementing a full standalone `ggml` backend device from day one
- fusing every possible operator around GEMM

These may become future extensions, but they are not necessary to deliver the initial value.

## 5. Hard Constraints

The implementation must satisfy all of the following:

1. The NPU compute core is assumed to perform matrix multiplication, not floating-point quantization or dequantization.
2. Weight quantization and activation quantization are performed on the CPU side.
3. Final output dequantization is also performed on the CPU side.
4. The NPU-visible buffers must live in the KV260 runtime's CMA heap allocated by `npu_mem_alloc()`.
5. The implementation must be able to fall back safely to the original CPU path if any NPU precondition is not met.
6. The integration must preserve current graph semantics unless an explicitly chosen fused path replaces them.

## 6. Data Types and Tensor Semantics

For the target `ggml` operator:

- `op = GGML_OP_MUL_MAT`
- `src0 = weight`
- `src1 = activation`
- `dst = output`

For the contiguous 2D case used here, the logical interpretation is:

- `src0` shape: `[K, N]`
- `src1` shape: `[K, M]`
- `dst` shape: `[N, M]`

Equivalent GEMM view:

- activation matrix `A`: shape `M x K`
- weight matrix `W`: shape `N x K`
- output matrix `C`: shape `M x N`

where:

- `K` is the reduction dimension
- `N` is the output-channel dimension
- `M` is the number of activation rows / tokens / patches in that call

This mapping is important because `per-output-channel` weight quantization means:

- one weight scale per output channel
- number of weight scales equals `N`

### Example

If a weight tensor has shape `[12288, 960]`, then:

- `K = 12288`
- `N = 960`
- there are `960` independent output channels
- there will be `960` weight scales in a per-output-channel scheme

## 7. Numerical Scheme Chosen for This Project

The chosen scheme is:

- weight quantization: `per-output-channel symmetric int8`
- activation quantization: `static asymmetric u8`
- NPU input interpretation for activation: `u8 - 128`
- NPU output: `int32`
- CPU final output: `f32`

### 7.1 Weight Path

Model weights may already be stored in `ggml` format such as `q8_0`. For the NPU path, that storage format is not treated as the final hardware format.

The conversion pipeline is:

1. read the original `ggml q8_0` weight
2. dequantize it to `fp32`
3. for each output channel `n`, compute a separate symmetric scale `w_scale[n]`
4. quantize that channel to signed `int8`
5. compute the signed sum of each output-channel row:
   - `sum_w[n] = sum_k W_q[n, k]`

The runtime / NPU GEMM sees only:

- `int8` weight values

The `w_scale[n]` array is not used inside the systolic array. It is only used later by the CPU for dequantization.

### 7.2 Activation Path

Activations are not dynamically quantized at runtime based on the current input alone. Instead, static calibration is used.

For each target weight tensor name, calibration produces:

- `a_scale`
- `zp_u8`
- optional clipping range such as `clip_min` and `clip_max`

At runtime:

1. the `f32` activation is clipped as required by calibration
2. it is quantized to `u8`
3. the `u8` bytes are copied into the runtime-managed CMA buffer
4. the NPU full GEMM interprets the activation as `u8 - 128`

This means the effective signed zero-point is:

- `zp_i8 = zp_u8 - 128`

### 7.3 Compensation Term

Because the activation is asymmetric and the NPU sees `u8 - 128`, a compensation term is required.

For each output channel:

- `comp[n] = -zp_i8 * sum_w[n]`

This compensation can be treated as an `int32` bias-like term.

### 7.4 Output Reconstruction

If the NPU returns:

- `acc_i32[m, n]`

then the CPU reconstructs:

- `out_f32[m, n] = (acc_i32[m, n] + comp[n]) * a_scale * w_scale[n]`

This is why the multiple weight scales do not interfere with the systolic array:

- the array still does pure `int8 * int8 -> int32`
- the per-channel scales are only part of CPU-side postprocessing

## 8. Bias Handling

There are two possible meanings of "bias" in this project:

1. compensation bias
   - generated from quantization math
   - needed only because activation quantization is asymmetric
2. model bias
   - the original learned bias tensor already present in the graph

These are different things and must not be confused.

## 9. Feasible Implementation Options

### Option A: NPU GEMM Only, Bias Add Remains in Graph

This is the simplest practical integration.

Behavior:

- NPU computes GEMM only
- CPU wrapper applies compensation and dequantization
- existing `ggml_add(..., model_bias)` nodes remain unchanged

Advantages:

- smallest graph change
- simplest debug path
- safest fallback
- enough to validate the end-to-end data path quickly

Disadvantages:

- model bias is not computed on the NPU
- not maximal fusion

### Option B: NPU GEMM + Model Bias Fusion

Behavior:

- NPU computes GEMM
- NPU also adds the compensation term
- NPU also adds the learned model bias
- graph must not execute the original bias `ADD` again

Advantages:

- closer to the fully fused target
- fewer CPU-side operations

Disadvantages:

- graph integration is meaningfully more complex
- the builder must replace `mul_mat + add(bias)` with one fused op or custom op
- it is easier to introduce silent double-bias bugs

### Option C: Full Custom Fused Graph Operators

Behavior:

- replace selected graph subexpressions such as `mul_mat + add + activation` with custom operators

Advantages:

- maximal control

Disadvantages:

- much higher integration risk
- larger validation surface

## 10. Recommended Phasing

The recommended implementation order is:

### Phase 1

Deliver `Option A`.

That means:

- use the NPU for full GEMM only
- do activation quantization on CPU
- do compensation and dequantization on CPU
- keep existing graph bias-add nodes unchanged

This phase proves:

- the runtime CMA allocation path
- weight preparation
- activation staging
- full GEMM runtime interface
- numerical correctness
- measurable speedup on the intended hotspot

### Phase 2

If Phase 1 passes correctness and performance gates, evaluate `Option B`.

That phase would:

- move compensation into NPU-side bias input
- optionally move learned model bias into NPU-side fused bias
- replace `mul_mat + add(bias)` in selected `clip.cpp` call sites with a fused path

## 11. Integration Target Inside This Repository

The target integration point is the multimodal projector path in:

- [tools/mtmd/clip.cpp](/home/gugugu/work/llama.cpp-kv260-20260407/tools/mtmd/clip.cpp)

The hardware runtime lives in:

- [npuruntime/kv260/runtime/npu_runtime.h](/home/gugugu/work/llama.cpp-kv260-20260407/npuruntime/kv260/runtime/npu_runtime.h)
- [npuruntime/kv260/runtime/npu_runtime.cpp](/home/gugugu/work/llama.cpp-kv260-20260407/npuruntime/kv260/runtime/npu_runtime.cpp)

The likely `ggml` integration path is the CPU extra-buffer / tensor-traits mechanism under:

- `ggml/src/ggml-cpu/`

This is preferred over building a new standalone backend for the initial version because:

- the scope is narrower
- the target operator class is limited
- the fallback story is simpler

## 12. Memory Model

The implementation should assume the following memory model:

- the KV260 runtime's CMA buffer is the only DMA-visible memory region used by this feature
- it is obtained by the runtime from the KV260 driver and sub-allocated via `npu_mem_alloc()`
- it is visible to CPU through `mmap`
- the runtime itself performs virtual-to-physical translation internally

Therefore:

- ordinary `ggml` host pointers do not need physical-address support
- any data needed by DMA must be copied or prepared into the CMA-backed runtime heap

### NPU-visible Buffers Needed

At minimum, the following data objects must live in the CMA-backed runtime heap:

- prepared `int8` weights
- quantized `u8` activations
- `int32` output buffer

If compensation or fused bias is handled on the NPU in a later phase, then the following also become NPU-visible:

- `int32` compensation vector
- possibly fused `int32` or hardware-form bias buffer

## 13. Proposed Full GEMM Runtime Contract

The intended high-level contract is:

- the integration layer does not tile `M`, `N`, or `K`
- it calls one "full GEMM" runtime operator
- the runtime or hardware implementation hides the internal tiling and DMA sequence

Recommended shape convention:

- activation input: `A[M][K]`, row-major
- weight input: `W[N][K]`, row-major
- output: `C[M][N]`, row-major

Recommended API shape:

```c
typedef void * kv260_npu_gemm_handle_t;

typedef struct {
    const int8_t * weight_nk;   // N x K, row-major, runtime CMA pointer
    uint32_t       n;
    uint32_t       k;
    uint32_t       ldw;         // usually K
    bool           asymmetric_activations;
} kv260_npu_gemm_prepare_desc;

int npu_full_gemm_prepare(
    const kv260_npu_gemm_prepare_desc * desc,
    kv260_npu_gemm_handle_t * out_handle);

typedef struct {
    kv260_npu_gemm_handle_t handle;
    const uint8_t * act_mk;     // M x K, row-major, runtime CMA pointer
    uint32_t        m;
    uint32_t        lda;        // usually K
    int32_t       * out_mn;     // M x N, row-major, runtime CMA pointer
    uint32_t        ldc;        // usually N
} kv260_npu_gemm_run_desc;

int npu_full_gemm_run(const kv260_npu_gemm_run_desc * desc);

void npu_full_gemm_destroy(kv260_npu_gemm_handle_t handle);
```

For the first software-integrated version, it is acceptable to implement this contract as a CPU stub in the runtime first, so the graph integration and correctness work can proceed before the real hardware GEMM is finished.

## 14. Calibration Requirements

Calibration is not optional for the chosen scheme.

The calibration data should be representative of the intended evaluation workload.

Target requirement already discussed for this project:

- use the current `acceval` data mix
- validate on 100 samples

Each calibrated entry must be bound to a concrete weight tensor name, not just an operator kind, to avoid ambiguity.

Recommended calibration record fields:

- model identifier
- weight tensor name
- `a_scale`
- `zp_u8`
- `clip_min`
- `clip_max`

## 15. Acceptance Criteria

An implementation is accepted only if all of the following are true.

### 15.1 Functional Correctness

- target `mmproj` calls execute successfully without graph corruption
- non-target operators continue to work unchanged
- NPU path falls back safely when preconditions are not met

### 15.2 Numerical Correctness

Compared with the current FP16 / reference path:

- `acceval` run on 100 samples must degrade by no more than 5 samples

This is the primary acceptance criterion for quality.

### 15.3 Performance

The implementation must show a real reduction in the targeted hotspot:

- `mmproj` profiling must show a reduction in the target `q8_0 weight + f32 activation` `MUL_MAT` time

The goal is not to speed up every `MUL_MAT`. It is specifically to reduce the dominant `q8-weight` projector workload.

### 15.4 Engineering Safety

- the design must preserve a clean CPU fallback path
- the integration must not require general-purpose physical-address support from `ggml`
- the runtime interface must remain stable when the real hardware GEMM replaces the software stub

## 16. Risks

The main risks are:

1. Calibration error
   - static activation quantization may fail to meet the quality target
2. Weight-layout mismatch
   - if the runtime and `ggml` integration disagree on `[K,N]` vs `W[N][K]`
3. Silent compensation mistakes
   - especially around `u8 -> i8` reinterpretation and zero-point handling
4. Double bias bugs
   - only relevant if bias fusion is attempted later
5. Memory pressure in the runtime CMA heap
   - prepared weights plus activation and output scratch must fit

## 17. Recommended First Deliverable

The recommended first deliverable is:

- NPU full GEMM contract defined
- runtime software stub implemented for that contract
- `ggml` integration for target `q8_0` `MUL_MAT`
- CPU activation quantization
- CPU compensation
- CPU dequantization
- existing graph bias add unchanged
- correctness and `acceval` gate passed

This is the shortest path to proving the architecture is valid.

## 18. Future Extension: Bias on NPU

If later we want bias to be computed on the NPU as well, that should be treated as a separate phase with its own design gate.

There are two possible sub-goals:

1. move compensation into NPU bias input
2. move both compensation and model bias into NPU fused bias input

The second requires changing graph construction so that the original `ADD bias` node does not run again.

That is a valid extension, but it is intentionally not the required starting point.

## 19. Final Decision

At the current stage, the most reasonable engineering decision is:

- use a full GEMM NPU operator
- keep quantization and dequantization on the CPU
- use per-output-channel weight quantization
- use static asymmetric activation quantization
- use the KV260 runtime CMA heap as the only DMA-visible memory pool
- start with GEMM acceleration first
- defer bias fusion until the simpler path is working and validated

That path is technically feasible, aligned with the existing runtime, and narrow enough to validate properly.
