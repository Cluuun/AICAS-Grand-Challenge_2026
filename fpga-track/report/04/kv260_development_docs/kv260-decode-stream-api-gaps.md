# KV260 Decode Stream API Gaps

Date: 2026-06-06

This note tracks temporary software workarounds in the llama.cpp decode NPU
path. They are intentionally kept visible because both cases should move into
the runtime/RTL contract before the path is treated as final.

## SwiGLU Down Projection Scale Combine

Current behavior:

- Gate and up projections run through `npu_stream_gemv_run()`.
- The gate projection can produce `silu(gate)`.
- The down projection also needs the AWQ input smooth scale for
  `ffn_down.weight`.
- The current stream API does not expose an operation that multiplies
  `silu(gate[col]) * down_inv_smooth[col]` before the down GEMV.
- llama.cpp therefore builds a combined FP16 act-scale vector on CPU and passes
  it as `act_scale_ptr` for the down projection.

Desired contract:

- Runtime/RTL should provide a stream-visible path that combines the gate/SILU
  result with the down AWQ activation scale before the down projection.
- The host should not need to materialize a full combined scale vector per
  decode layer.

## Attention K/V Projection KV Quant

Current behavior:

- The legacy split attention path could ask RTL to emit quantized K/V payload
  plus per-head scales.
- The release stream API only exposes `npu_stream_gemv_run()` and does not
  expose the KV-quant output payload or scale registers as a descriptor result.
- llama.cpp currently runs K and V projection as stream W4A16 FP16-output
  GEMVs, then CPU-quantizes the 5 KV heads and writes the existing CMA KV-cache
  layout:
  - K cache: token-major W8 payload plus per-token FP16 scale.
  - V cache: fixed-stride pre-transposed W8 payload plus per-token FP16 scale.
- Grouped QK stream output is compact group-major
  `prob[group][token]`, but grouped PV needs a 256B-aligned
  `act_group_stride_bytes`. llama.cpp therefore repacks grouped QK output from
  compact layout into the padded probability layout before PV.

Desired contract:

- Add a stream descriptor mode or result contract for KV projection that emits
  RTL-quantized K/V rows and their per-head FP16 scales.
- The host should only append or reorder this quantized result into the
  long-lived decode KV cache.
- The API should describe the K token-major and V pre-transposed output layout,
  scale layout, alignment, and cache-capacity rules.
- Add a grouped QK output-stride contract, or allow a QK-to-PV on-chip handoff
  through ACT/post buffers, so the host does not need to repack probabilities.

## Temporary Validation Rule

The temporary CPU paths are acceptable only for bring-up correctness. Throughput
numbers taken with these paths should be labeled as diagnostic until the missing
stream API/RTL features above are implemented.
