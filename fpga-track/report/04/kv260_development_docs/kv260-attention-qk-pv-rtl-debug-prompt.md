# KV260 Decode Attention QK/PV RTL Debug Prompt

## Goal

Reproduce and fix the remaining decode attention mismatch in the KV260 decode
RTL/runtime path. RoPE is already isolated to CPU. FFN SwiGLU scale-combine and
KV quant are also handled on CPU for now. The remaining failing path is grouped
QK softmax and grouped PV with the old 96-byte stream API.

## Source trees

- llama.cpp integration:
  `/home/gugugu/work/llama.cpp-kv260-20260407`
- old local stream runtime used by llama.cpp:
  `/home/gugugu/work/llama.cpp-kv260-20260407/tmp/decode_runtime_97a4824_stream`
- decode RTL source:
  `/mnt/c/vivado/versa_decode/rtl`
- board overlay used in the failing runs:
  `/mnt/c/vivado/KV260/out/dec_200m_grouprel_0606a_app`

## Evidence 1: old runtime synthetic test

New local old-API runtime test:

```text
tmp/decode_runtime_97a4824_stream/tests/stream_attention_qk_pv_isolation_test.cpp
tmp/decode_runtime_97a4824_stream/Makefile
```

Build:

```bash
source /home/gugugu/petalinux/sdk/kv260-2025.1/environment-setup-cortexa72-cortexa53-amd-linux
make -C /home/gugugu/work/llama.cpp-kv260-20260407/tmp/decode_runtime_97a4824_stream \
  -j8 kv260_stream_attention_qk_pv_isolation_test
```

Board run on `dec_200m_grouprel_0606a_app`:

```text
kv260_stream_attention_qk_pv_isolation_test: old 96B API QK softmax/PV isolation
qk_raw_stats group=0 min=-3.77051 max=3.08203 span=6.85254 mean=0.0466148 scaled_span=0.856567
qk_raw_stats group=1 min=-3.45703 max=3.5127 span=6.96973 mean=0.0754341 scaled_span=0.871216
qk_raw_stats group=2 min=-3.91992 max=4.25684 span=8.17676 mean=0.122639 scaled_span=1.02209
qk_raw rc=0
qk_raw checked=1503 mismatches=668 max_abs=5.85547 mean_abs=0.790828
qk_softmax rc=0 note=no kq_scale field in old 96B stream descriptor, expected kq_scale=0.125
qk_softmax_vs_scaled checked=1503 mismatches=0 max_abs=0.00262708 mean_abs=0.000222224
qk_softmax_vs_unscaled checked=1503 mismatches=11 max_abs=0.0699268 mean_abs=0.00153384
CLASSIFY:qk_softmax_scaled_ok
pv_fixed_stride_model_like rc=0 seq_len=501 capacity=4096 row_tile_stride=131072
pv_fixed_stride_model_like checked=192 mismatches=0 max_abs=0.00320872 mean_abs=0.00102547
SUMMARY:qk_raw=fail qk_softmax_scaled=ok pv_model_like=ok
```

Interpretation:

- Old 96B API does not expose `kq_scale`, but current RTL appears to apply the
  head_dim=64 scale internally for `POST_SOFTMAX`. Missing `kq_scale` is not the
  current text mismatch cause for this model.
- Synthetic grouped `PV` with `seq_len=501`, `capacity=4096`,
  `row_tile_stride=131072` passes.
- `QK POST_BYPASS` large/tail MVOUT fails, but the llama.cpp path uses
  `POST_SOFTMAX`; still worth checking MVOUT/reduce destination handling.

## Evidence 2: llama.cpp real-data compare

New llama.cpp diagnostic env:

```text
AICAS_TEXT_DECODE_ATTN_NPU_COMPARE_QKPV=1
AICAS_TEXT_DECODE_ATTN_NPU_COMPARE_QKPV_LAYER=<layer, default 0>
AICAS_TEXT_DECODE_ATTN_NPU_COMPARE_QKPV_POSITION=<optional token position>
```

Run with CPU RoPE and NPU QK/PV:

```bash
bash -o pipefail AICAS2026/aicas_semi/scripts/run_semi_eval_kv260.sh \
  --generic-fastpath \
  --decode-npu \
  --decode-overlay-dir /mnt/c/vivado/KV260/out/dec_200m_grouprel_0606a_app \
  --decode-overlay-app dec_200m_grouprel_0606a_app \
  --overlay-switch-mode fast \
  --throughput-only \
  --throughput-max-tokens 2 \
  --decode-profile \
  --skip-build \
  --server-bin /home/gugugu/work/llama.cpp-kv260-20260407/build-kv260-semi-arm/bin/llama-server \
  --server-env AICAS_TEXT_DECODE_AWQ_NPU_DEBUG=1 \
  --server-env AICAS_TEXT_DECODE_AWQ_NPU_GEMV_ONLY=0 \
  --server-env AICAS_TEXT_DECODE_AWQ_FUSED_FFN_NPU=1 \
  --server-env AICAS_TEXT_LM_HEAD_W8A16_NPU=1 \
  --server-env AICAS_TEXT_DECODE_ATTN_NPU=1 \
  --server-env AICAS_TEXT_DECODE_ATTN_NPU_CPU_ROPE=1 \
  --server-env AICAS_TEXT_DECODE_ATTN_NPU_COMPARE_QKPV=1 \
  --run-id 20260606Tdecode200m-qkpv-compare-layer0-2tok
```

Result text:

```text
Theִ
```

Key layer0 position501 signatures:

```text
qk_prob kv_head=0 elems=1536 mismatches=29 max_abs=4.24938 mean_abs=0.030597
qk_prob kv_head=1 elems=1536 mismatches=35 max_abs=4.22791 mean_abs=0.0306318
qk_prob kv_head=2 elems=1536 mismatches=22 max_abs=4.24786 mean_abs=0.0300427
qk_prob kv_head=3 elems=1536 mismatches=25 max_abs=4.24835 mean_abs=0.0300118
qk_prob kv_head=4 elems=1536 mismatches=30 max_abs=4.24923 mean_abs=0.0304356
```

Example QK probability failures:

```text
expected=2.2769e-05 got=-2.20117
expected=0.000227928 got=2.11328
expected=0.0195312 got=-3.12891
```

These `got` values are not probabilities. They look like raw GEMV/logit values
or stale output leaking through the QK softmax output stream for real llama.cpp
data near the token tail.

Run with CPU QK and NPU PV:

```bash
... same command, plus:
  --server-env AICAS_TEXT_DECODE_ATTN_NPU_CPU_QK=1 \
  --server-env AICAS_TEXT_DECODE_ATTN_NPU_COMPARE_QKPV=1 \
  --run-id 20260606Tdecode200m-pv-compare-cpuqk-layer0-2tok
```

Result text:

```text
The painting
```

PV compare using correct CPU probabilities:

```text
pv_out kv_head=0 elems=192 mismatches=20 max_abs=0.256348 mean_abs=0.0115942
pv_out kv_head=1 elems=192 mismatches=10 max_abs=0.0416107 mean_abs=0.00549685
pv_out kv_head=2 elems=192 mismatches=2 max_abs=0.0421753 mean_abs=0.00438934
pv_out kv_head=3 elems=192 mismatches=6 max_abs=0.0529785 mean_abs=0.00535594
pv_out kv_head=4 elems=192 mismatches=8 max_abs=0.237915 mean_abs=0.00772225
```

Several mismatches have `got=0` while CPU expects nonzero values, e.g.:

```text
kv_head=0 idx=34 expected=0.195435 got=0
kv_head=4 idx=142 expected=0.237915 got=0
```

## RTL simulation targets

Please build targeted RTL simulation around the exact stream descriptors below.

### QK softmax real-data failure

Descriptor shape:

```text
role=QK
mode=W8A16
post_op=SOFTMAX
dst=OUTPUT
m=seq_len around 502
n=64
group_count=3
act_group_stride_bytes=256
output_precision=FP16
```

Input layout used by llama.cpp:

- Q activation: grouped Q heads, one 64-FP16 vector per group, 256B stride.
- K payload: per KV head cache, scale region first, then INT8 K rows in
  token-major tiles of 32 tokens x 64 dims.
- K scale address: `token * 2`, equivalent to `tile * 64 + lane * 2`.

Checks:

- Confirm softmax writes valid probabilities for every valid token in every
  group, including token indices around 480..seq_len-1.
- Confirm no raw GEMV/logit values remain in MVOUT output after softmax.
- Confirm group output layout for `POST_SOFTMAX` is the same compact layout the
  runtime assumes, or explicitly expose/write the group stride.
- Confirm `dst=OUTPUT` plus `post_op=SOFTMAX` routes the reduced probabilities
  to the MVOUT source, not stale GEMV output or post buffer.
- Confirm the internal 1/sqrt(64) scale is actually applied for all groups and
  all tail rows, or expose a runtime `kq_scale` field if it is not guaranteed.

### PV real-data failure

Descriptor shape:

```text
role=PV
mode=W8A16
post_op=BYPASS
dst=OUTPUT
m=64
n=seq_len around 502
group_count=3
act_group_stride_bytes=ceil_align(seq_len * 2, 256)
weight_row_tile_stride_bytes=131072
weight_capacity_tokens=4096
flags=ENABLE_ACT_SCALE | KV_COL_SCALE | UNIT_WEIGHT_SCALE
output_precision=FP16
```

Input layout used by llama.cpp:

- Probabilities: FP16, group-major, each group padded to 256B stride.
- V scale: one FP16 scale per token, shared across groups.
- V payload: fixed-stride transposed layout:
  `row_tile * 131072 + col_tile * 2048 + row_lane * 64 + col_lane`.

Checks:

- Confirm every output dim in every group is written. The board compare saw
  exact zeros at several nonzero CPU-reference dims.
- Confirm V scale is read once per token and is not accidentally grouped.
- Confirm row_tile 0 and row_tile 1 are both read for m=64.
- Confirm `weight_capacity_tokens=4096` and `n≈502` do not cause token-tail or
  stride truncation.

## Current hypothesis

This is probably not a simple missing runtime `kq_scale` parameter for the
current head_dim=64 model. The stronger RTL candidates are:

1. QK `POST_SOFTMAX` real-data/tail handling leaves raw logits or stale output
   in the MVOUT stream for some valid tail tokens.
2. PV fixed-stride path has a real-data/layout corner where some dims are not
   written or some V rows/scales are skipped, despite synthetic PV passing.
3. If RTL expects a different grouped QK softmax output stride or PV fixed
   stride contract than the old runtime assumes, the API must expose that
   explicitly instead of relying on implicit compact copies.
