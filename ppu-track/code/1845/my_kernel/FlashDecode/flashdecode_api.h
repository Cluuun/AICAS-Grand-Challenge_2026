#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>

extern "C" {

void flashdecode_mma_init();
void flashdecode_mma_init_bf16();
size_t flashdecode_mma_workspace_size(int B, int S, int H_kv, int G);
void flashdecode_mma_run(
    const __half *q,
    const __half *const *k_ptrs,
    const __half *const *v_ptrs,
    const int *seq_lengths,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int H,
    int S,
    cudaStream_t stream);

void flashdecode_mma_run_qstride(
    const __half *q,
    const __half *const *k_ptrs,
    const __half *const *v_ptrs,
    const int *seq_lengths,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream);

void flashdecode_mma_run_qstride_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *const *k_ptrs,
    const __nv_bfloat16 *const *v_ptrs,
    const int *seq_lengths,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream);

void flashdecode_mma_run_qstride_contig(
    const __half *q,
    const __half *k_base,
    const __half *v_base,
    const int *seq_lengths,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream);

void flashdecode_mma_run_qstride_contig_noseqlens(
    const __half *q,
    const __half *k_base,
    const __half *v_base,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream);

void flashdecode_mma_run_qstride_contig_noseqlens_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k_base,
    const __nv_bfloat16 *v_base,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    cudaStream_t stream);

void flashdecode_mma_run_qstride_contig_causal(
    const __half *q,
    const __half *k_base,
    const __half *v_base,
    __half *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    int virtual_q,
    cudaStream_t stream);

void flashdecode_mma_run_qstride_contig_causal_bf16(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *k_base,
    const __nv_bfloat16 *v_base,
    __nv_bfloat16 *out,
    void *temp,
    int B,
    int H_kv,
    int G,
    int Q_stride,
    int H,
    int S,
    int virtual_q,
    cudaStream_t stream);

void flashdecode_run_decode_multi_masked(
    const __half *query,
    const __half *key,
    const __half *value,
    const __half *mask,
    __half *out,
    int B,
    int H_q,
    int H_kv,
    int G,
    int Q,
    int S,
    cudaStream_t stream);

void flashdecode_run_decode_multi_masked_bf16(
    const __nv_bfloat16 *query,
    const __nv_bfloat16 *key,
    const __nv_bfloat16 *value,
    const __nv_bfloat16 *mask,
    __nv_bfloat16 *out,
    int B,
    int H_q,
    int H_kv,
    int G,
    int Q,
    int S,
    cudaStream_t stream);

}
