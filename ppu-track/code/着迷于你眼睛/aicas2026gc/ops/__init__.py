import triton
import triton.language as tl
import torch
import math


# ================================================================
# Split-K GEMV (BS=1 decode 特化, Tensor Core)
# ================================================================
# Specialized Split-K GEMV kernels for decode (BS=1)
# Gate_Up: [1,2048] @ [12288,2048].T  N=12288, K=2048
# Down:    [1,6144] @ [2048,6144].T   N=2048,  K=6144
# QKV:     [1,2048] @ [4096,2048].T   N=4096,  K=2048
# O:       [1,2048] @ [2048,2048].T   N=2048,  K=2048
# ================================================================


# ---------- Gate_Up: Split-K=4, K_PER_SPLIT=512 ----------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 512}, num_warps=8, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _gate_up_sk4_kernel(
    x_ptr,
    w_ptr,
    partial_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    N: tl.constexpr = 12288
    K: tl.constexpr = 2048
    K_PER_SPLIT: tl.constexpr = 512

    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)  # 0..3

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * K_PER_SPLIT

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k in range(0, K_PER_SPLIT, BLOCK_K):
        offs_k = k_start + k + tl.arange(0, BLOCK_K)
        x_chunk = tl.load(x_ptr + offs_k)
        w_chunk = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
        acc += tl.sum(w_chunk * x_chunk[None, :], axis=1)

    tl.store(partial_ptr + pid_k * N + offs_n, acc)


@triton.jit
def _gate_up_sk4_reduce(partial_ptr, out_ptr, x_ptr, rms_eps, DO_RMSNORM: tl.constexpr):
    N: tl.constexpr = 12288
    K: tl.constexpr = 2048
    BLOCK_N: tl.constexpr = 256
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.load(partial_ptr + 0 * N + offs_n)
    acc += tl.load(partial_ptr + 1 * N + offs_n)
    acc += tl.load(partial_ptr + 2 * N + offs_n)
    acc += tl.load(partial_ptr + 3 * N + offs_n)
    # 融合 RMSNorm: acc *= rsqrt(mean(x²) + eps)
    if DO_RMSNORM:
        sum_x2 = tl.zeros([1], dtype=tl.float32)
        for k in range(0, K, 256):
            xk = tl.load(x_ptr + k + tl.arange(0, 256)).to(tl.float32)
            sum_x2 += tl.sum(xk * xk)
        acc = acc * tl.rsqrt(sum_x2 / K + rms_eps)
    tl.store(out_ptr + offs_n, acc.to(out_ptr.dtype.element_ty))


def gate_up_gemv_sk4(x, weight, partial_buf, out, rms_eps=0.0):
    """Gate_Up: split-k=4, fp32 partial"""
    grid = lambda meta: (triton.cdiv(12288, meta["BLOCK_N"]), 4)  # noqa: E731
    _gate_up_sk4_kernel[grid](x.view(-1), weight, partial_buf)
    do_rms = rms_eps > 0
    x_ptr = x.view(-1) if do_rms else None
    _gate_up_sk4_reduce[(12288 // 256,)](partial_buf, out.view(-1), x_ptr, rms_eps, DO_RMSNORM=do_rms)
    return out


# ---------- O: Split-K=32, fp16 partial (N=2048, K=2048, K_PER_SPLIT=64) ----------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _o_sk32_fp16_kernel(x_ptr, w_ptr, partial_ptr, BLOCK_N: tl.constexpr):
    N: tl.constexpr = 2048
    K: tl.constexpr = 2048
    K_PER_SPLIT: tl.constexpr = 64
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_start = pid_k * K_PER_SPLIT
    offs_k = k_start + tl.arange(0, K_PER_SPLIT)
    x_chunk = tl.load(x_ptr + offs_k)
    w_chunk = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
    acc = tl.sum(w_chunk * x_chunk[None, :], axis=1)
    tl.store(partial_ptr + pid_k * N + offs_n, acc.to(tl.float16))


@triton.jit
def _o_sk32_fp16_reduce(partial_ptr, out_ptr, residual_ptr):
    N: tl.constexpr = 2048
    BLOCK_N: tl.constexpr = 256
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.load(partial_ptr + 0 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 1 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 2 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 3 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 4 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 5 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 6 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 7 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 8 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 9 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 10 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 11 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 12 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 13 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 14 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 15 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 16 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 17 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 18 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 19 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 20 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 21 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 22 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 23 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 24 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 25 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 26 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 27 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 28 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 29 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 30 * N + offs_n).to(tl.float32)
    acc += tl.load(partial_ptr + 31 * N + offs_n).to(tl.float32)
    # 融合 residual add
    if residual_ptr:
        acc += tl.load(residual_ptr + offs_n).to(tl.float32)
    tl.store(out_ptr + offs_n, acc.to(tl.float16))


def o_gemv_sk32_fp16(x, weight, partial_buf, out, residual=None):
    """O: split-k=32, fp16 partial. residual 融合 residual.add_"""
    grid = lambda meta: (triton.cdiv(2048, meta["BLOCK_N"]), 32)  # noqa: E731
    _o_sk32_fp16_kernel[grid](x.view(-1), weight, partial_buf)
    residual_ptr = residual.view(-1) if residual is not None else None
    _o_sk32_fp16_reduce[(2048 // 256,)](partial_buf, out.view(-1), residual_ptr)
    return out


@triton.autotune(
    configs=[
        # --- [1] 平衡型：最不容易出错，通常在各种卡上表现都很稳 ---
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        # --- [2] 带宽狂暴型：调大 BLOCK_K，试图每次循环读大块 x，减少循环次数 ---
        # K=2048，BLOCK_K=512 的话只需要循环 4 次！
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 512}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=3),
        # --- [3] 极限延迟掩盖型：增加 num_stages，疯狂 prefetch (预取) ---
        # 适合 A100/H100 这种 SRAM 极大的卡
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 128}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=5),
        # --- [4] 大 N 块型：每个 Block 处理更多词汇，减少 Block 总数 ---
        # 注意：BLOCK_N 太大可能会导致寄存器溢出 (Register Spill)
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["N", "K"],
)
@triton.jit
def _fused_lm_head_argmax_kernel_v2(
    x_ptr,  # [2048]
    w_ptr,  # [151936, 2048]
    scratch_val_ptr,  # [2000]
    scratch_idx_ptr,  # [2000]
    N: tl.constexpr,  # 151936
    K: tl.constexpr,  # 2048
    stride_wn: tl.constexpr,  # 2048
    stride_wk: tl.constexpr,  # 1
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # 2^n
):
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N

    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    k_offsets = tl.arange(0, BLOCK_K)

    w_block_ptr = w_ptr + n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk
    x_block_ptr = x_ptr + k_offsets

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x_chunk = tl.load(x_block_ptr)
        w_chunk = tl.load(w_block_ptr, mask=n_mask[:, None], other=0.0)

        # w_chunk: [BLOCK_N, BLOCK_K]
        # x_chunk: [BLOCK_K]
        # acc += tl.sum(w_chunk * x_chunk[None, :], axis=1)

        # ----- 极致的 Tensor Core Hack -----
        x_2d = tl.expand_dims(x_chunk, 1)  # [BLOCK_K, 1]
        x_bcast = tl.broadcast_to(x_2d, [BLOCK_K, 16])  # [BLOCK_K, 16]
        dot_out = tl.dot(w_chunk, x_bcast)
        # 4. 16列数据完全一样，只取第一列
        acc += tl.max(dot_out, axis=1)

        x_block_ptr += BLOCK_K
        w_block_ptr += BLOCK_K

    acc = tl.where(n_mask, acc, float("-inf"))

    local_max_idx = tl.argmax(acc, axis=0)
    local_max_val = tl.max(acc, axis=0)
    global_max_idx = n_start + local_max_idx

    tl.store(scratch_val_ptr + pid, local_max_val)
    tl.store(scratch_idx_ptr + pid, global_max_idx)


def fused_lm_head_sample(hidden_states, lm_head_weight, result_buf, block_max_scores, block_max_tokens):
    """Fused lm_head GEMV (TC hack) + argmax."""
    x = hidden_states.view(-1)
    N, K = lm_head_weight.shape

    # fill_(-inf) 已删除: buffer 初始化时已全部为 -inf (torch.full),
    # producer kernel 每轮覆盖前 num_blocks 个 slot, 尾部 slot 永远保持 -inf
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731

    _fused_lm_head_argmax_kernel_v2[grid](
        x,
        lm_head_weight,
        block_max_scores,
        block_max_tokens,
        N,
        K,
        lm_head_weight.stride(0),
        lm_head_weight.stride(1),
    )

    _lm_head_global_argmax_kernel[(1,)](block_max_scores, block_max_tokens, result_buf, NUM_BLOCKS=8192)
    return result_buf


@triton.jit
def _lm_head_global_argmax_kernel(val_ptr, idx_ptr, out_ptr, NUM_BLOCKS: tl.constexpr):
    """Read all block-local max values, find the global argmax, write the winning token index."""
    offs = tl.arange(0, NUM_BLOCKS)
    vals = tl.load(val_ptr + offs)
    best = tl.argmax(vals, axis=0)
    best_idx = tl.load(idx_ptr + best)
    tl.store(out_ptr, best_idx)


# ================================================================
# v5: tl.sum + nomask (N=151936 = 32*4748 = 64*2374 = 128*1187)
# ================================================================
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 32, "BLOCK_K": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 512}, num_warps=8, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _lm_head_v5_kernel(
    x_ptr,
    w_ptr,
    scratch_val_ptr,
    scratch_idx_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    K: tl.constexpr = 2048
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_k
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        x_chunk = tl.load(x_ptrs)
        w_chunk = tl.load(w_ptrs)
        acc += tl.sum(w_chunk * x_chunk[None, :], axis=1)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    local_max_idx = tl.argmax(acc, axis=0)
    local_max_val = tl.max(acc, axis=0)
    tl.store(scratch_val_ptr + pid, local_max_val)
    tl.store(scratch_idx_ptr + pid, n_start + local_max_idx)


def fused_lm_head_v5(hidden_states, lm_head_weight, result_buf, block_max_scores, block_max_tokens):
    """Fused lm_head GEMV (tl.sum, no mask) + argmax."""
    x = hidden_states.view(-1)
    # fill_(-inf) 已删除: 同 fused_lm_head_sample 的理由
    grid = lambda meta: (151936 // meta["BLOCK_N"],)  # noqa: E731
    _lm_head_v5_kernel[grid](x, lm_head_weight, block_max_scores, block_max_tokens)
    _lm_head_global_argmax_kernel[(1,)](block_max_scores, block_max_tokens, result_buf, NUM_BLOCKS=8192)
    return result_buf


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
    ],
    key=["HEAD_DIM"],
)
@triton.jit
def fused_qkv_prep_kernel(
    qkv_raw_ptr,
    q_offset,  # [16, 128]
    k_offset,  # [8, 128]
    v_offset,  # [8, 128]
    q_norm_w_ptr,  # [128]
    k_norm_w_ptr,
    cos_ptr,  # [64]
    sin_ptr,
    k_cache_ptr,  # [4096, 8, 128]
    v_cache_ptr,
    q_out_ptr,  # [16, 128]
    seq_len_ptr,  # 存放历史长度，用于确定写入 Cache 的位置
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_kc_s: tl.constexpr,
    stride_kc_h: tl.constexpr,
    stride_kc_d: tl.constexpr,
    stride_vc_s: tl.constexpr,
    stride_vc_h: tl.constexpr,
    stride_vc_d: tl.constexpr,
    stride_oqh: tl.constexpr,
    stride_oqd: tl.constexpr,
    eps: tl.constexpr,
    gqa_group: tl.constexpr,  # 2
    HEAD_DIM: tl.constexpr,  # 128
    HALF_HEAD_DIM: tl.constexpr,  # 64
    sm_scale: tl.constexpr,  # 直接融合在 Q 的 RoPE 中乘掉
):
    q_head_idx = tl.program_id(0)
    kv_head_idx = q_head_idx // gqa_group

    # 基础指针重定向
    q_raw_ptr = qkv_raw_ptr + q_offset
    k_raw_ptr = qkv_raw_ptr + k_offset
    v_raw_ptr = qkv_raw_ptr + v_offset

    # 切分前后半段的 offset (用于 RoPE)
    offs_lo = tl.arange(0, HALF_HEAD_DIM)
    offs_hi = offs_lo + HALF_HEAD_DIM

    # ==========================================
    # 0. 加载权值与上下文状态
    # ==========================================
    seq_idx = tl.load(seq_len_ptr) - 1  # 当前 Token 要写入的 Cache 位置

    cos_lo = tl.load(cos_ptr + offs_lo)
    sin_lo = tl.load(sin_ptr + offs_lo)

    # ==========================================
    # 1. 计算 Query (RMS Norm -> RoPE -> Scale -> 写出)
    # ==========================================
    q_ptr_lo = q_raw_ptr + q_head_idx * stride_qh + offs_lo * stride_qd
    q_ptr_hi = q_raw_ptr + q_head_idx * stride_qh + offs_hi * stride_qd
    # RMSNorm 需要 FP32（平方和 + rsqrt 精度敏感）
    q_raw_lo = tl.load(q_ptr_lo).to(tl.float32)
    q_raw_hi = tl.load(q_ptr_hi).to(tl.float32)

    # Q - RMS Norm (推理场景 HEAD_DIM=128 累加量小，FP16 足够)
    qw_lo = tl.load(q_norm_w_ptr + offs_lo)
    qw_hi = tl.load(q_norm_w_ptr + offs_hi)
    q_var = (tl.sum(q_raw_lo * q_raw_lo) + tl.sum(q_raw_hi * q_raw_hi)) / HEAD_DIM
    q_rsqrt = tl.math.rsqrt(q_var + eps)
    q_norm_lo = q_raw_lo * q_rsqrt * qw_lo
    q_norm_hi = q_raw_hi * q_rsqrt * qw_hi

    # Q - RoPE 并且融合 sm_scale
    q_rope_lo = (q_norm_lo * cos_lo - q_norm_hi * sin_lo) * sm_scale
    q_rope_hi = (q_norm_hi * cos_lo + q_norm_lo * sin_lo) * sm_scale

    # 写入临时的 Q Buffer，供后续 Attention Kernel 读取
    q_out_ptr_lo = q_out_ptr + q_head_idx * stride_oqh + offs_lo * stride_oqd
    q_out_ptr_hi = q_out_ptr + q_head_idx * stride_oqh + offs_hi * stride_oqd
    tl.store(q_out_ptr_lo, q_rope_lo.to(q_out_ptr.dtype.element_ty))
    tl.store(q_out_ptr_hi, q_rope_hi.to(q_out_ptr.dtype.element_ty))

    # ==========================================
    # 2. 计算 K 和 V，并写入 Cache (GQA 去重保护)
    # ==========================================
    if q_head_idx % gqa_group == 0:
        # Load K
        k_ptr_lo = k_raw_ptr + kv_head_idx * stride_kh + offs_lo * stride_kd
        k_ptr_hi = k_raw_ptr + kv_head_idx * stride_kh + offs_hi * stride_kd
        k_raw_lo = tl.load(k_ptr_lo).to(tl.float32)
        k_raw_hi = tl.load(k_ptr_hi).to(tl.float32)

        # K - RMS Norm
        kw_lo = tl.load(k_norm_w_ptr + offs_lo)
        kw_hi = tl.load(k_norm_w_ptr + offs_hi)
        k_var = (tl.sum(k_raw_lo * k_raw_lo) + tl.sum(k_raw_hi * k_raw_hi)) / HEAD_DIM
        k_rsqrt = tl.math.rsqrt(k_var + eps)
        k_norm_lo = k_raw_lo * k_rsqrt * kw_lo
        k_norm_hi = k_raw_hi * k_rsqrt * kw_hi

        # K - RoPE (不乘 sm_scale)
        k_rope_lo = k_norm_lo * cos_lo - k_norm_hi * sin_lo
        k_rope_hi = k_norm_hi * cos_lo + k_norm_lo * sin_lo

        # Load V (直接读取，无需运算)
        v_ptr_lo = v_raw_ptr + kv_head_idx * stride_vh + offs_lo * stride_vd
        v_ptr_hi = v_raw_ptr + kv_head_idx * stride_vh + offs_hi * stride_vd
        v_lo = tl.load(v_ptr_lo)
        v_hi = tl.load(v_ptr_hi)

        # 写入 KV Cache
        k_cache_ptr_lo = k_cache_ptr + seq_idx * stride_kc_s + kv_head_idx * stride_kc_h + offs_lo * stride_kc_d
        k_cache_ptr_hi = k_cache_ptr + seq_idx * stride_kc_s + kv_head_idx * stride_kc_h + offs_hi * stride_kc_d
        tl.store(k_cache_ptr_lo, k_rope_lo.to(k_cache_ptr.dtype.element_ty))
        tl.store(k_cache_ptr_hi, k_rope_hi.to(k_cache_ptr.dtype.element_ty))

        v_cache_ptr_lo = v_cache_ptr + seq_idx * stride_vc_s + kv_head_idx * stride_vc_h + offs_lo * stride_vc_d
        v_cache_ptr_hi = v_cache_ptr + seq_idx * stride_vc_s + kv_head_idx * stride_vc_h + offs_hi * stride_vc_d
        tl.store(v_cache_ptr_lo, v_lo.to(v_cache_ptr.dtype.element_ty))
        tl.store(v_cache_ptr_hi, v_hi.to(v_cache_ptr.dtype.element_ty))


def algebra_prep_qkv_and_cache(
    qkv_states,  # [1, 1, 4096]
    position_embeddings,  # [1, 1, 64], [1, 1, 64]
    q_norm_weight,  # 128
    k_norm_weight,  # 128
    k_cache,  # [1, 4096, 8, 128]
    v_cache,  # [1, 4096, 8, 128]
    cache_seqlens,
    hidden_shape,
    qkv_odim,
    inplace=False,
    q_out=None,
):
    # 分解 Q, K, V (这只是给 Host 拿 stride 用的，底层 Kernel 会用 raw_ptr + offset 直接拿)
    query_states, key_states, value_states = qkv_states.split(qkv_odim, dim=-1)

    q_raw = query_states.reshape(-1, hidden_shape[-1])  # [16, 128]
    k_raw = key_states.reshape(-1, hidden_shape[-1])  # [8, 128]
    v_raw = value_states.reshape(-1, hidden_shape[-1])  # [8, 128]

    k_c = k_cache.squeeze(0)  # [4096, 8, 128]
    v_c = v_cache.squeeze(0)
    cos_1d = position_embeddings[0].flatten()  # [64]
    sin_1d = position_embeddings[1].flatten()

    NUM_Q_HEADS = q_raw.shape[0]  # 16
    NUM_KV_HEADS = k_raw.shape[0]  # 8
    GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 2
    HEAD_DIM = q_raw.shape[-1]  # 128

    if q_out is None:
        q_out = q_raw if inplace else torch.empty_like(q_raw)

    # 启动 1 维 Grid，大小为 Q_HEADS
    grid = (NUM_Q_HEADS,)

    fused_qkv_prep_kernel[grid](
        qkv_states,
        0,  # q_offset
        qkv_odim[0],  # k_offset
        qkv_odim[0] + qkv_odim[1],  # v_offset
        q_norm_weight,  # [128]
        k_norm_weight,  # [128]
        cos_1d,  # [64]
        sin_1d,  # [64]
        k_c,  # [4096, 8, 128]
        v_c,  # [4096, 8, 128]
        q_out,  # 算好的 Q 输出这里
        cache_seqlens,  # 写入 cache 的位置索引
        q_raw.stride(0),  # [16, 128]
        q_raw.stride(1),
        k_raw.stride(0),  # [8, 128]
        k_raw.stride(1),
        v_raw.stride(0),  # [8, 128]
        v_raw.stride(1),
        k_c.stride(0),
        k_c.stride(1),
        k_c.stride(2),
        v_c.stride(0),
        v_c.stride(1),
        v_c.stride(2),
        q_out.stride(0),
        q_out.stride(1),
        eps=1e-6,
        gqa_group=GQA_GROUP,
        HEAD_DIM=HEAD_DIM,
        HALF_HEAD_DIM=HEAD_DIM // 2,
        sm_scale=1.0 / math.sqrt(HEAD_DIM),
    )

    return q_out
