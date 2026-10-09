"""triton核心 triton生成的cuda代码天然支持cuda图"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    X_PTR, W_PTR, OUT_PTR,
    n_rows,
    EPS,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    x = tl.load(X_PTR + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_PTR + cols, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / H
    inv_var = tl.math.rsqrt(variance + EPS)

    out = x * inv_var * w
    tl.store(OUT_PTR + row * H + cols, out, mask=mask)


def triton_rms_norm(hidden_states, weight, epsilon):
    """RMS归一化"""
    orig_shape = hidden_states.shape
    H = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, H)
    n_rows = x.shape[0]

    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(H)
    grid = (n_rows,)

    # 自适应warps数 小行数用少点warps更高效
    if H <= 128:
        w = 1  # 128个元素32个线程 每线程处理4个
    elif H <= 256:
        w = 2
    elif H <= 1024:
        w = 4
    else:
        w = 8

    _rms_norm_kernel[grid](
        x, weight, out,
        n_rows, epsilon,
        H=H, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=w,
    )

    return out.reshape(orig_shape)


@triton.jit
def _silu_and_mul_kernel(
    GATE_UP_PTR, OUT_PTR,
    n_rows,
    HALF_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n_rows:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < HALF_D

    gate = tl.load(GATE_UP_PTR + row * 2 * HALF_D + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GATE_UP_PTR + row * 2 * HALF_D + HALF_D + cols, mask=mask, other=0.0).to(tl.float32)

    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) 也就是swish激活
    silu_gate = gate * tl.sigmoid(gate)
    out = silu_gate * up

    tl.store(OUT_PTR + row * HALF_D + cols, out, mask=mask)


def triton_silu_and_mul(gate_up):
    """SiLU门控 前半过激活再乘后半"""
    orig_shape = gate_up.shape
    d = gate_up.shape[-1] // 2
    x = gate_up.reshape(-1, 2 * d)
    n_rows = x.shape[0]

    out = torch.empty((n_rows, d), dtype=gate_up.dtype, device=gate_up.device)

    BLOCK_SIZE = triton.next_power_of_2(d)
    grid = (n_rows,)

    # 根据半维度自适应warps数
    if d <= 128:
        w = 1
    elif d <= 512:
        w = 2
    elif d <= 1024:
        w = 4
    else:
        w = 8

    _silu_and_mul_kernel[grid](
        x, out,
        n_rows,
        HALF_D=d, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=w,
    )

    return out.reshape(orig_shape[:-1] + (d,))


@triton.jit
def _fused_add_rms_norm_kernel(
    INPUT_PTR, RESIDUAL_PTR, W_PTR, OUT_PTR, RES_OUT_PTR,
    n_rows,
    EPS,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """融合残差加+RMS归一化"""
    row = tl.program_id(0)
    if row >= n_rows:
        return

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    inp = tl.load(INPUT_PTR + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(RESIDUAL_PTR + row * H + cols, mask=mask, other=0.0).to(tl.float32)

    # 残差相加
    res = res + inp

    # 存更新后的残差到独立缓冲区 不原地改
    tl.store(RES_OUT_PTR + row * H + cols, res, mask=mask)

    # RMS归一化
    w = tl.load(W_PTR + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(res * res, axis=0) / H
    inv_var = tl.math.rsqrt(variance + EPS)
    out = res * inv_var * w

    # 存归一化结果
    tl.store(OUT_PTR + row * H + cols, out, mask=mask)


def triton_fused_add_rms_norm(input_tensor, residual, weight, epsilon):
    """融合残差加+RMS归一化 非原地操作 torch.compile友好"""
    orig_shape = input_tensor.shape
    H = input_tensor.shape[-1]
    inp = input_tensor.reshape(-1, H)
    res = residual.reshape(-1, H)
    n_rows = inp.shape[0]

    out = torch.empty_like(inp)
    res_out = torch.empty_like(res)

    BLOCK_SIZE = triton.next_power_of_2(H)
    grid = (n_rows,)

    _fused_add_rms_norm_kernel[grid](
        inp, res, weight, out, res_out,
        n_rows, epsilon,
        H=H, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return out.reshape(orig_shape), res_out.reshape(orig_shape)


@triton.jit
def _fused_rope_kernel(
    Q_PTR, K_PTR, COS_PTR, SIN_PTR,
    Q_OUT_PTR, K_OUT_PTR,
    n_heads_q, n_heads_kv,
    HEAD_DIM: tl.constexpr,
):
    """decode用RoPE Q和K在一个核心里搞定"""
    pid = tl.program_id(0)
    HALF_DIM: tl.constexpr = HEAD_DIM // 2

    # 加载cos/sin 广播的 decode时所有头共用
    off_first = tl.arange(0, HALF_DIM)
    off_second = HALF_DIM + tl.arange(0, HALF_DIM)
    cos_f = tl.load(COS_PTR + off_first).to(tl.float32)
    cos_s = tl.load(COS_PTR + off_second).to(tl.float32)
    sin_f = tl.load(SIN_PTR + off_first).to(tl.float32)
    sin_s = tl.load(SIN_PTR + off_second).to(tl.float32)

    # 处理Q头
    if pid < n_heads_q:
        base = pid * HEAD_DIM
        x_first = tl.load(Q_PTR + base + off_first).to(tl.float32)
        x_second = tl.load(Q_PTR + base + off_second).to(tl.float32)
        # rotate_half: [-x_second, x_first]
        # 结果 = x * cos + rotated * sin
        out_first = x_first * cos_f + (-x_second) * sin_f
        out_second = x_second * cos_s + x_first * sin_s
        tl.store(Q_OUT_PTR + base + off_first, out_first)
        tl.store(Q_OUT_PTR + base + off_second, out_second)

    # 处理K头
    if pid < n_heads_kv:
        base = pid * HEAD_DIM
        x_first = tl.load(K_PTR + base + off_first).to(tl.float32)
        x_second = tl.load(K_PTR + base + off_second).to(tl.float32)
        out_first = x_first * cos_f + (-x_second) * sin_f
        out_second = x_second * cos_s + x_first * sin_s
        tl.store(K_OUT_PTR + base + off_first, out_first)
        tl.store(K_OUT_PTR + base + off_second, out_second)


def triton_fused_rope(query_states, key_states, cos, sin):
    """融合RoPE Q和K一次搞定 替代原来每层~12个GPU操作"""
    orig_q_shape = query_states.shape
    orig_k_shape = key_states.shape

    batch, n_heads_q, seq, head_dim = query_states.shape
    _, n_heads_kv, _, _ = key_states.shape

    q = query_states.reshape(n_heads_q, head_dim)
    k = key_states.reshape(n_heads_kv, head_dim)

    # cos/sin: [1, 1, 1, 128] → 展平成 [128]
    c = cos.reshape(head_dim)
    s = sin.reshape(head_dim)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = (max(n_heads_q, n_heads_kv),)

    _fused_rope_kernel[grid](
        q, k, c, s,
        q_out, k_out,
        n_heads_q, n_heads_kv,
        HEAD_DIM=head_dim,
        num_warps=4,
    )

    return q_out.reshape(orig_q_shape), k_out.reshape(orig_k_shape)


# ---------------------------------------------------------------------------
# prefill用融合RoPE (seq_len > 1)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_rope_prefill_kernel(
    Q_PTR, K_PTR, COS_PTR, SIN_PTR,
    Q_OUT_PTR, K_OUT_PTR,
    seq_len,
    n_heads_q: tl.constexpr,
    n_heads_kv: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """prefill用RoPE 每个program处理一个(头 位置)对"""
    pid = tl.program_id(0)
    HALF_DIM: tl.constexpr = HEAD_DIM // 2

    head = pid // seq_len
    pos = pid % seq_len

    off_first = tl.arange(0, HALF_DIM)
    off_second = HALF_DIM + tl.arange(0, HALF_DIM)

    # 加载该位置的cos/sin: [seq_len, head_dim]
    cos_pos = pos * HEAD_DIM
    cos_f = tl.load(COS_PTR + cos_pos + off_first).to(tl.float32)
    cos_s = tl.load(COS_PTR + cos_pos + off_second).to(tl.float32)
    sin_f = tl.load(SIN_PTR + cos_pos + off_first).to(tl.float32)
    sin_s = tl.load(SIN_PTR + cos_pos + off_second).to(tl.float32)

    # 处理Q头: [n_heads_q, seq_len, head_dim]
    if head < n_heads_q:
        q_base = head * seq_len * HEAD_DIM + pos * HEAD_DIM
        x_first = tl.load(Q_PTR + q_base + off_first).to(tl.float32)
        x_second = tl.load(Q_PTR + q_base + off_second).to(tl.float32)
        out_first = x_first * cos_f + (-x_second) * sin_f
        out_second = x_second * cos_s + x_first * sin_s
        tl.store(Q_OUT_PTR + q_base + off_first, out_first)
        tl.store(Q_OUT_PTR + q_base + off_second, out_second)

    # 处理K头: [n_heads_kv, seq_len, head_dim]
    if head < n_heads_kv:
        k_base = head * seq_len * HEAD_DIM + pos * HEAD_DIM
        x_first = tl.load(K_PTR + k_base + off_first).to(tl.float32)
        x_second = tl.load(K_PTR + k_base + off_second).to(tl.float32)
        out_first = x_first * cos_f + (-x_second) * sin_f
        out_second = x_second * cos_s + x_first * sin_s
        tl.store(K_OUT_PTR + k_base + off_first, out_first)
        tl.store(K_OUT_PTR + k_base + off_second, out_second)


def triton_fused_rope_prefill(query_states, key_states, cos, sin):
    """prefill用融合RoPE 一个核心搞定所有位置"""
    orig_q_shape = query_states.shape
    orig_k_shape = key_states.shape

    batch, n_heads_q, seq_len, head_dim = query_states.shape
    _, n_heads_kv, _, _ = key_states.shape

    # 重塑成 [n_heads, seq_len, head_dim] (batch=1)
    q = query_states.reshape(n_heads_q, seq_len * head_dim)
    k = key_states.reshape(n_heads_kv, seq_len * head_dim)
    c = cos.reshape(seq_len, head_dim)
    s = sin.reshape(seq_len, head_dim)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = (max(n_heads_q, n_heads_kv) * seq_len,)

    _fused_rope_prefill_kernel[grid](
        q, k, c, s,
        q_out, k_out,
        seq_len,
        n_heads_q=n_heads_q, n_heads_kv=n_heads_kv,
        HEAD_DIM=head_dim,
        num_warps=4,
    )

    return q_out.reshape(orig_q_shape), k_out.reshape(orig_k_shape)


# ---------------------------------------------------------------------------
# 视觉编码器用融合RoPE (内存布局不同: [seq, heads, dim])
# ---------------------------------------------------------------------------

@triton.jit
def _vision_rope_kernel(
    Q_PTR, K_PTR, COS_PTR, SIN_PTR,
    Q_OUT_PTR, K_OUT_PTR,
    seq_len,
    n_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """视觉编码器RoPE Q和K一起搞 布局是[seq, heads, dim]"""
    pid = tl.program_id(0)
    HALF_DIM: tl.constexpr = HEAD_DIM // 2

    head = pid // seq_len
    pos = pid % seq_len

    off_first = tl.arange(0, HALF_DIM)
    off_second = HALF_DIM + tl.arange(0, HALF_DIM)

    # 加载该位置的cos/sin: [seq_len, head_dim]
    cos_off = pos * HEAD_DIM
    cos_f = tl.load(COS_PTR + cos_off + off_first).to(tl.float32)
    cos_s = tl.load(COS_PTR + cos_off + off_second).to(tl.float32)
    sin_f = tl.load(SIN_PTR + cos_off + off_first).to(tl.float32)
    sin_s = tl.load(SIN_PTR + cos_off + off_second).to(tl.float32)

    # 内存布局: [seq_len, num_heads, head_dim]
    # 步长: seq_stride = n_heads * HEAD_DIM, head_stride = HEAD_DIM
    base = pos * n_heads * HEAD_DIM + head * HEAD_DIM

    # 处理Q
    x_first = tl.load(Q_PTR + base + off_first).to(tl.float32)
    x_second = tl.load(Q_PTR + base + off_second).to(tl.float32)
    out_first = x_first * cos_f + (-x_second) * sin_f
    out_second = x_second * cos_s + x_first * sin_s
    tl.store(Q_OUT_PTR + base + off_first, out_first)
    tl.store(Q_OUT_PTR + base + off_second, out_second)

    # 处理K (视觉编码器里K和Q形状一样)
    k_first = tl.load(K_PTR + base + off_first).to(tl.float32)
    k_second = tl.load(K_PTR + base + off_second).to(tl.float32)
    k_out_first = k_first * cos_f + (-k_second) * sin_f
    k_out_second = k_second * cos_s + k_first * sin_s
    tl.store(K_OUT_PTR + base + off_first, k_out_first)
    tl.store(K_OUT_PTR + base + off_second, k_out_second)


def triton_vision_rope(query_states, key_states, cos, sin):
    """视觉编码器融合RoPE 布局是[seq, heads, dim] 没有batch维度"""
    seq_len, num_heads, head_dim = query_states.shape

    # 调用方保证Q/K已连续（_make_patched_vision_attn_forward用3个独立mm）
    q = query_states if query_states.is_contiguous() else query_states.contiguous()
    k = key_states if key_states.is_contiguous() else key_states.contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    grid = (num_heads * seq_len,)

    _vision_rope_kernel[grid](
        q, k, cos, sin,
        q_out, k_out,
        seq_len,
        n_heads=num_heads,
        HEAD_DIM=head_dim,
        num_warps=4,
    )

    return q_out, k_out


# ---------------------------------------------------------------------------
# 融合QKV拆分+Q/K RMS归一化+RoPE (decode专用 seq_len=1)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_qkv_norm_rope_kernel(
    QKV_PTR, Q_W_PTR, K_W_PTR,
    COS_PTR, SIN_PTR,
    Q_OUT_PTR, K_OUT_PTR, V_OUT_PTR,
    Q_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    Q_EPS,
    K_EPS,
):
    """融合QKV拆分+Q/K归一化+RoPE 一次搞定"""
    pid = tl.program_id(0)

    off_first = tl.arange(0, HALF_DIM)
    off_second = HALF_DIM + tl.arange(0, HALF_DIM)

    # cos/sin只加载一次 所有头共用
    cos_f = tl.load(COS_PTR + off_first).to(tl.float32)
    cos_s = tl.load(COS_PTR + off_second).to(tl.float32)
    sin_f = tl.load(SIN_PTR + off_first).to(tl.float32)
    sin_s = tl.load(SIN_PTR + off_second).to(tl.float32)

    # --- 处理Q头 ---
    if pid < N_Q_HEADS:
        base = pid * HEAD_DIM
        q_first = tl.load(QKV_PTR + base + off_first).to(tl.float32)
        q_second = tl.load(QKV_PTR + base + off_second).to(tl.float32)
        qw_first = tl.load(Q_W_PTR + off_first).to(tl.float32)
        qw_second = tl.load(Q_W_PTR + off_second).to(tl.float32)

        # RMS归一化
        var = (tl.sum(q_first * q_first) + tl.sum(q_second * q_second)) / HEAD_DIM
        inv_var = tl.math.rsqrt(var + Q_EPS)
        q_nf = q_first * inv_var * qw_first
        q_ns = q_second * inv_var * qw_second

        # RoPE
        out_first = q_nf * cos_f + (-q_ns) * sin_f
        out_second = q_ns * cos_s + q_nf * sin_s
        tl.store(Q_OUT_PTR + pid * HEAD_DIM + off_first, out_first)
        tl.store(Q_OUT_PTR + pid * HEAD_DIM + off_second, out_second)

    # --- 处理K头 + 拷贝V ---
    if pid < N_KV_HEADS:
        k_base = Q_DIM + pid * HEAD_DIM
        k_first = tl.load(QKV_PTR + k_base + off_first).to(tl.float32)
        k_second = tl.load(QKV_PTR + k_base + off_second).to(tl.float32)
        kw_first = tl.load(K_W_PTR + off_first).to(tl.float32)
        kw_second = tl.load(K_W_PTR + off_second).to(tl.float32)

        # RMS归一化
        var = (tl.sum(k_first * k_first) + tl.sum(k_second * k_second)) / HEAD_DIM
        inv_var = tl.math.rsqrt(var + K_EPS)
        k_nf = k_first * inv_var * kw_first
        k_ns = k_second * inv_var * kw_second

        # RoPE
        out_first = k_nf * cos_f + (-k_ns) * sin_f
        out_second = k_ns * cos_s + k_nf * sin_s
        tl.store(K_OUT_PTR + pid * HEAD_DIM + off_first, out_first)
        tl.store(K_OUT_PTR + pid * HEAD_DIM + off_second, out_second)

        # 拷贝V 不归一化不加RoPE
        v_base = Q_DIM + KV_DIM + pid * HEAD_DIM
        full_off = tl.arange(0, HEAD_DIM)
        v_data = tl.load(QKV_PTR + v_base + full_off).to(tl.float32)
        tl.store(V_OUT_PTR + pid * HEAD_DIM + full_off, v_data)


def triton_fused_qkv_norm_rope(qkv, q_weight, k_weight, cos, sin,
                                q_dim, kv_dim, head_dim,
                                eps_q=1e-6, eps_k=1e-6):
    """decode用融合QKV拆分+归一化+RoPE 一次搞定 直接输出flash_attn格式"""
    n_q_heads = q_dim // head_dim
    n_kv_heads = kv_dim // head_dim

    batch = qkv.shape[0]
    seq = qkv.shape[1]

    qkv_flat = qkv.reshape(-1)
    c = cos.reshape(head_dim)
    s = sin.reshape(head_dim)

    q_out = torch.empty(n_q_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(n_kv_heads, head_dim, dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty(n_kv_heads, head_dim, dtype=qkv.dtype, device=qkv.device)

    grid = (max(n_q_heads, n_kv_heads),)

    _fused_qkv_norm_rope_kernel[grid](
        qkv_flat, q_weight, k_weight,
        c, s,
        q_out, k_out, v_out,
        Q_DIM=q_dim, KV_DIM=kv_dim, HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        N_Q_HEADS=n_q_heads, N_KV_HEADS=n_kv_heads,
        Q_EPS=eps_q, K_EPS=eps_k,
        num_warps=2,
    )

    return (q_out.reshape(batch, seq, n_q_heads, head_dim),
            k_out.reshape(batch, seq, n_kv_heads, head_dim),
            v_out.reshape(batch, seq, n_kv_heads, head_dim))


# ---------------------------------------------------------------------------
# 融合QKV拆分+Q/K RMS归一化+RoPE prefill版 (seq_len > 1)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_qkv_norm_rope_prefill_kernel(
    QKV_PTR, Q_W_PTR, K_W_PTR,
    COS_PTR, SIN_PTR,
    Q_OUT_PTR, K_OUT_PTR, V_OUT_PTR,
    seq_len,
    Q_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    Q_EPS,
    K_EPS,
):
    """prefill用融合QKV拆分+归一化+RoPE 每个program处理一个(头,位置)对"""
    pid = tl.program_id(0)

    off_first = tl.arange(0, HALF_DIM)
    off_second = HALF_DIM + tl.arange(0, HALF_DIM)
    full_off = tl.arange(0, HEAD_DIM)

    head = pid // seq_len
    pos = pid % seq_len

    # 加载该位置的cos/sin: [seq_len, head_dim]
    cos_off = pos * HEAD_DIM
    cos_f = tl.load(COS_PTR + cos_off + off_first).to(tl.float32)
    cos_s = tl.load(COS_PTR + cos_off + off_second).to(tl.float32)
    sin_f = tl.load(SIN_PTR + cos_off + off_first).to(tl.float32)
    sin_s = tl.load(SIN_PTR + cos_off + off_second).to(tl.float32)

    # --- 处理Q头 ---
    if head < N_Q_HEADS:
        # QKV布局: [seq_len, Q_DIM + 2*KV_DIM]
        row_base = pos * (Q_DIM + 2 * KV_DIM) + head * HEAD_DIM
        q_first = tl.load(QKV_PTR + row_base + off_first).to(tl.float32)
        q_second = tl.load(QKV_PTR + row_base + off_second).to(tl.float32)
        qw_first = tl.load(Q_W_PTR + off_first).to(tl.float32)
        qw_second = tl.load(Q_W_PTR + off_second).to(tl.float32)

        # RMS归一化
        var = (tl.sum(q_first * q_first) + tl.sum(q_second * q_second)) / HEAD_DIM
        inv_var = tl.math.rsqrt(var + Q_EPS)
        q_nf = q_first * inv_var * qw_first
        q_ns = q_second * inv_var * qw_second

        # RoPE
        out_first = q_nf * cos_f + (-q_ns) * sin_f
        out_second = q_ns * cos_s + q_nf * sin_s

        # 输出: [N_Q_HEADS, seq_len, HEAD_DIM]
        out_base = head * seq_len * HEAD_DIM + pos * HEAD_DIM
        tl.store(Q_OUT_PTR + out_base + off_first, out_first)
        tl.store(Q_OUT_PTR + out_base + off_second, out_second)

    # --- 处理K头 + 拷贝V ---
    if head < N_KV_HEADS:
        k_base_in = pos * (Q_DIM + 2 * KV_DIM) + Q_DIM + head * HEAD_DIM
        k_first = tl.load(QKV_PTR + k_base_in + off_first).to(tl.float32)
        k_second = tl.load(QKV_PTR + k_base_in + off_second).to(tl.float32)
        kw_first = tl.load(K_W_PTR + off_first).to(tl.float32)
        kw_second = tl.load(K_W_PTR + off_second).to(tl.float32)

        # RMS归一化
        var = (tl.sum(k_first * k_first) + tl.sum(k_second * k_second)) / HEAD_DIM
        inv_var = tl.math.rsqrt(var + K_EPS)
        k_nf = k_first * inv_var * kw_first
        k_ns = k_second * inv_var * kw_second

        # RoPE
        out_first = k_nf * cos_f + (-k_ns) * sin_f
        out_second = k_ns * cos_s + k_nf * sin_s

        out_base = head * seq_len * HEAD_DIM + pos * HEAD_DIM
        tl.store(K_OUT_PTR + out_base + off_first, out_first)
        tl.store(K_OUT_PTR + out_base + off_second, out_second)

        # 拷贝V 不归一化不加RoPE
        v_base_in = pos * (Q_DIM + 2 * KV_DIM) + Q_DIM + KV_DIM + head * HEAD_DIM
        v_data = tl.load(QKV_PTR + v_base_in + full_off).to(tl.float32)
        tl.store(V_OUT_PTR + head * seq_len * HEAD_DIM + pos * HEAD_DIM + full_off, v_data)


def triton_fused_qkv_norm_rope_prefill(qkv, q_weight, k_weight, cos, sin,
                                        q_dim, kv_dim, head_dim,
                                        eps_q=1e-6, eps_k=1e-6):
    """prefill用融合QKV拆分+归一化+RoPE 每层一个核心搞定"""
    n_q_heads = q_dim // head_dim
    n_kv_heads = kv_dim // head_dim

    batch = qkv.shape[0]
    seq = qkv.shape[1]

    qkv_flat = qkv.reshape(seq, q_dim + 2 * kv_dim)
    c = cos.reshape(seq, head_dim)
    s = sin.reshape(seq, head_dim)

    q_out = torch.empty(n_q_heads, seq, head_dim, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.empty(n_kv_heads, seq, head_dim, dtype=qkv.dtype, device=qkv.device)
    v_out = torch.empty(n_kv_heads, seq, head_dim, dtype=qkv.dtype, device=qkv.device)

    grid = (max(n_q_heads, n_kv_heads) * seq,)

    _fused_qkv_norm_rope_prefill_kernel[grid](
        qkv_flat, q_weight, k_weight,
        c, s,
        q_out, k_out, v_out,
        seq,
        Q_DIM=q_dim, KV_DIM=kv_dim, HEAD_DIM=head_dim,
        HALF_DIM=head_dim // 2,
        N_Q_HEADS=n_q_heads, N_KV_HEADS=n_kv_heads,
        Q_EPS=eps_q, K_EPS=eps_k,
        num_warps=4,
    )

    # 输出形状: [n_heads, seq, head_dim] → 转置成 [seq, n_heads, head_dim] → [batch, seq, n_heads, head_dim]
    return (q_out.permute(1, 0, 2).reshape(batch, seq, n_q_heads, head_dim),
            k_out.permute(1, 0, 2).reshape(batch, seq, n_kv_heads, head_dim),
            v_out.permute(1, 0, 2).reshape(batch, seq, n_kv_heads, head_dim))


# ---------------------------------------------------------------------------
# 融合 LayerNorm + QKV 投影 (ViT block 专用)
# 把 F.layer_norm(h) → mm(normed, Q_w_T) 合并成一个 kernel
# 每行 h 只读一次，省掉中间 normed 的 HBM 读写
# ---------------------------------------------------------------------------

@triton.jit
def _layernorm_linear_kernel(
    X_PTR,          # [seq, IN_DIM]  输入
    W_LN_PTR,       # [IN_DIM]       LayerNorm weight
    B_LN_PTR,       # [IN_DIM]       LayerNorm bias
    W_LINEAR_PTR,   # [OUT_DIM, IN_DIM]  线性层权重
    B_LINEAR_PTR,   # [OUT_DIM]      线性层 bias (可为 None)
    OUT_PTR,        # [seq, OUT_DIM] 输出
    eps,
    seq_len,
    IN_DIM:  tl.constexpr,
    OUT_DIM: tl.constexpr,
    BLOCK_IN:  tl.constexpr,   # 覆盖 IN_DIM 的 tile
    BLOCK_OUT: tl.constexpr,   # 每个 program 处理的输出列数
    HAS_BIAS: tl.constexpr,
):
    """
    每个 program 处理一行 x（一个 token）的一段输出列 [col_start, col_start+BLOCK_OUT)。
    Phase 1（只有 col==0 的 program 做）：计算 LayerNorm 统计量，写到 shared（用 tl.atomic）。
    实际上 Triton 没有 block 内 shared memory 跨 program 共享，所以每个 program 都独立算 LN。
    代价：IN_DIM 被读 ceil(OUT_DIM/BLOCK_OUT) 次，但省掉了 normed 的 HBM 写+读。
    对 IN_DIM=1024, OUT_DIM=1024 (proj) 或 4096 (fc1)，这是合算的。
    """
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    if row >= seq_len:
        return

    col_start = col_block * BLOCK_OUT
    col_offs = col_start + tl.arange(0, BLOCK_OUT)
    col_mask = col_offs < OUT_DIM

    # --- Phase 1: 计算 LayerNorm (每个 program 独立算，IN_DIM 次读) ---
    in_offs = tl.arange(0, BLOCK_IN)
    in_mask = in_offs < IN_DIM

    x = tl.load(X_PTR + row * IN_DIM + in_offs, mask=in_mask, other=0.0).to(tl.float32)
    w_ln = tl.load(W_LN_PTR + in_offs, mask=in_mask, other=1.0).to(tl.float32)
    b_ln = tl.load(B_LN_PTR + in_offs, mask=in_mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / IN_DIM
    x_c = x - mean
    var = tl.sum(x_c * x_c, axis=0) / IN_DIM
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_c * rstd * w_ln + b_ln   # [BLOCK_IN]

    # --- Phase 2: 线性投影 (dot product with weight rows) ---
    # W_LINEAR: [OUT_DIM, IN_DIM], 取 col_offs 行
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)
    # 逐块累加 dot product
    for k in range(0, IN_DIM, BLOCK_IN):
        k_offs = k + tl.arange(0, BLOCK_IN)
        k_mask = k_offs < IN_DIM
        # 重新加载这段 x_norm（因为 BLOCK_IN 可能 < IN_DIM 时需要分块）
        # 当 BLOCK_IN == IN_DIM 时只循环一次，x_norm 已经算好了
        if k == 0:
            xn = x_norm
        else:
            xr = tl.load(X_PTR + row * IN_DIM + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            wln = tl.load(W_LN_PTR + k_offs, mask=k_mask, other=1.0).to(tl.float32)
            bln = tl.load(B_LN_PTR + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            xn = xr  # 简化：只在 BLOCK_IN==IN_DIM 时正确，下面 constexpr 保证这点

        # W[col_offs, k_offs]: [BLOCK_OUT, BLOCK_IN]
        w = tl.load(
            W_LINEAR_PTR + col_offs[:, None] * IN_DIM + k_offs[None, :],
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * xn[None, :], axis=1)

    if HAS_BIAS:
        b = tl.load(B_LINEAR_PTR + col_offs, mask=col_mask, other=0.0).to(tl.float32)
        acc += b

    tl.store(OUT_PTR + row * OUT_DIM + col_offs, acc.to(tl.float16), mask=col_mask)


def triton_layernorm_linear(x, ln_weight, ln_bias, linear_weight, linear_bias, eps=1e-6):
    """
    融合 LayerNorm + Linear：y = Linear(LayerNorm(x))
    x: [seq, IN_DIM] FP16
    linear_weight: [OUT_DIM, IN_DIM] FP16
    返回: [seq, OUT_DIM] FP16
    """
    seq, IN_DIM = x.shape
    OUT_DIM = linear_weight.shape[0]

    # BLOCK_IN 必须 >= IN_DIM 且是 2 的幂（保证单次循环算完 LN）
    BLOCK_IN = triton.next_power_of_2(IN_DIM)
    # BLOCK_OUT: 每个 program 处理的输出列数，调优点
    BLOCK_OUT = 64

    out = torch.empty(seq, OUT_DIM, dtype=torch.float16, device=x.device)

    grid = (seq, triton.cdiv(OUT_DIM, BLOCK_OUT))

    _layernorm_linear_kernel[grid](
        x, ln_weight, ln_bias,
        linear_weight,
        linear_bias if linear_bias is not None else x,  # dummy ptr
        out,
        eps,
        seq,
        IN_DIM=IN_DIM,
        OUT_DIM=OUT_DIM,
        BLOCK_IN=BLOCK_IN,
        BLOCK_OUT=BLOCK_OUT,
        HAS_BIAS=(linear_bias is not None),
        num_warps=4,
    )
    return out

