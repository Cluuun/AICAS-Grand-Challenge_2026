import torch
import triton
import triton.language as tl


def rotate_half(x):  # NO NEED
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@triton.jit
def apply_rope_vision_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    stride_q_seq: tl.constexpr,
    stride_q_head: tl.constexpr,
    stride_q_dim: tl.constexpr,
    stride_k_seq: tl.constexpr,
    stride_k_head: tl.constexpr,
    stride_k_dim: tl.constexpr,
    stride_cos_seq: tl.constexpr,
    stride_cos_dim: tl.constexpr,
    stride_sin_seq: tl.constexpr,
    stride_sin_dim: tl.constexpr,
    stride_oq_seq: tl.constexpr,
    stride_oq_head: tl.constexpr,
    stride_oq_dim: tl.constexpr,
    stride_ok_seq: tl.constexpr,
    stride_ok_head: tl.constexpr,
    stride_ok_dim: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    pid_seq = tl.program_id(0)  # SEQ_LEN, 第 0 维
    pid_head = tl.program_id(1)  # NUM_HEADS, 第 1 维

    seq_offs = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    seq_mask = seq_offs < SEQ_LEN

    HALF_DIM: tl.constexpr = HEAD_DIM // 2
    d1_offs = tl.arange(0, HALF_DIM)
    d2_offs = HALF_DIM + tl.arange(0, HALF_DIM)

    q_base = q_ptr + seq_offs[:, None] * stride_q_seq + pid_head * stride_q_head
    q_ptrs_1 = q_base + d1_offs[None, :] * stride_q_dim
    q_ptrs_2 = q_base + d2_offs[None, :] * stride_q_dim

    k_base = k_ptr + seq_offs[:, None] * stride_k_seq + pid_head * stride_k_head
    k_ptrs_1 = k_base + d1_offs[None, :] * stride_k_dim
    k_ptrs_2 = k_base + d2_offs[None, :] * stride_k_dim

    cos_ptrs_1 = cos_ptr + seq_offs[:, None] * stride_cos_seq + d1_offs[None, :] * stride_cos_dim
    cos_ptrs_2 = cos_ptrs_1

    sin_ptrs_1 = sin_ptr + seq_offs[:, None] * stride_sin_seq + d1_offs[None, :] * stride_sin_dim
    sin_ptrs_2 = sin_ptrs_1
    mask_2d = seq_mask[:, None]

    q1 = tl.load(q_ptrs_1, mask=mask_2d, other=0.0).to(tl.float32)
    q2 = tl.load(q_ptrs_2, mask=mask_2d, other=0.0).to(tl.float32)
    k1 = tl.load(k_ptrs_1, mask=mask_2d, other=0.0).to(tl.float32)
    k2 = tl.load(k_ptrs_2, mask=mask_2d, other=0.0).to(tl.float32)

    cos1 = tl.load(cos_ptrs_1, mask=mask_2d, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_ptrs_2, mask=mask_2d, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptrs_1, mask=mask_2d, other=0.0).to(tl.float32)
    sin2 = tl.load(sin_ptrs_2, mask=mask_2d, other=0.0).to(tl.float32)

    out_q1 = q1 * cos1 - q2 * sin1
    out_k1 = k1 * cos1 - k2 * sin1

    out_q2 = q2 * cos2 + q1 * sin2
    out_k2 = k2 * cos2 + k1 * sin2

    oq_base = out_q_ptr + seq_offs[:, None] * stride_oq_seq + pid_head * stride_oq_head
    oq_ptrs_1 = oq_base + d1_offs[None, :] * stride_oq_dim
    oq_ptrs_2 = oq_base + d2_offs[None, :] * stride_oq_dim

    ok_base = out_k_ptr + seq_offs[:, None] * stride_ok_seq + pid_head * stride_ok_head
    ok_ptrs_1 = ok_base + d1_offs[None, :] * stride_ok_dim
    ok_ptrs_2 = ok_base + d2_offs[None, :] * stride_ok_dim

    tl.store(oq_ptrs_1, out_q1.to(tl.float16), mask=mask_2d)
    tl.store(oq_ptrs_2, out_q2.to(tl.float16), mask=mask_2d)

    tl.store(ok_ptrs_1, out_k1.to(tl.float16), mask=mask_2d)
    tl.store(ok_ptrs_2, out_k2.to(tl.float16), mask=mask_2d)


@torch.library.custom_op("qwen3vl::apply_rotary_pos_emb_vision_triton", mutates_args=())
def apply_rotary_pos_emb_vision_triton(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inplace: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # q,k [2688, 16, 64]
    # cos, sin [2688, 64]
    out_q = q if inplace else torch.empty_like(q)
    out_k = k if inplace else torch.empty_like(k)

    SEQ_LEN, NUM_HEADS, HEAD_DIM = q.shape
    BLOCK_SEQ = 64
    grid = (triton.cdiv(SEQ_LEN, BLOCK_SEQ), NUM_HEADS)

    apply_rope_vision_kernel[grid](
        q,
        k,
        cos,
        sin,
        out_q,
        out_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out_q.stride(0),
        out_q.stride(1),
        out_q.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        SEQ_LEN,
        HEAD_DIM=HEAD_DIM,
        BLOCK_SEQ=BLOCK_SEQ,
    )
    return out_q, out_k


@apply_rotary_pos_emb_vision_triton.register_fake
def _fake_apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inplace: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.  # [1, 16, 690, 128] 或者 [1, 16, 1, 128]
        k (`torch.Tensor`): The key tensor.    # [1, 8, 690, 128] 或者 [1, 8, 1, 128]
        cos (`torch.Tensor`): The cosine part of the rotary embedding.  # [1, 690, 128] 或者 [1, 1, 128]
        sin (`torch.Tensor`): The sine part of the rotary embedding.    # [1, 690, 128] 或者 [1, 1, 128]
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_s: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    # sin/cos 的 strides 依然是 B, S, D
    stride_s_b: tl.constexpr,
    stride_s_s: tl.constexpr,
    stride_s_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_s: tl.constexpr,
    stride_c_d: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_bs = tl.program_id(0)  # 融合了 batch 和 seq_len
    head_idx = tl.program_id(1)  # 当前处理的 head 索引

    batch_idx = pid_bs // seq_len
    seq_idx = pid_bs % seq_len

    x_offset = batch_idx * stride_x_b + seq_idx * stride_x_s + head_idx * stride_x_h
    o_offset = batch_idx * stride_o_b + seq_idx * stride_o_s + head_idx * stride_o_h

    c_offset = batch_idx * stride_c_b + seq_idx * stride_c_s
    s_offset = batch_idx * stride_s_b + seq_idx * stride_s_s

    x_row_ptr = x_ptr + x_offset
    out_row_ptr = out_ptr + o_offset
    c_row_ptr = cos_ptr + c_offset
    s_row_ptr = sin_ptr + s_offset

    half_dim = HEAD_DIM // 2

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    x1 = tl.load(x_row_ptr + offsets * stride_x_d, mask=mask, other=0.0)
    x2 = tl.load(x_row_ptr + (half_dim + offsets) * stride_x_d, mask=mask, other=0.0)

    c1 = tl.load(c_row_ptr + offsets * stride_c_d, mask=mask, other=0.0)
    c2 = tl.load(c_row_ptr + (half_dim + offsets) * stride_c_d, mask=mask, other=0.0)

    s1 = tl.load(s_row_ptr + offsets * stride_s_d, mask=mask, other=0.0)
    s2 = tl.load(s_row_ptr + (half_dim + offsets) * stride_s_d, mask=mask, other=0.0)

    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2

    tl.store(out_row_ptr + offsets * stride_o_d, out1, mask=mask)
    tl.store(out_row_ptr + (half_dim + offsets) * stride_o_d, out2, mask=mask)


# TODO: 所有的 triton kernel 都不再需要 mask，都是 2^n 直接特化


def apply_rotary_pos_emb_triton(q, k, cos, sin, inplace=False):
    """
    高效的 Triton 封装接口。
    默认输入形状：
      - q, k: (batch_size, seq_len, num_heads, head_dim)
      - cos, sin: (batch_size, seq_len, head_dim)
    """

    # 确保最后一维大小一致
    assert q.shape[-1] == k.shape[-1] == cos.shape[-1] == sin.shape[-1]

    batch_size, seq_len, n_q_heads, head_dim = q.shape
    _, _, n_k_heads, _ = k.shape

    # 准备输出张量 (支持 In-place 原地修改，极大节省显存)
    q_out = q if inplace else torch.empty_like(q)
    k_out = k if inplace else torch.empty_like(k)

    # Triton 的块大小必须是 2 的幂次方，我们按半个 head_dim 来分配
    BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    # 启动 Q 的 Kernel
    grid_q = (batch_size * seq_len, n_q_heads)
    rope_kernel[grid_q](
        q,
        cos,
        sin,
        q_out,
        seq_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  # q.shape: [1, 690, 16, 128]
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        q_out.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),  # sin.shape: [1, 690, 128]
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),  # cos.shape: [1, 690, 128]
        HEAD_DIM=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 启动 K 的 Kernel (单独启动完美兼容 GQA/MQA，即 Q 和 K 的 head 数量不同)
    grid_k = (batch_size * seq_len, n_k_heads)
    rope_kernel[grid_k](
        k,
        cos,
        sin,
        k_out,
        seq_len,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        k_out.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return q_out, k_out


@triton.jit
def fused_rmsnorm_rope_kernel_bs1(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    num_heads: tl.constexpr,  # 用于在一维 grid 中推导 seq_idx 和 head_idx
    stride_x_s: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_d: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    stride_s_s: tl.constexpr,
    stride_s_d: tl.constexpr,
    stride_c_s: tl.constexpr,
    stride_c_d: tl.constexpr,
    eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_BLOCK_SIZE: tl.constexpr,
):
    # BS=1 特化：使用一维 Grid
    pid = tl.program_id(0)

    # 直接从一维的 pid 中解构出当前处理的 seq 和 head
    seq_idx = pid // num_heads
    head_idx = pid % num_heads

    # 剔除了 Batch 的偏移计算，指令更少
    x_row_ptr = x_ptr + seq_idx * stride_x_s + head_idx * stride_x_h
    out_row_ptr = out_ptr + seq_idx * stride_o_s + head_idx * stride_o_h

    # cos / sin 在 BS=1 时，通常 shape 是 [1, seq_len, head_dim]
    c_row_ptr = cos_ptr + seq_idx * stride_c_s
    s_row_ptr = sin_ptr + seq_idx * stride_s_s

    half_dim = HEAD_DIM // 2

    offsets = tl.arange(0, HALF_BLOCK_SIZE)
    mask = offsets < half_dim

    # 1. 拆分加载 X
    x1 = tl.load(x_row_ptr + offsets * stride_x_d, mask=mask, other=0.0)
    x2 = tl.load(x_row_ptr + (half_dim + offsets) * stride_x_d, mask=mask, other=0.0)

    # 2. 计算 RMSNorm 的 Variance (FP32精度)
    x1_fp32 = x1.to(tl.float32)
    x2_fp32 = x2.to(tl.float32)

    var1 = tl.sum(x1_fp32 * x1_fp32, axis=0)
    var2 = tl.sum(x2_fp32 * x2_fp32, axis=0)
    variance = (var1 + var2) / HEAD_DIM
    rsqrt = tl.math.rsqrt(variance + eps)

    x1_normed = x1_fp32 * rsqrt
    x2_normed = x2_fp32 * rsqrt

    # 3. 乘上 RMSNorm 权重
    if w_ptr:
        w_row_ptr = w_ptr + head_idx * stride_w_h
        w1 = tl.load(w_row_ptr + offsets * stride_w_d, mask=mask, other=0.0)
        w2 = tl.load(w_row_ptr + (half_dim + offsets) * stride_w_d, mask=mask, other=0.0)
        x1_normed = x1_normed * w1
        x2_normed = x2_normed * w2

    x1_normed = x1_normed.to(x1.dtype)
    x2_normed = x2_normed.to(x2.dtype)

    # 4. 加载 RoPE 参数
    c1 = tl.load(c_row_ptr + offsets * stride_c_d, mask=mask, other=0.0)
    c2 = c1
    s1 = tl.load(s_row_ptr + offsets * stride_s_d, mask=mask, other=0.0)
    s2 = s1

    # 5. 寄存器内完成 RoPE
    out1 = x1_normed * c1 - x2_normed * s1
    out2 = x2_normed * c2 + x1_normed * s2

    # 6. 一次性写回
    tl.store(out_row_ptr + offsets * stride_o_d, out1, mask=mask)
    tl.store(out_row_ptr + (half_dim + offsets) * stride_o_d, out2, mask=mask)


@torch.library.custom_op("qwen3vl::fused_rms_norm_rope_triton_bs1", mutates_args=())
def fused_rms_norm_rope_triton_bs1(
    query_states: torch.Tensor,
    position_embeddings: list[torch.Tensor],
    q_norm_weight: torch.Tensor,
    hidden_shape: list[int],
    eps: float = 1e-6,
    inplace: bool = False,
) -> torch.Tensor:
    """Q-only RMSNorm + RoPE. K is now handled by _launch_k_rope_to_cache."""
    q = query_states.view(hidden_shape)  # [1, seq_len, 16, 128]

    batch_size, seq_len, n_q_heads, head_dim = q.shape

    cos, sin = position_embeddings

    q_out = q if inplace else torch.empty_like(q)

    HALF_BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    if q_norm_weight is None:
        q_sw_h, q_sw_d = 0, 0
    elif q_norm_weight.numel() == head_dim:
        q_sw_h, q_sw_d = 0, q_norm_weight.stride(0)
    else:
        q_sw_h, q_sw_d = head_dim, 1

    grid_q = (seq_len * n_q_heads,)

    fused_rmsnorm_rope_kernel_bs1[grid_q](
        q,
        q_norm_weight,
        cos,
        sin,
        q_out,
        n_q_heads,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q_sw_h,
        q_sw_d,
        q_out.stride(1),
        q_out.stride(2),
        q_out.stride(3),
        sin.stride(1),
        sin.stride(2),
        cos.stride(1),
        cos.stride(2),
        eps,
        HEAD_DIM=head_dim,
        HALF_BLOCK_SIZE=HALF_BLOCK_SIZE,
        num_warps=4,
    )

    return q_out


@fused_rms_norm_rope_triton_bs1.register_fake
def _fake_fused_rms_norm_rope_triton_bs1(
    query_states: torch.Tensor,
    position_embeddings: list[torch.Tensor],
    q_norm_weight: torch.Tensor,
    hidden_shape: list[int],
    eps: float = 1e-6,
    inplace: bool = False,
) -> torch.Tensor:
    return torch.empty_like(query_states).reshape(*hidden_shape)


def _launch_k_rope_to_cache(
    key_states: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    seq_len: int,
    hidden_shape: list[int],
    head_dim: int,
    eps: float = 1e-6,
):
    """Launch K RoPE kernel writing directly into k_cache[:, :seq_len], skipping temp alloc + copy."""
    k = key_states.view(hidden_shape)  # [1, seq_len, n_k_heads, head_dim]
    n_k_heads = k.shape[2]

    # k_cache slice: k_cache[:, :seq_len, :, :] — same seq/head/dim strides as k_cache
    # k_cache stride: (max_seq*n_k_heads*head_dim, n_k_heads*head_dim, head_dim, 1)
    # We use k_cache strides directly (stride(1), stride(2), stride(3)) — they're correct
    # because the kernel indexes via seq_idx * stride_o_s, which is along dim 1.

    HALF_BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    if k_norm_weight is None:
        k_sw_h, k_sw_d = 0, 0
    elif k_norm_weight.numel() == head_dim:
        k_sw_h, k_sw_d = 0, k_norm_weight.stride(0)
    else:
        k_sw_h, k_sw_d = head_dim, 1

    grid_k = (seq_len * n_k_heads,)
    fused_rmsnorm_rope_kernel_bs1[grid_k](
        k,
        k_norm_weight,
        cos,
        sin,
        k_cache,  # write directly into k_cache (base pointer at [0, 0, 0, 0])
        n_k_heads,
        k.stride(1),
        k.stride(2),
        k.stride(3),
        k_sw_h,
        k_sw_d,
        k_cache.stride(1),  # stride_o_s: same as k_out would have (n_k_heads * head_dim)
        k_cache.stride(2),  # stride_o_h: head_dim
        k_cache.stride(3),  # stride_o_d: 1
        sin.stride(1),
        sin.stride(2),
        cos.stride(1),
        cos.stride(2),
        eps,
        HEAD_DIM=head_dim,
        HALF_BLOCK_SIZE=HALF_BLOCK_SIZE,
        num_warps=4,
    )


def _launch_rms_norm_rope_to_out(
    x_states: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out: torch.Tensor,
    head_dim: int,
    eps: float = 1e-6,
):
    """RMSNorm + RoPE into caller-provided output.

    This is used by language prefill layers whose KV cache is not consumed by
    decode layer-skip. They still need K for the current prefill attention, but
    they do not need to dirty the persistent KV cache.
    """
    x = x_states.view(out.shape)
    n_heads = x.shape[2]

    HALF_BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)

    if norm_weight is None:
        sw_h, sw_d = 0, 0
    elif norm_weight.numel() == head_dim:
        sw_h, sw_d = 0, norm_weight.stride(0)
    else:
        sw_h, sw_d = head_dim, 1

    grid = (x.shape[1] * n_heads,)
    fused_rmsnorm_rope_kernel_bs1[grid](
        x,
        norm_weight,
        cos,
        sin,
        out,
        n_heads,
        x.stride(1),
        x.stride(2),
        x.stride(3),
        sw_h,
        sw_d,
        out.stride(1),
        out.stride(2),
        out.stride(3),
        sin.stride(1),
        sin.stride(2),
        cos.stride(1),
        cos.stride(2),
        eps,
        HEAD_DIM=head_dim,
        HALF_BLOCK_SIZE=HALF_BLOCK_SIZE,
        num_warps=4,
    )
    return out
