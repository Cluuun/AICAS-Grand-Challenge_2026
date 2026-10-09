import torch
import triton
import triton.language as tl


@triton.jit
def _decode_prefix_attention_qhead_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_pos_ptr,
    out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_t,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_t,
    stride_k_d,
    stride_v_b,
    stride_v_h,
    stride_v_t,
    stride_v_d,
    stride_o_b,
    stride_o_h,
    stride_o_t,
    stride_o_d,
    query_heads,
    kv_heads,
    scale,
    MAX_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // query_heads
    q_head = row - batch * query_heads
    kv_head = q_head // 2

    valid_len = tl.minimum(tl.load(cache_pos_ptr) + 1, MAX_K)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM

    q = tl.load(
        q_ptr + batch * stride_q_b + q_head * stride_q_h + offs_d * stride_q_d,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)

    m = tl.full((), -float("inf"), tl.float32)
    l = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_D,), tl.float32)

    start = 0
    while start < valid_len:
        offs_m = start + tl.arange(0, BLOCK_M)
        m_mask = offs_m < valid_len
        kv_mask = m_mask[:, None] & d_mask[None, :]

        k = tl.load(
            k_ptr
            + batch * stride_k_b
            + kv_head * stride_k_h
            + offs_m[:, None] * stride_k_t
            + offs_d[None, :] * stride_k_d,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr
            + batch * stride_v_b
            + kv_head * stride_v_h
            + offs_m[:, None] * stride_v_t
            + offs_d[None, :] * stride_v_d,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(m_mask, scores, -float("inf"))

        block_m = tl.max(scores, axis=0)
        new_m = tl.maximum(m, block_m)

        alpha = tl.exp(m - new_m)
        p = tl.exp(scores - new_m)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l = l * alpha + tl.sum(p, axis=0)
        m = new_m
        start += BLOCK_M

    out = acc / l
    tl.store(
        out_ptr + batch * stride_o_b + q_head * stride_o_h + offs_d * stride_o_d,
        out,
        mask=d_mask,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def fused_decode_prefix_attention_gqa2(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    scale: float,
    *,
    out: torch.Tensor | None = None,
    output_layout: str = "qhd",
    block_m: int = 64,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Decode-only split-query-head exact-prefix attention for Qwen3-VL GQA2.

    The kernel computes q_len=1 attention over keys/values in
    ``[0, cache_position]`` only. That matches the static-cache causal mask
    semantics without exposing padded cache slots to softmax.
    """
    if not (query.is_cuda and key_cache.is_cuda and value_cache.is_cuda and cache_position.is_cuda):
        raise ValueError("fused_decode_prefix_attention_gqa2 requires CUDA tensors")
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("fused_decode_prefix_attention_gqa2 currently supports fp16 tensors")
    if query.ndim != 4 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected query/key/value tensors with shape [B, H, T, D]")
    if int(query.shape[2]) != 1:
        raise ValueError("fused_decode_prefix_attention_gqa2 only supports q_len=1 decode")
    if int(query.shape[0]) != int(key_cache.shape[0]) or int(query.shape[0]) != int(value_cache.shape[0]):
        raise ValueError("query/key/value batch sizes must match")
    if int(key_cache.shape[1]) != int(value_cache.shape[1]):
        raise ValueError("key/value head counts must match")
    if int(query.shape[1]) != 2 * int(key_cache.shape[1]):
        raise ValueError("fused_decode_prefix_attention_gqa2 requires exactly 2 query heads per KV head")
    if int(query.shape[3]) != int(key_cache.shape[3]) or int(query.shape[3]) != int(value_cache.shape[3]):
        raise ValueError("query/key/value head dimensions must match")
    if cache_position.numel() != 1:
        raise ValueError("cache_position must contain a single decode index")

    batch = int(query.shape[0])
    query_heads = int(query.shape[1])
    kv_heads = int(key_cache.shape[1])
    head_dim = int(query.shape[3])
    max_k = int(key_cache.shape[2])
    block_d = _next_power_of_2(head_dim)

    if output_layout == "qhd":
        if out is None:
            out = torch.empty_like(query)
        if tuple(out.shape) != tuple(query.shape) or out.dtype != query.dtype or out.device != query.device:
            raise ValueError("qhd out must match query shape, dtype and device")
        stride_o_b = out.stride(0)
        stride_o_h = out.stride(1)
        stride_o_t = out.stride(2)
        stride_o_d = out.stride(3)
    elif output_layout == "flat":
        flat_shape = (batch, 1, query_heads * head_dim)
        if out is None:
            out = torch.empty(flat_shape, device=query.device, dtype=query.dtype)
        if tuple(out.shape) != flat_shape or out.dtype != query.dtype or out.device != query.device:
            raise ValueError("flat out must have shape [B, 1, query_heads * head_dim], dtype and device matching query")
        if not out.is_contiguous():
            raise ValueError("flat out must be contiguous")
        stride_o_b = out.stride(0)
        stride_o_h = head_dim * out.stride(2)
        stride_o_t = 0
        stride_o_d = out.stride(2)
    else:
        raise ValueError("output_layout must be 'qhd' or 'flat'")

    _decode_prefix_attention_qhead_kernel[(batch * query_heads,)](
        query,
        key_cache,
        value_cache,
        cache_position,
        out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        stride_o_b,
        stride_o_h,
        stride_o_t,
        stride_o_d,
        query_heads,
        kv_heads,
        float(scale),
        MAX_K=max_k,
        HEAD_DIM=head_dim,
        BLOCK_M=int(block_m),
        BLOCK_D=block_d,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out
