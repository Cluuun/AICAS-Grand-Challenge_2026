"""Post-decode Triton kernel for self-incrementing CUDA graph decode loop.

Runs AFTER each decode step inside a captured CUDA graph. Handles:
1. Write current decode output token to generated buffer
2. Copy output token to input_ids for next decode step
3. Update cos/sin position embeddings for next step (from pre-computed table)
4. Update cache_seqlens / cache_seqlens_next for next step
5. Increment step counter

With this kernel in the graph, the Python decode loop becomes:
  for _ in range(num_decode_steps): graph.replay()
  torch.cuda.synchronize()
No per-step Python bookkeeping or .item() sync needed.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode_post_step_kernel(
    cos_table_ptr, sin_table_ptr, cos_out_ptr, sin_out_ptr,
    cache_seqlens_ptr, cache_seqlens_next_ptr, base_seqlen_ptr,
    input_ids_ptr, next_token_ptr, generated_ptr,
    step_counter_ptr, HEAD_DIM: tl.constexpr,
):
    """Post-decode bookkeeping. step_counter = number of completed steps so far."""
    step = tl.load(step_counter_ptr).to(tl.int32)
    base = tl.load(base_seqlen_ptr)

    # Read decode step's output token
    token = tl.load(next_token_ptr)

    # Write to generated buffer: generated[step+1] (index 0 = prefill token)
    tl.store(generated_ptr + step + 1, token)

    # Prepare NEXT decode step
    next_step = step + 1

    # Copy token to input_ids for next decode step
    tl.store(input_ids_ptr, token)

    # Update cache sequence lengths for next step
    tl.store(cache_seqlens_ptr, base + next_step)
    tl.store(cache_seqlens_next_ptr, base + next_step + 1)

    # Index cos/sin from pre-computed table for next step
    offsets = tl.arange(0, HEAD_DIM)
    table_offset = next_step * HEAD_DIM + offsets
    cos_row = tl.load(cos_table_ptr + table_offset)
    tl.store(cos_out_ptr + offsets, cos_row)
    sin_row = tl.load(sin_table_ptr + table_offset)
    tl.store(sin_out_ptr + offsets, sin_row)

    # Increment step counter
    tl.store(step_counter_ptr, tl.cast(next_step, tl.int64))


def decode_post_step(
    cos_table_flat: torch.Tensor,
    sin_table_flat: torch.Tensor,
    cos_out_flat: torch.Tensor,
    sin_out_flat: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_seqlens_next: torch.Tensor,
    base_seqlen: torch.Tensor,
    input_ids_flat: torch.Tensor,
    next_token_flat: torch.Tensor,
    generated_flat: torch.Tensor,
    step_counter: torch.Tensor,
    head_dim: int = 128,
):
    _decode_post_step_kernel[(1,)](
        cos_table_flat, sin_table_flat, cos_out_flat, sin_out_flat,
        cache_seqlens, cache_seqlens_next, base_seqlen,
        input_ids_flat, next_token_flat, generated_flat,
        step_counter, HEAD_DIM=head_dim,
    )


def prewarm_decode_post_step(device, dtype=torch.bfloat16, head_dim=128):
    """Prewarm the Triton kernel to avoid JIT compile during graph capture."""
    cos_t = torch.zeros(4 * head_dim, device=device, dtype=dtype)
    sin_t = torch.zeros_like(cos_t)
    cos_o = torch.zeros(head_dim, device=device, dtype=dtype)
    sin_o = torch.zeros_like(cos_o)
    sl = torch.zeros(1, device=device, dtype=torch.int32)
    sl_n = torch.zeros(1, device=device, dtype=torch.int32)
    base = torch.zeros(1, device=device, dtype=torch.int32)
    iids = torch.zeros(1, device=device, dtype=torch.long)
    ntok = torch.zeros(1, device=device, dtype=torch.long)
    gen = torch.zeros(8, device=device, dtype=torch.long)
    sc = torch.zeros(1, device=device, dtype=torch.int64)
    decode_post_step(
        cos_t, sin_t, cos_o, sin_o,
        sl, sl_n, base, iids, ntok, gen, sc,
        head_dim=head_dim,
    )
    torch.cuda.synchronize()
