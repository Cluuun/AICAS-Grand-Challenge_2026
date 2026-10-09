"""Compiled decode functions + CUDA-graph capture.

Returns two compiled callables:
  decode_step       — one token from current state
  decode_n_steps    — N tokens unrolled, fed back through inv_perm if vocab
                      whitelist is active

B1 fix: mode='default' (cold autotune is the dominant cost on the harness'
fresh container; default still vectorises the matmuls and reuses fused
kernels we declared via custom_op).
"""
from __future__ import annotations

import torch


def make_decode_fns(
    *, language_model, lm_head_apply, inv_perm, fa_seqlens, n_steps,
):
    """Build (decode_step, decode_n_steps) compiled functions.

    `inv_perm` may be None (no vocab whitelist); when set it is a (V_small,)
    long tensor on device used to remap argmax output back to the original
    vocab id space before the next embed lookup.
    """
    lm = language_model

    if inv_perm is not None:
        @torch.compile(mode="default", dynamic=False)
        def decode_step(input_ids, position_ids, fa_seqlens):
            out = lm(input_ids=input_ids, position_ids=position_ids,
                     past_key_values=None, use_cache=False, return_dict=True)
            h = out.last_hidden_state[:, -1, :]
            small = lm_head_apply(h).argmax(dim=-1, keepdim=True)
            token = inv_perm[small.flatten()].view_as(small)
            fa_seqlens.add_(1)
            return token

        @torch.compile(mode="default", dynamic=False)
        def decode_n_steps(start_token, pos_n, fa_seqlens):
            cur = start_token
            tokens = []
            for i in range(n_steps):
                out = lm(input_ids=cur, position_ids=pos_n[i],
                         past_key_values=None, use_cache=False, return_dict=True)
                h = out.last_hidden_state[:, -1, :]
                small = lm_head_apply(h).argmax(dim=-1, keepdim=True)
                cur = inv_perm[small.flatten()].view_as(small)
                tokens.append(cur)
                fa_seqlens.add_(1)
            return torch.cat(tokens, dim=1)
    else:
        @torch.compile(mode="default", dynamic=False)
        def decode_step(input_ids, position_ids, fa_seqlens):
            out = lm(input_ids=input_ids, position_ids=position_ids,
                     past_key_values=None, use_cache=False, return_dict=True)
            h = out.last_hidden_state[:, -1, :]
            token = lm_head_apply(h).argmax(dim=-1, keepdim=True)
            fa_seqlens.add_(1)
            return token

        @torch.compile(mode="default", dynamic=False)
        def decode_n_steps(start_token, pos_n, fa_seqlens):
            cur = start_token
            tokens = []
            for i in range(n_steps):
                out = lm(input_ids=cur, position_ids=pos_n[i],
                         past_key_values=None, use_cache=False, return_dict=True)
                h = out.last_hidden_state[:, -1, :]
                cur = lm_head_apply(h).argmax(dim=-1, keepdim=True)
                tokens.append(cur)
                fa_seqlens.add_(1)
            return torch.cat(tokens, dim=1)

    return decode_step, decode_n_steps


def capture_decode_graphs(
    *, decode_step, decode_n_steps,
    decode_input_ids, decode_position_ids,
    decode_start_token, pos_n, fa_seqlens,
    warmup_iters: int = 3,
):
    """Warm-compile decode_step / decode_n_steps then capture CUDA graphs.

    Returns (graph_1, static_token_1, graph_n, static_tokens_n).
    """
    with torch.no_grad():
        fa_seqlens.fill_(512)
        for _ in range(warmup_iters):
            _ = decode_n_steps(decode_start_token, pos_n, fa_seqlens)
            _ = decode_step(decode_input_ids, decode_position_ids, fa_seqlens)
    torch.cuda.synchronize()

    fa_seqlens.fill_(512)

    graph_n = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(graph_n):
            static_tokens_n = decode_n_steps(
                decode_start_token, pos_n, fa_seqlens)

    graph_1 = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(graph_1):
            static_token_1 = decode_step(
                decode_input_ids, decode_position_ids, fa_seqlens)

    torch.cuda.synchronize()
    return graph_1, static_token_1, graph_n, static_tokens_n
