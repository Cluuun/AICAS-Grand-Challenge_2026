from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from my_kernel.static_cache_update_triton import (
    can_use_static_cache_multi_token_update,
    update_static_cache_multi_token_triton,
)


def _run_case(q_len: int, dtype: torch.dtype) -> None:
    device = "cuda"
    batch = 1
    heads = 8
    cache_len = 64
    dim = 128
    start = 17

    key_cache = torch.zeros(batch, heads, cache_len, dim, device=device, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    key_ref = key_cache.clone()
    value_ref = value_cache.clone()

    key_states = torch.randn(batch, heads, q_len, dim, device=device, dtype=dtype)
    value_states = torch.randn_like(key_states)
    positions = torch.arange(start, start + q_len, device=device, dtype=torch.long)

    if not can_use_static_cache_multi_token_update(
        key_cache,
        value_cache,
        key_states,
        value_states,
        positions,
    ):
        raise RuntimeError(f"multi-token cache update rejected q_len={q_len} dtype={dtype}")

    update_static_cache_multi_token_triton(
        key_cache,
        value_cache,
        key_states,
        value_states,
        positions,
    )
    key_ref[:, :, start:start + q_len, :].copy_(key_states)
    value_ref[:, :, start:start + q_len, :].copy_(value_states)
    torch.cuda.synchronize()

    k_diff = (key_cache - key_ref).abs().max().item()
    v_diff = (value_cache - value_ref).abs().max().item()
    print(f"q={q_len} dtype={dtype} k_max_abs={k_diff:.6f} v_max_abs={v_diff:.6f}")
    if k_diff != 0 or v_diff != 0:
        raise RuntimeError(f"cache update mismatch q_len={q_len} dtype={dtype}")


def _run_graph_case(q_len: int, dtype: torch.dtype) -> None:
    device = "cuda"
    batch = 1
    heads = 8
    cache_len = 64
    dim = 128
    start = 17

    key_cache = torch.zeros(batch, heads, cache_len, dim, device=device, dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    key_states = torch.empty(batch, heads, q_len, dim, device=device, dtype=dtype)
    value_states = torch.empty_like(key_states)
    positions = torch.arange(start, start + q_len, device=device, dtype=torch.long)

    if not can_use_static_cache_multi_token_update(
        key_cache,
        value_cache,
        key_states,
        value_states,
        positions,
    ):
        raise RuntimeError(f"graph multi-token cache update rejected q_len={q_len} dtype={dtype}")

    warm_key_cache = torch.zeros_like(key_cache)
    warm_value_cache = torch.zeros_like(value_cache)
    warm_key_states = torch.randn_like(key_states)
    warm_value_states = torch.randn_like(value_states)
    update_static_cache_multi_token_triton(
        warm_key_cache,
        warm_value_cache,
        warm_key_states,
        warm_value_states,
        positions,
    )
    torch.cuda.synchronize()

    first_key_states = torch.randn_like(key_states)
    first_value_states = torch.randn_like(value_states)
    key_states.copy_(first_key_states)
    value_states.copy_(first_value_states)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update_static_cache_multi_token_triton(
            key_cache,
            value_cache,
            key_states,
            value_states,
            positions,
        )

    replay_key_states = torch.randn_like(key_states)
    replay_value_states = torch.randn_like(value_states)
    key_cache.zero_()
    value_cache.zero_()
    key_states.copy_(replay_key_states)
    value_states.copy_(replay_value_states)
    graph.replay()
    torch.cuda.synchronize()

    key_ref = torch.zeros_like(key_cache)
    value_ref = torch.zeros_like(value_cache)
    key_ref[:, :, start:start + q_len, :].copy_(replay_key_states)
    value_ref[:, :, start:start + q_len, :].copy_(replay_value_states)
    k_diff = (key_cache - key_ref).abs().max().item()
    v_diff = (value_cache - value_ref).abs().max().item()
    k_abs = key_cache[:, :, start:start + q_len, :].abs().max().item()
    v_abs = value_cache[:, :, start:start + q_len, :].abs().max().item()
    print(
        f"graph q={q_len} dtype={dtype} "
        f"k_max_abs={k_diff:.6f} v_max_abs={v_diff:.6f} "
        f"k_abs={k_abs:.6f} v_abs={v_abs:.6f}"
    )
    if k_diff != 0 or v_diff != 0:
        raise RuntimeError(f"graph cache update mismatch q_len={q_len} dtype={dtype}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    for dtype in (torch.float16, torch.bfloat16):
        for q_len in (2, 3, 4, 8, 15, 16):
            _run_case(q_len, dtype)
            _run_graph_case(q_len, dtype)
    print("static cache multi-token update validation passed")


if __name__ == "__main__":
    main()
