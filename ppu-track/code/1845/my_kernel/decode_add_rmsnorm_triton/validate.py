from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from my_kernel.decode_add_rmsnorm_triton.runtime import _can_use, _fused_add_rmsnorm  # noqa: E402


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise ValueError(f"empty int list: {value!r}")
    return out


def _ptr_range(tensor: torch.Tensor) -> tuple[int, int]:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return (0, 0)
    start = int(tensor.untyped_storage().data_ptr())
    return (start, start + int(tensor.numel()) * int(tensor.element_size()))


def _tensor_desc(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu")
    if cpu.numel() == 0:
        return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} empty ptr={_ptr_range(tensor)}"
    if cpu.is_floating_point():
        lo = float(cpu.min().item())
        hi = float(cpu.max().item())
        return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} range=[{lo:.6g},{hi:.6g}] ptr={_ptr_range(tensor)}"
    lo = int(cpu.min().item())
    hi = int(cpu.max().item())
    return f"shape={tuple(tensor.shape)} dtype={tensor.dtype} range=[{lo},{hi}] ptr={_ptr_range(tensor)}"


def _snapshot_watch(watch: dict[str, torch.Tensor], names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {name: watch[name].detach().to(device="cpu").clone() for name in names if name in watch}


def _find_watch_change(
    watch: dict[str, torch.Tensor],
    before: dict[str, torch.Tensor],
) -> tuple[str, str] | None:
    for name, snap in before.items():
        after = watch[name].detach().to(device="cpu")
        if torch.equal(snap, after):
            continue
        diff = int((snap != after).sum().item()) if snap.shape == after.shape else -1
        msg = (
            f"watch.{name} changed diff={diff} ptr={_ptr_range(watch[name])} "
            f"before_shape={tuple(snap.shape)} after_shape={tuple(after.shape)} "
            f"before_range=[{snap.min().item()},{snap.max().item()}] "
            f"after_range=[{after.min().item()},{after.max().item()}]"
        )
        return name, msg
    return None


def _make_draft_like_watch(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    """Allocate long-lived tensors matching the buffers corrupted in draft_graph_replay.

    The tensors are not consumed by the kernel. They are deliberately kept live
    around graph capture/replay so an out-of-bounds graph kernel write has a
    realistic adjacent target to hit.
    """
    watch_len = int(args.watch_len)
    grid_h = int(args.grid_h)
    grid_w = int(args.grid_w)
    vocab = int(args.vocab_size)

    input_ids = torch.arange(watch_len, device=device, dtype=torch.long).view(1, -1)
    input_ids.remainder_(max(vocab, 1))
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.long)
    if bool(args.no_pixel_watch):
        pixel_values = torch.empty(0, device=device, dtype=torch.float32)
    else:
        pixel_values = torch.full(
            (int(args.pixel_tokens), int(args.pixel_dim)),
            -1.0,
            device=device,
            dtype=torch.float32,
        )
    attention_mask = torch.ones((1, watch_len), device=device, dtype=torch.long)
    mm_token_type_ids = torch.zeros((1, watch_len), device=device, dtype=torch.long)
    if watch_len > 0:
        mm_token_type_ids[:, max(0, watch_len - grid_h) :] = 1

    return {
        "grid_input_ids": input_ids,
        "grid_image_grid_thw": image_grid_thw,
        "grid_pixel_values": pixel_values,
        "grid_attention_mask": attention_mask,
        "grid_mm_token_type_ids": mm_token_type_ids,
    }


def _guarded_view(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    guard_elems: int,
    fill_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    arena = torch.empty(int(guard_elems) + numel + int(guard_elems), device=device, dtype=dtype)
    arena[:guard_elems].fill_(fill_value)
    arena[guard_elems + numel :].fill_(fill_value)
    view = arena[guard_elems : guard_elems + numel].view(shape)
    return arena, view, arena.detach().clone()


def _check_guard(label: str, arena: torch.Tensor, snap: torch.Tensor, guard_elems: int) -> None:
    if int(guard_elems) <= 0:
        return
    before = snap[:guard_elems]
    after = arena[:guard_elems]
    if not torch.equal(before, after):
        raise RuntimeError(f"{label} pre-guard corrupted")
    before = snap[-guard_elems:]
    after = arena[-guard_elems:]
    if not torch.equal(before, after):
        raise RuntimeError(f"{label} post-guard corrupted")


def _reference(
    residual: torch.Tensor,
    attn_out: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    added = (residual.float() + attn_out.float()).to(torch.bfloat16)
    s = added.float()
    inv = torch.rsqrt((s * s).mean(dim=-1, keepdim=True) + float(eps))
    normed = (s * inv).to(torch.bfloat16).float()
    out = (normed * weight.float()).to(torch.bfloat16)
    return added, out


def _assert_close(label: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    diff = (got.float() - ref.float()).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    if max_abs > float(tol):
        idx = int(diff.reshape(-1).argmax().item()) if diff.numel() else -1
        raise RuntimeError(
            f"{label} mismatch max_abs={max_abs:.6g} tol={float(tol):.6g} "
            f"idx={idx} got={float(got.reshape(-1)[idx].float().item()) if idx >= 0 else 'n/a'} "
            f"ref={float(ref.reshape(-1)[idx].float().item()) if idx >= 0 else 'n/a'}"
        )


def _run_eager_case(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(int(args.seed))
    rows = int(args.rows)
    hidden = int(args.hidden_size)
    eps = float(args.eps)
    dtype = torch.bfloat16

    residual = torch.randn(rows, hidden, device=device, dtype=dtype).view(1, rows, hidden)
    attn_out = torch.randn_like(residual)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    if not _can_use(residual, attn_out):
        raise RuntimeError(f"_can_use rejected residual={tuple(residual.shape)}")

    added, normed, _ws_add, _ws_norm, _ws_partial, _ws_sum = _fused_add_rmsnorm(
        residual,
        attn_out,
        weight,
        eps,
        None,
        None,
        None,
        None,
    )
    torch.cuda.synchronize(device)
    ref_added, ref_normed = _reference(residual, attn_out, weight, eps)
    _assert_close("eager.added", added, ref_added, float(args.add_tol))
    _assert_close("eager.normed", normed, ref_normed, float(args.norm_tol))
    print(f"decode_add_rmsnorm eager rows={rows} hidden={hidden} passed")


def _run_graph_case(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(int(args.seed) + 1)
    rows = int(args.rows)
    hidden = int(args.hidden_size)
    eps = float(args.eps)
    dtype = torch.bfloat16
    guard_elems = int(args.guard_elems)

    watch = _make_draft_like_watch(args, device)
    watch_names = ("grid_input_ids", "grid_attention_mask", "grid_image_grid_thw", "grid_mm_token_type_ids")

    residual = torch.empty(1, rows, hidden, device=device, dtype=dtype)
    attn_out = torch.empty_like(residual)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    if not _can_use(residual, attn_out):
        raise RuntimeError(f"_can_use rejected graph residual={tuple(residual.shape)}")

    added_arena, added_out, added_guard = _guarded_view(
        (rows, hidden),
        device=device,
        dtype=dtype,
        guard_elems=guard_elems,
        fill_value=-123.0,
    )
    norm_arena, norm_out, norm_guard = _guarded_view(
        (rows, hidden),
        device=device,
        dtype=dtype,
        guard_elems=guard_elems,
        fill_value=123.0,
    )
    partial_tiles = (hidden + 4095) // 4096
    partial_arena, partial_out, partial_guard = _guarded_view(
        (rows, partial_tiles),
        device=device,
        dtype=torch.float32,
        guard_elems=max(1, min(guard_elems, 1024)),
        fill_value=-777.0,
    )
    sum_arena, sum_out, sum_guard = _guarded_view(
        (rows,),
        device=device,
        dtype=torch.float32,
        guard_elems=max(1, min(guard_elems, 1024)),
        fill_value=777.0,
    )

    residual.copy_(torch.randn_like(residual))
    attn_out.copy_(torch.randn_like(attn_out))
    _fused_add_rmsnorm(
        residual,
        attn_out,
        weight,
        eps,
        added_out,
        norm_out,
        partial_out,
        sum_out,
    )
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(graph):
            graph_added, graph_normed, *_ = _fused_add_rmsnorm(
                residual,
                attn_out,
                weight,
                eps,
                added_out,
                norm_out,
                partial_out,
                sum_out,
            )

    for idx in range(int(args.iters)):
        residual.copy_(torch.randn_like(residual))
        attn_out.copy_(torch.randn_like(attn_out))
        watch_before = _snapshot_watch(watch, watch_names)
        graph.replay()
        torch.cuda.synchronize(device)

        ref_added, ref_normed = _reference(residual, attn_out, weight, eps)
        _assert_close(f"graph[{idx}].added", graph_added, ref_added, float(args.add_tol))
        _assert_close(f"graph[{idx}].normed", graph_normed, ref_normed, float(args.norm_tol))
        _check_guard("added_out", added_arena, added_guard, guard_elems)
        _check_guard("norm_out", norm_arena, norm_guard, guard_elems)
        _check_guard("partials", partial_arena, partial_guard, max(1, min(guard_elems, 1024)))
        _check_guard("sumsq", sum_arena, sum_guard, max(1, min(guard_elems, 1024)))

        changed = _find_watch_change(watch, watch_before)
        if changed is not None:
            name, msg = changed
            print(f"[decode_add_rmsnorm] replay{idx + 1} WATCH_MUTATED: {msg}", flush=True)
            for wn, wt in watch.items():
                print(f"[decode_add_rmsnorm]   watch.{wn}: {_tensor_desc(wt)}", flush=True)
            print(f"[decode_add_rmsnorm]   entry.residual: {_tensor_desc(residual)}", flush=True)
            print(f"[decode_add_rmsnorm]   entry.attn_out: {_tensor_desc(attn_out)}", flush=True)
            print(f"[decode_add_rmsnorm]   entry.added_out: {_tensor_desc(added_out)}", flush=True)
            print(f"[decode_add_rmsnorm]   entry.norm_out: {_tensor_desc(norm_out)}", flush=True)
            raise RuntimeError(f"decode_add_rmsnorm graph replay corrupted watch tensor: {name}")
        if int(args.log_every) > 0 and (idx + 1) % int(args.log_every) == 0:
            print(f"decode_add_rmsnorm graph replay iter={idx + 1}/{int(args.iters)} ok", flush=True)

    print(
        f"decode_add_rmsnorm graph rows={rows} hidden={hidden} "
        f"iters={int(args.iters)} passed",
        flush=True,
    )


def _make_graph_entry(
    args: argparse.Namespace,
    device: torch.device,
    *,
    rows: int,
    hidden: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    eps = float(args.eps)
    dtype = torch.bfloat16
    guard_elems = int(args.guard_elems)

    residual = torch.empty(1, rows, hidden, device=device, dtype=dtype)
    attn_out = torch.empty_like(residual)
    weight = torch.randn(hidden, device=device, dtype=dtype)
    if not _can_use(residual, attn_out):
        raise RuntimeError(f"_can_use rejected residual={tuple(residual.shape)}")

    added_arena, added_out, added_guard = _guarded_view(
        (rows, hidden),
        device=device,
        dtype=dtype,
        guard_elems=guard_elems,
        fill_value=-123.0,
    )
    norm_arena, norm_out, norm_guard = _guarded_view(
        (rows, hidden),
        device=device,
        dtype=dtype,
        guard_elems=guard_elems,
        fill_value=123.0,
    )
    partial_tiles = (hidden + 4095) // 4096
    partial_guard_elems = max(1, min(guard_elems, 1024))
    partial_arena, partial_out, partial_guard = _guarded_view(
        (rows, partial_tiles),
        device=device,
        dtype=torch.float32,
        guard_elems=partial_guard_elems,
        fill_value=-777.0,
    )
    sum_arena, sum_out, sum_guard = _guarded_view(
        (rows,),
        device=device,
        dtype=torch.float32,
        guard_elems=partial_guard_elems,
        fill_value=777.0,
    )

    residual.copy_(torch.randn_like(residual))
    attn_out.copy_(torch.randn_like(attn_out))
    _fused_add_rmsnorm(residual, attn_out, weight, eps, added_out, norm_out, partial_out, sum_out)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(graph):
            graph_added, graph_normed, *_ = _fused_add_rmsnorm(
                residual,
                attn_out,
                weight,
                eps,
                added_out,
                norm_out,
                partial_out,
                sum_out,
            )

    return {
        "rows": rows,
        "hidden": hidden,
        "eps": eps,
        "graph": graph,
        "residual": residual,
        "attn_out": attn_out,
        "weight": weight,
        "graph_added": graph_added,
        "graph_normed": graph_normed,
        "added_arena": added_arena,
        "added_guard": added_guard,
        "norm_arena": norm_arena,
        "norm_guard": norm_guard,
        "partial_arena": partial_arena,
        "partial_guard": partial_guard,
        "sum_arena": sum_arena,
        "sum_guard": sum_guard,
        "partial_guard_elems": partial_guard_elems,
    }


def _run_draft_like_replay_case(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(int(args.seed) + 2)
    watch = _make_draft_like_watch(args, device)
    watch_names = ("grid_input_ids", "grid_attention_mask", "grid_image_grid_thw", "grid_mm_token_type_ids")
    rows_pattern = _parse_int_list(args.rows_pattern)
    hidden_sizes = _parse_int_list(args.hidden_sizes)

    entries: list[dict[str, object]] = []
    for hidden in hidden_sizes:
        for rows in rows_pattern:
            entries.append(
                _make_graph_entry(
                    args,
                    device,
                    rows=int(rows),
                    hidden=int(hidden),
                    seed=int(args.seed) + 1000 + len(entries),
                )
            )
            print(f"decode_add_rmsnorm draft-like graph captured: hidden=(1,{int(rows)},{int(hidden)})", flush=True)

    for idx in range(int(args.iters)):
        entry = entries[idx % len(entries)]
        residual = entry["residual"]
        attn_out = entry["attn_out"]
        assert isinstance(residual, torch.Tensor)
        assert isinstance(attn_out, torch.Tensor)
        residual.copy_(torch.randn_like(residual))
        attn_out.copy_(torch.randn_like(attn_out))
        watch_before = _snapshot_watch(watch, watch_names)
        graph = entry["graph"]
        assert isinstance(graph, torch.cuda.CUDAGraph)
        graph.replay()
        torch.cuda.synchronize(device)

        changed = _find_watch_change(watch, watch_before)
        if changed is not None:
            name, msg = changed
            print(
                f"[decode_add_rmsnorm] replay{idx + 1} WATCH_MUTATED: "
                f"shape=(1,{int(entry['rows'])},{int(entry['hidden'])}) {msg}",
                flush=True,
            )
            for wn, wt in watch.items():
                print(f"[decode_add_rmsnorm]   watch.{wn}: {_tensor_desc(wt)}", flush=True)
            for bn in ("residual", "attn_out", "graph_added", "graph_normed"):
                bt = entry[bn]
                assert isinstance(bt, torch.Tensor)
                print(f"[decode_add_rmsnorm]   entry.{bn}: {_tensor_desc(bt)}", flush=True)
            raise RuntimeError(f"decode_add_rmsnorm draft-like replay corrupted watch tensor: {name}")

        should_check_math = idx < int(args.math_check_first) or (
            int(args.math_check_every) > 0 and (idx + 1) % int(args.math_check_every) == 0
        )
        if should_check_math:
            ref_added, ref_normed = _reference(
                residual,
                attn_out,
                entry["weight"],
                float(entry["eps"]),
            )
            _assert_close(f"draft_like[{idx}].added", entry["graph_added"], ref_added, float(args.add_tol))
            _assert_close(f"draft_like[{idx}].normed", entry["graph_normed"], ref_normed, float(args.norm_tol))

        _check_guard("added_out", entry["added_arena"], entry["added_guard"], int(args.guard_elems))
        _check_guard("norm_out", entry["norm_arena"], entry["norm_guard"], int(args.guard_elems))
        _check_guard("partials", entry["partial_arena"], entry["partial_guard"], int(entry["partial_guard_elems"]))
        _check_guard("sumsq", entry["sum_arena"], entry["sum_guard"], int(entry["partial_guard_elems"]))

        if int(args.log_every) > 0 and (idx + 1) % int(args.log_every) == 0:
            print(f"decode_add_rmsnorm draft-like replay iter={idx + 1}/{int(args.iters)} ok", flush=True)

    print(
        "decode_add_rmsnorm draft-like graph replay "
        f"graphs={len(entries)} iters={int(args.iters)} passed",
        flush=True,
    )


def _make_eagle3_config(args: argparse.Namespace):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextConfig

    cfg = Qwen3VLTextConfig(
        hidden_size=int(args.draft_hidden_size),
        intermediate_size=int(args.draft_intermediate_size),
        num_attention_heads=int(args.draft_num_heads),
        num_key_value_heads=int(args.draft_num_kv_heads),
        head_dim=int(args.draft_head_dim),
        num_hidden_layers=1,
        vocab_size=int(args.vocab_size),
        attention_bias=False,
        hidden_act="silu",
        rms_norm_eps=float(args.eps),
    )
    cfg._attn_implementation = "eager"
    return cfg


def _print_draft_step_entry(entry: dict[str, object] | None) -> None:
    if not isinstance(entry, dict):
        print("[decode_add_rmsnorm]   draft_step.entry: unavailable", flush=True)
        return
    for name in (
        "hidden",
        "input_ids",
        "position_ids",
        "past_k",
        "past_v",
        "tree_mask",
        "out_hidden",
        "out_k",
        "out_v",
        "topk_ids",
        "topk_vals",
    ):
        tensor = entry.get(name)
        if isinstance(tensor, torch.Tensor):
            print(f"[decode_add_rmsnorm]   draft_step.{name}: {_tensor_desc(tensor)}", flush=True)


def _run_draft_step_graph_case(args: argparse.Namespace, device: torch.device) -> None:
    import spec_decode.eagle3 as eagle3_mod

    os.environ["AICAS_EAGLE3_DRAFT_STEP_CUDAGRAPH"] = "1"
    os.environ["AICAS_EAGLE3_DRAFT_ADD_RMSNORM"] = "1"
    os.environ["AICAS_EAGLE3_DRAFT_QK_ROTARY"] = "1" if bool(args.draft_qk_rotary) else "0"
    os.environ.setdefault("AICAS_EAGLE3_DRAFT_RMSNORM", "1")
    os.environ.setdefault("AICAS_SPEC_KERNEL_MAX_Q", str(max(16, int(args.rows))))
    os.environ.setdefault("AICAS_EAGLE3_DRAFT_GRAPH_GLOBAL_CACHE", "0")

    torch.manual_seed(int(args.seed) + 3)
    cfg = _make_eagle3_config(args)
    model = eagle3_mod.Eagle3DraftModel(cfg, num_heads=int(args.draft_num_heads)).to(
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    q_len = int(args.rows)
    hidden_size = int(args.draft_hidden_size)
    past_len = int(args.draft_past_len)
    hidden_states = torch.empty(1, q_len, hidden_size * 3, device=device, dtype=torch.bfloat16)
    input_ids = torch.empty(1, q_len, device=device, dtype=torch.long)
    past_k = torch.randn(
        1,
        int(args.draft_num_kv_heads),
        past_len,
        int(args.draft_head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    past_v = torch.randn_like(past_k)
    tree_mask = torch.eye(q_len, device=device, dtype=torch.bool).view(1, 1, q_len, q_len)
    model.tree_mask = tree_mask

    watch = _make_draft_like_watch(args, device)
    watch_names = ("grid_input_ids", "grid_attention_mask", "grid_image_grid_thw", "grid_mm_token_type_ids")
    add_counts: dict[tuple[int, ...] | None, int] = {}
    original_add = eagle3_mod._fused_decode_add_rmsnorm

    hidden_samples = [
        torch.randn_like(hidden_states)
        for _ in range(int(args.iters))
    ]
    input_samples = [
        torch.randint(
            0,
            int(args.vocab_size),
            tuple(input_ids.shape),
            device=device,
            dtype=torch.long,
        )
        for _ in range(int(args.iters))
    ]

    def _counting_add_rmsnorm(residual, attn_out, norm_weight, eps, added_out, norm_out, partials_out=None, sumsq_out=None):
        shape = tuple(residual.shape) if isinstance(residual, torch.Tensor) else None
        add_counts[shape] = int(add_counts.get(shape, 0)) + 1
        return original_add(residual, attn_out, norm_weight, eps, added_out, norm_out, partials_out, sumsq_out)

    eagle3_mod._fused_decode_add_rmsnorm = _counting_add_rmsnorm
    try:
        for idx in range(int(args.iters)):
            hidden_states.copy_(hidden_samples[idx])
            input_ids.copy_(input_samples[idx])
            watch_before = _snapshot_watch(watch, watch_names)
            out_hidden, out_past, topk_ids, topk_vals = model._draft_step_graph(
                hidden_states,
                input_ids,
                ((past_k, past_v),),
                position_ids=None,
                logits_processor=None,
                topk_last_only=False,
            )
            torch.cuda.synchronize(device)
            if idx == 0:
                print(
                    "decode_add_rmsnorm draft-step graph captured: "
                    f"feature={tuple(hidden_states.shape)} q={q_len} "
                    f"past_len={past_len} add_rmsnorm_shapes={dict(add_counts)}",
                    flush=True,
                )
            changed = _find_watch_change(watch, watch_before)
            if changed is not None:
                name, msg = changed
                print(
                    f"[decode_add_rmsnorm] draft-step call{idx + 1} WATCH_MUTATED: {msg}",
                    flush=True,
                )
                for wn, wt in watch.items():
                    print(f"[decode_add_rmsnorm]   watch.{wn}: {_tensor_desc(wt)}", flush=True)
                key = model._draft_step_graph_key(
                    hidden_states,
                    input_ids,
                    None,
                    ((past_k, past_v),),
                    topk_last_only=False,
                )
                entry = model._draft_step_graph_state()[0].get(key) if key is not None else None
                _print_draft_step_entry(entry)
                print(f"[decode_add_rmsnorm]   add_rmsnorm_capture_counts: {dict(add_counts)}", flush=True)
                raise RuntimeError(f"decode_add_rmsnorm draft-step graph corrupted watch tensor: {name}")

            if bool(args.check_topk):
                topk_cpu = topk_ids.detach().to(device="cpu", dtype=torch.long)
                if int(topk_cpu.min().item()) < 0 or int(topk_cpu.max().item()) >= int(args.vocab_size):
                    raise RuntimeError(
                        f"draft-step topk ids out of range at iter={idx}: "
                        f"{_tensor_desc(topk_ids)} vocab={int(args.vocab_size)}"
                    )
                _ = out_hidden, out_past, topk_vals

            if int(args.log_every) > 0 and (idx + 1) % int(args.log_every) == 0:
                print(f"decode_add_rmsnorm draft-step iter={idx + 1}/{int(args.iters)} ok", flush=True)
    finally:
        eagle3_mod._fused_decode_add_rmsnorm = original_add

    if not add_counts:
        raise RuntimeError("draft-step graph did not call _fused_decode_add_rmsnorm")
    print(
        "decode_add_rmsnorm draft-step graph replay "
        f"iters={int(args.iters)} add_rmsnorm_shapes={dict(add_counts)} passed",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate decode_add_rmsnorm_triton for large hidden sizes and CUDA graph replay.")
    parser.add_argument("--mode", choices=("single", "draft-like", "draft-step", "all"), default="draft-step")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--rows-pattern", default="4,1,2,3,5")
    parser.add_argument("--hidden-sizes", default="2048")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--watch-len", type=int, default=1371)
    parser.add_argument("--grid-h", type=int, default=64)
    parser.add_argument("--grid-w", type=int, default=56)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--pixel-tokens", type=int, default=3584)
    parser.add_argument("--pixel-dim", type=int, default=1536)
    parser.add_argument("--no-pixel-watch", action="store_true")
    parser.add_argument("--guard-elems", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--add-tol", type=float, default=0.0)
    parser.add_argument("--norm-tol", type=float, default=0.03125)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--math-check-first", type=int, default=3)
    parser.add_argument("--math-check-every", type=int, default=500)
    parser.add_argument("--draft-hidden-size", type=int, default=2048)
    parser.add_argument("--draft-intermediate-size", type=int, default=6144)
    parser.add_argument("--draft-num-heads", type=int, default=16)
    parser.add_argument("--draft-num-kv-heads", type=int, default=8)
    parser.add_argument("--draft-head-dim", type=int, default=128)
    parser.add_argument("--draft-past-len", type=int, default=1371)
    parser.add_argument("--draft-qk-rotary", action="store_true")
    parser.add_argument("--check-topk", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/PPU device is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=device)
    if args.mode in ("single", "all"):
        _run_eager_case(args, device)
        _run_graph_case(args, device)
    if args.mode in ("draft-like", "all"):
        _run_draft_like_replay_case(args, device)
    if args.mode in ("draft-step", "all"):
        _run_draft_step_graph_case(args, device)
    print("decode_add_rmsnorm_triton validation passed", flush=True)


if __name__ == "__main__":
    main()
