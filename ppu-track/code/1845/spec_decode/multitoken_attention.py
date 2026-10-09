from __future__ import annotations

import math
import os
from contextlib import contextmanager

import torch
import triton
import triton.language as tl

try:
    from my_kernel.flashDecode.flashdecode_runtime import _load_flashdecode_ext
except Exception:
    _load_flashdecode_ext = None

_FLASHDECODE_RUNNER = None
_FLASHDECODE_OUT = {}
_BF16_EXT_UNAVAILABLE = False
_BF16_EXT_STATS = {"hit": 0, "fallback": 0, "logged_hit": False}
_TRACE_ONCE: set[str] = set()
_CHAIN_CHECK_STATE = {"count": 0}
_TREE_ROOT_STAGE_STATE = {"count": 0}


def _eager_attention_fallback(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.ndim == 4
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
        and key.ndim == 4
        and value.ndim == 4
    ):
        logical_kv = int(attention_mask.shape[-1])
        if logical_kv > 0 and int(key.shape[-2]) > logical_kv:
            key = key[:, :, :logical_kv, :]
            value = value[:, :, :logical_kv, :]

    fallback = qwen.eager_attention_forward
    return fallback(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


def _trace_once(key: str, message: str) -> None:
    if os.getenv("AICAS_SPEC_KERNEL_TRACE", "0") != "1" or key in _TRACE_ONCE:
        return
    _TRACE_ONCE.add(key)
    print(message)


def _tensor_abs_stats(tensor: torch.Tensor) -> tuple[float, float, float]:
    flat = tensor.detach().float().abs().reshape(-1)
    if int(flat.numel()) == 0:
        return 0.0, 0.0, 0.0
    return float(flat.max().item()), float(flat.mean().item()), float(flat.float().norm().item())


def _chain_stage_debug(
    *,
    module,
    backend_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    ref: torch.Tensor,
    step: int,
) -> None:
    if os.getenv("AICAS_SPEC_CHAIN_ATTN_STAGE_DEBUG", "0") != "1":
        return
    q_len = int(query.shape[-2])
    kv_len = int(key.shape[-2])
    try:
        q_idx = int(os.getenv("AICAS_SPEC_CHAIN_ATTN_STAGE_Q", "0"))
        h_idx = int(os.getenv("AICAS_SPEC_CHAIN_ATTN_STAGE_H", "0"))
        q_idx = max(0, min(q_idx, q_len - 1))
        h_idx = max(0, min(h_idx, int(query.shape[1]) - 1))
        hkv = h_idx // max(1, int(getattr(module, "num_key_value_groups", 2)))
        scale = float(getattr(module, "scaling", 1.0 / math.sqrt(int(query.shape[-1]))))
        prefix_len = kv_len - q_len
        end_pos = prefix_len + q_idx + 1
        q_vec = query[0, h_idx, q_idx, :].float()
        k_mat = key[0, hkv, :end_pos, :].float()
        v_mat = value[0, hkv, :end_pos, :].float()
        scores_fp32 = torch.matmul(k_mat, q_vec) * scale
        probs_fp32 = torch.softmax(scores_fp32, dim=-1, dtype=torch.float32)
        ref_stage = torch.matmul(probs_fp32.to(v_mat.dtype), v_mat).float()
        torch_bf16_stage = torch.matmul(
            probs_fp32.to(torch.bfloat16).view(1, -1),
            value[0, hkv, :end_pos, :],
        ).reshape(-1).float()
        scores_bf16 = (torch.matmul(k_mat.to(torch.bfloat16), q_vec.to(torch.bfloat16)).float() * scale)
        probs_bf16 = torch.softmax(scores_bf16, dim=-1, dtype=torch.float32).to(torch.bfloat16)
        bf16_stage = torch.matmul(probs_bf16.float(), v_mat).float()
        out_vec = out[0, q_idx, h_idx, :].float()
        ref_vec = ref[0, q_idx, h_idx, :].float()
        stage_ref_diff = (ref_stage - ref_vec).abs()
        torch_bf16_diff = (torch_bf16_stage - ref_vec).abs()
        stage_bf16_diff = (bf16_stage - ref_vec).abs()
        out_diff = (out_vec - ref_vec).abs()
        top_scores = torch.topk(scores_fp32, k=min(4, int(scores_fp32.numel())), dim=-1)
        top_probs = torch.topk(probs_fp32, k=min(4, int(probs_fp32.numel())), dim=-1)
        score_margin = (
            float((top_scores.values[0] - top_scores.values[1]).item())
            if int(top_scores.values.numel()) >= 2
            else float("inf")
        )
        prob_mass4 = float(top_probs.values.sum().item()) if int(top_probs.values.numel()) else 0.0
        q_abs = _tensor_abs_stats(q_vec)
        k_abs = _tensor_abs_stats(k_mat)
        v_abs = _tensor_abs_stats(v_mat)
        layer_idx = getattr(module, "layer_idx", None)
        print(
            "[spec_attn][stage] "
            f"step={step} layer={layer_idx} backend={backend_name} q={q_len} kv={kv_len} "
            f"probe=(q={q_idx},h={h_idx},hkv={hkv},end={end_pos}) "
            f"strides=q{tuple(int(x) for x in query.stride())} "
            f"k{tuple(int(x) for x in key.stride())} v{tuple(int(x) for x in value.stride())} "
            f"abs_q=max/mean/norm:{q_abs[0]:.4f}/{q_abs[1]:.4f}/{q_abs[2]:.4f} "
            f"abs_k=max/mean/norm:{k_abs[0]:.4f}/{k_abs[1]:.4f}/{k_abs[2]:.4f} "
            f"abs_v=max/mean/norm:{v_abs[0]:.4f}/{v_abs[1]:.4f}/{v_abs[2]:.4f} "
            f"score_top={[(int(i), float(v)) for i, v in zip(top_scores.indices.detach().cpu().tolist(), top_scores.values.detach().cpu().tolist())]} "
            f"score_margin={score_margin:.6f} prob_mass4={prob_mass4:.6f} "
            f"stage_ref_max={float(stage_ref_diff.max().item()):.6f} "
            f"torch_bf16_max={float(torch_bf16_diff.max().item()):.6f} "
            f"stage_bf16_max={float(stage_bf16_diff.max().item()):.6f} "
            f"out_max={float(out_diff.max().item()):.6f} out_mean={float(out_diff.mean().item()):.6f}",
            flush=True,
        )
        dump_path = os.getenv("AICAS_SPEC_CHAIN_ATTN_DUMP", "").strip()
        if dump_path:
            torch.save(
                {
                    "step": int(step),
                    "layer": layer_idx,
                    "backend": backend_name,
                    "query": query.detach().cpu(),
                    "key": key.detach().cpu(),
                    "value": value.detach().cpu(),
                    "out": out.detach().cpu(),
                    "ref": ref.detach().cpu(),
                    "q_idx": int(q_idx),
                    "h_idx": int(h_idx),
                    "hkv": int(hkv),
                    "end_pos": int(end_pos),
                    "scale": float(scale),
                    "query_stride": tuple(int(x) for x in query.stride()),
                    "key_stride": tuple(int(x) for x in key.stride()),
                    "value_stride": tuple(int(x) for x in value.stride()),
                },
                dump_path,
            )
            print(f"[spec_attn][stage] dumped={dump_path}", flush=True)
    except Exception as exc:
        print(f"[spec_attn][stage] step={step} error={type(exc).__name__}: {exc}", flush=True)


def _tree_root_stage_debug(
    *,
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout: float,
    scaling,
) -> None:
    if os.getenv("AICAS_SPEC_TREE_ROOT_STAGE_DEBUG", "0") != "1":
        return
    _TREE_ROOT_STAGE_STATE["count"] += 1
    step = int(_TREE_ROOT_STAGE_STATE["count"])
    start = int(os.getenv("AICAS_SPEC_TREE_ROOT_STAGE_START", "0"))
    limit = int(os.getenv("AICAS_SPEC_TREE_ROOT_STAGE_LIMIT", "16"))
    if step < start or step >= start + max(1, limit):
        return
    try:
        q_len = int(query.shape[-2])
        kv_len = int(key.shape[-2])
        groups = int(query.shape[1]) // int(key.shape[1])
        root_q = query[:, :, 0:1, :]
        if groups <= 1:
            key_rep = key
            value_rep = value
        else:
            key_rep = key[:, :, None, :, :].expand(
                int(key.shape[0]),
                int(key.shape[1]),
                groups,
                int(key.shape[2]),
                int(key.shape[3]),
            ).reshape(
                int(key.shape[0]),
                int(key.shape[1]) * groups,
                int(key.shape[2]),
                int(key.shape[3]),
            )
            value_rep = value[:, :, None, :, :].expand(
                int(value.shape[0]),
                int(value.shape[1]),
                groups,
                int(value.shape[2]),
                int(value.shape[3]),
            ).reshape(
                int(value.shape[0]),
                int(value.shape[1]) * groups,
                int(value.shape[2]),
                int(value.shape[3]),
            )
        scale = float(scaling) if scaling is not None else float(getattr(module, "scaling", 1.0 / math.sqrt(int(query.shape[-1]))))
        mask = attention_mask[:, :, :, :kv_len]
        root_mask = mask[:, :, 0:1, :]
        mask_flat = root_mask.reshape(-1)
        # Qwen causal masks use finfo.min for hidden positions, which is finite.
        visible = mask_flat.gt(torch.finfo(root_mask.dtype).min / 2)
        visible_idx = torch.nonzero(visible, as_tuple=False).reshape(-1)
        if int(visible_idx.numel()) == 0:
            print(f"[spec_attn][tree_root_stage] step={step} no_visible", flush=True)
            return

        tree_scores = torch.matmul(query, key_rep.transpose(2, 3)) * scale
        tree_scores = tree_scores + mask
        tree_probs = torch.softmax(tree_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        tree_out = torch.matmul(tree_probs, value_rep).transpose(1, 2).contiguous()
        tree_root = tree_out[:, 0:1, :, :]

        base_k = key_rep.index_select(2, visible_idx)
        base_v = value_rep.index_select(2, visible_idx)
        base_scores = torch.matmul(root_q, base_k.transpose(2, 3)) * scale
        base_probs = torch.softmax(base_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        base_out = torch.matmul(base_probs, base_v).transpose(1, 2).contiguous()

        qf = root_q.float()
        kf = base_k.float()
        vf = base_v.float()
        fp32_scores = torch.matmul(qf, kf.transpose(2, 3)) * scale
        fp32_probs = torch.softmax(fp32_scores, dim=-1, dtype=torch.float32)
        fp32_out = torch.matmul(fp32_probs, vf).transpose(1, 2).contiguous()

        tree_base = (tree_root.float() - base_out.float()).abs()
        base_fp32 = (base_out.float() - fp32_out.float()).abs()
        tree_fp32 = (tree_root.float() - fp32_out.float()).abs()
        score_root = tree_scores[:, :, 0:1, :].index_select(3, visible_idx)
        score_delta = (score_root.float() - base_scores.float()).abs()
        prob_root = tree_probs[:, :, 0:1, :].index_select(3, visible_idx)
        prob_delta = (prob_root.float() - base_probs.float()).abs()
        layer_idx = getattr(module, "layer_idx", None)
        print(
            "[spec_attn][tree_root_stage] "
            f"step={step} layer={layer_idx} q={q_len} kv={kv_len} visible={int(visible_idx.numel())} "
            f"tree_base_out_max={float(tree_base.max().item()):.6f} "
            f"tree_base_out_mean={float(tree_base.mean().item()):.6f} "
            f"score_max={float(score_delta.max().item()):.6f} "
            f"prob_max={float(prob_delta.max().item()):.8f} "
            f"base_fp32_out_max={float(base_fp32.max().item()):.6f} "
            f"tree_fp32_out_max={float(tree_fp32.max().item()):.6f} "
            f"dropout={dropout}",
            flush=True,
        )
    except Exception as exc:
        print(f"[spec_attn][tree_root_stage] step={step} error={type(exc).__name__}: {exc}", flush=True)


@triton.jit
def _spec_decode_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    scale: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qq: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_oq: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    BLOCK_S: tl.constexpr,
    D: tl.constexpr,
):
    pid_qh = tl.program_id(0)
    q_idx = pid_qh // 16
    hq = pid_qh - q_idx * 16
    hkv = hq // 2
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + hq * stride_qh + q_idx * stride_qq + offs_d * stride_qd).to(tl.float32)

    m = tl.full((), -3.4028234663852886e38, tl.float32)
    l = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((D,), tl.float32)

    end_pos = kv_len - q_len + q_idx + 1
    for s0 in range(0, kv_len, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        k = tl.load(
            k_ptr + hkv * stride_kh + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=offs_s[:, None] < end_pos,
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * scale
        scores = tl.where(offs_s < end_pos, scores, -3.4028234663852886e38)

        m_new = tl.maximum(m, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m - m_new)
        v = tl.load(
            v_ptr + hkv * stride_vh + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=offs_s[:, None] < end_pos,
            other=0.0,
        ).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l = l * alpha + tl.sum(p, axis=0)
        m = m_new

    out = acc / l
    tl.store(out_ptr + q_idx * stride_oq + hq * stride_oh + offs_d * stride_od, out)


@triton.jit
def _spec_tree_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    q_len: tl.constexpr,
    kv_len: tl.constexpr,
    scale: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qq: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_mq: tl.constexpr,
    stride_mk: tl.constexpr,
    stride_oq: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_od: tl.constexpr,
    BLOCK_S: tl.constexpr,
    D: tl.constexpr,
    DOT_M: tl.constexpr,
    ROUND_SCORES_BF16: tl.constexpr,
):
    pid_qh = tl.program_id(0)
    q_idx = pid_qh // 16
    hq = pid_qh - q_idx * 16
    hkv = hq // 2
    offs_d = tl.arange(0, D)
    offs_m = tl.arange(0, DOT_M)
    q = tl.load(q_ptr + hq * stride_qh + q_idx * stride_qq + offs_d * stride_qd)

    m = tl.full((), -3.4028234663852886e38, tl.float32)
    for s0 in range(0, kv_len, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        valid = offs_s < kv_len
        k = tl.load(
            k_ptr + hkv * stride_kh + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=valid[:, None],
            other=0.0,
        )
        q_mat = tl.broadcast_to(q[None, :], (DOT_M, D))
        qk = tl.dot(q_mat, tl.trans(k))
        scores = tl.sum(tl.where(offs_m[:, None] == 0, qk, 0.0), axis=0)
        if ROUND_SCORES_BF16:
            scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = scores * scale
        if ROUND_SCORES_BF16:
            scores = scores.to(tl.bfloat16).to(tl.float32)
        mask_vals = tl.load(mask_ptr + q_idx * stride_mq + offs_s * stride_mk, mask=valid, other=-3.4028234663852886e38)
        scores = scores + mask_vals.to(tl.float32)
        scores = tl.where(valid, scores, -3.4028234663852886e38)
        m = tl.maximum(m, tl.max(scores, axis=0))

    l = tl.full((), 0.0, tl.float32)
    for s0 in range(0, kv_len, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        valid = offs_s < kv_len
        k = tl.load(
            k_ptr + hkv * stride_kh + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=valid[:, None],
            other=0.0,
        )
        q_mat = tl.broadcast_to(q[None, :], (DOT_M, D))
        qk = tl.dot(q_mat, tl.trans(k))
        scores = tl.sum(tl.where(offs_m[:, None] == 0, qk, 0.0), axis=0)
        if ROUND_SCORES_BF16:
            scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = scores * scale
        if ROUND_SCORES_BF16:
            scores = scores.to(tl.bfloat16).to(tl.float32)
        mask_vals = tl.load(mask_ptr + q_idx * stride_mq + offs_s * stride_mk, mask=valid, other=-3.4028234663852886e38)
        scores = scores + mask_vals.to(tl.float32)
        scores = tl.where(valid, scores, -3.4028234663852886e38)
        l += tl.sum(tl.exp(scores - m), axis=0)

    acc = tl.zeros((D,), tl.float32)
    for s0 in range(0, kv_len, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        valid = offs_s < kv_len
        k = tl.load(
            k_ptr + hkv * stride_kh + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd,
            mask=valid[:, None],
            other=0.0,
        )
        q_mat = tl.broadcast_to(q[None, :], (DOT_M, D))
        qk = tl.dot(q_mat, tl.trans(k))
        scores = tl.sum(tl.where(offs_m[:, None] == 0, qk, 0.0), axis=0)
        if ROUND_SCORES_BF16:
            scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = scores * scale
        if ROUND_SCORES_BF16:
            scores = scores.to(tl.bfloat16).to(tl.float32)
        mask_vals = tl.load(mask_ptr + q_idx * stride_mq + offs_s * stride_mk, mask=valid, other=-3.4028234663852886e38)
        scores = scores + mask_vals.to(tl.float32)
        scores = tl.where(valid, scores, -3.4028234663852886e38)
        p = tl.exp(scores - m) / l
        if ROUND_SCORES_BF16:
            p = p.to(tl.bfloat16)
        v = tl.load(
            v_ptr + hkv * stride_vh + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=valid[:, None],
            other=0.0,
        )
        p_mat = tl.broadcast_to(p.to(tl.bfloat16)[None, :], (DOT_M, BLOCK_S))
        pv = tl.dot(p_mat, v)
        acc += tl.sum(tl.where(offs_m[:, None] == 0, pv, 0.0), axis=0)

    tl.store(out_ptr + q_idx * stride_oq + hq * stride_oh + offs_d * stride_od, acc)


def _can_use_spec_attn(query, key, value, attention_mask, dropout, scaling) -> bool:
    if os.getenv("AICAS_SPEC_MT_ATTN", "0") != "1":
        return False
    if attention_mask is not None or dropout != 0.0:
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        return False
    if query.dtype not in (torch.bfloat16, torch.float16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        return False
    if query.shape[1] != 16 or key.shape[1] != 8 or value.shape[1] != 8:
        return False
    if query.shape[-1] != 128 or key.shape[-1] != 128 or value.shape[-1] != 128:
        return False
    if int(query.shape[-2]) < 1 or int(query.shape[-2]) > int(os.getenv("AICAS_SPEC_MT_ATTN_MAX_Q", "16")):
        return False
    if key.shape[-2] != value.shape[-2] or int(key.shape[-2]) < int(query.shape[-2]):
        return False
    expected = 1.0 / math.sqrt(128.0)
    if scaling is not None and abs(float(scaling) - expected) > 1e-6:
        return False
    return True


def _can_use_spec_tree_attn(query, key, value, attention_mask, dropout, scaling) -> bool:
    if os.getenv("AICAS_SPEC_MT_ATTN", "0") != "1":
        return False
    if os.getenv("AICAS_SPEC_TREE_ATTN", "0") != "1":
        return False
    if attention_mask is None or dropout != 0.0:
        return False
    if not isinstance(attention_mask, torch.Tensor):
        return False
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or attention_mask.ndim != 4:
        return False
    if not (query.is_cuda and key.is_cuda and value.is_cuda and attention_mask.is_cuda):
        return False
    if query.dtype not in (torch.bfloat16, torch.float16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    if query.dtype != torch.bfloat16:
        return False
    if attention_mask.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        return False
    if query.shape[1] != 16 or key.shape[1] != 8 or value.shape[1] != 8:
        return False
    if query.shape[-1] != 128 or key.shape[-1] != 128 or value.shape[-1] != 128:
        return False
    q_len = int(query.shape[-2])
    kv_len = int(key.shape[-2])
    if q_len < 1 or q_len > int(os.getenv("AICAS_SPEC_MT_ATTN_MAX_Q", "16")):
        return False
    if key.shape[-2] != value.shape[-2] or kv_len < q_len:
        return False
    if attention_mask.shape[0] != 1 or attention_mask.shape[1] != 1:
        return False
    mask_q = int(attention_mask.shape[2])
    mask_k = int(attention_mask.shape[3])
    if mask_q != q_len:
        return False
    if mask_k < q_len:
        return False
    expected = 1.0 / math.sqrt(128.0)
    if scaling is not None and abs(float(scaling) - expected) > 1e-6:
        return False
    return True


def _get_flashdecode_runner(max_s: int, device_index: int):
    global _FLASHDECODE_RUNNER
    if _load_flashdecode_ext is None:
        return None
    runner_key = (int(max_s), int(device_index))
    if _FLASHDECODE_RUNNER is None or _FLASHDECODE_RUNNER[0] != runner_key:
        ext = _load_flashdecode_ext()
        _FLASHDECODE_RUNNER = (runner_key, ext.FlashDecodeRunner(1, 8, 2, max_s, device_index))
    return _FLASHDECODE_RUNNER[1]


def _logical_kv_len_from_module(module, q_len: int, kv_len: int) -> int:
    if os.getenv("AICAS_SPEC_MT_ATTN_TRIM_STATIC_KV", "1") != "1":
        return kv_len
    cache_position = getattr(module, "_aicas_spec_cache_position", None)
    if isinstance(cache_position, torch.Tensor) and cache_position.numel() > 0:
        try:
            logical = int(cache_position.reshape(-1)[-1].item()) + 1
            if q_len <= logical <= kv_len:
                return logical
        except Exception:
            pass
    return kv_len


def _trim_logical_kv(module, key, value, q_len: int):
    kv_len = int(key.shape[-2])
    logical_kv_len = _logical_kv_len_from_module(module, q_len, kv_len)
    if logical_kv_len < kv_len:
        key = key[:, :, :logical_kv_len, :]
        value = value[:, :, :logical_kv_len, :]
        _trace_once(
            "chain_static_trim",
            f"[spec_mt_attn] logical kv trim {kv_len}->{logical_kv_len} q_len={q_len}",
        )
    return key, value


def _flashdecode_spec_attn(module, query, key, value):
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    max_s = int(os.getenv("AICAS_FLASHDECODE_MAX_S", "4096"))
    if query.dtype == torch.bfloat16:
        os.environ.setdefault("AICAS_FLASHDECODE_BF16_ATTENTION", "1")
    runner = _get_flashdecode_runner(max_s, device_index)
    if runner is None:
        return None
    if not hasattr(runner, "run_decode_multi_into"):
        return None
    q = query.contiguous()
    q_len = int(q.shape[-2])
    key, value = _trim_logical_kv(module, key, value, q_len)
    k = key.contiguous()
    v = value.contiguous()
    out_key = (device_index, q.dtype, q_len)
    out = _FLASHDECODE_OUT.get(out_key)
    if out is None or out.device != q.device:
        out = torch.empty((1, q_len, 16, 128), device=q.device, dtype=q.dtype)
        _FLASHDECODE_OUT[out_key] = out
    runner.run_decode_multi_into(q, k, v, out)
    if (
        os.getenv("AICAS_SPEC_MT_ATTN_LOG_HIT", "0") == "1"
        and not _BF16_EXT_STATS.get("logged_flashdecode", False)
    ):
        print(
            "[spec_mt_attn] flashdecode first hit: "
            f"q_len={q_len}, kv_len={int(k.shape[-2])}, dtype={q.dtype}"
        )
        _BF16_EXT_STATS["logged_flashdecode"] = True
    return out


def _flashdecode_tree_spec_attn(query, key, value, attention_mask):
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    max_s = int(os.getenv("AICAS_FLASHDECODE_MAX_S", "4096"))
    if query.dtype == torch.bfloat16:
        os.environ.setdefault("AICAS_FLASHDECODE_BF16_ATTENTION", "1")
    runner = _get_flashdecode_runner(max_s, device_index)
    if runner is None or not hasattr(runner, "run_decode_multi_masked_into"):
        return None

    physical_kv_len = int(key.shape[-2])
    logical_kv_len = int(attention_mask.shape[-1])
    if logical_kv_len < physical_kv_len:
        key = key[:, :, :logical_kv_len, :]
        value = value[:, :, :logical_kv_len, :]
        _trace_once(
            "flashdecode_tree_static_trim",
            f"[spec_mt_attn] flashdecode tree logical kv trim {physical_kv_len}->{logical_kv_len} "
            f"q_len={int(query.shape[-2])}",
        )
    elif logical_kv_len > physical_kv_len:
        attention_mask = attention_mask[..., :physical_kv_len]

    q = query if query.is_contiguous() else query.contiguous()
    k = key if key.is_contiguous() else key.contiguous()
    v = value if value.is_contiguous() else value.contiguous()
    mask = attention_mask
    if mask.dtype != q.dtype:
        mask = mask.to(dtype=q.dtype)
    elif not mask.is_contiguous():
        mask = mask.contiguous()

    q_len = int(q.shape[-2])
    out_key = ("tree", device_index, q.dtype, q_len)
    out = _FLASHDECODE_OUT.get(out_key)
    if out is None or out.device != q.device:
        out = torch.empty((1, q_len, 16, 128), device=q.device, dtype=q.dtype)
        _FLASHDECODE_OUT[out_key] = out
    runner.run_decode_multi_masked_into(q, k, v, mask, out)
    if (
        os.getenv("AICAS_SPEC_MT_ATTN_LOG_HIT", "0") == "1"
        and not _BF16_EXT_STATS.get("logged_flashdecode_tree", False)
    ):
        print(
            "[spec_mt_attn] flashdecode tree first hit: "
            f"q_len={q_len}, kv_len={int(k.shape[-2])}, dtype={q.dtype}"
        )
        _BF16_EXT_STATS["logged_flashdecode_tree"] = True
    return out


def _bf16_ext_spec_attn(query, key, value):
    global _BF16_EXT_UNAVAILABLE
    if _BF16_EXT_UNAVAILABLE:
        return None
    try:
        from spec_decode.kernels.bf16_verify_attn.runtime import run_bf16_verify_attention
    except Exception as exc:
        _BF16_EXT_UNAVAILABLE = True
        if os.getenv("AICAS_SPEC_MT_ATTN_REQUIRE_BF16_EXT", "0") == "1":
            raise RuntimeError("bf16 verifier attention extension unavailable") from exc
        return None
    try:
        out = run_bf16_verify_attention(query, key, value)
        if out is not None:
            _BF16_EXT_STATS["hit"] += 1
            if (
                os.getenv("AICAS_SPEC_MT_ATTN_LOG_HIT", "0") == "1"
                and not _BF16_EXT_STATS["logged_hit"]
            ):
                print(
                    "[spec_mt_attn] bf16_ext first hit: "
                    f"q_len={int(query.shape[-2])}, kv_len={int(key.shape[-2])}, dtype={query.dtype}"
                )
                _BF16_EXT_STATS["logged_hit"] = True
        return out
    except Exception as exc:
        _BF16_EXT_STATS["fallback"] += 1
        if os.getenv("AICAS_SPEC_MT_ATTN_REQUIRE_BF16_EXT", "0") == "1":
            raise
        if os.getenv("AICAS_SPEC_MT_ATTN_LOG_FALLBACK", "0") == "1":
            print(f"[spec_mt_attn] bf16_ext fallback: {type(exc).__name__}: {exc}")
        return None


def _triton_tree_spec_attn(query, key, value, attention_mask):
    physical_kv_len = int(key.shape[-2])
    logical_kv_len = int(attention_mask.shape[-1])
    if logical_kv_len < physical_kv_len:
        key = key[:, :, :logical_kv_len, :]
        value = value[:, :, :logical_kv_len, :]
        _trace_once(
            "tree_static_trim",
            f"[spec_mt_attn] tree logical kv trim {physical_kv_len}->{logical_kv_len} "
            f"q_len={int(query.shape[-2])}",
        )
    q = query
    k = key
    v = value
    mask = attention_mask[:, :, :, : int(k.shape[-2])]
    q_len = int(q.shape[-2])
    kv_len = int(k.shape[-2])
    out = torch.empty((1, q_len, 16, 128), device=q.device, dtype=q.dtype)

    block_s = int(os.getenv("AICAS_SPEC_MT_ATTN_BLOCK_S", "64"))
    _spec_tree_attn_kernel[(q_len * 16,)](
        q,
        k,
        v,
        mask,
        out,
        q_len=q_len,
        kv_len=kv_len,
        scale=float(1.0 / math.sqrt(128.0)),
        stride_qh=int(q.stride(1)),
        stride_qq=int(q.stride(2)),
        stride_qd=int(q.stride(3)),
        stride_kh=int(k.stride(1)),
        stride_ks=int(k.stride(2)),
        stride_kd=int(k.stride(3)),
        stride_vh=int(v.stride(1)),
        stride_vs=int(v.stride(2)),
        stride_vd=int(v.stride(3)),
        stride_mq=int(mask.stride(2)),
        stride_mk=int(mask.stride(3)),
        stride_oq=int(out.stride(1)),
        stride_oh=int(out.stride(2)),
        stride_od=int(out.stride(3)),
        BLOCK_S=block_s,
        D=128,
        DOT_M=16,
        ROUND_SCORES_BF16=os.getenv("AICAS_SPEC_TREE_ATTN_ROUND_BF16", "0") == "1",
        num_warps=4,
        num_stages=3,
    )
    if os.getenv("AICAS_SPEC_TREE_ATTN_CHECK", "0") == "1":
        groups = int(q.shape[1]) // int(k.shape[1])
        ref_k = k[:, :, None, :, :].expand(-1, -1, groups, -1, -1).reshape(
            int(k.shape[0]),
            int(k.shape[1]) * groups,
            int(k.shape[2]),
            int(k.shape[3]),
        )
        ref_v = v[:, :, None, :, :].expand(-1, -1, groups, -1, -1).reshape(
            int(v.shape[0]),
            int(v.shape[1]) * groups,
            int(v.shape[2]),
            int(v.shape[3]),
        )
        ref_scores = torch.matmul(q, ref_k.transpose(2, 3)) * float(1.0 / math.sqrt(128.0))
        ref_scores = ref_scores + mask
        ref_probs = torch.softmax(ref_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        ref = torch.matmul(ref_probs, ref_v).transpose(1, 2).contiguous()
        diff = (out.float() - ref.float()).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        tol = float(os.getenv("AICAS_SPEC_TREE_ATTN_CHECK_TOL", "0.125"))
        if max_diff > tol:
            max_idx = int(diff.reshape(-1).argmax().item())
            dim = int(diff.shape[-1])
            head_count = int(diff.shape[2])
            q_bad = (max_idx // (head_count * dim)) % q_len
            h_bad = (max_idx // dim) % head_count
            d_bad = max_idx % dim
            raise RuntimeError(
                "triton tree attention mismatch against eager reference "
                f"q={q_len} kv={kv_len} max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
                f"at=(q={q_bad}, h={h_bad}, d={d_bad}) "
                f"got={float(out[0, q_bad, h_bad, d_bad].item()):.6f} "
                f"ref={float(ref[0, q_bad, h_bad, d_bad].item()):.6f}"
            )
    return out


class _SpecAttentionBackend:
    name = "base"

    def supports(self, module, query, key, value, attention_mask, dropout, scaling) -> bool:
        raise NotImplementedError

    def run(self, module, query, key, value, attention_mask, dropout, scaling):
        raise NotImplementedError


class _Bf16ExtChainBackend(_SpecAttentionBackend):
    name = "bf16_ext_chain"

    def supports(self, module, query, key, value, attention_mask, dropout, scaling) -> bool:
        return _can_use_spec_attn(query, key, value, attention_mask, dropout, scaling)

    def run(self, module, query, key, value, attention_mask, dropout, scaling):
        key, value = _trim_logical_kv(module, key, value, int(query.shape[-2]))
        out = _bf16_ext_spec_attn(query, key, value)
        return out


class _FlashDecodeChainBackend(_SpecAttentionBackend):
    name = "flashdecode_chain"

    def supports(self, module, query, key, value, attention_mask, dropout, scaling) -> bool:
        return _can_use_spec_attn(query, key, value, attention_mask, dropout, scaling)

    def run(self, module, query, key, value, attention_mask, dropout, scaling):
        return _flashdecode_spec_attn(module, query, key, value)


class _TritonChainBackend(_SpecAttentionBackend):
    name = "triton_chain"

    def supports(self, module, query, key, value, attention_mask, dropout, scaling) -> bool:
        return _can_use_spec_attn(query, key, value, attention_mask, dropout, scaling)

    def run(self, module, query, key, value, attention_mask, dropout, scaling):
        q = query.contiguous()
        key, value = _trim_logical_kv(module, key, value, int(q.shape[-2]))
        k = key.contiguous()
        v = value.contiguous()
        q_len = int(q.shape[-2])
        kv_len = int(k.shape[-2])
        out = torch.empty((1, q_len, 16, 128), device=q.device, dtype=q.dtype)

        block_s = int(os.getenv("AICAS_SPEC_MT_ATTN_BLOCK_S", "64"))
        _spec_decode_attn_kernel[(q_len * 16,)](
            q,
            k,
            v,
            out,
            q_len=q_len,
            kv_len=kv_len,
            scale=float(1.0 / math.sqrt(128.0)),
            stride_qh=int(q.stride(1)),
            stride_qq=int(q.stride(2)),
            stride_qd=int(q.stride(3)),
            stride_kh=int(k.stride(1)),
            stride_ks=int(k.stride(2)),
            stride_kd=int(k.stride(3)),
            stride_vh=int(v.stride(1)),
            stride_vs=int(v.stride(2)),
            stride_vd=int(v.stride(3)),
            stride_oq=int(out.stride(1)),
            stride_oh=int(out.stride(2)),
            stride_od=int(out.stride(3)),
            BLOCK_S=block_s,
            D=128,
            num_warps=4,
            num_stages=3,
        )
        return out


def _torch_chain_spec_attn(module, query, key, value):
    q_len = int(query.shape[-2])
    key, value = _trim_logical_kv(module, key, value, q_len)
    groups = int(query.shape[1]) // int(key.shape[1])
    if groups <= 1:
        key_rep = key
        value_rep = value
    else:
        key_rep = key[:, :, None, :, :].expand(
            int(key.shape[0]),
            int(key.shape[1]),
            groups,
            int(key.shape[2]),
            int(key.shape[3]),
        ).reshape(
            int(key.shape[0]),
            int(key.shape[1]) * groups,
            int(key.shape[2]),
            int(key.shape[3]),
        )
        value_rep = value[:, :, None, :, :].expand(
            int(value.shape[0]),
            int(value.shape[1]),
            groups,
            int(value.shape[2]),
            int(value.shape[3]),
        ).reshape(
            int(value.shape[0]),
            int(value.shape[1]) * groups,
            int(value.shape[2]),
            int(value.shape[3]),
        )
    scale = float(getattr(module, "scaling", 1.0 / math.sqrt(int(query.shape[-1]))))
    attn_weights = torch.matmul(query, key_rep.transpose(2, 3)) * scale
    if q_len > 1:
        kv_len = int(key.shape[-2])
        prefix_len = kv_len - q_len
        q_pos = torch.arange(q_len, device=query.device, dtype=torch.long).view(1, 1, q_len, 1)
        k_pos = torch.arange(kv_len, device=query.device, dtype=torch.long).view(1, 1, 1, kv_len)
        visible = k_pos <= (prefix_len + q_pos)
        attn_weights = attn_weights.masked_fill(~visible, torch.finfo(attn_weights.dtype).min)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(attn_weights, value_rep)
    return out.transpose(1, 2).contiguous()


class _TorchChainBackend(_SpecAttentionBackend):
    name = "torch_chain"

    def supports(self, module, query, key, value, attention_mask, dropout, scaling) -> bool:
        return _can_use_spec_attn(query, key, value, attention_mask, dropout, scaling)

    def run(self, module, query, key, value, attention_mask, dropout, scaling):
        return _torch_chain_spec_attn(module, query, key, value)


class _TritonTreeBackend(_SpecAttentionBackend):
    name = "triton_tree"

    def supports(self, module, query, key, value, attention_mask, dropout, scaling) -> bool:
        return _can_use_spec_tree_attn(query, key, value, attention_mask, dropout, scaling)

    def run(self, module, query, key, value, attention_mask, dropout, scaling):
        return _triton_tree_spec_attn(query, key, value, attention_mask)


class _FlashDecodeTreeBackend(_SpecAttentionBackend):
    name = "flashdecode_tree"

    def supports(self, module, query, key, value, attention_mask, dropout, scaling) -> bool:
        return _can_use_spec_tree_attn(query, key, value, attention_mask, dropout, scaling)

    def run(self, module, query, key, value, attention_mask, dropout, scaling):
        return _flashdecode_tree_spec_attn(query, key, value, attention_mask)


_SPEC_ATTENTION_BACKENDS: dict[str, _SpecAttentionBackend] = {
    "bf16_ext_chain": _Bf16ExtChainBackend(),
    "flashdecode_chain": _FlashDecodeChainBackend(),
    "triton_chain": _TritonChainBackend(),
    "torch_chain": _TorchChainBackend(),
    "flashdecode_tree": _FlashDecodeTreeBackend(),
    "triton_tree": _TritonTreeBackend(),
}


def _preferred_backend_names(attention_mask) -> list[str]:
    if attention_mask is not None:
        raw = os.getenv("AICAS_SPEC_TREE_ATTN_BACKEND", "flashdecode_tree")
        aliases = {
            "flashdecode": "flashdecode_tree",
            "fd": "flashdecode_tree",
            "triton": "triton_tree",
        }
        raw = aliases.get(raw.strip().lower(), raw)
        defaults = ["flashdecode_tree"]
    else:
        raw = os.getenv("AICAS_SPEC_MT_ATTN_BACKEND", "bf16_ext")
        aliases = {
            "bf16": "bf16_ext_chain",
            "cuda_bf16": "bf16_ext_chain",
            "bf16_ext": "bf16_ext_chain",
            "flashdecode": "flashdecode_chain",
            "fd": "flashdecode_chain",
            "triton": "triton_chain",
            "torch": "torch_chain",
            "eager_chain": "torch_chain",
        }
        raw = aliases.get(raw.strip().lower(), raw)
        defaults = ["bf16_ext_chain", "triton_chain"]
    names = []
    for part in raw.replace("+", ",").split(","):
        name = part.strip().lower()
        if name:
            names.append(name)
    for name in defaults:
        if name not in names:
            names.append(name)
    return names


def _run_registered_spec_attention(module, query, key, value, attention_mask, dropout, scaling):
    tried = []
    for name in _preferred_backend_names(attention_mask):
        backend = _SPEC_ATTENTION_BACKENDS.get(name)
        if backend is None:
            tried.append(f"{name}:missing")
            continue
        if not backend.supports(module, query, key, value, attention_mask, dropout, scaling):
            tried.append(f"{name}:unsupported")
            continue
        out = backend.run(module, query, key, value, attention_mask, dropout, scaling)
        if out is None:
            tried.append(f"{name}:unavailable")
            continue
        if attention_mask is None and os.getenv("AICAS_SPEC_CHAIN_ATTN_CHECK", "0") == "1":
            _CHAIN_CHECK_STATE["count"] += 1
            step = int(_CHAIN_CHECK_STATE["count"])
            start = int(os.getenv("AICAS_SPEC_CHAIN_ATTN_CHECK_START", "0"))
            limit = int(os.getenv("AICAS_SPEC_CHAIN_ATTN_CHECK_LIMIT", "16"))
            if step >= start and step < start + max(1, limit):
                ref, _ = _eager_attention_fallback(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    dropout=dropout,
                    scaling=scaling,
                )
                alt_msg = ""
                if os.getenv("AICAS_SPEC_CHAIN_ATTN_COMPARE_TRITON", "0") == "1" and backend.name != "triton_chain":
                    alt_backend = _SPEC_ATTENTION_BACKENDS.get("triton_chain")
                    if alt_backend is not None and alt_backend.supports(module, query, key, value, attention_mask, dropout, scaling):
                        try:
                            alt = alt_backend.run(module, query, key, value, attention_mask, dropout, scaling)
                            if alt is not None:
                                alt_diff = (alt.float() - ref.float()).abs()
                                alt_msg = (
                                    f" triton_max={float(alt_diff.max().item()):.6f}"
                                    f" triton_mean={float(alt_diff.mean().item()):.6f}"
                                )
                        except Exception as exc:
                            alt_msg = f" triton_error={type(exc).__name__}:{exc}"
                diff = (out.float() - ref.float()).abs()
                max_diff = float(diff.max().item())
                mean_diff = float(diff.mean().item())
                max_idx = int(diff.reshape(-1).argmax().item())
                d = int(diff.shape[-1])
                h = int(diff.shape[2])
                q_bad = (max_idx // (h * d)) % int(diff.shape[1])
                h_bad = (max_idx // d) % h
                d_bad = max_idx % d
                layer_idx = getattr(module, "layer_idx", None)
                print(
                    "[spec_attn][chain_check] "
                    f"step={step} layer={layer_idx} backend={backend.name} "
                    f"q={int(query.shape[-2])} kv={int(key.shape[-2])} "
                    f"max={max_diff:.6f} mean={mean_diff:.6f} at=(q={q_bad},h={h_bad},d={d_bad}) "
                    f"got={float(out[0, q_bad, h_bad, d_bad].item()):.6f} "
                    f"ref={float(ref[0, q_bad, h_bad, d_bad].item()):.6f}"
                    f"{alt_msg}",
                    flush=True,
                )
                _chain_stage_debug(
                    module=module,
                    backend_name=backend.name,
                    query=query,
                    key=key,
                    value=value,
                    out=out,
                    ref=ref,
                    step=step,
                )
        _trace_once(
            f"backend_{backend.name}",
            f"[spec_attn] backend={backend.name} q={int(query.shape[-2])} kv={int(key.shape[-2])} "
            f"mask={'tree' if attention_mask is not None else 'chain'}",
        )
        return out, backend.name
    raise RuntimeError(
        "speculative verifier attention has no optimized backend for "
        f"query={tuple(query.shape)} key={tuple(key.shape)} "
        f"mask={None if attention_mask is None else tuple(attention_mask.shape)} "
        f"dtype={query.dtype} dropout={dropout}; tried={tried}"
    )


def spec_multitoken_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    if attention_mask is not None:
        _tree_root_stage_debug(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=scaling,
        )
    if attention_mask is None and os.getenv("AICAS_SPEC_MT_ATTN_EAGER_CHAIN", "0") == "1":
        return _eager_attention_fallback(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
    if attention_mask is not None and os.getenv("AICAS_SPEC_TREE_ATTN", "0") != "1":
        return _eager_attention_fallback(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
    try:
        out, _ = _run_registered_spec_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout,
            scaling,
        )
        return out, None
    except RuntimeError:
        if os.getenv("AICAS_SPEC_MT_ATTN_ALLOW_EAGER_FALLBACK", "0") == "1":
            return _eager_attention_fallback(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                **kwargs,
            )
        raise


@contextmanager
def spec_multitoken_attention(model):
    if os.getenv("AICAS_SPEC_MT_ATTN", "0") != "1":
        yield
        return
    from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen

    impl_name = "aicas_spec_mt_attn"
    fallback_impl_name = getattr(model.config, "_attn_implementation", "sdpa")
    if fallback_impl_name == impl_name:
        fallback_impl_name = "sdpa"
    fallback = qwen.ALL_ATTENTION_FUNCTIONS.get_interface(
        fallback_impl_name,
        qwen.eager_attention_forward,
    )

    qwen.ALL_ATTENTION_FUNCTIONS.register(impl_name, spec_multitoken_attention_forward)
    configs = []
    for cfg in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ):
        if cfg is not None and hasattr(cfg, "_attn_implementation") and cfg not in configs:
            configs.append(cfg)
    old = [(cfg, getattr(cfg, "_attn_implementation")) for cfg in configs]
    lm = getattr(getattr(model, "model", None), "language_model", None)
    patched = []
    if lm is not None and hasattr(lm, "layers"):
        for layer in lm.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                patched.append((
                    attn,
                    getattr(attn, "_aicas_spec_mt_attn_fallback", None),
                    getattr(attn, "_aicas_spec_mt_attn_orig_forward", None),
                    getattr(attn, "_aicas_spec_mt_attn_patched_forward", False),
                ))
                setattr(attn, "_aicas_spec_mt_attn_fallback", fallback)
                if not getattr(attn, "_aicas_spec_mt_attn_patched_forward", False):
                    from types import MethodType

                    original_forward = attn.forward

                    def _make_forward(orig):
                        def _forward(
                            self,
                            hidden_states,
                            position_embeddings,
                            attention_mask=None,
                            past_key_values=None,
                            cache_position=None,
                            **kwargs,
                        ):
                            old_cache_position = getattr(self, "_aicas_spec_cache_position", None)
                            had_old = hasattr(self, "_aicas_spec_cache_position")
                            self._aicas_spec_cache_position = cache_position
                            try:
                                return orig(
                                    hidden_states,
                                    position_embeddings=position_embeddings,
                                    attention_mask=attention_mask,
                                    past_key_values=past_key_values,
                                    cache_position=cache_position,
                                    **kwargs,
                                )
                            finally:
                                if had_old:
                                    self._aicas_spec_cache_position = old_cache_position
                                else:
                                    try:
                                        delattr(self, "_aicas_spec_cache_position")
                                    except Exception:
                                        pass

                        return _forward

                    attn._aicas_spec_mt_attn_orig_forward = original_forward
                    attn.forward = MethodType(_make_forward(original_forward), attn)
                    attn._aicas_spec_mt_attn_patched_forward = True
    for cfg, _ in old:
        cfg._attn_implementation = impl_name
    try:
        yield
    finally:
        for cfg, value in old:
            cfg._attn_implementation = value
        for attn, old_fallback, old_forward, had_patched_forward in patched:
            if old_fallback is None:
                try:
                    delattr(attn, "_aicas_spec_mt_attn_fallback")
                except Exception:
                    pass
            else:
                setattr(attn, "_aicas_spec_mt_attn_fallback", old_fallback)
            if not had_patched_forward and old_forward is not None:
                attn.forward = old_forward
                try:
                    delattr(attn, "_aicas_spec_mt_attn_orig_forward")
                    delattr(attn, "_aicas_spec_mt_attn_patched_forward")
                except Exception:
                    pass
