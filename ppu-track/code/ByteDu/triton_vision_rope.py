"""Fused vision rotary embedding. Replaces apply_rotary_pos_emb_vision (fp32 upcast +
rotate_half slice/cat + ~6 elementwise kernels over [1,S,H,D]) with a single Triton
kernel. Reads strided q/k bf16 directly (no .contiguous() copy), computes in fp32,
writes contiguous bf16 — matching the reference's fp32-compute / bf16-store exactly.

Layout: q,k are [1, S, H, D] (D=head_dim=64). cos/sin are [S, D] where the second half
duplicates the first (emb = cat(freqs, freqs)), so cos[:, i] == cos[:, i+D/2]. Thus:
  out[..., :D/2] = q0*c - q1*s
  out[..., D/2:] = q1*c + q0*s     with c=cos[:, :D/2], s=sin[:, :D/2]
"""
import os
import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/junkrat_triton_cache")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None; tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _vision_rope_qk_kernel(
        q_ptr, k_ptr, qo_ptr, ko_ptr, cos_ptr, sin_ptr,
        s_stride_q, h_stride_q, s_stride_k, h_stride_k,
        S, H, HALF: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        s = pid // H
        h = pid % H
        half = tl.arange(0, HALF)
        # cos/sin: [S, D], use first half (second half duplicates it)
        c = tl.load(cos_ptr + s * (2 * HALF) + half).to(tl.float32)
        sn = tl.load(sin_ptr + s * (2 * HALF) + half).to(tl.float32)
        # q
        qb = q_ptr + s * s_stride_q + h * h_stride_q
        q0 = tl.load(qb + half).to(tl.float32)
        q1 = tl.load(qb + HALF + half).to(tl.float32)
        o0 = q0 * c - q1 * sn
        o1 = q1 * c + q0 * sn
        ob = qo_ptr + (s * H + h) * (2 * HALF)
        tl.store(ob + half, o0.to(qo_ptr.dtype.element_ty))
        tl.store(ob + HALF + half, o1.to(qo_ptr.dtype.element_ty))
        # k
        kb = k_ptr + s * s_stride_k + h * h_stride_k
        k0 = tl.load(kb + half).to(tl.float32)
        k1 = tl.load(kb + HALF + half).to(tl.float32)
        p0 = k0 * c - k1 * sn
        p1 = k1 * c + k0 * sn
        pb = ko_ptr + (s * H + h) * (2 * HALF)
        tl.store(pb + half, p0.to(ko_ptr.dtype.element_ty))
        tl.store(pb + HALF + half, p1.to(ko_ptr.dtype.element_ty))


def apply_rotary_pos_emb_vision_fused(query_states, key_states, cos, sin):
    """query/key_states: [1, S, H, D] (may be strided views). cos/sin: [S, D].
    Returns contiguous [1, S, H, D] in the inputs' original dtype."""
    assert query_states.dim() == 4 and query_states.shape[0] == 1
    _, S, H, D = query_states.shape
    half = D // 2
    q = query_states[0]; k = key_states[0]            # [S,H,D] views
    qo = torch.empty((S, H, D), device=q.device, dtype=q.dtype)
    ko = torch.empty((S, H, D), device=k.device, dtype=k.dtype)
    cos = cos.contiguous(); sin = sin.contiguous()
    _vision_rope_qk_kernel[(S * H,)](
        q, k, qo, ko, cos, sin,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        S, H, HALF=half, num_warps=2,
    )
    return qo.view(1, S, H, D), ko.view(1, S, H, D)
