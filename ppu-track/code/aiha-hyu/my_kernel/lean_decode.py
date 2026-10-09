"""
[SHLEE] Lean single-token decode forward pass for Qwen3-VL language model.

Bypasses HuggingFace nn.Module dispatch, pre-extracts all weight tensors,
pre-allocates scratch buffers, and uses FlashInfer fused kernels for norms
and activations.  Uses FA2 for attention (CUDA-graph-safe with StaticCache).
"""

import os
import sys
import torch

try:
    import flashinfer
    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False
    print("[lean_decode] flashinfer not available, using own kernels", file=sys.stderr)

if not _HAS_FLASHINFER:
    from my_kernel.fused_rmsnorm import _load as _load_rmsnorm_ext
    from my_kernel.fused_silu_mul_ver2 import _load as _load_silu_mul_ext
    _load_rmsnorm_ext()
    _load_silu_mul_ext()
    from my_kernel.fused_rmsnorm import _module as _rmsnorm_mod
    from my_kernel.fused_silu_mul_ver2 import _module as _silu_mul_mod

    class _FlashInferShim:
        @staticmethod
        def rmsnorm(x, w, eps, out=None):
            result = _rmsnorm_mod.fused_rmsnorm_forward(x, w, eps)
            if out is not None:
                out.copy_(result)
                return out
            return result

        @staticmethod
        def fused_add_rmsnorm(x, residual, w, eps):
            residual.add_(x)
            result = _rmsnorm_mod.fused_rmsnorm_forward(residual, w, eps)
            x.copy_(result)

        @staticmethod
        def silu_and_mul(gate_up, out=None):
            half = gate_up.shape[-1] // 2

            result = _silu_mul_mod.fused_silu_mul_forward_v2(gate_up, half)
            if out is not None:
                out.copy_(result)
                return out
            return result

    flashinfer = _FlashInferShim()

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_USE_CUSTOM_GEMV = os.environ.get("USE_CUSTOM_GEMV", "1") == "1"
_USE_CUSTOM_ATTN = os.environ.get("USE_CUSTOM_ATTN", "1") == "1"
_USE_ATTN_V2 = os.environ.get("USE_ATTN_V2", "1") == "1"

if os.environ.get("MY_KERNEL_SKIP_FUSED_ROPE", "0") == "1":

    _fused_rope = None
else:
    try:
        from my_kernel.fused_rope_ver6 import fused_apply_rotary_pos_emb as _fused_rope
    except Exception:
        _fused_rope = None

_fused_silu_mul_gemv = None
if _USE_CUSTOM_GEMV:
    try:
        from my_kernel.fast_gemv import (
            gemv as _fast_gemv,
            gemv_addres as _fast_gemv_addres,
            fused_silu_mul_gemv as _fused_silu_mul_gemv,
            _load as _load_gemv,
        )
        _load_gemv()
    except Exception:
        _fast_gemv = None
        _fast_gemv_addres = None
        _fused_silu_mul_gemv = None
else:
    _fast_gemv = None
    _fast_gemv_addres = None

_HAS_FUSED_ATTN = False
if _USE_CUSTOM_ATTN:
    try:
        from my_kernel.fused_attn_decode_v3 import (
            fused_norm_rope_cache_write as _fused_nrc,
            fused_decode_attn as _fused_decode_attn,
            alloc_workspace as _alloc_fused_ws,
            _load as _load_fused_attn,
        )
        _load_fused_attn()
        _HAS_FUSED_ATTN = True
        print("[lean_decode] Using fused_attn_decode_v3 (ptx_exp2)", file=sys.stderr)
    except Exception:
        pass
    if not _HAS_FUSED_ATTN and _USE_ATTN_V2:
        try:
            from my_kernel.fused_attn_decode_v2 import (
                fused_norm_rope_cache_write as _fused_nrc,
                fused_decode_attn as _fused_decode_attn,
                alloc_workspace as _alloc_fused_ws,
                _load as _load_fused_attn,
            )
            _load_fused_attn()
            _HAS_FUSED_ATTN = True
            print("[lean_decode] Using fused_attn_decode_v2 (256-thread)", file=sys.stderr)
        except Exception:
            pass
    if not _HAS_FUSED_ATTN:
        try:
            from my_kernel.fused_attn_decode import (
                fused_norm_rope_cache_write as _fused_nrc,
                fused_decode_attn as _fused_decode_attn,
                alloc_workspace as _alloc_fused_ws,
                _load as _load_fused_attn,
            )
            _load_fused_attn()
            _HAS_FUSED_ATTN = True
        except Exception:
            _HAS_FUSED_ATTN = False

try:
    from my_kernel.decode_attn import decode_attn as _decode_attn, alloc_attn_workspace as _alloc_attn_ws, _load as _load_decode_attn
    _load_decode_attn()
    _HAS_CUSTOM_ATTN = True
except Exception:
    _HAS_CUSTOM_ATTN = False

def _w(mod):
    return mod.weight.data

def _eps(mod):
    return mod.variance_epsilon if hasattr(mod, "variance_epsilon") else mod.eps

class LeanDecodeState:
    """Pre-extracted weights and pre-allocated buffers for lean decode."""

    def __init__(self, model, cache):
        cfg = model.config.get_text_config()
        lang = model.model.language_model
        self.cache = cache

        self.hidden_size = cfg.hidden_size
        self.num_q_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

        self.num_layers = len(lang.layers)
        self.q_dim = self.num_q_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.embed_tokens = lang.embed_tokens
        self.rotary_emb = lang.rotary_emb

        self.k_cache_views = None
        self.v_cache_views = None

        self.final_norm_w = _w(lang.norm)
        self.final_norm_eps = _eps(lang.norm)

        import torch.nn as _nn
        if isinstance(model.lm_head, _nn.Linear):
            self.lm_head_w = model.lm_head.weight.data
            self.lm_head_module = None
        else:
            self.lm_head_w = None
            self.lm_head_module = model.lm_head

        attn_impl = cfg._attn_implementation
        self.attn_fn = ALL_ATTENTION_FUNCTIONS[attn_impl]

        self.ln1_w, self.ln1_eps = [], []
        self.ln2_w, self.ln2_eps = [], []
        self.q_norm_w, self.q_norm_eps = [], []
        self.k_norm_w, self.k_norm_eps = [], []
        self.qkv_w, self.o_w = [], []
        self.gate_up_w, self.down_w = [], []
        self.attn_mods = []

        for layer in lang.layers:
            sa = layer.self_attn
            self.attn_mods.append(sa)

            self.ln1_w.append(_w(layer.input_layernorm))
            self.ln1_eps.append(_eps(layer.input_layernorm))
            self.ln2_w.append(_w(layer.post_attention_layernorm))
            self.ln2_eps.append(_eps(layer.post_attention_layernorm))

            if hasattr(sa, "fused_qkv"):
                fq = sa.fused_qkv
                self.qkv_w.append(fq.qkv_proj.weight.data)
                self.q_norm_w.append(_w(fq.q_norm))
                self.q_norm_eps.append(_eps(fq.q_norm))
                self.k_norm_w.append(_w(fq.k_norm))
                self.k_norm_eps.append(_eps(fq.k_norm))
            else:
                q_w = sa.q_proj.weight.data
                k_w = sa.k_proj.weight.data
                v_w = sa.v_proj.weight.data
                self.qkv_w.append(torch.cat([q_w, k_w, v_w], dim=0))
                self.q_norm_w.append(_w(sa.q_norm))
                self.q_norm_eps.append(_eps(sa.q_norm))
                self.k_norm_w.append(_w(sa.k_norm))
                self.k_norm_eps.append(_eps(sa.k_norm))

            self.o_w.append(sa.o_proj.weight.data)

            mlp = layer.mlp
            if hasattr(mlp, "gate_up_proj"):
                self.gate_up_w.append(mlp.gate_up_proj.weight.data)
                self.down_w.append(mlp.down_proj.weight.data)
            else:
                g_w = mlp.gate_proj.weight.data
                u_w = mlp.up_proj.weight.data
                self.gate_up_w.append(torch.cat([g_w, u_w], dim=0))
                self.down_w.append(mlp.down_proj.weight.data)

        self.intermediate_size = int(self.down_w[0].shape[1])
        for i, (gate_up_w, down_w) in enumerate(zip(self.gate_up_w, self.down_w)):
            if gate_up_w.shape[0] != 2 * self.intermediate_size or down_w.shape[1] != self.intermediate_size:
                raise RuntimeError(
                    f"[lean_decode] MLP shape mismatch at layer {i}: "
                    f"gate_up={tuple(gate_up_w.shape)} down={tuple(down_w.shape)} "
                    f"intermediate={self.intermediate_size}"
                )

        if self.lm_head_w is not None:
            device = self.lm_head_w.device
            dtype = self.lm_head_w.dtype
            vocab = self.lm_head_w.shape[0]
        else:
            device = self.lm_head_module.scales.device
            dtype = self.lm_head_module.scales.dtype
            vocab = self.lm_head_module.out_features
        hs = self.hidden_size
        qkv_total = self.q_dim + 2 * self.kv_dim
        inter2 = self.intermediate_size * 2

        from my_kernel.quant_swap import is_decoder_quant_enabled
        self.decoder_quant = is_decoder_quant_enabled()
        if self.decoder_quant:
            import os
            from my_kernel.fused_w4_gemv import (
                quantize_pack_w4_sym, _load as _load_w4,
            )
            _load_w4()
            self.quant_group = int(os.environ.get("MY_KERNEL_QUANT_GROUP", "128"))
            self.qkv_qw, self.qkv_qs = [], []
            self.o_qw,   self.o_qs   = [], []
            self.gu_qw,  self.gu_qs  = [], []
            self.dn_qw,  self.dn_qs  = [], []
            for i in range(len(self.qkv_w)):
                qw, sc = quantize_pack_w4_sym(self.qkv_w[i],     self.quant_group)
                self.qkv_qw.append(qw); self.qkv_qs.append(sc)
                qw, sc = quantize_pack_w4_sym(self.o_w[i],       self.quant_group)
                self.o_qw.append(qw);   self.o_qs.append(sc)
                qw, sc = quantize_pack_w4_sym(self.gate_up_w[i], self.quant_group)
                self.gu_qw.append(qw);  self.gu_qs.append(sc)
                qw, sc = quantize_pack_w4_sym(self.down_w[i],    self.quant_group)
                self.dn_qw.append(qw);  self.dn_qs.append(sc)

        self.h = torch.empty(1, hs, dtype=dtype, device=device)
        self.res = torch.empty(1, hs, dtype=dtype, device=device)
        self.qkv_buf = torch.empty(1, qkv_total, dtype=dtype, device=device)
        self.o = torch.empty(1, hs, dtype=dtype, device=device)
        self.gu = torch.empty(1, inter2, dtype=dtype, device=device)
        self.act = torch.empty(1, self.intermediate_size, dtype=dtype, device=device)
        self.mlp_out = torch.empty(1, hs, dtype=dtype, device=device)
        self.final_h = torch.empty(1, hs, dtype=dtype, device=device)
        self.logits_buf = torch.empty(1, 1, vocab, dtype=dtype, device=device)

        try:
            from my_kernel.quant_swap import _ScatteredLMHead
            if isinstance(self.lm_head_module, _ScatteredLMHead):
                self.logits_buf.fill_(-65000.0)
        except Exception:
            pass

        self.use_fused_attn = _HAS_FUSED_ATTN
        self.use_custom_attn = _HAS_CUSTOM_ATTN and not self.use_fused_attn

        import sys as _sys
        _attn_path = ("fused_nrc+fused_decode (v3)" if self.use_fused_attn
                      else "decode_attn (custom)" if self.use_custom_attn
                      else "HF eager")
        print(f"[lean_decode] attention path: {_attn_path}",
              file=_sys.stderr)

        max_cache_len = cache.max_cache_len
        self.attn_kv_len = torch.tensor(
            [max_cache_len], dtype=torch.int64, device=device)

        if _HAS_FUSED_ATTN:
            self.q_normed_buf = torch.empty(
                self.num_q_heads, self.head_dim, dtype=dtype, device=device)
            self.attn_out_2d = torch.empty(
                self.num_q_heads, self.head_dim, dtype=dtype, device=device)
            self.fused_attn_workspace = _alloc_fused_ws(
                self.num_q_heads, max_cache_len, device)

        if _HAS_CUSTOM_ATTN or _HAS_FUSED_ATTN:
            self.attn_out_buf = torch.empty(
                1, self.num_q_heads, 1, self.head_dim, dtype=dtype, device=device)

        self._kv_int8_sim = os.environ.get("MY_KERNEL_KV_INT8_SIM", "0") == "1"
        if self._kv_int8_sim:
            print(f"[lean_decode] KV INT8 simulation ON (per-token quant+dequant)",
                  file=sys.stderr)

        if _HAS_CUSTOM_ATTN:
            self.attn_workspace = _alloc_attn_ws(
                self.num_q_heads, max_cache_len, device)

        max_pos = max_cache_len + 512
        dummy_x = torch.zeros(1, 1, self.hidden_size, dtype=dtype, device=device)
        all_pos = torch.arange(max_pos, dtype=torch.long, device=device)
        pos_3d = all_pos.view(1, 1, -1).expand(3, 1, -1)
        with torch.no_grad():
            cos_all, sin_all = lang.rotary_emb(dummy_x, pos_3d)
        self.cos_table = cos_all.squeeze(0)
        self.sin_table = sin_all.squeeze(0)

def _init_cache_views(st):
    if st.k_cache_views is not None:
        return
    cache = st.cache
    st.k_cache_views = []
    st.v_cache_views = []
    for layer_cache in cache.layers:
        st.k_cache_views.append(
            layer_cache.keys.view(st.num_kv_heads, cache.max_cache_len, st.head_dim))
        st.v_cache_views.append(
            layer_cache.values.view(st.num_kv_heads, cache.max_cache_len, st.head_dim))

def _mm(x, W, out):
    """Dispatch: custom GEMV if available, else cuBLAS.  out = x @ W.T"""
    if _fast_gemv is not None:
        _fast_gemv(x, W, out)
    else:
        torch.mm(x, W.t(), out=out)

def _get_w4_mod():
    """Lazy module-level handle to the W4 CUDA extension."""
    global _w4_mod
    if _w4_mod is None:
        from my_kernel.fused_w4_gemv import _module
        _w4_mod = _module
    return _w4_mod

_w4_mod = None

def _mm_w4(x, qw, sc, out, group_size):
    """W4A16 GEMV — dispatches through env-driven variant selector."""
    from my_kernel.fused_w4_gemv import gemv_w4_dispatch
    gemv_w4_dispatch(x.reshape(-1), qw, sc, out, group_size)
    return

def _mm_w4_legacy(x, qw, sc, out, group_size):
    """W4A16 GEMV writing into preallocated `out`."""
    _get_w4_mod().gemv_w4_forward_out(
        x.reshape(-1), qw, sc, out, group_size)

def lean_forward(st, input_ids, position_ids, cache_position, max_layers=None):
    """
    Single-token decode through all transformer layers.

    Uses FlashInfer fused kernels and pre-allocated buffers.
    All operations are CUDA-graph safe (static shapes, no allocations).

    max_layers: optional cap on how many decoder layers to run (the rest are
    skipped — used by the EOS-shortcut tiny graph in cuda_graph_runner).
    """
    cache = st.cache
    _init_cache_views(st)
    tok_emb = st.embed_tokens(input_ids)

    if st.use_fused_attn and hasattr(st, 'cos_table'):
        pos_idx = position_ids.reshape(-1)[:1].long()
        cos = st.cos_table[pos_idx]
        sin = st.sin_table[pos_idx]
    else:
        pos_3d = position_ids
        if pos_3d.ndim == 3 and pos_3d.shape[0] != 3:
            pos_3d = pos_3d.view(1, 1, -1).expand(3, 1, -1)
        elif pos_3d.ndim == 2:
            pos_3d = pos_3d[None, ...].expand(3, pos_3d.shape[0], -1)
        cos, sin = st.rotary_emb(tok_emb, pos_3d)

    hidden = tok_emb.view(1, st.hidden_size)
    st.res.copy_(hidden)

    q_dim = st.q_dim
    kv_dim = st.kv_dim
    hd = st.head_dim

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

    flashinfer.rmsnorm(st.res, st.ln1_w[0], st.ln1_eps[0], out=st.h)
    normed = st.h

    n_layers = st.num_layers if max_layers is None else min(int(max_layers), st.num_layers)
    for i in range(n_layers):
        if st.decoder_quant:
            _mm_w4(normed, st.qkv_qw[i], st.qkv_qs[i], st.qkv_buf, st.quant_group)
        else:
            _mm(normed, st.qkv_w[i], st.qkv_buf)

        if st.use_fused_attn:
            k_cache_3d = st.k_cache_views[i]
            v_cache_3d = st.v_cache_views[i]

            _fused_nrc(
                st.qkv_buf,
                st.q_norm_w[i], st.k_norm_w[i],
                cos, sin,
                st.q_normed_buf,
                k_cache_3d, v_cache_3d,
                cache_position,
                st.q_norm_eps[i], st.k_norm_eps[i],
                st.num_q_heads, st.num_kv_heads,
            )

            if getattr(st, "_kv_int8_sim", False):
                pos = cache_position[0]

                k_slot = k_cache_3d.index_select(1, pos)
                v_slot = v_cache_3d.index_select(1, pos)
                k_abs = k_slot.abs().amax(-1, keepdim=True).clamp(min=1e-5)
                v_abs = v_slot.abs().amax(-1, keepdim=True).clamp(min=1e-5)
                k_s = (k_abs / 127.0)
                v_s = (v_abs / 127.0)
                k_q = torch.round(k_slot / k_s).clamp(-127, 127)
                v_q = torch.round(v_slot / v_s).clamp(-127, 127)
                k_cache_3d.index_copy_(1, pos, (k_q * k_s).to(k_cache_3d.dtype))
                v_cache_3d.index_copy_(1, pos, (v_q * v_s).to(v_cache_3d.dtype))

            _fused_decode_attn(
                st.q_normed_buf, k_cache_3d, v_cache_3d, st.attn_out_2d,
                st.attn_kv_len, st.scaling, st.fused_attn_workspace,
            )
            attn_out = st.attn_out_2d.view(1, st.num_q_heads, 1, hd)
        else:
            q = st.qkv_buf[:, :q_dim].view(1, st.num_q_heads, hd)
            k = st.qkv_buf[:, q_dim:q_dim + kv_dim].view(1, st.num_kv_heads, hd)

            flashinfer.rmsnorm(q, st.q_norm_w[i], st.q_norm_eps[i], out=q)
            flashinfer.rmsnorm(k, st.k_norm_w[i], st.k_norm_eps[i], out=k)

            q4 = q.view(1, st.num_q_heads, 1, hd)
            k4 = k.view(1, st.num_kv_heads, 1, hd)
            v4 = st.qkv_buf[:, q_dim + kv_dim:].view(1, st.num_kv_heads, 1, hd)

            if _fused_rope is not None:
                q_rot, k_rot = _fused_rope(q4, k4, cos, sin)
            else:
                cos_4d = cos.unsqueeze(0)
                sin_4d = sin.unsqueeze(0)
                q_rot = (q4 * cos_4d) + (_rotate_half(q4) * sin_4d)
                k_rot = (k4 * cos_4d) + (_rotate_half(k4) * sin_4d)

            k_full, v_full = cache.update(k_rot, v4, i, cache_kwargs)

            if st.use_custom_attn:
                _decode_attn(
                    q_rot, k_full, v_full, st.attn_out_buf,
                    st.attn_kv_len, st.scaling, st.attn_workspace,
                )
                attn_out = st.attn_out_buf
            else:
                attn_out, _ = st.attn_fn(
                    st.attn_mods[i], q_rot, k_full, v_full,
                    None,
                    dropout=0.0,
                    scaling=st.scaling,
                )

        if st.decoder_quant:
            _mm_w4(attn_out.reshape(1, st.hidden_size),
                   st.o_qw[i], st.o_qs[i], st.o, st.quant_group)
        else:
            _mm(attn_out.reshape(1, st.hidden_size), st.o_w[i], st.o)

        flashinfer.fused_add_rmsnorm(st.o, st.res, st.ln2_w[i], st.ln2_eps[i])

        if st.decoder_quant:
            _mm_w4(st.o, st.gu_qw[i], st.gu_qs[i], st.gu, st.quant_group)

            flashinfer.silu_and_mul(st.gu, out=st.act)
            _mm_w4(st.act, st.dn_qw[i], st.dn_qs[i], st.mlp_out, st.quant_group)
        else:
            _mm(st.o, st.gate_up_w[i], st.gu)
            if _fused_silu_mul_gemv is not None:
                _fused_silu_mul_gemv(st.gu, st.down_w[i], st.mlp_out)
            else:
                flashinfer.silu_and_mul(st.gu, out=st.act)
                _mm(st.act, st.down_w[i], st.mlp_out)

        if i < n_layers - 1:
            flashinfer.fused_add_rmsnorm(
                st.mlp_out, st.res, st.ln1_w[i + 1], st.ln1_eps[i + 1],
            )
            normed = st.mlp_out
        else:
            st.res.add_(st.mlp_out)

    flashinfer.rmsnorm(st.res, st.final_norm_w, st.final_norm_eps, out=st.final_h)
    if st.lm_head_module is None:
        _mm(st.final_h, st.lm_head_w, st.logits_buf[:, 0, :])
    elif hasattr(st.lm_head_module, "forward_into"):

        st.lm_head_module.forward_into(st.final_h, st.logits_buf[:, 0, :])
    else:
        out = st.lm_head_module(st.final_h)
        st.logits_buf[:, 0, :].copy_(out.view(-1))

    return st.logits_buf

def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
