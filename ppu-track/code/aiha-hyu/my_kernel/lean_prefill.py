"""
[SHLEE] Lean text prefill: bypasses nn.Module.__call__ dispatch overhead in 28 decoder layers.
"""

import os
import torch
import torch.nn.functional as F
import flashinfer
from flash_attn import flash_attn_func
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers import StaticCache
import transformers.models.qwen3_vl.modeling_qwen3_vl as _qmod

from my_kernel.fused_silu_mul_ver2 import fused_silu_mul

_PREFILL_ATTN = os.environ.get("MY_KERNEL_PREFILL_ATTN", "flashinfer").lower()

# POP — Prefill-Only Pruning (arxiv 2602.03295, Qwen3-VL-validated).
# Selected decoder layers skip attention + MLP during prefill but still
# populate their KV cache via the QKV projection, so decode runs on the
# full model and consumes the cached entries unchanged.
def _get_pop_skip_layers():
    """Parse MY_KERNEL_POP_SKIP_LAYERS.  Supports:
        ""           → no skip
        "24,25,26"   → individual layers
        "10-26"      → range (inclusive both ends)
        "8-12,24-26" → multiple ranges + individual
    """
    s = os.environ.get("MY_KERNEL_POP_SKIP_LAYERS", "").strip()
    if not s:
        return set()
    result = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return result


# Cosine-similarity early exit (LayerSkip / hidden-state convergence).
# When the residual stream stabilizes across consecutive decoder layers
# (cos_sim of the last-token hidden > threshold), further refinement is
# redundant and we route directly to the LM head.  Compatible with POP:
# POP fixes a static skip set, while cos_exit is sample-adaptive.
def _get_cos_exit_cfg():
    start  = int(os.environ.get("MY_KERNEL_COS_EXIT_START", "999"))
    thresh = float(os.environ.get("MY_KERNEL_COS_EXIT_THRESH", "1.001"))
    return start, thresh

def _prefill_attn(q_bnsh, k_bnsh, v_bnsh, S):
    """Run prefill attention given tensors in [1, num_h, S, hd] layout.

    Returns [1, S, num_q * hd] reshaped output, ready for o_proj.
    """
    if _PREFILL_ATTN == "flashinfer":

        q3 = q_bnsh.transpose(1, 2).squeeze(0).contiguous()
        k3 = k_bnsh.transpose(1, 2).squeeze(0).contiguous()
        v3 = v_bnsh.transpose(1, 2).squeeze(0).contiguous()
        out = flashinfer.single_prefill_with_kv_cache(
            q3, k3, v3, causal=True, kv_layout="NHD",
        )

        return out.reshape(1, S, -1)
    else:

        attn = flash_attn_func(
            q_bnsh.transpose(1, 2),
            k_bnsh.transpose(1, 2),
            v_bnsh.transpose(1, 2),
            causal=True,
        )
        return attn.reshape(1, S, -1)

def flashinfer_rmsnorm_compat(x, weight, eps):
    """Support flashinfer versions whose rmsnorm kernel only accepts 2D input."""
    if x.ndim == 2:
        return flashinfer.rmsnorm(x, weight, eps)

    orig_shape = x.shape
    y = flashinfer.rmsnorm(x.reshape(-1, orig_shape[-1]), weight, eps)
    return y.reshape(orig_shape)

class LeanPrefillState:
    """Pre-extracted weights and config for lean text prefill."""

    def __init__(self, model):
        lang = model.model.language_model
        cfg = model.config.get_text_config()
        self.num_q = cfg.num_attention_heads
        self.num_kv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.q_dim = self.num_q * self.hd
        self.kv_dim = self.num_kv * self.hd

        self.n_layers = len(lang.layers)

        self.ln1_w = []
        self.ln1_eps = []
        self.qkv_w = []
        self.qnorm_w = []
        self.qnorm_eps = []
        self.knorm_w = []
        self.knorm_eps = []
        self.o_w = []
        self.ln2_w = []
        self.ln2_eps = []
        self.gu_w = []
        self.down_w = []

        for layer in lang.layers:
            self.ln1_w.append(layer.input_layernorm.weight.data)
            self.ln1_eps.append(layer.input_layernorm.variance_epsilon)
            fq = layer.self_attn.fused_qkv
            self.qkv_w.append(fq.qkv_proj.weight.data)
            self.qnorm_w.append(fq.q_norm.weight.data)
            self.qnorm_eps.append(fq.q_norm.variance_epsilon)
            self.knorm_w.append(fq.k_norm.weight.data)
            self.knorm_eps.append(fq.k_norm.variance_epsilon)
            self.o_w.append(layer.self_attn.o_proj.weight.data)
            self.ln2_w.append(layer.post_attention_layernorm.weight.data)
            self.ln2_eps.append(layer.post_attention_layernorm.variance_epsilon)
            self.gu_w.append(layer.mlp.gate_up_proj.weight.data)
            self.down_w.append(layer.mlp.down_proj.weight.data)

        self.intermediate = int(self.down_w[0].shape[1])
        for i, (gu_w, down_w) in enumerate(zip(self.gu_w, self.down_w)):
            if gu_w.shape[0] != 2 * self.intermediate or down_w.shape[1] != self.intermediate:
                raise RuntimeError(
                    f"[lean_prefill] MLP shape mismatch at layer {i}: "
                    f"gate_up={tuple(gu_w.shape)} down={tuple(down_w.shape)} "
                    f"intermediate={self.intermediate}"
                )

        self.fnorm_w = lang.norm.weight.data
        self.fnorm_eps = lang.norm.variance_epsilon

        self.lang = lang
        self.rotary_emb = lang.rotary_emb

def lean_prefill_forward(st, inputs_embeds, position_ids, past_key_values,
                         cache_position, visual_pos_masks=None,
                         deepstack_visual_embeds=None):
    """
    Drop-in replacement for Qwen3VLTextModel.forward during prefill.
    Uses F.linear directly to avoid nn.Module.__call__ dispatch overhead.
    """
    from my_kernel.profiler import profiler
    profiler.mark("llm.start")

    h = inputs_embeds
    S = h.shape[1]

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        spatial_pid = position_ids[1:]
    else:
        spatial_pid = position_ids

    cos, sin = st.rotary_emb(h, spatial_pid)
    profiler.mark("llm.rotary_emb")
    _profile_quarter = max(1, st.n_layers // 4)

    q_dim = st.q_dim
    kv_dim = st.kv_dim
    num_q = st.num_q
    num_kv = st.num_kv
    hd = st.hd
    intermediate = st.intermediate
    hidden = num_q * hd

    is_static_cache = isinstance(past_key_values, StaticCache)
    seq_len = (cache_position[-1].item() + 1) if is_static_cache else None

    pop_skip = _get_pop_skip_layers()
    cos_exit_start, cos_exit_thresh = _get_cos_exit_cfg()
    prev_h_last = None

    residual_flat = h.reshape(-1, hidden).contiguous()
    normed_flat = torch.empty_like(residual_flat)
    flashinfer.rmsnorm(residual_flat, st.ln1_w[0], st.ln1_eps[0], out=normed_flat)

    for i in range(st.n_layers):

        if i in pop_skip:
            normed_3d = normed_flat.reshape(1, S, hidden)
            qkv = F.linear(normed_3d, st.qkv_w[i])
            q = qkv[..., :q_dim].reshape(S * num_q, hd)
            k = qkv[..., q_dim:q_dim + kv_dim].reshape(S * num_kv, hd)
            v = qkv[..., q_dim + kv_dim:].reshape(1, S, num_kv, hd).transpose(1, 2)
            q = flashinfer.rmsnorm(q, st.qnorm_w[i], st.qnorm_eps[i]).reshape(1, S, num_q, hd).transpose(1, 2)
            k = flashinfer.rmsnorm(k, st.knorm_w[i], st.knorm_eps[i]).reshape(1, S, num_kv, hd).transpose(1, 2)
            q, k = _qmod.apply_rotary_pos_emb(q, k, cos, sin)
            past_key_values.update(k, v, i, cache_kwargs={"cache_position": cache_position})

            has_deepstack = (deepstack_visual_embeds is not None
                             and i < len(deepstack_visual_embeds))
            if has_deepstack:
                h_3d = st.lang._deepstack_process(
                    residual_flat.reshape(1, S, hidden),
                    visual_pos_masks, deepstack_visual_embeds[i])
                residual_flat = h_3d.reshape(-1, hidden).contiguous()
            if i < st.n_layers - 1:
                normed_flat = torch.empty_like(residual_flat)
                flashinfer.rmsnorm(
                    residual_flat, st.ln1_w[i + 1], st.ln1_eps[i + 1],
                    out=normed_flat)

            if (i + 1) % _profile_quarter == 0:
                profiler.mark(f"llm.layers_{i + 1}")
            continue

        normed_3d = normed_flat.reshape(1, S, hidden)
        qkv = F.linear(normed_3d, st.qkv_w[i])
        q = qkv[..., :q_dim].reshape(S * num_q, hd)
        k = qkv[..., q_dim:q_dim + kv_dim].reshape(S * num_kv, hd)
        v = qkv[..., q_dim + kv_dim:].reshape(1, S, num_kv, hd).transpose(1, 2)
        q = flashinfer.rmsnorm(q, st.qnorm_w[i], st.qnorm_eps[i]).reshape(1, S, num_q, hd).transpose(1, 2)
        k = flashinfer.rmsnorm(k, st.knorm_w[i], st.knorm_eps[i]).reshape(1, S, num_kv, hd).transpose(1, 2)
        q, k = _qmod.apply_rotary_pos_emb(q, k, cos, sin)
        k_c, v_c = past_key_values.update(k, v, i, cache_kwargs={"cache_position": cache_position})
        if is_static_cache:
            k_c = k_c[:, :, :seq_len, :]
            v_c = v_c[:, :, :seq_len, :]
        attn = _prefill_attn(q, k_c, v_c, S)

        attn_o_flat = F.linear(
            attn.contiguous(), st.o_w[i]
        ).reshape(-1, hidden).contiguous()
        flashinfer.fused_add_rmsnorm(
            attn_o_flat, residual_flat, st.ln2_w[i], st.ln2_eps[i],
        )

        gu = F.linear(attn_o_flat.reshape(1, S, hidden), st.gu_w[i])
        act = fused_silu_mul(gu, intermediate)
        down_out_flat = F.linear(act, st.down_w[i]).reshape(-1, hidden).contiguous()

        has_deepstack = (deepstack_visual_embeds is not None
                         and i < len(deepstack_visual_embeds))

        if i < st.n_layers - 1 and not has_deepstack:

            flashinfer.fused_add_rmsnorm(
                down_out_flat, residual_flat, st.ln1_w[i + 1], st.ln1_eps[i + 1],
            )
            normed_flat = down_out_flat
        else:

            residual_flat.add_(down_out_flat)
            if has_deepstack:
                h_3d = st.lang._deepstack_process(
                    residual_flat.reshape(1, S, hidden),
                    visual_pos_masks, deepstack_visual_embeds[i])
                residual_flat = h_3d.reshape(-1, hidden).contiguous()
            if i < st.n_layers - 1:
                normed_flat = torch.empty_like(residual_flat)
                flashinfer.rmsnorm(
                    residual_flat, st.ln1_w[i + 1], st.ln1_eps[i + 1],
                    out=normed_flat)

        if (i + 1) % _profile_quarter == 0:
            profiler.mark(f"llm.layers_{i + 1}")

        if i >= cos_exit_start and i < st.n_layers - 1:
            try:
                last_curr = residual_flat[-1].detach().to(torch.float32).clone()
                if prev_h_last is not None:
                    sim_t = torch.nn.functional.cosine_similarity(
                        last_curr.unsqueeze(0),
                        prev_h_last.unsqueeze(0),
                        dim=-1,
                    )
                    sim_val = float(sim_t.flatten()[0].item())
                    if sim_val > cos_exit_thresh:
                        break
                prev_h_last = last_curr
            except Exception as _e:
                pass

    profiler.mark("llm.layers_done")
    h = residual_flat.reshape(1, S, hidden)
    h = flashinfer_rmsnorm_compat(h, st.fnorm_w, st.fnorm_eps)
    profiler.mark("llm.final_norm")
    profiler.flush()
    return BaseModelOutputWithPast(last_hidden_state=h, past_key_values=past_key_values)

def install_lean_prefill(model):
    """Monkey-patch lang.forward to use lean prefill for sequences > 1 token."""
    lang = model.model.language_model
    st = LeanPrefillState(model)
    orig_fwd = lang.forward

    def _patched_fwd(input_ids=None, attention_mask=None, position_ids=None,
                     past_key_values=None, inputs_embeds=None, use_cache=None,
                     cache_position=None, visual_pos_masks=None,
                     deepstack_visual_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = lang.embed_tokens(input_ids)

        if inputs_embeds.shape[1] <= 1:
            return orig_fwd(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                cache_position=cache_position, visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds, **kwargs)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1],
                                          device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        return lean_prefill_forward(
            st, inputs_embeds, position_ids, past_key_values, cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds)

    lang.forward = _patched_fwd
    return st
