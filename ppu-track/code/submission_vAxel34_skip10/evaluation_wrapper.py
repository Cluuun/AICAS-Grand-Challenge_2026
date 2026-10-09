"""vAxel33 — vAxel30 + flag_gems fused_add_rms_norm decode (env-gated).

Inherits the full vAxel30 init (B1/B2/B4 fixes, W8A16 decode, vocab whitelist,
ViT prune, FA2). Adds a single env-toggled hook in the decode path:

  AICAS_FAST_DECODE=1  → use aicas_runtime.fast_decode.make_fast_decode_fns
                          (re-implements LM forward with flag_gems
                          fused_add_rms_norm threading; saves ~56 elementwise
                          add ops per token by fusing residual+rmsnorm).

Structural change is contained inside `make_fast_decode_fns`; everything else
(prefill, ViT, lm_head dispatcher, decode-graph capture) is shared with
vAxel30. To roll back to vAxel30 behaviour set AICAS_FAST_DECODE=0.
"""
from __future__ import annotations

import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor, StaticCache

from prune_optim import (
    configure_processor_method_d,
    PRUNE_WARMUP_GRIDS, PRUNE_PREFILL_WARMUP_SEQS,
)
from quant_optim import (
    W8A16Weight, w8a16_linear,
    build_llm_decode_w8a16_bundle, install_w8a16_decode_patches,
    build_vit_fc1_w8a8_bundle,
    TextVQACalibrationLoader, collect_vit_fc1_activation_stats,
)

from aicas_runtime import (
    AICAS_QUANT_ENABLE, AICAS_QUANT_LM_HEAD, AICAS_QUANT_DECODE_LLM,
    AICAS_QUANT_VIT_FC1, AICAS_QUANT_VIT_FC1_LO, AICAS_QUANT_VIT_FC1_HI,
    AICAS_QUANT_VIT_CALIBRATE, AICAS_QUANT_N_CALIB, AICAS_QUANT_VIT_ALPHA,
    AICAS_QUANT_TEXTVQA_DIR,
    AICAS_VOCAB_WHITELIST, AICAS_VOCAB_K, AICAS_VOCAB_TEXTVQA_DIR,
    AICAS_VOCAB_CORPUS,
    AICAS_MAX_PIXELS, PRUNE_CONFIG,
    MAX_CACHE_LEN, MAX_CACHE_LEN_ACC, N_DECODE_STEPS,
    NUM_LLM_LAYERS, NUM_KV_HEADS, HEAD_DIM,
    CHAT_PREFIX,
    make_fast_apply_chat_template,
    make_fast_pos_embed_interpolate, make_fast_rot_pos_emb,
    make_fast_get_rope_index,
    install_llm_fusions,
    make_manual_vit_forward_fused,
    make_compiled_full_prefill,
    make_decode_fns, capture_decode_graphs,
    make_fast_decode_fns,
    build_vocab_whitelist,
)
import os
AICAS_FAST_DECODE = os.environ.get("AICAS_FAST_DECODE", "1") == "1"
AICAS_N_DECODE_STEPS = int(os.environ.get("AICAS_N_DECODE_STEPS", str(N_DECODE_STEPS)))


def _parse_skip_layers():
    # vAxel34 default: skip last 8 LLM layers in decode (prefill keeps all 28)
    raw = os.environ.get("AICAS_DECODE_SKIP_LAYERS", "18,19,20,21,22,23,24,25,26,27").strip()
    if not raw:
        return None
    skip = set(int(x) for x in raw.split(",") if x.strip())
    active = [i for i in range(NUM_LLM_LAYERS) if i not in skip]
    return active


def _parse_mlp_skip():
    raw = os.environ.get("AICAS_DECODE_MLP_SKIP", "").strip()
    if not raw:
        return None
    return set(int(x) for x in raw.split(",") if x.strip())


AICAS_DECODE_ACTIVE_LAYERS = _parse_skip_layers()
AICAS_DECODE_MLP_SKIP = _parse_mlp_skip()


class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self._device = device
        torch.cuda.empty_cache = lambda: None

        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device)
        self._model.eval()
        self._model.config.use_cache = True

        # ── Method D: cap max_pixels ──
        configure_processor_method_d(
            self._processor.image_processor, AICAS_MAX_PIXELS)

        # ── FastProcessor ──
        self._processor.apply_chat_template = make_fast_apply_chat_template(
            self._processor)
        print("[vAxel30] FastProcessor installed")

        # ── ViT overlap stream ──
        self._vit_stream = torch.cuda.Stream(device=device)
        print("[vAxel30] ViT stream created for overlap")

        # ── FA2 KV caches ──
        self._fa_k_cache = torch.zeros(
            NUM_LLM_LAYERS, 1, MAX_CACHE_LEN, NUM_KV_HEADS, HEAD_DIM,
            dtype=torch.float16, device=device)
        self._fa_v_cache = torch.zeros(
            NUM_LLM_LAYERS, 1, MAX_CACHE_LEN, NUM_KV_HEADS, HEAD_DIM,
            dtype=torch.float16, device=device)
        self._fa_cache_seqlens = torch.zeros(1, dtype=torch.int32, device=device)
        print("[vAxel30] FA2 caches allocated")

        # ── LLM fusion (MLP + QKV+FA2) ──
        n_mlp, n_qkv = install_llm_fusions(
            self._model, self._fa_k_cache, self._fa_v_cache,
            self._fa_cache_seqlens)
        print(f"[vAxel30] LLM Fused: {n_mlp} MLP, {n_qkv} QKV+FA2")

        # ── Stage-1 W8A16 quantization on LLM decode + lm_head ──
        self._w8a16_lm_head = None
        if AICAS_QUANT_ENABLE and AICAS_QUANT_DECODE_LLM:
            print(f"[vAxel30_quant] Building W8A16 bundle (decode QKV/MLP + lm_head={AICAS_QUANT_LM_HEAD})...")
            bundle = build_llm_decode_w8a16_bundle(
                self._model, quantize_lm_head=AICAS_QUANT_LM_HEAD)
            n_attn_q, n_mlp_q = install_w8a16_decode_patches(
                self._model, bundle,
                self._fa_k_cache, self._fa_v_cache, self._fa_cache_seqlens)
            print(f"[vAxel30_quant] W8A16 patched: {n_attn_q} attn, {n_mlp_q} mlp")
            self._w8a16_lm_head = bundle.lm_head
        elif AICAS_QUANT_ENABLE and AICAS_QUANT_LM_HEAD:
            self._w8a16_lm_head = W8A16Weight.from_fp16(self._model.lm_head.weight)
            print("[vAxel30_quant] W8A16 lm_head only (decoder kept FP16)")
        else:
            print("[vAxel30_quant] Quant disabled (env AICAS_QUANT_ENABLE=0)")

        # ── §1.10/§1.11 vocab whitelist ──
        self._inv_perm = None
        self._inv_perm_cpu = None
        self._lm_head_weight_sliced = None
        if AICAS_VOCAB_WHITELIST:
            import time as _t
            t0 = _t.time()
            print(f"[vAxel30] Building vocab whitelist (target K={AICAS_VOCAB_K}) "
                  f"corpus={AICAS_VOCAB_CORPUS} parquet={AICAS_VOCAB_TEXTVQA_DIR}")
            orig_lm_head_w = self._model.lm_head.weight
            V_orig = orig_lm_head_w.shape[0]
            whitelist, used_src = build_vocab_whitelist(
                self._processor.tokenizer,
                target_K=min(AICAS_VOCAB_K, V_orig),
                vocab_size=V_orig,
                corpus_path=AICAS_VOCAB_CORPUS,
                parquet_dir=AICAS_VOCAB_TEXTVQA_DIR,
            )
            V_small = len(whitelist)
            wl_t = torch.tensor(whitelist, dtype=torch.long, device=device)
            self._inv_perm = wl_t
            self._inv_perm_cpu = whitelist
            sliced_fp16 = orig_lm_head_w.index_select(0, wl_t).contiguous()
            if self._w8a16_lm_head is not None:
                self._w8a16_lm_head = W8A16Weight.from_fp16(sliced_fp16)
            else:
                self._lm_head_weight_sliced = sliced_fp16
            print(f"[vAxel30] vocab whitelist: V_orig={V_orig} -> V_small={V_small} "
                  f"({V_small/V_orig*100:.1f}%); built in {_t.time()-t0:.1f}s "
                  f"src={used_src}")

        # ── Patch ViT positional encodings + rope ──
        vit = self._model.model.visual
        fast_pos_embed = make_fast_pos_embed_interpolate(vit)
        fast_rot = make_fast_rot_pos_emb(vit)

        test_grid = torch.tensor([[1, 30, 40]], dtype=torch.int32, device=device)
        with torch.no_grad():
            orig_pos = vit.fast_pos_embed_interpolate(test_grid)
            new_pos = fast_pos_embed(test_grid)
            pos_ok = torch.allclose(orig_pos, new_pos, atol=5e-2)
            orig_rot = vit.rot_pos_emb(test_grid)
            new_rot = fast_rot(test_grid)
            rot_ok = torch.allclose(orig_rot, new_rot, atol=1e-5)
        print(f"[vAxel30] pos_embed: {'OK' if pos_ok else 'MISMATCH'}"
              f"  rot_pos_emb: {'OK' if rot_ok else 'MISMATCH'}")
        vit.fast_pos_embed_interpolate = fast_pos_embed
        vit.rot_pos_emb = fast_rot

        outer_model = self._model.model
        outer_model.get_rope_index = make_fast_get_rope_index(outer_model)
        print("[vAxel30] Patched fast_pos_embed_interpolate, rot_pos_emb, get_rope_index")

        # ── Stage-2 narrow W8A8 ViT fc1 (optional) ──
        vit_w8a8_bundle = None
        if AICAS_QUANT_ENABLE and AICAS_QUANT_VIT_FC1:
            act_stats = None
            if AICAS_QUANT_VIT_CALIBRATE:
                print(f"[vAxel30_quant] Running ViT fc1 calibration "
                      f"(N={AICAS_QUANT_N_CALIB}, α={AICAS_QUANT_VIT_ALPHA})...")
                cal_loader = TextVQACalibrationLoader(
                    AICAS_QUANT_TEXTVQA_DIR,
                    n_samples=AICAS_QUANT_N_CALIB,
                    image_processor=self._processor.image_processor,
                    device=device,
                )
                import time as _t
                _t0 = _t.time()
                act_stats = collect_vit_fc1_activation_stats(self._model, cal_loader)
                print(f"[vAxel30_quant] Calibration done in {_t.time()-_t0:.1f}s; "
                      f"{sum(1 for s in act_stats if s is not None)}/24 blocks covered")
            vit_w8a8_bundle = build_vit_fc1_w8a8_bundle(
                self._model,
                layer_range=(AICAS_QUANT_VIT_FC1_LO, AICAS_QUANT_VIT_FC1_HI),
                activation_stats=act_stats,
                alpha=AICAS_QUANT_VIT_ALPHA,
            )
            n_w8a8 = sum(1 for f in vit_w8a8_bundle.fc1 if f is not None)
            n_calib = sum(1 for f in vit_w8a8_bundle.fc1
                          if f is not None and f.smooth_inv_s is not None)
            print(f"[vAxel30_quant] ViT fc1 W8A8 bundle built: {n_w8a8} blocks "
                  f"in [{vit_w8a8_bundle.layer_range[0]}, {vit_w8a8_bundle.layer_range[1]}), "
                  f"calibrated={n_calib}")

        # ── Compiled manual ViT forward + prune warmup ──
        self._original_get_image_features = outer_model.get_image_features
        manual_vit_fn = make_manual_vit_forward_fused(
            vit, prune_config=PRUNE_CONFIG, vit_w8a8_fc1=vit_w8a8_bundle)
        compiled_manual_vit = torch.compile(
            manual_vit_fn, mode="default", dynamic=True)
        print(f"[vAxel30_prune] prune_config={PRUNE_CONFIG}")
        print("[vAxel30_prune] Pre-warming manual ViT forward...")
        for grid in PRUNE_WARMUP_GRIDS:
            t_, h_, w_ = grid
            dummy_pv = torch.zeros(t_ * h_ * w_, 1536, dtype=torch.float16, device=device)
            dummy_grid = torch.tensor([list(grid)], dtype=torch.long, device=device)
            with torch.no_grad():
                try:
                    merged, ds_feats, keep_blk = compiled_manual_vit(dummy_pv, dummy_grid)
                    kb_str = "None" if keep_blk is None else f"shape={tuple(keep_blk.shape)}"
                    print(f"  Grid {grid}: OK (merged={merged.shape}, ds={[d.shape for d in ds_feats]}, keep={kb_str})")
                except Exception as e:
                    print(f"  Grid {grid}: FAILED - {type(e).__name__}: {e}")
        self._compiled_manual_vit = compiled_manual_vit
        print("[vAxel30_prune] Compiled manual ViT forward wired")

        # ── StaticCache (acc fallback) ──
        self._static_cache = StaticCache(
            self._model.config.text_config,
            max_cache_len=MAX_CACHE_LEN,
            batch_size=1, device=device, dtype=torch.float16)
        self._static_cache_acc = StaticCache(
            self._model.config.text_config,
            max_cache_len=MAX_CACHE_LEN_ACC,
            batch_size=1, device=device, dtype=torch.float16)
        self._cache_pos_buf = torch.arange(MAX_CACHE_LEN, device=device, dtype=torch.long)
        self._cache_pos_buf_acc = torch.arange(MAX_CACHE_LEN_ACC, device=device, dtype=torch.long)

        lm = self._model.model.language_model
        self._embed_tokens = lm.embed_tokens
        self._llm_layers = lm.layers
        self._llm_norm = lm.norm
        self._rotary_emb = lm.rotary_emb
        self._lm_head_weight = self._model.lm_head.weight

        self._image_token_id = self._model.config.image_token_id
        self._video_token_id = getattr(self._model.config, 'video_token_id', None)

        # ── Per-layer attn params (fused QKV) ──
        self._attn_params = []
        for layer in lm.layers:
            attn = layer.self_attn
            self._attn_params.append({
                'qkv_weight': None, 'qkv_bias': None, 'o_weight': None,
                'q_norm': attn.q_norm, 'k_norm': attn.k_norm,
                'head_dim': attn.head_dim, 'scaling': attn.scaling,
                'q_dim': None, 'k_dim': None, 'v_dim': None,
            })
        for name, module in self._model.named_modules():
            if 'language_model.layers.' in name and name.endswith('.self_attn'):
                li = module.layer_idx
                q_w = module.q_proj.weight
                k_w = module.k_proj.weight
                v_w = module.v_proj.weight
                self._attn_params[li]['qkv_weight'] = torch.cat([q_w, k_w, v_w], dim=0)
                self._attn_params[li]['q_dim'] = q_w.shape[0]
                self._attn_params[li]['k_dim'] = k_w.shape[0]
                self._attn_params[li]['v_dim'] = v_w.shape[0]
                if module.q_proj.bias is not None:
                    self._attn_params[li]['qkv_bias'] = torch.cat([
                        module.q_proj.bias, module.k_proj.bias, module.v_proj.bias],
                        dim=0)
                self._attn_params[li]['o_weight'] = module.o_proj.weight

        # ── Build prefill layer params + lm_head dispatcher ──
        self._prefill_layer_params = []
        for layer_idx in range(NUM_LLM_LAYERS):
            ap = self._attn_params[layer_idx]
            layer = self._llm_layers[layer_idx]
            self._prefill_layer_params.append({
                'input_layernorm': layer.input_layernorm,
                'post_attention_layernorm': layer.post_attention_layernorm,
                'mlp': layer.mlp,
                'qkv_weight': ap['qkv_weight'], 'qkv_bias': ap['qkv_bias'],
                'o_weight': ap['o_weight'],
                'q_norm': ap['q_norm'], 'k_norm': ap['k_norm'],
                'head_dim': ap['head_dim'], 'scaling': ap['scaling'],
                'q_dim': ap['q_dim'], 'k_dim': ap['k_dim'], 'v_dim': ap['v_dim'],
            })

        lm_head_w8_ref = self._w8a16_lm_head
        lm_head_weight_ref = self._lm_head_weight
        lm_head_weight_sliced_ref = self._lm_head_weight_sliced

        if lm_head_w8_ref is not None:
            def _lm_head_apply(h):
                return w8a16_linear(h, lm_head_w8_ref)
        elif lm_head_weight_sliced_ref is not None:
            def _lm_head_apply(h):
                return F.linear(h, lm_head_weight_sliced_ref)
        else:
            def _lm_head_apply(h):
                return F.linear(h, lm_head_weight_ref)
        self._lm_head_apply = _lm_head_apply

        # ── Compiled FULL prefill ──
        self._compiled_full_prefill = make_compiled_full_prefill(
            num_layers=NUM_LLM_LAYERS,
            layer_params_list=self._prefill_layer_params,
            fa_k_cache=self._fa_k_cache, fa_v_cache=self._fa_v_cache,
            llm_norm=self._llm_norm, lm_head_apply=self._lm_head_apply,
        )
        print("[vAxel30] Compiled FULL prefill function registered")

        print("[vAxel30] Warming up compiled full prefill...")
        dummy_seq_len = 128
        dummy_embeds = torch.zeros(1, dummy_seq_len, 2048, dtype=torch.float16, device=device)
        dummy_cos = torch.ones(1, dummy_seq_len, 128, dtype=torch.float16, device=device)
        dummy_sin = torch.zeros(1, dummy_seq_len, 128, dtype=torch.float16, device=device)
        z = torch.zeros(1, dummy_seq_len, 2048, dtype=torch.float16, device=device)
        with torch.no_grad():
            self._fa_cache_seqlens.fill_(0)
            _ = self._compiled_full_prefill(dummy_embeds, dummy_cos, dummy_sin,
                                            z, z, z, self._fa_cache_seqlens)
        torch.cuda.synchronize()
        print("[vAxel30] Full prefill warmup: text-only OK")

        n_vis = 64
        ds0 = torch.zeros(1, dummy_seq_len, 2048, dtype=torch.float16, device=device)
        ds0[0, 10:10+n_vis, :] = torch.randn(n_vis, 2048, dtype=torch.float16, device=device)
        ds1 = torch.zeros(1, dummy_seq_len, 2048, dtype=torch.float16, device=device)
        ds1[0, 10:10+n_vis, :] = torch.randn(n_vis, 2048, dtype=torch.float16, device=device)
        ds2 = torch.zeros(1, dummy_seq_len, 2048, dtype=torch.float16, device=device)
        ds2[0, 10:10+n_vis, :] = torch.randn(n_vis, 2048, dtype=torch.float16, device=device)
        with torch.no_grad():
            self._fa_cache_seqlens.fill_(0)
            _ = self._compiled_full_prefill(dummy_embeds, dummy_cos, dummy_sin,
                                            ds0, ds1, ds2, self._fa_cache_seqlens)
        torch.cuda.synchronize()
        print("[vAxel30] Full prefill warmup: with image + DeepStack OK")

        for sl in PRUNE_PREFILL_WARMUP_SEQS:
            sl_embeds = torch.zeros(1, sl, 2048, dtype=torch.float16, device=device)
            sl_cos = torch.ones(1, sl, 128, dtype=torch.float16, device=device)
            sl_sin = torch.zeros(1, sl, 128, dtype=torch.float16, device=device)
            n_vis_w = max(min(sl - 60, 300), 16)
            sl_ds0 = torch.zeros(1, sl, 2048, dtype=torch.float16, device=device)
            sl_ds0[0, 4:4+n_vis_w, :] = torch.randn(n_vis_w, 2048, dtype=torch.float16, device=device)
            sl_ds1 = torch.zeros_like(sl_ds0); sl_ds1.copy_(sl_ds0)
            sl_ds2 = torch.zeros_like(sl_ds0); sl_ds2.copy_(sl_ds0)
            with torch.no_grad():
                self._fa_cache_seqlens.fill_(0)
                _ = self._compiled_full_prefill(sl_embeds, sl_cos, sl_sin,
                                                sl_ds0, sl_ds1, sl_ds2,
                                                self._fa_cache_seqlens)
            torch.cuda.synchronize()
        print(f"[vAxel30_prune] Full prefill warmup: seqs={PRUNE_PREFILL_WARMUP_SEQS} OK")

        # ── Compiled decode + CUDA graph capture ──
        n = AICAS_N_DECODE_STEPS
        self._decode_input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._decode_position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=device)
        self._decode_start_token = torch.zeros(1, 1, dtype=torch.long, device=device)
        self._pos_n = torch.zeros(n, 3, 1, 1, dtype=torch.long, device=device)
        self._pos_delta = torch.arange(n, dtype=torch.long, device=device)

        self._eos_token_id = self._processor.tokenizer.eos_token_id

        if AICAS_FAST_DECODE:
            if AICAS_DECODE_ACTIVE_LAYERS is not None:
                skipped = [i for i in range(NUM_LLM_LAYERS) if i not in AICAS_DECODE_ACTIVE_LAYERS]
                print(f"[vAxel34] fast_decode + layer-skip: active={len(AICAS_DECODE_ACTIVE_LAYERS)}/{NUM_LLM_LAYERS} skip={skipped}")
            elif AICAS_DECODE_MLP_SKIP is not None:
                print(f"[vAxel34b] fast_decode + MLP-only skip: skip MLP at {sorted(AICAS_DECODE_MLP_SKIP)}")
            else:
                print(f"[vAxel33] Using fast_decode (flag_gems fused_add_rms_norm path)")
            decode_step, decode_n_steps = make_fast_decode_fns(
                language_model=lm,
                lm_head_apply=self._lm_head_apply,
                inv_perm=self._inv_perm,
                fa_seqlens=self._fa_cache_seqlens,
                n_steps=n,
                active_layers=AICAS_DECODE_ACTIVE_LAYERS,
                mlp_skip_layers=AICAS_DECODE_MLP_SKIP,
            )
        else:
            print(f"[vAxel33] Using vAxel30 baseline decode (AICAS_FAST_DECODE=0)")
            decode_step, decode_n_steps = make_decode_fns(
                language_model=lm,
                lm_head_apply=self._lm_head_apply,
                inv_perm=self._inv_perm,
                fa_seqlens=self._fa_cache_seqlens,
                n_steps=n,
            )

        print(f"[vAxel33] Compiling decode functions (N={n}, default mode, FA2 path)...")
        graph_1, static_token_1, graph_n, static_tokens_n = capture_decode_graphs(
            decode_step=decode_step, decode_n_steps=decode_n_steps,
            decode_input_ids=self._decode_input_ids,
            decode_position_ids=self._decode_position_ids,
            decode_start_token=self._decode_start_token,
            pos_n=self._pos_n,
            fa_seqlens=self._fa_cache_seqlens,
        )
        self._graph_1 = graph_1
        self._static_token_1 = static_token_1
        self._graph_n = graph_n
        self._static_tokens_n = static_tokens_n
        print("[vAxel30] CUDA graph capture done!")

        # ── Pre-allocated DeepStack scatter buffers ──
        max_seq = MAX_CACHE_LEN
        self._ds_scattered_0 = torch.zeros(1, max_seq, 2048, dtype=torch.float16, device=device)
        self._ds_scattered_1 = torch.zeros(1, max_seq, 2048, dtype=torch.float16, device=device)
        self._ds_scattered_2 = torch.zeros(1, max_seq, 2048, dtype=torch.float16, device=device)
        print("[vAxel30] Pre-allocated DeepStack scatter buffers")

        self._model.generate = self._custom_generate
        print(f"[vAxel33] Ready | N={n} | fast_decode={AICAS_FAST_DECODE} | FA2 decode | FA2 ViT | fused ViT QKV/MLP | compiled full prefill | FastProcessor")

    # ───── runtime path ─────

    def _manual_prefill_fa2(self, inputs, fa_k_cache, fa_v_cache, fa_cache_seqlens):
        input_ids = inputs['input_ids']
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')

        device = self._device

        deepstack_visual_embeds = None
        keep_blocks = None
        has_image = (pixel_values is not None and image_grid_thw is not None)

        if has_image:
            with torch.cuda.stream(self._vit_stream):
                image_embeds, deepstack_visual_embeds, keep_blocks = self._compiled_manual_vit(
                    pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(device, torch.float16)

            position_ids, rope_deltas = self._model.model.get_rope_index(
                input_ids, image_grid_thw=image_grid_thw, attention_mask=None)
            self._model.model.rope_deltas = rope_deltas

            torch.cuda.current_stream().wait_stream(self._vit_stream)
        else:
            image_embeds = torch.zeros(0, 2048, dtype=torch.float16, device=device)
            position_ids, rope_deltas = self._model.model.get_rope_index(
                input_ids, image_grid_thw=image_grid_thw, attention_mask=None)
            self._model.model.rope_deltas = rope_deltas

        if has_image and keep_blocks is not None:
            n_orig = int(image_grid_thw[0, 0]) * int(image_grid_thw[0, 1]) * int(image_grid_thw[0, 2])
            n_merged_orig = n_orig // 4
            img_start = len(CHAT_PREFIX)
            seq_len_orig = input_ids.shape[1]
            suffix_start = img_start + n_merged_orig
            prefix_idx = torch.arange(img_start, device=device, dtype=torch.long)
            image_idx = img_start + keep_blocks.to(torch.long)
            suffix_idx = torch.arange(suffix_start, seq_len_orig, device=device, dtype=torch.long)
            kept_indices = torch.cat([prefix_idx, image_idx, suffix_idx])
            input_ids = input_ids.index_select(1, kept_indices)
            position_ids = position_ids.index_select(2, kept_indices)

        inputs_embeds = self._embed_tokens(input_ids)

        if has_image and image_embeds.shape[0] > 0:
            img_start = len(CHAT_PREFIX)
            img_end = img_start + image_embeds.shape[0]
            inputs_embeds[0, img_start:img_end] = image_embeds

        position_embeddings = self._rotary_emb(inputs_embeds, position_ids)
        cos_emb, sin_emb = position_embeddings

        seq_len = input_ids.shape[1]
        ds0 = self._ds_scattered_0[:, :seq_len, :]
        ds1 = self._ds_scattered_1[:, :seq_len, :]
        ds2 = self._ds_scattered_2[:, :seq_len, :]
        ds0.zero_(); ds1.zero_(); ds2.zero_()

        if has_image and deepstack_visual_embeds is not None:
            ds_fp16 = [ve.to(device, torch.float16) for ve in deepstack_visual_embeds]
            img_start = len(CHAT_PREFIX)
            img_end = img_start + ds_fp16[0].shape[0]
            ds0[0, img_start:img_end] = ds_fp16[0]
            ds1[0, img_start:img_end] = ds_fp16[1]
            ds2[0, img_start:img_end] = ds_fp16[2]

        fa_cache_seqlens.fill_(0)

        with torch.no_grad():
            logits = self._compiled_full_prefill(
                inputs_embeds, cos_emb, sin_emb, ds0, ds1, ds2, fa_cache_seqlens)

        prefill_len = input_ids.shape[1]
        fa_cache_seqlens.fill_(prefill_len)

        return logits, prefill_len

    def _custom_generate(self, *args, **kwargs):
        input_ids = kwargs.get('input_ids')
        max_new_tokens = kwargs.get('max_new_tokens', 128)
        inputs = {k: kwargs[k] for k in
                  ['input_ids', 'attention_mask', 'pixel_values',
                   'image_grid_thw', 'mm_token_type_ids']
                  if k in kwargs and kwargs[k] is not None}

        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

        input_len = input_ids.shape[1]
        use_perf_path = (input_len + max_new_tokens <= MAX_CACHE_LEN)

        if not use_perf_path:
            return self._generate_eager(inputs, input_ids, input_len, max_new_tokens)

        with torch.no_grad():
            logits, prefill_len = self._manual_prefill_fa2(
                inputs, self._fa_k_cache, self._fa_v_cache, self._fa_cache_seqlens)
        input_len = prefill_len
        first_small = logits.argmax(dim=-1).item()
        first_tid = (self._inv_perm_cpu[first_small]
                     if self._inv_perm_cpu is not None else first_small)

        generated_ids = [first_tid]

        if max_new_tokens <= 1 or first_tid == self._eos_token_id:
            gen = torch.tensor(generated_ids, dtype=torch.long, device=self._device).unsqueeze(0)
            return torch.cat([input_ids, gen], dim=1)

        self._decode_start_token[0, 0] = first_tid
        n = AICAS_N_DECODE_STEPS
        step = 1

        with torch.no_grad():
            while step < max_new_tokens:
                remaining = max_new_tokens - step

                if remaining >= n:
                    base = input_len + step
                    self._fa_cache_seqlens.fill_(base)
                    positions = self._pos_delta + base
                    self._pos_n[:, :, 0, 0] = positions.unsqueeze(1)

                    self._graph_n.replay()
                    tok_ids = self._static_tokens_n[0].tolist()

                    done = False
                    for tid in tok_ids:
                        generated_ids.append(tid)
                        if tid == self._eos_token_id:
                            done = True
                            break
                    if done:
                        break

                    self._decode_start_token[0, 0] = tok_ids[-1]
                    step += n

                else:
                    step_pos = input_len + step
                    self._fa_cache_seqlens.fill_(step_pos)
                    self._decode_input_ids.copy_(self._decode_start_token)
                    self._decode_position_ids[:, 0, 0] = step_pos

                    self._graph_1.replay()
                    tid = self._static_token_1.item()
                    generated_ids.append(tid)

                    if tid == self._eos_token_id:
                        break

                    self._decode_start_token[0, 0] = tid
                    step += 1

        gen = torch.tensor(generated_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        return torch.cat([input_ids, gen], dim=1)

    def _generate_eager(self, inputs, input_ids, input_len, max_new_tokens):
        self._static_cache_acc.reset()
        cache_position = self._cache_pos_buf_acc[:input_len]
        with torch.no_grad():
            outputs = self._model(
                **inputs, past_key_values=self._static_cache_acc,
                use_cache=True, cache_position=cache_position,
                return_dict=True, logits_to_keep=1)
        first_tid = outputs.logits[:, -1:, :].argmax(dim=-1).item()
        generated_ids = [first_tid]

        if max_new_tokens <= 1 or first_tid == self._eos_token_id:
            gen = torch.tensor(generated_ids, dtype=torch.long, device=self._device).unsqueeze(0)
            return torch.cat([input_ids, gen], dim=1)

        lm = self._model.model.language_model
        cur_input = torch.tensor([[first_tid]], dtype=torch.long, device=self._device)
        step = 1

        with torch.no_grad():
            while step < max_new_tokens:
                step_pos = input_len + step
                pos_ids = torch.full((3, 1, 1), step_pos, dtype=torch.long, device=self._device)
                cache_pos = torch.tensor([step_pos], dtype=torch.long, device=self._device)
                out = lm(input_ids=cur_input, position_ids=pos_ids,
                         cache_position=cache_pos,
                         past_key_values=self._static_cache_acc,
                         use_cache=True, return_dict=True)
                h = out.last_hidden_state[:, -1, :]
                small = self._lm_head_apply(h).argmax(dim=-1).item()
                tid = (self._inv_perm_cpu[small]
                       if self._inv_perm_cpu is not None else small)
                generated_ids.append(tid)
                if tid == self._eos_token_id:
                    break
                cur_input[0, 0] = tid
                step += 1

        gen = torch.tensor(generated_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        return torch.cat([input_ids, gen], dim=1)

    @property
    def processor(self): return self._processor
    @property
    def model(self): return self._model
    @property
    def device(self): return self._device

    def generate(self, image, question, max_new_tokens=128):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self._device)
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
        output_ids = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=0.0, use_cache=True)
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0][input_len:]
        text = self._processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        text = _post_process_answer(text)
        return {"text": text, "token_count": len(generated_ids)}


_SENT_RE = re.compile(r'[.!?]\s')


def _post_process_answer(text: str) -> str:
    if len(text) <= 200:
        return text
    matches = list(_SENT_RE.finditer(text))
    if len(matches) >= 2 and matches[1].end() <= 280:
        return text[:matches[1].end()].rstrip()
    if matches and matches[0].end() <= 280:
        return text[:matches[0].end()].rstrip()
    return text[:250].rstrip()
