"""
[SHLEE] In-process weight quantization swaps + lm_head channel pruning.

Walks the loaded FP16 Qwen3-VL-2B model and replaces selected linear
layers with quantized equivalents.  Controlled by env vars:

    MY_KERNEL_QUANT_MODE       - "off" | "lm_head" | "decoder" | "all"
    MY_KERNEL_QUANT_ALGO       - "rtn"
    MY_KERNEL_QUANT_GROUP      - group_size, default 128
    MY_KERNEL_LMHEAD_PRUNE_K   - keep top-K rows of lm_head (0 = off)
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _mode() -> str:
    return _env("MY_KERNEL_QUANT_MODE", "all").lower()

def is_quant_enabled() -> bool:
    return _mode() != "off"

def is_lm_head_quant_enabled() -> bool:
    return _mode() in ("lm_head", "all")

def is_decoder_quant_enabled() -> bool:
    return _mode() in ("decoder", "all")

def _algo() -> str:
    return _env("MY_KERNEL_QUANT_ALGO", "rtn").lower()

def _group() -> int:
    try:
        return int(_env("MY_KERNEL_QUANT_GROUP", "128"))
    except ValueError:
        return 128

def _lm_head_prune_k() -> int:
    try:
        return int(_env("MY_KERNEL_LMHEAD_PRUNE_K", "0"))
    except ValueError:
        return 0

def _lm_head_prune_low_id_cap() -> int:
    """[SHLEE] Always keep token IDs in [0, cap).  Qwen3 BPE assigns low IDs
    to high-frequency merges, so this protects common content vocab.
    """
    try:
        return int(_env("MY_KERNEL_LMHEAD_PRUNE_LOW_ID", "40000"))
    except ValueError:
        return 40000

def is_lm_head_pruning_enabled() -> bool:
    return _lm_head_prune_k() > 0

def _quantize_decoder_weights(model: nn.Module, device: str = "cuda:0"):
    from my_kernel.fused_w4_gemv import (
        quantize_pack_w4_sym, _load as _load_w4,
    )
    _load_w4()

    group = _group()
    lang = model.model.language_model
    cache = {
        "group": group,
        "qkv":     [],
        "o":       [],
        "gate_up": [],
        "down":    [],
    }
    for layer in lang.layers:
        fq = layer.self_attn.fused_qkv
        cache["qkv"].append(
            quantize_pack_w4_sym(fq.qkv_proj.weight.data, group))
        cache["o"].append(
            quantize_pack_w4_sym(layer.self_attn.o_proj.weight.data, group))
        cache["gate_up"].append(
            quantize_pack_w4_sym(layer.mlp.gate_up_proj.weight.data, group))
        cache["down"].append(
            quantize_pack_w4_sym(layer.mlp.down_proj.weight.data, group))
    model._w4_decoder_cache = cache
    print(f"[quant_swap] decoder W4 cache built: {len(cache['qkv'])} layers, "
          f"group={group}")

def _swap_lm_head(model: nn.Module, device: str = "cuda:0"):
    lm_head = model.lm_head
    if not isinstance(lm_head, nn.Linear):
        print(f"[quant_swap] lm_head is {type(lm_head).__name__}, skipping")
        return

    from my_kernel.fused_w4_gemv import W4Linear, _load as _load_w4
    _load_w4()

    out_features, in_features = lm_head.weight.shape
    print(f"[quant_swap] quantizing lm_head ({out_features}, {in_features}) "
          f"with W4 GEMV group={_group()}")
    new_lm_head = W4Linear(lm_head.weight.data.to(device), group_size=_group())
    new_lm_head = new_lm_head.to(device)
    model.lm_head = new_lm_head

class _ScatteredLMHead(nn.Module):
    """[SHLEE] Channel-pruned lm_head: reduced-vocab W4 GEMV + scatter to full.

    Holds a W4Linear over only K_keep rows; forward scatters its output back
    into a full-vocab tensor (pruned positions filled with -65000 so argmax
    never picks them).  Output shape stays (..., full_vocab) so all downstream
    code (graph capture, argmax, sampling) works unchanged.

    Proxies W4Linear-shaped attributes so introspectors (e.g. LeanDecodeState
    reading .scales for device/dtype) keep working.
    """
    def __init__(self, reduced, kept_idx: torch.Tensor, full_vocab: int):
        super().__init__()
        self.reduced = reduced
        self.register_buffer("kept_idx", kept_idx)
        self.full_vocab = full_vocab
        self.in_features = reduced.in_features
        self.out_features = full_vocab

        sc = reduced.scales
        self._reduced_buf = torch.empty(
            1, reduced.out_features, dtype=sc.dtype, device=sc.device,
        )

    @property
    def scales(self):
        return self.reduced.scales

    @property
    def qweight(self):
        return self.reduced.qweight

    @property
    def group_size(self):
        return self.reduced.group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced_logits = self.reduced(x)
        out = torch.full(
            x.shape[:-1] + (self.full_vocab,), -65000.0,
            dtype=reduced_logits.dtype, device=reduced_logits.device,
        )
        out.index_copy_(-1, self.kept_idx, reduced_logits)
        return out

    def forward_into(self, x: torch.Tensor, logits_out: torch.Tensor) -> None:
        """Write logits into a preallocated `logits_out` buffer.

        Resets to sentinel (-65000) each call so that stale logits from a
        previous decode step in a different module (e.g. main graph using
        a full lm_head) do not leak into argmax.
        """
        from my_kernel.fused_w4_gemv import gemv_w4_dispatch

        logits_out.fill_(-65000)

        gemv_w4_dispatch(
            x.reshape(-1),
            self.reduced.qweight, self.reduced.scales,
            self._reduced_buf.reshape(-1),
            self.reduced.group_size,
        )

        logits_out.index_copy_(-1, self.kept_idx, self._reduced_buf)

def _collect_mandatory_token_ids(model: nn.Module, vocab_size: int) -> set:
    """[SHLEE] Tokens that MUST stay in the kept set regardless of L2 norm.

    Without these, lm_head pruning drops EOS/special tokens and the model
    can never terminate → generate_answer runs to max_new_tokens=1024 every
    sample, demolishing wall-clock time AND accuracy.
    """
    mandatory = set()

    for cfg in (getattr(model, "generation_config", None),
                model.config.get_text_config(),
                model.config):
        if cfg is None: continue
        for attr in ("eos_token_id", "pad_token_id", "bos_token_id",
                     "decoder_start_token_id", "image_token_id",
                     "video_token_id", "vision_start_token_id",
                     "vision_end_token_id"):
            v = getattr(cfg, attr, None)
            if v is None: continue
            if isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, int): mandatory.add(x)
            elif isinstance(v, int):
                mandatory.add(v)

    SPECIAL_START = 151643
    for tid in range(SPECIAL_START, vocab_size):
        mandatory.add(tid)

    return mandatory

def _load_calibrated_keep_payload(N: int):
    """Returns (idx_tensor, mandatory_count) or None."""
    import json as _json
    from pathlib import Path as _Path
    default_path = _Path(__file__).with_name("lm_head_keep_indices.json")
    path = _Path(os.environ.get("MY_KERNEL_LMHEAD_KEEP_JSON", str(default_path)))
    if not path.exists():
        return None
    try:
        with path.open() as f:
            payload = _json.load(f)
    except Exception as e:
        print(f"[lm_head_prune] failed to read {path}: {e}")
        return None
    keep = payload.get("keep")
    if not isinstance(keep, list) or not keep:
        return None
    file_vocab = int(payload.get("vocab_size", N))
    if file_vocab != N:
        print(f"[lm_head_prune] {path}: vocab_size mismatch "
              f"{file_vocab}!={N}; ignoring file")
        return None
    idx = torch.tensor(keep, dtype=torch.long)
    mc = int(payload.get("calibration", {}).get("mandatory_count", 0))
    print(f"[lm_head_prune] loaded {path.name}: "
          f"{idx.numel()} keep, mandatory_count={mc}")
    return idx, mc


def _build_runtime_keep(calib_keep_idx, mandatory_count, mand_set,
                        K_keep: int, low_cap: int):
    """Build keep set respecting MY_KERNEL_LMHEAD_LOW_CAP partition.

    JSON layout: keep[:mandatory_count] = sorted(low_IDs + special), then
    keep[mandatory_count:] = freq-ranked.  At runtime we pick at most
    `low_cap` tokens from the low/special section, fill the rest from the
    freq section, up to K_keep.
    """
    keep_list = calib_keep_idx.tolist()
    low_pool  = [int(x) for x in keep_list[:mandatory_count] if int(x) not in mand_set]
    freq_pool = [int(x) for x in keep_list[mandatory_count:] if int(x) not in mand_set]

    keep = set(mand_set)
    budget = K_keep - len(keep)
    low_take = max(0, min(low_cap - len(mand_set), len(low_pool), budget))
    keep.update(low_pool[:low_take])
    budget = K_keep - len(keep)
    freq_take = max(0, min(budget, len(freq_pool)))
    keep.update(freq_pool[:freq_take])
    return sorted(keep), low_take, freq_take


def _load_calibrated_keep_ids(N: int) -> "torch.Tensor | None":
    res = _load_calibrated_keep_payload(N)
    if res is None:
        return None
    idx, _ = res
    return idx
    # tail kept for backward compatibility of fallback path
    import json as _json  # unreachable
    from pathlib import Path as _Path
    default_path = _Path(__file__).with_name("lm_head_keep_indices.json")
    path = _Path(os.environ.get("MY_KERNEL_LMHEAD_KEEP_JSON", str(default_path)))
    if not path.exists():
        return None
    try:
        with path.open() as f:
            payload = _json.load(f)
    except Exception as e:
        print(f"[lm_head_prune] failed to read {path}: {e}")
        return None
    keep = payload.get("keep")
    if not isinstance(keep, list) or not keep:
        print(f"[lm_head_prune] {path}: empty/invalid 'keep' field")
        return None
    file_vocab = int(payload.get("vocab_size", N))
    if file_vocab != N:
        print(f"[lm_head_prune] {path}: vocab_size mismatch "
              f"{file_vocab}!={N}; ignoring file")
        return None
    idx = torch.tensor(keep, dtype=torch.long)
    if int(idx.min()) < 0 or int(idx.max()) >= N:
        print(f"[lm_head_prune] {path}: out-of-range index; ignoring")
        return None
    print(f"[lm_head_prune] loaded calibrated keep set from {path.name}: "
          f"{idx.numel()} tokens "
          f"(meta: {payload.get('calibration', {})})")
    return idx

def _prune_lm_head(model: nn.Module, device: str = "cuda:0"):
    """[SHLEE] Drop low-importance vocab rows from the (W4) lm_head.

    Selection priority:
      1) calibrated keep set from lm_head_keep_indices.json if present
         (ranked by observed top-K argmax frequency on calibration corpus)
      2) FP16 weight L2-norm fallback

    Wraps the reduced W4Linear in _ScatteredLMHead.  Run AFTER _swap_lm_head().
    """
    K_keep = _lm_head_prune_k()
    if K_keep <= 0:
        return None

    from my_kernel.fused_w4_gemv import W4Linear, dequant_w4_to_fp16

    lm = model.lm_head
    if not isinstance(lm, W4Linear):
        print(f"[lm_head_prune] lm_head is {type(lm).__name__}, expected "
              f"W4Linear; skipping")
        return None

    N = lm.out_features
    K = lm.in_features
    if K_keep >= N:
        print(f"[lm_head_prune] K_keep={K_keep} >= vocab={N}; no-op")
        return None

    calib_payload = _load_calibrated_keep_payload(N)
    mandatory = _collect_mandatory_token_ids(model, N)
    if calib_payload is None:
        low_id_cap = min(_lm_head_prune_low_id_cap(), N)
        mandatory.update(range(low_id_cap))
        print(f"[lm_head_prune] always-kept: {len(mandatory)} "
              f"(low IDs [0,{low_id_cap}) + special tokens)")
    mandatory_t = torch.tensor(sorted(mandatory), dtype=torch.long, device=device)

    W_full = dequant_w4_to_fp16(lm.qweight, lm.scales, lm.group_size, K)

    if calib_payload is not None:
        calib_keep, mandatory_count = calib_payload
        mand_cpu = mandatory_t.cpu()
        mand_set = set(mand_cpu.tolist())
        low_cap_env = int(os.environ.get("MY_KERNEL_LMHEAD_LOW_CAP", "10000"))
        all_keep, low_take, freq_take = _build_runtime_keep(
            calib_keep, mandatory_count, mand_set, K_keep, low_cap_env)
        top_idx = torch.tensor(all_keep, dtype=torch.long, device=device)
        K_actual = top_idx.numel()
        print(f"[lm_head_prune] CALIBRATED: {len(mand_set)} special + "
              f"{low_take} low + {freq_take} freq "
              f"(LOW_CAP={low_cap_env}) → {K_actual}/{N} "
              f"({(1 - K_actual/N)*100:.1f}% rows skipped)")
    else:

        norms = W_full.float().norm(dim=1)
        norms_masked = norms.clone()
        norms_masked[mandatory_t.to(norms.device)] = -float("inf")
        n_norm_pick = max(0, K_keep - len(mandatory))
        if n_norm_pick > 0:
            _, top_norm_idx = norms_masked.topk(n_norm_pick, largest=True)
            top_idx = torch.cat([mandatory_t.to(top_norm_idx.device), top_norm_idx])
        else:
            top_idx = mandatory_t.to(norms.device)
        top_idx = torch.unique(top_idx)
        top_idx, _ = top_idx.sort()
        K_actual = top_idx.numel()
        print(f"[lm_head_prune] L2-norm fallback: {len(mandatory)} mandatory + "
              f"{max(0, K_actual - len(mandatory))} top by L2 norm → "
              f"{K_actual}/{N} ({(1 - K_actual/N)*100:.1f}% rows skipped)")

    W_reduced = W_full[top_idx].contiguous()
    reduced_w4 = W4Linear(W_reduced.to(device), group_size=lm.group_size).to(device)
    new_lm = _ScatteredLMHead(reduced_w4, top_idx.to(device).long(), N).to(device)
    model.lm_head = new_lm
    print(f"[lm_head_prune] reduced+scatter: GEMV ({K_actual}, {K})")
    return new_lm

class _ScatteredLMHeadFP16(nn.Module):
    def __init__(self, reduced_linear, kept_idx: torch.Tensor, full_vocab: int):
        super().__init__()
        self.reduced = reduced_linear
        self.reduced.weight.requires_grad_(False)
        self.register_buffer("kept_idx", kept_idx)
        self.full_vocab = full_vocab
        self.in_features = reduced_linear.in_features
        self.out_features = full_vocab
        w = reduced_linear.weight
        self._reduced_buf = torch.empty(
            1, reduced_linear.out_features, dtype=w.dtype, device=w.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduced(x)
        out = torch.full(
            x.shape[:-1] + (self.full_vocab,), -65000.0,
            dtype=reduced.dtype, device=reduced.device,
        )
        out.index_copy_(-1, self.kept_idx, reduced)
        return out

    def forward_into(self, x: torch.Tensor, logits_out: torch.Tensor) -> None:
        torch.mm(
            x.reshape(1, self.in_features),
            self.reduced.weight.t(),
            out=self._reduced_buf,
        )
        logits_out.index_copy_(-1, self.kept_idx, self._reduced_buf)


def _prune_lm_head_fp16(model: nn.Module, device: str = "cuda:0"):
    K_keep = _lm_head_prune_k()
    if K_keep <= 0:
        return None
    lm = model.lm_head
    if not isinstance(lm, nn.Linear):
        print(f"[lm_head_prune_fp16] lm_head is {type(lm).__name__}, skipping")
        return None
    N = lm.out_features
    K = lm.in_features
    if K_keep >= N:
        print(f"[lm_head_prune_fp16] K_keep={K_keep} >= vocab={N}; no-op")
        return None

    calib_payload = _load_calibrated_keep_payload(N)
    mandatory = _collect_mandatory_token_ids(model, N)
    if calib_payload is None:
        low_id_cap = min(_lm_head_prune_low_id_cap(), N)
        mandatory.update(range(low_id_cap))
        print(f"[lm_head_prune_fp16] always-kept: {len(mandatory)} "
              f"(low IDs [0,{low_id_cap}) + special tokens)")
    mandatory_t = torch.tensor(sorted(mandatory), dtype=torch.long, device=device)

    if calib_payload is not None:
        calib_keep, mandatory_count = calib_payload
        mand_set = set(mandatory_t.cpu().tolist())
        low_cap_env = int(os.environ.get("MY_KERNEL_LMHEAD_LOW_CAP", "10000"))
        all_keep, low_take, freq_take = _build_runtime_keep(
            calib_keep, mandatory_count, mand_set, K_keep, low_cap_env)
        top_idx = torch.tensor(all_keep, dtype=torch.long, device=device)
        K_actual = top_idx.numel()
        print(f"[lm_head_prune_fp16] CALIBRATED: {len(mand_set)} special + "
              f"{low_take} low + {freq_take} freq "
              f"(LOW_CAP={low_cap_env}) → {K_actual}/{N}")
    else:
        W_full = lm.weight.data
        norms = W_full.float().norm(dim=1)
        norms_masked = norms.clone()
        norms_masked[mandatory_t.to(norms.device)] = -float("inf")
        n_norm_pick = max(0, K_keep - len(mandatory))
        if n_norm_pick > 0:
            _, top_norm_idx = norms_masked.topk(n_norm_pick, largest=True)
            top_idx = torch.cat([mandatory_t.to(top_norm_idx.device), top_norm_idx])
        else:
            top_idx = mandatory_t.to(norms.device)
        top_idx = torch.unique(top_idx)
        top_idx, _ = top_idx.sort()
        K_actual = top_idx.numel()
        print(f"[lm_head_prune_fp16] L2-norm fallback: → {K_actual}/{N}")

    W_reduced = lm.weight.data[top_idx].contiguous().to(device)
    reduced = nn.Linear(K, K_actual, bias=False, device=device, dtype=lm.weight.dtype)
    reduced.weight.data.copy_(W_reduced)
    new_lm = _ScatteredLMHeadFP16(reduced, top_idx.to(device).long(), N).to(device)
    model.lm_head = new_lm
    print(f"[lm_head_prune_fp16] reduced+scatter: cuBLAS fp16 GEMV ({K_actual}, {K})")
    return new_lm


def make_tiny_lm_head(model: nn.Module, K_tiny: int, low_cap_tiny: int,
                      device: str = "cuda:0"):
    """Build a separate W4 reduced lm_head for the tiny-graph speculative
    continuation path.  The main lm_head (model.lm_head) is left untouched
    — long-form generation and the main decode graph keep using full
    vocabulary, while the tiny graph routes through this reduced module.

    Attaches result as model._tiny_lm_head (or returns None if unavailable).
    """
    from my_kernel.fused_w4_gemv import W4Linear, dequant_w4_to_fp16

    lm = model.lm_head
    if isinstance(lm, nn.Linear):
        W_full = lm.weight.data
        N, K = W_full.shape
        dtype = W_full.dtype
        kept_idx_main = None
    elif isinstance(lm, _ScatteredLMHead):
        N = lm.full_vocab
        K = lm.in_features
        kept_idx_main = lm.kept_idx.cpu().tolist()
        reduced_W = dequant_w4_to_fp16(
            lm.reduced.qweight, lm.reduced.scales, lm.reduced.group_size, K
        ).to(device)
        W_full = torch.zeros((N, K), dtype=torch.float16, device=device)
        W_full[lm.kept_idx.to(device)] = reduced_W
        dtype = torch.float16
    elif hasattr(lm, "qweight"):
        N = lm.out_features
        K = lm.in_features
        W_full = dequant_w4_to_fp16(lm.qweight, lm.scales, lm.group_size, K).to(device)
        dtype = torch.float16
        kept_idx_main = None
    else:
        print(f"[tiny_lm_head] lm_head is {type(lm).__name__}; skipping")
        return None

    payload = _load_calibrated_keep_payload(N)
    if payload is None:
        print(f"[tiny_lm_head] no calibration JSON; skipping")
        return None
    calib_keep, mandatory_count = payload

    mandatory = _collect_mandatory_token_ids(model, N)
    mand_set = set(sorted(mandatory))
    all_keep, low_take, freq_take = _build_runtime_keep(
        calib_keep, mandatory_count, mand_set, K_tiny, low_cap_tiny)

    if kept_idx_main is not None:
        kept_set_main = set(kept_idx_main)
        valid_keep = [g for g in all_keep if g in kept_set_main]
        dropped = len(all_keep) - len(valid_keep)
        if dropped:
            print(f"[tiny_lm_head] dropped {dropped} ids not in main lm_head kept set")
        all_keep = valid_keep

    top_idx = torch.tensor(all_keep, dtype=torch.long, device=device)
    K_actual = top_idx.numel()
    print(f"[tiny_lm_head] K_tiny={K_tiny}: special {len(mand_set)} + "
          f"low {low_take} + freq {freq_take} → {K_actual}/{N}")

    W_reduced = W_full[top_idx].contiguous().to(device).to(dtype)
    reduced_w4 = W4Linear(W_reduced, group_size=_group()).to(device)
    tiny_lm = _ScatteredLMHead(reduced_w4, top_idx.to(device).long(), N).to(device)
    model._tiny_lm_head = tiny_lm
    return tiny_lm


def apply_quant(model: nn.Module, device: str = "cuda:0"):
    mode = _mode()
    if mode == "off":
        if is_lm_head_pruning_enabled():
            _prune_lm_head_fp16(model, device=device)
        return
    print(f"[quant_swap] mode={mode} algo={_algo()} group={_group()}")
    if is_lm_head_quant_enabled():
        _swap_lm_head(model, device=device)
        if is_lm_head_pruning_enabled():
            _prune_lm_head(model, device=device)
    elif is_lm_head_pruning_enabled():
        _prune_lm_head_fp16(model, device=device)
    if is_decoder_quant_enabled():
        print("[quant_swap] decoder-layer quant: per-bucket in LeanDecodeState")
