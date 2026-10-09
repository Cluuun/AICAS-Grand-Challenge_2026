"""Low-rank MLP approximation for decode acceleration.

Replaces gate_up + down_proj (2 GEMVs, 18MB weight) with a single
fitted W [2048, 2048] (1 GEMV, 4MB weight) per target layer.
Only active during decode (is_decode=True); prefill uses the full MLP.
"""
import json
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _collect_real_activations(model, n_samples=30):
    """Run real samples through the model and collect MLP I/O at each layer."""
    from datasets import load_from_disk

    lm = model.model.language_model
    layers = lm.layers
    device = next(model.model.parameters()).device

    # Collect hooks: {layer_idx: [(input, output), ...]}
    hooks = {}
    handles = []

    def make_hook(layer_idx):
        collected = []
        def hook_fn(module, input, output):
            # input[0] is the hidden_states after RMS norm
            x = input[0].detach().float()
            y = output.detach().float()
            collected.append((x.cpu(), y.cpu()))
        hooks[layer_idx] = collected
        return hook_fn

    target_layers = set()
    spec = os.environ.get("AICAS_LOWRANK_MLP_LAYERS", "").strip()
    if spec:
        target_layers = {int(x) for x in spec.replace("+", ",").split(",") if x.strip()}

    for li in target_layers:
        if li < len(layers):
            h = layers[li].mlp.register_forward_hook(make_hook(li))
            handles.append(h)

    # Load a few real samples
    dataset = load_from_disk("./data")
    subset_path = "eval_subset_150.json"
    try:
        with open(subset_path) as f:
            indices = json.load(f)["indices"]
    except Exception:
        indices = list(range(min(n_samples, len(dataset))))

    from PIL import Image
    collected = 0
    for idx in indices[:n_samples]:
        try:
            sample = dataset[int(idx)]
            messages = [{"role": "user", "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"]},
            ]}]
            inputs = model.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                _ = model.model.generate(**inputs, max_new_tokens=5, do_sample=False)
            collected += 1
        except Exception as e:
            logger.warning("[lowrank] Calib sample failed: %s", e)

    # Remove hooks
    for h in handles:
        h.remove()

    # Extract per-layer activations
    # Each hook collected (x, y) pairs where x shape varies (prefill seq_len varies)
    result = {}
    for li in target_layers:
        if li not in hooks or not hooks[li]:
            continue
        # Concatenate all tokens from all samples
        xs, ys = [], []
        for x, y in hooks[li]:
            # x: [seq_len, 2048], y: [seq_len, 2048]
            xs.append(x.reshape(-1, x.shape[-1]))
            ys.append(y.reshape(-1, y.shape[-1]))
        result[li] = (torch.cat(xs, dim=0), torch.cat(ys, dim=0))

    logger.info("[lowrank] Collected real activations from %d samples (%d layers)",
                collected, len(result))
    return result


def setup_lowrank_mlp(model):
    """Compute and store low-rank MLP approximations for target layers.

    Reads AICAS_LOWRANK_MLP_LAYERS (e.g. "25,26,27") and
    AICAS_LOWRANK_MLP_RANK (default 256) from environment.
    """
    spec = os.environ.get("AICAS_LOWRANK_MLP_LAYERS", "").strip()
    if not spec:
        return 0
    target_layers = {int(x) for x in spec.replace("+", ",").split(",") if x.strip()}
    if not target_layers:
        return 0

    rank = int(os.environ.get("AICAS_LOWRANK_MLP_RANK", "256"))
    use_real_calib = os.environ.get("AICAS_LOWRANK_MLP_REAL_CALIB", "1") == "1"

    lm = model.model.language_model
    layers = lm.layers
    patched = 0

    # Collect real activations if enabled
    real_acts = {}
    if use_real_calib:
        try:
            n_calib_samples = int(os.environ.get("AICAS_LOWRANK_MLP_CALIB_SAMPLES", "20"))
            real_acts = _collect_real_activations(model, n_calib_samples)
        except Exception as e:
            logger.warning("[lowrank] Real calibration failed, falling back to random: %s", e)

    for layer_idx in sorted(target_layers):
        if layer_idx >= len(layers):
            continue
        mlp = layers[layer_idx].mlp
        if not hasattr(mlp, 'gate_up_proj'):
            continue

        gu_w = mlp.gate_up_proj.weight.data   # [12288, 2048]
        dp_w = mlp.down_proj.weight.data       # [2048, 6144]
        hidden_dim = gu_w.shape[1]             # 2048

        if layer_idx in real_acts:
            X, Y = real_acts[layer_idx]
            logger.info("[lowrank] Layer %d: using %d real activation tokens",
                        layer_idx, X.shape[0])
        else:
            # Fallback: random Gaussian calibration
            n_calib = 200
            torch.manual_seed(42 + layer_idx)
            X = torch.randn(n_calib, hidden_dim, device=gu_w.device, dtype=torch.float32)
            X = X / (hidden_dim ** 0.5)
            Y = []
            with torch.no_grad():
                for i in range(n_calib):
                    x = X[i].to(gu_w.dtype)
                    gate_up = torch.matmul(gu_w, x)
                    gate, up = gate_up.chunk(2)
                    mid = torch.nn.functional.silu(gate) * up
                    y = torch.matmul(dp_w, mid)
                    Y.append(y.float())
            Y = torch.stack(Y)
            logger.info("[lowrank] Layer %d: using random calibration (%d samples)",
                        layer_idx, n_calib)

        X = X.to(gu_w.device)
        Y = Y.to(gu_w.device)

        # Least-squares fit: Y ≈ X @ W^T
        lam = 1e-3
        XtX = X.T @ X + lam * torch.eye(hidden_dim, device=X.device)
        XtY = X.T @ Y
        W = torch.linalg.solve(XtX, XtY)  # [2048, 2048]
        mse_full = (Y - X @ W.T).pow(2).mean().item()

        # SVD truncate
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = min(rank, len(S))
        W_lr = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]

        mse_lr = (Y - X @ W_lr.T).pow(2).mean().item()
        logger.info(
            "[lowrank] Layer %d: rank=%d MSE_full=%.6f MSE_lr=%.6f ratio=%.3f",
            layer_idx, r, mse_full, mse_lr,
            mse_lr / max(mse_full, 1e-10))

        # INT8 per-channel quantization (scale must be float32 for gemv_int8 kernel)
        W_f = W_lr.float()
        W_absmax = W_f.abs().max(dim=1)[0].clamp(min=1e-8)
        W_scale = (W_absmax / 127.0).to(torch.float32)
        W_int8 = (W_f / W_scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        mlp._lr_W_int8 = W_int8.to(gu_w.device)
        mlp._lr_W_scale = W_scale.to(gu_w.device)
        patched += 1

    if patched:
        print(f"[lowrank_mlp] Patched {patched} modules")
    return patched
