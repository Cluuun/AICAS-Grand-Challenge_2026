"""MLP neuron pruning for decode acceleration.

Selects top-k neurons by weight magnitude importance, creates smaller
gate_up_proj and down_proj weight matrices. Preserves SiLU nonlinearity
and MLP structure — just uses fewer intermediate neurons.

Bandwidth savings: each pruned layer saves (6144 - k) * 6144 bytes INT8.
"""
import os
import torch


def _quantize_int8_per_channel(weight_fp16):
    """Quantize FP16 weight to INT8 with per-channel scaling."""
    w = weight_fp16.float()
    scale = w.abs().max(dim=1)[0].clamp(min=1e-8) / 127.0
    w_int8 = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale.to(torch.float32)


def setup_mlp_neuron_pruning(model):
    """Prune MLP neurons on target layers based on weight magnitude importance.

    Reads AICAS_MLP_PRUNE_LAYERS (e.g. "10,11,12,13") and
    AICAS_MLP_PRUNE_KEEP_RATIO (0.0-1.0, fraction of neurons to keep).
    """
    spec = os.environ.get("AICAS_MLP_PRUNE_LAYERS", "").strip()
    if not spec:
        return 0
    target_layers = {int(x) for x in spec.replace("+", ",").split(",") if x.strip()}
    if not target_layers:
        return 0

    keep_ratio = float(os.environ.get("AICAS_MLP_PRUNE_KEEP_RATIO", "0.5"))

    lm = model.model.language_model
    layers = lm.layers
    patched = 0

    for layer_idx in sorted(target_layers):
        if layer_idx >= len(layers):
            continue
        mlp = layers[layer_idx].mlp
        if not hasattr(mlp, 'gate_up_proj'):
            continue

        # Get FP16 weights (still available after INT8 quantization)
        gu_w = mlp.gate_up_proj.weight.data  # [12288, 2048] (gate rows 0-6143, up rows 6144-12287)
        dp_w = mlp.down_proj.weight.data      # [2048, 6144]
        intermediate_size = gu_w.shape[0] // 2  # 6144
        k = max(32, (int(intermediate_size * keep_ratio) // 32) * 32)

        # Compute neuron importance: product of gate, up, and down norms
        gate_norms = gu_w[:intermediate_size].float().norm(dim=1)       # [6144]
        up_norms = gu_w[intermediate_size:].float().norm(dim=1)         # [6144]
        down_norms = dp_w.float().norm(dim=0)                            # [6144]
        importance = gate_norms * up_norms * down_norms                   # [6144]

        # Select top-k neurons
        _, top_indices = torch.topk(importance, k)
        top_indices, _ = torch.sort(top_indices)  # sorted for better memory access

        # Create pruned weights
        # gate_up: select gate rows and up rows for top-k neurons
        gate_rows = top_indices
        up_rows = top_indices + intermediate_size
        all_rows = torch.cat([gate_rows, up_rows])
        pruned_gu_w = gu_w[all_rows]               # [2k, 2048]
        pruned_dp_w = dp_w[:, top_indices]           # [2048, k]

        # Quantize pruned weights to INT8
        gu_int8, gu_scale = _quantize_int8_per_channel(pruned_gu_w)
        dp_int8, dp_scale = _quantize_int8_per_channel(pruned_dp_w)

        # Replace weights
        mlp.gate_up_proj.weight_int8 = gu_int8.to(gu_w.device)
        mlp.gate_up_proj.weight_scale = gu_scale.to(gu_w.device)
        mlp.gate_up_proj.weight = torch.nn.Parameter(pruned_gu_w.half().to(gu_w.device))

        mlp.down_proj.weight_int8 = dp_int8.to(dp_w.device)
        mlp.down_proj.weight_scale = dp_scale.to(dp_w.device)
        mlp.down_proj.weight = torch.nn.Parameter(pruned_dp_w.half().to(dp_w.device))

        # Store metadata
        mlp._pruned_intermediate = k
        mlp._pruned_original = intermediate_size

        saved_mb = (intermediate_size - k) * 6144 / 1024 / 1024
        print(f"[neuron_prune] Layer {layer_idx}: kept {k}/{intermediate_size} neurons "
              f"({keep_ratio*100:.0f}%), saved {saved_mb:.1f}MB/step")
        patched += 1

    if patched:
        total_saved = patched * (6144 - max(1, int(6144 * keep_ratio))) * 6144 / 1024 / 1024
        print(f"[neuron_prune] Total: {patched} layers pruned, ~{total_saved:.0f}MB saved per step")
    return patched
