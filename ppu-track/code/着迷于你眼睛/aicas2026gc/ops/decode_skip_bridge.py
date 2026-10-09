from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_bridge_tmp_kernel(
    x_ptr,
    tmp_ptr,
    mean_in_ptr,
    low_a_ptr,
    H: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_r = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_H)
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)

    x = tl.load(x_ptr + offs_h).to(tl.float32)
    mean_in = tl.load(mean_in_ptr + offs_h).to(tl.float32)
    centered = x - mean_in
    a = tl.load(low_a_ptr + offs_h[:, None] * R + offs_r[None, :]).to(tl.float32)
    acc = tl.sum(centered[:, None] * a, axis=0)
    tl.store(tmp_ptr + offs_r, acc.to(tl.float32))


@triton.jit
def _decode_bridge_out_kernel(
    x_ptr,
    tmp_ptr,
    scale_ptr,
    bias_ptr,
    mean_out_ptr,
    low_b_ptr,
    H: tl.constexpr,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid_d = tl.program_id(0)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)

    tmp = tl.load(tmp_ptr + offs_r).to(tl.float32)
    b = tl.load(low_b_ptr + offs_r[:, None] * H + offs_d[None, :]).to(tl.float32)
    delta = tl.sum(tmp[:, None] * b, axis=0)

    x = tl.load(x_ptr + offs_d).to(tl.float32)
    scale = tl.load(scale_ptr + offs_d).to(tl.float32)
    bias = tl.load(bias_ptr + offs_d).to(tl.float32)
    mean_out = tl.load(mean_out_ptr + offs_d).to(tl.float32)
    out = x * scale + bias + delta + mean_out
    tl.store(x_ptr + offs_d, out)


def fused_decode_skip_bridge_(
    hidden_states: torch.Tensor,
    tmp: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    mean_in: torch.Tensor,
    mean_out: torch.Tensor,
    low_a: torch.Tensor,
    low_b: torch.Tensor,
) -> torch.Tensor:
    """Apply the decode skip LS affine + lowrank bridge in-place.

    This is specialized for the current BS=1 decode bridge:
    hidden_states shape [1, 1, 2048], rank 128, fp16 params.
    """
    if hidden_states.numel() != 2048:
        raise ValueError(f"hidden_states must have 2048 elements, got {hidden_states.numel()}")
    if low_a.shape != (2048, 128) or low_b.shape != (128, 2048):
        raise ValueError(f"expected low_a=(2048,128), low_b=(128,2048), got {tuple(low_a.shape)} {tuple(low_b.shape)}")
    if tmp.numel() < 128:
        raise ValueError(f"tmp must have at least 128 elements, got {tmp.numel()}")
    if not hidden_states.is_cuda:
        raise ValueError("hidden_states must be CUDA")
    if hidden_states.dtype != torch.float16:
        raise ValueError(f"hidden_states must be float16, got {hidden_states.dtype}")

    x = hidden_states.view(-1)
    tmp_1d = tmp.view(-1)

    _decode_bridge_tmp_kernel[(8,)](
        x,
        tmp_1d,
        mean_in.view(-1),
        low_a,
        H=2048,
        R=128,
        BLOCK_R=16,
        BLOCK_H=2048,
        num_warps=8,
        num_stages=1,
    )
    _decode_bridge_out_kernel[(64,)](
        x,
        tmp_1d,
        scale.view(-1),
        bias.view(-1),
        mean_out.view(-1),
        low_b,
        H=2048,
        R=128,
        BLOCK_D=32,
        BLOCK_R=128,
        num_warps=4,
        num_stages=1,
    )
    return hidden_states


def torch_decode_skip_bridge(
    hidden_states: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    mean_in: torch.Tensor,
    mean_out: torch.Tensor,
    low_a: torch.Tensor,
    low_b: torch.Tensor,
) -> torch.Tensor:
    source = hidden_states.clone()
    out = hidden_states * scale.view(1, 1, -1)
    out = out + bias.view(1, 1, -1)
    centered = source - mean_in.view(1, 1, -1)
    lowrank_hidden = torch.matmul(centered, low_a)
    lowrank_delta = torch.matmul(lowrank_hidden, low_b).contiguous()
    lowrank_delta.add_(mean_out.view(1, 1, -1))
    return out + lowrank_delta


def _load_default_payloads(device: torch.device):
    root = Path(__file__).resolve().parents[2]
    affine_path = root / "layer_stats/steering_all_layers_calib500_decode64/vectors/decode_skip_6_14_ls_affine.pt"
    lowrank_path = root / "layer_stats/steering_all_layers_calib500_decode64/lowrank_6_14/decode_skip_6_14_ls_residual_lowrank_k128.pt"
    affine = torch.load(affine_path, map_location="cpu")
    lowrank = torch.load(lowrank_path, map_location="cpu")
    return (
        affine["scale_vector"].to(device=device, dtype=torch.float16).contiguous(),
        affine["bias_vector"].to(device=device, dtype=torch.float16).contiguous(),
        lowrank["mean_in"].to(device=device, dtype=torch.float16).contiguous(),
        lowrank["mean_out"].to(device=device, dtype=torch.float16).contiguous(),
        lowrank["low_a"].to(device=device, dtype=torch.float16).contiguous(),
        lowrank["low_b"].to(device=device, dtype=torch.float16).contiguous(),
    )


if __name__ == "__main__":
    import triton.testing

    torch.manual_seed(0)
    device = torch.device("cuda")
    scale, bias, mean_in, mean_out, low_a, low_b = _load_default_payloads(device)
    x = torch.randn(1, 1, 2048, device=device, dtype=torch.float16)
    tmp = torch.empty(128, device=device, dtype=torch.float32)

    ref = torch_decode_skip_bridge(x, scale, bias, mean_in, mean_out, low_a, low_b)
    got = x.clone()
    fused_decode_skip_bridge_(got, tmp, scale, bias, mean_in, mean_out, low_a, low_b)
    torch.cuda.synchronize()

    diff = (got - ref).float()
    cos = torch.nn.functional.cosine_similarity(got.float().view(1, -1), ref.float().view(1, -1)).item()
    print(
        f"correctness max_abs={diff.abs().max().item():.6f} "
        f"mean_abs={diff.abs().mean().item():.6f} cos={cos:.8f}"
    )

    def torch_path():
        torch_decode_skip_bridge(x, scale, bias, mean_in, mean_out, low_a, low_b)

    def triton_path():
        y = x.clone()
        fused_decode_skip_bridge_(y, tmp, scale, bias, mean_in, mean_out, low_a, low_b)

    torch_ms = triton.testing.do_bench(torch_path)
    triton_ms = triton.testing.do_bench(triton_path)
    print(f"speed torch={torch_ms:.6f}ms triton={triton_ms:.6f}ms")
