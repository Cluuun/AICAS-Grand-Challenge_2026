from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spec_decode.kernels.bf16_qk_rotary.runtime import run_bf16_qk_rmsnorm_rotary


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = (xf * xf).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * weight.float()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
):
    q_len = int(q.shape[0]) // 16
    cos_f = cos.reshape(q_len, 128).float()
    sin_f = sin.reshape(q_len, 128).float()

    qn = _rmsnorm(q, q_weight, eps).reshape(q_len, 16, 128)
    kn = _rmsnorm(k, k_weight, eps).reshape(q_len, 8, 128)

    qrot = qn * cos_f[:, None, :] + _rotate_half(qn) * sin_f[:, None, :]
    krot = kn * cos_f[:, None, :] + _rotate_half(kn) * sin_f[:, None, :]

    qout = qrot.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16).contiguous()
    kout = krot.permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16).contiguous()
    return qout, kout


def _run_case(q_len: int) -> None:
    device = "cuda"
    eps = 1e-6
    q = torch.randn(q_len * 16, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(q_len * 8, 128, device=device, dtype=torch.bfloat16)
    cos = torch.randn(q_len, 128, device=device, dtype=torch.bfloat16)
    sin = torch.randn(q_len, 128, device=device, dtype=torch.bfloat16)
    q_weight = torch.randn(128, device=device, dtype=torch.bfloat16)
    k_weight = torch.randn(128, device=device, dtype=torch.bfloat16)

    got = run_bf16_qk_rmsnorm_rotary(q, k, cos, sin, q_weight, k_weight, eps)
    if got is None:
        raise RuntimeError(f"bf16 qk rotary backend rejected q_len={q_len}")
    q_got, k_got = got
    q_ref, k_ref = _reference(q, k, cos, sin, q_weight, k_weight, eps)
    torch.cuda.synchronize()

    q_diff = (q_got - q_ref).abs()
    k_diff = (k_got - k_ref).abs()
    q_max = q_diff.max().item()
    k_max = k_diff.max().item()
    q_mean = q_diff.float().mean().item()
    k_mean = k_diff.float().mean().item()
    print(
        f"q_len={q_len} q_max={q_max:.6f} q_mean={q_mean:.6f} "
        f"k_max={k_max:.6f} k_mean={k_mean:.6f}"
    )
    if max(q_max, k_max) > 0.03125:
        raise RuntimeError(f"diff too high for q_len={q_len}: q={q_max} k={k_max}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    for q_len in (1, 2, 3, 4, 8, 15, 16):
        _run_case(q_len)
    print("bf16 qk rmsnorm rotary validation passed")


if __name__ == "__main__":
    main()
