from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ref_attn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    q_len = int(query.shape[2])
    kv_len = int(key.shape[2])
    outs = []
    scale = 1.0 / math.sqrt(float(query.shape[-1]))
    for qi in range(q_len):
        end = kv_len - q_len + qi + 1
        q = query[:, :, qi: qi + 1, :].float()
        k = key[:, :, :end, :].float()
        v = value[:, :, :end, :].float()
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        prob = torch.softmax(scores, dim=-1)
        out = torch.matmul(prob, v).transpose(1, 2)
        outs.append(out)
    return torch.cat(outs, dim=1).to(query.dtype).contiguous()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ.setdefault("AICAS_FLASHDECODE_BF16_ATTENTION", "1")

    from my_kernel.flashDecode.flashdecode_runtime import _load_flashdecode_ext

    ext = _load_flashdecode_ext()
    device = torch.device("cuda")
    runner = ext.FlashDecodeRunner(1, 8, 2, 2112, torch.cuda.current_device())

    for dtype in (torch.float16, torch.bfloat16):
        for q_len in (1, 2, 3, 4):
            for kv_len in (q_len, 33, 257, 1024):
                torch.manual_seed(q_len * 100000 + kv_len)
                q = torch.randn((1, 16, q_len, 128), device=device, dtype=dtype) * 0.1
                k = torch.randn((1, 8, kv_len, 128), device=device, dtype=dtype) * 0.1
                v = torch.randn((1, 8, kv_len, 128), device=device, dtype=dtype)
                out = torch.empty((1, q_len, 16, 128), device=device, dtype=dtype)
                runner.run_decode_multi_into(q.contiguous(), k.contiguous(), v.contiguous(), out)
                ref = _ref_attn(q, k, v)
                diff = (out.float() - ref.float()).abs()
                max_abs = float(diff.max().item())
                mean_abs = float(diff.mean().item())
                print(
                    f"dtype={dtype} q={q_len} kv={kv_len} "
                    f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
                )
                tol = 0.02 if dtype is torch.bfloat16 else 0.01
                if max_abs > tol:
                    raise AssertionError(
                        f"flashdecode multi mismatch: dtype={dtype} q={q_len} kv={kv_len} max_abs={max_abs}"
                    )
    print("flashdecode multi validation passed")


if __name__ == "__main__":
    main()
