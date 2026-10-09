from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bench(label: str, fn, warmup: int, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    us = float(start.elapsed_time(end)) * 1000.0 / max(1, iters)
    print(f"{label}: {us:.3f} us")
    return us


def _expected_splits(kv_len: int, dynamic: bool) -> int:
    if dynamic:
        if kv_len <= 512:
            tile = kv_len
        elif kv_len <= 1024:
            tile = (kv_len + 1) // 2
        elif kv_len <= 2048:
            tile = (kv_len + 3) // 4
        else:
            tile = 512
    else:
        tile = 64 if kv_len <= 1024 else 128
    return max(1, (kv_len + max(1, tile) - 1) // max(1, tile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-lens", default="257,512,768,1024,1536,2048")
    parser.add_argument("--q-len", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from my_kernel.flashDecode.flashdecode_runtime import _load_flashdecode_ext

    ext = _load_flashdecode_ext()
    device = torch.device("cuda")
    device_index = torch.cuda.current_device()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    q_len = int(args.q_len)
    max_s = max(int(x) for x in args.kv_lens.split(",") if x.strip())
    runner = ext.FlashDecodeRunner(1, 8, 2, max_s, device_index)

    for kv_len_s in args.kv_lens.split(","):
        kv_len_s = kv_len_s.strip()
        if not kv_len_s:
            continue
        kv_len = int(kv_len_s)
        torch.manual_seed(20260607 + kv_len * 17 + q_len)
        q = torch.randn((1, 16, q_len, 128), device=device, dtype=dtype) * 0.1
        k = torch.randn((1, 8, kv_len, 128), device=device, dtype=dtype) * 0.1
        v = torch.randn((1, 8, kv_len, 128), device=device, dtype=dtype)
        out = torch.empty((1, q_len, 16, 128), device=device, dtype=dtype)

        for dynamic in (False, True):
            os.environ["AICAS_FD_MMA_DYNAMIC_SPLIT"] = "1" if dynamic else "0"
            label = (
                f"dynamic={int(dynamic)} q={q_len} kv={kv_len} "
                f"expected_splits={_expected_splits(kv_len, dynamic)}"
            )
            if q_len == 1:
                _bench(
                    label,
                    lambda: runner.run_decode_into(q.contiguous(), k.contiguous(), v.contiguous(), out),
                    int(args.warmup),
                    int(args.iters),
                )
            else:
                _bench(
                    label,
                    lambda: runner.run_decode_multi_into(q.contiguous(), k.contiguous(), v.contiguous(), out),
                    int(args.warmup),
                    int(args.iters),
                )


if __name__ == "__main__":
    main()
