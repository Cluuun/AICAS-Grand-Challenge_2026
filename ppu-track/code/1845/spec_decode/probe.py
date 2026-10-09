"""
Phase 1 probe: measure whether block verification is faster than sequential decode.
Decision rule: continue only if T_verify_4 < 4 * T_decode_1.
"""
import argparse
import time
import torch
from datasets import load_from_disk
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/root/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", default="/root/data")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--draft-lens", default="2,4,8")
    args = parser.parse_args()

    draft_lens = [int(x) for x in args.draft_lens.split(",")]
    device = "cuda:0"
    dtype = torch.bfloat16

    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    ds = load_from_disk(args.dataset_path)
    n_samples = min(args.num_samples, len(ds))

    # Warmup
    print("Warmup...")
    item = ds[0]
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": item["image"]},
        {"type": "text", "text": item["question"]}]}]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                           temperature=0.0, use_cache=True)
    torch.cuda.synchronize()
    print("Warmup done.")

    seq_times = []   # per-token sequential decode time
    verify_times = {k: [] for k in draft_lens}

    for i in range(n_samples):
        item = ds[i]
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": item["image"]},
            {"type": "text", "text": item["question"]}]}]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(device)

        with torch.no_grad():
            # ---- Sequential decode measurement ----
            # Run prefill then N decode steps, measure per-token
            out = model(**inputs, use_cache=True)
            input_len = inputs.input_ids.shape[1]
            generated_ids = out.logits[:, -1:, :].argmax(dim=-1)

            # Measure next 20 sequential decode steps
            for _ in range(20):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(
                    input_ids=generated_ids,
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                seq_times.append((time.perf_counter() - t0) * 1000)
                generated_ids = out.logits[:, -1:, :].argmax(dim=-1)

            # ---- Block verification measurement ----
            for k in draft_lens:
                # Re-prefill to get fresh KV
                out2 = model(**inputs, use_cache=True)
                # Generate k sequential tokens to simulate a candidate chain
                candidates = []
                cur_out = out2
                for _ in range(k):
                    tok = cur_out.logits[:, -1:, :].argmax(dim=-1)
                    candidates.append(tok)
                    cur_out = model(input_ids=tok, past_key_values=cur_out.past_key_values, use_cache=True)

                # Now verify all k tokens in ONE forward pass
                # We feed all k tokens at once into the model
                cand_ids = torch.cat(candidates, dim=1)  # [1, k]

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(
                    input_ids=cand_ids,
                    past_key_values=out2.past_key_values,
                    use_cache=False,
                )
                torch.cuda.synchronize()
                verify_times[k].append((time.perf_counter() - t0) * 1000)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples}")

    # ---- Report ----
    avg_seq = sum(seq_times) / len(seq_times)
    print(f"\n=== Probe Results ({n_samples} samples) ===")
    print(f"Decode steps measured: {len(seq_times)}")
    print(f"T_decode_1: {avg_seq:.2f} ms (avg per-token sequential)")

    for k in draft_lens:
        avg_v = sum(verify_times[k]) / len(verify_times[k])
        threshold = k * avg_seq
        faster = avg_v < threshold
        print(f"T_verify_{k}: {avg_v:.2f} ms  |  {k}*T_decode_1 = {threshold:.2f} ms  |  {'✓ FASTER' if faster else '✗ SLOWER'}")

    avg_v4 = sum(verify_times[4]) / len(verify_times[4])
    if avg_v4 < 4 * avg_seq:
        print(f"\nPASS: T_verify_4 ({avg_v4:.1f}ms) < 4*T_decode_1 ({4*avg_seq:.1f}ms)")
        print("→ Speculative decoding IS promising. Proceed to Phase 2.")
    else:
        print(f"\nFAIL: T_verify_4 ({avg_v4:.1f}ms) >= 4*T_decode_1 ({4*avg_seq:.1f}ms)")
        print("→ STOP. Speculative decoding not promising without deeper kernel work.")


if __name__ == "__main__":
    main()
