#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from collections import Counter
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from spec_decode.eagle3 import (
    Eagle3DraftModel,
    _parse_layer_indices,
    _select_eagle3_feature,
    save_eagle3_draft,
    shift_left,
)
from spec_decode.verifier import _temporary_model_attr
from utils import _effective_min_new_tokens


def _apply_safe_collection_env() -> None:
    """Use conservative generate paths while collecting long training caches."""
    os.environ.setdefault("AICAS_DISABLE_DECODE_CUDAGRAPH", "1")
    os.environ.setdefault("AICAS_FLASHDECODE", "0")
    os.environ.setdefault("AICAS_ENABLE_FLASHDECODE_ATTENTION", "0")
    os.environ.setdefault("AICAS_ENABLE_FLASHDECODE_FFN", "0")
    os.environ.setdefault("AICAS_DECODE_GRAPH_PREFILL_CACHE_REUSE", "0")
    os.environ.setdefault("AICAS_DECODE_FASTPATH_MAX_NEW_TOKENS", "0")
    os.environ.setdefault("AICAS_DECODE_CUDAGRAPH_MAX_NEW_TOKENS", "0")


def _prepare_inputs(processor, item, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": item["image"]},
            {"type": "text", "text": item["question"]},
        ],
    }]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)


def _target_forward_kwargs(processor, item, row, device):
    prompt_inputs = _prepare_inputs(processor, item, device)
    full_ids = row["input_ids"].to(device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = row["attention_mask"].to(device=device, dtype=torch.long).unsqueeze(0)
    kwargs = {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "output_hidden_states": True,
        "return_dict": True,
    }
    for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        value = getattr(prompt_inputs, key, None)
        if isinstance(value, torch.Tensor):
            kwargs[key] = value
    mm_token_type_ids = getattr(prompt_inputs, "mm_token_type_ids", None)
    if isinstance(mm_token_type_ids, torch.Tensor):
        mm_full = torch.zeros_like(full_ids, dtype=mm_token_type_ids.dtype)
        mm_full[:, : mm_token_type_ids.shape[1]] = mm_token_type_ids
        kwargs["mm_token_type_ids"] = mm_full
    return kwargs


def _build_draft_vocab_ids(rows, vocab_size: int, max_size: int) -> torch.Tensor | None:
    max_size = int(max_size)
    if max_size <= 0 or max_size >= int(vocab_size):
        return None
    counts = Counter()
    for row in rows:
        input_ids = row["input_ids"].reshape(-1).tolist()
        loss_mask = row["loss_mask"].reshape(-1).tolist()
        for token_id, use_token in zip(input_ids, loss_mask):
            if int(use_token) and 0 <= int(token_id) < int(vocab_size):
                counts[int(token_id)] += 1
    if not counts:
        raise RuntimeError("cannot build reduced draft vocab from empty training-token counts")
    ordered = [token_id for token_id, _ in counts.most_common(max_size)]
    if len(ordered) < max_size:
        present = set(ordered)
        for token_id in range(int(vocab_size)):
            if token_id not in present:
                ordered.append(token_id)
                if len(ordered) >= max_size:
                    break
    return torch.tensor(sorted(ordered[:max_size]), dtype=torch.long)


def _save_sequence_cache(args, rows, *, min_new: int, generator: str | None = None, completed: bool = False) -> None:
    os.makedirs(os.path.dirname(args.train_cache), exist_ok=True)
    payload = {
        "mode": "official_sequences_v1",
        "rows": rows,
        "train_length": int(args.train_length),
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(min_new),
        "completed": bool(completed),
    }
    if generator is not None:
        payload["generator"] = generator
    torch.save(payload, args.train_cache)


def _load_sequence_cache_if_compatible(args) -> tuple[list[dict], dict] | None:
    if not os.path.exists(args.train_cache):
        return None
    cache = torch.load(args.train_cache, map_location="cpu", weights_only=True)
    if cache.get("mode") != "official_sequences_v1":
        raise RuntimeError("cached mode mismatch; use a fresh --train-cache")
    if int(cache.get("train_length", args.train_length)) != int(args.train_length):
        raise RuntimeError("cached train_length mismatch; use a fresh --train-cache")
    if int(cache.get("max_new_tokens", args.max_new_tokens)) != int(args.max_new_tokens):
        raise RuntimeError("cached max_new_tokens mismatch; use a fresh --train-cache")
    requested_min_new = int(_effective_min_new_tokens(args.max_new_tokens, args.min_new_tokens))
    cached_min_new = int(cache.get("min_new_tokens", requested_min_new))
    if cached_min_new != requested_min_new:
        raise RuntimeError("cached min_new_tokens mismatch; use a fresh --train-cache")
    return list(cache.get("rows", [])), cache


def _draft_vocab_token_coverage(rows, draft_vocab_ids: torch.Tensor | None) -> tuple[int, int]:
    if not isinstance(draft_vocab_ids, torch.Tensor) or draft_vocab_ids.numel() <= 0:
        return 0, 0
    vocab = set(int(x) for x in draft_vocab_ids.reshape(-1).tolist())
    covered = 0
    total = 0
    for row in rows:
        input_ids = row["input_ids"].reshape(-1).tolist()
        loss_mask = row["loss_mask"].reshape(-1).tolist()
        for token_id, use_token in zip(input_ids, loss_mask):
            if not int(use_token):
                continue
            total += 1
            covered += int(int(token_id) in vocab)
    return covered, total


def _target_logits_for_draft_vocab(
    target_logits: torch.Tensor,
    draft_vocab_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    target_argmax = target_logits.argmax(dim=-1)
    if not isinstance(draft_vocab_ids, torch.Tensor) or draft_vocab_ids.numel() <= 0:
        return target_logits, None, target_argmax
    vocab_ids = draft_vocab_ids.to(device=target_logits.device, dtype=torch.long)
    target_in_vocab = torch.isin(target_argmax, vocab_ids)
    if bool(torch.all(vocab_ids[:-1] <= vocab_ids[1:]).item()):
        local_target = torch.searchsorted(vocab_ids, target_argmax).clamp_max(vocab_ids.numel() - 1)
    else:
        flat_target = target_argmax.reshape(-1)
        local_flat = torch.zeros_like(flat_target)
        for idx, token in enumerate(flat_target.tolist()):
            hit = torch.nonzero(vocab_ids.eq(int(token)), as_tuple=False)
            if hit.numel() > 0:
                local_flat[idx] = int(hit[0, 0].item())
        local_target = local_flat.reshape_as(target_argmax)
    return target_logits.index_select(-1, vocab_ids), target_in_vocab, local_target


def collect_sequence_cache(args, model, processor, ds, device):
    existing = _load_sequence_cache_if_compatible(args)
    rows = existing[0] if existing is not None and bool(args.resume_cache) else []
    seen = {int(row["dataset_index"]) for row in rows}
    n = min(args.num_samples, len(ds))
    min_new = _effective_min_new_tokens(args.max_new_tokens, args.min_new_tokens)
    pbar = tqdm(range(n), desc="Collect EAGLE3 sequences", initial=min(len(seen), n))
    for idx in pbar:
        if int(idx) in seen:
            continue
        item = ds[idx]
        inputs = _prepare_inputs(processor, item, device)
        input_len = int(inputs.input_ids.shape[1])
        if args.max_input_tokens > 0 and input_len > int(args.max_input_tokens):
            continue
        gen_kwargs = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        if min_new > 0:
            gen_kwargs["min_new_tokens"] = int(min_new)
        with torch.no_grad():
            gen = model.generate(**gen_kwargs)
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        full_ids = seq[:, : input_len + args.max_new_tokens].to(dtype=torch.long)
        if int(full_ids.shape[1]) <= input_len + args.train_length:
            continue
        attention_mask = torch.ones_like(full_ids)
        loss_mask = torch.zeros_like(full_ids)
        loss_mask[:, input_len:] = 1
        rows.append({
            "dataset_index": int(idx),
            "input_ids": full_ids[0].detach().cpu(),
            "attention_mask": attention_mask[0].detach().cpu(),
            "loss_mask": loss_mask[0].detach().cpu(),
        })
        seen.add(int(idx))
        if args.cache_save_every > 0 and len(rows) % int(args.cache_save_every) == 0:
            _save_sequence_cache(args, rows, min_new=int(min_new))

    _save_sequence_cache(args, rows, min_new=int(min_new), completed=True)
    return rows


def collect_sequence_cache_with_generator(args, generator_model, forward_model, processor, ds, device):
    existing = _load_sequence_cache_if_compatible(args)
    rows = existing[0] if existing is not None and bool(args.resume_cache) else []
    seen = {int(row["dataset_index"]) for row in rows}
    n = min(args.num_samples, len(ds))
    min_new = _effective_min_new_tokens(args.max_new_tokens, args.min_new_tokens)
    pbar = tqdm(range(n), desc="Collect EAGLE3 sequences", initial=min(len(seen), n))
    for idx in pbar:
        if int(idx) in seen:
            continue
        item = ds[idx]
        inputs = _prepare_inputs(processor, item, device)
        input_len = int(inputs.input_ids.shape[1])
        if args.max_input_tokens > 0 and input_len > int(args.max_input_tokens):
            continue
        gen_kwargs = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        if min_new > 0:
            gen_kwargs["min_new_tokens"] = int(min_new)
        with torch.no_grad():
            gen = generator_model.generate(**gen_kwargs)
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        full_ids = seq[:, : input_len + args.max_new_tokens].to(dtype=torch.long)
        if int(full_ids.shape[1]) <= input_len + args.train_length:
            continue
        attention_mask = torch.ones_like(full_ids)
        loss_mask = torch.zeros_like(full_ids)
        loss_mask[:, input_len:] = 1
        rows.append({
            "dataset_index": int(idx),
            "input_ids": full_ids[0].detach().cpu(),
            "attention_mask": attention_mask[0].detach().cpu(),
            "loss_mask": loss_mask[0].detach().cpu(),
        })
        seen.add(int(idx))
        if args.cache_save_every > 0 and len(rows) % int(args.cache_save_every) == 0:
            _save_sequence_cache(
                args,
                rows,
                min_new=int(min_new),
                generator="vlm_wrapper_current_policy",
            )

    _save_sequence_cache(
        args,
        rows,
        min_new=int(min_new),
        generator="vlm_wrapper_current_policy",
        completed=True,
    )
    return rows


def official_loss_for_row(
    draft,
    target_model,
    processor,
    ds,
    row,
    device,
    dtype,
    layer_indices,
    length,
    loss_weights,
    loss_mode: str,
    label_smoothing: float,
):
    item = ds[int(row["dataset_index"])]
    forward_kwargs = _target_forward_kwargs(processor, item, row, device)
    input_ids = forward_kwargs["input_ids"]
    attention_mask = forward_kwargs["attention_mask"]
    loss_mask = row["loss_mask"].to(device=device, dtype=dtype).unsqueeze(0)

    with torch.no_grad(), _temporary_model_attr(target_model, "_aicas_disable_prefill_compile", True):
        outputs = target_model(**forward_kwargs)
        feature = _select_eagle3_feature(outputs.hidden_states, layer_indices)
        target = outputs.logits

    draft_input_ids = shift_left(input_ids)
    target = shift_left(target)
    logits_list = draft.training_forward(feature, draft_input_ids, attention_mask, int(length))

    losses = []
    acces = []
    cur_loss_mask = loss_mask
    cur_target = target
    for idx, logits in enumerate(logits_list):
        position_mask = cur_loss_mask.unsqueeze(-1)
        cur_target_for_loss, target_in_vocab, hard_target = _target_logits_for_draft_vocab(
            cur_target,
            getattr(draft, "draft_vocab_ids", None),
        )
        if target_in_vocab is not None:
            position_mask = position_mask * target_in_vocab.unsqueeze(-1).to(dtype=position_mask.dtype)
        metric_mask = position_mask.squeeze(-1)
        out_logp = torch.log_softmax(logits.float(), dim=-1)
        if loss_mode == "kl":
            target_p = torch.softmax(cur_target_for_loss.float(), dim=-1).detach()
            loss = -(position_mask * target_p * out_logp).sum(dim=-1).sum() / position_mask.sum().clamp_min(1.0)
            metric_target = target_p.argmax(dim=-1)
        elif loss_mode == "ce_kl":
            target_p = torch.softmax(cur_target_for_loss.float(), dim=-1).detach()
            hard_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                hard_target.reshape(-1),
                reduction="none",
                label_smoothing=float(label_smoothing),
            ).view_as(metric_mask)
            hard_loss = (hard_loss * metric_mask).sum() / metric_mask.sum().clamp_min(1.0)
            soft_loss = -(position_mask * target_p * out_logp).sum(dim=-1).sum() / position_mask.sum().clamp_min(1.0)
            loss = hard_loss + 0.25 * soft_loss
            metric_target = hard_target
        else:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                hard_target.reshape(-1),
                reduction="none",
                label_smoothing=float(label_smoothing),
            ).view_as(metric_mask)
            loss = (loss * metric_mask).sum() / metric_mask.sum().clamp_min(1.0)
            metric_target = hard_target
        losses.append(loss)
        with torch.no_grad():
            correct = logits.argmax(dim=-1).eq(metric_target).to(metric_mask.dtype)
            acces.append(float((correct * metric_mask).sum().item() / metric_mask.sum().clamp_min(1.0).item()))
        if idx != len(logits_list) - 1:
            draft_input_ids = shift_left(draft_input_ids)
            cur_target = shift_left(cur_target)
            cur_loss_mask = shift_left(cur_loss_mask)

    total = sum(loss_weights[i] * losses[i] for i in range(len(losses)))
    return total, losses, acces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="../Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-path", default="../data")
    parser.add_argument("--output", default="/root/chusai_base/spec_decode/eagle3_official_draft.pt")
    parser.add_argument("--train-cache", default="/root/chusai_base/spec_decode/eagle3_vlm_policy_5000_len4_l0m14f28_cache.pt")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=None)
    parser.add_argument("--train-length", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layer-indices", default="0,1,2")
    parser.add_argument(
        "--loss-mode",
        choices=("kl", "ce", "ce_kl"),
        default=os.getenv("AICAS_EAGLE3_TRAIN_LOSS", "kl"),
        help="Training objective. kl matches the official EAGLE-3 distillation path; ce/ce_kl are A/B options for argmax acceptance.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=float(os.getenv("AICAS_EAGLE3_LABEL_SMOOTHING", "0.0")),
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=int(os.getenv("AICAS_EAGLE3_DRAFT_VOCAB_SIZE", "0")),
        help="Use a reduced draft vocabulary of this size. 0 keeps the full target vocabulary.",
    )
    parser.add_argument(
        "--collect-with-vlm-wrapper",
        action="store_true",
        help="Collect greedy sequences with evaluation_wrapper.VLMModel so training cache matches benchmark policy.",
    )
    parser.add_argument(
        "--unsafe-collect-fastpaths",
        action="store_true",
        help="Keep decode graph/flashdecode fastpaths on during training-cache collection. Faster but less stable for long samples.",
    )
    parser.add_argument("--cache-save-every", type=int, default=50)
    parser.add_argument("--resume-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help="Skip samples whose tokenized prompt exceeds this length. 0 keeps all samples.",
    )
    args = parser.parse_args()

    if int(args.batch_size) != 1:
        raise RuntimeError("Qwen3VL EAGLE3 official adapter currently trains with --batch-size 1")

    device = "cuda:0"
    dtype = torch.bfloat16
    generator_model = None
    if args.collect_with_vlm_wrapper:
        if not args.unsafe_collect_fastpaths:
            _apply_safe_collection_env()
        os.environ["AICAS_SPEC_DECODE"] = "0"
        from evaluation_wrapper import VLMModel

        wrapper = VLMModel(args.model_path)
        model = wrapper.model
        processor = wrapper.processor
        device = wrapper.device
        generator_model = wrapper.model
    else:
        model = AutoModelForImageTextToText.from_pretrained(args.model_path, dtype=dtype).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(args.model_path)
    for param in model.parameters():
        param.requires_grad_(False)
    ds = load_from_disk(args.dataset_path)

    n_hidden_states = int(model.config.text_config.num_hidden_layers) + 1
    layer_indices = _parse_layer_indices(args.layer_indices, n_hidden_states)
    train_length = int(args.train_length)

    existing_cache = _load_sequence_cache_if_compatible(args)
    if existing_cache is not None and bool(existing_cache[1].get("completed", False)):
        rows = existing_cache[0]
    else:
        if generator_model is not None:
            rows = collect_sequence_cache_with_generator(args, generator_model, model, processor, ds, device)
        else:
            rows = collect_sequence_cache(args, model, processor, ds, device)

    if not rows:
        raise RuntimeError("no EAGLE3 training rows collected")
    print(f"Training rows: {len(rows)}, layers={layer_indices}, length={train_length}")
    draft_vocab_ids = _build_draft_vocab_ids(
        rows,
        int(model.config.text_config.vocab_size),
        int(args.draft_vocab_size),
    )
    if draft_vocab_ids is not None:
        print(f"Reduced draft vocab: {int(draft_vocab_ids.numel())} / {int(model.config.text_config.vocab_size)}")
        covered, total = _draft_vocab_token_coverage(rows, draft_vocab_ids)
        if total > 0:
            print(f"Reduced draft vocab token coverage: {covered / total * 100:.2f}% ({covered}/{total})")

    text_model = getattr(getattr(model, "model", None), "language_model", None)
    embed_tokens = getattr(text_model, "embed_tokens", None)
    embed_weight = getattr(embed_tokens, "weight", None)
    draft = Eagle3DraftModel(
        model.config.text_config,
        train_length,
        lm_head_weight=model.lm_head.weight.data,
        embed_weight=embed_weight.data if isinstance(embed_weight, torch.Tensor) else None,
        draft_vocab_ids=draft_vocab_ids,
    ).to(device=device, dtype=dtype)
    draft.train()

    optimizer = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=args.lr)
    loss_weights = [0.8 ** i for i in range(train_length)]

    for epoch in range(args.epochs):
        perm = torch.randperm(len(rows)).tolist()
        total = 0.0
        pbar = tqdm(perm, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for row_idx in pbar:
            loss, losses, acces = official_loss_for_row(
                draft,
                model,
                processor,
                ds,
                rows[int(row_idx)],
                device,
                dtype,
                layer_indices,
                train_length,
                loss_weights,
                args.loss_mode,
                args.label_smoothing,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            pbar.set_postfix(loss=f"{float(loss.item()):.3f}", acc0=f"{acces[0]:.2f}")
        print(f"Epoch {epoch + 1}/{args.epochs} loss={total / max(len(rows), 1):.4f}")

    draft.eval()
    with torch.no_grad():
        totals = [0.0 for _ in range(train_length)]
        for row in rows:
            _, _, acces = official_loss_for_row(
                draft,
                model,
                processor,
                ds,
                row,
                device,
                dtype,
                layer_indices,
                train_length,
                loss_weights,
                args.loss_mode,
                args.label_smoothing,
            )
            for i, acc in enumerate(acces):
                totals[i] += acc
        for i in range(train_length):
            print(f"Position {i} masked accuracy: {totals[i] / max(len(rows), 1) * 100:.1f}%")

    save_eagle3_draft(draft, args.output, train_length, layer_indices)
    print(f"EAGLE3 draft saved to {args.output}")


if __name__ == "__main__":
    main()
