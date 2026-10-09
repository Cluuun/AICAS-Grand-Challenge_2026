#!/usr/bin/env python3
"""Test verify path v2: FIX cache_seqlens after prefill.

The bug: GraphStaticKVCache.get_seq_length() returns cache_seqlens[0],
but cache_seqlens is NOT updated by the prefill forward.
After prefill, _pos_val is advanced to prefill_len, but cache_seqlens stays at 0.
This causes decode to overwrite position 0 → garbage output.

Fix: after prefill, set cache_seqlens = layers[0]._pos_val
"""
import os, sys, torch, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ["AICAS_DISABLE_FUSED_GEMV_WARMUP"] = "1"


def test_verify_fixed():
    from evaluation_wrapper import VLMModel
    from datasets import load_from_disk
    from custom_kernels.cache.graph_cache import GraphStaticKVCache
    from custom_kernels.spec.batch_verify import _set_cache_positions
    import torch.nn.functional as F

    vlm = VLMModel('Qwen3-VL-2B-Instruct')
    processor = vlm._processor
    raw_model = vlm._model
    language_model = raw_model.model.language_model
    lm_head = raw_model.lm_head
    device = raw_model.device

    text_config = getattr(raw_model.config, 'text_config', raw_model.config)
    num_layers = text_config.num_hidden_layers
    num_kv_heads = text_config.num_key_value_heads
    head_dim = getattr(text_config, 'head_dim', text_config.hidden_size // text_config.num_attention_heads)

    dataset = load_from_disk('data')
    import json
    with open('eval_subset_150.json') as f:
        subset = json.load(f)
    qids = subset['question_ids'][:10]
    qid_to_idx = {dataset[i]['question_id']: i for i in range(len(dataset))}

    correct = 0
    total = 0

    for qi, qid in enumerate(qids):
        idx = qid_to_idx[qid]
        item = dataset[idx]
        q = item['question']
        if isinstance(q, list): q = ' '.join(q)
        img = item['image']
        msgs = [{'role': 'user', 'content': [
            {'type': 'image', 'image': img},
            {'type': 'text', 'text': q},
        ]}]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt'
        ).to(device)

        # Step 1: Get baseline answer
        with torch.no_grad():
            baseline_out = raw_model.generate(
                **inputs, max_new_tokens=128, do_sample=False, use_cache=True)
        baseline_answer = processor.tokenizer.decode(
            baseline_out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Step 2: Hook prefill
        captured = {}
        def prefill_hook(module, args, kwargs):
            if kwargs.get('inputs_embeds') is not None and 'inputs_embeds' not in captured:
                captured['inputs_embeds'] = kwargs['inputs_embeds'].detach().clone()
                captured['position_ids'] = kwargs.get('position_ids').detach().clone() if kwargs.get('position_ids') is not None else None
                captured['visual_pos_masks'] = kwargs.get('visual_pos_masks')
                captured['deepstack_visual_embeds'] = kwargs.get('deepstack_visual_embeds')
        handle = language_model.register_forward_pre_hook(prefill_hook, with_kwargs=True)
        with torch.no_grad():
            _ = raw_model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
        handle.remove()

        if 'inputs_embeds' not in captured:
            print(f"[{qi}] SKIP")
            continue

        # Step 3: Fresh cache + prefill
        fresh_cache = GraphStaticKVCache(
            num_layers=num_layers, num_kv_heads=num_kv_heads,
            head_dim=head_dim, max_seq_len=4096,
            dtype=torch.float16, device=device,
        )
        fresh_cache.cache_seqlens = torch.zeros(1, dtype=torch.int32, device=device)

        with torch.inference_mode():
            prefill_out = language_model(
                input_ids=None,
                position_ids=captured['position_ids'],
                past_key_values=fresh_cache,
                inputs_embeds=captured['inputs_embeds'],
                visual_pos_masks=captured.get('visual_pos_masks'),
                deepstack_visual_embeds=captured.get('deepstack_visual_embeds'),
                use_cache=True,
                output_hidden_states=True,
            )

            # ★ FIX: Get actual cache position from _pos_val, NOT get_seq_length()
            actual_cache_pos = fresh_cache.layers[0]._pos_val
            fresh_cache.cache_seqlens.fill_(actual_cache_pos)
            cache_pos = actual_cache_pos

            print(f"  [{qi}] prefill_len={cache_pos}, get_seq_length={fresh_cache.get_seq_length()}")

            # First token
            last_h = prefill_out.last_hidden_state[0, -1:].half()
            first_logits = lm_head(last_h)
            cur_token = first_logits.argmax(dim=-1).item()

        # Step 4: Decode sequentially (T=1)
        tokens_seq = []
        t0 = time.time()
        with torch.inference_mode():
            for step in range(128):
                _set_cache_positions(fresh_cache, cache_pos)
                tok = torch.tensor([[cur_token]], device=device, dtype=torch.long)
                out = language_model(
                    input_ids=tok, past_key_values=fresh_cache,
                    use_cache=True, output_hidden_states=False,
                )
                h = out.last_hidden_state[0, 0].half()
                logits = F.linear(h, lm_head.weight) if hasattr(lm_head, 'weight') else lm_head(h.unsqueeze(0)).squeeze(0)
                next_token = logits.argmax().item()
                tokens_seq.append(next_token)
                cache_pos += 1
                cur_token = next_token

        dt = time.time() - t0
        verify_answer = processor.tokenizer.decode(tokens_seq, skip_special_tokens=True)
        n_tok = len(tokens_seq)

        match = baseline_answer[:50] == verify_answer[:50]
        if match:
            correct += 1
        total += 1

        status = '✓' if match else '✗'
        print(f"[{qi}] {status} base={repr(baseline_answer[:60])}")
        print(f"       verify={repr(verify_answer[:60])} ({n_tok}tok, {dt:.2f}s)")

    print(f"\n{'='*60}")
    print(f"Verify path accuracy: {correct}/{total} ({correct/max(total,1)*100:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_verify_fixed()
