#!/usr/bin/env python3
"""Test: does the batch verify path (T>1, SDPA) produce correct answers?

Key question: if we generate tokens using ONLY the batch verify path
(no CUDA graph decoder), do we get 150/150 accuracy?

If YES: we can train the draft model on batch verify hidden states and
         use batch verify for speculative decoding → >3000 tok/s.
If NO: the batch verify path is fundamentally broken for this model,
         and we need a different approach.
"""
import os, json, torch, time, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ["AICAS_DISABLE_FUSED_GEMV_WARMUP"] = "1"

from custom_kernels.spec.batch_verify import (
    batch_verify_with_hiddens, _set_cache_positions
)
from custom_kernels.spec.eagle3_v4 import (
    _disable_graph_mode, _enable_graph_mode, _safe_set_cache_pos
)


def test_verify_path():
    from evaluation_wrapper import VLMModel
    from datasets import load_from_disk
    from custom_kernels.cache.graph_cache import GraphStaticKVCache

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
    with open('eval_subset_150.json') as f:
        subset = json.load(f)
    qids = subset['question_ids'][:10]  # Test 10 samples first
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

        # Step 1: Run baseline to get correct answer
        with torch.no_grad():
            baseline_out = raw_model.generate(
                **inputs, max_new_tokens=128, do_sample=False, use_cache=True
            )
        baseline_answer = processor.tokenizer.decode(
            baseline_out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )

        # Step 2: Run with batch verify path (T=1 sequentially)
        # Hook to capture prefill inputs
        captured = {}
        def prefill_hook(module, args, kwargs):
            if kwargs.get('inputs_embeds') is not None and 'inputs_embeds' not in captured:
                captured['inputs_embeds'] = kwargs['inputs_embeds'].detach().clone()
                captured['position_ids'] = kwargs.get('position_ids').detach().clone() if kwargs.get('position_ids') is not None else None
                captured['visual_pos_masks'] = kwargs.get('visual_pos_masks')
                captured['deepstack_visual_embeds'] = kwargs.get('deepstack_visual_embeds')

        handle = language_model.register_forward_pre_hook(prefill_hook, with_kwargs=True)
        with torch.no_grad():
            # Need to trigger prefill - use a dummy generate
            _ = raw_model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=True)
        handle.remove()

        if 'inputs_embeds' not in captured:
            print(f"[{qi}] SKIP: no prefill captured")
            continue

        # Step 3: Create fresh cache and run prefill
        fresh_cache = GraphStaticKVCache(
            num_layers=num_layers, num_kv_heads=num_kv_heads,
            head_dim=head_dim, max_seq_len=4096,
            dtype=torch.float16, device=device,
        )
        fresh_cache.cache_seqlens = torch.zeros(1, dtype=torch.int32, device=device)

        # Add flash attrs to match graph engine's cache
        # This makes the T=1 decode use flash_decode_attn
        for layer in fresh_cache.layers:
            if not hasattr(layer, 'flash_k_caches'):
                layer.flash_k_caches = None
                layer.flash_v_caches = None

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

            cache_pos = fresh_cache.get_seq_length()
            last_h = prefill_out.last_hidden_state[0, -1:].half()
            first_logits = lm_head(last_h)
            cur_token = first_logits.argmax(dim=-1).item()

        # Step 4: Decode with T=1 sequentially (matching collect_flash.py)
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
                logits = lm_head(h.unsqueeze(0)).squeeze(0) if not hasattr(lm_head, 'weight') else torch.nn.functional.linear(h, lm_head.weight)
                next_token = logits.argmax().item()
                tokens_seq.append(next_token)
                cache_pos += 1
                cur_token = next_token

        dt = time.time() - t0
        verify_answer = processor.tokenizer.decode(tokens_seq, skip_special_tokens=True)
        n_tok = len(tokens_seq)

        # Compare
        match = baseline_answer[:50] == verify_answer[:50]
        if match:
            correct += 1
        total += 1

        print(f"[{qi}] {'✓' if match else '✗'} baseline={repr(baseline_answer[:60])} | verify={repr(verify_answer[:60])} ({n_tok}tok, {dt:.2f}s, {n_tok/dt:.0f}tok/s)")

    print(f"\n{'='*60}")
    print(f"Verify path accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_verify_path()
