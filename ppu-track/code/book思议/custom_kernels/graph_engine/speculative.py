"""自投机解码 (Self-Speculative Decoding)：draft 跳层 + target 批量验证。"""

import os

import torch

from .decoder import (
    _get_or_create_decoder,
    _select_next_token_from_hidden,
    _select_tokens_from_hidden_sequence,
)
from .kv_cache import (
    _copy_cache_prefix,
    _copy_cache_range,
    _drop_flash_cache_attrs,
    _get_or_create_self_spec_cache,
    _parse_skip_layers,
    _set_cache_len,
)
from .utils import _temporary_env


def _self_spec_generate_from_prefill(model, decoder, full_cache, input_ids,
                                     prefill_len, rope_delta, first_token,
                                     max_new_tokens, eos_token_ids):
    """运行时自投机解码：draft跳层，target完整批量验证。

    这是显式开关的实验路径，不缓存答案token；只复用/维护KV特征。
    """
    block_size = max(1, int(os.environ.get("AICAS_SELF_SPEC_BLOCK", "8")))
    skip_layers = _parse_skip_layers(
        os.environ.get("AICAS_SELF_SPEC_SKIP_LAYERS", os.environ.get("AICAS_DECODE_SKIP_LAYERS", "-1")),
        decoder.num_layers,
    )
    non_skipped_layers = [i for i in range(decoder.num_layers) if i not in skip_layers]
    skip_spec = ",".join(str(i) for i in sorted(skip_layers))

    draft_cache = _get_or_create_self_spec_cache(
        num_layers=decoder.num_layers,
        num_kv_heads=decoder.num_kv_heads,
        head_dim=decoder.head_dim,
        max_seq_len=decoder.max_seq_len,
        dtype=torch.float16,
        device=input_ids.device,
    )
    _drop_flash_cache_attrs(draft_cache)
    _drop_flash_cache_attrs(full_cache)
    _copy_cache_prefix(full_cache, draft_cache, prefill_len)
    _set_cache_len(full_cache, prefill_len)

    token_buf = torch.empty(max_new_tokens, dtype=torch.long, device=input_ids.device)
    token_buf[0] = first_token[0, 0]
    num_generated = 1
    cache_len = prefill_len
    last_token = first_token.contiguous()
    eos_set = set(int(x) for x in eos_token_ids)

    accepted_total = 0
    drafted_total = 0
    old_skip_final_norm = getattr(model.model.language_model, "_skip_final_norm", False)
    model.model.language_model._skip_final_norm = False
    try:
        while num_generated < max_new_tokens:
            remaining = max_new_tokens - num_generated
            if remaining <= 0:
                break

            if remaining == 1:
                position_ids = torch.full([1, 1, 1], cache_len + rope_delta,
                                          dtype=torch.long, device=input_ids.device)
                outputs = model.model.language_model(
                    input_ids=last_token,
                    position_ids=position_ids,
                    past_key_values=full_cache,
                    use_cache=True,
                )
                next_token = _select_next_token_from_hidden(
                    model, decoder, outputs.last_hidden_state[:, -1, :]
                )
                token_buf[num_generated] = next_token[0, 0]
                num_generated += 1
                cache_len += 1
                last_token = next_token.contiguous()
                _set_cache_len(full_cache, cache_len)
                if next_token[0, 0].item() in eos_set:
                    break
                continue

            gamma = min(block_size, remaining - 1)
            _set_cache_len(draft_cache, cache_len, non_skipped_layers)
            _set_cache_len(full_cache, cache_len)
            draft_tokens = []
            tok = last_token
            with _temporary_env("AICAS_DECODE_SKIP_LAYERS", skip_spec):
                for _ in range(gamma):
                    pos = cache_len + len(draft_tokens) + rope_delta
                    position_ids = torch.full(
                        [1, 1, 1], pos, dtype=torch.long, device=input_ids.device
                    )
                    outputs = model.model.language_model(
                        input_ids=tok,
                        position_ids=position_ids,
                        past_key_values=draft_cache,
                        use_cache=True,
                    )
                    tok = _select_next_token_from_hidden(
                        model, decoder, outputs.last_hidden_state[:, -1, :]
                    )
                    draft_tokens.append(int(tok[0, 0].item()))
                    if draft_tokens[-1] in eos_set:
                        break

            if not draft_tokens:
                break

            draft = torch.tensor(draft_tokens, dtype=torch.long, device=input_ids.device)
            verify_input = torch.empty(1, draft.numel() + 1, dtype=torch.long, device=input_ids.device)
            verify_input[0, 0] = last_token[0, 0]
            verify_input[0, 1:] = draft

            position_ids = torch.arange(
                cache_len + rope_delta,
                cache_len + rope_delta + draft.numel() + 1,
                dtype=torch.long,
                device=input_ids.device,
            ).view(1, 1, -1)
            with _temporary_env("AICAS_DECODE_SKIP_LAYERS", None):
                outputs = model.model.language_model(
                    input_ids=verify_input,
                    position_ids=position_ids,
                    past_key_values=full_cache,
                    use_cache=True,
                )
            predicted = _select_tokens_from_hidden_sequence(model, decoder, outputs.last_hidden_state)

            matches = (predicted[:-1] == draft)
            match_len = int(matches.cumprod(dim=0).sum().item()) if matches.numel() else 0
            drafted_total += int(draft.numel())

            if match_len == draft.numel():
                take = min(int(draft.numel()), max_new_tokens - num_generated)
                if take > 0:
                    token_buf[num_generated:num_generated + take].copy_(draft[:take], non_blocking=True)
                    num_generated += take
                    accepted_total += take
                next_token = predicted[-1].view(1, 1)
                if num_generated < max_new_tokens:
                    token_buf[num_generated] = next_token[0, 0]
                    num_generated += 1
                    accepted_total += 1
                old_cache_len = cache_len
                cache_len += int(draft.numel()) + 1
                last_token = next_token.contiguous()
                _set_cache_len(full_cache, cache_len)
                _copy_cache_range(full_cache, draft_cache, old_cache_len + int(draft.numel()), cache_len, non_skipped_layers)
                _set_cache_len(draft_cache, cache_len, non_skipped_layers)
                if next_token[0, 0].item() in eos_set:
                    break
                continue

            accepted_prefix = draft[:match_len]
            next_token = predicted[match_len].view(1, 1)
            take = min(int(accepted_prefix.numel()) + 1, max_new_tokens - num_generated)
            if accepted_prefix.numel() > 0 and take > 0:
                prefix_take = min(int(accepted_prefix.numel()), take)
                token_buf[num_generated:num_generated + prefix_take].copy_(
                    accepted_prefix[:prefix_take], non_blocking=True
                )
                num_generated += prefix_take
                accepted_total += prefix_take
            if num_generated < max_new_tokens:
                token_buf[num_generated] = next_token[0, 0]
                num_generated += 1
                accepted_total += 1

            cache_len += int(accepted_prefix.numel()) + 1
            _set_cache_len(full_cache, cache_len)
            _set_cache_len(draft_cache, cache_len, non_skipped_layers)
            last_token = next_token.contiguous()
            if next_token[0, 0].item() in eos_set:
                break
    finally:
        model.model.language_model._skip_final_norm = old_skip_final_norm

    if os.environ.get("AICAS_SELF_SPEC_LOG", "0") == "1" and drafted_total:
        print(
            f"[self_spec] block={block_size} skip={skip_spec} "
            f"accepted={accepted_total}/{drafted_total}"
        )

    return torch.cat([input_ids, token_buf[:num_generated].unsqueeze(0)], dim=-1)
