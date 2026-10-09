"""FastProcessor — strip the slow tokenizer chat-template path on the hot path.

The HF apply_chat_template renders Jinja per call. We know the conversation
shape (single user turn with one image then text), so we skip Jinja and emit
the token IDs directly.
"""
from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature

from .constants import (
    CHAT_PREFIX, CHAT_SUFFIX_BEFORE_Q, CHAT_SUFFIX_AFTER_Q, IMAGE_PAD_ID,
)


def make_fast_apply_chat_template(processor):
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    spatial_merge_size = 2
    prefix_len = len(CHAT_PREFIX)

    def fast_apply_chat_template(
        conversation, chat_template=None, tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt", **kwargs
    ):
        images = []
        question = ""
        for msg in conversation:
            for content in msg.get("content", []):
                if content.get("type") == "image":
                    images.append(content["image"])
                elif content.get("type") == "text":
                    question = content["text"]

        if images:
            image_inputs = image_processor(images, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"]
            grid_thw = image_inputs["image_grid_thw"]
            n_img_tokens = int(
                (grid_thw[0, 1] // spatial_merge_size)
                * (grid_thw[0, 2] // spatial_merge_size))
            image_pad_tokens = [IMAGE_PAD_ID] * n_img_tokens
        else:
            pixel_values = None
            grid_thw = None
            image_pad_tokens = []

        q_tokens = tokenizer.encode(question, add_special_tokens=False)
        ids = (CHAT_PREFIX + image_pad_tokens + CHAT_SUFFIX_BEFORE_Q
               + q_tokens + CHAT_SUFFIX_AFTER_Q)
        input_ids = torch.tensor([ids], dtype=torch.long)

        n_img = len(image_pad_tokens)
        mm_ids = torch.zeros(1, len(ids), dtype=torch.long)
        mm_ids[0, prefix_len:prefix_len + n_img] = 1
        attention_mask = torch.ones(1, len(ids), dtype=torch.long)

        result = BatchFeature(data={
            "input_ids": input_ids, "attention_mask": attention_mask,
            "mm_token_type_ids": mm_ids,
        })
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = grid_thw
        return result

    return fast_apply_chat_template
