#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
from datasets import load_from_disk
from evaluation_wrapper import VLMModel
import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling_qwen3_vl
from transformers import StaticCache
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from cache import ASStaticCache
# from capture import init, forward

# modeling_qwen3_vl.Qwen3VLModel.forward = forward
# modeling_qwen3_vl.Qwen3VLModel.__init__ = init

# from module_final import Qwen3VLModel_, Qwen3VLModel_Forward
# modeling_qwen3_vl.Qwen3VLModel.__init__ = Qwen3VLModel_
# modeling_qwen3_vl.Qwen3VLModel.forward = Qwen3VLModel_Forward

torch._dynamo.config.disable = True

if os.environ.get("AICAS_CUDA_LAUNCH_BLOCKING", "").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class CUDAGraphModel:
    def __init__(self, model, processor, image, question):
        self.model = model
        self.processor = processor
        device = "cuda:0"
        self.device = device
        self.max_new_tokens = 128
        self.prefill_len = None
        self.is_first = False
        self.dtype = torch.float16
        self.max_cache_len = 900

        self.max_batch_size = 1
        self.past_key_values = ASStaticCache(
            config=self.model.config,
            max_batch_size=self.max_batch_size,
            max_cache_len=1145,
            device=self.device,
            dtype=self.dtype
        )
        # self.test_generate(image, question)
        
    def get_model(self):
        return self.model

    def test_generate(self, image, question):
        self.past_key_values.reset()
        # self.init_before_first_run()
        import time
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        # self.past_key_values.reset()
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        self.input_len = inputs.input_ids.shape[1]
        start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            use_cache=True
        )
        print(f"Generate Cost: {float(time.perf_counter()-start):.3f}")

        return outputs

def main():
    model_path = "./Qwen3-VL-2B-Instruct"
    model = VLMModel(model_path)
    dataset_path = "./data"
    dataset = load_from_disk(dataset_path)
    first_sample = dataset[0]
    image = first_sample["image"]
    question = first_sample["question"]
    question_id = first_sample.get("question_id", 0)
    print(f"\n问题ID: {question_id}")
    print(f"问题: {question}")

    mod = CUDAGraphModel(model.model, model.processor, image, question)

    res = mod.test_generate(dataset[0]['image'], dataset[0]['question'])
    generated_ids = res[0][mod.input_len:]
    text = model.processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(text)
    res = mod.test_generate(dataset[1]['image'], dataset[1]['question'])
    generated_ids = res[0][mod.input_len:]
    text = model.processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(text)
    res = mod.test_generate(dataset[2]['image'], dataset[2]['question'])
    generated_ids = res[0][mod.input_len:]
    text = model.processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    print(text)
    # res = mod.test_generate(dataset[3]['image'], dataset[3]['question'])
    # generated_ids = res[0][mod.input_len:]
    # text = model.processor.tokenizer.decode(
    #     generated_ids,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )
    # print(text)
    # res = mod.test_generate(dataset[4]['image'], dataset[4]['question'])
    # generated_ids = res[0][mod.input_len:]
    # text = model.processor.tokenizer.decode(
    #     generated_ids,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )
    # print(text)
    # res = mod.test_generate(dataset[1]['image'], dataset[1]['question'])
    # generated_ids = res[0][mod.input_len:]
    # text = model.processor.tokenizer.decode(
    #     generated_ids,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )
    # print(text)
    # res = mod.test_generate(dataset[2]['image'], dataset[2]['question'])
    # generated_ids = res[0][mod.input_len:]
    # text = model.processor.tokenizer.decode(
    #     generated_ids,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )
    # print(text)

if __name__ == "__main__":
    main()
