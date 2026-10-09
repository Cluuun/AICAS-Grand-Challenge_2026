"""VLMModel：参赛选手主类 — 加载模型、应用优化、生成答案。"""

from typing import Dict
import os
import time

try:
    from PIL import Image
except ImportError:
    class Image:
        pass
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from . import _debug_enabled, _debug_sync, _print_exception, apply_optimizations, _install_vision_feature_cache, _install_debug_generate_wrapper
from .processor import _PrewarmProcessor


class VLMModel:
    """
    Participant optimization class - modify this to implement optimizations.

    Important Notes:
    1. Benchmark directly calls self.model.generate() for performance testing.
    2. Your optimizations should modify self.model or its operators via Monkey Patch.
    3. All optimizations are applied in __init__ by calling optimization methods.
    """

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self._device = device
        self.model_path = model_path
        t_init0 = time.perf_counter()
        if _debug_enabled():
            print(
                "[AICAS_DEBUG][init] "
                f"pid={os.getpid()} device={device} torch={torch.__version__} "
                f"cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()} "
                f"env_debug={os.environ.get('AICAS_DEBUG')} "
                f"abort_sample={os.environ.get('AICAS_DEBUG_ABORT_AT_SAMPLE')} "
                f"abort_generate={os.environ.get('AICAS_DEBUG_ABORT_AT_GENERATE')}",
                flush=True,
            )

        # Load processor with the model's default fixed image budget. Keeping
        # image shapes stable preserves CUDA graph reuse on the finals PPU.
        print(f"[VLMModel] Loading processor from {model_path}...")
        t_processor0 = time.perf_counter()
        try:
            real_processor = AutoProcessor.from_pretrained(model_path)
        except Exception as e:
            if _debug_enabled():
                _print_exception("load_processor", e)
            raise
        if _debug_enabled():
            print(
                f"[AICAS_DEBUG][init] processor_load_ms="
                f"{(time.perf_counter() - t_processor0) * 1000:.3f}",
                flush=True,
            )
        default_image_size = dict(real_processor.image_processor.size)
        # Cap high-res questions to reduce KV cache length for throughput
        cap_pixels = int(os.environ.get("AICAS_CAP_HIGH_RES_PIXELS", "0"))
        if cap_pixels > 0:
            default_image_size["longest_edge"] = min(default_image_size.get("longest_edge", 16777216), cap_pixels)

        image_pixels = os.environ.get("AICAS_IMAGE_PIXELS")
        if image_pixels:
            pixels = int(image_pixels)
            real_processor.image_processor.size["shortest_edge"] = pixels
            real_processor.image_processor.size["longest_edge"] = pixels
            print(f"[VLMModel] Fixed image pixel budget: {pixels}")

        # Load model
        print(f"[VLMModel] Loading model with FP16 + SDPA...")
        t_model0 = time.perf_counter()
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map=device,
                attn_implementation="sdpa",
            )
            self._model.eval()
            _debug_sync(device)
        except Exception as e:
            if _debug_enabled():
                _print_exception("load_model", e)
            raise
        if _debug_enabled():
            print(
                f"[AICAS_DEBUG][init] model_load_ms="
                f"{(time.perf_counter() - t_model0) * 1000:.3f}",
                flush=True,
            )

        # Track applied optimizations
        self._optimizations_applied = []

        # Apply custom kernel optimizations unless explicitly disabled.
        # This is useful on PPU while isolating correctness regressions.
        if os.environ.get("AICAS_DISABLE_OPTIMIZATIONS", "0") != "1":
            t_opt0 = time.perf_counter()
            try:
                self._model, results = apply_optimizations(self._model)
                _debug_sync(device)
            except Exception as e:
                if _debug_enabled():
                    _print_exception("apply_optimizations", e)
                raise
            if _debug_enabled():
                print(
                    f"[AICAS_DEBUG][init] apply_optimizations_ms="
                    f"{(time.perf_counter() - t_opt0) * 1000:.3f} "
                    f"groups={list(results.keys())}",
                    flush=True,
                )
            self._optimizations_applied.extend(results.keys())
        else:
            print("[VLMModel] Custom optimizations disabled by AICAS_DISABLE_OPTIMIZATIONS=1")

        _install_vision_feature_cache(self._model)
        self._optimizations_applied.append("vision_feature_cache")

        # cuBLAS warmup: pre-tune GEMM algorithms for actual prefill shapes
        with torch.inference_mode():
            for _ in range(3):
                # Warmup for typical prefill seq_len ~256-1024
                for _seq in [128, 256, 384, 512, 640, 768, 1024]:
                    _a = torch.randn(_seq, 2048, device=device, dtype=torch.float16)
                    # QKV: [seq, 2048] × [2048, 4096]
                    (_a @ torch.randn(2048, 4096, device=device, dtype=torch.float16))
                    # O_proj: [seq, 2048] × [2048, 2048]
                    (_a @ torch.randn(2048, 2048, device=device, dtype=torch.float16))
                    # gate_up: [seq, 2048] × [2048, 12288]
                    (_a @ torch.randn(2048, 12288, device=device, dtype=torch.float16))
                    # down: [seq, 6144] × [6144, 2048]
                    (_a[:, :6144] @ torch.randn(6144, 2048, device=device, dtype=torch.float16) if _seq >= 6144 else
                     torch.randn(_seq, 6144, device=device, dtype=torch.float16) @ torch.randn(6144, 2048, device=device, dtype=torch.float16))
            torch.cuda.synchronize()

        # End-to-end warmup: exercise full prefill + decode + CUDA graph capture
        # to ensure all lazy initializations complete before TTFT measurement.
        try:
            _warmup_msgs = [{'role': 'user', 'content': [
                {'type': 'text', 'text': 'warmup'},
            ]}]
            _warmup_inp = self._processor.apply_chat_template(
                _warmup_msgs, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors='pt',
            ).to(device)
            with torch.inference_mode():
                self._model.generate(**_warmup_inp, max_new_tokens=5, do_sample=False)
            torch.cuda.synchronize()
        except Exception:
            pass

        if _debug_enabled():
            _install_debug_generate_wrapper(self._model, self._device)
            self._optimizations_applied.append("debug_generate_wrapper")

        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")
        if _debug_enabled():
            print(
                f"[AICAS_DEBUG][init] total_init_ms="
                f"{(time.perf_counter() - t_init0) * 1000:.3f}",
                flush=True,
            )

        self._processor = _PrewarmProcessor(
            real_processor, lambda: self._model, self._device, default_image_size
        )
        self._model._aicas_tokenizer = real_processor.tokenizer

    # Required properties for benchmark
    @property
    def processor(self):
        return self._processor

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 128
    ) -> Dict:
        """Generate answer (optional method, mainly for debugging)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True
            )

        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]

        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return {
            "text": text,
            "token_count": len(generated_ids)
        }
