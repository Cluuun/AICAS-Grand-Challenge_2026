"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
from typing import Dict
import json
import os
import sys
import types
from typing import Unpack as _TypingUnpack
try:
    from PIL import Image
except ImportError:
    # For testing without PIL
    class Image:
        pass
import torch


def _install_transformers_mixed_env_shims() -> None:
    """Avoid optional HF import paths that crash in the official mixed PPU env."""
    try:
        import transformers.utils.import_utils as import_utils

        # We do not use HF/bitsandbytes/xformers quantization paths. In the
        # official container they can resolve to leftover native extensions that
        # abort the process when Transformers probes optional backends.
        if hasattr(import_utils, "_bitsandbytes_available"):
            import_utils._bitsandbytes_available = False
        if hasattr(import_utils, "_xformers_available"):
            import_utils._xformers_available = False
    except Exception:
        pass

    if "transformers.processing_utils" not in sys.modules:
        processing_utils = types.ModuleType("transformers.processing_utils")
        processing_utils.Unpack = _TypingUnpack
        sys.modules["transformers.processing_utils"] = processing_utils

    if "transformers.quantizers.quantizers_utils" not in sys.modules:
        quantizers_utils = types.ModuleType("transformers.quantizers.quantizers_utils")

        def get_module_from_name(module, tensor_name: str):
            if "." in tensor_name:
                module_name, tensor_name = tensor_name.rsplit(".", 1)
                module = module.get_submodule(module_name)
            return module, tensor_name

        quantizers_utils.get_module_from_name = get_module_from_name
        sys.modules["transformers.quantizers.quantizers_utils"] = quantizers_utils

    if "transformers.quantizers.auto" not in sys.modules:
        quantizers_auto = types.ModuleType("transformers.quantizers.auto")

        def get_hf_quantizer(config, quantization_config, dtype, from_tf, from_flax, device_map, weights_only, user_agent):
            if quantization_config is not None or hasattr(config, "quantization_config"):
                raise RuntimeError("HF quantized checkpoints are disabled in this submission runtime")
            return None, config, dtype, device_map

        class AutoHfQuantizer:
            @staticmethod
            def supports_quant_method(_quantization_config_dict):
                return False

        class AutoQuantizationConfig:
            pass

        def register_quantization_config(_method):
            def decorator(cls):
                return cls

            return decorator

        def register_quantizer(_name):
            def decorator(cls):
                return cls

            return decorator

        quantizers_auto.get_hf_quantizer = get_hf_quantizer
        quantizers_auto.AutoHfQuantizer = AutoHfQuantizer
        quantizers_auto.AutoQuantizationConfig = AutoQuantizationConfig
        quantizers_auto.register_quantization_config = register_quantization_config
        quantizers_auto.register_quantizer = register_quantizer
        sys.modules["transformers.quantizers.auto"] = quantizers_auto

    if "transformers.quantizers" not in sys.modules:
        quantizers = types.ModuleType("transformers.quantizers")
        quantizers.__path__ = []

        class HfQuantizer:
            pass

        quantizers.HfQuantizer = HfQuantizer
        quantizers.AutoHfQuantizer = sys.modules["transformers.quantizers.auto"].AutoHfQuantizer
        quantizers.AutoQuantizationConfig = sys.modules["transformers.quantizers.auto"].AutoQuantizationConfig
        quantizers.get_module_from_name = sys.modules[
            "transformers.quantizers.quantizers_utils"
        ].get_module_from_name
        quantizers.register_quantization_config = sys.modules[
            "transformers.quantizers.auto"
        ].register_quantization_config
        quantizers.register_quantizer = sys.modules["transformers.quantizers.auto"].register_quantizer
        sys.modules["transformers.quantizers"] = quantizers

try:
    from torchvision.transforms import InterpolationMode as _InterpolationMode  # noqa: F401
except Exception:
    try:
        import transformers.utils.import_utils as _transformers_import_utils

        # Some PPU environments ship a torchvision wheel whose custom ops do not
        # register cleanly with the installed torch build. Qwen3-VL image-only
        # samples can use the slow PIL processor, so skip torchvision probing
        # instead of failing the whole transformers import.
        _transformers_import_utils._torchvision_available = False
        for _name in list(sys.modules):
            if _name == "torchvision" or _name.startswith("torchvision."):
                sys.modules.pop(_name, None)
    except Exception:
        pass

_install_transformers_mixed_env_shims()
from transformers import AutoModelForImageTextToText

from my_kernel.conf import conf_bool, conf_float
from my_kernel.prompt_short_answer import (
    finalize_prompt_len,
    finish_rendered_prompt,
    prepare_messages_for_render,
)
from my_kernel import (
    apply_resolution_bucket_processor,
    apply_cuda_graph_decode,
    apply_decode_kernel_fusions,
    apply_flash_kvcache_attention,
    apply_lookahead_decode,
    apply_runtime_int8_quant,
    apply_vision_kernel_fusions,
    apply_vision_prefill_cache,
)


class _ImageOnlyQwen3VLProcessor:
    """Small Qwen3-VL processor that avoids AutoProcessor's fast/video imports."""

    model_input_names = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")
        self.image_token_id = getattr(tokenizer, "image_token_id", None) or tokenizer.convert_tokens_to_ids(
            self.image_token
        )
        self.video_token_id = getattr(tokenizer, "video_token_id", None) or tokenizer.convert_tokens_to_ids(
            self.video_token
        )
        self.vision_start_token_id = getattr(
            tokenizer, "vision_start_token_id", None
        ) or tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = getattr(
            tokenizer, "vision_end_token_id", None
        ) or tokenizer.convert_tokens_to_ids(self.vision_end_token)

    @staticmethod
    def _extract_images(messages):
        images = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image" or "image" in item or "image_url" in item:
                    image = item.get("image", item.get("image_url"))
                    if image is not None:
                        images.append(image)
        return images

    def _render_chat(self, messages, add_generation_prompt=True, **kwargs):
        messages = prepare_messages_for_render(messages, processor=self)
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            )
            return finish_rendered_prompt(text)
        except Exception:
            # Benchmark inputs are a single user turn with one image and text.
            parts = ["<|im_start|>user\n"]
            for message in messages:
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image" or "image" in item or "image_url" in item:
                        parts.append("<|vision_start|><|image_pad|><|vision_end|>\n")
                    elif "text" in item:
                        parts.append(str(item["text"]))
            parts.append("<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return finish_rendered_prompt("".join(parts))

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        image_inputs = {}
        image_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs.get("image_grid_thw")

        if text is None:
            text = ""
        if not isinstance(text, list):
            text = [text]
        text = text.copy()

        if image_grid_thw is not None:
            merge_length = int(self.image_processor.merge_size) ** 2
            index = 0
            for i, item in enumerate(text):
                while self.image_token in item:
                    num_image_tokens = int(image_grid_thw[index].prod()) // merge_length
                    item = item.replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = item.replace("<|placeholder|>", self.image_token)

        text_inputs = self.tokenizer(
            text,
            padding=False,
            return_token_type_ids=False,
        )

        from transformers.feature_extraction_utils import BatchFeature

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        return_dict=False,
        return_tensors=None,
        **kwargs,
    ):
        text = self._render_chat(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        images = self._extract_images(messages)
        batch = self(images=images or None, text=text, return_tensors=return_tensors)
        if images:
            # benchmark warmup/performance reuse the same PIL objects; carry a stable per-process id
            # so the model-side vision cache can safely distinguish same-shape different images.
            batch["_vision_cache_key"] = tuple(id(image) for image in images)
        if return_dict:
            finalize_prompt_len(self, int(batch["input_ids"].shape[-1]))
            return batch
        finalize_prompt_len(self, int(batch["input_ids"].shape[-1]))
        return batch["input_ids"]

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


def _load_processor(model_path: str):
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

    with open(os.path.join(model_path, "preprocessor_config.json"), "r", encoding="utf-8") as f:
        processor_config = json.load(f)
    image_processor = Qwen2VLImageProcessor(
        image_mean=processor_config.get("image_mean"),
        image_std=processor_config.get("image_std"),
        patch_size=int(processor_config.get("patch_size", 16)),
        temporal_patch_size=int(processor_config.get("temporal_patch_size", 2)),
        merge_size=int(processor_config.get("merge_size", 2)),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("[VLMModel] using image-only Qwen3-VL processor (slow image processor, no AutoProcessor)")
    return _ImageOnlyQwen3VLProcessor(image_processor=image_processor, tokenizer=tokenizer)


def _configure_processor_pixels(processor) -> None:
    """用单个 IMAGE_RESIZE_RATIO 控制动态分辨率视觉 token 数。"""
    ratio = conf_float("IMAGE_RESIZE_RATIO", "1.0", minimum=0.5, maximum=1.5)
    min_px = max(1, int(round(65536 * ratio * ratio)))
    max_px = max(min_px, int(round(262144 * ratio * ratio)))
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min_px
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_px
    print(f"[VLMModel] image resize ratio={ratio:.2f} pixels: min={min_px} max={max_px}")


def _load_model_with_attn(model_path: str, device: str, dtype=torch.float16):
    """优先 flash_attention_2，失败回退默认实现。"""
    common = dict(dtype=dtype, device_map=device)
    if not conf_bool("ENABLE_FLASH_ATTENTION_2", "1"):
        return AutoModelForImageTextToText.from_pretrained(model_path, **common)
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            **common,
        )
        print("[VLMModel] attn_implementation=flash_attention_2")
        return model
    except Exception as exc:
        print(f"[VLMModel] flash_attention_2 unavailable ({exc}), fallback default attn")
        return AutoModelForImageTextToText.from_pretrained(model_path, **common)


class VLMModel:
    """
    Participant optimization class - modify this to implement optimizations.
    
    Optimization Architecture:
    - Split optimizations into separate methods for isolation and testing
    - Enable/disable each optimization independently in __init__
    - Each optimization method can be tested individually
    
    Important Notes:
    1. Benchmark directly calls self.model.generate() for performance testing.
    2. Your optimizations should modify self.model or its operators via Monkey Patch.
    3. All optimizations are applied in __init__ by calling optimization methods.
    """
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """
        Initialize model and apply optimizations.
        
        Args:
            model_path: Qwen3-VL-2B-Instruct model path
            device: CUDA device, e.g., "cuda:0"
        """
        self._device = device
        self.model_path = model_path
        self._optimizations_applied: list = []
        
        # Load processor
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = _load_processor(model_path)
        _configure_processor_pixels(self._processor)
        if apply_resolution_bucket_processor(self._processor):
            self._optimizations_applied.append("vision_resolution_policy")
        
        # Load model
        print(f"[VLMModel] Loading model with FP16...")
        self._model = _load_model_with_attn(model_path, device)
        self._model.eval()
        
        apply_vision_prefill_cache(self._model)
        apply_vision_kernel_fusions(self._model)
        quant_enabled = apply_runtime_int8_quant(self._model)
        apply_decode_kernel_fusions(self._model)
        apply_flash_kvcache_attention(self._model)
        apply_cuda_graph_decode(self._model)
        lookahead_enabled = apply_lookahead_decode(self._model)
        
        self._optimizations_applied.extend([
            "vision_prefill_cache",
            "vision_kernel_fusions",
            "decode_kernel_fusions",
            "flash_kvcache_attention",
            "cuda_graph_decode",
            "ttft_dedicated_forward",
        ])
        if quant_enabled:
            self._optimizations_applied.append("runtime_int8_quant")
        if lookahead_enabled:
            self._optimizations_applied.append("lookahead_decode")
        if conf_bool("ENABLE_FLASH_ATTENTION_2", "1"):
            self._optimizations_applied.append("flash_attention_2")
        
        # ================================================================
        # Participant Optimization Area - Enable/disable optimizations here
        # Uncomment the optimization methods you want to apply
        # ================================================================
        
        # 1. Vision Encoder Acceleration
        # self._optimize_vision_encoder()
        
        # 2. KV Cache Management
        # self._optimize_kv_cache()
        
        # 3. Cross-modal Connector Optimization
        # self._optimize_cross_modal_connector()
        
        # 4. Flash Attention Optimization
        # self._enable_flash_attention()
        
        # 5. Quantization
        # self._apply_quantization()
        
        # Optional: Explore model structure before optimization
        # self._explore_model_structure()
        
        # ================================================================
        
        print(f"[VLMModel] Model loaded successfully on {device}")
        if self._optimizations_applied:
            print(f"[VLMModel] Applied optimizations: {', '.join(self._optimizations_applied)}")
    
    # ================================================================
    # Optimization Methods - Implement your optimizations here
    # ================================================================
    
    def _explore_model_structure(self):
        """
        Helper method to explore model structure.
        
        Use this to understand the model architecture before implementing optimizations.
        This helps identify where to apply monkey patches.
        """
        print("=" * 60)
        print("Model Structure Exploration")
        print("=" * 60)
        
        # Explore vision model structure
        if hasattr(self._model, 'vision_model'):
            print(f"Vision Model: {type(self._model.vision_model)}")
            if hasattr(self._model.vision_model, 'encoder'):
                if hasattr(self._model.vision_model.encoder, 'layers'):
                    print(f"  Vision Encoder Layers: {len(self._model.vision_model.encoder.layers)}")
                    # Show first layer structure
                    if len(self._model.vision_model.encoder.layers) > 0:
                        print(f"  First Layer Type: {type(self._model.vision_model.encoder.layers[0])}")
        else:
            print("Vision Model: Not found (model structure may differ)")
        
        # Explore language model structure
        if hasattr(self._model, 'model'):
            print(f"Language Model: {type(self._model.model)}")
            if hasattr(self._model.model, 'layers'):
                print(f"  Language Model Layers: {len(self._model.model.layers)}")
        else:
            print("Language Model: Not found (model structure may differ)")
        
        # Explore cross-modal components
        cross_modal_attrs = ['connector', 'cross_attn', 'cross_attention', 'proj', 'projector']
        found_components = []
        for attr in cross_modal_attrs:
            if hasattr(self._model, attr):
                found_components.append(attr)
        if found_components:
            print(f"Cross-modal Components: {', '.join(found_components)}")
        else:
            print("Cross-modal Components: Explore manually (structure may vary)")
        
        print("=" * 60)
        print("Tip: Use print(self._model) to see full model structure")
        print("=" * 60)
    
    def _optimize_vision_encoder(self):
        """
        Optimize Vision Encoder for high-resolution image inputs.
        
        Optimization Directions:
        1. Patch embedding convolution optimization
        2. Vision Transformer attention mechanism optimization
        3. Layer normalization optimization
        4. Memory-efficient image processing
        
        Implementation Steps:
        1. Inspect model structure: call self._explore_model_structure()
        2. Identify bottlenecks using profiling tools (PyTorch Profiler, nsys, etc.)
        3. Implement optimized operators (Triton/CUDA kernels)
        4. Replace original operators via monkey patch
        
        Target Components:
        - self._model.vision_model (if exists)
        - Vision encoder layers and attention mechanisms
        - Convolution operations in patch embedding
        """
        # TODO: Implement your Vision Encoder optimization here
        # 
        # Example workflow:
        # 1. from your_optimization import optimized_attention, optimized_conv
        # 2. Inspect: print(self._model.vision_model) to find target layers
        # 3. Replace: layer.self_attn.forward = optimized_attention
        # 4. Test: Run benchmark to verify improvement
        
        if 'vision_encoder' not in self._optimizations_applied:
            self._optimizations_applied.append('vision_encoder')
    
    def _optimize_kv_cache(self):
        """
        Optimize KV Cache management to reduce memory fragmentation.
        
        Optimization Directions:
        1. Memory layout optimization (contiguous memory allocation)
        2. Fragmentation-free allocation strategies
        3. Efficient cache reuse patterns
        4. Dynamic cache sizing
        
        Implementation Steps:
        1. Understand current KV cache implementation in model layers
        2. Design memory-efficient cache allocation strategy
        3. Implement custom KV cache allocator if needed
        4. Apply optimizations via monkey patch or config modification
        
        Target Components:
        - self._model.config (cache configuration)
        - Attention layers (KV cache allocation)
        - Generation loop (cache management)
        """
        # Enable KV Cache first
        self._model.config.use_cache = True
        if hasattr(self._model.config, 'pad_token_id'):
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._model.config.eos_token_id
        
        # TODO: Implement advanced KV Cache optimizations here
        # 
        # Example workflow:
        # 1. from your_optimization import FragmentationFreeKVCache
        # 2. for layer in self._model.model.layers:
        # 3.     layer.attention.custom_kv_cache = FragmentationFreeKVCache()
        # 4. Test: Monitor memory usage and generation speed
        
        if 'kv_cache' not in self._optimizations_applied:
            self._optimizations_applied.append('kv_cache')
    
    def _optimize_cross_modal_connector(self):
        """
        Optimize Cross-modal Connector computation efficiency.
        
        Optimization Directions:
        1. Cross-attention mechanism optimization
        2. Vision-to-language projection optimization
        3. Multi-modal fusion layer efficiency
        4. Feature alignment and transformation optimization
        
        Implementation Steps:
        1. Identify cross-modal components using self._explore_model_structure()
        2. Profile cross-modal operations to find bottlenecks
        3. Implement optimized cross-attention or projection kernels
        4. Replace original operations via monkey patch
        
        Note: Qwen3-VL's cross-modal structure may vary.
        Use model exploration to identify actual component names and locations.
        """
        # TODO: Implement your Cross-modal Connector optimization here
        # 
        # Example workflow:
        # 1. Explore: self._explore_model_structure() to find connector components
        # 2. from your_optimization import optimized_cross_attention
        # 3. Identify: Inspect model to find cross-attention layers
        # 4. Replace: connector.cross_attention.forward = optimized_cross_attention
        # 5. Test: Verify accuracy and performance improvements
        
        if 'cross_modal' not in self._optimizations_applied:
            self._optimizations_applied.append('cross_modal')
    
    def _enable_flash_attention(self):
        """
        Enable or implement Flash Attention optimization.
        
        Implementation Approaches:
        
        Approach 1: Enable PyTorch's Built-in Flash Attention (Simple)
            - Uses torch.backends.cuda.enable_flash_sdp(True)
            - Easy to enable but limited customization
            - May not work for all attention patterns in Qwen3-VL
        
        Approach 2: Implement Custom Flash Attention (Advanced, Recommended)
            - Write custom Triton/CUDA kernels for attention computation
            - Replace torch.nn.functional.scaled_dot_product_attention
            - Full control over attention computation and memory layout
            - Better performance potential but requires more implementation effort
        
        Recommended: Implement Approach 2 for better performance gains.
        Use profiling to identify which attention operations benefit most from optimization.
        """
        # TODO: Choose and implement your Flash Attention approach
        
        # Approach 1: Simple (enable PyTorch built-in)
        # torch.backends.cuda.enable_flash_sdp(True)
        
        # Approach 2: Advanced (custom implementation - recommended)
        # from your_optimization import custom_flash_attention
        # torch.nn.functional.scaled_dot_product_attention = custom_flash_attention
        # 
        # Or replace at layer level:
        # for layer in self._model.model.layers:
        #     layer.self_attn.forward = custom_attention_with_flash
        
        if 'flash_attention' not in self._optimizations_applied:
            self._optimizations_applied.append('flash_attention')
    
    def _apply_quantization(self):
        """
        Apply quantization to reduce model size and speed up inference.
        
        Optimization Directions:
        1. INT8 quantization (8-bit integer)
        2. FP8 quantization (8-bit floating point)
        3. Mixed precision quantization
        4. Dynamic vs static quantization
        
        Implementation Steps:
        1. Choose quantization strategy based on accuracy/performance trade-off
        2. Use quantization libraries (BitsAndBytes, TensorRT, etc.)
        3. Calibrate quantized model on validation data
        4. Verify accuracy preservation
        
        Note: Quantization may require reloading the model with quantization config.
        Consider applying quantization before other optimizations if model reload is needed.
        """
        # TODO: Implement your quantization here
        # 
        # Example workflow:
        # 1. from transformers import BitsAndBytesConfig
        # 2. quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        # 3. Note: May need to reload model with quantization config
        # 4. Test: Verify accuracy and performance improvements
        
        if 'quantization' not in self._optimizations_applied:
            self._optimizations_applied.append('quantization')
    
    # Required properties for benchmark
    @property
    def processor(self):
        """
        Required by benchmark for input processing.
        
        Benchmark uses this to prepare inputs with unified tokenizer.
        """
        return self._processor
    
    @property
    def model(self):
        """
        Required by benchmark for direct model.generate() calls.
        
        Benchmark directly calls self.model.generate() for performance testing.
        Your optimizations should modify this model object or its operators.
        """
        return self._model
    
    @property
    def device(self):
        """
        Required by benchmark for device information.
        """
        return self._device
    
    def generate(
        self, 
        image: Image.Image, 
        question: str, 
        max_new_tokens: int = 128
    ) -> Dict:
        """
        Generate answer (optional method, mainly for debugging).
        
        Note: Benchmark uses self.model.generate() directly for performance testing.
        This method is provided for convenience and debugging purposes.
        
        Args:
            image: PIL Image object
            question: Question text
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Dict: {
                "text": str,        # Generated text answer
                "token_count": int  # Generated token count
            }
        """
        # Build Qwen3-VL message format
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]
        
        # Process inputs
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self._device)
        
        # Generate
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True
            )
        
        # Extract generated tokens (remove input part)
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[0][input_len:]
        
        # Decode
        text = self._processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return {
            "text": text,
            "token_count": len(generated_ids)
        }
