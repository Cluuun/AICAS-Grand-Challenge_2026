"""
AICAS 2026 - Participant Core Modification File

Participants should modify the VLMModel class to implement optimizations.

Note:
- Benchmark directly calls self.model.generate() for performance testing.
- Your optimizations should modify self.model or its operators in __init__ via Monkey Patch.
- The generate() method is optional and mainly for debugging.
"""
import os
from typing import Dict


_SUBMISSION_ROOT = os.path.dirname(os.path.abspath(__file__))


def _submission_path(*parts: str) -> str:
    return os.path.join(_SUBMISSION_ROOT, *parts)


def _set_submission_default_env() -> None:
    """Mirror the current eval.sh defaults for official wrapper-only evaluation."""
    defaults = {
        "USE_CUDAGRAPH": "1",
        "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "AUTOTUNE_CACHE_JSON": _submission_path("aicas2026gc", "runtime_autotune.json"),
        "AUTOTUNE_CACHE_LOG": "0",
        "INT8_LINEAR_LOG": "0",
        "DECODE_SKIP_LAYERS": "7,8,9,10,11,12,13,14,19,20,21,22,23",
        "DECODE_SKIP_SCALE_SPECS": (
            "15:"
            + _submission_path(
                "layer_stats",
                "steering_all_layers_calib500_decode64",
                "vectors",
                "decode_skip_6_14_ls_affine.pt",
            )
            + ",24:"
            + _submission_path(
                "layer_stats",
                "steering_all_layers_calib500_decode64",
                "vectors",
                "decode_skip_18_23_ls_affine.pt",
            )
        ),
        "DECODE_SKIP_LOWRANK_SPECS": (
            "15:"
            + _submission_path(
                "layer_stats",
                "steering_all_layers_calib500_decode64",
                "lowrank_6_14",
                "decode_skip_6_14_ls_residual_lowrank_k128.pt",
            )
            + ",24:"
            + _submission_path(
                "layer_stats",
                "steering_all_layers_calib500_decode64",
                "lowrank_18_23_k128",
                "decode_skip_18_23_ls_residual_lowrank_k128.pt",
            )
        ),
        "DECODE_SKIP_FUSED_BRIDGE": "1",
        "PREFILL_INT8_ATTN": "both",
        "PREFILL_INT8_MLP": "both",
        "PREFILL_INT8_AUTOTUNE": "1",
        "PREFILL_INT8_FUSED_QUANT": "1",
        "VISION_INT8_LINEAR": "both",
        "VISION_INT8_AUTOTUNE": "0",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


_set_submission_default_env()

try:
    from PIL import Image
except ImportError:
    # For testing without PIL
    class Image:
        pass
import torch
import torch.nn as nn
from transformers import AutoProcessor
from modeling_qwen3_vl import Qwen3VLForConditionalGeneration
# from modeling_qwen3_vl_bs1_decode_fast import Qwen3VLForConditionalGeneration

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
        
        # Load processor
        print(f"[VLMModel] Loading processor from {model_path}...")
        self._processor = AutoProcessor.from_pretrained(model_path)
        vision_max_pixels = os.environ.get("VISION_MAX_PIXELS", "").strip()
        vision_min_pixels = os.environ.get("VISION_MIN_PIXELS", "").strip()
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is not None and hasattr(image_processor, "size"):
            original_size = dict(image_processor.size)
            if vision_max_pixels:
                image_processor.size["longest_edge"] = int(vision_max_pixels)
            if vision_min_pixels:
                image_processor.size["shortest_edge"] = int(vision_min_pixels)
            print(f"[VisionResize] size {original_size} -> {dict(image_processor.size)}")
        elif vision_max_pixels or vision_min_pixels:
            print("[VisionResize] processor has no image_processor.size; env ignored")
        
        # Load model
        print(f"[VLMModel] Loading model with FP16...")
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device
        )
        self._model.eval()
        
        # Track applied optimizations
        self._optimizations_applied = []
        
        for name, module in self._model.named_modules():

            # 注意顺序, 要先 Qwen3VLTextDecoderLayer 再 Qwen3VLTextMLP
            if module.__class__.__name__ == "Qwen3VLTextDecoderLayer": # TODO
                module.mlp_post_init()

            elif module.__class__.__name__ == "Qwen3VLTextMLP": # TODO
                module.linear_post_init()

            elif module.__class__.__name__ == "Qwen3VLTextAttention": # TODO
                module.linear_post_init()
            
            elif module.__class__.__name__ == "Qwen3VLVisionBlock": # TODO
                module.post_init_fusion()

            elif module.__class__.__name__ == "Qwen3VLTextRotaryEmbedding": # TODO
                module.post_init_inv_freq_cache_sin_cos()
                module.post_register_buffer()

            elif module.__class__.__name__ == "Qwen3VLForConditionalGeneration": # TODO
                module.linear_post_init()
                
                # FastAdaptiveSafeRecallLMHead
                # module.fast_head._post_init(module.lm_head, rank=256) # head 整体替换一下
            
                

        # # 这里整体做一次 marlin 的替换试试
        # for name, module in self._model.named_modules():
        #     if isinstance(module, nn.Linear):
        #         pass


        
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
