"""[SHLEE] AICAS 2026"""
import gc, os, traceback

gc.disable()

os.environ.setdefault("MY_KERNEL_QUANT_MODE",         "all")
os.environ.setdefault("MY_KERNEL_LMHEAD_PRUNE_K",     "0") #disabled
os.environ.setdefault("MY_KERNEL_TINY_LM_HEAD_K",     "2048")
os.environ.setdefault("MY_KERNEL_TINY_LM_HEAD_LOW_CAP", "2000")
os.environ.setdefault("MY_KERNEL_EXACT_LAYERS",       "") #disabled
os.environ.setdefault("MY_KERNEL_LLM_DROP_TAIL",      "0") #disabled
os.environ.setdefault("MY_KERNEL_MLP_PRUNE",          "1")
os.environ.setdefault("MY_KERNEL_MAX_PIXELS",         "300000")
os.environ.setdefault("MY_KERNEL_W4_VARIANT",         "auto")
os.environ.setdefault("MY_KERNEL_PREFILL_ATTN",       "flash_attn")
os.environ.setdefault("MY_KERNEL_TINY_GRAPH_LAYERS",  "0")
os.environ.setdefault("MY_KERNEL_EARLY_EXIT_MARGIN",  "10") # based on branchynet value 5
os.environ.setdefault("MY_KERNEL_POP_SKIP_LAYERS",    "24-26")
os.environ.setdefault("MY_KERNEL_POP_SKIP_LAYERS_TINY", "1-27") #disabled
os.environ.setdefault("MY_KERNEL_COS_EXIT_START",     "999") #disabled
os.environ.setdefault("MY_KERNEL_COS_EXIT_THRESH",    "1.001") #disabled
os.environ.setdefault("MY_KERNEL_AWQ",                "1")
os.environ.setdefault("MY_KERNEL_AWQ_CALIB_DATA",     "lmms-lab/textvqa")
os.environ.setdefault("MY_KERNEL_AWQ_CALIB_SPLIT",    "train")
os.environ.setdefault("MY_KERNEL_AWQ_CALIB_N",        "256")
os.environ.setdefault("MY_KERNEL_AWQ_N_ALPHA",        "20")
os.environ.setdefault("MY_KERNEL_GPTQ",               "1")
os.environ.setdefault("MY_KERNEL_GPTQ_CALIB_N",       "256")
os.environ.setdefault("MY_KERNEL_GPTQ_PERCDAMP",      "0.01")
os.environ.setdefault("MY_KERNEL_MAX_PIXELS_LARGE_THRESH", "0") #disabled
os.environ.setdefault("MY_KERNEL_IMG_PREDOWN_THRESH", "0") #disabled
os.environ.setdefault("MY_KERNEL_PREFILL_RMSNORM_OURS","1")

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.99",
)

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from my_kernel.mlp_channel_prune import apply_mlp_channel_pruning


def _pick_attn_impl():
    try:
        import flash_attn
        return "flash_attention_2"
    except Exception:
        return "sdpa"


class VLMModel:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        max_pixels = int(os.environ["MY_KERNEL_MAX_PIXELS"])
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.image_processor.max_pixels = max_pixels
        self.processor.image_processor.min_pixels = min(max_pixels, 256 * 256)

        attn_impl = _pick_attn_impl()
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.float16, device_map=device,
            attn_implementation=attn_impl,
        )
        self._model.eval()

        if os.environ.get("MY_KERNEL_AWQ", "1") == "1":
            try:
                from my_kernel.awq_quant import (
                    apply_awq_inplace, _load_calibration_samples,
                )
                samples = _load_calibration_samples(
                    os.environ["MY_KERNEL_AWQ_CALIB_DATA"],
                    os.environ["MY_KERNEL_AWQ_CALIB_SPLIT"],
                    int(os.environ["MY_KERNEL_AWQ_CALIB_N"]),
                    max_pixels,
                )
                apply_awq_inplace(
                    self._model, samples, self.processor,
                    device=device, max_pixels=max_pixels,
                    n_alpha=int(os.environ["MY_KERNEL_AWQ_N_ALPHA"]),
                )
            except Exception as e:
                print(f"[evaluation_wrapper] AWQ failed: {e}")
                traceback.print_exc()

        if os.environ.get("MY_KERNEL_GPTQ", "1") == "1":
            try:
                from my_kernel.gptq_quant import apply_gptq_inplace
                from my_kernel.awq_quant import _load_calibration_samples
                gptq_samples = _load_calibration_samples(
                    os.environ["MY_KERNEL_AWQ_CALIB_DATA"],
                    os.environ["MY_KERNEL_AWQ_CALIB_SPLIT"],
                    int(os.environ["MY_KERNEL_GPTQ_CALIB_N"]),
                    max_pixels,
                )
                apply_gptq_inplace(
                    self._model, gptq_samples, self.processor,
                    device=device, max_pixels=max_pixels,
                    percdamp=float(os.environ["MY_KERNEL_GPTQ_PERCDAMP"]),
                )
            except Exception as e:
                print(f"[evaluation_wrapper] GPTQ failed: {e}")
                traceback.print_exc()

        if os.environ.get("MY_KERNEL_MLP_PRUNE", "1") == "1":
            self.mlp_prune_info = apply_mlp_channel_pruning(self._model)
        else:
            self.mlp_prune_info = None

        try:
            import my_kernel
            self._runner = my_kernel.apply_optimizations(
                self._model, self.processor, device=device
            )
        except Exception:
            traceback.print_exc()
            self._runner = None

    @property
    def model(self):
        return self._runner if self._runner is not None else self._model
