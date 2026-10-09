from dataclasses import dataclass

import torch


from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
)
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel

from transformers.utils import (
    auto_docstring,
)


from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
)
import torch._dynamo as dynamo

dynamo.config.error_on_recompile = True


@dataclass
@auto_docstring
class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
    r"""
    deepstack_features (`List[torch.FloatTensor]`, *optional*):
        List of hidden-states (feature maps) from deepstack layers.
    """

    deepstack_features: list[torch.FloatTensor] | None = None


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    all_layers_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    stage_names: tuple[str, ...] | None = None
    stage_hidden_states: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring
class Qwen3VLPreTrainedModel(PreTrainedModel):
    config: Qwen3VLConfig
    base_model_prefix = "model"
    input_modalities = ("image", "video", "text")
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    # from .language import Qwen3VLTextDecoderLayer, Qwen3VLTextAttention
    # _can_record_outputs = {
    #     "hidden_states": Qwen3VLTextDecoderLayer,
    #     "attentions": Qwen3VLTextAttention,
    # }

    def _init_weights(self, module):
        super()._init_weights(module)
        from aicas2026gc.qwen3vl2b.vision import Qwen3VLVisionRotaryEmbedding

        if isinstance(module, Qwen3VLVisionRotaryEmbedding):
            with torch.no_grad():
                inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
                # init.copy_(module.inv_freq, inv_freq)
                seq = torch.arange(module.max_seqlen, device=inv_freq.device, dtype=inv_freq.dtype)
                freq_tb = torch.outer(seq, inv_freq)
                # init.copy_(module.freqs_cos, freq_tb.cos())
                # init.copy_(module.freqs_sin, freq_tb.sin())
                module.inv_freq.copy_(inv_freq)
                module.freqs_cos.copy_(freq_tb.cos())
                module.freqs_sin.copy_(freq_tb.sin())
