from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.generation.utils import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
)

from transformers.modeling_outputs import ModelOutput

from transformers.processing_utils import Unpack

from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
)

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
import torch._dynamo as dynamo

dynamo.config.error_on_recompile = True

from aicas2026gc import RuntimeAutotuner
from aicas2026gc.qwen3vl2b.language import Qwen3VLTextModel
from aicas2026gc.qwen3vl2b.lm_head import init_lm_head, lm_head_argmax
from aicas2026gc.qwen3vl2b.vision import Qwen3VLVisionModel

from aicas2026gc.qwen3vl2b.basemodel import Qwen3VLPreTrainedModel, BaseModelOutputWithDeepstackFeatures


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Qwen3VLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    all_layers_hidden_states: tuple[torch.FloatTensor] | None = None


@auto_docstring
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    base_model_prefix = "model"
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.LongTensor:
        """
        Compute 3D positional indices for vision tokens using elegant Meshgrid operations.
        """
        # 提取并缩放 T, H, W 维度
        t, h, w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )

        # 🚀 5. 优雅重构：抛弃繁琐的 repeat_interleave，拥抱原生 Meshgrid
        # 修复了原版 Bug：Temporal ID 现在会根据 time_interval 和所在时间块正确递增
        t_grid = torch.arange(t, device=device) * time_interval + start_position
        h_grid = torch.arange(h, device=device) + start_position
        w_grid = torch.arange(w, device=device) + start_position

        # indexing='ij' 确保展平后的顺序为 (T块) -> (H行) -> (W列)
        grid_t, grid_h, grid_w = torch.meshgrid(t_grid, h_grid, w_grid, indexing="ij")

        # 一步到位：组合并展平，直接返回 shape 为 (3, sequence_length) 的 Tensor
        return torch.stack([grid_t.flatten(), grid_h.flatten(), grid_w.flatten()], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        position_ids_buffer: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's sizes. The utility expects a `vision + text`
        sequence and will error out otherwise. For pure text sequence, please rely on model's auto-inferred
        position ids. In a mixed vision + text sequence, vision tokens use 3D RoPE (temporal, height, width)
        while text tokens use standard 1D RoPE.

        Example:
            Temporal patches: 3; Height patches: 2; Width patches: 2
            Each vision input results in (temporal x height × width) positions. Here: 3 x 2 × 2 = 12 positions total.

            Temporal position IDs are spaced by:
                `interval = tokens_per_second * temporal_patch_size / fps`

                If fps = 1; tokens_per_second = 25; temporal_patch_size = 2, temporal IDs increase by 50 for each temporal patch:
                `[0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]`

            Height IDs repeat per row: `[0, 0, 1, 1, ...]`
            Width IDs alternate per column: `[0, 1, 0, 1, ...]`
            Text tokens follow standard 1D RoPE and the position IDs grow consequently with a step of `1`

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size  # 2

        B, S = input_ids.shape
        # 1. 预分配完整的输出 Tensor，彻底告别碎片化的 list.append 和 torch.cat
        position_ids = torch.zeros(3, B, S, dtype=input_ids.dtype, device=input_ids.device) if position_ids_buffer is None else position_ids_buffer

        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,  # None, 不处理视频
        }

        for batch_idx in range(B):
            types = mm_token_type_ids[batch_idx]

            # 🚀 3. 真正的优雅与极速：单次底层 C++ 调用完成游程统计 (Run-Length Encoding)
            # 完美替代 itertools.groupby 和低效的错位 != 切片
            values, counts = torch.unique_consecutive(types, return_counts=True)

            start = 0
            current_pos = 0

            # 4. 直接遍历模态及其对应的连续长度进行切片赋值
            for modality, length in zip(values.tolist(), counts.tolist()):
                end = start + length

                if modality == 0:
                    # 文本片段: 直接生成并利用广播机制赋值
                    pos = torch.arange(current_pos, current_pos + length, device=input_ids.device)
                    position_ids[:, batch_idx, start:end] = pos
                    current_pos += length
                else:
                    # 视觉片段: 提取网格，调用优雅版三维坐标生成器
                    grid_thw = next(grid_iters[modality])
                    pos = self.get_vision_position_ids(current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device)
                    position_ids[:, batch_idx, start:end] = pos
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

                start = end

        return position_ids, None

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)

        # TODO: 去掉傻逼 tolist
        grid_thw_list = image_grid_thw.tolist()
        pos_embeds = self.visual.real_fast_pos_embed_interpolate(grid_thw_list)
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.visual.elegant_rot_pos_emb(grid_thw_list)

        vision_output: BaseModelOutputWithDeepstackFeatures = self.visual(
            pixel_values, pos_embeds=pos_embeds, rotary_pos_emb_cos=rotary_pos_emb_cos, rotary_pos_emb_sin=rotary_pos_emb_sin, return_dict=True, **kwargs
        )
        return vision_output

    @torch.inference_mode()
    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,  # [4, 1, 690]
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLModelOutputWithPast:
        # nvtx.range_push("Qwen3VLModel.forward")
        if inputs_embeds is None:  # True
            # PREFILL, [1, 690] -> [1, 690, 2048]
            # DECODE,  [1, 1] -> [1, 1, 2048]
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        deepstack_image_embeds = None

        if pixel_values is not None:  # ONLY FOR PREFILL
            # nvtx.range_push("Qwen3VLVisionModel.forward")
            image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            # nvtx.range_pop()
            image_embeds = image_outputs.pooler_output  # tuple [  [672, 2048],   ]
            deepstack_image_embeds = image_outputs.deepstack_features  # tuple [  ([672, 2048] x 3  ]
            image_mask = input_ids == self.config.image_token_id
            inputs_embeds[image_mask] = image_embeds

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=None,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            cache_position=None,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deepstack_image_embeds,
            **kwargs,
        )

        # nvtx.range_pop()
        return Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


@dataclass
class Qwen3VLCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None
    all_layers_hidden_states: tuple[torch.FloatTensor] | None = None


class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)  # Bias 是 False!
        MAX_SEQ_LEN = 4096  # TODO: 4096 需要配置
        input_ids_buffer = torch.empty((1, MAX_SEQ_LEN), dtype=torch.long)
        position_ids_buffer = torch.empty((4, 1, MAX_SEQ_LEN), dtype=torch.long)
        torch.arange(MAX_SEQ_LEN, dtype=torch.long, out=position_ids_buffer[0, 0])
        self.register_buffer("input_ids_buffer", input_ids_buffer, persistent=False)
        self.register_buffer("position_ids_buffer", position_ids_buffer, persistent=False)
        self.register_buffer("lm_head_token_id_map", torch.empty(0, dtype=torch.int64), persistent=False)
        self.register_buffer("lm_head_signmatch_packed_signs", torch.empty(0, dtype=torch.int32), persistent=False)

        self.lm_head_signmatch_rows_per_block = 512
        self.lm_head_signmatch_local_k = 16
        self.lm_head_signmatch_workspace = None
        self.lm_head_signmatch_argmax_out = None
        self.lm_head_argmax_tuner = RuntimeAutotuner(
            fallback="exact",
            name=self.__class__.__name__ + "_lm_head_argmax",
            check_correctness=False,
            strict=False,
        )

        self.post_init()

        # 分配一块锁页内存，专门用来接收异步回传的 Token
        self.pinned_token_cpu_buffer = torch.empty((1,), dtype=torch.int64, pin_memory=True, device="cpu")

        # Decoder.norm.weight -> lm_head.weight
        self.fused_norm_weight_lm_head_weight = False

    def linear_post_init(self):
        if self.fused_norm_weight_lm_head_weight:
            return

        self.lm_head_token_id_map = init_lm_head(self.lm_head, self.model.language_model.norm.weight)
        self._init_lm_head_signmatch()

        self.fused_norm_weight_lm_head_weight = True

    def _init_lm_head_signmatch(self):
        from aicas2026gc.ops.lm_head_signmatch_argmax import (
            MAX_LOCAL_K,
            SignmatchArgmaxWorkspace,
            pack_lm_head_signs,
            signmatch_argmax_out,
        )

        rows_per_block = int(os.environ.get("LM_HEAD_SIGNMATCH_ROWS_PER_BLOCK", "512"))
        local_k = int(os.environ.get("LM_HEAD_SIGNMATCH_LOCAL_K", str(MAX_LOCAL_K)))
        if rows_per_block <= 0:
            raise ValueError(f"LM_HEAD_SIGNMATCH_ROWS_PER_BLOCK must be positive, got {rows_per_block}")
        if local_k <= 0 or local_k > MAX_LOCAL_K:
            raise ValueError(f"LM_HEAD_SIGNMATCH_LOCAL_K must be in [1, {MAX_LOCAL_K}], got {local_k}")
        if self.lm_head.weight.dtype != torch.float16:
            raise TypeError(f"LM_HEAD_SIGNMATCH requires float16 lm_head weight, got {self.lm_head.weight.dtype}")
        if not self.lm_head.weight.is_cuda:
            raise RuntimeError("LM_HEAD_SIGNMATCH requires lm_head weight on CUDA")

        self.lm_head_signmatch_rows_per_block = rows_per_block
        self.lm_head_signmatch_local_k = local_k
        self.lm_head_signmatch_packed_signs = pack_lm_head_signs(self.lm_head.weight)
        self.lm_head_signmatch_workspace = SignmatchArgmaxWorkspace(
            self.lm_head.weight.shape[0],
            self.lm_head.weight.device,
            rows_per_block=rows_per_block,
        )
        self.lm_head_signmatch_argmax_out = signmatch_argmax_out
        print(
            "[LMHeadSignmatch] enabled "
            f"allowed_vocab={self.lm_head.weight.shape[0]} "
            f"rows_per_block={rows_per_block} local_k={local_k}",
            flush=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = None,
        synced_gpus: bool | None = None,
        assistant_model: Optional[object] = None,
        streamer: Optional["BaseStreamer"] = None,  # noqa: F821
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        custom_generate: str | Callable | None = None,
        **kwargs,
    ) -> torch.LongTensor:

        # Check length values before updating the config with defaults. We'll use it later to define the final min/max length (# 6)
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)

        # 3. Define model inputs
        # inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        input_ids = model_kwargs.pop("input_ids", None)

        self._prepare_special_tokens(generation_config, True, device=input_ids.device)

        model_kwargs["position_ids"] = self._prepare_position_ids_for_generation(input_ids, model_kwargs)

        input_ids_length = input_ids.shape[1]
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=True,
            has_default_min_length=True,
            model_input_name="input_name",
            inputs_tensor=input_ids,
            input_ids_length=input_ids_length,
        )

        result = self._sample(
            input_ids,
            generation_config=generation_config,
            **model_kwargs,
        )

        return result

    def _prefill(
        self,
        input_ids: torch.LongTensor,
        model_kwargs: dict,
    ):

        model_inputs = self.prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=input_ids.shape[1],
            is_first_iteration=True,
            **model_kwargs,
        )
        return self.__call__(**model_inputs, return_dict=True)

    def _decode_one_step(self, input_ids: torch.LongTensor, seq_len: int, model_kwargs: dict):
        model_inputs = self.prepare_inputs_for_generation(input_ids[:, :seq_len], next_sequence_length=1, **model_kwargs)
        return self.__call__(**model_inputs, return_dict=True, is_decoding=True)

    def _decode_lm_head_argmax(self, hidden_states: torch.Tensor):
        candidates = {
            "exact": lambda hs: lm_head_argmax(hs, self.lm_head.weight, self.lm_head_token_id_map),
            f"signmatch_r{self.lm_head_signmatch_rows_per_block}_k{self.lm_head_signmatch_local_k}": lambda hs: self.lm_head_signmatch_argmax_out(
                hs[:, -1:, :],
                self.lm_head.weight,
                self.lm_head_signmatch_packed_signs,
                self.lm_head_signmatch_workspace,
                rows_per_block=self.lm_head_signmatch_rows_per_block,
                local_k=self.lm_head_signmatch_local_k,
                token_id_map=self.lm_head_token_id_map,
            ),
        }
        return self.lm_head_argmax_tuner.dispatch_once(candidates, name="decode_lm_head_argmax", args=(hidden_states,))

    @torch.inference_mode()
    def _sample(
        self,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        **model_kwargs,
    ):
        # init values
        max_length = generation_config.max_length
        eos_token_ids = set(generation_config.eos_token_id)

        input_ids_buffer = self.input_ids_buffer
        seq_len = input_ids.shape[1]
        initial_seq_len = seq_len
        input_ids_buffer[:, :seq_len] = input_ids

        prefill_consumed = False
        outputs = self._prefill(
            input_ids,
            model_kwargs,
        )

        copy_done_event = torch.cuda.Event()
        has_pending_copy = False

        while True:
            if prefill_consumed:
                outputs = self._decode_one_step(input_ids_buffer, seq_len, model_kwargs)
            prefill_consumed = True
            model_kwargs["position_ids"] = self.position_ids_buffer[:, :, : seq_len + 1]

            next_tokens = self._decode_lm_head_argmax(outputs.hidden_states)
            input_ids_buffer[:, seq_len] = next_tokens
            self.model.language_model.record_layer_stats_token(
                next_tokens=next_tokens,
                all_layers_hidden_states=outputs.all_layers_hidden_states,
                final_hidden_states=outputs.hidden_states,
                lm_head_weight=self.lm_head.weight,
                token_id_map=self.lm_head_token_id_map,
            )

            del outputs

            if has_pending_copy:
                copy_done_event.synchronize()
                if self.pinned_token_cpu_buffer[0].item() in eos_token_ids:
                    break

            self.pinned_token_cpu_buffer.copy_(next_tokens.view(-1), non_blocking=True)
            copy_done_event.record(torch.cuda.current_stream())
            has_pending_copy = True

            seq_len += 1
            if seq_len >= max_length:
                break

        self.model.language_model.end_layer_stats_sample(generated_tokens=seq_len - initial_seq_len)
        return input_ids_buffer[:, :seq_len]

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3VLCausalLMOutputWithPast:

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            **kwargs,
        )

        return Qwen3VLCausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=None,
            hidden_states=outputs[0],
            attentions=None,
            rope_deltas=None,
            all_layers_hidden_states=outputs.all_layers_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        position_ids=None,
        is_first_iteration=False,
        next_sequence_length=None,
        **kwargs,
    ):

        model_inputs = {}
        sequence_length = next_sequence_length

        model_inputs["input_ids"] = input_ids[:, -sequence_length:]
        model_inputs["position_ids"] = position_ids[..., -sequence_length:]

        model_inputs |= kwargs

        if not is_first_iteration:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        input_length = inputs_tensor.shape[-1]
        model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
        self.model.get_rope_index(inputs_tensor, position_ids_buffer=self.position_ids_buffer[1:, :, :input_length], **model_kwargs)

        part = torch.arange(1, 4096 - input_length + 1, dtype=torch.long, device=inputs_tensor.device)
        self.position_ids_buffer[1:, :, input_length:].copy_(self.position_ids_buffer[1:, :, input_length - 1, None] + part)

        return self.position_ids_buffer[:, :, :input_length]


__all__ = [
    "Qwen3VLVisionModel",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLTextModel",
]
