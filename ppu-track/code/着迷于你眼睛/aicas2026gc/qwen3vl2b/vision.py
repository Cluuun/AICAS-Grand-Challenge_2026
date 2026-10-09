import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLVisionConfig,
)
from transformers.activations import ACT2FN
from aicas2026gc import RuntimeAutotuner
from aicas2026gc.timing import timing_end, timing_start
from flash_attn import flash_attn_func
from .basemodel import BaseModelOutputWithDeepstackFeatures, Qwen3VLPreTrainedModel
from aicas2026gc.utils import shape_cache_decorator
from einops import rearrange
from transformers.utils.output_capturing import capture_outputs
from aicas2026gc.ops.fused_linear_add import fused_linear_add
from aicas2026gc.ops.fused_linear_gelu import fused_linear_gelu


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "t", "yes", "on")


def _parse_vision_int8_ops(value: str) -> set[str]:
    raw = value.strip().lower()
    if raw in ("", "0", "false", "f", "no", "off"):
        return set()
    valid = {"merger", "mlp_fc2"}
    if raw in ("1", "true", "t", "yes", "on", "all", "both"):
        return valid
    aliases = {
        "fc2": "mlp_fc2",
        "mlp": "mlp_fc2",
        "vision_mlp": "mlp_fc2",
        "vision_mlp_fc2": "mlp_fc2",
    }
    ops = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return {aliases.get(op, op) for op in ops if aliases.get(op, op) in valid}


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size  # 16
        self.temporal_patch_size = config.temporal_patch_size  # 2
        self.in_channels = config.in_channels  # 3
        self.embed_dim = config.hidden_size  # 1024

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]  # [2, 16, 16]
        self.proj = nn.Conv3d(
            self.in_channels,  # 3
            self.embed_dim,  # 1024
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor, pos_embeds: torch.Tensor = None) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype

        # [4096, 1536] -> [4096, 3, 2, 16, 16]
        # hidden_states = hidden_states.view(
        #     -1,
        #     self.in_channels,  # 3
        #     self.temporal_patch_size,  # 2
        #     self.patch_size,  # 16
        #     self.patch_size,  # 16
        # )
        # hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        # return hidden_states + pos_embeds

        # 权重从 [1024, 3, 2, 16, 16] 原地变为 [1024, 1536] (Zero-copy)
        weight_flat = self.proj.weight.view(self.embed_dim, -1)
        # [N, 1536] @ [1536, 1024] + [1024] + [N, 1024] -> [N, 1024]
        # return F.linear(hidden_states.to(dtype=target_dtype), weight_flat, self.proj.bias) + pos_embeds
        return fused_linear_add(pos_embeds, hidden_states, weight_flat, self.proj.bias)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seqlen = 128
        # Qwen3VLVisionModel.rot_pos_emb 中, max_hw 最大 64, 这里直接给 128
        #     self.rotary_pos_emb: Qwen3VLVisionRotaryEmbedding
        #     max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        #     freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        seq = torch.arange(self.max_seqlen, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(seq, inv_freq)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def forward(self, seqlen) -> torch.Tensor:
        raise NotImplementedError
        # seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs = torch.outer(seq, self.inv_freq)
        # return freqs


# 在 Qwen3VLVisionModel 中做中间的 self.deepstack_merger （真会起名字hh）和最终的 self.merger
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)  # 1024 * (2^2) -> 4096
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)  # 4096 or 1024
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)  # 4096
        self.act_fn = nn.GELU(approximate="tanh")
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)  # 4096 -> 2048
        self.fc1_int8 = None
        self.fc2_int8 = None
        self._int8_checked = False

    def _maybe_init_int8(self):
        if self._int8_checked:
            return
        self._int8_checked = True
        ops = _parse_vision_int8_ops(os.environ.get("VISION_INT8_LINEAR", "0"))
        if "merger" not in ops:
            return

        from aicas2026gc.ops.int8_linear_cublaslt import CublasLtInt8Linear

        workspace_mb = int(os.environ.get("VISION_INT8_WORKSPACE_MB", "32"))
        max_algos = int(os.environ.get("VISION_INT8_MAX_ALGOS", "8"))
        max_m = int(os.environ.get("VISION_INT8_MERGER_MAX_M", "1050"))
        autotune = _env_flag("VISION_INT8_AUTOTUNE", "1")
        suffix = "postshuffle" if self.use_postshuffle_norm else "final"
        self.fc1_int8 = CublasLtInt8Linear(
            self.linear_fc1.weight,
            self.linear_fc1.bias,
            name=f"vision_merger_{suffix}_fc1",
            workspace_mb=workspace_mb,
            max_algos=max_algos,
            max_m=max_m,
            autotune=autotune,
        )
        self.fc2_int8 = CublasLtInt8Linear(
            self.linear_fc2.weight,
            self.linear_fc2.bias,
            name=f"vision_merger_{suffix}_fc2",
            workspace_mb=workspace_mb,
            max_algos=max_algos,
            max_m=max_m,
            autotune=autotune,
        )
        print(
            f"[VisionINT8Merger] kind={suffix} max_m={max_m} max_algos={max_algos} "
            f"workspace_mb={workspace_mb} autotune={int(autotune)}",
            flush=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init_int8()
        if self.use_postshuffle_norm:
            x = x.view(-1, self.hidden_size)
        x = self.norm(x).view(-1, self.hidden_size)
        if self.fc1_int8 is None:
            x = self.linear_fc1(x)
        else:
            x = self.fc1_int8(x)
        x = self.act_fn(x)
        if self.fc2_int8 is None:
            x = self.linear_fc2(x)
        else:
            x = self.fc2_int8(x)
        return x


# 在 Qwen3VLVisionBlock 中，attn(Qwen3VLVisionAttention) + mlp(Qwen3VLVisionMLP)
class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)  # 这里需要和 Qwen3VLVisionBlock.norm1 的参数整合
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def origin_forward(
        self,
        hidden_states: torch.Tensor,  # [seq_len, hidden_dim]
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:

        origin_hidden_states = hidden_states
        seq_length, _ = hidden_states.shape

        # TODO: layernorm + qkv + rope 可以融合
        hidden_states = F.layer_norm(hidden_states, [hidden_states.shape[-1]], eps=1e-6)
        query_states, key_states, value_states = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).unbind(1)
        cos, sin = position_embeddings
        query_states, key_states = torch.ops.qwen3vl.apply_rotary_pos_emb_vision_triton(query_states, key_states, cos, sin, inplace=False)

        # TODO: 这里的特征是否可以压缩
        attn_output = flash_attn_func(query_states[None], key_states[None], value_states[None], causal=self.is_causal, softmax_scale=self.scaling).view(
            seq_length, -1
        )  # [seq_len, hidden_dim]

        # origin_hidden_states: [4096, 1024], attn_output: [4096, 1024], self.proj.weight: [1024, 1024]
        # return self.proj(attn_output) + origin_hidden_states
        return fused_linear_add(origin_hidden_states, attn_output, self.proj.weight, self.proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [seq_len, hidden_dim]
        cu_seqlens: torch.Tensor,  # 1d
        rotary_pos_emb: torch.Tensor | None = None,  # None
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.origin_forward(hidden_states, position_embeddings)


# 在 Qwen3VLVisionBlock 中，attn(Qwen3VLVisionAttention) + mlp(Qwen3VLVisionMLP)
class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size  # 1024
        self.intermediate_size = config.intermediate_size  # 4096
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]  # gelu_pytorch_tanh
        self.fc2_int8 = None
        self._int8_checked = False

    def _maybe_init_int8(self):
        if self._int8_checked:
            return
        self._int8_checked = True
        ops = _parse_vision_int8_ops(os.environ.get("VISION_INT8_LINEAR", "0"))
        if "mlp_fc2" not in ops:
            return

        from aicas2026gc.ops.int8_linear_cublaslt import CublasLtInt8Linear

        workspace_mb = int(os.environ.get("VISION_INT8_WORKSPACE_MB", "32"))
        max_algos = int(os.environ.get("VISION_INT8_MAX_ALGOS", "8"))
        max_m = int(os.environ.get("VISION_INT8_MLP_MAX_M", "4096"))
        autotune = _env_flag("VISION_INT8_AUTOTUNE", "1")
        self.fc2_int8 = CublasLtInt8Linear(
            self.linear_fc2.weight,
            self.linear_fc2.bias,
            name="vision_mlp_fc2_residual",
            workspace_mb=workspace_mb,
            max_algos=max_algos,
            max_m=max_m,
            autotune=autotune,
        )
        print(
            f"[VisionINT8MLP] op=fc2_residual max_m={max_m} max_algos={max_algos} "
            f"workspace_mb={workspace_mb} autotune={int(autotune)}",
            flush=True,
        )

    def forward(self, hidden_state):  # [4096, 1024]
        self._maybe_init_int8()
        origin_hidden_state = hidden_state  # [seq_len, hidden_size]
        hidden_state = F.layer_norm(hidden_state, [hidden_state.shape[-1]], eps=1e-6)  # 1024
        # activated = self.act_fn(self.linear_fc1(hidden_state))  # [seq_len, intermediate_size]
        activated = fused_linear_gelu(hidden_state, self.linear_fc1.weight, self.linear_fc1.bias)
        # out = origin_hidden_state + self.linear_fc2(activated)
        if self.fc2_int8 is None:
            out = fused_linear_add(origin_hidden_state, activated, self.linear_fc2.weight, self.linear_fc2.bias)
        else:
            out = self.fc2_int8(activated, residual=origin_hidden_state)
        return out


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        # 这俩 LayerNorm 的参数已经融合到 self.attn 和 self.mlp 中了，无参数的计算部分也在其 forward 中
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)  # 1024
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)  # 1024
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)
        self.has_post_init_fusion = False

    @torch.no_grad()
    def post_init_fusion(self):

        if self.has_post_init_fusion:
            return

        gamma = self.norm1.weight  # [in_features]
        beta = self.norm1.bias  # [in_features]
        W = self.attn.qkv.weight  # [out_features, in_features]
        b = self.attn.qkv.bias  # [out_features]
        W_fused = W * gamma
        b_fused = torch.matmul(W, beta) + b
        self.attn.qkv.weight.copy_(W_fused)
        self.attn.qkv.bias.copy_(b_fused)

        gamma = self.norm2.weight  # [in_features]
        beta = self.norm2.bias  # [in_features]
        W = self.mlp.linear_fc1.weight  # [out_features, in_features]
        b = self.mlp.linear_fc1.bias  # [out_features]
        W_fused = W * gamma
        b_fused = torch.matmul(W, beta) + b
        self.mlp.linear_fc1.weight.copy_(W_fused)
        self.mlp.linear_fc1.bias.copy_(b_fused)

        self.norm1.weight = None
        self.norm1.bias = None

        self.norm2.weight = None
        self.norm2.bias = None

        self.has_post_init_fusion = True

    def forward(
        self,
        hidden_states: torch.Tensor,  # [seq_len, hidden_dim]
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,  # None
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=None,
            rotary_pos_emb=None,
            position_embeddings=position_embeddings,
            attn_mask=attn_mask,
        )

        return self.mlp(hidden_states)


class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    config: Qwen3VLVisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["Qwen3VLVisionBlock"]
    _can_record_outputs = {
        "hidden_states": Qwen3VLVisionBlock,
        "attentions": Qwen3VLVisionAttention,
    }

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

        # ============ Vision CUDA Graph (精确 seq_len, 不分桶) ============
        MAX_VIS_SEQ = 4096
        vis_rope_dim = 32
        self.max_vis_seq = MAX_VIS_SEQ

        # 精确 seq_len 列表，每个独立 capture，不命中走 eager
        # fmt: off
        self.vis_exact_seq_lens: list[int] = [
            4096, 3968, 3840, 3712, 3584, 3456, 3328, 3200, 3136, 3120, 3100, 3072, 3024, 3016, 2976,
            2944, 2880, 2816, 2784, 2688, 2592, 2560, 2496, 2432, 2400,
            2304, 2176, 2048, 1920, 1792, 1664, 1536, 1408, 1280, 1024,
        ]
        # fmt: on
        self._vis_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._vis_graph_outputs: dict[int, tuple] = {}

        # 静态 input buffer（按 max 分配）
        in_pixel_dim = config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size  # 1536
        self.register_buffer("vis_input_buf", torch.zeros(MAX_VIS_SEQ, in_pixel_dim), persistent=False)
        self.register_buffer("vis_pos_buf", torch.zeros(MAX_VIS_SEQ, config.hidden_size), persistent=False)
        self.register_buffer("vis_cos_buf", torch.zeros(MAX_VIS_SEQ, vis_rope_dim), persistent=False)
        self.register_buffer("vis_sin_buf", torch.zeros(MAX_VIS_SEQ, vis_rope_dim), persistent=False)

        self._vis_mempool = None
        self._vis_graphs_captured = False
        self.using_cudagraph = os.environ.get("USE_CUDAGRAPH", "1").lower() in ["1", "true", "yes"]

        self.post_init()

    @shape_cache_decorator
    def elegant_rot_pos_emb(self, grid_thw_list) -> torch.Tensor:
        merge_size = self.spatial_merge_size  # 2
        freq_table_cos = self.rotary_pos_emb.freqs_cos  # [128, 16]
        freq_table_sin = self.rotary_pos_emb.freqs_sin  # [128, 16]
        device = freq_table_cos.device

        pos_ids_list = []

        for t, h, w in grid_thw_list:
            h_grid, w_grid = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            h_grid = rearrange(h_grid, "(h m1) (w m2) -> (h w m1 m2)", m1=merge_size, m2=merge_size)
            w_grid = rearrange(w_grid, "(h m1) (w m2) -> (h w m1 m2)", m1=merge_size, m2=merge_size)
            coords = torch.stack((h_grid, w_grid), dim=-1)  # shape: [h*w, 2]

            if t > 1:
                coords = coords.unsqueeze(0).expand(t, -1, -1).reshape(-1, 2)

            pos_ids_list.append(coords)

        # pos_ids = torch.cat(pos_ids_list, dim=0) # BS=1
        pos_ids = pos_ids_list[0]
        return freq_table_cos[pos_ids].flatten(1), freq_table_sin[pos_ids].flatten(1)

    @shape_cache_decorator
    def real_fast_pos_embed_interpolate(self, grid_thw_list):  # [bs, 3]

        merge_size = self.config.spatial_merge_size  # 2
        grid_size = self.num_grid_per_side  # 40

        # [H*W, C] -> [1, C, H, W]
        # [2304, 1024] -> [1, 1024, 48, 48]
        base_embed = self.pos_embed.weight.view(1, grid_size, grid_size, -1).permute(0, 3, 1, 2)

        # grid_thw_list = grid_thw.tolist()
        patch_pos_embeds_list = []

        for t, h, w in grid_thw_list:
            # align_corners=True 等价于原版的 torch.linspace(0, 47, h)
            interp = F.interpolate(base_embed, size=(h, w), mode="bilinear", align_corners=True)  # [1, C, h, w]

            interp = rearrange(
                interp,
                "b c (h m1) (w m2) -> b h w m1 m2 c",
                m1=merge_size,
                m2=merge_size,
            )

            if t > 1:
                interp = interp.expand(t, -1, -1, -1, -1, -1)
            patch_pos_embeds_list.append(interp.reshape(-1, interp.shape[-1]))

        # return torch.cat(patch_pos_embeds_list, dim=0)
        return patch_pos_embeds_list[0]

    def _vision_compute(self, hidden_states, pos_embeds, rotary_pos_emb_cos, rotary_pos_emb_sin):
        hidden_states = self.patch_embed(hidden_states, pos_embeds)  # [seq_len, 1536] -> [seq_len, 1024]
        position_embeddings = rotary_pos_emb_cos, rotary_pos_emb_sin

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=None,
                position_embeddings=position_embeddings,
                attn_mask=None,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)
        return hidden_states, merged_hidden_states, deepstack_feature_lists

    def _run_vision_eager(self, hidden_states, pos_embeds, rotary_pos_emb_cos, rotary_pos_emb_sin):
        return self._vision_compute(hidden_states, pos_embeds, rotary_pos_emb_cos, rotary_pos_emb_sin)

    def _run_vision_cudagraph(self, hidden_states, pos_embeds, rotary_pos_emb_cos, rotary_pos_emb_sin):
        seq_len = hidden_states.shape[0]

        self.vis_input_buf[:seq_len].copy_(hidden_states)
        self.vis_pos_buf[:seq_len].copy_(pos_embeds)
        self.vis_cos_buf[:seq_len].copy_(rotary_pos_emb_cos)
        self.vis_sin_buf[:seq_len].copy_(rotary_pos_emb_sin)

        if not self._vis_graphs_captured:
            self._capture_all_vision_graphs()

        if seq_len in self._vis_graphs:
            self._vis_graphs[seq_len].replay()
            return self._vis_graph_outputs[seq_len]

        print(f"[VisionCUDAGraph] eager fallback for seq_len={seq_len}")
        return self._run_vision_eager(hidden_states, pos_embeds, rotary_pos_emb_cos, rotary_pos_emb_sin)

    def _build_vision_output(self, out_hidden, out_merged, out_deepstack):
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=out_hidden,
            pooler_output=out_merged,
            deepstack_features=list(out_deepstack),
        )

    def _capture_all_vision_graphs(self):
        if self._vis_graphs_captured:
            return

        print(f"[VisionCUDAGraph] pre-capturing {len(self.vis_exact_seq_lens)} exact seq_lens")

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(s):
            for seq_len in self.vis_exact_seq_lens:
                inp = self.vis_input_buf[:seq_len]
                pos = self.vis_pos_buf[:seq_len]
                cos = self.vis_cos_buf[:seq_len]
                sin = self.vis_sin_buf[:seq_len]

                for _ in range(3):
                    self._run_vision_eager(inp, pos, cos, sin)

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._vis_mempool):
                    out_hidden, out_merged, out_deepstack = self._run_vision_eager(inp, pos, cos, sin)
                if self._vis_mempool is None:
                    self._vis_mempool = g.pool()

                self._vis_graphs[seq_len] = g
                self._vis_graph_outputs[seq_len] = (out_hidden, out_merged, out_deepstack)

        torch.cuda.current_stream().wait_stream(s)
        self._vis_graphs_captured = True
        print(f"[VisionCUDAGraph] all {len(self._vis_graphs)} graphs captured!")

    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_embeds: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        timing_token = timing_start("vision")
        if not self.using_cudagraph:
            outputs = self._run_vision_eager(hidden_states, pos_embeds, rotary_pos_emb_cos, rotary_pos_emb_sin)
        else:
            outputs = self._run_vision_cudagraph(hidden_states, pos_embeds, rotary_pos_emb_cos, rotary_pos_emb_sin)

        timing_end(timing_token)
        return self._build_vision_output(*outputs)
