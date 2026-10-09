import importlib
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import time
import torch.nn as nn
import triton
import triton.language as tl
from typing import Callable, Optional
from pathlib import Path
from datasets import load_from_disk
# from evaluation_wrapper import VLMModel
import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling_qwen3_vl
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLVisionModel,
    Qwen3VLTextModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLCausalLMOutputWithPast,
    BaseModelOutputWithDeepstackFeatures,
    capture_outputs,
    GradientCheckpointingLayer,
    Qwen3VLVisionMLP,
    ACT2FN,
    use_kernelized_func,
    apply_rotary_pos_emb,
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    merge_with_config_defaults,
    apply_rotary_pos_emb_vision,
    repeat_kv
    )
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load as load_torch_extension
import os
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"

_flashinfer = None
if os.environ.get("USE_FLASHINFER_DECODE_ATTENTION", "").lower() in {"1", "true", "yes", "on"}:
    try:
        import flashinfer as _flashinfer
    except Exception:
        _flashinfer = None

from cache import ASStaticCache
import os

torch.backends.cuda.enable_flash_sdp(True)

# torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
torch.backends.cuda.preferred_blas_library("cublaslt")
torch.set_grad_enabled(False)

from transformers.generation.utils import *

_LEAN_DECODE_CUDA_EXT = None
_LEAN_DECODE_CUDA_LOAD_ATTEMPTED = False
_LEAN_DECODE_CUDA_LOAD_ERROR = None
_DECODE_FRONT_CUDA_EXT = None
_DECODE_FRONT_CUDA_LOAD_ATTEMPTED = False
_DECODE_FRONT_CUDA_LOAD_ERROR = None


def _import_prebuilt_extension(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        return exc


def _load_lean_decode_cuda_extension():
    global _LEAN_DECODE_CUDA_EXT
    global _LEAN_DECODE_CUDA_LOAD_ATTEMPTED
    global _LEAN_DECODE_CUDA_LOAD_ERROR

    if _LEAN_DECODE_CUDA_LOAD_ATTEMPTED:
        return _LEAN_DECODE_CUDA_EXT

    _LEAN_DECODE_CUDA_LOAD_ATTEMPTED = True
    if not torch.cuda.is_available():
        return None

    prebuilt_extension = _import_prebuilt_extension("lean_decode_attention_sm80_fp16_v1")
    if not isinstance(prebuilt_extension, Exception):
        _LEAN_DECODE_CUDA_EXT = prebuilt_extension
        return _LEAN_DECODE_CUDA_EXT

    source_dir = Path(__file__).resolve().parent
    cpp_source = source_dir / "lean_decode_attention_cuda.cpp"
    cuda_source = source_dir / "lean_decode_attention_cuda_kernel.cu"
    if not (cpp_source.exists() and cuda_source.exists()):
        _LEAN_DECODE_CUDA_LOAD_ERROR = FileNotFoundError(
            f"Missing lean decode CUDA sources: {cpp_source.name}, {cuda_source.name}; "
            f"prebuilt import error: {prebuilt_extension}"
        )
        return None

    if "CUDA_HOME" not in os.environ:
        for candidate in ("/usr/local/cuda", "/usr/local/cuda-12.8"):
            if (Path(candidate) / "bin" / "nvcc").exists():
                os.environ["CUDA_HOME"] = candidate
                break
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

    try:
        _LEAN_DECODE_CUDA_EXT = load_torch_extension(
            name="lean_decode_attention_sm80_fp16_v1",
            sources=[str(cpp_source), str(cuda_source)],
            with_cuda=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-lineinfo",
                "-gencode=arch=compute_80,code=sm_80",
            ],
            verbose=os.environ.get("LEAN_DECODE_CUDA_VERBOSE", "").lower() in {"1", "true", "yes", "on"},
        )
    except Exception as exc:
        _LEAN_DECODE_CUDA_LOAD_ERROR = exc
        print(f"[lean-decode-cuda] load failed, fallback to Triton path: {exc}")
        return None

    return _LEAN_DECODE_CUDA_EXT


def _load_decode_front_cuda_extension():
    global _DECODE_FRONT_CUDA_EXT
    global _DECODE_FRONT_CUDA_LOAD_ATTEMPTED
    global _DECODE_FRONT_CUDA_LOAD_ERROR

    if _DECODE_FRONT_CUDA_LOAD_ATTEMPTED:
        return _DECODE_FRONT_CUDA_EXT

    _DECODE_FRONT_CUDA_LOAD_ATTEMPTED = True
    if not torch.cuda.is_available():
        return None

    prebuilt_extension = _import_prebuilt_extension("decode_attention_front_sm80_fp16_v1")
    if not isinstance(prebuilt_extension, Exception):
        _DECODE_FRONT_CUDA_EXT = prebuilt_extension
        return _DECODE_FRONT_CUDA_EXT

    source_dir = Path(__file__).resolve().parent
    cpp_source = source_dir / "decode_attention_front_cuda.cpp"
    cuda_source = source_dir / "decode_attention_front_cuda_kernel.cu"
    if not (cpp_source.exists() and cuda_source.exists()):
        _DECODE_FRONT_CUDA_LOAD_ERROR = FileNotFoundError(
            f"Missing decode front CUDA sources: {cpp_source.name}, {cuda_source.name}; "
            f"prebuilt import error: {prebuilt_extension}"
        )
        return None

    if "CUDA_HOME" not in os.environ:
        for candidate in ("/usr/local/cuda", "/usr/local/cuda-12.8"):
            if (Path(candidate) / "bin" / "nvcc").exists():
                os.environ["CUDA_HOME"] = candidate
                break
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

    try:
        _DECODE_FRONT_CUDA_EXT = load_torch_extension(
            name="decode_attention_front_sm80_fp16_v1",
            sources=[str(cpp_source), str(cuda_source)],
            with_cuda=True,
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "-lineinfo",
                "-gencode=arch=compute_80,code=sm_80",
            ],
            verbose=os.environ.get("DECODE_FRONT_CUDA_VERBOSE", "").lower() in {"1", "true", "yes", "on"},
        )
    except Exception as exc:
        _DECODE_FRONT_CUDA_LOAD_ERROR = exc
        print(f"[decode-front-cuda] load failed, fallback to Triton path: {exc}")
        return None

    return _DECODE_FRONT_CUDA_EXT

"""start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

end_event.record()
        torch.cuda.synchronize()
        print(start_event.elapsed_time(end_event)/1000)"""


@triton.jit
def _rmsnorm_forward_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride

    values = tl.load(input_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(values * values, axis=0) / n_cols
    inv_rms = tl.rsqrt(variance + eps)
    normalized = values * inv_rms * weights
    tl.store(output_row_ptr + cols, normalized, mask=mask)

def TextRMS_Forward(self, hidden_states):
    n_cols = hidden_states.shape[-1]
    hidden_states_2d = hidden_states.reshape(-1, n_cols)
    output = torch.empty_like(hidden_states_2d)

    block_size = triton.next_power_of_2(n_cols)
    if n_cols <= 128:
        num_warps = 2
    elif n_cols <= 1024:
        num_warps = 4
    else:
        num_warps = 8

    _rmsnorm_forward_kernel[(hidden_states_2d.shape[0],)](
        hidden_states_2d,
        self.weight,
        output,
        hidden_states_2d.stride(0),
        output.stride(0),
        n_cols,
        self.variance_epsilon,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output.view_as(hidden_states)

@merge_with_config_defaults
@capture_outputs
def Qwen3VLVisionModel_Forward(
    self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs
):
    # if not hasattr(self, "cache")
    hidden_states = self.patch_embed(hidden_states)

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    
    hidden_states = hidden_states + pos_embeds

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()

    deepstack_feature_lists = []
    def LiftTensor(
        hidden_states_,
        cu_seqlens_,
        position_embeddings_,
    ):
        hidden_states_[0].copy_(hidden_states_[1])
        cu_seqlens_[0].copy_(cu_seqlens_[1])
        position_embeddings_[0][0].copy_(position_embeddings_[1][0])
        position_embeddings_[0][1].copy_(position_embeddings_[1][1])

    # GRAPH
    lengths = [hidden_states.shape[0]]
    if hasattr(self, "shapes") and hidden_states.shape[0] in self.shapes:
        idx = self.shapes.index(hidden_states.shape[0])
        buffer = self.buffers[idx]
        LiftTensor(
            [buffer["hidden_states"], hidden_states],
            [buffer["cu_seqlens"], cu_seqlens],
            [buffer["position_embeddings"], position_embeddings]
        )
        buffer["graph"].replay()
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=None,
            pooler_output=self.merger(buffer["graph_outputs"].clone()),
            deepstack_features=deepstack_feature_lists,
        )
    elif not self.isInferStage:
        if not hasattr(self, "shapes"):
            self.shapes = []
            self.buffers = []
        if len(self.shapes) > 6: # STOP!
            self.isInferStage = True
        self.shapes.append(hidden_states.shape[0])
        graph = torch.cuda.CUDAGraph()
        graph_hidden_states = hidden_states
        graph_cu_seqlens = cu_seqlens
        graph_outputs = hidden_states.clone()
        graph_position_embeddings_0 = position_embeddings[0].clone()
        graph_position_embeddings_1 = position_embeddings[1].clone()
        buffer = {
            "hidden_states": graph_hidden_states,
            "cu_seqlens": graph_cu_seqlens,
            "graph": graph,
            "lengths": lengths,
            "kwargs": kwargs,
            "position_embeddings": (graph_position_embeddings_0, graph_position_embeddings_1),
            "graph_outputs": graph_outputs
        }
        self.buffers.append(buffer)
        with torch.cuda.graph(buffer["graph"]):
            for blk in self.blocks:
                buffer["hidden_states"] = blk(
                    buffer["hidden_states"],
                    cu_seqlens=buffer["cu_seqlens"],
                    position_embeddings=buffer["position_embeddings"],
                    lengths_ilovetakanashihoshino=buffer["lengths"],
                    **buffer["kwargs"],
                )
            buffer["graph_outputs"].copy_(buffer["hidden_states"])
        buffer["graph"].replay()

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=None,
            pooler_output=self.merger(buffer["graph_outputs"].clone()),
            deepstack_features=deepstack_feature_lists,
        )
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    merged_hidden_states = self.merger(hidden_states.clone())

    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
        deepstack_features=deepstack_feature_lists,
    )

def fused_qkv_proj_norm(hidden_states, q_proj, k_proj, v_proj, q_norm, k_norm, hidden_shape):
    query_states = q_norm(q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    return query_states, key_states, value_states

@triton.jit
def _silu_mul_kernel(gate_ptr,up_ptr,output_ptr,numel,BLOCK: tl.constexpr,):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    silu_gate = gate * tl.sigmoid(gate)
    tl.store(output_ptr + offsets, silu_gate * up, mask=mask)

# Want a clean toilet?
# class Toilet:
#     def __init__(self):
#         self.rooms = {}
#     def enter(self, shape, name, dtype = None):
#         room = (name, tuple(shape))
#         if room in self.rooms:
#             if room["used"]:
#                 return torch.empty(shape, device=self.device, dtype=dtype, **kwargs)
#             room = self.rooms["room"]
#             room["used"] = True
#             return room["permission"]
            
#     def leave(
#     def pee(self, )
    
def fused_silu_mul(gate_states, up_states):
    output = torch.empty_like(gate_states)
    gate_flat = gate_states.contiguous().view(-1)
    up_flat = up_states.contiguous().view(-1)
    output_flat = output.view(-1)

    block_size = 1024
    grid = (triton.cdiv(gate_flat.numel(), block_size),)
    _silu_mul_kernel[grid](
        gate_flat,
        up_flat,
        output_flat,
        gate_flat.numel(),
        BLOCK=block_size,
        num_warps=4,
        num_stages=1
    )
    return output

class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def _ready(self):
        self.gate_up_proj = nn.Linear(self.hidden_size,self.intermediate_size * 2,bias=False,device=self.gate_proj.weight.device,dtype=self.gate_proj.weight.dtype)
        with torch.no_grad():
            self.gate_up_proj.weight.copy_(torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0))

    def forward(self, x):
        gate_states, up_states = self.gate_up_proj(x).chunk(2, dim=-1)
        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        fused = fused_silu_mul(gate_states, up_states)
        return self.down_proj(fused)

# import flashinfer
# class Qwen3VLTextMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = ACT2FN[config.hidden_act]

#     def _ready(self):
#         self.gate_up_proj = nn.Linear(self.hidden_size,self.intermediate_size * 2,bias=False,device=self.gate_proj.weight.device,dtype=self.gate_proj.weight.dtype)
#         with torch.no_grad():
#             self.gate_up_proj.weight.copy_(torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0))

#     def forward(self, x):
#         gate_up_out = self.gate_up_proj(x)
#         fused = flashinfer.silu_and_mul(gate_up_out)
#         return self.down_proj(fused)

@triton.jit
def _fused_headwise_rmsnorm_rotary_kernel(
    input_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    half_dim,
    eps,
    BLOCK_HALF: tl.constexpr
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_HALF)
    mask = offsets < half_dim

    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride

    x_first = tl.load(input_row_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_second = tl.load(input_row_ptr + offsets + half_dim, mask=mask, other=0.0).to(tl.float32)

    weight_first = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight_second = tl.load(weight_ptr + offsets + half_dim, mask=mask, other=0.0).to(tl.float32)
    cos_first = tl.load(cos_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cos_second = tl.load(cos_ptr + offsets + half_dim, mask=mask, other=0.0).to(tl.float32)
    sin_first = tl.load(sin_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sin_second = tl.load(sin_ptr + offsets + half_dim, mask=mask, other=0.0).to(tl.float32)

    variance = (tl.sum(x_first * x_first, axis=0) + tl.sum(x_second * x_second, axis=0)) / (half_dim * 2)
    inv_rms = tl.rsqrt(variance + eps)

    norm_first = x_first * inv_rms * weight_first
    norm_second = x_second * inv_rms * weight_second

    out_first = norm_first * cos_first - norm_second * sin_first
    out_second = norm_second * cos_second + norm_first * sin_second

    tl.store(output_row_ptr + offsets, out_first, mask=mask)
    tl.store(output_row_ptr + offsets + half_dim, out_second, mask=mask)

def fused_text_qk_rotary(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_eps: float,
    k_eps: float,
    head_dim: int
):
    q_heads = q_proj.shape[-1] // head_dim
    k_heads = k_proj.shape[-1] // head_dim
    half_dim = head_dim // 2

    block_half = triton.next_power_of_2(head_dim // 2)
    """
    If u r not sm80
    APPLY THIS =>
    1. num_warps = 2
    2. num_stages = 1
    """
    if head_dim <= 128:
        num_warps = 2
        if max(q_heads, k_heads) >= 16:
            num_stages = 2
        else:
            num_stages = 1
    else:
        num_warps, num_stages = 4, 2

    q_proj_2d = q_proj.view(q_heads, head_dim)
    k_proj_2d = k_proj.view(k_heads, head_dim)
    q_out = torch.empty((1, q_heads, 1, head_dim), device=q_proj.device, dtype=q_proj.dtype)
    k_out = torch.empty((1, k_heads, 1, head_dim), device=k_proj.device, dtype=k_proj.dtype)
    q_out_2d = q_out.view(q_heads, head_dim)
    k_out_2d = k_out.view(k_heads, head_dim)
    cos_flat = cos.view(head_dim)
    sin_flat = sin.view(head_dim)

    _fused_headwise_rmsnorm_rotary_kernel[(q_heads,)](
        q_proj_2d,
        q_weight,
        cos_flat,
        sin_flat,
        q_out_2d,
        q_proj_2d.stride(0),
        q_out_2d.stride(0),
        half_dim,
        q_eps,
        BLOCK_HALF=block_half,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _fused_headwise_rmsnorm_rotary_kernel[(k_heads,)](
        k_proj_2d,
        k_weight,
        cos_flat,
        sin_flat,
        k_out_2d,
        k_proj_2d.stride(0),
        k_out_2d.stride(0),
        half_dim,
        k_eps,
        BLOCK_HALF=block_half,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return q_out, k_out


def _can_use_decode_attention_front_cuda(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor | None,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> bool:
    tensors = (q_proj, k_proj, v_proj, q_weight, k_weight, cos, sin)
    if cache_position is None or cache_position.numel() != 1:
        return False
    if any(t is None for t in tensors):
        return False
    if not all(t.is_cuda for t in (*tensors, cache_position)):
        return False
    if not all(t.dtype == torch.float16 for t in tensors):
        return False
    if q_proj.shape != (1, 1, num_attention_heads * head_dim):
        return False
    if k_proj.shape != (1, 1, num_key_value_heads * head_dim):
        return False
    if v_proj.shape != (1, 1, num_key_value_heads * head_dim):
        return False
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        return False
    if cos.shape != (1, 1, head_dim) or sin.shape != (1, 1, head_dim):
        return False
    if any(t.stride(-1) != 1 for t in (q_proj, k_proj, v_proj, cos, sin)):
        return False
    if q_weight.stride(0) != 1 or k_weight.stride(0) != 1:
        return False
    if num_attention_heads != 16 or num_key_value_heads != 8 or head_dim != 128:
        return False
    try:
        return torch.cuda.get_device_capability(q_proj.device) == (8, 0)
    except Exception:
        return False


def fused_decode_attention_front_cuda(
    past_key_values,
    layer_idx: int,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cache_position: torch.Tensor,
    q_eps: float,
    k_eps: float,
    query_buffer: torch.Tensor,
):
    extension = _load_decode_front_cuda_extension()
    if extension is None:
        raise RuntimeError(_DECODE_FRONT_CUDA_LOAD_ERROR or "decode front CUDA extension unavailable")

    layer = past_key_values.layers[layer_idx]
    if not layer.is_initialized:
        dummy_k = k_proj.view(1, 1, -1, 128).transpose(1, 2)
        dummy_v = v_proj.view(1, 1, -1, 128).transpose(1, 2)
        layer.lazy_initialization(dummy_k, dummy_v)

    query_states = extension.forward(
        q_proj,
        k_proj,
        v_proj,
        q_weight,
        k_weight,
        cos,
        sin,
        layer.keys,
        layer.values,
        cache_position,
        float(q_eps),
        float(k_eps),
        query_buffer,
    )
    return query_states, layer.keys, layer.values

@triton.jit
def _text_rotary_inplace_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    x_stride_b,
    x_stride_h,
    x_stride_s,
    x_stride_d,
    cos_stride_b,
    cos_stride_s,
    cos_stride_d,
    sin_stride_b,
    sin_stride_s,
    sin_stride_d,
    num_heads,
    seq_len,
    half_dim,
    BLOCK_HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    rows_per_batch = num_heads * seq_len
    batch_idx = pid // rows_per_batch
    row_idx = pid % rows_per_batch
    head_idx = row_idx // seq_len
    seq_idx = row_idx % seq_len

    offsets = tl.arange(0, BLOCK_HALF)
    mask = offsets < half_dim

    x_row_ptr = x_ptr + batch_idx * x_stride_b + head_idx * x_stride_h + seq_idx * x_stride_s
    cos_row_ptr = cos_ptr + batch_idx * cos_stride_b + seq_idx * cos_stride_s
    sin_row_ptr = sin_ptr + batch_idx * sin_stride_b + seq_idx * sin_stride_s

    x_first = tl.load(x_row_ptr + offsets * x_stride_d, mask=mask, other=0.0).to(tl.float32)
    x_second = tl.load(
        x_row_ptr + (offsets + half_dim) * x_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    cos_first = tl.load(cos_row_ptr + offsets * cos_stride_d, mask=mask, other=0.0).to(tl.float32)
    cos_second = tl.load(
        cos_row_ptr + (offsets + half_dim) * cos_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sin_first = tl.load(sin_row_ptr + offsets * sin_stride_d, mask=mask, other=0.0).to(tl.float32)
    sin_second = tl.load(
        sin_row_ptr + (offsets + half_dim) * sin_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    out_first = x_first * cos_first - x_second * sin_first
    out_second = x_second * cos_second + x_first * sin_second

    tl.store(x_row_ptr + offsets * x_stride_d, out_first, mask=mask)
    tl.store(x_row_ptr + (offsets + half_dim) * x_stride_d, out_second, mask=mask)

def _apply_text_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    half_dim = q.shape[-1] // 2
    block_half = triton.next_power_of_2(half_dim)
    q_grid = (q.shape[0] * q.shape[1] * q.shape[2],)
    k_grid = (k.shape[0] * k.shape[1] * k.shape[2],)

    _text_rotary_inplace_kernel[q_grid](
        q,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        q.shape[1],
        q.shape[2],
        half_dim,
        BLOCK_HALF=block_half,
        num_warps=2,
        num_stages=2,
    )
    _text_rotary_inplace_kernel[k_grid](
        k,
        cos,
        sin,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        k.shape[1],
        k.shape[2],
        half_dim,
        BLOCK_HALF=block_half,
        num_warps=2,
        num_stages=2,
    )
    return q, k


def single_token_update(
    past_key_values,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    
    layer = past_key_values.layers[layer_idx]

    if not layer.is_initialized:
        layer.lazy_initialization(key_states, value_states)
    
    head_dim = key_states.shape[-1]
    num_heads = key_states.shape[1]

    block_d = min(triton.next_power_of_2(head_dim), 256)
    cache_position = cache_position.to(torch.int32)

    grid = (num_heads,)
    
    single_token_update_kernel[grid](
        key_states,
        value_states,
        layer.keys,
        layer.values,
        cache_position,
        key_states.stride(0),
        key_states.stride(1),
        key_states.stride(2),
        key_states.stride(3),
        value_states.stride(0),
        value_states.stride(1),
        value_states.stride(2),
        value_states.stride(3),
        layer.keys.stride(0),
        layer.keys.stride(1),
        layer.keys.stride(2),
        layer.keys.stride(3),
        layer.values.stride(0),
        layer.values.stride(1),
        layer.values.stride(2),
        layer.values.stride(3),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    
    return layer.keys, layer.values

@triton.jit
def single_token_update_kernel(
    k_input_ptr,
    v_input_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cache_pos_ptr,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    kcache_stride_b, kcache_stride_h, kcache_stride_s, kcache_stride_d,
    vcache_stride_b, vcache_stride_h, vcache_stride_s, vcache_stride_d,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // tl.num_programs(1)
    head_idx = pid % tl.num_programs(1)

    head_idx = pid
    batch_idx = 0

    write_pos = tl.load(cache_pos_ptr)

    k_in_offset = batch_idx * k_stride_b + head_idx * k_stride_h
    v_in_offset = batch_idx * v_stride_b + head_idx * v_stride_h

    k_cache_offset = batch_idx * kcache_stride_b + head_idx * kcache_stride_h + write_pos * kcache_stride_s
    v_cache_offset = batch_idx * vcache_stride_b + head_idx * vcache_stride_h + write_pos * vcache_stride_s

    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < HEAD_DIM

    k_vals = tl.load(k_input_ptr + k_in_offset + offsets * k_stride_d, mask=mask)
    tl.store(k_cache_ptr + k_cache_offset + offsets * kcache_stride_d, k_vals, mask=mask)

    v_vals = tl.load(v_input_ptr + v_in_offset + offsets * v_stride_d, mask=mask)
    tl.store(v_cache_ptr + v_cache_offset + offsets * vcache_stride_d, v_vals, mask=mask)

def tokens_update(
    past_key_values,
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    
    layer = past_key_values.layers[layer_idx]

    if not layer.is_initialized:
        layer.lazy_initialization(key_states, value_states)
    
    head_dim = key_states.shape[-1]
    num_heads = key_states.shape[1]
    seq_len = key_states.shape[2]
    
    block_d = min(triton.next_power_of_2(head_dim), 256)
    block_t = min(triton.next_power_of_2(seq_len), 128)

    start_pos = 0
    
    grid = (num_heads, seq_len)
    
    tokens_update_kernel[grid](
        key_states,
        value_states,
        layer.keys,
        layer.values,
        start_pos,
        key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        layer.keys.stride(0), layer.keys.stride(1), layer.keys.stride(2), layer.keys.stride(3),
        layer.values.stride(0), layer.values.stride(1), layer.values.stride(2), layer.values.stride(3),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=2,
    )
    
    return layer.keys, layer.values

@triton.jit
def tokens_update_kernel(
    k_input_ptr, v_input_ptr,
    k_cache_ptr, v_cache_ptr,
    start_pos,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    kcache_stride_b, kcache_stride_h, kcache_stride_s, kcache_stride_d,
    vcache_stride_b, vcache_stride_h, vcache_stride_s, vcache_stride_d,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    head_idx = tl.program_id(0)
    token_idx = tl.program_id(1)
    batch_idx = 0
    write_pos = start_pos + token_idx

    k_in_offset = batch_idx * k_stride_b + head_idx * k_stride_h + token_idx * k_stride_s
    v_in_offset = batch_idx * v_stride_b + head_idx * v_stride_h + token_idx * v_stride_s

    k_cache_offset = batch_idx * kcache_stride_b + head_idx * kcache_stride_h + write_pos * kcache_stride_s
    v_cache_offset = batch_idx * vcache_stride_b + head_idx * vcache_stride_h + write_pos * vcache_stride_s

    d_offsets = tl.arange(0, BLOCK_D)
    mask = d_offsets < HEAD_DIM

    k_vals = tl.load(k_input_ptr + k_in_offset + d_offsets * k_stride_d, mask=mask)
    tl.store(k_cache_ptr + k_cache_offset + d_offsets * kcache_stride_d, k_vals, mask=mask)

    v_vals = tl.load(v_input_ptr + v_in_offset + d_offsets * v_stride_d, mask=mask)
    tl.store(v_cache_ptr + v_cache_offset + d_offsets * vcache_stride_d, v_vals, mask=mask)


@triton.jit
def _lean_decode_attention_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    cache_pos_ptr,
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    acc_stride_h, acc_stride_split, acc_stride_d,
    stats_stride_h, stats_stride_split,
    sm_scale,
    NUM_KV_GROUPS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    TOKENS_PER_SPLIT: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_head_idx = tl.program_id(0)
    split_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // NUM_KV_GROUPS

    kv_len = tl.load(cache_pos_ptr).to(tl.int32) + 1
    kv_len = tl.maximum(kv_len, 1)
    kv_len = tl.minimum(kv_len, NUM_BLOCKS * BLOCK_N)

    num_splits = (kv_len + TOKENS_PER_SPLIT - 1) // TOKENS_PER_SPLIT
    num_splits = tl.maximum(num_splits, 1)
    num_splits = tl.minimum(num_splits, MAX_SPLITS)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    acc_base = partial_acc_ptr + q_head_idx * acc_stride_h + split_idx * acc_stride_split
    stats_offset = q_head_idx * stats_stride_h + split_idx * stats_stride_split
    zero_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    if split_idx >= num_splits:
        tl.store(acc_base + offs_d * acc_stride_d, zero_acc, mask=mask_d)
        tl.store(partial_m_ptr + stats_offset, -1.0e6)
        tl.store(partial_l_ptr + stats_offset, 0.0)
        return

    q_ptrs = q_ptr + q_head_idx * q_stride_h + offs_d * q_stride_d
    q = tl.load(q_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    split_start = (split_idx * kv_len) // num_splits
    split_end = ((split_idx + 1) * kv_len) // num_splits

    acc = zero_acc
    m = -1.0e6
    l = 0.0

    for block_idx in tl.static_range(0, NUM_BLOCKS):
        block_start = split_start + block_idx * BLOCK_N
        if block_start < split_end:
            offs_n = block_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_end

            k_ptrs = (
                k_ptr
                + kv_head_idx * k_stride_h
                + offs_n[:, None] * k_stride_s
                + offs_d[None, :] * k_stride_d
            )
            v_ptrs = (
                v_ptr
                + kv_head_idx * v_stride_h
                + offs_n[:, None] * v_stride_s
                + offs_d[None, :] * v_stride_d
            )

            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            scores = tl.sum(k * q[None, :], axis=1) * sm_scale
            scores = tl.where(mask_n, scores, -1.0e6)

            m_new = tl.maximum(m, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            p = tl.where(mask_n, p, 0.0)
            alpha = tl.exp(m - m_new)

            acc = acc * alpha + tl.sum(v * p[:, None], axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = m_new

    tl.store(acc_base + offs_d * acc_stride_d, acc, mask=mask_d)
    tl.store(partial_m_ptr + stats_offset, m)
    tl.store(partial_l_ptr + stats_offset, l)


@triton.jit
def _lean_decode_attention_reduce_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    cache_pos_ptr,
    acc_stride_h, acc_stride_split, acc_stride_d,
    stats_stride_h, stats_stride_split,
    out_stride_b, out_stride_h, out_stride_s, out_stride_d,
    TOKENS_PER_SPLIT: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_head_idx = tl.program_id(0)

    kv_len = tl.load(cache_pos_ptr).to(tl.int32) + 1
    kv_len = tl.maximum(kv_len, 1)
    num_splits = (kv_len + TOKENS_PER_SPLIT - 1) // TOKENS_PER_SPLIT
    num_splits = tl.maximum(num_splits, 1)
    num_splits = tl.minimum(num_splits, MAX_SPLITS)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    m = -1.0e6
    l = 0.0

    for split_idx in tl.static_range(0, MAX_SPLITS):
        if split_idx < num_splits:
            stats_offset = q_head_idx * stats_stride_h + split_idx * stats_stride_split
            acc_base = partial_acc_ptr + q_head_idx * acc_stride_h + split_idx * acc_stride_split

            part_m = tl.load(partial_m_ptr + stats_offset).to(tl.float32)
            part_l = tl.load(partial_l_ptr + stats_offset).to(tl.float32)
            part_acc = tl.load(acc_base + offs_d * acc_stride_d, mask=mask_d, other=0.0).to(tl.float32)

            m_new = tl.maximum(m, part_m)
            alpha = tl.exp(m - m_new)
            beta = tl.exp(part_m - m_new)

            acc = acc * alpha + part_acc * beta
            l = l * alpha + part_l * beta
            m = m_new

    out = acc / l
    out_ptrs = out_ptr + q_head_idx * out_stride_h + offs_d * out_stride_d
    tl.store(out_ptrs, out, mask=mask_d)


def _can_use_lean_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor | None,
    num_kv_groups: int,
) -> bool:
    if cache_position is None or cache_position.numel() != 1:
        return False
    if any(t is None for t in (query_states, key_states, value_states)):
        return False
    if not all(t.is_cuda for t in (query_states, key_states, value_states, cache_position)):
        return False
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        return False
    if not (query_states.dtype == key_states.dtype == value_states.dtype):
        return False
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        return False
    if query_states.shape[0] != 1 or query_states.shape[2] != 1:
        return False
    if key_states.shape[0] != 1 or value_states.shape[0] != 1:
        return False
    if key_states.shape != value_states.shape:
        return False
    if query_states.shape[-1] != key_states.shape[-1]:
        return False
    if query_states.stride(-1) != 1 or key_states.stride(-1) != 1 or value_states.stride(-1) != 1:
        return False
    if num_kv_groups < 1 or query_states.shape[1] % key_states.shape[1] != 0:
        return False
    return True


def _can_use_lean_decode_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor | None,
    num_kv_groups: int,
    max_splits: int,
) -> bool:
    if not _can_use_lean_decode_attention(
        query_states,
        key_states,
        value_states,
        cache_position,
        num_kv_groups,
    ):
        return False
    if query_states.dtype != torch.float16:
        return False
    if query_states.shape[1] != 16 or key_states.shape[1] != 8:
        return False
    if query_states.shape[-1] != 128 or num_kv_groups != 2:
        return False
    if max_splits < 1 or max_splits > 8:
        return False
    try:
        return torch.cuda.get_device_capability(query_states.device) == (8, 0)
    except Exception:
        return False


def _lean_decode_attention_triton(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor,
    num_kv_groups: int,
    sm_scale: float,
    partial_acc: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    *,
    block_n: int,
    max_splits: int,
    tokens_per_split: int,
) -> torch.Tensor:
    head_dim = query_states.shape[-1]
    num_q_heads = query_states.shape[1]
    block_d = min(triton.next_power_of_2(head_dim), 256)
    num_blocks = triton.cdiv(key_states.shape[2], block_n)

    attn_output = torch.empty_like(query_states)

    _lean_decode_attention_split_kernel[(num_q_heads, max_splits)](
        query_states,
        key_states,
        value_states,
        partial_acc,
        partial_m,
        partial_l,
        cache_position,
        query_states.stride(0), query_states.stride(1), query_states.stride(2), query_states.stride(3),
        key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
        value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_m.stride(0), partial_m.stride(1),
        sm_scale,
        NUM_KV_GROUPS=num_kv_groups,
        NUM_BLOCKS=num_blocks,
        TOKENS_PER_SPLIT=tokens_per_split,
        MAX_SPLITS=max_splits,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )

    _lean_decode_attention_reduce_kernel[(num_q_heads,)](
        partial_acc,
        partial_m,
        partial_l,
        attn_output,
        cache_position,
        partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2),
        partial_m.stride(0), partial_m.stride(1),
        attn_output.stride(0), attn_output.stride(1), attn_output.stride(2), attn_output.stride(3),
        TOKENS_PER_SPLIT=tokens_per_split,
        MAX_SPLITS=max_splits,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return attn_output


def lean_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_position: torch.Tensor,
    num_kv_groups: int,
    sm_scale: float,
    partial_acc: torch.Tensor,
    partial_m: torch.Tensor,
    partial_l: torch.Tensor,
    *,
    block_n: int,
    max_splits: int,
    tokens_per_split: int,
) -> torch.Tensor:
    if _can_use_lean_decode_attention_cuda(
        query_states,
        key_states,
        value_states,
        cache_position,
        num_kv_groups,
        max_splits,
    ):
        extension = _load_lean_decode_cuda_extension()
        if extension is not None:
            return extension.forward(
                query_states,
                key_states,
                value_states,
                cache_position,
                partial_acc,
                partial_m,
                partial_l,
                float(sm_scale),
                int(max_splits),
                int(tokens_per_split),
            )

    return _lean_decode_attention_triton(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        cache_position=cache_position,
        num_kv_groups=num_kv_groups,
        sm_scale=sm_scale,
        partial_acc=partial_acc,
        partial_m=partial_m,
        partial_l=partial_l,
        block_n=block_n,
        max_splits=max_splits,
        tokens_per_split=tokens_per_split,
    )


def _can_use_flashinfer_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> bool:
    if _flashinfer is None:
        return False
    if any(t is None for t in (query_states, key_states, value_states)):
        return False
    if not all(t.is_cuda for t in (query_states, key_states, value_states)):
        return False
    if query_states.dtype != torch.float16 or key_states.dtype != torch.float16 or value_states.dtype != torch.float16:
        return False
    if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
        return False
    if query_states.shape[0] != 1 or query_states.shape[2] != 1:
        return False
    if key_states.shape[0] != 1 or value_states.shape[0] != 1:
        return False
    if key_states.shape != value_states.shape:
        return False
    if query_states.shape[-1] != key_states.shape[-1]:
        return False
    if query_states.shape[1] % key_states.shape[1] != 0:
        return False
    if query_states.stride(-1) != 1 or key_states.stride(-1) != 1 or value_states.stride(-1) != 1:
        return False
    return key_states[0].is_contiguous() and value_states[0].is_contiguous()


def flashinfer_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
) -> torch.Tensor:
    attn_output = _flashinfer.single_decode_with_kv_cache(
        query_states[0, :, 0, :],
        key_states[0],
        value_states[0],
        kv_layout="HND",
        pos_encoding_mode="NONE",
        use_tensor_cores=False,
    )
    return attn_output.view(1, query_states.shape[1], 1, query_states.shape[-1])


""" AAA """
""" AAA """
@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape
        # self._ready()
    def _ready(self):
        self.isReady = True
        with torch.no_grad():
            self.qkv_proj = nn.Linear(self.q_proj.in_features,self.q_proj.out_features + self.k_proj.out_features + self.v_proj.out_features,bias=self.q_proj.bias is not None,device=self.q_proj.weight.device,dtype=self.q_proj.weight.dtype)
            self.qkv_proj.weight.copy_(torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],dim=0))
            if self.qkv_proj.bias is not None:
                print("SS")
                self.qkv_proj.bias.copy_(torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias],dim=0))
            self.out_size = (self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features)
        
        # 融合!!!
        del self.q_proj
        del self.k_proj
        del self.v_proj
        # no-need now~
        self._decode_front_cuda_enabled = os.environ.get("USE_FUSED_DECODE_FRONT_CUDA", "1").lower() in {"1", "true", "yes", "on"}
        self._decode_front_query_states = torch.empty(
            (1, self.config.num_attention_heads, 1, self.head_dim),
            device=self.qkv_proj.weight.device,
            dtype=self.qkv_proj.weight.dtype,
        )
        self._lean_decode_enabled = os.environ.get("USE_LEAN_DECODE_ATTENTION", "").lower() in {"1", "true", "yes", "on"}
        self._lean_decode_block_n = 64
        self._lean_decode_max_splits = 8
        self._lean_decode_tokens_per_split = 128
        self._lean_decode_partial_acc = torch.empty(
            (self.config.num_attention_heads, self._lean_decode_max_splits, self.head_dim),
            device=self.qkv_proj.weight.device,
            dtype=torch.float32,
        )
        self._lean_decode_partial_m = torch.empty(
            (self.config.num_attention_heads, self._lean_decode_max_splits),
            device=self.qkv_proj.weight.device,
            dtype=torch.float32,
        )
        self._lean_decode_partial_l = torch.empty(
            (self.config.num_attention_heads, self._lean_decode_max_splits),
            device=self.qkv_proj.weight.device,
            dtype=torch.float32,
        )
        if self._decode_front_cuda_enabled and self.head_dim == 128 and self.config.num_attention_heads == 16 and self.config.num_key_value_heads == 8:
            _load_decode_front_cuda_extension()
        if self._lean_decode_enabled and self.head_dim == 128:
            _load_lean_decode_cuda_extension()
        self._flashinfer_decode_enabled = (
            os.environ.get("USE_FLASHINFER_DECODE_ATTENTION", "").lower() in {"1", "true", "yes", "on"}
            and _flashinfer is not None
        )
        if self._flashinfer_decode_enabled and self.head_dim == 128:
            try:
                dummy_q = torch.empty(
                    (self.config.num_attention_heads, self.head_dim),
                    device=self.qkv_proj.weight.device,
                    dtype=self.qkv_proj.weight.dtype,
                )
                dummy_k = torch.empty(
                    (self.config.num_key_value_heads, 1, self.head_dim),
                    device=self.qkv_proj.weight.device,
                    dtype=self.qkv_proj.weight.dtype,
                )
                _flashinfer.single_decode_with_kv_cache(
                    dummy_q,
                    dummy_k,
                    dummy_k,
                    kv_layout="HND",
                    pos_encoding_mode="NONE",
                    use_tensor_cores=False,
                )
            except Exception as exc:
                print(f"[flashinfer-decode] disabled after warmup failure: {exc}")
                self._flashinfer_decode_enabled = False
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        

        cos, sin = position_embeddings
        q_weight = self.q_norm.weight
        k_weight = self.k_norm.weight
        q_proj, k_proj, v_proj = self.qkv_proj(hidden_states).split(
            self.out_size,
            dim=-1,
        )
        value_states = None
        decode_front_fused = False
        if (
            self._decode_front_cuda_enabled
            and past_key_values is not None
            and _can_use_decode_attention_front_cuda(
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                q_weight=q_weight,
                k_weight=k_weight,
                cos=cos,
                sin=sin,
                cache_position=cache_position,
                num_attention_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.head_dim,
            )
        ):
            query_states, key_states, value_states = fused_decode_attention_front_cuda(
                past_key_values=past_key_values,
                layer_idx=self.layer_idx,
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                q_weight=q_weight,
                k_weight=k_weight,
                cos=cos,
                sin=sin,
                cache_position=cache_position,
                q_eps=self.q_norm.variance_epsilon,
                k_eps=self.k_norm.variance_epsilon,
                query_buffer=self._decode_front_query_states,
            )
            decode_front_fused = True

        if not decode_front_fused:
            value_states = v_proj.view(hidden_shape).transpose(1, 2)
            if not (any(t is None for t in (q_proj, k_proj, q_weight, k_weight, cos, sin)) or not all(t.is_cuda for t in (q_proj, k_proj, q_weight, k_weight, cos, sin)) or q_proj.dtype not in (torch.float16, torch.bfloat16) or not (q_proj.dtype == k_proj.dtype == q_weight.dtype == k_weight.dtype == cos.dtype == sin.dtype) or q_proj.ndim != 3 or k_proj.ndim != 3 or cos.ndim != 3 or sin.ndim != 3 or q_proj.shape[:2] != (1, 1) or k_proj.shape[:2] != (1, 1) or cos.shape != (1, 1, self.head_dim) or sin.shape != (1, 1, self.head_dim) or q_weight.shape != (self.head_dim,) or k_weight.shape != (self.head_dim,) or self.head_dim % 2 != 0 or self.head_dim > 256 or q_proj.shape[-1] % self.head_dim != 0 or k_proj.shape[-1] % self.head_dim != 0 or any(t.stride(-1) != 1 for t in (q_proj, k_proj, cos, sin)) or q_weight.stride(0) != 1 or k_weight.stride(0) != 1):
                query_states, key_states = fused_text_qk_rotary(
                    q_proj=q_proj,
                    k_proj=k_proj,
                    q_weight=q_weight,
                    k_weight=k_weight,
                    cos=cos,
                    sin=sin,
                    q_eps=self.q_norm.variance_epsilon,
                    k_eps=self.k_norm.variance_epsilon,
                    head_dim=self.head_dim,
                )
            else:
                query_states = self.q_norm(q_proj.view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(k_proj.view(hidden_shape)).transpose(1, 2)
                query_states, key_states = _apply_text_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if decode_front_fused:
                pass
            elif query_states.shape[-2] == 1:
                key_states, value_states = single_token_update(
                    past_key_values,
                    self.layer_idx,
                    key_states,
                    value_states,
                    cache_position,
                )
            else:
                key_states, value_states = tokens_update(
                    past_key_values,
                    self.layer_idx,
                    key_states,
                    value_states,
                    cache_position,
                )
                #key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = None
        attn_weights = None
        kwargs.pop("kv_length", None)
        if (
            query_states.shape[-2] == 1
            and self.head_dim <= 128
            and self._lean_decode_enabled
            and hasattr(self, "_lean_decode_partial_acc")
            and _can_use_lean_decode_attention(
                query_states,
                key_states,
                value_states,
                cache_position,
                self.num_key_value_groups,
            )
        ):
            attn_output = lean_decode_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                cache_position=cache_position,
                num_kv_groups=self.num_key_value_groups,
                sm_scale=self.scaling,
                partial_acc=self._lean_decode_partial_acc,
                partial_m=self._lean_decode_partial_m,
                partial_l=self._lean_decode_partial_l,
                block_n=self._lean_decode_block_n,
                max_splits=self._lean_decode_max_splits,
                tokens_per_split=self._lean_decode_tokens_per_split,
            )
        if (
            query_states.shape[-2] == 1
            and self.head_dim <= 128
            and attn_output == None
            and self._flashinfer_decode_enabled
            and _can_use_flashinfer_decode_attention(query_states, key_states, value_states)
        ):
            attn_output = flashinfer_decode_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
            )
        if ((query_states.shape[-2] == 1) and (self.head_dim <= 128) and attn_output == None):
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=None,
                    dropout_p=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    enable_gqa=self.num_key_value_groups > 1,
                )

        if attn_output is None:
            attention_interface = modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, modeling_qwen3_vl.eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.view(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

@triton.jit
def _add_rmsnorm_forward_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    residual_output_ptr,
    output_ptr,
    input_row_stride,
    residual_row_stride,
    residual_output_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    input_row_ptr = input_ptr + row_idx * input_row_stride
    residual_row_ptr = residual_ptr + row_idx * residual_row_stride
    residual_output_row_ptr = residual_output_ptr + row_idx * residual_output_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride

    values = tl.load(input_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual_values = tl.load(residual_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    summed = values + residual_values
    variance = tl.sum(summed * summed, axis=0) / n_cols
    inv_rms = tl.rsqrt(variance + eps)
    normalized = summed * inv_rms * weights

    tl.store(residual_output_row_ptr + cols, summed, mask=mask)
    tl.store(output_row_ptr + cols, normalized, mask=mask)

def fused_residual_rmsnorm(
    residual: torch.Tensor,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float
):
    n_cols = hidden_states.shape[-1]
    hidden_states_2d = hidden_states.reshape(-1, n_cols)
    residual_2d = residual.reshape(-1, n_cols)
    residual_output = torch.empty_like(hidden_states_2d)
    output = torch.empty_like(hidden_states_2d)
    def autoselect(n_cols: int, n_rows: int):
        block_size = triton.next_power_of_2(n_cols)
        if n_cols <= 128:
            return block_size, 2, 4 if n_rows >= 4096 else 3
        if n_cols <= 1024:
            return block_size, 4, 2
        if n_rows <= 8:
            return block_size, 4, 2
        return block_size, 2, 3
        """
        IF NOT A800, USE BELOW I SUPPOSE IT SHOULD BE BETTA
        """
        if n_cols <= 128:
            num_warps = 2
        elif n_cols <= 1024:
            num_warps = 4
        else:
            num_warps = 8
        return block_size, num_warps, 1

    block_size, num_warps, num_stages = autoselect(
        n_cols=n_cols,
        n_rows=hidden_states_2d.shape[0]
    )

    _add_rmsnorm_forward_kernel[(hidden_states_2d.shape[0],)](
        hidden_states_2d,
        residual_2d,
        weight,
        residual_output,
        output,
        hidden_states_2d.stride(0),
        residual_2d.stride(0),
        residual_output.stride(0),
        output.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return residual_output.view_as(hidden_states), output.view_as(hidden_states)

def Qwen3VLTextDecoderLayer_Forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    weight = self.post_attention_layernorm.weight
    if not (all(t is not None for t in (residual, hidden_states, weight)) and residual.is_cuda and hidden_states.is_cuda and weight.is_cuda and residual.dtype in (torch.float16, torch.bfloat16) and residual.dtype == hidden_states.dtype == weight.dtype and residual.shape == hidden_states.shape and residual.shape[-1] == weight.shape[0] and residual.ndim >= 2 and weight.ndim == 1 and residual.stride(-1) == 1 and hidden_states.stride(-1) == 1 and weight.stride(0) == 1):
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
    else:
        residual, hidden_states = fused_residual_rmsnorm(
            residual=residual,
            hidden_states=hidden_states,
            weight=weight,
            eps=self.post_attention_layernorm.variance_epsilon,
        )

    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states

class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, lengths_ilovetakanashihoshino=None, **kwargs):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_out = self.attn(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings, lengths=lengths_ilovetakanashihoshino, **kwargs)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out
        return hidden_states

class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False
        self._attention_interface = None
        self._uses_flash_attention = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        lengths: list = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if self._attention_interface is None:
            self._attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
        if self._uses_flash_attention is None:
            self._uses_flash_attention = is_flash_attention_requested(self.config)
        attention_interface: Callable = self._attention_interface

        if self._uses_flash_attention:
            # Flash Attention: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            if lengths is None and cu_seqlens.shape[0] == 2:
                attn_output = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
            else:
                # Other implementations: Process each chunk separately
                if lengths is None:
                    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                    split_lengths = lengths.tolist()
                else:
                    split_lengths = lengths

                splits = [
                    torch.split(tensor, split_lengths, dim=2) for tensor in (query_states, key_states, value_states)
                ]

                attn_outputs = [
                    attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for q, k, v in zip(*splits)
                ]
                attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

def _normalize_grid_cache_key(grid_thw, grid_key=None):
    if grid_key is None:
        if grid_thw is None:
            return None
        if isinstance(grid_thw, torch.Tensor):
            return tuple(tuple(int(value) for value in row) for row in grid_thw.tolist())
        return tuple(tuple(int(value) for value in row) for row in grid_thw)
    return tuple(tuple(int(value) for value in row) for row in grid_key)


def rot_pos_emb(self, grid_thw: torch.Tensor, grid_key=None) -> torch.Tensor:
    if not hasattr(self, 'rot_pos_emb_cache'):
        self.rot_pos_emb_cache = {}
    if grid_key is None:
        grid_key = getattr(self, "_image_grid_key", None)
    grid_key = _normalize_grid_cache_key(grid_thw, grid_key)
    if grid_key in self.rot_pos_emb_cache:
        return self.rot_pos_emb_cache[grid_key]
    grid_thw_list = [list(row) for row in grid_key]

    merge_size = self.spatial_merge_size

    max_hw = max(max(h, w) for _, h, w in grid_thw_list)
    freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
    device = freq_table.device

    total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw_list:
        merged_h, merged_w = height // merge_size, width // merge_size

        block_rows = torch.arange(merged_h, device=device)  # block row indices
        block_cols = torch.arange(merged_w, device=device)  # block col indices
        intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
        intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

        # Compute full-resolution positions
        row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

        row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    embeddings = freq_table[pos_ids]  # lookup rotary embeddings
    embeddings = embeddings.flatten(1)
    self.rot_pos_emb_cache[grid_key] = embeddings
    return embeddings

def fast_pos_embed_interpolate(self, grid_thw, grid_key=None):
    if not hasattr(self, 'fast_pos_embed_cache'):
        self.fast_pos_embed_cache = {}
    if grid_key is None:
        grid_key = getattr(self, "_image_grid_key", None)
    grid_key = _normalize_grid_cache_key(grid_thw, grid_key)
    if grid_key in self.fast_pos_embed_cache:
        return self.fast_pos_embed_cache[grid_key]
    grid_thw_list = [list(row) for row in grid_key]
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = self.pos_embed.weight.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    self.fast_pos_embed_cache[grid_key] = patch_pos_embeds
    return patch_pos_embeds


def _get_cached_vision_layout(visual, image_grid_thw, grid_key):
    if not hasattr(visual, "_codex_vision_layout_cache"):
        visual._codex_vision_layout_cache = {}
    layout = visual._codex_vision_layout_cache.get(grid_key)
    if layout is not None:
        return layout

    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw, grid_key=grid_key)
    seq_len = rotary_pos_emb.shape[0]
    rotary_flat = rotary_pos_emb.reshape(seq_len, -1)
    rotary_emb = torch.cat((rotary_flat, rotary_flat), dim=-1)
    cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=image_grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    spatial_merge = visual.spatial_merge_size**2
    split_sizes = [(t * h * w) // spatial_merge for t, h, w in grid_key]
    layout = {
        "cu_seqlens": cu_seqlens,
        "position_embeddings": (rotary_emb.cos(), rotary_emb.sin()),
        "split_sizes": split_sizes,
        "single_sequence": len(split_sizes) == 1,
    }
    visual._codex_vision_layout_cache[grid_key] = layout
    return layout


def _flatten_pooled_output(pooler_output):
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    if len(pooler_output) == 1:
        return pooler_output[0]
    return torch.cat(pooler_output, dim=0)


def _pack_image_feature_cache_entry(image_outputs):
    return (
        _flatten_pooled_output(image_outputs.pooler_output),
        image_outputs.deepstack_features,
    )


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)
        self.proj = self.proj.to(memory_format=torch.channels_last_3d)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype).to(memory_format=torch.channels_last_3d))
        hidden_states = hidden_states.view(-1, self.embed_dim)
        return hidden_states

def Qwen3VLVisionModel_GetImageFeatures(
    self,
    pixel_values: torch.FloatTensor,
    image_grid_thw: torch.LongTensor | None = None,
    image_grid_key=None,
    **kwargs,
) -> tuple | BaseModelOutputWithDeepstackFeatures:
    return_dict = kwargs.pop("return_dict", True)
    pixel_values = pixel_values.type(self.visual.dtype)
    grid_key = _normalize_grid_cache_key(image_grid_thw, image_grid_key)
    pos_embeds = self.visual.fast_pos_embed_interpolate(image_grid_thw, grid_key=grid_key)
    layout = _get_cached_vision_layout(self.visual, image_grid_thw, grid_key)
    hidden_states = self.visual.patch_embed(pixel_values)
    hidden_states = hidden_states + pos_embeds

    deepstack_feature_lists = []
    deepstack_indexes = list(getattr(self.visual, "deepstack_visual_indexes", ()))
    if deepstack_indexes:
        deepstack_map = {layer_num: idx for idx, layer_num in enumerate(deepstack_indexes)}
        deepstack_mergers = getattr(self.visual, "deepstack_merger_list", ())
    else:
        deepstack_map = {}
        deepstack_mergers = ()

    for layer_num, blk in enumerate(self.visual.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=layout["cu_seqlens"],
            position_embeddings=layout["position_embeddings"],
            **kwargs,
        )
        merger_idx = deepstack_map.get(layer_num)
        if merger_idx is not None:
            deepstack_feature_lists.append(deepstack_mergers[merger_idx](hidden_states))

    merged_hidden_states = self.visual.merger(hidden_states)
    if layout["single_sequence"]:
        image_embeds = (merged_hidden_states,)
    else:
        image_embeds = torch.split(merged_hidden_states, layout["split_sizes"])
    if not return_dict:
        return (None, image_embeds, deepstack_feature_lists)
    return BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=None,
        pooler_output=image_embeds,
        deepstack_features=deepstack_feature_lists,
    )

def register_graph(self, model, max_seq=1200, max_tokens=128, batch = 1, device="cuda:0", dtype=torch.float16):
    self.isCaptureStage = False
    self.milestones.append(max_seq)
    self.graph_lm_head = model.lm_head
    # 1, 1200
    # attention_mask = torch.zeros(batch, max_seq, dtype=torch.long, device=device)
    # 4, 1, 1200
    # 
    position_ids = torch.zeros(4, batch, 1 , dtype=torch.int64, device=device)
    attention_mask = torch.zeros(1, max_seq , dtype=torch.bool, device=device)
    # In kv cache, the max seq each time should be 1 <- :>
    inputs_embeds = torch.zeros(batch, 1 , self.hidden_states ,dtype=torch.float16, device=device)
    # 1, 1200, 2024
    last_hidden_state = torch.zeros(batch, 1, self.hidden_states, device=device, dtype=torch.float16)
    # [1]
    cache_position = torch.zeros(1, device=device, dtype=torch.long)

    # Prefill ~~
    position_ids_ff = torch.zeros(4, batch, max_seq ,dtype=torch.int64, device=device)
    attention_mask_ff = torch.zeros(1, 1, max_seq, max_seq ,dtype=torch.bool, device=device)
    inputs_embeds_ff = torch.zeros(batch, max_seq , self.hidden_states ,dtype=dtype, device=device)
    last_hidden_state_ff = torch.zeros(1, max_seq, self.hidden_states, device=device, dtype=dtype)
    cache_position_ff = torch.zeros(max_seq, device=device, dtype=torch.long)

    decode_logits = torch.zeros(batch, 1, model.lm_head.out_features, device=device, dtype=model.lm_head.weight.dtype)
    next_token_ids = torch.zeros(batch, 1, device=device, dtype=torch.long)
    chunk_next_token_ids = torch.zeros(batch, max_tokens, device=device, dtype=torch.long)
    chunk_position_ids = torch.zeros(4, batch, 1, dtype=torch.int64, device=device)
    chunk_inputs_embeds = torch.zeros(batch, 1, self.hidden_states, dtype=dtype, device=device)
    chunk_cache_position = torch.zeros(1, device=device, dtype=torch.long)
    
    cache = ASStaticCache(
        config=model.config,
        max_batch_size=batch,
        max_cache_len=max_seq,
        device=device,
        dtype=dtype
    )
    buffer = {
        'position_ids': position_ids,
        'attention_mask': attention_mask,
        'cache': cache,
        'decode_attention_len': 0,
        'decode_logits': decode_logits,
        'next_token_ids': next_token_ids,
        'chunk_next_token_ids': chunk_next_token_ids,
        'chunk_position_ids': chunk_position_ids,
        'chunk_inputs_embeds': chunk_inputs_embeds,
        'chunk_cache_position': chunk_cache_position,
        'inputs_embeds': inputs_embeds,
        'cache_position': cache_position,
        'last_hidden_state': last_hidden_state,
        'position_ids_ff': position_ids_ff,
        'attention_mask_ff': attention_mask_ff,
        'inputs_embeds_ff': inputs_embeds_ff,
        'last_hidden_state_ff': last_hidden_state_ff,
        'cache_position_ff': cache_position_ff
    }
    self.caches.append(cache)
    cache.reset()
    if len(self.buffers) == 0:
        # Pre-Warmup
        self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            cache_position=cache_position,
            use_cache=True,
        )
    self.buffers.append(buffer)
    self.cache_id_to_buffer[id(cache)] = buffer
    return buffer
def capture_graphs(self, func):
    # LOCKON
    self.isCaptureStage = True
    image = Image.new('1', (200, 100), color=0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = "Genius Machine?"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text(((200 - text_width) // 2, (100 - text_height) // 2), text, fill=255, font=font)
    for cache in self.caches:
        response = func(image, "Extract texts", 5, cache)
    print("Response of Warming Up: ", response["text"])
    self.isCaptureStage = False

def Qwen3VLModel_(self, config):
    super(Qwen3VLPreTrainedModel ,self).__init__(config)
    self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
    self.language_model = Qwen3VLTextModel._from_config(config.text_config)
    self.rope_deltas = None  # cache rope_deltas here

    # FeatureCache
    self.imageFeatures = {}
    # Initialize static tensors
    self.input_ids = None
    self.attention_mask = None
    self.position_ids = None
    self.past_key_values = None
    self.inputs_embeds = None
    self.pixel_values = None
    self.pixel_values_videos = None
    self.video_grid_thw = None
    self.batch_size = 1
    self.mm_token_type_ids = None
    self._dtype = torch.float16
    self.isFirst = True
    self.isSecond = False
    self.max_sequence_length = 2980
    self.graph = None
    self.graph_ = None
    self.destination = 0
    self.cache_position = None
    self.max_cache_len = 2980
    self._device = "cuda:0"
    # self.prefillOnly_model = torch.compile(self.language_model, mode="max-autotune")
    # Initialize weights and apply final processing
    self.post_init()
    # VERY IMPORTANT!!!
    self.last_hidden_state = None
    # ff
    self.position_ids_ff = None
    self.remaining_position_ids = None
    self.remaining_cache_position = None
    # registration
    self.hidden_states = 2048
    self.graphs = []
    self.caches = []
    self.buffers = []
    self.cache_id_to_buffer = {}
    self.milestones = []
    self.cache = None
    self.isCaptureStage = False
    # cache
    self.imageFeatures = {}

def AutoTensorLand(attention_mask_,
    position_ids_,
    inputs_embeds_,
    cache_position_,
    destination=None,
    current_attention_len=0,
    ):
    distance_before_landing = min(destination or 0, attention_mask_[0].shape[1])
    previous_attention_len = min(current_attention_len, attention_mask_[0].shape[1])

    if previous_attention_len > distance_before_landing:
        attention_mask_[0][:, distance_before_landing:previous_attention_len].zero_()
    elif previous_attention_len < distance_before_landing:
        attention_mask_[0][:, previous_attention_len:distance_before_landing].fill_(1)
    position_ids_[0].copy_(position_ids_[1])
    inputs_embeds_[0].copy_(inputs_embeds_[1])
    cache_position_[0].copy_(cache_position_[1])
    return distance_before_landing

# def AutoTensorReady(attention_mask_,
#     position_ids_,
#     inputs_embeds_,
#     cache_position_):
#     prefill_len = position_ids_[1].shape[2] # 597
#     position_ids_[0][:, :, :prefill_len].copy_(position_ids_[1])
#     attention_mask_[0][:, :, :prefill_len, :].copy_(attention_mask_[1])
#     inputs_embeds_[0][:, :prefill_len, :].copy_(inputs_embeds_[1])
#     cache_position_[0][:cache_position_[1].shape[0]].copy_(cache_position_[1])
def AutoTensorReady(attention_mask_,
    position_ids_,
    inputs_embeds_,
    cache_position_):
    prefill_len = min(position_ids_[0].shape[2], position_ids_[1].shape[2])
    position_ids_[0].zero_()
    attention_mask_[0].zero_()
    inputs_embeds_[0].zero_()
    cache_position_[0].zero_()
    position_ids_[0][:, :, :prefill_len].copy_(position_ids_[1])
    source_kv_len = min(attention_mask_[0].shape[3], attention_mask_[1].shape[3])
    attention_mask_[0][:, :, :prefill_len, :source_kv_len].copy_(attention_mask_[1][:, :, :prefill_len, :source_kv_len])
    inputs_embeds_[0][:, :prefill_len, :].copy_(inputs_embeds_[1])
    cache_position_[0][:prefill_len].copy_(cache_position_[1][:prefill_len])


def _advance_chunk_decode_sources(self, buffer, next_token_ids, position_ids, cache_position):
    buffer["chunk_inputs_embeds"].copy_(self.get_input_embeddings()(next_token_ids))
    buffer["chunk_position_ids"].copy_(position_ids)
    buffer["chunk_position_ids"].add_(1)
    buffer["chunk_cache_position"].copy_(cache_position)
    buffer["chunk_cache_position"].add_(1)
    return buffer["chunk_inputs_embeds"], buffer["chunk_position_ids"], buffer["chunk_cache_position"]


def _replay_decode_graph_steps(self, buffer, attention_mask, position_ids, inputs_embeds, cache_position, steps):
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = 1
    steps = max(1, min(steps, buffer["chunk_next_token_ids"].shape[1]))

    current_position_ids = position_ids
    current_inputs_embeds = inputs_embeds
    current_cache_position = cache_position
    destination = self.destination
    executed_steps = 0

    for step_idx in range(steps):
        buffer["decode_attention_len"] = AutoTensorLand(
            [buffer["attention_mask"], attention_mask],
            [buffer["position_ids"], current_position_ids],
            [buffer["inputs_embeds"], current_inputs_embeds],
            [buffer["cache_position"], current_cache_position],
            destination=destination,
            current_attention_len=buffer["decode_attention_len"],
        )
        buffer["graph"].replay()
        buffer["chunk_next_token_ids"][:, step_idx : step_idx + 1].copy_(buffer["next_token_ids"])
        executed_steps += 1
        destination += 1

        if step_idx + 1 >= steps:
            break

        current_inputs_embeds, current_position_ids, current_cache_position = _advance_chunk_decode_sources(
            self,
            buffer,
            buffer["next_token_ids"],
            current_position_ids,
            current_cache_position,
        )

    return executed_steps, destination


def _get_buffer_for_cache(self, past_key_values):
    buffer = self.cache_id_to_buffer.get(id(past_key_values))
    if buffer is not None:
        return buffer
    idx = self.caches.index(past_key_values)
    buffer = self.buffers[idx]
    self.cache_id_to_buffer[id(past_key_values)] = buffer
    return buffer

def Qwen3VLForConditionalGeneration_Forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: ASStaticCache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    image_feature_key=None,
    image_grid_key=None,
    **kwargs,
):
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
        mm_token_type_ids=mm_token_type_ids,
        image_feature_key=image_feature_key,
        image_grid_key=image_grid_key,
        **kwargs,
    )
    decode_logits = outputs.get("decode_logits")
    next_token_ids = outputs.get("next_token_ids")
    if decode_logits != None:
        logits = decode_logits
    else:
        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
    response = Qwen3VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )
    if next_token_ids is not None:
        response["next_token_ids"] = next_token_ids
    return response


@capture_outputs
def Qwen3VLModel_Forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: ASStaticCache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    image_feature_key=None,
    image_grid_key=None,
    **kwargs,
) -> tuple | Qwen3VLModelOutputWithPast:
    image_mask = None
    video_mask = None
    cache_position = kwargs.get("cache_position")
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        def get_hash_stats(pixel_values):
            return hash((
                pixel_values.shape,
                pixel_values.sum().item(),
                pixel_values.mean().item(),
                pixel_values.std().item(),
                pixel_values.min().item(),
                pixel_values.max().item()
            ))
        hashstr = image_feature_key if image_feature_key is not None else get_hash_stats(pixel_values)
        shape = cache_position.shape[0] if cache_position is not None else 0
        if hashstr in self.imageFeatures and shape > 2:
            image_embeds, deepstack_image_embeds = self.imageFeatures[hashstr]
        else:
            image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
                pixel_values,
                image_grid_thw,
                image_grid_key=image_grid_key,
                return_dict=True,
            )
            image_embeds, deepstack_image_embeds = _pack_image_feature_cache_entry(image_outputs)
            if shape > 2:
                self.imageFeatures[hashstr] = (image_embeds, deepstack_image_embeds)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds.masked_scatter_(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_outputs: BaseModelOutputWithDeepstackFeatures = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        video_embeds = _flatten_pooled_output(video_outputs.pooler_output)
        deepstack_video_embeds = video_outputs.deepstack_features
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds.masked_scatter_(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds
    if self.isCaptureStage and ((self.isFirst==False and visual_pos_masks!=None) or kwargs["cache_position"].shape[0] > 1):
        buffer = _get_buffer_for_cache(self, past_key_values)
        self.First = False
        self.isSecond = True
        buffer["ffgraph"] = torch.cuda.CUDAGraph()
        AutoTensorReady(
            [buffer["attention_mask_ff"], attention_mask],
            [buffer["position_ids_ff"], position_ids],
            [buffer["inputs_embeds_ff"], inputs_embeds],
            [buffer["cache_position_ff"], kwargs["cache_position"]])
        with torch.cuda.graph(buffer["ffgraph"]):
            res = self.language_model(
                input_ids=None,
                position_ids=buffer["position_ids_ff"],
                attention_mask=buffer["attention_mask_ff"],
                past_key_values=buffer["cache"],
                inputs_embeds=buffer["inputs_embeds_ff"],
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
                cache_position=buffer["cache_position_ff"],
                use_cache=True,
            )
            buffer["last_hidden_state_ff"].copy_(res.last_hidden_state)
        buffer["ffgraph"].replay()
        self.destination += 1
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=buffer["last_hidden_state_ff"],
            past_key_values=past_key_values,
            rope_deltas=self.rope_deltas,
        )
    elif (self.isFirst==False and visual_pos_masks!=None) or self.isFirst:
        self.destination = kwargs["cache_position"].shape[0]
        self.isSecond = True
        self.isFirst = False
        buffer = _get_buffer_for_cache(self, past_key_values)
        if "ffgraph" in buffer:
            self.destination += 1
            AutoTensorReady(
                [buffer["attention_mask_ff"], attention_mask],
                [buffer["position_ids_ff"], position_ids],
                [buffer["inputs_embeds_ff"], inputs_embeds],
                [buffer["cache_position_ff"], kwargs["cache_position"]])
            buffer["ffgraph"].replay()
            return Qwen3VLModelOutputWithPast(
                last_hidden_state=buffer["last_hidden_state_ff"][:, :position_ids.shape[2], :],
                past_key_values=past_key_values,
                rope_deltas=self.rope_deltas,
            )
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            cache_position=kwargs["cache_position"],
            use_cache=True,
        )
        self.destination += 1
        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=past_key_values,
            rope_deltas=self.rope_deltas,
        )
    elif self.isCaptureStage and self.isSecond:
        buffer = _get_buffer_for_cache(self, past_key_values)
        self.isSecond = False
        buffer["graph"] = torch.cuda.CUDAGraph()
        self.graphs.append(buffer["graph"])
        buffer["decode_attention_len"] = AutoTensorLand(
            [buffer["attention_mask"], attention_mask],
            [buffer["position_ids"], position_ids],
            [buffer["inputs_embeds"], inputs_embeds],
            [buffer["cache_position"], kwargs["cache_position"]],
            destination=self.destination,
            current_attention_len=buffer["decode_attention_len"])
        with torch.cuda.graph(buffer["graph"]):
            res = self.language_model(
                input_ids=None,
                position_ids=buffer["position_ids"],
                attention_mask=buffer["attention_mask"],
                past_key_values=buffer["cache"],
                inputs_embeds=buffer["inputs_embeds"],
                visual_pos_masks=None,
                deepstack_visual_embeds=None,
                cache_position=buffer["cache_position"],
                use_cache=True,
            )
            buffer["last_hidden_state"].copy_(res.last_hidden_state)
            buffer["decode_logits"].copy_(self.graph_lm_head(buffer["last_hidden_state"]))
            buffer["next_token_ids"].copy_(buffer["decode_logits"].argmax(dim=-1))
        buffer["graph"].replay()
        decode_steps_executed = 1
        next_token_ids_response = buffer["next_token_ids"]
        self.destination += decode_steps_executed
    else:
        buffer = _get_buffer_for_cache(self, past_key_values)
        decode_steps_executed, next_destination = _replay_decode_graph_steps(
            self,
            buffer,
            attention_mask,
            position_ids,
            inputs_embeds,
            kwargs["cache_position"],
            kwargs.get("decode_chunk_steps", 1),
        )
        next_token_ids_response = (
            buffer["chunk_next_token_ids"][:, :decode_steps_executed]
            if decode_steps_executed > 1
            else buffer["next_token_ids"]
        )
        self.destination = next_destination
    response = Qwen3VLModelOutputWithPast(
        last_hidden_state=buffer["last_hidden_state"],
        past_key_values=past_key_values,
        rope_deltas=self.rope_deltas,
    )
    # Code 200
    response["decode_logits"] = buffer["decode_logits"]
    response["next_token_ids"] = next_token_ids_response
    return response
