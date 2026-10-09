"""CUDA kernels for decode experiments."""

__all__ = [
    "cuda_decode_gate_up_swiglu",
    "cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp",
    "cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale",
    "cuda_decode_gate_up_swiglu_fp8",
    "cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_kscale",
    "cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_major",
    "cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_e4b15_kblock_major",
    "cuda_decode_prefix_stage2_o_add_rmsnorm_int4_sym_kblock_kscale_halfwarp",
    "cuda_decode_gemv_add_rmsnorm_int4_sym_kblock_kscale_halfwarp",
    "cuda_decode_qkv_gemv",
    "cuda_decode_qkv_gemv_fp8",
    "cuda_decode_qkv_fp8_hscale_qk_rmsnorm_rope_kv_update_gqa2",
    "cuda_decode_qkv_gemv_int4_sym_kblock_kscale",
    "cuda_decode_qkv_qk_rmsnorm_rope_kv_update_gqa2",
    "cuda_decode_qkv_qk_rmsnorm_rope_cache_kv_update_gqa2",
    "cuda_gemv_splitk_partial_fp8_e4b15_kblock_major",
    "cuda_gemv_splitk_partial_fp8_e4b15_kblock_kscale",
    "cuda_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp",
    "quantize_gate_up_e4m3_block128",
    "cuda_decode_prefix_attention_bucket_config",
    "cuda_decode_prefix_attention_bucketed_gqa2",
    "cuda_decode_prefix_attention_single_gqa2",
    "cuda_decode_prefix_attention_split_gqa2",
    "cuda_decode_prefix_attention_split_vgroup_atomic_gqa2",
    "cuda_decode_prefix_attention_split_vgroup_gqa2",
    "cuda_decode_prefix_attention_tiled_gqa2",
    "cuda_decode_prefix_attention_tiled_vgroup_gqa2",
    "cuda_decode_prefix_attention_tiled_warpred_gqa2",
    "make_prefix_attention_atomic_workspace",
    "make_prefix_attention_bucketed_workspace",
    "make_prefix_attention_split_workspace",
]


def __getattr__(name: str):
    if name == "cuda_decode_gate_up_swiglu":
        from .decode_gate_up_swiglu import cuda_decode_gate_up_swiglu

        return cuda_decode_gate_up_swiglu
    if name == "cuda_decode_gate_up_swiglu_fp8":
        from .decode_gate_up_swiglu import cuda_decode_gate_up_swiglu_fp8

        return cuda_decode_gate_up_swiglu_fp8
    if name == "cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp":
        from .decode_gate_up_int4_sym import cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp

        return cuda_decode_gate_up_swiglu_int4_sym_kblock_kscale_halfwarp
    if name == "cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale":
        from .decode_gate_up_fp8_e4b15 import cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale

        return cuda_decode_gate_up_swiglu_fp8_e4b15_kblock_kscale
    if name == "cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_kscale":
        from .decode_gemv_add_rmsnorm_fp8_e4b15 import (
            cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_kscale,
        )

        return cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_kscale
    if name == "cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_major":
        from .decode_gemv_add_rmsnorm_fp8_e4b15 import (
            cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_major,
        )

        return cuda_decode_gemv_add_rmsnorm_fp8_e4b15_kblock_major
    if name == "cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_e4b15_kblock_major":
        from .decode_gemv_add_rmsnorm_fp8_e4b15 import (
            cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_e4b15_kblock_major,
        )

        return cuda_decode_prefix_stage2_o_add_rmsnorm_fp8_e4b15_kblock_major
    if name == "cuda_decode_prefix_stage2_o_add_rmsnorm_int4_sym_kblock_kscale_halfwarp":
        from .decode_gemv_add_rmsnorm_fp8_e4b15 import (
            cuda_decode_prefix_stage2_o_add_rmsnorm_int4_sym_kblock_kscale_halfwarp,
        )

        return cuda_decode_prefix_stage2_o_add_rmsnorm_int4_sym_kblock_kscale_halfwarp
    if name == "cuda_decode_gemv_add_rmsnorm_int4_sym_kblock_kscale_halfwarp":
        from .decode_gemv_add_rmsnorm_fp8_e4b15 import (
            cuda_decode_gemv_add_rmsnorm_int4_sym_kblock_kscale_halfwarp,
        )

        return cuda_decode_gemv_add_rmsnorm_int4_sym_kblock_kscale_halfwarp
    if name == "quantize_gate_up_e4m3_block128":
        from .decode_gate_up_swiglu import quantize_gate_up_e4m3_block128

        return quantize_gate_up_e4m3_block128
    if name == "cuda_decode_qkv_gemv":
        from .decode_qkv_gemv import cuda_decode_qkv_gemv

        return cuda_decode_qkv_gemv
    if name == "cuda_decode_qkv_gemv_fp8":
        from .decode_qkv_gemv import cuda_decode_qkv_gemv_fp8

        return cuda_decode_qkv_gemv_fp8
    if name == "cuda_decode_qkv_fp8_hscale_qk_rmsnorm_rope_kv_update_gqa2":
        from .decode_qkv_gemv import cuda_decode_qkv_fp8_hscale_qk_rmsnorm_rope_kv_update_gqa2

        return cuda_decode_qkv_fp8_hscale_qk_rmsnorm_rope_kv_update_gqa2
    if name == "cuda_decode_qkv_gemv_int4_sym_kblock_kscale":
        from .decode_qkv_gemv import cuda_decode_qkv_gemv_int4_sym_kblock_kscale

        return cuda_decode_qkv_gemv_int4_sym_kblock_kscale
    if name == "cuda_decode_qkv_qk_rmsnorm_rope_kv_update_gqa2":
        from .decode_qkv_gemv import cuda_decode_qkv_qk_rmsnorm_rope_kv_update_gqa2

        return cuda_decode_qkv_qk_rmsnorm_rope_kv_update_gqa2
    if name == "cuda_decode_qkv_qk_rmsnorm_rope_cache_kv_update_gqa2":
        from .decode_qkv_gemv import cuda_decode_qkv_qk_rmsnorm_rope_cache_kv_update_gqa2

        return cuda_decode_qkv_qk_rmsnorm_rope_cache_kv_update_gqa2
    if name == "cuda_gemv_splitk_partial_fp8_e4b15_kblock_major":
        from .decode_gemv_partial_fp8_e4b15 import cuda_gemv_splitk_partial_fp8_e4b15_kblock_major

        return cuda_gemv_splitk_partial_fp8_e4b15_kblock_major
    if name == "cuda_gemv_splitk_partial_fp8_e4b15_kblock_kscale":
        from .decode_gemv_partial_fp8_e4b15 import cuda_gemv_splitk_partial_fp8_e4b15_kblock_kscale

        return cuda_gemv_splitk_partial_fp8_e4b15_kblock_kscale
    if name == "cuda_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp":
        from .decode_gemv_partial_int4_sym import cuda_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp

        return cuda_gemv_splitk_partial_int4_sym_kblock_kscale_halfwarp
    if name in {
        "cuda_decode_prefix_attention_bucket_config",
        "cuda_decode_prefix_attention_bucketed_gqa2",
        "cuda_decode_prefix_attention_single_gqa2",
        "cuda_decode_prefix_attention_split_gqa2",
        "cuda_decode_prefix_attention_split_vgroup_atomic_gqa2",
        "cuda_decode_prefix_attention_split_vgroup_gqa2",
        "cuda_decode_prefix_attention_tiled_gqa2",
        "cuda_decode_prefix_attention_tiled_vgroup_gqa2",
        "cuda_decode_prefix_attention_tiled_warpred_gqa2",
        "make_prefix_attention_atomic_workspace",
        "make_prefix_attention_bucketed_workspace",
        "make_prefix_attention_split_workspace",
    }:
        from . import prefix_attention

        return getattr(prefix_attention, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
