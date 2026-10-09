from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F


_ROOT_WRAPPER_PATH = Path(__file__).resolve()
_BASE_WRAPPER_PATH = _ROOT_WRAPPER_PATH.parent / "_base" / "evaluation_wrapper.py"
_BASE_MODULE_NAME = "aicas_prefill_gpuvisionnorm_base_prefillqkrotary_20260417"
_BASE_SPEC = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_WRAPPER_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"无法加载基础 wrapper: {_BASE_WRAPPER_PATH}")
_BASE_MODULE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE_MODULE)
_QWEN3_VL_MODELING = importlib.import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")

triton = getattr(_BASE_MODULE, "triton", None)
tl = getattr(_BASE_MODULE, "tl", None)
_choose_warp_count = getattr(_BASE_MODULE, "_choose_warp_count", lambda col_count: 4)


if triton is not None and tl is not None:

    @triton.jit
    def _prefill_qk_rmsnorm_rotary_pair_kernel(
        q_src_ptr,
        k_src_ptr,
        q_scale_ptr,
        k_scale_ptr,
        cos_ptr,
        sin_ptr,
        q_dst_ptr,
        k_dst_ptr,
        q_src_row_stride,
        k_src_row_stride,
        cos_row_stride,
        sin_row_stride,
        q_dst_row_stride,
        k_dst_row_stride,
        half_width,
        q_rows,
        k_rows,
        q_heads_per_token,
        k_heads_per_token,
        q_eps,
        k_eps,
        STORE_BF16: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        offs = tl.arange(0, BLOCK_HALF)
        base_mask = offs < half_width
        width = half_width * 2
        out_dtype = tl.bfloat16 if STORE_BF16 else tl.float16

        q_token_valid = row_idx < q_rows
        q_token_idx = row_idx // q_heads_per_token
        q_mask = q_token_valid & base_mask
        q_left = tl.load(q_src_ptr + row_idx * q_src_row_stride + offs, mask=q_mask, other=0.0).to(tl.float32)
        q_right = tl.load(
            q_src_ptr + row_idx * q_src_row_stride + half_width + offs,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        q_cos_left = tl.load(
            cos_ptr + q_token_idx * cos_row_stride + offs,
            mask=q_mask,
            other=1.0,
        ).to(tl.float32)
        q_cos_right = tl.load(
            cos_ptr + q_token_idx * cos_row_stride + half_width + offs,
            mask=q_mask,
            other=1.0,
        ).to(tl.float32)
        q_sin_left = tl.load(
            sin_ptr + q_token_idx * sin_row_stride + offs,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        q_sin_right = tl.load(
            sin_ptr + q_token_idx * sin_row_stride + half_width + offs,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        q_weight_left = tl.load(q_scale_ptr + offs, mask=base_mask, other=1.0).to(tl.float32)
        q_weight_right = tl.load(q_scale_ptr + half_width + offs, mask=base_mask, other=1.0).to(tl.float32)
        q_mean_square = tl.sum(q_left * q_left + q_right * q_right, axis=0) / width
        q_inv_rms = tl.rsqrt(q_mean_square + q_eps)
        q_norm_left = q_left * q_inv_rms * q_weight_left
        q_norm_right = q_right * q_inv_rms * q_weight_right
        q_out_left = q_norm_left * q_cos_left - q_norm_right * q_sin_left
        q_out_right = q_norm_right * q_cos_right + q_norm_left * q_sin_right
        tl.store(q_dst_ptr + row_idx * q_dst_row_stride + offs, q_out_left.to(out_dtype), mask=q_mask)
        tl.store(
            q_dst_ptr + row_idx * q_dst_row_stride + half_width + offs,
            q_out_right.to(out_dtype),
            mask=q_mask,
        )

        k_token_valid = row_idx < k_rows
        k_token_idx = row_idx // k_heads_per_token
        k_mask = k_token_valid & base_mask
        k_left = tl.load(k_src_ptr + row_idx * k_src_row_stride + offs, mask=k_mask, other=0.0).to(tl.float32)
        k_right = tl.load(
            k_src_ptr + row_idx * k_src_row_stride + half_width + offs,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        k_cos_left = tl.load(
            cos_ptr + k_token_idx * cos_row_stride + offs,
            mask=k_mask,
            other=1.0,
        ).to(tl.float32)
        k_cos_right = tl.load(
            cos_ptr + k_token_idx * cos_row_stride + half_width + offs,
            mask=k_mask,
            other=1.0,
        ).to(tl.float32)
        k_sin_left = tl.load(
            sin_ptr + k_token_idx * sin_row_stride + offs,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        k_sin_right = tl.load(
            sin_ptr + k_token_idx * sin_row_stride + half_width + offs,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        k_weight_left = tl.load(k_scale_ptr + offs, mask=base_mask, other=1.0).to(tl.float32)
        k_weight_right = tl.load(k_scale_ptr + half_width + offs, mask=base_mask, other=1.0).to(tl.float32)
        k_mean_square = tl.sum(k_left * k_left + k_right * k_right, axis=0) / width
        k_inv_rms = tl.rsqrt(k_mean_square + k_eps)
        k_norm_left = k_left * k_inv_rms * k_weight_left
        k_norm_right = k_right * k_inv_rms * k_weight_right
        k_out_left = k_norm_left * k_cos_left - k_norm_right * k_sin_left
        k_out_right = k_norm_right * k_cos_right + k_norm_left * k_sin_right
        tl.store(k_dst_ptr + row_idx * k_dst_row_stride + offs, k_out_left.to(out_dtype), mask=k_mask)
        tl.store(
            k_dst_ptr + row_idx * k_dst_row_stride + half_width + offs,
            k_out_right.to(out_dtype),
            mask=k_mask,
        )

else:
    _prefill_qk_rmsnorm_rotary_pair_kernel = None


class VLMModel(_BASE_MODULE.VLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefill_qk_rotary_warned_modules = set()
        self._enable_prefill_qk_norm_rotary_attention()
        self._source_config = self._build_source_config()

    def _build_source_config(self) -> dict[str, object]:
        processor_metadata = self._processor_load_config.get("metadata", {})
        if not isinstance(processor_metadata, dict):
            processor_metadata = {}

        applied_optimizations = [
            f"decode_runtime:{self._resolved_decode_runtime_variant}"
        ]
        if "prefill_qk_norm_rotary_attention" in getattr(self, "_optimizations_applied", []):
            applied_optimizations.append("prefill_attention:qk_norm_rotary_pair")

        wrapper_path = _ROOT_WRAPPER_PATH
        wrapper_sha256 = _BASE_MODULE._sha256_file(wrapper_path)
        decode_runtime_metadata = {
            "variant": self._resolved_decode_runtime_variant,
            "path_name": self._text_opt_config.get("path_name", ""),
            "enable_custom_text_patch": bool(
                self._text_opt_config.get("enable_custom_text_patch", False)
            ),
            "fused_text_rms_norm": bool(
                self._text_opt_config.get("fused_text_rms_norm", False)
            ),
            "fused_residual_rmsnorm": bool(
                self._text_opt_config.get("fused_residual_rmsnorm", False)
            ),
            "fused_qk_norm_rotary": bool(
                self._text_opt_config.get("fused_qk_norm_rotary", False)
            ),
            "decode_linear_backend": self._text_opt_config.get(
                "decode_linear_backend", "torch_linear"
            ),
            "decode_graph_enabled": bool(
                self._text_opt_config.get("decode_graph_enabled", False)
            ),
            "generate_mode": self._text_opt_config.get("generate_mode", ""),
            "prefill_backend": self._text_opt_config.get("prefill_backend", ""),
            "kv_impl": self._kv_cache_opt_config.get("impl", ""),
            "kv_bucket": self._kv_cache_opt_config.get("bucket", 0),
            "kv_pool_size": self._kv_cache_opt_config.get("pool_size", 0),
            "visual_prefix_reuse_enabled": bool(self._visual_prefix_reuse_enabled),
            "runtime_components": list(self._optimizations_applied),
            "prefill_qk_norm_rotary_attention": "prefill_qk_norm_rotary_attention"
            in getattr(self, "_optimizations_applied", []),
        }
        signature_payload = {
            "profile": _BASE_MODULE.DEFAULT_OPT_PROFILE,
            "variant": self._resolved_decode_runtime_variant,
            "model_dtype": self._requested_model_dtype_name,
            "processor": processor_metadata,
            "decode_runtime": decode_runtime_metadata,
            "wrapper_sha256": wrapper_sha256,
        }
        return {
            "requested_profile": _BASE_MODULE.DEFAULT_OPT_PROFILE,
            "profile": _BASE_MODULE.DEFAULT_OPT_PROFILE,
            "enabled_optimizations": ["decode_runtime", "prefill_attention"],
            "disabled_optimizations": [],
            "variants": {
                "decode_runtime": self._resolved_decode_runtime_variant,
                "vision_encoder": "default",
                "kv_cache": "default",
                "cross_modal": "default",
                "flash_attention": "default",
                "quantization": "default",
                "prefill_attention": "qk_norm_rotary_pair",
            },
            "unknown_options": [],
            "available_profiles": list(_BASE_MODULE.AVAILABLE_SOURCE_PROFILES),
            "decode_runtime": decode_runtime_metadata,
            "model_dtype": self._requested_model_dtype_name,
            "applied_optimizations": applied_optimizations,
            "wrapper_path": str(wrapper_path),
            "wrapper_sha256": wrapper_sha256,
            "wrapper_source_signature": _BASE_MODULE._stable_sha256(signature_payload),
            "env": {},
            "processor": dict(processor_metadata),
            "init_warmup": {"enabled": False},
        }

    def _get_triton_prefill_qk_norm_rotary_pair(self):
        if hasattr(self, "_triton_prefill_qk_norm_rotary_pair_fn"):
            return self._triton_prefill_qk_norm_rotary_pair_fn

        if triton is None or tl is None or _prefill_qk_rmsnorm_rotary_pair_kernel is None:
            self._triton_prefill_qk_norm_rotary_pair_fn = None
            self._triton_prefill_qk_norm_rotary_pair_reason = "triton prefill qk norm+rotary pair kernel unavailable"
            return None

        def _forward(
            q_states: torch.Tensor,
            k_states: torch.Tensor,
            q_weight: torch.Tensor,
            k_weight: torch.Tensor,
            q_eps: float,
            k_eps: float,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ):
            if q_states.ndim != 4 or k_states.ndim != 4:
                raise RuntimeError("prefill qk norm+rotary fast path requires 4D q/k tensors")
            if q_states.device != k_states.device or q_states.dtype != k_states.dtype:
                raise RuntimeError("prefill qk norm+rotary requires q/k on same device with same dtype")
            if not q_states.is_cuda or q_states.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError("prefill qk norm+rotary requires CUDA fp16/bf16 tensors")

            width = int(q_states.shape[-1])
            if width <= 0 or width % 2 != 0 or int(k_states.shape[-1]) != width:
                raise RuntimeError(
                    f"prefill qk norm+rotary requires matching even head_dim, got q={tuple(q_states.shape)}, k={tuple(k_states.shape)}"
                )
            if cos.ndim != 3 or sin.ndim != 3:
                raise RuntimeError(f"prefill rotary expects cos/sin [B,S,D], got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")
            if cos.shape != sin.shape or cos.shape[-1] != width:
                raise RuntimeError(f"prefill rotary cos/sin shape mismatch: cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, width={width}")
            if int(cos.shape[0]) != int(q_states.shape[0]) or int(cos.shape[1]) != int(q_states.shape[1]):
                raise RuntimeError(
                    f"prefill rotary cos/sin batch/seq mismatch: q={tuple(q_states.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
                )

            q_rows_per_token = int(q_states.shape[2])
            k_rows_per_token = int(k_states.shape[2])
            if q_rows_per_token <= 0 or k_rows_per_token <= 0:
                raise RuntimeError("prefill qk norm+rotary requires positive head counts")

            q_rows_2d = q_states.reshape(-1, width)
            k_rows_2d = k_states.reshape(-1, width)
            cos_2d = cos.reshape(-1, width)
            sin_2d = sin.reshape(-1, width)
            if q_rows_2d.stride(1) != 1:
                q_rows_2d = q_rows_2d.contiguous()
            if k_rows_2d.stride(1) != 1:
                k_rows_2d = k_rows_2d.contiguous()
            if cos_2d.stride(1) != 1:
                cos_2d = cos_2d.contiguous()
            if sin_2d.stride(1) != 1:
                sin_2d = sin_2d.contiguous()

            q_out_2d = torch.empty_like(q_rows_2d)
            k_out_2d = torch.empty_like(k_rows_2d)
            half_width = width // 2
            block_half = min(max(triton.next_power_of_2(half_width), 16), 256)
            _prefill_qk_rmsnorm_rotary_pair_kernel[(max(int(q_rows_2d.shape[0]), int(k_rows_2d.shape[0])),)](
                q_rows_2d,
                k_rows_2d,
                q_weight,
                k_weight,
                cos_2d,
                sin_2d,
                q_out_2d,
                k_out_2d,
                q_rows_2d.stride(0),
                k_rows_2d.stride(0),
                cos_2d.stride(0),
                sin_2d.stride(0),
                q_out_2d.stride(0),
                k_out_2d.stride(0),
                half_width,
                int(q_rows_2d.shape[0]),
                int(k_rows_2d.shape[0]),
                q_rows_per_token,
                k_rows_per_token,
                float(q_eps),
                float(k_eps),
                STORE_BF16=q_states.dtype == torch.bfloat16,
                BLOCK_HALF=block_half,
                num_warps=_choose_warp_count(width),
            )
            return q_out_2d.view_as(q_states), k_out_2d.view_as(k_states)

        self._triton_prefill_qk_norm_rotary_pair_fn = _forward
        return self._triton_prefill_qk_norm_rotary_pair_fn

    def _run_prefill_qk_norm_rotary_pair(
        self,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        q_eps: float,
        k_eps: float,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        fast_fn = self._get_triton_prefill_qk_norm_rotary_pair()
        if fast_fn is None:
            return self._fallback_prefill_qk_norm_rotary_pair(q_states, k_states, q_weight, k_weight, q_eps, k_eps, cos, sin)
        try:
            return fast_fn(q_states, k_states, q_weight, k_weight, q_eps, k_eps, cos, sin)
        except Exception as exc:
            if hasattr(self, "_record_fastpath_fallback"):
                self._record_fastpath_fallback("prefill_qk_norm_rotary", exc)
            return self._fallback_prefill_qk_norm_rotary_pair(q_states, k_states, q_weight, k_weight, q_eps, k_eps, cos, sin)

    def _fallback_prefill_qk_norm_rotary_pair(
        self,
        q_states: torch.Tensor,
        k_states: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        q_eps: float,
        k_eps: float,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        q_norm = F.rms_norm(q_states, (int(q_states.shape[-1]),), weight=q_weight, eps=float(q_eps))
        k_norm = F.rms_norm(k_states, (int(k_states.shape[-1]),), weight=k_weight, eps=float(k_eps))
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_rot = (q_norm * cos) + (self._rotate_half_chunks(q_norm) * sin)
        k_rot = (k_norm * cos) + (self._rotate_half_chunks(k_norm) * sin)
        return q_rot, k_rot

    def _run_prefill_attention_fast(
        self,
        mod,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        num_heads = int(getattr(mod, "num_heads", getattr(mod.config, "num_attention_heads", 0)))
        kv_heads = int(
            getattr(
                mod,
                "num_key_value_heads",
                getattr(mod, "num_kv_heads", getattr(mod.config, "num_key_value_heads", num_heads)),
            )
        )
        if num_heads <= 0 or kv_heads <= 0:
            raise RuntimeError("无法读取 prefill attention 头数")

        q_shape = (*input_shape, num_heads, mod.head_dim)
        kv_shape = (*input_shape, kv_heads, mod.head_dim)

        if hasattr(mod, "qkv_proj"):
            qkv_out = mod.qkv_proj(hidden_states)
            split_sizes = getattr(mod, "_packed_qkv_split_sizes", None)
            if split_sizes is None:
                q_dim = num_heads * mod.head_dim
                kv_dim = kv_heads * mod.head_dim
                split_sizes = (q_dim, kv_dim, kv_dim)
            q_out, k_out, v_out = torch.split(qkv_out, split_sizes, dim=-1)
        elif hasattr(mod, "_packed_qkv_weight"):
            qkv_out = F.linear(hidden_states, mod._packed_qkv_weight, mod._packed_qkv_bias)
            q_out, k_out, v_out = torch.split(qkv_out, mod._packed_qkv_split_sizes, dim=-1)
        else:
            q_out = mod.q_proj(hidden_states)
            k_out = mod.k_proj(hidden_states)
            v_out = mod.v_proj(hidden_states)

        q_states = q_out.view(q_shape)
        k_states = k_out.view(kv_shape)
        v_states = v_out.view(kv_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_states, k_states = self._run_prefill_qk_norm_rotary_pair(
            q_states,
            k_states,
            mod.q_norm.weight,
            mod.k_norm.weight,
            float(mod.q_norm.variance_epsilon),
            float(mod.k_norm.variance_epsilon),
            cos,
            sin,
        )
        query_states = q_states.transpose(1, 2)
        key_states = k_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, v_states = past_key_values.update(key_states, v_states, mod.layer_idx)

        attention_registry = _QWEN3_VL_MODELING.ALL_ATTENTION_FUNCTIONS
        if hasattr(attention_registry, "get_interface"):
            attention_interface = attention_registry.get_interface(
                mod.config._attn_implementation,
                _QWEN3_VL_MODELING.eager_attention_forward,
            )
        else:
            attention_interface = attention_registry.get(
                mod.config._attn_implementation,
                _QWEN3_VL_MODELING.eager_attention_forward,
            )
        attn_output, attn_weights = attention_interface(
            mod,
            query_states,
            key_states,
            v_states,
            attention_mask,
            dropout=0.0 if not mod.training else mod.attention_dropout,
            scaling=mod.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = mod.o_proj(attn_output)
        return attn_output, attn_weights

    def _enable_prefill_qk_norm_rotary_attention(self):
        patched = 0
        for name, mod in self._model.named_modules():
            if type(mod).__name__ != "Qwen3VLTextAttention":
                continue
            if getattr(mod, "_prefill_qk_norm_rotary_enabled", False):
                continue

            orig_forward = mod.forward

            def _forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values=None,
                _m=mod,
                _orig=orig_forward,
                _owner=self,
                _module_name=name,
                **kwargs,
            ):
                if (
                    _m.training
                    or hidden_states is None
                    or hidden_states.ndim != 3
                    or int(hidden_states.shape[1]) <= 1
                    or position_embeddings is None
                    or len(position_embeddings) != 2
                ):
                    return _orig(
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values=past_key_values,
                        **kwargs,
                    )

                try:
                    return _owner._run_prefill_attention_fast(
                        _m,
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values=past_key_values,
                        **kwargs,
                    )
                except Exception as exc:
                    if hasattr(_owner, "_record_fastpath_fallback"):
                        _owner._record_fastpath_fallback("prefill_attention_fast", exc)
                    if _module_name not in _owner._prefill_qk_rotary_warned_modules:
                        print(f"[VLMModel] {_module_name} prefill qk_norm+rotary attention 回退到原始实现: {exc}")
                        _owner._prefill_qk_rotary_warned_modules.add(_module_name)
                    return _orig(
                        hidden_states,
                        position_embeddings,
                        attention_mask,
                        past_key_values=past_key_values,
                        **kwargs,
                    )

            mod._orig_prefill_qk_norm_rotary_forward = orig_forward
            mod.forward = _forward
            mod._prefill_qk_norm_rotary_enabled = True
            patched += 1

        if patched > 0 and "prefill_qk_norm_rotary_attention" not in self._optimizations_applied:
            self._optimizations_applied.append("prefill_qk_norm_rotary_attention")
        print(f"[VLMModel] prefill qk_norm+rotary attention 已启用: Attention={patched}")
