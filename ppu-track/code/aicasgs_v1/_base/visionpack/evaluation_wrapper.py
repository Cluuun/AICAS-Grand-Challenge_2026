from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_BASE_WRAPPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "qkrotary"
    / "evaluation_wrapper.py"
)
_BASE_MODULE_NAME = "aicas_prefill_gpuvisionnorm_prefillqkrotary_visionpack_20260417"
_BASE_SPEC = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_WRAPPER_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"无法加载基础 wrapper: {_BASE_WRAPPER_PATH}")
_BASE_MODULE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE_MODULE)

triton = getattr(_BASE_MODULE, "triton", None)
tl = getattr(_BASE_MODULE, "tl", None)
_choose_warp_count = getattr(_BASE_MODULE, "_choose_warp_count", lambda col_count: 4)


if triton is not None and tl is not None:

    @triton.jit
    def _vision_qkv_rotary_pack_kernel(
        qkv_ptr,
        cos_ptr,
        sin_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        qkv_row_stride,
        cos_row_stride,
        sin_row_stride,
        q_head_stride,
        q_seq_stride,
        k_head_stride,
        k_seq_stride,
        v_head_stride,
        v_seq_stride,
        half_width,
        embed_dim,
        num_heads,
        total_rows,
        STORE_BF16: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= total_rows:
            return

        offs = tl.arange(0, BLOCK_HALF)
        mask = offs < half_width
        out_dtype = tl.bfloat16 if STORE_BF16 else tl.float16

        seq_idx = row_idx // num_heads
        head_idx = row_idx - seq_idx * num_heads
        head_offset = head_idx * (half_width * 2)
        qkv_base = qkv_ptr + seq_idx * qkv_row_stride + head_offset

        cos_left = tl.load(cos_ptr + seq_idx * cos_row_stride + offs, mask=mask, other=1.0).to(tl.float32)
        cos_right = tl.load(
            cos_ptr + seq_idx * cos_row_stride + half_width + offs,
            mask=mask,
            other=1.0,
        ).to(tl.float32)
        sin_left = tl.load(sin_ptr + seq_idx * sin_row_stride + offs, mask=mask, other=0.0).to(tl.float32)
        sin_right = tl.load(
            sin_ptr + seq_idx * sin_row_stride + half_width + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        q_left = tl.load(qkv_base + offs, mask=mask, other=0.0).to(tl.float32)
        q_right = tl.load(qkv_base + half_width + offs, mask=mask, other=0.0).to(tl.float32)
        q_out_left = q_left * cos_left - q_right * sin_left
        q_out_right = q_right * cos_right + q_left * sin_right

        k_base = qkv_base + embed_dim
        k_left = tl.load(k_base + offs, mask=mask, other=0.0).to(tl.float32)
        k_right = tl.load(k_base + half_width + offs, mask=mask, other=0.0).to(tl.float32)
        k_out_left = k_left * cos_left - k_right * sin_left
        k_out_right = k_right * cos_right + k_left * sin_right

        v_base = qkv_base + embed_dim * 2
        v_left = tl.load(v_base + offs, mask=mask, other=0.0).to(tl.float32)
        v_right = tl.load(v_base + half_width + offs, mask=mask, other=0.0).to(tl.float32)

        q_out_base = q_ptr + head_idx * q_head_stride + seq_idx * q_seq_stride
        k_out_base = k_ptr + head_idx * k_head_stride + seq_idx * k_seq_stride
        v_out_base = v_ptr + head_idx * v_head_stride + seq_idx * v_seq_stride

        tl.store(q_out_base + offs, q_out_left.to(out_dtype), mask=mask)
        tl.store(q_out_base + half_width + offs, q_out_right.to(out_dtype), mask=mask)
        tl.store(k_out_base + offs, k_out_left.to(out_dtype), mask=mask)
        tl.store(k_out_base + half_width + offs, k_out_right.to(out_dtype), mask=mask)
        tl.store(v_out_base + offs, v_left.to(out_dtype), mask=mask)
        tl.store(v_out_base + half_width + offs, v_right.to(out_dtype), mask=mask)

else:
    _vision_qkv_rotary_pack_kernel = None


class VLMModel(_BASE_MODULE.VLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vision_pack_warned_modules = set()
        self._enable_fused_vision_qkv_rotary_attention()

    def _get_triton_vision_qkv_rotary_pack(self):
        if hasattr(self, "_triton_vision_qkv_rotary_pack_fn"):
            return self._triton_vision_qkv_rotary_pack_fn

        if triton is None or tl is None or _vision_qkv_rotary_pack_kernel is None:
            self._triton_vision_qkv_rotary_pack_fn = None
            self._triton_vision_qkv_rotary_pack_reason = "triton vision qkv+rotary pack kernel unavailable"
            return None

        def _forward(
            qkv: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            num_heads: int,
            head_dim: int,
        ):
            if qkv.ndim != 2:
                raise RuntimeError(f"vision qkv pack expects [seq, 3*hidden], got {tuple(qkv.shape)}")
            if cos.ndim != 2 or sin.ndim != 2:
                raise RuntimeError(f"vision rotary expects cos/sin [seq, head_dim], got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")
            if qkv.device != cos.device or qkv.device != sin.device:
                raise RuntimeError("vision qkv pack requires qkv/cos/sin on same device")
            if not qkv.is_cuda or qkv.dtype not in (torch.float16, torch.bfloat16):
                raise RuntimeError("vision qkv pack requires CUDA fp16/bf16 tensors")

            seq_len = int(qkv.shape[0])
            embed_dim = int(num_heads) * int(head_dim)
            if seq_len <= 0 or embed_dim <= 0:
                raise RuntimeError("vision qkv pack got non-positive seq/head dimensions")
            if int(qkv.shape[1]) != embed_dim * 3:
                raise RuntimeError(
                    f"vision qkv pack shape mismatch: qkv={tuple(qkv.shape)}, num_heads={num_heads}, head_dim={head_dim}"
                )
            if int(cos.shape[0]) != seq_len or int(sin.shape[0]) != seq_len:
                raise RuntimeError(f"vision rotary seq mismatch: qkv={tuple(qkv.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}")
            if cos.shape != sin.shape or int(cos.shape[1]) != int(head_dim):
                raise RuntimeError(
                    f"vision rotary head_dim mismatch: cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, head_dim={head_dim}"
                )
            if int(head_dim) % 2 != 0:
                raise RuntimeError(f"vision qkv pack requires even head_dim, got {head_dim}")

            if qkv.stride(1) != 1:
                qkv = qkv.contiguous()
            if cos.stride(1) != 1:
                cos = cos.contiguous()
            if sin.stride(1) != 1:
                sin = sin.contiguous()

            q_out = torch.empty((int(num_heads), seq_len, int(head_dim)), device=qkv.device, dtype=qkv.dtype)
            k_out = torch.empty_like(q_out)
            v_out = torch.empty_like(q_out)
            half_width = int(head_dim) // 2
            block_half = min(max(triton.next_power_of_2(half_width), 16), 128)
            total_rows = seq_len * int(num_heads)

            _vision_qkv_rotary_pack_kernel[(total_rows,)](
                qkv,
                cos,
                sin,
                q_out,
                k_out,
                v_out,
                qkv.stride(0),
                cos.stride(0),
                sin.stride(0),
                q_out.stride(0),
                q_out.stride(1),
                k_out.stride(0),
                k_out.stride(1),
                v_out.stride(0),
                v_out.stride(1),
                half_width,
                embed_dim,
                int(num_heads),
                total_rows,
                STORE_BF16=qkv.dtype == torch.bfloat16,
                BLOCK_HALF=block_half,
                num_warps=_choose_warp_count(head_dim),
            )
            return q_out, k_out, v_out

        self._triton_vision_qkv_rotary_pack_fn = _forward
        return self._triton_vision_qkv_rotary_pack_fn

    def _run_vision_qkv_rotary_pack(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ):
        fast_fn = self._get_triton_vision_qkv_rotary_pack()
        if fast_fn is None:
            raise RuntimeError(getattr(self, "_triton_vision_qkv_rotary_pack_reason", "vision pack unavailable"))
        return fast_fn(qkv, cos, sin, num_heads, head_dim)

    def _enable_fused_vision_qkv_rotary_attention(self):
        triton_attn = self._get_triton_vision_attention()
        fast_pack = self._get_triton_vision_qkv_rotary_pack()
        if triton_attn is None or fast_pack is None:
            reason = getattr(self, "_triton_vision_qkv_rotary_pack_reason", "unknown")
            print(f"[VLMModel] 跳过 vision qkv+rotary pack attention: {reason}")
            return 0

        patched = 0
        for name, mod in self._model.named_modules():
            if type(mod).__name__ != "Qwen3VLVisionAttention":
                continue
            if getattr(mod, "_vision_qkv_rotary_pack_enabled", False):
                continue

            orig_forward = mod.forward

            def _attn_forward(
                hidden_states,
                cu_seqlens,
                rotary_pos_emb=None,
                position_embeddings=None,
                _m=mod,
                _orig=orig_forward,
                _owner=self,
                _module_name=name,
                _triton_attn=triton_attn,
                **kwargs,
            ):
                if (
                    not _owner._should_use_fast_vision_attention()
                    or _m.training
                    or hidden_states is None
                    or hidden_states.ndim != 2
                    or position_embeddings is None
                    or len(position_embeddings) != 2
                    or cu_seqlens is None
                    or cu_seqlens.dim() != 1
                    or int(cu_seqlens.numel()) != 2
                ):
                    return _orig(
                        hidden_states,
                        cu_seqlens,
                        rotary_pos_emb=rotary_pos_emb,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

                try:
                    seq_length, embed_dim = hidden_states.shape
                    num_heads = int(getattr(_m, "num_heads", 0))
                    head_dim = int(getattr(_m, "head_dim", 0))
                    if seq_length <= 0 or embed_dim != num_heads * head_dim:
                        raise RuntimeError(
                            f"invalid vision attention dims: hidden={tuple(hidden_states.shape)}, heads={num_heads}, head_dim={head_dim}"
                        )

                    cos, sin = position_embeddings
                    if cos.ndim == 3 and int(cos.shape[1]) == 1:
                        cos = cos.squeeze(1)
                    if sin.ndim == 3 and int(sin.shape[1]) == 1:
                        sin = sin.squeeze(1)

                    qkv = _m.qkv(hidden_states)
                    q_states, k_states, v_states = _owner._run_vision_qkv_rotary_pack(
                        qkv,
                        cos,
                        sin,
                        num_heads,
                        head_dim,
                    )

                    attn_output = _triton_attn(
                        q_states.unsqueeze(0),
                        k_states.unsqueeze(0),
                        v_states.unsqueeze(0),
                        float(_m.scaling),
                    )
                    attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, embed_dim).contiguous()
                    return _m.proj(attn_output)
                except Exception as exc:
                    if hasattr(_owner, "_record_fastpath_fallback"):
                        _owner._record_fastpath_fallback("vision_qkv_rotary_pack_attention", exc)
                    if _module_name not in _owner._vision_pack_warned_modules:
                        print(f"[VLMModel] {_module_name} vision qkv+rotary pack attention 回退到原始实现: {exc}")
                        _owner._vision_pack_warned_modules.add(_module_name)
                    return _orig(
                        hidden_states,
                        cu_seqlens,
                        rotary_pos_emb=rotary_pos_emb,
                        position_embeddings=position_embeddings,
                        **kwargs,
                    )

            mod._orig_vision_qkv_rotary_pack_forward = orig_forward
            mod.forward = _attn_forward
            mod._vision_qkv_rotary_pack_enabled = True
            patched += 1

        if patched > 0 and "vision_qkv_rotary_pack_attention" not in self._optimizations_applied:
            self._optimizations_applied.append("vision_qkv_rotary_pack_attention")
        print(f"[VLMModel] vision qkv+rotary pack attention 已启用: Attention={patched}")
        return patched
