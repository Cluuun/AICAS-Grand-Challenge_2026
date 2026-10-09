"""把自定义cuda/triton算子注册成torch.library.custom_op 这样torch.compile不会图断裂"""

import torch


def _get_cuda_ops():
    """懒加载cuda算子"""
    try:
        from custom_kernels.cuda import get_cuda_ops
        ops = get_cuda_ops()
        if ops is not None:
            return ops
    except Exception:
        pass
    return None


def _register_ops():
    """把所有自定义算子注册成torch.library.custom_op"""

    # 1. RMS Norm 归一化
    @torch.library.custom_op('custom::rms_norm', mutates_args=())
    def rms_norm_op(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        ops = _get_cuda_ops()
        if ops is not None:
            return ops.rms_norm(x, weight, eps)
        from custom_kernels.cuda.triton_kernels import triton_rms_norm
        return triton_rms_norm(x, weight, eps)

    @rms_norm_op.register_fake
    def _(x, weight, eps):
        return torch.empty_like(x)

    # 2. SiLU乘法融合
    @torch.library.custom_op('custom::silu_and_mul', mutates_args=())
    def silu_and_mul_op(gate_up: torch.Tensor) -> torch.Tensor:
        ops = _get_cuda_ops()
        if ops is not None:
            return ops.silu_and_mul(gate_up)
        from custom_kernels.cuda.triton_kernels import triton_silu_and_mul
        return triton_silu_and_mul(gate_up)

    @silu_and_mul_op.register_fake
    def _(gate_up):
        d = gate_up.shape[-1] // 2
        return torch.empty(gate_up.shape[:-1] + (d,), dtype=gate_up.dtype, device=gate_up.device)

    # 3. 残差加法+RMS Norm融合
    @torch.library.custom_op('custom::fused_add_rms_norm', mutates_args=())
    def fused_add_rms_norm_op(
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ops = _get_cuda_ops()
        if ops is not None:
            return ops.fused_add_rms_norm(input_tensor, residual, weight, epsilon)
        from custom_kernels.cuda.triton_kernels import triton_fused_add_rms_norm
        return triton_fused_add_rms_norm(input_tensor, residual, weight, epsilon)

    @fused_add_rms_norm_op.register_fake
    def _(input_tensor, residual, weight, epsilon):
        return torch.empty_like(input_tensor), torch.empty_like(residual)

    # 4. 融合RoPE（decode阶段，seq=1）
    @torch.library.custom_op('custom::fused_rope', mutates_args=())
    def fused_rope_op(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from custom_kernels.cuda.triton_kernels import triton_fused_rope
        return triton_fused_rope(query_states, key_states, cos, sin)

    @fused_rope_op.register_fake
    def _(query_states, key_states, cos, sin):
        return torch.empty_like(query_states), torch.empty_like(key_states)

    # 5. 融合RoPE（prefill阶段，seq>1）
    @torch.library.custom_op('custom::fused_rope_prefill', mutates_args=())
    def fused_rope_prefill_op(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from custom_kernels.cuda.triton_kernels import triton_fused_rope_prefill
        return triton_fused_rope_prefill(query_states, key_states, cos, sin)

    @fused_rope_prefill_op.register_fake
    def _(query_states, key_states, cos, sin):
        return torch.empty_like(query_states), torch.empty_like(key_states)

    # 6. QKV归一化+RoPE融合（prefill）
    @torch.library.custom_op('custom::fused_qkv_norm_rope_prefill', mutates_args=())
    def fused_qkv_norm_rope_prefill_op(
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        q_dim: int,
        kv_dim: int,
        head_dim: int,
        eps_q: float = 1e-6,
        eps_k: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from custom_kernels.cuda.triton_kernels import triton_fused_qkv_norm_rope_prefill
        return triton_fused_qkv_norm_rope_prefill(
            qkv, q_weight, k_weight, cos, sin,
            q_dim, kv_dim, head_dim, eps_q, eps_k,
        )

    @fused_qkv_norm_rope_prefill_op.register_fake
    def _(qkv, q_weight, k_weight, cos, sin, q_dim, kv_dim, head_dim, eps_q, eps_k):
        batch = qkv.shape[0]
        seq = qkv.shape[1]
        n_q = q_dim // head_dim
        n_kv = kv_dim // head_dim
        q_out = torch.empty(batch, seq, n_q, head_dim, dtype=qkv.dtype, device=qkv.device)
        k_out = torch.empty(batch, seq, n_kv, head_dim, dtype=qkv.dtype, device=qkv.device)
        v_out = torch.empty(batch, seq, n_kv, head_dim, dtype=qkv.dtype, device=qkv.device)
        return q_out, k_out, v_out

    # 7. 视觉RoPE
    @torch.library.custom_op('custom::vision_rope', mutates_args=())
    def vision_rope_op(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from custom_kernels.cuda.triton_kernels import triton_vision_rope
        return triton_vision_rope(query_states, key_states, cos, sin)

    @vision_rope_op.register_fake
    def _(query_states, key_states, cos, sin):
        return torch.empty_like(query_states), torch.empty_like(key_states)

    # 8. 视觉LayerNorm
    @torch.library.custom_op('custom::vision_layer_norm', mutates_args=())
    def vision_layer_norm_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        from custom_kernels.vision.opt import fused_layer_norm
        return fused_layer_norm(x, weight, bias, eps)

    @vision_layer_norm_op.register_fake
    def _(x, weight, bias, eps):
        return torch.empty_like(x)

    # 9. 视觉残差加法+LayerNorm融合
    @torch.library.custom_op('custom::vision_fused_add_layer_norm', mutates_args=())
    def vision_fused_add_layer_norm_op(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from custom_kernels.vision.opt import fused_add_layer_norm
        return fused_add_layer_norm(x, residual, weight, bias, eps)

    @vision_fused_add_layer_norm_op.register_fake
    def _(x, residual, weight, bias, eps):
        return torch.empty_like(x), torch.empty_like(residual)

    # 10. 视觉GELU
    @torch.library.custom_op('custom::vision_gelu', mutates_args=())
    def vision_gelu_op(x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        return F.gelu(x)  # 直接用PyTorch的GELU，省去clone开销

    @vision_gelu_op.register_fake
    def _(x):
        return torch.empty_like(x)


# 导入时自动注册
_register_ops()
