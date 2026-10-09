"""Fast per-row (per-token) int8 activation quantization for A8W8 prefill GEMMs.

A single Triton kernel reads a bf16 [M,K] activation, computes per-row absmax,
and writes int8 [M,K] + fp32 per-row scale (absmax/127). This replaces the ~5-kernel
eager path (abs/amax/div/round/clamp/to) that otherwise erases the int8 GEMM win.

Also provides rms_norm->int8 and (layernorm result)->int8 fused entry points so the
quantization rides on the norm's existing memory traffic.
"""
import os
import torch

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/junkrat_triton_cache")
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    triton = None; tl = None
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        try:
            from triton.language.extra.cuda.libdevice import rsqrt
        except ModuleNotFoundError:
            from triton.language.math import rsqrt


def _warps(block_size: int) -> int:
    if block_size >= 4096:
        return 16
    if block_size >= 2048:
        return 8
    return 4


if TRITON_AVAILABLE:

    @triton.jit
    def _quant_per_row_kernel(x_ptr, x_rs, q_ptr, q_rs, s_ptr, n_cols,
                              BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_rs + cols, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        scale = amax / 127.0
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = tl.extra.cuda.libdevice.round(x * inv)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(q_ptr + row * q_rs + cols, q, mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _layernorm_quant_kernel(x_ptr, x_rs, w_ptr, b_ptr, q_ptr, q_rs, s_ptr, n_cols, eps,
                                BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_rs + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / n_cols
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / n_cols
        xn = xc * rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = xn * w + b
        amax = tl.max(tl.abs(y), axis=0)
        scale = amax / 127.0
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(y * inv), -127.0), 127.0).to(tl.int8)
        tl.store(q_ptr + row * q_rs + cols, q, mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _gelu_tanh_quant_kernel(x_ptr, x_rs, q_ptr, q_rs, s_ptr, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_rs + cols, mask=mask, other=0.0).to(tl.float32)
        # gelu_pytorch_tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
        inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
        amax = tl.max(tl.abs(y), axis=0)
        scale = amax / 127.0
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.round(y * inv), -127.0), 127.0).to(tl.int8)
        tl.store(q_ptr + row * q_rs + cols, q, mask=mask)
        tl.store(s_ptr + row, scale)

    @triton.jit
    def _rms_norm_quant_kernel(x_ptr, x_rs, w_ptr, q_ptr, q_rs, s_ptr, n_cols, eps,
                               BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * x_rs + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / n_cols
        xn = x * rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = xn * w
        amax = tl.max(tl.abs(y), axis=0)
        scale = amax / 127.0
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = tl.extra.cuda.libdevice.round(y * inv)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(q_ptr + row * q_rs + cols, q, mask=mask)
        tl.store(s_ptr + row, scale)


def quant_per_row_i8(x: torch.Tensor):
    """x: [M,K] (any float). returns (q_i8 [M,K], scale_f32 [M])."""
    assert x.dim() == 2 and x.is_cuda
    M, K = x.shape
    x = x.contiguous()
    q = torch.empty((M, K), device=x.device, dtype=torch.int8)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(K)
    _quant_per_row_kernel[(M,)](x, x.stride(0), q, q.stride(0), s, K,
                                BLOCK=BLOCK, num_warps=_warps(BLOCK))
    return q, s


def rms_norm_quant_i8(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """RMSNorm(x)*weight, then per-row int8 quant — fused. x:[M,K]. returns (q,s)."""
    assert x.dim() == 2 and x.is_cuda
    M, K = x.shape
    x = x.contiguous()
    q = torch.empty((M, K), device=x.device, dtype=torch.int8)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(K)
    _rms_norm_quant_kernel[(M,)](x, x.stride(0), weight, q, q.stride(0), s, K, eps,
                                 BLOCK=BLOCK, num_warps=_warps(BLOCK))
    return q, s


# ─── A8W8 int8 linear (per-token act × per-channel weight) ──────────────────────
_acext = None


def _get_acext():
    global _acext
    if _acext is None:
        import acext as _a
        _acext = _a
    return _acext


def quantize_weight_per_channel_i8(weight: torch.Tensor):
    """weight [N,K] (out,in) -> (wq int8 [N,K], scale f32 [N]) per output channel."""
    w = weight.detach().to(torch.float32)
    s = (w.abs().amax(dim=1).clamp_min(1e-8) / 127.0)
    wq = torch.round(w / s[:, None]).clamp_(-127, 127).to(torch.int8).contiguous()
    return wq, s.to(torch.float32).contiguous()


def int8_linear_fused(x: torch.Tensor, weight_i8: torch.Tensor, weight_scale: torch.Tensor,
                      bias: torch.Tensor | None = None, out_dtype=torch.bfloat16):
    """Per-token int8 activation quant (fused triton) + acext int8 GEMM + optional bias.
    x: [..., K]; weight_i8: [N,K]; weight_scale: [N]. returns [..., N]."""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    q, s = quant_per_row_i8(x2)
    out = _get_acext().int8_gemm(q, weight_i8, weight_scale, s, bias, out_dtype)
    return out.view(*shp[:-1], weight_i8.shape[0])


# Custom op so torch.compile sees ONE opaque node (the inner triton quant kernel
# otherwise triggers per-shape recompiles / accumulated_recompile_limit). This keeps
# compile's glue fusion (layernorm/gelu/residual/rope) AND adds the int8 GEMM win.
INT8_LINEAR_OP = None
if TRITON_AVAILABLE and hasattr(torch.library, "custom_op"):
    try:
        @torch.library.custom_op("junkrat::int8_linear", mutates_args=())
        def _int8_linear_op(x: torch.Tensor, weight_i8: torch.Tensor,
                            weight_scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
            return int8_linear_fused(x, weight_i8, weight_scale, bias, torch.bfloat16)

        @_int8_linear_op.register_fake
        def _(x, weight_i8, weight_scale, bias):
            return x.new_empty((*x.shape[:-1], weight_i8.shape[0]), dtype=torch.bfloat16)

        INT8_LINEAR_OP = torch.ops.junkrat.int8_linear
    except Exception:
        INT8_LINEAR_OP = None


def int8_linear(x, weight_i8, weight_scale, bias=None):
    """Compile-safe entry: opaque custom op if available, else the raw fused path."""
    if INT8_LINEAR_OP is not None:
        return INT8_LINEAR_OP(x, weight_i8, weight_scale, bias)
    return int8_linear_fused(x, weight_i8, weight_scale, bias, torch.bfloat16)


def layernorm_quant_i8(x, weight, bias, eps):
    """LayerNorm(x)*w+b then per-row int8 quant — one kernel. x:[M,K] -> (q,s)."""
    M, K = x.shape
    x = x.contiguous()
    q = torch.empty((M, K), device=x.device, dtype=torch.int8)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(K)
    _layernorm_quant_kernel[(M,)](x, x.stride(0), weight, bias, q, q.stride(0), s, K, eps,
                                  BLOCK=BLOCK, num_warps=_warps(BLOCK))
    return q, s


def gelu_quant_i8(x):
    """gelu_pytorch_tanh(x) then per-row int8 quant — one kernel. x:[M,K] -> (q,s)."""
    M, K = x.shape
    x = x.contiguous()
    q = torch.empty((M, K), device=x.device, dtype=torch.int8)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(K)
    _gelu_tanh_quant_kernel[(M,)](x, x.stride(0), q, q.stride(0), s, K,
                                  BLOCK=BLOCK, num_warps=_warps(BLOCK))
    return q, s


def _ln_int8_linear_impl(x, ln_w, ln_b, eps, weight_i8, weight_scale, bias):
    shp = x.shape
    q, s = layernorm_quant_i8(x.reshape(-1, shp[-1]), ln_w, ln_b, eps)
    out = _get_acext().int8_gemm(q, weight_i8, weight_scale, s, bias, torch.bfloat16)
    return out.view(*shp[:-1], weight_i8.shape[0])


def _gelu_int8_linear_impl(x, weight_i8, weight_scale, bias):
    shp = x.shape
    q, s = gelu_quant_i8(x.reshape(-1, shp[-1]))
    out = _get_acext().int8_gemm(q, weight_i8, weight_scale, s, bias, torch.bfloat16)
    return out.view(*shp[:-1], weight_i8.shape[0])


LN_INT8_LINEAR_OP = None
GELU_INT8_LINEAR_OP = None
if TRITON_AVAILABLE and hasattr(torch.library, "custom_op"):
    try:
        @torch.library.custom_op("junkrat::ln_int8_linear", mutates_args=())
        def _ln_int8_op(x: torch.Tensor, ln_w: torch.Tensor, ln_b: torch.Tensor, eps: float,
                        weight_i8: torch.Tensor, weight_scale: torch.Tensor,
                        bias: torch.Tensor | None) -> torch.Tensor:
            return _ln_int8_linear_impl(x, ln_w, ln_b, eps, weight_i8, weight_scale, bias)

        @_ln_int8_op.register_fake
        def _(x, ln_w, ln_b, eps, weight_i8, weight_scale, bias):
            return x.new_empty((*x.shape[:-1], weight_i8.shape[0]), dtype=torch.bfloat16)

        @torch.library.custom_op("junkrat::gelu_int8_linear", mutates_args=())
        def _gelu_int8_op(x: torch.Tensor, weight_i8: torch.Tensor,
                          weight_scale: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
            return _gelu_int8_linear_impl(x, weight_i8, weight_scale, bias)

        @_gelu_int8_op.register_fake
        def _(x, weight_i8, weight_scale, bias):
            return x.new_empty((*x.shape[:-1], weight_i8.shape[0]), dtype=torch.bfloat16)

        LN_INT8_LINEAR_OP = torch.ops.junkrat.ln_int8_linear
        GELU_INT8_LINEAR_OP = torch.ops.junkrat.gelu_int8_linear
    except Exception:
        LN_INT8_LINEAR_OP = None
        GELU_INT8_LINEAR_OP = None


def ln_int8_linear(x, ln_w, ln_b, eps, weight_i8, weight_scale, bias=None):
    """Fused LayerNorm -> int8 quant -> int8 GEMM (+bias). Compile-safe opaque op."""
    if LN_INT8_LINEAR_OP is not None:
        return LN_INT8_LINEAR_OP(x, ln_w, ln_b, eps, weight_i8, weight_scale, bias)
    return _ln_int8_linear_impl(x, ln_w, ln_b, eps, weight_i8, weight_scale, bias)


def gelu_int8_linear(x, weight_i8, weight_scale, bias=None):
    """Fused gelu_tanh -> int8 quant -> int8 GEMM (+bias). Compile-safe opaque op."""
    if GELU_INT8_LINEAR_OP is not None:
        return GELU_INT8_LINEAR_OP(x, weight_i8, weight_scale, bias)
    return _gelu_int8_linear_impl(x, weight_i8, weight_scale, bias)
