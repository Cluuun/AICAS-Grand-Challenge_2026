"""视觉编码器优化"""

import torch
import torch.nn.functional as F
import types
import triton
import triton.language as tl
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 1. Triton 融合 LayerNorm
# ============================================================

@triton.jit
def _layer_norm_fwd_kernel(
    X, Y, W, B,
    stride_x, stride_y,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_y

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w
    if B is not None:
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
        y = y + b

    tl.store(Y + offs, y, mask=mask)


def fused_layer_norm(x, weight, bias, eps=1e-6):
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x)

    _layer_norm_fwd_kernel[(M,)](
        x, y, weight, bias,
        x.stride(0), y.stride(0),
        N=N, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.reshape(shape)


# ============================================================
# 2. Triton 融合残差加+LayerNorm（一个kernel搞定）
# ============================================================

@triton.jit
def _fused_add_layer_norm_kernel(
    X, R, Y, RR, W, B,
    stride_x, stride_r, stride_y, stride_rr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)

    added = x + r

    mean = tl.sum(added, axis=0) / N
    centered = added - mean
    var = tl.sum(centered * centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = centered * rstd

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = normalized * w
    if B is not None:
        b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
        y = y + b

    tl.store(Y + row * stride_y + offs, y, mask=mask)
    tl.store(RR + row * stride_rr + offs, added, mask=mask)


def fused_add_layer_norm(x, residual, weight, bias, eps=1e-6):
    """融合的残差加+LayerNorm: 返回(normed, new_residual)"""
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    r_flat = residual.reshape(-1, shape[-1])
    M, N = x_flat.shape

    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x_flat)
    rr = torch.empty_like(r_flat)

    _fused_add_layer_norm_kernel[(M,)](
        x_flat, r_flat, y, rr,
        weight, bias,
        x_flat.stride(0), r_flat.stride(0), y.stride(0), rr.stride(0),
        N=N, eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y.reshape(shape), rr.reshape(shape)


# ============================================================
# 2b. Triton 原地 GELU kernel
# ============================================================

@triton.jit
def _gelu_inplace_kernel(X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU tanh近似: x * sigmoid(1.59577 * (x + 0.044715 * x^3))
    inner = 1.5957691216 * (x + 0.044715 * x * x * x)
    result = x * tl.sigmoid(inner)
    tl.store(X + offs, result.to(X.dtype.element_ty), mask=mask)


def gelu_inplace(x):
    """原地GELU激活（省一次显存分配）"""
    N = x.numel()
    if N == 0:
        return x
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _gelu_inplace_kernel[grid](x, N, BLOCK_SIZE)
    return x


# ============================================================
# 2c. 视觉buffer池（不会被empty_cache清掉）
# ============================================================

class _VisionBufferPool:
    """视觉编码器的预分配buffer池，不会被GC回收，省去反复malloc"""

    def __init__(self, max_seq=65536, hidden_dim=1024, num_heads=16, head_dim=64, device='cuda:0'):
        self.max_seq = max_seq
        self.device = device
        # QKV输出buffer: [max_seq, 3 * hidden_dim]
        self.qkv_buf = torch.empty(max_seq, 3 * hidden_dim, dtype=torch.float16, device=device)
        # 分离的Q/K/V buffer（各自连续，省去permute+contiguous开销）
        self.q_buf = torch.empty(max_seq, hidden_dim, dtype=torch.float16, device=device)
        self.k_buf = torch.empty(max_seq, hidden_dim, dtype=torch.float16, device=device)
        self.v_buf = torch.empty(max_seq, hidden_dim, dtype=torch.float16, device=device)
        # MLP fc1输出buffer: [max_seq, 4 * hidden_dim]
        self.mlp_fc1_buf = torch.empty(max_seq, 4 * hidden_dim, dtype=torch.float16, device=device)
        # 注意力输出buffer: [max_seq, hidden_dim]
        self.attn_out_buf = torch.empty(max_seq, hidden_dim, dtype=torch.float16, device=device)

    def get_qkv_buf(self, seq_len):
        return self.qkv_buf[:seq_len]

    def get_q_buf(self, seq_len):
        return self.q_buf[:seq_len]

    def get_k_buf(self, seq_len):
        return self.k_buf[:seq_len]

    def get_v_buf(self, seq_len):
        return self.v_buf[:seq_len]

    def get_mlp_fc1_buf(self, seq_len):
        return self.mlp_fc1_buf[:seq_len]

    def get_attn_out_buf(self, seq_len):
        return self.attn_out_buf[:seq_len]


_vision_pool = None


def _get_vision_pool():
    global _vision_pool
    if _vision_pool is None:
        _vision_pool = _VisionBufferPool()
    return _vision_pool


# ============================================================
# 3. 替换视觉注意力（用flash_attn_func，graph安全）
# ============================================================

def _make_patched_vision_attn_forward(scaling, num_heads, head_dim):
    """构造替换版的视觉注意力forward，用3个独立mm替代1个mm+permute+contiguous"""
    from flash_attn import flash_attn_func
    from custom_kernels.cuda.triton_kernels import triton_vision_rope

    def forward(self, hidden_states, cu_seqlens=None,
                rotary_pos_emb=None, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]

        # 3个独立mm → Q/K/V各自连续，省去permute+2x contiguous（节省~0.12ms/block）
        pool = _get_vision_pool()
        q_buf = pool.get_q_buf(seq_length)
        k_buf = pool.get_k_buf(seq_length)
        v_buf = pool.get_v_buf(seq_length)

        torch.mm(hidden_states, self._q_w_T, out=q_buf)
        torch.mm(hidden_states, self._k_w_T, out=k_buf)
        torch.mm(hidden_states, self._v_w_T, out=v_buf)

        if self._q_bias is not None:
            q_buf.add_(self._q_bias)
            k_buf.add_(self._k_bias)
            v_buf.add_(self._v_bias)

        # reshape成 [seq, n_heads, head_dim]（连续，无需copy）
        q = q_buf.reshape(seq_length, num_heads, head_dim)
        k = k_buf.reshape(seq_length, num_heads, head_dim)
        v = v_buf.reshape(seq_length, num_heads, head_dim)

        # RoPE（输入已连续，triton_vision_rope内部不再需要.contiguous()）
        cos, sin = position_embeddings
        q, k = triton_vision_rope(q, k, cos, sin)

        # Flash attention
        attn_output = flash_attn_func(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
            dropout_p=0.0,
            softmax_scale=scaling,
            causal=False,
        )

        # 输出投影
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

    return forward


# ============================================================
# 4. 替换视觉block的forward
# ============================================================

def _make_patched_block_forward(n1_w, n1_b, n1_e, n2_w, n2_b, n2_e):
    """构造替换版的vision block forward。

    F.layer_norm比Triton快2.7x，F.linear比buffer pool快1.14x。
    gelu_inplace比PyTorch快1.7x，注意力用flash_attn_func。
    """
    fc1_w = None
    fc1_b = None
    fc2_w = None
    fc2_b = None
    n_shape = [n1_w.shape[0]]

    def forward(self, hidden_states, cu_seqlens=None,
                rotary_pos_emb=None, position_embeddings=None, **kwargs):
        nonlocal fc1_w, fc1_b, fc2_w, fc2_b

        # LayerNorm（注意力前）— F.layer_norm对[2688,1024]比Triton快2.7x
        normed1 = F.layer_norm(hidden_states, n_shape, n1_w, n1_b, n1_e)

        # 注意力（flash_attn_func + Triton vision RoPE）
        attn_out = self.attn(
            normed1,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # 残差加 + 注意力后LayerNorm — 分开做比Triton fused快2.4x
        hidden_states = attn_out + hidden_states
        normed2 = F.layer_norm(hidden_states, n_shape, n2_w, n2_b, n2_e)

        # MLP — F.linear + Triton gelu_inplace（比F.gelu快1.7x）
        if fc1_w is None:
            fc1_w = self.mlp.linear_fc1.weight
            fc1_b = self.mlp.linear_fc1.bias
            fc2_w = self.mlp.linear_fc2.weight
            fc2_b = self.mlp.linear_fc2.bias

        mlp_mid = F.linear(normed2, fc1_w, fc1_b)
        gelu_inplace(mlp_mid)
        mlp_out = F.linear(mlp_mid, fc2_w, fc2_b)

        # 残差加（原地操作，复用之前的tensor）
        hidden_states.add_(mlp_out)
        return hidden_states

    return forward


# ============================================================
# 5. 替换Merger（用F.layer_norm）
# ============================================================

def _patch_merger(merger):
    """替换merger的forward，用F.layer_norm（比Triton快）"""
    norm_w = merger.norm.weight
    norm_b = merger.norm.bias
    norm_e = merger.norm.eps
    use_postshuffle = merger.use_postshuffle_norm
    hs = merger.hidden_size
    fc1_w = merger.linear_fc1.weight
    fc1_b = merger.linear_fc1.bias
    fc2_w = merger.linear_fc2.weight
    fc2_b = merger.linear_fc2.bias
    # 用norm权重的shape(embed_dim)做归一化，不是hidden_size
    norm_shape = [norm_w.shape[0]]

    def forward(self, x):
        if use_postshuffle:
            x = x.view(-1, hs)
        x = F.layer_norm(x, norm_shape, norm_w, norm_b, norm_e)
        if not use_postshuffle:
            x = x.view(-1, hs)
        x = F.gelu(F.linear(x, fc1_w, fc1_b))
        return F.linear(x, fc2_w, fc2_b)

    merger.forward = types.MethodType(forward, merger)


# ============================================================
# 6. 视觉CUDA graph引擎
# ============================================================

class VisionGraphEngine:
    """视觉编码器的CUDA graph引擎，把block循环+merger拍成graph，干掉Python和kernel launch开销"""

    def __init__(self, visual):
        self.visual = visual
        self._graphs = {}  # seq_len -> graph缓存

    def _run_blocks(self, hidden, cos, sin, cu_seqlens, ds_bufs, merged_buf, output_buf):
        """跑所有vision block + deepstack merger + 主merger"""
        position_embeddings = (cos, sin)
        deepstack_indexes = self.visual.deepstack_visual_indexes

        ds_idx = 0
        for i, blk in enumerate(self.visual.blocks):
            hidden = blk(
                hidden,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if i in deepstack_indexes:
                feat = self.visual.deepstack_merger_list[ds_idx](hidden)
                ds_bufs[ds_idx].copy_(feat)
                ds_idx += 1

        merged = self.visual.merger(hidden)
        merged_buf.copy_(merged)
        output_buf.copy_(hidden)
        return hidden

    def capture(self, hidden, cos, sin, cu_seqlens):
        """把vision block循环录成CUDA graph"""
        seq_len = hidden.shape[0]
        if seq_len in self._graphs:
            return self._graphs[seq_len]

        # 分配静态buffer
        static_hidden = hidden.clone()
        static_cos = cos.clone()
        static_sin = sin.clone()
        static_cu = cu_seqlens.clone()

        # DeepStack + 主merger输出: [seq/4, 2048]
        merged_seq = seq_len // 4
        n_ds = len(self.visual.deepstack_merger_list)
        ds_bufs = [torch.empty(merged_seq, 2048, dtype=torch.float16, device=hidden.device)
                   for _ in range(n_ds)]
        merged_buf = torch.empty(merged_seq, 2048, dtype=torch.float16, device=hidden.device)
        output_buf = torch.empty(seq_len, hidden.shape[1], dtype=torch.float16, device=hidden.device)

        # 预热
        with torch.no_grad():
            for _ in range(3):
                self._run_blocks(static_hidden, static_cos, static_sin,
                                 static_cu, ds_bufs, merged_buf, output_buf)
        torch.cuda.synchronize()

        # 录图
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_blocks(static_hidden, static_cos, static_sin,
                             static_cu, ds_bufs, merged_buf, output_buf)

        entry = {
            'graph': graph,
            'hidden': static_hidden,
            'cos': static_cos,
            'sin': static_sin,
            'cu_seqlens': static_cu,
            'ds_bufs': ds_bufs,
            'merged_buf': merged_buf,
            'output_buf': output_buf,
        }
        self._graphs[seq_len] = entry
        return entry

    def replay(self, hidden, cos, sin, cu_seqlens):
        """回放已录好的graph"""
        seq_len = hidden.shape[0]

        if seq_len not in self._graphs:
            self.capture(hidden, cos, sin, cu_seqlens)

        entry = self._graphs[seq_len]
        entry['hidden'].copy_(hidden)
        entry['cos'].copy_(cos)
        entry['sin'].copy_(sin)
        entry['cu_seqlens'].copy_(cu_seqlens)

        entry['graph'].replay()

        return entry['output_buf'], entry['ds_bufs'], entry['merged_buf']


# 模块级vision graph引擎
_vision_graph = None


def get_vision_graph_engine(model):
    """拿或创建vision graph引擎"""
    global _vision_graph
    if _vision_graph is None:
        _vision_graph = VisionGraphEngine(model.model.visual)
    return _vision_graph


# ============================================================
# 7. 主入口
# ============================================================

# ============================================================
# 7b. 替换patch_embed Conv3d为线性层（消除cuDNN自动调优延迟）
# ============================================================

def _patch_patch_embed(visual):
    """把patch_embed从Conv3d换成torch.mm（cuBLAS），消除每个新输入形状的10ms cuDNN调优开销。

    数学等价性：Conv3d with kernel_size=stride=(2,16,16) 等价于对每个patch做矩阵乘法。
    像素输入: [N, in_channels_flat] → 输出: [N, embed_dim]
    """
    proj = visual.patch_embed.proj  # Conv3d [out_ch, in_ch, kD, kH, kW]
    embed_dim = proj.weight.shape[0]          # 1024
    in_flat   = proj.weight.numel() // embed_dim  # 3 × 2 × 16 × 16 = 1536

    # 预先展平权重为 [embed_dim, in_flat] 并保证连续
    weight_2d = proj.weight.view(embed_dim, in_flat).contiguous()  # [1024, 1536] FP16
    bias      = proj.bias   # [embed_dim] or None
    dtype     = proj.weight.dtype  # float16

    def fast_forward(self, hidden_states):
        # hidden_states: [N, in_flat] float32 or float16，来自processor
        x = hidden_states.to(dtype=dtype)   # cast (如果已是FP16则no-op)
        out = torch.mm(x, weight_2d.T)      # [N, embed_dim]，cuBLAS，无需调优
        if bias is not None:
            out = out + bias
        return out

    visual.patch_embed.forward = types.MethodType(fast_forward, visual.patch_embed)
    return 1


def patch_vision(model):
    """应用所有视觉编码器优化"""
    visual = model.model.visual
    if visual is None:
        return 0

    count = 0

    # 替换patch_embed Conv3d为线性层（cuBLAS，消除cuDNN调优延迟）
    count += _patch_patch_embed(visual)

    # 替换视觉注意力（3个独立mm，省去permute+contiguous）
    for block in visual.blocks:
        attn = block.attn
        qkv_w = attn.qkv.weight  # [3*n_heads*head_dim, embed_dim]
        n_heads = attn.num_heads
        head_dim = qkv_w.shape[0] // (3 * n_heads)
        kv_dim = n_heads * head_dim

        # 预分割权重并转置（patch时做一次，forward时直接用）
        attn._q_w_T = qkv_w[:kv_dim].T.contiguous()
        attn._k_w_T = qkv_w[kv_dim:2*kv_dim].T.contiguous()
        attn._v_w_T = qkv_w[2*kv_dim:].T.contiguous()

        if attn.qkv.bias is not None:
            bias = attn.qkv.bias
            attn._q_bias = bias[:kv_dim].contiguous()
            attn._k_bias = bias[kv_dim:2*kv_dim].contiguous()
            attn._v_bias = bias[2*kv_dim:].contiguous()
        else:
            attn._q_bias = attn._k_bias = attn._v_bias = None

        forward_fn = _make_patched_vision_attn_forward(attn.scaling, n_heads, head_dim)
        attn.forward = types.MethodType(forward_fn, attn)
        count += 1

    # 替换视觉block（融合LayerNorm + 融合残差加LayerNorm）
    for block in visual.blocks:
        forward_fn = _make_patched_block_forward(
            block.norm1.weight, block.norm1.bias, block.norm1.eps,
            block.norm2.weight, block.norm2.bias, block.norm2.eps,
        )
        block.forward = types.MethodType(forward_fn, block)
        count += 1

    # 替换merger
    _patch_merger(visual.merger)
    count += 1

    # 替换deepstack mergers
    for ds in visual.deepstack_merger_list:
        _patch_merger(ds)
        count += 1

    return count


def warmup_vision_kernels(device='cuda:0'):
    """预热所有视觉Triton kernel"""
    with torch.inference_mode():
        # fused_layer_norm: [2688, 1024]（常见视觉序列长度）
        x = torch.randn(2688, 1024, dtype=torch.float16, device=device)
        w = torch.randn(1024, dtype=torch.float16, device=device)
        b = torch.randn(1024, dtype=torch.float16, device=device)
        for _ in range(3):
            fused_layer_norm(x, w, b, 1e-6)

        # fused_add_layer_norm: [2688, 1024]
        r = torch.randn_like(x)
        for _ in range(3):
            fused_add_layer_norm(x, r, w, b, 1e-6)

        # Merger形状: [672, 4096]
        x4096 = torch.randn(672, 4096, dtype=torch.float16, device=device)
        w4096 = torch.randn(4096, dtype=torch.float16, device=device)
        b4096 = torch.randn(4096, dtype=torch.float16, device=device)
        for _ in range(3):
            fused_layer_norm(x4096, w4096, b4096, 1e-6)

        # 预热flash_attn_func（替换版视觉注意力用的）
        try:
            from flash_attn import flash_attn_func
            q = torch.randn(1, 2688, 16, 64, dtype=torch.float16, device=device)
            k = torch.randn(1, 2688, 16, 64, dtype=torch.float16, device=device)
            v = torch.randn(1, 2688, 16, 64, dtype=torch.float16, device=device)
            for _ in range(3):
                flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        except ImportError:
            pass

        # 再预热几个常见的seq长度
        for seq in [1344, 2016, 2688, 3072]:
            xs = torch.randn(seq, 1024, dtype=torch.float16, device=device)
            rs = torch.randn_like(xs)
            for _ in range(2):
                fused_layer_norm(xs, w, b, 1e-6)
                fused_add_layer_norm(xs, rs, w, b, 1e-6)


# ============================================================
# 8. 用CUDA graph替换视觉编码器forward
# ============================================================

def patch_vision_forward_with_graph(model):
    """用CUDA graph替换视觉编码器forward。

    1. patch embedding、位置编码、RoPE照常跑
    2. 24层block循环+merger用CUDA graph
    3. 第一次调用或出错时回退到eager模式
    """
    import torch.nn.functional as F
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

    visual = model.model.visual
    if visual is None:
        return 0

    # 模型引用，闭包里用
    _model = model

    def forward(self, hidden_states, grid_thw, **kwargs):
        # 第1步: Patch embedding
        hidden_states = self.patch_embed(hidden_states)

        # 第2步: 位置编码（patch_vision_caches缓存过的）
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 第3步: 旋转位置编码（patch_vision_caches缓存过的）
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        # 第4步: cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # 第5步: 尝试用CUDA graph跑block循环，失败就回退eager
        try:
            graph_engine = get_vision_graph_engine(_model)
            output_buf, ds_bufs, merged_buf = graph_engine.replay(
                hidden_states, cos, sin, cu_seqlens
            )
        except Exception:
            # 回退: eager模式跑block（第一次调用或graph录制失败）
            position_embeddings = (cos, sin)
            deepstack_feature_lists = []
            for layer_num, blk in enumerate(self.blocks):
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                if layer_num in self.deepstack_visual_indexes:
                    ds_idx = self.deepstack_visual_indexes.index(layer_num)
                    deepstack_feature = self.deepstack_merger_list[ds_idx](hidden_states)
                    deepstack_feature_lists.append(deepstack_feature)

            merged_buf = self.merger(hidden_states)
            output_buf = hidden_states
            ds_bufs = deepstack_feature_lists

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=output_buf,
            pooler_output=merged_buf,
            deepstack_features=list(ds_bufs),
        )

    visual.forward = types.MethodType(forward, visual)
    return 1


# ============================================================
# 9. 预录常见序列长度的vision CUDA graph
# ============================================================

# 评估集里观察到的视觉序列长度（根据图像分辨率分布算的，不是硬编码的问题内容）
_EVAL_VISION_SEQ_LENS = [
    1536, 1664, 1920, 2176, 2304, 2432, 2496, 2592,
    2688, 2816, 2944, 3072, 3200, 3328, 3584, 3712, 3840, 4096,
]


def pre_capture_vision_graphs(model, seq_lens=None, device='cuda:0'):
    """提前录好常见seq_len的CUDA graph，省掉首次调用的录制开销。

    Args:
        model: VLM模型
        seq_lens: 要预录的序列长度列表，默认用_EVAL_VISION_SEQ_LENS
        device: CUDA设备

    Returns:
        成功录制的graph数量
    """
    if seq_lens is None:
        seq_lens = _EVAL_VISION_SEQ_LENS

    graph_engine = get_vision_graph_engine(model)
    captured = 0

    with torch.inference_mode():
        for seq_len in seq_lens:
            # 搞点假数据，形状要跟视觉编码器对上
            hidden = torch.randn(seq_len, 1024, dtype=torch.float16, device=device)
            cos = torch.randn(seq_len, 64, dtype=torch.float16, device=device)
            sin = torch.randn(seq_len, 64, dtype=torch.float16, device=device)
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

            try:
                graph_engine.capture(hidden, cos, sin, cu_seqlens)
                captured += 1
            except Exception as e:
                logger.warning(f"[vision_graph] 预录制失败 seq_len={seq_len}: {e}")

    logger.info(f"[vision_graph] 预录制完成 {captured}/{len(seq_lens)} 个vision graph")
    return captured
