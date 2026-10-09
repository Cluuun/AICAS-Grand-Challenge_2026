"""
[SHLEE] Fused QKV projection: replaces three separate Linear (q_proj, k_proj, v_proj)
with a single matmul, then splits and applies per-head norms.

For Qwen3-VL-2B:
  hidden_size     = 2048
  num_heads       = 16  (Q heads)
  num_kv_heads    = 8   (K/V heads, GQA)
  head_dim        = 128
  q_dim = 16*128 = 2048, kv_dim = 8*128 = 1024
  fused weight: [hidden_size, q_dim + kv_dim + kv_dim] = [2048, 4096]
"""

import torch
import torch.nn as nn

class FusedQKVProjection(nn.Module):
    """Single matmul QKV followed by split + per-head RMSNorm on Q and K."""

    def __init__(self, attn: nn.Module):
        super().__init__()

        self.num_heads = attn.config.num_attention_heads
        self.num_kv_heads = attn.config.num_key_value_heads
        self.head_dim = attn.head_dim
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        w_q = attn.q_proj.weight.data
        w_k = attn.k_proj.weight.data
        w_v = attn.v_proj.weight.data

        fused_w = torch.cat([w_q, w_k, w_v], dim=0)
        self.qkv_proj = nn.Linear(
            fused_w.shape[1], fused_w.shape[0], bias=False,
        )
        self.qkv_proj.weight = nn.Parameter(fused_w)

        self.q_norm = attn.q_norm
        self.k_norm = attn.k_norm

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            query: [batch, num_heads, seq_len, head_dim]  (normed + ready for RoPE)
            key:   [batch, num_kv_heads, seq_len, head_dim]  (normed)
            value: [batch, num_kv_heads, seq_len, head_dim]
        """
        bsz_seq = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)

        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)

        q = self.q_norm(q.view(*bsz_seq, self.num_heads, self.head_dim)).transpose(-3, -2)
        k = self.k_norm(k.view(*bsz_seq, self.num_kv_heads, self.head_dim)).transpose(-3, -2)
        v = v.view(*bsz_seq, self.num_kv_heads, self.head_dim).transpose(-3, -2)

        return q, k, v
