"""
[SHLEE] Replace Conv3d patch embedding with nn.Linear.

Qwen3-VL's PatchEmbed receives 2D input [N, C*T*H*W] and reshapes to 5D
before Conv3d with kernel=(2,16,16) stride=(2,16,16).
Since stride==kernel_size, this is algebraically a reshape + linear.
Replacing Conv3d avoids cuDNN setup overhead.
"""
import torch
import torch.nn as nn

class LinearVisionPatchEmbed(nn.Module):
    def __init__(self, conv: nn.Conv3d, in_channels: int, temporal_patch_size: int,
                 patch_size: int, embed_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        in_features = in_channels * temporal_patch_size * patch_size * patch_size
        out_features = embed_dim

        w = conv.weight.data.reshape(out_features, in_features)
        b = conv.bias.data if conv.bias is not None else None

        self.linear = nn.Linear(in_features, out_features, bias=b is not None,
                                device=w.device, dtype=w.dtype)
        self.linear.weight.data.copy_(w)
        if b is not None:
            self.linear.bias.data.copy_(b)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.linear.weight.dtype
        return self.linear(hidden_states.to(dtype=target_dtype))
