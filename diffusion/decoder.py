import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        batch_size, num_channels, height, width = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view(batch_size, num_channels, height * width)
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features).
        # Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        x = x.transpose(-1, -2)

        # Perform self-attention WITHOUT mask
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view(batch_size, num_channels, height, width)

        return x + residue


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # TODO: why are we using group_norm here? Because of convolutions, features' relative locatlity
        self.group_norm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_block_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        self.group_norm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_block_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        # since, we have a residual connection at the end of the block
        # if the `in_channes` and `out_channels` do not match
        # we apply the `residual_layer` to handle the dimension mismatch
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_block_1(x)

        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_block_2(x)

        return x + self.residual_layer(residue)
