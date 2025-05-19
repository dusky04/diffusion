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


# TODO: Add dimension comments to indicate change in input dimensions -> output dimension
class VariationalAutoDecoder(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x /= 0.18215  # nullify the *magic* constant
        for module in self:
            x = module(x)
        return x
