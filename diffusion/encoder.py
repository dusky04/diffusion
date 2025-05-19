import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


# TODO: Add input channels and output channels
class VariationalAutoEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # bunch of convolutions
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            # (batch_size, channels, height, width) -> (batch_size, channels, height / stride, width / stride)
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=2),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, padding=0, stride=2),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            nn.GroupNorm(512, 512),
            # no particular reason why this is chosen
            nn.SiLU(),  # apparentl
            # howy works well for this application
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:  # type: ignore
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # (PaddingLeft, PaddingRight, PaddingTop, PaddingBottom)
                x = F.pad(x, (0, 1, 0, 1))  # asymmetrical padding?
            x = module(x)

        # (batch_size, 8, height / 8, width / 8) -> two tensors of shape (batch_size, 4, height / 8, width / 8)
        mean, log_variance = torch.chunk(
            x, chunks=2, dim=1
        )  # create two tensors along the third dimension

        log_variance = torch.clamp(
            log_variance, -30, 20
        )  # clamp the values within [-30, 20]

        variance = log_variance.exp()

        # get the standard deviation
        std_dev = variance.sqrt()

        # how do we sample from the N(mean, std_dev)
        # Z = N(0, 1) -> N(mean, std_dev)
        # x = mean + std_dev * Z
        x = mean + std_dev * noise

        # scale the output by this *magic* constant
        x *= 0.18215
