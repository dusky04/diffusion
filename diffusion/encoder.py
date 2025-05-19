import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super.__init__(nn.Conv2d(3, 128, kernel_size=3, padding=1))

        VAE_ResidualBlock(128, 128)  # bunch of convolutions
        VAE_ResidualBlock(128, 128)

        # (batch_size, channels, height, width) -> (batch_size, channels, height / stride, width / stride)
        nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=2)

        VAE_ResidualBlock(128, 256)
        VAE_ResidualBlock(256, 256)

        nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2)

        VAE_ResidualBlock(256, 512)
        VAE_ResidualBlock(512, 512)

        nn.Conv2d(512, 512, kernel_size=3, padding=0, stride=2)

        VAE_ResidualBlock(512, 512)
        VAE_ResidualBlock(512, 512)
        VAE_ResidualBlock(512, 512)

        VAE_AttentionBlock(512)

        nn.GroupNorm(512, 512)

        nn.SiLU()
