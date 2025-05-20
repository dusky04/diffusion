import math
import torch
from torch import nn
from torch.nn import functional as F


# TODO: Learn about Attention
class SelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_embeddings: int,
        in_projection_bias: bool = True,
        out_projection_bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_projection = nn.Linear(
            d_embeddings, 3 * d_embeddings, bias=in_projection_bias
        )
        self.out_projection = nn.Linear(
            d_embeddings, d_embeddings, bias=out_projection_bias
        )
        self.num_heads = num_heads
        self.d_heads = d_embeddings // num_heads

    def forward(self, x: torch.Tensor, casual_mask: bool = False) -> torch.Tensor:
        # x.shape -> (batch_size, sequence_length, dims)

        input_shape = x.shape
        batch_size, sequence_length, d_embeddings = input_shape

        interim_shape = (batch_size, sequence_length, self.num_heads, self.d_heads)

        q, k, v = self.in_projection(x).chunk(3, dim=-1)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_projection(output)

        return output
