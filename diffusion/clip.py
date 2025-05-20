import torch
from torch import nn
from vae.attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, num_tokens: int) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Parameter(torch.zeros(num_tokens, embedding_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(x) + self.position_embedding


class CLIPLayer(nn.Module):
    def __init__(self, num_heads: int, embedding_size: int) -> None:
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.attention = SelfAttention(num_heads, embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)
        self.linear_1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.linear_2 = nn.Linear(4 * embedding_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        # attention
        x = self.layer_norm_1(x)
        x = self.attention(x, casual_mask=True)
        x += residue

        # feed forward layer
        x = self.layer_norm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(
            1.702 * x
        )  # quickGeLU activation func - works better here
        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = CLIPEmbedding(
            49408, 768, 77
        )  # (vocabulary_size, embedding_size, sequence_len)

        self.layers = nn.ModuleList(
            [
                CLIPLayer(12, 768)  # num_heads
                for _ in range(12)
            ]
        )

        self.layer_norm = nn.LayerNorm(768)

    def forward(self, tokens: torch.Tensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layer_norm(state)
        return output
