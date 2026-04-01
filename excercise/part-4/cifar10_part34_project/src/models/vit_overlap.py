import torch
from torch import nn

from src.models.common import TransformerEncoder


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128, kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTOverlap(nn.Module):
    def __init__(self, num_classes: int = 10, image_size: int = 32, embed_dim: int = 128,
                 depth: int = 4, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.1,
                 kernel_size: int = 4, stride: int = 2, padding: int = 1):
        super().__init__()
        self.tokenizer = OverlapPatchEmbedding(embed_dim=embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            num_tokens = self.tokenizer(dummy).size(1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(depth=depth, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.dropout(x)
        x = self.encoder(x)
        return self.head(x[:, 0])
