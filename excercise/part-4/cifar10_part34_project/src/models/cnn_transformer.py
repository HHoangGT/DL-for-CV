import torch
from torch import nn

from src.models.common import TransformerEncoder


class CNNBackbone(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.project = nn.Conv2d(128, embed_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.project(x)
        return x


class CNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = CNNBackbone(embed_dim=embed_dim)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat = self.backbone(dummy)
            num_tokens = feat.shape[2] * feat.shape[3]

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(
            depth=depth,
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.dropout(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x)
