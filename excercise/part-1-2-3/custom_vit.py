import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CustomMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape

        q = (
            self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        )  # (B, num_heads, N, head_dim)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Q * K^T / sqrt(d_k)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, N)

        # Attn * V
        out = attn_weights @ v  # (B, num_heads, N, head_dim)

        # Concat heads
        out = out.transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.out_proj(out)
        return out


class CustomTransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CustomMultiHeadAttention(embed_dim, num_heads)
        self.dropout_1 = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Attention with residual
        x = x + self.dropout_1(self.attn(self.ln_1(x)))
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding[:, : x.size(1), :]


class ViTCustom(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            "Image dimensions must be divisible by the patch size."
        )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding(embed_dim, max_len=num_patches + 1)

        self.blocks = nn.ModuleList(
            [
                CustomTransformerEncoderBlock(embed_dim, heads, mlp_dim)
                for _ in range(depth)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        cls_output = x[:, 0]

        return self.mlp_head(cls_output)
