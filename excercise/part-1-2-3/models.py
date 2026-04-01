import torch
import torch.nn as nn


class SoftmaxRegression(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, num_classes)
        # Note: We don't apply softmax here because nn.CrossEntropyLoss inherently applies log-softmax

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, hidden_sizes=[512, 256], num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()

        layers = []
        in_features = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))

    def forward(self, x):
        # x shape: (seq_len, batch_size, embed_dim) if batch_first=False
        # x shape: (batch_size, seq_len, embed_dim) if batch_first=True
        return x + self.pos_embedding[:, : x.size(1), :]


class ViTPyTorch(nn.Module):
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding(embed_dim, max_len=num_patches + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.shape[0]

        # Patch embedding: (B, embed_dim, H/P, W/P)
        x = self.patch_embed(x)

        # Flatten patches: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)

        # Add positional embedding
        x = self.pos_embed(x)

        # Transformer
        x = self.transformer(x)

        # Use cls token output for classification
        cls_output = x[:, 0]

        return self.mlp_head(cls_output)
