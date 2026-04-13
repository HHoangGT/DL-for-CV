from __future__ import annotations

from dataclasses import dataclass
from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, dilation: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPBranch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        kernel_size = 1 if dilation == 1 else 3
        padding = 0 if dilation == 1 else dilation
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, atrous_rates: tuple[int, int, int] = (6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ASPPBranch(in_channels, out_channels, dilation=1),
                ASPPBranch(in_channels, out_channels, dilation=atrous_rates[0]),
                ASPPBranch(in_channels, out_channels, dilation=atrous_rates[1]),
                ASPPBranch(in_channels, out_channels, dilation=atrous_rates[2]),
            ]
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = self.pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        feats = [branch(x) for branch in self.branches] + [pooled]
        return self.project(torch.cat(feats, dim=1))


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, out_channels: int = 256):
        super().__init__()
        self.aspp = ASPP(high_channels, out_channels)
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU(out_channels + 48, out_channels, 3, padding=1),
            ConvBNReLU(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        high = self.aspp(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)
        x = torch.cat([low, high], dim=1)
        return self.fuse(x)


class TimmEncoder(nn.Module):
    def __init__(self, backbone_name: str, pretrained: bool = True, img_size: int = 512):
        super().__init__()
        self.backbone_name = backbone_name

        extra_kwargs = {}
        if "swin" in backbone_name:
            extra_kwargs["img_size"] = img_size

        try:
            self.encoder = timm.create_model(
                backbone_name,
                features_only=True,
                pretrained=pretrained,
                out_indices=(0, 1, 2, 3, 4),
                **extra_kwargs,
            )
        except Exception:
            self.encoder = timm.create_model(
                backbone_name,
                features_only=True,
                pretrained=pretrained,
                out_indices=(0, 1, 2, 3),
                **extra_kwargs,
            )

        self.channels = self.encoder.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.encoder(x)
        fixed_features = []

        for feat, ch in zip(features, self.channels):
            # Nếu feature đang ở dạng NHWC thì đổi sang NCHW
            if feat.ndim == 4 and feat.shape[-1] == ch and feat.shape[1] != ch:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            fixed_features.append(feat)

        return fixed_features


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        low_level_index: int = 0,
        high_level_index: int = -1,
        decoder_channels: int = 256,
        img_size: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = TimmEncoder(backbone_name, pretrained=pretrained, img_size=img_size)
        channels = self.encoder.channels

        self.low_level_index = low_level_index
        self.high_level_index = high_level_index

        # Chọn low/high channel an toàn theo số mức feature mà backbone thực sự trả về
        low_idx = min(low_level_index, len(channels) - 2)
        high_idx = len(channels) - 1 if high_level_index < 0 else min(high_level_index, len(channels) - 1)

        low_channels = channels[low_idx]
        high_channels = channels[high_idx]

        self.decoder = DeepLabV3PlusDecoder(
            low_channels=low_channels,
            high_channels=high_channels,
            out_channels=decoder_channels,
        )
        self.classifier = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x)

        low_idx = min(self.low_level_index, len(features) - 2)
        high_idx = len(features) - 1 if self.high_level_index < 0 else min(self.high_level_index, len(features) - 1)

        low = features[low_idx]
        high = features[high_idx]

        x = self.decoder(low, high)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
