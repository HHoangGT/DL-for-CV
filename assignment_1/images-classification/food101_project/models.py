from __future__ import annotations

import torch.nn as nn
from torchvision import models


_SUPPORTED = {'resnet50', 'efficientnet_b0', 'vit_b_16'}


def build_model(model_name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    model_name = model_name.lower()

    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f'Model không hỗ trợ: {model_name}. Hãy dùng một trong: {sorted(_SUPPORTED)}')

    return model


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
