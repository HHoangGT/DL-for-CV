from __future__ import annotations

import torch.nn as nn


def build_loss(ignore_index: int = 255) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=ignore_index)
