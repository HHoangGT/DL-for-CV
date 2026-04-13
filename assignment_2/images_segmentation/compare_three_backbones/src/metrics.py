from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SegmentationMetrics:
    miou: float
    dice: float
    pixel_acc: float


class RunningSegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.view(-1)
        targets = targets.view(-1)
        valid = targets != self.ignore_index
        preds = preds[valid]
        targets = targets[valid]
        if preds.numel() == 0:
            return
        k = (targets >= 0) & (targets < self.num_classes)
        inds = self.num_classes * targets[k].to(torch.int64) + preds[k].to(torch.int64)
        self.confusion += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> SegmentationMetrics:
        hist = self.confusion
        diag = torch.diag(hist)
        union = hist.sum(1) + hist.sum(0) - diag
        iou = diag / union.clamp(min=1)
        miou = iou.mean().item()

        dice = (2 * diag / (hist.sum(1) + hist.sum(0)).clamp(min=1)).mean().item()
        pixel_acc = (diag.sum() / hist.sum().clamp(min=1)).item()
        return SegmentationMetrics(miou=miou, dice=dice, pixel_acc=pixel_acc)
