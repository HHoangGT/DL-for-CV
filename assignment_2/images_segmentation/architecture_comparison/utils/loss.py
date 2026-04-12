"""
Loss functions for semantic segmentation.
"""

import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation."""

    def __init__(self, num_classes=21, ignore_index=255, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] logits.
            target: [B, H, W] class indices.
        """
        # Create valid mask
        valid_mask = target != self.ignore_index  # [B, H, W]

        # Clamp target to valid range for one-hot encoding
        target_clamped = target.clone()
        target_clamped[~valid_mask] = 0

        # One-hot encode target: [B, H, W] -> [B, C, H, W]
        target_onehot = F.one_hot(target_clamped, self.num_classes)  # [B, H, W, C]
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Softmax on predictions
        pred_soft = F.softmax(pred, dim=1)  # [B, C, H, W]

        # Apply valid mask
        valid_mask = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
        pred_soft = pred_soft * valid_mask
        target_onehot = target_onehot * valid_mask

        # Compute dice per class
        dims = (0, 2, 3)  # sum over batch, h, w
        intersection = (pred_soft * target_onehot).sum(dim=dims)
        cardinality = (pred_soft + target_onehot).sum(dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class CombinedLoss(nn.Module):
    """Combination of CrossEntropy and Dice Loss."""

    def __init__(
        self, num_classes=21, ignore_index=255, ce_weight=0.5, dice_weight=0.5
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice(
            pred, target
        )


def get_loss_fn(loss_type="ce", num_classes=21, ignore_index=255):
    """
    Factory function for loss.

    Args:
        loss_type: 'ce' for CrossEntropy, 'dice' for Dice, 'combined' for both.
        num_classes: Number of classes.
        ignore_index: Index to ignore in loss computation.

    Returns:
        nn.Module loss function.
    """
    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    elif loss_type == "combined":
        return CombinedLoss(num_classes=num_classes, ignore_index=ignore_index)
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Use 'ce', 'dice', or 'combined'."
        )
