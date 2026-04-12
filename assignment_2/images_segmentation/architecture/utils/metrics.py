"""
Evaluation metrics for semantic segmentation:
  - Mean IoU (mIoU)
  - Dice Score
  - Pixel Accuracy
"""

import numpy as np


def compute_miou(pred, target, num_classes=21, ignore_index=255):
    """
    Compute Mean Intersection over Union (mIoU).

    Args:
        pred: Tensor of shape [B, H, W] with predicted class indices.
        target: Tensor of shape [B, H, W] with ground truth class indices.
        num_classes: Number of classes.
        ignore_index: Class index to ignore (border pixels in VOC).

    Returns:
        miou: float, mean IoU averaged over present classes.
        per_class_iou: dict mapping class_id -> IoU.
    """
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    # Mask out ignored pixels
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    per_class_iou = {}
    iou_sum = 0.0
    count = 0

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            # Class not present in this batch
            continue

        iou = intersection / union
        per_class_iou[cls] = iou
        iou_sum += iou
        count += 1

    miou = iou_sum / max(count, 1)
    return miou, per_class_iou


def compute_dice(pred, target, num_classes=21, ignore_index=255):
    """
    Compute Mean Dice Score.

    Args:
        pred: Tensor of shape [B, H, W] with predicted class indices.
        target: Tensor of shape [B, H, W] with ground truth class indices.
        num_classes: Number of classes.
        ignore_index: Class index to ignore.

    Returns:
        mean_dice: float.
        per_class_dice: dict mapping class_id -> Dice.
    """
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    per_class_dice = {}
    dice_sum = 0.0
    count = 0

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = np.logical_and(pred_cls, target_cls).sum()
        total = pred_cls.sum() + target_cls.sum()

        if total == 0:
            continue

        dice = 2.0 * intersection / total
        per_class_dice[cls] = dice
        dice_sum += dice
        count += 1

    mean_dice = dice_sum / max(count, 1)
    return mean_dice, per_class_dice


def compute_pixel_accuracy(pred, target, ignore_index=255):
    """
    Compute overall pixel accuracy.

    Args:
        pred: Tensor of shape [B, H, W] with predicted class indices.
        target: Tensor of shape [B, H, W] with ground truth class indices.
        ignore_index: Class index to ignore.

    Returns:
        accuracy: float.
    """
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    correct = (pred == target).sum()
    total = valid.sum()

    return correct / max(total, 1)
