"""
Augmentation module for PASCAL VOC Segmentation.
Includes basic transforms and advanced Copy-Paste augmentation.
"""

import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF


# ──────────────────────────────────────────────
# 1. Joint transforms (applied to both image & mask synchronously)
# ──────────────────────────────────────────────


class Compose:
    """Apply a list of transforms to both image and mask."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class RandomCrop:
    def __init__(self, size, fill=0, ignore_index=255):
        self.size = size  # (h, w)
        self.fill = fill
        self.ignore_index = ignore_index

    def __call__(self, image, mask):
        w, h = image.size
        th, tw = self.size

        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)

        if pad_h > 0 or pad_w > 0:
            padding = [0, 0, pad_w, pad_h]  # left, top, right, bottom
            image = TF.pad(image, padding, fill=self.fill)
            mask = TF.pad(mask, padding, fill=self.ignore_index)

        i, j, hc, wc = T.RandomCrop.get_params(image, output_size=self.size)
        image = TF.crop(image, i, j, hc, wc)
        mask = TF.crop(mask, i, j, hc, wc)
        return image, mask


class RandomScale:
    def __init__(self, scale_range=(0.5, 2.0)):
        self.scale_range = scale_range

    def __call__(self, image, mask):
        scale = random.uniform(*self.scale_range)
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        image = TF.resize(
            image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.resize(
            mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST
        )
        return image, mask


class ColorJitter:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, image, mask):
        image = self.jitter(image)
        return image, mask


class ToTensorNormalize:
    """Convert PIL image & mask to tensors and normalize the image."""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = TF.to_tensor(image)  # [C, H, W], float32 in [0, 1]
        image = TF.normalize(image, mean=self.mean, std=self.std)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask


# ──────────────────────────────────────────────
# 2. Copy-Paste Augmentation (Extension)
# ──────────────────────────────────────────────


class CopyPasteAugmentation:
    """
    Copy-Paste augmentation for semantic segmentation (Google Brain, 2021).

    Given a source image-mask pair, extract objects of certain classes
    and paste them onto a target image-mask pair at a random position.
    """

    def __init__(self, dataset, paste_prob=0.5, max_paste_objects=3):
        """
        Args:
            dataset: The VOCSegDataset instance to sample source objects from.
            paste_prob: Probability of applying copy-paste on each sample.
            max_paste_objects: Maximum number of objects to paste.
        """
        self.dataset = dataset
        self.paste_prob = paste_prob
        self.max_paste_objects = max_paste_objects

    def __call__(self, image, mask):
        """
        Args:
            image: PIL Image (target).
            mask:  PIL Image (target mask).
        Returns:
            image: PIL Image with pasted objects.
            mask:  PIL Image with updated mask.
        """
        if random.random() > self.paste_prob:
            return image, mask

        # Sample a random source image from the dataset
        src_idx = random.randint(0, len(self.dataset) - 1)
        src_image, src_mask = self.dataset.get_raw(src_idx)

        src_mask_np = np.array(src_mask)
        tgt_image_np = np.array(image)
        tgt_mask_np = np.array(mask)

        # Find unique classes present in source (exclude background 0 and border 255)
        unique_classes = np.unique(src_mask_np)
        valid_classes = [c for c in unique_classes if c not in (0, 255)]

        if len(valid_classes) == 0:
            return image, mask

        # Randomly select a subset of classes to paste
        n_paste = min(self.max_paste_objects, len(valid_classes))
        selected_classes = random.sample(valid_classes, n_paste)

        src_image_np = np.array(src_image)

        for cls_id in selected_classes:
            # Create binary mask for this class
            obj_mask = src_mask_np == cls_id

            if obj_mask.sum() < 100:  # Skip very small objects
                continue

            # Find bounding box of the object
            rows = np.any(obj_mask, axis=1)
            cols = np.any(obj_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Crop the object
            obj_crop = src_image_np[rmin : rmax + 1, cmin : cmax + 1].copy()
            mask_crop = obj_mask[rmin : rmax + 1, cmin : cmax + 1]

            oh, ow = obj_crop.shape[:2]
            th, tw = tgt_image_np.shape[:2]

            if oh >= th or ow >= tw:
                continue

            # Random position in target
            paste_y = random.randint(0, th - oh)
            paste_x = random.randint(0, tw - ow)

            # Paste: overwrite target pixels where object mask is True
            region = tgt_image_np[paste_y : paste_y + oh, paste_x : paste_x + ow]
            region[mask_crop] = obj_crop[mask_crop]
            tgt_image_np[paste_y : paste_y + oh, paste_x : paste_x + ow] = region

            # Update target mask
            mask_region = tgt_mask_np[paste_y : paste_y + oh, paste_x : paste_x + ow]
            mask_region[mask_crop] = cls_id
            tgt_mask_np[paste_y : paste_y + oh, paste_x : paste_x + ow] = mask_region

        image = Image.fromarray(tgt_image_np)
        mask = Image.fromarray(tgt_mask_np)

        return image, mask


# ──────────────────────────────────────────────
# 3. Convenience factory functions
# ──────────────────────────────────────────────


def get_train_transforms(crop_size=(512, 512)):
    """Standard training transforms (without Copy-Paste, which is handled separately)."""
    return Compose(
        [
            RandomScale(scale_range=(0.5, 2.0)),
            RandomCrop(crop_size),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            ToTensorNormalize(),
        ]
    )


def get_val_transforms(size=(512, 512)):
    """Validation / test transforms."""
    return Compose(
        [
            Resize(size),
            ToTensorNormalize(),
        ]
    )
