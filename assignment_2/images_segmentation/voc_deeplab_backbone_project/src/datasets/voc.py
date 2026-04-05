from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class VOCSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        image_dir: str,
        mask_dir: str,
        split_dir: str,
        split_file: str,
        image_size: int,
        mean: list[float],
        std: list[float],
        train: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_root = self.root_dir / image_dir
        self.mask_root = self.root_dir / mask_dir
        self.split_file = self.root_dir / split_dir / split_file
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.train = train

        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        with self.split_file.open("r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.ids)

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image, str]:
        sample_id = self.ids[idx]
        image_path = self.image_root / f"{sample_id}.jpg"
        mask_path = self.mask_root / f"{sample_id}.png"
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        return image, mask, sample_id

    def _train_transform(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        scale = random.uniform(0.5, 2.0)
        new_w = max(1, int(image.width * scale))
        new_h = max(1, int(image.height * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        pad_w = max(0, self.image_size - new_w)
        pad_h = max(0, self.image_size - new_h)
        if pad_w > 0 or pad_h > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=255)

        i, j, h, w = RandomCrop.get_params(image, output_size=(self.image_size, self.image_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor

    def _val_transform(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, self.mean, self.std)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor

    def __getitem__(self, idx: int):
        image, mask, sample_id = self._load_pair(idx)

        if self.train:
            image_tensor, mask_tensor = self._train_transform(image, mask)
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "id": sample_id,
            }
        else:
            raw_image = image.copy()
            image_tensor, mask_tensor = self._val_transform(image, mask)
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "id": sample_id,
                "raw_image": raw_image,
            }
