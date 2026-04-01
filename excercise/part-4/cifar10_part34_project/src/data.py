from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.utils import CIFAR10_MEAN, CIFAR10_STD


class KaggleCIFAR10TrainDataset(Dataset):
    def __init__(self, image_dir: str, labels_csv: str, transform=None):
        self.image_dir = Path(image_dir)
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform

        # Kaggle CIFAR-10 labels are strings, so map them to integer ids
        classes = sorted(self.labels_df["label"].unique().tolist())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        self.samples = []
        for _, row in self.labels_df.iterrows():
            image_id = str(row["id"])
            label_name = row["label"]
            image_path = self.image_dir / f"{image_id}.png"
            label_idx = self.class_to_idx[label_name]
            self.samples.append((image_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class KaggleCIFAR10TestDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Sort numerically: 1.png, 2.png, 3.png, ...
        self.image_paths = sorted(
            self.image_dir.glob("*.png"),
            key=lambda p: int(p.stem)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Return image and filename/id for later prediction export if needed
        image_id = int(image_path.stem)
        return image, image_id


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, test_transform


def create_dataloaders(
    batch_size: int,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, test_transform = build_transforms()

    train_dir = "./data/train"
    test_dir = "./data/test"
    labels_csv = "./data/trainLabels.csv"

    full_train = KaggleCIFAR10TrainDataset(
        image_dir=train_dir,
        labels_csv=labels_csv,
        transform=train_transform,
    )

    val_base = KaggleCIFAR10TrainDataset(
        image_dir=train_dir,
        labels_csv=labels_csv,
        transform=test_transform,
    )

    test_set = KaggleCIFAR10TestDataset(
        image_dir=test_dir,
        transform=test_transform,
    )

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset_idx, val_subset_idx = random_split(
        range(len(full_train)),
        [train_size, val_size],
        generator=generator,
    )

    train_indices = list(train_subset_idx)
    val_indices = list(val_subset_idx)

    train_subset = torch.utils.data.Subset(full_train, train_indices)
    val_subset = torch.utils.data.Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader