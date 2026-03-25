from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DATA_ROOT, DATASET_DIRNAME, IMG_SIZE, SEED, VAL_RATIO

PathLike = Union[str, Path]


@dataclass
class Food101Paths:
    root: Path
    images_dir: Path
    meta_dir: Path
    train_txt: Path
    test_txt: Path
    classes_txt: Path


def resolve_food101_paths(data_root: PathLike = DATA_ROOT) -> Food101Paths:
    data_root = Path(data_root)
    candidates = [
        data_root / DATASET_DIRNAME,
        data_root,
        data_root / 'food101',
        data_root / 'Food-101',
    ]
    for base in candidates:
        images_dir = base / 'images'
        meta_dir = base / 'meta'
        train_txt = meta_dir / 'train.txt'
        test_txt = meta_dir / 'test.txt'
        classes_txt = meta_dir / 'classes.txt'
        if images_dir.exists() and train_txt.exists() and test_txt.exists() and classes_txt.exists():
            return Food101Paths(base, images_dir, meta_dir, train_txt, test_txt, classes_txt)
    raise FileNotFoundError(
        'Không tìm thấy Food-101. Hãy giải nén dataset vào food101_project/data/food-101/ sao cho bên trong có images/ và meta/.'
    )


def load_classes(paths: Food101Paths) -> List[str]:
    with open(paths.classes_txt, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def _load_split(txt_path: Path, images_dir: Path, class_to_idx: Dict[str, int]) -> pd.DataFrame:
    rows = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for raw in f:
            rel = raw.strip()
            if not rel:
                continue
            label_name = rel.split('/')[0]
            rows.append(
                {
                    'relative_path': rel + '.jpg',
                    'img_path': str(images_dir / (rel + '.jpg')),
                    'label_name': label_name,
                    'label_idx': class_to_idx[label_name],
                }
            )
    return pd.DataFrame(rows)


def build_splits(data_root: PathLike = DATA_ROOT, val_ratio: float = VAL_RATIO, seed: int = SEED):
    paths = resolve_food101_paths(data_root)
    classes = load_classes(paths)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    train_df = _load_split(paths.train_txt, paths.images_dir, class_to_idx)
    test_df = _load_split(paths.test_txt, paths.images_dir, class_to_idx)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_df['label_idx'],
    )
    return classes, class_to_idx, train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_train_transform(img_size: int = IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_plain_train_transform(img_size: int = IMG_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform(img_size: int = IMG_SIZE):
    return get_plain_train_transform(img_size)


class Food101CustomDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, return_path: bool = False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row['img_path']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, int(row['label_idx']), row['img_path']
        return image, int(row['label_idx'])


def create_dataloaders(
    batch_size: int,
    num_workers: int,
    img_size: int = IMG_SIZE,
    data_root: PathLike = DATA_ROOT,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
    use_augmentation: bool = True,
):
    classes, class_to_idx, train_df, val_df, test_df = build_splits(data_root=data_root, val_ratio=val_ratio, seed=seed)
    train_transform = get_train_transform(img_size) if use_augmentation else get_plain_train_transform(img_size)
    train_ds = Food101CustomDataset(train_df, transform=train_transform)
    val_ds = Food101CustomDataset(val_df, transform=get_eval_transform(img_size))
    test_ds = Food101CustomDataset(test_df, transform=get_eval_transform(img_size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    metadata = {
        'num_classes': len(classes),
        'classes': classes,
        'class_to_idx': class_to_idx,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'use_augmentation': use_augmentation,
    }
    return train_loader, val_loader, test_loader, metadata
