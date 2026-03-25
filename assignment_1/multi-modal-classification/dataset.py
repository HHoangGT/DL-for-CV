"""
Dataset utilities for Flickr30k multimodal classification.

Loads Flickr30k from HuggingFace, assigns pseudo-labels via CLIP similarity,
and provides DataLoaders for zero-shot and few-shot experiments.

NOTE: Images are loaded LAZILY (on-the-fly) to avoid storing all 31K images
in RAM at once, which would exceed 32GB.
"""

import random
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from loguru import logger


from config import (
    CATEGORIES,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_SAMPLES,
)


# ─── PyTorch Dataset (Lazy Loading) ─────────────────────────────────────────


class MultimodalDataset(Dataset):
    """
    Dataset that loads images from the HuggingFace CIFAR-100 dataset.
    Images are loaded on-the-fly in __getitem__.
    """

    def __init__(self, hf_dataset, labels, transform=None, max_samples=None):
        self.hf_dataset = hf_dataset
        self.labels = labels
        self.transform = transform
        self.max_samples = max_samples or len(labels)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        img = sample["img"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        label = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ─── Few-shot sampling ───────────────────────────────────────────────────────


def sample_few_shot(labels, k, seed=SEED):
    """
    Sample k examples per class for few-shot training.
    """
    rng = random.Random(seed)
    class_indices = defaultdict(list)

    for i, label in enumerate(labels):
        class_indices[label].append(i)

    train_indices = []
    test_indices = []

    for cls, indices in class_indices.items():
        rng.shuffle(indices)
        if len(indices) <= k:
            train_indices.extend(indices)
        else:
            train_indices.extend(indices[:k])
            test_indices.extend(indices[k:])

    return train_indices, test_indices


# ─── Data preparation pipeline ───────────────────────────────────────────────


def load_cifar100():
    """Load CIFAR-100 dataset from HuggingFace."""
    from datasets import load_dataset

    logger.info("Loading CIFAR-100 from HuggingFace...")
    ds = load_dataset("cifar100", split="test")
    logger.success(f"Loaded {len(ds)} samples.")
    return ds


def prepare_dataset(max_samples=None):
    """
    Data preparation pipeline for CIFAR-100:
    1. Load CIFAR-100 test split
    2. Extract ground truth labels
    3. Return MultimodalDataset

    Returns:
        dataset: MultimodalDataset
        label_dist: dict mapping category index to count
    """
    global MAX_SAMPLES
    if max_samples is not None:
        original = MAX_SAMPLES
        MAX_SAMPLES = max_samples

    hf_ds = load_cifar100()
    labels = hf_ds["fine_label"]

    if max_samples is not None:
        MAX_SAMPLES = original

    dataset = MultimodalDataset(
        hf_ds,
        labels,
        transform=None,  # Transforms will be applied dynamically if needed
        max_samples=len(labels),
    )

    label_dist = defaultdict(int)
    for l in labels:
        label_dist[l] += 1

    logger.info("Label distribution:")
    for idx in range(min(5, len(CATEGORIES))):
        logger.info(f"  {CATEGORIES[idx]}: {label_dist[idx]}")
    logger.info(f"  ... and {len(CATEGORIES) - 5} more classes.")

    return dataset, dict(label_dist)


def get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    """Create a DataLoader from a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
