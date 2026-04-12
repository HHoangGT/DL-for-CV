"""
PASCAL VOC 2012 Semantic Segmentation Dataset wrapper.
Uses torchvision.datasets.VOCSegmentation under the hood.
"""

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation

from .augmentations import (
    get_train_transforms,
    get_val_transforms,
    CopyPasteAugmentation,
)

# PASCAL VOC class names (21 classes: 0=background + 20 objects)
VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
NUM_CLASSES = 21
IGNORE_INDEX = 255

# VOC color palette for visualization
VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class VOCSegDataset(Dataset):
    """
    Wrapper around torchvision VOCSegmentation that applies
    joint image-mask transforms and supports Copy-Paste augmentation.
    """

    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transforms=None,
        copy_paste: bool = False,
        copy_paste_prob: float = 0.5,
    ):
        """
        Args:
            root: Root directory for VOC data (will contain VOCdevkit/).
            year: Dataset year ('2007' or '2012').
            image_set: 'train', 'val', or 'trainval'.
            download: Whether to download the dataset.
            transforms: Joint (image, mask) transform callable.
            copy_paste: Whether to enable Copy-Paste augmentation.
            copy_paste_prob: Probability of applying Copy-Paste per sample.
        """
        self.voc = VOCSegmentation(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
        )
        self.transforms = transforms
        self.copy_paste_aug = None

        if copy_paste:
            # Deferred: we pass `self` so copy-paste can sample from this dataset
            self.copy_paste_aug = CopyPasteAugmentation(
                dataset=self,
                paste_prob=copy_paste_prob,
            )

    def __len__(self):
        return len(self.voc)

    def get_raw(self, index):
        """Return raw PIL image and mask (no transforms). Used by Copy-Paste."""
        image, mask = self.voc[index]
        return image, mask

    def __getitem__(self, index):
        image, mask = self.voc[index]

        # Apply Copy-Paste augmentation first (on PIL images)
        if self.copy_paste_aug is not None:
            image, mask = self.copy_paste_aug(image, mask)

        # Apply standard transforms
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask


def get_dataloaders(
    root: str,
    batch_size: int = 8,
    crop_size: tuple = (512, 512),
    num_workers: int = 4,
    download: bool = False,
    copy_paste: bool = False,
):
    """
    Create train and val DataLoaders for PASCAL VOC 2012 segmentation.

    Args:
        root: Path to store/load VOC data.
        batch_size: Batch size for DataLoader.
        crop_size: (H, W) crop size for training.
        num_workers: Number of data loading workers.
        download: Whether to download the dataset.
        copy_paste: Whether to apply Copy-Paste augmentation on training set.

    Returns:
        train_loader, val_loader
    """
    train_transforms = get_train_transforms(crop_size=crop_size)
    val_transforms = get_val_transforms(size=crop_size)

    train_dataset = VOCSegDataset(
        root=root,
        year="2012",
        image_set="train",
        download=download,
        transforms=train_transforms,
        copy_paste=copy_paste,
    )

    val_dataset = VOCSegDataset(
        root=root,
        year="2012",
        image_set="val",
        download=download,
        transforms=val_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
