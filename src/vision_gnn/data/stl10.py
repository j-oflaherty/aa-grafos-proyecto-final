from __future__ import annotations

import torch.utils.data
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import STL10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# STL-10 images are 96x96; resize to 224 for ImageNet-pretrained backbones.
_train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

_val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class STL10DataModule(LightningDataModule):
    """LightningDataModule for STL-10 (10 classes, 96×96 RGB).

    The labeled training set has 5 000 images. A fraction (`val_split`) is
    held out as a validation set; the rest is used for training. The test
    set contains 8 000 images.
    """

    num_classes: int = 10

    def __init__(
        self,
        data_dir: str = "data/stl10",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            train_full = STL10(
                self.data_dir, split="train", download=True, transform=_train_transform
            )
            val_full = STL10(
                self.data_dir, split="train", download=True, transform=_val_transform
            )
            n_val = int(len(train_full) * self.val_split)
            n_train = len(train_full) - n_val
            indices = list(range(len(train_full)))
            self.train_ds = Subset(train_full, indices[:n_train])
            self.val_ds = Subset(val_full, indices[n_train:])

        if stage in ("test", None):
            self.test_ds = STL10(
                self.data_dir, split="test", download=True, transform=_val_transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
