from __future__ import annotations

import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

_val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        val_subdir: str = "val",
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_subdir = val_subdir

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = ImageFolder(
                os.path.join(self.data_dir, "train"),
                transform=_train_transform,
            )
            self.val_ds = ImageFolder(
                os.path.join(self.data_dir, self.val_subdir),
                transform=_val_transform,
            )

        if stage in ("test", None):
            self.test_ds = ImageFolder(
                os.path.join(self.data_dir, self.val_subdir),
                transform=_val_transform,
            )

    @property
    def num_classes(self) -> int:
        return len(self.train_ds.classes)

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
