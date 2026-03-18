from __future__ import annotations

import torch
import torch.utils.data
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_train_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

_val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class _FilteredMNIST(torch.utils.data.Dataset):
    """Wraps an MNIST dataset, keeping only `classes` and remapping labels to 0-based indices."""

    def __init__(self, base: MNIST, classes: list[int]) -> None:
        self.base = base
        self.label_map = {orig: new for new, orig in enumerate(classes)}
        mask = torch.isin(base.targets, torch.tensor(classes))
        self.indices = mask.nonzero(as_tuple=True)[0].tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.base[self.indices[idx]]
        return img, self.label_map[label]


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/mnist",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        classes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.classes = sorted(classes) if classes is not None else list(range(10))
        self.num_classes = len(self.classes)

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            # Use separate dataset objects so transforms don't bleed across splits.
            train_full = _FilteredMNIST(
                MNIST(self.data_dir, train=True, download=True, transform=_train_transform),
                self.classes,
            )
            val_full = _FilteredMNIST(
                MNIST(self.data_dir, train=True, download=True, transform=_val_transform),
                self.classes,
            )
            n_val = int(len(train_full) * self.val_split)
            n_train = len(train_full) - n_val
            indices = list(range(len(train_full)))
            self.train_ds = Subset(train_full, indices[:n_train])
            self.val_ds = Subset(val_full, indices[n_train:])

        if stage in ("test", None):
            self.test_ds = _FilteredMNIST(
                MNIST(self.data_dir, train=False, download=True, transform=_val_transform),
                self.classes,
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
