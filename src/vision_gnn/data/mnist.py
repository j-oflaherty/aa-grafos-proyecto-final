from __future__ import annotations

import torch
import torch.utils.data
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from .augmentation import (
    AugmentationConfig,
    MixupCutmixCollate,
    RepeatAugSampler,
    build_post_tensor_transforms,
    build_pre_tensor_transforms,
    build_spatial_transform,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _make_train_transform(aug: AugmentationConfig | None) -> transforms.Compose:
    pre = build_pre_tensor_transforms(aug) if aug else []
    post = build_post_tensor_transforms(aug) if aug else []
    return transforms.Compose(
        [
            build_spatial_transform(aug, size=224),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            *pre,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            *post,
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
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.classes = sorted(classes) if classes is not None else list(range(10))
        self.num_classes = len(self.classes)
        self.augmentation = augmentation

    def setup(self, stage: str | None = None) -> None:
        train_tf = _make_train_transform(self.augmentation)

        if stage in ("fit", None):
            # Use separate dataset objects so transforms don't bleed across splits.
            train_full = _FilteredMNIST(
                MNIST(self.data_dir, train=True, download=True, transform=train_tf),
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
        aug = self.augmentation
        use_repeat = aug is not None and aug.repeated_augment > 1
        use_mix = aug is not None and (aug.mixup_alpha > 0 or aug.cutmix_alpha > 0)

        sampler = (
            RepeatAugSampler(self.train_ds, num_repeats=aug.repeated_augment)
            if use_repeat
            else None
        )
        collate_fn = (
            MixupCutmixCollate(
                num_classes=self.num_classes,
                mixup_alpha=aug.mixup_alpha,
                cutmix_alpha=aug.cutmix_alpha,
            )
            if use_mix
            else None
        )

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
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
