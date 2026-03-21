from __future__ import annotations

import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .augmentation import (
    AugmentationConfig,
    MixupCutmixCollate,
    RepeatAugSampler,
    build_post_tensor_transforms,
    build_pre_tensor_transforms,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _make_train_transform(aug: AugmentationConfig | None) -> transforms.Compose:
    pre = build_pre_tensor_transforms(aug) if aug else []
    post = build_post_tensor_transforms(aug) if aug else []
    # ImageNet always uses RandomResizedCrop — it is essential for this dataset.
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            *pre,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            *post,
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
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_subdir = val_subdir
        self.augmentation = augmentation

    def setup(self, stage: str | None = None) -> None:
        train_tf = _make_train_transform(self.augmentation)

        if stage in ("fit", None):
            self.train_ds = ImageFolder(
                os.path.join(self.data_dir, "train"),
                transform=train_tf,
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
