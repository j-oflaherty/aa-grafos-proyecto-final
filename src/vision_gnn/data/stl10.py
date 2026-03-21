from __future__ import annotations

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import STL10

from .augmentation import (
    AugmentationConfig,
    MixupCutmixCollate,
    RepeatAugSampler,
    build_post_tensor_transforms,
    build_pre_tensor_transforms,
    build_spatial_transform,
)

STL10_MEAN = (0.485, 0.456, 0.406)
STL10_STD = (0.2682, 0.2610, 0.2686)


def _make_train_transform(aug: AugmentationConfig | None) -> transforms.Compose:
    pre = build_pre_tensor_transforms(aug) if aug else []
    post = build_post_tensor_transforms(aug) if aug else []
    return transforms.Compose(
        [
            build_spatial_transform(aug, size=224),
            transforms.RandomHorizontalFlip(),
            *pre,
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
            *post,
        ]
    )


_val_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD),
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
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.augmentation = augmentation

    def setup(self, stage: str | None = None) -> None:
        train_tf = _make_train_transform(self.augmentation)

        if stage in ("fit", None):
            train_full = STL10(
                self.data_dir, split="train", download=True, transform=train_tf
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
