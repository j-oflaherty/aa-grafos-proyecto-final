"""Training-time data augmentation utilities.

Supports:
- RandomResizedCrop  (replaces plain Resize in the spatial step)
- ColorJitter        (brightness / contrast / saturation)
- RandAugment        (transforms.RandAugment)
- Mixup              (batch-level, via MixupCutmixCollate)
- CutMix             (batch-level, via MixupCutmixCollate)
- Random Erasing     (transforms.RandomErasing, post-tensor)
- Repeated Augment   (RepeatAugSampler)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data
from torchvision import transforms


@dataclass
class AugmentationConfig:
    """All augmentation knobs, suitable for YAML / Lightning CLI."""

    # --- Spatial crop (replaces plain Resize when enabled) ---
    random_resized_crop: bool = False

    # --- ColorJitter ---
    color_jitter: bool = False
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1

    # --- RandAugment ---
    rand_augment: bool = False
    rand_augment_num_ops: int = 2
    rand_augment_magnitude: int = 9

    # --- Mixup (0 = disabled) ---
    mixup_alpha: float = 0.0

    # --- CutMix (0 = disabled) ---
    cutmix_alpha: float = 0.0

    # --- Random Erasing (0 = disabled) ---
    random_erasing_prob: float = 0.0

    # --- Repeated Augment (1 = disabled, >1 = repeats per epoch) ---
    repeated_augment: int = 1


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------


def build_spatial_transform(cfg: AugmentationConfig | None, size: int = 224):
    """Return the primary spatial transform: RandomResizedCrop or plain Resize."""
    if cfg is not None and cfg.random_resized_crop:
        return transforms.RandomResizedCrop(size)
    return transforms.Resize(size)


def build_pre_tensor_transforms(cfg: AugmentationConfig) -> list:
    """Extra transforms to insert *before* ToTensor (operate on PIL images)."""
    extra: list = []
    if cfg.color_jitter:
        extra.append(
            transforms.ColorJitter(
                brightness=cfg.color_jitter_brightness,
                contrast=cfg.color_jitter_contrast,
                saturation=cfg.color_jitter_saturation,
                hue=cfg.color_jitter_hue,
            )
        )
    if cfg.rand_augment:
        extra.append(
            transforms.RandAugment(
                num_ops=cfg.rand_augment_num_ops,
                magnitude=cfg.rand_augment_magnitude,
            )
        )
    return extra


def build_post_tensor_transforms(cfg: AugmentationConfig) -> list:
    """Extra transforms to insert *after* Normalize (operate on float tensors)."""
    extra: list = []
    if cfg.random_erasing_prob > 0:
        extra.append(
            transforms.RandomErasing(
                p=cfg.random_erasing_prob,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
            )
        )
    return extra


# ---------------------------------------------------------------------------
# Mixup / CutMix collate
# ---------------------------------------------------------------------------


def _rand_bbox(h: int, w: int, lam: float) -> tuple[int, int, int, int]:
    cut_rat = math.sqrt(1.0 - lam)
    cut_h = int(h * cut_rat)
    cut_w = int(w * cut_rat)
    cy = random.randint(0, h)
    cx = random.randint(0, w)
    y1 = max(cy - cut_h // 2, 0)
    x1 = max(cx - cut_w // 2, 0)
    y2 = min(cy + cut_h // 2, h)
    x2 = min(cx + cut_w // 2, w)
    return y1, x1, y2, x2


class MixupCutmixCollate:
    """Collate function that applies Mixup and/or CutMix at batch level.

    Returns ``(images, soft_labels)`` where ``soft_labels`` is a
    ``float32`` tensor of shape ``[B, num_classes]``.  If neither
    ``mixup_alpha`` nor ``cutmix_alpha`` is positive the original integer
    labels are returned unchanged so the lightning module can detect the
    difference via ``labels.is_floating_point()``.

    When both are enabled, the method is chosen uniformly at random per batch.
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
    ) -> None:
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        use_mixup = self.mixup_alpha > 0
        use_cutmix = self.cutmix_alpha > 0

        if not use_mixup and not use_cutmix:
            return images, labels

        # One-hot encode
        soft = torch.zeros(len(labels), self.num_classes)
        soft.scatter_(1, labels.unsqueeze(1), 1.0)

        if use_mixup and use_cutmix:
            do_cutmix = random.random() < 0.5
        else:
            do_cutmix = use_cutmix

        if do_cutmix:
            images, soft = self._cutmix(images, soft)
        else:
            images, soft = self._mixup(images, soft)

        return images, soft

    def _mixup(self, images, soft):
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        perm = torch.randperm(images.size(0))
        images = lam * images + (1 - lam) * images[perm]
        soft = lam * soft + (1 - lam) * soft[perm]
        return images, soft

    def _cutmix(self, images, soft):
        lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        perm = torch.randperm(images.size(0))
        h, w = images.size(2), images.size(3)
        y1, x1, y2, x2 = _rand_bbox(h, w, lam)
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
        soft = lam * soft + (1 - lam) * soft[perm]
        return images, soft


# ---------------------------------------------------------------------------
# Repeated Augmentation sampler
# ---------------------------------------------------------------------------


class RepeatAugSampler(torch.utils.data.Sampler):
    """Yields each dataset index ``num_repeats`` times per epoch.

    Each repeat sees a *different* random augmentation because the transform
    pipeline is applied lazily at load time.  The indices are shuffled
    globally so the same image does not appear in consecutive batches.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_repeats: int = 3,
    ) -> None:
        if num_repeats < 1:
            raise ValueError("num_repeats must be >= 1")
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self) -> int:
        return len(self.dataset) * self.num_repeats

    def __iter__(self):
        indices = list(range(len(self.dataset))) * self.num_repeats
        random.shuffle(indices)
        return iter(indices)
