import random

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# ImageNet stats — standard for transfer-learning-style training on VOC
_VOC_MEAN = (0.485, 0.456, 0.406)
_VOC_STD = (0.229, 0.224, 0.225)

# Applied to the image channel only during training
_TRAIN_COLOR_JITTER = transforms.ColorJitter(
    brightness=0.4, contrast=0.4, saturation=0.4
)


class _SqueezeToLong:
    """Converts a [1, H, W] uint8 mask tensor to [H, W] int64."""

    def __call__(self, m):
        return m.squeeze(0).long()


class _PairedSegDataset(Dataset):
    """Wraps a VOCSegmentation dataset with geometry-consistent transforms.

    Geometric transforms (scale, crop, flip) are applied identically to the
    image and mask.  Photometric transforms (colour jitter) are image-only.
    The base dataset must be constructed with ``transform=None`` and
    ``target_transform=None`` so that raw PIL images are returned here.
    """

    def __init__(self, base_ds: VOCSegmentation, train: bool):
        self._base = base_ds
        self._train = train

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int):
        img, mask = self._base[idx]  # PIL images

        if self._train:
            # Random scale: resize so the shortest edge is at least 224.
            scale = random.uniform(0.5, 2.0)
            new_h = max(224, round(img.height * scale))
            new_w = max(224, round(img.width * scale))
            img = TF.resize(img, [new_h, new_w], InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [new_h, new_w], InterpolationMode.NEAREST)

            # Random crop to 224×224
            i, j, th, tw = transforms.RandomCrop.get_params(img, (224, 224))
            img = TF.crop(img, i, j, th, tw)
            mask = TF.crop(mask, i, j, th, tw)

            # Random horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Colour jitter — image only
            img = _TRAIN_COLOR_JITTER(img)
        else:
            img = TF.resize(img, [224, 224], InterpolationMode.BILINEAR)
            mask = TF.resize(mask, [224, 224], InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, _VOC_MEAN, _VOC_STD)
        mask = TF.pil_to_tensor(mask).squeeze(0).long()
        return img, mask


def _make_voc(root: str, image_set: str, train: bool) -> _PairedSegDataset:
    """Build a paired segmentation dataset, falling back to 'train' if the
    requested split file is not found (e.g. when trainaug is not set up)."""
    try:
        base = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=False,
            transform=None,
            target_transform=None,
        )
    except (FileNotFoundError, ValueError):
        if image_set == "trainaug":
            import warnings
            warnings.warn(
                "trainaug split not found; falling back to 'train' (1 464 images). "
                "Set up the SBD augmentation for the full 10 582-image training set.",
                stacklevel=3,
            )
            base = VOCSegmentation(
                root=root,
                year="2012",
                image_set="train",
                download=False,
                transform=None,
                target_transform=None,
            )
        else:
            raise
    return _PairedSegDataset(base, train=train)


class VOCSegmentationDataModule(L.LightningDataModule):
    """Lightning DataModule for PASCAL VOC 2012 semantic segmentation.

    Loads images and segmentation masks from the local VOCdevkit directory.
    Classes 0-20 correspond to background + 20 object categories; 255 marks
    ambiguous boundary pixels that should be ignored during training.

    Training augmentation
    ---------------------
    - Random scale (0.5–2.0×) then random 224×224 crop
    - Random horizontal flip
    - Colour jitter (brightness, contrast, saturation ±0.4)

    Validation/test: deterministic 224×224 resize only.

    Data contract
    -------------
    Each batch is a ``(images, masks)`` tuple:
    - ``images``: ``Tensor[B, 3, 224, 224]`` float32, normalised with
      ImageNet mean/std.
    - ``masks``:  ``Tensor[B, 224, 224]`` int64 with values in ``{0..20, 255}``.

    Args:
        data_dir:    Path to the directory that directly contains
                     ``VOCdevkit/`` (torchvision appends that subdirectory
                     internally). Defaults to ``"data/VOC_dataset"``.
        batch_size:  Samples per batch.
        num_workers: DataLoader worker processes.
        train_split: Image-set name for training. Use ``"trainaug"`` (default)
                     for the 10 582-image SBD-augmented set; falls back to
                     ``"train"`` (1 464 images) automatically if not found.
    """

    num_classes: int = 21

    def __init__(
        self,
        data_dir: str = "data/VOC_dataset",
        batch_size: int = 16,
        num_workers: int = 4,
        train_split: str = "trainaug",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_ds = _make_voc(self.data_dir, self.train_split, train=True)
            self.val_ds = _make_voc(self.data_dir, "val", train=False)
        if stage in ("test", None):
            self.test_ds = _make_voc(self.data_dir, "val", train=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
