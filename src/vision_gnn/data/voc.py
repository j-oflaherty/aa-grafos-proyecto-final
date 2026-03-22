import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode

# ImageNet stats — standard for transfer-learning-style training on VOC
_VOC_MEAN = (0.485, 0.456, 0.406)
_VOC_STD = (0.229, 0.224, 0.225)


class _SqueezeToLong:
    """Converts a [1, H, W] uint8 mask tensor to [H, W] int64."""

    def __call__(self, m):
        return m.squeeze(0).long()


_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_VOC_MEAN, _VOC_STD),
])

# Nearest-neighbour resize preserves integer class labels (0-20) and the
# ignore boundary value (255).
_MASK_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    transforms.PILToTensor(),  # [1, H, W] uint8
    _SqueezeToLong(),          # [H, W] int64
])


class VOCSegmentationDataModule(L.LightningDataModule):
    """Lightning DataModule for PASCAL VOC 2012 semantic segmentation.

    Loads images and segmentation masks from the local VOCdevkit directory.
    Classes 0-20 correspond to background + 20 object categories; 255 marks
    ambiguous boundary pixels that should be ignored during training.

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
    """

    num_classes: int = 21

    def __init__(
        self,
        data_dir: str = "data/VOC_dataset",
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_ds = VOCSegmentation(
                root=self.data_dir,
                year="2012",
                image_set="train",
                download=False,
                transform=_IMG_TRANSFORM,
                target_transform=_MASK_TRANSFORM,
            )
            self.val_ds = VOCSegmentation(
                root=self.data_dir,
                year="2012",
                image_set="val",
                download=False,
                transform=_IMG_TRANSFORM,
                target_transform=_MASK_TRANSFORM,
            )
        if stage in ("test", None):
            self.test_ds = VOCSegmentation(
                root=self.data_dir,
                year="2012",
                image_set="val",
                download=False,
                transform=_IMG_TRANSFORM,
                target_transform=_MASK_TRANSFORM,
            )

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
