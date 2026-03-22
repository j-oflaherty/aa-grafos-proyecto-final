from .augmentation import AugmentationConfig
from .mnist import MNISTDataModule
from .imagenet import ImageNetDataModule
from .stl10 import STL10DataModule
from .svhn_superpixel import SVHNSuperpixelDataModule
from .mnist_superpixel import MNISTSuperpixelDataModule
from .stl10_superpixel import STL10SuperpixelDataModule
from .stl10_grid import STL10GridDataModule
from .mnist_grid import MNISTGridDataModule
from .voc import VOCSegmentationDataModule

__all__ = [
    "AugmentationConfig",
    "MNISTDataModule",
    "ImageNetDataModule",
    "STL10DataModule",
    "SVHNSuperpixelDataModule",
    "MNISTSuperpixelDataModule",
    "STL10SuperpixelDataModule",
    "STL10GridDataModule",
    "MNISTGridDataModule",
    "VOCSegmentationDataModule",
]
