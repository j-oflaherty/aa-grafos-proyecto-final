from .mnist import MNISTDataModule
from .imagenet import ImageNetDataModule
from .stl10 import STL10DataModule
from .svhn_superpixel import SVHNSuperpixelDataModule
from .mnist_superpixel import MNISTSuperpixelDataModule
from .stl10_superpixel import STL10SuperpixelDataModule

__all__ = [
    "MNISTDataModule",
    "ImageNetDataModule",
    "STL10DataModule",
    "SVHNSuperpixelDataModule",
    "MNISTSuperpixelDataModule",
    "STL10SuperpixelDataModule",
]
