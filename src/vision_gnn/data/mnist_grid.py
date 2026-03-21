from __future__ import annotations

import os

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

from vision_gnn.models.superpixel_gat.graph_utils import (
    GridGraphDataset,
    superpixel_collate,
)


class MNISTGridDataModule(LightningDataModule):
    """LightningDataModule for MNIST via regular N×N grid graphs.

    Each 28×28 grayscale image is divided into grid_size×grid_size equal patches.
    Edges connect 4-connected grid neighbors (bidirectional + self-loops).

    Args:
        data_dir: Path where MNIST data is downloaded.
        grid_size: Patches per spatial axis (default 7 → 49 nodes of 4×4 px).
        batch_size: Number of graphs per batch.
        num_workers: DataLoader worker processes.
        val_split: Fraction of training data held out for validation.
        use_raw_pixels: If ``False`` (default), node features are
            ``[gray_mean, x_mean, y_mean]`` (3 features).
            If ``True``, node features are the flattened patch pixels
            (4×4×1 = 16 features for the default grid_size=7).
    """

    num_classes: int = 10
    _IMAGE_H: int = 28
    _IMAGE_W: int = 28
    _IMAGE_C: int = 1

    def __init__(
        self,
        data_dir: str = "data/mnist",
        grid_size: int = 7,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
        use_raw_pixels: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.use_raw_pixels = use_raw_pixels

        ph = self._IMAGE_H // grid_size
        pw = self._IMAGE_W // grid_size
        self.num_features = ph * pw * self._IMAGE_C if use_raw_pixels else self._IMAGE_C + 2

    def _cache(self, split: str) -> str:
        suffix = "_raw" if self.use_raw_pixels else ""
        return os.path.join(self.data_dir, f"mnist_{split}_grid{self.grid_size}{suffix}.pt")

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            raw = MNIST(self.data_dir, train=True, download=True)
            # raw.data: Tensor (N, 28, 28) uint8  →  (N, 28, 28, 1) float32 /255
            images = raw.data.numpy()[:, :, :, None].astype(np.float32) / 255.0
            labels = raw.targets.numpy()

            n_total = len(labels)
            n_val = int(n_total * self.val_split)
            n_train = n_total - n_val
            indices = list(range(n_total))

            full_ds = GridGraphDataset(
                images, labels, self.grid_size, self._cache("train"), self.use_raw_pixels
            )
            self.train_ds = Subset(full_ds, indices[:n_train])
            self.val_ds = Subset(full_ds, indices[n_train:])

        if stage in ("test", None):
            raw = MNIST(self.data_dir, train=False, download=True)
            images = raw.data.numpy()[:, :, :, None].astype(np.float32) / 255.0
            labels = raw.targets.numpy()

            self.test_ds = GridGraphDataset(
                images, labels, self.grid_size, self._cache("test"), self.use_raw_pixels
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=superpixel_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=superpixel_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=superpixel_collate,
        )
