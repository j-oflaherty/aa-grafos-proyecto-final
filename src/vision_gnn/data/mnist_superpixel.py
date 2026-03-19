from __future__ import annotations

import os

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

from vision_gnn.models.superpixel_gat.graph_utils import (
    SuperpixelGraphDataset,
    superpixel_collate,
)


class MNISTSuperpixelDataModule(LightningDataModule):
    """LightningDataModule for MNIST via superpixel graphs.

    Each 28×28 grayscale image is converted to a graph of superpixels.
    Node features are: [gray_mean, x_mean, y_mean] (3 features).

    Args:
        data_dir: Path where MNIST data is downloaded.
        desired_nodes: Target superpixel count per image.
        batch_size: Number of graphs per batch.
        num_workers: DataLoader worker processes.
        val_split: Fraction of training data held out for validation.
    """

    num_classes: int = 10
    num_features: int = 3  # Gray (1) + XY (2)

    def __init__(
        self,
        data_dir: str = "data/mnist",
        desired_nodes: int = 75,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.desired_nodes = desired_nodes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

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

            cache = os.path.join(self.data_dir, f"mnist_train_n{self.desired_nodes}.pt")
            full_ds = SuperpixelGraphDataset(images, labels, self.desired_nodes, cache)

            self.train_ds = Subset(full_ds, indices[:n_train])
            self.val_ds = Subset(full_ds, indices[n_train:])

        if stage in ("test", None):
            raw = MNIST(self.data_dir, train=False, download=True)
            images = raw.data.numpy()[:, :, :, None].astype(np.float32) / 255.0
            labels = raw.targets.numpy()

            cache = os.path.join(self.data_dir, f"mnist_test_n{self.desired_nodes}.pt")
            self.test_ds = SuperpixelGraphDataset(images, labels, self.desired_nodes, cache)

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
