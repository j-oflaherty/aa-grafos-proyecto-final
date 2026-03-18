from typing import List

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from .pvig import PyramidDeepGCN


class PVigLightningModule(L.LightningModule):
    """PyTorch Lightning wrapper for the Pyramid Vision GNN (pViG) model.

    Datamodule contract
    -------------------
    Batches must be ``(images, labels)`` where:
    - ``images``: ``Tensor[B, 3, 224, 224]`` — RGB, float32.
    - ``labels``: ``Tensor[B]`` — integer class indices.

    Args:
        num_classes:      Output classes (10 for MNIST, 1000 for ImageNet).
        blocks:           Grapher+FFN blocks per stage, e.g. ``[2,2,6,2]``.
        channels:         Feature channels per stage, e.g. ``[48,96,240,384]``.
        k:                KNN neighbors (constant across all blocks).
        conv:             Graph conv type: ``'mr'`` | ``'edge'`` | ``'sage'`` | ``'gin'``.
        act:              Activation: ``'relu'`` | ``'gelu'`` | ``'prelu'`` | ``'leakyrelu'`` | ``'hswish'``.
        norm:             Normalisation: ``'batch'`` | ``'instance'``.
        bias:             Bias in conv layers.
        use_stochastic:   Stochastic graph construction.
        epsilon:          Stochastic graph epsilon.
        drop_path_rate:   Max stochastic depth drop rate.
        dropout:          Dropout before the classifier head.
        in_channels:      Input channels (3 for RGB).
        lr:               Peak learning rate for AdamW.
        weight_decay:     AdamW weight decay.
        warmup_epochs:    Linear-warmup length (epochs).
        max_epochs:       Total training epochs (for cosine decay end).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        blocks: List[int] = (2, 2, 6, 2),
        channels: List[int] = (48, 96, 240, 384),
        k: int = 9,
        conv: str = "mr",
        act: str = "gelu",
        norm: str = "batch",
        bias: bool = True,
        use_stochastic: bool = False,
        epsilon: float = 0.2,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        in_channels: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = PyramidDeepGCN(
            num_classes=num_classes,
            blocks=list(blocks),
            channels=list(channels),
            k=k,
            conv=conv,
            act=act,
            norm=norm,
            bias=bias,
            use_stochastic=use_stochastic,
            epsilon=epsilon,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            in_channels=in_channels,
        )

    # ------------------------------------------------------------------
    # Preset constructors matching the four official pViG variants
    # ------------------------------------------------------------------

    @classmethod
    def from_tiny(cls, **kwargs) -> "PVigLightningModule":
        """pViG-Ti: channels=[48,96,240,384], blocks=[2,2,6,2]."""
        return cls(blocks=[2, 2, 6, 2], channels=[48, 96, 240, 384], **kwargs)

    @classmethod
    def from_small(cls, **kwargs) -> "PVigLightningModule":
        """pViG-S: channels=[80,160,400,640], blocks=[2,2,6,2]."""
        return cls(blocks=[2, 2, 6, 2], channels=[80, 160, 400, 640], **kwargs)

    @classmethod
    def from_medium(cls, **kwargs) -> "PVigLightningModule":
        """pViG-M: channels=[96,192,384,768], blocks=[2,2,16,2]."""
        return cls(blocks=[2, 2, 16, 2], channels=[96, 192, 384, 768], **kwargs)

    @classmethod
    def from_base(cls, **kwargs) -> "PVigLightningModule":
        """pViG-B: channels=[128,256,512,1024], blocks=[2,2,18,2]."""
        return cls(blocks=[2, 2, 18, 2], channels=[128, 256, 512, 1024], **kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        top1 = accuracy(
            logits,
            labels,
            task="multiclass",
            num_classes=self.hparams.num_classes,
            top_k=1,
        )
        top5 = accuracy(
            logits,
            labels,
            task="multiclass",
            num_classes=self.hparams.num_classes,
            top_k=5,
        )

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        self.log(f"{stage}/acc_top1", top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc_top5", top5, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.hparams.max_epochs - self.hparams.warmup_epochs),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
