from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from .model import GAT_MNIST


class GATLightningModule(L.LightningModule):
    """PyTorch Lightning wrapper for GAT_MNIST on superpixel graphs.

    Datamodule contract
    -------------------
    Batches must be ``(h, adj, src, tgt, Msrc, Mtgt, Mgraph, labels)`` produced
    by ``superpixel_collate``, where:
    - ``h``: ``Tensor[N_total, num_features]`` — concatenated node features.
    - ``adj``: ``Tensor[N_total, N_total]`` — block-diagonal adjacency.
    - ``src``, ``tgt``: ``Tensor[M_total]`` — edge endpoint indices.
    - ``Msrc``, ``Mtgt``: ``Tensor[N_total, M_total]`` — incidence matrices.
    - ``Mgraph``: ``Tensor[N_total, B]`` — node-to-graph assignment.
    - ``labels``: ``Tensor[B]`` — integer class indices.

    Args:
        num_features: Input features per node (3 for MNIST, 5 for SVHN).
        num_classes: Number of output classes.
        num_heads: Attention heads per GAT layer (list of 3 ints).
        lr: Peak learning rate for AdamW.
        weight_decay: AdamW weight decay.
        warmup_epochs: Linear-warmup length (epochs).
        max_epochs: Total training epochs (for cosine decay end).
    """

    def __init__(
        self,
        num_features: int = 5,
        num_classes: int = 10,
        num_heads: list[int] = [2, 2, 2],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = GAT_MNIST(
            num_features=num_features,
            num_classes=num_classes,
            num_heads=list(num_heads),
        )

    def forward(self, h, adj, src, tgt, Msrc, Mtgt, Mgraph):
        return self.model(h, adj, src, tgt, Msrc, Mtgt, Mgraph)

    def _shared_step(self, batch, stage: str):
        h, adj, src, tgt, Msrc, Mtgt, Mgraph, labels = batch
        logits = self(h, adj, src, tgt, Msrc, Mtgt, Mgraph)
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
            top_k=min(5, self.hparams.num_classes),
        )
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc_top1", top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc_top5", top5, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

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
