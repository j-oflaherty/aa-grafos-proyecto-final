import lightning as L
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from torchmetrics.functional import accuracy

from .resnet import MiniResNet

# torchvision ResNet variants supported via model_name
_TORCHVISION_RESNETS = {
    "resnet18": tvm.resnet18,
    "resnet34": tvm.resnet34,
    "resnet50": tvm.resnet50,
    "resnet101": tvm.resnet101,
    "resnet152": tvm.resnet152,
}


def _build_model(
    model_name: str,
    num_classes: int,
    in_channels: int,
    hidden_dim: int,
    num_layers: int,
) -> torch.nn.Module:
    if model_name == "mini":
        return MiniResNet(
            num_classes=num_classes,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    if model_name in _TORCHVISION_RESNETS:
        net = _TORCHVISION_RESNETS[model_name](weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        return net
    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Choose 'mini' or one of {sorted(_TORCHVISION_RESNETS)}."
    )


class ResNetLightningModule(L.LightningModule):
    """PyTorch Lightning wrapper for ResNet backbones.

    Datamodule contract
    -------------------
    Batches must be ``(images, labels)`` where:
    - ``images``: ``Tensor[B, 3, 224, 224]`` — RGB, float32.
    - ``labels``: ``Tensor[B]`` — integer class indices.

    Model selection via ``model_name``
    -----------------------------------
    - ``'mini'``      — MiniResNet (custom lightweight backbone, ~701 k params at defaults).
                        Uses ``hidden_dim`` and ``num_layers`` to control width/depth.
    - ``'resnet18'``  — torchvision ResNet-18  (~11.2 M params)
    - ``'resnet34'``  — torchvision ResNet-34  (~21.3 M params)
    - ``'resnet50'``  — torchvision ResNet-50  (~25.6 M params)
    - ``'resnet101'`` — torchvision ResNet-101 (~44.5 M params)
    - ``'resnet152'`` — torchvision ResNet-152 (~60.2 M params)

    All torchvision models are initialised with ``weights=None`` (train from scratch).
    ``hidden_dim`` and ``num_layers`` are ignored for torchvision variants.

    Args:
        model_name:    Backbone to use (see above).
        num_classes:   Output classes.
        in_channels:   Input channels (only used by ``'mini'``).
        hidden_dim:    MiniResNet stem channels (only used by ``'mini'``).
        num_layers:    MiniResNet residual stages (only used by ``'mini'``).
        lr:            Peak learning rate for AdamW.
        weight_decay:  AdamW weight decay.
        warmup_epochs: Linear-warmup length (epochs).
        max_epochs:    Total training epochs (for cosine decay end).
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 10,
        in_channels: int = 3,
        hidden_dim: int = 32,
        num_layers: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = _build_model(model_name, num_classes, in_channels, hidden_dim, num_layers)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # Mixup / CutMix yield soft (float) labels — use argmax for accuracy.
        hard_labels = labels.argmax(1) if labels.is_floating_point() else labels

        top1 = accuracy(
            logits, hard_labels,
            task="multiclass",
            num_classes=self.hparams.num_classes,
            top_k=1,
        )
        top5 = accuracy(
            logits, hard_labels,
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
