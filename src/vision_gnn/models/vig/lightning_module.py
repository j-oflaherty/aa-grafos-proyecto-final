import lightning as L
import torch
import torch.nn.functional as F
from timm.utils import ModelEma
from torchmetrics.functional import accuracy

from .vig import DeepGCN


class VigLightningModule(L.LightningModule):
    """PyTorch Lightning wrapper for the Vision GNN (ViG) model.

    Datamodule contract
    -------------------
    Batches must be ``(images, labels)`` where:
    - ``images``: ``Tensor[B, 3, 224, 224]`` — RGB, float32.
    - ``labels``: ``Tensor[B]`` — integer class indices.

    MNIST notes
    -----------
    Resize 28×28 → 224×224 and convert grayscale to 3-channel before feeding.
    Example transform::

        transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    Args:
        num_classes:      Output classes (10 for MNIST, 1000 for ImageNet).
        n_filters:        Channel width of deep features.
        n_blocks:         Number of Grapher+FFN blocks in backbone.
        k:                Base KNN neighbors (grows linearly to 2k).
        conv:             Graph conv type: ``'mr'`` | ``'edge'`` | ``'sage'`` | ``'gin'``.
        act:              Activation: ``'relu'`` | ``'gelu'`` | ``'prelu'`` | ``'leakyrelu'`` | ``'hswish'``.
        norm:             Normalisation: ``'batch'`` | ``'instance'``.
        bias:             Bias in conv layers.
        use_dilation:     Use dilated KNN graph.
        use_stochastic:   Stochastic graph construction.
        epsilon:          Stochastic graph epsilon.
        drop_path_rate:   Max stochastic depth drop rate.
        dropout:          Dropout before the classifier head.
        in_channels:      Input channels (3 for RGB, 1 for grayscale).
        lr:               Peak learning rate for AdamW.
        weight_decay:     AdamW weight decay.
        warmup_epochs:    Linear-warmup length (epochs).
        max_epochs:       Total training epochs (for cosine decay end).
        ema_decay:        EMA decay for shadow weights via timm ModelEma (0.0 = disabled).
        label_smoothing:  Label smoothing passed to cross-entropy loss (0.0 = disabled).
    """

    def __init__(
        self,
        num_classes: int = 1000,
        n_filters: int = 192,
        n_blocks: int = 12,
        k: int = 9,
        conv: str = "mr",
        act: str = "gelu",
        norm: str = "batch",
        bias: bool = True,
        use_dilation: bool = True,
        use_stochastic: bool = False,
        epsilon: float = 0.2,
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        in_channels: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        ema_decay: float = 0.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DeepGCN(
            num_classes=num_classes,
            n_filters=n_filters,
            n_blocks=n_blocks,
            k=k,
            conv=conv,
            act=act,
            norm=norm,
            bias=bias,
            use_dilation=use_dilation,
            use_stochastic=use_stochastic,
            epsilon=epsilon,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            in_channels=in_channels,
        )

        self._ema: ModelEma | None = None  # lazily created after model is on device

    # ------------------------------------------------------------------
    # Preset constructors matching the three official ViG variants
    # ------------------------------------------------------------------

    @classmethod
    def from_tiny(cls, **kwargs) -> "VigLightningModule":
        """ViG-Ti: 192 filters, 12 blocks, GELU."""
        return cls(n_filters=192, n_blocks=12, act="gelu", **kwargs)

    @classmethod
    def from_small(cls, **kwargs) -> "VigLightningModule":
        """ViG-S: 320 filters, 16 blocks, GELU."""
        return cls(n_filters=320, n_blocks=16, act="gelu", **kwargs)

    @classmethod
    def from_base(cls, **kwargs) -> "VigLightningModule":
        """ViG-B: 640 filters, 16 blocks, GELU."""
        return cls(n_filters=640, n_blocks=16, act="gelu", **kwargs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str, model=None):
        images, labels = batch
        m = model if model is not None else self.model
        logits = m(images)
        loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Mixup / CutMix yield soft (float) labels — use argmax for accuracy.
        hard_labels = labels.argmax(1) if labels.is_floating_point() else labels

        top1 = accuracy(
            logits,
            hard_labels,
            task="multiclass",
            num_classes=self.hparams.num_classes,
            top_k=1,
        )
        top5 = accuracy(
            logits,
            hard_labels,
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

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.hparams.ema_decay > 0.0:
            if self._ema is None:
                self._ema = ModelEma(self.model, decay=self.hparams.ema_decay)
            self._ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        ema = self._ema.ema if self._ema is not None else None
        self._shared_step(batch, "val", model=ema)

    def test_step(self, batch, batch_idx):
        ema = self._ema.ema if self._ema is not None else None
        self._shared_step(batch, "test", model=ema)

    # ------------------------------------------------------------------
    # Checkpoint persistence for EMA state
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self._ema is not None:
            checkpoint["ema_state_dict"] = self._ema.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if "ema_state_dict" in checkpoint:
            if self._ema is None:
                self._ema = ModelEma(self.model, decay=self.hparams.ema_decay)
            self._ema.ema.load_state_dict(checkpoint["ema_state_dict"])

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
