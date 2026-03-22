"""Small ResNet for MNIST/low-resolution experiments.

Architecture (MiniResNet)
-------------------------
stem    → Conv(in_channels, hidden_dim, 7×7, stride=2) → BN → ReLU → MaxPool(3, stride=2)
stage 0 → 2 × BasicBlock(hidden_dim,       hidden_dim,       stride=1)
stage 1 → 2 × BasicBlock(hidden_dim,       hidden_dim*2,     stride=2)
...
stage N → 2 × BasicBlock(hidden_dim*2^(N-1), hidden_dim*2^N, stride=2)
head    → AdaptiveAvgPool → Linear(hidden_dim * 2^(num_layers-1), num_classes)

Defaults (hidden_dim=32, num_layers=3): ~701 k parameters for 10-class MNIST.
"""

import torch.nn as nn
import torchvision.models as tvm

from vision_gnn.models.seg_head import FPNDecoder


class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample: nn.Sequential | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(
    in_ch: int, out_ch: int, num_blocks: int, stride: int = 1
) -> nn.Sequential:
    blocks = [_BasicBlock(in_ch, out_ch, stride)]
    for _ in range(1, num_blocks):
        blocks.append(_BasicBlock(out_ch, out_ch))
    return nn.Sequential(*blocks)


class MiniResNet(nn.Module):
    """Compact ResNet for 224×224 3-channel input.

    Args:
        num_classes:  Output classes.
        in_channels:  Input channels.
        hidden_dim:   Stem output channels; each subsequent stage doubles this.
        num_layers:   Number of residual stages (min 1).
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        hidden_dim: int = 32,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stages = []
        in_ch = hidden_dim
        for i in range(num_layers):
            out_ch = hidden_dim * (2 ** i)
            stride = 1 if i == 0 else 2
            stages.append(_make_layer(in_ch, out_ch, num_blocks=2, stride=stride))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class ResNet18Seg(nn.Module):
    """ResNet-18 backbone + FPN decoder for semantic segmentation.

    Uses torchvision ResNet-18 as a feature extractor, discarding the average
    pool and fc head.  Features are collected after each of the four residual
    stages, matching the spatial resolutions of pViG:

    ======  ========  ==========
    Stage   Channels  Resolution
    ======  ========  ==========
    layer1     64      56 × 56
    layer2    128      28 × 28
    layer3    256      14 × 14
    layer4    512       7 × 7
    ======  ========  ==========

    An :class:`~vision_gnn.models.seg_head.FPNDecoder` fuses these features
    and upsamples back to the input resolution.

    Args:
        num_classes:   Segmentation classes (21 for VOC2012).
        fpn_channels:  Uniform channel width inside the FPN (default 256).
        dropout:       Spatial dropout rate in the FPN decoder.
    """

    def __init__(
        self,
        num_classes: int = 21,
        fpn_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        backbone = tvm.resnet18(weights=None)

        # Stem: conv1 + bn1 + relu + maxpool → 56×56, 64 ch
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 56×56, 64 ch
        self.layer2 = backbone.layer2  # 28×28, 128 ch
        self.layer3 = backbone.layer3  # 14×14, 256 ch
        self.layer4 = backbone.layer4  #  7×7,  512 ch

        self.decoder = FPNDecoder(
            in_channels=[64, 128, 256, 512],
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x: ``[B, 3, 224, 224]``

        Returns:
            ``[B, num_classes, 224, 224]`` logits.
        """
        f0 = self.layer1(self.layer0(x))  # 56×56, 64 ch
        f1 = self.layer2(f0)              # 28×28, 128 ch
        f2 = self.layer3(f1)              # 14×14, 256 ch
        f3 = self.layer4(f2)              #  7×7,  512 ch
        return self.decoder((f0, f1, f2, f3))
