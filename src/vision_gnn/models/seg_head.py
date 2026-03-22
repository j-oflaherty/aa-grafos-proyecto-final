from typing import List

import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    """Lightweight FPN decoder for dense prediction.

    Takes a tuple of 4 feature maps at strides (4, 8, 16, 32) relative to the
    input image (i.e. 56×56, 28×28, 14×14, 7×7 for 224×224 input), runs a
    top-down lateral-connection pathway, and produces per-pixel logits at the
    original image resolution.

    Args:
        in_channels:  Channel widths of the 4 encoder stages, finest first.
                      e.g. ``[48, 96, 240, 384]`` for pViG-Ti or
                      ``[64, 128, 256, 512]`` for ResNet-18.
        fpn_channels: Uniform channel width inside the FPN (default 256).
        num_classes:  Number of output classes.
        dropout:      Spatial dropout rate before the final 1×1 conv.
    """

    def __init__(
        self,
        in_channels: List[int],
        fpn_channels: int = 256,
        num_classes: int = 21,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Lateral 1×1 projections — one per encoder stage
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, fpn_channels, 1) for c in in_channels]
        )
        self.dropout = nn.Dropout2d(dropout)
        self.output_conv = nn.Conv2d(fpn_channels, num_classes, 1)

    def forward(self, features: tuple):
        """
        Args:
            features: tuple of 4 tensors ``(f0, f1, f2, f3)`` where ``f0`` is
                      the finest (56×56) and ``f3`` is the coarsest (7×7).

        Returns:
            Tensor ``[B, num_classes, H, W]`` at the original image resolution
            (4 × the finest feature map, i.e. 224×224 for 224-px input).
        """
        # Project each stage to fpn_channels
        laterals = [conv(f) for conv, f in zip(self.laterals, features)]

        # Top-down pathway: start from coarsest, upsample and add
        p = laterals[-1]
        for lat in reversed(laterals[:-1]):
            p = F.interpolate(p, size=lat.shape[-2:], mode="bilinear", align_corners=False)
            p = p + lat

        # p is now at the finest feature resolution (56×56 for 224-px input).
        # Upsample ×4 back to the original image size.
        p = F.interpolate(p, scale_factor=4, mode="bilinear", align_corners=False)
        return self.output_conv(self.dropout(p))
