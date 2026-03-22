from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn import Sequential as Seq

from vision_gnn.models.seg_head import FPNDecoder
from vision_gnn.models.vig.modelling import Grapher, act_layer


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act="relu",
        drop_path=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


class Stem(nn.Module):
    """Overlapping patch embedding: 224 → 56 (stride 4)."""

    def __init__(self, in_dim=3, out_dim=48, act="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            act_layer(act),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.convs(x)


class Downsample(nn.Module):
    """Stride-2 conv to halve spatial resolution between pyramid stages."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.conv(x)


class PyramidDeepGCN(nn.Module):
    """Pyramid Vision GNN backbone (pvig).

    Builds a 4-stage hierarchical GNN where each stage operates at a different
    spatial scale, with ``Downsample`` layers between stages.

    Args:
        num_classes:      Output classes.
        blocks:           Number of Grapher+FFN blocks per stage, e.g. [2,2,6,2].
        channels:         Feature channels per stage, e.g. [48,96,240,384].
        k:                Base KNN neighbors (constant across all blocks).
        conv:             Graph conv type: ``'mr'`` | ``'edge'`` | ``'sage'`` | ``'gin'``.
        act:              Activation: ``'relu'`` | ``'gelu'`` | ``'prelu'`` | ``'leakyrelu'`` | ``'hswish'``.
        norm:             Normalisation: ``'batch'`` | ``'instance'``.
        bias:             Bias in conv layers.
        use_stochastic:   Stochastic graph construction.
        epsilon:          Epsilon for stochastic graph.
        drop_path_rate:   Max stochastic depth drop rate.
        dropout:          Dropout before the classifier head.
        in_channels:      Input image channels.
    """

    _REDUCE_RATIOS = [4, 2, 1, 1]  # spatial reduce ratio per stage

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
    ):
        super().__init__()
        n_blocks = sum(blocks)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        num_knn = [k] * n_blocks
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(in_dim=in_channels, out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))

        self.backbone = nn.ModuleList()
        idx = 0
        HW = (224 // 4) ** 2  # 56×56 after stem

        for stage, n_stage_blocks in enumerate(blocks):
            if stage > 0:
                self.backbone.append(Downsample(channels[stage - 1], channels[stage]))
                HW = HW // 4  # halve H and W each → /4 tokens

            reduce = self._REDUCE_RATIOS[stage]
            dilation = min(idx // 4 + 1, max_dilation)

            for _ in range(n_stage_blocks):
                self.backbone.append(
                    Seq(
                        Grapher(
                            channels[stage],
                            num_knn[idx],
                            dilation,
                            conv,
                            act,
                            norm,
                            bias,
                            use_stochastic,
                            epsilon,
                            reduce,
                            n=HW,
                            drop_path=dpr[idx],
                            relative_pos=True,
                        ),
                        FFN(
                            channels[stage],
                            channels[stage] * 4,
                            act=act,
                            drop_path=dpr[idx],
                        ),
                    )
                )
                idx += 1

        self.prediction = Seq(
            nn.Conv2d(channels[-1], 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            act_layer(act),
            nn.Dropout(dropout),
            nn.Conv2d(1024, num_classes, 1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.stem(x) + self.pos_embed
        for block in self.backbone:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


class PyramidDeepGCNSeg(nn.Module):
    """pViG backbone for dense prediction — returns multi-scale feature maps.

    Identical block construction to :class:`PyramidDeepGCN` but organises the
    backbone into per-stage module lists so that ``forward`` can capture the
    output after each stage.  No classification head.

    Args:
        Same as :class:`PyramidDeepGCN` except ``num_classes`` (removed) and
        ``dropout`` (removed).

    Returns (from ``forward``):
        Tuple ``(f0, f1, f2, f3)`` where:
        - ``f0``: ``[B, channels[0], 56, 56]``
        - ``f1``: ``[B, channels[1], 28, 28]``
        - ``f2``: ``[B, channels[2], 14, 14]``
        - ``f3``: ``[B, channels[3],  7,  7]``

    Note:
        Input resolution is fixed at 224×224; ``pos_embed`` and the Grapher
        ``n`` parameter are both tied to the 56×56 stem output.
    """

    _REDUCE_RATIOS = [4, 2, 1, 1]

    def __init__(
        self,
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
        in_channels: int = 3,
    ):
        super().__init__()
        n_blocks = sum(blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        num_knn = [k] * n_blocks
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(in_dim=in_channels, out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        idx = 0
        HW = (224 // 4) ** 2

        for stage, n_stage_blocks in enumerate(blocks):
            if stage > 0:
                self.downsamples.append(Downsample(channels[stage - 1], channels[stage]))
                HW = HW // 4

            reduce = self._REDUCE_RATIOS[stage]
            dilation = min(idx // 4 + 1, max_dilation)
            stage_blocks = nn.ModuleList()

            for _ in range(n_stage_blocks):
                stage_blocks.append(
                    Seq(
                        Grapher(
                            channels[stage],
                            num_knn[idx],
                            dilation,
                            conv,
                            act,
                            norm,
                            bias,
                            use_stochastic,
                            epsilon,
                            reduce,
                            n=HW,
                            drop_path=dpr[idx],
                            relative_pos=True,
                        ),
                        FFN(
                            channels[stage],
                            channels[stage] * 4,
                            act=act,
                            drop_path=dpr[idx],
                        ),
                    )
                )
                idx += 1

            self.stages.append(stage_blocks)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.stem(x) + self.pos_embed
        features = []
        for i, stage_blocks in enumerate(self.stages):
            if i > 0:
                x = self.downsamples[i - 1](x)
            for block in stage_blocks:
                x = block(x)
            features.append(x)
        return tuple(features)


class PVigSeg(nn.Module):
    """Pyramid Vision GNN for semantic segmentation.

    Combines :class:`PyramidDeepGCNSeg` backbone with :class:`FPNDecoder` to
    produce per-pixel logits at the original image resolution.

    Args:
        num_classes:    Segmentation classes (21 for VOC2012).
        blocks:         Grapher+FFN blocks per stage, e.g. ``[2,2,6,2]``.
        channels:       Feature channels per stage, e.g. ``[48,96,240,384]``.
        k:              KNN neighbors (constant across all blocks).
        conv:           Graph conv type: ``'mr'`` | ``'edge'`` | ``'sage'`` | ``'gin'``.
        act:            Activation function name.
        norm:           Normalisation: ``'batch'`` | ``'instance'``.
        bias:           Bias in conv layers.
        use_stochastic: Stochastic graph construction.
        epsilon:        Stochastic graph epsilon.
        drop_path_rate: Max stochastic depth drop rate.
        dropout:        Spatial dropout in FPN decoder.
        fpn_channels:   Uniform channel width inside the FPN.
        in_channels:    Input image channels.
    """

    def __init__(
        self,
        num_classes: int = 21,
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
        dropout: float = 0.1,
        fpn_channels: int = 256,
        in_channels: int = 3,
    ):
        super().__init__()
        self.backbone = PyramidDeepGCNSeg(
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
            in_channels=in_channels,
        )
        self.decoder = FPNDecoder(
            in_channels=list(channels),
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
        features = self.backbone(x)
        return self.decoder(features)
