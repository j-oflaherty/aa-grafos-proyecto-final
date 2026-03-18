import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from torch.nn import Sequential as Seq

from vision_gnn.model import Grapher, act_layer


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
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """Image to Visual Word Embedding (overlapping patch embed)."""

    def __init__(self, img_size=224, in_dim=3, out_dim=768, act="relu"):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 8),
            act_layer(act),
            nn.Conv2d(out_dim // 8, out_dim // 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 4),
            act_layer(act),
            nn.Conv2d(out_dim // 4, out_dim // 2, 3, stride=2, padding=1),
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


class DeepGCN(nn.Module):
    """Vision GNN backbone.

    Args:
        num_classes:    Number of output classes.
        n_filters:      Channel width of deep features.
        n_blocks:       Number of Grapher+FFN blocks.
        k:              Base number of KNN neighbors (grows linearly to 2k across blocks).
        conv:           Graph conv type: 'mr' | 'edge' | 'sage' | 'gin'.
        act:            Activation: 'relu' | 'gelu' | 'prelu' | 'leakyrelu' | 'hswish'.
        norm:           Normalization: 'batch' | 'instance'.
        bias:           Use bias in conv layers.
        use_dilation:   Use dilated KNN graph (increases receptive field).
        use_stochastic: Stochastic graph construction.
        epsilon:        Epsilon for stochastic graph.
        drop_path_rate: Max stochastic depth drop rate.
        dropout:        Dropout rate before the classification head.
        in_channels:    Input image channels (3 for RGB, 1 for grayscale).
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
    ):
        super().__init__()
        self.n_blocks = n_blocks

        self.stem = Stem(out_dim=n_filters, act=act, in_dim=in_channels)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k, 2 * k, n_blocks)]
        max_dilation = 196 // max(num_knn)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_filters, 14, 14))

        dilation_fn = lambda i: min(i // 4 + 1, max_dilation) if use_dilation else 1

        self.backbone = Seq(
            *[
                Seq(
                    Grapher(
                        n_filters,
                        num_knn[i],
                        dilation_fn(i),
                        conv,
                        act,
                        norm,
                        bias,
                        use_stochastic,
                        epsilon,
                        1,
                        drop_path=dpr[i],
                    ),
                    FFN(n_filters, n_filters * 4, act=act, drop_path=dpr[i]),
                )
                for i in range(n_blocks)
            ]
        )

        self.prediction = Seq(
            nn.Conv2d(n_filters, 1024, 1, bias=True),
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
