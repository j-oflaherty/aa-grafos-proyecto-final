import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayerAdj(nn.Module):
    """More didactic (also memory-hungry) GAT layer."""

    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super().__init__()
        self.f = nn.Linear(2 * d_i, d_o)
        self.w = nn.Linear(2 * d_i, 1)
        self.act = act
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        N = x.size()[0]
        hsrc = x.unsqueeze(0).expand(N, -1, -1)
        htgt = x.unsqueeze(1).expand(-1, N, -1)
        h = torch.cat([hsrc, htgt], dim=2)

        a = self.w(h)
        a_sqz = a.squeeze(2)
        a_zro = -1e16 * torch.ones_like(a_sqz)
        a_msk = torch.where(adj > 0, a_sqz, a_zro)
        a_att = F.softmax(a_msk, dim=1)

        y = self.act(self.f(h))
        y_att = a_att.unsqueeze(-1) * y
        o = y_att.sum(dim=1).squeeze()
        return o


class GATLayerEdgeAverage(nn.Module):
    """GAT layer with average, instead of softmax, attention distribution."""

    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super().__init__()
        self.f = nn.Linear(2 * d_i, d_o)
        self.w = nn.Linear(2 * d_i, 1)
        self.act = act
        self.eps = eps
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        hsrc = x[src]
        htgt = x[tgt]
        h = torch.cat([hsrc, htgt], dim=1)
        y = self.act(self.f(h))
        a = self.w(h)
        a_sum = torch.mm(Mtgt, a) + self.eps
        o = torch.mm(Mtgt, y * a) / a_sum
        assert not torch.isnan(o).any()
        return o


class GATLayerEdgeSoftmax(nn.Module):
    """GAT layer with softmax attention distribution."""

    def __init__(self, d_i, d_o, act=F.relu, eps=1e-6):
        super().__init__()
        self.f = nn.Linear(2 * d_i, d_o)
        self.w = nn.Linear(2 * d_i, 1)
        self.act = act
        self.eps = eps
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.f.weight)
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        hsrc = x[src]
        htgt = x[tgt]
        h = torch.cat([hsrc, htgt], dim=1)
        y = self.act(self.f(h))
        a = self.w(h)
        assert not torch.isnan(a).any()
        a_base, _ = torch.max(a, 0, keepdim=True)
        assert not torch.isnan(a_base).any()
        a_norm = a - a_base
        assert not torch.isnan(a_norm).any()
        a_exp = torch.exp(a_norm)
        assert not torch.isnan(a_exp).any()
        a_sum = torch.mm(Mtgt, a_exp) + self.eps
        assert not torch.isnan(a_sum).any()
        o = torch.mm(Mtgt, y * a_exp) / a_sum
        assert not torch.isnan(o).any()
        return o


class GATLayerMultiHead(nn.Module):

    def __init__(self, d_in, d_out, num_heads):
        super().__init__()
        self.GAT_heads = nn.ModuleList(
            [GATLayerEdgeSoftmax(d_in, d_out) for _ in range(num_heads)]
        )

    def forward(self, x, adj, src, tgt, Msrc, Mtgt):
        return torch.cat([l(x, adj, src, tgt, Msrc, Mtgt) for l in self.GAT_heads], dim=1)


class GAT_MNIST(nn.Module):

    def __init__(self, num_features, num_classes, num_heads=[2, 2, 2]):
        super().__init__()

        self.layer_heads = [1] + num_heads
        self.GAT_layer_sizes = [num_features, 32, 64, 64]
        self.MLP_layer_sizes = [self.layer_heads[-1] * self.GAT_layer_sizes[-1], 32, num_classes]
        self.MLP_acts = [F.relu, lambda x: x]

        self.GAT_layers = nn.ModuleList(
            [
                GATLayerMultiHead(d_in * heads_in, d_out, heads_out)
                for d_in, d_out, heads_in, heads_out in zip(
                    self.GAT_layer_sizes[:-1],
                    self.GAT_layer_sizes[1:],
                    self.layer_heads[:-1],
                    self.layer_heads[1:],
                )
            ]
        )
        self.MLP_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(self.MLP_layer_sizes[:-1], self.MLP_layer_sizes[1:])
            ]
        )

    def forward(self, x, adj, src, tgt, Msrc, Mtgt, Mgraph):
        for l in self.GAT_layers:
            x = l(x, adj, src, tgt, Msrc, Mtgt)
        x = torch.mm(Mgraph.t(), x)
        for layer, act in zip(self.MLP_layers, self.MLP_acts):
            x = act(layer(x))
        return x
