from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from skimage.segmentation import slic
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SuperpixelGraph:
    """Container for one superpixel graph generated from an input image.

    Attributes:
        node_features: Float32 tensor of shape (N, C+2), where each row is
            [channel_means..., x_mean, y_mean].
        edge_index: Int64 tensor of shape (2*M, 2) with directed edges
            (including reverse edges and self-loops).
        adjacency: Float32 tensor of shape (N, N), dense adjacency matrix.
        segments: Int64 tensor of shape (H, W), per-pixel superpixel id map.
    """

    node_features: Tensor
    edge_index: Tensor
    adjacency: Tensor
    segments: Tensor


def _build_graph_arrays(
    image: np.ndarray,
    desired_nodes: int,
    compactness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build superpixel graph arrays from one image.

    Returns:
        h_np: Node features, shape (N, C+2), float32.
        edges_np: Directed edge list, shape (2*M, 2), int64.
        segments: Remapped superpixel ids, shape (H, W), int64.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    segments = slic(
        image,
        n_segments=desired_nodes,
        compactness=compactness,
        slic_zero=True,
        channel_axis=-1,
    )
    H, W = segments.shape[:2]
    n_channels = image.shape[2] if image.ndim == 3 else 1

    # slic_zero may produce non-contiguous IDs; remap to 0..n_nodes-1
    unique_ids = np.unique(segments)
    n_nodes = len(unique_ids)
    remap = np.zeros(int(segments.max()) + 1, dtype=np.int64)
    remap[unique_ids] = np.arange(n_nodes)
    segments = remap[segments]

    # --- Vectorised node features ---
    flat_seg = segments.ravel()  # (H*W,)

    # Pixel colour accumulation
    rgb_sums = np.zeros((n_nodes, n_channels), dtype=np.float64)
    np.add.at(rgb_sums, flat_seg, image.reshape(-1, n_channels).astype(np.float64))

    # Pixel position accumulation (normalised to [0, 1])
    ys, xs = np.mgrid[0:H, 0:W]
    pos = np.stack([xs.ravel() / W, ys.ravel() / H], axis=-1)  # (H*W, 2)
    pos_sums = np.zeros((n_nodes, 2), dtype=np.float64)
    np.add.at(pos_sums, flat_seg, pos)

    counts = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(counts, flat_seg, 1.0)

    rgb_mean = (rgb_sums / counts[:, None]).astype(np.float32)
    pos_mean = (pos_sums / counts[:, None]).astype(np.float32)
    h_np = np.concatenate([rgb_mean, pos_mean], axis=1)  # (N, C+2)

    # --- Edge list (array-shift method) ---
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    # Remove self-loops from neighbour edges (will be added explicitly)
    non_self = bneighbors[:, bneighbors[0] != bneighbors[1]]  # (2, K)

    # Self-loops
    arange = np.arange(n_nodes)
    self_loops = np.stack([arange, arange])  # (2, N)

    # Forward edges: non-self adjacency + self-loops
    fwd = np.concatenate([non_self, self_loops], axis=1).T.astype(np.int64)  # (M, 2)
    rev = fwd[:, [1, 0]]  # (M, 2) — reversed

    edges_np = np.concatenate([fwd, rev], axis=0)  # (2*M, 2)
    return h_np, edges_np, segments.astype(np.int64)


def get_superpixel_graph_from_image(
    image: np.ndarray,
    desired_nodes: int = 75,
    compactness: float = 0.1,
) -> SuperpixelGraph:
    """Return the full superpixel graph produced from one input image.

    This is useful for visualization/debugging because it exposes both the
    SLIC segmentation map and graph tensors in one object.
    """
    h_np, edges_np, segments_np = _build_graph_arrays(
        image,
        desired_nodes,
        compactness=compactness,
    )

    node_features = torch.from_numpy(h_np)
    edge_index = torch.from_numpy(edges_np)
    segments = torch.from_numpy(segments_np)

    n_nodes = node_features.shape[0]
    adjacency = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    adjacency[edge_index[:, 0], edge_index[:, 1]] = 1.0

    return SuperpixelGraph(
        node_features=node_features,
        edge_index=edge_index,
        adjacency=adjacency,
        segments=segments,
    )


def get_graph_from_image(
    image: np.ndarray,
    desired_nodes: int = 75,
    compactness: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Convert an image to a superpixel graph.

    Args:
        image: Float32 array of shape (H, W, C), values in [0, 1].
        desired_nodes: Target number of superpixels.
        compactness: Balances color similarity vs grid regularity in SLIC.

    Returns:
        h: Float32 tensor of shape (N, C+2) — per-node features (mean colour + mean xy position).
        edges: Int64 tensor of shape (2*M, 2) — bidirectional edge list (src, tgt).
    """
    h_np, edges_np, _ = _build_graph_arrays(
        image,
        desired_nodes,
        compactness=compactness,
    )

    return (
        torch.from_numpy(h_np),
        torch.from_numpy(edges_np),
    )


def get_grid_graph_from_image(
    image: np.ndarray,
    grid_size: int = 8,
    use_raw_pixels: bool = False,
) -> tuple[Tensor, Tensor]:
    """Convert an image to a regular N×N patch grid graph.

    The image is divided into ``grid_size × grid_size`` equal-sized patches.

    Args:
        image: Float32 array of shape (H, W, C), values in [0, 1].
            H and W must each be divisible by ``grid_size``.
        grid_size: Number of patches along each spatial axis.
        use_raw_pixels: If ``False`` (default), node features are
            ``[mean_colour..., x_centre, y_centre]`` — shape ``(N, C+2)``.
            If ``True``, node features are the flattened patch pixels —
            shape ``(N, ph*pw*C)``.

    Returns:
        h: Float32 tensor of shape ``(N, F)`` where N = grid_size².
        edges: Int64 tensor of shape (2*M, 2) — bidirectional edge list.
    """
    H, W, C = image.shape
    assert H % grid_size == 0 and W % grid_size == 0, (
        f"Image dimensions ({H}×{W}) must be divisible by grid_size={grid_size}"
    )
    ph, pw = H // grid_size, W // grid_size

    # Reshape to (grid_size, grid_size, ph, pw, C) — shared by both modes
    patches = image.reshape(grid_size, ph, grid_size, pw, C).transpose(0, 2, 1, 3, 4)

    if use_raw_pixels:
        # Flatten each patch's pixels: (G, G, ph*pw*C)
        h_np = patches.reshape(grid_size * grid_size, ph * pw * C).astype(np.float32)
    else:
        # Per-patch colour means + normalised centre coordinates
        colour_means = patches.mean(axis=(2, 3)).astype(np.float32)  # (G, G, C)
        row_c = ((np.arange(grid_size) * ph + ph / 2) / H).astype(np.float32)
        col_c = ((np.arange(grid_size) * pw + pw / 2) / W).astype(np.float32)
        yy, xx = np.meshgrid(row_c, col_c, indexing="ij")  # (G, G)
        h_np = np.concatenate(
            [colour_means.reshape(-1, C), np.stack([xx.ravel(), yy.ravel()], axis=-1)],
            axis=1,
        )  # (N, C+2)

    # 4-connected edge list (right + below neighbors)
    node_ids = np.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    right_src = node_ids[:, :-1].ravel()
    right_tgt = node_ids[:, 1:].ravel()
    below_src = node_ids[:-1, :].ravel()
    below_tgt = node_ids[1:, :].ravel()

    non_self = np.stack(
        [np.concatenate([right_src, below_src]), np.concatenate([right_tgt, below_tgt])]
    )  # (2, K)

    arange = np.arange(grid_size * grid_size)
    self_loops = np.stack([arange, arange])  # (2, N)

    fwd = np.concatenate([non_self, self_loops], axis=1).T.astype(np.int64)  # (M, 2)
    rev = fwd[:, [1, 0]]

    edges_np = np.concatenate([fwd, rev], axis=0)  # (2*M, 2)

    return torch.from_numpy(h_np), torch.from_numpy(edges_np)


def batch_graphs(gs: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, ...]:
    """Batch a list of (h, edges) graphs into block-diagonal tensors.

    Args:
        gs: List of (h, edges) pairs from ``get_graph_from_image``.

    Returns:
        h, adj, src, tgt, Msrc, Mtgt, Mgraph — all torch tensors.
    """
    G = len(gs)
    N = sum(g[0].shape[0] for g in gs)
    M = sum(g[1].shape[0] for g in gs)

    h = torch.cat([g[0] for g in gs])  # (N, F)
    adj = torch.zeros(N, N)
    src = torch.zeros(M, dtype=torch.long)
    tgt = torch.zeros(M, dtype=torch.long)
    Msrc = torch.zeros(N, M)
    Mtgt = torch.zeros(N, M)
    Mgraph = torch.zeros(N, G)

    n_acc = 0
    m_acc = 0
    for g_idx, (node_h, edges) in enumerate(gs):
        n = node_h.shape[0]
        m = edges.shape[0]

        s = edges[:, 0] + n_acc  # (m,)
        t = edges[:, 1] + n_acc  # (m,)
        e_idx = torch.arange(m_acc, m_acc + m)

        adj[s, t] = 1
        adj[t, s] = 1
        src[e_idx] = s
        tgt[e_idx] = t
        Msrc[s, e_idx] = 1
        Mtgt[t, e_idx] = 1
        Mgraph[n_acc : n_acc + n, g_idx] = 1

        n_acc += n
        m_acc += m

    return h, adj, src, tgt, Msrc, Mtgt, Mgraph


class SuperpixelGraphDataset(Dataset):
    """Pre-computes superpixel graphs from raw images with optional disk cache.

    Args:
        images: Array of float32 images, shape (N, H, W, C), values in [0, 1].
        labels: 1-D array of integer class labels, length N.
        desired_nodes: Target superpixel count per image.
        cache_path: Optional ``.pt`` file path. Graphs are loaded from disk if
            the file exists; otherwise they are computed and saved there.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels,
        desired_nodes: int = 75,
        cache_path: Optional[str] = None,
    ) -> None:
        self.labels = labels

        if cache_path is not None and os.path.exists(cache_path):
            self.graphs = torch.load(cache_path, weights_only=False)
        else:
            self.graphs = [
                get_graph_from_image(images[i], desired_nodes)
                for i in range(len(images))
            ]
            if cache_path is not None:
                torch.save(self.graphs, cache_path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[tuple[Tensor, Tensor], int]:
        return self.graphs[idx], int(self.labels[idx])


class GridGraphDataset(Dataset):
    """Pre-computes regular grid graphs from raw images with optional disk cache.

    Args:
        images: Float32 array of shape (N, H, W, C), values in [0, 1].
        labels: 1-D array of integer class labels, length N.
        grid_size: Number of patches along each spatial axis (N×N grid).
        cache_path: Optional ``.pt`` file path. Graphs are loaded from disk if
            the file exists; otherwise they are computed and saved there.
        use_raw_pixels: If ``False`` (default), node features are
            ``[mean_colour..., x_centre, y_centre]``.
            If ``True``, node features are the flattened patch pixels.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels,
        grid_size: int = 8,
        cache_path: Optional[str] = None,
        use_raw_pixels: bool = False,
    ) -> None:
        self.labels = labels

        if cache_path is not None and os.path.exists(cache_path):
            self.graphs = torch.load(cache_path, weights_only=False)
        else:
            self.graphs = [
                get_grid_graph_from_image(images[i], grid_size, use_raw_pixels)
                for i in range(len(images))
            ]
            if cache_path is not None:
                torch.save(self.graphs, cache_path)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[tuple[Tensor, Tensor], int]:
        return self.graphs[idx], int(self.labels[idx])


def superpixel_collate(batch) -> tuple:
    """Custom collate function for variable-size superpixel graphs.

    Args:
        batch: List of ``((h, edges), label)`` from ``SuperpixelGraphDataset``.

    Returns:
        ``(h, adj, src, tgt, Msrc, Mtgt, Mgraph, labels)`` — all tensors.
    """
    graphs, labels = zip(*batch)
    h, adj, src, tgt, Msrc, Mtgt, Mgraph = batch_graphs(list(graphs))
    return h, adj, src, tgt, Msrc, Mtgt, Mgraph, torch.tensor(labels, dtype=torch.long)
