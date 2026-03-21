from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, STL10

from .lightning_module import VigLightningModule

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def _tensor_to_hwc_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to HWC float image in [0, 1]."""
    x = tensor.detach().cpu().numpy().astype(np.float32)
    x = np.transpose(x, (1, 2, 0))
    x = x * IMAGENET_STD[None, None, :] + IMAGENET_MEAN[None, None, :]
    return np.clip(x, 0.0, 1.0)


def _grid_xy(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y pixel coordinates for flattened node indices in raster order."""
    ys, xs = np.divmod(np.arange(h * w, dtype=np.int64), w)
    return xs.astype(np.float32), ys.astype(np.float32)


@torch.no_grad()
def _save_layer_overlay(
    image_hwc: np.ndarray,
    edge_index: torch.Tensor,
    hw: tuple[int, int],
    output_path: Path,
    alpha: float,
    linewidth: float,
    node_size: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for graph visualization. "
            "Install dev deps with: uv sync --group dev"
        ) from exc

    h_nodes, w_nodes = hw
    src = edge_index[0, 0].reshape(-1).numpy()
    dst = edge_index[1, 0].reshape(-1).numpy()

    xg, yg = _grid_xy(h_nodes, w_nodes)

    img_h, img_w = image_hwc.shape[:2]
    xs = (xg + 0.5) / w_nodes * img_w
    ys = (yg + 0.5) / h_nodes * img_h

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_hwc)

    seen: set[tuple[int, int]] = set()
    for u, v in zip(src.tolist(), dst.tolist(), strict=False):
        a = int(u)
        b = int(v)
        if a == b:
            continue
        edge = (a, b) if a < b else (b, a)
        if edge in seen:
            continue
        seen.add(edge)
        ax.plot(
            [xs[a], xs[b]],
            [ys[a], ys[b]],
            color="deepskyblue",
            alpha=alpha,
            linewidth=linewidth,
        )

    ax.scatter(xs, ys, s=node_size, c="tomato", edgecolors="white", linewidths=0.5)
    ax.set_title(f"nodes={h_nodes * w_nodes} edges={len(seen)}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _save_random_patch_neighbors_overlay(
    image_hwc: np.ndarray,
    edge_index: torch.Tensor,
    hw: tuple[int, int],
    output_path: Path,
    alpha: float,
    linewidth: float,
    node_size: float,
    n_random_patches: int,
    seed: int,
) -> None:
    """Draw only edges from a small random subset of query patches."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for graph visualization. "
            "Install dev deps with: uv sync --group dev"
        ) from exc

    h_nodes, w_nodes = hw
    src = edge_index[0, 0].reshape(-1).numpy()
    dst = edge_index[1, 0].reshape(-1).numpy()

    xg, yg = _grid_xy(h_nodes, w_nodes)
    img_h, img_w = image_hwc.shape[:2]
    xs = (xg + 0.5) / w_nodes * img_w
    ys = (yg + 0.5) / h_nodes * img_h

    total_nodes = h_nodes * w_nodes
    n_select = max(1, min(n_random_patches, total_nodes))
    rng = np.random.default_rng(seed)
    selected = rng.choice(total_nodes, size=n_select, replace=False)
    selected_set = {int(i) for i in selected.tolist()}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_hwc)

    drawn_edges: list[tuple[int, int]] = []
    for u, v in zip(src.tolist(), dst.tolist(), strict=False):
        a = int(u)
        b = int(v)
        if b not in selected_set:
            continue
        drawn_edges.append((a, b))
        ax.plot(
            [xs[b], xs[a]],
            [ys[b], ys[a]],
            color="gold",
            alpha=alpha,
            linewidth=linewidth,
        )

    # Context nodes.
    ax.scatter(
        xs,
        ys,
        s=max(3.0, node_size * 0.45),
        c="lightgray",
        edgecolors="none",
        alpha=0.65,
    )

    # Neighbor nodes attached to selected query patches.
    neighbor_nodes = sorted({u for u, _ in drawn_edges})
    if neighbor_nodes:
        ax.scatter(
            xs[neighbor_nodes],
            ys[neighbor_nodes],
            s=max(5.0, node_size * 0.8),
            c="deepskyblue",
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
            zorder=3,
        )

    # Selected query patches.
    selected_sorted = sorted(selected_set)
    ax.scatter(
        xs[selected_sorted],
        ys[selected_sorted],
        s=max(10.0, node_size * 2.0),
        c="crimson",
        edgecolors="white",
        linewidths=0.9,
        zorder=4,
    )

    ax.set_title(f"query_patches={len(selected_set)} shown_edges={len(drawn_edges)}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_dataset(dataset: str, split: str, data_dir: str):
    tfm = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3)
            if dataset == "mnist"
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
        ]
    )

    if dataset == "mnist":
        if split not in {"train", "test"}:
            raise ValueError("For mnist, split must be one of: train, test")
        return MNIST(data_dir, train=(split == "train"), download=True, transform=tfm)

    if dataset == "stl10":
        valid = {"train", "test", "unlabeled", "train+unlabeled"}
        if split not in valid:
            raise ValueError(
                "For stl10, split must be one of: "
                "train, test, unlabeled, train+unlabeled"
            )
        return STL10(data_dir, split=split, download=True, transform=tfm)

    raise ValueError("dataset must be one of: mnist, stl10")


def _parse_layers(raw: str | None, n_blocks: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return list(range(n_blocks))
    out = []
    for token in raw.split(","):
        idx = int(token.strip())
        if idx < 0 or idx >= n_blocks:
            raise ValueError(f"Layer index {idx} out of range [0, {n_blocks - 1}]")
        out.append(idx)
    return sorted(set(out))


def _resolve_checkpoint_path(
    checkpoint: str | None,
    run_name: str | None,
    version: str,
    logs_dir: str,
) -> Path | None:
    """Resolve which checkpoint to load.

    Priority:
    1) --checkpoint explicit path
    2) --run-name + --version from logs
    3) None (randomly initialised model)
    """
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    if not run_name:
        return None

    run_root = Path(logs_dir) / run_name
    if not run_root.exists():
        raise FileNotFoundError(f"Run directory not found: {run_root}")

    versions = sorted(
        [p for p in run_root.glob("version_*") if p.is_dir()],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not versions:
        raise FileNotFoundError(f"No version_* directories found in: {run_root}")

    if version == "latest":
        version_dir = versions[-1]
    else:
        version_dir = run_root / f"version_{version}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {version_dir}")

    ckpt_files = sorted(
        [p for p in (version_dir / "checkpoints").glob("*.ckpt") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpt_files:
        raise FileNotFoundError(
            f"No .ckpt files found in: {version_dir / 'checkpoints'}"
        )

    return ckpt_files[-1]


@torch.no_grad()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Visualize dynamic KNN graphs produced by ViG Grapher layers."
    )
    parser.add_argument("--dataset", choices=["mnist", "stl10"], default="mnist")
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a VigLightningModule checkpoint (.ckpt). "
            "If omitted, uses random weights."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help=(
            "Experiment folder under logs (e.g. mnist-full, vig-stl10). "
            "Used when --checkpoint is not provided."
        ),
    )
    parser.add_argument(
        "--version",
        default="latest",
        help=(
            "Version index under run-name (e.g. 0, 1, 2) or 'latest'. "
            "Used when --run-name is set."
        ),
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Root logs directory containing experiment runs.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Dataset root directory. "
            "Defaults: data/mnist for mnist, data/stl10 for stl10."
        ),
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated Grapher block indices to render. Default: all layers.",
    )
    parser.add_argument("--output-dir", default="outputs/vig_layer_graphs")
    parser.add_argument("--edge-alpha", type=float, default=0.35)
    parser.add_argument("--edge-width", type=float, default=0.6)
    parser.add_argument("--node-size", type=float, default=14.0)
    parser.add_argument(
        "--random-patches",
        type=int,
        default=0,
        help=(
            "If > 0, save an extra overlay showing only edges from N "
            "random query patches to their neighbors."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for selecting query patches.",
    )
    args = parser.parse_args()

    default_data_dir = "data/mnist" if args.dataset == "mnist" else "data/stl10"
    data_dir = args.data_dir or default_data_dir

    resolved_ckpt = _resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        run_name=args.run_name,
        version=args.version,
        logs_dir=args.logs_dir,
    )

    if resolved_ckpt is not None:
        lit = VigLightningModule.load_from_checkpoint(
            str(resolved_ckpt), map_location="cpu"
        )
        LOGGER.info("Loaded checkpoint: %s", resolved_ckpt.resolve())
    else:
        lit = VigLightningModule(
            num_classes=10,
            in_channels=3,
            use_stochastic=False,
            dropout=0.0,
            drop_path_rate=0.0,
        )
        LOGGER.info("No checkpoint selected, using randomly initialized ViG model.")

    lit.eval()
    lit.model.eval()
    lit.model.set_graph_capture(True)

    ds = _build_dataset(args.dataset, args.split, data_dir)

    max_count = min(args.count, len(ds))
    batch = torch.stack([ds[i][0] for i in range(max_count)], dim=0)

    _ = lit.model(batch)
    graphs = lit.model.get_captured_graphs()
    layer_ids = _parse_layers(args.layers, len(graphs))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(max_count):
        image_hwc = _tensor_to_hwc_image(batch[sample_idx])
        sample_dir = out_root / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in layer_ids:
            graph = graphs[layer_idx]
            edge_index = graph["edge_index"]
            hw = graph["hw"]
            if edge_index is None or hw is None:
                continue
            out_path = sample_dir / f"layer_{layer_idx:02d}.png"
            _save_layer_overlay(
                image_hwc=image_hwc,
                edge_index=edge_index[:, sample_idx : sample_idx + 1],
                hw=hw,
                output_path=out_path,
                alpha=args.edge_alpha,
                linewidth=args.edge_width,
                node_size=args.node_size,
            )

            if args.random_patches > 0:
                random_out_path = sample_dir / (
                    f"layer_{layer_idx:02d}_random{args.random_patches}.png"
                )
                local_seed = args.seed + sample_idx * 1000 + layer_idx
                _save_random_patch_neighbors_overlay(
                    image_hwc=image_hwc,
                    edge_index=edge_index[:, sample_idx : sample_idx + 1],
                    hw=hw,
                    output_path=random_out_path,
                    alpha=min(1.0, args.edge_alpha + 0.2),
                    linewidth=max(0.5, args.edge_width),
                    node_size=args.node_size,
                    n_random_patches=args.random_patches,
                    seed=local_seed,
                )

    LOGGER.info("Saved graph visualizations to: %s", out_root.resolve())


if __name__ == "__main__":
    main()
