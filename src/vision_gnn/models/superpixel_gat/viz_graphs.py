from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.segmentation import mark_boundaries
from torchvision.datasets import MNIST, STL10

from .graph_utils import get_graph_from_image, get_superpixel_graph_from_image


def _to_hwc_float32(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert image to float32 HWC in [0, 1]."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 2:
        image = image[:, :, None]
    elif image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def _save_original_image(image: np.ndarray | torch.Tensor, output_path: Path) -> Path:
    """Save original image without overlays."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for graph visualization. "
            "Install dev deps with: uv sync --group dev"
        ) from exc

    image_hwc = _to_hwc_float32(image)
    if image_hwc.shape[-1] == 1:
        image_to_save = image_hwc[:, :, 0]
    else:
        image_to_save = image_hwc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image_to_save)
    return output_path


def _save_superpixel_only_image(
    image: np.ndarray | torch.Tensor,
    output_path: Path,
    desired_nodes: int = 75,
    compactness: float = 0.1,
    boundary_mode: str = "inner",
) -> Path:
    """Save image with superpixel boundaries only (no graph edges or nodes)."""
    image_hwc = _to_hwc_float32(image)
    superpixel_graph = get_superpixel_graph_from_image(
        image_hwc,
        desired_nodes=desired_nodes,
        compactness=compactness,
    )

    canvas = image_hwc
    if image_hwc.shape[2] == 1:
        canvas = np.repeat(image_hwc, 3, axis=2)

    overlay = mark_boundaries(
        canvas,
        superpixel_graph.segments.numpy(),
        color=(1, 1, 0),
        mode=boundary_mode,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for graph visualization. "
            "Install dev deps with: uv sync --group dev"
        ) from exc

    plt.imsave(output_path, overlay)
    return output_path


def save_graph_overlay(
    image: np.ndarray | torch.Tensor,
    output_path: str | Path,
    desired_nodes: int = 75,
    compactness: float = 0.1,
    draw_boundaries: bool = True,
    edge_alpha: float = 0.5,
    edge_width: float = 0.8,
    edge_color: str = "deepskyblue",
    node_size: float = 10.0,
    node_color: str = "red",
    node_edge_color: str = "white",
    node_edge_width: float = 0.6,
    boundary_mode: str = "inner",
) -> Path:
    """Build a SLIC graph from one image and save an overlay figure."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for graph visualization. Install dev deps with: uv sync --group dev"
        ) from exc

    image_hwc = _to_hwc_float32(image)
    h, edges = get_graph_from_image(
        image_hwc,
        desired_nodes=desired_nodes,
        compactness=compactness,
    )

    # Build segments once for optional boundary rendering.
    superpixel_graph = get_superpixel_graph_from_image(
        image_hwc,
        desired_nodes=desired_nodes,
        compactness=compactness,
    )

    xy = h[:, -2:].numpy()
    img_h, img_w = image_hwc.shape[:2]
    xs = xy[:, 0] * img_w
    ys = xy[:, 1] * img_h
    edges_np = edges.numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    canvas = image_hwc
    if image_hwc.shape[2] == 1:
        canvas = np.repeat(image_hwc, 3, axis=2)

    if draw_boundaries:
        canvas = mark_boundaries(
            canvas,
            superpixel_graph.segments.numpy(),
            color=(1, 1, 0),
            mode=boundary_mode,
        )

    ax.imshow(canvas)

    for src, tgt in edges_np:
        if src < tgt:
            ax.plot(
                [xs[src], xs[tgt]],
                [ys[src], ys[tgt]],
                color=edge_color,
                alpha=edge_alpha,
                linewidth=edge_width,
            )

    ax.scatter(
        xs,
        ys,
        s=node_size,
        c=node_color,
        edgecolors=node_edge_color,
        linewidths=node_edge_width,
        zorder=3,
    )
    ax.set_title(f"nodes={h.shape[0]} edges={edges.shape[0]}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_graph_batch(
    images: np.ndarray | torch.Tensor,
    output_dir: str | Path,
    desired_nodes: int = 75,
    compactness: float = 0.1,
    draw_boundaries: bool = True,
    edge_alpha: float = 0.5,
    edge_width: float = 0.8,
    edge_color: str = "deepskyblue",
    node_size: float = 10.0,
    node_color: str = "red",
    node_edge_color: str = "white",
    node_edge_width: float = 0.6,
    boundary_mode: str = "inner",
    prefix: str = "graph",
    save_originals: bool = False,
    save_superpixel_only: bool = False,
) -> list[Path]:
    """Save graph overlays for a batch of images.

    Args:
        images: Batch with shape (B, H, W), (B, H, W, C) or (B, C, H, W).
        output_dir: Directory where PNG files are saved.
        desired_nodes: Target number of SLIC superpixels.
        prefix: Prefix for output file names.
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    images = np.asarray(images)
    if images.ndim not in (3, 4):
        raise ValueError("Expected a batch with 3 or 4 dimensions.")

    # (B, H, W) -> (B, H, W, 1)
    if images.ndim == 3:
        images = images[:, :, :, None]

    saved_paths: list[Path] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    originals_dir = output_dir / "originals"
    superpixels_dir = output_dir / "superpixels"

    for idx in range(images.shape[0]):
        out_path = output_dir / f"{prefix}_{idx:04d}.png"
        if save_originals:
            original_path = originals_dir / f"{prefix}_{idx:04d}_original.png"
            _save_original_image(images[idx], original_path)
        if save_superpixel_only:
            superpixel_path = superpixels_dir / f"{prefix}_{idx:04d}_superpixel.png"
            _save_superpixel_only_image(
                images[idx],
                superpixel_path,
                desired_nodes=desired_nodes,
                compactness=compactness,
                boundary_mode=boundary_mode,
            )
        saved_paths.append(
            save_graph_overlay(
                images[idx],
                out_path,
                desired_nodes=desired_nodes,
                compactness=compactness,
                draw_boundaries=draw_boundaries,
                edge_alpha=edge_alpha,
                edge_width=edge_width,
                edge_color=edge_color,
                node_size=node_size,
                node_color=node_color,
                node_edge_color=node_edge_color,
                node_edge_width=node_edge_width,
                boundary_mode=boundary_mode,
            )
        )

    return saved_paths


def _load_dataset_batch(
    dataset_name: str,
    split: str,
    count: int,
    data_dir: str,
) -> np.ndarray:
    """Load a batch of images from MNIST or STL10."""
    dataset_name = dataset_name.lower()
    split = split.lower()

    if dataset_name == "mnist":
        ds = MNIST(data_dir, train=(split == "train"), download=True)
        count = min(count, len(ds))
        imgs = [
            np.array(ds[i][0], dtype=np.float32)[:, :, None] / 255.0
            for i in range(count)
        ]
        return np.stack(imgs, axis=0)

    if dataset_name == "stl10":
        valid_splits = {"train", "test", "unlabeled", "train+unlabeled"}
        if split not in valid_splits:
            raise ValueError(
                "For stl10, split must be one of: train, test, unlabeled, train+unlabeled"
            )

        ds = STL10(data_dir, split=split, download=True)
        count = min(count, len(ds))
        imgs = []
        for i in range(count):
            sample = ds[i]
            image = sample[0] if isinstance(sample, tuple) else sample
            imgs.append(np.array(image, dtype=np.float32) / 255.0)
        return np.stack(imgs, axis=0)

    raise ValueError("dataset must be one of: mnist, stl10")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize SLIC superpixel graphs over input images."
    )
    parser.add_argument("--dataset", choices=["mnist", "stl10"], default="mnist")
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (mnist: train/test, stl10: train/test/unlabeled/train+unlabeled).",
    )
    parser.add_argument(
        "--count", type=int, default=8, help="Number of images to visualize."
    )
    parser.add_argument("--desired-nodes", type=int, default=75)
    parser.add_argument("--compactness", type=float, default=0.1)
    parser.add_argument("--edge-alpha", type=float, default=0.75)
    parser.add_argument("--edge-width", type=float, default=1.2)
    parser.add_argument("--edge-color", default="deepskyblue")
    parser.add_argument("--node-size", type=float, default=26.0)
    parser.add_argument("--node-color", default="red")
    parser.add_argument("--node-edge-color", default="white")
    parser.add_argument("--node-edge-width", type=float, default=0.8)
    parser.add_argument(
        "--boundary-mode",
        choices=["thick", "inner", "outer", "subpixel"],
        default="inner",
        help="Superpixel boundary style. Use 'inner' for thinner boundaries.",
    )
    parser.add_argument(
        "--no-boundaries",
        action="store_true",
        help="Disable drawing SLIC boundaries on top of images.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root directory. Defaults: data/mnist for mnist, data/stl10 for stl10.",
    )
    parser.add_argument("--output-dir", default="outputs/graph_viz")
    parser.add_argument("--prefix", default="graph")
    parser.add_argument(
        "--save-originals",
        action="store_true",
        help="Save original input images in output_dir/originals.",
    )
    parser.add_argument(
        "--save-superpixel-only",
        action="store_true",
        help=(
            "Save images with only superpixel boundaries (no graph edges/nodes) "
            "in output_dir/superpixels."
        ),
    )
    args = parser.parse_args()

    default_data_dir = "data/mnist" if args.dataset == "mnist" else "data/stl10"
    data_dir = args.data_dir or default_data_dir

    images = _load_dataset_batch(
        dataset_name=args.dataset,
        split=args.split,
        count=args.count,
        data_dir=data_dir,
    )

    saved = save_graph_batch(
        images=images,
        output_dir=args.output_dir,
        desired_nodes=args.desired_nodes,
        compactness=args.compactness,
        draw_boundaries=not args.no_boundaries,
        edge_alpha=args.edge_alpha,
        edge_width=args.edge_width,
        edge_color=args.edge_color,
        node_size=args.node_size,
        node_color=args.node_color,
        node_edge_color=args.node_edge_color,
        node_edge_width=args.node_edge_width,
        boundary_mode=args.boundary_mode,
        prefix=args.prefix,
        save_originals=args.save_originals,
        save_superpixel_only=args.save_superpixel_only,
    )
    print(f"Saved {len(saved)} graph overlays to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
