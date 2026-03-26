from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torchvision import transforms
from torchvision.datasets import MNIST, STL10


def _to_hwc(image):
    """Convert a CHW tensor to HWC numpy array in [0, 1]."""
    array = image.detach().cpu().numpy().astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={array.shape}")
    return np.transpose(array, (1, 2, 0))


def _sample_indices(size: int, count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    count = min(count, size)
    return rng.choice(size, size=count, replace=False).tolist()


def _label_to_text(label, class_names: list[str] | None) -> str:
    idx = int(label)
    if class_names is None or idx < 0 or idx >= len(class_names):
        return str(idx)
    return class_names[idx]


def _save_grid(
    dataset,
    indices: list[int],
    rows: int,
    cols: int,
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required. Install with: uv sync --group dev"
        ) from exc

    class_names = getattr(dataset, "classes", None)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    axes_flat = np.array(axes).reshape(-1)

    for ax, idx in zip(axes_flat, indices, strict=False):
        image, label = dataset[idx]
        image_hwc = _to_hwc(image)
        if image_hwc.shape[2] == 1:
            ax.imshow(image_hwc[:, :, 0], cmap="gray")
        else:
            ax.imshow(image_hwc)
        ax.set_title(_label_to_text(label, class_names), fontsize=8)
        ax.axis("off")

    # If there are extra axes (unlikely for 3x3), hide them.
    for ax in axes_flat[len(indices) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 3x3 sample grids for MNIST and STL10."
    )
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/example_grids")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mnist-split", choices=["train", "test"], default="test")
    parser.add_argument(
        "--stl10-split",
        choices=["train", "test", "unlabeled", "train+unlabeled"],
        default="test",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    total = args.rows * args.cols
    transform = transforms.ToTensor()

    mnist = MNIST(
        root=str(Path(args.data_root) / "mnist"),
        train=(args.mnist_split == "train"),
        download=True,
        transform=transform,
    )
    stl10 = STL10(
        root=str(Path(args.data_root) / "stl10"),
        split=args.stl10_split,
        download=True,
        transform=transform,
    )

    mnist_indices = _sample_indices(len(mnist), total, seed=args.seed)
    stl10_indices = _sample_indices(len(stl10), total, seed=args.seed + 1)

    output_dir = Path(args.output_dir)
    _save_grid(
        dataset=mnist,
        indices=mnist_indices,
        rows=args.rows,
        cols=args.cols,
        output_path=output_dir / "mnist_grid_3x3.png",
        title=f"MNIST ({args.mnist_split}) - 3x3 samples",
    )
    _save_grid(
        dataset=stl10,
        indices=stl10_indices,
        rows=args.rows,
        cols=args.cols,
        output_path=output_dir / "stl10_grid_3x3.png",
        title=f"STL10 ({args.stl10_split}) - 3x3 samples",
    )

    print(f"Saved: {output_dir / 'mnist_grid_3x3.png'}")
    print(f"Saved: {output_dir / 'stl10_grid_3x3.png'}")


if __name__ == "__main__":
    main()
