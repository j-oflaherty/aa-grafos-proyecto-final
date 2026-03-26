# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parent / "informe/images"
RESULTS_DIR = ROOT / "results"
TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13
LEGEND_TITLE_FONTSIZE = 14


def _prepare_curves(curves: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for df in curves.values():
        # Convert absolute UNIX time to elapsed minutes since each run started.
        df["Relative time (min)"] = (df["Wall time"] - df["Wall time"].iloc[0]) / 60.0
    return curves


def load_mnist_curves() -> dict[str, pd.DataFrame]:
    curves = {
        "ViG": pd.read_csv(RESULTS_DIR / "mnist_vig.csv"),
        "Superpixel GAT": pd.read_csv(RESULTS_DIR / "gat-mnist.csv"),
        "ResNet": pd.read_csv(RESULTS_DIR / "resnet-mnist.csv"),
    }
    return _prepare_curves(curves)


def load_stl10_curves() -> dict[str, pd.DataFrame]:
    curves = {
        "ViG": pd.read_csv(RESULTS_DIR / "vig-ti-stl10.csv"),
        "PyramidViG": pd.read_csv(RESULTS_DIR / "pvig-ti-stl10.csv"),
        "Superpixel GAT": pd.read_csv(RESULTS_DIR / "gat-stl10.csv"),
        "ResNet18": pd.read_csv(RESULTS_DIR / "resnet18-stl10.csv"),
    }
    return _prepare_curves(curves)


def load_stl10_losses() -> dict[str, pd.DataFrame]:
    return {
        "ViG": pd.read_csv(RESULTS_DIR / "vig-ti-stl10_loss.csv"),
        "PyramidViG": pd.read_csv(RESULTS_DIR / "pvig-ti-stl10_loss.csv"),
        "Superpixel GAT": pd.read_csv(RESULTS_DIR / "gat-stl10_loss.csv"),
        "ResNet18": pd.read_csv(RESULTS_DIR / "resnet18-stl10_loss.csv"),
    }


def plot_against_step(
    curves: dict[str, pd.DataFrame], dataset_name: str, output_name: str
) -> None:
    plt.figure(figsize=(10, 5))
    for label, df in curves.items():
        plt.plot(
            df["Step"],
            df["Value"],
            marker="o",
            linewidth=2,
            markersize=3,
            label=label,
        )

    plt.title(
        f"Exactitud Top-1 en {dataset_name} (eje x: pasos)",
        fontsize=TITLE_FONTSIZE,
    )
    plt.xlabel("Pasos", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Exactitud Top-1", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(alpha=0.25)
    plt.legend(
        title="Modelo",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=180)


def plot_against_relative_time(
    curves: dict[str, pd.DataFrame], dataset_name: str, output_name: str
) -> None:
    plt.figure(figsize=(10, 5))
    for label, df in curves.items():
        plt.plot(
            df["Relative time (min)"],
            df["Value"],
            marker="o",
            linewidth=2,
            markersize=3,
            label=label,
        )

    plt.title(
        f"Exactitud Top-1 en {dataset_name} (eje x: tiempo relativo)",
        fontsize=TITLE_FONTSIZE,
    )
    plt.xlabel("Tiempo relativo (minutos)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Exactitud Top-1", fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(alpha=0.25)
    plt.legend(
        title="Modelo",
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=180)


def _loss_at_step(loss_df: pd.DataFrame, step: int) -> tuple[float, int]:
    same_step = loss_df[loss_df["Step"] == step]
    if not same_step.empty:
        return float(same_step["Value"].iloc[0]), step

    nearest_idx = (loss_df["Step"] - step).abs().idxmin()
    nearest_step = int(loss_df.loc[nearest_idx, "Step"])
    loss_value = float(loss_df.loc[nearest_idx, "Value"])
    return loss_value, nearest_step


def print_max_top1_with_loss(
    curves: dict[str, pd.DataFrame],
    losses: dict[str, pd.DataFrame],
    dataset_name: str,
) -> None:
    print(f"Max Top-1 y loss asociada por modelo en {dataset_name}:")
    for model_name, acc_df in curves.items():
        max_idx = acc_df["Value"].idxmax()
        max_top1 = float(acc_df.loc[max_idx, "Value"])
        step_at_max = int(acc_df.loc[max_idx, "Step"])
        loss_value, matched_step = _loss_at_step(losses[model_name], step_at_max)
        print(
            f"- {model_name}: max Top-1={max_top1:.4f} "
            f"(step={step_at_max}), loss={loss_value:.4f} (step={matched_step})"
        )


# %%
mnist_curves = load_mnist_curves()

# %%
plot_against_step(
    mnist_curves,
    dataset_name="MNIST",
    output_name="training_curves_step.pdf",
)

# %%
plot_against_relative_time(
    mnist_curves,
    dataset_name="MNIST",
    output_name="training_curves_relative_time.pdf",
)

# %%
stl10_curves = load_stl10_curves()
stl10_losses = load_stl10_losses()

# %%
plot_against_step(
    stl10_curves,
    dataset_name="STL10",
    output_name="training_curves_step_stl10.pdf",
)

# %%
plot_against_relative_time(
    stl10_curves,
    dataset_name="STL10",
    output_name="training_curves_relative_time_stl10.pdf",
)

# %%
print_max_top1_with_loss(
    stl10_curves,
    stl10_losses,
    dataset_name="STL10",
)

# %%
