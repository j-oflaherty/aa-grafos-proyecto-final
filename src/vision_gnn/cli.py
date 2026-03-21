"""Lightning CLI entry point for training runs.

Usage examples
--------------
# ViG on MNIST
vig-train fit --config configs/vig/mnist/full.yaml

# ResNet baseline on MNIST
vig-train fit --config configs/resnet/mnist.yaml

# Override batch size on the fly
vig-train fit --config configs/resnet/mnist.yaml --data.batch_size 128

# Resume from checkpoint
vig-train fit --config configs/resnet/mnist.yaml --ckpt_path checkpoints/last.ckpt
"""

import torch
import lightning as L
from lightning.pytorch.cli import LightningCLI

torch.set_float32_matmul_precision("high")

_SUBCOMMANDS = {"fit", "validate", "test", "predict"}


def main() -> None:
    LightningCLI(
        model_class=L.LightningModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={
            cmd: {"default_config_files": ["configs/default.yaml"]}
            for cmd in _SUBCOMMANDS
        },
    )


if __name__ == "__main__":
    main()
