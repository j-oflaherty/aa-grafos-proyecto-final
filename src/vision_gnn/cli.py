"""Lightning CLI entry point for ViG training runs.

Usage examples
--------------
# MNIST with defaults (TensorBoard logs → logs/vig)
vig-train fit --config configs/mnist.yaml

# Override batch size and max epochs on the fly
vig-train fit --config configs/mnist.yaml --data.batch_size 128 --trainer.max_epochs 50

# ImageNet
vig-train fit --config configs/imagenet.yaml

# Resume from checkpoint
vig-train fit --config configs/mnist.yaml --ckpt_path checkpoints/last.ckpt
"""

from lightning.pytorch.cli import LightningCLI

from vision_gnn.lightning_module import VigLightningModule

_SUBCOMMANDS = {"fit", "validate", "test", "predict"}


def main() -> None:
    LightningCLI(
        model_class=VigLightningModule,
        subclass_mode_data=True,
        parser_kwargs={
            cmd: {"default_config_files": ["configs/default.yaml"]}
            for cmd in _SUBCOMMANDS
        },
    )


if __name__ == "__main__":
    main()
