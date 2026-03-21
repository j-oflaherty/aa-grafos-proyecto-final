# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Train a model
uv run vig-train fit --config configs/<model>/<dataset>.yaml

# Validate / test
uv run vig-train validate --config configs/vig/mnist/full.yaml
uv run vig-train test     --config configs/vig/mnist/full.yaml

# Resume from checkpoint
uv run vig-train fit --config configs/vig/mnist/full.yaml --ckpt_path checkpoints/last.ckpt

# Lint and format
uv run ruff check --fix src/
uv run ruff format src/
```

## Architecture

**Entry point**: `src/vision_gnn/cli.py` — Lightning CLI with `fit`, `validate`, `test`, `predict` subcommands. Loads `configs/default.yaml` first, then merges the user-specified config, then applies CLI overrides.

**Models** (`src/vision_gnn/models/`):
- `vig/` — Vision GNN: flat single-stage. Stem convolutions → dense KNN graph construction → Grapher blocks (MRConv/EdgeConv/SAGE/GIN) + FFN blocks → classifier.
- `pvig/` — Pyramid Vision GNN: 4-stage hierarchical. Each stage downsamples and runs Grapher+FFN blocks at different spatial scales.
- `superpixel_gat/` — Multi-head GAT on SLIC superpixel graphs. Images → SLIC → node features (mean color + xy) → edge list → GAT layers → pooled prediction.
- `resnet/` — ResNet baselines (MiniResNet or torchvision variants).

Each model has a `lightning_module.py` wrapping a backbone with training/val/test steps, AdamW + linear warmup + cosine annealing, and Mixup/CutMix soft-label support.

**Data** (`src/vision_gnn/data/`):
- Standard image datasets (`mnist.py`, `stl10.py`, `imagenet.py`): resize to 224×224, optional augmentation pipeline.
- Superpixel datasets (`mnist_superpixel.py`, `stl10_superpixel.py`, `svhn_superpixel.py`): run SLIC, build block-diagonal batched graphs.
- `augmentation.py`: `AugmentationConfig` dataclass + collate functions for Mixup/CutMix.

**Config system** (`configs/`):
```
configs/default.yaml          ← trainer defaults (TensorBoard logger, ModelCheckpoint, EarlyStopping)
configs/<model>/<dataset>.yaml ← model class_path, init_args, data class_path, overrides
```
Config files use `class_path` / `init_args` YAML keys — Lightning CLI resolves and instantiates these automatically.

**Graph construction** (ViG/PVIG): Dense KNN over image patch tokens; `k` increases linearly across blocks. Optional dilated KNN and stochastic perturbation for regularization.

**Superpixel collation**: Graphs are batched as block-diagonal adjacency matrices with an assignment matrix for graph-level pooling.

## Key files

| Purpose | Path |
|---------|------|
| CLI / entry point | `src/vision_gnn/cli.py` |
| ViG backbone | `src/vision_gnn/models/vig/vig.py` |
| Graph conv layers | `src/vision_gnn/models/vig/modelling/torch_vertex.py` |
| KNN graph builder | `src/vision_gnn/models/vig/modelling/torch_edge.py` |
| GAT model | `src/vision_gnn/models/superpixel_gat/model.py` |
| Superpixel utils | `src/vision_gnn/models/superpixel_gat/graph_utils.py` |
| Augmentation utils | `src/vision_gnn/data/augmentation.py` |
| Default config | `configs/default.yaml` |
