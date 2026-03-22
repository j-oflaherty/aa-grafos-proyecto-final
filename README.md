# AA Grafos Proyecto Final

## Setup

Install dependencies:

```bash
uv sync --group dev
```

The visualization scripts save PNG files under the `outputs/` folder.

## Graph Visualization

This project supports graph visualization for two models:

1. ViG dynamic KNN graphs from GNN layers
2. Superpixel-GAT input SLIC graphs

---

## 1) ViG Layer Graphs

Entry point:

```bash
uv run vig-viz-graphs --help
```

### A) Visualize all node-to-node edges (full layer graph)

MNIST trained run:

```bash
uv run vig-viz-graphs \
	--dataset mnist \
	--split test \
	--count 10 \
	--layers 0,5,7 \
	--run-name mnist-full \
	--version latest \
	--output-dir outputs/vig_layer_graphs_mnist
```

STL10 trained run:

```bash
uv run vig-viz-graphs \
	--dataset stl10 \
	--split test \
	--count 10 \
	--layers 0,5,7 \
	--run-name vig-stl10 \
	--version latest \
	--output-dir outputs/vig_layer_graphs_stl10
```

### B) Visualize random query patches and their neighbors

This keeps the full graph output and adds extra images named like:

- `layer_00_random4.png`

MNIST example (4 random patches):

```bash
uv run vig-viz-graphs \
	--dataset mnist \
	--split test \
	--count 10 \
	--layers 0,5,7 \
	--run-name mnist-full \
	--version latest \
	--random-patches 4 \
	--seed 7 \
	--output-dir outputs/vig_layer_graphs_mnist_random
```

STL10 example (10 random patches):

```bash
uv run vig-viz-graphs \
	--dataset stl10 \
	--split test \
	--count 10 \
	--layers 0,5,7 \
	--run-name vig-stl10 \
	--version latest \
	--random-patches 10 \
	--seed 7 \
	--output-dir outputs/vig_layer_graphs_stl10_random
```

### Model selection options (ViG)

- Use `--checkpoint path/to/model.ckpt` to select an exact checkpoint.
- Or use `--run-name` + `--version` to auto-pick the newest checkpoint under `logs/<run-name>/version_<n>/checkpoints/`.
- If neither is passed, a randomly initialized ViG model is used.

---

## 2) Superpixel-GAT Input Graphs

Entry point:

```bash
uv run python -m vision_gnn.models.superpixel_gat.viz_graphs --help
```

MNIST example:

```bash
uv run python -m vision_gnn.models.superpixel_gat.viz_graphs \
	--dataset mnist \
	--split test \
	--count 10 \
	--desired-nodes 75 \
	--output-dir outputs/superpixel_mnist_graphs
```

STL10 example:

```bash
uv run python -m vision_gnn.models.superpixel_gat.viz_graphs \
	--dataset stl10 \
	--split test \
	--count 10 \
	--desired-nodes 75 \
	--output-dir outputs/superpixel_stl10_graphs
```

This script visualizes the SLIC superpixel graph built from each input image.
