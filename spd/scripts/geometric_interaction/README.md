# Geometric Interaction Strength vs Coactivation

Analyses how much components within the same module geometrically interfere with each other, and whether that geometric overlap correlates with how often they actually fire together.

## Background

In SPD, each component in a module has a U matrix (shape `[C, d_out]`) and V matrix (shape `[d_in, C]`). The weight contribution of component `i` is the rank-one outer product `V_i @ U_i`. When two components are simultaneously active, their U vectors both contribute to the output — so if `|U_i|` and `|U_j|` point in similar directions, component `i`'s activation will bleed into the subspace that component `j` occupies.

This script quantifies that overlap (Geometric Interaction Strength) and compares it against empirical coactivation statistics from harvest data.

## Metrics

### Geometric Interaction Strength (GIS)

The effective output vector of component `i` is `||v_i|| * u_i` — the input activation projects onto `v_i` with magnitude proportional to `||v_i||`, then maps through `u_i`. We define V-norm-scaled vectors `w_i = ||v_i|| * |u_i|` and compute:

```
GIS(i → j) = w_i^T w_j / ||w_i||^2
```

where `w_i = ||v_i|| * |u_i|`.

- `|u_i|` is the element-wise absolute value of U_i (shape `[d_out]`)
- `||v_i||` is the L2 norm of column `i` of V — this scales u_i to reflect the component's actual input-side magnitude
- The absolute value is taken before the inner product because we care about overlap in magnitude, not sign
- **Asymmetric**: GIS(i→j) ≠ GIS(j→i) in general. GIS(i→j) measures "what fraction of component i's effective output energy overlaps with component j". A component with large V-norm will have high GIS toward many smaller components, but not vice versa.
- Range: [0, ≥1] — can exceed 1 because the inner product uses absolute values while the norm uses raw values. Components with very different V-norms can produce large GIS values.

### Coactivation Fraction

```
Coact(i, j) = P(i active | j active) = count_ij / count_j
```

- Loaded from harvest `component_correlations.pt` (computed over the full dataset, typically hundreds of millions of tokens)
- `count_ij` = number of tokens where both i and j had CI above the harvest threshold
- `count_j` = number of tokens where j was active
- **Asymmetric**: P(i active | j active) ≠ P(j active | i active). A rarely-firing component i that only fires when component j fires will have Coact(i,j) close to 1 (i almost always fires when j does), but Coact(j,i) will be low (j fires plenty of times without i).
- Range: [0, 1]

### Alive components

Components are filtered by **activation density** — the fraction of all tokens where a component fires. Components with density below `alive_density_threshold` (default 0.0001) are excluded. The remaining "alive" components are sorted by density in descending order for the heatmap visualizations.

## Data sources

The script does **not** instantiate the full ComponentModel or run any forward passes. It loads two things:

1. **U/V matrices** — extracted directly from the checkpoint state dict by parsing keys like `_components.h-0-mlp-c_fc.U`. This means it works with any model regardless of CI function architecture (e.g. models with global shared transformer CI fns that aren't in the current branch).

2. **Coactivation counts** — from `SPD_OUT_DIR/harvest/<run_id>/<harvest_id>/component_correlations.pt`, which contains:
   - `component_keys`: list of `"module_name:component_idx"` strings
   - `count_i`: per-component firing count across the dataset
   - `count_ij`: pairwise coactivation count matrix (all components × all components)
   - `count_total`: total tokens processed

   The script splits the global `count_ij` matrix into per-module blocks.

## Outputs

All outputs go to `out/<run_id>/` relative to this script (or a custom `output_dir`).

```
out/<run_id>/
├── scatter/                    # One scatter plot per module
│   ├── h_0_mlp_c_fc.png
│   ├── h_0_mlp_down_proj.png
│   └── ...
├── heatmaps/
│   ├── gis.png                 # GIS matrices (Reds), one heatmap per module stacked vertically
│   ├── coactivation.png        # Coactivation matrices (Blues)
│   └── gis_x_coactivation.png  # Element-wise product GIS × Coact (Purples)
└── data.pt                     # Raw tensors for further analysis
```

### Scatter plots

Each scatter plot shows all (i, j) pairs from the alive-component submatrix (full matrix, both directions — not just upper triangle). X = GIS(i→j), Y = Coact(i, j). Each module gets its own plot.

### Heatmaps

Components are sorted by activation density (most active top-left). Three heatmap grids:
- **GIS** — are geometrically similar components clustered together?
- **Coactivation** — which components tend to fire together?
- **GIS × Coactivation** — highlights pairs that are both geometrically overlapping AND frequently coactive (the most "interfering" pairs)

### data.pt

Contains `gis_matrices`, `coactivation_fractions`, `activation_density` (all `dict[str, Tensor]` keyed by module name), plus the config and run_id. Load with `torch.load(path)`.

## Usage

```bash
# Via config file
python spd/scripts/geometric_interaction/geometric_interaction.py spd/scripts/geometric_interaction/config.yaml

# Via CLI args
python spd/scripts/geometric_interaction/geometric_interaction.py --model_path="wandb:goodfire/spd/runs/s-55ea3f9b"

# With specific harvest and output dir
python spd/scripts/geometric_interaction/geometric_interaction.py \
    --model_path="wandb:goodfire/spd/runs/s-55ea3f9b" \
    --harvest_id="h-20260319_121635" \
    --output_dir="/tmp/gis_analysis"
```

## Prerequisites

- The model checkpoint must be cached locally (download happens automatically for wandb paths)
- Harvest must have been run on the model (`spd-harvest`) with correlation data present
- No GPU required — everything runs on CPU
