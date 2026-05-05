# QK Component Static Interaction Strengths

Analyzes how PD components in query and key projections interact to produce attention patterns, using only learned weights (no activations needed at inference time). The central output is a **standardized static interaction strength** metric that quantifies, for each (query component, key component) pair, how strongly they contribute to the attention logit at each relative position offset.

## Motivation

In a PD attention layer, the query and key projections are each split into components. Each component has an input-side vector V (which determines *when* it activates) and an output-side vector U (which determines *what* it writes to the query/key space). By taking dot products between Q-component U vectors and K-component U vectors -- accounting for RoPE rotations -- we can measure pairwise static interaction strengths purely from weights.

This is useful for understanding:
- Which component pairs have strong interaction at specific relative positions
- Whether certain components specialize in local vs. distant attention
- How individual heads differ in their component interaction structure

## How the Metric Is Calculated

### Step 1: Filter alive components

For each layer, Q and K components are filtered to those with mean causal importance (CI) above a threshold (default 0.01), then sorted by CI descending.

### Step 2: Build scaled U vectors

Each component `c` in a `LinearComponents` module has:
- `V[:, c]` -- input projection vector (d_in,)
- `U[c]` -- output projection vector (d_out,)

The output vector is scaled to incorporate both the input magnitude and activation sign:

```
U_scaled[c] = U[c] * ||V[:, c]||_2 * sign(mean_component_activation[c])
```

The V-norm scaling accounts for the fact that components with larger input norms will produce proportionally larger activations. The sign correction ensures the weight-only dot product reflects the typical sign of the actual (data-dependent) contribution -- without it, a component whose activations are typically negative would appear to contribute in the wrong direction.

After scaling, U vectors are reshaped to separate attention heads: `(n_components, n_heads, head_dim)`. For GQA models, K vectors are expanded to match the number of Q heads.

### Step 3: Compute RoPE-aware dot products

Standard attention applies Rotary Position Embeddings (RoPE) before the Q-K dot product, making the result depend on the relative position offset between query and key. The script decomposes this into offset-independent coefficients:

For query component `q` and key component `k` in head `h`, split the head dimension into first-half and second-half (non-adjacent-pairs RoPE layout):

```
q1, q2 = U_q[q, h, :half], U_q[q, h, half:]
k1, k2 = U_k[k, h, :half], U_k[k, h, half:]

A_d = q1_d * k1_d + q2_d * k2_d    (content-aligned coefficient)
B_d = q1_d * k2_d - q2_d * k1_d    (cross-half coefficient)
```

Then the static interaction strength at relative offset delta is:

```
W(delta) = sum_d [ A_d * cos(delta * theta_d) + B_d * sin(delta * theta_d) ]
```

where `theta_d` are the RoPE frequencies. At delta=0 only the A term contributes (since sin(0)=0), while at nonzero offsets both terms contribute.

This is computed by `compute_qk_rope_coefficients` and `evaluate_qk_at_offsets` in `param_decomp/scripts/rope_aware_qk.py`.

### Step 4: Z-score standardization (per head)

The raw dot products have different scales across heads. To make heads comparable, interaction strengths are z-scored independently per head:

```
mu_h  = mean over all (delta, q, k) of W_h(delta, q, k)
sigma_h = std over all (delta, q, k) of W_h(delta, q, k)

StandardizedStaticInteractionStrength_h(delta, q, k) = (W_h(delta, q, k) - mu_h) / sigma_h
```

The final output tensor has shape `(n_offsets, n_q_heads, n_q_alive, n_k_alive)`. A value of +2.0 means "this pair's interaction strength is 2 standard deviations above the head's average" -- a strong positive attention signal at that offset.

## Usage

```bash
python -m param_decomp.scripts.plot_qk_c_attention_contributions.plot_qk_c_attention_contributions \
    wandb:goodfire/spd/runs/<run_id> \
    [--offsets 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] \
    [--top_n_pairs 10] \
    [--plots all] \
    [--recompute]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `wandb_path` | (required) | WandB run path, e.g. `wandb:goodfire/spd/runs/s-55ea3f9b` |
| `offsets` | `0..16` | Relative position offsets to evaluate |
| `top_n_pairs` | `10` | Number of top (q, k) pairs to highlight in line plots |
| `plots` | `all` | Comma-separated subset of plot types (see below), or `all` |
| `recompute` | `False` | Force recomputation even if cached results exist |

**Prerequisites:** The run must have harvest data available (run `pd-harvest` first). The model must be a `LlamaSimpleMLP` with non-adjacent-pairs RoPE layout.

## Plot Types

### `heatmaps`
One image per offset. Grid shows mean-across-heads plus individual head heatmaps. Rows = Q components (by CI rank), columns = K components. Colormap: RdBu_r (red = positive, blue = negative).

**File:** `layer{L}_qk_attention_contributions_offset{D}.png`

### `heatmaps_per_head`
One image per head (plus Mean). Grid shows all offsets as sub-heatmaps.

**File:** `heatmap_offsets_per_head/layer{L}_qk_attention_{mean|h0|...}.png`

### `scatter`
Per-head interaction strength (y) vs mean-across-heads interaction strength (x). Each head uses a distinct colormap; darker shades indicate larger offsets.

**File:** `scatter_head_vs_sum/layer{L}_head_vs_sum_scatter.png`

### `diffs`
Heatmaps of `W(delta) - W(0)` for each nonzero offset, isolating the position-dependent component of attention. Colormap: PiYG.

**File:** `diffs/layer{L}_qk_attention_diff_offset{D}.png`

### `lines`
Line plot of mean-across-heads interaction strength vs offset for each (q, k) pair. Top-N pairs (ranked by peak absolute mean-across-heads interaction strength over offsets) highlighted in color, rest in faint gray.

**File:** `lines/layer{L}_qk_pair_lines.png`

### `lines_per_head`
Line plot of per-head interaction strength vs offset for the top-N (head, q, k) triples, ranked by peak absolute interaction strength.

**File:** `lines_per_head/layer{L}_qk_pair_lines_per_head.png`

### `lines_single_head`
Grid with one subplot per head (3 columns, rows adapt to head count). Global top-K pairs (ranked by peak absolute interaction strength across all heads and offsets) in color (consistent across subplots), remaining pairs in gray, sum of all pairs as a thick black line.

**File:** `lines_single_head/layer{L}_qk_pair_lines_grid.png`

### `lines_combined`
Combined layout: top row has mean-across-heads plot alongside a legend panel; lower rows show individual heads in a 3x2 grid (assumes 6 query heads). Global top-N pairs (ranked by peak absolute interaction strength across all heads and offsets) share consistent colors across all subplots. All subplots share y-axis limits.

**File:** `lines_combined/layer{L}_qk_pair_lines_combined.png`

## Output Structure

```
out/<run_id>/
  cache/                         # Cached computation (reused across plot runs)
    layer0.npz
    layer1.npz
    ...
  layer*_qk_attention_*.png      # heatmaps
  heatmap_offsets_per_head/      # heatmaps_per_head
  scatter_head_vs_sum/           # scatter
  diffs/                         # diffs
  lines/                         # lines
  lines_per_head/                # lines_per_head
  lines_single_head/             # lines_single_head
  lines_combined/                # lines_combined
```

The cache stores the full `(n_offsets, n_q_heads, n_q_alive, n_k_alive)` tensor per layer. If offsets change, the cache is invalidated and all layers are recomputed. Use `--recompute` to force a fresh computation.

## Source Files

- `plot_qk_c_attention_contributions.py` -- Main script (computation, caching, all plot types)
- `../rope_aware_qk.py` -- RoPE coefficient decomposition (`compute_qk_rope_coefficients`, `evaluate_qk_at_offsets`)
