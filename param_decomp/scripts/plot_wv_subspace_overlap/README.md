# OV Subspace Overlap Analysis

Measures how much the OV circuits of different attention heads overlap in the residual stream — do they read from the same input subspaces, write to the same output subspaces, or implement similar linear maps? Optionally weights the analysis by which subspaces the data actually uses.

## Background

Each attention head `h` has an OV circuit `W_OV^h = W_O^h @ W_V^h` (shape `d_model x d_model`, rank `d_head`) that maps from the residual stream it reads to the residual stream it writes to. The attention pattern (from the QK circuit) determines how much influence each previous token's OV output has on the current timestep.

We want to know:
1. **Read overlap**: Do two heads amplify the same input directions? (What they attend to in the residual stream)
2. **Write overlap**: Do two heads write to the same output directions? (What they contribute back)
3. **Raw similarity**: Are the full linear maps similar? (Captures whether singular vectors are paired the same way, even if the read/write subspaces overlap)

## Scripts

### `plot_wv_subspace_overlap.py` — Main analysis

Computes pairwise head overlap and produces heatmap plots. Also saves intermediate `.npy` files for the semantic analysis script.

```bash
# Full run (100 batches, all individual + combined plots)
python -m param_decomp.scripts.plot_wv_subspace_overlap.plot_wv_subspace_overlap \
    wandb:goodfire/spd/runs/<run_id> --layer 1

# Fast development run (10 batches, skips individual variant plots)
python -m param_decomp.scripts.plot_wv_subspace_overlap.plot_wv_subspace_overlap \
    wandb:goodfire/spd/runs/<run_id> --layer 1 --fast

# With K-filtered data weighting (filter activations to tokens where
# K component 329 is causally important — the attended-to positions)
python -m param_decomp.scripts.plot_wv_subspace_overlap.plot_wv_subspace_overlap \
    wandb:goodfire/spd/runs/<run_id> --layer 1 --k_filter 329
```

### `analyze_ov_subspace_semantics.py` — Semantic analysis

Identifies which PD V and O components most align with each head's OV subspace. Requires `plot_wv_subspace_overlap.py` to have been run first (uses saved `.npy` files). Auto-discovers any K-filtered SVD data and produces a variant for each.

```bash
python -m param_decomp.scripts.plot_wv_subspace_overlap.analyze_ov_subspace_semantics \
    wandb:goodfire/spd/runs/<run_id> --layer 1
```

## How the analysis works

### Step 1: Collect activations

Run the pretrained target model (not the PD model) on dataset batches and collect the post-RMSNorm residual stream activations at the target layer. These are the inputs that the attention layer sees. Shape: `(total_tokens, d_model)`.

### Step 2: Compute data SVDs

Two SVDs are computed on the activation matrix:

**Mean-centered SVD (PCA)** — used for all paper figures and data-weighted analysis:

```
X_centered = X - mean(X)
X_centered = U_bar @ diag(S_bar) @ Z_bar^T
```

Here `Z_bar^T` has shape `(d_model, d_model)` with **rows** as principal axes of variation (following `torch.linalg.svd` convention). `S_bar` contains singular values proportional to standard deviation along each axis. This is what the code stores as `var_svectors` (= `Z_bar^T`) and `var_singular_values` (= `S_bar`).

**Non-centered SVD** — used only for legacy individual variant plots (skipped in `--fast` mode):

```
X = U @ diag(S) @ Z^T
```

Stored as `data_svectors` / `singular_values`. The first singular vector predominantly captures the mean direction rather than variance.

### Step 3: Extract per-head W_OV

For each head `h`:
- `W_V^h`: rows `[h*d_head : (h+1)*d_head]` of `v_proj.weight` — shape `(d_head, d_model)`
- `W_O^h`: columns `[h*d_head : (h+1)*d_head]` of `o_proj.weight` — shape `(d_model, d_head)`
- `W_OV^h = W_O^h @ W_V^h` — shape `(d_model, d_model)`, rank `d_head`

### Step 4: Compute Frobenius cosine similarity (FCS)

Three types of pairwise comparison, each answering a different question:

**Read overlap** — "Do these heads amplify the same input directions?"

Gram matrix: `M_read^h = W_OV^{hT} W_OV^h`. For any input direction `x`, `x^T M_read^h x = ||W_OV^h x||^2`.

FCS between two heads:
```
FCS(a, b) = tr(M_read^a @ M_read^b) / (||M_read^a||_F * ||M_read^b||_F)
```

**Write overlap** — "Do these heads write to the same output directions?"

Gram matrix: `M_write^h = W_OV^h @ W_OV^{hT}`. For any output direction `y`, `y^T M_write^h y = ||W_OV^{hT} y||^2`.

**Raw FCS** — "Are the full linear maps similar?"

Flattens each `W_OV^h` as a vector in `R^{d_model^2}` and computes cosine similarity directly. This is stricter: two heads can have identical read and write subspaces but different raw FCS if their singular vectors are paired differently.

### Step 5: Data weighting

The raw FCS treats all residual stream directions equally. But the network doesn't use every direction equally. Data weighting ensures overlap in heavily-used subspaces counts more.

To apply data weighting, right-multiply each `W_OV^h` by `Z_bar @ diag(S_bar)` before computing Gram matrices. In code, `var_svectors` stores `Z_bar^T` (rows = principal axes), so we transpose it to get `Z_bar` (columns = principal axes) and scale each column:

```python
Z_diag_s = var_svectors.T * var_singular_values[None, :]  # = Z_bar @ diag(S_bar)
W_eff = W_OV^h @ Z_diag_s
```

This rotates into the PCA basis and scales each axis by how much the data varies along it. The resulting Gram matrices reflect overlap *in the subspace the data actually uses*.

### Step 6: Random baselines

Each heatmap includes a random baseline: the expected FCS between Gram matrices of random Gaussian matrices with the same dimensions.

**Analytical formulas** (for random `(m x n)` matrices with i.i.d. N(0,1) entries):
- Read Gram FCS: `m / (m + n + 1)`
- Write Gram FCS: `n / (m + n + 1)`
- Raw FCS: `0`

For W_OV (`m = n = 768`): read and write baselines are both ~0.500. For data-weighted versions, baselines are higher (computed empirically via Monte Carlo, 1000 trials). The analytical formula is verified against the empirical computation on every run.

Values above baseline indicate genuine structural similarity; values at or below baseline are consistent with random matrices.

### Step 7: K-filtered data weighting

The data weighting in Step 5 uses all tokens. But the OV circuit reads from key/value positions — the attended-to tokens. When studying a particular attention behavior, the relevant input distribution is the activations at positions where the K component fires.

To filter:

1. Run the PD ComponentModel on the data with `cache_type="input"`
2. Compute causal importance (CI) for the specified K component (using `sampling="continuous"`)
3. Keep only tokens where the K component has CI > 0.5
4. Compute PCA on the filtered activations
5. Use the filtered PCA for data weighting

This produces a separate set of OV overlap figures and saves filtered SVD data for the semantic analysis script.

### Step 8: Component-head amplification

For each PD value component `c` and head `h`, compute `||W_OV^h v_hat_c||` where `v_hat_c = V[:, c] / ||V[:, c]||` — how much the head's OV circuit amplifies the component's read direction (normalized to unit length). Plotted as a `(n_components x n_heads)` heatmap sorted by max amplification.

Note: the plot function is named `_plot_component_head_amplification` with a `v_weight_per_head` parameter name (a legacy from when it only analyzed W_V), but at runtime it receives `ov_weight_per_head`.

### Step 9: Subspace semantics (`analyze_ov_subspace_semantics.py`)

For each head, identifies which PD components are most aligned with its OV subspace.

**Read-side** (V components): Each v_proj component has weight `U_v[c] @ V_v[c]^T`. Since U and V are unnormalized, we scale the read vector by the norm of the other factor to reflect the component's true contribution:
```
v_scaled_c = V_v[:, c] * ||U_v[c, :]||
read_alignment[h, c] = ||W_OV^h @ v_scaled_c||
```

**Write-side** (O components): Each o_proj component has weight `U_o[c] @ V_o[c]^T`. Similarly:
```
u_scaled_c = U_o[c, :] * ||V_o[:, c]||
write_alignment[h, c] = ||W_OV^{hT} @ u_scaled_c||
```

Three variants are produced:
- **Raw** (`raw/`): alignment on raw `W_OV`
- **Data weighted** (`data_weighted/`): alignment on `W_OV @ Z_diag_s` (all-token PCA)
- **K-filtered** (`k_*/`): alignment on `W_OV @ Z_diag_s_filtered` (auto-discovered from saved K-filtered SVD data)

Each outputs a markdown file with per-head tables of the top-20 V and O components ranked by alignment, with their autointerp labels.

## Output structure

### From `plot_wv_subspace_overlap.py`:

```
out/<run_id>/ov/
├── layer1_ov_paper_figure.png              # 1x3: data-weighted read/write/raw FCS
├── layer1_ov_paper_figure_unweighted.png   # 1x3: unweighted read/write/raw FCS
├── layer1_read_overlap_combined.png        # 1x2: unweighted + data-weighted read overlap
├── layer1_write_overlap_combined.png       # 1x2: unweighted + data-weighted write overlap
├── layer1_component_head_amplification.png # (n_components x n_heads) heatmap
├── layer1_ov_weight_per_head.npy           # (n_heads, d_model, d_model) — for semantics script
├── layer1_var_svectors.npy                 # (d_model, d_model) — PCA right singular vectors
├── layer1_var_singular_values.npy          # (d_model,) — PCA singular values
└── k_329/                                  # only if --k_filter was used
    ├── layer1_ov_paper_figure_k_329.png
    ├── layer1_var_svectors.npy             # K-filtered PCA vectors
    └── layer1_var_singular_values.npy      # K-filtered PCA singular values
```

In full (non-`--fast`) mode, these additional individual variant plots are also produced:
```
├── layer1_wv_subspace_overlap.png          # unweighted read Gram FCS
├── layer1_wv_strength_weighted_overlap.png # strength-weighted overlap
├── layer1_wv_data_weighted_overlap.png     # non-centered SVD data-weighted
├── layer1_wv_variance_weighted_overlap.png # PCA data-weighted
└── layer1_wv_data_strength_weighted_overlap.png
```

### From `analyze_ov_subspace_semantics.py`:

```
out/<run_id>/ov/
├── raw/
│   └── layer1_subspace_semantics.md        # component alignment on raw W_OV
├── data_weighted/
│   └── layer1_subspace_semantics.md        # component alignment on PCA-weighted W_OV
└── k_329/                                  # auto-discovered from saved K-filtered SVD
    └── layer1_subspace_semantics.md        # component alignment with K-filtered weighting
```

## Key equations reference

| Metric | Formula | Measures |
|--------|---------|----------|
| Read FCS | `tr(M_a M_b) / (norms)` where `M = W^T W` | Input subspace overlap |
| Write FCS | `tr(M_a M_b) / (norms)` where `M = W W^T` | Output subspace overlap |
| Raw FCS | `<vec(W_a), vec(W_b)> / (norms)` | Full linear map similarity |
| Data weighting | `W_eff = W @ Z_bar @ diag(S_bar)` | Upweight data-relevant directions |
| Read baseline | `m / (m + n + 1)` for random `(m x n)` | Expected FCS of read Grams |
| Write baseline | `n / (m + n + 1)` for random `(m x n)` | Expected FCS of write Grams |
| Raw baseline | `0` | Expected FCS of random matrices |
| Read alignment | `\|\|W v_scaled\|\|` where `v_scaled = V * \|\|U\|\|` | V component -> head read affinity |
| Write alignment | `\|\|W^T u_scaled\|\|` where `u_scaled = U * \|\|V\|\|` | O component -> head write affinity |
