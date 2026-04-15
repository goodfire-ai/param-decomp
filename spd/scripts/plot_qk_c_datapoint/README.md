# Data-Specific QK Component Contribution Plots

This script decomposes the pre-softmax attention logits for a **single prompt and query position** into contributions from individual (q\_component, k\_component) pairs. It answers the question: "for this specific token attending to each previous token, which component pairs are driving the attention pattern?"

## Background: How This Relates to the Weight-Only Script

The existing `plot_qk_c_attention_contributions` script computes **weight-only** QK interactions. It takes each component's U vector (the output-side weight), scales it by `||V|| * sign(mean_activation)` as a proxy for typical activation strength, and computes a RoPE-aware dot product at each abstract offset (0, 1, 2, ..., 16). The result is a data-averaged picture of which component pairs tend to cooperate.

This script does something different: it uses the **actual component activations** on a specific input to compute data-dependent contributions. Instead of abstract offsets on the x-axis, it plots actual key positions (with token labels). Instead of heuristic scaling, it uses the real activation values `v_c^T * x` at each position.

## Mathematical Derivation

### The Decomposition

SPD decomposes each linear projection `W` as `V @ U`, where:
- `V` is `(d_in, C)` — the input-side matrix
- `U` is `(C, d_out)` — the output-side matrix
- `C` is the number of components

For an input `x`, the **component activation** of component `c` is:

```
act_c(x) = v_c^T * x       (a scalar)
```

And the full projection is:

```
W @ x = sum_c  act_c(x) * u_c
```

### Applying to Attention

For a given head `h`, the pre-softmax attention logit between query position `t_q` and key position `t_k` is:

```
logit[h, t_q, t_k] = (1 / sqrt(d_head)) * q_h[t_q] . k_h[t_k]
```

where `q_h` and `k_h` are the RoPE-rotated query and key vectors for head `h`. Substituting the component decomposition:

```
q_h[t_q] = sum_i  act_q_i[t_q] * RoPE(u_q_i_h, t_q)
k_h[t_k] = sum_j  act_k_j[t_k] * RoPE(u_k_j_h, t_k)
```

The logit becomes:

```
logit[h, t_q, t_k] = (1 / sqrt(d_head)) * sum_{i,j}  act_q_i[t_q] * act_k_j[t_k] * W_rope(u_q_i_h, u_k_j_h, t_q - t_k)
```

where `W_rope(u_q, u_k, delta)` is the RoPE-modulated dot product at relative offset `delta = t_q - t_k`. This is computed by the shared `rope_aware_qk.py` module using the identity:

```
W_rope(delta) = sum_d [ A_d * cos(delta * theta_d) + B_d * sin(delta * theta_d) ]
```

where `A` and `B` are content-aligned and cross-half coefficients derived from the U vectors.

### What Each Term Means

Each term `act_q_i[t_q] * act_k_j[t_k] * W_rope(...)` is **the contribution of component pair (i, j) to the attention logit at this (query, key) position**. It has three factors:

1. **`act_q_i[t_q]`** — how strongly q-component `i` activates on the query token
2. **`act_k_j[t_k]`** — how strongly k-component `j` activates on the key token
3. **`W_rope(u_q_i, u_k_j, delta)`** — the weight-space affinity between these components at this relative position (this is what the weight-only script plots)

### The Weight Delta

The component decomposition `V @ U` does not perfectly reconstruct the target model weights `W_target`. There is a weight delta `W_target - V @ U`. This means the sum of all component contributions will not exactly equal the ground-truth logits — there will be a small residual from cross-terms involving the delta. The validation plot shows this residual per head. Typical residuals are ~0.4-0.5 in absolute value, small relative to the logit range.

## Two Modes

### Weighted Mode (default)

Each pair's contribution is scaled by the actual component activations:

```
contribution(i, j, t_k) = (1/sqrt(d_head)) * act_q_i[t_q] * act_k_j[t_k] * W_rope(i, j, t_q - t_k)
```

The sum over all (i, j) pairs should approximately match the ground-truth pre-softmax logits. This is the primary mode for validating the decomposition.

### Binary Mode

Each pair's contribution uses the weight-only dot product, gated by whether both components are "active" (per-token causal importance exceeds `--ci_threshold`):

```
contribution(i, j, t_k) = (1/sqrt(d_head)) * W_rope(i, j, t_q - t_k)    if CI_q_i[t_q] > threshold AND CI_k_j[t_k] > threshold
                         = 0                                                otherwise
```

This shows the structural contribution pattern — which pairs are active regardless of activation magnitude. Because binary mode doesn't scale by activations, its values are in different units than the actual pre-softmax logits. The plot therefore **does not overlay ground truth** and the y-axis auto-scales to the binary contribution range. No validation plot is produced either, since the comparison is not meaningful.

Note that CI values in this model are typically very sparse — most components have CI near 0 at any given position. The default `--ci_threshold` of 0.01 reflects this. If your plot looks flat at zero, try lowering the threshold further or switching to weighted mode.

## Output Plots

### 1. Combined Lines (`*_combined.png`)

A 4x2 grid matching the layout of the weight-only script's `lines_combined` plot:

- **Top-left**: Mean across all heads
- **Top-right**: Legend
- **Bottom 3x2**: One subplot per head (H0-H5)

In each subplot:
- **Colored lines with markers**: Top-N (q, k) component pairs ranked by peak absolute contribution. Only "alive" pairs (mean CI > `--min_mean_ci`) are shown as individual lines.
- **Gray lines**: Remaining alive pairs
- **Black solid line**: Sum over **all** components (not just alive ones)
- **Red dashed line** (weighted mode only): Ground-truth pre-softmax logits from the target model
- **X-axis**: Key positions labeled with their tokens

In weighted mode, the black and red lines should track each other closely — any gap is the weight delta residual. In binary mode, no ground truth is shown (different units) and the y-axis auto-scales to the contribution range.

### 2. Validation (`*_validation.png`) — weighted mode only

A 3x2 grid (one subplot per head) showing:
- **Black line**: Sum of all component contributions
- **Red dashed line**: Ground-truth logits
- **Gray bars**: Residual (component sum minus ground truth) on a secondary y-axis

The title reports `max |residual|` across all heads and positions. This plot is not produced in binary mode.

## Usage

```bash
python -m spd.scripts.plot_qk_c_datapoint.plot_qk_c_datapoint \
    wandb:goodfire/spd/runs/<run_id> \
    --prompt "The cat sat on the mat" \
    --query_pos 5 \
    --layer 1 \
    --mode weighted
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `wandb_path` | (required) | WandB run path, e.g. `wandb:goodfire/spd/runs/s-55ea3f9b` |
| `--prompt` | (required) | Text prompt to tokenize and analyze |
| `--query_pos` | (required) | 0-indexed position of the query token |
| `--layer` | (required) | Layer index (0-based) |
| `--mode` | `weighted` | `"weighted"` or `"binary"` |
| `--ci_threshold` | `0.01` | Per-token CI threshold for binary mode |
| `--top_n_pairs` | `10` | Number of top pairs to highlight in color |
| `--min_mean_ci` | `0.001` | Mean CI threshold for "alive" filtering (display only) |

### Prerequisites

- The SPD run must have harvest data available (run `spd-harvest` first)
- Currently supports `LlamaSimpleMLP` target models only

### Output Location

Plots are saved to `spd/scripts/plot_qk_c_datapoint/out/<run_id>/`.

## Key Differences from the Weight-Only Script

| | Weight-only (`plot_qk_c_attention_contributions`) | Data-specific (this script) |
|---|---|---|
| **U vector scaling** | `U * \|\|V\|\| * sign(mean_activation)` — a heuristic proxy | Raw U vectors; actual activations carry the scaling |
| **Activation weights** | None (weight-only analysis) | `act_q_i[t_q] * act_k_j[t_k]` per position |
| **X-axis** | Abstract offsets 0..16 | Actual key positions with token labels |
| **Normalization** | Z-scored per head (for cross-head comparison) | Raw pre-softmax logit scale (for ground-truth matching) |
| **1/sqrt(d_head)** | Not applied (z-scoring absorbs it) | Applied (needed to match actual logits) |
| **Ground truth** | None | Overlaid from target model's actual attention |
| **Alive filtering** | Controls which components are computed | Controls which pairs get individual lines; sum uses ALL components |

## Code Structure

### `_compute_datapoint_contributions()`

The core computation function. Steps:

1. Forward pass through `ComponentModel` with `cache_type="input"` to cache pre-weight activations
2. `get_all_component_acts(cache)` computes `v_c^T * x` for every component at every position
3. Extract q activations at `query_pos` and k activations at positions `0..query_pos`
4. For each head, compute RoPE-aware weight dot products using `compute_qk_rope_coefficients` + `evaluate_qk_at_offsets` on raw U vectors
5. Multiply weight dot products by activation weights (weighted mode) or CI-gated binary mask (binary mode)
6. Apply `1/sqrt(d_head)` scaling
7. Collect ground-truth logits via `collect_attention_patterns_with_logits` on the target model

### `_plot_combined_lines()`

Generates the 4x2 combined plot. The contribution tensor has shape `(n_q_heads, C_q, C_k, n_key_pos)` where `C_q` and `C_k` are the **total** component counts. For display, it indexes into the alive subset `W[:, q_alive][:, :, k_alive]` to draw individual pair lines, but the sum line uses the full tensor.

### `_plot_validation()`

Generates the per-head validation overlay. Simply sums contributions over all (q, k) pairs and compares to ground truth.
