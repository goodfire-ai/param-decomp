# Data-Specific QK Component Contribution Plots

This script decomposes the pre-softmax attention logits for **individual dataset samples** into contributions from (q\_component, k\_component) pairs. It answers the question: "for this specific token attending to each previous token, which component pairs are driving the attention pattern?"

## Background

The existing `plot_qk_c_attention_contributions` script computes **weight-only** QK interactions averaged over data. This script uses **actual component activations** on specific inputs to compute data-dependent contributions, validating that the decomposition tracks ground-truth logits.

## Mathematical Derivation

SPD decomposes each linear projection `W` as `V @ U` (V: d\_in x C, U: C x d\_out). For input `x`, component `c`'s activation is `act_c(x) = v_c^T * x`.

For head `h`, query position `t_q`, key position `t_k`:

```
logit[h, t_q, t_k] = (1/sqrt(d_head)) * sum_{i,j} act_q_i[t_q] * act_k_j[t_k] * W_rope(u_q_i_h, u_k_j_h, t_q - t_k)
```

where `W_rope` is the RoPE-modulated weight dot product computed by `rope_aware_qk.py`.

## Two Modes

### Weighted (default)

Contributions scaled by actual component activations. The sum over all pairs should approximately match ground-truth pre-softmax logits (residual is from the weight delta — `V@U` doesn't perfectly reconstruct target weights).

### Binary

Contributions gated by per-token CI threshold — weight-only dot product if both components are active, else zero. Ground truth is not overlaid (different units). Y-axis auto-scales to the binary contribution range.

## Output Plot

A 4x2 grid per (sample, layer, query\_pos):

- **Top-left**: Mean across all heads
- **Top-right**: Legend (two columns when > 12 entries)
- **Bottom 3x2**: Per-head subplots

In each subplot:
- **Colored lines**: Top-N (q, k) pairs ranked by peak absolute contribution on this datapoint (consistent across heads)
- **Black solid**: Sum over all components
- **Red dashed** (weighted only): Ground-truth pre-softmax logits
- **X-axis**: Key positions with token labels

Per-head subplots mask out pairs whose peak contribution in that head is below 5% of the head's logit range, reducing clutter while keeping the global pair set consistent.

## Usage

```bash
# Single sample
python -m spd.scripts.plot_qk_c_datapoint.plot_qk_c_datapoint \
    wandb:goodfire/spd/runs/<run_id> \
    --layer 1 \
    --sample_indices 0 \
    --query_positions 5

# Multiple samples (fire list syntax)
python -m spd.scripts.plot_qk_c_datapoint.plot_qk_c_datapoint \
    wandb:goodfire/spd/runs/<run_id> \
    --layer 1 \
    --sample_indices='[0,1,2,3,4]' \
    --query_positions 5
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `wandb_path` | (required) | WandB run path |
| `--layer` | (required) | Layer index (0-based) |
| `--sample_indices` | `0` | Dataset sample index or list of indices |
| `--query_positions` | `5` | Query position(s). Single int broadcast to all samples, or list matched 1:1 |
| `--mode` | `weighted` | `"weighted"` or `"binary"` |
| `--ci_threshold` | `0.01` | Per-token CI threshold for binary mode |
| `--top_n_pairs` | `20` | Number of top pairs to highlight |

### Output Location

```
spd/scripts/plot_qk_c_datapoint/out/<run_id>/layer<L>/sample<N>_pos<P>_<mode>.png
```

## Key Differences from the Weight-Only Script

| | Weight-only (`plot_qk_c_attention_contributions`) | Data-specific (this script) |
|---|---|---|
| **U scaling** | `U * \|\|V\|\| * sign(mean_activation)` | Raw U; actual activations carry scaling |
| **X-axis** | Abstract offsets 0..16 | Actual key positions with token labels |
| **Normalization** | Z-scored per head | Raw pre-softmax logit scale |
| **Pair ranking** | Harvest mean CI threshold | Peak abs contribution on this datapoint |
| **Ground truth** | None | Overlaid (weighted mode) |

## Weight Delta Residual

The component sum won't exactly match ground truth because `V@U` doesn't perfectly reconstruct the target weights. The weight delta is typically ~11% of the target weight Frobenius norm, producing logit residuals of ~0.3-0.6.
