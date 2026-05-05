# Interactive QK Contribution Heatmaps

This tool visualizes how individual PD components contribute to attention patterns. It decomposes the attention score between any query and key position into contributions from specific (q_component, k_component) pairs, letting you see which components drive attention between which tokens.

## Background: PD Component Decomposition

In PD, each linear layer's weight matrix `W` (shape `d_out x d_in`, matching `nn.Linear` convention) is decomposed as a sum of rank-1 terms:

```
W = sum_c  u_c @ v_c^T
```

where `v_c` is the input-direction vector (column of `V`, shape `d_in`) and `u_c` is the output-direction vector (row of `U`, shape `d_out`). There are `C` components per layer (e.g. 512). In code, `V` has shape `(d_in, C)` and `U` has shape `(C, d_out)`, so `weight = (V @ U)^T`.

When input `x` passes through this layer, the output is:

```
output = sum_c  u_c * (v_c . x)
       = sum_c  u_c * a_c
```

where `a_c = v_c . x` is the **component activation** -- a scalar measuring how strongly the input aligns with component `c`'s input direction. The full output is a weighted sum of the output directions `u_c`, weighted by activations `a_c`.

## How Attention Scores Decompose

### Standard Attention

In standard (single-head) attention, the pre-softmax attention logit from query position `q` to key position `k` is:

```
logit(q, k) = (1/sqrt(d)) * RoPE(W_Q x_q) . RoPE(W_K x_k)
```

where `RoPE(.)` applies rotary position embeddings and `d` is the head dimension.

### Decomposed Attention

With PD, `W_Q` and `W_K` are each decomposed into components. The query vector at position `q` becomes:

```
query(q) = W_Q x_q = sum_i  u_i^Q * a_i^Q(q)
```

and the key vector at position `k` becomes:

```
key(k) = W_K x_k = sum_j  u_j^K * a_j^K(k)
```

where `a_i^Q(q) = v_i^Q . x_q` and `a_j^K(k) = v_j^K . x_k` are the component activations at those positions.

Substituting into the attention logit (and distributing the dot product over the sums):

```
logit(q, k) = (1/sqrt(d)) * sum_i sum_j  a_i^Q(q) * a_j^K(k) * RoPE(u_i^Q) . RoPE(u_j^K)
```

This decomposes the attention logit into a sum over all (q_component `i`, k_component `j`) pairs. Each pair's contribution is:

```
contribution(i, j, q, k) = (1/sqrt(d)) * a_i^Q(q) * a_j^K(k) * W_ij(delta)
```

where `delta = q - k` is the relative position offset and `W_ij(delta)` is the **interaction strength** -- the RoPE-modulated dot product between the output directions of q_component `i` and k_component `j`.

### The Interaction Strength W_ij(delta)

The interaction strength `W_ij(delta)` captures how much the weight geometry of components `i` and `j` promotes attention at relative offset `delta`. It is computed using a RoPE-aware decomposition:

```
W_ij(delta) = sum_d [ A_ijd * cos(delta * theta_d) + B_ijd * sin(delta * theta_d) ]
```

where:
- `A_ijd = u_i^Q[first_half]_d * u_j^K[first_half]_d + u_i^Q[second_half]_d * u_j^K[second_half]_d` (content-aligned terms)
- `B_ijd = u_i^Q[first_half]_d * u_j^K[second_half]_d - u_i^Q[second_half]_d * u_j^K[first_half]_d` (cross-half terms)
- `theta_d` are the RoPE rotation frequencies

This is a closed-form expression: given the component weight vectors, we can evaluate the interaction strength at any offset without running the model. The `A` coefficients capture position-independent alignment (they dominate at `delta=0`), while the `B` coefficients capture position-dependent interactions (they vanish at `delta=0` since `sin(0)=0`).

Note: this decomposition assumes the non-adjacent-pairs RoPE layout (first-half / second-half dimension split), which is what LlamaSimpleMLP uses.

### Why the Sum Exactly Recovers the Attention Score

The decomposition is exact (not an approximation) because it's just distributing the dot product. Concretely:

1. The query vector is `sum_i u_i^Q * a_i^Q(q)` (by definition of the decomposed weight matrix)
2. The key vector is `sum_j u_j^K * a_j^K(k)` (same)
3. The dot product of two sums equals the sum of all pairwise dot products:
   ```
   (sum_i u_i * a_i) . (sum_j u_j * a_j) = sum_i sum_j a_i * a_j * (u_i . u_j)
   ```
4. RoPE is applied element-wise before the dot product, which is absorbed into `W_ij(delta)`

Therefore: **summing the contributions of ALL (q_component, k_component) pairs exactly reproduces the attention logit**. There is no residual or approximation error. In the viewer, selecting all pairs and applying softmax produces attention patterns that match the model's actual attention to floating-point precision (~1e-7).

In practice, we only include "alive" components (those with firing density above a threshold). The validation in the viewer compares against attention computed using only these alive components' reconstructed weights, so the match remains exact within the alive set.

### Factored Structure

Each pair's contribution factorizes into three independent terms:

```
contribution(i, j, q, k) = (1/sqrt(d))  *  W_ij(delta)  *  a_i^Q(q)  *  a_j^K(k)
                            \_________/     \__________/     \________/     \________/
                              scale        interaction      q activation   k activation
                                           strength         at pos q       at pos k
                                          (weights only)   (data-dependent)
```

- **W_ij(delta)** depends only on the learned weights and the relative position -- it is the same for all inputs. It tells you "how much do these two components structurally promote attention at this offset?"
- **a_i^Q(q)** and **a_j^K(k)** are data-dependent scalars. They tell you "how active is each component on this particular input at this particular position?"
- The contribution is their product: a pair only contributes meaningfully to attention if (a) the weight geometry promotes attention at that offset AND (b) both components are active on the input.

## What the Viewer Shows

### Heatmap Rows

1. **Alive-only model attention (ground truth)**: The actual attention patterns produced by the model using only the alive components' weights. This is always shown as the reference.

2. **Sum of selected pairs (softmax attention)**: The attention pattern that would result from only the selected (q, k) component pairs. Computed by summing the selected pairs' pre-softmax logits, then applying softmax. When all pairs are selected, this matches row 1 exactly.

3. **Sum of selected pairs (pre-softmax logits)**: The raw logits before softmax, shown with a diverging blue-white-red colormap. Positive values promote attention, negative values suppress it.

4. **Individual pair rows**: Each selected pair gets its own row showing its contribution heatmap. This lets you see which specific component interactions drive attention to particular positions.

### Controls

- **Prompt/Layer dropdowns**: Select which input and layer to visualize.
- **Pair checkboxes**: Toggle individual (q_component, k_component) pairs. Pairs are ranked by peak absolute contribution.
- **Select all / Clear**: Bulk selection controls.

## Usage

There are two precompute entrypoints, picked by who consumes the output:

- **`compute_data.py`** — full alive×alive sweep + ranked `top_pairs`, for the local `viewer.html` research tool.
- **`compute_pair_data.py`** — explicit `(q, k)` pairs from a YAML config, for the blog widget (`vpd-blog-replit/js/attention_qk_grid.js`).

### `compute_data.py` — full sweep

```bash
python -m param_decomp.scripts.interactive_qk_contributions.compute_data \
    wandb:goodfire/spd/runs/<run_id> --layer 1 \
    --prompts_file path/to/prompts.json
```

Or sample from the dataset:

```bash
python -m param_decomp.scripts.interactive_qk_contributions.compute_data \
    wandb:goodfire/spd/runs/<run_id> --layer 1 \
    --dataset_samples 30 --seq_len 24
```

Options:
- `--layer N` -- which layer to compute (single int, required)
- `--min_density 0.001` -- firing density threshold for alive components (read from harvest)
- `--top_k N` -- keep only the top-N pairs by peak contribution and slice `alive_q`/`alive_k`/`W` to the components involved in those pairs (shrinks files for layers with many alive components)
- `--output path/to/file.json` -- override output path (default: `out/<run_id>/prompts.json`)

Open the resulting JSON in `viewer.html`.

### `compute_pair_data.py` — explicit pairs

```bash
python -m param_decomp.scripts.interactive_qk_contributions.compute_pair_data path/to/config.yaml
```

Config schema (`PairDataConfig`):
```yaml
wandb_path: wandb:goodfire/spd/runs/<run_id>
layer: 1
prompts:
  - "..."
pairs:
  - [<q_idx>, <k_idx>]
```

Output goes to `out/<run_id>/pairs.json` by default; override with `--output path/to/file.json`. No harvest dependency, no ranking — `alive_q`/`alive_k` are derived from the unique component indices in `pairs`. The output shape matches `compute_data.py` minus `q_ci`/`k_ci`/`top_pairs` (the blog widget doesn't read them).

### `build_targeted_prompts.py` — generate prompts from harvest examples

```bash
python -m param_decomp.scripts.interactive_qk_contributions.build_targeted_prompts \
    wandb:goodfire/spd/runs/<run_id> \
    --components 'h.1.attn.q_proj:308,h.1.attn.k_proj:218'
```

Pulls reservoir-sampled activation examples for the listed components from harvest and decodes them into a `targeted_prompts.json` flat list. Pipe into `compute_data.py --prompts_file`.

### Prompts

Edit `handwritten_prompts.json` for curated prompts (consumed by `compute_data.py --prompts_file`). To sample from the training dataset instead, use `--dataset_samples N --seq_len T`. To target specific components, use `build_targeted_prompts.py`.

## File Structure

```
interactive_qk_contributions/
    README.md                       # This file
    compute_data.py                 # Full sweep, output for viewer.html
    compute_pair_data.py            # Explicit pairs, output for blog widget
    build_targeted_prompts.py       # Harvest examples → prompts.json
    viewer.html                     # Local research viewer (reads compute_data.py output)
    handwritten_prompts.json        # Curated prompt list
    targeted_prompts.json           # Output of build_targeted_prompts.py
    out/<run_id>/prompts.json       # compute_data.py default output
    out/<run_id>/pairs.json         # compute_pair_data.py default output
```

### JSON Data Format

```
{
  "prompts": [
    {
      "tokens": [...],          // [T]
      "label": "...",
      "layer_idx": 1,
      "n_heads": 6, "head_dim": 128, "scale": 0.0884,
      "alive_q": [...],         // [n_q]
      "alive_k": [...],         // [n_k]
      "W": [...],               // [n_heads, n_offsets, n_q, n_k]
      "q_acts": [...],          // [T, n_q]
      "k_acts": [...],          // [T, n_k]
      "q_ci": [...],            // [T, n_q]   — compute_data.py only
      "k_ci": [...],            // [T, n_k]   — compute_data.py only
      "component_model_attn": [...],  // [n_heads, T, T]
      "top_pairs": [[qi, ki, score], ...],   // compute_data.py only
      "pairs": [[q_idx, k_idx], ...]         // compute_pair_data.py only (echoes config)
    }
  ]
}
```

Each prompt entry is flat. `layer_idx` is repeated per prompt but identical across all prompts in a single file -- the file is single-layer.

The viewer assembles heatmaps client-side (causal: only for `q >= k`):
```
heatmap[h][q][k] = scale * sum_{(i,j) in selected} W[h][q-k][i][j] * q_acts[q][i] * k_acts[k][j]
```
