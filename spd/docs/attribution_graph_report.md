# Attribution Graph Computation in SPD

## 1. Overview

This document describes the computation of **dataset attribution graphs** in the SPD (Stochastic Parameter Decomposition) framework. An attribution graph is a weighted directed graph whose nodes are *parameter components* (the learned units of a decomposed neural network) and whose edges quantify how strongly each source component influences each target component, aggregated over a large corpus of data.

Formally, let a trained SPD model decompose the weight matrix of each target module $\ell$ into $C_\ell$ rank-1 components:

$$W_\ell \approx \sum_{i=1}^{C_\ell} V_{\ell,i} \, U_{\ell,i}^\top$$

where $V_{\ell,i} \in \mathbb{R}^{d_\text{in}}$ and $U_{\ell,i} \in \mathbb{R}^{d_\text{out}}$ are the input and output directions of component $i$ at layer $\ell$. The component activation of component $i$ on input $x$ is $a_{\ell,i}(x) = V_{\ell,i}^\top x$, and the component's output contribution is $a_{\ell,i}(x) \cdot U_{\ell,i}$.

The attribution graph answers the question: *in aggregate across the training data, how much does source component $s$ influence target component $t$?*

## 2. The Attribution Formula

### 2.1 Core Definition

The dataset attribution from source component $s$ to target component $t$ is defined as:

$$\text{attr}(s \to t) = \sum_{\text{batch}} \sum_{\text{pos}} \frac{\partial \, a_{t}(\text{pos})}{\partial \, a_{s}(\text{pos})} \cdot a_{s}(\text{pos})$$

where:
- $a_t(\text{pos})$ is the pre-mask component activation of target component $t$ at position `pos`
- $a_s(\text{pos})$ is the post-mask component activation of source component $s$ at position `pos`
- The partial derivative measures the linear sensitivity of the target activation to the source activation, propagated through all intervening layers of the network

This is the **gradient $\times$ activation** formula, sometimes called the *path attribution* or *integrated gradient approximation at the operating point*. It decomposes the total contribution of the source into a product of (a) how sensitive the target is to the source (the gradient) and (b) how strongly the source fires (the activation).

### 2.2 Two Metrics

Two variants of the attribution are computed:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `attr` | $\mathbb{E}\bigl[\frac{\partial a_t}{\partial a_s} \cdot a_s\bigr]$ | Signed mean attribution. Positive values indicate that the source promotes the target; negative values indicate suppression. |
| `attr_abs` | $\sum_\text{pos} \frac{\partial \lvert a_t(\text{pos}) \rvert}{\partial a_s(\text{pos})} \cdot a_s(\text{pos})$ | Attribution to the absolute value of the target at each position. Captures the source's influence on the *magnitude* of the target, irrespective of sign. Only computed for component targets, not output targets. |

The `attr_abs` metric requires a separate backward pass. The implementation first applies `.abs()` element-wise to each position's activation and *then* sums over positions: `target_acts_abs = target_acts_raw.abs().sum(dim=(0, 1))`. This is the sum of absolute values, not the absolute value of the sum. The distinction matters because the sign of $a_t(\text{pos})$ varies across positions, so the gradient $\frac{\partial}{\partial a_s} \sum_\text{pos} |a_t(\text{pos})| = \sum_\text{pos} \text{sgn}(a_t(\text{pos})) \cdot \frac{\partial a_t(\text{pos})}{\partial a_s}$ is data-dependent and cannot be factored out of the summation. This necessitates a separate backward pass through the position-summed absolute-value scalar.

Note: `attr_abs` is **not computed for output (unembed) targets**. The residual-space storage optimization used for output edges is incompatible with the nonlinear absolute value operation.

### 2.3 CI-Weighted Source Activations

For **component sources** (as opposed to embedding sources), the source activation in the attribution formula is weighted by the component's causal importance (CI):

$$\text{attr}(s \to t) = \sum_{\text{batch}} \sum_{\text{pos}} \frac{\partial \, a_t}{\partial \, a_s} \cdot a_s \cdot \text{CI}(s, \text{pos})$$

where $\text{CI}(s, \text{pos}) \in [0, 1]$ is the causal importance of source component $s$ at that position, specifically the `lower_leaky` sigmoid output of the CI function. This weighting ensures that only "alive" components --- those the CI function deems causally relevant --- contribute to the attribution. Components with CI near zero on a given datapoint are effectively masked out, preventing dead components from contributing noise to the attribution graph.

## 3. Graph Topology Discovery

Before computing attributions, the system must discover which (source, target) pairs are connected by gradient flow. Not all pairs have a valid gradient path --- for example, a component in layer 3 has no gradient path to a component in layer 1.

### 3.1 Gradient Connectivity Probing

The function `get_sources_by_target()` (`spd/topology/gradient_connectivity.py`) discovers the graph topology by:

1. **Creating a dummy batch** (2 sequences of length 3) filled with zeros.
2. **Running a forward pass** with all component masks set to 1 (all components active, `routing_masks="all"`), with `cache_type="component_acts"` to capture component activations at every layer. A hook on the embedding module enables gradients on its output, and the model's final output logits are stored as the unembed target's `pre_detach` entry (since the unembedding is not a decomposed component, it has no component-level detach boundary).
3. **Probing all source-target pairs**: For each candidate pair $(s, t)$ where $s \ne t$:
   - Compute $\frac{\partial \, \text{out}_t[0, 0, 0]}{\partial \, \text{in}_s}$ via `torch.autograd.grad` with `allow_unused=True`, where $\text{out}_t$ is the `pre_detach` cache entry for target $t$ and $\text{in}_s$ is the `post_detach` cache entry for source $s$.
   - If the gradient is non-`None`, a gradient path exists: add $s$ to the source list of $t$.

**Valid nodes**:
- **Sources**: the embedding layer and all component layers (not the unembedding)
- **Targets**: all component layers and the unembedding (not the embedding)

The result is a dictionary `sources_by_target: dict[target_layer, list[source_layer]]` describing the computational graph over which attributions will be computed.

### 3.2 Alive Component Filtering

Not all components within a layer fire meaningfully. The harvester loads **firing density** statistics from a prior harvest run and builds a boolean mask `component_alive[layer]` of shape `(C_ℓ,)`. A component is "alive" if its firing density (fraction of datapoints on which it activates above a threshold) is strictly greater than zero.

Only alive components are processed as **targets** during gradient computation. All components are available as sources (their contribution is naturally suppressed by low CI values).

## 4. The Detach-and-Reattach Mechanism

A central design pattern enables efficient per-component gradient computation. During the forward pass with `cache_type="component_acts"`, each `Components.forward()` (both `LinearComponents` and `EmbeddingComponents`) performs:

```python
component_acts = self.get_component_acts(x)    # V^T x, shape (... C)
# --- detach boundary ---
component_acts_cache["pre_detach"] = component_acts
component_acts = component_acts.detach().requires_grad_(True)
component_acts_cache["post_detach"] = component_acts
# --- continue forward ---
if mask is not None:
    component_acts = component_acts * mask
out = component_acts @ U
```

This creates two views of the component activations at each layer, both of shape `(batch, seq, C)` and holding the same numerical values:
- **`pre_detach`**: retains the gradient connection to all upstream computations. This is the "target" view --- differentiating through it propagates gradients to upstream source activations.
- **`post_detach`**: a fresh leaf tensor with `requires_grad=True`. This is the "source" view --- its gradient tells us how downstream targets depend on this layer's component activations.

Crucially, both `pre_detach` and `post_detach` are captured **before** the component mask is applied. The mask multiplication happens after the detach boundary, so `post_detach` represents the raw, unmasked component activation $V_\ell^\top x$. This means the gradient $\frac{\partial a_t}{\partial a_s}$ computed via `autograd.grad(target_pre_detach, source_post_detach)` reflects how the target depends on the source's unmasked activation, which is then multiplied by the source activation and CI weight during accumulation.

For the embedding and unembedding modules --- which are not decomposed into components --- the detach boundary is handled differently. A forward hook on the embedding module calls `output.requires_grad_(True)` and stores the result as its `post_detach` entry. For the unembedding target, the pre-unembed residual stream (captured via a forward pre-hook on the unembed module) serves as the `pre_detach` entry, since there is no component-level decomposition at the output.

The detach boundary ensures that `torch.autograd.grad(target_pre_detach[t_idx], source_post_detach)` computes exactly $\frac{\partial a_t}{\partial a_s}$ for the path *between* those two layers, without conflating gradients from paths that bypass the source layer entirely.

## 5. Batch Processing Pipeline

### 5.1 Per-Batch Computation

For each batch of tokens, `AttributionHarvester.process_batch()` executes:

**Step 1: Compute Causal Importances.** A forward pass with `cache_type="input"` captures pre-weight activations. These are fed to the CI function to produce per-component importance values $\text{CI}(\ell, i) \in [0, 1]$ for every component. This step runs under `torch.no_grad()`.

**Step 2: Forward with All Components Active.** A second forward pass runs with:
- All component masks set to 1 (every component fully active)
- `cache_type="component_acts"` to capture both `pre_detach` and `post_detach` activations at every layer
- `torch.enable_grad()` to build the computation graph for gradient computation
- Forward hooks on the embedding and unembedding modules to capture the embedding output (with gradients enabled) and the pre-unembedding residual stream

**Step 3: Process Each Target Layer.** Target layers are processed in two distinct ways depending on whether the target is a component layer or the output (unembedding) layer:

#### Component Targets

For each component target layer $t$:

1. **Sum over positions.** The target activations `pre_detach` are summed over batch and sequence dimensions *before* computing gradients. Two scalars are produced per target component: $\bar{a}_{t,i} = \sum_{\text{pos}} a_{t,i}(\text{pos})$ for the signed metric, and $\overline{|a|}_{t,i} = \sum_{\text{pos}} |a_{t,i}(\text{pos})|$ for the absolute metric.

2. **Accumulate squared activations.** The sum of squared `post_detach` activations is accumulated for normalization: `_square_component_act_accumulator[t] += post_detach.square().sum(dim=(0, 1))`.

3. **For each alive target component $t_i$:** Compute gradients of $\bar{a}_{t,i}$ (a scalar) with respect to all source-layer `post_detach` activations simultaneously via a single `torch.autograd.grad()` call with `retain_graph=True`:

   ```python
   grads = torch.autograd.grad(target_acts[t_idx], source_acts, retain_graph=True)
   ```

   This returns one gradient tensor per source layer, each of shape `(batch, seq, C_source)`.

4. **Accumulate attributions.** For each source layer and its gradient:
   - **Component sources**: `attr[t_i, :] += (grad * source_post_detach * ci[source]).sum(dim=(0, 1))`
   - **Embedding sources**: `attr[t_i, :] += scatter_add(grad * embed_output, tokens)` --- accumulates per-token using `scatter_add_` for efficiency

5. **Repeat for `attr_abs`.** A second `autograd.grad` call computes gradients of $\overline{|a|}_{t,i}$ and accumulates into the `_abs` accumulators using the same formula.

#### Output (Unembed) Targets

The output target is handled fundamentally differently. Rather than looping over alive components, the harvester loops over the $d_\text{model}$ dimensions of the **pre-unembed residual stream**:

1. **Sum residual over positions.** The pre-unembed residual $r \in \mathbb{R}^{B \times S \times d_\text{model}}$ is summed over batch and sequence: $\bar{r} = \sum_{\text{pos}} r(\text{pos})$, yielding a vector in $\mathbb{R}^{d_\text{model}}$.

2. **For each residual dimension $d$:** Compute gradients of $\bar{r}_d$ (a scalar) with respect to all source-layer `post_detach` activations:

   ```python
   grads = torch.autograd.grad(out_residual_sum[d_idx], source_acts, retain_graph=True)
   ```

3. **Accumulate attributions in residual space.** For each source layer:
   - **Component sources**: `unembed_attr[source][d, :] += (grad * source_post_detach * ci[source]).sum(dim=(0, 1))`
   - **Embedding sources**: `straight_through_attr[d, :] += scatter_add(grad * embed_output, tokens)`

This residual-space accumulation requires $d_\text{model}$ backward passes per batch (typically 256--4096), independent of vocabulary size. No `attr_abs` is computed for output targets because the absolute value is a nonlinear operation that cannot be factored through the linear $W_\text{unembed}$ projection used at query time.

### 5.2 Accumulator Structure

The harvester maintains six groups of accumulators, all storing raw unnormalized sums:

| Accumulator | Shape | Description |
|-------------|-------|-------------|
| `_regular_layers_acc[t][s]` | `(C_t, C_s)` | Component $\to$ component signed attribution |
| `_regular_layers_acc_abs[t][s]` | `(C_t, C_s)` | Component $\to$ component absolute attribution |
| `_embed_tgts_acc[t]` | `(C_t, V)` | Embedding $\to$ component signed attribution |
| `_embed_tgts_acc_abs[t]` | `(C_t, V)` | Embedding $\to$ component absolute attribution |
| `_unembed_srcs_acc[s]` | `(d_\text{model}, C_s)` | Component $\to$ output signed attribution (residual space) |
| `_straight_through_attr_acc` | $(d_\text{model}, V)$ | Embedding $\to$ output signed attribution (residual space) |

Additionally, normalization metadata is accumulated:

| Metadata | Shape | Purpose |
|----------|-------|---------|
| `_ci_sum_accumulator[ℓ]` | `(C_ℓ,)` | Sum of CI values per component (source denominator) |
| `_square_component_act_accumulator[ℓ]` | `(C_ℓ,)` | Sum of squared component activations (target RMS) |
| `_logit_sq_sum` | `(V,)` | Sum of squared logits (output target RMS) |
| `_embed_token_count` | `(V,)` | Per-token occurrence count (embedding source denominator) |
| `n_tokens` | scalar | Total tokens processed |

## 6. Four Edge Types

The attribution graph contains four structurally distinct edge types, reflecting the different source and target node types:

### 6.1 Component $\to$ Component (Regular Edges)

The most common edge type. Source is a component in an earlier layer; target is a component in a later layer.

- **Storage**: `regular_attr[target_layer][source_layer]` of shape `(C_target, C_source)`
- **Both signed and absolute variants** are stored
- **Gradient formula**: $\frac{\partial a_t}{\partial a_s}$ flows through all intermediate layers between source and target

### 6.2 Embedding $\to$ Component (Embed Edges)

The embedding layer is not decomposed into components in the same way --- instead, each vocabulary token is treated as a distinct "source". The attribution measures how much a particular input token influences a downstream component.

- **Storage**: `embed_attr[target_layer]` of shape `(C_target, V)` where $V$ is vocabulary size
- **Both signed and absolute variants**
- **Accumulation**: uses `scatter_add_` to efficiently bin per-token attributions by token identity across all positions in the batch

### 6.3 Component $\to$ Output (Unembed Edges)

These edges connect components to the model's output logits. A key optimization avoids materializing the full `(V, C_source)` attribution matrix:

- **Storage**: `unembed_attr[source_layer]` of shape `(d_model, C_source)` --- stored in the **residual stream space** rather than vocabulary space
- **Only signed variant** (no absolute). The residual-space storage trick is incompatible with `attr_abs` because the absolute value is a nonlinear operation that cannot be factored through the linear projection $W_\text{unembed}$
- **Query-time projection**: When a caller requests the attribution of source component $s$ to output token $v$, the storage computes $W_\text{unembed}[:, v]^\top \cdot \text{unembed\_attr}[\text{source}][:, s]$ on the fly

This reduces both computation during harvesting (avoiding an $O(V \times d_\text{model} \times C)$ matmul per batch) and storage ($d_\text{model} \ll V$ typically).

### 6.4 Embedding $\to$ Output (Straight-Through Edges)

The direct path from input embeddings to output logits, bypassing all intermediate components.

- **Storage**: `embed_unembed_attr` of shape `(d_model, V)` --- also in residual space
- **Only signed variant**
- Same query-time projection as unembed edges

## 7. Multi-GPU Parallelism and Merging

### 7.1 Data-Parallel Sharding

The pipeline supports multi-GPU execution via data-parallel sharding. Each of $W$ GPU workers processes a disjoint subset of batches determined by:

```python
if batch_idx % world_size != rank:
    continue  # skip this batch
```

Each worker independently accumulates its own set of attribution sums and metadata. Workers do not communicate during harvesting.

### 7.2 Merge

After all workers complete, a CPU-only merge job combines the partial results. Because all stored quantities are **raw sums** (not averages or normalized values), merging is simple element-wise addition:

```python
for target, sources in other._regular_attr.items():
    for source, tensor in sources.items():
        merged._regular_attr[target][source] += tensor
```

All accumulators are summed: attribution tensors (regular, embed, unembed, and straight-through), CI sums, squared component activation sums, squared logit sums, and embedding token counts. The `n_tokens_processed` fields are also summed to obtain the total.

## 8. Normalization

Raw attribution sums are not directly comparable across components because:
1. Components that fire more frequently accumulate larger sums (source frequency bias)
2. Components with larger activation magnitudes produce larger gradients (target scale bias)

Normalization is applied **at query time**, not during accumulation, which preserves the ability to merge partial results and allows experimentation with different normalization schemes.

### 8.1 Normalization Formula

$$\text{normed}(s \to t) = \frac{\text{raw}(s \to t)}{\text{source\_denom}(s) \cdot \text{target\_rms}(t)}$$

**Source denominator** (puts attributions on a per-occurrence scale):
- For component sources: $\text{source\_denom}(s) = \sum_{\text{data}} \text{CI}(s)$ --- the total CI mass, analogous to the number of times the component was "active"
- For embedding sources: $\text{source\_denom}(v) = \text{count}(v)$ --- the number of times token $v$ appeared in the data

**Target RMS** (normalizes for target activation scale):
- For component targets: $\text{target\_rms}(t) = \sqrt{\frac{1}{N} \sum_{\text{data}} a_t^2}$ --- the root-mean-square component activation
- For output targets: $\text{target\_rms}(v) = \sqrt{\frac{1}{N} \sum_{\text{data}} \text{logit}_v^2}$ --- the root-mean-square logit for token $v$

A small epsilon ($10^{-10}$) is added to denominators to avoid division by zero for dead components or rare tokens.

## 9. Canonical Addressing

The neural network's concrete module paths (e.g., `transformer.h.0.mlp.c_fc`) are translated to **canonical addresses** (e.g., `0.mlp.up`) at the storage boundary. The `TransformerTopology` class provides this mapping via `target_to_canon()` (concrete $\to$ canonical).

This translation happens once in `AttributionHarvester.finalize()`, which applies `topology.target_to_canon` to every layer key in the accumulator dictionaries before constructing the `DatasetAttributionStorage`. All stored attribution data thus uses architecture-independent canonical names. This is essential because:
1. Different model architectures use different concrete path conventions
2. Downstream consumers (the web app, graph interpretation) operate on canonical addresses
3. It enables consistent querying across models

## 10. Query Interface

The `DatasetAttributionStorage` class provides two primary query methods:

### 10.1 `get_top_sources(target_key, k, sign, metric)`

Returns the top-$k$ source components that most strongly influence a given target component.

**Algorithm**:
1. Parse the target key (e.g., `"0.mlp.up:5"`) into layer and component index
2. For each source layer connected to this target:
   - Extract the raw attribution row `raw[target_idx, :]`
   - Normalize: `raw / source_denom / target_rms`
3. Concatenate all normalized source attributions into a single vector
4. Apply `torch.topk` with `largest=(sign == "positive")`
5. Filter to retain only values matching the requested sign
6. Map flat indices back to `(source_layer, component_idx)` pairs

For **output targets** (e.g., `"output:42"`), the raw attributions must first be projected from residual space: `w_unembed[:, token_id] @ unembed_attr[source_layer]`.

### 10.2 `get_top_targets(source_key, k, sign, metric)`

Returns the top-$k$ target components most strongly influenced by a given source.

The algorithm mirrors `get_top_sources` but iterates over target layers instead, extracting column vectors `raw[:, source_idx]` from each target's attribution matrix.

For **component sources** targeting the **output**, the projection `unembed_attr[:, source_idx] @ w_unembed` yields per-token attributions in vocabulary space.

## 11. Downstream Consumption: Graph Interpretation

The attribution graph serves as the backbone for **graph interpretation** (`spd/graph_interp/`), a three-phase LLM-based labeling pipeline that assigns human-readable descriptions to components using their graph context.

### 11.1 Output Pass (Late $\to$ Early)

Processing layers in reverse order, each component's prompt includes its **top-$k$ downstream targets** (from `get_top_targets`), along with any labels already assigned to those targets. This answers: *"What does this component contribute to?"*

### 11.2 Input Pass (Early $\to$ Late)

Processing layers in forward order, each component's prompt includes its **top-$k$ upstream sources** (from `get_top_sources`), along with labels from already-processed earlier layers. This answers: *"What triggers this component?"*

### 11.3 Unification

A third LLM call synthesizes the output and input labels into a single unified description per component.

### 11.4 Prompt Edge Storage

Each (component, related_component) pair used in prompts is stored as a **prompt edge** in an SQLite database, recording the attribution strength, the pass direction, and the related component's label at the time of prompting. This creates an auditable record of the graph context that informed each label.

In addition to attribution, each related component is enriched with **pointwise mutual information (PMI)** from a correlation storage (computed during harvest), measuring how often two components co-fire beyond what chance would predict. PMI is included in the LLM prompt context but is not persisted in the prompt edges table.

## 12. Computational Complexity

Let $L$ be the number of layers, $C$ the typical number of components per layer, $B$ the batch size, $S$ the sequence length, and $d$ the model dimension.

### 12.1 Per-Batch Cost

- **Forward passes**: 2 (one for CI under `no_grad`, one for component activations with gradients)
- **Backward passes per component target layer**: $2 \times C_\text{alive}$ (one for `attr`, one for `attr_abs`, per alive component)
- **Backward passes for the output target**: $d_\text{model}$ (one per residual dimension, signed metric only)
- **Total backward passes per batch**: $\approx 2 \sum_\ell C_{\ell,\text{alive}} + d_\text{model}$

The position-summation optimization reduces the component-target cost from $O(B \cdot S \cdot \sum C_\ell)$ backward passes to $O(\sum C_\ell)$ --- a factor of $B \cdot S$ improvement. The same optimization applies to the output target (residual summed over positions before backward).

### 12.2 Memory

Peak memory is dominated by the computation graph retained for `retain_graph=True` backward passes. The graph must be kept alive for all target components within a layer, then can be freed before processing the next target layer.

### 12.3 Storage

For a model with $L$ layers of $C$ components, vocabulary size $V$, and model dimension $d$:
- Regular edges: $O(L^2 \cdot C^2)$ (sparse --- only connected pairs)
- Embed edges: $O(L \cdot C \cdot V)$
- Unembed edges: $O(L \cdot C \cdot d)$ (not $O(L \cdot C \cdot V)$ due to residual-space storage)

## 13. Interpreting Attribution Edges: Direct Paths and the Detach Boundary

### 13.1 Intuition: Why Detach?

To understand what the attribution edges mean, it helps to first ask: what would happen *without* the detach mechanism --- if we simply computed `autograd.grad(component_C_activation, component_A_activation)` straight through the network?

**Without detach**, the gradient from C back to A flows through *every* computational path in the network, including paths that pass through intermediate component B's activations. The resulting `grad * activation` attribution for the A $\to$ C edge would include A's influence on C that is *mediated* by B: A fires, B reads A's output from the residual stream and fires differently as a result, and C then reads B's modified output. This indirect influence would be counted in the A $\to$ C edge, but it would *also* appear in the A $\to$ B and B $\to$ C edges. Every non-adjacent edge would overcount because multi-hop effects pile up inside it.

**With detach**, the gradient from C back to A is blocked at B's component boundary. It can only flow through the residual stream's skip connections and non-decomposed operations (like LayerNorm). So the A $\to$ C edge measures specifically: *how much does A's output, as it persists in the residual stream without being transformed by any intermediate component, directly affect C's input?* The indirect path through B is excluded from this edge --- it is captured separately by the A $\to$ B and B $\to$ C edges.

An analogy: imagine mapping influence in an organization. Without detach, asking "how does Alice influence Charlie?" gives you *total* influence --- including chains where Alice influences Bob, who reinterprets and passes it on to Charlie. Every edge would overcount because multi-hop effects are conflated with direct ones. With detach, each edge captures only the *direct* relationship: Alice's memos that Charlie reads himself, not the ones that Bob reads and reinterprets for Charlie. The indirect path is still represented in the graph, but decomposed into its constituent hops (Alice $\to$ Bob, Bob $\to$ Charlie).

This is why the detach-and-reattach mechanism exists: it ensures that attribution edges decompose the network's computation into **non-overlapping single-hop contributions**, where each edge measures influence through the residual stream without double-counting paths that pass through intermediate components.

**The role of `pre_detach` and `post_detach`.** Recall from Section 4 that each decomposed module stores two copies of the component activation at the detach boundary: `pre_detach` (connected to all upstream computation, used as the target/differentiation-origin in gradient computation) and `post_detach` (a fresh leaf tensor, used as the source that gradient is computed *with respect to*). Computing `grad(C.pre_detach, A.post_detach)` asks: *what is the local sensitivity of C's component activation to A's, through the residual stream only?* The detach at intermediate components ensures that the gradient captures only the direct residual-stream path --- intermediate components don't get to "react" to A's change in the gradient computation. (Section 13.3 discusses a subtle caveat: LayerNorm creates a weak second-order coupling even through detached components.)

**What attribution values mean.** A large $\text{attr}(A \to C)$ indicates that A's output direction ($U_A$) aligns well with C's input direction ($V_C$) through the residual stream, and that A fires strongly (weighted by CI) on the same data where this alignment matters. It is a measure of direct residual-stream influence: how much one component's output is read by another's input projection, in practice across the training distribution.

### 13.2 What the Detach Blocks (Technical Detail)

Recall that at every decomposed module, the computation graph is severed:

```
pre_detach = V^T @ x         # depends on all upstream computations
post_detach = pre_detach.detach().requires_grad_(True)  # NEW leaf
output = (post_detach * mask) @ U   # depends only on post_detach
```

When `autograd.grad(C.pre_detach, A.post_detach)` is computed, the gradient can only flow along paths where every intermediate tensor is part of the live computation graph. The detach at every intermediate component layer B creates a wall: gradient flows freely *forward* from `B.post_detach` through B's mask and $U$ matrix into the residual stream, but it cannot flow *backward* through `B.post_detach` to `B.pre_detach` and hence to A.

### 13.3 Tracing the Gradient Path in a Transformer

Consider a transformer with residual connections and three decomposed component layers A, B, C (e.g., MLP up-projections in blocks 0, 1, 2). A typical transformer block has the structure `x = x + attn(norm(x)); x = x + mlp(norm(x))`, so the residual stream evolves as:

$$r_1 = r_0 + f_A(\text{A.post\_detach})$$
$$r_2 = r_1 + g_1(r_1) + f_B(\text{B.post\_detach})$$
$$r_3 = r_2 + g_2(r_2) + f_C(\text{C.post\_detach})$$

Here $f_X(\text{X.post\_detach})$ denotes the output of a single decomposed module: `(post_detach * mask) @ U` (plus optional bias). This is a purely linear function of `post_detach` --- no activation function is applied within a decomposed module. The $g_i(r_i)$ terms denote non-decomposed operations that read from the residual and add back to it. These include LayerNorm/RMSNorm, attention sub-blocks (if attention is not decomposed), and crucially, **activation functions** (e.g., GELU) that sit between two decomposed modules within the same MLP. For example, in a GELU MLP, both `c_fc` (up-projection) and `down_proj` (down-projection) are decomposed, but the GELU nonlinearity between them is part of the target model's forward pass and is *not* decomposed --- it is a $g$ term that gradient flows through freely. C's `pre_detach` is $V_C^\top \cdot \text{Norm}(r_2)$, where Norm is the RMSNorm/LayerNorm applied before C's module.

An important subtlety in these equations: each component's `post_detach` is *numerically* a function of the residual stream that feeds it. B's activation is computed as $V_B^\top \cdot \text{Norm}(r_1)$, so `B.post_detach`'s value depends on $r_1$ (and hence on A's output). However, the `.detach()` call severs this dependency in the **computation graph** --- `B.post_detach` becomes a fresh leaf tensor. In the equations above, `B.post_detach` should therefore be read as "a value that was computed from $r_1$ but whose gradient connection to $r_1$ has been cut." The notation $f_B(\text{B.post\_detach})$ emphasizes the computation-graph view: as far as gradient flow is concerned, $f_B$ depends only on the detached leaf, not on $r_1$.

The gradient $\frac{\partial \text{C.pre\_detach}}{\partial \text{A.post\_detach}}$ flows backward:

1. Through $V_C^\top$ and the pre-C normalization
2. Into $r_2$, which depends on $r_1$ (via the additive residual connection) and $g_1(r_1)$
3. Through $r_1$, which depends on $r_0$ and $f_A(\text{A.post\_detach})$
4. Back to `A.post_detach` via the chain $f_A$

Crucially, $r_2$ also includes $f_B(\text{B.post\_detach})$. While `B.post_detach`'s *value* was computed from $r_1$, its *gradient connection* to $r_1$ (and hence to `A.post_detach`) has been severed by the detach. The gradient therefore does **not** flow backward through B's component computation to reach A. It can only flow through the residual stream's carry-forward of A's contribution (the $r_1$ term in $r_2 = r_1 + \ldots$), and through non-component operations $g_i$ that transform the residual.

Note that LayerNorm/RMSNorm introduces a subtle coupling: because normalization divides by the standard deviation of the full residual vector, the Jacobian $\frac{\partial \text{Norm}(r)}{\partial r}$ is not diagonal --- it depends on all elements of $r$, including contributions from B's (detached) output. This means the gradient from C to A is *modulated* by B's output (through the normalization statistics), even though it doesn't flow *through* B's component computation. This is a second-order effect: B's output changes the normalization scale, which changes how strongly A's contribution is read by C.

### 13.4 What Each Edge Measures

This means each attribution edge $\text{attr}(A \to C)$ measures A's influence on C through the **residual-stream path**, including:

- The **identity/skip connection** that carries A's output forward unchanged (modulo normalization)
- **Non-decomposed operations** between A and C (LayerNorm, attention mechanisms if they are not decomposed, activation functions between decomposed MLP projections within the same block)

But **excluding**:

- Paths that are mediated by intermediate component activations (e.g., A's output being read by B, transformed by B's components, and B's output then influencing C)

Each edge is therefore best understood as measuring the **direct residual-stream influence** of one component on another, where "direct" means "not mediated by any other decomposed component's activations". In a fully decomposed model, this is the path through the residual stream's additive skip connections and normalization layers only.

### 13.5 Non-Additivity of Paths

A natural question is whether the attribution from A to C equals the sum (or some other combination) of attributions through intermediate components:

$$\text{attr}(A \to C) \stackrel{?}{=} \sum_i f\bigl(\text{attr}(A \to B_i),\; \text{attr}(B_i \to C)\bigr)$$

This does **not** hold, for two independent reasons:

1. **Different gradient paths.** $\text{attr}(A \to C)$ measures the residual-stream path, while $\text{attr}(A \to B_i)$ and $\text{attr}(B_i \to C)$ each measure their own residual-stream paths. The composition of two residual-stream paths is not the same as the direct residual-stream path --- the former goes through B's component computation, the latter doesn't.

2. **Incompatible activation weights.** Even if the gradients could be composed, the attribution formula multiplies gradients by source activations and CI values. $\text{attr}(A \to B)$ uses $a_A \cdot \text{CI}_A$ while $\text{attr}(B \to C)$ uses $a_B \cdot \text{CI}_B$. There is no algebraic identity that relates their product to $\text{attr}(A \to C)$, which uses $a_A \cdot \text{CI}_A$ with a completely different gradient.

### 13.6 Implications for Graph Interpretation

This edge semantics has several practical consequences:

**The graph is not a flow network.** Unlike circuit-level analyses where edge weights compose multiplicatively along paths, the attribution graph's edges are independent measurements. The "importance" of a multi-hop path A $\to$ B $\to$ C cannot be computed from the individual edge weights. Each edge stands alone as a measure of direct residual-stream influence.

**Non-adjacent edges are meaningful.** An edge from layer 0 to layer 5 is not redundant with the chain of edges 0 $\to$ 1, 1 $\to$ 2, ..., 4 $\to$ 5. It captures a genuinely different quantity: how much layer 0's output persists in the residual stream all the way to layer 5's input without being transformed by any intermediate component.

**Strong non-adjacent edges indicate residual-stream persistence.** If $\text{attr}(A \to C)$ is large despite A and C being many layers apart, it means A's contribution to the residual stream survives normalization and other non-component operations across multiple layers and is directly read by C's input projection $V_C$. This is a meaningful structural finding: it identifies components whose output directions are preserved through the residual stream.

**Adjacent edges capture the most "mediated" influence.** The edge from B to C (adjacent layers) captures B's influence through the most direct residual-stream path. Since no intermediate component detach boundary intervenes, this edge includes the full gradient through whatever non-component operations exist between B and C (typically just a LayerNorm/RMSNorm). These edges tend to have the strongest attribution values.

**Attention creates cross-sequence edges.** When attention projections (Q, K, V, O) are decomposed, the attention mechanism sits between source projections (K, V) and the output projection (O) within the same block. The softmax attention operation is a non-decomposed operation that mixes information across sequence positions. This means a K/V component at position $p$ can influence the O projection at position $q$ --- the gradient flows through the attention weights. The system tracks these as `is_cross_seq` edges, distinguishing them from same-position residual-stream edges.

**The graph is a one-hop decomposition of the network.** Each edge decomposes the network's total computation into single-hop contributions through the residual stream. To reconstruct the full multi-hop influence of A on C, one would need to account for all paths: the direct residual path (captured by $\text{attr}(A \to C)$) plus all indirect paths through intermediate components (not captured by any single edge, and not equal to any simple combination of edges). The attribution graph thus provides a *local* view of component interactions rather than a *global* view of information flow.

## 14. Summary of Key Design Decisions

1. **Raw sums with deferred normalization.** All accumulators store unnormalized sums. This enables trivial merging of parallel workers (element-wise addition) and allows experimentation with normalization schemes without re-harvesting.

2. **Residual-space output storage.** Output-target attributions are stored in the $d_\text{model}$-dimensional residual space rather than the $V$-dimensional vocabulary space. The $W_\text{unembed}$ projection is applied at query time. This reduces both computation and storage by a factor of $V / d_\text{model}$.

3. **Position summation before backward.** Target activations are summed over the batch and sequence dimensions before computing gradients. This reduces the number of backward passes from $O(B \cdot S \cdot C)$ to $O(C)$ per target layer without changing the result, because gradient is a linear operator and the attributions are summed over positions anyway.

4. **Detach-and-reattach at component boundaries.** The computation graph is severed at each layer's component activations, creating clean "source" and "target" attachment points. This ensures that `autograd.grad(target_pre_detach, source_post_detach)` captures exactly the inter-layer gradient, enabling efficient per-component attribution without full Jacobian computation.

5. **CI weighting of source activations.** Source activations are multiplied by their causal importance values before accumulation. This focuses the attribution on components that are causally relevant on each datapoint, suppressing noise from components that the model has learned to mask out.

6. **Separate `attr_abs` backward passes.** The absolute-value metric requires its own backward pass because `.abs()` is applied per-position before summation, and the sign of each element varies across positions, preventing factorization. This doubles the backward pass count for component targets (but is not computed for output targets, where residual-space storage precludes the nonlinear absolute value).

7. **Gradient connectivity probing.** Rather than assuming a fixed graph topology, the system empirically discovers which layer pairs are connected by gradient flow using a single dummy forward-backward pass. This handles arbitrary model architectures including skip connections.

8. **Canonical addressing at the storage boundary.** Concrete module paths are translated to architecture-independent canonical names exactly once, at finalization time. This ensures consistent downstream consumption across different model families.
