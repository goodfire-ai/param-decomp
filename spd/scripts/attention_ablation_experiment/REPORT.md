# Attention Ablation Experiment — Technical Report

## Overview

The script runs two types of experiments:
1. **Head ablation**: Zero out specific attention heads in the *target model* (no SPD involved)
2. **Component ablation**: Zero out specific SPD parameter components within attention projections (q/k/v/o\_proj)

For each, it does a baseline forward pass and an ablated forward pass, captures attention patterns from both, and compares logits.

---

## Head Ablation

**Model used:** The raw pretrained `LlamaSimpleMLP` target model, loaded independently — no SPD model involved.

**Mechanism:** The `patched_attention_forward` context manager monkey-patches every `CausalSelfAttention.forward` method. The patched forward:

1. Calls `q_proj(x)`, `k_proj(x)`, `v_proj(x)` — these are the *original* target model linear layers (no component decomposition)
2. Reshapes to `(B, n_heads, T, head_dim)`, applies rotary embeddings, repeats KV for GQA
3. Computes explicit `softmax(QK^T / sqrt(d))` (flash attention is disabled)
4. Stores the attention pattern `(n_heads, T, T)` averaged over the batch dim
5. Computes `y = att @ v` giving `(B, n_heads, T, head_dim)`
6. **Ablation**: for specified `(layer, head)` pairs, zeros out `y[:, head, :, :]` (or at specific positions)
7. Reshapes and passes through `o_proj`

**Baseline pass**: patched forward with no ablations — captures patterns, produces logits.

**Ablated pass**: patched forward with head zeroing — captures patterns, produces logits.

**What "ablating a head" means here**: The head's QKV computation and attention still happens normally. What's zeroed is the head's *contribution to the output* — `att @ v` for that head is set to 0 before the `o_proj` linear layer. So from `o_proj`'s perspective, that head contributes nothing.

---

## Component Ablation — Deterministic Mode

**Models used:** The SPD `ComponentModel` wrapping the target model. `target_model = spd_model.target_model` — same instance.

**Mechanism:**

1. **Mask construction**: For every module in the SPD model, creates a mask tensor of shape `(batch, C)` where `C` is the number of components. Baseline: all ones. Ablated: all ones except the target component indices are set to 0.

2. These are wrapped via `make_mask_infos()` into `ComponentsMaskInfo` objects (containing `component_mask` and `routing_mask="all"`).

3. **Forward pass**: `spd_model(input_ids, mask_infos=...)` registers PyTorch forward hooks on each target module (e.g. `h.1.attn.q_proj`). During the forward:
   - The hook intercepts the module's input `x` and output
   - Instead of using the original module output, it calls `components(x, mask=component_mask)` which computes: **`output = sum_c(mask[c] * outer(U[c], V[:, c]) @ x)`** — i.e. the reconstructed output is a masked sum of rank-1 component contributions
   - With `routing_mask="all"`, *all positions* use the component reconstruction (not the original module)

4. The `patched_attention_forward` context manager is also active on `target_model` (same instance), so it captures attention patterns from within the attention block. The q/k/v projections fire through SPD's hooks (producing component-masked outputs), then the patched attention forward computes softmax attention manually.

**What "ablating a component" means here**: Setting `mask[c] = 0` removes that component's rank-1 contribution from the module's reconstructed weight matrix. With all-ones mask: `W_recon = sum_c(U[c] V[c]^T)`. With component `c` zeroed: `W_recon = sum_{i!=c}(U[i] V[i]^T)`.

---

## Component Ablation — Stochastic Mode

**Additional step**: Before masking, runs a forward pass with `cache_type="input"` to cache activations, then computes CI (causal importance) via `spd_model.calc_causal_importances()`.

**Mask formula** (from `calc_stochastic_component_mask_info`):
```
mask[c] = CI[c] + (1 - CI[c]) * random_source[c]
```
Where `random_source` is `torch.rand_like(ci)` for "continuous" sampling.

- If `CI[c] = 1`: mask is always 1 (component always fully active)
- If `CI[c] = 0`: mask is uniformly random in [0, 1)
- If `CI[c] = 0.5`: mask is in [0.5, 1)

**Ablation**: For the target components, CI is forced to 0 *before* sampling, so their masks become purely random rather than biased toward 1.

**Averaging**: Runs `n_mask_samples` (default 10) stochastic forward passes for both baseline and ablated. Averages logits and attention patterns across samples. This gives an expectation over the stochastic masking distribution.

**Important detail**: Both baseline and ablated use stochastic masks — so the baseline is *not* the all-ones deterministic case. The baseline has stochastic noise too, just with the original CI values. The comparison isolates the effect of removing CI for specific components.

---

## Component Ablation — Adversarial Mode

Runs PGD (projected gradient descent) to find worst-case masks:

1. Computes CI as in stochastic mode
2. Calls `pgd_masked_recon_loss_update()` which optimizes adversarial sources to maximize reconstruction loss subject to the CI constraint (`mask = CI + (1-CI) * source`)
3. Reports baseline and ablated PGD loss

**However**, for the attention pattern visualization and prediction table, it **falls back to deterministic masks**. The PGD losses are logged but the plots show the same thing as deterministic mode.

---

## Attention Pattern Capture

**What's stored**: `att.float().mean(dim=0).detach().cpu()` — shape `(n_heads, T, T)`, averaged over the batch dimension.

**Implication**: With `batch_size=1` (default), the batch mean is a no-op. With larger batches, patterns would be averaged across different sequences within the batch.

**Across samples**: Patterns are accumulated and divided by `n_samples` to get the mean.

---

## Metrics

**`frac_top1_changed`**: Total positions where `argmax(baseline) != argmax(ablated)`, divided by total positions across all samples. Uses first item in batch only.

**`mean_kl_divergence`**: `KL(baseline || ablated)` averaged over positions, then averaged over samples. Computed as:
```python
kl = F.kl_div(ablated_log_probs, baseline_log_probs.exp(), reduction="batchmean")
```
This is `sum_pos sum_vocab p_baseline * log(p_baseline / p_ablated) / n_positions`. Also uses first batch item only.

---

## Potential Validity Concerns

1. **Adversarial mode plots don't show adversarial masks**: The PGD loss is computed but plots/predictions use deterministic masks. This means the adversarial mode's attention patterns and prediction tables are identical to deterministic mode.

2. **Stochastic baseline also has noise**: The baseline in stochastic mode isn't the "true" model — it's a stochastic reconstruction. So the comparison measures "effect of removing this component from the stochastic ensemble" rather than "effect of removing this component from the faithful reconstruction."

3. **Batch dimension in patterns**: Attention patterns are averaged over batch dim. With `batch_size=1` this is fine. With larger batches, different sequences' patterns would be mixed.

4. **KL direction**: `KL(baseline || ablated)` — measures how surprised baseline would be by ablated. If ablated is more uniform, KL can be moderate even if top-1 changes a lot.

5. **Head ablation zeros post-attention, not pre-attention**: The head still computes QKV and attention weights. Only its contribution to the residual stream (via `o_proj`) is zeroed. This is a standard choice (same as what TransformerLens does) but worth noting.

6. **Component ablation in deterministic mode uses all-ones baseline**: This means the baseline is the SPD reconstruction `W = sum_c(U[c]V[c]^T)`, not the original target model weights. If the reconstruction isn't perfect, the baseline already differs from the original model.
