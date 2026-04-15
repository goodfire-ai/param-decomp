# Analyzing Attention Behaviors with VPD Parameter Components

This document describes the analysis pipeline for writing paper sections about specific attention behaviors identified by VPD. It is based on the flow used for the "previous token behavior", "previous syntax boundary movement", and "copula verb attention modulation" sections in the paper.

## Overview

The goal is to identify an attention behavior implemented by parameter subcomponents, build evidence for what it does, and describe how it connects to the OV circuit. The pipeline has five stages:

1. **Static SIS analysis** — identify interesting QK pairs from weight-only data
2. **Component characterization** — understand what the components detect
3. **Dynamic multiprompt analysis** — verify data-dependent contributions across many prompts
4. **K-V coactivation and OV overlap** — connect QK routing to what value information is moved
5. **Cross-check and write-up** — verify claims, add figure references, draft the section

Each stage builds on the previous one. Do not skip stages or make claims from a later stage without verifying them against earlier stages.

## Prerequisites

- A completed VPD run (e.g., `s-55ea3f9b`)
- Harvest data (component summaries, correlations, autointerp labels)
- Pre-computed outputs from the analysis scripts (or the ability to run them)

The main run ID appears in script output paths as `out/<run_id>/`.

## Stage 1: Static SIS Analysis

**Script:** `spd/scripts/plot_qk_c_attention_contributions/plot_qk_c_attention_contributions.py`

**What it produces:** Z-scored Standardized Static Interaction Strength for all alive (Q, K) pairs across offsets and heads. Cached as NPZ files in `out/<run_id>/cache/layerN.npz`.

**How to use it:**

Load the cached NPZ:
```python
data = np.load('out/<run_id>/cache/layer1.npz')
W = data['W']  # (n_offsets, n_heads, n_q_alive, n_k_alive) — z-scored
q_alive = data['q_alive']
k_alive = data['k_alive']
```

Rank all pairs by peak |z-scored SIS| to find candidates:
```python
for qi in range(len(q_alive)):
    for ki in range(len(k_alive)):
        peak = np.max(np.abs(W[:, :, qi, ki]))
```

Look for pairs that show:
- **Head concentration** — strong in specific heads rather than distributed (contrast with Q.316 pairs)
- **Offset dependence** — varying interaction strength at different relative positions
- **Unusual patterns** — anything qualitatively different from the already-studied behaviors

**Interpreting the z-scored SIS sign:**

The SIS sign correction uses the mean component activation on **firing tokens** (tokens where CI > threshold), computed from the reservoir-sampled activation examples in the harvest DB. This ensures the sign reflects the typical activation direction when the component is causally important, not the global mean which is diluted by the vast majority of tokens where the component doesn't fire.

The z-scoring then subtracts the per-head mean and divides by std. The per-head means are typically small (~0.005–0.014 for layer 1), so the z-scored sign matches the pre-z sign for any non-trivial interaction.

For strong interactions, you can read the SIS sign directly: positive means the pair boosts the attention score when both components activate on their trigger tokens, negative means it reduces it. If you need to verify:
1. The pre-z SIS = sign_q × sign_k × ||V_q|| × ||V_k|| × (U_q^T R U_k), where sign_q/sign_k use the firing-filtered mean
2. The sign of the pre-z SIS equals the sign of the data-dependent contribution when both components activate in their typical direction on firing tokens
3. The script logs the global and firing means for each component at INFO level — check these if a sign seems surprising

**Note on old data:** If running against harvest data that predates the firing-filtered fix, or if `firing_component_activation` is not available, the script falls back to the global mean. In that case, the sign can be wrong for rare components whose global mean has the opposite sign from their activation on trigger tokens (e.g., K.485 had global mean +0.40 but firing mean -4.25).

**Paper figure reference:** `<ref>fig:attn_contrib_grid</ref>` for the SIS line plots.

## Stage 2: Component Characterization

**Data sources:**
- Harvest summary: `HarvestRepo.open_most_recent(run_id).get_summary()` — firing density, mean CI, mean activation
- Autointerp labels: `InterpRepo.open(run_id).get_interpretation(key)` — LLM-generated descriptions

**What to record for each component:**
- Firing density (how often CI > threshold)
- Mean activation and its sign (important for SIS interpretation)
- Autointerp label
- Whether it's always-active (density > 90%) or conditional

**Key distinction:** Subcomponent activation (`V^T h`) vs causal importance (CI). The interactive data and SIS use activations. CI determines when a component actually matters for the model's output. A component can have high activation but low CI (it interacts with the data but doesn't affect the output). The firing density from harvest reflects CI, not activation.

## Stage 3: Dynamic Multiprompt Analysis

**Script:** `spd/scripts/interactive_qk_contributions/compute_data.py`

**What it produces:** Per-prompt JSON files in `out/<run_id>/` containing:
- `W`: raw `U_q^T R U_k` (NOT sign-corrected, NOT V-norm-scaled, NOT z-scored)
- `q_acts`, `k_acts`: raw `V^T h` activations per token
- `alive_q`, `alive_k`: which component indices are alive
- `component_model_attn`: actual attention patterns from the component model

**How to compute data-dependent contributions:**
```python
# Contribution of pair (qi, ki) at offset, head h:
contrib = q_acts[query_pos][qi] * W[h, offset, qi, ki] * k_acts[key_pos][ki]
```

This is the exact term in the attention score equation: `(V_q^T x) (U_q^T R U_k) (V_k^T x')`.

**Relationship to the SIS:**
- SIS_pre_z = sign_q × sign_k × ||V_q|| × ||V_k|| × raw_W (where signs use firing-filtered means)
- Data contrib = q_act × raw_W × k_act
- When both components activate in their typical firing direction: sign(contrib) = sign(SIS_pre_z)
- On individual tokens, a component may activate opposite to its firing mean, which flips the contribution sign for that token. This is uncommon by definition (the firing mean reflects the typical direction on firing tokens)

**Analysis approach:**

Do NOT cherry-pick a single prompt. Scan all available prompts systematically:
```python
for prompt_idx in range(31):
    # Load prompt data
    # Find positions where the query component has high activation (e.g., > 4.0)
    # Compute total contribution per head across all K interactions
    # Record results
```

Then compute statistics: mean, % positive, range per head across ALL examples. The claim "Q.308 boosts H3 and H5" requires showing this holds in >90% of examples, not just one prompt.

**Checking multiple offsets:** Repeat the analysis at offsets (for example) 1, 2, 3, 5, 8, 12 to verify the pattern holds at different distances, not just previous-token.

**Paper figure references:** `<ref>fig:dynamic-1</ref>` or `<ref>fig:dynamic-2</ref>` for the interactive multiprompt viewer.

## Stage 4: K-V Coactivation and OV Overlap

This stage connects the QK circuit (which determines WHERE attention goes) to the OV circuit (which determines WHAT information is moved).

### 4a: K-V Coactivation (PKV)

**Data source:** Harvest correlation data:
```python
from spd.harvest.storage import CorrelationStorage
corr = CorrelationStorage.load(harvest_dir / 'component_correlations.pt')
# P(K active | V active) = count_ij[k_idx, v_idx] / count_i[v_idx]
```

This is a cross-module coactivation: K and V subcomponents are in different weight matrices but act at the same sequence position.

**Two directions, two questions:**
- `P(K|V) = count_ij[k, v] / count_i[v]` — "When V fires, how often does K also fire?" High P(K|V) means V almost always co-occurs with K. Useful for finding V components that are specific to K's context.
- `P(V|K) = count_ij[k, v] / count_i[k]` — "When K fires, how often does V also fire?" High P(V|K) means K's activations nearly always include this V component. Useful for finding the V components that carry the value information at K-selected positions.

Compute both. High P(K|V) with low P(V|K) means V is a rare specialist that almost only fires in K's context. High P(V|K) with low P(K|V) means V is broadly active and happens to also fire when K does.

**What it reveals:** The VALUE information available at key positions where a specific K component is active. For example, if K.485 (copula predictor) coactivates >90% with V.614/V.596/V.622 (existential construction detectors), this tells you that when attention routes to a "there" token, the value information being read specifically encodes "there-is" construction features.

**When this is useful vs not:**
- Useful when a *conditional* K component selects specific key positions (like K.119 selecting syntax boundaries, or K.485 selecting "there"/"it")
- NOT useful for always-active K components (K.329 is active everywhere, so P(K.329|V) ≈ 1 for all V)
- For behaviors mediated primarily by always-active keys, the PKV analysis is most informative for the OUTLIER cases (when conditional K components add extra contributions)

**Paper figure reference:** `<ref>fig:pkv</ref>`

### 4b: OV Subspace Overlap

**Script:** `spd/scripts/plot_wv_subspace_overlap/plot_wv_subspace_overlap.py`

**What it produces:** Data-weighted Frobenius cosine similarity between each head's W_OV = W_O @ W_V matrices, measuring:
- **Read similarity** (from M_read = W_OV^T W_OV): do two heads read from similar residual stream subspaces?
- **Write similarity** (from M_write = W_OV W_OV^T): do they write to similar subspaces?
- **Raw similarity**: do they compute similar overall transformations?

Pre-computed outputs are in `out/<run_id>/ov/`. The `--k_filter N` option re-weights the data PCA using only tokens where K component N is causally important.

**How to interpret:**
- High read + low write: heads extract similar information but project differently (complementary outputs)
- Low read + low write: heads operate on distinct subspaces entirely
- Compare to the random baseline (~0.14 for unweighted, ~0.56 for data-weighted, from paper appendix)

**Supporting script:** `spd/scripts/plot_wv_subspace_overlap/analyze_ov_subspace_semantics.py` lists the top V and O subcomponents aligned with each head's OV circuit, with autointerp labels.

**Paper figure references:** `<ref>fig:prev_tok_ov_overlap_k_329</ref>`, `<ref>fig:prev_tok_ov_overlap_k_119</ref>`

## Stage 5: Cross-check and Write-up

Before writing, verify every claim against the data:

**Claims that need verification:**
- "Component X activates on Y" → Check autointerp label AND activation patterns on multiple prompts
- "Pair has positive/negative SIS" → Verify z-scored sign matches pre-z sign (check per-head means are near zero)
- "Q.308 boosts H3" → Show this with aggregate statistics (% positive, N examples), not a single prompt
- "K.485 coactivates with V.614" → Verify the P(K|V) numbers from harvest correlation data
- "H3 and H5 read from similar subspaces" → Compute the actual Frobenius cosine similarity

**Claims to be careful about:**
- Do not conflate subcomponent activation with causal importance (CI). The interactive data has activations; CI comes from the CI function.
- Do not claim the z-scored SIS sign directly indicates the sign of the attention score contribution. It indicates the relative strength within a head. Verify with data-dependent contributions.
- Do not claim a behavior is "distributed across heads" or "localized to one head" based on weight norms alone. The weight norms show where a component has parameters; the SIS and dynamic analysis show where it has computational effect.

**Figure references to include:**
- `<ref>fig:qk_comp_weight_norm</ref>` — component weight norms per head
- `<ref>fig:attn_contrib_grid</ref>` — SIS line plots per head and offset
- `<ref>fig:dynamic-2</ref>` — interactive multiprompt attention viewer
- `<ref>fig:pkv</ref>` — P(K|V) coactivation
- `<ref>fig:prev_tok_ov_overlap_k_329</ref>` / `<ref>fig:prev_tok_ov_overlap_k_119</ref>` — OV overlap

**Structure of a behavior section (following Behavior 2's pattern):**
1. Identify the pair from the static SIS — what stands out and why
2. Characterize the components — what they detect, firing densities, autointerp labels
3. Note the conditional nature — when does this behavior engage?
4. Dynamic analysis on multiple prompts — systematic evidence for the behavior's effect
5. K-V coactivation — what value information gets moved
6. OV overlap — how the relevant heads' OV circuits relate to each other
7. Interpretation — what is the model using this for?
8. Contrast with other behaviors — what makes this behavior different

## Script Quick Reference

| Script | What it computes | Key outputs |
|--------|-----------------|-------------|
| `plot_qk_c_attention_contributions` | Z-scored SIS per head/offset | `cache/layerN.npz`, line plots |
| `plot_component_head_norms` | Weight norm per head per component | `layerN_qk_combined.png` |
| `detect_prev_token_heads` | Mean attention at offset 1 | `prev_token_scores_combined.png` |
| `attention_ablation_experiment` | Attention change when ablating components | Various ablation plots |
| `interactive_qk_contributions` | Per-prompt raw W, activations, attention | Per-prompt JSON files |
| `plot_wv_subspace_overlap` | OV Frobenius cosine similarity | `layerN_ov_paper_figure*.png` |
| `analyze_ov_subspace_semantics` | Top V/O components per head | Markdown tables |
| `rope_aware_qk` | RoPE-aware QK coefficients | (utility, no standalone output) |
| `collect_attention_patterns` | Raw attention patterns from model | (utility, no standalone output) |

All scripts are in `spd/scripts/` and take the wandb path as their first argument:
```bash
python -m spd.scripts.<script_module>.<script_name> wandb:goodfire/spd/runs/<run_id> [--layer N] [--k_filter M]
```

## Choosing Between Ablation and Multiprompt Dynamic Analysis

Behavior 1 (previous token) used ablation experiments to confirm the behavior: ablating Q.316 reduced previous-token attention. Behavior 2 (syntax boundary) and Behavior 3 (copula verb) skipped ablation and relied on dynamic multiprompt analysis + PKV + OV overlap.

**When ablation works well:**
- The component has high firing density (Q.316 at 97%) so its ablation has a visible effect on aggregate attention statistics
- The expected effect is clear and testable (e.g., "ablating this Q component should reduce attention to recent tokens")
- The ablation script is `spd/scripts/attention_ablation_experiment/attention_ablation_experiment.py`

**When multiprompt dynamic analysis is better:**
- The component has low firing density (Q.308 at 0.55%) — its ablation may not show up in aggregate attention statistics because it's only active on rare tokens
- The behavior is more subtle than "attention to X goes up/down" — e.g., head-selective amplification
- You want to understand the mechanism (which K interactions mediate the effect) rather than just confirming the effect exists

For most new behaviors beyond the obvious dominant ones, the Behavior 2 flow (multiprompt dynamic + PKV + OV) is more productive.

## Prompt Selection for Interactive Data

The precomputed interactive data (`spd/scripts/interactive_qk_contributions/out/<run_id>/`) covers 31 prompts from two sources:
- `spd/scripts/interactive_qk_contributions/handwritten_prompts.json` — curated prompts
- `spd/scripts/plot_prompt_attention/dataset_prompts.json` — sampled from training data

For common behaviors (Q.316-mediated, high firing density), 31 prompts typically provides enough examples. For rare behaviors (Q.308 at 0.55%), check how many examples the existing prompts yield before committing to the analysis. In our case, 31 prompts yielded 67 query positions with Q.308 activation > 4.0, which was sufficient.

If the existing prompts don't have enough examples, you can generate new interactive data:
```bash
python -m spd.scripts.interactive_qk_contributions.compute_data \
    wandb:goodfire/spd/runs/<run_id> \
    --prompts_file path/to/new_prompts.json
```

Choose prompts that are likely to exercise the behavior (e.g., sentences with copula verbs for Q.308). But also include control prompts where the component should NOT activate, to verify specificity.

## Lessons Learned

1. **Your section must be consistent with the paper figures.** The reader will see the z-scored SIS plot (`fig:attn_contrib_grid`). With the firing-filtered sign correction, the SIS sign should correctly reflect the contribution direction on trigger tokens. If you see a discrepancy between the SIS plot and your data-dependent analysis, check that the plot was generated with the firing-filtered fix (look for `firing_mean=` in the script's INFO logs). If the signs still disagree, investigate — don't paper over the discrepancy.

2. **Spot-check SIS signs against data-dependent contributions.** The firing-filtered sign correction handles the common case, but it's still good practice to verify a few key pairs by computing `q_act * raw_W * k_act` on specific prompts and confirming the sign matches the SIS.

3. **Aggregate before claiming.** A finding on one prompt is an anecdote. Sweep all available prompts and report N, mean, % positive, range. The copula verb analysis used 67 examples across 31 prompts.

4. **The always-active key (K.329) mediates most interactions.** For query components whose primary interaction is with K.329, the behavior is "amplification of existing attention patterns" rather than "routing to specific positions." This is qualitatively different from behaviors mediated by conditional key components.

5. **PKV is most informative for conditional K components.** P(K.329|V) ≈ 1 for all V since K.329 is always active. The PKV analysis adds insight only for conditional keys (K.119, K.218, K.485) where it reveals what value information is specifically available at the positions those keys select.

6. **Follow the Behavior 2 flow, not Behavior 1.** Behavior 1 used ablation experiments, which require a clear causal prediction (ablating Q.316 should reduce previous-token attention). For more subtle behaviors where the component has low firing density, the multiprompt dynamic analysis + PKV + OV overlap flow (Behavior 2) is more productive.
