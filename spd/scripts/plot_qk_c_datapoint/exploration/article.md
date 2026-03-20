# Two Programs in One Layer: How Distributed Components Trade Off Local Attention and Induction

**Model:** 4-layer LlamaSimpleMLP (6 heads/layer) trained on SimpleStories
**Run:** `s-55ea3f9b` | **Layer analyzed:** Layer 2

---

## The Puzzle

Layer 2 of this model contains an **induction head** — H4 (score 0.676) attends to the token *after* previous occurrences of the current token, implementing the copying mechanism central to in-context learning. But the same layer's Q and K weight matrices also produce content-dependent attention patterns in other heads that have nothing to do with induction. What is going on?

SPD decomposes those shared weight matrices into components and reveals two independent programs coexisting in the same parameters. They use different components, target different heads, and compete for influence over the model's predictions.

---

## 1. A Focus Dial in H0 and H5

We start with the most striking quantitative finding. Four components in Layer 2's Q/K projections — which we call the **semantic modulators** — jointly explain **73% of H0's** and **45% of H5's** nearby-offset attention variance, while explaining **0% of H4's**.

![Variance explained by head](out/variance_explained_by_head.png)
*Figure 1: R-squared of attention (offsets 0--5) regressed on four semantic component CIs. The semantic components dominate H0 and H5 but have zero explanatory power over H4.*

These four components implement a **focus dial** that controls whether H0 and H5 attend locally or broadly, depending on content:

- **Q335** and **K206** (active ~70% of positions, on most natural language) concentrate H0/H5 attention on self and immediately preceding tokens. K206 alone explains 21% of H0's self-attention variance.
- **Q270** (active ~24%, on research/technical text) does the opposite: it broadens H0/H5 attention to a wider context window of 2--8 tokens back, while suppressing self-attention.

Q335/K206 and Q270 are functional antagonists. They modulate the same two heads in opposite directions:

![CI-attention correlation heatmaps](out/continuous_all_components_summary.png)
*Figure 2: Correlation between each component's CI and attention at each (head, offset). Q335 and K206 promote nearby attention in H0/H5 (red at small offsets). Q270 promotes broad attention (red at offsets 2--8, blue at offset 0). The H4 row is blank for all of them.*

The remaining K components target other heads: K224 targets H3 (self-attention, r=+0.10) and H0 (previous-token, r=+0.25), K347 targets H1 (self-attention, r=+0.52), and Q279 promotes H3 self-attention on social content. Each non-induction head has at least one dedicated modulator.

---

## 2. H4 Runs a Separate Induction Program

While the semantic modulators control H0, H1, H3, and H5, H4 ignores them entirely. Its induction behavior is driven by a completely different set of components.

We identify these by correlating each component's CI with H4's attention to induction targets at 85,594 repeated-token events. The top-correlated components (Q195, Q499, Q259, K166, K5, K251) have zero overlap with the semantic modulators:

![Induction driver correlations](out/induction_drivers_correlation.png)
*Figure 3: Top components correlated with H4 induction. Red bars mark semantic modulators — none appear in the top ranks.*

At the datapoint level, we can directly see the separation. Decomposing the attention logit at an induction position into contributions from every (Q, K) component pair:

![Category decomposition](out/decompose_category_comparison.png)
*Figure 4: Component pair contributions at the induction target vs. previous token, categorized as semantic (red/orange/green) or non-semantic (blue). H4's induction logit (+12) is almost entirely non-semantic. H0/H5 produce negative logits at both positions using semantic pairs — they attend elsewhere.*

H4 produces a +12 logit at the induction target, driven by non-semantic pairs like Q203xK373 and Q432xK185. H0 and H5 produce negative logits at *both* the induction target and previous token — their attention is directed to self/nearby by the semantic focus dial. H2 serves as a weaker backup induction head (r=0.51 with H4 across events).

---

## 3. The Competition

The two programs are functionally independent — different components, different heads. But they share weight matrices, and empirically, they compete: strengthening one weakens the other.

We show this with **causal ablation** — subtracting a component's weight contribution (`V_c @ U_c`) from the Q or K projection matrix and recomputing attention.

![Causal ablation detail](out/causal_ablation_h0_h5_detail.png)
*Figure 5: H0/H5 attention profiles under ablation. Removing Q335 or K206 collapses nearby attention. The baseline (black) shows how much of these heads' behavior depends on these two components.*

The ablation results reveal a **double dissociation**. Removing semantic components *helps* H4 induction (by removing competition). Removing induction components *hurts* it (by removing signal):

![Causal induction summary](out/causal_induction_summary.png)
*Figure 6: Causal effects on H4 induction (blue) and H0 local attention (red) at 43,505 repeated-token positions.*

| Ablation | H4 induction | H0 local attention | Interpretation |
|----------|-------------|-------------------|----------------|
| -Q335 (semantic) | **+0.055** | -0.163 | Less competition, more induction |
| -Top 5 induction Q | **-0.039** | ~0 | Less signal, less induction |
| -Q270 (semantic) | -0.004 | -0.020 | Minimal effect on either |

The strongest single result: removing Q335 boosts H4 induction by +0.055 (from 0.124 to 0.179). Q335 computes nothing related to induction — it competes by making H0/H5 louder.

The competition is also visible in natural variation, without ablation. At positions where K206 happens to be active at the induction target's key position, H4 induction drops by 0.048 on average (0.163 to 0.115 across 75,610 events). At the extremes: a code/markup position where all semantic components are inactive shows H4 induction = 0.99; a natural language position where K206 and Q335 are both fully active shows H4 induction = 0.01.

![Induction exemplar distribution](out/induction_exemplar_distribution.png)
*Figure 7: H4 induction weight distribution by K206 state at the key position. K206 OFF: ~80% of events above 0.05. K206 ON: only ~40%.*

---

## 4. What Matters for Prediction

The semantic focus dial is not cosmetic — it is essential for the model's predictions:

| Ablation | Loss increase |
|----------|-------------|
| -Q335 (local focus) | +16.4% |
| -K206 (local focus) | +11.7% |
| -Q335 and -K206 | **+22.4%** |
| -Q270 (broad context) | **+0.6%** |

![Downstream loss comparison](out/downstream_loss_comparison.png)
*Figure 8: Prediction loss under ablation. Local-focus components are critical; the broad-context mode is dispensable.*

Removing the local-focus components together increases cross-entropy loss by 22%. But Q270 — which creates clearly visible changes in attention patterns — contributes almost nothing to prediction quality. On this dataset, the model relies on local attention for next-token prediction. This also means the induction-suppressing competition from the focus dial (Section 3) is not wasteful — it reflects the model prioritizing local context over token copying, a trade-off that improves predictions overall.

---

## 5. Context: How This Differs from Layer 1

Layer 1 provides a useful contrast. Its two main components (Q316, K329) are **always on** (CI ~0.9) and implement uniform previous-token attention across all heads — H1 in Layer 1 attends to the previous token with weight 0.62 regardless of content. Layer 2's semantic components are conditionally active (bimodal CI: mostly 0 or 1 at each position) and modulate specific heads.

![L1 vs L2 contrast](out/l1_vs_l2_contrast.png)
*Figure 9: L1 (always-on, stable across content) vs L2 (conditional, content-dependent). Panels D/E show that L1 attention profiles are identical across content types while L2 profiles shift with Q270 activation.*

The layers are linked through the residual stream: L1 K329's CI predicts L2 K206 (r=+0.34) and anti-predicts L2 Q270 (r=-0.25), suggesting that content properties influence both layers' routing.

---

## Summary

SPD decomposes Layer 2's Q/K projections into components that reveal two independent programs sharing the same weight matrices:

1. **A semantic focus dial** (Q335, K206, Q270, and others) modulates H0 and H5 based on content type — concentrating or broadening their attention. It explains 73% of H0's attention variance and is essential for predictions (+22% loss when removed). Each K component targets a specific head.

2. **An induction program** (Q195, Q499, K166, and others) drives token-copying in H4, with H2 as a backup. It is causally necessary (top-5 Q ablation: -0.039) and completely independent of the semantic components (R-squared = 0).

3. **The two programs compete.** Removing the local-focus component Q335 causally boosts H4 induction by +0.055. In natural variation, K206 activation at key positions swings H4 from near-perfect induction (0.99) to near-zero (0.01). This competition is not wasteful — local attention is what the model needs for prediction (+22% loss when removed).

This structure is invisible at the head level — every component spans all six heads in its weight norms. The head-specific effects emerge through coordination, visible only through decomposition.

---

## Methods

CI values: `ComponentModel.calc_causal_importances(sampling="continuous").lower_leaky`. Attention: `collect_attention_patterns_with_logits` on the target model. Ablation: subtract `V_c @ U_c` from projection weights. QK decomposition: RoPE-aware weight dot products. All correlations: Pearson r. R-squared: sklearn linear regression. N = 43,505--148,800 depending on analysis. 18 scripts, 51 plots in `exploration/`.
