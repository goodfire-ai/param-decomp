# Two Programs in One Layer: How Distributed Components Trade Off Local Attention and Induction

**Model:** 4-layer LlamaSimpleMLP (6 heads/layer, 768-dim) trained on The Pile
**Run:** `s-55ea3f9b` | **Layer analyzed:** Layer 2 | **Eval data:** Pile validation split

---

## The Puzzle

Layer 2 of this model contains an **induction head** — H4 (score 0.676) attends to the token *after* previous occurrences of the current token, implementing a copying mechanism central to in-context learning. But the same layer's Q and K weight matrices also produce content-dependent attention patterns in other heads that have nothing to do with induction. What is going on?

SPD decomposes those shared weight matrices into components and reveals two independent programs coexisting in the same parameters. They use different components, target different heads, and compete for influence over the model's predictions.

---

## 1. Identifying the Components

### 1.1 Most components are dead; a few are conditionally active

SPD decomposes Layer 2's Q projection into 512 components and K projection into 512 components. Each component has a per-position **causal importance** (CI) value between 0 and 1, measuring how much that component contributes to the output at that position.

Most components are dead (CI near 0). But 14 Q components and 14 K components have moderate mean CI (0.1--0.7), making Layer 2 uniquely rich compared to other layers. Layer 1, for instance, has only 2 non-dead components (Q316/K329, both always-on at CI~0.9).

Critically, these moderate components are **bimodal at the position level**: at any given position, a component's CI is near 0 (OFF) or near 1 (ON). The "moderate" mean CI reflects how often the component switches on, not a graded activation strength.

![CI bimodality](out/article_ci_bimodality.png)
*Figure 1: Per-position CI distributions for four key components. Each is bimodal — mostly 0 or 1 — with the mean CI reflecting the ON fraction. Q335 is ON at 60% of positions; Q270 at only 19%.*

### 1.2 What makes a component switch on?

We examine which tokens co-occur with each component being ON vs OFF:

![Token evidence](out/article_token_evidence.png)
*Figure 2: Most frequent tokens when each component is ON (green) vs OFF (red). Q335 and K206 are OFF on LaTeX/math tokens (`mathcal`, `operatorname`) and ON on everything else (including JSON and natural language). Q270 is more selective — ON for content words (`game`, `other`, `character`), OFF for formatting of all kinds.*

The pattern: **Q335 and K206 switch OFF specifically for LaTeX/math markup** (`mathcal`, `operatorname`, `}}^{{\\`) and stay ON for most other content — including JSON, natural language, and general code. Q270 is more selective: it activates on content words in narrative/research text (`game`, `other`, `character`) and is OFF on formatting of all kinds. We call these **content-dependent** components because their activation depends on what kind of text the model is processing, not on positional features. (The label is imperfect — Q335 and K206 are better described as "anti-LaTeX" than as "natural language detectors." We keep "content-dependent" as shorthand.)

### 1.3 Two separate populations

We can identify which components drive different behaviors by correlating their CI with attention patterns. This reveals two non-overlapping groups:

- **Content-dependent modulators**: Q335, K206, Q270, K224, K347, Q279 — their CIs correlate with attention at specific offsets in H0/H5 (details in Section 2)
- **Induction drivers**: Q195, Q499, Q259, K166, K5 — their CIs correlate with H4's attention to induction targets (details in Section 3)

![Component landscape](out/article_component_landscape.png)
*Figure 3: All Q components in Layer 2 by mean CI. Red: content-dependent modulators. Blue: induction drivers. Gray: other (mostly dead/low). The two populations are distinct.*

---

## 2. The Focus Dial: Content Controls Where H0 and H5 Attend

### 2.1 Three modes of attention

For each component, we correlate its per-position CI with each head's attention weight at each offset (tokens back), across 148,800 positions. This reveals that the content-dependent components modulate attention in **specific heads at specific offsets**:

![CI-attention correlation heatmaps](out/continuous_all_components_summary.png)
*Figure 4: Pearson r between component CI and attention at each (head, offset). Red = higher CI means more attention there. Q335 and K206 are red at offset 0 in H0/H5 (promoting self/nearby). Q270 is blue at offset 0 and red at offsets 2--8 (promoting broad context). The H4 row is uniformly blank.*

Three modes emerge:

- **Local mode** (Q335 + K206, ON at ~60% and ~53% of positions respectively): Concentrates H0/H5 attention on self and immediately preceding tokens. K206 at H0 offset 0: r=+0.46.
- **Broad mode** (Q270, ON at ~19% of positions): Shifts H0/H5 attention from self to a wider window of 2--8 tokens back. Q270 at H0 offset 0: r=-0.26; at offset 3: r=+0.30.
- **Technical mode** (K224, ON at ~15% of positions): Promotes self-attention in H3 and previous-token in H0.

Q335/K206 and Q270 act as **functional antagonists** on the same two heads — they push H0/H5 attention in opposite directions depending on content:

![Offset profiles by Q270 state](out/cross_layer_regime_comparison.png)
*Figure 5: Attention offset profiles conditioned on Q270 activation. When Q270 is ON (blue, research text), H0/H5 spread attention broadly. When OFF (red), attention concentrates locally. L1 patterns (top row) are stable regardless — the content-dependent routing is specific to L2.*

### 2.2 Quantifying the effect

A linear regression of attention at each (head, offset) on the four main component CIs (Q335, Q270, K206, K224) gives the variance explained (R-squared):

![Variance explained by head](out/variance_explained_by_head.png)
*Figure 6: Joint R-squared (offsets 0--5) by head. Four content-dependent components explain 73% of H0's and 45% of H5's nearby attention variance, and 0% of H4's.*

| Head | Joint R-squared | Role |
|------|-----------------|------|
| **H0** | **0.73** | Heavily modulated |
| **H5** | **0.45** | Heavily modulated |
| H3 | 0.14 | Moderate (K224) |
| H1 | 0.13 | Moderate (K347) |
| H2 | 0.06 | Weakly modulated |
| **H4** | **0.00** | Independent |

### 2.3 Head-specific K component targeting

Each K component primarily targets a different head:

| K Component | Primary Head | Evidence |
|-------------|-------------|---------|
| K206 | H0, H5 | r=+0.46 at H0 self; ablation: -0.175 |
| K224 | H3, H0 | r=+0.25 with H0 prev-tok; R²=0.10 at H3 |
| K347 | H1 | r=+0.52 with H1 self-attention |

The model distributes content-dependent control across heads through dedicated K-component modulators. Most non-induction heads have at least one; H2 is the exception (R²=0.06, no single dominant modulator).

---

## 3. H4 Runs a Separate Induction Program

### 3.1 Different components drive induction

At 85,594 repeated-token events, we correlate each component's CI with H4's attention to the induction target. The top-correlated components are entirely different from the content-dependent modulators:

![Induction driver correlations](out/induction_drivers_correlation.png)
*Figure 7: Top Q and K components correlated with H4 induction attention. Red bars mark the content-dependent modulators — none appear in the top ranks.*

### 3.2 Decomposing a single position

At a specific induction position, we decompose the pre-softmax attention logit into contributions from every (Q_component, K_component) pair, then categorize each pair:

![Category decomposition](out/decompose_category_comparison.png)
*Figure 8: Summed QK pair contributions at the induction target vs previous token, by category. Blue = both components are outside the content-dependent set; other colors = at least one component is in the content-dependent set. H4's +12 logit is almost entirely blue (non-content-dependent pairs). H0/H5 produce negative logits at both positions using content-dependent pairs.*

H4 produces a +12 logit at the induction target via non-content-dependent pairs (Q203xK373: +2.86, Q432xK185: +1.88, Q202xK308: +1.71). H0/H5 produce negative logits at both the induction target and previous token — they attend elsewhere, directed by the content-dependent focus dial. H2 serves as a weaker backup induction head (r=0.51 with H4 across events).

---

## 4. The Competition

The two programs use different components and target different heads. But they share the same Q and K projection weight matrices — when a content-dependent component like Q335 contributes its `V_c @ U_c` to the Q projection, it changes the query vectors for *all* heads, including H4. Even though Q335's CI does not predict H4's attention *variation* across positions (R-squared = 0, Section 2.2), the component's weight contribution is always present in the shared matrix, and removing it changes the baseline query vectors that H4 uses everywhere. We demonstrate this with causal ablation — subtracting a component's weight contribution from the projection matrix and recomputing attention.

### 4.1 Double dissociation

![Causal induction summary](out/causal_induction_summary.png)
*Figure 9: Causal effects on H4 induction (blue) and H0 local attention (red), measured at 43,505 repeated-token positions. Removing Q335 (content-dependent) frees +0.055 for H4. Removing induction components costs H4 -0.039.*

| Ablation | H4 induction | H0 local attention | Interpretation |
|----------|-------------|-------------------|----------------|
| -Q335 (content-dep.) | **+0.055** | -0.163 | Less competition, more induction |
| -Top 5 induction Q | **-0.039** | ~0 | Less signal, less induction |
| -Q270 (content-dep.) | -0.004 | -0.020 | Minimal effect |

Removing Q335 boosts H4 induction by +0.055 (from 0.124 to 0.179). Q335 computes nothing related to induction — it competes by making H0/H5 louder in the shared Q projection matrix.

### 4.2 Visible in natural variation

The competition is also visible in observational data (without ablation, so potentially confounded by content type). At positions where K206 happens to be active at the induction target's key position (63,176 events), H4 induction averages 0.115; when K206 is off (12,434 events), it averages 0.163 — a difference of 0.048.

At the extremes: a code position where all content-dependent components are inactive shows H4 induction = 0.99. A natural language position where K206 and Q335 are both fully active shows H4 induction = 0.01.

![Induction exemplar distribution](out/induction_exemplar_distribution.png)
*Figure 10: H4 induction weight distribution by K206 state at key. K206 OFF: ~80% of events above 0.05. K206 ON: only ~40%.*

---

## 5. What Matters for Prediction

The local-focus mode is essential for the model's predictions. The broad-context mode is not.

| Ablation | Loss increase |
|----------|-------------|
| -Q335 (local focus) | +16.4% |
| -K206 (local focus) | +11.7% |
| -Q335 and -K206 | **+22.4%** |
| -Q270 (broad context) | **+0.6%** |

![Downstream loss comparison](out/downstream_loss_comparison.png)
*Figure 11: Prediction loss under ablation. The local-focus components are critical; the broad-context mode is dispensable for average loss.*

Removing the local-focus components together increases cross-entropy loss by 22%. Q270 — which creates clearly visible attention pattern changes — contributes almost nothing to prediction quality on this dataset. The local-focus components are doing real work for prediction, not just rearranging attention cosmetically. Whether the competition with induction (Section 4) is specifically beneficial — or just a side effect of packing two programs into shared weight matrices — remains an open question.

---

## 6. Value Components: What Gets Transported

The Q/K analysis above describes *where* heads attend. But what information do they transport? The V (value) projection determines what each head reads from attended positions, and the O (output) projection determines what it writes to the residual stream.

### 6.1 V components are mostly low-activity

Unlike Q/K, which have 14 moderate components each, the V projection has only **1 moderate component** (V22, CI=0.11) and 243 low-activity components. The O projection has 1 always-on (O735, CI=0.95) and 4 moderate. Most value computation in Layer 2 is carried by low-activity components that are always weakly contributing, rather than by a few conditionally-active ones.

![V/O CI distributions](out/value_ci_distribution.png)
*Figure 12: CI distributions for V and O components. Unlike Q/K, these are mostly low-activity rather than bimodal.*

### 6.2 V components weakly co-activate with Q/K content-dependent components

![V vs Q/K correlation](out/value_vs_qk_correlation.png)
*Figure 13: Correlation between top V component CIs and Q/K content-dependent component CIs. The strongest correlations are with K206 (r up to +0.22), suggesting some V components preferentially activate when the local-focus mode is engaged.*

Several V components (V928, V600, V363, V397) positively correlate with both Q335 (r~+0.15) and K206 (r~+0.17-0.22). V22 (the only moderate V component) anti-correlates with both Q335 (r=-0.22) and Q270 (r=-0.17) — it activates on the code/LaTeX content that the content-dependent Q/K components avoid.

The weak correlations (|r| < 0.22) suggest that **the content-dependent routing in Q/K and the information transported via V are partially but not tightly coupled**. The attention routing (where to look) is more content-dependent than the value extraction (what to read). This has an architectural implication: the focus dial primarily controls *which positions* contribute to the residual stream update, while the *kind of information* read from those positions is relatively constant. The model decides where to look based on content, but reads similar features from wherever it looks.

### 6.3 Per-head V component norms

![V/O component norms](out/value_component_norms.png)
*Figure 14: Per-head weight norms for top V and O components. Like Q/K components, V components span multiple heads rather than being localized.*

---

## 7. Context: Layer 1's Always-On Previous-Token Attention

Layer 1 provides a useful contrast. Its two main components (Q316, K329) are **always on** (CI ~0.9) and implement uniform previous-token attention across all heads — H1 in Layer 1 attends to the previous token with weight 0.62 regardless of content. Layer 2's content-dependent components are conditionally active (bimodal CI) and modulate specific heads.

![L1 vs L2 contrast](out/l1_vs_l2_contrast.png)
*Figure 15: L1 vs L2 contrast. (A,B) L1 CIs peak near 1; L2 CIs are bimodal. (D,E) L1 attention is stable across content types; L2 shifts.*

The layers are linked: L1 K329's CI predicts L2 K206 (r=+0.34) and anti-predicts L2 Q270 (r=-0.25). Content properties that influence L1's behavior also prime which L2 mode activates.

---

## Summary

SPD decomposes Layer 2's Q/K projections into components that reveal two independent programs sharing the same weight matrices:

1. **A content-dependent focus dial** (Q335, K206, Q270, and others) modulates H0 and H5 based on content type — concentrating or broadening their attention. It explains 73% of H0's attention variance and is essential for predictions (+22% loss when removed). Each K component targets a specific head.

2. **An induction program** (Q195, Q499, K166, and others) drives token-copying in H4, with H2 as a backup. It is causally necessary (top-5 Q ablation: -0.039) and completely independent of the content-dependent components (R-squared = 0).

3. **The two programs compete.** Removing Q335 causally boosts H4 induction by +0.055. In natural variation, K206 activation swings H4 from 0.99 to 0.01 induction. This competition is not wasteful — local attention improves predictions (+22% loss when removed).

4. **V components are less content-dependent.** Unlike Q/K, the value projection has few conditionally-active components. The content-dependent routing (where to look) is more modulated than the value extraction (what to read).

This structure is invisible at the head level — every component spans all six heads in its weight norms (Figure 15, panel C). The head-specific effects emerge through coordination between components, visible only through decomposition.

---

## Appendix A: Oddities and Unresolved Questions

### A.1 K206 weight ablation hurts H4 induction (-0.023)

In the causal ablation (Section 4.1), removing K206 *reduces* H4 induction by -0.023. This is the opposite of what the conditional analysis predicts (K206 ON at key positions suppresses H4 by -0.048). The resolution: weight-level ablation removes K206's contribution from ALL positions, including positions where K206 contributes positively to the K projection that H4 uses for induction targets. The conditional analysis compares positions where K206 is naturally ON vs OFF, while the ablation changes the weight matrix globally. These measure different things.

### A.2 Induction K components are dispensable

Ablating the top 5 induction-correlated K components (K166, K5, K251, K55, K1) causes H4 induction to *increase* by +0.004 — they appear dispensable. This contrasts with the top 5 induction Q components, which cause -0.039 when ablated. Possible explanation: the induction pattern is primarily carried by specific Q vectors that can find appropriate K representations across many K components, while K components are more redundant — there is no single K component whose removal disrupts induction.

### A.3 Q270 is dispensable for loss (+0.6%)

Q270 creates clearly visible attention pattern changes (shifting H0/H5 from local to broad context) but removing it barely affects prediction loss (+0.6%). This could mean: (a) the broad-context information is redundant with information already in the residual stream from Layer 1, (b) the model rarely needs broad context for next-token prediction on this dataset, or (c) other components partially compensate when Q270 is removed. We have not distinguished between these explanations.

### A.4 Q335 is ON at 60% of positions, not "~70%"

The per-position CI analysis (Figure 1) shows Q335 ON (>0.9) at 60% of positions, with 17% in a "mid" range (0.1-0.9). The "~70%" figure cited for Q335's activation rate in the exploration log reflects the mean CI (0.70) rather than the ON fraction. For K206, the ON fraction is 53% with 27% mid. The mid-range positions represent cases where the component is partially active, which we did not deeply investigate.

### A.5 V components are weakly structured

The V projection has far fewer conditionally-active components than Q/K (1 moderate vs 14 each). The top V component correlations with Q/K content-dependent components are weak (|r| < 0.22). This suggests the model's content-dependent behavior is primarily expressed through *where it attends* (Q/K) rather than *what it reads* (V). This is architecturally interesting — it means the attention routing is the primary site of content-dependent computation in this layer, with value extraction being more generic.

### A.6 Cross-layer CI correlations may be confounded

The cross-layer correlations (L1 K329 → L2 K206: r=+0.34) could reflect genuine causal influence through the residual stream, or they could reflect shared sensitivity to the same content properties (both components respond to "natural language vs code/LaTeX" in the input). We have not disentangled these with a controlled experiment (e.g., ablating L1 and measuring L2 CI changes).

---

## Methods

CI values: `ComponentModel.calc_causal_importances(sampling="continuous").lower_leaky`. Attention: `collect_attention_patterns_with_logits` on the target model. Causal ablation: subtract `V_c @ U_c` from projection weights. QK decomposition: RoPE-aware weight dot products via `compute_qk_rope_coefficients`. Correlations: Pearson r. R-squared: sklearn linear regression. Token analysis: `AutoTokenizer` decoding of positions where CI > 0.5 vs CI < 0.1. N = 43,505--148,800 depending on analysis.

20 analysis scripts and 55+ output plots in `exploration/`.
