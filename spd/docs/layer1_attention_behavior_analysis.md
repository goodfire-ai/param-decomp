# Layer 1 Attention Behavior Analysis

This document presents an analysis of attention behaviors in layer 1 of the VPD-decomposed 67M-parameter language model, following the methodology described in the VPD paper. Where the paper studied two behaviors in detail — previous token behavior (Q.316/K.329) and previous syntax boundary movement (Q.316/K.119) — this analysis extends coverage to additional behaviors identified by the decomposition.

**Run:** `s-55ea3f9b`

## Summary of Layer 1 Attention Decomposition

VPD identifies 5 alive Q subcomponents and 13 alive K subcomponents in layer 1 (at mean CI cutoff 0.001). The model has 6 attention heads per layer, each with head dimension 128 and model dimension 768.

### Q Subcomponents

| Component | Mean CI | Firing Act | Label |
|-----------|---------|------------|-------|
| Q.316 | 0.857 | +27.4 | always fires / bias component |
| Q.308 | 0.004 | +0.64 | existence and state verbs (is, was, there are/is) |
| Q.53 | 0.003 | +0.77 | domain-specific technical and academic terms |
| Q.149 | 0.002 | +0.39 | common punctuation and stopwords |
| Q.497 | 0.002 | -0.07 | repeating identical tokens or characters |

Q.316 dominates, with a mean CI two orders of magnitude above any other Q component. It is effectively always active, functioning as a bias-like term that establishes a baseline query signal at every position.

### K Subcomponents

| Component | Mean CI | Firing Act | Label |
|-----------|---------|------------|-------|
| K.329 | 0.891 | -51.1 | always fires |
| K.119 | 0.129 | -2.74 | punctuation, brackets, and newlines |
| K.290 | 0.018 | +0.03 | newlines, predicts indentation or newlines |
| K.357 | 0.004 | -0.03 | miscellaneous text and punctuation |
| K.315 | 0.003 | +0.07 | beginning of sequences |
| K.339 | 0.003 | -0.23 | uninterpretable / polysemantic |
| K.272 | 0.002 | +0.08 | first token of sequence |
| K.220 | 0.002 | +0.84 | markdown citation and reference opening tags |
| K.327 | 0.002 | -0.16 | sequence boundaries and document starts |
| K.121 | 0.001 | -0.18 | first token of sequence |
| K.218 | 0.001 | -0.32 | pronoun 'it', predicting subsequent verbs |
| K.147 | 0.001 | -0.26 | early sequence tokens |
| K.485 | 0.001 | +0.40 | predicts existence or copula verbs after "there"/"it" |

K.329 is the always-active key, analogous to Q.316. K.119 is the next most important, active on ~16% of tokens. The remaining K components are conditional, each active on narrow semantic or positional categories.

## Already-Studied Behaviors

The paper covers two behaviors in detail:

1. **Previous token behavior** (Q.316 × K.329): The interaction between the two always-active components exhibits strong positive static interaction strength (SIS) at small offsets and negative SIS at larger offsets, implementing cross-head previous token attention. Peak |z-scored SIS| = 11.20 (rank 1 overall).

2. **Previous syntax boundary movement** (Q.316 × K.119): This pair has strong positive SIS at medium-to-large offsets, implementing attention to punctuation and formatting boundaries further back in the sequence. Peak |z-scored SIS| = 9.51 (rank 2).

## Novel Behaviors

### Behavior 3: Newline / Paragraph Boundary Attention (Q.316 × K.290)

**Peak |z-scored SIS|:** 4.66 (rank 3 overall)

**Components:**
- **Q.316** (always active, CI = 0.857): The generic query signal.
- **K.290** (CI = 0.018, label: "fires on newlines, predicts indentation or newlines"): Active on ~2% of tokens, specifically on newlines and whitespace that precedes indented or structured content.

**Static Analysis:**

The SIS profile for this pair is strongly offset-dependent and peaks at medium-far offsets (Δ=8–12), distinct from both the near-offset peak of previous token behavior and the broadly increasing profile of syntax boundary movement:

| Offset | Mean SIS | H0 | H1 | H2 | H3 | H4 | H5 |
|--------|----------|------|------|------|------|------|------|
| 0 | +0.04 | | | | | | |
| 1 | -0.13 | | | | | | |
| 3 | +0.50 | | | | | | |
| 5 | +1.06 | | | | | | |
| 7 | +1.25 | | | | | | |
| 9 | +1.84 | +2.26 | +0.16 | **+4.66** | -0.29 | +2.36 | +1.91 |
| 11 | +1.91 | | | | | | |
| 13 | +1.15 | | | | | | |
| 16 | +0.59 | | | | | | |

The interaction is strongest in **H2** (z-scored SIS = 4.66 at Δ=9), with substantial contributions from H0 and H4. The bell-shaped offset profile, peaking around Δ=8–11, suggests this pair implements attention to newlines and paragraph breaks at a *specific range* — not the immediately preceding newline, but one roughly a sentence-length back. This is qualitatively different from K.119 (syntax boundary), which has broader and more monotonically increasing offset dependence.

**Head distribution:** The weight norms for K.290 across heads are relatively uniform (H0=0.84, H1=0.85, H2=0.93, H3=0.83, H4=0.69, H5=1.08), indicating this behavior is distributed rather than localized. However, the SIS analysis shows the computation is primarily effective in H0, H2, H4, and H5. H1 and H3 contribute minimally.

**Dynamic Analysis:**

Across 31 prompts, 23,061 (q,k) position pairs had both |q_act| > 1 and |k_act| > 1. Per-head data-dependent contribution statistics:

| Head | Mean | Std | % Positive | Range |
|------|------|-----|------------|-------|
| H0 | +0.012 | 1.60 | 50.1% | [-12.2, +15.9] |
| H1 | +0.030 | 1.39 | 61.8% | [-13.8, +4.9] |
| H2 | +0.112 | 2.08 | 58.7% | [-14.4, +18.0] |
| H3 | -0.032 | 1.50 | 40.2% | [-5.5, +15.2] |
| H4 | +0.030 | 1.44 | 34.8% | [-4.4, +12.6] |
| H5 | +0.080 | 2.25 | 60.6% | [-16.3, +10.5] |

The mean contributions are modest relative to the variance, consistent with a behavior that operates across a wide offset range rather than concentrating its effect at a single offset. H2 has the strongest mean positive contribution (+0.112) and H5 the second strongest (+0.080), matching the static analysis.

The offset-dependent profile in H2 shows a steady increase from Δ=0 to Δ=9 (mean contribution +0.054 → +0.214), confirming the static prediction that this pair becomes progressively more important at medium-far offsets.

**K-V Coactivation:**

When K.290 fires, the V components most likely to also be active are:

| V Component | P(V|K.290) | P(K.290|V) | Label |
|-------------|------------|------------|-------|
| V.919 | 0.962 | 0.422 | fires on newlines and indentation |
| V.494 | 0.859 | 0.646 | predicts line breaks or indentation in formatted text |
| V.72 | 0.771 | 0.126 | fires on punctuation to predict newlines and connectors |
| V.346 | 0.492 | 0.055 | distinguishes function vs content words |
| V.932 | 0.448 | 0.644 | (not in top-20 by CI) |
| V.940 | 0.299 | 0.799 | fires on line breaks and sequence boundaries |

The coactivating V components paint a coherent picture: when K.290 fires on a newline token, the value information available is overwhelmingly about newlines, indentation, and line structure. V.919 ("newlines and indentation") fires on 96.2% of K.290-active positions, and V.494 ("predicts line breaks") on 85.9%. The high P(K.290|V) for V.932 (0.644) and V.940 (0.799) indicates these components are quite specific to K.290's context.

**Interpretation:** This pair implements **paragraph/block boundary attention** — the model looks back to find the most recent newline or paragraph break at a medium distance. The combination of (a) the bell-shaped offset profile peaking at Δ~9, (b) the newline/indentation semantics of both K.290 and its coactivating V components, and (c) distributed computation across H0, H2, H4, H5 suggests the model uses this mechanism to track document structure at the paragraph level. This is complementary to K.119's syntax boundary mechanism, which operates on a broader set of delimiters at varying distances.

### Behavior 4: Copula Verb Attention Modulation (Q.308 × K.218 and Q.308 × K.485)

**Peak |z-scored SIS|:** 3.11 (Q.308/K.218), 2.72 (Q.308/K.485)

These two pairs share the same Q component and have semantically related K components, forming a coherent attention behavior around copula/existential verb constructions.

**Components:**
- **Q.308** (CI = 0.004, label: "existence and state verbs (is, was, there are/is)"): A rare, conditional query that activates on tokens where the model is about to predict a copula or state verb. Weight norm is concentrated in **H3** (1.13) with a secondary peak in H5 (0.67), compared to 0.34–0.60 in other heads.
- **K.218** (CI = 0.001, label: "fires on the pronoun 'it', predicting subsequent verbs"): Activates on the word "it" (and variants). Weight norm concentrated in H3 (0.80) and H5 (0.93).
- **K.485** (CI = 0.001, label: "predicts existence or copula verbs after 'there'/'it'"): Activates on "there" and "it" in existential constructions. Weight norm similarly concentrated in H3 (0.79) and H5 (0.80).

**Static Analysis:**

Both pairs show a distinctive offset profile that is approximately flat, with a mild peak at Δ=1–2 and a slow decline at larger offsets:

Q.308/K.218:
| Offset | Mean SIS | H3 | H5 |
|--------|----------|------|------|
| 0 | +0.53 | | |
| 1 | +0.59 | | |
| 2 | +0.61 | **+3.11** | +1.21 |
| 3 | +0.59 | | |
| 5 | +0.43 | | |
| 8 | +0.23 | | |

Q.308/K.485:
| Offset | Mean SIS | H3 | H5 |
|--------|----------|------|------|
| 0 | +0.44 | | |
| 1 | +0.53 | | |
| 2 | +0.58 | **+2.72** | +0.77 |
| 3 | +0.58 | | |
| 5 | +0.46 | | |
| 8 | +0.30 | | |

The SIS is heavily concentrated in **H3** for both pairs, with modest contributions from **H5**. This stands in sharp contrast to the broadly distributed behaviors of Q.316 pairs. The relatively flat offset profile with a slight near-offset peak means this behavior activates when "it" or "there" appeared recently — consistent with looking for the subject of an upcoming copula verb.

**Head Concentration:** Both K.218 and K.485 have their largest weight norms in H3 (0.80 and 0.79 respectively) and H5 (0.93 and 0.80), and Q.308's largest norm is also in H3 (1.13). This makes the Q.308 × K.218/K.485 interactions among the most head-concentrated in layer 1.

**Dynamic Analysis:**

Q.308/K.218 (1,908 active pairs at |act| > 1.0):

| Head | Mean | % Positive | Range |
|------|------|------------|-------|
| H0 | -0.004 | 40.7% | [-0.87, +2.47] |
| H1 | +0.019 | 67.5% | [-1.66, +0.50] |
| H2 | +0.015 | 68.4% | [-1.87, +0.34] |
| **H3** | **-0.151** | **31.1%** | **[-4.24, +27.60]** |
| H4 | +0.000 | 54.2% | [-0.71, +0.20] |
| H5 | -0.078 | 31.0% | [-1.72, +8.78] |

Q.308/K.485 (1,940 active pairs at |act| > 1.0):

| Head | Mean | % Positive | Range |
|------|------|------------|-------|
| H0 | -0.021 | 36.5% | [-0.66, +1.05] |
| **H3** | **-0.111** | **36.4%** | **[-3.70, +9.73]** |
| H5 | -0.035 | 38.6% | [-1.10, +2.57] |

H3 dominates both interactions, with the largest absolute mean and widest range, confirming the static prediction. The sign of the mean contribution is negative, indicating that in the typical case, Q.308's interaction with these K components *reduces* the attention score in H3 at the relevant positions. However, the large positive outliers (up to +27.6 for K.218 in H3) suggest that on specific tokens — likely the ones where copula verbs are actually being predicted — the contribution can be very strongly positive. The 31% positive rate means the positive contributions are rare but large.

**K-V Coactivation:**

K.485 has highly specific V coactivation:

| V Component | P(V|K.485) | P(K.485|V) | Label |
|-------------|------------|------------|-------|
| V.180 | 0.908 | 0.295 | fires on pronouns and dummy subjects (it, there) |
| V.744 | 0.851 | 0.047 | fires on pronouns and determiners |
| V.448 | 0.607 | 0.616 | fires on 'there/where/here' predicting 'to be' verbs |
| V.614 | 0.567 | **0.955** | predicting forms of 'to be' after 'there'/'here' |
| V.622 | 0.521 | **0.924** | fires on "there" to predict existence verbs |
| V.596 | 0.521 | **0.934** | predicts 'to be' verbs and 'exists' after 'there' |

The value information available at K.485-selected positions is strikingly coherent. V.180 ("pronouns and dummy subjects") and V.744 ("pronouns and determiners") encode the syntactic category of the attended token. V.448 ("there/where/here predicting 'to be' verbs") directly encodes the existential construction context. Most remarkably, V.614 ("predicting forms of 'to be' after there/here"), V.622 ("fires on 'there' to predict existence verbs"), and V.596 ("predicts 'to be' verbs and 'exists' after 'there'") have extremely high P(K.485|V) values (>0.92), meaning they almost exclusively fire when K.485 is also active. These three components form a tightly coupled functional group that specifically encodes the prediction of copula/existence verbs following "there" — exactly the information the downstream MLP would need to boost "is", "are", "was", etc.

K.218 has a different coactivation pattern:

| V Component | P(V|K.218) | P(K.218|V) | Label |
|-------------|------------|------------|-------|
| V.180 | 0.991 | 0.429 | fires on pronouns and dummy subjects (it, there) |
| V.744 | 0.960 | 0.070 | fires on pronouns and determiners |
| V.694 | 0.144 | 0.010 | fires on pronouns related to people |
| V.946 | 0.080 | 0.001 | distinguishes content words from function words/symbols |
| V.731 | 0.023 | 0.002 | fires on forms of 'to be' |

V.180 ("pronouns and dummy subjects") fires on 99.1% of K.218-active positions, and V.744 ("pronouns and determiners") on 96.0%. These two components — shared with K.485 — encode the general pronoun/determiner identity of the "it" token. Unlike K.485, however, K.218 does not coactivate strongly with the existential-construction-specific V components (V.614, V.622, V.596). Instead, the value information at "it" positions is more generic: pronoun identity (V.180, V.744, V.694) and, weakly, verb-type information (V.731, "forms of 'to be'"). This makes sense — "it" appears in many contexts beyond existential constructions (e.g., "it seems", "it was raining", anaphoric "it"), so the value information at "it" positions is less specialized.

**Interpretation:** Q.308/K.218 and Q.308/K.485 together implement **copula verb attention routing** — a mechanism concentrated in H3 (and secondarily H5) that modulates attention to positions containing "it" or "there" when the model is about to predict a copula verb. The V coactivation reveals a clear division of labor: at "there" positions (K.485), the OV circuit carries highly specific existential construction information via V.614/V.622/V.596, directly encoding the expectation of upcoming "to be" verbs. At "it" positions (K.218), the carried information is more generic — pronoun identity rather than construction-specific predictions. This is the most head-concentrated behavior in layer 1, with Q.308 and both K components having their largest weight norms in H3.

### Behavior 5: Sequence-Start and First-Token Attention (Q.316 × K.315 and Q.316 × K.121)

**Peak |z-scored SIS|:** 4.19 (Q.316/K.315), 2.12 (Q.316/K.121)

**Components:**
- **K.315** (CI = 0.003, label: "fires at the beginning of sequences")
- **K.121** (CI = 0.001, label: "fires on the first token of a sequence")

Both K components activate on early sequence positions, but they participate in qualitatively different interactions.

**Q.316 × K.315 — Sequence-Beginning Attention:**

The SIS profile increases monotonically with offset, reaching its peak at the maximum offset tested (Δ=16, mean SIS = +1.50). This means the pair's interaction is strongest when the query position is far from the key position — i.e., looking back to sequence-start positions from deep within the sequence. The effect is concentrated in **H2** (4.19 at Δ=10) and **H5** (up to 2.94 at Δ=16), with H0 also contributing (+0.59 at Δ=10).

Dynamic analysis (19,115 active pairs) confirms broadly positive contributions across most heads: H5 has mean +0.563, H0 +0.497, H2 +0.441, all with ~57-58% positive.

K.315 has very specific V coactivation: V.602, V.127, V.571, V.861, V.290, V.633 all have P(V|K.315) > 0.67. Several of these (V.861 at P(K|V)=1.000, V.172 at P(K|V)=1.000) are completely exclusive to K.315's context, indicating the OV circuit at sequence-start positions carries specialized information.

**Q.316 × K.121 — First-Token Attention Suppression:**

Unlike Q.316/K.315, this pair has predominantly **negative** SIS at medium-far offsets, and the negative contribution strengthens with distance. In H5, which has the largest effect (|z-scored SIS| = 2.12 at Δ=12), the dynamic analysis shows mean = -0.621 with only 27.3% positive contributions.

This suggests Q.316 × K.121 implements **first-token attention suppression** — the further away from the first token, the more this pair reduces attention to it. Combined with Q.316/K.315's sequence-beginning *boosting*, the two behaviors appear complementary: K.315 increases attention to early sequence content while K.121 specifically reduces attention to the very first token. This may serve to distinguish the BOS/first token (which often has less informative content) from early content tokens.

## OV Circuit Analysis

### Unweighted Frobenius Cosine Similarity

The random baseline for the unweighted FCS between two heads of dimension 128 in a 768-dimensional space is 0.143.

**Read overlap (W_OV^T W_OV):**

|      | H0    | H1    | H2    | H3    | H4    | H5    |
|------|-------|-------|-------|-------|-------|-------|
| H0   | 1.000 | 0.188 | 0.153 | 0.133 | 0.113 | 0.153 |
| H1   | 0.188 | 1.000 | 0.250 | 0.140 | 0.077 | 0.105 |
| H2   | 0.153 | 0.250 | 1.000 | 0.121 | 0.237 | 0.202 |
| H3   | 0.133 | 0.140 | 0.121 | 1.000 | 0.066 | 0.105 |
| H4   | 0.113 | 0.077 | 0.237 | 0.066 | 1.000 | 0.110 |
| H5   | 0.153 | 0.105 | 0.202 | 0.105 | 0.110 | 1.000 |

Mean off-diagonal: 0.144 (approximately at the random baseline).

The read subspace overlaps are close to or slightly above the random baseline for most pairs. The H1-H2 pair (0.250) and the H2-H4 pair (0.237) stand out as reading from somewhat more similar subspaces than expected by chance. H3 has notably low overlap with all other heads (0.066–0.140), consistent with its specialized role in copula verb processing (Behavior 4).

**Write overlap (W_OV W_OV^T):**

|      | H0    | H1    | H2    | H3    | H4    | H5    |
|------|-------|-------|-------|-------|-------|-------|
| H0   | 1.000 | 0.214 | 0.388 | 0.316 | 0.387 | 0.502 |
| H1   | 0.214 | 1.000 | 0.212 | 0.263 | 0.120 | 0.188 |
| H2   | 0.388 | 0.212 | 1.000 | 0.138 | 0.332 | 0.434 |
| H3   | 0.316 | 0.263 | 0.138 | 1.000 | 0.247 | 0.308 |
| H4   | 0.387 | 0.120 | 0.332 | 0.247 | 1.000 | 0.390 |
| H5   | 0.502 | 0.188 | 0.434 | 0.308 | 0.390 | 1.000 |

Mean off-diagonal: 0.296 (well above the random baseline of 0.143).

Write subspace overlaps are substantially higher than read overlaps. The H0-H5 pair (0.502), H2-H5 (0.434), and H0-H2 (0.388) show the most write similarity. H1 is the most isolated writer (mean off-diagonal 0.199), while H5 has the highest average overlap with others.

**Raw W_OV overlap:**

The raw W_OV similarities are near zero or slightly negative (mean off-diagonal = -0.047), indicating that despite reading from somewhat similar and writing to more similar subspaces, the full OV circuits implement different overall transformations. This is consistent with each head extracting and projecting distinct information channels.

### Summary: Read vs Write Asymmetry

The most striking finding is the **asymmetry between read and write overlap**:
- Read overlaps cluster near the random baseline (~0.14), indicating heads read from largely orthogonal residual stream subspaces.
- Write overlaps are roughly 2× higher (~0.30 on average), indicating heads write to more overlapping subspaces.

This pattern — reading from distinct subspaces but writing to somewhat shared output directions — suggests the heads in layer 1 extract diverse information from the residual stream but contribute to a more limited set of output directions. This may reflect the layer's role in consolidating diverse input signals (from embeddings and layer 0) into a smaller set of output features for downstream processing.

## Comparative Summary of Layer 1 Attention Behaviors

| Behavior | Pair | Peak |SIS| | Head Focus | Offset Profile | Mechanism |
|----------|------|------------|------------|----------------|-----------|
| 1. Previous token | Q.316/K.329 | 11.20 | Distributed (H1 peak) | Strong near, negative far | Cross-head recency bias |
| 2. Syntax boundary | Q.316/K.119 | 9.51 | Distributed (H3 peak) | Monotonically increasing | Delimiter/punctuation lookback |
| 3. Paragraph boundary | Q.316/K.290 | 4.66 | H0, H2, H4, H5 | Bell-shaped, peak Δ~9 | Medium-distance newline attention |
| 4. Copula verb routing | Q.308/K.218,K.485 | 3.11 | **H3** (concentrated) | Flat with near peak | Copula construction detection |
| 5a. Sequence-start boost | Q.316/K.315 | 4.19 | H2, H5 | Monotonically increasing | Attend to early sequence content |
| 5b. First-token suppression | Q.316/K.121 | 2.12 | H5 | Negative, increasing magnitude | Suppress BOS/first token |

**Key observations:**

1. **Most behaviors are Q.316-mediated.** Four of the five novel behaviors use the always-active Q.316 as their query component. The exception is the copula verb behavior (Q.308), which is the only behavior that is conditional on *both* the query and the key.

2. **Head concentration varies dramatically.** The copula verb behavior is strongly concentrated in H3, while the paragraph boundary and previous token behaviors are distributed across 4+ heads. This suggests parameter decomposition can distinguish between localized and distributed attention computations within the same layer.

3. **Offset profiles differentiate function.** Each behavior has a distinctive offset profile:
   - Near-peaked: previous token (Δ=0–2), copula (Δ=0–3)
   - Bell-shaped: paragraph boundary (Δ=5–12)
   - Far-peaked/monotonic: sequence start (increasing), first-token suppression (negatively increasing)
   - Broadly increasing: syntax boundary (Δ=3–16)

4. **K-V coactivation reveals the OV circuit's content.** For conditional K components, the PKV analysis reveals what value information is available at the attended positions. K.290 (newlines) coactivates with newline/indentation V components; K.485 (copula predictor) has extremely specific V coactivation (V.614, V.622, V.596 with >92% exclusivity).

5. **The OV circuit shows a read-diverse, write-convergent pattern.** Heads in layer 1 read from largely orthogonal subspaces (mean read FCS ≈ random baseline) but write to more overlapping subspaces (mean write FCS ≈ 2× baseline). This architectural pattern may reflect consolidation of diverse input signals.
