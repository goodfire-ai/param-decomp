# Graph Analysis: `z = y ? x :` → predicts ` y` — Abs-Target Attributions

**Graph 11** (optimized) from the Jose model. L0=209, 3047 edges.
Optimization: CE loss on token ` y` (id 340) at position 5.

This analysis uses **abs-target attributions** (`∂|y|/∂x · x = sign(y) · ∂y/∂x · x`) instead of signed attributions (`∂y/∂x · x`). A positive abs-target edge means the source *amplifies* the target's magnitude; a negative edge means it *suppresses* it.

Since graph 11 predates the `edgesAbs` feature, abs-target values were derived analytically: `abs_edge = sign(target_activation) * signed_edge`.

## 1. Output node attribution breakdown

Target: `output:5:340` (logit for ` y`). Logit value = +28.0, so `sign(y) = +1`.

**All edges into this output node are identical to the signed version** — the positive target logit means no sign flips. The two dominant pathways remain layer 3 MLP (net +8.60) and layer 2 attention (net +3.02, led by O:210).

| Layer | sum(val) | sum(\|val\|) | n |
|-------|----------|-------------|---|
| `3.mlp.down` | +8.60 | 8.68 | 33 |
| `2.attn.o` | +3.02 | 3.21 | 15 |
| `3.attn.o` | -1.21 | 1.54 | 2 |
| `2.mlp.down` | -0.31 | 0.71 | 2 |
| `1.mlp.down` | +0.31 | 0.31 | 1 |
| `0.mlp.down` | -0.17 | 0.24 | 3 |
| `embed` | +0.05 | 0.05 | 1 |
| `0.attn.o` | +0.02 | 0.02 | 1 |

## 2. Induction-like components on source tokens

Component inventory is unchanged: 10 `2.attn.v` components at pos 2, 8 at pos 4, with 8 shared and 2 pos2-only (363, 406).

### V component activations

These activation signs determine how edges *into* V components change between signed and abs-target:

| Component | act(pos2) | act(pos4) | sign2 | sign4 |
|-----------|-----------|-----------|-------|-------|
| v:\*:188 | +6.02 | +3.82 | +1 | +1 |
| v:\*:363 | +1.99 | — | +1 | — |
| v:\*:391 | **-9.79** | **-11.96** | **-1** | **-1** |
| v:\*:406 | -4.42 | — | -1 | — |
| v:\*:475 | -6.33 | -4.43 | -1 | -1 |
| v:\*:627 | **-4.57** | **+1.94** | **-1** | **+1** |
| v:\*:632 | +4.97 | +3.76 | +1 | +1 |
| v:\*:731 | +10.96 | +10.00 | +1 | +1 |
| v:\*:899 | -2.26 | -1.95 | -1 | -1 |
| v:\*:1011 | +5.23 | +6.12 | +1 | +1 |

### V → O cross-sequence edges to dominant O:210

`2.attn.o:5:210` has activation **-3.47** (sign = -1), so all V→O:210 edges **flip sign** compared to the signed analysis. In the signed view these were all negative; in abs-target they are all **positive** — V components uniformly *amplify* O:210's magnitude.

| Source | abs-tgt | signed |
|--------|---------|--------|
| `v:4:391` | +1.03 | -1.03 |
| `v:2:391` | +0.79 | -0.79 |
| `v:4:731` | +0.79 | -0.79 |
| `v:2:731` | +0.75 | -0.75 |
| `v:4:1011` | +0.04 | -0.04 |
| `v:2:188` | +0.03 | -0.03 |

Position sums: pos 2 = +1.63, pos 4 = +1.87 (same magnitudes as signed, signs flipped).

### V-side asymmetry analysis (abs-target)

Summing all outgoing V→O edges across **all** O targets:

| Component | sum(pos2→O) | sum(pos4→O) | diff | signed diff | note |
|-----------|-------------|-------------|------|-------------|------|
| v:\*:627 | +0.32 | -0.15 | **+0.46** | +0.46 | **SIGN FLIP** |
| v:\*:406 | +0.35 | — | **+0.35** | +0.21 | pos2 only |
| v:\*:188 | +0.44 | +0.35 | +0.09 | +0.06 | |
| v:\*:363 | +0.05 | — | +0.05 | +0.05 | pos2 only |
| v:\*:475 | +0.38 | +0.36 | +0.02 | +0.03 | |
| v:\*:632 | +0.41 | +0.40 | +0.01 | -0.01 | |
| v:\*:899 | +0.04 | +0.04 | -0.00 | +0.01 | |
| v:\*:731 | +1.68 | +1.87 | -0.20 | -0.07 | |
| v:\*:391 | +0.69 | +0.90 | **-0.21** | +0.28 | **FLIPPED** |
| v:\*:1011 | +0.44 | +0.71 | -0.27 | -0.03 | |
| **TOTAL** | **+4.78** | **+4.49** | **+0.30** | **+0.98** | |

**Key changes from signed analysis:**

1. **Total asymmetry shrinks from +0.98 to +0.30** — a 70% reduction. Much of the signed asymmetry was an artifact of different O target activation signs, not genuine pos2-vs-pos4 differentiation.

2. **V:391 flips direction** — from +0.28 (signed, appearing to favor pos 2) to -0.21 (abs-target, actually favors pos 4). V:391 at pos 4 (act = -11.96) has larger magnitude than at pos 2 (act = -9.79), so it amplifies downstream O magnitudes more. The signed view was misleading: pos 4's "more negative" contributions went to negatively-activated O nodes, inflating the apparent asymmetry.

3. **V:627 is unchanged at +0.46** — the only component whose asymmetry is robust across both views. Its sign flip between positions (negative at pos 2, positive at pos 4) creates genuine directional asymmetry that isn't an artifact of target signs. This confirms V:627 as the primary token-identity discriminator.

4. **V:406 increases from +0.21 to +0.35** — its pos2-only contribution is larger in magnitude terms.

### Upstream: what feeds the V components

Edges into V:391 all flip sign (V:391 has negative activation), while edges into V:731 (positive activation) are unchanged:

| Source | → v:2:391 | → v:4:391 | → v:2:731 | → v:4:731 |
|--------|-----------|-----------|-----------|-----------|
| `0.mlp.down:X:3082` | +1.61 | +1.33 | +2.56 | +2.16 |
| `0.mlp.down:X:1025` | +1.57 | **+4.25** | +0.46 | +0.92 |
| `0.mlp.down:X:1847` | -1.20 | -1.58 | -2.23 | -2.05 |
| `1.attn.o:X:630` | -0.94 | -1.10 | -0.93 | -0.84 |
| `0.mlp.down:X:2264` | +0.84 | — | +1.18 | — |
| `1.mlp.down:X:1217` | +0.37 | +0.77 | +0.34 | +0.48 |

In the abs-target view, all top sources into V:391 are now **positive** (they amplify V:391's magnitude). `0.mlp.down:X:1025` still feeds V:391 much more strongly at pos 4 (+4.25) than pos 2 (+1.57), explaining why V:391's magnitude is larger at pos 4.

Sources into V:731 are unchanged (positive activation → no sign flip). `0.mlp.down:X:3082` and `0.mlp.down:X:1847` dominate with opposite signs — 3082 amplifies magnitude, 1847 suppresses it.

### K → O attribution (abs-target)

Total K attribution to `2.attn.o:5:210` by source position (all flipped since O:210 is negative):

| Position | Token | abs-tgt sum | signed sum |
|----------|-------|-------------|------------|
| 0 | `z` | -1.78 | +1.78 |
| 1 | ` =` | +0.01 | -0.01 |
| 2 | ` y` | +3.75 | -3.75 |
| 4 | ` x` | +4.15 | -4.15 |

The conclusion is unchanged: **K components don't strongly favor pos 2 over pos 4** for the dominant O:210 head. Both variable positions contribute roughly equally to amplifying O:210's magnitude.

Across **all** O targets: pos 2 K total = +12.79, pos 4 K total = +11.88 (diff = +0.91, or 7% proportional asymmetry). The signed view showed +2.34 vs -0.45, which appeared as a dramatic asymmetry but was largely an artifact of O target activation signs.

### Revised interpretation

The core conclusion holds: **asymmetry comes primarily from the V side, not attention routing.** But the abs-target view substantially refines the picture:

- **Signed view**: 5 of 10 V components favor pos 2, total asymmetry = +0.98. V:627, V:391, and V:406 are the top three contributors.
- **Abs-target view**: Only 4 components contribute meaningfully to asymmetry (total = +0.30). V:391 actually favors pos 4 in magnitude terms. V:627 (+0.46) and V:406 (+0.35) account for essentially all of the genuine asymmetry.

The mechanism is more concentrated than the signed analysis suggested: **V:627** (y-letter detector with sign-flip behavior) and **V:406** (pos2-only component) are the two components that genuinely discriminate ` y` from ` x`. The other shared components contribute roughly symmetrically when measured by magnitude amplification.

## 3. Layer 3 attention: `3.attn.o:5:806`

`3.attn.o:5:806` has activation **-6.62** (sign = -1).

**Edge to output**: -1.38 (unchanged — output logit is positive).

**Inputs** (all flipped since O:806 is negative):

| Source | abs-tgt | signed | note |
|--------|---------|--------|------|
| `3.attn.v:5:76` | **+1.12** | -1.12 | cross-seq, from pos 5 ` :` |
| `3.attn.v:3:76` | **+0.65** | -0.65 | cross-seq, from pos 3 ` ?` |
| `3.attn.v:1:76` | **+0.21** | -0.21 | cross-seq, from pos 1 ` =` |

In abs-target: V:76 uniformly **amplifies** O:806's magnitude from the structural/syntax positions.

### Key finding: O:806 → 3.mlp.up cancellation

| View | net | sum(\|val\|) | cancellation |
|------|-----|-------------|--------------|
| Signed | -7.22 | 22.16 | **67%** |
| Abs-target | -22.16 | 22.16 | **0%** |

In the abs-target view, **all 17 edges** from O:806 to 3.mlp.up are **negative**. O:806 uniformly *suppresses* the magnitude of all downstream MLP components. The 67% cancellation in the signed view was entirely due to different 3.mlp.up components having different activation signs — not genuine mixed effects.

The chain in abs-target terms: V:76 (at syntax positions) amplifies O:806's magnitude → O:806 suppresses 3.mlp.up magnitude. This is a cleaner description than tracking sign chains through the signed view.

For comparison, `2.attn.o → 3.mlp.up` retains 59% cancellation in the abs-target view (down from ~79% in signed), indicating genuinely mixed effects from the attention layer 2 outputs.

## 4. Global cancellation analysis

| Scope | cancel (signed) | cancel (abs-target) |
|-------|----------------|---------------------|
| All edges | 87% | 46% |

Layer pairs with the largest cancellation reduction:

| Source → Target | sum\|val\| | cancel (signed) | cancel (abs-target) | Δ |
|-----------------|-----------|-----------------|---------------------|---|
| `embed → 0.mlp.up` | 86.8 | 88% | 0% | -88% |
| `0.attn.o → 0.mlp.up` | 6.6 | 98% | 0% | -98% |
| `1.attn.o → 2.attn.v` | 11.4 | 90% | 4% | -86% |
| `2.attn.q → 2.attn.o` | 15.5 | 91% | 6% | -86% |
| `3.attn.o → 3.mlp.up` | 24.3 | 74% | 13% | -61% |
| `2.attn.v → 2.attn.o` | 16.0 | 90% | 35% | -55% |
| `2.attn.k → 2.attn.o` | 33.4 | 96% | 41% | -55% |

No layer pair has *more* cancellation in the abs-target view. The abs-target representation consistently reduces or eliminates cancellation, particularly for feed-forward connections (embed → MLP, attn.o → MLP) where the signed view's cancellation was entirely an artifact of mixed target activation signs.
