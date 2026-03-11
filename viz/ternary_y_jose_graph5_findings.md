# Graph Analysis: `z = y ? x : y` → predicts ` y` (Jose model)

**Graph 5** (optimized) from prompt ID 3. L0=209, 3026 edges.
Optimization: CE loss on token ` y` (id 340) at position 5.

## 1. Output node attribution breakdown

All 60 edges into `output:5:340` (` y`), grouped by source layer:

| Layer | sum(val) | sum(\|val\|) | n |
|-------|----------|-------------|---|
| `3.mlp.down` | +8.60 | 8.68 | 33 |
| `2.attn.o` | +3.02 | 3.21 | 15 |
| `3.attn.o` | -1.21 | 1.54 | 2 |
| `2.mlp.down` | -0.31 | 0.71 | 2 |
| `1.attn.o` | -0.32 | 0.32 | 1 |
| `1.mlp.down` | +0.32 | 0.32 | 2 |
| `0.mlp.down` | -0.17 | 0.24 | 3 |
| `embed` | +0.05 | 0.05 | 1 |
| `0.attn.o` | +0.02 | 0.02 | 1 |

Top individual edges:

| Source | val |
|--------|-----|
| `2.attn.o:5:210` | +2.02 |
| `3.attn.o:5:806` | -1.38 |
| `3.mlp.down:5:2756` | +1.21 |
| `3.mlp.down:5:176` | +0.67 |
| `3.mlp.down:5:431` | +0.57 |
| `2.mlp.down:5:170` | -0.51 |
| `3.mlp.down:5:442` | +0.42 |
| `3.mlp.down:5:3405` | +0.41 |
| `3.mlp.down:5:263` | +0.38 |
| `3.mlp.down:5:1118` | +0.35 |
| `1.attn.o:5:300` | -0.32 |
| `1.mlp.down:5:1217` | +0.31 |

The two dominant pathways are **layer 3 MLP** (net +8.60, with very little cancellation unlike the previous model) and **layer 2 attention** (net +3.02, led by O component 210). Layer 3 attention provides a notable negative contribution (-1.21), with the top 3 MLP components relating to math/code domains (2756: math instruction keywords, 176: algebraic text, 431: code punctuation).

## 2. Induction-like components on source tokens

Pos 2 (` y`) activates 10 `2.attn.v` components; pos 4 (` x`) activates 8. All have CI=1.0. Eight are shared; two (363, 406) appear only at pos 2.

| Shared (8) | Pos 2 only (2) |
|------------|----------------|
| 188, 391, 475, 627, 632, 731, 899, 1011 | 363, 406 |

### V → O cross-sequence edges to dominant O:210

| Source | val |
|--------|-----|
| `v:4:391` | -1.03 |
| `v:2:391` | -0.79 |
| `v:4:731` | -0.79 |
| `v:2:731` | -0.75 |
| `v:4:1011` | -0.04 |
| `v:2:188` | -0.03 |

Both positions contribute roughly equally to O:210 (pos 2 sum = -1.63, pos 4 sum = -1.87). Components 391 and 731 dominate, unlike the previous model where only 391 carried large attribution.

### V-side asymmetry analysis

When summing all outgoing V→O edges across **all** O targets (not just O:210), a clear asymmetry emerges:

| Component | sum(pos2→O) | sum(pos4→O) | diff | note |
|-----------|-------------|-------------|------|------|
| v:\*:627 | +0.31 | -0.14 | **+0.46** | **SIGN FLIP** |
| v:\*:391 | -0.81 | -1.09 | +0.28 | |
| v:\*:406 | +0.21 | — | +0.21 | pos2 only |
| v:\*:188 | +0.26 | +0.19 | +0.06 | |
| v:\*:363 | +0.05 | — | +0.05 | pos2 only |
| v:\*:475 | +0.16 | +0.14 | +0.03 | |
| v:\*:899 | +0.06 | +0.06 | +0.01 | |
| v:\*:632 | +0.17 | +0.18 | -0.01 | |
| v:\*:1011 | +0.02 | +0.05 | -0.03 | |
| v:\*:731 | +0.16 | +0.23 | -0.07 | |
| **TOTAL** | **+0.59** | **-0.39** | **+0.98** | |

**Key asymmetry sources:**

1. **V:627** (y/Y/j/J letter detector): the single largest differentiator (+0.46). Its outgoing attributions *flip sign* between positions — positive at pos 2 (` y`), negative at pos 4 (` x`). This makes it a token-identity-sensitive component: it constructively contributes when it sees the letter y, and destructively when it sees x.

2. **V:391** (high-frequency, 3.7%): contributes +0.28 to the asymmetry. Both positions produce negative V→O attributions to O:210, but pos 4 is more negative, so across all O nodes pos 2's total is less negative.

3. **V:406** (word-stem detector, pos2-only): contributes +0.21. Since it doesn't fire at pos 4 at all, its entire contribution is asymmetric. Its largest outgoing edge goes to O:947 (+0.22).

### Upstream: what feeds the V components

The dominant upstream sources of V:391 and V:731 at both positions are:

| Source | → v:2:391 | → v:4:391 | → v:2:731 | → v:4:731 |
|--------|-----------|-----------|-----------|-----------|
| `0.mlp.down:X:3082` | -1.61 | -1.33 | +2.56 | +2.16 |
| `0.mlp.down:X:1025` | -1.57 | **-4.25** | +0.46 | +0.92 |
| `0.mlp.down:X:1847` | +1.20 | +1.58 | -2.23 | -2.05 |
| `1.attn.o:X:630` | +0.94 | +1.10 | -0.93 | -0.84 |
| `0.mlp.down:X:2264` | -0.84 | — | +1.18 | — |
| `1.mlp.down:X:1217` | -0.37 | -0.77 | +0.34 | +0.48 |

Notable: `0.mlp.down:X:1025` (fires on tokens that determine the next token) feeds V:391 much more strongly at pos 4 (-4.25) than pos 2 (-1.57). This explains part of why V:391 has larger magnitude at pos 4.

The earliest asymmetry source is `0.mlp.up:2441`, a y-letter detector ("overwhelmingly activates on tokens that consist of or end in 'y'"). It is active only at pos 2 (CI=1.0), absent at pos 4. It feeds into `0.mlp.down:2:3082` (+0.81) and `0.mlp.down:2:1025` (-0.30), creating differentiated V-component activations from the very first MLP layer.

### K → O attribution (attention routing)

5 K components are shared between pos 2 and pos 4 (206, 224, 320, 327, 439). Pos 2 has one additional (83); pos 4 has two additional (130, 306).

Total K attribution to `2.attn.o:5:210` by source position:

| Position | Token | sum(val) |
|----------|-------|----------|
| 0 | `z` | -0.21 |
| 1 | ` =` | -0.01 |
| 2 | ` y` | -3.75 |
| 4 | ` x` | -4.15 |

For the dominant O:210, **K components don't strongly favor pos 2 over pos 4** — both variable positions get large negative K attribution (pos 4 slightly more negative). This is consistent with the previous model.

However, across **all** O targets combined, there is some K-side asymmetry: pos 2 total K = +2.34 vs pos 4 total K = -0.45. This suggests that while the dominant induction head (O:210) routes equally to both positions, some secondary O nodes are influenced more by pos 2 K values.

### Interpretation

The same high-level deduction holds: **the asymmetry that makes the output ` y` instead of ` x` comes primarily from the V side, not attention routing.** Both positions are attended to roughly equally by the dominant O:210 head.

However, the mechanism is more distributed in this model compared to the previous one:

- **Previous model**: 2 shared V components, with V:391 carrying ~3x more attribution from pos 2 than pos 4 to O:1015.
- **This model**: 8 shared V components + 2 pos2-only, with the biggest differentiator being V:627 (a y/Y letter detector with sign-flipping behavior). The asymmetry is spread across multiple components rather than concentrated in one.

The sign flip in V:627 is particularly interesting: the same component index at both positions produces *opposite-sign* contributions. This is because V:627 fires on y/Y letters, so at pos 2 (` y`) it produces constructive values, while at pos 4 (` x`) it produces destructive values. The upstream `0.mlp.up:2441` (also a y-letter detector) being active only at pos 2 further amplifies this token-identity signal from layer 0.

## 3. Layer 3 attention: `3.attn.o:5:806`

This node contributes -1.38 to the output (suppressing ` y`). It has only 3 inputs, all from the same V component:

| Source | val |
|--------|-----|
| `3.attn.v:5:76` | -1.13 (cross-seq, from pos 5 ` :`) |
| `3.attn.v:3:76` | -0.65 (cross-seq, from pos 3 ` ?`) |
| `3.attn.v:1:76` | -0.21 (cross-seq, from pos 1 ` =`) |

V:76 fires at positions 1, 3, 5 — the structural/syntax positions (` =`, ` ?`, ` :`), not the variable-value positions. Its dominant upstream input at each position is `0.mlp.down:X:328` (~+0.9 at all three positions), a component that "almost exclusively targets programming-related input" (7.8% firing rate). No K nodes are present in the graph for layer 3 attention, meaning routing is not decomposed.

The main effect of `3.attn.o:5:806` is not its direct -1.38 to the output but its large-magnitude edges to `3.mlp.up:5:*` (18 edges, up to 3.58 magnitude). The net sum to `3.mlp.up` is -9.10 with total |sum| = 24.0, making it the dominant driver of layer 3 MLP alongside `2.attn.o` (which contributes +8.79 net to `3.mlp.up`). The other L3 attention O node (`3.attn.o:5:682`) contributes a modest +0.17 to output.
