# Graph Analysis: `z = y ? x :` → predicts ` y` (Jose model)

**Graph 57** (optimized) from the SPD app on port 8000. L0=210, 3057 edges.
Optimization: CE loss on token ` y` (id 340) at position 5.
CI-masked label prob: 99.99%, stochastic-masked: 99.93%, adversarial PGD: 69.92%.

## 1. Output node attribution breakdown

All 59 edges into `output:5:340` (` y`), grouped by source layer:

| Layer | sum(val) | sum(\|val\|) | n |
|-------|----------|-------------|---|
| `3.mlp.down` | +8.60 | 8.68 | 33 |
| `2.attn.o` | +3.02 | 3.21 | 15 |
| `3.attn.o` | -1.21 | 1.54 | 2 |
| `2.mlp.down` | -0.31 | 0.71 | 2 |
| `1.attn.o` | -0.32 | 0.32 | 1 |
| `1.mlp.down` | +0.31 | 0.31 | 1 |
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

The two dominant pathways are **layer 3 MLP** (net +8.60, very little internal cancellation) and **layer 2 attention** (net +3.02, led by O:210). Layer 3 attention provides a notable negative contribution (-1.21). The top L3 MLP components are math/code-domain detectors (2756: "math and programming expressions", 176: "algebraic expressions", 431: "code identifiers and syntax").

## 2. Layer 0: token identity detection

Six `0.mlp.up` components fire only at position 2 (` y`) and are completely absent at position 4 (` x`). No position-4-only components exist.

| Component | act | label |
|-----------|-----|-------|
| `0.mlp.up:2441` | +6.86 | word completions after tokens ending in 'y' |
| `0.mlp.up:582` | +2.72 | word stem completion (stems to suffixes) |
| `0.mlp.up:1275` | +2.33 | predicts continuations of stems and content words |
| `0.mlp.up:2301` | -2.11 | word stem completion |
| `0.mlp.up:784` | -1.50 | word stem completion |
| `0.mlp.up:363` | +0.78 | word stem completion (stems to suffixes) |

`0.mlp.up:2441` is the most important: a y-letter detector with the largest activation. It feeds downstream via:

| Target | val |
|--------|-----|
| `0.mlp.down:2:3374` | +0.81 |
| `0.mlp.down:2:3082` | +0.81 |
| `0.mlp.down:2:2264` | -0.60 |
| `0.mlp.down:2:168` | +0.37 |
| `0.mlp.down:2:897` | -0.32 |
| `0.mlp.down:2:1847` | +0.31 |
| `0.mlp.down:2:1025` | -0.30 |

## 3. Layer 2 attention: induction-like copying with V-side asymmetry

### O:210 — the dominant attention head

`2.attn.o:5:210` (act=-3.47) has the largest single effect: +2.02 direct to output and +3.32 into L3 MLP (~5.3 total). It receives 46 edges: 40 cross-sequence (V and K) and 6 same-sequence (Q).

**K-side (attention routing):** K attribution to O:210 is similar for both variable positions — the head attends roughly equally to both.

| Position | Token | K sum |
|----------|-------|-------|
| 0 | `z` | +1.57 |
| 1 | ` =` | -0.01 |
| 2 | ` y` | -3.74 |
| 4 | ` x` | -4.15 |

**V-side (content copied):** V→O:210 contributions by position:

| Position | Token | V sum |
|----------|-------|-------|
| 1 | ` =` | -0.01 |
| 2 | ` y` | -1.63 |
| 3 | ` ?` | +0.01 |
| 4 | ` x` | -1.87 |
| 5 | ` :` | +0.01 |

Both positions contribute similarly to O:210 individually. The asymmetry emerges when summing V→O edges across **all** O targets at position 5.

### V-side asymmetry across all O nodes

| Component | sum(pos2→O) | sum(pos4→O) | diff | note |
|-----------|-------------|-------------|------|------|
| v:\*:627 | +0.31 | -0.14 | **+0.46** | **SIGN FLIP** |
| v:\*:391 | -0.81 | -1.09 | +0.28 | |
| v:\*:406 | +0.21 | — | +0.21 | pos2 only |
| v:\*:731 | +0.16 | +0.23 | -0.07 | |
| v:\*:188 | +0.26 | +0.19 | +0.06 | |
| v:\*:363 | +0.05 | — | +0.05 | pos2 only |
| v:\*:475 | +0.16 | +0.14 | +0.03 | |
| v:\*:899 | +0.06 | +0.06 | +0.01 | |
| v:\*:632 | +0.17 | +0.18 | -0.01 | |
| v:\*:1011 | +0.02 | +0.05 | -0.03 | |
| **TOTAL** | **+0.59** | **-0.39** | **+0.98** | |

**V:627** ("completions for words starting with j or y") is the single largest differentiator. Its activation flips sign between positions: act=-4.57 at pos 2 (` y`) vs act=+1.94 at pos 4 (` x`). It constructively contributes when it sees y, and destructively when it sees x.

**V:391** ("predicts highly probable word and phrase completions") contributes +0.28 to the asymmetry. Both positions produce negative V→O:210 attribution, but pos 4 is more negative, so pos 2 wins on net.

**V:406** ("multi-token word completion") is pos2-only, contributing +0.21. Its largest outgoing edge goes to O:947.

### Upstream of V:627: the sign-flip chain

V:627 at pos 2 (act=-4.57) and pos 4 (act=+1.94) receives different upstream signals:

| Source | → v:2:627 | → v:4:627 |
|--------|-----------|-----------|
| `0.mlp.down:X:3082` | -1.93 | -1.72 |
| `0.mlp.down:X:1025` | +0.38 | +0.88 |
| `0.mlp.down:X:1847` | +0.63 | -0.54 |
| `1.attn.o:X:630` | +0.54 | — |

`0.mlp.down:3082` dominates at both positions, but the mix of secondary sources creates the activation difference. The chain `0.mlp.up:2441 → 0.mlp.down:2:3082 → V:2:627` (product: 0.81 x -1.93 = -1.56) is the primary path by which the y-identity signal from layer 0 reaches V:627.

### K-side asymmetry across secondary O nodes

While K doesn't differentiate positions for O:210, it does across all O nodes combined:

| Position | Token | K→O total |
|----------|-------|-----------|
| 0 | `z` | -0.36 |
| 1 | ` =` | -0.02 |
| 2 | ` y` | **+2.28** |
| 4 | ` x` | **-0.45** |

Several secondary O nodes receive K-routing that favors pos 2:
- O:353: K pos2 = +1.42, K pos4 = +0.09
- O:880: K pos2 = +1.66, K pos4 = +0.48
- O:888: K pos2 = +1.47, K pos4 = -0.29

So while the dominant head O:210 routes equally, some secondary heads preferentially attend to the y-position via K.

## 4. Layer 3 attention: `3.attn.o:5:806`

This node contributes -1.38 to the output (suppressing ` y`) and -9.10 into L3 MLP (with massive cancellation: |sum| = 24.0). It has only 2 inputs, both from V:76 at syntax positions:

| Source | val |
|--------|-----|
| `3.attn.v:5:76` | -1.13 (from pos 5 ` :`) |
| `3.attn.v:1:76` | -0.21 (from pos 1 ` =`) |

V:76 fires at structural positions (` =`, ` :`), not variable-value positions. Its dominant upstream input is `0.mlp.down:X:328` (~+0.9-1.0 at both positions), a "programming-related input" component. This pathway responds to the structural ternary pattern, not to the variable identities.

The other L3 attention node (`3.attn.o:5:682`) contributes a modest +0.17 to output and +1.13 to L3 MLP.

## 5. Interpretation

### How the model outputs ` y`

The mechanism operates across three stages:

**Stage 1 — Token identity (Layer 0):** `0.mlp.up:2441` (y-letter detector) fires only at position 2 (act=+6.86). No corresponding x-detector exists. This creates differentiated `0.mlp.down` activations at pos 2 vs pos 4.

**Stage 2 — Content-based copying (Layer 2 attention):** Multiple attention heads at position 5 attend to both variable positions (2 and 4). The asymmetry that selects ` y` over ` x` comes primarily from the V-side (what content is read), not from attention routing (which positions are attended to). V:627's sign-flipping behavior is the single largest contributor: it produces opposite-sign attributions depending on whether it sees y or x. The total V-side asymmetry is +0.98 favoring position 2.

**Stage 3 — Amplification (Layer 3 MLP):** 33 `3.mlp.down` components at position 5 amplify the layer 2 attention output into a +8.60 net contribution to the output logit. `3.attn.o:806` provides a large-magnitude suppressive signal based on the structural ternary pattern (responding to ` =` and ` :` syntax tokens), but its effect is largely cancelled within L3 MLP.

```
Layer 0:  embed → 0.mlp.up:2441 (y-detector, pos2 only)
              → 0.mlp.down:3082, :1025, :1847 (stem completers)
                  |
Layer 2:  V:627 (sign flip: neg at y, pos at x)
          V:391, V:731 (shared, slight pos2 advantage)
          V:406, V:363 (pos2 only)
              → 2.attn.o:210 (+5.3 total effect)
              → 2.attn.o:279, :947 (secondary, feed L3)
                  |
Layer 3:  3.mlp.down (33 components, +8.60 to output)
          3.attn.o:806 (-1.38 to output, structural/syntax)
                  |
Output:   " y" token 340 at pos 5 (prob ~ 1.0)
```
