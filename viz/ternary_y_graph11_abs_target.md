# Abs-Target Attribution Analysis: `z = y ? x :` → predicts ` y`

**Graph 57** (optimized, app ID 57 / internally "graph 11"), Jose model (`goodfire/spd/s-55ea3f9b`).
L0 = 210 active nodes, 3057 edges. Optimization: CE loss on token ` y` (id 340) at position 5, sparsity coeff 0.05, 2000 steps, p-norm 0.3.

**Attribution type**: abs-target (`∂|y|/∂x · x`). A positive edge means the source **amplifies** the target's magnitude; a negative edge means it **suppresses** it. This measures how much each source contributes to making its target larger in absolute value, regardless of sign.

## 1. Output node breakdown

All edges into `output:5:340` (logit for ` y`). The output logit is positive (+28.0), so abs-target edges into the output are identical to standard attributions.

| Layer | sum(val) | sum(\|val\|) | n |
|-------|----------|-------------|---|
| `3.mlp.down` | **+8.60** | 8.68 | 33 |
| `2.attn.o` | **+3.02** | 3.21 | 15 |
| `3.attn.o` | -1.21 | 1.54 | 2 |
| `2.mlp.down` | -0.31 | 0.71 | 2 |
| `1.attn.o` | -0.32 | 0.32 | 1 |
| `1.mlp.down` | +0.31 | 0.31 | 1 |
| `0.mlp.down` | -0.17 | 0.24 | 3 |
| `embed` | +0.05 | 0.05 | 1 |
| `0.attn.o` | +0.02 | 0.02 | 1 |

**Two dominant amplifiers of the ` y` logit:**

1. **Layer 3 MLP** (net +8.60, very little cancellation). The top contributors are `3.mlp.down:5:2756` (+1.21), `3.mlp.down:5:176` (+0.67), `3.mlp.down:5:431` (+0.57), and `3.mlp.down:5:442` (+0.42).

2. **Layer 2 attention** (net +3.02), led by `2.attn.o:5:210` (+2.02). This is the induction-like attention head that reads variable values from earlier positions.

**One notable suppressor:** `3.attn.o:5:806` (-1.38) suppresses the output logit. See §4 for its role in the 3.mlp pathway.

## 2. Layer 2 attention: V components at variable positions

### Inventory

10 V components are active at pos 2 (` y`), 8 at pos 4 (` x`). Eight are shared; two (363, 406) fire only at pos 2.

| Component | act(pos2) | act(pos4) | note |
|-----------|-----------|-----------|------|
| v:\*:188 | +6.02 | +3.82 | |
| v:\*:363 | +1.99 | — | pos2 only |
| v:\*:391 | -9.79 | -11.96 | largest magnitude |
| v:\*:406 | -4.42 | — | pos2 only |
| v:\*:475 | -6.33 | -4.43 | |
| v:\*:627 | **-4.57** | **+1.94** | **sign flip** |
| v:\*:632 | +4.97 | +3.76 | |
| v:\*:731 | +10.96 | +10.00 | |
| v:\*:899 | -2.26 | -1.95 | |
| v:\*:1011 | +5.23 | +6.12 | |

**V:627** is the only component whose activation sign differs between positions — negative at pos 2, positive at pos 4. This has direct consequences for how its downstream edges behave (§2.2).

### 2.1 V → O edges to the dominant O:210

`2.attn.o:5:210` has activation -3.47. All V→O:210 edges are **positive** — every V component uniformly amplifies O:210's magnitude.

| Source | val | note |
|--------|-----|------|
| `v:4:391` | +1.03 | from ` x` |
| `v:2:391` | +0.79 | from ` y` |
| `v:4:731` | +0.79 | from ` x` |
| `v:2:731` | +0.75 | from ` y` |
| `v:4:1011` | +0.04 | |
| `v:2:188` | +0.03 | |
| `v:2:1011` | +0.03 | |

V:391 and V:731 dominate. Position sums: pos 2 = +1.63, pos 4 = +1.87. Both positions amplify O:210 roughly equally — O:210 does not strongly discriminate between ` y` and ` x`.

### 2.2 V-side asymmetry across all O targets

Summing all outgoing V→O edges per component reveals the genuine magnitude-based asymmetry between positions:

| Component | sum(pos2→O) | sum(pos4→O) | diff | note |
|-----------|-------------|-------------|------|------|
| v:\*:627 | +0.32 | -0.15 | **+0.46** | **sign flip** |
| v:\*:406 | +0.35 | — | **+0.35** | pos2 only |
| v:\*:188 | +0.44 | +0.35 | +0.09 | |
| v:\*:363 | +0.05 | — | +0.05 | pos2 only |
| v:\*:475 | +0.38 | +0.36 | +0.02 | |
| v:\*:632 | +0.41 | +0.40 | +0.01 | |
| v:\*:899 | +0.04 | +0.04 | -0.00 | |
| v:\*:731 | +1.68 | +1.87 | -0.20 | |
| v:\*:391 | +0.69 | +0.90 | -0.21 | |
| v:\*:1011 | +0.44 | +0.71 | -0.27 | |
| **TOTAL** | **+4.78** | **+4.49** | **+0.30** | |

**Two components account for essentially all asymmetry:**

1. **V:627** (+0.46 diff) — the y/Y letter detector with sign-flipping activation. At pos 2, V:627 is negative (act = -4.57) and its outgoing edges amplify downstream O magnitudes. At pos 4, V:627 is positive (act = +1.94) and its edges suppress O magnitudes. This makes V:627 a genuine token-identity discriminator: it treats ` y` and ` x` in opposite ways.

2. **V:406** (+0.35 diff) — active only at pos 2, absent at pos 4. Its dominant upstream source is `0.mlp.down:2:2522` (+3.64), which itself is absent at pos 4 (CI = 0). V:406 distributes its influence across many O components, with the largest edge going to O:947 (+0.22).

The remaining 8 shared components contribute **nearly symmetrically** — their total diff is only -0.52 (slightly favoring pos 4 in aggregate). V:731, V:391, and V:1011 each slightly favor pos 4, consistent with their slightly larger activations there.

The total V-side asymmetry is +0.30 in favor of pos 2, entirely driven by V:627 and V:406.

### 2.3 K routing

K components into O:210 by source position:

| Position | Token | sum(val) | n |
|----------|-------|----------|---|
| 0 | `z` | -1.57 | 3 |
| 1 | ` =` | +0.01 | 1 |
| 2 | ` y` | +3.74 | 6 |
| 4 | ` x` | +4.15 | 7 |

Both variable positions strongly amplify O:210's magnitude (pos 2: +3.74, pos 4: +4.15). K routing does not discriminate between ` y` and ` x` for the dominant head — it attends to both roughly equally.

6 K components are shared between pos 2 and pos 4 (130, 206, 224, 320, 327, 439). Pos 4 has one additional (306).

Across **all** O targets: pos 2 K total = +12.62, pos 4 K total = +11.88 — a 6% proportional difference, confirming minimal K-side asymmetry.

### 2.4 Upstream of V components

The dominant upstream sources feeding V:391 and V:731 at both positions:

| Source | → v:2:391 | → v:4:391 | → v:2:731 | → v:4:731 |
|--------|-----------|-----------|-----------|-----------|
| `0.mlp.down:X:3082` | +1.61 | +1.33 | +2.56 | +2.16 |
| `0.mlp.down:X:1025` | +1.57 | **+4.25** | +0.46 | +0.92 |
| `0.mlp.down:X:1847` | -1.20 | -1.58 | -2.23 | -2.05 |
| `1.attn.o:X:630` | -0.94 | -1.10 | -0.93 | -0.84 |
| `0.mlp.down:X:2264` | +0.84 | — | +1.18 | — |
| `1.mlp.down:X:1217` | +0.37 | +0.77 | +0.34 | +0.48 |

All top sources into V:391 are **positive** — they uniformly amplify its magnitude. `0.mlp.down:X:1025` amplifies V:391 much more at pos 4 (+4.25) than pos 2 (+1.57), explaining V:391's larger magnitude at pos 4 (-11.96 vs -9.79).

For V:627, the upstream pattern is strikingly different between positions:

| Source | → v:2:627 | → v:4:627 |
|--------|-----------|-----------|
| `0.mlp.down:X:3082` | **+1.93** | **-1.72** |
| `0.mlp.down:X:1025` | -0.38 | +0.88 |
| `0.mlp.down:X:1847` | -0.62 | -0.54 |
| `1.attn.o:X:630` | -0.54 | — |

`0.mlp.down:X:3082` **amplifies** V:627's magnitude at pos 2 (+1.93) but **suppresses** it at pos 4 (-1.72). Since V:627 is negative at pos 2 and positive at pos 4, this means 3082 consistently pushes V:627 toward negative values regardless of position. The difference in V:627's sign between positions is what converts this uniform directional push into opposite magnitude effects.

### 2.5 The earliest asymmetry: `0.mlp.up:2:2441`

`0.mlp.up:2441` is a y-letter detector (activates on tokens consisting of or ending in 'y'). It fires at pos 2 (act = +6.86, CI = 1.0) and is inactive at all other positions (CI = 0.0).

Its downstream edges amplify several layer-0 MLP-down components:

| Target | val |
|--------|-----|
| `0.mlp.down:2:3374` | +0.81 |
| `0.mlp.down:2:3082` | +0.81 |
| `0.mlp.down:2:2264` | +0.60 |
| `0.mlp.down:2:168` | +0.37 |
| `0.mlp.down:2:897` | +0.32 |
| `0.mlp.down:2:1847` | +0.31 |
| `0.mlp.down:2:1025` | -0.30 |

The +0.81 amplification of `0.mlp.down:2:3082` is notable: 3082 is the dominant upstream source of V:627 at pos 2 (§2.4). Since 2441 is absent at pos 4, this extra amplification of 3082's magnitude at pos 2 contributes to V:627's stronger (and negative) activation there.

## 3. Interpretation: what makes the model predict ` y`

**The asymmetry that selects ` y` over ` x` comes from the V side, not attention routing.** K components route attention roughly equally to both variable positions. The discrimination happens through two mechanisms:

1. **V:627** (y-letter detector, +0.46 asymmetry): Its activation **flips sign** between positions — negative at pos 2 (` y`), positive at pos 4 (` x`). This sign flip converts its downstream edges from magnitude-amplifying at pos 2 to magnitude-suppressing at pos 4. The root cause traces to the upstream `0.mlp.up:2441` (also a y-letter detector), which is active only at pos 2 and amplifies `0.mlp.down:3082`, the dominant feeder of V:627.

2. **V:406** (pos2-only, +0.35 asymmetry): Absent at pos 4 entirely. Its upstream `0.mlp.down:2522` is likewise absent at pos 4, creating a pos2-exclusive pathway.

The total V-side asymmetry is **+0.30** in favor of pos 2. This is modest relative to the total V→O magnitude flow (~4.6 per position), meaning the mechanism relies on a small differential signal from two components while the remaining eight contribute nearly symmetrically.

## 4. Layer 3 attention: V:76 → O:806 → 3.mlp suppression

`3.attn.o:5:806` (act = -6.62) suppresses the output logit by -1.38, but its primary role is shaping the 3.mlp pathway.

### Inputs

V:76 fires at syntax/structural positions (` =` and ` :`, absent at ` ?` in this graph) and uniformly **amplifies** O:806's magnitude:

| Source | val | position |
|--------|-----|----------|
| `3.attn.v:5:76` | +1.12 | ` :` |
| `3.attn.v:1:76` | +0.21 | ` =` |

V:76's dominant upstream source at both positions is `0.mlp.down:X:328` (a programming-related component), amplifying V:76 with +1.01 (pos 1) and +0.89 (pos 5).

### O:806 → 3.mlp.up: uniform magnitude suppression

O:806 sends 18 edges to 3.mlp.up components. **All 18 are negative** — O:806 uniformly suppresses the magnitude of every downstream MLP component with zero cancellation:

| | net | sum(\|val\|) | cancellation |
|-|-----|-------------|--------------|
| O:806 → 3.mlp.up | -24.04 | 24.04 | **0%** |

Top suppressed nodes:

| Target | val | target act |
|--------|-----|-----------|
| `3.mlp.up:5:945` | -3.58 | +14.34 |
| `3.mlp.up:5:1767` | -3.12 | +12.58 |
| `3.mlp.up:5:1565` | -1.96 | -7.19 |
| `3.mlp.up:5:724` | -1.88 | +6.94 |
| `3.mlp.up:5:1005` | -1.66 | +5.56 |
| `3.mlp.up:5:2169` | -1.61 | -6.27 |
| `3.mlp.up:5:2878` | -1.41 | +5.68 |
| `3.mlp.up:5:1861` | -1.30 | -6.40 |

### Competing inputs at 3.mlp.up

The key 3.mlp.up nodes receive opposing inputs from layer 2 attention (amplifying) and O:806 (suppressing):

| Node | act | from 2.attn.o | from 3.attn.o | from other |
|------|-----|--------------|--------------|------------|
| `3.mlp.up:5:945` | +14.34 | +2.16 | -3.23 | +1.85 |
| `3.mlp.up:5:1767` | +12.58 | +1.86 | -3.28 | +4.21 |
| `3.mlp.up:5:724` | +6.94 | +4.30 | -1.68 | -0.75 |

Layer 2 attention amplifies while O:806 suppresses. The net result is determined by their competition plus contributions from earlier layers. This is a fundamentally different picture than the V→O pathway: instead of fine-grained token discrimination, this is a broad tug-of-war over 3.mlp.up magnitudes.

### 2.attn.o → 3.mlp.up: mixed effects

In contrast to O:806's uniform suppression, layer 2 attention's contribution to 3.mlp.up has **49% cancellation** (143 positive edges, 127 negative, net +13.25 from sum|val| = 26.04). This means layer 2 attention genuinely amplifies some 3.mlp.up components while suppressing others — a selective, mixed influence compared to O:806's blanket suppression.

### The secondary L3 head: O:682

`3.attn.o:5:682` (act = +1.62) receives the same V:76 inputs (+0.46, +0.08) and contributes a modest +0.17 to the output logit and +1.19 net to 3.mlp.up (18 edges). It amplifies the same MLP nodes that O:806 suppresses, adding to the tug-of-war but with much smaller magnitude.

## 5. Full pathway structure

```
Layer 0:  0.mlp.up:2:2441 (y-detector, pos2 only)
            ↓ +0.81
          0.mlp.down:*:3082, 1025, 1847 (shared across pos 2,4)
            ↓
Layer 2:  2.attn.v at pos 2,4 (8 shared + 2 pos2-only comps)
            ↓  V:627 sign-flip (+0.46 asymmetry)
            ↓  V:406 pos2-only (+0.35 asymmetry)
          2.attn.o at pos 5 (led by O:210)
            ↓ +3.02 direct          ↓ +13.25 net (49% cancel)
          output:5:340           3.mlp.up:5:*
                                   ↕ O:806 suppresses (-24.04, 0% cancel)
Layer 3:                        3.mlp.down:5:*
                                   ↓ +8.60
                                output:5:340
```

The model amplifies the ` y` logit through two main pathways:

1. **Direct path** via `2.attn.o` (+3.02): The induction-like head O:210 reads values from both variable positions, with V:627 and V:406 providing the discriminating signal that favors ` y`.

2. **Indirect path** via layer 3 MLP (+8.60): Layer 2 attention feeds into 3.mlp.up with mixed, selective amplification/suppression (+13.25 net). O:806 acts as a broad suppressor of 3.mlp.up magnitudes (-24.04). The competition between these determines which 3.mlp.down components are active, and 33 of them collectively amplify the output by +8.60 with very little cancellation.
