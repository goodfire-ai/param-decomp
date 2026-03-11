# Graph 9 — Attribution Analysis: " Dallas , Texas . Cleveland ," → " Ohio"

## Overview

| Property | Value |
|----------|-------|
| Graph ID | 9 (optimized) |
| Prompt | ` Dallas , Texas . Cleveland ,` |
| Target token | ` Ohio` (id 10128) at position 5 |
| L0 | 295 active components |
| Total edges | 6,527 |
| Max |attribution| | 9.46 |
| CI-masked prob | 0.990 |
| Stoch-masked prob | 0.986 |
| Adversarial PGD prob | 0.605 |

This is a **geographic association** task: the model must recognize that "Cleveland" is a city
and predict its associated state ("Ohio"), using the pattern established by "Dallas, Texas."

## Output Node Attribution Breakdown

| Source layer | sum(val) | sum(|val|) | n edges |
|-------------|----------|------------|---------|
| 3.mlp.down | +4.64 | 10.55 | 46 |
| 3.attn.o | -3.02 | 4.82 | 6 |
| 2.attn.o | +2.44 | 2.44 | 22 |
| 2.mlp.down | +1.11 | 1.79 | 7 |
| 0.mlp.down | +0.27 | 0.86 | 8 |
| 1.mlp.down | +0.43 | 0.63 | 4 |
| embed | -0.34 | 0.34 | 1 |

The dominant pathway is **3.mlp.down** (net +4.64), which serves as the primary "output head"
projecting the residual stream into vocabulary space. Layer-3 attention is the second-largest
contributor by |val| but is net-negative (-3.02), acting as a **suppressive** force. Layer-2
attention contributes a net positive +2.44.

### Top individual edges to output

| Source | val |
|--------|-----|
| 3.attn.o:5:806 | -3.92 |
| 3.mlp.down:5:3532 | -2.11 |
| 2.attn.o:5:845 | +0.87 |
| 3.mlp.down:5:1053 | +0.85 |
| 3.mlp.down:5:3102 | +0.66 |

## Attention Mechanism Analysis

### Layer-3 Attention: Two Opposing O Components

The layer-3 attention block has two key O components with **opposing roles**:

**O:806 (suppressive, -3.92 → output):**
- Receives large V contributions from Dallas (pos 0): -3.66
- Also from Cleveland (pos 4): -1.81
- All V contributions are negative → this head writes a negative direction that suppresses the output logit
- However, O:806 also feeds heavily into 3.mlp.up (+13.9 total), meaning its "suppression"
  at the output is partially offset by its constructive contribution through the MLP pathway

**O:399 (constructive, +0.42 → output):**
- Receives large positive V from Cleveland (pos 4): +1.55
- Small positive from Texas (pos 2): +0.19
- Near-zero from Dallas (pos 0): -0.01
- Feeds into 3.mlp.up with net -3.84 (opposite sign to O:806)

**Key insight**: O:806 and O:399 form a complementary pair that jointly modulate the
3.mlp.up layer. They share V component 403, which produces opposite-sign outputs to the
two O nodes — a **sign-flip detector** pattern.

### Layer-2 Attention: O:845 (constructive, +0.87 → output)

- Attends mainly to Cleveland (pos 4, V_sum = -1.01) and Texas (pos 2, V_sum = -0.39)
- Despite the negative V sums, the O:845 component inverts the sign, producing a positive
  contribution to the output
- K routing focuses exclusively on Cleveland (pos 4, K_sum = -1.63)

### V-Side Asymmetry: Why " Ohio" and Not Something Else

The model sees both "Dallas" (pos 0) and "Cleveland" (pos 4) as city names, but must
produce different state associations for each. The asymmetry analysis reveals:

**Position-unique V components** (fire at Cleveland only):
- 2.attn.v:345 — Large negative contributor (-0.78), fed by `0.mlp.down:4:3455` (-2.84)
- 2.attn.v:672, 573, 365 — Additional Cleveland-specific components

**Shared component with asymmetric strength — V:403 (layer 3):**
- Dallas (pos 0): -0.07
- Cleveland (pos 4): -0.74
- 10x stronger at Cleveland, creating net asymmetry

**Dallas-unique component — V:677 (layer 3):**
- Dallas only: -3.68 total (the single largest V contribution in the graph)
- Zero at Cleveland
- This is the key "Dallas identity detector" — it fires strongly only for Dallas and
  drives the suppressive O:806 head

**Cleveland-favoring components** (positive at Cleveland, absent at Dallas):
- 2.attn.v:47 (+0.64), 2.attn.v:686 (+0.52), 2.attn.v:549 (+0.50), 2.attn.v:416 (+0.50)
- These are Cleveland-specific feature detectors that constructively contribute to " Ohio"

### K-Side Analysis (Attention Routing)

**Layer-3 K routing** is roughly symmetric between positions:
- K:145 routes attention to both Cleveland (+1.27 to O:399, +0.31 to O:806) and Texas (+0.16, +0.12)
- The routing itself doesn't strongly differentiate — **asymmetry comes from the V side**

**Layer-2 K routing** focuses exclusively on Cleveland (pos 4), with K:206 (-0.84) as the dominant component.

## Upstream Tracing: Origins of Asymmetry

### Layer 0 MLP — The Token Identity Layer

The earliest differentiation happens at **layer-0 MLP**, where different `0.mlp.down` components
activate for different city tokens:

| Component | Role |
|-----------|------|
| 0.mlp.down:4:3455 | Cleveland-specific, feeds 2.attn.v:345 (-2.84) and 3.attn.v:403 (-1.88) |
| 0.mlp.down:4:3257 | Feeds 1.attn.v:428 (-0.45) and 2.attn.v:345 (+0.95) — opposing roles |
| 0.mlp.down:4:1377 | Feeds 2.attn.v:757 (+3.63) — the shared Texas/Cleveland component |
| 0.mlp.down:4:2344 | Feeds 1.attn.v:428 (+1.41) — Cleveland pathway |
| 0.mlp.down:0:3257 | Dallas-specific, feeds 3.attn.v:677 (-0.08) |

These layer-0 MLP components are themselves fed by **embed:4:0** (the Cleveland embedding)
and **0.attn.o:4:389** (layer-0 attention, contributing contextual information from prior tokens).

### Full Path: embed → 0.mlp → V → O → 3.mlp → output

The complete dominant pathway for the Cleveland → Ohio prediction:
1. **embed:4:0** (Cleveland token embedding) activates layer-0 MLP up components
2. **0.mlp.down:4:3455/2344/1377** (city-identity features) produce Cleveland-specific representations
3. **2.attn.v:345/757** and **3.attn.v:403** (value components) encode city-to-state mappings
4. **2.attn.o:845** and **3.attn.o:399/806** (output projections) write into the residual stream at pos 5
5. **3.mlp.up/down** (final MLP) transforms the attended features into the ` Ohio` logit

## Secondary Pathways

### 3.mlp.down:5:3532 (second-largest edge, -2.11)

This MLP component acts as a **suppressor**, partially counteracting the positive MLP contributions.
It receives mixed-sign inputs from 3.mlp.up components, some of which are themselves driven
by O:806 (the suppressive attention head).

### Layer-1 attention (1.attn.v:428)

A secondary attention pathway at layer 1 sends Cleveland information (V_sum = +0.95) forward.
This component is fed primarily by 0.mlp.down:4:2344 (+1.41) and 0.mlp.down:4:3455 (+0.78).

## Interpretation

The model predicts " Ohio" after " Cleveland ," through a multi-layer mechanism:

1. **Token identity encoding** (layer 0): The Cleveland embedding activates a distinct set of
   layer-0 MLP components (3455, 2344, 1377) that differ from those activated by "Dallas."
   This is where geographic identity is first established.

2. **Cross-token information transfer** (layers 2-3): Attention heads at layers 2 and 3 look
   back from position 5 (the comma after Cleveland) to retrieve city-specific features.
   V components carry the city→state mapping information, while K components route attention
   to the relevant source positions.

3. **Asymmetry mechanism**: The prediction of "Ohio" vs "Texas" is driven primarily by the
   **V side**, not the K side. The same attention heads attend to both Dallas and Cleveland,
   but the V values written are different because different layer-0 MLP features are active
   at each city position. Key asymmetry sources:
   - V:677 (layer 3): fires only at Dallas, creating strong negative contribution
   - V:345 (layer 2): fires only at Cleveland, creating a Cleveland-specific pathway
   - V:403 (layer 3): fires at both but 10x stronger at Cleveland

4. **Output computation** (layer-3 MLP): The final MLP reads from the residual stream at
   position 5, which now contains the attended city-specific information, and projects it
   into vocabulary space to boost the " Ohio" logit. The net positive contribution (+4.64)
   from 3.mlp.down is the largest single-layer contributor to the output.

5. **Suppressive regulation**: O:806 directly suppresses the output (-3.92) but simultaneously
   excites 3.mlp.up (+13.9). This suggests a **"suppress-then-refine"** pattern where the
   attention output is partially cancelled at the direct path but amplified through the
   MLP pathway, allowing the MLP to perform further nonlinear processing of the attended features.
