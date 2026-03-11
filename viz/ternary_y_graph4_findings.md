# Graph Analysis: `z = y ? x : y` → predicts ` y`

**Graph 4** (optimized) from prompt ID 2. L0=176, 1846 edges.
Optimization: CE loss on token ` y` (id 340) at position 5, imp_min_coeff=0.05, 2000 steps, pnorm=0.3.

Key metrics: CI-masked prob=0.999, stochastic=0.985, adversarial PGD=0.567.

## 1. Output node attribution breakdown

All 42 edges into `output:5:340` (` y`), grouped by source layer:

| Layer | sum(val) | sum(\|val\|) | n |
|-------|----------|-------------|---|
| `3.mlp.down` | +5.17 | 10.74 | 23 |
| `2.attn.o` | +2.98 | 2.99 | 7 |
| `3.attn.o` | -0.89 | 1.19 | 2 |
| `1.mlp.down` | +0.61 | 0.61 | 1 |
| `1.attn.o` | -0.35 | 0.35 | 1 |
| `0.mlp.down` | -0.10 | 0.25 | 5 |

Top individual edges:

| Source | val |
|--------|-----|
| `2.attn.o:5:1015` | +2.31 |
| `3.mlp.down:5:341` | +1.94 |
| `3.mlp.down:5:3532` | -1.42 |
| `3.attn.o:5:805` | -1.04 |
| `3.mlp.down:5:1443` | +0.89 |
| `3.mlp.down:5:723` | +0.77 |
| `3.mlp.down:5:817` | -0.69 |
| `3.mlp.down:5:361` | -0.67 |
| `1.mlp.down:5:1217` | +0.61 |

The two dominant pathways are **layer 2 attention** (net +2.98, led by O component 1015) and **layer 3 MLP** (net +5.17, but with large cancellation — both strong positive and negative contributors). Layer 3 attention provides modest negative attribution (-0.89).

## 2. Induction-like components on source tokens

Both pos 2 (` y`) and pos 4 (` x`) activate the same two `2.attn.v` components: **0** and **391**. Both have CI=1.0 at both positions.

### V → O cross-sequence edges (to position 5)

All outgoing edges are cross-sequence to `2.attn.o` at pos 5:

| Target | v:2:0 | v:4:0 | v:2:391 | v:4:391 |
|--------|-------|-------|---------|---------|
| `o:5:1015` | +0.10 | +0.06 | **+0.96** | **+0.29** |
| `o:5:216` | -0.11 | -0.07 | -0.23 | -0.07 |
| `o:5:354` | -0.03 | -0.01 | **-0.39** | -0.12 |
| `o:5:666` | +0.02 | +0.01 | -0.13 | -0.04 |

Component 391 carries much larger attributions than component 0. The pos 2 (` y`) V nodes attribute ~3x more than pos 4 (` x`) to the dominant O node 1015.

### What feeds the V components

Both V nodes receive from the same upstream MLP-0 components at their respective positions:

| Source comp | → v:2:0 | → v:4:0 | → v:2:391 | → v:4:391 |
|-------------|---------|---------|-----------|-----------|
| `0.mlp.down:X:606` | +2.77 | +2.45 | -1.10 | -1.28 |
| `0.mlp.down:X:51` | +0.49 | +1.02 | -1.94 | **-4.72** |
| `0.mlp.down:X:1891` | -0.08 | +0.18 | +1.44 | +0.53 |
| `1.attn.o:X:630` | -0.54 | -0.19 | +1.05 | +0.01 |
| `1.mlp.down:X:1217` | +0.20 | +0.03 | -0.57 | +0.07 |

### K → O cross-sequence edges (attention routing)

6 K components are shared between pos 2 and pos 4 (206, 224, 286, 313, 320, 439). Pos 2 has one additional component (1).

Total K attribution to `2.attn.o:5:1015` by source position:

| Position | Token | sum(val) |
|----------|-------|----------|
| 0 | `z` | -1.65 |
| 1 | ` =` | +0.00 |
| 2 | ` y` | +4.66 |
| 3 | ` ?` | +0.01 |
| 4 | ` x` | +5.04 |

**The K components don't favor pos 2 over pos 4.** Both variable positions get roughly equal positive K attribution (pos 4 slightly higher). The K/Q mechanism says "attend to both variable-value positions, suppress non-variable positions" — pos 0 (`z`) gets -1.65.

### Interpretation

The asymmetry that makes the output ` y` instead of ` x` does not come from attention routing. Both positions are attended to roughly equally. Instead it comes from the **V side**: the same V component indices (0, 391) fire at both positions but receive different upstream activations (since the tokens differ), producing different value vectors. The values from pos 2 (` y`) contribute ~3x more to the dominant `2.attn.o:5:1015` than pos 4 (` x`). The output logits are effectively a weighted sum of both contributions, with ` y`-encoding values dominating.

## 3. Layer 3 attention: `3.attn.o:5:805`

This node contributes -1.04 to the output (suppressing ` y`). Its dominant input is `3.attn.v:0:224` (-2.25, cross-seq from pos 0 `z`), with smaller contributions from the same V component 224 at pos 2 (-0.34) and pos 4 (-0.28), plus `3.attn.v:5:573` (-1.02, self-attention). Its main outgoing effect is driving `3.mlp.up` at pos 5 (10 edges, up to 2.5 magnitude), not its direct -1.04 to the output. All four components involved are high-frequency: `3.attn.o:805` (87%, unclear), `3.attn.v:224` (30%, "predictable word/subword completion"), `3.attn.v:573` (16%, "source code domain bias"), `3.attn.k:413` (80%, unclear).
