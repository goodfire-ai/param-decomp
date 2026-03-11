# Attribution Graph Analysis: Graph 10

## Header

| Field | Value |
|-------|-------|
| Graph ID | 10 |
| Prompt | ` Dallas , Texas . Cleveland ,` |
| Target token | ` Ohio` (id 10128) at position 5 |
| Graph type | Optimized (CE loss, 2000 steps, p=0.3) |
| L0 | 263 active components |
| Edges | 5,494 |
| CI-masked prob | 0.9878 |
| Stoch-masked prob | 0.9666 |
| Adv PGD prob | 0.5625 |

## Output Node Attribution Breakdown

The model predicts ` Ohio` with 98.8% probability after the prompt ` Dallas , Texas . Cleveland ,`. The graph is optimized to explain this prediction.

### Layer Summary

| Source Layer | Sum Val | Sum |Val| | # Edges |
|-------------|---------|-----------|---------|
| 3.mlp.down | +4.511 | 10.245 | 43 |
| 2.attn.o | +2.271 | 2.271 | 21 |
| 2.mlp.down | +1.114 | 1.791 | 7 |
| 3.attn.o | +0.621 | 0.621 | 4 |
| 1.mlp.down | +0.434 | 0.625 | 4 |
| 0.mlp.down | +0.274 | 0.864 | 8 |
| embed | -0.340 | 0.340 | 1 |
| **Total** | **+8.924** | | **89** |

**Dominant pathways:**
1. **Layer 3 MLP** (43 components, net +4.5) — the largest contributor by far, with a mix of positive and negative attributions. The single largest edge is `3.mlp.down:5:3532` at -2.109 (a high-density baseline bias component).
2. **Layer 2 attention** (21 O components, net +2.3) — the main cross-sequence pathway, dominated by `2.attn.o:5:845` (+0.867).
3. **Layer 3 attention** (4 O components, net +0.6) — secondary cross-sequence pathway via `3.attn.o:5:399` (+0.420).

### Top Individual Edges to Output

| Source | Attribution |
|--------|------------|
| `3.mlp.down:5:3532` | -2.109 |
| `2.attn.o:5:845` | +0.867 |
| `3.mlp.down:5:1053` | +0.852 |
| `3.mlp.down:5:3102` | +0.656 |
| `3.mlp.down:5:2790` | +0.531 |
| `3.mlp.down:5:502` | -0.473 |
| `2.mlp.down:5:1560` | +0.449 |
| `3.attn.o:5:399` | +0.420 |

## Layer 2 Attention: The Primary Cross-Sequence Pathway

### V → O Breakdown

The dominant L2 O node `2.attn.o:5:845` receives V inputs almost exclusively from position 4 (Cleveland):

| Source Position | Token | Sum V Attribution | # V components |
|----------------|-------|-------------------|----------------|
| 4 | Cleveland | -1.018 | 25 |
| 2 | Texas | -0.391 | 2 |
| 0 | Dallas | -0.047 | 1 |

The attention is overwhelmingly focused on Cleveland, with minor contributions from Texas and negligible from Dallas.

Other L2 O nodes (`2.attn.o:5:828`, `2.attn.o:5:606`, `2.attn.o:5:8`) show similar patterns — all dominated by Cleveland V components with small Texas contributions.

### K → O Analysis (Attention Routing)

K edges into `2.attn.o:5:845` come **exclusively from position 4 (Cleveland)** with total K attribution of -1.617 across 9 K components. The dominant K component is `2.attn.k:4:206` (-0.836), a very high-density component (fires ~80% of tokens) that acts as a generic attention bias.

This is notable: the L2 attention head is not attending symmetrically to both city-state pairs. It strongly attends to Cleveland specifically, likely because Cleveland is the most recent city token before the prediction position.

### V-Side Asymmetry

Since attention routing is Cleveland-focused (not symmetric), the V-side asymmetry here is less about "why Cleveland and not Dallas" and more about "what information about Cleveland is being written into the residual stream."

Key V components active at Cleveland (pos 4):

| Component | Total Cross-Seq Attr | Interpretation |
|-----------|---------------------|----------------|
| `2.attn.v:4:686` | +0.529 | Fires on tokens starting with 'C'/'c', produces suffix completions |
| `2.attn.v:4:549` | +0.492 | Word stem completion (first part of incomplete words → suffixes) |
| `2.attn.v:4:416` | +0.489 | Incomplete words → completions (e.g., Bluetooth, glycemic) |
| `2.attn.v:4:920` | +0.485 | Word stems/fragments → suffixes (hypothal→amus, infil→tration) |
| `2.attn.v:4:345` | -0.804 | Word stem/name completion (taxp→ayer, citiz→en) |
| `2.attn.v:4:757` | -0.702 | Geographic entity completion (Venezuel→a, cryptocur→rency) |
| `2.attn.v:4:672` | -0.280 | Word suffix completion (earthqu→ake, infil→tration) |

The **shared component** `2.attn.v:757` is active at both Texas (pos 2) and Cleveland (pos 4):
- At Texas: total cross-seq attr = -0.982
- At Cleveland: total cross-seq attr = -0.702
- This is a **geographic entity completion** component. Its stronger activation at Texas makes sense — "Texas" is a more prototypical geographic completion target.

## Layer 3 Attention: Secondary Cross-Sequence Pathway

`3.attn.o:5:399` (interpretation: "proper noun completions like Angeles, Aires, Francisco, Lanka") receives:

| Source | V Attribution |
|--------|--------------|
| `3.attn.v:4:403` | +1.516 (Cleveland) |
| `3.attn.v:2:403` | +0.164 (Texas) |
| `3.attn.v:0:403` | +0.130 (Dallas) |

Component 403 is a **word stem completion** component. It's active at all three city/state positions but contributes overwhelmingly from Cleveland — again because attention routing (K) strongly favors Cleveland at `3.attn.k:4:145` (+1.273).

The L3 attention pathway feeds into the L3 MLP via `3.mlp.up:5:2554` (attr +0.836 from `3.attn.o:5:399`), which then feeds `3.mlp.down:5:2790` — the **geographic state name** component (outputs: Zealand, Angeles, York, Aires, Arabia, Ohio).

## Layer 0 MLP: The Earliest Differentiators

Layer 0 MLP components show the earliest asymmetry between positions:

| Component | Dallas (0) | Texas (2) | Cleveland (4) | Interpretation |
|-----------|-----------|-----------|---------------|----------------|
| `0.mlp.down:1377` | NOT ACTIVE | act=-3.67 | act=-2.47 | Geographic/political proper noun first halves (United, Czech, Hawaii) |
| `0.mlp.down:2344` | NOT ACTIVE | act=-2.48 | act=-1.81 | Capitalized tokens and proper noun parts (New, United, Gl) |
| `0.mlp.down:3455` | NOT ACTIVE | NOT ACTIVE | act=-3.28 | Word stems with high precision (extremely selective) |
| `0.mlp.down:3523` | NOT ACTIVE | NOT ACTIVE | act=-1.81 | Word prefixes (disappe→, c→) |
| `0.mlp.down:3257` | act=+2.80 | act=+3.17 | act=+4.57 | Content words (nouns, verbs, adjectives) — active everywhere |

**Critical finding:** `0.mlp.down:1377` (geographic proper noun detector) is active at Texas and Cleveland but **not at Dallas**. This is the earliest point in the network where the city tokens are differentiated. It feeds directly into the key V components:
- → `2.attn.v:4:757` with attr +3.625 (the geographic entity completion V component)
- → `2.attn.v:2:757` with attr +5.063 (same component at Texas position)

Similarly, `0.mlp.down:3455` and `3523` fire only at Cleveland, feeding `1.attn.v:4:428` (word continuation) which provides a dedicated Cleveland → output pathway through L1 attention.

## L1 Attention: Cleveland-Specific Sub-Pathway

`1.attn.v:4:428` is a **word continuation** component active only at Cleveland. It feeds `1.attn.o:5:37` with attr +0.871. From there, `1.attn.o:5:37` feeds into L3 MLP components:
- → `3.mlp.up:5:2554` with +0.447 (which feeds the geographic state name MLP component)
- → `1.mlp.up:5:477` with -4.750 (large magnitude, feeds further MLP processing)

This is upstream of `0.mlp.down:4:2344` (capitalized tokens) and `0.mlp.down:4:3455` (word stems), both Cleveland-only L0 components.

## Layer 3 MLP: The Final Output Stage

The L3 MLP contributes the most total attribution (+4.511 net) through 43 components. Key ones:

| Component | Attr → Output | Interpretation |
|-----------|--------------|----------------|
| `3.mlp.down:5:3532` | -2.109 | 99.5% firing rate — broad baseline bias. Suppresses Ohio logit. |
| `3.mlp.down:5:1053` | +0.852 | Biographical contexts — promotes Ohio logit |
| `3.mlp.down:5:3102` | +0.656 | Fires on commas — promotes geographic continuations after commas |
| `3.mlp.down:5:2790` | +0.531 | Geographic state names (Zealand, Angeles, York, Arabia) — key Ohio promoter |
| `3.mlp.down:5:502` | -0.473 | 47% firing rate — broad function word bias, suppresses Ohio |

The pattern is clear: high-density baseline components (`3532`, `502`) provide suppressive pressure, while more selective components (`1053` for biographical contexts, `3102` for comma-following geography, `2790` for US state names) provide the positive push toward Ohio.

## Interpretation: Why Does the Model Predict " Ohio"?

The model predicts " Ohio" after " Dallas , Texas . Cleveland ," through a multi-layered mechanism:

### 1. Pattern Recognition (Layer 0)
The prompt establishes a **"City, State"** pattern. Layer 0 MLP components detect geographic proper nouns (`0.mlp.down:1377`) and capitalized tokens (`0.mlp.down:2344`) at the Texas and Cleveland positions. Notably, some components fire only at Cleveland (`3455`, `3523`), creating position-specific representations from the earliest layer.

### 2. Information Transfer via Attention (Layers 1-3)
Layer 2 attention is the primary cross-sequence information pathway. The attention heads at position 5 (the comma after Cleveland) **strongly attend to position 4 (Cleveland)** — not symmetrically to both cities. This makes sense: the model needs to know what city was most recently mentioned to predict its corresponding state.

The V components that carry information from Cleveland are primarily **word/entity completion** components — they encode that Cleveland is an incomplete geographic entity that needs its state suffix.

Layer 3 attention provides a secondary pathway through a **proper noun completion** component (`3.attn.o:5:399`), also focused on Cleveland.

### 3. State Name Production (Layer 3 MLP)
The final MLP layer transforms the Cleveland-specific information into the actual " Ohio" prediction. The `3.mlp.down:5:2790` component is a dedicated **US state name producer** (outputs: Zealand, Angeles, York, Ohio, Arabia). It receives input from both the L3 attention pathway and earlier MLP processing.

### 4. The In-Context Learning Aspect
The " Dallas , Texas ." prefix serves as an **in-context example** that establishes the pattern. While the graph focuses on the Cleveland→Ohio pathway, the Texas position contributes through shared V components (especially `2.attn.v:757`, the geographic entity completion component) that help the model recognize this as a city→state completion task.

### Key Mechanistic Insight
The asymmetry is **not** primarily on the V side (as is often the case in attribution graphs). Instead, the **attention routing itself is asymmetric** — K components at Cleveland dominate the attention pattern. The model has learned that the relevant city for state prediction is the most recent one (Cleveland at position 4), not the earlier one (Dallas at position 0). The V components then carry Cleveland-specific information (geographic entity features) to the output position, where specialized MLP components convert it to " Ohio".
