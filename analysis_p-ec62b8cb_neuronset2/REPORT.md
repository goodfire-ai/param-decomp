# Subcomponent support on layer-18 MLP neurons (neuron set 2)

**Run:** `goodfire/param-decomp-llama/p-ec62b8cb`  (checkpoint `model_100000.pth`)
**Target model:** meta-llama/Llama-3.1-8B
**Decomposed module:** `model.layers.18.mlp` — `gate_proj`, `up_proj`, `down_proj`, each C=24576 subcomponents (global shared-transformer CI, `use_delta_component=true`).
**MLP neuron axis:** d_ffn = 14336.
**Target neurons (15):** 8791, 11651, 12527, 391, 2405, 11942, 4549, 2923, 14013, 13662, 6230, 10722, 4287, 12898, 13598.

## Method

Each subcomponent is a rank-1 outer product `W_c = V[:,c] ⊗ U[c,:]`. The residual-stream-touching
vector is normalised to unit L2 norm and the MLP-neuron-touching vector is rescaled by that same
norm (the outer product is scale-invariant, so this fixes a canonical scale). The **connecting
weight** to neuron *n* is the entry of the rescaled MLP-side vector at index *n*. Subcomponents are
ranked by `|connecting weight|`.

- `gate_proj` / `up_proj` (d_model→d_ffn): residual side = `V[:,c]` (d_model=4096); MLP side =
  `U[c,:]` (d_ffn=14336). Connecting weight = `U[c,n]·‖V[:,c]‖`.
- `down_proj` (d_ffn→d_model): residual side = `U[c,:]` (d_model); MLP side = `V[:,c]` (d_ffn).
  Connecting weight = `V[n,c]·‖U[c,:]‖`.

## Aggregate top subcomponents across the neuron set

`max |w|` = strongest single connection to any neuron in the set; `sum |w|` = total connection
magnitude across all 15 neurons (broad support).

### gate_proj
- by **max |w|**: c6731(0.176), c7548(0.160), c13603(0.150), c19773(0.141), c12602(0.135), c17298(0.126), c5420(0.123), c11229(0.118), c13881(0.103), c7478(0.100)
- by **sum |w|**: c11229(0.767), c13603(0.654), c7478(0.641), c17747(0.589), c5147(0.573), c7548(0.572), c6731(0.566), c23529(0.559), c12602(0.544), c4112(0.544)

### up_proj
- by **max |w|**: c422(0.122), c5065(0.118), c3608(0.113), c19800(0.111), c19132(0.099), c4821(0.098), c3999(0.097), c3194(0.096), c12433(0.092), c12194(0.090)
- by **sum |w|**: c19132(0.396), c12433(0.396), c4821(0.374), c422(0.368), c5065(0.342), c7130(0.342), c1025(0.324), c12194(0.310), c17957(0.309), c24323(0.305)

### down_proj
- by **max |w|**: c4368(0.095), c10852(0.089), c24322(0.082), c24102(0.080), c18329(0.078), c14730(0.076), c5658(0.075), c961(0.073), c14167(0.064), c14201(0.064)
- by **sum |w|**: c5658(0.308), c24322(0.306), c11143(0.302), c24102(0.296), c5946(0.295), c21045(0.294), c15225(0.291), c18329(0.290), c4851(0.287), c18606(0.284)

## Files
- `top_subcomponents.csv` — top-50 subcomponents per (proj, neuron) with signed + abs weights.
- `per_neuron_top.json` — top-20 per (proj, neuron).
- `conn_matrices.pt` — `{proj: tensor[24576, 15]}` of signed connecting weights (columns ordered as the target-neuron list above).
