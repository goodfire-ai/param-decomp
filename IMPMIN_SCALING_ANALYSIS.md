# Importance-minimality: parity check + scaling analysis for the 9-layer (27-site) run

**Context.** We are about to launch a from-scratch decomposition of **9 MLP layers
(Llama-8B layers 18–26 inclusive), 3 matrices each = 27 decomposition targets (sites)**,
`C = 49,152` per site. The reference run decomposed **1 layer (3 sites)** with
`ImportanceMinimalityLoss`: `coeff = 5e-6`, `beta = 0.2`, `pnorm = 2.0` annealing to
`0.4`, `eps = 1e-12`, batch `512 × seq 2048`. The new run will use a smaller batch (TBD).

Three questions: (1) does the JAX imp-min match the torch oracle exactly; (2) how does
the per-component sparsity pressure scale with the **number of sites** (1→9 layers); (3)
how does it scale with **batch size**. Below: side-by-side parity, the equivalence-test
result, the gradient derivation, the batch math, and a recommendation table.

---

## 1. Semantic parity — JAX vs torch oracle

### Side by side

**JAX** (`param_decomp_jax/jax_single_pool/losses.py::importance_minimality_terms`,
branch `feature/jax`):

```python
def importance_minimality_terms(
    ci_upper: dict[str, Float[Array, "*leading _"]], pnorm: Float[Array, ""], eps: float
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    lp = jnp.zeros((), jnp.float32)
    entropy = jnp.zeros((), jnp.float32)
    for ci in ci_upper.values():
        ci = ci.astype(jnp.float32)  # (*leading, C)
        leading_axes = tuple(range(ci.ndim - 1))
        n_positions = math.prod(ci.shape[:-1])
        per_component_sums = jnp.sum((ci + eps) ** pnorm, axis=leading_axes)  # (C,)
        per_component_means = per_component_sums / n_positions
        lp = lp + jnp.sum(per_component_means)
        entropy = entropy + jnp.sum(per_component_means * jnp.log2(1.0 + per_component_sums))
    return lp, entropy
```

Combined in `train.py` (lines 416–417, 474):

```python
imp_lp, imp_entropy = importance_minimality_terms(ci.upper, pnorm, imp_min.eps)
imp_loss = imp_lp + imp_min.beta * imp_entropy
...
total_loss = faith_coeff * faith_loss + imp_coeff * imp_loss
```

**Torch oracle** (`git show torch-oracle:param_decomp/metrics/importance_minimality.py`):

```python
def per_component_lp_sums(ci_upper_leaky, pnorm, eps):
    out = {}
    for layer_name, layer_ci in ci_upper_leaky.items():
        result = (layer_ci + eps) ** pnorm
        out[layer_name] = result.sum(dim=tuple(range(result.dim() - 1)))
    n_examples = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return out, n_examples

def lp_and_entropy_terms(per_component_sums, n_examples):
    lp = torch.zeros(())
    entropy = torch.zeros(())
    for layer_sums in per_component_sums.values():
        per_component_mean = layer_sums / n_examples
        lp = lp + per_component_mean.sum()
        entropy = entropy + (per_component_mean * torch.log2(1 + layer_sums)).sum()
    return lp, entropy

def finalize_imp_min(per_component_sums, n_examples, beta):
    lp, entropy = lp_and_entropy_terms(per_component_sums, n_examples)
    return lp + beta * entropy
```

(The torch `Metric.update` then scales by the loss-metric `coeff` outside this function,
exactly as JAX's `imp_coeff * imp_loss`.)

### Term-by-term equivalence

| Aspect | torch oracle | JAX | match |
|---|---|---|---|
| per-component reduction over positions | `result.sum(dim=tuple(range(result.dim()-1)))` → `(C,)` | `jnp.sum(..., axis=tuple(range(ndim-1)))` → `(C,)` | ✓ |
| `eps` placement | `(ci + eps) ** pnorm` | `(ci + eps) ** pnorm` | ✓ |
| mean over positions | `layer_sums / n_examples`, `n_examples = shape[:-1].numel()` | `per_component_sums / n_positions`, `n_positions = prod(shape[:-1])` | ✓ |
| `lp` | `Σ_c per_component_mean` | `Σ_c per_component_means` | ✓ |
| entropy | `Σ_c (mean_c · log2(1 + sum_c))` | `Σ_c (mean_c · log2(1.0 + sum_c))` | ✓ |
| `log2` argument | **global per-component SUM** (not the mean) | **global per-component SUM** | ✓ |
| per-site grouping | accumulate per layer dict, `lp/entropy` summed across layers | iterate `ci_upper.values()`, summed across sites | ✓ |
| total loss | `coeff · (lp + beta·entropy)` | `imp_coeff · (imp_lp + beta·imp_entropy)` | ✓ |
| fp32 reduction | torch default fp32 | explicit `.astype(jnp.float32)` | ✓ |

**Global-sum-inside-the-log2.** Both implementations put the **full-batch per-component
sum** (not the mean) inside `log2(1 + ·)`. Under DDP the torch `ImportanceMinimalityLoss`
SUM-reduces `per_component_sums` across ranks with an autograd-aware all-reduce so the
convex `log2` sees the true full-batch sum (avoiding a per-rank Jensen upward bias). The
JAX version gets this for free: under GSPMD the `*leading` axes are the global batch, so
`jnp.sum` is already the exact global per-component sum (XLA reduces across shards inside
the graph). This is documented in the JAX docstring and matches the oracle's intent.

**Divergence found: none.** The JAX `importance_minimality_terms` is a line-for-line
semantic match to the torch oracle's `per_component_lp_sums` + `lp_and_entropy_terms`.
The only torch surface absent from JAX is the `_no_beta` diagnostic key (`lp` alone),
which the oracle emits **only on the eval path** (`Metric.compute`), never from the train
step — so it has no effect on the optimized loss. p-norm annealing
(`annealed_pnorm`) is also identical (linear ramp `pnorm → p_anneal_final_p` over
`[start_frac, end_frac]`).

### Equivalence-test result

`param_decomp_jax/jax_single_pool/tests/equivalence/test_equivalence.py::test_jax_matches_torch_reference[imp]`
runs the JAX `importance_minimality_terms` on the committed fixtures and compares against
the frozen `torch_reference.json` golden (produced by the torch oracle's
`torch_reference.py`).

```
$ .venv/bin/python -m pytest jax_single_pool/tests/equivalence/test_equivalence.py -k imp -q
.                                                                        [100%]
1 passed, 10 deselected in 17.96s
```

Numeric values on the fixtures (`IMP_P=0.4`, `IMP_BETA=0.2`, `IMP_EPS=1e-12`):

| term | JAX | torch golden | rel err |
|---|---|---|---|
| imp (`lp + beta·entropy`) | `4.23450012e+01` | `4.23450012e+01` | **0.00e+00** |
| faith | `1.00019515e-01` | `1.00019522e-01` | 7.45e-08 |
| stoch | `4.06340907e-02` | `4.06340659e-02` | 6.11e-07 |
| ppgd | `9.31269675e-02` | `9.31269377e-02` | 3.20e-07 |

The imp-min term is **bit-exact** (zero relative error) against the oracle on these
fixtures. The bf16-seam test (`test_imp_min_bf16_seam.py`) also passes. **Parity is
confirmed.**

---

## 2. Scaling with the number of sites/layers (the key question)

### The claim under test

> *"The per-component imp-min gradient is INVARIANT to the number of decomposed sites,
> because the loss is a sum of independent per-component terms; therefore to preserve the
> same effective per-component sparsity pressure when going 1→9 layers, `coeff` should
> stay `5e-6` and should NOT be divided by the total component count."*

### Setup

Let the sites be indexed `s`, components `c`, leading positions `i ∈ [1..N]` where
`N = n_positions = batch · seq` (uniform across sites in one forward). Write the upper
CI value as `a_{s,i,c} := ci_upper[s][i, c]`. Define the per-site, per-component sum over
positions:

```
S_{s,c} = Σ_i (a_{s,i,c} + eps)^p
M_{s,c} = S_{s,c} / N          (per-component mean)
```

The total imp-min loss that the optimizer sees is (`train.py:474`):

```
L = coeff · ( lp + beta · entropy )
  = coeff · Σ_s Σ_c [ M_{s,c}  +  beta · M_{s,c} · log2(1 + S_{s,c}) ]
```

Every summand depends **only on its own** `(s, c)` through `S_{s,c}` and `M_{s,c}`. There
is no cross-site or cross-component coupling: site `s` appears only in its own
`Σ_c [...]`. Adding more sites simply appends more independent additive blocks to the sum.

### Gradient of a single CI value

Differentiate `L` w.r.t. one CI value `a_{s,i,c}`. Only the `(s,c)` block depends on it.
Let `b := (a_{s,i,c} + eps)^{p-1}`, so `∂S_{s,c}/∂a_{s,i,c} = p·b` and
`∂M_{s,c}/∂a_{s,i,c} = (p·b)/N`.

**`lp` term:**

```
∂/∂a (coeff · M_{s,c}) = coeff · (p·b)/N
```

**`beta·entropy` term.** With `E_{s,c} = M_{s,c} · log2(1 + S_{s,c})`, and
`d/dS log2(1+S) = 1 / ((1+S)·ln2)`:

```
∂E/∂a = (∂M/∂a)·log2(1+S) + M·(1/((1+S)·ln2))·(∂S/∂a)
      = (p·b/N)·log2(1+S_{s,c}) + M_{s,c}·(p·b)/((1+S_{s,c})·ln2)
```

So the full per-component gradient is

```
∂L/∂a_{s,i,c} = coeff · (p·b/N) · [ 1 + beta·log2(1+S_{s,c}) + beta·S_{s,c} / ((1+S_{s,c})·N·ln2) · (N/M... ) ]
```

— but the structural point is simpler than the algebra: **`∂L/∂a_{s,i,c}` depends only on
`coeff`, `p`, `eps`, `N`, and the quantities `a_{s,i,c}`, `S_{s,c}`, `M_{s,c}` of that one
site's one component.** It contains **no factor of the number of sites and no factor of
the total component count `Σ_s C_s`.** Adding 8 more layers does not change any term in
this expression for a component that is held at the same CI distribution.

### Verdict on the claim

**Confirmed.** Both the `lp` term and the `beta·entropy` term are per-component-local:

- `lp` contributes `coeff·(p·b)/N` — no site-count dependence.
- `beta·entropy` couples a component **only to its own** `S_{s,c}` (through `log2(1+S)`
  and the `1/(1+S)` chain term). It does **not** couple to other components or other
  sites. The "entropy"/`log2` term is local, not global-across-components.

Because the loss is `coeff · Σ_s Σ_c (independent per-component block)`, the per-component
gradient — i.e. the actual sparsity pressure each component feels — is **invariant to the
number of decomposed sites.** Going 1→9 layers (3→27 sites) does not dilute or amplify
the pressure on any individual component; it just adds more independently-penalized
components to the objective.

**Therefore: keep `coeff = 5e-6`. Do NOT divide by the site count or by the total
component count `27 · 49152`.** Dividing would weaken each component's sparsity pressure
by 9× (relative to the reference run), which is exactly the wrong move — the per-component
penalty was calibrated on the 1-layer run and the math says it transfers unchanged.

A useful sanity intuition: `coeff` is the price per unit of *one component's* mean
importance. That price is a property of a component, not of how many components exist.
The total loss magnitude will grow ~9× (9× more additive blocks), and so will the total
imp-min gradient-norm summed over all parameters — but that is the correct behavior:
there are 9× more components to keep sparse, each penalized at the same rate. The faith
and recon terms are themselves normalized (faith by `Σ numel`, recon by `n_positions`),
so the relative balance among the three loss families is preserved per-component.

---

## 3. Scaling with batch size

`N = n_positions = batch · seq`. Look at how `lp` and `entropy` depend on `N`, holding the
CI **distribution** fixed (i.e. the typical per-position CI value is a property of the
model, not of how many positions we average over).

### `lp` is batch-invariant

```
lp = Σ_s Σ_c M_{s,c} = Σ_s Σ_c (1/N) Σ_i (a_{s,i,c}+eps)^p
   = Σ_s Σ_c  E_i[(a_{s,i,c}+eps)^p]
```

`lp` is a **mean over positions**. As `N` changes, the mean over a fixed CI distribution
is unchanged in expectation (only its sampling variance shrinks with larger `N`). So `lp`
— the beta-independent sparsity proxy — is **batch-invariant**. Its gradient
`coeff·(p·b)/N` per position is `1/N`, but there are `N` positions, so the summed `lp`
gradient per component is also batch-invariant in expectation. **Shrinking the batch does
not change the `lp` pressure.**

### `entropy` has a `log2(N)` dependence

```
entropy = Σ_s Σ_c M_{s,c} · log2(1 + S_{s,c})
```

Here `S_{s,c} = N · M_{s,c}` is the **global sum**, which grows linearly with `N`. Write
`m := M_{s,c}` (the batch-invariant mean). Then

```
entropy term per component = m · log2(1 + N·m)
```

For `N·m ≫ 1` (the usual regime: even a modestly-alive component over `N = 512·2048 ≈
1.05M` positions has `N·m` enormous),

```
m · log2(1 + N·m) ≈ m · [ log2(N) + log2(m) ]   (when N·m ≫ 1)
```

So the entropy term carries an explicit **`+ m·log2(N)`** piece. Changing the batch from
`N_ref` to `N_new` shifts each component's entropy contribution by approximately

```
Δentropy_{s,c} ≈ m_{s,c} · ( log2(N_new) − log2(N_ref) ) = m_{s,c} · log2(N_new / N_ref)
```

i.e. the entropy term scales by `+ log2(N_new/N_ref)` **bits per unit of mean CI**, and
the full imp-min contribution of the entropy family shifts by
`coeff · beta · Σ_{s,c} m_{s,c} · log2(N_new/N_ref)`.

### How big is the batch effect, with beta = 0.2?

The total imp-min loss is `coeff·(lp + beta·entropy)`. The entropy term's coefficient
inside the brackets is `beta = 0.2`. A batch change multiplies the **`log2(N)` part** of
each entropy summand, not the whole loss. Concretely, halving or quartering the reference
batch (`512 → 256` or `512 → 128`, seq fixed):

| batch change | `log2(N_new/N_ref)` | shift in entropy term | per-component imp-min shift (`beta·Δentropy / [lp + beta·entropy]`) |
|---|---|---|---|
| 512 → 256 (½×) | `−1` bit | `−1 · m` per comp | entropy family weakens by `0.2·m` per comp |
| 512 → 128 (¼×) | `−2` bits | `−2 · m` per comp | entropy family weakens by `0.4·m` per comp |
| 512 → 256 → 9-layer (no change to per-comp `m`) | `−1` | as above | as above |

To put a number on the relative size: the entropy contribution per component is
`m·log2(1+N·m)`. At the reference scale `N_ref = 512·2048 ≈ 1.05e6`, `log2(N_ref) ≈ 20.0`
bits. The CI-distribution-dependent `log2(m)` offset is the same regardless of batch, so
the **relative** change in the entropy term from `512 → 256` is
`Δlog2 / log2(1+N·m) ≈ −1 / (20 + log2(m))`. For an alive-ish component
(`m ~ 0.01`, `log2(m) ≈ −6.6`), that is `≈ −1/13.4 ≈ −7.5%` of the entropy term; for a
near-dead component (`m ~ 1e-4`, `log2(m) ≈ −13.3`) it's `≈ −1/6.7 ≈ −15%`. Because the
entropy term is itself only `beta = 0.2`-weighted inside the bracket, the effect on the
**total** imp-min loss is roughly a quarter to a half of those percentages.

### Is this material?

**Marginally — and in the direction that makes a smaller batch slightly *weaker* on the
entropy (frequency-sparsity) component, not the `lp` component.** Quantitatively, going
`512 → 256` shifts the entropy term down by ~7–15% (component-dependent), and the total
imp-min loss by roughly a few percent. It does **not** rescale the dominant `lp` sparsity
proxy at all. This is well within the noise of a from-scratch run and does **not** by
itself warrant a `coeff` change. If we want to hold the entropy term's `log2(N)` operating
point exactly fixed across a `512 → 128` change, the only knob that touches it is `beta`,
and the correction would be tiny (a ≤2-bit shift on a ~20-bit quantity). Not worth it.

---

## 4. Recommendation table

| knob | reference (1 layer / 3 sites, batch 512) | 9 layers / 27 sites | rationale |
|---|---|---|---|
| **`coeff`** | `5e-6` | **`5e-6` (UNCHANGED)** | Per-component gradient is invariant to site count; the loss is `coeff·Σ_s Σ_c(independent block)`. Do NOT divide by `27` or by `27·49152`. Dividing would weaken per-component pressure 9×. |
| **`beta`** | `0.2` | `0.2` (unchanged) | Per-component-local; site-count invariant. Batch change shifts entropy by only ≤2 bits on a ~20-bit term — not worth a `beta` nudge. |
| **`pnorm`** | `2.0 → 0.4` (annealed) | unchanged | Annealing schedule is per-component; identical math, no scaling interaction. |
| **`eps`** | `1e-12` | unchanged | Inside `(ci+eps)^p`; numerical, scale-free. |
| **batch** | `512 × 2048` | smaller (e.g. 128–256) is fine | `lp` is batch-invariant (mean over positions). Entropy carries a `+log2(N)` term: `512→256` shifts it ~−7–15% per component (`512→128` ~−15–30%), total imp-min a few %. Immaterial; **no `coeff`/`beta` compensation needed.** If you ever want it exactly fixed, nudge `beta`, not `coeff`. |

### One-line bottom line

**Keep `coeff = 5e-6`, `beta = 0.2`, `pnorm 2.0→0.4`, `eps = 1e-12` exactly as the
1-layer reference.** The imp-min loss is a sum of independent per-component blocks, so its
per-component sparsity pressure is invariant to the jump from 3→27 sites — dividing the
coeff by the component count would be a 9× under-penalty bug. A smaller batch only shifts
the `beta`-weighted entropy term by a sub-`log2(N)` amount and leaves the dominant `lp`
proxy untouched; no compensation is warranted. The JAX implementation is a confirmed,
bit-exact semantic match to the torch oracle (equivalence test passes, rel err 0.00e+00).
