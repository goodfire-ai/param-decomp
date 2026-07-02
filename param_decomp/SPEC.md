# Single-pool VPD — semantics spec

Pins the **meaning** of the single-pool VPD training step. An implementation (torch,
JAX, anything) is correct iff it satisfies this document. Ground truth: the on-branch
single-pool torch core in `goodfire-ai/param-decomp` @ `feature/jax` (`param_decomp/`),
file pointers in §9 (non-normative). The runnable torch Metric for the production
*stochastic* recon term (`ChunkwiseSubsetReconLoss`) lives off-branch on the
`feature/fsdp-lm-trainer` n-pool lineage — the parity fixtures pin it; there is no
on-branch Metric counterpart (see §9). Production constants are from the production yaml
(`llama8b_l18_b512_2pool_lr_mid.yaml`, n-pool lineage), extended 1 → N decomposed layers.

**How to read.** Normative content is: the pseudocode (§4), the invariants (§5–§8),
and the tables (§2, §3, §6). Prose between them is orientation only. Notation:

- `sg[x]` — stop-gradient. `x ~ D` — a fresh independent draw from distribution `D`.
- Shapes in brackets: `[B,T,C]`. `B,T` are *global* batch and sequence length. Every
  loss and every gradient in §4 is defined on the GLOBAL batch — §4 is single-machine
  math; how a sharded implementation reproduces it is §8's contract, and is never
  annotated inline.
- Pseudocode names match the implementation (`train.py`) verbatim, so the spec-to-code
  mapping is line-for-line.
- `UPPER_SNAKE` names are **variation points**: pluggable functions with the valid
  instantiation set in §6. ★ marks the production choice.
- Invariants are numbered (`S_`, `N_`, `R_`, `D_`) for citation by audits and tests.
- Bit-exactness with torch is NOT required; identical math / distributions /
  detachment-structure / ordering is.

---

## 1. Glossary — terms that bite

| term | means | NOT to be confused with |
|---|---|---|
| **site** | one decomposed weight matrix (any selected matrix — MLP, q/k/v/o, embedding) | a transformer layer (a layer may own several sites) |
| **component** | one rank-1 slice `V[:,c] ⊗ U[c,:]` of a site's decomposition | a site; a CI-fn unit |
| **recon plan** | the list of `(live_sites, routing sampler)` entries defining the stochastic-recon forwards (§4.3) | a routing draw (those are fresh per step) |
| **live sites** | the sites running their decomposed path in one forward; all others take the frozen `x @ W` path (~9× cheaper, no `(B,T,C)` tensors) | the routed-True positions *within* a live site |
| **chunk** | one plan entry's live-site set (production: sequential triples) | a data chunk / micro-batch |
| **"layerwise"/"chunkwise" recon** | historical names for recon-plan instantiations (per-site / subset-chunk); the loss is ALWAYS on final logits | ⚠ recon evaluated *at* that layer's output. Site-local recon is NOT this method |
| **clean forward** | the full frozen forward (every site on its frozen `x @ W` path) | the decomposed forward with all masks = 1 (≈ equal only in exact arithmetic) |
| **source** | a `[0,1]` value per channel that a mask is built *from* (stochastic or adversarial) | the mask itself |
| **mask** | `ci + (1−ci)·source`, what the forward consumes | the source; the CI value |
| **delta component** | the `(C+1)`-th maskable channel carrying `x @ (W − V@U)` | the faithfulness loss (same residual, different role) |
| **CI** (causal importance) | the CI fn's per-component prediction in `[0,1]`; `ci=1` ⇒ mask pinned 1 (protected) | a probability; an attribution score |
| **lower / upper leaky** | the two squashings of the SAME CI logits; lower → masks, upper → imp-min | two CI functions (there is one) |
| **faithfulness warmup** | pre-loop phase: V/U trained on `L_faith` alone | the other two "warmups" → |
| **PPGD warmup (`n_warmup`)** | supplemental source-ascent iterations inside each step | LR-schedule warmup; faithfulness warmup |
| **LR warmup (`warmup_pct`)** | linear ramp 0 → start of a schedule's value | the above two |
| **persistent** (PGD) | sources + their optimizer moments survive across training steps | re-initialized-per-step PGD (the eval-only `PGDReconLoss`) |

---

## 2. Production constants

| | |
|---|---|
| target | Llama-3.1-8B, frozen, bf16 storage |
| decomposed | `layers[18..18].mlp.{gate,up,down}_proj` (bench: `20..31`), right-mult `W: [d_in, d_out]` |
| C | 24576 (bench: 8192); the delta component is always built → source channel dim `C+1` |
| data | fineweb, seq `T=2048`, global batch `B=512`, fresh batch per step |
| coeffs | faith `1e5` · imp `5e-6` · stoch `0.5` · ppgd `0.5` |
| imp-min | `eps 1e-12`, `p: 2.0 → 0.4` linear over `[0, 1]`-frac of training; `frequency` coeff `1e-6` (= imp `5e-6` · old beta `0.2`), `reference_token_count = B·T = 1048576` |
| stoch | plan = sequential 3-site chunks × `uniform_k_routing(chunk, n_draws=1)` |
| PPGD | scope `sc`, `n_warmup 2`, clamp-parameterization, Adam(β₁ .5, β₂ .99, ε 1e-8), lr const `0.01` w/ 2.5% LR-warmup |
| components opt | AdamW(.9, .999, ε 1e-8, wd 0), lr `1.5e-4` cosine → `0.1×`, **grad-clip 0.01** |
| CI opt | AdamW(.9, .999, ε 1e-8, wd 0), lr `5e-5` cosine → `0.1×`, no clip |
| faith warmup | 400 steps, AdamW lr `1e-3`, wd 0 |
| CI fn | chunkwise transformer (§4.6): per-chunk independent transformer, `d_model 4096`, 4 blocks, 64 heads, mlp `[16384]`, RoPE base 10000, bidirectional; leaky-hard sigmoid (hardwired) |
| steps | 100k; checkpoint cadence per config |

## 3. State

| symbol | shape / type | trains via | persists in ckpt |
|---|---|---|---|
| `W_s` | `[d_in, d_out]` per site | frozen | no (rebuilt from HF) |
| `components = {V_s [d_in,C], U_s [C,d_out]}` | fp32 master | AdamW (components opt) | yes (+ moments) |
| `ci_fn` (params) | fp32 master | AdamW (CI opt) | yes (+ moments) |
| `sources[s]` | `[scope-dims, C+1]` per site (§6 SCOPE; ★ `[1, T, C+1]`) | SRC_STEP ascent | yes (+ SRC_STEP moments) |
| `step`, schedules `pnorm(step)`, `lr(step)` | scalar | — | yes |

A **site** is any weight matrix selected for decomposition (torch decomposes any
`nn.Linear`/`Embedding`/`Conv1D`); sites may be heterogeneous in shape AND in `C`, and
a fixed, documented site order is part of the configuration. The current target
implementation decomposes any per-layer Llama matrices — sites named
`layers.{i}.self_attn.{q,k,v,o}_proj` / `layers.{i}.mlp.{gate,up,down}_proj`, each with
its own `C` (production: the MLP family of one layer at a single `C`). q/k/v sites are
decomposed before RoPE/SDPA (the masked site output feeds the attention math); o after.

---

## 4. Normative pseudocode

### 4.1 Forward semantics

```
def site_out(x[.., d_in], s, mask[.., C]|ONES, delta_mask[..]|ONES, route[..]|ALL) -> [.., d_out]:
    Δ_s    = W_s − V_s @ U_s
    y_dec  = ((x @ V_s) * mask) @ U_s  +  (x @ Δ_s) * delta_mask
    return where(route, y_dec, x @ W_s)                  # route=ALL ⇒ y_dec everywhere

def masked_forward(batch, live_sites, masks, delta_masks, routes) -> logits[B,T,vocab]:
    full forward (embed batch, all blocks, final norm, LM head);
    each site s ∈ live_sites computes site_out(x, s, masks[s], delta_masks[s], routes[s]);
    each site s ∉ live_sites computes x @ W_s            # frozen path, NOT y_dec(mask=1)   (S2)

def clean_output(batch)   = masked_forward(batch, live_sites=∅)                             (S3)
def read_activations(batch, wanted: tuple[str,...]) -> {tap_key: [B,T,d_tap]}               (S4)
    # general clean-path activation accessor keyed by OPAQUE tap keys. `wanted` is the
    # CI fn's static `input_names`; the target is the sole key→activation interpreter.
    # LM: residual-stream taps `resid.{layer}` (the residual entering block `layer`), and/or
    #     per-site matrix-input keys (site names) for harvest.
    # positionless toys: the per-site matrix inputs, keyed by site name.
```

### 4.2 CI

```
def ci(ci_fn, taps) -> CI{logits, lower, upper}:         # each {site: [B,T,C]}, sites PARTITION the model
    logits = CI_ARCH(ci_fn, taps)                        # architecture pinned in §4.6 (★ chunkwise)
    return CI.from_logits(logits)                        # the two squashings, centralized (S5)

CI.from_logits(logits): lower = lower_leaky_hard(logits); upper = upper_leaky_hard(logits)
    # `logits` is a KEPT view (histograms / heatmaps plot the pre-squash value).
    # `ci_lower` ≡ CI.lower feeds every mask; `ci_upper` ≡ CI.upper feeds imp-min only.

lower_leaky_hard: fwd clamp(x,0,1); CUSTOM bwd (nested where, boundaries are <=):
                  pass g on 0<x≤1; at x≤0 pass α·g ONLY where g<0, else 0; at x>1 zero.
                  Tie-break: x=0 resolves to the LOWER (x≤0) branch.  α=0.01               (S6)
upper_leaky_hard: fwd x>1 ? 1+α(x−1) : clamp(x,0,1); ordinary autodiff of that expr.       (S6)
```

### 4.3 Masks and losses

```
def make_masks(ci_lower_s[B,T,C], source_s[B,T,C+1]) -> (mask, delta_mask):
    mask       = ci_lower_s + (1 − ci_lower_s) * source_s[..., :C]
    delta_mask = source_s[..., C]              # delta channel raw: NO ci interpolation     (S1)

def kl_per_position(masked_output, clean_output) =
    Σ_{b,t} KL(softmax(clean_output[b,t]) ‖ softmax(masked_output[b,t])) / (B·T)    # fp32 (N3)

def faithfulness_loss(components) = ( Σ_s ‖W_s − V_s@U_s‖_F² ) / ( Σ_s numel(W_s) )        (S17)

def imp_min_terms(ci_upper, pnorm, reference_token_count):   # per-site grouping             (S7)
    for s: per_component_sums[c] = Σ_{b,t} (ci_upper_s[b,t,c] + eps) ** pnorm           (S8,S9)
           f[s,c] = per_component_sums[c] / (B·T)            # per-token firing frequency
    lp   = Σ_s Σ_c f[s,c]                                    # bare mean term (imp coeff)
    freq = Σ_s Σ_c f[s,c] · log2(1 + a' · f[s,c])           # a' = reference_token_count (freq coeff)
    return lp, freq          # total imp contribution = imp_coeff·lp + freq_coeff·freq   (S8')
    # freq is omitted (0) when no `frequency` is configured. Setting a' = B·T recovers the
    # old rolled `Σ_c f·(1 + beta·log2(1 + B·T·f))` with freq_coeff = imp_coeff·beta.

# RECON_PLAN: a static list of entries (live_sites, SAMPLE_ROUTING); each entry's sampler
# returns a statically-sized FAMILY of routing draws, each draw = one forward (§6).
def stochastic_recon_loss(components, ci_lower, residual, clean_output):
    total, n_forwards = 0, 0
    for (live_sites, SAMPLE_ROUTING) in RECON_PLAN:
        for routes in SAMPLE_ROUTING(key, [B,T]):        # fresh per step                 (R1,S11)
            masks, delta_masks = make_masks(ci_lower_s, source_s ~ U[0,1]^[B,T,C+1])  ∀ s ∈ live_sites
            total += kl_per_position(masked_forward(residual, live_sites, masks, delta_masks, routes),
                                     clean_output)
            n_forwards += 1
    return total / n_forwards                                                               (S10)

def adversarial_recon_loss(components, ci_lower, sources, residual, clean_output):     # all sites (S12)
    masks, delta_masks = make_masks(ci_lower_s, expand(SCOPE, sources[s]))    ∀ s ∈ sites
    return kl_per_position(masked_forward(residual, ALL_SITES, masks, delta_masks, ALL),
                           clean_output)
```

### 4.4 The adversary

```
def sources_update(sources, sources_grad, opt_state):
    sources, opt_state = SRC_STEP(sources, +sources_grad, opt_state)   # ASCENT on L_ppgd
    return PROJ(sources), opt_state                                                         (S15)
```

### 4.5 One training step

```
def train_step(state, batch, step):                  # batch = fresh token batch            (S18)
    cln      = sg[ clean_output(batch) ]                                                     (S3)
    CI = ci(ci_fn, read_activations(batch, ci_fn.input_names))      # ONE conceptual CI eval/step;
    ci_lower, ci_upper = CI.lower, CI.upper                         # recompute allowed (deterministic)
    # -- supplemental adversary ascents (components & CI detached) --
    set SRC_STEP lr = sched_src(step)                # stepped once per TRAINING step       (S13)
    repeat n_warmup:
        sources_grad = ∂/∂sources adversarial_recon_loss(sg[components], sg[ci_lower],
                                                  EFFECTIVE(sources), residual, cln)
        sources, sources_opt = sources_update(sources, sources_grad, sources_opt)

    # -- main losses: live components & ci_fn; the ppgd term's sources detached --
    L = 1e5·faithfulness_loss(components) + 5e-6·importance_minimality_loss(ci_upper, pnorm(step))
        + 0.5·stochastic_recon_loss(components, ci_lower, residual, cln)
        + 0.5·adversarial_recon_loss(components, ci_lower, sg[EFFECTIVE(sources)], residual, cln)

    sources_grad     = ∂/∂sources of that same ppgd term   # PRE-update components, live
                                                           # ci_lower — the SAME graph as
                                                           # the main backward             (S14)
    components_grad  = ∂L/∂components
    ci_fn_grad       = ∂L/∂ci_fn

    sources, sources_opt = sources_update(sources, sources_grad, sources_opt)  # (n_warmup+1)-th (S13)
    components = adamw(components, clip_global_norm(components_grad, 0.01), lr(step))       (S19)
    ci_fn      = adamw(ci_fn, ci_fn_grad, lr_ci(step))                                      (S20)
    return state'

before the loop:                                                                            (S21)
    repeat 400: components = adamw_warm(components, ∂faithfulness_loss/∂components, lr=1e-3)
```

### 4.6 CI architecture (pinned: chunkwise transformer)

**AMENDED 2026-06-22** (Oli-approved): the abstract contract is UNCHANGED — `CI_fn:
read_activations → CI` — but the pinned REALIZATION changes from one global transformer
over every site's input concatenated to a **chunkwise transformer**: the sites partition
into chunks, and each chunk runs its OWN independent transformer. (The CI fn is a protocol
`dict[InputTap,Array] → CI`; the input keyspace — clean-path tap keys — is independent of
the output keyspace — the decomposition sites — and the output sites must PARTITION the
model's sites.)

```
in:   {tap_key: x [B,T,d_tap]}  (clean-path taps = ci_fn.input_names; opaque keys, §4.1 S4)
sites partition into CHUNKS; each chunk c declares input_taps ⊆ tap_keys and output_sites.
per chunk c (an INDEPENDENT pre-norm bidirectional-RoPE transformer):
1.    h_c = concat_{k ∈ input_taps_c}( rms_norm(x_k) )   # weightless per tap;
                                                          # eps = finfo(fp32).eps ~1.19e-7 (S4)
                                                          # → [B,T, input_dim]
2.    h   = h_c @ W_in_c + b_in_c                         # → [B,T,d_model]; NO nonlinearity here
3.    × n_blocks (pre-norm):
        h += attn(rms_norm_weightless(h))      # bidirectional MHA, rotate-half RoPE
                                               # base 10000; q/k/v/out bias-FREE
                                               # RoPE inv_freq is a stop-gradient buffer (S27)
        h += mlp(rms_norm_weightless(h))       # Linear(d→16384)+b → GELU(erf, NOT tanh) → Linear(→d)+b (S6)
4.    c_chunk_c = h @ W_out_c + b_out_c        # → [B,T, Σ_{s ∈ output_sites_c} C_s]
                                               # split back per site, in the chunk's site order
the per-chunk transformers are STACKED along a leading n_chunks axis and run under ONE
eqx.filter_vmap → requires HOMOGENEOUS chunks: equal total input width (`input_dim`) and
equal `c_chunk` (Σ C over the chunk's output sites). Asserted at init.
init: biases zero; weights fan-in scaled (Kaiming: relu-gain √2 on in_proj / MLP-in,
      linear gain 1 on out / MLP-out; PyTorch-default U(±1/√fan_in) on the attn projections)
```

`input_dim` is a generic linear-input width (a plain in_proj fan-in), NOT a residual-dim
or transformer concept in core — the lab computes it from the taps it authored (their
widths summed) and core stays agnostic to what the taps mean. Layerwise (one site per
chunk) and global (all sites in one chunk) are degenerate chunkings of this same form.
The positionless toys (`expects_axes=()`) are the MLP siblings (`LayerwiseMLPCIFn` /
`GlobalMLPCIFn`), not transformers.

CI-fn numerics still unify with the torch oracle's per-block primitives (#624/#625/#730,
resolved "unify — match torch"): GELU is exact-erf (`jax.nn.gelu(..., approximate=False)`,
matching torch `nn.GELU()`), the weightless RMSNorm uses `eps = finfo(fp32).eps`
(`CI_FN_RMS_EPS`, matching torch `F.rms_norm`'s default `eps=None → finfo(x.dtype).eps`;
RMS upcasts to fp32, so fp32 finfo governs), and RoPE is base-10000 bidirectional with
`inv_freq` stop-gradient'd (S27). **Torch-oracle scope (AMENDED 2026-06-22):** the
CI-fn ARCHITECTURE is now JAX-native and INTENTIONALLY no longer bit-faithful to the
torch oracle's single global-concat CI fn — the chunkwise partition has no torch
counterpart, so the prior "torch→JAX CI-fn weight transfer is bit-faithful on the clean
path" clause is RETIRED. Everything else remains torch-oracle-grounded — recon,
faithfulness, sources/PPGD, the squashings (S5/S6), the imp-min reduction, the grad clip
(S19), the schedules (S20) — and the `tests/equivalence/` suite still pins them. Refs:
`ci_fn.py` (`ChunkwiseTransformerCIFn`, `CIBlock`, GELU line, `CI_FN_RMS_EPS`, `inv_freq`).

---

## 5. Semantic invariants

| id | invariant |
|---|---|
| S1 | `mask = ci + (1−ci)·source` per component channel; the delta channel is the raw source value, never ci-interpolated; CI has no delta output. |
| S2 | A site not live in a forward runs the frozen `x @ W_s` path — zero V/U gradient, zero decomposition rounding, and none of the live path's ~9× compute or `(B,T,C)` activations. |
| S3 | The recon target `clean_output` is the frozen-path forward (the full model: embed the token batch, all blocks, final norm, LM head), stop-gradient. Never the `mask=1, delta=1` decomposed identity (differs in bf16 and pollutes the graph). A subset decomposition simply leaves non-decomposed blocks on the frozen `x @ W` path. **AMENDED 2026-06-24** (Oli-approved): removed residual-start — the model takes the token batch and embeds internally, so `clean_output` IS the whole-model frozen forward (was: the suffix-only forward over a separately-harvested residual, argued equal to the whole-model forward under sg of a frozen prefix). See REMOVE_RESIDUAL_START_DESIGN.md. |
| S4 | **AMENDED 2026-06-22** (Oli-approved): CI inputs come from `read_activations(batch, ci_fn.input_names)` — a GENERAL clean-path activation accessor keyed by OPAQUE tap keys (was the fixed `site_inputs` seam). `input_names` is the CI fn's static declaration; the target is the SOLE key→activation interpreter, producing exactly the requested taps off the frozen path of the same batch. For the LM these are residual-stream taps `resid.{layer}` (the residual entering block `layer`) and/or per-site matrix-input keys (site names) for harvest; for positionless toys they are the per-site matrix inputs. Core never parses a key. (Was: "CI inputs are the clean site inputs from the frozen path of the same batch.") |
| S5 | **AMENDED 2026-06-22** (Oli-approved): the CI fn returns a `CI{logits, lower, upper}` bundle (was `(ci_lower, ci_upper)`). `lower` and `upper` are two squashings of the SAME `logits`, centralized in `CI.from_logits` (no impl re-triplicates them); `logits` is KEPT as a consumed view (histograms / heatmaps plot the pre-squash value). `ci_lower ≡ CI.lower` feeds every mask; `ci_upper ≡ CI.upper` feeds imp-min only; no other crossing. The squashings themselves (S6) are UNCHANGED — only the bundling and the `logits` view changed. The CI fn is a protocol `dict[InputTap,Array] → CI` whose output sites PARTITION the model's sites (input keyspace independent of output keyspace). |
| S6 | The squashings' forward/backward are exactly §4.2 — including `lower_leaky_hard`'s grad-sign-gated lower leak (a custom VJP, not autodiff of the forward). The backward is a nested `where` with `<=` boundaries: on `0 < x <= 1` pass `g`; at `x <= 0` pass `α·g` ONLY where `g < 0`, else `0`; at `x > 1` pass `0`; `α=0.01`. The boundary tie at `x=0` resolves to the LOWER (`x <= 0`) branch — this `<=` placement is exactly what makes torch (`ci_sigmoids.py`) and JAX (`ci_fn.py`, `_lhs_b`) bit-identical and is load-bearing, not incidental. Grad-checked by `tests/test_lower_leaky_hard_grad.py` (`g<0` vs `g>0` at `x<0`, `x∈(0,1]`, `x>1`; #789). |
| S7 | Imp-min groups per site: the frequency penalty's `log2(1 + a'·f_c)` consumes one site's per-component frequency. Merging sites/layers into one group is incorrect (convexity). |
| S8 | The per-component frequencies `f_c = (Σ_{b,t} ψ(c)) / B·T` are over the **global batch**, formed before the `log2`. (Per-shard `f_c` combined after the log are incorrect — Jensen; see D2.) The two imp-min terms (`lp = Σ_c f_c`, `freq = Σ_c f_c·log2(1 + a'·f_c)`) share these per-component sums in one pass. |
| S8' | Imp-min contributes `imp_coeff·lp + freq_coeff·freq` with INDEPENDENT coefficients; the frequency normalizer `a' = reference_token_count` is explicit (batch-invariant at fixed firing rate), not the implicit `B·T`. `freq` is absent when no `frequency` is configured. `a' = B·T` recovers the old rolled `imp_coeff·Σ_c f·(1 + beta·log2(1 + B·T·f))` with `freq_coeff = imp_coeff·beta`. |
| S9 | `pnorm(step)` anneals linearly `2.0 → 0.4` over the configured frac window; `eps` sits inside the power. **JAX narrowing:** annealing is REQUIRED — `annealed_pnorm` asserts `cfg.p_anneal_final_p is not None` (`losses.py:53`) and `train.py:122` asserts it too. Torch supports a constant-p config (`importance_minimality.py:16-37` returns `initial_p` when no annealing window). Constant-p in JAX is expressed by setting `p_anneal_final_p == pnorm` (a flat schedule); any other torch constant-p config is REFUSED (fail-fast assert), never silently approximated. |
| S10′ | The recon objective is a static tuple of coefficiented loss TERMS (one per configured recon loss metric, in config order). Each term is a static plan of `(live_sites, SAMPLE_ROUTING, MASK_SOURCE)` entries; the term's loss = mean over ALL its forwards (every draw of every entry) of `kl_per_position`; the total adds `coeff · term` per term. Plan structures (live-sets, sampler identities, family sizes, strategy kinds) are fixed across steps. The §4 pseudocode shows the production two-term instantiation (`stochastic_recon_loss` + `adversarial_recon_loss`). Recon KL direction is pinned by S25; the mean-over-forwards ≡ accumulator identity by S26. |
| S11 | `uniform_k_routing`, per position: `k ~ U{1..|live_sites|}` then a uniform `k`-subset of the live sites routes True; non-live sites are not live at all. Routing draws are fresh per step, sampled inside the step. |
| S12′ | An adversarial term's loss forward consumes its sources as LEAVES (no ascent-graph history); gradient flows to components and (through `ci_lower`) to the CI fn — and, for persistent sources, to the leaves themselves (S14′). The PRODUCTION adversarial term masks ALL sites and routes everywhere; subset-routed adversarial terms route per their plan. |
| S13′ | Per persistent term: source updates per training step = `n_warmup + 1`, all through THAT term's persistent SRC_STEP optimizer state; its source LR schedule advances once per training step. **`warmup_pct==0` edge (accepted seam):** at `warmup_pct==0` torch short-circuits to full LR at step 0 (`warmup_steps=0`), while JAX clamps `warmup_steps = max(floor(...), 1)` → source LR `=0` at step 0 (`losses.py:61`, `train.py:265`). A one-step divergence, only when `warmup_pct==0`; production uses 2.5% warmup and is unaffected. Accepted, not matched. |
| S14′ | Each persistent term's final ascent gradient comes from the SAME graph as the main backward (pre-update components, live `ci_lower`), unscaled by THAT term's coeff. It is applied after backward; it must not use post-update params. |
| S15 | Every source update ends with `PROJ` (★ clamp to `[0,1]`). Init: ★ `sources ~ U[0,1]` i.i.d. |
| S16 | Shared-scope sources are identical on every data-parallel replica at every step (identical init, identical updates from the global-batch gradient). `bsc` sources shard with the batch instead. (Implementation mapping in §8/§9.) |
| S17 | `faithfulness_loss` is the global mean of squared delta entries over all sites' parameters (Σ‖Δ‖² / Σ numel), recomputed from live V/U each step. |
| S18 | Each training step consumes a fresh token batch (a pure function of `(seed, step)` for O(1) resume); the model embeds it internally. **AMENDED 2026-06-24** (Oli-approved): removed the prefix harvest (residual-start) — there is no separate prefix forward; `clean_output` / `read_activations` / `masked_output` take the token batch directly. |
| S19 | Components gradients are global-norm-clipped at `0.01`, before the optimizer step. CI fn is unclipped (production). The clip coefficient uses torch's eps convention: `clip_coef = min(1, max_norm / (total_norm + 1e-6))` (`torch.nn.utils.clip_grad_norm_`). This `+1e-6` is canonical and matched JAX-side (#643); plain `optax.clip_by_global_norm` divides by `max(total_norm, max_norm)` with NO eps, which at `clip=0.01` (clip fires almost every step) gives a ~1e-4 relative component-grad difference each step — a real per-step deviation the JAX clip must avoid by reproducing the `+1e-6`. |
| S20 | Both main optimizers are AdamW, `wd=0`, betas `(0.9, 0.999)`, eps `1e-8`; LR cosine to `0.1×` start, no warmup, stepped per training step. Cosine convention is canonical-torch: `progress = (step − warmup) / (decay_steps − 1)`, so LR reaches `0.1×` at `step = steps − 1` (`schedule.py`). This is matched JAX-side (#642); plain `optax.cosine_decay_schedule(lr, steps, alpha=0.1)` divides by `steps` and reaches `0.1×` one update LATER (at `count = steps`), a genuine formula divergence of ~O(1/steps) ≈ 2.5e-6/step at 400k that the JAX schedule must avoid by using the `decay_steps − 1` denominator. |
| S21 | Faithfulness warmup (400 × AdamW lr `1e-3` on `faithfulness_loss` alone) precedes step 0; its optimizer is discarded. |
| S22 | Checkpoints round-trip ALL trajectory state of §3 — including every persistent term's sources + SRC_STEP moments + step/schedule counters — such that a resumed run continues the same trajectory (modulo RNG streams and kernel nondeterminism, cf. D4). **SIGTERM lifecycle (decision):** a SLURM-delivered SIGTERM sets a flag that is serviced at the main train-step boundary (synchronous save of the completed step, then exit for requeue), inside the faith-warmup loop (clean exit WITHOUT a save — no valid checkpoint exists pre-step-0, and resume skips warmup whenever any checkpoint is present, so a partial step-0 save would resume as if fully warmed; the requeue redoes warmup), and inside the in-loop eval pass (abandon the partial pass unlogged, fall through to the step-boundary save). The **first-jit-compile window remains unserviced** — a SIGTERM there is honored only once compilation finishes and control reaches the next servicing point. Periodic `save_every` is the BACKSTOP guarantee for every window; the SIGTERM→save path is the low-latency fast path, not the sole guarantee (lore `jsp_sigterm_save_never_fired`: 6/6 historical preemptions fell back to periodic ckpts; warmup/eval servicing closes the two largest unserviced windows). |
| S23 | A persistent source bundle feeds exactly ONE loss term. (The fused-backward S14′ unscaling divides that term's coeff out of the source gradient; a bundle shared across terms would make the division wrong.) |
| S24 | A persistent term's WARMUP ascents forward all sites, routed everywhere — regardless of the term's loss plan (torch parity: `persistent_pgd_state.warmup` hardcodes route-all). A fresh-PGD entry draws its routing ONCE per step, shared by all its ascents and its main loss forward (torch parity: `pgd_masked_recon_loss_update`). |
| S25 | Recon KL direction is `KL(softmax(clean_output) ‖ softmax(masked_output))` — `P = clean`, `Q = masked`. Equivalently `Σ p_clean · (log p_clean − log p_masked)`. Torch `recon_loss_kl` realizes this as `F.kl_div(log_softmax(pred=masked), softmax(target=clean), reduction='sum')` (`batch_and_loss_fns.py::recon_loss_kl`); reversing the arguments is a silent, plausible bug, so the direction is semantic, not incidental. |
| S26 | Normalization identity: `(Σ_forwards sum_kl) / (Σ_forwards n_positions) == mean_forwards(sum_kl / n_positions)`, valid **iff** every forward shares the same `(B,T)` (so `n_positions` is constant across forwards). This precondition is what makes JAX's mean-over-forwards (S10′) equal torch's `(Σ sum_kl)/(Σ n_positions)` accumulator; uniform `(B,T)` across all forwards in a term is therefore required. (Stated in LOSS_PARITY_DESIGN §4e; cross-ref S10′.) |
| S27 | The CI transformer's RoPE `inv_freq` (§4.6) is a non-trained buffer and MUST be stop-gradient'd in the CI fn. In JAX it is a pytree leaf and would otherwise be optax-updated, silently drifting the rotary frequencies. In the chunkwise CI fn `inv_freq` is shared across chunks (a single buffer, NOT mapped over `n_chunks`) and `stop_gradient`'d inside `ChunkwiseTransformerCIFn.__call__` before the `filter_vmap`. Ref `ci_fn.py` (`ChunkwiseTransformerCIFn.__call__`, `jax.lax.stop_gradient(self.inv_freq)`). |
| S28 | Eval runs in-loop in TWO tiers (§10): a FAST scalar tier on cadence `eval.every`, and a SLOW/plot tier on cadence `eval.slow_every` (a multiple of `every`, so it lands on a fast-eval step and REUSES that step's in-memory eval batches — no second forward, no second data read). Slow/plot eval is IN-LOOP ONLY — there is no offline/retrospective CLI (`pd-slow-eval` and `run_offline_slow_eval` are removed; `slow_eval.py` is a pure library of the accumulate/render/metric fns the in-loop tier calls). The slow tier renders the plot metrics (`CIHistograms`, `ComponentActivationDensity`, `CIMeanPerComponent` → the shared `render_slow_eval_figures` figure set), the config-gated CI-heatmap / `PermutedCIPlots` figures + the `IdentityCIError` scalars (both driven by the cheap `(T, C)` per-site position-CI matrix from `accumulate_position_ci`, gated on the config naming them), the two hidden-acts recon scalars (`compute_hidden_acts_metrics`), and — when the config names it — the `UVPlots` figure. `UVPlots` is a config-gated figure metric usable for ANY decomposition (the torch `Metric` pattern: it returns a wandb figure): cheap for the toys (TMS/ResidMLP — small, replicated, on-host V/U, no gather; rendered off the toy probe CI by `render_uv_figure`), and a NAIVE full host gather of the C-sharded V/U for the LM in-loop tier (`run.py` gathers V/U only when `want_uv_plots` and passes `components` to `render_permutation_figures`). The LM gather OOMs / breaks at production C BY DESIGN (per Oli) — no special handling, the gather (collective, on the eval pass) is the cost; zero cost when `UVPlots` is not named. The slow tier is forward-only (one target forward + the CI-fn forward + sigmoid + C-sized reductions; no backward / masking grid / PPGD / optimizer state), so its peak HBM ≤ the train step's own high-water mark (the UVPlots gather aside) — where training fits, in-loop slow eval fits. **AMENDED 2026-06-18** (Oli-approved): reverses the migration-era "slow tier is offline-only, no in-loop analog" decision (and supersedes the retired torch-era push-triggered `pd-offline-eval` sidecar); the slow tier runs in-loop next to the fast pass. **RE-AMENDED 2026-06-19** (Oli-approved): drops the "`pd-slow-eval` CLI retained" clause — slow eval is in-loop only — and makes `UVPlots` an in-loop config-gated figure metric (cheap for toys, naive/breaks-at-scale for the LM by design) rather than offline-only. The lineage of why the out-of-process sidecar is a known dead end is recorded in lore `2026-06-18--in-loop-slow-eval-proposal`. |
| S28a | **Multi-host split (in-loop slow tier).** The slow tier separates a COLLECTIVE part — run in lockstep on ALL ranks — from a pure-HOST part — run on rank 0 only, OFF the train loop. Collective: the jitted forward + the device→host pull (`accumulate_site_reductions` / `compute_hidden_acts_metrics` / `accumulate_position_ci` materialize C-sharded reductions to numpy; `np.asarray` triggers the all-gather every rank must join — `accumulate_position_ci` runs only when the config names a CI-heatmap / permutation / identity-error metric, so it adds zero cost otherwise). When the config names `UVPlots`, the C-sharded V/U is ALSO gathered to host here (`np.asarray` over `state.components.vu`) — a NAIVE collective gather that OOMs / breaks at production C by design (S28); gated on `want_uv_plots`, so zero cost otherwise. Host: matplotlib render + `wandb.log` of the figure PNGs (the base set + the config-driven CI heatmaps + `UVPlots` when the V/U was gathered), handed to a `SlowEvalRenderer` BACKGROUND THREAD on rank 0 (`run.py`) — the main loop proceeds immediately on every rank, near-zero cross-rank divergence. The thread touches ZERO jax/device state; at most one render is in flight (a new submit `join()`s the prior first); an `atexit` join flushes the last render before process exit (the trainer never calls `wandb.finish`). The hidden-acts SCALARS and the `IdentityCIError` SCALARS ride the live `_step` axis (the fast eval record, computed on the collective path and logged synchronously by the sink at the eval step — cheap, and scalars must stay `_step`-monotonic, so they are NOT deferred to the background thread); the FIGURES log on `_step` at `step=now_step` from the background thread — a render that lands after the head advances past `now_step` is dropped by wandb's monotonic-`_step` rule (a benign one-figure-set miss, warned not raised), which slow eval's forward-only seconds against a coarse `slow_every` is not expected to hit. The toys run the same `UVPlots` figure synchronously on CPU (single-process, tiny V/U): `toy_uv_eval.log_uv_figure` renders + logs it from the toy `eval_fn` on the live `_step` axis when the config names it. |
| S29 | JAX `EvalConfig` carries `batch_size`, `every`, `n_steps`, `slow_every`, `slow_on_first_step`, `slow_n_batches_accum`, `density_heatmap_n_bins` (+ `rounding_threshold`, `ci_alive_threshold`). `slow_every` / `slow_on_first_step` are read from the canonical schema and drive the in-loop slow tier (S28); `slow_n_batches_accum` is the `CIHistograms.n_batches_accum` histogram-sample cap (None = uncapped); `density_heatmap_n_bins` is the `CIHistograms.density_heatmap_n_bins` opt-in for the per-token CI density heatmap (None = off — the add-on shares the slow-eval forward's `lower`, adds only an on-device per-component bincount `(C, n_bins + 1)` — column 0 = underflow (CI < 1e-9, incl. exact-0), columns 1..n_bins = log-spaced `[1e-9, 1]` bands — accumulated over EVERY batch, and emits `figures/ci_density_heatmap` on a log y-axis; no raw-value host transfer). `eval` is atomic-optional (`EvalConfig | None`): `None` disables in-loop eval (both tiers). **AMENDED 2026-06-18**: re-adds `slow_every` / `slow_on_first_step`, deliberately dropped in the original S29 when the slow tier was offline-only. **AMENDED 2026-07-01**: adds `density_heatmap_n_bins` (opt-in per-token CI density heatmap sharing the CIHistograms forward). |
| S30 | `cfg.cadence.log_every` divides `cfg.eval.every` (`eval.every % log_every == 0`, asserted at `run.py:255`) so every eval step is also a train-log step. |
| S31 | The two hidden-acts recon metrics (`CIHiddenActsReconLoss`, `StochasticHiddenActsReconLoss`) are STANDALONE OFFLINE EVAL metrics in JAX (`hidden_acts_eval.py`, wired into `jsp-slow-eval`) — **NOT** recon-grid training terms. `build_loss_terms` still refuses them as training losses (the parameterized recon loss stays KL-on-final-logits only, §2.3–2.5); their objective is per-ELEMENT MSE on each decomposed site's OUTPUT activations, which as a *training* loss is exactly the site-local recon the trainer treats as a conceptual no-no (LOSS_PARITY_DESIGN §4c). The port adds a fifth per-target seam `masked_site_outputs(vu, batch, masks, delta_masks, routes, live, has_delta) -> dict[site, (B,T,d_out)]` (`lm.py`), factored out of `masked_output` (the shared masked forward with a per-site `collect` — the masked per-site output is an intermediate of that forward, so no logic is duplicated). The clean (target) per-site output is the frozen `x @ W`, obtained from the same seam by routing FALSE everywhere (`_site_out`'s frozen branch). Per site, `MSE(masked_site_output, clean_site_output)` with `reduction="sum"` accumulated host-side as `(Σ sum_mse, Σ n_elements)` (token-weighted, exact under micro-batching), divided once at the end; log keys mirror torch exactly (`<ClassName>/<site>` + a combined `<ClassName>` = Σmse/Σn over all sites). `CIHiddenActsReconLoss` is the deterministic `lower_leaky` CI mask, no delta, one forward (tight torch parity); `StochasticHiddenActsReconLoss` draws `n_mask_samples` stochastic CI masks (`mask = ci + (1−ci)·s`) WITH weight deltas (the delta component is always built in JAX runs) — its draws are NOT seed-aligned to torch, so exact bitwise parity is impossible there (expected). Masked + clean run in COMPUTE_DT (bf16, matching the trained model, mirroring `load_run.py`); the MSE reduction is fp32. **AMENDED 2026-06-16** (Oli-approved): superseded the prior "keep-on-bridge / seam refused" decision — these now have a native JAX eval path; the `pd-offline-eval` torch bridge remains available for cross-framework parity checks. |
| S33 | Fine-tune init (`ExperimentConfig.resume_provenance`, LM-only): a fresh run whose own `ckpts/` is EMPTY and whose `resume_provenance is not None` initializes from a PARENT checkpoint — it loads the parent's `ckpts/<parent_step>` onto the fresh reference `TrainState` and keeps ONLY the trained `components` (V/U) + `ci_fn`; the optimizer states, persistent sources, and `step` are the FRESH reference's (`step = 0`). Rationale: a fine-tune runs a NEW LR/p-anneal schedule computed over the new `cfg.steps` from 0, so carrying stale Adam momentum / a stale adversary would mis-scale the restart; faith warmup is also skipped (the parent's V/U is already faithful). The run records lineage via `resume_provenance` in `config.yaml` + `wandb.config`. The parent's decomposition STRUCTURE (sites names + C, ci-fn arch) must match the new config's — asserted from the parent's pinned `config.yaml` (`run.py::assert_finetune_structural_compat`) before the orbax restore; only LR / coeffs / eps / seq / batch / steps may change. This is distinct from same-config requeue-resume (S22): on a subsequent requeue the run's own `ckpts/` is non-empty, so `restore_latest` from its own dir wins and provenance is ignored. |

## 6. Variation points

| point | valid instantiations | production |
|---|---|---|
| `RECON_TERMS` | any static tuple of coefficiented terms, each a static plan of `(live_sites, SAMPLE_ROUTING, MASK_SOURCE)` entries: `subset_chunk_plan` · `per_site_plan` (the torch "layerwise" shape) · `all_sites_plan` · custom subset families (pairs, covers, …); built from the shared torch loss configs by `build_loss_terms` | ★ stochastic subset term + persistent-PGD term |
| `MASK_SOURCE` | `stochastic` (fresh U[0,1]/Bernoulli per draw) · `constant(v)` (`v=0` CI-masked, `v=1` unmasked; no delta path) · `fresh_pgd(init, n_steps, step_size, scope)` · `persistent(state_key)` | ★ stochastic + persistent |
| `SAMPLE_ROUTING` | `(key, [B,T]) → tuple of routing draws`, statically sized; draws may be jointly sampled (independent repeats, antithetic/complementary subsets, per-step covers) | ★ uniform_k_routing(·, 1) |
| `SRC_STEP` | `adam(β₁,β₂,ε)` with bias correction; `sign` (`sources += lr·sign(grad)`) | ★ adam(.5, .99, 1e-8) |
| `PROJ` / `EFFECTIVE` | **clamp**: PROJ = clamp[0,1], EFFECTIVE = identity, init U[0,1] · **sigmoid**: PROJ = identity (unbounded raw), EFFECTIVE = sigmoid, init N(0,1) | ★ clamp |
| `SCOPE` | persistent: `c (1,1)` · `sc (1,T)` · `nsc (n,T), n|B` · `bsc (B,T)` (jax: `sc` + `bsc` today) — fresh-PGD: `c` · `bc` · `bsc` | ★ sc |
| stoch sampling | `continuous U[0,1]` · `binomial {0,1}` | ★ continuous |
| delta component | on (`C+1` channels) · off (no delta path, no delta mask) | ★ on |
| CI squashing | `leaky_hard` pair (§4.2); other registry sigmoids exist in torch but are out of spec scope | ★ leaky_hard |

A variant choice must hold every invariant not explicitly parameterized by it.

## 7. Numerics (N) and randomness (R)

| id | rule |
|---|---|
| N1 | Components and CI-fn master params fp32; both AdamW moment sets fp32; SRC_STEP moments fp32. Forward compute may be bf16; the frozen target may be stored bf16. The persistent-source Adam (`SRC_STEP = adam`) places eps AFTER the sqrt, inside the denom: `denom = sqrt(v_hat) + eps` (not `sqrt(v_hat + eps)`) — a classic eps-before-vs-after-sqrt parity trap. Refs: torch `persistent_pgd_state.py:125` (`v_hat.sqrt().add_(eps)`), jax `adversary.py:113`. |
| N2 | Faithfulness deltas `W − V@U` are computed in fp32 outside any autocast; the sum-of-squares is fp32. (The delta on the masked-forward PATH may be bf16-computed — a documented bf16-rounding divergence from torch, which forms it in fp32 and casts at use.) |
| N3 | `kl_per_position` (softmaxes + KL sum) and the imp-min reduction are fp32. Loss scalars and gradient accumulation fp32. **imp-min cast point (accepted seam):** the *reduction* is fp32 on both sides, but the elementwise `(ci + eps)**p` intermediate differs — torch forms it as a bf16 intermediate (autocast leaves the pow in the input dtype) then sums in fp32 (`importance_minimality.py:49,146`; `train_step.py:142,225`), while JAX casts `ci → fp32` BEFORE the power (`losses.py:43,45`). This is a small per-component-sum rounding asymmetry on the imp-min input that the fp32-only equivalence fixture does not exercise; accepted as a seam because the fp32 masters dominate the trajectory. **eval cast point (accepted seam):** `CEandKLLosses` carries the SAME bf16-input asymmetry — torch computes the eval softmax under bf16 autocast (`param_decomp_lab/eval_metrics/ce_and_kl_losses.py:91-176`), JAX casts to fp32 before `log_softmax` (`eval.py:31-40,110-166`). Both seams are bf16-vs-fp32 input cast-point divergences, not direction/reduction divergences; the expected `kl_<v>` delta on a fixed batch is the bf16-rounding floor (rel ≪ 1e-2), accepted rather than bounded by a fixture (cf. E17). |
| R1 | Every stochastic draw (mask sources, routing, source init) is independent across sites, positions, forwards, steps — distributions as stated. |
| R2 | RNG stream order/bits need not match torch. |
| R3 | Draws over a sharded batch are independent across ranks (distinct streams). |

## 8. Data-parallel contract (D)

§4 never mentions sharding because it doesn't have to: every loss and gradient there
is global-batch math. This section is the whole answer to "now shard it":

| id | rule |
|---|---|
| D1 | Per-shard means + averaged grads must compose to §4's global-batch values for faith/stoch/ppgd (uniform shards). For the *shared-scope source* gradient this means: AVG the per-replica source-grads (each is ∂(local-shard mean)/∂sources) — torch's `reduce_source_grads`; under GSPMD it falls out of autodiff of the global-mean loss. Getting this reduction wrong (e.g. SUM over independent per-position sources) was a real historical bug. |
| D2 | Imp-min requires the exact global per-component sums *inside* the `log2` (S8) — the one term where mean-of-shard-results ≠ global result (Jensen). The reduction must also be autograd-aware so gradient reaches each shard's CI values. The eval-path imp-min reuses the one `importance_minimality_terms` impl (there is no separate non-autograd eval reduce as in torch), so the value is device-count invariant by D4; guarded directly at 1 vs N devices by `tests/test_imp_min_global_reduction.py`. |
| D3 | Shared PPGD sources stay replica-identical per S16: identical init (broadcast or identical seeding), updates computed from the D1 gradient, identical optimizer steps. |
| D4 | Validation property: with global batch + seed fixed, the metric trajectory is invariant to device count up to floating-point reassociation (cross-shard reduction order; observed rel ≤ ~1e-5 on the tiny-target harness, `experiments/invariance_check.py`). JAX's counter-based RNG makes even the stochastic draws identical across layouts. |

---

## 9. Non-normative: torch ground-truth pointers & rationale

All pointers below resolve to the on-branch single-pool torch core (`param_decomp/`) at
`feature/jax`. The one exception is the production *stochastic* recon term: its runnable
torch Metric `ChunkwiseSubsetReconLoss` lives off-branch on the `feature/fsdp-lm-trainer`
n-pool lineage (`param_decomp_lab/metrics/chunkwise_subset_recon.py`) — there is no
on-branch Metric counterpart, but the torch↔JAX parity fixtures (`tests/equivalence/`)
pin its math (R-1). The KL primitive it composes (`recon_loss_kl`) and the recon-plan
ancestry are on-branch (rows below).

| spec | torch source |
|---|---|
| §4.1 site/forward, routing | `param_decomp/components.py` (`LinearComponents.forward`), `param_decomp/masks.py` |
| §4.2 squashings | `param_decomp/ci_sigmoids.py` (`LowerLeakyHardSigmoidFunction`, `upper_leaky_hard_sigmoid`) |
| §4.3 faith / imp / KL | `param_decomp/metrics/faithfulness.py` (`faithfulness_loss`) · `param_decomp/metrics/importance_minimality.py` · `param_decomp_lab/batch_and_loss_fns.py::recon_loss_kl` |
| §4.3 recon term (stochastic) | OFF-BRANCH: `param_decomp_lab/metrics/chunkwise_subset_recon.py` (`ChunkwiseSubsetReconLoss`) on `feature/fsdp-lm-trainer`; pinned by `tests/equivalence/` fixtures, no on-branch Metric |
| §4.4–4.5 adversary, ordering | `param_decomp/metrics/persistent_pgd_state.py` (init/warmup/step/scopes; source Adam denom `v_hat.sqrt().add_(eps)` @ `:125` — N1), `param_decomp/metrics/persistent_pgd_recon.py` (`before_backward`/`after_backward`), `param_decomp/metrics/pgd_utils.py`, `param_decomp/train_step.py::run_loss_step` (hook order), `param_decomp/optimize.py` (clip → step) |
| §4.5 warmup, schedules | `param_decomp/faithfulness_warmup.py`, `param_decomp_config/schedule.py::get_scheduled_value` |
| §4.6 CI arch (torch oracle, RETIRED for arch) | `param_decomp/ci_fns.py::GlobalSharedTransformerCiFn`, `param_decomp/ci_nn_blocks.py` — the torch global-concat CI fn, no longer bit-faithful to the pinned JAX arch (§4.6 AMENDED 2026-06-22); JAX arch is native (`ci_fn.py::ChunkwiseTransformerCIFn`) |

The JAX implementation (`jax_single_pool/train.py`) uses these pseudocode names
verbatim: `clean_output`, `read_activations`, `source_masks`, `stochastic_recon_loss`,
`adversarial_recon_loss`, `sources_adam_ascend_project`, `ReconPlan`/`ReconForward`,
`uniform_k_routing`, `subset_chunk_plan`, `per_site_plan`.

Rationale worth keeping: the two squashings give each consumer gradient only in its
permitted direction (masks may push CI up out of saturation; the sparsity penalty may
not push it below 0). The adversary is *persistent* because re-finding the worst-case
ablation from scratch each step under-trains the adversary at any affordable inner-step
count. The `freq` term is the description-length / frequency penalty (`L_freq` in the VPD
paper); normalizing its `log2` by an explicit `a' = reference_token_count` (rather than the
implicit `B·T`) makes its curvature batch-invariant at a fixed firing rate, so batch size
and frequency-penalty strength are independently tunable. Its convexity is why S8 demands
the true global per-component frequency. Fused-linear-KL
and LM-head-bypass are memory/throughput optimizations and must be semantically
invisible (cf. `recon_loss_kl` equivalence).

---

## 10. Eval (two in-loop tiers)

Eval is normative only at the boundary; the metric *values* it reports are not part of
the training-step semantics. The two tiers (S28–S30):

- **FAST tier — in-loop.** Scalar eval metrics run inside the training process on
  cadence `eval.every` (`run.py::_make_lm_eval_fn`). This is the torch `EvalLoop` analog
  restricted to scalars (CE/KL, CI-L0, the fresh-PGD probe, attn-patterns).
- **SLOW / plot tier — in-loop ONLY (S28/S28a).** On cadence `eval.slow_every` (a multiple
  of `every`, so it lands on a fast-eval step and reuses that step's in-memory eval batches)
  the slow tier renders the plot figures and the hidden-acts recon scalars, forward-only,
  next to the fast pass. The collective forward + device→host pull run on all ranks; the
  matplotlib render + `wandb.log` run on a rank-0 background thread (`SlowEvalRenderer`)
  off the loop. The hidden-acts scalars ride the live `_step` axis; the figures log on
  `_step` at the eval step. There is NO offline/retrospective slow-eval CLI — `slow_eval.py`
  is a library only. `UVPlots` is a config-gated figure metric usable for any decomposition:
  cheap for the toys, a naive V/U host gather for the LM in-loop tier that breaks at
  production C by design (S28).
- **Cadence coupling.** `log_every` divides `eval.every` (asserted in `train`), and
  `eval.every` divides `eval.slow_every` (asserted in `train`), so every eval step is a
  train-log step and every slow step is an eval step.
- **Cast-point seam.** `CEandKLLosses` carries the bf16-input cast-point asymmetry recorded
  in N3 (torch softmax under bf16 autocast; JAX fp32 before `log_softmax`), an accepted seam.
