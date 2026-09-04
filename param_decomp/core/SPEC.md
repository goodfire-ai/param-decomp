# Single-pool VPD — semantics spec

Pins the **meaning** of the single-pool VPD training step. An implementation (torch,
JAX, anything) is correct iff it satisfies this document. Ground truth: the on-branch
single-pool torch core in `goodfire-ai/param-decomp` @ `feature/jax` (`param_decomp/`),
file pointers in §9 (non-normative). The runnable torch Metric for the *chunked
ancestor* of the stochastic recon term (`ChunkwiseSubsetReconLoss`) lives off-branch on
the `feature/fsdp-lm-trainer` n-pool lineage — the parity fixtures pin it; there is no
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
| **recon term** | one coefficiented `(SAMPLE_ROUTING, MASK_SOURCE)` forward family over all sites (§4.3) | a routing draw (those are fresh per step) |
| **decomposed layers** | the contiguous layer span owning the model's sites — every masked forward runs their decomposed path; layers outside it take the frozen `x @ W` path (~9× cheaper, no `(B,T,C)` tensors). Structural (from the model's site set), not a per-forward choice (S2) | the routed-True positions *within* a site |
| **chunk** | one independent CI transformer's `input_taps → output_sites` group (§4.6) | a data chunk / micro-batch |
| **recon locality** | the recon loss is ALWAYS on the final model output — every recon forward threads masks through the full target forward | ⚠ recon evaluated *at* a site's output. Site-local recon is NOT this method |
| **clean forward** | the full frozen forward (every site on its frozen `x @ W` path) | the decomposed forward with all masks = 1 (≈ equal only in exact arithmetic) |
| **source** | a `[0,1]` value per channel that a mask is built *from* (stochastic or adversarial) | the mask itself |
| **mask** | `ci + (1−ci)·source`, what the forward consumes | the source; the CI value |
| **delta component** | the `(C+1)`-th maskable channel carrying `x @ (W − V@U)` | the faithfulness loss (same residual, different role) |
| **CI** (causal importance) | the CI fn's per-component prediction in `[0,1]`; `ci=1` ⇒ mask pinned 1 (protected) | a probability; an attribution score |
| **lower / upper leaky** | the two squashings of the SAME CI preactivations; lower → masks, upper → imp-min | two CI functions (there is one) |
| **faithfulness warmup** | pre-loop phase: V/U trained on `L_faith` alone | the other two "warmups" → |
| **PPGD warmup (`n_warmup`)** | supplemental source-ascent iterations inside each step | LR-schedule warmup; faithfulness warmup |
| **LR warmup** (ramp knots) | a schedule's initial `frac 0 → 1` knot ramp | the above two |
| **persistent** (PGD) | sources + their optimizer moments survive across training steps | re-initialized-per-step PGD (the eval-only `PGDReconLoss`) |

---

## 2. Production constants

| | |
|---|---|
| target | Llama-3.1-8B, frozen, bf16 storage |
| decomposed | `layers[18..18].mlp.{gate,up,down}_proj` (bench: `20..31`), right-mult `W: [d_in, d_out]` |
| C | 24576 (bench: 8192); the delta component is always built → source channel dim `C+1` |
| data | fineweb, seq `T=2048`, global batch `B=512`, fresh batch per step |
| coeffs | faith `1e3` · imp `5e-6` · stoch `0.5` · ppgd `0.5` |
| imp-min | smooth-L0 `gamma: 1.0 → 0.01` linear over `[0, 1]`-frac of training; `frequency` coeff `1e-6` (= imp `5e-6` · old beta `0.2`), `reference_datapoint_count = B·T = 1048576` |
| stoch | one `uniform_k_routing(sites, n_draws=1)` family over all decomposed sites (**AMENDED 2026-08-05**: the torch-lineage sequential 3-site chunks are retired with recon chunking) |
| PPGD | source_shape `sc`, `n_warmup 2`, clamp-parameterization, Adam(β₁ .5, β₂ .99, ε 1e-8), lr const `0.01` w/ 2.5% LR-warmup |
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
| `sources[s]` | `[source_shape dims, C+1]` per site (§6 SCOPE; ★ `[1, T, C+1]`) | SRC_STEP ascent | yes (+ SRC_STEP moments) |
| `step`, schedules `gamma(step)`, `lr(step)` | scalar | — | yes |

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

wanted = (canonical activation keys)       # immutable, target-owned vocabulary       (S4)

def masked_forward(batch, masks, delta_masks, routes, wanted) -> ForwardResult:
    target validates wanted and derives its private sparse slot layout on first trace;
    assert masks cover exactly the model's sites                                          (S2)
    full forward (embed batch, all blocks, final norm, target output);
    each site s computes site_out(x, s, masks[s], delta_masks[s], routes[s]);
    each non-decomposed layer computes its frozen block  # frozen path, NOT y_dec(mask=1)  (S2)
    values = one_physical_value_per_requested_key(wanted)
    return ForwardResult.from_producer(output, wanted, values)

def clean_forward(batch, wanted) -> ForwardResult:                                         (S3/S4)
    the all-frozen full forward plus the SAME target-owned activation points;
    wanted=() takes the untouched output-only path.

def captured(result) -> {target_key: [B,T,d_key]}:                                         (S4)
    return result.captures
    # Keys and arrays are structurally one-to-one. Core never parses target keys or point
    # types. Heterogeneous widths remain separate dictionary leaves rather than being stacked.
```

### 4.2 CI

```
def ci(ci_fn, taps) -> CI{preactivations, lower, upper}:  # each {site: [B,T,C]}, sites PARTITION the model
    preactivations = CI_ARCH(ci_fn, taps)                   # architecture pinned in §4.6 (★ chunkwise)
    return CI.from_preactivations(preactivations)           # the two squashings, centralized (S5)

CI.from_preactivations(preactivations):
    lower = lower_leaky_hard(preactivations)
    upper = upper_leaky_hard(preactivations)
    # `preactivations` is a KEPT view (histograms / heatmaps plot the pre-squash value).
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

def faithfulness_loss(components) = mean_s ( ‖W_s − V_s@U_s‖_F² / ‖W_s‖_F² )              (S17)

def nonlinearity_loss(components, t):     # OPTIONAL weight-space term          (S36)
    # over the sites declaring nonlinearity_partition only; N = the site's unit count, each U_s[c]
    # grouped into N contiguous per-unit blocks along d_out
    for s, c: f_u         = ‖U_s[c] block u‖² / ‖U_s[c]‖²  # squared-norm fraction; V cancels;
                                                            # an exact-zero U_s[c] counts exactly 0
              count[s,c] = Σ_u f_u / (f_u + t/N)           # soft count; threshold RELATIVE
                                                            # to the uniform fraction 1/N
    mean[kind] = ( Σ_{s: kind} Σ_c count[s,c] ) / ( Σ_{s: kind} C_s )       # one mean per unit kind
    return Σ_kind w[kind] · mean[kind]                     # w = unit_kind_coefficients; a
                                                            # None-weighted kind is excluded
                                                            # outright (no reduction built)

def imp_min_terms(ci_upper, gamma, reference_datapoint_count):   # per-site grouping             (S7)
    for s: per_component_sums[c] = Σ_{b,t} ci² / (ci² + gamma²)   # ci = ci_upper_s[b,t,c] (S8,S9)
           f[s,c] = per_component_sums[c] / (B·T)            # per-token firing frequency
    activity = Σ_s Σ_c f[s,c]                                # bare mean term (imp coeff)
    freq     = Σ_s Σ_c f[s,c] · log2(1 + a' · f[s,c])       # a' = reference_datapoint_count (freq coeff)
    return activity, freq   # total imp contribution = imp_coeff·activity + freq_coeff·freq (S8')
    # freq is omitted (0) when no `frequency` is configured. Setting a' = B·T recovers the
    # old rolled `Σ_c f·(1 + beta·log2(1 + B·T·f))` with freq_coeff = imp_coeff·beta.
    # With `frequency.ema_halflife_steps = h` set, the freq term is evaluated at a debiased
    # EMA of f instead of the single-batch estimate (S8''), Φ(f) = f·log2(1 + a'·f):
    #   ema'   = decay·ema + (1−decay)·sg(f)        decay = 2^(−1/h), ema_0 = 0, sg = stop-grad
    #   f̂     = ema' / (1 − decay^(step+1))         # debias: f̂ = f exactly at step 0
    #   freq   = Σ_s Σ_c Φ(f̂) + sg(Φ'(f̂))·(f − sg(f))   # value Φ(f̂), gradient Φ'(f̂)·∂f/∂θ

# Each recon term: a SAMPLE_ROUTING returning a statically-sized FAMILY of routing draws
# over ALL sites, each draw = one forward (§6).
def stochastic_recon_loss(components, ci_lower, residual, clean_output):
    total, n_forwards = 0, 0
    for routes in SAMPLE_ROUTING(key, [B,T]):            # fresh per step                 (R1,S11)
        masks, delta_masks = make_masks(ci_lower_s, source_s ~ U[0,1]^[B,T,C+1])  ∀ s ∈ sites
        total += kl_per_position(masked_forward(residual, ALL_SITES, masks, delta_masks, routes),
                                 clean_output)
        n_forwards += 1
    return total / n_forwards                                                               (S10′)

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
def train_step(state, batch, step):                  # batch = fresh target input          (S18)
    ci_keys = ci_fn.capture_keys                       # immutable canonical keys          (S4)
    cln_result = sg[ clean_forward(batch, ci_keys) ]                                              (S3)
    cln = cln_result.output
    CI = ci(ci_fn, captured(cln_result))             # ONE conceptual CI eval/step;
    ci_lower, ci_upper = CI.lower, CI.upper                   # recompute allowed (deterministic)
    # -- supplemental adversary ascents (components & CI detached) --
    set SRC_STEP lr = sched_src(step)                # stepped once per TRAINING step       (S13)
    repeat n_warmup:
        sources_grad = ∂/∂sources adversarial_recon_loss(sg[components], sg[ci_lower],
                                                  EFFECTIVE(sources), residual, cln)
        sources, sources_opt = sources_update(sources, sources_grad, sources_opt)

    # -- main losses: live components & ci_fn; the ppgd term's sources detached --
    L = 1e3·faithfulness_loss(components) + 5e-6·importance_minimality_loss(ci_upper, gamma(step))
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
target-owned clean activation captures → CI` — but the pinned REALIZATION changes from one global transformer
over every site's input concatenated to a **chunkwise transformer**: the sites partition
into chunks, and each chunk runs its OWN independent transformer. (The CI fn is a protocol
`dict[InputTap,Array] → CI`; the input keyspace — clean-path tap keys — is independent of
the output keyspace — the decomposition sites — and the output sites must PARTITION the
model's sites.)

```
in:   {tap_key: x [B,T,d_tap]}  (`ci_fn.capture_keys` resolved by the target; opaque to core, §4.1 S4)
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
or transformer concept in core — the composition layer computes it from the taps it authored (their
widths summed) and core stays agnostic to what the taps mean. Layerwise (one site per
chunk) and global (all sites in one chunk) are degenerate chunkings of this same form.
The positionless toys (`has_position_axis=False`) are the MLP siblings (`LayerwiseMLPCIFn` /
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
(S19), the schedules (S20) — and the `param_decomp/tests/targets/equivalence/` suite still pins them. Refs:
`ci_fn.py` (`ChunkwiseTransformerCIFn`, `CIBlock`, GELU line, `CI_FN_RMS_EPS`, `inv_freq`).

---

## 5. Semantic invariants

| id | invariant |
|---|---|
| S1 | `mask = ci + (1−ci)·source` per component channel; the delta channel is the raw source value, never ci-interpolated; CI has no delta output. |
| S2 | A masked forward's masks (and routes, when present) cover exactly the model's sites — asserted fail-closed at the target boundary. Non-decomposed layers run the frozen `x @ W` path — zero V/U gradient, zero decomposition rounding, none of the decomposed path's ~9× compute or `(B,T,C)` activations — as a structural fact of the model's site set, never a per-forward choice. **AMENDED 2026-08-05** (Oli, dechunk): per-forward partial live-site sets are retired with recon chunking; masked forwards are total over the model's sites. |
| S3 | The recon target is the `.output` of the step's all-frozen `clean_forward(batch, clean_capture)`: the frozen-path full model (embed the token batch, all blocks, final norm, target head), stop-gradient. Never the `mask=1, delta=1` decomposed identity (differs in bf16 and pollutes the graph). A subset decomposition simply leaves non-decomposed blocks on the frozen `x @ W` path. **AMENDED 2026-06-24** (Oli-approved): removed residual-start — the model takes the target batch and embeds internally, so the clean forward is the whole-model frozen forward (was: a suffix over a separately harvested residual). **AMENDED 2026-07-30** (Dan/Oli design review): the specialized `clean_output` protocol method is retired into the empty-key arm of the unified clean-forward result; semantics are unchanged. |
| S4 | **RE-AMENDED 2026-08-03 (Oli review):** clean and masked activation access use one selective, read-only surface with a strict **one key = one activation = one array** contract. Core passes an immutable frozenset of canonical keys directly to `clean_forward` or `masked_forward`; no resolved plan type crosses the target boundary. Each target owns a closed point algebra, wire syntax, validation, and sparse lowering, and canonicalizes the set into private sources/slots when the forward is first traced. Invalid keys therefore fail deterministically at first trace rather than when the step factory is built. Both forwards return the keyed arrays in `ForwardResult.captures`; the same key denotes the same tensor on clean and masked paths. Site names are never alternate names for matrix inputs. Physical activations are named directly; no site→input alias table exists. Consumers ask for the vectors they actually need. Component-activation harvest is target-owned and must fuse its site-input needs with the caller's capture keys into one frozen forward. Core owns and parses no target anatomy. An empty frozenset routes through the untouched output-only computation. On an active mesh every captured tensor is pinned at this producer boundary with its leading batch axis over `(replicate, fsdp)`, before consumers fan out. `site_output_keys(sites)` preserves request order: the returned key at index `i` names the output of `sites[i]`. Capture keys are static per compiled executable: a different set selects/compiles another graph; dynamic/superset fallback is excluded. Capturing a value may legitimately change HLO/fusion and retains that requested tensor across remat; unrequested values do not escape. This supersedes both the 2026-06-22 specialized `read_activations(batch, ci_fn.input_names)` accessor and the briefly-public `resolve_capture` plan abstraction. Transformer-private points currently include residual boundaries, post-attention residuals, canonical block taps, and matrix outputs; toy targets use the same one-name-per-activation rule with their own target-local grammars. |
| S5 | **AMENDED 2026-06-22** (Oli-approved): the CI fn returns a `CI{preactivations, lower, upper}` bundle (was `(ci_lower, ci_upper)`). `lower` and `upper` are two squashings of the SAME `preactivations`, centralized in `CI.from_preactivations` (no impl re-triplicates them); `preactivations` is KEPT as a consumed view (histograms / heatmaps plot the pre-squash value). `ci_lower ≡ CI.lower` feeds every mask; `ci_upper ≡ CI.upper` feeds imp-min only; no other crossing. The squashings themselves (S6) are UNCHANGED — only the bundling and the `preactivations` view changed. The CI fn is a protocol `dict[InputTap,Array] → CI` whose output sites PARTITION the model's sites (input keyspace independent of output keyspace). |
| S6 | The squashings' forward/backward are exactly §4.2 — including `lower_leaky_hard`'s grad-sign-gated lower leak (a custom VJP, not autodiff of the forward). The backward is a nested `where` with `<=` boundaries: on `0 < x <= 1` pass `g`; at `x <= 0` pass `α·g` ONLY where `g < 0`, else `0`; at `x > 1` pass `0`; `α=0.01`. The boundary tie at `x=0` resolves to the LOWER (`x <= 0`) branch — this `<=` placement is exactly what makes torch (`ci_sigmoids.py`) and JAX (`ci_fn.py`, `_lhs_b`) bit-identical and is load-bearing, not incidental. Grad-checked by `param_decomp/tests/core/test_lower_leaky_hard_grad.py` (`g<0` vs `g>0` at `x<0`, `x∈(0,1]`, `x>1`; #789). |
| S7 | Imp-min groups per site: the frequency penalty's `log2(1 + a'·f_c)` consumes one site's per-component frequency. Merging sites/layers into one group is incorrect (convexity). |
| S8 | The per-component frequencies `f_c = (Σ_{b,t} ψ(c)) / B·T` are over the **global batch**, formed before the `log2`. (Per-shard `f_c` combined after the log are incorrect — Jensen; see D2.) The two imp-min terms (`activity = Σ_c f_c`, `freq = Σ_c f_c·log2(1 + a'·f_c)`) share these per-component sums in one pass. |
| S8' | Imp-min contributes `imp_coeff·activity + freq_coeff·freq` with INDEPENDENT coefficients; the frequency normalizer `a' = reference_datapoint_count` is explicit (batch-invariant at fixed firing rate), not the implicit `B·T`. `freq` is absent when no `frequency` is configured. `a' = B·T` recovers the old rolled `imp_coeff·Σ_c f·(1 + beta·log2(1 + B·T·f))` with `freq_coeff = imp_coeff·beta`. |
| S8'' | **ADDED 2026-07-31** (#135): with `frequency.ema_halflife_steps = h`, the freq term is evaluated at a debiased EMA of the per-component frequencies instead of the single-batch `f_c` — a noisy estimator that the convex `Φ(f) = f·log2(1 + a'·f)` turns into a systematic over-penalty on rare components (Jensen). `ema' = decay·ema + (1−decay)·sg(f)`, `decay = 2^(−1/h)`, `ema_0 = 0`, debiased `f̂ = ema'/(1 − decay^(step+1))` — so `f̂ = f` exactly at step 0. Value `Σ_s Σ_c Φ(f̂)`; gradient through the current batch at FULL scale via the surrogate `sg(Φ'(f̂))·f` (the naive `Φ(EMA)` shrinks gradients by `1−decay`, silently retuning `freq_coeff`), the estimate stop-gradded so it is not a lever the CI fn can steer. State: per-site `(C,)` fp32 `TrainingItem.freq_ema`, None iff unconfigured (checkpoint tree unchanged for existing configs); the EMA consumes S7/S8's per-site global-batch `f`. Defined for the plain PD objective only: the EMA carries one frequency stream per site, while the targeted (tPD) step takes the penalty on two independent streams (target + non-target, each on its own CI population) — tPD EMA is not implemented, and the knob is refused loudly at parse and at `build_targeted_objective` rather than half-applied. While `f` moves faster than the halflife the loss VALUE lags it (an average of a moving trajectory, amplified by convex `Φ`) — a large smoothed-vs-batch gap during the first few halflives is the estimator converging, not an instability. |
| S9 | The smooth-L0 width `gamma(step)` follows its `ScheduleConfig` (canonical: linear `1.0 → 0.01` over the run), evaluated by `schedule.scheduled_value_traced`. Constant-`gamma` is a bare float; every knot must keep `frac > 0` (`build_objective`). **AMENDED 2026-07-02** (#915): the former windowed linear anneal (`{p,gamma}_anneal_{start,end}_frac`) is retired — every config on record used the trivial `[0, 1]` window — and the anneal now shares the schedule's `decay_steps − 1` denominator (S20), reaching the final value AT `step = steps − 1` instead of one step past the run (a ≤O(1/steps) per-step trajectory change vs the old `step / steps`). **AMENDED 2026-07-16** (Oli-approved, expressive schedules): `ScheduleConfig` is knot-based (see S20); the canonical p-anneal is `max_val=2.0` over knots `frac 1.0 → 0.2` (pointwise identical to the old linear form). Constant-`p` is a bare float; the warmup refusal generalizes to "every knot `frac > 0`" (`build_loss_terms`). **AMENDED 2026-08-27:** the `L_p` penalty — `(ci + eps)**pnorm`, its `p: 2.0 → 0.4` anneal and `eps` floor — is retired with `ImportanceMinimalityLoss`; smooth-L0 is the one per-value penalty and this row (formerly the S9/S9′ pair) is the gamma rule alone. |
| S10′ | The recon objective is a static tuple of coefficiented loss TERMS (one per configured recon loss metric, in config order). Each term is one `(SAMPLE_ROUTING, MASK_SOURCE)` family over ALL sites; the term's loss = mean over its routing draws (each draw one forward) of `kl_per_position`; the total adds `coeff · term` per term. Term structures (sampler identities, family sizes, strategy kinds) are fixed across steps. The §4 pseudocode shows the production two-term instantiation (`stochastic_recon_loss` + `adversarial_recon_loss`). Recon KL direction is pinned by S25; the mean-over-forwards ≡ accumulator identity by S26. **AMENDED 2026-08-05** (Oli, dechunk): the per-term live-set plan — chunked (`sites_per_chunk`) and per-site ("layerwise") entry lists — is retired; every term routes over all sites, matching the published VPD recipe, and masked forwards are total over the model's sites (S2). |
| S11 | `uniform_k_routing`, per position: `k ~ U{1..|sites|}` then a uniform `k`-subset of the model's sites routes True. Routing draws are fresh per step, sampled inside the step. |
| S12′ | An adversarial term's loss forward consumes its sources as LEAVES (no ascent-graph history); gradient flows to components and (through `ci_lower`) to the CI fn — and, for persistent sources, to the leaves themselves (S14′). The PRODUCTION adversarial term masks ALL sites and routes everywhere; subset-routed adversarial terms route per their `SAMPLE_ROUTING`. |
| S13′ | Per persistent term: source updates per training step = `n_warmup + 1`, all through THAT term's persistent SRC_STEP optimizer state; its source LR schedule advances once per training step. The source LR is `scheduled_value_traced` over the term's `ScheduleConfig` — any fn_type (**AMENDED 2026-07-17**, Oli, #18: the constant-after-warmup-only restriction is retired — it guarded a long-gone specialized evaluator with no decay branch (#646). ★ production remains constant `0.01` w/ 2.5% warmup). **AMENDED 2026-07-02** (#915): the former `warmup_pct==0` accepted seam (JAX clamped `warmup_steps = max(floor(...), 1)` → source LR `=0` at step 0) is retired — at `warmup_pct==0` JAX now matches torch's full LR at step 0. Production uses 2.5% warmup and is unaffected either way. **AMENDED 2026-07-16** (Oli-approved, expressive schedules; ported 2026-07-27): under knot schedules the source LR keeps "any shape" — the 07-16 draft predated the 07-17 relaxation and restated the decay refusal; the relaxation stands. The 2.5% warmup is now the explicit ramp knots `(0, 0) → (0.025, 1.0) → (1.0, 1.0)` on normalized time (the old `warmup_pct` ramp counted `int(steps·pct)` whole steps — an O(pct) reparameterization, not pointwise-representable). |
| S14′ | A persistent term defaults to `adversary_objective: e2e`: the complete term stays in the outer components/CI backward, but every persistent-source ascent maximizes output reconstruction alone. Warmups use the output-only spec. When the outer term also includes S35 hidden-activation reconstruction, the final ascent retakes `dL_e2e/d(sources)` against the SAME draws with pre-update components and detached CI, replacing the fused source gradient; without hidden-activation reconstruction, the fused gradient already is `dL_e2e/d(sources)` and is reused. Explicit `adversary_objective: term` instead takes its final ascent gradient from the SAME graph as the main backward and ascends the complete term. The term's coeff is applied to its MODEL-SIDE cotangents only (`model_cotangents_scaled` wraps prepared weights + CI — bit-identical forward, cotangents scaled by the per-step coeff), and the term enters the differentiated total at weight 1, so either source objective stays coeff-unscaled. Components/CI receive exactly `coeff·dL_term/dθ`; the final ascent stays full-strength through coefficient gates; reported `total` remains `Σ coeff·L`. Applied after backward; never use post-update params. **Amended 2026-08-06 twice** (schedulable coefficients; coeff moved to model-side cotangents); **amended 2026-08-17** (`e2e` source-only objective, made default). |
| S15 | Every source update ends with `PROJ` (★ clamp to `[0,1]`). Init: ★ `sources ~ U[0,1]` i.i.d. |
| S16 | Batch-1 sources are identical on every data-parallel replica at every step (identical init, identical updates from the global-batch gradient). batch-B sources (`bc`/`bsc`) shard with the batch instead. (Implementation mapping in §8/§9.) |
| S17 | `faithfulness_loss` is the mean over sites of each site's relative squared Frobenius error `‖W_s − V_sU_s‖²/‖W_s‖²`, recomputed from live V/U each step. Frozen target norms are computed once at engine setup, slot-aligned with the persistence stacks, and validated for exact site coverage and finite positive values before JIT; the in-step reduction runs per-slot on the stacked deltas (per-site slices of the stack-sharded layout would redistribute cross-node, task 577). |
| S18 | Each training step consumes a fresh target input batch (a pure function of `(seed, step)` for O(1) resume); the model embeds it internally where applicable. **AMENDED 2026-06-24** (Oli-approved): removed the prefix harvest (residual-start) — there is no separate prefix forward; clean and masked forwards take the token batch directly. **AMENDED 2026-07-30:** the unified `clean_forward` / `masked_forward` names replace the former `clean_output` / `read_activations` / `masked_output` method grid; input semantics are unchanged. |
| S19 | Components gradients are global-norm-clipped at `0.01`, before the optimizer step. CI fn is unclipped (production). The clip coefficient uses torch's eps convention: `clip_coef = min(1, max_norm / (total_norm + 1e-6))` (`torch.nn.utils.clip_grad_norm_`). This `+1e-6` is canonical and matched JAX-side (#643); plain `optax.clip_by_global_norm` divides by `max(total_norm, max_norm)` with NO eps, which at `clip=0.01` (clip fires almost every step) gives a ~1e-4 relative component-grad difference each step — a real per-step deviation the JAX clip must avoid by reproducing the `+1e-6`. |
| S20 | Both main optimizers are AdamW, `wd=0`, betas `(0.9, 0.999)`, eps `1e-8`; LR cosine to `0.1×` start, no warmup, stepped per training step. Cosine convention is canonical-torch: `progress = (step − warmup) / (decay_steps − 1)`, so LR reaches `0.1×` at `step = steps − 1` (`schedule.py`). This is matched JAX-side (#642); plain `optax.cosine_decay_schedule(lr, steps, alpha=0.1)` divides by `steps` and reaches `0.1×` one update LATER (at `count = steps`), a genuine formula divergence of ~O(1/steps) ≈ 2.5e-6/step at 400k that the JAX schedule must avoid by using the `decay_steps − 1` denominator. Both main LRs are evaluated by `scheduled_value_traced` honoring the full `ScheduleConfig`; the cosine-to-0.1-no-warmup shape is the METHOD's recipe, authored per config, not a constraint the schema enforces. **AMENDED 2026-07-02** (config-gated, default off = oracle parity): `type: muon` on a main optimizer config swaps that group's update rule for `optax.contrib.muon` (Newton-Schulz-orthogonalized momentum for the group's matrix leaves, Adam fallback for the rest), under the same cosine schedule and the same S19 clip chain. Every config without `type: muon` deserializes to `type: adamw` and keeps the canonical trajectory bit-identical. **AMENDED 2026-07-11** (muon × ci-fn matrix experiment): the muon matrix-leaf labeling is per-group (`run_state.build_optimizers`) — V/U and the MLP CI fns use optax's default 2D rule; the chunkwise CI fn passes `MuonDimensionNumbers` marking its 3D leaves as chunk-batched matrices (trailing two axes orthogonalized) and its 2D bias stacks as Adam-fallback, since the default 2D rule would label that tree exactly backwards. **AMENDED 2026-07-12** (fast muon): `impl: stacked` on a muon config swaps the per-leaf NS for `muon_stacked.py` — same-shape leaves batched into one NS with the stack axis sharded over `(replicate, fsdp)`, making each orthogonalization device-local (GSPMD otherwise lowers per-leaf NS on the ÷N masters into per-iteration full-Gram all-reduces with the largest matmul replicated on every device — the measured 3.3× muon-ci step hit). Same `MuonState` pytree (checkpoints round-trip across impls); trajectories match `impl: optax` up to float reassociation (the D4 tolerance class). **AMENDED 2026-08-19** (per-kind NS): the stacked NS batches per SEMANTIC KIND — each muon leaf (one kind's stack) gets its own batched NS at its placement `ns_compute` waypoint verbatim, every preset declaring the node-axis stack split (`{stack: replicate}`); a kind whose stack does not tile the split refuses at the stacked-muon consumer's claim (`placement.assert_stacked_muon_*_staging`, fired at optimizer build and the LM pre-submit gate) — only stacked-muon runs consume the row, so non-muon runs keep any-stack-length placement. Cross-kind shape-grouping, the concat, and stack padding are REMOVED (shape-grouping is banned by design decision); the waypoint is reached by a one-axis-per-reshard hop chain (`muon_stacked.staging_hops`, the anti-remat spelling). Values are unchanged (bitwise vs the grouped path in the migration probe — batching never crossed matrix boundaries), so the D4 tolerance class and the `MuonState` round-trip both stand. `ns_steps` (default 5) and `ns_dtype` (default float32; bfloat16 = the Kimi recipe, stacked-only, masters/momentum stay fp32 per N1) are config knobs. Default `impl: optax` keeps the 07-02/07-11 experiment arms' exact semantics. **AMENDED 2026-07-16** (Oli-approved, expressive schedules): `ScheduleConfig` is now KNOT-BASED — `max_val` (the one sweepable magnitude) × a piecewise curve over knots `{at, frac, interp ∈ linear|cosine|hold}` on normalized time `t = step / (total_steps − 1)`, so the `at = 1.0` knot lands exactly ON `step = steps − 1` (the same torch-parity endpoint the old `decay_steps − 1` denominator pinned; a no-warmup cosine/linear decay is POINTWISE IDENTICAL to the old form at every step a run trains — `test_schedule.py::test_migrated_seat_is_numerically_unchanged`; at the one-past-the-end `step == total_steps` the knot form HOLDS the final value where the retired evaluator extrapolated through it, a count no update consumes). `hold` keeps the previous knot's value then jumps AT its knot (step functions, delayed starts); a bare float parses as the constant schedule; `frac ∈ [0,1]` with the peak `frac = 1.0` attained. The method's LR shape is knots `(0, 1.0) → (1.0, 0.1, cosine)`, authored per config — no schema gate pins it (`assert_canonical_algorithm_config` was deleted on the release line). The merged stochastic/persistent loss's `adv_fraction` uses the same schedule abstraction, with `max_val <= 1` enforcing its probability range. |
| S21 | Faithfulness warmup (400 × AdamW lr `1e-3` on the S17 faith term alone) precedes step 0; its optimizer is discarded. |
| S22 | Checkpoints round-trip ALL trajectory state of §3 — including every persistent term's sources + SRC_STEP moments + step/schedule counters — such that a resumed run continues the same trajectory (modulo RNG streams and kernel nondeterminism, cf. D4). **AMENDED 2026-07-06** (checkpoint split): each step stores TWO orbax items — `decomposition` (the trained product: V/U `components` + `ci_fn`) and `training` (the process: both optimizer states, persistent adversaries, `step`). `TrainState` COMPOSES exactly these two (`TrainState{decomposition, training}`) — one representation shared by the trainer and the checkpoint, so save/restore map onto its own fields with no regrouping. Trainer save/restore round-trips both (S22 semantics unchanged); fine-tune init (S33) and every downstream consumer restores ONLY `decomposition`, with zero knowledge of training's optimizer/adversary structure. Both items save sharded (no on-loop full-gather). No in-code compat for pre-split single-`default`-item checkpoints: this is a clean format break — pre-split runs must be migrated by an ad-hoc script before a tip trainer/consumer can read them, not carried by a dual-read fallback. **SIGTERM lifecycle (decision):** a SLURM-delivered SIGTERM sets a flag that is serviced at the main train-step boundary (synchronous save of the completed step, then exit for requeue), inside the faith-warmup loop (clean exit WITHOUT a save — no valid checkpoint exists pre-step-0, and resume skips warmup whenever any checkpoint is present, so a partial step-0 save would resume as if fully warmed; the requeue redoes warmup), and inside the in-loop eval pass (abandon the partial pass unlogged, fall through to the step-boundary save). The **first-jit-compile window remains unserviced** — a SIGTERM there is honored only once compilation finishes and control reaches the next servicing point. Periodic `save_every` is the BACKSTOP guarantee for every window; the SIGTERM→save path is the low-latency fast path, not the sole guarantee; historical preemptions have fallen back to periodic checkpoints, and warmup/eval servicing closes the two largest unserviced windows. **AMENDED 2026-08-17 (no-checkpoint cadence):** `cadence.checkpointing` is a closed union — `{kind: periodic, save_every, retention}` keeps THIS row's semantics unchanged, while the authored `{kind: none}` arm writes NO checkpoints at all: no periodic saves, no final-step save, and no SIGTERM save either (the flag still exits the loop cleanly at the step boundary; nothing is written) — for probe/measurement runs whose trajectory is throwaway. With nothing to resume from, a no-checkpoint run id is single-entry: the engine drops a marker on first entry and refuses re-entry (requeue or --run-id rerun) loudly, so nothing can silently retrain from step 0. |
| S23 | A persistent source bundle feeds exactly ONE loss term. (The fused-backward S14′ unscaling divides that term's coeff out of the source gradient; a bundle shared across terms would make the division wrong.) |
| S24 | A persistent term's WARMUP ascents forward all sites, routed everywhere — regardless of the term's own routing (torch parity: `persistent_pgd_state.warmup` hardcodes route-all). A fresh-PGD term draws its routing ONCE per step, shared by all its ascents and its main loss forward (torch parity: `pgd_masked_recon_loss_update`). |
| S25 | Recon KL direction is `KL(softmax(clean_output) ‖ softmax(masked_output))` — `P = clean`, `Q = masked`. Equivalently `Σ p_clean · (log p_clean − log p_masked)`. Torch `recon_loss_kl` realizes this as `F.kl_div(log_softmax(pred=masked), softmax(target=clean), reduction='sum')` (`batch_and_loss_fns.py::recon_loss_kl`); reversing the arguments is a silent, plausible bug, so the direction is semantic, not incidental. |
| S26 | Normalization identity: `(Σ_forwards sum_kl) / (Σ_forwards n_positions) == mean_forwards(sum_kl / n_positions)`, valid **iff** every forward shares the same `(B,T)` (so `n_positions` is constant across forwards). This precondition is what makes JAX's mean-over-forwards (S10′) equal torch's `(Σ sum_kl)/(Σ n_positions)` accumulator; uniform `(B,T)` across all forwards in a term is therefore required. (Stated in LOSS_PARITY_DESIGN §4e; cross-ref S10′.) |
| S27 | The CI transformer's RoPE `inv_freq` (§4.6) is a non-trained buffer and MUST be stop-gradient'd in the CI fn. In JAX it is a pytree leaf and would otherwise be optax-updated, silently drifting the rotary frequencies. In the chunkwise CI fn `inv_freq` is shared across chunks (a single buffer, NOT mapped over `n_chunks`) and `stop_gradient`'d inside `ChunkwiseTransformerCIFn.__call__` before the `filter_vmap`. Ref `ci_fn.py` (`ChunkwiseTransformerCIFn.__call__`, `jax.lax.stop_gradient(self.inv_freq)`). |
| S28 | Eval runs in-loop in TWO tiers (§10): a FAST scalar tier on cadence `eval.every`, and a SLOW/plot tier on cadence `eval.slow_every` (a multiple of `every`, so it lands on a fast-eval step and REUSES that step's in-memory eval batches — no second forward, no second data read). Slow/plot eval is IN-LOOP ONLY — there is no offline/retrospective CLI (`pd-slow-eval` and `run_offline_slow_eval` are removed; `slow_eval.py` is a pure library of the accumulate/render/metric fns the in-loop tier calls). The slow tier renders the plot metrics (`CIHistograms`, `ComponentActivationDensity`, `CIMeanPerComponent` → the shared `render_slow_eval_figures` figure set), the config-gated CI-heatmap / `PermutedCIPlots` figures + the `IdentityCIError` scalars (both driven by the cheap `(T, C)` per-site position-CI matrix from `accumulate_position_ci`, gated on the config naming them), and — when the config names it — the `UVPlots` figure. `UVPlots` is a config-gated figure metric usable for ANY decomposition (the metric-operation pattern: it returns a typed PNG value): cheap for the toys (TMS/ResidMLP — small, replicated, on-host V/U, no gather; rendered off the toy probe CI by `render_uv_figure`), and a NAIVE full host gather of the C-sharded V/U for the LM in-loop tier (`run.py` gathers V/U only when `want_uv_plots` and passes `components` to `render_permutation_figures`). The LM gather OOMs / breaks at production C BY DESIGN — no special handling, the gather (collective, on the eval pass) is the cost; zero cost when `UVPlots` is not named. The slow tier is forward-only (one target forward + the CI-fn forward + sigmoid + C-sized reductions; no backward / masking grid / PPGD / optimizer state), so its peak HBM ≤ the train step's own high-water mark (the UVPlots gather aside) — where training fits, in-loop slow eval fits. **AMENDED 2026-06-18:** reverses the migration-era "slow tier is offline-only, no in-loop analog" decision (and supersedes the retired torch-era push-triggered `pd-offline-eval` sidecar); the slow tier runs in-loop next to the fast pass. **RE-AMENDED 2026-06-19:** drops the "`pd-slow-eval` CLI retained" clause — slow eval is in-loop only — and makes `UVPlots` an in-loop config-gated figure metric (cheap for toys, naive/breaks-at-scale for the LM by design) rather than offline-only. |
| S28a | **Multi-host split (in-loop slow tier).** The slow tier separates a COLLECTIVE part — run in lockstep on ALL ranks — from a pure-HOST part — run on rank 0 only, OFF the train loop. Collective: the jitted forward + the device→host pull (`accumulate_site_reductions` / `accumulate_position_ci` materialize C-sharded reductions to numpy; `np.asarray` triggers the all-gather every rank must join — `accumulate_position_ci` runs only when the config names a CI-heatmap / permutation / identity-error metric, so it adds zero cost otherwise). When the config names `UVPlots`, the C-sharded V/U is ALSO gathered to host here (`np.asarray` over `state.components.vu`) — a NAIVE collective gather that OOMs / breaks at production C by design (S28); gated on `want_uv_plots`, so zero cost otherwise. Host: pure matplotlib rendering of the figure PNGs (the base set + the config-driven CI heatmaps + `UVPlots` when the V/U was gathered), handed to a `BackgroundRenderer` BACKGROUND THREAD on rank 0 (`run.py`); the renderer returns a `DeferredMediaRecord` and the shared `MetricsSink` alone performs serialized W&B transport — the main loop proceeds immediately on every rank, near-zero cross-rank divergence. The thread touches ZERO jax/device state; at most one render is in flight (a new submit `join()`s the prior first); an `atexit` join flushes the last render before process exit (the trainer never calls `wandb.finish`). The `IdentityCIError` SCALARS ride the live `_step` axis (the fast eval record, computed on the collective path and logged synchronously by the sink at the eval step — cheap, and scalars must stay `_step`-monotonic, so they are NOT deferred to the background thread); the FIGURES carry `slow_eval/figure_step=now_step` as a dedicated W&B metric axis and log without an explicit `_step`, so a background render that finishes after synchronous metrics advance `_step` is retained rather than rejected as an out-of-order write. The toys render the same `UVPlots` figure synchronously on CPU (single-process, tiny V/U): `toy_uv_eval.render_uv_metric` returns a typed `PNGImage` from a toy `EvalOperation`; the normal collision-checked operation record then reaches `MetricsSink`, which alone owns W&B transport on the live `_step` axis. An authored toy `UVPlots` operation fails while binding when no W&B transport is configured. |
| S29 | JAX `EvalConfig` carries `batch_size`, `every`, `n_steps`, `slow_every` / `slow_on_first_step` + the authored `metrics` list; the two cadences drive the two tiers (S28) for EVERY family. WHICH tier a metric runs in is not configurable at all: each eval-metric config declares `slow: ClassVar[bool]` beside its own definition (no default; swept at import in `experiments/eval_config.py`, which also refuses a metric that merely inherits one), and the shared `schedule_for(metric, eval)` turns that declaration plus the two cadences into the operation's `EvalSchedule`. `ClassVar` is what makes it unauthorable — pydantic ignores it, and `extra="forbid"` refuses a YAML `slow:` key. The remaining knobs are per-metric fields on the metric configs: `CIHistograms.n_batches_accum` is the histogram-sample cap (None = uncapped), `CEandKLLosses.rounding_threshold` and `CI_L0.ci_alive_threshold` ride their own metric configs, and `CIHistograms.density_heatmap_n_bins` is the opt-in for the per-token CI density heatmap (None = off — the add-on shares the slow-eval forward's `lower`, adds only an on-device per-component bincount `(C, n_bins + 1)` — column 0 = underflow (CI < 1e-9, incl. exact-0), columns 1..n_bins = log-spaced `[1e-9, 1]` bands — accumulated over EVERY batch, and emits `figures/ci_density_heatmap` on a log y-axis; no raw-value host transfer). `eval` is atomic-optional (`EvalConfig | None`): `None` disables in-loop eval (both tiers). **AMENDED 2026-06-18**: re-adds `slow_every` / `slow_on_first_step`, deliberately dropped in the original S29 when the slow tier was offline-only. **AMENDED 2026-08-12**: the two `CIHistograms` value histograms are binned ON DEVICE (`jnp.histogram` over the data's own min/max — exactly the edges `ax.hist` picks), so only `(counts, lo, hi)` crosses to the host; the `(*leading, C)` values are never gathered. They are gathered/binned ONLY for a metric that renders a value histogram — `ComponentActivationDensity` / `CIMeanPerComponent` read pre-reduced `(C,)` arrays and transfer nothing. Because each batch bins against its own min/max and counts on different edges do not sum, a `CIHistograms` operation REQUIRES `eval.n_steps=1` and asserts loudly otherwise; `n_batches_accum` correspondingly admits only None/1. Subsampling the values instead was rejected: `lo`/`hi` are order statistics, so a sample renders a narrower x-axis and empties bins holding a rare-but-real fraction of the mass. **AMENDED 2026-08-12**: `slow_on_first_step` fires the slow tier at step 0 — the untrained baseline, before the first optimizer step — not at the first `every` pass. The engine dispatches one eval pass at step 0 on a fresh run (`start_step == 0`); `eval_due` refuses step 0 for a bare `Every` cadence, since every cadence divides 0 and modular arithmetic alone would fire every operation there. A resumed run has no step 0 and takes no baseline. **AMENDED 2026-07-01**: adds `density_heatmap_n_bins` (opt-in per-token CI density heatmap sharing the CIHistograms forward). **AMENDED 2026-07-30**: restores the torch-era ontology — the tier is declared by the metric, not assigned by the binder, and `slow_every` / `slow_on_first_step` move from the (now deleted) `LMEvalConfig` onto the shared `EvalConfig` so a non-LM family can honour a declaration. `UVPlots`, previously slow under the LM binder and fast under the toy one, resolves to slow (torch's answer). |
| S30 | `cfg.cadence.train_log_every` divides `cfg.eval.every` (`eval.every % train_log_every == 0`, asserted LM-side in `experiments/lm/training.py`) so every eval step is also a train-log step. |
| S31 | **RETIRED 2026-08-25:** the two standalone hidden-acts recon eval metrics (`CIHiddenActsReconLoss`, `StochasticHiddenActsReconLoss`, formerly `hidden_acts_eval.py` on the slow tier) are removed — S35's per-term `hidden_acts_reconstruction` is the one hidden-activation measurement. What survives this row: per-element MSE on a decomposed site's OUTPUT activation remains unspellable as a training loss — site-local recon is the conceptual no-no this trainer refuses (LOSS_PARITY_DESIGN §4c) — and the capture seam the 2026-08-03 amendment built (`site_output_keys(sites)` + site outputs riding the ordinary clean/masked forwards) stands, used by the attn-patterns eval. |
| S33 | Fine-tune init (`ExperimentConfig.resume_provenance`, LM-only): a fresh run whose own `ckpts/` is EMPTY and whose `resume_provenance is not None` initializes from a PARENT checkpoint — it restores the parent's `ckpts/<parent_step>` `decomposition` item (the trained `components` (V/U) + `ci_fn`, cf. S22) onto the fresh reference's decomposition; the optimizer states, persistent sources, and `step` are the FRESH reference's (`step = 0`). Rationale: a fine-tune runs a NEW LR/gamma-anneal schedule computed over the new `cfg.steps` from 0, so carrying stale Adam momentum / a stale adversary would mis-scale the restart; faith warmup is also skipped (the parent's V/U is already faithful). The run records lineage via `resume_provenance` in `launch_config.yaml` + `wandb.config`. The parent's decomposition STRUCTURE (sites names + C, ci-fn arch) must match the new config's — asserted from the parent's pinned `launch_config.yaml` (`experiments/lm/training.py::assert_finetune_structural_compat`) before the orbax restore; only LR / coeffs / gamma / seq / batch / steps may change. This is distinct from same-config requeue-resume (S22): on a subsequent requeue the run's own `ckpts/` is non-empty, so `restore_latest` from its own dir wins and provenance is ignored. |
| S34 | `mixed_persistent_stochastic(state_key, adv_fraction)` (`MergedStochasticSubsetPPGDReconLossConfig` / `MixedPersistentStochasticSources`, #6): per forward, ONE `assign ~ Bernoulli(adv_fraction)` draw per BATCH ELEMENT (`adv_fraction` is a `ScheduleConfig` evaluated per step like S9's gamma; both endpoints must lie in [0, 1]), broadcast over all position axes — per-sample by design (the R1 "distribution as stated"), so no sample is scored against a mixed-family attention context. `assign` → the term's persistent bundle (component sources + delta channel), routing forced all-live; `¬assign` → fresh `U[0,1]` per (position, component[+delta]), routed per the term's `SAMPLE_ROUTING`. Masks compose per S1; the mask-level `where` gates the source gradient to adversarial-assigned samples (S24's route-all warmup ascents remain the full-coverage backstop). This is the one mask source that modulates its term's routing (`route = assign OR routing_draw`). A term carrying it IS a persistent term for S12′–S16 and S22–S24. Expectation identity, exact since batch elements are forward-independent: `E_assign[term] = adv_fraction · (all-sites persistent term) + (1 − adv_fraction) · (stochastic term)` — so `coeff 1.0, adv_fraction 0.5` replaces the canonical `0.5·stoch + 0.5·persistent` pair in one masked forward; the semantic delta vs the pair is adversary coverage/staleness only (each persistent slot updates on the ~`adv_fraction` of steps its sample draws adversarial). |
| S35 | Optional per-term hidden-activation reconstruction (#94): any recon loss config may carry `hidden_acts_reconstruction: {coeff, points}`, adding `coeff · mean_points(relative squared error at the named activations)` to THAT term's masked-vs-clean comparison on every draw. It shares the term's outer-objective forwards; their capture keys include the named points. `points` name activations in the TARGET's own tap vocabulary and have NO default (which internal activations matter is an experiment question). Per-point normalization is by the clean activation's own squared scale. Logged per point under `loss/<term>/hidden_acts_reconstruction/<point>` plus the term aggregate. S14′'s default `adversary_objective: e2e` excludes hidden-activation reconstruction from source ascents only; explicit `term` mode makes the adversary ascend the complete loss. The `hidden_acts_reconstruction` option is TARGET-PASS-ONLY: a tPD non-target term refuses it (T5). **Added 2026-08-05** — this row documents shipped #94 behavior whose code already cited S35; **amended 2026-08-17** for S14′'s source-only objective split. |
| S36 | **ADDED 2026-07-25 (renumbered from S35 when the hidden-acts reconstruction extension took that ID). AMENDED 2026-08-25:** the counted unit is one USE of a write block in an attention nonlinearity, not one block. Partitions are `Neurons`/`QueryHeads`/`KVHeads`: a query head's block is used once; under GQA a kv block is used `n_head/n_kv_head` times (`KVHeads.use_multiplicity`, the only partition where multiplicity exists). Soft count = use_multiplicity × the block-grid soft count; a component uniform over kv heads counts the same as one uniform over query heads. Unit kinds remain {`neuron`, `attention_head`} — q and kv pool into one mean because uses put them on one scale; W&B soft-count keys are named `soft_use_count_*`.** At most one optional `NonlinearityLocalityLossConfig` extends the objective to `c·faith + c·imp + Σ c·recon [+ c·nonlinearity]`. This is a weight-space prior: it makes each rank-1 component's write vector concentrate on fewer nonlinearity-facing neurons or attention heads; it does not inspect activations or nonlinear functions. A site participates only when `SiteSpec.nonlinearity_partition` splits its `d_out` axis into equal contiguous units: one coordinate per MLP neuron (`Neurons`, kind `neuron`), or one output block per attention head (kind `attention_head`) — `QueryHeads(n_head)` for the q projection, `KVHeads(n_kv_head, use_multiplicity)` for k/v. For component `c`, the unit squared-norm fraction is `f_u = ‖U_{c,u}‖² / ‖U_c‖²`, and the soft count is `use_multiplicity · Σ_u f_u / (f_u + t/unit_count)`. The counted unit is one USE of a write block in an attention nonlinearity: a query head's block is consumed by exactly its own attention nonlinearity, while under GQA a kv block is written once but consumed by `n_head / n_kv_head` of them — so `KVHeads` carries `use_multiplicity = n_head / n_kv_head` (with `n_head % n_kv_head == 0` asserted), the ONLY partition where multiplicity is a field; `Neurons` and `QueryHeads` are multiplicity-1 by construction, so a non-unit multiplicity off kv is unrepresentable. The multiplier is exactly the use-grid soft count with each block's energy fraction split evenly across its uses and the threshold taken relative to the use grid. With `m` equally loaded blocks out of `N`, the count is `use_multiplicity · m/(1 + t·m/N)` (one-hot `use_multiplicity/(1+t/N)`, uniform `use_multiplicity · N/(1+t)`), so it is monotone but not a literal support count; a component uniform over kv heads counts `n_head/(1+t)` — the same footing as a component uniform over `n_head` query heads, which is why q and kv pool into the one `attention_head` mean: uses put them on one scale. An exact-zero **write vector `U_c`** contributes zero; a zero rank-1 matrix caused only by `V_c=0` does not. The loss is one component mean per unit kind, combined as `Σ_kind w_kind · mean[kind]` with the strictly-positive authored weights `unit_kind_coefficients`, which must name exactly the unit kinds the target's partitions declare (asserted at step build; no default weights). A `None` weight excludes its kind from the objective outright — no reduction is built and its `loss/<name>_<kind>` metric is absent, deliberately distinct from weight zero — and at least one kind must be trained. `relative_threshold` supplies `t`; every schedule knot is strictly positive. The term is CI-ungated. The standing nonlinearity eval reports the same soft count at fixed `t=4` and the L1 effective use count per subcomponent `use_multiplicity · (Σ_u ‖U_{c,u}‖)² / Σ_u ‖U_{c,u}‖²`. It averages each statistic over all components and over the corpus-level stratum `{c : mean_x lower_CI(x,c) > 0}`, both per site and pooled across sites of the same unit kind, where `x` spans the diagnostic's finite evaluation sample (LM slow-eval batches or a toy's single-feature probe). The zero cutoff is fixed for cross-run comparability and is deliberately independent of `CI_L0.ci_alive_threshold`: CI L0 applies its authored cutoff per token before averaging, while this stratum averages CI before applying its cutoff. W&B keys expose every fixed cutoff: the soft-count statistic is named `soft_use_count_relative_threshold_4` (renamed from `soft_unit_count_*` when the unit became a use — stored data must not mix the two semantics under one key), selected-stratum paths contain `mean_ci_gt_0`, and all keys live under `eval/nonlinearity/sites` or `eval/nonlinearity/aggregates`, so stored data cannot silently mix thresholds, reduction order, or site-level and pooled values. |

## 6. Variation points

| point | valid instantiations | production |
|---|---|---|
| `RECON_TERMS` | any static tuple of coefficiented terms, each one `(SAMPLE_ROUTING, MASK_SOURCE)` family over all sites; built from the shared loss configs by `objective.build_objective` | ★ stochastic subset term + persistent-PGD term |
| `MASK_SOURCE` | `stochastic` (fresh U[0,1]/Bernoulli per draw) · `constant(v)` (`v=0` CI-masked, `v=1` unmasked; no delta path) · `fresh_pgd(init, n_steps, step_size, source_shape)` · `persistent(state_key)` · `mixed_persistent_stochastic(state_key, adv_fraction)` (per-BATCH-ELEMENT Bernoulli family selection — persistent bundle routed all-live vs fresh U[0,1] per the term's routing; contract in S34) | ★ stochastic + persistent |
| `SAMPLE_ROUTING` | `(key, [B,T]) → tuple of routing draws`, statically sized; draws may be jointly sampled (independent repeats, antithetic/complementary subsets, per-step covers) | ★ uniform_k_routing(·, 1) |
| `SRC_STEP` | `adam(β₁,β₂,ε)` with bias correction; `sign` (`sources += lr·sign(grad)`) | ★ adam(.5, .99, 1e-8) |
| `PROJ` / `EFFECTIVE` | **clamp**: PROJ = clamp[0,1], EFFECTIVE = identity, init U[0,1] · **sigmoid**: PROJ = identity (unbounded raw), EFFECTIVE = sigmoid, init N(0,1) | ★ clamp |
| `SCOPE` | `source_shape` (one vocabulary, AMENDED 2026-07-22) — persistent: `c (1,1)` · `bc (B,1)` · `sc (1,T)` · `bsc (B,T)` on a positioned target, `c (1)` · `bc (B)` positionless (`sc`/`bsc` raise there); `nsc (n,T), n|B` unimplemented — fresh-PGD: `c` · `bc` · `bsc`, rejects `sc`. Legacy spellings (`scope:` objects, `mask_scope`) alias at validation | ★ persistent sc |
| stoch sampling | `continuous U[0,1]` · `binomial {0,1}` | ★ continuous |
| delta component | on (`C+1` channels) · off (no delta path, no delta mask) | ★ on |
| CI squashing | `leaky_hard` pair (§4.2); other registry sigmoids exist in torch but are out of spec scope | ★ leaky_hard |

A variant choice must hold every invariant not explicitly parameterized by it.

## 7. Numerics (N) and randomness (R)

| id | rule |
|---|---|
| N1 | Components and CI-fn master params fp32; both AdamW moment sets fp32; SRC_STEP moments fp32. Forward compute may be bf16; the frozen target may be stored bf16. **AMENDED 2026-08-04 (uniform CI compute):** `evaluate_ci` casts both the fp32-master CI parameters and captured taps to `COMPUTE_DT` before every CI call. This intentionally replaces the prior caller-dependent promotion: pre-amendment toy and fp32-target checkpoints remain loadable but are not trajectory-identical on resume, and their CI metrics and training trajectories may change because of bf16 input quantization; bf16-target LM runs are unchanged. The persistent-source Adam (`SRC_STEP = adam`) places eps AFTER the sqrt, inside the denom: `denom = sqrt(v_hat) + eps` (not `sqrt(v_hat + eps)`) — a classic eps-before-vs-after-sqrt parity trap. Refs: torch `persistent_pgd_state.py:125` (`v_hat.sqrt().add_(eps)`), jax `adversary.py::sources_adam_ascend_project`. |
| N2 | Faithfulness deltas `W − V@U` are computed in fp32 outside any autocast; the sum-of-squares is fp32. (The delta on the masked-forward PATH may be bf16-computed — a documented bf16-rounding divergence from torch, which forms it in fp32 and casts at use.) |
| N3 | `kl_per_position` (softmaxes + KL sum) and the imp-min reduction are fp32; the smooth-L0 per-value penalty computes on `ci` cast to fp32 first (`losses.py::_per_component_frequencies`). Loss scalars and gradient accumulation fp32. **eval cast point (accepted seam):** `CEandKLLosses` — torch computes the eval softmax under bf16 autocast (`param_decomp/eval_metrics/ce_and_kl_losses.py:91-176`), JAX casts to fp32 before `log_softmax` (`eval.py:31-40,110-166`) — a bf16-vs-fp32 input cast-point divergence, not a direction/reduction divergence; the expected `kl_<v>` delta on a fixed batch is the bf16-rounding floor (rel ≪ 1e-2), accepted rather than bounded by a fixture (cf. E17). **AMENDED 2026-08-27:** the former torch-vs-JAX imp-min cast seam — torch's bf16 `(ci + eps)**p` intermediate vs JAX's fp32-first cast — retired with the `L_p` penalty; its bounding fixture (`test_imp_min_bf16_seam.py`) is deleted. |
| R1 | Every stochastic draw (mask sources, routing, source init) is independent across sites, positions, forwards, steps — distributions as stated. |
| R2 | RNG stream order/bits need not match torch. |
| R3 | Draws over a sharded batch are independent across ranks (distinct streams). |

## 8. Data-parallel contract (D)

§4 never mentions sharding because it doesn't have to: every loss and gradient there
is global-batch math. This section is the whole answer to "now shard it":

| id | rule |
|---|---|
| D1 | Per-shard means + averaged grads must compose to §4's global-batch values for faith/stoch/ppgd (uniform shards). For the *batch-1 source* gradient this means: AVG the per-replica source-grads (each is ∂(local-shard mean)/∂sources) — torch's `reduce_source_grads`; under GSPMD it falls out of autodiff of the global-mean loss. Getting this reduction wrong (e.g. SUM over independent per-position sources) was a real historical bug. |
| D2 | Imp-min requires the exact global per-component sums *inside* the `log2` (S8) — the one term where mean-of-shard-results ≠ global result (Jensen). The reduction must also be autograd-aware so gradient reaches each shard's CI values. The eval-path imp-min reuses the one `importance_minimality_terms` impl (there is no separate non-autograd eval reduce as in torch), so the value is device-count invariant by D4; guarded directly at 1 vs N devices by `param_decomp/tests/core/test_imp_min_global_reduction.py`. |
| D3 | Shared PPGD sources stay replica-identical per S16: identical init (broadcast or identical seeding), updates computed from the D1 gradient, identical optimizer steps. |
| D4 | Validation property: with global batch + seed fixed, the metric trajectory is invariant to device count up to floating-point reassociation (cross-shard reduction order; observed rel ≤ ~1e-5 on the tiny-target harness, `param_decomp/targets/invariance_check.py`). JAX's counter-based RNG makes even the stochastic draws identical across layouts. **AMENDED 2026-07-15** (owner-partitioned persistence): the trainable V/U masters (and their optimizer moments) persist as same-shape STACKS — one `(Vs [g, d_in, C], Us [g, C, d_out])` pair per shape group (`components.ComponentStacks.stacks` + static `site_slots`), placed by the hybrid HSDP rule: stack axis ÷`replicate` (whole matrices owned per node-group — zero cross-node weight collectives per step; muon NS node-local), matrix d dims ÷`fsdp`, C ÷`tp` (total ÷N, same memory as the retired per-site intra-matrix ZeRO-1); a group whose stack length does not tile `replicate` falls back per-group to intra-matrix data sharding behind the stack axis. Same math, different layout — covered by THIS invariant's reassociation tolerance (re-validated on the harness at 4 sim devices). Checkpoint trees change shape (2·n_shapes stack leaves, not 2·n_sites): pre-change checkpoints need a one-off layout migration to restore at tip. Muon leaf labeling: the V/U tree is now all-3D, so both groups use `stacked_muon_dimension_numbers` (optax's default 2D rule would silently Adam every V/U leaf). **AMENDED 2026-07-21** (fallback opt-in): the non-tiling-group fallback is config-OPT-IN, not automatic — enabled by the `owner+zero1` preset (or an explicit table's declared `components.optimizer_state_fallback` row); strict `owner` errors on a group whose stack length does not tile `replicate`. **AMENDED 2026-07-21** (bidirectional config claim): the sharding preset/table is a BIDIRECTIONAL claim, checked at config build — the per-group persist-vs-zero1 assignment is resolved once, at `PlacementRules` construction from the run's resolved site set and its config-implied mesh; a declared `components.optimizer_state_fallback` row that no shape group takes is equally an error, and the trainer boundary only validates the received assignment against the arrays it holds — it never re-decides. **AMENDED 2026-08-18** (strict placement): the per-group fallback is REMOVED entirely — the `owner+zero1` preset and the explicit-table fallback rows (`optimizer_state_fallback`, `faithfulness_weights_fallback`, `faithfulness_deltas_fallback`) no longer exist, so mixed per-group placement is unrepresentable and the bidirectional claim is vacuous (nothing left to claim). One set of component rows places every shape group; a group whose stack length does not tile a stack-sharded row REFUSES at config build, the refusal naming the non-tiling groups, their stack lengths, the sharded extent, and the remedies (a tiling mesh, or a placement with no stack sharding). `zero1`'s faithfulness rows are now its intra-matrix master layout for ALL groups (the weights transition is the identity; previously tiling groups took the stack-preferring pair) — same math, layout covered by this invariant's reassociation tolerance. |

**D4 amendment 2026-08-11:** the target, not numeric shape equality, declares each
shape-homogeneous persistence group. LM targets group by matrix kind, so a full-kind
layer stack is already the scan input leaf; placement remains a separate rule over its
semantic `stack`, `d_in`, `d_out`, and `C` axes. This supersedes D4's earlier
same-numeric-shape grouping key without changing its optimizer or collective semantics.

---

## 9. Non-normative: torch ground-truth pointers & rationale

All pointers below resolve to the on-branch single-pool torch core (`param_decomp/`) at
`feature/jax`. The one exception is the stochastic recon term: its runnable torch Metric
is the chunked ancestor `ChunkwiseSubsetReconLoss`, off-branch on the
`feature/fsdp-lm-trainer` n-pool lineage
(`param_decomp/metrics/chunkwise_subset_recon.py`) — there is no on-branch Metric
counterpart, but the torch↔JAX parity fixtures (`param_decomp/tests/targets/equivalence/`) pin its math
(R-1). The KL primitive it composes (`recon_loss_kl`) and the recon-term ancestry are
on-branch (rows below).

| spec | torch source |
|---|---|
| §4.1 site/forward, routing | `param_decomp/components.py` (`LinearComponents.forward`), `param_decomp/masks.py` |
| §4.2 squashings | `param_decomp/ci_sigmoids.py` (`LowerLeakyHardSigmoidFunction`, `upper_leaky_hard_sigmoid`) |
| §4.3 faith / imp / KL | `param_decomp/metrics/faithfulness.py` (`faithfulness_loss`) · `param_decomp/metrics/importance_minimality.py` · `param_decomp/batch_and_loss_fns.py::recon_loss_kl` |
| §4.3 recon term (stochastic) | OFF-BRANCH (chunked ancestor): `param_decomp/metrics/chunkwise_subset_recon.py` (`ChunkwiseSubsetReconLoss`) on `feature/fsdp-lm-trainer`; pinned by `param_decomp/tests/targets/equivalence/` fixtures, no on-branch Metric |
| §4.4–4.5 adversary, ordering | `param_decomp/metrics/persistent_pgd_state.py` (init/warmup/step/scopes; source Adam denom `v_hat.sqrt().add_(eps)` @ `:125` — N1), `param_decomp/metrics/persistent_pgd_recon.py` (`before_backward`/`after_backward`), `param_decomp/metrics/pgd_utils.py`, `param_decomp/train_step.py::run_loss_step` (hook order), `param_decomp/optimize.py` (clip → step) |
| §4.5 warmup, schedules | `param_decomp/faithfulness_warmup.py`, `param_decomp_config/schedule.py::get_scheduled_value` |
| §4.6 CI arch (torch oracle, RETIRED for arch) | `param_decomp/ci_fns.py::GlobalSharedTransformerCiFn`, `param_decomp/ci_nn_blocks.py` — the torch global-concat CI fn, no longer bit-faithful to the pinned JAX arch (§4.6 AMENDED 2026-06-22); JAX arch is native (`ci_fn.py::ChunkwiseTransformerCIFn`) |

The JAX implementation (`train.py`) realizes these semantics with
`clean_forward` / `masked_forward`, target-owned capture-key lowering, `masks_from_sources`, the
coefficiented `ReconLossTerm` terms, and
`sources_adam_ascend_project`. Pseudocode names describe semantics, not a required method
grid.

Rationale worth keeping: the two squashings give each consumer gradient only in its
permitted direction (masks may push CI up out of saturation; the sparsity penalty may
not push it below 0). The adversary is *persistent* because re-finding the worst-case
ablation from scratch each step under-trains the adversary at any affordable inner-step
count. The `freq` term is the description-length / frequency penalty (`L_freq` in the VPD
paper); normalizing its `log2` by an explicit `a' = reference_datapoint_count` (rather than the
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
  cadence `eval.every` as independent operations scheduled by the core loop. This is the torch `EvalLoop` analog
  restricted to scalars (CE/KL, CI-L0, the fresh-PGD probe, attn-patterns).
- **SLOW / plot tier — in-loop ONLY (S28/S28a).** On cadence `eval.slow_every` (a multiple
  of `every`, so it lands on a fast-eval step and reuses that step's in-memory eval batches)
  the slow tier renders the plot figures, forward-only,
  next to the fast pass. The collective forward + device→host pull run on all ranks; the
  pure matplotlib render runs on a rank-0 background thread (`BackgroundRenderer`) off the
  loop, then the shared `MetricsSink` serializes the deferred W&B write. Deferred figures use
  `slow_eval/figure_step` as their semantic eval-step axis. There is NO offline/retrospective slow-eval CLI — `slow_eval.py`
  is a library only. `UVPlots` is a config-gated figure metric usable for any decomposition:
  cheap for the toys, a naive V/U host gather for the LM in-loop tier that breaks at
  production C by design (S28).
- **Cadence coupling.** `log_every` divides `eval.every` (asserted in `train`), and
  `eval.every` divides `eval.slow_every` (asserted in `train`), so every eval step is a
  train-log step and every slow step is an eval step.
- **Cast-point seam.** `CEandKLLosses` carries the bf16-input cast-point asymmetry recorded
  in N3 (torch softmax under bf16 autocast; JAX fp32 before `log_softmax`), an accepted seam.

---

## 11. Targeted parameter decomposition (tPD)

tPD decomposes a target model's mechanism for a NARROW behavior while holding the rest of
the model harmless. It is plain VPD plus a second data stream and one new device: two
distributions — the **target** stream (the behavior whose mechanism we want in
components) and the **non-target** stream (the broad distribution we intend only not to
disturb) — and a **weight delta as the off-target escape valve**: over the non-target
stream every delta mask is pinned fully on, so `components + Δ` reproduce the frozen
model there and the components are NOT needed to carry off-target behavior. Free
components then sort themselves into target-mechanism vs general. The delta is never
penalized: faithfulness pressure drives `Δ → 0`, and a zero delta has no valve — so a
targeted objective has **no faithfulness role at all** (not a coeff-0 term; the role does
not exist).

The step is two passes summed into ONE backward. The TARGET pass is the full
decomposition objective (recon grid, adversary ascents, imp-min) on the target stream.
The NON-TARGET pass is exactly three things — delta-pinned reconstruction against the
broad stream's frozen output, the unmasked-no-delta term (T4's one delta-off exception),
and importance-minimality at its own coefficient. Nothing else has a job off-target:
with the delta pinned on, adversarial probing and internal-activation matching would
constrain the very behavior tPD declines to decompose. The non-target surface is
**authored directly** (`configs.NontargetConfig`), never derived from the target pass's
loss list — there is no keep/drop filtering anywhere, and the types the non-target
schema admits are closed (`NontargetReconLossMetricConfig`: the stochastic/
constant-source recon types plus `UnmaskedNoDeltaReconLoss` only).

Engine surface: `run_targeted_decomposition_training` /
`train.make_targeted_train_step` — siblings of the plain entry/factory over the same
step vocabulary, and a separate composition root per domain (a targeted run is a
different top-level config shape, not a mode flag on a plain one).

| id | invariant |
|---|---|
| T1 | One optimizer step = `∇(L_target + L_nontarget)` from a single `value_and_grad`; both passes see the same pre-update parameters, the same live CI fn, and the same annealed schedule parameters. The CI fn's gradient is the sum of the two streams' pullbacks. |
| T2 | Two independent streams, each an ordinary `sample_batch(step)` seam. The TARGET stream's global batch is `pd.batch_size` and its geometry is the run's `positions` — everything core sizes off those (persistent sources, S16/D1 shapes) is target-pass state. The non-target stream's global batch is `nontarget.batch_size`, its sequence length the broad dataset's own. |
| T3 | No faithfulness role exists in a targeted objective, and none is spellable: `TargetedPDConfig`'s loss union has no `FaithfulnessLoss` member and the shape carries no `faithfulness_warmup_*` fields at all — refusal is a parse error, with one terse library-boundary assert behind it for programmatically-built loss lists. The weight delta is never penalized, by construction. |
| T4 | Every non-target forward of a delta-pinned term runs with every weight-delta mask pinned to `1.0`. Component masks compose per S1 from the NON-TARGET batch's own CI envelope; constant-source terms (which carry no delta path in the plain objective) carry a pinned-on delta here. ONE enumerated exception: an `UnmaskedNoDeltaReconLoss` term runs every component mask at `1.0` and every weight-delta mask at `0.0` — the paper's unmasked reconstruction term for CSS-only decompositions (all `m_i = 1`, `m_Δ = 0`; Method details), which makes the FULL component sum alone reconstruct so that dead components that never activate cannot interfere with the reconstruction. The polarity rides its own strategy type (`UnmaskedNoDeltaSources`), never a flag on the pinned arms. **Amended 2026-08-06**: was "every non-target forward"; the unmasked-no-delta arm is the one delta-off exception. |
| T5 | The non-target surface is closed: delta-pinned stochastic/constant recon (KL against the broad stream's frozen clean output, scored full-length), the unmasked-no-delta term (T4's one delta-off exception), and imp-min — nothing else. Adversarial and mixed sources are unrepresentable in `NontargetReconLossMetricConfig`; the S35 hidden-acts rider is refused on non-target terms; and the narrowing is carried IN THE TYPE — `NontargetPass.recon` is `ReconLossTerm[StochasticSources | ConstantSources | UnmaskedNoDeltaSources]`, so the non-target grid's dispatch is exhaustive over exactly the three enumerated arms, with one terse factory assert as the library boundary for objectives built past the type. **Amended 2026-08-06**: admitted `UnmaskedNoDeltaReconLoss` (T4's exception) into the closed set. |
| T6 | Both passes share ONE importance-minimality config — penalty kind, anneal schedule, frequency block — and the same annealed parameter at every step, by construction (`build_targeted_objective` reuses the target term's config object). Only the coefficients differ: the target pass's is the authored term's, the non-target pass's is `nontarget.impmin_coeff` (absolute, not a ratio). |
| T7 | Adversaries (persistent and fresh) exist in the TARGET pass only: warmup ascents (S24), final ascents (S14′), and source sizing all run against the target stream. Non-target masks never read adversarial sources. |
| T8 | The target stream is UNPADDED BY CONSTRUCTION: every prompt in the pool must tokenize to ONE shared length (asserted when the pool is built), and the target pass runs at that natural geometry. No pad position exists, so every objective — the recon grid's comparisons, the adversary ascents, importance-minimality, and the CI fn's (bidirectional) attention — covers every target position with nothing to exclude. Each stream keeps its own sequence length (T2); scoring is full-length on both passes. **AMENDED 2026-08-06**: supersedes the windowed formulation (`TargetScoring` prefix slicing of scored observations + pad-masked CI attention + prefix-windowed imp-min), which existed only to make END-padding to the broad stream's seq_len unreachable; the padding itself is gone, per the tPD paper's own per-stream geometry, and the window machinery was deleted with it. |
| T9 | RNG: one step key. The target grid's per-term keys derive at offset 1 — byte-identical to the plain step's derivation (R1) — and the non-target grid's offset past them (`1 + n_target_terms + i`), so the two grids' draws are disjoint. |
| T10 | The TARGET pass treats the delta exactly as plain VPD does: stochastic terms draw a `U[0,1]` delta source, adversarial terms carry their source's delta channel (S1). **OPEN** — the algorithm's author-definition may instead pin the delta OFF on-target ("components alone reconstruct the target behavior"); every implementation to date, and every recorded run, uses the stochastic form. Resolving this changes only the target pass's mask derivation. |
| T11 | CI-scaled weight decay on subcomponent vectors (`TargetedPDConfig.ci_scaled_weight_decay`, optional — absent means no decay, and absence is byte-inert: the step is the pre-T11 step). AFTER each optimizer step, every subcomponent's V column and U row are scaled by `keep_c = 1 − lr(step)·wd·(1 − max_c)`: `wd` the authored coefficient, `lr(step)` the components optimizer's CURRENT scheduled LR (AdamW's decoupled-decay convention), `max_c` that subcomponent's max CI over BOTH streams' batches — every leading axis, batch and positions alike; a component important on either stream is not dead. The CI read is the step's own pre-update forward's `lower` (no clamp needed: `lower ≡ clip(upper, 0, 1)` pointwise per S6, so the two squashings agree on the statistic). This is an update rule, not a loss term: nothing differentiates through it, and the CI fn, imp-min, and the adversaries never see it. Targeted-only by design: in plain PD faithfulness penalizes `‖W − VU‖`, so shrinking a dead component's V/U GROWS the residual and the decay would fight faithfulness head-on; in tPD the delta is the free escape valve (T3), so the never-important components' stale vectors — which imp-min cannot shrink (it only pushes their CI down) — get dragged to zero. Ref: the tPD paper's "weight decay of 0.1 on subcomponent vectors, where the weight decay is scaled by 1 − max_batch(CI)". |

Ref: the paper *Targeted Recovery of Weight-Space Mechanisms From Neural Networks*
(Antoine's targeted decomposition; standup 2026-07-14 elevated it to first-class).
Open items beyond T10: whether T3's faithfulness exclusion is definitional or a
regime choice awaits the same author-definition; it is currently enforced structurally
(the objective type has no faithfulness field) with the config-level refusal as the
boundary message.
