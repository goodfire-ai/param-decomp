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
| **clean forward** | suffix forward with every site on its frozen `x @ W` path | the decomposed forward with all masks = 1 (≈ equal only in exact arithmetic) |
| **source** | a `[0,1]` value per channel that a mask is built *from* (stochastic or adversarial) | the mask itself |
| **mask** | `ci + (1−ci)·source`, what the forward consumes | the source; the CI value |
| **delta component** | the `(C+1)`-th maskable channel carrying `x @ (W − V@U)` | the faithfulness loss (same residual, different role) |
| **CI** (causal importance) | the CI fn's per-component prediction in `[0,1]`; `ci=1` ⇒ mask pinned 1 (protected) | a probability; an attribution score |
| **lower / upper leaky** | the two squashings of the SAME CI logits; lower → masks, upper → imp-min | two CI functions (there is one) |
| **faithfulness warmup** | pre-loop phase: V/U trained on `L_faith` alone | the other two "warmups" → |
| **PPGD warmup (`n_warmup`)** | supplemental source-ascent iterations inside each step | LR-schedule warmup; faithfulness warmup |
| **LR warmup (`warmup_pct`)** | linear ramp 0 → start of a schedule's value | the above two |
| **persistent** (PGD) | sources + their optimizer moments survive across training steps | re-initialized-per-step PGD (the eval-only `PGDReconLoss`) |
| **residual-start** | suffix forward from a harvested residual; prefix runs once per batch, never in the graph | KV-caching; activation checkpointing |

---

## 2. Production constants

| | |
|---|---|
| target | Llama-3.1-8B, frozen, bf16 storage |
| decomposed | `layers[18..18].mlp.{gate,up,down}_proj` (bench: `20..31`), right-mult `W: [d_in, d_out]` |
| C | 24576 (bench: 8192); `use_delta_component: true` → source channel dim `C+1` |
| data | fineweb, seq `T=2048`, global batch `B=512`, fresh batch per step |
| coeffs | faith `1e5` · imp `5e-6` · stoch `0.5` · ppgd `0.5` |
| imp-min | `beta 0.2`, `eps 1e-12`, `p: 2.0 → 0.4` linear over `[0, 1]`-frac of training |
| stoch | plan = sequential 3-site chunks × `uniform_k_routing(chunk, n_draws=1)` |
| PPGD | scope `broadcast_across_batch`, `n_warmup 2`, clamp-parameterization, Adam(β₁ .5, β₂ .99, ε 1e-8), lr const `0.01` w/ 2.5% LR-warmup |
| components opt | AdamW(.9, .999, ε 1e-8, wd 0), lr `1.5e-4` cosine → `0.1×`, **grad-clip 0.01** |
| CI opt | AdamW(.9, .999, ε 1e-8, wd 0), lr `5e-5` cosine → `0.1×`, no clip |
| faith warmup | 400 steps, AdamW lr `1e-3`, wd 0 |
| CI fn | shared transformer: `d_model 4096`, 4 blocks, 64 heads, mlp `[16384]`, RoPE base 10000, bidirectional; `sigmoid_type leaky_hard` |
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

def masked_forward(residual[B,T,d], live_sites, masks, delta_masks, routes) -> logits[B,T,vocab]:
    suffix forward (layers first..n−1, final norm, LM head);
    each site s ∈ live_sites computes site_out(x, s, masks[s], delta_masks[s], routes[s]);
    each site s ∉ live_sites computes x @ W_s            # frozen path, NOT y_dec(mask=1)   (S2)

def clean_logits(residual)   = masked_forward(residual, live_sites=∅)                       (S3)
def site_inputs(residual)    = the activation entering each site's weight on the clean path
    # MLP sites: gate_in = up_in = post-ln2 residual; down_in = silu(gate)·up               (S4)
    # attn sites: q_in = k_in = v_in = post-ln1 residual; o_in = pre-o_proj attn output
```

### 4.2 CI

```
def ci(ci_fn, site_inputs) -> (ci_lower, ci_upper):      # each {site: [B,T,C]}
    logits = CI_TRANSFORMER(ci_fn, site_inputs)          # architecture pinned in §4.6
    return lower_leaky_hard(logits), upper_leaky_hard(logits)                               (S5)

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

def kl_per_position(masked_logits, clean_logits) =
    Σ_{b,t} KL(softmax(clean_logits[b,t]) ‖ softmax(masked_logits[b,t])) / (B·T)    # fp32 (N3)

def faithfulness_loss(components) = ( Σ_s ‖W_s − V_s@U_s‖_F² ) / ( Σ_s numel(W_s) )        (S17)

def importance_minimality_loss(ci_upper, pnorm):         # per-site grouping                (S7)
    for s: per_component_sums[c] = Σ_{b,t} (ci_upper_s[b,t,c] + eps) ** pnorm           (S8,S9)
    return Σ_s Σ_c (per_component_sums[c]/(B·T)) · (1 + beta · log2(1 + per_component_sums[c]))

# RECON_PLAN: a static list of entries (live_sites, SAMPLE_ROUTING); each entry's sampler
# returns a statically-sized FAMILY of routing draws, each draw = one forward (§6).
def stochastic_recon_loss(components, ci_lower, residual, clean_logits):
    total, n_forwards = 0, 0
    for (live_sites, SAMPLE_ROUTING) in RECON_PLAN:
        for routes in SAMPLE_ROUTING(key, [B,T]):        # fresh per step                 (R1,S11)
            masks, delta_masks = make_masks(ci_lower_s, source_s ~ U[0,1]^[B,T,C+1])  ∀ s ∈ live_sites
            total += kl_per_position(masked_forward(residual, live_sites, masks, delta_masks, routes),
                                     clean_logits)
            n_forwards += 1
    return total / n_forwards                                                               (S10)

def adversarial_recon_loss(components, ci_lower, sources, residual, clean_logits):     # all sites (S12)
    masks, delta_masks = make_masks(ci_lower_s, expand(SCOPE, sources[s]))    ∀ s ∈ sites
    return kl_per_position(masked_forward(residual, ALL_SITES, masks, delta_masks, ALL),
                           clean_logits)
```

### 4.4 The adversary

```
def sources_update(sources, sources_grad, opt_state):
    sources, opt_state = SRC_STEP(sources, +sources_grad, opt_state)   # ASCENT on L_ppgd
    return PROJ(sources), opt_state                                                         (S15)
```

### 4.5 One training step

```
def train_step(state, batch, step):
    residual = sg[ prefix_forward(batch) ]           # per fresh batch                      (S18)
    cln      = sg[ clean_logits(residual) ]                                                  (S3)
    ci_lower, ci_upper = ci(ci_fn, site_inputs(residual))   # ONE conceptual CI eval/step;
                                                            # recompute allowed (deterministic)
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

### 4.6 CI transformer architecture (pinned)

```
in:   {site: x_s [B,T,d_in_s]}  (clean site inputs, fixed site order)
1.    h_s = rms_norm(x_s)                      # weightless (no learnable scale)
2.    h   = concat_s(h_s) @ W_in + b_in        # → [B,T,d_model]; NO nonlinearity here
3.    × n_blocks (pre-norm):
        h += attn(rms_norm_weightless(h))      # bidirectional MHA, rotate-half RoPE
                                               # base 10000; q/k/v/out bias-FREE
                                               # RoPE inv_freq is a stop-gradient buffer (S27)
        h += mlp(rms_norm_weightless(h))       # Linear(d→16384)+b → GELU → Linear(→d)+b
4.    logits = h @ W_out + b_out               # → [B,T, Σ_s C_s], split per site in order
init: biases zero; weights fan-in scaled (torch init_param_)
```

---

## 5. Semantic invariants

| id | invariant |
|---|---|
| S1 | `mask = ci + (1−ci)·source` per component channel; the delta channel is the raw source value, never ci-interpolated; CI has no delta output. |
| S2 | A site not live in a forward runs the frozen `x @ W_s` path — zero V/U gradient, zero decomposition rounding, and none of the live path's ~9× compute or `(B,T,C)` activations. |
| S3 | The recon target `clean_logits` is the frozen-path forward, stop-gradient. Never the `mask=1, delta=1` decomposed identity (differs in bf16 and pollutes the graph). Under residual-start (S18), `clean_logits` is the SUFFIX-only clean forward over the harvested residual (`clean_suffix_logits`); this equals the whole-model frozen forward because the frozen prefix is stop-gradient'd and runs once outside the graph — the two forms are identical under sg of the frozen prefix. The on-branch torch single-pool reference computes the whole-model frozen forward; JAX computes the suffix-only form; they agree by this equivalence. |
| S4 | CI inputs are the clean site inputs from the frozen path of the same batch. |
| S5 | `ci_lower` and `ci_upper` are two squashings of the SAME logits. `ci_lower` feeds every mask; `ci_upper` feeds imp-min only; no other crossing. |
| S6 | The squashings' forward/backward are exactly §4.2 — including `lower_leaky_hard`'s grad-sign-gated lower leak (a custom VJP, not autodiff of the forward). The backward is a nested `where` with `<=` boundaries: on `0 < x <= 1` pass `g`; at `x <= 0` pass `α·g` ONLY where `g < 0`, else `0`; at `x > 1` pass `0`; `α=0.01`. The boundary tie at `x=0` resolves to the LOWER (`x <= 0`) branch — this `<=` placement is exactly what makes torch (`ci_sigmoids.py`) and JAX (`ci_fn.py`, `_lhs_b`) bit-identical and is load-bearing, not incidental. Grad-checked by `tests/test_lower_leaky_hard_grad.py` (`g<0` vs `g>0` at `x<0`, `x∈(0,1]`, `x>1`; #789). |
| S7 | Imp-min groups per site: the `log2(1+sum)` consumes one site's per-component sum. Merging sites/layers into one group is incorrect (convexity). |
| S8 | The per-component sums are over the **global batch**, accumulated before the `log2`. (Per-shard results combined after the log are incorrect — Jensen; see D2.) |
| S9 | `pnorm(step)` anneals linearly `2.0 → 0.4` over the configured frac window; `eps` sits inside the power. **JAX narrowing:** annealing is REQUIRED — `annealed_pnorm` asserts `cfg.p_anneal_final_p is not None` (`losses.py:53`) and `train.py:122` asserts it too. Torch supports a constant-p config (`importance_minimality.py:16-37` returns `initial_p` when no annealing window). Constant-p in JAX is expressed by setting `p_anneal_final_p == pnorm` (a flat schedule); any other torch constant-p config is REFUSED (fail-fast assert), never silently approximated. |
| S10′ | The recon objective is a static tuple of coefficiented loss TERMS (one per configured recon loss metric, in config order). Each term is a static plan of `(live_sites, SAMPLE_ROUTING, MASK_SOURCE)` entries; the term's loss = mean over ALL its forwards (every draw of every entry) of `kl_per_position`; the total adds `coeff · term` per term. Plan structures (live-sets, sampler identities, family sizes, strategy kinds) are fixed across steps. The §4 pseudocode shows the production two-term instantiation (`stochastic_recon_loss` + `adversarial_recon_loss`). Recon KL direction is pinned by S25; the mean-over-forwards ≡ accumulator identity by S26. |
| S11 | `uniform_k_routing`, per position: `k ~ U{1..|live_sites|}` then a uniform `k`-subset of the live sites routes True; non-live sites are not live at all. Routing draws are fresh per step, sampled inside the step. |
| S12′ | An adversarial term's loss forward consumes its sources as LEAVES (no ascent-graph history); gradient flows to components and (through `ci_lower`) to the CI fn — and, for persistent sources, to the leaves themselves (S14′). The PRODUCTION adversarial term masks ALL sites and routes everywhere; subset-routed adversarial terms route per their plan. |
| S13′ | Per persistent term: source updates per training step = `n_warmup + 1`, all through THAT term's persistent SRC_STEP optimizer state; its source LR schedule advances once per training step. **`warmup_pct==0` edge (accepted seam):** at `warmup_pct==0` torch short-circuits to full LR at step 0 (`warmup_steps=0`), while JAX clamps `warmup_steps = max(floor(...), 1)` → source LR `=0` at step 0 (`losses.py:61`, `train.py:265`). A one-step divergence, only when `warmup_pct==0`; production uses 2.5% warmup and is unaffected. Accepted, not matched. |
| S14′ | Each persistent term's final ascent gradient comes from the SAME graph as the main backward (pre-update components, live `ci_lower`), unscaled by THAT term's coeff. It is applied after backward; it must not use post-update params. |
| S15 | Every source update ends with `PROJ` (★ clamp to `[0,1]`). Init: ★ `sources ~ U[0,1]` i.i.d. |
| S16 | Shared-scope sources are identical on every data-parallel replica at every step (identical init, identical updates from the global-batch gradient). `per_batch_per_position` sources shard with the batch instead. (Implementation mapping in §8/§9.) |
| S17 | `faithfulness_loss` is the global mean of squared delta entries over all sites' parameters (Σ‖Δ‖² / Σ numel), recomputed from live V/U each step. |
| S18 | Each training step consumes a fresh data batch; the prefix harvest runs per batch, outside every gradient graph. |
| S19 | Components gradients are global-norm-clipped at `0.01`, before the optimizer step. CI fn is unclipped (production). The clip coefficient uses torch's eps convention: `clip_coef = min(1, max_norm / (total_norm + 1e-6))` (`torch.nn.utils.clip_grad_norm_`). This `+1e-6` is canonical and matched JAX-side (#643); plain `optax.clip_by_global_norm` divides by `max(total_norm, max_norm)` with NO eps, which at `clip=0.01` (clip fires almost every step) gives a ~1e-4 relative component-grad difference each step — a real per-step deviation the JAX clip must avoid by reproducing the `+1e-6`. |
| S20 | Both main optimizers are AdamW, `wd=0`, betas `(0.9, 0.999)`, eps `1e-8`; LR cosine to `0.1×` start, no warmup, stepped per training step. Cosine convention is canonical-torch: `progress = (step − warmup) / (decay_steps − 1)`, so LR reaches `0.1×` at `step = steps − 1` (`schedule.py`). This is matched JAX-side (#642); plain `optax.cosine_decay_schedule(lr, steps, alpha=0.1)` divides by `steps` and reaches `0.1×` one update LATER (at `count = steps`), a genuine formula divergence of ~O(1/steps) ≈ 2.5e-6/step at 400k that the JAX schedule must avoid by using the `decay_steps − 1` denominator. |
| S21 | Faithfulness warmup (400 × AdamW lr `1e-3` on `faithfulness_loss` alone) precedes step 0; its optimizer is discarded. |
| S22 | Checkpoints round-trip ALL trajectory state of §3 — including every persistent term's sources + SRC_STEP moments + step/schedule counters — such that a resumed run continues the same trajectory (modulo RNG streams and kernel nondeterminism, cf. D4). |
| S23 | A persistent source bundle feeds exactly ONE loss term. (The fused-backward S14′ unscaling divides that term's coeff out of the source gradient; a bundle shared across terms would make the division wrong.) |
| S24 | A persistent term's WARMUP ascents forward all sites, routed everywhere — regardless of the term's loss plan (torch parity: `persistent_pgd_state.warmup` hardcodes route-all). A fresh-PGD entry draws its routing ONCE per step, shared by all its ascents and its main loss forward (torch parity: `pgd_masked_recon_loss_update`). |
| S25 | Recon KL direction is `KL(softmax(clean_logits) ‖ softmax(masked_logits))` — `P = clean`, `Q = masked`. Equivalently `Σ p_clean · (log p_clean − log p_masked)`. Torch `recon_loss_kl` realizes this as `F.kl_div(log_softmax(pred=masked), softmax(target=clean), reduction='sum')` (`batch_and_loss_fns.py::recon_loss_kl`); reversing the arguments is a silent, plausible bug, so the direction is semantic, not incidental. |
| S26 | Normalization identity: `(Σ_forwards sum_kl) / (Σ_forwards n_positions) == mean_forwards(sum_kl / n_positions)`, valid **iff** every forward shares the same `(B,T)` (so `n_positions` is constant across forwards). This precondition is what makes JAX's mean-over-forwards (S10′) equal torch's `(Σ sum_kl)/(Σ n_positions)` accumulator; uniform `(B,T)` across all forwards in a term is therefore required. (Stated in LOSS_PARITY_DESIGN §4e; cross-ref S10′.) |
| S27 | The CI transformer's RoPE `inv_freq` (§4.6) is a non-trained buffer and MUST be stop-gradient'd in the CI fn. In JAX it is a pytree leaf and would otherwise be optax-updated, silently drifting the rotary frequencies; torch carries it as a registered buffer (no grad). Ref `ci_fn.py:115`. |
| S28 | Eval splits across two processes (§10): a FAST scalar tier run in-loop on cadence `eval.every`, and a SLOW/plot tier delegated entirely to `pd-offline-eval` on exported checkpoints. The torch `slow_every` / `slow_on_first_step` cadence (`optimize.py`'s `EvalLoop`) has NO JAX in-loop analog — the slow tier is offline, keyed off the checkpoint-save cadence, by design. |
| S29 | JAX `EvalConfig` carries `batch_size`, `every`, `n_steps` (+ `rounding_threshold`, `ci_alive_threshold`) and DELIBERATELY omits `slow_every` / `slow_on_first_step` (those have no in-loop tier — S28). `eval` is atomic-optional (`EvalConfig | None`): `None` disables in-loop eval. |
| S30 | `cfg.cadence.log_every` divides `cfg.eval.every` (`eval.every % log_every == 0`, asserted at `run.py:255`) so every eval step is also a train-log step. |
| S31 | `StochasticHiddenActsReconLoss` is NOT a JAX training loss — `build_recon_terms` refuses it (`recon.py` final `case _`: `raise AssertionError(f"unsupported training loss {cfg.type!r}")`). Its objective is per-element MSE on each target module's OUTPUT activations, which `DecomposedLM`'s four-fn table deliberately does not expose (the table returns final logits only); porting it would need a fifth per-target seam `masked_site_outputs(...) -> dict[site, (B,T,d_out)]` plus a per-element (not per-position) normalization, and as a training loss it is exactly the site-local recon the trainer treats as a conceptual no-no (LOSS_PARITY_DESIGN §4c). It rides the OFFLINE BRIDGE instead: it lives only in `torch_config.OFFLINE_EVAL_METRIC_TYPES`, and `pd-offline-eval` computes it bit-faithfully on exported checkpoints (S28's slow tier). Keep-on-bridge is the decision; the seam is built ONLY if a training-loss use case appears, and then as an explicit amendment to this invariant (LOSS_PARITY_DESIGN §6 stage 4). |
| S32 | A persistent term with `start_frac > 0` contributes nothing — no loss, no source/optimizer update — until `step/total_steps >= start_frac` (torch parity: `persistent_pgd_recon.update()` returns `None` before `start_frac`, the term absent and its state frozen at init). Sources are RNG-initialized at step 0 but untouched while inactive, so activation is distributionally identical to torch's lazy state construction (the init draw is RNG-pure). `start_frac == 0.0` is the always-active common case and keeps the unguarded, byte-exact path (`train.py` only inserts the `term_active` `where`-gating when `start_frac > 0`). The source-LR schedule (S13′) is a pure function of `step`, so the LR at the activation step already reflects whatever warmup/decay that step prescribes. |

## 6. Variation points

| point | valid instantiations | production |
|---|---|---|
| `RECON_TERMS` | any static tuple of coefficiented terms, each a static plan of `(live_sites, SAMPLE_ROUTING, MASK_SOURCE)` entries: `subset_chunk_plan` · `per_site_plan` (the torch "layerwise" shape) · `all_sites_plan` · custom subset families (pairs, covers, …); built from the shared torch loss configs by `build_recon_terms` | ★ stochastic subset term + persistent-PGD term |
| `MASK_SOURCE` | `stochastic` (fresh U[0,1]/Bernoulli per draw) · `constant(v)` (`v=0` CI-masked, `v=1` unmasked; no delta path) · `fresh_pgd(init, n_steps, step_size, scope)` · `persistent(state_key)` | ★ stochastic + persistent |
| `SAMPLE_ROUTING` | `(key, [B,T]) → tuple of routing draws`, statically sized; draws may be jointly sampled (independent repeats, antithetic/complementary subsets, per-step covers) | ★ uniform_k_routing(·, 1) |
| `SRC_STEP` | `adam(β₁,β₂,ε)` with bias correction; `sign` (`sources += lr·sign(grad)`) | ★ adam(.5, .99, 1e-8) |
| `PROJ` / `EFFECTIVE` | **clamp**: PROJ = clamp[0,1], EFFECTIVE = identity, init U[0,1] · **sigmoid**: PROJ = identity (unbounded raw), EFFECTIVE = sigmoid, init N(0,1) | ★ clamp |
| `SCOPE` | persistent: `c (1,1)` · `sc (1,T)` · `nsc (n,T), n|B` · `bsc (B,T)` (jax: `sc` only today) — fresh-PGD: `c` · `bc` · `bsc` | ★ sc |
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
| §4.6 CI arch | `param_decomp/ci_fns.py::GlobalSharedTransformerCiFn` (`:156`), `param_decomp/ci_nn_blocks.py` |

The JAX implementation (`jax_single_pool/train.py`) uses these pseudocode names
verbatim: `clean_logits`, `site_inputs`, `source_masks`, `stochastic_recon_loss`,
`adversarial_recon_loss`, `sources_adam_ascend_project`, `ReconPlan`/`ReconForward`,
`uniform_k_routing`, `subset_chunk_plan`, `per_site_plan`.

Rationale worth keeping: the two squashings give each consumer gradient only in its
permitted direction (masks may push CI up out of saturation; the sparsity penalty may
not push it below 0). The adversary is *persistent* because re-finding the worst-case
ablation from scratch each step under-trains the adversary at any affordable inner-step
count. The `log2` term approximates a description-length / frequency penalty (`L_freq`
in the VPD paper) — its convexity is why S8 demands the true global sum. Fused-linear-KL
and LM-head-bypass are memory/throughput optimizations and must be semantically
invisible (cf. `recon_loss_kl` equivalence).

---

## 10. Eval (two-process split)

Eval is normative only at the boundary; the metric *values* it reports are not part of
the training-step semantics. The split (S28–S30):

- **FAST tier — in-loop.** Scalar eval metrics run inside the training process on
  cadence `eval.every` (JAX `run.py:320-348`). This is the torch `EvalLoop` analog
  restricted to scalars. `EvalConfig` (`config.py:90-104`) carries
  `batch_size`/`every`/`n_steps` and omits `slow_every`/`slow_on_first_step` — there is
  no in-loop slow tier.
- **SLOW / plot tier — offline.** All plot/heavy metrics are delegated to
  `pd-offline-eval`, run on exported checkpoints and keyed off the checkpoint-save
  cadence — a separate process, not the training loop. The torch
  `slow_every`/`slow_on_first_step` cadence (`optimize.py`'s `EvalLoop`) has no JAX
  in-loop analog by design.
- **Cadence coupling.** `log_every` divides `eval.every` (asserted `run.py:255`), so
  every eval step is also a train-log step.
- **Cast-point seam.** The offline `CEandKLLosses` carries the bf16-input cast-point
  asymmetry recorded in N3 (torch softmax under bf16 autocast; JAX fp32 before
  `log_softmax`), an accepted seam.
