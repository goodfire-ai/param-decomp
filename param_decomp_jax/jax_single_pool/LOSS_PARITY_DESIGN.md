# Loss parity: every torch loss Metric in the JAX trainer — design

Status: IMPLEMENTED (stages 1-3; 2026-06-11) — `recon.py` holds the strategies/terms/`build_recon_terms`, `train.py` the multi-term step, SPEC amended (S10′/S12′/S13′/S14′, S23/S24). Deferred per §6 stage 4: `nsc` scope, sigmoid parameterization, `start_frac>0`, the hidden-acts seam. Scope: the 15 `LOSS_METRIC_CLASSES` entries +
`ChunkwiseSubsetReconLoss` (lab) as the torch surface; `recon.py` / `adversary.py` /
`train.py` / `SPEC.md` as the JAX surface. Every torch `update()` /
`before_backward` / `after_backward` path was read, not just class names.

## Verdict on the hypothesis

**"All these recon losses are just a scoped recon with a source generation strategy"
— holds, with one clean exception and one quirk list.**

13 of the 16 loss classes are *the same function*: a recon plan (which sites are
live, how positions route) × a **mask-source strategy** (where the `[0,1]` source
values come from) × a per-datapoint **scope** (how sources are shared across
batch/seq). The class distinctions `CIMasked`/`Stochastic`/`Unmasked`/`PGD`/
`PersistentPGD` × `_`/`Subset`/`Layerwise` are exactly the cartesian product
(source strategy × plan shape) baked into class names. The JAX `ReconForward`
already carries `(live_sites, sample_routing)`; adding a `sources` strategy field
and lifting the single hardcoded `(stoch_coeff, adversary_coeff)` pair into a tuple
of coefficiented terms completes the factorization.

The exceptions:

- `FaithfulnessLoss`, `ImportanceMinimalityLoss` — not recons (weight-space /
  CI-space objectives). Already implemented; nothing to do.
- `StochasticHiddenActsReconLoss` — a recon in spirit but its objective is MSE on
  *internal* activations, which the `DecomposedLM` fn-table deliberately does not
  expose. The one genuine seam-breaker. Recommendation: keep on the offline bridge
  (it is already in `torch_config.OFFLINE_EVAL_METRIC_TYPES`), per Oli's stance
  that hidden-acts is an eval metric; see §4c.
- Quirks that survive the factorization but need explicit decisions: PPGD warmup's
  route-all override, fresh-PGD's once-per-batch routing draw, `start_frac`,
  `nsc` global-vs-per-rank tiling. All documented in §4.

## 1. Inventory

Axes: **objective** (what's compared), **sources** (where mask values come from),
**plan** (live-sites + routing structure), **scope** (per-datapoint sharing of
sources, shape-spelled `c`/`bc`/`sc`/`nsc`/`bsc`), **misc** (what doesn't fit).

All recon-KL losses share one normalization: torch returns `(Σ losses, Σ
n_examples)` and the live loss is `sum/n`; with uniform forward shapes this is
exactly the JAX "mean over all forwards of `kl_per_position`" (§4e).

| torch class | objective | sources | plan | scope | misc |
|---|---|---|---|---|---|
| `FaithfulnessLoss` | weight-space: `Σ‖Δ‖²/Σnumel` | — | — | — | already JAX (`losses.faithfulness_loss`) |
| `ImportanceMinimalityLoss` | CI-space Lp + entropy | — | — | — | already JAX; p-anneal, β, D2 global sums |
| `UnmaskedReconLoss` | KL final logits | const **1** (mask≡1, CI irrelevant) | 1 fwd, all sites, route-all | n/a (constant) | **no delta path** (delta_mask ≡ 0) |
| `CIMaskedReconLoss` | KL | const **0** (mask = ci) | 1 fwd, all sites, route-all | n/a | no delta path |
| `CIMaskedReconSubsetLoss` | KL | const 0 | 1 fwd, all sites live, routed subset (`uniform_k` / `static_p` / `all`) | n/a | router over the full site set |
| `CIMaskedReconLayerwiseLoss` | KL | const 0 | one fwd per site, route-all (= `per_site_plan`) | n/a | |
| `StochasticReconLoss` | KL | fresh U[0,1] (or Bernoulli) per draw | `n_mask_samples` fwds, all sites, route-all | `bsc` (fresh full-shape) | delta_mask ~ U[0,1] per position |
| `StochasticReconSubsetLoss` | KL | fresh stochastic | `n_mask_samples` fwds, all sites, routed per `cfg.routing` | `bsc` | = production loss at one chunk |
| `StochasticReconLayerwiseLoss` | KL | fresh stochastic | `per_site_plan` × `n_mask_samples` | `bsc` | torch samples all sites' masks jointly per draw, applies one per fwd — ≡ independent per-fwd draws (sources are site-independent anyway) |
| `ChunkwiseSubsetReconLoss` (lab) | KL | fresh stochastic | `subset_chunk_plan(sites_per_chunk, n_samples)` | `bsc` | `use_fused_kl` (impl detail, §4e); **this is the JAX trainer's existing stochastic loss** |
| `StochasticHiddenActsReconLoss` | **per-module MSE on cached module outputs** | fresh stochastic | all sites, route-all × `n_mask_samples` | `bsc` | needs `forward_with_output_acts` — no JAX seam (§4c); per-**element** normalization, not per-position |
| `PGDReconLoss` | KL | fresh sign-PGD: `init`∈{random,ones,zeroes}, `n_steps`, `step_size` | all sites, route-all | `c`/`bc`/`bsc` | already JAX twice (training adversary + eval probe); `c`-scope grads AVG-reduced cross-rank |
| `PGDReconSubsetLoss` | KL | fresh sign-PGD | routed subset; **routing drawn ONCE per batch**, fixed for all `n_steps` + final eval | `c`/`bc`/`bsc` | see §4 quirk Q2 |
| `PGDReconLayerwiseLoss` | KL | fresh sign-PGD, **independent PGD per site** | loop over sites: `LayerRouter` (route only that site) per inner PGD | `c`/`bc`/`bsc` | n_sites separate inner ascent loops |
| `PersistentPGDReconLoss` | KL | persistent sources + Adam/sign moments, cross-step | all sites, route-all, × `n_samples` | `c`/`sc`/`nsc`/`bsc` | warmup ascents, S14 fused final ascent, sigmoid param, `start_frac`, eval-time hidden-acts extras |
| `PersistentPGDReconSubsetLoss` | KL | persistent | loss fwds routed per `cfg.routing` (fresh draw per sample); **warmup fwds route ALL** (quirk Q1) | same | |

Not in `LOSS_METRIC_CLASSES` (eval-only, lab-side): `CIHiddenActsReconLoss`,
`CIMaskedAttnPatternsReconLoss`, `StochasticAttnPatternsReconLoss` — covered in
§4c/§4d; they stay on the bridge.

Trainer-level torch knobs that parameterize these (not per-loss):
`pd.n_mask_samples`, `pd.sampling` (`continuous`/`binomial`),
`pd.use_delta_component` — all already in the shared config; the converter folds
them into the generated plans.

## 2. The unified model

### 2.1 Types (sketch)

```python
# recon.py — ReconForward gains a source strategy; a step has a TUPLE of recon terms.

@dataclass(frozen=True)
class StochasticSources:
    sampling: Literal["continuous", "binomial"]   # component sources; delta ~ U[0,1]

@dataclass(frozen=True)
class ConstantSources:
    value: float          # 0.0 → CI-masked (mask = ci); 1.0 → unmasked (mask ≡ 1)
                          # delta_mask ≡ 0 (no delta path; §4b)

@dataclass(frozen=True)
class FreshPGDSources:    # torch PGDRecon* family
    init: Literal["random", "ones", "zeroes"]
    n_steps: int
    step_size: float
    scope: Literal["c", "bc", "bsc"]              # source leading shape

@dataclass(frozen=True)
class PersistentSources:  # torch PersistentPGDRecon* family
    state_key: str        # index into TrainState.sources / .sources_opt_state
    # scope/optimizer/warmup config rides the shared PersistentPGD*LossConfig,
    # resolved by the step factory; this is just the state pointer.

MaskSourceStrategy = StochasticSources | ConstantSources | FreshPGDSources | PersistentSources

@dataclass(frozen=True)
class ReconForward:
    live_sites: tuple[str, ...]
    sample_routing: RoutingSampler     # unchanged; gains static_probability_routing()
    sources: MaskSourceStrategy        # NEW

@dataclass(frozen=True)
class ReconLossTerm:
    name: str                          # = Metric.instance_key → metric log key
    coeff: float
    plan: tuple[ReconForward, ...]
    # term loss = mean over ALL draws of ALL entries of kl_per_position (S10, per term)

ReconLossTerms = tuple[ReconLossTerm, ...]
```

All of this is static structure closed over by the jit'd step (exactly like today's
`ReconPlan`) — term count, plan shapes, strategy kinds never vary per step, so no
retracing. Only `PersistentSources.state_key` indexes runtime state.

### 2.2 TrainState

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    ...
    sources: dict[str, dict[str, Array]]            # state_key → site → source
    sources_opt_state: dict[str, SourcesAdamState]  # state_key → moments (sign: empty)
```

Production configs have exactly one persistent term, so the dicts are singleton —
checkpoint layout changes shape but not substance (orbax handles nested dicts;
old checkpoints migrate by nesting under the term name — a deliberate one-time
break, no legacy shim per repo policy, or accept incompatibility since the
generalization lands between runs).

### 2.3 The step (replaces the `stochastic_recon_loss` / `adversarial_recon_loss` pair)

```python
def recon_term_loss(term, frozen, components_bf16, ci_lower, persistent_sources_for_term,
                    residual, clean_logits, key) -> Array:
    total, n = 0, 0
    for entry in term.plan:
        for routes in entry.sample_routing(entry_key, (B, T)):
            masks, delta_masks = materialize(entry.sources, ci_lower, entry.live_sites, draw_key)
            total += kl_per_position(masked_forward(..., routes, entry.live_sites), clean_logits)
            n += 1
    return total / n
```

`materialize` dispatches on the strategy:

- `StochasticSources` — today's inline draw (`U[0,1]` component sources +
  `U[0,1]` delta mask), per draw per site.
- `ConstantSources` — `mask = ci + (1−ci)·v` (`v=0` ⇒ `ci`; `v=1` ⇒ ones),
  `delta_mask = 0`. No RNG.
- `FreshPGDSources` — before the main `loss_fn`: init per `scope`/`init`, run the
  `n_steps` sign-ascent `lax.scan` against *this term's* recon (today's fresh-PGD
  branch, generalized to the term's plan), `stop_gradient` the result, then the
  main loss re-forwards through them (gradient reaches components/CI through the
  interpolation, sources are constants — torch-identical).
- `PersistentSources` — read `state.sources[state_key]`; these stay **live leaves**
  in the fused backward (S14).

Step skeleton:

```python
# 1. clean/CI as today.
# 2. For each persistent term: n_warmup supplemental ascents (params+CI detached),
#    each ascending ONLY that term's sources against ONLY that term's recon loss
#    (torch: each Metric's warmup uses its own loss; terms are sequential, as in torch).
# 3. For each fresh-PGD term: the inner sign-ascent scan; stop_gradient.
# 4. loss_fn over (components, ci_fn, {state_key: sources}):
#        total = faith_coeff·faith + imp_coeff·imp + Σ_term term.coeff · recon_term_loss(term, ...)
# 5. One fused backward. For each persistent term:
#        sources_grad = grads.sources[state_key] / term.coeff      # S14 unscaling, per term
#        ascend + project via that term's optimizer state.
# 6. Components/CI optimizer steps as today.
```

The per-term coeff division is exact because each persistent source dict is a
distinct pytree leaf-set appearing in **exactly one** term — this must become a
spec invariant (§6, new S23), since a source bundle shared across terms would make
the unscaling wrong. Torch gets the same result differently: `run_loss_step` calls
`m.before_backward(losses[metric_name])` with the **un-coeffed** live loss and PPGD
runs a separate `torch.autograd.grad(live_loss, sources, retain_graph=True)`
(`persistent_pgd_recon.py:214-218`) before `total_loss.backward()`; JAX reuses the
fused backward and divides — numerically identical, one fewer backward.

### 2.4 Config surface

No new config mirrors. The JAX `ExperimentConfig` currently flattens the loss list
into `faith_coeff / stoch_coeff / imp_min / adversary / ReconConfig` — replace that
with carrying the shared configs through:

```python
class ExperimentConfig:
    ...
    loss_metrics: tuple[AnyLossMetricConfig, ...]   # the shared pydantic union, as-is
    n_mask_samples: int
    sampling: Literal["continuous", "binomial"]
```

`make_train_step` (or a pure `build_recon_terms(loss_metrics, site_names, ...)`
helper) maps each shared config → `ReconLossTerm` and asserts the supported
subset, exactly the current `torch_config._losses` philosophy moved one level
down. `torch_config.py`'s converter then shrinks: `_losses` stops pattern-matching
into four slots and just passes the validated list through (still refusing
unsupported types, e.g. `StochasticHiddenActsReconLoss` as a *training* loss).
Duplicate same-class entries are keyed by `name` (torch `instance_key` semantics)
— assert distinct names exactly as `instantiate_metrics` does.

## 3. Simplification wins — which classes dissolve

The entire recon cartesian product collapses; the class names become converter
table rows:

| torch class | parameterization (`ReconLossTerm` with…) |
|---|---|
| `UnmaskedReconLoss` | 1 entry: all sites, `route_all`, `ConstantSources(1.0)` |
| `CIMaskedReconLoss` | 1 entry: all sites, `route_all`, `ConstantSources(0.0)` |
| `CIMaskedReconSubsetLoss` | 1 entry: all sites, router-per-`cfg.routing` (n_draws=1), `ConstantSources(0.0)` |
| `CIMaskedReconLayerwiseLoss` | `per_site_plan` entries, `ConstantSources(0.0)` |
| `StochasticReconLoss` | 1 entry: all sites, `route_all` → sampler with `n_draws=n_mask_samples`, `StochasticSources(sampling)` |
| `StochasticReconSubsetLoss` | 1 entry: all sites, `cfg.routing` sampler × `n_mask_samples`, `StochasticSources` |
| `StochasticReconLayerwiseLoss` | `per_site_plan` × `n_mask_samples`, `StochasticSources` (joint-then-split sampling ≡ independent; R1 already permits) |
| `ChunkwiseSubsetReconLoss` | `subset_chunk_plan(sites_per_chunk, n_samples)`, `StochasticSources` — **the existing production term, unchanged** |
| `PGDReconLoss` | 1 entry: all sites, `route_all`, `FreshPGDSources(init, n_steps, step_size, scope)` |
| `PGDReconSubsetLoss` | 1 entry: all sites, `cfg.routing` (drawn once per step — Q2), `FreshPGDSources` |
| `PGDReconLayerwiseLoss` | `per_site_plan` entries, each with its **own** `FreshPGDSources` ascent (torch runs an independent PGD per site) |
| `PersistentPGDReconLoss` | 1 entry: all sites, `route_all` × `n_samples`, `PersistentSources(key)` |
| `PersistentPGDReconSubsetLoss` | loss plan: `cfg.routing` × `n_samples`, `PersistentSources(key)`; warmup plan: route-all (Q1) |

`FaithfulnessLoss` / `ImportanceMinimalityLoss` stay as the two non-recon terms
(already done). `StochasticHiddenActsReconLoss` is the only class that does not
dissolve (different objective seam, §4c).

Net: 13 torch loss classes → 4 strategy dataclasses + 3 plan builders + 1 loss
function. The torch `Subset`/`Layerwise` suffix classes were never semantics, only
plan shapes — the hypothesis is confirmed at the strongest reading.

## 4. Counterexample hunt — where it strains

**(a) Per-entry / layerwise PERSISTENT PGD.** Torch has no
`PersistentPGDReconLayerwiseLoss`, so nothing to be parity with — but the
factorization handles it if wanted: one persistent term per site
(`state_key="ppgd/{site}"`), each with its own moments. The S14 fused ascent works
per term because each term's sources are distinct leaves of the grad pytree; one
backward yields all of them, each divided by its own coeff. Strains: (i) warmup
ascents are sequential per term — n_sites × n_warmup extra forwards per step
(torch would pay the same; it's inherent, not a JAX problem); (ii) `TrainState`
grows n_sites × `(1,T,C+1)` sources + 2× moments — fine. If instead one
*persistent state* were shared across multiple plan entries **within one term**,
the summed entry-grads are exactly the ascent gradient of the term's mean loss —
also fine. The only forbidden topology is one source bundle in *two terms* (breaks
coeff-unscaling) — new invariant S23.

**(b) Delta-mask under scopes/strategies.** Falls out cleanly: the delta source is
the trailing channel of whatever the strategy produces, so it automatically
inherits the scope (`c`-scope fresh PGD ⇒ one delta scalar for everything;
`sc` PPGD ⇒ per-position shared across batch — exactly torch's
`expanded_adv_sources[k][..., -1]`). Two real notes: (i) torch's CI-masked and
unmasked losses pass `weight_deltas_and_masks=None` — the delta path is *absent*,
not zero-masked. `delta_mask=0` is mathematically identical (`y += (x@Δ)·0`) but
pays the `x@Δ` matmul; the in-loop eval already accepts this (`eval.py`
`zeros_delta`). If it matters, add a static `has_delta: bool` per entry so XLA
skips the delta matmul — static, retrace-safe. (ii) Stochastic delta masks are
always full `(B,T)` fresh draws regardless of anything — consistent with torch.

**(c) Hidden-acts losses.** `StochasticHiddenActsReconLoss.update` calls
`model.forward_with_output_acts(batch)` twice (target acts, then masked acts) and
MSEs per module — it needs every target module's *output activation*, which
`DecomposedLM` deliberately does not expose (the fn-table returns final logits
only). The seam would be a fifth fn,
`masked_site_outputs(frozen, vu, residual, masks, delta_masks, routes, live) ->
dict[site, (B,T,d_out)]`, implemented per target. That is mechanical but it (i)
adds a per-target implementation burden ×2 targets, (ii) normalizes per-element
(MSE over numel), a different reduction from every other term, and (iii) as a
*training* loss it is exactly the "site-local recon" this trainer's CLAUDE.md
calls a conceptual no-no. Oli's stance (hidden-acts ≈ eval metric) matches the
plumbing reality: it is already in `OFFLINE_EVAL_METRIC_TYPES` and `pd-offline-eval`
computes it bit-faithfully on exported checkpoints. **Recommendation:
keep-on-bridge; do not build the seam unless a training-loss use case appears, and
then as an explicit spec amendment.** (Note: PPGD's eval-time hidden-acts extras
in `_accum_hidden_acts` are eval-only decoration on the metric, not part of the
training loss — they ride the same bridge.)

**(d) Attn-pattern losses.** `CIMaskedAttnPatternsReconLoss` /
`StochasticAttnPatternsReconLoss` are **eval-only** (lab `eval_metrics`, not in
`LOSS_METRIC_CLASSES`) — no training-parity obligation exists. They need pre-RoPE
q/k site outputs + RoPE + masked-softmax patterns, strictly deeper internals than
(c). Keep-on-bridge, unconditionally.

**(e) `use_fused_kl` and normalization mismatches.** Audited every recon `update`:
all return `(Σ sum_kl, Σ n_positions)` accumulated across forwards; live loss
`sum/n` = mean-over-forwards of `kl_per_position` whenever all forwards share
`(B,T)` — always true here. `ChunkwiseSubsetReconLoss` computes
`Σ_f (loss_f/n_pos) / n_forwards` directly — same number. `use_fused_kl` is a
memory optimization required to be semantically invisible (`recon_loss_kl`
equivalence; SPEC §9 already says so) — correctly ignored by the converter. The
two true reduction outliers are non-recon or bridged: imp-min (global-sum-inside-
log2, D2, done) and hidden-acts (per-element, bridged). MSE `ReconstructionLoss`
(TMS/ResidMLP) is out of scope for this LM trainer. **No counterexample.**

**(f) Multiple simultaneous adversary losses.** Torch allows it today: 
`loss_metrics` is a list; two same-class entries need distinct `name`s
(`instantiate_metrics` asserts), different-class (e.g. `PersistentPGDReconLoss` +
`PGDReconLoss` + `PersistentPGDReconSubsetLoss`) coexist freely, each with its own
state and its own `before_backward`/`after_backward` (each does its own
`autograd.grad(its_live_loss, its_sources, retain_graph=True)`). The unified model
covers this exactly: N terms, N state keys, one fused backward, N per-term
unscalings. `TrainState` needs only the §2.2 dicts. The fresh-PGD inner loops and
persistent warmups run sequentially before `loss_fn` — same cost structure as
torch. One torch behavior to preserve: `validate_pgd_scope` (per-rank batch
divisibility for `nsc`) becomes a converter assert.

**Residual quirks (decisions needed, all small):**

- **Q1 — PPGD warmup routes ALL even for the Subset variant.**
  `persistent_pgd_state.py:321-327`:
  ```python
  all_layers = AllLayersRouter()
  for _ in range(self._n_warmup_steps):
      sum_loss, n = self.compute_recon_sum_and_n(..., router=all_layers)
  ```
  while the main-loss forwards use `self._router` with fresh draws per sample. The
  unified model needs the persistent term to carry a *warmup plan* distinct from
  its *loss plan* (both static). For the production all-sites term they coincide.
  Propose: replicate torch (warmup plan = route-all) and record it in the spec.
- **Q2 — fresh-PGD routing is drawn once per batch.** `pgd_masked_recon_loss_update`
  samples `routing_masks` once, then closes over it through all `n_steps` ascents
  AND the final evaluation; PPGD by contrast redraws per `compute_recon_sum_and_n`
  call. Replicate: the `FreshPGDSources` ascent and the final loss forward share
  one routing draw per step.
- **Q3 — `start_frac`.** Torch returns `None` before `start_frac` (term simply
  absent, state lazily constructed at first active step). Under a static jit plan
  the analog is `coeff_t = where(step ≥ start_frac·total, coeff, 0)` plus gating
  source/moment updates with `where` — semantics match (sources init at step 0 but
  untouched until active; distributionally identical since init is RNG-pure), cost
  is paying the adversary forward before activation. Either accept that cost or
  keep refusing `start_frac > 0` (current assert). Propose: keep refusing until a
  config actually needs it.
- **Q4 — `nsc` tiling.** Torch tiles `n_sources` over the **per-rank** batch slice
  (synced replicas ⇒ each source covers `B/n` global elements, rank-interleaved);
  JAX would tile over the global batch (contiguous). Same multiset of assignments,
  different element→source pairing; batch elements are exchangeable, so this is a
  distributional no-op — document as an R2-style divergence, don't chase layout
  parity.
- **Q5 — eval/loss double-duty.** Torch auto-evaluates loss metrics and allows a
  second eval-only instance via `name` (e.g. 20-step PGD probe). The JAX in-loop
  eval already has the PGD probe; per-term eval logging falls out of per-term
  metrics keys (`loss/<term.name>`). No design change needed.

## 5. Effort map

| torch loss | tier | notes |
|---|---|---|
| `FaithfulnessLoss` | **already-runnable** | `losses.faithfulness_loss`, S17/N2 |
| `ImportanceMinimalityLoss` | **already-runnable** | S7–S9, D2; `p_anneal_final_p=None` (constant p) is a trivial assert-relax |
| `ChunkwiseSubsetReconLoss` | **already-runnable** | the production stochastic term |
| `StochasticReconSubsetLoss` (uniform_k) | **already-runnable** | converted today as 1-chunk plan |
| `PersistentPGDReconLoss` (sc/bsc, Adam, clamp) | **already-runnable** | the production adversary; `bsc` is batch-sharded (`P("dp", None, None)`), no replica sync |
| `PGDReconLoss` (as the single adversary; as eval probe) | **already-runnable** | both paths exist |
| `UnmaskedReconLoss` | **composition-only** | `ConstantSources(1.0)`; optional `has_delta` static flag |
| `CIMaskedReconLoss` / `Subset` / `Layerwise` | **composition-only** | `ConstantSources(0.0)` × plan shape; `static_probability` routing sampler is ~5 lines |
| `StochasticReconLoss` / `Layerwise` | **composition-only** | plan shapes; `binomial` sampling = one `random.bernoulli` branch |
| `PGDReconSubsetLoss` / `Layerwise` | **composition-only** | `FreshPGDSources` per entry + Q2 routing-draw sharing; `bc` scope shape exists in `init_fresh_pgd_sources` already |
| `PersistentPGDReconSubsetLoss` | **composition-only** | needs warmup-plan/loss-plan split (Q1) + routed loss fwds |
| PPGD scopes `c`/`nsc` | **composition-only** | source-shape variants of `init_persistent_sources` + Q4 note (`bsc` now implemented: batch-sharded, no replica sync per S16) |
| PPGD `sign` SRC_STEP, sigmoid parameterization, `n_samples>1` | **composition-only** | SPEC §6 already names them as variation points |
| multiple simultaneous loss/adversary terms | **composition-only** | §2.2 TrainState dicts + per-term S14 |
| PPGD `start_frac > 0` | **composition-only (deferred)** | Q3 — keep the refusing assert until needed |
| `StochasticHiddenActsReconLoss` | **needs-new-seam → recommend-keep-on-bridge** | `masked_site_outputs` fn per target; conflicts with the one-recon-semantics rule; already offline (§4c) |
| attn-pattern eval losses | **keep-on-bridge** | eval-only in torch too (§4d) |

**`torch_config.py` converter changes:** `_losses` stops slotting into
`(faith, stoch, imp, adversary)` and emits `(loss_metrics_passthrough, …)`;
per-type asserts move into `build_recon_terms` next to the structures they
guard; add `validate_pgd_scope`-equivalent divisibility asserts; keep refusing
`StochasticHiddenActsReconLoss`-as-training-loss and `start_frac>0`; the
`OFFLINE_EVAL_METRIC_TYPES` set is unchanged. `ExperimentConfig` swaps
`faith_coeff/stoch_coeff/imp_min/adversary/ReconConfig.{sites_per_chunk,n_samples}`
for `loss_metrics + n_mask_samples + sampling` (`remat_forwards` stays).

**SPEC amendments (explicit, with Oli):**

- **S10′** — generalize from "the stochastic recon loss" to *recon loss terms*:
  each term is a static plan of `(live_sites, sampler, source-strategy)` entries;
  term loss = mean over its forwards of `kl_per_position`; total = Σ coeff·term.
- **S12′** — "the adversarial term masks ALL sites" becomes a property of the
  *production* term, not of adversaries per se; subset-routed adversarial terms
  route per their plan. Source detachment statement unchanged.
- **S13′/S14′** — per persistent term: its own optimizer state, its own
  `n_warmup+1` updates, final ascent from the fused backward unscaled by *its*
  coeff.
- **NEW S23** — a persistent source bundle feeds exactly one loss term (the
  coeff-unscaling validity condition).
- **NEW S24** — warmup-plan vs loss-plan for persistent terms (Q1: warmup routes
  everywhere, torch parity); fresh-PGD terms share one routing draw per step (Q2).
- **§6 table** — add `ConstantSources` / `FreshPGDSources` as MASK_SOURCE
  variation points; extend SCOPE with the fresh-PGD `bc`; note Q4 for `nsc`.
- **§2 constants** — unchanged (production config is untouched by all of this).

## 6. Recommended implementation order

1. **Stage 1 — multi-term recon + deterministic/stochastic strategies.**
   `ReconLossTerm` tuple in the step factory, `ConstantSources`/`StochasticSources`,
   `static_probability` + `route_all` samplers, per-term metric keys, converter
   passthrough. Unlocks: Unmasked, CIMasked×3, Stochastic×3, binomial. Pure
   refactor of `train.py`'s two closures into one; production configs must produce
   a bit-identical trajectory (regenerate nothing; the equivalence fixtures
   already pin the four production terms — extend `tests/equivalence` with one
   golden per new strategy).
2. **Stage 2 — fresh-PGD as a term strategy.** Lift the existing fresh-PGD branch
   into `FreshPGDSources` per term, Q2 routing-draw sharing, `per_site_plan`
   composition. Unlocks PGDSubset/Layerwise and PGD-train + PGD-eval coexistence.
3. **Stage 3 — persistent terms generalized.** `TrainState.sources` → keyed dicts
   (checkpoint-shape change — land between runs), per-term warmup with the Q1
   route-all plan, per-term S14 unscaling, scopes `c`/`nsc`/`bsc`, `sign` SRC_STEP,
   sigmoid parameterization, `n_samples>1`. SPEC amendments S13′/S14′/S23/S24 +
   §6 land in the same PR as the code, cited by ID.
4. **Stage 4 — only on demand.** `start_frac` step-gating (Q3); the
   `masked_site_outputs` seam for hidden-acts *iff* someone actually wants it as a
   training loss — otherwise it stays on `jsp-export` → `pd-offline-eval`
   permanently, alongside the attn-pattern metrics.

Each stage keeps the step a single jit'd function with static structure; nothing
in the design introduces per-step Python branching on traced values.
