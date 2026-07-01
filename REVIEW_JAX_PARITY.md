# JAX↔main parity review ledger

Line-by-line walkthrough of `feature/jax` core against `main` (torch = ground truth),
also correcting `SPEC.md` where it has drifted. Oli is the final arbiter.

- **main worktree**: `/mnt/home/oli/param-decomp-main-review`
- **order**: data-flow bottom-up (leaves → train loop)
- Each chunk records: JAX-vs-main verdict, spec corrections, main bugs Oli flags.

Legend: ✅ agreed faithful · ⚠️ flagged, awaiting decision · 🔧 spec needs fix · ❌ divergence

## Divergence taxonomy (Oli's framing)

Every flag is tagged by class:

- **Class A — graph correctness** (CRITICAL): what data operates on what, gradient flow,
  stop-gradients, structure of the computation. A bug here = wrong algorithm.
- **Class B — in-value correctness** (fixable): inits, epsilons, dtypes, constants. Right
  shape of computation, possibly wrong number.
- **Class C — software/scope**: config surface, missing features, capability gaps.

Graph correctness (A) is the priority; B and C are noted but not blocking unless Oli says so.

---

## Chunk 1 — component representation & decomposed-linear primitive

**JAX**: `components.py` (`SiteC`/`SiteSpec`, `DecompVU`, `init_decomp_vu`, `site_out`, fp8 QAG)
**main**: `components.py` (`Components`/`LinearComponents`/`EmbeddingComponents`, `init_param_`, `make_components`)
**spec**: §4.1 forward semantics

### Math verdict (core)
- **Init**: JAX `V ~ N(0, d_in^-0.5)`, `U ~ N(0, C^-0.5)`. Torch `init_param_` with `gain("linear")=1.0` ⇒ identical std. ✅
- **Forward `((x@V)*mask)@U`**: identical. ✅
- **Delta path**: torch forms `weight_delta=[d_out,d_in]` and does `x@Δ.T`; JAX expands to activation space `x@W.T − (x@V)@U` and never forms Δ. Algebraically identical (`x@(W−(V@U).T).T = x@W.T − (x@V)@U`). bf16 rounding divergence is documented + reconciled by N2 (faithfulness uses fp32 `weight_deltas`, a separate path). ✅ (pending review of that path)
- **SPEC §4.1**: matches both. Uses `Δ_s=W_s−V_s@U_s` / `x@W_s` with W_s in `[d_in,d_out]` orientation (JAX stores `[d_out,d_in]`, transposes). Faithful; orientation could be noted. ✅

### Flags (resolved with Oli)
- **(Class C) Frozen bias dropped.** Torch `LinearComponents.forward` re-adds a frozen `bias`
  (and `make_components` carries it from Linear/Conv1D targets). JAX `site_out` has NO bias
  term; SPEC §4.1 omits it too. **Status: non-blocking but in principle necessary** (Oli).
  Llama is bias-free so currently moot. TODO if a bias-bearing target is ever decomposed.
- **(Class C) Embedding decomposition dropped from the primitive.** Torch had
  `EmbeddingComponents` (`V[x]` indexing). JAX `site_out` is linear-only (`x@V`).
  **Status: currently unused, a technical gap, not urgent** (Oli). Confirm in target review
  that no JAX run decomposes the embedding.
- **(Class A) Routing folded into `site_out` — VERIFIED FAITHFUL.** Torch
  (`component_model.py:272`) selects `where(routing_mask[...,None], components_out, frozen_output)`
  via a forward hook (`"all"` sentinel → decomposed everywhere). JAX folds the same
  `where(route[...,None], y_dec, x@W.T)` into the primitive (`route is None` → y_dec
  everywhere). Graph-identical: both compute both branches everywhere then select; V/U get
  gradient only at routed positions (`∂where/∂a = cond·g`); frozen W never differentiated in
  either; grad to input `x` flows through the live branch per position. Two-level routing
  (live_sites whole-site gate + per-position route) preserved; the live_sites gate is in
  `masked_forward` (verify in lm.py). Mechanism differs (hooks vs explicit thread) — lm.py
  concern, not the primitive.
- **(perf, out of scope) fp8 QAG + `with_sharding_constraint`** — JAX-only; no semantic effect.

**Verdict: ✅ FAITHFUL (graph).** Class-C gaps (bias, embedding) noted as non-urgent.

---

## Chunk 2 — `DecomposedModel` interface (the contract)

**JAX**: `lm.py` (Protocol only — no computation; `SiteMasks`/`SiteDeltaMasks`/`SiteRoutes`,
`all_false_routes`, `chunk_sites`)
**main**: `component_model.py` (`ComponentModel`, hook-driven `forward`)
**spec**: §4.1 `masked_forward`/`clean_output`, §4.6

`lm.py` is the interface; the actual masked-forward graph is in `targets/llama8b.py` (next chunk).
Surface maps cleanly to torch's single hook-based `ComponentModel`:
`clean_output`↔`forward(batch)`, `masked_output`↔`forward(batch,mask_infos)`,
`read_activations`↔`forward(cache_type="input")`, `masked_site_outputs`↔`forward(cache_type="output")`,
`weight_deltas`↔`W−V@U`. `prepare_compute_weights` is JAX-only (÷N→÷fsdp gather).

### Commitments to verify downstream
- **(Class A) `live` is STATIC under jit.** Torch chose decomposed modules dynamically via
  `mask_infos` keys. Equivalent IFF the recon plan enumerates `live` sets correctly → verify
  in recon. (This is chunk-1 routing level-(a); confirmed present: non-live sites = frozen `x@W`.)
- **(Class A) `has_delta` static gate** skips `x@Δ` when delta-mask const 0. Torch:
  `wd = ctx.weight_deltas if ctx.use_delta_component else None`. Verify same firing cases → recon.
- **(Class A, IMPORTANT) `read_activations` uses residual taps `resid.{layer}`** vs torch CI
  fn consuming per-module INPUT acts (`cache_type="input"`). Must confirm torch's CI fn also
  read residuals (SPEC pins chunkwise transformer on residual taps) → **nail in ci_fn chunk.**
- **(perf) `prepare_compute_weights`** claimed read-only/mask-independent bf16 gather. Confirm
  it doesn't alter rounding feeding the loss (ties to N2 fp32-faithfulness) → target chunk.

**Verdict: ✅ contract faithful; 4 commitments deferred to downstream chunks.**

---

## Chunk 3 — concrete masked forward (`LlamaDecomposedModel`)

**JAX**: `targets/llama8b.py` (`_run_masked_forward`, `masked_site`/`block`, `clean_output`,
`read_activations`, `prepare_compute_weights`, `weight_deltas`, `FrozenAttn`/`LlamaLayer`)
**main**: HF transformers Llama + `component_model.py` forward hooks
**spec**: §4.1 `masked_forward`/`clean_output`, S2/S3

### Graph verdicts
- **(Class A) Site placement faithful.** Sites splice in at the exact inputs to each frozen
  Linear: q/k/v ← post-LN1 `h1`, o ← attention output `attn_y`, gate/up ← post-LN2 `h2`,
  down ← `silu(gate)·up`. Matches where torch hooks fired; matches `read_activations` taps. ✅
- **(Class A) Two-level live/frozen dispatch faithful.** `kind not in decomposed_kinds` → pure
  frozen; else `lax.cond(e["live"], decomp, frozen)` per (layer,kind). Decomp = `site_out`
  (incl. per-position `route`); frozen = `x@W.T`. Grad flows only through the taken branch
  (lax.cond) → live sites get V/U grad, frozen don't = torch hooked/not-hooked. ✅
- **(Class A) S2/S3 holds.** Non-live = `x@W.T`, NOT the `V@U+(W−V@U)` mask-1 identity. Verified
  `clean_output` ≡ all-frozen `masked_output` (same RMSNorm/attn/MLP). No V/U grad, no bf16
  identity rounding on frozen sites. ✅ (one of the sharp-teeth invariants — confirmed)
- **(Class A/N2) `weight_deltas` = fp32 `W−(V@U).T`** feeds faithfulness only; recon delta stays
  bf16 activation-space in `site_out`. Consistent with chunk 1. Verify consumer in losses.py. ✅
- **remat** (`nothing_saveable`/`dots_saveable`) grad-transparent recompute; zero numerics
  change. ✅
- **(Class B) `lax.scan` over layers + `PD_UNROLL_K`** reassociate float ops vs torch's Python
  loop — within fp32 tol, documented. ℹ️ acceptable.
- **(perf) fp8 QAG / `PD_REPLICATE_WEIGHTS`** — env knobs, default off; fp8 perturbs numerics
  only when enabled. Out of scope.

### To confirm later
- **(Class A, load-bearing) frozen forward = recon target** is a JAX reimpl of HF Llama
  (`vendored_jax.llama`). Confirm an equivalence test pins `clean_output` vs HF → tests chunk.
- **`live` tuples** built by the recon plan → verify in recon chunk.
- **Embedding (chunk-1 Class C)**: confirmed — `embed` is a plain frozen lookup, NEVER a
  decomposed site here (sites regex = attn/MLP proj only). Embedding decomposition genuinely
  absent in this target.

**Verdict: ✅ FAITHFUL (graph).**

---

## Chunk 4 — CI function

**JAX**: `ci_fn.py` (squashings, `CI`/`from_logits`, `CIBlock`, `ChunkwiseTransformerCIFn`;
toy `LayerwiseMLPCIFn`/`GlobalMLPCIFn`)
**main**: `ci_sigmoids.py`, `ci_fns.py` (`MLPCiFn`/`VectorMLPCiFn`/`GlobalSharedMLPCiFn`/
`GlobalSharedTransformerCiFn`), `ci_nn_blocks.py`
**spec**: §4.2, §4.6, S5/S6
**theory**: paper §"VPD CI function" — CI fn may be a function of ~any activations; torch & jax
implement different, partially-overlapping tap subsets. Oli: OK if both valid instances +
graph-correct.

### Graph verdicts
- **(Class A, the critical one) Squashings graph-identical.** `lower_leaky_hard`: fwd
  `clip(x,0,1)`; custom bwd `x≤0→(g<0?0.01g:0)`, `0<x≤1→g`, `x>1→0`. `upper_leaky_hard`:
  `where(x>1,1+0.01(x−1),clip(x,0,1))` ordinary autodiff. Byte-for-byte = torch
  `LowerLeakyHardSigmoidFunction` / `upper_leaky_hard_sigmoid`; α=0.01; x=0 tie-break → lower
  branch. SPEC S6 describes it correctly. ✅
- **(theory) `from_logits` lower→masks / upper→imp-min** matches paper (g∈[0,1] mask floor +
  upper-leak gradient on saturated logits). torch recon uses `ci.lower_leaky` (confirmed
  stochastic_recon.py:96); imp-min uses upper → confirm in losses chunk.
- **(Class C, Oli-sanctioned) Architecture & input taps DIFFER.** torch: per-site matrix
  inputs + one global transformer + per-site slice. jax: residual `resid.{layer}` taps +
  per-chunk transformers (`lax.scan`) + per-slot heads. Both valid theory instances;
  jax per-chunk receptive field narrower but residual stream is cumulative — legitimate.
- **(Class A) `inv_freq` stop_gradient** (line 395), the ONLY stop-grad; no stray detaches. ✅
- **(Class B, already unified)** GELU exact-erf; RMS eps `finfo(fp32).eps`; weightless RMS =
  torch `F.rms_norm(x,(dim,))`. fp32 masters→bf16 (N1). ✅
- **remat / scan-vs-vmap (`PD_CI_BROADCAST`)** grad-transparent / reassociation. perf.

### To confirm in train.py (Class A, load-bearing)
- **CI fn inputs must be FROZEN-path activations** (`read_activations` on clean path), NOT the
  V/U-dependent masked forward. Else recon-loss grad would leak into V/U via the CI inputs.

**Verdict: ✅ valid theory instance, graph-correct. Squashings identical; tap/arch difference
sanctioned. One train.py check pending.**

---

## Chunk 5 — faithfulness + importance-minimality losses

**JAX**: `losses.py` (`faithfulness_loss`, `_imp_min_terms`/`importance_minimality_terms`/
`smooth_l0_*`, `imp_min_terms`, annealing helpers, `kl_per_position`)
**main**: `metrics/faithfulness.py`, `metrics/importance_minimality.py`
**spec**: §2.3, S7/S8/S8′/S9, S17, N3

### Verdicts
- **(Class A) Faithfulness identical.** `Σ_s‖Δ_s‖²/Σ_s numel` over fp32 deltas (S17). Consumes
  fp32 `weight_deltas` = `W−(V@U).T` from chunk 3 → N2 fp32 path confirmed feeding faithfulness.
  `float` denom = int32-overflow fix, not a divergence. ✅ (closes parked thread)
- **(Class A) imp-min `L_p`, eps, annealing faithful + consumes `ci.upper`.** `(ci+eps)^pnorm`,
  eps=1e-12, `lp=Σ_sΣ_c f_c`. `ci_upper` = torch `ctx.ci.upper_leaky` ✅ (closes parked thread).
  `_linear_anneal` ≡ torch `_get_linear_annealed_p`. ✅
- **❌→deliberate: imp-min FREQUENCY term diverges from main (AWAITING OLI CONFIRM).** Two
  intentional changes: (1) **global vs per-rank reduction** — jax `jnp.sum` = global batch sum
  under GSPMD, `f_c` = true global frequency inside the convex `log2`; torch used per-rank `f_c`
  + `log2(1+per_rank_sum·world_size)` then DDP-averaged (Jensen-biased multi-rank). IDENTICAL at
  world_size=1; jax strictly more correct multi-rank. (2) **`a'`=reference_token_count decoupled
  from B·T** (batch-invariant reparam, SPEC S8′, lore project_impmin_scaling). Exact recovery of
  old torch under `a'=B·T`, 1 rank, `freq_coeff=imp_coeff·beta`. Form is a faithful
  generalization; value-level it's an intentional algo change. **CONFIRMED INTENDED by Oli
  (2026-06-30): the frequency reference batch size is deliberate.** ✅
- **(Class C) smooth-L0 (Geman–McClure) is a jax ADDITION**, not in torch. Nothing to reconcile.
- **(Class C, verify train) `annealed_pnorm` asserts `p_anneal_final_p is not None`** whereas
  torch returns `initial_p` when None (no-anneal default). train.py must guard no-anneal case.

**Verdict: faithfulness ✅; imp-min L_p/anneal/upper ✅; imp-min FREQUENCY = deliberate
divergence pending Oli's confirm; smooth-L0 = added capability.**

---

## Chunk 6 — reconstruction (the "very differently implemented" one)

**JAX**: `recon.py` (source strategies, `ReconForward`/`ReconLossTerm`/`LossSurface`, routing
samplers, chunking helpers, `make_plan`, `build_loss_terms`)
**main**: `metrics/{ci_masked,unmasked,stochastic,pgd_masked,persistent_pgd}_recon{,_layerwise,_subset}.py`,
`masks.py`, `batch_and_loss_fns.py` + lab `recon_loss_kl`
**spec**: §4.3, S2/S10/S10′/S11

The torch loss-class cartesian product = **chunking × routing × source strategy**. Every torch
class maps to a JAX plan with identical graph (verified CIMasked, Unmasked, CIMaskedLayerwise
line-by-line; others by the same factoring).

### Class-A verdicts (all confirmed)
- **KL direction & reduction identical.** Both `KL(clean‖masked)=Σ p_clean(log p_clean−log
  q_masked)`, sum over positions ÷ n_pos. torch `recon_loss_kl(pred=masked,target=clean)` =
  `F.kl_div(log_softmax(masked), softmax(clean), "sum")`. ✅
- **Normalization identical.** Mean over ALL (forward×position): jax `kl_per_position` then mean
  over draws/entries ≡ torch `Σ sum_loss / Σ n`. ✅
- **`live`-sites gate faithful** (CLOSES parked thread). jax `live_sites` ↔ torch
  `mask_infos={subset}`: only listed sites decomposed (lax.cond), rest frozen `x@W`. per_site ↔
  *Layerwise (one site live, mean over sites). ✅
- **`has_delta` gate faithful** (CLOSES parked thread). `ConstantSources` → no delta = torch
  `weight_deltas_and_masks=None` for CI-masked/unmasked. ✅
- **uniform-k routing distributionally correct.** jax single-argsort `perms<k` vs torch
  double-argsort `rank<k` — both uniform random k-subset; `k~U{1..n}` matches. ✅
- **chunking map**: one_chunk↔full, per_site↔layerwise, into_groups↔chunkwise/subset. ✅
- **source map**: ConstantSources(0)=CImask, (1)=unmask; StochasticSources=U[0,1] comp+delta;
  Fresh/Persistent PGD plan-level only → ascent verified in adversary chunk. ✅

### Notes
- **(Class C) `build_loss_terms` requires exactly 1 faith + 1 imp + ≥1 recon.** torch accepted
  arbitrary metric lists. Scope narrowing to production objective.
- **(spec-overfit, 🔧) SPEC §4.3 under-describes** — only enumerates stochastic + adversarial;
  real surface is the chunking×routing×source factoring (in LOSS_PARITY_DESIGN.md, not SPEC).
  Amend SPEC §4.3 to the general factoring.
- **(Class B trivial) unmasked mask** `ci+(1−ci)·1.0` vs torch literal `1.0`: ≤1 ulp.

### Deferred to adversary chunk
- FreshPGDSources + PersistentSources ascent mechanics (sign-PGD, projection, persistent state).

**Verdict: ✅ recon graph-faithful (refactor preserves per-class semantics exactly). PGD source
ascent pending adversary.py.**

---

## Chunk 7 — adversary (PGD sources) — IN PROGRESS

**JAX**: `adversary.py` + the producer side in `train.py` (`step`, the fused `value_and_grad`,
warmup/fresh ascents, `final_ascend` routing)
**main**: `metrics/persistent_pgd_state.py`, `metrics/persistent_pgd_recon.py`, `metrics/pgd_*`
**spec**: §4.4, S12/S13/S14/S15/S23/S24/S32
**theory**: paper §recon-motivation — adversarial (worst-case) ablation for robust ablatability.

### Understood deeply (with Oli, the gradient-sharing architecture)
- **One fused `value_and_grad` over `(prepared, components, ci, warmed_sources)`** produces param
  grads AND `∂L/∂source` in one backward. `prepared`/`ci` are computed ONCE outside with vjps
  captured (`recon_vjp`/`ci_vjp`) and composed after — avoids a 2nd forward of the ~10×-target
  CI fn and the ÷N→÷fsdp gather; also sequences those backwards AFTER the main backward (peak
  win). remat preserved (it's INSIDE the vjp'd fn via `remat=remat_ci_fn`).
- **Why this structure is forced**: the free final ascent (S14) needs `∂L/∂source` from the main
  backward → sources must be an INPUT LEAF → warmup must run BEFORE `value_and_grad` → warmup
  needs detached `ci`/`prepared` → vjp factoring to keep CI to one forward. (n_warmup=0 would
  collapse the whole thing to a single value_and_grad + grad-flip; but warmup is empirically
  important per Oli, so the complexity stays.)
- **Gradient hygiene (Class A) correct**: same mask `ci_lower+(1−ci_lower)·source` → `∂L/∂ci`
  descends CI fn, `∂L/∂source` ascends adversary, in ONE backward. Warmed sources
  `stop_gradient`'d vs the warmup scan but live as the value_and_grad leaf. Source is a constant
  leaf for the param update.
- **`÷coeff` exact via S23**: source feeds exactly one term, so `∂total/∂source = coeff·∂term/∂source`;
  `final_ascend` divides it out to ascend on raw KL. Primary reason = SCALE-CONSISTENCY with the
  warmup grads that share the same persistent Adam `m/v` (Adam is ~scale-invariant, so it's not
  about LR). Exact ONLY because one-source-one-term (S23 has teeth).
- **Choreography**: `n_warmup` paid Adam ascents (own forwards, route-all detached scoring loss,
  S24 quirk) + 1 free final ascent (reuses main backward). Fresh PGD = sign-ascent scan, own
  forwards, no state.

### Faithfulness confirmed
- **(Class A/B) Adam ascent is an EXACT mirror of torch `AdamPGDOptimizer.step`** — same moment
  recursions, bias correction, eps-outside-sqrt, `add_`=ascent. Defaults β1=.9/β2=.999/eps=1e-8.
  Both torch and JAX hand-roll it. ✅
- **(Class A) projection `clip(·,0,1)` every step** = S13/S15 (torch projects too). ✅

### Action items / notes
- **🔧 REFACTOR (deferred, post-parity): replace the hand-rolled source Adam with optax.** Main
  optimizers already use optax; only the adversary hand-rolls (to mirror torch's hand-roll, which
  torch needed because torch.optim couldn't feed an externally-supplied/cross-backward grad —
  fine in JAX/optax). Oli wants the proper optimizer pattern, NOT now; requires re-baselining the
  PPGD goldens (optax arithmetic ≠ torch bit-exact). Logged as an exercise outcome.
- **(Class C) scope narrowing**: torch has 4 persistent scopes (Single/Broadcast/Repeat/
  PerBatchPerPosition); JAX persistent = sc/bsc. Confirm mapping in scope check.

### Choreography faithfulness — CONFIRMED (torch `persistent_pgd_recon.py` + `_state.py`)
- **(Class A) JAX fused-backward + ÷coeff ≡ torch separate-backward on the unscaled live_loss.**
  Torch: `before_backward` does `get_grads(live_loss, retain_graph=True)` = `autograd.grad(live_loss,
  sources)` where `live_loss` is the RAW term recon loss (NO coeff; coeff applied only into
  total_loss); then `after_backward` Adam-steps. So torch's source grad is raw-by-construction via
  a SECOND backward. JAX gets `coeff·∂term/∂source` from the ONE fused backward and `÷coeff`s →
  same raw grad. The ÷coeff is exactly the compensation that makes JAX's one-backward shortcut
  equal torch's two-backward approach. ✅ Faithful (JAX cheaper by one backward).
- **(S24) warmup = route-all, all-sites, n_warmup paid ascents, params/CI not updated** — torch
  hardcodes `AllLayersRouter()` in `warmup` (line 323) regardless of subset; JAX
  `warmup_scoring_loss` routes None/all-sites. Matches. ✅
- **Param update uses WARMED sources; final source grad at PRE-UPDATE θ.** Matches. ✅
- warmup does n_warmup fresh forwards (torch loop / JAX scan); route-all makes n_samples
  redundant in warmup (identical draws). Matches.

### Flags
- **⚠️ (Class A, distributed — verify in sharding/multi-host):** torch `get_grads` does
  `all_reduce(g, AVG)` across ranks for the source grad. JAX relies on GSPMD to reduce the
  `sc`-shared source's gradient across batch shards. Confirm SUM-vs-AVG + that the cross-shard
  reduction actually happens. `bsc` = no cross-shard reduce (matches torch `_skip_all_reduce`).
- **(Class C) sigmoid parameterization**: torch `use_sigmoid_parameterization`; JAX refuses
  (clamp-only, deliberate).

### `source_masks` (S1) + C+1 memory — RESOLVED (measured)
- `mask = ci_lower + (1−ci_lower)·source[:-1]`; `delta = source[-1]`. Matches torch
  `interpolate_component_mask` + delta-channel split (`get_ppgd_mask_infos`). ✅
- **C+1 memory question settled by HLO/memory_analysis** (CPU, bsc shapes B=8,T=128,C=512):
  `temp_size = 1,050,624 B` = exactly ONE `(B,T,C+1)` bf16 buffer (the fp32→bf16 source cast).
  Both slices `[:-1]`/`[-1]` are FUSED into their consumers (`add_convert_fusion` / `convert_
  bitcast_fusion`), no standalone `(B,T,C)` copy, no `copy(` ops. The `C+1` packing + slices add
  ZERO extra buffers. (Correction to earlier: `source[:-1]` IS ci-aligned/ci-sized at bsc — the
  no-dup conclusion rests on the fusion, now verified, not on source being "small".) The lone
  temp is the N1 fp32-master cast (vanishes if sources stored bf16). Backend-agnostic (slice→
  elementwise fusion; bsc shards batch, slice is on C). ✅

### `_gate` / start_frac (S32) — REMOVED (Oli executive decision, 2026-06-30)
Was faithful to torch (compute-then-gate ≡ torch's early `return None`), but always the no-op
path (start_frac=0.0 everywhere). Oli decided the team won't miss it → REMOVED entirely:
`_gate` + `start_frac` field (`adversary.py`), the config field + docstring (`configs.py`),
the construction kwarg (`run_state.py`), the loss-gating block + comment (`train.py`), the
docstring mention (`recon.py`), SPEC row **S32 deleted** (ID gap left, not renumbered), and all
construction sites (invariance_check, 5 tests, compile_probe gen). Verified: basedpyright clean;
stacked-parity + equivalence + checkpoint tests pass (9 passed / 11 xfailed) → trajectory
byte-exact (start_frac=0 was the unguarded path, so removal changes nothing).

**Chunk 7 VERDICT: ✅ FAITHFUL.** Gradient-sharing architecture, Adam ascent (exact torch
mirror), choreography (÷coeff ≡ torch unscaled-live_loss backward), source_masks/C+1 (no dup,
measured), source ascent + projection — all confirmed. Carried forward:
- ⚠️ distributed source-grad AVG (sc scope) → verify in sharding chunk.
- 🔧 optax refactor for the source Adam → deferred post-parity cleanup (Oli wants it).

---

## Chunk 8 — train.py (the step, non-adversary plumbing)

**JAX**: `train.py` (`make_train_step`/`step`, helpers `masked_forward`/`constant_entry_masks`/
`entry_loss_for_sources`, the recon dispatch, `make_faith_warmup_step`) + `lm.py`
`stochastic_site_masks`/`run_stochastic_masked_output` + `targets/llama8b.py`
`_attach_per_kind_stochastic`/in-block draw
**main**: `optimize.py` (train loop), `metrics/stochastic_recon*`, `faithfulness_warmup.py`
**spec**: §4.5, S3, S12, S21

### Verdicts
- **(Class A) Stochastic recon faithful (formula), not bit-parity (RNG).** Generic
  `stochastic_site_masks` and the Llama in-block recompute BOTH = `mask=ci+(1−ci)·U[0,1]`,
  `delta=U[0,1]` — matches torch `calc_stochastic_component_mask_info` (`rand_like`/`rand`).
  In-block draws the source inside the checkpointed scan (deterministic recompute, same per-layer
  key fwd+bwd) so the mask is never held (memory win). Exact draws differ (jax keys vs torch RNG;
  the two jax paths even differ from each other) — inherent cross-framework, expected (xfail). ✅
- **(Class A) CHUNK-4 PARKED CHECK CLOSED: CI fn reads FROZEN-path activations, no V/U leak.**
  `taps = read_activations(batch)` runs the frozen forward (only frozen weights + batch, no V/U /
  CI params) and is computed OUTSIDE `loss_fn` (train.py:248); `ci`/`ci_vjp` close over fixed
  taps → recon-loss grad cannot leak into V/U via the CI inputs. ✅
- **(Class A) helpers faithful by construction**: `masked_forward`=`masked_output`+shard+remat
  (chunk 3); `constant_entry_masks`=`ci+(1−ci)·value` (chunk 6, CImask/unmask); both lean on
  verified pieces. ✅
- **Step order = SPEC §4.5**: clean (stop_gradient, S3) → CI envelope → warmup+fresh ascents →
  fused value_and_grad (faith + imp + recon dispatch) → vjp-composed param grads → final ascend →
  optimizer update → step+1. ✅
- **faith warmup (S21)** = minimize `‖W−V@U‖²` over components only (verified `faithfulness_loss`). ✅

### Notes / deferred
- batch-sharding pins (`batch_sharded`/`ci_shard`/`ci_batch_sharded`/`replicate_for_ascend`) →
  sharding chunk (perf/distributed, not algorithm).
- per-term RNG (`fold_in(key,1+i)` etc., SPEC R1): determinism/resume structure, not torch bit-parity.

**Verdict: ✅ FAITHFUL.**

**Verdict: IN PROGRESS — gradient architecture + Adam parity confirmed; masks/gate/choreography pending.**
