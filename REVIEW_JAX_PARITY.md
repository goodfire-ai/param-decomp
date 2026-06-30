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
