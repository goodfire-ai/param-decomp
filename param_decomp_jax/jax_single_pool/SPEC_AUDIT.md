# SPEC_AUDIT.md — rigor pass on the single-pool VPD training-step math

> Historical analysis from the torch-parity phase; superseded by `TRANSITION.md` (the fully-JAX pivot). Kept for reference.

Independent re-derivation of the training-step **math** (bucket 1) from the **on-branch**
torch single-pool impl (`param_decomp/`, `param_decomp_lab/` @ `feature/jax`), cross-checked
against `SPEC.md` invariant-by-invariant and against the JAX impl (`param_decomp_jax/jax_single_pool/`).

**Scope note that frames everything below.** `SPEC.md` §1/§5 declares its ground truth to be
the torch impl on `feature/fsdp-lm-trainer` (the n-pool lineage), and §9's pointers name trees
(`three_pool/`, `param_decomp_lab/metrics/`) that **do not exist on this branch**. The actual
torch reference a JAX run on this branch must match is the **single-pool core** in
`param_decomp/` + `param_decomp_lab/`. I audited against *that*. Where the two diverge it is
called out (this is itself a SPEC-STALE class of finding: the spec anchors numerics on an
off-branch lineage).

Verdict vocabulary:
- **CONFIRMED** — torch (this branch) + JAX + SPEC all agree.
- **SPEC-STALE** — SPEC is wrong, outdated, or anchored on the wrong reference vs on-branch torch.
- **JAX-DEVIATES** — JAX differs from on-branch torch *semantics* (not just RNG bits / reassociation).
- **SPEC-SILENT** — a real on-branch torch behavior that no invariant captures.

Each verdict carries a primary evidence pointer I read directly (✓ = spot-checked in this pass).

---

## 1. Per-invariant verdicts

### S-series (semantic invariants)

| id | verdict | evidence (file:line) | justification |
|---|---|---|---|
| **S1** `mask = ci+(1−ci)·source`; delta channel raw, never ci-interp; CI has no delta output | **CONFIRMED** | torch `masks.py:174` ✓ (`ci + (1-ci)*source`), `masks.py:236` ✓ (delta mask = raw `torch.rand`, no interp); jax `adversary.py:134`, `train.py:198,214` | Interpolation identical; delta is the trailing/separate raw source on both sides. C+1 channel layout matches. |
| **S2** non-live site runs frozen `x@W` (zero V/U grad, no `(B,T,C)` acts) | **CONFIRMED** | torch `component_model.py:369-377` (where-fallback) + single-key `make_mask_infos` (omitted sites → frozen); jax `llama8b.py:363-379` (`live` tuple) | Two realizations (per-position route-False; site not in live-set) both target frozen `x@W`. On-branch single-pool only ever uses per-position routing (never a fully-non-live site); the static `live`-set path is exercised only by `subset_chunk_plan`. See risk R-2. |
| **S3** recon target = frozen-path forward under sg, NOT `mask=1` identity | **CONFIRMED** | torch `train_step.py:141,150` (`cache_type='input'`, `mask_infos=None`); jax `llama8b.py:319-326`, `train.py:249` | Both stop-grad a true all-frozen forward. Structural nuance: torch single-pool computes the WHOLE-model frozen forward; JAX uses residual-start suffix-only. Equivalent under frozen+sg prefix. See S18/SPEC-SILENT. |
| **S4** CI inputs = clean site inputs from the frozen path | **CONFIRMED** | torch `train_step.py:141-146` (cached pre-weight acts); jax `llama8b.py:329-360` | MLP: gate_in=up_in=post-ln2, down_in=silu(gate)·up; attn: q/k/v_in=post-ln1, o_in=pre-o attn out. Both on the frozen residual. |
| **S5** `ci_lower`/`ci_upper` two squashings of the SAME logits; lower→masks, upper→imp-min only | **CONFIRMED** | torch `ci_sigmoids.py:82-87` + caller; jax `ci_fn.py:118-131`, `train.py:368,370` | Single `logits` tensor; lower feeds all masks, upper feeds imp-min only. No crossing. |
| **S6** squashings' fwd/bwd exactly §4.2 incl. lower_leaky's grad-sign-gated custom VJP | **CONFIRMED** ✓ | torch `ci_sigmoids.py:24-53` ✓ (custom `Function`, nested `where`: `x<=0` → `α·g` iff `g<0`); jax `ci_fn.py:34-48` | Read the torch custom backward line-for-line: `where(x<=0, where(g<0, α·g, 0), where(x<=1, g, 0))`. JAX `_lhs_b` is bit-identical including the `<=` tie-breaks. Upper is ordinary autodiff of the §4.2 expr on both. **Risk:** zero fixture/grad-test coverage of this VJP anywhere (R-5). |
| **S7** imp-min groups per site; `log2(1+sum)` consumes ONE site's per-component sum | **CONFIRMED** ✓ | torch `importance_minimality.py:55-74` ✓ (per-`layer_name` loop, no merge); jax `losses.py:42-49` | Both accumulate per-site `lp`/`entropy` into running scalars; no site merging. Convexity preserved. |
| **S8** per-component sums over GLOBAL batch, accumulated BEFORE log2 | **CONFIRMED** ✓ | torch `importance_minimality.py:161-167` ✓ (autograd-aware `dist_fn.all_reduce(SUM)` on sums, `n*world_size`, then `finalize`); jax `losses.py:45` (`jnp.sum` over global `(0,1)` axes) | Torch SUM-reduces sums across ranks inside the log2 with the autograd-aware all_reduce; JAX gets the global sum from GSPMD over the dp-sharded `(b,t)` axes. Mechanism differs, semantics identical. |
| **S9** `pnorm` anneals linearly 2.0→0.4 over frac window; eps inside the power | **CONFIRMED** ✓ | torch `importance_minimality.py:16-37` ✓ (3-branch clamp), `:49` ✓ (`(ci+eps)**pnorm`); jax `losses.py:45,52-58` | Same boundary behavior (`frac<start`→initial, `≥end`→final, linear between); eps inside `(·)^p` on both. **JAX narrowing:** JAX *asserts* `p_anneal_final_p is not None` (`train.py:122`), refusing torch's constant-p config (`importance_minimality.py:24` returns `initial_p`). Unreachable in production; spec-silent on it → see "new invariants". |
| **S10′** static tuple of coefficiented recon TERMS; each term = static plan of `(live_sites, SAMPLE_ROUTING, MASK_SOURCE)`; term loss = mean-over-forwards of kl; total = Σ coeff·term | **CONFIRMED** | torch per-Metric `(Σ kl, Σ n)` → `sum/n`; jax `recon.py:282`, `train.py:415-418` | The normalization identity `Σkl/Σn == mean-over-forwards of kl_per_position` holds because every forward shares `(B,T)` (LOSS_PARITY_DESIGN §4e). JAX `build_recon_terms` maps each shared config → one term. **Caveat:** the production stochastic term's torch Metric (`ChunkwiseSubsetReconLoss`) is **not** a runnable Metric on this branch (only its config exists, not in `LOSS_METRIC_CLASSES`) — its parity reference is off-branch. See risk R-1. |
| **S11** `uniform_k_routing`: `k~U{1..|live|}` then uniform k-subset; fresh per step | **CONFIRMED** | torch `masks.py` (`UniformKSubsetRouter`); jax `recon.py:192` | Distribution matches (torch double-argsort, JAX single-argsort < k both yield uniform k-subset). R2 permits the RNG-bit divergence. |
| **S12′** adversarial term consumes sources as LEAVES; grad → components + (via ci_lower) ci_fn + (persistent) the leaves; production term masks ALL sites route-all | **CONFIRMED** | torch `persistent_pgd_recon.py:155-168` (sources as leaves), `pgd_masked_recon.py:42` (fresh: sources detached for main loss); jax `train.py:303,356,436` | Persistent sources are live leaves in the fused backward; fresh sources are stop-gradient'd into the main loss. Production routes all sites everywhere. |
| **S13′** per persistent term: `n_warmup+1` updates through THAT term's SRC_STEP state; source LR advances once per training step | **CONFIRMED** ✓ | torch `persistent_pgd_state.py:329` ✓ (warmup loop ×`n_warmup`) + `persistent_pgd_recon.py:225` (after_backward = +1), `:142` (`update_lr` once, skipped in eval); jax `train.py:263-294,438` | n_warmup supplemental ascents + 1 fused-backward ascent = n_warmup+1; LR stepped once per step. |
| **S14′** final ascent grad from SAME graph as main backward, pre-update params, UNSCALED by term coeff | **CONFIRMED** ✓ | torch `persistent_pgd_recon.py:217` ✓ (`get_grads(live_loss=sum/n, retain_graph=True)` — un-coeffed) called in `train_step.py:263` BEFORE `:265` backward; jax `train.py:441` (`/coeff`) | Torch takes a SECOND autograd.grad on the un-coeffed per-term loss; JAX reuses the fused total grad and divides by coeff (`∂(coeff·L)/∂s ÷ coeff = ∂L/∂s`). Algebraically identical; JAX is actually *stricter* (literally the same graph, can't use post-update params). Requires S23. |
| **S15** every source update ends with PROJ (clamp [0,1]); init U[0,1] | **CONFIRMED** ✓ | torch `persistent_pgd_state.py:126` (`source.add_(...)`) + clamp in `step`, `:220` (`init_fn = torch.rand` for clamp param); jax `adversary.py:111-112,48` | Clamp after every ascent (warmup + final + fresh sign steps); clamp-param init is U[0,1]. |
| **S16** shared-scope sources replica-identical (identical init+updates); per_batch_per_position shards with batch | **CONFIRMED** | torch `persistent_pgd_state.py:189-194,229` (assert + broadcast init), `reduce_source_grads` AVG; jax `llama8b_sharding.py:113` (replicated leaf), `adversary.py:36` | Torch maintains identity by broadcast-init + deterministic AVG-grad + identical step. JAX gets it structurally (replicated leaf via `out_shardings`). JAX implements **sc only** (S-SILENT/JAX-DEVIATES partial — see §1 Scope). |
| **S17** faithfulness = global mean sq-delta over all params (Σ‖Δ‖²/Σnumel), from live V/U each step | **CONFIRMED** ✓ | torch `faithfulness.py:14-25` ✓ (`Σ(delta**2).sum() / Σ numel`); jax `losses.py:21-28` | Identical. Equivalence fixture `faith=0.10001952` matches at rtol 2e-4. fp32 deltas on both (N2). |
| **S18** fresh batch per step; prefix harvest per batch, outside every grad graph | **CONFIRMED** | torch `optimize.py:386-392` (loader), `train_step.py` clean fwd under sg; jax `run.py:238,291`, `train.py:249` | Both consume a fresh batch and stop-grad the prefix. JAX uses residual-start (suffix-only); torch single-pool uses whole-model frozen forward. Data **orderings differ** (na-by-design). |
| **S19** components grads global-norm-clipped at 0.01 before step; CI unclipped | **CONFIRMED (math)**, **SPEC-SILENT (eps)** ✓ | torch `optimize.py:489-494` ✓ (`clip_grad_norm_`, ci clip only if `grad_clip_norm is not None` → None in prod); jax `run_state.py:34` (`optax.clip_by_global_norm`) | Clip target/timing match. **Numeric seam:** torch `clip_grad_norm_` uses `clip_coef = clamp(max_norm/(total_norm+1e-6), max=1)`; optax uses `max_norm/max(norm, max_norm)` with **no +1e-6**. At clip=0.01 the clip almost always fires → ~1e-4 relative scale difference on component grads **every step**. SPEC/S19 silent on the eps. See risk R-3. |
| **S20** both opts AdamW wd=0, betas(0.9,0.999), eps1e-8; LR cosine→0.1×, no warmup, stepped per step | **CONFIRMED (opt)**, **SPEC-SILENT (cosine convention)** ✓ | torch `schedule.py:52,60-64` ✓ (`progress=(step-warmup)/(decay_steps-1)`); jax `run_state.py:31-35` (`cosine_decay_schedule(lr,steps,alpha=0.1)`) | AdamW hyperparams match exactly (torch relies on its 1e-8 default eps; JAX sets it). **Cosine off-by-one:** torch divides by `decay_steps-1` (reaches 0.1× at `step=steps-1`); optax divides by `steps` (reaches 0.1× at `count=steps`, one update past the last). Per-step LR delta ~O(1/steps)~2.5e-6 at 400k. Genuine formula divergence; S20 doesn't pin the convention. See risk R-4. |
| **S21** faith warmup (400×AdamW lr1e-3 on faith alone) precedes step 0; opt discarded | **CONFIRMED** ✓ | torch `faithfulness_warmup.py:34-51` ✓ (separate AdamW, discarded), `optimize.py:394-395` (only at step 0); jax `run.py:197-224`, `train.py:494-508` | Both run only on fresh start, discard the opt, re-init the VU optimizer. **JAX narrowing:** torch warmup honors `faithfulness_warmup_weight_decay` (config, default 0); JAX hardcodes wd=0 and *asserts* torch's is 0 (`torch_config.py:235`). A non-zero warmup wd is refused, not silently dropped — acceptable. |
| **S22** checkpoints round-trip ALL §3 trajectory state incl. persistent sources + moments + step | **CONFIRMED** | torch `persistent_pgd_state.py:133-147,284-296` (state_dict m/v/step_count + sources); jax `train.py:70-73`, `adversary.py:28-33` | Both round-trip sources + (m,v,step_count) + opt state + step; frozen target excluded (rebuilt). Subtleties (step_count int vs f32 array; topology-independence mechanism) are impl details, not math. |
| **S23** a persistent source bundle feeds exactly ONE term (coeff-unscaling validity) | **CONFIRMED** | torch (one Metric instance owns its sources); jax `train.py:130` (`set(state_keys)==set(persistent)` assert) | Torch keys sources by Metric instance; JAX by `state_key`, asserted 1:1. The S14′ `/coeff` is exact only under this. |
| **S24** persistent WARMUP routes ALL regardless of loss plan; fresh-PGD draws routing ONCE per step | **CONFIRMED** ✓ | torch `persistent_pgd_state.py:321-334` ✓ (hardcoded `AllLayersRouter()` + explicit WARNING), `pgd_utils.py:156` (fresh: one `get_masks` reused); jax `train.py:273-275,314-316,382` | Read the torch warmup: it ascends against route-all even for the Subset variant (with a code WARNING that this is undocumented PR-#486 inheritance). JAX replicates (warmup forwards `routes=None`). Fresh-PGD single routing draw matched via `fixed_routes`. **Explicitly provisional** per SPEC + CLAUDE.md "pending team decision." See risk R-6. |

### N-series (numerics)

| id | verdict | evidence | justification |
|---|---|---|---|
| **N1** masters fp32; moments fp32; forward/frozen may be bf16 | **CONFIRMED** | torch `components.py:71` (fp32 Params), `optimize.py` AdamW; jax `ci_fn.py:10-12`, `train.py:367-368`, `run_state.py:35` (wd=0) | V/U + CI-fn fp32 masters, both AdamW moment sets fp32, SRC_STEP moments fp32; bf16 compute. CLAUDE.md A7: optax default wd 1e-4 vs torch 0 → explicitly set 0. |
| **N2** faith deltas `W−V@U` fp32 outside autocast; masked-PATH delta may be bf16 | **CONFIRMED** | torch `train_step.py:222-223` (`calc_weight_deltas()` OUTSIDE autocast), `components.py:154-160`; jax `train.py:369`, `llama8b.py:437-443` (fp32), `:301-305` (bf16 path) | Torch forms the faith delta in fp32 outside autocast; the on-PATH delta is bf16. JAX mirrors: `weight_deltas_fp32` casts to fp32; the masked-path delta is bf16. The bf16 path-delta divergence is documented and out of scope for faithfulness. N2 wording accurate. |
| **N3** kl + imp-min reductions fp32; loss scalars + grad accum fp32 | **CONFIRMED (training)**, **SPEC-SILENT (bf16-input asymmetry)** | torch `importance_minimality.py:49-52` (pow/sum), `batch_and_loss_fns.py:61` (KL); jax `losses.py:12-18,43-45` | Both reductions land in fp32. **Subtle seam SPEC omits:** torch forms `(ci_bf16+eps)**p` as a **bf16 intermediate** (autocast leaves elementwise pow in input dtype) *then* sums in fp32; JAX casts ci→fp32 BEFORE the power. Small per-component-sum rounding difference on the imp-min input, not covered by the fp32-only equivalence fixture. N3 says "the imp-min reduction is fp32" but is silent on the bf16-input intermediate. Accept-as-seam (fp32 masters dominate). **Bounded** by `tests/equivalence/test_imp_min_bf16_seam.py`: identical bf16 ci into both reductions, worst-case rel error lp 2.06e-4 / entropy 2.76e-4 (uniform-[0,1] ci, pnorm 2; tol 1e-3). |

### D-series (data-parallel contract)

| id | verdict | evidence | justification |
|---|---|---|---|
| **D1** per-shard means + averaged grads compose to global; shared-source grad AVG (not SUM) | **CONFIRMED** | torch `persistent_pgd_state.py:243-253` (`reduce_source_grads` AVG), `pgd_utils.py:76-79` (fresh c-scope AVG); jax `train.py:362,438`, `llama8b_sharding.py:113` | Torch AVG-reduces shared-source grads; JAX gets AVG from GSPMD autodiff of the global-mean loss on a replicated leaf (kl_per_position normalizes by global B·T). This is the historical SUM-bug spot (MEMORY). **Risk R-7:** the JAX AVG is implicit (not an explicit reduce) — confirm via HLO/invariance_check that the cotangent into the replicated source leaf is MEAN not SUM. |
| **D2** imp-min exact global per-component sums inside log2; autograd-aware | **CONFIRMED** ✓ | torch `importance_minimality.py:161-167` ✓; jax `losses.py:45` | Exactly the S8 mechanism. Torch's `dist_fn.all_reduce` is autograd-aware; JAX's reduction is in the differentiated graph. Identical. |
| **D3** shared PPGD sources stay replica-identical (init+grad+step) | **CONFIRMED** | torch `persistent_pgd_state.py:229-233`; jax `llama8b_sharding.py:130-132` | Torch broadcast-inits; JAX inits via jit with `out_shardings=replicated` so XLA produces the same value on every device. JAX never needs the broadcast. |
| **D4** trajectory invariant to device count up to float reassociation; counter-RNG identical across layouts | **CONFIRMED** | jax `run.py:294` (`fold_in(run_key, step)` identical across procs); `experiments/invariance_check.py` | JAX's counter-based RNG satisfies a *stronger* invariance than torch's R3 (torch deliberately diverges per-rank streams). Spec-blessed deviation (D4 states it). |

### R-series (randomness)

| id | verdict | evidence | justification |
|---|---|---|---|
| **R1** every draw independent across sites/positions/forwards/steps | **CONFIRMED** | torch `masks.py:225-227,236`; jax `recon.py:148`, `train.py:178` | Independent stochastic sources + delta masks + routing per draw on both. |
| **R2** RNG stream order/bits need not match torch | **CONFIRMED** | (definitional) | JAX uses Threefry counter keys; torch uses its global PRNG. Distributions match; bits don't. As intended. |
| **R3** draws over a sharded batch independent across ranks | **CONFIRMED (torch) / JAX stronger** | torch `distributed.py:198,206` (`seed_per_rank`); jax `run.py:294` | Torch diverges streams per rank (R3 literal). JAX makes draws *identical* across layouts (D4) — a deliberate, spec-acknowledged strengthening. Not a deviation. |

---

## 2. (a) NEW semantics that should become invariants

These are real, load-bearing on-branch torch behaviors with no current invariant, surfaced and verified in this pass. Each is a candidate `S`/`N` addition.

1. **Recon KL direction is fixed: P=clean, Q=masked, `KL(P‖Q)`.** Torch `recon_loss_kl` uses `pred=masked, target=clean` with `F.kl_div(log_softmax(pred), softmax(target), 'sum')` ⇒ `Σ p_clean·(log p_clean − log q_masked)`; JAX `losses.py:18` computes the same direction. §4.3 writes `KL(softmax(clean) ‖ softmax(masked))` in pseudocode but no numbered invariant pins the *direction* as semantic. Make it one — getting it backwards is a silent, plausible bug.

2. **The recon normalization identity** (`Σ sum_kl / Σ n_positions == mean-over-forwards of kl_per_position`, valid iff all forwards share `(B,T)`). This is the linchpin that lets JAX's "mean over forwards" equal torch's accumulator; it is stated in LOSS_PARITY_DESIGN §4e but not as a SPEC invariant. It deserves one because it is the *condition under which the whole multi-term factorization is parity-correct*.

3. **eps inside the AdamW-source denom is `sqrt(v_hat)+eps` (after the sqrt).** Verified `persistent_pgd_state.py:125` and `adversary.py:114`. A classic parity trap (eps-before-vs-after sqrt). §6 names `adam(β₁,β₂,ε)` as a variation point but pins no eps placement. Add to N1 or §6.

4. **`inv_freq` RoPE buffer is stop-gradient'd in the CI fn.** Torch registers it as a buffer (never a Param); JAX must explicitly `stop_gradient` it (it's a pytree leaf and would otherwise be optax-updated). CLAUDE.md flags it "sharp teeth" but SPEC has no numbered invariant. Add one.

5. **JAX deliberately collapses imp-min to annealing-required** (`p_anneal_final_p is not None` asserted; constant-p must be expressed as `final_p==pnorm`). S9 describes annealing as always-on; it should record this narrowing so a torch constant-p config is known to be refused, not silently approximated.

6. **The `<=` tie-break structure of `lower_leaky_hard`'s backward is load-bearing** (lower branch wins ties at x=0; `g<0` gate). §4.2/S6 describe the leak but the tie-break at the boundaries is exactly what makes torch and JAX bit-identical — worth pinning explicitly given there is *no test* (R-5).

## 2. (b) Prioritized SPEC.md amendments

**P0 — fix the reference anchor (SPEC-STALE, affects every numeric claim).**
§1 and §5 declare ground truth = `feature/fsdp-lm-trainer` and §9 points at trees absent on this branch (`three_pool/`, `param_decomp_lab/metrics/`). The actual on-branch torch reference is the single-pool core (`param_decomp/metrics/`, `param_decomp/train_step.py`, `param_decomp/optimize.py`). Re-point §9 at the **on-branch** files (all verified above) and state explicitly that the production stochastic term's runnable torch Metric (`ChunkwiseSubsetReconLoss`) lives off-branch (parity fixtures pin it; no on-branch Metric counterpart). Without this, every "torch parity" claim cites a tree the reader can't open.

**P1 — pin the two undocumented numeric conventions (SPEC-SILENT, real per-step deltas).**
- Amend **S20** to pin the cosine convention: torch `progress=(step−warmup)/(decay_steps−1)` vs optax `count/steps`. Declare one canonical (recommend optax's, and note torch reaches 0.1× one update earlier). ~2.5e-6/step at 400k, but it is a genuine formula divergence the spec should own.
- Amend **S19** to record the grad-clip eps: torch `max_norm/(norm+1e-6)`, optax `max_norm/max(norm,max_norm)` (no eps). At clip=0.01 the clip fires almost every step ⇒ ~1e-4 relative component-grad difference *each step*. Either pin optax's form as canonical or add the +1e-6 to the JAX clip.

**P2 — add the new invariants from (a)** (KL direction; recon normalization identity; AdamW eps-after-sqrt; inv_freq stop-grad; imp-min annealing-required; lower-leaky tie-break).

**P3 — amend N3** to acknowledge the bf16-input intermediate asymmetry in imp-min (torch `(ci_bf16+eps)**p` bf16 intermediate then fp32 sum vs JAX cast-to-fp32-first). Document as an accepted seam, not silence. **Done:** bounded by `tests/equivalence/test_imp_min_bf16_seam.py` (worst-case rel error lp 2.06e-4 / entropy 2.76e-4).

**P4 — record the warmup_pct==0 source-LR edge** (S13). torch short-circuits to full LR at step 0 (`warmup_steps=0`); JAX `max(floor(...),1)` yields LR=0 at step 0. Diverges for exactly one step only when `warmup_pct==0`; production uses 2.5%. One-line note or a JAX fix to match torch's short-circuit.

**P5 — add a one-line S3 note** that `clean_output` = suffix-only clean forward when residual-start is active (the on-branch torch single-pool reference is NOT residual-start; equivalent under sg/frozen prefix). Tightens the contract wording.

## 2. (c) Genuine jax-vs-torch correctness risks surfaced

Ordered by stakes. None is a confirmed bug; all are unguarded seams worth a targeted check before relying on the affected path at scale.

- **R-1 (production parity has no on-branch torch fixture).** The JAX production stochastic recon term (`ChunkwiseSubsetReconLoss`) has **no runnable torch Metric on this branch** — only the config schema exists; it is absent from `LOSS_METRIC_CLASSES`. Its parity reference is the off-branch n-pool lineage. So the single most important loss in production is validated against a tree not present here. *Action:* either vendor the lab Metric onto this branch for the equivalence harness, or have SPEC §9 state the production-term torch counterpart is off-branch by design.

- **R-2 / R-4 (cosine endpoint + clip eps).** Covered in P1. Both are per-step numeric deltas that accumulate over a 100k–400k-step trajectory; small but real and currently undocumented.

- **R-5 (the most subtle numeric in the codebase is untested).** `lower_leaky_hard`'s grad-sign-gated custom VJP, the weightless RMS, and the bidirectional RoPE in the CI transformer are **entirely absent from the equivalence fixture harness** (CI values are fed in pre-computed). A 1-input grad check (`g<0` vs `g>0` at `x<0`, `x∈(0,1]`, `x>1`) would close the highest-confidence gap. The CI transformer's two *known* numeric divergences (tanh vs erf GELU; rms eps 1e-5 vs torch finfo ~1.19e-7) are now UNIFIED with torch (#624/#625/#730 resolved "match torch": `approximate=False` + `CI_FN_RMS_EPS = finfo(fp32).eps`), so those ops no longer block torch→JAX CI-fn weight transfer or shift the alive set near the clamp boundary.

- **R-6 (S24 is parity-correct but provisional).** The PPGD warmup-routes-all-even-for-Subset quirk is faithfully replicated from a torch path that **carries its own WARNING** ("parameterize before relying on Subset PPGD"). Unreachable in JAX today (subset persistent refused, sc-only), but if torch ever fixes the warmup-routing quirk, JAX must follow — a coupled change to track.

- **R-7 (implicit DP source-grad AVG).** D1's shared-source AVG falls out of GSPMD autodiff in JAX rather than an explicit `reduce_source_grads(op=AVG)` like torch. This is the historical SUM-bug class. The reasoning (kl normalized by global B·T ⇒ replicated-leaf grad sums normalized partials = global mean) is sound but *implicit*. A targeted `invariance_check.py` assertion (or HLO inspection) at >1 sim device for sc-scope fresh/persistent PGD would de-risk it. Same applies to fresh-PGD c-scope: `sign(avg(g))==sign(sum(g))` so they agree, but no multi-device invariance test pins it.

- **R-8 (site-ordering convention).** On-branch torch derives site order from `named_modules()` traversal / `sorted()` (lexicographic over module-path strings); JAX forces canonical `(layer-asc, KIND_ORDER)`. These coincide on the production single-MLP-layer family but **diverge for multi-layer or mixed attn+mlp configs** — affecting BOTH the CI concat/split alignment AND the per-step RNG key-folding order (recon keys derive from term/site index). SPEC S10 says site order is "config and RNG-load-bearing" but doesn't pin WHICH order. Needs a spec decision + a multi-site equivalence fixture before any non-production topology is trained.

---

**Verification summary:** I directly spot-checked the highest-stakes claims against on-branch torch source — `lower_leaky_hard` custom VJP (`ci_sigmoids.py:24-53`), imp-min global-sum-inside-log2 with autograd-aware SUM all_reduce (`importance_minimality.py:55-74,161-167`), faithfulness `Σ‖Δ‖²/Σnumel` (`faithfulness.py:14-25`), cosine LR `(decay_steps-1)` denominator (`schedule.py:52`), the fused PPGD ascent reusing the un-coeffed `before_backward` grad (`persistent_pgd_recon.py:217` + `train_step.py:263-268`), warmup hardcoded route-all with its own WARNING (`persistent_pgd_state.py:321-334`), AdamW source step eps-after-sqrt (`persistent_pgd_state.py:125`), S1 interpolation (`masks.py:174`), torch `clip_grad_norm_` vs `optax.clip_by_global_norm`, and the JAX counterparts in `losses.py`/`run_state.py`. All merged-reader findings I relied on were corroborated. The document above is my deliverable.
