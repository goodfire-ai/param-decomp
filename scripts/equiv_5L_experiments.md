# 5-layer GPT-2 equivalence-test experiment log

Each experiment runs the same `pd:`/`runtime:`/`target:`/`data:` config —
HFWeightsInVendored GPT2Simple on `openai-community/gpt2`, layers 0-4,
batch_size=24, steps=200. Only the pool topology varies.

The "1-pool" baseline is a single run repeated to confirm determinism. The
pool variants are compared against 1-pool via `metrics.jsonl`'s
`train/loss/total` at each train-log step.

## Tracking

| name | yaml | topology | n_gpus | nodes | job | run_dir | status | max rel_err |
|------|------|----------|--------|-------|-----|---------|--------|-------------|
| 1pool-baseline-A | equiv_5L_1pool.yaml | DDP=2 | 2 | 1 | 33484 | p-307f927d-cleanup → p-07073b1e | DONE | (baseline) |
| 2pool-baseline | equiv_5L_2pool.yaml | 5×1 LW + 1 pool-B | 6 | 1 | 33488 | p-3b22d0d4 | DONE | 8e-4 vs 1pool |
| 3pool-7r-single-rank | equiv_5L_3pool.yaml (v1) | 5×1 LW + 1 CI + 1 PPGD | 7 | 1 | 33489 | p-0ec0f476 | HUNG (single-rank groups) | n/a |
| 3pool-8r-single-rank | equiv_5L_3pool.yaml (v2) | 5×1 LW + 1 CI + 2 PPGD | 8 | 1 | 33490 | (partial) | FAILED at PG 5 | n/a |
| 3pool-1block-4r | equiv_5L_3pool.yaml (v3) | 1×4 LW + 2 CI + 2 PPGD | 8 | 1 | 33491 | (pending) | RUNNING | tbd |

## Active hypotheses

1. **Single-rank `dist.new_group` calls deadlock when combined with cross-pool
   broadcast groups** (each cross_pool_bcast_group is `[block_leader, *ppgd_ranks]`).
   Evidence: NCCL error referenced PG index 5 = `block_group_groups[1]` (single-rank `[1]`).
   Counterevidence pending: 3pool-1block-4r should bypass this entirely.

## 2026-05-25: Per-loss raw (num, den) aggregation fix + 7-way sweep (jobs 33498-33504)

**Logging bug identified.** The old cross-pool aggregator AVG'd per-rank
scalars across the pool. That's wrong when ranks own disjoint site sets (imp
is summed across the rank's owned sites — AVG gives ``global / n_blocks``)
or when the value is a ratio of two additive quantities (faith = ``sum_sq /
numel`` — AVG'ing the ratios is not the global ratio). Backprop was
unaffected.

**Fix.** Each pool's step function now emits raw additive ingredients
``_raw/<name>_{num,den}``. The aggregator SUMs them across the pool with a
``1 / n_per_block`` pre-scale (LW pool) that's equivalent to AVG-within-block
+ SUM-across-blocks. Final scalars on rank 0:

  ``faith_global = SUM(faith_num) / SUM(faith_den)``
  ``imp_global   = SUM(imp_num)``
  ``stoch_global = SUM(stoch_num) / SUM(stoch_den)``
  ``ppgd_global  = SUM(ppgd_num) / SUM(ppgd_den)``

Re-ran 7 variants with the new aggregator (all sharing git snapshot
``p-28abdc1b``):

| variant                     | run_dir    | total @ step 190 | rel_err vs 1pool |
|-----------------------------|------------|-----------------:|-----------------:|
| 1pool (DDP=2)               | p-562ace8e | 16087.61         | ref              |
| 2pool (5×1)                 | p-eff9f833 | 16075.22         | −7.7e-4          |
| 2pool-poolb2 (5×1, 2 PPGD)  | p-118728f7 | 16076.29         | −7.0e-4          |
| 2pool-nperblock2 (3×2)      | p-aa504a30 | 16161.50         | +4.6e-3          |
| 2pool-1block4r (1×4)        | p-fe08bd5e | 16240.18         | +9.5e-3          |
| 3pool (1×4)                 | p-af9f80d6 | 16236.74         | +9.3e-3          |
| 3pool-2blocks (2×2)         | p-d819abff | 16204.59         | +7.3e-3          |

**Verdict.**

* **faith and imp now match across all topologies within ~1%** (RNG noise from
  ``seed_per_rank`` differences). Aggregation is correct.
* **stoch shows a ~10% time-averaged gap** between 1-pool (~0.294) and all
  multi-pool variants (~0.315–0.330). Multi-pool variants cluster within 2%
  of each other. Probable cause: 1-pool draws stochastic masks for all 30
  sites in one ``calc_stochastic_component_mask_info`` call (shared RNG state),
  whereas multi-pool's per-site streaming layerwise draws masks independently
  per site. Different RNG paths sampled from the same distribution — not a
  correctness bug.
* **Step-190 total loss is within 1% across every topology.** Single-rank-block
  2-pool tracks 1-pool to 0.1%. Multi-rank-block topologies (DDP-within-block)
  drift up to 1%, dominated by mask-sampling RNG variance.

**Confidence assessment.** 2-pool and 3-pool produce optimization trajectories
equivalent to 1-pool modulo well-understood RNG variance. The earlier-observed
"3-pool drift" was logging artifact + mask-sampling noise, not algorithmic
bug. Ready to scale to longer GPT-2 XL runs.

## 2026-05-25 (canon config audit + faith scaling fix)

**Setup change.** All 7 equiv yamls re-ported to mirror ``gpt2_xl_full.yaml``
exactly: LR 5e-4/1e-4 cosine to 0.1× final, ``grad_clip_norm=0.01`` on
components, faith coeff 1e8, stoch coeff 50, ImpMin pnorm=2.0 with anneal to
0.3, PPGD adam (β1=0.5 β2=0.99), warmup=400 steps. Multi-seed sweep with
canon settings.

**Two new bugs surfaced**:

1. ``grad_clip_norm`` was silently ignored in 2-pool / 3-pool — the field was
   only implemented in ``param_decomp.optimize`` (1-pool). Implemented
   ``param_decomp.grad_clip.cross_pool_clip_grad_norm`` that all-reduces the
   sum-of-squares across the relevant pool group with ``/n_replicas``
   deduplication, then scales grads uniformly.
2. ``_faithfulness_loss`` divided by ``numel_owned`` (rank-local) not
   ``numel_global``. Single-pool's per-element gradient is ``∝ 1 / numel_global``;
   multi-pool's was ``n_blocks×`` larger, so the unclipped 400-step
   faithfulness warmup over-converged V/U. Step-0 faith differed by ~5% and
   training never caught up (step-190 total ~2% lower in 2-pool across all
   seeds, t=-377). Fix: divide by ``numel_global`` (computed once at
   runtime-build time from resolved decomposition targets) so per-element grad
   matches single-pool exactly.

**Multi-seed result after grad-clip + faith-scaling fixes (N=10 each, canon
config):** 1-pool ≡ 2-pool 5×1, delta = -0.009%, t = -1.18 (not significant).

## 2026-05-25 (RNG sync bug — DDP partners had different V/U init)

7-way sweep after the previous two fixes showed single-rank-block topologies
matching 1-pool to 0.01%, but multi-rank-block topologies (3×2, 1×4) still
drifting up to 1.4%. Diagnosed by logging the pre-clip global grad norm:
2-pool 1×4 reported 785 at step 0, vs 1-pool's 992 (20% lower).

**Root cause.** ``seed_all_ranks`` was never called before ComponentModel
construction in 2-pool / 3-pool. The single-pool trainer (
``param_decomp.optimize.Trainer.__init__``) does this explicitly so DDP
partners initialize V/U + CI fn from the same RNG state. Between
``set_seed(pd.seed)`` in ``_fresh_main`` and ``ComponentModel.__init__`` in
the multi-pool trainers, the dataloader build + distributed setup advance the
RNG by rank-dependent amounts, so partners initialized with different V/U.
The in-block grad all-reduce averages grads but cannot bring divergent params
back into sync — partners stayed at slightly different points in V/U space
through training, biasing the trajectory.

**Fix:** call ``seed_all_ranks(pd_config.seed)`` immediately before
``ComponentModel(...)`` in both ``TwoPoolTrainer.__init__`` and
``ThreePoolTrainer.__init__``, then ``seed_per_rank(pd_config.seed)``
afterwards (matches single-pool's order).

**Final 7-way (N=1, canon config, all 3 fixes):**

| variant | step-190 total | rel vs 1pool |
|---------|---------------:|-------------:|
| 1pool DDP=2 | 1018578 | (ref) |
| 2pool 5×1 | 1018717 | +0.014% |
| 2pool poolb2 | 1018716 | +0.014% |
| 2pool nperblock2 (3×2) | 1018781 | +0.020% |
| 2pool 1block4r (1×4) | 1018512 | -0.007% |
| 3pool 1×4 | 1018505 | -0.007% |
| 3pool 2blocks (2×2) | 1018411 | -0.016% |

All within ±0.02% of 1-pool. Bias direction is RNG-consistent (mixed signs,
not systematic).

**Final multi-seed verdict (N=10, 1pool DDP=2 vs 2pool 5×1):**

| metric | 1-pool | 2-pool 5×1 | delta | t |
|--------|--------|------------|-------|---|
| step-0 faith (post-warmup) | 0.010572 ± 2e-6 | 0.010572 ± 2e-6 | -1e-7 | -0.10 |
| step-190 total | 1018600 ± 166 | 1018557 ± 142 | -43 (-0.004%) | -0.63 |

t-stat well below 1 → statistically indistinguishable. Warmup converges to
identical V/U; training trajectory is equivalent up to RNG variance.

**Summary of bugs surfaced and fixed:**

1. Per-loss aggregator AVG'd per-rank scalars across pool, wrong for
   disjoint-site sums and ratios. → raw `(num, den)` + cross-block SUM.
2. `grad_clip_norm` silently ignored in 2-pool / 3-pool.
   → `cross_pool_clip_grad_norm` with `/n_replicas` dedup.
3. `_faithfulness_loss` divided by `numel_owned`, not `numel_global`.
   → divide by `numel_global` so per-element grad matches single-pool.
4. RNG diverged across DDP partners before V/U init.
   → `seed_all_ranks` before ComponentModel construction.

**Ready to scale to GPT-2 XL.**

## 2026-05-25 (multi-node equivalence — jobs 33757-33771)

`compute_utils.py` from main was lost in the modular refactor. Restored as
``param_decomp_lab.infra.slurm.torchrun_command`` (1-GPU / single-node DDP /
multi-node DDP under one function), with ``CUDA_FLAGS`` plumbing.

N=5 seeds × 3 cohorts on the 5L canon config (batch=64 to divide both 16
and 4):

| cohort | step-0 faith | step-190 faith | step-190 total |
|--------|--------------|----------------|----------------|
| 1pool DDP=16 (2-node) | 0.010572 ± 2e-6 | 0.010186 ± 2e-6 | 1018583 ± 156 |
| 2pool 6×2+4PPGD (2-node) | 0.010571 ± 1e-6 | 0.010185 ± 1e-6 | 1018533 ± 115 |
| 3pool 5×2+2CI+4PPGD (2-node) | 0.010572 ± 1e-6 | 0.010185 ± 1e-6 | 1018558 ± 139 |

Pairwise vs 1pool-mnode (Welch t):
* 2pool: delta=-50.14 (−0.005%), t=-0.58 — NS
* 3pool: delta=-25.47 (−0.003%), t=-0.27 — NS

Faith trajectories track within 1e-6 across all three pools at every logged
step. ``seed_all_ranks`` works correctly across nodes (step-0 faith identical
to 6 decimals after 400-step warmup).

(1pool DDP=2 at batch=64 OOMs — pre-existing memory issue in
``_stochastic_recon_layerwise_loss_update`` which accumulates 30 forward
graphs before backward at single-pool scale. Independent of multi-node.)
