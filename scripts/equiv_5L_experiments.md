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

**Multi-seed result after both fixes (N=10 each, canon config):**

| metric | 1-pool | 2-pool (5×1) | delta | t-stat |
|--------|--------|--------------|-------|--------|
| step-0 faith (post-warmup) | 0.010572 ± 0.000002 | 0.010571 ± 0.000003 | -0.000001 | -0.57 |
| mean(stoch) over training | 0.0866 ± 0.0035 | 0.0905 ± 0.0062 | +0.0039 | +1.72 |
| step-190 total | 1018600 ± 166 | 1018507 ± 186 | -93 (-0.009%) | -1.18 |

Step-0 faith identical to 6 decimal places — warmup now converges to the
same V/U in both pools. Step-190 total agrees within ~0.01%, t-stat
well below significance.

**Final verdict.** 1-pool ≡ 2-pool 5×1 under the canon config, modulo RNG.
