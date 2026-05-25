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

## Next batch (pending submission)

- **2pool-nperblock2**: 5 LW × 2 ranks + 1 pool-B = 11 GPUs. Tests in-block all_reduce.
- **2pool-poolb2**: 5 LW × 1 + 2 pool-B = 7 GPUs. Tests pool-B DDP.
- **2pool-poolb4**: 5 LW × 1 + 4 pool-B = 9 GPUs.
- **2pool-1block-4r**: 1 LW × 4 + 1 pool-B = 5 GPUs. Single-block 2-pool.
- **2pool-long**: baseline but steps=1000 — convergence-shape comparison.
- **3pool-2blocks**: 2 LW × 2 ranks + 2 CI + 2 PPGD = 8 GPUs. Per-block fanout with no single-rank groups.
