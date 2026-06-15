# `recon.remat_forwards` characterization

When does rematerializing the recon forwards (`jax.checkpoint` around each masked
suffix forward) help vs hinder? All numbers: B200 nodes, seq 2048, fp32 masters +
bf16 compute, jax 0.10.1. Memory numbers are AOT `mem_probe.py` temp-arena sizes
(`compile().memory_analysis()`), not noisy runtime peaks; add args+out (~43 GiB at
8 GPU, shrinks with mesh size) for the per-device total.

## Throughput — L18, C=24576, 8 GPU, per-rank batch 8 (jobs 50467/50468, steps 40–50)

| remat | step time | tok/s/GPU |
|---|---|---|
| on | 1.77 s | 9,266 |
| **off** | **1.54 s** | **10,640 (+14.8%)** |

The b512 semi-prod run (64 GPU, bl8, remat off) reproduces this: ~10.1k tok/s/GPU.

## Memory — temp arena GiB at 8 GPU

| config | bl | remat on | remat off | off − on |
|---|---|---|---|---|
| L18 (1 chunk, 3 sites, C=24576) | 4 | 44.5 | 56.6 | +12.1 |
| L18 | 8 | 79.0 | 102.6 | +23.6 |
| L18 | 16 | 125.8 | — (would not fit) | |
| L20–31 (12 chunks, 36 sites, C=8192), pre-pin | 1 | 67.2 | 71.2 | +4.0 |
| L20–31, pre-pin | 2 | 78.9 | 118.8 | +39.9 |
| L20–31, post-`batch_sharded_ci`-pin | 1 | 49.7 | not yet probed | |

(The multi-chunk pre-pin rows predate the `batch_sharded_ci` fix; subtract roughly
the ~17 GiB of per-consumer CI all-to-alls for post-pin estimates. The remat
*deltas* are the meaningful column.)

## Recommendation

**Default `remat_forwards: false` whenever the off-arena fits the device; flip on
only to buy batch.** Remat is purely a memory-for-compute trade here and the
recompute is expensive: each recon forward's backward re-runs a full masked suffix
forward, costing ~15% of step throughput at the production L18 shape.

- L18 at bl8 (the production semi-prod shape): off fits with margin → off.
- L18 at bl16: only remat-on fits 8-GPU nodes → on, or shard the batch wider.
- Multi-chunk at bl1: the arena is dominated by CI resharding + per-chunk logits,
  not recon activations — remat saves almost nothing (4 GiB pre-pin) and costs 12
  chunk-forward recomputes. Post-pin, off should fit easily and is the likely win;
  the live comparison leg (`jax-l20-31-b32-cmp32`, job 50583, remat on per its
  pinned config, 1.54 s/step) keeps remat on because its run-dir config is frozen.
- Multi-chunk at bl2: remat buys ~40 GiB — there it earns its keep.

Open: a multi-chunk remat-off throughput point (cheap 30-step smoke once the
comparison leg finishes; do NOT change the live leg's config).
