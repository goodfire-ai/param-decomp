# full32L HSDP perf — synthesized into Lore

The full perf investigation (V/U reconstruction hoist → buffer donation → b128/4-seq-per-GPU
at ~3.6× the b32 throughput) has been distilled into Lore. This file is intentionally short:
a 1100-line chronological trail of hypotheses, corrections, and dead-ends is not worth carrying
on the trunk.

**Canonical understanding** (mechanisms, exact memory anatomy, throughput, open levers):
lore `2026-06-29--full32l-hsdp-donation-canonical-scaling-memory`.

**Closed levers / measured-null results** (so they aren't re-chased — collective-combine,
latency-hiding, NCCL NVLS/CUMEM/PROTO, command buffers, unroll-by-K, scan-unroll,
replicate-weights, TP-without-SP): lore `2026-06-29--full32l-mfu-closed-levers-and-measured-facts`.

**Open comm lever** (CI-fn weight-grad cross-node reduce un-batched in the scan backward):
lore `2026-06-29--full32l-ci-fn-weight-grad-reduce-unbatched-in-scan`.

**Trainer-architecture canon** (JAX single-pool; torch 2/3-pool retired):
lore `2026-06-29--state--trainer-architecture`.

The full chronological trail is preserved in the git history of the `perf/hsdp-mfu` and
`perf/gather-coalesce-unroll-k` branches (this file's prior revisions).
