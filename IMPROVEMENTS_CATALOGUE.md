# Recent improvements — catalogue & where they live (2026-06-26)

Audit of every recent perf/memory/correctness improvement to the JAX trainer, **how
it's enabled**, and **which branch actually has it**. Motivated by finding the full32L
runs silently defaulting `remat_ci_fn=false` (→ 82-min compile + near-OOM).

## TL;DR — the work is fragmented; no single branch has everything

The improvements were built in parallel worktrees and never unified. The branch we've
been **launching from** (`feature/jax-full32L-scan`) has the **launcher** but is **42–44
commits behind** and carries **none of the trainer memory/perf wins**. The wins live on
`feature/jax-full32L-port` (5 of 6) and `feature/stable-imp-min` (the 6th).

## Catalogue

| # | Improvement | Commit | How enabled | scan (current/launch) | port | stable-imp-min | bf16-worktree |
|---|---|---|---|:--:|:--:|:--:|:--:|
| 1 | scan-over-layers fwd + cuDNN attn (q/k/v head-parallel) sharding | cd88aaa8c (+port's own) | code | ✓ | ✓ | ✓ | ✓ |
| 2 | **per-layer remat** of recon forwards (scan-body `jax.checkpoint`) | 3a5922207 | code | ✗ | ✓ | ✗ | ✓ |
| 3 | **CI-gather scan** (`ChunkwiseTransformerCIFn`: vmap→`lax.scan` over chunks) | 58e3b09c7 | code | ✗ | ✓ | ✗ | ✓ |
| 4 | **`remat_ci_fn`** — checkpoint the ~31B CI fn (the big compile/mem lever) | 3937f336a | **config flag** `runtime.remat_ci_fn` (default **false**) | ✗ | ✓ | ✓ | ✗ |
| 5 | CI-fn **Megatron/TP placement** (model-owned per-leaf sharding) | ac1973bdf | code | ✗ | ✓ | ✓ | ✗ |
| 6 | **smooth-L0 imp-min** (Geman–McClure penalty) | 8dfb86103 | config (`SmoothL0ImportanceMinimalityLossConfig`) | ✗ | ✓ | ✓ | ✗ |
| 7 | frequency-minimality split (batch-invariant metric) | 67462638a | code | ✗ | ✗ | ✓ | ✗ |
| 8 | **bf16 PGD sources** (halves source+Adam, ~41→21 GiB) | 238254ed1 | **config flag** `source_dtype` (default float32) | ✗ | ✗ | ✗ | ✓ |
| — | multi-host (1-proc/node) launcher + full-model config + cuSPARSE libpath | 08299eb4e | infra | ✓ | ✗ | ✗ | ✗ |

Lineage: all branches share ancestor `bd1306e2a` (2026-06-25). `port` = +2 (#2,#3),
`stable-imp-min` = ci-fn-chunk-scan +1 (#7). `port` already contains #4,#5,#6.
`feature/jax-full32L-scan` forked **2026-06-19**, before any of #2–#8.

## Config flags to set in the full32L configs (independent of the branch merge)

| flag | default | set to | why |
|---|---|---|---|
| `runtime.remat_ci_fn` | false | **true** | the 82-min-compile / near-OOM lever; off in p-7bb3b645 |
| `runtime.remat_recon_forwards` | false | **true** | already set in our configs ✓ |
| `source_dtype` (PersistentPGD) | float32 | **bf16** (optional) | halves PPGD source+Adam (~20 GiB at full-32L); needs #8 merged; stability-watch |
| `use_fused_kl` (ChunkwiseSubset) | true | true | default OK ✓ |
| smooth-L0 imp-min | L_p | optional | swap loss type if we want the smoother penalty |

## The consolidation gap

To "use them all" we need ONE branch with: the **launcher** (only on `scan`) + **#2–#6**
(on `port`) + **#7** (on `stable-imp-min`) + optionally **#8** (bf16 worktree). Merging
`port`→`scan` touches 7 overlapping core files (`launch.py`, `llama8b.py`, `run.py`,
`sharding.py`, `losses.py`, the config yaml, a parity test) — real conflicts, esp. if
scan-over-layers was implemented twice. Needs a careful merge, not a fast-forward.
