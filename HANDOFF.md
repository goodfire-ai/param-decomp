# HANDOFF — full-model Llama-3.1-8B decomposition (scan + attn-sharding)

Branch: `feature/jax-full32L-scan` (forked off `feature/jax` @ `0a82fb68`, 2026-06-22).
This doc is a temporary handoff; delete it once the work merges into `feature/jax`.

## Goal
First **full-model** decomposition of Llama-3.1-8B: all 32 layers × 7 matrices
(q/k/v/o/gate/up/down) = **224 sites**, vs the prior 18–27-site (MLP-only, few-layer) R&D
runs. Per-matrix C from the 2026-06-22 botec (lore `2026-06-22--full-llama8b-pd-memory-botec`),
rounded to multiples of 128 for the `dp=128` C-shard: q/k 2048, v/o 4096, gate/up 8192,
down 10240. Config: `param_decomp/configs/llama8b_full32L_seq512_b128_dp128.yaml`
(seq 512, B 128 → per-rank 1, dp 128 = 16 nodes).

## Status: what's done / validated
- **Compile wall SOLVED.** The unrolled per-layer suffix forward made XLA's
  all-reduce-combiner / SPMD passes blow up (1h+ on 128 GPU). The suffix forward is now a
  `lax.scan` over the layer stack (one compiled block body):
  - `clean_suffix_logits` — scan one frozen block.
  - `_run_masked_suffix` — scan + per-site `lax.cond(decomp, frozen)`, preserving the
    SPEC S2 layer-split; **unrolled fallback** for heterogeneous-C configs
    (`_per_kind_dims_uniform` dispatch — the per-kind stacking needs uniform dims).
  - The `all-reduce-combiner` pass collapsed **205s → 9.2s** (CPU profile). On GPU the
    faith warmup compiled and ran to convergence (`faith 3.56e-4`) — the suffix compile is
    no longer the blocker.
- **cuDNN attn-sharding bug FIXED + GPU-validated (job 108953).** cuDNN flash attention's
  `custom_partitioner` requires q/k/v identically sharded; under scan+`cond` XLA shards `q`
  but **replicates the small GQA `k`/`v`** → "Query, key and value should have same
  sharding". Fix: a guarded `with_sharding_constraint` in `FrozenAttn.core` pins q/k/v to
  the batch-sharded layout (+ `jax.set_mesh(mesh)` in `run.py` so the bare-`PartitionSpec`
  resolves). Validated: no-fix → error, with-fix → compiles.
  **Note:** this bug is invisible to the CPU test suite — CPU uses the `xla` attn path
  (no custom partitioner); only GPU+cuDNN+multi-device triggers it.
- **int32 overflow FIXED.** `faithfulness_loss`'s `Σnumel` denominator (~7e9) overflowed
  the int32 jax materializes a Python int into under jit; now `float`.
- **Tests:** 225 pass / 7 skip. The scan reassociates float ops vs the unrolled loop
  (within tolerance, not byte-identical): torch-equivalence + trajectory tests hold; the
  stacked-parity clean-output golden was relaxed from `array_equal` to fp32 tolerance.

## Status: what's NOT done (open work, priority order)
1. **No training step / throughput / memory yet.** Fastest path: run with
   `faithfulness_warmup_steps: 0` (skips the slow weight_deltas compile) to confirm the
   real masked forward compiles with the attn fix, takes a step, and measure tok/s + HBM.
2. **Faith warmup / `weight_deltas` is slow (~2.2h).** `weight_deltas` is NOT scanned —
   it's 224 C-sharded `W−V@U` collectives (same compile/collective hotspot the suffix had).
   Best fix: **site-parallel reshard** — faith warmup is embarrassingly parallel over sites,
   so reshard V/U C-sharded → site-sharded (each device owns whole sites) for the warmup →
   zero collectives. (Literal 1-GPU infeasible: V/U + fp32 masters + Adam ≈ 250 GB.)
   Cheaper-but-compile-only alternative: vectorize the per-kind matmuls.
3. **Main-step throughput at 128-host** — the deepest unknown. The slow faith warmup is a
   canary that the main step (suffix forward × chunks + PPGD, all C-sharded) may be
   collective-bound. Unlike faith warmup it CAN'T be site-parallel (the suffix forward
   threads activations through all layers → needs the C-sharded activation layout).

## How to run (esp. on a different cluster)
- `make install-dev`, set up `.env` (WandB creds).
- **Config:** edit `data_files` in the config to the local tokenized dataset
  (`fineweb_llama_tok_512`), and `dp` to your node count × 8.
- **Launch:** `pd-lm <config.yaml>` (config-driven via `runtime.dp`).
- **Launch flags caveat:** the srun flags in `experiments/lm/launch.py`
  (`--ntasks-per-node=1`, no `--cpus-per-task`/`--distribution`) and
  `--xla_gpu_autotune_level=0` + the cuSPARSE `LD_LIBRARY_PATH` in `_RANK_ENV` were tuned
  for the polished-lake h200 cluster (`CR_Pack_Nodes`, cuDNN). The **1-process-per-node
  model is portable** (`sharding.init_distributed` claims all local GPUs); other clusters
  may want different srun/cpu-bind flags. The cuSPARSE libpath is harmless elsewhere.
- If launching from inside an interactive SLURM job, scrub inherited `SLURM_*`/`SBATCH_*`
  env before `pd-lm` (they leak into the batch job and break placement).

## Merging into `feature/jax` (careful follow-up)
`feature/jax` advanced with refactors that touch the same files: `llama8b.py` moved to
`param_decomp/targets/`, `DecompVU`/`site_out` extracted to `components.py`,
`DecomposedModel` made a Protocol, `config.py → built_run.py`. So a rebase is NOT clean —
the scan + attn-sharding changes must be **re-applied onto the relocated/refactored files**
and **re-validated on GPU**. Don't fast-forward; port deliberately.

## Reference
- Daily logs of the full arc: `~/logs/2026_06_22.md`, `~/logs/2026_06_23.md` (polished-lake).
- Scratch probes (compile-scaling, scan+cond collective, attn-sharding dump):
  `~/pd-investigation-probes/` on polished-lake (not committed; easily recreated).
- GPU validation of the attn fix: SLURM job 108953.
