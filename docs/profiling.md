# Profiling PD training

A small, strategy-agnostic toolkit for answering two questions about a PD run:

1. **Where does the time go?** (what to optimize)
2. **Which distributed setup is fastest / fits in memory?** (how to scale)

It deliberately leans on standard tools — `torch.profiler` → **Perfetto** (the
[ui.perfetto.dev](https://ui.perfetto.dev) web viewer) and PyTorch's
[memory_viz](https://pytorch.org/memory_viz) — plus a thin layer of our own glue. Nothing
here is coupled to the current 3-pool taxonomy: it all keys off ranks, GPU kernels, NCCL
comm, generic PD *operations*, and memory, so it keeps working if the distribution strategy
changes.

## The three layers

### 1. Always-on comparison metrics (no profiler, works at any scale)

Logged every `train_log_every` steps to the sink/wandb. CUPTI-free, so safe at full scale:

- `perf/step_ms` — per-step wall (CPU dispatch time, MAX over the LW pool).
- `perf/tokens_per_s` — throughput, normalized for batch size. **The headline number for
  comparing configs/strategies.**
- `train/mem/{lw,ci,ppgd}_peak_gb` — peak memory per pool (the binding-constraint signal).

To compare two distribution strategies, just read these — no trace needed.

### 2. `pd/*` operation labels (make traces readable)

`torch.profiler.record_function("pd/...")` labels on the PD *algorithm primitives*, in
core `param_decomp/` (so any strategy that reuses them gets readable traces for free) plus
the LW step:

| label | where | pool that hits it |
|---|---|---|
| `pd/ci_fn_forward` | `ci_fns.py` (GlobalSharedTransformerCiFn.forward) | CI |
| `pd/calc_causal_importances` | `component_model.py` | CI |
| `pd/calc_weight_deltas` | `component_model.py` | PPGD (and others) |
| `pd/ppgd_warmup` / `pd/ppgd_recon` | `metrics/persistent_pgd_state.py` | PPGD |
| `pd/target_forward` / `pd/layerwise_recon` | `three_pool/step_layerwise.py` | LW |

These are no-ops when no profiler is active (zero overhead in production). In Perfetto they
appear as named spans on the timeline; in `key_averages` tables they appear as rows.

### 3. Per-kernel compute floors (single-GPU, no NCCL, CUPTI-safe)

For the dominant phases, get a per-kernel breakdown (GEMM vs attention vs elementwise) at
true production scale on ONE GPU — no distributed run, so no CUPTI↔NCCL deadlock risk:

```bash
srun --gres=gpu:1 python scripts/probe_ci_fn_bl_ceiling.py  --profile   # CI fn fwd+bwd, bl=8
srun --gres=gpu:1 python scripts/probe_ppgd_bl_ceiling.py   --profile   # PPGD warmup+recon, bl=8
srun --gres=gpu:1 python scripts/profile_lw_rank_step.py               # LW streaming recon, bl_lw=256
```

(Without `--profile`, the `probe_*` scripts still run their memory-ceiling sweep.) This is
the COMPUTE floor for each pool; the gap between it and the live distributed step time is
cross-pool wait/comm.

## End-to-end: profiling a distributed run

```bash
# 1. launch with profiling (one rank per pool). Reuse the existing env knobs; the launcher
#    below injects them into the multi-node SLURM job.
python scripts/launch_b256_profile.py        # 64-GPU b256 profile (see the script header)

# 2a. read it visually:  open <out>/traces/trace_{ci,layerwise,ppgd}_rank*.json at ui.perfetto.dev
#     - solid GPU-stream blocks = compute; gaps = that pool waiting on a peer; nccl = comm.
# 2b. read the numbers:
python scripts/analyze_3pool_trace.py <out>/traces      # per-pool compute / nccl / idle
# 2c. memory:  open <out>/mem/mem_rank*.pickle at pytorch.org/memory_viz
```

### Generating a profile config

`scripts/gen_b256_profile_config.py` derives a ≤96-GPU profile variant of the b256 setup.
The key constraint (see its header): the CI fn and PPGD per-rank cost scale with the *number
of decomposition sites*, so you cannot shrink GPU count by dropping sites without making
those pools unrepresentative. Instead keep all sites and drop the global batch.

## The CUPTI↔NCCL caveat (read before profiling at scale)

`torch.profiler` historically **deadlocked at ≥64 ranks** (CUPTI collection collided with a
rank-0 `.item()`-after-`recv` on an NCCL stream). Mitigations baked into the workflow:

- Profile **one rank per pool**, not all ranks.
- Place the profiler's active window **off the metric-log steps** (`launch_b256_profile.py`
  uses `skip_first=43, active=4` with `train_log_every=10`) so rank 0 never does a
  `.item()`-after-`recv` while CUPTI is collecting.
- Set `PD_TORCH_PROFILE_MEMORY=0` (CUPTI memory tracking is the heaviest part); capture
  memory separately via `PD_MEMORY_PROFILE_RANKS` (CUPTI-free).
- **If a profiled run hangs** past where it should be: it's the deadlock. Re-run without
  `PD_TORCH_PROFILE_RANKS` (always-on metrics + memory pickles still work) and get
  per-kernel detail from the single-GPU probes (layer 3) instead.

## Env-var reference

| var | meaning |
|---|---|
| `PD_TORCH_PROFILE_RANKS` | comma-separated global ranks to trace (e.g. `0,48,56`) |
| `PD_TORCH_PROFILE_OUT` | trace output dir (writes `trace_{pool}_rank{r}.json`) |
| `PD_TORCH_PROFILE_SKIP_FIRST` / `_ACTIVE` | schedule: skip N steps, then record N |
| `PD_TORCH_PROFILE_MEMORY` | `0` to disable CUPTI memory tracking (recommended at scale) |
| `PD_TORCH_PROFILE_SHAPES` | `1` to record tensor shapes (cheap; helps GEMM attribution) |
| `PD_MEMORY_PROFILE_RANKS` / `_OUT` | CUDA memory-history recorder (CUPTI-free) → memory_viz pickles |
