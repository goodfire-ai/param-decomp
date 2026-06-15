# `param_decomp_lab/fsdp/` — porting log (Agent B)

What got ported from `/mnt/home/oli/pd-nano-jax/param_decomp/`, and what was deferred.

## Ported

| Source | Destination | Form |
|---|---|---|
| `sdpa_strict.py` | `fsdp/sdpa_strict.py` | Clean reimplementation. `verify_flash_attention_available(...)` — startup probe that FA can dispatch on our production SDPA shapes. Defaults dropped: all args are supplied by the one caller (`FsdpLMTrainer.__init__`). |
| `grad_clip.py` | `fsdp/grad_clip.py` | **Thin wrapper, NOT a port of the disjoint-subset reduction.** See below. |
| `fused_linear_kl.py` | — | **Not duplicated.** Already lives in core at `param_decomp/fused_linear_kl.py` (byte-identical to the source) and is consumed by `three_pool/recon_loss_strategy.py`. The FSDP path imports `from param_decomp.fused_linear_kl import fused_linear_kl_div`. |

### grad_clip rationale

The source `grad_clip.py` solves a 3-pool-specific problem: ranks own **disjoint**
parameter subsets, so the global L2 norm is `sqrt(sum-over-ranks of local sum-sq)` and
must be reduced by hand (with a `/n_replicas` correction for DDP-replicated blocks).

The single-pool FSDP path has no disjoint cross-pool subsets — it shards one flat param
list with FSDP2/DTensor. **torch 2.11's `torch.nn.utils.clip_grad_norm_` already handles
DTensor grads correctly:** `aten.linalg_vector_norm` has a registered DTensor strategy
(`NormReduction`, in `torch.distributed.tensor._ops._math_ops`) that reduces the norm
across the mesh, and the in-place scale touches each local shard. Verified empirically
(sharded DTensor grad → global norm matches the full-tensor norm; post-clip norm == max_norm).

So `fsdp/grad_clip.py` is just two thin wrappers over `clip_grad_norm_`:
`clip_grad_norm_no_sync` (drops the returned norm to stay off the per-step device→host
sync) and `clip_grad_norm_with_norm` (returns it for logging, accepting the sync). The
module could even be dropped entirely in favour of calling `clip_grad_norm_` directly;
it is kept only to document the no-sync intent at the call site.

## Deferred (not implemented now)

Each lives in `experiments/lm/three_pool_run.py` (or `three_pool/`); referenced here so a
later port is mechanical.

| Feature | Where in source | Rationale for deferral |
|---|---|---|
| **First-fail markers** | `three_pool_run.py::_install_first_fail_marker` (~L249) — writes `$HOME/pd_first_fail/$SLURM_JOB_ID/rank<R>.json`, chains `sys.excepthook` | Multi-node hang-diagnosis aid; only worth porting once the FSDP path hits real multi-node crashes. |
| **`PD_TORCH_PROFILE_*` hooks** | `three_pool_run.py::_maybe_build_torch_profiler` (~L314) + `_PROFILE_ENV_PREFIXES` (~L123) | `torch.profiler` schedule wiring (`trainer.run(profiler=...)`); add when profiling the FSDP step, not before there's a step to profile. |
| **`PD_MEMORY_PROFILE_*` hooks** | `three_pool_run.py::_maybe_enable_memory_profile` (~L282) — `torch.cuda.memory._record_memory_history` + dump | Same: a memory-debug aid for when FSDP sharding/OOM needs investigating. |
| **PG-timeout widening on compile** | `three_pool/optimize.py::_resolve_pg_timeout` (~L132), `_COMPILE_PG_TIMEOUT` / `_DEFAULT_PG_TIMEOUT` (~L124-129), `PD_3POOL_PG_TIMEOUT_S` override; passed to `pg_timeout=` at PG init | The FSDP trainer compiles too (step-0 compile stalls the first collective). Port the timeout-widening when `FsdpLMTrainer` sets up its process group — but it's a single PG, so this is a one-line `init_process_group(timeout=...)`, not the multi-subgroup machinery. |
| **`phase_timer` instrumentation** | source `param_decomp/phase_timer.py` (`PhaseTimer`, `set_active`, `phase(...)`, `format_phase_table`); call sites in `three_pool/{step_pool_a,step_chunkwise,two_pool_optimize}.py` | CUDA-event per-phase step timing; no-op unless installed. Port alongside the profiler hooks when optimizing the FSDP step. The shared `train_step.py` helpers the FSDP path reuses are not yet `phase(...)`-instrumented. |
