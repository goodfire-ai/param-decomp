# `param_decomp_lab/fsdp/` — interface contracts (design draft)

Temporary design doc so the modules of the single-pool FSDP LM path compose without drift.
Will be folded into a proper `CLAUDE.md` once the path stabilises. **If a contract here is
wrong or awkward, push back — surfacing type/composition/config problems is the point.**

Environment confirmed: **torch 2.11.0** → `torch.distributed.fsdp.fully_shard` (FSDP2) and
`torch.distributed.checkpoint` (DCP) both import cleanly.

## What we're building

A single-process-group LM PD trainer (`pd-lm-fsdp`) that runs the full VPD algorithm
(faith + importance-min + stochastic-recon + persistent-PGD, authored as ordinary
`pd.loss_metrics`) through the SHARED step helpers in `param_decomp/train_step.py`, but
scales via **FSDP2 (memory) + torch.compile (speed)** on the **vendored** `LMComponentModel`,
with residual-start, DCP sharded saves, off-loop async consolidation + slow eval, and SLURM
requeue + resume-in-place. Pools (`three_pool/`, `two_pool_run`, `three_pool_run`) are
untouched. Core `param_decomp/optimize.py` `Trainer` (DDP) is untouched in behaviour.

## Shared step helpers already available (reuse, do not reimplement)

`from param_decomp.train_step import (run_loss_step, run_eval_pass, scheduled_lrs,
EvalLoop, _install_sigterm_flag, empty_cuda_cache_and_collect)`

- `run_loss_step(*, batch, step, device, wrapped_model, component_model, loss_metrics,
  config: PDConfig, reconstruction_loss, autocast_bf16) -> (total_loss, batch_log_data)` —
  computes weight_deltas (outside autocast), builds the `MetricContext`, runs every loss
  metric's `update` under autocast, sums `coeff*loss`, runs `before_backward` /
  `backward` / `after_backward`. Does NOT zero_grad / clip / step / log. The FSDP trainer
  wraps THIS CALL in `with component_model.use_cached_residual(batch):` for residual-start.
- `run_eval_pass(*, eval_iterator, n_steps, slow_step, all_instances, step, device,
  wrapped_model, component_model, config, reconstruction_loss, autocast_bf16) ->
  (fast_metrics, slow_metrics)` — caller logs the dicts and calls
  `empty_cuda_cache_and_collect()` after.
- `scheduled_lrs(step, *, total_steps, config) -> (components_lr, ci_fn_lr)`.

For FSDP, `wrapped_model is component_model` (the adapter) — `fully_shard` mutates in place,
there is no DDP wrapper object.

## Module ownership (one owner each — no shared files)

### `component_adapter.py` — Agent A
`FsdpComponentAdapter(nn.Module)` wrapping a vendored `LMComponentModel` so it presents the
exact surface `train_step.py` + the loss/eval metrics consume from the core `ComponentModel`.
The LM `batch` is a bare `[B,S]` Int token-id tensor → **`batch is idx`** (confirm via
`collate_token_column` in `experiments/lm/data.py`). Required surface:
- `forward(batch, mask_infos=None, cache_type="none") -> OutputWithCache | Tensor`
  - `cache_type="input"` → `OutputWithCache(output=logits, cache=pre_weight_acts)` where
    `pre_weight_acts` is exactly the dict `LMComponentModel.calc_causal_importances` consumes
    (`forward_with_pre_weight_acts` already returns it, path-keyed).
  - `cache_type="output"` → not used by core metrics directly; route to
    `forward_with_output_acts` if a caller asks, else assert unsupported.
  - `cache_type="none"` (default) → bare logits `Tensor`.
- `forward_with_output_acts(batch, mask_infos=None) -> (Tensor, dict[str,Tensor])`
- delegate properties/methods: `calc_causal_importances`, `calc_weight_deltas`,
  `module_to_c`, `target_module_paths`, `components`, `ci_fn`, `use_cached_residual`,
  `model` (the inner ComponentTarget). Keep `nn.Module` (so it can be a fully_shard root /
  hold the inner module), but it has NO params of its own.
- **Typing note (resolved):** `MetricContext.model`, `Metric.bind(model=...)`,
  `instantiate_metrics(...)`, and the shared `train_step.py` helpers are typed against
  `ComponentModelProtocol` (defined in `param_decomp/component_model.py`) — the structural
  surface the training step + loss/eval metrics consume. The adapter (`FsdpComponentAdapter`),
  the core `ComponentModel`, and the vendored `LMComponentModel` all satisfy it, so the
  `cast(ComponentModel, cast(object, lm))` idiom is gone. The Protocol's `__call__` uses a
  3-way `cache_type` (`"input"`/`"output"`/`"none"`) — the subset consumers request — so the
  adapter's narrower forward satisfies it while the concrete model's 4-way forward also does.

### `config.py` — Agent C
`FsdpRuntimeConfig(RuntimeConfig)` — subclass core `param_decomp.configs.RuntimeConfig`
(inherits `autocast_bf16`, `device`, `dp`), adding:
- `compile_model: bool = True` — `torch.compile` the vendored `model` (masked forward).
- `compile_ci_fn: bool = True` — `torch.compile` the CI fn.
- `checkpoint_blocks: bool = True` — per-block activation checkpointing on the target.
- `shard_frozen_target: bool = True` — convert frozen target buffers to no-grad params so
  FSDP2 actually shards the 8B target (else buffers are replicated).
Do NOT subclass the pools' `PooledRuntimeConfig`. Keep it a plain `BaseConfig` subclass via
`RuntimeConfig`.

### `sdpa_strict.py`, `grad_clip.py`, `fused_linear_kl.py`, `PORTING.md` — Agent B
Crib from `/mnt/home/oli/pd-nano-jax/param_decomp/{sdpa_strict,grad_clip,fused_linear_kl}.py`,
adapting imports to this repo. `verify_flash_attention_available(...)` is called once at
trainer init. `grad_clip` provides an FSDP/DTensor-aware global grad-norm clip (the trainer
uses it instead of `torch.nn.utils.clip_grad_norm_`, which mishandles DTensors). `PORTING.md`
lists deferred ports: first-fail markers, `PD_TORCH_PROFILE_*` / `PD_MEMORY_PROFILE_*` hooks,
PG-timeout widening on compile, `phase_timer`.

### `checkpoint.py` + `consolidate.py` — Agent D
On-loop sharded save + off-loop consolidation. Save TRAINABLE-ONLY state (components V/U +
CI fn + optimizers + step + each loss-metric `state_dict()` — NOT the frozen target).
- `save_dcp(model, optimizers: dict[str, Optimizer], *, step, loss_metric_states: dict[str,dict],
  out_dir: Path) -> None` — `torch.distributed.checkpoint.save` of a state dict built via
  `torch.distributed.checkpoint.state_dict.get_model_state_dict` /
  `get_optimizer_state_dict` (sharded). Writes `out_dir/.dcp/step_<S>/`. No full-gather.
- `load_dcp(model, optimizers, *, step, in_dir) -> dict[str,dict]` — `dcp.load` directly into
  the FSDP2-sharded model via `set_model_state_dict` / `set_optimizer_state_dict`; returns the
  loss-metric states for the trainer to load. Used by resume-in-place; NO dependency on
  consolidation.
- `latest_dcp_step(run_dir: Path) -> int | None` — newest `.dcp/step_<S>/`.
- `consolidate.py::consolidate(run_dir, step, *, build_full_state) -> None` — off-loop:
  read the DCP shards into a FULL state dict, write `model_<step>.pth` (downstream-compatible
  `LMComponentModel` state) + `training_<step>.pth` (`TrainingState`), prune old, idempotent.
  Reuse `TrainingState` from `param_decomp.training_state`. Look at
  `three_pool/consolidate.py` for the pruning + idempotency + filenames it already uses.

### `trainer.py` — Agent E
`FsdpLMTrainer`:
- `__init__(self, *, target_model: nn.Module, pd_config: PDConfig,
  runtime_config: FsdpRuntimeConfig)`. Build sequence:
  1. `insert_identity_operations_` (if `pd.identity_decomposition_targets`) + freeze + eval +
     `resolve_decomposition_targets(target_model, pd.all_decomposition_target_configs)`
     (mirror core `Trainer.__init__`).
  2. `LMComponentModel.build(target_model, decomposition_targets, pd.ci_config, pd.sigmoid_type)`.
  3. If `shard_frozen_target`: convert frozen target `register_buffer` weights →
     `nn.Parameter(requires_grad=False)` so FSDP2 shards them.
  4. If `checkpoint_blocks`: enable per-block activation checkpointing on the vendored model.
  5. `fully_shard` each transformer block (`lm.model._layers` / `._h`) + each CI-fn block +
     the root model; then `.to(device)`.
  6. Per-rank `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR`; then `lm.model.compile()` /
     `lm.ci_fn.compile()` if the flags are on.
  7. Wrap in `FsdpComponentAdapter`; build the two AdamW optimizers (components params /
     ci_fn params) like core `Trainer`; `instantiate_metrics(pd_config, adapter, device)`.
  8. `verify_flash_attention_available(...)`.
- `run(self, train_loader, sink, cadence, scratch_dir: Path, eval_loop=None) -> None` —
  mirror core `Trainer.run` loop but: residual-start wrap around `run_loss_step`; FSDP-aware
  grad clip (`grad_clip.py`); DCP save via `checkpoint.save_dcp` → fire `sink`'s `on_save`
  (async consolidate+eval); requeue/sigterm save. In-train eval is FAST-ONLY (slow eval is
  async). Reuse `run_loss_step` / `run_eval_pass` / `scheduled_lrs`.
- `from_dcp(...)` classmethod (or a `load_dcp` call in `__init__`-then-load pattern) for
  resume-in-place. Cross-run resume loads the consolidated `training_<step>.pth` (mirror core
  `Trainer.from_snapshot`).
- **Design to surface:** `loss_metrics` here include PPGD, whose `before_backward` uses
  `torch.autograd.grad(retain_graph=True)`. Activation checkpointing recompute is
  nondeterministic under that → if a PPGD metric is configured, assert/force
  `checkpoint_blocks=False` (or document the conflict). Report your call.

### `experiments/lm/fsdp_run.py` + console script — Agent F
Mirror `experiments/lm/three_pool_run.py`: `main` / `_submit_slurm` /
`_fresh_or_requeue_main` (scan latest `.dcp/step_<S>/`, not `training_<step>.pth`) /
`_resume_in_place` / `_run_resume` / `submit_slurm_async_consolidate_and_eval`,
`SlurmConfig(requeue=True)`. New `FsdpLMExperimentConfig(BaseConfig)` with
`pd: PDConfig`, `runtime: FsdpRuntimeConfig`, `cadence: Cadence`, `target: LMTargetConfig`,
`data: LMDataConfig`, `eval`, `wandb`, `resume_provenance`. **Loader is data-parallel** —
`build_lm_loader(..., dist_state=dist_state, ...)` (DistributedSampler shard, like `pd-lm`),
NOT the pools' full-batch-per-rank `dist_state=None`. Reuse `build_target`, `make_run_batch`,
`_build_eval_loop` (include_slow=False), `_split_metrics_by_slow` from `experiments/lm/run.py`.
`SavedFsdpLMRun` reload class (loads consolidated `model_<step>.pth` via the vendored loader,
cf. `SavedThreePoolLMRun.load_model` `case "vendored"`). Add `pd-lm-fsdp =
"param_decomp_lab.experiments.lm.fsdp_run:cli"` to `param_decomp_lab/pyproject.toml`. Extend
`experiments/lm/async_eval.py` with a `variant="fsdp"` that consolidates DCP shards then runs
slow eval (or document the wiring if it diverges from the pooled async_eval).

## Cross-cutting conventions
- Repo style: fail-fast asserts, narrow types, PEP604 unions, lowercase generics, no
  `from __future__ import annotations`, basedpyright clean, ruff clean (`make check`).
- Every module owner: at the END of your work, append a short **"Design concerns"** section
  to this file (under your module heading) listing any type/composition/config awkwardness
  you hit or had to cast around. That list is the deliverable we review next.
