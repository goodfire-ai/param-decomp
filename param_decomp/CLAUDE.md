# param_decomp — agent notes

Single-pool VPD trainer in JAX, **generic over vendored LM targets**. The semantics
source of truth is `SPEC.md` (normative pseudocode + numbered invariants, grounded in
the stable torch `param_decomp` impl). See `README.md` for the file map.

Open items: persistent-source scopes `c`/`nsc` and sigmoid parameterization are
deliberately refused. The hidden-acts seam is now BUILT (SPEC S31 amended 2026-06-16):
`CIHiddenActsReconLoss` / `StochasticHiddenActsReconLoss` are standalone eval metrics
(`hidden_acts_eval.py`, computed in-loop on `eval.slow_every`) over
a fifth model fn `masked_site_outputs` — NOT recon-grid training terms (the recon loss
stays KL-on-final-logits). `sc` and `bsc` are supported (`bsc` is batch-sharded:
an independent source per batch element and position, no cross-replica sync — SPEC
S16/D1). SPEC S24's two torch-parity quirks (PPGD warmup route-all, fresh-PGD
single routing draw) are pinned pending a team decision. CI-fn numerics are unified
with the torch oracle (#624/#625/#730 resolved): GELU is exact-erf
(`approximate=False`) and RMSNorm eps is `finfo(fp32).eps` (`CI_FN_RMS_EPS`).

## The one rule

**Every change is checked against SPEC.md, by invariant ID.** If a change deviates
from an invariant, either fix the change or (deliberately, with Oli) amend the spec —
never silently diverge. Cite IDs (`S14`, `N1`, …) in commit messages and reviews.

## Architecture in one breath

`lm.py` defines `DecomposedModel` — a `@runtime_checkable Protocol`: ordered `sites` +
`leading_axes` + the methods `clean_output`, `read_activations`, `masked_output`,
`masked_site_outputs`, `weight_deltas`, and a `recon_loss_fn` (LM: `kl_per_position`). The
concrete impl per target is an `eqx.Module` (`LlamaDecomposedModel`,
`SimpleMLPDecomposedModel`, `TMSDecomposedModel`, `ResidMLPDecomposedModel`) carrying its
FROZEN target weights as ARRAY FIELDS; the TRAINABLE V/U (`vu: DecompVU`) stays an explicit
METHOD ARG (separate lifecycle — own optimizer + checkpoint, C-sharded while the frozen
weights replicate). Flat site-name-keyed dicts at the boundary; the model threads into the
jitted step as a pytree ARG (never a jit-closure constant — an 8B target becomes a multi-GB
HLO constant; see "HLO-baking rule" below). The activation waist is GENERIC `[*leading, d]` (masks/CI
`[*leading, C]`), `leading = (batch,) + named position axes`: masking / routing / sources
/ imp-min all read an opaque `leading = residual.shape[:-1]`; reductions are
`math.prod(shape[:-1])` / `axis=tuple(range(ndim-1))`. CI is independent over every leading
axis (no per-axis CI semantics, only axis NAMES).
`DecomposedModel.leading_axes` names the position axes (`("sequence",)` for LM, `()` for
TMS); `CIFn.expects_axes` mirrors it, and `init_train_state` asserts they're equal (early
fail) so the CI fn stays per-domain (RoPE over `sequence`) without the core adapting. The
three EDGES are generic so non-LM (bio-style) targets fit (#828): the model INPUT
(the opaque batch `clean_output` / `read_activations` / `masked_output` consume, typed
`Any` — token ids for an LM, a dict for bio), the model OUTPUT (`clean_output`/`masked_output` return `Any` — logits, a tuple
of heads, coords; field NAMES stay `*_logits` pending a deferred rename), and the recon
comparison (`recon_loss_fn(clean_output, masked_output) -> scalar`, default
`kl_per_position` so the LM path is byte-identical). The waist shape contract (all per-site
tensors in one forward share one `*leading` prefix) is enforced at trace time by
`@jaxtyped(typechecker=beartype)` on the core `step`, `masked_forward`, and the loss fns.
`train.py` is the generic step factory
(fp32 masters / bf16 compute) over a flat tuple of self-describing loss TERMS
(`recon.LossTerms` — faithfulness, importance-minimality, and the recon terms, iterated
uniformly; S10′ — the recon loss-class cartesian product factored as chunking × routing ×
mask-source strategy: a chunking helper (`one_chunk`/`per_site`/`into_groups`) feeds the
single `make_plan` constructor, built from the shared configs by `recon.build_loss_terms`;
see LOSS_PARITY_DESIGN.md),
consuming `losses.py` (pure loss terms + schedules) and `adversary.py` (persistent
vs fresh source machinery — semantically distinct adversaries sharing only
`source_masks`); `ci_fn.py` the shared CI transformer; `targets/llama8b.py` + `targets/llama8b_sharding.py` the first target. There is ONE
recon semantics: masks thread through the full token-input forward, loss is KL on final logits
(SPEC §2.3–2.5). Site-local recon is a conceptual no-no, not a "simplification".
`targets/llama_simple_mlp.py` is the second target (the pile-pretrained `LlamaSimpleMLP`,
t-9d2b8f02; sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}`) —
config dispatch is `TargetConfig` (llama8b) vs `LlamaSimpleMLPTargetConfig`, both LAB-side
(`param_decomp_lab/experiments/lm/config.py`, which reads the canonical schema DIRECTLY —
`build_experiment_config`/`load_config` — routing `kind: pretrained` specs + `h.*`
wildcards), target build in the LM composition root
`param_decomp_lab/experiments/lm/run.py::main`. The slow plot metrics are computed
NATIVELY in JAX (`slow_eval.py`) — no torch export round-trip (the torch offline-eval
bridge `jsp-export` / `pd-offline-eval` was retired). They run IN-LOOP ONLY on
`eval.slow_every` next to the fast pass (SPEC S28/S29; there is NO offline/retrospective
CLI — `slow_eval.py` is a pure library): the collective
forward + device→host pull in lockstep on all ranks, the matplotlib render + `wandb.log`
on a rank-0 background thread (`run.py::SlowEvalRenderer`), reusing the fast pass's eval
batches and logging on the live `_step` axis. The config-gated position-CI metrics
(`PermutedCIPlots` / CI heatmaps + `IdentityCIError`) ALSO run in-loop off the cheap
`(T, C)` position-CI matrix (`accumulate_position_ci`, collective; the heatmap figures on
the background thread, the `IdentityCIError` scalars synchronously on `_step`). `UVPlots`
is a config-gated figure metric usable for ANY decomposition (the torch `Metric` pattern —
returns a wandb figure): for the LM in-loop tier the LM composition's `eval_fn` does a
NAIVE host gather of the C-sharded V/U (gated on `want_uv_plots`) and passes `components` to
`render_permutation_figures` — it OOMs / breaks at production C BY DESIGN (per Oli), no
special handling; for the positionless toys (TMS/ResidMLP) `toy_uv_eval.log_uv_figure`
renders it off the small on-host V/U + the probe CI as permutation source (cheap, no
gather), sharing `slow_eval.render_uv_figure` / `plot_uv_matrices` with the LM path.

**The toys (TMS, ResidMLP) live in the lab, not the core.** The core trainer carries ZERO
toy-specific code — the toy *targets* (`DecomposedModel`s, pretrain, identity-CI eval) are
all lab-side. CI-fn *architectures* are NOT toy-specific code: core owns every CI-fn arch
regardless of which experiments use it. The positionless MLPs and the sequence transformer
are peers in `ci_fn.py` (differing by domain, not status), not a toy carve-out. The
generic engine is `run.py::run_decomposition_training(pd, cadence, run, raw_cfg, lm, frozen,
ci_fn, data, remat_recon_forwards, sample_batch, eval_fn, eval_every, perf_tokens_per_step,
mesh)` — the ONE train loop every target runs through (init/restore/finetune/faith-warmup
via `_init_or_restore_state`, the recon-grid step factory, orbax checkpointing, schedules,
SIGTERM-save). It reads the pydantic `PDConfig` / `Cadence` (`param_decomp.configs`)
DIRECTLY — optimizers / loss metrics / faith warmup / seed / steps — so there is
NO flattened mirror DC (the old `config.ExperimentConfig`); the run identity rides in
`config.RunInstance`, and the lab-built objects (`ci_fn` arch, `data`, the decomposed target)
pass alongside. A target injects exactly three seams: the data source
(`sample_batch(step) -> residual`), the eval metric (`eval_fn(state, now_step) -> dict`, run
every `eval_every`), and (for the LM) the perf token count.
`param_decomp_lab/experiments/lm/run.py::train` is the thin LM caller (parquet
`sample_batch` + the CEandKL/CI-L0/PGD/attn-patterns `eval_fn` in `_make_lm_eval_fn`); that
LM composition root is LM-ONLY (`experiments.lm.config.build_from_schema` validates
`LMExperimentConfig` and returns a `config.BuiltRun`; `main`'s `match built.target` covers
only `TargetConfig` / `LlamaSimpleMLPTargetConfig`). `BuiltRun.target` is typed by the core
`config.TargetSites` protocol (just `.sites`), `BuiltRun.data` is `DataConfig | None` (None
for a toy run). The shared schema validation + run-identity / CI-fn-arch helpers are public
lab-side for the toys to reuse: `experiments.config.assert_canonical_algorithm_config` /
`run_instance` / `ci_arch`.

The TMS + ResidMLP targets now live under `param_decomp_lab/experiments/{tms,resid_mlp}/`
(`model.py` = the JAX `DecomposedModel` + frozen target + in-process pretrain + identity-CI
eval; `run.py` = the `pd-tms` / `pd-resid-mlp` CPU CLI that builds the `ExperimentConfig`
from the canonical schema and calls `run_decomposition_training`). They are positionless
(`leading_axes=()`) and use the MLP CI fns. All CI-fn architectures live together in
`ci_fn.py`: `LayerwiseMLPCIFn` (`expects_axes=()`, one independent MLP per site mapping
`site_input [B,d_in] -> [B,C]`), `GlobalMLPCIFn` (`expects_axes=()`, one shared MLP over all
sites jointly, concat/split in canonical site order), and the LM `ChunkwiseTransformerCIFn`
(`expects_axes=("sequence",)`, per-chunk transformers reading residual taps, stacked +
`lax.scan`'d with per-chunk remat, and **N per-site output heads** (one `[d_model, C_j]` per
site-slot). NOTE: this is the pure-HSDP backup branch — the mesh is `(replicate, fsdp)` with
NO tensor-parallel / Megatron-C axis (`fsdp` = the 8 intra-node NVLink GPUs, `replicate` =
across nodes). The CI output C axis is NEVER sharded, so the per-site heads are a layout
convenience here (they were load-bearing under the prior TP layout, which sliced a tp-sharded
glued-ΣC head mid-site). **ZeRO-1 ÷N**: the trainable V/U + CI-fn fp32 masters AND their Adam
m/v shard ÷N over the FULL mesh (`("replicate","fsdp")` on V's d_in / U's d_out / the CI fn's
d_model) — the dominant optimizer-state memory scales 1/N, not the fixed 1/fsdp. The bf16
COMPUTE weights are reconstructed to the `fsdp`-sharded (÷fsdp) layout ONCE per step in ENTRY
(the cross-`replicate` gather, off the hot path — `llama8b._reconstruct_compute_weights` /
`ci_fn._reconstruct_ci_compute_weights` pin `P(None,"fsdp",...)` BEFORE the per-layer /
per-chunk scan), landing a SMALL ÷fsdp-resident stack; the scan body then gathers ONE layer's
`fsdp` shard to full d_in transiently (NVLink, freed each iteration) — NEVER a
full-model `[n_layer, full_d_in, C]` weight stack resident.
`run_state.init_train_state` dispatches CI-fn construction on `cfg.ci_fn`
(`MLPCIArch` / `GlobalMLPCIArch` / `ChunkwiseTransformerCIArch`) and uses replicated (not
C-sharded) V/U + CI for the tiny toys; the core `config.CIFnArch` admits all three and the
lab `experiments.config.ci_arch` builds the layerwise / global arch from the toy
ci_config (validated end-to-end on CPU via
`pd-resid-mlp`). Harvest / slow-eval / export over the toys are NOT wired
(`experiments.lm.load_run.build_target` / `run_metadata` are LM-only).

## The HLO-baking rule (filter_jit discipline)

The decomposed model is an `eqx.Module` whose frozen target weights are ARRAY FIELDS — a
Llama-8B target is multi-GB. Therefore:

- **Every `jax.jit` that touches a model is an `eqx.filter_jit` with the model as a TRACED
  ARG** — never `@jax.jit` over a function that CLOSES OVER an array-bearing model. A closed-
  over model is a jit *constant*: its arrays bake into the HLO (multi-GB constant tensors,
  recompiled per concrete model). As a traced arg, the array leaves are dynamic inputs and
  the static fields (`sites`, `eps`, `leading_axes`) bake harmlessly.
- **Step factories read only STATIC config off the closed-over `lm` at trace-setup**
  (`lm.site_names`, `lm.sites`, `lm.recon_loss_fn` — `recon_loss_fn` is a `@staticmethod`,
  pure, holds no arrays, so closing over it is safe). All ARRAY access goes through the
  model ARG (named `model` inside the jitted fn). `make_train_step`, `make_eval_step`,
  `make_*_hidden_acts_step`, `make_slow_eval_step`, `make_position_ci_step`,
  `make_*_attn_patterns_step`, and `make_faith_warmup_step` all follow this; each carries a
  comment at the step factory. The toy `run.py`s and `load_run.py`'s harvest `forward`
  thread the model as a filter_jit arg too.
- This is why the methods take only the *runtime-varying* args (`vu`, `resid`, masks, …) and
  the frozen weights ride on `self`: `self` reaches the trace as the traced model arg.

## Invariants with sharp teeth (the ones that have actually bitten)

- **S3**: the recon target is the FROZEN-path forward (`clean_output`), never the
  `mask=1` decomposed identity (bf16 rounding + V/U in the stopped graph).
- **S13/S15**: source updates go through the persistent Adam AND project to [0,1]
  after EVERY ascent — an unprojected drift past 1 has zero `clip` gradient and the
  entry dies.
- **S14**: the final source ascent reuses the main backward's source-grad
  (pre-update θ), unscaled by the ppgd coeff. No extra forward.
- **N1**: fp32 masters everywhere (`optax.adamw(..., weight_decay=0.0)` — optax's
  default wd is 1e-4, torch's is 0; this was audit finding A7).
- **`inv_freq` is a buffer, not a param** — `stop_gradient` in `CIFn.__call__`.
- **S10/S11**: chunking is sequential `sites_per_chunk` groups in canonical site
  order; routing is uniform-k over the chunk's sites only.

## Validation stack (run all before claiming correctness)

1. `pytest param_decomp/tests/` — at the default device count AND
   `XLA_FLAGS="--xla_force_host_platform_device_count=4"`.
2. `tests/equivalence/` — fixture-driven JAX-vs-frozen-golden per-term numeric
   equivalence (fp32, no RNG, zeroed attn). The torch references are FROZEN committed
   goldens (`torch_reference.json`, `simple_mlp_equivalence/*.npz`,
   `tools/export_fixtures/*`); the torch generators/verifier that produced them are
   deleted so `param_decomp` imports no torch (push-1). Regenerate goldens only when
   the MATH changes: redraw fixtures JAX-side with `gen_fixtures.py`, then check out the
   `torch-oracle` git tag in a torch-venv worktree and run that revision's
   `torch_reference.py` / `gen_torch_fixtures.py` / `gen_export_fixture.py`, copying the
   emitted goldens back here.
3. `experiments/invariance_check.py` at 4 sim devices — trajectory invariant to
   device count up to float reassociation (SPEC D4).

`basedpyright` over the whole workspace must be clean (run `make type`); `param_decomp`
is in the root `[tool.pyright]` include and is checked in the one venv, one pass,
alongside lab.

## The training pipeline

The generic ENGINE `run.py::run_decomposition_training` is a pure library (no `main`, no
YAML). The composition root + only I/O layer is LAB-side:
`python -m param_decomp_lab.experiments.lm.run <config.yaml>` reads the YAML, builds the
target + data loader + `ExperimentConfig`, and calls the engine; the step stays pure. Data
is a pre-tokenized parquet artifact under
`$DATA_MOUNT/artifacts/mechanisms/param-decomp/datasets/` (`fineweb_llama_tok_2048`
for Llama-8B, `pile_neox_tok_512` for `LlamaSimpleMLP`) — NEVER stream/tokenize from
HF at run time (the 80-rank thunderherd lesson). The batch schedule is a pure
function of `(seed, step)` (O(1) resume, no replay); checkpoints are orbax sharded
saves (no on-loop full-gather); SIGTERM → save → SLURM requeue → resume from latest.
Resume with a changed config is refused (byte-compare). Smokes before a long run
MUST exercise save AND resume at the production per-rank shape.

A run config is ONE self-contained yaml: the experiment schema
(`param_decomp_lab.experiments.config.ExperimentConfig` over the core
`param_decomp.configs` pieces — `pd`/`data`/`eval`/`cadence`/`runtime`/`target`/`wandb`)
plus the run-instance fields
the schema now also carries — top-level `run_name`/`run_id`/`out_dir`, the
`runtime.remat_recon_forwards` memory/compute knob, and `wandb.group`/`wandb.tags`.
`run_id`/`out_dir` are absent in a hand-authored config; `pd-lm` mints + stamps them.

**Fine-tune from a parent checkpoint** (`resume_provenance`, SPEC S33, LM-only). A fresh
run can initialize its trained decomposition (V/U + ci_fn) from a PARENT run's checkpoint
and continue under a DIFFERENT config (changed LR / coeffs / eps / seq / batch / steps —
NOT changed C / sites / ci-fn arch). Add to the config:

```yaml
resume_provenance:
  # ABSOLUTE path — the trainer runs with cwd = <workspace> (the repo root), so a
  # relative path would resolve under the workspace, not the output runs dir.
  parent_run_dir: /mnt/data/artifacts/mechanisms/param-decomp/runs/p-bd3cd4d4
  parent_step: 175000
```

On the FIRST entry (own `ckpts/` empty) the trainer loads `parent_run_dir/ckpts/175000`
onto the fresh reference and keeps ONLY the components + ci_fn; the optimizer states,
persistent sources, and `step` are FRESH (`step = 0`, no faith warmup) so the new LR /
p-anneal schedule recomputes over the new `cfg.steps` from 0. A subsequent SLURM requeue
(own `ckpts/` now non-empty) resumes from the run's own dir and ignores provenance.
`run.py::assert_finetune_structural_compat` reads the parent's pinned `launch_config.yaml` and
asserts matching sites (names + C) + ci-fn arch before the restore. Provenance flows into
`config.yaml` + `wandb.config`. Launch as usual via `pd-lm <config.yaml>`.

**Launch is CONFIG-DRIVEN via `runtime.dp`** (lab-side `pd-lm <config.yaml>`): there are NO
`--nodes` / `--local` / `--distributed` flags. The mode is a pure function of the config's
`runtime.dp`:
- `dp = null` → run the trainer INLINE in the current process (single device, no SLURM, no
  workspace). For smoke / debug.
- `dp = N` (a multiple of 8) → submit to SLURM across `nodes = N // 8` nodes,
  `--ntasks-per-node=8`. Mints the `p-` run id, snapshots the tree to
  `refs/runs/snapshot/<id>`, materializes an immutable shared-FS workspace (clone + the one
  CUDA venv) at `$PARAM_DECOMP_OUT_DIR/workspaces/<id>`, stamps the id (+ out_dir / wandb
  group / tags) into the workspace's single config yaml, and sbatches. The srun command is
  bare `python -m param_decomp_lab.experiments.lm.run <config>` (no rank/topology flags).

Requeues re-enter the workspace, never the live checkout. `--run_id` resubmits an existing
workspace. Don't hand-write sbatch files.

`main` enables JAX's persistent compilation cache
(`_enable_persistent_compilation_cache`) at `$PARAM_DECOMP_OUT_DIR/xla_compilation_cache`
— a SIBLING of `runs/` (derived from `out_dir.parent`), shared across all runs and all
8N ranks, NOT per-run. The multi-minute chunkwise-step compile is keyed by HLO + backend +
topology + jax/xla version, so a requeue/resume or a fresh run at the same config+topology
loads the executable from disk in seconds. Set after `init_distributed` (the write gate
reads the distributed state) and before the first compile; threshold 60s
(`jax_persistent_cache_min_compile_time_secs`) so only the big compiles cache. Multi-host
safe on jax 0.10.1: jax gates the cache WRITE on `process_id == 0` (`compiler.py` — "Only
write cache entries from the first process … contention for writes on some filesystems"),
so all ranks read but only rank 0 writes — no shared-FS race. Requires the cache dir on a
shared FS, which `$PARAM_DECOMP_OUT_DIR` already is.

### Cold-start compile budget (measured 2026-07-06, btdr, isolated-cache probe grid)

Method: one production-shape probe run per mutation (12 steps, eval/checkpoints off,
`JAX_LOG_COMPILES=1`), each launched with its own `PARAM_DECOMP_OUT_DIR` so runs never
share an XLA cache — cache isolation is a submit-time env var, nothing repo-side. Logs +
launch configs archived on btdr (`compile_probe_grid_2026-07-06_logs.tgz`, sibling of
`runs/`). Findings at the full 32L/224-site production shape:

- The cold start is a SUM of serial compiles, and the inits used to dominate it:
  `jit_init_decomp_vu` 211s + `jit_build_ci_fn` 167s + `jit_init_persistent_sources` 55s
  before `jit_step` even started. All three are now vmap-stacked (few-outputs-under-jit;
  `init_decomp_vu_placed` / `init_chunkwise_transformer_ci_fn` / `init_sources_sharded`),
  bit-identically: measured after, V/U 16s, build_ci_fn 4.4s, sources <2s — the
  historical "~24 min before step 0" was mostly this, not the step compile.
- **`jit_step` compile is flat across GRAPH-STRUCTURE knobs**: recon chunk count 1/2/4/8,
  halved C, 2-block CI fn, PPGD warmup on/off all land in the same ~4–6 min band at dp32
  (PPGD removal saves ~1 min). Don't chase graph-shrink refactors for compile time
  without new evidence.
- **Topology slope is mild**: dp32 ≈ 290–306s, dp64 ≈ 345s (same code/hardware).
- Per-pass attribution (TF_CPP_VMODULE=hlo_pass_pipeline=1, the `passes` arm):
  `jit_step` pass time is ~83% priority-fusion (+ multi-output-fusion); shardy/SPMD and
  layout-assignment are minor at these mesh sizes. The rest of the wall is
  backend codegen + (cold-cache) autotuning; the per-fusion autotune cache is
  persistent and shared, so real runs pay it once per new GEMM shape, not per config.
- `keep new seeded inits few-outputs-under-jit`: a jit returning n_sites (hundreds of)
  sharded outputs is a multi-minute SPMD/layout pass; vmap-stack per shape/C group, then
  fan out with a trivial donated slice jit (`init_decomp_vu_placed` is the template).

## Gotchas

- **`init_distributed(dp)` is config-driven, NEVER SLURM-sniffing** (`sharding.py`):
  distributedness comes from `runtime.dp` ONLY. `dp is None` → no-op (single device);
  `dp = N` → `jax.distributed.initialize` + assert `process_count() == N`. SLURM env
  (`SLURM_LOCALID`) is read only for the rank, once `dp` has decided we're distributed.
  Do NOT revert to inferring it from ambient `SLURM_PROCID` — that env is present in EVERY
  process on a SLURM box (incl. a pytest worker), so sniffing it wrongly fires
  `jax.distributed.initialize` mid-suite (the `test_pretrain` smoke failure).
- **`shard_batch` topology** (`sharding.py`): uses `make_array_from_process_local_data`
  so it's correct for BOTH single-process-many-devices and multi-process-1-device.
  Do NOT revert to the per-`process_index()`-slice idiom — it silently replicates one
  slice on single-process multi-device CPU.
- **`vendored_jax` is a repo-root sibling package in the same `param-decomp` distribution**;
  no `sys.path` hacks anywhere. If an import fails, the install is broken — fix the env
  (`make install-dev`), don't add a path shim.
- **Bench schedules**: `llama8b_real.py` anneals over `--total_steps` (default 100k),
  not the benched `--steps` — short benches measure start-of-training semantics.
