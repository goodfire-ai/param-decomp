# param_decomp

A JAX implementation of the **single-pool** Parameter Decomposition (VPD) training
loop — the four-term loss (faithfulness + importance-minimality + chunkwise stochastic
recon + persistent-PGD adversarial recon) as one `jax.jit` step, GSPMD-sharded,
**generic over vendored targets**: the engine sees a target only through the
`DecomposedModel` protocol; the concrete targets live in the sibling subpackage
[`param_decomp.targets`](../targets/README.md).

The semantics are pinned by [`SPEC.md`](SPEC.md) (normative: pseudocode + numbered
invariants, grounded in the torch oracle at git tag `torch-oracle`). It realized the
"single-pool SPMD collapse" hypothesis: XLA + whole-step `jit` + GSPMD sharding replaces
the hand-written-NCCL multi-pool design with zero manual collectives.

`param_decomp/core/` is the engine layer of the one `param-decomp` library, beside its
sibling subpackages `param_decomp.targets` (the concrete targets),
`param_decomp.pretrain` (the in-house target-LM pretrainer), `param_decomp.vendored_jax`
(bit-parity JAX archs), and the composition/consumer layers (`experiments`, `harvest`,
…). Imports point only downward — pinned by `tests/test_runtime_standalone.py`. Install
the whole workspace into the one venv with `make install-dev`.

## What's here

| file | what |
|---|---|
| `model.py` | `DecomposedModel` — the interface a vendored LM target implements (ordered sites, flat site-keyed dicts, frozen pytree as runtime arg) + generic chunking |
| `train.py` | the step factory: one fused jit step over faith + imp-min + the recon loss TERMS, per-persistent-term fused final ascents, fp32 masters + bf16 compute |
| `losses.py` | the pure loss terms (KL/(B·T), faithfulness, imp-min lp+entropy split) + schedules (p-anneal, source-LR warmup) |
| `adversary.py` | adversarial source machinery: persistent state + Adam ascents, fresh sign-PGD init, `source_masks` |
| `recon.py` | the flat loss surface (LOSS_PARITY_DESIGN.md): the self-describing `LossTerm` union (`FaithfulnessTerm` / `ImportanceMinimalityTerm` / `ReconLossTerm`), mask-source strategies × plans × routing samplers, and `build_loss_terms` — the shared torch loss configs mapped onto a flat tuple of terms |
| `ci_fn.py` | shared-transformer CI fn over ordered site specs; the two leaky-hard squashings (SPEC §4.6, S5/S6) |
| `checkpoint.py` | orbax sharded save/resume of `TrainState` (adversary sources + moments included, no full-gather on the loop, SPEC S22) |
| `eval.py` | in-loop eval pass: the six CE/KL masking variants + per-site CI-L0 in one jitted step, logged under the torch `EvalLoop` keys (`eval/ce_kl/*`, `eval/l0/*`) — enabled by the optional `eval:` config block |
| `slow_eval.py` | LIBRARY for the in-loop slow (plot) tier (SPEC S28, in-loop only — no offline CLI): the `CIHistograms` / `ComponentActivationDensity` / `CIMeanPerComponent` reductions + renders, the config-gated `PermutedCIPlots` / `IdentityCIError` (off the `(T, C)` position CI), the `UVPlots` figure (`render_uv_figure` / `plot_uv_matrices`, shared by the LM in-loop naive-gather path and the toy `toy_uv_eval` cheap path), and the hidden-acts recon scalars. Torch-free numpy/matplotlib; logged under `slow_eval/figures/*` |
| `components.py` | the decomposition representation: `ComponentStacks` — V/U masters persisted as same-shape STACKS (owner-partitioned, SPEC D4 amendment 2026-07-15), `site(name)` per-site views, `site_out` the decomposed-linear primitive |
| `run_state.py` | optimizer + initial-`TrainState` construction (`init_train_state(pd, model, ci_fn_arch, positions, …)`; orbax restores onto this reference) |
| `tools/` | debug tools (`liverange_peak.py`, `memreport.py`) |
| `sharding.py` | generic GSPMD helpers (`init_distributed`, `hsdp_mesh`, `place_via_shardings`, `place_target`, `shard_batch`) |
| `init_placed.py` | seeded init → placed arrays with no host-side full tree (`init_component_stacks_placed` / `init_ci_fn_placed` / `init_sources_sharded`; the few-outputs-under-jit compile doctrine) |
| `family.py` | `ArchFamily` (a target's matrix grammar as data: vocabulary + `name_of`/`parse`) + the family-parameterized `canonical_site_cs`/`site_specs` the targets delegate to. The block-structured `SiteTree` + `resolve_site_tree` (tiled c-spec → tree) live composition-side with the LM schema (`param_decomp/experiments/lm/config.py`) |
| `run.py` | the generic ENGINE `run_decomposition_training` (pure library, no `main`/YAML): faith warmup, loop, metrics jsonl/wandb, in-loop slow renderer, orbax checkpoints, SIGTERM-save + requeue-resume. The LM composition root that reads YAML + builds the target lives composition-side (`param_decomp/experiments/lm/run.py`) |
| `data.py` | deterministic batch schedule over the pre-tokenized fineweb parquet shards; O(1) resume addressing, per-process slices |
| `hf_http.py` | `configure_hf_http_retries` — idempotent retrying-adapter install on huggingface_hub (cold-cache 8N-rank startup burst); no-op without huggingface_hub; JAX-side analog of `param_decomp/infra/hf_http.py` |
| `built_run.py` | the built-run bundle the engine consumes: `BuiltRun` / `DataConfig` / `EvalConfig` / `RunInstance` + the `TargetSites` protocol. Domain-agnostic — the YAML→bundle CONVERSION lives composition-side (`experiments/config.py` shared + `experiments/lm/config.py` LM) |
| `configs.py` | the torch-free pydantic config SCHEMA: routing + the `explicit` (toy) site spec + loss-metric + eval-metric configs, `PDConfig` / `RuntimeConfig` / `Cadence` / `WandbConfig` / `ResumeProvenance`, and the `wandb.config` shaping helpers. The authored `decomposition.ci` configs AND the tiled LM site specs (`GluTransformerCSpec`/`SimpleMlpCSpec`, `LayerSelection`) speak each domain's vocabulary and live with the domain schemas, composition-side (`experiments/lm/config.py` chunkwise + tiled sites, `experiments/config.py` toy MLPs); core carries only the RESOLVED CI-fn arches (`ci_fn.py`) and resolved flat sites |
| `base_config.py` | `BaseConfig` (frozen `extra=forbid` pydantic `BaseModel` + YAML/JSON round-trip), `Probability` |
| `schedule.py` | `ScheduleConfig` + its two evaluators, host `get_scheduled_value` / traced `scheduled_value_traced` (warmup → constant/linear/cosine decay; every scheduled quantity routes through here) |
| `configs/` | the single self-contained run yamls (one file per run; no wrapper/schema split) |
| `tests/` | tiny-target unit tests (incl. attention sites + heterogeneous per-site C), checkpoint resume, sharding, and the layering test (`test_runtime_standalone.py`, pinning the downward-only subpackage layering — composition → targets → core — and the wrapper-never-imported rule). The per-target parity/golden suites (torch↔JAX equivalence, stacked parity, Qwen3 HF parity, SimpleMLP torch fixtures) live with the targets: `param_decomp/targets/tests/` |

## Run

```bash
# From the repo root — one venv for the whole workspace:
make install-dev && source .venv/bin/activate

pytest param_decomp/tests/ param_decomp/targets/tests/

# GSPMD device-count invariance (simulated devices on CPU), SPEC D4:
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  python -m param_decomp.targets.invariance_check --steps 3
```

## Design

- **Generic over vendored targets.** The trainer sees only the `DecomposedModel` fn-table
  (`model.py`): ordered `sites`, `clean_output`, `read_activations`, `masked_output`,
  `masked_site_outputs` (the hidden-acts eval seam, SPEC S31), `weight_deltas` — all
  pure, all taking the frozen pytree as a *runtime arg* (a frozen
  8B target closed over as a jit constant bakes multi-GB weights into the HLO). Adding
  a target (e.g. GPT-2) = implementing that table; no TMS/ResidMLP-style generality.
- **One jit'd step, functional minimax.** The persistent adversary (per-site sources +
  their Adam moments) lives in `TrainState` and is threaded through; `n_warmup`
  supplemental ascents + one final ascent whose gradient comes from the same backward
  as the param grads (SPEC S13/S14).
- **GSPMD, not pools.** Data `P('dp')`, params placed by the target's sharding plan,
  `jax.jit` inserts every collective. The torch `reduce_source_grads` dance is absorbed
  by autodiff of the global-mean loss. Validated by `invariance_check.py`: the
  trajectory is device-count-invariant up to float reassociation.
