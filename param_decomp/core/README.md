# param_decomp

A JAX implementation of the **single-pool** Parameter Decomposition (VPD) training
loop — the four-term loss (faithfulness + importance-minimality + stochastic subset
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
(bit-parity JAX archs), and the composition/consumer layers (`experiments`,
`clustering`, …). Imports point only downward — pinned by
`param_decomp/tests/core/test_runtime_standalone.py`. Install
the library into one virtual environment with `make install-dev`.

## What's here

| file | what |
|---|---|
| `model.py` | `DecomposedModel` — the interface a vendored LM target implements (ordered sites, flat site-keyed dicts, frozen pytree as runtime arg) |
| `train.py` | the step factory: one fused jit step over faith + imp-min + recon + optional nonlinearity, per-persistent-term fused final ascents, fp32 masters + bf16 compute |
| `losses.py` | pure faithfulness, importance-minimality, reconstruction-comparison, and nonlinearity losses + the jnp schedule evaluators every in-step scheduled quantity uses |
| `adversary.py` | adversarial source machinery: persistent state + Adam ascents, fresh sign-PGD init |
| `masking.py` | construction and materialization of explicit/stochastic component and weight-delta masks |
| `recon.py` | the recon term vocabulary (LOSS_PARITY_DESIGN.md): `ReconLossTerm` (one all-sites forward family = routing sampler × mask-source strategy), the mask-source strategies, the routing samplers, and the reconstruction specs |
| `objective.py` | `LossSurface` (`FaithfulnessTerm` / `ImportanceMinimalityTerm` / the `ReconLossTerm` tuple / optional `NonlinearityTerm`) and `build_objective` — the shared loss configs compiled onto the explicit-role surface |
| `nonlinearity_eval.py` | standing per-component nonlinearity-unit statistics |
| `ci_fn.py` | shared-transformer CI fn over ordered site specs; the two leaky-hard squashings (SPEC §4.6, S5/S6) |
| `checkpoint.py` | orbax sharded save/resume of `TrainState` (adversary sources + moments included, no full-gather on the loop, SPEC S22) |
| `recon_eval.py` | target-generic fresh-PGD reconstruction eval: opaque model inputs/outputs, model-owned `recon_loss_fn`, arbitrary leading axes |
| `slow_eval.py` | LIBRARY for the in-loop slow (plot) tier (SPEC S28, in-loop only — no offline CLI): the `CIHistograms` / `ComponentActivationDensity` / `CIMeanPerComponent` reductions + renders, the config-gated `PermutedCIPlots` / `IdentityCIError` (off the `(T, C)` position CI), the `UVPlots` figure (`render_uv_figure` / `plot_uv_matrices`, shared by the LM in-loop naive-gather path and the toy `toy_uv_eval` cheap path), and the hidden-acts recon scalars. Torch-free numpy/matplotlib; logged under `slow_eval/figures/*` |
| `well_temperedness.py` | whether higher causal importance preactivations mean larger reconstruction-loss changes when components are ablated one at a time, compared across all heads and layers |
| `well_temperedness_eval.py` | target-generic eval operation and rendering for the well-temperedness measurement |
| `components.py` | the decomposition representation: `ComponentStacks` — V/U masters in target-declared semantic stacks, `site(name)` per-site views, `activation_axes` the one spelling of the waist's semantic axes |
| `decomposed_linear.py` | the placed decomposed-linear primitive: `site_forward`/`site_out` (SPEC §4.1) executing one site under `PlacementRules`, a precompiled `PlannedComponentLinear`, or unplaced `None`; `constrain_component_activation` pins `[*leading, C]` tensors to the component-waist row |
| `placement.py` | `PlacementRules` — the typed placement table (components / ci_fn / activations / target rows), preset resolution (`from_config`: `owner` / `zero1` / `ddp`), the compute-weight materialization (`materialize_reduced_weights`), and the stacked-muon staging claims (see PLACEMENT_DESIGN.md) |
| `muon_stacked.py` | per-kind batched Newton-Schulz for `impl: stacked` muon; `staging_hops`, the one-axis-per-reshard waypoint chain |
| `run_state.py` | optimizer + initial-`TrainState` construction (`init_train_state(pd, model, ci_fn_arch, positions, …)`; orbax restores onto this reference) |
| `tools/` | debug tools (`memreport.py` — proto memory-report + live-range peak attribution, `hlo_census.py`, `fit_check.py` — the AOT GPU-fit check) |
| `sharding.py` | generic GSPMD helpers (`initialize_topology`, `hsdp_mesh`, `place_via_shardings`, `place_target`, `shard_batch`) |
| `init_placed.py` | seeded init → placed arrays with no host-side full tree (`init_component_stacks_placed` / `init_ci_fn_placed` / `init_sources_sharded`; the few-outputs-under-jit compile doctrine) |
| `family.py` | `ArchFamily` (a target's matrix grammar as data: vocabulary + `name_of`/`parse`) + the family-parameterized `canonical_site_cs`/`site_specs` the targets delegate to. The block-structured `SiteTree` + `resolve_site_tree` (tiled c-spec → tree) live composition-side with the LM schema (`param_decomp/experiments/lm/config.py`) |
| `run.py` | the generic ENGINE `run_decomposition_training` (pure library, no `main`/YAML): faith warmup, loop, metrics jsonl/wandb, in-loop slow renderer, orbax checkpoints, SIGTERM-save + requeue-resume. The LM composition root that reads YAML + builds the target lives composition-side (`param_decomp/experiments/lm/run.py`) |
| `built_run.py` | generic `BuiltRun[DataT, TargetT, PDT]`, `RunInstance`, and the target-sites protocol; domain data/eval plans live composition-side |
| `configs.py` | the torch-free pydantic config SCHEMA: routing + the `explicit` (toy) site spec + loss-metric + eval-metric configs, `PDConfig` / `Cadence` / `WandbConfig` / `ResumeProvenance` / `PlacementTableConfig`, and the `wandb.config` shaping helpers. The authored `decomposition.ci` configs, the tiled LM site specs (`GluTransformerCSpec`/`SimpleMlpCSpec`, `LayerSelection`) AND the LM's compute substrate (`RuntimeConfig` / `LaunchEnv`) speak each domain's vocabulary and live with the domain schemas, composition-side (`experiments/lm/config.py` chunkwise + tiled sites, `experiments/lm/runtime.py` the `runtime:` section, `experiments/toy_config.py` toy MLPs); core carries only the RESOLVED CI-fn arches (`ci_fn.py`) and resolved flat sites |
| `base_config.py` | `BaseConfig` (frozen `extra=forbid` pydantic `BaseModel` + YAML/JSON round-trip), `Probability` |
| `schedule.py` | `ScheduleConfig` + the host evaluator `get_scheduled_value` (the traced twin `scheduled_value_traced` lives in `losses.py`) — a knot-based piecewise curve `max_val × frac(t)`, interp linear/cosine/hold; every scheduled quantity — LRs, gamma, merged-loss `adv_fraction` — routes through here |
| `../experiments/*/configs/` | the domain-owned self-contained run YAMLs (one file per run; no wrapper/schema split) |
| `../tests/core/` | tiny-target engine tests (incl. attention sites + heterogeneous per-site C), checkpoint resume, sharding, and the layering test (`test_runtime_standalone.py`, pinning composition → targets → core and the wrapper-never-imported rule) |
| `../tests/targets/` | per-target parity/golden suites (torch↔JAX equivalence, stacked parity, Qwen3 HF parity, SimpleMLP torch fixtures) |

## Run

```bash
# From the repo root — one venv for the whole workspace:
make install-dev && source .venv/bin/activate

pytest param_decomp/tests/core/ param_decomp/tests/targets/

# GSPMD device-count invariance (simulated devices on CPU), SPEC D4:
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  python -m param_decomp.targets.invariance_check --steps 3
```

## Design

- **Generic over vendored targets.** The trainer sees only the `DecomposedModel` fn-table
  (`model.py`): ordered `sites`, one `clean_forward`, one `masked_forward`, and
  `weight_deltas`. Both forwards accept an immutable frozenset of canonical activation keys
  and return `ForwardResult`; its `.captures` dictionary contains exactly one value per
  requested physical activation. Each target validates and deterministically orders those
  keys privately on first trace;
  core owns no activation grammar or capture-plan type. Every method is pure and the frozen
  pytree remains a *runtime arg* (a frozen 8B target closed over as a jit constant bakes
  multi-GB weights into the HLO). An empty key set takes the untouched no-capture path.
- **One jit'd step, functional minimax.** The persistent adversary (per-site sources +
  their Adam moments) lives in `TrainState` and is threaded through; `n_warmup`
  supplemental ascents + one final ascent whose gradient comes from the same backward
  as the param grads (SPEC S13/S14).
- **GSPMD, not pools.** Data `P('dp')`, params placed by the target's sharding plan,
  `jax.jit` inserts every collective. The torch `reduce_source_grads` dance is absorbed
  by autodiff of the global-mean loss. Validated by `invariance_check.py`: the
  trajectory is device-count-invariant up to float reassociation.
