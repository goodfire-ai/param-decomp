# `param_decomp/experiments/`

Experiment glue + the per-domain COMPOSITION ROOTS, torch-free. Training is JAX through the
generic core engine (`param_decomp.core.run.run_decomposition_training`, a pure library that reads
the pydantic `PDConfig` / `Cadence` directly). Each toy domain's `run.py` and LM's `training.py` are composition roots: read the run YAML → build the target / data loader / `config.BuiltRun` → call the
engine. LM runs through `python -m param_decomp.experiments.lm.run` in an allocation
provided by the caller; the toy domains (TMS, ResidMLP) run on CPU
in-process via their module mains (`python -m param_decomp.experiments.{tms,resid_mlp}.run`). The shared experiment YAML schema + the shared
run-identity helper (`run_instance`) live in `experiments/config.py`; the toy CI-arch
builder is `experiments/toy_config.py::build_toy_ci_arch`; each domain's `config.py`
carries its own target/data schema + (for the LM) its `BuiltRun` build.

## `pd` optimizers

`pd.components_optimizer` (the V/U group) and `pd.ci_fn_optimizer` are
`core.configs.AnyOptimizerConfig` — a union discriminated on `type`, **not** a single
`OptimizerConfig`:

| `type` | class | notes |
|---|---|---|
| `adamw` | `AdamWOptimizerConfig` | the canonical one; `type` may be omitted (a `type`-less optimizer block validates as `adamw`) |
| `muon` | `MuonOptimizerConfig` | experimental, must be spelled explicitly |

The literal is `adamw`, never `adam`. `adam` IS a valid literal elsewhere in the schema —
`AdamPGDConfig.type`, the persistent-PGD adversary's own source optimizer
(`pd.loss_metrics[].optimizer` under a `PersistentPGD*` term) — a different field with a
different union; the two never substitute.

Both blocks are honored exactly as written: `run_state.build_optimizers` reads the full
`ScheduleConfig` (an arbitrary knot curve), both `betas` and `weight_decay`, and chains a
global-norm clip only where `grad_clip_norm` is non-null. A schedule is `max_val` times a
piecewise `frac` curve over normalized time `t = step / (total_steps - 1)`; a bare float
is the constant schedule, so the seats spell the cosine decay out as knots. The shape
below is the METHOD's recipe (SPEC S20) — what the maintained seats run, not a subspace
the schema enforces:

```yaml
pd:
  components_optimizer:            # type omitted => adamw
    lr_schedule:
      max_val: 7.0e-05
      points:
        - {at: 0.0, frac: 1.0}
        - {at: 1.0, frac: 0.1, interp: cosine}
    betas: [0.9, 0.999]
    weight_decay: 0.0
    grad_clip_norm: 0.01
  ci_fn_optimizer:
    lr_schedule:
      max_val: 7.0e-05
      points:
        - {at: 0.0, frac: 1.0}
        - {at: 1.0, frac: 0.1, interp: cosine}
    betas: [0.9, 0.999]
    weight_decay: 0.0
    grad_clip_norm: null
```

## Toy domains (TMS, ResidMLP)

The TMS and ResidualMLP toys are LAB experiments that call the core engine as a library
(the core itself has zero target-specific code). The toy *targets* live in the targets
distribution — `param_decomp/targets/{tms,resid_mlp}.py`: the JAX `DecomposedModel`
(sites, pure fns, MSE `recon_loss_fn`), the frozen target (`eqx.Module`), from-scratch
in-process pretrain (`pretrain_*_target`), the ground-truth identity-CI eval
(`identity_ci_error` + the single-feature probe), and the `*TargetConfig` dataclass
carried on `BuiltRun.target` (satisfies the core `built_run.TargetSites` protocol).
Each `experiments/{tms,resid_mlp}/` carries:
- `run.py` — the toy composition root (module main): builds the core `BuiltRun` from the
  canonical schema via the public shared helpers
  (`config.run_instance` / `toy_config.build_toy_ci_arch`),
  pretrains + builds the target, and calls `run_decomposition_training` with a synthetic
  `sample_batch` plus domain-bound identity/PGD/UV eval operations. CPU, synchronous, no
  SLURM — and no `runtime:` section in the YAML at all: a toy is single-device by
  construction (`sharding.single_device_mesh`, which asserts that world rather than
  absorbing whatever devices are visible), so the engine's substrate arguments are
  literals here (`ddp` placement, no remat, no `compiler_options`), not config.
  The native toy operation ALSO logs the per-site-permuted CI heatmap alongside each checkpoint
  (`toy_uv_eval.render_permuted_ci_heatmap`, unconditional, no config gate — the visual
  companion to the `IdentityCIError`/dense-CI-error scalars: each site permutes toward
  ITS target pattern, identity via Hungarian assignment or dense via column-mass sort —
  e.g. TMS's frozen `hidden_layers.*` and ResidMLP's `mlp_out` target dense, not identity)
  and renders the config-gated `UVPlots` figure when the run's `eval.metrics` names it
  (`toy_uv_eval.render_uv_metric`): the toys feed `UVPlots` their probe CI as the
  column-permutation source and their small on-host V/U, sharing `slow_eval.render_uv_figure`
  / `plot_uv_matrices` with the LM in-loop tier (SPEC S28). Toy `eval:` is a domain-specific
  closed schema: fresh `PGDReconLoss` runs against the target's own `recon_loss_fn` on
  independent synthetic batches; optional `UVPlots` and target-generic `WellTemperedness`
  operations run on the slow cadence (the latter samples the actual eval distribution, not the
  single-feature probe) — read off
  its own `slow` declaration, the same one the LM binder reads (`eval_config.schedule_for`;
  SPEC S29), never a per-family choice. LM-only
  token metrics refuse when toy evaluator construction reaches them. Ground-truth identity/dense CI scoring remains the
  toy runner's native validation pass on the train-log cadence.
- `configs/*.yaml` — the canonical `experiments.{tms,resid_mlp}.config` schema (TMS: 5-2 /
  40-10 / the `-id` deeper variants; ResidMLP: 1l/2l/3l).

The TMS deeper variant (`n_hidden_layers>0`, the `-id` configs) and the toy `global_mlp`
CI arch (`type: global_mlp`) are wired end-to-end (the global arch dispatches through the
core `init_train_state` via `toy_config.build_toy_ci_arch`). The shipped ResidMLP configs
all use per-site `layerwise_mlp` with `hidden_dims: [400]` at `C: 200` per site — wide
enough to avoid the output bottleneck the `global_mlp` variant escapes.
Toy clustering is not wired (`load_run` is LM-only).

## Picking a CI-fn arch — and `n_blocks: 0`

`CIFn.has_position_axis` must equal the target's (`core.run_state.init_decomposition` asserts
it). The chunkwise transformer is the positioned arch whose blocks self-attend OVER the
position axis; the LM schema also admits `type: global_mlp` (the tPD paper's LM CI net) —
ONE shared MLP over every decomposed block's `input_tap` taps, pointwise per token, so it
is positioned with no cross-position read at all.

For a sequence target that is the point. It is fatal when the position axis is large and
derived: a pair-shaped target (an AF2-style pair representation, where a position is a residue
PAIR) turns a 128-residue crop into 16384 positions — 16k×16k attention per chunk per forward,
several times per step. Infeasible, not slow.

**The answer is `LayerwiseMLPCIArch(has_position_axis=True)`.** The MLP arches were positionless by
DECLARATION, not by construction — `SiteMLP.__call__` is `[*leading, d_in] -> [*leading, C]`,
pointwise over every leading axis, so the same weights serve `[batch, d]` and
`[batch, position, d]` alike. The arch now carries the axis (the target's shape, not the MLP's
property) and the runtime checks it against the model. Pinned by
`param_decomp/tests/core/test_ci_fn_positioned_mlp.py`.

`n_blocks: 0` on the chunkwise arch also runs and is also position-local (pinned by
`test_ci_fn_zero_blocks.py`), but it is a weaker instrument: the chunkwise path RMS-norms
every tap before `in_proj`, so a blockless chunk is `RMSNorm → affine` — no learned
nonlinearity, no hidden layer, and **invariant to tap magnitude** (measured: `f(x) == f(7x)`
to 3.6e-7, where the MLP's outputs differ by 14). Use it as a cheap baseline, not as the
position-local CI function.

The LM schema (`ChunkwiseTransformerCiConfig.n_blocks`) stays `PositiveInt`: an LM's positions
are tokens, cross-position CI is what that arch is for there, and 0 would be a typo rather than
a choice. `n_blocks: 0` is available to any lab whose own schema admits it.

## Layout

The `ExperimentConfig` schema base (domain subclasses bind concrete
`target`/`decomposition`/`data`) + the shared validation / run-identity helpers live in
`experiments/config.py`; `EvalConfig` lives in `experiments/eval_config.py`; the toy
authored CI configs (`LayerwiseMlpCiConfig` / `GlobalMlpCiConfig`) plus
`build_toy_ci_arch` live in `experiments/toy_config.py` (`WandbConfig` / `ResumeProvenance` are
core, in `param_decomp.core.configs`; the engine's `BuiltRun` bundle is core, in
`param_decomp.core.built_run`); the LM schema + LM build (`LMExperimentConfig`, `LMTargetConfig`,
`LMDataConfig`, the `target.spec` union, the authored LM CI union (`LMCiConfig`:
`ChunkwiseTransformerCiConfig` + its `attention`/`ffn` unions, and the LM `GlobalMlpCiConfig`
— its own class, not the toy one: it selects taps via `ChunkInputTap`, resolved by
`resolve_lm_ci_arch` over ALL decomposed blocks) AND the
tiled site specs + their resolution (`GluTransformerCSpec`/`SimpleMlpCSpec` over
`LayerSelection`, keys typed by each target family's matrix vocabulary;
`resolve_site_tree` → the block-structured `SiteTree` the chunkwise CI resolver consumes) —
target-anatomy vocabulary, so it lives in the domain that IS transformers —
`build_from_schema` / `load_config`) in `experiments/lm/config.py`; the toy schemas in
`experiments/{tms,resid_mlp}/config.py`. Core (`param_decomp.core.configs`) carries no authored
`decomposition.ci` config and no tiled site spec — only the resolved CI-fn arches
(`param_decomp.core.ci_fn`) and the family grammar (`param_decomp.core.family`).

```
experiments/
├── lm/
│   ├── run.py               # python -m param_decomp.experiments.lm.run — pre-JAX env bootstrap deferring to training.py, the LM composition root
│   ├── run_targeted.py      # the tPD (SPEC §11) twin: bootstrap deferring to training_targeted.py; a targeted run is its own top-level config shape (LMTargetedExperimentConfig: prompts: + nontarget:), never a mode flag
│   ├── targeted_data.py     # the tPD TARGET stream: kind-discriminated prompt pools, tokenized once at startup, unpadded at one shared prompt length (T8)
│   ├── resolved.py          # LM-only resolved data/run types (ResolvedLMData, LMRun)
│   ├── eval.py              # token CE/KL + CI-L0 fast pass
│   ├── attn_patterns_eval.py / arithmetic_eval.py
│   ├── data.py              # tokenize_and_concatenate (offline helper for prestage)
│   ├── prestage_tokenized.py  # HF text -> int32 parquet shards for the JAX trainer
│   └── arithmetic_probe.py    # a x b arithmetic grid spec -> in-memory eval probe (ArithmeticCIGrid)
├── tms/                     # TMS (CPU): run.py + configs/ (target: param_decomp/targets/tms.py; also the tPD engine's test fixture — no shipped toy tPD shape)
└── resid_mlp/               # ResidMLP (CPU): run.py + configs/ (target: param_decomp/targets/resid_mlp.py)

Tests mirror these roots under `param_decomp/tests/experiments/`.
```

## Sites and the family grammar

Read `param_decomp/core/family.py`'s module docstring before authoring a
`decomposition.sites` c-spec or a new target. In short: site names are layer-indexed
(`layers.{i}.self_attn.q_proj`, `h.{i}.mlp.c_fc`, …), the spelling and the within-block
matrix order are declared as DATA by the target's `ArchFamily`, and the c-spec's `cs` keys
are typed by that same vocabulary — so which layers get decomposed is a config choice
(`layers: {kind: all | range | list}`), not a target property. A new target declares its
family and gets any layer subset for free; layers without sites run the plain frozen block.

## LM `target.spec`

The LM target is a discriminated union on `kind`:

```yaml
target:
  spec:
    kind: hf                            # HuggingFace model
    model_class: transformers.LlamaForCausalLM
    model_name: meta-llama/Llama-3.1-8B

# or
target:
  spec:
    kind: pretrained                    # in-repo lab-pretrained model
    model_class: param_decomp.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP
    run_path: <entity>/<project>/runs/<run_id>   # the W&B pretrain run — see below

# or
target:
  spec:
    kind: hf_weights_in_vendored        # HF weights loaded into a vendored, componentizable arch
    model_class: param_decomp.experiments.lm.vendored.llama_3_1.model.VendoredLlama
    model_name: meta-llama/Llama-3.1-8B

# or
target:
  spec:
    kind: hf                            # Qwen3-8B-Base (same vendored JAX target + QK-norm)
    model_class: transformers.Qwen3ForCausalLM
    model_name: Qwen/Qwen3-8B-Base
```

### `kind: pretrained` — `run_path`

`run_path` names the W&B pretrain run whose checkpoint is the target's weights — a name,
never a location. It resolves to the local store entry
`<data_root>/pretrain_cache/<project>-<run_id>/` (one `model_step_<N>.safetensors` plus
`model_config.yaml`), fetched from W&B on first use and read from disk ever after
(`infra/pretrain_cache.py::resolved_cache_dir` — a complete entry short-circuits, so cold
starts are idempotent across ranks and requeues; `targets/` stays network-free). A local
`param_decomp.pretrain.train` run writes the same layout directly as its output, so
pretrain-here-then-decompose never touches W&B.

Torch-era pretrain runs ship `model_step_<N>.pt`, which this loader can't read. The fetch
downloads it anyway and then fails pointing at the local file and the converter at git
tag `torch-oracle` — conversion needs torch, which the library deliberately doesn't depend on.

`kind: hf`/`hf_weights_in_vendored` model names must be in `experiments/lm/config.py::HF_MODEL_VARIANTS`
(Llama-3.1-8B; dense Qwen3 0.6B/1.7B/4B/8B/14B Base and post-trained) — anything else
refuses at convert time. A Qwen3 run needs a Qwen3-tokenized prestaged dataset
(`prestage_tokenized --tokenizer_name <the selected Qwen checkpoint>`).

## LM `data`

`data` carries two required dataset references — `train` and the held-out `eval` split —
each a discriminated union on `kind`:

```yaml
data:
  train:
    kind: name                    # a named store dataset (the portable form)
    name: fineweb_llama_tok_2048  # pile_neox_tok_512 for LlamaSimpleMLP
  eval:
    kind: name
    name: fineweb_llama_tok_2048_eval

# or, per split:
  train:
    kind: dir                     # ad-hoc escape hatch: an explicit shard dir
    dir: /abs/path/to/shards
```

A store name resolves to `<data_root>/datasets/<name>` (`infra.dataset_store`); the
deployment populates the store (the root CLAUDE.md describes ours). The dataset dir is self-describing:
`meta.json` (`infra.dataset_store.DatasetMeta`) carries its seq_len and tokenizer, read
at load time — prestage writes it alongside the shards. Tip reads pinned configs through
this same strict schema; older shapes require their original revision or an external converter.

The JAX prediction tensor is always the final logits (there is no `output_extract` —
it was a torch-era field and current configs reject it). The `model_class` strings
are NOT imported by the JAX trainer — `experiments/lm/config.py::resolve_decomposition`
only asserts the class identity (`kind: hf` matches the family's full class string; the
other kinds match the class-name suffix) and routes to its own vendored JAX arch
(`pretrained` LlamaSimpleMLP -> the pretrain-cache loader, `hf_weights_in_vendored`
Llama -> `vendored_jax`). The dotted `model_class` is a stable identifier only, never
imported.

The path schemas (`topology/path_schemas.py`) cover the pretrain (`GPT2*`,
`LlamaSimple*`) and HF GLU (`Llama`, `Qwen3`) architectures used to name harvested
sites consistently.

## `runtime.launch_env` (rank env / XLA flags)

The rank env (XLA client flags, NCCL/host-memory knobs) is config-driven via
`runtime.launch_env` (`param_decomp.experiments.lm.runtime.LaunchEnv`), so `config.yaml`
fully captures the environment a run executed with — A/B a flag in the YAML, not in
the launcher. `lm/run.py` exports `LaunchEnv.as_env()` before importing JAX, so it applies
on every path including a direct module invocation. Machine-specific environment such as
`LD_LIBRARY_PATH` belongs to the caller rather than the authored run configuration.

XLA *compiler* flags go through `runtime.compiler_options` instead (passed natively to
every jit, in the compile-cache key). REQUIRED, no default, no merge — every run's
flags trace to a visible authored token: `tuned-v1` = the frozen production set
(`TUNED_V1_COMPILER_OPTIONS` in `lm/runtime.py`, the one code copy — a changed tuned
set is a new preset name, never an edit); `bare` = `{}` (true XLA defaults, the
debugging baseline); or an explicit `xla_*`-keyed dict, used VERBATIM as the run's
complete flag set (non-`xla_*` keys refuse at parse).

```yaml
runtime:
  replicate: 4
  fsdp: 8
  tp: 1
  compiler_options: tuned-v1   # or bare, or an explicit complete xla_* dict
  launch_env:
    xla_python_client_allocator: platform
    env: { SOME_ONE_OFF_VAR: "1" }
```

## W&B grouping and tags

The shipped TMS and ResidualMLP configs include `wandb:` and require authentication. Tests
that must not contact W&B should use a temporary config with `wandb: null`; local
`metrics.jsonl` output is still written in the run directory.

Launchers may stamp a W&B group and tags when `wandb:` is configured:

- **`--group`** sets wandb's first-class `group` field — used by the UI's native
  collapsing and matched by workspace filters via `ws.Metric("Group")`.
- **`--tags`** adds wandb tags — orthogonal to `group`, many per run, user-defined.
