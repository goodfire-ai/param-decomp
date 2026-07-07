# `param_decomp_lab/experiments/`

Experiment glue + the per-domain COMPOSITION ROOTS, torch-free. Training is JAX through the
generic core engine (`param_decomp.run.run_decomposition_training`, a pure library that reads
the pydantic `PDConfig` / `Cadence` directly). Each domain's `run.py` is its composition
root: read the run YAML → build the target / data loader / `config.BuiltRun` → call the
engine. LM runs go to SLURM via `pd-lm` (which sbatches
`python -m param_decomp_lab.experiments.lm.run`); the toy domains (TMS, ResidMLP) run on CPU
in-process via `pd-tms` / `pd-resid-mlp`. The shared experiment YAML schema + the shared
validation / run-identity helpers (`assert_canonical_algorithm_config` / `run_instance` /
`ci_arch`) live in `experiments/config.py`; each domain's `config.py` carries its own
target/data schema + (for the LM) its `BuiltRun` build.
autointerp/clustering read a run's target topology from
`experiments.lm.load_run.run_metadata` (config + pretrain cache, no checkpoint restore) —
see `param_decomp_lab/adapters/pd.py`.

## Toy domains (TMS, ResidMLP)

The TMS and ResidualMLP toys are LAB experiments that call the core engine as a library
(the core itself has zero toy-specific code). Each `experiments/{tms,resid_mlp}/` carries:

- `model.py` — the JAX `DecomposedModel` (sites, pure fns, MSE `recon_loss_fn`), the frozen
  target (`eqx.Module`), from-scratch in-process pretrain (`pretrain_*_target`), the
  ground-truth identity-CI eval (`identity_ci_error` + the single-feature probe), and the
  lab `*TargetConfig` dataclass carried on `config.BuiltRun.target` (satisfies the core
  `config.TargetSites` protocol).
- `run.py` — the `pd-tms` / `pd-resid-mlp` CLI: builds the `config.BuiltRun` from the
  canonical schema via the public shared helpers
  (`config.assert_canonical_algorithm_config` / `run_instance` / `ci_arch`),
  pretrains + builds the target, and calls `run_decomposition_training` with a synthetic
  `sample_batch` + an `identity_ci_error` `eval_fn`. CPU, synchronous, no SLURM. The toy
  `eval_fn` ALSO renders the config-gated `UVPlots` figure when the run's `eval.metrics`
  names it (`toy_uv_eval.log_uv_figure`): the toys feed `UVPlots` their probe CI as the
  column-permutation source and their small on-host V/U, sharing `slow_eval.render_uv_figure`
  / `plot_uv_matrices` with the LM in-loop tier (SPEC S28). The toy `BuiltRun.eval`
  stays `None` (the toy validates via the target-CI metric, not the LM scalar pass); the
  `eval.metrics` list is read straight off the raw schema dict (`toy_uv_eval.toy_uv_spec`).
- `configs/*.yaml` — the canonical `experiments.{tms,resid_mlp}.config` schema (TMS: 5-2 /
  40-10 / the `-id` deeper variants; ResidMLP: 1l/2l/3l + the global-CI variant).

TMS deeper variant (`n_hidden_layers>0`, the `-id` configs) + the ResidMLP `global` CI arch
(`fn_type=global_shared_mlp`) are restored and wired end-to-end (the global arch dispatches
through the core `init_train_state` via `experiments.config.ci_arch`). Toy harvest /
autointerp / clustering is NOT yet wired (`load_run` is LM-only) — the remaining Phase-3 bucket.

## Layout

The `ExperimentConfig[T,D]` schema generic + `EvalConfig` + the shared validation /
run-identity helpers live in `experiments/config.py` (`WandbConfig` / `ResumeProvenance` are
core, in `param_decomp.configs`; the engine's `BuiltRun` bundle is core, in
`param_decomp.built_run`); the LM schema + LM build (`LMExperimentConfig`, `LMTargetConfig`,
`LMDataConfig`, the `target.spec` union, `build_from_schema` / `load_config`) in
`experiments/lm/config.py`; the toy schemas in `experiments/{tms,resid_mlp}/config.py`.

```
experiments/
├── utils.py                 # EXPERIMENT_CONFIG_FILENAME
├── lm/
│   ├── launch.py        # pd-lm: snapshot + shared-FS workspace + sbatch
│   ├── data.py              # tokenize_and_concatenate (offline helper for prestage)
│   ├── prestage_tokenized.py  # HF text -> int32 parquet shards for the JAX trainer
│   └── arithmetic_probe.py    # a x b arithmetic grid spec -> in-memory eval probe (ArithmeticCIGrid)
├── tms/                     # pd-tms (CPU): model.py + run.py + configs/ + test_tms.py
└── resid_mlp/               # pd-resid-mlp (CPU): model.py + run.py + configs/ + test
```

## LM `target.spec`

The LM target is a discriminated union on `kind`:

```yaml
target:
  spec:
    kind: hf                            # HuggingFace model
    model_class: transformers.GPT2LMHeadModel
    model_name: openai-community/gpt2

# or
target:
  spec:
    kind: pretrained                    # in-repo lab-pretrained model
    model_class: param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP
    run_path: goodfire/spd/runs/<run_id>

# or
target:
  spec:
    kind: hf_weights_in_vendored        # HF weights loaded into a vendored, componentizable arch
    model_class: param_decomp_lab.experiments.lm.vendored.llama_3_1.model.VendoredLlama
    model_name: meta-llama/Llama-3.1-8B
```

The JAX prediction tensor is always the final logits (there is no `output_extract` —
it was a torch-era field, stripped on load for back-compat). The `model_class` strings
are NOT imported by the JAX trainer — `param_decomp.built_run` only asserts the class-name
suffix and routes to its own vendored JAX arch (`pretrained` LlamaSimpleMLP -> the
pretrain-cache loader, `hf_weights_in_vendored` Llama -> `vendored_jax`). The dotted
`model_class` is a stable identifier only, never imported.

The path schemas (`topology/path_schemas.py`) cover the GPT-2 and `LlamaSimple*` archs —
so `PDAdapter`'s layer-description path is exercised by `kind: pretrained` runs (the
pile `LlamaSimpleMLP` decompositions), the production target.

## `runtime.launch_env` (rank env / XLA flags)

The SLURM rank env (XLA flags, NCCL/host-memory knobs, `PD_*` profiling toggles) is
config-driven via `runtime.launch_env` (`param_decomp.configs.LaunchEnv`), so `config.yaml`
fully captures the environment a run executed with — A/B a flag in the YAML, not in
`launch.py`. `launch.py::_render_rank_env` renders `LaunchEnv.as_env()` into the exported
bash block; `LD_LIBRARY_PATH` is computed at submit time (machine-specific) and stays in the
launcher. A profiling run is a config (`runtime.launch_env.profile`), not an env hack. The
defaults mirror the values the launcher used to hardcode. Applies to the SLURM path only;
the inline `dp is None` path inherits the caller's environment.

```yaml
runtime:
  dp: 32
  launch_env:
    xla_flags: { gpu_enable_command_buffer: "", gpu_autotune_level: "0" }
    xla_python_client_allocator: platform   # was the old `pd-lm --allocator` flag
    profile: { mem_profile: true, no_checkpoint: true }
    env: { SOME_ONE_OFF_VAR: "1" }           # escape hatch, merged last (overrides)
```

## `--group` and `--tags`

Every `pd-*` run command accepts `--group <id>` and `--tags a,b,c` (no-ops when
`wandb:` is omitted):

- **`--group`** sets wandb's first-class `group` field — used by the UI's native
  collapsing and matched by workspace filters via `ws.Metric("Group")`.
- **`--tags`** adds wandb tags — orthogonal to `group`, many per run, user-defined.
