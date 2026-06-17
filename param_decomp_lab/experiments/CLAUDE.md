# `param_decomp_lab/experiments/`

LM experiment glue, torch-free. Training is JAX (`jsp-train`, launched via `pd-jax-lm`).
The torch `build_target` bridge + the `pretrain/` dir were DELETED with the rest of torch:
autointerp/clustering read a run's target topology from
`jax_single_pool.load_run.run_metadata` (config + pretrain cache, no checkpoint restore) —
see `param_decomp_lab/adapters/jax_pd.py`. The torch TMS and ResidualMLP experiment dirs
were deleted too: those domains now live only as JAX targets
(`param_decomp_jax/jax_single_pool/tms.py`, `resid_mlp.py`). The torch-free config schemas
(`param_decomp_config/{tms,resid_mlp}.py`) remain, since the JAX trainer reads them.

## Layout

The `ExperimentConfig[T,D]` generic + `EvalConfig` + `WandbConfig` +
`ResumeProvenance` live in `param_decomp_config/experiment.py`; the LM schema
(`LMExperimentConfig`, `LMTargetConfig`, `LMDataConfig`, the `target.spec` union) in
`param_decomp_config/lm.py`.

```
experiments/
├── utils.py                 # EXPERIMENT_CONFIG_FILENAME
└── lm/
    ├── jax_launch.py        # pd-jax-lm: snapshot + shared-FS workspace + sbatch
    ├── data.py              # tokenize_and_concatenate (offline helper for prestage)
    └── prestage_tokenized.py  # HF text -> int32 parquet shards for the JAX trainer
```

## LM `target.spec`

The LM target is a discriminated union on `kind`:

```yaml
target:
  spec:
    kind: hf                            # HuggingFace model
    model_class: transformers.GPT2LMHeadModel
    model_name: openai-community/gpt2
  output_extract: logits

# or
target:
  spec:
    kind: pretrained                    # in-repo lab-pretrained model
    model_class: param_decomp_lab.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP
    run_path: goodfire/spd/runs/<run_id>
  output_extract: 0

# or
target:
  spec:
    kind: hf_weights_in_vendored        # HF weights loaded into a vendored, componentizable arch
    model_class: param_decomp_lab.experiments.lm.vendored.llama_3_1.model.VendoredLlama
    model_name: meta-llama/Llama-3.1-8B
  output_extract: 0
```

`output_extract` (default `"logits"`) is the key/index used to pull the prediction
tensor out of the model's forward output. The `model_class` strings are NOT imported by
the JAX trainer — `jax_single_pool.config` only asserts the class-name suffix and routes
to its own vendored JAX arch (`pretrained` LlamaSimpleMLP -> the pretrain-cache loader,
`hf_weights_in_vendored` Llama -> `vendored_jax`). They reference the deleted torch
`pretrain/` module only as identifiers.

The path schemas (`topology/path_schemas.py`) cover the GPT-2 and `LlamaSimple*` archs —
so `JaxPDAdapter`'s layer-description path is exercised by `kind: pretrained` runs (the
pile `LlamaSimpleMLP` decompositions), the production target.

## `--group` and `--tags`

Every `pd-*` run command accepts `--group <id>` and `--tags a,b,c` (no-ops when
`wandb:` is omitted):

- **`--group`** sets wandb's first-class `group` field — used by the UI's native
  collapsing and matched by workspace filters via `ws.Metric("Group")`.
- **`--tags`** adds wandb tags — orthogonal to `group`, many per run, user-defined.
