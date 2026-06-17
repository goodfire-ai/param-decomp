# `param_decomp_lab/experiments/`

Composition roots for the in-repo experiments. Training is JAX now (`jsp-train`, launched
via `pd-jax-lm`); what survives here on the torch side is the **`build_target` bridge** —
`lm/run.py::build_target` rebuilds the LM target *architecture* from config so
`JaxPDAdapter` can read its layer topology (`n_blocks`, canonical layer descriptions).
Torch-run *loading* (the `SavedLMRun` reload + `component_model_io` + vendored archs) was
dropped with the torch-trainer shed and returns JAX-native as the #10 torch->jax adapter.

The torch TMS and ResidualMLP experiment dirs were deleted: those domains now live only as
JAX targets (`param_decomp_jax/jax_single_pool/tms.py`, `resid_mlp.py`). Their torch-free
config schemas (`param_decomp_config/tms.py`, `resid_mlp.py`) remain, since the JAX trainer
reads them.

## Layout

The `ExperimentConfig[T,D]` generic + `EvalConfig` + `WandbConfig` +
`ResumeProvenance` live in `param_decomp_config/experiment.py`; the LM schema
(`LMExperimentConfig`, `LMTargetConfig`, `LMDataConfig`, the `target.spec` union) in
`param_decomp_config/lm.py`.

```
experiments/
├── utils.py                 # EXPERIMENT_CONFIG_FILENAME (the JaxPDAdapter reload contract name)
└── lm/
    ├── run.py               # build_target bridge (architecture only, no checkpoint restore)
    ├── jax_launch.py        # pd-jax-lm: snapshot + shared-FS workspace + sbatch
    ├── data.py
    ├── prestage_tokenized.py
    └── pretrain/            # see lm/pretrain/CLAUDE.md
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
tensor out of the model's forward output.

`hf_weights_in_vendored` resolves `model_class` dynamically and requires it to expose a
`from_hf_pretrained(model_name)` classmethod. The vendored torch archs that class once
pointed at were dropped with torch-run loading; the spec kind survives in the config
schema (the JAX trainer's own `param_decomp_jax/vendored_jax/` loads these weights), and
`build_target` only resolves that branch if a future re-add supplies a matching class.
Note `TransformerTopology` (`path_schemas.py`) only has path schemas for the GPT-2 and
`LlamaSimple*` pretrain archs — so `JaxPDAdapter`'s topology path is exercised by
`kind: pretrained` runs (the pile `LlamaSimpleMLP` decompositions), not the raw-HF/vendored
Llama specs.

## `lm/run.py`

```python
def build_target(target_cfg: LMTargetConfig) -> nn.Module: ...
```

Loads the LM target in eval mode, dispatching on `target_cfg.spec.kind`. The only caller
is `JaxPDAdapter`. `LMExperimentConfig` is imported from `param_decomp_config.lm`;
pydantic validation against the wrong `ExperimentConfig` subclass fails fast at YAML load
time.

### `--group` and `--tags`

Every `pd-*` run command accepts `--group <id>` and `--tags a,b,c` (no-ops when
`wandb:` is omitted):

- **`--group`** sets wandb's first-class `group` field — used by the UI's native
  collapsing and matched by workspace filters via `ws.Metric("Group")`.
- **`--tags`** adds wandb tags — orthogonal to `group`, many per run, user-defined.
