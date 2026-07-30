# `param_decomp_lab/experiments/`

Composition roots for the in-repo experiments. Each experiment is a plain Python script
that parses a YAML, builds the target / loaders / metrics, and runs a `Trainer`.

There is no central registry — each `run.py` declares its own `<Name>ExperimentConfig`
+ build functions + `Saved<Name>Run` reload class, and post-processing callers import
the concrete reload class directly.

## YAML schema

One validated pydantic tree (extra keys raise):

```yaml
pd:      { ... PDConfig ... }
runtime: { ... RuntimeConfig ... }
cadence: { train_log_every, save_every }
target:  { ... per-experiment target config ... }
data:    { ... per-experiment data config ... }
label:   "<series id>"                              # optional: groups runs in the index
eval:    { batch_size, n_steps, every, slow_every, slow_on_first_step,
           metrics: [ {type: "...", ...}, ... ] }   # optional: omit to skip eval
wandb:   { project: ..., entity: ... }              # optional: omit to skip wandb
nontarget: { data: {...}, batch_size, eval_batch_size,
             impmin_coeff_ratio }                   # optional: targeted (tPD) runs only
```

`eval.metrics` entries are dispatched via `EVAL_METRIC_CLASSES` (see
[`../eval_metrics/CLAUDE.md`](../eval_metrics/CLAUDE.md)). `slow_every` must be a
multiple of `every`.

The optional `nontarget:` block (`NontargetConfig` in `param_decomp_lab/targeted.py`)
turns the run into a targeted decomposition: a second per-step pass over the nontarget
distribution with the delta component forced on. Requires `pd.use_delta_component: true`,
no `FaithfulnessLoss`, and `faithfulness_warmup_steps: 0`. Toy data configs take
`active_indices` to restrict the target distribution; LM data configs take a
`prompts_file` (exactly one of `dataset_name` / `prompts_file`). Example:
`tms/tms_40-10-id_targeted_config.yaml`.

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
```

`output_extract` (default `"logits"`) is the key/index `make_run_batch` uses to pull
the prediction tensor out of the model's forward output.

## Anatomy of `run.py`

Every experiment script exposes the same five top-level shapes:

```python
class <Name>ExperimentConfig(ExperimentConfig[<Name>TargetConfig, <Name>DataConfig]):
    pass

def build_target(target_cfg) -> nn.Module: ...

def build_<name>_loader(
    target_cfg, data_cfg, *,
    split: Literal["train", "eval"], device: str, batch_size: int,
    dist_state=None, seed=None,
) -> DataLoader: ...

def make_run_batch(target_cfg) -> RunBatch: ...

@dataclass(frozen=True)
class Saved<Name>Run:
    cfg: <Name>ExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "Saved<Name>Run": ...
    def load_model(self) -> ComponentModel: ...
```

The loader is per-experiment-named (`build_lm_loader`, `build_tms_loader`,
`build_resid_mlp_loader`) so cross-imports don't shadow each other. The reload class
deliberately does *not* re-export the loader as a method — post-processing code calls
the free function directly with `pd_run.cfg.target` / `pd_run.cfg.data`.

`main()` calls the module-level build functions directly; `Saved<Name>Run` delegates
to those same functions, so there's no duplication between "fresh run from YAML" and
"reload from disk" paths.

`main()` writes the resolved `<Name>ExperimentConfig` to `out_dir / EXPERIMENT_CONFIG_FILENAME`
via `cfg.to_file(...)`. There is no kind discriminator on disk — each post-processing
caller imports the concrete `Saved<Name>Run` it expects:

```python
from param_decomp_lab.experiments.lm.run import SavedLMRun
pd_run = SavedLMRun.from_path("entity/project/runs/<run_id>")
```

Pydantic validation against the wrong `ExperimentConfig` subclass fails fast at YAML
load time.

## Adding a new experiment

Drop a `run.py` next to a YAML config. Implement the five shapes above. Then either:

1. Invoke `python -m param_decomp_lab.experiments.<name>.run config.yaml`, or
2. Add a console script in `param_decomp_lab/pyproject.toml`:
   ```toml
   pd-<name> = "param_decomp_lab.experiments.<name>.run:cli"
   ```

No central registry to update — post-processing callers `import` the new
`Saved<Name>Run` directly from its module.

## Sink + wandb wiring

`utils.py::init_pd_run(cfg, group, tags)` does the standard sink construction:

- If `cfg.wandb is None` → `RunSink.local(out_dir)`.
- Otherwise → `RunSink.with_wandb(...)` with the full `ExperimentConfig` dumped into
  `wandb.config`. Nested lists of typed configs (loss / eval metrics) are flattened
  into queryable flat keys via `flatten_typed_lists` in `infra/wandb.py`.

Non-main DDP ranks get `RunSink.silent()`.

### Resume in place (walltime-split legs)

`pd-lm --resume <yaml>` continues the parent run **in place** by default: with no
explicit `--run_id`, it reuses the parent run id (`from_run.name`), so all legs share
one run folder and one wandb run. `init_pd_run(..., resume=True)` reopens the wandb run
(`resume="allow"`) and preserves the original `started_at`. Pass an explicit `--run_id`
to branch a separate run instead. The `~/pd_scratch/autoresume.py` walltime harness
relies on this — it resubmits each leg with no run_id rather than minting `-rN` runs.

A fine-tune leg that changes the algorithm mid-trajectory sets `pd:` in the resume
YAML (`ResumeConfig.pd`) — a full replacement `PDConfig` (copy the parent's block and
edit) that overrides the snapshot's saved one. `steps` must exceed the resumed step;
structural fields must match the parent (optimizer/PPGD state comes from the
snapshot). Typical use: extend `steps` and switch on a γ-anneal for a fine-tune,
branched with `--run_id`.

### `--group` and `--tags`

Every `pd-*` run command accepts `--group <id>` and `--tags a,b,c` (no-ops when
`wandb:` is omitted):

- **`--group`** sets wandb's first-class `group` field — used by the UI's native
  collapsing and matched by workspace filters via `ws.Metric("Group")`.
  `pd-lm-layerwise` auto-generates a `lw-...` group id and stamps every child run with
  it. Manual users can pass `--group` to mark ad-hoc multi-launches.
- **`--tags`** adds wandb tags — orthogonal to `group`, many per run, user-defined.

## Saved-run layout

```
PARAM_DECOMP_OUT_DIR/runs/<run_id>/
  experiment_config.yaml     # the full ExperimentConfig
  run_metadata.json          # started_at/finished_at + git_commit/uncommitted_changes
  model_<step>.pth           # checkpoints (RunSink.checkpoint)
  metrics.jsonl              # local logs (RunSink.log)
```

Post-decomposition pipelines (harvest, autointerp, attributions, graph_interp) nest
their own sub-directories under this same `runs/<run_id>/` dir — see each module's
CLAUDE.md.

## Canonical references

- `param_decomp_lab/experiments/tms/run.py` — smallest, simplest reference.
- `param_decomp_lab/experiments/resid_mlp/run.py` — toy model with feature_importances.
- `param_decomp_lab/experiments/lm/run.py` — discriminated `target.spec`, DDP via
  `with_distributed_cleanup`.
