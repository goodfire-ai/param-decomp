# `param_decomp_lab/experiments/`

Composition roots for the in-repo experiments. Training is JAX now (`jsp-train`, launched
via `pd-jax-lm`); what survives here on the torch side is the **LM consumer bridge** — the
`SavedLMRun` reload class that post-processing (harvest / app / eval) imports to load a
saved decomposition off disk.

The torch TMS and ResidualMLP experiment dirs were deleted: those domains now live only as
JAX targets (`param_decomp_jax/jax_single_pool/tms.py`, `resid_mlp.py`). Their torch-free
config schemas (`param_decomp_config/tms.py`, `resid_mlp.py`) remain, since the JAX trainer
reads them.

There is no central registry — `lm/run.py` declares its own `LMExperimentConfig` +
build functions + `SavedLMRun` reload class, and post-processing callers import the
concrete reload class directly.

## Layout

The `ExperimentConfig[T,D]` generic + `EvalConfig` + `WandbConfig` +
`ResumeProvenance` live in `param_decomp_config/experiment.py`; the LM schema
(`LMExperimentConfig`, `LMTargetConfig`, `LMDataConfig`, the `target.spec` union) in
`param_decomp_config/lm.py`.

```
experiments/
├── utils.py                 # init_pd_run + EXPERIMENT_CONFIG_FILENAME
└── lm/
    ├── run.py               # LM consumer bridge (SavedLMRun reload)
    ├── jax_launch.py        # pd-jax-lm: snapshot + shared-FS workspace + sbatch
    ├── layerwise.py         # split LM YAML into per-matrix configs + SLURM-array submit
    ├── data.py
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

`output_extract` (default `"logits"`) is the key/index `make_run_batch` uses to pull
the prediction tensor out of the model's forward output. (Vendored targets return bare logits,
so `output_extract: 0`.)

`hf_weights_in_vendored` requires `model_class` to expose a `from_hf_pretrained(model_name)`
classmethod that loads real HF weights into a checkpointable, componentizable vendored arch. The vendored models live in `experiments/lm/vendored/`:
`gpt2.py` (GPT-2) and the **`llama_3_1/`** package (Llama-3.1 — `config` / `model` /
`components`). **Do not confuse the vendored `llama_3_1` with `pretrain/models/llama_simple.py`**:
the latter is a separate, small pretrain-only architecture (different MLP, ties embeddings, no
llama3 RoPE scaling) and is NOT the real-Llama decomposition target — don't retrofit it.

## Anatomy of `lm/run.py`

The LM bridge exposes the same shapes a fresh-run script and a reload path share:

```python
class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    ...  # lives in param_decomp_config/lm.py

def build_target(target_cfg) -> nn.Module: ...

def build_lm_loader(
    target_cfg, data_cfg, *,
    split: Literal["train", "eval"], device: str, batch_size: int,
    dist_state=None, seed=None,
) -> DataLoader: ...

def make_run_batch(target_cfg) -> RunBatch: ...

@dataclass(frozen=True)
class SavedLMRun:
    cfg: LMExperimentConfig
    checkpoint_path: Path

    @classmethod
    def from_path(cls, path: ModelPath) -> "SavedLMRun": ...
    def load_model(self) -> ComponentModel: ...
```

The reload class deliberately does *not* re-export the loader as a method — post-processing
code calls the free function directly with `pd_run.cfg.target` / `pd_run.cfg.data`.

There is no kind discriminator on disk — each post-processing caller imports the concrete
`SavedLMRun` it expects:

```python
from param_decomp_lab.experiments.lm.run import SavedLMRun
pd_run = SavedLMRun.from_path("entity/project/runs/<run_id>")
```

(`LMExperimentConfig` itself is imported from `param_decomp_config.lm`.)

Pydantic validation against the wrong `ExperimentConfig` subclass fails fast at YAML
load time.

## Sink + wandb wiring

`utils.py::init_pd_run(cfg, group, tags)` does the standard sink construction:

- If `cfg.wandb is None` → `RunSink.local(out_dir)`.
- Otherwise → `RunSink.with_wandb(...)` with the full `ExperimentConfig` dumped into
  `wandb.config`. Nested lists of typed configs (loss / eval metrics) are flattened
  into queryable flat keys via `flatten_typed_lists` in
  `param_decomp_config/wandb_config.py` (torch-free, so the JAX trainer logs the same
  key layout; the `short_name` table there is drift-guarded against the torch metric
  registry).

Non-main DDP ranks get `RunSink.silent()`.

### `--group` and `--tags`

Every `pd-*` run command accepts `--group <id>` and `--tags a,b,c` (no-ops when
`wandb:` is omitted):

- **`--group`** sets wandb's first-class `group` field — used by the UI's native
  collapsing and matched by workspace filters via `ws.Metric("Group")`.
  `pd-lm-layerwise` auto-generates a `lw-...` group id and stamps every child run with
  it. Manual users can pass `--group` to mark ad-hoc multi-launches.
- **`--tags`** adds wandb tags — orthogonal to `group`, many per run, user-defined.
