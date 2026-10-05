# `param_decomp/experiments/`

Experiment glue + the per-domain COMPOSITION ROOTS, torch-free. Training is JAX through the
generic core engine (`param_decomp.core.run.run_decomposition_training`, a pure library that reads
the pydantic `PDConfig` / `Cadence` directly). Each toy domain's `run.py` and LM's
`training.py` are composition roots: read the run YAML → build the target / data loader /
`config.BuiltRun` → call the engine. LM runs through
`python -m param_decomp.experiments.lm.run` in an allocation provided by the caller; the toy
domains (TMS, ResidMLP) run on CPU in-process via their module mains
(`python -m param_decomp.experiments.{tms,resid_mlp}.run`). The shared experiment YAML
schema + the shared run-identity helper (`run_instance`) live in `experiments/config.py`;
the toy CI-arch builder is `experiments/toy_config.py::build_toy_ci_arch`; each domain's
`config.py` carries its own target/data schema + (for the LM) its `BuiltRun` build.

## Training logs

Training metrics log every 20 steps by default, plus the final step. Maintained
configs omit `cadence.train_log_every` to use this shared default. Evaluation and
checkpoint schedules are independent.

## Runtime topology

`RuntimeConfig.world_size` derives a `SingleNode | MultiNode` value from `runtime.mesh`.
A single node has one to eight GPUs; multiple nodes each have eight GPUs and require
at least two nodes. Unsupported mesh totals fail during runtime parsing. Consumers
use `world_size.device_count` when they need the total as an integer. YAML continues
to author only the logical mesh, so the physical world cannot disagree with it.

## Attention implementations

LM configurations name both `target.attention_implementation` and
`decomposition.ci.attention.implementation` for transformer CI functions. Each field is
required and accepts exactly `flash` or `xla`. Both MHA and GQA carry the CI choice through
resolution to execution. The JAX boundary maps `flash` to its `cudnn` backend.
Neither schema validation nor loading a stored run chooses a
backend for missing fields; pins that omit the choice or spell `auto` must be updated
explicitly before use.
Transformer CI attention also requires an explicit `mask: bidirectional | causal`.
The mask choice is independent of the backend and is carried unchanged into the CI architecture.

GPU corpus configurations use `flash` where its shape and dtype requirements hold. An
unsupported cuDNN request fails at the attention boundary. CPU runs and the short,
unpadded Llama arithmetic prompts explicitly use `xla`. The full Qwen3.6 configurations
retain their target `xla` setting and choose CI attention independently. Target and CI
choices are construction inputs; loaders do not repair models after construction.

MoE configurations also name `target.expert_implementation` and
`decomposition.ci.expert_implementation` independently. Both use the same closed
implementation set: `dense_masked`, `ragged_dot`, `tokamax`, or `tokamax_split_vjp`.
Dense expert computation preserves the selected CI interface and pinned routing.

## Shared evaluation batches

Fresh PGD uses distinct training and evaluation config types. Training
`PGDReconLossConfig` / `PGDReconSubsetLossConfig` require `bc | bsc` sources;
`EvalPGDReconLossConfig` / `SlowPGDReconLossConfig` require `source_shape: c` and
`init: random`. The training and evaluation unions select the appropriate schema
for the `PGDReconLoss` YAML tag. Both evaluation cadences bind through the closed
`AnyPGDEvalConfig` union to the same probe; their tags select the schedule and
default metric name.

The context step returns `LMBatchForwardProducts[TargetIn, PreparedT, Conditioning]`,
a registered JAX dataclass containing one clean forward's tokens, output, requested
captures, CI envelope, compute weights and conditioning. `LMBatchContext.forward`
keeps these device values together alongside the batch index and persistent sources.
The input type retains document layout; consumers read the product directly instead
of interpreting a positional return tuple.

## `pd` optimizers

`pd.components_optimizer` (the V/U group) and `pd.ci_fn_optimizer` are
`core.configs.AnyOptimizerConfig` — a union discriminated on `type`, **not** a single
`OptimizerConfig`:

| `type` | class | notes |
|---|---|---|
| `adamw` | `AdamWOptimizerConfig` | the canonical one; `type` may be omitted (a `type`-less optimizer block validates as `adamw`) |
| `muon` | `MuonOptimizerConfig` | experimental, must be spelled explicitly |

Muon defaults to `ns_dtype: bfloat16` for Newton–Schulz orthogonalization; master
weights and momentum remain FP32. Set `ns_dtype: float32` explicitly when FP32
orthogonalization is required.

`consistent_rms` defaults to `0.2`, approximately matching AdamW's empirical update
RMS. Use `consistent_rms: null` explicitly for Optax's original width scaling.

The literal is `adamw`, never `adam`. `adam` IS a valid literal elsewhere in the schema —
`AdamPGDConfig.type`, the persistent-PGD adversary's own source optimizer
(`pd.loss_metrics[].optimizer` under a `PersistentPGD*` term) — a different field with a
different union; the two never substitute.

Both blocks are honored exactly as written: `run_state.build_optimizers` reads the full
`ScheduleConfig` (an arbitrary knot curve), both `betas` and `weight_decay`, and chains a
global-norm clip only where `grad_clip_norm` is non-null. A schedule is `max_val` times a
piecewise `frac` curve over normalized time `t = step / (total_steps - 1)`; a bare float
is the constant schedule, so maintained configs spell the cosine decay out as knots. The shape
below is the method's recipe, not a subspace
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

The TMS and ResidualMLP toys are small experiments that call the core engine as a library
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
  scheduler configuration — and no `runtime:` section in the YAML at all: a toy is single-device by
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
  / `plot_uv_matrices` with the LM in-loop tier. Toy `eval:` is a domain-specific
  closed schema: fresh `PGDReconLoss` runs against the target's own `recon_loss_fn` on
  independent synthetic batches; the optional `UVPlots` operation runs on the slow cadence —
  read off its own `slow` declaration, the same one the LM binder reads
  (`eval_config.schedule_for`), never a per-family choice. LM-only metrics
  (`CEandKLLosses`, `WellTemperedness`) refuse when toy evaluator construction reaches them. Ground-truth identity/dense CI scoring remains the
  toy runner's native validation pass on the train-log cadence.
- `configs/*.yaml` — the canonical `experiments.{tms,resid_mlp}.config` schema (TMS: 5-2 /
  40-10 / the `-id` deeper variants; ResidMLP: 1l/2l/3l).

The TMS deeper variant (`n_hidden_layers>0`, the `-id` configs) and the toy `global_mlp`
CI arch (`type: global_mlp`) are wired end-to-end (the global arch dispatches through the
core `init_pd_state` via `toy_config.build_toy_ci_arch`). The shipped ResidMLP configs
all use per-site `layerwise_mlp` with `hidden_dims: [400]` at `C: 200` per site — wide
enough to avoid the output bottleneck the `global_mlp` variant escapes.

`CIMeanPerComponent` also binds on the toys: it samples held-out eval batches,
uses the shared example-weighted CI reductions, and emits linear/log PNGs through the
configured W&B transport. Its standalone operation compiles at plan preparation, like
the fast toy metrics, using the configured fixed eval-batch shape.

Toy offline consumers are not wired (`load_run` is LM-only).

## Picking a CI-fn arch — and `n_blocks: 0`

`CIFn.has_position_axis` must equal the target's (`core.run_state.init_decomposition` asserts
it). The chunkwise transformer is the positioned arch whose blocks self-attend OVER the
position axis. The global transformer (`type: global_transformer`) runs one transformer
over the selected taps of every decomposed block, with an output head per site; its blocks
stack along `depth` and each semantic group's heads along `site`, which its `owner` rows
own over `replicate`.
`component_conditioning: {kind: affine_clipped_component_activation, output_scale_init: …,
calibration: {min_n_tokens: …}}` on the global transformer config also
conditions every site's CI on its own component activations `h = x @ V` of the clean site
input: positive and negative arms clamping affine maps of `h` and `-h`, and a free bias.
Every arm's output scale starts at `output_scale_init`; every arm's input scale starts at
`1 / q`, `q` each component's exact 99.9th percentile of `|h|` over the whole step-0
training batch. That batch must hold no padding, at least `min_n_tokens` tokens, and a
whole number of the calibration's `CALIBRATION_CHUNK_TOKENS`-token chunks;
targeted training calibrates on its non-target stream. Both fields are required
(`experiments/lm/ci_fn_init.py`, the place for the LM's data-dependent CI init). V stays a
component parameter; its CI gradient joins the component update. It requires a dense transformer target
(GLU or SimpleMLP), whose anatomy names each site's input capture. The LM schema also admits
`type: global_mlp` (the tPD paper's LM CI net) —
ONE shared MLP over every decomposed block's `input_tap` taps, pointwise per token, so it
is positioned with no cross-position read at all.

For a sequence target that is the point. It is fatal when the position axis is large and
derived: a pair-shaped target (an AF2-style pair representation, where a position is a residue
PAIR) turns a 128-residue crop into 16384 positions — 16k×16k attention per chunk per forward,
several times per step. Infeasible, not slow.

**The answer is `LayerwiseMLPCIFnArch(has_position_axis=True)`.** The MLP arches were positionless by
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

The LM schema (`ChunkwiseTransformerCIFnConfig.n_blocks`) stays `PositiveInt`: an LM's positions
are tokens, cross-position CI is what that arch is for there, and 0 would be a typo rather than
a choice. `n_blocks: 0` is available to any domain whose own schema admits it.

## Layout

Shared LM batches, input access, and shard loading live in `param_decomp/lm/`;
`param_decomp/pretrain/` owns target-LM pretraining. These modules do not depend on
experiment composition roots. The generic core does not import the LM package.

The `ExperimentConfig` schema base (domain subclasses bind concrete
`target`/`decomposition`/`data`) + the shared validation / run-identity helpers live in
`experiments/config.py`; `EvalConfig` lives in `experiments/eval_config.py`; the toy
authored CI configs (`LayerwiseMlpCIFnConfig` / `GlobalMlpCIFnConfig`) plus
`build_toy_ci_fn_arch` live in `experiments/toy_config.py` (`WandbConfig` / `ResumeProvenance` are
core, in `param_decomp.core.configs`; the engine's `BuiltRun` bundle is core, in
`param_decomp.core.built_run`); the LM schema + LM build (`LMExperimentConfig`, `LMTargetConfig`,
`LMDataConfig` (`lm/run_data.py`), the `target.spec` union, the authored LM CI union (`LMCIFnConfig`:
`ChunkwiseTransformerCIFnConfig` and `GlobalTransformerCIFnConfig` over their shared
`TransformerCIFnConfig` backbone + its `attention`/`ffn` unions, and the LM `GlobalMlpCIFnConfig`
— its own class, not the toy one: it selects taps via `CIFnInputTapSelection`, resolved by
`resolve_lm_ci_fn_arch` over ALL decomposed blocks) AND the
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
│   ├── run_targeted.py      # the tPD twin: bootstrap deferring to training_targeted.py; a targeted run is its own top-level config shape (LMTargetedExperimentConfig: prompts: + nontarget:), never a mode flag
│   ├── targeted_data.py     # the tPD TARGET stream: kind-discriminated prompt pools, tokenized once at startup, unpadded at one shared prompt length
│   ├── resolved.py          # LM-only resolved data/run types (ResolvedLMData, LMRun)
│   ├── eval.py              # token CE/KL + CI-L0 fast pass
│   ├── attn_patterns_eval.py / arithmetic_eval.py
│   ├── ci_position_eval.py  # the `CIActiveCountsPerPosition` slow metric: CI_L0's count at every sequence position (CI > 0 / 0.01 / 0.1), one native W&B line chart at `eval/l0/per_position`
│   ├── router_divergence_eval.py  # the `RouterDivergence` metric: per MoE layer, the masked model's expert router vs the target's under the target's routing pinned everywhere
│   ├── well_temperedness.py / well_temperedness_eval.py  # the `WellTemperedness` slow metric (kernel + operation)
│   ├── data.py              # pack encoded documents
│   ├── llama3/              # Llama 3 preprocessing and staging
│   ├── abstract_inputs.py   # abstract batches for trace and fit checks
│   └── arithmetic_probe.py    # a x b arithmetic grid spec -> in-memory prompt grid (the tPD pool's layer) + eval probe (ArithmeticCIGrid, adds the single-token-answer premise)
├── tms/                     # TMS (CPU): run.py + configs/ (target: param_decomp/targets/tms.py; also the tPD engine's test fixture — no shipped toy tPD shape)
└── resid_mlp/               # ResidMLP (CPU): run.py + configs/ (target: param_decomp/targets/resid_mlp.py)

Tests mirror these roots under `param_decomp/tests/experiments/`.
```

`lm/well_temperedness.py` measures whether components with higher causal-importance
preactivations damage reconstruction more when ablated one at a time, at sampled token
positions: components are compared across all heads and layers, separately for
preactivations below 0, between 0 and 1, and above 1, and named `groups` add the same
measurement for site subsets (attention vs MLP, say). It is LM-only — the kernel dispatches
on the `LMOutput` edge and needs a position axis — so
`lm/eval_operations.py` binds it and the toy binder refuses it.

## Router divergence evaluation

`RouterDivergence` reports KL over all expert probabilities, top-k overlap and
`weight_mae` over the selected mixing weights. Each strategy produces one router
readout per batch; stochastic masking uses one draw. Sums normalize by tokens. A persistent `bc` or
`bsc` source samples one complete training row per evaluation sequence, shared across
sites, while `c` and `sc` sources broadcast directly. This row sampler is separate
from grouped source pools; pooled terms are not accepted by this evaluation strategy.

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

# or
target:
  spec:
    kind: pretrained_qwen35_moe         # a lab-pretrained Qwen35Moe toy on the qwen36_moe engine
    run_path: <entity>/<project>/<run_id>       # the W&B pretrain run — arch from its model_config.yaml
```

### `kind: pretrained` / `pretrained_qwen35_moe` — `run_path`

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
refuses at convert time. Each registry entry pairs its immutable architecture with a loader
parameterized by that exact config type. Its load method resolves sites against that
architecture and forwards the authored dtype, output edge, and attention implementation
to construction. A dense Qwen3 run requires Qwen3-tokenized, document-aware shards: source IDs are
preserved in standard training, targeted non-target batches and evaluation. Token-only
Qwen3 inputs are rejected; historical replay uses its original pinned code and data.


## LM `target.output_edge`

Required on every LM config — a discriminated union on `kind`: `materialized` (the
forward forms its full `[B, S, vocab]` logits in the target's native dtype) or
`streamed` (`n_vocab_chunks` chunks; the forward returns the factored
`targets.lm_output.StreamedLinearOutput` package and every comparison streams over the
vocab axis with fp32-accumulated chunk logits, `targets.losses`). Both edges serve all
LM families. The chunk count must divide the target vocabulary; Llama-3.1-8B accepts
32 chunks of 4008 tokens, and Qwen3.6 accepts 32 chunks of 7760 tokens.

```yaml
target:
  output_edge: {kind: streamed, n_vocab_chunks: 32}   # 248320 = 32 · 7760
  # or
  output_edge: {kind: materialized}
```

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

A store name resolves to `<data_root>/datasets/<name>` (`infra.dataset_store`). Provision
that directory before running; `experiments.lm.llama3.prestage_tokenized` can create
Llama shards and metadata. The dataset directory is self-describing:
`meta.json` (`infra.dataset_store.DatasetMeta`) records its sequence length, tokenizer,
and preprocessing policy. Document-aware shards pair token IDs with document IDs; Qwen3.6
requires token-only shards. Tip reads pinned configs through this same strict schema; older shapes require their original revision or an external converter.

The JAX prediction is the final logits or their factored streamed representation
(there is no `output_extract`; current configs reject that torch-era field). The `model_class` strings
are NOT imported by the JAX trainer — `experiments/lm/config.py::resolve_decomposition`
only asserts the class identity (`kind: hf` matches the family's full class string; the
other kinds match the class-name suffix) and routes to its own vendored JAX arch
(`pretrained` LlamaSimpleMLP -> the pretrain-cache loader, `hf_weights_in_vendored`
Llama -> `target_ports`). The dotted `model_class` is a stable identifier only, never
imported. `pretrained_qwen35_moe` carries no `model_class`: its kind already names the
engine, and the arch is read from the cache entry's `model_config.yaml`.

The path schemas (`topology/path_schemas.py`) cover the pretrain (`GPT2*`,
`LlamaSimple*`) and HF GLU (`Llama`, `Qwen3`) architectures used to name model sites consistently.

## `runtime.launch_env` (rank env / XLA flags)

The rank environment (XLA client flags, NCCL/host-memory knobs) is config-driven via
`runtime.launch_env` (`param_decomp.experiments.lm.runtime.LaunchEnv`), so `config.yaml`
fully captures authored settings. `lm/run.py` exports
`LaunchEnv.as_env(os.environ.get("XLA_FLAGS"))` before importing JAX, so it applies to
direct module invocation. Existing `XLA_FLAGS` compose with the config rather than being
replaced: disjoint flags append, identical duplicates dedupe, and conflicting values fail
loudly. Machine-specific environment such as `LD_LIBRARY_PATH` belongs to the caller rather
than the authored run configuration.

XLA *compiler* flags go through `runtime.compiler_options` instead (passed natively to
every jit, in the compile-cache key). REQUIRED, no default, no merge — every run's
flags trace to a visible authored token: `tuned-v2` = the frozen tuned set
(`TUNED_V2_COMPILER_OPTIONS` in `lm/runtime.py`, the one code copy — a changed tuned
set is a new preset name, never an edit). It keeps while-loop double-buffering off
(that pass keeps O(1) extra copies of the while tuple — fatal for the
`*-replicated-resident` placements, whose loops carry whole-depth ÷tp weight stacks as
xs), so the one set serves every placement. `tuned-v2-autotune1` = tuned-v2 plus
`xla_gpu_autotune_level: 1` (`TUNED_V2_AUTOTUNE1_COMPILER_OPTIONS`) — the faster-iteration
preset: shorter first compile, and since autotune picks steer fusion, its arena
and step time do not transfer to tuned-v2; `bare` = `{}` (true XLA
defaults, the debugging baseline); or an explicit `xla_*`-keyed dict, used VERBATIM as
the run's complete flag set (non-`xla_*` keys refuse at parse). The retired `tuned-v1`
family refuses at parse naming its successor: jaxlib 0.11 removed the three
`xla_gpu_enable_pipelined_*` compile options those presets froze.

```yaml
runtime:
  mesh: {replicate: 4, fsdp: 8, tp: 1}   # or {data: D, tp: T} for *-replicated-resident
  compiler_options: tuned-v2   # or tuned-v2-autotune1, bare, or an explicit xla_* dict
  launch_env:
    xla_python_client_allocator: platform
    env: { SOME_ONE_OFF_VAR: "1" }
```

## W&B grouping and tags

The shipped TMS and ResidualMLP configs include `wandb:` and require authentication. Tests
that must not contact W&B should use a temporary config with `wandb: null`; local
`metrics.jsonl` output is still written in the run directory.

The TMS and ResidualMLP module mains accept `--group <id>` and `--tags a,b,c`
(no-ops when `wandb:` is omitted). Callers may supply the same W&B fields:

- **`--group`** sets wandb's first-class `group` field — used by the UI's native
  collapsing and matched by workspace filters via `ws.Metric("Group")`.
- **`--tags`** adds wandb tags — orthogonal to `group`, many per run, user-defined.

## Auxiliary reconstruction comparisons

Training and fresh-PGD evaluation use the same explicit `auxiliaries` groups. Each
entry specifies `name`, `coeff`, and `comparisons: [{capture, distance}]`. Capture
names come from the target; distance is `relative_squared_error` or
`categorical_kl_from_logits`. Each group contributes its coefficient times its own
mean, and its name determines only its logged metric prefix. An empty list gives
output-only reconstruction. Evaluation accepts constant zero coefficients to retain
readouts without changing ascent; training accepts positive constants or schedules.

For router reconstruction, enumerate the Qwen target's `router_logits_capture_keys`
with categorical KL, using the name `router_kl` to preserve existing metric paths.
The router's frozen weights and clean expert selections are unchanged; the captured
logits measure counterfactual scores under the ordinary masked forward. Hidden readouts
use relative squared error and can retain the name `hidden_acts_reconstruction`.

Stored configs using the former `hidden_acts_reconstruction: {coeff, points}` or
`router_kl: {coeff}` syntax must be translated to these explicit groups before loading
with this schema. No automatic compatibility conversion runs when loading a config.
Current training checkpoints also store frequency estimators inside the algorithm-specific
training-state record. Older training items require a one-off external migration before
resume; there is no compatibility reader. The separately checkpointed `decomposition`
item is unchanged.
