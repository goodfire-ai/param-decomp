# Parameter Decomposition

Training tools for parameter decomposition on neural networks. For a compact implementation of
the core method, see [`nano_param_decomp/`](nano_param_decomp/).

## Research guidance

Read the [parameter-decomposition handbook](docs/handbook.md) for the
science, evidence standards, and failure modes. The
[parameter-decomposition skill](docs/skill.md) is the experiment-driving
guide for target implementation, sweep design, convergence, selection, and analysis.

## References

- **Tiny Qwen3.5 MoE:** the current canonical decomposition run is
  [p-63932d23](https://wandb.ai/goodfire/param-decomp/runs/p-63932d23), targeting
  [t-82543a24](https://wandb.ai/goodfire/param-decomp/runs/t-82543a24)
  (four layers, width 512, 42 experts, top-8 routing). For a fresh run on main, use
  [the maintained config](param_decomp/experiments/lm/configs/pile_qwen3_5_moe-4L.yaml);
  its comments identify differences from that run.
- **SPD paper (June 2025):** https://arxiv.org/abs/2506.20790. [SPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/v1).

## Install

The public package is self-contained. Create the environment from the locked repository:

```bash
uv sync --frozen --no-dev                    # CPU
uv sync --frozen --no-dev --extra cuda       # NVIDIA driver r525-r579
uv sync --frozen --no-dev --extra cuda13     # NVIDIA driver r580+
```

**Blackwell GPUs require `--extra cuda13` and driver r580 or newer.** The CUDA-12 lock
contains cuBLAS older than 13.2, which can silently corrupt execution on Blackwell rather
than merely failing. Use `--extra cuda` only for Ampere, Ada, or Hopper hosts whose driver
cannot load the CUDA-13 wheels.

`make install` is the library-only shorthand. For development setup, see
[Contributing](#contributing).

## Run Experiments

A run is one self-contained YAML configuration. Start from a shipped config, make a copy
for the experiment, and run it inside the GPU allocation supplied by your compute system:

```bash
uv run python -m param_decomp.experiments.lm.run <config.yaml> \
  --data-root <data-root> --local-device-count <devices-per-process>
```

The `runtime.mesh` axes' product must equal the allocation's total GPU count.
Process-local allocation size is a launch argument, not a logical mesh axis. The
command does not submit a job or choose a cluster. For example, the current JAX reference config
[`param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml`](param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml)
sets `mesh: {replicate: 1, fsdp: 32, tp: 1}`, and therefore needs 32 GPUs.

### Maintained LM configs

- Maintain **one LM decomposition config per model** in
  [`param_decomp/experiments/lm/configs/`](param_decomp/experiments/lm/configs/), not one
  per experiment, optimizer, layer selection, topology, or test.
- Configs on `main` must stay runnable with the documented data, weights, and hardware;
  schema changes must update affected configs in the same PR.
- Link the latest **human-selected canonical decomposition run**, if one exists, in a
  comment at the top of each config. Only humans designate canonical runs, and rarely.
- A maintained config is the team's **current best guess for that model**, not a frozen
  copy of its canonical run. Update it when justified, even before a new run completes
  or becomes canonical. Comment beside each departure from the linked run, explaining
  what differs and why, with evidence where available (e.g. a new eval absent from that
  run, or a new initialization supported by a linked report on another model).
  Updating the config does **not** change which run is canonical.
- Keep one-off configs outside the repo and stored runs' pinned configs immutable.

### Datasets

- LM training reads staged, pre-tokenized Parquet shards; it does not tokenize or
  stream source text at runtime.
- Set named train/eval refs in `data.train` and `data.eval`; each resolves to
  `<data-root>/datasets/<name>/`. See the [config/data guide](param_decomp/experiments/CLAUDE.md#lm-data).
- Match the dataset tokenizer and row width to the target; use the
  [maintained configs](param_decomp/experiments/lm/configs/) for model-specific choices.
- Maintained Llama 3.1 and dense Qwen3 configs require stored document IDs; Pile 4L
  and hybrid Qwen retain token-only rows. For Llama preparation, see the
  [prestager](param_decomp/experiments/lm/llama3/prestage_tokenized.py).
- Dense Qwen3 data uses text + `<|endoftext|>` (151643), no BOS. Token-only Qwen3
  data is unsupported.

### Pretrained target weights

For `target.spec.kind: pretrained` (and `pretrained_qwen35_moe`, the Qwen3.5-MoE toys),
`run_path` names a W&B pretrain run such as
`goodfire/spd/runs/t-9d2b8f02`; it is never a filesystem path. On first use, the library
fetches `model_config.yaml` and `model_step_<N>.safetensors` into
`<data-root>/pretrain_cache/<project>-<run-id>/`. Later runs read that cache without
network access. `python -m param_decomp.pretrain.train` writes the same layout directly
when training a target locally.

TMS and ResidualMLP run the same way — in-process module mains, on CPU:
`uv run python -m param_decomp.experiments.tms.run <config.yaml> --data-root <data-root>`
(likewise `...experiments.resid_mlp.run`). The shipped toy configs log to W&B by default,
so authenticate with `wandb login` or set `WANDB_API_KEY` before running them. For local-only
tests, copy the config and set `wandb: null`; the run still writes `metrics.jsonl` under
`<data-root>/runs/<run-id>/`. See [the experiments guide](param_decomp/experiments/CLAUDE.md)
for the complete LM config schema.

## Metrics

W&B logging defaults to the existing metric names. New runs can opt into two-level
names with `wandb.metric_schema: grouped`. See [W&B metric names](docs/wandb-metrics.md).

Training losses are configured in `pd.loss_metrics` as a list of `{type: "<ClassName>",
...}` entries; eval metrics in `eval.metrics`. Both are validated by the torch-free pydantic
schema in core (`param_decomp.core.configs`) and computed by the JAX trainer
(`param_decomp/core/losses.py`, `param_decomp/core/slow_eval.py`).

Fresh and persistent training PGD require `source_shape: bc` or `bsc`, giving each
batch row independent adversarial state. Fresh evaluation PGD uses separate config
types requiring `source_shape: c` and `init: random`; the `PGDReconLoss` YAML tag is
shared, with its schema selected by the containing training or evaluation list.
Changing an older `sc` training config to `bsc` changes its adversary and multiplies
source and optimizer-buffer storage by the global batch size. Stored-run pins require
their original revision. Training checkpoints created before frequency estimators moved
into the plain and targeted training-state records require a one-off external migration
before resume; the loader intentionally has no compatibility reader. The separately
checkpointed `decomposition` item is unchanged.

The [analytical MFU calculator](param_decomp/experiments/lm/model_flops.py) estimates useful FLOPs from an LM
decomposition config and reports MFU with and without optimizer work from measured
step time, without loading weights or compiling the training step.

The root `pyproject.toml` builds the `param-decomp` library (the whole `param_decomp/`
package). It declares no console scripts: every runnable surface is a module main, so
the library never submits a job or chooses a machine for you.

## Contributing

Run `make install-dev` at the repository root to install the library, development tools,
and pre-commit hooks. Run commands with `uv run` or activate `.venv`.

### Checks

Use the narrowest useful check while iterating:

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make lint      # what CI enforces: ruff without fixes, the `|=` check, basedpyright
make test      # testmon-selected tests, excluding slow
make test-all  # all tests
```

CI runs `make test-all`'s selection, every test including those marked `slow`, on every
pull request and push to `main`, sharded across parallel jobs. The shards are an exact
partition of the suite pinned by `param_decomp/tests/infra/test_ci_shards.py`; a new test
directory must be placed in one.

Python `|=` is rejected by `make format`, pre-commit, and CI. Use
`dict_safe_update_` for dictionary composition so duplicate keys fail; use explicit
`.update()` for intentional replacement or set unions, and assignment for bitwise operations.

Never bypass pre-commit hooks. Add files explicitly rather than with `git add .`.

### Code style and architecture

- `param_decomp/` is the public library. It must not know where it runs: no scheduler,
  submission, code-shipping, cluster path, mount, partition, or team namespace. Paths are
  explicit required inputs, and configs identify external resources by portable names.
- Training uses the JAX single-pool engine. The retired Torch implementation is only a
  semantic oracle at git tag `torch-oracle`; `nano_param_decomp/` is a standalone Torch
  reference and is not imported by either package.
- Keep the functional core pure and put I/O at entry points. Encode invariants in types when
  possible and assert the rest. Fail closed rather than adding fallbacks, compatibility shims,
  or degraded modes.
- Dispatch on a tag or kind is an exhaustive `match` with every arm written out — no `case _`
  catch-alls. Where basedpyright proves the match exhaustive, that proof is the fail-closed
  mechanism (a trailing `case _: raise` is dead code and is rejected); where it cannot, the last
  arm raises.
- Operations are locally truthful: each is correct given only what its input types promise, and
  the types carry those guarantees across seams. An operation whose correctness rests on an
  informal bound elsewhere in the program (a value "known" to be in range because something
  upstream clips it) is a defect — encode the bound in the type at the seam (a bounded type, a
  discriminated union) or make the operation correct for every value its input type admits.
- No code is written for a test's convenience. A test that needs a known state computes its
  expectation from real, in-domain inputs; production code never gains an out-of-range knob,
  debug flag, or special arm so a test can reach a state.
- Import public names from the modules that define them; package-level re-exports are
  exceptional. Update the nearest guide when changing a documented structure or interface.

For local architecture and interfaces, read the nearest module's `README.md` or
`CLAUDE.md`.
