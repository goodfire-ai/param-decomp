# Parameter Decomposition

Training tools for parameter decomposition on neural networks. For a compact implementation of
the core method, see [`nano_param_decomp/`](nano_param_decomp/).

## References

- **VPD paper (April 2026):** https://www.goodfire.ai/research/interpreting-lm-parameters. [VPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/vpd-paper)
  Published 4L-Pile decomposition: https://wandb.ai/goodfire/spd/runs/s-55ea3f9b.
  Current JAX reference config: [`param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml`](param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml), validated by https://wandb.ai/goodfire/param-decomp/runs/p-76082aa1.
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

`make install` is the shorthand for the first line; `make install-dev` also installs the
development dependencies and the pre-commit hooks.

## Run Experiments

A run is one self-contained YAML configuration. Start from a shipped config, make a copy
for the experiment, and run it inside the GPU allocation supplied by your compute system:

```bash
uv run python -m param_decomp.experiments.lm.run <config.yaml> --data-root <data-root>
```

`runtime.dp` in the config must equal the allocation's total GPU count;
`runtime.gpus_per_node` describes its node shape. The command does not submit a job or
choose a cluster. For example, the current JAX reference config
[`param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml`](param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml)
sets `dp: 8` and therefore needs 8 GPUs.

### Datasets

LM training reads pre-tokenized parquet shards and never tokenizes or streams source text
at run time. A named dataset resolves under `<data-root>/datasets/<name>/`; the directory
contains `meta.json` plus `shard_*.parquet`. Prepare an immutable dataset with:

```bash
uv run python -m param_decomp.experiments.lm.prestage_tokenized \
  --out-dir <data-root>/datasets/<name> \
  --dataset-repo <huggingface-dataset> --subdir <parquet-subdir> --revision <sha> \
  --column-name text --tokenizer-name <target-tokenizer> \
  --seq-len <sequence-length> --num-files <count> --skip-files 0 \
  --task-id 0 --num-tasks 1 --num-proc <cpus>
```

Then set `data.train: {kind: name, name: <name>}` in the run config. Eval metrics read
the held-out split named by `data.eval` (same `{kind: name}` shape, required): prestage
it from the same source with `--skip-files` set past the training split's file range, so
the two splits are file-disjoint. The tokenizer must already
be in the Hugging Face cache, and its identity and sequence length must match the target.
For ad-hoc local data, put the explicit `{kind: dir, dir: <path>}` escape arm under each
of `data.train` and `data.eval`.

### Pretrained target weights

For `target.spec.kind: pretrained`, `run_path` names a W&B pretrain run such as
`goodfire/spd/runs/t-9d2b8f02`; it is never a filesystem path. On first use, the library
fetches `model_config.yaml` and `model_step_<N>.safetensors` into
`<data-root>/pretrain_cache/<project>-<run-id>/`. Later runs read that cache without
network access. `python -m param_decomp.pretrain.train` writes the same layout directly
when training a target locally.

TMS and ResidualMLP run the same way — in-process module mains, on CPU:
`uv run python -m param_decomp.experiments.tms.run <config.yaml> --data-root <data-root>`
(likewise `...experiments.resid_mlp.run`). The torch
trainer is preserved only at git tag `torch-oracle`; current training uses the JAX
single-pool engine. See `param_decomp/core/SPEC.md` for its numerical contract and
`param_decomp/experiments/CLAUDE.md` for the complete LM config schema.

## Metrics

Training losses are configured in `pd.loss_metrics` as a list of `{type: "<ClassName>",
...}` entries; eval metrics in `eval.metrics`. Both are validated by the torch-free pydantic
schema in core (`param_decomp.core.configs`) and computed by the JAX trainer
(`param_decomp/core/losses.py`, `param_decomp/core/slow_eval.py`).

## Packaging

The root `pyproject.toml` builds the `param-decomp` library (the whole `param_decomp/`
package). It declares no console scripts: every runnable surface is a module main, so
the library never submits a job or chooses a machine for you.

## Development

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```
