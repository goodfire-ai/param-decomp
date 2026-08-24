# experiments/lm/pretrain — in-house target-LM pretraining (JAX)

Pretrains the FROZEN in-house target LMs that the decomposition trainer (`param_decomp.experiments.lm.run`)
then decomposes — e.g. the pile-pretrained `llama_simple_mlp` (target `t-9d2b8f02`). Small,
simple ML: next-token CE, AdamW, cosine LR + warmup. JAX-native (equinox), torch-free.

The torch original (`torch-oracle:param_decomp/experiments/lm/pretrain/`) was a
torchrun trainer; this is a capability reimplementation (NOT bit-exact), reusing the JAX
single-pool trainer's data/sharding/checkpoint substrate.

## Process boundary

The trainer is a library subpackage under `param_decomp/pretrain/`:

- **`param_decomp/pretrain/`** — the trainer:
  - `models.py` — trainable equinox defs for all three archs (`GPT2Simple`, `LlamaSimple`,
    `LlamaSimpleMLP`). Weights are stored in torch `nn.Linear` orientation `(d_out, d_in)`
    and `state_dict()` emits the EXACT keys the decomposition loader reads.
  - `config.py` — `PretrainConfig` (the self-contained run yaml schema).
  - `train.py` — `python -m param_decomp.pretrain.train <config.yaml>`: the composition root + only I/O layer. fp32
    masters, AdamW (decay on 2D weights only), cosine+warmup, grad clip, orbax sharded
    checkpoints, SIGTERM→save→requeue→resume. Reuses `param_decomp.pretrain.batch_data` (offline
    pre-tokenized parquet, never streamed) + `param_decomp.core.sharding`.
  - `cache.py` — writes the decomposition trainer's `pretrain_cache/<project>-<run_id>/`
    layout (safetensors + `model_config.yaml`) at every save.
  - `configs/` — the run yamls (`pile_llama_simple_mlp-*`, `gpt2_simple-2L`,
    `pile_llama_simple-4L-768`, `*_SMOKE`).
- **`param_decomp/experiments/lm/pretrain/`** (here):
  - `run_info.py` — `find_pretrain_cache(data_root, project, run_id)`: an unused legacy
    read-side index. The cache resolver consumers actually use is
    `param_decomp/infra/pretrain_cache.py::resolved_cache_dir`.

## Cache compatibility (load-bearing)

A freshly-pretrained target is decomposable with NO conversion. `pretrain.train` writes
`<data_root>/pretrain_cache/<project>-<run_id>/model_step_<N>.safetensors` keyed
`h.{i}.attn.{q,k,v,o}_proj.weight`, `h.{i}.mlp.{c_fc,down_proj}.weight`,
`h.{i}.rms_{1,2}.weight`, `wte.weight`, `ln_f.weight` (NO `lm_head.weight` — tied), every
weight `(d_out, d_in)` — exactly what `param_decomp.targets.llama_simple_mlp` reads. A
decomposition config points at it via `target.spec`:

```yaml
target:
  spec:
    kind: pretrained
    model_class: param_decomp.experiments.lm.pretrain.models.llama_simple_mlp.LlamaSimpleMLP
    run_path: <entity>/<project>/runs/<run_id>
```

(`run_path` resolves to `pretrain_cache/<project>-<run_id>`.) The pretrain model's forward
is bit-identical to the decomposition loader's clean-forward round-trip — pinned by
`param_decomp/core/tests/test_pretrain.py`.

## Data

Offline pre-tokenized parquet ONLY (the prestage tool's output;
`param_decomp.pretrain.batch_data.ShardServer`). Shards are `block_size + 1` wide; the trainer serves
the full row and splits `x = tokens[:, :block]`, `y = tokens[:, 1:]` inside the step. The
ported configs point at the staged `datasets/pile_neox_tok_512` (the torch configs'
SimpleStories/streaming data is not staged — runtime tokenization is deliberately
unsupported).

## Usage

The mode is config-driven via `dp`; there are no `--nodes` or `--local` flags. `dp = N`
declares a distributed world of `N` devices and requires the caller to start the matching
process topology. `dp = null` runs on the devices visible to one process.

```bash
# Single process; the config leaves `dp` unset.
python -m param_decomp.pretrain.train \
  param_decomp/pretrain/configs/pile_llama_simple_mlp-2L-128_SMOKE.yaml
```
