# jax_single_pool

A JAX implementation of the **single-pool** Parameter Decomposition (VPD) training
loop — the four-term loss (faithfulness + importance-minimality + chunkwise stochastic
recon + persistent-PGD adversarial recon) as one `jax.jit` step, GSPMD-sharded,
**generic over vendored LM targets**.

The semantics are pinned by [`SPEC.md`](SPEC.md) (normative: pseudocode + numbered
invariants, grounded in the stable torch `param_decomp` implementation). The current
implementation-vs-spec audit lives in [`AUDIT.md`](AUDIT.md). This is the research
counterpart to the torch FSDP single-pool path (`param_decomp_lab/fsdp/`), testing the
"single-pool SPMD collapse" hypothesis: XLA + whole-step `jit` + GSPMD sharding
replaces the hand-written-NCCL multi-pool design with zero manual collectives.

## What's here

| file | what |
|---|---|
| `lm.py` | `DecomposedLM` — the interface a vendored LM target implements (ordered sites, flat site-keyed dicts, frozen pytree as runtime arg) + generic chunking |
| `train.py` | the step factory: one fused jit step over the four losses + adversary, fp32 masters + bf16 compute, `make_train_step` / `make_faith_warmup_step` |
| `losses.py` | the pure loss terms (KL/(B·T), faithfulness, imp-min lp+entropy split) + schedules (p-anneal, source-LR warmup) |
| `adversary.py` | both adversaries' source machinery: persistent (PPGD: state + Adam ascents) vs fresh sign-PGD (per-step), `source_masks` |
| `recon.py` | stochastic-recon plans: `ReconForward`/`ReconPlan`, uniform-k routing samplers, `subset_chunk_plan` |
| `ci_fn.py` | shared-transformer CI fn over ordered site specs; the two leaky-hard squashings (SPEC §4.6, S5/S6) |
| `checkpoint.py` | orbax sharded save/resume of `TrainState` (adversary sources + moments included, no full-gather on the loop, SPEC S22) |
| `eval.py` | in-loop eval pass: the six CE/KL masking variants + per-site CI-L0 in one jitted step, logged under the torch `EvalLoop` keys (`eval/ce_kl/*`, `eval/l0/*`) — enabled by the optional `eval:` config block |
| `torch_config.py` | the shared-config route: a wrapper yaml (`torch_config:` + run identity + the remat knob) routes through `param-decomp-config`'s `LMExperimentConfig` and converts the supported subspace onto `ExperimentConfig` — asserts loudly on anything this trainer doesn't implement |
| `run_state.py` | optimizer + initial-`TrainState` construction from an `ExperimentConfig` — shared by `run.py` and the exporter (orbax restores onto this reference) |
| `export.py` | `jsp-export <run_dir> [--step N]` — orbax checkpoint → `<run_dir>/export/model_<step>.safetensors` with the torch `LMComponentModel`'s exact state-dict keys (V/U destacked per site, CI fn in-proj/out-head permuted to torch's sorted site order, frozen target included), so the torch eval/harvest/postprocess stack runs on JAX runs |
| `tools/` | export round-trip verification: `gen_export_fixture.py` (JAX venv) + `verify_export_torch.py` (torch venv, rebuilds the real torch modules from the safetensors and matches forwards at fp32 tolerance) |
| `sharding.py` | generic GSPMD helpers (`init_distributed`, `dp_mesh`, `replicate`, `shard_batch`) |
| `llama8b.py` | Llama-3.1-8B target: residual-start suffix + `Prefix` harvest, arbitrary per-layer matrix sites (`q/k/v/o/gate/up/down`, per-site C; q/k/v decomposed before RoPE/SDPA), per-site `DecompVU`, HF safetensors loader, `llama_decomposed_lm(cfg, sites)` |
| `llama_simple_mlp.py` | `LlamaSimpleMLP` pile-pretrained target (`goodfire/spd/runs/t-9d2b8f02`: 4L, d768, GELU MLP, plain rotate-half RoPE, tied head): sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}` with `h.*` wildcard expansion, pretrain-cache safetensors loader (one-off `.pt` conversion: `tools/convert_llama_simple_mlp_checkpoint.py`), `llama_simple_mlp_decomposed_lm(cfg, sites)`; frozen weights small enough to replicate (`replicate_frozen`), V/U/CI/source placement reuses the generic per-site plan |
| `run.py` | the training entrypoint (`jsp-train <config.yaml>`): data, faith warmup, loop, metrics jsonl/wandb, orbax checkpoints, SIGTERM-save + requeue-resume |
| `data.py` | deterministic batch schedule over the pre-tokenized fineweb parquet shards; O(1) resume addressing, per-process slices |
| `config.py` | the trainer's internal `ExperimentConfig` (built only by `torch_config.py`): shared pydantic loss/adversary configs passed through + jax-runtime knob structs |
| `configs/` | wrapper yamls (`*_from_torch.yaml`) + the torch `LMExperimentConfig` yamls they reference under `torch/` |
| `slurm/` | push-triggered offline-eval sbatch scripts (training launches go through `pd-jax-lm`, which generates the job script) |
| `llama8b_sharding.py` | the 8B placement plan (frozen replicated; per-site V/U + CI + Adam C-sharded; source replicated; batch sharded) |
| `experiments/llama8b_real.py` | the runnable 8B step + tok/s/GPU bench |
| `experiments/invariance_check.py` | device-count invariance harness (SPEC D4) |
| `tests/` | tiny-target unit tests (incl. attention sites + heterogeneous per-site C), checkpoint resume, sharding, `tests/equivalence/` — the fixture-driven torch↔JAX loss-term equivalence harness — `tests/stacked_parity/` — fixtures pinning the pre-site-generality stacked implementation (clean logits bit-identical, train trajectory rel ≤ ~1e-5) — and `tests/simple_mlp_equivalence/` — torch-fixture logits parity for the LlamaSimpleMLP target (tiny random model max abs diff ~2e-7; real t-9d2b8f02 weights ~5e-5 fp32) |

## Run

```bash
cd nano_param_decomp_jax
uv venv .venv && source .venv/bin/activate && uv pip install -e ../param_decomp_config -e .
# (`vendored_jax` — the bit-parity JAX Llama — is part of this distribution.)

pytest jax_single_pool/tests/

# GSPMD device-count invariance (simulated devices on CPU), SPEC D4:
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  python -m jax_single_pool.experiments.invariance_check --steps 3

# tiny single-device smoke of the real step (random weights):
python -m jax_single_pool.experiments.llama8b_real --per_gpu_batch 1 --steps 6 \
  --C 2048 --faith_warmup 0

# the real thing (HF weights, 8 GPU, C-sharded):
python -m jax_single_pool.experiments.llama8b_real --real_weights --first_layer 20 \
  --last_layer 31 --C 8192 --per_gpu_batch 1 --shard
```

## Design

- **Generic over vendored LMs.** The trainer sees only the `DecomposedLM` fn-table
  (`lm.py`): ordered `sites`, `clean_logits`, `site_inputs`, `masked_logits`,
  `weight_deltas` — all pure, all taking the frozen pytree as a *runtime arg* (a frozen
  8B target closed over as a jit constant bakes multi-GB weights into the HLO). Adding
  a target (e.g. GPT-2) = implementing that table; no TMS/ResidMLP-style generality.
- **One jit'd step, functional minimax.** The persistent adversary (per-site sources +
  their Adam moments) lives in `TrainState` and is threaded through; `n_warmup`
  supplemental ascents + one final ascent whose gradient comes from the same backward
  as the param grads (SPEC S13/S14).
- **GSPMD, not pools.** Data `P('dp')`, params placed by the target's sharding plan,
  `jax.jit` inserts every collective. The torch `reduce_source_grads` dance is absorbed
  by autodiff of the global-mean loss. Validated by `invariance_check.py`: the
  trajectory is device-count-invariant up to float reassociation.
