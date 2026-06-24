# param_decomp

A JAX implementation of the **single-pool** Parameter Decomposition (VPD) training
loop — the four-term loss (faithfulness + importance-minimality + chunkwise stochastic
recon + persistent-PGD adversarial recon) as one `jax.jit` step, GSPMD-sharded,
**generic over vendored LM targets**.

The semantics are pinned by [`SPEC.md`](SPEC.md) (normative: pseudocode + numbered
invariants, grounded in the torch oracle at git tag `torch-oracle`). It realized the
"single-pool SPMD collapse" hypothesis: XLA + whole-step `jit` + GSPMD sharding replaces
the hand-written-NCCL multi-pool design with zero manual collectives.

`param_decomp/` is the core of the root `param-decomp` distribution, living at the repo
root with sibling packages `pretrain/` (the in-house target-LM pretrainer) and
`vendored_jax/` (bit-parity JAX archs). Install the whole workspace into the one venv with
`make install-dev`.

## What's here

| file | what |
|---|---|
| `lm.py` | `DecomposedModel` — the interface a vendored LM target implements (ordered sites, flat site-keyed dicts, frozen pytree as runtime arg) + generic chunking |
| `train.py` | the step factory: one fused jit step over faith + imp-min + the recon loss TERMS, per-persistent-term fused final ascents, fp32 masters + bf16 compute |
| `losses.py` | the pure loss terms (KL/(B·T), faithfulness, imp-min lp+entropy split) + schedules (p-anneal, source-LR warmup) |
| `adversary.py` | adversarial source machinery: persistent state + Adam ascents, fresh sign-PGD init, `source_masks` |
| `recon.py` | the flat loss surface (LOSS_PARITY_DESIGN.md): the self-describing `LossTerm` union (`FaithfulnessTerm` / `ImportanceMinimalityTerm` / `ReconLossTerm`), mask-source strategies × plans × routing samplers, and `build_loss_terms` — the shared torch loss configs mapped onto a flat tuple of terms |
| `ci_fn.py` | shared-transformer CI fn over ordered site specs; the two leaky-hard squashings (SPEC §4.6, S5/S6) |
| `checkpoint.py` | orbax sharded save/resume of `TrainState` (adversary sources + moments included, no full-gather on the loop, SPEC S22) |
| `eval.py` | in-loop eval pass: the six CE/KL masking variants + per-site CI-L0 in one jitted step, logged under the torch `EvalLoop` keys (`eval/ce_kl/*`, `eval/l0/*`) — enabled by the optional `eval:` config block |
| `slow_eval.py` | LIBRARY for the in-loop slow (plot) tier (SPEC S28, in-loop only — no offline CLI): the `CIHistograms` / `ComponentActivationDensity` / `CIMeanPerComponent` reductions + renders, the config-gated `PermutedCIPlots` / `IdentityCIError` (off the `(T, C)` position CI), the `UVPlots` figure (`render_uv_figure` / `plot_uv_matrices`, shared by the LM in-loop naive-gather path and the toy `toy_uv_eval` cheap path), and the hidden-acts recon scalars. Torch-free numpy/matplotlib; logged under `slow_eval/figures/*` |
| `run_state.py` | optimizer + initial-`TrainState` construction from an `ExperimentConfig` (orbax restores onto this reference) |
| `tools/` | `convert_llama_simple_mlp_checkpoint.py` (torch venv) — one-off `.pt` → safetensors conversion of the pile pretrain checkpoint; `migrate_c49k_checkpoint.py` — one-off remap of the frozen C49k clone's orbax `TrainState` (legacy `components.{Vg..Ud}` `(1,*,*)` + flat `sources.<site>`) onto the current layout (site-keyed `components.vu`, `sources.<state_key>.<site>`) so a fine-tune can `restore_latest` it |
| `sharding.py` | generic GSPMD helpers (`init_distributed`, `dp_mesh`, `replicate`, `shard_batch`) |
| `targets/llama8b.py` | Llama-3.1-8B target: residual-start suffix + `Prefix` harvest, arbitrary per-layer matrix sites (`q/k/v/o/gate/up/down`, per-site C; q/k/v decomposed before RoPE/SDPA), per-site `DecompVU`, HF safetensors loader, `llama_decomposed_lm(cfg, sites)` |
| `targets/llama_simple_mlp.py` | `LlamaSimpleMLP` pile-pretrained target (`goodfire/spd/runs/t-9d2b8f02`: 4L, d768, GELU MLP, plain rotate-half RoPE, tied head): sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}` with `h.*` wildcard expansion, pretrain-cache safetensors loader (one-off `.pt` conversion: `tools/convert_llama_simple_mlp_checkpoint.py`), `llama_simple_mlp_decomposed_lm(cfg, sites)`; frozen weights small enough to replicate (`replicate_frozen`), V/U/CI/source placement reuses the generic per-site plan |
| `run.py` | the generic ENGINE `run_decomposition_training` (pure library, no `main`/YAML): faith warmup, loop, metrics jsonl/wandb, in-loop slow renderer, orbax checkpoints, SIGTERM-save + requeue-resume. The LM composition root that reads YAML + builds the target lives lab-side (`param_decomp_lab/experiments/lm/run.py`) |
| `data.py` | deterministic batch schedule over the pre-tokenized fineweb parquet shards; O(1) resume addressing, per-process slices |
| `hf_http.py` | `configure_hf_http_retries` — idempotent retrying-adapter install on huggingface_hub (cold-cache 8N-rank startup burst); no-op without huggingface_hub; JAX-side analog of `param_decomp_lab/infra/hf_http.py` |
| `config.py` | the trainer's internal runtime `ExperimentConfig` dataclasses (the typed config the engine + `run_state` consume): `DataConfig` / `EvalConfig` / `CadenceConfig` / optimizer structs + the `TargetSites` protocol + `CIFnArch`. Domain-agnostic — the YAML→dataclass CONVERSION lives lab-side (`experiments/config.py` shared + `experiments/lm/config.py` LM) |
| `configs.py` | the torch-free pydantic config SCHEMA: routing + decomposition-target + ci-fn + loss-metric + eval-metric configs, `PDConfig` / `RuntimeConfig` / `Cadence` / `WandbConfig` / `ResumeProvenance`, and the `wandb.config` shaping helpers (was the dissolved `param-decomp-config` distribution) |
| `base_config.py` | `BaseConfig` (frozen `extra=forbid` pydantic `BaseModel` + YAML/JSON round-trip), `Probability` |
| `schedule.py` | `ScheduleConfig` + `get_scheduled_value` (warmup → constant/linear/cosine decay) |
| `configs/` | the single self-contained run yamls (one file per run; no wrapper/schema split) |
| `targets/llama8b_sharding.py` | the 8B placement plan (frozen replicated; per-site V/U + CI + Adam C-sharded; source replicated; batch sharded) |
| `experiments/llama8b_real.py` | the runnable 8B step + tok/s/GPU bench |
| `experiments/invariance_check.py` | device-count invariance harness (SPEC D4) |
| `tests/` | tiny-target unit tests (incl. attention sites + heterogeneous per-site C), checkpoint resume, sharding, `tests/equivalence/` — the fixture-driven torch↔JAX loss-term equivalence harness — `tests/stacked_parity/` — fixtures pinning the pre-site-generality stacked implementation (clean logits bit-identical, train trajectory rel ≤ ~1e-5) — and `tests/simple_mlp_equivalence/` — torch-fixture logits parity for the LlamaSimpleMLP target (tiny random model max abs diff ~2e-7; real t-9d2b8f02 weights ~5e-5 fp32) |

## Run

```bash
# From the repo root — one venv for the whole workspace:
make install-dev && source .venv/bin/activate

pytest param_decomp/tests/

# GSPMD device-count invariance (simulated devices on CPU), SPEC D4:
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
  python -m param_decomp.experiments.invariance_check --steps 3

# tiny single-device smoke of the real step (random weights):
python -m param_decomp.experiments.llama8b_real --per_gpu_batch 1 --steps 6 \
  --C 2048 --faith_warmup 0

# the real thing (HF weights, 8 GPU, C-sharded):
python -m param_decomp.experiments.llama8b_real --real_weights --first_layer 20 \
  --last_layer 31 --C 8192 --per_gpu_batch 1 --shard
```

## Design

- **Generic over vendored LMs.** The trainer sees only the `DecomposedModel` fn-table
  (`lm.py`): ordered `sites`, `clean_output`, `read_activations`, `masked_output`,
  `masked_site_outputs` (the hidden-acts eval seam, SPEC S31), `weight_deltas` — all
  pure, all taking the frozen pytree as a *runtime arg* (a frozen
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
