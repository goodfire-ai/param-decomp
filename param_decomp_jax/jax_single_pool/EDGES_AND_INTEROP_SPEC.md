# Single-pool VPD — edges & interop spec

Pins the **meaning** of everything *around* the pure training step: checkpoint/resume,
logging, eval, model/data/config loading, the launcher, and — the load-bearing part —
the **interop contract** that lets the unchanged torch postprocessing stack consume a
JAX-trained run. `SPEC.md` is the companion document for the step math (invariants
`S_/N_/R_/D_`); this one numbers edge behaviors `E1, E2, …` and interop clauses
`I1, I2, …`.

**Ground truth.** The stable torch impl in `goodfire-ai/param-decomp`. The single-pool
core lives under `param_decomp/` + `param_decomp_lab/` on this branch (`feature/jax`);
the n-pool lineage (`three_pool/`, vendored `LMComponentModel`) that some torch
pointers reference is NOT on this branch and is flagged where it matters.

**How to read.** Same conventions as `SPEC.md`: normative content is the tables and the
numbered invariants; prose is orientation. Bit-exactness with torch is NOT required —
identical state set / format contract / cadence / detachment is. Every clause is
grounded in `file:line`. Each interop clause is tagged **CONFIRMED** (validated by the
empirical fixture verification against the real torch modules) or **ASSUMED** (derived
from source, not exercised end-to-end).

Notation: `sg[x]` stop-gradient; `<run_dir>` = `PARAM_DECOMP_OUT_DIR/runs/<run_id>/`;
`p-<8hex>` the canonical run id.

---

## 1. Glossary — terms that bite

| term | means | NOT to be confused with |
|---|---|---|
| **trajectory state** | the mutable state that defines where optimization is: V/U masters, CI-fn masters, both AdamW states, every persistent adversary's sources + Adam moments + step_count, and the global step counter | the frozen target (rebuilt from HF, never persisted) |
| **wrapper yaml** | the JAX-only launch config: `{torch_config, run_name, out_dir, remat_recon_forwards}` (+ `run_id` once stamped) | the torch `LMExperimentConfig` it points at |
| **pinned configs** | the two yamls a run dir copies on first write — `config.yaml` (wrapper) + `experiment_config.yaml` (torch config) — byte-compared on resume | the launch-relative paths in the wrapper, which are ignored on reload |
| **fast metric** | a scalar eval metric the JAX trainer runs in-loop (CE/KL, CI L0, fresh-PGD probe) | a slow/plot metric (histograms, hidden-acts, autointerp), offline-only in JAX |
| **export** | the safetensors a JAX run emits so torch neighbors can load it (`<run_dir>/export/model_<step>.safetensors`) | the orbax checkpoint (`<run_dir>/ckpts/`), which is JAX-internal and not torch-readable |
| **site order** | the order sites are concatenated/split in the CI fn. JAX uses computation order (`KIND_ORDER`); torch uses lexicographic `sorted()` — they differ, and the export permutes between them | the per-site computation order inside one forward |

---

## 2. Checkpoint / resume (E1–E8)

The contract: a run stops (user, SLURM preemption→requeue) and continues without
perturbing the trajectory. Schedules carry no own state — they are pure functions of the
restored step counter, so round-tripping the integer step resumes every schedule (LR
cosine, imp-min p-anneal, source-LR warmup).

| ID | invariant | grounding |
|---|---|---|
| E1 | The persisted unit is exactly the §3/`S22` trajectory state: V/U + CI masters, both AdamW states, each persistent term's sources + Adam moments + `step_count`, and the global step. The frozen target is excluded and rebuilt from HF on resume. | `train.py:65-74`; `checkpoint.py:6-9`; SPEC `S22` |
| E2 | The checkpoint format is an orbax **sharded** save under `<run_dir>/ckpts/`: every process writes its own shards, no on-loop full-gather; restore maps stored shards onto a freshly-built reference `TrainState`'s shardings. | `checkpoint.py:30-50` |
| E3 | Saves are **synchronous** (`enable_async_checkpointing=False` + `wait_until_finished()`): a SIGTERM-triggered save must be on disk before exit for requeue. | `checkpoint.py:33-43` |
| E4 | Resume is **in-place**: re-entering the same workspace `restore_latest`-loads the newest checkpoint, sets `start_step` to it, and the loop continues the same `run_id` / wandb run. There is no separate resume command. | `run.py:189-194` |
| E5 | The data position at step `S` is reproduced O(1): the batch schedule is a pure function of `(seed, step)` (`server.local_batch(step)`), not a replayed loader. | `run.py:244-248`, `291`; SPEC `S18`, `R1` |
| E6 | Resume with a changed config is **refused**: the run dir byte-compares the incoming `config.yaml` and `experiment_config.yaml` against the pinned copies and asserts equality (whitespace-sensitive, intentionally sharp). First run copies them in. | `run.py:399-407`, `421-427` |
| E7 | SLURM lifecycle: `--requeue` + `--signal=TERM@300`. A module-global SIGTERM flag is checked once per training-loop iteration; on save-cadence OR final step OR sigterm the loop saves then breaks for requeue. | `run.py:350-360`; `jax_launch.py:81-91` |
| E8 | SIGTERM-save is **best-effort, not the guarantee.** The flag is only serviced at the main-loop step boundary — a preemption during the first jit compile, the faith-warmup loop (`run.py` warmup has no sigterm check), or an eval pass is missed; the periodic `save_every` is the actual safety net. (Observed: 6/6 preemptions fell back to periodic ckpts.) | `run.py:206`, `350`; lore `jsp_sigterm_save_never_fired` |

**Step convention (non-normative caveat).** JAX saves at `now_step = step+1` and resumes
at `range(start_step, …)` re-executing `start_step`; torch saves at `self.step==S` and
re-executes `S`. Both never re-apply a gradient twice, but a torch-vs-JAX checkpoint "at
the same step" is offset by one — align the axes before cross-impl trajectory diffs.
(`run.py:291`, `350`.)

**Persistent-source `step_count` is a fp32 scalar array** (`+1.0` each ascent),
round-tripped as a pytree leaf; torch's is a Python int. Bias-correction math agrees; the
first post-resume ascent must use count `N+1`. (`adversary.py:103`; SPEC `S15`.)

---

## 3. Logging / run-sink / wandb (E9–E14)

The sink is the trainer's side-effect boundary: rank-0-only fan-out to `metrics.jsonl` +
wandb + console; non-main ranks get a silent no-op. SPEC.md is silent on logging; this
section is the authoritative key-namespace reference.

| ID | invariant | grounding |
|---|---|---|
| E9 | Logging is rank-aware: non-main ranks no-op (`_jsonl=_wandb=None`, early return). Only rank 0 touches disk/wandb/console. | `run.py:124-127`, `148-149` |
| E10 | Each `log()` appends one JSON line `{step, **scalars}` to `<run_dir>/metrics.jsonl` (handle held open, flushed per write). The sink is **scalar-only** — no figure/chart side-channel (the JAX metric set emits no figures). | `run.py:127`, `147-161` |
| E11 | The logged-key namespace MUST match torch for overlay (stated intent `run.py:102-105`): `train/loss/total`, `train/loss/<TermName>`, `train/loss/FaithfulnessLoss`, `train/loss/ImportanceMinimalityLoss(+_no_beta)`, `train/grad_norms/summary/{components,ci_fns,total}`, `train/schedules/lr/{components,ci_fn}`, `eval/<key>`, `slow_eval/<key>` (+ `slow_eval/step`). | `run.py:106-116`, `150-156`; `train.py:473-485` |
| E12 | Logged scalars are **global-batch** quantities. Losses are computed over the global sharded batch (kl divides by global `B·T`, imp-min sums over the global batch per `S8`), so `float(v)` on proc 0 reads an already-global value — no explicit cross-rank average needed. | `run.py:304-308`; `train.py:415-418`; SPEC `S8`, `D2` |
| E13 | The `slow_eval/*` keys ride a dedicated `slow_eval/step` axis (`wandb.define_metric`), defined up front so the async offline-eval writer can log retroactively (the live run's `_step` has advanced). The live JAX trainer emits `eval/*` only; ALL slow metrics are deferred to `pd-offline-eval`. | `run.py:139-144`; SPEC `S22` |
| E14 | JAX-only additive keys (absent on torch runs, harmless for overlay): `train/perf/{step_time_s,tok_per_s,tok_per_s_per_gpu}`, `train/mem/peak_gb_per_rank`, `train/schedules/{p_imp,lr/src}`, and `jax_runtime` config sub-dict (actual `n_devices`/`n_processes`/topology). | `run.py:309-316`, `278-284` |

**Known deviations from torch (non-normative).** Per-leaf grad-norm keys differ
(equinox keystr vs torch dotted path) — only `summary/*` overlays. Config dump is the
raw nested torch dict, not torch's flattened-typed-lists shape. No `log_code` (the
snapshot workspace supersedes it). **Two robustness gaps worth closing**: (1) no
CommError guard around `wandb.log` — a transient wandb outage crashes a JAX run that
torch survives; (2) no `isfinite` assert on the logged loss — a NaN logs silently where
torch crashes at source (`train.py:417`; `run.py:308`).

---

## 4. Eval (E15–E20)

Eval is split across two processes in JAX: a **fast** scalar tier in-loop, and a
**slow/plot** tier delegated entirely to `pd-offline-eval` on exported checkpoints. SPEC
is silent on eval; this is the reference.

| ID | invariant | grounding |
|---|---|---|
| E15 | The in-loop eval runs the FAST tier only, on cadence `eval.every` (must divide `log_every`; the tok/s window resets after eval). JAX `EvalConfig` carries `every`/`n_steps`/`batch_size` and **no** `slow_every`/`slow_on_first_step` — the slow tier is offline, keyed off the checkpoint save cadence. | `run.py:254-258`, `320-348`; `config.py:90-104` |
| E16 | FAST metrics implemented in-loop: `CEandKLLosses` (six masking variants), `CI_L0` (per-site + grouped sums), and an optional fresh sign-PGD `PGDReconLoss` probe. The torch fast attn-pattern recon metrics are not ported (absent from both production eval blocks). | `eval.py:43-218` |
| E17 | `CEandKLLosses`: six all-sites no-routing forwards — `ci_masked`/`unmasked`/`stoch_masked`/`random_masked`/`rounded_masked`/`zero_masked`. Only `stoch_masked` carries a weight-delta mask. Emits `kl_<v>`, `ce_difference_<v>`, `ce_unrecovered_<v>`; softmaxes/KL in fp32. KL direction P=clean, Q=masked. | `eval.py:110-166`, `31-40`; SPEC `S1`, `N3` |
| E18 | `CI_L0`: per site, mean over batch·seq of `count(ci_lower > threshold)`; groups (fnmatch patterns) **sum** member-site L0s under `l0/<thresh>_<group>`; key format matches torch byte-for-byte. | `eval.py:63-71`, `167-177` |
| E19 | The eval `PGDReconLoss` probe is the eval-only adversary (NOT the persistent training adversary): per site init `(1,1,C+1)` (c-scope), `n_steps` sign ascents on global-mean-KL source-grad, clamp `[0,1]` each step; KL at the final source. JAX implements c-scope only (both production yamls use it). | `eval.py:179-215`; `config.py:81-104`; SPEC `S1`, `S15` |
| E20 | SLOW/plot metrics (`CIHistograms`, `ComponentActivationDensity`, `CIMeanPerComponent`, hidden-acts recon, `IdentityCIError`, `PermutedCIPlots`, `UVPlots`, `AutointerpLabels`) are **not** computed in-loop; `pd-offline-eval` rebuilds the torch `LMComponentModel`, strict-loads the export, and runs them through the shared torch `run_eval_pass`, logging `slow_eval/*` retroactively. Frozen target forced to bf16 to match the JAX run (`N1`/`N2`). | `eval.py:1-14`; `offline_eval.py:60-149` |

**Eval averaging caveat (non-normative).** JAX averages eval metrics `sum/n_steps`
(uniform-batch assumption); torch metrics accumulate position-weighted. Equal under fixed
`(B,T)` (which holds). For any non-mean eval metric (log2-of-global-sum L0/imp), torch's
accumulate-then-compute differs from per-batch-mean-then-average (Jensen, same class as
`S8`) — verify per metric before relying on offline/in-loop agreement. (`run.py:340`.)

**SimpleMLP eval gap.** Export/offline-eval are llama8b-only (`run.py:354`); a
`LlamaSimpleMLP` run's configured slow metrics are never computed.

---

## 5. Model / data / config loading (E21–E28)

| ID | invariant | grounding |
|---|---|---|
| E21 | Both impls validate the SAME torch-free `param_decomp_config` pydantic schema. The JAX trainer parses the wrapper, validates its torch yaml as `LMExperimentConfig`, and `convert_torch_lm_config` accepts a defined subspace, asserting loudly on the rest — no silent approximation. | `torch_config.py:304-327`, `319` |
| E22 | The wrapper yaml keys are exactly `{torch_config, run_id, run_name, out_dir, remat_recon_forwards}` (`WRAPPER_KEYS`); `run_id` is the canonical `p-<8hex>`. | `torch_config.py:300-301`, `313-315` |
| E23 | Target dispatch is a discriminated union: `hf` / `hf_weights_in_vendored` → `TargetConfig` (vendored or HF Llama-3.1-8B); `pretrained` → `LlamaSimpleMLPTargetConfig`. Any other model family (GPT-2, other HF) is refused at convert time. | `torch_config.py:96`; `config.py:20`; `run.py:439` |
| E24 | The frozen target is stored **bf16**; V/U + CI-fn masters are fp32; AdamW + source moments are fp32 (`N1`). A torch config requesting an fp32 frozen target is **downgraded to bf16 with a printed warning, not refused** — such a run is not bit-reproducible in JAX (≈5e-4 nats KL), a spec-tolerated divergence. | `llama8b.py:39`; `torch_config.py:237-245`; SPEC `N1`, `N2` |
| E25 | Faithfulness deltas `W−V@U` for the loss are formed in fp32 (`weight_deltas_fp32`). The masked-forward delta PATH is bf16-computed — a documented bf16-rounding divergence, faithfulness untouched. | `llama8b.py:434-444`, `301-305`; SPEC `N2` |
| E26 | Data is a pre-tokenized **parquet** contract, validated at convert: `is_tokenized and not streaming`, `dataset_name=="parquet"`, `column_name=="input_ids"`, a `*.parquet` glob. The JAX trainer NEVER streams/tokenizes from HF at run time. | `torch_config.py:173-181` |
| E27 | The batch schedule is a pure seeded function: shard-order permutation, in-shard row permutation, consecutive row-windows; shard tails (`rows % global_batch`) dropped; rows of width `seq_len` or `seq_len+1` accepted (the `+1` label column truncated). | `data.py:53`, `114` |
| E28 | Supported PD subspace is asserted, not approximated: `sigmoid_type=='leaky_hard'`, `use_delta_component` True, `tied_weights` None, `identity_decomposition_targets` None, persistent scope `sc` only, AdamW `wd=0`/betas `(0.9,0.999)`, cosine-to-`0.1×` no-warmup LR, components grad-clip set / ci-fn unclipped, faith-warmup `wd==0`. | `torch_config.py:85`, `148`, `232`, `247`; SPEC `S19`, `S20`, `S21` |

**Site-order parity (open).** torch resolves sites by fnmatch over `named_modules()`
(definition order); JAX canonicalizes `(layer-asc, KIND_ORDER)`. They coincide on the
production configs but the orderings are not provably equal for arbitrary multi-matrix
configs — and site order is RNG- and concat-load-bearing (`S10`). See I7.

---

## 6. Launcher / SLURM (E29–E33)

The JAX launch path is deliberately NOT polymorphic with torch's `torchrun`/elastic
model. `pd-jax-lm <wrapper.yaml> --nodes N` (lab-side, torch venv).

| ID | invariant | grounding |
|---|---|---|
| E29 | `pd-jax-lm` mints `p-<8hex>`, snapshots the working tree to `refs/runs/snapshot/<id>`, and materializes an **immutable shared-FS workspace** at `$PARAM_DECOMP_OUT_DIR/workspaces/<id>` (git clone + BOTH venvs built at submit time, since jsp-train runs 8 srun tasks/node and in-job per-node cloning would race), stamping `run_id` into the workspace wrapper. | `jax_launch.py:45-70`, `146-172` |
| E30 | One srun task per GPU: `srun --kill-on-bad-exit=1 --ntasks-per-node=8 --distribution=block:block`, each rank activating the cuda venv and `exec jsp-train`. There is no torchrun/elastic; `jax.distributed.initialize(local_device_ids=[SLURM_LOCALID])` brings up GSPMD over the SLURM topology. | `jax_launch.py:32`, `94-96`; `sharding.py:23`, `38` |
| E31 | The sbatch omits `--partition` (org policy: cluster default), passes `--signal=TERM@300` (no `B:` prefix — delivers to srun ranks, where the SIGTERM handler runs), `--requeue`, and a `--comment`. | `jax_launch.py:81-92`; `slurm.py:58-69` |
| E32 | Requeue re-enters the immutable workspace, never the live checkout; `--run_id` resubmits an existing workspace. Identity is the `run_id` (run-dir name, wandb id, ckpt dir). | `jax_launch.py:74`, `169`; `run.py:131` |
| E33 | On each non-zero checkpoint save (main rank, llama8b target only), the trainer fire-and-forgets `sbatch offline_eval_once.sbatch` with `--dependency=singleton` (per-run serialized) + marker-file dedup. A failed submission MUST NOT kill training (the one place graceful beats fail-fast). | `run.py:350-394` |

**Operational (non-normative).** XLA env: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92` (default
0.75 OOMs at production step), `--xla_gpu_enable_command_buffer=` empty (disables CUDA-graph
capture; ~0% throughput cost; avoids intermittent `STREAM_CAPTURE_INVALIDATED`).
Cold-cache HF weight load on every rank at startup has no retry adapter (torch's
`hf_http` guard has no JAX analog) — pre-warm the snapshot. (`jax_launch.py:40-42`,
`95`; lore `pd_jax_lm_launcher`.)

---

## 7. INTEROP CONTRACT (I1–I12)

What a JAX run MUST emit so the **unchanged** torch postprocessing stack (harvest,
autointerp, dataset_attributions, graph_interp, clustering, app, offline-eval) can
consume it. The torch consumer path is `SavedLMRun.from_path(run_id)` → reads
`experiment_config.yaml` + the latest checkpoint → `load_component_model` rebuilds the
torch `ComponentModel` and strict-loads the state dict.

Each clause is **CONFIRMED** (the empirical verification built the real torch
`LMComponentModel` / `LinearComponents` / `GlobalSharedTransformerCiFn` on this branch
and round-tripped the committed export fixtures) or **ASSUMED** (source-derived, not
exercised end-to-end). The verification used the Llama-8B-layout fixtures
(`tools/export_fixtures/`); the SimpleMLP key naming has NO fixture coverage and NO
export support.

### 7.1 Run-dir layout & config

| ID | clause | status | grounding |
|---|---|---|---|
| I1 | A run dir pins TWO yamls side by side: `config.yaml` (the wrapper) and `experiment_config.yaml` (the referenced torch `LMExperimentConfig`, the torch `SavedLMRun` contract name). The launch-relative path in the wrapper is ignored on reload — the pinned copy is the source of truth. | CONFIRMED | `torch_config.py:330-347`; `run.py:421-427` |
| I2 | The exported weights live at `<run_dir>/export/model_<step>.safetensors`. **This does NOT match the torch discovery path**: `resolve_run_files` globs `*.pth` directly in the run-dir root and `torch.load`s it; neither finds an `export/*.safetensors`. A consumer must (a) load via `safetensors.torch.load_file` and (b) point at `export/`, OR the contract must change. | CONFIRMED (gap) | `export.py:181-185`; `run_files.py:270-274` |
| I3 | `experiment_config.yaml` must validate as `LMExperimentConfig` AND `build_target` must succeed (torch rebuilds the frozen target from `cfg.target.spec`, not from the checkpoint). Verified for the Llama-8B HF/vendored spec; the `pretrained` + `h.*`-wildcard SimpleMLP round-trip is untested. | ASSUMED | `offline_eval.py:60-118`; `torch_config.py:330-347` |

### 7.2 Safetensors key / shape / orientation

The export targets the **vendored `LMComponentModel`** state-dict layout (`model.*` frozen
target inlined, `model.<site>.components.{V,U}`, `model.<site>.target_weight`,
`ci_fn._global_ci_fn.*`). The verifier confirmed this layout loads `strict=True` into the
real torch `LMComponentModel.build`.

| ID | clause | status | grounding |
|---|---|---|---|
| I4 | Exported keys EXACTLY partition the torch `LMComponentModel.state_dict()`: the exported set == the trainable subset (`model.<site>.components.{V,U}` + `ci_fn._global_ci_fn.*`); the frozen subset (`model.*` target weights, `model.<site>.target_weight` for decomposed sites) completes the strict load. All shapes match; `load_state_dict(strict=True)` succeeds with no missing/unexpected/mismatched keys. | CONFIRMED | `export.py:85-149`; verify result `contract_confirmations[0-1]` |
| I5 | V/U orientation is `V (d_in,C)`, `U (C,d_out)` (same both sides); the effective weight is `(V@U).T` to match torch `[d_out,d_in]` storage. `LinearComponents` `((x@V)*mask)@U` round-trips to the JAX component output within `max_rel ≤ 1.1e-4`. | CONFIRMED | `export.py:85-92`; `components.py:71-72`; verify `contract_confirmations[2]` |
| I6 | CI-fn key mapping: `in_proj_{w,b}→_input_projector.{W,b}`, `out_{w,b}→_output_head.{W,b}` (both `(d_in,d_out)`, `x@W+b`, NO transpose); `blocks[i].w{q,k,v,o}→_blocks.{i}.attn.{q,k,v,out}_proj.weight` (`(d_out,d_in)`, `x@W.T`); `blocks[i].{w1,b1,w2,b2}→_blocks.{i}.mlp.{0,2}.{W,b}`; `inv_freq→_blocks.{i}.attn.rope.inv_freq` (persistent buffer). All fp32. | CONFIRMED | `export.py:95-121`; verify `contract_confirmations[4]` |
| I7 | **The site-order permutation — the one real trap.** torch concatenates CI inputs and splits CI outputs in `sorted()` (lexicographic) module-path order (`GlobalSharedTransformerCiFn.layer_order`); JAX uses computation order (`KIND_ORDER` q,k,v,o,gate,up,down). The export MUST reorder the `in_proj` ROW blocks (by `d_in`) and the out-head COLUMN blocks + bias (by `C`) from JAX order to `sorted(site_names)`. `in_proj_b` and per-block weights are order-invariant. Verified against the real torch module: full CI-fn forward (lower/upper leaky-hard) round-trips within the verifier's `rtol 2e-4 / atol 1e-6` gate. | CONFIRMED | `export.py:66-82`, `95-121`; verify `contract_confirmations[3]`; SPEC `S10`, `S11` |
| I8 | Tensors are fp32 (`_f32`): V/U are already fp32 masters (lossless); the frozen bf16 target upcasts exactly, matching torch's fp32 HF load path. | CONFIRMED | `export.py:62-63`, `144-149`; SPEC `N1` |

### 7.3 Deliberate omissions

| ID | clause | status | grounding |
|---|---|---|---|
| I9 | The export carries V/U + CI-fn params + frozen target ONLY. Adversary persistent sources and optimizer/Adam state are training-only and intentionally excluded — no torch consumer (harvest pre-weight-acts, `calc_causal_importances`, app attribution, dataset_attributions) needs them; the app reconstructs eval PGD fresh. | CONFIRMED | `export.py:10`; SPEC `S13`, `S14`, `S15` |

### 7.4 Known interop gaps (must be resolved before relying on the contract)

| ID | clause | status | grounding |
|---|---|---|---|
| I10 | **SimpleMLP export is unimplemented.** `jsp-export` asserts `isinstance(cfg.target, TargetConfig)` (llama8b only) and aborts on `LlamaSimpleMLPTargetConfig`. EVERY on-disk JAX run with orbax checkpoints targets `LlamaSimpleMLP`, so no finished run can currently be exported. A SimpleMLP branch (dispatch + `llama_simple_mlp` lm/frozen loader + a `h.N.mlp.{c_fc,down_proj}` / `attn.{q,k,v,o}_proj` fixture) is required. | CONFIRMED (blocker) | `export.py:160-162`; verify `notes`, `run_used` |
| I11 | **Layout vs the wired loader.** The export emits the VENDORED `LMComponentModel` layout, but the loader `SavedLMRun.load_model` invokes on this branch is `load_component_model` (the CORE `ComponentModel`, keys `target_model.*` + `_components.<dashed-path>.{V,U}`). The vendored-layout loader exists but is unreachable from the wired path. The canonical export target for `feature/jax` must be decided: emit the core layout (preferred — it's what the branch builds, and `build_target` already supplies the frozen target, so inlining the 8B target is redundant), or wire `SavedLMRun` to dispatch to the vendored loader. | CONFIRMED (gap) | `export.py:5-31`; `component_model.py:164-171`; `component_model_io.py:117-127` |
| I12 | **The parity verifier is rotted.** `tools/verify_export_torch.py` imports `param_decomp_lab.three_pool.checkpoint.is_trainable_component_key` (no such module/symbol on this branch) and asserts the vendored layout — it cannot run, and the interop contract has NO working end-to-end parity test in CI. The empirical check reproduced its assertions inline; the contract must be re-pinned by a runnable verifier against whichever layout I11 settles on. | CONFIRMED (gap) | `verify_export_torch.py:35`; verify `notes` |

**CI-fn numerics: UNIFIED with torch (#624/#625/#730 resolved "match torch").** JAX CIFn
uses exact-erf GELU (`jax.nn.gelu(..., approximate=False)`, matching torch `nn.GELU()`)
and weightless-RMS eps `finfo(fp32).eps` ≈1.19e-7 (`CI_FN_RMS_EPS`, matching torch
`F.rms_norm`'s default; RMS upcasts to fp32 so fp32 finfo governs). The former CI-logit
divergence near the leaky-hard clamp boundary is resolved — these ops are now bit-faithful
torch→JAX, so the only remaining CI-fn-transfer blocker is the site-order permutation (I7).
(`ci_fn.py` GELU line + `CI_FN_RMS_EPS`; torch `ci_nn_blocks.py:167,174`; SPEC `S4`/`S6`.)

---

## 8. Non-normative: pointers & rationale

- Checkpoint/resume: `checkpoint.py`, `run.py:188-194` / `350-407`, `train.py:65-74`.
- Logging/sink: `run.py:124-163`, `276-348`; torch `param_decomp_lab/run_sink.py`,
  `infra/wandb.py`.
- Eval: `eval.py`, `run.py:320-348`; offline bridge `offline_eval.py`,
  `slurm/offline_eval_once.sbatch`.
- Config/data/target: `torch_config.py`, `config.py`, `data.py`, `llama8b.py`,
  `llama_simple_mlp.py`.
- Launcher: `param_decomp_lab/experiments/lm/jax_launch.py`, `infra/slurm.py`,
  `infra/git.py`.
- Interop: `export.py`, `tools/verify_export_torch.py`, `tools/export_fixtures/`;
  torch consumers `component_model_io.py`, `adapters/pd.py`, `experiments/lm/run.py`.
