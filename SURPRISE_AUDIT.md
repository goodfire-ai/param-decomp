# Surprise audit — code in a state a reader would not guess

Adversarially-verified output of the `surprise-audit` workflow (2026-06-26): **38 confirmed** of 43 candidates (5 dismissed as justified). Each finding survived a skeptic agent that tried to dismiss it. Severity 1 (cosmetic) .. 5 (real waste/error).

## Severity 4  (5)

### `param_decomp/targets/llama8b.py:191-211`  — _claim-vs-code_
- **Assumed:** A reader of the 9-line comment block (lines 191-206) concludes the target's attention is TENSOR-PARALLEL over heads: it explicitly says 'pin all three head-parallel: batch on `dp`, HEADS on `tp` (`P("dp","tp",None,None)`)', 'clean tensor-parallel attention', and 'the 2-D head-parallel form is the fix'.
- **Actual:** The actual code at lines 208-210 pins q/k/v to `PartitionSpec("dp", None, None, None)` — BATCH-parallel with heads REPLICATED, exactly the opposite of the comment block above it. The one-line inline comment on line 210 ('batch-parallel') and `FrozenAttn.shardings` (lines 159-165, 'the HEAD dim stays REPLICATED (NOT TP'd)') both agree with the code, leaving the long block as the lone, contradicting voice.
- **Fix:** Delete/replace the stale block (191-206) with the truth it currently contradicts: q/k/v are pinned `P("dp",None,None,None)` — batch-parallel, heads REPLICATED (not TP'd), because cuDNN flash attn requires q/k/v identically sharded and TP'ing heads would give q (n_head) and k/v (n_kv_head) different per-rank head counts.

### `param_decomp_lab/experiments/lm/config.py:103-108`  — _heavy-default_
- **Assumed:** `weights_dtype: Literal["float32", "bfloat16"] = "float32"` — a reader sees float32 as the safe/conservative default; the docstring frames bfloat16 as the opt-in optimization ("halves the footprint ... at some numerical risk"), implying float32 is the normal path you get by omitting the field.
- **Actual:** float32 is the ALWAYS-REFUSED path. Both LM targets declare `supported_weights_dtypes = frozenset({"bfloat16"})` (lines 150, 166), and `assert_supported_weights_dtype` (line 400, called from `build_from_schema` line 435) hard-asserts `weights_dtype in supported`. So omitting the field (defaulting to float32) crashes every LM run at convert time with 'No silent downgrade'. The only value that works is the non-default `bfloat16`.
- **Fix:** Make the unsupported value unrepresentable: drop the float32 literal and the default — `weights_dtype: Literal["bfloat16"]` (required, no default) — so the schema forces the only value any target accepts, and delete the now-misleading "opt-in optimization" framing from the docstring.

### `param_decomp/configs.py:813-819 (default) vs param_decomp/run.py:431`  — _heavy-default_
- **Assumed:** `keep_last_n_checkpoints: PositiveInt | None = None` with a docstring stating: 'None (the default) keeps all checkpoints — the conservative choice for research where prior steps may matter.' A reader concludes that omitting the field is supported and keeps every checkpoint.
- **Actual:** The trainer refuses None. `run.py:431` asserts `cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None`. A config that omits `keep_last_n_checkpoints` (relying on the documented 'keep all' default) crashes at training start with an AssertionError. The documented 'keep all checkpoints' behavior is unreachable through the default.
- **Fix:** Drop the `= None` default and the 'keeps all checkpoints' docstring; make it required (`keep_last_n_checkpoints: PositiveInt`) so omission fails at schema validation with a clear message instead of the docstring promising an unreachable behavior.

### `param_decomp/configs/llama8b_full32L_seq512_b128_dp128.yaml:14-15, 593-594, 26 (run_name)`  — _temp-riding-prod_
- **Assumed:** A reader sees a clean canonical name (jax-full32L-allmat-seq512-b128-dp128) and a confident header ("the botec puts per-rank-1 at ~105 GiB, safe under the 180 GiB B200 cap") and assumes this is the viable de-risking launch config for the full-32L decomposition — a per-DP-1 lightweight-activation topology.
- **Actual:** The config sets runtime.tp: 8 with dp: 128, so dp_axis = dp/tp = 16 and per-DP batch = 128/16 = 8, NOT per-rank-1. The sibling config llama8b_full32L_seq512_b32_dp128_tp4.yaml explicitly records that THIS EXACT topology already OOM'd: "b128/tp8 OOM'd (~72 GiB alloc) because tp=8 -> dp_axis=16 -> per-DP=8, ~8x these activations." The b32/tp4 file is the actual high-water-mark fix (a7b31ae37 "b32/dp128/tp4 high-water-mark (per-DP=1)"); the b128 file is the superseded sweep attempt left behind with its pre-fix header reasoning intact.
- **Fix:** In the b128 header replace the "per-rank 1 / ~105 GiB safe" claim with the truth — "tp:8 -> dp_axis=16 -> per-DP=8; this topology OOM'd (~72 GiB), superseded by llama8b_full32L_seq512_b32_dp128_tp4.yaml (per-DP=1)" — or delete this superseded config.

### `param_decomp/targets/llama8b.py:444-482 (read_activations) vs 429-442 (clean_output) & 558-565 (masked forward)`  — _wrong-granularity_
- **Assumed:** A reader who has internalized this file's own design — `clean_output` and `_run_masked_forward` both use `jax.lax.scan` over the block stack, each carrying a comment that the scan is "the compile fix for the full model" (so XLA compiles ONE block body, not 32) — would assume `read_activations`, which feeds the CI fn EVERY training step, scans the layers the same way. Its docstring even brags "Stops once the last requested key's block is fully covered (no wasted block compute past it)", reading as if it is careful about per-layer cost.
- **Actual:** `read_activations` is an unrolled Python `for layer in range(self.n_layer)` loop that slices `self.stacked` per layer (`jax.tree.map(lambda a: a[li], ...)`). With the full32L config every layer is decomposed and the chunkwise CI fn requests `resid.{first_block_of_chunk}` taps (config.py:282), so `last` = 31 and the loop materializes all 32 distinct attention+MLP layer bodies — defeating exactly the scan-based compile fix the sibling methods document. The 'stops at last' optimization only helps when no high layer is tapped; at full depth it stops at layer 31, i.e. never early.
- **Fix:** Rewrite read_activations as a lax.scan over self.stacked (like _run_masked_forward), stacking all candidate taps and indexing out the wanted subset afterward, OR add a comment at line 461 stating WHY it must unroll (and drop the misleading "no wasted block compute" docstring line, since at full depth the break never fires early).

## Severity 3  (19)

### `param_decomp/hidden_acts_eval.py:1-3`  — _claim-vs-code_
- **Assumed:** Module docstring: 'JAX-native hidden-activation reconstruction eval metrics ... the OFFLINE counterparts of the torch eval metrics of the same names (`param_decomp/metrics/stochastic_hidden_acts_recon.py`).' A reader assumes these are offline/retrospective metrics and that the cited torch file exists as reference.
- **Actual:** These metrics run IN-LOOP in the live eval pass (`compute_hidden_acts_metrics` called at experiments/lm/run.py:300 inside `eval_fn`, on the slow tier), not offline — the offline/export bridge was retired (per CLAUDE.md). And `param_decomp/metrics/` does not exist (verified: no such directory), so the cited reference path is dead.
- **Fix:** Reword line 2 to "the JAX eval-metric forms of the retired torch metrics of the same names" and drop the dead `param_decomp/metrics/...` path (the torch oracle lives at git tag torch-oracle); optionally note they run in-loop on eval.slow_every.

### `param_decomp/lm.py:143`  — _claim-vs-code_
- **Assumed:** `masked_site_outputs` docstring: 'For the OFFLINE hidden-acts recon eval metrics only — never the recon grid'. A reader assumes this seam exists to serve an offline/export path.
- **Actual:** The seam serves the IN-LOOP slow-eval hidden-acts metrics (hidden_acts_eval.py → experiments/lm/run.py:300, run every eval.slow_every during training). There is no offline path — the torch offline-eval bridge was retired. The same stale 'offline' word recurs in the llama8b impl (`masked_site_outputs`, line 601-602, which is more accurate — 'SPEC S31').
- **Fix:** lm.py:142-143: drop "offline" — e.g. "For the in-loop hidden-acts / attn-pattern eval metrics only (SPEC S31, run on eval.slow_every) — never the recon grid, which stays KL-on-final-logits." (matches the llama8b impl wording and the actual in-loop call site).

### `param_decomp/sharding.py:1-13`  — _claim-vs-code_
- **Assumed:** Module docstring: 'data is sharded `P('dp')` over a 1-D device mesh ... see `jax_spike/SYNTHESIS.md`'. A reader assumes the trainer runs on a one-dimensional `dp` mesh and can consult the cited spike doc.
- **Actual:** The mesh is 2-D: `dp_mesh(tp)` builds `Mesh(devices.reshape(n//tp, tp), axis_names=('dp','tp'))` (lines 56-65), and the 2-D `(dp, tp)` mesh is used throughout (CIBlock/ChunkTransformer/FrozenAttn shardings reference `tp`). `jax_spike/` does not exist (verified). The docstring predates the tp axis being added.
- **Fix:** Rewrite the module docstring to describe the 2-D `(dp, tp)` mesh (dp = data/chunk parallel, tp = Megatron tensor-parallel; tp=1 degenerates to pure dp) and drop the dead `jax_spike/SYNTHESIS.md` reference.

### `param_decomp_lab/experiments/lm/config.py:103-108`  — _plumbed-but-ignored_
- **Assumed:** `weights_dtype` (default "float32") chooses the FROZEN target's weight precision; the docstring describes a real float32-vs-bf16 numerical tradeoff ("bfloat16 halves the target's resident footprint ... only changes residual/norm accumulation precision (measured ~5e-4 nats KL)") as if both are live, runnable code paths.
- **Actual:** The knob is a no-op. The only consumer is `assert_supported_weights_dtype` (line 400), a validator. Both real targets declare `supported_weights_dtypes = frozenset({"bfloat16"})` (lines 150, 166), and the loaders hardcode bf16 (`DT = jnp.bfloat16`; the line-167 comment says "jnp.bfloat16 hardcoded at the call site"). The value never reaches a loader. Worse, the DEFAULT "float32" is rejected by every real target, so the documented default cannot run.
- **Fix:** Change the default to the only runnable value and stop documenting a non-existent capability: `weights_dtype: Literal["bfloat16"] = "bfloat16"` (drop the float32 member and the tradeoff prose), since both loaders hardcode bf16 and no target supports float32.

### `param_decomp_lab/experiments/lm/config.py:98-102`  — _plumbed-but-ignored_
- **Assumed:** Setting `activation_checkpointing: true` enables per-block gradient checkpointing on the frozen target forward — the docstring calls it "the main lever for raising b_per_rank on deep targets" (trades ~33% compute for ~10-15x less activation memory under 3-pool).
- **Actual:** The field is never read anywhere in the JAX codebase (the only hit is its own definition). `enable_activation_checkpointing()` is never called. It references the retired 3-pool torch trainer; the JAX engine gates recon-forward remat via `runtime.remat_recon_forwards` instead. Toggling this field does nothing.
- **Fix:** Delete the dead `activation_checkpointing` field and its docstring (config.py:98-102); the real lever is `runtime.remat_recon_forwards` / `runtime.remat_ci_fn` in `param_decomp.configs.RuntimeConfig`.

### `param_decomp/configs.py:278 (def); 263 (docstring)`  — _plumbed-but-ignored_
- **Assumed:** `ChunkwiseSubsetReconLoss.use_fused_kl` (default True) selects between a fused-linear-KL and a non-fused KL computation. The class docstring reinforces this: "the recon is the fused-linear-KL against the clean logits (when `use_fused_kl`)."
- **Actual:** `use_fused_kl` is never read anywhere (only its definition and the docstring mention it). In `recon.build_loss_terms` the `ChunkwiseSubsetReconLossConfig` case (recon.py:404-412) consumes `sites_per_chunk`, `routing`, `n_samples` but ignores `use_fused_kl` entirely. The recon loss is unconditionally KL-on-final-logits.
- **Fix:** Drop the "(when `use_fused_kl`)" parenthetical and either delete the inert `use_fused_kl` field or note in its docstring that it is a semantically-invisible memory optimization currently ignored by the JAX trainer (per LOSS_PARITY_DESIGN §4e), e.g.: "recon is KL against the clean logits; `use_fused_kl` is a memory-only optimization with no effect on the result and is currently not threaded into the JAX recon path."

### `param_decomp/configs.py:500`  — _plumbed-but-ignored_
- **Assumed:** `ComponentActivationDensity.ci_alive_threshold` (default 0.0) sets the CI cutoff used by the ComponentActivationDensity slow-eval plot (mirrors the sibling `CI_L0.ci_alive_threshold` at line 457).
- **Actual:** This per-metric field is never read. The slow-eval density step (`make_slow_eval_step(lm, ci_alive_threshold)`, slow_eval.py:108/130) is fed `eval.ci_alive_threshold`, which `_eval` (lm/config.py:374) sources EXCLUSIVELY from the CI_L0 metric (`ci_l0.ci_alive_threshold`). The ComponentActivationDensity config's own threshold is dropped on the floor.
- **Fix:** Delete the unused `ci_alive_threshold` field from `ComponentActivationDensityConfig` (configs.py:500) so the density plot's threshold is unambiguously sourced from CI_L0; or, if a separate threshold is wanted, wire it through `make_slow_eval_step` instead of reusing `eval.ci_alive_threshold`.

### `param_decomp/configs.py:813-819`  — _documented-unimplemented_
- **Assumed:** `Cadence.keep_last_n_checkpoints` (and `save_every`) genuinely default to `None`, and `None` means 'keep all checkpoints — the conservative choice for research where prior steps may matter.' So a hand-authored config can omit them and get the keep-all behavior.
- **Actual:** Both fields are asserted NON-None before the trainer can run: `run.py:431` (`assert cadence.save_every is not None and cadence.keep_last_n_checkpoints is not None`) and the config-load gate `param_decomp_lab/experiments/config.py:163` (`assert_canonical_algorithm_config`). `make_checkpoint_manager(ckpt_dir, keep_last: int)` even types the arg as a required `int`. A config relying on the documented `None`/keep-all default crashes at convert/run time. Every real yaml sets explicit values (one even comments 'keep_last null -> 2', pile_pgd1.yaml:16).
- **Fix:** Make both fields required (`save_every: PositiveInt` and `keep_last_n_checkpoints: PositiveInt`) and rewrite the docstring to state a value is always required — or, if keep-all is genuinely wanted, implement it (orbax unlimited retention) and drop the two asserts; do not leave a documented default the trainer rejects.

### `param_decomp_lab/experiments/lm/config.py:92-97`  — _documented-unimplemented_
- **Assumed:** `output_extract` (default `"logits"`) is wired: it 'is passed to `make_run_batch`' to pull the prediction tensor out of the model's forward output — i.e. changing it changes which tensor is used as the model output.
- **Actual:** `make_run_batch` does not exist anywhere in the codebase (it was a torch-era function). `output_extract` is declared and set in configs but never read by any code path: the JAX trainer routes by class-name suffix to vendored archs and computes outputs via `DecomposedModel.clean_output`/`masked_output`, ignoring `output_extract` entirely.
- **Fix:** Either delete the inert `output_extract` field (and its docstring + the probe-config setter), or, if kept as a reserved stub, change the docstring to state it is currently UNUSED by the JAX trainer (which routes by class-name suffix and computes outputs via DecomposedModel.clean_output/masked_output) and drop the dead `make_run_batch` reference; fix the matching line in experiments/CLAUDE.md too.

### `param_decomp_lab/experiments/lm/config.py:103-108`  — _documented-unimplemented_
- **Assumed:** `weights_dtype` defaults to `"float32"`, the implied baseline frozen-target precision; `"bfloat16"` is the opt-in footprint optimization. Omitting the field gives a working float32 run.
- **Actual:** Both targets set `supported_weights_dtypes = frozenset({"bfloat16"})` (lines 150, 166) and `assert_supported_weights_dtype` (line 400) asserts `weights_dtype in supported_weights_dtypes`. So the field's own DEFAULT value `"float32"` is unconditionally REJECTED at convert time for every existing target. Every real yaml sets `weights_dtype: bfloat16` explicitly; one comments 'upstream's default fp32 -> bfloat16. The JAX targets are [bf16-only]'.
- **Fix:** Change the default to the only value that works — `weights_dtype: Literal["bfloat16"] = "bfloat16"` (or drop the field's default and add a docstring line noting all current targets are bf16-only) — so the declared default is not an always-rejected value.

### `param_decomp_lab/experiments/lm/config.py:98-102`  — _documented-unimplemented_
- **Assumed:** Setting `activation_checkpointing: true` enables per-block gradient checkpointing on the frozen target forward (the docstring describes a ~33% compute / ~10-15x memory trade via `enable_activation_checkpointing()`, 'the main lever for raising b_per_rank on deep targets').
- **Actual:** `activation_checkpointing` is never consumed anywhere in the JAX codebase (only the field declaration + docstring exist; the only other hit is the torch-era compile-probe generator). The JAX trainer's actual remat levers are `RuntimeConfig.remat_recon_forwards` / `remat_ci_fn`. The described `enable_activation_checkpointing()` 3-pool mechanism is torch-era.
- **Fix:** Delete the unused `activation_checkpointing` field and its docstring from `LMTargetConfig` (config.py:98-102); the real memory levers are `RuntimeConfig.remat_recon_forwards` / `remat_ci_fn`. (If a target-side remat toggle is genuinely wanted later, wire it through `_resolve_target` and the engine — don't leave a no-op field advertising a torch-era mechanism.)

### `param_decomp_lab/experiments/lm/config.py:132-135`  — _documented-unimplemented_
- **Assumed:** `streaming` / `buffer_size` / `shuffle_each_epoch` configure an on-the-fly streaming dataloader (stream from HF with a shuffle buffer, reshuffle each epoch). `is_tokenized` defaults False (so the trainer can tokenize text on the fly) and `column_name` defaults 'text'.
- **Actual:** The JAX trainer reads only pre-tokenized parquet: `_data` (config.py:310) asserts `is_tokenized and not streaming` and `column_name == "input_ids"`. `buffer_size` and `shuffle_each_epoch` are referenced ONLY by the torch-era compile-probe config generator — never read by the trainer. `streaming` is read only in the negative assertion. The defaults (`is_tokenized=False`, `streaming=False`, `column_name='text'`) describe a streaming/tokenize mode the trainer explicitly refuses.
- **Fix:** Delete the four dead fields (is_tokenized, streaming, buffer_size, shuffle_each_epoch) and the misleading defaults; require column_name="input_ids" and dataset_name="parquet" in the schema (the only values _data accepts), so LMDataConfig describes only the pre-tokenized-parquet mode the trainer actually supports — and update gen_probe_config.py to stop emitting the removed keys.

### `param_decomp_lab/experiments/lm/config.py:132-133 vs 310`  — _heavy-default_
- **Assumed:** `is_tokenized: bool = Field(default=False)` and `streaming: bool = Field(default=False)` — a reader sees the dataloader defaulting to 'not pre-tokenized, not streaming', i.e. it will tokenize text at run time as the normal path.
- **Actual:** `_data` (line 310) asserts `data.is_tokenized and not data.streaming` with message 'JAX trainer reads pre-tokenized parquet shards; tokenize offline first'. So the default `is_tokenized=False` is REFUSED — the trainer only accepts pre-tokenized parquet. Lines 313-316 further require `dataset_name == "parquet"` and `column_name == "input_ids"`, but `column_name` defaults to "text" (line 128) — also the wrong/refused value.
- **Fix:** Make the only-supported mode the default and document it: `is_tokenized: bool = Field(default=True)`, `column_name: str = Field(default="input_ids", ...)`, and either drop `streaming` or add a description noting the JAX trainer only reads pre-tokenized parquet shards (runtime HF tokenization/streaming is unsupported).

### `param_decomp_lab/experiments/lm/config.py:98-102`  — _heavy-default_
- **Assumed:** `activation_checkpointing: bool = False` with a docstring: 'If True ... turn on per-block gradient checkpointing on the frozen target forward. Trades ~33% extra compute for ~10-15x less stored activation memory under 3-pool — the main lever for raising b_per_rank on deep targets.' A reader concludes the default False means checkpointing is OFF and flipping it to True will enable the documented memory lever for deep targets.
- **Actual:** The field is never read anywhere in the JAX build/train path (grep for `.activation_checkpointing` / `activation_checkpointing=` returns only the declaration + one YAML). It is dead config: the docstring describes torch-3-pool behavior; the real JAX memory levers are `RuntimeConfig.remat_recon_forwards` / `remat_ci_fn`. Setting it True does nothing.
- **Fix:** Delete the dead `activation_checkpointing` field + docstring (and remove `activation_checkpointing: true` from llama8b_l18_b128_cmp32.yaml); the JAX memory levers are RuntimeConfig.remat_recon_forwards / remat_ci_fn.

### `param_decomp/run.py:503-504`  — _reported-ne-executed_
- **Assumed:** The logged `train/schedules/lr/components` and `train/schedules/lr/ci_fn` report the learning rate the optimizer actually applied in the step that just completed — `build_optimizers`' docstring even claims the schedule fns are returned so 'the log path reports the exact LR the optimizer applies (single source of truth)'.
- **Actual:** They log `sched_vu(now_step)` / `sched_ci(now_step)` where `now_step = step + 1`, but optax evaluates the schedule at the optimizer's internal `count`, which equals the pre-increment `state.step == now_step - 1`. So the logged LR is the LR for the NEXT step, off by one. Empirically confirmed: optax's first `.update()` uses `schedule(0)` (zero update magnitude under an identity schedule), while the log at `now_step=1` reports `schedule(1)`.
- **Fix:** Log at the pre-increment count to match what the optimizer applied: use the loop `step` (not `now_step`), i.e. `record["train/schedules/lr/components"] = float(jnp.asarray(sched_vu(step)))` and `... sched_ci(step)` — this aligns the logged LR with optax's pre-increment `state.count` and with p_imp/src_lr in the same record.

### `param_decomp/run.py:462-477`  — _reported-ne-executed_
- **Assumed:** The comment 'flatten the metric lists into the same flat keys torch logs (E14) so cross-impl wandb config queries line up' plus the MetricsSink comment 'not just the flattened wandb.config dict' say wandb.config carries the run's full algorithm config flattened into torch-parity keys (e.g. `pd.loss_metrics.ImpMin.coeff`), queryable for cross-impl comparison.
- **Actual:** `flatten_typed_lists(...)` is applied to a dict containing ONLY `jax_runtime={n_devices, n_processes, remat_*, run_id, run_dir}` — which has no typed lists, so the flatten is a no-op. wandb.config never contains `pd.loss_metrics`, optimizers, seed, eps, coeffs, etc. The full config lives only in the separately-saved `config.yaml` file (run.py:176-178), not in wandb's queryable config.
- **Fix:** Drop the stale flatten call and metric-list comment: `wandb_config = {\"jax_runtime\": {...}}` directly, comment it as 'topology only; full algorithm config is in the saved config.yaml, not wandb.config', and fix the MetricsSink comment at run.py:174-175 (remove 'not just the flattened wandb.config dict'). If cross-impl config queries are actually wanted, restore `dict(raw_cfg, jax_runtime=...)` so the loss-metric flattening runs again.

### `param_decomp/configs/llama8b_full32L_seq512_b128_dp128.yaml:10-11`  — _temp-riding-prod_
- **Assumed:** Reader assumes the V/U C-axis shards over the dp=128 mesh and that init enforces C % dp == 0 (so per-matrix C values rounded to multiples of 128 are sized to the dp axis).
- **Actual:** After the 2-D mesh refactor, the C axis shards on the tp axis, not dp: components.py:62-63 says "V (d_in, C) shards d_in on dp + C on tp", and the assert is C % tp. The sibling b32/tp4 config carries the corrected text ("the V/U C-axis shards on the `tp` mesh axis; init asserts C % tp == 0"); this b128 file still says "shards over the dp=128 mesh; init asserts C % dp == 0".
- **Fix:** Update lines 10-11 to match the sibling config: "(the V/U C-axis shards on the `tp` mesh axis; init asserts C % tp == 0)".

### `param_decomp/configs.py:267-269, 278`  — _half-migration_
- **Assumed:** `ChunkwiseSubsetReconLoss` is wired by routing onto `recon.subset_chunk_plan` (via a `chunkwise_plan` builder), and `use_fused_kl` toggles whether recon uses the fused-linear-KL.
- **Actual:** The production path (`recon.build_loss_terms`, recon.py:404-412) inlines `make_plan(into_groups(...))` and never calls `subset_chunk_plan`; there is no `chunkwise_plan` builder at all. `use_fused_kl` is parsed but read NOWHERE — the JAX trainer has one recon semantics (KL on final logits), with no fused/unfused toggle.
- **Fix:** In ChunkwiseSubsetReconLossConfig's docstring: drop the false `subset_chunk_plan`/`chunkwise_plan` wiring claim (say build_loss_terms inlines `make_plan(into_groups(...))`), and delete the dead `use_fused_kl` field plus its docstring sentence (the JAX trainer has one recon semantics, KL on final logits, with no fused/unfused toggle).

### `param_decomp/recon.py:290-300`  — _half-migration_
- **Assumed:** `subset_chunk_plan` is "The production plan" that the engine uses for `ChunkwiseSubsetReconLoss` (as its docstring says, echoed by tests at test_stacked_parity.py:189 "what reaches the engine").
- **Actual:** The production engine path (`build_loss_terms`, recon.py:404) re-derives the plan inline with `cfg.routing`; `subset_chunk_plan` is called ONLY from tests. It also hardcodes a fresh `UniformKSubsetRoutingConfig()` rather than threading the config's routing, so it is a parallel reimplementation that merely coincides with production today (kept in sync only by the line-405 assertion).
- **Fix:** Change recon.py:296-297 docstring to state it is a TEST-ONLY plan builder mirroring the inline production case in `build_loss_terms` (recon.py:404-412), and correct the matching claims at test_stacked_parity.py:189 and configs.py:267.

## Severity 2  (12)

### `param_decomp_lab/experiments/tms/model.py:4`  — _claim-vs-code_
- **Assumed:** Module docstring: 'Torch reference (read-only ground truth): `param_decomp_lab/experiments/tms/models.py`.' A reader treats this as a live pointer to the authoritative torch implementation to consult for semantics.
- **Actual:** No `models.py` exists in that directory (verified: the dir holds config.py, model.py [singular — this file itself], run.py, test_tms.py). The torch reference was deleted in the de-torching. The pointer dangles.
- **Fix:** Point at the tag, not the dead working-tree path: `Torch reference (read-only ground truth): param_decomp_lab/experiments/tms/models.py at git tag torch-oracle.`

### `param_decomp_lab/experiments/resid_mlp/model.py:5-7`  — _claim-vs-code_
- **Assumed:** Module docstring: 'Torch reference (read-only ground truth): `param_decomp_lab/experiments/resid_mlp/` (`models.py` the architecture, `train_resid_mlp.py` the read-off pretrain objective, `data.py` the synthetic sparse features).' A reader expects these three torch files to exist alongside as the authoritative reference.
- **Actual:** None of `models.py`, `train_resid_mlp.py`, `data.py` exist in that directory (verified: only config.py, model.py, run.py, test_resid_mlp.py). All deleted in the de-torching. Three dangling pointers presented as ground truth.
- **Fix:** Repoint the docstring at the tag, e.g. "Torch reference (read-only ground truth, at git tag `torch-oracle`): `param_decomp_lab/experiments/resid_mlp/{models.py, train_resid_mlp.py, data.py}`" — since those files no longer exist at HEAD.

### `param_decomp/eval.py:10`  — _claim-vs-code_
- **Assumed:** 'Variant semantics mirror `param_decomp_lab/eval_metrics/ce_and_kl_losses.py`'. A reader expects to open that torch file to confirm the six CE/KL masking variants' semantics.
- **Actual:** `param_decomp_lab/eval_metrics/` is empty (verified: only `__pycache__`, no .py files). The cited file was deleted with the torch eval metrics. (Same dead-reference recurs at slow_eval.py:10 → `plotting.py` and attn_patterns_eval.py:4 → `attn_patterns_recon_loss.py`.)
- **Fix:** Either restore the reference to the surviving authority — "Variant semantics mirror the torch oracle's `CEandKLLosses` (git tag `torch-oracle`)" — or drop the dead path and point at the frozen golden `param_decomp/tests/equivalence/` that actually pins the semantics; apply the same fix to slow_eval.py:10 and attn_patterns_eval.py:4.

### `param_decomp/configs.py:777 (def); param_decomp/run.py:348 (use)`  — _plumbed-but-ignored_
- **Assumed:** `PDConfig.faithfulness_warmup_weight_decay` (NonNegativeFloat, documented "Weight decay for warmup phase optimizer") sets the AdamW weight decay used during the faithfulness-warmup phase.
- **Actual:** The warmup optimizer hardcodes it: `optax.adamw(pd.faithfulness_warmup_lr, weight_decay=0.0)` (run.py:348) — `faithfulness_warmup_weight_decay` is never read into the optimizer. Its only consumer is the lab assert `cfg.pd.faithfulness_warmup_weight_decay == 0.0` (experiments/config.py:151), which forbids any nonzero value. The knob can only hold the value the call site already hardcodes.
- **Fix:** Delete the `faithfulness_warmup_weight_decay` field from PDConfig (configs.py:777) and the now-redundant assert at experiments/config.py:151; the warmup AdamW already hardcodes weight_decay=0.0 (per N1, optax's default wd of 1e-4 must be overridden anyway). If a tunable is genuinely wanted instead, change run.py:348 to `optax.adamw(pd.faithfulness_warmup_lr, weight_decay=pd.faithfulness_warmup_weight_decay)` and drop the ==0.0 assert.

### `param_decomp_lab/experiments/lm/config.py:347-349`  — _plumbed-but-ignored_
- **Assumed:** The eval-tier `PGDReconLoss` probe honors all four PGD fields it requires (`init`, `step_size`, `n_steps`, `mask_scope`) — they are non-defaulted required fields on `PGDConfig`, so a reader assumes each shapes the probe.
- **Actual:** Only `n_steps` and `step_size` flow into `EvalPGDConfig`; `init` and `mask_scope` are asserted to fixed values (`metric.init == "random" and metric.mask_scope == "c"`). The eval step hardcodes a `(1,1,C+1)` random source (eval.py:89-94, 160-176), so those two required fields can only ever be `"random"`/`"c"`.
- **Fix:** In configs.py give the eval probe its own narrowed config (no init/mask_scope fields) instead of reusing the full PGDConfig base, OR add a docstring/Field note on PGDConfig that the eval tier only accepts init=random, mask_scope=c; minimally, the convert-time assert message at config.py:348 should state the eval probe only supports those two values.

### `param_decomp_lab/experiments/lm/config.py:130-131`  — _documented-unimplemented_
- **Assumed:** `train_split` / `eval_split` select which dataset splits the trainer and in-loop eval read (the names and the run.py:212 comment 'Mirrors the torch eval_split: train stream' imply split selection).
- **Actual:** Neither field is read by the trainer. The eval pass (`run.py:219`) builds its `BatchSchedule` over the SAME parquet shard dir as training with a different seed (`pd.seed + 1`); there is no split selection. `train_split` appears only in the compile-probe generator; `eval_split` only in a descriptive comment.
- **Fix:** Delete the unused `train_split`/`eval_split` fields from LMDataConfig (and drop them from the probe config + the run.py:212 comment); the eval stream is just the training corpus read with seed `pd.seed + 1`, not a split selection.

### `param_decomp_lab/experiments/lm/config.py:92-97`  — _heavy-default_
- **Assumed:** `output_extract: int | str | None = "logits"` with docstring 'passed to `make_run_batch`, pulls the prediction tensor out of the model's forward output (default "logits")'. A reader assumes the default extracts the logits field and that production targets rely on it.
- **Actual:** `output_extract` is never read in the JAX build path (grep for `.output_extract` returns nothing) and `make_run_batch` no longer exists in the codebase (it was a torch artifact; the JAX trainer routes by class-name suffix). The default "logits" is also the WRONG value for both production targets, whose YAMLs all set `output_extract: 0` — but since the field is inert this never bites.
- **Fix:** Delete the dead `output_extract` field (and its docstring paragraph) from LMTargetConfig and drop it from the YAMLs; the JAX trainer routes by class-name suffix and the vendored arch returns logits directly, so nothing reads it. (If keeping for forward-compat, at minimum fix the docstring: remove the `make_run_batch` reference and mark the field unused by the JAX trainer.)

### `param_decomp/run.py:505-507`  — _reported-ne-executed_
- **Assumed:** `train/mem/peak_gb_per_rank` reports the peak memory of the rank (the whole process).
- **Actual:** It reads `jax.local_devices()[0].memory_stats()['peak_bytes_in_use']` — only device 0's peak. It is per-DEVICE, not per-rank. It coincides with per-rank only because production launches 1 device per process (`--ntasks-per-node=8`); on any multi-device-per-process topology (e.g. an inline multi-GPU debug run) it silently under-reports by ignoring devices 1..n.
- **Fix:** Aggregate across all local devices: `peak = max(d.memory_stats()["peak_bytes_in_use"] for d in jax.local_devices())` (and rename to `peak_gb_per_device` if a per-device figure is actually intended).

### `param_decomp/configs.py:655-663`  — _temp-riding-prod_
- **Assumed:** remat_ci_fn defaults to False, so a reader writing a new big-target config and omitting it gets the default lightweight path.
- **Actual:** On big targets the non-remat path is the OOM/compile-wall path: both production full32L configs must set remat_ci_fn: true, and the b128 header item #5 says "without it the step is the GPU compile wall — ~80 min + near-OOM — since the whole CI forward materialises." So the documented default (False) is the path you must NOT use at production scale; the safe value lives only in the yamls. This is the same default-is-the-heavy-trap shape as the prior remat_ci_fn catch (which fixed the hardcoded-False call site in c204b6afc), now reincarnated as a misleading config default.
- **Fix:** Amend the configs.py:655-663 Field description to warn that big/full-model targets MUST set remat_ci_fn: true (default False is only viable for toys/small targets) — e.g. add "On full-model targets the non-remat path is the ~80-min compile wall + near-OOM since the whole CI forward materialises; the production full32L configs set true."

### `param_decomp/recon.py:436-437`  — _half-migration_
- **Assumed:** `StochasticHiddenActsReconLoss` is an offline-eval-only loss living on a torch bridge, not implemented in the JAX trainer.
- **Actual:** The hidden-acts recon metrics are fully implemented JAX-native and run IN-LOOP (`hidden_acts_eval.py` + `slow_eval.py:671-672`). `slow_eval.py:27` even states this was "amended 2026-06-16 from keep-on-bridge". The recon.py comment is stale from the pre-amendment design.
- **Fix:** Rewrite the comment to: "StochasticHiddenActsReconLoss / CIHiddenActsReconLoss are not recon-grid TRAINING terms; they're standalone in-loop eval metrics computed JAX-native in slow_eval.py (SPEC S31, amended 2026-06-16). They must not appear in a training-loss config." — drop the "keep-on-bridge / offline-eval only" wording.

### `param_decomp/hidden_acts_eval.py:2-3`  — _half-migration_
- **Assumed:** The JAX hidden-acts metrics are "offline counterparts" of torch metrics that live at `param_decomp/metrics/stochastic_hidden_acts_recon.py`.
- **Actual:** `param_decomp/metrics/` does not exist in the repo (only in stale worktrees) — the torch metric module was deleted, and these JAX metrics are the in-loop primary implementation, not an "offline counterpart" of anything live.
- **Fix:** Rewrite lines 2-3 to: "the JAX-native in-loop eval metrics CIHiddenActsReconLoss / StochasticHiddenActsReconLoss (computed on eval.slow_every via slow_eval.py); they preserve the log-key/reduction semantics of the now-deleted torch metrics of the same name." Drop the dead path reference and the word 'offline'.

### `param_decomp_lab/experiments/lm/load_run.py:1-2, 108-109`  — _half-migration_
- **Assumed:** `open_jax_run` is consumed by harvest, clustering, autointerp, slow-eval, AND the app; and `LoadedJaxRun` mirrors "the torch `PDAdapter`".
- **Actual:** The app package was removed (`param_decomp_lab/app` does not exist) and slow-eval runs in-loop only (no offline CLI per CLAUDE.md). Actual consumers are harvest + clustering. `PDAdapter` (adapters/pd.py) is explicitly torch-FREE — calling it "the torch `PDAdapter`" implies a torch version that no longer exists.
- **Fix:** Module docstring: list consumers as "harvest and clustering" (autointerp/slow-eval/app use run_metadata or were removed). LoadedJaxRun docstring: drop "torch" — these fields mirror the (JAX-native, torch-free) `PDAdapter` fields the harvest pipeline keys on.

## Severity 1  (2)

### `param_decomp/run.py:4-10`  — _half-migration_
- **Assumed:** The engine signature is `(pd, cadence, run, lm, ci_fn, data, remat_recon_forwards, sample_batch, eval_fn, eval_every, mesh)` and the target injects TWO seams (sample_batch, eval_fn).
- **Actual:** The real signature (run.py:384-397) inserts `remat_ci_fn` between `remat_recon_forwards` and `sample_batch`, and the function's own docstring (run.py:409) says THREE seams (sample_batch, eval_fn, eval_every).
- **Fix:** In run.py:4-5, add the missing `remat_ci_fn` parameter to the module-docstring signature (after `remat_recon_forwards`) so the inline arg list matches the real 12-arg signature at run.py:384-397; optionally reconcile "two seams" (line 10) with the function docstring's "three seams" (line 409).

### `param_decomp_lab/adapters/pd.py:13-18`  — _half-migration_
- **Assumed:** From `is_jax_run`'s docstring ("a torch run instead has `model_*.pth`"), torch and JAX runs coexist and the adapter discriminates between two loadable kinds.
- **Actual:** The repo is torch-free; there is no torch adapter and torch runs cannot be loaded. `is_jax_run` is used purely as an assertion guard (adapters/__init__.py:12) that the dir is a JAX run — there is no torch branch anywhere.
- **Fix:** Rewrite the docstring to state what it does — assert a dir is a valid JAX run (has `config.yaml` and an orbax `ckpts/`) — and drop the obsolete `model_*.pth` "torch run" clause, since torch runs are no longer loadable.
