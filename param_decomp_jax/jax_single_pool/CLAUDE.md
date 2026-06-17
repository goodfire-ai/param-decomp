# jax_single_pool — agent notes

Single-pool VPD trainer in JAX, **generic over vendored LM targets**. The semantics
source of truth is `SPEC.md` (normative pseudocode + numbered invariants, grounded in
the stable torch `param_decomp` impl). See `README.md` for the file map.

Open items: persistent-source scopes `c`/`nsc` and sigmoid parameterization are
deliberately refused. The hidden-acts seam is now BUILT (SPEC S31 amended 2026-06-16):
`CIHiddenActsReconLoss` / `StochasticHiddenActsReconLoss` are standalone offline eval
metrics (`hidden_acts_eval.py`, via `jsp-slow-eval`) over a fifth model fn
`masked_site_outputs` — NOT recon-grid training terms (the recon loss stays
KL-on-final-logits). `sc` and `bsc` are supported (`bsc` is batch-sharded:
an independent source per batch element and position, no cross-replica sync — SPEC
S16/D1). Persistent `start_frac>0` is now implemented (SPEC S32, `term_active`
`where`-gating); SPEC S24's two torch-parity quirks (PPGD warmup route-all, fresh-PGD
single routing draw) are pinned pending a team decision. CI-fn numerics are unified
with the torch oracle (#624/#625/#730 resolved): GELU is exact-erf
(`approximate=False`) and RMSNorm eps is `finfo(fp32).eps` (`CI_FN_RMS_EPS`).

## The one rule

**Every change is checked against SPEC.md, by invariant ID.** If a change deviates
from an invariant, either fix the change or (deliberately, with Oli) amend the spec —
never silently diverge. Cite IDs (`S14`, `N1`, …) in commit messages and reviews.

## Architecture in one breath

`lm.py` defines `DecomposedModel` — ordered `sites` + `leading_axes` + five pure fns
(`clean_output`, `site_inputs`, `masked_output`, `weight_deltas`) plus a pluggable
`recon_loss_fn` (default `kl_per_position`), flat site-name-keyed dicts at the boundary,
frozen pytree always a runtime arg (never a jit closure constant — an 8B target becomes a
multi-GB HLO constant). The activation waist is GENERIC `[*leading, d]` (masks/CI
`[*leading, C]`), `leading = (batch,) + named position axes`: masking / routing / sources
/ imp-min all read an opaque `leading = residual.shape[:-1]`; reductions are
`math.prod(shape[:-1])` / `axis=tuple(range(ndim-1))`. CI is independent over every leading
axis (no per-axis CI semantics, only axis NAMES — see AXIS_SEMANTICS_DESIGN.md).
`DecomposedModel.leading_axes` names the position axes (`("sequence",)` for LM, `()` for
TMS); `CIFn.expects_axes` mirrors it, and `init_train_state` asserts they're equal (early
fail) so the CI fn stays per-domain (RoPE over `sequence`) without the core adapting. The
three EDGES are generic so non-LM (bio-style) targets fit (#828): the model INPUT
(`prefix_residual_fn(prefix, inputs)` in `run.py` takes `Any` — tokens for an LM, a dict
for bio), the model OUTPUT (`clean_output`/`masked_output` return `Any` — logits, a tuple
of heads, coords; field NAMES stay `*_logits` pending a deferred rename), and the recon
comparison (`recon_loss_fn(clean_output, masked_output) -> scalar`, default
`kl_per_position` so the LM path is byte-identical). The waist shape contract (all per-site
tensors in one forward share one `*leading` prefix) is enforced at trace time by
`@jaxtyped(typechecker=beartype)` on the core `step`, `masked_forward`, and the loss fns.
`train.py` is the generic step factory
(fp32 masters / bf16 compute) over a static tuple of recon loss TERMS (S10′ — the
torch loss-class cartesian product factored as chunking × routing × mask-source
strategy: a chunking helper (`one_chunk`/`per_site`/`into_groups`) feeds the single
`make_plan` constructor, built from the shared configs by `recon.build_recon_terms`;
see LOSS_PARITY_DESIGN.md),
consuming `losses.py` (pure loss terms + schedules) and `adversary.py` (persistent
vs fresh source machinery — semantically distinct adversaries sharing only
`source_masks`); `ci_fn.py` the shared CI transformer; `llama8b.py` + `llama8b_sharding.py` the first target. There is ONE
recon semantics: masks thread through the suffix forward, loss is KL on final logits
(SPEC §2.3–2.5). Site-local recon is a conceptual no-no, not a "simplification".
`llama_simple_mlp.py` is the second target (the pile-pretrained `LlamaSimpleMLP`,
t-9d2b8f02; sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}`) —
config dispatch is `TargetConfig` (llama8b) vs `LlamaSimpleMLPTargetConfig` in
`config.py` (which also reads the canonical `param_decomp_config` schema DIRECTLY —
`build_experiment_config`/`load_config` — routing `kind: pretrained` specs + `h.*`
wildcards), target build in `run.py::main`. The slow plot metrics are computed
NATIVELY in JAX via `jsp-slow-eval` (`slow_eval.py`) — no torch export round-trip
(the torch offline-eval bridge `jsp-export` / `pd-offline-eval` was retired).
`tms.py` is the third target and the **first non-LM one** (`leading_axes=()`, no position
axis; the waist is `[B, n_features]`) — the proof the generic `[*leading, d]` core fits a
positionless target. The TMS target is the Anthropic toy (`out = relu(linear2(linear1(x)))`,
weights TIED) with an UNTIED 2-site decomposition (`linear1`/`linear2`, independent V/U);
the recon comparison is `recon_loss_fn = tms_mse` (MSE on the post-ReLU output, NOT KL),
there is NO prefix (the whole model is decomposed, residual = raw input `x`), and the
frozen target is **pretrained from scratch in-process** (`pretrain_tms_target`, the
`mean((|x|-out)²)` objective — no wandb/cache). `ci_fn_mlp.py` is the second CI-fn arch:
the **layerwise per-site MLP** (`fn_type=mlp`, `expects_axes=()`, `LayerwiseMLPCIFn`) —
one independent MLP per site mapping `site_input [B,d_in] -> [B,C]` logits through the
same `lower/upper_leaky_hard` squashings; `run_state.init_train_state` dispatches CI-fn
construction on `cfg.ci_fn` (`CIArch` transformer vs `MLPCIArch`) and uses replicated
(not C-sharded) V/U + CI for the tiny TMS. Config dispatch: `config._is_tms_schema`
(structural `target.n_hidden` marker) routes the canonical schema to
`build_tms_experiment_config` (torch-free `param_decomp_config.tms.TMSExperimentConfig`);
`TMSTargetConfig` / `TMSDataConfig` join the `AnyTargetConfig` / `AnyDataConfig` unions;
`run.py::main` builds + pretrains the target and calls `train_tms` (the TMS composition
over the unified core — same `init_train_state` / faith warmup / `make_train_step` /
orbax checkpointing, but synthetic data + the ground-truth target-CI metric
`tms.identity_ci_error` instead of the LM CEandKLLosses eval pass). Harvest / slow-eval /
export over TMS are NOT wired (`load_run.build_target` / `export` / `slow_eval` assert
LM). **Finding (axis-semantics):** the core needed ZERO change for TMS — only ADDED a
target + a CI arch + dispatch. The one deviation from the torch `fn_type=mlp` is by
design: torch's scalar `MLPCiFn` feeds `get_component_acts(x)=x@V` (per-component scalar),
which couples the CI-fn input to the trained components and so does NOT fit the generic
`ci_fn(site_inputs)` waist; the vector-input per-site MLP (consumes the raw `site_input`)
fits the waist with no core change and recovers a clean identity in the non-superposed
(`n_hidden==n_features`) regime (validated end-to-end in `tests/test_tms.py`).
`resid_mlp.py` is the fourth target and the **second non-LM one** — the SPD/APD
residual-stream toy (`leading_axes=()`, no position axis; the waist is the residual
stream `[B, d_embed]`). A FIXED input embedding `W_E` (`n_features→d_embed`), `n_layers`
MLP blocks each reading/writing the `d_embed` residual stream, a FIXED unembed `W_U =
W_Eᵀ`; the DECOMPOSITION targets the per-layer MLP matrices (sites `layers.{i}.mlp_in` /
`layers.{i}.mlp_out`, UNTIED V/U). Unlike TMS it HAS a prefix (`W_E`, so
`resid_mlp_input_residual(frozen, x) = x @ W_E` — the prefix `W_E` is carried inside the
frozen target itself, no separate `prefix` slot), and unlike LM the recon comparison is
`recon_loss_fn = resid_mlp_mse` (MSE on the model output `[B, n_features]`, NOT KL). The
frozen target is **pretrained from scratch in-process** (`pretrain_resid_mlp_target`, the
read-off `mean((out − (act_fn(x)+x))²)` objective with trivial unit label coeffs — no
wandb/cache). It REUSES `LayerwiseMLPCIFn` (`fn_type=mlp`, no new CI arch) and the same
ground-truth identity-CI metric (`resid_mlp.identity_ci_error`, the single-feature probe
through `W_E`). Config dispatch: `config._is_resid_mlp_schema` (structural `target.d_embed`
marker, disjoint from TMS's `n_hidden` and the LM's `spec`) routes the canonical schema to
`build_resid_mlp_experiment_config` (torch-free
`param_decomp_config.resid_mlp.ResidMLPExperimentConfig`); `ResidMLPTargetConfig` /
`ResidMLPDataConfig` join the `AnyTargetConfig` / `AnyDataConfig` unions and `ResidMLPTarget`
joins `AnyFrozenTarget`; `run.py::main` builds + pretrains the target and calls
`train_resid_mlp` (mirroring `train_tms`). Harvest / slow-eval / export over ResidMLP are
NOT wired (`load_run.build_target` asserts against it; `export` is llama8b-only). **Finding:**
again ZERO core change — only ADDED a target + dispatch (the CI arch was reused). The
identity-embedding regime sets `W_U = W_Eᵀ = I` (the salvaged design choice; torch leaves
`W_U` at its discarded random init there since only `mlp_in` CI structure is asserted), the
unambiguous clean per-feature ground truth (validated end-to-end pretrain→decompose→identity
recovery in `tests/test_resid_mlp.py`).

## Invariants with sharp teeth (the ones that have actually bitten)

- **S3**: the recon target is the FROZEN-path forward (`clean_output`), never the
  `mask=1` decomposed identity (bf16 rounding + V/U in the stopped graph).
- **S13/S15**: source updates go through the persistent Adam AND project to [0,1]
  after EVERY ascent — an unprojected drift past 1 has zero `clip` gradient and the
  entry dies.
- **S14**: the final source ascent reuses the main backward's source-grad
  (pre-update θ), unscaled by the ppgd coeff. No extra forward.
- **N1**: fp32 masters everywhere (`optax.adamw(..., weight_decay=0.0)` — optax's
  default wd is 1e-4, torch's is 0; this was audit finding A7).
- **`inv_freq` is a buffer, not a param** — `stop_gradient` in `CIFn.__call__`.
- **S10/S11**: chunking is sequential `sites_per_chunk` groups in canonical site
  order; routing is uniform-k over the chunk's sites only.

## Validation stack (run all before claiming correctness)

1. `pytest jax_single_pool/tests/` — at the default device count AND
   `XLA_FLAGS="--xla_force_host_platform_device_count=4"`.
2. `tests/equivalence/` — fixture-driven JAX-vs-frozen-golden per-term numeric
   equivalence (fp32, no RNG, zeroed attn). The torch references are FROZEN committed
   goldens (`torch_reference.json`, `simple_mlp_equivalence/*.npz`,
   `tools/export_fixtures/*`); the torch generators/verifier that produced them are
   deleted so `param_decomp_jax` imports no torch (push-1). Regenerate goldens only when
   the MATH changes: redraw fixtures JAX-side with `gen_fixtures.py`, then check out the
   `torch-oracle` git tag in a torch-venv worktree and run that revision's
   `torch_reference.py` / `gen_torch_fixtures.py` / `gen_export_fixture.py`, copying the
   emitted goldens back here.
3. `experiments/invariance_check.py` at 4 sim devices — trajectory invariant to
   device count up to float reassociation (SPEC D4).

`basedpyright jax_single_pool/` must be clean; the package stays out of the repo
`[tool.pyright]` include (torch venv type-checks the torch side).

## The training pipeline (`run.py`)

`jsp-train <config.yaml>` is the composition root and the only I/O layer; the step
stays pure. Data is a pre-tokenized parquet artifact under
`$DATA_MOUNT/artifacts/mechanisms/param-decomp/datasets/` (`fineweb_llama_tok_2048`
for Llama-8B, `pile_neox_tok_512` for `LlamaSimpleMLP`) — NEVER stream/tokenize from
HF at run time (the 80-rank thunderherd lesson). The batch schedule is a pure
function of `(seed, step)` (O(1) resume, no replay); checkpoints are orbax sharded
saves (no on-loop full-gather); SIGTERM → save → SLURM requeue → resume from latest.
Resume with a changed config is refused (byte-compare). Smokes before a long run
MUST exercise save AND resume at the production per-rank shape.

A run config is ONE self-contained yaml: the `param_decomp_config` experiment schema
(`pd`/`data`/`eval`/`cadence`/`runtime`/`target`/`wandb`) plus the run-instance fields
the schema now also carries — top-level `run_name`/`run_id`/`out_dir`, the
`runtime.remat_recon_forwards` memory/compute knob, and `wandb.group`/`wandb.tags`.
`run_id`/`out_dir` are absent in a hand-authored config; `pd-jax-lm` mints + stamps them.

**Launch via `pd-jax-lm <config.yaml> --nodes N`** (lab-side, torch venv): mints the
`p-` run id, snapshots the tree to `refs/runs/snapshot/<id>`, materializes an
immutable shared-FS workspace (clone + both venvs) at
`$PARAM_DECOMP_OUT_DIR/workspaces/<id>`, stamps the id (+ out_dir / wandb group / tags)
into the workspace's single config yaml, and sbatches. Requeues re-enter the workspace,
never the live checkout. `--run_id` resubmits an existing workspace. Don't hand-write
sbatch files.

## Gotchas

- **`shard_batch` topology** (`sharding.py`): uses `make_array_from_process_local_data`
  so it's correct for BOTH single-process-many-devices and multi-process-1-device.
  Do NOT revert to the per-`process_index()`-slice idiom — it silently replicates one
  slice on single-process multi-device CPU.
- **`vendored_jax` is part of this distribution** (moved from `jax_spike/`); no
  `sys.path` hacks anywhere. If an import fails, the install is broken — fix the env
  (`uv pip install -e .`), don't add a path shim. (The old `jax_spike` stage scripts
  that imported it by cwd are superseded by this package.)
- **Bench schedules**: `llama8b_real.py` anneals over `--total_steps` (default 100k),
  not the benched `--steps` — short benches measure start-of-training semantics.
