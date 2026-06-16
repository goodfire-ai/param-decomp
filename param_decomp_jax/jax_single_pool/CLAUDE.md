# jax_single_pool — agent notes

Single-pool VPD trainer in JAX, **generic over vendored LM targets**. The semantics
source of truth is `SPEC.md` (normative pseudocode + numbered invariants, grounded in
the stable torch `param_decomp` impl). See `README.md` for the file map.

Open items: persistent-source scopes beyond `sc`, sigmoid parameterization, and the
hidden-acts seam are deliberately refused (SPEC S31, LOSS_PARITY_DESIGN §6 stage 4 —
hidden-acts is keep-on-bridge); persistent `start_frac>0` is now implemented (SPEC S32,
`term_active` `where`-gating); SPEC S24's two torch-parity quirks (PPGD warmup
route-all, fresh-PGD single routing draw) are pinned pending a team decision; CI-fn
numerics (gelu erf, rms eps) unify with torch at a run boundary.

## The one rule

**Every change is checked against SPEC.md, by invariant ID.** If a change deviates
from an invariant, either fix the change or (deliberately, with Oli) amend the spec —
never silently diverge. Cite IDs (`S14`, `N1`, …) in commit messages and reviews.

## Architecture in one breath

`lm.py` defines `DecomposedLM` — ordered `sites` + four pure fns (`clean_logits`,
`site_inputs`, `masked_logits`, `weight_deltas`), flat site-name-keyed dicts at the
boundary, frozen pytree always a runtime arg (never a jit closure constant — an 8B
target becomes a multi-GB HLO constant). `train.py` is the generic step factory
(fp32 masters / bf16 compute) over a static tuple of recon loss TERMS (S10′ — the
torch loss-class cartesian product factored as plan × mask-source strategy, built
from the shared configs by `recon.build_recon_terms`; see LOSS_PARITY_DESIGN.md),
consuming `losses.py` (pure loss terms + schedules) and `adversary.py` (persistent
vs fresh source machinery — semantically distinct adversaries sharing only
`source_masks`); `ci_fn.py` the shared CI transformer; `llama8b.py` + `llama8b_sharding.py` the first target. There is ONE
recon semantics: masks thread through the suffix forward, loss is KL on final logits
(SPEC §2.3–2.5). Site-local recon is a conceptual no-no, not a "simplification".
`llama_simple_mlp.py` is the second target (the pile-pretrained `LlamaSimpleMLP`,
t-9d2b8f02; sites `h.{i}.attn.{q,k,v,o}_proj` / `h.{i}.mlp.{c_fc,down_proj}`) —
config dispatch is `TargetConfig` (llama8b) vs `LlamaSimpleMLPTargetConfig` in
`config.py` (which also reads the canonical `param_decomp_config` schema DIRECTLY —
`build_experiment_config`/`load_wrapper` — routing `kind: pretrained` specs + `h.*`
wildcards), target build in `run.py::main`. Export/offline-eval stay llama8b-only
(guarded).

## Invariants with sharp teeth (the ones that have actually bitten)

- **S3**: the recon target is the FROZEN-path forward (`clean_logits`), never the
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
2. `tests/equivalence/` — fixture-driven torch↔JAX per-term numeric equivalence
   (fp32, no RNG, zeroed attn). Regenerate goldens only when the MATH changes:
   `gen_fixtures.py` (JAX env) then `torch_reference.py` (torch env).
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

**Launch via `pd-jax-lm <wrapper.yaml> --nodes N`** (lab-side, torch venv): mints the
`p-` run id, snapshots the tree to `refs/runs/snapshot/<id>`, materializes an
immutable shared-FS workspace (clone + both venvs) at
`$PARAM_DECOMP_OUT_DIR/workspaces/<id>`, stamps the id into the workspace's wrapper
yaml, and sbatches. Requeues re-enter the workspace, never the live checkout.
`--run_id` resubmits an existing workspace. Don't hand-write sbatch files.

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
