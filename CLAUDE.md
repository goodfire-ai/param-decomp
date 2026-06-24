# CLAUDE.md

Guidance for Claude Code agents working in this repo. Sub-CLAUDE.md files carry the
module-specific detail; see [Module pointers](#module-pointers).

## Environment

**Always activate the venv before running Python or git:**

```bash
source .venv/bin/activate
```

In a worktree, run `uv sync` first so the worktree has its own `.venv`. Do NOT `cd` to
the main repo — all commands (including git) run in the worktree.

`.env` file with WandB credentials required (see `.env.example`).

**The repo is torch-free.** All torch was deleted (trainer is JAX; torch consumers were
de-torched; the torch oracle lives at git tag `torch-oracle`). The one torch island left
is `nano_param_decomp/` — a standalone single-file VPD reference impl for paper readers,
excluded from `make type` and not imported by any package.

**One venv.** There is a single `.venv` at the repo root, built by `make install-dev`
(one `uv sync --all-packages` over the workspace + pre-commit). jax is a normal
dependency of the root `param-decomp` distribution (CPU jax base); the cluster CUDA
wheels come from a `[cuda]` extra (`uv sync --extra cuda`), which the per-run launch
workspace installs. Use `make install-dev`, never a bare `uv sync` (it would skip the
lab dev deps).

## Project overview

PD is a research framework for sparse parameter decomposition: target-model parameters are
decomposed into a sum of components; per-datapoint **causal importance (CI)** quantifies
how much each component can be masked; multiple loss terms balance faithfulness,
reconstruction, and sparsity.

Three experimental domains: TMS (Toy Model of Superposition), ResidualMLP, and Language
Models. The LM experiment decomposes any HuggingFace-loadable model whose target modules
are `nn.Linear`, `nn.Embedding`, or `transformers.modeling_utils.Conv1D`.

Three research papers describe the method, in lineage order (newest first). When this
repo says "the method", it means **VPD** — or just "PD" generically. It is *not* "SPD":
SPD is the predecessor VPD builds on.

- **VPD** — adVersarial Parameter Decomposition — [`papers/Adversarial_Parameter_Decomposition/post.md`](papers/Adversarial_Parameter_Decomposition/post.md) ("Interpreting Language Model Parameters"). The current method and framing: scales parameter decomposition to full language models and replaces SPD's *stochastic* ablation sampling with *adversarially-chosen* (persistent-PGD) ablations. This is what the repo implements today.
- **SPD** — Stochastic Parameter Decomposition — [`papers/Stochastic_Parameter_Decomposition/spd_paper.md`](papers/Stochastic_Parameter_Decomposition/spd_paper.md). Predecessor; introduced the causal-importance / stochastic-masking framing VPD builds on. Concepts still right, but VPD supersedes it.
- **APD** — Attribution-based Parameter Decomposition — [`papers/Attribution_based_Parameter_Decomposition/apd_paper.md`](papers/Attribution_based_Parameter_Decomposition/apd_paper.md). Original precursor; introduced linear parameter decomposition.

## Package layout

Two flat-layout distributions, deliberately split — **core is a pure trainer library;
lab is composition / IO / CLI / experiment assembly**:

- **`param-decomp`** (root: `param_decomp/` + sibling `pretrain/` + sibling
  `vendored_jax/`) — the core: the generic JAX single-pool VPD trainer ENGINE
  (`param_decomp/`: run.py = `run_decomposition_training`, lm.py, train.py, ci_fn.py,
  targets/llama8b.py, …), the torch-free pydantic config SCHEMA it now carries directly
  (`base_config.py` = `BaseConfig`, `schedule.py`, `configs.py` = `PDConfig` /
  `RuntimeConfig` / `Cadence` / loss + eval-metric configs / routing / ci-fn / wandb
  shaping), the built-run bundle (`built_run.py`: `BuiltRun` / `DataConfig` /
  `EvalConfig` / `RunInstance` / `TargetSites`), the
  `param_decomp.log` logger, the in-house target-LM pretrainer (`pretrain/`), and the
  bit-parity vendored JAX archs (`vendored_jax/`). Carries jax + pydantic as deps. NO
  CLI entrypoints, NO `main()`, NO YAML/experiment reading — the engine takes built
  objects. The repo root IS the uv workspace root.
- **`param-decomp-lab`** (`param_decomp_lab/`) — team tooling AND the composition roots.
  The per-domain in-job entry (`experiments/lm/run.py` / `experiments/{tms,resid_mlp}/run.py`:
  read the run YAML → build the target / data loader / `ExperimentConfig` → call the core
  engine), the YAML→`ExperimentConfig` conversion (`experiments/config.py` shared +
  `experiments/lm/config.py` LM), the experiment YAML schemas, run-loading
  (`experiments/lm/load_run.py`), launchers, post-processing pipelines, infra. Churns
  freely; depends on core. The reverse edge (`param_decomp → param_decomp_lab`) is
  FORBIDDEN — pinned by `param_decomp/tests/test_runtime_standalone.py`.

`make install-dev` syncs both editably via the uv workspace in the root `pyproject.toml`
into the one `.venv`. The root `pyproject.toml` declares NO core console scripts — the
trainers run as modules (`python -m param_decomp_lab.experiments.lm.run` /
`python -m pretrain.train`), not as `pd-*` scripts; the launcher and post-pipeline `pd-*`
scripts live in `param_decomp_lab/pyproject.toml`. (Slow/plot eval is in-loop only — there
is no `pd-slow-eval` CLI.)

## Training (JAX) <a id="training-jax"></a>

**Training is JAX now.** The torch `Trainer` was retired from HEAD (the JAX single-pool
trainer is faster and is what we run; the torch trainer is preserved as the *semantic
oracle* at git tag `torch-oracle`).

The trainer lives in `param_decomp/` (the core of the root `param-decomp` distribution,
the one venv). The semantics source of truth is its `SPEC.md` (normative pseudocode +
numbered invariants, grounded in the torch oracle — JAX **conforms** to it). For the real
entry points, read `param_decomp/CLAUDE.md` and `SPEC.md`. In one breath:

- **`DecomposedModel`** (`param_decomp/lm.py`) — THE model interface: ordered `sites` +
  pure fns (`clean_output` / `site_inputs` / `masked_output` / `weight_deltas`) over
  `(frozen, vu)` pytrees. Generic over vendored LM targets. There is one recon
  semantics: chunkwise masking through the suffix forward, KL on final logits.
- **`run_decomposition_training(...)`** (`param_decomp/run.py`) — the generic ENGINE: the
  one train loop every target runs through (init/restore/finetune/faith-warmup, the
  recon-grid step factory, orbax checkpointing, schedules, metrics, in-loop slow eval,
  SIGTERM-save). A pure library — no `main()`, reads no YAML; takes built objects.
- **`python -m param_decomp_lab.experiments.lm.run <config.yaml>`**
  (`param_decomp_lab/experiments/lm/run.py`) — the LM composition root + only I/O layer:
  reads the canonical schema, builds the target / prefix / data loader / `ExperimentConfig`,
  then calls the engine. Orbax sharded checkpoints; SIGTERM → save → SLURM requeue → resume.
- **Launch from the lab side** via `pd-lm <config.yaml>` (login-node submission wrapper;
  CONFIG-DRIVEN via `runtime.dp`, no `--nodes` / `--local` flags). `dp = N` (multiple of 8)
  → snapshots the tree to an immutable shared-FS workspace, installs the `[cuda]` extra
  there, sbatches `python -m param_decomp_lab.experiments.lm.run` across `N // 8` nodes;
  `dp = null` → runs the trainer inline single-process. `lab → param_decomp` is a fine
  dependency; only
  `param_decomp → lab` is forbidden.

## Public API (consumer substrate)

Alongside the trainer, `param_decomp/` exposes a thin substrate the **consumers**
(harvest / autointerp / clustering / intruder) lean on. Import names from where they're
defined — no package-level re-exports, `__init__.py` files are bare:

```python
from param_decomp.configs import Cadence, LossMetricConfig, PDConfig, RuntimeConfig
from param_decomp.log import logger
from param_decomp_lab.experiments.lm.load_run import open_jax_run, run_metadata
```

- `PDConfig` — algorithm config: seed, CI fn, loss metrics, optimizers, decomposition
  targets. The torch-free pydantic schema now lives in core
  (`param_decomp.configs`, alongside `base_config` / `schedule`); the engine reads the
  derived runtime `ExperimentConfig` (`param_decomp.built_run`). (The eval-metric *config*
  classes likewise live in `param_decomp.configs`; only their torch `Metric` *impls* were
  dropped.)
- `RuntimeConfig` — compute substrate: `autocast_bf16`, `device`, `dp`. Perturbs numerics
  without changing the algorithm.
- `param_decomp.log` — the logger every consumer uses (folded into the core trainer
  package).
- `param_decomp_lab.experiments.lm.load_run.{open_jax_run, run_metadata}` — the JAX
  consumer entry (lab-side, since it builds the LM target): a run opened for a forward pass
  (`open_jax_run`, restores orbax) or just its target topology (`run_metadata`:
  `n_blocks`/`vocab`/per-site `(name, C)` from config + cache, no restore). `JaxPDAdapter`
  keys autointerp/clustering metadata off `run_metadata`.

The torch run-loading surface (`ComponentModel`, the loss `Metric` impls, `RunSink`,
`RunBatch` / `ReconstructionLoss`, `component_model_io`, the vendored archs) was dropped
and returns JAX-native as the #10 torch->jax adapter.

## Where things live

- `param_decomp/` — the JAX trainer core. The pydantic config SCHEMA (`base_config.py` =
  `BaseConfig` / `Probability`; `schedule.py`; `configs.py` = routing +
  decomposition-target + ci-fn + loss + eval-metric configs + `PDConfig` / `RuntimeConfig`
  / `Cadence` / `WandbConfig` / `ResumeProvenance` + the wandb-shaping helpers). The
  built-run bundle the engine consumes (`built_run.py`: `BuiltRun` /
  `DataConfig` / `EvalConfig` / … + the `TargetSites` protocol). The engine + numerics
  (`run.py` = `run_decomposition_training`, `lm.py` / `train.py` / `ci_fn.py` /
  `targets/llama8b.py` / `targets/llama_simple_mlp.py` / `adversary.py` / `recon.py` / `losses.py` /
  `checkpoint.py` / `sharding.py` / `eval.py` / `slow_eval.py` + `log.py`) plus `configs/`
  (the self-contained run yamls) and `tests/` (incl. the `tests/equivalence/` frozen
  torch↔JAX goldens). The torch oracle lives at git tag `torch-oracle`.
- `pretrain/` (repo-root sibling) — the in-house target-LM pretrainer (`pretrain.train`):
  trainable equinox archs whose `state_dict()` keys the decomposition loader reads.
- `vendored_jax/` (repo-root sibling) — bit-parity JAX Llama / GPT-2 archs the trainer
  decomposes.
- `param_decomp_lab/adapters/` — `JaxPDAdapter`: torch-free autointerp/clustering metadata
  for a JAX run, keyed off `experiments.lm.load_run.run_metadata` (config + cache, no
  orbax restore). The torch `build_target` bridge was deleted with the rest of torch.
- `param_decomp_lab/experiments/` — `config.py` (the shared `ExperimentConfig[T, D]` YAML
  schema + the shared YAML→`ExperimentConfig` conversion). `experiments/lm/`: `run.py`
  (the LM composition root — `python -m param_decomp_lab.experiments.lm.run`), `config.py`
  (LM schema + LM build), `load_run.py` (open a finished JAX run), `data.py` /
  `prestage_tokenized.py` (offline tokenize → parquet shards), `jax_launch.py` (`pd-lm`).
  The TMS and ResidualMLP domains live under `experiments/{tms,resid_mlp}/`
  (`run.py` + `config.py` + `model.py`; `pd-tms` / `pd-resid-mlp`), calling the core
  engine as a library.
- `param_decomp_lab/{harvest,autointerp,clustering,investigate}/`
  — post-pipeline stages, each with its own CLAUDE.md.
- `param_decomp_lab/postprocess/` — orchestrates the post-pipeline stages.
- `param_decomp_lab/infra/` — settings, paths, slurm, wandb, sqlite, git, run_files,
  markdown, pydantic helpers.

## Module pointers

| Module | CLAUDE.md | What it covers |
|---|---|---|
| `param_decomp_lab/experiments/` | `param_decomp_lab/experiments/CLAUDE.md` | LM `target.spec` schema, the offline prestage tool, JAX launch |
| `param_decomp_lab/postprocess/` | `param_decomp_lab/postprocess/CLAUDE.md` | Pipeline orchestration: harvest → autointerp / intruder |
| `param_decomp_lab/harvest/` | `param_decomp_lab/harvest/CLAUDE.md` | Component-statistics collection pipeline |
| `param_decomp_lab/autointerp/` | `param_decomp_lab/autointerp/CLAUDE.md` | LLM-based component interpretation |
| `param_decomp_lab/clustering/` | `param_decomp_lab/clustering/CLAUDE.md` | Hierarchical clustering of components |
| `param_decomp_lab/investigate/` | `param_decomp_lab/investigate/CLAUDE.md` | Agent investigation of a research question |

> **The torch web-app (`param_decomp_lab/app/`) was temporarily removed during the JAX
> migration** to shed torch surface for the JAX-primary merge. It is slated for re-add,
> likely as a JAX-native viewer. The reusable tokenizer-display helpers it once owned
> (`AppTokenizer`, `escape_for_display`, `delimit_tokens`) now live in
> `param_decomp_lab/tokenizer_display.py`. See the removal PR for the full re-add log.

## Saved-run layout

Every artifact for a decomposition lives under one dir per run:

```
PARAM_DECOMP_OUT_DIR/runs/<run_id>/
  config.yaml                # the single self-contained run config (the trainer reads it; resume byte-compares)
  ckpts/<step>/...           # orbax sharded checkpoints (JAX trainer)
  metrics.jsonl              # local logs
  harvest/h-*/...            # pd-harvest output
  autointerp/a-*/...         # pd-autointerp output
```

Both training output and the W&B download cache write here. Per-stage subdirs are
populated by their respective pipelines.

`PARAM_DECOMP_OUT_DIR` defaults to `$DATA_MOUNT/artifacts/mechanisms/param-decomp` on
cluster (e.g. `/mnt/data/artifacts/mechanisms/param-decomp` when `DATA_MOUNT=/mnt/data`)
and the relative `out/` off cluster (no `DATA_MOUNT`). Set the `PARAM_DECOMP_OUT_DIR` env
var to override either. Defined in `param_decomp_lab/infra/settings.py`. (A stale shell
may export a wrong value — e.g. an old `/mnt/polished-lake/...` — which overrides the
correct default; check `echo $PARAM_DECOMP_OUT_DIR` if outputs land somewhere unexpected.)

## Development commands

| Command | Purpose |
|---|---|
| `make install-dev` | All workspace packages + dev deps + pre-commit, into the one `.venv` (see [Environment](#environment)) |
| `make install` | Core only |
| `make install-lab` | Core + lab, no dev deps |
| `make check` | basedpyright + ruff lint + format |
| `make type` | basedpyright over the whole workspace (core + config + lab) |
| `make format` | ruff lint + format |
| `make test` | Tests excluding slow |
| `make test-all` | All tests |

Run a single test: `python -m pytest path/to/test_file.py::test_name`.

## CLI entry points

The root `pyproject.toml` declares no core console scripts; the launchers and
post-pipeline scripts live in `param_decomp_lab/pyproject.toml`. The composition roots are
NOT console scripts — run them as modules (the lab launchers sbatch the same module-run
command). LM training is `python -m param_decomp_lab.experiments.lm.run` (JAX), launched
via `pd-lm`. Slow/plot eval is in-loop only (no CLI).

| Command | Entry point | Purpose |
|---|---|---|
| `python -m param_decomp_lab.experiments.lm.run` | `param_decomp_lab/experiments/lm/run.py` | The LM decomposition composition root (reads YAML, builds the target, calls the core engine; run inside a launch workspace) |
| `python -m pretrain.train` | `pretrain/train.py` | The core in-house target-LM pretrainer |
| `pd-lm` | `experiments/lm/launch.py` | Launch a decomposition trainer run; config-driven via `runtime.dp` (`dp=N` → snapshot + workspace + sbatch across `N//8` nodes; `dp=null` → inline) |
| `pd-pretrain` | `experiments/lm/pretrain/launch.py` | Launch a pretrainer run; config-driven via `dp` (`dp=N` → sbatch; `dp=null` → inline) |
| `pd-tms` / `pd-resid-mlp` | `experiments/{tms,resid_mlp}/run.py` | The CPU toy decomposition CLIs |
| `pd-harvest` | `harvest/scripts/run_slurm_cli.py` | Submit harvest SLURM job |
| `pd-autointerp` | `autointerp/scripts/run_slurm_cli.py` | Submit autointerp SLURM job |
| `pd-clustering` / `pd-cluster-merge` / `pd-cluster-distances` | `clustering/scripts/` | Clustering ensemble / merge / consensus distances |
| `pd-postprocess` | `postprocess/cli.py` | Unified postprocessing pipeline |
| `pd-intruder` | `harvest/scripts/run_intruder_slurm_cli.py` | Submit intruder eval job |
| `pd-investigate` | `investigate/scripts/run_slurm_cli.py` | Submit agent-investigation job |

All `pd-*` run commands accept `--group <id>` (wandb group field, used for UI
collapsing) and `--tags a,b,c` (wandb tags). Both no-op when `wandb:` is omitted from
the YAML.

## Cluster usage

- Monitor your jobs: `squeue --format="%.18i %.9P %.15j %.12u %.12T %.10M %.9l %.6D %b %R" --me`

## Files to skip when searching

Use `param_decomp/` or `param_decomp_lab/` as the search root, not the repo root.

Always skip: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `node_modules/`,
`.git/`, `.data/`, `wandb/`, `notebooks/`.

Usually skip unless relevant: `param_decomp/tests/`, `param_decomp_lab/tests/`, `papers/`.

---

# Coding guidelines

This is research code, not production. Prioritize simplicity and fail-fast over
defensive programming.

## Fail fast

- If you have an invariant in your head, **assert it**. Asserting isn't a sign you
  distrust the code — it's the opposite. Codify the trust.
- Don't write `if everything_is_ok: continue_happy_path()`. Just `assert everything_is_ok`.
- Have a VERY good reason to handle an error gracefully. If the program isn't working as
  it should, it shouldn't be running — fix it instead.
- Avoid `try/except` unless it's the right tool. Never use it for control flow.
- Write for the golden path. Don't pre-handle edge cases — raise instead, and handle
  them when they actually bite.

```python
# BAD
def get_config(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

# GOOD
def get_config(path: Path) -> Config:
    assert path.exists(), f"config not found: {path}"
    with open(path) as f:
        return Config(**json.load(f))  # pydantic validates
```

## No legacy support

- Don't add fallbacks for old formats or migration shims. Change it; migrate manually
  if needed.
- Delete unused code. If an argument is always the same value, inline it.

## Types & arguments

- Encode invariants in types. If two fields jointly vary (both present or neither),
  put them in an optional tuple — don't make them independently optional.
- Avoid `| None` unless null is semantically meaningful. Differentiate `None` from `[]`
  when it matters.
- Don't use bare dicts for heterogeneous values. `{<id>: <val>}` good; `{"tokens": ...,
  "loss": ...}` bad — use a dataclass.
- PEP 604 unions (`X | Y`, `X | None`) — not `Union[X, Y]` or `Optional[X]`.
- Lowercase generics (`list`, `dict`, `tuple`) — not `List`, `Dict`, `Tuple`.
- Type-checker: **basedpyright** (not mypy).
- Don't use `from __future__ import annotations`. Use quoted forward references when needed.
- Don't add redundant annotations (`x: int = 5` when `x = 5` infers fine).
- Default arguments are rarely a good idea. Have a very good reason — especially if the
  caller also defaults to the same value. Keep defaults high in the call stack.
- Be explicit about naming, even if names end up long. If a name has to be long to be
  honest about what the thing is, that's fine — if it feels silly, the abstraction is
  probably wrong upstream.

## Control flow

- Keep I/O as high as possible. Make as many functions pure as possible.
- Prefer `match` over `if/elif/else` for dispatching on a tag or kind — more declarative.

## Tensor operations

- Prefer einops for clarity.
- Use jaxtyping for shape annotations (we don't runtime-check, but they document).
- Assert shapes liberally.

## Comments

Comments hide sloppy code. If you feel a comment coming on, consider: better names,
extract a function, extract a named local.

Comments describe what the code is, not what changed about it. No narrativizing:

- `# the function now uses y instead of x` — bad
- `# changed to be faster` — bad
- `# we now traverse in reverse` — bad

## Docstrings

Docstrings carry information the signature doesn't.

- Default to a single line — or none at all, even on public classes / configs /
  functions, when name + type carry everything. `class DistributedState:` doesn't need
  `"""Immutable snapshot of the distributed runtime state for this process."""`.
- Skip `Args:` / `Attributes:` entries that just paraphrase the name and type.
- Don't restate the function name in English. If the name needs translation, fix the name.
- Keep: non-obvious semantics, invariants, gotchas, shape constraints not in jaxtyping,
  side effects, ordering requirements, cross-references.
- No Sphinx/RST markup. Single backticks, not double.
- No `Raises:` for `AssertionError` — asserts are programmer errors, not part of the
  contract.
- Don't re-document a Protocol or abstract method in its impl unless there's
  impl-specific behavior to note.
- Module docstrings: one orienting line. Anything longer belongs in CLAUDE.md.

**Load-bearing public entrypoints in `param_decomp/` are an exception** — there, a full
Google-style `Args:` block is worth the bookkeeping, because IDE hover surfaces it and
the callers are external. Concretely: `ComponentModel.__init__` / `forward` /
`calc_causal_importances`, `RunSink` protocol methods, `Metric.bind` / `update` /
`reset` / `compute`, `make_components`, `make_ci_fn_wrapper`. For everything else,
*including internal helpers in `param_decomp/`*, prefer better parameter names and
clearer parameterisation over docstrings — name parameters by their role inside the
function, not just their type.

## Tests

Tests catch obvious bugs; they're not insurance against production outages — there's
no production. Skip heavy end-to-end tests when they require lots of overhead. The
codebase is run interactively constantly, so the user catches issues cheaply.

## Other

- **Update CLAUDE.md files** when changing code structure, adding/removing files, or
  modifying key interfaces. Update the CLAUDE.md in the same directory (or nearest
  parent) as the changed files.

## GitHub

- Use `gh` for issues and PRs (`gh issue view 28`, `gh pr view 30`).
- PR template: `.github/pull_request_template.md`.
- Before committing: verify you're on the right branch. Don't `git add .` — add
  specific files.
- Branch names: `refactor/X`, `feature/Y`, `fix/Z`.
- **Never** `--no-verify`. Pre-commit hooks exist for a reason. If they fail, fix the
  underlying issue.
