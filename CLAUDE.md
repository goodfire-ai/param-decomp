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

**Two-stack gotcha (torch + JAX in the main venv).** The JAX trainer distribution
(`param_decomp_jax/`) keeps its own venvs (`make install-jax` / `install-jax-cuda`) —
its CUDA wheels conflict with torch's. But `param_decomp_jax` is NOT a uv-workspace
member, so a bare `uv sync --all-packages` strips jax from the main `.venv`. The
JAX-run bridge workers (e.g. `param_decomp_lab/harvest/scripts/run_worker_jax.py`)
live in the lab venv and `import jax` + `from jax_single_pool ...`, and `make type`
over them needs both stacks resolvable. `make install-dev` handles this: it re-adds
jax/jaxlib (CPU) + the editable `param_decomp_jax` source (`--no-deps`) into the main
venv right after the sync. Use `make install-dev`, never a bare `uv sync`.

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

Three flat-layout distributions, deliberately split:

- **`param-decomp-config`** (`param_decomp_config/`) — torch-free pydantic config
  schema. Every config class in the project lives here (`BaseConfig`, `PDConfig`,
  loss/eval-metric configs, experiment YAML schemas). Depends only on pydantic, numpy,
  pyyaml, annotated-types — so non-torch consumers (e.g. the JAX repo) can validate the
  same YAML run configs without pulling torch/transformers/wandb. Keep it that way.
- **`param-decomp`** (`param_decomp/`) — core torch library. Since the torch trainer was
  retired (see [Training](#training-jax)), what survives here is **bridge substrate**:
  `ComponentModel`, the CI fns / sigmoids / masks / decomposition targets, the `RunSink`
  protocol, and the `RunBatch` / `ReconstructionLoss` protocols — the surface the torch
  consumers (harvest / app / eval) reach through `component_model_io.py` to load JAX
  exports. Depends on config. The live torch optimization loop and loss metrics are gone
  from HEAD (preserved at git tag `torch-oracle`).
- **`param-decomp-lab`** (`param_decomp_lab/`) — team tooling. Experiment scripts, the
  post-processing pipelines, the app, infra, eval metrics, lab-side helpers. Churns
  freely; depends on core + config.

`make install-dev` syncs all three editably via the uv workspace in the root
`pyproject.toml`. The `pd-*` console scripts all live in
`param_decomp_lab/pyproject.toml`.

## Training (JAX) <a id="training-jax"></a>

**Training is JAX now.** The torch `Trainer` was retired from HEAD (the JAX single-pool
trainer is faster and is what we run; the torch trainer is preserved as the *semantic
oracle* at git tag `torch-oracle`). The pivot and its scope are documented in
[`param_decomp_jax/jax_single_pool/TRANSITION.md`](param_decomp_jax/jax_single_pool/TRANSITION.md);
that's the settled plan, read it first.

The trainer lives in `param_decomp_jax/jax_single_pool/` (its own distribution, own
venv). The semantics source of truth is its `SPEC.md` (normative pseudocode + numbered
invariants, grounded in the torch oracle — JAX **conforms** to it). For the real entry
points, read `param_decomp_jax/jax_single_pool/CLAUDE.md` and `SPEC.md`. In one breath:

- **`DecomposedModel`** (`jax_single_pool/lm.py`) — THE model interface: ordered `sites` +
  pure fns (`clean_output` / `site_inputs` / `masked_output` / `weight_deltas`) over
  `(frozen, vu)` pytrees. Generic over vendored LM targets. There is one recon
  semantics: chunkwise masking through the suffix forward, KL on final logits.
- **`jsp-train <config.yaml>`** (`jax_single_pool/run.py`) — the composition root and the
  only I/O layer. Reads the canonical `param_decomp_config` schema directly. Orbax
  sharded checkpoints; SIGTERM → save → SLURM requeue → resume.
- **Launch from the lab side** via `pd-jax-lm <wrapper.yaml> --nodes N` (login-node
  submission wrapper; runs in the torch lab venv, snapshots the tree to an immutable
  shared-FS workspace, sbatches). `lab → param_decomp_jax` is a fine dependency; only
  `param_decomp_jax → lab` is forbidden.

## Public API (bridge substrate)

What remains of the torch core after the trainer's retirement is the surface the torch
**consumers** (harvest / app / eval / autointerp / clustering) use to load and read a
saved JAX decomposition. Import names from where they're defined — no package-level
re-exports, `__init__.py` files are bare:

```python
from param_decomp_config.pd import Cadence, PDConfig, RuntimeConfig
from param_decomp.run_sink import RunSink
from param_decomp.metrics.base import Metric
from param_decomp_config.losses import LossMetricConfig
from param_decomp.batch_and_loss_fns import RunBatch, ReconstructionLoss
```

- `PDConfig` — algorithm config: seed, CI fn, loss metrics, optimizers, decomposition
  targets, tied weights. The torch-free schema in `param_decomp_config`; the JAX trainer
  reads it directly.
- `RuntimeConfig` — compute substrate: `autocast_bf16`, `device`, `dp`. Perturbs numerics
  without changing the algorithm.
- `RunBatch` / `ReconstructionLoss` — protocols in `param_decomp/batch_and_loss_fns.py`,
  consumed by the torch consumer bridges (`SavedLMRun` / `SavedTMSRun` reload paths).
- `RunSink` — Protocol with three methods (`log`, `console`, `checkpoint`). Concrete
  impl in `param_decomp_lab.run_sink.RunSink` (local files + wandb + rank-aware no-op),
  built via `.local(...)`, `.with_wandb(...)`, or `.silent()`.
- `Metric` — base class with `__init__(cfg)` + `bind(model, device)`. Each config carries
  a `type: Literal["<ClassName>"]` discriminator. See `param_decomp/metrics/CLAUDE.md`
  for the loss-metric wiring (canonical, curated) and
  `param_decomp_lab/eval_metrics/CLAUDE.md` for the eval-metric wiring
  (user-extensible).

## Where things live

- `param_decomp_config/` — all pydantic configs, one module per domain: `base`
  (`BaseConfig`, `Probability`, `runtime_cast`), `schedule`, `routing`, `ci_fn`,
  `decomposition_target`, `losses`, `pd`, `experiment`, `lm`, `eval_metrics`,
  `autointerp`.
- `param_decomp/` — core library (see [Public API](#public-api)). Module docstrings
  describe each file.
- `param_decomp/metrics/` — loss `Metric` classes and dispatch.
- `param_decomp_lab/experiments/{tms,resid_mlp,lm}/run.py` — consumer bridges: pure
  builders (`build_target` / loader / `make_run_batch`) + the `Saved<Name>Run` reload
  classes that load a saved decomposition off disk. The torch training drivers were
  retired with the torch trainer; training is `jsp-train` (JAX) launched via `pd-jax-lm`.
- `param_decomp_lab/{harvest,autointerp,clustering,investigate,app}/`
  — post-pipeline + app, each with its own CLAUDE.md.
- `param_decomp_lab/postprocess/` — orchestrates the post-pipeline stages.
- `param_decomp_lab/eval_metrics/` — batteries-included eval-metric set.
- `param_decomp_lab/infra/` — settings, paths, slurm, ddp_launch (single-/multi-node
  torchrun wrapper), wandb, sqlite, git, run_files, markdown, pydantic helpers.
- `param_decomp_lab/{seed.py, distributed.py, batch_and_loss_fns.py, component_model_io.py, run_sink.py}`
  — lab-side helpers that aren't big enough to warrant their own subdir.

## Module pointers

| Module | CLAUDE.md | What it covers |
|---|---|---|
| `param_decomp/metrics/` | `param_decomp/metrics/CLAUDE.md` | Loss-metric dispatch, config placement rule, sources vs masks, PPGD |
| `param_decomp_lab/experiments/` | `param_decomp_lab/experiments/CLAUDE.md` | Adding an experiment, YAML schema, LM `target.spec`, `Saved<Name>Run` |
| `param_decomp_lab/eval_metrics/` | `param_decomp_lab/eval_metrics/CLAUDE.md` | Eval-metric dispatch — user-extensible (vs canonical loss metrics) |
| `param_decomp_lab/postprocess/` | `param_decomp_lab/postprocess/CLAUDE.md` | Pipeline orchestration: harvest → autointerp / intruder |
| `param_decomp_lab/harvest/` | `param_decomp_lab/harvest/CLAUDE.md` | Component-statistics collection pipeline |
| `param_decomp_lab/autointerp/` | `param_decomp_lab/autointerp/CLAUDE.md` | LLM-based component interpretation |
| `param_decomp_lab/clustering/` | `param_decomp_lab/clustering/CLAUDE.md` | Hierarchical clustering of components |
| `param_decomp_lab/investigate/` | `param_decomp_lab/investigate/CLAUDE.md` | Agent investigation of a research question |
| `param_decomp_lab/app/` | `param_decomp_lab/app/CLAUDE.md` | Web visualization (FastAPI + Svelte) |
| `param_decomp_lab/experiments/lm/pretrain/` | `param_decomp_lab/experiments/lm/pretrain/CLAUDE.md` | LM target-model pretraining |

## Saved-run layout

Every artifact for a decomposition lives under one dir per run:

```
PARAM_DECOMP_OUT_DIR/runs/<run_id>/
  <wrapper>.yaml             # the stamped run config (jsp-train reads it; resume byte-compares)
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
| `make install-dev` | All workspace packages + dev deps + pre-commit, and bridge jax into the main venv (see [Environment](#environment)) |
| `make install` | Core only |
| `make install-lab` | Core + lab, no dev deps |
| `make install-jax` / `install-jax-cuda` | The JAX trainer distribution's own venv (`param_decomp_jax/.venv`); CUDA wheels for the cluster |
| `make check-jax` / `test-jax` | basedpyright / pytest the JAX distribution (in its own venv) |
| `make check` | basedpyright + ruff lint + format |
| `make type` | basedpyright |
| `make format` | ruff lint + format |
| `make test` | Tests excluding slow |
| `make test-all` | All tests |
| `make app` | Launch the PD app (backend + frontend) |

Run a single test: `python -m pytest path/to/test_file.py::test_name`.

## CLI entry points

All declared in `param_decomp_lab/pyproject.toml`.

The torch training drivers (`pd-tms` / `pd-resid-mlp` / `pd-lm` / `pd-lm-layerwise`)
were retired with the torch trainer; the oracle lives at git tag `torch-oracle`.
Training is now `jsp-train` (JAX), submitted via `pd-jax-lm`.

| Command | Entry point | Purpose |
|---|---|---|
| `pd-jax-lm` | `experiments/lm/jax_launch.py` | Submit a JAX `jsp-train` run: snapshot ref + shared-FS workspace + sbatch |
| `pd-offline-eval` | `experiments/lm/offline_eval.py` | Torch offline eval of a JAX run's exported checkpoint |
| `pd-pretrain` | `experiments/lm/pretrain/cli.py` | Pretrain target models |
| `pd-harvest` | `harvest/scripts/run_slurm_cli.py` | Submit harvest SLURM job |
| `pd-autointerp` | `autointerp/scripts/run_slurm_cli.py` | Submit autointerp SLURM job |
| `pd-postprocess` | `postprocess/cli.py` | Unified postprocessing pipeline |
| `pd-cluster-merge` | `clustering/scripts/run_merge.py` | Merge from a membership snapshot (CPU only) |
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
