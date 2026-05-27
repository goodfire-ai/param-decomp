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

## Project overview

PD is a research framework for sparse parameter decomposition: target-model parameters are
decomposed into a sum of components; per-datapoint **causal importance (CI)** quantifies
how much each component can be masked; multiple loss terms balance faithfulness,
reconstruction, and sparsity.

Three experimental domains: TMS (Toy Model of Superposition), ResidualMLP, and Language
Models. The LM experiment decomposes any HuggingFace-loadable model whose target modules
are `nn.Linear`, `nn.Embedding`, or `transformers.modeling_utils.Conv1D`.

Two research papers describe the method:

- **SPD** — [`papers/Stochastic_Parameter_Decomposition/spd_paper.md`](papers/Stochastic_Parameter_Decomposition/spd_paper.md). The current framing. Repo has evolved since publication but the concepts are still right.
- **APD** — [`papers/Attribution_based_Parameter_Decomposition/apd_paper.md`](papers/Attribution_based_Parameter_Decomposition/apd_paper.md). Precursor; introduced linear parameter decomposition.

## Package layout

Two flat-layout distributions, deliberately split:

- **`param-decomp`** (`param_decomp/`) — core library. The reusable, publishable surface:
  the optimization loop, configs, `ComponentModel`, loss metrics, the `RunSink` protocol.
  Treat as a stable API; changes here are deliberate.
- **`param-decomp-lab`** (`param_decomp_lab/`) — team tooling. Experiment scripts, the
  post-processing pipelines, the app, infra, eval metrics, lab-side helpers. Churns
  freely; depends on core.

`make install-dev` syncs both editably via the uv workspace in the root `pyproject.toml`.
The `pd-*` console scripts all live in `param_decomp_lab/pyproject.toml`.

## Public API

Import names from where they're defined. No package-level re-exports — `__init__.py`
files are bare. The canonical entrypoint and the protocols / configs it consumes:

```python
from param_decomp.optimize import EvalLoop, optimize
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.run_sink import RunSink
from param_decomp.metrics.base import LossMetricConfig, Metric
from param_decomp.batch_and_loss_fns import RunBatch, ReconstructionLoss
```

- `optimize(target_model, train_loader, run_batch, reconstruction_loss, pd_config,
  runtime_config, sink, cadence, eval_loop=None)` — the only entrypoint. Builds the
  `ComponentModel`, binds eval metrics, runs the loop. Side effects all flow through `sink`.
- `PDConfig` — algorithm: seed, CI fn, loss metrics, optimizers, decomposition targets,
  tied weights, faithfulness warmup. Flipping a field here changes what algorithm runs.
- `RuntimeConfig` — compute substrate: `autocast_bf16`, `device`, `dp`. Perturbs numerics
  without changing the algorithm.
- `Cadence` — train-log / save period predicates. Train-log fires every
  `train_log_every` steps; `save_every` is optional and `should_save` is false at
  step 0. `optimize()` always checkpoints at the final step regardless of `save_every`.
- `EvalLoop` — frozen dataclass in `param_decomp/optimize.py` bundling the eval-loop
  triple (`loader`, `metrics`, `n_steps`) with its timing (`every`, `slow_every`,
  `slow_on_first_step`). Atomic optional: pass `None` to disable eval. `slow_every` must
  be a multiple of `every`.
- `RunSink` — Protocol with three methods (`log`, `console`, `checkpoint`). Concrete
  impl in `param_decomp_lab.run_sink.RunSink` (local files + wandb + rank-aware no-op),
  built via `.local(...)`, `.with_wandb(...)`, or `.silent()`.
- `Metric` — base class with `__init__(cfg)` + `bind(model, device)`. Each config carries
  a `type: Literal["<ClassName>"]` discriminator. See `param_decomp/metrics/CLAUDE.md`
  for the loss-metric wiring (canonical, curated) and
  `param_decomp_lab/eval_metrics/CLAUDE.md` for the eval-metric wiring
  (user-extensible).

## Where things live

- `param_decomp/` — core library (see [Public API](#public-api)). Module docstrings
  describe each file.
- `param_decomp/metrics/` — loss `Metric` classes and dispatch.
- `param_decomp_lab/experiments/{tms,resid_mlp,lm}/run.py` — composition roots; each
  parses a YAML, builds objects, calls `optimize()`.
- `param_decomp_lab/{harvest,autointerp,clustering,dataset_attributions,graph_interp,investigate,app}/`
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
| `param_decomp_lab/postprocess/` | `param_decomp_lab/postprocess/CLAUDE.md` | Pipeline orchestration: harvest → autointerp / attributions / intruder → graph_interp |
| `param_decomp_lab/harvest/` | `param_decomp_lab/harvest/CLAUDE.md` | Component-statistics collection pipeline |
| `param_decomp_lab/autointerp/` | `param_decomp_lab/autointerp/CLAUDE.md` | LLM-based component interpretation |
| `param_decomp_lab/clustering/` | `param_decomp_lab/clustering/CLAUDE.md` | Hierarchical clustering of components |
| `param_decomp_lab/dataset_attributions/` | `param_decomp_lab/dataset_attributions/CLAUDE.md` | Aggregated component-to-component attributions |
| `param_decomp_lab/graph_interp/` | `param_decomp_lab/graph_interp/CLAUDE.md` | Context-aware labelling using the attribution graph |
| `param_decomp_lab/investigate/` | `param_decomp_lab/investigate/CLAUDE.md` | Agent investigation of a research question |
| `param_decomp_lab/app/` | `param_decomp_lab/app/CLAUDE.md` | Web visualization (FastAPI + Svelte) |
| `param_decomp_lab/experiments/lm/pretrain/` | `param_decomp_lab/experiments/lm/pretrain/CLAUDE.md` | LM target-model pretraining |

## Saved-run layout

```
PARAM_DECOMP_OUT_DIR/decompositions/<run_id>/
  model_<step>.pth           # checkpoints (RunSink.checkpoint)
  metrics.jsonl              # local logs (RunSink.log)
```

`PARAM_DECOMP_OUT_DIR` is `/mnt/polished-lake/artifacts/mechanisms/param-decomp/` on
cluster, `~/param_decomp_out/` off cluster. Defined in
`param_decomp_lab/infra/settings.py`.

## Development commands

| Command | Purpose |
|---|---|
| `make install-dev` | Install all workspace packages + dev deps + pre-commit |
| `make install` | Core only |
| `make install-lab` | Core + lab, no dev deps |
| `make check` | basedpyright + ruff lint + format |
| `make type` | basedpyright |
| `make format` | ruff lint + format |
| `make test` | Tests excluding slow |
| `make test-all` | All tests |
| `make app` | Launch the PD app (backend + frontend) |

Run a single test: `python -m pytest path/to/test_file.py::test_name`.

## CLI entry points

All declared in `param_decomp_lab/pyproject.toml`.

| Command | Entry point | Purpose |
|---|---|---|
| `pd-tms` | `experiments/tms/run.py` | Run TMS experiment from a YAML |
| `pd-resid-mlp` | `experiments/resid_mlp/run.py` | Run ResidMLP from a YAML |
| `pd-lm` | `experiments/lm/run.py` | Run LM from a YAML |
| `pd-lm-layerwise` | `experiments/lm/layerwise.py` | Split an LM YAML into per-matrix configs, submit as a SLURM array |
| `pd-pretrain` | `experiments/lm/pretrain/cli.py` | Pretrain target models |
| `pd-harvest` | `harvest/scripts/run_slurm_cli.py` | Submit harvest SLURM job |
| `pd-autointerp` | `autointerp/scripts/run_slurm_cli.py` | Submit autointerp SLURM job |
| `pd-attributions` | `dataset_attributions/scripts/run_slurm_cli.py` | Submit dataset-attribution SLURM job |
| `pd-graph-interp` | `graph_interp/scripts/run_slurm_cli.py` | Submit graph-interp SLURM job |
| `pd-postprocess` | `postprocess/cli.py` | Unified postprocessing pipeline |
| `pd-clustering` | `clustering/scripts/run_pipeline.py` | Clustering ensemble pipeline |
| `pd-cluster-harvest` | `clustering/scripts/run_harvest.py` | Harvest activations → membership snapshot |
| `pd-cluster-merge` | `clustering/scripts/run_merge.py` | Merge from snapshot (CPU only) |
| `pd-intruder` | `harvest/scripts/run_intruder_slurm_cli.py` | Submit intruder eval job |
| `pd-investigate` | `investigate/scripts/run_slurm_cli.py` | Submit agent-investigation job |

All `pd-*` run commands accept `--group <id>` (wandb group field, used for UI
collapsing) and `--tags a,b,c` (wandb tags). Both no-op when `wandb:` is omitted from
the YAML. `pd-lm-layerwise` auto-generates a `lw-...` group id and propagates it (and
any `--tags`) to every child run.

## Cluster usage

- **Do not use more than 8 GPUs at one time** — this includes simultaneous sweeps / evals.
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
the callers are external. Concretely: `optimize`, `ComponentModel.__init__` / `forward`
/ `calc_causal_importances`, `RunSink` protocol methods, `Metric.bind` / `update` /
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
