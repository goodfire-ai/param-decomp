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
(`uv sync` + pre-commit). jax is a normal dependency of the `param-decomp` distribution
(CPU jax base); GPU hosts add the `[cuda]` / `[cuda13]` extra (`uv sync --extra cuda`).

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

ONE generic library, `param_decomp/`. **The library is just a library — mostly pure
functions, mostly just logic**; nothing infra-ish (schedulers, submission, cluster
paths, code-shipping) lives in it. The library is the deliverable — everything in it is
written to be shippable:

The distribution is `param-decomp`; the package is enumerated layers, each a subpackage,
importing only DOWNWARD (pinned by `param_decomp/core/tests/test_runtime_standalone.py`
— an allowlist; a new subpackage must be enumerated there deliberately):

- `core/` — the generic JAX single-pool VPD trainer ENGINE (`run.py` =
  `run_decomposition_training`, `model.py` = the `DecomposedModel` protocol,
  `train.py`, `ci_fn.py`, …; ZERO concrete target code) plus the torch-free pydantic
  config SCHEMA it reads directly (`base_config.py`, `schedule.py`, `configs.py` =
  `PDConfig` / `Cadence` / loss + eval-metric configs) and the built-run bundle
  (`built_run.py`). No `main()`, no YAML — takes built objects.
- `targets/` — every `DecomposedModel` implementation, one slice per architecture:
  `glu_transformer.py` (+ `llama8b.py`/`qwen3_8b.py` family files),
  `llama_simple_mlp.py`, `transformer_taps.py`, the toys `tms.py`/`resid_mlp.py`;
  per-target parity/golden suites in `targets/tests/`. Imports core only.
- `vendored_jax/` — bit-parity JAX archs (leaf; excluded from typecheck).
- `pretrain/` — the in-house target-LM pretrainer (`python -m
  param_decomp.pretrain.train`).
- The composition/consumer layers (the former lab, merged in): `experiments/` (the
  per-domain composition roots + YAML authoring schemas — `python -m
  param_decomp.experiments.lm.run <config.yaml>` is the LM in-job entry),
  `harvest/`, `autointerp/`, `clustering/`,
  `topology/`, `adapters/`, `migrations/`, `infra/`. Library-level tests in
  `param_decomp/tests/`.

**Datasets are named, never located.** `data.train` / `data.eval` carry
`{kind: name, name: <name>}` and resolve under `<data_root>/datasets/<name>/`. A dataset
dir is self-describing — `meta.json` (`infra.dataset_store.DatasetMeta`: seq_len +
tokenizer) rides with the shards, so the config carries only the identity. Names are
immutable versions: a changed dataset is a new name. A dataset is born by materializing
shards + meta, which `prestage_tokenized` writes together.

`make install-dev` installs the library editably, with dev deps, into the one `.venv`.
The library ships no console scripts — every runnable surface is a module main.
Slow/plot eval is in-loop only: it has no entry point of its own.

## The library rule

`param_decomp` is JUST a library: logic and mostly-pure functions over explicit inputs.
It may do I/O that *is* the work — checkpoints, datasets, HF weights, wandb logging,
LLM calls — but it must not know *where it runs*: no scheduler (SLURM/sbatch), no
submission or code-shipping, no cluster paths, mounts, partitions, or team namespaces.
Deployment fit belongs to whatever launcher invokes the library. A launcher composes
library entrypoints and may import the library; the library may never import a launcher
(enforced fail-closed by `param_decomp/core/tests/test_runtime_standalone.py`). The
library reads NO ambient environment for paths: output roots are explicit, required
parameters threaded from entry points, so a launcher passes its own resolved root as an
argument. Credentials and third-party-tool conventions (`WANDB_*`, `*_API_KEY`, `HF_*`,
`CUDA_*`) may follow their ecosystem's own env contract, resolved at entry points.
Everything else takes typed values. A launcher is not privileged: if it needs
something the library doesn't publicly expose, that is a library bug — never a reason
for a private hook. A SLURM *mention* in library prose is legitimate only when it
documents a generic contract (e.g. SIGTERM→save semantics), never a dependency.

**Configs are portable: names, never locations.** A config references the world only
through names — a dataset name, a run id, an HF hub id, a wandb ref — each with exactly
one resolver, rooted in a global registry (HF hub, W&B) or the explicit `data_root`.
Locations (absolute paths) are representable only inside tagged escape arms
(`data: {kind: dir, ...}`), so non-portability is always loud in the document; names
are immutable versions, so a pin means the same thing on every machine and every year.
CI-enforced for the seats (CONFIGS.md rule 5). Known tracked exception:
`resume_provenance.parent_run_dir` is an absolute path that should be a run id.

## Training (JAX) <a id="training-jax"></a>

**Training is JAX now.** The torch `Trainer` was retired from HEAD (the JAX single-pool
trainer is faster and is what we run; the torch trainer is preserved as the *semantic
oracle* at git tag `torch-oracle`).

The trainer lives in `param_decomp/core/`. The semantics source of truth is its
`SPEC.md` (normative pseudocode + numbered invariants, grounded in the torch oracle —
JAX **conforms** to it). For the real entry points, read `param_decomp/core/CLAUDE.md`
and `param_decomp/core/SPEC.md`. In one breath:

- **`DecomposedModel`** (`param_decomp/core/model.py`) — THE model interface: ordered `sites` +
  pure fns (`clean_output` / `read_activations` / `masked_output` / `weight_deltas`) on an
  `eqx.Module` carrying the frozen target weights as fields (the trainable `vu` is an
  explicit method arg). Generic over vendored LM targets. There is one recon
  semantics: chunkwise masking through the full token-input forward, KL on final logits.
- **`run_decomposition_training(...)`** (`param_decomp/core/run.py`) — the generic ENGINE: the
  one train loop every target runs through (init/restore/finetune/faith-warmup, the
  recon-grid step factory, orbax checkpointing, schedules, metrics, in-loop slow eval,
  SIGTERM-save). A pure library — no `main()`, reads no YAML; takes built objects.
- **`python -m param_decomp.experiments.lm.run <config.yaml>`**
  (`param_decomp/experiments/lm/run.py`) — the LM composition root + only I/O layer:
  reads the canonical schema, builds the target / data loader / `ExperimentConfig`,
  then calls the engine. Orbax sharded checkpoints; SIGTERM → save → SLURM requeue → resume.
- **Topology derives from config; submission is a verb.** `dp <= gpus_per_node` → one
  process over exactly `dp` local devices (asserted at startup); `dp > gpus_per_node` →
  one process per node via `jax.distributed`.
  `python -m param_decomp.experiments.lm.run <config>` runs HERE, in the current
  allocation. Submission is somebody else's verb: an external scheduler owns it and
  invokes this module inside the allocation it created.

## Public API (consumer substrate)

Alongside the trainer, `param_decomp/` exposes a thin substrate the **consumers**
(harvest / autointerp / clustering / intruder) lean on. Import names from where they're
defined — no package-level re-exports; `__init__.py` files are bare except two
deliberate exceptions (`topology/__init__.py` re-exports `path_schema_for_model_type`;
`adapters/__init__.py` defines the adapter constructors):

```python
from param_decomp.core.configs import Cadence, LossMetricConfig, PDConfig
from param_decomp.core.log import logger
from param_decomp.experiments.lm.load_run import open_jax_run, run_metadata
from param_decomp.experiments.lm.runtime import RuntimeConfig
```

- `PDConfig` — algorithm config: seed, CI fn, loss metrics, optimizers, decomposition
  targets. The torch-free pydantic schema now lives in core
  (`param_decomp.core.configs`, alongside `base_config` / `schedule`); the engine reads the
  derived runtime `ExperimentConfig` (`param_decomp.core.built_run`). (The eval-metric *config*
  classes likewise live in `param_decomp.core.configs`; only their torch `Metric` *impls* were
  dropped.)
- `RuntimeConfig` — the LM run's compute substrate (`runtime:`): `dp`,
  `remat_recon_forwards` / `remat_ci_fn`, and `launch_env` (the pre-process env each rank
  runs with — single source of truth, so `config.yaml` captures the run's environment).
  Perturbs numerics without changing the algorithm. Composition-side
  (`param_decomp.experiments.lm.runtime`): core accepts it in no form, and the toys have no
  substrate to author.
- `param_decomp.core.log` — the logger every consumer uses (folded into the core trainer
  package).
- `param_decomp.experiments.lm.load_run.{open_jax_run, run_metadata}` — the JAX
  consumer entry (composition-side, since it builds the LM target): a run opened for a forward pass
  (`open_jax_run`, restores orbax) or just its target topology (`run_metadata`:
  `n_blocks`/`vocab`/per-site `(name, C)` from config + cache, no restore). `JaxPDAdapter`
  keys autointerp/clustering metadata off `run_metadata`.

The torch run-loading surface (`ComponentModel`, the loss `Metric` impls, `RunSink`,
`RunBatch` / `ReconstructionLoss`, `component_model_io`, the vendored archs) was dropped
and returns JAX-native as the #10 torch->jax adapter.

## Where things live

- `param_decomp/core/` — the JAX trainer engine. The pydantic config SCHEMA (`base_config.py` =
  `BaseConfig` / `Probability`; `schedule.py`; `configs.py` = routing +
  the explicit (toy) site spec + loss + eval-metric configs + `PDConfig`
  / `Cadence` / `WandbConfig` / `ResumeProvenance` + the wandb-shaping helpers; the
  authored `decomposition.ci` configs, the tiled LM site specs AND the LM's `runtime:`
  substrate live with their domain schemas, composition-side). The
  built-run bundle the engine consumes (`built_run.py`: `BuiltRun` /
  `DataConfig` / `EvalConfig` / … + the `TargetSites` protocol). The engine + numerics
  (`run.py` = `run_decomposition_training`, `model.py` / `train.py` / `ci_fn.py` /
  `family.py` / `adversary.py` / `recon.py` / `losses.py` /
  `checkpoint.py` / `sharding.py` / `init_placed.py` / `recon_eval.py` / `slow_eval.py` +
  `log.py`) plus `configs/`
  (the self-contained run yamls) and `tests/`. The concrete targets live in the sibling
  subpackage `param_decomp/targets/` (`glu_transformer.py` +
  `{llama8b,qwen3_8b}.py` family files, `llama_simple_mlp.py`, the toys
  `{tms,resid_mlp}.py`; its `tests/` holds the frozen torch↔JAX equivalence goldens and
  the other parity suites). The torch oracle lives at git tag `torch-oracle`.
- `param_decomp/pretrain/` — the in-house target-LM pretrainer (`python -m
  param_decomp.pretrain.train`): trainable equinox archs whose `state_dict()` keys the
  decomposition loader reads.
- `param_decomp/vendored_jax/` — bit-parity JAX Llama / GPT-2 archs the trainer
  decomposes.
- `param_decomp/adapters/` — `JaxPDAdapter`: torch-free autointerp/clustering metadata
  for a JAX run, keyed off `experiments.lm.load_run.run_metadata` (config + cache, no
  orbax restore). The torch `build_target` bridge was deleted with the rest of torch.
- `param_decomp/experiments/` — `config.py` (the shared `ExperimentConfig` YAML
  schema base — each domain subclass binds concrete `target`/`decomposition`/`data` —
  plus the shared YAML→`ExperimentConfig` conversion). `experiments/lm/`: `run.py` (pre-JAX environment bootstrap —
  `python -m param_decomp.experiments.lm.run`), `training.py` (the LM composition root), `config.py`
  (LM schema + LM build), `resolved.py` (resolved LM data/run types), `load_run.py` (open a finished
  JAX run), `eval.py` / `attn_patterns_eval.py` / `arithmetic_eval.py` / `data.py` /
  `hf_http.py`, `data.py` / `prestage_tokenized.py` (offline tokenize → parquet shards).
  The TMS and ResidualMLP domains live under `experiments/{tms,resid_mlp}/`
  (`run.py` + `config.py`; module mains), calling the core engine as a
  library over the toy targets (`param_decomp/targets/{tms,resid_mlp}.py`).
- `param_decomp/{harvest,autointerp,clustering}/`
  — post-pipeline stages, each with its own CLAUDE.md.
- `param_decomp/infra/` — settings, paths, wandb, sqlite, run_files, markdown,
  tokenizer_display, pydantic helpers.

## Module pointers

| Module | CLAUDE.md | What it covers |
|---|---|---|
| `param_decomp/experiments/` | `param_decomp/experiments/CLAUDE.md` | LM `target.spec` schema, the offline prestage tool, JAX launch |
| `param_decomp/harvest/` | `param_decomp/harvest/CLAUDE.md` | Component-statistics collection pipeline |
| `param_decomp/autointerp/` | `param_decomp/autointerp/CLAUDE.md` | LLM-based component interpretation |
| `param_decomp/clustering/` | `param_decomp/clustering/CLAUDE.md` | Hierarchical clustering of components |

> **The torch web-app (`param_decomp/app/`) was temporarily removed during the JAX
> migration** to shed torch surface for the JAX-primary merge. It is slated for re-add,
> likely as a JAX-native viewer. The reusable tokenizer-display helpers it once owned
> (`AppTokenizer`, `escape_for_display`, `delimit_tokens`) now live in
> `param_decomp/infra/tokenizer_display.py`. See the removal PR for the full re-add log.

## Saved-run layout

Every artifact for a decomposition lives under one dir per run:

```
<data_root>/runs/<run_id>/
  launch_config.yaml         # the single self-contained run config (the trainer reads it; resume byte-compares). NOT config.yaml: that basename collides with wandb's reserved run-config file, which wandb.save would symlink onto and clobber
  ckpts/<step>/{decomposition,training}/  # orbax sharded checkpoints (JAX trainer): the trained product vs the trainer-only tail (pre-split default/-item runs need an ad-hoc migration, no in-code compat)
  metrics.jsonl              # local logs
  harvest/h-*/...            # harvest output
  autointerp/a-*/...         # autointerp output
```

Both training output and the W&B download cache write here. Per-stage subdirs are
populated by their respective pipelines.

`data_root` is the ONE root of the library's local world — runs (outputs), the dataset
store, the pretrain and compilation caches all hang under it. It is an explicit,
required parameter, never an env var and never defaulted: every entry edge (composition
roots, worker mains, consumer functions like `open_jax_run`) refuses to run without it,
so a deployment cannot silently write into a cwd-relative directory. A launcher passes
its resolved root explicitly — the `--data_root` / `--data-root` flag, or the
pretrainer's stamped `data_root`.

## Development commands

**Use `make test` for the default inner loop.** It uses testmon to rerun only tests affected
by the current changes and excludes `slow`; do not run a bare full-workspace `pytest` while
iterating. Use `make test-all` for exhaustive validation.

| Command | Purpose |
|---|---|
| `make install-dev` | Library + dev deps + pre-commit, into the one `.venv` (see [Environment](#environment)) |
| `make install` | Library only, no dev deps |
| `make check` | basedpyright + ruff lint + format |
| `make type` | basedpyright over `param_decomp/` |
| `make format` | ruff lint + format |
| `make test` | Tests excluding slow |
| `make test-all` | All tests |

Run a single test: `python -m pytest path/to/test_file.py::test_name`.

## CLI entry points

The library ships no console scripts: every runnable surface is a module main, run
inside whatever allocation your scheduler gave you. LM training is
`python -m param_decomp.experiments.lm.run` (JAX). Slow/plot eval is in-loop only (no
entry point).

| Command | Entry point | Purpose |
|---|---|---|
| `python -m param_decomp.experiments.lm.run` | `param_decomp/experiments/lm/run.py` | The LM decomposition composition root (reads YAML, builds the target, calls the core engine) |
| `python -m param_decomp.pretrain.train` | `param_decomp/pretrain/train.py` | The core in-house target-LM pretrainer |
| `python -m param_decomp.experiments.{tms,resid_mlp}.run` | `experiments/{tms,resid_mlp}/run.py` | The CPU toy decomposition composition roots |
| `python -m param_decomp.harvest.scripts.{run_worker,run_merge,run_intruder}` | `harvest/scripts/` | Component-statistics harvest: per-rank worker, merge, intruder eval |
| `python -m param_decomp.autointerp.scripts.run_interpret` | `autointerp/scripts/` | LLM interpretation of harvested components |
| `python -m param_decomp.clustering.scripts.{run_worker,run_merge,calc_distances}` | `clustering/scripts/` | Clustering harvest / merge / consensus distances |

The toy run commands (`experiments.{tms,resid_mlp}.run`) accept `--group <id>` (wandb
group field, used for UI collapsing) and `--tags a,b,c`. Both no-op when `wandb:` is
omitted from the YAML. The LM and pretrain roots take their wandb group and tags from
the config's `wandb:` block instead.

## Files to skip when searching

Use `param_decomp/` as the search root, not the repo root.

Always skip: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `node_modules/`,
`.git/`, `.data/`, `wandb/`, `notebooks/`.

Usually skip unless relevant: the `tests/` subtrees (`param_decomp/core/tests/`,
`param_decomp/targets/tests/`, `param_decomp/tests/`), `papers/`.

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
- **"Unused" is judged against the right caller set.** The two rules above apply to
  internal code, whose callers are enumerable in-repo. Config schema fields, CLI flags,
  and other user-facing surface have users the tree deliberately does not contain:
  uncommitted sweeps and stored runs' pinned `launch_config.yaml`s (see CONFIGS.md — the
  repo's job is more than running committed configs). "No committed config sets it" is
  not evidence for removal; before deleting external surface, check the canonical seats,
  stored-run pins, and intent. The fix for an untested capability flag is a test, not
  deletion.

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

**Comments point inwards, not outwards.** A comment carries a constraint, invariant, or
gotcha the code itself can't show. It must not narrate transient outward state:
campaign measurements tied to a particular config ("~5x at the production plan"),
strategy attributions ("per Oli: no TP"), PR war stories, current-canon claims. That
content lives in lore, the PR thread, or an in-repo design doc (which MAY carry
history) — with at most a short pointer from the code. SPEC invariant-ID citations
(`S14`, `D4`, …) are sanctioned and required as before.

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
the callers are external. Concretely: `run_decomposition_training`, the
`DecomposedModel` protocol methods, `open_jax_run` / `run_metadata`. For everything else,
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
