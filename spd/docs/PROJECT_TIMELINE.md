# SPD Project Timeline

> _Stochastic Parameter Decomposition — Goodfire fork, June 2025 – May 2026._
>
> Compiled from the `main` branch git history (1,039 commits, 17 distinct author identities).
> All dates are UTC. Durations are computed from the first to the last commit touching the
> subsystem on `main`.

---

## Executive summary

The SPD repository was bootstrapped on **2025-06-11** by importing the precursor APD codebase from Apollo Research and has been actively developed by Goodfire ever since — **1,039 commits over ~47 weeks**, currently sitting at ~110K lines of code (Python + frontend) across 597 tracked files.

The project moved through six clear phases:

1. **Bootstrap (Jun–Jul 2025)** — APD code imported; CLI, registry, and the open-source contributor surface (CONTRIBUTING, STYLE, papers) put in place.
2. **Capability buildout (Aug–Oct 2025)** — Data parallelism, GPT-2 support, streaming datasets, hooks-based architecture, `gate → ci_fn` rename, losses+evals consolidation. **`v1` tag** cut on 2025-09-04 to mark the paper-aligned release.
3. **Visualization + multi-node (Nov 2025)** — The interactive interp **app** debuts; **multi-node training** unlocks larger LM experiments.
4. **Offline-analysis pipeline (Dec 2025 – Jan 2026)** — Harvest, autointerp, clustering, dataset attributions, and post-processing all land within ~5 weeks; the **`Claude SPD1` AI agent** begins contributing on 2026-01-26.
5. **Persistent PGD era (Feb 2026)** — The project's peak month at **274 commits**: PPGD productionized, pile_llama target models, unified `spd-postprocess` CLI, pretrain migrated in-tree, DDP correctness fixes.
6. **Mechanistic analysis + paper/blog prep (Mar–May 2026)** — `graph_interp`, `investigate`, and `editing` modules added in one day; multi-provider autointerp; a concentrated **attention/QK/VO geometry analysis** push; package renamed to "Stochastic Parameter Decomposition" for paper alignment; final Protocol-based loss refactor.

**Headline patterns leadership should note:**
- **Single-person resilience:** Dan Braun has been present every month for 12 months and accounts for 35.5% of commits. The project has a strong central architect.
- **High-throughput AI assistance:** the `Claude SPD1` agent has contributed 347 commits in 5 months (33.4% of project lifetime commits). In Feb 2026 it produced more commits in one month (157) than any human in any single month of the project. These are human-supervised commits — they reflect the team's ability to drive an AI coding agent productively, not autonomous activity.
- **Concentrated specialist bursts:** Lee Sharkey landed 8 attention-analysis PRs (#447–#454) across two consecutive days (2026-03-17/18). Lucius Bushnaq's 8 lifetime commits are all high-leverage loss-function changes. The project benefits from short, focused interventions by domain experts.
- **Velocity tracks new capabilities, not maintenance:** commit volume scaled from ~30/month (Aug–Nov 2025) to a peak of 274/month (Feb 2026) and tapered to 49/month (Apr 2026) — consistent with the buildout → analysis → polish lifecycle of a research codebase.

---

## At a glance

| Metric | Value |
|---|---|
| Project age (main branch) | **2025-06-11 → 2026-05-04** (~10 months, 3 weeks) |
| Commits on `main` (non-trivial branch) | **1,039** |
| Commits across all branches | **6,714** |
| Distinct author identities | **17** (≈12 unique humans + 1 AI agent + bots) |
| Files tracked | **597** (≈61.7K Python LOC + ≈46.5K frontend LOC) |
| Top contributor | Dan Braun, **369 commits** (35.5%) |
| First-ever commit | `f5b649f8` — Dan Braun, 2025-06-11, LICENSE only |
| Origin of the codebase | Imported `2025-06-11` from `ApolloResearch/apd@feature/paper-rename` |
| Companion paper | [arXiv:2506.20790 — Stochastic Parameter Decomposition](https://arxiv.org/abs/2506.20790) (paper snapshot lives on the `spd-paper` branch) |
| Tagged releases | one tag — `v1` at `c6314c9f` (2025-09-04) — README polish for paper-aligned release |

---

## 1. Chronological timeline (by month)

### 2025-06 (Bootstrap) — 65 commits
**Theme: import APD, set up the experiment framework, get the lights on.**

| Date | Event | Author |
|---|---|---|
| **2025-06-11** | `f5b649f8` Initial commit (LICENSE only). | Dan Braun |
| **2025-06-11** | `a1cf5a03` **Import code from `ApolloResearch/apd@feature/paper-rename`** — 56 files, ~8.9K LOC. This is the genesis of the working codebase: `run_spd.py`, `configs.py`, `losses.py`, `models/`, the TMS / ResidMLP / LM experiments, the test suite. | Dan Braun |
| **2025-06-19** | `cb5d74a3` Create centralized **experiment registry** (`spd/registry.py`). | Dan Braun |
| **2025-06-25** | First external PRs land (#5, #6 — torch bump, high-mem init fix). | Oli Clive-Griffin |

### 2025-07 (Open-source onboarding) — 62 commits
**Theme: papers added, CLI consolidated, induction-head experiment, contributor onboarding.**

- **2025-07-09** `ad2eb876` (#35) `spd-evals` + `spd-sweep` merged into single `spd-run` CLI.
- **2025-07-10** `b387b239` (#38) Sweep params: flatten nested `parameters` keys.
- **2025-07-11** `7b81b253` (#39) **APD + SPD papers added** to `papers/` in markdown.
- **2025-07-14** `4f00ba5e` Add `CONTRIBUTING.md` — repo is now formally open to PRs.
- **2025-07-15** `4ce552a9` (#47) Add `STYLE.md` for code style.
- **2025-07-22** `54d2125b` (#71) Metric + figure parameterization (foundation of `spd/figures.py`).
- **2025-07-23** `94d45463` (#73) Configurable `AliveComponentsTracker`.
- **2025-07-28** `87b35ca7` (#41) **Induction-head experiment** lands (`spd/experiments/ih/`).
- **2025-07-31** `a9e559af` (#94) Refactor model loading across the codebase.

### 2025-08 (Data parallelism + GPT-2) — 31 commits
**Theme: scale-out capability and second model family.**

- **2025-08-08** `ab53a105` (#102) **Data parallelism** support — first multi-GPU training capability.
- **2025-08-13** `d7e9b32a` (#106) **p-annealing for adaptive L_p sparsity loss** (nathu-goodfire).
- **2025-08-14** `0033ea8e` (#109) **GPT-2 support** (Casper Lützhøft Christensen) — the project gains a second target-model family beyond Llama.
- **2025-08-14** `011d2533` (#107) Fix `spd-run` CLI (mivanit).
- **2025-08-14** `24bc5c0a` Configs updated for GPT-2.
- **2025-08-15** `d8da1256` (#113) Support streaming and tokenized datasets with and without DDP.
- **2025-08-15** `12ae8839` Avoid OOMs with default LM configs at dp=1.

### 2025-09 (Refactors + new loss machinery) — 34 commits

- **2025-09-03** `b29b5b19` (#124) Fix bias-handling in component model + GPT-2 implementation update (Casper).
- **2025-09-04** `c6314c9f` — **`v1` tag** (README polish to align with paper release).
- **2025-09-05** `30527040` (#136) Support custom GPT-2 models from `simple_stories_train`.
- **2025-09-08** `54177615` (#142) Option to use **binomial sampling** (Lucius Bushnaq).
- **2025-09-08** `662d1210` (#140) Remove `init_from_target_weight`; init U,V in `__init__` (Lucius Bushnaq).
- **2025-09-10** `be623758` (#146) `SubsetReconstructionLoss` evaluation metric (nathu-goodfire).
- **2025-09-11** `efea2e81` (#141) Allow inserting **identity matrices before any module**.
- **2025-09-18** `03129623` (#148) **Layerwise global CI function**.
- **2025-09-22** `e46da337` (#166) Remove unused loss functions + improve naming.
- **2025-09-23** `52b39547` (#165) **Refactor to use hooks** instead of `ComponentsOrModule` — major architectural simplification.
- **2025-09-23** `eaddaf22` (#154) Geometric similarity comparison between two trained models (Lee Sharkey).
- **2025-09-24** `49108628` (#168) **Routing / Subset reconstruction loss** (Oli Clive-Griffin).

### 2025-10 (Loss/eval consolidation) — 42 commits

- **2025-10-03** `f931fb25` (#162) **Consolidate losses and evals** — birth of `spd/metrics/`.
- **2025-10-03** `761bf0fb` (#175) **Rename `gate` → `ci_fn`** across the codebase — major vocabulary change still visible today.
- **2025-10-07** `9a04b3d2` (#182) Handle list of discriminated unions in sweep.
- **2025-10-24** `d2c50907` (#230) `ss_llama_simple` replaces `ss_llama`.
- **2025-10-28** `9d500ad8` (#231) **New Interp App** lands at top-level `app/` (Oli Clive-Griffin). Moved to `spd/app/` a week later on 2025-11-07 (#238).

### 2025-11 (App + Multi-node) — 29 commits

- **2025-11-06** `d378a0f2` (#245) **Move to new cluster** — infra migration.
- **2025-11-07** `2ad491b5` (#238) **App moved into `spd/app/`** — Svelte frontend + FastAPI backend now lives under the package.
- **2025-11-24** `298a108b` (#271) Backend reachability warnings.
- **2025-11-27** `9475050c` (#264) **Multi-node training** lands.
- **2025-11-27** `9facd69c` (#279) App performance pass.

### 2025-12 (Harvest + Autointerp + Clustering) — 100 commits
**Theme: the offline-analysis pipeline is built out — first big jump in monthly velocity.**

- **2025-12-04** `ad628384` (#285) Local attribution graphs in the app.
- **2025-12-19** `2a13d8bf` (#315) **Autointerp pipeline + app refactor**.
- **2025-12-22** `1a517f93` (#301) **Birth of `spd/harvest/` and `spd/autointerp/`** — many app features and harvest/autointerp scripts.
- **2025-12-22** `33daca01` Rename `spd-interpret` → `spd-autointerp`.
- **2025-12-23** `b9884af1` (#198) **Clustering subsystem** lands (`spd/clustering/`) — squash of long-running PR #43 (mivanit; clustering work itself had been in flight since earlier in the autumn on a feature branch).
- **2025-12-31** `f8cc9968` (#318) Cosine half-period + `ScheduleConfig`.

### 2026-01 (CI functions, PGD prelude, AI agent debut) — 212 commits
**Theme: largest month of contributor diversity; Claude SPD1 begins contributing.**

- **2026-01-07** `9465f977` (#324) Subcomponent activation display.
- **2026-01-14** `82a669d5` Refactor CI config to discriminated union.
- **2026-01-14** `34afa064` (#338) Log2-based importance minimality loss (Lucius Bushnaq).
- **2026-01-16** `f4f8ac46` (#339) `ss_llama_simple_mlp-2L-wide` experiment config.
- **2026-01-20** `2624852b` (#317) **App updates** — first `spd/dataset_attributions/` code lands here.
- **2026-01-20** `50bee82f` (#346) Improve post-processing.
- **2026-01-20** `1a5b5ed9` Sweep configs for **global reverse residual** experiments.
- **2026-01-23** `1b616aed` (#348) **Global Reverse Residual CI Function with Attention Support** (Lee Sharkey).
- **2026-01-26** `5fb58d61` **First-ever commit by `Claude SPD1`** ("Use autointerp labels for nodes"). Same day: `4ee20424` (#349) "Optimize loss at a specific position in the app" — first merged PR from the agent (authored under `claude-spd1`).
- **2026-01-29** `450af3c1` **First `ContinuousPGD` commit** — adversarial-mask experiment kicked off.
- **2026-01-30** `0760c7c3` ContinuousPGD configs for TMS + ResidMLP.

### 2026-02 (Persistent PGD era) — 274 commits — **PEAK MONTH**
**Theme: PGD scaffolding hardens into PPGD, pile-llama experiments, postprocess unification, pretrain migration.**

- **2026-02-02** `2eda1217` **Rename `ContinuousPGD` → `PersistentPGD`** + simplify mask shape.
- **2026-02-02** `43ffc26a` `PersistentPGDReconSubsetLoss` for subset routing.
- **2026-02-04** Heavy PPGD scope/config work (5+ commits in a single day).
- **2026-02-06** `24c480a5` (#357) **Migrate `simple_stories_train` into `spd/pretrain/`**.
- **2026-02-06** `f9083a3d` (#358) `batch_invariant` scope for PPGD.
- **2026-02-07** `3687c163` (#359) **Autocast to bf16** — up to 25% SPD speedup.
- **2026-02-08** `dad6b5ba` Unified `spd-postprocess` CLI for SLURM dependency-chained postprocessing.
- **2026-02-09** `3f917fbe` Centralize all SLURM job scheduling in postprocess pipeline.
- **2026-02-09** `b71c7b20` (#371) Setup tokenized pile dataset.
- **2026-02-09** `f23352e7` (#374) **Add `pile_llama_simple_mlp` 2L/4L configs** — production target models.
- **2026-02-09** `f2b42167` (#368) PPGD: scope renames, sources terminology, sigmoid param, optimizer abstraction.
- **2026-02-10** `e0b8c10d` (#373) **Generalize app to arbitrary transformer models** (`AppTokenizer`).
- **2026-02-10** `0a86d1ac` Move postprocess into `spd/scripts/postprocess/` (later promoted to `spd/postprocess/`).
- **2026-02-12** Intruder eval decoupled from harvest, promoted to top-level postprocess stage.
- **2026-02-13** `4aef8b8f` (#389) Fix PPGD state source shape for non-sequence (MSE) inputs.
- **2026-02-15** `14465583` (#390) **Fix PGD source DDP semantics** + distributed tests.
- **2026-02-15** `811857eb` (#391) `dist.broadcast` replaces pickle-based PGD source sync.
- **2026-02-16** `1da22a1a` (#388) DDP importance-minimality log-term fix (Lucius Bushnaq).
- **2026-02-16** `f94ceecd` (#383) Adversarial PGD loss in app CI optimization.
- **2026-02-17** `3859e55e` (#396) AVG instead of SUM for PPGD source gradient all-reduce.
- **2026-02-18** `39de8fc3` (#394) Separate `launch_id` from `run_id`; consolidate experiment setup.
- **2026-02-18** `5a8b598f` (#398) Generalize harvest data model + adapters over decomposition methods.
- **2026-02-18** `6dd71624` (#400) Simplify autointerp LLM API + generalize evals.
- **2026-02-18** `78d35772` (#401) **Dual-view autointerp** strategy.
- **2026-02-23** `4c0a470e` (#412) Diverge global RNG across DDP ranks.
- **2026-02-25** `ac5dc77b` (#402) **`StochasticAttentionPatternsReconLoss` metric** (Antoine Vigouroux).
- **2026-02-25** `e4b2d286` (#409) CI and PGD variants of hidden-acts recon loss.

### 2026-03 (Graph interp, investigate, autointerp variants) — 136 commits

- **2026-03-06** `b6ce5cbd` (#421) **Merge 1/3: Data pipelines refactor**.
- **2026-03-06** `af880b5e` (#427) **Add `spd/graph_interp/` module** — context-aware component labeling.
- **2026-03-06** `3d02e8b4` (#428) **Add `spd/investigate/` module** — Claude Code-driven investigation agent.
- **2026-03-06** `29a9ad83` (#429) Add editing module.
- **2026-03-06** `285e2e2d` (#430) Improve postprocess module.
- **2026-03-12** `e9485298` (#438) Enable PPGD late during training (Antoine Vigouroux).
- **2026-03-13** `b7f866a3` / `70b0e277` (#442/#443) Autointerp prompt context + metric explanations.
- **2026-03-17/18** Lee Sharkey burst across two days: PRs #447–#454 — **sweep summary stats**, **fixed-source (r) sweep**, **QK attention contributions**, **prev-token-head detection**, **attention ablation experiments**, **component head norms**, **attention offset profile**, **W_V subspace overlap**.
- **2026-03-17** `a0ef0aee` (#419) **Transcoder harvest integration**.
- **2026-03-17** `d426b733` (#446) **Multi-provider LLM support for autointerp** (Google AI provider added 03-20).
- **2026-03-18** `a67e43c6` (#455) **`rich_examples` autointerp strategy + compare tab**.
- **2026-03-20** `b2a6a152` **`canon` autointerp strategy** with detailed SPD prompt.
- **2026-03-23** `5702e117` Add clustering runs.
- **2026-03-27** `bf435634` **First blog data export scripts** using `AppTokenizer`.
- **2026-03-28** `4dcdd895` Activation-based co-occurrence harvest + GIS comparison plots.
- **2026-03-30** `6d0f2005` (#463) **Clustering core**: compressed memberships, harvest-merge split.

### 2026-04 (Analysis push) — 49 commits

- **2026-04-06** PRs #468/#469/#470 — PPGD warmup mask, `n_samples` for PPGD, clustering distances toggle.
- **2026-04-08** `47167983` **Interactive QK contribution heatmap viewer**.
- **2026-04-09** `9f167366` Consolidated blog export scripts (`scripts/blog/`).
- **2026-04-10** `7979979d` **W_OV overlap analysis** with read/write/raw FCS, QK-filtered data weighting, semantic analysis.
- **2026-04-10** `43d0d117` Rename **"Attention Contribution" → "Static Interaction Strength"** in QK plots.
- **2026-04-21** `85ccd3cf` **Rename package to "Stochastic Parameter Decomposition"** (final paper-aligned naming).
- **2026-04-23** `44f6c3e5` (#474) Add `RunBatch` and `ReconstructionLoss` Protocols.
- **2026-04-23** `c0952a62` `scripts/spd_lm_minimal.py` — minimal LM-decomposition script.
- **2026-04-26** `30f0501f` Intruder-score comparison data export.
- **2026-04-29** `3c23aa74` KV co-activation plotting script.
- **2026-04-30** `36246817` / `995b77b8` / `c024d703` Interactive QK viewer polish: CI tick markers, `build_targeted_prompts` script, var rename + README.
- **2026-04-30** `dcfe27b2` **Geometric interaction analysis scripts** package.

### 2026-05 (Polish) — 5 commits (so far)

- **2026-05-01** `5bc4e051` Restructure `plot_component_activations` into a package.
- **2026-05-01** `7d519298` VO combined component head-norms plot.
- **2026-05-04** `fd482eb5` Lower `MIN_MEAN_CI` in `plot_component_head_norms` to 0.001 — **HEAD**.

---

## 2. Thematic timeline (initiatives and durations)

Where “Duration” is calendar time from first → last commit on the subsystem. Where “Active” is given, it’s the months in which ≥5 commits touched the subsystem (a heuristic for sustained activity vs. drive-by maintenance).

### A. Core SPD optimization (`spd/run_spd.py`, `losses.py`, `models/`)
- **Span:** entire project (2025-06-11 → 2026-05-04, **~47 weeks**).
- **Inflection points:**
  - 2025-09-23 — hooks-based refactor (`#165`).
  - 2025-10-03 — losses + evals consolidation into `spd/metrics/` (`#162`).
  - 2025-10-03 — `gate` → `ci_fn` rename (`#175`).
  - 2026-01-23 — Global Reverse Residual CI fn lands (`#348`).
  - 2026-04-23 — `RunBatch`/`ReconstructionLoss` Protocols (`#474`) — final architectural cleanup.

### B. Experiment surfaces (TMS, ResidMLP, IH, LM)
- **TMS / ResidMLP** — alive since day 1 (imported from APD).
- **Induction Heads (`spd/experiments/ih/`)** — born 2025-07-28; last touched 2026-04-23. **Duration ≈ 9 months**, mostly maintenance after initial PR.
- **LM experiments (`spd/experiments/lm/`)** — alive since day 1; last touched 2026-04-23. **Duration ≈ 47 weeks**, the most active experiment surface. Progression: SimpleStories Llama → SimpleStories GPT-2 → Pile Llama (2L/4L/12L).
- **Pile Llama configs** — first appear **2026-02-09** (`#374`); this is when the project shifts from "toy" LMs to "production" target models.

### C. Persistent PGD (PPGD)
- **Inception:** 2026-01-29 (`ContinuousPGD` first commit).
- **Rename:** 2026-02-02 → `PersistentPGD`.
- **Module file `spd/persistent_pgd.py` born:** 2026-02-02.
- **Heaviest activity:** 2026-02-02 → 2026-02-25 (≈3.5 weeks of near-daily commits).
- **Long-tail polish:** through 2026-04-06 (PPGD warmup, `n_samples`).
- **Total active span:** **~10 weeks**, ~50+ commits with PGD in the subject.
- **Cross-cutting fixes touched:** DDP source sync (pickle → `dist.broadcast`), all-reduce semantics (AVG vs SUM), source shape for MSE inputs, late-enable option, scope abstraction.

### D. Web app (`spd/app/`)
- **First commit:** 2025-10-28 (`#231`, top-level `app/`); moved into `spd/app/` 2025-11-07 (`#238`).
- **Last commit:** 2026-04-23.
- **Duration:** **~26 weeks**.
- **Commits touching the app:** 116 backend + 137 frontend (some overlap).
- **Frontend size today:** ~46.5K LOC across 3,868 files (Svelte + TS).
- **Inflection points:**
  - 2025-12 — local attribution graphs (`#285`).
  - 2026-01-20 — major app updates (`#317`), dataset_attributions appears here.
  - 2026-02-10 — generalize app to arbitrary transformer models (`#373`).
  - 2026-03-06 — app overhaul + clustering integration + cleanup (`c3e6c8e3`).

### E. Harvest / Autointerp / Clustering / Attributions / Graph-interp / Investigate / Editing
| Subsystem | Born | Last commit | Span | Notes |
|---|---|---|---|---|
| `spd/harvest/` | 2025-12-22 | 2026-04-26 | **~18 weeks** | Birthed alongside autointerp in PR #301. SQLite-on-NFS pain mid-Feb 2026. |
| `spd/autointerp/` | 2025-12-22 | 2026-03-29 | **~14 weeks** | Multi-provider LLM support 2026-03-17; rich_examples, canon, dual-view strategies follow. |
| `spd/clustering/` | 2025-12-23 | 2026-04-23 | **~17 weeks** | Compressed memberships + harvest-merge split (PR #463) in late March. |
| `spd/dataset_attributions/` | 2026-01-20 | 2026-04-23 | **~13 weeks** | Born inside PR #317; later promoted to its own module. |
| `spd/pretrain/` | 2026-02-06 | 2026-02-27 | **~3 weeks** active | One-time migration of `simple_stories_train` into the repo. |
| `spd/postprocess/` | 2026-02-10 | 2026-03-07 | **~4 weeks** active | Unification of harvest/autointerp/attributions into one CLI pipeline. |
| `spd/graph_interp/` | 2026-03-06 | 2026-03-29 | **~3 weeks** | Context-aware labeling on top of attributions + correlations. |
| `spd/investigate/` | 2026-03-06 | 2026-03-07 | **~1 day** | Single-PR feature: a Claude Code-driven research agent runner. |

### F. CI functions (gate / `ci_fn`) — naming and architectural evolution
- **2025-10-03** — first formal `gate` → `ci_fn` rename across the codebase (`#175`).
- **2026-01-14** to **2026-01-23** — Lee Sharkey arc: CI config refactored to discriminated union; Global CI fn; Global Reverse Residual CI with attention support (`#348`).
- **2026-04-23** — `GlobalReverseResidualCiFn` deleted (`f869a6d5`) as the architecture moved to `RunBatch`/`ReconstructionLoss` Protocols.

### G. Distributed training (DDP / SLURM / multi-node)
- **2025-11-27** Multi-node training lands (`#264`).
- **2026-01-29** Multi-node SPD fixes (`#352`).
- **2026-02-15** PGD source DDP semantics fix + distributed tests (`#390`).
- **2026-02-15** `dist.broadcast` replaces pickle for PGD source sync (`#391`).
- **2026-02-23** Diverge global RNG across DDP ranks (`#412`).
- **Active window:** ~14 weeks (late Nov 2025 – late Feb 2026), tail of small DDP fixes after.

### H. Postprocessing / SLURM dependency graph
- **2026-02-08** `dad6b5ba` Unified `spd-postprocess` CLI debuts.
- **2026-02-09** Centralize all SLURM job scheduling in postprocess (`3f917fbe`).
- **2026-02-09** Make autointerp a functional unit that owns its eval jobs.
- **2026-02-12** Intruder eval decoupled from harvest, promoted to top-level postprocess stage.
- **2026-02-18** `subrun_id` → `manifest_id` rename + harvest data model generalization.
- **Settled:** by 2026-03-06 the pipeline structure (`harvest → autointerp / attributions → graph_interp`) is essentially today’s shape.

### I. Attention / QK / VO geometry analysis
- **2026-03-17/18** Major Lee Sharkey burst across two days: PRs #447–#454 — sweep summary stats, fixed-source-r sweep, QK attention contributions, prev-token-head detection, attention ablation suite, component head norms, attention-offset profile, W_V subspace overlap.
- **2026-04-08–04-30** Claude SPD1 + Lee follow-up: interactive QK contribution heatmap viewer, per-prompt attention heatmaps, W_OV overlap analysis, KV co-activation plots, VO combined component head norms.
- **Active window:** ~7 weeks of sustained attention-geometry tooling — the most concentrated analysis push in the project.

### J. Blog / external communication push
- **2026-03-27** First blog export scripts using `AppTokenizer`.
- **2026-04-09** Consolidated blog export scripts in `scripts/blog/`.
- **2026-04-10** Component binning + dataset attributions to export; 50 examples per export.
- **2026-04-21** Package renamed to "Stochastic Parameter Decomposition" (paper-aligned).

---

## 3. Contributors

### Headline numbers (commits to `main`, all-time)
> Aliases collapsed where evident from email/identity.

| Rank | Contributor | Commits | Share | Active months |
|---:|---|---:|---:|---|
| 1 | **Dan Braun** | 369 | 35.5% | All 12 (Jun-25 → May-26) |
| 2 | **Claude SPD1** (AI agent) | 347 | 33.4% | Jan-26 → May-26 (5 months) |
| 3 | **Oli Clive-Griffin** (Oli / Oliver / oli) | 213 | 20.5% | Jun-25 → Apr-26 (intermittent until Nov; heavy from Dec) |
| 4 | **Lee Sharkey** (Lee / lee-goodfire) | 75 | 7.2% | Jun-25 → Apr-26 (concentrated bursts) |
| 5 | mivanit | 8 | 0.8% | Jul, Aug, Oct, Dec 2025 |
| 6 | Lucius Bushnaq | 8 | 0.8% | Jul, Sep, Nov 2025; Jan, Feb 2026 |
| 7 | Casper Lützhøft Christensen | 5 | 0.5% | Jul, Aug, Sep 2025 |
| 8 | nathu-goodfire | 4 | 0.4% | Jul, Aug, Sep 2025 |
| 9 | dependabot[bot] | 3 | 0.3% | Nov 2025 |
| 10 | Antoine Vigouroux | 3 | 0.3% | Feb, Mar 2026 |
| 11 | Scott Brenner | 2 | 0.2% | Nov 2025 |
| 12 | kabir jamadar | 1 | 0.1% | Oct 2025 |
| 13 | silico-goodfire-andromeda[bot] | 1 | 0.1% | Apr 2026 |

### Per-contributor narrative

**Dan Braun (lead author, 369 commits)** — present every single month; produced the initial APD import, the `spd-run` CLI, the experiment registry, the multi-node training, the pretrain migration, the bulk of the PPGD work, the package rename, and the final Protocol-based refactor. Has been the consistent thread holding the project's architecture together. Peak months: Feb 2026 (55) and Dec 2025 (42).

**Claude SPD1 (AI agent, 347 commits — `Claude SPD1` + `claude-spd1` identities)** — first commit `5fb58d61` on 2026-01-26 21:35 UTC ("Use autointerp labels for nodes"). In Feb 2026 alone, contributed 157 commits — more than any human author has produced in any single month of the project. Predominantly worked on: app UI (token pills, color modes, navigation, log viewers), PPGD scaffolding (early `ContinuousPGD` configs, scope refactors), autointerp evals, postprocess pipeline, analysis scripts (LaTeX tables, sweep summaries, interactive QK viewer, KV co-activation, component head norms). The two identities are the same agent (single email `claude_spd1@proton.me`): `Claude SPD1` for direct local commits, `claude-spd1` for merged PRs.

**Oli Clive-Griffin (213 commits)** — present from week 2; mostly platform/infra work. Owned: the new interp app (PR #238, Nov 2025), the autointerp pipeline overhaul (PR #315, Dec 2025), the harvest data-model generalization (#398), the unified `spd-postprocess` CLI (Feb 2026), the graph_interp / investigate / editing modules (March 2026), and the blog export push (March–April 2026). Peak month: Mar 2026 (79 commits).

**Lee Sharkey (75 commits)** — three notable arcs:
1. **Jul 2025:** added the APD + SPD papers and contributed to early docs.
2. **Jan 2026 (39 commits):** introduced the Global Reverse Residual CI Function with attention support (`#348`), refactored the CI config to a discriminated union, added the component-activation scatter plot (#343).
3. **Mar 2026 (8 commits across 2 days, 03-17/18):** the attention-analysis suite — sweep summary stats (#447), fixed-source-r sweep (#448), QK contributions (#449), prev-token-head detection (#450), attention ablation experiment suite (#451), component head norms (#452), attention offset profile (#453), W_V subspace overlap (#454).

**Lucius Bushnaq (8 commits)** — small but high-leverage: terms to reduce high-frequency components in ImpMin loss (#335), log2-based importance minimality loss (#338), DDP importance-minimality log-term fix (#388).

**Antoine Vigouroux (3 commits)** — `StochasticAttentionPatternsReconLoss` metric (#402), `dataset_seed` option (#416), late-PPGD-enable option (#438).

**Casper Lützhøft Christensen (5 commits)** — GPT-2 bias-handling fix (#124), induction-head pos-encoding fix (#88), and the induction-head experiment itself (#41).

**mivanit, nathu-goodfire, Scott Brenner, kabir jamadar** — occasional small contributions (typo fixes, single PRs).

---

## 4. Project velocity over time

### Commits-per-month on `main`

```
2025-06  ██████████████████████████████████████████████████████████████████  65
2025-07  ███████████████████████████████████████████████████████████████     62
2025-08  ███████████████████████████                                          31
2025-09  ██████████████████████████████                                       34
2025-10  ██████████████████████████████████████                               42
2025-11  ██████████████████████████                                           29
2025-12  ████████████████████████████████████████████████████████████████████ 100
2026-01  ███████████████████████████████████████████████████████████████████████████████████████████████████████████ 212
2026-02  ████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 274  ★
2026-03  ████████████████████████████████████████████████████████████████████████████████████████ 136
2026-04  ████████████████████████████████                                     49
2026-05  ██                                                                    5  (partial)
```

### Notable patterns
- **3-month ramp** (Aug–Oct 2025) of stabilization at ~30–40 commits/month after the open-source onboarding burst.
- **2.7× jump** in Dec 2025 (29 → 100) when harvest + autointerp + clustering all came online in the same week.
- **2.1× jump** in Jan 2026 (100 → 212) coinciding with Claude SPD1's onset.
- **Peak**: Feb 2026 at 274 commits — driven by simultaneous PPGD, postprocess, app generalization, and pretrain-migration workstreams.
- **Wind-down** to 49 commits in Apr 2026 as the focus shifted from feature work to analysis and blog/paper artifacts.

---

## 5. Notable architectural inflection points

These are the commits that re-shape what the codebase is, not just what it does.

| Date | Commit | Change | Why it matters |
|---|---|---|---|
| 2025-06-11 | `a1cf5a03` | Import APD code | Defined the starting surface. |
| 2025-06-19 | `cb5d74a3` | Create `registry.py` | Single source of truth for experiments. |
| 2025-09-23 | `52b39547` (#165) | Hooks-based refactor | Removed `ComponentsOrModule` abstraction. |
| 2025-10-03 | `f931fb25` (#162) | Consolidate losses + evals | Birth of `spd/metrics/`. |
| 2025-10-03 | `761bf0fb` (#175) | `gate` → `ci_fn` rename | Vocabulary alignment with paper. |
| 2025-11-07 | `2ad491b5` (#238) | First `spd/app/` commit | Interactive visualization becomes first-class. |
| 2025-11-27 | `9475050c` (#264) | Multi-node training | Project can now train large LMs across nodes. |
| 2025-12-22 | `1a517f93` (#301) | Birth of harvest + autointerp | Offline-analysis pipeline is born. |
| 2025-12-23 | `b9884af1` (#198) | Clustering subsystem | Component grouping becomes a first-class tool. |
| 2026-01-20 | `2624852b` (#317) | Dataset attributions | Cross-component attribution analysis enabled. |
| 2026-01-23 | `1b616aed` (#348) | Global Reverse Residual CI fn | New CI architecture for attention models. |
| 2026-02-02 | `2eda1217` | ContinuousPGD → PersistentPGD | Adversarial-mask scheme stabilizes. |
| 2026-02-06 | `24c480a5` (#357) | `simple_stories_train` → `spd/pretrain/` | Target-model training in-tree. |
| 2026-02-08 | `dad6b5ba` | Unified `spd-postprocess` CLI | Single pipeline entry point. |
| 2026-02-09 | `f23352e7` (#374) | `pile_llama_simple_mlp` configs | Production target models. |
| 2026-02-10 | `e0b8c10d` (#373) | App generalized to arbitrary transformers | App decoupled from any specific model family. |
| 2026-03-06 | `af880b5e` / `3d02e8b4` / `29a9ad83` | graph_interp + investigate + editing | Three new modules in one day. |
| 2026-03-17/18 | PRs #447–#454 | Attention-analysis suite (Lee Sharkey, 8 PRs / 2 days) | The mechanistic-interp toolkit moves from probes to QK/VO geometry. |
| 2026-04-21 | `85ccd3cf` | Package rename to "Stochastic Parameter Decomposition" | Paper-aligned external naming finalized. |
| 2026-04-23 | `44f6c3e5` (#474) | `RunBatch`/`ReconstructionLoss` Protocols | Final architectural simplification of the loss layer. |

---

## 6. Open questions / caveats

These are things this document **does not** answer from the git history alone — flag them if leadership asks:

- **Why** the project rebooted from APD on 2025-06-11 (the prior history is on `ApolloResearch/apd`).
- The full **Goodfire ↔ Apollo Research collaboration** picture lives in chat / wandb / org repos, not here.
- The **`spd-paper` branch** is referenced in README but not analyzed here; the arXiv paper (2506.20790) was already public when this repo opened.
- The 6,714 all-branch commits include experiment branches and Claude-Code worktrees that never landed on `main`; this doc only audits `main`.
- The **AI agent (Claude SPD1)** contribution count is real but reflects human-supervised commits — they correspond to prompts/reviews by Dan, Lee, and Oli rather than autonomous activity. Per-PR review history is not analyzed here.
- This timeline does not yet correlate commits with **WandB runs** or **experiment outcomes**; doing so would require pulling from `SPD_OUT_DIR` and the `goodfire/spd-*` WandB projects.

---

_Generated from `git log` on the `dev` branch (HEAD `fd482eb5`, 2026-05-04). Re-run the analysis after major merges to refresh._
