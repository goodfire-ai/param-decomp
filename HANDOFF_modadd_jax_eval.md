# Handoff — Integrate Antoine's modular-addition experiments as a JAX eval

**Author:** Claude (handoff for Lee, lee@goodfire.ai)
**Date:** 2026-06-30
**Repo:** `goodfire-ai/param-decomp` (GitHub remote `origin`, `git@github.com:...`). On this
cluster it lives at `~/param-decomp`; on the new cluster, clone fresh and `git fetch origin`.

---

## The goal (Lee's plan)

Integrate **some of Antoine Vigouroux's modular-addition / arithmetic experiments** into the
**JAX implementation** (`feature/jax`) **as an eval** for a particular model being decomposed.
Concretely: a new eval that measures how well a decomposition reproduces the target model's
behavior on modular-arithmetic prompts (addition first; subtraction/multiplication follow).

---

## Two branches that matter

### 1. `feature/jax` — the target (where the eval should land)
- Ground-up **JAX rewrite** of param-decomp. The old PyTorch `spd/` package is deleted and
  replaced by a two-distribution layout:
  - `param_decomp` — core JAX library (training engine `run.run_decomposition_training`,
    JAX targets `tms.py`/`resid_mlp.py`).
  - `param_decomp_lab` — experiments, app, postprocessing, the `pd-*` CLI tooling.
  - `nano_param_decomp` — compact reference impl; `vendored_jax/` — JAX gpt2/llama defs.
- Diverged from `dev` at a 2026-04-21 merge-base; 592 ahead / 723 behind `dev` (long-lived,
  likely *replaces* the torch line rather than merging back).
- Install: `make install-dev` (core + lab + dev). Run LM experiments: `pd-lm <yaml> --nodes N`.
- **Eval system (the integration point):**
  - `param_decomp/eval.py` — main eval
  - `param_decomp/slow_eval.py`, `hidden_acts_eval.py`, `attn_patterns_eval.py`
  - lab-side example: `param_decomp_lab/experiments/toy_uv_eval.py`
  - Tests: `param_decomp/tests/test_eval.py`, `test_slow_eval.py`,
    `test_eval_averaging_parity.py` — **mirror these for the new eval.**

### 2. `experiment/8B_targeted` — Antoine's source material
- **110 commits by Antoine** (`Antoine Vigouroux <antvig@protonmail.com>`),
  2026-06-09 → 2026-06-27. HEAD `25cde4237`.
- **Targeted Parameter Decomposition (TPD) of arithmetic on Llama-8B's layer-18 MLP.**
- ⚠️ **Still largely torch-based** (43 torch files vs 24 jax in `param_decomp/`); **no
  `vendored_jax/`**. Forked from the JAX line at `1130c1fe6` (2026-06-11) and is now
  **552 commits behind `feature/jax`** (110 ahead). So porting his eval to `feature/jax`
  means **re-expressing it against the current JAX eval interfaces**, not cherry-picking.
- Related infra branch: `feature/targeted` (the generic "targeted PD" base; `8B_targeted`
  is 90 commits ahead of it, diverged — not a clean descendant).

---

## What Antoine actually built (the reusable pieces)

**Roadmap & progress notes (read these first):**
- `notes/old_roadmaps/roadmap_optimize_arithmetic_tpd.md` — the original weekend roadmap
  (Obj 2: optimize the addition decomposition; Obj 3: reconstruct only the last `=` position;
  Obj 4: subtraction; later: multiplication + add/sub/mult comparisons).
- `notes/roadmap_addition_analysis.md`, `notes/roadmap_addition_analysis_log.md` — analysis
  progress + findings.
- `notes/targeted_implementation/{targeted_implementation_main,targeted_plan,targeted_log}.md`.

**Dataset (simple & portable):** `param_decomp_lab/experiments/lm/prompts_dataset.py`
- Arithmetic "dataset" = a **text file of prompts, one per line**, all tokenizing to the
  **same length**, with the answer at a **fixed position** (padding disabled).
- `StaticBatchLoader` samples rows from that fixed pool (seeded).
- Conceptually trivial to regenerate in JAX; the porting effort is the eval/recon loss, not
  the data.

**Validation tooling** (`param_decomp_lab/scripts/validation/`, see its `CLAUDE.md`,
`commands.md`, `spec.md`):
- `map_arithmetic.py`, `build_arith_ablation_explorer.py` (+ `arith_ablation_explorer_app.html`)
- `measure_model_accuracy.py` / `model_accuracy_notebook.py` — **per-prompt model accuracy**
  (closest thing to "the eval" Lee wants).
- `compute_subcomp_periods.py` (additive vs **logarithmic** period detection),
  `build_subspace_scatter.py` (+ app), `build_neuron_investigator.py` (+ app),
  `find_alive_components.py`, `ablate_component_groups.py`, `collect_ablation_kl.py`.

**Key success metrics Antoine used** (good basis for the eval's reported numbers):
- `RoundedReconLoss` and total **L0**; **PGD recon loss < 0.1**.
- **`n_alive`** — number of components active (CI > 0.1) at least once over the eval batch,
  one value per decomposed matrix. (Antoine added this as a new eval — worth porting too.)
- Baseline run to beat: **`llama8b-add-02`** (standard addition decomposition, layer-18 MLP).

---

## Suggested approach for the new Claude

1. `git fetch origin`; check out `feature/jax` (it's the base). Skim `param_decomp/eval.py`
   and `param_decomp_lab/experiments/toy_uv_eval.py` to learn the **current** eval contract
   (how an eval is registered, what it receives, how it logs to wandb / `metrics.jsonl`).
2. Read Antoine's `notes/roadmap_addition_analysis*.md` on `experiment/8B_targeted` to see
   what worked and what the eval needs to report.
3. Inspect his eval/accuracy logic:
   `git show origin/experiment/8B_targeted:param_decomp_lab/scripts/validation/measure_model_accuracy.py`
   (and `prompts_dataset.py`). Decide what to port vs. rebuild against JAX.
4. Generate the arithmetic prompt set in the JAX world (mirror `prompts_dataset.py` semantics:
   fixed-length prompts, answer at a fixed position; support "last position only" for the
   `=`-token recon variant from Obj 3).
5. Add the new eval next to `param_decomp/eval.py`, wire it into the LM experiment, and add a
   test mirroring `param_decomp/tests/test_eval.py`. Consider also porting **`n_alive`**.
6. Confirm which **specific model** Lee is decomposing — *this is the one open input below.*

---

## Open items / things to confirm with Lee
- **Which model** is being decomposed on the new cluster, and at which layer/site? (Antoine's
  work is Llama-8B layer-18 MLP; Lee's target may differ — the eval should be parameterized.)
- Cluster-specific bits do **not** transfer: SLURM wrappers, `pd-lm --nodes/--dp` submission,
  partition/GPU defaults are tuned to the `reno` cluster. Re-check launch tooling on the new
  cluster before running GPU jobs.

## Workspace state (this cluster — informational, won't transfer)
- `~/param-decomp` currently on local `feature/jax` (tracks `origin/feature/jax`, up to date).
- ~103 GB of **untracked, regenerable** output sits in the tree and was intentionally left
  alone: `spd/` (27 GB of analysis outputs — NOT the package), `.claude/worktrees/` (76 GB of
  `wt`/`wtt` worktrees, dominated by `wandb/` logs), plus `ONBOARDING.md`. None of it is on any
  branch; none blocks anything. 13 stashes exist (unrelated to this work).
