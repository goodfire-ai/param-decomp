# Research Track: Speeding Up the Core PD Method

**Status:** plan / proposal
**Owner:** Dan
**Audience:** the AI coding agents that will execute this track, plus the human reviewing them.

> **Normative contract = [`track2/README.md`](track2/README.md)** — the concise rules the agent
> obeys each iteration (objective, gates, thresholds, workflow). This plan is the *rationale +
> execution detail* behind it. If the two ever disagree, README wins and should be reconciled.

---

## 0. TL;DR

We run two parallel tracks. **Track 1 (scaling)** applies today's method to llama-3.1-8b and
beyond. **Track 2 (this plan, "method speedups")** asks a different question: *can we change the
core algorithm so the loop runs much faster for similar quality, before we pay to scale it?*

Track 2 is small-model only. We iterate on two cheap models and a result that works on
`pile_llama_simple_mlp-4L` is a finished result — we do **not** validate on 12L or 8b in this track.

So the plan is mostly about three things:

1. A **two-tier model ladder** (`ss-2L` for fast iteration, `pile-4L` for confirmation) so we move
   fast without fooling ourselves.
2. A **measurement contract** that pins down "similar loss" and "X times faster" *before* an agent
   touches code — so agents can't accidentally win by moving the goalposts.
3. An **autonomous agent workflow** (worktrees, specs, background jobs, a shared ledger) tuned
   around the known failure modes of coding agents, with the human in the loop only via **concise
   status reports**, not approvals.

Scope guardrails up front:
- **Base training only.** Everything is `pd-lm` with plain **DDP**. **No 3-pool / multipool** — that
  is Track 1's memory-ceiling machinery and is explicitly out of scope here.
- **Core is fully fair game.** `param_decomp/` is our library; agents may rewrite it freely. There
  is no "stable API" constraint in this track. Reproducibility of the baseline comes from the
  *recorded baseline run artifacts* (§3.1) plus worktree isolation, not from keeping an old code
  path around.

The actual experiment ideas are a *backlog* (§9), deliberately the smallest part of this document.

---

## 1. Goals & what counts as a win

A Track-2 result is a change to the core method (`param_decomp/`) that is one of:

- **Faithful speedup** — same/better quality at materially lower wall-clock or GPU cost
  (e.g. "PPGD warmup in bf16 → ~X% faster step, recon/CE/L0 within noise").
- **Faithful simplification** — fewer moving parts / less compute for similar quality
  (e.g. "hidden-state MSE replaces logit-KL in warmup with no measurable quality loss", or
  "an approximate component forward matches the exact one within tolerance").

**"Done" = it holds on `pile_llama_simple_mlp-4L` under the measurement contract (§3).** No higher
tier is required.

### What we're actually looking for (the objective)

A speedup is only a win if it preserves decomposition *quality*. Concretely, in priority order:

- **Primary — what a change must hold (or improve):**
  - **PPGD recon loss** — measured at eval via `PGDReconLoss` (a fresh PGD attack on the masked
    reconstruction). We use the *eval* metric, not the train-time `PersistentPGDReconLoss`, precisely
    because many of our speedups change the training-time PPGD loop — the eval attack stays a fair,
    procedure-independent yardstick.
  - **L0** (`eval/l0/0.0_total`) — sparsity / number of active components.
- **Secondary — weighted less:** stochastic-mask recon (`StochasticHiddenActsReconLoss`) and
  CI-mask recon (`CIHiddenActsReconLoss`).
- **Gate (not a soft guardrail) — faithfulness under the CI mask** (`ce_difference_ci_masked`,
  `kl_ci_masked`, `ce_unrecovered_ci_masked`). This is held to a **near-equality constraint**: a
  change that regresses faithfulness past the band **fails, regardless of how good PPGD/L0 look**.
  (Otherwise a "win" can just be the optimizer trading faithfulness for lower L0 — a worse
  decomposition that scores better.)

So a result is: **≥10% faster** (wall-clock/step or GPU-h — our floor for "material"), with
**faithfulness held within band**, PPGD-recon and L0 within band or better, and secondary not
meaningfully worse. The decision rule is faithfulness-gated then primary-driven, encoded in
`param_decomp_lab/speedup/quality_bundle.py` + `pd-speedup-compare` (which prints an overall verdict).

What we can change (no assumption about which dominates — **measure, don't guess**, §9): the **PPGD
inner loop**, the **CI function** (`global_shared_transformer`), and the **target-model forward**
(frozen, so its activations are cacheable). The benchmark profiler tells us where the time actually
goes.

Non-goals: 3-pool / model-parallel / 8b memory work (Track 1); any change whose only justification
is "it should help at 8b but we can't see it on small models."

---

## 2. The two-tier model ladder

The bet is that wins on small models transfer to the 4L paper model. We cap the ladder at two tiers
to keep iteration fast — the user's explicit priority over deeper-tier assurance.

| Tier | Model (config) | Role | Measured cost | Used for |
|---|---|---|---|---|
| **T0** | `ss_llama_simple_mlp-2L` (SimpleStories, tiny) | **Fast iteration** | ~0.29 s/step, 28 GB peak (1×H100, b64/s512); ~32 GPU-h @ 400k (≈8h at dp4) | Correctness *and* fast quality/speed iteration. Where ideas live day-to-day. |
| **T1** | `pile_llama_simple_mlp-4L` (the paper model, real Pile) | **Confirmation / the bar** | ~25.8 GPU-h @ 400k (from Jose's locked run `s-55ea3f9b`) | Anything promising on T0 is confirmed here. A win here is a finished result. |

> Both checkpoints are already pretrained/cached — Track 2 only *decomposes*, never pretrains.
> Costs above are **measured** (2026-06-08): T0 via `pd-speedup-bench`; T1 from the wall-clock of
> Jose's baseline run (which is **not re-run** by us — we read its stored W&B metrics, §3.1).
>
> **Experiments run a reduced budget, not the full curve.** ~**50k steps is enough signal** at both
> sizes (incl. Jose-size 4L); judge there (compare against the baseline's *cached trajectory at the
> same step*, §3.3). Only a *promising* result graduates to a longer run for confirmation. So the
> per-experiment T0/T1 *experiment* config is the **50k baseline config**
> (`ss_llama_simple_mlp-2L-baseline.yaml` for T0), not the 400k canonical — experiments must use the
> same config as the baseline they're compared against.

### How the two tiers divide labor

- **Iterate everything on T0**, including quality. T0 is cheap enough to be the default workbench:
  run smoke, measure speed, read the quality bundle, kill obviously-bad ideas here.
- **Confirm anything promising on T1.** T0 is SimpleStories and tiny, so a T0 quality win is a
  *strong hypothesis*, not the result. Promotion to T1 is the agent's call (no human gate, §8) once
  T0 looks good; the **T1 number is the headline** in every report.
- **Speedups are scale-dependent**, so report them at *both* tiers — a T0 speedup can shrink or grow
  at T1. Never carry a T0 speedup number up as if it were the T1 result.

**Operating rule:** iterate on T0; **confirm on T1**; the T1 result is what we claim.

---

## 3. The measurement contract (the anti-reward-hacking spine)

Coding agents are very good at making a number move the right way — by quietly breaking correctness,
by changing the eval, or by concluding from one short single-seed run. The defense is to fix the
scoring rules *before* any agent edits code. Note: agents may rewrite the core algorithm freely, but
they may **not** edit the eval harness or baseline pointers as part of an experiment — those are the
ruler, not the thing being measured.

### 3.1 Locked baselines
- **T0: 3 seeds** of the reduced baseline config (`ss_llama_simple_mlp-2L-baseline.yaml`, 50k/b32).
  The 3 seeds give the **noise floor** — the per-metric band is the observed seed spread (this is
  how we judge "within band" without guessing). `pd-speedup-compare` takes the three run ids
  comma-separated and derives the band.
- **T1: Jose's locked run `s-55ea3f9b`** (single seed; we can't re-run it). Verified to use the
  *current* config/schema, so it's a clean bar; its full trajectory is cached for `--at_step`.
- Agents compare against these exact artifacts, never a baseline they re-derive. Re-lock only when a
  human deliberately rebases (keep the old one).

> **Status (2026-06-08).** **T1 LOCKED** → `s-55ea3f9b` (cached at `runs/s-55ea3f9b/`).
> **T0 RUNNING (3 seeds)** of `ss_llama_simple_mlp-2L-baseline.yaml` (50k/b32): seed-0 `p-db5adc3b`,
> seed-1 `p-89a42376`, seed-2 `p-993ae14a`. On completion, record all three + the per-metric band
> (seed spread) in `track2/ledger/baselines.md`; then T0 is locked. Until then, experiments can
> smoke/benchmark but not be quality-judged on T0.
>
> Quality-bundle values (tiered; T0 = mean of 3 seeds, _fill on completion_):
>
> | tier | metric | T0 (3-seed) | T1 (`s-55ea3f9b`) |
> |---|---|---|---|
> | primary | PGD recon (PPGD) | _TBD_ | 0.65225 |
> | primary | L0 total | _TBD_ | 201.45 |
> | secondary | Stochastic hidden-acts recon | _TBD_ | 0.41470 |
> | secondary | CI-masked hidden-acts recon | _TBD_ | 0.85740 |
> | gate | CE diff (CI-masked) | _TBD_ | 0.28603 |
> | gate | KL (CI-masked) | _TBD_ | 0.34315 |
> | gate | CE unrecovered (CI-masked) | _TBD_ | 0.0042753 |

### 3.2 Frozen eval + metric bundle (agents don't edit these)
The **whole `eval:` block is part of the ruler**, not just the metric classes: the eval metrics,
their hyperparameters (crucially the **PGD-attack strength** — `PGDReconLoss` `n_steps`/`step_size`),
the eval batch size, and the cadence. An experiment runs the *same* `eval:` block as the baseline;
weakening the attack or changing the eval batch would make the comparison meaningless (and is a
reward-hack). `pd-speedup-compare` checks eval-config parity and flags a mismatch.

- **Speed metric** (the thing we trade): wall-clock per step + tokens/sec on a **pinned GPU type**,
  fixed batch/seq, plus a torch.profiler op breakdown — produced by `pd-speedup-bench` on the base
  DDP `pd-lm` path. (No `pd/*` `record_function` labels on `main`; the breakdown is aten-op level.)
- **Quality bundle (the objective)** — the eval metrics as a *tiered vector*, defined in
  `quality_bundle.py` (§1): **primary** = PPGD recon (`PGDReconLoss`) + `l0_total`; **secondary** =
  `StochasticHiddenActsReconLoss` + `CIHiddenActsReconLoss`; **gate** = CI-masked faithfulness
  (`ce_difference_ci_masked`, `kl_ci_masked`, `ce_unrecovered_ci_masked`). Plus visual guardrails
  (`ComponentActivationDensity`, `ci_mean_per_component`) eyeballed in wandb. `pd-speedup-compare`
  reports the whole vector and an **overall verdict** (faithfulness-gated, then primary-driven).

### 3.3 Statistics, not vibes
- **Judge early — don't wait for full training.** ~**50k steps is enough signal** at both sizes
  (incl. Jose-size 4L). Run experiments to ~50k and compare against the baseline's *cached
  trajectory at the same step* (`pd-speedup-compare --at_step N`, `N` on a slow-eval step). Per-step
  *speed* needs only a short run. **A promising result then graduates to a longer run** to confirm
  the win holds later in training — required before *merge* (§7), since early ranking can cross over.
- **The band is the measured noise floor.** The **T0 baseline is 3 seeds**; the per-metric band is
  the observed seed spread (floored at `--tol_pct`, default ±2%). `pd-speedup-compare` takes the
  three seed run-ids comma-separated and derives the band — this is how "within band" is defined,
  not a guess. (T1's baseline is single-seed, so T1 uses the ±`tol_pct` floor only; treat T1 deltas
  near that floor as noise and lean on the T0 band.)
- **Experiments may be single-seed** for speed — they're judged against the 3-seed *baseline* band.
  Re-run a borderline experiment at multiple seeds only when its Δ sits near the band edge.
- The decision is **faithfulness-gated, then primary-driven** (§1), reported as a Pareto point
  (speed vs the quality vector) — not a single scalar.

### 3.4 Make gaming structurally hard
- Eval harness, **the experiment's `eval:` config block** (metrics + PGD-attack strength + eval
  batch + cadence, §3.2), baseline pointers, and this contract are **read-only for an experiment**;
  changing them is a separate, explicitly-flagged change the human can see. `pd-speedup-compare`
  flags eval-config drift between variant and baseline.
- Every result cites **artifacts** (`run_id`, `metrics.jsonl` path, wandb URL). No artifact → it
  didn't happen. (Blocks hallucinated results.)
- Keep fail-fast asserts on (`isfinite`, shape checks): a speedup that trips an assert is a failure,
  not a finding.

---

## 4. The iteration harness (built — status in §10)

All on the **base DDP `pd-lm`** path, in `param_decomp_lab/speedup/` + the smoke/baseline configs:

1. **Smoke configs** per tier (`*-smoke.yaml`): tiny `steps`, short seq, no wandb, fast eval. One
   command, finishes in minutes, exercises the full code path. First thing every agent runs after
   editing.
2. **`pd-speedup-bench`**: config → step time, tokens/sec, peak memory, profiler breakdown — pinned
   GPU, fixed batch/seq, eval excluded. Produces the §3.2 speed metric reproducibly.
3. **`pd-speedup-compare`**: baseline run-id(s) + variant(s) → tiered quality-bundle diff + overall
   verdict (reads `metrics.jsonl`; `--at_step` for early judging; takes the 3 T0 seeds for the band).
4. **Baseline lock** (§3.1): T0 = 3 seeds (spread defines "noise"); T1 = `s-55ea3f9b` (single).

---

## 5. Autonomous agent workflow (Claude Code)

Agents run unattended under **bypass permissions** (§11.4). Design for isolation,
reproducibility, and shared memory — the three things multi-agent setups lose. This section is the
*principles*; **§11 is the concrete execution** (how to actually start and run the loop).

### 5.1 One experiment = one worktree = one branch = one spec
- Each experiment runs in its **own git worktree** (branched from `feature/track2-setup`, §11.0) so
  parallel agents can't clobber each other and dead ends are deleted, not untangled. Repo rule:
  `uv sync --all-packages` in the worktree, never `cd` back to main.
- Branch `feature/spd-<short-idea>` (off `feature/track2-setup`). Edit core `param_decomp/` freely —
  no flag-gating
  required; the committed baseline artifacts (§3.1) are what make comparison reproducible, not an
  in-tree fallback path.
- Each experiment is a short **spec file** from a template: hypothesis, claim type, config/code diff
  vs baseline, T0+T1 plan, success/kill thresholds quoted from §3, artifact links. The spec is the
  unit in the ledger (§7).

### 5.2 Long runs are background jobs
- Submit via SLURM/`pd-lm` (default partition; no custom CPU/mem; `--qos=opportunistic` or
  `scavenge` for speculative / over-quota work). **GPU budget (2026-06-08): a single run may use up
  to 16 GPUs**, and keep **≤16 GPUs in flight** across all simultaneous Track-2 experiments. (>8
  GPUs/run is multi-node — `--dp N`, `N` a multiple of 8. This supersedes the global ≤8 note in
  `CLAUDE.md` for Track-2 runs.)
- Launch as **background work** and let the harness re-invoke on completion rather than blocking. For
  the SLURM queue (which the harness can't watch), poll with a sane cadence
  (`ScheduleWakeup`/`Monitor`) — not a tight loop.
- Each agent monitors *only its own* jobs (`squeue --me`); never cancel others' jobs.

### 5.3 Claude Code surfaces, used sparingly
- **Subagents (Explore/Plan)** for fan-out reading so the main agent's context stays on the change.
- **`/code-review`** on the diff before merging.
- **Plan mode** is *optional* (handy for large core rewrites like an approximate forward), not
  required.
- **Memory** for durable cross-session facts; the **ledger** (§7) for experiment state.
- Permission posture: **bypass** (§11.4). Keep CLAUDE.md current as configs change. Do
  **not** add MCP servers or heavier orchestration without a concrete need.

---

## 6. Common AI-agent failure modes & mitigations

| Failure mode | What it looks like here | Mitigation |
|---|---|---|
| **Reward hacking the speed metric** | Step faster because correctness silently broke; or trade faithfulness for lower L0 | Quality bundle is a mandatory *vector* with a **faithfulness gate** (§1/§3.2); asserts stay on; equivalence tests for approximations |
| **Moving the goalposts** | Agent edits eval / weakens the PGD attack / changes eval batch so "loss is similar" | Eval **config block** + baseline read-only (§3.4); `pd-speedup-compare` flags eval-config drift |
| **Premature conclusion** | Promote off noise on one early run | **3-seed baseline band = measured noise floor** (§3.3); judge at a matched step; **longer confirmation run required before merge** (§7) |
| **Overfitting to the tiny model** | "5x faster, same loss" on 2L only | Confirm on T1; report speedup at both tiers, never carry T0 up |
| **Hallucinated results** | A report with numbers but no run | Artifacts required or it didn't happen (§3.4) |
| **Bogus speedup** | Wall-clock "win" that's GPU/noise/eval overhead | Pinned GPU, fixed batch/seq, profiler breakdown, eval excluded (§4) |
| **Scope creep / collisions** | Parallel agents clobber configs / wander | Worktree-per-experiment; `/code-review` before merge (§5.1) |
| **Context loss on long tasks** | Forgets the contract mid-session | Self-contained spec + ledger; contract lives in-repo, re-read each session |
| **Silent numerical drift** | bf16/approx NaNs late in training | Fail-fast `isfinite` asserts; smoke first; watch curves, not just the final number |
| **Zombie GPU jobs** | Crashed run holds GPUs | Verify `squeue --me` after a crash (known error-path issue) |

---

## 7. Experiment lifecycle & promotion

`proposed → T0 (iterate) → T1 (confirm) → merged / killed / parked`

No human gate before T1 — agents promote T0→T1 themselves once T0 looks good. The human reads the
result afterward (§8). Every gate needs **≥10% speedup** (per-step or GPU-h) — below that it's not
worth the complexity. Promotion criteria:

- **→ T0:** smoke passes, unit + equivalence tests green, no NaNs.
- **T0 → T1:** at ~50k steps, `pd-speedup-compare` overall verdict is WIN or NEUTRAL vs the **3-seed
  T0 band** (faithfulness gate held, primary within-or-better) **and** ≥10% measured speedup on
  `ss-2L`. Single experiment seed is fine (judged against the 3-seed baseline band).
- **T1 → merged:** same verdict vs `s-55ea3f9b` at ~50k, ≥10% speedup — **then a longer
  confirmation run** (further into training) showing the win holds. This guards against early/late
  crossover. This is a finished result; flip the change to default.

Killing is first-class: a clean "tried X, here's the run, it regressed Y" with artifacts is a good
outcome and goes in the ledger so nobody re-runs it.

### The ledger (shared memory + the report)
A directory of per-experiment spec/result markdown files (`track2/ledger/experiments/`) plus **one
top-level index table** (`track2/ledger/README.md`) — this is both the cross-agent memory and the
thing the human reads. Cheap, greppable, no extra infra. (We considered moving this to a `lore`
knowledge base for better cross-worktree sharing, but chose to keep the simpler in-repo setup.)

---

## 8. Human-in-the-loop: concise reports only

The human does **not** approve specs or gate compute. Within the ≤16-GPU budget (§5.2), agents run T0→T1
autonomously. The human's interface is the ledger, kept **very concise**:

- **Index line per experiment** — one row:
  `<id> | <one-line idea> | <claim type> | <stage: T0/T1/merged/killed/parked> | <headline result>`
- **Result card per experiment** (only when stage changes) — ≤5 lines: hypothesis, the T1 (or
  current-tier) speedup, the quality bundle verdict (within-band / regressed-X), and artifact links.

That's the whole report surface: *which experiments are at which stage, and the headline number.* The
human owns the baselines and the measurement contract; agents never silently change either.

---

## 9. Idea backlog (seed list — not prescriptive)

There are **three compute centers** an experiment can attack — the **PPGD inner loop**, the **CI
function** (`global_shared_transformer`), and the **target-model forward** (frozen). **We do not
assume which dominates** — `pd-speedup-bench`'s profiler shows where the time actually goes on each
tier; start an idea where the profile says the cost is. Any idea, listed or not, is judged the same
way (§§1–8). Categorized so agents can work non-overlapping areas in parallel:

**PPGD inner loop**
- **Precision** — lower precision for the PPGD inner loop; per-phase autocast (today
  `RuntimeConfig.autocast_bf16` is all-or-nothing); fp32 only where faithfulness residuals need it.
- **Cheaper warmup objective** — hidden-state MSE instead of logit-KL during faithfulness warmup;
  shorter / scheduled warmup.
- **Inner-loop budget** — fewer `n_warmup_steps` / `n_samples`; smaller batch for the inner PPGD
  loop than the outer step; cheaper source optimizer/scope.
- **Approximate component forward** — the masked forwards through components are a big cost;
  subsample components, gate by CI threshold, or low-rank/low-precision the component matmul,
  validated by an equivalence test vs the exact forward.

**CI function** (`global_shared_transformer`) — it runs every step to produce causal importances:
cheaper architecture / fewer blocks / lower precision / approximations, traded against the quality
bundle.

**Target-model forward** — the target is **frozen**, so for a fixed batch its activations are
constant: **cache / precompute** them instead of recomputing the target forward each step (esp. with
a fixed data order), or cheaper target evaluation.

> The aspirational wins ("simplify PPGD so the loop is 5× faster for similar loss"; "an approximate
> component forward that works just as well"; "cache the frozen target forward") are open-ended — the
> contract is what makes any of them trustworthy.

---

## 10. Bootstrap checklist (first steps, in order)

Status as of **2026-06-08** (✅ done / ◐ partial / ☐ todo):

1. ✅ **Iteration harness (§4)** — all on the base DDP `pd-lm` path, validated on an H100:
   - smoke configs `…-2L-smoke.yaml` / `…-4L-smoke.yaml` (both run clean through step 20);
   - `pd-speedup-bench` (`param_decomp_lab/speedup/benchmark.py`) — step time / tokens-sec / peak
     mem / profiler breakdown;
   - `pd-speedup-compare` (`compare_runs.py`) + `quality_bundle.py` — the frozen quality-bundle diff.
   - (Unblocked T0: fixed a legacy-config migration assert in `pretrain/run_info.py` that prevented
     loading the `ss-2L` target.)
2. ◐ **Lock baselines.** T1 ✅ locked to Jose's `s-55ea3f9b` (W&B metrics cached locally). T0 ◐
   3 seeds running (`p-db5adc3b`/`p-89a42376`/`p-993ae14a`, 50k/b32); band = seed spread (§3.3).
3. ✅ **Real cost numbers** filled into the §2 ladder.
4. ✅ **Spec template + ledger index** (in-repo flat ledger, `track2/ledger/`); eval harness +
   quality bundle + baselines marked **read-only for experiments** (`track2/README.md`).
5. ☐ *Only then* pull the first ideas off the backlog, one worktree each, running autonomously.

> The harness/contract/ledger live in `track2/` (docs) + `param_decomp_lab/speedup/` (code). See
> `track2/README.md` for the operating manual the agents follow.

---

## 11. Execution — running the loop in Claude Code

How we actually set this off and keep it running unattended. The model: **one orchestrator
session** owns the ledger + GPU budget and the long waits; it dispatches bounded work to
subagents and runs each experiment in its own worktree.

### 11.0 Prerequisites (once, before kickoff)
- **T0 baseline locked** — all 3 seeds (`p-db5adc3b`, `p-89a42376`, `p-993ae14a`) finished and the
  per-metric band (seed spread) written into `track2/ledger/baselines.md` (and §3.1). The ruler must
  be locked before any experiment can be quality-judged on T0.
- **Track-2 lives on `feature/track2-setup` — do NOT merge to `main`** (owner's call). All Track-2
  work runs off that branch: per-experiment worktrees **branch from `feature/track2-setup`**
  (`git worktree add <path> -b feature/spd-<idea> feature/track2-setup`), and a *won* experiment
  merges **back into `feature/track2-setup`**, never `main`. In each worktree run
  `uv sync --all-packages` first (plain `uv sync` misses the lab CLIs).
- **Permission posture: bypass** (decided — 11.4); launch with `claude --dangerously-skip-permissions`.

### 11.1 How to start it
Start one self-paced loop in a session at the repo root:

```
/loop Run the next Track-2 experiment per track2/README.md and plan.md §11. First reconcile
in-flight SLURM jobs (squeue --me) and record any finished results in the ledger; then, if our
GPU use is ≤16 and the backlog (§9) has an untried idea, start the next experiment. Stop when the
backlog is exhausted.
```

- **No interval → self-paced:** the agent chooses its own wake cadence via `ScheduleWakeup`
  (short while setting up, long while waiting on training). It ends the loop by not rescheduling
  once the backlog is empty.
- Durable state lives in files (ledger, baselines, this contract, CLAUDE.md), so context
  compaction or a fresh session loses nothing — each iteration re-reads them. `/loop` auto-expires
  after 7 days; re-kick for a longer campaign.

### 11.2 One orchestrator iteration (the loop body)
1. **Reconcile.** `squeue --me`; for each of *our* finished experiment runs:
   `pd-speedup-compare <T0 3 seed ids, comma-sep> <run> --at_step 50000` (use `s-55ea3f9b` as the T1
   baseline) + `pd-speedup-bench`, then write the result card + index row (promote/kill/park) in the
   ledger. Read the overall verdict; faithfulness gate first. Never touch jobs that aren't ours.
2. **Dedup.** Grep the ledger for the candidate idea — if tried/killed, skip it.
3. **Budget.** Sum GPUs across our in-flight jobs; only start a new one if it keeps us **≤16**
   (a single run may itself use up to 16 — multi-node `--dp`, multiple of 8; §5.2).
4. **Start next experiment** (11.3) if budget + backlog allow. Several experiments' runs may be
   in the SLURM queue at once (that's the parallelism — not multiple agent processes).
5. **Wait efficiently.** After launching a run, start a **background Bash `until`-loop** that
   blocks until the job leaves the queue (`until ! squeue -j <id> -h -t R,PD | grep -q .; do
   sleep 120; done`). When it exits the harness re-invokes the orchestrator — cheaper and faster
   than `ScheduleWakeup` polling.

### 11.3 One experiment (worktree-isolated)
`git worktree add <path> -b feature/spd-<idea> feature/track2-setup`; `uv sync --all-packages` in
it; **never `cd` back to main**. A won experiment merges back into `feature/track2-setup`.
1. Spec: `track2/ledger/experiments/<id>.md` from `TEMPLATE.md` (hypothesis, claim type,
   thresholds quoted from §3, artifact links).
2. Make the core change in `param_decomp/` (free rein). Add an **equivalence test** for any
   approximation (`param_decomp/tests/`).
3. **Smoke** (`…-2L-smoke.yaml`) — must pass, no NaNs.
4. **Benchmark** both tiers (`pd-speedup-bench`) — confirm ≥10% speedup, T0 and T1.
5. Launch the comparison run **from the worktree**: `pd-lm <baseline-cfg> --dp N --job_name ai-pd-lm`
   — the config is the **baseline config** (`…-2L-baseline.yaml`, 50k/b32 for T0) with *only* the
   change under test; **eval block unchanged** (it's the ruler). Always prefix the SLURM job name
   with `ai-`. The snapshot captures the worktree's edits, so the run is reproducible. Record
   run_id + wandb URL in the spec.
6. On completion: full-bundle `pd-speedup-compare`; `/code-review` the diff before any merge;
   promote T0→T1→merge or kill — always with artifacts.

### 11.4 Permissions (unattended) — decided: bypass
Agents run with **bypass permissions** (`claude --dangerously-skip-permissions`), per the owner's
call (2026-06-08) — full autonomy, no per-step approval. The guardrails that matter here are the
contract (read-only ruler, artifacts, asserts, faithfulness gate) and the `ai-` SLURM job prefix for
attribution, not interactive permission prompts.

### 11.5 Claude Code primitives this uses
- **`/loop` (self-paced)** — the driver (11.1).
- **Worktrees** — `git worktree` per experiment (or an `isolation: worktree` subagent); isolation
  so parallel work can't clobber, dead ends are deleted not untangled.
- **Background Bash `until`-loop** — turns an external SLURM wait into harness-tracked work that
  re-invokes the orchestrator on completion (no polling cost).
- **Subagents** — `Explore` for "has this been tried / where's the relevant code" fan-out (keeps
  the orchestrator's context lean); `Plan` for designing a large core rewrite; `/code-review` on
  the diff before merge. *You don't invoke these — the execution steps tell the orchestrator to.*
- **The ledger + CLAUDE.md** — durable state across compaction/sessions.

> **Do I need to do something special to invoke it?** Just start a session with
> `claude --dangerously-skip-permissions` (bypass, 11.4) and run the `/loop` in 11.1. You do **not**
> manually request subagents or worktrees — those are driven by the execution steps above.
> Optionally wrap 11.3 in a custom slash command (`/pd-new-experiment`) later if it earns it (§4).

### 11.6 Gotchas
- **Context compaction** over many iterations — never rely on chat memory; the ledger + baselines +
  contract are the source of truth, re-read each loop.
- **GPU budget ≤16 in flight** (single run up to 16, multi-node). Still check the whole `squeue`
  for Track-1's jobs and overall pressure; manage only your own.
- **Prefix every SLURM job name with `ai-`** (`--job_name ai-pd-lm`) so AI-launched jobs are
  identifiable in `squeue`.
- **Zombie jobs** after a crash — verify `squeue --me` (known error-path issue) and `scancel` only
  your own.
- **Never edit the ruler** (eval harness, `quality_bundle.py`, `baselines.md`) inside an experiment.
- **7-day `/loop` expiry** — re-kick for longer campaigns.

---

## Appendix — where the compute goes (for the agents)

Grounding so agents don't re-derive it. Verify before relying on any line number.

- **Entry / loop:** `Trainer.__init__` + `Trainer.run` in `param_decomp/optimize.py`. One step:
  weight-deltas (fp32, outside autocast) → target forward + CI → each loss metric `.update(ctx)` →
  weighted sum → `before_backward` → `backward` → `after_backward` (PPGD steps its persistent
  sources) → optimizer step. Eval interleaves via `EvalLoop`.
- **Compute centers (don't assume — profile with `pd-speedup-bench`):** (1) **PPGD** =
  *Persistent Projected Gradient Descent*, `param_decomp/metrics/persistent_pgd_recon.py` +
  `persistent_pgd_state.py` — `(n_warmup_steps + n_samples)` masked forwards through components;
  (2) the **CI function** forward (`global_shared_transformer`), run every step; (3) the **target
  forward** (frozen → cacheable). Which dominates is tier-dependent; let the profiler say.
- **Config surface:** `PDConfig` (`loss_metrics` + `coeff`, `n_mask_samples`, optimizers,
  `faithfulness_warmup_*`, `steps`, `batch_size`), `RuntimeConfig.autocast_bf16` (global today),
  `Cadence`. Many ablations are YAML-only, but core code edits are fully allowed in this track.
- **Component forward:** `param_decomp/component_model.py` (`forward`,
  `calc_causal_importances`, `calc_weight_deltas`) + `components.py` — all C components computed in
  one batched einsum (`V^T x` then `@ U`), not per-component; the target for "approximate forward".
- **Precision:** `RuntimeConfig.autocast_bf16` (global); warmup (`faithfulness_warmup.py`) currently
  runs outside autocast.
- **Fast validation without a run:** `param_decomp/tests/` (`test_optimize.py`,
  `test_component_model.py`, …) via `make test`; the seed for the equivalence tests every
  approximation must add.
- **Profiling:** plain `torch.profiler` via `pd-speedup-bench` (the `pd/*` `record_function` labels
  and `docs/profiling.md` are on Track-1 branches, not `main`). Add finer labels inside an
  experiment if a change needs them.
