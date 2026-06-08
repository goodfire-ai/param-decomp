# Research Track: Speeding Up the Core PD Method

**Status:** plan / proposal
**Owner:** Dan
**Audience:** the AI coding agents that will execute this track, plus the human reviewing them.

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
- **Guardrail — must not blow up:** faithfulness under the CI mask (`ce_difference_ci_masked`,
  `kl_ci_masked`, `ce_unrecovered_ci_masked`). Sanity, not the thing we optimize.

So a result is: *materially faster/simpler, with PPGD-recon and L0 within band (or better) vs the
baseline, secondary metrics not meaningfully worse, and faithfulness not blown up.* This is encoded,
tier by tier, in `param_decomp_lab/speedup/quality_bundle.py` and surfaced by `pd-speedup-compare`.

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
> Costs above are **measured** (2026-06-08): T0 via `pd-speedup-bench` on the canonical config;
> T1 from the wall-clock of Jose's baseline run. The T1 baseline is **not re-run** by us — we read
> its stored metrics from W&B (see §3.1).

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

### 3.1 Locked baselines (one per tier)
- Run the canonical config on T0 and T1. Record each `run_id`, `metrics.jsonl`, and wandb link, and
  commit them as **the baselines**, referenced by short code. Agents compare against these exact
  artifacts, never against a baseline they re-derive.
- Re-run a baseline only when the human deliberately rebases it (old one kept).

> **Status (2026-06-08).** **T1 LOCKED** to Jose's existing run `s-55ea3f9b` (`goodfire/spd`,
> 400k steps) — we can't re-run it, so the harness reads its stored W&B metrics; an 81-row
> `metrics.jsonl` artifact is cached at `runs/s-55ea3f9b/` so `pd-speedup-compare s-55ea3f9b …`
> resolves it by short code. **T0 RUNNING** — `p-db5adc3b`, a single-seed *reduced* run
> (`ss_llama_simple_mlp-2L-baseline.yaml`: 50k steps / batch 32, current code) as a
> qualitatively-representative comparison point. Both tiers are **single-seed** initially (§3.3).
>
> **T0 baseline results — FILL ON COMPLETION of `p-db5adc3b`:** record its `metrics.jsonl` path
> and the final quality-bundle values in `track2/ledger/baselines.md`, then this tier is locked.
> Until then, experiments can smoke/benchmark but cannot be quality-judged on T0.
>
> | bundle metric | T0 (`p-db5adc3b`) | T1 (`s-55ea3f9b`) |
> |---|---|---|
> | CE diff (CI-masked) | _TBD_ | 0.28603 |
> | CE unrecovered (CI-masked) | _TBD_ | 0.0042753 |
> | KL (CI-masked) | _TBD_ | 0.34315 |
> | L0 total | _TBD_ | 201.45 |
> | Stochastic hidden-acts recon | _TBD_ | 0.41470 |
> | PGD recon | _TBD_ | 0.65225 |

### 3.2 Frozen eval harness + metric bundle (agents don't edit these)
Same eval split, seeds, and metric set both tiers. Two groups:

- **Primary** (the thing we trade): wall-clock per step + tokens/sec on a **pinned GPU type**, fixed
  batch/seq, plus a step-time breakdown from a profiler (torch.profiler + `pd/*` labels). The
  existing `analyze_3pool_trace.py` is 3-pool-specific — for this track we time the **base DDP**
  `pd-lm` path, so the benchmark wraps the plain profiler, not that analyzer.
- **Quality bundle (the objective)** — the eval metrics as a *tiered vector* (so a change can't hide
  a regression in one by improving another), defined in `quality_bundle.py` and described in §1:
  **primary** = PPGD recon (`PGDReconLoss`) + `l0_total`; **secondary** = `StochasticHiddenActsReconLoss`
  + `CIHiddenActsReconLoss`; **guardrail** = CI-masked faithfulness (`ce_difference_ci_masked`,
  `kl_ci_masked`, `ce_unrecovered_ci_masked`). Plus visual guardrails (`ComponentActivationDensity`,
  `ci_mean_per_component`) eyeballed in wandb. A change reports the **whole** vector vs baseline via
  `pd-speedup-compare` (which groups by tier).

### 3.3 Statistics, not vibes
- **Judge early — don't wait for full training (esp. the 4L Pile run).** The baselines have their
  *whole trajectory* cached, so an experiment can be evaluated **at a matched intermediate step**
  against the baseline at that same step (`pd-speedup-compare --at_step N`). Run experiments to a
  reduced step budget and compare the trajectory, rather than running to convergence. A clear
  early divergence (primary metrics tracking the baseline, or clearly worse) is enough signal to
  promote or kill. Pick `N` on a slow-eval step (multiple of `slow_every`) so the slow metrics
  exist. Per-step *speed* needs only a short run regardless. Reserve a longer run only to confirm a
  borderline winner.
- **Seeds — single-seed initially** (user's call, 2026-06-08): iterate and even confirm on T1 at one
  seed to move fast. Escalate to **≥3 seeds** only before *claiming/merging* a result, or when a
  Δ sits near the band edge. Report mean ± spread once you have multiple seeds.
- **"Within band" with a single-seed baseline:** we have no measured seed spread (the T1 baseline is
  one run we can't repeat), so within-band is a **fixed relative tolerance** (`pd-speedup-compare
  --tol_pct`, default ±2%) rather than a measured σ. Replace with a real spread if/when a baseline
  is ever run multi-seed.
- The decision object is a **Pareto point** (speed vs the quality vector), not a single scalar.

### 3.4 Make gaming structurally hard
- Eval harness, baseline pointers, and this contract are **read-only for an experiment**; changing
  them is a separate, explicitly-flagged change the human can see.
- Every result cites **artifacts** (`run_id`, `metrics.jsonl` path, wandb URL). No artifact → it
  didn't happen. (Blocks hallucinated results.)
- Keep fail-fast asserts on (`isfinite`, shape checks): a speedup that trips an assert is a failure,
  not a finding.

---

## 4. Iteration harness to build *first* (before any experiment)

Infra is ~85% there; the gaps are what slow agents down. Build these *small* pieces first; resist
gold-plating (the user wants minimal setup). Everything targets the **base DDP `pd-lm`** path.

1. **Smoke configs** per tier (`*-smoke.yaml`): tiny `steps`, truncated splits, no wandb, fast-only
   eval. One command, finishes in minutes, exercises the full code path. First thing every agent
   runs after editing.
2. **A benchmark command**: given a config, report step time, tokens/sec, peak memory, and the
   profiler breakdown — pinned GPU, fixed batch/seq, eval excluded from timing. Produces the §3.2
   primary metric reproducibly.
3. **A compare-runs report**: given baseline `run_id` + variant `run_id`(s), emit one markdown/plot
   diff over the full §3.2 quality bundle (just reads `metrics.jsonl`). Every experiment report
   embeds this, so "compare to baseline" is one command.
4. **Baseline lock**: run §3.1 on both T0 and T1; record each baseline's seed spread (defines
   "noise").

Keep these as plain scripts + configs in `param_decomp_lab/`. Optionally wrap the scaffold/report in
**one or two custom slash commands** (`/pd-new-experiment`, `/pd-report`) only if they earn it.

---

## 5. Autonomous agent workflow (Claude Code)

Bypass-permissions mode is on, so agents run unattended. Design for isolation, reproducibility, and
shared memory — the three things multi-agent setups lose.

### 5.1 One experiment = one worktree = one branch = one spec
- Each experiment runs in its **own git worktree** (`isolation: worktree` / `EnterWorktree`) so
  parallel agents can't clobber each other and dead ends are deleted, not untangled. Repo rule:
  `uv sync` in the worktree, never `cd` back to main.
- Branch `feature/spd-<short-idea>`. Edit core `param_decomp/` as freely as needed — no flag-gating
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
- Permissions allowlisting is moot under bypass mode. Keep CLAUDE.md current as configs change. Do
  **not** add MCP servers or heavier orchestration without a concrete need.

---

## 6. Common AI-agent failure modes & mitigations

| Failure mode | What it looks like here | Mitigation |
|---|---|---|
| **Reward hacking the speed metric** | Step faster because correctness silently broke | Quality bundle is a mandatory *vector* (§3.2); asserts stay on; equivalence tests for approximations |
| **Moving the goalposts** | Agent edits eval/baseline so "loss is similar" | Eval + baseline read-only for experiments (§3.4); changes are separate + visible |
| **Premature conclusion** | Promote off one short single-seed T0 run | Min step budget + ≥3 seeds + Pareto decision (§3.3); T0 win is a hypothesis, T1 is the bar |
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
result afterward (§8). Promotion criteria:

- **→ T0:** smoke passes, unit + equivalence tests green, no NaNs.
- **T0 → T1:** quality bundle within band + a real measured speedup on `ss-2L`, ≥3 seeds.
- **T1 → merged:** holds on `pile_llama_simple_mlp-4L`, full quality bundle within band vs baseline,
  measured speedup, ≥3 seeds. This is a finished result; flip the change to default.

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

Categorized so agents can work non-overlapping areas in parallel. The point of §§1–8 is that *any*
idea, including ones we haven't listed, gets evaluated the same way.

- **Precision** — lower precision for the PPGD warmup inner loop; per-phase autocast (today
  `RuntimeConfig.autocast_bf16` is all-or-nothing); fp32 only where faithfulness residuals need it.
- **Cheaper warmup objective** — hidden-state MSE instead of logit-KL during PPGD warmup; shorter /
  scheduled faithfulness warmup.
- **Inner-loop budget** — fewer `n_warmup_steps` / `n_samples`; smaller batch for the inner PPGD
  loop than the outer step; cheaper source optimizer/scope.
- **Approximate component forward** — the dominant cost is `(n_warmup_steps + n_samples)` masked
  forwards through components; subsample components, gate by CI threshold, or low-rank/low-precision
  the component matmul, validated by an equivalence test vs the exact forward.

> The aspirational wins ("simplify PPGD so the loop is 5x faster for similar loss"; "an approximate
> component forward that works just as well") live in the last two buckets — the backlog is open, and
> the contract is what makes any of them trustworthy.

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
2. ◐ **Lock baselines.** T1 ✅ locked to Jose's `s-55ea3f9b` (W&B metrics cached locally). T0 ☐
   pending one single-seed canonical run. (Single-seed initially — §3.3.)
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
- **Baseline `p-db5adc3b` finished and locked** — its final quality-bundle values written into
  `track2/ledger/baselines.md` (and §3.1). The ruler must be locked before any experiment can be
  quality-judged on T0.
- **Land the Track-2 setup on `main`.** It's committed on branch `feature/track2-setup` (speedup
  harness, smoke + baseline configs, the `run_info.py` fix, `track2/`). Per-experiment worktrees
  branch from `main`, so **merge this branch to `main` before kickoff** (or branch experiment
  worktrees from `feature/track2-setup`) — otherwise worktrees won't inherit the harness/fix.
- **Pick the permission posture** (11.4).

### 11.1 How to start it
Start one self-paced loop in a session at the repo root:

```
/loop Run the next Track-2 experiment per track2/README.md and plan.md §11. First reconcile
in-flight SLURM jobs (squeue --me) and record any finished results in the ledger; then, if our
GPU use is <8 and the backlog (§9) has an untried idea, start the next experiment. Stop when the
backlog is exhausted.
```

- **No interval → self-paced:** the agent chooses its own wake cadence via `ScheduleWakeup`
  (short while setting up, long while waiting on training). It ends the loop by not rescheduling
  once the backlog is empty.
- Durable state lives in files (ledger, baselines, this contract, CLAUDE.md), so context
  compaction or a fresh session loses nothing — each iteration re-reads them. `/loop` auto-expires
  after 7 days; re-kick for a longer campaign.

### 11.2 One orchestrator iteration (the loop body)
1. **Reconcile.** `squeue --me`; for each of *our* finished experiment runs: `pd-speedup-compare
   p-db5adc3b <run>` + `pd-speedup-bench`, then write the result card + index row
   (promote/kill/park) in the ledger. Never touch jobs that aren't ours.
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
Branch `feature/spd-<idea>`; new git worktree; `uv sync` in it; **never `cd` back to main**.
1. Spec: `track2/ledger/experiments/<id>.md` from `TEMPLATE.md` (hypothesis, claim type,
   thresholds quoted from §3, artifact links).
2. Make the core change in `param_decomp/` (free rein). Add an **equivalence test** for any
   approximation (`param_decomp/tests/`).
3. **Smoke** (`…-2L-smoke.yaml`) — must pass, no NaNs.
4. **Benchmark** both tiers (`pd-speedup-bench`) — confirm a real speedup, T0 and T1.
5. Launch the comparison run **from the worktree** (`pd-lm <cfg> --dp N`) — config identical to
   the baseline except the change under test. The SLURM snapshot captures the worktree's edits, so
   the run is reproducible. Record run_id + wandb URL in the spec.
6. On completion: full-bundle `pd-speedup-compare`; `/code-review` the diff before any merge;
   promote T0→T1→merge or kill — always with artifacts.

### 11.4 Permissions (unattended) — *your call*
The loop edits files and runs bash without per-step approval. Two options:
- **Allowlist (safer, recommended on a shared cluster):** pre-approve the command patterns the
  loop needs in `.claude/settings.json` (`git`, `uv`, `python -m param_decomp_lab.*`, `pd-lm`,
  `pd-speedup-*`, `squeue`, `pytest`); anything else still prompts. The `fewer-permission-prompts`
  / update-config skills can build this list.
- **Bypass (full autonomy):** launch `claude --dangerously-skip-permissions`. Skips all prompts —
  only appropriate in a sandbox/container. This is a shared cluster, so prefer the allowlist unless
  the environment is isolated.

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

> **Do I need to do something special to invoke it?** Only two things: (1) start the `/loop` in
> 11.1, and (2) choose the permission posture in 11.4. You do **not** manually request subagents
> or worktrees — those are driven by the execution steps above. Optionally wrap 11.3 in a custom
> slash command (`/pd-new-experiment`) later if it earns it (§4).

### 11.6 Gotchas
- **Context compaction** over many iterations — never rely on chat memory; the ledger + baselines +
  contract are the source of truth, re-read each loop.
- **GPU budget ≤16 in flight** (single run up to 16, multi-node). Still check the whole `squeue`
  for Track-1's jobs and overall pressure; manage only your own.
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
- **The hot loop:** PPGD = *Persistent Projected Gradient Descent*.
  `param_decomp/metrics/persistent_pgd_recon.py` + `persistent_pgd_state.py`. Cost per step is
  dominated by `(n_warmup_steps + n_samples)` masked forwards through components — the main lever.
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
- **Profiling:** torch.profiler + `pd/*` labels (`docs/profiling.md`). For this track, profile the
  base DDP path — not the 3-pool analyzer.
