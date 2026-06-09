# Handover: `feature/adv-lee` — adversarial-mask robustness for SPD

_To the next Claude (on another cluster). Written 2026-06-09 from the `reno` cluster. This
branch is pushed to `origin/feature/adv-lee` (HEAD `130caacc0`). Read this, then see
`param_decomp/metrics/CLAUDE.md` and `param_decomp_lab/eval_metrics/CLAUDE.md`._

## TL;DR — what to do next
Launch the **head-init-PGD v2** run on your cluster and watch whether it fixes an
oscillation in the success metric:
```bash
# in a worktree/checkout of feature/adv-lee, with .venv active:
pd-lm param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_adv-headpgd-v2.yaml \
  --dp 8 --time 144:00:00 --group adv-lee --tags "adv-lee,headpgd,head-and-random"
```
Then watch (wandb group `adv-lee`): **`eval/loss/PGDReconLoss`** (the goal — should fall and
**stay** low, not oscillate), `eval/loss/HeadInitPGDReconLoss/random_restart_win_frac`, and the
new **`figures/.../pgd_adversarial_source_values`** histogram.

## The goal & the success metric
SPD decomposes a target model's weights into components gated by per-datapoint causal
importances (CI). We want the decomposition to be **robust to adversarial masking**. The
**success metric is `eval/loss/PGDReconLoss`**: it runs a 20-step sign-PGD attack (random
init, `shared_across_batch`) that searches the `[0,1]` mask box for the worst-case
`mask = ci + (1-ci)*source`, and reports recon (KL) under it. **Lower = more robust.**

Reference points (all `pile_llama_simple_mlp-4L`, 8×H200):
- **Control** = exact replica of the "Jose" VPD-paper run (`goodfire/spd/s-55ea3f9b`), which
  trains a **PersistentPGD** adversary. It gets eval `PGDReconLoss ≈ 1.0–1.4` and is stable.
  This is the bar to beat / match.
- **All our learned-adversary arms are stuck at ~45–104** (≈50–70× worse). Why: a single-shot
  amortized adversary head is far weaker than a 20-step free PGD optimizer — the defender
  trivially beats it (its own-adversary recon is ~0.1–11), so it learns ~no PGD robustness.
  This is the central finding; see `head_init_pgd_recon.py`'s module docstring.

## What's on this branch (3 commits on origin/main `5ce561df4`)
Core losses (`param_decomp/metrics/`):
- **`adversarial_distribution_recon.py`** — `AdversarialDistributionReconLoss`. A head off the
  CI trunk emits per-component distribution params; reparameterized samples are the mask
  sources. `distribution`: `deterministic` (sigmoid), `gaussian_sigmoid`, `beta` (bounded
  [1,100]). `trunk_grad`: `detach` (head-only) or `reverse` (gradient-reversal co-opts the
  trunk). Adversary ascends via its own AdamW.
- **`head_init_pgd_recon.py`** — `HeadInitPGDReconLoss` (**the promising approach**). A
  *detached* deterministic MLP head predicts a PGD **initialization**; sign-PGD refines it a
  **random number of steps** (`pgd_steps_min/max`) and optionally from a **random restart**
  (two *distinct* randomnesses — keep the names separate). The **defender** descends recon at
  the PGD endpoint (PGD-family ⇒ transfers to eval); the **head** is trained by **distillation**
  (`MSE(head_output, pgd_endpoint.detach())`) — a stable regressor, no minimax.
- Supporting: `ci_fns.py` `GlobalSharedTransformerCiFn.trunk_features()` (pure refactor) +
  `GlobalCiFnWrapper.transformer`; `ci_sigmoids.py` `double_leaky_hard`; `pgd_utils.pgd_attack()`
  (returns final sources); config-union + dispatch wiring.

Eval metric (`param_decomp_lab/eval_metrics/`):
- **`PGDSourceHistogram`** — runs the eval PGD attack and plots per-layer histograms of the
  worst-case source values. Added to the headpgd-v2 config's `eval.metrics`. Use it to see what
  the adversary's solution looks like (see "Solution characteristics" below).

Configs (`param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L_adv-*.yaml`): `control`,
`gauss`/`beta`/`deterministic` (+ `-detachdeep` variants), `headpgd`, **`headpgd-v2`** (the
one to run).

## The story so far (what we tried, what we learned)
1. **Distribution-head adversaries** (gaussian/beta/deterministic, both `reverse` and
   `detach-deep`): all stuck ~45–104 on eval PGDRecon. The amortized head is too weak.
2. **Deterministic-`reverse` diverged** (ImpMin → 1e8). Cause: `double_leaky_hard`'s
   *non-vanishing* rail gradient drives the *ascending* adversary's logits to ±∞, and via the
   reverse-coupled shared trunk + CI's `leaky_hard` lower-leak that blows up `ci²`. **Fixed**
   by (a) deterministic squashing → plain `sigmoid` (gradient vanishes at saturation, self-
   limits) and (b) `ci_fn_optimizer.grad_clip_norm: 1.0` (it was *unclipped* in Jose's config;
   the reverse coupling inflates CI-fn grads 10–60× over control).
3. **head-init-PGD** got eval PGDRecon down to **~5** (best learned arm) **then oscillated
   5↔39**. Cause: winner-take-all PGD-endpoint selection + a good head ⇒
   `random_restart_win_frac = 0` always ⇒ the random restart was **discarded** ⇒ defender
   overfit to the head's *narrow* attack mode, brittle on the broad eval PGD. **Fix =
   `defender_target: head_and_random`** (pool BOTH the head-init and random-init endpoints) →
   that's **headpgd-v2, which has NOT yet run at scale** (it was stuck PENDING on contended
   reno — the reason for this cluster move). **This is the experiment to run.**

## Solution characteristics (answer to "is it all 0s/1s?")
From logged source stats (`source_frac_saturated`, mean, std):
- The strong (distribution) adversaries converge to a **near-binary, bimodal mask**: ~87–99%
  of source values pinned at 0/1, `std ≈ 0.46–0.50` (max for [0,1]), mean ~0.54–0.69 → **roughly
  half-to-two-thirds of components driven fully ON** (source→1), the rest left at ci (source→0).
- The head-init-PGD adversary is **soft/interior** (`frac_saturated ≈ 0.20`).
- The new `PGDSourceHistogram` gives the full per-layer shape going forward.

## Cluster / infra notes (important)
- **Env:** `source .venv/bin/activate`. In a fresh worktree: `uv sync --all-packages` (you need
  the `param-decomp-lab` package for `pd-lm` + wandb). Needs `.env` or `wandb login` (jobs auth
  via `~/.netrc`; the snapshot excludes `.env` since it's gitignored).
- **Launch:** `pd-lm <config> --dp 8`. `--dp 8` submits a single-node 8-GPU DDP SLURM job.
- **Throughput:** control/PPGD ≈ **3.1 steps/s** (~36 h for 400k); head-init-PGD ≈ **1.2 steps/s**
  (~93 h+) because of the PGD inner loop (~10 model forwards/step). **Use a long `--time`**
  (e.g. `144:00:00`) — there is **no auto-requeue**; `save_every: 5000` is set for manual
  `--resume` if a job dies.
- **Snapshots:** `pd-lm --dp` snapshots the working tree (incl. untracked files) to a pushed
  `refs/runs/snapshot/<run_id>` ref and runs that — so **commit/clean the branch before
  launching** (it's clean now). Each run's exact code is recoverable from its snapshot ref.
- **GPU policy:** repo CLAUDE.md says ≤8 GPUs at once; the user (lee) has explicitly OK'd
  exceeding that for this comparison.
- **Monitoring:** `monitor_jobs <id> ...` (on PATH, run in background); `logs <id>`; wandb
  group `adv-lee`. Don't `--partition`; don't set CPU/RAM on GPU jobs.

## Still running on reno (data persists in wandb; you don't need to manage these)
- control (Jose replica, PPGD): job 61155 / `p-df25d4c5` — the eval-PGDRecon ≈ 1.4 baseline.
- head-init-PGD **winner-take-all** (the shaky one): job 61259 / `p-4838d26a` — its oscillation
  is the A/B baseline for the v2 fix.
(The distribution-arm runs `p-91205fbc`, `p-03428711`, `p-053a5541`, `p-7dd141ec`, `p-9b35aec0`,
`p-0c888b92` have crashed/finished; their curves are still in wandb for comparison.)

## If headpgd-v2 still struggles, levers to try (untested)
- Raise/`pgd_steps` toward eval's 20 (closer attack match) — costs throughput.
- Match the eval threat model (`shared_across_batch`) in the *training* attack.
- Tune the distillation / head LR; reduce the step-count variance.
- Reconsider whether a learned adversary can ever beat a free 20-step PGD optimizer at its own
  metric — control (PGD-trained) may simply be the right tool, and the learned-adversary work is
  about amortization/insight.

## Misc
- The user is **lee@goodfire.ai**. Share wandb URLs + SLURM job ids when launching runs.
- Project memory on reno: `~/.claude/.../memory/project_adv_lee_runs.md` (won't travel unless the
  home is shared — this doc is the portable record).
- Delete this file before opening a PR.
