---
name: parameter-decomposition
display-name: Parameter Decomposition
description: >
  Train a decomposition of model parameters using adVersarial Parameter Decomposition (VPD). This method splits models into simple, computationally faithful 'mechanisms'. It works by splitting model parameters into rank-1 subcomponents and trains them alongside a 'causal importance function'. The causal importance function identifies the minimal set of subcomponents in the target model that are 'causally (un)important' on a given input. Causally unimportant subcomponents can be completely, or partially masked (in any combination, including adversarially), and the masked network will still perform the same output.
user-invocable: true
---

# Parameter Decomposition

Neural networks use different parts of their parameters for different computations. How do we split up their parameters into parts that each do particular computations? This is the job of adVersarial Parameter Decomposition (VPD).

## The core reframe

Interpretability analyses commonly start by decomposing a network's activations, e.g. using SAEs or manifold discovery methods. But these methods can identify structure that the network doesn't actually _use_ (e.g. feature splitting/absorption in SAEs, or finding manifold structure that is identifiable in the input data). Parameter decomposition gives us a way to know which parts are actually used by the model for computation. Instead of first looking for the representations (variables) on which computations might be done, parameter decomposition identifies the computations first; then we can identify the representations they use.

Another core reframe is that we're doing causal ablations on parameters, rather than on activations. Our causal importance function amortizes the results of many ablation experiments into a function that produces a single set of numbers: The causal importance values for each parameter subcomponent on a given datapoint.

## When to reach for this skill

- "Decompose this model's weights into mechanisms / parameter components."
- "Run VPD / param-decomp / parameter decomposition on {this model}"
-  "Decompose the attention layers of this model (even though their computations may be distributed across attention heads)"
- "Find all the circuits in this model"
- "Identify the computations done by this model"
- "What circuit in the weights computes {behavior}? I want hand-editable components."
- "Find the minimal set of weight components a prediction causally depends on."

## Recipe

A VPD run, typically applied to a transformer language model (lm), is one validated YAML driving the command `pd-lm`. The most important hyperparameters (mainly the importance-minimality coeff) aren't knowable a priori, **the default arc is to run a sweep over importance-minimality coefficient. Depending on which pathologies this sweep surfaces, sweeps over other hyperparameters may be necessary too (often frequency-minimality coefficient or C). Skip the sweep only if you're reusing known-good hyperparameters from a prior VPD run on a comparable target. Runs must go to convergence - analyzing or comparing unconverged decompositions is almost always worthless.


### 1. Confirm the plan up front

Present these four things to the researcher and iterate until you get explicit confirmation on all of them:

- **Target** (`target`) — a `spec` plus a sibling `output_extract` at `target.output_extract`,
*not* inside `spec` (the spec models are `extra="forbid"`, so a nested `output_extract` is
rejected at config load; a sibling one is a legacy torch-era field that's stripped on load —
inert but conventionally kept to match the reference YAMLs). `spec` is `kind: hf`
(+ `model_class` = the HF class, `model_name` = the hub id) with `output_extract: logits` for
an off-the-shelf HuggingFace LM, or `kind: pretrained` (+ `model_class` = the in-repo class,
`run_path` = a W&B run) with `output_extract: 0` for a model trained in-repo with `pd-pretrain`.
- **Base config** — the reference config plus the values for hyperparameters that matter, all at reference values except the one(s) you sweep.
- **Sweep grid** — the axes and their values (see step 3).
- **Selection rule** — If you're doing a sweep of decompositions on toy models or models with ground truth, you're probably using those models to explore selection rules. If you're doing it on a model without a ground truth, there are various model selection criteria that should be considered and pathologies to be avoided (see below). A default, rule of thumb heuristic for a decomposition with no obvious pathologies should be *"the lowest importance-minimality coeff whose adversarial PGD-recon KL stays under `T` while L0 stays ≪ rank for some reasonable `T`"*
**Compute plan** — grid size × steps × GPUs (Ask the user if it's okay to use more than 8 GPUs).

### 2. Build the base config from the nearest reference

Identify the closest reference YAML from `param_decomp_lab/experiments/lm/` (`ss_llama_simple_mlp-2L.yaml`, `pile_llama_simple_mlp-4L.yaml`, …). Match on target model family and depth. Override only the hyperparameters you're sweeping. **Open and read it** — it's the source of truth for field names and for every value you're __not__ changing; the YAML is one validated tree, so a misspelled field fails fast at load.

Size the causal-importance (CI) function to the target. For transformer language models, the CI function is itself a transformer (`ci_config.fn_type: global_shared_transformer`) that reads the target's concatenated hidden activations and emits all CI values. Residual width must be *wider* than the target's (it holds all the target's hidden acts — the paper used 2048 for a 768-wide target); depth ½–2× the target's; MLP ~4× its own width. The reference config sets this for *its* target — if your target is larger/deeper, scale the CI transformer up too, or CI quality caps out.


### 3. Define the grid — 1D over the coefficient

The standard hyperparameter to sweep first is **the importance-minimality coeff**. (too-low → many subcomponents per input;  too-high → CI shrinks and recon blows up; just-right → clean monosemantic components, with low L_0 and low adversarial recon loss). Use ~5 log-spaced points **centered on your target's reference value**, bracketing it roughly an order of magnitude either side. Your sweep should be wide enough to surface pathologies on both ends of the sweep. If that means the sweep is too imprecise around the middle, such that e.g. Pareto curves are too coarse around crucial areas, you should use a denser sweep. Run more runs if you have to. **With VPD it is better to take a long time than to get results that are faulty.**
Keep the sweep 1-dimensional: Handle `C` and frequency-minimality coefficient __separately__, not as second/third axes in the sweep. For `C`, you should over-provision it (e.g. 2x the rank of the corresponding parameter matrix) and later reduce it only if you confirm you can get similar solutions with lower.

### 4. Main run: First smoke-test one point in the grid (modified), then launch the full sweep

**A smoke test is not just `steps: 50`.** At the reference cadence, 50 steps would be entirely swallowed by `faithfulness_warmup_steps` (no PD/adversarial step ever runs) and `eval.every: 1000` means the `PGDReconLoss` eval never fires — so it wouldn't test what you need. Scale these together so warmup ends early, real steps run, and a __slow__ eval cycle (where `PGDReconLoss` lives) lands inside the run; use `streaming` so you don't tokenize the whole dataset up front:

```YAML
pd:      { steps: 50, faithfulness_warmup_steps: 5 }
cadence: { train_log_every: 10 }
eval:    { every: 10, slow_every: 20, slow_on_first_step: true }   # slow_every must be a multiple of every
data:    { streaming: true }                                       # smoke-only: skip upfront full-dataset tokenization
```

Confirm four things: the YAML validates, VRAM fits, a real `PGDReconLoss` value is logged (not just the cheap metrics), and loss isn't NaN past warmup. (Even so, budget a couple of minutes — data prep, not the 50 steps, dominates.) Then fan the grid out as a throttled SLURM array under one W&B group, tagging each task with its coefficient. There is no `pd-sweep` — roll the fan-out yourself.

N.B. `pd-lm-layerwise` is a per-matrix array, a different axis:

```
# Pre-generate cfg-0.yaml … cfg-5.yaml, each patched with the matching coeff:
sbatch --array=0-5%8 --job-name=vpd-<target>-sweep \
  --time=12:00:00 --output=<experiment-dir>/logs/vpd-sweep-%A_%a.out <<'EOF'
#!/bin/bash
cd /path/to/param-decomp
COEFFS=(2e-4 5e-4 1e-3 2e-3 5e-3 1e-2)   # centered on the reference coeff (knob #1); re-center for your target
IMPMIN=${COEFFS[$SLURM_ARRAY_TASK_ID]}
uv run --frozen pd-lm <experiment-dir>/sweep/cfg-${SLURM_ARRAY_TASK_ID}.yaml \
  --group vpd-<target>-impmin --tags "vpd,sweep,impmin=${IMPMIN}"
EOF
monitor_jobs <array_job_id>   # waits across the whole array; run in the background
```

- (`%8` caps concurrency when the grid exceeds 8 — You may use more if appropriate.)

Sweep runs are not merely exploratory: They should run to convergence. Not just 'a reasonably long run' -- we need actual _convergence_. You can't usually tell before convergence how well hyperparameters are doing relative to each other. Sometimes, they will converge before the full step count, and only then should you draw inferences on which runs are doing best. You can sometimes tell early if a run is doing very badly compared with other runs; in those cases, it is okay to stop those runs early in order to free up compute, if they are using lots of resources.

Confirm convergence before interpreting anything (§ checks): **PGD-recon (`PGDReconLoss`) plateaued**, **L0 (`CI_L0`) settled ≪ matrix rank**, and the **unmasked model matches the target — `kl_unmasked ≈ 0` (`CEandKLLosses`)**.

- *PGD-recon isn't plateauing / L0 still falling* → under-trained → more steps (and potentially a stronger adversary: Raise `PGDReconLoss.n_steps`).
- *CI collapsed, recon blew up* → importance minimality coeff too high.
- *L0 not ≪ matrix rank* → coeff too low.
- *kl_unmasked not ≈ 0* → the unmasked subcomponent sum isn't reproducing the target → check parameter faithfulness (`FaithfulnessLoss`).

Output lands at `PARAM_DECOMP_OUT_DIR/runs/<run_id>/` (the lab's run-output dir, set in the environment).

## What a good result looks like, and how to check it

- **Good result:** Adversarial **PGD-recon** (`PGDReconLoss`; freshly-initialized masks, ~20 PGD steps), has *plateaued* low — faithful in the worst case, not just on average — **and** **L0 per datapoint** (`CI_L0`) has settled `≪` matrix rank, so it's a genuine simplification. Both must hold: low PGD-recon with high L0 is faithful but not minimal; the reverse is sparse but unfaithful. Note: VPD trains against a persistent PGD adversary (`PersistentPGDReconLoss`, "PPGD") that warm-starts its attack across steps and hunts the most damaging mask consistent with the predicted CI. Mechanistic faithfulness must hold even for these masks (but it's okay if it's somewhat worse than non-adversarial masks). But we judge run quality on a separate, stricter check: a fresh PGD attack at eval (`PGDReconLoss`, `mask_scope: shared_across_batch`), re-initialized from random and run for more steps. It's shared across the batch *on purpose*: a fully unconstrained per-datapoint adversary is too strict — it can exploit uncorrelated superposition-interference noise — so we force it to find **systematic** defects rather than noise-fit single points.
- **Verify:** the unmasked model matches the target — **`kl_unmasked ≈ 0`** (the output KL of the all-components-on model vs the target, logged by `CEandKLLosses`); if it isn't, check parameter faithfulness (`FaithfulnessLoss`) and the Δ component. The `CIMeanPerComponent` spectrum shows a sharp alive/dead cutoff and at least some CI values sit ~1 for most inputs (if none do, the importance minimality coefficient is too high); a known mechanism, if you have one, comes back as expected (an inserted identity matrix → one high-rank component).
- **Also monitor, but never optimize:** stochastic-recon (average-case ablation) and — to confirm the causally-important set is sufficient — **CI-masked** and **rounded-CI-masked** recon. Keep the latter two out of the *training* loss: they're trivially driven to ~0 by cheating, so they're only useful metrics because nothing optimizes them directly. Adversarial PGD recon (and, less so, stochastic recon) is the bar that resists cheating.
- **Failure mode that superficially looks informative (but is not):**
  - *Under-training read as "no structure."* L0 falls slowly under p-annealing (the importance penalty's p-norm anneals 2.0→0.4 over the run, which is what steadily drives L0 down) and the adversary takes many steps to bite. So an unconverged run shows undertrained solutions that look like 'no structure.' Confirm PGD-recon has plateaued and L0 settled before drawing that conclusion — most apparent VPD nulls are under-training.
  - *Non-categorical output read as a null.* The recon/eval losses (`PGDReconLoss`, `kl_unmasked` / `CEandKLLosses`) assume a **categorical** output distribution. On a regression/scalar target (`d_vocab_out = 1`) the softmax is trivially 1 and KL ≡ 0, so the run optimizes a vacuous objective and yields a degenerate decomposition that looks like a clean null. The default LM configs use the KL/CE path; if your target's output isn't categorical, switch to the MSE reconstruction objective (`recon_loss_mse`) before trusting anything.

If no runs in the sweep look good, reason through how to modify the hyperparameters in order to fix the observed pathologies, and run a new sweep.

## Helpful info when setting hyperparameters for sweeps

| # | Hyperparameter | Reference value | When to change |
|---|---|---|---|
| 1 | **Importance-minimality coeff** — `loss_metrics[ImportanceMinimalityLoss].coeff` | `1e-3` (2L ref; deeper LMs ~`2e-4`) | **Most sensitive — sweep this first.** Too high → CI collapses below 1, recon blows up; too low → too many subcomponents fire per input (not minimal). |
| 2 | **Component / CI learning rate** — `components_optimizer` / `ci_fn_optimizer` `lr_schedule.start_val` | `5e-5` (4L Pile paper run; cosine →10%) · ~`1e-3` (toy) | Standard LR tuning. (The 2L SimpleStories config uses `3e-4`.) |
| 3 | **Adversarial LR** — `PersistentPGDReconLoss.optimizer.lr_schedule.start_val` | `1e-2` | Keep `n_adv · lr_adv ≈ 2` (`n_adv` = total adversarial updates ≈ 3: 2 warmup + the outer step). Deeper models → more steps, proportionally lower LR. |
| 4 | **Frequency-minimality `beta`** — `ImportanceMinimalityLoss.beta` | `0.5` | Lower → fewer, more polysemantic subcomponents. Interacts with #1. (Imp-min and freq-min are fused into one minimality term in code, `beta` = inner weight on the frequency part — so logs show a single minimality loss.) |
| 5 | **Subcomponents `C` per module** — `decomposition_targets[].C` | see § C defaults | Err high; aim within ~2× of the true count. |
| 6 | **Δ-L2 faithfulness coeff** — `FaithfulnessLoss.coeff` | `1e7` | Insensitive; raise 10× until `kl_unmasked` is negligible (the raw `Σ(UV)` reproduces the target). Safe to overshoot. |


The canonical values for an example 4-layer Pile model (`d_resid = 768`; scale `C` with width):

| Module | `C` per layer |
|--------|------|
| `c_fc` (MLP up, 768×3072) | 3072 |
| `down_proj` (MLP down, 3072×768) | 3584 |
| `q_proj`, `k_proj` (768×768) | 512 |
| `v_proj` (768×768) | 1024 |
| `o_proj` (768×768) | 1024 |

If unsure, over-provision and do a converged run, then read the `CIMeanPerComponent` spectrum: there's usually a sharp alive/dead cutoff revealing a rough estimate of the count. Aim within ~2× of it.


## Toolbox

VPD runs from the `param-decomp` library (core `param_decomp` + lab tooling
`param_decomp_lab`):

```bash
cd /path/to/param-decomp && uv run pd-lm <config.yaml> --group <id> --tags vpd
```

- `pd-lm <config.yaml> [--dp N] [--resume <resume.yaml>]` — train an LM decomposition (`--dp N` = N-process DDP). `pd-tms` / `pd-resid-mlp` for toy models.
- `pd-lm-layerwise <config.yaml>` — split a config into per-matrix configs + submit as a SLURM array (a per-matrix axis, *not* a hyperparameter sweep).
- Reload: `SavedLMRun.from_path("entity/project/runs/<id>").load_model() -> ComponentModel`.
- Downstream (separate stages, own configs): `pd-harvest`, `pd-autointerp`, `pd-attributions`, `pd-graph-interp`, `pd-clustering`, `pd-postprocess`.
- Cluster: no `--partition`, no CPU/mem on GPU jobs; wait on jobs with `monitor_jobs <id>...`.

## Background / citations

- **VPD** is the main method, published in the "Interpreting Language Model Parameters" paper (Bushnaq et al. 2026: https://static.goodfire.ai/vpd-blog-post/post.md). It fixes the (insufficient) uniform stochastic masking procedure introduced in the SPD paper by introducing masks that are chosen adversarially, not just stochastically.
- **SPD** — *Stochastic Parameter Decomposition* (Bushnaq, Braun & Sharkey, 2025): An out-of-date method. Superseded by VPD. First to introduces the causal importance function.
- **APD** — *Attribution-based Parameter Decomposition* (Braun, Bushnaq & Sharkey, 2025):
  The precursor method on which VPD and SPD are conceptually based.

N.B. **From subcomponents to components.** A run produces rank-1 *subcomponents*. The *components* (the learned "mechanisms" this skill keeps referring to) come from clustering co-activating subcomponents — a separate, required step via `pd-clustering` (MDL clustering + stochastic hierarchical merging; you pick the merge threshold α). "Components" are not subcomponents, and vice versa. But subcomponents are often useful objects in and of themselves.
