# Parameter Decomposition

This document is a practical guide to implementing, training, and evaluating a parameter decomposition with this repository. The science lives in the [parameter-decomposition handbook](handbook.md) — why mechanisms are sought in the weights rather than the activations, the four properties a trustworthy decomposition must exhibit (parameter faithfulness, minimality, mechanistic faithfulness, simplicity), the evidence standards and failure modes, and what a finished decomposition licenses downstream.

## Implementing a target

The core trainer is agnostic to target architecture: it drives anything satisfying the `DecomposedModel` protocol (`param_decomp/core/model.py`), including the target-owned reconstruction metric `recon_loss_fn`. **Expect to implement your own target, and often your own composition root.** The targets that ship with the library (TMS and ResidMLP toys, a 4-layer Llama-style LM, GLU-transformer targets loading Llama-3.1-8B / Qwen3-8B-Base HF weights) are working examples from our own research. Read the one nearest your shape before writing yours, and fit the library to your problem rather than your problem to these examples. The reconstruction comparison rides on the target, not on config: `recon_loss_fn` is a static method on your `DecomposedModel`, and every recon term — the training losses and the fresh-PGD eval probe alike — scores through it. The LM targets return `kl_per_position`; TMS and ResidMLP return MSE. Use what your output space calls for, and implement it if nothing shipped matches your shape.

This is a research repo. When the library is awkward or gets in your way, patch your checkout directly and note the diff

In some cases, you may be able to use the included implementations of target models and composition roots. The runnable roots are [`param_decomp/experiments/lm/training.py`](../param_decomp/experiments/lm/training.py) for LMs, [`param_decomp/experiments/tms/run.py`](../param_decomp/experiments/tms/run.py) for TMS, and [`param_decomp/experiments/resid_mlp/run.py`](../param_decomp/experiments/resid_mlp/run.py) for ResidMLP. They are thin wrappers around the core trainer that load the config, target, and data. [`param_decomp/experiments/toy_eval.py`](../param_decomp/experiments/toy_eval.py) only binds authored toy evaluation operations; it is not a runnable composition root. In general, treat the included implementations as references, and use the universal library entrypoint `param_decomp/core/run.py` when writing your own root.

To implement a target model,

- Check for an existing JAX implementation to start from — many language models have JAX implementations on HuggingFace or elsewhere.
- `core/model.py` is the contract. Read it in full, docstrings included, before writing a line — it carries the waist geometry, the mask semantics, and the per-method requirements.
- For an LM-shaped target, start from `param_decomp/targets/llama_simple_mlp.py` and copy its structure; read `SyntheticDecomposedModel` in `param_decomp/core/tests/test_generic_model_io.py` (~120 lines, driven through the real train step) when you want the protocol with nothing else attached. Every shipped `DecomposedModel` lives in `param_decomp/targets/` — when you reuse a module from one, read it first; each bakes in its own family's choices (attention flavor, positional encoding, norm placement).
- Verify the port against an independent reference forward — a torch implementation when one exists, a float64 NumPy forward you write from the architecture spec otherwise; the value of the oracle is its independence from your JAX code. The repo's parity-golden machinery is the shape to copy: frozen goldens under `param_decomp/targets/tests/`, with `qwen3_hf_parity/gen_hf_fixtures.py` as a turnkey generator (tiny-random and `--real` modes, its docstring carries the torch-venv recipe).
- **On GPU, set `jax_default_matmul_precision = "highest"` for parity checks**: fp32 matmuls default to TF32 on Ampere+, and a correct port will fail every gate with depth-growing deviation that looks exactly like a wrong block. Check parity on the real target as well as any tiny fixture — the TF32 effect grows with depth, and a 3-layer fixture can read as rounding noise.
- Before allocating any GPU: parse your config through the experiment schema and run the placement assertions (`assert_placement_claims`) on CPU. Schema, loss-builder, and placement failures are all free to find statically, and their assertion messages are the documentation.

For agents: If you cannot establish an equivalent `DecomposedModel` implementation of the target model, do not continue, and explain to the user.

## Recipe

Mapping from our conceptual-level losses (from the handbook) to their (sometimes multiple) instantiations.

- **Parameter faithfulness:** `FaithfulnessLoss`. The squared elementwise error between the component sum and the target weights, averaged over the total decomposed parameter count. Equivalently, the (normalized) sum of squared values in the ∆-component.
- **Minimality:** `ImportanceMinimalityLoss` (Lp), or the newer `SmoothL0ImportanceMinimalityLoss`. Also measured using L0 as a non-differentiable eval.
- **Simplicity** — implemented as the nested `frequency:` block on the two importance-minimality losses above.
- **Mechanistic faithfulness:** Reconstruction Loss, of which there are many different variations. Loss configs take the rough form `*Recon*Loss`, with the wildcards expanding to variations. Fundamentally a reconstruction loss asks "how well does the decomposition approximate the target model's behaviour" but this leaves unbound various details. There are numerous schemes for masking causally unimportant subcomponents (stochastic masking, adversarial masking, raw CI-value masking), and many other tweaks which improve training dynamics, for example only using the decomposition parameters for a subset of the model, as in our `*ReconSubsetLoss`es or `*ReconLayerwiseLoss`es. In *Interpreting Language Model Parameters*, VPD uses separate stochastic and adversarially masked reconstruction passes, but this is just one potential combination of many valid configurations.

A good default which we have recently settled on:

- `FaithfulnessLoss`
- `SmoothL0ImportanceMinimalityLoss` + `frequency` term
- `MergedStochasticSubsetPPGDReconLoss` — a compound loss combining adversarial (PPGD) and stochastic (StochasticSubset) reconstruction into one pass.

The 4L-Pile reference config (`param_decomp/experiments/lm/configs/pile_llama_simple_mlp-4L.yaml`) carries this recipe with tuned coefficients; the toy reference configs carry the same shape at their own coefficients. In general you should start with a reference config nearest your target.

The most important hyperparameters are not knowable a priori, so **the default arc is to sweep the importance-minimality coefficient** (`SmoothL0ImportanceMinimalityLoss.coeff`). Depending on the pathologies this surfaces, sweep `frequency.coeff` or the subcomponent budget `C` separately. Runs must go to convergence — analyzing or comparing unconverged decompositions is almost always worthless.

### 0. Background

Fresh LM decompositions are usually compute-heavy and often need several sweep rounds.

#### Setting hyperparameters for sweeps

| Hyperparameter | Reference value | When to change |
| :---- | :---- | :---- |
| **Importance-minimality coeff** — `pd.loss_metrics[SmoothL0ImportanceMinimalityLoss].coeff` | `8e-4` (TMS 5→2); `6e-5` (TMS 40→10); `2e-4` (4L-Pile flagship) | **Most sensitive — sweep this first.** Too high → CI collapses below 1 and recon blows up; too low → too many subcomponents fire per input. |
| **Component / CI learning rate** — `pd.components_optimizer.lr_schedule.start_val` and `pd.ci_fn_optimizer.lr_schedule.start_val` | `1e-3` both (TMS); `5e-5` both (4L-Pile flagship; cosine →10%) | Standard LR tuning. The flagship keeps the two equal; we vaguely think components at ~3× the CI-fn LR is a good starting point in LMs — unswept, a hunch rather than a tuned fact. |
| **Frequency-minimality coeff** — `...[SmoothL0ImportanceMinimalityLoss].frequency.coeff` | `4e-4` (TMS 5→2); `3e-5` (TMS 40→10); `1e-4` (4L flagship) | Keep near half the importance-minimality coeff initially. Lower → fewer, more polysemantic subcomponents. **This is an independent loss coefficient; there is no frequency `beta` like in the paper.** |
| **Smooth-L0 width schedule** — `...[SmoothL0ImportanceMinimalityLoss].gamma` | linear `1.0 → 0.01` | Annealing sharpens the smooth active-count proxy over training. Copy the whole schedule from the flagship config unless intentionally studying it. |
| **Subcomponents `C`** — toy `decomposition.sites.sites[*].C`; LM `decomposition.sites.cs.<matrix>` | TMS 5→2: `20` per site; TMS 40→10: `100`; LM: see the C table below | Start from the matching shipped config. For LMs, calibrate down from a converged `CIMeanPerComponent` spectrum; usually keep C at or above the matrix rank — below rank, exact faithfulness is structurally impossible. |
| **Δ-L2 parameter-faithfulness coeff** — `pd.loss_metrics[FaithfulnessLoss].coeff` | `1` (TMS); `1e7` (4L flagship) | Insensitive; raise 10× until `kl_unmasked` is negligible. Overshooting is safe within reason, but an excessively large value can still impair optimization — stop raising once `kl_unmasked` is flat. |

Each decomposed matrix gets a subcomponent budget at `decomposition.sites.cs.<matrix>`. The matrix names are the site vocabulary of `decomposition.sites.kind`. The shipped 4-layer simple-MLP reference (`d_resid = 768`; scale with width) uses:

| Module | `C` per layer |
| --- | ---: |
| `c_fc` (MLP up, 768×3072) | 3072 |
| `down_proj` (MLP down, 3072×768) | 3584 |
| `q_proj`, `k_proj` (768×768) | 768 |
| `v_proj` (768×768) | 1024 |
| `o_proj` (768×768) | 1024 |

The shipped 32-layer GLU-transformer reference (`d_model = 4096`) uses `q: 2048`, `k: 2048`, `v: 4096`, `o: 4096`, `gate: 8192`, `up: 8192`, and `down: 10240` per layer; see [`llama8b_full32L_HSDP_b64_dp64.yaml`](../param_decomp/experiments/lm/configs/llama8b_full32L_HSDP_b64_dp64.yaml). These are reference budgets, not universal defaults: scale them for the target width and architecture, and start from the matching shipped config when one exists.

In general, we think a safe approach early in tuning is to make C slightly larger than the maximum dimension of a given matrix. It certainly should be larger, usually, than the rank of the matrix, otherwise faithfulness becomes structurally impossible.

So, for an LM with no close reference, over-provision and do a converged run, then read the `CIMeanPerComponent` spectrum: there's usually a sharp alive/dead cutoff after training, where a long tail of subcomponents have mean ci below ≈1e-6 or 1e-7. We take this to reveal a rough estimate of the count of the "alive" subcomponents. Aim within ~2× of it.

For toy models, do not use this LM heuristic. E.g. for TMS we use `20` per site for 5→2 and `100` for 40→10, which deliberately exceeds the known feature count.

### 1. Define the experimental plan

Fix the following decisions before launching an experiment:

- **Domain and runtime** — identify the target model. Confirm that the public JAX package can be installed in the execution environment. TMS and ResidMLP run on CPU and need no GPU; any non-toy experiments will normally need GPU compute.
- **LM target** (`target`) — `weights_dtype` (required, no default) plus a `spec`. Use `bfloat16` unless you have a reason not to: `float32` works, but it drops off cuDNN flash attention, so an 8B target materialises the `[B, H, T, T]` scores and will likely OOM rather than error.
- **Data distribution**
- **Base config** — the reference config plus the values for hyperparameters that matter
- **Sweep grid (optional)** — the axes and their values (see step 2).
- **Selection rule** — Choose it by domain. For an LM without ground truth, use the converged reconstruction/minimality Pareto front; a reasonable default is the lowest importance coefficient whose adversarial recon stays below a stated threshold while `CI_L0` is materially simpler than the relevant dense baseline. For TMS, use the ground-truth target-CI metric and expected feature pattern below — the toys do emit a fresh-PGD recon scalar on their own MSE metric, but ground truth outranks it whenever it exists.
- **Compute plan** — first decide whether GPUs are needed at all: TMS and ResidMLP use CPU; LM sweeps usually use GPUs. Record grid size × steps × devices; later rounds may be needed.

For agents: Get explicit approval from your operator on these points before going ahead with costly experiments.

### 2. Define the grid — 1D over the coefficient

Sweep the importance-minimality coeff first. Too low gives many active subcomponents; too high collapses CI and breaks reconstruction. For LMs, the useful region has low `CI_L0` and low adversarial recon loss; for toys, use the runner's ground-truth metric and reconstruction training loss. Use about five log-spaced points centered on the relevant reference and bracket roughly an order of magnitude either side. The grid must be wide enough to expose both pathologies; add points if the Pareto front is too coarse. **With VPD it is better to take longer than to trust a faulty decomposition.**

Keep the first sweep one-dimensional. Hold `frequency.coeff`, `gamma`, and the domain's `C` fields fixed. If a later frequency sweep is needed, sweep its coefficient directly — there is no `beta` — while holding the importance coefficient fixed. Over-provision `C` on a first run, and reduce it only after a converged run shows a stable alive/dead cutoff.

### 3. Happy-path test one point, then launch the full sweep

A useful initial test must do more than set `steps: 50`. Schedules use normalized progress, so truncating 400k steps to 1k fully anneals them on a compressed schedule rather than reproducing the first 1/400th of the reference run. The test must pass faithfulness warmup, perform a PD or adversarial update, fit in memory under the intended CPU/GPU topology, and run an evaluation pass. For LMs, include the slow evaluation pass. For toy models, include the configured evaluation and the runner's ground-truth metric. Launch the complete sweep after one configuration passes these checks.

Run sweep configurations to convergence. Compare hyperparameters only after the relevant metrics plateau, even when that happens before the full step count. Stop a clearly dominated configuration early when doing so frees substantial compute.

Confirm convergence before interpreting anything. **For LMs:** eval `PGDReconLoss` plateaued, `CI_L0` settled at a meaningful simplification relative to the relevant dense baseline, and `kl_unmasked ≈ 0`. **For toys:** training losses and the in-loop ground-truth metric settled; do not wait for LM eval metrics the runner does not compute.

- *PGD-recon is not plateauing or L0 is still falling:* the run needs more steps. If stochastic reconstruction and average PPGD training loss look good but adversarial PGD reconstruction is poor during evaluation, strengthen the persistent-PGD training loss (`PersistentPGDReconLoss`, or the merged loss's adversarial term). Increase `n_warmup_steps` for more inner Adam ascents per training step; total ascents are `n_warmup_steps + 1`. `optimizer.lr_schedule` controls the ascent learning rate. Lower the learning rate proportionally when increasing the ascent count. The training loss has no `n_steps` or `step_size`; those settings belong to the evaluation probe.
- *Importance-Minimality collapsed, recon blew up* → importance minimality coeff too high.
- *L0 remains high relative to the domain's expected active-mechanism count or sweep Pareto front* → Importance-Minimality coeff too low.
- *kl_unmasked not ≈ 0* → the unmasked subcomponent sum isn't reproducing the target → check parameter faithfulness (`FaithfulnessLoss`). At sufficiently good faithfulness loss kl_unmasked should be ≈0 by construction. Remember this will be impossible if C < rank in your target parameters.

## What a good result looks like, and how to check it

- **Good result:** (This is key info)
  - Adversarial **PGD-recon** (`PGDReconLoss`; freshly initialized masks, ~20 PGD steps) has *plateaued* low — mechanistically faithful in the worst case, not just on average — **and** **L0 per datapoint** (`CI_L0`) has settled `≪` matrix rank, so it is an actual simplification. Both must hold. VPD trains against a persistent PGD adversary that warm-starts its attack across steps; run quality is judged on the separate stricter eval probe: the `PGDReconLoss` entry in `eval.metrics` with `init: random` and `source_shape: c`, reinitialized fresh at each eval. Why the eval adversary is fresh and shared across the batch, and how much adversarial robustness to demand, are covered in the [handbook's evidence standards](handbook.md#evidence-standards).
- **Verify:** the unmasked model matches the target — `kl_unmasked ≈ 0` (the output KL of the all-components-on model vs the target, logged by `CEandKLLosses`) — this is the **parameter-faithfulness** check; if it fails, raise the Δ-L2 coefficient (`FaithfulnessLoss`) and inspect the Δ component. The `CIMeanPerComponent` spectrum shows a sharp alive/dead cutoff and at least some CI values sit ~1 for most inputs (if none do, the importance minimality coefficient is too high); a known mechanism, if you have one, comes back as expected (an inserted identity matrix → one high-rank component).
- **Metric noise should be accounted for:** The end-of-training value of training/eval metrics is just a point estimate of a noisy quantity. Plot the metric curves over training and compare runs on windows of settled values — don't compare single end-of-run point estimates; you risk just comparing noise.
- **Monitor but do not optimize CI-masked and rounded-CI-masked reconstruction.** These metrics test whether the causally important set is sufficient. Keep them out of the training loss because optimization can trivially drive them near zero; see the [handbook](handbook.md#evidence-standards). Stochastic reconstruction is a training term. Use its evaluation counterpart, `kl_stoch_masked`, as the average-case result while noting that it is optimized. Adversarial PGD reconstruction is the stronger check against this failure.
- **Failure mode that superficially looks informative (but is not):**
  - *Under-training read as "no structure."* Minimality arrives late while the Smooth-L0 `gamma` schedule sharpens (the reference anneals `1.0 → 0.01`), and the adversary takes many steps to bite. If using `ImportanceMinimalityLoss` instead, its `pnorm` schedule plays the analogous role (canonical `2.0 → 0.4`). For LMs, confirm PGD-recon and L0 have settled; for toys, confirm the training losses and target-CI metric have settled. Most apparent VPD nulls are under-training.

If no runs in the sweep look good, reason through how to modify the hyperparameters in order to fix the observed pathologies, and run a new sweep. Do not proceed to downstream analysis until you've got hyperparameters that you're confident are good.

For agents: If the new sweep materially changes or expands the approved compute plan, get explicit approval from your operator before launching it.

### Eval metrics: the canonical block

The eval pass is **authored-only**: exactly the metrics listed in `eval.metrics` run, and nothing silently re-adds one you drop. LM runs should keep the flagship's full **metric list**; the block's sizing knobs are yours — `eval.batch_size` is a **global** batch (scale it to your allocation, like the training batch). Four gates the schema enforces: each metric's logged identity must be unique — its `name` if it has one, otherwise its `type`, so a second `PGDReconLoss` probe at different `n_steps` is fine provided you `name` it — eval entries must not set a `coeff` (training-only), `CI_L0` requires `groups` (write `groups: null` for none), and the cadences must nest — `eval.every` a multiple of `cadence.train_log_every`, and `eval.slow_every` a multiple of `eval.every`.

| Metric | What it logs | How to read it |
| :---- | :---- | :---- |
| `CEandKLLosses` | Output-level faithfulness battery under `eval/ce_kl/`: `kl_<variant>` for all six variants — `unmasked`, `ci_masked`, `rounded_masked` (CI binarised at the required `rounding_threshold`), `stoch_masked`, `random_masked`, `zero_masked` — plus `ce_difference_<variant>` for the same set *minus* `zero_masked`, so six `kl_` keys and five `ce_difference_` keys. LM-only: these keys read next-token CE and KL over a categorical output distribution. The toy binding refuses the metric outright, but the LM binding's only construction-time guard is `has_position_axis` — a positioned non-categorical target clears it and logs meaningless numbers, so omit this metric unless your target emits logits. | The workhorse. `kl_unmasked ≈ 0` = parameter faithfulness holds. `kl_stoch_masked` = the headline average-case recon number for comparing runs. `kl_rounded_masked` low = the binarised causally-important set alone suffices. For cross-model comparison, normalize the `kl_` numbers against the `kl_zero_masked` ceiling yourself (there is no `ce_difference_zero_masked` to normalize against). |
| `PGDReconLoss` (`init: random`, `n_steps: 20`, `step_size: 0.1`, `source_shape: c`) | Fresh 20-step adversarial attack at eval, with one source shared across batch and positions. `step_size` is required, and only `init: random` with `source_shape: c` executes — anything else asserts at construction. An optional `name` sets the logged key (the flagship logs `eval/loss/PGDReconLoss_20step`). Copy the whole entry from the flagship. | The strict quality bar (see "What a good result looks like"). By far the *noisiest* metric — read the plateau/trend, never a single point. The 20-step budget is pragmatism, not principle: the working assumption is that robustness to 20 steps is enough, and at large scale the attack is still climbing well past 20 steps — treat absolute levels with suspicion. |
| `CI_L0` (per-layer `groups` + `total`) | Mean active CI values per position | Compare runs on the **Pareto plot of L0 versus reconstruction KL**. Runs on the same tight Pareto front represent similar-quality hyperparameter tradeoffs. |
| `CIMeanPerComponent` | Sorted mean-CI-per-component spectrum (linear + log) | Sharp alive/dead cutoff ≈ true component count → calibrates `C` (see the C table above). No values near 1 → imp-min coeff too high. |
| `ComponentActivationDensity` | Per-layer histogram of each component's firing density | Alive-vs-dead split; a smear with no bimodality = components aren't specializing. |
| `CIHistograms` | Per-layer histograms of CI values (lower-leaky + pre-sigmoid) | Want mass piled at 0 and near 1; everything pinned strictly below 1 = CI collapse (coeff too high). |
| `StochasticHiddenActsReconLoss` / `CIHiddenActsReconLoss` | Per-module hidden-activation MSE under stochastic / CI masks | Localizers: when output KL is bad, these tell you *which module's* reconstruction is failing. Monitor-only. |

Check every metric above and record the anomalies you find. If one shows the run is broken, launch another sweep that tests a fix rather than stopping at the breakage alone.

## Analysis of a decomposition

For agents: Depending on what the user wants, you might at this point choose to perform different analyses.

For agents: Show the key metrics and their plots to the user, and discuss why they indicate that the decomposition is acceptable or not. It will usually be a presentation of acceptability, because if you didn't find a good decomposition on one sweep, you should typically continue with more sweeps in order to find hyperparameters that work.

Assuming your decomposition looks good according to the various metrics, the next step is typically the 'visualization' of the parameter subcomponents. This involves highlighting 'activating dataset examples', where activating here means both 'the subcomponent is causally important on these dataset examples' and 'the subcomponent has a subcomponent activation above a threshold'. You should typically visualize both of these as separate options. For causal importances, the subcomponent should be considered active if it has a causal importance above 0.0, whereas for the subcomponent activation, the threshold is set heuristically (e.g. anything with magnitude larger than 0.5 standard deviations (±) of the subcomponent activation for that subcomponent counts as 'active').

Another standard analysis is the visualization of the QK circuit. Details of this analysis can be found in the VPD paper. In particular, you should display plots of 'Standardized Static Interaction Strength' for pairs of subcomponents (where you should display each head separately as well as one aggregate measure of SSIS), and dynamic interaction strength, where you should display the contributions of various pairs of subcomponents to the attention scores of each head. You should show this by displaying the attention scores and attention patterns of each head as heatmaps, which vary according to the datapoint you display. Visualization should let users explore a number of different input datapoints, which should result in a number of different heatmap arrays. Further descriptions can be found in the VPD paper.

Another standard analysis you should consider performing is running inference on the model with and without particular interpreted subcomponents, and observing the change in behavior. Users often use VPD in order to be able to interpret the effects of weight changes in their model, and this analysis is a first step in that direction.

## Background / citations

The [parameter-decomposition handbook](handbook.md) carries the science and cites the literature — the method implemented here is VPD, from *Interpreting Language Model Parameters* (Bushnaq et al. 2026: [https://static.goodfire.ai/vpd-blog-post/post.md](https://static.goodfire.ai/vpd-blog-post/post.md)).

**From subcomponents to components.** A run produces rank-1 *subcomponents*. The *components* (the learned "mechanisms" this skill keeps referring to) come from clustering co-activating subcomponents — a separate step via the clustering pipeline (MDL clustering + stochastic hierarchical merging; you pick the merge threshold α), needed when the analysis wants mechanism-level objects. "Components" are groups of subcomponents that tend to be 'used' together by the network. But subcomponents are often useful objects for interpretability by themselves.
